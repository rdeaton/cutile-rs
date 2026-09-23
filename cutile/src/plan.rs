/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Launch plans: resolve a kernel launch once per argument layout, then
//! replay it by writing only pointers and scalars.
//!
//! A generated launcher resolves every launch from scratch: it probes the
//! launch-site cache, validates each argument against the compiled
//! specialization, runs the hoisted launch checks, infers the grid, and
//! marshals every argument. All of that is a function of the argument
//! *layout* — dtypes, shapes, strides, partitions, pointer alignment, and
//! the divisibility of integer scalars — not of the pointer values. (Plan
//! capture refuses a kernel whose hoisted checks read anything else.) A
//! [`LaunchPlan`] does that work once, through the normal launcher, and keeps
//! the result:
//!
//! ```rust,ignore
//! // Once per layout: JIT, validation, grid, argument template. No launch,
//! // so the returned `z` is untouched.
//! let (plan, (z, _, _)) = kernels::add(z.partition([128]), &x, &y).plan_on(&stream)?;
//!
//! // Safe replay: checks the layout, retains the tensors like any launch,
//! // writes the new pointers, and calls cuLaunchKernel.
//! let (z, _, _) = plan.launch((z, &x2, &y2)).sync_on(&stream)?;
//!
//! // Unsafe replay: no cutile tensors at all, for callers that manage
//! // device memory themselves. The guard keeps the kernel's module loaded.
//! let launch = unsafe {
//!     plan.launch_raw(&stream, &[RawArg::ptr(z_ptr), RawArg::ptr(x_ptr), RawArg::ptr(y_ptr)])?
//! };
//! stream.synchronize()?;
//! drop(launch);
//! ```
//!
//! A plan is exact: it accepts only arguments whose layout equals the one it
//! was built for. A kernel used at several shapes needs one plan per shape;
//! [`PlanCache`] keeps one kernel's plans and builds missing ones on demand,
//! and [`plan_launch!`](crate::plan_launch) gives a call site its own cache.
//!
//! Builder options — `.grid(..)`, `.const_grid(..)`, `.generics(..)`,
//! `.compile_options(..)` — are baked into the plan when it is built and
//! recorded as its [`PlanOptions`]. They are not a function of the
//! arguments (a grid computed from a scalar changes while the layout does
//! not), so [`PlanCache`] looks plans up by options *and* layout.
//!
//! A plan's type says how it may launch. For an entry point `f`:
//!
//! - `LaunchPlan<f::Kernel>` (that is, `S = `[`Safe`]) is built by
//!   `.plan_on()` and replayed by [`LaunchPlan::launch`]. Its arguments are
//!   type-checked against `f::Kernel::Params` (see [`PlanArgs`]), so a wrong
//!   count or kind is a compile error; the layout is checked at launch.
//! - A plan of an `unsafe` entry point, or of a launcher opted into
//!   programmatic dependent launch, is `LaunchPlan<f::Kernel, Unchecked>`,
//!   built by `.plan_unchecked_on()`. It has only the `unsafe` launch
//!   methods, [`LaunchPlan::launch_unchecked`] and [`LaunchPlan::launch_raw`]
//!   (and its check-free form, [`LaunchPlan::launch_raw_unchecked`]).
//! - A raw `DevicePointer` parameter (only allowed on `unsafe` entry points)
//!   has no [`PlanArg`] ([`param::Ptr`]), so such plans launch only through
//!   [`LaunchPlan::launch_raw`] or [`LaunchPlan::launch_raw_unchecked`].
//! - A kernel with a `MappedPartitionMut` parameter ([`param::Unsupported`])
//!   cannot be planned: its launcher has no plan methods.

use crate::tensor::{Partition, Tensor, TensorView};
use crate::tile_kernel::{CacheEpoch, CompileOptions};
use cuda_async::device_context::with_default_device_policy;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext, GraphNode, ReplayResource};
use cuda_async::error::DeviceError;
use cuda_async::launch::AsyncKernelLaunch;
use cuda_async::predicate::{Atom, LaunchCheck};
use cuda_core::sys::CUdeviceptr;
use cuda_core::{
    f4e2m1fnx2, f8e4m3fn, f8e5m2, f8e5m3fnu, f8e8m0fnu, launch_kernel, launch_kernel_pdl, tf32,
    DType, DTypeId, Function, LaunchConfig, Stream,
};
use cutile_compiler::specialization::{DivHint, SpecializationBits};
use half::{bf16, f16};
use rustc_hash::{FxBuildHasher, FxHasher};
use std::collections::HashMap;
use std::ffi::c_void;
use std::future::IntoFuture;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock, Weak};

// ── Layouts ─────────────────────────────────────────────────────────────────

/// Everything about one kernel argument that the resolved launch depends on,
/// apart from pointer and scalar values.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub enum ArgLayout {
    Tensor(TensorLayout),
    /// `hint` is `Some` for integer scalars, whose divisibility the kernel
    /// is specialized on.
    Scalar {
        dtype: DTypeId,
        hint: Option<DivHint>,
    },
    Pointer {
        dtype: DTypeId,
        hint: DivHint,
    },
}

/// A parameter binding that launch plans cannot record, as returned by
/// `KernelInputStored::plan_layout` and `KernelOutputStored::plan_layout`.
/// Building a plan with such a binding fails.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedParam(pub &'static str);

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct TensorLayout {
    dtype: DTypeId,
    device_id: usize,
    shape: Vec<i32>,
    strides: Vec<i32>,
    spec: SpecializationBits,
    /// `Some` for a `&mut Tensor` parameter: the partition it is bound with.
    partition: Option<PartitionLayout>,
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    prefix_coverage: bool,
}

/// A tensor layout, borrowed from a stored [`TensorLayout`] or a live
/// [`TensorArg`]. Plans compare and hash layouts only through this type, so
/// a stored layout and an argument that has it are equal, and hash equally,
/// by construction.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorLayoutRef<'a> {
    dtype: DTypeId,
    device_id: usize,
    shape: &'a [i32],
    strides: &'a [i32],
    spec: &'a SpecializationBits,
    partition: Option<PartitionParts<'a>>,
}

/// The part of an argument's layout a [`PlanCache`] key hashes: all of it
/// except a scalar's divisibility hint, which depends on the scalar's value
/// rather than its type. [`arg_matches`] checks the hint separately. A
/// pointer's hint is its alignment, part of its layout like a tensor's.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutKey<'a> {
    Tensor(TensorLayoutRef<'a>),
    Scalar(DTypeId),
    Pointer(DTypeId, DivHint),
}

impl ArgLayout {
    /// A scalar parameter. `hint` is the launcher's divisibility hint for the
    /// value: `Some` for integer scalars declared with a concrete integer
    /// type.
    pub fn scalar<T: DType>(hint: Option<DivHint>) -> Self {
        ArgLayout::Scalar {
            dtype: T::DTYPE,
            hint,
        }
    }

    /// A raw device pointer parameter.
    pub fn pointer<T: DType>(ptr: CUdeviceptr) -> Self {
        ArgLayout::Pointer {
            dtype: T::DTYPE,
            hint: DivHint::from_ptr(ptr),
        }
    }

    /// How many driver arguments this parameter marshals to. Plans locate
    /// slots from what the launcher recorded; this only cross-checks it.
    fn arg_count(&self) -> usize {
        match self {
            ArgLayout::Tensor(t) => {
                let partition = t
                    .partition
                    .as_ref()
                    .map_or(0, |p| p.shape.len() + p.strides.len());
                1 + t.shape.len() + t.strides.len() + partition
            }
            ArgLayout::Scalar { .. } | ArgLayout::Pointer { .. } => 1,
        }
    }

    /// This layout's [`PlanCache`] key; equal to
    /// [`ErasedPlanArg::layout_key`] of every argument that matches it.
    fn key(&self) -> LayoutKey<'_> {
        match self {
            ArgLayout::Tensor(t) => LayoutKey::Tensor(t.borrowed()),
            ArgLayout::Scalar { dtype, .. } => LayoutKey::Scalar(*dtype),
            ArgLayout::Pointer { dtype, hint } => LayoutKey::Pointer(*dtype, *hint),
        }
    }
}

/// Whether `arg` has exactly `layout`.
fn arg_matches(arg: &dyn ErasedPlanArg, layout: &ArgLayout) -> bool {
    arg.layout_key() == layout.key()
        && match layout {
            ArgLayout::Scalar { hint: Some(h), .. } => arg.value_hint() == Some(*h),
            _ => true,
        }
}

impl TensorLayout {
    fn of(layout: TensorLayoutRef<'_>) -> Self {
        TensorLayout {
            dtype: layout.dtype,
            device_id: layout.device_id,
            shape: layout.shape.to_vec(),
            strides: layout.strides.to_vec(),
            spec: layout.spec.clone(),
            partition: layout.partition.map(|p| PartitionLayout {
                shape: p.shape.to_vec(),
                strides: p.strides.to_vec(),
                prefix_coverage: p.prefix_coverage,
            }),
        }
    }

    fn borrowed(&self) -> TensorLayoutRef<'_> {
        TensorLayoutRef {
            dtype: self.dtype,
            device_id: self.device_id,
            shape: &self.shape,
            strides: &self.strides,
            spec: &self.spec,
            partition: self.partition.as_ref().map(|p| PartitionParts {
                shape: &p.shape,
                strides: &p.strides,
                prefix_coverage: p.prefix_coverage,
            }),
        }
    }
}

/// A tensor parameter's layout, for `KernelInputStored::plan_layout` and
/// `KernelOutputStored::plan_layout`.
pub(crate) fn tensor_layout<B: TensorArg>(arg: &B) -> ArgLayout {
    ArgLayout::Tensor(TensorLayout::of(arg.layout()))
}

/// The value's bytes in a zeroed 16-byte slot, as the launcher stores it.
fn scalar_bits<T: DType>(value: T) -> u128 {
    const { assert!(std::mem::size_of::<T>() <= 16) };
    let mut bits = 0u128;
    // SAFETY: `T` fits in the slot (asserted above), and `DType` promises it
    // has no padding bytes, so every byte of `bits` stays initialized.
    unsafe { std::ptr::write_unaligned(&mut bits as *mut u128 as *mut T, value) };
    bits
}

/// Whether a plan pins `atom`'s value, so that a launch check reading it holds
/// on every replay once it held at build time. Shapes and partitions are part
/// of the argument layout; the grid is fixed by the plan.
///
/// Exhaustive on purpose: a new atom kind must decide here whether a replay
/// with new pointers and scalars can change it. If it can, return `false`,
/// and plans refuse kernels whose checks read it.
fn fixed_by_plan(atom: &Atom) -> bool {
    match atom {
        Atom::Dim { .. } | Atom::ViewExtent { .. } | Atom::TileCount { .. } => true,
        Atom::NumTileBlocks(_) => true,
        // Device values; `evaluate_launch_check` rejects them too.
        Atom::Iv(_) | Atom::TileBlockId(_) => false,
    }
}

// ── Plan arguments ──────────────────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
    pub trait SealedTensor {}
    pub trait SealedSafety {}
}

/// The kinds of kernel parameter, as types. [`Kernel::Params`] lists one
/// per parameter, and a [`PlanArg<P>`] is a value a parameter of kind `P`
/// accepts, so [`LaunchPlan::launch`] checks the argument count and kinds at
/// compile time. Uninhabited: never values.
pub mod param {
    use std::marker::PhantomData;

    /// A `&Tensor` input.
    pub enum In {}
    /// A `&mut Tensor` output, bound through a partition.
    pub enum Out {}
    /// A scalar of type `T`.
    pub struct Scalar<T>(PhantomData<T>, std::convert::Infallible);
    /// A scalar of one of the entry point's type parameters. Any scalar
    /// type is accepted; the plan checks the dtype when it launches.
    pub enum AnyScalar {}
    /// A raw device pointer, of an `unsafe` entry point. No [`PlanArg`]
    /// accepts one: such plans launch only through
    /// [`LaunchPlan::launch_raw`].
    ///
    /// [`PlanArg`]: super::PlanArg
    /// [`LaunchPlan::launch_raw`]: super::LaunchPlan::launch_raw
    pub enum Ptr {}
    /// A parameter launch plans cannot record, such as a
    /// `MappedPartitionMut`. No [`PlanArg`] accepts one, and the kernel's
    /// launcher has no plan methods.
    ///
    /// [`PlanArg`]: super::PlanArg
    pub enum Unsupported {}
}

/// A value that can be passed for a kernel parameter of kind `P` (see
/// [`param`]): a tensor input (`&Tensor`, `Tensor`, `Arc<Tensor>`,
/// `&TensorView`) for [`param::In`], a tensor output (`Partition<Tensor>`,
/// `Partition<&mut Tensor>`) for [`param::Out`], or a scalar for
/// [`param::Scalar`] of its type or [`param::AnyScalar`].
pub trait PlanArg<P>: sealed::Sealed + ErasedPlanArg {}

/// What a replay does with one argument, whatever its kind.
#[doc(hidden)]
pub trait ErasedPlanArg: Send {
    /// This argument's layout, apart from a scalar's divisibility hint.
    fn layout_key(&self) -> LayoutKey<'_>;
    /// The divisibility hint the launcher computes for this value: `Some`
    /// for integer scalars.
    fn value_hint(&self) -> Option<DivHint>;
    /// Registers this argument's device access with the submission, exactly
    /// as the generated launcher does.
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError>;
    /// The value written into this parameter's first argument slot: a device
    /// pointer, or the scalar's bytes.
    fn slot_value(&self) -> u128;
}

/// A tuple with one [`PlanArg`] per element of `P`, in order: the arguments
/// of a kernel whose [`Kernel::Params`] is `P`.
///
/// Implemented for tuples of up to 12 elements, so plans of a kernel with
/// more than 12 parameters launch only through [`LaunchPlan::launch_raw`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not match the kernel's parameters `{P}`",
    note = "pass one argument per kernel parameter, in order: a tensor input for \
            `param::In`, a partitioned output for `param::Out`, and a scalar of \
            the declared type for `param::Scalar`"
)]
pub trait PlanArgs<P>: sealed::Sealed + ArgTuple {}

/// A tuple of arguments, whatever their kinds.
#[doc(hidden)]
pub trait ArgTuple: Send {
    const LEN: usize;
    /// Calls `f` on each argument in order, stopping at the first error.
    fn try_for_each<E>(
        &self,
        f: impl FnMut(usize, &dyn ErasedPlanArg) -> Result<(), E>,
    ) -> Result<(), E>;
}

/// A tensor argument of a plan. The layout recorded when the plan is built
/// ([`tensor_layout`]) and the one a replay matches come from these same
/// accessors, and [`device_ptr`](Self::device_ptr) is the pointer
/// `push_kernel_args` passes to the kernel, so a replay writes what a
/// generated launch would.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not match this kernel parameter",
    note = "a `param::In` takes a tensor input (`&Tensor`, `Tensor`, `Arc<Tensor>`, `&TensorView`), \
            a `param::Out` a partitioned output (`Partition<Tensor>`, `Partition<&mut Tensor>`), \
            and a `param::Scalar` a scalar of its own type"
)]
pub trait TensorArg: sealed::SealedTensor + Send {
    type Elem: DType;
    /// [`param::In`] or [`param::Out`].
    type Kind;
    /// The tensor owning the storage.
    fn tensor(&self) -> &Tensor<Self::Elem>;
    /// The pointer the kernel receives: the storage's, plus any view offset.
    fn device_ptr(&self) -> CUdeviceptr;
    fn shape(&self) -> &[i32];
    fn strides(&self) -> &[i32];
    fn spec(&self) -> &SpecializationBits;
    /// `Some` for an output, which the kernel writes through a partition.
    fn partition(&self) -> Option<PartitionParts<'_>>;

    /// This argument's layout, as a plan records and matches it.
    fn layout(&self) -> TensorLayoutRef<'_> {
        TensorLayoutRef {
            dtype: Self::Elem::DTYPE,
            device_id: self.tensor().device_id(),
            shape: self.shape(),
            strides: self.strides(),
            spec: self.spec(),
            partition: self.partition(),
        }
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PartitionParts<'a> {
    shape: &'a [usize],
    strides: &'a [usize],
    prefix_coverage: bool,
}

impl<'a> PartitionParts<'a> {
    fn of<O>(partition: &'a Partition<O>) -> Self {
        PartitionParts {
            shape: &partition.partition_shape,
            strides: &partition.partition_strides,
            prefix_coverage: partition.prefix_coverage,
        }
    }
}

/// `TensorArg` for types that deref to a whole `Tensor` input.
macro_rules! whole_tensor_input {
    ($([$($bounds:tt)*] $ty:ty),* $(,)?) => {$(
        impl<$($bounds)*> sealed::SealedTensor for $ty {}
        impl<$($bounds)*> TensorArg for $ty {
            type Elem = T;
            type Kind = param::In;
            fn tensor(&self) -> &Tensor<T> {
                self
            }
            fn device_ptr(&self) -> CUdeviceptr {
                self.cu_deviceptr()
            }
            fn shape(&self) -> &[i32] {
                &self.shape
            }
            fn strides(&self) -> &[i32] {
                &self.strides
            }
            fn spec(&self) -> &SpecializationBits {
                &self.spec
            }
            fn partition(&self) -> Option<PartitionParts<'_>> {
                None
            }
        }
    )*};
}
whole_tensor_input!(
    [T: DType + Sync] &Tensor<T>,
    // An owned input, as a launcher returns it for a `Tensor` argument.
    [T: DType] Tensor<T>,
    [T: DType] Arc<Tensor<T>>,
);

impl<T: DType + Sync> sealed::SealedTensor for &TensorView<'_, T> {}
impl<T: DType + Sync> TensorArg for &TensorView<'_, T> {
    type Elem = T;
    type Kind = param::In;
    fn tensor(&self) -> &Tensor<T> {
        self.base
    }
    fn device_ptr(&self) -> CUdeviceptr {
        self.base.cu_deviceptr() + self.offset_bytes as u64
    }
    fn shape(&self) -> &[i32] {
        TensorView::shape(self)
    }
    fn strides(&self) -> &[i32] {
        TensorView::strides(self)
    }
    fn spec(&self) -> &SpecializationBits {
        TensorView::spec(self)
    }
    fn partition(&self) -> Option<PartitionParts<'_>> {
        None
    }
}

/// `TensorArg` for a partitioned output; `$object` borrows its tensor.
macro_rules! partition_output {
    ($($ty:ty => |$p:ident| $object:expr),* $(,)?) => {$(
        impl<T: DType> sealed::SealedTensor for $ty {}
        impl<T: DType> TensorArg for $ty {
            type Elem = T;
            type Kind = param::Out;
            fn tensor(&self) -> &Tensor<T> {
                let $p = self;
                $object
            }
            fn device_ptr(&self) -> CUdeviceptr {
                self.tensor().cu_deviceptr()
            }
            fn shape(&self) -> &[i32] {
                &self.tensor().shape
            }
            fn strides(&self) -> &[i32] {
                &self.tensor().strides
            }
            fn spec(&self) -> &SpecializationBits {
                &self.tensor().spec
            }
            fn partition(&self) -> Option<PartitionParts<'_>> {
                Some(PartitionParts::of(self))
            }
        }
    )*};
}
partition_output!(
    Partition<Tensor<T>> => |p| &p.object,
    Partition<&mut Tensor<T>> => |p| p.object,
);

impl<B: TensorArg> sealed::Sealed for B {}
impl<B: TensorArg> PlanArg<B::Kind> for B {}
impl<B: TensorArg> ErasedPlanArg for B {
    fn layout_key(&self) -> LayoutKey<'_> {
        LayoutKey::Tensor(self.layout())
    }
    fn value_hint(&self) -> Option<DivHint> {
        None
    }
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        // Outputs are written; inputs only read.
        self.tensor()
            .storage
            .retain(ctx, self.partition().is_some())
    }
    fn slot_value(&self) -> u128 {
        scalar_bits(self.device_ptr())
    }
}

/// A scalar type a plan argument can be: every scalar a kernel parameter
/// can declare.
pub trait PlanScalar: DType + sealed::Sealed {
    /// The divisibility hint the launcher computes for this value.
    #[doc(hidden)]
    fn div_hint(self) -> Option<DivHint>;
}

// A blanket `impl<T: PlanScalar> ErasedPlanArg for T` would overlap the
// tensor impls under coherence, so scalars are listed. The hinted types are
// the launcher's (`is_integer_scalar`), with its expression for the hint.
macro_rules! scalar_plan_args {
    (hinted: $($int:ty),*; unhinted: $($other:ty),* $(,)?) => {
        $(scalar_plan_args!(@one $int, |v| Some(DivHint::from_value(v as i32)));)*
        $(scalar_plan_args!(@one $other, |_v| None);)*
    };
    (@one $ty:ty, |$v:ident| $hint:expr) => {
        impl sealed::Sealed for $ty {}
        impl PlanScalar for $ty {
            fn div_hint(self) -> Option<DivHint> {
                let $v = self;
                $hint
            }
        }
        impl PlanArg<param::Scalar<$ty>> for $ty {}
        impl PlanArg<param::AnyScalar> for $ty {}
        impl ErasedPlanArg for $ty {
            fn layout_key(&self) -> LayoutKey<'_> {
                LayoutKey::Scalar(<$ty as DType>::DTYPE)
            }
            fn value_hint(&self) -> Option<DivHint> {
                self.div_hint()
            }
            fn retain(&self, _ctx: &ExecutionContext) -> Result<(), DeviceError> {
                Ok(())
            }
            fn slot_value(&self) -> u128 {
                scalar_bits(*self)
            }
        }
    };
}
scalar_plan_args!(
    hinted: u8, u16, u32, u64, i8, i16, i32, i64;
    unhinted: bool, f16, bf16, f32, tf32, f64, f8e4m3fn, f8e5m2, f8e8m0fnu, f8e5m3fnu, f4e2m1fnx2,
);

/// Calls `f` with the tuple's elements as its arguments; lets
/// [`plan_launch!`](crate::plan_launch) pass a kernel by name.
#[doc(hidden)]
pub trait SpreadArgs<F, R> {
    fn spread(self, f: F) -> R;
}

macro_rules! tuple_plan_args {
    ($len:expr; $($idx:tt $name:ident $kind:ident),+) => {
        impl<$($name: ErasedPlanArg),+> sealed::Sealed for ($($name,)+) {}
        impl<$($name: ErasedPlanArg),+> ArgTuple for ($($name,)+) {
            const LEN: usize = $len;
            fn try_for_each<E>(
                &self,
                mut f: impl FnMut(usize, &dyn ErasedPlanArg) -> Result<(), E>,
            ) -> Result<(), E> {
                $(f($idx, &self.$idx)?;)+
                Ok(())
            }
        }
        impl<$($name: PlanArg<$kind>, $kind),+> PlanArgs<($($kind,)+)> for ($($name,)+) {}
        impl<F, R, $($name),+> SpreadArgs<F, R> for ($($name,)+)
        where
            F: FnOnce($($name),+) -> R,
        {
            fn spread(self, f: F) -> R {
                f($(self.$idx),+)
            }
        }
    };
}
tuple_plan_args!(1; 0 A0 P0);
tuple_plan_args!(2; 0 A0 P0, 1 A1 P1);
tuple_plan_args!(3; 0 A0 P0, 1 A1 P1, 2 A2 P2);
tuple_plan_args!(4; 0 A0 P0, 1 A1 P1, 2 A2 P2, 3 A3 P3);
tuple_plan_args!(5; 0 A0 P0, 1 A1 P1, 2 A2 P2, 3 A3 P3, 4 A4 P4);
tuple_plan_args!(6; 0 A0 P0, 1 A1 P1, 2 A2 P2, 3 A3 P3, 4 A4 P4, 5 A5 P5);
tuple_plan_args!(7; 0 A0 P0, 1 A1 P1, 2 A2 P2, 3 A3 P3, 4 A4 P4, 5 A5 P5, 6 A6 P6);
tuple_plan_args!(8; 0 A0 P0, 1 A1 P1, 2 A2 P2, 3 A3 P3, 4 A4 P4, 5 A5 P5, 6 A6 P6, 7 A7 P7);
tuple_plan_args!(9; 0 A0 P0, 1 A1 P1, 2 A2 P2, 3 A3 P3, 4 A4 P4, 5 A5 P5, 6 A6 P6, 7 A7 P7, 8 A8 P8);
tuple_plan_args!(10; 0 A0 P0, 1 A1 P1, 2 A2 P2, 3 A3 P3, 4 A4 P4, 5 A5 P5, 6 A6 P6, 7 A7 P7, 8 A8 P8, 9 A9 P9);
tuple_plan_args!(11; 0 A0 P0, 1 A1 P1, 2 A2 P2, 3 A3 P3, 4 A4 P4, 5 A5 P5, 6 A6 P6, 7 A7 P7, 8 A8 P8, 9 A9 P9, 10 A10 P10);
tuple_plan_args!(12; 0 A0 P0, 1 A1 P1, 2 A2 P2, 3 A3 P3, 4 A4 P4, 5 A5 P5, 6 A6 P6, 7 A7 P7, 8 A8 P8, 9 A9 P9, 10 A10 P10, 11 A11 P11);

// A kernel without parameters.
impl sealed::Sealed for () {}
impl ArgTuple for () {
    const LEN: usize = 0;
    fn try_for_each<E>(
        &self,
        _f: impl FnMut(usize, &dyn ErasedPlanArg) -> Result<(), E>,
    ) -> Result<(), E> {
        Ok(())
    }
}
impl PlanArgs<()> for () {}
impl<F: FnOnce() -> R, R> SpreadArgs<F, R> for () {
    fn spread(self, f: F) -> R {
        f()
    }
}

/// One argument to [`LaunchPlan::launch_raw`]: a device pointer for a tensor
/// or pointer parameter, or a scalar's value.
///
/// Opaque, so a scalar's dtype is always its value's: `launch_raw` checks
/// scalar dtypes, and that check is only as good as the recorded dtype.
#[derive(Debug, Clone, Copy)]
pub struct RawArg(RawArgKind);

#[derive(Debug, Clone, Copy)]
enum RawArgKind {
    Ptr(CUdeviceptr),
    Scalar {
        dtype: DTypeId,
        hint: Option<DivHint>,
        bits: u128,
    },
}

impl RawArg {
    /// A device pointer, for a tensor or pointer parameter.
    pub fn ptr(ptr: CUdeviceptr) -> Self {
        RawArg(RawArgKind::Ptr(ptr))
    }

    /// A scalar parameter's value.
    pub fn scalar<T: PlanScalar>(value: T) -> Self {
        RawArg(RawArgKind::Scalar {
            dtype: T::DTYPE,
            hint: value.div_hint(),
            bits: scalar_bits(value),
        })
    }
}

// ── Options ─────────────────────────────────────────────────────────────────

/// The builder options a plan was built with: the launcher settings that
/// shape a launch without being a function of its arguments.
///
/// Mirrors the generated builder methods; apply one to a launcher with its
/// `.with_options(&options)`, and read a plan's with [`LaunchPlan::options`].
/// The default is a launcher with no options set: an inferred grid, inferred
/// generics, and default compile options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct PlanOptions {
    grid: PlanGrid,
    generics: Option<Vec<String>>,
    compile_options: CompileOptions,
}

/// How a plan's launch grid is chosen.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum PlanGrid {
    /// Inferred from the partitioned outputs, as when no grid is set.
    #[default]
    Inferred,
    /// Explicit, as `.grid(..)`.
    Grid((u32, u32, u32)),
    /// Explicit and specialized on, as `.const_grid(..)`.
    ConstGrid((u32, u32, u32)),
}

impl PlanOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// An explicit launch grid, as `.grid(..)`. As on the launcher,
    /// `(0, 0, 0)` means no grid: the grid is inferred.
    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.grid = if grid == (0, 0, 0) {
            PlanGrid::Inferred
        } else {
            PlanGrid::Grid(grid)
        };
        self
    }

    /// An explicit grid the kernel is also specialized on, as `.const_grid(..)`.
    /// As on the launcher, `(0, 0, 0)` cannot be launched: building a plan
    /// with it fails.
    pub fn const_grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.grid = PlanGrid::ConstGrid(grid);
        self
    }

    /// Explicit kernel generics, as `.generics(..)`.
    pub fn generics(mut self, generics: Vec<String>) -> Self {
        self.generics = Some(generics);
        self
    }

    /// Compile options, as `.compile_options(..)`.
    pub fn compile_options(mut self, options: CompileOptions) -> Self {
        self.compile_options = options;
        self
    }

    pub fn get_grid(&self) -> PlanGrid {
        self.grid
    }

    pub fn get_generics(&self) -> Option<&[String]> {
        self.generics.as_deref()
    }

    pub fn get_compile_options(&self) -> &CompileOptions {
        &self.compile_options
    }

    /// The grid as a launcher's `(_grid, _const_grid)` fields, where
    /// `(0, 0, 0)` is "not set". Inverse of the grid in
    /// [`from_launcher`](Self::from_launcher).
    #[doc(hidden)]
    pub fn launcher_grid(&self) -> ((u32, u32, u32), bool) {
        match self.grid {
            PlanGrid::Inferred => ((0, 0, 0), false),
            PlanGrid::Grid(grid) => (grid, false),
            PlanGrid::ConstGrid(grid) => (grid, true),
        }
    }

    /// The options a launcher holds, as recorded by its plan mode. A grid of
    /// `(0, 0, 0)` is the launcher's "not set".
    #[doc(hidden)]
    pub fn from_launcher(
        grid: (u32, u32, u32),
        const_grid: bool,
        generics: Option<Vec<String>>,
        compile_options: CompileOptions,
    ) -> Self {
        let grid = match (grid, const_grid) {
            (grid, true) => PlanGrid::ConstGrid(grid),
            ((0, 0, 0), false) => PlanGrid::Inferred,
            (grid, false) => PlanGrid::Grid(grid),
        };
        PlanOptions {
            grid,
            generics,
            compile_options,
        }
    }
}

// ── The plan ────────────────────────────────────────────────────────────────

/// How a [`LaunchPlan`] may be launched: [`Safe`] or [`Unchecked`]. The
/// [module docs](crate::plan) say which launchers build which.
pub trait PlanSafety: sealed::SealedSafety + 'static {}

/// A plan that launches safely, through [`LaunchPlan::launch`]. Built by
/// `.plan()` / `.plan_on(&stream)`.
pub enum Safe {}

/// A plan that launches only through the `unsafe`
/// [`LaunchPlan::launch_unchecked`] and [`LaunchPlan::launch_raw`]. Built by
/// `.plan_unchecked()` / `.plan_unchecked_on(&stream)`.
pub enum Unchecked {}

impl sealed::SealedSafety for Safe {}
impl PlanSafety for Safe {}
impl sealed::SealedSafety for Unchecked {}
impl PlanSafety for Unchecked {}

/// A launch of kernel `K` resolved for one argument layout, launched as `S`
/// allows (see the [module docs](crate::plan)). Cheap to clone.
///
/// The plan holds its compiled function, so the module stays loaded for as
/// long as the plan, or a launch of it, lives, even across a kernel-cache
/// eviction. After an eviction, the plan and the generated launcher run
/// different instances of the module: device globals (`Global`) written
/// through one are not visible through the other.
pub struct LaunchPlan<K: Kernel, S: PlanSafety = Safe> {
    inner: Arc<PlanInner>,
    _marker: PhantomData<fn() -> (K, S)>,
}

impl<K: Kernel, S: PlanSafety> Clone for LaunchPlan<K, S> {
    fn clone(&self) -> Self {
        LaunchPlan::wrap(Arc::clone(&self.inner))
    }
}

struct PlanInner {
    /// The entry point's path, `module::function`.
    kernel: &'static str,
    function: Arc<Function>,
    cfg: LaunchConfig,
    programmatic_dependent_launch: bool,
    device_id: usize,
    /// Read before the launcher resolved the function.
    epoch: CacheEpoch,
    unsafe_entry: bool,
    options: PlanOptions,
    /// The launch's argument values, as the launcher marshalled them.
    values: Box<[u128]>,
    /// Slot index of each driver argument.
    offsets: Box<[usize]>,
    params: Box<[PlanParam]>,
}

struct PlanParam {
    layout: ArgLayout,
    /// Slot holding this parameter's pointer or scalar value.
    slot: usize,
}

impl<K: Kernel, S: PlanSafety> std::fmt::Debug for LaunchPlan<K, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let plan = &*self.inner;
        f.debug_struct("LaunchPlan")
            .field("kernel", &plan.kernel)
            .field("safety", &std::any::type_name::<S>())
            .field("cfg", &plan.cfg)
            .field("options", &plan.options)
            .field(
                "programmatic_dependent_launch",
                &plan.programmatic_dependent_launch,
            )
            .field("device_id", &plan.device_id)
            .field(
                "params",
                &plan.params.iter().map(|p| &p.layout).collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Collects the plan from a launcher running in plan mode.
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct PlanSink(Arc<Mutex<Option<Arc<PlanInner>>>>);

/// A generated launcher's fully resolved launch, handed to
/// [`PlanSink::capture`] in place of submitting it.
#[doc(hidden)]
pub struct CapturedLaunch<'a> {
    /// The validated launch of the function for `kernel`.
    pub launch: AsyncKernelLaunch,
    /// [`Kernel::PATH`] of the entry point's marker.
    pub kernel: &'static str,
    /// Read before the launcher resolved the function.
    pub epoch: CacheEpoch,
    /// The function's hoisted launch checks, which the launcher has passed.
    pub launch_checks: &'a [LaunchCheck],
    /// Each parameter's layout, in order.
    pub layouts: Vec<Result<ArgLayout, UnsupportedParam>>,
    /// Each parameter's first driver argument, in order: the launch's
    /// [`arg_count`](AsyncKernelLaunch::arg_count) before the parameter
    /// was pushed.
    pub param_starts: &'a [usize],
    /// Whether the entry point is `unsafe`.
    pub unsafe_entry: bool,
    /// The builder options the launcher was given.
    pub options: PlanOptions,
}

impl PlanSink {
    /// Called by the generated launcher in place of submitting its launch.
    ///
    /// # Safety
    /// The captured plan is what [`finish`](Self::finish) returns as a
    /// [`Safe`] plan of `captured.kernel`'s marker, so `captured` must be
    /// exactly what the generated launcher for that entry point produces,
    /// as documented on each field of [`CapturedLaunch`].
    pub unsafe fn capture(
        &self,
        ctx: &ExecutionContext,
        captured: CapturedLaunch<'_>,
    ) -> Result<(), DeviceError> {
        let plan = PlanInner::from_launch(ctx, captured)?;
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(plan));
        Ok(())
    }

    /// The captured plan, as a [`Safe`] plan of `K`.
    ///
    /// The generated `.plan_on()` exists only on standard launchers of safe
    /// entry points, but the check here is still load-bearing: code in the
    /// kernel's module can set a launcher's private sink directly, so this
    /// is what keeps an unsafe entry point or a programmatic dependent
    /// launch from coming back as a `Safe` plan. Do not remove it.
    pub fn finish<K: Kernel>(&self) -> Result<LaunchPlan<K, Safe>, DeviceError> {
        let plan = self.take::<K>()?;
        if plan.unsafe_entry || plan.programmatic_dependent_launch {
            return Err(DeviceError::Internal(format!(
                "a plan of `{}` needs an unsafe launch but was built as `Safe`",
                plan.kernel
            )));
        }
        Ok(LaunchPlan::wrap(plan))
    }

    /// The captured plan, as an [`Unchecked`] plan of `K`.
    pub fn finish_unchecked<K: Kernel>(&self) -> Result<LaunchPlan<K, Unchecked>, DeviceError> {
        Ok(LaunchPlan::wrap(self.take::<K>()?))
    }

    fn take<K: Kernel>(&self) -> Result<Arc<PlanInner>, DeviceError> {
        let plan = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| {
                DeviceError::Launch("the launcher finished without resolving a launch".into())
            })?;
        if plan.kernel != K::PATH {
            return Err(DeviceError::Internal(format!(
                "a plan of `{}` was finished as a plan of `{}`",
                plan.kernel,
                K::PATH
            )));
        }
        Ok(plan)
    }
}

impl PlanInner {
    fn from_launch(
        ctx: &ExecutionContext,
        captured: CapturedLaunch<'_>,
    ) -> Result<Self, DeviceError> {
        let CapturedLaunch {
            launch,
            kernel,
            epoch,
            launch_checks,
            layouts,
            param_starts,
            unsafe_entry,
            options,
        } = captured;
        // Replays skip the launch checks, so each must be a function of what
        // the plan pins; otherwise a replay with new values could violate it.
        if let Some(check) = launch_checks
            .iter()
            .find(|check| !check.predicate.atoms().all(fixed_by_plan))
        {
            return Err(DeviceError::Launch(format!(
                "launch plans do not support this kernel: a launch check depends on more \
                 than argument layouts and the grid ({})",
                check.cause
            )));
        }
        let parts = launch.into_parts();
        let cfg = parts
            .cfg
            .ok_or_else(|| DeviceError::Internal("planned launch has no launch config".into()))?;
        // `submit` builds argument pointers from these offsets unchecked.
        if let Some(&offset) = parts.offsets.iter().find(|&&o| o >= parts.values.len()) {
            return Err(DeviceError::Internal(format!(
                "planned launch has an argument at slot {offset} of {}",
                parts.values.len()
            )));
        }
        // Programmatic dependent launch needs no check here: the launcher ran
        // `validate_programmatic_dependent_launch` for this device before
        // capture, and every replay is refused on any other device.
        if param_starts.len() != layouts.len() {
            return Err(DeviceError::Internal(format!(
                "planned launch recorded {} parameter starts for {} parameters",
                param_starts.len(),
                layouts.len()
            )));
        }
        // No argument may precede the first parameter's.
        if param_starts.first().copied().unwrap_or(parts.offsets.len()) != 0 {
            return Err(DeviceError::Internal(
                "planned launch has arguments outside every parameter".into(),
            ));
        }
        // Each parameter's slots come from where the launcher actually
        // pushed it. A replay writes into the first one, so each span is
        // also checked against its layout: a disagreement would put a
        // pointer or scalar into another argument's slot.
        let mut params = Vec::with_capacity(layouts.len());
        for (i, layout) in layouts.into_iter().enumerate() {
            let layout = layout.map_err(|UnsupportedParam(what)| {
                DeviceError::Launch(format!(
                    "launch plans do not support parameter {i} ({what})"
                ))
            })?;
            let start = param_starts[i];
            let end = param_starts
                .get(i + 1)
                .copied()
                .unwrap_or(parts.offsets.len());
            if end.checked_sub(start) != Some(layout.arg_count()) {
                return Err(DeviceError::Internal(format!(
                    "planned launch marshalled parameter {i} as arguments {start}..{end}; \
                     its layout accounts for {}",
                    layout.arg_count()
                )));
            }
            let slot = *parts.offsets.get(start).ok_or_else(|| {
                DeviceError::Internal(format!(
                    "planned launch has no argument {start} for parameter {i}"
                ))
            })?;
            params.push(PlanParam { layout, slot });
        }
        Ok(PlanInner {
            kernel,
            function: parts.func,
            cfg,
            programmatic_dependent_launch: parts.programmatic_dependent_launch,
            device_id: ctx.get_device_id(),
            epoch,
            unsafe_entry,
            options,
            values: parts.values.into_boxed_slice(),
            offsets: parts.offsets.into_boxed_slice(),
            params: params.into_boxed_slice(),
        })
    }

    /// `args` must be [`PlanArgs`] of this plan's kernel's `Params`: the
    /// seal on [`Kernel`] then guarantees one argument per parameter.
    fn matches<A: ArgTuple>(&self, args: &A) -> bool {
        debug_assert_eq!(A::LEN, self.params.len());
        args.try_for_each(|i, arg| {
            if arg_matches(arg, &self.params[i].layout) {
                Ok(())
            } else {
                Err(())
            }
        })
        .is_ok()
    }

    /// As [`matches`](Self::matches), with the mismatch as an error.
    fn check<A: ArgTuple>(&self, args: &A) -> Result<(), DeviceError> {
        let params = &self.params;
        debug_assert_eq!(A::LEN, params.len());
        args.try_for_each(|i, arg| {
            if arg_matches(arg, &params[i].layout) {
                Ok(())
            } else {
                Err(DeviceError::Launch(format!(
                    "argument {i} does not match the layout the plan was built for: {:?}",
                    params[i].layout
                )))
            }
        })
    }

    /// This plan's [`PlanCache`] key: equal to [`args_key`] of every
    /// device, options and arguments it serves.
    fn cache_key(&self) -> u64 {
        let mut h = key_hasher(self.device_id, &self.options);
        for param in &self.params {
            param.layout.key().hash(&mut h);
        }
        h.finish()
    }

    /// Whether a [`PlanCache`] would serve `other` wherever it serves this
    /// plan: same generation, device, options and layout.
    fn same_entry(&self, other: &PlanInner) -> bool {
        self.epoch == other.epoch
            && self.device_id == other.device_id
            && self.options == other.options
            && self.params.len() == other.params.len()
            && self
                .params
                .iter()
                .zip(&other.params)
                .all(|(p, q)| p.layout == q.layout)
    }

    /// Copies the argument template, lets `write` patch it, and launches.
    ///
    /// # Safety
    /// The patched values must be valid arguments for the plan's kernel.
    unsafe fn submit(
        &self,
        stream: &Stream,
        write: impl FnOnce(&mut [u128]) -> Result<(), DeviceError>,
    ) -> Result<(), DeviceError> {
        const INLINE: usize = 64;
        let (n, m) = (self.values.len(), self.offsets.len());
        if n > INLINE || m > INLINE {
            let mut values = self.values.to_vec();
            write(&mut values)?;
            // One base pointer for every slot: reborrowing `values` per slot
            // would invalidate the pointers taken before it.
            let base = values.as_mut_ptr();
            let mut ptrs: Vec<*mut c_void> = self
                .offsets
                .iter()
                // SAFETY: every offset indexes a slot below `n` (checked in
                // `from_launch`).
                .map(|&o| unsafe { base.add(o) } as *mut c_void)
                .collect();
            return unsafe { self.driver_launch(stream, &mut ptrs) };
        }
        // Stack buffers: the steady state allocates nothing.
        let mut value_buf = MaybeUninit::<[u128; INLINE]>::uninit();
        let mut ptr_buf = MaybeUninit::<[*mut c_void; INLINE]>::uninit();
        // SAFETY: `n, m <= INLINE`; the first `n` values are initialized by
        // the copy before the slice is formed, and each of the first `m`
        // pointers is written before its slice is formed. Every offset
        // indexes a slot below `n` (checked in `from_launch`).
        unsafe {
            let values = value_buf.as_mut_ptr() as *mut u128;
            std::ptr::copy_nonoverlapping(self.values.as_ptr(), values, n);
            let values = std::slice::from_raw_parts_mut(values, n);
            write(values)?;
            let base = values.as_mut_ptr();
            let ptrs = ptr_buf.as_mut_ptr() as *mut *mut c_void;
            for (i, &offset) in self.offsets.iter().enumerate() {
                ptrs.add(i).write(base.add(offset) as *mut c_void);
            }
            self.driver_launch(stream, std::slice::from_raw_parts_mut(ptrs, m))
        }
    }

    /// The checks [`LaunchPlan::launch_raw`] makes of its arguments.
    fn check_raw(&self, stream: &Stream, args: &[RawArg]) -> Result<(), DeviceError> {
        if args.len() != self.params.len() {
            return Err(DeviceError::Launch(format!(
                "launch plan takes {} arguments, got {}",
                self.params.len(),
                args.len()
            )));
        }
        if stream.device().ordinal() != self.device_id {
            return Err(DeviceError::Launch(format!(
                "launch plan was built for device {}, the stream is on device {}",
                self.device_id,
                stream.device().ordinal()
            )));
        }
        // Divisibility is checked as "at least what the kernel assumes". The
        // safe path instead requires the exact hint, because there a
        // different hint means a different layout, and so a different plan.
        for (i, (param, arg)) in self.params.iter().zip(args).enumerate() {
            let ok = match (&param.layout, &arg.0) {
                (ArgLayout::Tensor(t), RawArgKind::Ptr(p)) => {
                    p.is_multiple_of(t.spec.base_ptr_div.divisor as u64)
                }
                (ArgLayout::Pointer { hint, .. }, RawArgKind::Ptr(p)) => {
                    p.is_multiple_of(hint.divisor as u64)
                }
                (
                    ArgLayout::Scalar { dtype, hint },
                    RawArgKind::Scalar {
                        dtype: d,
                        hint: raw,
                        ..
                    },
                ) => dtype == d && hint.is_none_or(|h| raw.is_some_and(|r| r.divisor >= h.divisor)),
                _ => false,
            };
            if !ok {
                return Err(DeviceError::Launch(format!(
                    "raw argument {i} ({arg:?}) does not satisfy the plan's layout {:?}",
                    param.layout
                )));
            }
        }
        Ok(())
    }

    /// Launches with `args` written over the template.
    ///
    /// # Safety
    /// `args` must pass [`check_raw`](Self::check_raw), and satisfy
    /// [`LaunchPlan::launch_raw`]'s safety contract.
    #[inline]
    unsafe fn submit_raw(&self, stream: &Stream, args: &[RawArg]) -> Result<(), DeviceError> {
        // SAFETY: the caller's.
        unsafe {
            self.submit(stream, |values| {
                for (param, arg) in self.params.iter().zip(args) {
                    values[param.slot] = match arg.0 {
                        RawArgKind::Ptr(p) => scalar_bits(p),
                        RawArgKind::Scalar { bits, .. } => bits,
                    };
                }
                Ok(())
            })
        }
    }

    unsafe fn driver_launch(
        &self,
        stream: &Stream,
        params: &mut [*mut c_void],
    ) -> Result<(), DeviceError> {
        let launch = if self.programmatic_dependent_launch {
            launch_kernel_pdl
        } else {
            launch_kernel
        };
        unsafe {
            launch(
                self.function.cu_function(),
                self.cfg.grid_dim,
                self.cfg.block_dim,
                self.cfg.shared_mem_bytes,
                stream.cu_stream(),
                params,
            )?
        };
        Ok(())
    }
}

impl<K: Kernel, S: PlanSafety> LaunchPlan<K, S> {
    fn wrap(inner: Arc<PlanInner>) -> Self {
        LaunchPlan {
            inner,
            _marker: PhantomData,
        }
    }

    /// The path of the entry point this plan launches, `module::function`.
    /// Generic instances of one entry point share it.
    pub fn kernel(&self) -> &'static str {
        K::PATH
    }

    /// The builder options this plan was built with.
    pub fn options(&self) -> &PlanOptions {
        &self.inner.options
    }

    /// The compiled function the plan launches. Tests observe its lifetime.
    #[doc(hidden)]
    pub fn function(&self) -> &Arc<Function> {
        &self.inner.function
    }

    /// Whether the plan was built with programmatic dependent launch.
    pub fn is_programmatic_dependent_launch(&self) -> bool {
        self.inner.programmatic_dependent_launch
    }

    /// Whether `args` have exactly the layout this plan was built for.
    pub fn matches<A: PlanArgs<K::Params>>(&self, args: &A) -> bool {
        self.inner.matches(args)
    }

    /// Launches this plan on `stream` with raw pointers, bypassing cutile's
    /// tensors, access tracking and execution lock. One [`RawArg`] per kernel
    /// parameter, in order: a pointer for each tensor or pointer parameter,
    /// a scalar for each scalar parameter.
    ///
    /// Checks the argument count and kinds, scalar dtypes, the device, and
    /// that every pointer and integer scalar satisfies the divisibility the
    /// kernel was specialized for. Extents cannot be checked from a raw
    /// pointer; they are the caller's to uphold.
    ///
    /// Returns a [`RawLaunch`] guard that keeps the kernel's module loaded.
    ///
    /// # Safety
    /// Each pointer must address a live device allocation on `stream`'s
    /// device holding at least the extent (shape and strides) the plan was
    /// built with, in the plan's dtype, and must stay valid until the kernel
    /// completes. An output's extent must not overlap any other argument's,
    /// as the borrows of a safe launch guarantee. The caller orders all
    /// conflicting accesses, on this and every other stream. The returned
    /// guard must not be dropped before the kernel completes. For an
    /// [`Unchecked`] plan, the caller also upholds the obligations listed on
    /// [`launch_unchecked`](LaunchPlan::launch_unchecked).
    pub unsafe fn launch_raw(
        &self,
        stream: &Stream,
        args: &[RawArg],
    ) -> Result<RawLaunch, DeviceError> {
        let plan = &*self.inner;
        plan.check_raw(stream, args)?;
        // SAFETY: argument kinds and values were checked against the plan
        // above; the caller guarantees the pointers and keeps the returned
        // guard, and so the module, until the kernel completes.
        unsafe { plan.submit_raw(stream, args)? };
        Ok(RawLaunch {
            _function: Arc::clone(&plan.function),
        })
    }

    /// [`launch_raw`](Self::launch_raw) without its checks or its guard,
    /// for when every nanosecond of host overhead counts: it copies the
    /// argument template, writes `args` into it, and calls the driver.
    ///
    /// Nothing about `args` is checked in release builds: not their count,
    /// kinds or scalar dtypes, the stream's device, or divisibility. Debug
    /// builds run `launch_raw`'s checks and panic if one fails.
    ///
    /// # Safety
    /// Everything [`launch_raw`](Self::launch_raw) requires, and also every
    /// condition it checks: exactly one [`RawArg`] per kernel parameter, of
    /// the parameter's kind and, for a scalar, its dtype; `stream` on the
    /// device the plan was built for; and every pointer and integer scalar
    /// at least as divisible as the plan's layout records. With no guard
    /// returned, this plan, or a clone of it, must outlive the kernel, so
    /// the module stays loaded until it completes.
    #[inline]
    pub unsafe fn launch_raw_unchecked(
        &self,
        stream: &Stream,
        args: &[RawArg],
    ) -> Result<(), DeviceError> {
        let plan = &*self.inner;
        #[cfg(debug_assertions)]
        if let Err(e) = plan.check_raw(stream, args) {
            panic!("launch_raw_unchecked: {e}");
        }
        // SAFETY: the caller upholds every condition `launch_raw` checks,
        // guarantees the pointers, and keeps the plan, and so the module,
        // alive until the kernel completes.
        unsafe { plan.submit_raw(stream, args) }
    }
}

/// A kernel submitted by [`LaunchPlan::launch_raw`]. Holds the compiled
/// function, so the kernel's module stays loaded while the guard lives,
/// even if a kernel-cache eviction drops every other reference.
///
/// Keep it until the kernel completes, such as until its stream is
/// synchronized. `launch_raw`'s caller promises not to drop it sooner.
#[must_use = "dropping a RawLaunch before its kernel completes may unload the running module"]
pub struct RawLaunch {
    _function: Arc<Function>,
}

impl<K: Kernel> LaunchPlan<K, Safe> {
    /// A launch of this plan over `args`, as a [`DeviceOp`] that returns the
    /// arguments. Validates the layout and retains every tensor like a
    /// generated launcher does, so it is safe, and graph-recordable.
    pub fn launch<A: PlanArgs<K::Params>>(&self, args: A) -> PlannedLaunch<A> {
        PlannedLaunch::new(Arc::clone(&self.inner), args, LayoutCheck::Pending)
    }
}

impl<K: Kernel> LaunchPlan<K, Unchecked> {
    /// [`launch`](LaunchPlan::launch) for an [`Unchecked`] plan. The layout
    /// is still checked and every tensor still retained.
    ///
    /// # Safety
    /// For a plan of an `unsafe` entry point, the caller upholds that entry
    /// point's contract, as if calling its launcher. For a plan built with
    /// programmatic dependent launch, the caller upholds that opt-in's
    /// contract for *this* launch and its predecessor on the stream: every
    /// predecessor-dependent access is token-ordered after `gdc_wait_tko`,
    /// work before the wait does not race the predecessor, neither kernel
    /// depends on overlap for progress, and resources outlive both kernels.
    ///
    /// If the launch is recorded into a graph, these obligations hold for
    /// every replay of that graph, with each replay's predecessor.
    pub unsafe fn launch_unchecked<A: PlanArgs<K::Params>>(&self, args: A) -> PlannedLaunch<A> {
        PlannedLaunch::new(Arc::clone(&self.inner), args, LayoutCheck::Pending)
    }
}

/// Whether a [`PlannedLaunch`] still has to check its arguments' layout.
enum LayoutCheck {
    Pending,
    /// Already matched against the plan, by [`PlanCache::launch`].
    Done,
}

/// A [`LaunchPlan`] bound to its arguments; see [`LaunchPlan::launch`].
///
/// Only a [`Safe`] plan's `launch`, or an `unsafe` caller of
/// [`LaunchPlan::launch_unchecked`], can create one, so executing it needs no
/// further safety check.
#[must_use = "a PlannedLaunch does nothing until it is synced or awaited"]
pub struct PlannedLaunch<A> {
    plan: Arc<PlanInner>,
    args: A,
    layout: LayoutCheck,
}

impl<A> PlannedLaunch<A> {
    fn new(plan: Arc<PlanInner>, args: A, layout: LayoutCheck) -> Self {
        PlannedLaunch { plan, args, layout }
    }
}

impl<A: ArgTuple> DeviceOp for PlannedLaunch<A> {
    type Output = A;

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<A, DeviceError> {
        let plan = &*self.plan;
        if ctx.get_device_id() != plan.device_id {
            return Err(DeviceError::Launch(format!(
                "launch plan was built for device {}, the stream is on device {}",
                plan.device_id,
                ctx.get_device_id()
            )));
        }
        if let LayoutCheck::Pending = self.layout {
            plan.check(&self.args)?;
        }
        self.args.try_for_each(|_, arg| arg.retain(ctx))?;
        // The kernel cache may no longer hold the function (a plan outlives
        // evictions), so this launch may hold the last reference to its
        // module. Keep it until the kernel completes, or for as long as a
        // recording graph can replay it: unloading a module whose grid is
        // running is undefined behavior.
        if ctx.is_recording() {
            ctx.record_resource(Arc::new(FunctionLease(Arc::clone(&plan.function))));
        } else {
            ctx.retain(Arc::clone(&plan.function))?;
        }
        // SAFETY: the layout matches the plan and every tensor is retained
        // for the submission, which is what the generated launcher
        // establishes before it submits; the function outlives the kernel.
        unsafe {
            plan.submit(ctx.get_cuda_stream(), |values| {
                self.args.try_for_each(|i, arg| {
                    values[plan.params[i].slot] = arg.slot_value();
                    Ok(())
                })
            })?
        };
        Ok(self.args)
    }
}

// Launch only: no allocation, and every access is registered by `retain`.
impl<A: ArgTuple> GraphNode for PlannedLaunch<A> {}

/// Keeps a recorded launch's function loaded for as long as the graph lives.
/// A replay needs nothing more: the graph holds this lease, and each replay
/// holds the graph until it completes.
struct FunctionLease(Arc<Function>);

impl ReplayResource for FunctionLease {
    fn retain_for_launch(&self, _ctx: &ExecutionContext) -> Result<(), DeviceError> {
        Ok(())
    }
    fn replay_identity(&self) -> (usize, bool) {
        // Distinct from every storage identity: a different live allocation.
        (Arc::as_ptr(&self.0) as usize, false)
    }
}

impl<A: ArgTuple> IntoFuture for PlannedLaunch<A> {
    type Output = Result<A, DeviceError>;
    type IntoFuture = DeviceFuture<A, PlannedLaunch<A>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

// ── Plan cache ──────────────────────────────────────────────────────────────

/// Seals [`Kernel`] and [`PlanLauncher`], which have it as a supertrait:
/// implementing either by hand takes an `unsafe impl`. Not part of the
/// public API.
///
/// # Safety
/// Only `#[cutile::entry]` implements this. Plans rely on every [`Kernel`]
/// being a generated marker, whose `PATH` and `Params` are exactly its entry
/// point's, and on every [`PlanLauncher`] being that entry point's generated
/// launcher.
#[doc(hidden)]
pub unsafe trait EntryGenerated {}

/// Names one entry point in types. `#[cutile::entry]` generates a marker for
/// every entry point `f`, `f::Kernel`, alongside the launcher function `f`.
/// Only generated markers implement it.
pub trait Kernel: EntryGenerated + 'static {
    /// The entry point's path, as [`LaunchPlan::kernel`] reports it.
    const PATH: &'static str;
    /// The entry point's parameter kinds, in order: a tuple of [`param`]
    /// markers. A plan's arguments must be [`PlanArgs<Params>`](PlanArgs),
    /// which covers at most 12 parameters.
    type Params;
}

/// A generated launcher that can build a [`Safe`] [`LaunchPlan`].
/// Implemented by `#[cutile::entry]` for safe entry points only: plans of
/// `unsafe` ones are [`Unchecked`], which a [`PlanCache`] cannot hold.
pub trait PlanLauncher: EntryGenerated + Sized {
    /// The entry point's marker, `f::Kernel`.
    type Kernel: Kernel;
    /// The arguments the launch returns.
    type Args;
    /// `self.with_options(options).plan_on(stream)`.
    fn plan_with(
        self,
        options: &PlanOptions,
        stream: &Arc<Stream>,
    ) -> Result<(LaunchPlan<Self::Kernel>, Self::Args), DeviceError>;
}

/// One kernel's [`Safe`] plans, keyed by device, builder options and
/// argument layout, built on demand. [`plan_launch!`](crate::plan_launch)
/// keeps one per call site.
///
/// ```rust,ignore
/// static ADD: PlanCache<kernels::add::Kernel> = PlanCache::new();
///
/// let opts = PlanOptions::new().grid((rows as u32, 1, 1));
/// let (z, _, _) = ADD
///     .launch(&opts, (z.partition_prefix([128]), &x, &y), &stream, |(z, x, y)| {
///         kernels::add(z, x, y)
///     })?
///     .sync_on(&stream)?;
/// ```
///
/// On a miss, the closure turns the arguments back into a launcher, and the
/// cache applies `options` to it and builds the plan on `stream`, which it
/// synchronizes. The cache cannot store a wrong plan: the types require a
/// [`Safe`] plan of kernel `K` and a launcher that returns the arguments'
/// type, the cache's options override whatever the closure set, and the
/// plan must have been built from the caller's own arguments, in order.
///
/// A cache is unbounded: it keeps one plan per device, options and layout
/// until [`clear`](Self::clear) or a kernel-cache eviction, so use it for a
/// bounded set of shapes. Where shapes are unbounded (one per sequence
/// length, say), launch through the generated launcher instead.
///
/// Every eviction from the kernel cache empties every `PlanCache`, as it
/// does every launch site, so a cache never keeps an evicted module loaded.
/// (A [`LaunchPlan`] held elsewhere still does, until it drops.)
pub struct PlanCache<K: Kernel> {
    /// Created by the first insert, which registers it in [`PLAN_STORES`].
    plans: OnceLock<Arc<PlanStore>>,
    _kernel: PhantomData<fn() -> K>,
}

/// A cache's plans, type-erased so that evictions can reach every cache.
/// Every entry is a [`Safe`] plan of the owning cache's kernel: only
/// [`PlanCache::insert`] adds to it.
type PlanStore = RwLock<PlanMap>;

#[derive(Default)]
struct PlanMap {
    /// The most recently stored plan, also in `by_key`: a call site that
    /// launches at one layout finds it without hashing.
    newest: Option<Arc<PlanInner>>,
    /// Every plan, by [`PlanInner::cache_key`]; a bucket holds the rare
    /// plans whose keys collide.
    by_key: HashMap<u64, Vec<Arc<PlanInner>>, FxBuildHasher>,
}

/// A hasher seeded with the parts of a [`PlanCache`] key that are not
/// argument layouts.
fn key_hasher(device_id: usize, options: &PlanOptions) -> FxHasher {
    let mut h = FxHasher::default();
    device_id.hash(&mut h);
    options.hash(&mut h);
    h
}

/// The [`PlanCache`] key of a lookup for `args` on `device_id` with
/// `options`.
fn args_key<A: ArgTuple>(device_id: usize, options: &PlanOptions, args: &A) -> u64 {
    let mut h = key_hasher(device_id, options);
    let _ = args.try_for_each(|_, arg| {
        arg.layout_key().hash(&mut h);
        Ok::<_, std::convert::Infallible>(())
    });
    h.finish()
}

/// Every plan cache that has stored a plan; evictions empty them all. Weak,
/// so a dropped cache is not kept alive; dead entries are pruned whenever
/// a cache registers.
static PLAN_STORES: Mutex<Vec<Weak<PlanStore>>> = Mutex::new(Vec::new());

/// Empties every plan cache. Called by every kernel-cache eviction, after it
/// bumps the epoch. The plans are dropped after the locks are released:
/// dropping the last reference to a function unloads its module.
pub(crate) fn clear_plan_caches() {
    let mut stores = PLAN_STORES.lock().unwrap_or_else(PoisonError::into_inner);
    stores.retain(|store| store.strong_count() > 0);
    let live: Vec<Arc<PlanStore>> = stores.iter().filter_map(Weak::upgrade).collect();
    let drained: Vec<PlanMap> = live
        .iter()
        .map(|store| std::mem::take(&mut *store.write().unwrap_or_else(PoisonError::into_inner)))
        .collect();
    drop(stores);
    drop(drained);
    drop(live);
}

impl<K: Kernel> PlanCache<K> {
    pub const fn new() -> Self {
        PlanCache {
            plans: OnceLock::new(),
            _kernel: PhantomData,
        }
    }

    /// The store, creating and registering it on first use.
    fn store(&self) -> &PlanStore {
        self.plans.get_or_init(|| {
            let store = Arc::new(PlanStore::default());
            let mut stores = PLAN_STORES.lock().unwrap_or_else(PoisonError::into_inner);
            stores.retain(|s| s.strong_count() > 0);
            stores.push(Arc::downgrade(&store));
            store
        })
    }

    fn read(&self) -> Option<std::sync::RwLockReadGuard<'_, PlanMap>> {
        let store = self.plans.get()?;
        Some(store.read().unwrap_or_else(PoisonError::into_inner))
    }

    /// The cached plan for `stream`'s device, built with `options`, whose
    /// layout `args` match, if any.
    ///
    /// The device is part of the key because the layout alone need not name
    /// one: a kernel without tensor parameters has the same layout on every
    /// device, and a plan launches only on the device it was built for.
    pub fn get<A: PlanArgs<K::Params>>(
        &self,
        options: &PlanOptions,
        args: &A,
        stream: &Stream,
    ) -> Option<LaunchPlan<K>> {
        let plans = self.read()?;
        let device_id = stream.device().ordinal();
        // Evictions empty every cache, but only after bumping the epoch;
        // this rejects a plan in between.
        let epoch = CacheEpoch::now();
        let serves = |plan: &&Arc<PlanInner>| {
            plan.epoch == epoch
                && plan.device_id == device_id
                && &plan.options == options
                && plan.matches(args)
        };
        let plan = match plans.newest.as_ref().filter(serves) {
            Some(plan) => plan,
            None => plans
                .by_key
                .get(&args_key(device_id, options, args))?
                .iter()
                .find(serves)?,
        };
        Some(LaunchPlan::wrap(Arc::clone(plan)))
    }

    /// Adds `plan`, unless the cache already holds one for the same device,
    /// options and layout (as when two threads miss at once), which it keeps.
    /// A plan resolved before the latest eviction is not stored: its function
    /// may be one the eviction removed.
    pub fn insert(&self, plan: LaunchPlan<K>) {
        let plan = plan.inner;
        let key = plan.cache_key();
        // No eviction outlives a stored plan, by the argument on
        // `LaunchSite::store`: `store()` registers this cache before the
        // epoch is checked under its lock, so an eviction either bumped
        // before the check (which then discards) or empties the cache after
        // the plan is in.
        let mut plans = self.store().write().unwrap_or_else(PoisonError::into_inner);
        let keep = plan.epoch.is_current()
            && !plans
                .by_key
                .get(&key)
                .is_some_and(|bucket| bucket.iter().any(|p| p.same_entry(&plan)));
        let released = if keep {
            plans.by_key.entry(key).or_default().push(Arc::clone(&plan));
            plans.newest.replace(plan)
        } else {
            Some(plan)
        };
        // Unlocked: a stale plan may hold the last reference to its module.
        drop(plans);
        drop(released);
    }

    /// A launch over `args` with the cached plan for `stream`'s device,
    /// `options`, and their layout. On a miss, `launcher` turns the arguments
    /// back into this kernel's launcher, `|(a, b)| kernel(a, b)`, and the
    /// cache plans it with `options` on `stream`, synchronizing `stream`.
    /// On a hit, the returned launch does not re-check the layout: the
    /// lookup already matched it.
    ///
    /// `launcher` must pass the arguments through unchanged and in order: a
    /// miss fails if the plan was built from other pointers or scalars.
    pub fn launch<A, L>(
        &self,
        options: &PlanOptions,
        args: A,
        stream: &Arc<Stream>,
        launcher: impl FnOnce(A) -> L,
    ) -> Result<PlannedLaunch<A>, DeviceError>
    where
        A: PlanArgs<K::Params>,
        L: PlanLauncher<Kernel = K, Args = A>,
    {
        if let Some(plan) = self.get(options, &args, stream) {
            return Ok(PlannedLaunch::new(plan.inner, args, LayoutCheck::Done));
        }
        let mut given = Vec::with_capacity(A::LEN);
        args.try_for_each(|_, arg| {
            given.push(arg.slot_value());
            Ok::<_, DeviceError>(())
        })?;
        let (plan, args) = launcher(args).plan_with(options, stream)?;
        // The seal makes `L` a generated launcher, whose `plan_with` applies
        // `options` whole.
        debug_assert_eq!(plan.options(), options);
        plan.inner.check(&args)?;
        // A closure that reorders or replaces arguments of the same layout
        // would pass `check`, and this launch would then differ from every
        // cache hit. Require the plan to have marshalled, and the launcher to
        // have returned, exactly the caller's pointers and scalars.
        args.try_for_each(|i, arg| {
            let marshalled = plan.inner.values[plan.inner.params[i].slot];
            if arg.slot_value() == given[i] && marshalled == given[i] {
                Ok(())
            } else {
                Err(DeviceError::Launch(format!(
                    "the launcher closure changed argument {i}; it must pass the \
                     arguments through unchanged and in order, as `|(a, b)| kernel(a, b)`"
                )))
            }
        })?;
        self.insert(plan.clone());
        Ok(PlannedLaunch::new(plan.inner, args, LayoutCheck::Done))
    }

    pub fn len(&self) -> usize {
        self.read()
            .map_or(0, |plans| plans.by_key.values().map(Vec::len).sum())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        let Some(store) = self.plans.get() else {
            return;
        };
        // Dropped unlocked, as in `clear_plan_caches`.
        let drained = std::mem::take(&mut *store.write().unwrap_or_else(PoisonError::into_inner));
        drop(drained);
    }
}

impl<K: Kernel> Default for PlanCache<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// Launches a kernel through a plan cache owned by this call site.
///
/// `plan_launch!(kernel(args..), &options, &stream)` evaluates to a
/// `Result<`[`PlannedLaunch`]`, DeviceError>`: the cached plan for the
/// arguments' layout and `options`, built on a miss as
/// `kernel(args..).with_options(options).plan_on(stream)`.
///
/// ```rust,ignore
/// let opts = PlanOptions::new();
/// let (z, _, _) = plan_launch!(kernels::add(z.partition([128]), &x, &y), &opts, &stream)?
///     .sync_on(&stream)?;
/// ```
///
/// `kernel` is a path to a safe entry point with at most 12 parameters. Set
/// kernel generics through `options`.
///
/// The call site's cache is a [`PlanCache`](crate::plan::PlanCache): it
/// never evicts on its own, and keeps a plan for every layout the call site
/// launches, so use it where the set of shapes is bounded.
#[macro_export]
macro_rules! plan_launch {
    ($($kernel:ident)::+ ( $($arg:expr),* $(,)? ), $options:expr, $stream:expr $(,)?) => {{
        static __CUTILE_PLANS: $crate::plan::PlanCache<$($kernel)::+::Kernel> =
            $crate::plan::PlanCache::new();
        __CUTILE_PLANS.launch($options, ($($arg,)*), $stream, |__args| {
            $crate::plan::SpreadArgs::spread(__args, $($kernel)::+)
        })
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(options: &PlanOptions) -> PlanOptions {
        let (grid, const_grid) = options.launcher_grid();
        PlanOptions::from_launcher(grid, const_grid, None, CompileOptions::default())
    }

    #[test]
    fn grid_options_round_trip_through_the_launcher() {
        for options in [
            PlanOptions::new(),
            PlanOptions::new().grid((0, 0, 0)),
            PlanOptions::new().grid((4, 1, 1)),
            PlanOptions::new().const_grid((4, 1, 1)),
            PlanOptions::new().const_grid((0, 0, 0)),
        ] {
            assert_eq!(round_trip(&options), options);
        }
        assert_eq!(PlanOptions::new().grid((0, 0, 0)), PlanOptions::new());
    }

    /// Plans hint exactly the scalar types the launcher hints: another
    /// hinted type would make a plan's layout disagree with its launch.
    #[test]
    fn plan_scalars_are_hinted_as_the_launcher_hints_them() {
        fn case<T: PlanScalar>() -> (&'static str, bool) {
            (T::DTYPE.as_str(), T::one().div_hint().is_some())
        }
        for (name, hinted) in [
            case::<bool>(),
            case::<u8>(),
            case::<u16>(),
            case::<u32>(),
            case::<u64>(),
            case::<i8>(),
            case::<i16>(),
            case::<i32>(),
            case::<i64>(),
            case::<f16>(),
            case::<bf16>(),
            case::<f32>(),
            case::<tf32>(),
            case::<f64>(),
            case::<f8e4m3fn>(),
            case::<f8e5m2>(),
            case::<f8e8m0fnu>(),
            case::<f8e5m3fnu>(),
            case::<f4e2m1fnx2>(),
        ] {
            assert_eq!(
                hinted,
                cutile_compiler::specialization::is_integer_scalar(name),
                "{name}"
            );
        }
    }
}
