/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Launch plans: a plan replays exactly what the generated launcher would
//! have launched, refuses arguments of any other layout, and its raw form
//! refuses values that break the kernel's specialization. A plan cache keys
//! on builder options as well as layout, and holds only plans that launch
//! safely.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cuda_async::cuda_graph::CudaGraph;
use cuda_core::{Device, Stream};
use cutile::plan::{Kernel, LaunchPlan, PlanCache, PlanOptions, RawArg};
use cutile::plan_launch;
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use cutile_ir::requirements::Feature;

use crate::common;

#[cutile::module]
mod plan_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn axpy<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
        a: f32,
    ) {
        let tx = x.load_like(z);
        let ty = y.load_like(z);
        let s: Tile<f32, { [B] }> = a.broadcast(z.shape());
        z.store(tx * s + ty);
    }

    /// One block: copies tile `n / B` of `x`. `n` is an integer scalar, so
    /// the kernel is specialized on its divisibility.
    #[cutile::entry()]
    fn pick<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>, n: i32) {
        let p = x.partition(shape![B]);
        z.store(p.load([n / B]));
    }

    /// Only the module-lifetime tests resolve this kernel: they observe
    /// when its function is released, which another holder would mask.
    #[cutile::entry()]
    fn scale<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>, a: f32) {
        let tx = x.load_like(z);
        let s: Tile<f32, { [B] }> = a.broadcast(z.shape());
        z.store(tx * s);
    }

    unsafe fn view<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        make_tensor_view(ptr_tile, shape, strides, new_token_unordered())
    }

    /// An `unsafe` entry point over tensors only: its plans must keep the
    /// `unsafe` obligation even though every argument is a tensor.
    #[cutile::entry()]
    unsafe fn double<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let tx = x.load_like(z);
        z.store(tx + tx);
    }

    #[cutile::entry()]
    unsafe fn add_ptr<T: ElementType>(z_ptr: *mut T, x_ptr: *mut T, y_ptr: *mut T, len: i32) {
        let mut z: Tensor<T, { [-1] }> = view(z_ptr, len);
        let x: Tensor<T, { [-1] }> = view(x_ptr, len);
        let y: Tensor<T, { [-1] }> = view(y_ptr, len);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = shape![4i32];
        let tx = x.partition(tile_shape).load([pid.0]);
        let ty = y.partition(tile_shape).load([pid.0]);
        z.partition_mut(tile_shape).store(tx + ty, [pid.0]);
    }
}

use plan_module::{add_ptr, axpy, double, pick, scale};

const B: usize = 16;

fn stream() -> Arc<Stream> {
    Device::new(0)
        .expect("device")
        .new_stream()
        .expect("stream")
}

fn filled(len: usize, f: impl Fn(usize) -> f32, stream: &Arc<Stream>) -> Tensor<f32> {
    let host: Vec<f32> = (0..len).map(f).collect();
    api::copy_host_vec_to_device(&Arc::new(host))
        .sync_on(stream)
        .expect("upload")
}

fn host(t: &Tensor<f32>, stream: &Arc<Stream>) -> Vec<f32> {
    t.dup().to_host_vec().sync_on(stream).expect("download")
}

fn ptr(t: &Tensor<f32>) -> u64 {
    t.device_pointer().cu_deviceptr()
}

#[test]
fn plan_replays_match_the_launcher_bit_for_bit() {
    common::with_test_stack(|| {
        let s = stream();
        let len = 64;
        let x = filled(len, |i| i as f32 * 0.37, &s);
        let y = filled(len, |i| 1.0 / (i as f32 + 1.0), &s);
        let mut z = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");

        let (plan, _) = axpy((&mut z).partition([B]), &x, &y, 2.0f32)
            .plan_on(&s)
            .expect("plan");
        assert!(
            host(&z, &s).iter().all(|&v| v == 0.0),
            "building a plan must not launch"
        );

        // Fresh tensors (new pointers), a new scalar, same layout.
        for (round, a) in [3.0f32, -0.5, 1e-3].into_iter().enumerate() {
            let x2 = filled(len, |i| (i + round) as f32 * 1.1, &s);
            let y2 = filled(len, |i| (i * round) as f32 - 7.0, &s);
            let mut planned = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
            let mut launched = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
            plan.launch(((&mut planned).partition([B]), &x2, &y2, a))
                .sync_on(&s)
                .expect("planned launch");
            axpy((&mut launched).partition([B]), &x2, &y2, a)
                .sync_on(&s)
                .expect("launcher");
            let (p, l) = (host(&planned, &s), host(&launched, &s));
            assert!(
                p.iter().zip(&l).all(|(a, b)| a.to_bits() == b.to_bits()),
                "round {round}: plan {p:?} != launcher {l:?}"
            );
        }
    });
}

#[test]
fn plan_refuses_other_layouts() {
    common::with_test_stack(|| {
        let s = stream();
        let x = filled(64, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        let (plan, _) = axpy((&mut z).partition([B]), &x, &x, 1.0f32)
            .plan_on(&s)
            .expect("plan");

        let x48 = filled(48, |i| i as f32, &s);
        let mut z48 = api::zeros::<f32>(&[48]).sync_on(&s).expect("z");
        let args = ((&mut z48).partition([B]), &x48, &x48, 1.0f32);
        assert!(!plan.matches(&args));
        let err = plan.launch(args).sync_on(&s).err().expect("other shape");
        assert!(err.to_string().contains("does not match"), "{err}");

        // Same shapes, other partition: a different launch.
        let err = plan
            .launch(((&mut z).partition([8]), &x, &x, 1.0f32))
            .sync_on(&s)
            .err()
            .expect("other partition");
        assert!(err.to_string().contains("does not match"), "{err}");
    });
}

#[test]
fn plan_cache_serves_alternating_shapes() {
    common::with_test_stack(|| {
        // Counts cache entries; evictions in other tests would empty them.
        let _guard = common::cache_test_lock();
        let s = stream();
        let cache = PlanCache::<axpy::Kernel>::new();
        let opts = PlanOptions::new();
        let builds = Cell::new(0);
        for round in 0..3 {
            // Two layouts through one call site: one plan each.
            for len in [64usize, 48] {
                let x = filled(len, |i| i as f32, &s);
                let mut z = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
                let a = round as f32 + 1.0;
                cache
                    .launch(
                        &opts,
                        ((&mut z).partition([B]), &x, &x, a),
                        &s,
                        |(z, x, y, a)| {
                            builds.set(builds.get() + 1);
                            axpy(z, x, y, a)
                        },
                    )
                    .expect("cache lookup")
                    .sync_on(&s)
                    .expect("cached launch");
                let out = host(&z, &s);
                for (i, v) in out.iter().enumerate() {
                    assert_eq!(*v, i as f32 * a + i as f32, "len {len}, index {i}");
                }
            }
        }
        assert_eq!(builds.get(), 2, "one build per layout");
        assert_eq!(cache.len(), 2);
    });
}

#[test]
fn raw_launch_writes_only_pointers_and_scalars() {
    common::with_test_stack(|| {
        let s = stream();
        let len = 64;
        let x = filled(len, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
        let (plan, _) = axpy((&mut z).partition([B]), &x, &x, 1.0f32)
            .plan_on(&s)
            .expect("plan");

        let x2 = filled(len, |i| 2.0 * i as f32, &s);
        let y2 = filled(len, |_| 5.0, &s);
        let z2 = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
        unsafe {
            let launch = plan
                .launch_raw(
                    &s,
                    &[
                        RawArg::ptr(ptr(&z2)),
                        RawArg::ptr(ptr(&x2)),
                        RawArg::ptr(ptr(&y2)),
                        RawArg::scalar(0.5f32),
                    ],
                )
                .expect("raw launch");
            s.synchronize().expect("sync");
            drop(launch);
        }
        let out = host(&z2, &s);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as f32 + 5.0, "index {i}");
        }

        // The unchecked form writes the same slots.
        let z3 = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
        unsafe {
            plan.launch_raw_unchecked(
                &s,
                &[
                    RawArg::ptr(ptr(&z3)),
                    RawArg::ptr(ptr(&y2)),
                    RawArg::ptr(ptr(&x2)),
                    RawArg::scalar(2.0f32),
                ],
            )
            .expect("unchecked raw launch");
            s.synchronize().expect("sync");
        }
        let out = host(&z3, &s);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 10.0 + 2.0 * i as f32, "index {i}");
        }

        let good = [
            RawArg::ptr(ptr(&z2)),
            RawArg::ptr(ptr(&x2)),
            RawArg::ptr(ptr(&y2)),
            RawArg::scalar(0.5f32),
        ];
        // The kernel assumes 16-byte-aligned bases.
        let mut misaligned = good;
        misaligned[1] = RawArg::ptr(ptr(&x2) + 4);
        // A scalar of another dtype.
        let mut wrong_dtype = good;
        wrong_dtype[3] = RawArg::scalar(0.5f64);
        // A pointer where the kernel takes a scalar.
        let mut wrong_kind = good;
        wrong_kind[3] = RawArg::ptr(ptr(&y2));
        for (what, args) in [
            ("misaligned", &misaligned[..]),
            ("wrong dtype", &wrong_dtype[..]),
            ("wrong kind", &wrong_kind[..]),
            ("too few", &good[..3]),
        ] {
            let result = unsafe { plan.launch_raw(&s, args) };
            assert!(result.is_err(), "{what} must be refused");
            // Debug builds run the same checks in the unchecked form.
            if cfg!(debug_assertions) {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    plan.launch_raw_unchecked(&s, args)
                }));
                assert!(result.is_err(), "{what} must panic in a debug build");
            }
        }
    });
}

#[test]
fn integer_scalars_keep_their_divisibility() {
    common::with_test_stack(|| {
        let s = stream();
        let x = filled(64, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[B]).sync_on(&s).expect("z");
        // Built with n = 16: the kernel assumes n divisible by 16.
        let (plan, _) = pick((&mut z).partition([B]), &x, 16i32)
            .grid((1, 1, 1))
            .plan_on(&s)
            .expect("plan");

        // n = 32 has the same hint: tile 2.
        plan.launch(((&mut z).partition([B]), &x, 32i32))
            .sync_on(&s)
            .expect("same hint");
        assert_eq!(host(&z, &s)[0], 32.0);

        // n = 8 would break the assumption.
        assert!(plan
            .launch(((&mut z).partition([B]), &x, 8i32))
            .sync_on(&s)
            .is_err());
        let raw = |n: i32| {
            [
                RawArg::ptr(ptr(&z)),
                RawArg::ptr(ptr(&x)),
                RawArg::scalar(n),
            ]
        };
        assert!(unsafe { plan.launch_raw(&s, &raw(8)) }.is_err());
        // The raw form accepts any value at least as divisible.
        unsafe {
            let launch = plan.launch_raw(&s, &raw(48)).expect("n = 48");
            s.synchronize().expect("sync");
            drop(launch);
        }
        assert_eq!(host(&z, &s)[0], 48.0);
    });
}

#[test]
fn unsafe_entry_points_replay_through_the_raw_form() {
    common::with_test_stack(|| {
        let s = stream();
        let len = 32usize;
        let x = filled(len, |i| i as f32, &s);
        let y = filled(len, |_| 1.0, &s);
        let z = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
        let (plan, _) = unsafe {
            add_ptr(
                z.device_pointer(),
                x.device_pointer(),
                y.device_pointer(),
                len as i32,
            )
        }
        .grid(((len / 4) as u32, 1, 1))
        .plan_unchecked_on(&s)
        .expect("plan");

        let x2 = filled(len, |i| 3.0 * i as f32, &s);
        unsafe {
            let launch = plan
                .launch_raw(
                    &s,
                    &[
                        RawArg::ptr(ptr(&z)),
                        RawArg::ptr(ptr(&x2)),
                        RawArg::ptr(ptr(&y)),
                        RawArg::scalar(len as i32),
                    ],
                )
                .expect("raw launch");
            s.synchronize().expect("sync");
            drop(launch);
        }
        let out = host(&z, &s);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 3.0 * i as f32 + 1.0, "index {i}");
        }
    });
}

#[test]
fn planned_launches_record_into_graphs() {
    common::with_test_stack(|| {
        let s = stream();
        let len = 64;
        let x = filled(len, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
        let (plan, _) = axpy((&mut z).partition([B]), &x, &x, 1.0f32)
            .plan_on(&s)
            .expect("plan");

        let y = filled(len, |_| 1.0, &s);
        let mut out = api::zeros::<f32>(&[len]).sync_on(&s).expect("out");
        let graph = CudaGraph::scope(&s, |g| {
            g.record(plan.launch(((&mut out).partition([B]), &x, &y, 2.0f32)))?;
            Ok(())
        })
        .expect("capture");
        assert!(
            host(&out, &s).iter().all(|&v| v == 0.0),
            "capture must not run"
        );
        graph.launch().sync_on(&s).expect("replay");
        let out = host(&out, &s);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 2.0 * i as f32 + 1.0, "index {i}");
        }
    });
}

#[test]
fn plans_record_their_builder_options() {
    common::with_test_stack(|| {
        let s = stream();
        let x = filled(64, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");

        let (plan, _) = axpy((&mut z).partition([B]), &x, &x, 1.0f32)
            .plan_on(&s)
            .expect("plan");
        assert_eq!(plan.options(), &PlanOptions::new());

        // Any non-default option; `lineinfo` is valid in every build profile.
        let mut compile_options = CompileOptions::default();
        compile_options.lineinfo = !compile_options.lineinfo;
        let opts = PlanOptions::new()
            .grid((4, 1, 1))
            .generics(vec!["16".to_string()])
            .compile_options(compile_options);
        let (plan, _) = axpy((&mut z).partition([B]), &x, &x, 1.0f32)
            .with_options(&opts)
            .plan_on(&s)
            .expect("plan");
        assert_eq!(plan.options(), &opts);
    });
}

#[test]
fn plan_cache_keys_on_the_launch_grid() {
    common::with_test_stack(|| {
        // Counts cache entries; evictions in other tests would empty them.
        let _guard = common::cache_test_lock();
        let s = stream();
        let cache = PlanCache::<axpy::Kernel>::new();
        let builds = Cell::new(0);
        let len = 64; // four tiles of 16
        let x = filled(len, |i| i as f32, &s);
        let y = filled(len, |_| 1.0, &s);
        // A prefix partition leaves the grid to the caller: the layout is
        // identical for every grid, so only the options tell the plans apart.
        // Serving the 2-block plan for a 4-block request would silently
        // leave half of `z` unwritten.
        for tiles in [2u32, 4, 2, 4] {
            let opts = PlanOptions::new().grid((tiles, 1, 1));
            let mut z = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
            cache
                .launch(
                    &opts,
                    ((&mut z).partition_prefix([B]), &x, &y, 1.0f32),
                    &s,
                    |(z, x, y, a)| {
                        builds.set(builds.get() + 1);
                        axpy(z, x, y, a)
                    },
                )
                .expect("cache lookup")
                .sync_on(&s)
                .expect("cached launch");
            let out = host(&z, &s);
            let covered = tiles as usize * B;
            for (i, v) in out.iter().enumerate() {
                let expected = if i < covered { i as f32 + 1.0 } else { 0.0 };
                assert_eq!(*v, expected, "{tiles} blocks, index {i}");
            }
        }
        assert_eq!(builds.get(), 2, "one build per grid");
        assert_eq!(cache.len(), 2);
    });
}

#[test]
fn plans_are_typed_by_their_kernel() {
    common::with_test_stack(|| {
        // Reads cache entries; evictions in other tests would empty them.
        let _guard = common::cache_test_lock();
        let s = stream();
        let x = filled(64, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[B]).sync_on(&s).expect("z");
        let (axpy_plan, _): (LaunchPlan<axpy::Kernel>, _) =
            axpy((&mut z).partition([B]), &x, &x, 1.0f32)
                .plan_on(&s)
                .expect("axpy plan");
        let (pick_plan, _): (LaunchPlan<pick::Kernel>, _) =
            pick((&mut z).partition([B]), &x, 16i32)
                .grid((1, 1, 1))
                .plan_on(&s)
                .expect("pick plan");
        assert_eq!(axpy_plan.kernel(), <axpy::Kernel as Kernel>::PATH);
        assert!(axpy_plan.kernel().ends_with("plan_module::axpy"));
        assert_eq!(pick_plan.kernel(), <pick::Kernel as Kernel>::PATH);

        // A cache takes only its own kernel's plans; handing it another's is
        // a type error (see the UI tests).
        let cache = PlanCache::<axpy::Kernel>::new();
        cache.insert(axpy_plan);
        assert_eq!(cache.len(), 1);
    });
}

#[test]
fn plan_launch_keeps_a_cache_per_call_site() {
    common::with_test_stack(|| {
        let s = stream();
        let opts = PlanOptions::new();
        let one_block = PlanOptions::new().grid((1, 1, 1));
        for round in 0..3 {
            for len in [64usize, 48] {
                let x = filled(len, |i| i as f32, &s);
                let mut z = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
                let a = round as f32 + 1.0;
                plan_launch!(axpy((&mut z).partition([B]), &x, &x, a), &opts, &s)
                    .expect("axpy lookup")
                    .sync_on(&s)
                    .expect("axpy launch");
                let out = host(&z, &s);
                for (i, v) in out.iter().enumerate() {
                    assert_eq!(*v, i as f32 * a + i as f32, "len {len}, index {i}");
                }

                // Another kernel, same arguments' shapes, its own cache.
                let mut t = api::zeros::<f32>(&[B]).sync_on(&s).expect("t");
                plan_launch!(pick((&mut t).partition([B]), &x, 16i32), &one_block, &s)
                    .expect("pick lookup")
                    .sync_on(&s)
                    .expect("pick launch");
                assert_eq!(host(&t, &s)[0], 16.0);
            }
        }
    });
}

#[test]
fn plan_cache_options_override_the_launcher() {
    common::with_test_stack(|| {
        // Reads cache entries; evictions in other tests would empty them.
        let _guard = common::cache_test_lock();
        let s = stream();
        let cache = PlanCache::<axpy::Kernel>::new();
        let x = filled(64, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        let asked = PlanOptions::new().grid((2, 1, 1));
        cache
            .launch(
                &asked,
                ((&mut z).partition_prefix([B]), &x, &x, 1.0f32),
                &s,
                // A different grid than the key: the cache's options win, so
                // the plan cannot be filed under the wrong key.
                |(z, x, y, a)| axpy(z, x, y, a).grid((4, 1, 1)),
            )
            .expect("cache lookup")
            .sync_on(&s)
            .expect("cached launch");
        let out = host(&z, &s);
        for (i, v) in out.iter().enumerate() {
            let expected = if i < 2 * B { 2.0 * i as f32 } else { 0.0 };
            assert_eq!(*v, expected, "index {i}");
        }
        let args = ((&mut z).partition_prefix([B]), &x, &x, 1.0f32);
        let plan = cache
            .get(&asked, &args, &s)
            .expect("filed under the asked options");
        assert_eq!(plan.options(), &asked);
    });
}

#[test]
fn plan_cache_refuses_a_closure_that_changes_the_arguments() {
    common::with_test_stack(|| {
        // Counts cache entries; evictions in other tests would empty them.
        let _guard = common::cache_test_lock();
        let s = stream();
        let cache = PlanCache::<axpy::Kernel>::new();
        let x = filled(64, |i| i as f32, &s);
        let y = filled(64, |_| 5.0, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        let opts = PlanOptions::new();

        // Same layouts, so the plan's layout check alone would accept these,
        // and the first launch would differ from every later cache hit.
        let swapped = cache
            .launch(
                &opts,
                ((&mut z).partition([B]), &x, &y, 1.0f32),
                &s,
                |(z, x, y, a)| axpy(z, y, x, a),
            )
            .err()
            .expect("swapped inputs must be refused");
        assert!(
            swapped.to_string().contains("changed argument 1"),
            "{swapped}"
        );
        let rescaled = cache
            .launch(
                &opts,
                ((&mut z).partition([B]), &x, &y, 1.0f32),
                &s,
                |(z, x, y, a)| axpy(z, x, y, a * 2.0),
            )
            .err()
            .expect("a changed scalar must be refused");
        assert!(
            rescaled.to_string().contains("changed argument 3"),
            "{rescaled}"
        );
        assert!(cache.is_empty());

        // The same tensor twice is indistinguishable, and harmless, swapped.
        cache
            .launch(
                &opts,
                ((&mut z).partition([B]), &x, &x, 1.0f32),
                &s,
                |(z, x, y, a)| axpy(z, y, x, a),
            )
            .expect("aliased swap")
            .sync_on(&s)
            .expect("launch");
        assert_eq!(cache.len(), 1);
    });
}

#[test]
fn owned_tensor_inputs_plan_and_replay() {
    common::with_test_stack(|| {
        let s = stream();
        let opts = PlanOptions::new();
        let y = filled(64, |_| 5.0, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        let mut x = filled(64, |i| i as f32, &s);
        for round in 0..3 {
            let a = round as f32 + 1.0;
            // The launcher returns the owned `x`, which feeds the next round.
            let (_, x_back, _, _) =
                plan_launch!(axpy((&mut z).partition([B]), x, &y, a), &opts, &s)
                    .expect("lookup")
                    .sync_on(&s)
                    .expect("launch");
            x = x_back;
            let out = host(&z, &s);
            for (i, v) in out.iter().enumerate() {
                assert_eq!(*v, i as f32 * a + 5.0, "round {round}, index {i}");
            }
        }

        // An explicit plan replays with an owned input too.
        let (plan, (_, x, _, _)) = axpy((&mut z).partition([B]), x, &y, 1.0f32)
            .plan_on(&s)
            .expect("plan");
        plan.launch(((&mut z).partition([B]), x, &y, 3.0f32))
            .sync_on(&s)
            .expect("replay");
        let out = host(&z, &s);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as f32 * 3.0 + 5.0, "index {i}");
        }
    });
}

#[test]
fn plans_of_unsafe_entry_points_need_unsafe_launches() {
    common::with_test_stack(|| {
        let s = stream();
        let x = filled(64, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        // An `unsafe` entry point has only `plan_unchecked_on`, whose plan
        // has no safe `launch` and no place in a `PlanCache` (see the UI
        // tests).
        let (plan, _) = unsafe { double((&mut z).partition([B]), &x) }
            .plan_unchecked_on(&s)
            .expect("plan");

        unsafe { plan.launch_unchecked(((&mut z).partition([B]), &x)) }
            .sync_on(&s)
            .expect("unsafe launch");
        let out = host(&z, &s);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 2.0 * i as f32, "index {i}");
        }

        // The layout is still checked on the unsafe path.
        let mut short = api::zeros::<f32>(&[32]).sync_on(&s).expect("short");
        let err = unsafe { plan.launch_unchecked(((&mut short).partition([B]), &x)) }
            .sync_on(&s)
            .err()
            .expect("other layout");
        assert!(err.to_string().contains("does not match"), "{err}");
    });
}

#[test]
fn plans_with_programmatic_dependent_launch_need_unsafe_launches() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[Feature::ProgrammaticDependentLaunch]) {
            return;
        }
        let s = stream();
        let x = filled(64, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");

        // A `Safe` plan cannot carry the opt-in: an opted-in launcher has no
        // `plan_on` and fills no `PlanCache` (tests/ui/plan_safety_is_typed.rs).
        // SAFETY: `axpy` reads only buffers that are complete before it is
        // enqueued (every earlier operation here synchronizes the stream),
        // so overlapping its predecessor cannot race.
        let (plan, _) = unsafe {
            axpy((&mut z).partition([B]), &x, &x, 1.0f32).programmatic_dependent_launch()
        }
        .plan_unchecked_on(&s)
        .expect("plan");
        assert!(plan.is_programmatic_dependent_launch());

        // SAFETY: as above; the stream is idle when this is enqueued.
        unsafe { plan.launch_unchecked(((&mut z).partition([B]), &x, &x, 1.0f32)) }
            .sync_on(&s)
            .expect("unsafe launch");
        let out = host(&z, &s);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 2.0 * i as f32, "index {i}");
        }
    });
}

#[test]
fn zero_const_grids_are_refused_by_launchers_and_plans() {
    common::with_test_stack(|| {
        let s = stream();
        let x = filled(64, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        // The kernel would be specialized on an empty grid while the launch
        // inferred another; both paths refuse it instead.
        let err = axpy((&mut z).partition([B]), &x, &x, 1.0f32)
            .const_grid((0, 0, 0))
            .sync_on(&s)
            .err()
            .expect("zero const grid launch");
        assert!(err.to_string().contains("const grid"), "{err}");
        let err = axpy((&mut z).partition([B]), &x, &x, 1.0f32)
            .with_options(&PlanOptions::new().const_grid((0, 0, 0)))
            .plan_on(&s)
            .err()
            .expect("zero const grid plan");
        assert!(err.to_string().contains("const grid"), "{err}");
    });
}

#[test]
fn plan_caches_keep_one_plan_per_options_and_layout() {
    common::with_test_stack(|| {
        // Counts cache entries; evictions in other tests would empty them.
        let _guard = common::cache_test_lock();
        let s = stream();
        let cache = PlanCache::<axpy::Kernel>::new();
        let x = filled(64, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        // Two plans resolved independently for one layout, as when two
        // threads miss at once: the cache keeps the first.
        let (first, _) = axpy((&mut z).partition([B]), &x, &x, 1.0f32)
            .plan_on(&s)
            .expect("first plan");
        let (second, _) = axpy((&mut z).partition([B]), &x, &x, 2.0f32)
            .plan_on(&s)
            .expect("second plan");
        cache.insert(first.clone());
        cache.insert(second);
        cache.insert(first);
        assert_eq!(cache.len(), 1);

        // Other options are another entry, even an explicit grid equal to
        // the inferred one.
        let (gridded, _) = axpy((&mut z).partition([B]), &x, &x, 1.0f32)
            .grid((4, 1, 1))
            .plan_on(&s)
            .expect("gridded plan");
        cache.insert(gridded);
        assert_eq!(cache.len(), 2);
    });
}

#[test]
fn evictions_empty_every_plan_cache() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        static STATIC: PlanCache<axpy::Kernel> = PlanCache::new();
        let s = stream();
        let local = PlanCache::<axpy::Kernel>::new();
        let x = filled(64, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        let opts = PlanOptions::new();
        let fill = |cache: &PlanCache<axpy::Kernel>, z: &mut Tensor<f32>| {
            cache
                .launch(
                    &opts,
                    (z.partition([B]), &x, &x, 1.0f32),
                    &s,
                    |(z, x, y, a)| axpy(z, x, y, a),
                )
                .expect("cache lookup")
                .sync_on(&s)
                .expect("cached launch");
        };
        fill(&STATIC, &mut z);
        fill(&local, &mut z);
        assert_eq!((STATIC.len(), local.len()), (1, 1));

        // SAFETY: every launch above was synchronized.
        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };
        assert!(STATIC.is_empty(), "a static cache is emptied");
        assert!(local.is_empty(), "a local cache is emptied");

        // A dropped cache is skipped (and pruned) by later evictions.
        drop(local);
        // SAFETY: as above.
        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };

        // The emptied cache rebuilds on the next launch.
        fill(&STATIC, &mut z);
        assert_eq!(STATIC.len(), 1);
        let out = host(&z, &s);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 2.0 * i as f32, "index {i}");
        }
    });
}

#[test]
fn view_inputs_replay_at_their_offset() {
    common::with_test_stack(|| {
        let s = stream();
        let x = filled(128, |i| i as f32, &s);
        let y = filled(64, |i| 1000.0 + i as f32, &s);
        let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        let built = x.slice(&[16..80]).expect("slice");
        let (plan, _) = axpy((&mut z).partition([B]), &built, &y, 2.0f32)
            .plan_on(&s)
            .expect("plan");

        // Another window of the same tensor: same layout, another offset. The
        // replay must read from the view's own start, not the tensor's.
        let window = x.slice(&[32..96]).expect("slice");
        let mut planned = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        let mut launched = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
        plan.launch(((&mut planned).partition([B]), &window, &y, 2.0f32))
            .sync_on(&s)
            .expect("planned launch");
        axpy((&mut launched).partition([B]), &window, &y, 2.0f32)
            .sync_on(&s)
            .expect("launcher");
        let (p, l) = (host(&planned, &s), host(&launched, &s));
        assert_eq!(p, l, "plan and launcher disagree");
        for (i, v) in p.iter().enumerate() {
            assert_eq!(*v, 2.0 * (32 + i) as f32 + 1000.0 + i as f32, "index {i}");
        }
    });
}

#[test]
fn plan_cache_inserts_racing_evictions_keep_only_current_plans() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let s = stream();
        let cache = PlanCache::<axpy::Kernel>::new();
        let opts = PlanOptions::new();
        let x = filled(64, |i| i as f32, &s);
        // One layout per worker: two partitions of the same output shape.
        let tiles = [B, B / 2];
        for round in 0..3 {
            let building = AtomicUsize::new(tiles.len());
            std::thread::scope(|scope| {
                for tile in tiles {
                    let (cache, x, s, building) = (&cache, &x, &s, &building);
                    scope.spawn(move || {
                        let mut z = api::zeros::<f32>(&[64]).sync_on(s).expect("z");
                        for _ in 0..8 {
                            let (plan, _) = axpy((&mut z).partition([tile]), x, x, 1.0f32)
                                .plan_on(s)
                                .expect("plan");
                            cache.insert(plan);
                        }
                        building.fetch_sub(1, Ordering::Release);
                    });
                }
                // Evict until every plan is in, so the last inserts race an
                // eviction with no later one to clean up after them.
                scope.spawn(|| {
                    while building.load(Ordering::Acquire) > 0 {
                        // SAFETY: this test launches no kernel; building a
                        // plan only resolves one.
                        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };
                        std::thread::sleep(Duration::from_millis(1));
                    }
                });
            });
            // Every stored plan is current, so the cache serves everything it
            // holds: a stale plan would count in `len` but never be served.
            let mut z = api::zeros::<f32>(&[64]).sync_on(&s).expect("z");
            let served = tiles
                .iter()
                .filter(|&&tile| {
                    cache
                        .get(&opts, &((&mut z).partition([tile]), &x, &x, 1.0f32), &s)
                        .is_some()
                })
                .count();
            assert_eq!(cache.len(), served, "round {round}: unserved plans held");
        }
    });
}

/// A launch keeps its plan's function loaded until the kernel completes.
/// After an eviction the plan may hold the last reference, and unloading a
/// module whose grid is still running is undefined behavior.
#[test]
fn planned_launches_keep_their_module_loaded_until_complete() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let s = stream();
        let len = 64;
        let x = filled(len, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
        let (plan, _) = scale((&mut z).partition([B]), &x, 1.0f32)
            .plan_on(&s)
            .expect("plan");
        // SAFETY: building a plan launches nothing.
        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };
        let function = Arc::downgrade(plan.function());
        let op = plan.launch(((&mut z).partition([B]), &x, 3.0f32));
        drop(plan);

        // Runs after the launch is submitted, before the stream is synced:
        // the launch op has dropped its reference by then.
        let loaded_in_flight = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&loaded_in_flight);
        let observed = function.clone();
        op.inspect(move |_| flag.store(observed.upgrade().is_some(), Ordering::SeqCst))
            .sync_on(&s)
            .expect("planned launch");
        assert!(
            loaded_in_flight.load(Ordering::SeqCst),
            "the submission must hold the function until the kernel completes"
        );
        assert!(
            function.upgrade().is_none(),
            "the function is released once the launch completes"
        );
        let out = host(&z, &s);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 3.0 * i as f32, "index {i}");
        }
    });
}

/// A recorded planned launch keeps its function loaded for as long as the
/// graph can replay it, even after the plan and the kernel cache drop it.
#[test]
fn graphs_keep_their_planned_modules_loaded() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let s = stream();
        let len = 64;
        let x = filled(len, |i| i as f32, &s);
        let mut z = api::zeros::<f32>(&[len]).sync_on(&s).expect("z");
        let (plan, _) = scale((&mut z).partition([B]), &x, 1.0f32)
            .plan_on(&s)
            .expect("plan");
        let function = Arc::downgrade(plan.function());
        let graph = CudaGraph::scope(&s, |g| {
            g.record(plan.launch(((&mut z).partition([B]), &x, 3.0f32)))?;
            Ok(())
        })
        .expect("capture");
        drop(plan);
        // SAFETY: nothing has run; the capture launched nothing.
        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };
        assert!(function.upgrade().is_some(), "the graph holds the function");

        graph.launch().sync_on(&s).expect("replay");
        drop(graph);
        assert!(
            function.upgrade().is_none(),
            "the function is released with the graph"
        );
        let out = host(&z, &s);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 3.0 * i as f32, "index {i}");
        }
    });
}
