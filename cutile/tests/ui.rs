/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-fail tests: API misuse the type system must reject.
//!
//! Each case in `tests/ui/*.rs` is a program that must NOT compile, paired
//! with the expected diagnostics in the `.stderr` file next to it. These need
//! no GPU; they only build against `cutile`. Regenerate the expectations with
//! `TRYBUILD=overwrite cargo test -p cutile --test ui` after an intentional
//! diagnostic change.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    // `api::memcpy` holds bare device pointers; the borrow it carries is what
    // stops the copy from executing after its tensors were freed.
    t.compile_fail("tests/ui/memcpy_outlives_tensors.rs");
    // A launcher is only a `GraphNode` when its argument op is; an allocating
    // input (`api::zeros(..).partition(..)`) cannot be recorded into a scope.
    t.compile_fail("tests/ui/graph_scope_rejects_allocating_input.rs");
    // A plan cache's closure must produce the cache's kernel, and plans of
    // `unsafe` entry points cannot be cached.
    t.compile_fail("tests/ui/plan_cache_is_typed_by_kernel.rs");
    // Plans that need an `unsafe` launch are `Unchecked` in their type: no
    // safe `launch`, no `PlanCache`, and no `Safe` plan of an `unsafe` entry
    // point or an opted-in PDL launcher. A cache takes only its own kernel's
    // plans.
    t.compile_fail("tests/ui/plan_safety_is_typed.rs");
    // A plan's arguments must match its kernel's parameter count and kinds.
    t.compile_fail("tests/ui/plan_args_are_typed.rs");
    // A kernel with a `MappedPartitionMut` parameter cannot be planned: its
    // launcher has no plan methods, and its `Params` accept no arguments.
    t.compile_fail("tests/ui/plan_rejects_mapped_partitions.rs");
    // `#[cutile::entry(..)]` keys and literal kinds are checked at expansion:
    // a typo is no longer silently ignored, a non-literal no longer panics
    // inside the JIT at first launch.
    t.compile_fail("tests/ui/entry_unknown_key.rs");
    t.compile_fail("tests/ui/entry_non_literal_value.rs");
    t.compile_fail("tests/ui/global_*.rs");
    t.compile_fail("tests/ui/unordered_partition_requires_unsafe.rs");
}
