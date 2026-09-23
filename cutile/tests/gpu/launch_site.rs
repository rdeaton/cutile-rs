/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The per-launch-site resolution cache: steady-state launches skip key
//! construction entirely, so these tests pin the two ways a site must NOT
//! serve a stale resolution — a changed specialization (alternating
//! generics through one call site) and an evicted global cache.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cuda_async::device_context::Validator;
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{Device, Function};
use cutile::prelude::*;
use cutile::tile_kernel::{
    compile_from_context, contains_cuda_function, CacheEpoch, CompileOptions, LaunchSite,
    SiteResolution,
};

use crate::common;

#[cutile::module]
mod site_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn scale<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let tile = x.load_like(z);
        z.store(tile + tile);
    }
}

use site_module::scale;

fn run(len: usize, tile: usize) -> Vec<f32> {
    scale(
        api::arange::<f32>(len).partition([tile]),
        api::arange::<f32>(len),
    )
    .grid(((len / tile) as u32, 1, 1))
    .first()
    .unpartition()
    .to_host_vec()
    .sync()
    .expect("scale kernel")
}

#[test]
fn alternating_specializations_through_one_site_stay_correct() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        // Two tile sizes = two specializations through the same call site.
        // The site keeps both resolutions; results must stay correct.
        for _ in 0..3 {
            for tile in [4usize, 8] {
                let host = run(32, tile);
                for (i, v) in host.iter().enumerate() {
                    assert_eq!(*v, 2.0 * i as f32, "tile {tile}, index {i}");
                }
            }
        }
    });
}

#[test]
fn cache_eviction_invalidates_hot_launch_sites() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        // Key-scoped observations: concurrent tests in this binary compile
        // their own kernels, so global compile counts race, but this key is
        // ours alone.
        let key = scale(
            api::arange::<f32>(32).partition([16]),
            api::arange::<f32>(32),
        )
        .generics(vec!["16".to_string()])
        .l1_cache_key()
        .expect("key");
        run(32, 16); // fill the site (and the global cache)
        assert!(contains_cuda_function(&key), "filled");
        // Quiesced (nothing of ours in flight after sync): evicting must
        // force the hot site to re-resolve, not serve its stale
        // Arc<Function> — the epoch check.
        unsafe {
            cutile::tile_kernel::clear_kernel_cache_for_tests();
        }
        assert!(!contains_cuda_function(&key), "evicted");
        let host = run(32, 16);
        assert!(
            contains_cuda_function(&key),
            "a launch after the clear re-resolved through the global cache"
        );
        for (i, v) in host.iter().enumerate() {
            assert_eq!(*v, 2.0 * i as f32, "index {i}");
        }
    });
}

/// Resolves `scale` for `B = 16` as the generated launcher does: the epoch
/// is read before the function is.
fn resolve(ctx: &ExecutionContext) -> (CacheEpoch, Arc<Function>, Arc<Validator>) {
    let epoch = CacheEpoch::now();
    let (function, validator) = compile_from_context(
        ctx,
        site_module::__module_ast_self,
        "site_module",
        "scale",
        "scale_entry",
        vec!["16".to_string()],
        vec![("z".to_string(), vec![1]), ("x".to_string(), vec![1])],
        vec![],
        vec![],
        None,
        CompileOptions::default(),
        site_module::_SOURCE_HASH,
    )
    .expect("compile");
    (epoch, function, validator)
}

/// The site entry for a [`resolve`] result.
fn resolution(
    epoch: CacheEpoch,
    function: &Arc<Function>,
    validator: &Arc<Validator>,
) -> SiteResolution {
    SiteResolution::new(
        epoch,
        0,
        vec!["16".to_string()],
        vec![],
        vec![],
        None,
        CompileOptions::default(),
        Arc::clone(function),
        Arc::clone(validator),
    )
}

/// The probe of every [`resolution`], as `LaunchSite::get` takes it.
fn probe(site: &LaunchSite) -> Option<(Arc<Function>, Arc<Validator>)> {
    site.get(
        0,
        &["16".to_string()],
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
}

#[test]
fn launch_sites_never_keep_an_evicted_function() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        static SITE: LaunchSite = LaunchSite::new();
        let stream = Device::new(0)
            .expect("device")
            .new_stream()
            .expect("stream");
        let ctx = ExecutionContext::new(stream);

        // An eviction between resolving and storing: the function it removed
        // must not be stored, or the site would keep its module loaded.
        let (epoch, function, validator) = resolve(&ctx);
        // SAFETY: nothing has launched this kernel.
        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };
        assert!(!epoch.is_current());
        SITE.store(resolution(epoch, &function, &validator));
        assert_eq!(
            Arc::strong_count(&function),
            1,
            "stale resolution discarded"
        );

        // Stored while current, then evicted: the eviction empties the site.
        let (epoch, function, validator) = resolve(&ctx);
        SITE.store(resolution(epoch, &function, &validator));
        assert_eq!(
            Arc::strong_count(&function),
            3,
            "held by the site and the cache"
        );
        // SAFETY: as above.
        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };
        assert_eq!(
            Arc::strong_count(&function),
            1,
            "site emptied by the eviction"
        );
    });
}

#[test]
fn launch_site_stores_racing_evictions_keep_only_current_entries() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        static SITE: LaunchSite = LaunchSite::new();
        let stream = Device::new(0)
            .expect("device")
            .new_stream()
            .expect("stream");
        let ctx = ExecutionContext::new(stream);
        let (_, function, validator) = resolve(&ctx);
        // SAFETY: nothing has launched this kernel. From here on only this
        // test's handles hold the function, so the site's share is visible
        // in its count.
        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };

        for round in 0..20 {
            let storing = AtomicUsize::new(4);
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(|| {
                        for _ in 0..20 {
                            // Stamped as a launcher stamps it; the pause
                            // stands in for resolving, so an eviction often
                            // lands between the stamp and the store.
                            let epoch = CacheEpoch::now();
                            std::thread::sleep(Duration::from_micros(50));
                            SITE.store(resolution(epoch, &function, &validator));
                        }
                        storing.fetch_sub(1, Ordering::Release);
                    });
                }
                // Evict until every store is in, so the last stores race an
                // eviction with no later one to clean up after them.
                scope.spawn(|| {
                    while storing.load(Ordering::Acquire) > 0 {
                        // SAFETY: nothing launches this kernel.
                        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };
                        std::thread::sleep(Duration::from_micros(10));
                    }
                });
            });
            // Every stored entry is current, so the site holds the function
            // exactly when it serves it: a stale entry would hold it unserved.
            let served = probe(&SITE).is_some();
            assert_eq!(
                Arc::strong_count(&function),
                1 + served as usize,
                "round {round}: the site holds only entries it serves"
            );
        }
        // SAFETY: as above.
        unsafe { cutile::tile_kernel::clear_kernel_cache_for_tests() };
        assert_eq!(Arc::strong_count(&function), 1, "emptied");
    });
}
