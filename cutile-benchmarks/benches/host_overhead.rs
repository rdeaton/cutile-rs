/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side cost of enqueueing work, isolated from GPU time.
//!
//! Times only the enqueue (`async_on`), never the wait: a batch of launches
//! is timed, then the stream is drained outside the timer so the launch
//! queue never fills. Two shapes that matter for decode-style serving:
//!
//! - `launch`: one tiny 4-tensor kernel launch (argument retention, launch
//!   validation, `cuLaunchKernel`).
//! - `graph_replay`: one replay of a graph recording 32 such launches over
//!   8 buffers (per-replay resource reacquisition plus `cuGraphLaunch`).
//! - `plan_launch` / `plan_raw` / `plan_raw_unchecked`: the same launch
//!   replayed from a [`LaunchPlan`](cutile::plan::LaunchPlan), safely,
//!   through raw pointers, and through raw pointers without checks.
//! - `*_alternating`: two launches per iteration whose inputs alternate
//!   between two specializations (extent divisible by 16, and by 8 only),
//!   through the generated launcher and through a `PlanCache`.
//! - `launch_12_specializations`: twelve launches per iteration, each at a
//!   different specialization, through the generated launcher: a kernel run
//!   at many shapes must stay on its launch site's fast path.
//! - `plan_cache_12_specializations`: the same twelve launches through a
//!   `PlanCache`, which must not scan every plan on each launch;
//!   `plan_cache_get_12_specializations` times the lookups alone.

use criterion::{criterion_group, criterion_main, Criterion};
use cuda_async::cuda_graph::CudaGraph;
use cuda_core::{Device, Stream};
use cutile::plan::{PlanCache, PlanOptions, RawArg};
use cutile::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn add3(
        z: &mut Tensor<f32, { [128] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
        c: &Tensor<f32, { [-1] }>,
    ) {
        let ta: Tile<f32, { [128] }> = a.load_like(z);
        let tb: Tile<f32, { [128] }> = b.load_like(z);
        let tc: Tile<f32, { [128] }> = c.load_like(z);
        z.store(ta + tb + tc);
    }
}

const N: usize = 128;
/// Launches (or replays) per drain of the stream.
const BATCH: u64 = 32;

fn host_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("host_overhead");
    if cfg!(feature = "smoke-test") {
        group
            .warm_up_time(Duration::from_millis(1))
            .sample_size(10)
            .measurement_time(Duration::from_millis(1));
    } else {
        group
            .warm_up_time(Duration::from_millis(500))
            .sample_size(100)
            .measurement_time(Duration::from_millis(2000));
    }

    let device = Device::new(0).expect("device");
    let stream = device.new_stream().expect("stream");

    let a: Arc<Tensor<f32>> = api::ones::<f32>(&[N]).sync_on(&stream).expect("a").into();
    let b: Arc<Tensor<f32>> = api::ones::<f32>(&[N]).sync_on(&stream).expect("b").into();
    let c3: Arc<Tensor<f32>> = api::ones::<f32>(&[N]).sync_on(&stream).expect("c").into();
    let fresh_z = || -> Partition<Tensor<f32>> {
        api::zeros::<f32>(&[N])
            .sync_on(&stream)
            .expect("z")
            .partition([N])
    };

    // JIT + steady-state warmup.
    let mut z = fresh_z();
    for _ in 0..200 {
        let (local_z, _, _, _) = kernels::add3(z, a.clone(), b.clone(), c3.clone())
            .sync_on(&stream)
            .expect("warmup");
        z = local_z;
    }
    drop(z);

    group.bench_function("launch", |bench| {
        bench.iter_custom(|iters| {
            let mut z = fresh_z();
            let mut elapsed = Duration::ZERO;
            let mut done = 0;
            while done < iters {
                let batch = BATCH.min(iters - done);
                let start = Instant::now();
                for _ in 0..batch {
                    let (local_z, _, _, _) = unsafe {
                        kernels::add3(z, a.clone(), b.clone(), c3.clone())
                            .async_on(&stream)
                            .expect("launch")
                    };
                    z = local_z;
                }
                elapsed += start.elapsed();
                unsafe { stream.synchronize() }.expect("drain");
                done += batch;
            }
            elapsed
        });
    });

    // ── Launch plans ────────────────────────────────────────────────────
    let mut zt = api::zeros::<f32>(&[N]).sync_on(&stream).expect("zt");
    let (plan, _) = kernels::add3((&mut zt).partition([N]), &*a, &*b, &*c3)
        .plan_on(&stream)
        .expect("plan");

    group.bench_function("plan_launch", |bench| {
        bench.iter_custom(|iters| {
            enqueue_time(iters, &stream, || {
                unsafe {
                    plan.launch(((&mut zt).partition([N]), &*a, &*b, &*c3))
                        .async_on(&stream)
                }
                .expect("planned launch");
            })
        });
    });

    let ptr = |t: &Tensor<f32>| t.device_pointer().cu_deviceptr();
    let raw_args = [
        RawArg::ptr(ptr(&zt)),
        RawArg::ptr(ptr(&a)),
        RawArg::ptr(ptr(&b)),
        RawArg::ptr(ptr(&c3)),
    ];
    group.bench_function("plan_raw", |bench| {
        bench.iter_custom(|iters| {
            enqueue_time(iters, &stream, || {
                // `plan` outlives every launch, so the guard can drop here.
                let _launch = unsafe { plan.launch_raw(&stream, &raw_args) }.expect("raw launch");
            })
        });
    });

    group.bench_function("plan_raw_unchecked", |bench| {
        bench.iter_custom(|iters| {
            enqueue_time(iters, &stream, || {
                // `plan` outlives every launch, keeping the module loaded.
                unsafe { plan.launch_raw_unchecked(&stream, &raw_args) }.expect("raw launch");
            })
        });
    });

    // ── Alternating specializations ─────────────────────────────────────
    // Inputs of 128 elements (divisible by 16) and 136 (by 8 only): the
    // same kernel at two specializations, as a model forward runs it.
    let odd = |_| -> Arc<Tensor<f32>> {
        api::ones::<f32>(&[N + 8])
            .sync_on(&stream)
            .expect("odd")
            .into()
    };
    let (a2, b2, c2) = (odd(()), odd(()), odd(()));
    kernels::add3((&mut zt).partition([N]), &*a2, &*b2, &*c2)
        .sync_on(&stream)
        .expect("warm second specialization");

    group.bench_function("launch_alternating", |bench| {
        bench.iter_custom(|iters| {
            enqueue_time(iters, &stream, || {
                for (x, y, w) in [(&a, &b, &c3), (&a2, &b2, &c2)] {
                    unsafe {
                        kernels::add3((&mut zt).partition([N]), &**x, &**y, &**w).async_on(&stream)
                    }
                    .expect("launch");
                }
            })
        });
    });

    let cache = PlanCache::<kernels::add3::Kernel>::new();
    let options = PlanOptions::new();
    group.bench_function("plan_cache_alternating", |bench| {
        bench.iter_custom(|iters| {
            enqueue_time(iters, &stream, || {
                for (x, y, w) in [(&a, &b, &c3), (&a2, &b2, &c2)] {
                    let op = cache
                        .launch(
                            &options,
                            ((&mut zt).partition([N]), &**x, &**y, &**w),
                            &stream,
                            |(z, x, y, w)| kernels::add3(z, x, y, w),
                        )
                        .expect("plan cache");
                    unsafe { op.async_on(&stream) }.expect("planned launch");
                }
            })
        });
    });

    // Extents divisible by 16, 8, 4, 2 and 1: `a` and `b` at independent
    // divisibilities give twelve specializations of one kernel.
    let extents = [N, N + 8, N + 4, N + 2, N + 1];
    let input = |len: usize| -> Arc<Tensor<f32>> {
        api::ones::<f32>(&[len])
            .sync_on(&stream)
            .expect("input")
            .into()
    };
    let many: Vec<(Arc<Tensor<f32>>, Arc<Tensor<f32>>)> = extents
        .iter()
        .flat_map(|&la| extents[..3].iter().map(move |&lb| (la, lb)))
        .take(12)
        .map(|(la, lb)| (input(la), input(lb)))
        .collect();
    for (x, y) in &many {
        kernels::add3((&mut zt).partition([N]), &**x, &**y, &*c3)
            .sync_on(&stream)
            .expect("warm specialization");
    }
    group.bench_function("launch_12_specializations", |bench| {
        bench.iter_custom(|iters| {
            enqueue_time(iters, &stream, || {
                for (x, y) in &many {
                    unsafe {
                        kernels::add3((&mut zt).partition([N]), &**x, &**y, &*c3).async_on(&stream)
                    }
                    .expect("launch");
                }
            })
        });
    });

    let many_cache = PlanCache::<kernels::add3::Kernel>::new();
    group.bench_function("plan_cache_12_specializations", |bench| {
        bench.iter_custom(|iters| {
            enqueue_time(iters, &stream, || {
                for (x, y) in &many {
                    let op = many_cache
                        .launch(
                            &options,
                            ((&mut zt).partition([N]), &**x, &**y, &*c3),
                            &stream,
                            |(z, x, y, w)| kernels::add3(z, x, y, w),
                        )
                        .expect("plan cache");
                    unsafe { op.async_on(&stream) }.expect("planned launch");
                }
            })
        });
    });

    // The lookups alone, without launching: the cache's own cost.
    for (x, y) in &many {
        many_cache
            .launch(
                &options,
                ((&mut zt).partition([N]), &**x, &**y, &*c3),
                &stream,
                |(z, x, y, w)| kernels::add3(z, x, y, w),
            )
            .expect("plan cache")
            .sync_on(&stream)
            .expect("fill plan cache");
    }
    group.bench_function("plan_cache_get_12_specializations", |bench| {
        bench.iter(|| {
            for (x, y) in &many {
                let args = ((&mut zt).partition([N]), &**x, &**y, &*c3);
                std::hint::black_box(many_cache.get(&options, &args, &stream))
                    .expect("cached plan");
            }
        })
    });

    // A graph of 32 launches over 8 buffers, replayed.
    let mut bufs: Vec<Tensor<f32>> = (0..8)
        .map(|_| api::zeros::<f32>(&[N]).sync_on(&stream).expect("buf"))
        .collect();
    let graph = CudaGraph::scope(&stream, |s| {
        for i in 0..32usize {
            let (out, rest) = bufs.split_at_mut(1);
            let out = &mut out[0];
            let src = &rest[i % 7];
            s.record(kernels::add3((&mut *out).partition([N]), src, &a, &b))?;
        }
        Ok(())
    })
    .expect("capture");
    for _ in 0..50 {
        graph.launch().sync_on(&stream).expect("graph warmup");
    }

    group.bench_function("graph_replay", |bench| {
        bench.iter_custom(|iters| {
            let mut elapsed = Duration::ZERO;
            let mut done = 0;
            while done < iters {
                let batch = 8u64.min(iters - done);
                let start = Instant::now();
                for _ in 0..batch {
                    unsafe { graph.launch().async_on(&stream) }.expect("replay");
                }
                elapsed += start.elapsed();
                unsafe { stream.synchronize() }.expect("drain");
                done += batch;
            }
            elapsed
        });
    });

    group.finish();
}

/// Host time to enqueue `iters` calls of `enqueue`, draining the stream every
/// [`BATCH`] calls outside the timer.
fn enqueue_time(iters: u64, stream: &Stream, mut enqueue: impl FnMut()) -> Duration {
    let mut elapsed = Duration::ZERO;
    let mut done = 0;
    while done < iters {
        let batch = BATCH.min(iters - done);
        let start = Instant::now();
        for _ in 0..batch {
            enqueue();
        }
        elapsed += start.elapsed();
        unsafe { stream.synchronize() }.expect("drain");
        done += batch;
    }
    elapsed
}

fn bench_config() -> Criterion {
    if cfg!(feature = "smoke-test") {
        Criterion::default()
            .without_plots()
            .save_baseline("smoke-discard".to_string())
    } else {
        Criterion::default()
    }
}
criterion_group!(name = benches; config = bench_config(); targets = host_overhead);
criterion_main!(benches);
