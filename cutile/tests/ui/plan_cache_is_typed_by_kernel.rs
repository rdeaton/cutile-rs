// A PlanCache serves one kernel: on a hit the build closure never runs, so a
// cache shared by two kernels could answer one with the other's plan. The
// closure must produce the cache's kernel, and only safe entry points can
// build cached plans at all.
use cutile::plan::{PlanCache, PlanOptions};
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn double<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let tx = x.load_like(z);
        z.store(tx + tx);
    }

    #[cutile::entry()]
    fn triple<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let tx = x.load_like(z);
        z.store(tx + tx + tx);
    }

    #[cutile::entry()]
    unsafe fn raw<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let tx = x.load_like(z);
        z.store(tx);
    }
}

fn wrong_kernel(
    stream: &std::sync::Arc<cuda_core::Stream>,
    z: &mut Tensor<f32>,
    x: &Tensor<f32>,
) {
    let cache = PlanCache::<kernels::double::Kernel>::new();
    let _ = cache.launch(&PlanOptions::new(), (z.partition([4]), x), stream, |(z, x)| {
        kernels::triple(z, x)
    });
}

fn unsafe_entry(
    stream: &std::sync::Arc<cuda_core::Stream>,
    z: &mut Tensor<f32>,
    x: &Tensor<f32>,
) {
    let cache = PlanCache::<kernels::raw::Kernel>::new();
    let _ = cache.launch(&PlanOptions::new(), (z.partition([4]), x), stream, |(z, x)| unsafe {
        kernels::raw(z, x)
    });
}

fn main() {}
