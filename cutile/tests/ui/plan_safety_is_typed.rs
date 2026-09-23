// A plan that needs an `unsafe` launch is a `LaunchPlan<_, Unchecked>`: it
// has no safe `launch`, and a `PlanCache` holds only `Safe` plans of its own
// kernel. An `unsafe` entry point, or a launcher opted into programmatic
// dependent launch, cannot build a `Safe` plan at all.
use cutile::plan::{LaunchPlan, PlanCache, Unchecked};
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

fn unchecked_plans_have_no_safe_launch(
    plan: LaunchPlan<kernels::raw::Kernel, Unchecked>,
    z: &mut Tensor<f32>,
    x: &Tensor<f32>,
) {
    let _ = plan.launch((z.partition([4]), x));
}

fn unsafe_entry_points_have_no_safe_plan(
    stream: &std::sync::Arc<cuda_core::Stream>,
    z: &mut Tensor<f32>,
    x: &Tensor<f32>,
) {
    let _ = unsafe { kernels::raw(z.partition([4]), x) }.plan_on(stream);
}

fn caches_take_only_safe_plans(
    plan: LaunchPlan<kernels::double::Kernel, Unchecked>,
) {
    PlanCache::<kernels::double::Kernel>::new().insert(plan);
}

fn caches_take_only_their_kernel(plan: LaunchPlan<kernels::triple::Kernel>) {
    PlanCache::<kernels::double::Kernel>::new().insert(plan);
}

fn opted_in_launchers_have_no_safe_plan(
    stream: &std::sync::Arc<cuda_core::Stream>,
    z: &mut Tensor<f32>,
    x: &Tensor<f32>,
) {
    let _ = unsafe { kernels::double(z.partition([4]), x).programmatic_dependent_launch() }
        .plan_on(stream);
}

fn opted_in_launchers_fill_no_cache(
    stream: &std::sync::Arc<cuda_core::Stream>,
    z: &mut Tensor<f32>,
    x: &Tensor<f32>,
) {
    let cache = PlanCache::<kernels::double::Kernel>::new();
    let _ = cache.launch(
        &cutile::plan::PlanOptions::new(),
        (z.partition([4]), x),
        stream,
        |(z, x)| unsafe { kernels::double(z, x).programmatic_dependent_launch() },
    );
}

fn main() {}
