// A kernel with a `MappedPartitionMut` parameter cannot be planned, so its
// launcher has no plan methods, and no `PlanArg` fills its `Params`.
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry]
    fn copy<const BN: i32, const MAP_SHAPE: [i32; 1]>(
        mut z: MappedPartitionMut<f32, { [BN] }, MAP_SHAPE>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let part_x = x.partition(shape![BN]);
        for index in z.iter_indices() {
            let coords = index.coords();
            let tile = part_x.load([coords[0]]);
            z.store(tile, index);
        }
    }
}

fn mapped_kernels_have_no_plan(
    stream: &std::sync::Arc<cuda_core::Stream>,
    z: Tensor<f32>,
    x: &Tensor<f32>,
) {
    let _ = kernels::copy(z.partition([4]).map([1], 2), x).plan_on(stream);
}

fn mapped_kernels_have_no_unchecked_plan(
    stream: &std::sync::Arc<cuda_core::Stream>,
    z: Tensor<f32>,
    x: &Tensor<f32>,
) {
    let _ = kernels::copy(z.partition([4]).map([1], 2), x).plan_unchecked_on(stream);
}

fn mapped_kernels_take_no_plan_args(
    plan: cutile::plan::LaunchPlan<kernels::copy::Kernel>,
    z: Tensor<f32>,
    x: &Tensor<f32>,
) {
    let _ = plan.launch((z.partition([4]), x));
}

fn main() {}
