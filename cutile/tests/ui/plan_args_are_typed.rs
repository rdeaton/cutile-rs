// A plan's arguments are checked against its kernel's parameters at compile
// time: their count, and whether each is a tensor input, a partitioned
// output, or a scalar of the declared type.
use cutile::plan::LaunchPlan;
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn scale<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>, a: f32) {
        let tx = x.load_like(z);
        let s: Tile<f32, { [B] }> = a.broadcast(z.shape());
        z.store(tx * s);
    }
}

fn too_few_arguments(plan: LaunchPlan<kernels::scale::Kernel>, z: &mut Tensor<f32>, x: &Tensor<f32>) {
    let _ = plan.launch((z.partition([4]), x));
}

fn an_input_for_an_output(plan: LaunchPlan<kernels::scale::Kernel>, z: &Tensor<f32>, x: &Tensor<f32>) {
    let _ = plan.launch((z, x, 1.0f32));
}

fn a_scalar_of_another_type(plan: LaunchPlan<kernels::scale::Kernel>, z: &mut Tensor<f32>, x: &Tensor<f32>) {
    let _ = plan.launch((z.partition([4]), x, 1.0f64));
}

fn main() {}
