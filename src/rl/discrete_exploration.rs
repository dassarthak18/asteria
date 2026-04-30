use crate::clab::tensor::Tensor;
use crate::clab::random_generator::RandomGenerator;
use crate::clab::Interpolation;

pub enum DiscreteExplorationMethod {
    Egreedy,
    Boltzman,
}

pub struct DiscreteExploration {
    method: DiscreteExplorationMethod,
    param: f32,
    interpolation: Option<Box<dyn Interpolation>>,
}

impl DiscreteExploration {
    pub fn new(method: DiscreteExplorationMethod, exploration_parameter: f32, interpolation: Option<Box<dyn Interpolation>>) -> Self {
        DiscreteExploration {
            method,
            param: exploration_parameter,
            interpolation,
        }
    }

    pub fn explore(&self, values: &Tensor) -> Tensor {
        let mut output = Tensor::value(vec![values.size()], 0.0);
        match self.method {
            DiscreteExplorationMethod::Egreedy => self.explore_egreedy(&mut output, values),
            DiscreteExplorationMethod::Boltzman => self.explore_boltzman(&mut output, values),
        }
        output
    }

    pub fn update(&mut self, timestep: i32) {
        if let Some(ref interpolation) = self.interpolation {
            self.param = interpolation.interpolate(timestep);
        }
    }

    fn explore_egreedy(&self, output: &mut Tensor, values: &Tensor) {
        let random = RandomGenerator::instance().random();
        let action = if random < self.param {
            RandomGenerator::instance().random_range(0, (values.size() - 1) as i32) as usize
        } else {
            values.max_index(0)[0]
        };

        output.fill(0.0);
        output.set(vec![action], 1.0);
    }

    fn explore_boltzman(&self, output: &mut Tensor, values: &Tensor) {
        let mut evals = Tensor::value(vec![values.size()], 0.0);
        let mut sum = 0.0;

        for i in 0..values.size() {
            let val = (values.get(vec![i]) / self.param).exp();
            evals.set(vec![i], val);
            sum += val;
        }

        for i in 0..values.size() {
            evals.set(vec![i], evals.get(vec![i]) / sum);
        }

        let action = RandomGenerator::instance().choice(evals.data());

        output.fill(0.0);
        output.set(vec![action], 1.0);
    }
}
