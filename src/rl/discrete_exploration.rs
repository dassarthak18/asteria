use crate::clab::tensor::Tensor;
use crate::clab::random_generator::RandomGenerator;
use crate::clab::Interpolation;

/// Strategy used by [`DiscreteExploration`] to select an action.
pub enum DiscreteExplorationMethod {
    /// ε-greedy: with probability ε choose a uniformly random action; otherwise choose the greedy action.
    Egreedy,
    /// Boltzmann (softmax) sampling: sample an action proportional to `exp(Q(s,a) / τ)`,
    /// where `τ` is the exploration parameter (temperature).
    Boltzman,
}

/// Exploration wrapper for discrete action spaces.
///
/// Wraps an underlying Q-value (or policy) tensor and converts it to an action using one
/// of two exploration strategies. The exploration parameter (ε for ε-greedy, τ for Boltzmann)
/// can optionally decay over time via an [`Interpolation`] schedule.
///
/// Call [`explore`](DiscreteExploration::explore) at each step to obtain a one-hot action
/// tensor, and [`update`](DiscreteExploration::update) at the end of each episode/step to
/// advance the decay schedule.
pub struct DiscreteExploration {
    method: DiscreteExplorationMethod,
    param: f32,
    interpolation: Option<Box<dyn Interpolation>>,
}

impl DiscreteExploration {
    /// Creates a new discrete exploration policy.
    ///
    /// - `method`: ε-greedy or Boltzmann sampling.
    /// - `exploration_parameter`: initial ε (ε-greedy) or τ (Boltzmann).
    /// - `interpolation`: optional schedule that decays the parameter over time;
    ///   pass `None` to keep it constant.
    pub fn new(method: DiscreteExplorationMethod, exploration_parameter: f32, interpolation: Option<Box<dyn Interpolation>>) -> Self {
        DiscreteExploration {
            method,
            param: exploration_parameter,
            interpolation,
        }
    }

    /// Applies the exploration strategy to `values` and returns a one-hot action tensor.
    ///
    /// For ε-greedy, `values` are interpreted as Q-values and the greedy action is `argmax`.
    /// For Boltzmann, `values` are treated as logits and sampled proportional to `exp(v / τ)`.
    pub fn explore(&self, values: &Tensor) -> Tensor {
        let mut output = Tensor::value(vec![values.size()], 0.0);
        match self.method {
            DiscreteExplorationMethod::Egreedy => self.explore_egreedy(&mut output, values),
            DiscreteExplorationMethod::Boltzman => self.explore_boltzman(&mut output, values),
        }
        output
    }

    /// Advances the exploration parameter decay by one step.
    ///
    /// Has no effect if no interpolation schedule was provided at construction.
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
        let max_val = (0..values.size())
            .map(|i| values.get(vec![i]))
            .fold(f32::NEG_INFINITY, f32::max);
        let mut evals = Tensor::value(vec![values.size()], 0.0);
        let mut sum = 0.0;

        for i in 0..values.size() {
            let val = ((values.get(vec![i]) - max_val) / self.param).exp();
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
