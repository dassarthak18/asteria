use crate::clab::tensor::{Tensor, Init};
use crate::clab::loss_functions::{MseFunction, LossFunction};
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::forward_model::ForwardModel;

/// Metacritic intrinsic motivation module.
///
/// `Metacritic` extends [`ForwardModel`]-based curiosity by adding a *metacritic* network
/// that learns to *predict the forward model's prediction error*. The intrinsic reward is
/// high when the actual error deviates from the metacritic's estimate by more than `sigma`
/// (i.e. when the agent visits a truly novel region, not just a region the forward model
/// happens to be inaccurate on).
///
/// The reward is:
///
/// ```text
/// if |error − estimate| > σ:
///     reward = max(0, tanh((error/estimate + estimate/error − 2)))
/// else:
///     reward = 0
/// ```
///
/// This reward is then component-wise max'd with the forward model's base `tanh(error)` reward.
pub struct Metacritic {
    /// Metacritic network m(s, a) → error_estimate; input is the concatenation of state and action.
    pub network: NeuralNetwork,
    /// Optimizer used to update the metacritic network.
    pub optimizer: Box<dyn Optimizer>,
    /// Underlying forward model for next-state prediction.
    pub forward_model: ForwardModel,
    /// Threshold σ: only rewards when `|error − estimate| > σ`.
    pub sigma: f32,
    /// Reusable input buffer for the concatenated `[state, action]` tensor.
    pub input: Tensor,
    /// Prediction error from the forward model.
    pub error: Tensor,
    /// Intrinsic reward tensor.
    pub reward: Tensor,
    /// MSE loss function used for training the metacritic.
    pub loss_function: MseFunction,
}

impl Metacritic {
    /// Creates a new metacritic agent.
    ///
    /// - `network`: metacritic network with `state_dim + action_dim` inputs and 1 output.
    /// - `optimizer`: update rule for the metacritic.
    /// - `forward_model`: a pre-constructed [`ForwardModel`].
    /// - `sigma`: novelty threshold; rewards only transitions where `|error − estimate| > σ`.
    pub fn new(
        network: NeuralNetwork,
        optimizer: Box<dyn Optimizer>,
        forward_model: ForwardModel,
        sigma: f32,
    ) -> Self {
        Metacritic {
            network,
            optimizer,
            forward_model,
            sigma,
            input: Tensor::new(),
            error: Tensor::new(),
            reward: Tensor::new(),
            loss_function: MseFunction,
        }
    }

    /// Trains both the forward model and the metacritic on one `(state, action, next_state)` transition.
    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor) {
        let input = Tensor::concat(&[state, action], 1);
        self.forward_model.train(state, action, next_state);

        let target = self.forward_model.error(state, action, next_state).clone();
        let prediction = self.network.forward(&input).clone();

        let mut delta = self.loss_function.backward(&prediction, &target);
        self.network.backward(&mut delta);
        self.optimizer.update(&mut self.network);
    }

    /// Returns the metacritic intrinsic reward for a `(state, action, next_state)` triple.
    ///
    /// Reward is non-zero only when the forward model's actual error exceeds the metacritic's
    /// prediction by more than `sigma`.
    pub fn reward(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor) -> &Tensor {
        let input = Tensor::concat(&[state, action], 1);
        let error = self.forward_model.error(state, action, next_state).clone();
        let error_estimate = self.network.forward(&input).clone();
        let pe_reward = self.forward_model.reward(state, action, next_state).clone();

        self.reward.resize(vec![error.size], Init::Zero);

        for i in 0..error.size {
            if (error.data[i] - error_estimate.data[i]).abs() > self.sigma {
                let val = (error.data[i] / error_estimate.data[i] + error_estimate.data[i] / error.data[i] - 2.0).tanh();
                self.reward.data[i] = val.max(0.0);
            } else {
                self.reward.data[i] = 0.0;
            }
            self.reward.data[i] = self.reward.data[i].max(pe_reward.data[i]);
        }
        &self.reward
    }

    /// Returns the metacritic's predicted forward-model error for `(state, action)`.
    pub fn error(&mut self, state: &Tensor, action: &Tensor) -> &Tensor {
        let input = Tensor::concat(&[state, action], 1);
        self.error = self.network.forward(&input).clone();
        &self.error
    }
}
