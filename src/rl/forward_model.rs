use crate::clab::tensor::{Tensor, Init};
use crate::clab::loss_functions::{MseFunction, LossFunction};
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

/// World model that predicts the next state given a `(state, action)` pair.
///
/// `ForwardModel` is trained to minimise the MSE between its predicted next state and the
/// actual observed next state. It produces intrinsic curiosity rewards proportional to its
/// own prediction error: novel (hard-to-predict) transitions yield higher intrinsic reward.
///
/// The intrinsic reward is computed as `tanh(MSE)`, which bounds the signal to `[0, 1)`.
/// Combine with extrinsic reward to encourage exploration: see [`Metacritic`](crate::rl::metacritic::Metacritic)
/// for a variant that also predicts its own error.
pub struct ForwardModel {
    /// Network fˆ(s, a) → sˆ'; input is the concatenation of state and action.
    pub network: NeuralNetwork,
    /// Optimizer used to update the forward model's parameters.
    pub optimizer: Box<dyn Optimizer>,
    /// Reusable input buffer for the concatenated `[state, action]` tensor.
    pub input: Tensor,
    /// Per-row MSE between predicted and actual next state (shape `[batch, 1]`).
    pub error: Tensor,
    /// Intrinsic reward tensor derived from `error` via `tanh` (same shape as `error`).
    pub reward: Tensor,
    /// MSE loss function used for both training and error computation.
    pub loss_function: MseFunction,
}

impl ForwardModel {
    /// Creates a new forward model.
    ///
    /// - `network`: a network whose input dimension equals `state_dim + action_dim` and
    ///   output dimension equals `state_dim`.
    /// - `optimizer`: parameter update rule.
    pub fn new(network: NeuralNetwork, optimizer: Box<dyn Optimizer>) -> Self {
        ForwardModel {
            network,
            optimizer,
            input: Tensor::new(),
            error: Tensor::new(),
            reward: Tensor::new(),
            loss_function: MseFunction,
        }
    }

    /// Trains the model on one `(state, action, next_state)` transition.
    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor) {
        let input = Tensor::concat(&[state, action], 1);
        let predicted_state = self.network.forward(&input).clone();
        let mut delta = self.loss_function.backward(&predicted_state, next_state);
        self.network.backward(&mut delta);
        self.optimizer.update(&mut self.network);
    }

    /// Returns the intrinsic reward for a `(state, action, next_state)` transition.
    ///
    /// Computed as `tanh(per-dimension-MSE)` for each output dimension, bounding the
    /// signal to `[0, 1)`. High prediction error → high intrinsic reward.
    pub fn reward(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor) -> &Tensor {
        let error = self.error(state, action, next_state).clone();
        self.reward.resize(vec![1, error.size], Init::Zero);
        for i in 0..error.size {
            self.reward.data[i] = error.data[i].tanh();
        }
        &self.reward
    }

    /// Returns the per-row prediction MSE for a `(state, action, next_state)` triple.
    ///
    /// Shape: `[batch, 1]`. Used internally by [`Metacritic`](crate::rl::metacritic::Metacritic)
    /// as a training target.
    pub fn error(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor) -> &Tensor {
        let input = Tensor::concat(&[state, action], 1);
        let predicted_state = self.network.forward(&input).clone();

        self.error.resize(vec![predicted_state.shape[0], 1], Init::Zero);

        let rows = predicted_state.shape[0];
        let cols = predicted_state.shape[1];

        for i in 0..rows {
            let mut sum_sq_err = 0.0;
            for j in 0..cols {
                let index = i * cols + j;
                sum_sq_err += (predicted_state.data[index] - next_state.data[index]).powi(2);
            }
            self.error.data[i] = sum_sq_err / cols as f32;
        }

        &self.error
    }
}
