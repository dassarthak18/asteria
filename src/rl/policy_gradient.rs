use crate::clab::tensor::{Tensor, Init};
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

/// REINFORCE-style policy gradient component used inside actor-critic agents.
///
/// `PolicyGradient` maintains a stochastic actor network π(a|s) and updates it using
/// the policy gradient theorem:
///
/// ```text
/// ∇J ∝ −δ · ∇ log π(a|s)
/// ```
///
/// where `δ` is a scalar advantage signal supplied by a critic (TD error, Q-value, etc.).
/// The gradient of `log π` is approximated as `δ / π(a|s)` for the selected action `a`.
pub struct PolicyGradient {
    /// Actor network π(a|s); output should be a probability distribution (Softmax activation).
    pub network: NeuralNetwork,
    /// Optimizer used to update the actor's parameters.
    pub optimizer: Box<dyn Optimizer>,
    /// Reusable buffer for the per-step policy gradient loss tensor.
    pub loss: Tensor,
}

impl PolicyGradient {
    /// Creates a new policy gradient actor.
    ///
    /// - `network`: actor network with Softmax output.
    /// - `optimizer`: parameter update rule.
    pub fn new(network: NeuralNetwork, optimizer: Box<dyn Optimizer>) -> Self {
        PolicyGradient {
            network,
            optimizer,
            loss: Tensor::empty(),
        }
    }

    /// Updates the actor with one policy gradient step.
    ///
    /// - `state`: current observation.
    /// - `action`: one-hot tensor encoding the action taken.
    /// - `delta`: loss-gradient convention, consistent with the critic's own backward signal.
    ///   **Negative** delta increases the probability of `action` (action was better than
    ///   expected); positive delta decreases it. The [`TD`](crate::rl::td::TD) critic returns
    ///   `V(s) − target`, which is negative when the action was good — this convention is
    ///   used throughout the codebase.
    pub fn train(&mut self, state: &Tensor, action: &Tensor, delta: &Tensor) {
        let mut loss = self.loss_function(state, action, delta);
        self.network.backward(&mut loss);
        self.optimizer.update(&mut self.network);
    }

    /// Returns the actor's output probabilities for the given `state`.
    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        self.network.forward(state).clone()
    }

    fn loss_function(&mut self, state: &Tensor, action: &Tensor, delta: &Tensor) -> Tensor {
        let output = self.network.forward(state).clone();
        let action_index = action.max_index(0)[0];

        self.loss.resize(vec![1, output.size()], Init::Zero);
        let val = delta.get(vec![0, 0]) / output.get(vec![0, action_index]);
        self.loss.set(vec![0, action_index], val);

        self.loss.clone()
    }
}
