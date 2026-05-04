use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

/// Tabular Q-learning implemented with a neural network function approximator.
///
/// `Qlearning` trains a network Q(s, a) using the one-step Bellman update:
///
/// ```text
/// Q(s, a) ← Q(s, a) + α · (r + γ · max_{a'} Q(s', a') − Q(s, a))
/// ```
///
/// At each step, only the gradient for the selected action's Q-value is non-zero;
/// all other action outputs are left unchanged (sparse gradient).
pub struct Qlearning {
    /// Q-network mapping observations to per-action Q-values.
    pub network: NeuralNetwork,
    /// Optimizer used to update the Q-network parameters.
    pub optimizer: Box<dyn Optimizer>,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// Most recently selected action (one-hot, shape `[n_actions]`).
    pub action: Tensor,
    /// TD error for the selected action (shape `[1, 1]`); used by QAC as an advantage signal.
    pub delta: Tensor,
    /// Full Q-value loss tensor from the last update (shape `[1, n_actions]`).
    pub loss: Tensor,
}

impl Qlearning {
    /// Creates a new Q-learning agent.
    ///
    /// - `network`: Q-network with one output per action.
    /// - `optimizer`: parameter update rule.
    /// - `gamma`: discount factor.
    pub fn new(network: NeuralNetwork, optimizer: Box<dyn Optimizer>, gamma: f32) -> Self {
        Qlearning {
            network,
            optimizer,
            gamma,
            action: Tensor::empty(),
            delta: Tensor::value(vec![1, 1], 0.0),
            loss: Tensor::empty(),
        }
    }

    /// Returns the greedy action for `state` as a one-hot tensor (shape `[n_actions]`).
    pub fn get_action(&mut self, state: &Tensor) -> &Tensor {
        let q_values = self.network.forward(state);
        self.action = Tensor::value(vec![q_values.size()], 0.0);
        let max_idx = q_values.max_index(0)[0];
        self.action.set(vec![max_idx], 1.0);
        &self.action
    }

    /// Performs one Q-learning update using the given transition.
    ///
    /// Computes the Bellman target, backpropagates the loss for the selected action,
    /// and applies a gradient step.
    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) {
        let mut loss = self.loss_function(state, action, next_state, reward, final_state);
        self.network.backward(&mut loss);
        self.optimizer.update(&mut self.network);
    }

    /// Returns the TD error `Q(s,a) − target` from the most recent update (shape `[1, 1]`).
    ///
    /// Follows the same sign convention as [`TD::delta`](crate::rl::td::TD::delta):
    /// positive when Q overestimates the target (action worse than expected); negative when
    /// Q underestimates (action better than expected). [`QAC`](crate::rl::qac::QAC) uses
    /// this value as the advantage signal for its actor.
    pub fn delta(&self) -> &Tensor {
        &self.delta
    }

    fn loss_function(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) -> Tensor {
        let q_next_values = self.network.forward(next_state).clone();
        let max_q_value = q_next_values.get(vec![0, q_next_values.max_index(0)[0]]);

        let q_values = self.network.forward(state).clone();
        self.loss = Tensor::value(q_values.shape.clone(), 0.0);

        let index = action.max_index(0)[0];
        let q_sa = q_values.get(vec![0, index]);
        let td_error = if final_state {
            q_sa - reward
        } else {
            q_sa - (reward + self.gamma * max_q_value)
        };
        self.loss.set(vec![0, index], td_error);
        // Actor advantage uses same sign convention as TD: positive loss = overestimate = reduce prob.
        self.delta.set(vec![0, 0], td_error);

        self.loss.clone()
    }
}
