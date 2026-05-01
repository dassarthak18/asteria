use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

/// On-policy SARSA (State-Action-Reward-State-Action) agent.
///
/// SARSA is an on-policy TD control algorithm. Unlike Q-learning, which bootstraps
/// from the *greedy* next action, SARSA bootstraps from the action the policy
/// *actually takes* next:
///
/// ```text
/// Q(s, a) ← Q(s, a) + α · (r + γ · Q(s', a') − Q(s, a))
/// ```
///
/// The agent must call [`get_action`](SARSA::get_action) for both the current and next
/// state and pass both actions to [`train`](SARSA::train).
pub struct SARSA {
    /// Q-network mapping observations to per-action Q-values.
    pub critic: NeuralNetwork,
    /// Optimizer used to update the Q-network parameters.
    pub critic_optimizer: Box<dyn Optimizer>,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// Most recently selected greedy action (one-hot, shape `[n_actions]`).
    pub action: Tensor,
    /// Full Q-value loss tensor from the last update.
    pub critic_loss: Tensor,
}

impl SARSA {
    /// Creates a new SARSA agent.
    ///
    /// - `critic`: Q-network with one output per action.
    /// - `critic_optimizer`: parameter update rule.
    /// - `gamma`: discount factor.
    pub fn new(critic: NeuralNetwork, critic_optimizer: Box<dyn Optimizer>, gamma: f32) -> Self {
        SARSA {
            critic,
            critic_optimizer,
            gamma,
            action: Tensor::new(),
            critic_loss: Tensor::new(),
        }
    }

    /// Returns the greedy action for `state` as a one-hot tensor (shape `[n_actions]`).
    pub fn get_action(&mut self, state: &Tensor) -> &Tensor {
        let q_values = self.critic.forward(state);
        self.action = Tensor::with_shape_val(vec![q_values.size()], 0.0);
        let max_idx = q_values.max_index(0)[0];
        self.action.set(vec![max_idx], 1.0);
        &self.action
    }

    /// Performs one SARSA update using a full `(s, a, s', a', r, done)` tuple.
    ///
    /// Both `action` and `next_action` must be one-hot tensors obtained by calling
    /// [`get_action`](SARSA::get_action) (optionally wrapped by an exploration policy).
    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, next_action: &Tensor, reward: f32, final_state: bool) {
        let mut loss = self.critic_loss_function(state, action, next_state, next_action, reward, final_state);
        self.critic.backward(&mut loss);
        self.critic_optimizer.update(&mut self.critic);
    }

    fn critic_loss_function(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, next_action: &Tensor, reward: f32, final_state: bool) -> Tensor {
        let q_next_values = self.critic.forward(next_state).clone();
        let a1_index = next_action.max_index(0)[0];
        let qs1a1 = q_next_values.get(vec![0, a1_index]);

        let q_values = self.critic.forward(state).clone();
        let a0_index = action.max_index(0)[0];
        let qs0a0 = q_values.get(vec![0, a0_index]);

        self.critic_loss = Tensor::with_shape_val(q_values.shape.clone(), 0.0);
        if final_state {
            self.critic_loss.set(vec![0, a0_index], qs0a0 - reward);
        } else {
            self.critic_loss.set(vec![0, a0_index], qs0a0 - (reward + self.gamma * qs1a1));
        }

        self.critic_loss.clone()
    }
}
