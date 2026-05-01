use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::policy_gradient::PolicyGradient;
use crate::rl::qlearning::Qlearning;

/// Q-value actor-critic (QAC).
///
/// `QAC` replaces the TD(0) critic from the standard [`AC`](crate::rl::ac::AC) agent with
/// a Q-learning critic Q(s, a). The critic's temporal-difference error for the selected
/// action serves as the advantage signal for the [`PolicyGradient`] actor.
///
/// Unlike AC, which computes state-value advantages `V(s') − V(s) + r`, QAC uses the
/// action-value estimate `Q(s, a)` directly. This allows the policy to be conditioned on
/// action values rather than state values alone.
pub struct QAC {
    /// Policy gradient actor π(a|s).
    pub actor: PolicyGradient,
    /// Q-learning critic Q(s, a).
    pub critic: Qlearning,
}

impl QAC {
    /// Creates a new QAC agent.
    ///
    /// - `actor_network` / `actor_optimizer`: policy network and its update rule.
    /// - `critic_network` / `critic_optimizer`: Q-network (outputs one Q-value per action) and its update rule.
    /// - `gamma`: discount factor passed to the Q-learning critic.
    pub fn new(
        actor_network: NeuralNetwork,
        actor_optimizer: Box<dyn Optimizer>,
        critic_network: NeuralNetwork,
        critic_optimizer: Box<dyn Optimizer>,
        gamma: f32,
    ) -> Self {
        QAC {
            actor: PolicyGradient::new(actor_network, actor_optimizer),
            critic: Qlearning::new(critic_network, critic_optimizer, gamma),
        }
    }

    /// Returns the actor's action probability distribution for the given `state`.
    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        self.actor.get_action(state)
    }

    /// Performs one QAC update step.
    ///
    /// First trains the Q-critic on `(state, action, next_state, reward, final_state)`,
    /// then updates the actor using the Q-value delta as the advantage signal.
    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) {
        self.critic.train(state, action, next_state, reward, final_state);
        self.actor.train(state, action, self.critic.delta());
    }
}
