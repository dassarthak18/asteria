use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::policy_gradient::PolicyGradient;
use crate::rl::td::TD;

/// Online actor-critic agent (AC).
///
/// `AC` combines a [`PolicyGradient`] actor and a [`TD`] critic. On every step the critic
/// produces a one-step TD error `δ`, which the actor uses as an advantage signal to
/// update the policy. Both networks are updated after every transition (no replay buffer).
///
/// Use [`AC2`](crate::rl::ac2::AC2) for the batched A2C variant or [`AC3`](crate::rl::ac3::AC3)
/// for the asynchronous multi-worker variant.
pub struct AC {
    /// Policy gradient actor π(a|s).
    pub actor: PolicyGradient,
    /// TD(0) value critic V(s).
    pub critic: TD,
}

impl AC {
    /// Creates a new online actor-critic agent.
    ///
    /// - `actor_network` / `actor_optimizer`: policy network and its update rule.
    /// - `critic_network` / `critic_optimizer`: value network (single scalar output) and its update rule.
    /// - `gamma`: discount factor passed to the TD critic.
    pub fn new(
        actor_network: NeuralNetwork,
        actor_optimizer: Box<dyn Optimizer>,
        critic_network: NeuralNetwork,
        critic_optimizer: Box<dyn Optimizer>,
        gamma: f32,
    ) -> Self {
        AC {
            actor: PolicyGradient::new(actor_network, actor_optimizer),
            critic: TD::new(critic_network, critic_optimizer, gamma),
        }
    }

    /// Returns the actor's action probability distribution for the given `state`.
    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        self.actor.get_action(state)
    }

    /// Performs one AC update step using the given transition `(state, action, next_state, reward, final_state)`.
    ///
    /// First updates the critic with the TD error, then uses that error as the advantage
    /// signal to update the actor.
    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) {
        self.critic.train(state, next_state, reward, final_state);
        self.actor.train(state, action, self.critic.delta());
    }
}
