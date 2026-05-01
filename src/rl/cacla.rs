use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::td::TD;

/// Continuous Actor-Critic Learning Automaton (CACLA).
///
/// CACLA is an online actor-critic algorithm for continuous action spaces. Its key
/// distinction from standard AC is a conditional actor update: the actor is only updated
/// when the TD error is *positive* (i.e. the observed return exceeded the critic's
/// prediction). This avoids reinforcing actions that led to below-average outcomes.
///
/// The actor loss is the mean-squared error between the actor's output and the target
/// action, gated by the sign of the TD error:
///
/// ```text
/// if δ > 0:  update actor toward a_target
/// if δ ≤ 0:  skip actor update
/// ```
pub struct CACLA {
    /// Actor network mapping states to continuous actions.
    pub actor: NeuralNetwork,
    /// Optimizer used to update the actor.
    pub actor_optimizer: Box<dyn Optimizer>,
    /// TD(0) value critic V(s).
    pub critic: TD,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// Reusable buffer for the actor loss tensor.
    pub actor_loss: Tensor,
}

impl CACLA {
    /// Creates a new CACLA agent.
    ///
    /// - `actor` / `actor_optimizer`: continuous-action actor network and its update rule.
    /// - `critic_network` / `critic_optimizer`: value network (single scalar output) and its update rule.
    /// - `gamma`: discount factor.
    pub fn new(
        actor: NeuralNetwork,
        actor_optimizer: Box<dyn Optimizer>,
        critic_network: NeuralNetwork,
        critic_optimizer: Box<dyn Optimizer>,
        gamma: f32,
    ) -> Self {
        CACLA {
            actor,
            actor_optimizer,
            critic: TD::new(critic_network, critic_optimizer, gamma),
            gamma,
            actor_loss: Tensor::new(),
        }
    }

    /// Returns the actor's continuous action for `state`.
    pub fn get_action(&mut self, state: &Tensor) -> &Tensor {
        self.actor.forward(state)
    }

    /// Performs one CACLA update step.
    ///
    /// The critic is updated unconditionally. The actor is updated only if the TD error
    /// `δ > 0`, nudging the actor output toward the executed `action`.
    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) {
        self.critic.train(state, next_state, reward, final_state);
        if self.critic.delta().get(vec![0, 0]) > 0.0 {
            let mut loss = self.actor_loss_function(state, action);
            self.actor.backward(&mut loss);
            self.actor_optimizer.update(&mut self.actor);
        }
    }

    fn actor_loss_function(&mut self, state: &Tensor, action: &Tensor) -> Tensor {
        let output = self.actor.forward(state).clone();
        let mut diff = output.clone();
        for i in 0..output.size {
            diff.data[i] -= action.data[i];
        }
        self.actor_loss = diff;
        self.actor_loss.clone()
    }
}
