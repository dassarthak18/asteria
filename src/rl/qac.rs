use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::policy_gradient::PolicyGradient;
use crate::rl::qlearning::Qlearning;

pub struct QAC {
    pub actor: PolicyGradient,
    pub critic: Qlearning,
}

impl QAC {
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

    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        self.actor.get_action(state)
    }

    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) {
        self.critic.train(state, action, next_state, reward, final_state);
        self.actor.train(state, action, self.critic.delta());
    }
}
