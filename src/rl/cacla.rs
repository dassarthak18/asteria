use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::td::TD;

pub struct CACLA {
    pub actor: NeuralNetwork,
    pub actor_optimizer: Box<dyn Optimizer>,
    pub critic: TD,
    pub gamma: f32,
    pub actor_loss: Tensor,
}

impl CACLA {
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

    pub fn get_action(&mut self, state: &Tensor) -> &Tensor {
        self.actor.forward(state)
    }

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
