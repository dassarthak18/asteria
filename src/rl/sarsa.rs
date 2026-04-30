use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

pub struct SARSA {
    pub critic: NeuralNetwork,
    pub critic_optimizer: Box<dyn Optimizer>,
    pub gamma: f32,
    pub action: Tensor,
    pub critic_loss: Tensor,
}

impl SARSA {
    pub fn new(critic: NeuralNetwork, critic_optimizer: Box<dyn Optimizer>, gamma: f32) -> Self {
        SARSA {
            critic,
            critic_optimizer,
            gamma,
            action: Tensor::new(),
            critic_loss: Tensor::new(),
        }
    }

    pub fn get_action(&mut self, state: &Tensor) -> &Tensor {
        let q_values = self.critic.forward(state);
        self.action = Tensor::with_shape_val(vec![q_values.size()], 0.0);
        let max_idx = q_values.max_index(0)[0];
        self.action.set(vec![max_idx], 1.0);
        &self.action
    }

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
