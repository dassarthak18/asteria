use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

pub struct Qlearning {
    pub network: NeuralNetwork,
    pub optimizer: Box<dyn Optimizer>,
    pub gamma: f32,
    pub action: Tensor,
    pub delta: Tensor,
    pub loss: Tensor,
}

impl Qlearning {
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

    pub fn get_action(&mut self, state: &Tensor) -> &Tensor {
        let q_values = self.network.forward(state);
        self.action = Tensor::value(vec![q_values.size()], 0.0);
        let max_idx = q_values.max_index(0)[0];
        self.action.set(vec![max_idx], 1.0);
        &self.action
    }

    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) {
        let mut loss = self.loss_function(state, action, next_state, reward, final_state);
        self.network.backward(&mut loss);
        self.optimizer.update(&mut self.network);
    }

    pub fn delta(&self) -> &Tensor {
        &self.delta
    }

    fn loss_function(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) -> Tensor {
        let q_next_values = self.network.forward(next_state).clone();
        let max_q_value = q_next_values.get(vec![0, q_next_values.max_index(0)[0]]);

        let q_values = self.network.forward(state).clone();
        self.loss = Tensor::value(q_values.shape.clone(), 0.0);

        let index = action.max_index(0)[0];

        self.delta.set(vec![0, 0], q_values.get(vec![0, index]));

        if final_state {
            self.loss.set(vec![0, index], q_values.get(vec![0, index]) - reward);
        } else {
            self.loss.set(vec![0, index], q_values.get(vec![0, index]) - (reward + self.gamma * max_q_value));
        }

        self.loss.clone()
    }
}
