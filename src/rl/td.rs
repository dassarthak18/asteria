use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

pub struct TD {
    pub network: NeuralNetwork,
    pub optimizer: Box<dyn Optimizer>,
    pub gamma: f32,
    pub loss: Tensor,
}

impl TD {
    pub fn new(network: NeuralNetwork, optimizer: Box<dyn Optimizer>, gamma: f32) -> Self {
        TD {
            network,
            optimizer,
            gamma,
            loss: Tensor::value(vec![1, 1], 0.0),
        }
    }

    pub fn train(&mut self, state: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) {
        let mut loss = self.loss_function(state, next_state, reward, final_state);
        self.network.backward(&mut loss);
        self.optimizer.update(&mut self.network);
    }

    pub fn delta(&self) -> &Tensor {
        &self.loss
    }

    fn loss_function(&mut self, state: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) -> Tensor {
        let v1 = self.network.forward(next_state).get(vec![0, 0]);
        let v0 = self.network.forward(state).get(vec![0, 0]);

        let mut delta = v0;

        if final_state {
            delta -= reward;
        } else {
            delta -= reward + self.gamma * v1;
        }

        self.loss.set(vec![0, 0], delta);
        self.loss.clone()
    }
}
