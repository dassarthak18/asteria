use crate::clab::tensor::{Tensor, Init};
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

pub struct PolicyGradient {
    pub network: NeuralNetwork,
    pub optimizer: Box<dyn Optimizer>,
    pub loss: Tensor,
}

impl PolicyGradient {
    pub fn new(network: NeuralNetwork, optimizer: Box<dyn Optimizer>) -> Self {
        PolicyGradient {
            network,
            optimizer,
            loss: Tensor::empty(),
        }
    }

    pub fn train(&mut self, state: &Tensor, action: &Tensor, delta: &Tensor) {
        let mut loss = self.loss_function(state, action, delta);
        self.network.backward(&mut loss);
        self.optimizer.update(&mut self.network);
    }

    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        self.network.forward(state).clone()
    }

    fn loss_function(&mut self, state: &Tensor, action: &Tensor, delta: &Tensor) -> Tensor {
        let output = self.network.forward(state).clone();
        let action_index = action.max_index(0)[0];

        self.loss.resize(vec![1, output.size()], Init::Zero);
        let val = delta.get(vec![0, 0]) / output.get(vec![0, action_index]);
        self.loss.set(vec![0, action_index], val);

        self.loss.clone()
    }
}
