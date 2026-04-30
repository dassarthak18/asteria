use crate::clab::tensor::{Tensor, Init};
use crate::clab::loss_functions::{MseFunction, LossFunction};
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

pub struct ForwardModel {
    pub network: NeuralNetwork,
    pub optimizer: Box<dyn Optimizer>,
    pub input: Tensor,
    pub error: Tensor,
    pub reward: Tensor,
    pub loss_function: MseFunction,
}

impl ForwardModel {
    pub fn new(network: NeuralNetwork, optimizer: Box<dyn Optimizer>) -> Self {
        ForwardModel {
            network,
            optimizer,
            input: Tensor::new(),
            error: Tensor::new(),
            reward: Tensor::new(),
            loss_function: MseFunction,
        }
    }

    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor) {
        let input = Tensor::concat(&[state, action], 1);
        let predicted_state = self.network.forward(&input).clone();
        let mut delta = self.loss_function.backward(&predicted_state, next_state);
        self.network.backward(&mut delta);
        self.optimizer.update(&mut self.network);
    }

    pub fn reward(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor) -> &Tensor {
        let error = self.error(state, action, next_state).clone();
        self.reward.resize(vec![1, error.size], Init::Zero);
        for i in 0..error.size {
            self.reward.data[i] = error.data[i].tanh();
        }
        &self.reward
    }

    pub fn error(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor) -> &Tensor {
        let input = Tensor::concat(&[state, action], 1);
        let predicted_state = self.network.forward(&input).clone();

        self.error.resize(vec![predicted_state.shape[0], 1], Init::Zero);

        let rows = predicted_state.shape[0];
        let cols = predicted_state.shape[1];

        for i in 0..rows {
            let mut sum_sq_err = 0.0;
            for j in 0..cols {
                let index = i * cols + j;
                sum_sq_err += (predicted_state.data[index] - next_state.data[index]).powi(2);
            }
            self.error.data[i] = sum_sq_err / cols as f32;
        }

        &self.error
    }
}
