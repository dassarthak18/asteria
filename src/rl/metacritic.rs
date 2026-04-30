use crate::clab::tensor::{Tensor, Init};
use crate::clab::loss_functions::{MseFunction, LossFunction};
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::forward_model::ForwardModel;

pub struct Metacritic {
    pub network: NeuralNetwork,
    pub optimizer: Box<dyn Optimizer>,
    pub forward_model: ForwardModel,
    pub sigma: f32,
    pub input: Tensor,
    pub error: Tensor,
    pub reward: Tensor,
    pub loss_function: MseFunction,
}

impl Metacritic {
    pub fn new(
        network: NeuralNetwork,
        optimizer: Box<dyn Optimizer>,
        forward_model: ForwardModel,
        sigma: f32,
    ) -> Self {
        Metacritic {
            network,
            optimizer,
            forward_model,
            sigma,
            input: Tensor::new(),
            error: Tensor::new(),
            reward: Tensor::new(),
            loss_function: MseFunction,
        }
    }

    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor) {
        let input = Tensor::concat(&[state, action], 1);
        self.forward_model.train(state, action, next_state);

        let target = self.forward_model.error(state, action, next_state).clone();
        let prediction = self.network.forward(&input).clone();

        let mut delta = self.loss_function.backward(&prediction, &target);
        self.network.backward(&mut delta);
        self.optimizer.update(&mut self.network);
    }

    pub fn reward(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor) -> &Tensor {
        let input = Tensor::concat(&[state, action], 1);
        let error = self.forward_model.error(state, action, next_state).clone();
        let error_estimate = self.network.forward(&input).clone();
        let pe_reward = self.forward_model.reward(state, action, next_state).clone();

        self.reward.resize(vec![error.size], Init::Zero);

        for i in 0..error.size {
            if (error.data[i] - error_estimate.data[i]).abs() > self.sigma {
                let val = (error.data[i] / error_estimate.data[i] + error_estimate.data[i] / error.data[i] - 2.0).tanh();
                self.reward.data[i] = val.max(0.0);
            } else {
                self.reward.data[i] = 0.0;
            }
            self.reward.data[i] = self.reward.data[i].max(pe_reward.data[i]);
        }
        &self.reward
    }

    pub fn error(&mut self, state: &Tensor, action: &Tensor) -> &Tensor {
        let input = Tensor::concat(&[state, action], 1);
        self.error = self.network.forward(&input).clone();
        &self.error
    }
}
