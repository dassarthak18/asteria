use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

/// Temporal-difference value estimator (one-step TD(0)).
///
/// `TD` maintains a critic network V(s) and trains it by minimising the one-step
/// Bellman residual: `δ = V(s) − (r + γ · V(s'))`. The TD error `δ` is exposed via
/// [`delta`](TD::delta) so that actor-critic algorithms can use it as an advantage signal
/// for the actor update.
pub struct TD {
    /// Critic network V(s) that maps observations to scalar value estimates.
    pub network: NeuralNetwork,
    /// Optimizer used to update the critic's parameters.
    pub optimizer: Box<dyn Optimizer>,
    /// Discount factor γ ∈ [0, 1]; higher values assign more weight to future rewards.
    pub gamma: f32,
    /// Most recently computed TD error tensor; shape `[1, 1]`.
    pub loss: Tensor,
}

impl TD {
    /// Creates a new TD critic.
    ///
    /// - `network`: a neural network with a single scalar output.
    /// - `optimizer`: parameter update rule (e.g. ADOPT, Adam).
    /// - `gamma`: discount factor.
    pub fn new(network: NeuralNetwork, optimizer: Box<dyn Optimizer>, gamma: f32) -> Self {
        TD {
            network,
            optimizer,
            gamma,
            loss: Tensor::value(vec![1, 1], 0.0),
        }
    }

    /// Performs one TD(0) update step.
    ///
    /// Computes `δ = V(s) − (r + γ·V(s'))`, backpropagates it into the critic, and
    /// applies a gradient step. The TD error is saved and accessible via [`delta`](TD::delta).
    pub fn train(&mut self, state: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) {
        let mut loss = self.loss_function(state, next_state, reward, final_state);
        self.network.backward(&mut loss);
        self.optimizer.update(&mut self.network);
    }

    /// Returns the TD error `δ` from the most recent [`train`](TD::train) call (shape `[1, 1]`).
    ///
    /// Actor-critic agents use this as a scalar advantage signal to scale the policy gradient.
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
