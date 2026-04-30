use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;

/// Vanilla stochastic gradient descent: `param -= alpha * gradient`.
///
/// No momentum or adaptive rates. Useful as a baseline or when simplicity matters.
pub struct Sgd {
    /// Learning rate applied to every gradient.
    pub alpha: f32,
}

impl Sgd {
    /// Creates an SGD optimizer with the given learning rate.
    pub fn new(alpha: f32) -> Self {
        Sgd { alpha }
    }
}

impl Optimizer for Sgd {
    fn set_lr(&mut self, lr: f32) { self.alpha = lr; }
    fn get_lr(&self) -> f32 { self.alpha }

    fn update(&mut self, network: &mut NeuralNetwork) {
        for param_arc in network.model.model.values() {
            let mut param = param_arc.lock().unwrap();
            for i in 0..param.params.size {
                param.params.data[i] -= self.alpha * param.gradient.data[i];
            }
        }
    }
}
