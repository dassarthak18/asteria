use crate::core::neural_network::NeuralNetwork;
use crate::clab::tensor::{Tensor, Init};
use crate::core::optimizer::Optimizer;

/// Adam optimizer with bias-corrected first and second moment estimates.
///
/// Optional `weight_decay` adds L2 regularisation; set to `0.0` to disable.
pub struct Adam {
    /// Learning rate.
    pub alpha: f32,
    /// L2 weight decay coefficient.
    pub weight_decay: f32,
    /// First moment decay (default 0.9).
    pub beta1: f32,
    /// Second moment decay (default 0.999).
    pub beta2: f32,
    /// Numerical stabiliser (default 1e-8).
    pub epsilon: f32,
    /// Global step counter used for bias correction.
    pub t: u32,
    /// Per-parameter first moment estimates.
    pub m: std::collections::HashMap<String, Tensor>,
    /// Per-parameter second moment estimates.
    pub v: std::collections::HashMap<String, Tensor>,
}

impl Adam {
    /// Creates an Adam optimizer with the given learning rate and sensible defaults.
    ///
    /// Defaults: `weight_decay=0.0`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`.
    pub fn new(alpha: f32) -> Self {
        Adam {
            alpha,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: std::collections::HashMap::new(),
            v: std::collections::HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn set_lr(&mut self, lr: f32) { self.alpha = lr; }
    fn get_lr(&self) -> f32 { self.alpha }

    fn update(&mut self, network: &mut NeuralNetwork) {
        self.t += 1;
        let denb1 = 1.0 - self.beta1.powi(self.t as i32);
        let denb2 = 1.0 - self.beta2.powi(self.t as i32);

        for (id, param_arc) in &network.model.model {
            let mut param = param_arc.lock().unwrap();
            
            if !self.m.contains_key(id) {
                self.m.insert(id.clone(), Tensor::with_shape(param.params.shape.clone(), Init::Zero));
                self.v.insert(id.clone(), Tensor::with_shape(param.params.shape.clone(), Init::Zero));
            }

            let m = self.m.get_mut(id).unwrap();
            let v = self.v.get_mut(id).unwrap();

            for i in 0..param.params.size {
                let g = param.gradient.data[i] + self.weight_decay * param.params.data[i];
                m.data[i] = self.beta1 * m.data[i] + (1.0 - self.beta1) * g;
                v.data[i] = self.beta2 * v.data[i] + (1.0 - self.beta2) * g * g;

                let m_corr = m.data[i] / denb1;
                let v_corr = v.data[i] / denb2;

                param.params.data[i] -= self.alpha * m_corr / (v_corr.sqrt() + self.epsilon);
            }
        }
    }
}
