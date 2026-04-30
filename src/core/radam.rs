use crate::core::neural_network::NeuralNetwork;
use crate::clab::tensor::{Tensor, Init};
use crate::core::optimizer::Optimizer;

/// Rectified Adam (RAdam) optimizer. Falls back to SGD-with-momentum when the effective
/// second-moment estimate is unreliable (low step count), avoiding the high-variance warm-up
/// phase of standard Adam.
pub struct RAdam {
    /// Learning rate.
    pub alpha: f32,
    /// First moment decay (default 0.9).
    pub beta1: f32,
    /// Second moment decay (default 0.999).
    pub beta2: f32,
    /// Numerical stabiliser (default 1e-8).
    pub epsilon: f32,
    /// Global step counter.
    pub t: u32,
    /// Per-parameter first moment estimates.
    pub m: std::collections::HashMap<String, Tensor>,
    /// Per-parameter second moment estimates.
    pub v: std::collections::HashMap<String, Tensor>,
    /// Maximum SMA length `ρ∞ = 2/(1-β₂) - 1`; determines the threshold for adaptive updates.
    pub sma_inf: f32,
}

impl RAdam {
    /// Creates a RAdam optimizer with the given learning rate and sensible defaults.
    pub fn new(alpha: f32) -> Self {
        let beta1 = 0.9;
        let beta2 = 0.999;
        RAdam {
            alpha,
            beta1,
            beta2,
            epsilon: 1e-8,
            t: 0,
            m: std::collections::HashMap::new(),
            v: std::collections::HashMap::new(),
            sma_inf: 2.0 / (1.0 - beta2) - 1.0,
        }
    }
}

impl Optimizer for RAdam {
    fn set_lr(&mut self, lr: f32) { self.alpha = lr; }
    fn get_lr(&self) -> f32 { self.alpha }

    fn update(&mut self, network: &mut NeuralNetwork) {
        self.t += 1;
        let beta1_t = self.beta1.powi(self.t as i32);
        let sma_t = self.sma_inf - 2.0 * (self.t as f32) * self.beta2.powi(self.t as i32) / (1.0 - self.beta2.powi(self.t as i32));

        for (id, param_arc) in &network.model.model {
            let mut param = param_arc.lock().unwrap();
            
            if !self.m.contains_key(id) {
                self.m.insert(id.clone(), Tensor::with_shape(param.params.shape.clone(), Init::Zero));
                self.v.insert(id.clone(), Tensor::with_shape(param.params.shape.clone(), Init::Zero));
            }

            let m = self.m.get_mut(id).unwrap();
            let v = self.v.get_mut(id).unwrap();

            for i in 0..param.params.size {
                let g = param.gradient.data[i];
                m.data[i] = self.beta1 * m.data[i] + (1.0 - self.beta1) * g;
                v.data[i] = self.beta2 * v.data[i] + (1.0 - self.beta2) * g * g;

                let m_corr = m.data[i] / (1.0 - beta1_t);
                if sma_t > 5.0 {
                    let v_corr = (v.data[i] / (1.0 - self.beta2.powi(self.t as i32))).sqrt();
                    let r_t = ((sma_t - 4.0) * (sma_t - 2.0) * self.sma_inf / ((self.sma_inf - 4.0) * (self.sma_inf - 2.0) * sma_t)).sqrt();
                    param.params.data[i] -= self.alpha * r_t * m_corr / (v_corr + self.epsilon);
                } else {
                    param.params.data[i] -= self.alpha * m_corr;
                }
            }
        }
    }
}
