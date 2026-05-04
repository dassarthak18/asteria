use crate::core::neural_network::NeuralNetwork;
use crate::clab::tensor::{Tensor, Init};
use crate::core::optimizer::Optimizer;

/// ADOPT optimizer — an Adam variant with gradient clipping and a bootstrap step that
/// initialises the second moment before any parameter updates, improving early-training stability.
pub struct ADOPT {
    /// Learning rate.
    pub alpha: f32,
    /// First moment (momentum) decay rate.
    pub beta1: f32,
    /// Second moment (variance) decay rate.
    pub beta2: f32,
    /// Denominator stabiliser for the normalised gradient.
    pub epsilon: f32,
    /// L2 regularisation coefficient added to each gradient.
    pub weight_decay: f32,
    /// When `true`, clips the normalised gradient by `t^0.25` for extra stability.
    pub clip: bool,
    /// Global step counter; step 0 is the bootstrap pass (no parameter update).
    pub t: u32,
    /// Per-parameter first moment estimates.
    pub m: std::collections::HashMap<String, Tensor>,
    /// Per-parameter second moment estimates.
    pub v: std::collections::HashMap<String, Tensor>,
}

impl ADOPT {
    /// Creates an ADOPT optimizer with the given learning rate and sensible defaults.
    ///
    /// Defaults: `beta1=0.9`, `beta2=0.9999`, `epsilon=1e-6`, `weight_decay=0.0`, `clip=true`.
    pub fn new(alpha: f32) -> Self {
        ADOPT {
            alpha,
            beta1: 0.9,
            beta2: 0.9999,
            epsilon: 1e-6,
            weight_decay: 0.0,
            clip: true,
            t: 0,
            m: std::collections::HashMap::new(),
            v: std::collections::HashMap::new(),
        }
    }
}

impl Optimizer for ADOPT {
    fn set_lr(&mut self, lr: f32) { self.alpha = lr; }
    fn get_lr(&self) -> f32 { self.alpha }

    fn update(&mut self, network: &mut NeuralNetwork) {
        for (id, param_arc) in &network.model.model {
            let mut param = param_arc.lock().unwrap();

            if !self.m.contains_key(id) {
                self.m.insert(id.clone(), Tensor::with_shape(param.params.shape.clone(), Init::Zero));
                self.v.insert(id.clone(), Tensor::with_shape(param.params.shape.clone(), Init::Zero));
            }

            let m = self.m.get_mut(id).unwrap();
            let v = self.v.get_mut(id).unwrap();

            if self.t == 0 {
                for i in 0..param.params.size {
                    let g = param.gradient.data[i];
                    v.data[i] += g * g;
                }
            } else {
                let clip_val = if self.clip { (self.t as f32).powf(0.25) } else { f32::INFINITY };

                for i in 0..param.params.size {
                    let g = param.gradient.data[i] + self.weight_decay * param.params.data[i];

                    let denom = v.data[i].sqrt().max(self.epsilon);
                    let normed = (g / denom).clamp(-clip_val, clip_val);

                    m.data[i] = self.beta1 * m.data[i] + (1.0 - self.beta1) * normed;
                    param.params.data[i] -= self.alpha * m.data[i];
                    v.data[i] = self.beta2 * v.data[i] + (1.0 - self.beta2) * g * g;
                }
            }
        }
        self.t += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clab::tensor::Tensor;
    use crate::clab::activation_functions::ActivationType;
    use crate::clab::tensor_initializer::{TensorInitializer, InitializerType};
    use crate::core::neural_network::NeuralNetwork;
    use crate::core::dense_layer::DenseLayer;

    fn xor_network() -> NeuralNetwork {
        let mut net = NeuralNetwork::new();
        net.add_layer(DenseLayer::new("h".to_string(), 8, ActivationType::Sigmoid,
            TensorInitializer::new(InitializerType::LecunUniform), 2));
        net.add_layer(DenseLayer::new("o".to_string(), 1, ActivationType::Sigmoid,
            TensorInitializer::new(InitializerType::LecunUniform), 0));
        net.add_connection("h", "o");
        net.init();
        net
    }

    #[test]
    fn bootstrap_step_skips_param_update() {
        let mut net = xor_network();
        let mut opt = ADOPT::new(1e-2);

        let input = Tensor::from_data(vec![1, 2], vec![1.0, 0.0]);
        net.forward(&input);

        let params_before: Vec<f32> = net.model.model.values()
            .flat_map(|p| p.lock().unwrap().params.data.clone())
            .collect();

        for p in net.model.model.values() {
            let mut p = p.lock().unwrap();
            p.gradient.data.fill(0.1);
        }
        opt.update(&mut net);

        let params_after: Vec<f32> = net.model.model.values()
            .flat_map(|p| p.lock().unwrap().params.data.clone())
            .collect();

        assert_eq!(params_before, params_after, "step 0 must not update parameters");
        assert_eq!(opt.t, 1);
    }

    #[test]
    fn step1_updates_params() {
        let mut net = xor_network();
        let mut opt = ADOPT::new(1e-2);

        let input = Tensor::from_data(vec![1, 2], vec![1.0, 0.0]);
        net.forward(&input);

        for p in net.model.model.values() {
            let mut p = p.lock().unwrap();
            p.gradient.data.fill(0.1);
        }
        opt.update(&mut net);

        let params_before: Vec<f32> = net.model.model.values()
            .flat_map(|p| p.lock().unwrap().params.data.clone())
            .collect();

        for p in net.model.model.values() {
            let mut p = p.lock().unwrap();
            p.gradient.data.fill(0.1);
        }
        opt.update(&mut net);

        let params_after: Vec<f32> = net.model.model.values()
            .flat_map(|p| p.lock().unwrap().params.data.clone())
            .collect();

        assert_ne!(params_before, params_after, "step 1 must update parameters");
    }

    #[test]
    fn xor_converges() {
        let mut net = xor_network();
        let mut opt = ADOPT::new(1e-2);

        let input = Tensor::from_data(vec![4, 2], vec![0.0,0.0, 0.0,1.0, 1.0,0.0, 1.0,1.0]);
        let target = Tensor::from_data(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]);

        for _ in 0..1000 {
            let output = net.forward(&input).clone();
            let mut delta = Tensor::with_shape(vec![4, 1], crate::clab::tensor::Init::Zero);
            for i in 0..4 {
                delta.data[i] = (output.data[i] - target.data[i]) / 4.0;
            }
            net.backward(&mut delta);
            opt.update(&mut net);
        }

        let output = net.forward(&input).clone();
        assert!(output.data[0] < 0.2, "XOR(0,0) should be near 0, got {}", output.data[0]);
        assert!(output.data[1] > 0.8, "XOR(0,1) should be near 1, got {}", output.data[1]);
        assert!(output.data[2] > 0.8, "XOR(1,0) should be near 1, got {}", output.data[2]);
        assert!(output.data[3] < 0.2, "XOR(1,1) should be near 0, got {}", output.data[3]);
    }
}
