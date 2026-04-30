use crate::clab::tensor::Tensor;
use crate::clab::random_generator::RandomGenerator;

/// Strategy used by [`TensorInitializer`] to fill a weight tensor before training.
#[derive(Clone, Debug)]
pub enum InitializerType {
    /// Fills every element with the same constant; useful for deterministic unit tests.
    Debug(f32),
    /// Draws each element uniformly from `[low, high)`.
    Uniform(f32, f32),
    /// Draws each element from N(`mean`, `sigma`).
    Normal(f32, f32),
    /// LeCun uniform: scale based on fan-in; recommended for Tanh/Sigmoid networks.
    LecunUniform,
    /// LeCun normal: like `LecunUniform` but Gaussian draws.
    LecunNormal,
    /// Glorot/Xavier uniform: scale based on fan-in + fan-out.
    GlorotUniform,
    /// Glorot/Xavier normal: Gaussian variant of `GlorotUniform`.
    GlorotNormal,
    /// Alias for `GlorotUniform` (same formula).
    XavierUniform,
    /// Xavier normal with a slightly wider spread than `GlorotNormal`.
    XavierNormal,
}

/// Applies a chosen weight-initialisation strategy to a [`Tensor`] in-place.
#[derive(Clone)]
pub struct TensorInitializer {
    /// The initialisation strategy to use.
    pub init_type: InitializerType,
}

impl TensorInitializer {
    /// Creates an initializer using the given strategy.
    pub fn new(init_type: InitializerType) -> Self {
        TensorInitializer { init_type }
    }

    /// Fills `tensor` according to the configured [`InitializerType`].
    pub fn init(&self, tensor: &mut Tensor) {
        let rg = RandomGenerator::instance();
        match self.init_type {
            InitializerType::Debug(val) => {
                for x in tensor.data.iter_mut() {
                    *x = val;
                }
            }
            InitializerType::Uniform(low, high) => {
                for x in tensor.data.iter_mut() {
                    *x = rg.random_float(low, high);
                }
            }
            InitializerType::Normal(mean, sigma) => {
                for x in tensor.data.iter_mut() {
                    *x = rg.normal_random(mean, sigma);
                }
            }
            InitializerType::LecunUniform => {
                let limit = (3.0 / tensor.shape[0] as f32).sqrt();
                self.init_uniform(tensor, -limit, limit);
            }
            InitializerType::LecunNormal => {
                let sigma = (1.0 / tensor.shape[0] as f32).sqrt();
                self.init_normal(tensor, 0.0, sigma);
            }
            InitializerType::GlorotUniform => {
                let limit = (6.0 / (tensor.shape[0] + tensor.shape[1]) as f32).sqrt();
                self.init_uniform(tensor, -limit, limit);
            }
            InitializerType::GlorotNormal => {
                let sigma = (2.0 / (tensor.shape[0] + tensor.shape[1]) as f32).sqrt();
                self.init_normal(tensor, 0.0, sigma);
            }
            InitializerType::XavierUniform => {
                let limit = (6.0 / (tensor.shape[0] + tensor.shape[1]) as f32).sqrt();
                self.init_uniform(tensor, -limit, limit);
            }
            InitializerType::XavierNormal => {
                let sigma = (3.0 / (tensor.shape[0] + tensor.shape[1]) as f32).sqrt();
                self.init_normal(tensor, 0.0, sigma);
            }
        }
    }

    fn init_uniform(&self, tensor: &mut Tensor, low: f32, high: f32) {
        let rg = RandomGenerator::instance();
        for x in tensor.data.iter_mut() {
            *x = rg.random_float(low, high);
        }
    }

    fn init_normal(&self, tensor: &mut Tensor, mean: f32, sigma: f32) {
        let rg = RandomGenerator::instance();
        for x in tensor.data.iter_mut() {
            *x = rg.normal_random(mean, sigma);
        }
    }
}
