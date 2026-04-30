use crate::clab::tensor::Tensor;

/// A single trainable parameter tensor paired with its accumulated gradient.
///
/// Shared via `Arc<Mutex<Param>>` so that layers and optimizers can access the same
/// allocation without copying.
#[derive(Clone, Debug)]
pub struct Param {
    /// Unique string key used to look up this parameter in [`ParamModel`](super::param_model::ParamModel).
    pub id: String,
    /// Current parameter values (weights or biases).
    pub params: Tensor,
    /// Gradient accumulated during the backward pass; zeroed implicitly by the next backward.
    pub gradient: Tensor,
}

impl Param {
    /// Allocates zero-initialised parameter and gradient tensors with the given `shape`.
    pub fn new(shape: Vec<usize>) -> Self {
        Param {
            id: String::new(),
            params: Tensor::with_shape(shape.clone(), crate::clab::tensor::Init::Zero),
            gradient: Tensor::with_shape(shape, crate::clab::tensor::Init::Zero),
        }
    }
}
