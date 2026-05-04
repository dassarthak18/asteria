use crate::clab::tensor::Tensor;

/// A container for a tensor value and its associated error gradient (delta).
///
/// Used inside [`NeuralNetwork`](crate::core::neural_network::NeuralNetwork) to track
/// intermediate activations and backpropagated errors for each layer in the graph.
#[derive(Clone, Debug)]
pub struct Variable {
    /// The current value (activation) of the variable.
    pub value: Tensor,
    /// The error gradient $\partial L/\partial x$ w.r.t. the variable's value.
    pub delta: Tensor,
}

impl Variable {
    /// Creates a new empty variable.
    pub fn new() -> Self {
        Variable {
            value: Tensor::new(),
            delta: Tensor::new(),
        }
    }
}
