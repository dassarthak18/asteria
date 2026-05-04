use crate::clab::tensor::{Tensor, Init};
use crate::clab::tensor_operator::TensorOperator;
use crate::core::param::Param;
use std::sync::{Arc, Mutex};

/// A neural network operator that performs a linear transformation: $y = xW + b$.
///
/// It encapsulates references to shared weight and bias parameters and handles the
/// accumulation of gradients during backpropagation.
pub struct LinearOperator {
    /// Reference to the weight matrix $[d_{in}, d_{out}]$.
    pub weights: Arc<Mutex<Param>>,
    /// Reference to the bias vector $[d_{out}]$.
    pub bias: Arc<Mutex<Param>>,
    /// Cached input from the last forward pass, required for gradient calculation.
    pub input: Option<Tensor>,
}

impl LinearOperator {
    /// Creates a operator linked to the provided parameters.
    pub fn new(weights: Arc<Mutex<Param>>, bias: Arc<Mutex<Param>>) -> Self {
        LinearOperator {
            weights,
            bias,
            input: None,
        }
    }

    /// Computes $y = xW + b$.
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input = Some(input.clone());
        let weights = self.weights.lock().unwrap();
        let bias = self.bias.lock().unwrap();
        
        let mut output = Tensor::with_shape(vec![input.shape[0], weights.params.shape[1]], Init::Zero);
        TensorOperator::mul(input, &weights.params, &mut output);
        
        let mut final_output = output.clone();
        TensorOperator::add(&output, &bias.params, &mut final_output);
        final_output
    }

    /// Computes gradients $\partial L/\partial W$ and $\partial L/\partial b$ and returns upstream delta $\partial L/\partial x$.
    pub fn backward(&mut self, delta: &Tensor) -> Tensor {
        let input = self.input.as_ref().expect("Forward must be called before backward");
        let mut weights = self.weights.lock().unwrap();
        let mut bias = self.bias.lock().unwrap();

        let mut next_delta = Tensor::with_shape(vec![delta.shape[0], weights.params.shape[0]], Init::Zero);
        
        weights.gradient.data.fill(0.0);

        let mut w_t = weights.params.clone();
        w_t.t();
        TensorOperator::mul(delta, &w_t, &mut next_delta);

        let mut i_t = input.clone();
        i_t.t();
        TensorOperator::mul(&i_t, delta, &mut weights.gradient);

        TensorOperator::reduce_sum(delta, &mut bias.gradient);
        
        next_delta
    }
}
