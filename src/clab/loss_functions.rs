use crate::clab::tensor::{Tensor, Init};

/// Differentiable scalar loss over batched predictions and targets.
pub trait LossFunction {
    /// Computes the scalar loss value (used for monitoring, not for backprop).
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> f32;
    /// Returns the gradient of the loss with respect to `prediction`.
    fn backward(&self, prediction: &Tensor, target: &Tensor) -> Tensor;
}

/// Fused softmax + cross-entropy loss. Use with Linear output activation.
/// forward: applies softmax to logits, then computes CE.
/// backward: returns (softmax(logits) - target) / batch, bypassing the layer's softmax Jacobian.
pub struct SoftmaxCrossEntropyFunction;

impl LossFunction for SoftmaxCrossEntropyFunction {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> f32 {
        let rows = prediction.shape[0];
        let cols = prediction.shape[1];
        let mut result = 0.0f32;
        for i in 0..rows {
            let max = (0..cols).map(|j| prediction.data[i*cols+j]).fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = (0..cols).map(|j| (prediction.data[i*cols+j] - max).exp()).sum();
            for j in 0..cols {
                let p = (prediction.data[i*cols+j] - max).exp() / sum;
                result -= target.data[i*cols+j] * p.max(1e-10).ln();
            }
        }
        result / rows as f32
    }

    fn backward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let rows = prediction.shape[0];
        let cols = prediction.shape[1];
        let mut gradient = Tensor::with_shape(prediction.shape.clone(), Init::Zero);
        for i in 0..rows {
            let max = (0..cols).map(|j| prediction.data[i*cols+j]).fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = (0..cols).map(|j| (prediction.data[i*cols+j] - max).exp()).sum();
            for j in 0..cols {
                let p = (prediction.data[i*cols+j] - max).exp() / sum;
                gradient.data[i*cols+j] = (p - target.data[i*cols+j]) / rows as f32;
            }
        }
        gradient
    }
}

/// Mean squared error: `sum((pred - target)^2) / (2 * batch_size)`.
///
/// Gradient is `(pred - target) / batch_size`. Suitable for regression and
/// intrinsic motivation signals (e.g. [`ForwardModel`](crate::rl::forward_model::ForwardModel)).
pub struct MseFunction;

impl LossFunction for MseFunction {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> f32 {
        let mut result = 0.0;
        for i in 0..prediction.size {
            result += (prediction.data[i] - target.data[i]).powi(2);
        }
        result / (2.0 * prediction.shape[0] as f32)
    }

    fn backward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let mut gradient = Tensor::with_shape(vec![prediction.shape[0], prediction.shape[1]], crate::clab::tensor::Init::Zero);
        for i in 0..prediction.size {
            gradient.data[i] = (prediction.data[i] - target.data[i]) / prediction.shape[0] as f32;
        }
        gradient
    }
}
