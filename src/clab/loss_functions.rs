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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clab::tensor::Tensor;

    // ── MSE ───────────────────────────────────────────────────────────────────

    #[test]
    fn mse_forward_zero_on_perfect_prediction() {
        let pred = Tensor::from_data(vec![1, 3], vec![1.0, 2.0, 3.0]);
        let target = Tensor::from_data(vec![1, 3], vec![1.0, 2.0, 3.0]);
        assert!((MseFunction.forward(&pred, &target)).abs() < 1e-6);
    }

    #[test]
    fn mse_forward_known_value() {
        // batch=1, errors=[1, 2]: MSE = (1² + 2²) / (2*1) = 5/2 = 2.5
        let pred = Tensor::from_data(vec![1, 2], vec![2.0, 4.0]);
        let target = Tensor::from_data(vec![1, 2], vec![1.0, 2.0]);
        assert!((MseFunction.forward(&pred, &target) - 2.5).abs() < 1e-5);
    }

    #[test]
    fn mse_backward_gradient_is_diff_over_batch() {
        let pred = Tensor::from_data(vec![2, 2], vec![3.0, 1.0, 5.0, 2.0]);
        let target = Tensor::from_data(vec![2, 2], vec![1.0, 1.0, 3.0, 2.0]);
        let grad = MseFunction.backward(&pred, &target);
        // gradient = (pred - target) / batch_size
        assert!((grad.data[0] - 1.0).abs() < 1e-6); // (3-1)/2
        assert!((grad.data[1] - 0.0).abs() < 1e-6); // (1-1)/2
        assert!((grad.data[2] - 1.0).abs() < 1e-6); // (5-3)/2
        assert!((grad.data[3] - 0.0).abs() < 1e-6); // (2-2)/2
    }

    // ── SoftmaxCrossEntropy ───────────────────────────────────────────────────

    #[test]
    fn sce_forward_near_zero_on_confident_correct_prediction() {
        // Large positive logit for correct class → softmax ≈ 1 → CE ≈ 0
        let pred = Tensor::from_data(vec![1, 3], vec![10.0, 0.0, 0.0]);
        let target = Tensor::from_data(vec![1, 3], vec![1.0, 0.0, 0.0]);
        assert!(SoftmaxCrossEntropyFunction.forward(&pred, &target) < 0.01);
    }

    #[test]
    fn sce_forward_high_on_wrong_class() {
        // Large positive logit for wrong class → high CE
        let pred = Tensor::from_data(vec![1, 3], vec![0.0, 10.0, 0.0]);
        let target = Tensor::from_data(vec![1, 3], vec![1.0, 0.0, 0.0]);
        assert!(SoftmaxCrossEntropyFunction.forward(&pred, &target) > 5.0);
    }

    #[test]
    fn sce_backward_shape_matches_input() {
        let pred = Tensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let target = Tensor::from_data(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let grad = SoftmaxCrossEntropyFunction.backward(&pred, &target);
        assert_eq!(grad.shape, vec![2, 3]);
    }

    #[test]
    fn sce_backward_sums_near_zero_per_row() {
        // (softmax(z) - y) sums to 0 for any one-hot y, before the /batch division
        let pred = Tensor::from_data(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let target = Tensor::from_data(vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]);
        let grad = SoftmaxCrossEntropyFunction.backward(&pred, &target);
        let row_sum: f32 = grad.data.iter().sum();
        assert!(row_sum.abs() < 1e-6, "per-row gradient sum should be ~0, got {row_sum}");
    }
}
