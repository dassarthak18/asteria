use crate::clab::tensor::Tensor;

/// In-place element-wise activation applied during the forward and backward passes.
///
/// Implementors must be `Send + Sync` so networks can be shared across threads.
pub trait ActivationFunction: Send + Sync {
    /// Applies the activation in-place to `input` (forward pass).
    fn forward(&self, input: &mut Tensor);
    /// Multiplies `delta` element-wise by the activation's local gradient (backward pass).
    ///
    /// `input` holds the post-activation values from the forward pass.
    fn backward(&self, input: &Tensor, delta: &mut Tensor);
    /// Returns a heap-allocated clone; required because trait objects cannot use `Clone` directly.
    fn clone_box(&self) -> Box<dyn ActivationFunction>;
}

/// Identity activation — passes values through unchanged. Use as the output activation when
/// the loss function (e.g. [`SoftmaxCrossEntropyFunction`](super::loss_functions::SoftmaxCrossEntropyFunction)) handles non-linearity internally.
pub struct Linear;
impl ActivationFunction for Linear {
    fn forward(&self, _input: &mut Tensor) {}
    fn backward(&self, _input: &Tensor, _delta: &mut Tensor) {}
    fn clone_box(&self) -> Box<dyn ActivationFunction> { Box::new(Linear) }
}

/// Logistic sigmoid σ(x) = 1/(1+e^{-x}); output in (0,1).
pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn forward(&self, input: &mut Tensor) {
        for x in input.data.iter_mut() {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
    }
    fn backward(&self, input: &Tensor, delta: &mut Tensor) {
        for (i, d) in delta.data.iter_mut().enumerate() {
            let s = input.data[i];
            *d *= s * (1.0 - s);
        }
    }
    fn clone_box(&self) -> Box<dyn ActivationFunction> { Box::new(Sigmoid) }
}

/// Hyperbolic tangent tanh(x); output in (-1,1), zero-centred.
pub struct Tanh;
impl ActivationFunction for Tanh {
    fn forward(&self, input: &mut Tensor) {
        for x in input.data.iter_mut() {
            *x = x.tanh();
        }
    }
    fn backward(&self, input: &Tensor, delta: &mut Tensor) {
        for (i, d) in delta.data.iter_mut().enumerate() {
            let t = input.data[i];
            *d *= 1.0 - t * t;
        }
    }
    fn clone_box(&self) -> Box<dyn ActivationFunction> { Box::new(Tanh) }
}

/// Rectified Linear Unit: max(0, x). Dead neurons have zero gradient for negative inputs.
pub struct Relu;
impl ActivationFunction for Relu {
    fn forward(&self, input: &mut Tensor) {
        for x in input.data.iter_mut() {
            *x = x.max(0.0);
        }
    }
    fn backward(&self, input: &Tensor, delta: &mut Tensor) {
        for (i, d) in delta.data.iter_mut().enumerate() {
            if input.data[i] <= 0.0 {
                *d *= 0.0;
            }
        }
    }
    fn clone_box(&self) -> Box<dyn ActivationFunction> { Box::new(Relu) }
}

/// Row-wise softmax; converts logits to a probability distribution. Numerically stabilised
/// by subtracting the row maximum before exponentiation.
pub struct Softmax;
impl ActivationFunction for Softmax {
    fn forward(&self, input: &mut Tensor) {
        let rows = input.shape[0];
        let cols = input.shape[1];
        for i in 0..rows {
            let mut max = f32::NEG_INFINITY;
            for j in 0..cols {
                max = max.max(input.data[i * cols + j]);
            }
            let mut sum = 0.0;
            for j in 0..cols {
                input.data[i * cols + j] = (input.data[i * cols + j] - max).exp();
                sum += input.data[i * cols + j];
            }
            for j in 0..cols {
                input.data[i * cols + j] /= sum;
            }
        }
    }
    fn backward(&self, input: &Tensor, delta: &mut Tensor) {
        let rows = delta.shape[0];
        let cols = delta.shape[1];
        for n in 0..rows {
            let dot: f32 = (0..cols)
                .map(|i| delta.data[n * cols + i] * input.data[n * cols + i])
                .sum();
            for j in 0..cols {
                delta.data[n * cols + j] =
                    input.data[n * cols + j] * (delta.data[n * cols + j] - dot);
            }
        }
    }
    fn clone_box(&self) -> Box<dyn ActivationFunction> { Box::new(Softmax) }
}

/// Discriminant used to select an activation at construction time without generics.
pub enum ActivationType {
    /// No-op activation (see [`Linear`]).
    Linear,
    /// Logistic sigmoid (see [`Sigmoid`]).
    Sigmoid,
    /// Hyperbolic tangent (see [`Tanh`]).
    Tanh,
    /// Rectified linear unit (see [`Relu`]).
    Relu,
    /// Row-wise softmax (see [`Softmax`]).
    Softmax,
}

/// Instantiates the activation function corresponding to `t` and boxes it.
pub fn create_activation(t: ActivationType) -> Box<dyn ActivationFunction> {
    match t {
        ActivationType::Linear => Box::new(Linear),
        ActivationType::Sigmoid => Box::new(Sigmoid),
        ActivationType::Tanh => Box::new(Tanh),
        ActivationType::Relu => Box::new(Relu),
        ActivationType::Softmax => Box::new(Softmax),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clab::tensor::Tensor;

    fn make(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::from_data(shape, data)
    }

    // ── Linear ────────────────────────────────────────────────────────────────

    #[test]
    fn linear_forward_is_identity() {
        let mut t = make(vec![1.0, -2.0, 3.0], vec![1, 3]);
        let orig = t.data.clone();
        Linear.forward(&mut t);
        assert_eq!(t.data, orig);
    }

    #[test]
    fn linear_backward_passes_delta_unchanged() {
        let input = make(vec![1.0, 2.0], vec![1, 2]);
        let mut delta = make(vec![0.5, -0.5], vec![1, 2]);
        let orig = delta.data.clone();
        Linear.backward(&input, &mut delta);
        assert_eq!(delta.data, orig);
    }

    // ── Sigmoid ───────────────────────────────────────────────────────────────

    #[test]
    fn sigmoid_forward_at_zero_is_half() {
        let mut t = make(vec![0.0], vec![1, 1]);
        Sigmoid.forward(&mut t);
        assert!((t.data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_forward_output_in_unit_interval() {
        let mut t = make(vec![-10.0, 0.0, 10.0], vec![1, 3]);
        Sigmoid.forward(&mut t);
        assert!(t.data.iter().all(|&x| x > 0.0 && x < 1.0));
    }

    #[test]
    fn sigmoid_backward_at_half() {
        // σ(0) = 0.5, σ'(0) = 0.25
        let input = make(vec![0.5], vec![1, 1]);
        let mut delta = make(vec![1.0], vec![1, 1]);
        Sigmoid.backward(&input, &mut delta);
        assert!((delta.data[0] - 0.25).abs() < 1e-6);
    }

    // ── Tanh ──────────────────────────────────────────────────────────────────

    #[test]
    fn tanh_forward_at_zero_is_zero() {
        let mut t = make(vec![0.0], vec![1, 1]);
        Tanh.forward(&mut t);
        assert!(t.data[0].abs() < 1e-6);
    }

    #[test]
    fn tanh_forward_output_in_minus1_1() {
        let mut t = make(vec![-3.0, 0.0, 3.0], vec![1, 3]);
        Tanh.forward(&mut t);
        // tanh is bounded in (-1, 1) for finite inputs; use strict bounds for non-saturating inputs
        assert!(t.data.iter().all(|&x| x > -1.0 && x < 1.0));
    }

    #[test]
    fn tanh_backward_at_zero() {
        // tanh'(0) = 1 - tanh(0)² = 1
        let input = make(vec![0.0], vec![1, 1]);
        let mut delta = make(vec![2.0], vec![1, 1]);
        Tanh.backward(&input, &mut delta);
        assert!((delta.data[0] - 2.0).abs() < 1e-6);
    }

    // ── ReLU ──────────────────────────────────────────────────────────────────

    #[test]
    fn relu_forward_zeroes_negatives() {
        let mut t = make(vec![-3.0, 0.0, 2.0], vec![1, 3]);
        Relu.forward(&mut t);
        assert_eq!(t.data, vec![0.0, 0.0, 2.0]);
    }

    #[test]
    fn relu_backward_zeroes_dead_neurons() {
        let input = make(vec![-1.0, 0.0, 1.0], vec![1, 3]);
        let mut delta = make(vec![5.0, 5.0, 5.0], vec![1, 3]);
        Relu.backward(&input, &mut delta);
        assert_eq!(delta.data, vec![0.0, 0.0, 5.0]);
    }

    // ── Softmax ───────────────────────────────────────────────────────────────

    #[test]
    fn softmax_forward_sums_to_one_per_row() {
        let mut t = make(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        Softmax.forward(&mut t);
        for row in 0..2 {
            let sum: f32 = (0..3).map(|j| t.data[row * 3 + j]).sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {row} sums to {sum}");
        }
    }

    #[test]
    fn softmax_forward_numerically_stable() {
        // Large values should not overflow
        let mut t = make(vec![1000.0, 1001.0, 1002.0], vec![1, 3]);
        Softmax.forward(&mut t);
        let sum: f32 = t.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(t.data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn softmax_backward_jacobian() {
        // For a one-hot delta [1, 0, 0] with softmax output [s0, s1, s2]:
        // backward should produce s_j * (δ_j - s_0) for j=0,1,2
        let mut input = make(vec![2.0, 1.0, 0.0], vec![1, 3]);
        Softmax.forward(&mut input);
        let s = input.data.clone();

        let mut delta = make(vec![1.0, 0.0, 0.0], vec![1, 3]);
        Softmax.backward(&input, &mut delta);

        // dot = Σ δ_i * s_i = 1.0 * s[0]
        let dot = s[0];
        for j in 0..3 {
            let expected = s[j] * (delta.data[j] + (if j == 0 { 1.0 } else { 0.0 }) - dot);
            // after backward, delta[j] = s[j] * (original_delta[j] - dot)
            let expected_post = s[j] * ((if j == 0 { 1.0 } else { 0.0 }) - dot);
            assert!((delta.data[j] - expected_post).abs() < 1e-5,
                "j={j}: got {}, expected {expected_post}", delta.data[j]);
            let _ = expected; // suppress unused warning
        }
    }
}
