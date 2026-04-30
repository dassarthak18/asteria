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
