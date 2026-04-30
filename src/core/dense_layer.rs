use crate::clab::tensor::Tensor;
use crate::clab::activation_functions::{ActivationFunction, create_activation, ActivationType};
use crate::clab::tensor_initializer::TensorInitializer;
use crate::core::param::Param;
use crate::core::param_model::ParamModel;
use crate::core::linear_operator::LinearOperator;
use std::sync::{Arc, Mutex};

/// A fully-connected (linear + activation) layer that participates in a [`crate::core::neural_network::NeuralNetwork`] DAG.
///
/// Weights and bias are stored as shared [`Param`] objects so the optimizer can update them
/// without holding a mutable reference to the layer itself.
pub struct DenseLayer {
    /// Unique identifier used to wire layers together in the network graph.
    pub id: String,
    /// Output dimensionality of this layer.
    pub dim: usize,
    /// Non-zero only for network-input layers; sets the external input width.
    pub input_dim: usize,
    /// Cached output tensor from the last forward pass.
    pub output: Tensor,
    /// Shared weight matrix `[input_dim, dim]`.
    pub weights: Option<Arc<Mutex<Param>>>,
    /// Shared bias vector `[dim]`.
    pub bias: Option<Arc<Mutex<Param>>>,
    /// Weight initialisation strategy applied in [`init`](DenseLayer::init).
    pub initializer: TensorInitializer,
    /// Activation function applied after the linear transform.
    pub af: Box<dyn ActivationFunction>,
    /// The linear operator (matmul + bias add) wired to the shared parameters.
    pub op: Option<LinearOperator>,
}

impl Clone for DenseLayer {
    fn clone(&self) -> Self {
        DenseLayer {
            id: self.id.clone(),
            dim: self.dim,
            input_dim: self.input_dim,
            output: self.output.clone(),
            weights: self.weights.clone(),
            bias: self.bias.clone(),
            initializer: self.initializer.clone(),
            af: self.af.clone_box(),
            op: None, 
        }
    }
}

impl DenseLayer {
    /// Constructs a layer; weights are not allocated until [`init`](Self::init) is called.
    ///
    /// Set `input_dim > 0` only for the first layer(s) that receive external data directly.
    pub fn new(id: String, dim: usize, activation: ActivationType, initializer: TensorInitializer, input_dim: usize) -> Self {
        DenseLayer {
            id,
            dim,
            input_dim,
            output: Tensor::new(),
            weights: None,
            bias: None,
            initializer,
            af: create_activation(activation),
            op: None,
        }
    }

    /// Registers weight and bias parameters with `model` and wires them to a [`LinearOperator`].
    ///
    /// `input_dims` lists the output widths of predecessor layers in the DAG; they are summed
    /// together with `self.input_dim` to determine the weight matrix width.
    pub fn init(&mut self, model: &mut ParamModel, input_dims: &[usize]) {
        let mut total_input_dim = self.input_dim;
        for &dim in input_dims {
            total_input_dim += dim;
        }

        let weights = model.add_param(vec![total_input_dim, self.dim]);
        self.initializer.init(&mut weights.lock().unwrap().params);
        self.weights = Some(weights.clone());

        let bias = model.add_param(vec![self.dim]);
        self.bias = Some(bias.clone());

        self.op = Some(LinearOperator::new(weights, bias));
    }

    /// Computes the linear transform followed by the activation, caching the result in `self.output`.
    pub fn forward(&mut self, input: &Tensor) -> &Tensor {
        let op = self.op.as_mut().expect("Layer must be initialized before forward");
        let mut linear_output = op.forward(input);
        self.af.forward(&mut linear_output);
        self.output = linear_output;
        &self.output
    }

    /// Applies the activation's local gradient to `delta`, then backpropagates through the linear
    /// operator, accumulating weight/bias gradients and returning the upstream delta.
    pub fn backward(&mut self, delta: &mut Tensor) -> Tensor {
        let op = self.op.as_mut().expect("Layer must be initialized before backward");
        self.af.backward(&self.output, delta);
        op.backward(delta)
    }
}
