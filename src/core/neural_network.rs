use std::collections::{HashMap, VecDeque};
use crate::clab::tensor::{Tensor, Init};
use crate::core::dense_layer::DenseLayer;
use crate::core::param_model::ParamModel;
use crate::core::variable::Variable;

/// DAG-based feedforward neural network supporting arbitrary layer connections.
///
/// Build a network by calling [`add_layer`](NeuralNetwork::add_layer) and
/// [`add_connection`](NeuralNetwork::add_connection), then call [`init`](NeuralNetwork::init) once
/// to allocate weights and compute topological order before any forward/backward passes.
pub struct NeuralNetwork {
    /// Shared parameter store; optimizers iterate over this to apply gradient updates.
    pub model: ParamModel,
    /// Map from layer id to its [`DenseLayer`].
    pub layers: HashMap<String, DenseLayer>,
    /// Adjacency list: `graph[to]` = list of `from` ids (inputs to `to`).
    pub graph: HashMap<String, Vec<String>>,
    /// Layer ids in topological (forward) order, computed by [`init`](NeuralNetwork::init).
    pub forward_graph: Vec<String>,
    /// Reverse of `forward_graph`, used during backprop.
    pub backward_graph: Vec<String>,
    /// Ids of layers that accept external input tensors.
    pub input_layers: Vec<String>,
    /// Id of the final layer whose output is returned by [`forward`](NeuralNetwork::forward).
    pub output_layer: String,
    /// Per-layer activation and gradient bookkeeping.
    pub layer_variables: HashMap<String, LayerVariable>,
    /// External input tensors, keyed by input-layer id.
    pub inputs: HashMap<String, Variable>,
}

/// Per-layer intermediate tensors needed for the forward/backward pass.
#[derive(Clone)]
pub struct LayerVariable {
    /// Concatenated input to this layer during the last forward pass.
    pub input: Tensor,
    /// Gradient flowing into this layer during the last backward pass.
    pub delta: Tensor,
    /// Ordered list of ids whose outputs feed this layer's input.
    pub input_list: Vec<String>,
    /// Ordered list of ids that receive gradient from this layer.
    pub delta_list: Vec<String>,
}

impl Clone for NeuralNetwork {
    fn clone(&self) -> Self {
        let mut new_net = NeuralNetwork::new();
        for layer in self.layers.values() {
            new_net.add_layer(layer.clone());
        }
        new_net.graph = self.graph.clone();
        new_net.init();
        new_net
    }
}

impl NeuralNetwork {
    /// Creates an empty network with no layers or connections.
    pub fn new() -> Self {
        NeuralNetwork {
            model: ParamModel::new(),
            layers: HashMap::new(),
            graph: HashMap::new(),
            forward_graph: Vec::new(),
            backward_graph: Vec::new(),
            input_layers: Vec::new(),
            output_layer: String::new(),
            layer_variables: HashMap::new(),
            inputs: HashMap::new(),
        }
    }

    /// Soft- or hard-copies parameters from `other` into `self`.
    ///
    /// `ratio = 1.0` performs a hard copy; `ratio < 1.0` performs a Polyak average
    /// `self = ratio * other + (1 - ratio) * self`, used for target network updates.
    pub fn copy_params(&mut self, other: &NeuralNetwork, ratio: f32) {
        for (id, param) in &mut self.model.model {
            if let Some(other_param) = other.model.model.get(id) {
                let mut p = param.lock().unwrap();
                let op = other_param.lock().unwrap();
                if ratio < 1.0 {
                    for i in 0..p.params.size {
                        p.params.data[i] = op.params.data[i] * ratio + p.params.data[i] * (1.0 - ratio);
                    }
                } else {
                    p.params.data.copy_from_slice(&op.params.data);
                }
            }
        }
    }

    /// Registers a layer; layers with `input_dim > 0` are treated as network inputs.
    pub fn add_layer(&mut self, layer: DenseLayer) {
        let id = layer.id.clone();
        if layer.input_dim > 0 {
            self.input_layers.push(id.clone());
            let mut var = Variable::new();
            var.delta.resize(vec![1, layer.input_dim], Init::Zero);
            self.inputs.insert(id.clone(), var);
        }
        self.layers.insert(id.clone(), layer);
        self.graph.insert(id.clone(), Vec::new());
        self.layer_variables.insert(id.clone(), LayerVariable {
            input: Tensor::new(),
            delta: Tensor::new(),
            input_list: Vec::new(),
            delta_list: Vec::new(),
        });
    }

    /// Declares that layer `from`'s output feeds into layer `to`'s input.
    pub fn add_connection(&mut self, from: &str, to: &str) {
        if let Some(adj) = self.graph.get_mut(to) {
            adj.push(from.to_string());
        }
    }

    /// Computes topological order and allocates weight tensors for every layer.
    ///
    /// Must be called once after all layers and connections are registered, and again after
    /// structural changes. Subsequent calls to [`forward`](Self::forward) and
    /// [`backward`](Self::backward) rely on the order computed here.
    pub fn init(&mut self) {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for id in self.layers.keys() {
            in_degree.insert(id.clone(), 0);
        }

        for (to, froms) in &self.graph {
            in_degree.insert(to.clone(), froms.len());
        }

        let mut queue: VecDeque<String> = VecDeque::new();
        for id in &self.input_layers {
            queue.push_back(id.clone());
        }

        self.forward_graph.clear();
        let mut visited = HashMap::new();
        for id in self.layers.keys() { visited.insert(id.clone(), false); }

        while let Some(v) = queue.pop_front() {
            if *visited.get(&v).unwrap() { continue; }
            self.forward_graph.push(v.clone());
            visited.insert(v.clone(), true);

            for (to, froms) in &self.graph {
                if froms.contains(&v) && !*visited.get(to).unwrap() {
                    queue.push_back(to.clone());
                }
            }
        }

        self.backward_graph = self.forward_graph.clone();
        self.backward_graph.reverse();

        if let Some(last) = self.forward_graph.last() {
            self.output_layer = last.clone();
        }

        let layer_ids: Vec<String> = self.forward_graph.clone();
        for id in layer_ids {
            let input_ids = self.graph.get(&id).unwrap().clone();
            
            {
                let lv = self.layer_variables.get_mut(&id).unwrap();
                lv.input_list.clear();
                for in_id in &input_ids {
                    lv.input_list.push(in_id.clone());
                }
                if self.inputs.contains_key(&id) {
                    lv.input_list.push(id.clone());
                }
            }

            let mut input_dims = Vec::new();
            for in_id in &input_ids {
                input_dims.push(self.layers.get(in_id).unwrap().dim);
            }
            
            let layer = self.layers.get_mut(&id).unwrap();
            layer.init(&mut self.model, &input_dims);
        }
    }

    /// Runs a forward pass through all layers in topological order and returns the output tensor.
    ///
    /// `input` is broadcast to all input layers; rank-1 tensors are automatically reshaped to
    /// `[1, size]`. The returned reference is valid until the next call to `forward`.
    pub fn forward(&mut self, input: &Tensor) -> &Tensor {
        for id in &self.input_layers {
            let var = self.inputs.get_mut(id).unwrap();
            if input.rank == 1 {
                var.value = Tensor::from_data(vec![1, input.size], input.data.clone());
            } else {
                var.value = input.clone();
            }
        }

        let ids = self.forward_graph.clone();
        for id in ids {
            let mut inputs_to_concat = Vec::new();
            {
                let lv = self.layer_variables.get(&id).unwrap();
                for in_id in &lv.input_list {
                    if in_id == &id {
                        inputs_to_concat.push(self.inputs.get(in_id).unwrap().value.clone());
                    } else if let Some(in_layer) = self.layers.get(in_id) {
                        inputs_to_concat.push(in_layer.output.clone());
                    } else if let Some(in_var) = self.inputs.get(in_id) {
                        inputs_to_concat.push(in_var.value.clone());
                    }
                }
            }

            let final_input = if inputs_to_concat.len() > 1 {
                 let rows = inputs_to_concat[0].shape[0];
                 let mut cols = 0;
                 for t in &inputs_to_concat { cols += t.shape[1]; }
                 let mut res = Tensor::with_shape(vec![rows, cols], Init::Zero);
                 let mut offset = 0;
                 for t in &inputs_to_concat {
                     for i in 0..rows {
                         for j in 0..t.shape[1] {
                             res.data[i * cols + offset + j] = t.data[i * t.shape[1] + j];
                         }
                     }
                     offset += t.shape[1];
                 }
                 res
            } else {
                inputs_to_concat[0].clone()
            };

            let layer = self.layers.get_mut(&id).unwrap();
            layer.forward(&final_input);
        }

        &self.layers.get(&self.output_layer).unwrap().output
    }

    /// Saves all network parameters to a binary file.
    ///
    /// Parameters are written in creation order (param_0, param_1, …) as
    /// little-endian f32 bytes. The file can be loaded back with [`Self::load`]
    /// as long as the network has the identical architecture.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        let mut keys: Vec<String> = self.model.model.keys().cloned().collect();
        keys.sort_by_key(|k| k.strip_prefix("param_").and_then(|n| n.parse::<u32>().ok()).unwrap_or(0));
        for key in &keys {
            let param = self.model.model[key].lock().unwrap();
            for &v in &param.params.data {
                file.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Loads parameters from a file previously written by [`Self::save`].
    ///
    /// The network must already be initialised with the same architecture.
    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut keys: Vec<String> = self.model.model.keys().cloned().collect();
        keys.sort_by_key(|k| k.strip_prefix("param_").and_then(|n| n.parse::<u32>().ok()).unwrap_or(0));
        for key in &keys {
            let mut param = self.model.model[key].lock().unwrap();
            let n = param.params.size;
            let mut buf = vec![0u8; n * 4];
            file.read_exact(&mut buf)?;
            for (i, chunk) in buf.chunks_exact(4).enumerate() {
                param.params.data[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        Ok(())
    }

    /// Propagates `delta` (loss gradient w.r.t. output) backwards through the network,
    /// accumulating gradients in each layer's parameter tensors ready for an optimizer step.
    pub fn backward(&mut self, delta: &mut Tensor) {
        self.layer_variables.get_mut(&self.output_layer).unwrap().delta = delta.clone();

        let ids = self.backward_graph.clone();
        for id in ids {
            let mut d = self.layer_variables.get(&id).unwrap().delta.clone();
            let layer = self.layers.get_mut(&id).unwrap();
            let next_delta = layer.backward(&mut d);
            
            let input_list = self.layer_variables.get(&id).unwrap().input_list.clone();
            if input_list.len() == 1 {
                let in_id = &input_list[0];
                if let Some(in_lv) = self.layer_variables.get_mut(in_id) {
                    in_lv.delta = next_delta;
                } else if let Some(in_var) = self.inputs.get_mut(in_id) {
                    in_var.delta = next_delta;
                }
            } else {
                let mut offset = 0;
                for in_id in &input_list {
                    let in_dim = if let Some(l) = self.layers.get(in_id) { l.dim } else { self.inputs.get(in_id).unwrap().value.shape[1] };
                    let rows = next_delta.shape[0];
                    let mut split_delta = Tensor::with_shape(vec![rows, in_dim], Init::Zero);
                    for i in 0..rows {
                        for j in 0..in_dim {
                            split_delta.data[i * in_dim + j] = next_delta.data[i * next_delta.shape[1] + offset + j];
                        }
                    }
                    if let Some(in_lv) = self.layer_variables.get_mut(in_id) {
                        in_lv.delta = split_delta;
                    } else if let Some(in_var) = self.inputs.get_mut(in_id) {
                        in_var.delta = split_delta;
                    }
                    offset += in_dim;
                }
            }
        }
    }
}
