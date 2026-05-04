use std::collections::HashMap;
use crate::core::param::Param;
use std::sync::{Arc, Mutex};

/// Registry of all trainable parameters in a network, keyed by auto-generated ids.
///
/// Optimizers iterate over `model` to apply gradient updates; layers hold `Arc` clones
/// of individual entries so they can mutate weights during the forward/backward passes.
pub struct ParamModel {
    /// All registered parameters, keyed by `"param_N"` in insertion order.
    pub model: HashMap<String, Arc<Mutex<Param>>>,
    next_id: u32,
}

impl ParamModel {
    /// Creates an empty parameter registry.
    pub fn new() -> Self {
        ParamModel {
            model: HashMap::new(),
            next_id: 0,
        }
    }

    /// Allocates and registers a new parameter with the given `shape`.
    pub fn add_param(&mut self, shape: Vec<usize>) -> Arc<Mutex<Param>> {
        let mut p = Param::new(shape);
        p.id = format!("param_{}", self.next_id);
        self.next_id += 1;
        let arc = Arc::new(Mutex::new(p));
        self.model.insert(arc.lock().unwrap().id.clone(), arc.clone());
        arc
    }

    /// Registers an existing parameter reference (e.g. for weight sharing).
    pub fn add_existing_param(&mut self, param: Arc<Mutex<Param>>) {
        let id = param.lock().unwrap().id.clone();
        self.model.insert(id, param);
    }
}
