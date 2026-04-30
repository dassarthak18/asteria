use crate::clab::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct Variable {
    pub value: Tensor,
    pub delta: Tensor,
}

impl Variable {
    pub fn new() -> Self {
        Variable {
            value: Tensor::new(),
            delta: Tensor::new(),
        }
    }
}
