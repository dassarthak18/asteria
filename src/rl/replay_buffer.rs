use crate::clab::tensor::Tensor;
use rand::prelude::*;

#[derive(Clone, Debug)]
pub struct MdpTransition {
    pub s0: Tensor,
    pub a: Tensor,
    pub s1: Tensor,
    pub r: f32,
    pub final_state: bool,
}

pub struct ReplayBuffer<T> {
    pub buffer: Vec<T>,
    pub max_size: usize,
}

impl<T: Clone> ReplayBuffer<T> {
    pub fn new(max_size: usize) -> Self {
        ReplayBuffer {
            buffer: Vec::with_capacity(max_size),
            max_size,
        }
    }

    pub fn add_item(&mut self, item: T) {
        if self.buffer.len() == self.max_size {
            self.buffer.remove(0);
        }
        self.buffer.push(item);
    }

    pub fn sample(&self, sample_size: usize) -> Vec<T> {
        let mut rng = rand::rng();
        let n = sample_size.min(self.buffer.len());
        self.buffer.sample(&mut rng, n).cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}
