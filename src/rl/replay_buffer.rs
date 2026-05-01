use crate::clab::tensor::Tensor;
use rand::prelude::*;

/// A single `(s₀, a, s₁, r, done)` transition collected during an episode.
#[derive(Clone, Debug)]
pub struct MdpTransition {
    /// Observation at the start of the transition.
    pub s0: Tensor,
    /// Action taken (one-hot for discrete; continuous vector for continuous action spaces).
    pub a: Tensor,
    /// Observation after the action was applied.
    pub s1: Tensor,
    /// Scalar reward received on this transition.
    pub r: f32,
    /// `true` if `s1` is a terminal (absorbing) state.
    pub final_state: bool,
}

/// Fixed-capacity circular experience replay buffer used by off-policy algorithms.
///
/// Once the buffer reaches `max_size`, inserting a new transition evicts the oldest one
/// (FIFO). Uniform random mini-batches are drawn by [`sample`](ReplayBuffer::sample)
/// for experience replay (DQN, DDPG, PPO, …).
pub struct ReplayBuffer<T> {
    /// Stored transitions in insertion order.
    pub buffer: Vec<T>,
    /// Maximum number of transitions the buffer can hold.
    pub max_size: usize,
}

impl<T: Clone> ReplayBuffer<T> {
    /// Creates an empty buffer with the given capacity.
    pub fn new(max_size: usize) -> Self {
        ReplayBuffer {
            buffer: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Inserts `item`, evicting the oldest entry when the buffer is at capacity.
    pub fn add_item(&mut self, item: T) {
        if self.buffer.len() == self.max_size {
            self.buffer.remove(0);
        }
        self.buffer.push(item);
    }

    /// Returns up to `sample_size` transitions chosen uniformly at random (without replacement).
    pub fn sample(&self, sample_size: usize) -> Vec<T> {
        let mut rng = rand::rng();
        let n = sample_size.min(self.buffer.len());
        self.buffer.sample(&mut rng, n).cloned().collect()
    }

    /// Returns the number of transitions currently stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}
