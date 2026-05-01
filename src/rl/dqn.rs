use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::replay_buffer::{ReplayBuffer, MdpTransition};

/// Deep Q-Network (DQN) with experience replay and a frozen target network.
///
/// DQN stabilises Q-learning with two techniques:
/// - **Experience replay**: transitions are stored in a [`ReplayBuffer`] and sampled
///   uniformly at random, breaking harmful temporal correlations in the training data.
/// - **Target network**: a periodically-updated copy of the Q-network is used to
///   compute Bellman targets, preventing the moving-target instability that arises
///   when both sides of the update use the same weights.
///
/// The target network is hard-copied from the online network every `target_update_frequency`
/// gradient steps.
pub struct DQN {
    /// Online Q-network updated every step.
    pub network: NeuralNetwork,
    /// Optimizer used to update the online network.
    pub optimizer: Box<dyn Optimizer>,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// Experience replay buffer.
    pub memory: ReplayBuffer<MdpTransition>,
    /// Number of transitions per mini-batch.
    pub sample_size: usize,
    /// Frozen target network used to compute Bellman targets.
    pub critic_target: NeuralNetwork,
    /// Number of gradient steps between hard target-network updates.
    pub target_update_frequency: usize,
    /// Counter tracking gradient steps since the last target update.
    pub target_update_step: usize,
    /// Batched state tensor assembled from the current replay sample.
    pub batch_state: Tensor,
    /// Batched action tensor (one-hot) assembled from the current replay sample.
    pub batch_action: Tensor,
    /// Batched next-state tensor assembled from the current replay sample.
    pub batch_next_state: Tensor,
    /// Batched reward tensor (shape `[sample_size, 1]`).
    pub batch_reward: Tensor,
    /// Batched termination mask (1.0 = non-terminal, 0.0 = terminal; shape `[sample_size, 1]`).
    pub batch_mask: Tensor,
    /// Loss tensor from the last critic update.
    pub loss: Tensor,
}

impl DQN {
    /// Creates a new DQN agent.
    ///
    /// - `network`: Q-network with one output per action.
    /// - `optimizer`: parameter update rule.
    /// - `gamma`: discount factor.
    /// - `memory_size`: replay buffer capacity.
    /// - `sample_size`: mini-batch size for each update.
    /// - `target_update_frequency`: gradient steps between target-network hard copies.
    pub fn new(network: NeuralNetwork, optimizer: Box<dyn Optimizer>, gamma: f32, memory_size: usize, sample_size: usize, target_update_frequency: usize) -> Self {
        let critic_target = network.clone();
        DQN {
            network,
            optimizer,
            gamma,
            memory: ReplayBuffer::new(memory_size),
            sample_size,
            critic_target,
            target_update_frequency,
            target_update_step: 0,
            batch_state: Tensor::new(),
            batch_action: Tensor::new(),
            batch_next_state: Tensor::new(),
            batch_reward: Tensor::new(),
            batch_mask: Tensor::new(),
            loss: Tensor::new(),
        }
    }

    /// Stores the transition and, once the replay buffer has enough data, performs a DQN update.
    ///
    /// Training only begins once `memory.len() >= sample_size`. The target network is
    /// hard-copied every `target_update_frequency` gradient steps.
    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) {
        self.memory.add_item(MdpTransition {
            s0: state.clone(),
            a: action.clone(),
            s1: next_state.clone(),
            r: reward,
            final_state,
        });

        if self.memory.len() >= self.sample_size {
            self.process_sample();

            let mut loss = self.critic_loss_function();
            self.network.backward(&mut loss);
            self.optimizer.update(&mut self.network);

            self.target_update_step += 1;
            if self.target_update_step == self.target_update_frequency {
                self.critic_target.copy_params(&self.network, 1.0);
                self.target_update_step = 0;
            }
        }
    }

    fn process_sample(&mut self) {
        let sample = self.memory.sample(self.sample_size);
        let actual_sample_size = sample.len();

        self.batch_reward = Tensor::with_shape_val(vec![actual_sample_size, 1], 0.0);
        self.batch_mask = Tensor::with_shape_val(vec![actual_sample_size, 1], 0.0);

        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut next_states = Vec::new();

        for (i, transition) in sample.iter().enumerate() {
            states.push(&transition.s0);
            actions.push(&transition.a);
            next_states.push(&transition.s1);
            self.batch_reward.set(vec![i, 0], transition.r);
            self.batch_mask.set(vec![i, 0], if transition.final_state { 0.0 } else { 1.0 });
        }

        self.batch_state = Tensor::concat(&states, 0);
        self.batch_action = Tensor::concat(&actions, 0);
        self.batch_next_state = Tensor::concat(&next_states, 0);
    }

    fn critic_loss_function(&mut self) -> Tensor {
        let q_next_values = self.critic_target.forward(&self.batch_next_state).clone();
        let n_actions = q_next_values.shape[1];
        // max_index(0) returns per-row column indices; convert to flat buffer offsets for gather.
        let a_max_cols = q_next_values.max_index(0);
        let a_max_flat: Vec<usize> = a_max_cols.iter().enumerate().map(|(i, &c)| i * n_actions + c).collect();
        let max_q_values = q_next_values.gather(&a_max_flat);

        let q_all = self.network.forward(&self.batch_state).clone();
        let a0_cols = self.batch_action.max_index(0);
        let a0_flat: Vec<usize> = a0_cols.iter().enumerate().map(|(i, &c)| i * n_actions + c).collect();
        let q_values = q_all.gather(&a0_flat);

        let mut loss = Tensor::zero_like(&q_all);

        for i in 0..self.sample_size {
            let val = (q_values.get(vec![i]) - (self.batch_reward.get(vec![i, 0]) + self.gamma * self.batch_mask.get(vec![i, 0]) * max_q_values.get(vec![i]))) / self.sample_size as f32;
            loss.data[a0_flat[i]] = val;
        }
        loss
    }

    /// Returns the greedy action for `state` as a one-hot tensor (shape `[n_actions]`).
    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        let q_values = self.network.forward(state);
        let mut action = Tensor::with_shape_val(vec![q_values.shape[1]], 0.0);
        let max_idx = q_values.max_index(0)[0];
        action.set(vec![max_idx], 1.0);
        action
    }
}
