use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::replay_buffer::{ReplayBuffer, MdpTransition};

pub struct DQN {
    pub network: NeuralNetwork,
    pub optimizer: Box<dyn Optimizer>,
    pub gamma: f32,
    pub memory: ReplayBuffer<MdpTransition>,
    pub sample_size: usize,
    pub critic_target: NeuralNetwork,
    pub target_update_frequency: usize,
    pub target_update_step: usize,
    pub batch_state: Tensor,
    pub batch_action: Tensor,
    pub batch_next_state: Tensor,
    pub batch_reward: Tensor,
    pub batch_mask: Tensor,
    pub loss: Tensor,
}

impl DQN {
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
        // max_index(0) now returns column indices (one per row)
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

    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        let q_values = self.network.forward(state);
        let mut action = Tensor::with_shape_val(vec![q_values.shape[1]], 0.0);
        let max_idx = q_values.max_index(0)[0];
        action.set(vec![max_idx], 1.0);
        action
    }
}
