use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::replay_buffer::{ReplayBuffer, MdpTransition};

pub struct PPO {
    pub actor: NeuralNetwork,
    pub critic: NeuralNetwork,
    pub actor_optimizer: Box<dyn Optimizer>,
    pub critic_optimizer: Box<dyn Optimizer>,
    pub gamma: f32,
    pub epsilon: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub memory: ReplayBuffer<MdpTransition>,
}

impl PPO {
    pub fn new(
        actor: NeuralNetwork,
        actor_optimizer: Box<dyn Optimizer>,
        critic: NeuralNetwork,
        critic_optimizer: Box<dyn Optimizer>,
        gamma: f32,
        epsilon: f32,
        epochs: usize,
        batch_size: usize,
        memory_size: usize,
    ) -> Self {
        PPO {
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            gamma,
            epsilon,
            epochs,
            batch_size,
            memory: ReplayBuffer::new(memory_size),
        }
    }

    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        self.actor.forward(state).clone()
    }

    pub fn train(&mut self, state: &Tensor, action: &Tensor, next_state: &Tensor, reward: f32, final_state: bool) {
        self.memory.add_item(MdpTransition {
            s0: state.clone(),
            a: action.clone(),
            s1: next_state.clone(),
            r: reward,
            final_state,
        });

        if self.memory.len() >= self.batch_size {
            self.update();
            self.memory.buffer.clear();
        }
    }

    fn update(&mut self) {
        let transitions = self.memory.buffer.clone();
        let n = transitions.len();

        let mut states = Vec::new();
        let mut rewards = Vec::new();
        let mut next_states = Vec::new();
        let mut masks = Vec::new();

        for t in &transitions {
            states.push(&t.s0);
            rewards.push(t.r);
            next_states.push(&t.s1);
            masks.push(if t.final_state { 0.0 } else { 1.0 });
        }

        let batch_state = Tensor::concat(&states, 0);

        let old_probs = self.actor.forward(&batch_state).clone();

        for _ in 0..self.epochs {
            let v_values = self.critic.forward(&batch_state).clone();
            let v_next_values = self.critic.forward(&Tensor::concat(&next_states, 0)).clone();
            
            let mut targets = Tensor::with_shape_val(vec![n, 1], 0.0);
            let mut advantages = Tensor::with_shape_val(vec![n, 1], 0.0);
            for i in 0..n {
                let target = rewards[i] + self.gamma * masks[i] * v_next_values.get(vec![i, 0]);
                targets.set(vec![i, 0], target);
                advantages.set(vec![i, 0], target - v_values.get(vec![i, 0]));
            }

            let mut critic_delta = Tensor::with_shape_val(targets.shape.clone(), 0.0);
            for i in 0..n {
                critic_delta.set(vec![i, 0], (v_values.get(vec![i, 0]) - targets.get(vec![i, 0])) / n as f32);
            }
            self.critic.backward(&mut critic_delta);
            self.critic_optimizer.update(&mut self.critic);

            let new_probs = self.actor.forward(&batch_state).clone();
            let mut actor_delta = Tensor::with_shape_val(new_probs.shape.clone(), 0.0);
            
            for i in 0..n {
                let act_idx = transitions[i].a.max_index(0)[0];
                let prob = new_probs.get(vec![i, act_idx]);
                let old_prob = old_probs.get(vec![i, act_idx]);
                let ratio = prob / (old_prob + 1e-8);
                let adv = advantages.get(vec![i, 0]);

                let surr1 = ratio * adv;
                let surr2 = ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * adv;
                
                let d_ratio = 1.0 / (old_prob + 1e-8);
                let grad = if surr1 < surr2 {
                    adv * d_ratio
                } else if ratio > 1.0 + self.epsilon || ratio < 1.0 - self.epsilon {
                    0.0
                } else {
                    adv * d_ratio
                };

                actor_delta.set(vec![i, act_idx], -grad / n as f32);
            }
            
            self.actor.backward(&mut actor_delta);
            self.actor_optimizer.update(&mut self.actor);
        }
    }
}
