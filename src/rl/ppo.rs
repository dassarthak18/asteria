use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::replay_buffer::{ReplayBuffer, MdpTransition};

/// Proximal Policy Optimisation (PPO) with clipped surrogate objective.
///
/// PPO improves on vanilla policy gradient by limiting how far the updated policy can
/// deviate from the behaviour policy that collected the data. The clipped objective is:
///
/// ```text
/// L_CLIP = E[ min(r·A, clip(r, 1−ε, 1+ε)·A) ]
/// ```
///
/// where `r = π_θ(a|s) / π_θ_old(a|s)` and `A` is the advantage estimate.
/// The critic (value function) is updated with a standard MSE loss over the same rollout.
///
/// The agent accumulates `batch_size` transitions per rollout, then runs `epochs`
/// optimisation passes over the batch before discarding it.
pub struct PPO {
    /// Actor network π(a|s); output should be probabilities (Softmax activation).
    pub actor: NeuralNetwork,
    /// Value critic network V(s); output should be a single scalar (Linear activation).
    pub critic: NeuralNetwork,
    /// Optimizer used to update the actor.
    pub actor_optimizer: Box<dyn Optimizer>,
    /// Optimizer used to update the critic.
    pub critic_optimizer: Box<dyn Optimizer>,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// PPO clip parameter ε; controls the trust region size.
    pub epsilon: f32,
    /// Number of gradient passes over each rollout batch.
    pub epochs: usize,
    /// Number of transitions per rollout.
    pub batch_size: usize,
    /// Local rollout buffer; cleared after each update.
    pub memory: ReplayBuffer<MdpTransition>,
}

impl PPO {
    /// Creates a new PPO agent.
    ///
    /// - `actor` / `actor_optimizer`: policy network and its update rule.
    /// - `critic` / `critic_optimizer`: value network and its update rule.
    /// - `gamma`: discount factor.
    /// - `epsilon`: clip parameter (typically 0.1 – 0.2).
    /// - `epochs`: gradient passes per rollout (typically 4 – 10).
    /// - `batch_size`: rollout length before each update.
    /// - `memory_size`: capacity of the rollout buffer (usually ≥ `batch_size`).
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

    /// Returns the actor's action probability distribution for `state`.
    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        self.actor.forward(state).clone()
    }

    /// Adds the transition to the rollout buffer and triggers a PPO update when `batch_size` is reached.
    ///
    /// Each update runs `epochs` gradient passes over the full rollout using the clipped
    /// surrogate objective. The buffer is cleared after each update.
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
