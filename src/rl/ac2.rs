use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::replay_buffer::{ReplayBuffer, MdpTransition};

/// Advantage Actor-Critic with mini-batch updates (A2C).
///
/// `AC2` is the batched variant of [`AC`](crate::rl::ac::AC). Rather than updating after
/// every transition, it accumulates `batch_size` transitions in a local buffer and then
/// performs a single vectorised actor-critic update:
///
/// - **Critic**: minimises `(V(s) − target)²` where `target = r + γ·V(s')`.
/// - **Actor**: maximises `Σ A(s,a) · log π(a|s)` where advantage `A = target − V(s)`.
///
/// A2C is synchronous (single-threaded). For the asynchronous multi-worker variant
/// see [`AC3`](crate::rl::ac3::AC3).
pub struct AC2 {
    /// Actor network π(a|s); output should be a probability distribution (Softmax activation).
    pub actor: NeuralNetwork,
    /// Value critic network V(s); output should be a single scalar (Linear activation).
    pub critic: NeuralNetwork,
    /// Optimizer used to update the actor.
    pub actor_optimizer: Box<dyn Optimizer>,
    /// Optimizer used to update the critic.
    pub critic_optimizer: Box<dyn Optimizer>,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// Number of transitions to accumulate before each update.
    pub batch_size: usize,
    /// Local rollout buffer; cleared after each update.
    pub memory: ReplayBuffer<MdpTransition>,
}

impl AC2 {
    /// Creates a new A2C agent.
    ///
    /// - `actor` / `actor_optimizer`: policy network and its update rule.
    /// - `critic` / `critic_optimizer`: value network and its update rule.
    /// - `gamma`: discount factor.
    /// - `batch_size`: rollout length before each update step.
    pub fn new(
        actor: NeuralNetwork,
        actor_optimizer: Box<dyn Optimizer>,
        critic: NeuralNetwork,
        critic_optimizer: Box<dyn Optimizer>,
        gamma: f32,
        batch_size: usize,
    ) -> Self {
        AC2 {
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            gamma,
            batch_size,
            memory: ReplayBuffer::new(batch_size * 2),
        }
    }

    /// Returns the actor's action probability distribution for `state`.
    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        self.actor.forward(state).clone()
    }

    /// Adds the transition to the rollout buffer and triggers an update when `batch_size` is reached.
    ///
    /// The buffer is cleared after each update, so the agent always trains on fresh on-policy data.
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
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut next_states = Vec::new();
        let mut masks = Vec::new();

        for t in &transitions {
            states.push(&t.s0);
            actions.push(&t.a);
            rewards.push(t.r);
            next_states.push(&t.s1);
            masks.push(if t.final_state { 0.0 } else { 1.0 });
        }

        let batch_state = Tensor::concat(&states, 0);

        // next_states forward first so that batch_state is the last forward,
        // keeping the correct cached inputs for the critic backward pass.
        let v_next_values = self.critic.forward(&Tensor::concat(&next_states, 0)).clone();
        let v_values = self.critic.forward(&batch_state).clone();

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

        let probs = self.actor.forward(&batch_state).clone();
        let mut actor_delta = Tensor::with_shape_val(probs.shape.clone(), 0.0);

        for i in 0..n {
            let act_idx = transitions[i].a.max_index(0)[0];
            let prob = probs.get(vec![i, act_idx]);
            let adv = advantages.get(vec![i, 0]);

            actor_delta.set(vec![i, act_idx], -adv / (prob + 1e-8) / n as f32);
        }

        self.actor.backward(&mut actor_delta);
        self.actor_optimizer.update(&mut self.actor);
    }
}
