use crate::clab::tensor::{Tensor, Init};
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::replay_buffer::{ReplayBuffer, MdpTransition};

/// Deep Deterministic Policy Gradient (DDPG).
///
/// DDPG is an off-policy actor-critic algorithm for continuous action spaces. It combines
/// experience replay (as in DQN) with a deterministic policy and soft target-network
/// updates:
///
/// - **Critic** Q(s, a) is trained with the Bellman target
///   `r + γ · Q_target(s', μ_target(s'))`.
/// - **Actor** μ(s) is trained by ascending the critic's action-value gradient
///   `∇_a Q(s, a)|_{a=μ(s)}`, propagated back through the actor via the chain rule.
/// - **Target networks** `μ_target` and `Q_target` are updated each step with
///   soft (Polyak) averaging: `θ_target ← τ·θ + (1−τ)·θ_target`.
pub struct DDPG {
    /// Online actor network μ(s).
    pub actor: NeuralNetwork,
    /// Soft-averaged target actor μ_target(s).
    pub actor_target: NeuralNetwork,
    /// Online critic network Q(s, a).
    pub critic: NeuralNetwork,
    /// Soft-averaged target critic Q_target(s, a).
    pub critic_target: NeuralNetwork,
    /// Optimizer used to update the online actor.
    pub actor_optimizer: Box<dyn Optimizer>,
    /// Optimizer used to update the online critic.
    pub critic_optimizer: Box<dyn Optimizer>,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// Polyak averaging coefficient τ for soft target updates (typically 0.005).
    pub tau: f32,
    /// Experience replay buffer.
    pub memory: ReplayBuffer<MdpTransition>,
    /// Mini-batch size for each update.
    pub sample_size: usize,
    /// Dimensionality of the state observation.
    pub state_dim: usize,
    /// Dimensionality of the continuous action space.
    pub action_dim: usize,
    /// Batched state tensor assembled from the current replay sample.
    pub batch_state: Tensor,
    /// Batched action tensor assembled from the current replay sample.
    pub batch_action: Tensor,
    /// Batched next-state tensor assembled from the current replay sample.
    pub batch_next_state: Tensor,
    /// Batched reward tensor (shape `[sample_size, 1]`).
    pub batch_reward: Tensor,
    /// Batched termination mask (1.0 = non-terminal, 0.0 = terminal; shape `[sample_size, 1]`).
    pub batch_mask: Tensor,
}

impl DDPG {
    /// Creates a new DDPG agent.
    ///
    /// - `actor` / `actor_optimizer`: deterministic policy network and its update rule.
    /// - `critic` / `critic_optimizer`: Q-network (takes concatenated `[state, action]` input) and its update rule.
    /// - `gamma`: discount factor.
    /// - `memory_size`: replay buffer capacity.
    /// - `sample_size`: mini-batch size.
    /// - `tau`: Polyak coefficient for soft target updates.
    /// - `state_dim` / `action_dim`: used to split the gradient flowing back from the critic to the actor.
    pub fn new(
        actor: NeuralNetwork,
        actor_optimizer: Box<dyn Optimizer>,
        critic: NeuralNetwork,
        critic_optimizer: Box<dyn Optimizer>,
        gamma: f32,
        memory_size: usize,
        sample_size: usize,
        tau: f32,
        state_dim: usize,
        action_dim: usize,
    ) -> Self {
        let actor_target = actor.clone();
        let critic_target = critic.clone();
        DDPG {
            actor,
            actor_target,
            critic,
            critic_target,
            actor_optimizer,
            critic_optimizer,
            gamma,
            tau,
            memory: ReplayBuffer::new(memory_size),
            sample_size,
            state_dim,
            action_dim,
            batch_state: Tensor::new(),
            batch_action: Tensor::new(),
            batch_next_state: Tensor::new(),
            batch_reward: Tensor::new(),
            batch_mask: Tensor::new(),
        }
    }

    /// Returns the actor's continuous action for `state`.
    pub fn get_action(&mut self, state: &Tensor) -> &Tensor {
        self.actor.forward(state)
    }

    /// Stores the transition and, once the buffer has enough data, performs a DDPG update.
    ///
    /// Each update:
    /// 1. Draws a random mini-batch.
    /// 2. Updates the critic with the Bellman target.
    /// 3. Updates the actor by ascending the critic's action-value gradient.
    /// 4. Soft-updates both target networks.
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

            let mut critic_loss = self.critic_loss_function();
            self.critic.backward(&mut critic_loss);
            self.critic_optimizer.update(&mut self.critic);

            let mut actor_loss = self.actor_loss_function();
            self.actor.backward(&mut actor_loss);
            self.actor_optimizer.update(&mut self.actor);

            self.actor_target.copy_params(&self.actor, self.tau);
            self.critic_target.copy_params(&self.critic, self.tau);
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
        let next_actions = self.actor_target.forward(&self.batch_next_state).clone();
        let critic_target_input = Tensor::concat(&[&self.batch_next_state, &next_actions], 1);
        let next_q_values = self.critic_target.forward(&critic_target_input).clone();

        let critic_input = Tensor::concat(&[&self.batch_state, &self.batch_action], 1);
        let q_values = self.critic.forward(&critic_input).clone();

        let mut loss = Tensor::with_shape_val(q_values.shape.clone(), 0.0);
        for i in 0..self.sample_size {
            let target_q = self.batch_reward.get(vec![i, 0]) + self.gamma * self.batch_mask.get(vec![i, 0]) * next_q_values.get(vec![i, 0]);
            loss.set(vec![i, 0], (q_values.get(vec![i, 0]) - target_q) / self.sample_size as f32);
        }
        loss
    }

    fn actor_loss_function(&mut self) -> Tensor {
        let current_actions = self.actor.forward(&self.batch_state).clone();

        let critic_input = Tensor::concat(&[&self.batch_state, &current_actions], 1);
        self.critic.forward(&critic_input);

        let mut critic_grad = Tensor::with_shape_val(vec![self.sample_size, 1], -1.0 / self.sample_size as f32);
        self.critic.backward(&mut critic_grad);

        let input_id = self.critic.input_layers[0].clone();
        let input_delta = self.critic.layer_variables[&input_id].delta.clone();

        let n = self.sample_size;
        let total = self.state_dim + self.action_dim;
        let mut actor_grad = Tensor::with_shape(vec![n, self.action_dim], Init::Zero);
        for i in 0..n {
            for j in 0..self.action_dim {
                actor_grad.data[i * self.action_dim + j] =
                    input_delta.data[i * total + self.state_dim + j];
            }
        }
        actor_grad
    }
}
