use crate::clab::tensor::Tensor;

/// Environment interface for reinforcement learning agents.
///
/// Implementors model a Markov Decision Process (MDP): they hold mutable internal state,
/// expose the current observation via [`get_state`](IEnvironment::get_state), apply
/// actions via [`do_action`](IEnvironment::do_action), and report the resulting scalar
/// reward and termination flag.
///
/// All RL agents in this library interact with environments exclusively through this trait,
/// so the same agent code runs against any environment that implements it.
pub trait IEnvironment {
    /// Returns a tensor encoding of the current environment state.
    fn get_state(&mut self) -> Tensor;
    /// Applies `action` to the environment, advancing its internal state by one step.
    fn do_action(&mut self, action: &Tensor);
    /// Returns the reward earned by the most recent [`do_action`](IEnvironment::do_action) call.
    fn get_reward(&mut self) -> f32;
    /// Resets the environment to a fresh initial state, ready for a new episode.
    fn reset(&mut self);
    /// Returns `true` when the episode has reached a terminal condition.
    fn is_finished(&mut self) -> bool;
    /// Dimensionality of the state observation (number of features).
    fn state_dim(&self) -> usize;
    /// Dimensionality of the action space (number of discrete actions or continuous control dims).
    fn action_dim(&self) -> usize;
}
