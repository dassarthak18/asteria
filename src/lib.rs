//! Asteria — a pure-Rust reinforcement learning and neural network library.
//!
//! Three top-level modules cover the full stack:
//! - [`clab`]: tensor engine, activation/loss functions, weight initializers, and noise generators.
//! - [`core`]: graph-based neural networks, dense layers, parameter management, optimizers
//!   (Adam, RAdam, ADOPT, SGD), and LR schedulers (StepDecay, CosineAnnealing, LinearDecay,
//!   WarmupCosine, ReduceLROnPlateau).
//! - [`rl`]: environment trait, exploration strategies, and RL algorithms (Q-Learning, SARSA,
//!   DQN, AC, A2C, A3C, CACLA, DDPG, PPO, QAC, TD, ForwardModel, Metacritic).

#![forbid(unsafe_code)]

pub mod clab;
pub mod core;
pub mod rl;
