//! Graph-based neural network construction, parameter management, optimizers, and LR schedulers.
//!
//! The main entry points are [`neural_network::NeuralNetwork`] and [`dense_layer::DenseLayer`].
//! Optimizers ([`adam::Adam`], [`adopt::ADOPT`], [`radam::RAdam`], [`sgd::Sgd`]) all implement
//! [`optimizer::Optimizer`] and are interchangeable at runtime.
//!
//! Learning-rate schedules live in [`lr_scheduler`]: [`lr_scheduler::StepDecay`],
//! [`lr_scheduler::CosineAnnealing`], [`lr_scheduler::LinearDecay`],
//! [`lr_scheduler::WarmupCosine`], and [`lr_scheduler::ReduceLROnPlateau`].

pub mod neural_network;
pub mod dense_layer;
pub mod optimizer;
pub mod adam;
pub mod adopt;
pub mod radam;
pub mod sgd;
pub mod lr_scheduler;
pub mod variable;
pub mod param;
pub mod param_model;
pub mod linear_operator;
