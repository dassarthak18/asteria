use crate::core::neural_network::NeuralNetwork;

/// Gradient-based parameter update rule applied after each backward pass.
///
/// All built-in optimizers ([`Adam`](super::adam::Adam), [`ADOPT`](super::adopt::ADOPT),
/// [`RAdam`](super::radam::RAdam), [`Sgd`](super::sgd::Sgd)) implement this trait so they can
/// be stored as `Box<dyn Optimizer>` and swapped without changing the training loop.
/// Use [`set_lr`](Optimizer::set_lr) together with an [`LrScheduler`](super::lr_scheduler::LrScheduler)
/// to apply a learning-rate schedule each epoch.
pub trait Optimizer: Send {
    /// Reads accumulated gradients from `network` and updates its parameters in-place.
    fn update(&mut self, network: &mut NeuralNetwork);
    /// Overrides the current learning rate (used by LR schedulers).
    fn set_lr(&mut self, lr: f32);
    /// Returns the current learning rate.
    fn get_lr(&self) -> f32;
}
