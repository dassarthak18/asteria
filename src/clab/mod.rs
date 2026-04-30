//! Tensor engine and mathematical utilities.
//!
//! Provides the core [`tensor::Tensor`] type along with activation functions, loss functions,
//! weight initializers, random number generation, and interpolation helpers used throughout
//! the rest of the library.

pub mod tensor;
pub mod tensor_operator;
pub mod activation_functions;
pub mod loss_functions;
pub mod random_generator;
pub mod tensor_initializer;
pub mod exponential_interpolation;
pub mod linear_interpolation;
pub mod ounoise;

/// Scheduler-driven scalar interpolation, used to anneal exploration parameters over time.
pub trait Interpolation {
    /// Returns the interpolated value at timestep `t`.
    fn interpolate(&self, t: i32) -> f32;
}
