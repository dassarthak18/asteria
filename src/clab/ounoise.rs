use crate::clab::tensor::Tensor;
use crate::clab::random_generator::RandomGenerator;

/// Ornstein-Uhlenbeck process for temporally correlated noise.
///
/// Models the stochastic differential equation $dX_t = \theta(\mu - X_t)dt + \sigma dW_t$.
/// Used for continuous action exploration in DDPG to encourage smooth, directed trajectories.
pub struct OUNoise {
    dim: usize,
    mu: f32,
    theta: f32,
    sigma: f32,
    dt: f32,
    state: Tensor,
}

impl OUNoise {
    /// Creates a new OU process.
    ///
    /// - `dim`: Dimensionality of the action space.
    /// - `mu`: Long-term mean (usually 0.0).
    /// - `sigma`: Volatility / noise scale.
    /// - `theta`: Rate of mean reversion.
    /// - `dt`: Timestep size (e.g. 0.01).
    pub fn new(dim: usize, mu: f32, sigma: f32, theta: f32, dt: f32) -> Self {
        let state = Tensor::value(vec![dim], mu);
        OUNoise {
            dim,
            mu,
            theta,
            sigma,
            dt,
            state,
        }
    }

    /// Resets the process state to `mu`.
    pub fn reset(&mut self) {
        self.state = Tensor::value(vec![self.dim], self.mu);
    }

    /// Generates a noise vector and adds it in-place to the `action` tensor.
    pub fn noise(&mut self, action: &mut Tensor) {
        let sqrt_dt = self.dt.sqrt();
        let rg = RandomGenerator::instance();
        for i in 0..self.dim {
            self.state[i] += self.theta * (self.mu - self.state[i]) * self.dt + self.sigma * rg.normal_random(0.0, 1.0) * sqrt_dt;
            action[i] += self.state[i];
        }
    }

    /// Updates the volatility parameter.
    pub fn set_sigma(&mut self, sigma: f32) {
        self.sigma = sigma;
    }
}
