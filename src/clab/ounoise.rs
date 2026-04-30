use crate::clab::tensor::Tensor;
use crate::clab::random_generator::RandomGenerator;

pub struct OUNoise {
    dim: usize,
    mu: f32,
    theta: f32,
    sigma: f32,
    dt: f32,
    state: Tensor,
}

impl OUNoise {
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

    pub fn reset(&mut self) {
        self.state = Tensor::value(vec![self.dim], self.mu);
    }

    pub fn noise(&mut self, action: &mut Tensor) {
        let sqrt_dt = self.dt.sqrt();
        let rg = RandomGenerator::instance();
        for i in 0..self.dim {
            self.state[i] += self.theta * (self.mu - self.state[i]) * self.dt + self.sigma * rg.normal_random(0.0, 1.0) * sqrt_dt;
            action[i] += self.state[i];
        }
    }

    pub fn set_sigma(&mut self, sigma: f32) {
        self.sigma = sigma;
    }
}
