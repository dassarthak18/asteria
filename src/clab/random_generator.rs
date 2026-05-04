use rand::prelude::*;
use rand_distr::{Normal, Exp, Distribution};

/// Thread-local random number generator with support for common distributions.
pub struct RandomGenerator;

impl RandomGenerator {
    /// Returns a new instance of the random generator.
    pub fn instance() -> Self {
        RandomGenerator
    }

    /// Returns a random float in the range `[lower, upper)`.
    pub fn random_float(&self, lower: f32, upper: f32) -> f32 {
        let mut rng = rand::rng();
        rng.random_range(lower..upper)
    }

    /// Returns a random float in the range `[0, 1)`.
    pub fn random(&self) -> f32 {
        let mut rng = rand::rng();
        rng.random()
    }

    /// Returns a random integer in the range `[lower, upper]`.
    pub fn random_int(&self, lower: i32, upper: i32) -> i32 {
        let mut rng = rand::rng();
        rng.random_range(lower..=upper)
    }

    /// Alias for [`random_int`](Self::random_int).
    pub fn random_range(&self, lower: i32, upper: i32) -> i32 {
        self.random_int(lower, upper)
    }

    /// Samples from a Normal distribution $\mathcal{N}(\mu, \sigma)$.
    pub fn normal_random(&self, mean: f32, sigma: f32) -> f32 {
        let mut rng = rand::rng();
        let normal = Normal::new(mean, sigma).unwrap();
        normal.sample(&mut rng)
    }

    /// Samples from an Exponential distribution $\text{Exp}(\lambda)$.
    pub fn exp_random(&self, lambda: f32) -> f32 {
        let mut rng = rand::rng();
        let exp = Exp::new(lambda).unwrap();
        exp.sample(&mut rng)
    }

    /// Weighted random choice from a slice of probabilities; returns the selected index.
    pub fn choice(&self, prob: &[f32]) -> usize {
        let mut rng = rand::rng();
        let r: f32 = rng.random();
        let mut p = 0.0;
        for (i, &prob_val) in prob.iter().enumerate() {
            if r >= p && r < p + prob_val {
                return i;
            }
            p += prob_val;
        }
        prob.len() - 1
    }
}
