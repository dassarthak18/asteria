use rand::prelude::*;
use rand_distr::{Normal, Exp, Distribution};

pub struct RandomGenerator;

impl RandomGenerator {
    pub fn instance() -> Self {
        RandomGenerator
    }

    pub fn random_float(&self, lower: f32, upper: f32) -> f32 {
        let mut rng = rand::rng();
        rng.random_range(lower..upper)
    }

    pub fn random(&self) -> f32 {
        let mut rng = rand::rng();
        rng.random()
    }

    pub fn random_int(&self, lower: i32, upper: i32) -> i32 {
        let mut rng = rand::rng();
        rng.random_range(lower..=upper)
    }

    pub fn random_range(&self, lower: i32, upper: i32) -> i32 {
        self.random_int(lower, upper)
    }

    pub fn normal_random(&self, mean: f32, sigma: f32) -> f32 {
        let mut rng = rand::rng();
        let normal = Normal::new(mean, sigma).unwrap();
        normal.sample(&mut rng)
    }

    pub fn exp_random(&self, lambda: f32) -> f32 {
        let mut rng = rand::rng();
        let exp = Exp::new(lambda).unwrap();
        exp.sample(&mut rng)
    }

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
