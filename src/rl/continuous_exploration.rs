use crate::clab::tensor::Tensor;
use crate::clab::random_generator::RandomGenerator;
use crate::clab::Interpolation;

pub enum ContinuousExplorationMethod {
    Gaussian,
    OuNoise,
}

pub struct OuNoise {
    dim: usize,
    mu: f32,
    theta: f32,
    sigma: f32,
    dt: f32,
    state: Tensor,
}

impl OuNoise {
    pub fn new(dim: usize, mu: f32, sigma: f32, theta: f32, dt: f32) -> Self {
        OuNoise {
            dim,
            mu,
            theta,
            sigma,
            dt,
            state: Tensor::value(vec![dim], mu),
        }
    }

    pub fn reset(&mut self) {
        self.state = Tensor::value(vec![self.dim], self.mu);
    }

    pub fn noise(&mut self, action: &mut Tensor) {
        for i in 0..self.dim {
            let noise = self.theta * (self.mu - self.state.get(vec![i])) * self.dt 
                        + self.sigma * RandomGenerator::instance().normal_random(0.0, 1.0) * self.dt.sqrt();
            let new_state = self.state.get(vec![i]) + noise;
            self.state.set(vec![i], new_state);
            action.set(vec![i], action.get(vec![i]) + new_state);
        }
    }

    pub fn set_sigma(&mut self, sigma: f32) {
        self.sigma = sigma;
    }
}

pub struct ContinuousExploration {
    method: Option<ContinuousExplorationMethod>,
    interpolation: Option<Box<dyn Interpolation>>,
    ou_noise: Option<OuNoise>,
    sigma: f32,
}

impl ContinuousExploration {
    pub fn new(interpolation: Option<Box<dyn Interpolation>>) -> Self {
        ContinuousExploration {
            method: None,
            interpolation,
            ou_noise: None,
            sigma: 0.0,
        }
    }

    pub fn init_gaussian(&mut self, sigma: f32) {
        self.method = Some(ContinuousExplorationMethod::Gaussian);
        self.sigma = sigma;
    }

    pub fn init_ou_noise(&mut self, dim: usize, mu: f32, sigma: f32, theta: f32) {
        self.method = Some(ContinuousExplorationMethod::OuNoise);
        self.sigma = sigma;
        self.ou_noise = Some(OuNoise::new(dim, mu, sigma, theta, 0.01));
    }

    pub fn explore(&mut self, action: &Tensor) -> Tensor {
        let mut output = action.clone();
        match self.method {
            Some(ContinuousExplorationMethod::Gaussian) => self.explore_gaussian(&mut output),
            Some(ContinuousExplorationMethod::OuNoise) => self.explore_ou_noise(&mut output),
            None => {}
        }
        output
    }

    pub fn update(&mut self, timestep: i32) {
        if let Some(ref interpolation) = self.interpolation {
            self.sigma = interpolation.interpolate(timestep);
        }

        if let Some(ref mut ou_noise) = self.ou_noise {
            ou_noise.set_sigma(self.sigma);
        }
    }

    pub fn reset(&mut self) {
        if let Some(ref mut ou_noise) = self.ou_noise {
            ou_noise.reset();
        }
    }

    fn explore_gaussian(&self, action: &mut Tensor) {
        for i in 0..action.size() {
            let rand = if self.sigma > 0.0 {
                RandomGenerator::instance().normal_random(0.0, self.sigma)
            } else {
                0.0
            };
            action.set(vec![i], action.get(vec![i]) + rand);
        }
    }

    fn explore_ou_noise(&mut self, action: &mut Tensor) {
        if let Some(ref mut ou_noise) = self.ou_noise {
            ou_noise.noise(action);
        }
    }
}
