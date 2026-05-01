use crate::clab::tensor::Tensor;
use crate::clab::random_generator::RandomGenerator;
use crate::clab::Interpolation;

/// Strategy used by [`ContinuousExploration`] to perturb continuous actions.
pub enum ContinuousExplorationMethod {
    /// Adds i.i.d. Gaussian noise N(0, σ²) to each action dimension.
    Gaussian,
    /// Adds correlated Ornstein-Uhlenbeck noise for temporally-smooth exploration.
    OuNoise,
}

/// Ornstein-Uhlenbeck process for temporally-correlated exploration noise.
///
/// Models a mean-reverting stochastic process:
///
/// ```text
/// dX_t = θ · (μ − X_t) · dt + σ · √dt · N(0, 1)
/// ```
///
/// Produces smoother action sequences than i.i.d. Gaussian noise, which is
/// beneficial for physical control tasks where abrupt action changes are penalised.
pub struct OuNoise {
    dim: usize,
    mu: f32,
    theta: f32,
    sigma: f32,
    dt: f32,
    state: Tensor,
}

impl OuNoise {
    /// Creates a new OU-noise process.
    ///
    /// - `dim`: number of action dimensions.
    /// - `mu`: long-run mean (typically 0.0).
    /// - `sigma`: noise magnitude.
    /// - `theta`: mean-reversion rate (higher ⇒ faster reversion).
    /// - `dt`: time step size (typically 0.01).
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

    /// Resets the noise state to the long-run mean `μ` (call at the start of each episode).
    pub fn reset(&mut self) {
        self.state = Tensor::value(vec![self.dim], self.mu);
    }

    /// Advances the OU process and adds the resulting noise to `action` in place.
    pub fn noise(&mut self, action: &mut Tensor) {
        for i in 0..self.dim {
            let noise = self.theta * (self.mu - self.state.get(vec![i])) * self.dt
                        + self.sigma * RandomGenerator::instance().normal_random(0.0, 1.0) * self.dt.sqrt();
            let new_state = self.state.get(vec![i]) + noise;
            self.state.set(vec![i], new_state);
            action.set(vec![i], action.get(vec![i]) + new_state);
        }
    }

    /// Updates the noise magnitude σ (used by [`ContinuousExploration::update`] for decay).
    pub fn set_sigma(&mut self, sigma: f32) {
        self.sigma = sigma;
    }
}

/// Exploration wrapper for continuous action spaces.
///
/// Supports two noise strategies (Gaussian and OU) with optional magnitude decay via an
/// [`Interpolation`] schedule. Initialise with one of [`init_gaussian`](ContinuousExploration::init_gaussian)
/// or [`init_ou_noise`](ContinuousExploration::init_ou_noise) before calling [`explore`](ContinuousExploration::explore).
pub struct ContinuousExploration {
    method: Option<ContinuousExplorationMethod>,
    interpolation: Option<Box<dyn Interpolation>>,
    ou_noise: Option<OuNoise>,
    sigma: f32,
}

impl ContinuousExploration {
    /// Creates a new continuous exploration wrapper with no method selected.
    ///
    /// - `interpolation`: optional schedule to decay the noise magnitude over time.
    pub fn new(interpolation: Option<Box<dyn Interpolation>>) -> Self {
        ContinuousExploration {
            method: None,
            interpolation,
            ou_noise: None,
            sigma: 0.0,
        }
    }

    /// Selects Gaussian noise with standard deviation `sigma`.
    pub fn init_gaussian(&mut self, sigma: f32) {
        self.method = Some(ContinuousExplorationMethod::Gaussian);
        self.sigma = sigma;
    }

    /// Selects Ornstein-Uhlenbeck noise.
    ///
    /// - `dim`: number of action dimensions.
    /// - `mu`: long-run mean of the OU process.
    /// - `sigma`: initial noise magnitude.
    /// - `theta`: mean-reversion rate.
    pub fn init_ou_noise(&mut self, dim: usize, mu: f32, sigma: f32, theta: f32) {
        self.method = Some(ContinuousExplorationMethod::OuNoise);
        self.sigma = sigma;
        self.ou_noise = Some(OuNoise::new(dim, mu, sigma, theta, 0.01));
    }

    /// Returns a clone of `action` perturbed by the selected noise strategy.
    pub fn explore(&mut self, action: &Tensor) -> Tensor {
        let mut output = action.clone();
        match self.method {
            Some(ContinuousExplorationMethod::Gaussian) => self.explore_gaussian(&mut output),
            Some(ContinuousExplorationMethod::OuNoise) => self.explore_ou_noise(&mut output),
            None => {}
        }
        output
    }

    /// Advances the noise decay schedule and updates the OU process sigma if applicable.
    pub fn update(&mut self, timestep: i32) {
        if let Some(ref interpolation) = self.interpolation {
            self.sigma = interpolation.interpolate(timestep);
        }

        if let Some(ref mut ou_noise) = self.ou_noise {
            ou_noise.set_sigma(self.sigma);
        }
    }

    /// Resets OU noise state (call at the start of each episode when using OU noise).
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
