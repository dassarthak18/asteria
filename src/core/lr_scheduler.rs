/// Adjusts a learning rate over time according to a fixed rule.
///
/// Call [`step`](LrScheduler::step) once per epoch (or per mini-batch, depending on the
/// scheduler) and pass the returned value to [`Optimizer::set_lr`](super::optimizer::Optimizer::set_lr).
///
/// # Example
/// ```rust,ignore
/// let mut sched = CosineAnnealing::new(1e-3, 1e-5, epochs);
/// for epoch in 0..epochs {
///     // … training loop …
///     let lr = sched.step();
///     opt.set_lr(lr);
/// }
/// ```
pub trait LrScheduler {
    /// Advances the internal step counter and returns the new learning rate.
    fn step(&mut self) -> f32;
    /// Returns the current learning rate without advancing the counter.
    fn current_lr(&self) -> f32;
}

// ── Step Decay ────────────────────────────────────────────────────────────────

/// Multiplies the learning rate by `gamma` every `step_size` calls to [`step`](LrScheduler::step).
///
/// ```text
/// lr = lr_initial * gamma ^ floor(epoch / step_size)
/// ```
pub struct StepDecay {
    lr: f32,
    initial_lr: f32,
    gamma: f32,
    step_size: usize,
    epoch: usize,
}

impl StepDecay {
    /// Creates a new step-decay scheduler.
    ///
    /// - `initial_lr`: learning rate at epoch 0.
    /// - `gamma`: multiplicative factor applied every `step_size` epochs (e.g. `0.1`).
    /// - `step_size`: number of epochs between each decay.
    pub fn new(initial_lr: f32, gamma: f32, step_size: usize) -> Self {
        StepDecay { lr: initial_lr, initial_lr, gamma, step_size, epoch: 0 }
    }
}

impl LrScheduler for StepDecay {
    fn step(&mut self) -> f32 {
        self.epoch += 1;
        let factor = self.gamma.powi((self.epoch / self.step_size) as i32);
        self.lr = self.initial_lr * factor;
        self.lr
    }
    fn current_lr(&self) -> f32 { self.lr }
}

// ── Cosine Annealing ──────────────────────────────────────────────────────────

/// Decays the learning rate following a half-cosine curve from `lr_max` to `lr_min`.
///
/// ```text
/// lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))
/// ```
///
/// At `t = 0` the LR equals `lr_max`; at `t = T_max` it equals `lr_min`.
pub struct CosineAnnealing {
    lr_max: f32,
    lr_min: f32,
    t_max: usize,
    t: usize,
    lr: f32,
}

impl CosineAnnealing {
    /// Creates a cosine annealing scheduler.
    ///
    /// - `lr_max`: learning rate at the start (t = 0).
    /// - `lr_min`: learning rate at `t = t_max` (floor value; typically 0 or 1e-5).
    /// - `t_max`: total number of steps (usually total epochs).
    pub fn new(lr_max: f32, lr_min: f32, t_max: usize) -> Self {
        CosineAnnealing { lr_max, lr_min, t_max, t: 0, lr: lr_max }
    }
}

impl LrScheduler for CosineAnnealing {
    fn step(&mut self) -> f32 {
        let t = self.t.min(self.t_max) as f32;
        self.lr = self.lr_min
            + 0.5 * (self.lr_max - self.lr_min) * (1.0 + (std::f32::consts::PI * t / self.t_max as f32).cos());
        self.t += 1;
        self.lr
    }
    fn current_lr(&self) -> f32 { self.lr }
}

// ── Linear Decay ─────────────────────────────────────────────────────────────

/// Decays the learning rate linearly from `lr_start` to `lr_end` over `total_steps`.
///
/// Stays at `lr_end` for any steps beyond `total_steps`.
pub struct LinearDecay {
    lr_start: f32,
    lr_end: f32,
    total_steps: usize,
    t: usize,
    lr: f32,
}

impl LinearDecay {
    /// Creates a linear decay scheduler.
    ///
    /// - `lr_start`: learning rate at step 0.
    /// - `lr_end`: learning rate at `total_steps` (and beyond).
    /// - `total_steps`: number of steps over which to interpolate.
    pub fn new(lr_start: f32, lr_end: f32, total_steps: usize) -> Self {
        LinearDecay { lr_start, lr_end, total_steps, t: 0, lr: lr_start }
    }
}

impl LrScheduler for LinearDecay {
    fn step(&mut self) -> f32 {
        let frac = (self.t as f32 / self.total_steps as f32).min(1.0);
        self.lr = self.lr_start + (self.lr_end - self.lr_start) * frac;
        self.t += 1;
        self.lr
    }
    fn current_lr(&self) -> f32 { self.lr }
}

// ── Warmup + Cosine ───────────────────────────────────────────────────────────

/// Linear warm-up for `warmup_steps`, then cosine annealing to `lr_min`.
///
/// Useful for Adam/ADOPT where the second-moment estimate is unreliable in the first steps.
/// During warm-up the LR rises linearly from `0` to `lr_max`; afterwards it follows the
/// half-cosine curve down to `lr_min`.
pub struct WarmupCosine {
    lr_max: f32,
    lr_min: f32,
    warmup_steps: usize,
    total_steps: usize,
    t: usize,
    lr: f32,
}

impl WarmupCosine {
    /// Creates a warm-up + cosine schedule.
    ///
    /// - `lr_max`: peak learning rate reached at the end of warm-up.
    /// - `lr_min`: final learning rate at `total_steps`.
    /// - `warmup_steps`: number of steps for the linear ramp-up phase.
    /// - `total_steps`: total steps (warm-up + cosine decay).
    pub fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self {
        WarmupCosine { lr_max, lr_min, warmup_steps, total_steps, t: 0, lr: 0.0 }
    }
}

impl LrScheduler for WarmupCosine {
    fn step(&mut self) -> f32 {
        self.lr = if self.t < self.warmup_steps {
            self.lr_max * (self.t as f32 + 1.0) / self.warmup_steps as f32
        } else {
            let decay_steps = (self.total_steps - self.warmup_steps).max(1);
            let t_decay = (self.t - self.warmup_steps).min(decay_steps) as f32;
            self.lr_min
                + 0.5 * (self.lr_max - self.lr_min)
                * (1.0 + (std::f32::consts::PI * t_decay / decay_steps as f32).cos())
        };
        self.t += 1;
        self.lr
    }
    fn current_lr(&self) -> f32 { self.lr }
}

// ── Reduce LR on Plateau ─────────────────────────────────────────────────────

/// Reduces the learning rate by `factor` when a tracked metric stops improving.
///
/// Unlike the other schedulers this one is *metric-driven*: call [`step`](ReduceLROnPlateau::step)
/// with the current validation loss (or any scalar — lower is assumed to be better).
/// If the metric has not improved for `patience` consecutive calls, the LR is multiplied by
/// `factor` (clamped to `min_lr`).
pub struct ReduceLROnPlateau {
    lr: f32,
    factor: f32,
    patience: usize,
    min_lr: f32,
    best: f32,
    wait: usize,
}

impl ReduceLROnPlateau {
    /// Creates a plateau-based scheduler.
    ///
    /// - `initial_lr`: starting learning rate.
    /// - `factor`: multiplicative reduction factor (e.g. `0.5` to halve).
    /// - `patience`: number of non-improving steps before reducing.
    /// - `min_lr`: lower bound on the learning rate.
    pub fn new(initial_lr: f32, factor: f32, patience: usize, min_lr: f32) -> Self {
        ReduceLROnPlateau { lr: initial_lr, factor, patience, min_lr, best: f32::INFINITY, wait: 0 }
    }

    /// Advances the scheduler given the current `metric` value (lower = better).
    ///
    /// Returns the (possibly reduced) learning rate.
    pub fn step(&mut self, metric: f32) -> f32 {
        if metric < self.best {
            self.best = metric;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.lr = (self.lr * self.factor).max(self.min_lr);
                self.wait = 0;
            }
        }
        self.lr
    }

    /// Returns the current learning rate without advancing.
    pub fn current_lr(&self) -> f32 { self.lr }
}
