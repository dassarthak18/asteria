use crate::clab::Interpolation;

/// Exponential interpolation between two points over a fixed interval.
///
/// Follows the form $f(t) = p_1 \cdot (p_2/p_1)^{t/T}$, where $T$ is the interval.
pub struct ExponentialInterpolation {
    /// Starting value.
    pub point1: f32,
    /// Target value reached at `t = interval`.
    pub point2: f32,
    /// Duration of the transition.
    pub interval: i32,
}

impl ExponentialInterpolation {
    /// Creates a new exponential interpolation. If points are zero, a small epsilon is added
    /// to avoid division by zero.
    pub fn new(mut point1: f32, mut point2: f32, interval: i32) -> Self {
        if point1 == 0.0 {
            point1 += 1e-8;
        }
        if point2 == 0.0 {
            point2 += 1e-8;
        }
        ExponentialInterpolation {
            point1,
            point2,
            interval,
        }
    }
}

impl Interpolation for ExponentialInterpolation {
    fn interpolate(&self, t: i32) -> f32 {
        let mut f_t = 1.0;

        if t <= self.interval {
            f_t = t as f32 / self.interval as f32;
        }

        self.point1 * (self.point2 / self.point1).powf(f_t)
    }
}
