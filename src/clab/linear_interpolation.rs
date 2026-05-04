use crate::clab::Interpolation;

/// Linear interpolation between two points over a fixed interval.
///
/// Follows the form $f(t) = (1 - \frac{t}{T})p_1 + \frac{t}{T}p_2$, where $T$ is the interval.
pub struct LinearInterpolation {
    /// Starting value.
    pub point1: f32,
    /// Target value reached at `t = interval`.
    pub point2: f32,
    /// Duration of the transition.
    pub interval: i32,
}

impl LinearInterpolation {
    /// Creates a new linear interpolation.
    pub fn new(point1: f32, point2: f32, interval: i32) -> Self {
        LinearInterpolation {
            point1,
            point2,
            interval,
        }
    }
}

impl Interpolation for LinearInterpolation {
    fn interpolate(&self, t: i32) -> f32 {
        let mut f_t = 1.0;

        if t <= self.interval {
            f_t = t as f32 / self.interval as f32;
        }

        (1.0 - f_t) * self.point1 + f_t * self.point2
    }
}
