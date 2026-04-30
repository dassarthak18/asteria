use crate::clab::Interpolation;

pub struct ExponentialInterpolation {
    pub point1: f32,
    pub point2: f32,
    pub interval: i32,
}

impl ExponentialInterpolation {
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
