use crate::clab::tensor::Tensor;

/// Low-level mathematical operations on [`Tensor`] buffers.
///
/// Most operations support broadcasting over a leading batch dimension.
pub struct TensorOperator;

impl TensorOperator {
    /// Element-wise addition: `z = x + y`. Supports broadcasting.
    pub fn add(x: &Tensor, y: &Tensor, z: &mut Tensor) {
        if x.size == y.size && x.size == z.size {
            for i in 0..x.size {
                z.data[i] = x.data[i] + y.data[i];
            }
        } else if x.size < y.size {
            Self::add_broadcast_x(x, y, z);
        } else {
            Self::add_broadcast_y(x, y, z);
        }
    }

    /// Adds a scalar to every element: `z = x + y`.
    pub fn const_add(x: &Tensor, y: f32, z: &mut Tensor) {
        for i in 0..x.size {
            z.data[i] = x.data[i] + y;
        }
    }

    /// Element-wise subtraction: `z = x - y`. Supports broadcasting.
    pub fn sub(x: &Tensor, y: &Tensor, z: &mut Tensor) {
        if x.size == y.size && x.size == z.size {
            for i in 0..x.size {
                z.data[i] = x.data[i] - y.data[i];
            }
        } else if x.size < y.size {
            Self::sub_broadcast_x(x, y, z);
        } else {
            Self::sub_broadcast_y(x, y, z);
        }
    }

    /// Subtracts a scalar from every element: `z = x - y`.
    pub fn const_sub(x: &Tensor, y: f32, z: &mut Tensor) {
        for i in 0..x.size {
            z.data[i] = x.data[i] - y;
        }
    }

    /// Subtracts every element from a scalar: `z = x - y`.
    pub fn const_sub_lhs(x: f32, y: &Tensor, z: &mut Tensor) {
        for i in 0..y.size {
            z.data[i] = x - y.data[i];
        }
    }

    /// Matrix multiplication: `z = x * y`.
    ///
    /// Respects the `transpose_flag` of both `x` and `y` to perform $(AB,\ A^TB,\ AB^T,\ A^TB^T)$
    /// without physical buffer transpositions. `z` is automatically resized to the correct dimensions.
    pub fn mul(x: &Tensor, y: &Tensor, z: &mut Tensor) {
        let (rows, common, cols) = if !x.transpose_flag && !y.transpose_flag {
            (x.shape[0], x.shape[1], y.shape[1])
        } else if x.transpose_flag && !y.transpose_flag {
            (x.shape[1], x.shape[0], y.shape[1])
        } else if !x.transpose_flag && y.transpose_flag {
            (x.shape[0], x.shape[1], y.shape[0])
        } else {
            (x.shape[1], x.shape[0], y.shape[0])
        };

        z.resize(vec![rows, cols], crate::clab::tensor::Init::Zero);

        match (x.transpose_flag, y.transpose_flag) {
            (false, false) => Self::mul_ab(x, y, z, rows, common, cols),
            (true, false) => Self::mul_atb(x, y, z, rows, common, cols),
            (false, true) => Self::mul_abt(x, y, z, rows, common, cols),
            (true, true) => Self::mul_atbt(x, y, z, rows, common, cols),
        }
    }

    /// Multiplies every element by a scalar: `z = x * y`.
    pub fn const_mul(x: &Tensor, y: f32, z: &mut Tensor) {
        for i in 0..x.size {
            z.data[i] = x.data[i] * y;
        }
    }

    /// Divides a scalar by every element: `z = x / y`.
    pub fn const_div(x: f32, y: &Tensor, z: &mut Tensor) {
        for i in 0..y.size {
            z.data[i] = x / y.data[i];
        }
    }

    /// Sums elements along the batch axis (dimension 0).
    pub fn reduce_sum(x: &Tensor, y: &mut Tensor) {
        if x.shape[0] > 1 {
            let y_size = x.size / x.shape[0];
            y.resize(vec![1, y_size], crate::clab::tensor::Init::Zero);
            for i in 0..x.shape[0] {
                for j in 0..y_size {
                    y.data[j] += x.data[i * y_size + j];
                }
            }
        } else {
            y.data.copy_from_slice(&x.data);
        }
    }

    fn mul_ab(x: &Tensor, y: &Tensor, z: &mut Tensor, rows: usize, common: usize, cols: usize) {
        for i in 0..rows {
            for k in 0..common {
                let temp = x.data[i * common + k];
                for j in 0..cols {
                    z.data[i * cols + j] += temp * y.data[k * cols + j];
                }
            }
        }
    }

    fn mul_atb(x: &Tensor, y: &Tensor, z: &mut Tensor, rows: usize, common: usize, cols: usize) {
        for k in 0..common {
            for i in 0..rows {
                let temp = x.data[k * rows + i];
                for j in 0..cols {
                    z.data[i * cols + j] += temp * y.data[k * cols + j];
                }
            }
        }
    }

    fn mul_abt(x: &Tensor, y: &Tensor, z: &mut Tensor, rows: usize, common: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..common {
                    sum += x.data[i * common + k] * y.data[j * common + k];
                }
                z.data[i * cols + j] = sum;
            }
        }
    }

    fn mul_atbt(x: &Tensor, y: &Tensor, z: &mut Tensor, rows: usize, common: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..common {
                    sum += x.data[k * rows + i] * y.data[j * common + k];
                }
                z.data[i * cols + j] = sum;
            }
        }
    }

    fn add_broadcast_x(x: &Tensor, y: &Tensor, z: &mut Tensor) {
        let iterations = y.size / x.size;
        for i in 0..iterations {
            for j in 0..x.size {
                z.data[i * x.size + j] = x.data[j] + y.data[i * x.size + j];
            }
        }
    }

    fn add_broadcast_y(x: &Tensor, y: &Tensor, z: &mut Tensor) {
        let iterations = x.size / y.size;
        for i in 0..iterations {
            for j in 0..y.size {
                z.data[i * y.size + j] = x.data[i * y.size + j] + y.data[j];
            }
        }
    }

    fn sub_broadcast_x(x: &Tensor, y: &Tensor, z: &mut Tensor) {
        let iterations = y.size / x.size;
        for i in 0..iterations {
            for j in 0..x.size {
                z.data[i * x.size + j] = x.data[j] - y.data[i * x.size + j];
            }
        }
    }

    fn sub_broadcast_y(x: &Tensor, y: &Tensor, z: &mut Tensor) {
        let iterations = x.size / y.size;
        for i in 0..iterations {
            for j in 0..y.size {
                z.data[i * y.size + j] = x.data[i * y.size + j] - y.data[j];
            }
        }
    }
}
