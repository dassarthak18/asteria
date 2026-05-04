use std::ops::{Index, IndexMut};
use std::fmt;

/// N-dimensional dense tensor backed by a flat `Vec<f32>` with row-major strides.
///
/// All neural network inputs, outputs, weights, gradients, and intermediate values
/// are represented as `Tensor`. Indexing via `tensor[i]` addresses the flat buffer directly.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Flat, row-major storage of all elements.
    pub data: Vec<f32>,
    /// Size of each dimension, e.g. `[rows, cols]` for a matrix.
    pub shape: Vec<usize>,
    /// Row-major strides computed from `shape`.
    pub stride: Vec<usize>,
    /// Number of dimensions (`shape.len()`).
    pub rank: usize,
    /// Total number of elements (`shape.iter().product()`).
    pub size: usize,
    /// When `true`, matrix operations treat this tensor as transposed (logical flag only).
    pub transpose_flag: bool,
}

/// Controls how a tensor's data buffer is initialised on creation or resize.
#[derive(Copy, Clone, Debug)]
pub enum Init {
    /// Leave the buffer uninitialised (all zeros from allocation).
    None,
    /// Fill with `0.0`.
    Zero,
    /// Fill every element with the given constant.
    Value(f32),
    /// Set to the identity matrix; panics if the tensor is not square 2-D.
    Identity,
}

impl Tensor {
    /// Creates an empty zero-rank tensor; useful as a placeholder before resize.
    pub fn new() -> Self {
        Tensor {
            data: Vec::new(),
            shape: Vec::new(),
            stride: Vec::new(),
            rank: 0,
            size: 0,
            transpose_flag: false,
        }
    }

    /// Allocates a tensor with the given `shape` and applies `init` to every element.
    pub fn with_shape(shape: Vec<usize>, init: Init) -> Self {
        let rank = shape.len();
        let size = shape.iter().product();
        let stride = Self::compute_stride(&shape);
        let mut data = vec![0.0; size];

        match init {
            Init::Zero => {}
            Init::Value(v) => data.fill(v),
            Init::Identity => {
                if rank == 2 && shape[0] == shape[1] {
                    for i in 0..shape[0] {
                        data[i * shape[0] + i] = 1.0;
                    }
                } else {
                    panic!("Identity initialization used for non-square matrix");
                }
            }
            Init::None => {}
        }

        Tensor {
            data,
            shape,
            stride,
            rank,
            size,
            transpose_flag: false,
        }
    }

    /// Wraps an existing `data` buffer with the given `shape`; panics if lengths disagree.
    pub fn from_data(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let rank = shape.len();
        let size = shape.iter().product();
        assert_eq!(data.len(), size, "Data size does not match shape");
        let stride = Self::compute_stride(&shape);

        Tensor {
            data,
            shape,
            stride,
            rank,
            size,
            transpose_flag: false,
        }
    }

    fn compute_stride(shape: &[usize]) -> Vec<usize> {
        let mut stride = vec![1; shape.len()];
        let mut s = 1;
        for i in (0..shape.len()).rev() {
            stride[i] = s;
            s *= shape[i];
        }
        stride
    }

    /// Creates a tensor filled with `0.0`.
    pub fn zero(shape: Vec<usize>) -> Self {
        Self::with_shape(shape, Init::Zero)
    }

    /// Creates a tensor where every element equals `val`.
    pub fn value(shape: Vec<usize>, val: f32) -> Self {
        Self::with_shape(shape, Init::Value(val))
    }

    /// Alias for [`value`](Self::value); provided for naming consistency.
    pub fn with_shape_val(shape: Vec<usize>, val: f32) -> Self {
        Self::value(shape, val)
    }

    /// Alias for [`new`](Self::new); semantically signals an intentionally empty tensor.
    pub fn empty() -> Self {
        Self::new()
    }

    /// Shared slice over the flat element buffer.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Exclusive slice over the flat element buffer.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Changes the logical shape without reallocating; total element count must stay the same.
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(self.size, new_size, "Reshape size mismatch");
        self.shape = new_shape;
        self.rank = self.shape.len();
        self.stride = Self::compute_stride(&self.shape);
    }

    /// Reallocates the buffer if `new_shape` differs from the current shape, then applies `init`.
    pub fn resize(&mut self, new_shape: Vec<usize>, init: Init) {
        let new_size: usize = new_shape.iter().product();
        if self.shape != new_shape {
            self.shape = new_shape;
            self.rank = self.shape.len();
            self.stride = Self::compute_stride(&self.shape);
            self.size = new_size;
            self.data = vec![0.0; self.size];
            match init {
                Init::Zero => {}
                Init::Value(v) => self.data.fill(v),
                Init::Identity => {
                    if self.rank == 2 && self.shape[0] == self.shape[1] {
                        for i in 0..self.shape[0] {
                            self.data[i * self.shape[0] + i] = 1.0;
                        }
                    } else {
                        panic!("Identity initialization used for non-square matrix");
                    }
                }
                Init::None => {}
            }
        } else {
            match init {
                Init::Zero => self.data.fill(0.0),
                Init::Value(v) => self.data.fill(v),
                Init::Identity => {
                    self.data.fill(0.0);
                    for i in 0..self.shape[0] {
                        self.data[i * self.shape[0] + i] = 1.0;
                    }
                }
                Init::None => {}
            }
        }
        self.transpose_flag = false;
    }

    /// Toggles the logical transpose flag (no data movement).
    pub fn t(&mut self) {
        self.transpose_flag = !self.transpose_flag;
    }

    /// Sets every element to `val`.
    pub fn fill(&mut self, val: f32) {
        self.data.fill(val);
    }

    /// Returns the index of the maximum element along `dim` for each slice.
    ///
    /// For rank-1 tensors `dim` is ignored and a single index is returned.
    pub fn max_index(&self, dim: usize) -> Vec<usize> {
        let mut result = Vec::new();
        if self.rank == 1 {
            let mut max_idx = 0;
            let mut max_val = self.data[0];
            for i in 1..self.size {
                if self.data[i] > max_val {
                    max_val = self.data[i];
                    max_idx = i;
                }
            }
            result.push(max_idx);
        } else if self.rank == 2 {
            if dim == 0 {
                for i in 0..self.shape[0] {
                    let mut max_idx = 0;
                    let mut max_val = self.data[i * self.shape[1]];
                    for j in 1..self.shape[1] {
                        if self.data[i * self.shape[1] + j] > max_val {
                            max_val = self.data[i * self.shape[1] + j];
                            max_idx = j;
                        }
                    }
                    result.push(max_idx);
                }
            } else {
                for j in 0..self.shape[1] {
                    let mut max_idx = 0;
                    let mut max_val = self.data[j];
                    for i in 1..self.shape[0] {
                        if self.data[i * self.shape[1] + j] > max_val {
                            max_val = self.data[i * self.shape[1] + j];
                            max_idx = i;
                        }
                    }
                    result.push(max_idx);
                }
            }
        }
        result
    }

    /// Collects elements at the given flat `indices` into a new rank-1 tensor.
    pub fn gather(&self, indices: &[usize]) -> Tensor {
        let mut res = Tensor::with_shape(vec![indices.len()], Init::Zero);
        for (i, &idx) in indices.iter().enumerate() {
            res.data[i] = self.data[idx];
        }
        res
    }

    /// Returns the largest element across the entire tensor.
    pub fn max(&self) -> f32 {
        self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Returns the smallest element across the entire tensor.
    pub fn min(&self) -> f32 {
        self.data.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }

    /// Averages across the axis perpendicular to `dim` and returns the result.
    ///
    /// - `dim = 0`: averages each row across its columns; returns shape `[rows, 1]`.
    /// - `dim = 1`: averages each column across its rows; returns shape `[1, cols]`.
    pub fn mean(&self, dim: usize) -> Tensor {
        if self.rank == 1 {
            let sum: f32 = self.data.iter().sum();
            Tensor::from_data(vec![1], vec![sum / self.size as f32])
        } else if self.rank == 2 {
            if dim == 0 {
                let mut res = Tensor::with_shape(vec![self.shape[0], 1], Init::Zero);
                for i in 0..self.shape[0] {
                    let mut sum = 0.0;
                    for j in 0..self.shape[1] {
                        sum += self.data[i * self.shape[1] + j];
                    }
                    res.data[i] = sum / self.shape[1] as f32;
                }
                res
            } else {
                let mut res = Tensor::with_shape(vec![1, self.shape[1]], Init::Zero);
                for j in 0..self.shape[1] {
                    let mut sum = 0.0;
                    for i in 0..self.shape[0] {
                        sum += self.data[i * self.shape[1] + j];
                    }
                    res.data[j] = sum / self.shape[0] as f32;
                }
                res
            }
        } else {
            panic!("Mean not implemented for rank > 2");
        }
    }

    /// Returns the total number of elements.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Reads the element at a multi-dimensional `index` using stride arithmetic.
    pub fn get(&self, index: Vec<usize>) -> f32 {
        let mut offset = 0;
        for (i, &idx) in index.iter().enumerate() {
            offset += idx * self.stride[i];
        }
        self.data[offset]
    }

    /// Writes `val` at a multi-dimensional `index` using stride arithmetic.
    pub fn set(&mut self, index: Vec<usize>, val: f32) {
        let mut offset = 0;
        for (i, &idx) in index.iter().enumerate() {
            offset += idx * self.stride[i];
        }
        self.data[offset] = val;
    }

    /// Concatenates a slice of tensors along `dim` (rows=0, cols=1 for rank-2).
    pub fn concat(tensors: &[&Tensor], dim: usize) -> Tensor {
        if tensors.is_empty() {
            return Tensor {
                data: Vec::new(),
                shape: Vec::new(),
                stride: Vec::new(),
                rank: 0,
                size: 0,
                transpose_flag: false,
            };
        }
        let mut new_shape = tensors[0].shape.clone();
        let mut total_dim = 0;
        for t in tensors {
            total_dim += t.shape[dim];
        }
        new_shape[dim] = total_dim;
        
        let mut res = Tensor::with_shape(new_shape, Init::Zero);
        let mut offset = 0;
        
        for t in tensors {
            if t.rank == 2 {
                if dim == 0 {
                    for i in 0..t.shape[0] {
                        for j in 0..t.shape[1] {
                            res.data[(offset + i) * res.shape[1] + j] = t.data[i * t.shape[1] + j];
                        }
                    }
                    offset += t.shape[0];
                } else {
                    for i in 0..t.shape[0] {
                        for j in 0..t.shape[1] {
                            res.data[i * res.shape[1] + offset + j] = t.data[i * t.shape[1] + j];
                        }
                    }
                    offset += t.shape[1];
                }
            } else if t.rank == 1 {
                 for i in 0..t.shape[0] {
                    res.data[offset + i] = t.data[i];
                 }
                 offset += t.shape[0];
            }
        }
        res
    }

    /// Allocates a zero tensor with the same shape as `other`.
    pub fn zero_like(other: &Tensor) -> Tensor {
        Tensor::with_shape(other.shape.clone(), Init::Zero)
    }

    /// Overwrites `self`'s data with a byte-copy of `other`'s data; shapes must match.
    pub fn copy_params(&mut self, other: &Tensor) {
        self.data.copy_from_slice(&other.data);
    }
}

impl Index<usize> for Tensor {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.rank == 1 {
            for (i, val) in self.data.iter().enumerate() {
                write!(f, "{}", val)?;
                if i < self.size - 1 { write!(f, ",")?; }
            }
        } else if self.rank == 2 {
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    write!(f, "{}", self.data[i * self.shape[1] + j])?;
                    if j < self.shape[1] - 1 { write!(f, ",")?; }
                }
                writeln!(f)?;
            }
        } else {
            write!(f, "Tensor(rank={}, shape={:?})", self.rank, self.shape)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn from_data_roundtrip() {
        let t = Tensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.size, 6);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn value_fills_constant() {
        let t = Tensor::value(vec![2, 2], 3.14);
        assert!(t.data.iter().all(|&x| (x - 3.14).abs() < 1e-6));
    }

    #[test]
    fn identity_init() {
        let t = Tensor::with_shape(vec![3, 3], Init::Identity);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(t.data[i * 3 + j], expected);
            }
        }
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    #[test]
    fn get_set_2d() {
        let mut t = Tensor::with_shape(vec![3, 4], Init::Zero);
        t.set(vec![1, 2], 7.0);
        assert_eq!(t.get(vec![1, 2]), 7.0);
        assert_eq!(t.get(vec![0, 0]), 0.0);
    }

    // ── Reshape ───────────────────────────────────────────────────────────────

    #[test]
    fn reshape_preserves_data() {
        let mut t = Tensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        t.reshape(vec![3, 2]);
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ── max_index semantics (critical invariant) ──────────────────────────────

    #[test]
    fn max_index_dim0_returns_per_row_col_index() {
        // Shape [2, 3]: row 0 max at col 2, row 1 max at col 0
        let t = Tensor::from_data(vec![2, 3], vec![1.0, 2.0, 5.0, 9.0, 3.0, 4.0]);
        let idx = t.max_index(0);
        assert_eq!(idx, vec![2, 0], "max_index(0) must return column indices per row");
    }

    #[test]
    fn max_index_dim1_returns_per_col_row_index() {
        // Shape [3, 2]: col 0 max at row 2, col 1 max at row 0
        let t = Tensor::from_data(vec![3, 2], vec![1.0, 9.0, 4.0, 3.0, 7.0, 2.0]);
        let idx = t.max_index(1);
        assert_eq!(idx, vec![2, 0], "max_index(1) must return row indices per column");
    }

    #[test]
    fn max_index_rank1() {
        let t = Tensor::from_data(vec![4], vec![0.1, 0.7, 0.05, 0.15]);
        assert_eq!(t.max_index(0), vec![1]);
    }

    // ── gather ────────────────────────────────────────────────────────────────

    #[test]
    fn gather_picks_correct_flat_indices() {
        let t = Tensor::from_data(vec![2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        // flat index 2 → 30.0, flat index 4 → 50.0
        let g = t.gather(&[2, 4]);
        assert_eq!(g.data, vec![30.0, 50.0]);
    }

    // ── concat ────────────────────────────────────────────────────────────────

    #[test]
    fn concat_rows_dim0() {
        let a = Tensor::from_data(vec![1, 3], vec![1.0, 2.0, 3.0]);
        let b = Tensor::from_data(vec![2, 3], vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let c = Tensor::concat(&[&a, &b], 0);
        assert_eq!(c.shape, vec![3, 3]);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn concat_cols_dim1() {
        let a = Tensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::from_data(vec![2, 1], vec![5.0, 6.0]);
        let c = Tensor::concat(&[&a, &b], 1);
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(c.data, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    // ── max / min / mean ──────────────────────────────────────────────────────

    #[test]
    fn max_min_correct() {
        let t = Tensor::from_data(vec![2, 2], vec![-1.0, 3.0, 0.5, 2.0]);
        assert_eq!(t.max(), 3.0);
        assert_eq!(t.min(), -1.0);
    }

    #[test]
    fn mean_dim0_per_row_mean() {
        // Shape [2, 2]: mean(0) averages each row across its columns.
        // Row 0 mean: (1+2)/2 = 1.5, row 1 mean: (3+4)/2 = 3.5
        let t = Tensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let m = t.mean(0);
        assert_eq!(m.shape, vec![2, 1]);
        assert!((m.data[0] - 1.5).abs() < 1e-6);
        assert!((m.data[1] - 3.5).abs() < 1e-6);
    }

    // ── zero_like ─────────────────────────────────────────────────────────────

    #[test]
    fn zero_like_matches_shape() {
        let t = Tensor::from_data(vec![3, 4], vec![1.0; 12]);
        let z = Tensor::zero_like(&t);
        assert_eq!(z.shape, vec![3, 4]);
        assert!(z.data.iter().all(|&x| x == 0.0));
    }
}
