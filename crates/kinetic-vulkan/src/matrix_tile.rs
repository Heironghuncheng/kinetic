/// Rows and tokens of a workgroup's tile in the kernels on the matrix
/// units. build.rs reads this file too and hands it to mm.wgsl as `TILE`,
/// whose layout (four subgroups of 32 by 32 quarters) is written for 64.
pub(crate) const MATRIX_TILE: usize = 64;
