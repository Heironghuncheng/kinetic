//! How the device reads each weight type: in its own bytes where a kernel
//! reads that type, or else expanded on the host into [`EXPANDED`], which
//! one does. One path for every tensor: ordinary matrices, which GEMV
//! kernels read ([`DeviceMatrix::reads`]), and routed experts, which expert
//! kernels read ([`EXPERT_TYPES`]). Host and device run an expanded weight
//! from the same expanded bytes, so their results agree bit for bit.

use std::cell::RefCell;

use kinetic_cuda::{DeviceMatrix, EXPERT_TYPES};
use kinetic_quant::{dequantize, quantize, Matrix, TensorType};

/// The type weights without a kernel of their own are expanded to.
pub(super) const EXPANDED: TensorType = TensorType::Q8_0;

/// Whether a tensor is an ordinary matrix or a routed expert's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Use {
    Matrix,
    Expert,
}

/// The type the device reads weights of type `ty` used as `u` in.
pub(super) fn device_type(ty: TensorType, u: Use) -> TensorType {
    let kernel = match u {
        Use::Matrix => DeviceMatrix::reads(ty),
        Use::Expert => EXPERT_TYPES.contains(&ty),
    };
    if kernel {
        ty
    } else {
        EXPANDED
    }
}

/// Whether weights of type `ty` can be expanded: whether the host reads it.
pub(super) fn expandable(ty: TensorType) -> bool {
    let mut row = vec![0.0; ty.block_len()];
    dequantize(ty, &vec![0; ty.block_bytes()], &mut row).is_ok()
}

/// A row of `cols` weights of type `ty` expanded into [`EXPANDED`].
pub(super) fn row(ty: TensorType, row: &[u8], cols: usize) -> Vec<u8> {
    let mut x = vec![0.0; cols];
    dequantize(ty, row, &mut x).expect("expanded types dequantize");
    let mut out = vec![0; cols / EXPANDED.block_len() * EXPANDED.block_bytes()];
    assert!(
        quantize(EXPANDED, &x, &mut out),
        "the expanded type quantizes"
    );
    out
}

/// Every row of `m` expanded into [`EXPANDED`].
pub(super) fn matrix(m: &Matrix) -> Vec<u8> {
    (0..m.rows)
        .flat_map(|r| row(m.ty, m.row(r), m.cols))
        .collect()
}

/// The weight types a load expanded and the device bytes that took beyond
/// the file's, to say once the model is up.
#[derive(Default)]
pub(super) struct Expanded(RefCell<Vec<(TensorType, isize)>>);

impl Expanded {
    pub fn add(&self, ty: TensorType, extra: isize) {
        self.0.borrow_mut().push((ty, extra));
    }

    /// One plain line naming the expanded types and their cost, if any.
    pub fn report(&self) -> Option<String> {
        let all = self.0.borrow();
        if all.is_empty() {
            return None;
        }
        let mut types: Vec<String> = all.iter().map(|(t, _)| t.to_string()).collect();
        types.sort();
        types.dedup();
        let extra: isize = all.iter().map(|(_, e)| e).sum();
        let mib = extra.unsigned_abs() as f64 / (1 << 20) as f64;
        let size = match extra {
            0.. => format!("{mib:.0} MiB more"),
            _ => format!("{mib:.0} MiB less"),
        };
        Some(format!(
            "kinetic: CUDA has no kernel for {} weights; they run expanded to {EXPANDED}, which takes {size} than the file",
            types.join(", "),
        ))
    }
}
