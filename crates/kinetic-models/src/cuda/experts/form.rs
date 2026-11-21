//! The form an expert takes in a device slot: each matrix in its file's
//! type where an expert kernel reads that type ([`EXPERT_TYPES`]), or else
//! expanded on the host into [`EXPANDED`], which one does; then, in files
//! that have them, its scales of the gate, up and down products.

use std::borrow::Cow;

use kinetic_cuda::EXPERT_TYPES;
use kinetic_quant::{dequantize, quantize, TensorType};

use super::cache::SlotLayout;
use crate::blocks::{Experts, Moe};

/// The type expert matrices without a kernel of their own are expanded to.
pub(super) const EXPANDED: TensorType = TensorType::Q8_0;

/// How the device stores and reads expert matrices of type `ty`.
pub(super) fn device_type(ty: TensorType) -> TensorType {
    if EXPERT_TYPES.contains(&ty) {
        ty
    } else {
        EXPANDED
    }
}

/// Whether experts of type `ty` can run on the device, read as they are or
/// expanded.
pub(super) fn supported(ty: TensorType) -> bool {
    let mut row = vec![0.0; ty.block_len()];
    EXPERT_TYPES.contains(&ty) || dequantize(ty, &vec![0; ty.block_bytes()], &mut row).is_ok()
}

fn matrix_bytes(m: &Experts) -> usize {
    let ty = device_type(m.ty);
    m.rows * m.cols / ty.block_len() * ty.block_bytes()
}

/// Where each part of an expert of `moe` lies in a slot.
pub(super) fn layout(moe: &Moe) -> SlotLayout {
    SlotLayout {
        gate: matrix_bytes(&moe.gate),
        up: matrix_bytes(&moe.up),
        down: matrix_bytes(&moe.down),
        scales: if moe.scales.iter().any(Option::is_some) {
            3 * size_of::<f32>()
        } else {
            0
        },
    }
}

/// Row `r` of `m`'s expert `e` in device form.
pub(super) fn row<'a>(m: &Experts<'a>, e: usize, r: usize) -> Cow<'a, [u8]> {
    let row = m.expert(e).row(r);
    if device_type(m.ty) == m.ty {
        return Cow::Borrowed(row);
    }
    let mut x = vec![0.0; m.cols];
    dequantize(m.ty, row, &mut x).expect("supported types dequantize");
    let mut out = vec![0; m.cols / EXPANDED.block_len() * EXPANDED.block_bytes()];
    assert!(
        quantize(EXPANDED, &x, &mut out),
        "the expanded type quantizes"
    );
    Cow::Owned(out)
}

/// Writes expert `e` of `moe` in its slot form at the start of `dst` and
/// returns its length.
pub(super) fn stage(moe: &Moe, e: usize, dst: &mut [u8]) -> usize {
    let mut at = 0;
    for m in [&moe.gate, &moe.up, &moe.down] {
        if device_type(m.ty) == m.ty {
            let data = m.expert(e).data;
            dst[at..at + data.len()].copy_from_slice(data);
            at += data.len();
            continue;
        }
        for r in 0..m.rows {
            let row = row(m, e, r);
            dst[at..at + row.len()].copy_from_slice(&row);
            at += row.len();
        }
    }
    if layout(moe).scales > 0 {
        for s in &moe.scales {
            let v = s.as_ref().map_or(1.0, |s| s[e]);
            dst[at..at + 4].copy_from_slice(&v.to_le_bytes());
            at += 4;
        }
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The types of the published Inkling 2 files, and how each reaches the
    /// device.
    #[test]
    fn every_published_expert_type_has_a_device_form() {
        use TensorType::*;
        let kernel = [IQ4NL, NVFP4, Q8_0, Q5K, Q3K, IQ3XXS, IQ2XXS, IQ2XS];
        let expanded = [IQ2S, IQ3S, IQ1S, MXFP4, Q2_0];
        for ty in kernel {
            assert_eq!(device_type(ty), ty, "{ty}");
        }
        for ty in expanded {
            assert!(supported(ty), "{ty}");
            assert_eq!(device_type(ty), EXPANDED, "{ty}");
        }
    }
}
