//! The form an expert takes in a device slot: each matrix as `expand.rs`
//! says the device reads its type, then, in files that have them, its
//! scales of the gate, up and down products.

use std::borrow::Cow;

use kinetic_cpu::pool;
use kinetic_cuda::EXPERT_TYPES;
use kinetic_quant::TensorType;

use super::super::expand::{self, Use};
use super::cache::SlotLayout;
use crate::blocks::{Experts, Moe};

/// How the device stores and reads expert matrices of type `ty`.
pub(super) fn device_type(ty: TensorType) -> TensorType {
    expand::device_type(ty, Use::Expert)
}

/// Whether experts of type `ty` can run on the device, read as they are or
/// expanded.
pub(super) fn supported(ty: TensorType) -> bool {
    EXPERT_TYPES.contains(&ty) || expand::expandable(ty)
}

pub(super) fn matrix_bytes(m: &Experts) -> usize {
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
    Cow::Owned(expand::row(m.ty, row, m.cols))
}

/// Writes expert `e` of `moe` in its slot form at the start of `dst` and
/// returns its length.
pub(super) fn stage(moe: &Moe, e: usize, dst: &mut [u8]) -> usize {
    stage_with(moe, e, dst, false)
}

/// [`stage`] with the copy spread over the thread pool, for an expert
/// staged while the pool has nothing else to do.
pub(super) fn stage_spread(moe: &Moe, e: usize, dst: &mut [u8]) -> usize {
    stage_with(moe, e, dst, true)
}

/// Bytes one task of [`stage_spread`] copies.
const PIECE: usize = 1 << 16;

fn stage_with(moe: &Moe, e: usize, dst: &mut [u8], spread: bool) -> usize {
    let mut at = 0;
    for m in [&moe.gate, &moe.up, &moe.down] {
        if device_type(m.ty) == m.ty {
            let data = m.expert(e).data;
            let out = &mut dst[at..at + data.len()];
            at += data.len();
            match spread {
                true => pool::for_each_chunk(out, PIECE, |i, piece| {
                    piece.copy_from_slice(&data[i * PIECE..][..piece.len()])
                }),
                false => out.copy_from_slice(data),
            }
            continue;
        }
        let len = matrix_bytes(m);
        let out = &mut dst[at..at + len];
        at += len;
        let write = |r: usize, piece: &mut [u8]| piece.copy_from_slice(&row(m, e, r));
        match spread {
            true => pool::for_each_chunk(out, len / m.rows, write),
            false => out
                .chunks_exact_mut(len / m.rows)
                .enumerate()
                .for_each(|(r, piece)| write(r, piece)),
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
        let kernel = [
            IQ4NL, NVFP4, Q8_0, Q5K, Q3K, IQ3XXS, IQ2XXS, IQ2XS, IQ2S, IQ1S, Q2_0,
        ];
        let expanded = [IQ3S, MXFP4, Q5_0, BF16];
        for ty in kernel {
            assert_eq!(device_type(ty), ty, "{ty}");
        }
        for ty in expanded {
            assert!(supported(ty), "{ty}");
            assert_eq!(device_type(ty), expand::EXPANDED, "{ty}");
        }
    }

    /// Staging spread over the pool writes the bytes staging alone does.
    #[test]
    fn spread_staging_writes_the_same_bytes() {
        use crate::blocks::{Experts, Gating, Glu};
        use kinetic_quant::Matrix;
        let (d, f, count) = (256, 512, 3);
        let bytes = |len: usize, k: u8| -> Vec<u8> {
            (0..len).map(|i| (i * 31 + k as usize) as u8).collect()
        };
        for (gate_ty, down_ty) in [
            (TensorType::IQ3XXS, TensorType::NVFP4),
            (TensorType::IQ3S, TensorType::Q5_0),
        ] {
            let size = |ty: TensorType, rows: usize, cols: usize| {
                count * rows * cols / ty.block_len() * ty.block_bytes()
            };
            let (gate, up, down) = (
                bytes(size(gate_ty, f, d), 1),
                bytes(size(gate_ty, f, d), 2),
                bytes(size(down_ty, d, f), 3),
            );
            let router = vec![0u8; 4 * d * count];
            fn experts(
                ty: TensorType,
                data: &[u8],
                (rows, cols, count): (usize, usize, usize),
            ) -> Experts<'_> {
                Experts {
                    ty,
                    data,
                    rows,
                    cols,
                    stride: data.len() / count,
                }
            }
            let moe = Moe {
                router: Matrix::new(TensorType::F32, &router, count, d),
                gate: experts(gate_ty, &gate, (f, d, count)),
                up: experts(gate_ty, &up, (f, d, count)),
                down: experts(down_ty, &down, (d, f, count)),
                n_used: 2,
                shared_gate: None,
                gating: Gating::Softmax,
                glu: Glu::Silu,
                weight_scale: 1.0,
                scales: [Some(vec![0.5, 1.0, 2.0]), None, None],
                norms: None,
            };
            let room = layout(&moe).bytes();
            for e in 0..count {
                let (mut want, mut got) = (vec![0; room], vec![0; room]);
                let len = stage(&moe, e, &mut want);
                assert_eq!(stage_spread(&moe, e, &mut got), len);
                assert!(want == got, "{gate_ty}/{down_ty} expert {e}");
            }
        }
    }
}
