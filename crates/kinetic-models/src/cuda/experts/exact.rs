//! Experts run on the host exactly as the device runs them in their slot
//! form (see `form.rs`), so that where an expert runs never changes a
//! token's result: types the device reads per 256 values through the
//! host's own kernels, the others as `moe_rows_grouped` in moe.cuh sums
//! them. The host's own path for those types sums otherwise and stays as
//! it is.

use half::f16;
use kinetic_cpu::{pool, Q8K};
use kinetic_cuda::{expert_block, Q8K_BLOCK};
use kinetic_quant::grids::{ue4m3_half, MXFP4_VALUES};
use kinetic_quant::iq::IQ4NL_VALUES;
use kinetic_quant::TensorType;

use super::form::{device_type, row};
use crate::blocks::{packed_dot, Experts, Job, Moe};

/// Input values per quantization block of the grouped types.
const BLOCK: usize = 32;

/// `x` quantized per block of 32 as GEMV inputs are on the device: the
/// scale `max |x| / 127` and each value divided by it, rounded half to even.
fn quantize(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut qs = Vec::with_capacity(x.len());
    let mut ds = Vec::with_capacity(x.len() / BLOCK);
    for block in x.chunks_exact(BLOCK) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        qs.extend(block.iter().map(|&v| match amax {
            0.0 => 0,
            _ => (v / d).round_ties_even() as i8,
        }));
        ds.push(d);
    }
    (qs, ds)
}

/// A row of a grouped type against a quantized input: each group's integer
/// sum times its scale and its input block's, the groups added in order.
fn grouped_dot(ty: TensorType, row: &[u8], qs: &[i8], ds: &[f32]) -> f32 {
    let half = |b: &[u8]| f16::from_le_bytes([b[0], b[1]]).to_f32();
    // (weight scale, input block, integer sum) of each group in order
    let mut acc = 0.0f32;
    let mut add = |scale: f32, block: usize, sum: i32| acc += scale * ds[block] * sum as f32;
    match ty {
        TensorType::IQ4NL => {
            for (b, (w, x)) in row.chunks_exact(18).zip(qs.chunks_exact(32)).enumerate() {
                let sum = (0..16)
                    .map(|j| {
                        IQ4NL_VALUES[(w[2 + j] & 0xf) as usize] as i32 * x[j] as i32
                            + IQ4NL_VALUES[(w[2 + j] >> 4) as usize] as i32 * x[j + 16] as i32
                    })
                    .sum();
                add(half(w), b, sum);
            }
        }
        TensorType::NVFP4 => {
            for (b, (w, x)) in row.chunks_exact(36).zip(qs.chunks_exact(64)).enumerate() {
                for g in 0..4 {
                    let (q, x) = (&w[4 + 8 * g..][..8], &x[16 * g..][..16]);
                    let sum = (0..8)
                        .map(|j| {
                            MXFP4_VALUES[(q[j] & 0xf) as usize] as i32 * x[j] as i32
                                + MXFP4_VALUES[(q[j] >> 4) as usize] as i32 * x[j + 8] as i32
                        })
                        .sum();
                    add(ue4m3_half(w[g]), 2 * b + g / 2, sum);
                }
            }
        }
        TensorType::Q8_0 => {
            for (b, (w, x)) in row.chunks_exact(34).zip(qs.chunks_exact(32)).enumerate() {
                let sum = (0..32).map(|j| w[2 + j] as i8 as i32 * x[j] as i32).sum();
                add(half(w), b, sum);
            }
        }
        _ => unreachable!("{ty} is not a grouped expert type"),
    }
    acc
}
/// `m`'s expert `e` applied to each of `xs` as the device computes it, one
/// output row per input; each weight row is read once for all of them.
fn products(m: &Experts, e: usize, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let ty = device_type(m.ty);
    let mut out = vec![vec![0.0; m.rows]; xs.len()];
    if expert_block(ty) == Q8K_BLOCK {
        let qs: Vec<Q8K> = xs.iter().map(|x| Q8K::quantize(x)).collect();
        let w = m.expert(e);
        for r in 0..m.rows {
            for (o, q) in out.iter_mut().zip(&qs) {
                o[r] = packed_dot(ty, w.row(r), q);
            }
        }
        return out;
    }
    let qs: Vec<(Vec<i8>, Vec<f32>)> = xs.iter().map(|x| quantize(x)).collect();
    for r in 0..m.rows {
        let row = row(m, e, r);
        for (o, (q, d)) in out.iter_mut().zip(&qs) {
            o[r] = grouped_dot(ty, &row, q, d);
        }
    }
    out
}

/// Whether [`Moe::run_experts`] already computes what the device does:
/// when every matrix is read per 256 values in its own type.
pub(super) fn host_path_is_exact(moe: &Moe) -> bool {
    [&moe.gate, &moe.up, &moe.down]
        .iter()
        .all(|m| device_type(m.ty) == m.ty && expert_block(m.ty) == Q8K_BLOCK)
}

/// [`Moe::run_experts`] with the device's arithmetic: one output row per
/// job, in order. Each expert runs once for all of its jobs, so its weights
/// are read once per call.
pub(super) fn run_experts(moe: &Moe, h: &[f32], jobs: &[Job]) -> Vec<f32> {
    let d = moe.router.cols;
    let mut experts: Vec<(usize, Vec<usize>)> = Vec::new();
    for (s, j) in jobs.iter().enumerate() {
        match experts.iter_mut().find(|(e, _)| *e == j.expert as usize) {
            Some((_, slots)) => slots.push(s),
            None => experts.push((j.expert as usize, vec![s])),
        }
    }
    let scale = |i: usize, e: usize, rows: &mut [Vec<f32>]| {
        if let Some(s) = &moe.scales[i] {
            rows.iter_mut().flatten().for_each(|v| *v *= s[e]);
        }
    };
    let done = pool::map(experts.len(), |k| {
        let (e, slots) = (experts[k].0, &experts[k].1);
        let xs: Vec<Vec<f32>> = slots
            .iter()
            .map(|&s| h[jobs[s].row as usize * d..][..d].to_vec())
            .collect();
        let mut gate = products(&moe.gate, e, &xs);
        let mut up = products(&moe.up, e, &xs);
        scale(0, e, &mut gate);
        scale(1, e, &mut up);
        let acts: Vec<Vec<f32>> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| {
                g.iter()
                    .zip(u)
                    .map(|(&g, &u)| moe.glu.apply(g, u))
                    .collect()
            })
            .collect();
        let mut down = products(&moe.down, e, &acts);
        scale(2, e, &mut down);
        down
    });
    let mut out = vec![0.0; jobs.len() * moe.down.rows];
    for ((_, slots), rows) in experts.iter().zip(done) {
        for (&s, row) in slots.iter().zip(rows) {
            out[s * moe.down.rows..][..moe.down.rows].copy_from_slice(&row);
        }
    }
    out
}
