//! Experts of types in blocks of 32 values run on the host exactly as the
//! device runs them (`moe_rows32` in moe.cuh), so that where an expert runs
//! never changes a token's result. The host's own path for these types
//! sums otherwise and stays as it is.

use half::f16;
use kinetic_cpu::pool;
use kinetic_quant::iq::IQ4NL_VALUES;
use kinetic_quant::TensorType;

use crate::blocks::{Experts, Job, Moe};

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

/// One row of IQ4_NL weights against a quantized input: each block's
/// integer sum times the product of the two scales, added in order.
fn dot(row: &[u8], qs: &[i8], ds: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (b, block) in row.chunks_exact(18).enumerate() {
        let wd = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let x = &qs[b * BLOCK..][..BLOCK];
        let mut sum = 0i32;
        for (j, &q) in block[2..].iter().enumerate() {
            sum += IQ4NL_VALUES[(q & 0xf) as usize] as i32 * x[j] as i32;
            sum += IQ4NL_VALUES[(q >> 4) as usize] as i32 * x[j + 16] as i32;
        }
        acc += wd * ds[b] * sum as f32;
    }
    acc
}

/// `m`'s rows of expert `e` against `rows` quantized inputs.
fn products(m: &Experts, e: usize, inputs: &[&(Vec<i8>, Vec<f32>)]) -> Vec<f32> {
    let w = m.expert(e);
    (0..m.rows)
        .flat_map(|r| inputs.iter().map(move |(qs, ds)| dot(w.row(r), qs, ds)))
        .collect()
}

/// Whether the host runs experts of `moe` here rather than through
/// [`Moe::run_experts`].
pub(super) fn handles(moe: &Moe) -> bool {
    [moe.gate.ty, moe.up.ty, moe.down.ty]
        .iter()
        .all(|&t| t == TensorType::IQ4NL)
}

/// [`Moe::run_experts`] with the device's arithmetic: one output row per
/// job, in order.
pub(super) fn run_experts(moe: &Moe, h: &[f32], jobs: &[Job]) -> Vec<f32> {
    let d = moe.router.cols;
    let x: Vec<(Vec<i8>, Vec<f32>)> = pool::map(h.len() / d, |t| quantize(&h[t * d..][..d]));
    let act: Vec<(Vec<i8>, Vec<f32>)> = pool::map(jobs.len(), |s| {
        let job = jobs[s];
        let input = [&x[job.row as usize]];
        let gate = products(&moe.gate, job.expert as usize, &input);
        let up = products(&moe.up, job.expert as usize, &input);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| moe.glu.apply(g, u))
            .collect();
        quantize(&act)
    });
    let mut out = vec![0.0; jobs.len() * moe.down.rows];
    pool::for_each_chunk(&mut out, moe.down.rows, |s, row| {
        let y = products(&moe.down, jobs[s].expert as usize, &[&act[s]]);
        row.copy_from_slice(&y);
    });
    out
}
