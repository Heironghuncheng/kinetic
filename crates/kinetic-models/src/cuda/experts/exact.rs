//! Experts run on the host exactly as the device runs them in their slot
//! form (see `form.rs`), so that where an expert runs never changes a
//! token's result: types the device reads per 256 values through the
//! host's own kernels, the others as `moe_rows_grouped` in moe.cuh sums
//! them. The host's own path for those types sums otherwise and stays as
//! it is.

use std::cell::RefCell;

use half::f16;
use kinetic_cpu::{pool, Q8K};
use kinetic_cuda::{expert_block, Q8K_BLOCK};
use kinetic_quant::grids::{ue4m3_half, IQ1S_GRID, IQ2S_GRID, MXFP4_VALUES};
use kinetic_quant::iq::IQ4NL_VALUES;
use kinetic_quant::TensorType;

use super::form::{device_type, row};
use crate::blocks::{packed_dot, unpack, unpacked_dot, Experts, Job, Moe, Unpacked};

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
        TensorType::IQ2S => {
            for (b, (w, x)) in row.chunks_exact(82).zip(qs.chunks_exact(256)).enumerate() {
                for g in 0..16 {
                    let sum = (2 * g..2 * g + 2)
                        .map(|k| {
                            let index = w[2 + k] as usize
                                | (w[66 + k / 4] as usize >> (2 * (k % 4)) & 3) << 8;
                            let grid = &IQ2S_GRID[index];
                            (0..8)
                                .map(|j| {
                                    let v = grid[j] as i32;
                                    let v = if w[34 + k] >> j & 1 == 1 { -v } else { v };
                                    v * x[8 * k + j] as i32
                                })
                                .sum::<i32>()
                        })
                        .sum();
                    let s = w[74 + g / 2] >> (4 * (g % 2)) & 0xf;
                    add(half(w) * (0.5 + s as f32) * 0.25, 8 * b + g / 2, sum);
                }
            }
        }
        TensorType::IQ1S => {
            for (b, (w, x)) in row.chunks_exact(50).zip(qs.chunks_exact(256)).enumerate() {
                for sub in 0..8 {
                    let qh = u16::from_le_bytes([w[34 + 2 * sub], w[35 + 2 * sub]]) as usize;
                    let delta = if qh & 0x8000 != 0 { -1 } else { 1 };
                    let sum = (0..4)
                        .map(|l| {
                            let index = w[2 + 4 * sub + l] as usize | (qh >> (3 * l) & 7) << 8;
                            (0..8)
                                .map(|j| {
                                    (8 * IQ1S_GRID[index][j] as i32 + delta)
                                        * x[32 * sub + 8 * l + j] as i32
                                })
                                .sum::<i32>()
                        })
                        .sum();
                    let scale = (2 * (qh >> 12 & 7) + 1) as f32;
                    add(half(w) * scale * 0.125, 8 * b + sub, sum);
                }
            }
        }
        TensorType::Q2_0 => {
            for (b, (w, x)) in row.chunks_exact(18).zip(qs.chunks_exact(64)).enumerate() {
                for h in 0..2 {
                    let sum = (32 * h..32 * h + 32)
                        .map(|j| ((w[2 + j / 4] >> (2 * (j % 4)) & 3) as i32 - 1) * x[j] as i32)
                        .sum();
                    add(half(w), 2 * b + h, sum);
                }
            }
        }
        _ => unreachable!("{ty} is not a grouped expert type"),
    }
    acc
}
/// Weight rows one parallel task computes for every job of one expert.
const ROWS_PER_TASK: usize = 32;

/// Rows of inputs quantized as the device quantizes them for one type.
enum Inputs {
    PerK(Vec<Q8K>),
    Per32(Vec<(Vec<i8>, Vec<f32>)>),
}

impl Inputs {
    /// The `width`-value rows of `x` quantized for weights of type `ty`.
    fn new(ty: TensorType, x: &[f32], width: usize) -> Inputs {
        let rows = x.len() / width;
        let row = |i: usize| &x[i * width..][..width];
        match expert_block(ty) {
            Q8K_BLOCK => Inputs::PerK(pool::map(rows, |i| Q8K::quantize(row(i)))),
            _ => Inputs::Per32(pool::map(rows, |i| quantize(row(i)))),
        }
    }

    /// A weight row of type `ty` times each of the input rows `rows`, onto
    /// `out`. A row read for several inputs is decoded once.
    fn dots(
        &self,
        ty: TensorType,
        row: &[u8],
        rows: &[usize],
        fast: Option<Dots>,
        out: &mut Vec<f32>,
    ) {
        match self {
            Inputs::PerK(q) if rows.len() == 1 => out.push(packed_dot(ty, row, &q[rows[0]])),
            Inputs::PerK(q) => UNPACKED.with_borrow_mut(|unpacked| {
                unpacked.resize(row.len() / ty.block_bytes(), Unpacked::default());
                unpack(ty, row, unpacked);
                out.extend(rows.iter().map(|&i| unpacked_dot(unpacked, &q[i])));
            }),
            Inputs::Per32(q) => match fast {
                Some(dots) => {
                    for rows in rows.chunks(MOST) {
                        let xs: Vec<(&[i8], &[f32])> =
                            rows.iter().map(|&i| (&q[i].0[..], &q[i].1[..])).collect();
                        let start = out.len();
                        out.resize(start + rows.len(), 0.0);
                        dots(row, &xs, &mut out[start..]);
                    }
                }
                None => out.extend(rows.iter().map(|&i| grouped_dot(ty, row, &q[i].0, &q[i].1))),
            },
        }
    }
}

thread_local! {
    /// The unpacked weight row each thread reuses.
    static UNPACKED: RefCell<Vec<Unpacked>> = RefCell::default();
}

#[cfg(target_arch = "x86_64")]
use super::grouped::{dots_fn, Dots, MOST};

#[cfg(not(target_arch = "x86_64"))]
type Dots = fn(&[u8], &[(&[i8], &[f32])], &mut [f32]);
#[cfg(not(target_arch = "x86_64"))]
const MOST: usize = 8;
#[cfg(not(target_arch = "x86_64"))]
fn dots_fn(_: TensorType) -> Option<Dots> {
    None
}

/// Each of `ms`, alike in shape, applied as the device computes it to the
/// input row `input(s)` of every job slot `s` of each expert in `experts`,
/// and the products of each expert merged into one by `merge`: `slots` rows
/// of the matrices' height. A parallel task computes and merges some rows
/// of one expert for all of its slots, so each weight row is read once.
fn products<const N: usize>(
    ms: [&Experts; N],
    experts: &[(usize, Vec<usize>)],
    x: &Inputs,
    input: impl Fn(usize) -> usize + Sync,
    slots: usize,
    merge: impl Fn(usize, [Vec<f32>; N]) -> Vec<f32> + Sync,
) -> Vec<f32> {
    let rows = ms[0].rows;
    let tasks: Vec<(usize, usize)> = (0..experts.len())
        .flat_map(|k| (0..rows).step_by(ROWS_PER_TASK).map(move |r| (k, r)))
        .collect();
    let kinds = ms.map(|m| (device_type(m.ty), dots_fn(device_type(m.ty))));
    let done: Vec<Vec<f32>> = pool::map(tasks.len(), |t| {
        let (k, r0) = tasks[t];
        let (e, picks) = &experts[k];
        let inputs: Vec<usize> = picks.iter().map(|&s| input(s)).collect();
        let mut i = 0;
        let outs = ms.map(|m| {
            let (ty, fast) = kinds[i];
            i += 1;
            let mut out = Vec::with_capacity(ROWS_PER_TASK * picks.len());
            for r in r0..rows.min(r0 + ROWS_PER_TASK) {
                x.dots(ty, &row(m, *e, r), &inputs, fast, &mut out);
            }
            out
        });
        merge(*e, outs)
    });
    let mut y = vec![0.0; slots * rows];
    for (&(k, r0), out) in tasks.iter().zip(&done) {
        let picks = &experts[k].1;
        for (r, values) in out.chunks_exact(picks.len()).enumerate() {
            for (&s, &v) in picks.iter().zip(values) {
                y[s * rows + r0 + r] = v;
            }
        }
    }
    y
}

/// Whether [`Moe::run_experts`] already computes what the device does:
/// when every matrix is read per 256 values in its own type.
pub(super) fn host_path_is_exact(moe: &Moe) -> bool {
    [&moe.gate, &moe.up, &moe.down]
        .iter()
        .all(|m| device_type(m.ty) == m.ty && expert_block(m.ty) == Q8K_BLOCK)
}

/// [`Moe::run_experts`] with the device's arithmetic: one output row per
/// job, in order. Each expert's weights are read once per call, by every
/// thread a share of its rows.
pub(super) fn run_experts(moe: &Moe, h: &[f32], jobs: &[Job]) -> Vec<f32> {
    let (d, f) = (moe.router.cols, moe.gate.rows);
    let mut experts: Vec<(usize, Vec<usize>)> = Vec::new();
    for (s, j) in jobs.iter().enumerate() {
        match experts.iter_mut().find(|(e, _)| *e == j.expert as usize) {
            Some((_, slots)) => slots.push(s),
            None => experts.push((j.expert as usize, vec![s])),
        }
    }
    let scaled = |i: usize, e: usize, v: f32| match &moe.scales[i] {
        Some(s) => v * s[e],
        None => v,
    };
    let x = Inputs::new(device_type(moe.gate.ty), h, d);
    let act = products(
        [&moe.gate, &moe.up],
        &experts,
        &x,
        |s| jobs[s].row as usize,
        jobs.len(),
        |e, [gate, up]| {
            gate.iter()
                .zip(&up)
                .map(|(&g, &u)| moe.glu.apply(scaled(0, e, g), scaled(1, e, u)))
                .collect()
        },
    );
    let x = Inputs::new(device_type(moe.down.ty), &act, f);
    products(
        [&moe.down],
        &experts,
        &x,
        |s| s,
        jobs.len(),
        |e, [out]| out.into_iter().map(|v| scaled(2, e, v)).collect(),
    )
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    /// Every grouped type's vector form gives the scalar form's bits, for
    /// one input and for several.
    #[test]
    fn vector_grouped_dots_match_the_scalar_ones() {
        use TensorType::*;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (ty, inputs) in [IQ4NL, NVFP4, Q8_0, Q2_0, IQ2S, IQ1S]
            .into_iter()
            .flat_map(|ty| [(ty, 1), (ty, 3)])
        {
            let Some(fast) = dots_fn(ty) else {
                eprintln!("skipped: no AVX2");
                return;
            };
            // an odd count of 64-value blocks where the type allows
            let cols = if ty.block_len() == 256 { 512 } else { 576 };
            for _ in 0..20 {
                let xs: Vec<(Vec<i8>, Vec<f32>)> = (0..inputs)
                    .map(|_| {
                        let x: Vec<f32> = (0..cols)
                            .map(|_| (next() % 2001) as f32 / 400.0 - 2.5)
                            .collect();
                        quantize(&x)
                    })
                    .collect();
                let mut row: Vec<u8> = (0..cols / ty.block_len() * ty.block_bytes())
                    .map(|_| next() as u8)
                    .collect();
                // finite scales: halves below 2 and E4M3 bytes below 0x7f
                for block in row.chunks_exact_mut(ty.block_bytes()) {
                    match ty {
                        NVFP4 => block[..4].iter_mut().for_each(|b| *b &= 0x5f),
                        _ => block[1] &= 0x3b,
                    }
                }
                let want: Vec<u32> = xs
                    .iter()
                    .map(|(qs, ds)| grouped_dot(ty, &row, qs, ds).to_bits())
                    .collect();
                let inputs: Vec<(&[i8], &[f32])> =
                    xs.iter().map(|(q, d)| (&q[..], &d[..])).collect();
                let mut got = vec![0.0; xs.len()];
                fast(&row, &inputs, &mut got);
                let got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                assert_eq!(got, want, "{ty}");
            }
        }
    }
}
