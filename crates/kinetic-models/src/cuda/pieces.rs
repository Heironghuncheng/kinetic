//! Each piece of Inkling 2 on the device against the host block it ports,
//! on synthetic f32 weights: attention by relative distance with its
//! convolutions, the mix over depth and a looped layer's control. Each test
//! passes without a device.

use half::f16;
use kinetic_cuda::{Batch, Context, Device, DeviceBuffer, GemvInput};
use kinetic_quant::{Matrix, TensorType};

use super::experts::{CudaExperts, ExpertDims, ExpertWork, Offload};
use super::network::control_on;
use super::relative::RelLayer;
use super::Gpu;
use crate::blocks::{mix, Control, Experts, Gating, Glu, Moe, RelAttention};

fn gpu() -> Option<Gpu> {
    let ctx = Context::new(Device::get(0).ok()?).ok()?;
    Gpu::new(ctx).ok()
}

/// `n` values `((i * mul) % m - m / 2) * scale`.
fn pattern(n: usize, mul: usize, m: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * mul % m) as f32 - (m / 2) as f32) * scale)
        .collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn matrix(b: &[u8], rows: usize, cols: usize) -> Matrix<'_> {
    Matrix::new(TensorType::F32, b, rows, cols)
}

fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want.iter().fold(1.0f32, |m, w| m.max(w.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * scale,
            "{what}, value {i}: {g} on the device against {w} on the host"
        );
    }
}

fn upload(g: &Gpu, x: &[f32]) -> DeviceBuffer<f32> {
    DeviceBuffer::from_slice(&g.ctx, x).unwrap()
}

const D: usize = 64;
const HD: usize = 64;
const TAPS: usize = 4;

/// The weights of one layer, `heads` query heads over one key-value head.
fn weights(heads: usize) -> Vec<Vec<u8>> {
    let w = heads * HD;
    [
        pattern(w * D, 3, 7, 0.05),
        pattern(HD * D, 5, 9, 0.04),
        pattern(HD * D, 7, 11, 0.03),
        pattern(D * w, 5, 13, 0.03),
        pattern(w * D, 2, 5, 0.06),
        pattern(heads * D, 3, 5, 0.05),
        pattern(2 * heads * D, 3, 11, 0.06),
        pattern(TAPS * D, 5, 11, 0.05),
        pattern(TAPS * D, 7, 13, 0.05),
    ]
    .iter()
    .map(|w| bytes(w))
    .collect()
}

fn attention(w: &[Vec<u8>], heads: usize, full: bool) -> RelAttention<'_> {
    let width = heads * HD;
    let extent = if full { 6 } else { 3 };
    let plus_one = |v: Vec<f32>| v.into_iter().map(|v| v + 1.0).collect::<Vec<_>>();
    RelAttention {
        norms: [
            plus_one(pattern(D, 1, 3, 0.25)),
            plus_one(pattern(D, 2, 5, 0.2)),
            plus_one(pattern(D, 3, 7, 0.1)),
        ],
        q: matrix(&w[0], width, D),
        k: matrix(&w[1], HD, D),
        v: matrix(&w[2], HD, D),
        o: matrix(&w[3], D, width),
        gate: matrix(&w[4], width, D),
        sink: matrix(&w[5], heads, D),
        q_scale: plus_one(pattern(width, 1, 3, 0.5)),
        rel: matrix(&w[6], 2 * heads, D),
        rel_scale: plus_one(pattern(2 * heads, 1, 3, 0.5)),
        rel_bias: pattern(2 * extent, 2, 7, 0.5),
        n_rel: 2,
        convs: [
            pattern(TAPS * HD, 5, 7, 0.2),
            pattern(TAPS * HD, 3, 5, 0.3),
            pattern(TAPS * D, 7, 9, 0.1),
            pattern(TAPS * D, 5, 11, 0.2),
        ],
        taps: TAPS,
        dynamic: full.then(|| [matrix(&w[7], TAPS, D), matrix(&w[8], TAPS, D)]),
        n_head: heads,
        n_head_kv: 1,
        head_dim: HD,
        window: (!full).then_some(3),
        qk_clip: 1.5,
        rel_clip: 0.8,
        log_scale: full.then_some((2.0, 0.5)),
        eps: 1e-6,
    }
}

/// Attention and the feed-forward convolution of seven tokens, run in
/// batches of `sizes` on the host and on the device, from the query, key
/// and value inputs, after each step of the device's.
#[test]
fn relative_attention_matches_the_host() {
    let Some(g) = gpu() else { return };
    let tokens = 18;
    let inputs = [
        pattern(tokens * D, 3, 17, 0.25),
        pattern(tokens * D, 5, 19, 0.2),
        pattern(tokens * D, 7, 23, 0.15),
    ];
    let w = weights(2);
    for full in [false, true] {
        let a = attention(&w, 2, full);
        let layer = RelLayer::upload(&g, &a).unwrap();
        let mut host = a.new_state();
        let mut state = layer.new_state(&g, 64, 16).unwrap();
        let mut b = layer.buffers(&g, 16).unwrap();
        let mut batch = Batch::new(&g.ctx, 16).unwrap();
        let mut out = DeviceBuffer::<f32>::new(&g.ctx, 16 * D).unwrap();
        let mut ffn = DeviceBuffer::<f32>::new(&g.ctx, 16 * D).unwrap();
        let mut at = 0;
        for n in [4, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1] {
            let rows = |x: &Vec<f32>| x[at * D..(at + n) * D].to_vec();
            let x = [rows(&inputs[0]), rows(&inputs[1]), rows(&inputs[2])];
            let want = a.forward([&x[0], &x[1], &x[2]], at, None, &mut host);
            let want_ffn = a.ffn_conv(&want, None, &mut host);

            batch.set(at, &vec![0; n], &g.stream).unwrap();
            let dx = x.each_ref().map(|x| upload(&g, x));
            layer
                .forward(&g, &mut b, dx.each_ref(), &mut state, &batch, &mut out)
                .unwrap();
            let y = upload(&g, &want);
            layer
                .ffn_conv(&g, &y, &mut state, &batch, &mut ffn)
                .unwrap();
            g.stream.synchronize().unwrap();
            let what = format!("full {full}, tokens {at}..{}", at + n);
            let got = |buf: &DeviceBuffer<f32>| buf.to_vec().unwrap()[..n * D].to_vec();
            close(&format!("{what}: attention"), &got(&out), &want, 2e-3);
            close(
                &format!("{what}: feed-forward convolution"),
                &got(&ffn),
                &want_ffn,
                1e-4,
            );
            at += n;
        }
    }
}

#[test]
fn depth_mix_matches_the_host() {
    let Some(g) = gpu() else { return };
    let n = 3;
    let states: Vec<Vec<f32>> = (0..4).map(|i| pattern(n * D, 3 + 2 * i, 17, 0.1)).collect();
    let queries = pattern(3 * D, 5, 13, 0.2);
    let want = mix(
        &states.iter().map(|s| &s[..]).collect::<Vec<_>>(),
        &queries,
        D,
    );
    let dev: Vec<DeviceBuffer<f32>> = states.iter().map(|s| upload(&g, s)).collect();
    let mut outs: Vec<DeviceBuffer<f32>> = (0..3).map(|_| upload(&g, &vec![0.0; n * D])).collect();
    let q = upload(&g, &queries);
    let refs: Vec<&DeviceBuffer<f32>> = dev.iter().collect();
    let mut out_refs: Vec<&mut DeviceBuffer<f32>> = outs.iter_mut().collect();
    g.kernels
        .depth_mix(&refs, &q, &mut out_refs, D, n, &g.stream)
        .unwrap();
    for (o, w) in outs.iter().zip(&want) {
        close("mix over depth", &o.to_vec().unwrap(), w, 1e-5);
    }
}

#[test]
fn control_matches_the_host() {
    let Some(g) = gpu() else { return };
    let (n, rank) = (3, 32);
    let w = [
        bytes(&pattern(rank * 2 * D, 7, 11, 0.05)),
        bytes(&pattern(D * rank, 5, 7, 0.1)),
        bytes(&pattern(D * rank, 3, 5, 0.1)),
    ];
    let c = Control {
        down: matrix(&w[0], rank, 2 * D),
        gain: matrix(&w[1], D, rank),
        forget: matrix(&w[2], D, rank),
    };
    let x = pattern(n * D, 3, 13, 0.3);
    let message = pattern(n * D, 5, 17, 0.2);
    let mut want = x.clone();
    c.apply(&mut want, &message);
    let got = control_on(&g, &c, &x, &message, n).unwrap();
    close("control", &got, &want, 1e-5);
}

/// IQ4_NL blocks of 32 with a small scale and nibbles from `seed`.
fn iq4_nl(values: usize, seed: usize) -> Vec<u8> {
    (0..values / 32)
        .flat_map(|b| {
            let mut block = f16::from_f32(0.002).to_le_bytes().to_vec();
            block.extend((0..16).map(|j| ((b * 16 + j) * (2 * seed + 5) % 251) as u8));
            block
        })
        .collect()
}

/// Routed experts as Inkling 2 has them: gate and up in one IQ4_NL tensor,
/// SITU, sigmoid routing with a bias and a weight scale, a width that is
/// not a multiple of 256.
#[test]
fn routed_experts_match_the_host() {
    let Some(g) = gpu() else { return };
    let (count, ff, n) = (4, 96, 3);
    let gate_up = iq4_nl(count * 2 * ff * D, 1);
    let down = iq4_nl(count * D * ff, 2);
    let router = bytes(&pattern(count * D, 5, 11, 0.05));
    let experts = |data: &'static [u8], rows, cols| Experts {
        ty: TensorType::IQ4NL,
        data,
        rows,
        cols,
        stride: data.len() / count,
    };
    let (gate_up, down): (&'static [u8], &'static [u8]) = (gate_up.leak(), down.leak());
    let (gate, up) = experts(gate_up, 2 * ff, D).halves();
    let moe = Moe {
        router: matrix(&router, count, D),
        gate,
        up,
        down: experts(down, D, ff),
        n_used: 2,
        shared_gate: None,
        gating: Gating::Sigmoid {
            bias: pattern(count, 3, 5, 0.01),
        },
        glu: Glu::Situ {
            beta: 4.0,
            linear: 25.0,
        },
        weight_scale: 2.0,
    };
    let h = pattern(n * D, 7, 19, 0.3);
    let mut want = vec![0.0; n * D];
    moe.forward(&h, &mut want);

    let e = CudaExperts::upload(&g, &moe, None, false).unwrap();
    let mut off = Offload::new(&g, vec![moe.clone()], 1 << 30, None, 0).unwrap();
    assert!(off.is_resident());
    let dims = ExpertDims {
        d: D,
        rows: 16,
        picks: 2,
        n_ff: ff,
        scores: e.score_rows(),
    };
    let mut work = ExpertWork::new(&g, &dims).unwrap();
    let mut input = GemvInput::batch(&g.ctx, D, 16).unwrap();
    input.upload(&g.kernels, &h, &g.stream).unwrap();
    let mut y = DeviceBuffer::<f32>::new(&g.ctx, n * D).unwrap();
    e.score(&g, &input, n, &mut work.gates).unwrap();
    e.mix(0, &g, input.values(), &mut y, &mut work, &mut off, n)
        .unwrap();
    close("routed experts", &y.to_vec().unwrap(), &want, 2e-3);
}
