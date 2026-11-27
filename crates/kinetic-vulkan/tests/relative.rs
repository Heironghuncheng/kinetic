//! Attention by relative distance and the kernels around it against host
//! references in f64, on every Vulkan device that can run them. Each test
//! passes with a message when there is none.

use std::rc::Rc;

use half::f16;
use kinetic_vulkan::{
    AttentionGate, AttentionShape, Batch, Commands, ConvRing, Device, DeviceBuffer, Error,
    GemvInput, Instance, Kernels, KvCache, RelativeShape, States, Strided,
};

struct Gpu {
    device: Rc<Device>,
    kernels: Kernels,
}

fn gpus() -> Vec<Gpu> {
    let physical = match Instance::new().and_then(|i| i.physical_devices()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipped: {e}");
            return Vec::new();
        }
    };
    physical
        .iter()
        .filter_map(|p| match Device::new(p) {
            Ok(device) => Some(Gpu {
                kernels: Kernels::load(&device).unwrap(),
                device,
            }),
            Err(e @ Error::DeviceUnsupported(_)) => {
                eprintln!("skipped {}: {e}", p.info().name);
                None
            }
            Err(e) => panic!("{}: {e}", p.info().name),
        })
        .collect()
}

/// `n` values in [-scale, scale] from a fixed sequence.
fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * scale
        })
        .collect()
}

fn close(what: &str, got: &[f32], want: &[f64]) {
    let scale = want.iter().fold(1e-6f64, |m, v| m.max(v.abs()));
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g as f64 - w).abs() <= 1e-5 * scale,
            "{what}, value {i}: {g} on the device against {w}"
        );
    }
}

/// The f16 values `bits` as f64, each within one f16 step of `want`.
fn stored(what: &str, bits: &[u16], want: &[f64]) -> Vec<f64> {
    let got: Vec<f64> = bits.iter().map(|&b| f16::from_bits(b).to_f64()).collect();
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= w.abs() / 1024.0 + 1e-7,
            "{what}, value {i}: {g} stored for {w}"
        );
    }
    got
}

/// Scales `x` by `s`, then down to a norm of at most `most`, then by `tau`.
fn clip(x: &mut [f64], s: &[f32], most: f64, tau: f64) {
    x.iter_mut().zip(s).for_each(|(x, &s)| *x *= s as f64);
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-6;
    let f = (most / norm).clamp(0.0, 1.0);
    x.iter_mut().for_each(|v| *v *= f * tau);
}

#[test]
fn relative_attention_matches_the_reference() {
    for window in [None, Some(2)] {
        check_relative(window);
    }
}

fn check_relative(window: Option<usize>) {
    let (nh, kv, hd, n_rel, extent, tokens) = (2, 1, 64, 2, 3, 4);
    let (gate_at, rel_at) = (nh * hd, 2 * nh * hd);
    let sink_at = rel_at + nh * n_rel;
    let (k_at, v_at) = (sink_at + nh, sink_at + nh + kv * hd);
    let stride = v_at + kv * hd;
    let shape = RelativeShape {
        heads: AttentionShape {
            n_head: nh,
            n_head_kv: kv,
            head_dim: hd,
            rope_dims: 0,
            rope_adjacent: false,
            norm_after_rope: false,
            gate: AttentionGate::None,
            window,
            scale: 1.0 / (hd as f32).sqrt(),
            eps: 1e-6,
        },
        n_rel,
        extent,
        stride,
        gate_at,
        rel_at,
        sink_at,
        k_at,
        v_at,
        qk_clip: 1.5,
        rel_clip: 0.8,
        log_scale: Some((2.0, 0.5)),
    };
    let qkv = values(tokens * stride, 1, 2.0);
    let scales: Vec<f32> = values(nh * (hd + n_rel), 2, 0.5)
        .iter()
        .map(|v| v + 1.0)
        .collect();
    let rel_bias = values(extent * n_rel, 3, 1.0);
    for gpu in gpus() {
        let dev = &gpu.device;
        let device_qkv = DeviceBuffer::from_slice(dev, &qkv).unwrap();
        let device_scales = DeviceBuffer::from_slice(dev, &scales).unwrap();
        let device_bias = DeviceBuffer::from_slice(dev, &rel_bias).unwrap();
        let cache = KvCache::new(dev, &shape.heads, 16, tokens).unwrap();
        let out = GemvInput::batch(dev, nh * hd, tokens).unwrap();
        let mut batch = Batch::new(dev, tokens, 1, 16, 0).unwrap();
        batch.set(tokens, 0, 0..0, None).unwrap();
        let mut c = Commands::new(dev).unwrap();
        batch.record_upload(&mut c).unwrap();
        gpu.kernels
            .relative_attention(
                &batch,
                &shape,
                &device_qkv,
                &device_scales,
                &device_bias,
                &cache,
                &out,
                &mut c,
            )
            .unwrap();
        c.submit().unwrap();
        let got = out.values().to_vec().unwrap();
        let readied = device_qkv.to_vec().unwrap();

        // The host reference, with keys and values stored as f16.
        let row = |t: usize, at: usize, len: usize| -> Vec<f64> {
            qkv[t * stride + at..][..len]
                .iter()
                .map(|&v| v as f64)
                .collect()
        };
        // The reference reads the f16 values the cache holds, whichever way
        // the driver rounds them, once they are within a step of exact.
        let (kc, vc) = cache.save(0..tokens).unwrap();
        let mut keys = Vec::new();
        let mut vals = Vec::new();
        for t in 0..tokens {
            let mut k = row(t, k_at, hd);
            clip(&mut k, &[1.0; 64], 1.5, 1.0);
            let mut v = row(t, v_at, hd);
            let ms = v.iter().map(|x| x * x).sum::<f64>() / hd as f64;
            v.iter_mut().for_each(|x| *x /= (ms + 1e-6).sqrt());
            close(
                &format!("{window:?} key {t}"),
                &readied[t * stride + k_at..][..hd],
                &k,
            );
            close(
                &format!("{window:?} value {t}"),
                &readied[t * stride + v_at..][..hd],
                &v,
            );
            keys.push(stored(
                &format!("{window:?} cached key {t}"),
                &kc[t * hd..][..hd],
                &k,
            ));
            vals.push(stored(
                &format!("{window:?} cached value {t}"),
                &vc[t * hd..][..hd],
                &v,
            ));
        }
        for t in 0..tokens {
            let tau = 1.0 + 0.5 * ((t as f64 + 1.0) / 2.0).max(1.0).ln();
            for h in 0..nh {
                let mut q = row(t, h * hd, hd);
                clip(&mut q, &scales[h * hd..][..hd], 1.5, tau);
                let mut r = row(t, rel_at + h * n_rel, n_rel);
                clip(&mut r, &scales[nh * hd + h * n_rel..][..n_rel], 0.8, tau);
                let what = format!("{window:?} token {t} head {h}");
                close(
                    &format!("{what} queries"),
                    &readied[t * stride + h * hd..][..hd],
                    &q,
                );
                let sink = qkv[t * stride + sink_at + h] as f64;
                let first = window.map_or(0, |w| (t + 1).saturating_sub(w));
                let mut scores: Vec<f64> = (first..=t)
                    .map(|p| {
                        let dot: f64 = q.iter().zip(&keys[p]).map(|(a, b)| a * b).sum();
                        let dist = t - p;
                        let bias: f64 = match dist < extent {
                            true => (0..n_rel)
                                .map(|i| rel_bias[dist * n_rel + i] as f64 * r[i])
                                .sum(),
                            false => 0.0,
                        };
                        dot / (hd as f64).sqrt() + bias - sink
                    })
                    .collect();
                scores.push(0.0);
                let top = scores.iter().copied().fold(f64::MIN, f64::max);
                let total: f64 = scores.iter().map(|s| (s - top).exp()).sum();
                let want: Vec<f64> = (0..hd)
                    .map(|e| {
                        let o: f64 = (first..=t)
                            .zip(&scores)
                            .map(|(p, s)| (s - top).exp() / total * vals[p][e])
                            .sum();
                        let g = qkv[t * stride + gate_at + h * hd + e] as f64;
                        o / (1.0 + (-g).exp())
                    })
                    .collect();
                close(
                    &format!("{what} output"),
                    &got[(t * nh + h) * hd..][..hd],
                    &want,
                );
            }
        }
    }
}

/// The residual convolution with dynamic taps against the host formula.
#[test]
fn residual_conv_matches_the_reference() {
    let (len, taps, heads, tokens) = (64, 3, 2, 4);
    let head_len = len / heads;
    let stride = len + heads * taps + 8;
    let src = values(tokens * stride, 4, 1.0);
    let weights = values(len * taps, 5, 0.5);
    for gpu in gpus() {
        let dev = &gpu.device;
        let device_src = DeviceBuffer::from_slice(dev, &src).unwrap();
        let device_weights = DeviceBuffer::from_slice(dev, &weights).unwrap();
        let ring = ConvRing::new(dev, len, taps, tokens).unwrap();
        let out = DeviceBuffer::<f32>::new(dev, tokens * len).unwrap();
        let mut batch = Batch::new(dev, tokens, 1, 16, 0).unwrap();
        batch.set(tokens, 0, 0..0, None).unwrap();
        let mut c = Commands::new(dev).unwrap();
        batch.record_upload(&mut c).unwrap();
        let x = Strided {
            data: &device_src,
            stride,
            at: 0,
        };
        let y = Strided {
            data: &out,
            stride: len,
            at: 0,
        };
        let dynamic = Some((len, head_len));
        gpu.kernels
            .residual_conv(&batch, x, dynamic, &device_weights, &ring, y, false, &mut c)
            .unwrap();
        c.submit().unwrap();
        let got = out.to_vec().unwrap();
        let input = |t: usize, ch: usize| src[t * stride + ch] as f64;
        let mut want = Vec::new();
        for t in 0..tokens {
            for ch in 0..len {
                let mut acc = 0.0;
                for j in 0..taps {
                    let back = taps - 1 - j;
                    if t >= back {
                        acc += weights[ch * taps + j] as f64 * input(t - back, ch);
                    }
                }
                let lags = &src[t * stride + len + ch / head_len * taps..][..taps];
                for (lag, &l) in lags.iter().enumerate() {
                    if t >= lag {
                        acc += l as f64 * input(t - lag, ch);
                    }
                }
                want.push((input(t, ch) + acc) / ((2 * taps + 1) as f64).sqrt());
            }
        }
        close("residual convolution", &got, &want);
    }
}

/// The mix over depth against the host formula.
#[test]
fn depth_mix_matches_the_reference() {
    let (d, tokens, stacked) = (64, 3, 2);
    let stack = values(stacked * tokens * d, 6, 2.0);
    let cur = values(tokens * d, 7, 2.0);
    let query = values(d, 8, 1.0);
    for gpu in gpus() {
        let dev = &gpu.device;
        let device_stack = DeviceBuffer::from_slice(dev, &stack).unwrap();
        let device_cur = DeviceBuffer::from_slice(dev, &cur).unwrap();
        let device_query = DeviceBuffer::from_slice(dev, &query).unwrap();
        let out = DeviceBuffer::<f32>::new(dev, tokens * d).unwrap();
        let mut batch = Batch::new(dev, tokens, 1, 16, 0).unwrap();
        batch.set(tokens, 0, 0..0, None).unwrap();
        let mut c = Commands::new(dev).unwrap();
        batch.record_upload(&mut c).unwrap();
        let states = States {
            stack: &device_stack,
            count: stacked,
            stride: tokens * d,
            cur: Some(&device_cur),
        };
        gpu.kernels
            .depth_mix(&batch, &states, &device_query, &out, &mut c)
            .unwrap();
        c.submit().unwrap();
        let got = out.to_vec().unwrap();
        let mut want = Vec::new();
        for t in 0..tokens {
            let rows: Vec<Vec<f64>> = (0..stacked)
                .map(|s| &stack[(s * tokens + t) * d..][..d])
                .chain([&cur[t * d..][..d]])
                .map(|r| {
                    let ms = r.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / d as f64;
                    r.iter().map(|&v| v as f64 / (ms + 1e-6).sqrt()).collect()
                })
                .collect();
            let scores: Vec<f64> = rows
                .iter()
                .map(|r| {
                    r.iter()
                        .zip(&query)
                        .map(|(a, &q)| a * q as f64)
                        .sum::<f64>()
                        / (d as f64).sqrt()
                })
                .collect();
            let top = scores.iter().copied().fold(f64::MIN, f64::max);
            let total: f64 = scores.iter().map(|s| (s - top).exp()).sum();
            for e in 0..d {
                want.push(
                    rows.iter()
                        .zip(&scores)
                        .map(|(r, s)| (s - top).exp() / total * r[e])
                        .sum(),
                );
            }
        }
        close("mix over depth", &got, &want);
    }
}
