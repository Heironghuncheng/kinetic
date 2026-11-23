//! Tiny Inkling 2 models built in memory: every other layer windowed,
//! blocks of two layers and a core of one block that runs twice.

#![allow(dead_code)]

use kinetic_gguf::{Gguf, TensorInfo, TensorOut, TensorSource, TensorType, Value};

/// A file held in memory.
pub struct Memory {
    pub header: Gguf,
    bytes: Vec<u8>,
}

impl TensorSource for Memory {
    fn header(&self) -> &Gguf {
        &self.header
    }

    fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        &self.bytes[t.offset as usize..][..t.size as usize]
    }
}

pub const N_PHYS: usize = 6;
pub const LOOP: std::ops::Range<usize> = 2..4;

/// The widths of a tiny model.
pub struct Size {
    pub d: u64,
    pub head_dim: u64,
    pub n_ff: u64,
    pub rank: u64,
    /// The step between weight values, smaller for wider models so a
    /// layer's gain stays about that of the reference size.
    pub step: f32,
}

/// The size the reference logits were computed for.
pub const REFERENCE: Size = Size {
    d: 8,
    head_dim: 4,
    n_ff: 4,
    rank: 3,
    step: 0.07,
};

/// The smallest width whose rows are whole blocks of every weight type.
pub const TYPED: Size = Size {
    d: 256,
    head_dim: 64,
    n_ff: 256,
    rank: 32,
    step: 0.01,
};

/// The smallest size every device runs: widths of whole blocks of 32 and
/// heads of 64.
pub const DEVICE: Size = Size {
    d: 64,
    head_dim: 64,
    n_ff: 32,
    rank: 32,
    step: 0.025,
};

/// The tensors of a tiny looped model of size `s`.
fn tensors(s: &Size) -> Vec<(String, Vec<u64>)> {
    let (d, hd, ff, rank) = (s.d, s.head_dim, s.n_ff, s.rank);
    let (heads, rel, taps, experts) = (2, 2, 3, 4);
    let mut t: Vec<(String, Vec<u64>)> = [
        ("token_embd.weight", vec![d, 12]),
        ("token_embd_norm.weight", vec![d]),
        ("output.weight", vec![d, 12]),
        ("output_norm.weight", vec![d]),
        ("output_res_query.weight", vec![d]),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect();
    for i in 0..N_PHYS {
        let windowed = i % 2 == 0;
        let mut add = |name: &str, dims: Vec<u64>| t.push((format!("blk.{i}.{name}"), dims));
        add("attn_q.weight", vec![d, heads * hd]);
        add("attn_k.weight", vec![d, hd]);
        add("attn_v.weight", vec![d, hd]);
        add("attn_output.weight", vec![heads * hd, d]);
        add("attn_gate.weight", vec![d, heads * hd]);
        add("attn_sink_proj.weight", vec![d, heads]);
        add("attn_q_clip.weight", vec![hd, heads]);
        add("attn_rel.weight", vec![d, rel * heads]);
        add(
            "attn_rel_bias.weight",
            vec![rel, if windowed { 2 } else { 3 }],
        );
        add("attn_rel_clip.weight", vec![rel, heads]);
        add("attn_k_conv.weight", vec![taps, hd]);
        add("attn_v_conv.weight", vec![taps, hd]);
        add("attn_post_conv.weight", vec![taps, d]);
        add("ffn_post_conv.weight", vec![taps, d]);
        if windowed {
            add("attn_norm.weight", vec![d]);
        } else {
            add("attn_q_inp_norm.weight", vec![d]);
            add("attn_k_inp_norm.weight", vec![d]);
            add("attn_v_inp_norm.weight", vec![d]);
            add("attn_k_conv_proj.weight", vec![d, taps]);
            add("attn_v_conv_proj.weight", vec![d, taps]);
        }
        if i > 0 {
            add(
                "attn_res_query.weight",
                if windowed { vec![d] } else { vec![d, 3] },
            );
        }
        add("ffn_res_query.weight", vec![d]);
        add("ffn_norm.weight", vec![d]);
        add("ffn_gate_inp.weight", vec![d, experts]);
        add("exp_probs_b.bias", vec![experts]);
        add("ffn_gate_up_exps.weight", vec![d, 2 * ff, experts]);
        add("ffn_down_exps.weight", vec![ff, d, experts]);
        if LOOP.contains(&i) {
            add("loop_msg_query.weight", vec![d, 4]);
            for x in ["q", "k", "v", "ffn"] {
                add(&format!("loop_{x}_ctrl_down.weight"), vec![2 * d, rank]);
                add(&format!("loop_{x}_ctrl_gain.weight"), vec![rank, d]);
                add(&format!("loop_{x}_ctrl_forget.weight"), vec![rank, d]);
            }
        }
    }
    t
}

/// Hyperparameters of a tiny model of size `s` with `slots` layers listed,
/// without those of its loop.
pub fn keys(s: &Size, slots: usize) -> Vec<(&'static str, Value)> {
    let per_layer = |v: Value| Value::Array(vec![v; slots]);
    vec![
        ("block_count", Value::U32(slots as u32)),
        ("context_length", Value::U32(64)),
        ("embedding_length", Value::U32(s.d as u32)),
        ("feed_forward_length", Value::U32(s.n_ff as u32)),
        ("attention.head_count", per_layer(Value::U32(2))),
        ("attention.head_count_kv", per_layer(Value::U32(1))),
        ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("expert_count", Value::U32(4)),
        ("expert_used_count", Value::U32(2)),
        ("attention.key_length", Value::U32(s.head_dim as u32)),
        ("attention.value_length", Value::U32(s.head_dim as u32)),
        ("vocab_size", Value::U32(12)),
        ("vocab_size_unpadded", Value::U32(10)),
        ("expert_feed_forward_length", Value::U32(s.n_ff as u32)),
        ("rope.dimension_count", Value::U32(0)),
        ("attention.sliding_window", Value::U32(2)),
        (
            "attention.sliding_window_pattern",
            Value::Array((0..slots).map(|i| Value::Bool(i % 2 == 0)).collect()),
        ),
        ("shortconv.l_cache", Value::U32(3)),
        ("expert_weights_scale", Value::F32(2.0)),
        ("activation.situ_beta", Value::F32(4.0)),
        ("activation.situ_linear_beta", Value::F32(25.0)),
        ("attn_res.block_size", Value::U32(2)),
        ("final_logit_softcapping", Value::F32(30.0)),
        ("attention.relative_length", Value::U32(2)),
        ("attention.relative_extent", Value::U32(3)),
        ("attention.qk_clip", Value::F32(1.5)),
        ("attention.relative_clip", Value::F32(0.8)),
        ("attention.log_scale_floor", Value::U32(2)),
        ("attention.log_scale_alpha", Value::F32(0.5)),
        ("loop.gate_rank", Value::U32(s.rank as u32)),
    ]
}

/// The tensors of a tiny model of size `s` with their values, the same in
/// the NumPy script the reference logits come from.
pub fn weights(s: &Size) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    tensors(s)
        .into_iter()
        .enumerate()
        .map(|(k, (name, dims))| {
            let n: u64 = dims.iter().product();
            let shift = if name.contains("norm") || name.contains("_clip") {
                1.0
            } else {
                0.0
            };
            let values = (0..n as usize)
                .map(|i| (((i * (2 * k + 3)) % 17) as f32 - 8.0) * s.step + shift)
                .collect();
            (name, dims, values)
        })
        .collect()
}

/// One tensor of a file: its name, dimensions, type and bytes.
pub type Tensor = (String, Vec<u64>, TensorType, Vec<u8>);

pub fn write(arch: &str, keys: Vec<(&str, Value)>, tensors: &[Tensor]) -> Memory {
    let mut metadata = vec![(
        "general.architecture".to_string(),
        Value::String(arch.into()),
    )];
    metadata.extend(keys.into_iter().map(|(k, v)| (format!("{arch}.{k}"), v)));
    let out: Vec<TensorOut> = tensors
        .iter()
        .map(|(name, dims, ty, _)| TensorOut {
            name: name.clone(),
            dims: dims.clone(),
            ty: *ty,
        })
        .collect();
    let mut bytes = Vec::new();
    kinetic_gguf::write(&mut bytes, &metadata, &out, |k| Ok(tensors[k].3.clone())).unwrap();
    Memory {
        header: Gguf::parse(&bytes).unwrap(),
        bytes,
    }
}

/// The tensors that take the weight type under test: the experts, and the
/// query, key and value projections.
fn typed(name: &str) -> bool {
    [
        "_exps.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
    ]
    .iter()
    .any(|s| name.ends_with(s))
}

/// Blocks of `ty` holding `n` values, random but for their scales, which
/// give the values a root mean square near 0.01 whatever the type's grid;
/// floats are drawn directly.
pub fn random_blocks(ty: TensorType, n: usize, seed: u64) -> Vec<u8> {
    let first = blocks(ty, n, seed, 1e-3);
    if matches!(ty, TensorType::MXFP4 | TensorType::NVFP4) || ty.block_len() == 1 {
        return first;
    }
    let mut values = vec![0.0; n];
    kinetic_quant::dequantize(ty, &first, &mut values).unwrap();
    let rms = (values.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
    blocks(ty, n, seed, 1e-3 * 0.01 / rms)
}

/// [`random_blocks`] with f16 block scales near `d`.
fn blocks(ty: TensorType, n: usize, seed: u64, d: f32) -> Vec<u8> {
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut small = || (next() as f32 / u32::MAX as f32 - 0.5) * 0.04;
    match ty {
        TensorType::F32 => return (0..n).flat_map(|_| small().to_le_bytes()).collect(),
        TensorType::F16 => {
            return (0..n)
                .flat_map(|_| half::f16::from_f32(small()).to_le_bytes())
                .collect()
        }
        TensorType::BF16 => {
            return (0..n)
                .flat_map(|_| half::bf16::from_f32(small()).to_le_bytes())
                .collect()
        }
        _ => {}
    }
    let size = ty.block_bytes();
    let mut data: Vec<u8> = (0..n / ty.block_len() * size)
        .map(|_| next() as u8)
        .collect();
    for blk in data.chunks_exact_mut(size) {
        let scale = half::f16::from_f32(d * (1.0 + (next() % 1000) as f32 * 1e-3));
        match ty {
            // the f16 scale in the top nibbles of the four scale words
            TensorType::IQ1M => {
                let bits = scale.to_bits();
                for i in 0..4 {
                    let at = 48 + 2 * i;
                    let w = u16::from_le_bytes([blk[at], blk[at + 1]]) & 0x0FFF;
                    let w = w | ((bits >> (4 * i)) & 0xF) << 12;
                    blk[at..at + 2].copy_from_slice(&w.to_le_bytes());
                }
            }
            // an E8M0 scale of 2^-9, halved
            TensorType::MXFP4 => blk[0] = 119,
            // E4M3 scales between 2^-8 and 2^-7
            TensorType::NVFP4 => blk[..4]
                .iter_mut()
                .for_each(|b| *b = 0x08 + (next() % 8) as u8),
            _ => {
                let at: &[usize] = match ty {
                    TensorType::Q4K | TensorType::Q5K => &[0, 2],
                    TensorType::Q6K => &[208],
                    TensorType::Q3K => &[108],
                    _ => &[0],
                };
                for &o in at {
                    blk[o..o + 2].copy_from_slice(&scale.to_le_bytes());
                }
            }
        }
    }
    data
}

/// The tensors of a tiny model of size `s` with the typed ones in `ty`,
/// the others in f32.
pub fn tensors_of(s: &Size, ty: TensorType) -> Vec<Tensor> {
    weights(s)
        .into_iter()
        .enumerate()
        .map(
            |(k, (name, dims, values))| match typed(&name) && ty != TensorType::F32 {
                true => {
                    let bytes = random_blocks(ty, values.len(), k as u64);
                    (name, dims, ty, bytes)
                }
                false => {
                    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    (name, dims, TensorType::F32, bytes)
                }
            },
        )
        .collect()
}

/// A tiny looped model of size `s` in ggml-org's layout.
pub fn model(s: &Size) -> Memory {
    model_of(s, TensorType::F32)
}

/// [`model`] with the experts and the query, key and value projections in
/// `ty`.
pub fn model_of(s: &Size, ty: TensorType) -> Memory {
    let mut keys = keys(s, N_PHYS);
    keys.extend([
        ("loop.start", Value::U32(LOOP.start as u32)),
        ("loop.end", Value::U32(LOOP.end as u32)),
        ("loop.count", Value::U32(2)),
        ("loop.varying", Value::Bool(false)),
    ]);
    write("inkling2_loop", keys, &tensors_of(s, ty))
}

/// [`model_of`] as Unsloth lays it out: `inkling2` with every pass of the
/// core in `block_count` and the per-layer arrays, its own loop keys, the
/// controls named `attn`, `k`, `v` and `mlp`, and the message queries of
/// all core layers in two tables. The renaming is what the bytes of
/// Unsloth's files and of their copies in Mainline_Support show.
pub fn unrolled_of(s: &Size, ty: TensorType) -> Memory {
    let d = s.d as usize;
    let mut keys = keys(s, N_PHYS + LOOP.len());
    keys.extend([
        ("num_loops", Value::U32(2)),
        ("loop.core_start", Value::U32(LOOP.start as u32)),
        ("loop.core_end", Value::U32(LOOP.end as u32)),
    ]);
    let mut tensors = Vec::new();
    let (mut res, mut kv) = (Vec::new(), Vec::new());
    for (name, dims, t, bytes) in tensors_of(s, ty) {
        if name.ends_with("loop_msg_query.weight") {
            let q = |j: usize| &bytes[4 * j * d..][..4 * d];
            res.extend([q(0), q(3)].concat());
            kv.extend([q(1), q(2)].concat());
            continue;
        }
        let renamed = ["q", "k", "v", "ffn"]
            .into_iter()
            .zip(["attn", "k", "v", "mlp"])
            .fold(name, |n, (from, to)| {
                n.replace(&format!("loop_{from}_ctrl_"), &format!("loop_{to}_"))
            });
        tensors.push((renamed, dims, t, bytes));
    }
    let rows = 2 * LOOP.len() as u64;
    let f32 = TensorType::F32;
    tensors.push(("loop_res_query.weight".into(), vec![s.d, rows], f32, res));
    tensors.push(("loop_kv_res_query.weight".into(), vec![s.d, rows], f32, kv));
    write("inkling2", keys, &tensors)
}
