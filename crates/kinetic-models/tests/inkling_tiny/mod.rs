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

pub fn write(
    arch: &str,
    keys: Vec<(&str, Value)>,
    tensors: &[(String, Vec<u64>, Vec<f32>)],
) -> Memory {
    let mut metadata = vec![(
        "general.architecture".to_string(),
        Value::String(arch.into()),
    )];
    metadata.extend(keys.into_iter().map(|(k, v)| (format!("{arch}.{k}"), v)));
    let out: Vec<TensorOut> = tensors
        .iter()
        .map(|(name, dims, _)| TensorOut {
            name: name.clone(),
            dims: dims.clone(),
            ty: TensorType::F32,
        })
        .collect();
    let mut bytes = Vec::new();
    kinetic_gguf::write(&mut bytes, &metadata, &out, |k| {
        Ok(tensors[k].2.iter().flat_map(|v| v.to_le_bytes()).collect())
    })
    .unwrap();
    Memory {
        header: Gguf::parse(&bytes).unwrap(),
        bytes,
    }
}

/// A tiny looped model of size `s` in ggml-org's layout.
pub fn model(s: &Size) -> Memory {
    let mut keys = keys(s, N_PHYS);
    keys.extend([
        ("loop.start", Value::U32(LOOP.start as u32)),
        ("loop.end", Value::U32(LOOP.end as u32)),
        ("loop.count", Value::U32(2)),
        ("loop.varying", Value::Bool(false)),
    ]);
    write("inkling2_loop", keys, &weights(s))
}
