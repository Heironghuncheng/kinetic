//! Inkling 2 on CUDA: a tiny looped model held in memory, its experts in
//! IQ4_NL and its projections in Q8_0 as in the published file, against the
//! CPU, and the GPU path's exactness: batches, rollbacks, host copies,
//! shared passes and speculative decoding change no bit.
//!
//! cargo test --release -p kinetic-models --features cuda --test inkling_cuda -- --ignored

#![cfg(feature = "cuda")]

use half::f16;
use kinetic_gguf::{Gguf, TensorInfo, TensorOut, TensorSource, TensorType, Value};
use kinetic_models::cuda::{CudaModel, CudaState, Run};
use kinetic_models::speculative::{Decoder, Greedy, Lookup, Policy};

struct Memory {
    header: Gguf,
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

const D: u64 = 64;
const N_PHYS: usize = 6;
const LOOP: std::ops::Range<usize> = 2..4;
const VOCAB: u64 = 64;
/// Long enough for the windowed layers to wrap their caches.
const WINDOW: u32 = 4;

/// The tensors of the model and their types: every other layer windowed,
/// blocks of two layers and a core of one block that runs three times.
fn tensors() -> Vec<(String, Vec<u64>, TensorType)> {
    let (heads, hd, rel, taps, experts, ff, rank) = (2, 64, 2, 4, 4, 96, 32);
    let (f32, q8, iq4) = (TensorType::F32, TensorType::Q8_0, TensorType::IQ4NL);
    let mut t: Vec<(String, Vec<u64>, TensorType)> = vec![
        ("token_embd.weight".into(), vec![D, VOCAB], q8),
        ("token_embd_norm.weight".into(), vec![D], f32),
        ("output.weight".into(), vec![D, VOCAB], q8),
        ("output_norm.weight".into(), vec![D], f32),
        ("output_res_query.weight".into(), vec![D], f32),
    ];
    for i in 0..N_PHYS {
        let windowed = i % 2 == 0;
        let mut add =
            |name: &str, dims: Vec<u64>, ty| t.push((format!("blk.{i}.{name}"), dims, ty));
        add("attn_q.weight", vec![D, heads * hd], q8);
        add("attn_k.weight", vec![D, hd], q8);
        add("attn_v.weight", vec![D, hd], q8);
        add("attn_output.weight", vec![heads * hd, D], q8);
        add("attn_gate.weight", vec![D, heads * hd], q8);
        add("attn_sink_proj.weight", vec![D, heads], q8);
        add("attn_q_clip.weight", vec![hd, heads], f32);
        add("attn_rel.weight", vec![D, rel * heads], q8);
        let extent = if windowed { WINDOW as u64 } else { 6 };
        add("attn_rel_bias.weight", vec![rel, extent], f32);
        add("attn_rel_clip.weight", vec![rel, heads], f32);
        add("attn_k_conv.weight", vec![taps, hd], f32);
        add("attn_v_conv.weight", vec![taps, hd], f32);
        add("attn_post_conv.weight", vec![taps, D], f32);
        add("ffn_post_conv.weight", vec![taps, D], f32);
        if windowed {
            add("attn_norm.weight", vec![D], f32);
        } else {
            add("attn_q_inp_norm.weight", vec![D], f32);
            add("attn_k_inp_norm.weight", vec![D], f32);
            add("attn_v_inp_norm.weight", vec![D], f32);
            add("attn_k_conv_proj.weight", vec![D, taps], q8);
            add("attn_v_conv_proj.weight", vec![D, taps], q8);
        }
        if i > 0 {
            let query = if windowed { vec![D] } else { vec![D, 3] };
            add("attn_res_query.weight", query, f32);
        }
        add("ffn_res_query.weight", vec![D], f32);
        add("ffn_norm.weight", vec![D], f32);
        add("ffn_gate_inp.weight", vec![D, experts], f32);
        add("exp_probs_b.bias", vec![experts], f32);
        add("ffn_gate_up_exps.weight", vec![D, 2 * ff, experts], iq4);
        add("ffn_down_exps.weight", vec![ff, D, experts], iq4);
        if LOOP.contains(&i) {
            add("loop_msg_query.weight", vec![D, 4], f32);
            for x in ["q", "k", "v", "ffn"] {
                add(
                    &format!("loop_{x}_ctrl_down.weight"),
                    vec![2 * D, rank],
                    f32,
                );
                add(&format!("loop_{x}_ctrl_gain.weight"), vec![rank, D], f32);
                add(&format!("loop_{x}_ctrl_forget.weight"), vec![rank, D], f32);
            }
        }
    }
    t
}

/// Small values that vary with the tensor `k` and position `i`; norms and
/// clips around one.
fn data(k: usize, name: &str, n: usize, ty: TensorType) -> Vec<u8> {
    let step = |i: usize| ((i * (2 * k + 3)) % 17) as i32 - 8;
    match ty {
        TensorType::Q8_0 => (0..n / 32)
            .flat_map(|b| {
                let mut block = f16::from_f32(0.07).to_le_bytes().to_vec();
                block.extend((0..32).map(|j| step(b * 32 + j) as i8 as u8));
                block
            })
            .collect(),
        TensorType::IQ4NL => (0..n / 32)
            .flat_map(|b| {
                let mut block = f16::from_f32(0.001).to_le_bytes().to_vec();
                block.extend((0..16).map(|j| ((b * 16 + j) * (2 * k + 5) % 251) as u8));
                block
            })
            .collect(),
        _ => {
            let shift = if name.contains("norm") || name.contains("_clip") {
                1.0
            } else {
                0.0
            };
            (0..n)
                .flat_map(|i| (step(i) as f32 * 0.07 + shift).to_le_bytes())
                .collect()
        }
    }
}

fn tiny_model() -> Memory {
    let arch = "inkling2_loop";
    let per_layer = |v: Value| Value::Array(vec![v; N_PHYS]);
    let mut metadata = vec![(
        "general.architecture".to_string(),
        Value::String(arch.into()),
    )];
    let keys = [
        ("block_count", Value::U32(N_PHYS as u32)),
        ("context_length", Value::U32(128)),
        ("embedding_length", Value::U32(D as u32)),
        ("feed_forward_length", Value::U32(96)),
        ("attention.head_count", per_layer(Value::U32(2))),
        ("attention.head_count_kv", per_layer(Value::U32(1))),
        ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("expert_count", Value::U32(4)),
        // every expert, so that rounding never changes which ones run and the
        // comparison with the CPU stays smooth
        ("expert_used_count", Value::U32(4)),
        ("attention.key_length", Value::U32(64)),
        ("attention.value_length", Value::U32(64)),
        ("vocab_size", Value::U32(VOCAB as u32)),
        ("vocab_size_unpadded", Value::U32(60)),
        ("expert_feed_forward_length", Value::U32(96)),
        ("rope.dimension_count", Value::U32(0)),
        ("attention.sliding_window", Value::U32(WINDOW)),
        (
            "attention.sliding_window_pattern",
            Value::Array((0..N_PHYS).map(|i| Value::Bool(i % 2 == 0)).collect()),
        ),
        ("shortconv.l_cache", Value::U32(4)),
        ("expert_weights_scale", Value::F32(2.0)),
        ("activation.situ_beta", Value::F32(4.0)),
        ("activation.situ_linear_beta", Value::F32(25.0)),
        ("attn_res.block_size", Value::U32(2)),
        ("final_logit_softcapping", Value::F32(30.0)),
        ("attention.relative_length", Value::U32(2)),
        ("attention.relative_extent", Value::U32(6)),
        ("attention.qk_clip", Value::F32(1.5)),
        ("attention.relative_clip", Value::F32(0.8)),
        ("attention.log_scale_floor", Value::U32(2)),
        ("attention.log_scale_alpha", Value::F32(0.5)),
        ("loop.start", Value::U32(LOOP.start as u32)),
        ("loop.end", Value::U32(LOOP.end as u32)),
        ("loop.count", Value::U32(3)),
        ("loop.gate_rank", Value::U32(32)),
        ("loop.varying", Value::Bool(false)),
    ];
    metadata.extend(keys.into_iter().map(|(k, v)| (format!("{arch}.{k}"), v)));
    let tensors = tensors();
    let out: Vec<TensorOut> = tensors
        .iter()
        .map(|(name, dims, ty)| TensorOut {
            name: name.clone(),
            dims: dims.clone(),
            ty: *ty,
        })
        .collect();
    let mut bytes = Vec::new();
    kinetic_gguf::write(&mut bytes, &metadata, &out, |k| {
        let (name, dims, ty) = &tensors[k];
        Ok(data(k, name, dims.iter().product::<u64>() as usize, *ty))
    })
    .unwrap();
    Memory {
        header: Gguf::parse(&bytes).unwrap(),
        bytes,
    }
}

const PROMPT: [u32; 12] = [1, 40, 10, 23, 50, 7, 31, 12, 33, 6, 45, 24];
const NEXT: [u32; 6] = [30, 3, 59, 7, 17, 9];

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// One test, so one model on the device: capturing the decode graph on one
/// thread forbids other threads' synchronous copies in the same context.
#[test]
#[ignore = "needs a CUDA device"]
fn inkling_on_cuda_matches_the_cpu_and_changes_no_bit() {
    let file = tiny_model();
    let model = kinetic_models::load(&file).unwrap();
    let gpu = CudaModel::load(&model, 128).unwrap();
    assert!(
        gpu.plan()
            .is_some_and(|p| p.pools.iter().all(|p| p.slots == p.layers * 4)),
        "every expert on the device"
    );
    logits_match_the_cpu(&gpu, &model);
    a_batch_matches_token_by_token(&gpu);
    rollback_leaves_the_state_as_if_never_run(&gpu);
    a_host_copy_restores_the_state(&gpu, &model);
    a_reset_state_is_a_fresh_one(&gpu);
    shared_passes_match_running_alone(&gpu);
    speculative_greedy_is_plain_greedy(&gpu);
}

/// The GPU's logits after every token of a prompt and of single steps are
/// the CPU's within the rounding of the device's f16 cache and its own
/// order of sums.
fn logits_match_the_cpu(gpu: &CudaModel, model: &kinetic_models::Model) {
    let mut cpu = model.new_state();
    let mut want = model.run(&mut cpu, &PROMPT, true).logits;
    let mut state = gpu.new_state().unwrap();
    let mut got = gpu.run(&mut state, &PROMPT, true).unwrap().logits;
    for &t in &NEXT {
        want.extend(model.run(&mut cpu, &[t], true).logits);
        got.extend(gpu.run(&mut state, &[t], true).unwrap().logits);
    }
    assert_eq!(got.len(), want.len());
    let vocab = gpu.n_vocab();
    let rows: Vec<f32> = got
        .chunks_exact(vocab)
        .zip(want.chunks_exact(vocab))
        .map(|(g, w)| {
            g.iter()
                .zip(w)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0, f32::max)
        })
        .collect();
    let worst = rows.iter().fold(0.0f32, |m, &r| m.max(r));
    let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
    eprintln!("largest logit difference from the CPU per token: {rows:?} (logits up to {scale})");
    assert!(
        worst <= 0.01 * scale.max(1.0),
        "{worst} against logits up to {scale}"
    );
}

fn a_batch_matches_token_by_token(gpu: &CudaModel) {
    let vocab = gpu.n_vocab();
    let mut batched = gpu.new_state().unwrap();
    gpu.run(&mut batched, &PROMPT, false).unwrap();
    let all = gpu.run(&mut batched, &NEXT, true).unwrap().logits;

    let mut single = gpu.new_state().unwrap();
    gpu.run(&mut single, &PROMPT, false).unwrap();
    for (i, &t) in NEXT.iter().enumerate() {
        let logits = gpu.run(&mut single, &[t], false).unwrap().logits;
        assert_eq!(bits(&logits), bits(&all[i * vocab..][..vocab]), "row {i}");
    }
}

fn rollback_leaves_the_state_as_if_never_run(gpu: &CudaModel) {
    let mut rolled = gpu.new_state().unwrap();
    gpu.run(&mut rolled, &PROMPT, false).unwrap();
    gpu.checkpoint(&mut rolled).unwrap();
    gpu.run(&mut rolled, &NEXT, true).unwrap();
    gpu.rollback(&mut rolled, 2).unwrap();
    let after_rollback = gpu.run(&mut rolled, &[42], false).unwrap().logits;

    let mut plain = gpu.new_state().unwrap();
    gpu.run(&mut plain, &PROMPT, false).unwrap();
    gpu.run(&mut plain, &NEXT[..2], false).unwrap();
    let expected = gpu.run(&mut plain, &[42], false).unwrap().logits;
    assert_eq!(rolled.len(), plain.len());
    assert_eq!(bits(&after_rollback), bits(&expected));
}

fn a_host_copy_restores_the_state(gpu: &CudaModel, model: &kinetic_models::Model) {
    let mut state = gpu.new_state().unwrap();
    gpu.run(&mut state, &PROMPT, false).unwrap();
    let host = gpu.host_state(model, &state).unwrap();
    let mut restored = gpu.device_state(&host).unwrap();
    let expected = gpu.run(&mut state, &NEXT, true).unwrap().logits;
    let logits = gpu.run(&mut restored, &NEXT, true).unwrap().logits;
    assert_eq!(bits(&logits), bits(&expected));
}

fn a_reset_state_is_a_fresh_one(gpu: &CudaModel) {
    let mut fresh = gpu.new_state().unwrap();
    let expected = gpu.run(&mut fresh, &NEXT, true).unwrap().logits;
    let mut reused = gpu.new_state().unwrap();
    gpu.run(&mut reused, &PROMPT, false).unwrap();
    gpu.reset_state(&mut reused).unwrap();
    let logits = gpu.run(&mut reused, &NEXT, true).unwrap().logits;
    assert_eq!(bits(&logits), bits(&expected));
}

/// Sequences run together, one verifying a draft inside a checkpoint, get
/// exactly what each gets alone.
fn shared_passes_match_running_alone(gpu: &CudaModel) {
    let prompts: [&[u32]; 2] = [&PROMPT, &NEXT];
    let fresh = || -> Vec<CudaState> {
        prompts
            .iter()
            .map(|p| {
                let mut s = gpu.new_state().unwrap();
                gpu.run(&mut s, p, false).unwrap();
                s
            })
            .collect()
    };
    let (mut shared, mut alone) = (fresh(), fresh());
    gpu.checkpoint(&mut shared[1]).unwrap();
    gpu.checkpoint(&mut alone[1]).unwrap();
    let tokens: [&[u32]; 2] = [&[42], &[7, 8, 9]];
    let mut runs: Vec<Run> = shared
        .iter_mut()
        .zip(tokens)
        .map(|(state, tokens)| Run {
            state,
            tokens,
            all_rows: true,
            logits: true,
        })
        .collect();
    let got = gpu.run_many(&mut runs).unwrap();
    for ((state, tokens), ran) in alone.iter_mut().zip(tokens).zip(&got) {
        let want = gpu.run(state, tokens, true).unwrap();
        assert_eq!(bits(&ran.forward.logits), bits(&want.logits));
    }
}

fn speculative_greedy_is_plain_greedy(gpu: &CudaModel) {
    let prompt: Vec<u32> = PROMPT.repeat(3);
    let n = 24;
    let mut state = gpu.new_state().unwrap();
    let mut context = prompt.clone();
    let (mut scores, _) = Greedy.score(gpu, &mut state, &prompt, false);
    while context.len() < prompt.len() + n {
        let t = Greedy.choose(gpu, &scores, 0, &context);
        context.push(t);
        scores = Greedy.score(gpu, &mut state, &[t], false).0;
    }
    let want = context.split_off(prompt.len());

    let mut decoder = Decoder::new(Box::new(Lookup::default()));
    decoder.row_cost = f32::INFINITY;
    let mut state = gpu.new_state().unwrap();
    let mut context = prompt.clone();
    decoder.prefill(gpu, &mut Greedy, &mut state, &context, 0);
    while context.len() < prompt.len() + n {
        let limit = prompt.len() + n - context.len();
        decoder.step(gpu, &mut Greedy, &mut state, &mut context, limit, |_| false);
    }
    assert_eq!(context.split_off(prompt.len()), want);
}
