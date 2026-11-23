//! Inkling 2 on CUDA: a tiny looped model held in memory, its experts in
//! IQ4_NL and its projections in Q8_0 as in the published file, against the
//! CPU, and the GPU path's exactness: batches, rollbacks, host copies,
//! shared passes and speculative decoding change no bit.
//!
//! cargo test --release -p kinetic-models --features cuda --test inkling_cuda -- --ignored

#![cfg(feature = "cuda")]

use half::f16;
use kinetic_gguf::{Gguf, TensorInfo, TensorOut, TensorSource, TensorType, Value};
use std::cell::Cell;

use kinetic_models::cuda::{CudaModel, CudaRunner, CudaState, Placement, Run};
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

const N_PHYS: usize = 6;
const LOOP: std::ops::Range<usize> = 2..4;
const VOCAB: u64 = 64;
/// Long enough for the windowed layers to wrap their caches.
const WINDOW: u32 = 4;

/// Passes of the core.
const PASSES: usize = 3;

/// A variant of the tiny model: the types of its dense weights and of its
/// experts, how many of the four experts each token uses, and whether the
/// file unrolls the loop as Unsloth's do.
#[derive(Clone, Copy, Debug)]
struct Tiny {
    dense: TensorType,
    experts: TensorType,
    used: u32,
    unrolled: bool,
    /// Widths that weight types in blocks of 256 fit: a width of 256 and
    /// four heads, rather than 64 and two.
    wide: bool,
}

impl Tiny {
    /// The width, query heads and expert width.
    fn dims(&self) -> (u64, u64, u64) {
        match self.wide {
            true => (256, 4, 256),
            false => (64, 2, 128),
        }
    }
}

/// The tensors of the model in the looped layout, with their types: every
/// other layer windowed, blocks of two layers and a core of one block that
/// runs three times. NVFP4 experts come with per-expert scales.
fn tensors(c: &Tiny) -> Vec<(String, Vec<u64>, TensorType)> {
    let (d, heads, ff) = c.dims();
    let (hd, rel, taps, experts, rank) = (64, 2, 4, 4, 32);
    let (f32, q8, exps) = (TensorType::F32, c.dense, c.experts);
    let mut t: Vec<(String, Vec<u64>, TensorType)> = vec![
        ("token_embd.weight".into(), vec![d, VOCAB], q8),
        ("token_embd_norm.weight".into(), vec![d], f32),
        ("output.weight".into(), vec![d, VOCAB], q8),
        ("output_norm.weight".into(), vec![d], f32),
        ("output_res_query.weight".into(), vec![d], f32),
    ];
    for i in 0..N_PHYS {
        let windowed = i % 2 == 0;
        let mut add =
            |name: &str, dims: Vec<u64>, ty| t.push((format!("blk.{i}.{name}"), dims, ty));
        add("attn_q.weight", vec![d, heads * hd], q8);
        add("attn_k.weight", vec![d, hd], q8);
        add("attn_v.weight", vec![d, hd], q8);
        add("attn_output.weight", vec![heads * hd, d], q8);
        add("attn_gate.weight", vec![d, heads * hd], q8);
        add("attn_sink_proj.weight", vec![d, heads], q8);
        add("attn_q_clip.weight", vec![hd, heads], f32);
        add("attn_rel.weight", vec![d, rel * heads], q8);
        let extent = if windowed { WINDOW as u64 } else { 6 };
        add("attn_rel_bias.weight", vec![rel, extent], f32);
        add("attn_rel_clip.weight", vec![rel, heads], f32);
        add("attn_k_conv.weight", vec![taps, hd], f32);
        add("attn_v_conv.weight", vec![taps, hd], f32);
        add("attn_post_conv.weight", vec![taps, d], f32);
        add("ffn_post_conv.weight", vec![taps, d], f32);
        if windowed {
            add("attn_norm.weight", vec![d], f32);
        } else {
            add("attn_q_inp_norm.weight", vec![d], f32);
            add("attn_k_inp_norm.weight", vec![d], f32);
            add("attn_v_inp_norm.weight", vec![d], f32);
            add("attn_k_conv_proj.weight", vec![d, taps], q8);
            add("attn_v_conv_proj.weight", vec![d, taps], q8);
        }
        if i > 0 {
            let query = if windowed { vec![d] } else { vec![d, 3] };
            add("attn_res_query.weight", query, f32);
        }
        add("ffn_res_query.weight", vec![d], f32);
        add("ffn_norm.weight", vec![d], f32);
        add("ffn_gate_inp.weight", vec![d, experts], f32);
        add("exp_probs_b.bias", vec![experts], f32);
        add("ffn_gate_up_exps.weight", vec![d, 2 * ff, experts], exps);
        add("ffn_down_exps.weight", vec![ff, d, experts], exps);
        if exps == TensorType::NVFP4 {
            add("ffn_gate_up_exps.scale", vec![experts], f32);
            add("ffn_down_exps.scale", vec![experts], f32);
        }
        if LOOP.contains(&i) {
            add("loop_msg_query.weight", vec![d, 4], f32);
            for x in ["q", "k", "v", "ffn"] {
                add(
                    &format!("loop_{x}_ctrl_down.weight"),
                    vec![2 * d, rank],
                    f32,
                );
                add(&format!("loop_{x}_ctrl_gain.weight"), vec![rank, d], f32);
                add(&format!("loop_{x}_ctrl_forget.weight"), vec![rank, d], f32);
            }
        }
    }
    t
}

/// Small values that vary with the tensor `k` and position `i`; norms,
/// clips and expert scales around one. The controls are kept gentle: with
/// full-size ones the looped model is chaotic, and a weight changed by a
/// part in a thousand moves the CPU's own logits as much as the device's
/// rounding does.
fn data(k: usize, name: &str, n: usize, ty: TensorType) -> Vec<u8> {
    let step = |i: usize| ((i * (2 * k + 3)) % 17) as i32 - 8;
    let codes =
        |b: usize, len: usize| (0..len).map(move |j| ((b * len + j) * (2 * k + 5) % 251) as u8);
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
                block.extend(codes(b, 16));
                block
            })
            .collect(),
        // E4M3 scale bytes 2 to 6 give 0.002 to 0.006
        TensorType::NVFP4 => (0..n / 64)
            .flat_map(|b| {
                (0..4)
                    .map(move |g| (2 + (b + g) % 5) as u8)
                    .chain(codes(b, 32))
            })
            .collect(),
        TensorType::F32 => {
            let shift =
                if name.contains("norm") || name.contains("_clip") || name.ends_with(".scale") {
                    1.0
                } else {
                    0.0
                };
            let gain = if name.contains("ctrl") { 0.2 } else { 1.0 };

            (0..n)
                .flat_map(|i| (step(i) as f32 * 0.07 * gain + shift).to_le_bytes())
                .collect()
        }
        TensorType::F16 | TensorType::BF16 => (0..n)
            .flat_map(|i| {
                let v = step(i) as f32 * 0.07;
                match ty {
                    TensorType::F16 => f16::from_f32(v).to_le_bytes(),
                    _ => half::bf16::from_f32(v).to_le_bytes(),
                }
            })
            .collect(),
        _ => scaled(ty, n, k, if name.contains("_exps") { 0.06 } else { 0.3 }),
    }
}

/// The variant's tensors with their data, in its layout. Unsloth's unrolls
/// the loop into `block_count`, names the controls `attn`, `k`, `v` and
/// `mlp` and keeps the message queries of all core layers in two tables;
/// its data is that of the looped layout.
fn tensors_with_data(c: &Tiny) -> Vec<(String, Vec<u64>, TensorType, Vec<u8>)> {
    let looped = tensors(c)
        .into_iter()
        .enumerate()
        .map(|(k, (name, dims, ty))| {
            let data = data(k, &name, dims.iter().product::<u64>() as usize, ty);
            (name, dims, ty, data)
        });
    if !c.unrolled {
        return looped.collect();
    }
    let (mut t, mut res, mut kv) = (Vec::new(), Vec::new(), Vec::new());
    for (name, dims, ty, data) in looped {
        if name.ends_with("loop_msg_query.weight") {
            let q = |j: usize| &data[j * 4 * c.dims().0 as usize..][..4 * c.dims().0 as usize];
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
        t.push((renamed, dims, ty, data));
    }
    let rows = 2 * LOOP.len() as u64;
    t.push((
        "loop_res_query.weight".into(),
        vec![c.dims().0, rows],
        TensorType::F32,
        res,
    ));
    t.push((
        "loop_kv_res_query.weight".into(),
        vec![c.dims().0, rows],
        TensorType::F32,
        kv,
    ));
    t
}

fn tiny_model(c: Tiny) -> Memory {
    let arch = if c.unrolled {
        "inkling2"
    } else {
        "inkling2_loop"
    };
    let slots = match c.unrolled {
        true => N_PHYS + (PASSES - 1) * LOOP.len(),
        false => N_PHYS,
    };
    let per_layer = |v: Value| Value::Array(vec![v; slots]);
    let mut metadata = vec![(
        "general.architecture".to_string(),
        Value::String(arch.into()),
    )];
    let mut keys = vec![
        ("block_count", Value::U32(slots as u32)),
        ("context_length", Value::U32(128)),
        ("embedding_length", Value::U32(c.dims().0 as u32)),
        ("feed_forward_length", Value::U32(c.dims().2 as u32)),
        (
            "attention.head_count",
            per_layer(Value::U32(c.dims().1 as u32)),
        ),
        ("attention.head_count_kv", per_layer(Value::U32(1))),
        ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("expert_count", Value::U32(4)),
        ("expert_used_count", Value::U32(c.used)),
        ("attention.key_length", Value::U32(64)),
        ("attention.value_length", Value::U32(64)),
        ("vocab_size", Value::U32(VOCAB as u32)),
        ("vocab_size_unpadded", Value::U32(60)),
        ("expert_feed_forward_length", Value::U32(c.dims().2 as u32)),
        ("rope.dimension_count", Value::U32(0)),
        ("attention.sliding_window", Value::U32(WINDOW)),
        // the core's layers alternate too, so each pass repeats the pattern
        (
            "attention.sliding_window_pattern",
            Value::Array((0..slots).map(|i| Value::Bool(i % 2 == 0)).collect()),
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
        ("loop.gate_rank", Value::U32(32)),
    ];
    keys.extend(match c.unrolled {
        true => vec![
            ("num_loops", Value::U32(PASSES as u32)),
            ("loop.core_start", Value::U32(LOOP.start as u32)),
            ("loop.core_end", Value::U32(LOOP.end as u32)),
        ],
        false => vec![
            ("loop.start", Value::U32(LOOP.start as u32)),
            ("loop.end", Value::U32(LOOP.end as u32)),
            ("loop.count", Value::U32(PASSES as u32)),
            ("loop.varying", Value::Bool(false)),
        ],
    });
    metadata.extend(keys.into_iter().map(|(k, v)| (format!("{arch}.{k}"), v)));
    let tensors = tensors_with_data(&c);
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
const PROMPT: [u32; 12] = [1, 40, 10, 23, 50, 7, 31, 12, 33, 6, 45, 24];
const NEXT: [u32; 6] = [30, 3, 59, 7, 17, 9];

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// One test, so one model on the device at a time: capturing the decode
/// graph on one thread forbids other threads' synchronous copies in the
/// same context.
///
/// With f32 dense weights and every expert used, the device follows the
/// CPU within its rounding. With the file's types and two experts of four,
/// rounding may change which experts run, so those models check exactness
/// alone. Both layouts of the published files, and their two expert types
/// that the device reads in their own bytes, run every check.
#[test]
#[ignore = "needs a CUDA device"]
fn inkling_on_cuda_matches_the_cpu_and_changes_no_bit() {
    for experts in [TensorType::IQ4NL, TensorType::NVFP4] {
        let mut looped_logits = None;
        for unrolled in [false, true] {
            let close = Tiny {
                dense: TensorType::F32,
                experts,
                used: 4,
                unrolled,
                wide: false,
            };
            let file = tiny_model(close);
            let model = kinetic_models::load(&file).unwrap();
            let gpu = CudaModel::load(&model, 128).unwrap();
            assert!(
                gpu.plan()
                    .is_some_and(|p| p.pools.iter().all(|p| p.slots == p.layers * 4)),
                "every expert on the device"
            );
            let logits = logits_match_the_cpu(&gpu, &model);
            greedy_tokens_match_the_cpu(&gpu, &model);
            // the layouts hold one model
            assert_eq!(
                looped_logits.get_or_insert_with(|| logits.clone()),
                &logits,
                "{close:?}"
            );
            drop(gpu);

            let exact = Tiny {
                dense: TensorType::Q8_0,
                used: 2,
                ..close
            };
            let file = tiny_model(exact);
            let model = kinetic_models::load(&file).unwrap();
            let gpu = CudaModel::load(&model, 128).unwrap();
            exactness(&gpu, &model);
            drop(gpu);

            // Three slots for all 24 experts: most picks run on the host or
            // come over for a batch, and the result must not depend on
            // where they ran.
            let slot = gpu_slot_bytes(experts);
            let placement = Placement {
                expert_cache: Some(3 * slot),
                ..Placement::default()
            };
            let gpu = CudaModel::load_with(&model, 128, placement).unwrap();
            let plan = gpu.plan().unwrap();
            assert_eq!(
                plan.pools.iter().map(|p| p.slots).sum::<usize>(),
                3,
                "{plan}"
            );
            exactness(&gpu, &model);
            let stats = gpu.stats().unwrap();
            eprintln!("{exact:?} offloaded: {stats:?}");
            assert!(stats.host_experts > 0 && stats.loaded > 0, "{stats:?}");
        }
    }
}

/// Device bytes of one expert of the tiny model with experts of `ty`, and
/// with NVFP4 its three scales.
fn gpu_slot_bytes(ty: TensorType) -> usize {
    let values = 3 * 128 * 64;
    match ty {
        TensorType::NVFP4 => values / 64 * 36 + 12,
        _ => values / 32 * 18,
    }
}

/// Greedy decoding from the prompt picks the same tokens on both.
fn greedy_tokens_match_the_cpu(gpu: &CudaModel, model: &kinetic_models::Model) {
    let argmax = kinetic_models::speculative::argmax;
    let mut cpu = model.new_state();
    let mut dev = gpu.new_state().unwrap();
    let mut want = vec![argmax(&model.run(&mut cpu, &PROMPT, false).logits)];
    let mut got = vec![argmax(&gpu.run(&mut dev, &PROMPT, false).unwrap().logits)];
    for _ in 0..12 {
        want.push(argmax(
            &model.run(&mut cpu, &want[want.len() - 1..], false).logits,
        ));
        got.push(argmax(
            &gpu.run(&mut dev, &got[got.len() - 1..], false)
                .unwrap()
                .logits,
        ));
    }
    assert_eq!(got, want);
}
fn exactness<'m>(gpu: &CudaModel<'m>, model: &kinetic_models::Model<'m>) {
    a_batch_matches_token_by_token(gpu);
    rollback_leaves_the_state_as_if_never_run(gpu);
    a_host_copy_restores_the_state(gpu, model);
    a_reset_state_is_a_fresh_one(gpu);
    shared_passes_match_running_alone(gpu);
    speculative_greedy_is_plain_greedy(gpu);
    prompt_cuts_change_no_bit(gpu);
    truncating_to_a_mark_matches_stopping_there(gpu, model);
}

/// The GPU's logits after every token of a prompt and of single steps are
/// the CPU's within the device's rounding: its f16 cache and inputs of
/// quantized products in blocks of 32. The first token, which no earlier
/// rounding reaches through attention, within two parts in a thousand; later
/// ones, where the tiny looped model amplifies rounding about as much as a
/// part-in-a-thousand change of its weights moves the CPU's own logits,
/// within three parts in a hundred, which still catches any wiring error.
fn logits_match_the_cpu(gpu: &CudaModel, model: &kinetic_models::Model) -> Vec<f32> {
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
    assert!(rows[0] <= 2e-3 * scale.max(1.0), "first token: {rows:?}");
    assert!(
        worst <= 0.03 * scale.max(1.0),
        "{worst} against logits up to {scale}"
    );
    got
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

/// A prompt long enough to run in prefill passes gives the same bits cut
/// anywhere on the prefill grid.
fn prompt_cuts_change_no_bit(gpu: &CudaModel) {
    let prompt: Vec<u32> = (0..96).map(|i| (i * 7 % 59) as u32).collect();
    let run = |cuts: &[usize]| {
        let mut state = gpu.new_state().unwrap();
        let (mut hidden, mut logits) = (Vec::new(), Vec::new());
        for part in cuts.windows(2) {
            let out = gpu
                .run(&mut state, &prompt[part[0]..part[1]], false)
                .unwrap();
            hidden.extend(bits(&out.hidden));
            logits = bits(&out.logits);
        }
        (hidden, logits)
    };
    assert!(run(&[0, 96]) == run(&[0, 32, 64, 96]));
}

/// A state cut back to a mark, as the server cuts a conversation it
/// continues differently, gives what a state that stopped there gives.
fn truncating_to_a_mark_matches_stopping_there<'m>(
    gpu: &CudaModel<'m>,
    model: &kinetic_models::Model<'m>,
) {
    use kinetic_models::warm::Runner;
    let runner = CudaRunner {
        gpu,
        model,
        spare: Cell::new(None),
    };
    let mut state = gpu.new_state().unwrap();
    gpu.run(&mut state, &PROMPT[..8], false).unwrap();
    runner.mark(&mut state);
    gpu.run(&mut state, &PROMPT[8..], false).unwrap();
    assert_eq!(gpu.truncate(model, &mut state, 8).unwrap(), 8);
    let got = gpu.run(&mut state, &NEXT, true).unwrap().logits;

    let mut plain = gpu.new_state().unwrap();
    gpu.run(&mut plain, &PROMPT[..8], false).unwrap();
    let want = gpu.run(&mut plain, &NEXT, true).unwrap().logits;
    assert_eq!(bits(&got), bits(&want));
}

/// Bytes of `n` weights of `ty`, a type this test writes no encoder for:
/// code bytes that vary with `k`, and every block's scale set so that the
/// weights' root mean square is `rms`.
fn scaled(ty: TensorType, n: usize, k: usize, rms: f32) -> Vec<u8> {
    let (len, size) = (ty.block_len(), ty.block_bytes());
    let mut bytes: Vec<u8> = (0..n / len * size)
        .map(|i| (i * (2 * k + 5) % 251) as u8)
        .collect();
    let set = |bytes: &mut [u8], s: f32| {
        let half = f16::from_f32(s).to_le_bytes();
        let at: &[usize] = match ty {
            TensorType::Q2K => &[80, 82],
            TensorType::Q3K => &[108],
            TensorType::Q4K | TensorType::Q5K => &[0, 2],
            TensorType::Q6K => &[208],
            _ => &[0],
        };
        for b in bytes.chunks_exact_mut(size) {
            match ty {
                TensorType::MXFP4 => b[0] = (128.0 + s.log2().round()) as u8,
                _ => at.iter().for_each(|&a| b[a..a + 2].copy_from_slice(&half)),
            }
        }
    };
    set(&mut bytes, 1.0);
    let mut x = vec![0.0; n];
    kinetic_quant::dequantize(ty, &bytes, &mut x).unwrap();
    let r = (x.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
    set(&mut bytes, rms / r);
    bytes
}
