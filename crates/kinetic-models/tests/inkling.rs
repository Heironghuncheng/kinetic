//! Inkling 2: a tiny looped model against reference logits, and the
//! ignored tests against real files. Only a real file's header matters, so
//! it may be a copy whose tensor data is a sparse hole of the same length.
//! Loading fails on any tensor or `<arch>.` key the family neither reads
//! nor names as skipped. KINETIC_TEST_INKLING names one file, or a folder
//! whose `.gguf` files are all checked, such as one holding every
//! published quantization.
//!
//! KINETIC_TEST_INKLING=folder cargo test --release -p kinetic-models --test inkling -- --ignored --test-threads=1

use std::path::PathBuf;

mod inkling_tiny;

use kinetic_gguf::{GgufFile, TensorType, Value};

use inkling_tiny::{model, unrolled_of, weights, write, Memory, Tensor, LOOP, N_PHYS, REFERENCE};

const D: u64 = REFERENCE.d;

fn files() -> Vec<PathBuf> {
    let path =
        PathBuf::from(std::env::var("KINETIC_TEST_INKLING").expect("set KINETIC_TEST_INKLING"));
    if !path.is_dir() {
        return vec![path];
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    files.sort();
    files
}

#[test]
#[ignore]
fn reads_every_tensor_and_key_of_every_file() {
    for path in files() {
        let file = GgufFile::open(&path).unwrap();
        let model =
            kinetic_models::load(&file).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let key = |k: &str| {
            let arch = file.header().architecture().unwrap();
            file.header()
                .get(&format!("{arch}.{k}"))
                .and_then(|v| v.as_u64())
        };
        // block_count lists every pass in Unsloth's layout, the physical
        // layers only in ggml-org's
        let listed = key("block_count").unwrap();
        let extra = match (key("loop.start"), key("loop.end"), key("loop.count")) {
            (Some(start), Some(end), Some(count)) => (end - start) * (count - 1),
            _ => 0,
        };
        assert_eq!(
            model.n_layers() as u64,
            listed + extra,
            "{}",
            path.display()
        );
        assert_eq!(Some(model.n_vocab() as u64), key("vocab_size_unpadded"));
        println!("{}: loads, {} layers", path.display(), model.n_layers());
    }
}

/// Runs tokens through every layer and pass, each weight type through its
/// product. With sparse copies the weights are zeros, so this checks the
/// shapes, the types and the order of the layers, not the values.
#[test]
#[ignore]
fn runs_tokens_through_every_pass_of_every_file() {
    for path in files() {
        let file = GgufFile::open(&path).unwrap();
        let model = kinetic_models::load(&file).unwrap();
        let mut state = model.new_state();
        let logits = model.forward_batch(&mut state, &[1, 2]);
        assert_eq!(logits.len(), model.n_vocab());
        assert!(logits.iter().all(|l| l.is_finite()), "{}", path.display());
        let logits = model.forward(&mut state, 3);
        assert!(logits.iter().all(|l| l.is_finite()), "{}", path.display());
    }
}

fn unrolled_model() -> Memory {
    unrolled_of(&REFERENCE, TensorType::F32)
}

fn logits(model: &kinetic_models::Model, batches: &[usize]) -> Vec<f32> {
    let tokens = [3, 7, 1, 9];
    let mut state = model.new_state();
    let mut logits = Vec::new();
    let mut at = 0;
    for &n in batches {
        logits.extend(model.run(&mut state, &tokens[at..at + n], true).logits);
        at += n;
    }
    logits
}

/// Logits of the tiny model after each of the tokens 3, 7, 1 and 9,
/// whichever way they are batched, against values computed in NumPy
/// (float64) from the graphs of llama.cpp's src/models/inkling2.cpp and
/// inkling2-loop.cpp on the same weights.
#[test]
fn tiny_model_matches_the_reference() {
    let file = model(&REFERENCE);
    let model = kinetic_models::load(&file).unwrap();
    assert_eq!(model.n_layers(), N_PHYS + LOOP.len());
    assert_eq!(model.n_vocab(), 10);
    let mut first: Option<Vec<f32>> = None;
    for batches in [&[4][..], &[2, 1, 1], &[1, 1, 1, 1]] {
        let logits = logits(&model, batches);
        for (i, (got, want)) in logits.iter().zip(LOGITS).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "{batches:?}, logit {i}: {got} against {want}"
            );
        }
        // a batch gives each token what it gets alone, bit for bit
        assert_eq!(first.get_or_insert_with(|| logits.clone()), &logits);
    }
}

/// Unsloth's layout of the same weights gives the same logits, bit for bit.
#[test]
fn the_unrolled_layout_runs_the_same_model() {
    let unrolled = unrolled_model();
    let looped = model(&REFERENCE);
    let model = kinetic_models::load(&unrolled).unwrap();
    assert_eq!(model.n_layers(), N_PHYS + LOOP.len());
    let want = logits(&kinetic_models::load(&looped).unwrap(), &[4]);
    assert_eq!(logits(&model, &[4]), want);
}

/// Taps change no logit and no final state, count the visits of every pass
/// alike in both layouts, and the first visit reads the normalized
/// embedding.
#[test]
fn taps_add_layer_inputs_and_change_nothing_else() {
    let d = D as usize;
    let tokens = [3, 7, 1, 9];
    let run = |file: &Memory, taps: Vec<usize>| {
        let mut model = kinetic_models::load(file).unwrap();
        model.set_taps(taps);
        let mut state = model.new_state();
        let mut f = model.run(&mut state, &tokens[..3], true);
        let last = model.run(&mut state, &tokens[3..], true);
        f.logits.extend(last.logits);
        f.hidden.extend(last.hidden);
        (f.logits, f.hidden)
    };
    let (looped, unrolled) = (model(&REFERENCE), unrolled_model());
    let (logits, hidden) = run(&looped, Vec::new());
    let taps = vec![0, 4, 6, 3];
    let (tapped_logits, tapped) = run(&looped, taps.clone());
    assert_eq!(tapped_logits, logits);
    assert_eq!(tapped.len(), hidden.len() * (1 + taps.len()));
    let row = d * (1 + taps.len());
    for (r, h) in hidden.chunks_exact(d).enumerate() {
        assert_eq!(&tapped[r * row..][..d], h);
    }
    assert_eq!(
        run(&unrolled, taps.clone()),
        (tapped_logits, tapped.clone())
    );

    let weights = weights(&REFERENCE);
    let table = |name: &str| &weights.iter().find(|w| w.0 == name).unwrap().2;
    let (embd, norm) = (table("token_embd.weight"), table("token_embd_norm.weight"));
    for (r, &t) in tokens.iter().enumerate() {
        let e = &embd[t as usize * d..][..d];
        let rms = (e.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-6).sqrt();
        for (i, got) in tapped[r * row + d..][..d].iter().enumerate() {
            let want = e[i] / rms * norm[i];
            assert!((got - want).abs() < 1e-5, "row {r} value {i}: {got} {want}");
        }
    }
}

/// An unrolled array whose later passes do not repeat the core is refused.
#[test]
fn an_unrolled_array_must_repeat_the_core() {
    let mut file = unrolled_model();
    let key = "inkling2.attention.sliding_window_pattern";
    let (_, v) = file
        .header
        .metadata
        .iter_mut()
        .find(|(k, _)| k == key)
        .unwrap();
    *v = Value::Array(
        (0..N_PHYS + LOOP.len())
            .map(|i| Value::Bool(i < 7))
            .collect(),
    );
    let Err(e) = kinetic_models::load(&file) else {
        panic!("a mismatched pattern loaded")
    };
    assert!(e.0.contains("differs between passes"), "{e}");
}

/// The tensors of a tiny DSpark drafter for the tiny model, which reads the
/// inputs of its layer visits 1, 4 and 6 and its final state.
fn tiny_drafter_tensors() -> Vec<(String, Vec<u64>)> {
    let (q, kv, ff, rank) = (8, 4, 8, 3);
    let mut t: Vec<(String, Vec<u64>)> = [
        ("dspark_fuse.weight", vec![4, 4]),
        ("dspark_hidden_out.bias", vec![D]),
        ("dspark_hidden_out.weight", vec![rank, D]),
        ("dspark_norm.weight", vec![D]),
        ("joint_proj.bias", vec![3 * rank]),
        ("joint_proj.weight", vec![2 * rank + 2 * D, 3 * rank]),
        ("markov_w1.weight", vec![rank, 12]),
        ("mask_embd.weight", vec![D]),
        ("output_norm.weight", vec![D]),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect();
    for i in 0..3 {
        let mut add = |name: &str, dims: Vec<u64>| t.push((format!("blk.{i}.{name}"), dims));
        add("attn_q.weight", vec![D, q]);
        add("attn_k.weight", vec![D, kv]);
        add("attn_v.weight", vec![D, kv]);
        add("attn_output.weight", vec![q, D]);
        add("attn_k_ctx.weight", vec![D, kv]);
        add("attn_v_ctx.weight", vec![D, kv]);
        add("attn_q_norm.weight", vec![4]);
        add("attn_k_norm.weight", vec![4]);
        add("attn_k_ctx_norm.weight", vec![4]);
        add("attn_norm.weight", vec![D]);
        add("dflare_norm.weight", vec![D]);
        add("ffn_norm.weight", vec![D]);
        add("ffn_gate.weight", vec![D, ff]);
        add("ffn_up.weight", vec![D, ff]);
        add("ffn_down.weight", vec![ff, D]);
    }
    t
}

fn tiny_drafter_weights() -> Vec<(String, Vec<u64>, Vec<f32>)> {
    tiny_drafter_tensors()
        .into_iter()
        .enumerate()
        .map(|(k, (name, dims))| {
            let n: u64 = dims.iter().product();
            let shift = if name.contains("norm") { 1.0 } else { 0.0 };
            let values = (0..n as usize)
                .map(|i| (((i * (2 * k + 5) + 3 * k) % 19) as f32 - 9.0) * 0.09 + shift)
                .collect();
            (name, dims, values)
        })
        .collect()
}

fn tiny_drafter_keys() -> Vec<(&'static str, Value)> {
    vec![
        ("block_count", Value::U32(3)),
        ("context_length", Value::U32(64)),
        ("embedding_length", Value::U32(D as u32)),
        ("feed_forward_length", Value::U32(8)),
        ("attention.head_count", Value::U32(2)),
        ("attention.head_count_kv", Value::U32(1)),
        ("rope.freq_base", Value::F32(1e6)),
        ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("attention.key_length", Value::U32(4)),
        ("attention.value_length", Value::U32(4)),
        ("vocab_size", Value::U32(12)),
        ("vocab_size_unpadded", Value::U32(10)),
        ("block_size", Value::U32(8)),
        (
            "target_layers",
            Value::Array([1, 4, 6, 8].map(Value::U32).to_vec()),
        ),
        ("dspark.markov_rank", Value::U32(3)),
        ("sample_from_anchor", Value::Bool(false)),
        ("has_confidence_head", Value::Bool(false)),
        ("attention.causal", Value::Bool(false)),
        ("attention.sliding_window", Value::U32(3)),
        (
            "attention.sliding_window_pattern",
            Value::Array([true, false, true].map(Value::Bool).to_vec()),
        ),
        ("rope.dimension_count", Value::U32(4)),
        ("final_logit_softcapping", Value::F32(30.0)),
    ]
}

/// `weights` as f32 tensors of a file.
fn f32s(weights: &[(String, Vec<u64>, Vec<f32>)]) -> Vec<Tensor> {
    weights
        .iter()
        .map(|(name, dims, values)| {
            let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (name.clone(), dims.clone(), TensorType::F32, bytes)
        })
        .collect()
}

fn tiny_drafter() -> Memory {
    write(
        "dflash",
        tiny_drafter_keys(),
        &f32s(&tiny_drafter_weights()),
    )
}

#[test]
fn the_drafter_loads_through_the_guard() {
    let file = tiny_drafter();
    let dspark = kinetic_models::Dspark::load(&file).unwrap();
    assert_eq!(dspark.target_layers(), [1, 4, 6, 8]);
    assert_eq!(dspark.max_draft(), 7);

    let mut weights = tiny_drafter_weights();
    weights.push(("blk.0.attn_k_ctx.bias".into(), vec![4], vec![0.0; 4]));
    let extra = write("dflash", tiny_drafter_keys(), &f32s(&weights));
    let Err(e) = kinetic_models::Dspark::load(&extra) else {
        panic!("a drafter with an unread tensor loaded")
    };
    assert!(e.0.ends_with(": blk.0.attn_k_ctx.bias"), "{e}");
}

/// The published drafter loads, every tensor and key read, and drafts a
/// whole block from made-up hidden states of the target's width through a
/// made-up output head, since the target is not needed for that.
///
/// KINETIC_TEST_DSPARK=dspark-Inkling-2-Mini-Q8_0.gguf cargo test --release -p kinetic-models --test inkling -- --ignored drafter
#[test]
#[ignore]
fn the_published_drafter_loads_and_drafts() {
    let path = std::env::var("KINETIC_TEST_DSPARK").expect("set KINETIC_TEST_DSPARK");
    let file = GgufFile::open(&path).unwrap();
    let dspark = kinetic_models::Dspark::load(&file).unwrap();
    assert_eq!(dspark.target_layers(), [5, 23, 41, 48]);
    assert_eq!(dspark.max_draft(), 7);
    let (d, vocab) = (dspark.n_embd(), dspark.vocab());
    let mut seed = 1u32;
    let mut noise = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1 << 24) as f32 - 0.5) * scale
            })
            .collect()
    };
    let mut state = dspark.new_state();
    dspark.commit(&mut state, &noise(5 * 4 * d, 4.0));
    dspark.commit(&mut state, &noise(4 * d, 4.0));
    assert_eq!(state.len(), 6);
    let head = noise(d, 1.0);
    let draft = dspark.draft(&state, 1000, &noise(d, 2.0), 7, |h| {
        (0..vocab).map(|i| h[i % d] * head[i * 7 % d]).collect()
    });
    assert_eq!(draft.len(), 7);
    for (token, p) in draft {
        assert!((token as usize) < vocab && p.is_finite() && p > 0.0 && p <= 1.0);
    }
}

/// The tiny model's run of `tokens` with the taps the tiny drafter reads:
/// each position's features in the drafter's order.
fn tiny_features(model: &mut kinetic_models::Model, tokens: &[u32]) -> Vec<f32> {
    let d = D as usize;
    model.set_taps(vec![1, 4, 6]);
    let hidden = model.run(&mut model.new_state(), tokens, true).hidden;
    hidden
        .chunks_exact(4 * d)
        .flat_map(|row| [&row[d..], &row[..d]].concat())
        .collect()
}

/// Speculative greedy decoding with the drafter, after rejected drafts and
/// rollbacks too, gives plain greedy decoding token for token.
#[test]
fn speculative_greedy_with_the_drafter_is_plain_greedy() {
    use kinetic_models::speculative::{
        CompanionDrafter, Decoder, DsparkCompanion, Greedy, Lookup, Policy,
    };

    let (target, drafter) = (model(&REFERENCE), tiny_drafter());
    let prompt = [3, 7, 1, 9, 3, 7];
    let n = 40;
    let plain = {
        let model = kinetic_models::load(&target).unwrap();
        let mut state = model.new_state();
        let mut context = prompt.to_vec();
        let (mut scores, _) = Greedy.score(&model, &mut state, &prompt, false);
        while context.len() < prompt.len() + n {
            let t = Greedy.choose(&model, &scores, 0, &context);
            context.push(t);
            scores = Greedy.score(&model, &mut state, &[t], false).0;
        }
        context
    };
    for lookup in [false, true] {
        let mut model = kinetic_models::load(&target).unwrap();
        let dspark = kinetic_models::Dspark::load(&drafter).unwrap();
        let companion = Box::new(DsparkCompanion::new(dspark));
        let mut drafter = CompanionDrafter::new(companion, &mut model).unwrap();
        drafter.lookup = lookup.then(Lookup::default);
        let mut decoder = Decoder::new(drafter);
        decoder.max_pause = 0;
        let mut state = model.new_state();
        let mut context = prompt.to_vec();
        decoder.prefill(&model, &mut Greedy, &mut state, &context, 0);
        let (mut drafted, mut accepted) = (0, 0);
        while context.len() < prompt.len() + n {
            let limit = prompt.len() + n - context.len();
            let step = decoder.step(&model, &mut Greedy, &mut state, &mut context, limit, |_| {
                false
            });
            drafted += step.drafted;
            accepted += step.tokens - 1;
        }
        assert_eq!(context, plain, "lookup {lookup:?}");
        assert!(
            0 < accepted && accepted < drafted,
            "{accepted} of {drafted}"
        );
    }
}

/// A drafter beside its target is found when its header fits the target,
/// refused when it does not, and nothing is found when there is none.
#[test]
fn the_companion_beside_a_model_is_found_and_checked() {
    use kinetic_models::speculative::find_companion;

    let folder = std::env::temp_dir().join(format!("kinetic-companion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&folder);
    std::fs::create_dir_all(&folder).unwrap();
    let target = model(&REFERENCE);
    let model = folder.join("tiny.gguf");
    std::fs::write(&model, &target.bytes).unwrap();
    let find =
        |explicit: Option<&std::path::Path>| find_companion(&model, &target.header, explicit);

    assert_eq!(find(None).unwrap(), None);
    let drafter = folder.join("dspark-tiny-Q8_0.gguf");
    std::fs::write(&drafter, &tiny_drafter().bytes).unwrap();
    assert_eq!(find(None).unwrap(), Some(drafter.clone()));

    let mut keys = tiny_drafter_keys();
    keys.retain(|(k, _)| *k != "embedding_length");
    keys.push(("embedding_length", Value::U32(16)));
    let wide = folder.join("wide.gguf");
    std::fs::write(
        &wide,
        &write("dflash", keys, &f32s(&tiny_drafter_weights())).bytes,
    )
    .unwrap();
    let e = find(Some(&wide)).unwrap_err();
    assert!(
        e.0.contains("is not a drafter for this inkling2_loop model"),
        "{e}"
    );
    std::fs::remove_dir_all(&folder).unwrap();
}

/// One draft of the tiny drafter against a float64 reading of llama.cpp's
/// Inkling 2 DSpark graph (Unsloth's src/models/dspark.cpp) and its
/// noncausal mask, in which the whole block shares the anchor's window.
#[test]
fn a_draft_matches_the_reference() {
    let d = D as usize;
    let tokens = [3u32, 7, 1, 9, 3, 7];
    let anchor = 2u32;
    let target = model(&REFERENCE);
    let mut model = kinetic_models::load(&target).unwrap();
    let features = tiny_features(&mut model, &tokens);
    let drafter = tiny_drafter();
    let dspark = kinetic_models::Dspark::load(&drafter).unwrap();

    let tw = weights(&REFERENCE);
    let dw = tiny_drafter_weights();
    let from = |weights: &[(String, Vec<u64>, Vec<f32>)], name: &str| -> Vec<f64> {
        let t = weights.iter().find(|t| t.0 == name).unwrap();
        t.2.iter().map(|&v| v as f64).collect()
    };
    let w = |name: &str| from(&dw, name);
    let matvec = |m: &[f64], x: &[f64]| -> Vec<f64> {
        m.chunks_exact(x.len())
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    };
    let rms = |x: &[f64], w: &[f64]| -> Vec<f64> {
        let s = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + 1e-6).sqrt();
        x.iter().zip(w).map(|(x, w)| x / s * w).collect()
    };
    let rope = |x: &mut [f64], pos: usize| {
        for i in 0..2 {
            let theta = pos as f64 * 1e6f64.powf(-(2.0 * i as f64) / 4.0);
            let (a, b) = (x[i], x[i + 2]);
            x[i] = a * theta.cos() - b * theta.sin();
            x[i + 2] = a * theta.sin() + b * theta.cos();
        }
    };
    let softmax = |x: &mut [f64]| {
        let m = x.iter().cloned().fold(f64::MIN, f64::max);
        x.iter_mut().for_each(|v| *v = (*v - m).exp());
        let s: f64 = x.iter().sum();
        x.iter_mut().for_each(|v| *v /= s);
    };
    let fuse = w("dspark_fuse.weight");
    let mix = |pos: usize, row: usize, norm: &[f64]| -> Vec<f64> {
        let mut weights = fuse[row * 4..][..4].to_vec();
        softmax(&mut weights);
        let f = &features[pos * 4 * d..][..4 * d];
        let x: Vec<f64> = (0..d)
            .map(|i| (0..4).map(|k| weights[k] * f[k * d + i] as f64).sum())
            .collect();
        rms(&x, norm)
    };

    let pos0 = tokens.len();
    let embd = from(&tw, "token_embd.weight");
    let anchor_row = rms(
        &embd[anchor as usize * d..][..d],
        &from(&tw, "token_embd_norm.weight"),
    );
    let mask = w("mask_embd.weight");
    let mut x: Vec<Vec<f64>> = (0..8)
        .map(|t| {
            if t == 0 {
                anchor_row.clone()
            } else {
                mask.clone()
            }
        })
        .collect();
    for il in 0..3 {
        let t = |s: &str| w(&format!("blk.{il}.{s}"));
        let visible = if il == 1 { 0..pos0 } else { pos0 - 3..pos0 };
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for p in visible {
            let c = mix(p, il, &t("dflare_norm.weight"));
            let mut k = rms(
                &matvec(&t("attn_k_ctx.weight"), &c),
                &t("attn_k_ctx_norm.weight"),
            );
            rope(&mut k, p);
            keys.push(k);
            values.push(matvec(&t("attn_v_ctx.weight"), &c));
        }
        let h: Vec<Vec<f64>> = x.iter().map(|r| rms(r, &t("attn_norm.weight"))).collect();
        let mut queries = Vec::new();
        for (j, h) in h.iter().enumerate() {
            let q = matvec(&t("attn_q.weight"), h);
            queries.push(
                q.chunks_exact(4)
                    .map(|q| {
                        let mut q = rms(q, &t("attn_q_norm.weight"));
                        rope(&mut q, pos0 + j);
                        q
                    })
                    .collect::<Vec<_>>(),
            );
            let mut k = rms(&matvec(&t("attn_k.weight"), h), &t("attn_k_norm.weight"));
            rope(&mut k, pos0 + j);
            keys.push(k);
            values.push(matvec(&t("attn_v.weight"), h));
        }
        for (j, row) in x.iter_mut().enumerate() {
            let mut out = Vec::new();
            for q in &queries[j] {
                let mut s: Vec<f64> = keys
                    .iter()
                    .map(|k| k.iter().zip(q).map(|(a, b)| a * b).sum::<f64>() * 0.5)
                    .collect();
                softmax(&mut s);
                out.extend(
                    (0..4).map(|i| s.iter().zip(&values).map(|(s, v)| s * v[i]).sum::<f64>()),
                );
            }
            let o = matvec(&t("attn_output.weight"), &out);
            row.iter_mut().zip(o).for_each(|(x, o)| *x += o);
            let h = rms(row, &t("ffn_norm.weight"));
            let (g, u) = (
                matvec(&t("ffn_gate.weight"), &h),
                matvec(&t("ffn_up.weight"), &h),
            );
            let a: Vec<f64> = g
                .iter()
                .zip(u)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let f = matvec(&t("ffn_down.weight"), &a);
            row.iter_mut().zip(f).for_each(|(x, f)| *x += f);
        }
    }
    let hidden: Vec<Vec<f64>> = x.iter().map(|r| rms(r, &w("output_norm.weight"))).collect();
    let context = mix(pos0 - 1, 3, &w("dspark_norm.weight"));
    let (markov, joint, joint_b) = (
        w("markov_w1.weight"),
        w("joint_proj.weight"),
        w("joint_proj.bias"),
    );
    let (corr, corr_b, out) = (
        w("dspark_hidden_out.weight"),
        w("dspark_hidden_out.bias"),
        from(&tw, "output.weight"),
    );
    let mut state = [0.0; 3];
    let mut prev = anchor as usize;
    let mut want = Vec::new();
    for h in &hidden[1..] {
        let input = [&state[..], &markov[prev * 3..][..3], h, &context].concat();
        let j: Vec<f64> = matvec(&joint, &input)
            .iter()
            .zip(&joint_b)
            .map(|(a, b)| a + b)
            .collect();
        for i in 0..3 {
            let (g, c) = (1.0 / (1.0 + (-j[i]).exp()), j[3 + i].tanh());
            state[i] = c + g * (state[i] - c);
        }
        let latent: Vec<f64> = j[6..].iter().map(|v| v.tanh()).collect();
        let c = matvec(&corr, &latent);
        let y: Vec<f64> = (0..d).map(|i| h[i] + c[i] + corr_b[i]).collect();
        let logits: Vec<f64> = matvec(&out[..10 * d], &y)
            .iter()
            .map(|l| 30.0 * (l / 30.0).tanh())
            .collect();
        let best = (0..10).fold(0, |b, i| if logits[i] > logits[b] { i } else { b });
        let total: f64 = logits.iter().map(|l| (l - logits[best]).exp()).sum();
        want.push((best as u32, 1.0 / total));
        prev = best;
    }

    let mut state = dspark.new_state();
    dspark.commit(&mut state, &features[..4 * 4 * d]);
    dspark.commit(&mut state, &features[4 * 4 * d..]);
    let anchor_f32: Vec<f32> = anchor_row.iter().map(|&v| v as f32).collect();
    let head: Vec<f32> = out[..10 * d].iter().map(|&v| v as f32).collect();
    let got = dspark.draft(&state, anchor, &anchor_f32, 7, |h| {
        head.chunks_exact(d)
            .map(|r| r.iter().zip(h).map(|(a, b)| a * b).sum())
            .collect()
    });
    assert_eq!(got.len(), 7);
    for (i, ((t, p), (wt, wp))) in got.iter().zip(&want).enumerate() {
        assert_eq!(t, wt, "token {i}");
        assert!(
            (*p as f64 - wp).abs() < 1e-4,
            "chance {i}: {p} against {wp}"
        );
    }
}

const LOGITS: [f32; 40] = [
    1.337009, 0.8481389, -0.7500277, -1.971547, 0.6931028, 0.5891846, -2.125962, 1.414415,
    0.4340913, -0.6725019, -0.4141479, 0.8115726, -1.191427, -0.212345, 0.7890617, 0.707845,
    -0.2348708, -0.4028865, 0.6853302, -1.180181, 0.2231518, 0.5435061, -0.398955, -1.543489,
    0.460775, 0.9439149, -1.626012, 0.2645262, 0.8612356, -0.3575845, 0.08001616, 0.8101404,
    -1.594792, 0.02124436, 0.5619222, 1.058711, -0.2271008, 0.2041879, 0.8106044, -1.470942,
];
