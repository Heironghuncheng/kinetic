//! Inkling 2 on a Vulkan GPU against the CPU, on a tiny looped model built
//! in memory: the stream after every layer of the first pass, the logits of
//! every token, and the same bits for every way of batching the tokens and
//! after a rollback. Each test passes with a message when no device can run
//! it.
//!
//! cargo test --release -p kinetic-models --features vulkan --test inkling_vulkan

#![cfg(feature = "vulkan")]

mod inkling_tiny;

use kinetic_models::vulkan::{devices, VulkanModel};
use kinetic_models::Model;

use inkling_tiny::{model, DEVICE, LOOP, N_PHYS};

const TOKENS: [u32; 4] = [3, 7, 1, 9];

/// How far a device value may be from the CPU's, relative to the largest
/// magnitude of the values compared. The device keeps keys and values as
/// f16, rounded toward zero on some drivers, about 5e-4 off in the first
/// layer, and this model's loop controls and experts grow that tenfold by
/// its last; a wrong index or formula is off by far more. The kernels
/// themselves are checked closely in kinetic-vulkan's tests.
const TOLERANCE: f32 = 3e-2;

fn close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let scale = want.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= TOLERANCE * scale,
            "{what}, value {i}: {g} on the device against {w} on the CPU"
        );
    }
}

/// The tiny model on the CPU and on the first device, if there is one.
fn with_device(f: impl FnOnce(&Model, &VulkanModel)) {
    let file = model(&DEVICE);
    let cpu = kinetic_models::load(&file).unwrap();
    let Some(physical) = devices().ok().and_then(|d| d.into_iter().next()) else {
        eprintln!("skipped: no Vulkan device can run models");
        return;
    };
    let gpu = VulkanModel::load_on(&cpu, 64, &physical).unwrap();
    f(&cpu, &gpu);
}

/// The stream after each layer of the first pass: the first that differs
/// names the piece to look at.
#[test]
fn every_layer_matches_the_cpu() {
    with_device(|cpu, gpu| {
        let want = cpu.trace_layers(&TOKENS);
        let got = gpu.trace_layers(&TOKENS).unwrap();
        assert_eq!(want.len(), N_PHYS + LOOP.len());
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            close(&format!("stream after layer step {i}"), g, w);
        }
    });
}

/// Logits of every token, in any batches and after a rollback, against the
/// CPU; the device gives every way the same bits.
#[test]
fn logits_match_the_cpu_in_any_batches() {
    with_device(|cpu, gpu| {
        let mut state = cpu.new_state();
        let want = cpu.run(&mut state, &TOKENS, true).logits;
        let mut first: Option<Vec<f32>> = None;
        for batches in [&[4][..], &[2, 1, 1], &[1, 1, 1, 1]] {
            let mut state = gpu.new_state().unwrap();
            let mut logits = Vec::new();
            let mut at = 0;
            for &n in batches {
                let run = gpu.run(&mut state, &TOKENS[at..at + n], true).unwrap();
                logits.extend(run.logits);
                at += n;
            }
            close(&format!("logits in batches {batches:?}"), &logits, &want);
            assert_eq!(first.get_or_insert_with(|| logits.clone()), &logits);
        }

        // A draft whose last two tokens are wrong, rolled back to its first,
        // leaves the state that token alone leaves.
        let mut state = gpu.new_state().unwrap();
        let mut logits = gpu.run(&mut state, &TOKENS[..1], true).unwrap().logits;
        gpu.checkpoint(&mut state).unwrap();
        let draft = gpu.run(&mut state, &[TOKENS[1], 2, 2], true).unwrap();
        logits.extend(&draft.logits[..gpu.n_vocab()]);
        gpu.rollback(&mut state, 1).unwrap();
        logits.extend(gpu.run(&mut state, &TOKENS[2..], true).unwrap().logits);
        assert_eq!(first.as_ref(), Some(&logits), "after a rollback");
    });
}
