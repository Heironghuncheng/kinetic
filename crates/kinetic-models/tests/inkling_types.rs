//! Every weight type a published Inkling 2 file holds, on a Vulkan GPU
//! against the CPU: a tiny looped model whose experts and query, key and
//! value projections are in that type, in ggml-org's and in Unsloth's
//! layout, with room on the device for the experts of half its layers.
//! Types with a kernel run on the device, the experts of the other layers
//! on the CPU; types without one run from the host. All must give the same
//! answers. The next type is one more row of [`TYPES`]. Passes
//! with a message when no device can run models.
//!
//! cargo test --release -p kinetic-models --features vulkan --test inkling_types

#![cfg(feature = "vulkan")]

mod inkling_tiny;

use kinetic_gguf::TensorType;
use kinetic_models::vulkan::{devices, VulkanModel};

use inkling_tiny::{model_of, unrolled_of, Memory, N_PHYS, TYPED};

/// The types of the experts and other tensors of the thirteen published
/// files, IQ1_M besides.
const TYPES: &[TensorType] = &[
    TensorType::NVFP4,
    TensorType::MXFP4,
    TensorType::Q2_0,
    TensorType::IQ1S,
    TensorType::IQ1M,
    TensorType::IQ2XXS,
    TensorType::IQ2XS,
    TensorType::IQ2S,
    TensorType::IQ3XXS,
    TensorType::IQ3S,
    TensorType::IQ4NL,
    TensorType::IQ4XS,
    TensorType::Q3K,
    TensorType::Q4K,
    TensorType::Q5K,
    TensorType::Q6K,
    TensorType::Q5_0,
    TensorType::Q8_0,
    TensorType::BF16,
    TensorType::F32,
];

const TOKENS: [u32; 4] = [3, 7, 1, 9];

/// How far a logit may be from the CPU's, relative to the largest, and how
/// far apart the CPU's two best must be for the device to pick the same.
/// Random weights leave near-ties in routing, where the f16 caches of the
/// device and the 8-bit inputs of the CPU's quantized products may pick
/// different experts and move a token's logits by a tenth or so; a type
/// read wrongly gives logits unrelated to the CPU's.
const TOLERANCE: f32 = 0.25;
const CLEAR: f32 = 0.1;

/// Logits after every token: the CPU's, and the device's with room for
/// the experts of half the layers.
fn run(file: &Memory, ty: TensorType) -> Option<(Vec<f32>, Vec<f32>)> {
    let cpu = kinetic_models::load(file).unwrap_or_else(|e| panic!("{ty}: {e}"));
    let physical = devices().ok()?.into_iter().next()?;
    let (len, bytes) = (ty.block_len(), ty.block_bytes());
    let (d, ff) = (TYPED.d as usize, TYPED.n_ff as usize);
    let layer = 4 * (2 * ff * d + d * ff) / len * bytes;
    let gpu = VulkanModel::load_capped(&cpu, 64, &physical, true, Some(N_PHYS / 2 * layer))
        .unwrap_or_else(|e| panic!("{ty}: {e}"));
    let want = cpu.run(&mut cpu.new_state(), &TOKENS, true).logits;
    let mut state = gpu.new_state().unwrap();
    let got = gpu.run(&mut state, &TOKENS, true).unwrap().logits;
    Some((got, want))
}

#[test]
fn every_published_type_runs_on_the_device() {
    if devices().map_or(true, |d| d.is_empty()) {
        eprintln!("skipped: no Vulkan device can run models");
        return;
    }
    for &ty in TYPES {
        for (layout, file) in [
            ("ggml-org", model_of(&TYPED, ty)),
            ("Unsloth", unrolled_of(&TYPED, ty)),
        ] {
            let (got, want) = run(&file, ty).expect("a device");
            let what = format!("{ty} in {layout}'s layout");
            let scale = want.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= TOLERANCE * scale,
                    "{what}, logit {i}: {g} on the device against {w} on the CPU"
                );
            }
            // the same greedy token wherever the CPU's choice is clear
            let vocab = want.len() / TOKENS.len();
            for (t, (g, w)) in got.chunks(vocab).zip(want.chunks(vocab)).enumerate() {
                let mut order: Vec<usize> = (0..vocab).collect();
                order.sort_by(|&a, &b| w[b].total_cmp(&w[a]));
                let clear = w[order[0]] - w[order[1]] > CLEAR * scale;
                let top = (0..vocab).max_by(|&a, &b| g[a].total_cmp(&g[b]));
                assert!(!clear || top == Some(order[0]), "{what}, token {t}");
            }
            eprintln!("{what}: same answers");
        }
    }
}
