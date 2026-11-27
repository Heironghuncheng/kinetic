//! A mixture-of-experts model on a Vulkan GPU with room for only some of
//! its experts, on a real file: the experts of the other layers run on the
//! CPU, and the logits stay close to those of every expert on the device.
//!
//! KINETIC_TEST_GGUF=moe.gguf cargo test --release -p kinetic-models --features vulkan --test vulkan_experts -- --ignored

#![cfg(feature = "vulkan")]

use kinetic_gguf::GgufFile;
use kinetic_models::vulkan::{devices, VulkanModel};

/// How far a logit may be from those of every expert on the device,
/// relative to the largest, and how far apart the two best of those must
/// be for the same pick. The CPU's 8-bit inputs round apart from the
/// device's, and a near-tie in a router then picks other experts in later
/// layers; experts run on the wrong rows are off by far more.
const TOLERANCE: f32 = 0.25;
const CLEAR: f32 = 0.1;

#[test]
#[ignore = "needs KINETIC_TEST_GGUF of a mixture-of-experts model and a Vulkan device"]
fn experts_on_the_cpu_keep_the_logits() {
    let path = std::env::var("KINETIC_TEST_GGUF").expect("set KINETIC_TEST_GGUF");
    let file = GgufFile::open(&path).unwrap();
    let model = kinetic_models::load(&file).unwrap();
    let physical = devices()
        .unwrap()
        .into_iter()
        .next()
        .expect("a Vulkan device");
    let tokens: Vec<u32> = (0..24).map(|i| 100 + 37 * i).collect();
    let run = |cap: Option<usize>| {
        let gpu = VulkanModel::load_capped(&model, 64, &physical, false, cap).unwrap();
        let mut state = gpu.new_state().unwrap();
        let logits = gpu.run(&mut state, &tokens, false).unwrap().logits;
        (logits, gpu.expert_report())
    };
    let (want, report) = run(None);
    assert!(report.is_none(), "every expert fits on the device");
    let scale = want.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
    let mut order: Vec<usize> = (0..want.len()).collect();
    order.sort_by(|&a, &b| want[b].total_cmp(&want[a]));
    let clear = want[order[0]] - want[order[1]] > CLEAR * scale;

    // experts take most of a mixture-of-experts file: room for a third of
    // them, and for none
    let third = std::fs::metadata(&path).unwrap().len() as usize / 3;
    for cap in [third, 0] {
        let (got, report) = run(Some(cap));
        let report = report.expect("experts run on the CPU");
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        eprintln!("{cap} bytes: {report}; worst logit off by {worst} of {scale}");
        assert!(worst <= TOLERANCE * scale, "{cap} bytes");
        let top = (0..got.len()).max_by(|&a, &b| got[a].total_cmp(&got[b]));
        assert!(!clear || top == Some(order[0]), "{cap} bytes");
    }
}
