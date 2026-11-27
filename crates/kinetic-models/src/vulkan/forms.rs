//! How each weight type runs on each device: with a kernel of the device's
//! own, or from the host. On the host, routed experts run on the CPU's
//! kernels and other matrices are expanded to f32 once at load; either way
//! the values are those the CPU path computes with.

use std::collections::BTreeMap;
use std::fmt;

use kinetic_quant::{dequantize_row, Matrix, TensorType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    Kernel,
    Host,
}

/// The weight types Vulkan has kernels for.
pub(crate) const VULKAN: &[TensorType] = &[
    TensorType::F32,
    TensorType::F16,
    TensorType::Q8_0,
    TensorType::Q4K,
    TensorType::Q5K,
    TensorType::Q6K,
    TensorType::IQ4NL,
    TensorType::NVFP4,
    TensorType::IQ3XXS,
    TensorType::IQ2S,
    TensorType::IQ3S,
    TensorType::Q3K,
    TensorType::IQ2XS,
];

pub(crate) fn on_vulkan(ty: TensorType) -> Form {
    match VULKAN.contains(&ty) {
        true => Form::Kernel,
        false => Form::Host,
    }
}

/// `m` as f32 rows, the values the CPU path reads.
pub(crate) fn expanded(m: &Matrix) -> Vec<u8> {
    let mut row = vec![0.0; m.cols];
    let mut out = Vec::with_capacity(4 * m.rows * m.cols);
    for r in 0..m.rows {
        dequantize_row(m.ty, m.data, r, &mut row).expect("every type expands");
        out.extend(row.iter().flat_map(|v| v.to_le_bytes()));
    }
    out
}

/// The bytes of the matrices of each type that run from the host, packed
/// and as a device holds them.
#[derive(Default)]
pub struct HostWeights(pub BTreeMap<String, (usize, usize)>);

impl HostWeights {
    pub(crate) fn add(&mut self, ty: TensorType, packed: usize, held: usize) {
        let e = self.0.entry(ty.to_string()).or_default();
        e.0 += packed;
        e.1 += held;
    }
}

impl fmt::Display for HostWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const MIB: f64 = (1 << 20) as f64;
        let parts: Vec<String> = self
            .0
            .iter()
            .map(|(ty, &(packed, held))| {
                format!(
                    "{ty} {:.0} MiB as {:.0} MiB",
                    packed as f64 / MIB,
                    held as f64 / MIB
                )
            })
            .collect();
        write!(f, "{}", parts.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table says what the Vulkan crate runs, for every type.
    #[test]
    fn the_vulkan_table_matches_its_kernels() {
        for ty in (0..=u8::MAX as u32).filter_map(TensorType::from_id) {
            let kernel = kinetic_vulkan::has_kernel(ty);
            assert_eq!(on_vulkan(ty) == Form::Kernel, kernel, "{ty}");
        }
    }
}
