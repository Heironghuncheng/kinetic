use kinetic_quant::{Matrix, TensorType};
use kinetic_vulkan::{Commands, DeviceBuffer, DeviceMatrix, Out, Rows, Shared};

use super::layer::R;
use super::work::Work;
use super::Gpu;
use crate::blocks::{Experts, Gating, Moe};
use crate::LoadError;

/// The routed experts of a layer on the device, every expert resident; see
/// [`Moe`].
pub(super) struct VulkanMoe {
    router: DeviceMatrix,
    gating: kinetic_vulkan::Gating,
    /// Added to the probabilities to choose experts: zeros for a softmax.
    bias: DeviceBuffer<f32>,
    gate: DeviceMatrix,
    up: DeviceMatrix,
    down: DeviceMatrix,
    /// The shared expert's gate as a one-row matrix.
    shared_gate: Option<DeviceMatrix>,
    pub used: usize,
}

fn stacked<'a>(e: &Experts<'a>) -> Matrix<'a> {
    Matrix::new(e.ty, e.data, e.count * e.rows, e.cols)
}

impl VulkanMoe {
    pub fn upload(gpu: &Gpu, m: &Moe) -> Result<Self, LoadError> {
        let n = m.n_experts();
        let (gating, bias) = match &m.gating {
            Gating::Softmax => (kinetic_vulkan::Gating::Softmax, vec![0.0; n]),
            Gating::Sigmoid { bias } => (kinetic_vulkan::Gating::Sigmoid, bias.clone()),
        };
        let shared_gate = m
            .shared_gate
            .as_ref()
            .map(|g| {
                let bytes: Vec<u8> = g.iter().flat_map(|v| v.to_le_bytes()).collect();
                gpu.matrix(&Matrix::new(TensorType::F32, &bytes, 1, g.len()))
            })
            .transpose()?;
        Ok(VulkanMoe {
            router: gpu.matrix(&m.router)?,
            gating,
            bias: gpu.vector(&bias)?,
            gate: gpu.matrix(&stacked(&m.gate))?,
            up: gpu.matrix(&stacked(&m.up))?,
            down: gpu.matrix(&stacked(&m.down))?,
            shared_gate,
            used: m.n_used,
        })
    }

    pub fn n_experts(&self) -> usize {
        self.router.rows()
    }

    /// The width of one expert's feed-forward.
    pub fn n_ff(&self) -> usize {
        self.gate.rows() / self.n_experts()
    }

    pub fn size_bytes(&self) -> usize {
        [&self.router, &self.gate, &self.up, &self.down]
            .iter()
            .map(|m| m.size_bytes())
            .sum()
    }

    /// Reads `w.h` and leaves the layer's output in `w.y`, which holds the
    /// shared expert's output if `shared`.
    pub fn record(&self, gpu: &Gpu, w: &Work, shared: bool, c: &mut Commands) -> R {
        let k = &gpu.kernels;
        let (n_exp, n_ff, used, d) = (self.n_experts(), self.n_ff(), self.used, w.d);
        let e = w.experts.as_ref().expect("work has expert buffers");
        c.independent(|c| {
            let out = Out {
                stride: n_exp,
                offset: 0,
            };
            k.gemv_rows(
                &self.router,
                &w.h,
                &e.logits,
                out,
                Rows::Tokens,
                &w.batch,
                c,
            )?;
            if let Some(g) = &self.shared_gate {
                let out = Out {
                    stride: 1,
                    offset: 0,
                };
                k.gemv_rows(g, &w.h, &e.gate, out, Rows::Tokens, &w.batch, c)?;
            }
            Ok(())
        })?;
        k.route(
            &w.batch,
            self.gating,
            &e.logits,
            &self.bias,
            n_exp,
            used,
            &e.picks,
            c,
        )?;
        c.independent(|c| {
            for (m, offset) in [(&self.gate, 0), (&self.up, n_ff)] {
                let out = Out {
                    stride: 2 * n_ff,
                    offset,
                };
                k.gemv_experts(
                    m, n_exp, used, &e.picks, &w.h, used, &e.gu, out, &w.batch, c,
                )?;
            }
            Ok(())
        })?;
        let act = e.activations(n_ff);
        k.swiglu(&w.batch, used, &e.gu, act, c)?;
        let out = Out {
            stride: d,
            offset: 0,
        };
        k.gemv_experts(
            &self.down, n_exp, used, &e.picks, act, 1, &e.routed, out, &w.batch, c,
        )?;
        let shared = match (shared, &self.shared_gate) {
            (false, _) => Shared::None,
            (true, None) => Shared::Plain,
            (true, Some(_)) => Shared::Gated(&e.gate),
        };
        k.mix_experts(&w.batch, used, &e.picks, &e.routed, shared, d, &w.y, c)
    }
}
