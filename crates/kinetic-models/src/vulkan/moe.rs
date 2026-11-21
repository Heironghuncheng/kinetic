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
    weight_scale: f32,
    glu: kinetic_vulkan::Glu,
    n_ff: usize,
    weights: ExpertWeights,
    /// The shared expert's gate as a one-row matrix.
    shared_gate: Option<DeviceMatrix>,
    pub used: usize,
}

/// Expert matrices of equal shape one after another, `count` of each.
pub(super) struct ExpertWeights {
    /// The gate and up products and where each writes in a row of both: two
    /// matrices, or one whose experts hold their gate rows and then their up
    /// rows.
    pub gate_up: Vec<(DeviceMatrix, usize)>,
    pub down: DeviceMatrix,
    pub count: usize,
}

/// Every expert of `e` as one matrix, if they follow each other in memory.
fn stacked<'a>(e: &Experts<'a>, count: usize) -> Option<Matrix<'a>> {
    let bytes = e.expert_bytes();
    (e.stride == bytes).then(|| Matrix::new(e.ty, &e.data[..count * bytes], count * e.rows, e.cols))
}

/// The gate and up halves of one tensor as that tensor, if they are.
fn fused<'a>(gate: &Experts<'a>, up: &Experts<'a>, count: usize) -> Option<Matrix<'a>> {
    let half = gate.expert_bytes();
    let joined = gate.stride == 2 * half
        && (up.stride, up.rows, up.cols, up.ty) == (gate.stride, gate.rows, gate.cols, gate.ty)
        && gate.data.get(half..).map(<[u8]>::as_ptr) == Some(up.data.as_ptr());
    joined.then(|| {
        Matrix::new(
            gate.ty,
            &gate.data[..count * gate.stride],
            count * 2 * gate.rows,
            gate.cols,
        )
    })
}

impl VulkanMoe {
    pub fn upload(gpu: &Gpu, m: &Moe) -> Result<Self, LoadError> {
        let n = m.n_experts();
        let n_ff = m.gate.rows;
        let gate_up = match (
            fused(&m.gate, &m.up, n),
            stacked(&m.gate, n),
            stacked(&m.up, n),
        ) {
            (Some(f), _, _) => vec![(gpu.matrix(&f)?, 0)],
            (None, Some(g), Some(u)) => vec![(gpu.matrix(&g)?, 0), (gpu.matrix(&u)?, n_ff)],
            _ => {
                return Err(LoadError(
                    "expert gate and up matrices are interleaved".into(),
                ))
            }
        };
        let down = stacked(&m.down, n)
            .ok_or_else(|| LoadError("expert down matrices are interleaved".into()))?;
        if m.scales.iter().any(Option::is_some) {
            return Err(LoadError(
                "per-expert weight scales do not run on Vulkan yet".into(),
            ));
        }
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
            weight_scale: m.weight_scale,
            glu: match m.glu {
                Glu::Silu => kinetic_vulkan::Glu::Silu,
                Glu::Situ { beta, linear } => kinetic_vulkan::Glu::Situ { beta, linear },
            },
            n_ff,
            weights: ExpertWeights {
                gate_up,
                down: gpu.matrix(&down)?,
                count: n,
            },
            shared_gate,
            used: m.n_used,
        })
    }

    pub fn n_experts(&self) -> usize {
        self.router.rows()
    }

    /// The width of one expert's feed-forward.
    pub fn n_ff(&self) -> usize {
        self.n_ff
    }

    pub fn size_bytes(&self) -> usize {
        let w = &self.weights;
        w.gate_up
            .iter()
            .map(|(m, _)| m)
            .chain([&self.router, &w.down])
            .map(DeviceMatrix::size_bytes)
            .sum()
    }

    /// Reads `w.h` and leaves the layer's output in `w.y`, which holds the
    /// shared expert's output if `shared`.
    pub fn record(&self, gpu: &Gpu, w: &Work, shared: bool, c: &mut Commands) -> R {
        self.record_route(gpu, w, c)?;
        self.record_products(gpu, w, &self.weights, shared, c)
    }

    /// Records the routing of `w.h` into the picks of `w`.
    pub fn record_route(&self, gpu: &Gpu, w: &Work, c: &mut Commands) -> R {
        let k = &gpu.kernels;
        let e = w.experts.as_ref().expect("work has expert buffers");
        let n_exp = self.n_experts();
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
            self.used,
            self.weight_scale,
            &e.picks,
            c,
        )
    }

    /// Records the products of the picked experts of `weights`, which the
    /// picks name, and their mix into `w.y`.
    pub fn record_products(
        &self,
        gpu: &Gpu,
        w: &Work,
        weights: &ExpertWeights,
        shared: bool,
        c: &mut Commands,
    ) -> R {
        let k = &gpu.kernels;
        let (n_ff, used, d, count) = (self.n_ff, self.used, w.d, weights.count);
        let e = w.experts.as_ref().expect("work has expert buffers");
        c.independent(|c| {
            for (m, offset) in &weights.gate_up {
                let out = Out {
                    stride: 2 * n_ff,
                    offset: *offset,
                };
                k.gemv_experts(
                    m, count, used, &e.picks, &w.h, used, &e.gu, out, &w.batch, c,
                )?;
            }
            Ok(())
        })?;
        let act = e.activations(n_ff);
        k.swiglu(&w.batch, used, self.glu, &e.gu, act, c)?;
        let out = Out {
            stride: d,
            offset: 0,
        };
        k.gemv_experts(
            &weights.down,
            count,
            used,
            &e.picks,
            act,
            1,
            &e.routed,
            out,
            &w.batch,
            c,
        )?;
        let shared = match (shared, &self.shared_gate) {
            (false, _) => Shared::None,
            (true, None) => Shared::Plain,
            (true, Some(_)) => Shared::Gated(&e.gate),
        };
        k.mix_experts(&w.batch, used, &e.picks, &e.routed, shared, d, &w.y, c)
    }
}
