use kinetic_quant::{Matrix, TensorType};
use kinetic_vulkan::{Commands, DeviceBuffer, DeviceMatrix, Out, Rows, Shared};

use super::layer::R;
use super::work::Work;
use super::Gpu;
use crate::blocks::{Experts, Gating, Moe};
use crate::LoadError;

/// The routed experts of a layer on the device; see [`Moe`]. Their
/// matrices are resident, or for a model larger than the device held in
/// slots that its offload fills.
pub(super) struct VulkanMoe {
    router: DeviceMatrix,
    gating: kinetic_vulkan::Gating,
    /// Added to the probabilities to choose experts: zeros for a softmax.
    bias: DeviceBuffer<f32>,
    weight_scale: f32,
    glu: kinetic_vulkan::Glu,
    n_ff: usize,
    weights: Option<ExpertWeights>,
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

/// [`ExpertWeights`] wherever they are held.
pub(super) struct ExpertView<'a> {
    pub gate_up: Vec<(&'a DeviceMatrix, usize)>,
    pub down: &'a DeviceMatrix,
    pub count: usize,
}

impl ExpertWeights {
    fn view(&self) -> ExpertView<'_> {
        ExpertView {
            gate_up: self.gate_up.iter().map(|(m, at)| (m, *at)).collect(),
            down: &self.down,
            count: self.count,
        }
    }
}

/// The experts of `m` the products run: the gate and up experts, each with
/// where it writes in a row of both (one product when they are the halves
/// of one tensor), then the down experts.
pub(super) fn sources<'a>(m: &Moe<'a>) -> (Vec<(Experts<'a>, usize)>, Experts<'a>) {
    let (gate, up) = (m.gate, m.up);
    let half = gate.expert_bytes();
    let joined = gate.stride == 2 * half
        && (up.stride, up.rows, up.cols, up.ty) == (gate.stride, gate.rows, gate.cols, gate.ty)
        && gate.data.get(half..).map(<[u8]>::as_ptr) == Some(up.data.as_ptr());
    let gate_up = match joined {
        true => vec![(
            Experts {
                rows: 2 * gate.rows,
                ..gate
            },
            0,
        )],
        false => vec![(gate, 0), (up, gate.rows)],
    };
    (gate_up, m.down)
}

/// Every expert of `e` as one matrix, if they follow each other in memory.
fn stacked<'a>(e: &Experts<'a>, count: usize) -> Result<Matrix<'a>, LoadError> {
    let bytes = e.expert_bytes();
    (e.stride == bytes)
        .then(|| Matrix::new(e.ty, &e.data[..count * bytes], count * e.rows, e.cols))
        .ok_or_else(|| LoadError("expert matrices are interleaved".into()))
}

impl VulkanMoe {
    /// Uploads the routing of `m`, and its experts if `resident`.
    pub fn upload(gpu: &Gpu, m: &Moe, resident: bool) -> Result<Self, LoadError> {
        let n = m.n_experts();
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
        let mut moe = VulkanMoe {
            router: gpu.matrix(&m.router)?,
            gating,
            bias: gpu.vector(&bias)?,
            weight_scale: m.weight_scale,
            glu: match m.glu {
                Glu::Silu => kinetic_vulkan::Glu::Silu,
                Glu::Situ { beta, linear } => kinetic_vulkan::Glu::Situ { beta, linear },
            },
            n_ff: m.gate.rows,
            weights: None,
            shared_gate,
            used: m.n_used,
        };
        if resident {
            moe.upload_experts(gpu, m)?;
        }
        Ok(moe)
    }

    /// Uploads every expert of `m`, the routed experts of this layer.
    pub fn upload_experts(&mut self, gpu: &Gpu, m: &Moe) -> Result<(), LoadError> {
        let n = m.n_experts();
        let (gate_up, down) = sources(m);
        let gate_up = gate_up
            .iter()
            .map(|(e, at)| Ok((gpu.matrix(&stacked(e, n)?)?, *at)))
            .collect::<Result<_, LoadError>>()?;
        self.weights = Some(ExpertWeights {
            gate_up,
            down: gpu.matrix(&stacked(&down, n)?)?,
            count: n,
        });
        Ok(())
    }

    pub fn n_experts(&self) -> usize {
        self.router.rows()
    }

    /// The width of one expert's feed-forward.
    pub fn n_ff(&self) -> usize {
        self.n_ff
    }

    pub fn size_bytes(&self) -> usize {
        let experts = self.weights.as_ref().map_or(0, |w| {
            let gate_up: usize = w.gate_up.iter().map(|(m, _)| m.size_bytes()).sum();
            gate_up + w.down.size_bytes()
        });
        self.router.size_bytes() + experts
    }

    /// Reads `w.h` and leaves the layer's output in `w.y`, which holds the
    /// shared expert's output if `shared`. The experts must be resident.
    pub fn record(&self, gpu: &Gpu, w: &Work, shared: bool, c: &mut Commands) -> R {
        let weights = self.weights.as_ref().expect("experts are on the device");
        self.record_route(gpu, w, c)?;
        self.record_products(gpu, w, &weights.view(), c)?;
        self.record_mix(gpu, w, shared, c)
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

    /// Records the products of the picks of `w`, which name experts of
    /// `weights`, into the rows of each pick's output; picks of experts
    /// past `weights` are left alone.
    pub fn record_products(
        &self,
        gpu: &Gpu,
        w: &Work,
        weights: &ExpertView,
        c: &mut Commands,
    ) -> R {
        let k = &gpu.kernels;
        let (n_ff, used, d, count) = (self.n_ff, self.used, w.d, weights.count);
        let e = w.experts.as_ref().expect("work has expert buffers");
        c.independent(|c| {
            for &(m, offset) in &weights.gate_up {
                let out = Out {
                    stride: 2 * n_ff,
                    offset,
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
            weights.down,
            count,
            used,
            &e.picks,
            act,
            1,
            &e.routed,
            out,
            &w.batch,
            c,
        )
    }

    /// Records the mix of the picks' outputs into `w.y`, which holds the
    /// shared expert's output if `shared`.
    pub fn record_mix(&self, gpu: &Gpu, w: &Work, shared: bool, c: &mut Commands) -> R {
        let e = w.experts.as_ref().expect("work has expert buffers");
        let shared = match (shared, &self.shared_gate) {
            (false, _) => Shared::None,
            (true, None) => Shared::Plain,
            (true, Some(_)) => Shared::Gated(&e.gate),
        };
        let (used, d) = (self.used, w.d);
        gpu.kernels
            .mix_experts(&w.batch, used, &e.picks, &e.routed, shared, d, &w.y, c)
    }
}
