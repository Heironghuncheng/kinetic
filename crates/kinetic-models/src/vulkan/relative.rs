use std::rc::Rc;

use kinetic_quant::Matrix;
use kinetic_vulkan::{
    AttentionGate, AttentionShape, Commands, ConvRing, Device, DeviceBuffer, DeviceMatrix,
    GemvInput, KvCache, Out, RelativeShape, Rows, Strided,
};

use super::layer::R;
use super::work::{Needs, Work};
use super::Gpu;
use crate::blocks::RelAttention;
use crate::LoadError;

/// Attention by relative distance on the device; see [`RelAttention`].
///
/// A token's row of projections holds the queries, gates, relative terms
/// and sink logits, then the keys and values before their convolutions,
/// then on full-attention layers the dynamic taps of keys and values, then
/// the keys and values after their convolutions.
pub(super) struct VulkanRelative {
    pub shape: RelativeShape,
    /// RMS norms of the query, key and value inputs.
    pub norms: [DeviceBuffer<f32>; 3],
    eps: f32,
    q: DeviceMatrix,
    gate: DeviceMatrix,
    rel: DeviceMatrix,
    sink: DeviceMatrix,
    k: DeviceMatrix,
    v: DeviceMatrix,
    o: DeviceMatrix,
    dynamic: Option<[DeviceMatrix; 2]>,
    /// The queries' scales, then the relative terms'.
    scales: DeviceBuffer<f32>,
    rel_bias: DeviceBuffer<f32>,
    /// Taps of the key, value, attention output and feed-forward output
    /// convolutions.
    convs: [DeviceBuffer<f32>; 4],
    taps: usize,
    /// Where the keys and values before their convolutions, and their
    /// dynamic taps, start in a row.
    raw_k: usize,
    raw_v: usize,
    taps_k: usize,
    taps_v: usize,
}

/// What a sequence keeps of one such layer: its keys and values, and the
/// last inputs of each convolution.
pub(super) struct RelativeState {
    pub kv: KvCache,
    pub rings: [ConvRing; 4],
}

impl RelativeState {
    pub fn size_bytes(&self) -> usize {
        self.kv.size_bytes() + self.rings.iter().map(ConvRing::size_bytes).sum::<usize>()
    }
}

impl VulkanRelative {
    pub fn upload(gpu: &Gpu, a: &RelAttention) -> Result<Self, LoadError> {
        let (nh, hd, kv) = (a.n_head, a.head_dim, a.n_head_kv);
        let kv_dim = kv * hd;
        let dyn_taps = a.dynamic.as_ref().map_or(0, |_| a.taps * kv);
        let gate_at = nh * hd;
        let rel_at = gate_at + nh * hd;
        let sink_at = rel_at + nh * a.n_rel;
        let raw_k = sink_at + nh;
        let raw_v = raw_k + kv_dim;
        let taps_k = raw_v + kv_dim;
        let taps_v = taps_k + dyn_taps;
        let k_at = taps_v + dyn_taps;
        let v_at = k_at + kv_dim;
        let shape = RelativeShape {
            heads: AttentionShape {
                n_head: nh,
                n_head_kv: kv,
                head_dim: hd,
                rope_dims: 0,
                rope_adjacent: false,
                norm_after_rope: false,
                gate: AttentionGate::None,
                window: a.window,
                scale: 1.0 / (hd as f32).sqrt(),
                eps: a.eps,
            },
            n_rel: a.n_rel,
            extent: a.rel_bias.len() / a.n_rel,
            stride: v_at + kv_dim,
            gate_at,
            rel_at,
            sink_at,
            k_at,
            v_at,
            qk_clip: a.qk_clip,
            rel_clip: a.rel_clip,
            log_scale: a.log_scale,
        };
        let m = |m: &Matrix| gpu.matrix(m);
        let mut scales = a.q_scale.clone();
        scales.extend(&a.rel_scale);
        Ok(VulkanRelative {
            norms: [
                gpu.vector(&a.norms[0])?,
                gpu.vector(&a.norms[1])?,
                gpu.vector(&a.norms[2])?,
            ],
            eps: a.eps,
            q: m(&a.q)?,
            gate: m(&a.gate)?,
            rel: m(&a.rel)?,
            sink: m(&a.sink)?,
            k: m(&a.k)?,
            v: m(&a.v)?,
            o: m(&a.o)?,
            dynamic: match &a.dynamic {
                Some([k, v]) => Some([m(k)?, m(v)?]),
                None => None,
            },
            scales: gpu.vector(&scales)?,
            rel_bias: gpu.vector(&a.rel_bias)?,
            convs: [
                gpu.vector(&a.convs[0])?,
                gpu.vector(&a.convs[1])?,
                gpu.vector(&a.convs[2])?,
                gpu.vector(&a.convs[3])?,
            ],
            taps: a.taps,
            raw_k,
            raw_v,
            taps_k,
            taps_v,
            shape,
        })
    }

    pub fn size_bytes(&self) -> usize {
        [
            &self.q, &self.gate, &self.rel, &self.sink, &self.k, &self.v, &self.o,
        ]
        .into_iter()
        .chain(self.dynamic.iter().flatten())
        .map(DeviceMatrix::size_bytes)
        .sum()
    }

    pub fn needs(&self) -> Needs {
        let h = &self.shape.heads;
        Needs {
            proj: self.shape.stride,
            width: h.n_head * h.head_dim,
            rec: 0,
        }
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn new_state(
        &self,
        dev: &Rc<Device>,
        positions: usize,
        rows: usize,
    ) -> kinetic_vulkan::Result<RelativeState> {
        let (d, kv_dim) = (self.o.rows(), self.shape.heads.kv_dim());
        let ring = |c| ConvRing::new(dev, c, self.taps, rows);
        Ok(RelativeState {
            kv: KvCache::new(dev, &self.shape.heads, positions, rows)?,
            rings: [ring(kv_dim)?, ring(kv_dim)?, ring(d)?, ring(d)?],
        })
    }

    /// Records attention on the normalized query, key and value inputs
    /// `ins` and leaves its projected output in `w.y`, before its
    /// convolution.
    pub fn record(
        &self,
        gpu: &Gpu,
        w: &Work,
        ins: [&GemvInput; 3],
        state: &RelativeState,
        c: &mut Commands,
    ) -> R {
        let (k, s) = (&gpu.kernels, &self.shape);
        let stride = s.stride;
        let mut products = vec![
            (&self.q, ins[0], 0),
            (&self.gate, ins[0], s.gate_at),
            (&self.rel, ins[0], s.rel_at),
            (&self.sink, ins[0], s.sink_at),
            (&self.k, ins[1], self.raw_k),
            (&self.v, ins[2], self.raw_v),
        ];
        if let Some([dk, dv]) = &self.dynamic {
            products.extend([(dk, ins[1], self.taps_k), (dv, ins[2], self.taps_v)]);
        }
        // The projections write disjoint columns of each row.
        c.independent(|c| {
            for (m, x, offset) in products {
                let out = Out { stride, offset };
                k.gemv_rows(m, x, &w.proj, out, Rows::Tokens, &w.batch, c)?;
            }
            Ok(())
        })?;
        let head_len = s.heads.head_dim;
        let parts = [
            (self.raw_k, self.taps_k, s.k_at),
            (self.raw_v, self.taps_v, s.v_at),
        ];
        c.independent(|c| {
            for (i, (raw, taps, at)) in parts.into_iter().enumerate() {
                let x = Strided {
                    data: &w.proj,
                    stride,
                    at: raw,
                };
                let out = Strided { at, ..x };
                let dynamic = self.dynamic.as_ref().map(|_| (taps, head_len));
                let (weights, ring) = (&self.convs[i], &state.rings[i]);
                k.residual_conv(&w.batch, x, dynamic, weights, ring, out, false, c)?;
            }
            Ok(())
        })?;
        let att = w.mixed(s.heads.n_head * head_len);
        k.relative_attention(
            &w.batch,
            s,
            &w.proj,
            &self.scales,
            &self.rel_bias,
            &state.kv,
            att,
            c,
        )?;
        let out = Out {
            stride: w.d,
            offset: 0,
        };
        k.gemv_rows(&self.o, att, &w.y, out, Rows::Tokens, &w.batch, c)
    }

    /// Records the convolution of `w.y`, the attention output (`i` 2) or the
    /// feed-forward output (`i` 3), into `out`, added to it with
    /// `accumulate`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_conv(
        &self,
        gpu: &Gpu,
        w: &Work,
        i: usize,
        state: &RelativeState,
        out: &DeviceBuffer<f32>,
        accumulate: bool,
        c: &mut Commands,
    ) -> R {
        let rows = |data| Strided {
            data,
            stride: w.d,
            at: 0,
        };
        gpu.kernels.residual_conv(
            &w.batch,
            rows(&w.y),
            None,
            &self.convs[i],
            &state.rings[i],
            rows(out),
            accumulate,
            c,
        )
    }
}
