//! Attention by relative distance on the device (Inkling 2). Its layers run
//! inside their network, see `network.rs`; each pass of the loop has its
//! own state but the weights of its physical layer.

use std::rc::Rc;

use kinetic_cuda::{
    AttentionShape, Batch, ConvState, DeviceBuffer, DeviceMatrix, GemvInput, KvCache, RelIo,
    RelShape, RelWeights, RmsNorm, Stream, MAX_BATCH,
};

use super::network::gemv_bytes;
use super::{cuda_error, Gpu};
use crate::blocks::RelAttention;
use crate::LoadError;

/// The weights of one physical layer.
pub(super) struct RelLayer {
    norms: [DeviceBuffer<f32>; 3],
    q: DeviceMatrix,
    k: DeviceMatrix,
    v: DeviceMatrix,
    o: DeviceMatrix,
    gate: DeviceMatrix,
    sink: DeviceMatrix,
    rel: DeviceMatrix,
    dynamic: Option<[DeviceMatrix; 2]>,
    q_scale: DeviceBuffer<f32>,
    rel_scale: DeviceBuffer<f32>,
    rel_bias: DeviceBuffer<f32>,
    convs: [DeviceBuffer<f32>; 4],
    taps: usize,
    shape: RelShape,
}

/// A layer slot: its physical layer's weights, owned by the slot of the
/// first pass and shared by those of later passes.
pub(super) struct CudaRelative {
    pub weights: Rc<RelLayer>,
    pub shared: bool,
}

/// The keys and values of a slot, and the inputs of its key, value,
/// attention output and feed-forward output convolutions.
pub(super) struct RelState {
    pub kv: KvCache,
    pub convs: [ConvState; 4],
}

/// Working buffers of attention, for batches of up to `rows` tokens.
pub(super) struct RelBuffers {
    normed: [GemvInput; 3],
    q: DeviceBuffer<f32>,
    r: DeviceBuffer<f32>,
    sink: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    taps: [DeviceBuffer<f32>; 2],
    convolved: [DeviceBuffer<f32>; 2],
    partial: DeviceBuffer<f32>,
    out: GemvInput,
    y: DeviceBuffer<f32>,
}

impl RelLayer {
    pub fn upload(gpu: &Gpu, a: &RelAttention) -> Result<RelLayer, LoadError> {
        let m = |m| gpu.matrix(m).map_err(cuda_error);
        let v = |v: &[f32]| gpu.vector(v).map_err(cuda_error);
        let shape = RelShape {
            n_head: a.n_head,
            n_head_kv: a.n_head_kv,
            head_dim: a.head_dim,
            n_rel: a.n_rel,
            window: a.window,
            qk_clip: a.qk_clip,
            rel_clip: a.rel_clip,
            log_scale: a.log_scale,
            eps: a.eps,
        };
        Ok(RelLayer {
            norms: [v(&a.norms[0])?, v(&a.norms[1])?, v(&a.norms[2])?],
            q: m(&a.q)?,
            k: m(&a.k)?,
            v: m(&a.v)?,
            o: m(&a.o)?,
            gate: m(&a.gate)?,
            sink: m(&a.sink)?,
            rel: m(&a.rel)?,
            dynamic: match &a.dynamic {
                Some([k, v]) => Some([m(k)?, m(v)?]),
                None => None,
            },
            q_scale: v(&a.q_scale)?,
            rel_scale: v(&a.rel_scale)?,
            rel_bias: v(&a.rel_bias)?,
            convs: [
                v(&a.convs[0])?,
                v(&a.convs[1])?,
                v(&a.convs[2])?,
                v(&a.convs[3])?,
            ],
            taps: a.taps,
            shape,
        })
    }

    pub fn size_bytes(&self) -> usize {
        let matrices = [
            &self.q, &self.k, &self.v, &self.o, &self.gate, &self.sink, &self.rel,
        ];
        let small = self.norms.iter().chain(&self.convs).chain([
            &self.q_scale,
            &self.rel_scale,
            &self.rel_bias,
        ]);
        matrices
            .into_iter()
            .chain(self.dynamic.iter().flatten())
            .map(DeviceMatrix::size_bytes)
            .sum::<usize>()
            + 4 * small.map(DeviceBuffer::len).sum::<usize>()
    }

    /// Query heads, key-value heads and relative terms per head, which
    /// size the buffers every layer of a network shares.
    pub fn heads(&self) -> (usize, usize, usize) {
        let s = &self.shape;
        (s.n_head, s.n_head_kv, s.n_rel)
    }

    fn kv_dim(&self) -> usize {
        self.shape.n_head_kv * self.shape.head_dim
    }

    pub fn window(&self) -> Option<usize> {
        self.shape.window
    }

    /// Channels and taps of each convolution, in the order of the state's.
    pub fn convs(&self) -> [(usize, usize); 4] {
        let d = self.o.rows();
        [self.kv_dim(), self.kv_dim(), d, d].map(|c| (c, self.taps))
    }

    /// A fresh state for up to `capacity` positions run in batches of up
    /// to `rows`, with room for the tokens a pass that verifies a draft
    /// runs between a checkpoint and its rollback.
    pub fn new_state(
        &self,
        gpu: &Gpu,
        capacity: usize,
        rows: usize,
    ) -> kinetic_cuda::Result<RelState> {
        let s = &self.shape;
        let attention = AttentionShape {
            n_head: s.n_head,
            n_head_kv: s.n_head_kv,
            head_dim: s.head_dim,
            rope_dims: 0,
            rope_base: 0.0,
            rope_adjacent: false,
            rope_mscale: 1.0,
            norm_after_rope: false,
            scale: 1.0,
            eps: s.eps,
            window: s.window,
            interleaved_gate: false,
            v_norm: true,
        };
        let slots = match s.window {
            Some(w) => w + rows.max(MAX_BATCH) + MAX_BATCH,
            None => capacity,
        };
        let since = super::wide_verify(rows.max(MAX_BATCH));
        let conv = |(c, taps)| ConvState::new(&gpu.ctx, &gpu.stream, c, taps, since);
        let [a, b, c, d] = self.convs();
        Ok(RelState {
            kv: KvCache::new(&gpu.ctx, &attention, slots)?,
            convs: [conv(a)?, conv(b)?, conv(c)?, conv(d)?],
        })
    }

    pub fn buffers(&self, gpu: &Gpu, rows: usize) -> kinetic_cuda::Result<RelBuffers> {
        let s = &self.shape;
        let d = self.q.cols();
        let width = s.n_head * s.head_dim;
        // windowed layers have no dynamic taps, full ones of the same network do
        let heads_kv = self.taps * s.n_head_kv;
        let f32s = |len: usize| DeviceBuffer::<f32>::new(&gpu.ctx, (rows * len).max(1));
        let input = || GemvInput::batch(&gpu.ctx, d, rows);
        Ok(RelBuffers {
            normed: [input()?, input()?, input()?],
            q: f32s(width)?,
            r: f32s(s.n_head * s.n_rel)?,
            sink: f32s(s.n_head)?,
            gate: f32s(width)?,
            k: f32s(self.kv_dim())?,
            v: f32s(self.kv_dim())?,
            taps: [f32s(heads_kv)?, f32s(heads_kv)?],
            convolved: [f32s(self.kv_dim())?, f32s(self.kv_dim())?],
            partial: DeviceBuffer::new(&gpu.ctx, s.partial_len())?,
            out: GemvInput::batch(&gpu.ctx, width, rows)?,
            y: f32s(d)?,
        })
    }

    /// Queues attention over the query, key and value `inputs` of the batch,
    /// before their norms, and leaves its output after its convolution in
    /// `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        gpu: &Gpu,
        b: &mut RelBuffers,
        inputs: [&DeviceBuffer<f32>; 3],
        state: &mut RelState,
        batch: &Batch,
        out: &mut DeviceBuffer<f32>,
    ) -> kinetic_cuda::Result<()> {
        let (k, stream, side) = (&gpu.kernels, &gpu.stream, &gpu.side);
        let n = batch.len();
        let norm = |j: usize| RmsNorm {
            weight: &self.norms[j],
            eps: self.shape.eps,
        };
        let [x0, x1, x2] = &mut b.normed;
        k.rms_norm(inputs[0], norm(0), x0, 0, n, stream)?;
        // The keys and values run beside the query side's products.
        gpu.fork()?;
        k.rms_norm(inputs[1], norm(1), x1, 0, n, side)?;
        k.rms_norm(inputs[2], norm(2), x2, 0, n, side)?;
        let gemv = |w: &DeviceMatrix, x: &GemvInput, y: &mut DeviceBuffer<f32>, on: &Stream| {
            k.gemv_rows(w, w.rows(), x, 0..n, y, on)
        };
        gemv(&self.k, x1, &mut b.k, side)?;
        gemv(&self.v, x2, &mut b.v, side)?;
        let heads = self.shape.n_head_kv;
        let [kc, vc] = &mut b.convolved;
        let [kt, vt] = &mut b.taps;
        let dynamic = match &self.dynamic {
            Some([kd, vd]) => {
                gemv(kd, x1, kt, side)?;
                gemv(vd, x2, vt, side)?;
                [Some((&*kt, heads)), Some((&*vt, heads))]
            }
            None => [None, None],
        };
        let [ks, vs, os, _] = &mut state.convs;
        k.rel_conv(&self.convs[0], &b.k, dynamic[0], ks, kc, batch, side)?;
        k.rel_conv(&self.convs[1], &b.v, dynamic[1], vs, vc, batch, side)?;
        gemv(&self.q, x0, &mut b.q, stream)?;
        gemv(&self.rel, x0, &mut b.r, stream)?;
        gemv(&self.sink, x0, &mut b.sink, stream)?;
        gemv(&self.gate, x0, &mut b.gate, stream)?;
        gpu.join()?;
        let weights = RelWeights {
            q_scale: &self.q_scale,
            rel_scale: &self.rel_scale,
            rel_bias: &self.rel_bias,
        };
        let io = RelIo {
            q: &mut b.q,
            r: &mut b.r,
            sink: &b.sink,
            gate: &b.gate,
            k: kc,
            v: vc,
            partial: &mut b.partial,
            out: &mut b.out,
        };
        k.rel_attention(&self.shape, weights, io, &mut state.kv, batch, stream)?;
        gemv(&self.o, &b.out, &mut b.y, stream)?;
        k.rel_conv(&self.convs[2], &b.y, None, os, out, batch, stream)
    }

    /// Queues the feed-forward output convolution of `y` into `out`.
    pub fn ffn_conv(
        &self,
        gpu: &Gpu,
        y: &DeviceBuffer<f32>,
        state: &mut RelState,
        batch: &Batch,
        out: &mut DeviceBuffer<f32>,
    ) -> kinetic_cuda::Result<()> {
        let (k, stream) = (&gpu.kernels, &gpu.stream);
        k.rel_conv(
            &self.convs[3],
            y,
            None,
            &mut state.convs[3],
            out,
            batch,
            stream,
        )
    }
}

impl RelBuffers {
    pub fn size_bytes(&self) -> usize {
        let f32s = [
            &self.q,
            &self.r,
            &self.sink,
            &self.gate,
            &self.k,
            &self.v,
            &self.partial,
            &self.y,
        ]
        .into_iter()
        .chain(&self.taps)
        .chain(&self.convolved);
        4 * f32s.map(DeviceBuffer::len).sum::<usize>()
            + self
                .normed
                .iter()
                .chain([&self.out])
                .map(gemv_bytes)
                .sum::<usize>()
    }
}

impl RelState {
    pub fn size_bytes(&self) -> usize {
        self.kv.size_bytes() + self.convs.iter().map(ConvState::size_bytes).sum::<usize>()
    }
}
