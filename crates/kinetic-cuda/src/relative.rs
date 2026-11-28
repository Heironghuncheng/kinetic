use crate::attention::KvCache;
use crate::batch::Batch;
use crate::error::{Error, Result};
use crate::gemv::{GemvInput, MAX_BATCH};
use crate::kernels::{launch_struct, Kernels};
use crate::memory::DeviceBuffer;
use crate::module::LaunchConfig;
use crate::short_conv::ConvState;
use crate::stream::Stream;

/// Parts each head's keys split into; must match relative.cuh.
const PARTS: usize = 32;
const THREADS: u32 = 128;
const MAX_HEAD: usize = 256;
const MAX_REL: usize = 32;

/// The fixed shape of an attention layer by relative distance (Inkling 2).
#[derive(Clone, Copy, Debug)]
pub struct RelShape {
    pub n_head: usize,
    pub n_head_kv: usize,
    /// A multiple of 32 up to 256.
    pub head_dim: usize,
    /// Relative terms per query head, at most 32.
    pub n_rel: usize,
    pub window: Option<usize>,
    /// The largest norm of a query or key head, and of a head's relative
    /// terms.
    pub qk_clip: f32,
    pub rel_clip: f32,
    /// The floor and alpha of `1 + alpha ln(max(1, (pos + 1) / floor))`,
    /// which scales queries and relative terms at position `pos`.
    pub log_scale: Option<(f32, f32)>,
    pub eps: f32,
}

impl RelShape {
    fn check(&self) -> Result<()> {
        let ok = self.head_dim.is_multiple_of(32)
            && (32..=MAX_HEAD).contains(&self.head_dim)
            && (1..=MAX_REL).contains(&self.n_rel)
            && self.n_head_kv > 0
            && self.n_head.is_multiple_of(self.n_head_kv)
            && self.window != Some(0);
        if ok {
            Ok(())
        } else {
            Err(Error::Shape(format!(
                "unsupported relative attention shape {self:?}"
            )))
        }
    }

    /// Scratch values [`Kernels::rel_attention`] needs.
    pub fn partial_len(&self) -> usize {
        MAX_BATCH * self.n_head * PARTS * (self.head_dim + 2)
    }
}

/// The small per-layer tensors of an attention layer by relative distance.
#[derive(Clone, Copy)]
pub struct RelWeights<'a> {
    /// Multiply the query heads and the relative terms before they are
    /// clipped.
    pub q_scale: &'a DeviceBuffer<f32>,
    pub rel_scale: &'a DeviceBuffer<f32>,
    /// `n_rel` weights per distance below the extent; farther keys get no
    /// bias.
    pub rel_bias: &'a DeviceBuffer<f32>,
}

/// The projections of a batch, rows of `n_head` heads (`n_head_kv` for keys
/// and values), and the buffers attention writes.
pub struct RelIo<'a> {
    /// Clipped in place, as are the relative terms.
    pub q: &'a mut DeviceBuffer<f32>,
    pub r: &'a mut DeviceBuffer<f32>,
    /// One logit per head and token, a key whose value is zero.
    pub sink: &'a DeviceBuffer<f32>,
    /// Gates every value of the output through a sigmoid.
    pub gate: &'a DeviceBuffer<f32>,
    /// Clipped and normalized in place, then stored.
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a mut DeviceBuffer<f32>,
    /// At least [`RelShape::partial_len`] values.
    pub partial: &'a mut DeviceBuffer<f32>,
    pub out: &'a mut GemvInput,
}

#[repr(C)]
struct ConvArgs {
    batch: u64,
    x: u64,
    w: u64,
    dynamic: u64,
    ring: u64,
    y: u64,
    n: u32,
    channels: u32,
    taps: u32,
    ring_len: u32,
    group: u32,
    scale: f32,
}

#[repr(C)]
struct PrepareArgs {
    batch: u64,
    q: u64,
    r: u64,
    k: u64,
    v: u64,
    q_scale: u64,
    rel_scale: u64,
    k_cache: u64,
    v_cache: u64,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    n_rel: u32,
    slots: u32,
    qk_clip: f32,
    rel_clip: f32,
    alpha: f32,
    floor_inv: f32,
    eps: f32,
}

#[repr(C)]
struct AttendArgs {
    pos: u64,
    q: u64,
    r: u64,
    k_cache: u64,
    v_cache: u64,
    rel_bias: u64,
    partial: u64,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    n_rel: u32,
    slots: u32,
    window: u32,
    extent: u32,
    scale: f32,
}

#[repr(C)]
struct CombineArgs {
    partial: u64,
    sink: u64,
    gate: u64,
    out: crate::gemv::Act,
    n_head: u32,
}

impl Kernels {
    /// Queues `y = (x + c) / sqrt(k + 1)` for every token of `batch`, where
    /// `c` is the causal convolution of each channel of `x` with its taps in
    /// `w`, oldest input first, over the inputs before the batch that
    /// `state` holds, and `k` the number of taps. With `dynamic`, every
    /// token also has as many taps per head of channels, newest input
    /// first, that add to `c` and count in `k`. Stores the batch's inputs
    /// in `state`.
    #[allow(clippy::too_many_arguments)]
    pub fn rel_conv(
        &self,
        w: &DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        dynamic: Option<(&DeviceBuffer<f32>, usize)>,
        state: &mut ConvState,
        y: &mut DeviceBuffer<f32>,
        batch: &Batch,
        stream: &Stream,
    ) -> Result<()> {
        let (n, c, taps) = (batch.len(), state.channels, state.taps);
        if w.len() != c * taps || dynamic.is_some_and(|(_, heads)| !c.is_multiple_of(heads)) {
            return Err(Error::Shape(format!(
                "convolution of {c} channels with {} taps",
                w.len()
            )));
        }
        x.check_holds(n * c, "convolution input")?;
        y.check_holds(n * c, "convolution output")?;
        if let Some((d, heads)) = dynamic {
            d.check_holds(n * heads * taps, "dynamic taps")?;
        }
        let k = taps * if dynamic.is_some() { 2 } else { 1 };
        let args = ConvArgs {
            batch: batch.device_ptr(),
            x: x.device_ptr(),
            w: w.device_ptr(),
            dynamic: dynamic.map_or(0, |(d, _)| d.device_ptr()),
            ring: state.ring.device_ptr(),
            y: y.device_ptr(),
            n: n as u32,
            channels: c as u32,
            taps: taps as u32,
            ring_len: state.ring_len as u32,
            group: dynamic.map_or(c, |(_, heads)| c / heads) as u32,
            scale: 1.0 / ((k + 1) as f32).sqrt(),
        };
        let blocks = (c as u32).div_ceil(THREADS);
        let config = LaunchConfig::linear(blocks, THREADS);
        // SAFETY: matches `RelConvArgs`; buffers hold what `n` tokens of `c`
        // channels touch, as checked, and ring rows wrap below `ring_len`.
        unsafe {
            let all = LaunchConfig {
                grid: [blocks, n as u32, 1],
                ..config
            };
            launch_struct(&self.rel_conv, all, stream, &args)?;
            launch_struct(&self.rel_conv_store, config, stream, &args)
        }
    }

    /// Queues attention by relative distance for every token of `batch`:
    /// clips the query heads and relative terms, scaled by position on
    /// layers with a log scale, and the key heads, normalizes the value
    /// heads, stores keys and values in `cache` and attends each query over
    /// its causal (and windowed) past, each key biased by the dot product of
    /// the query's relative terms with the bias weights of its distance, and
    /// over its sink, then gates the result into `io.out`. A full cache
    /// stores position `p` at slot `p`; a windowed one must hold the window
    /// and the batch. Tokens attend [`MAX_BATCH`] at a time, so a batch
    /// gives each token what it gets alone.
    pub fn rel_attention(
        &self,
        s: &RelShape,
        w: RelWeights,
        io: RelIo,
        cache: &mut KvCache,
        batch: &Batch,
        stream: &Stream,
    ) -> Result<()> {
        s.check()?;
        let (n, hd) = (batch.len(), s.head_dim);
        let (width, kv_dim) = (s.n_head * hd, s.n_head_kv * hd);
        let slots = cache.slots();
        let extent = w.rel_bias.len() / s.n_rel;
        let fits = match s.window {
            Some(win) => slots >= win + n,
            None => batch.pos() + n <= slots,
        };
        if !fits || cache.kv_dim() != kv_dim || io.out.width() != width {
            return Err(Error::Shape(format!(
                "{n} tokens at {} into a cache of {slots} for {s:?}",
                batch.pos()
            )));
        }
        io.q.check_holds(n * width, "queries")?;
        io.r.check_holds(n * s.n_head * s.n_rel, "relative terms")?;
        io.sink.check_holds(n * s.n_head, "sinks")?;
        io.gate.check_holds(n * width, "gates")?;
        io.k.check_holds(n * kv_dim, "keys")?;
        io.v.check_holds(n * kv_dim, "values")?;
        io.partial
            .check_holds(s.partial_len(), "attention partials")?;
        w.q_scale.check_holds(width, "query scales")?;
        w.rel_scale
            .check_holds(s.n_head * s.n_rel, "relative scales")?;
        io.out.check_rows(n)?;
        let (k_cache, v_cache) = cache.f16_ptrs()?;
        let (floor, alpha) = s.log_scale.unwrap_or((1.0, 0.0));
        let prepare = PrepareArgs {
            batch: batch.device_ptr(),
            q: io.q.device_ptr(),
            r: io.r.device_ptr(),
            k: io.k.device_ptr(),
            v: io.v.device_ptr(),
            q_scale: w.q_scale.device_ptr(),
            rel_scale: w.rel_scale.device_ptr(),
            k_cache,
            v_cache,
            n_head: s.n_head as u32,
            n_head_kv: s.n_head_kv as u32,
            head_dim: hd as u32,
            n_rel: s.n_rel as u32,
            slots: slots as u32,
            qk_clip: s.qk_clip,
            rel_clip: s.rel_clip,
            alpha,
            floor_inv: 1.0 / floor,
            eps: s.eps,
        };
        // SAFETY: matches `RelPrepareArgs`; the projections hold `n` tokens
        // and the cache every slot taken mod `slots`, as checked.
        unsafe {
            launch_struct(
                &self.rel_prepare,
                LaunchConfig::linear(n as u32, THREADS),
                stream,
                &prepare,
            )?
        };
        let out = io.out.act();
        for first in (0..n).step_by(MAX_BATCH) {
            let m = (n - first).min(MAX_BATCH);
            let attend = AttendArgs {
                pos: batch.step_ptr(first),
                q: io.q.ptr_at(first * width),
                r: io.r.ptr_at(first * s.n_head * s.n_rel),
                k_cache,
                v_cache,
                rel_bias: w.rel_bias.device_ptr(),
                partial: io.partial.device_ptr(),
                n_head: s.n_head as u32,
                n_head_kv: s.n_head_kv as u32,
                head_dim: hd as u32,
                n_rel: s.n_rel as u32,
                slots: slots as u32,
                window: s.window.unwrap_or(0) as u32,
                extent: extent as u32,
                scale: 1.0 / (hd as f32).sqrt(),
            };
            let combine = CombineArgs {
                partial: io.partial.device_ptr(),
                sink: io.sink.ptr_at(first * s.n_head),
                gate: io.gate.ptr_at(first * width),
                out: out.skip_inputs(first, width),
                n_head: s.n_head as u32,
            };
            let parts = LaunchConfig {
                grid: [PARTS as u32, s.n_head as u32, m as u32],
                block: [THREADS, 1, 1],
                shared_bytes: 0,
            };
            let heads = LaunchConfig {
                grid: [s.n_head as u32, m as u32, 1],
                block: [hd as u32, 1, 1],
                shared_bytes: 0,
            };
            // SAFETY: match `RelAttendArgs` and `RelCombineArgs`; the step
            // reads `m` tokens from `first` and the cache, and the partials
            // hold a step's parts, as checked.
            unsafe {
                launch_struct(&self.rel_attend, parts, stream, &attend)?;
                launch_struct(&self.rel_combine, heads, stream, &combine)?;
            }
        }
        Ok(())
    }
}
