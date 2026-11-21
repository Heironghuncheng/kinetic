use crate::attention::{AttentionShape, KvCache};
use crate::batch::Batch;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::gemv::GemvInput;
use crate::kernels::Kernels;
use crate::memory::DeviceBuffer;

/// Most relative terms per head.
const MAX_REL: usize = 16;

/// One layer of attention by relative distance: the heads, and where each
/// part sits in a token's row of projections, whose queries start it.
#[derive(Clone, Debug)]
pub struct RelativeShape {
    /// Heads, their width, the window, the score scale and the epsilon of
    /// the values' norm; its rotary and gate fields are not read.
    pub heads: AttentionShape,
    /// Relative terms per query head.
    pub n_rel: usize,
    /// Distances that get a bias; farther keys get none.
    pub extent: usize,
    /// Values per token row.
    pub stride: usize,
    /// Where the per-value gates, the relative terms, the sink logits, the
    /// keys and the values start in a row.
    pub gate_at: usize,
    pub rel_at: usize,
    pub sink_at: usize,
    pub k_at: usize,
    pub v_at: usize,
    /// The largest norm of a query or key head, and of a head's relative
    /// terms.
    pub qk_clip: f32,
    pub rel_clip: f32,
    /// The floor and alpha of the log scale of queries and relative terms.
    pub log_scale: Option<(f32, f32)>,
}

impl RelativeShape {
    fn fits(&self, qkv: usize, scales: usize, rel_bias: usize, rows: usize) -> bool {
        let h = &self.heads;
        let parts = [
            (0, h.n_head * h.head_dim),
            (self.gate_at, h.n_head * h.head_dim),
            (self.rel_at, h.n_head * self.n_rel),
            (self.sink_at, h.n_head),
            (self.k_at, h.kv_dim()),
            (self.v_at, h.kv_dim()),
        ];
        h.check().is_ok()
            && (1..=MAX_REL).contains(&self.n_rel)
            && parts.iter().all(|&(at, len)| at + len <= self.stride)
            && qkv >= rows * self.stride
            && scales == h.n_head * (h.head_dim + self.n_rel)
            && rel_bias == self.extent * self.n_rel
    }
}

impl Kernels {
    /// Records attention by relative distance for every token of the
    /// batch. `qkv` holds each token's projections as `shape` says; queries,
    /// relative terms, keys and values are readied in place, keys and values
    /// stored in `cache`, and the gated outputs written to `out`. `scales`
    /// holds the queries' scales, head after head, then the relative
    /// terms'; `rel_bias` the `n_rel` weights of each distance.
    #[allow(clippy::too_many_arguments)]
    pub fn relative_attention(
        &self,
        batch: &Batch,
        shape: &RelativeShape,
        qkv: &DeviceBuffer<f32>,
        scales: &DeviceBuffer<f32>,
        rel_bias: &DeviceBuffer<f32>,
        cache: &KvCache,
        out: &GemvInput,
        commands: &mut Commands,
    ) -> Result<()> {
        let h = &shape.heads;
        let rows = batch.rows();
        let fits = shape.fits(qkv.len(), scales.len(), rel_bias.len(), rows)
            && cache.positions() >= batch.positions()
            && cache.rows >= rows
            && 2 * cache.k.len() == cache.slots() * h.kv_dim()
            && out.width() == h.n_head * h.head_dim
            && out.rows() >= rows;
        if !fits {
            return Err(Error::Shape(format!(
                "relative attention of {rows} rows of shape {shape:?}"
            )));
        }
        let (floor, alpha) = shape.log_scale.unwrap_or((1.0, 0.0));
        let attend = match h.head_dim {
            64 => &self.attend_64,
            128 => &self.attend_128,
            _ => &self.attend_256,
        };
        let batch_words = batch.device().allocation();
        let [ox, oq, od] = out.outputs();
        // SAFETY: relative.wgsl takes the batch, qkv, scales and both caches
        // at bindings 0..4 and then its shape; attend.wgsl the batch, qkv,
        // both caches, the three output buffers and the bias table at 0..7
        // and its shape. Every index they form stays inside the sizes checked
        // above: token rows below the batch's rows, parts inside a row,
        // positions below the batch's positions, distances below the extent.
        // Workgroups past the pass's tokens return early.
        unsafe {
            commands.dispatch(
                &self.relative,
                &[
                    batch_words,
                    qkv.allocation(),
                    scales.allocation(),
                    cache.k.allocation(),
                    cache.v.allocation(),
                ],
                &[
                    h.n_head as u32,
                    h.n_head_kv as u32,
                    h.head_dim as u32,
                    shape.n_rel as u32,
                    shape.stride as u32,
                    shape.rel_at as u32,
                    shape.k_at as u32,
                    shape.v_at as u32,
                    shape.qk_clip.to_bits(),
                    shape.rel_clip.to_bits(),
                    floor.to_bits(),
                    alpha.to_bits(),
                    h.eps.to_bits(),
                    cache.slots() as u32,
                    0,
                    (h.n_head * h.head_dim) as u32,
                ],
                [(h.n_head + 2 * h.n_head_kv) as u32, rows as u32, 1],
            );
            commands.dispatch(
                attend,
                &[
                    batch_words,
                    qkv.allocation(),
                    cache.k.allocation(),
                    cache.v.allocation(),
                    ox,
                    oq,
                    od,
                    rel_bias.allocation(),
                ],
                &[
                    h.n_head as u32,
                    h.n_head_kv as u32,
                    shape.stride as u32,
                    h.head_dim as u32,
                    h.scale.to_bits(),
                    3,
                    shape.gate_at as u32,
                    h.window.map_or(u32::MAX, |w| w as u32),
                    cache.slots() as u32,
                    shape.n_rel as u32,
                    shape.extent as u32,
                    shape.rel_at as u32,
                    shape.sink_at as u32,
                ],
                [h.n_head as u32, rows as u32, 1],
            );
        }
        Ok(())
    }
}
