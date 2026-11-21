use std::rc::Rc;

use crate::batch::Batch;
use crate::commands::Commands;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::gemv::GemvInput;
use crate::kernels::Kernels;
use crate::memory::DeviceBuffer;

/// The recent inputs of a causal convolution: position `p` in slot
/// `p % slots` of a ring with room for `taps - 1` positions and a pass
/// more, so neither a pass nor a rollback overwrites a position still read.
pub struct ConvRing {
    data: DeviceBuffer<f32>,
    channels: usize,
    taps: usize,
    rows: usize,
}

impl ConvRing {
    /// A ring for `channels` channels of `taps` taps run in passes of up
    /// to `rows` tokens.
    pub fn new(device: &Rc<Device>, channels: usize, taps: usize, rows: usize) -> Result<Self> {
        if taps < 2 {
            return Err(Error::Shape(format!("convolution of {taps} taps")));
        }
        Ok(ConvRing {
            data: DeviceBuffer::new(device, channels * (taps - 1 + rows))?,
            channels,
            taps,
            rows,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn taps(&self) -> usize {
        self.taps
    }

    pub fn slots(&self) -> usize {
        self.taps - 1 + self.rows
    }

    /// Every slot, `channels` values each.
    pub fn save(&self) -> Result<Vec<f32>> {
        self.data.to_vec()
    }

    /// The `taps - 1` positions before `pos` as the CPU keeps them, channel
    /// by channel: value `ch * (taps - 1) + i` is channel `ch` of position
    /// `pos - (taps - 1) + i`, zero before the start. Reads only those
    /// slots, which a ring for long passes holds few of.
    pub fn save_past(&self, pos: usize) -> Result<Vec<f32>> {
        let len = self.taps - 1;
        let mut past = vec![0.0; self.channels * len];
        let mut row = vec![0.0; self.channels];
        for i in 0..len {
            let Some(p) = (pos + i).checked_sub(len) else {
                continue;
            };
            self.data
                .copy_range_to(p % self.slots() * self.channels, &mut row)?;
            for (ch, &x) in row.iter().enumerate() {
                past[ch * len + i] = x;
            }
        }
        Ok(past)
    }

    /// Writes every slot, as [`ConvRing::save`] gives them.
    pub fn load(&mut self, slots: &[f32]) -> Result<()> {
        self.data.copy_from(slots)
    }

    pub(crate) fn data(&self) -> &DeviceBuffer<f32> {
        &self.data
    }

    /// Device bytes held.
    pub fn size_bytes(&self) -> usize {
        4 * self.data.len()
    }

    pub(crate) fn check(&self, batch: &Batch, channels: usize, taps: usize) -> Result<()> {
        if self.channels == channels && self.taps == taps && self.rows >= batch.rows() {
            Ok(())
        } else {
            Err(Error::Shape(format!(
                "ring of {} channels and {} taps for passes of {}",
                self.channels, self.taps, self.rows
            )))
        }
    }
}

/// Rows of `stride` values from element `at` on: a part of each token's
/// row of a buffer.
#[derive(Clone, Copy)]
pub struct Strided<'a> {
    pub data: &'a DeviceBuffer<f32>,
    pub stride: usize,
    pub at: usize,
}

impl Strided<'_> {
    fn holds(&self, rows: usize, len: usize) -> bool {
        (rows <= 1 || self.at + len <= self.stride)
            && (rows.max(1) - 1) * self.stride + self.at + len <= self.data.len()
    }
}

impl Kernels {
    /// Records the residual short convolution of every token of the batch,
    /// `(x + c) / sqrt(k + 1)`, where `c` is the causal convolution of each
    /// channel of `x` with its `taps` weights, oldest input first, and `k`
    /// the number of taps. With `dynamic`, the rows of `x.data` also hold
    /// `taps` more weights per head of `head_len` channels at element `at`,
    /// newest input first, that add to `c` and count in `k`. With
    /// `accumulate` the result is added to `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn residual_conv(
        &self,
        batch: &Batch,
        x: Strided,
        dynamic: Option<(usize, usize)>,
        weight: &DeviceBuffer<f32>,
        ring: &ConvRing,
        out: Strided,
        accumulate: bool,
        commands: &mut Commands,
    ) -> Result<()> {
        let (len, taps, rows) = (ring.channels, ring.taps, batch.rows());
        let (taps_at, head_len) = dynamic.unwrap_or((0, len));
        let lags = Strided { at: taps_at, ..x };
        let fits = head_len > 0
            && len.is_multiple_of(head_len)
            && weight.len() == len * taps
            && x.holds(rows, len)
            && (dynamic.is_none() || lags.holds(rows, len / head_len * taps))
            && out.holds(rows, len);
        ring.check(batch, len, taps)?;
        if !fits {
            return Err(Error::Shape(format!(
                "residual convolution of {rows} rows of {len}"
            )));
        }
        // SAFETY: matches residual_conv.wgsl: the batch, the input rows,
        // weights, ring and output at bindings 0..4, then its shape. Every
        // row the batch can name holds its parts, checked above, the ring
        // its slots; invocations past them return.
        unsafe {
            commands.dispatch(
                &self.residual_conv,
                &[
                    batch.device().allocation(),
                    x.data.allocation(),
                    weight.allocation(),
                    ring.data.allocation(),
                    out.data.allocation(),
                ],
                &[
                    len as u32,
                    taps as u32,
                    ring.slots() as u32,
                    x.stride as u32,
                    x.at as u32,
                    taps_at as u32,
                    dynamic.is_some() as u32,
                    head_len as u32,
                    out.stride as u32,
                    out.at as u32,
                    accumulate as u32,
                ],
                [len.div_ceil(256) as u32, rows as u32, 1],
            )
        };
        Ok(())
    }

    /// Records the gated short convolution of every token of the batch:
    /// a row of `bcx` holds the input gates, output gates and values,
    /// `out.width()` each; `weight` holds `taps` weights per channel, the
    /// oldest input's first.
    pub fn short_conv(
        &self,
        batch: &Batch,
        bcx: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        ring: &ConvRing,
        out: &GemvInput,
        commands: &mut Commands,
    ) -> Result<()> {
        let (len, rows) = (out.width(), batch.rows());
        ring.check(batch, len, ring.taps)?;
        if bcx.len() < 3 * rows * len || weight.len() != len * ring.taps || out.rows() < rows {
            return Err(Error::Shape(format!(
                "short convolution of {rows} rows of {len}"
            )));
        }
        let [ox, oq, od] = out.outputs();
        // SAFETY: matches short_conv.wgsl: the batch, bcx, weights, ring and
        // the three output buffers at bindings 0..6, then length, taps and
        // slots. Every buffer holds the batch's rows and the ring its
        // slots, checked above; invocations past them return.
        unsafe {
            commands.dispatch(
                &self.short_conv,
                &[
                    batch.device().allocation(),
                    bcx.allocation(),
                    weight.allocation(),
                    ring.data.allocation(),
                    ox,
                    oq,
                    od,
                ],
                &[len as u32, ring.taps as u32, ring.slots() as u32],
                [len.div_ceil(256) as u32, rows as u32, 1],
            )
        };
        Ok(())
    }
}
