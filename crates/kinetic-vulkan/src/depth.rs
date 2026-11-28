use crate::batch::Batch;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::gemv::GemvInput;
use crate::kernels::Kernels;
use crate::memory::DeviceBuffer;

/// Most states one mix over depth reads.
pub const MAX_STATES: usize = 8;

/// The states a mix over depth reads: `count` of them `stride` values apart
/// in `stack`, then, if given, `cur`.
pub struct States<'a> {
    pub stack: &'a DeviceBuffer<f32>,
    pub count: usize,
    pub stride: usize,
    pub cur: Option<&'a DeviceBuffer<f32>>,
}

impl Kernels {
    /// Records attention over depth for each token row of the batch into
    /// `out`: the states normalized by their root mean square, weighted by
    /// the softmax of their dot products with `query` over the square root
    /// of its length.
    pub fn depth_mix(
        &self,
        batch: &Batch,
        states: &States,
        query: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        commands: &mut Commands,
    ) -> Result<()> {
        let (rows, d) = (batch.rows(), query.len());
        let n = states.count + states.cur.is_some() as usize;
        let fits = d > 0
            && (1..=MAX_STATES).contains(&n)
            && (states.count < 2 || states.stride >= rows * d)
            && (states.count == 0
                || (states.count - 1) * states.stride + rows * d <= states.stack.len())
            && states.cur.is_none_or(|c| c.len() >= rows * d)
            && out.len() >= rows * d;
        if !fits {
            return Err(Error::Shape(format!(
                "mix over {n} states of {rows} rows of {d}"
            )));
        }
        let cur = states.cur.unwrap_or(states.stack);
        // SAFETY: matches depth.wgsl: the batch, stack, current state, query
        // and output at bindings 0..4, then its shape. Every state and the
        // output hold the batch's rows, checked above, and workgroups past
        // the pass's tokens return early.
        unsafe {
            commands.dispatch(
                &self.depth,
                &[
                    batch.device().allocation(),
                    states.stack.allocation(),
                    cur.allocation(),
                    query.allocation(),
                    out.allocation(),
                ],
                &[
                    d as u32,
                    states.count as u32,
                    states.stride as u32,
                    states.cur.is_some() as u32,
                ],
                [rows as u32, 1, 1],
            )
        };
        Ok(())
    }

    /// Records, for each token row of the batch, the row of `x` and then
    /// that of `m`, each normalized by its root mean square, as one row of
    /// `out`, twice as wide.
    pub fn join_norm(
        &self,
        batch: &Batch,
        x: &DeviceBuffer<f32>,
        m: &DeviceBuffer<f32>,
        eps: f32,
        out: &GemvInput,
        commands: &mut Commands,
    ) -> Result<()> {
        let (d, rows) = (out.width() / 2, batch.rows());
        if !out.width().is_multiple_of(64)
            || x.len() < rows * d
            || m.len() < rows * d
            || out.rows() < rows
        {
            return Err(Error::Shape(format!("joined norms of {rows} rows of {d}")));
        }
        let [ox, oq, od] = out.outputs();
        // SAFETY: matches join_norm.wgsl: the batch, x, m and the three
        // output buffers at bindings 0..5, then the width and epsilon; every
        // buffer holds the batch's rows, checked above.
        unsafe {
            commands.dispatch(
                &self.join_norm,
                &[
                    batch.device().allocation(),
                    x.allocation(),
                    m.allocation(),
                    ox,
                    oq,
                    od,
                ],
                &[d as u32, eps.to_bits()],
                [rows as u32, 1, 1],
            )
        };
        Ok(())
    }

    /// Records `x = (x - tanh(forget) x) + tanh(gain) m` for the first `d`
    /// values of each token row of the batch.
    #[allow(clippy::too_many_arguments)]
    pub fn control(
        &self,
        batch: &Batch,
        x: &DeviceBuffer<f32>,
        m: &DeviceBuffer<f32>,
        gain: &DeviceBuffer<f32>,
        forget: &DeviceBuffer<f32>,
        d: usize,
        commands: &mut Commands,
    ) -> Result<()> {
        let rows = batch.rows();
        if [x, m, gain, forget].iter().any(|b| b.len() < rows * d) {
            return Err(Error::Shape(format!("control of {rows} rows of {d}")));
        }
        // SAFETY: matches control.wgsl: the batch, x, m, gain and forget at
        // bindings 0..4, then the width; every buffer holds the batch's
        // rows, checked above.
        unsafe {
            commands.dispatch(
                &self.control,
                &[
                    batch.device().allocation(),
                    x.allocation(),
                    m.allocation(),
                    gain.allocation(),
                    forget.allocation(),
                ],
                &[d as u32],
                [d.div_ceil(256) as u32, rows as u32, 1],
            )
        };
        Ok(())
    }
}
