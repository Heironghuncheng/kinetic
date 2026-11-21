use crate::batch::Batch;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::kernels::Kernels;
use crate::memory::DeviceBuffer;

/// How router logits become expert probabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gating {
    Softmax,
    /// A sigmoid per expert.
    Sigmoid,
}

/// What a mixture-of-experts layer's output starts from.
#[derive(Clone, Copy)]
pub enum Shared<'a> {
    /// Nothing: the layer has no shared expert.
    None,
    /// The shared expert's output, already in the output rows.
    Plain,
    /// The same, scaled by the sigmoid of each token's gate value.
    Gated(&'a DeviceBuffer<f32>),
}

/// Threads per workgroup of route.wgsl, one per token.
const ROUTE_THREADS: usize = 64;

impl Kernels {
    /// Records the routing of each token of the batch: its `experts` logits
    /// in `logits` become probabilities in place, and `picks` gets the
    /// `used` experts of the largest probability plus `bias`, largest first,
    /// each followed by the bits of its share of their probabilities; two
    /// words per pick.
    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &self,
        batch: &Batch,
        gating: Gating,
        logits: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        experts: usize,
        used: usize,
        picks: &DeviceBuffer<u32>,
        commands: &mut Commands,
    ) -> Result<()> {
        let rows = batch.rows();
        if used == 0
            || used > experts
            || bias.len() < experts
            || logits.len() < rows * experts
            || picks.len() < 2 * rows * used
        {
            return Err(Error::Shape(format!(
                "routing of {rows} rows to {used} of {experts} experts"
            )));
        }
        // SAFETY: matches route.wgsl: the batch, logits, bias and picks at
        // bindings 0..3, then the expert count, picks per token and the
        // gating; every buffer holds the batch's rows, checked above.
        unsafe {
            commands.dispatch(
                &self.route,
                &[
                    batch.device().allocation(),
                    logits.allocation(),
                    bias.allocation(),
                    picks.allocation(),
                ],
                &[
                    experts as u32,
                    used as u32,
                    (gating == Gating::Sigmoid) as u32,
                ],
                [rows.div_ceil(ROUTE_THREADS) as u32, 1, 1],
            )
        };
        Ok(())
    }

    /// Records the output rows `y` of a mixture-of-experts layer, `d` wide:
    /// what `shared` says, plus the `used` rows of `routed` of each token
    /// times their weights in `picks`.
    #[allow(clippy::too_many_arguments)]
    pub fn mix_experts(
        &self,
        batch: &Batch,
        used: usize,
        picks: &DeviceBuffer<u32>,
        routed: &DeviceBuffer<f32>,
        shared: Shared,
        d: usize,
        y: &DeviceBuffer<f32>,
        commands: &mut Commands,
    ) -> Result<()> {
        let rows = batch.rows();
        let (mode, gate) = match shared {
            Shared::None => (0, routed),
            Shared::Plain => (1, routed),
            Shared::Gated(g) => (2, g),
        };
        if picks.len() < 2 * rows * used
            || routed.len() < rows * used * d
            || y.len() < rows * d
            || (mode == 2 && gate.len() < rows)
        {
            return Err(Error::Shape(format!(
                "mixing {used} experts per row into {rows} rows of {d}"
            )));
        }
        // SAFETY: matches mix_experts.wgsl: the batch, picks, routed rows,
        // gates and output at bindings 0..4, then the width, picks per token
        // and the mode, which reads the gates only when they are given;
        // every buffer holds the batch's rows, checked above.
        unsafe {
            commands.dispatch(
                &self.mix_experts,
                &[
                    batch.device().allocation(),
                    picks.allocation(),
                    routed.allocation(),
                    gate.allocation(),
                    y.allocation(),
                ],
                &[d as u32, used as u32, mode],
                [d.div_ceil(256) as u32, rows as u32, 1],
            )
        };
        Ok(())
    }
}
