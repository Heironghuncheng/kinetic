use crate::batch::Batch;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::gemv::GemvInput;
use crate::kernels::Kernels;
use crate::memory::DeviceBuffer;

/// How a gated feed-forward combines its gate and up values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Glu {
    /// `silu(gate) * up`.
    Silu,
    /// `beta tanh(gate / beta) sigmoid(gate) * up`, with `up` first taken to
    /// `linear tanh(up / linear)` unless `linear` is zero.
    Situ { beta: f32, linear: f32 },
}

impl Kernels {
    /// Records `out = glu(gate, up)` for each row of the batch, where a
    /// token has `per_token` rows and a row of `gu` holds `out.width()` gate
    /// values and then as many up values.
    pub fn swiglu(
        &self,
        batch: &Batch,
        per_token: usize,
        glu: Glu,
        gu: &DeviceBuffer<f32>,
        out: &GemvInput,
        commands: &mut Commands,
    ) -> Result<()> {
        let (len, rows) = (out.width(), batch.rows() * per_token);
        let (beta, linear) = match glu {
            Glu::Silu => (0.0, 0.0),
            Glu::Situ { beta, linear } => (beta, linear),
        };
        if !(beta >= 0.0 && linear >= 0.0) || gu.len() < 2 * rows * len || out.rows() < rows {
            return Err(Error::Shape(format!(
                "SwiGLU of {rows} rows of {len} from {} values",
                gu.len()
            )));
        }
        let [ox, oq, od] = out.outputs();
        // SAFETY: matches swiglu.wgsl: the batch, gu and the three output
        // buffers at bindings 0..4, then the length, rows per token, beta and
        // linear beta; the buffers hold the batch's rows, checked above, and
        // invocations past them return.
        unsafe {
            commands.dispatch(
                &self.swiglu,
                &[batch.device().allocation(), gu.allocation(), ox, oq, od],
                &[
                    len as u32,
                    per_token as u32,
                    beta.to_bits(),
                    linear.to_bits(),
                ],
                [len.div_ceil(256) as u32, rows as u32, 1],
            )
        };
        Ok(())
    }
}
