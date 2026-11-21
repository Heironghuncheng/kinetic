use crate::error::{Error, Result};
use crate::kernels::{arg, launch_struct, Kernels};
use crate::memory::DeviceBuffer;
use crate::module::LaunchConfig;
use crate::stream::Stream;

/// Most states one mix reads and outputs it writes; must match depth.cuh.
pub const DEPTH_MAX_STATES: usize = 8;
const MAX_OUTS: usize = 4;
const MIX_THREADS: u32 = 256;
const CONTROL_THREADS: u32 = 256;

#[repr(C)]
struct MixArgs {
    states: [u64; DEPTH_MAX_STATES],
    outs: [u64; MAX_OUTS],
    queries: u64,
    n_states: u32,
    n_queries: u32,
    n_outs: u32,
    d: u32,
}

impl Kernels {
    /// Queues, for each of `n` tokens and each of `outs`, the mix over
    /// depth (Inkling 2): the token's rows of `states`, `d` values each,
    /// normalized by their root mean square and weighted by the softmax of
    /// their dot products with a query of `queries` over the square root of
    /// `d`. Output `o` takes query `o`, or the last one when there are
    /// fewer.
    pub fn depth_mix(
        &self,
        states: &[&DeviceBuffer<f32>],
        queries: &DeviceBuffer<f32>,
        outs: &mut [&mut DeviceBuffer<f32>],
        d: usize,
        n: usize,
        stream: &Stream,
    ) -> Result<()> {
        let n_queries = queries.len() / d.max(1);
        if states.is_empty()
            || states.len() > DEPTH_MAX_STATES
            || outs.is_empty()
            || outs.len() > MAX_OUTS
            || n_queries == 0
            || queries.len() != n_queries * d
        {
            return Err(Error::Shape(format!(
                "mix of {} states by {n_queries} queries into {} outputs",
                states.len(),
                outs.len()
            )));
        }
        for s in states {
            s.check_holds(n * d, "mixed state")?;
        }
        let mut args = MixArgs {
            states: [0; DEPTH_MAX_STATES],
            outs: [0; MAX_OUTS],
            queries: queries.device_ptr(),
            n_states: states.len() as u32,
            n_queries: n_queries as u32,
            n_outs: outs.len() as u32,
            d: d as u32,
        };
        for (p, s) in args.states.iter_mut().zip(states) {
            *p = s.device_ptr();
        }
        for (p, o) in args.outs.iter_mut().zip(outs.iter()) {
            o.check_holds(n * d, "mix output")?;
            *p = o.device_ptr();
        }
        if n == 0 {
            return Ok(());
        }
        // SAFETY: matches `DepthMixArgs`; every state and output holds `n`
        // rows of `d` values and the queries `n_queries` of them, as checked.
        unsafe {
            launch_struct(
                &self.depth_mix,
                LaunchConfig::linear(n as u32, MIX_THREADS),
                stream,
                &args,
            )
        }
    }

    /// Queues `x = x - tanh(forget) x + tanh(gain) message` over the first
    /// `len` values, the gate of a looped layer's message.
    pub fn loop_control(
        &self,
        x: &mut DeviceBuffer<f32>,
        message: &DeviceBuffer<f32>,
        gain: &DeviceBuffer<f32>,
        forget: &DeviceBuffer<f32>,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        x.check_holds(len, "controlled input")?;
        message.check_holds(len, "message")?;
        gain.check_holds(len, "gain")?;
        forget.check_holds(len, "forget gate")?;
        let (xp, mp, gp, fp, n) = (
            x.device_ptr(),
            message.device_ptr(),
            gain.device_ptr(),
            forget.device_ptr(),
            len as u32,
        );
        let mut args = [arg(&xp), arg(&mp), arg(&gp), arg(&fp), arg(&n)];
        let config = LaunchConfig::linear(n.div_ceil(CONTROL_THREADS), CONTROL_THREADS);
        // SAFETY: matches `loop_control(float*, const float*, const float*,
        // const float*, unsigned)`; every buffer holds `len` values.
        unsafe { self.loop_control.launch(config, stream, &mut args) }
    }
}
