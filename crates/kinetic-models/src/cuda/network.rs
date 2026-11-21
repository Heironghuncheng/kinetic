//! Inkling 2's network on the device: the steps [`Network::steps`] gives,
//! each the device form of what `arch/inkling2.rs` does on the host. The
//! stream, the blocks the mixes over depth read and the bank of a looped
//! pass are buffers here; layer slots are the model's layers, each with its
//! own state.

use std::rc::Rc;

use kinetic_cuda::{DeviceBuffer, DeviceMatrix, GemvInput, RmsNorm, DEPTH_MAX_STATES};

use super::experts::{CudaExperts, ExpertWork, Offload};
use super::layer::{CudaLayer, CudaMixer, LayerState};
use super::relative::{CudaRelative, RelBuffers, RelLayer};
use super::scratch::Work;
use super::{cuda_error, Gpu};
use crate::arch::inkling2::Network;
use crate::arch::inkling2_steps::Step;
use crate::blocks::{Control, Mixer};
use crate::{LoadError, Model};

/// Epsilon of the norms inside the mixes and controls, as `blocks/depth.rs`
/// fixes it.
const DEPTH_EPS: f32 = 1e-6;

pub(super) struct CudaNetwork {
    d: usize,
    eps: f32,
    steps: Vec<Step>,
    /// Per physical layer, whether it starts a block.
    starts: Vec<bool>,
    most_blocks: (usize, usize),
    embd_norm: DeviceBuffer<f32>,
    out_query: DeviceBuffer<f32>,
    /// The norms inside the controls take no weights.
    ones: DeviceBuffer<f32>,
    wiring: Vec<Wiring>,
    /// Per physical layer.
    ffn_norms: Vec<DeviceBuffer<f32>>,
    experts: Vec<CudaExperts>,
    rank: usize,
}

struct Wiring {
    attn_query: Option<DeviceBuffer<f32>>,
    ffn_query: DeviceBuffer<f32>,
    looped: Option<(DeviceBuffer<f32>, [CudaControl; 4])>,
}

struct CudaControl {
    down: DeviceMatrix,
    gain: DeviceMatrix,
    forget: DeviceMatrix,
}

/// A state's buffers for running the network on batches of up to `rows`.
pub(super) struct NetBuffers {
    cur: DeviceBuffer<f32>,
    prelude: DeviceBuffer<f32>,
    ffn_in: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    blocks: Vec<DeviceBuffer<f32>>,
    bank: Vec<DeviceBuffer<f32>>,
    inputs: [DeviceBuffer<f32>; 3],
    messages: [DeviceBuffer<f32>; 4],
    normed: GemvInput,
    control: ControlBuffers,
    rel: RelBuffers,
}

struct ControlBuffers {
    joined: GemvInput,
    code: DeviceBuffer<f32>,
    code_in: GemvInput,
    gain: DeviceBuffer<f32>,
    forget: DeviceBuffer<f32>,
}

/// Where a layer's run stands among the buffers: blocks kept, and the
/// bank's length and the blocks before the core, in a looped pass.
#[derive(Default)]
struct Depth {
    blocks: usize,
    prelude: usize,
    banked: usize,
}

/// Uploads the layers of `model`, whose layers run in `net`: one slot per
/// layer of the model, those of later passes sharing the weights of their
/// physical layer.
pub(super) fn upload(
    gpu: &Gpu,
    model: &Model,
    net: &Network,
) -> Result<(Vec<CudaLayer>, CudaNetwork), LoadError> {
    let n_phys = net.wiring.len();
    let most_blocks = net.most_blocks();
    if most_blocks.0 + 1 > DEPTH_MAX_STATES || most_blocks.1 > DEPTH_MAX_STATES {
        return Err(LoadError(format!(
            "Inkling 2 mixes over {} blocks, more than CUDA's {DEPTH_MAX_STATES}",
            most_blocks.0 + 1
        )));
    }
    let v = |x: &[f32]| gpu.vector(x).map_err(cuda_error);
    let m = |x: &kinetic_quant::Matrix| gpu.matrix(x).map_err(cuda_error);
    let physical = &model.layers[..n_phys];
    let weights = physical
        .iter()
        .map(|l| match &l.mixer {
            Mixer::Relative(a) => RelLayer::upload(gpu, a).map(Rc::new),
            _ => unreachable!("Inkling 2 layers attend by relative distance"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if weights.iter().any(|w| w.heads() != weights[0].heads()) {
        return Err(LoadError(
            "Inkling 2 with head counts that vary by layer does not run on CUDA yet".into(),
        ));
    }
    let layers = (0..model.layers.len())
        .map(|slot| {
            let (layer, shared) = match net.steps().iter().find_map(|s| match *s {
                Step::Layer { layer, slot: x, .. } if x == slot => Some(layer),
                _ => None,
            }) {
                Some(l) => (l, slot >= n_phys),
                None => unreachable!("every slot runs"),
            };
            let mixer = CudaMixer::Relative(CudaRelative {
                weights: Rc::clone(&weights[layer]),
                shared,
            });
            CudaLayer::in_network(mixer, model.eps)
        })
        .collect();
    let experts = physical
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let moe = l.experts.as_ref().expect("Inkling 2 layers have experts");
            let next = physical.get(i + 1).and_then(|n| n.experts.as_ref());
            CudaExperts::upload(gpu, moe, next.map(|n| &n.router), false)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let control = |c: &Control| -> Result<CudaControl, LoadError> {
        Ok(CudaControl {
            down: m(&c.down)?,
            gain: m(&c.gain)?,
            forget: m(&c.forget)?,
        })
    };
    let wiring = net
        .wiring
        .iter()
        .map(|w| -> Result<Wiring, LoadError> {
            Ok(Wiring {
                attn_query: w.attn_query.as_deref().map(v).transpose()?,
                ffn_query: v(&w.ffn_query)?,
                looped: match &w.looped {
                    Some((q, [a, b, c, d])) => {
                        Some((v(q)?, [control(a)?, control(b)?, control(c)?, control(d)?]))
                    }
                    None => None,
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rank = net
        .wiring
        .iter()
        .find_map(|w| w.looped.as_ref())
        .map_or(0, |(_, c)| c[0].down.rows);
    let d = model.n_embd;
    let network = CudaNetwork {
        d,
        eps: model.eps,
        steps: net.steps(),
        starts: (0..n_phys).map(|i| net.starts_block(i)).collect(),
        most_blocks,
        embd_norm: v(&net.embd_norm)?,
        out_query: v(&net.out_query)?,
        ones: v(&vec![1.0; d])?,
        wiring,
        ffn_norms: physical
            .iter()
            .map(|l| v(l.ffn_norm.as_deref().expect("Inkling 2 layers norm")))
            .collect::<Result<_, _>>()?,
        experts,
        rank,
    };
    Ok((layers, network))
}

impl CudaNetwork {
    pub fn size_bytes(&self) -> usize {
        let wiring: usize = self
            .wiring
            .iter()
            .map(|w| {
                let looped = w.looped.as_ref().map_or(0, |(q, cs)| {
                    4 * q.len()
                        + cs.iter()
                            .map(|c| {
                                c.down.size_bytes() + c.gain.size_bytes() + c.forget.size_bytes()
                            })
                            .sum::<usize>()
                });
                4 * (w.attn_query.as_ref().map_or(0, DeviceBuffer::len) + w.ffn_query.len())
                    + looped
            })
            .sum();
        let experts: usize = self.experts.iter().map(CudaExperts::size_bytes).sum();
        wiring + experts + 4 * self.d * (3 + self.ffn_norms.len())
    }

    pub fn experts(&self) -> &[CudaExperts] {
        &self.experts
    }

    pub fn buffers(
        &self,
        gpu: &Gpu,
        layers: &[CudaLayer],
        rows: usize,
    ) -> kinetic_cuda::Result<NetBuffers> {
        let d = self.d;
        let f32s = |len: usize| DeviceBuffer::<f32>::new(&gpu.ctx, (rows * len).max(1));
        let rows_of = || f32s(d);
        let many = |n: usize| {
            (0..n)
                .map(|_| rows_of())
                .collect::<kinetic_cuda::Result<Vec<_>>>()
        };
        let rank = self.rank.max(32);
        Ok(NetBuffers {
            cur: rows_of()?,
            prelude: rows_of()?,
            ffn_in: rows_of()?,
            h: rows_of()?,
            y: rows_of()?,
            attn: rows_of()?,
            blocks: many(self.most_blocks.0)?,
            bank: many(self.most_blocks.1)?,
            inputs: [rows_of()?, rows_of()?, rows_of()?],
            messages: [rows_of()?, rows_of()?, rows_of()?, rows_of()?],
            normed: GemvInput::batch(&gpu.ctx, d, rows)?,
            control: ControlBuffers {
                joined: GemvInput::batch(&gpu.ctx, 2 * d, rows)?,
                code: f32s(rank)?,
                code_in: GemvInput::batch(&gpu.ctx, rank, rows)?,
                gain: rows_of()?,
                forget: rows_of()?,
            },
            rel: relative(&layers[0]).buffers(gpu, rows)?,
        })
    }

    /// Queues the network over the batch of `work`, whose embeddings are in
    /// its scratch, and leaves the output normalized by `output_norm` in
    /// the scratch for the head.
    pub fn queue(
        &self,
        gpu: &Gpu,
        layers: &[CudaLayer],
        work: &mut Work,
        offload: &mut Offload,
        output_norm: &DeviceBuffer<f32>,
    ) -> kinetic_cuda::Result<()> {
        let (k, stream) = (&gpu.kernels, &gpu.stream);
        let (n, d) = (work.batch.len(), self.d);
        let len = n * d;
        let b = work.net.as_mut().expect("a state of a network model");
        let experts = work.experts.as_mut().expect("a state of an expert model");
        let norm = RmsNorm {
            weight: &self.embd_norm,
            eps: self.eps,
        };
        k.rms_norm(&work.scratch.x, norm, &mut b.normed, 0, n, stream)?;
        b.cur
            .copy_from_device(0, b.normed.values(), 0..len, stream)?;
        let mut depth = Depth::default();
        for &step in &self.steps {
            match step {
                Step::Prelude => {
                    b.prelude.copy_from_device(0, &b.cur, 0..len, stream)?;
                    depth.prelude = depth.blocks;
                }
                Step::Pass => {
                    depth.banked = 0;
                    let kept = depth.prelude + 1..depth.blocks;
                    for (i, from) in kept.enumerate() {
                        b.bank[i].copy_from_device(0, &b.blocks[from], 0..len, stream)?;
                        depth.banked += 1;
                    }
                    b.bank[depth.banked].copy_from_device(0, &b.cur, 0..len, stream)?;
                    depth.banked += 1;
                    b.cur.copy_from_device(0, &b.prelude, 0..len, stream)?;
                    depth.blocks = depth.prelude;
                }
                Step::Layer {
                    layer,
                    slot,
                    looped,
                } => {
                    let LayerState::Relative(state) = &mut work.layers[slot] else {
                        unreachable!("network slots attend by relative distance")
                    };
                    let w = relative(&layers[slot]);
                    let run = LayerRun { layer, looped, n };
                    self.layer(gpu, &run, w, state, b, &mut depth, &work.batch)?;
                    self.experts[layer].score(gpu, &b.normed, n, &mut experts.gates)?;
                    self.feed(gpu, &run, w, state, b, experts, offload, &work.batch)?;
                }
            }
        }
        let mut states: Vec<&DeviceBuffer<f32>> = b.blocks[..depth.blocks].iter().collect();
        states.push(&b.cur);
        k.depth_mix(&states, &self.out_query, &mut [&mut b.h], d, n, stream)?;
        let norm = RmsNorm {
            weight: output_norm,
            eps: self.eps,
        };
        k.rms_norm(&b.h, norm, &mut work.scratch.h, 0, n, stream)
    }

    /// The front of a layer as `Network::layer` runs it: its attention
    /// inputs mixed from the blocks, controlled in a looped pass, the
    /// attention, the feed-forward input, and that input mixed, controlled
    /// and normalized into `b.normed`.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        gpu: &Gpu,
        run: &LayerRun,
        w: &RelLayer,
        state: &mut super::relative::RelState,
        b: &mut NetBuffers,
        depth: &mut Depth,
        batch: &kinetic_cuda::Batch,
    ) -> kinetic_cuda::Result<()> {
        let (k, stream) = (&gpu.kernels, &gpu.stream);
        let (n, d) = (run.n, self.d);
        let len = n * d;
        let wiring = &self.wiring[run.layer];
        let controls = wiring.looped.as_ref().filter(|_| run.looped);
        if let Some((query, _)) = controls {
            let bank: Vec<&DeviceBuffer<f32>> = b.bank[..depth.banked].iter().collect();
            let mut outs: Vec<&mut DeviceBuffer<f32>> = b.messages.iter_mut().collect();
            k.depth_mix(&bank, query, &mut outs, d, n, stream)?;
        }
        let NetBuffers {
            cur,
            blocks,
            inputs,
            messages,
            control,
            rel,
            attn,
            ffn_in,
            h,
            normed,
            ..
        } = b;
        let inputs: [&DeviceBuffer<f32>; 3] = match &wiring.attn_query {
            None => [&*cur; 3],
            Some(query) => {
                let three = controls.is_some() || query.len() == 3 * d;
                let mut states: Vec<&DeviceBuffer<f32>> = blocks[..depth.blocks].iter().collect();
                states.push(cur);
                let mut outs: Vec<&mut DeviceBuffer<f32>> =
                    inputs.iter_mut().take(if three { 3 } else { 1 }).collect();
                k.depth_mix(&states, query, &mut outs, d, n, stream)?;
                if let Some((_, cs)) = controls {
                    for (j, x) in outs.into_iter().enumerate() {
                        self.control(gpu, &cs[j], x, &messages[j], control, n)?;
                    }
                }
                match three {
                    true => [&inputs[0], &inputs[1], &inputs[2]],
                    false => [&inputs[0]; 3],
                }
            }
        };
        w.forward(gpu, rel, inputs, state, batch, attn)?;

        // The first layer of a block starts from the attention output alone
        // and hands its input to the blocks later layers mix.
        if self.starts[run.layer] {
            blocks[depth.blocks].copy_from_device(0, cur, 0..len, stream)?;
            depth.blocks += 1;
            ffn_in.copy_from_device(0, attn, 0..len, stream)?;
        } else {
            ffn_in.copy_from_device(0, cur, 0..len, stream)?;
            k.add(ffn_in, attn, len, stream)?;
        }
        let mut states: Vec<&DeviceBuffer<f32>> = blocks[..depth.blocks].iter().collect();
        states.push(ffn_in);
        k.depth_mix(&states, &wiring.ffn_query, &mut [&mut *h], d, n, stream)?;
        if let Some((_, cs)) = controls {
            self.control(gpu, &cs[3], h, &messages[3], control, n)?;
        }
        let norm = RmsNorm {
            weight: &self.ffn_norms[run.layer],
            eps: self.eps,
        };
        k.rms_norm(h, norm, normed, 0, n, stream)
    }

    /// The back of a layer: its routed experts on `b.normed`, scored into
    /// `experts.gates`, their output's convolution, and the new stream.
    #[allow(clippy::too_many_arguments)]
    fn feed(
        &self,
        gpu: &Gpu,
        run: &LayerRun,
        w: &RelLayer,
        state: &mut super::relative::RelState,
        b: &mut NetBuffers,
        experts: &mut ExpertWork,
        offload: &mut Offload,
        batch: &kinetic_cuda::Batch,
    ) -> kinetic_cuda::Result<()> {
        let (k, stream) = (&gpu.kernels, &gpu.stream);
        let len = run.n * self.d;
        let e = &self.experts[run.layer];
        e.mix(
            run.layer,
            gpu,
            b.normed.values(),
            &mut b.y,
            experts,
            offload,
            run.n,
        )?;
        w.ffn_conv(gpu, &b.y, state, batch, &mut b.attn)?;
        b.cur.copy_from_device(0, &b.ffn_in, 0..len, stream)?;
        k.add(&mut b.cur, &b.attn, len, stream)
    }

    /// `Control::apply` on the device: `x` takes in `message` through the
    /// low-rank gate of `c`.
    fn control(
        &self,
        gpu: &Gpu,
        c: &CudaControl,
        x: &mut DeviceBuffer<f32>,
        message: &DeviceBuffer<f32>,
        b: &mut ControlBuffers,
        n: usize,
    ) -> kinetic_cuda::Result<()> {
        let (k, stream) = (&gpu.kernels, &gpu.stream);
        let d = self.d;
        let norm = RmsNorm {
            weight: &self.ones,
            eps: DEPTH_EPS,
        };
        k.rms_norm(x, norm, &mut b.joined, 0, n, stream)?;
        k.rms_norm(message, norm, &mut b.joined, d, n, stream)?;
        k.gemv_rows(&c.down, c.down.rows(), &b.joined, 0..n, &mut b.code, stream)?;
        k.input_rows(&b.code, &mut b.code_in, n, stream)?;
        k.gemv_rows(&c.gain, d, &b.code_in, 0..n, &mut b.gain, stream)?;
        k.gemv_rows(&c.forget, d, &b.code_in, 0..n, &mut b.forget, stream)?;
        k.loop_control(x, message, &b.gain, &b.forget, n * d, stream)
    }
}

impl NetBuffers {
    /// Device bytes held.
    pub fn size_bytes(&self) -> usize {
        let c = &self.control;
        let f32s = [
            &self.cur,
            &self.prelude,
            &self.ffn_in,
            &self.h,
            &self.y,
            &self.attn,
            &c.code,
            &c.gain,
            &c.forget,
        ]
        .into_iter()
        .chain(&self.blocks)
        .chain(&self.bank)
        .chain(&self.inputs)
        .chain(&self.messages);
        let inputs = [&self.normed, &c.joined, &c.code_in];
        4 * f32s.map(DeviceBuffer::len).sum::<usize>()
            + inputs.iter().map(|i| gemv_bytes(i)).sum::<usize>()
            + self.rel.size_bytes()
    }
}

/// Device bytes of a GEMV input: f32 values, their int8 copy and scales.
pub(super) fn gemv_bytes(i: &GemvInput) -> usize {
    i.capacity() * i.width() * 6
}

struct LayerRun {
    layer: usize,
    looped: bool,
    n: usize,
}

fn relative(l: &CudaLayer) -> &RelLayer {
    match &l.mixer {
        CudaMixer::Relative(r) => &r.weights,
        _ => unreachable!("network layers attend by relative distance"),
    }
}
