//! A network of layers without one residual stream (Inkling 2) on the
//! device, recorded in the order [`Network::steps`] gives, which the host
//! path follows too. The number of blocks each mix reads is known at every
//! step, so a pass records once like any other.

use kinetic_vulkan::{
    Commands, DeviceBuffer, DeviceMatrix, GemvInput, HostBuffer, Out, Rows, States,
};

use super::layer::{LayerState, VulkanLayer, R};
use super::relative::RelativeState;
use super::work::Work;
use super::Gpu;
use crate::arch::inkling2::Network;
use crate::arch::inkling2_steps::Step;
use crate::blocks::Control;
use crate::LoadError;

/// The epsilon of the norms of a control's inputs, fixed by the reference.
const CONTROL_EPS: f32 = 1e-6;

/// A low-rank control on the device; see [`Control`].
struct VulkanControl {
    down: DeviceMatrix,
    gain: DeviceMatrix,
    forget: DeviceMatrix,
}

/// Where a physical layer reads its inputs from; see `Wiring`.
struct VulkanWiring {
    /// One query for the input of a windowed layer, one each for the query,
    /// key and value inputs of a full one; none for the first layer.
    attn_queries: Vec<DeviceBuffer<f32>>,
    ffn_query: DeviceBuffer<f32>,
    /// Queries of the messages to the query, key, value and feed-forward
    /// inputs of a layer of the core, and their controls.
    looped: Option<([DeviceBuffer<f32>; 4], [VulkanControl; 4])>,
}

pub(super) struct VulkanNetwork {
    steps: Vec<Step>,
    starts_block: Vec<bool>,
    embd_norm: DeviceBuffer<f32>,
    out_query: DeviceBuffer<f32>,
    eps: f32,
    wiring: Vec<VulkanWiring>,
    /// Most blocks a mix reads besides the stream, most a bank holds, and
    /// the rank of the controls.
    pub most: (usize, usize, usize),
}

/// The buffers of a network's pass, besides those of [`Work`].
pub(super) struct NetWork {
    /// The blocks the mixes read, rows of the pass each.
    stack: DeviceBuffer<f32>,
    /// The stream before the core, and the bank of a later pass.
    prelude: DeviceBuffer<f32>,
    bank: DeviceBuffer<f32>,
    /// The query, key and value inputs, then the feed-forward input.
    ins: [DeviceBuffer<f32>; 4],
    msgs: [DeviceBuffer<f32>; 4],
    /// The key and value inputs after their norms.
    normed: [GemvInput; 2],
    joined: GemvInput,
    code: GemvInput,
    gain: DeviceBuffer<f32>,
    forget: DeviceBuffer<f32>,
}

impl NetWork {
    pub fn new(
        gpu: &Gpu,
        net: &VulkanNetwork,
        d: usize,
        rows: usize,
    ) -> kinetic_vulkan::Result<Self> {
        let dev = &gpu.device;
        let (blocks, bank, rank) = net.most;
        let f32s = |n: usize| DeviceBuffer::<f32>::new(dev, (n * rows * d).max(1));
        Ok(NetWork {
            stack: f32s(blocks)?,
            prelude: f32s(1)?,
            bank: f32s(bank)?,
            ins: [f32s(1)?, f32s(1)?, f32s(1)?, f32s(1)?],
            msgs: [f32s(1)?, f32s(1)?, f32s(1)?, f32s(1)?],
            normed: [
                GemvInput::batch(dev, d, rows)?,
                GemvInput::batch(dev, d, rows)?,
            ],
            joined: GemvInput::batch(dev, 2 * d, rows)?,
            code: GemvInput::batch(dev, rank.max(32), rows)?,
            gain: f32s(1)?,
            forget: f32s(1)?,
        })
    }

    pub fn size_bytes(&self) -> usize {
        let floats: usize = [
            &self.stack,
            &self.prelude,
            &self.bank,
            &self.gain,
            &self.forget,
        ]
        .into_iter()
        .chain(&self.ins)
        .chain(&self.msgs)
        .map(DeviceBuffer::len)
        .sum();
        let inputs: usize = self
            .normed
            .iter()
            .chain([&self.joined, &self.code])
            .map(|i| i.len() * 21 / 4)
            .sum();
        4 * floats + inputs
    }
}

/// `n` queries of `d` values from `table`, each on the device.
fn queries<const N: usize>(
    gpu: &Gpu,
    table: &[f32],
    d: usize,
) -> Result<[DeviceBuffer<f32>; N], LoadError> {
    let mut out = Vec::with_capacity(N);
    for q in table.chunks_exact(d).take(N) {
        out.push(gpu.vector(q)?);
    }
    out.try_into()
        .map_err(|_| LoadError("a query table is too short".into()))
}

impl VulkanNetwork {
    pub fn upload(gpu: &Gpu, n: &Network, d: usize) -> Result<Self, LoadError> {
        let control = |c: &Control| -> Result<VulkanControl, LoadError> {
            Ok(VulkanControl {
                down: gpu.matrix(&c.down)?,
                gain: gpu.matrix(&c.gain)?,
                forget: gpu.matrix(&c.forget)?,
            })
        };
        let mut rank = 0;
        let wiring = n
            .wiring
            .iter()
            .map(|w| {
                let attn_queries = match &w.attn_query {
                    Some(q) => q
                        .chunks_exact(d)
                        .map(|q| gpu.vector(q))
                        .collect::<Result<_, _>>()?,
                    None => Vec::new(),
                };
                let looped = match &w.looped {
                    Some((q, c)) => {
                        rank = rank.max(c[0].down.rows);
                        let controls = [
                            control(&c[0])?,
                            control(&c[1])?,
                            control(&c[2])?,
                            control(&c[3])?,
                        ];
                        Some((queries::<4>(gpu, q, d)?, controls))
                    }
                    None => None,
                };
                Ok(VulkanWiring {
                    attn_queries,
                    ffn_query: gpu.vector(&w.ffn_query)?,
                    looped,
                })
            })
            .collect::<Result<Vec<_>, LoadError>>()?;
        let (blocks, bank) = n.most_blocks();
        if blocks + 1 > kinetic_vulkan::MAX_STATES || bank > kinetic_vulkan::MAX_STATES {
            return Err(LoadError(format!(
                "networks of {blocks} blocks do not run on Vulkan yet"
            )));
        }
        if !rank.is_multiple_of(32) {
            return Err(LoadError(format!(
                "loop controls of rank {rank} do not run on Vulkan, which needs a multiple of 32"
            )));
        }
        Ok(VulkanNetwork {
            steps: n.steps(),
            starts_block: (0..n.wiring.len()).map(|i| n.starts_block(i)).collect(),
            embd_norm: gpu.vector(&n.embd_norm)?,
            out_query: gpu.vector(&n.out_query)?,
            eps: n.eps,
            wiring,
            most: (blocks, bank, rank),
        })
    }

    /// The physical layer each state slot belongs to.
    pub fn slots(&self) -> Vec<usize> {
        let mut slots = Vec::new();
        for step in &self.steps {
            if let Step::Layer { layer, slot, .. } = *step {
                if slots.len() <= slot {
                    slots.resize(slot + 1, 0);
                }
                slots[slot] = layer;
            }
        }
        slots
    }

    /// Layers a pass runs.
    pub fn layer_steps(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| matches!(s, Step::Layer { .. }))
            .count()
    }

    /// Records the network over the embeddings in `w.x` and leaves the
    /// input of the head in `w.h`, normalized by `output_norm`. With
    /// `trace`, the stream after each layer is copied there, rows of the
    /// pass each.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        gpu: &Gpu,
        layers: &[VulkanLayer],
        w: &Work,
        states: &[LayerState],
        output_norm: &DeviceBuffer<f32>,
        trace: Option<&HostBuffer<f32>>,
        c: &mut Commands,
    ) -> R {
        let k = &gpu.kernels;
        let net = w.net.as_ref().expect("work has network buffers");
        let (d, rows) = (w.d, w.batch.rows());
        let len = rows * d;
        let cur = &w.x;
        k.norm(&w.batch, cur, None, &self.embd_norm, self.eps, &w.h, c)?;
        c.copy_buffer(w.h.values(), 0, cur, 0, len)?;
        let mut blocks = 0;
        let (mut kept, mut banked, mut traced) = (0, 0, 0);
        for step in &self.steps {
            match *step {
                Step::Prelude => {
                    kept = blocks;
                    c.copy_buffer(cur, 0, &net.prelude, 0, len)?;
                }
                Step::Pass => {
                    banked = blocks - kept;
                    let later = (blocks - kept - 1) * len;
                    if later > 0 {
                        c.copy_buffer(&net.stack, (kept + 1) * len, &net.bank, 0, later)?;
                    }
                    c.copy_buffer(cur, 0, &net.bank, later, len)?;
                    c.copy_buffer(&net.prelude, 0, cur, 0, len)?;
                    blocks = kept;
                }
                Step::Layer {
                    layer,
                    slot,
                    looped,
                } => {
                    let LayerState::Relative(state) = &states[slot] else {
                        unreachable!("network layers attend by relative distance")
                    };
                    let bank = States {
                        stack: &net.bank,
                        count: banked,
                        stride: len,
                        cur: None,
                    };
                    let ctx = Ctx {
                        gpu,
                        w,
                        net,
                        layer: &layers[layer],
                        wiring: &self.wiring[layer],
                        state,
                        bank: looped.then_some(&bank),
                    };
                    blocks = ctx.record(blocks, self.starts_block[layer], self.eps, c)?;
                    if let Some(trace) = trace {
                        c.download(cur, 0, trace, traced * len, len)?;
                        traced += 1;
                    }
                }
            }
        }
        let all = States {
            stack: &net.stack,
            count: blocks,
            stride: len,
            cur: Some(cur),
        };
        k.depth_mix(&w.batch, &all, &self.out_query, &net.ins[3], c)?;
        k.norm(&w.batch, &net.ins[3], None, output_norm, self.eps, &w.h, c)
    }
}

/// What one layer of a pass works with.
struct Ctx<'a> {
    gpu: &'a Gpu,
    w: &'a Work,
    net: &'a NetWork,
    layer: &'a VulkanLayer,
    wiring: &'a VulkanWiring,
    state: &'a RelativeState,
    /// The previous pass's blocks and output, in a later pass of the core.
    bank: Option<&'a States<'a>>,
}

impl Ctx<'_> {
    /// Records the layer after `blocks` blocks; returns how many there are
    /// after it.
    fn record(
        &self,
        mut blocks: usize,
        starts_block: bool,
        eps: f32,
        c: &mut Commands,
    ) -> kinetic_vulkan::Result<usize> {
        let (gpu, w, net) = (self.gpu, self.w, self.net);
        let k = &gpu.kernels;
        let attn = self
            .layer
            .relative()
            .expect("network layers attend by relative distance");
        let len = w.batch.rows() * w.d;
        let cur = &w.x;
        let stack = |count| States {
            stack: &net.stack,
            count,
            stride: len,
            cur: Some(cur),
        };
        let looped = match (self.bank, &self.wiring.looped) {
            (Some(bank), Some((queries, controls))) => {
                for (q, m) in queries.iter().zip(&net.msgs) {
                    k.depth_mix(&w.batch, bank, q, m, c)?;
                }
                Some(controls)
            }
            _ => None,
        };

        // The query, key and value inputs: the stream itself in the first
        // layer, else mixes of it and the blocks; one input serves all three
        // in a windowed layer unless controls make them differ.
        let queries = &self.wiring.attn_queries;
        for (q, x) in queries.iter().zip(&net.ins) {
            k.depth_mix(&w.batch, &stack(blocks), q, x, c)?;
        }
        let separate = looped.is_some() || attn.shape.heads.window.is_none();
        let inputs = if separate { 3 } else { 1 };
        let source = |j: usize| match queries.len() {
            0 => cur,
            1 => &net.ins[0],
            _ => &net.ins[j],
        };
        if let Some(controls) = looped {
            if queries.len() == 1 {
                for j in 1..3 {
                    c.copy_buffer(&net.ins[0], 0, &net.ins[j], 0, len)?;
                }
            }
            for ((control, x), m) in controls.iter().zip(&net.ins).zip(&net.msgs).take(3) {
                self.control(control, x, m, c)?;
            }
        }
        let normed = [&w.h, &net.normed[0], &net.normed[1]];
        for (j, (norm, out)) in attn.norms.iter().zip(normed).enumerate().take(inputs) {
            let x = if looped.is_some() {
                &net.ins[j]
            } else {
                source(j)
            };
            k.norm(&w.batch, x, None, norm, attn.eps(), out, c)?;
        }
        let ins = [0, 1, 2].map(|j| normed[j.min(inputs - 1)]);
        attn.record(gpu, w, ins, self.state, c)?;

        // The first layer of a block starts from the attention output alone
        // and hands its input to the blocks later layers mix.
        if starts_block {
            c.copy_buffer(cur, 0, &net.stack, blocks * len, len)?;
            blocks += 1;
        }
        attn.record_conv(gpu, w, 2, self.state, cur, !starts_block, c)?;

        let h = &net.ins[3];
        k.depth_mix(&w.batch, &stack(blocks), &self.wiring.ffn_query, h, c)?;
        if let Some(controls) = looped {
            self.control(&controls[3], h, &net.msgs[3], c)?;
        }
        k.norm(&w.batch, h, None, self.layer.ffn_norm(), eps, &w.h, c)?;
        let moe = self.layer.experts().expect("network layers have experts");
        moe.record(gpu, w, false, c)?;
        attn.record_conv(gpu, w, 3, self.state, cur, true, c)?;
        Ok(blocks)
    }

    /// Records `control` applied to `x` with the message `m`.
    fn control(
        &self,
        control: &VulkanControl,
        x: &DeviceBuffer<f32>,
        m: &DeviceBuffer<f32>,
        c: &mut Commands,
    ) -> R {
        let (k, w, net) = (&self.gpu.kernels, self.w, self.net);
        k.join_norm(&w.batch, x, m, CONTROL_EPS, &net.joined, c)?;
        let out = |stride| Out { stride, offset: 0 };
        let code = net.code.values();
        k.gemv_rows(
            &control.down,
            &net.joined,
            code,
            out(net.code.width()),
            Rows::Tokens,
            &w.batch,
            c,
        )?;
        net.code.record_quantize(k, c)?;
        c.independent(|c| {
            k.gemv_rows(
                &control.gain,
                &net.code,
                &net.gain,
                out(w.d),
                Rows::Tokens,
                &w.batch,
                c,
            )?;
            k.gemv_rows(
                &control.forget,
                &net.code,
                &net.forget,
                out(w.d),
                Rows::Tokens,
                &w.batch,
                c,
            )
        })?;
        k.control(&w.batch, x, m, &net.gain, &net.forget, w.d, c)
    }
}
