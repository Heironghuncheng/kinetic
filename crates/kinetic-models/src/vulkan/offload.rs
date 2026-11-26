//! Flywheel on Vulkan: where the routed experts of a model larger than the
//! device run.
//!
//! Everything else lives on the device. The layers whose experts run most
//! often per token keep every expert there (a layer the loop of its network
//! repeats runs once per pass), as many layers as fit beside the rest of
//! the model, one sequence's state and a reserve. The experts of the other
//! layers run on the CPU where they are, in the mapped file, and nothing is
//! copied to the device: a pass stops after such a layer's routing, the
//! CPU runs the picks on the layer's input, and the device mixes their
//! outputs. Experts of a type the device has no kernel for run there too.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::fmt;
use std::time::{Duration, Instant};

use kinetic_vulkan::{Commands, Graph, HostBuffer};

use super::forms::{on_vulkan, Form};
use super::layer::{VulkanLayer, R};
use super::moe::sources;
use super::work::Work;
use super::{Gpu, VulkanModel, VulkanState};
use crate::blocks::{Job, Moe};
use crate::offload::plan::DEVICE_RESERVE;
use crate::offload::stats::Stats;
use crate::{LoadError, Model};

/// Seconds spent per part of the passes, since load or the last reset.
#[derive(Clone, Copy, Debug, Default)]
pub struct Times {
    /// Running the layers up to each routing on the CPU's experts, and
    /// reading the picks back.
    pub layers: Duration,
    /// Running experts on the CPU, with their inputs and outputs.
    pub host: Duration,
}

pub(super) struct VulkanOffload<'m> {
    /// Per physical layer, if the CPU runs its experts.
    host: Vec<Option<Moe<'m>>>,
    /// Layers with routed experts, and the device bytes of those it holds.
    layers: usize,
    device_bytes: usize,
    stats: RefCell<Stats>,
    times: RefCell<Times>,
}

/// Device memory a plan leaves free on a device of `total` bytes. The
/// desktop and other programs take and return memory while a model runs,
/// and a driver out of dedicated memory spills to system memory, where the
/// model runs at a fraction of its speed.
fn reserve(total: usize) -> usize {
    DEVICE_RESERVE.max(total / 20)
}

/// Bytes of every expert of `m`.
fn bytes(m: &Moe) -> usize {
    let (gate_up, down) = sources(m);
    let one = gate_up.iter().map(|(e, _)| e.expert_bytes()).sum::<usize>() + down.expert_bytes();
    m.n_experts() * one
}

/// Whether the device has kernels for every expert matrix of `m`.
fn kernels(m: &Moe) -> bool {
    let (gate_up, down) = sources(m);
    let mut types = gate_up.iter().map(|(e, _)| e.ty).chain([down.ty]);
    types.all(|ty| on_vulkan(ty) == Form::Kernel)
}

/// Uploads the routed experts of `model`'s layers, of which a token runs
/// physical layer `l` `runs[l]` times: every expert if they fit in what
/// the device has left after `elsewhere` bytes, and in at most `cap`
/// bytes; else whole layers, those that run most first, and the offload
/// that runs the rest on the CPU.
pub(super) fn place<'m>(
    gpu: &Gpu,
    model: &Model<'m>,
    layers: &mut [VulkanLayer],
    runs: &[usize],
    elsewhere: usize,
    cap: Option<usize>,
) -> Result<Option<VulkanOffload<'m>>, LoadError> {
    let moes: Vec<Option<&Moe<'m>>> = model.layers[..layers.len()]
        .iter()
        .map(|l| l.experts.as_ref())
        .collect();
    if moes.iter().all(Option::is_none) {
        return Ok(None);
    }
    // what the driver says is free now, less what is still to come and
    // what the plan leaves free
    let card = gpu.device.info().device_local_bytes as usize;
    let held: usize = layers.iter().map(VulkanLayer::size_bytes).sum();
    let need = elsewhere + reserve(card);
    let free = gpu.device.free_bytes().map(|f| f as usize);
    let free = free.unwrap_or(card.saturating_sub(held));
    let mut budget = free.saturating_sub(need).min(cap.unwrap_or(usize::MAX));

    let mut order: Vec<usize> = (0..moes.len())
        .filter(|&l| moes[l].is_some_and(kernels))
        .collect();
    order.sort_by_key(|&l| Reverse(runs[l]));
    let mut resident = Vec::new();
    for l in order {
        let m = moes[l].expect("a layer of experts");
        if bytes(m) <= budget {
            budget -= bytes(m);
            let moe = layers[l].experts_mut().expect("uploaded routing");
            moe.upload_experts(gpu, m)?;
            resident.push(l);
        }
    }
    // others may have taken memory meanwhile: keep the reserve free
    while let Some(&l) = resident.last() {
        let left = gpu.device.free_bytes().map_or(usize::MAX, |f| f as usize);
        if left >= need {
            break;
        }
        layers[l].experts_mut().expect("resident").drop_experts();
        resident.pop();
    }
    let with_experts = moes.iter().flatten().count();
    if resident.len() == with_experts {
        return Ok(None);
    }
    let host = moes
        .iter()
        .enumerate()
        .map(|(l, m)| m.filter(|_| !resident.contains(&l)).cloned())
        .collect();
    Ok(Some(VulkanOffload {
        host,
        layers: with_experts,
        device_bytes: resident.iter().flat_map(|&l| moes[l]).map(bytes).sum(),
        stats: RefCell::default(),
        times: RefCell::default(),
    }))
}

/// A pass cut after the routing of each layer whose experts run on the
/// CPU: the command lists up to each cut and the physical layer of each.
pub(super) struct Segments {
    before: Vec<Graph>,
    layers: Vec<usize>,
    /// The picks and the input each cut reads back, and the outputs of
    /// their experts.
    picks: HostBuffer<u32>,
    inputs: HostBuffer<f32>,
    outputs: HostBuffer<f32>,
}

impl Segments {
    fn new(gpu: &Gpu, w: &Work) -> kinetic_vulkan::Result<Segments> {
        let e = w.experts.as_ref().expect("work has expert buffers");
        Ok(Segments {
            before: Vec::new(),
            layers: Vec::new(),
            picks: HostBuffer::new(&gpu.device, e.picks.len())?,
            inputs: HostBuffer::new(&gpu.device, w.h.len())?,
            outputs: HostBuffer::new(&gpu.device, e.routed.len())?,
        })
    }

    /// Ends the commands `c` records with the readback of the picks and
    /// input of physical layer `layer`, keeps them, and leaves `c` empty.
    pub fn cut(&mut self, gpu: &Gpu, w: &Work, layer: usize, c: &mut Commands) -> R {
        let e = w.experts.as_ref().expect("work has expert buffers");
        c.download(&e.picks, 0, &self.picks, 0, self.picks.len())?;
        c.download(w.h.values(), 0, &self.inputs, 0, self.inputs.len())?;
        let done = std::mem::replace(c, Commands::new(&gpu.device)?);
        self.before.push(done.finish()?);
        self.layers.push(layer);
        Ok(())
    }
}

impl VulkanOffload<'_> {
    /// Runs the part of a pass of `n` tokens before its last cut: each
    /// segment, then the experts of its layer on the CPU.
    pub fn run(
        &self,
        gpu: &Gpu,
        layers: &[VulkanLayer],
        s: &mut Segments,
        w: &Work,
        n: usize,
    ) -> R {
        let e = w.experts.as_ref().expect("work has expert buffers");
        let mut times = self.times.borrow_mut();
        let mut stats = self.stats.borrow_mut();
        stats.tokens += n as u64;
        for k in 0..s.before.len() {
            let start = Instant::now();
            s.before[k].launch()?;
            let launched = Instant::now();
            times.layers += launched - start;
            let layer = s.layers[k];
            let used = layers[layer].experts().expect("a layer of experts").used;
            let moe = self.host[layer].as_ref().expect("experts the CPU runs");
            let jobs: Vec<Job> = s.picks.as_slice()[..2 * n * used]
                .iter()
                .step_by(2)
                .enumerate()
                .map(|(j, &expert)| Job {
                    row: (j / used) as u32,
                    expert,
                })
                .collect();
            let out = moe.run_experts(&s.inputs.as_slice()[..n * w.d], &jobs);
            s.outputs.as_mut_slice()[..out.len()].copy_from_slice(&out);
            let mut c = Commands::new(&gpu.device)?;
            c.upload(&s.outputs, 0, &e.routed, 0, out.len())?;
            c.submit()?;
            stats.picks += jobs.len() as u64;
            stats.host_experts += jobs.len() as u64;
            times.host += launched.elapsed();
        }
        Ok(())
    }
}

/// Where the routed experts of a model larger than the device run, and
/// what that took, for the run summary.
pub struct ExpertReport {
    /// Layers with routed experts, and those whose experts the device holds.
    pub layers: usize,
    pub device_layers: usize,
    /// Device bytes those experts take.
    pub bytes: usize,
    pub stats: Stats,
    pub times: Times,
}

impl fmt::Display for ExpertReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tokens = self.stats.tokens.max(1) as f64;
        let ms = |d: Duration| d.as_secs_f64() * 1e3 / tokens;
        write!(
            f,
            "experts of {} of {} layers on the GPU ({:.0} MiB), the rest on the CPU; \
             ms per token: {:.1} on the GPU, {:.1} on the CPU",
            self.device_layers,
            self.layers,
            self.bytes as f64 / (1 << 20) as f64,
            ms(self.times.layers),
            ms(self.times.host),
        )
    }
}

impl VulkanModel<'_> {
    /// Command lists for the parts of a pass over `w`, if it is cut.
    pub(super) fn segments_for(&self, w: &Work) -> kinetic_vulkan::Result<Option<Segments>> {
        self.offload
            .as_ref()
            .map(|_| Segments::new(&self.gpu, w))
            .transpose()
    }

    /// Where the routed experts run and what that took, if they do not
    /// all fit on the device.
    pub fn expert_report(&self) -> Option<ExpertReport> {
        let o = self.offload.as_ref()?;
        Some(ExpertReport {
            layers: o.layers,
            device_layers: o.layers - o.host.iter().flatten().count(),
            bytes: o.device_bytes,
            stats: *o.stats.borrow(),
            times: *o.times.borrow(),
        })
    }

    /// One line on the weights that run from the host, if any do.
    pub fn host_weights(&self) -> Option<String> {
        let mut parts = Vec::new();
        let expanded = self.gpu.expanded.borrow();
        if !expanded.0.is_empty() {
            parts.push(format!(
                "no Vulkan kernels for some weights, expanded to f32 on the device: {expanded}"
            ));
        }
        if let Some(o) = &self.offload {
            let mut types: Vec<String> = o
                .host
                .iter()
                .flatten()
                .flat_map(|m| [m.gate.ty, m.down.ty])
                .map(|t| t.to_string())
                .collect();
            types.sort();
            types.dedup();
            parts.push(format!(
                "experts of {} layers ({}) run on the CPU, in host memory",
                o.host.iter().flatten().count(),
                types.join(", ")
            ));
        }
        (!parts.is_empty()).then(|| parts.join("; "))
    }

    /// One line on the device memory this model takes with `state`, and
    /// what the driver says is left free.
    pub fn memory_line(&self, state: &VulkanState) -> String {
        const MIB: usize = 1 << 20;
        let used = self.weight_bytes() + self.state_bytes(state);
        match self.gpu.device.free_bytes() {
            Some(free) => format!(
                "the model takes {} MiB of the GPU, {} MiB left free",
                used / MIB,
                free as usize / MIB
            ),
            None => format!("the model takes {} MiB of the GPU", used / MIB),
        }
    }

    /// Starts the counts of [`VulkanModel::expert_report`] over.
    pub fn reset_expert_report(&self) {
        if let Some(o) = &self.offload {
            *o.stats.borrow_mut() = Stats::default();
            *o.times.borrow_mut() = Times::default();
        }
    }
}
