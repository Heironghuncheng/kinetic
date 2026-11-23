//! Routed experts of a model larger than the device.
//!
//! Everything else lives on the device; what device memory is left holds
//! slots for each layer's experts. A pass stops after each layer's routing,
//! the picks come back, and [`assign`] says which slot each pick runs from
//! and which experts to copy there first, least recently used slots taken
//! first; a pass needing more experts than a layer's slots runs that layer
//! in rounds. The products run from the slots, then the pass goes on. Where
//! an expert sits never changes the bits of its product.

use std::cell::RefCell;
use std::fmt;
use std::time::{Duration, Instant};

use kinetic_vulkan::{Commands, ExpertSlots, Graph, HostBuffer};

use super::layer::{VulkanLayer, R};
use super::moe::{sources, ExpertView};
use super::work::Work;
use super::{vulkan_error, Gpu, VulkanModel};
use crate::blocks::Experts;
use crate::offload::assign::assign;
use crate::offload::cache::ExpertCache;
use crate::offload::plan::DEVICE_RESERVE;
use crate::offload::stats::Stats;
use crate::{LoadError, Model};

/// Experts staged in host memory per copy to the device.
const STAGED: usize = 16;

/// The slots of one layer and where its experts come from.
struct LayerSlots<'m> {
    /// Slots of the gate and up products, each with where it writes in a
    /// row of both, then of the down product.
    gate_up: Vec<(ExpertSlots, usize)>,
    down: ExpertSlots,
    sources: (Vec<(Experts<'m>, usize)>, Experts<'m>),
}

impl LayerSlots<'_> {
    /// Bytes of one of this layer's experts.
    fn bytes(&self) -> usize {
        self.parts().map(|(s, _)| s.expert_bytes()).sum()
    }

    fn view(&self) -> ExpertView<'_> {
        ExpertView {
            gate_up: self
                .gate_up
                .iter()
                .map(|(s, at)| (s.matrix(), *at))
                .collect(),
            down: self.down.matrix(),
            count: self.down.slots(),
        }
    }

    fn parts(&self) -> impl Iterator<Item = (&ExpertSlots, &Experts<'_>)> {
        let gate_up = self.gate_up.iter().zip(&self.sources.0);
        gate_up
            .map(|((s, _), (e, _))| (s, e))
            .chain([(&self.down, &self.sources.1)])
    }
}

/// Seconds spent per part of the passes, since load or the last reset.
#[derive(Clone, Copy, Debug, Default)]
pub struct Times {
    /// Running the layers up to each routing, and reading the picks back.
    pub layers: Duration,
    /// Writing experts into host staging.
    pub staging: Duration,
    /// Copying them to the device.
    pub copies: Duration,
    /// Running the experts.
    pub experts: Duration,
}

pub(super) struct VulkanOffload<'m> {
    /// Per physical layer, if it has routed experts.
    layers: Vec<Option<LayerSlots<'m>>>,
    /// Experts of a layer.
    experts: usize,
    cache: RefCell<ExpertCache>,
    stats: RefCell<Stats>,
    times: RefCell<Times>,
    staging: RefCell<HostBuffer<u8>>,
}

/// Plans the routed experts of `model`, whose other weights `layers` hold:
/// every expert on the device if they fit in what it has left after
/// `elsewhere` bytes, and at most `cap` bytes; else slots for as many as
/// fit, and the offload that fills them.
pub(super) fn place<'m>(
    gpu: &Gpu,
    model: &Model<'m>,
    layers: &mut [VulkanLayer],
    elsewhere: usize,
    cap: Option<usize>,
) -> Result<Option<VulkanOffload<'m>>, LoadError> {
    let moes: Vec<_> = model.layers[..layers.len()]
        .iter()
        .map(|l| l.experts.as_ref())
        .collect();
    let expert_bytes = |m: &crate::blocks::Moe| {
        let (gate_up, down) = sources(m);
        gate_up.iter().map(|(e, _)| e.expert_bytes()).sum::<usize>() + down.expert_bytes()
    };
    let total: usize = moes
        .iter()
        .flatten()
        .map(|m| m.n_experts() * expert_bytes(m))
        .sum();
    let used: usize = layers.iter().map(VulkanLayer::size_bytes).sum::<usize>() + elsewhere;
    let free =
        (gpu.device.info().device_local_bytes as usize).saturating_sub(used + DEVICE_RESERVE);
    let budget = cap.map_or(free, |c| c.min(free));
    if total <= budget {
        for (layer, m) in layers.iter_mut().zip(&moes) {
            if let (Some(vm), Some(m)) = (layer.experts_mut(), m) {
                vm.upload_experts(gpu, m)?;
            }
        }
        return Ok(None);
    }
    // a slot in every layer, each the size of that layer's experts
    let per_slot: usize = moes.iter().flatten().map(|m| expert_bytes(m)).sum();
    // staging holds the largest experts
    let slot_bytes = moes
        .iter()
        .flatten()
        .map(|m| expert_bytes(m))
        .max()
        .unwrap_or(0);
    let most = moes
        .iter()
        .flatten()
        .map(|m| m.n_experts())
        .max()
        .unwrap_or(0);
    let slots = (budget / per_slot.max(1)).min(most);
    let fewest = moes.iter().flatten().map(|m| m.n_used).max().unwrap_or(0);
    if slots < fewest {
        return Err(LoadError(format!(
            "the device has room for {slots} experts per layer and a token needs {fewest}"
        )));
    }
    let device = &gpu.device;
    let mut planned = Vec::with_capacity(moes.len());
    for m in &moes {
        planned.push(match m {
            Some(m) => {
                let (gate_up, down) = sources(m);
                let pool = |e: &Experts| ExpertSlots::new(device, e.ty, slots, e.rows, e.cols);
                Some(LayerSlots {
                    gate_up: gate_up
                        .iter()
                        .map(|(e, at)| Ok((pool(e)?, *at)))
                        .collect::<kinetic_vulkan::Result<_>>()
                        .map_err(vulkan_error)?,
                    down: pool(&down).map_err(vulkan_error)?,
                    sources: (gate_up, down),
                })
            }
            None => None,
        });
    }
    let counts: Vec<usize> = planned
        .iter()
        .map(|p| if p.is_some() { slots } else { 0 })
        .collect();
    Ok(Some(VulkanOffload {
        cache: RefCell::new(ExpertCache::new(&counts, (0..counts.len()).collect(), most)),
        layers: planned,
        experts: most,
        stats: RefCell::default(),
        times: RefCell::default(),
        staging: RefCell::new(HostBuffer::new(device, STAGED * slot_bytes).map_err(vulkan_error)?),
    }))
}

/// A pass cut after each layer's routing: the command lists up to each
/// cut, the physical layer each cut belongs to, and each layer's products.
pub(super) struct Segments {
    before: Vec<Graph>,
    layers: Vec<usize>,
    products: Vec<Option<Graph>>,
    /// The picks each cut reads back, and the slots it runs them from.
    picks: HostBuffer<u32>,
    slots: HostBuffer<u32>,
}

impl Segments {
    /// Ends the commands `c` records with the readback of the picks of
    /// physical layer `layer`, keeps them, and leaves `c` empty.
    pub fn cut(&mut self, gpu: &Gpu, w: &Work, layer: usize, c: &mut Commands) -> R {
        let e = w.experts.as_ref().expect("work has expert buffers");
        c.download(&e.picks, 0, &self.picks, 0, self.picks.len())?;
        let done = std::mem::replace(c, Commands::new(&gpu.device)?);
        self.before.push(done.finish()?);
        self.layers.push(layer);
        Ok(())
    }
}

impl VulkanOffload<'_> {
    /// The command lists of a pass over `w` before the network records it.
    pub fn segments(
        &self,
        gpu: &Gpu,
        layers: &[VulkanLayer],
        w: &Work,
    ) -> kinetic_vulkan::Result<Segments> {
        let e = w.experts.as_ref().expect("work has expert buffers");
        let products = layers
            .iter()
            .zip(&self.layers)
            .map(|(layer, slots)| {
                let (Some(slots), Some(moe)) = (slots, layer.experts()) else {
                    return Ok(None);
                };
                let mut c = Commands::new(&gpu.device)?;
                moe.record_products(gpu, w, &slots.view(), &mut c)?;
                c.finish().map(Some)
            })
            .collect::<kinetic_vulkan::Result<_>>()?;
        Ok(Segments {
            before: Vec::new(),
            layers: Vec::new(),
            products,
            picks: HostBuffer::new(&gpu.device, e.picks.len())?,
            slots: HostBuffer::new(&gpu.device, e.picks.len())?,
        })
    }

    /// Runs the part of a pass of `n` tokens before its last cut: each
    /// segment, then the experts of its layer from their slots.
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
        self.stats.borrow_mut().tokens += n as u64;
        for (k, graph) in s.before.iter().enumerate() {
            let start = Instant::now();
            graph.launch()?;
            times.layers += start.elapsed();
            let layer = s.layers[k];
            let used = layers[layer].experts().expect("a layer of experts").used;
            let jobs = n * used;
            let experts: Vec<u32> = s.picks.as_slice()[..2 * jobs]
                .iter()
                .step_by(2)
                .copied()
                .collect();
            let slots = self.layers[layer]
                .as_ref()
                .expect("slots of a layer of experts");
            let rounds = assign(
                &mut self.cache.borrow_mut(),
                layer,
                &experts,
                slots.bytes() as u64,
                &mut self.stats.borrow_mut(),
            );
            for round in rounds {
                for chunk in round.copies.chunks(STAGED) {
                    self.copy(gpu, slots, chunk, &mut times)?;
                }
                let picks = s.picks.as_slice()[..2 * jobs].to_vec();
                for (j, (dst, src)) in s
                    .slots
                    .as_mut_slice()
                    .chunks_exact_mut(2)
                    .zip(picks.chunks_exact(2))
                    .take(jobs)
                    .enumerate()
                {
                    dst[0] = round.slots[j];
                    dst[1] = src[1];
                }
                let start = Instant::now();
                let mut c = Commands::new(&gpu.device)?;
                c.upload(&s.slots, 0, &e.picks, 0, 2 * jobs)?;
                c.submit()?;
                s.products[layer]
                    .as_ref()
                    .expect("products of a layer of experts")
                    .launch()?;
                times.experts += start.elapsed();
            }
        }
        Ok(())
    }

    /// Stages the experts of `copies` and copies each to its slot.
    fn copy(&self, gpu: &Gpu, slots: &LayerSlots, copies: &[(usize, u32)], times: &mut Times) -> R {
        let start = Instant::now();
        let mut staging = self.staging.borrow_mut();
        let mut c = Commands::new(&gpu.device)?;
        let mut at = 0;
        for &(expert, slot) in copies {
            for (pool, source) in slots.parts() {
                let bytes = pool.expert_bytes();
                pool.stage(
                    &source.expert(expert),
                    &mut staging.as_mut_slice()[at..at + bytes],
                )?;
                pool.record_copy(&staging, at, slot as usize, &mut c)?;
                at += bytes;
            }
        }
        let staged = Instant::now();
        times.staging += staged - start;
        c.submit()?;
        times.copies += staged.elapsed();
        Ok(())
    }
}

/// What the offload of routed experts holds and did, for the run summary.
pub struct ExpertReport {
    /// Slots per layer, layers with experts, and experts per layer.
    pub slots: usize,
    pub layers: usize,
    pub experts: usize,
    /// Device bytes the slots take.
    pub bytes: usize,
    pub stats: Stats,
    pub times: Times,
}

impl fmt::Display for ExpertReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.stats;
        let tokens = s.tokens.max(1) as f64;
        let ms = |d: Duration| d.as_secs_f64() * 1e3 / tokens;
        let t = &self.times;
        write!(
            f,
            "experts: {} of {} per layer on the device ({:.0} MiB), hit rate {:.1}%, {:.1} MiB \
             copied per token; ms per token: {:.1} layers, {:.1} staging, {:.1} copies, {:.1} experts",
            self.slots,
            self.experts,
            self.bytes as f64 / (1 << 20) as f64,
            100.0 * s.hit_rate(),
            s.loaded_bytes as f64 / tokens / (1 << 20) as f64,
            ms(t.layers),
            ms(t.staging),
            ms(t.copies),
            ms(t.experts),
        )
    }
}

impl VulkanModel<'_> {
    /// Command lists for the parts of a pass over `w`, if it is cut.
    pub(super) fn segments_for(&self, w: &Work) -> kinetic_vulkan::Result<Option<Segments>> {
        let o = self.offload.as_ref();
        o.map(|o| o.segments(&self.gpu, &self.layers, w))
            .transpose()
    }

    /// What the offload of routed experts holds and did, if the model's
    /// experts do not all fit on the device.
    pub fn expert_report(&self) -> Option<ExpertReport> {
        let o = self.offload.as_ref()?;
        let held: Vec<&LayerSlots> = o.layers.iter().flatten().collect();
        let first = held.first()?;
        let bytes = held
            .iter()
            .flat_map(|l| l.parts())
            .map(|(s, _)| s.matrix().size_bytes())
            .sum();
        Some(ExpertReport {
            slots: first.down.slots(),
            layers: held.len(),
            experts: o.experts,
            bytes,
            stats: *o.stats.borrow(),
            times: *o.times.borrow(),
        })
    }

    /// Starts the counts of [`VulkanModel::expert_report`] over.
    pub fn reset_expert_report(&self) {
        if let Some(o) = &self.offload {
            *o.stats.borrow_mut() = Stats::default();
            *o.times.borrow_mut() = Times::default();
        }
    }
}
