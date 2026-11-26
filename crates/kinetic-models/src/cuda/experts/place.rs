//! Where each picked expert runs, the copies that bring experts into the
//! device cache, and the copies made while the host waits for the device.

use std::time::Instant;

use kinetic_cpu::pool;
use kinetic_cuda::Event;

use super::form::{stage, stage_spread};
use super::offload::{Offload, DEMAND_STAGING, PREFETCH_STAGING, TRANSIT_SLOTS};
use super::Gpu;
use crate::blocks::{Job, Pick};

/// Where an expert of the current layer runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Place {
    Unused,
    /// In this slot of the cache.
    Device(u32),
    /// In this transit slot, for this batch only.
    Transit(u32),
    Host,
}

/// Fewest tokens of a batch an expert must have to be copied to the device
/// while its layer waits; with fewer the host runs it, since the copy and
/// the product on the device take longer.
const COPY_FROM: usize = 4;

/// Most experts one layer adds to the queue of copies into the cache, so
/// that a layer of rare experts does not evict common ones at once.
const ADMITTED_PER_LAYER: usize = 2;

impl Offload<'_> {
    /// Places every expert of layer `l` that `picks` use: in its cache slot,
    /// on the host, or, when that finishes sooner, copied into a cache slot
    /// it may take or else into a transit slot. Experts the host runs that
    /// are used more than one the cache holds are queued to be copied into
    /// it later. Counts the picks. The copies are queued on the main stream.
    pub(super) fn route(
        &mut self,
        l: usize,
        picks: &[Pick],
        gpu: &Gpu,
    ) -> kinetic_cuda::Result<Vec<Place>> {
        let n_exp = self.n_experts();
        self.cache.release();
        if self.in_flight > 0 && self.copied.is_done()? {
            self.settle();
        }
        let mut tokens_of = vec![0usize; n_exp];
        for p in picks {
            tokens_of[p.expert as usize] += 1;
        }
        let used: Vec<usize> = (0..n_exp).filter(|&e| tokens_of[e] > 0).collect();
        // Use counts once per batch, so a long prompt does not outweigh the
        // tokens generated after it.
        for &e in &used {
            self.usage[l * n_exp + e] += 1;
        }
        if l == 0 {
            self.stats.tokens += (picks.len() / self.moes[0].n_used) as u64;
        }
        self.stats.picks += picks.len() as u64;
        self.stats.used += used.len() as u64;
        self.stats.predicted_used += used
            .iter()
            .filter(|&&e| self.predicted.contains(&(e as u32)))
            .count() as u64;

        let bytes = self.layouts[l].bytes();
        // A copy into the cache still under way is waited for only by an
        // expert of many tokens, when the host would take longer than a copy.
        let waiting = used.iter().any(|&e| {
            self.cache.slot(l, e).is_some_and(|slot| {
                self.cache.is_pending(l, slot)
                    && tokens_of[e] >= COPY_FROM
                    && self.costs.host(bytes, tokens_of[e]) > self.costs.copy(bytes)
            })
        });
        if waiting {
            self.stage_all()?;
        }
        let mut place = vec![Place::Unused; n_exp];
        let mut cold = Vec::new();
        for &e in &used {
            match self.cache.slot(l, e) {
                Some(slot) if !self.cache.is_pending(l, slot) => {
                    self.cache.lookup(l, e);
                    self.cache.protect(l, slot);
                    place[e] = Place::Device(slot);
                }
                _ => cold.push(e),
            }
        }
        cold.sort_by_key(|&e| std::cmp::Reverse(tokens_of[e]));

        let usage = &self.usage;
        let uses = |layer: usize, e: usize| usage[layer * n_exp + e];
        // Staging a copy and running an expert both take the host, one after
        // the other, so an expert runs there only when that alone is cheaper.
        let mut host_time = 0.0;
        let mut loads = Vec::new();
        let mut transit = 0;
        let mut admitted = 0;
        for e in cold {
            let host = self.costs.host(bytes, tokens_of[e]);
            let copy = self.costs.copy(bytes);
            let pending = self.cache.slot(l, e).is_some();
            let few = tokens_of[e] < COPY_FROM;
            let target = if pending || few || host < copy {
                Place::Host
            } else if let Some(slot) = self.cache.admit(l, e, uses, uses(l, e), false) {
                self.cache.protect(l, slot);
                Place::Device(slot)
            } else if transit < TRANSIT_SLOTS {
                transit += 1;
                Place::Transit(transit as u32 - 1)
            } else {
                Place::Host
            };
            place[e] = target;
            if target != Place::Host {
                loads.push((e, target));
                continue;
            }
            host_time += host;
            let room = admitted < ADMITTED_PER_LAYER && self.admissions.len() < PREFETCH_STAGING;
            if !pending && room {
                if let Some(slot) = self.cache.admit(l, e, uses, uses(l, e), true) {
                    self.admissions.push_back((l, e, slot));
                    admitted += 1;
                }
            }
        }
        self.expected_host = host_time;
        self.stats.hits += picks
            .iter()
            .filter(|p| matches!(place[p.expert as usize], Place::Device(_)))
            .count() as u64;
        let host_experts = place.iter().filter(|p| **p == Place::Host).count();
        self.stats.host_experts += host_experts as u64;
        self.stats.host_bytes += (host_experts * bytes) as u64;
        self.stats.loaded += loads.len() as u64;
        self.stats.loaded_bytes += (loads.len() * bytes) as u64;
        self.load(l, &loads, gpu)?;
        Ok(place)
    }

    /// Copies `loads`, experts of layer `l` and their cache or transit
    /// slots, on the main stream, staging a round of them at a time in
    /// parallel.
    pub(super) fn load(
        &mut self,
        l: usize,
        loads: &[(usize, Place)],
        gpu: &Gpu,
    ) -> kinetic_cuda::Result<()> {
        let region = self.staging.region;
        let half = DEMAND_STAGING / 2;
        let pool = self.cache.pool_of(l);
        let slot_bytes = self.plan.pools[pool].slot_bytes;
        // Rounds alternate between the two halves of the staging, so one
        // half is filled while the other's copies run; a half is refilled
        // only once its copies, of this call or an earlier one, are done.
        for (round, batch) in loads.chunks(half).enumerate() {
            let side = round % 2;
            self.staged[side].synchronize()?;
            let moe = &self.moes[l];
            let staging = &mut self.staging.demand.as_mut_slice()[side * half * region..];
            pool::for_each_chunk(&mut staging[..batch.len() * region], region, |i, dst| {
                stage(moe, batch[i].0, dst);
            });
            let len = self.layouts[l].bytes();
            for (i, &(_, target)) in batch.iter().enumerate() {
                let at = (side * half + i) * region;
                let (dst, start) = match target {
                    Place::Device(slot) => (&mut self.arenas[pool], slot as usize * slot_bytes),
                    Place::Transit(slot) => (&mut self.transit, slot as usize * region),
                    Place::Unused | Place::Host => unreachable!("loads go to the device"),
                };
                // SAFETY: this half of the staging is rewritten only after
                // its event, recorded below, or a synchronization of this
                // stream.
                unsafe {
                    self.staging
                        .demand
                        .upload(at..at + len, dst, start, &gpu.stream)?
                };
            }
            self.staged[side].record(&gpu.stream)?;
        }
        Ok(())
    }

    /// Queues copies into the cache of the experts `predicted` for layer
    /// `l` that it lacks, as long as each evicts only experts used less
    /// often. Their slots wait for the copy.
    pub fn admit_predicted(&mut self, l: usize, predicted: &[u32]) {
        self.predicted = predicted.to_vec();
        self.stats.predicted += predicted.len() as u64;
        let n_exp = self.n_experts();
        for &e in predicted {
            let e = e as usize;
            if self.admissions.len() >= PREFETCH_STAGING || self.cache.slot(l, e).is_some() {
                continue;
            }
            let usage = &self.usage;
            let uses = |layer: usize, e: usize| usage[layer * n_exp + e];
            if let Some(slot) = self.cache.admit(l, e, uses, uses(l, e) + 1, true) {
                self.admissions.push_back((l, e, slot));
            }
        }
    }

    /// Runs `jobs` of layer `l` on `h` on the host, timing them, and returns
    /// their outputs, one row per job.
    pub fn run_jobs(&mut self, l: usize, h: &[f32], jobs: &[Job]) -> Vec<f32> {
        if jobs.is_empty() {
            self.costs.observe_host(0.0, 0.0);
            return Vec::new();
        }
        let start = Instant::now();
        let outputs = super::run_host(&self.moes[l], h, jobs);
        self.costs
            .observe_host(self.expected_host, start.elapsed().as_secs_f64());
        outputs
    }

    /// Waits for `ready`, meanwhile staging queued copies into the cache.
    pub fn wait(&mut self, ready: &Event) -> kinetic_cuda::Result<()> {
        while !ready.is_done()? {
            if !self.stage_next()? {
                return ready.synchronize();
            }
        }
        Ok(())
    }

    /// Marks the copies into the cache done, all but those still queued.
    fn settle(&mut self) {
        self.cache.settle();
        self.in_flight = 0;
        for &(l, _, slot) in &self.admissions {
            self.cache.set_pending(l, slot);
        }
    }

    /// Stages every queued copy and waits for them all.
    fn stage_all(&mut self) -> kinetic_cuda::Result<()> {
        loop {
            if !self.stage_next()? {
                self.copied.synchronize()?;
                self.settle();
            }
            if self.admissions.is_empty() && self.in_flight == 0 {
                return Ok(());
            }
        }
    }

    /// Stages the next queued expert and queues its copy on the second
    /// stream; false when nothing is queued or every staging region waits
    /// for its copy.
    fn stage_next(&mut self) -> kinetic_cuda::Result<bool> {
        if self.admissions.is_empty() {
            return Ok(false);
        }
        if self.in_flight == PREFETCH_STAGING {
            if !self.copied.is_done()? {
                return Ok(false);
            }
            self.settle();
        }
        let (l, e, slot) = self.admissions.pop_front().expect("one is queued");
        let region = self.staging.region;
        let at = self.in_flight * region;
        let staging = &mut self.staging.prefetch.as_mut_slice()[at..];
        let len = stage_spread(&self.moes[l], e, staging);
        let pool = self.cache.pool_of(l);
        let slot_bytes = self.plan.pools[pool].slot_bytes;
        // SAFETY: this staging region is rewritten only after `copied` has
        // passed, and the slot is read only once its copy is settled.
        unsafe {
            self.staging.prefetch.upload(
                at..at + len,
                &mut self.arenas[pool],
                slot as usize * slot_bytes,
                &self.copies,
            )?
        };
        self.copied.record(&self.copies)?;
        self.in_flight += 1;
        self.stats.prefetched += 1;
        self.stats.prefetched_bytes += len as u64;
        Ok(true)
    }
}
