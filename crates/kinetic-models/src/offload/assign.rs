//! Which slot of a device's expert cache each routed pick of a pass runs
//! from, and which experts must be copied there first. A pass whose layer
//! needs more experts than the cache has slots runs in rounds, each with
//! the experts it can hold; a pick runs in exactly one round.

use super::cache::ExpertCache;
use super::stats::Stats;

/// What a pick runs from in rounds it is not part of.
pub(crate) const SKIP: u32 = u32::MAX;

/// One round of a layer's pass.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Round {
    /// The slot of each pick in this round, [`SKIP`] for the others.
    pub slots: Vec<u32>,
    /// Experts to copy to their slots before the round runs.
    pub copies: Vec<(usize, u32)>,
}

/// The rounds that run the picks `experts` of `layer`, least recently used
/// slots taken first. Counts in `stats` the picks, those whose expert was
/// already in place (hits), and the copies, of `expert_bytes` each.
pub(crate) fn assign(
    cache: &mut ExpertCache,
    layer: usize,
    experts: &[u32],
    expert_bytes: u64,
    stats: &mut Stats,
) -> Vec<Round> {
    let mut rounds = vec![Round::default()];
    let mut done = vec![false; experts.len()];
    while done.iter().any(|d| !d) {
        let round = rounds.last_mut().expect("a round");
        round.slots = vec![SKIP; experts.len()];
        let mut full = false;
        for (j, &e) in experts.iter().enumerate() {
            if done[j] {
                continue;
            }
            let e = e as usize;
            let slot = match cache.lookup(layer, e) {
                Some(slot) => {
                    stats.hits += 1;
                    Some(slot)
                }
                None if full => None,
                None => {
                    let slot = cache.admit(layer, e, |_, _| 0, u32::MAX, false);
                    if let Some(slot) = slot {
                        round.copies.push((e, slot));
                        stats.loaded += 1;
                        stats.loaded_bytes += expert_bytes;
                    }
                    full = slot.is_none();
                    slot
                }
            };
            if let Some(slot) = slot {
                cache.protect(layer, slot);
                round.slots[j] = slot;
                done[j] = true;
            }
        }
        cache.release();
        if done.iter().any(|d| !d) {
            rounds.push(Round::default());
        }
    }
    stats.picks += experts.len() as u64;
    rounds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(cache: &mut ExpertCache, experts: &[u32], stats: &mut Stats) -> Vec<Round> {
        assign(cache, 0, experts, 10, stats)
    }

    #[test]
    fn picks_reuse_cached_experts_and_copy_the_rest() {
        let mut cache = ExpertCache::new(&[4], vec![0], 8);
        let mut stats = Stats::default();
        let rounds = run(&mut cache, &[3, 5, 3], &mut stats);
        assert_eq!(rounds.len(), 1);
        assert_eq!(rounds[0].slots, [0, 1, 0]);
        assert_eq!(rounds[0].copies, [(3, 0), (5, 1)]);
        let rounds = run(&mut cache, &[5, 6], &mut stats);
        assert_eq!(rounds[0].slots, [1, 2]);
        assert_eq!(rounds[0].copies, [(6, 2)]);
        assert_eq!((stats.picks, stats.hits, stats.loaded), (5, 2, 3));
    }

    #[test]
    fn the_least_recently_used_slot_goes_first() {
        let mut cache = ExpertCache::new(&[2], vec![0], 8);
        let mut stats = Stats::default();
        run(&mut cache, &[1], &mut stats);
        run(&mut cache, &[2], &mut stats);
        run(&mut cache, &[1], &mut stats);
        let rounds = run(&mut cache, &[3], &mut stats);
        assert_eq!(rounds[0].copies, [(3, 1)]);
        assert_eq!(cache.slot(0, 1), Some(0));
        assert_eq!(cache.slot(0, 2), None);
    }

    /// More experts than slots: every pick runs once, in a round whose
    /// copies put its expert where it reads it.
    #[test]
    fn a_pass_larger_than_the_cache_runs_in_rounds() {
        let mut cache = ExpertCache::new(&[2], vec![0], 8);
        let mut stats = Stats::default();
        let experts = [0, 1, 2, 3, 1, 4];
        let rounds = run(&mut cache, &experts, &mut stats);
        assert_eq!(rounds.len(), 3);
        let mut held = [None; 2];
        let mut ran = vec![0; experts.len()];
        for round in &rounds {
            for &(e, slot) in &round.copies {
                held[slot as usize] = Some(e as u32);
            }
            for (j, &slot) in round.slots.iter().enumerate() {
                if slot != SKIP {
                    assert_eq!(held[slot as usize], Some(experts[j]));
                    ran[j] += 1;
                }
            }
        }
        assert_eq!(ran, [1; 6]);
    }
}
