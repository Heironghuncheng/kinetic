//! Where each part of a mixture-of-experts model lives: everything but the
//! routed experts on the device, then as many experts as the device has
//! room for in a cache, the rest in host memory or, beyond it, on disk.

use std::fmt;

use super::cache::SlotLayout;

/// Device memory the planner leaves free on a device of `total` bytes:
/// the larger of [`MIN_RESERVE`] and a twentieth of the device. The desktop
/// and other programs take and return memory while a model runs, and on
/// Windows an allocation past the dedicated memory does not fail but lands
/// in system memory, reached over PCIe, which makes every kernel that
/// touches it many times slower. A margin that stays free keeps everything
/// in dedicated memory.
pub fn device_reserve(total: usize) -> usize {
    MIN_RESERVE.max(total / 20)
}

/// The least [`device_reserve`] keeps free.
pub const MIN_RESERVE: usize = 300 << 20;

/// The placement chosen at load time.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Device bytes of the dense weights, head and caches of one sequence.
    pub dense_bytes: usize,
    /// Bytes of every routed expert of the model.
    pub expert_bytes: usize,
    /// Per slot size: bytes per slot, slots and the layers using them.
    pub pools: Vec<PoolPlan>,
    /// Pinned host bytes used to stage copies to the device.
    pub staging_bytes: usize,
    /// Measured bytes per second of expert weights the host runs through
    /// for one token, and copies to the device.
    pub host_rate: f64,
    pub copy_rate: f64,
}

#[derive(Clone, Debug)]
pub struct PoolPlan {
    pub slot_bytes: usize,
    pub slots: usize,
    pub layers: usize,
}

impl Plan {
    /// Device bytes the expert cache holds.
    pub fn cache_bytes(&self) -> usize {
        self.pools.iter().map(|p| p.slot_bytes * p.slots).sum()
    }

    /// Fraction of all routed experts the device caches.
    pub fn cached_share(&self) -> f64 {
        self.cache_bytes() as f64 / self.expert_bytes.max(1) as f64
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const MIB: f64 = (1 << 20) as f64;
        write!(
            f,
            "device: {:.0} MiB dense, {:.0} MiB expert cache ({:.0}% of {:.0} MiB of experts in",
            self.dense_bytes as f64 / MIB,
            self.cache_bytes() as f64 / MIB,
            100.0 * self.cached_share(),
            self.expert_bytes as f64 / MIB,
        )?;
        for p in &self.pools {
            write!(
                f,
                " {} slots of {:.2} MiB for {} layers;",
                p.slots,
                p.slot_bytes as f64 / MIB,
                p.layers
            )?;
        }
        write!(
            f,
            "); host: {:.0} MiB pinned staging, the rest mapped from the file; host experts at {:.1} GB/s, copies at {:.1} GB/s",
            self.staging_bytes as f64 / MIB,
            self.host_rate / 1e9,
            self.copy_rate / 1e9,
        )
    }
}

/// Groups layers by slot size and shares `budget` device bytes among the
/// groups in proportion to the bytes of their experts times how many times
/// per token their layers run, `visits`, so that a layer run three times a
/// token caches three times the share of its experts another one does. A
/// group whose share would hold all its experts takes just those, and the
/// rest is shared again. Returns the slot sizes, the slots of each and the
/// group of every layer.
pub fn share_cache(
    layouts: &[SlotLayout],
    n_experts: usize,
    visits: &[usize],
    budget: usize,
) -> (Vec<PoolPlan>, Vec<usize>) {
    let mut sizes: Vec<usize> = layouts.iter().map(SlotLayout::bytes).collect();
    sizes.sort_unstable();
    sizes.dedup();
    let pool_of: Vec<usize> = layouts
        .iter()
        .map(|l| sizes.binary_search(&l.bytes()).expect("size is listed"))
        .collect();
    let layers = |p: usize| pool_of.iter().filter(|&&q| q == p).count();
    let weight = |p: usize| -> f64 {
        let visited: usize = (0..layouts.len())
            .filter(|&l| pool_of[l] == p)
            .map(|l| visits[l])
            .sum();
        (sizes[p] * n_experts * visited) as f64
    };
    let all = |p: usize| sizes[p] * n_experts * layers(p);
    let mut slots = vec![0; sizes.len()];
    let mut open: Vec<usize> = (0..sizes.len()).collect();
    let mut left = budget;
    while !open.is_empty() {
        let total: f64 = open.iter().map(|&p| weight(p)).sum();
        let share = |p: usize| (left as f64 * weight(p) / total.max(1.0)) as usize;
        let (full, rest): (Vec<usize>, Vec<usize>) =
            open.iter().partition(|&&p| share(p) >= all(p));
        if full.is_empty() {
            for &p in &rest {
                slots[p] = share(p) / sizes[p];
            }
            break;
        }
        for p in full {
            slots[p] = n_experts * layers(p);
            left -= all(p);
        }
        open = rest;
    }
    let pools = sizes
        .iter()
        .enumerate()
        .map(|(p, &slot_bytes)| PoolPlan {
            slot_bytes,
            slots: slots[p],
            layers: layers(p),
        })
        .collect();
    (pools, pool_of)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_share_the_budget_by_expert_bytes() {
        let small = SlotLayout {
            gate: 100,
            up: 100,
            down: 100,
            scales: 0,
        };
        let large = SlotLayout {
            gate: 200,
            up: 200,
            down: 200,
            scales: 0,
        };
        let (pools, pool_of) = share_cache(&[small, large, small], 10, &[1; 3], 6000);
        assert_eq!(pool_of, [0, 1, 0]);
        // 6000 + 6000 bytes of experts per group: 3000 bytes each.
        assert_eq!((pools[0].slots, pools[0].layers), (10, 2));
        assert_eq!((pools[1].slots, pools[1].layers), (5, 1));
        let (pools, _) = share_cache(&[small], 10, &[1], 1 << 30);
        assert_eq!(pools[0].slots, 10);
    }

    #[test]
    fn layers_run_more_often_get_more_of_the_budget() {
        let layout = |bytes| SlotLayout {
            gate: bytes,
            up: bytes,
            down: bytes,
            scales: 0,
        };
        let (once, thrice) = (layout(100), layout(200));
        // weights 3000 and 3 * 6000: the second group's share of 9000 is
        // more than its 6000 bytes, which it takes, leaving 3000 to the first
        let (pools, _) = share_cache(&[once, thrice], 10, &[1, 3], 9000);
        assert_eq!((pools[0].slots, pools[1].slots), (10, 10));
        // 4200 * 3000 / 21000 = 600 bytes, two slots; 3600 bytes, six slots
        let (pools, _) = share_cache(&[once, thrice], 10, &[1, 3], 4200);
        assert_eq!((pools[0].slots, pools[1].slots), (2, 6));
    }
}
