//! The search for the largest expert cache that leaves the device the room a
//! sequence needs. What a placement leaves free is only known once it is
//! made: large allocations can take more than they hold, and the desktop
//! takes and returns memory meanwhile.

/// Least growth of the expert cache worth placing it again for.
pub const GROW_STEP: usize = 64 << 20;

/// Most placements one search makes.
pub const MOST_PLACEMENTS: usize = 12;

pub struct BudgetSearch {
    need: usize,
    /// The largest budget seen to leave the room needed, and the cache
    /// bytes it held; the smallest seen not to.
    fit: usize,
    held: usize,
    over: usize,
}

impl BudgetSearch {
    /// A search for budgets that leave `need` bytes free.
    pub fn new(need: usize) -> BudgetSearch {
        BudgetSearch {
            need,
            fit: 0,
            held: 0,
            over: usize::MAX,
        }
    }

    /// After a placement with `budget` that left `left` bytes free and holds
    /// `cached` bytes of experts: the budget to place with next, or `None`
    /// to keep this placement.
    pub fn next(&mut self, budget: usize, left: usize, cached: usize) -> Option<usize> {
        if budget == 0 {
            return None;
        }
        if left >= self.need {
            // A larger budget the cache did not take holds nothing more:
            // every expert is in, or the cache is capped.
            if self.fit > 0 && cached <= self.held {
                return None;
            }
            self.fit = budget;
            self.held = cached;
            // The next one is between the two seen, and never past what
            // was left.
            let next = (budget + (left - self.need)).min(budget + (self.over - budget) / 2);
            return (next >= budget + GROW_STEP).then_some(next);
        }
        // A budget that fit before and no longer does: the device has less
        // room than it had.
        if budget <= self.fit {
            self.fit = 0;
        }
        self.over = budget;
        Some(
            budget
                .saturating_sub(self.need - left)
                .max(self.fit + (budget - self.fit) / 2),
        )
    }

    /// Whether no budget has left the room needed yet.
    pub fn short(&self) -> bool {
        self.fit == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    /// The budget each placement of a search uses, on a device of `free`
    /// bytes whose cache holds at most `experts` bytes and takes `extra`
    /// bytes per byte it holds on top.
    fn search(need: usize, free: usize, experts: usize, extra: f64) -> Vec<usize> {
        let mut s = BudgetSearch::new(need);
        let mut budget = free - need;
        let mut tried = vec![budget];
        for _ in 0..MOST_PLACEMENTS {
            let cached = budget.min(experts);
            let taken = cached + (cached as f64 * extra) as usize;
            let left = free.saturating_sub(taken);
            match s.next(budget, left, cached) {
                Some(next) => budget = next,
                None => return tried,
            }
            tried.push(budget);
        }
        panic!("no placement kept after {tried:?}");
    }

    #[test]
    fn every_expert_fitting_ends_at_once() {
        let tried = search(500 * MIB, 5000 * MIB, 100 * MIB, 0.0);
        assert!(tried.len() <= 2, "{tried:?}");
    }

    #[test]
    fn a_capped_cache_ends_at_once() {
        let tried = search(500 * MIB, 5000 * MIB, 3 * MIB, 0.1);
        assert!(tried.len() <= 2, "{tried:?}");
    }

    #[test]
    fn allocations_that_take_more_end_on_a_budget_that_fits() {
        let (need, free) = (500 * MIB, 5000 * MIB);
        let tried = search(need, free, 50_000 * MIB, 0.08);
        let kept = *tried.last().unwrap();
        assert!(
            free - kept - (kept as f64 * 0.08) as usize >= need,
            "{tried:?}"
        );
    }

    #[test]
    fn a_device_that_lost_room_shrinks_the_cache() {
        let need = 500 * MIB;
        let mut s = BudgetSearch::new(need);
        assert_eq!(s.next(4000 * MIB, 600 * MIB, 4000 * MIB), Some(4100 * MIB));
        // the same budget no longer fits
        let next = s.next(4000 * MIB, 300 * MIB, 4000 * MIB).unwrap();
        assert!(next < 4000 * MIB);
        assert!(s.short());
    }
}
