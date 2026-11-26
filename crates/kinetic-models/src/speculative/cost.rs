use std::time::Duration;

/// How fast a measured value follows a step that took longer than it, and
/// one that took less. Other work on the machine only ever slows a step,
/// so a slow one moves it little.
const RISE: f32 = 0.1;
const FALL: f32 = 0.3;

/// Steps after which a row count's cost, not measured again since, is
/// forgotten, so a pass judged dear on a busy moment is tried again.
const STALE: usize = 64;

/// What a decode step costs by the rows its pass runs, measured on the
/// model as it decodes, so whatever makes a wide pass dear on this model
/// and placement counts: layers or experts on the CPU, passes a recorded
/// step would have saved, rollbacks, a slow drafter.
#[derive(Clone, Debug, Default)]
pub struct Costs {
    /// Seconds of a one-row step.
    plain: Option<f32>,
    /// What a step of two rows on costs in one-row steps, and the step it
    /// was last measured at.
    ratios: Vec<Option<(f32, usize)>>,
    steps: usize,
    /// Row counts a step has run with. The first step of each warms up
    /// what it touches, so it is not counted.
    seen: Vec<bool>,
    fixed: bool,
}

impl Costs {
    /// Costs that are never measured: a step of two rows on costs what
    /// `ratios` gives in one-row steps, and longer ones what the last does.
    pub fn fixed(ratios: &[f32]) -> Self {
        Costs {
            plain: Some(1.0),
            ratios: ratios.iter().map(|&r| Some((r, 0))).collect(),
            steps: 0,
            seen: Vec::new(),
            fixed: true,
        }
    }

    /// Counts a step of `rows` rows that took `took`.
    pub fn record(&mut self, rows: usize, took: Duration) {
        if self.seen.len() < rows {
            self.seen.resize(rows, false);
        }
        let first = !std::mem::replace(&mut self.seen[rows - 1], true);
        let t = took.as_secs_f32();
        match (rows, self.plain) {
            _ if self.fixed || first => return,
            (1, _) => follow(&mut self.plain, t),
            (_, Some(plain)) => {
                if self.ratios.len() < rows - 1 {
                    self.ratios.resize(rows - 1, None);
                }
                let mut ratio = self.ratios[rows - 2].map(|(r, _)| r);
                follow(&mut ratio, t / plain);
                self.ratios[rows - 2] = ratio.map(|r| (r, self.steps));
            }
            (_, None) => {}
        }
        self.steps += 1;
        for r in &mut self.ratios {
            if r.is_some_and(|(_, at)| self.steps - at > STALE) {
                *r = None;
            }
        }
    }

    /// What a step of `rows` rows costs in one-row steps: as measured, else
    /// on the line between the nearest measured row counts around it, else
    /// what the nearest below costs, so a longer draft is tried before it is
    /// judged.
    fn ratio(&self, rows: usize) -> f32 {
        let measured = |r: usize| match r {
            1 => Some((1, 1.0)),
            _ => self
                .ratios
                .get(r - 2)
                .copied()
                .flatten()
                .map(|(x, _)| (r, x)),
        };
        let (a, below) = (1..=rows)
            .rev()
            .find_map(measured)
            .expect("one row costs one step");
        match (rows..=self.ratios.len() + 1).find_map(measured) {
            Some((b, above)) if b > a => {
                below + (above - below) * (rows - a) as f32 / (b - a) as f32
            }
            _ => below,
        }
    }

    /// How many leading tokens of a draft whose tokens come true with the
    /// chances given, each with all before it, to verify: the number that
    /// gives the most tokens per second expected, zero when a plain step
    /// gives as many. Zero too until a plain step was measured.
    pub fn best_len(&self, chances: &[f32]) -> usize {
        if self.plain.is_none() {
            return 0;
        }
        let mut best = (0, 1.0);
        let mut expected = 1.0;
        for (k, &chance) in chances.iter().enumerate() {
            expected += chance;
            let rate = expected / self.ratio(k + 2);
            if rate > best.1 {
                best = (k + 1, rate);
            }
        }
        best.0
    }
}

fn follow(mean: &mut Option<f32>, x: f32) {
    *mean = Some(match *mean {
        None => x,
        Some(m) if x > m => m + RISE * (x - m),
        Some(m) => m + FALL * (x - m),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Costs after each step given, as rows and milliseconds, ran twice.
    fn measured(steps: &[(usize, f32)]) -> Costs {
        let mut costs = Costs::default();
        for &(rows, ms) in steps {
            for _ in 0..2 {
                costs.record(rows, Duration::from_secs_f32(ms / 1000.0));
            }
        }
        costs
    }

    #[test]
    fn a_plain_step_is_measured_first() {
        let mut costs = Costs::default();
        costs.record(1, Duration::from_millis(500));
        assert_eq!(costs.best_len(&[1.0; 8]), 0);
        costs.record(1, Duration::from_millis(10));
        costs.record(9, Duration::from_millis(500));
        assert_eq!(costs.best_len(&[0.5; 8]), 8);
    }

    #[test]
    fn rows_between_measured_ones_cost_what_the_line_gives() {
        let costs = measured(&[(1, 10.0), (3, 14.0), (9, 26.0)]);
        let near = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(near(costs.ratio(2), 1.2));
        assert!(near(costs.ratio(6), 2.0));
        assert!(near(costs.ratio(16), 2.6));
    }

    /// A pass that costs what as many plain steps cost never pays, however
    /// likely the draft; one that costs little more than a plain step pays
    /// for even an unlikely draft.
    #[test]
    fn drafts_are_verified_only_where_they_pay() {
        let linear: Vec<f32> = (2..=9).map(|r| r as f32).collect();
        assert_eq!(Costs::fixed(&linear).best_len(&[0.99; 8]), 0);
        let flat = Costs::fixed(&[1.05]);
        assert_eq!(flat.best_len(&[0.1, 0.01, 0.001]), 3);
    }

    /// Rows on the CPU make every row dear: the draft is cut where its next
    /// token's chance no longer pays for its row.
    #[test]
    fn dear_rows_cut_the_draft() {
        let split: Vec<f32> = (1..=8).map(|r| 1.0 + 0.3 * r as f32).collect();
        let chances = [0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05];
        assert_eq!(Costs::fixed(&split).best_len(&chances), 3);
        assert_eq!(Costs::fixed(&split).best_len(&[0.2; 8]), 0);
    }

    /// A step slowed by other work barely moves the costs; a run of faster
    /// steps brings them down soon.
    #[test]
    fn costs_shrug_off_a_slow_step() {
        let mut costs = measured(&[(1, 10.0), (3, 15.0)]);
        costs.record(1, Duration::from_millis(100));
        assert!(costs.plain.unwrap() < 0.02);
        costs.record(3, Duration::from_millis(150));
        assert_eq!(costs.best_len(&[0.9, 0.8]), 2);
        for _ in 0..8 {
            costs.record(3, Duration::from_millis(11));
        }
        assert!(costs.ratio(3) < 1.2);
    }

    /// A row count judged too dear is tried again once its cost is stale.
    #[test]
    fn a_dear_pass_is_tried_again_later() {
        let mut costs = measured(&[(1, 10.0), (2, 40.0)]);
        assert_eq!(costs.best_len(&[0.9]), 0);
        for _ in 0..STALE {
            costs.record(1, Duration::from_millis(10));
        }
        assert_eq!(costs.best_len(&[0.9]), 1);
    }
}
