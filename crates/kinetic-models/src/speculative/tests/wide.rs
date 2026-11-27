//! Drafts that fill passes of up to 128 rows, as a GPU verifies on paths
//! that run such a pass whole.

use super::*;

const WIDE: usize = 127;
const LONG: usize = 600;

fn plain_long<P: Policy<Toy>>(policy: &mut P) -> Vec<u32> {
    let mut state = ToyState::default();
    let mut context = PROMPT.to_vec();
    let (mut scores, _) = policy.score(&Toy, &mut state, &PROMPT, false);
    for _ in 0..LONG {
        let t = policy.choose(&Toy, &scores, 0, &context);
        context.push(t);
        scores = policy.score(&Toy, &mut state, &[t], false).0;
    }
    context.split_off(PROMPT.len())
}

/// The output and every step taken.
fn long<P: Policy<Toy>, D: Drafter<Toy>>(policy: &mut P, drafter: D) -> (Vec<u32>, Vec<Step>) {
    let mut decoder = drafting(drafter);
    decoder.max_draft = WIDE;
    let mut state = ToyState::default();
    let mut context = PROMPT.to_vec();
    decoder.prefill(&Toy, policy, &mut state, &context, 0);
    let mut steps = Vec::new();
    while context.len() - PROMPT.len() < LONG {
        let limit = LONG - (context.len() - PROMPT.len());
        steps.push(decoder.step(&Toy, policy, &mut state, &mut context, limit, |_| false));
        assert_eq!(state.tokens, context, "the state holds what was kept");
    }
    (context.split_off(PROMPT.len()), steps)
}

/// Right for `.0` tokens, then wrong.
struct MissAt(usize);

impl Drafter<Toy> for MissAt {
    fn propose(&mut self, model: &Toy, context: &[u32], limit: usize) -> Vec<u32> {
        let mut draft = Greedier.propose(model, context, limit);
        if let Some(t) = draft.get_mut(self.0) {
            *t = (*t + 1) % VOCAB as u32;
        }
        draft
    }
}

/// Drafts accepted whole double until they fill the widest pass, and every
/// one of them is kept.
#[test]
fn whole_hits_grow_drafts_to_the_widest_pass() {
    let (tokens, steps) = long(&mut Greedy, Greedier);
    assert_eq!(tokens, plain_long(&mut Greedy));
    let grown: Vec<usize> = steps.iter().map(|s| s.drafted).take(5).collect();
    assert_eq!(grown, [8, 16, 32, 64, WIDE]);
    assert!(steps.iter().all(|s| s.tokens == s.drafted + 1));
}

/// A miss deep in a wide pass keeps the rows before it and the token drawn
/// at it, and nothing of the rows after.
#[test]
fn a_miss_deep_in_a_wide_pass_rolls_back_the_rest() {
    for k in [0, 1, 40, 70, 126] {
        let (tokens, steps) = long(&mut Greedy, MissAt(k));
        assert_eq!(tokens, plain_long(&mut Greedy), "miss at {k}");
        if k >= 64 {
            assert!(
                steps.iter().any(|s| s.drafted == WIDE && s.tokens == k + 1),
                "miss at {k}: {steps:?}"
            );
        }
    }
}

#[test]
fn seeded_sampling_through_wide_passes_is_plain_sampling() {
    for seed in 0..8 {
        let expected = plain_long(&mut Sampler::new(sampling(seed)));
        for drafter in [
            Box::new(Greedier) as Box<dyn Drafter<Toy>>,
            Box::new(Counting),
        ] {
            let (tokens, _) = long(&mut Sampler::new(sampling(seed)), drafter);
            assert_eq!(tokens, expected, "seed {seed}");
        }
    }
}
