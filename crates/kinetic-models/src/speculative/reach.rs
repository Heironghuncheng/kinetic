/// How fast a chance follows the latest drafts.
const RATE: f32 = 0.2;

/// How far drafts come true: for drafts of each length, the chance that
/// the token at each depth matches what was generated, all before it
/// included, kept apart for drafts that follow one accepted whole, as
/// copied text gives, and drafts that follow a miss. Every draft counts
/// once the tokens it guessed are known, whether or not it was verified,
/// so drafts that are not verified are still judged.
#[derive(Clone, Debug, Default)]
pub struct Reach {
    /// After a miss and after a draft accepted whole, by draft length from
    /// one on; empty until a draft of it is judged.
    chances: [Vec<Vec<f32>>; 2],
    /// Drafts not yet judged, each with the context length it continues
    /// and whether it followed a draft accepted whole.
    open: Vec<(usize, Vec<u32>, bool)>,
}

impl Reach {
    /// Notes a draft of what follows the first `at` tokens of the context,
    /// made after a draft accepted `whole` or not.
    pub fn guess(&mut self, at: usize, tokens: &[u32], whole: bool) {
        if !tokens.is_empty() {
            self.open.push((at, tokens.to_vec(), whole));
        }
    }

    /// Judges every open draft whose outcome `context` shows: one that
    /// missed, or whose tokens have all been generated.
    pub fn judge(&mut self, context: &[u32]) {
        let mut open = std::mem::take(&mut self.open);
        open.retain(|(at, draft, whole)| {
            let known = context.get(*at..).unwrap_or_default();
            let hits = draft.iter().zip(known).take_while(|(a, b)| a == b).count();
            let judged = hits < known.len().min(draft.len()) || known.len() >= draft.len();
            if judged {
                self.count(*whole, draft.len(), hits);
            }
            !judged
        });
        self.open = open;
    }

    /// Forgets drafts of a sequence that has ended.
    pub fn close(&mut self) {
        self.open.clear();
    }

    /// The chance of each depth of a draft of `len` tokens made after a
    /// draft accepted `whole` or not; sure until such a draft is judged, so
    /// it is tried.
    pub fn chances(&self, len: usize, whole: bool) -> Vec<f32> {
        match self.chances[whole as usize].get(len.wrapping_sub(1)) {
            Some(c) if !c.is_empty() => c.clone(),
            _ => vec![1.0; len],
        }
    }

    fn count(&mut self, whole: bool, len: usize, hits: usize) {
        let by_len = &mut self.chances[whole as usize];
        if by_len.len() < len {
            by_len.resize(len, Vec::new());
        }
        let chances = &mut by_len[len - 1];
        chances.resize(len, 1.0);
        for (depth, chance) in chances.iter_mut().enumerate() {
            let hit = if depth < hits { 1.0 } else { 0.0 };
            *chance += RATE * (hit - *chance);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafts_are_judged_once_their_outcome_is_known() {
        let mut reach = Reach::default();
        reach.guess(2, &[5, 6, 7], false);
        reach.judge(&[1, 2, 5, 6]);
        assert_eq!(reach.chances(3, false), [1.0; 3], "still open");
        reach.judge(&[1, 2, 5, 6, 9]);
        assert_eq!(reach.chances(3, false), [1.0, 1.0, 0.8]);
        reach.guess(5, &[4, 4, 4], false);
        reach.judge(&[1, 2, 5, 6, 9, 3]);
        assert_eq!(reach.chances(3, false), [0.8, 0.8, 0.64]);
        assert_eq!(reach.chances(2, false), [1.0; 2]);
        assert_eq!(reach.chances(3, true), [1.0; 3]);
    }

    #[test]
    fn drafts_of_each_length_and_kind_are_judged_apart() {
        let mut reach = Reach::default();
        for at in 0..20 {
            reach.guess(2 * at, &[0], false);
            reach.guess(2 * at, &[1; 4], false);
            reach.guess(2 * at, &[0; 4], true);
            reach.judge(&vec![1; 2 * at + 4]);
        }
        assert!(reach.chances(1, false)[0] < 0.02);
        assert_eq!(reach.chances(4, false), [1.0; 4]);
        assert!(reach.chances(4, true)[0] < 0.02);
    }
}
