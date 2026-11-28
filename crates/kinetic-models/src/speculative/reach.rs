/// How fast a chance follows the latest drafts.
const RATE: f32 = 0.2;

/// How far drafts come true: for drafts of each length, the chance that
/// the token at each depth matches what was generated, all before it
/// included, kept apart for drafts that follow one accepted whole, as
/// copied text gives, and drafts that follow a miss. Every draft counts,
/// whether or not it was verified, each depth as soon as the token there is
/// known, so drafts are judged for nothing while none is verified.
#[derive(Clone, Debug, Default)]
pub struct Reach {
    /// After a miss and after a draft accepted whole, by draft length from
    /// one on, then by depth; `None` until a token at that depth is known.
    chances: [Vec<Vec<Option<f32>>>; 2],
    open: Vec<Open>,
}

/// A draft not yet judged to its end.
#[derive(Clone, Debug)]
struct Open {
    /// The context length it continues.
    at: usize,
    tokens: Vec<u32>,
    /// Whether it followed a draft accepted whole.
    whole: bool,
    /// Depths counted so far.
    counted: usize,
}

impl Reach {
    /// Notes a draft of what follows the first `at` tokens of the context,
    /// made after a draft accepted `whole` or not.
    pub fn guess(&mut self, at: usize, tokens: &[u32], whole: bool) {
        if !tokens.is_empty() {
            self.open.push(Open {
                at,
                tokens: tokens.to_vec(),
                whole,
                counted: 0,
            });
        }
    }

    /// Counts every depth of the open drafts that `context` now shows: up to
    /// the tokens generated after each, or all of a draft once it missed.
    pub fn judge(&mut self, context: &[u32]) {
        let mut open = std::mem::take(&mut self.open);
        for d in &mut open {
            let known = context.get(d.at..).unwrap_or_default();
            let len = d.tokens.len();
            let hits = d
                .tokens
                .iter()
                .zip(known)
                .take_while(|(a, b)| a == b)
                .count();
            let upto = match hits < known.len().min(len) {
                true => len,
                false => known.len().min(len),
            };
            for depth in d.counted..upto {
                self.count(d.whole, len, depth, depth < hits);
            }
            d.counted = d.counted.max(upto);
        }
        open.retain(|d| d.counted < d.tokens.len());
        self.open = open;
    }

    /// Forgets drafts of a sequence that has ended.
    pub fn close(&mut self) {
        self.open.clear();
    }

    /// The chance of each depth of a draft of `len` tokens made after a
    /// draft accepted `whole` or not. A length no draft of that kind was
    /// judged at takes the nearest judged length's chances, of the other
    /// kind if none, and a depth not yet known the chance of the one before
    /// it. With nothing judged every chance is zero: drafts are judged
    /// before any is verified.
    pub fn chances(&self, len: usize, whole: bool) -> Vec<f32> {
        let nearest = |kind: bool| {
            let rows = &self.chances[kind as usize];
            (0..rows.len())
                .filter(|&i| rows[i].first().copied().flatten().is_some())
                .min_by_key(|&i| (i + 1).abs_diff(len))
                .map(|i| &rows[i])
        };
        let Some(row) = nearest(whole).or_else(|| nearest(!whole)) else {
            return vec![0.0; len];
        };
        let mut last = 0.0;
        (0..len)
            .map(|depth| {
                last = row.get(depth).copied().flatten().unwrap_or(last);
                last
            })
            .collect()
    }

    fn count(&mut self, whole: bool, len: usize, depth: usize, hit: bool) {
        let by_len = &mut self.chances[whole as usize];
        if by_len.len() < len {
            by_len.resize(len, Vec::new());
        }
        let row = &mut by_len[len - 1];
        row.resize(len, None);
        let x = if hit { 1.0 } else { 0.0 };
        row[depth] = Some(row[depth].map_or(x, |c| c + RATE * (x - c)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depths_are_judged_as_their_tokens_are_known() {
        let mut reach = Reach::default();
        assert_eq!(reach.chances(3, false), [0.0; 3], "nothing judged yet");
        reach.guess(2, &[5, 6, 7], false);
        reach.judge(&[1, 2, 5]);
        assert_eq!(reach.chances(3, false), [1.0; 3]);
        reach.judge(&[1, 2, 5, 6, 9]);
        assert_eq!(reach.chances(3, false), [1.0, 1.0, 0.0]);
        reach.guess(5, &[4, 4, 4], false);
        reach.judge(&[1, 2, 5, 6, 9, 3]);
        assert_eq!(reach.chances(3, false), [0.8, 0.8, 0.0]);
        assert_eq!(reach.chances(2, false), [0.8, 0.8], "the nearest length");
        assert_eq!(
            reach.chances(4, true),
            [0.8, 0.8, 0.0, 0.0],
            "the other kind"
        );
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
        assert_eq!(reach.chances(1, false), [0.0]);
        assert_eq!(reach.chances(4, false), [1.0; 4]);
        assert_eq!(reach.chances(4, true), [0.0; 4]);
    }
}
