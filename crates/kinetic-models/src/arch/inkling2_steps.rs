//! The order in which an Inkling 2 network runs its layers, for devices that
//! run it step by step: which physical layer runs on which state slot, and
//! where the loop's passes start. The host path in `inkling2.rs` follows
//! the same order.

use super::inkling2::Network;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// Physical layer `layer` on state slot `slot`; `looped` in the passes
    /// after the first of the core, which take messages from the bank.
    Layer {
        layer: usize,
        slot: usize,
        looped: bool,
    },
    /// Before the core: keep the stream and how many blocks there are.
    Prelude,
    /// Before a later pass of the core: the blocks after the first of the
    /// core, then the stream, become the bank the pass's messages mix; the
    /// stream and the blocks go back to what [`Step::Prelude`] kept.
    Pass,
}

impl Network<'_> {
    /// Every step of a run, in order.
    pub(crate) fn steps(&self) -> Vec<Step> {
        let n = self.wiring.len();
        let layer = |layer, slot, looped| Step::Layer {
            layer,
            slot,
            looped,
        };
        let Some(c) = &self.core else {
            return (0..n).map(|i| layer(i, i, false)).collect();
        };
        let mut steps: Vec<Step> = (0..c.start).map(|i| layer(i, i, false)).collect();
        steps.push(Step::Prelude);
        steps.extend((c.start..c.end).map(|i| layer(i, i, false)));
        for pass in 1..c.passes {
            steps.push(Step::Pass);
            let first = n + (pass - 1) * (c.end - c.start);
            steps.extend((c.start..c.end).map(|i| layer(i, first + i - c.start, true)));
        }
        steps.extend((c.end..n).map(|i| layer(i, i, false)));
        steps
    }

    /// Whether `layer` starts a block, whose input then joins the blocks
    /// later layers mix.
    pub(crate) fn starts_block(&self, layer: usize) -> bool {
        layer.is_multiple_of(self.block)
    }

    /// Most blocks a mix reads besides the stream, and most a bank holds.
    #[cfg(feature = "vulkan")]
    pub(crate) fn most_blocks(&self) -> (usize, usize) {
        let blocks = self.wiring.len().div_ceil(self.block);
        let bank = self
            .core
            .as_ref()
            .map_or(0, |c| (c.end - c.start) / self.block);
        (blocks, bank)
    }
}
