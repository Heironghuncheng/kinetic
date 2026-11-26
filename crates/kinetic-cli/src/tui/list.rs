//! A list to move through with the arrows, narrowed by typing.

use super::keys::Key;
use super::paint::{Ink, Line};
use crate::fuzzy;

pub struct Item {
    /// Columns, lined up with the same columns of the other items.
    pub cells: Vec<String>,
    /// What typing is matched against.
    pub key: String,
    pub ink: Ink,
    /// Shown whatever is typed.
    pub pinned: bool,
}

impl Item {
    pub fn new(cells: Vec<String>, ink: Ink) -> Item {
        Item {
            key: cells.first().cloned().unwrap_or_default(),
            cells,
            ink,
            pinned: false,
        }
    }
}

#[derive(Default)]
pub struct List {
    pub items: Vec<Item>,
    pub filter: String,
    /// Position of the selection among the items shown.
    at: usize,
    top: usize,
}

impl List {
    pub fn new(items: Vec<Item>) -> List {
        List {
            items,
            ..List::default()
        }
    }

    /// The items the filter lets through: all in order when it is empty,
    /// else the closest first, pinned items last.
    pub fn shown(&self) -> Vec<usize> {
        if self.filter.trim().is_empty() {
            return (0..self.items.len()).collect();
        }
        let mut hits: Vec<(usize, usize)> = (0..self.items.len())
            .filter(|&i| !self.items[i].pinned)
            .filter_map(|i| Some((fuzzy::score(&self.filter, &self.items[i].key)?, i)))
            .collect();
        hits.sort();
        let pinned = (0..self.items.len()).filter(|&i| self.items[i].pinned);
        hits.into_iter().map(|(_, i)| i).chain(pinned).collect()
    }

    /// The selected item.
    pub fn selected(&self) -> Option<usize> {
        self.shown().get(self.at).copied()
    }

    pub fn select(&mut self, item: usize) {
        if let Some(at) = self.shown().iter().position(|&i| i == item) {
            self.at = at;
        }
    }

    /// Moves or types for `key`; false when the key is not the list's.
    pub fn handle(&mut self, key: Key) -> bool {
        let last = self.shown().len().saturating_sub(1);
        match key {
            Key::Up => self.at = self.at.saturating_sub(1),
            Key::Down => self.at = (self.at + 1).min(last),
            Key::PageUp => self.at = self.at.saturating_sub(10),
            Key::PageDown => self.at = (self.at + 10).min(last),
            Key::Home => self.at = 0,
            Key::End => self.at = last,
            Key::Char(c) => {
                self.filter.push(c);
                self.at = 0;
            }
            Key::Backspace if !self.filter.is_empty() => {
                self.filter.pop();
                self.at = 0;
            }
            Key::Esc if !self.filter.is_empty() => {
                self.filter.clear();
                self.at = 0;
            }
            _ => return false,
        }
        true
    }

    /// The rows that fit in `height` by `cols`, the selection marked and
    /// kept in view; the first column gives way when they are too wide, and
    /// the last is cut at the edge.
    pub fn lines(&mut self, height: usize, cols: usize) -> Vec<Line> {
        let shown = self.shown();
        let height = height.max(1);
        if self.at < self.top {
            self.top = self.at;
        } else if self.at >= self.top + height {
            self.top = self.at + 1 - height;
        }
        self.top = self.top.min(shown.len().saturating_sub(height));
        let columns = self.items.iter().map(|i| i.cells.len()).max().unwrap_or(0);
        let mut widths: Vec<usize> = (0..columns)
            .map(|c| {
                self.items
                    .iter()
                    .filter_map(|i| i.cells.get(c))
                    .map(|s| s.chars().count())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        // the last column is cut at the edge; the ones between keep their width
        let between: usize = widths
            .iter()
            .skip(1)
            .take(columns.saturating_sub(2))
            .map(|w| w + GAP.len())
            .sum();
        let last = widths.last().map_or(0, |w| (*w).min(MIN_NAME));
        let rest = between + if columns > 1 { GAP.len() + last } else { 0 };
        if let Some(first) = widths.first_mut() {
            *first = (*first).min(cols.saturating_sub(MARGIN + rest).max(MIN_NAME));
        }
        shown
            .iter()
            .enumerate()
            .skip(self.top)
            .take(height)
            .map(|(at, &i)| {
                let item = &self.items[i];
                let picked = at == self.at;
                let text: Vec<String> = item
                    .cells
                    .iter()
                    .zip(&widths)
                    .map(|(cell, &w)| format!("{:<w$}", shorten(cell, w)))
                    .collect();
                let text = text.join(GAP);
                match picked {
                    true => Line::new(Ink::Accent, format!("  › {}", text.trim_end())),
                    false => Line::new(item.ink, format!("    {}", text.trim_end())),
                }
            })
            .collect()
    }
}

const GAP: &str = "   ";
/// Columns before the first cell: the indent and the selection mark.
const MARGIN: usize = 4;
/// The first column never shrinks below this.
const MIN_NAME: usize = 12;

/// `text` in at most `width` characters, its middle left out for `…`, so
/// both the start and the quantization at the end of a name stay.
pub fn shorten(text: &str, width: usize) -> String {
    let n = text.chars().count();
    if n <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    let tail = (width - 1) / 2;
    let head = width - 1 - tail;
    let start: String = text.chars().take(head).collect();
    let end: String = text.chars().skip(n - tail).collect();
    format!("{start}…{end}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_names_lose_their_middle() {
        assert_eq!(shorten("unsloth/Qwen3.5-4B-MTP-GGUF", 11), "unslo…-GGUF");
        assert_eq!(shorten("short", 11), "short");
        assert_eq!(shorten("abc", 1), "…");
    }

    #[test]
    fn columns_fit_the_width_by_shortening_the_first() {
        let mut l = List::new(vec![Item::new(
            vec!["a-very-long-model-name-Q4_K_M".into(), "2.50 GiB".into()],
            Ink::Plain,
        )]);
        let line = &l.lines(5, 30)[0];
        assert!(line.width() <= 30, "{}", line.0[0].text);
        assert!(line.0[0].text.contains("…"));
        assert!(line.0[0].text.ends_with("2.50 GiB"));
    }

    fn list(names: &[&str]) -> List {
        let mut items: Vec<Item> = names
            .iter()
            .map(|n| Item::new(vec![n.to_string()], Ink::Plain))
            .collect();
        items.last_mut().unwrap().pinned = true;
        List::new(items)
    }

    #[test]
    fn typing_narrows_to_the_closest_and_keeps_pinned_rows() {
        let mut l = list(&[
            "LFM3 1.7B",
            "Qwen3 4B",
            "qwen3.5 4b mtp",
            "Browse a folder...",
        ]);
        assert_eq!(l.shown(), [0, 1, 2, 3]);
        for c in "qwen4".chars() {
            l.handle(Key::Char(c));
        }
        assert_eq!(l.shown(), [1, 2, 3]);
        assert_eq!(l.selected(), Some(1));
        l.handle(Key::Down);
        assert_eq!(l.selected(), Some(2));
        l.handle(Key::Esc);
        assert_eq!(l.shown(), [0, 1, 2, 3]);
        assert!(!l.handle(Key::Esc));
    }

    #[test]
    fn the_selection_stays_inside_the_list_and_in_view() {
        let mut l = list(&["a", "b", "c", "d", "e"]);
        l.handle(Key::End);
        assert_eq!(l.selected(), Some(4));
        let rows = l.lines(2, 80);
        assert_eq!(rows.len(), 2);
        assert!(rows[1].0[0].text.contains("› e"));
        l.handle(Key::PageUp);
        assert_eq!(l.selected(), Some(0));
        assert!(l.lines(2, 80)[0].0[0].text.contains("› a"));
    }
}
