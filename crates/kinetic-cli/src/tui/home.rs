//! The home screen: the Kinetic wordmark, six tiles to move between with
//! the arrows, and the night scene under them.

use std::io;
use std::time::Duration;

use super::art::{self, ICON, ICONS};
use super::keys::Key;
use super::list::shorten;
use super::paint::{Colors, Ink, Line, Rgb, DARK};
use super::{quit, scene, Term};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    Chat,
    Models,
    Api,
    Bench,
    Doctor,
    Settings,
}

const TILES: [(Tile, &str, &str); 6] = [
    (Tile::Chat, "Chat", "Talk to a model here, in the terminal"),
    (
        Tile::Models,
        "Models",
        "Every model on this computer, and ones to download",
    ),
    (
        Tile::Api,
        "API",
        "Serve a model to apps such as OpenCode and Open WebUI",
    ),
    (
        Tile::Bench,
        "Benchmark",
        "Measure how fast a model runs on this computer",
    ),
    (
        Tile::Doctor,
        "Doctor",
        "Check this computer: memory, GPUs, drivers and disk",
    ),
    (
        Tile::Settings,
        "Settings",
        "Sampling, context, device and speed technologies",
    ),
];

const GUTTER: usize = 3;
/// Columns between the mark and the wordmark.
const GAP: usize = 3;
/// Inside a tile, a shade apart from the sky behind it.
const TILE_BG: Rgb = (27, 24, 35);
const TILE_MIN: usize = 18;
const TILE_MAX: usize = 32;
/// Tile heights from roomiest to plainest: icon, gap and name; icon and
/// name; the name alone.
const TILE_ROWS: [usize; 3] = [10, 9, 3];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Wordmark {
    Large,
    Small,
    Text,
}

impl Wordmark {
    fn size(self) -> (usize, usize) {
        match self {
            Wordmark::Large => (art::mark_width(8) + GAP + 55, 8),
            Wordmark::Small => (art::mark_width(5) + GAP + 38, 5),
            Wordmark::Text => (7, 1),
        }
    }

    /// The mark, then the wordmark to its right with "by Velocity" under
    /// it when there is room.
    fn lines(self) -> Vec<Line> {
        let (rows, mut word) = match self {
            Wordmark::Text => return vec![Line::new(Ink::Bright, "KINETIC")],
            Wordmark::Small => (5, art::wordmark("KINETIC", false)),
            Wordmark::Large => (8, art::wordmark("KINETIC", true)),
        };
        let width = word[0].width();
        if word.len() < rows {
            let mut by = Line::new(Ink::Pixels(None, None), " ".repeat(width - 11));
            by.push(Ink::Dim, "by Velocity");
            word.push(by);
        }
        let above = (rows - word.len()) / 2;
        let blank = || Line::new(Ink::Pixels(None, None), " ".repeat(width));
        let word = (0..above)
            .map(|_| blank())
            .chain(word)
            .chain(std::iter::repeat_with(blank));
        art::mark(rows)
            .into_iter()
            .zip(word)
            .map(|(mut line, word)| {
                line.push(Ink::Pixels(None, None), " ".repeat(GAP));
                line.0.extend(word.0);
                line
            })
            .collect()
    }
}

/// How the screen is laid out at a size.
#[derive(Debug, PartialEq)]
struct Plan {
    wordmark: Wordmark,
    /// Tiles per row and their width; `None` lists them one per line.
    grid: Option<(usize, usize)>,
    tile_rows: usize,
    /// Rows above the wordmark.
    top: usize,
}

fn plan(cols: usize, rows: usize) -> Plan {
    let grid = [3, 2].into_iter().find_map(|per: usize| {
        let w = cols.saturating_sub(4 + GUTTER * (per - 1)) / per;
        (w >= TILE_MIN).then_some((per, w.min(TILE_MAX)))
    });
    let grid_lines = |tile_rows: usize| match grid {
        Some((per, _)) => {
            let n = TILES.len().div_ceil(per);
            n * tile_rows + n - 1
        }
        None => TILES.len(),
    };
    // a row on top, one under the wordmark, one under the tiles, the
    // focused tile's line and the keys
    let avail = rows.saturating_sub(5);
    let heights: &[usize] = match grid {
        Some(_) => &TILE_ROWS,
        None => &[1],
    };
    let marks = [Wordmark::Large, Wordmark::Small, Wordmark::Text];
    let (wordmark, tile_rows) = heights
        .iter()
        .flat_map(|&t| marks.iter().map(move |&m| (m, t)))
        .find(|&(m, t)| m.size().0 + 2 <= cols && m.size().1 + grid_lines(t) <= avail)
        .unwrap_or((Wordmark::Text, heights[heights.len() - 1]));
    let left = avail.saturating_sub(wordmark.size().1 + grid_lines(tile_rows));
    Plan {
        wordmark,
        grid,
        tile_rows,
        top: 1 + left * 2 / 5,
    }
}

/// Row `r` of tile `i`, `width` by `rows`.
fn tile_row(i: usize, focused: bool, width: usize, rows: usize, r: usize) -> Line {
    let border = if focused { Ink::Accent } else { Ink::Muted };
    let inner = width - 2;
    if r == 0 || r + 1 == rows {
        let (a, b) = if r == 0 {
            ('╭', '╮')
        } else {
            ('╰', '╯')
        };
        return Line::new(border, format!("{a}{}{b}", "─".repeat(inner)));
    }
    let body = rows - 2;
    let icon_rows = ICON.1 / 2;
    let (icon, gap) = match body {
        8.. => (icon_rows, 1),
        7 => (icon_rows, 0),
        _ => (0, 0),
    };
    let top = (body - icon - gap - 1) / 2;
    let at = r - 1;
    let mut line = Line::new(border, "│");
    if icon > 0 && (top..top + icon).contains(&at) {
        let pad = (inner - ICON.0 / 2) / 2;
        line.push(Ink::Plain, " ".repeat(pad));
        line.0.extend(art::icon_row(&ICONS[i], at - top, focused).0);
        line.push(Ink::Plain, " ".repeat(inner - pad - ICON.0 / 2));
    } else if at == top + icon + gap {
        let name = shorten(TILES[i].1, inner);
        let n = name.chars().count();
        let pad = (inner - n) / 2;
        line.push(Ink::Plain, " ".repeat(pad));
        line.push(if focused { Ink::Accent } else { Ink::Plain }, name);
        line.push(Ink::Plain, " ".repeat(inner - pad - n));
    } else {
        line.push(Ink::Plain, " ".repeat(inner));
    }
    line.push(border, "│");
    line
}

#[derive(Clone, Copy)]
enum Cell {
    /// Two pixels of the night scene.
    Scene(Rgb, Rgb),
    Text(char, Ink, Option<Rgb>),
}

/// The screen cell by cell: the scene behind, or the plain background
/// without full colour, and lines put over it.
struct Canvas(Vec<Vec<Cell>>);

impl Canvas {
    fn new(cols: usize, rows: usize, colors: Colors, tick: u64) -> Canvas {
        match colors {
            Colors::True => Canvas(
                scene::cells(cols, rows, tick)
                    .into_iter()
                    .map(|r| r.into_iter().map(|(u, d)| Cell::Scene(u, d)).collect())
                    .collect(),
            ),
            _ => Canvas(vec![vec![Cell::Text(' ', Ink::Plain, None); cols]; rows]),
        }
    }

    /// Puts `line` at `row` from column `col`: over `bg` when given, else
    /// over what is there, which shows around the text.
    fn put(&mut self, row: usize, col: usize, line: &Line, bg: Option<Rgb>) {
        let Some(cells) = self.0.get_mut(row) else {
            return;
        };
        // a span's spaces between its words hide the scene, the rest show it
        let chars = line.0.iter().flat_map(|s| {
            let text: Vec<char> = s.text.chars().collect();
            let first = text.iter().position(|&c| c != ' ').unwrap_or(text.len());
            let last = text.iter().rposition(|&c| c != ' ').unwrap_or(0);
            let pixels = matches!(s.ink, Ink::Pixels(..));
            text.into_iter().enumerate().map(move |(i, c)| {
                let shown = c != ' ' || (!pixels && (first..=last).contains(&i));
                (c, s.ink, shown)
            })
        });
        for ((ch, ink, shown), cell) in chars.zip(cells.iter_mut().skip(col)) {
            let under = match *cell {
                Cell::Scene(u, d) => Some(mix(u, d)),
                Cell::Text(_, _, b) => b,
            };
            if bg.is_some() || shown {
                // text right on the scene gets a darker patch of it behind
                let back = match ink {
                    Ink::Pixels(..) => under,
                    _ => under.map(|u| mix(u, DARK)),
                };
                *cell = Cell::Text(ch, ink, bg.or(back));
            }
        }
    }

    fn lines(&self) -> Vec<Line> {
        self.0
            .iter()
            .map(|cells| {
                let mut line = Line::default();
                for cell in cells {
                    match *cell {
                        Cell::Scene(u, d) => {
                            let c = if u == d { '█' } else { '▀' };
                            line.push_on(Ink::Pixels(Some(u), Some(d)), c, None)
                        }
                        Cell::Text(c, ink, bg) => line.push_on(ink, c, bg),
                    };
                }
                line
            })
            .collect()
    }
}

fn mix(a: Rgb, b: Rgb) -> Rgb {
    let m = |x: u8, y: u8| ((x as u16 + y as u16) / 2) as u8;
    (m(a.0, b.0), m(a.1, b.1), m(a.2, b.2))
}

/// The whole screen at `cols` by `rows`, tile `focus` lit.
fn layout(
    cols: usize,
    rows: usize,
    focus: usize,
    colors: Colors,
    badge: Option<&str>,
    tick: u64,
) -> Vec<Line> {
    let plan = plan(cols, rows);
    let mut canvas = Canvas::new(cols, rows, colors, tick);
    let centre = |line: &Line| cols.saturating_sub(line.width()) / 2;
    let mut y = plan.top;
    let mark = plan.wordmark.lines();
    let x = cols.saturating_sub(mark.iter().map(Line::width).max().unwrap_or(0)) / 2;
    for line in &mark {
        canvas.put(y, x, line, None);
        y += 1;
    }
    y += 1;
    match plan.grid {
        Some((per, width)) => {
            let left = (cols - per * width - GUTTER * (per - 1)) / 2;
            for i in 0..TILES.len() {
                let tx = left + (i % per) * (width + GUTTER);
                let ty = y + (i / per) * (plan.tile_rows + 1);
                for r in 0..plan.tile_rows {
                    let row = tile_row(i, i == focus, width, plan.tile_rows, r);
                    canvas.put(ty + r, tx, &row, Some(TILE_BG));
                }
            }
            let n = TILES.len().div_ceil(per);
            y += n * plan.tile_rows + n - 1;
        }
        None => {
            for (i, (_, name, _)) in TILES.iter().enumerate() {
                let mark = if i == focus { "›" } else { " " };
                let text = format!("{:<1$}", format!(" {mark} {name}"), cols.saturating_sub(4));
                let ink = if i == focus { Ink::Accent } else { Ink::Plain };
                canvas.put(y, 2, &Line::new(ink, text), Some(TILE_BG));
                y += 1;
            }
        }
    }
    y += 1;
    let what = Line::new(Ink::Dim, shorten(TILES[focus].2, cols.saturating_sub(2)));
    canvas.put(y, centre(&what), &what, None);
    let keys = Line::new(Ink::Dim, shorten("←↑↓→ move   Enter open   q quit", cols));
    canvas.put(rows.saturating_sub(1), centre(&keys), &keys, None);
    if let Some(badge) = badge {
        let badge = Line::new(Ink::Accent, shorten(badge, cols.saturating_sub(2)));
        canvas.put(0, cols.saturating_sub(badge.width() + 2), &badge, None);
    }
    canvas.lines()
}

/// Shows the home screen until a tile is opened; an `Interrupted` error
/// when the user quits.
pub fn show(term: &mut Term, focus: &mut usize, badge: Option<&str>) -> io::Result<Tile> {
    let mut tick = 0;
    loop {
        let (cols, rows) = term.size();
        let colors = term.colors;
        term.draw(&layout(cols, rows, *focus, colors, badge, tick))?;
        let per = plan(cols, rows).grid.map_or(1, |(per, _)| per);
        // the stars twinkle only where there is a scene
        let wait = match colors {
            Colors::True => Duration::from_millis(600),
            _ => Duration::from_secs(3600),
        };
        let Some(key) = term.poll(wait)? else {
            tick += 1;
            continue;
        };
        let last = TILES.len() - 1;
        match key {
            Key::CtrlC | Key::Esc | Key::Char('q' | 'Q') => return Err(quit()),
            Key::Enter | Key::Char(' ') => return Ok(TILES[*focus].0),
            Key::Left => *focus = focus.saturating_sub(1),
            Key::Right | Key::Tab => *focus = (*focus + 1).min(last),
            Key::Up if per == 1 => *focus = focus.saturating_sub(1),
            Key::Down if per == 1 => *focus = (*focus + 1).min(last),
            Key::Up => *focus = focus.saturating_sub(per),
            Key::Down if *focus + per <= last => *focus += per,
            Key::Home => *focus = 0,
            Key::End => *focus = last,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZES: [(usize, usize); 7] = [
        (80, 24),
        (100, 30),
        (120, 40),
        (200, 60),
        (300, 80),
        (60, 40),
        (40, 12),
    ];

    #[test]
    fn every_size_fills_the_window_and_shows_every_tile() {
        for colors in [Colors::True, Colors::Sixteen, Colors::None] {
            for (cols, rows) in SIZES {
                let lines = layout(cols, rows, 1, colors, Some("Kinetic 0.1.2 is out"), 0);
                assert_eq!(lines.len(), rows, "{cols}x{rows}");
                for l in &lines {
                    assert_eq!(l.width(), cols, "{cols}x{rows}");
                }
                let text: String = lines
                    .iter()
                    .flat_map(|l| &l.0)
                    .map(|s| s.text.as_str())
                    .collect();
                for (_, name, _) in &TILES {
                    assert!(text.contains(name), "{cols}x{rows} lacks {name}");
                }
                assert!(text.contains("Enter open"), "{cols}x{rows}");
                assert!(text.contains("0.1.2 is out"), "{cols}x{rows}");
            }
        }
    }

    #[test]
    fn every_cell_of_a_tile_has_the_tile_background() {
        let (cols, rows) = (120, 40);
        let p = plan(cols, rows);
        let (per, width) = p.grid.unwrap();
        let left = (cols - per * width - GUTTER * (per - 1)) / 2;
        let top = p.top + p.wordmark.size().1 + 1;
        let lines = layout(cols, rows, 0, Colors::True, None, 0);
        for line in &lines[top..top + p.tile_rows] {
            let mut x = 0;
            for span in &line.0 {
                let n = span.text.chars().count();
                let inside = (x..x + n).any(|c| c >= left && c < left + width);
                if inside {
                    assert_eq!(span.bg, Some(TILE_BG), "column {x}");
                }
                x += n;
            }
        }
    }

    #[test]
    fn the_layout_steps_down_as_the_window_shrinks() {
        assert_eq!(plan(300, 80).wordmark, Wordmark::Large);
        assert_eq!(plan(300, 80).grid, Some((3, TILE_MAX)));
        assert_eq!(plan(300, 80).tile_rows, 10);
        assert_eq!(plan(80, 24).grid.map(|g| g.0), Some(3));
        assert_eq!(plan(60, 40).grid.map(|g| g.0), Some(2));
        assert_eq!(plan(40, 12).grid, None);
    }
}
