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
    /// The newer release the badge names.
    Update,
    /// Every command, under the tiles.
    More,
}

const MORE: (&str, &str) = (
    "Every command",
    "run, pull, fit, info, quantize, perplexity and the rest, with all their options",
);

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
    fn lines(self, mono: bool) -> Vec<Line> {
        let (rows, mut word) = match self {
            Wordmark::Text => return vec![Line::new(Ink::Bright, "KINETIC")],
            Wordmark::Small => (5, art::wordmark("KINETIC", false, !mono)),
            Wordmark::Large => (8, art::wordmark("KINETIC", true, !mono)),
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
        // the mark is its colours; without them, the letters alone
        let mark = match mono {
            true => (0..rows).map(|_| Line::default()).collect(),
            false => art::mark(rows),
        };
        mark.into_iter()
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
    // button for every command, the focused tile's line and the keys
    let avail = rows.saturating_sub(6);
    // a bigger wordmark first, then taller tiles, then the plainest
    let tiles: &[usize] = match grid {
        Some(_) => &TILE_ROWS,
        None => &[1],
    };
    let (roomy, plain) = tiles.split_at(tiles.len() - 1);
    let marks = [Wordmark::Large, Wordmark::Small, Wordmark::Text];
    let mut tried = marks
        .iter()
        .flat_map(|&m| roomy.iter().map(move |&t| (m, t)))
        .chain(marks.iter().map(|&m| (m, plain[0])));
    let (wordmark, tile_rows) = tried
        .find(|&(m, t)| m.size().0 + 2 <= cols && m.size().1 + grid_lines(t) <= avail)
        .unwrap_or((Wordmark::Text, plain[0]));
    let left = avail.saturating_sub(wordmark.size().1 + grid_lines(tile_rows));
    Plan {
        wordmark,
        grid,
        tile_rows,
        top: 1 + left * 2 / 5,
    }
}

/// Row `r` of tile `i`, `width` by `rows`.
fn tile_row(i: usize, focused: bool, width: usize, rows: usize, r: usize, mono: bool) -> Line {
    let border = if focused { Ink::Accent } else { Ink::Muted };
    let inner = width - 2;
    if r == 0 || r + 1 == rows {
        let (a, b, line) = match (r == 0, focused) {
            (true, false) => ('╭', '╮', '─'),
            (false, false) => ('╰', '╯', '─'),
            (true, true) => ('┏', '┓', '━'),
            (false, true) => ('┗', '┛', '━'),
        };
        return Line::new(border, format!("{a}{}{b}", line.to_string().repeat(inner)));
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
    let side = if focused { "┃" } else { "│" };
    let mut line = Line::new(border, side);
    if icon > 0 && (top..top + icon).contains(&at) {
        let pad = (inner - ICON.0 / 2) / 2;
        line.push(Ink::Plain, " ".repeat(pad));
        line.0
            .extend(art::icon_row(&ICONS[i], at - top, focused, mono).0);
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
    line.push(border, side);
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
    asking: bool,
) -> Vec<Line> {
    let plan = plan(cols, rows);
    let mut canvas = Canvas::new(cols, rows, colors, tick);
    let centre = |line: &Line| cols.saturating_sub(line.width()) / 2;
    let mut y = plan.top;
    let mono = colors == Colors::None;
    let mark = plan.wordmark.lines(mono);
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
                    let row = tile_row(i, i == focus, width, plan.tile_rows, r, mono);
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
    let more = format!("  {} ›  ", MORE.0);
    let button = match focus == TILES.len() {
        true => Line::new(Ink::Accent, more),
        false => Line::new(Ink::Plain, more),
    };
    canvas.put(y, centre(&button), &button, Some(TILE_BG));
    y += 1;
    let what = TILES.get(focus).map_or(MORE.1, |t| t.2);
    let what = Line::new(Ink::Dim, shorten(what, cols.saturating_sub(2)));
    canvas.put(y, centre(&what), &what, None);
    let keys = Line::new(
        Ink::Dim,
        shorten("←↑↓→ move   Enter open   Esc or q quit", cols),
    );
    canvas.put(rows.saturating_sub(1), centre(&keys), &keys, None);
    if let Some(badge) = badge {
        let badge = Line::new(Ink::Accent, shorten(badge, cols.saturating_sub(2)));
        canvas.put(0, cols.saturating_sub(badge.width() + 2), &badge, None);
    }
    if asking {
        let width = 32.min(cols);
        let inner = width.saturating_sub(2);
        let top = rows.saturating_sub(7) / 2;
        let x = cols.saturating_sub(width) / 2;
        let text = ["", "Quit Kinetic?", "", "Enter quit    Esc stay", ""];
        let edge =
            |l: char, r: char| Line::new(Ink::Accent, format!("{l}{}{r}", "─".repeat(inner)));
        canvas.put(top, x, &edge('╭', '╮'), Some(TILE_BG));
        for (i, t) in text.into_iter().enumerate() {
            let pad = inner.saturating_sub(t.chars().count()) / 2;
            let ink = if i == 1 { Ink::Bright } else { Ink::Dim };
            let mut line = Line::new(Ink::Accent, "│");
            line.push(
                ink,
                format!("{:<1$}", format!("{}{t}", " ".repeat(pad)), inner),
            );
            line.push(Ink::Accent, "│");
            canvas.put(top + 1 + i, x, &line, Some(TILE_BG));
        }
        canvas.put(top + 6, x, &edge('╰', '╯'), Some(TILE_BG));
    }
    canvas.lines()
}

/// Shows the home screen until a tile is opened; an `Interrupted` error
/// when the user quits.
pub fn show(term: &mut Term, focus: &mut usize, badge: Option<&str>) -> io::Result<Tile> {
    let mut tick = 0;
    let mut asking = false;
    loop {
        let (cols, rows) = term.size();
        let colors = term.colors;
        term.draw(&layout(cols, rows, *focus, colors, badge, tick, asking))?;
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
        let last = TILES.len();
        if asking {
            match key {
                Key::Enter | Key::CtrlC | Key::Char('q' | 'Q' | 'y' | 'Y') => return Err(quit()),
                Key::Resize => {}
                _ => asking = false,
            }
            continue;
        }
        match key {
            Key::CtrlC | Key::Char('q' | 'Q') => return Err(quit()),
            Key::Esc => asking = true,
            Key::Char('u' | 'U') if badge.is_some() => return Ok(Tile::Update),
            Key::Enter | Key::Char(' ') => {
                return Ok(TILES.get(*focus).map_or(Tile::More, |t| t.0))
            }
            Key::Left => *focus = focus.saturating_sub(1),
            Key::Right | Key::Tab => *focus = (*focus + 1).min(last),
            Key::Up if per == 1 => *focus = focus.saturating_sub(1),
            Key::Down if per == 1 => *focus = (*focus + 1).min(last),
            Key::Up if *focus == last => *focus = last - per + (per - 1) / 2,
            Key::Up => *focus = focus.saturating_sub(per),
            Key::Down if *focus + per < last => *focus += per,
            Key::Down => *focus = last,
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
                let lines = layout(
                    cols,
                    rows,
                    1,
                    colors,
                    Some("Kinetic 0.1.2 is out"),
                    0,
                    false,
                );
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
                assert!(text.contains("Every command"), "{cols}x{rows}");
                assert!(text.contains("0.1.2 is out"), "{cols}x{rows}");
                let asked = layout(cols, rows, 1, colors, None, 0, true);
                let asked: String = asked
                    .iter()
                    .flat_map(|l| &l.0)
                    .map(|s| s.text.as_str())
                    .collect();
                assert!(asked.contains("Quit Kinetic?"), "{cols}x{rows}");
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
        let lines = layout(cols, rows, 0, Colors::True, None, 0, false);
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
    fn each_tile_draws_its_icon_once() {
        let cells = |l: &Line| -> Vec<(char, Ink)> {
            l.0.iter()
                .flat_map(|s| s.text.chars().map(move |c| (c, s.ink)))
                .collect()
        };
        for (cols, rows) in SIZES {
            let screen: Vec<_> = layout(cols, rows, 0, Colors::True, None, 0, false)
                .iter()
                .map(cells)
                .collect();
            let shown = usize::from(plan(cols, rows).tile_rows >= 9);
            for (i, icon) in ICONS.iter().enumerate() {
                for r in 0..ICON.1 / 2 {
                    let row = cells(&art::icon_row(icon, r, i == 0, false));
                    let found: usize = screen
                        .iter()
                        .map(|l| {
                            l.windows(row.len())
                                .filter(|w| *w == row.as_slice())
                                .count()
                        })
                        .sum();
                    assert_eq!(found, shown, "{cols}x{rows} tile {i} row {r}");
                }
            }
        }
    }

    #[test]
    fn the_layout_steps_down_as_the_window_shrinks() {
        assert_eq!(plan(300, 80).wordmark, Wordmark::Large);
        assert_eq!(plan(300, 80).grid, Some((3, TILE_MAX)));
        assert_eq!(plan(300, 80).tile_rows, 10);
        assert_eq!(plan(80, 24).grid.map(|g| g.0), Some(3));
        assert_eq!(plan(100, 30).wordmark, Wordmark::Small);
        assert_eq!(plan(100, 30).tile_rows, 9);
        assert_eq!(plan(60, 40).grid.map(|g| g.0), Some(2));
        assert_eq!(plan(40, 12).grid, None);
    }
}
