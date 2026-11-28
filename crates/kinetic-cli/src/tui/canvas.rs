//! The screen cell by cell: the night scene behind, where the terminal
//! has full colour, and lines and panels put over it.

use super::paint::{Colors, Ink, Line, Rgb, DARK};
use super::scene;

/// Inside a tile or a panel, a shade apart from the sky behind it.
pub const TILE_BG: Rgb = (27, 24, 35);

#[derive(Clone, Copy)]
enum Cell {
    /// Two pixels of the night scene.
    Scene(Rgb, Rgb),
    Text(char, Ink, Option<Rgb>),
}

/// The screen cell by cell: the scene behind, or the plain background
/// without full colour, and lines put over it.
pub struct Canvas(Vec<Vec<Cell>>);

impl Canvas {
    pub fn new(cols: usize, rows: usize, colors: Colors, tick: u64) -> Canvas {
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
    pub fn put(&mut self, row: usize, col: usize, line: &Line, bg: Option<Rgb>) {
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

    pub fn lines(&self) -> Vec<Line> {
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
