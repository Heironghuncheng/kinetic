//! Lines of coloured text, cut to the window, in the Kinetic colours where
//! the terminal has them and in its sixteen colours where it does not.

use std::io::IsTerminal;

use unicode_width::UnicodeWidthChar;

pub type Rgb = (u8, u8, u8);

pub const ORANGE: Rgb = (221, 138, 74);
pub const PURPLE: Rgb = (122, 102, 136);
pub const LAVENDER: Rgb = (220, 213, 226);
pub const GREY: Rgb = (150, 141, 160);
pub const WHITE: Rgb = (241, 239, 243);
pub const DARK: Rgb = (28, 26, 31);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Colors {
    None,
    Sixteen,
    True,
}

impl Colors {
    pub fn detect() -> Colors {
        if std::env::var_os("NO_COLOR").is_some() || !std::io::stdout().is_terminal() {
            return Colors::None;
        }
        let truecolor = cfg!(windows)
            || std::env::var("COLORTERM").is_ok_and(|v| v == "truecolor" || v == "24bit")
            || std::env::var_os("WT_SESSION").is_some();
        match truecolor {
            true => Colors::True,
            false => Colors::Sixteen,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Ink {
    Plain,
    Dim,
    Bright,
    Accent,
    Muted,
    /// A cell of two pixels, the upper and the lower.
    Pixels(Option<Rgb>, Option<Rgb>),
}

pub struct Span {
    pub ink: Ink,
    pub text: String,
    /// The background in full colour; the window's when `None`.
    pub bg: Option<Rgb>,
}

#[derive(Default)]
pub struct Line(pub Vec<Span>);

impl Line {
    pub fn new(ink: Ink, text: impl Into<String>) -> Line {
        let mut line = Line::default();
        line.push(ink, text);
        line
    }

    pub fn push(&mut self, ink: Ink, text: impl Into<String>) -> &mut Line {
        self.push_on(ink, text, None)
    }

    /// Adds `text` in `ink` over `bg`, to the last span when it looks the
    /// same.
    pub fn push_on(&mut self, ink: Ink, text: impl Into<String>, bg: Option<Rgb>) -> &mut Line {
        match self.0.last_mut() {
            Some(last) if last.ink == ink && last.bg == bg => last.text += &text.into(),
            _ => self.0.push(Span {
                ink,
                text: text.into(),
                bg,
            }),
        }
        self
    }

    /// The line cut to at most `width` columns.
    pub fn cut(&self, width: usize) -> Line {
        let mut line = Line::default();
        let mut used = 0;
        for span in &self.0 {
            for c in span.text.chars() {
                used += c.width().unwrap_or(0);
                if used > width {
                    return line;
                }
                line.push_on(span.ink, c, span.bg);
            }
        }
        line
    }

    pub fn width(&self) -> usize {
        self.0
            .iter()
            .flat_map(|s| s.text.chars())
            .map(|c| c.width().unwrap_or(0))
            .sum()
    }

    /// The line as the terminal takes it, at most `cols` wide.
    pub fn render(&self, cols: usize, colors: Colors) -> String {
        let mut out = String::new();
        let mut used = 0;
        for span in &self.0 {
            out += &start(span.ink, span.bg, colors);
            for c in span.text.chars() {
                let w = c.width().unwrap_or(0);
                if used + w > cols {
                    return out;
                }
                used += w;
                out.push(c);
            }
        }
        out += &start(Ink::Plain, None, colors);
        out
    }
}

/// The escape that sets `ink` over `back`, or the window's background, in
/// full colour; clear pixels take the background.
pub fn start(ink: Ink, back: Option<Rgb>, colors: Colors) -> String {
    let back = back.unwrap_or(DARK);
    let fg = |(r, g, b): Rgb| format!("38;2;{r};{g};{b}");
    let bg = |(r, g, b): Rgb| format!("48;2;{r};{g};{b}");
    match colors {
        Colors::None => String::new(),
        Colors::True => {
            let code = match ink {
                Ink::Plain => fg(LAVENDER),
                Ink::Dim => fg(GREY),
                Ink::Bright => format!("1;{}", fg(WHITE)),
                Ink::Accent => format!("1;{}", fg(ORANGE)),
                Ink::Muted => fg(PURPLE),
                Ink::Pixels(up, down) => {
                    format!("{};{}", fg(up.unwrap_or(back)), bg(down.unwrap_or(back)))
                }
            };
            let back = match ink {
                Ink::Pixels(..) => String::new(),
                _ => format!(";{}", bg(back)),
            };
            format!("\x1b[0;{code}{back}m")
        }
        Colors::Sixteen => {
            let code = match ink {
                Ink::Plain => "37".to_owned(),
                Ink::Dim => "90".to_owned(),
                Ink::Bright => "1;97".to_owned(),
                Ink::Accent => "1;33".to_owned(),
                Ink::Muted => "35".to_owned(),
                Ink::Pixels(up, down) => {
                    let mut code = up.map_or("39".to_owned(), |c| sixteen(c).to_string());
                    if let Some(c) = down {
                        code += &format!(";{}", sixteen(c) + 10);
                    }
                    code
                }
            };
            format!("\x1b[0;{code}m")
        }
    }
}

/// The nearest of the sixteen colours for a pixel of the mark.
fn sixteen((r, g, b): Rgb) -> u8 {
    let light = r as u16 + g as u16 + b as u16;
    if r > 180 && g < 170 && b < 140 {
        33
    } else if light > 560 {
        97
    } else if light > 300 {
        35
    } else {
        90
    }
}

const QUADRANTS: [char; 16] = [
    ' ', '▘', '▝', '▀', '▖', '▌', '▞', '▛', '▗', '▚', '▐', '▜', '▄', '▙', '▟', '█',
];

/// Row `row` of a picture `width` pixels wide, two by two pixels a cell.
/// A cell shows two colours at most: the commonest pixel colour, and the
/// clear background or the next commonest, which the others round to.
pub fn quadrants(width: usize, row: usize, px: impl Fn(usize, usize) -> Option<Rgb>) -> Line {
    let mut line = Line::default();
    for cx in 0..width.div_ceil(2) {
        let cell = [
            px(2 * cx, 2 * row),
            px(2 * cx + 1, 2 * row),
            px(2 * cx, 2 * row + 1),
            px(2 * cx + 1, 2 * row + 1),
        ];
        let mut counts: Vec<(Rgb, usize)> = Vec::new();
        for c in cell.iter().flatten() {
            match counts.iter_mut().find(|(k, _)| k == c) {
                Some((_, n)) => *n += 1,
                None => counts.push((*c, 1)),
            }
        }
        counts.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        let Some(&(fg, _)) = counts.first() else {
            line.push(Ink::Pixels(None, None), " ");
            continue;
        };
        let bg = match cell.contains(&None) {
            true => None,
            false => counts.get(1).map(|&(c, _)| c),
        };
        let near = |a: Rgb, b: Rgb| {
            let d = |x: u8, y: u8| (x as i32 - y as i32).pow(2);
            d(a.0, b.0) + d(a.1, b.1) + d(a.2, b.2)
        };
        let mask = cell.iter().enumerate().fold(0, |mask, (i, c)| {
            let lit = match (c, bg) {
                (None, _) => false,
                (Some(_), None) => true,
                (Some(c), Some(b)) => near(*c, fg) <= near(*c, b),
            };
            mask | (usize::from(lit) << i)
        });
        line.push(Ink::Pixels(Some(fg), bg), QUADRANTS[mask].to_string());
    }
    line
}
