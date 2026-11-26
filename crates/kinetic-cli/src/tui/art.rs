//! The home screen's pixel art: the wordmark's letters and the tiles'
//! icons, one character a pixel, drawn two by two pixels a cell.

use super::paint::{quadrants, Line, Rgb, GREY, LAVENDER, ORANGE, PURPLE, WHITE};

/// Pixels across and down an icon.
pub const ICON: (usize, usize) = (24, 12);

const CHAT: [&str; 12] = [
    "....llllllllllllllll....",
    "..llwwwlllllllllllllll..",
    ".llwlllllllllllllllllll.",
    ".llllllllllllllllllllll.",
    ".llloooolloooolloooolll.",
    ".llloooolloooolloooolll.",
    ".llllllllllllllllllllll.",
    ".llllllllllllllllllllll.",
    "..pllllllllllllllllllp..",
    "...pppppppppppppppppp...",
    "...ppp..................",
    "..pp....................",
];
const MODELS: [&str; 12] = [
    "......ll...ll...ll......",
    "......ll...ll...ll......",
    "....lllllllllllllllp....",
    "lllllwwlllllllllllpppppp",
    "llllllllwwoooooollpppppp",
    "....lllloooooooollpp....",
    "....lllloooooooollpp....",
    "lllllllloooooooollpppppp",
    "llllllllllllllllllpppppp",
    "....pppppppppppppppp....",
    "......pp...pp...pp......",
    "......pp...pp...pp......",
];
const API: [&str; 12] = [
    ".......ww....ww.........",
    ".......ll....ll.........",
    ".......ll....ll.........",
    "....llllllllllllll......",
    "....lwwlllllllllll......",
    "....llllllllllllll......",
    ".....pllllllllllp.......",
    "......pppppppppp........",
    "........pppppp..........",
    "..........oo............",
    "..........ooo...........",
    "...........ooooooooooooo",
];
const BENCH: [&str; 12] = [
    "........................",
    ".......llllllllll.......",
    "....pppllllllllllooo....",
    "..ppppp..........ooooo..",
    ".pppp..........ww..oooo.",
    "pppp.........www....oooo",
    "ppp.......wwww.......ooo",
    "........................",
    "...................oooo.",
    "..............llll.oooo.",
    ".........llll.llll.oooo.",
    "....pppp.llll.llll.oooo.",
];
const DOCTOR: [&str; 12] = [
    "...oooooo......oooooo...",
    ".oooooooooow.oooooooooo.",
    "oooooooooowwwooooooooooo",
    "oooooooooowowooooooooooo",
    "ooooooooowoowooooooooooo",
    "wwwwwwwwwoooowoowwwwwwww",
    "..ooooooooooowowoooooo..",
    "....ooooooooowwooooo....",
    "......oooooooowooo......",
    "........oooooooo........",
    "..........oooo..........",
    "........................",
];
const SETTINGS: [&str; 12] = [
    ".........llllll.........",
    "...ll....llllll....ll...",
    "..lllllllllllllllllllp..",
    "...llllllllllllllllpp...",
    "...llllooo....ooopppp...",
    "lllllllo........oppppppp",
    "lllllllo........oppppppp",
    "...llllooo....ooopppp...",
    "...llllllpppppppppppp...",
    "..lllllppppppppppppppp..",
    "...ll....pppppp....pp...",
    ".........pppppp.........",
];

/// The icons in the order of the tiles.
pub const ICONS: [[&str; 12]; 6] = [CHAT, MODELS, API, BENCH, DOCTOR, SETTINGS];

/// Row `row` of `icon`, in full colour when `lit`, softer otherwise.
pub fn icon_row(icon: &[&str; 12], row: usize, lit: bool) -> Line {
    quadrants(ICON.0, row, |x, y| {
        let c = icon.get(y)?.as_bytes().get(x)?;
        Some(match (c, lit) {
            (b'l', true) => LAVENDER,
            (b'p', true) => PURPLE,
            (b'o', true) => ORANGE,
            (b'w', true) => WHITE,
            (b'l', false) => GREY,
            (b'p', false) => (84, 72, 96),
            (b'o', false) => (156, 108, 74),
            (b'w', false) => (200, 195, 206),
            _ => return None,
        })
    })
}

/// Letters 9 pixels tall.
const BIG: [(char, [&str; 9]); 6] = [
    (
        'K',
        [
            "##...##", "##..##.", "##.##..", "####...", "###....", "####...", "##.##..", "##..##.",
            "##...##",
        ],
    ),
    (
        'I',
        [
            "######", "..##..", "..##..", "..##..", "..##..", "..##..", "..##..", "..##..",
            "######",
        ],
    ),
    (
        'N',
        [
            "##...##", "###..##", "###..##", "####.##", "##.####", "##..###", "##..###", "##...##",
            "##...##",
        ],
    ),
    (
        'E',
        [
            "#######", "##.....", "##.....", "##.....", "######.", "##.....", "##.....", "##.....",
            "#######",
        ],
    ),
    (
        'T',
        [
            "########", "...##...", "...##...", "...##...", "...##...", "...##...", "...##...",
            "...##...", "...##...",
        ],
    ),
    (
        'C',
        [
            ".#####.", "##...##", "##.....", "##.....", "##.....", "##.....", "##.....", "##...##",
            ".#####.",
        ],
    ),
];

/// Letters 7 pixels tall, for smaller windows.
const SMALL: [(char, [&str; 7]); 6] = [
    (
        'K',
        [
            "#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#",
        ],
    ),
    ('I', ["###", ".#.", ".#.", ".#.", ".#.", ".#.", "###"]),
    (
        'N',
        [
            "#...#", "##..#", "##..#", "#.#.#", "#..##", "#..##", "#...#",
        ],
    ),
    (
        'E',
        [
            "#####", "#....", "#....", "####.", "#....", "#....", "#####",
        ],
    ),
    (
        'T',
        [
            "#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#..",
        ],
    ),
    (
        'C',
        [
            ".####", "#....", "#....", "#....", "#....", "#....", ".####",
        ],
    ),
];

/// `text` in the pixel letters, each letter pixel two pixels wide so it
/// comes out square, a letter pixel between letters; `#` is lit.
pub fn word(text: &str, big: bool) -> Vec<String> {
    let glyph = |c: char| -> Vec<&str> {
        match big {
            true => BIG.iter().find(|(k, _)| *k == c).map(|(_, g)| g.to_vec()),
            false => SMALL.iter().find(|(k, _)| *k == c).map(|(_, g)| g.to_vec()),
        }
        .unwrap_or_default()
    };
    let height = if big { 9 } else { 7 };
    (0..height)
        .map(|y| {
            text.chars()
                .map(|c| {
                    glyph(c).get(y).map_or(String::new(), |row| {
                        row.replace('#', "##").replace('.', "..")
                    })
                })
                .collect::<Vec<_>>()
                .join("..")
        })
        .collect()
}

/// The rows of `text` in the pixel letters: white fading to lavender, with
/// a purple shadow a pixel down and to the right.
pub fn wordmark(text: &str, big: bool) -> Vec<Line> {
    let rows = word(text, big);
    let width = rows.first().map_or(0, |r| r.len()) + 2;
    let height = rows.len() + 1;
    let lit = |x: usize, y: usize| rows.get(y).and_then(|r| r.as_bytes().get(x)) == Some(&b'#');
    let px = |x: usize, y: usize| -> Option<Rgb> {
        if lit(x, y) {
            let t = y as f32 / (rows.len() - 1).max(1) as f32;
            let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
            return Some((
                mix(WHITE.0, LAVENDER.0),
                mix(WHITE.1, LAVENDER.1),
                mix(WHITE.2, LAVENDER.2),
            ));
        }
        (x >= 2 && y >= 1 && lit(x - 2, y - 1)).then_some(PURPLE)
    };
    (0..height.div_ceil(2))
        .map(|r| quadrants(width, r, px))
        .collect()
}

/// The squares of the Kinetic mark as assets/kinetic-lockup-on-dark.svg
/// draws them: centre, side, degrees turned, colour; later ones on top.
const MARK: [((f32, f32), f32, f32, Rgb); 13] = [
    ((18.0, 70.0), 14.0, -70.59, (0x3A, 0x36, 0x40)),
    ((25.75, 50.75), 9.72, -64.84, (0x59, 0x54, 0x5F)),
    ((33.5, 37.0), 6.66, -54.83, (0x6F, 0x6A, 0x75)),
    ((41.25, 28.75), 4.83, -35.36, (0x84, 0x7F, 0x8A)),
    ((49.0, 26.0), 4.21, 0.0, (0x97, 0x91, 0x9D)),
    ((56.75, 28.75), 4.83, 35.36, (0xA9, 0xA3, 0xAF)),
    ((64.5, 37.0), 6.66, 54.83, (0xBB, 0xB4, 0xC1)),
    ((72.25, 50.75), 9.72, 64.84, (0xCC, 0xC5, 0xD2)),
    ((80.0, 70.0), 15.0, 70.59, (0xA8, 0x5A, 0x26)),
    ((80.0, 70.0), 12.45, 70.59, (0xBC, 0x75, 0x44)),
    ((80.0, 70.0), 9.9, 70.59, (0xCF, 0x90, 0x62)),
    ((80.0, 70.0), 7.35, 70.59, (0xE2, 0xAC, 0x80)),
    ((80.0, 70.0), 4.8, 70.59, (0xF6, 0xC7, 0x9E)),
];
/// The rounded square the mark sits on, as the app icon has it: its
/// edges and the radius of its corners.
const PANEL: (f32, f32, f32) = (4.0, 96.0, 18.0);
const PANEL_COLOUR: Rgb = (0x1C, 0x1A, 0x1F);
/// The part of the drawing shown: left, top, right, bottom.
const MARK_BOX: (f32, f32, f32, f32) = (4.0, 4.0, 96.0, 96.0);

/// Units of the mark's drawing a pixel is tall when it is `rows` cells
/// tall; a pixel is half as wide.
fn mark_unit(rows: usize) -> f32 {
    (MARK_BOX.3 - MARK_BOX.1) / (2 * rows) as f32
}

/// Cells across the mark `rows` cells tall.
pub fn mark_width(rows: usize) -> usize {
    let pixels = ((MARK_BOX.2 - MARK_BOX.0) / (mark_unit(rows) / 2.0)).ceil() as usize;
    pixels.div_ceil(2)
}

/// What is at `(x, y)` in the mark's drawing: the panel as 0, a square
/// as its place in [`MARK`] plus one, or nothing.
fn mark_at(x: f32, y: f32) -> Option<usize> {
    let square = MARK.iter().rposition(|&((cx, cy), side, turn, _)| {
        let (sin, cos) = (-turn.to_radians()).sin_cos();
        let (dx, dy) = (x - cx, y - cy);
        let (u, v) = (dx * cos - dy * sin, dx * sin + dy * cos);
        u.abs() <= side / 2.0 && v.abs() <= side / 2.0
    });
    if let Some(i) = square {
        return Some(i + 1);
    }
    let (lo, hi, r) = PANEL;
    let mid = (lo + hi) / 2.0;
    let inner = (hi - lo) / 2.0 - r;
    let (qx, qy) = ((x - mid).abs() - inner, (y - mid).abs() - inner);
    let inside = x >= lo && x <= hi && y >= lo && y <= hi;
    (inside && (qx <= 0.0 || qy <= 0.0 || qx.hypot(qy) <= r)).then_some(0)
}

fn mark_colour(at: usize) -> Rgb {
    match at {
        0 => PANEL_COLOUR,
        i => MARK[i - 1].3,
    }
}

/// The mark `rows` cells tall, drawn from its squares: a pixel takes the
/// square that covers most of it, so the small dots stay.
pub fn mark(rows: usize) -> Vec<Line> {
    const SAMPLES: usize = 4;
    let unit = mark_unit(rows);
    let width = 2 * mark_width(rows);
    let pixel = |x: usize, y: usize| -> Option<Rgb> {
        let mut hits = [0usize; MARK.len() + 1];
        for i in 0..SAMPLES * SAMPLES {
            let fx = (i % SAMPLES) as f32 / SAMPLES as f32 + 0.125;
            let fy = (i / SAMPLES) as f32 / SAMPLES as f32 + 0.125;
            let sx = MARK_BOX.0 + (x as f32 + fx) * unit / 2.0;
            let sy = MARK_BOX.1 + (y as f32 + fy) * unit;
            if let Some(k) = mark_at(sx, sy) {
                hits[k] += 1;
            }
        }
        // a square that covers a quarter of the pixel wins over the panel
        let square = (1..hits.len())
            .filter(|&k| hits[k] >= SAMPLES * SAMPLES / 4)
            .max_by_key(|&k| (hits[k], k));
        let panel = (hits[0] > SAMPLES * SAMPLES / 2).then_some(0);
        square.or(panel).map(mark_colour)
    };
    let grid: Vec<Vec<Option<Rgb>>> = (0..2 * rows)
        .map(|y| (0..width).map(|x| pixel(x, y)).collect())
        .collect();
    (0..rows)
        .map(|r| quadrants(width, r, |x, y| grid.get(y)?.get(x).copied().flatten()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letters_and_icons_are_even() {
        for icon in ICONS {
            assert!(icon.iter().all(|r| r.len() == ICON.0), "{icon:?}");
        }
        for big in [true, false] {
            let rows = word("KINETIC", big);
            assert!(rows.iter().all(|r| r.len() == rows[0].len()));
        }
        assert_eq!(word("KINETIC", true)[0].len(), 108);
        assert_eq!(wordmark("KINETIC", true).len(), 5);
        assert_eq!(wordmark("KINETIC", false)[0].width(), 38);
        for rows in [4, 6] {
            let m = mark(rows);
            assert_eq!(m.len(), rows);
            assert!(m.iter().all(|l| l.width() == mark_width(rows)));
        }
        assert_eq!(mark_at(80.0, 70.0), Some(13));
        assert_eq!(mark_at(49.0, 26.0), Some(5));
        assert_eq!(mark_at(50.0, 60.0), Some(0));
        assert_eq!(mark_at(5.0, 5.0), None);
    }
}
