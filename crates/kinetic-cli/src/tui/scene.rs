//! A night over the mountains in pixels, two to a cell, drawn the same at
//! every launch and fitted to any width.

use super::paint::{Rgb, ORANGE};

const SKY_TOP: Rgb = (11, 15, 42);
const SKY_LOW: Rgb = (26, 34, 86);
const STAR: Rgb = (232, 235, 255);
const STAR_DIM: Rgb = (185, 175, 195);
const LIT: Rgb = (52, 66, 146);
const SHADE: Rgb = (38, 50, 116);
const FAR: Rgb = (28, 38, 94);
const SNOW: Rgb = (176, 184, 228);
const SNOW_TIP: Rgb = (241, 239, 243);
const HILL: Rgb = (22, 34, 77);
const HILL_LOW: Rgb = (17, 26, 63);
const GROUND: Rgb = (8, 9, 16);
const WHITE: Rgb = (241, 239, 243);

const BAYER: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

fn hash(a: usize, b: usize) -> u32 {
    let mut h = (a as u32).wrapping_mul(0x9e37_79b1) ^ (b as u32).wrapping_mul(0x85eb_ca77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^ (h >> 12)
}

fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    (m(a.0, b.0), m(a.1, b.1), m(a.2, b.2))
}

/// A peak: its middle, its height in pixels, how fast its sides fall.
struct Peak {
    x: f32,
    height: f32,
    slope: f32,
}

/// The scene's pixels: `width` wide, `height` tall.
struct Scene {
    width: usize,
    height: usize,
    near: Vec<Peak>,
    far: Vec<Peak>,
}

impl Scene {
    fn new(width: usize, height: usize) -> Scene {
        let peaks = |every: usize, seed: usize, tall: f32| -> Vec<Peak> {
            let n = (width / every).max(2);
            (0..n)
                .map(|i| {
                    let r = hash(i, seed);
                    let jitter = (r % 100) as f32 / 100.0 - 0.5;
                    Peak {
                        x: (i as f32 + 0.5 + jitter * 0.6) * width as f32 / n as f32,
                        height: height as f32 * tall * (0.6 + ((r >> 8) % 40) as f32 / 100.0),
                        slope: 1.0 + ((r >> 16) % 60) as f32 / 100.0,
                    }
                })
                .collect()
        };
        Scene {
            width,
            height,
            near: peaks(22, 1, 0.4),
            far: peaks(15, 2, 0.32),
        }
    }

    /// Height of a ridge above the bottom at column `x`, and the peak that
    /// makes it.
    fn ridge<'p>(&self, peaks: &'p [Peak], x: usize) -> (f32, Option<&'p Peak>) {
        let jag = (hash(x, 7) % 3) as f32 * 0.5;
        peaks
            .iter()
            .map(|p| (p.height - (x as f32 - p.x).abs() * p.slope - jag, Some(p)))
            .fold((0.0, None), |a, b| if b.0 > a.0 { b } else { a })
    }

    fn hill(&self, x: usize) -> f32 {
        let t = x as f32 / 9.0;
        self.height as f32 * 0.1 + (t.sin() * 1.6 + (t * 0.37 + 1.3).sin() * 2.2).max(-2.0)
    }

    fn pixel(&self, x: usize, y: usize, tick: u64) -> Rgb {
        let up = (self.height - y) as f32;
        let hill = self.hill(x);
        if y + 2 >= self.height {
            return GROUND;
        }
        if up <= hill {
            for (at, light) in [(0.21, ORANGE), (0.64, WHITE), (0.83, ORANGE)] {
                let lx = (self.width as f32 * at) as usize;
                if x == lx && (hill - up - 1.5).abs() < 0.6 {
                    return light;
                }
            }
            return if up < hill - 3.0 { HILL_LOW } else { HILL };
        }
        let (near, peak) = self.ridge(&self.near, x);
        if up <= near {
            let cap =
                peak.is_some_and(|p| p.height > self.height as f32 * 0.3 && up > p.height - 3.5);
            if cap && up > near - 2.0 {
                return if up > near - 1.0 { SNOW_TIP } else { SNOW };
            }
            return match peak.is_some_and(|p| (x as f32) < p.x) {
                true => LIT,
                false => SHADE,
            };
        }
        if up <= self.ridge(&self.far, x).0 {
            return FAR;
        }
        let r = hash(x, y + 1000);
        if r.is_multiple_of(149) && up > near + 2.0 {
            let twinkles = r.is_multiple_of(3) && (tick + u64::from(r >> 5)).is_multiple_of(4);
            return match r.is_multiple_of(5) ^ twinkles {
                true => STAR,
                false => STAR_DIM,
            };
        }
        let t = y as f32 / self.height as f32;
        let steps = 5.0;
        let level = t * steps + BAYER[y % 4][x % 4] as f32 / 16.0 - 0.5;
        mix(SKY_TOP, SKY_LOW, (level.round().clamp(0.0, steps)) / steps)
    }
}

/// The scene filling `cols` by `rows` cells, the upper and lower pixel of
/// each; `tick` moves the twinkling stars.
pub fn cells(cols: usize, rows: usize, tick: u64) -> Vec<Vec<(Rgb, Rgb)>> {
    let scene = Scene::new(cols, rows * 2);
    (0..rows)
        .map(|row| {
            (0..cols)
                .map(|x| {
                    (
                        scene.pixel(x, 2 * row, tick),
                        scene.pixel(x, 2 * row + 1, tick),
                    )
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scene_fills_the_window_the_same_every_time() {
        for (cols, rows) in [(80, 24), (300, 80)] {
            let a = cells(cols, rows, 0);
            assert_eq!(a.len(), rows);
            assert!(a.iter().all(|r| r.len() == cols));
            assert_eq!(a, cells(cols, rows, 0));
            assert_eq!(a[rows - 1][0], (GROUND, GROUND));
        }
    }
}
