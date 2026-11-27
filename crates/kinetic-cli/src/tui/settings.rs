//! Every setting `chat`, `serve`, `run` and `bench` share, changed with the
//! arrows and saved for the next run, each beside the flag it stands for.

use std::io;

use super::keys::Key;
use super::list::{Item, List};
use super::paint::{Ink, Line};
use super::{frame, quit, Here, Term};
use crate::config::{Settings, UPDATE_CHECK};

/// The speed technologies `--off` takes, as shown and as typed.
const TECHS: [(&str, &str); 6] = [
    ("RAD-1", "rad1"),
    ("Warm Start", "warm-start"),
    ("Zero Wait", "zero-wait"),
    ("Shortlist", "shortlist"),
    ("Torque", "torque"),
    ("Convoy", "convoy"),
];

enum Row {
    /// A flag and its values; the first, empty, is the default.
    Choice {
        name: &'static str,
        flag: &'static str,
        values: Vec<String>,
        default: &'static str,
    },
    /// A speed technology, on unless `--off` names it.
    Tech(&'static str, &'static str),
    /// The daily check for a new release, off unless turned on.
    UpdateCheck,
}

fn steps(from: f32, to: f32, step: f32, digits: usize) -> Vec<String> {
    let n = ((to - from) / step).round() as usize;
    let mut v = vec![String::new()];
    v.extend((0..=n).map(|i| format!("{:.*}", digits, from + step * i as f32)));
    v
}

fn list(values: &[&str]) -> Vec<String> {
    std::iter::once(String::new())
        .chain(values.iter().map(|v| (*v).to_owned()))
        .collect()
}

fn rows(here: &Here) -> Vec<Row> {
    let choice = |name, flag, values, default| Row::Choice {
        name,
        flag,
        values,
        default,
    };
    let devices: Vec<&str> = here.devices().into_iter().skip(1).collect();
    let cuda = here.gpus.iter().filter(|g| g.api == "CUDA").count();
    let mut gpus: Vec<String> = (0..cuda).map(|i| i.to_string()).collect();
    if cuda > 1 {
        gpus.push(
            (0..cuda)
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    let threads: Vec<String> = (1..=here.machine.threads).map(|n| n.to_string()).collect();
    let mut rows = vec![
        choice("Device", "--device", list(&devices), "picked for you"),
        choice(
            "GPUs",
            "--gpus",
            list(&gpus.iter().map(String::as_str).collect::<Vec<_>>()),
            "the first",
        ),
        choice(
            "Context",
            "--ctx-size",
            list(&[
                "2048", "4096", "8192", "16384", "32768", "65536", "131072", "0",
            ]),
            "32768, or what fits",
        ),
        choice(
            "Threads",
            "--threads",
            list(&threads.iter().map(String::as_str).collect::<Vec<_>>()),
            "all",
        ),
        choice(
            "Requests at once",
            "--parallel",
            list(&["1", "2", "3", "4", "6", "8"]),
            "what memory allows",
        ),
        choice(
            "Temperature",
            "--temp",
            steps(0.0, 2.0, 0.1, 1),
            "the model's",
        ),
        choice("Top-p", "--top-p", steps(0.05, 1.0, 0.05, 2), "the model's"),
        choice(
            "Top-k",
            "--top-k",
            list(&["0", "5", "10", "20", "40", "50", "100", "200"]),
            "the model's",
        ),
        choice("Min-p", "--min-p", steps(0.0, 0.3, 0.01, 2), "the model's"),
        choice(
            "Repeat penalty",
            "--repeat-penalty",
            steps(1.0, 1.5, 0.05, 2),
            "the model's",
        ),
        choice(
            "Most tokens per answer",
            "--n-predict",
            list(&["256", "512", "1024", "2048", "4096", "8192"]),
            "no limit",
        ),
        choice("Reasoning", "--reasoning", list(&["off"]), "on"),
    ];
    rows.extend(TECHS.iter().map(|&(name, id)| Row::Tech(name, id)));
    rows.push(Row::UpdateCheck);
    rows
}

fn off(s: &Settings) -> Vec<String> {
    s.get("--off")
        .map(|v| v.split(',').map(str::to_owned).collect())
        .unwrap_or_default()
}

impl Row {
    fn cells(&self, s: &Settings) -> Vec<String> {
        match self {
            Row::Choice {
                name,
                flag,
                default,
                ..
            } => match s.get(flag) {
                Some(v) => {
                    let shown = if *flag == "--ctx-size" && v == "0" {
                        "all the model takes"
                    } else {
                        v
                    };
                    vec![
                        (*name).into(),
                        format!("‹ {shown} ›"),
                        format!("{flag} {v}"),
                    ]
                }
                None => vec![(*name).into(), format!("‹ {default} ›"), String::new()],
            },
            Row::Tech(name, id) => match off(s).iter().any(|o| o == id) {
                true => vec![
                    format!("{name} (speed)"),
                    "[   ] off".into(),
                    format!("--off {id}"),
                ],
                false => vec![format!("{name} (speed)"), "[ ✓ ] on".into(), String::new()],
            },
            Row::UpdateCheck => vec![
                "Check for new releases once a day".into(),
                match s.checks_updates() {
                    true => "[ ✓ ] on".into(),
                    false => "[   ] off".into(),
                },
                "asks GitHub's releases API, sends nothing else".into(),
            ],
        }
    }

    /// Moves to the value `by` steps away, or flips a technology.
    fn change(&self, s: &mut Settings, by: isize) {
        match self {
            Row::Choice { flag, values, .. } => {
                let now = s.get(flag).unwrap_or("");
                let at = values.iter().position(|v| v == now).unwrap_or(0);
                let next = (at as isize + by).clamp(0, values.len() as isize - 1) as usize;
                s.set(flag, &values[next]);
            }
            Row::Tech(_, id) => {
                let mut off = off(s);
                match off.iter().position(|o| o == id) {
                    Some(i) => {
                        off.remove(i);
                    }
                    None => off.push((*id).to_owned()),
                }
                s.set("--off", &off.join(","));
            }
            Row::UpdateCheck => {
                let on = if s.checks_updates() { "" } else { "on" };
                s.set(UPDATE_CHECK, on);
            }
        }
    }
}

pub fn show(term: &mut Term, here: &Here) -> io::Result<()> {
    let rows = rows(here);
    let mut settings = Settings::load();
    let mut at = 0;
    let mut note = String::new();
    loop {
        let (cols, height) = super::inner(term.size());
        let ink = Ink::Plain;
        let mut list = List::new(
            rows.iter()
                .map(|r| Item::new(r.cells(&settings), ink))
                .collect(),
        );
        list.select(at);
        let flags = settings.flags_for("chat");
        let head = vec![
            Line::new(Ink::Bright, "  Settings"),
            Line::new(
                Ink::Dim,
                "  Saved for chat, serve, run and bench; a flag you type wins.",
            ),
            Line::default(),
        ];
        let mut body = list.lines(height.saturating_sub(head.len() + 6), cols);
        body.push(Line::default());
        body.push(Line::new(Ink::Dim, "  Chat runs as:"));
        body.push(Line::new(
            Ink::Plain,
            format!("  kinetic chat MODEL {}", flags.join(" ")),
        ));
        let keys = "↑↓ move   ←→ change   Enter switch   Backspace default   Esc back   q quit";
        term.draw(&frame(term, head, body, &note, keys))?;
        let key = term.key()?;
        let row = &rows[at];
        let before = settings.clone();
        match key {
            Key::CtrlC | Key::Char('q' | 'Q') => return Err(quit()),
            Key::Esc => return Ok(()),
            Key::Up => at = at.saturating_sub(1),
            Key::Down => at = (at + 1).min(rows.len() - 1),
            Key::Home => at = 0,
            Key::End => at = rows.len() - 1,
            Key::Left => row.change(&mut settings, -1),
            Key::Right | Key::Enter | Key::Char(' ') => row.change(&mut settings, 1),
            Key::Backspace | Key::Char('0') => {
                if let Row::Choice { flag, .. } = row {
                    settings.set(flag, "");
                }
            }
            _ => {}
        }
        if settings != before {
            note = match settings.save() {
                Ok(()) => String::new(),
                Err(e) => format!("Not saved: {e}"),
            };
        }
    }
}
