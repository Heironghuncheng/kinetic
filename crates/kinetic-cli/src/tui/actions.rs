//! What to do with the model picked, and where it runs.

use std::io;

use super::keys::Key;
use super::list::{shorten, Item, List};
use super::models::{Model, Target};
use super::paint::{Ink, Line};
use super::{frame, quit, Term};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Chat,
    Serve,
    Bench,
    Info,
}

const ACTIONS: [(Action, &str, &str); 4] = [
    (Action::Chat, "Chat", "talk to it here, in the terminal"),
    (
        Action::Serve,
        "Serve for apps",
        "an address to paste into OpenCode, Open WebUI and other apps",
    ),
    (Action::Bench, "Benchmark", "measure how fast it runs here"),
    (Action::Info, "Info", "what the file holds"),
];

/// How a `--device` value reads on screen.
pub fn device_name(device: &str) -> &'static str {
    match device {
        "cpu" => "the processor (CPU)",
        "cuda" => "the NVIDIA GPU (CUDA)",
        "vulkan" => "the GPU through Vulkan",
        _ => "picked for you",
    }
}

/// Asks what to do with `model`; `devices[*device]` is where it runs, and
/// `d` moves to the next. `None` when the user goes back.
pub fn choose(
    term: &mut Term,
    model: &Model,
    devices: &[&str],
    device: &mut usize,
) -> io::Result<Option<Action>> {
    let offered: Vec<_> = ACTIONS
        .iter()
        .filter(|(a, ..)| *a != Action::Info || matches!(model.target, Target::Disk(_)))
        .collect();
    let mut list = List::new(
        offered
            .iter()
            .enumerate()
            .map(|(i, (_, name, what))| {
                Item::new(
                    vec![format!("{} {name}", i + 1), (*what).into()],
                    Ink::Plain,
                )
            })
            .collect(),
    );
    let place = match &model.target {
        Target::Disk(p) => p.display().to_string(),
        Target::Hub(spec) => format!("downloads from Hugging Face first: {spec}"),
    };
    loop {
        let (cols, rows) = term.size();
        let on = match devices[*device] {
            "auto" => match model.fit.text().as_str() {
                "" => device_name("auto").to_owned(),
                on => format!("{on} ({})", device_name("auto")),
            },
            d => device_name(d).to_owned(),
        };
        let mut runs = Line::new(Ink::Dim, "  Runs on   ");
        runs.push(Ink::Bright, on);
        if devices.len() > 1 {
            runs.push(Ink::Dim, "   d changes");
        }
        let head = vec![
            Line::new(
                Ink::Bright,
                format!("  {}", shorten(&model.name, cols.saturating_sub(4))),
            ),
            Line::new(
                Ink::Dim,
                format!("  {}, {}, {place}", model.size_text(), model.store),
            ),
            Line::default(),
            runs,
            Line::default(),
            Line::new(Ink::Plain, "  What would you like to do?"),
            Line::default(),
        ];
        let body = list.lines(rows.saturating_sub(head.len() + 3), cols);
        let keys = "↑↓ move   Enter start   d device   Esc back   q quit";
        term.draw(&frame(head, body, "", keys, rows))?;
        match term.key()? {
            Key::CtrlC | Key::Char('q' | 'Q') => return Err(quit()),
            Key::Esc => return Ok(None),
            Key::Char('d' | 'D') => *device = (*device + 1) % devices.len(),
            Key::Enter => {
                if let Some(i) = list.selected() {
                    return Ok(Some(offered[i].0));
                }
            }
            Key::Char(c) => {
                let n = c.to_digit(10).unwrap_or(0) as usize;
                if (1..=offered.len()).contains(&n) {
                    return Ok(Some(offered[n - 1].0));
                }
            }
            key => {
                list.handle(key);
            }
        }
    }
}
