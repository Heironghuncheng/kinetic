//! Screens driven by the arrow keys: the home screen `kinetic` opens, the
//! model list, settings, and the pickers commands open when no model is
//! named.

pub mod actions;
mod art;
mod folder;
pub mod home;
mod keys;
pub mod last;
mod list;
pub mod models;
mod paint;
mod scene;
pub mod settings;
mod term;

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

pub use actions::Action;
pub use models::{Here, Model, Target};
pub use term::Term;

use crate::library::{self, Local};
use paint::{Ink, Line};

/// What a screen returns when the user quits with Ctrl+C or q.
fn quit() -> io::Error {
    io::ErrorKind::Interrupted.into()
}

pub fn is_quit(e: &io::Error) -> bool {
    e.kind() == io::ErrorKind::Interrupted
}

/// `head` and `body` from the top, then `note` and the line of keys on the
/// last two rows of a window `rows` high.
fn frame(head: Vec<Line>, body: Vec<Line>, note: &str, keys: &str, rows: usize) -> Vec<Line> {
    let room = rows.saturating_sub(2);
    let mut lines: Vec<Line> = head.into_iter().chain(body).take(room).collect();
    lines.resize_with(room, Line::default);
    lines.push(Line::new(Ink::Accent, format!("  {note}")));
    lines.push(Line::new(Ink::Dim, format!("  {keys}")));
    lines
}

/// The model list for a command that was given none: the file picked,
/// downloaded first if it is not here; `None` when the user leaves.
pub fn pick_model(title: &str) -> crate::Result<Option<PathBuf>> {
    println!("Looking for models on this computer...");
    let here = Here::check();
    let picked = match models::pick(&mut Term::enter()?, &here, title) {
        Err(e) if is_quit(&e) => None,
        picked => picked?,
    };
    picked.map(|m| fetch(&m)).transpose()
}

/// The file of `model`, downloaded as `kinetic pull` does when it is not on
/// disk yet, remembered as the last model.
pub fn fetch(model: &Model) -> crate::Result<PathBuf> {
    let path = match &model.target {
        Target::Disk(path) => path.clone(),
        Target::Hub(spec) => {
            println!("Downloading {}; this happens only once.", model.name);
            crate::pull::run(&crate::pull::Pull {
                spec,
                file: None,
                projector: true,
                dir: None,
            })?
        }
    };
    last::remember(&path);
    Ok(path)
}

/// The model `arg` names, as [`library::locate`] finds it; when it finds
/// none and a person is at the terminal, the models with names like it are
/// offered instead.
pub fn locate(arg: &Path) -> crate::Result<Local> {
    let missed = match library::locate(arg) {
        Ok(model) => return Ok(model),
        Err(e) => e,
    };
    let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
    if !interactive || arg.extension().is_some_and(|e| e == "gguf" || e == "ktc") {
        return Err(missed);
    }
    let query = arg.to_string_lossy();
    let mut models = library::discover();
    let close = library::resembling(&models, &query);
    let picked = match close[..] {
        [] => return Err(missed),
        [i] => {
            print!(
                "No model called {query} here. Did you mean {}? [Enter] yes, Esc no ",
                models[i].name
            );
            io::stdout().flush()?;
            matches!(
                crate::console::read_key(),
                Some(b'\r' | b'\n' | b'y' | b'Y')
            )
            .then_some(i)
        }
        _ => {
            println!("No model called {query} here. Did you mean:");
            for (n, &i) in close.iter().enumerate() {
                println!("  {}  {}", n + 1, models[i].name);
            }
            print!("Press 1-{} or Esc for none: ", close.len());
            io::stdout().flush()?;
            let key = crate::console::read_key().map(char::from);
            let n = key.and_then(|k| k.to_digit(10)).unwrap_or(0) as usize;
            close.get(n.wrapping_sub(1)).copied()
        }
    };
    println!();
    match picked {
        Some(i) => Ok(models.swap_remove(i)),
        None => Err(missed),
    }
}
