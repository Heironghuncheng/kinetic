//! Walking folders with the arrows to a .gguf file anywhere on disk.

use std::io;
use std::path::{Path, PathBuf};

use super::keys::Key;
use super::list::{Item, List};
use super::paint::{Ink, Line};
use super::{frame, quit, Term};

enum Entry {
    Up,
    Dir(PathBuf),
    File(PathBuf),
}

/// The folders and model files in `dir`, folders first, or the drives when
/// `dir` is `None`.
fn entries(dir: Option<&Path>) -> Vec<Entry> {
    let Some(dir) = dir else {
        return drives().into_iter().map(Entry::Dir).collect();
    };
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            files.push(path);
        }
    }
    dirs.sort();
    files.sort();
    let up = (dir.parent().is_some() || cfg!(windows)).then_some(Entry::Up);
    up.into_iter()
        .chain(dirs.into_iter().map(Entry::Dir))
        .chain(files.into_iter().map(Entry::File))
        .collect()
}

#[cfg(windows)]
fn drives() -> Vec<PathBuf> {
    (b'A'..=b'Z')
        .map(|d| PathBuf::from(format!("{}:\\", d as char)))
        .filter(|p| p.is_dir())
        .collect()
}

#[cfg(not(windows))]
fn drives() -> Vec<PathBuf> {
    vec![PathBuf::from("/")]
}

fn name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

fn item(entry: &Entry) -> Item {
    match entry {
        Entry::Up => Item::new(vec!["..".into(), String::new()], Ink::Muted),
        Entry::Dir(p) => Item::new(vec![format!("{}/", name(p)), String::new()], Ink::Plain),
        Entry::File(p) => {
            let size = std::fs::metadata(p).map_or(0, |m| m.len());
            let size = format!("{:.2} GiB", size as f64 / (1u64 << 30) as f64);
            Item::new(vec![name(p), size], Ink::Bright)
        }
    }
}

/// Lets the user walk to a model file from the current folder; `None` when
/// they go back.
pub fn browse(term: &mut Term) -> io::Result<Option<PathBuf>> {
    let mut dir = std::env::current_dir().ok();
    let mut came_from: Option<PathBuf> = None;
    loop {
        let found = entries(dir.as_deref());
        let mut list = List::new(found.iter().map(item).collect());
        if let Some(from) = came_from.take() {
            let at = found
                .iter()
                .position(|e| matches!(e, Entry::Dir(p) if *p == from));
            if let Some(at) = at {
                list.select(at);
            }
        }
        let place = dir
            .as_ref()
            .map_or("This computer".to_owned(), |d| d.display().to_string());
        let next = loop {
            let (cols, rows) = super::inner(term.size());
            let filter = match list.filter.is_empty() {
                true => Line::new(Ink::Dim, "  Type to filter"),
                false => Line::new(Ink::Plain, format!("  Filter: {}", list.filter)),
            };
            let head = vec![
                Line::new(Ink::Bright, "  Pick a model file"),
                Line::new(Ink::Dim, format!("  {place}")),
                Line::default(),
                filter,
                Line::default(),
            ];
            let mut body = list.lines(rows.saturating_sub(head.len() + 3), cols);
            if found.iter().all(|e| matches!(e, Entry::Up)) {
                body.push(Line::new(Ink::Dim, "    No folders or .gguf files here."));
            }
            let keys = "↑↓ move   Enter open   ← up a folder   type to filter   Esc back";
            term.draw(&frame(term, head, body, "", keys))?;
            let key = term.key()?;
            if list.handle(key) {
                continue;
            }
            let up = || match &dir {
                Some(d) => (d.parent().map(Path::to_path_buf), Some(d.clone())),
                None => (None, None),
            };
            match key {
                Key::CtrlC => return Err(quit()),
                Key::Esc => return Ok(None),
                Key::Left | Key::Backspace if dir.is_some() => break up(),
                Key::Enter | Key::Right => match list.selected().map(|i| &found[i]) {
                    Some(Entry::Up) => break up(),
                    Some(Entry::Dir(p)) => break (Some(p.clone()), None),
                    Some(Entry::File(p)) if key == Key::Enter => return Ok(Some(p.clone())),
                    _ => {}
                },
                _ => {}
            }
        };
        (dir, came_from) = next;
    }
}
