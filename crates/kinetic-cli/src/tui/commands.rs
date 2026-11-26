//! Every command and every option it takes, set with the arrows and typing
//! instead of a command line, which is shown as it grows.

use std::io;

use clap::{Arg, CommandFactory};

use super::keys::Key;
use super::list::{Item, List};
use super::paint::{Ink, Line};
use super::{frame, models, quit, Here, Term};

/// One option of a command, as the form holds it.
struct Field {
    arg: Arg,
    /// What is set; empty for the command's default.
    value: String,
}

impl Field {
    fn switch(&self) -> bool {
        !self.arg.get_action().takes_values()
    }

    fn choices(&self) -> Vec<String> {
        self.arg
            .get_possible_values()
            .iter()
            .filter(|v| !v.is_hide_set())
            .map(|v| v.get_name().to_owned())
            .collect()
    }

    fn label(&self) -> String {
        match (self.arg.get_long(), self.arg.is_positional()) {
            (Some(long), _) => format!("--{long}"),
            (None, true) => self
                .arg
                .get_value_names()
                .and_then(|n| n.first())
                .map_or(self.arg.get_id().to_string(), |n| n.to_string()),
            (None, false) => self.arg.get_id().to_string(),
        }
    }

    fn shown(&self, editing: bool) -> String {
        let cursor = if editing { "▏" } else { "" };
        match (self.switch(), self.value.as_str()) {
            (true, "") => "[   ]".into(),
            (true, _) => "[ ✓ ]".into(),
            (false, "") if !self.choices().is_empty() => "‹ default ›".into(),
            (false, v) if !self.choices().is_empty() => format!("‹ {v} ›"),
            (false, "") => format!("{cursor}default"),
            (false, v) => format!("{v}{cursor}"),
        }
    }

    /// The next choice `by` steps away, the first being the default.
    fn step(&mut self, by: isize) {
        if self.switch() {
            self.value = if self.value.is_empty() {
                "on".into()
            } else {
                String::new()
            };
            return;
        }
        let mut all = vec![String::new()];
        all.extend(self.choices());
        if all.len() > 1 {
            let at = all.iter().position(|v| *v == self.value).unwrap_or(0) as isize;
            self.value = all[(at + by).rem_euclid(all.len() as isize) as usize].clone();
        }
    }

    fn is_model(&self) -> bool {
        self.arg
            .get_value_names()
            .and_then(|n| n.first())
            .is_some_and(|n| n.as_str() == "MODEL")
    }
}

/// The command line the form stands for, as `kinetic` would be given it.
fn command_line(name: &str, fields: &[Field]) -> Vec<String> {
    let mut args = vec!["kinetic".to_owned(), name.to_owned()];
    for f in fields
        .iter()
        .filter(|f| f.arg.is_positional() && !f.value.is_empty())
    {
        args.push(f.value.clone());
    }
    for f in fields
        .iter()
        .filter(|f| !f.arg.is_positional() && !f.value.is_empty())
    {
        args.push(f.label());
        if !f.switch() {
            args.push(f.value.clone());
        }
    }
    args
}

fn quoted(args: &[String]) -> String {
    args.iter()
        .map(|a| match a.contains(' ') || a.is_empty() {
            true => format!("\"{a}\""),
            false => a.clone(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Lets the user pick a command and set its options; the command line to
/// run, or `None` when they go back.
pub fn pick(term: &mut Term, here: &Here) -> io::Result<Option<Vec<String>>> {
    let cli = crate::Cli::command();
    let subs: Vec<_> = cli.get_subcommands().filter(|c| !c.is_hide_set()).collect();
    let mut list = List::new(
        subs.iter()
            .map(|c| {
                let about = c.get_about().map(|a| a.to_string()).unwrap_or_default();
                Item::new(vec![c.get_name().to_owned(), about], Ink::Plain)
            })
            .collect(),
    );
    loop {
        let (cols, rows) = term.size();
        let head = vec![
            Line::new(Ink::Bright, "  Every command"),
            Line::new(
                Ink::Dim,
                "  Each opens with its options to set; the command line shows below them.",
            ),
            Line::default(),
        ];
        let body = list.lines(rows.saturating_sub(head.len() + 3), cols);
        let keys = "↑↓ move   Enter open   type to filter   Esc back";
        term.draw(&frame(head, body, "", keys, rows))?;
        let key = term.key()?;
        if list.handle(key) {
            continue;
        }
        match key {
            Key::CtrlC => return Err(quit()),
            Key::Esc => return Ok(None),
            Key::Enter => {
                if let Some(i) = list.selected() {
                    if let Some(args) = form(term, here, subs[i])? {
                        return Ok(Some(args));
                    }
                }
            }
            _ => {}
        }
    }
}

fn form(term: &mut Term, here: &Here, cmd: &clap::Command) -> io::Result<Option<Vec<String>>> {
    let name = cmd.get_name().to_owned();
    let mut fields: Vec<Field> = cmd
        .get_arguments()
        .filter(|a| !a.is_hide_set() && !matches!(a.get_id().as_str(), "help" | "version"))
        .map(|a| Field {
            arg: a.clone(),
            value: String::new(),
        })
        .collect();
    fields.sort_by_key(|f| !f.arg.is_positional());
    let mut at = 0;
    let mut note = String::new();
    loop {
        let (cols, rows) = term.size();
        let run = 0;
        let mut items = vec![Item::new(
            vec![format!("Run kinetic {name}"), String::new(), String::new()],
            Ink::Accent,
        )];
        items.extend(fields.iter().enumerate().map(|(i, f)| {
            let help = f.arg.get_help().map(|h| h.to_string()).unwrap_or_default();
            let ink = if f.value.is_empty() {
                Ink::Dim
            } else {
                Ink::Plain
            };
            Item::new(vec![f.label(), f.shown(at == i + 1), help], ink)
        }));
        let mut list = List::new(items);
        list.select(at);
        let line = quoted(&command_line(&name, &fields));
        let head = vec![
            Line::new(Ink::Bright, format!("  kinetic {name}")),
            Line::new(
                Ink::Dim,
                format!(
                    "  {}",
                    cmd.get_about().map(|a| a.to_string()).unwrap_or_default()
                ),
            ),
            Line::default(),
        ];
        let mut body = list.lines(rows.saturating_sub(head.len() + 6), cols);
        body.push(Line::default());
        body.push(Line::new(Ink::Dim, "  Runs as:"));
        body.push(Line::new(Ink::Plain, format!("  {line}")));
        let keys = "↑↓ move   ←→ change   type a value   Enter run or pick   Esc back";
        term.draw(&frame(head, body, &note, keys, rows))?;
        let key = term.key()?;
        if key != Key::Resize {
            note.clear();
        }
        let field = at.checked_sub(1).and_then(|i| fields.get_mut(i));
        match (key, field) {
            (Key::CtrlC, _) => return Err(quit()),
            (Key::Esc, _) => return Ok(None),
            (Key::Up, _) => at = at.saturating_sub(1),
            (Key::Down, _) => at = (at + 1).min(fields.len()),
            (Key::PageUp, _) => at = at.saturating_sub(10),
            (Key::PageDown, _) => at = (at + 10).min(fields.len()),
            (Key::Enter, None) if at == run => {
                return Ok(Some(command_line(&name, &fields)));
            }
            (Key::Enter, Some(f)) if f.is_model() => {
                if let Some(m) = models::pick(term, here, "Pick the model")? {
                    match &m.target {
                        super::Target::Disk(p) => f.value = p.to_string_lossy().into_owned(),
                        super::Target::Hub(spec) => {
                            note = format!("{} is not downloaded yet: kinetic pull {spec}", m.name)
                        }
                    }
                }
            }
            (Key::Enter | Key::Right | Key::Char(' '), Some(f))
                if f.switch() || !f.choices().is_empty() =>
            {
                f.step(1)
            }
            (Key::Left, Some(f)) if f.switch() || !f.choices().is_empty() => f.step(-1),
            (Key::Enter, Some(_)) => at = 0,
            (Key::Backspace, Some(f)) => {
                f.value.pop();
            }
            (Key::Char(c), Some(f)) if !f.switch() && f.choices().is_empty() => f.value.push(c),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    fn fields(name: &str) -> Vec<Field> {
        let cli = crate::Cli::command();
        let cmd = cli.find_subcommand(name).unwrap();
        cmd.get_arguments()
            .map(|a| Field {
                arg: a.clone(),
                value: String::new(),
            })
            .collect()
    }

    #[test]
    fn every_command_has_a_form_that_reads_back_as_its_command_line() {
        let cli = crate::Cli::command();
        for cmd in cli.get_subcommands() {
            let fields = fields(cmd.get_name());
            assert_eq!(
                command_line(cmd.get_name(), &fields),
                ["kinetic", cmd.get_name()]
            );
        }
        let mut chat = fields("chat");
        let set = |fs: &mut Vec<Field>, label: &str, v: &str| {
            fs.iter_mut().find(|f| f.label() == label).unwrap().value = v.into();
        };
        set(&mut chat, "MODEL", "m.gguf");
        set(&mut chat, "--temp", "0.5");
        chat.iter_mut()
            .find(|f| f.label() == "--device")
            .unwrap()
            .step(1);
        chat.iter_mut()
            .find(|f| f.label() == "--mlock")
            .unwrap()
            .step(1);
        let line = command_line("chat", &chat);
        assert_eq!(line[..3], ["kinetic", "chat", "m.gguf"]);
        assert!(quoted(&line).contains("--temp 0.5"));
        assert!(quoted(&line).contains("--device auto"));
        assert!(quoted(&line).contains("--mlock"));
        let parsed = crate::Cli::try_parse_from(&line);
        assert!(parsed.is_ok(), "{}", quoted(&line));
    }
}
