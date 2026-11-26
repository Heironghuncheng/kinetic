//! Settings saved from the Settings screen, which `chat`, `serve`, `run`
//! and `bench` take as if they had been typed after the command; a flag
//! that is typed wins over the saved one.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// A flag, its short form, and the commands that take it.
pub struct Flag {
    pub long: &'static str,
    short: Option<&'static str>,
    commands: &'static [&'static str],
}

const ALL: &[&str] = &["chat", "serve", "run", "bench"];
const LOADED: &[&str] = &["chat", "serve", "run"];
const WRITING: &[&str] = &["chat", "run"];

pub const FLAGS: [Flag; 13] = [
    flag("--device", None, ALL),
    flag("--gpus", None, ALL),
    flag("--ctx-size", Some("-c"), LOADED),
    flag("--threads", Some("-t"), ALL),
    flag("--parallel", None, &["serve"]),
    flag("--temp", None, WRITING),
    flag("--top-p", None, WRITING),
    flag("--top-k", None, WRITING),
    flag("--min-p", None, WRITING),
    flag("--repeat-penalty", None, WRITING),
    flag("--n-predict", Some("-n"), WRITING),
    flag("--reasoning", None, WRITING),
    flag("--off", None, ALL),
];

const fn flag(
    long: &'static str,
    short: Option<&'static str>,
    commands: &'static [&'static str],
) -> Flag {
    Flag {
        long,
        short,
        commands,
    }
}

/// Saved values by flag, in the order of [`FLAGS`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Settings(pub Vec<(String, String)>);

impl Settings {
    pub fn get(&self, flag: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(f, _)| f == flag)
            .map(|(_, v)| v.as_str())
    }

    /// Sets `flag` to `value`; an empty value goes back to the default.
    pub fn set(&mut self, flag: &str, value: &str) {
        self.0.retain(|(f, _)| f != flag);
        if !value.is_empty() {
            self.0.push((flag.to_owned(), value.to_owned()));
            let at = |f: &str| FLAGS.iter().position(|x| x.long == f);
            self.0.sort_by_key(|(f, _)| at(f));
        }
    }

    /// Puts the saved sampling settings into `s`.
    pub fn sample(&self, s: &mut kinetic_server::Sampling) {
        let get = |flag| self.get(flag).and_then(|v| v.parse::<f32>().ok());
        if let Some(v) = get("--temp") {
            s.temperature = v;
        }
        if let Some(v) = get("--top-p") {
            s.top_p = v;
        }
        if let Some(v) = self.get("--top-k").and_then(|v| v.parse().ok()) {
            s.top_k = v;
        }
        if let Some(v) = get("--min-p") {
            s.min_p = v;
        }
        if let Some(v) = get("--repeat-penalty") {
            s.repeat_penalty = v;
        }
    }

    fn parse(text: &str) -> Settings {
        let mut s = Settings::default();
        for line in text.lines() {
            if let Some((flag, value)) = line.trim().split_once('=') {
                if FLAGS.iter().any(|f| f.long == flag.trim()) {
                    s.set(flag.trim(), value.trim());
                }
            }
        }
        s
    }

    fn text(&self) -> String {
        self.0.iter().map(|(f, v)| format!("{f}={v}\n")).collect()
    }

    pub fn load() -> Settings {
        file()
            .and_then(|f| fs::read_to_string(f).ok())
            .map_or_else(Settings::default, |t| Settings::parse(&t))
    }

    pub fn save(&self) -> io::Result<()> {
        let file = file().ok_or_else(|| io::Error::other("no folder for settings"))?;
        save_to(&file, &self.text())
    }

    /// The flags this adds to `command`, as they would be typed.
    pub fn flags_for(&self, command: &str) -> Vec<String> {
        self.0
            .iter()
            .filter(|(f, _)| {
                FLAGS
                    .iter()
                    .any(|x| x.long == f && x.commands.contains(&command))
            })
            .flat_map(|(f, v)| [f.clone(), v.clone()])
            .collect()
    }

    /// `args` with the saved flags their command takes and they do not
    /// already give inserted after the command.
    pub fn apply(&self, mut args: Vec<String>) -> Vec<String> {
        let Some(command) = args.get(1).cloned() else {
            return args;
        };
        let typed = |flag: &Flag| {
            args[2..].iter().any(|a| {
                a == flag.long
                    || a.starts_with(&format!("{}=", flag.long))
                    || flag
                        .short
                        .is_some_and(|s| a.starts_with(s) && !a.starts_with("--"))
            })
        };
        let compare = args
            .iter()
            .any(|a| a == "--compare" || a.starts_with("--compare="));
        let mut added = Vec::new();
        for (f, v) in &self.0 {
            let Some(flag) = FLAGS.iter().find(|x| x.long == f) else {
                continue;
            };
            // --off does not apply to --compare
            let skip = compare && flag.long == "--off";
            if flag.commands.contains(&command.as_str()) && !typed(flag) && !skip {
                added.extend([f.clone(), v.clone()]);
            }
        }
        args.splice(2..2, added);
        args
    }
}

fn file() -> Option<PathBuf> {
    Some(crate::library::state_dir()?.join("settings"))
}

fn save_to(file: &Path, text: &str) -> io::Result<()> {
    if let Some(dir) = file.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(file, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn saved_flags_go_after_the_command_unless_typed() {
        let s = Settings::parse(
            "--temp=0.7\n--ctx-size=8192\n--off=rad1,torque\nnonsense\n--bogus=1\n",
        );
        assert_eq!(
            s.apply(args("kinetic chat m.gguf --temp 0.2")),
            args("kinetic chat --ctx-size 8192 --off rad1,torque m.gguf --temp 0.2")
        );
        assert_eq!(
            s.apply(args("kinetic serve m.gguf -c4096")),
            args("kinetic serve --off rad1,torque m.gguf -c4096")
        );
        assert_eq!(
            s.apply(args("kinetic bench m.gguf --compare")),
            args("kinetic bench m.gguf --compare")
        );
        assert_eq!(s.apply(args("kinetic models")), args("kinetic models"));
        assert_eq!(s.apply(args("kinetic")), args("kinetic"));
    }

    #[test]
    fn settings_read_back_as_written_in_flag_order() {
        let mut s = Settings::default();
        s.set("--temp", "0.5");
        s.set("--device", "cpu");
        s.set("--temp", "");
        s.set("--top-k", "40");
        assert_eq!(s.text(), "--device=cpu\n--top-k=40\n");
        assert_eq!(Settings::parse(&s.text()), s);
        assert_eq!(s.flags_for("serve"), args("--device cpu"));
    }
}
