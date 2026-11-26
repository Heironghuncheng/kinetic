//! Every model on this computer in one list, with where it runs and how
//! fast, the downloads worth having, and a way into any folder.

use std::cmp::Reverse;
use std::io;
use std::path::{Path, PathBuf};

use super::keys::Key;
use super::list::{Item, List};
use super::paint::{Ink, Line};
use super::{folder, frame, quit, Term};
use crate::fit::{Runs, Speeds};
use crate::library::{self, Local};
use crate::system::{self, Gpu, Machine};

const GIB: f64 = (1u64 << 30) as f64;

/// A model worth downloading.
struct Suggestion {
    name: &'static str,
    /// What `kinetic pull` takes.
    spec: &'static str,
    /// Bytes of the file, as Hugging Face lists it.
    size: u64,
    /// GPU memory it is offered with; twice this in free RAM does too.
    memory: u64,
}

const SUGGESTIONS: [Suggestion; 2] = [
    Suggestion {
        name: "Qwen3.5 4B",
        spec: "unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M",
        size: 2_834_975_040,
        memory: 4 << 30,
    },
    Suggestion {
        name: "LFM3 1.7B",
        spec: "LiquidAI/LFM3-1.7B-GGUF:Q4_K_M",
        size: 1_099_627_808,
        memory: 0,
    },
];

/// This computer, as far as where models run on it goes.
pub struct Here {
    pub machine: Machine,
    pub gpus: Vec<Gpu>,
    speeds: Speeds,
}

impl Here {
    pub fn check() -> Here {
        let machine = system::machine();
        let (gpus, _) = system::gpus();
        let speeds = Speeds::measure(&gpus);
        Here {
            machine,
            gpus,
            speeds,
        }
    }

    fn gpu_memory(&self) -> u64 {
        let free = |g: &Gpu| g.memory_free.unwrap_or(g.memory_total);
        self.gpus.iter().map(free).max().unwrap_or(0)
    }

    /// One line: the processor, the memory and the GPU.
    pub fn describe(&self) -> String {
        let gib = |b: u64| b as f64 / GIB;
        let m = &self.machine;
        let gpu = match self.gpus.iter().max_by_key(|g| g.memory_total) {
            Some(g) => match g.memory_free {
                Some(free) => format!("{} ({:.1} GiB free)", g.name, gib(free)),
                None => format!("{} ({:.1} GiB)", g.name, gib(g.memory_total)),
            },
            // a build without CUDA and Vulkan sees no GPU, whatever is there
            None if cfg!(any(feature = "cuda", feature = "vulkan")) => {
                "no GPU this build can use, so models run on the processor".into()
            }
            None => "this build runs models on the processor".into(),
        };
        let free = match gib(m.ram_available) {
            f if f < 0.1 => "almost none available now".to_owned(),
            f => format!("{f:.1} GiB available now"),
        };
        format!(
            "{}, {:.0} GiB memory ({free}), {gpu}",
            m.cpu,
            gib(m.ram_total)
        )
    }

    /// The `--device` values worth offering here, automatic first.
    pub fn devices(&self) -> Vec<&'static str> {
        let mut devices = vec!["auto", "cpu"];
        let has = |api: &str| {
            self.gpus
                .iter()
                .any(|g| g.api == api && g.problem.is_none())
        };
        if cfg!(feature = "cuda") && has("CUDA") {
            devices.push("cuda");
        }
        if cfg!(feature = "vulkan") && has(crate::gpu::VULKAN) {
            devices.push("vulkan");
        }
        devices
    }

    /// Where `kinetic fit` places a model, when that is sure: on a GPU the
    /// model fits, its dense part on the GPU and its experts on the CPU,
    /// or the CPU of a computer with no GPU. Nothing when the engine may
    /// split it some other way.
    fn place(&self, runs: Runs) -> Option<&'static str> {
        match runs.0 {
            "GPU" => Some("GPU"),
            "GPU, experts on CPU" => Some("GPU + CPU"),
            "CPU" if self.gpus.is_empty() => Some("CPU"),
            _ => None,
        }
    }

    pub fn model(&self, l: Local) -> Model {
        let fit = match &l.header {
            Err(_) => Fit::Unsupported("unreadable".into()),
            Ok(h) if h.support() != "yes" => {
                let why = h.support();
                Fit::Unsupported(why.strip_prefix("no, ").unwrap_or(&h.arch).to_owned())
            }
            Ok(_) if crate::is_embedding_model(&l.path).unwrap_or(false) => Fit::Embeddings,
            Ok(_) => match self.speeds.file(&l.path) {
                Ok(runs) => Fit::Runs(self.place(runs)),
                Err(_) => Fit::Unsupported("unreadable".into()),
            },
        };
        let extras = match l.projector {
            Some(_) => vec!["vision"],
            None => Vec::new(),
        };
        Model {
            size: l.header.as_ref().map_or(0, |h| h.size),
            name: l.name,
            store: l.store.label(),
            fit,
            target: Target::Disk(l.path),
            last: false,
            extras,
        }
    }

    /// Every model found and the suggestions not here yet, in the order
    /// [`order`] gives, `last` first.
    pub fn models(&self, last: Option<&Path>) -> Vec<Model> {
        let (drafters, found): (Vec<Local>, Vec<Local>) = library::discover()
            .into_iter()
            .filter(|l| !l.header.as_ref().is_ok_and(|h| h.is_projector()))
            .partition(is_drafter);
        let mut models: Vec<Model> = found.into_iter().map(|l| self.model(l)).collect();
        for d in &drafters {
            let base = drafts_for(&library::model_stem(&d.path));
            let model = models
                .iter_mut()
                .find(|m| letters(&m.name).contains(&letters(&base)));
            if let Some(m) = model {
                m.extras.push("drafter");
            }
        }
        let gpu = self.gpu_memory();
        let ram = self.machine.ram_available;
        let fresh: Vec<Model> = SUGGESTIONS
            .iter()
            .filter(|s| gpu >= s.memory || ram >= 2 * s.memory)
            // the same file under another name is already here
            .filter(|s| !models.iter().any(|m| m.size == s.size))
            .map(|s| Model {
                name: format!(
                    "{} ({})",
                    s.name,
                    s.spec.split(':').next().unwrap_or(s.spec)
                ),
                size: s.size,
                store: "download",
                fit: Fit::Runs(self.place(self.speeds.dense(s.size))),
                target: Target::Hub(s.spec),
                last: false,
                extras: Vec::new(),
            })
            .collect();
        models.extend(fresh);
        for m in &mut models {
            m.last = matches!((&m.target, last), (Target::Disk(p), Some(l)) if p == l);
        }
        order(&mut models);
        models
    }
}

pub enum Target {
    Disk(PathBuf),
    /// Downloaded as `kinetic pull` does when picked.
    Hub(&'static str),
}

/// A drafter, a prediction block or another file that only runs beside
/// a model.
fn is_drafter(l: &Local) -> bool {
    let stem = library::model_stem(&l.path).to_lowercase();
    let arch = l.header.as_ref().map_or("", |h| h.arch.as_str());
    ["dspark-", "dflash-", "mtp-", "draft-"]
        .iter()
        .any(|p| stem.starts_with(p))
        || arch.contains("dflash")
        || arch.ends_with("-assistant")
}

/// The model name in a drafter's file name: `dspark-Inkling-2-Mini-Q8_0`
/// drafts for `Inkling-2-Mini`.
fn drafts_for(stem: &str) -> String {
    let rest = stem.split_once('-').map_or(stem, |(_, r)| r);
    let quant = |p: &str| {
        let p = p.to_lowercase();
        ["q", "iq", "f16", "bf16", "f32"]
            .iter()
            .any(|q| p.starts_with(q))
    };
    match rest.rsplit_once('-') {
        Some((head, last)) if quant(last) => head.to_owned(),
        _ => rest.to_owned(),
    }
}

/// Lowercase letters and digits only, for matching names loosely.
fn letters(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[derive(Debug, PartialEq)]
pub enum Fit {
    /// Runs here, and where when that is sure.
    Runs(Option<&'static str>),
    Embeddings,
    Unsupported(String),
}

impl Fit {
    pub fn text(&self) -> String {
        match self {
            Fit::Runs(on) => on.unwrap_or_default().to_owned(),
            Fit::Embeddings => "makes embeddings, not answers".into(),
            Fit::Unsupported(why) => format!("not supported yet: {why}"),
        }
    }

    pub fn usable(&self) -> bool {
        matches!(self, Fit::Runs(_))
    }
}

pub struct Model {
    pub name: String,
    pub size: u64,
    pub store: &'static str,
    pub fit: Fit,
    pub target: Target,
    /// The model picked last time.
    pub last: bool,
    /// What was found with it, such as its vision projector.
    pub extras: Vec<&'static str>,
}

impl Model {
    /// The app a model came from, by the name people know it by; nothing
    /// for Kinetic's own folder and plain folders.
    pub fn from(&self) -> &'static str {
        match self.store {
            "ollama" => "Ollama",
            "lm studio" => "LM Studio",
            "hugging face" => "Hugging Face",
            "llama.cpp" => "llama.cpp",
            "download" => "to download",
            _ => "",
        }
    }

    pub fn size_text(&self) -> String {
        format!("{:.2} GiB", self.size as f64 / GIB)
    }

    fn item(&self) -> Item {
        let mut fit = self.fit.text();
        if self.last {
            fit = format!("{fit}   last used").trim_start().to_owned();
        }
        let with: String = self.extras.iter().map(|e| format!(" + {e}")).collect();
        let cells = vec![
            format!("{}{with}", self.name),
            self.size_text(),
            self.from().into(),
            fit,
        ];
        let ink = match self.fit.usable() {
            true => Ink::Plain,
            false => Ink::Dim,
        };
        Item::new(cells, ink)
    }
}

/// The last model first, then those that run here, largest first, then
/// downloads, then those that do not run.
pub fn order(models: &mut [Model]) {
    models.sort_by_key(|m| {
        let rank = match (&m.target, m.last, m.fit.usable()) {
            (_, true, _) => 0,
            (Target::Hub(_), ..) => 2,
            (_, _, true) => 1,
            _ => 3,
        };
        (rank, Reverse(m.size))
    });
}

/// Lists the models under `title` and returns the one picked; `None` when
/// the user goes back.
pub fn pick(term: &mut Term, here: &Here, title: &str) -> io::Result<Option<Model>> {
    let mut models = here.models(super::last::recall().as_deref());
    let mut items: Vec<Item> = models.iter().map(Model::item).collect();
    let mut browse = Item::new(
        vec![
            "Browse a folder...".into(),
            String::new(),
            String::new(),
            "any .gguf file".into(),
        ],
        Ink::Muted,
    );
    browse.pinned = true;
    items.push(browse);
    let browse = items.len() - 1;
    let mut list = List::new(items);
    let mut note = String::new();
    let described = here.describe();
    loop {
        let (cols, rows) = super::inner(term.size());
        let filter = match list.filter.is_empty() {
            true => Line::new(Ink::Dim, "  Type to filter"),
            false => Line::new(Ink::Plain, format!("  Filter: {}", list.filter)),
        };
        let head = vec![
            Line::new(Ink::Bright, format!("  {title}")),
            Line::new(Ink::Dim, format!("  {described}")),
            Line::default(),
            filter,
            Line::default(),
        ];
        let body = list.lines(rows.saturating_sub(head.len() + 3), cols);
        let keys = "↑↓ move   Enter pick   type to filter   Esc back";
        term.draw(&frame(term, head, body, &note, keys))?;
        let key = term.key()?;
        if key != Key::Resize {
            note.clear();
        }
        if list.handle(key) {
            continue;
        }
        match key {
            Key::CtrlC => return Err(quit()),
            Key::Esc => return Ok(None),
            Key::Enter => match list.selected() {
                Some(i) if i == browse => {
                    let Some(path) = folder::browse(term)? else {
                        continue;
                    };
                    let model = library::locate(&path).map(|l| here.model(l));
                    match model {
                        Ok(m) if m.fit.usable() => return Ok(Some(m)),
                        Ok(m) => note = format!("{}: {}", m.name, m.fit.text()),
                        Err(e) => note = e.to_string(),
                    }
                }
                Some(i) if models[i].fit.usable() => return Ok(Some(models.swap_remove(i))),
                Some(i) => note = format!("{}: {}", models[i].name, models[i].fit.text()),
                None => {}
            },
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str, size: u64, fit: Fit, hub: bool) -> Model {
        Model {
            name: name.into(),
            size,
            store: "kinetic",
            fit,
            target: match hub {
                true => Target::Hub("org/repo"),
                false => Target::Disk(PathBuf::from(name)),
            },
            last: false,
            extras: Vec::new(),
        }
    }

    #[test]
    fn the_last_model_leads_then_what_runs_largest_first() {
        let runs = || Fit::Runs(Some("GPU"));
        let mut models = vec![
            model("small", 1, runs(), false),
            model("odd", 5, Fit::Unsupported("bert".into()), false),
            model("hub", 3, runs(), true),
            model("big", 4, Fit::Runs(None), false),
            model("used", 2, runs(), false),
        ];
        models[4].last = true;
        order(&mut models);
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["used", "big", "small", "hub", "odd"]);
    }

    #[test]
    fn drafters_name_the_model_they_draft_for() {
        assert_eq!(drafts_for("dspark-Inkling-2-Mini-Q8_0"), "Inkling-2-Mini");
        assert_eq!(drafts_for("mtp-Qwen3.8-35B-A3B-Q8_0"), "Qwen3.8-35B-A3B");
        assert!(letters("Inkling-2-Mini-UD-IQ3_XXS").contains(&letters("Inkling-2-Mini")));
    }

    #[test]
    fn a_place_shows_only_when_it_is_known() {
        assert_eq!(Fit::Runs(Some("GPU + CPU")).text(), "GPU + CPU");
        assert_eq!(Fit::Runs(None).text(), "");
        assert!(!Fit::Embeddings.usable());
    }
}
