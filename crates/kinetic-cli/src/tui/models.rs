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
/// Slowest decode, in tokens per second, that still reads comfortably.
const READABLE: f64 = 8.0;

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
            None => "no GPU, so models run on the processor".into(),
        };
        format!(
            "{}, {:.0} GiB memory ({:.1} GiB free), {gpu}",
            m.cpu,
            gib(m.ram_total),
            gib(m.ram_available)
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

    pub fn model(&self, l: Local) -> Model {
        let fit = match &l.header {
            Err(_) => Fit::Unsupported("unreadable".into()),
            Ok(h) if h.support() != "yes" => {
                let why = h.support();
                Fit::Unsupported(why.strip_prefix("no, ").unwrap_or(&h.arch).to_owned())
            }
            Ok(_) if crate::is_embedding_model(&l.path).unwrap_or(false) => Fit::Embeddings,
            Ok(h) if h.size > self.machine.ram_available + self.gpu_memory() => Fit::TooBig,
            Ok(_) => self
                .speeds
                .file(&l.path)
                .map_or(Fit::Unsupported("unreadable".into()), Fit::Runs),
        };
        Model {
            size: l.header.as_ref().map_or(0, |h| h.size),
            name: l.name,
            store: l.store.label(),
            fit,
            target: Target::Disk(l.path),
            last: false,
        }
    }

    /// Every model found and the suggestions not here yet, in the order
    /// [`order`] gives, `last` first.
    pub fn models(&self, last: Option<&Path>) -> Vec<Model> {
        let mut models: Vec<Model> = library::discover()
            .into_iter()
            .filter(|l| !l.header.as_ref().is_ok_and(|h| h.is_projector()))
            .map(|l| self.model(l))
            .collect();
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
                fit: Fit::Runs(self.speeds.dense(s.size)),
                target: Target::Hub(s.spec),
                last: false,
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

#[derive(Debug, PartialEq)]
pub enum Fit {
    /// Where it runs and its decode speed range.
    Runs(Runs),
    TooBig,
    Embeddings,
    Unsupported(String),
}

impl Fit {
    pub fn text(&self) -> String {
        match self {
            Fit::Runs((on, Some((lo, hi)))) => format!("{on}, {lo:.0}-{hi:.0} tokens/s"),
            Fit::Runs((on, None)) => (*on).to_owned(),
            Fit::TooBig => "too big for this computer".into(),
            Fit::Embeddings => "makes embeddings, not answers".into(),
            Fit::Unsupported(why) => format!("not supported yet: {why}"),
        }
    }

    pub fn usable(&self) -> bool {
        matches!(self, Fit::Runs(_))
    }

    fn rank(&self) -> u8 {
        match self {
            Fit::Runs((_, Some((lo, hi)))) if (lo + hi) / 2.0 < READABLE => 2,
            Fit::Runs(_) => 1,
            Fit::TooBig => 4,
            Fit::Embeddings | Fit::Unsupported(_) => 5,
        }
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
}

impl Model {
    pub fn size_text(&self) -> String {
        format!("{:.2} GiB", self.size as f64 / GIB)
    }

    fn item(&self) -> Item {
        let mut fit = self.fit.text();
        if self.last {
            fit += "   last used";
        }
        let cells = vec![self.name.clone(), self.size_text(), self.store.into(), fit];
        let ink = match self.fit.usable() {
            true => Ink::Plain,
            false => Ink::Dim,
        };
        Item::new(cells, ink)
    }
}

/// The last model first, then those that run at a readable speed, largest
/// first, then slower ones, downloads, and those that do not run here.
pub fn order(models: &mut [Model]) {
    models.sort_by_key(|m| {
        let rank = match (&m.target, m.last) {
            (_, true) => 0,
            (Target::Hub(_), _) => 3,
            _ => m.fit.rank(),
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
        let (cols, rows) = term.size();
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
        term.draw(&frame(head, body, &note, keys, rows))?;
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
        }
    }

    #[test]
    fn the_last_model_leads_then_what_runs_well_largest_first() {
        let fast = |lo| Fit::Runs(("GPU", Some((lo, lo + 10.0))));
        let mut models = vec![
            model("slow", 9, fast(1.0), false),
            model("odd", 5, Fit::Unsupported("bert".into()), false),
            model("small", 1, fast(90.0), false),
            model("hub", 3, fast(50.0), true),
            model("big", 4, fast(30.0), false),
            model("huge", 99, Fit::TooBig, false),
            model("used", 2, fast(60.0), false),
        ];
        models[6].last = true;
        order(&mut models);
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            ["used", "big", "small", "slow", "hub", "huge", "odd"]
        );
    }

    #[test]
    fn fit_reads_as_a_place_and_a_speed() {
        assert_eq!(
            Fit::Runs(("CPU", Some((9.4, 15.6)))).text(),
            "CPU, 9-16 tokens/s"
        );
        assert_eq!(Fit::Runs(("GPU", None)).text(), "GPU");
        assert!(!Fit::TooBig.usable());
    }
}
