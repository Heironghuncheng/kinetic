//! `kinetic` alone in a terminal: the home screen, and what its tiles open.

use std::io;
use std::net::TcpListener;

use crate::tui::home::{self, Tile};
use crate::tui::{self, models, Action, Here, Model, Term};
use crate::{console, Result};

pub fn run() -> Result<()> {
    println!("Welcome to Kinetic. Checking this computer...");
    let here = Here::check();
    let settings = crate::config::Settings::load();
    let badge = crate::update::available(&settings, false)
        .map(|v| format!("Kinetic {v} is out   u updates"));
    let mut term = Term::enter()?;
    let mut focus = 0;
    loop {
        match step(&mut term, &here, &mut focus, badge.as_deref()) {
            Err(e) if tui::is_quit(&e) => return Ok(()),
            Err(e) => return Err(e.into()),
            Ok(None) => {}
            Ok(Some((action, model, device))) => {
                term.suspend(|| launch(action, &model, device))?;
            }
        }
    }
}

type Todo = Option<(Action, Model, &'static str)>;

/// The home screen and the screens a tile opens, up to something to run.
fn step(term: &mut Term, here: &Here, focus: &mut usize, badge: Option<&str>) -> io::Result<Todo> {
    let tile = home::show(term, focus, badge)?;
    let direct = |term: &mut Term, title, action| -> io::Result<Todo> {
        let model = models::pick(term, here, title)?;
        Ok(model.map(|m| (action, m, "auto")))
    };
    match tile {
        Tile::Chat => direct(term, "Pick a model to chat with", Action::Chat),
        Tile::Api => direct(term, "Pick a model to serve to your apps", Action::Serve),
        Tile::Bench => direct(term, "Pick a model to measure", Action::Bench),
        Tile::Models => browse(term, here),
        Tile::Doctor => {
            term.suspend(|| {
                if let Err(e) = crate::doctor::run() {
                    crate::report(&*e);
                }
                wait();
            })?;
            Ok(None)
        }
        Tile::Settings => {
            tui::settings::show(term, here)?;
            Ok(None)
        }
        Tile::More => {
            if let Some(args) = tui::commands::pick(term, here)? {
                let chat = args.get(1).is_some_and(|c| c == "chat");
                term.suspend(|| {
                    if let Err(e) = crate::launch(args) {
                        crate::report(&*e);
                    }
                    if !chat {
                        wait();
                    }
                })?;
            }
            Ok(None)
        }
        Tile::Update => {
            term.suspend(|| {
                if let Err(e) = crate::update::run(false) {
                    crate::report(&*e);
                }
                wait();
            })?;
            Ok(None)
        }
    }
}

/// The model list, then what to do with the model picked.
fn browse(term: &mut Term, here: &Here) -> io::Result<Todo> {
    let devices = here.devices();
    loop {
        let Some(model) = models::pick(term, here, "Models on this computer")? else {
            return Ok(None);
        };
        let mut device = 0;
        loop {
            match tui::actions::choose(term, &model, &devices, &mut device)? {
                None => break,
                Some(Action::Info) => {
                    if let tui::Target::Disk(path) = &model.target {
                        term.suspend(|| {
                            if let Err(e) = crate::info::print(path) {
                                crate::report(&*e);
                            }
                            wait();
                        })?;
                    }
                }
                Some(action) => return Ok(Some((action, model, devices[device]))),
            }
        }
    }
}

/// Runs `action` on `model` in the normal terminal, then waits for a key
/// unless it was a chat, which ends when the user leaves it.
fn launch(action: Action, model: &Model, device: &str) {
    let run = || -> Result<()> {
        let path = tui::fetch(model)?;
        let command = match action {
            Action::Chat => "chat",
            Action::Serve => "serve",
            Action::Bench => "bench",
            Action::Info => "info",
        };
        let mut args = vec![
            "kinetic".to_owned(),
            command.to_owned(),
            path.to_string_lossy().into_owned(),
        ];
        if device != "auto" {
            args.extend(["--device".to_owned(), device.to_owned()]);
        }
        if action == Action::Serve {
            let port = free_port();
            args.extend(["--port".to_owned(), port.to_string()]);
            println!("\nPaste this address into OpenCode, Open WebUI or any app that takes an");
            println!("OpenAI-compatible base URL:\n\n    http://127.0.0.1:{port}/v1\n");
            println!(
                "Apps made for Ollama take http://127.0.0.1:{port}. Ctrl+C stops the server.\n"
            );
        }
        crate::launch(args)
    };
    if let Err(e) = run() {
        crate::report(&*e);
    }
    if action != Action::Chat {
        wait();
    }
}

/// 8080, or the first port after it nothing listens on.
fn free_port() -> u16 {
    (8080..8180)
        .find(|&p| TcpListener::bind(("127.0.0.1", p)).is_ok())
        .unwrap_or(8080)
}

fn wait() {
    println!("\nPress any key to go back to Kinetic.");
    console::read_key();
}
