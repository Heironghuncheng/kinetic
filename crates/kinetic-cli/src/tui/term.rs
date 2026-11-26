//! The terminal while a screen is up: keys one at a time, the alternate
//! screen, and everything put back on exit, on Ctrl+C and on a panic.

use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::{Mutex, Once};
use std::time::{Duration, Instant};

use super::keys::Key;
use super::paint::{self, Colors, Ink, Line};
use crate::render::screen;

static SAVED: Mutex<Option<platform::Mode>> = Mutex::new(None);

pub struct Term {
    /// Columns and rows of the window.
    view: (usize, usize),
    /// Keys read and not yet handed out.
    events: VecDeque<Key>,
    /// The rows on screen, as last written.
    lines: Vec<String>,
    /// The colours the terminal shows.
    output: Colors,
}

impl Term {
    pub fn enter() -> io::Result<Term> {
        static HOOK: Once = Once::new();
        HOOK.call_once(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                restore();
                previous(info);
            }));
        });
        open()?;
        Ok(Term {
            view: size(),
            events: VecDeque::new(),
            lines: Vec::new(),
            output: Colors::detect(),
        })
    }

    /// Columns and rows of the window.
    pub fn size(&self) -> (usize, usize) {
        self.view
    }

    pub fn colors(&self) -> Colors {
        self.output
    }

    /// The next key, or [`Key::Resize`] when the window changed size.
    pub fn key(&mut self) -> io::Result<Key> {
        loop {
            if let Some(key) = self.poll(Duration::from_millis(100))? {
                return Ok(key);
            }
        }
    }

    /// As [`Term::key`], but `None` when no key comes within `wait`.
    pub fn poll(&mut self, wait: Duration) -> io::Result<Option<Key>> {
        let until = Instant::now() + wait;
        loop {
            if let Some(key) = self.events.pop_front() {
                return Ok(Some(key));
            }
            let now = size();
            if now != self.view {
                self.view = now;
                self.lines.clear();
                return Ok(Some(Key::Resize));
            }
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Ok(None);
            }
            self.events
                .extend(platform::read(left.min(Duration::from_millis(100)))?);
        }
    }

    /// Puts `lines` on the screen from its top, cut to the window, writing
    /// only the rows that changed since the last call.
    pub fn draw(&mut self, lines: &[Line]) -> io::Result<()> {
        let (cols, rows) = self.view;
        let fresh: Vec<String> = (0..rows)
            .map(|i| {
                let line = lines
                    .get(i)
                    .map_or_else(String::new, |l| l.render(cols, self.output));
                line + "[K"
            })
            .collect();
        let mut out = String::new();
        if self.lines.len() != fresh.len() {
            out += &paint::start(Ink::Plain, None, self.output);
            out += "[2J";
            self.lines.clear();
        }
        for (i, line) in fresh.iter().enumerate() {
            if self.lines.get(i) != Some(line) {
                out += &format!("[{};1H{line}", i + 1);
            }
        }
        self.lines = fresh;
        if out.is_empty() {
            return Ok(());
        }
        let mut stdout = io::stdout().lock();
        stdout.write_all(out.as_bytes())?;
        stdout.flush()
    }

    /// Leaves the screen for the normal terminal while `f` runs.
    pub fn suspend<T>(&mut self, f: impl FnOnce() -> T) -> io::Result<T> {
        restore();
        let t = f();
        open()?;
        self.events.clear();
        self.lines.clear();
        self.view = size();
        Ok(t)
    }
}

impl Drop for Term {
    fn drop(&mut self) {
        restore();
    }
}

fn open() -> io::Result<()> {
    let mode = platform::raw()?;
    *SAVED.lock().unwrap_or_else(|e| e.into_inner()) = Some(mode);
    let mut out = io::stdout().lock();
    out.write_all(b"\x1b[?1049h\x1b[?25l\x1b[H")?;
    out.flush()
}

/// Back to the screen and the input mode from before; once, however many
/// times it is called.
fn restore() {
    let saved = SAVED.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(mode) = saved {
        let mut out = io::stdout().lock();
        let _ = out.write_all(b"\x1b[0m\x1b[?25h\x1b[?1049l");
        let _ = out.flush();
        platform::put_back(&mode);
    }
}

fn size() -> (usize, usize) {
    screen::size().unwrap_or((80, 24))
}

#[cfg(windows)]
mod platform {
    use std::io;
    use std::time::{Duration, Instant};

    use windows_sys::Win32::System::Console::{
        GetConsoleMode, GetNumberOfConsoleInputEvents, GetStdHandle, ReadConsoleInputW,
        SetConsoleMode, CONSOLE_MODE, ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT,
        ENABLE_PROCESSED_OUTPUT, ENABLE_VIRTUAL_TERMINAL_INPUT, ENABLE_VIRTUAL_TERMINAL_PROCESSING,
        ENABLE_WINDOW_INPUT, INPUT_RECORD, KEY_EVENT, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
        WINDOW_BUFFER_SIZE_EVENT,
    };

    use super::Key;

    pub struct Mode {
        input: CONSOLE_MODE,
        output: CONSOLE_MODE,
    }

    pub fn raw() -> io::Result<Mode> {
        // SAFETY: console calls on this process's own standard handles
        unsafe {
            let (i, o) = (
                GetStdHandle(STD_INPUT_HANDLE),
                GetStdHandle(STD_OUTPUT_HANDLE),
            );
            let (mut input, mut output) = (0, 0);
            if GetConsoleMode(i, &mut input) == 0 || GetConsoleMode(o, &mut output) == 0 {
                return Err(io::Error::last_os_error());
            }
            let keys = ENABLE_LINE_INPUT
                | ENABLE_ECHO_INPUT
                | ENABLE_PROCESSED_INPUT
                | ENABLE_VIRTUAL_TERMINAL_INPUT;
            SetConsoleMode(i, (input & !keys) | ENABLE_WINDOW_INPUT);
            SetConsoleMode(
                o,
                output | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING,
            );
            Ok(Mode { input, output })
        }
    }

    pub fn put_back(mode: &Mode) {
        // SAFETY: as in raw
        unsafe {
            SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), mode.input);
            SetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), mode.output);
        }
    }

    /// The keys pressed within `wait`.
    pub fn read(wait: Duration) -> io::Result<Vec<Key>> {
        let until = Instant::now() + wait;
        // SAFETY: reads this process's own console input into a local buffer
        unsafe {
            let input = GetStdHandle(STD_INPUT_HANDLE);
            loop {
                let mut n = 0;
                if GetNumberOfConsoleInputEvents(input, &mut n) == 0 {
                    return Err(io::Error::last_os_error());
                }
                if n > 0 {
                    break;
                }
                if Instant::now() >= until {
                    return Ok(Vec::new());
                }
                std::thread::sleep(Duration::from_millis(15));
            }
            let mut records: [INPUT_RECORD; 32] = std::mem::zeroed();
            let mut n = 0;
            if ReadConsoleInputW(input, records.as_mut_ptr(), records.len() as u32, &mut n) == 0 {
                return Err(io::Error::last_os_error());
            }
            let mut keys = Vec::new();
            for r in &records[..n as usize] {
                match u32::from(r.EventType) {
                    KEY_EVENT => {
                        let k = r.Event.KeyEvent;
                        if k.bKeyDown == 0 {
                            continue;
                        }
                        if let Some(key) = key(k.wVirtualKeyCode, k.uChar.UnicodeChar) {
                            keys.extend(std::iter::repeat_n(key, k.wRepeatCount.max(1).into()));
                        }
                    }
                    WINDOW_BUFFER_SIZE_EVENT => keys.push(Key::Resize),
                    _ => {}
                }
            }
            Ok(keys)
        }
    }

    fn key(code: u16, unit: u16) -> Option<Key> {
        Some(match (code, unit) {
            (_, 3) => Key::CtrlC,
            (0x26, _) => Key::Up,
            (0x28, _) => Key::Down,
            (0x25, _) => Key::Left,
            (0x27, _) => Key::Right,
            (0x21, _) => Key::PageUp,
            (0x22, _) => Key::PageDown,
            (0x24, _) => Key::Home,
            (0x23, _) => Key::End,
            (0x0d, _) => Key::Enter,
            (0x1b, _) => Key::Esc,
            (0x08, _) => Key::Backspace,
            (0x09, _) => Key::Tab,
            (_, 0) => return None,
            (_, unit) => Key::Char(char::from_u32(unit.into()).filter(|c| !c.is_control())?),
        })
    }
}

#[cfg(unix)]
mod platform {
    use std::io;
    use std::time::Duration;

    use super::super::keys::{self, Key};

    pub struct Mode(libc::termios);

    pub fn raw() -> io::Result<Mode> {
        // SAFETY: tcgetattr fills the zeroed termios; the copy changed is ours
        unsafe {
            let mut saved: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut saved) != 0 {
                return Err(io::Error::last_os_error());
            }
            let mut raw = saved;
            raw.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG | libc::IEXTEN);
            raw.c_iflag &= !(libc::IXON | libc::ICRNL);
            raw.c_cc[libc::VMIN] = 1;
            raw.c_cc[libc::VTIME] = 0;
            libc::tcsetattr(0, libc::TCSANOW, &raw);
            Ok(Mode(saved))
        }
    }

    pub fn put_back(mode: &Mode) {
        // SAFETY: puts back the settings tcgetattr read
        unsafe { libc::tcsetattr(0, libc::TCSANOW, &mode.0) };
    }

    fn ready(wait: Duration) -> bool {
        let mut fd = libc::pollfd {
            fd: 0,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: polls one descriptor described by a local pollfd
        unsafe { libc::poll(&mut fd, 1, wait.as_millis() as libc::c_int) > 0 }
    }

    fn bytes() -> io::Result<Vec<u8>> {
        let mut buf = [0u8; 256];
        // SAFETY: reads at most buf.len() bytes into buf
        let n = unsafe { libc::read(0, buf.as_mut_ptr().cast(), buf.len()) };
        match n {
            1.. => Ok(buf[..n as usize].to_vec()),
            _ => Err(io::ErrorKind::UnexpectedEof.into()),
        }
    }

    /// The keys pressed within `wait`.
    pub fn read(wait: Duration) -> io::Result<Vec<Key>> {
        if !ready(wait) {
            return Ok(Vec::new());
        }
        let mut got = bytes()?;
        // an escape alone may be the start of an arrow still on its way
        if got == [0x1b] && ready(Duration::from_millis(30)) {
            got.extend(bytes()?);
        }
        Ok(keys::parse(&got))
    }
}
