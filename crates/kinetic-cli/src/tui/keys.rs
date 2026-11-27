//! Keys as the screens see them, read from the bytes a Unix terminal sends.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Up,
    Down,
    Left,
    Right,
    PageUp,
    PageDown,
    Home,
    End,
    Enter,
    Esc,
    Backspace,
    Tab,
    CtrlC,
    Char(char),
    /// The window changed size.
    Resize,
}

/// The keys in `bytes`, as a terminal in raw mode sends them; a lone escape
/// is the Esc key, and sequences this does not know are dropped.
#[cfg_attr(windows, allow(dead_code))]
pub fn parse(bytes: &[u8]) -> Vec<Key> {
    let mut keys = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let (key, used) = one(&bytes[i..]);
        keys.extend(key);
        i += used;
    }
    keys
}

fn one(b: &[u8]) -> (Option<Key>, usize) {
    match b[0] {
        0x1b => match b.get(1) {
            Some(b'[' | b'O') => escape(b),
            _ => (Some(Key::Esc), 1),
        },
        b'\r' | b'\n' => (Some(Key::Enter), 1),
        0x7f | 0x08 => (Some(Key::Backspace), 1),
        b'\t' => (Some(Key::Tab), 1),
        0x03 => (Some(Key::CtrlC), 1),
        c if c < 0x20 => (None, 1),
        _ => {
            let len = match b[0] {
                0xf0.. => 4,
                0xe0.. => 3,
                0xc0.. => 2,
                _ => 1,
            };
            let len = len.min(b.len());
            let c = std::str::from_utf8(&b[..len])
                .ok()
                .and_then(|s| s.chars().next());
            (c.map(Key::Char), len)
        }
    }
}

/// `ESC [` or `ESC O`, parameters, then the final byte.
fn escape(b: &[u8]) -> (Option<Key>, usize) {
    let Some(end) = b[2..].iter().position(|c| (0x40..=0x7e).contains(c)) else {
        return (None, b.len());
    };
    let params = &b[2..2 + end];
    let key = match (b[2 + end], params) {
        (b'A', _) => Some(Key::Up),
        (b'B', _) => Some(Key::Down),
        (b'C', _) => Some(Key::Right),
        (b'D', _) => Some(Key::Left),
        (b'H', _) => Some(Key::Home),
        (b'F', _) => Some(Key::End),
        (b'~', b"1" | b"7") => Some(Key::Home),
        (b'~', b"4" | b"8") => Some(Key::End),
        (b'~', b"5") => Some(Key::PageUp),
        (b'~', b"6") => Some(Key::PageDown),
        _ => None,
    };
    (key, 3 + end)
}

#[cfg(test)]
mod tests {
    use super::{parse, Key};

    #[test]
    fn reads_arrows_text_and_a_lone_escape() {
        assert_eq!(
            parse(b"\x1b[A\x1bOBq\r\x1b"),
            [Key::Up, Key::Down, Key::Char('q'), Key::Enter, Key::Esc]
        );
        assert_eq!(
            parse(b"\x1b[5~\x1b[1;5C\x7f\x03"),
            [Key::PageUp, Key::Right, Key::Backspace, Key::CtrlC]
        );
        assert_eq!(parse("ż".as_bytes()), [Key::Char('ż')]);
        assert_eq!(parse(b"\x1b[200~a"), [Key::Char('a')]);
    }
}
