//! The model picked last, offered first next time.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn file() -> Option<PathBuf> {
    Some(crate::library::state_dir()?.join("last-model"))
}

pub fn remember(model: &Path) {
    if let Some(file) = file() {
        let _ = save(&file, model);
    }
}

/// The model picked last, while its file is still there.
pub fn recall() -> Option<PathBuf> {
    read(&file()?)
}

fn save(file: &Path, model: &Path) -> io::Result<()> {
    if let Some(dir) = file.parent() {
        fs::create_dir_all(dir)?;
    }
    let model = std::path::absolute(model)?;
    fs::write(file, model.to_string_lossy().as_bytes())
}

fn read(file: &Path) -> Option<PathBuf> {
    let model = PathBuf::from(fs::read_to_string(file).ok()?.trim());
    model.is_file().then_some(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remembers_a_model_while_its_file_is_there() {
        let dir = std::env::temp_dir().join(format!("kinetic-last-{}", std::process::id()));
        let model = dir.join("model.gguf");
        let file = dir.join("state").join("last-model");
        assert_eq!(read(&file), None);
        fs::create_dir_all(&dir).unwrap();
        fs::write(&model, b"GGUF").unwrap();
        save(&file, &model).unwrap();
        let back = read(&file).unwrap();
        assert_eq!(back, model);
        fs::remove_file(&model).unwrap();
        assert_eq!(read(&file), None);
        let _ = fs::remove_dir_all(&dir);
    }
}
