//! `kinetic update`: the newest release for this computer and build,
//! checked against the release's SHA256SUMS, in place of this binary; and
//! the daily check that says, in one line, when there is one.

use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Result;

const REPO: &str = "Veloresearch/kinetic";
const DAY: u64 = 24 * 60 * 60;

/// Where release lists and files come from: GitHub, or `KINETIC_RELEASES`
/// for a mirror, serving `latest`, `tags/<tag>` and `download/<tag>/<file>`.
fn urls(tag: &str) -> (String, String) {
    match std::env::var("KINETIC_RELEASES") {
        Ok(base) if !base.is_empty() => (format!("{base}/{tag}"), format!("{base}/download")),
        _ => (
            format!("https://api.github.com/repos/{REPO}/releases/{tag}"),
            format!("https://github.com/{REPO}/releases/download"),
        ),
    }
}

fn curl(args: &[&str], stdout: Stdio) -> Result<Vec<u8>> {
    let out = Command::new(crate::pull::curl().ok_or("updating needs curl on PATH")?)
        .args(["-fL", "-A", "kinetic-update"])
        .args(args)
        .stdout(stdout)
        .stderr(if std::io::stderr().is_terminal() {
            Stdio::inherit()
        } else {
            Stdio::null()
        })
        .output()?;
    if !out.status.success() {
        return Err(format!("curl failed with code {}", out.status.code().unwrap_or(-1)).into());
    }
    Ok(out.stdout)
}

/// The tag of the newest release, or of the nightly prerelease.
fn newest(nightly: bool, wait: &str) -> Result<String> {
    let which = if nightly { "tags/nightly" } else { "latest" };
    let body = curl(&["-sS", "-m", wait, &urls(which).0], Stdio::piped())?;
    let release: serde_json::Value = serde_json::from_slice(&body)?;
    let tag = release["tag_name"]
        .as_str()
        .ok_or("the release has no tag")?;
    Ok(tag.trim_start_matches('v').to_owned())
}

/// The numbers of a version such as `0.1` or `0.1.2 (nightly)`.
fn numbers(version: &str) -> Vec<u64> {
    let version = version.split_whitespace().next().unwrap_or_default();
    let mut n: Vec<u64> = version.split('.').map_while(|p| p.parse().ok()).collect();
    while n.last() == Some(&0) {
        n.pop();
    }
    n
}

fn newer(candidate: &str, current: &str) -> bool {
    let (a, b) = (numbers(candidate), numbers(current));
    !a.is_empty() && a > b
}

/// The archive of this platform and build.
fn archive(version: &str) -> String {
    let (target, ext) = if cfg!(windows) {
        ("windows-x86_64", "zip")
    } else if cfg!(target_os = "macos") {
        ("macos-arm64", "tar.gz")
    } else {
        ("linux-x86_64", "tar.gz")
    };
    let variant = if cfg!(target_os = "macos") {
        "metal"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "cpu"
    };
    format!("kinetic-{version}-{target}-{variant}.{ext}")
}

/// The checksum SHA256SUMS lists for `name`.
fn listed(sums: &str, name: &str) -> Option<String> {
    sums.lines().find_map(|line| {
        let (hash, file) = line.split_once(char::is_whitespace)?;
        (file.trim().trim_start_matches('*') == name).then(|| hash.to_lowercase())
    })
}

/// `kinetic update`.
pub fn run(nightly: bool) -> Result<()> {
    let current = env!("KINETIC_VERSION");
    let version = newest(nightly, "30")?;
    if !nightly && !newer(&version, current) {
        println!("Kinetic {current} is the newest release.");
        return Ok(());
    }
    let tag = if nightly {
        version.clone()
    } else {
        format!("v{version}")
    };
    let name = archive(&version);
    let base = format!("{}/{tag}", urls("latest").1);
    let tmp = std::env::temp_dir().join(format!("kinetic-update-{}", std::process::id()));
    fs::create_dir_all(&tmp)?;
    let result = (|| -> Result<PathBuf> {
        println!("Downloading Kinetic {version}");
        let file = tmp.join(&name);
        curl(
            &[
                "-#",
                "-o",
                &file.to_string_lossy(),
                &format!("{base}/{name}"),
            ],
            Stdio::inherit(),
        )?;
        let sums = curl(&["-sS", &format!("{base}/SHA256SUMS")], Stdio::piped())?;
        let expected = listed(&String::from_utf8_lossy(&sums), &name)
            .ok_or_else(|| format!("SHA256SUMS does not list {name}"))?;
        let actual = crate::pull::sha256(&file)?;
        if actual != expected {
            return Err(
                format!("checksum mismatch for {name}: expected {expected}, got {actual}").into(),
            );
        }
        let status = Command::new(tar())
            .arg("-xf")
            .arg(&file)
            .arg("-C")
            .arg(&tmp)
            .status()?;
        if !status.success() {
            return Err(format!("could not unpack {name}").into());
        }
        let exe = format!("kinetic{}", std::env::consts::EXE_SUFFIX);
        find(&tmp, &exe).ok_or_else(|| format!("{name} holds no {exe}").into())
    })();
    let installed = result.and_then(|new| replace(&new));
    let _ = fs::remove_dir_all(&tmp);
    installed?;
    remember(&version);
    println!("Kinetic {version} is installed; it runs from the next start.");
    Ok(())
}

/// Windows' own tar, which unpacks zip files, ahead of any other on PATH.
fn tar() -> PathBuf {
    match std::env::var_os("SystemRoot") {
        Some(root) if cfg!(windows) => Path::new(&root).join("System32").join("tar.exe"),
        _ => PathBuf::from("tar"),
    }
}

fn find(dir: &Path, name: &str) -> Option<PathBuf> {
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|n| n == name) {
            return Some(path);
        }
    }
    None
}

/// The running binary moved aside, as Windows does not let it be
/// overwritten, so the new one can take its place.
fn aside(exe: &Path) -> PathBuf {
    let mut name = exe.as_os_str().to_owned();
    name.push(".old");
    PathBuf::from(name)
}

fn replace(new: &Path) -> Result<()> {
    let exe = std::env::current_exe()?;
    let old = aside(&exe);
    let _ = fs::remove_file(&old);
    fs::rename(&exe, &old).map_err(|e| format!("cannot move {} aside: {e}", exe.display()))?;
    if let Err(e) = fs::copy(new, &exe) {
        let _ = fs::rename(&old, &exe);
        return Err(format!("cannot write {}: {e}", exe.display()).into());
    }
    Ok(())
}

/// Removes the binary an update moved aside, once it no longer runs.
pub fn tidy() {
    if let Ok(exe) = std::env::current_exe() {
        let _ = fs::remove_file(aside(&exe));
    }
}

fn state() -> Option<PathBuf> {
    Some(crate::library::state_dir()?.join("update-check"))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// When the check last ran and the newest release it saw.
fn last_check() -> Option<(u64, String)> {
    let text = fs::read_to_string(state()?).ok()?;
    let (when, version) = text.trim().split_once(' ')?;
    Some((when.parse().ok()?, version.to_owned()))
}

fn remember(version: &str) {
    write_check(now(), version);
}

fn write_check(when: u64, version: &str) {
    if let Some(file) = state() {
        let tmp = file.with_extension("tmp");
        let _ = file
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|()| fs::write(&tmp, format!("{when} {version}")))
            .and_then(|()| fs::rename(&tmp, &file));
    }
}

/// Whether the daily check is off, by `KINETIC_NO_UPDATE_CHECK`.
fn off() -> bool {
    std::env::var_os("KINETIC_NO_UPDATE_CHECK").is_some_and(|v| !v.is_empty() && v != "0")
}

/// The newer release the last check saw, if any; asks GitHub again, once,
/// in the background, when that check is a day old. Nothing when off.
pub fn available() -> Option<String> {
    if off() {
        return None;
    }
    let seen = last_check();
    if seen
        .as_ref()
        .is_none_or(|(when, _)| now().saturating_sub(*when) >= DAY)
    {
        // a run that ends before the answer asks again in an hour, not at once
        let known = seen.as_ref().map_or("-", |(_, v)| v.as_str());
        write_check(now().saturating_sub(DAY - 3600), known);
        std::thread::spawn(|| {
            if let Ok(version) = newest(false, "5") {
                remember(&version);
            }
        });
    }
    let (_, version) = seen?;
    newer(&version, env!("KINETIC_VERSION")).then_some(version)
}

/// One line on stderr when a newer release is out, for a person at a
/// terminal only.
pub fn notice(args: &[String]) {
    let terminal = std::io::stdin().is_terminal()
        && std::io::stdout().is_terminal()
        && std::io::stderr().is_terminal();
    let json = args.iter().any(|a| a == "--json");
    let updating = args.get(1).is_some_and(|a| a == "update");
    if terminal && !json && !updating {
        if let Some(version) = available() {
            eprintln!("Kinetic {version} is out: run kinetic update");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_by_their_numbers() {
        assert!(newer("0.1.1", "0.1"));
        assert!(newer("0.2", "0.1.9 (nightly 2026-11-20 abc)"));
        assert!(!newer("0.1.0", "0.1"));
        assert!(!newer("0.1", "0.1.1"));
        assert!(!newer("nightly", "0.1"));
    }

    #[test]
    fn checksums_are_read_by_file_name() {
        let sums = "abc123  kinetic-0.1.1-linux-x86_64-cpu.tar.gz\nDEF456 *kinetic-0.1.1-windows-x86_64-cuda.zip\n";
        assert_eq!(
            listed(sums, "kinetic-0.1.1-windows-x86_64-cuda.zip").as_deref(),
            Some("def456")
        );
        assert_eq!(
            listed(sums, "kinetic-0.1.1-linux-x86_64-cpu.tar.gz").as_deref(),
            Some("abc123")
        );
        assert_eq!(listed(sums, "other.zip"), None);
    }
}
