//! Wall-time profile of named phases, for finding where an eval's time goes.
//!
//! Off unless `MWDEC_PROFILE=<file>` is set; then [`span`] guards and [`add`] calls accumulate
//! (count, seconds) per phase name in the process, and [`dump`] appends one report (tagged with
//! the process id and a caller label) to the file. Phases nest freely: a phase's time includes
//! the phases inside it.
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn target() -> Option<&'static str> {
    static T: OnceLock<Option<String>> = OnceLock::new();
    T.get_or_init(|| std::env::var("MWDEC_PROFILE").ok().filter(|s| !s.is_empty())).as_deref()
}

/// Whether profiling is on (`MWDEC_PROFILE` set).
pub fn enabled() -> bool {
    target().is_some()
}

fn table() -> &'static Mutex<BTreeMap<&'static str, (u64, f64)>> {
    static T: OnceLock<Mutex<BTreeMap<&'static str, (u64, f64)>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Add one occurrence of `name` taking `secs`.
pub fn add(name: &'static str, secs: f64) {
    if enabled() {
        let mut t = table().lock().unwrap();
        let e = t.entry(name).or_default();
        e.0 += 1;
        e.1 += secs;
    }
}

/// Times its scope as phase `name` (nothing when profiling is off).
pub struct Span(Option<(&'static str, Instant)>);

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((name, t)) = self.0 {
            add(name, t.elapsed().as_secs_f64());
        }
    }
}

pub fn span(name: &'static str) -> Span {
    Span(enabled().then(|| (name, Instant::now())))
}

/// Append the accumulated phases to the profile file (`label`: which process / command).
pub fn dump(label: &str, wall_secs: f64) {
    let Some(path) = target() else { return };
    let t = table().lock().unwrap();
    let mut s = format!("== {label} pid {} wall {wall_secs:.1}s\n", std::process::id());
    for (k, (n, secs)) in t.iter() {
        s += &format!("{k:<32} {n:>8} {secs:>10.1}s\n");
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(s.as_bytes());
    }
}
