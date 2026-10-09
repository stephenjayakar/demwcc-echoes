//! Removal of scratch files that processes which ended without cleaning up left in a work dir.
//!
//! Scratch names carry the creating process id (`<stem>_<pid>_<n>[.ext]`, see `Mwcc::unique`).
//! A file or directory whose process is no longer running and which is older than [`MIN_AGE`]
//! can't be in use: it is deleted. Each work dir is swept once per process, in the background.
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Scratch younger than this is never removed (whatever its process).
pub const MIN_AGE: Duration = Duration::from_secs(30 * 60);

/// Sweep `work`'s scratch directories (`tmp`, `pch`) once per process, on a background thread.
pub fn sweep_once(work: &Path) {
    static DONE: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    if std::env::var_os("MWDEC_NO_SWEEP").is_some() {
        return;
    }
    if !DONE.get_or_init(Default::default).lock().unwrap().insert(work.to_path_buf()) {
        return;
    }
    let work = work.to_path_buf();
    let _ = std::thread::Builder::new().name("mwcc-sweep".into()).spawn(move || {
        for sub in ["tmp", "pch"] {
            sweep_dir(&work.join(sub), MIN_AGE);
        }
    });
}

/// The process id in a scratch name `<stem>_<pid>_<n>[.ext]`.
pub fn scratch_pid(name: &str) -> Option<u32> {
    let base = name.split('.').next()?;
    let mut it = base.rsplitn(3, '_');
    let n = it.next()?;
    let pid = it.next()?;
    let stem = it.next()?;
    if stem.is_empty() || n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) || !pid.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    pid.parse().ok()
}

/// Delete entries of `dir` whose process isn't running and that are older than `min_age`.
/// Returns the number of entries removed.
pub fn sweep_dir(dir: &Path, min_age: Duration) -> usize {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    let me = std::process::id();
    let mut alive: std::collections::HashMap<u32, bool> = Default::default();
    let mut n = 0;
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(pid) = scratch_pid(&name) else { continue };
        if pid == me || *alive.entry(pid).or_insert_with(|| pid_alive(pid)) {
            continue;
        }
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|d| d >= min_age);
        if !old {
            continue;
        }
        let p = e.path();
        let ok = if p.is_dir() { std::fs::remove_dir_all(&p).is_ok() } else { std::fs::remove_file(&p).is_ok() };
        n += ok as usize;
    }
    n
}

/// Whether process `pid` is running (unknown counts as running: nothing is removed then).
#[cfg(windows)]
pub fn pid_alive(pid: u32) -> bool {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut c_void;
        fn GetExitCodeProcess(h: *mut c_void, code: *mut u32) -> i32;
        fn CloseHandle(h: *mut c_void) -> i32;
        fn GetLastError() -> u32;
    }
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const STILL_ACTIVE: u32 = 259;
    const ERROR_INVALID_PARAMETER: u32 = 87;
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            // no such process; any other failure (access denied: it exists) counts as running
            return GetLastError() != ERROR_INVALID_PARAMETER;
        }
        let mut code = 0u32;
        let ok = GetExitCodeProcess(h, &mut code) != 0;
        CloseHandle(h);
        !ok || code == STILL_ACTIVE
    }
}

#[cfg(not(windows))]
pub fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists() || !Path::new("/proc/self").exists()
}

