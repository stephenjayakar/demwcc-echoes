//! First drafts in a child process (`mwdec draft-server`), so one pathological function (an IR
//! explosion in the lifter, a runaway inline match, a stack overflow) is reported as
//! `oom/too-big` / `crash` / `timeout` for that function instead of killing the whole eval.
//!
//! The child gets its own (nested) job-object cap (`DRAFT_MEM_MB`, below the parent's), keeps the
//! inputs of the current unit only (TypeDb, inline library), and answers one JSON line per
//! request on stdout. The parent no longer holds any TypeDb or inline library.
use super::search_cmds::{draft_variant, draft_with, unit_inputs, Compilers, NoDraft, UnitInputs};
use super::{find_unit, load_project};
use anyhow::Result;
use mwdec_core::*;
use mwdec_project::Project;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Memory cap of the draft child (MB), below the parent's cap.
pub const DRAFT_MEM_MB: u64 = 1536;

/// Outcome of a draft request.
#[derive(Debug, Clone, Default)]
pub struct DraftReply {
    /// "ok", "hdr-inline", "implicit", "lift-err", "missing", "unit-err", "oom/too-big", "crash", "timeout".
    pub status: String,
    /// Draft (with folded header inlines where enabled).
    pub src: Option<String>,
    /// Draft without folded inlines, when it differs.
    pub plain: Option<String>,
    /// Kind of a function that can't exist as standalone source.
    pub implicit: Option<String>,
    pub error: Option<String>,
}

impl DraftReply {
    fn err(status: &str, error: impl Into<String>) -> DraftReply {
        DraftReply { status: status.into(), error: Some(error.into()), ..Default::default() }
    }
}

/// Draft one function in-process (the child's work; also the fallback without a child).
pub fn draft_local(ui: &UnitInputs, symbol: &str, include_implicit: bool) -> DraftReply {
    let Some(f) = mwdec_obj::find_function(&ui.target, symbol) else {
        return DraftReply { status: "missing".into(), ..Default::default() };
    };
    let mut r = DraftReply::default();
    if let Some(db) = &ui.db {
        if let mwdec_project::standalone::Standalone::Implicit(k) = mwdec_project::standalone::standalone(&mwdec_lift::sig::sig_of(&f.name, Some(db)), db) {
            r.implicit = Some(k.to_string());
        }
    }
    match draft_with(ui, f, include_implicit) {
        Ok(s) => {
            r.status = "ok".into();
            if ui.inlines.enabled {
                r.plain = draft_variant(ui, f, false, true).ok().filter(|p| *p != s);
            }
            r.src = Some(s);
        }
        Err(NoDraft::HeaderInline) => r.status = "hdr-inline".into(),
        Err(NoDraft::Implicit(_)) => r.status = "implicit".into(),
        Err(NoDraft::Lift(e)) => {
            r.status = "lift-err".into();
            r.error = Some(e);
        }
    }
    r
}

fn reply_json(r: &DraftReply) -> String {
    serde_json::json!({"status": r.status, "src": r.src, "plain": r.plain, "implicit": r.implicit, "error": r.error}).to_string()
}

fn parse_reply(line: &str) -> DraftReply {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
        return DraftReply::err("crash", format!("bad draft-server reply: {}", line.chars().take(200).collect::<String>()));
    };
    let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
    DraftReply { status: s("status").unwrap_or_default(), src: s("src"), plain: s("plain"), implicit: s("implicit"), error: s("error") }
}

/// `mwdec draft-server`: one JSON request per stdin line ({"unit","symbol","include_implicit"}),
/// one reply per stdout line. Keeps only the current unit's inputs.
pub fn serve(root: &Path, work: &Path, no_db: bool) -> Result<()> {
    let p = load_project(root)?;
    // Probe compiles of the inline library: 2 slots (the parent's search workers of the job
    // waiting for this draft aren't compiling meanwhile).
    let cc = Compilers::new(root, work, 2);
    // The module's target objects: vtables (with the TypeDb) and literal values the target only
    // references. The main module's index is built once and layered under each REL's.
    let mut main_ext: Option<Arc<mwdec_mwcc::ExternIndex>> = None;
    let mut cur: Option<(String, Result<UnitInputs, String>)> = None;
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        let Ok(req) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
        let unit = req.get("unit").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let symbol = req.get("symbol").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let include_implicit = req.get("include_implicit").and_then(|x| x.as_bool()).unwrap_or(false);
        if cur.as_ref().map_or(true, |c| c.0 != unit) {
            drop(cur.take()); // drop the previous unit before loading the next
            let ui = find_unit(&p, &unit).map_err(|e| e.to_string()).and_then(|u| {
                let main = main_ext.get_or_insert_with(|| Arc::new(mwdec_mwcc::ExternIndex::new(p.load_module_data("main")))).clone();
                let m = Project::module_of(&u.name);
                let ext = mwdec_mwcc::ExternIndex::layered(main, if m == "main" { vec![] } else { p.load_module_data(m) });
                let objs: &[Arc<ObjectFile>] = if no_db { &[] } else { &ext.objs };
                unit_inputs(&p, u, &cc, objs, !no_db, Some(&ext)).map_err(|e| e.to_string())
            });
            cur = Some((unit.clone(), ui));
        }
        // Test hook for the guard: simulate an IR explosion on this symbol.
        if std::env::var("MWDEC_DRAFT_TEST_OOM").is_ok_and(|s| s == symbol) {
            let mut hog: Vec<Vec<u8>> = Vec::new();
            loop {
                hog.push(vec![1u8; 64 << 20]);
            }
        }
        let r = match &cur.as_ref().unwrap().1 {
            Ok(ui) => draft_local(ui, &symbol, include_implicit),
            Err(e) => DraftReply::err("unit-err", e.clone()),
        };
        writeln!(out, "{}", reply_json(&r))?;
        out.flush()?;
    }
    Ok(())
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<String>,
    err_tail: Arc<Mutex<VecDeque<String>>>,
    err_thread: Option<std::thread::JoinHandle<()>>,
}

/// Parent side: a lazily (re)started draft child shared by the eval workers.
pub struct DraftClient {
    root: PathBuf,
    work: PathBuf,
    no_db: bool,
    pub timeout: Duration,
    proc_: Mutex<Option<Proc>>,
}

impl DraftClient {
    pub fn new(root: &Path, work: &Path, no_db: bool) -> DraftClient {
        DraftClient { root: root.into(), work: work.into(), no_db, timeout: Duration::from_secs(300), proc_: Mutex::new(None) }
    }

    fn spawn(&self) -> std::io::Result<Proc> {
        let exe = std::env::current_exe()?;
        let mem = DRAFT_MEM_MB.min(mwdec_core::memcap::limit_mb().max(512));
        let mut cmd = Command::new(exe);
        cmd.arg("--root").arg(&self.root).arg("--work").arg(&self.work).arg("draft-server");
        if self.no_db {
            cmd.arg("--no-db");
        }
        let mut child = cmd
            .env("MWDEC_MEM_MB", mem.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for l in BufReader::new(stdout).lines() {
                let Ok(l) = l else { break };
                if tx.send(l).is_err() {
                    break;
                }
            }
        });
        let err_tail: Arc<Mutex<VecDeque<String>>> = Default::default();
        let tail = err_tail.clone();
        let err_thread = std::thread::spawn(move || {
            for l in BufReader::new(stderr).lines() {
                let Ok(l) = l else { break };
                eprintln!("draft: {l}");
                let mut t = tail.lock().unwrap();
                t.push_back(l);
                if t.len() > 20 {
                    t.pop_front();
                }
            }
        });
        Ok(Proc { child, stdin, rx, err_tail, err_thread: Some(err_thread) })
    }

    /// Draft `symbol` of `unit` in the child; a dead or hung child is reported for this function
    /// and restarted on the next request.
    pub fn draft(&self, unit: &str, symbol: &str, include_implicit: bool) -> DraftReply {
        let mut g = self.proc_.lock().unwrap();
        if g.is_none() {
            match self.spawn() {
                Ok(p) => *g = Some(p),
                Err(e) => return DraftReply::err("crash", format!("spawning draft-server: {e}")),
            }
        }
        let p = g.as_mut().unwrap();
        let req = serde_json::json!({"unit": unit, "symbol": symbol, "include_implicit": include_implicit}).to_string();
        let sent = writeln!(p.stdin, "{req}").and_then(|_| p.stdin.flush());
        let res = if sent.is_ok() { p.rx.recv_timeout(self.timeout) } else { Err(RecvTimeoutError::Disconnected) };
        match res {
            Ok(line) => parse_reply(&line),
            Err(e) => {
                let mut p = g.take().unwrap();
                let timed_out = e == RecvTimeoutError::Timeout;
                if timed_out {
                    let _ = p.child.kill(); // our own child, by handle
                }
                let status = p.child.wait().ok();
                if let Some(t) = p.err_thread.take() {
                    let _ = t.join();
                }
                let tail: Vec<String> = p.err_tail.lock().unwrap().iter().cloned().collect();
                let last = tail.iter().rev().find(|l| !l.trim().is_empty()).cloned().unwrap_or_default();
                if timed_out {
                    DraftReply::err("timeout", format!("draft took over {}s", self.timeout.as_secs()))
                } else if tail.iter().any(|l| l.contains("memory allocation of")) {
                    DraftReply::err("oom/too-big", last)
                } else {
                    DraftReply::err("crash", format!("draft-server exited ({}): {last}", status.map(|s| s.to_string()).unwrap_or_default()))
                }
            }
        }
    }
}

impl Drop for DraftClient {
    fn drop(&mut self) {
        if let Some(mut p) = self.proc_.lock().unwrap().take() {
            drop(p.stdin); // EOF: the child exits
            let _ = p.child.wait();
        }
    }
}
