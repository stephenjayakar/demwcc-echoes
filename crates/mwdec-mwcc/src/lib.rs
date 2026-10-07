//! Run the real compiler and compare functions strictly (see DESIGN.md).
//!
//! - [`Mwcc`]: compiler driver bound to a project root (cwd for `-i` include paths), with
//!   precompiled per-unit contexts ([`Mwcc::precompile`] / [`UnitContext`]), a content-hash
//!   cache of compiled candidates (memory + disk), and a bounded process pool (default 6).
//! - [`compile`]: the simple free-function form (no PCH, no cache).
//! - [`compare`] / [`compare_detailed`]: the strict comparator.
//!
//! MWCC quirk: path arguments must be absolute Windows paths with backslashes; forward slashes in
//! `-precompile` / `-prefix` arguments are rejected ("filename is invalid").
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub mod compare;
pub use compare::{compare, compare_detailed, compare_indexed, Detailed, DiffClass, ExternIndex, ObjIndex};

/// Default project root (read-only inputs); override with `MWDEC_ROOT`.

/// Default scratch root; override with `MWDEC_WORK`.

/// Default compiler, relative to the project root.
pub const DEFAULT_COMPILER: &str = "build/compilers/GC/2.7/mwcceppc.exe";
/// Bump to invalidate on-disk caches when the cache format/semantics change.
const CACHE_SALT: &str = "mwdec-mwcc-cache-v2";

pub fn default_root() -> PathBuf {
    std::env::var_os("MWDEC_ROOT").map(PathBuf::from).unwrap_or_else(mwdec_core::paths::project_root)
}

pub fn default_work() -> PathBuf {
    std::env::var_os("MWDEC_WORK").map(PathBuf::from).unwrap_or_else(|| mwdec_core::paths::work_dir("mwdec-mwcc"))
}

/// Typed compiler error.
#[derive(Debug, Clone)]
pub enum MwccError {
    /// The compiler ran and rejected the input; `messages` is its full diagnostic output.
    Compile { status: Option<i32>, messages: String },
    /// The compiler crashed (e.g. access violation 0xC0000005, seen with some PCHs) or hit an
    /// internal compiler error. Never cached; with a PCH context, `compile_in` retries without it.
    Crash { status: Option<i32>, messages: String },
    /// The compiler did not finish in time (it was killed by PID).
    Timeout { seconds: u64 },
    /// Spawning / file I/O failure.
    Io(String),
}

impl std::fmt::Display for MwccError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MwccError::Compile { status, messages } => {
                write!(f, "mwcc failed (status {status:?}):\n{}", messages.trim_end())
            }
            MwccError::Crash { status, messages } => {
                write!(f, "mwcc crashed (status {status:?}):
{}", messages.trim_end())
            }
            MwccError::Timeout { seconds } => write!(f, "mwcc timed out after {seconds}s"),
            MwccError::Io(s) => write!(f, "mwcc I/O error: {s}"),
        }
    }
}

impl std::error::Error for MwccError {}

impl MwccError {
    /// Compiler diagnostics (empty for I/O errors).
    pub fn messages(&self) -> &str {
        match self {
            MwccError::Compile { messages, .. } | MwccError::Crash { messages, .. } => messages,
            _ => "",
        }
    }
}

/// A failed run that is not an ordinary diagnostic: abnormal exit status (Windows exceptions
/// such as 0xC0000005 show up as negative codes) or an internal compiler error.
fn failure(status: Option<i32>, messages: String) -> MwccError {
    let abnormal = !matches!(status, Some(0..=255));
    let lower = messages.to_ascii_lowercase();
    if abnormal || lower.contains("internal compiler error") || lower.contains("access violation") || lower.contains("unhandled exception") {
        MwccError::Crash { status, messages }
    } else {
        MwccError::Compile { status, messages }
    }
}

fn io<E: std::fmt::Display>(ctx: &str) -> impl FnOnce(E) -> MwccError + '_ {
    move |e| MwccError::Io(format!("{ctx}: {e}"))
}

/// Absolute Windows path with backslashes and no `\\?\` prefix.
pub fn win_path(p: &Path) -> String {
    let abs = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    abs.to_string_lossy().replace('/', "\\")
}

// ---------------------------------------------------------------- hashing

/// FNV-1a 128 over length-prefixed parts (stable across runs/toolchains; used for cache keys).
pub fn content_hash(parts: &[&[u8]]) -> u128 {
    const OFF: u128 = 0x6c62272e07bb014262b821756295c58d;
    const PRIME: u128 = 0x0000000001000000000000000000013B;
    let mut h = OFF;
    for p in parts {
        for b in (p.len() as u64).to_le_bytes().iter().chain(p.iter()) {
            h ^= *b as u128;
            h = h.wrapping_mul(PRIME);
        }
    }
    h
}

fn flags_bytes(cflags: &[String]) -> Vec<u8> {
    cflags.join("\u{1}").into_bytes()
}

// ---------------------------------------------------------------- pool

/// Counting semaphore bounding concurrent compiler processes.
struct Pool {
    slots: Mutex<usize>,
    cv: Condvar,
}

impl Pool {
    fn new(n: usize) -> Self {
        Pool { slots: Mutex::new(n.max(1)), cv: Condvar::new() }
    }
    fn acquire(&self) -> PoolGuard<'_> {
        let mut s = self.slots.lock().unwrap();
        while *s == 0 {
            s = self.cv.wait(s).unwrap();
        }
        *s -= 1;
        PoolGuard(self)
    }
}

struct PoolGuard<'a>(&'a Pool);

impl Drop for PoolGuard<'_> {
    fn drop(&mut self) {
        *self.0.slots.lock().unwrap() += 1;
        self.0.cv.notify_one();
    }
}

// ---------------------------------------------------------------- driver

/// Output of one successful compile.
#[derive(Debug, Clone)]
pub struct Compiled {
    /// Object bytes (also written to `obj_path` unless served from the memory cache).
    pub obj: Arc<Vec<u8>>,
    /// Where the object was written (may be a cache file).
    pub obj_path: PathBuf,
    /// Compiler warnings/notes (stdout+stderr), usually empty.
    pub messages: String,
    /// Wall time of the compiler process (0 for cache hits).
    pub ms: f64,
    pub cache_hit: bool,
}

/// A precompiled unit context: `.mch` built from the context TU with the unit's flags.
#[derive(Debug, Clone)]
pub struct UnitContext {
    pub cflags: Vec<String>,
    /// The context TU text (include lines).
    pub context: String,
    /// Precompiled header, passed with `-prefix`. `None` = no PCH (context is prepended as text).
    pub mch: Option<PathBuf>,
    /// Identity of (compiler, flags, context) for cache keys.
    pub hash: u128,
}

type CacheEntry = Result<Arc<Vec<u8>>, MwccError>;

/// Compiler driver bound to a project root.
pub struct Mwcc {
    /// Project root; compiler cwd.
    pub root: PathBuf,
    /// Compiler path relative to `root`.
    pub compiler: String,
    /// Scratch dir for TUs/objects.
    pub work: PathBuf,
    /// On-disk candidate cache (`<hash>.o` / `<hash>.err`); `None` disables it.
    pub disk_cache: Option<PathBuf>,
    pub timeout: Duration,
    pool: Pool,
    counter: AtomicU64,
    mem_cache: Mutex<HashMap<u128, CacheEntry>>,
    pch_locks: Mutex<HashMap<u128, Arc<Mutex<()>>>>,
}

impl Mwcc {
    /// Driver with `jobs` parallel compiler processes, scratch under `work`
    /// (`work/tmp`, `work/pch`, `work/cache`).
    pub fn new(root: &Path, work: &Path, jobs: usize) -> Mwcc {
        Mwcc {
            root: root.to_path_buf(),
            compiler: DEFAULT_COMPILER.to_string(),
            work: work.to_path_buf(),
            disk_cache: Some(work.join("cache")),
            timeout: Duration::from_secs(120),
            pool: Pool::new(jobs),
            counter: AtomicU64::new(0),
            mem_cache: Mutex::new(HashMap::new()),
            pch_locks: Mutex::new(HashMap::new()),
        }
    }

    /// `Mwcc::new(default_root(), default_work(), 6)`.
    pub fn default_driver() -> Mwcc {
        Mwcc::new(&default_root(), &default_work(), 6)
    }

    /// Process-unique scratch stem. The counter is process-wide: several `Mwcc` instances (one per
    /// compiler version, `check`/`ctx` helpers, ...) may share one work dir, and per-instance
    /// counters made them reuse the same temp names (the compiler then fails with "OSErr -48").
    fn unique(&self, stem: &str) -> String {
        static UNIQUE: AtomicU64 = AtomicU64::new(0);
        let n = UNIQUE.fetch_add(1, Ordering::Relaxed);
        self.counter.fetch_add(1, Ordering::Relaxed);
        format!("{stem}_{}_{n}", std::process::id())
    }

    /// Run the compiler with `args` (cwd = root), bounded by the pool. Returns (status ok, messages, ms).
    fn run(&self, args: &[String], log_stem: &Path) -> Result<(bool, Option<i32>, String, f64), MwccError> {
        let _slot = self.pool.acquire();
        let out_path = log_stem.with_extension("log");
        let out = std::fs::File::create(&out_path).map_err(io("creating compiler log"))?;
        let err = out.try_clone().map_err(io("cloning log handle"))?;
        let exe = self.root.join(&self.compiler);
        let t = Instant::now();
        let mut child = Command::new(&exe)
            .args(args)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::from(out))
            .stderr(Stdio::from(err))
            .spawn()
            .map_err(io("spawning mwcceppc"))?;
        // Watchdog: kill our own child (by PID) if it exceeds the timeout.
        let pid = child.id();
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let timeout = self.timeout;
        let watchdog = std::thread::spawn(move || {
            if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = rx.recv_timeout(timeout) {
                let _ = Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
                return true;
            }
            false
        });
        let status = child.wait().map_err(io("waiting for mwcceppc"));
        let _ = tx.send(());
        let timed_out = watchdog.join().unwrap_or(false);
        if timed_out {
            let _ = std::fs::remove_file(&out_path);
            return Err(MwccError::Timeout { seconds: self.timeout.as_secs() });
        }
        let status = status?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let messages = String::from_utf8_lossy(&std::fs::read(&out_path).unwrap_or_default()).into_owned();
        let _ = std::fs::remove_file(&out_path);
        Ok((status.success(), status.code(), messages, ms))
    }

    /// Compile a full TU `source` with `cflags` (+ optional `-prefix <mch>`), writing into
    /// `work_dir`. No caching. The object is left at the returned path.
    pub fn compile_tu(
        &self,
        source: &str,
        cflags: &[String],
        prefix: Option<&Path>,
        work_dir: &Path,
    ) -> Result<Compiled, MwccError> {
        std::fs::create_dir_all(work_dir).map_err(io("creating work dir"))?;
        let stem = work_dir.join(self.unique("tu"));
        let src = stem.with_extension("cpp");
        let obj = stem.with_extension("o");
        std::fs::write(&src, source).map_err(io("writing TU"))?;
        let mut args: Vec<String> = cflags.to_vec();
        if let Some(p) = prefix {
            args.push("-prefix".into());
            args.push(win_path(p));
        }
        args.extend(["-c".into(), win_path(&src), "-o".into(), win_path(&obj)]);
        let res = self.run(&args, &stem);
        let _ = std::fs::remove_file(&src);
        let (ok, status, messages, ms) = res?;
        if !ok {
            let _ = std::fs::remove_file(&obj);
            return Err(failure(status, messages));
        }
        let bytes = std::fs::read(&obj).map_err(io("reading object"))?;
        Ok(Compiled { obj: Arc::new(bytes), obj_path: obj, messages, ms, cache_hit: false })
    }

    /// Precompile a unit context (`context` = include lines) with `cflags` into an `.mch`,
    /// cached by content hash under `work/pch`. Thread-safe; concurrent callers share one build.
    /// If the compiler crashes while precompiling, falls back to [`Mwcc::plain_context`].
    pub fn precompile(&self, context: &str, cflags: &[String]) -> Result<UnitContext, MwccError> {
        match self.precompile_mch(context, cflags) {
            Err(MwccError::Crash { .. }) => Ok(self.plain_context(context, cflags)),
            r => r,
        }
    }

    fn precompile_mch(&self, context: &str, cflags: &[String]) -> Result<UnitContext, MwccError> {
        let hash = content_hash(&[CACHE_SALT.as_bytes(), self.compiler.as_bytes(), &flags_bytes(cflags), context.as_bytes()]);
        let dir = self.work.join("pch");
        let mch = dir.join(format!("{hash:032x}.mch"));
        let lock = self.pch_locks.lock().unwrap().entry(hash).or_default().clone();
        let _g = lock.lock().unwrap();
        if !mch.exists() {
            std::fs::create_dir_all(&dir).map_err(io("creating pch dir"))?;
            let stem = dir.join(self.unique("ctx"));
            // The extension selects precompilation in the driver: .pch++ = C++, .pch = C.
            let is_c = cflags.iter().any(|f| f == "-lang=c" || f == "-lang=c99");
            let src = stem.with_extension(if is_c { "pch" } else { "pch++" });
            let tmp_mch = stem.with_extension("mch");
            std::fs::write(&src, context).map_err(io("writing context"))?;
            let mut args: Vec<String> = cflags.to_vec();
            args.extend([
                "-c".into(),
                win_path(&src),
                "-o".into(),
                win_path(&dir),
                "-precompile".into(),
                win_path(&tmp_mch),
            ]);
            let res = self.run(&args, &stem);
            let _ = std::fs::remove_file(&src);
            let (ok, status, messages, _ms) = res?;
            if !ok || !tmp_mch.exists() {
                let _ = std::fs::remove_file(&tmp_mch);
                return Err(failure(status, messages));
            }
            if std::fs::rename(&tmp_mch, &mch).is_err() {
                // Another process won the race; keep theirs.
                let _ = std::fs::remove_file(&tmp_mch);
            }
        }
        Ok(UnitContext { cflags: cflags.to_vec(), context: context.to_string(), mch: Some(mch), hash })
    }

    /// A context without PCH: the context text is prepended to every candidate (slow path,
    /// and the reference for verifying that PCH does not change codegen).
    pub fn plain_context(&self, context: &str, cflags: &[String]) -> UnitContext {
        let hash = content_hash(&[CACHE_SALT.as_bytes(), b"plain", self.compiler.as_bytes(), &flags_bytes(cflags), context.as_bytes()]);
        UnitContext { cflags: cflags.to_vec(), context: context.to_string(), mch: None, hash }
    }

    /// Compile candidate code (function definitions etc.) in a unit context. Results (objects and
    /// compile errors) are cached by content hash in memory and on disk.
    pub fn compile_in(&self, ctx: &UnitContext, code: &str) -> Result<Compiled, MwccError> {
        let key = content_hash(&[&ctx.hash.to_le_bytes(), code.as_bytes()]);
        if let Some(hit) = self.mem_cache.lock().unwrap().get(&key).cloned() {
            return hit.map(|obj| Compiled {
                obj,
                obj_path: self.cache_path(key, "o").unwrap_or_default(),
                messages: String::new(),
                ms: 0.0,
                cache_hit: true,
            });
        }
        if let Some(hit) = self.disk_get(key) {
            self.mem_cache.lock().unwrap().insert(key, hit.clone());
            return hit.map(|obj| Compiled {
                obj,
                obj_path: self.cache_path(key, "o").unwrap_or_default(),
                messages: String::new(),
                ms: 0.0,
                cache_hit: true,
            });
        }
        let tmp = self.work.join("tmp");
        let plain = || {
            let mut tu = ctx.context.clone();
            if !tu.is_empty() && !tu.ends_with('\n') {
                tu.push('\n');
            }
            tu.push_str(code);
            self.compile_tu(&tu, &ctx.cflags, None, &tmp)
        };
        let mut res = match &ctx.mch {
            Some(m) => match self.compile_tu(code, &ctx.cflags, Some(m), &tmp) {
                // MWCC sometimes crashes with a PCH; the plain context gives the same codegen.
                Err(MwccError::Crash { .. }) => plain(),
                r => r,
            },
            None => plain(),
        };
        // Cache only deterministic outcomes (not crashes / timeouts / I/O failures).
        match &mut res {
            Ok(c) => {
                if let Some(p) = self.disk_put(key, Ok(&c.obj)) {
                    let _ = std::fs::remove_file(&c.obj_path);
                    c.obj_path = p;
                }
                self.mem_cache.lock().unwrap().insert(key, Ok(c.obj.clone()));
            }
            Err(e @ MwccError::Compile { .. }) => {
                let _ = self.disk_put(key, Err(e.messages()));
                self.mem_cache.lock().unwrap().insert(key, Err(e.clone()));
            }
            Err(_) => {}
        }
        res
    }

    /// Compile many candidates in one context concurrently (bounded by the pool).
    pub fn compile_many(&self, ctx: &UnitContext, codes: &[String]) -> Vec<Result<Compiled, MwccError>> {
        std::thread::scope(|s| {
            let hs: Vec<_> = codes.iter().map(|c| s.spawn(move || self.compile_in(ctx, c))).collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(MwccError::Io("panic".into())))).collect()
        })
    }

    fn cache_path(&self, key: u128, ext: &str) -> Option<PathBuf> {
        self.disk_cache.as_ref().map(|d| d.join(format!("{:02x}", (key >> 120) as u8)).join(format!("{key:032x}.{ext}")))
    }

    fn disk_get(&self, key: u128) -> Option<CacheEntry> {
        let o = self.cache_path(key, "o")?;
        if let Ok(b) = std::fs::read(&o) {
            return Some(Ok(Arc::new(b)));
        }
        let e = self.cache_path(key, "err")?;
        if let Ok(m) = std::fs::read_to_string(&e) {
            return Some(Err(MwccError::Compile { status: Some(1), messages: m }));
        }
        None
    }

    fn disk_put(&self, key: u128, v: Result<&[u8], &str>) -> Option<PathBuf> {
        let (ext, bytes): (&str, &[u8]) = match v {
            Ok(b) => ("o", b),
            Err(m) => ("err", m.as_bytes()),
        };
        let p = self.cache_path(key, ext)?;
        std::fs::create_dir_all(p.parent()?).ok()?;
        let tmp = p.with_extension(format!("{ext}.{}", self.unique("w")));
        std::fs::write(&tmp, bytes).ok()?;
        if std::fs::rename(&tmp, &p).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        Some(p)
    }
}

/// Compile `source` (a full TU) with `cflags`; returns the path of the produced object.
/// Uses [`default_root`] as the compiler cwd. Failures are `MwccError` (downcastable from anyhow).
pub fn compile(source: &str, cflags: &[String], work_dir: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    let m = Mwcc::new(&default_root(), work_dir, 1);
    let c = m.compile_tu(source, cflags, None, work_dir)?;
    Ok(c.obj_path)
}

