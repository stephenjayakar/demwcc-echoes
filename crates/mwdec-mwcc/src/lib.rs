//! Run the real compiler and compare functions strictly (see DESIGN.md).
//!
//! - [`Mwcc`]: compiler driver bound to a project root (cwd for `-i` include paths), with
//!   precompiled per-unit contexts ([`Mwcc::precompile`] / [`UnitContext`]), a content-hash
//!   cache of compiled candidates (memory + disk), and a bounded process pool (default 6).
//! - [`compile`]: the simple free-function form (no PCH, no cache).
//! - [`compare`] / [`compare_detailed`]: the strict comparator.
//! - Fast path ([`Mwcc::enable_fast`], module `fast`): candidates compiled by persistent compiler
//!   processes that parse the context once; failures and exact matches are redone normally.
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
pub mod placeholder;
mod fast;
mod split;
pub mod sweep;
pub use fast::FastStats;
pub use compare::{compare, compare_detailed, compare_indexed, Detailed, DiffClass, ExternIndex, ObjIndex};
pub use placeholder::PlaceholderProver;

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
    /// Object bytes (also at `obj_path` when that is set: the on-disk cache entry, or the
    /// object a `compile_tu` call leaves for its caller).
    pub obj: Arc<Vec<u8>>,
    /// Where the object was written (may be a cache file).
    pub obj_path: PathBuf,
    /// Compiler warnings/notes (stdout+stderr), usually empty.
    pub messages: String,
    /// Wall time of the compiler process (0 for cache hits).
    pub ms: f64,
    pub cache_hit: bool,
    /// Produced by the fast path (a persistent compiler, `Mwcc::enable_fast`): confirm an exact
    /// match with [`Mwcc::compile_in_normal`] before reporting it.
    pub fast: bool,
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
    /// Text compiled in front of each candidate after the PCH (split PCH contexts: the context
    /// lines left out of the PCH; see `split`). Empty normally.
    pub text: String,
    /// Indexes of the context lines left out of the PCH (split contexts).
    pub excluded: Vec<usize>,
    /// File name candidate TUs are compiled as (`CFoo.cpp`): the compiler names the static
    /// initializer (`__sinit_CFoo_cpp`) and the anonymous namespace (`@unnamed@CFoo_cpp@`)
    /// after it. `None`: a unique scratch name.
    pub tu_name: Option<String>,
}

impl UnitContext {
    /// Candidates compiled as the unit's own source file name (`main/Dir/CFoo` ->
    /// `CFoo.cpp`, `.c` for C units).
    pub fn named(mut self, unit: &str) -> UnitContext {
        let base = unit.rsplit('/').next().unwrap_or(unit);
        let c = self.cflags.iter().any(|f| f == "-lang=c" || f == "-lang=c99");
        if !base.is_empty() && base.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' || ch == '-') {
            self.tu_name = Some(format!("{base}.{}", if c { "c" } else { "cpp" }));
        }
        self
    }
}

/// Object (and whether the fast path produced it) or a compile error.
type CacheEntry = Result<(Arc<Vec<u8>>, bool), MwccError>;

/// Default byte budget of the in-memory candidate cache (`MWDEC_MEMCACHE_MB` overrides).
pub const MEM_CACHE_MB: usize = 96;

/// In-memory candidate cache bounded by total bytes (objects + messages); oldest entries are
/// evicted first. Unbounded, it grew by every candidate of every function of an eval.
struct MemCache {
    map: HashMap<u128, CacheEntry>,
    order: std::collections::VecDeque<u128>,
    bytes: usize,
    budget: usize,
}

fn entry_bytes(e: &CacheEntry) -> usize {
    64 + match e {
        Ok((o, _)) => o.len(),
        Err(e) => e.messages().len(),
    }
}

impl MemCache {
    fn new() -> MemCache {
        let mb = std::env::var("MWDEC_MEMCACHE_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(MEM_CACHE_MB);
        MemCache { map: HashMap::new(), order: Default::default(), bytes: 0, budget: mb << 20 }
    }
    fn get(&self, k: &u128) -> Option<&CacheEntry> {
        self.map.get(k)
    }
    fn insert(&mut self, k: u128, v: CacheEntry) {
        let n = entry_bytes(&v);
        if n > self.budget / 4 {
            return;
        }
        if let Some(old) = self.map.insert(k, v) {
            self.bytes -= entry_bytes(&old);
        } else {
            self.order.push_back(k);
        }
        self.bytes += n;
        while self.bytes > self.budget {
            let Some(o) = self.order.pop_front() else { break };
            if let Some(e) = self.map.remove(&o) {
                self.bytes -= entry_bytes(&e);
            }
        }
    }
}

/// Messages of a compile failure a persistent compiler reported without confirmation by a normal
/// compile (a trusted failure of the fast path).
pub const UNCONFIRMED_FAILURE: &str = "compile failed (persistent compiler; messages not collected)";

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
    pool: Arc<Pool>,
    /// Persistent-compiler fast path for candidate compiles (`enable_fast`).
    fast: std::sync::RwLock<Option<Arc<fast::FastPool>>>,
    counter: AtomicU64,
    mem_cache: Mutex<MemCache>,
    pch_locks: Mutex<HashMap<u128, Arc<Mutex<()>>>>,
    /// Split PCH replacements per context hash (`split`), established on the first crash.
    splits: Mutex<HashMap<u128, Arc<std::sync::OnceLock<Option<UnitContext>>>>>,
}

impl Mwcc {
    /// Driver with `jobs` parallel compiler processes, scratch under `work`
    /// (`work/tmp`, `work/pch`, `work/cache`).
    pub fn new(root: &Path, work: &Path, jobs: usize) -> Mwcc {
        // scratch that ended processes left behind (once per work dir and process)
        sweep::sweep_once(work);
        Mwcc {
            root: root.to_path_buf(),
            compiler: DEFAULT_COMPILER.to_string(),
            work: work.to_path_buf(),
            disk_cache: Some(work.join("cache")),
            timeout: Duration::from_secs(120),
            pool: Arc::new(Pool::new(jobs)),
            fast: std::sync::RwLock::new(None),
            counter: AtomicU64::new(0),
            mem_cache: Mutex::new(MemCache::new()),
            pch_locks: Mutex::new(HashMap::new()),
            splits: Mutex::new(HashMap::new()),
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
        let w = Instant::now();
        let _slot = self.pool.acquire();
        mwdec_core::prof::add("mwcc.slot_wait", w.elapsed().as_secs_f64());
        let _p = mwdec_core::prof::span("mwcc.process");
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
        self.compile_tu_as(source, cflags, prefix, work_dir, None)
    }

    /// `compile_tu` with the source file named `tu_name` (in a private directory).
    pub fn compile_tu_as(
        &self,
        source: &str,
        cflags: &[String],
        prefix: Option<&Path>,
        work_dir: &Path,
        tu_name: Option<&str>,
    ) -> Result<Compiled, MwccError> {
        std::fs::create_dir_all(work_dir).map_err(io("creating work dir"))?;
        let stem = work_dir.join(self.unique("tu"));
        let (src, own_dir) = match tu_name {
            Some(n) => {
                std::fs::create_dir_all(&stem).map_err(io("creating TU dir"))?;
                (stem.join(n), true)
            }
            None => (stem.with_extension("cpp"), false),
        };
        let obj = stem.with_extension("o");
        // A compiler that could not open one of our work files (another process sharing the work
        // dir was replacing or deleting it, a virus scanner held it, ...) is retried, and reported
        // as an I/O failure (never cached as a compile error) if it keeps failing.
        let mut attempt = 0;
        let mut starts = 0;
        let res = loop {
            std::fs::write(&src, source).map_err(io("writing TU"))?;
            let res = self.compile_src(&src, &obj, &stem, cflags, prefix);
            match &res {
                Err(MwccError::Compile { messages, .. }) if self.transient_io(messages) => {
                    attempt += 1;
                    if attempt >= IO_RETRIES {
                        break Err(MwccError::Io(format!("compiler could not open a work file: {}", messages.lines().take(3).collect::<Vec<_>>().join(" | "))));
                    }
                    std::thread::sleep(Duration::from_millis(150 * attempt as u64));
                }
                // the compiler process could not start or allocate (the machine is out of
                // resources): back off and run it again
                Err(MwccError::Crash { status: Some(s), .. }) if resource_status(*s) && starts < START_RETRIES => {
                    starts += 1;
                    std::thread::sleep(Duration::from_millis(500 << starts));
                }
                _ => break res,
            }
        };
        if own_dir {
            let _ = std::fs::remove_dir_all(&stem);
        }
        res
    }

    /// The compiler failed to open, read or write one of our own files (under the work dir):
    /// an I/O race, not a property of the source.
    fn transient_io(&self, messages: &str) -> bool {
        let lower = messages.to_ascii_lowercase();
        let marker = ["cannot be opened", "cannot open", "could not write file", "could not load file", "could not find or load precompiled", "oserr"]
            .iter()
            .any(|m| lower.contains(m));
        let work = win_path(&self.work).to_ascii_lowercase();
        marker && (lower.contains(&work) || lower.contains(&work.replace('\\', "/")) || lower.contains("oserr"))
    }

    fn compile_src(&self, src: &Path, obj: &Path, stem: &Path, cflags: &[String], prefix: Option<&Path>) -> Result<Compiled, MwccError> {
        let (src, obj, stem) = (src.to_path_buf(), obj.to_path_buf(), stem.to_path_buf());
        let mut args: Vec<String> = cflags.to_vec();
        if let Some(p) = prefix {
            args.push("-prefix".into());
            args.push(win_path(p));
        }
        args.extend(["-c".into(), win_path(&src), "-o".into(), win_path(&obj)]);
        let res = self.run(&args, &stem);
        let _ = std::fs::remove_file(&src);
        if res.is_err() {
            // (a compiler killed on timeout may have left a partial object)
            let _ = std::fs::remove_file(&obj);
        }
        let (ok, status, messages, ms) = res?;
        if !ok {
            let _ = std::fs::remove_file(&obj);
            return Err(failure(status, messages));
        }
        let bytes = std::fs::read(&obj).map_err(io("reading object"))?;
        Ok(Compiled { obj: Arc::new(bytes), obj_path: obj, messages, ms, cache_hit: false, fast: false })
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
            let tmp_mch = self.build_mch(context, cflags, &dir, "ctx")?;
            publish(&tmp_mch, &mch);
        }
        Ok(UnitContext { cflags: cflags.to_vec(), context: context.to_string(), mch: Some(mch), hash, text: String::new(), excluded: vec![], tu_name: None })
    }

    /// Precompile `context` into a private `.mch` (a unique file the caller owns and deletes).
    pub(crate) fn precompile_private(&self, context: &str, cflags: &[String]) -> Result<UnitContext, MwccError> {
        let hash = content_hash(&[CACHE_SALT.as_bytes(), self.compiler.as_bytes(), &flags_bytes(cflags), context.as_bytes()]);
        let mch = self.build_mch(context, cflags, &self.work.join("pch"), "trial")?;
        Ok(UnitContext { cflags: cflags.to_vec(), context: context.to_string(), mch: Some(mch), hash, text: String::new(), excluded: vec![], tu_name: None })
    }

    /// Run the precompiler on `context` into a new uniquely named `.mch` in `dir` (retrying when
    /// the compiler could not open one of our files).
    fn build_mch(&self, context: &str, cflags: &[String], dir: &Path, stem_name: &str) -> Result<PathBuf, MwccError> {
        let _p = mwdec_core::prof::span("mwcc.pch_build");
        std::fs::create_dir_all(dir).map_err(io("creating pch dir"))?;
        let mut attempt = 0;
        loop {
            let stem = dir.join(self.unique(stem_name));
            // The extension selects precompilation in the driver: .pch++ = C++, .pch = C.
            let is_c = cflags.iter().any(|f| f == "-lang=c" || f == "-lang=c99");
            let src = stem.with_extension(if is_c { "pch" } else { "pch++" });
            let tmp_mch = stem.with_extension("mch");
            std::fs::write(&src, context).map_err(io("writing context"))?;
            let mut args: Vec<String> = cflags.to_vec();
            args.extend(["-c".into(), win_path(&src), "-o".into(), win_path(dir), "-precompile".into(), win_path(&tmp_mch)]);
            let res = self.run(&args, &stem);
            let _ = std::fs::remove_file(&src);
            let (ok, status, messages, _ms) = res?;
            if ok && tmp_mch.exists() {
                return Ok(tmp_mch);
            }
            let _ = std::fs::remove_file(&tmp_mch);
            let e = failure(status, messages);
            match &e {
                MwccError::Compile { messages, .. } if self.transient_io(messages) => {
                    attempt += 1;
                    if attempt >= IO_RETRIES {
                        return Err(MwccError::Io(format!("precompiling: {}", messages.lines().take(3).collect::<Vec<_>>().join(" | "))));
                    }
                    std::thread::sleep(Duration::from_millis(150 * attempt as u64));
                }
                _ => return Err(e),
            }
        }
    }

    /// A context without PCH: the context text is prepended to every candidate (slow path,
    /// and the reference for verifying that PCH does not change codegen).
    pub fn plain_context(&self, context: &str, cflags: &[String]) -> UnitContext {
        let hash = content_hash(&[CACHE_SALT.as_bytes(), b"plain", self.compiler.as_bytes(), &flags_bytes(cflags), context.as_bytes()]);
        UnitContext { cflags: cflags.to_vec(), context: context.to_string(), mch: None, hash, text: String::new(), excluded: vec![], tu_name: None }
    }

    /// Compile candidate code (function definitions etc.) in a unit context. Results (objects and
    /// compile errors) are cached by content hash in memory and on disk. With the fast path on
    /// ([`Mwcc::enable_fast`]) the object may come from a persistent compiler (`Compiled::fast`).
    pub fn compile_in(&self, ctx: &UnitContext, code: &str) -> Result<Compiled, MwccError> {
        self.compile_in_mode(ctx, code, true)
    }

    /// [`Mwcc::compile_in`] without the fast path: a normal compiler run (or a cached result of
    /// one). Use it to confirm a match found with a fast-path object.
    pub fn compile_in_normal(&self, ctx: &UnitContext, code: &str) -> Result<Compiled, MwccError> {
        self.compile_in_mode(ctx, code, false)
    }

    fn compile_in_mode(&self, ctx: &UnitContext, code: &str, allow_fast: bool) -> Result<Compiled, MwccError> {
        let key = match &ctx.tu_name {
            Some(n) => content_hash(&[&ctx.hash.to_le_bytes(), b"tu:", n.as_bytes(), code.as_bytes()]),
            None => content_hash(&[&ctx.hash.to_le_bytes(), code.as_bytes()]),
        };
        let hit = |e: CacheEntry, path: PathBuf| {
            e.map(|(obj, fast)| Compiled { obj, obj_path: path, messages: String::new(), ms: 0.0, cache_hit: true, fast })
        };
        let cached = self.mem_cache.lock().unwrap().get(&key).cloned();
        match cached {
            Some(Ok((_, true))) if !allow_fast => {}
            Some(e) => {
                mwdec_core::prof::add("mwcc.hit_mem", 0.0);
                return hit(e, self.cache_path(key, "o").unwrap_or_default());
            }
            None => {}
        }
        if let Some(e) = self.disk_get(key) {
            mwdec_core::prof::add("mwcc.hit_disk", 0.0);
            self.mem_cache.lock().unwrap().insert(key, e.clone());
            return hit(e, self.cache_path(key, "o").unwrap_or_default());
        }
        let mut fast_failed = None;
        // Test hook: a persistent compiler that wrongly reports (and trusts) a compile failure
        // for candidates containing this text (`*`: for every candidate).
        if allow_fast && std::env::var("MWDEC_TEST_TRUSTED_FAIL").is_ok_and(|t| t == "*" || (!t.is_empty() && code.contains(t.as_str()))) {
            return Err(MwccError::Compile { status: Some(1), messages: UNCONFIRMED_FAILURE.into() });
        }
        if allow_fast {
            if let Some((fp, spec)) = self.fast_spec(ctx) {
                let t = Instant::now();
                match fp.compile(&spec, &fast_body(ctx, code)) {
                    fast::Outcome::Obj(obj) => {
                        let obj = Arc::new(obj);
                        if std::env::var_os("MWDEC_PERSIST_VERIFY").is_some() {
                            // check mode: every fast object against a normal compile
                            if let Ok(n) = self.compile_in_mode(ctx, code, false) {
                                let same = split::same_code(&obj, &n.obj);
                                fp.note_verify(same);
                                if !same {
                                    eprintln!("mwdec-mwcc: fast path and normal compile differ for:\n{code}");
                                    return Ok(n);
                                }
                            }
                        }
                        self.mem_cache.lock().unwrap().insert(key, Ok((obj.clone(), true)));
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        mwdec_core::prof::add("mwcc.fast_obj", ms / 1000.0);
                        return Ok(Compiled { obj, obj_path: PathBuf::new(), messages: String::new(), ms, cache_hit: false, fast: true });
                    }
                    fast::Outcome::Failed(w) => {
                        mwdec_core::prof::add("mwcc.fast_failed", t.elapsed().as_secs_f64());
                        if fp.trusted_failure(w) {
                            // a compile error without its messages (not cached: unconfirmed)
                            return Err(MwccError::Compile { status: Some(1), messages: UNCONFIRMED_FAILURE.into() });
                        }
                        fast_failed = Some((fp, w))
                    }
                    fast::Outcome::NotReady => {}
                }
            }
        }
        let res = {
            let _p = mwdec_core::prof::span(if ctx.mch.is_some() { "mwcc.compile_pch" } else { "mwcc.compile_plain" });
            self.compile_in_slow(ctx, code, key)
        };
        if let (Some((fp, w)), Err(MwccError::Compile { .. })) = (&fast_failed, &res) {
            fp.failure_confirmed(*w);
        }
        if let (Some((fp, w)), Ok(_)) = (&fast_failed, &res) {
            fp.poisoned(*w);
            if std::env::var_os("MWDEC_MWCC_LOG").is_some() {
                eprintln!("mwdec-mwcc: fast path failed, normal compile succeeded");
            }
        }
        res
    }

    /// The normal compile behind `compile_in` (PCH with crash workarounds, or the plain context).
    fn compile_in_slow(&self, ctx: &UnitContext, code: &str, key: u128) -> Result<Compiled, MwccError> {
        let log = std::env::var_os("MWDEC_MWCC_LOG").is_some();
        let mut res = match &ctx.mch {
            Some(_) => {
                let split = self.known_split(ctx);
                match self.compile_pch(split.as_ref().unwrap_or(ctx), code) {
                    // MWCC sometimes crashes with a PCH: find a split PCH that doesn't (once per
                    // context), else the plain context (same codegen, much slower).
                    Err(MwccError::Crash { status, messages }) => {
                        if log {
                            eprintln!("mwcc: PCH compile crashed (status {status:?}, code#{:08x}, split {}). {}", content_hash(&[code.as_bytes()]) as u32, split.is_some(), messages.lines().take(3).collect::<Vec<_>>().join(" | "));
                        }
                        let rep = if split.is_none() { self.repair(ctx, code) } else { None };
                        match rep.map(|sp| self.compile_pch(&sp, code)) {
                            Some(r @ (Ok(_) | Err(MwccError::Compile { .. }))) => r,
                            _ => {
                                if log {
                                    eprintln!("mwcc: plain-context compile for code#{:08x}", content_hash(&[code.as_bytes()]) as u32);
                                }
                                self.plain_compile(ctx, code)
                            }
                        }
                    }
                    r => r,
                }
            }
            None => self.plain_compile(ctx, code),
        };
        // Cache only deterministic outcomes (not crashes / timeouts / I/O failures).
        match &mut res {
            Ok(c) => {
                // the scratch object is not kept: its bytes are in `c.obj` (and the caches)
                let _ = std::fs::remove_file(&c.obj_path);
                c.obj_path = self.disk_put(key, Ok(&c.obj)).unwrap_or_default();
                self.mem_cache.lock().unwrap().insert(key, Ok((c.obj.clone(), false)));
            }
            Err(e @ MwccError::Compile { .. }) => {
                let _ = self.disk_put(key, Err(e.messages()));
                self.mem_cache.lock().unwrap().insert(key, Err(e.clone()));
            }
            Err(_) => {}
        }
        res
    }

    /// Turn on the fast path for candidate compiles: up to `workers` (at most [`MAX_FAST_WORKERS`]) persistent
    /// compiler threads (module `fast`), started on first use. GC/2.7 only; contexts compiled
    /// with line information (`-sym`, `-g`) always compile normally. `MWDEC_PERSIST=0` in the
    /// environment keeps it off. Returns whether it is on.
    pub fn enable_fast(&self, workers: usize) -> bool {
        if fast_disabled_by_env() || !self.compiler.replace('\\', "/").contains("GC/2.7/") {
            return false;
        }
        let mut f = self.fast.write().unwrap();
        if f.is_none() {
            *f = Some(Arc::new(fast::FastPool::new(workers.clamp(1, MAX_FAST_WORKERS), self.pool.clone())));
        }
        true
    }

    /// Make sure `n` fast-path workers hold a persistent compiler for `ctx` before a compile
    /// loop starts (instead of starting them on demand while candidates compile normally).
    /// Waits at most `wait`; returns how many hold it (0 when the fast path does not apply).
    pub fn warm_fast(&self, ctx: &UnitContext, n: usize, wait: std::time::Duration) -> usize {
        match self.fast_spec(ctx) {
            Some((fp, spec)) => fp.warm(&spec, n, wait),
            None => 0,
        }
    }

    /// End the fast path's workers and their compiler processes (it stays off afterwards).
    pub fn shutdown_fast(&self) {
        let f = self.fast.write().unwrap().take();
        if let Some(f) = f {
            f.shutdown();
        }
    }

    /// Counters of the fast path (`None` if it is off).
    pub fn fast_stats(&self) -> Option<FastStats> {
        self.fast.read().unwrap().as_ref().map(|f| f.stats())
    }

    /// Record the normal-compile confirmation of an exact match found with a fast-path object
    /// (`agreed`: the normal compile matched too).
    pub fn note_fast_confirm(&self, agreed: bool) {
        if let Some(f) = self.fast.read().unwrap().as_ref() {
            f.note_confirm(agreed);
        }
        if !agreed {
            eprintln!("mwdec-mwcc: an exact match from the fast path was not confirmed by a normal compile");
        }
    }

    /// The fast path and the persistent-compiler spec for `ctx`, if both apply.
    fn fast_spec(&self, ctx: &UnitContext) -> Option<(Arc<fast::FastPool>, Arc<fast::Spec>)> {
        let fp = self.fast.read().unwrap().clone()?;
        if ctx.cflags.iter().any(|f| f == "-sym" || f.starts_with("-sym") || f == "-g") {
            return None;
        }
        // Keyed by the context text (not the PCH): the plain and PCH forms share one process.
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (&self.compiler, &ctx.cflags, &ctx.context, &ctx.tu_name).hash(&mut h);
            h.finish()
        };
        let spec = fp.spec(key, || {
            let c_lang = ctx.cflags.iter().any(|f| f == "-lang=c" || f == "-lang=c99");
            let version = self.compiler.replace('\\', "/").trim_start_matches("build/compilers/").trim_end_matches("/mwcceppc.exe").to_string();
            fast::Spec {
                key,
                comp: mwdec_oracle::compile::Compiler {
                    root: self.root.clone(),
                    work: self.work.join("persist"),
                    profile: if c_lang { mwdec_oracle::flags::Profile::SdkC } else { mwdec_oracle::flags::Profile::Game },
                    extra: vec![],
                    version: Some(version),
                    cflags: Some(ctx.cflags.clone()),
                },
                context: ctx.context.clone(),
                file_name: ctx.tu_name.clone(),
            }
        });
        Some((fp, spec))
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
            return Some(Ok((Arc::new(b), false)));
        }
        let e = self.cache_path(key, "err")?;
        if let Ok(m) = std::fs::read_to_string(&e) {
            return Some(Err(MwccError::Compile { status: Some(1), messages: m }));
        }
        None
    }

    fn disk_put(&self, key: u128, v: Result<&[u8], &str>) -> Option<PathBuf> {
        // (cache entries are only ever replaced by identical content: a rename over one is safe)
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

/// Most persistent-compiler workers of one driver (each holds up to two context snapshots of a
/// few MB plus an idle compiler process).
pub const MAX_FAST_WORKERS: usize = 6;

/// Fast-path workers for search drivers: `MWDEC_FAST_WORKERS` (1..=MAX_FAST_WORKERS), else
/// `default`.
pub fn fast_workers_from_env(default: usize) -> usize {
    std::env::var("MWDEC_FAST_WORKERS").ok().and_then(|v| v.trim().parse().ok()).map_or(default, |n: usize| n.clamp(1, MAX_FAST_WORKERS))
}

/// Exit statuses of a compiler that failed for lack of system resources, not because of the
/// source: DLL initialization failed (0xC0000142, process creation under resource exhaustion),
/// no memory (0xC0000017), commitment limit (0xC000012D), insufficient resources (0xC000009A).
pub fn resource_status(s: i32) -> bool {
    matches!(s as u32, 0xC000_0142 | 0xC000_0017 | 0xC000_012D | 0xC000_009A)
}

/// Re-runs of a compile that failed with a [`resource_status`] (backing off 1, 2, 4, 8 s).
const START_RETRIES: u32 = 4;

/// Attempts of a compile whose compiler could not open one of our work files.
const IO_RETRIES: u32 = 3;

/// Move a finished shared file (`.mch`) into place without ever replacing an existing one: other
/// processes sharing the work dir may be compiling against it right now (a rename over it fails
/// or, worse, swaps the file under a compiler that is about to open it). The first writer wins.
pub(crate) fn publish(tmp: &Path, dst: &Path) {
    if !dst.exists() && std::fs::hard_link(tmp, dst).is_err() && !dst.exists() {
        // no hard links on this file system: plain rename (still only when absent)
        let _ = std::fs::rename(tmp, dst);
    }
    let _ = std::fs::remove_file(tmp);
}

/// `MWDEC_PERSIST=0` (or `off` / `no`) disables the fast path.
pub fn fast_disabled_by_env() -> bool {
    std::env::var("MWDEC_PERSIST").is_ok_and(|v| matches!(v.trim(), "0" | "off" | "no" | "false"))
}

/// The candidate as the persistent compiler sees it: after the context and a marker line, so a
/// `#line` directive restores the line numbers of a normal compile (`__LINE__`): line 1 after a
/// PCH (the candidate is the whole file), after the context text otherwise.
fn fast_body(ctx: &UnitContext, code: &str) -> String {
    let line = if ctx.mch.is_some() || ctx.context.is_empty() {
        1
    } else {
        ctx.context.matches('\n').count() + 1 + usize::from(!ctx.context.ends_with('\n'))
    };
    format!("#line {line}\n{code}")
}

/// Compile `source` (a full TU) with `cflags`; returns the path of the produced object.
/// Uses [`default_root`] as the compiler cwd. Failures are `MwccError` (downcastable from anyhow).
pub fn compile(source: &str, cflags: &[String], work_dir: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    let m = Mwcc::new(&default_root(), work_dir, 1);
    let c = m.compile_tu(source, cflags, None, work_dir)?;
    Ok(c.obj_path)
}

