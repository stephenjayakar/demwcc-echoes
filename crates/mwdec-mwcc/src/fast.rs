//! Fast path for candidate compiles ([`Mwcc::enable_fast`]): persistent compiler processes
//! (`mwdec_oracle::persist`) that parse a unit context once and then compile each candidate after
//! it in a few milliseconds.
//!
//! - **Workers**: dedicated long-lived threads (started on first use), each owning up to
//!   [`persist::CACHE_PER_THREAD`] persistent compilers (the debugger API binds a compiler process
//!   to the thread that started it). At most `workers * CACHE_PER_THREAD` context snapshots exist
//!   at once ([`FastStats::live`]).
//! - **Routing**: a request goes to an idle worker that already holds the context. A context no
//!   worker holds yet is started (from its third request on) in the background on an idle
//!   worker while the request itself is compiled normally ([`Outcome::NotReady`]), so one-off
//!   compiles never wait for a start. A context in demand spreads to more workers, up to a fair
//!   share of the cache slots when several contexts are in use at once.
//! - **Results**: object bytes only. The persistent compiler reports no messages: on a failure the
//!   caller recompiles normally, and an exact match is confirmed with a normal compile.
//! - Every compiler process at work (fast or normal) holds a slot of the driver's pool.
//! - **Shutdown**: dropping the pool closes the queues; each worker ends its compiler processes
//!   as it exits (the pool waits for that).
use super::Pool;
use mwdec_oracle::compile::Compiler;
use mwdec_oracle::persist;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Contexts used within this window count as active (fair share of workers).
const ACTIVE: Duration = Duration::from_secs(5);
/// A context gets its first persistent compiler on this request (drafts compile a context once
/// or twice; a start costs about one plain-context compile of CPU).
const START_AFTER: u32 = 3;
/// A request waits at most this long for a busy worker holding its context.
const MAX_WAIT: Duration = Duration::from_secs(20);
/// Confirmed compile errors before a worker's failures are trusted (`FastPool::trusted_failure`).
const TRUST_AFTER: usize = 4;
/// A trusted worker's failures are still re-checked with a normal compile this often.
const RECHECK_EVERY: usize = 8;

/// What a persistent compiler is started on: compiler + flags + context text + file name.
pub(crate) struct Spec {
    pub key: u64,
    pub comp: Compiler,
    pub context: String,
    pub file_name: Option<String>,
}

/// Result of a fast-path request.
pub(crate) enum Outcome {
    /// Object bytes of a successful compile.
    Obj(Vec<u8>),
    /// The persistent compile on this worker failed (compile error, crash, or an unusable
    /// process): compile normally for the messages; if the normal compile succeeds, report it
    /// with [`FastPool::poisoned`].
    Failed(usize),
    /// No persistent compiler is ready for this context (one may be starting): compile normally.
    NotReady,
}

/// Counters of a driver's fast path.
#[derive(Clone, Copy, Debug, Default)]
pub struct FastStats {
    /// Successful persistent compiles.
    pub compiles: u64,
    /// Persistent compiles that failed (redone normally).
    pub failed: u64,
    /// ... of which the normal compile succeeded (the fast path should never reject a candidate
    /// the compiler accepts).
    pub failed_normal_ok: u64,
    /// Requests compiled normally because no persistent compiler was ready.
    pub not_ready: u64,
    /// Persistent compilers started / failed to start.
    pub starts: u64,
    pub start_failures: u64,
    /// Total milliseconds in successful persistent compiles / in starts.
    pub compile_ms: f64,
    pub start_ms: f64,
    /// Exact matches re-checked with a normal compile, and how many of them disagreed.
    pub confirms: u64,
    pub confirm_mismatches: u64,
    /// `MWDEC_PERSIST_VERIFY`: fast results compared with a normal compile, and disagreements.
    pub verified: u64,
    pub verify_mismatches: u64,
    /// Context snapshots alive now / at most so far.
    pub live: usize,
    pub live_max: usize,
}

impl FastStats {
    pub fn add(&mut self, o: &FastStats) {
        self.compiles += o.compiles;
        self.failed += o.failed;
        self.failed_normal_ok += o.failed_normal_ok;
        self.not_ready += o.not_ready;
        self.starts += o.starts;
        self.start_failures += o.start_failures;
        self.compile_ms += o.compile_ms;
        self.start_ms += o.start_ms;
        self.confirms += o.confirms;
        self.confirm_mismatches += o.confirm_mismatches;
        self.verified += o.verified;
        self.verify_mismatches += o.verify_mismatches;
        self.live += o.live;
        self.live_max += o.live_max;
    }

    /// Fast compiles and normal compiles that disagreed (should be zero).
    pub fn mismatches(&self) -> u64 {
        self.failed_normal_ok + self.confirm_mismatches + self.verify_mismatches
    }

    /// One-line summary.
    pub fn line(&self) -> String {
        let avg = if self.compiles > 0 { self.compile_ms / self.compiles as f64 } else { 0.0 };
        let avg_start = if self.starts > 0 { self.start_ms / self.starts as f64 } else { 0.0 };
        format!(
            "fast path: {} compiles ({avg:.1} ms avg), {} failed ({} accepted by a normal compile), {} not ready; \
             {} starts ({avg_start:.0} ms avg, {} failed), max {} snapshots; {} exact confirmations ({} disagreed), \
             {} verified ({} disagreed)",
            self.compiles,
            self.failed,
            self.failed_normal_ok,
            self.not_ready,
            self.starts,
            self.start_failures,
            self.live_max,
            self.confirms,
            self.confirm_mismatches,
            self.verified,
            self.verify_mismatches
        )
    }
}

#[derive(Default)]
struct Counters {
    compiles: AtomicU64,
    failed: AtomicU64,
    failed_normal_ok: AtomicU64,
    not_ready: AtomicU64,
    starts: AtomicU64,
    start_failures: AtomicU64,
    compile_us: AtomicU64,
    start_us: AtomicU64,
    confirms: AtomicU64,
    confirm_mismatches: AtomicU64,
    verified: AtomicU64,
    verify_mismatches: AtomicU64,
    live_max: AtomicUsize,
}

enum Msg {
    Run(Job),
    /// End the worker's persistent compilers (they are restarted on demand).
    Reset,
}

struct Job {
    spec: Arc<Spec>,
    body: String,
    /// `None`: a start (warm-up) without a waiting caller.
    reply: Option<Sender<Option<Vec<u8>>>>,
}

#[derive(Default)]
struct State {
    /// Per worker: busy with a job.
    busy: Vec<bool>,
    /// Per worker: contexts it holds, least recently used first (mirrors the worker's
    /// `persist::compile_cached` cache, which uses the same policy).
    held: Vec<VecDeque<u64>>,
    /// Starts in flight per context.
    starting: HashMap<u64, usize>,
    /// Contexts whose persistent compiler failed to start (normal compiles only).
    failed: HashSet<u64>,
    /// Last request per context.
    last_use: HashMap<u64, Instant>,
    /// Requests per context (bounded: cleared when large).
    requests: HashMap<u64, u32>,
}

impl State {
    fn live(&self) -> usize {
        self.held.iter().map(|h| h.len()).sum()
    }
}

struct Shared {
    state: Mutex<State>,
    cv: Condvar,
    counters: Counters,
    pool: Arc<Pool>,
}

/// The fast path of one driver: worker threads with persistent compilers.
pub(crate) struct FastPool {
    n: usize,
    /// Per worker: consecutive failures a normal compile confirmed (a real compile error), and
    /// failures reported since the last confirmation (see [`FastPool::trusted_failure`]).
    trust: Vec<AtomicUsize>,
    unchecked: Vec<AtomicUsize>,
    shared: Arc<Shared>,
    workers: Mutex<Vec<(Sender<Msg>, std::thread::JoinHandle<()>)>>,
    specs: Mutex<HashMap<u64, Arc<Spec>>>,
}

impl FastPool {
    pub fn new(n: usize, pool: Arc<Pool>) -> FastPool {
        let n = n.max(1);
        let state = State { busy: vec![false; n], held: vec![VecDeque::new(); n], ..Default::default() };
        FastPool {
            n,
            trust: (0..n).map(|_| AtomicUsize::new(0)).collect(),
            unchecked: (0..n).map(|_| AtomicUsize::new(0)).collect(),
            shared: Arc::new(Shared { state: Mutex::new(state), cv: Condvar::new(), counters: Counters::default(), pool }),
            workers: Mutex::new(Vec::new()),
            specs: Mutex::new(HashMap::new()),
        }
    }

    /// The shared spec for `key` (built once by `make`; bounded).
    pub fn spec(&self, key: u64, make: impl FnOnce() -> Spec) -> Arc<Spec> {
        let mut m = self.specs.lock().unwrap();
        if let Some(s) = m.get(&key) {
            return s.clone();
        }
        if m.len() >= 64 {
            m.clear();
        }
        let s = Arc::new(make());
        m.insert(key, s.clone());
        s
    }

    fn send(&self, w: usize, msg: Msg) -> bool {
        let mut ws = self.workers.lock().unwrap();
        if ws.is_empty() {
            for id in 0..self.n {
                let (tx, rx) = channel::<Msg>();
                let shared = self.shared.clone();
                match std::thread::Builder::new().name(format!("mwcc-fast-{id}")).spawn(move || worker(id, rx, shared)) {
                    Ok(h) => ws.push((tx, h)),
                    Err(_) => return false,
                }
            }
        }
        ws.get(w).is_some_and(|(tx, _)| tx.send(msg).is_ok())
    }

    /// Compile `body` after the spec's context (see the module docs).
    pub fn compile(&self, spec: &Arc<Spec>, body: &str) -> Outcome {
        let c = &self.shared.counters;
        let key = spec.key;
        let deadline = Instant::now() + MAX_WAIT;
        let mut st = self.shared.state.lock().unwrap();
        if st.failed.contains(&key) {
            c.not_ready.fetch_add(1, Relaxed);
            return Outcome::NotReady;
        }
        let now = Instant::now();
        st.last_use.insert(key, now);
        if st.requests.len() > 4096 {
            st.requests.clear();
        }
        let nreq = {
            let n = st.requests.entry(key).or_default();
            *n = n.saturating_add(1);
            *n
        };
        if st.last_use.len() > 64 {
            st.last_use.retain(|_, t| now.duration_since(*t) < ACTIVE);
        }
        loop {
            let holders: Vec<usize> = (0..self.n).filter(|&w| st.held[w].contains(&key)).collect();
            if let Some(&w) = holders.iter().find(|&&w| !st.busy[w]) {
                st.busy[w] = true;
                drop(st);
                let (tx, rx) = channel();
                if !self.send(w, Msg::Run(Job { spec: spec.clone(), body: body.to_string(), reply: Some(tx) })) {
                    self.shared.state.lock().unwrap().busy[w] = false;
                    c.not_ready.fetch_add(1, Relaxed);
                    return Outcome::NotReady;
                }
                return match rx.recv() {
                    Ok(Some(obj)) => Outcome::Obj(obj),
                    _ => Outcome::Failed(w),
                };
            }
            // Spread the context to one more worker if it is below its share.
            let active = st.last_use.values().filter(|t| now.duration_since(**t) < ACTIVE).count().max(1);
            // each worker can hold CACHE_PER_THREAD contexts: share the slots, not the workers
            let share = (self.n * persist::CACHE_PER_THREAD).div_ceil(active).clamp(1, self.n);
            let starting = st.starting.get(&key).copied().unwrap_or(0);
            if starting == 0 && holders.len() < share && (nreq >= START_AFTER || !holders.is_empty()) {
                if let Some(w) = self.start_target(&st, key, share, now) {
                    st.busy[w] = true;
                    *st.starting.entry(key).or_default() += 1;
                    if !self.send(w, Msg::Run(Job { spec: spec.clone(), body: String::new(), reply: None })) {
                        st.busy[w] = false;
                        st.starting.remove(&key);
                    }
                }
            }
            if holders.is_empty() || Instant::now() >= deadline {
                c.not_ready.fetch_add(1, Relaxed);
                return Outcome::NotReady;
            }
            st = self.shared.cv.wait_timeout(st, Duration::from_millis(250)).unwrap().0;
            if st.failed.contains(&key) {
                c.not_ready.fetch_add(1, Relaxed);
                return Outcome::NotReady;
            }
        }
    }

    /// Start persistent compilers for `spec` until `n` workers (at most the pool's) hold it, or
    /// `wait` has passed, or a start failed. Returns the number of holders.
    pub fn warm(&self, spec: &Arc<Spec>, n: usize, wait: Duration) -> usize {
        let key = spec.key;
        let deadline = Instant::now() + wait;
        let n = n.clamp(1, self.n);
        let mut st = self.shared.state.lock().unwrap();
        loop {
            let holders = st.held.iter().filter(|h| h.contains(&key)).count();
            if holders >= n || st.failed.contains(&key) || Instant::now() >= deadline {
                return holders;
            }
            let now = Instant::now();
            st.last_use.insert(key, now);
            let starting = st.starting.get(&key).copied().unwrap_or(0);
            if holders + starting < n {
                if let Some(w) = self.start_target(&st, key, n, now) {
                    st.busy[w] = true;
                    *st.starting.entry(key).or_default() += 1;
                    if !self.send(w, Msg::Run(Job { spec: spec.clone(), body: String::new(), reply: None })) {
                        st.busy[w] = false;
                        if let Some(k) = st.starting.get_mut(&key) {
                            *k -= 1;
                        }
                        return holders;
                    }
                    continue;
                }
            }
            st = self.shared.cv.wait_timeout(st, Duration::from_millis(100)).unwrap().0;
        }
    }

    /// An idle worker to start `key` on: one with a free cache slot, else one whose least
    /// recently used context is inactive or held by more workers than its share.
    fn start_target(&self, st: &State, key: u64, share: usize, now: Instant) -> Option<usize> {
        let idle = (0..self.n).filter(|&w| !st.busy[w] && !st.held[w].contains(&key));
        let mut best: Option<(usize, usize)> = None;
        for w in idle {
            let rank = if st.held[w].len() < persist::CACHE_PER_THREAD {
                0
            } else {
                let victim = st.held[w][0];
                let hot = st.last_use.get(&victim).is_some_and(|t| now.duration_since(*t) < ACTIVE);
                let victim_holders = st.held.iter().filter(|h| h.contains(&victim)).count();
                if !hot {
                    1
                } else if victim_holders > share {
                    2
                } else {
                    continue;
                }
            };
            if best.is_none_or(|(r, _)| rank < r) {
                best = Some((rank, w));
            }
        }
        best.map(|(_, w)| w)
    }

    /// Worker `w` failed a candidate that a normal compile accepted: its persistent compilers are
    /// in a bad state (process state the snapshot does not cover, e.g. after an exception in an
    /// error path). End them; they restart on the next use.
    pub fn poisoned(&self, w: usize) {
        self.trust[w].store(0, Relaxed);
        self.shared.counters.failed_normal_ok.fetch_add(1, Relaxed);
        self.shared.state.lock().unwrap().held[w].clear();
        self.send(w, Msg::Reset);
    }
    /// A failure of worker `w` can be reported as a compile error without the normal compile
    /// that would give its messages: the worker's last [`TRUST_AFTER`] failures were all real
    /// compile errors (a candidate the normal compiler accepts resets it), and one in
    /// [`RECHECK_EVERY`] failures is still re-checked. Compile errors are frequent among search
    /// candidates and each normal compile costs as much as dozens of fast ones.
    pub fn trusted_failure(&self, w: usize) -> bool {
        if self.trust[w].load(Relaxed) < TRUST_AFTER {
            return false;
        }
        let n = self.unchecked[w].fetch_add(1, Relaxed) + 1;
        if n >= RECHECK_EVERY {
            self.unchecked[w].store(0, Relaxed);
            return false;
        }
        true
    }

    /// A failure of worker `w` that a normal compile confirmed as a compile error.
    pub fn failure_confirmed(&self, w: usize) {
        self.trust[w].fetch_add(1, Relaxed);
    }

    pub fn note_confirm(&self, agreed: bool) {
        self.shared.counters.confirms.fetch_add(1, Relaxed);
        if !agreed {
            self.shared.counters.confirm_mismatches.fetch_add(1, Relaxed);
        }
    }
    pub fn note_verify(&self, agreed: bool) {
        self.shared.counters.verified.fetch_add(1, Relaxed);
        if !agreed {
            self.shared.counters.verify_mismatches.fetch_add(1, Relaxed);
        }
    }

    pub fn stats(&self) -> FastStats {
        let c = &self.shared.counters;
        let live = self.shared.state.lock().unwrap().live();
        FastStats {
            compiles: c.compiles.load(Relaxed),
            failed: c.failed.load(Relaxed),
            failed_normal_ok: c.failed_normal_ok.load(Relaxed),
            not_ready: c.not_ready.load(Relaxed),
            starts: c.starts.load(Relaxed),
            start_failures: c.start_failures.load(Relaxed),
            compile_ms: c.compile_us.load(Relaxed) as f64 / 1000.0,
            start_ms: c.start_us.load(Relaxed) as f64 / 1000.0,
            confirms: c.confirms.load(Relaxed),
            confirm_mismatches: c.confirm_mismatches.load(Relaxed),
            verified: c.verified.load(Relaxed),
            verify_mismatches: c.verify_mismatches.load(Relaxed),
            live,
            live_max: c.live_max.load(Relaxed),
        }
    }

    /// End every worker and its compiler processes (waits for jobs in flight).
    pub fn shutdown(&self) {
        let ws: Vec<_> = std::mem::take(&mut *self.workers.lock().unwrap());
        let handles: Vec<_> = ws.into_iter().map(|(tx, h)| {
            drop(tx);
            h
        }).collect();
        for h in handles {
            let _ = h.join();
        }
        let mut st = self.shared.state.lock().unwrap();
        for h in st.held.iter_mut() {
            h.clear();
        }
        st.busy.iter_mut().for_each(|b| *b = false);
        st.starting.clear();
    }
}

impl Drop for FastPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker(id: usize, rx: Receiver<Msg>, shared: Arc<Shared>) {
    for msg in rx {
        let job = match msg {
            Msg::Run(job) => job,
            Msg::Reset => {
                persist::clear_cache();
                continue;
            }
        };
        let key = job.spec.key;
        let start = job.reply.is_none();
        // a start compiles an empty candidate (the context alone)
        let new_instance = !shared.state.lock().unwrap().held[id].contains(&key);
        let t = Instant::now();
        let r = {
            let _slot = shared.pool.acquire();
            persist::compile_cached(&job.spec.comp, &job.spec.context, job.spec.file_name.as_deref(), &job.body)
        };
        let us = t.elapsed().as_micros() as u64;
        let c = &shared.counters;
        if let Err(e) = &r {
            if std::env::var_os("MWDEC_MWCC_LOG").is_some() {
                eprintln!("mwcc-fast-{id}: {e:#}");
            }
        }
        {
            let mut st = shared.state.lock().unwrap();
            st.busy[id] = false;
            if start {
                if let Some(n) = st.starting.get_mut(&key) {
                    *n -= 1;
                    if *n == 0 {
                        st.starting.remove(&key);
                    }
                }
            }
            if new_instance {
                c.starts.fetch_add(1, Relaxed);
                c.start_us.fetch_add(us, Relaxed);
            } else if r.is_ok() {
                c.compiles.fetch_add(1, Relaxed);
                c.compile_us.fetch_add(us, Relaxed);
            }
            if r.is_err() {
                if new_instance {
                    c.start_failures.fetch_add(1, Relaxed);
                    st.failed.insert(key);
                    if st.failed.len() > 1024 {
                        st.failed.clear();
                    }
                } else {
                    c.failed.fetch_add(1, Relaxed);
                }
            }
            // same policy as `persist::compile_cached`: most recently used last, oldest dropped
            let held = &mut st.held[id];
            held.retain(|k| *k != key);
            held.push_back(key);
            while held.len() > persist::CACHE_PER_THREAD {
                held.pop_front();
            }
            let live = st.live();
            c.live_max.fetch_max(live, Relaxed);
        }
        shared.cv.notify_all();
        if let Some(tx) = job.reply {
            if new_instance && r.is_ok() {
                // started on a request (the instance was dropped): counts as a compile too
                c.compiles.fetch_add(1, Relaxed);
                c.compile_us.fetch_add(us, Relaxed);
            }
            let _ = tx.send(r.ok().map(|o| o.object));
        }
    }
    persist::clear_cache();
}
