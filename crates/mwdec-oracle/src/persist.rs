//! Prototype: a persistent real-compiler process for many compiles that share one context.
//!
//! The candidate source is laid out as `<context>` + a marker `#define` line + a padded slot. The
//! compiler runs under a debugger until the preprocessor reaches the marker directive in the main
//! file (`CPrep_ParseDirective`, GC/2.7 0x436470, with the text position at `[0x5e9f3c]`): the whole
//! context (headers, declarations) is parsed by then. At that moment the process's writable
//! memory (the executable's data, the heap, the 32-bit stack) and the thread context are
//! snapshotted. Each compile then restores the snapshot, writes the candidate text into the slot
//! of the in-memory source buffer, and lets the compiler run until it calls `exit` (0x402180); the
//! object file is read from the usual output path (at the `ExitProcess` call of `exit`, 0x402249,
//! after the streams are flushed). The context is parsed once instead of per
//! compile, and no process is created per compile.
//!
//! Restored per compile: every committed writable private region and the executable's data, with
//! the commit state of the snapshot (regions allocated later are released, pages committed later
//! are decommitted, pages released later are committed again), plus the state the C runtime keeps
//! outside them: the 32-bit thread's TLS slots and last error, and the process TLS bitmap. The
//! compiler's line buffer uses CR line ends; candidates are converted.
//!
//! The snapshot is taken only when the directive being parsed is the marker line itself (a
//! directive just before it, e.g. the context's last `#include`, would leave that header to be
//! parsed again by every compile).
//!
//! Measured (`persist-bench`, 2.7, identical listings in every case): context `CStateManager.hpp` +
//! `CActor.hpp`, 120 mixed candidates (header inlines, statics, strings, switches, virtual classes,
//! templates): 908 -> 17.7 ms per compile (+1.6 s once to start); same context, 30 simple candidates:
//! 692 -> 14.2 ms; `CVector3f.hpp`: 162 -> 6.4 ms; a two-line context: 34 -> 4.6 ms. The restore
//! (6.7 MB) costs 2-6 ms of that.
//!
//! Debug aids (environment): `MWDEC_PERSIST_DEBUG` (slot contents), `MWDEC_PERSIST_PROFILE[=depth]`
//! (sample the compiler's call sites every millisecond and print the hottest), `MWDEC_PERSIST_MARK=
//! addr,...` (print the time at which each address is first reached).
//!
//! The debugger API ties the process to the thread that started it: use a `PersistentCompiler` on
//! one thread only (it is not `Send`).
//!
//! Limits: GC/2.7 only; the candidate must fit in the slot (`capacity` bytes); compiler messages are
//! not captured (a failing compile reports only its exit status: re-run it with [`Compiler`] to see
//! the diagnostics); line numbers after the marker are shifted (`__LINE__`, `-sym on` line tables).
//! After a timeout the process is unusable (`compile` reports it): start a new one. Diagnostic and
//! speed tool: always confirm a match with a normal compile.

use crate::compile::{CompileOutput, Compiler};
use crate::tracer::{ffi, quote_arg, wide};
use anyhow::{bail, Context, Result};
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::path::PathBuf;

const BP_DIRECTIVE: u32 = 0x436470; // CPrep_ParseDirective
const BP_EXIT: u32 = 0x402249; // exit(): the ExitProcess call, after atexit handlers and stream flushing; eax = status
const A_TEXT_POS: u32 = 0x5e9f3c; // current text position (char*)
const A_FILE_INDEX: u32 = 0x5ef1d4; // s16 current file index (0 = main file)
const MARKER: &str = "MWDEC_PERSIST_SPLIT_7E1";
const WOW64_CONTEXT_ALL: u32 = 0x0001_003F;
const STATUS_INVALID_HANDLE: u32 = 0xC000_0008;
const COMPILE_FAILED: &str = "compile failed";

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Mbi {
    base: usize,
    alloc_base: usize,
    alloc_protect: u32,
    partition: u16,
    size: usize,
    state: u32,
    protect: u32,
    typ: u32,
}

#[repr(C)]
#[derive(Default)]
struct MemCounters {
    cb: u32,
    page_faults: u32,
    peak_working_set: usize,
    working_set: usize,
    quota_peak_paged: usize,
    quota_paged: usize,
    quota_peak_nonpaged: usize,
    quota_nonpaged: usize,
    pagefile: usize,
    peak_pagefile: usize,
}

#[repr(C)]
#[derive(Default)]
struct ThreadBasicInfo {
    exit_status: i32,
    teb: usize,
    client_id: [usize; 2],
    affinity: usize,
    priority: i32,
    base_priority: i32,
}

#[allow(non_snake_case)]
mod ffi2 {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        pub fn VirtualQueryEx(h: *mut c_void, addr: *const c_void, mbi: *mut super::Mbi, len: usize) -> usize;
        pub fn VirtualFreeEx(h: *mut c_void, addr: *mut c_void, size: usize, ty: u32) -> i32;
        pub fn VirtualAllocEx(h: *mut c_void, addr: *mut c_void, size: usize, ty: u32, prot: u32) -> *mut c_void;
        pub fn SuspendThread(h: *mut c_void) -> u32;
        pub fn K32GetProcessMemoryInfo(h: *mut c_void, pmc: *mut super::MemCounters, cb: u32) -> i32;
        pub fn ResumeThread(h: *mut c_void) -> u32;
        pub fn VirtualProtectEx(h: *mut c_void, addr: *mut c_void, size: usize, prot: u32, old: *mut u32) -> i32;
    }
    #[link(name = "ntdll")]
    extern "system" {
        pub fn NtQueryInformationThread(h: *mut c_void, class: u32, info: *mut c_void, len: u32, ret: *mut u32) -> i32;
    }
}

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_DECOMMIT: u32 = 0x4000;
const MEM_FREE: u32 = 0x10000;
const MEM_RELEASE: u32 = 0x8000;
const MEM_PRIVATE: u32 = 0x20000;
const MEM_IMAGE: u32 = 0x100_0000;

fn writable(p: u32) -> bool {
    matches!(p & 0xff, 0x04 | 0x08 | 0x40 | 0x80)
}

pub struct PersistentCompiler {
    hp: ffi::HANDLE,
    ht: ffi::HANDLE,
    pid: u32,
    tid: u32,
    /// (pid, tid) of the debug event the debuggee is stopped in
    stopped: Option<(u32, u32)>,
    ctx: ffi::WOW64_CONTEXT,
    snapshot: Vec<(u32, Vec<u8>)>,
    /// allocation bases of the private regions that existed at the snapshot
    known: BTreeSet<u32>,
    /// private regions (base, end) that were reserved but not committed at the snapshot
    reserved: Vec<(u32, u32)>,
    /// allocation base of the 32-bit stack (its guard page moves; never decommitted)
    stack_alloc: u32,
    /// (address, contents) of the 32-bit TEB and PEB at the snapshot (debug comparison)
    teb_peb: Vec<(u32, Vec<u8>)>,
    /// excluded ranges (TEBs/PEBs, 64-bit stack)
    exclude: Vec<(u32, u32)>,
    slot: u32,
    capacity: usize,
    obj: PathBuf,
    src: PathBuf,
    exit_orig: u8,
    pub compiles: u64,
    /// the process has exited (its exit event was consumed)
    exited: bool,
    /// compiles that failed once and were retried
    pub retries: u64,
    /// MWDEC_PERSIST_MARK: one-shot timing breakpoints (address -> original byte)
    marks: std::collections::HashMap<u32, u8>,
    /// total time spent restoring the snapshot
    pub restore_time: std::time::Duration,
    /// bytes restored per compile
    pub snapshot_bytes: usize,
    /// How it was started (compiler, context, slot size, file name): a fresh instance replaces
    /// one whose snapshot keeps crashing.
    restart: Option<(Compiler, String, usize, Option<String>)>,
}

impl PersistentCompiler {
    fn read(&self, a: u32, n: usize) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; n];
        let mut got = 0usize;
        let ok = unsafe { ffi::ReadProcessMemory(self.hp, a as usize as *const _, buf.as_mut_ptr() as *mut _, n, &mut got) };
        (ok != 0 && got == n).then_some(buf)
    }
    fn write(&self, a: u32, data: &[u8]) -> bool {
        let mut got = 0usize;
        let ok = unsafe { ffi::WriteProcessMemory(self.hp, a as usize as *mut _, data.as_ptr() as *const _, data.len(), &mut got) };
        ok != 0 && got == data.len()
    }
    fn u32(&self, a: u32) -> u32 {
        self.read(a, 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).unwrap_or(0)
    }

    /// Start the compiler on `context` (a TU prefix of complete declarations) and stop it right
    /// after the context is parsed. `capacity`: maximum candidate size in bytes.
    pub fn start(comp: &Compiler, context: &str, capacity: usize) -> Result<Self> {
        Self::start_as(comp, context, capacity, None)
    }

    /// [`start`](Self::start) with the source file name the candidates are compiled as (e.g.
    /// `CFoo.cpp`): the compiler names static initializers (`__sinit_CFoo_cpp`) and the anonymous
    /// namespace after it. `None`: a unique scratch name.
    pub fn start_as(comp: &Compiler, context: &str, capacity: usize, file_name: Option<&str>) -> Result<Self> {
        let ver = comp.version.clone().unwrap_or_else(|| comp.profile.version().to_string());
        if ver != "GC/2.7" {
            bail!("persistent compiler: GC/2.7 only (got {ver})");
        }
        std::fs::create_dir_all(&comp.work)?;
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let stem = format!("pc{}_{}", std::process::id(), n);
        let ext = if comp.profile.is_c() { "c" } else { "cpp" };
        let (src, obj) = match file_name {
            Some(name) => {
                // own directory, so several processes can use the same file name
                let dir = comp.work.join(&stem);
                std::fs::create_dir_all(&dir)?;
                (dir.join(name), dir.join(format!("{stem}.o")))
            }
            None => (comp.work.join(format!("{stem}.{ext}")), comp.work.join(format!("{stem}.o"))),
        };
        let mut text = String::with_capacity(context.len() + capacity + 64);
        text.push_str(context);
        text.push_str(&format!("\n#define {MARKER} 1\n"));
        for k in 0..capacity {
            text.push(if k % 128 == 127 { '\n' } else { ' ' });
        }
        text.push('\n');
        std::fs::write(&src, &text)?;
        let mut parts = vec![quote_arg(&comp.exe().to_string_lossy().replace('/', "\\"))];
        parts.extend(comp.flags().iter().map(|f| quote_arg(f)));
        parts.push("-c".into());
        parts.push(quote_arg(&src.to_string_lossy()));
        parts.push("-o".into());
        parts.push(quote_arg(&obj.to_string_lossy()));
        let mut cmdw = wide(&parts.join(" "));
        let cwdw = wide(&comp.work.to_string_lossy());
        let mut si: ffi::STARTUPINFOW = unsafe { std::mem::zeroed() };
        si.cb = std::mem::size_of::<ffi::STARTUPINFOW>() as u32;
        let mut pi: ffi::PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
        let ok = unsafe {
            ffi::CreateProcessW(
                std::ptr::null(),
                cmdw.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                ffi::DEBUG_ONLY_THIS_PROCESS | ffi::CREATE_NO_WINDOW,
                std::ptr::null_mut(),
                cwdw.as_ptr(),
                &mut si,
                &mut pi,
            )
        };
        if ok == 0 {
            bail!("CreateProcessW failed: {}", std::io::Error::last_os_error());
        }
        let mut pc = PersistentCompiler {
            hp: pi.hProcess,
            ht: pi.hThread,
            pid: pi.dwProcessId,
            tid: pi.dwThreadId,
            stopped: None,
            ctx: unsafe { std::mem::zeroed() },
            snapshot: vec![],
            known: BTreeSet::new(),
            reserved: vec![],
            stack_alloc: 0,
            teb_peb: vec![],
            exclude: vec![],
            slot: 0,
            capacity,
            obj,
            src,
            exit_orig: 0,
            compiles: 0,
            exited: false,
            retries: 0,
            marks: Default::default(),
            restore_time: std::time::Duration::ZERO,
            snapshot_bytes: 0,
            restart: Some((comp.clone(), context.to_string(), capacity, file_name.map(|s| s.to_string()))),
        };
        pc.run_to_marker().context("persistent compiler: start")?;
        Ok(pc)
    }

    fn run_to_marker(&mut self) -> Result<()> {
        let mut dir_orig = 0u8;
        let start = std::time::Instant::now();
        loop {
            let mut ev: ffi::DEBUG_EVENT = unsafe { std::mem::zeroed() };
            if unsafe { ffi::WaitForDebugEvent(&mut ev, 60_000) } == 0 || start.elapsed().as_secs() > 300 {
                bail!("timeout parsing the context");
            }
            let mut status = ffi::DBG_CONTINUE;
            match ev.dwDebugEventCode {
                ffi::CREATE_PROCESS_DEBUG_EVENT => {
                    dir_orig = self.read(BP_DIRECTIVE, 1).map(|b| b[0]).unwrap_or(0);
                    self.exit_orig = self.read(BP_EXIT, 1).map(|b| b[0]).unwrap_or(0);
                    self.write(BP_DIRECTIVE, &[0xcc]);
                    self.write(BP_EXIT, &[0xcc]);
                    unsafe { ffi::FlushInstructionCache(self.hp, std::ptr::null(), 0) };
                }
                ffi::EXIT_PROCESS_DEBUG_EVENT => {
                    unsafe { ffi::ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, ffi::DBG_CONTINUE) };
                    self.exited = true;
                    bail!("compiler exited before the marker (compile error in the context?)")
                }
                ffi::EXCEPTION_DEBUG_EVENT => {
                    let er = unsafe { ev.u.Exception.ExceptionRecord };
                    let addr = er.ExceptionAddress as usize as u32;
                    let code = er.ExceptionCode;
                    let bp = code == ffi::STATUS_BREAKPOINT || code == ffi::STATUS_WX86_BREAKPOINT;
                    if bp && addr == BP_EXIT {
                        self.stopped = Some((ev.dwProcessId, ev.dwThreadId));
                        bail!("compiler exited before the marker (compile error in the context?)");
                    }
                    if bp && addr == BP_DIRECTIVE {
                        let th = unsafe { ffi::OpenThread(ffi::THREAD_ALL_ACCESS, 0, ev.dwThreadId) };
                        let mut c: ffi::WOW64_CONTEXT = unsafe { std::mem::zeroed() };
                        c.ContextFlags = WOW64_CONTEXT_ALL;
                        unsafe { ffi::Wow64GetThreadContext(th, &mut c) };
                        let main_file = self.read(A_FILE_INDEX, 2).map_or(false, |b| i16::from_le_bytes([b[0], b[1]]) == 0);
                        let pos = self.u32(A_TEXT_POS);
                        let win = if main_file { self.read(pos.saturating_sub(64), 256) } else { None };
                        // the directive being parsed must be the marker line itself (the line holding
                        // the text position), not a directive shortly before it such as the last
                        // #include of the context
                        let at = win.as_ref().and_then(|w| {
                            let m = MARKER.as_bytes();
                            let k = w.windows(m.len()).position(|x| x == m)?;
                            let p = 64usize.min(w.len());
                            let eol = |b: &u8| *b == 13 || *b == 10;
                            let ls = w[..k].iter().rposition(eol).map_or(0, |x| x + 1);
                            let le = w[k..].iter().position(eol).map_or(w.len(), |x| k + x);
                            (ls <= p && p <= le + 1).then_some(k)
                        });
                        if let (Some(w), Some(k)) = (win, at) {
                            // slot = after the end of the marker line
                            let base = pos.saturating_sub(64);
                            // the compiler keeps lines ending in '\r' in its buffer
                            let Some(nl) = w[k..].iter().position(|&b| b == b'\r' || b == b'\n').map(|x| k + x + 1) else {
                                bail!("persistent compiler: marker line end not found");
                            };
                            self.slot = base + nl as u32;
                            // remove the directive breakpoint, rewind to it, snapshot here
                            self.write(BP_DIRECTIVE, &[dir_orig]);
                            unsafe { ffi::FlushInstructionCache(self.hp, std::ptr::null(), 0) };
                            c.Eip = BP_DIRECTIVE;
                            unsafe { ffi::Wow64SetThreadContext(th, &c) };
                            self.ctx = c;
                            self.tid = ev.dwThreadId;
                            unsafe { ffi::CloseHandle(th) };
                            self.stopped = Some((ev.dwProcessId, ev.dwThreadId));
                            self.take_snapshot()?;
                            return Ok(());
                        }
                        // another directive: step over the breakpoint and re-arm
                        self.write(BP_DIRECTIVE, &[dir_orig]);
                        c.Eip = BP_DIRECTIVE;
                        c.EFlags |= 0x100;
                        unsafe {
                            ffi::FlushInstructionCache(self.hp, std::ptr::null(), 0);
                            ffi::Wow64SetThreadContext(th, &c);
                            ffi::CloseHandle(th);
                        }
                    } else if code == ffi::STATUS_SINGLE_STEP || code == ffi::STATUS_WX86_SINGLE_STEP {
                        self.write(BP_DIRECTIVE, &[0xcc]);
                        unsafe { ffi::FlushInstructionCache(self.hp, std::ptr::null(), 0) };
                    } else if !bp {
                        status = ffi::DBG_EXCEPTION_NOT_HANDLED;
                    }
                }
                _ => {}
            }
            unsafe { ffi::ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, status) };
        }
    }

    fn regions(&self) -> Vec<Mbi> {
        let mut out = vec![];
        let mut a: usize = 0x10000;
        while a < 0x8000_0000 {
            let mut m = Mbi::default();
            let r = unsafe { ffi2::VirtualQueryEx(self.hp, a as *const c_void, &mut m, std::mem::size_of::<Mbi>()) };
            if r == 0 || m.size == 0 {
                break;
            }
            out.push(m);
            a = m.base + m.size;
        }
        out
    }

    fn excluded(&self, base: u32, size: u32) -> bool {
        self.exclude.iter().any(|&(s, e)| base < e && s < base + size)
    }

    fn take_snapshot(&mut self) -> Result<()> {
        // TEBs / PEBs and the 64-bit stack must not be touched
        let mut tbi = ThreadBasicInfo::default();
        let mut ret = 0u32;
        let th = unsafe { ffi::OpenThread(ffi::THREAD_ALL_ACCESS, 0, self.tid) };
        unsafe {
            ffi2::NtQueryInformationThread(th, 0, &mut tbi as *mut _ as *mut c_void, std::mem::size_of::<ThreadBasicInfo>() as u32, &mut ret);
            ffi::CloseHandle(th);
        }
        let teb64 = tbi.teb as u32;
        let teb32 = teb64 + 0x2000;
        let stack64 = (self.read(teb64 + 0x10, 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).unwrap_or(0),
            self.read(teb64 + 0x8, 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).unwrap_or(0));
        let peb32 = self.u32(teb32 + 0x30);
        self.exclude = vec![(teb64, teb32 + 0x1000), (peb32.saturating_sub(0x1000), peb32 + 0x1000)];
        self.teb_peb = [teb32, peb32].iter().filter_map(|&a| self.read(a, 0x1000).map(|b| (a, b))).collect();
        if stack64.0 != 0 && stack64.1 > stack64.0 {
            self.exclude.push(stack64);
        }
        let image_end = 0x0060_0000u32;
        let mut snap = vec![];
        let mut total = 0usize;
        for m in self.regions() {
            let (base, size) = (m.base as u32, m.size as u32);
            if m.typ == MEM_PRIVATE && m.state != MEM_FREE {
                self.known.insert(m.alloc_base as u32);
                if m.state == MEM_RESERVE {
                    self.reserved.push((base, base + size));
                }
                if (base..base + size).contains(&self.ctx.Esp) {
                    self.stack_alloc = m.alloc_base as u32;
                }
            }
            if m.state != MEM_COMMIT || !writable(m.protect) {
                continue;
            }
            let ok_type = m.typ == MEM_PRIVATE || (m.typ == MEM_IMAGE && base >= 0x40_0000 && base < image_end);
            if !ok_type || self.excluded(base, size) {
                continue;
            }
            if let Some(b) = self.read(base, size as usize) {
                total += b.len();
                snap.push((base, b));
            }
        }
        self.snapshot = snap;
        self.snapshot_bytes = total;
        Ok(())
    }

    /// Make `[a, a+len)` committed and writable, sub-region by sub-region.
    fn recommit(&self, a: u32, len: usize) {
        let end = a as usize + len;
        let mut p = a as usize;
        while p < end {
            let mut m = Mbi::default();
            let r = unsafe { ffi2::VirtualQueryEx(self.hp, p as *const c_void, &mut m, std::mem::size_of::<Mbi>()) };
            if r == 0 || m.size == 0 {
                break;
            }
            let sub_end = (m.base + m.size).min(end);
            let n = sub_end - p;
            unsafe {
                match m.state {
                    MEM_COMMIT => {
                        if !writable(m.protect) || m.protect & 0x100 != 0 {
                            let mut old = 0u32;
                            ffi2::VirtualProtectEx(self.hp, p as *mut c_void, n, 0x04, &mut old);
                        }
                    }
                    MEM_RESERVE => {
                        ffi2::VirtualAllocEx(self.hp, p as *mut c_void, n, MEM_COMMIT, 0x04);
                    }
                    _ => {
                        ffi2::VirtualAllocEx(self.hp, p as *mut c_void, n, MEM_COMMIT | MEM_RESERVE, 0x04);
                    }
                }
            }
            p = sub_end;
        }
    }

    fn restore(&mut self) -> Result<()> {
        if std::env::var_os("MWDEC_PERSIST_DEBUG").is_some() {
            for (a, old) in &self.teb_peb {
                if let Some(now) = self.read(*a, old.len()) {
                    for k in (0..old.len()).step_by(4) {
                        if old[k..k + 4] != now[k..k + 4] {
                            eprintln!("persist: {:#x}+{k:#x}: {:02x?} -> {:02x?}", a, &old[k..k + 4], &now[k..k + 4]);
                        }
                    }
                }
            }
        }
        // release private regions allocated after the snapshot, and decommit pages committed since
        // (the heaps expect pages they commit again to read as zero)
        for m in self.regions() {
            let (base, size) = (m.base as u32, m.size as u32);
            if m.typ != MEM_PRIVATE || m.state != MEM_COMMIT || self.excluded(base, size) {
                continue;
            }
            let ab = m.alloc_base as u32;
            if !self.known.contains(&ab) {
                unsafe { ffi2::VirtualFreeEx(self.hp, m.alloc_base as *mut c_void, 0, MEM_RELEASE) };
                continue;
            }
            if ab == self.stack_alloc {
                continue;
            }
            for &(rs, re) in &self.reserved {
                let (s0, e0) = (rs.max(base), re.min(base + size));
                if s0 < e0 {
                    unsafe { ffi2::VirtualFreeEx(self.hp, s0 as usize as *mut c_void, (e0 - s0) as usize, MEM_DECOMMIT) };
                }
            }
        }
        // thread/process state outside the snapshot that the C runtime changes at exit: the TLS
        // slots (TEB 0xe10..0xf10) and the last error (TEB 0x34, 0xbf4) of the 32-bit thread, and the
        // TLS allocation bitmap (PEB 0x44..0x4c)
        if let [(teb, tb), (peb, pb)] = &self.teb_peb[..] {
            for &(o, n) in &[(0x34usize, 4usize), (0xbf4, 4), (0xe10, 0x100)] {
                self.write(*teb + o as u32, &tb[o..o + n]);
            }
            self.write(*peb + 0x44, &pb[0x44..0x4c]);
        }
        for (a, b) in &self.snapshot {
            if !self.write(*a, b) {
                // part of the region was decommitted, released or re-protected since the snapshot:
                // bring every page of it back to committed read/write, then write again
                self.recommit(*a, b.len());
                if !self.write(*a, b) {
                    bail!("restore failed at {a:#x} ({} bytes)", b.len());
                }
            }
        }
        let th = unsafe { ffi::OpenThread(ffi::THREAD_ALL_ACCESS, 0, self.tid) };
        let mut c = self.ctx;
        c.ContextFlags = WOW64_CONTEXT_ALL;
        unsafe {
            ffi::Wow64SetThreadContext(th, &c);
            ffi::CloseHandle(th);
        }
        Ok(())
    }

    /// (peak working set, committed private bytes) of the compiler process, in bytes.
    pub fn memory(&self) -> Option<(usize, usize)> {
        let mut m = MemCounters { cb: std::mem::size_of::<MemCounters>() as u32, ..Default::default() };
        let ok = unsafe { ffi2::K32GetProcessMemoryInfo(self.hp, &mut m, m.cb) };
        (ok != 0).then_some((m.peak_working_set, m.pagefile))
    }

    /// Compile `body` (the text after the context) and return the object (exit status 0) or an
    /// error with the exit status. A compile that produced no object for another reason is
    /// retried once (the object file can be briefly locked by another process, e.g. a virus
    /// scanner, after the previous compile); a real compile error (nonzero status) is not.
    pub fn compile(&mut self, body: &str) -> Result<CompileOutput> {
        match self.compile_once(body) {
            Ok(o) => Ok(o),
            Err(e) if self.stopped.is_some() && body.len() <= self.capacity && !e.to_string().starts_with(COMPILE_FAILED) => {
                self.retries += 1;
                std::thread::sleep(std::time::Duration::from_millis(20));
                let r = self.compile_once(body).map_err(|e2| e2.context(format!("first attempt: {e:#}")));
                // crashed again from the restored snapshot: the snapshot itself may be bad (a
                // loaded machine), so the instance is replaced by a fresh one and the candidate
                // compiled once more; if that fails too, the instance ends (callers fall back to a
                // normal compile)
                // (or the process died: exited, or not stopped at its snapshot any more)
                let dead = |m: &str| m.matches("compiler exception").count() >= 2 || m.contains("exited") || m.contains("not stopped");
                if r.as_ref().is_err_and(|e2| dead(&format!("{e2:#}"))) {
                    if !self.exited {
                        unsafe { self.terminate() };
                    }
                    if let Some((c, ctx, cap, fname)) = self.restart.clone() {
                        if let Ok(fresh) = Self::start_as(&c, &ctx, cap, fname.as_deref()) {
                            let (compiles, retries) = (self.compiles, self.retries);
                            *self = fresh;
                            self.compiles = compiles;
                            self.retries = retries + 1;
                            return self.compile_once(body).map_err(|e3| e3.context("after a restart"));
                        }
                    }
                }
                r
            }
            // the process ended under the candidate (killed, out of memory): a fresh instance
            Err(e) if format!("{e:#}").contains("exited") && body.len() <= self.capacity => match self.restart.clone() {
                Some((c, ctx, cap, fname)) => match Self::start_as(&c, &ctx, cap, fname.as_deref()) {
                    Ok(fresh) => {
                        let (compiles, retries) = (self.compiles, self.retries);
                        *self = fresh;
                        self.compiles = compiles;
                        self.retries = retries + 1;
                        self.compile_once(body).map_err(|e3| e3.context(format!("after a restart: {e:#}")))
                    }
                    Err(_) => Err(e),
                },
                None => Err(e),
            },
            Err(e) => Err(e),
        }
    }

    fn compile_once(&mut self, body: &str) -> Result<CompileOutput> {
        if body.len() > self.capacity {
            bail!("candidate ({} bytes) exceeds the slot ({} bytes)", body.len(), self.capacity);
        }
        let Some((pid, tid)) = self.stopped else { bail!("persistent compiler is not stopped (an earlier compile hung or the process died)") };
        let t_restore = std::time::Instant::now();
        self.restore()?;
        self.restore_time += t_restore.elapsed();
        // the in-memory source has '\r' line ends
        let mut slot: Vec<u8> = body.replace("\r\n", "\r").replace('\n', "\r").into_bytes();
        slot.resize(self.capacity, b' ');
        if !self.write(self.slot, &slot) {
            bail!("cannot write the candidate slot");
        }
        self.stopped = None;
        if std::env::var_os("MWDEC_PERSIST_DEBUG").is_some() {
            let back = self.read(self.slot.saturating_sub(48), 120).unwrap_or_default();
            eprintln!("persist: slot {:#x} pos-now {:#x}: {:?}", self.slot, self.u32(A_TEXT_POS), String::from_utf8_lossy(&back));
        }
        let _ = std::fs::remove_file(&self.obj);
        // MWDEC_PERSIST_PROFILE: sample the compiler's instruction pointer every millisecond
        let prof_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let prof = std::env::var_os("MWDEC_PERSIST_PROFILE").is_some().then(|| {
            let stop = prof_stop.clone();
            let tid = self.tid;
            let hp = self.hp as usize;
            std::thread::spawn(move || {
                let rd = |a: u32| -> u32 {
                    let mut b = [0u8; 4];
                    let mut got = 0usize;
                    unsafe { ffi::ReadProcessMemory(hp as ffi::HANDLE, a as usize as *const _, b.as_mut_ptr() as *mut _, 4, &mut got) };
                    u32::from_le_bytes(b)
                };
                let mut hist = std::collections::HashMap::<u32, u32>::new();
                let th = unsafe { ffi::OpenThread(ffi::THREAD_ALL_ACCESS, 0, tid) };
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    unsafe {
                        if ffi2::SuspendThread(th) == u32::MAX {
                            break;
                        }
                        let mut c: ffi::WOW64_CONTEXT = std::mem::zeroed();
                        c.ContextFlags = ffi::WOW64_CONTEXT_CONTROL | ffi::WOW64_CONTEXT_INTEGER;
                        if ffi::Wow64GetThreadContext(th, &mut c) != 0 {
                            // first return address inside the compiler (scan the stack)
                            let deep = std::env::var("MWDEC_PERSIST_PROFILE").map_or(1, |v| v.parse::<usize>().unwrap_or(1));
                            let mut sites = vec![];
                            if (0x401000..0x580000).contains(&c.Eip) {
                                sites.push(c.Eip);
                            }
                            for k in 0..1024u32 {
                                if sites.len() >= deep {
                                    break;
                                }
                                let v = rd(c.Esp + 4 * k);
                                if (0x401000..0x580000).contains(&v) {
                                    sites.push(v);
                                }
                            }
                            for s in sites {
                                *hist.entry(s).or_default() += 1;
                            }
                        }
                        ffi2::ResumeThread(th);
                    }
                }
                unsafe { ffi::CloseHandle(th) };
                hist
            })
        });
        if let Ok(m) = std::env::var("MWDEC_PERSIST_MARK") {
            for a in m.split(',').filter_map(|x| u32::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok()) {
                if !self.marks.contains_key(&a) {
                    let o = self.read(a, 1).map(|b| b[0]).unwrap_or(0);
                    self.marks.insert(a, o);
                }
                self.write(a, &[0xcc]);
            }
            unsafe { ffi::FlushInstructionCache(self.hp, std::ptr::null(), 0) };
        }
        unsafe { ffi::ContinueDebugEvent(pid, tid, ffi::DBG_CONTINUE) };
        let r = self.run_compile();
        prof_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = prof {
            if let Ok(hist) = h.join() {
                let mut v: Vec<(u32, u32)> = hist.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1));
                let total: u32 = v.iter().map(|x| x.1).sum();
                eprintln!("persist profile: {total} samples; top: {:x?}", &v[..v.len().min(40)]);
            }
        }
        r
    }

    fn run_compile(&mut self) -> Result<CompileOutput> {
        let start = std::time::Instant::now();
        loop {
            let mut ev: ffi::DEBUG_EVENT = unsafe { std::mem::zeroed() };
            if unsafe { ffi::WaitForDebugEvent(&mut ev, 30_000) } == 0 || start.elapsed().as_secs() > 60 {
                bail!("persistent compile timed out");
            }
            match ev.dwDebugEventCode {
                ffi::EXIT_PROCESS_DEBUG_EVENT => {
                    unsafe { ffi::ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, ffi::DBG_CONTINUE) };
                    self.exited = true;
                    bail!("compiler process exited")
                }
                ffi::EXCEPTION_DEBUG_EVENT => {
                    let er = unsafe { ev.u.Exception.ExceptionRecord };
                    let addr = er.ExceptionAddress as usize as u32;
                    let code = er.ExceptionCode;
                    let bp = code == ffi::STATUS_BREAKPOINT || code == ffi::STATUS_WX86_BREAKPOINT;
                    if bp && addr == BP_EXIT {
                        self.stopped = Some((ev.dwProcessId, ev.dwThreadId));
                        self.compiles += 1;
                        let th = unsafe { ffi::OpenThread(ffi::THREAD_ALL_ACCESS, 0, ev.dwThreadId) };
                        let mut c: ffi::WOW64_CONTEXT = unsafe { std::mem::zeroed() };
                        c.ContextFlags = ffi::WOW64_CONTEXT_CONTROL | ffi::WOW64_CONTEXT_INTEGER;
                        unsafe {
                            ffi::Wow64GetThreadContext(th, &mut c);
                            ffi::CloseHandle(th);
                        }
                        let status = c.Eax;
                        if status != 0 {
                            bail!("{COMPILE_FAILED} (exit status {status}); recompile normally for the messages");
                        }
                        let object = std::fs::read(&self.obj).context("persistent compile: no object")?;
                        return Ok(CompileOutput { object, messages: String::new() });
                    }
                    if bp {
                        if let Some(&o) = self.marks.get(&addr) {
                            eprintln!("persist mark {addr:#x} +{:.1} ms", start.elapsed().as_secs_f64() * 1000.0);
                            self.write(addr, &[o]);
                            let th = unsafe { ffi::OpenThread(ffi::THREAD_ALL_ACCESS, 0, ev.dwThreadId) };
                            let mut c: ffi::WOW64_CONTEXT = unsafe { std::mem::zeroed() };
                            c.ContextFlags = ffi::WOW64_CONTEXT_CONTROL;
                            unsafe {
                                ffi::FlushInstructionCache(self.hp, std::ptr::null(), 0);
                                ffi::Wow64GetThreadContext(th, &mut c);
                                c.Eip = addr;
                                ffi::Wow64SetThreadContext(th, &c);
                                ffi::CloseHandle(th);
                            }
                        }
                    }
                    if code == STATUS_INVALID_HANDLE {
                        // raised only because a debugger is attached (closing a stale handle on the
                        // compiler's error path); without one the call just fails, so carry on
                        unsafe { ffi::ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, ffi::DBG_CONTINUE) };
                        continue;
                    }
                    if !bp {
                        // a crash inside the compiler: report; the next compile restores the snapshot
                        self.stopped = Some((ev.dwProcessId, ev.dwThreadId));
                        bail!("compiler exception {code:#x} at {addr:#x}");
                    }
                    unsafe { ffi::ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, ffi::DBG_CONTINUE) };
                }
                _ => unsafe {
                    ffi::ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, ffi::DBG_CONTINUE);
                },
            }
        }
    }
}

impl Drop for PersistentCompiler {
    fn drop(&mut self) {
        unsafe {
            if !self.exited {
                self.terminate();
            }
            ffi::CloseHandle(self.ht);
            ffi::CloseHandle(self.hp);
        }
        let _ = self.pid;
        let _ = std::fs::remove_file(&self.src);
        let _ = std::fs::remove_file(&self.obj);
        if let Some(dir) = self.src.parent().filter(|d| d.file_name().is_some_and(|n| n.to_string_lossy().starts_with("pc"))) {
            let _ = std::fs::remove_dir(dir);
        }
    }
}

impl PersistentCompiler {
    /// Kill the process and consume its debug events up to the exit event.
    unsafe fn terminate(&mut self) {
        unsafe {
            ffi::TerminateProcess(self.hp, 1);
            if let Some((pid, tid)) = self.stopped.take() {
                ffi::ContinueDebugEvent(pid, tid, ffi::DBG_CONTINUE);
            }
            // drain until the exit event so the process handle is released cleanly
            let start = std::time::Instant::now();
            loop {
                let mut ev: ffi::DEBUG_EVENT = std::mem::zeroed();
                if ffi::WaitForDebugEvent(&mut ev, 2000) == 0 || start.elapsed().as_secs() > 5 {
                    break;
                }
                let done = ev.dwDebugEventCode == ffi::EXIT_PROCESS_DEBUG_EVENT;
                ffi::ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, ffi::DBG_CONTINUE);
                if done {
                    break;
                }
            }
        }
        self.exited = true;
    }
}

// ------------------------------------------------------------------ per-thread cache

struct CacheEntry {
    key: u64,
    /// None: starting failed for this key (do not retry every call)
    pc: Option<PersistentCompiler>,
}

thread_local! {
    static CACHE: std::cell::RefCell<Vec<CacheEntry>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Instances kept per thread (each is one idle compiler process of a few tens of MB).
pub const CACHE_PER_THREAD: usize = 2;
const DEFAULT_CAPACITY: usize = 64 * 1024;

fn cache_key(comp: &Compiler, context: &str, file_name: Option<&str>, capacity: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    comp.exe().hash(&mut h);
    comp.flags().hash(&mut h);
    comp.work.hash(&mut h);
    context.hash(&mut h);
    file_name.hash(&mut h);
    capacity.hash(&mut h);
    h.finish()
}

/// Compile `context` + `body` through a persistent compiler cached on the calling thread
/// (started on first use for this compiler/flags/context/file name, at most
/// [`CACHE_PER_THREAD`] per thread, least recently used dropped). An error means "use a normal
/// compile": a context that fails to start is remembered and fails fast afterwards; an instance
/// whose process became unusable is dropped and restarted on the next call. Bodies larger than
/// the default slot (64 KB) get an instance with a larger slot.
pub fn compile_cached(comp: &Compiler, context: &str, file_name: Option<&str>, body: &str) -> Result<CompileOutput> {
    let mut capacity = DEFAULT_CAPACITY;
    while capacity < body.len() {
        capacity *= 2;
    }
    let key = cache_key(comp, context, file_name, capacity);
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let idx = match cache.iter().position(|e| e.key == key) {
            Some(i) => i,
            None => {
                while cache.len() >= CACHE_PER_THREAD {
                    cache.remove(0);
                }
                let pc = PersistentCompiler::start_as(comp, context, capacity, file_name).ok();
                cache.push(CacheEntry { key, pc });
                cache.len() - 1
            }
        };
        // most recently used last
        let entry = cache.remove(idx);
        cache.push(entry);
        let last = cache.len() - 1;
        let Some(pc) = cache[last].pc.as_mut() else { bail!("persistent compiler unavailable for this context") };
        let r = pc.compile(body);
        if r.is_err() && pc.stopped.is_none() {
            cache.remove(last);
        }
        r
    })
}

/// Drop this thread's cached persistent compilers (their processes end).
pub fn clear_cache() {
    CACHE.with(|c| c.borrow_mut().clear());
}
