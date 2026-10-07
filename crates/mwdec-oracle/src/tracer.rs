//! Real-compiler tracer: runs the unmodified GC/2.7 `mwcceppc.exe` under a minimal Win32 debugger
//! (WOW64, int3 breakpoints in the debuggee's memory only) and reads the compiler's own data
//! structures at chosen points:
//!
//! - **inliner**: every `CInline_InlineFunctionCheck` decision (caller, callee, expansion level,
//!   estimated cost vs `inline_max_size`, result and the code path that decided it), and the order
//!   in which function bodies are expanded/generated (deferred inlining order);
//! - **colouring**: per function and register class, every interference-graph node: vreg number,
//!   variable name (params / named locals / FE temps), value kind, IG degree, assigned register,
//!   coalescing, and the exact colouring (pop) order;
//! - **PCode** (optional): the backend instruction stream with virtual registers at three stages.
//!
//! Addresses/layouts are for GC/2.7 only (see learnings/mwcc-codegen/feasibility.md). Diagnostic
//! only: matching always uses the plain compiler.

use crate::compile::Compiler;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};

// ------------------------------------------------------------------ GC/2.7 address table
const BP_EXPAND: u32 = 0x55ee80; // CInline_Expand(stmt); expanding_function = [0x5e3ada]
const BP_IFC: u32 = 0x560c80; // CInline_InlineFunctionCheck(ENode*)
const BP_IFC_SIZE: u32 = 0x560d26; // after CInline_EstimateSizeOfFunc: eax = size, [esp+8] = level
const IFC_RETS: [(u32, &str); 9] = [
    (0x560cdd, "always_inline"),
    (0x560cf3, "no_body_yet"),
    (0x560d34, "too_big"),
    (0x560d53, "no_body_yet"),
    (0x560d61, "stmt_limit_10"),
    (0x560d75, "stmt_limit_7"),
    (0x560d89, "stmt_limit_3"),
    (0x560d93, "ok"),
    (0x560d97, "not_inline_candidate"),
];
const BP_GLOBALOPT: u32 = 0x500420;
const BP_SCHED: u32 = 0x507b90;
const BP_COLORGRAPH: u32 = 0x5087b0;
const BP_AFTER_COLOR: u32 = 0x508616;
const A_EXPANDING_FUNC: u32 = 0x5e3ada;
const A_INLINE_MAX: u32 = 0x5e3ae0;
const A_CINLINE_LEVEL: u32 = 0x5e3b4c;
const A_CLASS: u32 = 0x5ef2cf;
const A_IG: u32 = 0x5ea768;
const A_USED: u32 = 0x5eaa2c;
const A_NREAL: u32 = 0x5ea710;
const A_FIRST_FE_TEMP: u32 = 0x5ef210; // short[class]
const A_FIRST_TEMP: u32 = 0x5e9f58; // int[class]
const A_ARGUMENTS: u32 = 0x5eaa28; // ObjectList*
const A_BLOCKS: u32 = 0x5ea748;
const A_OPINFO: u32 = 0x5c0fa8;

// ------------------------------------------------------------------ results
#[derive(Clone, Debug, Serialize)]
pub struct InlineDecision {
    /// function whose body was being expanded
    pub caller: String,
    pub callee: String,
    /// expansion pass (cinline_level) of the caller
    pub pass: i16,
    /// depth budget passed to the size estimate
    pub level: Option<i32>,
    /// estimated cost of the callee's (expanded) body
    pub cost: Option<i32>,
    pub max: i32,
    pub inlined: bool,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VregKind {
    Param,
    Named,
    FeTemp,
    Temp,
}

#[derive(Clone, Debug, Serialize)]
pub struct IgNode {
    pub vreg: u16,
    pub name: Option<String>,
    pub kind: VregKind,
    /// IG adjacency size (physical + virtual neighbours) before simplification
    pub degree: u16,
    /// final register number (0..31) or None if spilled/unresolved
    pub reg: Option<u8>,
    pub coalesced_into: Option<i16>,
    pub spilled: bool,
    pub neighbours: Vec<i16>,
    /// position in colouring order (0 = coloured first); None for coalesced nodes
    pub order: Option<usize>,
    /// pushed in a later simplify pass (degree >= K), i.e. coloured ahead of its vreg rank
    pub blocked: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ColorRound {
    pub function: String,
    /// "GPR" or "FPR"
    pub class: String,
    pub nreal: u32,
    pub first_fe_temp: u32,
    pub first_temp: u32,
    pub nodes: Vec<IgNode>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PcodeDump {
    pub stage: String,
    pub function: String,
    pub lines: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Trace {
    pub inline: Vec<InlineDecision>,
    /// functions in the order their bodies were expanded (= code generation order)
    pub expand_order: Vec<String>,
    pub rounds: Vec<ColorRound>,
    pub pcode: Vec<PcodeDump>,
    #[serde(skip)]
    pub object: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct TraceOptions {
    pub inline: bool,
    pub coloring: bool,
    /// any of "initial", "presched", "precolor"
    pub pcode: Vec<String>,
}

// ------------------------------------------------------------------ Win32 FFI (no crates)
#[allow(non_snake_case, non_camel_case_types, dead_code)]
mod ffi {
    use std::ffi::c_void;
    pub type HANDLE = *mut c_void;
    #[repr(C)]
    pub struct STARTUPINFOW {
        pub cb: u32,
        pub lpReserved: *mut u16,
        pub lpDesktop: *mut u16,
        pub lpTitle: *mut u16,
        pub dwX: u32,
        pub dwY: u32,
        pub dwXSize: u32,
        pub dwYSize: u32,
        pub dwXCountChars: u32,
        pub dwYCountChars: u32,
        pub dwFillAttribute: u32,
        pub dwFlags: u32,
        pub wShowWindow: u16,
        pub cbReserved2: u16,
        pub lpReserved2: *mut u8,
        pub hStdInput: HANDLE,
        pub hStdOutput: HANDLE,
        pub hStdError: HANDLE,
    }
    #[repr(C)]
    pub struct PROCESS_INFORMATION {
        pub hProcess: HANDLE,
        pub hThread: HANDLE,
        pub dwProcessId: u32,
        pub dwThreadId: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct EXCEPTION_RECORD {
        pub ExceptionCode: u32,
        pub ExceptionFlags: u32,
        pub ExceptionRecord: *mut c_void,
        pub ExceptionAddress: *mut c_void,
        pub NumberParameters: u32,
        pub ExceptionInformation: [usize; 15],
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct EXCEPTION_DEBUG_INFO {
        pub ExceptionRecord: EXCEPTION_RECORD,
        pub dwFirstChance: u32,
    }
    #[repr(C)]
    pub union DEBUG_EVENT_U {
        pub Exception: EXCEPTION_DEBUG_INFO,
        pub raw: [u8; 160],
    }
    #[repr(C)]
    pub struct DEBUG_EVENT {
        pub dwDebugEventCode: u32,
        pub dwProcessId: u32,
        pub dwThreadId: u32,
        pub u: DEBUG_EVENT_U,
    }
    #[repr(C)]
    pub struct WOW64_CONTEXT {
        pub ContextFlags: u32,
        pub Dr: [u32; 6],
        pub FloatSave: [u8; 112],
        pub SegGs: u32,
        pub SegFs: u32,
        pub SegEs: u32,
        pub SegDs: u32,
        pub Edi: u32,
        pub Esi: u32,
        pub Ebx: u32,
        pub Edx: u32,
        pub Ecx: u32,
        pub Eax: u32,
        pub Ebp: u32,
        pub Eip: u32,
        pub SegCs: u32,
        pub EFlags: u32,
        pub Esp: u32,
        pub SegSs: u32,
        pub ExtendedRegisters: [u8; 512],
    }
    #[link(name = "kernel32")]
    extern "system" {
        pub fn CreateProcessW(
            app: *const u16,
            cmd: *mut u16,
            pa: *mut c_void,
            ta: *mut c_void,
            inherit: i32,
            flags: u32,
            env: *mut c_void,
            cwd: *const u16,
            si: *mut STARTUPINFOW,
            pi: *mut PROCESS_INFORMATION,
        ) -> i32;
        pub fn WaitForDebugEvent(ev: *mut DEBUG_EVENT, ms: u32) -> i32;
        pub fn ContinueDebugEvent(pid: u32, tid: u32, status: u32) -> i32;
        pub fn ReadProcessMemory(h: HANDLE, addr: *const c_void, buf: *mut c_void, n: usize, got: *mut usize) -> i32;
        pub fn WriteProcessMemory(h: HANDLE, addr: *mut c_void, buf: *const c_void, n: usize, got: *mut usize) -> i32;
        pub fn FlushInstructionCache(h: HANDLE, addr: *const c_void, n: usize) -> i32;
        pub fn OpenThread(access: u32, inherit: i32, tid: u32) -> HANDLE;
        pub fn Wow64GetThreadContext(h: HANDLE, c: *mut WOW64_CONTEXT) -> i32;
        pub fn Wow64SetThreadContext(h: HANDLE, c: *const WOW64_CONTEXT) -> i32;
        pub fn CloseHandle(h: HANDLE) -> i32;
        pub fn TerminateProcess(h: HANDLE, code: u32) -> i32;
        pub fn GetExitCodeProcess(h: HANDLE, code: *mut u32) -> i32;
    }
    pub const DEBUG_ONLY_THIS_PROCESS: u32 = 2;
    pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    pub const EXCEPTION_DEBUG_EVENT: u32 = 1;
    pub const CREATE_PROCESS_DEBUG_EVENT: u32 = 3;
    pub const EXIT_PROCESS_DEBUG_EVENT: u32 = 5;
    pub const DBG_CONTINUE: u32 = 0x0001_0002;
    pub const DBG_EXCEPTION_NOT_HANDLED: u32 = 0x8001_0001;
    pub const STATUS_BREAKPOINT: u32 = 0x8000_0003;
    pub const STATUS_SINGLE_STEP: u32 = 0x8000_0004;
    pub const STATUS_WX86_BREAKPOINT: u32 = 0x4000_001F;
    pub const STATUS_WX86_SINGLE_STEP: u32 = 0x4000_001E;
    pub const WOW64_CONTEXT_CONTROL: u32 = 0x0001_0001;
    pub const WOW64_CONTEXT_INTEGER: u32 = 0x0001_0002;
    pub const THREAD_ALL_ACCESS: u32 = 0x001F_FFFF;
}

struct Debuggee {
    h: ffi::HANDLE,
    bps: HashMap<u32, u8>,
    rearm: HashMap<u32, u32>,
    names: HashMap<u32, String>,
}

impl Debuggee {
    fn read(&self, addr: u32, n: usize) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; n];
        let mut got = 0usize;
        let ok = unsafe {
            ffi::ReadProcessMemory(self.h, addr as usize as *const _, buf.as_mut_ptr() as *mut _, n, &mut got)
        };
        if ok == 0 || got != n {
            None
        } else {
            Some(buf)
        }
    }
    fn write(&self, addr: u32, data: &[u8]) {
        let mut got = 0usize;
        unsafe {
            ffi::WriteProcessMemory(self.h, addr as usize as *mut _, data.as_ptr() as *const _, data.len(), &mut got);
            ffi::FlushInstructionCache(self.h, addr as usize as *const _, data.len());
        }
    }
    fn u32(&self, a: u32) -> u32 {
        self.read(a, 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).unwrap_or(0)
    }
    fn u16(&self, a: u32) -> u16 {
        self.read(a, 2).map(|b| u16::from_le_bytes([b[0], b[1]])).unwrap_or(0)
    }
    fn i16(&self, a: u32) -> i16 {
        self.u16(a) as i16
    }
    fn set_bp(&mut self, a: u32) {
        if let Some(b) = self.read(a, 1) {
            self.bps.insert(a, b[0]);
            self.write(a, &[0xcc]);
        }
    }
    fn cstr(&self, a: u32, max: usize) -> String {
        match self.read(a, max) {
            Some(b) => {
                let e = b.iter().position(|&c| c == 0).unwrap_or(b.len());
                String::from_utf8_lossy(&b[..e]).into_owned()
            }
            None => "?".into(),
        }
    }
    /// Object* -> name (Object.name @0xa -> HashNameNode, name @0xa).
    fn obj_name(&mut self, obj: u32) -> String {
        if obj == 0 {
            return "?".into();
        }
        if let Some(n) = self.names.get(&obj) {
            return n.clone();
        }
        let hn = self.u32(obj + 0xa);
        let n = if hn == 0 { "?".into() } else { self.cstr(hn + 0xa, 128) };
        self.names.insert(obj, n.clone());
        n
    }
}

fn quote_arg(a: &str) -> String {
    if a.is_empty() || a.contains(' ') || a.contains('"') {
        format!("\"{}\"", a.replace('"', "\\\""))
    } else {
        a.to_string()
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Trace one compile of `source` with the compiler's flag profile.
pub fn trace_source(comp: &Compiler, source: &str, opts: &TraceOptions) -> Result<Trace> {
    let ver = comp.version.clone().unwrap_or_else(|| comp.profile.version().to_string());
    if ver != "GC/2.7" {
        bail!("tracer address table only exists for GC/2.7 (got {ver})");
    }
    std::fs::create_dir_all(&comp.work)?;
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stem = format!("tr{}_{}", std::process::id(), n);
    let ext = if comp.profile.is_c() { "c" } else { "cpp" };
    let src = comp.work.join(format!("{stem}.{ext}"));
    let obj = comp.work.join(format!("{stem}.o"));
    std::fs::write(&src, source)?;
    let mut parts = vec![quote_arg(&comp.exe().to_string_lossy().replace('/', "\\"))];
    parts.extend(comp.flags().iter().map(|f| quote_arg(f)));
    parts.push("-c".into());
    parts.push(quote_arg(&src.to_string_lossy()));
    parts.push("-o".into());
    parts.push(quote_arg(&obj.to_string_lossy()));
    let cmd = parts.join(" ");
    let res = run_debugged(&cmd, &comp.work.to_string_lossy(), opts);
    let _ = std::fs::remove_file(&src);
    let mut trace = res?;
    trace.object = std::fs::read(&obj).with_context(|| "tracer: compiler produced no object (compile error?)")?;
    let _ = std::fs::remove_file(&obj);
    Ok(trace)
}

struct PendingInline {
    caller: String,
    callee: String,
    pass: i16,
    level: Option<i32>,
    cost: Option<i32>,
    max: i32,
}

fn run_debugged(cmd: &str, cwd: &str, opts: &TraceOptions) -> Result<Trace> {
    let mut si: ffi::STARTUPINFOW = unsafe { std::mem::zeroed() };
    si.cb = std::mem::size_of::<ffi::STARTUPINFOW>() as u32;
    let mut pi: ffi::PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    let mut cmdw = wide(cmd);
    let cwdw = wide(cwd);
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
    let mut d = Debuggee { h: pi.hProcess, bps: HashMap::new(), rearm: HashMap::new(), names: HashMap::new() };
    let mut t = Trace::default();
    let mut current_func = String::new();
    let mut pending: Option<PendingInline> = None;
    let mut pending_order: Option<(u8, Vec<u16>)> = None;
    let start = std::time::Instant::now();
    loop {
        let mut ev: ffi::DEBUG_EVENT = unsafe { std::mem::zeroed() };
        if unsafe { ffi::WaitForDebugEvent(&mut ev, 60_000) } == 0 || start.elapsed().as_secs() > 300 {
            unsafe { ffi::TerminateProcess(pi.hProcess, 1) };
            bail!("tracer: timeout waiting for the compiler");
        }
        let mut status = ffi::DBG_CONTINUE;
        match ev.dwDebugEventCode {
            ffi::CREATE_PROCESS_DEBUG_EVENT => {
                if opts.coloring {
                    d.set_bp(BP_COLORGRAPH);
                    d.set_bp(BP_AFTER_COLOR);
                }
                if opts.inline || opts.coloring || !opts.pcode.is_empty() {
                    d.set_bp(BP_EXPAND);
                }
                if opts.inline {
                    d.set_bp(BP_IFC);
                    d.set_bp(BP_IFC_SIZE);
                    for (a, _) in IFC_RETS {
                        d.set_bp(a);
                    }
                }
                if opts.pcode.iter().any(|s| s == "initial") {
                    d.set_bp(BP_GLOBALOPT);
                }
                if opts.pcode.iter().any(|s| s == "presched") {
                    d.set_bp(BP_SCHED);
                }
                if opts.pcode.iter().any(|s| s == "precolor") && !opts.coloring {
                    d.set_bp(BP_COLORGRAPH);
                }
            }
            ffi::EXIT_PROCESS_DEBUG_EVENT => break,
            ffi::EXCEPTION_DEBUG_EVENT => {
                let er = unsafe { ev.u.Exception.ExceptionRecord };
                let addr = er.ExceptionAddress as usize as u32;
                let code = er.ExceptionCode;
                if (code == ffi::STATUS_BREAKPOINT || code == ffi::STATUS_WX86_BREAKPOINT) && d.bps.contains_key(&addr) {
                    let th = unsafe { ffi::OpenThread(ffi::THREAD_ALL_ACCESS, 0, ev.dwThreadId) };
                    let mut c: ffi::WOW64_CONTEXT = unsafe { std::mem::zeroed() };
                    c.ContextFlags = ffi::WOW64_CONTEXT_CONTROL | ffi::WOW64_CONTEXT_INTEGER;
                    unsafe { ffi::Wow64GetThreadContext(th, &mut c) };
                    handle_bp(&mut d, &mut t, addr, &c, opts, &mut current_func, &mut pending, &mut pending_order);
                    // restore, rewind, single-step, re-arm
                    let orig = d.bps[&addr];
                    d.write(addr, &[orig]);
                    c.Eip = addr;
                    c.EFlags |= 0x100;
                    unsafe {
                        ffi::Wow64SetThreadContext(th, &c);
                        ffi::CloseHandle(th);
                    }
                    d.rearm.insert(ev.dwThreadId, addr);
                } else if (code == ffi::STATUS_SINGLE_STEP || code == ffi::STATUS_WX86_SINGLE_STEP)
                    && d.rearm.contains_key(&ev.dwThreadId)
                {
                    let a = d.rearm.remove(&ev.dwThreadId).unwrap();
                    d.write(a, &[0xcc]);
                } else if code == ffi::STATUS_BREAKPOINT || code == ffi::STATUS_WX86_BREAKPOINT {
                    // loader breakpoints
                } else {
                    status = ffi::DBG_EXCEPTION_NOT_HANDLED;
                }
            }
            _ => {}
        }
        unsafe { ffi::ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, status) };
    }
    unsafe {
        ffi::CloseHandle(pi.hThread);
        ffi::CloseHandle(pi.hProcess);
    }
    Ok(t)
}

#[allow(clippy::too_many_arguments)]
fn handle_bp(
    d: &mut Debuggee,
    t: &mut Trace,
    addr: u32,
    c: &ffi::WOW64_CONTEXT,
    opts: &TraceOptions,
    current_func: &mut String,
    pending: &mut Option<PendingInline>,
    pending_order: &mut Option<(u8, Vec<u16>)>,
) {
    let esp = c.Esp;
    match addr {
        BP_EXPAND => {
            let f = d.u32(A_EXPANDING_FUNC);
            *current_func = d.obj_name(f);
            t.expand_order.push(current_func.clone());
        }
        BP_IFC => {
            let expr = d.u32(esp + 4);
            let mut obj = d.u32(expr + 0xe);
            if d.read(obj + 2, 1).map(|b| b[0]) == Some(6) {
                obj = d.u32(obj + 0x2a); // DALIAS -> aliased object
            }
            *pending = Some(PendingInline {
                caller: current_func.clone(),
                callee: d.obj_name(obj),
                pass: d.i16(A_CINLINE_LEVEL),
                level: None,
                cost: None,
                max: d.u32(A_INLINE_MAX) as i32,
            });
        }
        BP_IFC_SIZE => {
            if let Some(p) = pending.as_mut() {
                p.cost = Some(c.Eax as i32);
                p.level = Some(d.u32(esp + 8) as i32);
            }
        }
        BP_GLOBALOPT | BP_SCHED => {
            if addr == BP_SCHED && d.u32(esp + 4) & 0xff != 0 {
                return;
            }
            let stage = if addr == BP_GLOBALOPT { "initial" } else { "presched" };
            let lines = dump_pcode(d);
            t.pcode.push(PcodeDump { stage: stage.into(), function: current_func.clone(), lines });
        }
        BP_COLORGRAPH => {
            let cls = d.read(A_CLASS, 1).map(|b| b[0]).unwrap_or(0);
            if opts.pcode.iter().any(|s| s == "precolor") {
                let f = d.obj_name(c.Ebp);
                t.pcode.push(PcodeDump {
                    stage: format!("precolor-{}", if cls == 3 { "fpr" } else { "gpr" }),
                    function: f,
                    lines: dump_pcode(d),
                });
            }
            let mut order = vec![];
            let mut p = d.u32(esp + 4);
            let mut guard = 0;
            while p != 0 && guard < 100_000 {
                order.push(d.u16(p + 0x10));
                p = d.u32(p);
                guard += 1;
            }
            *pending_order = Some((cls, order));
        }
        BP_AFTER_COLOR => {
            let Some((cls, order)) = pending_order.take() else { return };
            let ig = d.u32(A_IG);
            let used = d.u32(A_USED + 4 * cls as u32);
            let nreal = d.u32(A_NREAL + 4 * cls as u32);
            let first_fe = d.i16(A_FIRST_FE_TEMP + 2 * cls as u32) as u32;
            let first_tmp = d.u32(A_FIRST_TEMP + 4 * cls as u32);
            // params = objects on the `arguments` list
            let mut params: HashSet<u32> = HashSet::new();
            let mut l = d.u32(A_ARGUMENTS);
            let mut g = 0;
            while l != 0 && g < 1000 {
                params.insert(d.u32(l + 4));
                l = d.u32(l);
                g += 1;
            }
            let fname = d.obj_name(c.Ebp);
            let mut raw: BTreeMap<u16, (u32, i16, i16, u16, u16, Vec<i16>)> = BTreeMap::new();
            for i in nreal..used {
                let p = d.u32(ig + 4 * i);
                if p == 0 {
                    continue;
                }
                let Some(h) = d.read(p, 0x1a) else { continue };
                let spill = u32::from_le_bytes([h[4], h[5], h[6], h[7]]);
                let vreg = u16::from_le_bytes([h[0x10], h[0x11]]);
                let color = i16::from_le_bytes([h[0x14], h[0x15]]);
                let flags = u16::from_le_bytes([h[0x16], h[0x17]]);
                let asize = i16::from_le_bytes([h[0x18], h[0x19]]).max(0) as usize;
                let neigh: Vec<i16> = d
                    .read(p + 0x1a, 2 * asize)
                    .map(|b| b.chunks_exact(2).map(|x| i16::from_le_bytes([x[0], x[1]])).collect())
                    .unwrap_or_default();
                raw.insert(vreg, (spill, color, i16::from_le_bytes([h[0x12], h[0x13]]), flags, asize as u16, neigh));
            }
            let pos: HashMap<u16, usize> = order.iter().enumerate().map(|(k, &v)| (v, k)).collect();
            // blocked: in colouring order, runs of descending vregs; all but the last run were
            // pushed in later simplify passes.
            let mut run_of: HashMap<u16, usize> = HashMap::new();
            let mut run = 0usize;
            for (k, &v) in order.iter().enumerate() {
                if k > 0 && v > order[k - 1] {
                    run += 1;
                }
                run_of.insert(v, run);
            }
            let last_run = run;
            let resolve = |mut col: i16| -> Option<u8> {
                let mut guard = 0;
                while col >= nreal as i16 && guard < 64 {
                    col = raw.get(&(col as u16)).map(|x| x.1).unwrap_or(-1);
                    guard += 1;
                }
                if (0..32).contains(&col) {
                    Some(col as u8)
                } else {
                    None
                }
            };
            let mut nodes = vec![];
            for (&v, (spill, color, _deg_left, flags, asize, neigh)) in &raw {
                let name = if *spill != 0 { Some(d.obj_name(*spill)) } else { None };
                let kind = if *spill != 0 && params.contains(spill) {
                    VregKind::Param
                } else if (v as u32) < first_fe {
                    VregKind::Named
                } else if (v as u32) < first_tmp {
                    VregKind::FeTemp
                } else {
                    VregKind::Temp
                };
                let coalesced = flags & 4 != 0;
                nodes.push(IgNode {
                    vreg: v,
                    name,
                    kind,
                    degree: *asize,
                    reg: resolve(*color),
                    coalesced_into: if coalesced { Some(*color) } else { None },
                    spilled: flags & 1 != 0,
                    neighbours: neigh.clone(),
                    order: pos.get(&v).copied(),
                    blocked: run_of.get(&v).map_or(false, |&r| r < last_run),
                });
            }
            t.rounds.push(ColorRound {
                function: fname,
                class: if cls == 3 { "FPR".into() } else if cls == 4 { "GPR".into() } else { format!("class{cls}") },
                nreal,
                first_fe_temp: first_fe,
                first_temp: first_tmp,
                nodes,
            });
        }
        _ => {
            if let Some((_, why)) = IFC_RETS.iter().find(|(a, _)| *a == addr) {
                if let Some(p) = pending.take() {
                    t.inline.push(InlineDecision {
                        caller: p.caller,
                        callee: p.callee,
                        pass: p.pass,
                        level: p.level,
                        cost: p.cost,
                        max: p.max,
                        inlined: c.Eax & 0xff != 0,
                        reason: why.to_string(),
                    });
                }
            }
        }
    }
}

/// Walk pcbasicblocks and render the PCode with virtual registers.
fn dump_pcode(d: &Debuggee) -> Vec<String> {
    let mut out = vec![];
    let mut opnames: HashMap<i16, String> = HashMap::new();
    let mut b = d.u32(A_BLOCKS);
    let mut nb = 0;
    while b != 0 && nb < 20_000 {
        out.push(format!("B{}:", d.u32(b + 0x1c)));
        let mut p = d.u32(b + 0x14);
        let mut n = 0;
        while p != 0 && n < 200_000 {
            let Some(h) = d.read(p, 0x24) else { break };
            let op = i16::from_le_bytes([h[0x20], h[0x21]]);
            let argc = i16::from_le_bytes([h[0x22], h[0x23]]);
            if !(0..471).contains(&op) || !(0..200).contains(&argc) {
                out.push(format!("   ?? op={op} argc={argc}"));
                break;
            }
            let name = opnames
                .entry(op)
                .or_insert_with(|| {
                    let np = d.u32(A_OPINFO + 18 * op as u32);
                    d.cstr(np, 16).to_lowercase()
                })
                .clone();
            let raw = d.read(p + 0x24, 12 * argc as usize).unwrap_or_default();
            let mut args = vec![];
            for k in 0..argc as usize {
                let a = &raw[12 * k..12 * k + 12];
                let (kind, cls) = (a[0], a[1]);
                let eff = u16::from_le_bytes([a[2], a[3]]);
                let reg = i16::from_le_bytes([a[4], a[5]]);
                let imm = i32::from_le_bytes([a[2], a[3], a[4], a[5]]);
                args.push(match kind {
                    0 => {
                        let pre = match cls {
                            4 => "r",
                            3 => "f",
                            1 => "cr",
                            0 => "spr",
                            _ => "v",
                        };
                        format!("{pre}{reg}{}", if eff & 3 == 2 { "=" } else { "" })
                    }
                    2 => format!("#{imm}"),
                    _ => format!("<{kind}:{imm}>"),
                });
            }
            if matches!(name.as_str(), "bl" | "bctrl" | "blrl") && args.len() > 4 {
                let w = args.iter().filter(|a| a.ends_with('=')).count();
                args.truncate(1);
                args.push(format!("({} operands, {} written regs)", argc, w));
            }
            out.push(format!("   {:<8} {}", name, args.join(", ")));
            p = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
            n += 1;
        }
        b = d.u32(b);
        nb += 1;
    }
    out
}

impl ColorRound {
    /// Nodes that received a callee-saved register, in colouring order.
    pub fn callee_saved(&self) -> Vec<&IgNode> {
        let lo = if self.class == "FPR" { 14 } else { 14 };
        let mut v: Vec<&IgNode> =
            self.nodes.iter().filter(|n| n.reg.map_or(false, |r| r >= lo) && n.coalesced_into.is_none()).collect();
        v.sort_by_key(|n| n.order.unwrap_or(usize::MAX));
        v
    }
}

// ------------------------------------------------------------------ targeted register fixes

impl Trace {
    /// The colouring round of the first function whose name contains `func` for "GPR"/"FPR".
    pub fn round(&self, func: &str, class: &str) -> Option<&ColorRound> {
        self.rounds.iter().find(|r| r.function.contains(func) && r.class == class)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RegisterFix {
    /// the candidate's value that sits in `from` but should be in `to`
    pub vreg: u16,
    pub name: Option<String>,
    pub kind: VregKind,
    pub from: u8,
    pub to: u8,
    /// the candidate value currently holding `to` (if any)
    pub holder: Option<(u16, Option<String>, VregKind)>,
    pub suggestion: String,
}

fn label(n: &IgNode) -> String {
    match &n.name {
        Some(s) => format!("`{s}` (v{})", n.vreg),
        None => format!("temp v{}", n.vreg),
    }
}

/// For a register-only diff: `moves` = (candidate register, target register) pairs for callee-saved
/// registers that disagree. Returns which candidate value is involved, who holds the wanted
/// register, and the source-level mutation that changes their colouring order (regalloc.md rules).
pub fn register_fixes(round: &ColorRound, moves: &[(u8, u8)]) -> Vec<RegisterFix> {
    let mut out = vec![];
    // the callee-saved value per register: the one coloured LAST into it overlaps the diff region
    // poorly; we take the first in colouring order (values sharing a register do not overlap).
    let by_reg = |r: u8| -> Vec<&IgNode> {
        let mut v: Vec<&IgNode> = round.nodes.iter().filter(|n| n.reg == Some(r) && n.coalesced_into.is_none() && n.degree > 0).collect();
        v.sort_by_key(|n| n.order.unwrap_or(usize::MAX));
        v
    };
    for &(from, to) in moves {
        for n in by_reg(from) {
            let holder = by_reg(to).into_iter().find(|h| h.neighbours.contains(&(n.vreg as i16)));
            let earlier = to > from; // higher register number = coloured earlier (r31 first)
            let s = if n.blocked {
                format!(
                    "{} is K-blocked (degree {} >= K): it is coloured before everything else; its register only changes \
                     if its degree drops below K (fewer overlapping temps) or other blocked values change",
                    label(n),
                    n.degree
                )
            } else if let Some(h) = holder {
                let (hn, hk) = (label(h), h.kind);
                match (n.kind, hk, earlier) {
                    (VregKind::Named, VregKind::Named, true) => format!("declare {} before {}", label(n), hn),
                    (VregKind::Named, VregKind::Named, false) => format!("declare {} after {}", label(n), hn),
                    (VregKind::Temp, VregKind::Temp, true) => format!("create {} after {} (evaluate it later; args run right to left)", label(n), hn),
                    (VregKind::Temp, VregKind::Temp, false) => format!("create {} before {}", label(n), hn),
                    (VregKind::Named, VregKind::Temp, true) => format!(
                        "make {} temp-class (all uses non-move: unname it or remove its plain register uses), or make {} named (give it a plain move-use)",
                        label(n),
                        hn
                    ),
                    (VregKind::Temp, VregKind::Named, false) => format!(
                        "make {} named (add a plain move-use, e.g. pass it as-is), or make {} temp-class",
                        label(n),
                        hn
                    ),
                    (VregKind::Param, _, _) | (_, VregKind::Param, _) => format!(
                        "params rank lowest in fixed order ({} vs {}); change which value is a param copy, \
                         or use `const` on a by-value param passed as a call argument (becomes an FE temp)",
                        label(n),
                        hn
                    ),
                    (VregKind::FeTemp, _, _) | (_, VregKind::FeTemp, _) => format!(
                        "{} vs {}: FE temps (inline-expansion locals, const-param copies) rank above named locals and below codegen temps; \
                         inline/out-of-line the helper or change the local into/out of an inlined body",
                        label(n),
                        hn
                    ),
                    _ => format!("{} and {} would need their class order reversed", label(n), hn),
                }
            } else {
                format!(
                    "{} gets r{} because the lowest free already-obtained register is reused; to reach r{}, the value holding it \
                     must overlap {} (or be coloured later)",
                    label(n),
                    from,
                    to,
                    label(n)
                )
            };
            out.push(RegisterFix {
                vreg: n.vreg,
                name: n.name.clone(),
                kind: n.kind,
                from,
                to,
                holder: holder.map(|h| (h.vreg, h.name.clone(), h.kind)),
                suggestion: s,
            });
            break;
        }
    }
    out
}
