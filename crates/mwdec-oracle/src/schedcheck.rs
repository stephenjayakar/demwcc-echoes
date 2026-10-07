//! Scheduling dependence checker for scheduling-only diffs.
//!
//! Given the target function and a candidate compile of the same function (same instructions, some
//! in a different order), it pairs instructions per basic block and classifies every pair whose
//! relative order differs:
//! - **Forced**: a true register dependence or a memory-order dependence orders the two
//!   instructions in one of the programs. MWCC's scheduler can never swap them, so the *source
//!   statements* producing them must be reordered (or the dependence removed, e.g. by caching a
//!   load in a local). With `-sym on` line info the report names the source lines.
//! - **RegAlloc**: only a register anti/output dependence (an artefact of register choice) orders
//!   them: fix the register assignment first.
//! - **Priority**: independent; the list scheduler chose by latency/critical path. Statement order
//!   matters only as a tie-break (swap statements as a low-confidence try).
//! Basic blocks end at branches, labels and calls (MWCC blocks end at calls; nothing moves across).
//!
//! Memory model (the GC/2.7 alias analysis, verified with compiler experiments, see
//! `tests/schedcheck_alias.rs`):
//! - accesses through a pointer of unknown origin share one "worst case" alias set;
//! - the worst case set contains every global/static object and every stack local whose address
//!   *escapes*: passed to a call, stored to memory, copied/compared as a value. Such a local is
//!   ordered against pointer accesses, even when its address is taken in a later block;
//! - a stack local whose address is only used as a load/store base (indexed array access, struct
//!   copies) stays its own object: independent of pointer accesses;
//! - named objects (globals, stack locals) are disambiguated by object and byte range: different
//!   objects never alias, disjoint fields of one object do not alias.
//! Escapes and object extents are recovered from the code (`addi rX, r1, N` and how rX is used);
//! exact local extents can be supplied from the candidate's DWARF ([`stack_objects`]).

use crate::asm::{self, Func, Obj};
use ppc750cl::{Argument, Ins, Opcode};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Debug)]
struct I {
    off: u32,
    key: String,   // exact text (registers included)
    loose: String, // mnemonic + non-register operands (relocs/immediates)
    defs: u64,
    uses: u64,
    mem: Option<Mem>,
    store: bool,
    call: bool,
    frame: bool,
}

/// Memory class of a load/store (see the module docs for the alias rules).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Mem {
    /// stack object bytes `[off, off+size)` relative to r1; `escaped`: its address escapes, so it
    /// is in the worst-case set and aliases pointer accesses
    Stack { off: i32, size: u32, escaped: bool },
    /// named global/static object (relocation target) and byte range (`size` u32::MAX = unknown)
    Global { sym: String, off: i64, size: u32 },
    /// through a pointer of unknown origin (worst case)
    Pointer,
}

/// Whether two memory accesses may alias under the GC/2.7 rules.
pub fn mem_alias(a: &Mem, b: &Mem) -> bool {
    let overlap = |x: i64, sx: u32, y: i64, sy: u32| x < y + sy as i64 && y < x + sx as i64;
    match (a, b) {
        (Mem::Stack { off: x, size: sx, .. }, Mem::Stack { off: y, size: sy, .. }) => {
            overlap(*x as i64, *sx, *y as i64, *sy)
        }
        (Mem::Global { sym: x, off: ox, size: sx }, Mem::Global { sym: y, off: oy, size: sy }) => {
            x == y && overlap(*ox, *sx, *oy, *sy)
        }
        (Mem::Stack { escaped, .. }, Mem::Pointer) | (Mem::Pointer, Mem::Stack { escaped, .. }) => *escaped,
        (Mem::Stack { .. }, Mem::Global { .. }) | (Mem::Global { .. }, Mem::Stack { .. }) => false,
        _ => true,
    }
}

/// A stack-resident local of a function: r1-relative offset and extent.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StackObject {
    pub name: String,
    pub off: i32,
    pub size: u32,
}

fn access_size(op: Opcode) -> u32 {
    use Opcode::*;
    match op {
        Lbz | Lbzu | Lbzx | Lbzux | Stb | Stbu | Stbx | Stbux => 1,
        Lhz | Lha | Lhzu | Lhau | Lhzx | Lhax | Lhzux | Lhaux | Sth | Sthu | Sthx | Sthux | Lhbrx | Sthbrx => 2,
        Lfd | Lfdu | Lfdx | Lfdux | Stfd | Stfdu | Stfdx | Stfdux => 8,
        PsqL | PsqSt | PsqLu | PsqStu | PsqLx | PsqStx => 8,
        _ => 4,
    }
}

fn bit(r: u8) -> u64 {
    1u64 << r
}

/// What a register is known to hold (address tracking for the alias model).
#[derive(Clone, Debug)]
enum Addr {
    /// r1 + `exact` (when known), derived from the stack address `base` (object start)
    Stack { base: i32, exact: Option<i32> },
    /// address of a global (+ offset when known)
    Global { sym: String, off: Option<i64> },
}

/// Raw per-instruction facts used by the memory classification.
struct Raw {
    ins: Ins,
    mnem: String,
    args: Vec<Argument>,
    load: bool,
    store: bool,
    indexed: bool,
    /// base register (d-form) or the address registers (indexed)
    addr_regs: Vec<u8>,
    /// stored value register (integer stores)
    value_reg: Option<u8>,
    disp: i32,
    reloc: Option<(String, i64)>,
    call: bool,
}

fn gprs(args: &[Argument]) -> Vec<u8> {
    args.iter().filter_map(|a| if let Argument::GPR(g) = a { Some(g.0) } else { None }).collect()
}

fn simm(args: &[Argument]) -> Option<i32> {
    args.iter().find_map(|a| match a {
        Argument::Simm(s) => Some(s.0 as i32),
        Argument::Uimm(u) => Some(u.0 as i32),
        _ => None,
    })
}

/// Register saved/restored by a frame instruction candidate: (is float, register).
fn saved_reg(r: &Raw) -> Option<(bool, u8)> {
    if r.addr_regs != [1] {
        return None;
    }
    match r.mnem.as_str() {
        "stw" | "lwz" | "stmw" | "lmw" => gprs(&r.args).first().copied().filter(|&g| g >= 14).map(|g| (false, g)),
        "stfd" | "lfd" | "psq_st" | "psq_l" => r.args.iter().find_map(|a| match a {
            Argument::FPR(f) if f.0 >= 14 => Some((true, f.0)),
            _ => None,
        }),
        _ => None,
    }
}

/// Frame size (from `stwu r1, -N(r1)`).
fn frame_size(raws: &[Raw]) -> i32 {
    raws.iter()
        .filter(|r| r.mnem == "stwu" && gprs(&r.args).first() == Some(&1) && r.addr_regs == [1])
        .map(|r| -r.disp)
        .max()
        .unwrap_or(0)
}

/// Prologue/epilogue instructions: stack pointer update, LR save/restore, callee-saved register
/// saves (a store of a callee-saved register before the function writes it) and the matching
/// restores (loads of the same register from the same slot).
fn frame_flags(raws: &[Raw]) -> Vec<bool> {
    let size = frame_size(raws);
    let mut written: BTreeSet<(bool, u8)> = BTreeSet::new();
    let mut saves: BTreeSet<(bool, u8, i32)> = BTreeSet::new();
    let mut out = vec![false; raws.len()];
    for (i, r) in raws.iter().enumerate() {
        let regs = gprs(&r.args);
        out[i] = match r.mnem.as_str() {
            "stwu" => regs.first() == Some(&1) && r.addr_regs == [1],
            "addi" => regs.len() == 2 && regs[0] == 1 && regs[1] == 1,
            "mflr" | "mtlr" => regs.first() == Some(&0),
            "stw" | "lwz" if regs.first() == Some(&0) && r.addr_regs == [1] => r.disp == size + 4,
            _ => false,
        };
        if r.store {
            if let Some((fl, g)) = saved_reg(r) {
                if !written.contains(&(fl, g)) {
                    out[i] = true;
                    saves.insert((fl, g, r.disp));
                    if r.mnem == "stmw" {
                        for k in g..32 {
                            saves.insert((false, k, r.disp + 4 * (k - g) as i32));
                        }
                    }
                }
            }
        } else if r.load {
            if let Some((fl, g)) = saved_reg(r) {
                if saves.contains(&(fl, g, r.disp)) {
                    out[i] = true;
                }
            }
        }
        for a in r.ins.defs().iter() {
            match a {
                Argument::GPR(g) => {
                    written.insert((false, g.0));
                }
                Argument::FPR(f) => {
                    written.insert((true, f.0));
                }
                _ => {}
            }
        }
    }
    out
}

/// Lowest offset of the callee-saved register save area (= end of the locals area), else the
/// frame size.
fn locals_end(raws: &[Raw]) -> i32 {
    let size = frame_size(raws);
    let fr = frame_flags(raws);
    raws.iter()
        .zip(fr.iter())
        .filter(|(r, &f)| f && r.store && saved_reg(r).is_some() && r.disp > 0)
        .map(|(r, _)| r.disp)
        .min()
        .unwrap_or(size)
}

fn raw_facts(f: &Func) -> Vec<Raw> {
    let rel_at: BTreeMap<u32, &asm::Rel> = f.relocs.iter().map(|r| (r.offset & !3, r)).collect();
    let mut out = vec![];
    for (i, c) in f.code.chunks_exact(4).enumerate() {
        let off = (i * 4) as u32;
        let w = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        let ins = Ins::new(w);
        let s = ins.simplified();
        let mnem = s.mnemonic.to_string();
        let args: Vec<Argument> = s.args_iter().cloned().collect();
        let load = mnem.starts_with('l') && !matches!(mnem.as_str(), "li" | "lis");
        let store = mnem.starts_with("st") || mnem.starts_with("psq_st");
        let mut indexed = false;
        let mut addr_regs = vec![];
        let mut value_reg = None;
        let mut disp = 0i32;
        if load || store {
            let regs = gprs(&args);
            let has_off = args.iter().any(|a| matches!(a, Argument::Offset(_)));
            disp = args.iter().find_map(|a| if let Argument::Offset(o) = a { Some(o.0 as i32) } else { None }).unwrap_or(0);
            let is_fp = args.iter().any(|a| matches!(a, Argument::FPR(_)));
            if has_off || mnem == "stmw" || mnem == "lmw" {
                // d-form: [rT/fT,] off(rA)
                addr_regs = regs.last().copied().into_iter().collect();
                if store && !is_fp && mnem != "stmw" {
                    value_reg = regs.first().copied();
                }
            } else {
                // x-form: rT/fT, rA, rB
                indexed = true;
                let skip = if is_fp { 0 } else { 1 };
                addr_regs = regs.iter().skip(skip).copied().collect();
                if store && !is_fp {
                    value_reg = regs.first().copied();
                }
            }
        }
        let reloc = rel_at.get(&off).map(|r| (r.target.clone(), r.addend));
        let call = (ins.op == Opcode::B && (w & 1) == 1) || mnem == "bctrl" || mnem == "blrl";
        out.push(Raw { ins, mnem, args, load, store, indexed, addr_regs, value_reg, disp, reloc, call });
    }
    out
}

#[derive(Clone)]
enum Via {
    Direct(i32),
    Stack { base: i32, exact: Option<i32> },
    Global(String, Option<i64>),
    Unknown,
}

/// Number of integer argument registers (r3..) a call to the CodeWarrior-mangled `sym` uses
/// (`this` included), or None when the name cannot be parsed.
pub fn gpr_arg_count(sym: &str) -> Option<u8> {
    let b = sym.as_bytes();
    // find the "__" that starts the signature: the last one followed by a class qualifier or 'F'
    let mut start = None;
    let mut i = 1;
    while i + 2 < b.len() {
        if b[i] == b'_' && b[i + 1] == b'_' && (b[i + 2] == b'F' || b[i + 2] == b'Q' || b[i + 2].is_ascii_digit()) {
            start = Some(i + 2);
            if b[i + 2] == b'F' {
                break;
            }
        }
        i += 1;
    }
    let mut p = start?;
    let mut n: u8 = 0;
    if b[p] != b'F' {
        // member function: class qualifier, `this` in r3
        p = skip_name(b, p)?;
        n += 1;
        while p < b.len() && (b[p] == b'C' || b[p] == b'V') {
            p += 1;
        }
        if p >= b.len() || b[p] != b'F' {
            return Some(n); // e.g. static member data or unusual form
        }
    }
    p += 1;
    while p < b.len() {
        if b[p] == b'_' || b[p] == b'e' {
            break;
        }
        let (fl, gp, np) = mangled_type(b, p)?;
        if !fl {
            n += gp;
        }
        p = np;
    }
    Some(n)
}

fn skip_name(b: &[u8], mut p: usize) -> Option<usize> {
    if b.get(p) == Some(&b'Q') {
        let k = (*b.get(p + 1)? as char).to_digit(10)? as usize;
        p += 2;
        for _ in 0..k {
            p = skip_name(b, p)?;
        }
        return Some(p);
    }
    let mut len = 0usize;
    let s = p;
    while p < b.len() && b[p].is_ascii_digit() {
        len = len * 10 + (b[p] - b'0') as usize;
        p += 1;
    }
    if p == s {
        return None;
    }
    Some(p + len)
}

/// (is float, GPRs used as a parameter, next position)
fn mangled_type(b: &[u8], mut p: usize) -> Option<(bool, u8, usize)> {
    while p < b.len() && matches!(b[p], b'C' | b'V' | b'U' | b'S') {
        p += 1;
    }
    let c = *b.get(p)?;
    Some(match c {
        b'P' | b'R' => {
            let (_, _, np) = mangled_type(b, p + 1)?;
            (false, 1, np)
        }
        b'A' => {
            p += 1;
            while p < b.len() && b[p].is_ascii_digit() {
                p += 1;
            }
            p += 1; // '_'
            let (_, _, np) = mangled_type(b, p)?;
            (false, 1, np)
        }
        b'F' => {
            p += 1;
            while p < b.len() && b[p] != b'_' {
                let (_, _, np) = mangled_type(b, p)?;
                p = np;
            }
            let (_, _, np) = mangled_type(b, p + 1)?;
            (false, 1, np)
        }
        b'M' => {
            let np = skip_name(b, p + 1)?;
            let (_, _, np) = mangled_type(b, np)?;
            (false, 1, np)
        }
        b'Q' | b'0'..=b'9' => (false, 1, skip_name(b, p)?),
        b'f' | b'd' => (true, 0, p + 1),
        b'x' => (false, 2, p + 1),
        b'v' => (false, 0, p + 1),
        _ => (false, 1, p + 1),
    })
}

/// Memory classes per instruction (None for non-memory instructions).
fn classify_memory(f: &Func, stack: Option<&[StackObject]>) -> Vec<Option<Mem>> {
    let raws = raw_facts(f);
    let region_end = locals_end(&raws);
    // all stack address bases (object starts) seen in the code
    let mut bases: BTreeSet<i32> = BTreeSet::new();
    for r in &raws {
        if r.mnem == "addi" && r.reloc.is_none() {
            let regs = gprs(&r.args);
            if regs.len() == 2 && regs[1] == 1 && regs[0] != 1 {
                if let Some(imm) = simm(&r.args) {
                    bases.insert(imm);
                }
            }
        }
    }
    let object_of = |at: i32| -> (i32, u32) {
        if let Some(objs) = stack {
            if let Some(o) = objs.iter().find(|o| at >= o.off && at < o.off + o.size as i32) {
                return (o.off, o.size);
            }
        }
        let next_obj = stack.and_then(|objs| objs.iter().map(|o| o.off).filter(|&o| o > at).min());
        let next_base = bases.range(at + 1..).next().copied();
        let end = [next_base, next_obj].into_iter().flatten().fold(region_end, i32::min).max(at + 4);
        (at, (end - at).max(1) as u32)
    };
    // pass 1: address tracking, escapes, per-access address origin
    let mut via: Vec<Option<Via>> = vec![None; raws.len()];
    let mut escaped: BTreeSet<i32> = BTreeSet::new();
    let mut st: HashMap<u8, Addr> = HashMap::new();
    // basic block of each tracked register's definition: argument registers set up for a call are
    // always defined in the call's own block (calls end MWCC blocks)
    let targets: BTreeSet<usize> = raws
        .iter()
        .enumerate()
        .filter(|(_, r)| r.ins.is_branch() && !r.call)
        .filter_map(|(i, r)| r.ins.branch_dest((i * 4) as u32).map(|d| d as usize / 4))
        .collect();
    let mut def_blk: HashMap<u8, usize> = HashMap::new();
    // registers whose stack address was already used as a load/store address: a register used that
    // way is a local address temp, not an argument prepared for a following call
    let mut addr_used: BTreeSet<u8> = BTreeSet::new();
    let mut blk = 0usize;
    for (i, r) in raws.iter().enumerate() {
        if targets.contains(&i) {
            blk += 1;
        }
        let regs = gprs(&r.args);
        let defs: Vec<u8> =
            r.ins.defs().iter().filter_map(|a| if let Argument::GPR(g) = a { Some(g.0) } else { None }).collect();
        let uses: Vec<u8> =
            r.ins.uses().iter().filter_map(|a| if let Argument::GPR(g) = a { Some(g.0) } else { None }).collect();
        let mut new: Option<(u8, Addr)> = None;
        // registers used as addresses here (not escapes)
        let mut addr_uses: Vec<u8> = vec![];
        if r.load || r.store {
            if let Some((sym, add)) = &r.reloc {
                via[i] = Some(Via::Global(sym.clone(), Some(*add)));
            } else if r.addr_regs == [1] {
                via[i] = Some(Via::Direct(r.disp));
            } else {
                let tracked: Vec<&Addr> = r.addr_regs.iter().filter_map(|g| st.get(g)).collect();
                via[i] = Some(match tracked.first() {
                    Some(Addr::Stack { base, exact }) => Via::Stack {
                        base: *base,
                        exact: if r.indexed || tracked.len() > 1 { None } else { exact.map(|e| e + r.disp) },
                    },
                    Some(Addr::Global { sym, off }) => {
                        Via::Global(sym.clone(), if r.indexed { None } else { off.map(|o| o + r.disp as i64) })
                    }
                    None => Via::Unknown,
                });
            }
            addr_uses.extend(r.addr_regs.iter().copied());
        } else if (r.mnem == "addi" || r.mnem == "subi") && regs.len() == 2 {
            if let Some((sym, add)) = &r.reloc {
                new = Some((regs[0], Addr::Global { sym: sym.clone(), off: Some(*add) }));
                addr_uses.push(regs[1]);
            } else if regs[1] == 1 && regs[0] != 1 {
                let imm = simm(&r.args).unwrap_or(0);
                new = Some((regs[0], Addr::Stack { base: imm, exact: Some(imm) }));
            } else if let Some(a) = st.get(&regs[1]) {
                let imm = simm(&r.args).unwrap_or(0);
                let imm = if r.mnem == "subi" { -imm } else { imm };
                new = Some((
                    regs[0],
                    match a {
                        Addr::Stack { base, exact } => Addr::Stack { base: *base, exact: exact.map(|e| e + imm) },
                        Addr::Global { sym, off } => Addr::Global { sym: sym.clone(), off: off.map(|o| o + imm as i64) },
                    },
                ));
                addr_uses.push(regs[1]);
            }
        } else if r.mnem == "add" && regs.len() == 3 {
            if let Some(a) = st.get(&regs[1]).or_else(|| st.get(&regs[2])).cloned() {
                new = Some((
                    regs[0],
                    match a {
                        Addr::Stack { base, .. } => Addr::Stack { base, exact: None },
                        Addr::Global { sym, .. } => Addr::Global { sym, off: None },
                    },
                ));
                addr_uses.extend([regs[1], regs[2]]);
            }
        }
        // escapes: a stack address used as a value (stored, copied, compared, passed to a call)
        let mut esc = |a: Option<&Addr>| {
            if let Some(Addr::Stack { base, .. }) = a {
                escaped.insert(*base);
            }
        };
        if let Some(v) = r.value_reg {
            esc(st.get(&v));
        }
        if r.call {
            // argument registers from the callee's mangled signature when known
            let nargs = r.reloc.as_ref().and_then(|(sym, _)| gpr_arg_count(sym));
            for g in 3..=10u8 {
                let is_arg = match nargs {
                    Some(n) => (g - 3) < n,
                    None => def_blk.get(&g) == Some(&blk) && !addr_used.contains(&g),
                };
                if is_arg {
                    esc(st.get(&g));
                }
            }
        } else {
            for g in &uses {
                if *g == 1 || (addr_uses.contains(g) && Some(*g) != r.value_reg) {
                    continue;
                }
                esc(st.get(g));
            }
        }
        // state update
        if r.load || r.store {
            addr_used.extend(r.addr_regs.iter().copied());
        }
        for g in &defs {
            st.remove(g);
            addr_used.remove(g);
        }
        if r.call {
            for g in [0u8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
                st.remove(&g);
            }
        }
        if let Some((g, a)) = new {
            st.insert(g, a);
            def_blk.insert(g, blk);
        }
        // update forms (stwu/lwzu/...): the base register now holds the accessed address
        let update = (r.load || r.store) && !r.indexed && r.mnem != "stmw" && r.mnem != "lmw" && r.mnem.ends_with('u');
        if update {
            if let (Some(&b), Some(v)) = (r.addr_regs.first(), via[i].as_ref()) {
                if b != 1 {
                    let a = match v {
                        Via::Global(sym, off) => Some(Addr::Global { sym: sym.clone(), off: *off }),
                        Via::Stack { base, exact } => Some(Addr::Stack { base: *base, exact: *exact }),
                        _ => None,
                    };
                    if let Some(a) = a {
                        st.insert(b, a);
                        def_blk.insert(b, blk);
                    }
                }
            }
        }
        if matches!(r.mnem.as_str(), "b" | "blr" | "bctr") {
            st.clear();
        }
        if r.call || r.ins.is_branch() {
            blk += 1;
        }
    }
    let is_escaped = |off: i32, size: u32| {
        escaped.iter().any(|&b| {
            let (o, s) = object_of(b);
            off < o + s as i32 && o < off + size as i32
        })
    };
    raws.iter()
        .enumerate()
        .map(|(i, r)| {
            let size = access_size(r.ins.op);
            Some(match via[i].clone()? {
                Via::Direct(off) => Mem::Stack { off, size, escaped: is_escaped(off, size) },
                Via::Stack { exact: Some(e), .. } => Mem::Stack { off: e, size, escaped: is_escaped(e, size) },
                Via::Stack { base, exact: None } => {
                    let (o, s) = object_of(base);
                    Mem::Stack { off: o, size: s, escaped: is_escaped(o, s) }
                }
                Via::Global(sym, Some(o)) => Mem::Global { sym, off: o, size },
                Via::Global(sym, None) => Mem::Global { sym, off: i32::MIN as i64, size: u32::MAX },
                Via::Unknown => Mem::Pointer,
            })
        })
        .collect()
}

/// Stack-resident locals of `func` from the object's DWARF 1 `.debug` section (compile with
/// `-sym on`). Extents run to the next local's offset (or the saved-register area).
pub fn stack_objects(obj: &Obj, func: &str) -> Vec<StackObject> {
    let Some(d) = obj.sections.get(".debug") else { return vec![] };
    let rd32 = |p: usize| d.get(p..p + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    let rd16 = |p: usize| d.get(p..p + 2).map(|b| u16::from_be_bytes([b[0], b[1]]));
    let mut found: Vec<(String, i32)> = vec![];
    let mut in_func = false;
    let mut o = 0usize;
    while let Some(len) = rd32(o) {
        let len = len as usize;
        if len < 8 {
            o += len.max(4);
            continue;
        }
        if o + len > d.len() {
            break;
        }
        let tag = rd16(o + 4).unwrap_or(0);
        let mut p = o + 6;
        let (mut name, mut mangled, mut loc): (Option<String>, Option<String>, Option<Vec<u8>>) = (None, None, None);
        while p + 2 <= o + len {
            let at = rd16(p).unwrap_or(0);
            p += 2;
            let val = p;
            match at & 0xf {
                1 | 2 | 6 => p += 4,
                5 => p += 2,
                7 => p += 8,
                3 => {
                    let n = rd16(p).unwrap_or(0) as usize;
                    if at == 0x0023 {
                        loc = d.get(p + 2..p + 2 + n).map(|b| b.to_vec());
                    }
                    p += 2 + n;
                }
                4 => p += 4 + rd32(p).unwrap_or(0) as usize,
                8 => {
                    let e = d[p..o + len].iter().position(|&c| c == 0).map_or(o + len, |x| p + x);
                    let s = String::from_utf8_lossy(&d[p..e]).into_owned();
                    match at {
                        0x0038 => name = Some(s),
                        0x2008 => mangled = Some(s),
                        _ => {}
                    }
                    p = e + 1;
                }
                _ => break,
            }
            let _ = val;
        }
        match tag {
            0x06 | 0x14 => in_func = mangled.or(name.clone()).as_deref() == Some(func),
            0x0c if in_func => {
                // location: OP_BASEREG 1, OP_CONST off, OP_ADD
                if let Some(b) = loc {
                    if b.len() == 11 && b[0] == 0x02 && b[1..5] == [0, 0, 0, 1] && b[5] == 0x04 && b[10] == 0x07 {
                        let off = i32::from_be_bytes([b[6], b[7], b[8], b[9]]);
                        found.push((name.unwrap_or_default(), off));
                    }
                }
            }
            _ => {}
        }
        o += len;
    }
    found.sort_by_key(|x| x.1);
    found.dedup_by_key(|x| x.1);
    let end = obj.funcs.iter().find(|f| f.name == func).map(|f| locals_end(&raw_facts(f))).unwrap_or(i32::MAX);
    (0..found.len())
        .map(|k| {
            let next = found.get(k + 1).map_or(end, |x| x.1).max(found[k].1 + 1);
            StackObject { name: found[k].0.clone(), off: found[k].1, size: (next - found[k].1) as u32 }
        })
        .collect()
}

fn decode(obj: &Obj, f: &Func, stack: Option<&[StackObject]>) -> Vec<I> {
    let lines = asm::disasm_func(obj, f, asm::AsmOpts { offsets: true, literals: false });
    let texts: BTreeMap<u32, String> = lines
        .iter()
        .filter_map(|l| {
            let (o, t) = l.trim().split_once(": ")?;
            Some((u32::from_str_radix(o.trim(), 16).ok()?, t.trim().to_string()))
        })
        .collect();
    let rel_at: BTreeMap<u32, &asm::Rel> = f.relocs.iter().map(|r| (r.offset & !3, r)).collect();
    let lit_re = regex::Regex::new(r"@\d+|lbl_[0-9A-Fa-f]+|\.L_[0-9a-f]+").unwrap();
    let mems = classify_memory(f, stack);
    let frames = frame_flags(&raw_facts(f));
    let mut out = vec![];
    for (i, c) in f.code.chunks_exact(4).enumerate() {
        let off = (i * 4) as u32;
        let w = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        let ins = Ins::new(w);
        let s = ins.simplified();
        let key = texts.get(&off).cloned().unwrap_or_else(|| s.to_string());
        let key = lit_re.replace_all(&key, "@L").into_owned();
        let loose = {
            let mut parts = vec![s.mnemonic.to_string()];
            for a in s.args_iter() {
                match a {
                    Argument::GPR(_) | Argument::FPR(_) | Argument::CRField(_) => {}
                    _ => parts.push(a.to_string()),
                }
            }
            if let Some(r) = rel_at.get(&off) {
                parts.push(format!("{}+{}", lit_re.replace_all(&r.target, "@L"), r.addend));
            }
            parts.join(" ")
        };
        let mut defs = 0u64;
        let mut uses = 0u64;
        for a in ins.defs().iter() {
            match a {
                Argument::GPR(g) => defs |= bit(g.0),
                Argument::FPR(f) => defs |= bit(32 + f.0),
                _ => {}
            }
        }
        for a in ins.uses().iter() {
            match a {
                Argument::GPR(g) => uses |= bit(g.0),
                Argument::FPR(f) => uses |= bit(32 + f.0),
                _ => {}
            }
        }
        let call = ins.op == Opcode::B && (w & 1) == 1;
        let store = s.mnemonic.starts_with("st") || s.mnemonic.starts_with("psq_st");
        let mem = mems[i].clone();
        let frame = frames[i];
        out.push(I { off, key, loose, defs, uses, store: store && mem.is_some(), mem, call, frame });
    }
    out
}

/// Memory class of every load/store of `f` as (offset within the function, class): what the
/// alias model believes about each access. `stack`: exact locals ([`stack_objects`]) if known.
pub fn memory_classes(f: &Func, stack: Option<&[StackObject]>) -> Vec<(u32, Mem)> {
    classify_memory(f, stack).into_iter().enumerate().filter_map(|(i, m)| Some(((i * 4) as u32, m?))).collect()
}

/// Function offsets of the prologue/epilogue instructions (excluded from the comparison).
pub fn frame_offsets(f: &Func) -> Vec<u32> {
    frame_flags(&raw_facts(f)).iter().enumerate().filter(|(_, &x)| x).map(|(i, _)| (i * 4) as u32).collect()
}

/// Whether the memory accesses at function offsets `a` and `b` are ordered by the alias model
/// (both access memory, at least one is a store, and they may alias). Frame instructions included.
pub fn memory_dependent(f: &Func, a: u32, b: u32, stack: Option<&[StackObject]>) -> bool {
    let cls = classify_memory(f, stack);
    let raws = raw_facts(f);
    let (ia, ib) = ((a / 4) as usize, (b / 4) as usize);
    match (cls.get(ia).cloned().flatten(), cls.get(ib).cloned().flatten()) {
        (Some(ma), Some(mb)) => (raws[ia].store || raws[ib].store) && mem_alias(&ma, &mb),
        _ => false,
    }
}

/// Direct dependence between the instructions at function offsets `a` and `b` (either order):
/// `Some(Data | Memory)`: MWCC's scheduler can never swap them; `Some(RegAlloc)`: only register
/// reuse orders them; `None`: independent.
pub fn dependence(obj: &Obj, f: &Func, a: u32, b: u32, stack: Option<&[StackObject]>) -> Option<DepKind> {
    let ins = decode(obj, f, stack);
    let (x, y) = if a <= b { (a, b) } else { (b, a) };
    let ia = ins.iter().position(|i| i.off == x)?;
    let ib = ins.iter().position(|i| i.off == y)?;
    dep(&ins[ia], &ins[ib])
}

/// Block boundaries: after branches/calls, before branch targets.
fn blocks(ins: &[I], f: &Func) -> Vec<(usize, usize)> {
    let n = ins.len();
    let mut leaders = std::collections::BTreeSet::new();
    leaders.insert(0usize);
    for (i, c) in f.code.chunks_exact(4).enumerate() {
        let w = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        let x = Ins::new(w);
        if x.is_branch() || ins[i].call {
            if i + 1 < n {
                leaders.insert(i + 1);
            }
            if !ins[i].call {
                if let Some(d) = x.branch_dest((i * 4) as u32) {
                    if (d as usize / 4) < n {
                        leaders.insert(d as usize / 4);
                    }
                }
            }
        }
    }
    let l: Vec<usize> = leaders.into_iter().collect();
    l.iter().enumerate().map(|(k, &s)| (s, l.get(k + 1).copied().unwrap_or(n))).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum DepKind {
    /// true register dependence (value flows from first to second)
    Data,
    /// memory order (store/load through possibly aliasing addresses)
    Memory,
    /// register anti/output dependence only (register-allocation artefact)
    RegAlloc,
}

/// Direct dependence of b on a (a earlier in program order), strongest kind.
fn dep(a: &I, b: &I) -> Option<DepKind> {
    if a.defs & b.uses != 0 {
        return Some(DepKind::Data);
    }
    if let (Some(ma), Some(mb)) = (&a.mem, &b.mem) {
        if (a.store || b.store) && mem_alias(ma, mb) {
            return Some(DepKind::Memory);
        }
    }
    if a.call || b.call {
        return Some(DepKind::Memory);
    }
    if a.uses & b.defs != 0 || a.defs & b.defs != 0 {
        return Some(DepKind::RegAlloc);
    }
    None
}

/// Transitive closure over the given instruction indices (program order): for each pair (i, j),
/// i before j, the best dependence kind on any path (Data/Memory if a path of real edges exists,
/// else RegAlloc), keyed by (i, j).
fn closure(ins: &[I], idx: &[usize]) -> std::collections::HashMap<(usize, usize), DepKind> {
    let n = idx.len();
    // real[a][b], any[a][b] over positions
    let mut real = vec![vec![false; n]; n];
    let mut any = vec![vec![false; n]; n];
    let mut first_kind: Vec<Vec<Option<DepKind>>> = vec![vec![None; n]; n];
    let mut has_mem = vec![vec![false; n]; n];
    for b in 0..n {
        for a in (0..b).rev() {
            if let Some(k) = dep(&ins[idx[a]], &ins[idx[b]]) {
                any[a][b] = true;
                if k != DepKind::RegAlloc {
                    real[a][b] = true;
                    first_kind[a][b] = Some(k);
                    has_mem[a][b] = k == DepKind::Memory;
                }
            }
        }
    }
    // propagate: process b ascending; reach(a, b) |= reach(a, m) && edge(m, b)
    let edge_real = real.clone();
    let edge_mem = has_mem.clone();
    let edge_any = any.clone();
    for b in 0..n {
        for m in 0..b {
            if !edge_any[m][b] {
                continue;
            }
            for a in 0..m {
                if any[a][m] {
                    any[a][b] = true;
                }
                if real[a][m] && edge_real[m][b] {
                    real[a][b] = true;
                    if has_mem[a][m] || edge_mem[m][b] {
                        has_mem[a][b] = true;
                    }
                    if first_kind[a][b].is_none() {
                        first_kind[a][b] = first_kind[a][m];
                    }
                }
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    for a in 0..n {
        for b in a + 1..n {
            if real[a][b] {
                out.insert((idx[a], idx[b]), if has_mem[a][b] { DepKind::Memory } else { DepKind::Data });
            } else if any[a][b] {
                out.insert((idx[a], idx[b]), DepKind::RegAlloc);
            }
        }
    }
    out
}

#[derive(Clone, Debug, Serialize)]
pub struct InstrRef {
    pub text: String,
    pub cand_off: u32,
    pub target_off: u32,
    /// candidate source line (when the candidate was compiled with `-sym on`)
    pub line: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Inversion {
    /// earlier in the candidate, later in the target
    pub cand_first: InstrRef,
    pub cand_second: InstrRef,
    /// Some(kind) = ordered by a dependence (in the candidate or in the target); None = priority only
    pub forced: Option<DepKind>,
    /// which program's dependence forces it ("candidate" / "target")
    pub forced_in: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BlockReport {
    pub index: usize,
    pub cand_range: (u32, u32),
    pub target_range: (u32, u32),
    pub inversions: Vec<Inversion>,
    /// instructions present in only one of the two blocks (not a scheduling-only difference)
    pub unmatched_target: Vec<String>,
    pub unmatched_cand: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SchedReport {
    pub blocks: Vec<BlockReport>,
    /// block structure differs (different count): not a scheduling-only diff
    pub structure_differs: bool,
    /// summary: (line that must move earlier, line it must precede, dependence kind)
    pub statement_moves: Vec<(u32, u32, DepKind)>,
}

/// Source line per candidate instruction, from an UNSCHEDULED compile of the same candidate
/// (`#pragma scheduling off` keeps statement order, so its `-sym on` line table is exact).
/// `line_offset` is subtracted (e.g. 1 when a pragma line was prepended).
pub fn attribute_lines(cand_obj: &Obj, cand: &Func, uns_obj: &Obj, uns: &Func, line_offset: u32) -> Vec<Option<u32>> {
    let ci = decode(cand_obj, cand, None);
    let ui = decode(uns_obj, uns, None);
    let cb = blocks(&ci, cand);
    let ub = blocks(&ui, uns);
    let mut out = vec![None; ci.len()];
    for (&(cs, ce), &(us, ue)) in cb.iter().zip(ub.iter()) {
        let mut used = vec![false; ui.len()];
        for pass in 0..2 {
            for c in cs..ce {
                if out[c].is_some() {
                    continue;
                }
                let m = (us..ue).find(|&u| {
                    !used[u] && if pass == 0 { ui[u].key == ci[c].key } else { ui[u].loose == ci[c].loose }
                });
                if let Some(u) = m {
                    used[u] = true;
                    out[c] = uns_obj.line_at(uns.address + ui[u].off).map(|l| l.saturating_sub(line_offset));
                }
            }
        }
    }
    out
}

/// Compare `cand` against `target`. `lines`: optional source line per candidate instruction
/// (see [`attribute_lines`]); otherwise `cand_obj`'s own line table is used if present.
pub fn check(target_obj: &Obj, target: &Func, cand_obj: &Obj, cand: &Func, lines: Option<&[Option<u32>]>) -> SchedReport {
    check_with(target_obj, target, cand_obj, cand, &CheckOptions { lines, ..Default::default() })
}

/// Options for [`check_with`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CheckOptions<'a> {
    /// source line per candidate instruction ([`attribute_lines`])
    pub lines: Option<&'a [Option<u32>]>,
    /// exact stack locals of the candidate ([`stack_objects`] on a `-sym on` compile)
    pub cand_stack: Option<&'a [StackObject]>,
    /// exact stack locals of the target; for a scheduling-only diff the candidate's layout is the
    /// target's, so passing the candidate's objects here is usually right
    pub target_stack: Option<&'a [StackObject]>,
}

/// [`check`] with explicit options (stack object extents for the alias model).
pub fn check_with(target_obj: &Obj, target: &Func, cand_obj: &Obj, cand: &Func, opts: &CheckOptions) -> SchedReport {
    let lines = opts.lines;
    let ti = decode(target_obj, target, opts.target_stack);
    let ci = decode(cand_obj, cand, opts.cand_stack);
    let tb = blocks(&ti, target);
    let cb = blocks(&ci, cand);
    let structure_differs = tb.len() != cb.len();
    let mut report = SchedReport { blocks: vec![], structure_differs, statement_moves: vec![] };
    for (k, (&(ts, te), &(cs, ce))) in tb.iter().zip(cb.iter()).enumerate() {
        // body instructions (prologue/epilogue excluded), indices within the function
        let tbody: Vec<usize> = (ts..te).filter(|&i| !ti[i].frame).collect();
        let cbody: Vec<usize> = (cs..ce).filter(|&i| !ci[i].frame).collect();
        // pair instructions: exact text first, then loose (register-insensitive), in order
        let mut t_of_c: BTreeMap<usize, usize> = BTreeMap::new();
        let mut used_t = vec![false; ti.len()];
        for pass in 0..2 {
            for &c in &cbody {
                if t_of_c.contains_key(&c) {
                    continue;
                }
                let found = tbody.iter().copied().find(|&t| {
                    !used_t[t] && if pass == 0 { ti[t].key == ci[c].key } else { ti[t].loose == ci[c].loose }
                });
                if let Some(t) = found {
                    used_t[t] = true;
                    t_of_c.insert(c, t);
                }
            }
        }
        let unmatched_target: Vec<String> = tbody.iter().filter(|&&t| !used_t[t]).map(|&t| ti[t].key.clone()).collect();
        let unmatched_cand: Vec<String> =
            cbody.iter().filter(|c| !t_of_c.contains_key(c)).map(|&c| ci[c].key.clone()).collect();
        let cclo = closure(&ci, &cbody);
        let tclo = closure(&ti, &tbody);
        let mut inversions = vec![];
        let pairs: Vec<(usize, usize)> = t_of_c.iter().map(|(&c, &t)| (c, t)).collect();
        for x in 0..pairs.len() {
            for y in x + 1..pairs.len() {
                let (ca, ta) = pairs[x];
                let (cb_, tb_) = pairs[y];
                // candidate order ca < cb_ (pairs sorted by candidate index); inversion if target reversed
                if ta < tb_ {
                    continue;
                }
                let fc = cclo.get(&(ca, cb_)).copied();
                let ft = tclo.get(&(tb_, ta)).copied();
                let pick = |k: Option<DepKind>| k.filter(|&d| d != DepKind::RegAlloc);
                let (forced, forced_in) = match (pick(fc), pick(ft)) {
                    (Some(k), _) => (Some(k), Some("candidate".to_string())),
                    (None, Some(k)) => (Some(k), Some("target".to_string())),
                    _ => match (fc, ft) {
                        (Some(DepKind::RegAlloc), _) => (Some(DepKind::RegAlloc), Some("candidate".into())),
                        (_, Some(DepKind::RegAlloc)) => (Some(DepKind::RegAlloc), Some("target".into())),
                        _ => (None, None),
                    },
                };
                let r = |c: usize, t: usize| InstrRef {
                    text: ci[c].key.clone(),
                    cand_off: ci[c].off,
                    target_off: ti[t].off,
                    line: match lines {
                        Some(l) => l.get(c).copied().flatten(),
                        None => cand_obj.line_at(cand.address + ci[c].off),
                    },
                };
                inversions.push(Inversion { cand_first: r(ca, ta), cand_second: r(cb_, tb_), forced, forced_in });
            }
        }
        // statement moves: forced data/memory inversions between different source lines
        for inv in &inversions {
            if let (Some(k @ (DepKind::Data | DepKind::Memory)), Some(l1), Some(l2)) =
                (inv.forced, inv.cand_first.line, inv.cand_second.line)
            {
                if l1 != l2 && !report.statement_moves.iter().any(|m| m.0 == l2 && m.1 == l1) {
                    report.statement_moves.push((l2, l1, k));
                }
            }
        }
        report.blocks.push(BlockReport {
            index: k,
            cand_range: (ci[cs].off, ci[ce - 1].off),
            target_range: (ti[ts].off, ti[te - 1].off),
            inversions,
            unmatched_target,
            unmatched_cand,
        });
    }
    report
}

