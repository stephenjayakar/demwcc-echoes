//! Recover register "webs" (def-use connected values) for callee-saved registers from machine code,
//! with their live ranges, interference and call crossings. This is what MWCC's colouring decided
//! on; [`crate::regalloc`] models how it decided.
//!
//! Register numbering: 0..31 = r0..r31, 32..63 = f0..f31.

use ppc750cl::{Argument, Ins, Opcode};
use std::collections::{BTreeMap, BTreeSet};

pub type RegId = u8;

pub fn reg_name(r: RegId) -> String {
    if r < 32 {
        format!("r{r}")
    } else {
        format!("f{}", r - 32)
    }
}

pub fn is_callee_saved(r: RegId) -> bool {
    (14..=31).contains(&r) || (46..=63).contains(&r)
}

#[derive(Clone, Debug)]
pub struct Instr {
    pub off: u32,
    pub ins: Ins,
    pub text: String,
    pub defs: u64,
    pub uses: u64,
    pub is_call: bool,
    /// call target symbol (from relocation), if known
    pub call_target: Option<String>,
    /// mr/fmr rD, rS
    pub mov: Option<(RegId, RegId)>,
    /// li rD, imm
    pub li: Option<(RegId, i32)>,
    /// `mr. rD, rS` (record form: not a move for MWCC's allocator, but a copy for classification)
    pub mov_dot: Option<(RegId, RegId)>,
}

fn bit(r: RegId) -> u64 {
    1u64 << r
}

fn arg_reg(a: &Argument) -> Option<RegId> {
    match a {
        Argument::GPR(g) => Some(g.0),
        Argument::FPR(f) => Some(32 + f.0),
        _ => None,
    }
}

/// Decode `code` into instructions with def/use masks. `call_targets` maps instruction offset ->
/// relocation target for `bl`.
pub fn decode(code: &[u8], call_targets: &BTreeMap<u32, String>) -> Vec<Instr> {
    let mut out = Vec::new();
    for (i, c) in code.chunks_exact(4).enumerate() {
        let off = (i * 4) as u32;
        let w = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        let ins = Ins::new(w);
        let mut defs = 0u64;
        let mut uses = 0u64;
        for a in ins.defs().iter() {
            if let Some(r) = arg_reg(a) {
                defs |= bit(r);
            }
        }
        for a in ins.uses().iter() {
            if let Some(r) = arg_reg(a) {
                uses |= bit(r);
            }
        }
        let s = ins.simplified();
        let text = s.to_string();
        match ins.op {
            Opcode::Stmw => {
                if let Some(Argument::GPR(g)) = s.args.first() {
                    for r in g.0..32 {
                        uses |= bit(r);
                    }
                }
            }
            Opcode::Lmw => {
                if let Some(Argument::GPR(g)) = s.args.first() {
                    for r in g.0..32 {
                        defs |= bit(r);
                    }
                }
            }
            _ => {}
        }
        let is_call = ins.op == Opcode::B && (w & 1) == 1;
        if is_call {
            // clobbers volatile regs
            for r in [0u8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
                defs |= bit(r);
            }
            for r in 32u8..46 {
                defs |= bit(r);
            }
        }
        let mov = match (s.mnemonic, s.args.first().and_then(arg_reg), s.args.get(1).and_then(arg_reg)) {
            ("mr", Some(d), Some(src)) | ("fmr", Some(d), Some(src)) if s.args.get(2).map_or(true, |a| matches!(a, Argument::None)) => {
                Some((d, src))
            }
            _ => None,
        };
        let mov_dot = match (s.mnemonic, s.args.first().and_then(arg_reg), s.args.get(1).and_then(arg_reg)) {
            ("mr.", Some(d), Some(src)) => Some((d, src)),
            _ => None,
        };
        let li = match (s.mnemonic, s.args.first(), s.args.get(1)) {
            ("li", Some(Argument::GPR(g)), Some(Argument::Simm(v))) => Some((g.0, v.0 as i32)),
            _ => None,
        };
        out.push(Instr {
            off,
            ins,
            text,
            defs,
            uses,
            is_call,
            call_target: call_targets.get(&off).cloned(),
            mov,
            li,
            mov_dot,
        });
    }
    out
}

#[derive(Clone, Debug)]
pub struct Block {
    pub start: usize,
    pub end: usize, // exclusive
    pub succs: Vec<usize>,
}

pub fn blocks(ins: &[Instr]) -> Vec<Block> {
    let n = ins.len();
    let mut leaders = BTreeSet::new();
    leaders.insert(0usize);
    let mut has_bctr = false;
    for (i, x) in ins.iter().enumerate() {
        if x.is_call {
            continue;
        }
        if x.ins.is_branch() {
            if i + 1 < n {
                leaders.insert(i + 1);
            }
            if let Some(d) = x.ins.branch_dest(x.off) {
                if (d as usize) / 4 < n {
                    leaders.insert(d as usize / 4);
                }
            }
            if x.ins.op == Opcode::Bcctr {
                has_bctr = true;
            }
        }
    }
    let ls: Vec<usize> = leaders.into_iter().collect();
    let idx_of = |i: usize| ls.binary_search(&i).unwrap();
    let mut bs = Vec::new();
    for (k, &s) in ls.iter().enumerate() {
        let e = ls.get(k + 1).copied().unwrap_or(n);
        let last = &ins[e - 1];
        let mut succs = Vec::new();
        if last.ins.is_branch() && !last.is_call {
            if let Some(d) = last.ins.branch_dest(last.off) {
                if (d as usize) / 4 < n {
                    succs.push(idx_of(d as usize / 4));
                }
            }
            let uncond = last.ins.is_unconditional_branch();
            if !uncond && e < n {
                succs.push(idx_of(e));
            }
            if last.ins.op == Opcode::Bcctr && has_bctr {
                // jump table: conservatively every block may follow
                succs.extend(0..ls.len());
            }
        } else if e < n {
            succs.push(idx_of(e));
        }
        succs.sort();
        succs.dedup();
        bs.push(Block { start: s, end: e, succs });
    }
    bs
}

#[derive(Clone, Debug)]
pub struct Web {
    pub id: usize,
    pub reg: RegId,
    /// instruction indices defining this web
    pub defs: Vec<usize>,
    /// instruction indices using this web
    pub uses: Vec<usize>,
    /// live across at least one call
    pub crosses_call: bool,
    pub interferes: BTreeSet<usize>,
    /// classification guess from the code shape
    pub origin: Origin,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    /// `mr rX, rK` from incoming argument register rK at function entry
    Param { arg_reg: RegId },
    /// `mr rX, r3`/`fmr fX, f1` directly from a call result (temp-class signature)
    CallResult { call: usize },
    /// call result moved through r0 (`mr r0,r3 ... mr rX,r0`): named-local signature
    CallResultHop { call: usize },
    /// copy from another register
    Copy { src: RegId },
    /// arithmetic on a call result (e.g. `fadds f31, f0, f1` right after a bl)
    FromCallResult { call: usize },
    /// anything else (load, arithmetic...)
    Computed,
}

/// Build webs for callee-saved registers.
pub fn webs(ins: &[Instr]) -> Vec<Web> {
    webs_ext(ins, false)
}

fn is_allocatable(r: RegId) -> bool {
    !(r == 1 || r == 2 || r == 13)
}

/// Build webs; `all` = every allocatable GPR/FPR (r0, r3..r31, f0..f31), else callee-saved only.
pub fn webs_ext(ins: &[Instr], all: bool) -> Vec<Web> {
    let want = |r: RegId| if all { is_allocatable(r) } else { is_callee_saved(r) };
    let bs = blocks(ins);
    let n = ins.len();
    // ---- reaching definitions, per register, def ids = instruction index; ENTRY = n
    let entry = n;
    // block-level: for each reg tracked, set of reaching defs at block entry
    let tracked: Vec<RegId> = (0u8..64).filter(|&r| is_allocatable(r)).collect();
    let mut rd_in: Vec<BTreeMap<RegId, BTreeSet<usize>>> = vec![BTreeMap::new(); bs.len()];
    for &r in &tracked {
        rd_in[0].entry(r).or_default().insert(entry);
    }
    let transfer = |b: &Block, mut cur: BTreeMap<RegId, BTreeSet<usize>>| {
        for i in b.start..b.end {
            for &r in &tracked {
                if ins[i].defs & bit(r) != 0 {
                    let s = cur.entry(r).or_default();
                    s.clear();
                    s.insert(i);
                }
            }
        }
        cur
    };
    let mut preds: Vec<Vec<usize>> = vec![vec![]; bs.len()];
    for (k, b) in bs.iter().enumerate() {
        for &s in &b.succs {
            preds[s].push(k);
        }
    }
    let mut changed = true;
    let mut rd_out: Vec<BTreeMap<RegId, BTreeSet<usize>>> = vec![BTreeMap::new(); bs.len()];
    while changed {
        changed = false;
        for k in 0..bs.len() {
            let mut inn: BTreeMap<RegId, BTreeSet<usize>> = if k == 0 { rd_in[0].clone() } else { BTreeMap::new() };
            for &p in &preds[k] {
                for (r, s) in &rd_out[p] {
                    inn.entry(*r).or_default().extend(s.iter().copied());
                }
            }
            let out = transfer(&bs[k], inn.clone());
            if out != rd_out[k] || inn != rd_in[k] {
                rd_out[k] = out;
                rd_in[k] = inn;
                changed = true;
            }
        }
    }
    // reaching defs before each instruction
    let mut rd_at: Vec<BTreeMap<RegId, BTreeSet<usize>>> = vec![BTreeMap::new(); n];
    for (k, b) in bs.iter().enumerate() {
        let mut cur = rd_in[k].clone();
        for i in b.start..b.end {
            rd_at[i] = cur.clone();
            for &r in &tracked {
                if ins[i].defs & bit(r) != 0 {
                    let s = cur.entry(r).or_default();
                    s.clear();
                    s.insert(i);
                }
            }
        }
    }
    // ---- union-find over (def, reg)
    let mut parent: BTreeMap<(usize, RegId), (usize, RegId)> = BTreeMap::new();
    fn find(p: &mut BTreeMap<(usize, RegId), (usize, RegId)>, x: (usize, RegId)) -> (usize, RegId) {
        let mut r = x;
        while let Some(&q) = p.get(&r) {
            if q == r {
                break;
            }
            r = q;
        }
        p.insert(x, r);
        r
    }
    let mut use_web: Vec<(usize, RegId, Vec<usize>)> = Vec::new();
    for i in 0..n {
        for r in 0u8..64 {
            if !want(r) || ins[i].uses & bit(r) == 0 {
                continue;
            }
            let Some(rds) = rd_at[i].get(&r) else { continue };
            let defs: Vec<usize> = rds.iter().copied().filter(|&d| d != entry).collect();
            if defs.is_empty() {
                continue; // prologue save of the caller's value
            }
            for &d in &defs {
                parent.entry((d, r)).or_insert((d, r));
            }
            for w in defs.windows(2) {
                let a = find(&mut parent, (w[0], r));
                let b = find(&mut parent, (w[1], r));
                if a != b {
                    parent.insert(a, b);
                }
            }
            use_web.push((i, r, defs));
        }
    }
    let mut web_of: BTreeMap<(usize, RegId), usize> = BTreeMap::new();
    let mut out: Vec<Web> = Vec::new();
    let keys: Vec<(usize, RegId)> = parent.keys().copied().collect();
    for k in keys {
        let root = find(&mut parent, k);
        let id = *web_of.entry(root).or_insert_with(|| {
            out.push(Web {
                id: out.len(),
                reg: root.1,
                defs: vec![],
                uses: vec![],
                crosses_call: false,
                interferes: BTreeSet::new(),
                origin: Origin::Computed,
            });
            out.len() - 1
        });
        web_of.insert(k, id);
        out[id].defs.push(k.0);
    }
    for (i, r, defs) in &use_web {
        let id = web_of[&(defs[0], *r)];
        out[id].uses.push(*i);
    }
    for w in out.iter_mut() {
        w.defs.sort();
        w.defs.dedup();
        w.uses.sort();
        w.uses.dedup();
    }
    // ---- liveness per tracked callee-saved reg (only uses with real defs count)
    let real_use = |i: usize, r: RegId| -> bool {
        ins[i].uses & bit(r) != 0 && rd_at[i].get(&r).map_or(false, |s| s.iter().any(|&d| d != entry))
    };
    let cs: Vec<RegId> = (0u8..64).filter(|&r| want(r)).collect();
    let mut live_in: Vec<u64> = vec![0; bs.len()];
    let mut live_out: Vec<u64> = vec![0; bs.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for k in (0..bs.len()).rev() {
            let mut out_s = 0u64;
            for &s in &bs[k].succs {
                out_s |= live_in[s];
            }
            let mut cur = out_s;
            for i in (bs[k].start..bs[k].end).rev() {
                for &r in &cs {
                    if ins[i].defs & bit(r) != 0 {
                        cur &= !bit(r);
                    }
                }
                for &r in &cs {
                    if real_use(i, r) {
                        cur |= bit(r);
                    }
                }
            }
            if cur != live_in[k] || out_s != live_out[k] {
                live_in[k] = cur;
                live_out[k] = out_s;
                changed = true;
            }
        }
    }
    // live-after sets per instruction
    let mut live_after: Vec<u64> = vec![0; n];
    for (k, b) in bs.iter().enumerate() {
        let mut cur = live_out[k];
        for i in (b.start..b.end).rev() {
            live_after[i] = cur;
            for &r in &cs {
                if ins[i].defs & bit(r) != 0 {
                    cur &= !bit(r);
                }
            }
            for &r in &cs {
                if real_use(i, r) {
                    cur |= bit(r);
                }
            }
        }
    }
    // web live at a point = reg live there and reaching def belongs to the web
    let web_at = |i: usize, r: RegId, after: bool| -> Option<usize> {
        // reaching defs after instruction i (or before i if !after)
        if after && ins[i].defs & bit(r) != 0 {
            return web_of.get(&(i, r)).copied();
        }
        let rds = rd_at[i].get(&r)?;
        rds.iter().filter(|&&d| d != entry).find_map(|&d| web_of.get(&(d, r)).copied())
    };
    let mut inter: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut crosses: BTreeSet<usize> = BTreeSet::new();
    for i in 0..n {
        let la = live_after[i];
        let live_webs: Vec<usize> = cs.iter().filter(|&&r| la & bit(r) != 0).filter_map(|&r| web_at(i, r, true)).collect();
        if ins[i].is_call {
            for &w in &live_webs {
                // the web must also be live before the call (not defined by it)
                crosses.insert(w);
            }
        }
        for &r in &cs {
            if ins[i].defs & bit(r) == 0 {
                continue;
            }
            let Some(&dw) = web_of.get(&(i, r)) else { continue };
            for &lw in &live_webs {
                if lw == dw {
                    continue;
                }
                // MWCC: a move's source does not interfere with its destination
                if let Some((_, src)) = ins[i].mov {
                    if out[lw].reg == src {
                        continue;
                    }
                }
                inter.insert((dw.min(lw), dw.max(lw)));
            }
        }
    }
    for (a, b) in inter {
        out[a].interferes.insert(b);
        out[b].interferes.insert(a);
    }
    for w in crosses {
        out[w].crosses_call = true;
    }
    // ---- origins
    for w in out.iter_mut() {
        let d = w.defs[0];
        w.origin = match ins[d].mov.or(ins[d].mov_dot) {
            Some((_, src)) => {
                let src_defs = rd_at[d].get(&src).cloned().unwrap_or_default();
                if src_defs.len() == 1 && src_defs.contains(&entry) && ((3..=10).contains(&src) || (33..=45).contains(&src)) {
                    Origin::Param { arg_reg: src }
                } else if (src == 3 || src == 33) && src_defs.len() == 1 && src_defs.iter().all(|&s| s < n && ins[s].is_call) {
                    Origin::CallResult { call: *src_defs.iter().next().unwrap() }
                } else if src == 0 || src == 32 {
                    // hop: r0 defined by mr r0, r3 after a call
                    let hop = src_defs.iter().next().copied().filter(|&s| s < n);
                    match hop.and_then(|h| ins[h].mov.map(|m| (h, m))) {
                        Some((h, (_, s3))) if s3 == 3 || s3 == 33 => {
                            let c = rd_at[h].get(&s3).and_then(|s| s.iter().next().copied()).filter(|&c| c < n && ins[c].is_call);
                            match c {
                                Some(c) => Origin::CallResultHop { call: c },
                                None => Origin::Copy { src },
                            }
                        }
                        _ => Origin::Copy { src },
                    }
                } else {
                    Origin::Copy { src }
                }
            }
            None => {
                // defined directly as the destination of an instruction; note if an operand is a
                // call result (r3/f1 straight from a bl)
                let mut o = Origin::Computed;
                for src in [3u8, 33u8] {
                    if ins[d].uses & bit(src) != 0 {
                        if let Some(rds) = rd_at[d].get(&src) {
                            if rds.len() == 1 {
                                let c = *rds.iter().next().unwrap();
                                if c < n && ins[c].is_call {
                                    o = Origin::FromCallResult { call: c };
                                }
                            }
                        }
                    }
                }
                o
            }
        };
    }
    out
}

/// For a call instruction, the immediate loaded into r3 most recently before it (if any).
pub fn call_imm_arg(ins: &[Instr], call: usize) -> Option<i32> {
    for i in (0..call).rev() {
        if ins[i].is_call {
            return None;
        }
        if let Some((3, v)) = ins[i].li {
            return Some(v);
        }
        if ins[i].defs & bit(3) != 0 {
            return None;
        }
    }
    None
}

/// Estimated interference-graph degree (MWCC's IGNode x12 before simplification) of every web
/// from an all-register web set (`webs_ext(ins, true)`): number of interfering virtual webs
/// (webs not pinned to a physical register by a call result) plus physical neighbours (all 11/14
/// volatile registers for call-crossing values). Argument/return registers have no web (their
/// values are consumed by bl/blr, which we do not model as uses), so they are not counted.
pub fn estimate_degrees(ins: &[Instr], ws: &[Web]) -> Vec<u32> {
    let mode = std::env::var("MWDEC_DEG_MODE").unwrap_or_default();
    let pinned: Vec<bool> = ws.iter().map(|w| mode != "nopin" && w.defs.iter().any(|&d| ins[d].is_call)).collect();
    ws.iter()
        .map(|w| {
            let float = w.reg >= 32;
            let mut virt = 0u32;
            let mut phys: BTreeSet<RegId> = BTreeSet::new();
            for &j in &w.interferes {
                if (ws[j].reg >= 32) != float {
                    continue;
                }
                if pinned[j] {
                    phys.insert(ws[j].reg);
                } else {
                    virt += 1;
                }
            }
            let p = if w.crosses_call { if float { 14 } else { 11 } } else { phys.len() as u32 };
            let bias: i32 = std::env::var("MWDEC_DEG_BIAS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            ((virt + p) as i32 + bias).max(0) as u32
        })
        .collect()
}
