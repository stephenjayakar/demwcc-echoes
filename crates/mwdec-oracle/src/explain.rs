//! Scheduling diffs explained by the real scheduler: compile the candidate under the tracer, find
//! every instruction pair the target orders differently (`schedcheck`), and ask the recorded
//! post-RA and pre-RA schedules *why* the candidate issued them in its order. The verdict says what
//! to change in the source:
//! - `Forced`: a data/memory dependence orders them; swapping statements cannot help (remove the
//!   dependence, e.g. cache a load, or change which object is accessed);
//! - `RegAlloc`: only a register reuse (anti/output dependence after allocation) orders them: fix
//!   the register assignment first;
//! - `SwapStatements`: both were ready with equal priority and program order decided: move the
//!   statement of the second instruction before the first's;
//! - `WaitFor`: the second instruction was not ready; the instruction it waited for must come
//!   earlier (move that statement);
//! - `Priority`: the pick rule decided by deadline / successors uncovered / height / opcode rank /
//!   unit availability: statement order does not matter; change the dependence chains instead.
//!
//! The entry block is compared too: the post-RA pass interleaves the prologue's register saves
//! with the first body instructions, and whether a load through a parameter can move above the
//! saves is decided by the parameter's declared type (compiler experiments, GC/2.7): the
//! pointee of a pointer/reference-to-const parameter (`this` of a const member function) is a
//! known object that no store reaches, so its loads are free; a non-const pointer, or a pointee
//! class with a `mutable` non-pointer member anywhere inside (e.g. an `auto_ptr`'s ownership
//! flag), is of unknown origin and every load through it waits for the saves. Verdicts:
//! - `ConstBase`: the target reads before a save the candidate's load depends on: the target's
//!   parameter is pointer/reference-to-const;
//! - `NonConstBase`: the target waits for a save the candidate's (free) load does not: the target's
//!   parameter is not trusted read-only.

use crate::asm::{self, Func, Obj};
use crate::compile::Compiler;
use crate::sched::{Explanation, PickReason, SchedBlock, WhyKind};
use crate::schedcheck::{self, CheckOptions, InstrRef};
use crate::tracer::{trace_source, TraceOptions};
use anyhow::{anyhow, Result};
use ppc750cl::Ins;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub enum Verdict {
    Forced(String),
    RegAlloc,
    /// move source line `.0` before line `.1`
    SwapStatements(Option<u32>, Option<u32>),
    /// the second instruction waits for this instruction (text, candidate source line)
    WaitFor(String, Option<u32>),
    Priority(PickReason),
    /// entry block: the target loads through parameter register `.0` before a register save
    /// (the parameter is pointer/reference-to-const there)
    ConstBase(u8),
    /// entry block: the target's load through parameter register `.0` waits for the register
    /// saves (a non-const parameter, or a pointee class with a mutable non-pointer member)
    NonConstBase(u8),
    Unknown,
}

#[derive(Clone, Debug, Serialize)]
pub struct OrderAdvice {
    /// candidate order: `first` before `second`; the target wants `second` first
    pub first: InstrRef,
    pub second: InstrRef,
    pub post_ra: Option<Explanation>,
    pub pre_ra: Option<Explanation>,
    pub verdict: Verdict,
    pub text: String,
}

/// Canonical base mnemonic so PCode opcode names and disassembler (simplified) mnemonics compare.
fn canon(m: &str) -> &str {
    let m = m.trim_end_matches('.');
    match m {
        "li" | "addi" | "subi" | "la" => "addi",
        "lis" | "addis" | "subis" => "addis",
        "mr" | "or" => "or",
        "slwi" | "srwi" | "clrlwi" | "clrrwi" | "rotlwi" | "rotrwi" | "extlwi" | "extrwi" | "rlwinm" | "clrlslwi" => "rlwinm",
        "inslwi" | "insrwi" | "rlwimi" => "rlwimi",
        "rotlw" | "rlwnm" => "rlwnm",
        "cmpwi" | "cmpi" => "cmpi",
        "cmplwi" | "cmpli" => "cmpli",
        "cmpw" | "cmp" => "cmp",
        "cmplw" | "cmpl" => "cmpl",
        "sub" | "subf" => "subf",
        "subc" | "subfc" => "subfc",
        "not" | "nor" => "nor",
        "nop" | "ori" => "ori",
        "mtlr" | "mtctr" | "mtspr" | "mtxer" => "mtspr",
        "mflr" | "mfctr" | "mfspr" | "mfxer" => "mfspr",
        "fmr" => "fmr",
        _ if m.starts_with('b') && !m.starts_with("bl") && m != "b" && !m.starts_with("bctr") => "bc",
        _ => m,
    }
}

/// Register-insensitive key of a PCode text (mnemonic + immediates + memory operands).
fn loose_key(text: &str) -> String {
    let mut it = text.split_whitespace();
    let m = canon(it.next().unwrap_or("")).to_string();
    let rest: String = it.collect::<Vec<_>>().join(" ");
    let ops: Vec<String> = rest
        .split(", ")
        .map(|o| o.trim().trim_end_matches('=').to_string())
        .filter(|o| {
            let reg = |p: &str| o.strip_prefix(p).map_or(false, |n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
            !(reg("r") || reg("f") || reg("cr") || reg("spr") || o.starts_with('<') || o.is_empty())
        })
        .collect();
    format!("{m} {}", ops.join(","))
}

/// Final-code offsets of the nodes of every recorded block. Blocks the post-RA pass rescheduled
/// (prologue/epilogue blocks and blocks merged into them) are matched first, as windows of the
/// final code in issue order; the remaining code comes from the pre-RA schedule (the post-RA pass
/// skips blocks it already scheduled), matched as a subsequence because later passes delete
/// coalesced moves and fold instructions.
fn align(func: &Func, post: &[&SchedBlock], pre: &[&SchedBlock]) -> (Vec<Vec<Option<u32>>>, Vec<Vec<Option<u32>>>) {
    let mn: Vec<String> = func
        .code
        .chunks_exact(4)
        .map(|c| canon(Ins::new(u32::from_be_bytes([c[0], c[1], c[2], c[3]])).simplified().mnemonic).to_string())
        .collect();
    let seq = |b: &SchedBlock| -> Vec<String> {
        b.picks.iter().map(|p| canon(b.nodes[p.node].text.split_whitespace().next().unwrap_or("")).to_string()).collect()
    };
    let mut claimed = vec![false; mn.len()];
    let mut post_maps = vec![];
    let mut cursor = 0usize;
    for b in post {
        let picks = seq(b);
        let n = picks.len();
        let mut map = vec![None; b.nodes.len()];
        let mut found = None;
        for start in cursor..mn.len().saturating_sub(n.saturating_sub(1)) {
            let score = (0..n).filter(|&k| mn.get(start + k) == Some(&picks[k])).count();
            if n > 0 && score * 10 >= n * 8 {
                found = Some(start);
                break;
            }
        }
        if let Some(st) = found {
            for (k, p) in b.picks.iter().enumerate() {
                map[p.node] = Some(((st + k) * 4) as u32);
                claimed[st + k] = true;
            }
            cursor = st + n;
        }
        post_maps.push(map);
    }
    let free: Vec<usize> = (0..mn.len()).filter(|&i| !claimed[i]).collect();
    let mut pre_maps = vec![];
    let mut pos = 0usize;
    for b in pre {
        let picks = seq(b);
        let mut map = vec![None; b.nodes.len()];
        for (k, p) in b.picks.iter().enumerate() {
            if let Some(j) = (pos..(pos + 4).min(free.len())).find(|&j| mn[free[j]] == picks[k]) {
                map[p.node] = Some((free[j] * 4) as u32);
                pos = j + 1;
            }
        }
        pre_maps.push(map);
    }
    (post_maps, pre_maps)
}

fn plain_name(func: &str) -> String {
    match func.find("__") {
        Some(i) if i > 0 => func[..i].to_string(),
        _ => func.to_string(),
    }
}

/// Explain every instruction-order difference between the candidate source (compiled with `comp`)
/// and the target function `target` of `target_obj`. `func` is the function's (mangled) name.
pub fn explain_sched_diff(comp: &Compiler, cand_src: &str, func: &str, target_obj: &Obj, target: &Func) -> Result<Vec<OrderAdvice>> {
    let opts = TraceOptions { sched: true, sched_filter: Some(plain_name(func)), ..Default::default() };
    let t = trace_source(comp, cand_src, &opts)?;
    let cobj = asm::parse(&t.object)?;
    let pick = |o: &Obj| -> Option<Func> {
        o.funcs.iter().find(|f| f.name == func).or_else(|| o.funcs.iter().find(|f| f.name.contains(func))).cloned()
    };
    let cf = pick(&cobj).ok_or_else(|| anyhow!("{func} not in candidate"))?;
    // unscheduled -sym on compile: exact statement lines and local extents
    let mut c2 = comp.clone();
    c2.extra.extend(["-sym".to_string(), "on".to_string()]);
    let uns = c2.compile(&format!("#pragma scheduling off\n{cand_src}")).ok().and_then(|o| asm::parse(&o.object).ok());
    let lines = uns.as_ref().and_then(|u| pick(u).map(|uf| schedcheck::attribute_lines(&cobj, &cf, u, &uf, 1)));
    let objs = uns.as_ref().and_then(|u| pick(u).map(|uf| schedcheck::stack_objects(u, &uf.name))).filter(|o| !o.is_empty());
    let rep = schedcheck::check_with(
        target_obj,
        target,
        &cobj,
        &cf,
        &CheckOptions { lines: lines.as_deref(), cand_stack: objs.as_deref(), target_stack: objs.as_deref() },
    );
    let post: Vec<&SchedBlock> = t.sched.iter().filter(|b| !b.pre_ra).collect();
    let pre: Vec<&SchedBlock> = t.sched.iter().filter(|b| b.pre_ra).collect();
    let (post_maps, pre_maps) = align(&cf, &post, &pre);
    // (is post-RA, block, node) of a final instruction
    let locate = |off: u32| -> Option<(bool, usize, usize)> {
        let find = |maps: &Vec<Vec<Option<u32>>>| {
            maps.iter().enumerate().find_map(|(bi, m)| m.iter().position(|&o| o == Some(off)).map(|ni| (bi, ni)))
        };
        find(&post_maps).map(|(b, n)| (true, b, n)).or_else(|| find(&pre_maps).map(|(b, n)| (false, b, n)))
    };
    // pre-RA node with the same register-insensitive key (first unused per block)
    let find_pre = |text: &str, other: &str| -> Option<(usize, usize, usize)> {
        let (ka, kb) = (loose_key(text), loose_key(other));
        for (bi, b) in pre.iter().enumerate() {
            let a = b.nodes.iter().position(|n| loose_key(&n.text) == ka);
            let c = b.nodes.iter().position(|n| loose_key(&n.text) == kb);
            if let (Some(a), Some(c)) = (a, c) {
                return Some((bi, a, c));
            }
        }
        None
    };
    // prologue/epilogue instructions carry the function's first/last line: never a statement to move
    let frame = schedcheck::frame_offsets(&cf);
    let line_of = |off: u32| -> Option<u32> {
        if frame.contains(&off) {
            return None;
        }
        lines.as_ref().and_then(|l| l.get((off / 4) as usize).copied().flatten())
    };
    // loads through a parameter register (base never written before them)
    let param_load: BTreeMap<u32, u8> =
        entry_region(&cobj, &cf, false).into_iter().filter_map(|x| Some((x.off, x.param_load?.0))).collect();
    let mut out = vec![];
    for blk in &rep.blocks {
        for inv in &blk.inversions {
            let unframe = |r: &InstrRef| {
                let mut r = r.clone();
                if frame.contains(&r.cand_off) {
                    r.line = None;
                }
                r
            };
            let (f, s) = (&unframe(&inv.cand_first), &unframe(&inv.cand_second));
            let mut adv = OrderAdvice {
                first: f.clone(),
                second: s.clone(),
                post_ra: None,
                pre_ra: None,
                verdict: Verdict::Unknown,
                text: String::new(),
            };
            let (Some((pf, bf, nf)), Some((ps, bs, ns))) = (locate(f.cand_off), locate(s.cand_off)) else {
                adv.text = "instructions not found in the recorded schedule".into();
                out.push(adv);
                continue;
            };
            if bf != bs || pf != ps {
                adv.text = "different scheduling blocks".into();
                out.push(adv);
                continue;
            }
            let b = if pf { post[bf] } else { pre[bf] };
            let maps = if pf { &post_maps } else { &pre_maps };
            let e = b.why_before(nf, ns);
            adv.verdict = match &e.why {
                WhyKind::Dependence(path) => {
                    if path.iter().all(|k| matches!(k, crate::sched::EdgeKind::Anti | crate::sched::EdgeKind::Output)) {
                        Verdict::RegAlloc
                    } else {
                        Verdict::Forced(format!("{path:?}"))
                    }
                }
                WhyKind::NotReady { waiting_for: Some(w), .. } => {
                    let woff = maps[bf][*w];
                    Verdict::WaitFor(b.nodes[*w].text.clone(), woff.and_then(line_of))
                }
                WhyKind::Priority(PickReason::ProgramOrder) => Verdict::SwapStatements(s.line, f.line),
                WhyKind::Priority(r) => Verdict::Priority(*r),
                _ => Verdict::Unknown,
            };
            // program order of the post-RA pass is the pre-RA schedule: ask the pre-RA pass too
            if pf && matches!(adv.verdict, Verdict::SwapStatements(..)) {
                if let Some((pb, pa, pc)) = find_pre(&b.nodes[nf].text, &b.nodes[ns].text) {
                    let pe = pre[pb].why_before(pa, pc);
                    adv.verdict = match &pe.why {
                        WhyKind::Dependence(path) => Verdict::Forced(format!("pre-RA {path:?}")),
                        WhyKind::NotReady { waiting_for: Some(w), .. } => Verdict::WaitFor(pre[pb].nodes[*w].text.clone(), None),
                        WhyKind::Priority(PickReason::ProgramOrder) | WhyKind::NotBefore => {
                            Verdict::SwapStatements(s.line, f.line)
                        }
                        WhyKind::Priority(r) => Verdict::Priority(*r),
                        _ => adv.verdict.clone(),
                    };
                    adv.pre_ra = Some(pe);
                }
            }
            // a load through a parameter held back by a store (or register save) that the target
            // issues before it: the target's pointee is a known object (pointer-to-const parameter)
            if let Some(&r) = param_load.get(&s.cand_off) {
                let memory = |k: crate::sched::EdgeKind| k == crate::sched::EdgeKind::Memory;
                let held = match &e.why {
                    WhyKind::Dependence(p) => p.iter().any(|&k| memory(k)),
                    WhyKind::NotReady { waiting_for: Some(w), .. } => b.nodes[*w].succs.iter().any(|x| x.to == ns && memory(x.kind)),
                    _ => false,
                };
                if held && matches!(adv.verdict, Verdict::Forced(_) | Verdict::WaitFor(..)) {
                    adv.verdict = Verdict::ConstBase(r);
                }
            }
            adv.text = match &adv.verdict {
                Verdict::Forced(p) => format!("[{}] depends on [{}] ({p}): change the dependence, not the order", s.text, f.text),
                Verdict::RegAlloc => "only register reuse orders them: fix the register assignment first".into(),
                Verdict::SwapStatements(a, b) => match (a, b) {
                    (Some(a), Some(b)) if a != b => format!("program order decides: move line {a} before line {b}"),
                    _ => "program order decides: emit the second instruction's expression first".into(),
                },
                Verdict::WaitFor(w, l) => format!(
                    "[{}] waited for [{w}]{}: move that computation earlier",
                    s.text,
                    l.map(|x| format!(" (line {x})")).unwrap_or_default()
                ),
                Verdict::Priority(r) => format!("the scheduler's priority ({r:?}) decides, not statement order"),
                Verdict::ConstBase(r) => const_base_text(*r),
                Verdict::NonConstBase(r) => non_const_base_text(*r),
                Verdict::Unknown => e.text.clone(),
            };
            if pf {
                adv.post_ra = Some(e);
            } else {
                adv.pre_ra = Some(e);
            }
            out.push(adv);
        }
    }
    let explained: Vec<u32> = out
        .iter()
        .filter(|a| matches!(a.verdict, Verdict::ConstBase(_) | Verdict::NonConstBase(_)))
        .map(|a| a.second.cand_off)
        .collect();
    out.extend(entry_advice(&cobj, &cf, target_obj, target, &post, &post_maps).into_iter().filter(|a| {
        let load = if matches!(a.verdict, Verdict::ConstBase(_)) { a.second.cand_off } else { a.first.cand_off };
        !explained.contains(&load)
    }));
    Ok(out)
}

fn const_base_text(r: u8) -> String {
    format!("the target issues the load through r{r} before a store or register save the candidate orders before it: the pointee is a known object there, i.e. the parameter is declared pointer/reference-to-const (`this`: a const member function whose class has no mutable non-pointer member)")
}

fn non_const_base_text(r: u8) -> String {
    format!("the target's load through r{r} waits for the register save: the parameter is not trusted read-only there (a non-const pointer/reference, or the pointee class has a mutable non-pointer member, e.g. an auto_ptr's ownership flag)")
}

/// One instruction of a function's entry region (up to the first branch or call).
struct EntryIns {
    off: u32,
    text: String,
    /// register save of the prologue (`stw`/`stfd`/`stmw`/`psq_st` of the frame)
    save: bool,
    /// load through a parameter register no earlier instruction wrote: (register, mnemonic +
    /// displacement)
    param_load: Option<(u8, String)>,
}

/// The instructions of `f` with their save / parameter-load facts; `entry`: only the entry region
/// (up to the first branch or call), else the whole function in address order.
fn entry_region(obj: &Obj, f: &Func, entry: bool) -> Vec<EntryIns> {
    let frame = schedcheck::frame_offsets(f);
    let lines = asm::disasm_func(obj, f, asm::AsmOpts { offsets: true, literals: false });
    let texts: BTreeMap<u32, String> = lines
        .iter()
        .filter_map(|l| {
            let (o, t) = l.trim().split_once(": ")?;
            Some((u32::from_str_radix(o.trim(), 16).ok()?, t.trim().to_string()))
        })
        .collect();
    let reg = |t: &str| t.strip_prefix('r').and_then(|n| n.parse::<u8>().ok());
    let mut written = [false; 32];
    let mut out = vec![];
    for (i, c) in f.code.chunks_exact(4).enumerate() {
        let off = (i * 4) as u32;
        let ins = Ins::new(u32::from_be_bytes([c[0], c[1], c[2], c[3]]));
        let text = texts.get(&off).cloned().unwrap_or_else(|| ins.simplified().to_string());
        let m = text.split_whitespace().next().unwrap_or("").to_string();
        if m.starts_with('b') {
            if entry {
                break;
            }
            continue;
        }
        let ops: Vec<&str> = text.splitn(2, ' ').nth(1).unwrap_or("").split(", ").map(|o| o.trim()).collect();
        let save = frame.contains(&off) && matches!(m.as_str(), "stw" | "stfd" | "stmw" | "psq_st");
        let mut param_load = None;
        let is_load = m.starts_with('l') && !matches!(m.as_str(), "li" | "lis");
        if is_load && !frame.contains(&off) {
            if let Some((disp, base)) = ops.get(1).and_then(|o| o.strip_suffix(')')).and_then(|o| o.split_once('(')) {
                if let Some(b) = reg(base) {
                    if (3..=10).contains(&b) && !written[b as usize] {
                        param_load = Some((b, format!("{m} {disp}")));
                    }
                }
            }
        }
        // the register this instruction writes (first operand of everything but stores/compares)
        if !m.starts_with("st") && !m.starts_with("cmp") {
            if let Some(d) = ops.first().and_then(|o| reg(o)) {
                written[d as usize] = true;
            }
        }
        out.push(EntryIns { off, text, save, param_load });
    }
    out
}

/// Entry-block order differences between the prologue's register saves and the loads through
/// parameters, explained by the alias class of the load (see the module docs).
fn entry_advice(cobj: &Obj, cf: &Func, tobj: &Obj, tf: &Func, post: &[&SchedBlock], post_maps: &[Vec<Option<u32>>]) -> Vec<OrderAdvice> {
    let c = entry_region(cobj, cf, true);
    let t = entry_region(tobj, tf, true);
    let mut used = vec![false; t.len()];
    // the target counterpart: saves by exact text, loads by mnemonic + displacement
    let mut pairs: Vec<(usize, usize)> = vec![];
    for (i, x) in c.iter().enumerate() {
        if !x.save && x.param_load.is_none() {
            continue;
        }
        let k = t.iter().enumerate().position(|(j, y)| {
            !used[j]
                && match (&x.param_load, &y.param_load) {
                    (Some((_, a)), Some((_, b))) => a == b,
                    (None, None) => x.save && y.save && x.text == y.text,
                    _ => false,
                }
        });
        if let Some(k) = k {
            used[k] = true;
            pairs.push((i, k));
        }
    }
    // (block, node) of a candidate offset in the post-RA schedule
    let node = |off: u32| post_maps.iter().enumerate().find_map(|(b, m)| m.iter().position(|&o| o == Some(off)).map(|n| (b, n)));
    let mut out = vec![];
    for &(li, lk) in pairs.iter().filter(|&&(i, _)| c[i].param_load.is_some()) {
        let r = c[li].param_load.as_ref().map_or(0, |p| p.0);
        for &(si, sk) in pairs.iter().filter(|&&(i, _)| c[i].save) {
            let cand_save_first = c[si].off < c[li].off;
            if cand_save_first == (t[sk].off < t[lk].off) {
                continue;
            }
            let (Some((bs, ns)), Some((bl, nl))) = (node(c[si].off), node(c[li].off)) else { continue };
            if bs != bl {
                continue;
            }
            let e = if cand_save_first { post[bs].why_before(ns, nl) } else { post[bs].why_before(nl, ns) };
            let memory = matches!(&e.why, WhyKind::Dependence(p) if p.contains(&crate::sched::EdgeKind::Memory));
            let (verdict, text) = if cand_save_first && memory {
                (Verdict::ConstBase(r), const_base_text(r))
            } else if !cand_save_first && !matches!(&e.why, WhyKind::Dependence(_)) {
                (Verdict::NonConstBase(r), non_const_base_text(r))
            } else {
                continue;
            };
            let iref = |i: usize, k: usize| InstrRef { text: c[i].text.clone(), cand_off: c[i].off, target_off: t[k].off, line: None };
            let (first, second) = if cand_save_first { (iref(si, sk), iref(li, lk)) } else { (iref(li, lk), iref(si, sk)) };
            out.push(OrderAdvice { first, second, post_ra: Some(e), pre_ra: None, verdict, text });
            break;
        }
    }
    out
}

