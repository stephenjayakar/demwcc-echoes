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

use crate::asm::{self, Func, Obj};
use crate::compile::Compiler;
use crate::sched::{Explanation, PickReason, SchedBlock, WhyKind};
use crate::schedcheck::{self, CheckOptions, InstrRef};
use crate::tracer::{trace_source, TraceOptions};
use anyhow::{anyhow, Result};
use ppc750cl::Ins;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub enum Verdict {
    Forced(String),
    RegAlloc,
    /// move source line `.0` before line `.1`
    SwapStatements(Option<u32>, Option<u32>),
    /// the second instruction waits for this instruction (text, candidate source line)
    WaitFor(String, Option<u32>),
    Priority(PickReason),
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
    Ok(out)
}
