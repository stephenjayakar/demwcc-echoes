//! Diff localisation: which source statements of a candidate produce the instructions that differ
//! from the target, and which statement moves the scheduler cannot do by itself.
//!
//! The candidate is compiled twice more with `-sym on` (line table): once as is (`-sym on` changes
//! no instruction selection inside functions, only literal label numbering) and once with
//! `#pragma scheduling off` prepended, whose line table is exact per instruction
//! (`mwdec_oracle::schedcheck::attribute_lines`). Both compiles are diagnostics only: candidates are
//! still judged by the plain compile. The pragma never appears in emitted output.
//!
//! - [`Locator::analyze`] -> [`Analysis`]: candidate source lines of differing instructions (drives
//!   focused mutations), plus `schedcheck` statement moves for reorder diffs.
use crate::score::align_pairs;
use mwdec_core::{Function, RelocKind};
use mwdec_mwcc::compare::masked_words;
use mwdec_mwcc::{Mwcc, UnitContext};
use mwdec_oracle::asm;
use mwdec_oracle::schedcheck::{self, DepKind};
use std::collections::BTreeMap;

/// Diagnostic compiles for one unit.
pub struct Locator<'a> {
    pub mwcc: &'a Mwcc,
    /// The unit context with `-sym on` appended to the flags.
    pub ctx: UnitContext,
    pub target: asm::Obj,
    pub tf: asm::Func,
    tw: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Analysis {
    /// 1-based candidate source lines with differing instructions, most differences first.
    pub lines: Vec<u32>,
    /// (line that must move earlier, line it must precede, dependence) from schedcheck.
    pub moves: Vec<(u32, u32, DepKind)>,
    /// Independent pairs scheduled in the other order (line first in the target, line first in ours).
    pub swaps: Vec<(u32, u32)>,
    /// Register definitions of differing candidate instructions: (line, FPR?, register).
    pub defs: Vec<(u32, bool, u8)>,
}

impl Analysis {
    /// Lines whose differing instructions define register `reg` of the class.
    pub fn def_lines(&self, fpr: bool, reg: u8) -> Vec<u32> {
        let mut v: Vec<u32> = self.defs.iter().filter(|d| d.1 == fpr && d.2 == reg).map(|d| d.0).collect();
        v.sort();
        v.dedup();
        v
    }
}

fn r_type(k: RelocKind) -> u32 {
    match k {
        RelocKind::Addr32 => 1,
        RelocKind::Addr16Lo => 4,
        RelocKind::Addr16Hi => 5,
        RelocKind::Addr16Ha => 6,
        RelocKind::Rel24 => 10,
        RelocKind::Rel14 => 11,
        RelocKind::EmbSda21 => 109,
        RelocKind::Other(x) => x,
    }
}

/// `mwdec_core::Function` as the oracle's `asm::Func` (relocations are function-relative in both).
pub fn to_asm(f: &Function) -> asm::Func {
    asm::Func {
        name: f.name.clone(),
        weak: false,
        local: false,
        section: ".text".into(),
        code: f.code.clone(),
        relocs: f
            .relocs
            .iter()
            .map(|r| asm::Rel { offset: r.offset, r_type: r_type(r.kind), target: r.target.clone(), addend: r.addend as i64 })
            .collect(),
        address: f.address,
    }
}

impl<'a> Locator<'a> {
    /// `None` if the `-sym on` context cannot be built.
    pub fn new(mwcc: &'a Mwcc, ctx: &UnitContext, tf: &Function) -> Option<Locator<'a>> {
        let mut flags = ctx.cflags.clone();
        flags.extend(["-sym".to_string(), "on".to_string()]);
        let mut sctx = if ctx.mch.is_some() {
            mwcc.precompile(&ctx.context, &flags).ok()?
        } else {
            mwcc.plain_context(&ctx.context, &flags)
        };
        sctx.tu_name = ctx.tu_name.clone();
        let f = to_asm(tf);
        Some(Locator { mwcc, ctx: sctx, target: asm::Obj { funcs: vec![f.clone()], ..Default::default() }, tf: f, tw: masked_words(tf) })
    }

    fn compile(&self, src: &str, symbol: &str) -> Option<(asm::Obj, asm::Func, Vec<u32>)> {
        let c = self.mwcc.compile_in(&self.ctx, src).ok()?;
        let o = asm::parse(&c.obj).ok()?;
        let f = o.funcs.iter().find(|f| f.name == symbol)?.clone();
        let core = mwdec_obj::load_object_bytes("sym.o", &c.obj).ok()?;
        let w = masked_words(mwdec_obj::find_function(&core, symbol)?);
        Some((o, f, w))
    }

    /// Localise the differences of candidate `src` (which must contain `symbol`).
    pub fn analyze(&self, src: &str, symbol: &str, reorder: bool) -> Option<Analysis> {
        let (cobj, cf, cw) = self.compile(src, symbol)?;
        let (uobj, uf, _) = self.compile(&format!("#pragma scheduling off\n{src}"), symbol)?;
        let lines = schedcheck::attribute_lines(&cobj, &cf, &uobj, &uf, 1);
        // Differing candidate instructions via the same opcode alignment as the penalty.
        let tw = &self.tw;
        let pairs = align_pairs(tw, &cw);
        let mut diff = vec![false; cw.len()];
        let mut matched = vec![false; cw.len()];
        let (mut pi, mut pj) = (0usize, 0usize);
        let mut bounds = pairs.clone();
        bounds.push((tw.len(), cw.len()));
        for &(i, j) in &bounds {
            if j < cw.len() {
                matched[j] = true;
                if tw[i] != cw[j] {
                    diff[j] = true;
                }
            }
            // A gap: unmatched instructions on either side.
            if i > pi || j > pj {
                for d in diff.iter_mut().take(j).skip(pj) {
                    *d = true;
                }
                if i > pi && j == pj {
                    // target-only instructions: blame the neighbours
                    if j < cw.len() {
                        diff[j] = true;
                    }
                    if j > 0 {
                        diff[j - 1] = true;
                    }
                }
            }
            pi = i + 1;
            pj = j + 1;
        }
        let mut count: BTreeMap<u32, usize> = BTreeMap::new();
        for (j, &d) in diff.iter().enumerate() {
            if d {
                if let Some(Some(l)) = lines.get(j) {
                    *count.entry(*l).or_default() += 1;
                }
            }
        }
        let mut ls: Vec<(u32, usize)> = count.into_iter().collect();
        ls.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut an = Analysis { lines: ls.into_iter().map(|x| x.0).collect(), ..Default::default() };
        for (j, &d) in diff.iter().enumerate() {
            let (true, Some(Some(l))) = (d, lines.get(j)) else { continue };
            for a in ppc750cl::Ins::new(cw[j]).defs().iter() {
                match a {
                    ppc750cl::Argument::GPR(g) => an.defs.push((*l, false, g.0)),
                    ppc750cl::Argument::FPR(f) => an.defs.push((*l, true, f.0)),
                    _ => {}
                }
            }
        }
        if reorder {
            let rep = schedcheck::check(&self.target, &self.tf, &cobj, &cf, Some(&lines));
            an.moves = rep.statement_moves.clone();
            for b in &rep.blocks {
                for inv in &b.inversions {
                    if inv.forced.is_none() {
                        if let (Some(l1), Some(l2)) = (inv.cand_first.line, inv.cand_second.line) {
                            if l1 != l2 && !an.swaps.contains(&(l2, l1)) {
                                an.swaps.push((l2, l1));
                            }
                        }
                    }
                }
            }
        }
        Some(an)
    }
}

/// 1-based line of byte offset `pos`.
pub fn line_of(src: &str, pos: usize) -> u32 {
    src.as_bytes()[..pos.min(src.len())].iter().filter(|&&b| b == b'\n').count() as u32 + 1
}

/// Byte ranges `[start, end)` of 1-based `lines` in `src`.
pub fn line_ranges(src: &str, lines: &[u32]) -> Vec<(usize, usize)> {
    let mut starts = vec![0usize];
    for (i, b) in src.bytes().enumerate() {
        if b == b'\n' {
            starts.push(i + 1);
        }
    }
    lines
        .iter()
        .filter_map(|&l| {
            let s = *starts.get(l as usize - 1)?;
            let e = starts.get(l as usize).copied().unwrap_or(src.len());
            Some((s, e))
        })
        .collect()
}

/// The outermost statement of the target function's body that starts on 1-based `line`
/// (compound statements excluded).
fn stmt_at_line(cst: &crate::cst::Cst, body: usize, line: u32) -> Option<usize> {
    let mut best: Option<usize> = None;
    for n in cst.descendants(body) {
        if n == body || !crate::func::is_stmt(cst.kind(n)) || cst.kind(n) == "compound_statement" {
            continue;
        }
        if line_of(&cst.src, cst.nodes[n].start) != line {
            continue;
        }
        match best {
            Some(b) if cst.contains(b, n) => {}
            _ => best = Some(n),
        }
    }
    best
}

/// Sibling statements (same block) containing the statements at lines `x` and `y`.
fn siblings(cst: &crate::cst::Cst, body: usize, x: u32, y: u32) -> Option<(usize, usize)> {
    let sx = stmt_at_line(cst, body, x)?;
    let sy = stmt_at_line(cst, body, y)?;
    let mut a = sx;
    loop {
        let p = cst.parent(a)?;
        if cst.kind(p) == "compound_statement" && cst.contains(p, sy) {
            let mut b = sy;
            while cst.parent(b)? != p {
                b = cst.parent(b)?;
            }
            return (a != b).then_some((a, b));
        }
        if p == body {
            return None;
        }
        a = p;
    }
}

fn indent_at(src: &str, pos: usize) -> String {
    let ls = src[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
    src[ls..pos].chars().take_while(|c| c.is_whitespace()).collect()
}

/// Remove statement `n` with its own line when it is alone on it.
fn removal(cst: &crate::cst::Cst, n: usize) -> crate::cst::Edit {
    let (s, e) = (cst.nodes[n].start, cst.nodes[n].end);
    let ls = cst.src[..s].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let le = cst.src[e..].find('\n').map(|i| e + i + 1).unwrap_or(e);
    if cst.src[ls..s].trim().is_empty() && cst.src[e..le].trim().is_empty() {
        crate::cst::Edit { start: ls, end: le, text: String::new() }
    } else {
        crate::cst::Edit { start: s, end: e, text: String::new() }
    }
}

/// Source edits for "the statement at line `x` must come before the one at line `y`": move x's
/// statement just before y's, or y's just after x's (both at the level of their common block).
/// Empty when the statements are already in that order or not in a common block.
pub fn move_before(src: &str, symbol: &str, x: u32, y: u32) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let cst = Cst::parse(src);
    let Some(def) = crate::func::find_target(&cst, symbol) else { return vec![] };
    let Some(body) = cst.child(def, "body") else { return vec![] };
    let Some((a, b)) = siblings(&cst, body, x, y) else { return vec![] };
    if cst.nodes[a].start < cst.nodes[b].start {
        return vec![];
    }
    let mut out = Vec::new();
    let ind_b = indent_at(src, cst.nodes[b].start);
    // a before b
    let e1 = vec![Edit::insert(cst.nodes[b].start, format!("{}\n{ind_b}", cst.text(a))), removal(&cst, a)];
    // b after a
    let ind_a = indent_at(src, cst.nodes[a].start);
    let e2 = vec![Edit::insert(cst.nodes[a].end, format!("\n{ind_a}{}", cst.text(b))), removal(&cst, b)];
    for e in [e1, e2] {
        if let Some(s) = apply(src, &e) {
            if Cst::parse(&s).errors <= cst.errors && !out.contains(&s) {
                out.push(s);
            }
        }
    }
    out
}

/// Swap the statements at lines `x` and `y` (at the level of their common block).
pub fn swap_lines(src: &str, symbol: &str, x: u32, y: u32) -> Option<String> {
    use crate::cst::{apply, Cst, Edit};
    let cst = Cst::parse(src);
    let def = crate::func::find_target(&cst, symbol)?;
    let body = cst.child(def, "body")?;
    let (a, b) = siblings(&cst, body, x, y)?;
    let s = apply(src, &[Edit::replace(&cst, a, cst.text(b)), Edit::replace(&cst, b, cst.text(a))])?;
    (Cst::parse(&s).errors <= cst.errors).then_some(s)
}

