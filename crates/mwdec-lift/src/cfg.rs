//! Basic blocks, control-flow edges (incl. jump tables), dominators, post-dominators, loops.

use crate::insn::Insn;
use mwdec_core::{Function, ObjectFile, RelocKind};
use ppc750cl::Opcode;
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    /// Falls into the next block.
    Fall(usize),
    Jump(usize),
    /// Conditional branch at the last instruction: (taken, fallthrough).
    Cond { taken: usize, fall: usize },
    Return,
    /// Conditional return (`beqlr`): (fallthrough block).
    CondReturn { fall: usize },
    /// Jump table: targets by index, the table symbol.
    Switch { targets: Vec<usize>, table: String },
    /// `b sym` out of the function (tail call).
    TailCall,
    /// Unknown indirect jump / falls off the end.
    Stop,
}

#[derive(Clone, Debug)]
pub struct Block {
    /// Instruction index range [start, end).
    pub start: usize,
    pub end: usize,
    pub term: Term,
    pub succs: Vec<usize>,
    pub preds: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Cfg {
    pub blocks: Vec<Block>,
    /// Instruction index -> block.
    pub block_of: Vec<usize>,
    pub rpo: Vec<usize>,
    /// Immediate dominator (entry maps to itself; unreachable: usize::MAX).
    pub idom: Vec<usize>,
    /// Immediate post-dominator; `exit()` is a virtual exit node index (== blocks.len()).
    pub ipdom: Vec<usize>,
    /// Extra edges for the post-dominator computation only: an early return continues (for
    /// structuring purposes) at the shared tail it jumped over.
    pub pd_extra: Vec<(usize, usize)>,
}

pub fn decode(f: &Function) -> Vec<Insn> {
    let mut relocs: HashMap<u32, mwdec_core::Reloc> = HashMap::new();
    for r in &f.relocs {
        // relocations on halfword fields point at offset+2; normalize to the instruction start
        relocs.insert(r.offset & !3, r.clone());
    }
    f.words()
        .enumerate()
        .map(|(i, w)| {
            let off = i as u32 * 4;
            Insn { off, ins: ppc750cl::Ins::new(w), reloc: relocs.get(&off).cloned() }
        })
        .collect()
}

/// Find the jump table symbol for a `bctr` at instruction `i` and resolve its targets
/// (function-relative offsets).
fn jump_table(obj: &ObjectFile, f: &Function, insns: &[Insn], i: usize) -> Option<(String, Vec<u32>)> {
    let lo = i.saturating_sub(12);
    for j in (lo..i).rev() {
        let ins = &insns[j];
        if let Some(r) = &ins.reloc {
            if matches!(r.kind, RelocKind::Addr16Lo | RelocKind::Addr16Ha) {
                let d = obj.data.get(&r.target)?;
                let mut targets = Vec::new();
                let n = d.size / 4;
                for k in 0..n {
                    let rr = d.relocs.iter().find(|x| x.offset == k * 4)?;
                    let off = if rr.target == f.name {
                        rr.addend as u32
                    } else if let Some(g) = obj.functions.iter().find(|g| g.name == rr.target) {
                        // label symbol inside .text: convert via section addresses
                        (g.address as i64 + rr.addend - f.address as i64) as u32
                    } else {
                        return None;
                    };
                    targets.push(off);
                }
                return Some((r.target.clone(), targets));
            }
        }
    }
    None
}

impl Cfg {
    pub fn exit(&self) -> usize {
        self.blocks.len()
    }

    pub fn build(obj: &ObjectFile, f: &Function, insns: &[Insn]) -> Cfg {
        let n = insns.len();
        let mut leaders = BTreeSet::new();
        leaders.insert(0usize);
        let mut tables: HashMap<usize, (String, Vec<u32>)> = HashMap::new();
        for (i, ins) in insns.iter().enumerate() {
            let is_branch = ins.is_jump() || ins.is_cond_branch() || ins.is_blr() || ins.is_bctr() || ins.is_cond_blr();
            if ins.is_jump() && ins.reloc.is_none() || ins.is_cond_branch() && ins.reloc.is_none() {
                if let Some(t) = ins.target() {
                    if (t as usize) / 4 < n {
                        leaders.insert(t as usize / 4);
                    }
                }
            }
            if ins.is_bctr() {
                if let Some(tb) = jump_table(obj, f, insns, i) {
                    for &t in &tb.1 {
                        if (t as usize) / 4 < n {
                            leaders.insert(t as usize / 4);
                        }
                    }
                    tables.insert(i, tb);
                }
            }
            if is_branch && i + 1 < n {
                leaders.insert(i + 1);
            }
        }
        let starts: Vec<usize> = leaders.into_iter().collect();
        let mut block_of = vec![0usize; n];
        let mut blocks = Vec::new();
        for (bi, &s) in starts.iter().enumerate() {
            let e = starts.get(bi + 1).copied().unwrap_or(n);
            for k in s..e {
                block_of[k] = bi;
            }
            blocks.push(Block { start: s, end: e, term: Term::Stop, succs: vec![], preds: vec![] });
        }
        let nb = blocks.len();
        for bi in 0..nb {
            let last = blocks[bi].end - 1;
            let ins = &insns[last];
            let next = if bi + 1 < nb { Some(bi + 1) } else { None };
            let tgt = |o: u32| block_of.get(o as usize / 4).copied();
            let term = if ins.is_blr() {
                Term::Return
            } else if ins.is_cond_blr() {
                match next {
                    Some(nx) => Term::CondReturn { fall: nx },
                    None => Term::Return,
                }
            } else if ins.is_jump() {
                if ins.reloc.is_some() {
                    Term::TailCall
                } else {
                    match ins.target().and_then(tgt) {
                        Some(t) => Term::Jump(t),
                        None => Term::Stop,
                    }
                }
            } else if ins.is_cond_branch() && ins.reloc.is_none() {
                match (ins.target().and_then(tgt), next) {
                    (Some(t), Some(nx)) => Term::Cond { taken: t, fall: nx },
                    (Some(t), None) => Term::Jump(t),
                    _ => Term::Stop,
                }
            } else if ins.is_bctr() {
                match tables.get(&last) {
                    Some((name, ts)) => Term::Switch {
                        targets: ts.iter().map(|&t| block_of.get(t as usize / 4).copied().unwrap_or(0)).collect(),
                        table: name.clone(),
                    },
                    None => Term::Stop,
                }
            } else {
                match next {
                    Some(nx) => Term::Fall(nx),
                    None => Term::Stop,
                }
            };
            let succs: Vec<usize> = match &term {
                Term::Fall(t) | Term::Jump(t) => vec![*t],
                Term::Cond { taken, fall } => {
                    if taken == fall {
                        vec![*taken]
                    } else {
                        vec![*taken, *fall]
                    }
                }
                Term::CondReturn { fall } => vec![*fall],
                Term::Switch { targets, .. } => {
                    let mut v: Vec<usize> = Vec::new();
                    for t in targets {
                        if !v.contains(t) {
                            v.push(*t);
                        }
                    }
                    v
                }
                _ => vec![],
            };
            blocks[bi].term = term;
            blocks[bi].succs = succs;
        }
        for bi in 0..nb {
            for s in blocks[bi].succs.clone() {
                blocks[s].preds.push(bi);
            }
        }
        let mut cfg = Cfg { blocks, block_of, rpo: vec![], idom: vec![], ipdom: vec![], pd_extra: vec![] };
        cfg.compute_orders();
        cfg
    }

    /// Turn block `b`'s jump into a return (an early `return` the source wrote as such), and
    /// recompute the orders and (post-)dominators.
    pub fn make_return(&mut self, b: usize, continue_at: usize) {
        for s in std::mem::take(&mut self.blocks[b].succs) {
            self.blocks[s].preds.retain(|&p| p != b);
        }
        self.blocks[b].term = Term::Return;
        self.pd_extra.push((b, continue_at));
        self.compute_orders();
    }

    /// Turn block `b`'s conditional branch into a conditional return (`if (c) return;` branching
    /// straight to the epilogue), and recompute the orders and (post-)dominators.
    pub fn make_cond_return(&mut self, b: usize) {
        if let Term::Cond { taken, fall } = self.blocks[b].term {
            if taken == fall {
                return;
            }
            self.blocks[b].succs.retain(|&s| s != taken);
            self.blocks[taken].preds.retain(|&p| p != b);
            self.blocks[b].term = Term::CondReturn { fall };
            self.compute_orders();
        }
    }

    /// Turn block `b`'s jump into a plain return (the function's last statement).
    pub fn make_return_plain(&mut self, b: usize) {
        for s in std::mem::take(&mut self.blocks[b].succs) {
            self.blocks[s].preds.retain(|&p| p != b);
        }
        self.blocks[b].term = Term::Return;
        self.compute_orders();
    }

    fn compute_orders(&mut self) {
        let nb = self.blocks.len();
        // RPO by iterative DFS
        let mut visited = vec![false; nb];
        let mut post = Vec::new();
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        visited[0] = true;
        while let Some(&mut (b, ref mut k)) = stack.last_mut() {
            if *k < self.blocks[b].succs.len() {
                let s = self.blocks[b].succs[*k];
                *k += 1;
                if !visited[s] {
                    visited[s] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        post.reverse();
        self.rpo = post;
        // dominators (Cooper-Harvey-Kennedy)
        let mut order = vec![usize::MAX; nb];
        for (i, &b) in self.rpo.iter().enumerate() {
            order[b] = i;
        }
        let mut idom = vec![usize::MAX; nb];
        idom[0] = 0;
        let mut changed = true;
        let mut fuel = crate::fuel::Fuel::new("cfg.dominators", crate::fuel::CAP_FIXPOINT);
        while changed && fuel.burn() {
            changed = false;
            for &b in self.rpo.iter().skip(1) {
                let mut new: Option<usize> = None;
                for &p in &self.blocks[b].preds {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(q) => intersect(&idom, &order, p, q),
                    });
                }
                if let Some(nd) = new {
                    if idom[b] != nd {
                        idom[b] = nd;
                        changed = true;
                    }
                }
            }
        }
        self.idom = idom;
        // post-dominators on the reversed graph with a virtual exit (index nb)
        let exit = nb;
        let mut rsuccs: Vec<Vec<usize>> = vec![vec![]; nb + 1]; // reversed edges: node -> its preds in reverse graph = succs
        let mut rpreds: Vec<Vec<usize>> = vec![vec![]; nb + 1];
        for &(b, s) in &self.pd_extra {
            rsuccs[s].push(b);
            rpreds[b].push(s);
        }
        for b in 0..nb {
            if self.blocks[b].succs.is_empty() && !self.pd_extra.iter().any(|e| e.0 == b) {
                rsuccs[exit].push(b);
                rpreds[b].push(exit);
            }
            for &s in &self.blocks[b].succs {
                rsuccs[s].push(b);
                rpreds[b].push(s);
            }
        }
        // infinite loops: blocks that can't reach exit get an edge to exit from their last block
        let mut reach = vec![false; nb + 1];
        let mut st = vec![exit];
        reach[exit] = true;
        while let Some(x) = st.pop() {
            for &y in &rsuccs[x] {
                if !reach[y] {
                    reach[y] = true;
                    st.push(y);
                }
            }
        }
        for b in 0..nb {
            if !reach[b] && visited[b] {
                rsuccs[exit].push(b);
                rpreds[b].push(exit);
            }
        }
        // RPO of reverse graph from exit
        let mut vis = vec![false; nb + 1];
        let mut post = Vec::new();
        let mut stack: Vec<(usize, usize)> = vec![(exit, 0)];
        vis[exit] = true;
        while let Some(&mut (b, ref mut k)) = stack.last_mut() {
            if *k < rsuccs[b].len() {
                let s = rsuccs[b][*k];
                *k += 1;
                if !vis[s] {
                    vis[s] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        post.reverse();
        let mut order = vec![usize::MAX; nb + 1];
        for (i, &b) in post.iter().enumerate() {
            order[b] = i;
        }
        let mut ipdom = vec![usize::MAX; nb + 1];
        ipdom[exit] = exit;
        let mut changed = true;
        let mut fuel = crate::fuel::Fuel::new("cfg.postdominators", crate::fuel::CAP_FIXPOINT);
        while changed && fuel.burn() {
            changed = false;
            for &b in post.iter().skip(1) {
                let mut new: Option<usize> = None;
                for &p in &rpreds[b] {
                    if ipdom[p] == usize::MAX {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(q) => intersect(&ipdom, &order, p, q),
                    });
                }
                if let Some(nd) = new {
                    if ipdom[b] != nd {
                        ipdom[b] = nd;
                        changed = true;
                    }
                }
            }
        }
        self.ipdom = ipdom;
    }

    pub fn dominates(&self, a: usize, mut b: usize) -> bool {
        let mut fuel = crate::fuel::Fuel::new("cfg.dominates", self.idom.len() + 1);
        loop {
            if !fuel.burn() {
                return false;
            }
            if a == b {
                return true;
            }
            let d = self.idom[b];
            if d == b || d == usize::MAX {
                return false;
            }
            b = d;
        }
    }

    pub fn postdominates(&self, a: usize, mut b: usize) -> bool {
        let mut fuel = crate::fuel::Fuel::new("cfg.postdominates", self.ipdom.len() + 1);
        loop {
            if !fuel.burn() {
                return false;
            }
            if a == b {
                return true;
            }
            let d = self.ipdom[b];
            if d == b || d == usize::MAX {
                return false;
            }
            b = d;
        }
    }

    /// Natural loops: header -> set of body blocks (incl. header), and the back-edge sources.
    pub fn loops(&self) -> Vec<Loop> {
        let mut out: Vec<Loop> = Vec::new();
        for b in 0..self.blocks.len() {
            for &s in &self.blocks[b].succs {
                if self.idom[b] != usize::MAX && self.dominates(s, b) {
                    // back edge b -> s
                    let mut body = BTreeSet::new();
                    body.insert(s);
                    let mut st = vec![b];
                    while let Some(x) = st.pop() {
                        if body.insert(x) {
                            for &p in &self.blocks[x].preds {
                                st.push(p);
                            }
                        }
                    }
                    if let Some(l) = out.iter_mut().find(|l| l.header == s) {
                        l.body.extend(body);
                        l.latches.push(b);
                    } else {
                        out.push(Loop { header: s, body, latches: vec![b] });
                    }
                }
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct Loop {
    pub header: usize,
    pub body: BTreeSet<usize>,
    pub latches: Vec<usize>,
}

fn intersect(idom: &[usize], order: &[usize], mut a: usize, mut b: usize) -> usize {
    let mut fuel = crate::fuel::Fuel::new("cfg.intersect", 3 * idom.len() + 3);
    while a != b {
        if !fuel.burn() {
            return a;
        }
        while order[a] > order[b] {
            if !fuel.burn() {
                return a;
            }
            a = idom[a];
        }
        while order[b] > order[a] {
            if !fuel.burn() {
                return a;
            }
            b = idom[b];
        }
    }
    a
}

/// True for opcodes that end a block.
pub fn is_terminator(op: Opcode) -> bool {
    matches!(op, Opcode::B | Opcode::Bc | Opcode::Bcctr | Opcode::Bclr)
}
