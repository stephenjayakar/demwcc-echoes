//! CFG -> structured statements: if/else with &&/|| chains (m2c's reduction), while/do-while/
//! infinite loops with break/continue, jump-table switches, gotos as a fallback.

use crate::cfg::{Cfg, Loop, Term};
use crate::insn::Insn;
use crate::ir::*;
use crate::switchtree::{self, Lab, TItem};
use crate::translate::BlockOut;
use std::collections::{BTreeSet, HashMap, HashSet};

struct LoopCtx {
    header: usize,
    exit: Option<usize>,
    body: BTreeSet<usize>,
}

pub struct Structurer<'a> {
    cfg: &'a Cfg,
    blocks: &'a mut Vec<BlockOut>,
    vars: &'a [Var],
    emitted: Vec<bool>,
    loops: HashMap<usize, Loop>,
    stack: Vec<LoopCtx>,
    pub gotos: HashSet<usize>,
    in_loop_build: HashSet<usize>,
    ret_void: bool,
    insns: &'a [Insn],
}

enum TreeCheck {
    /// MWCC builds this tree from the case set plus these empty-body case groups.
    Match(Vec<(Vec<i64>, usize)>),
    NoMatch,
    /// the tree can't be simulated (register compares, unsigned narrow selectors, ...)
    Unknown,
}

/// Case targets at or above this are empty-body case groups found by the tree simulation
/// (`case k: break;`), numbered from here.
const EXTRA_CASE: usize = usize::MAX / 2;

impl<'a> Structurer<'a> {
    pub fn new(cfg: &'a Cfg, blocks: &'a mut Vec<BlockOut>, vars: &'a [Var], ret_void: bool) -> Self {
        let loops = cfg.loops().into_iter().map(|l| (l.header, l)).collect();
        let n = cfg.blocks.len();
        Structurer {
            cfg,
            blocks,
            vars,
            emitted: vec![false; n],
            loops,
            stack: vec![],
            gotos: HashSet::new(),
            in_loop_build: HashSet::new(),
            ret_void,
            insns: &[],
        }
    }

    /// Target instructions, for checking switch trees against MWCC's tree builder.
    pub fn with_insns(mut self, insns: &'a [Insn]) -> Self {
        self.insns = insns;
        self
    }

    pub fn run(mut self) -> Vec<Stmt> {
        let mut out = vec![];
        self.build(0, None, false, &mut out);
        // unreachable-from-structure blocks that are goto targets but never emitted: emit after
        let mut pending: Vec<usize> = (0..self.cfg.blocks.len()).filter(|&b| !self.emitted[b] && self.cfg.idom[b] != usize::MAX).collect();
        while let Some(b) = pending.pop() {
            if !self.emitted[b] {
                self.build(b, None, false, &mut out);
            }
        }
        // remove labels that are never targeted
        let gotos = self.gotos.clone();
        Stmt::for_each_block_mut(&mut out, &mut |blk| blk.retain(|s| !matches!(s, Stmt::Label(l) if !gotos.contains(l))));
        out
    }

    fn exit_node(&self) -> usize {
        self.cfg.exit()
    }

    fn cond_of(&self, b: usize) -> Expr {
        self.blocks[b].cond.clone().unwrap_or(Expr::Unknown { text: "cond".into(), ty: mwdec_core::Type::Bool })
    }

    fn has_stmts(&self, b: usize) -> bool {
        self.blocks[b].stmts.iter().any(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_)))
    }

    fn cond_edges(&self, b: usize) -> Option<(usize, usize)> {
        match self.cfg.blocks[b].term {
            Term::Cond { taken, fall } if taken != fall => Some((taken, fall)),
            _ => None,
        }
    }

    fn build(&mut self, start: usize, end: Option<usize>, first_ok: bool, out: &mut Vec<Stmt>) {
        let mut cur = start;
        let mut first = true;
        loop {
            let skip_checks = first && first_ok;
            first = false;
            if cur == self.exit_node() {
                return;
            }
            if !skip_checks {
                if Some(cur) == end {
                    return;
                }
                if let Some(ctx) = self.stack.iter().rev().find(|c| c.header != usize::MAX) {
                    if cur == ctx.header {
                        out.push(Stmt::Continue);
                        return;
                    }
                }
                if let Some(ctx) = self.stack.last() {
                    if Some(cur) == ctx.exit {
                        out.push(Stmt::Break);
                        return;
                    }
                }
            }
            if self.emitted[cur] {
                // an already-emitted block that only returns can be duplicated instead of goto
                if let Some(r) = self.small_return(cur) {
                    out.extend(r);
                    return;
                }
                // a small shared tail that flows straight into this region's end (e.g. the
                // `result = 0;` of several failed checks): duplicate it instead of a goto
                let n = self.blocks[cur].stmts.iter().filter(|s| !matches!(s, Stmt::Label(_))).count();
                let next = match self.cfg.blocks[cur].term {
                    Term::Fall(t) | Term::Jump(t) => Some(t),
                    _ => None,
                };
                if n <= 2 && !self.blocks[cur].stmts.iter().any(|s| matches!(s, Stmt::Expr(_))) {
                    if let Some(t) = next {
                        if Some(t) == end || t == self.exit_node() {
                            out.extend(self.blocks[cur].stmts.iter().filter(|s| !matches!(s, Stmt::Label(_))).cloned());
                            return;
                        }
                        if let Some(r) = self.small_return(t) {
                            out.extend(self.blocks[cur].stmts.iter().filter(|s| !matches!(s, Stmt::Label(_))).cloned());
                            out.extend(r);
                            return;
                        }
                    }
                }
                out.push(Stmt::Goto(cur));
                self.gotos.insert(cur);
                return;
            }
            if self.loops.contains_key(&cur) && !self.in_loop_build.contains(&cur) {
                let (stmt, exit) = self.build_loop(cur);
                out.push(stmt);
                match exit {
                    Some(x) => {
                        cur = x;
                        continue;
                    }
                    None => return,
                }
            }
            self.emitted[cur] = true;
            out.push(Stmt::Label(cur));
            out.extend(self.blocks[cur].stmts.clone());
            match self.cfg.blocks[cur].term.clone() {
                Term::Return => {
                    let r = self.blocks[cur].ret.clone();
                    out.push(Stmt::Return(r));
                    return;
                }
                Term::TailCall => {
                    match self.blocks[cur].ret.clone() {
                        Some(e) if !self.ret_void => out.push(Stmt::Return(Some(e))),
                        Some(e) => {
                            out.push(Stmt::Expr(e));
                            out.push(Stmt::Return(None));
                        }
                        None => out.push(Stmt::Return(None)),
                    }
                    return;
                }
                Term::Stop => return,
                Term::Fall(t) | Term::Jump(t) => cur = t,
                Term::Cond { taken, fall } if taken == fall => {
                    // a test whose branch goes nowhere: the source had an empty `if`
                    if let Some(c) = self.blocks[cur].cond.clone() {
                        out.push(Stmt::If { cond: c, then: vec![], els: vec![] });
                    }
                    cur = taken
                }
                Term::Cond { .. } => {
                    let join = self.join_of(cur);
                    if let Some((e, cases, default, nodes)) = self.case_tree(cur).or_else(|| self.guarded_table(cur)) {
                        for n in &nodes {
                            self.emitted[*n] = true;
                        }
                        let j = self.join_of_set(&nodes).or(join);
                        let stmt = self.build_cases(e, cases, default, j);
                        out.push(stmt);
                        match j {
                            Some(j) => {
                                cur = j;
                                continue;
                            }
                            None => return,
                        }
                    }
                    self.build_if(cur, join, out);
                    match join {
                        Some(j) => cur = j,
                        None => return,
                    }
                }
                Term::CondReturn { fall } => {
                    let c = self.cond_of(cur);
                    let r = self.blocks[cur].ret.clone();
                    out.push(Stmt::If { cond: c, then: vec![Stmt::Return(r)], els: vec![] });
                    cur = fall;
                }
                Term::Switch { targets, .. } => {
                    let join = self.join_of(cur);
                    let stmt = self.build_switch(cur, &targets, join);
                    out.push(stmt);
                    match join {
                        Some(j) => cur = j,
                        None => return,
                    }
                }
            }
        }
    }

    /// Does every path from `b` return within a few straight-line blocks (`ctor(); return;`)?
    fn only_returns(&self, b: usize, body: &BTreeSet<usize>) -> bool {
        let mut cur = b;
        for _ in 0..4 {
            if body.contains(&cur) {
                return false;
            }
            match self.cfg.blocks[cur].term {
                Term::Return | Term::TailCall => return true,
                Term::Fall(t) | Term::Jump(t) => {
                    if t == self.exit_node() {
                        return true;
                    }
                    cur = t;
                }
                _ => return false,
            }
        }
        false
    }

    /// A block that is just `return x;` (possibly with an epilogue) can be duplicated.
    fn small_return(&self, b: usize) -> Option<Vec<Stmt>> {
        if matches!(self.cfg.blocks[b].term, Term::Return) && !self.has_stmts(b) {
            return Some(vec![Stmt::Return(self.blocks[b].ret.clone())]);
        }
        None
    }

    /// Join point for a branching block: immediate post-dominator, clamped to the current loop.
    fn join_of(&self, b: usize) -> Option<usize> {
        let mut j = self.cfg.ipdom[b];
        if j == usize::MAX || j == self.exit_node() {
            return None;
        }
        if let Some(ctx) = self.stack.iter().rev().find(|c| c.header != usize::MAX) {
            if !ctx.body.contains(&j) && Some(j) != ctx.exit {
                j = ctx.header;
            }
        }
        Some(j)
    }

    fn build_if(&mut self, start: usize, join: Option<usize>, out: &mut Vec<Stmt>) {
        let mut chain = vec![start];
        let mut cur = start;
        loop {
            let Some((_, fall)) = self.cond_edges(cur) else { break };
            let n = fall;
            let ok = self.cond_edges(n).is_some()
                && Some(n) != join
                && n != self.exit_node()
                && !self.emitted[n]
                && !self.loops.contains_key(&n)
                && self.cfg.blocks[n].preds.iter().all(|p| chain.contains(p))
                && !self.has_stmts(n)
                && join.map_or(true, |j| self.cfg.postdominates(j, n))
                && self.stack.last().map_or(true, |c| n != c.header && Some(n) != c.exit);
            if !ok {
                break;
            }
            chain.push(n);
            cur = n;
        }
        let (cond, if_node, else_node) = loop {
            if let Some(r) = self.try_make_cond(&chain, join) {
                break r;
            }
            chain.pop();
            if chain.is_empty() {
                unreachable!()
            }
        };
        for &c in &chain[1..] {
            self.emitted[c] = true;
        }
        let mut then = vec![];
        let mut els = vec![];
        if let Some(e) = else_node {
            self.build(e, join, false, &mut els);
        }
        self.build(if_node, join, false, &mut then);
        // a branch over an empty arm that only jumps (`b join`) plus a single assignment on the
        // other arm is MWCC's ternary layout: `v = c ? v : e`
        let empty_jump = |b: usize, me: &Self| !me.has_stmts(b) && matches!(me.cfg.blocks[b].term, Term::Jump(_));
        let strip_labels = |v: &Vec<Stmt>| v.iter().filter(|s| !matches!(s, Stmt::Label(_))).cloned().collect::<Vec<_>>();
        let (t2, e2) = (strip_labels(&then), strip_labels(&els));
        if t2.is_empty() && empty_jump(if_node, self) {
            if let Some((v, src)) = ternary_assign(&e2) {
                let ty = self.vars[v].ty.clone();
                out.push(Stmt::Assign {
                    dst: Expr::Var(v),
                    src: Expr::Ternary { c: Box::new(cond), t: Box::new(Expr::Var(v)), f: Box::new(src), ty },
                });
                return;
            }
        }
        if let Some(e) = else_node {
            if e2.is_empty() && empty_jump(e, self) {
                if let Some((v, src)) = ternary_assign(&t2).map(|(v, s)| (v, s)) {
                    let src = &src;
                    let v = &v;
                    let ty = self.vars[*v].ty.clone();
                    out.push(Stmt::Assign {
                        dst: Expr::Var(*v),
                        src: Expr::Ternary { c: Box::new(cond), t: Box::new(src.clone()), f: Box::new(Expr::Var(*v)), ty },
                    });
                    return;
                }
            }
        }
        // `if (c) {} else {...}` -> `if (!c) {...}`
        if then.is_empty() && !els.is_empty() {
            out.push(Stmt::If { cond: cond.negate(self.vars), then: els, els: vec![] });
        } else if then.is_empty() && els.is_empty() {
            // condition with no effect: keep calls in it
            if cond.has_call() {
                out.push(Stmt::If { cond, then: vec![], els: vec![] });
            }
        } else {
            out.push(Stmt::If { cond, then, els });
        }
    }

    /// m2c's T2-style reduction of a chain of conditional nodes into one condition.
    /// Returns (cond for entering if_node, if_node, else_node or None when else == join).
    fn try_make_cond(&self, chain: &[usize], join: Option<usize>) -> Option<(Expr, usize, Option<usize>)> {
        let last = *chain.last().unwrap();
        let (else_node, if_node) = self.cond_edges(last)?; // taken = else, fallthrough = if
        let mut allowed: HashSet<usize> = chain.iter().copied().collect();
        allowed.insert(if_node);
        allowed.insert(else_node);
        // node -> (cond (taken), taken, fall)
        let mut edges: Vec<(usize, Expr, usize, usize)> = vec![];
        for &n in chain {
            let (t, f) = self.cond_edges(n)?;
            if !allowed.contains(&t) || !allowed.contains(&f) {
                return None;
            }
            allowed.remove(&n);
            edges.push((n, self.cond_of(n), t, f));
        }
        loop {
            let mut did = false;
            let ids: Vec<usize> = edges.iter().map(|e| e.0).collect();
            for ci in 0..edges.len() {
                let child = edges[ci].0;
                let parents: Vec<usize> = (0..edges.len()).filter(|&p| edges[p].2 == child || edges[p].3 == child).collect();
                if parents.len() != 1 {
                    continue;
                }
                let pi = parents[0];
                let (_, ref pc, pt, pf) = edges[pi];
                let (_, ref cc, ct, cf) = edges[ci];
                let _ = &ids;
                let (nc, nt, nf) = if pt == ct && pf == child {
                    (Expr::cmp(BinOp::LogOr, pc.clone(), cc.clone()), pt, cf)
                } else if pt == cf && pf == child {
                    (Expr::cmp(BinOp::LogOr, pc.clone(), cc.clone().negate(self.vars)), pt, ct)
                } else if pt == child && pf == ct {
                    (Expr::cmp(BinOp::LogAnd, pc.clone(), cc.clone().negate(self.vars)), cf, pf)
                } else if pt == child && pf == cf {
                    (Expr::cmp(BinOp::LogAnd, pc.clone(), cc.clone()), ct, pf)
                } else {
                    continue;
                };
                edges[pi].1 = nc;
                edges[pi].2 = nt;
                edges[pi].3 = nf;
                edges.remove(ci);
                did = true;
                break;
            }
            if !did {
                break;
            }
        }
        if edges.len() != 1 || edges[0].0 != chain[0] {
            return None;
        }
        let (_, c, t, f) = edges.remove(0);
        // c is the condition for going to t
        let cond = if (t, f) == (if_node, else_node) {
            c
        } else if (t, f) == (else_node, if_node) {
            c.negate(self.vars)
        } else {
            return None;
        };
        if Some(else_node) == join {
            Some((cond, if_node, None))
        } else if Some(if_node) == join {
            Some((cond.negate(self.vars), else_node, None))
        } else {
            Some((cond, if_node, Some(else_node)))
        }
    }

    fn build_loop(&mut self, h: usize) -> (Stmt, Option<usize>) {
        let l = self.loops[&h].clone();
        // exit: first node outside the loop on the ipdom chain from h
        let mut x = h;
        let mut guard = 0;
        while l.body.contains(&x) && guard < 10000 {
            x = self.cfg.ipdom[x];
            guard += 1;
            if x == usize::MAX {
                break;
            }
        }
        let mut exit = if x == usize::MAX || x == self.exit_node() { None } else { Some(x) };
        // a loop whose body returns (`for (...) { if (c) return x; }`) post-dominates nothing
        // but the return: its exit is where the loop test falls out, when every other way out
        // only returns
        let test_exit = |b: usize, me: &Self| -> Option<usize> {
            let (t, f) = me.cond_edges(b)?;
            match (l.body.contains(&t), l.body.contains(&f)) {
                (true, false) => Some(f),
                (false, true) => Some(t),
                _ => None,
            }
        };
        if let Some(c) = test_exit(h, self).or_else(|| l.latches.iter().find_map(|&lt| test_exit(lt, self))) {
            let exit_returns = exit.map_or(true, |x| self.only_returns(x, &l.body));
            if Some(c) != exit && exit_returns && c != self.exit_node() {
                let others_return = l.body.iter().all(|&b| self.cfg.blocks[b].succs.iter().all(|&s| l.body.contains(&s) || s == c || self.only_returns(s, &l.body)));
                if others_return {
                    exit = Some(c);
                }
            }
        }
        self.in_loop_build.insert(h);
        self.stack.push(LoopCtx { header: h, exit, body: l.body.clone() });
        let stmt;
        // (a2) while with an && chain: header tests that each leave the loop or go on to the
        // next test, the last one entering the body (`while (p && p->x != k)`)
        let a2 = (|| {
            let x = exit?;
            if self.has_stmts(h) {
                return None;
            }
            let mut chain: Vec<usize> = vec![];
            let mut conds: Vec<Expr> = vec![];
            let mut cur = h;
            loop {
                let (t, f) = self.cond_edges(cur)?;
                let (stay, next) = if t == x && l.body.contains(&f) {
                    (self.cond_of(cur).negate(self.vars), f)
                } else if f == x && l.body.contains(&t) {
                    (self.cond_of(cur), t)
                } else {
                    return None;
                };
                chain.push(cur);
                conds.push(stay);
                let more = next != h
                    && self.cond_edges(next).map_or(false, |(t2, f2)| t2 == x || f2 == x)
                    && !self.has_stmts(next)
                    && self.cfg.blocks[next].preds.iter().all(|p| chain.contains(p))
                    && !self.loops.contains_key(&next);
                if !more {
                    if chain.len() < 2 {
                        return None;
                    }
                    for &c in &chain[1..] {
                        self.emitted[c] = true;
                    }
                    let cond = conds.into_iter().reduce(|a, b| Expr::cmp(BinOp::LogAnd, a, b))?;
                    return Some((cond, next));
                }
                cur = next;
            }
        })();
        // (a) while: header has only the condition, one edge into the body and one out
        let a = a2.or_else(|| self.cond_edges(h).and_then(|(t, f)| {
            if self.has_stmts(h) {
                return None;
            }
            if l.body.contains(&t) && !l.body.contains(&f) && Some(f) == exit {
                Some((self.cond_of(h), t))
            } else if l.body.contains(&f) && !l.body.contains(&t) && Some(t) == exit {
                Some((self.cond_of(h).negate(self.vars), f))
            } else {
                None
            }
        }));
        // (b) do-while: a latch with a conditional back edge whose other edge leaves the loop
        let b = l.latches.iter().copied().find_map(|lt| {
            let (t, f) = self.cond_edges(lt)?;
            if !self.cfg.postdominates(lt, h) && lt != h {
                return None;
            }
            if t == h && Some(f) == exit {
                Some((lt, self.cond_of(lt)))
            } else if f == h && Some(t) == exit {
                Some((lt, self.cond_of(lt).negate(self.vars)))
            } else {
                None
            }
        });
        if let Some((cond, body_start)) = a {
            self.emitted[h] = true;
            let mut body = vec![Stmt::Label(h)];
            self.build(body_start, Some(h), false, &mut body);
            strip_trailing_continue(&mut body);
            stmt = Stmt::While { cond, body };
        } else if let Some((lt, cond)) = b {
            let mut body = vec![];
            if lt == h {
                self.emitted[h] = true;
                body.push(Stmt::Label(h));
                body.extend(self.blocks[h].stmts.clone());
            } else {
                self.build(h, Some(lt), true, &mut body);
                self.emitted[lt] = true;
                body.push(Stmt::Label(lt));
                body.extend(self.blocks[lt].stmts.clone());
            }
            stmt = Stmt::DoWhile { body, cond };
        } else {
            let mut body = vec![];
            self.build(h, Some(h), true, &mut body);
            strip_trailing_continue(&mut body);
            stmt = Stmt::While { cond: Expr::Int { value: 1, ty: mwdec_core::Type::Bool }, body };
        }
        self.stack.pop();
        self.in_loop_build.remove(&h);
        (stmt, exit)
    }

    fn build_switch(&mut self, b: usize, targets: &[usize], join: Option<usize>) -> Stmt {
        let (e, base) = switch_selector(self.blocks[b].switch.clone().unwrap_or(Expr::Unknown { text: "switch index".into(), ty: t_s32() }));
        let mut cases: Vec<(Vec<i64>, usize)> = vec![];
        for (i, &t) in targets.iter().enumerate() {
            match cases.iter_mut().find(|(_, x)| *x == t) {
                Some(c) => c.0.push(i as i64 + base),
                None => cases.push((vec![i as i64 + base], t)),
            }
        }
        self.build_cases(e, cases, None, join)
    }

    /// Emit a switch: case bodies in address order (MWCC lays bodies out in source order).
    fn build_cases(&mut self, e: Expr, mut cases: Vec<(Vec<i64>, usize)>, default: Option<usize>, join: Option<usize>) -> Stmt {
        let mut order: Vec<usize> = cases.iter().map(|c| c.1).chain(default).collect();
        order.sort_by_key(|&t| if t < self.cfg.blocks.len() { self.cfg.blocks[t].start } else { usize::MAX });
        order.dedup();
        let mut out = vec![];
        // switch context: reaching the join means break
        self.stack.push(LoopCtx { header: usize::MAX, exit: join, body: BTreeSet::new() });
        for t in order {
            let values: Vec<i64> = cases.iter_mut().filter(|c| c.1 == t).flat_map(|c| std::mem::take(&mut c.0)).collect();
            let is_default = default == Some(t);
            if values.is_empty() && !is_default {
                continue;
            }
            let mut body = vec![];
            if Some(t) == join || t >= EXTRA_CASE {
                body.push(Stmt::Break);
            } else {
                self.build(t, join, false, &mut body);
                if !matches!(body.last(), Some(Stmt::Return(_) | Stmt::Break | Stmt::Goto(_) | Stmt::Continue)) && join.is_some() {
                    body.push(Stmt::Break);
                }
            }
            out.push(SwitchCase { values, is_default, body });
        }
        self.stack.pop();
        // a default that only breaks is implicit
        out.retain(|c| !(c.is_default && c.values.is_empty() && c.body == vec![Stmt::Break]));
        Stmt::Switch { e, cases: out }
    }

    /// Common post-dominator of a set of blocks (the tree's exit).
    fn join_of_set(&self, nodes: &[usize]) -> Option<usize> {
        let mut j = *nodes.first()?;
        loop {
            if nodes.iter().all(|&n| self.cfg.postdominates(j, n)) && !nodes.contains(&j) {
                break;
            }
            let p = self.cfg.ipdom[j];
            if p == usize::MAX || p == j {
                return None;
            }
            j = p;
        }
        if j == self.exit_node() {
            return None;
        }
        if let Some(ctx) = self.stack.iter().rev().find(|c| c.header != usize::MAX) {
            if !ctx.body.contains(&j) && Some(j) != ctx.exit {
                return Some(ctx.header);
            }
        }
        Some(j)
    }

    /// `cmplwi x, span; bgt default` guarding a jump-table block: one switch with a default.
    fn guarded_table(&self, b: usize) -> Option<(Expr, Vec<(Vec<i64>, usize)>, Option<usize>, Vec<usize>)> {
        let (taken, fall) = self.cond_edges(b)?;
        let sw = fall;
        let Term::Switch { targets, .. } = &self.cfg.blocks[sw].term else { return None };
        if self.cfg.blocks[sw].preds.len() != 1 || self.has_stmts(sw) || self.emitted[sw] {
            return None;
        }
        let (e, base) = switch_selector(self.blocks[sw].switch.clone()?);
        let mut cases: Vec<(Vec<i64>, usize)> = vec![];
        for (i, &t) in targets.iter().enumerate() {
            if t == taken {
                continue;
            }
            match cases.iter_mut().find(|(_, x)| *x == t) {
                Some(c) => c.0.push(i as i64 + base),
                None => cases.push((vec![i as i64 + base], t)),
            }
        }
        Some((e, cases, Some(taken), vec![b, sw]))
    }

    /// A binary-search tree of `cmpwi x, K` blocks: rebuild the value -> target map.
    fn case_tree(&self, root: usize) -> Option<(Expr, Vec<(Vec<i64>, usize)>, Option<usize>, Vec<usize>)> {
        let sel = |b: usize| -> Option<(Expr, BinOp, i64)> {
            // statements before the root's compare are emitted before the switch
            if b != root && self.has_stmts(b) {
                return None;
            }
            let c = self.blocks[b].cond.as_ref()?;
            let mut c = c.clone();
            let mut neg = false;
            if let Expr::Unary { op: UnOp::Not, e, .. } = c {
                c = *e;
                neg = true;
            }
            if let Expr::Binary { op, l, r, .. } = c {
                let k = r.as_int()?;
                if !op.is_cmp() {
                    return None;
                }
                let op = if neg { op.negate_cmp()? } else { op };
                Some((strip_casts(*l), op, k))
            } else {
                None
            }
        };
        let (x, _, _) = sel(root)?;
        let mut nodes = vec![];
        let mut leaves: Vec<(i64, i64, usize)> = vec![];
        let mut work: Vec<(usize, Vec<(i64, i64)>)> = vec![(root, vec![(i32::MIN as i64, i32::MAX as i64)])];
        let mut guard = 0;
        while let Some((b, set)) = work.pop() {
            guard += 1;
            if guard > 64 {
                return None;
            }
            let is_node = b == root
                || (sel(b).map_or(false, |(y, _, _)| y == x) && self.cfg.blocks[b].preds.iter().all(|p| nodes.contains(p) || *p == root) && !self.emitted[b] && !self.loops.contains_key(&b));
            // plain `b target` blocks inside the tree forward their set
            let is_jump = b != root && !self.has_stmts(b) && matches!(self.cfg.blocks[b].term, Term::Jump(_)) && self.cfg.blocks[b].preds.len() == 1 && nodes.contains(&self.cfg.blocks[b].preds[0]);
            if is_jump {
                if let Term::Jump(t) = self.cfg.blocks[b].term {
                    nodes.push(b);
                    work.push((t, set));
                    continue;
                }
            }
            if !is_node {
                for (lo, hi) in set {
                    leaves.push((lo, hi, b));
                }
                continue;
            }
            nodes.push(b);
            let (_, op, k) = sel(b)?;
            let (taken, fall) = self.cond_edges(b)?;
            let (mut ts, mut fs) = (vec![], vec![]);
            for (lo, hi) in set {
                for v in [(lo, hi)] {
                    let (a, z) = v;
                    let split = |pred: &dyn Fn(i64) -> bool, ts: &mut Vec<(i64, i64)>, fs: &mut Vec<(i64, i64)>| {
                        // ranges are split at k / k+1 boundaries only
                        let cuts = [a, k.clamp(a, z + 1), (k + 1).clamp(a, z + 1), z + 1];
                        for w in cuts.windows(2) {
                            if w[0] < w[1] {
                                if pred(w[0]) {
                                    ts.push((w[0], w[1] - 1));
                                } else {
                                    fs.push((w[0], w[1] - 1));
                                }
                            }
                        }
                    };
                    let pred: Box<dyn Fn(i64) -> bool> = match op {
                        BinOp::Eq => Box::new(move |v| v == k),
                        BinOp::Ne => Box::new(move |v| v != k),
                        BinOp::Lt => Box::new(move |v| v < k),
                        BinOp::Le => Box::new(move |v| v <= k),
                        BinOp::Gt => Box::new(move |v| v > k),
                        BinOp::Ge => Box::new(move |v| v >= k),
                        _ => return None,
                    };
                    split(&*pred, &mut ts, &mut fs);
                }
            }
            if !ts.is_empty() {
                work.push((taken, ts));
            }
            if !fs.is_empty() {
                work.push((fall, fs));
            }
        }
        let tree_nodes = nodes.iter().filter(|&&n| self.cond_edges(n).is_some()).count();
        if tree_nodes < 2 && self.insns.is_empty() {
            return None;
        }
        // default: the leaf owning the unbounded ranges
        let unbounded: Vec<usize> = leaves.iter().filter(|(lo, hi, _)| *lo == i32::MIN as i64 || *hi == i32::MAX as i64).map(|l| l.2).collect();
        let default = unbounded.first().copied();
        if unbounded.iter().any(|d| Some(*d) != default) {
            return None;
        }
        let mut cases: Vec<(Vec<i64>, usize)> = vec![];
        for (lo, hi, t) in leaves {
            if Some(t) == default {
                continue;
            }
            if hi - lo > 64 {
                return None;
            }
            for v in lo..=hi {
                match cases.iter_mut().find(|(_, x)| *x == t) {
                    Some(c) => c.0.push(v),
                    None => cases.push((vec![v], t)),
                }
            }
        }
        if cases.len() + (default.is_some() as usize) < 2 {
            return None;
        }
        for c in cases.iter_mut() {
            c.0.sort();
        }
        cases.sort_by_key(|c| c.0[0]);
        if !self.insns.is_empty() {
            match self.check_tree(root, &cases, default, &nodes) {
                TreeCheck::Match(extra) => cases.extend(extra),
                // not MWCC's tree for any case set tried: an if/else-if chain on one value
                TreeCheck::NoMatch => return None,
                // a one-node "tree" must be MWCC's single-case switch (`beq case; b end`)
                TreeCheck::Unknown if tree_nodes < 2 || self.has_stmts(root) => return None,
                TreeCheck::Unknown => {}
            }
        }
        Some((x, cases, default, nodes))
    }

    /// Compare the target's compare tree with MWCC's tree for this case set. `Some(extra)` when
    /// it matches, `extra` being the empty-body case groups (`case k: break;`) needed for that;
    /// `None` when no case set tried reproduces the tree.
    fn check_tree(&self, root: usize, cases: &[(Vec<i64>, usize)], default: Option<usize>, nodes: &[usize]) -> TreeCheck {
        use ppc750cl::Opcode;
        let blk = &self.cfg.blocks[root];
        // the root's compare: `cmpwi x, K`, or a record form (`mr. r30, r6`) for K = 0
        let is_cmp = |k: usize| matches!(self.insns[k].op(), Opcode::Cmpi | Opcode::Cmpli | Opcode::Cmp | Opcode::Cmpl);
        let is_rec = |k: usize| !is_cmp(k) && self.insns[k].rc() && !matches!(self.insns[k].op(), Opcode::Bc | Opcode::B | Opcode::Bclr | Opcode::Bcctr);
        let Some(ci) = (blk.start..blk.end).rev().find(|&k| is_cmp(k) || is_rec(k)) else {
            return TreeCheck::Unknown;
        };
        let ins = &self.insns[ci];
        // `lis rT, hi; [addi|ori rT, rX, lo]; cmpw x, rT` for constants beyond 16 bits (the
        // `lis` may be shared: `addi r0, r4, 4`)
        fn reg_const(insns: &[Insn], k: usize, r: u8, depth: u32) -> Option<i64> {
            if depth > 4 {
                return None;
            }
            let lo_bound = k.saturating_sub(48);
            let mut j = k;
            while j > lo_bound {
                j -= 1;
                let q = &insns[j];
                if !crate::insn::defs_uses(q).0.contains(&crate::insn::gpr(r)) {
                    continue;
                }
                return match q.op() {
                    ppc750cl::Opcode::Addis if q.ra() == 0 => Some(((q.simm() as i64) << 16) as i32 as i64),
                    ppc750cl::Opcode::Addi if q.ra() == 0 => Some(q.simm() as i64),
                    ppc750cl::Opcode::Addi => Some((reg_const(insns, j, q.ra(), depth + 1)? + q.simm() as i64) as i32 as i64),
                    ppc750cl::Opcode::Ori if q.ra() == r => Some(reg_const(insns, j, q.rs(), depth + 1)? | q.uimm() as i64),
                    _ => None,
                };
            }
            None
        }
        let reg_const = |k: usize, r: u8| reg_const(self.insns, k, r, 0);
        let sel: Vec<u8> = if is_rec(ci) {
            vec![ins.ra(), ins.rs()]
        } else if matches!(ins.op(), Opcode::Cmpi | Opcode::Cmp) && ins.ins.field_crfd() == 0 {
            vec![ins.ra()]
        } else {
            return TreeCheck::Unknown;
        };
        // the tree region: selector compares and cr0 branches, other scheduled-in code dropped
        let mut items: Vec<(u32, TItem)> = vec![];
        if is_rec(ci) {
            items.push((ins.off, TItem::Cmp(0)));
        }
        for (k, i) in self.insns.iter().enumerate().skip(ci) {
            match i.op() {
                Opcode::Cmpi if i.ins.field_crfd() == 0 && sel.contains(&i.ra()) => items.push((i.off, TItem::Cmp(i.simm() as i64))),
                Opcode::Cmp if i.ins.field_crfd() == 0 && sel.contains(&i.ra()) => match reg_const(k, i.rb()) {
                    Some(v) => items.push((i.off, TItem::Cmp(v))),
                    None => return TreeCheck::Unknown,
                },
                // a big constant's setup for the next compare
                Opcode::Addis | Opcode::Addi | Opcode::Ori => {}
                Opcode::Bc if i.is_cond_branch() && i.ins.field_bi() < 4 => {
                    let (bo, bi) = (i.ins.field_bo(), i.ins.field_bi());
                    let c = match (bo & 0x08 != 0, bi) {
                        (true, 2) => switchtree::Cond::Eq,
                        (false, 0) => switchtree::Cond::Ge,
                        (true, 0) => switchtree::Cond::Lt,
                        _ => break,
                    };
                    let Some(t) = i.target() else { return TreeCheck::Unknown };
                    items.push((i.off, TItem::Bc(c, t)));
                }
                Opcode::B if i.is_jump() && i.reloc.is_none() => {
                    let Some(t) = i.target() else { return TreeCheck::Unknown };
                    items.push((i.off, TItem::B(t)))
                }
                _ if i.is_call() || i.is_blr() || i.is_bctr() => break,
                // scheduled-in code only shares the root block; past it the case bodies start
                _ if k >= blk.end => break,
                _ => {}
            }
            if items.len() > 64 {
                break;
            }
        }
        let off_of = |b: usize| -> Option<u32> { self.cfg.blocks.get(b).map(|bb| self.insns[bb.start].off) };
        let join = self.join_of_set(nodes);
        let def_off = default.and_then(off_of);
        let join_off = join.and_then(off_of);
        let addr = |l: Lab| -> Option<u32> {
            match l {
                Lab::Default => def_off,
                Lab::Case(b) => off_of(b),
                Lab::Extra(_) => join_off,
                Lab::Internal(_) => None,
            }
        };
        let base: Vec<(i64, Lab)> = cases.iter().flat_map(|(vs, t)| vs.iter().map(move |&v| (v, Lab::Case(*t)))).collect();
        if std::env::var_os("MWDEC_DUMP").is_some() {
            eprintln!("switch tree at block {root}: cases {base:?} default {def_off:?} join {join_off:?}
  target {items:?}
  plain {:?}", switchtree::simulate(&base, &addr));
        }
        let try_set = |extra: &[(i64, Lab)]| -> bool {
            let mut all = base.clone();
            all.extend_from_slice(extra);
            !switchtree::uses_table(&all) && switchtree::matches(&switchtree::simulate(&all, &addr), &items, &addr)
        };
        if try_set(&[]) {
            return TreeCheck::Match(vec![]);
        }
        if join_off.is_none() {
            return TreeCheck::NoMatch;
        }
        // empty-body case labels near the compared constants
        let used: HashSet<i64> = base.iter().map(|c| c.0).collect();
        let mut cand: Vec<i64> = vec![];
        for (_, it) in &items {
            if let TItem::Cmp(k) = it {
                for d in -2..=2 {
                    let v = k + d;
                    if !used.contains(&v) && !cand.contains(&v) {
                        cand.push(v);
                    }
                }
            }
        }
        cand.sort();
        if cand.len() > 14 {
            return TreeCheck::Unknown;
        }
        let n = cand.len();
        let mut subsets: Vec<Vec<i64>> = vec![];
        for a in 0..n {
            subsets.push(vec![cand[a]]);
        }
        for a in 0..n {
            for b in a + 1..n {
                subsets.push(vec![cand[a], cand[b]]);
            }
        }
        for a in 0..n {
            for b in a + 1..n {
                for c in b + 1..n {
                    subsets.push(vec![cand[a], cand[b], cand[c]]);
                }
            }
        }
        for sub in &subsets {
            // one shared empty body (`case a: case b: break;`) or one per value
            let shared: Vec<(i64, Lab)> = sub.iter().map(|&v| (v, Lab::Extra(0))).collect();
            if try_set(&shared) {
                return TreeCheck::Match(vec![(sub.clone(), EXTRA_CASE)]);
            }
            if sub.len() > 1 {
                let own: Vec<(i64, Lab)> = sub.iter().enumerate().map(|(k, &v)| (v, Lab::Extra(k as u32))).collect();
                if try_set(&own) {
                    return TreeCheck::Match(sub.iter().enumerate().map(|(k, &v)| (vec![v], EXTRA_CASE + k)).collect());
                }
            }
        }
        TreeCheck::NoMatch
    }

}

/// A statement list that only assigns one variable, possibly through an if/else whose arms both
/// assign it: (variable, value as a (nested) ternary).
fn ternary_assign(stmts: &[Stmt]) -> Option<(VarId, Expr)> {
    let s: Vec<&Stmt> = stmts.iter().filter(|s| !matches!(s, Stmt::Label(_))).collect();
    match s.as_slice() {
        [Stmt::Assign { dst: Expr::Var(v), src }] => Some((*v, src.clone())),
        // `v = a; v = f(v);` (a clamp's second bound): `v = f(a)`, a read again
        [Stmt::Assign { dst: Expr::Var(v), src: a }, Stmt::Assign { dst: Expr::Var(w), src: b }]
            if v == w && !a.has_call() && !a.uses_var(*v) && b.uses_var(*v) && matches!(b, Expr::Ternary { .. }) =>
        {
            let mut e = b.clone();
            e.rewrite(&mut |x| {
                if matches!(x, Expr::Var(y) if *y == *v) {
                    *x = a.clone();
                }
            });
            Some((*v, e))
        }
        [Stmt::If { cond, then, els }] if !then.is_empty() && !els.is_empty() => {
            let (a, ea) = ternary_assign(then)?;
            let (b, eb) = ternary_assign(els)?;
            if a != b {
                return None;
            }
            Some((a, Expr::Ternary { c: Box::new(cond.clone()), t: Box::new(ea), f: Box::new(eb), ty: mwdec_core::Type::Unknown { size: 4 } }))
        }
        _ => None,
    }
}

fn strip_casts(e: Expr) -> Expr {
    match e {
        Expr::Cast { e, .. } => strip_casts(*e),
        e => e,
    }
}

/// Jump-table selector `x - first` -> (x, first).
fn switch_selector(e: Expr) -> (Expr, i64) {
    match e {
        Expr::Binary { op: BinOp::Sub, l, r, .. } if r.as_int().is_some() => (strip_casts(*l), r.as_int().unwrap()),
        Expr::Binary { op: BinOp::Add, l, r, .. } if r.as_int().is_some() => (strip_casts(*l), -r.as_int().unwrap()),
        e => (strip_casts(e), 0),
    }
}

fn strip_trailing_continue(body: &mut Vec<Stmt>) {
    while matches!(body.last(), Some(Stmt::Continue)) {
        body.pop();
    }
}
