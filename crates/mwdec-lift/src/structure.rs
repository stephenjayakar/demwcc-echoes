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
    /// Variables the structurer introduced (numbered after `vars`).
    pub extra_vars: Vec<Var>,
    /// The last `case_tree` was confirmed against MWCC's tree builder.
    tree_confirmed: std::cell::Cell<bool>,
    /// A loop header `bool_region` may start at (its value is the loop test).
    bool_header: Option<usize>,
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

/// The return of a void function as a switch-tree leaf (`blr`, `bgelr`): the switch's end.
const RET_LEAF: usize = usize::MAX - 7;

/// Offset standing for "returns" in compare-tree items.
const RET_OFF: u32 = u32::MAX;

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
            extra_vars: vec![],
            tree_confirmed: std::cell::Cell::new(false),
            bool_header: None,
        }
    }

    /// Target instructions, for checking switch trees against MWCC's tree builder.
    pub fn with_insns(mut self, insns: &'a [Insn]) -> Self {
        self.insns = insns;
        self
    }

    pub fn run(self) -> Vec<Stmt> {
        self.run_with_vars().0
    }

    /// Structure the function; also returns the variables introduced on the way.
    pub fn run_with_vars(mut self) -> (Vec<Stmt>, Vec<Var>) {
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
        (out, self.extra_vars)
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
        let mut fuel = crate::fuel::Fuel::new("structure.build", crate::fuel::CAP_WALK);
        loop {
            if !fuel.burn() {
                return;
            }
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
                // unreachable after a statement that always leaves (both arms returned)
                if !diverges(out) {
                    out.push(Stmt::Goto(cur));
                    self.gotos.insert(cur);
                }
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
                    if let Some(next) = self.one_case_switch(cur, out) {
                        match next {
                            Some(j) => {
                                cur = j;
                                continue;
                            }
                            None => return,
                        }
                    }
                    if let Some(next) = self.bool_region(cur, out) {
                        match next {
                            Some(j) => {
                                cur = j;
                                continue;
                            }
                            None => return,
                        }
                    }
                    // an arm that never comes back (a loop left only by returns) leaves no post-
                    // dominator: the function's shared final return is still where the other
                    // arm goes on
                    let join = self.join_of(cur).or_else(|| {
                        let r = self.final_return()?;
                        let (t, f) = self.cond_edges(cur)?;
                        let other = if t == r { f } else if f == r { t } else { return None };
                        self.reaches_loop_before(other, r).then_some(r)
                    });
                    if let Some((e, cases, default, nodes))= self.case_tree(cur).or_else(|| self.guarded_table(cur)) {
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
                    // (after the switch trees: a compare tree's last test can branch over a `b`)
                    if let Some(next) = self.or_return_chain(cur, out) {
                        cur = next;
                        continue;
                    }
                    if let Some(next) = self.shared_return_or_chain(cur, out) {
                        cur = next;
                        continue;
                    }
                    self.build_if(cur, join, out);
                    match join {
                        Some(j) => cur = j,
                        None => return,
                    }
                }
                Term::CondReturn { fall } => {
                    // a void leaf's compare tree can start with a conditional return
                    if self.ret_void && self.blocks[cur].ret.is_none() {
                        self.tree_confirmed.set(false);
                        if let Some((e, cases, default, nodes)) = self.case_tree(cur) {
                            if self.tree_confirmed_last() {
                                for n in &nodes {
                                    self.emitted[*n] = true;
                                }
                                let stmt = self.build_cases(e, cases, default, None);
                                out.push(stmt);
                                return;
                            }
                        }
                    }
                    if let Some(next) = self.leaf_or_return(cur, out) {
                        cur = next;
                        continue;
                    }
                    if let Some(next) = self.one_case_switch(cur, out) {
                        match next {
                            Some(j) => {
                                cur = j;
                                continue;
                            }
                            None => return,
                        }
                    }
                    if let Some(next) = self.bool_region(cur, out) {
                        match next {
                            Some(j) => {
                                cur = j;
                                continue;
                            }
                            None => return,
                        }
                    }
                    let c = self.cond_of(cur);
                    let r = self.blocks[cur].ret.clone();
                    // `cmplwi x, N ; bgtlr` guarding a jump table whose switch ends the function:
                    // the switch's own range check (no default)
                    let table_guard = self.ret_void
                        && fall < self.cfg.blocks.len()
                        && matches!(self.cfg.blocks[fall].term, Term::Switch { .. })
                        && !self.has_stmts(fall)
                        && self.join_of(fall).is_none()
                        && matches!(&c, Expr::Binary { op: BinOp::Gt, .. });
                    if table_guard {
                        cur = fall;
                        continue;
                    }
                    // the same with a value in place (`bgtlr` returning the selector's source):
                    // `switch (x) { case ...: return ...; } return v;`
                    let value_guard = !self.ret_void
                        && fall < self.cfg.blocks.len()
                        && matches!(self.cfg.blocks[fall].term, Term::Switch { .. })
                        && !self.has_stmts(fall)
                        && self.cfg.blocks[fall].preds.len() == 1
                        && self.join_of(fall).is_none()
                        && matches!(&c, Expr::Binary { op: BinOp::Gt, .. })
                        && matches!(r, Some(Expr::Var(_)));
                    if value_guard {
                        if let Term::Switch { targets, .. } = self.cfg.blocks[fall].term.clone() {
                            self.emitted[fall] = true;
                            let stmt = self.build_switch(fall, &targets, None);
                            out.push(stmt);
                            out.push(Stmt::Return(r));
                            return;
                        }
                    }
                    // `li r3,A ; b<c>lr ; li r3,B ; blr` is MWCC's select layout for `return !c ?
                    // B : A` (the else value is loaded first); `if (c) return A; return B;`
                    // would load B first
                    let tail = (fall < self.cfg.blocks.len() && matches!(self.cfg.blocks[fall].term, Term::Return) && !self.has_stmts(fall) && !self.emitted[fall] && self.cfg.blocks[fall].preds.len() == 1)
                        .then(|| self.blocks[fall].ret.clone())
                        .flatten();
                    if let (Some(a), Some(b), false) = (r.clone(), tail, self.ret_void) {
                        let nc = c.clone().negate(self.vars);
                        // the tail computes its value (`subi; clrlwi; extsb; blr`) rather than
                        // loading it, the other value a variable already in place: `if (!c) return b;
                        // return a;` (a ternary would compute b first; a computed `a` is a ternary)
                        let simple_tail = self.insns.is_empty() || {
                            use ppc750cl::Opcode;
                            let fb = &self.cfg.blocks[fall];
                            let real: Vec<&Insn> = (fb.start..fb.end).map(|k| &self.insns[k]).filter(|i| !i.is_blr()).collect();
                            real.len() <= 2 && real.iter().all(|i| matches!(i.op(), Opcode::Addi | Opcode::Addis | Opcode::Or | Opcode::Ori | Opcode::Fmr | Opcode::Lfs | Opcode::Lfd))
                        };
                        if !simple_tail && matches!(a, Expr::Var(_)) {
                            self.emitted[fall] = true;
                            out.push(Stmt::If { cond: nc, then: vec![Stmt::Return(Some(b))], els: vec![] });
                            out.push(Stmt::Return(Some(a)));
                            return;
                        }
                        // constants MWCC would select without a branch (`c ? K : 0`, `c ? K+1 :
                        // K`): the source assigned a variable (`v = a; if (!c) v = b; return v;`)
                        let branchless = match (a.as_int(), b.as_int()) {
                            (Some(x), Some(y)) => x == 0 || y == 0 || (x - y).abs() == 1,
                            // `c ? v : 0` is a mask too (`and`/`andc`) for integers (not for pointers)
                            (Some(0), None) => matches!(&b, Expr::Var(x) if matches!(strip_cv(&self.vars[*x].ty), mwdec_core::Type::Int { .. })),
                            (None, Some(0)) => matches!(&a, Expr::Var(x) if matches!(strip_cv(&self.vars[*x].ty), mwdec_core::Type::Int { .. })),
                            _ => false,
                        };
                        if branchless {
                            self.emitted[fall] = true;
                            let v = self.vars.len() + self.extra_vars.len();
                            let ty = match (&a, &b) {
                                (Expr::Var(x), _) | (_, Expr::Var(x)) => self.vars[*x].ty.clone(),
                                _ => t_s32(),
                            };
                            self.extra_vars.push(Var { name: "var_r3".into(), ty, kind: VarKind::Local });
                            out.push(Stmt::Assign { dst: Expr::Var(v), src: a });
                            out.push(Stmt::If { cond: nc, then: vec![Stmt::Assign { dst: Expr::Var(v), src: b }], els: vec![] });
                            out.push(Stmt::Return(Some(Expr::Var(v))));
                            return;
                        }
                        if !matches!(nc, Expr::Unary { op: UnOp::Not, .. }) {
                            self.emitted[fall] = true;
                            let ty = mwdec_core::Type::Unknown { size: 4 };
                            out.push(Stmt::Return(Some(Expr::Ternary { c: Box::new(nc), t: Box::new(b), f: Box::new(a), ty })));
                            return;
                        }
                    }
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
        // every path from b returns without coming back into the loop (`if (c) { f(); return
        // true; } return false;` included), within a small region
        let mut seen: HashSet<usize> = HashSet::new();
        let mut work = vec![b];
        while let Some(cur) = work.pop() {
            if cur == self.exit_node() {
                continue;
            }
            if body.contains(&cur) || !seen.insert(cur) || seen.len() > 12 {
                return false;
            }
            match &self.cfg.blocks[cur].term {
                Term::Return | Term::TailCall => {}
                Term::Fall(t) | Term::Jump(t) => work.push(*t),
                Term::Cond { taken, fall } => {
                    work.push(*taken);
                    work.push(*fall);
                }
                Term::CondReturn { fall } => work.push(*fall),
                _ => return false,
            }
        }
        true
    }

    /// A `&&`/`||` chain in value context (MWCC's `gen_LOGICAL`: `li r,0; cmp; b..; cmp; b..;
    /// li r,1`): a single-entry DAG of tests from `s` whose exits only produce the constants 0
    /// and 1, either as returns (`li r3,0; ...; bnelr; li r3,1; blr`) or into a variable that
    /// was 0 before the first test (`v = 0; ...; v = 1; J:`). Emits `return a && b;` /
    /// `v = a && b;` (statements already pushed for `s` are in `out`). A branch-context chain
    /// (`if (a || b) v = 1;`) is not the same code: MWCC turns `==` chains into range tests
    /// there. Returns the block to continue at (None: the region returned).
    fn bool_region(&mut self, s: usize, out: &mut Vec<Stmt>) -> Option<Option<usize>> {
        const T: usize = usize::MAX - 1;
        const F: usize = usize::MAX - 2;
        let nblocks = self.cfg.blocks.len();
        let edges_of = |me: &Self, b: usize| -> Option<(usize, Option<usize>)> {
            // (fallthrough or taken target, taken target if a block): CondReturn has no taken block
            match me.cfg.blocks[b].term {
                Term::Cond { taken, fall } if taken != fall => Some((fall, Some(taken))),
                Term::CondReturn { fall } => Some((fall, None)),
                _ => None,
            }
        };
        edges_of(self, s)?;
        if self.loops.contains_key(&s) && self.bool_header != Some(s) {
            return None;
        }
        // statements a test may carry: definitions of temps only read by later tests
        let temp_defs = |me: &Self, b: usize| -> Option<Vec<(VarId, Expr)>> {
            let mut defs = vec![];
            for st in &me.blocks[b].stmts {
                match st {
                    Stmt::Label(_) | Stmt::Comment(_) => {}
                    // single-assignment temps only: substituting a variable's value into a later test
                    // must not see a reassignment in between
                    Stmt::Assign { dst: Expr::Var(v), src } if !src.has_call() && me.vars[*v].name.starts_with("temp_") && !matches!(me.vars[*v].kind, VarKind::Param { .. } | VarKind::This) => defs.push((*v, src.clone())),
                    _ => return None,
                }
            }
            Some(defs)
        };
        let mut set: Vec<usize> = vec![s];
        let mut fuel = crate::fuel::Fuel::new("structure.bool_region", crate::fuel::CAP_WALK);
        loop {
            if !fuel.burn() {
                return None;
            }
            let mut grew = false;
            let mut cands: Vec<usize> = vec![];
            for &n in &set {
                let (f, t) = edges_of(self, n).unwrap();
                cands.push(f);
                if let Some(t) = t {
                    cands.push(t);
                }
            }
            cands.sort_by_key(|&b| if b < nblocks { self.cfg.blocks[b].start } else { usize::MAX });
            for x in cands {
                if x >= nblocks || set.contains(&x) || x == self.exit_node() || self.emitted[x] || self.loops.contains_key(&x) {
                    continue;
                }
                if edges_of(self, x).is_none() || temp_defs(self, x).is_none() || !self.cfg.blocks[x].preds.iter().all(|p| set.contains(p)) {
                    continue;
                }
                if self.stack.last().map_or(false, |c| x == c.header || Some(x) == c.exit) {
                    continue;
                }
                set.push(x);
                grew = true;
            }
            if !grew {
                break;
            }
        }
        if set.len() < 2 || set.len() > 16 {
            return None;
        }
        // classify the exits
        let const_ret = |me: &Self, b: usize| -> Option<i64> {
            if b < nblocks && matches!(me.cfg.blocks[b].term, Term::Return) && !me.has_stmts(b) {
                return me.blocks[b].ret.as_ref()?.as_int();
            }
            None
        };
        // (var, value, join) of a block that only sets a variable to a constant
        let set_block = |me: &Self, b: usize| -> Option<(VarId, i64, usize)> {
            if b >= nblocks || me.emitted[b] {
                return None;
            }
            let st: Vec<&Stmt> = me.blocks[b].stmts.iter().filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))).collect();
            let [Stmt::Assign { dst: Expr::Var(v), src }] = st.as_slice() else { return None };
            let k = src.as_int()?;
            match me.cfg.blocks[b].term {
                Term::Fall(j) | Term::Jump(j) => Some((*v, k, j)),
                _ => None,
            }
        };
        #[derive(Clone, Copy, PartialEq)]
        enum Exit {
            Val(i64),
            Set(VarId, i64, usize, usize),
            Join(usize),
        }
        let mut exits: Vec<(usize, Exit)> = vec![];
        for &n in &set {
            let (f, t) = edges_of(self, n).unwrap();
            let mut targets = vec![f];
            match t {
                Some(t) => targets.push(t),
                None => {
                    // CondReturn: the returned value must be a constant
                    let k = self.blocks[n].ret.as_ref()?.as_int()?;
                    exits.push((usize::MAX, Exit::Val(k)));
                }
            }
            for x in targets {
                if set.contains(&x) || exits.iter().any(|(b, _)| *b == x) {
                    continue;
                }
                let e = if let Some(k) = const_ret(self, x) {
                    Exit::Val(k)
                } else if let Some((v, k, j)) = set_block(self, x) {
                    Exit::Set(v, k, j, x)
                } else {
                    Exit::Join(x)
                };
                exits.push((x, e));
            }
        }
        // variant 1: every exit returns a constant 0/1, both values present
        let vals: Vec<i64> = exits.iter().filter_map(|(_, e)| if let Exit::Val(k) = e { Some(*k) } else { None }).collect();
        let mut map: HashMap<usize, usize> = HashMap::new();
        let mut cr_val: Option<usize> = None;
        let variant;
        if vals.len() == exits.len() {
            // MWCC loads 0 before the tests (the conditional returns: `bnelr`), 1 is set in a
            // single shared return block (`li r3,1; blr`); two plain
            // return blocks are an if/else of returns, not a value chain
            let cr: Vec<i64> = exits.iter().filter(|(b, _)| *b == usize::MAX).filter_map(|(_, e)| if let Exit::Val(k) = e { Some(*k) } else { None }).collect();
            let rb: Vec<(usize, i64)> = exits.iter().filter(|(b, _)| *b != usize::MAX).filter_map(|(b, e)| if let Exit::Val(k) = e { Some((*b, *k)) } else { None }).collect();
            if cr.is_empty() || rb.len() != 1 || cr.iter().any(|&k| k != 0) || rb[0].1 != 1 {
                return None;
            }
            cr_val = Some(if cr[0] != 0 { T } else { F });
            map.insert(rb[0].0, if rb[0].1 != 0 { T } else { F });
            variant = 1;
        } else {
            // variant 2: exits are one `v = k` block and its join, v held the other value before
            let sets: Vec<(VarId, i64, usize, usize)> = exits.iter().filter_map(|(_, e)| if let Exit::Set(v, k, j, b) = e { Some((*v, *k, *j, *b)) } else { None }).collect();
            let [(v, k, j, sb)] = sets.as_slice() else { return None };
            if *k != 1 || exits.len() != 2 || !exits.iter().any(|(_, e)| *e == Exit::Join(*j)) {
                return None;
            }
            if self.cfg.blocks[*sb].preds.iter().any(|p| !set.contains(p)) {
                return None;
            }
            // the value before the tests: the last assignment to v in s's statements
            let pre = self.blocks[s].stmts.iter().rev().find_map(|st| match st {
                Stmt::Assign { dst: Expr::Var(x), src } if x == v => Some(src.as_int()),
                _ => None,
            })??;
            if pre != 1 - *k {
                return None;
            }
            map.insert(*sb, if *k != 0 { T } else { F });
            map.insert(*j, if *k != 0 { F } else { T });
            variant = 2;
        }
        // temps defined in tests after the first: substituted into the later tests when nothing
        // else reads them
        let mut subst: Vec<(VarId, Expr)> = vec![];
        for &n in &set[1..] {
            for (v, src) in temp_defs(self, n).unwrap() {
                subst.push((v, src));
            }
        }
        for (v, _) in &subst {
            if self.var_read_outside(*v, &set) {
                return None;
            }
        }
        let mut edges: Vec<(usize, Expr, usize, usize)> = vec![];
        let resolve = |x: usize| -> usize { map.get(&x).copied().unwrap_or(x) };
        for &n in &set {
            let mut c = self.cond_of(n);
            for (v, src) in subst.iter().rev() {
                c.rewrite(&mut |e| {
                    if matches!(e, Expr::Var(x) if x == v) {
                        *e = src.clone();
                    }
                });
            }
            let (f, t) = edges_of(self, n).unwrap();
            let t = match t {
                Some(t) => resolve(t),
                None => cr_val?,
            };
            edges.push((n, c, t, resolve(f)));
        }
        if edges.iter().any(|(_, _, t, f)| (*t != T && *t != F && !set.contains(t)) || (*f != T && *f != F && !set.contains(f))) {
            return None;
        }
        let (c, t, _) = reduce_cond_dag(edges, s, self.vars)?;
        let cond = if t == T { c } else { c.negate(self.vars) };
        for &n in &set {
            self.emitted[n] = true;
        }
        if variant == 1 {
            for b in map.keys() {
                // a shared `return 1;` may be the target of other code too: only drop it when the
                // region is its only way in
                if self.cfg.blocks[*b].preds.iter().all(|p| set.contains(p)) {
                    self.emitted[*b] = true;
                }
            }
            out.push(Stmt::Return(Some(cond)));
            return Some(None);
        }
        let (v, sb, j) = exits
            .iter()
            .find_map(|(_, e)| if let Exit::Set(v, _, j, b) = e { Some((*v, *b, *j)) } else { None })
            .unwrap();
        self.emitted[sb] = true;
        // drop the preload pushed with s's statements
        if let Some(i) = out.iter().rposition(|st| matches!(st, Stmt::Assign { dst: Expr::Var(x), src } if *x == v && src.as_int().is_some())) {
            out.remove(i);
        }
        out.push(Stmt::Assign { dst: Expr::Var(v), src: cond });
        Some(Some(j))
    }

    /// `if (a || b || !c) return;` in a void function: the tests before the last branch straight
    /// to the epilogue, the last one branches over a lone `b epilogue` (`bne L; b end; L:`), which
    /// MWCC emits for the last term of an `||` guarding a `return` (a single test branches to the
    /// epilogue itself). Returns the block the code goes on at.
    fn or_return_chain(&mut self, s: usize, out: &mut Vec<Stmt>) -> Option<usize> {
        if !self.ret_void || self.loops.contains_key(&s) {
            return None;
        }
        let is_end = |me: &Self, b: usize| b < me.cfg.blocks.len() && matches!(me.cfg.blocks[b].term, Term::Return) && !me.has_stmts(b) && me.blocks[b].ret.is_none();
        let mut chain = vec![s];
        let mut conds = vec![];
        let mut cur = s;
        let end;
        let mut fuel = crate::fuel::Fuel::new("structure.or_return_chain", crate::fuel::CAP_WALK);
        loop {
            if !fuel.burn() {
                return None;
            }
            let (t, f) = self.cond_edges(cur)?;
            if cur != s && (self.has_stmts(cur) || self.cfg.blocks[cur].preds.len() != 1 || self.emitted[cur] || self.loops.contains_key(&cur)) {
                return None;
            }
            // the last test: over `b end`
            if !self.has_stmts(f) && f < self.cfg.blocks.len() && self.cfg.blocks[f].preds.len() == 1 {
                if let Term::Jump(e) = self.cfg.blocks[f].term {
                    if is_end(self, e) && chain.len() >= 2 && conds.len() == chain.len() - 1 {
                        conds.push(self.cond_of(cur).negate(self.vars));
                        self.emitted[f] = true;
                        end = (t, e, f);
                        break;
                    }
                }
            }
            if !is_end(self, t) {
                return None;
            }
            if chain.len() > 1 {
                // all earlier tests go to the same end
                let first_end = self.cond_edges(s)?.0;
                if t != first_end {
                    return None;
                }
            }
            conds.push(self.cond_of(cur));
            cur = f;
            chain.push(cur);
        }
        let (cont, e, _) = end;
        if self.cond_edges(s)?.0 != e {
            return None;
        }
        for &n in &chain[1..] {
            self.emitted[n] = true;
        }
        let cond = conds.into_iter().reduce(|a, b| Expr::cmp(BinOp::LogOr, a, b))?;
        out.push(Stmt::If { cond, then: vec![Stmt::Return(None)], els: vec![] });
        Some(cont)
    }

    /// See `build_loop`: (condition, body start) of a loop whose header computes its test as a
    /// value-context `&&`/`||` into a flag the next block tests.
    fn bool_loop_test(&mut self, h: usize, body: &BTreeSet<usize>) -> Option<(Expr, usize)> {
        let saved = self.emitted.clone();
        self.bool_header = Some(h);
        let mut tmp = vec![];
        let r = self.bool_region(h, &mut tmp);
        self.bool_header = None;
        let res = (|| {
            let j = r??;
            let [Stmt::Assign { dst: Expr::Var(v), src: c }] = tmp.as_slice() else { return None };
            // the header carries nothing but the flag's preload
            let real: Vec<&Stmt> = self.blocks[h].stmts.iter().filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))).collect();
            if !matches!(real.as_slice(), [Stmt::Assign { dst: Expr::Var(x), .. }] if x == v) {
                return None;
            }
            if !body.contains(&j) || self.has_stmts(j) || tested_var_any(&self.cond_of(j)) != Some(*v) || self.reads_of(*v) != 1 {
                return None;
            }
            let (t, f) = self.cond_edges(j)?;
            let mut jc = self.cond_of(j);
            jc.rewrite(&mut |e| {
                if matches!(e, Expr::Var(x) if x == v) {
                    *e = c.clone();
                }
            });
            self.emitted[j] = true;
            match (body.contains(&t), body.contains(&f)) {
                (true, false) => Some((jc, t)),
                (false, true) => Some((jc.negate(self.vars), f)),
                _ => None,
            }
        })();
        if res.is_none() {
            self.emitted = saved;
        }
        res
    }

    /// `if (a || b) return k;` with one return block shared by the tests: the first ones branch
    /// to it, the last falls into it. A test after the first may load a value the code after the
    /// `if` reuses (`if (!p || p->n == 0) return -1; ... p->n ...`): the load is substituted into
    /// the test and repeated after the `if` (MWCC CSEs the two reads).
    fn shared_return_or_chain(&mut self, s: usize, out: &mut Vec<Stmt>) -> Option<usize> {
        let nb = self.cfg.blocks.len();
        let (t0, _) = self.cond_edges(s)?;
        let r = t0;
        if r >= nb || !matches!(self.cfg.blocks[r].term, Term::Return) || self.has_stmts(r) || self.emitted[r] {
            return None;
        }
        let mut conds = vec![self.cond_of(s)];
        let mut chain = vec![s];
        let mut defs: Vec<(VarId, Expr)> = vec![];
        let (_, mut cur) = self.cond_edges(s)?;
        let mut fuel = crate::fuel::Fuel::new("structure.shared_return_or_chain", crate::fuel::CAP_WALK);
        loop {
            if !fuel.burn() {
                return None;
            }
            if cur >= nb || cur == r || self.emitted[cur] || self.cfg.blocks[cur].preds.len() != 1 || self.loops.contains_key(&cur) {
                return None;
            }
            for st in &self.blocks[cur].stmts {
                match st {
                    Stmt::Label(_) | Stmt::Comment(_) => {}
                    Stmt::Assign { dst: Expr::Var(v), src } if !src.has_call() && self.vars[*v].name.starts_with("temp_") => defs.push((*v, src.clone())),
                    _ => return None,
                }
            }
            let (mut t, mut f) = self.cond_edges(cur)?;
            let mut c = self.cond_of(cur);
            if c.has_call() {
                return None;
            }
            // a flag set to 0 or 1 by the test and tested next (`entryIsDir(n)` as `x ? 1 : 0`):
            // the flag's test with the ternary in place
            let set01 = |me: &Self, b: usize| -> Option<(VarId, i64, usize)> {
                if b >= nb || me.cfg.blocks[b].preds.len() != 1 || me.emitted[b] {
                    return None;
                }
                let st: Vec<&Stmt> = me.blocks[b].stmts.iter().filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))).collect();
                let [Stmt::Assign { dst: Expr::Var(v), src }] = st.as_slice() else { return None };
                let k = src.as_int().filter(|k| *k == 0 || *k == 1)?;
                match me.cfg.blocks[b].term {
                    Term::Fall(m) | Term::Jump(m) => Some((*v, k, m)),
                    _ => None,
                }
            };
            if let (Some((vt, kt, mt)), Some((vf, kf, mf))) = (set01(self, t), set01(self, f)) {
                if vt == vf && mt == mf && kt != kf && mt < nb && !self.has_stmts(mt) && !self.emitted[mt] && self.cfg.blocks[mt].preds.len() == 2 {
                    // (the fallthrough value is the ternary's first arm)
                    let flag = Expr::Ternary { c: Box::new(c.clone().negate(self.vars)), t: Box::new(Expr::int(kf)), f: Box::new(Expr::int(kt)), ty: t_s32() };
                    let (t2, f2) = self.cond_edges(mt)?;
                    let mut c2 = self.cond_of(mt);
                    if tested_var_any(&c2) != Some(vt) || self.reads_of(vt) != 1 {
                        return None;
                    }
                    c2.rewrite(&mut |e| {
                        if matches!(e, Expr::Var(x) if *x == vt) {
                            *e = flag.clone();
                        }
                    });
                    chain.push(t);
                    chain.push(f);
                    // (the flag blocks and the test of the flag join the chain)
                    chain.push(cur);
                    cur = mt;
                    c = c2;
                    t = t2;
                    f = f2;
                }
            }
            for (v, src) in defs.iter().rev() {
                c.rewrite(&mut |e| {
                    if matches!(e, Expr::Var(x) if x == v) {
                        *e = src.clone();
                    }
                });
            }
            chain.push(cur);
            if t == r {
                conds.push(c);
                cur = f;
                continue;
            }
            if f == r {
                // the last test falls into the shared return
                conds.push(c.negate(self.vars));
                chain.dedup();
                if !self.cfg.blocks[r].preds.iter().all(|p| chain.contains(p)) || chain.len() < 2 {
                    return None;
                }
                for &n in &chain[1..] {
                    self.emitted[n] = true;
                }
                self.emitted[r] = true;
                let cond = conds.into_iter().reduce(|a, b| Expr::cmp(BinOp::LogOr, a, b))?;
                out.push(Stmt::If { cond, then: vec![Stmt::Return(self.blocks[r].ret.clone())], els: vec![] });
                for (v, src) in defs {
                    out.push(Stmt::Assign { dst: Expr::Var(v), src });
                }
                return Some(t);
            }
            return None;
        }
    }

    /// The leaf form of `or_return_chain`: `b<c>lr` for the first terms, the last one branching
    /// over a lone `blr` (`bne L; blr; L:`): `if (a || !b) return;`.
    fn leaf_or_return(&mut self, s: usize, out: &mut Vec<Stmt>) -> Option<usize> {
        if !self.ret_void || self.insns.is_empty() || self.blocks[s].ret.is_some() {
            return None;
        }
        let lone_blr = |me: &Self, b: usize| {
            b < me.cfg.blocks.len()
                && matches!(me.cfg.blocks[b].term, Term::Return)
                && me.cfg.blocks[b].end == me.cfg.blocks[b].start + 1
                && me.insns[me.cfg.blocks[b].start].is_blr()
                && me.cfg.blocks[b].preds.len() == 1
        };
        let mut conds = vec![self.cond_of(s)];
        let mut chain = vec![s];
        let Term::CondReturn { fall } = self.cfg.blocks[s].term else { return None };
        let mut cur = fall;
        let mut fuel = crate::fuel::Fuel::new("structure.leaf_or_return", crate::fuel::CAP_WALK);
        loop {
            if !fuel.burn() {
                return None;
            }
            if cur >= self.cfg.blocks.len() || self.has_stmts(cur) || self.emitted[cur] || self.cfg.blocks[cur].preds.len() != 1 || self.loops.contains_key(&cur) {
                return None;
            }
            match self.cfg.blocks[cur].term {
                Term::CondReturn { fall } if self.blocks[cur].ret.is_none() => {
                    conds.push(self.cond_of(cur));
                    chain.push(cur);
                    cur = fall;
                }
                Term::Cond { taken, fall } if taken != fall && lone_blr(self, fall) => {
                    conds.push(self.cond_of(cur).negate(self.vars));
                    chain.push(cur);
                    chain.push(fall);
                    for &n in &chain {
                        self.emitted[n] = true;
                    }
                    let cond = conds.into_iter().reduce(|a, b| Expr::cmp(BinOp::LogOr, a, b))?;
                    out.push(Stmt::If { cond, then: vec![Stmt::Return(None)], els: vec![] });
                    return Some(taken);
                }
                _ => return None,
            }
        }
    }

    /// `lis rT, hi; addi rT, rT, lo; cmpw x, rT; bne end`: an equality test against a constant
    /// beyond 16 bits built in a register is MWCC's single-label switch (an `if` compares such a
    /// constant with `subis; cmplwi`).
    fn one_case_switch(&mut self, b: usize, out: &mut Vec<Stmt>) -> Option<Option<usize>> {
        use ppc750cl::Opcode;
        if self.insns.is_empty() || self.loops.contains_key(&b) {
            return None;
        }
        let blk = &self.cfg.blocks[b];
        let last = blk.end.checked_sub(1)?;
        let br = &self.insns[last];
        if !(br.is_cond_branch() || matches!(br.op(), Opcode::Bclr)) || br.ins.field_bi() != 2 {
            return None;
        }
        let ci = (blk.start..last).rev().find(|&k| matches!(self.insns[k].op(), Opcode::Cmpi | Opcode::Cmpli | Opcode::Cmp | Opcode::Cmpl))?;
        let ins = &self.insns[ci];
        if ins.op() != Opcode::Cmp || ins.ins.field_crfd() != 0 {
            return None;
        }
        let k = reg_const(self.insns, ci, ins.rb(), 0)?;
        if (-0x8000..0x8000).contains(&k) {
            return None;
        }
        let c = self.blocks[b].cond.clone()?;
        let Expr::Binary { op, l, r, .. } = &c else { return None };
        if r.as_int() != Some(k) {
            return None;
        }
        let sel = strip_casts((**l).clone());
        let (case, end) = match (self.cfg.blocks[b].term.clone(), *op) {
            (Term::Cond { taken, fall }, BinOp::Ne) => (fall, Some(taken)),
            (Term::Cond { taken, fall }, BinOp::Eq) => (taken, Some(fall)),
            (Term::CondReturn { fall }, BinOp::Ne) if self.ret_void => (fall, None),
            _ => return None,
        };
        if case >= self.cfg.blocks.len() || self.emitted[case] || self.cfg.blocks[case].preds.len() != 1 {
            return None;
        }
        let end = end.filter(|&e| e != self.exit_node());
        if let Some(e) = end {
            if !self.cfg.postdominates(e, case) && self.join_of(b) != Some(e) {
                return None;
            }
        }
        let stmt = self.build_cases(sel, vec![(vec![k], case)], None, end);
        out.push(stmt);
        Some(end)
    }

    /// Does a path from `b` enter a loop before reaching `stop`?
    fn reaches_loop_before(&self, b: usize, stop: usize) -> bool {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut work = vec![b];
        while let Some(x) = work.pop() {
            if x == stop || x >= self.cfg.blocks.len() || !seen.insert(x) || seen.len() > 64 {
                continue;
            }
            if self.loops.contains_key(&x) {
                return true;
            }
            work.extend(self.cfg.blocks[x].succs.iter().copied());
        }
        false
    }

    /// Whether the last `case_tree` call was confirmed by the tree builder (call `case_tree`
    /// after resetting it via `confirmed_tree` or directly).
    fn tree_confirmed_last(&self) -> bool {
        self.tree_confirmed.get()
    }

    /// Is `n` the root of a compare tree MWCC's tree builder reproduces (not merely a run of
    /// tests on one value)?
    fn confirmed_tree(&self, n: usize) -> bool {
        self.tree_confirmed.set(false);
        self.case_tree(n).is_some() && self.tree_confirmed.get()
    }

    /// The function's last block when it is a return shared by several paths.
    fn final_return(&self) -> Option<usize> {
        let nb = self.cfg.blocks.len();
        let last = (0..nb).filter(|&b| self.cfg.idom[b] != usize::MAX).max_by_key(|&b| self.cfg.blocks[b].start)?;
        (matches!(self.cfg.blocks[last].term, Term::Return) && self.cfg.blocks[last].preds.len() >= 2 && !self.emitted[last]).then_some(last)
    }

    /// Number of reads of `v` in all blocks (statements, conditions, returns, switches).
    fn reads_of(&self, v: VarId) -> usize {
        let mut n = 0;
        for bo in self.blocks.iter() {
            for st in &bo.stmts {
                match st {
                    Stmt::Assign { dst: Expr::Var(x), src } if *x == v => src.walk(&mut |e| n += matches!(e, Expr::Var(y) if *y == v) as usize),
                    _ => Stmt::walk_exprs(std::slice::from_ref(st), &mut |e| n += matches!(e, Expr::Var(y) if *y == v) as usize),
                }
            }
            for e in [&bo.cond, &bo.ret, &bo.switch].into_iter().flatten() {
                e.walk(&mut |x| n += matches!(x, Expr::Var(y) if *y == v) as usize);
            }
        }
        n
    }

    /// Is `v` read anywhere except in the conditions of `region` (its definitions there aside)?
    fn var_read_outside(&self, v: VarId, region: &[usize]) -> bool {
        for (b, bo) in self.blocks.iter().enumerate() {
            let mut found = false;
            for st in &bo.stmts {
                match st {
                    Stmt::Assign { dst: Expr::Var(x), src } if *x == v => found |= src.uses_var(v),
                    _ => Stmt::walk_exprs(std::slice::from_ref(st), &mut |e| found |= matches!(e, Expr::Var(x) if *x == v)),
                }
            }
            if !region.contains(&b) {
                found |= bo.cond.as_ref().map_or(false, |c| c.uses_var(v));
            }
            found |= bo.ret.as_ref().map_or(false, |c| c.uses_var(v));
            found |= bo.switch.as_ref().map_or(false, |c| c.uses_var(v));
            if found {
                return true;
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
        let mut fuel = crate::fuel::Fuel::new("structure.build_if", crate::fuel::CAP_WALK);
        loop {
            if !fuel.burn() {
                break;
            }
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
            // the root of one of MWCC's compare trees starts a switch, not another `&&` term
            if !ok || self.confirmed_tree(n) {
                break;
            }
            chain.push(n);
            cur = n;
        }
        let (cond, if_node, else_node) = loop {
            if let Some(r) = self.try_make_cond(&chain, join) {
                break r;
            }
            if chain.len() == 1 {
                // no reduction possible: the head's own test
                let (t, f) = self.cond_edges(start).unwrap();
                let c = self.cond_of(start);
                break if Some(t) == join { (c.negate(self.vars), f, None) } else if Some(f) == join { (c, t, None) } else { (c.negate(self.vars), f, Some(t)) };
            }
            chain.pop();
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
            let reads = |w: VarId| self.reads_of(w);
            if let Some((v, src)) = ternary_assign(&e2, &reads) {
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
                let reads = |w: VarId| self.reads_of(w);
                if let Some((v, src)) = ternary_assign(&t2, &reads) {
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
        let mut fuel = crate::fuel::Fuel::new("structure.try_make_cond", crate::fuel::CAP_WALK);
        loop {
            if !fuel.burn() {
                return None;
            }
            let mut did = false;
            let ids: Vec<usize> = edges.iter().map(|e| e.0).collect();
            for ci in 0..edges.len() {
                let child = edges[ci].0;
                // the chain's head is nobody's child (a one-block loop branches to itself)
                if child == chain[0] {
                    continue;
                }
                let parents: Vec<usize> = (0..edges.len()).filter(|&p| edges[p].2 == child || edges[p].3 == child).collect();
                if parents.len() != 1 || parents[0] == ci {
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
        // a header that only steps the counter it tests (`subic. r7, r7, 1; bge`): `--n >= 0`
        if let ([Stmt::Assign { dst: Expr::Var(v), src: Expr::Binary { op, l: x, r: k, .. } }], Some(c)) = (self.blocks[h].stmts.as_slice(), self.blocks[h].cond.clone()) {
            let delta = match (op, k.as_int()) {
                (BinOp::Add, Some(1)) | (BinOp::Sub, Some(-1)) => Some(1),
                (BinOp::Sub, Some(1)) | (BinOp::Add, Some(-1)) => Some(-1),
                _ => None,
            };
            let v = *v;
            let mut uses = 0;
            c.walk(&mut |e| uses += matches!(e, Expr::Var(y) if *y == v) as usize);
            if let (Some(delta), true, 1) = (delta, matches!(**x, Expr::Var(y) if y == v), uses) {
                // (not the CTR counter: the CTR loop recovery reads its decrement)
                if matches!(strip_cv(&self.vars[v].ty), mwdec_core::Type::Int { .. }) && !self.vars[v].name.starts_with("var_ctr") {
                    let mut c2 = c;
                    c2.rewrite(&mut |e| {
                        if matches!(e, Expr::Var(y) if *y == v) {
                            *e = Expr::IncDec { e: Box::new(Expr::Var(v)), delta, post: false };
                        }
                    });
                    self.blocks[h].stmts.clear();
                    self.blocks[h].cond = Some(c2);
                }
            }
        }
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
        // (a chain that never leaves the body: every way out returns)
        let mut exit = if x == usize::MAX || x == self.exit_node() || l.body.contains(&x) { None } else { Some(x) };
        // a loop whose body returns (`for (...) { if (c) return x; }`) post-dominates nothing
        // but the return: its exit is where the loop test falls out, when every other way out
        // only returns
        // (an early return continuing inside the loop for post-dominance is no loop exit, unless
        // a latch's test leaves to it)
        let test_exit = |b: usize, me: &Self| -> Option<usize> {
            let (t, f) = me.cond_edges(b)?;
            let out = match (l.body.contains(&t), l.body.contains(&f)) {
                (true, false) => f,
                (false, true) => t,
                _ => return None,
            };
            if !l.latches.contains(&b) && me.cfg.pd_extra.iter().any(|&(r, c)| r == out && l.body.contains(&c)) {
                return None;
            }
            Some(out)
        };
        // a CTR loop's `bdnz` latch is the loop test; a test in the body leaving the loop is an
        // early exit (`for (...) { if (a[i] == x) return i; }`)
        let ctr_latch = l.latches.iter().copied().find(|&lt| {
            let mut ctr = false;
            if let Some(c) = &self.blocks[lt].cond {
                c.walk(&mut |e| ctr |= matches!(e, Expr::Var(v) if self.vars[*v].name.starts_with("var_ctr")));
            }
            ctr && test_exit(lt, self).is_some()
        });
        let first_exit = match ctr_latch {
            Some(lt) => test_exit(lt, self),
            None => test_exit(h, self).or_else(|| l.latches.iter().find_map(|&lt| test_exit(lt, self))),
        };
        if let Some(c) = first_exit {
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
        // a CTR loop is a counted loop whose tests in the body only break out of it
        let a2 = (|| {
            if ctr_latch.is_some() {
                return None;
            }
            let x = exit?;
            if self.has_stmts(h) {
                return None;
            }
            let mut chain: Vec<usize> = vec![];
            let mut conds: Vec<Expr> = vec![];
            let mut cur = h;
            let mut fuel = crate::fuel::Fuel::new("structure.loop_test_chain", crate::fuel::CAP_WALK);
            loop {
                if !fuel.burn() {
                    return None;
                }
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
                // MWCC lays the tests of `while (a && b)` out in order: each one falls into the
                // next (a test that branches back to the body top is the body's own `if`)
                let more = next != h
                    && next == f
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
        let a = a2.or_else(|| self.cond_edges(h).filter(|_| ctr_latch.is_none()).and_then(|(t, f)| {
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
        // the test at the top and an unconditional back edge: `for (;;) { if (!c) break; ... }`
        // (MWCC moves a `while`'s test to the bottom)
        let top_test = l.body.iter().all(|&b| self.cfg.blocks[b].start >= self.cfg.blocks[h].start)
            && l.latches.iter().all(|&lt| matches!(self.cfg.blocks[lt].term, Term::Jump(t) if t == h));
        // the loop test is an `&&`/`||` computed into a flag (`li r0,0; ...; li r0,1;` then
        // `clrlwi. r0; bne body`, an inlined `it != end`): the chain is the while condition
        let a = a.or_else(|| self.bool_loop_test(h, &l.body));
        if let Some((cond, body_start)) = a {
            self.emitted[h] = true;
            let mut body = vec![Stmt::Label(h)];
            if top_test {
                body.push(Stmt::If { cond: cond.clone().negate(self.vars), then: vec![Stmt::Break], els: vec![] });
            }
            self.build(body_start, Some(h), false, &mut body);
            strip_trailing_continue(&mut body);
            stmt = if top_test { Stmt::While { cond: Expr::Int { value: 1, ty: mwdec_core::Type::Bool }, body } } else { Stmt::While { cond, body } };
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
        // an empty case body MWCC kept as a dead `b end` before the other bodies sits there in
        // the source too
        let first_body = order.iter().filter(|&&t| t < self.cfg.blocks.len()).map(|&t| self.cfg.blocks[t].start).min().unwrap_or(usize::MAX);
        let join_off = join.and_then(|j| self.cfg.blocks.get(j)).map(|b| b.start);
        let dead_pos = (!self.insns.is_empty())
            .then(|| {
                self.cfg.blocks.iter().enumerate().filter(|(bi, bb)| {
                    self.cfg.idom[*bi] == usize::MAX
                        && bb.end == bb.start + 1
                        && bb.start < first_body
                        && ((self.insns[bb.start].is_jump() && self.insns[bb.start].reloc.is_none() && join_off.map(|o| self.insns[o].off) == self.insns[bb.start].target())
                            || (join.is_none() && self.ret_void && matches!(self.insns[bb.start].op(), ppc750cl::Opcode::Bclr) && self.insns[bb.start].ins.field_bo() & 0x14 == 0x14))
                }).map(|(_, bb)| bb.start).max()
            })
            .flatten();
        order.sort_by_key(|&t| if t < self.cfg.blocks.len() { self.cfg.blocks[t].start } else if t >= EXTRA_CASE { dead_pos.unwrap_or(usize::MAX) } else { usize::MAX });
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
            if t >= EXTRA_CASE && join.is_none() && self.ret_void && t != RET_LEAF {
                // an empty case of a switch ending the function: `return;` (MWCC keeps its `blr`)
                body.push(Stmt::Return(None));
            } else if Some(t) == join || t >= EXTRA_CASE {
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
        let mut fuel = crate::fuel::Fuel::new("structure.join_of_set", crate::fuel::CAP_WALK);
        loop {
            if !fuel.burn() {
                return None;
            }
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
        let nblocks = self.cfg.blocks.len();
        while let Some((mut b, set)) = work.pop() {
            guard += 1;
            if guard > 64 {
                return None;
            }
            // void leaf functions: a bare `blr` is the end of the switch
            let lone_blr = |b: usize| !self.insns.is_empty() && self.cfg.blocks[b].end == self.cfg.blocks[b].start + 1 && self.insns[self.cfg.blocks[b].start].is_blr();
            if b < nblocks && b != root && self.ret_void && matches!(self.cfg.blocks[b].term, Term::Return) && !self.has_stmts(b) && self.blocks[b].ret.is_none() && lone_blr(b) {
                if !nodes.contains(&b) {
                    nodes.push(b);
                }
                b = RET_LEAF;
            }
            if b >= nblocks {
                for (lo, hi) in set {
                    leaves.push((lo, hi, b));
                }
                continue;
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
            let (taken, fall) = match self.cfg.blocks[b].term {
                // `bgelr`: a tree leaf that returns
                Term::CondReturn { fall } if self.ret_void && self.blocks[b].ret.is_none() => (RET_LEAF, fall),
                _ => self.cond_edges(b)?,
            };
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
            // a test that doesn't split its values (the same compare again, as a destructor's own
            // `if (this)` after the caller's test) is no tree node of MWCC's
            if ts.is_empty() || fs.is_empty() {
                return None;
            }
            work.push((taken, ts));
            work.push((fall, fs));
        }
        let tree_nodes = nodes.iter().filter(|&&n| self.cond_edges(n).is_some() || matches!(self.cfg.blocks[n].term, Term::CondReturn { .. })).count();
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
                TreeCheck::Match(extra) => {
                    self.tree_confirmed.set(true);
                    cases.extend(extra)
                }
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
                    // a second unconditional branch in a row is dead code after the tree (an
                    // empty case body), not part of it
                    if matches!(items.last(), Some((o, TItem::B(_))) if *o + 4 == i.off) {
                        break;
                    }
                    let Some(t) = i.target() else { return TreeCheck::Unknown };
                    items.push((i.off, TItem::B(t)))
                }
                // a void function's returns inside the tree (`bgelr`, `blr` for the default)
                Opcode::Bclr if self.ret_void && i.ins.field_bi() < 4 => {
                    let (bo, bi) = (i.ins.field_bo(), i.ins.field_bi());
                    if bo & 0x14 == 0x14 {
                        if matches!(items.last(), Some((o, TItem::B(_))) if *o + 4 == i.off) {
                            break;
                        }
                        items.push((i.off, TItem::B(RET_OFF)));
                    } else {
                        let c = match (bo & 0x08 != 0, bi) {
                            (true, 2) => switchtree::Cond::Eq,
                            (false, 0) => switchtree::Cond::Ge,
                            (true, 0) => switchtree::Cond::Lt,
                            _ => break,
                        };
                        items.push((i.off, TItem::Bc(c, RET_OFF)));
                    }
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
        let def_off = if default == Some(RET_LEAF) { Some(RET_OFF) } else { default.and_then(off_of) };
        // (a void leaf's tree ending in returns: empty cases return too)
        let join_off = join.and_then(off_of).or((default == Some(RET_LEAF)).then_some(RET_OFF));
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
            // a dead `b join` right after the tree is the body of an empty case whose value
            // doesn't change the tree (`case 2: break;` after cases 0 and 1): MWCC threads the tree's
            // jumps past it but keeps the block
            if let (Some(j), Some(&(last_off, _))) = (join_off, items.last()) {
                let dead_b = self.cfg.blocks.iter().enumerate().any(|(bi, bb)| {
                    self.cfg.idom[bi] == usize::MAX
                        && bb.end == bb.start + 1
                        && self.insns[bb.start].off == last_off + 4
                        && self.insns[bb.start].is_jump()
                        && self.insns[bb.start].reloc.is_none()
                        && self.insns[bb.start].target() == Some(j)
                });
                if dead_b {
                    // MWCC folds an empty case next to a default range into that range (checked:
                    // `case 2: break;` after cases 0 and 1 leaves the tree as it is, `case 5:` or
                    // `case -1:` don't), so the value is the one after the largest case
                    let hi = base.iter().map(|c| c.0).max().unwrap_or(0);
                    return TreeCheck::Match(vec![(vec![hi + 1], EXTRA_CASE)]);
                }
            }
            return TreeCheck::Match(vec![]);
        }
        if join_off.is_none() && default.is_none() {
            return TreeCheck::NoMatch;
        }
        // empty-body case labels near the compared constants, or labels on the default's body
        // (`case 0: default:`)
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
        // non-negative values first (equivalent trees: the plainer label wins)
        cand.sort_by_key(|&v| (v < 0, v.abs()));
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
        // a dead `b end` after the tree is an empty case body: empty cases before labels on the
        // default's body
        // (a void leaf's tree: a dead lone `blr`)
        let dead_jump = join_off.map_or(false, |j| {
            self.cfg.blocks.iter().enumerate().any(|(bi, bb)| {
                self.cfg.idom[bi] == usize::MAX
                    && bb.end == bb.start + 1
                    && ((self.insns[bb.start].is_jump() && self.insns[bb.start].reloc.is_none() && self.insns[bb.start].target() == Some(j))
                        || (j == RET_OFF && self.insns[bb.start].is_blr() && matches!(self.insns[bb.start].op(), ppc750cl::Opcode::Bclr) && self.insns[bb.start].ins.field_bo() & 0x14 == 0x14))
            })
        });
        for sub in &subsets {
            if let (Some(def), false) = (default, dead_jump) {
                let on_default: Vec<(i64, Lab)> = sub.iter().map(|&v| (v, Lab::Default)).collect();
                if try_set(&on_default) {
                    return TreeCheck::Match(vec![(sub.clone(), def)]);
                }
            }
            if join_off.is_none() {
                continue;
            }
            // one shared empty body (`case a: case b: break;`) or one per value
            let shared: Vec<(i64, Lab)> = sub.iter().map(|&v| (v, Lab::Extra(0))).collect();
            if try_set(&shared) {
                // the kept dead `b end` also takes the value after the largest case (folded into
                // the default range on its own, see above)
                if dead_jump {
                    let hi = base.iter().map(|c| c.0).max().unwrap_or(0) + 1;
                    // (the tree simulation doesn't model that fold: no re-check)
                    if !sub.contains(&hi) && !used.contains(&hi) {
                        let mut vs = sub.clone();
                        vs.push(hi);
                        vs.sort();
                        return TreeCheck::Match(vec![(vs, EXTRA_CASE)]);
                    }
                }
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

/// `lis rT, hi; [addi|ori rT, rX, lo]; cmpw x, rT` for constants beyond 16 bits (the
/// `lis` may be shared: `addi r0, r4, 4`)
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

/// m2c's reduction of a DAG of tests `(node, cond for taken, taken, fall)` rooted at `root` into
/// one `&&`/`||` condition: (cond for taken, taken, fall) over the non-node targets.
fn reduce_cond_dag(mut edges: Vec<(usize, Expr, usize, usize)>, root: usize, vars: &[Var]) -> Option<(Expr, usize, usize)> {
    let mut fuel = crate::fuel::Fuel::new("structure.reduce_cond_dag", crate::fuel::CAP_WALK);
    loop {
        if !fuel.burn() {
            return None;
        }
        let mut did = false;
        for ci in 0..edges.len() {
            let child = edges[ci].0;
            if child == root {
                continue;
            }
            let parents: Vec<usize> = (0..edges.len()).filter(|&p| edges[p].2 == child || edges[p].3 == child).collect();
            if parents.len() != 1 {
                continue;
            }
            let pi = parents[0];
            let (_, ref pc, pt, pf) = edges[pi];
            let (_, ref cc, ct, cf) = edges[ci];
            let (nc, nt, nf) = if pt == ct && pf == child {
                (Expr::cmp(BinOp::LogOr, pc.clone(), cc.clone()), pt, cf)
            } else if pt == cf && pf == child {
                (Expr::cmp(BinOp::LogOr, pc.clone(), cc.clone().negate(vars)), pt, ct)
            } else if pt == child && pf == ct {
                (Expr::cmp(BinOp::LogAnd, pc.clone(), cc.clone().negate(vars)), cf, pf)
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
    if edges.len() != 1 || edges[0].0 != root {
        return None;
    }
    let (_, c, t, f) = edges.remove(0);
    Some((c, t, f))
}

/// A statement list that only assigns one variable, possibly through an if/else whose arms both
/// assign it: (variable, value as a (nested) ternary).
fn ternary_assign(stmts: &[Stmt], reads: &dyn Fn(VarId) -> usize) -> Option<(VarId, Expr)> {
    let once = |w: VarId| reads(w) == 1;
    let s: Vec<&Stmt> = stmts.iter().filter(|s| !matches!(s, Stmt::Label(_))).collect();
    // a temp computed first and only read by the selected value (`t = a + b / c; v = v < t ? v : t;`)
    if let [Stmt::Assign { dst: Expr::Var(t), src }, rest @ ..] = s.as_slice() {
        if !rest.is_empty() && !src.has_call() {
            let rest: Vec<Stmt> = rest.iter().map(|x| (*x).clone()).collect();
            if let Some((v, mut e)) = ternary_assign(&rest, reads) {
                let mut n = 0;
                e.walk(&mut |x| n += matches!(x, Expr::Var(y) if y == t) as usize);
                if v != *t && n > 0 && n == reads(*t) && !src.uses_var(v) {
                    e.rewrite(&mut |x| {
                        if matches!(x, Expr::Var(y) if y == t) {
                            *x = src.clone();
                        }
                    });
                    return Some((v, e));
                }
            }
        }
    }
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
        // a materialised `&&`/`||` tested once (`b = x || y; if (b) v = p; else v = q;`): MWCC
        // evaluates a ternary's logical condition as a value, `v = (x || y) ? p : q`
        [Stmt::Assign { dst: Expr::Var(w), src: c @ Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, .. } }, Stmt::If { cond, then, els }]
            if tested_var(cond) == Some(*w) && once(*w) && !then.is_empty() && !els.is_empty() =>
        {
            let (a, ea) = ternary_assign(then, reads)?;
            let (b, eb) = ternary_assign(els, reads)?;
            if a != b {
                return None;
            }
            Some((a, Expr::Ternary { c: Box::new(c.clone()), t: Box::new(ea), f: Box::new(eb), ty: mwdec_core::Type::Unknown { size: 4 } }))
        }
        [Stmt::If { cond, then, els }] if !then.is_empty() && !els.is_empty() => {
            let (a, ea) = ternary_assign(then, reads)?;
            let (b, eb) = ternary_assign(els, reads)?;
            if a != b {
                return None;
            }
            Some((a, Expr::Ternary { c: Box::new(cond.clone()), t: Box::new(ea), f: Box::new(eb), ty: mwdec_core::Type::Unknown { size: 4 } }))
        }
        _ => None,
    }
}

/// The variable a test compares (`v == 0`, `(T)v != 0`, `v`).
fn tested_var_any(e: &Expr) -> Option<VarId> {
    match e {
        Expr::Var(v) => Some(*v),
        Expr::Cast { e, .. } => tested_var_any(e),
        Expr::Unary { op: UnOp::Not, e, .. } => tested_var_any(e),
        Expr::Binary { op: BinOp::Ne | BinOp::Eq, l, r, .. } if r.as_int() == Some(0) => tested_var_any(l),
        _ => None,
    }
}

/// `v`, `(T)v`, `v != 0`: the variable a condition tests for truth.
fn tested_var(e: &Expr) -> Option<VarId> {
    match e {
        Expr::Var(v) => Some(*v),
        Expr::Cast { e, .. } => tested_var(e),
        Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => tested_var(l),
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

/// Does control never fall out of the end of this statement list?
fn diverges(b: &[Stmt]) -> bool {
    match b.iter().rev().find(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))) {
        Some(Stmt::Return(_) | Stmt::Goto(_) | Stmt::Break | Stmt::Continue) => true,
        Some(Stmt::If { then, els, .. }) => !then.is_empty() && !els.is_empty() && diverges(then) && diverges(els),
        // every case (default included) leaves; a `break` leaves only the switch
        Some(Stmt::Switch { cases, .. }) => {
            cases.iter().any(|c| c.is_default) && cases.iter().all(|c| diverges(&c.body) && !matches!(c.body.iter().rev().find(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))), Some(Stmt::Break)))
        }
        _ => false,
    }
}

fn strip_trailing_continue(body: &mut Vec<Stmt>) {
    while matches!(body.last(), Some(Stmt::Continue)) {
        body.pop();
    }
}

/// MWCC swaps the arms of `if (!x) return a; return b;` (each arm a single return) into `if (x)
/// return b; return a;`, so a structured `if (!x)` over two returns would lose its layout;
/// `x == 0` keeps it (that form is not swapped).
pub fn guard_not_swap(body: &mut Vec<Stmt>) {
    Stmt::for_each_block_mut(body, &mut |blk| {
        let real: Vec<usize> = (0..blk.len()).filter(|&i| !matches!(blk[i], Stmt::Label(_) | Stmt::Comment(_))).collect();
        for (k, &i) in real.iter().enumerate() {
            let next_ret = real.get(k + 1).map_or(false, |&j| matches!(blk[j], Stmt::Return(Some(_))));
            let Stmt::If { cond, then, els } = &mut blk[i] else { continue };
            let single_ret = |v: &Vec<Stmt>| {
                let r: Vec<&Stmt> = v.iter().filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))).collect();
                matches!(r.as_slice(), [Stmt::Return(Some(_))])
            };
            let els_empty = els.iter().all(|s| matches!(s, Stmt::Label(_) | Stmt::Comment(_)));
            // (the same swap for two single assignments of one variable: `if (!p) v = a; else v = b;`)
            let single_assign = |v: &Vec<Stmt>| -> Option<VarId> {
                let r: Vec<&Stmt> = v.iter().filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))).collect();
                match r.as_slice() {
                    [Stmt::Assign { dst: Expr::Var(x), .. }] => Some(*x),
                    _ => None,
                }
            };
            let assigns = single_assign(then).is_some() && single_assign(then) == single_assign(els);
            if !assigns && (!single_ret(then) || !(single_ret(els) || (els_empty && next_ret))) {
                continue;
            }
            if let Expr::Unary { op: UnOp::Not, e, .. } = cond {
                if matches!(**e, Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, .. }) {
                    continue;
                }
                let x = (**e).clone();
                let zero = Expr::Int { value: 0, ty: t_s32() };
                *cond = Expr::cmp(BinOp::Eq, x, zero);
            }
        }
    });
}

/// A loop that only re-tests memory (`while (p->done == 0) {}`) reloads it every iteration in
/// the target: MWCC would hoist a plain load out of the loop, so the source read is volatile.
pub fn volatile_spin_loads(body: &mut Vec<Stmt>) {
    Stmt::for_each_block_mut(body, &mut |blk| {
        for s in blk.iter_mut() {
            let (Stmt::While { cond, body } | Stmt::DoWhile { body, cond }) = s else { continue };
            if body.iter().any(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))) || cond.has_call() {
                continue;
            }
            if matches!(cond, Expr::Int { .. }) {
                continue;
            }
            cond.rewrite(&mut |e| {
                if let Expr::Load { ty, .. } = e {
                    if !matches!(ty, mwdec_core::Type::Volatile(_)) {
                        *ty = mwdec_core::Type::Volatile(Box::new(ty.clone()));
                    }
                }
            });
        }
    });
}

/// A CTR loop counting an index up from a start of its own: `ctr = X - v; if (v < X) { do {
/// B; v = v + 1; ctr--; } while (ctr != 0); }` is `for (; v < X; v++) { B }` (MWCC derives the
/// trip count `X - v` from the test).
pub fn offset_ctr_loops(body: &mut Vec<Stmt>, vars: &[Var]) {
    fn is_dec_of(s: &Stmt, c: VarId, k: i64) -> bool {
        matches!(s, Stmt::Assign { dst: Expr::Var(x), src: Expr::Binary { op, l, r, .. } }
            if *x == c && matches!(**l, Expr::Var(y) if y == c) && ((*op == BinOp::Sub && r.as_int() == Some(k)) || (*op == BinOp::Add && r.as_int() == Some(-k))))
    }
    let uses = |b: &[Stmt], v: VarId| {
        let mut n = 0;
        Stmt::walk_exprs(b, &mut |e| n += matches!(e, Expr::Var(x) if *x == v) as usize);
        n
    };
    let whole = body.clone();
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut k = 0;
        while k < b.len() {
            let Stmt::If { cond, then, els } = &b[k] else {
                k += 1;
                continue;
            };
            if !els.is_empty() || then.len() != 1 {
                k += 1;
                continue;
            }
            let Stmt::DoWhile { body: lb, cond: lc } = &then[0] else {
                k += 1;
                continue;
            };
            // the counter and its initialisation `ctr = X - v` earlier in this list
            let ctr = match lc {
                Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => match strip_casts((**l).clone()) {
                    Expr::Var(c) if vars[c].name.starts_with("var_ctr") => c,
                    _ => {
                        k += 1;
                        continue;
                    }
                },
                _ => {
                    k += 1;
                    continue;
                }
            };
            let Some(ip) = b[..k].iter().rposition(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if *x == ctr)) else {
                k += 1;
                continue;
            };
            let Stmt::Assign { src: Expr::Binary { op: BinOp::Sub, l: bound, r: start, .. }, .. } = &b[ip] else {
                k += 1;
                continue;
            };
            let Expr::Var(v) = strip_casts((**start).clone()) else {
                k += 1;
                continue;
            };
            // the guard is the loop test `v < X`
            let guard_ok = matches!(cond, Expr::Binary { op: BinOp::Lt, l, r, .. }
                if strip_casts((**l).clone()) == Expr::Var(v) && strip_casts((**r).clone()) == strip_casts((**bound).clone()));
            let n = lb.len();
            let tail_ok = n >= 2 && is_dec_of(&lb[n - 1], ctr, 1) && is_dec_of(&lb[n - 2], v, -1);
            // the counter is only initialised, decremented and tested
            let ctr_uses = uses(&whole, ctr);
            if !guard_ok || !tail_ok || ctr_uses != 4 || b[ip + 1..k].iter().any(|s| uses(std::slice::from_ref(s), v) > 0 && matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if *x == v)) {
                k += 1;
                continue;
            }
            let mut lb2 = lb.clone();
            lb2.truncate(n - 2);
            let cond2 = cond.clone();
            b[k] = Stmt::For {
                init: vec![],
                cond: cond2,
                step: vec![Stmt::Expr(Expr::IncDec { e: Box::new(Expr::Var(v)), delta: 1, post: true })],
                body: lb2,
            };
            b.remove(ip);
            k = k.saturating_sub(1);
        }
    });
}

/// A CTR loop whose body breaks out: `ctr = n; if (n > 0) { while (1) { if (c) { B; ctr--; if
/// (ctr != 0) continue; } else { X } break; } }` is `for (i = 0; i < n; i++) { if (!c) { X;
/// break; } B }` (likewise with the arms the other way round).
pub fn ctr_break_loops(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
    let mut new_vars: Vec<Var> = vec![];
    let base = vars.len();
    let vars_ro: &[Var] = vars;
    let uses_of = |b: &[Stmt], v: VarId| {
        let mut n = 0;
        Stmt::walk_exprs(b, &mut |e| n += matches!(e, Expr::Var(x) if *x == v) as usize);
        n
    };
    let whole = body.clone();
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut k = 0;
        while k < b.len() {
            let found = (|| {
                let Stmt::If { cond: g, then, els } = &b[k] else { return None };
                if !els.is_empty() {
                    return None;
                }
                let [Stmt::While { cond: Expr::Int { value: 1, .. }, body: lb }] = then.as_slice() else { return None };
                let [Stmt::If { cond: c, then: t0, els: e0 }, Stmt::Break] = lb.as_slice() else { return None };
                // the arm that stays in the loop ends `ctr--; if (ctr != 0) continue;`; the other
                // arm (`X`, maybe empty) leaves it: `if (exit) { X; break; }`
                let stays = |a: &[Stmt]| a.len() >= 2 && matches!(a.last(), Some(Stmt::If { then, els, .. }) if then.as_slice() == [Stmt::Continue] && els.is_empty());
                let (t, x, exit) = if stays(t0) {
                    (t0, e0, c.clone().negate(vars_ro))
                } else if stays(e0) {
                    (e0, t0, c.clone())
                } else {
                    return None;
                };
                let n = t.len();
                let Stmt::If { cond: lc, then: lt, els: le } = &t[n - 1] else { return None };
                if lt.as_slice() != [Stmt::Continue] || !le.is_empty() {
                    return None;
                }
                let ctr = match lc {
                    Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => match strip_casts((**l).clone()) {
                        Expr::Var(v) if vars_ro[v].name.starts_with("var_ctr") => v,
                        _ => return None,
                    },
                    _ => return None,
                };
                let dec_ok = matches!(&t[n - 2], Stmt::Assign { dst: Expr::Var(x), src: Expr::Binary { op: BinOp::Sub, l, r, .. } }
                    if *x == ctr && matches!(**l, Expr::Var(y) if y == ctr) && r.as_int() == Some(1));
                if !dec_ok || uses_of(&whole, ctr) != 4 {
                    return None;
                }
                let ip = b[..k].iter().rposition(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if *x == ctr))?;
                let Stmt::Assign { src: count, .. } = &b[ip] else { return None };
                let signed = match g {
                    Expr::Binary { op: BinOp::Gt, l, r, .. } if r.as_int() == Some(0) && strip_casts((**l).clone()) == strip_casts(count.clone()) => !matches!(**l, Expr::Cast { ty: mwdec_core::Type::Int { signed: false, .. }, .. }),
                    Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) && strip_casts((**l).clone()) == strip_casts(count.clone()) => false,
                    _ => return None,
                };
                let mut leave = x.clone();
                leave.push(Stmt::Break);
                let mut inner = vec![Stmt::If { cond: exit, then: leave, els: vec![] }];
                inner.extend(t[..n - 2].iter().cloned());
                Some((ip, count.clone(), signed, inner))
            })();
            let Some((ip, count, signed, inner)) = found else {
                k += 1;
                continue;
            };
            let ty = mwdec_core::Type::Int { size: 4, signed };
            let i = base + new_vars.len();
            new_vars.push(Var { name: "i".into(), ty: ty.clone(), kind: VarKind::Local });
            let bound = if signed { count } else { Expr::cast(ty.clone(), count) };
            b[k] = Stmt::For {
                init: vec![Stmt::Assign { dst: Expr::Var(i), src: Expr::Int { value: 0, ty: ty.clone() } }],
                cond: Expr::cmp(BinOp::Lt, Expr::Var(i), bound),
                step: vec![Stmt::Expr(Expr::IncDec { e: Box::new(Expr::Var(i)), delta: 1, post: true })],
                body: inner,
            };
            b.remove(ip);
            k = k.saturating_sub(1);
        }
    });
    for v in new_vars {
        vars.push(v);
        is_temp.push(false);
    }
}

/// `while (1) { if (c) { S; return x; } B }` (the other arm `B; continue;` or after the `if`), with
/// no other way out of the loop, is `while (!c) { B } S; return x;` (variant): the source had the
/// exit code after a bottom-tested loop (a top-tested one keeps the return in the loop).
pub fn loop_exit_returns_after(body: &mut Vec<Stmt>, vars: &[Var]) {
    fn labels(b: &[Stmt]) -> bool {
        let mut any = false;
        for s in b {
            match s {
                Stmt::Label(_) | Stmt::Goto(_) => any = true,
                Stmt::If { then, els, .. } => any |= labels(then) || labels(els),
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => any |= labels(body),
                Stmt::For { init, step, body, .. } => any |= labels(init) || labels(step) || labels(body),
                Stmt::Switch { cases, .. } => any |= cases.iter().any(|c| labels(&c.body)),
                _ => {}
            }
        }
        any
    }
    Stmt::for_each_block_mut(body, &mut |blk| {
        let mut k = 0;
        while k < blk.len() {
            let found = (|| {
                let Stmt::While { cond: Expr::Int { value: 1, .. }, body: lb } = &blk[k] else { return None };
                let (Stmt::If { cond: c, then: t, els: e }, rest) = lb.split_first()? else { return None };
                if !matches!(t.last(), Some(Stmt::Return(_))) || crate::ctrloop::has_own_jump(t, false) || labels(lb) {
                    return None;
                }
                // the arm that stays: `B; continue;` (nothing after the if) or `B` then the rest
                let mut stay: Vec<Stmt> = e.clone();
                if matches!(stay.last(), Some(Stmt::Continue)) {
                    stay.pop();
                    if !rest.is_empty() {
                        return None;
                    }
                } else if !e.is_empty() && rest.is_empty() {
                    return None;
                }
                stay.extend(rest.iter().cloned());
                if crate::ctrloop::has_own_jump(&stay, false) {
                    return None;
                }
                if !crate::variants::alt(crate::variants::LOOP_EXIT_AFTER) {
                    return None;
                }
                Some((c.clone().negate(vars), stay, t.clone()))
            })();
            let Some((cond, stay, exit)) = found else {
                k += 1;
                continue;
            };
            blk[k] = Stmt::While { cond, body: stay };
            let n = exit.len();
            for (j, s) in exit.into_iter().enumerate() {
                blk.insert(k + 1 + j, s);
            }
            k += 1 + n;
        }
    });
}

/// Warning text of `invariant_loop_conditions` (eval rows flag drafts carrying it).
pub const WARN_INVARIANT_LOOP: &str = "condition never changes in the loop";

/// Loops whose condition reads only variables nothing in the loop changes (no memory, no call)
/// and that are not `while (1)`: a source loop never looks like that (it would not end), so it is
/// a structuring or loop-recovery error. Returns one line per such loop.
pub fn invariant_loop_conditions(body: &[Stmt]) -> Vec<String> {
    fn assigned(b: &[Stmt], out: &mut Vec<VarId>) {
        for s in b {
            match s {
                Stmt::Assign { dst: Expr::Var(x), .. } => out.push(*x),
                Stmt::If { then, els, .. } => {
                    assigned(then, out);
                    assigned(els, out);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => assigned(body, out),
                Stmt::For { init, step, body, .. } => {
                    assigned(init, out);
                    assigned(step, out);
                    assigned(body, out);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        assigned(&c.body, out);
                    }
                }
                _ => {}
            }
        }
        Stmt::walk_exprs(b, &mut |e| {
            if let Expr::IncDec { e: x, .. } = e {
                if let Expr::Var(v) = **x {
                    out.push(v);
                }
            }
            // `&v` passed anywhere may change v
            if let Expr::AddrOf(x) = e {
                if let Expr::Var(v) = **x {
                    out.push(v);
                }
            }
        });
    }
    fn check(cond: &Expr, parts: &[&[Stmt]], what: &str, out: &mut Vec<String>) {
        if matches!(cond, Expr::Int { .. }) {
            return;
        }
        let mut opaque = false;
        let mut vs = vec![];
        cond.walk(&mut |e| match e {
            Expr::Var(v) => vs.push(*v),
            // (an unknown value, like a time base read, may change on its own)
            Expr::Load { .. } | Expr::Global { .. } | Expr::Member { .. } | Expr::Index { .. } | Expr::Call { .. } | Expr::IncDec { .. } | Expr::BitField { .. } | Expr::New { .. } | Expr::Unknown { .. } => opaque = true,
            _ => {}
        });
        if opaque || vs.is_empty() {
            return;
        }
        let mut a = vec![];
        for p in parts {
            assigned(p, &mut a);
        }
        if !vs.iter().any(|v| a.contains(v)) {
            out.push(format!("{what} {WARN_INVARIANT_LOOP}"));
        }
    }
    let mut out = vec![];
    fn walk(b: &[Stmt], out: &mut Vec<String>) {
        for s in b {
            match s {
                Stmt::While { cond, body } => {
                    check(cond, &[body], "while", out);
                    walk(body, out);
                }
                Stmt::DoWhile { body, cond } => {
                    check(cond, &[body], "do-while", out);
                    walk(body, out);
                }
                Stmt::For { cond, step, body, .. } => {
                    check(cond, &[body, step], "for", out);
                    walk(body, out);
                }
                Stmt::If { then, els, .. } => {
                    walk(then, out);
                    walk(els, out);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        walk(&c.body, out);
                    }
                }
                _ => {}
            }
        }
    }
    walk(body, &mut out);
    out
}
