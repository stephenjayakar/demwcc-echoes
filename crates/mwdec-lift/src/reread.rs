//! Locals holding a pure memory read (`t = this->a; ... t ...`) are often MWCC's CSE of reads
//! the source repeated (`this->a ... this->a`); the two compile differently (register choice,
//! inline argument evaluation). Offered as a draft variant: re-read the value at every use.
use crate::ir::*;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq)]
enum Ev {
    Def(VarId),
    Use(VarId),
    Effect,
    /// a store through the object a variable points to (None: another lvalue)
    Store(Option<VarId>),
}

/// The variable at the root of an lvalue's address (`this->a.b`, `p[i].c`, `*(T*)&o`).
fn root_var(e: &Expr) -> Option<VarId> {
    match e {
        Expr::Var(v) => Some(*v),
        Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } | Expr::Index { base, .. } => root_var(base),
        Expr::AddrOf(x) | Expr::Cast { e: x, .. } => root_var(x),
        _ => None,
    }
}

type Path = Vec<(usize, u8)>;

struct Evs {
    list: Vec<(Ev, bool, Path)>,
    path: Path,
    next_if: usize,
    /// the outermost loop of each event
    loops: Vec<Option<usize>>,
    cur_loop: Option<usize>,
    next_loop: usize,
}

impl Evs {
    fn push(&mut self, e: Ev, l: bool) {
        self.list.push((e, l, self.path.clone()));
        self.loops.push(self.cur_loop);
    }

    /// Run `f` inside a loop (numbered when it is an outermost one).
    fn in_loop(&mut self, f: impl FnOnce(&mut Evs)) {
        let outer = self.cur_loop;
        if outer.is_none() {
            self.cur_loop = Some(self.next_loop);
            self.next_loop += 1;
        }
        f(self);
        self.cur_loop = outer;
    }
}

/// Two events in different arms of one `if` never run one after the other.
fn exclusive(a: &Path, b: &Path) -> bool {
    a.iter().any(|(i, x)| b.iter().any(|(j, y)| i == j && x != y))
}

fn expr_events(e: &Expr, in_loop: bool, out: &mut Evs) {
    let mut effect = false;
    let mut uses = vec![];
    e.walk(&mut |x| {
        if let Expr::Var(v) = x {
            uses.push(*v);
        }
        if matches!(x, Expr::Call { .. } | Expr::New { .. } | Expr::IncDec { .. }) && !x.is_pure_call() {
            effect = true;
        }
    });
    for v in uses {
        out.push(Ev::Use(v), in_loop);
    }
    if effect {
        out.push(Ev::Effect, in_loop);
    }
}

fn events(b: &[Stmt], in_loop: bool, out: &mut Evs) {
    for s in b {
        match s {
            Stmt::Assign { dst: Expr::Var(v), src } => {
                expr_events(src, in_loop, out);
                out.push(Ev::Def(*v), in_loop);
            }
            Stmt::Assign { dst, src } => {
                expr_events(src, in_loop, out);
                expr_events(dst, in_loop, out);
                out.push(Ev::Store(root_var(dst)), in_loop);
            }
            Stmt::Expr(e) | Stmt::Return(Some(e)) => expr_events(e, in_loop, out),
            Stmt::If { cond, then, els } => {
                expr_events(cond, in_loop, out);
                let id = out.next_if;
                out.next_if += 1;
                out.path.push((id, 0));
                events(then, in_loop, out);
                out.path.pop();
                out.path.push((id, 1));
                events(els, in_loop, out);
                out.path.pop();
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => out.in_loop(|out| {
                expr_events(cond, true, out);
                events(body, true, out);
            }),
            Stmt::For { init, cond, step, body } => out.in_loop(|out| {
                events(init, true, out);
                expr_events(cond, true, out);
                events(step, true, out);
                events(body, true, out);
            }),
            Stmt::Switch { e, cases } => {
                expr_events(e, in_loop, out);
                for c in cases {
                    events(&c.body, in_loop, out);
                }
            }
            // control leaving the path: later code may run after either arm
            Stmt::Goto(_) | Stmt::Label(_) => out.push(Ev::Effect, true),
            _ => {}
        }
    }
}

fn has_load(e: &Expr) -> bool {
    let mut f = false;
    e.walk(&mut |x| f |= matches!(x, Expr::Load { .. } | Expr::Global { .. } | Expr::Member { .. } | Expr::Index { .. }));
    f
}

/// Candidates: (var, its value).
fn candidates(body: &[Stmt], vars: &[Var]) -> Vec<(VarId, Expr)> {
    let mut evs = Evs { list: vec![], path: vec![], next_if: 0, loops: vec![], cur_loop: None, next_loop: 0 };
    events(body, false, &mut evs);
    let loops = evs.loops;
    let ev: Vec<(Ev, bool, Path)> = evs.list;
    // a use in a loop sees the value read before it on every iteration when nothing in the loop
    // stores or calls
    let calm_loop = |u: usize| loops[u].is_some_and(|l| !(0..ev.len()).any(|k| loops[k] == Some(l) && matches!(ev[k].0, Ev::Effect | Ev::Store(_))));
    let mut defs: HashMap<VarId, usize> = HashMap::new();
    for (e, _, _) in &ev {
        if let Ev::Def(v) = e {
            *defs.entry(*v).or_insert(0) += 1;
        }
    }
    let mut srcs: HashMap<VarId, Expr> = HashMap::new();
    fn collect(b: &[Stmt], out: &mut HashMap<VarId, Expr>) {
        for s in b {
            match s {
                Stmt::Assign { dst: Expr::Var(v), src } => {
                    out.insert(*v, src.clone());
                }
                Stmt::If { then, els, .. } => {
                    collect(then, out);
                    collect(els, out);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        collect(&c.body, out);
                    }
                }
                _ => {}
            }
        }
    }
    collect(body, &mut srcs);
    let mut out = vec![];
    for (&v, src) in &srcs {
        if defs.get(&v) != Some(&1) || !matches!(vars[v].kind, VarKind::Local) || src.has_call() || !has_load(src) {
            continue;
        }
        // operands that never change
        let mut stable = true;
        src.walk(&mut |x| {
            if let Expr::Var(y) = x {
                stable &= defs.get(y).copied().unwrap_or(0) == 0 || (defs.get(y) == Some(&1) && *y != v);
            }
        });
        if !stable {
            continue;
        }
        let Some(d) = ev.iter().position(|(e, _, _)| *e == Ev::Def(v)) else { continue };
        if ev[d].1 {
            continue;
        }
        let uses: Vec<usize> = ev.iter().enumerate().filter(|(_, (e, _, _))| *e == Ev::Use(v)).map(|(i, _)| i).collect();
        if uses.len() < 2 || uses.iter().any(|&u| u < d || (ev[u].1 && !calm_loop(u)) || exclusive(&ev[u].2, &ev[d].2)) {
            continue;
        }
        // no call can run between the read and any use, nor a store that may reach what it
        // reads: a store through `this` or a parameter leaves reads through the others (MWCC
        // keeps such a read across it)
        let roots: Vec<VarId> = {
            let mut r = vec![];
            src.walk(&mut |x| {
                if let Expr::Var(y) = x {
                    r.push(*y);
                }
            });
            r
        };
        let objectish = |w: VarId| matches!(vars[w].kind, VarKind::This | VarKind::Param { .. });
        let may_reach = |k: usize| match ev[k].0 {
            Ev::Effect => true,
            Ev::Store(Some(r)) => !(objectish(r) && roots.iter().all(|&x| objectish(x) && x != r)),
            Ev::Store(None) => true,
            _ => false,
        };
        let blocked = uses.iter().any(|&u| (d..u).any(|k| may_reach(k) && !exclusive(&ev[k].2, &ev[u].2)));
        if blocked {
            continue;
        }
        out.push((v, src.clone()));
    }
    out.sort_by_key(|c| c.0);
    out
}

/// `v = E; *p = v;` at the very end of the function (nothing reads `v` afterwards): `*p = E`.
fn forward_final_stores(b: &mut Vec<Stmt>, tail: bool) -> bool {
    let mut changed = false;
    let n = b.len();
    for j in 0..n {
        let t = tail && j + 1 == n;
        match &mut b[j] {
            Stmt::If { then, els, .. } => {
                changed |= forward_final_stores(then, t);
                changed |= forward_final_stores(els, t);
            }
            _ => {}
        }
    }
    if tail && n >= 2 {
        if let (Stmt::Assign { dst: Expr::Var(v), src: e }, Stmt::Assign { dst, src: Expr::Var(w) }) = (&b[n - 2], &b[n - 1]) {
            if v == w && !matches!(dst, Expr::Var(_)) && !e.has_call() && !dst.uses_var(*v) {
                let (dst, e) = (dst.clone(), e.clone());
                b.truncate(n - 2);
                b.push(Stmt::Assign { dst, src: e });
                changed = true;
            }
        }
    }
    changed
}

/// Ask the draft variant when there is something to re-read; substitute when it is taken.
pub fn reread_temps(body: &mut Vec<Stmt>, vars: &[Var]) {
    let mut trial = body.clone();
    forward_final_stores(&mut trial, true);
    let cands = candidates(&trial, vars);
    if cands.is_empty() || !crate::variants::alt(crate::variants::REREAD_TEMPS) {
        return;
    }
    *body = trial;
    for (v, src) in cands {
        Stmt::for_each_block_mut(body, &mut |b| b.retain(|s| !matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if *x == v)));
        Stmt::rewrite_exprs(body, &mut |e| {
            if matches!(e, Expr::Var(x) if *x == v) {
                *e = src.clone();
            }
        });
    }
}
