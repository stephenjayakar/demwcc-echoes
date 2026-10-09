//! The rewriting traversal. Statements are visited in order with a map of the locals whose
//! current value is a known pure expression (reaching definitions), so a register reused for
//! several values (`p = &data.mLayout` kept live across inlined accessors) is still looked
//! through where only one definition reaches.

use crate::matcher::{expand, scalar_shallow, Defs, Env, Index};
use crate::InlineLib;
use mwdec_core::TypeDb;
use mwdec_lift::{Expr, Stmt, Var, VarId, VarKind};

pub struct Ctx<'a> {
    pub db: &'a TypeDb,
    pub vars: &'a [Var],
    /// single-definition temps valid everywhere (look-through safe)
    pub global: &'a Defs,
    pub lib: &'a InlineLib,
    pub idx: &'a Index,
    /// the body at the start of the round (use counts)
    pub whole: &'a [Stmt],
}

fn assigned_deep(b: &[Stmt], out: &mut Vec<VarId>) {
    for s in b {
        match s {
            Stmt::Assign { dst: Expr::Var(v), .. } => out.push(*v),
            Stmt::If { then, els, .. } => {
                assigned_deep(then, out);
                assigned_deep(els, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => assigned_deep(body, out),
            Stmt::For { init, step, body, .. } => {
                assigned_deep(init, out);
                assigned_deep(step, out);
                assigned_deep(body, out);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    assigned_deep(&c.body, out);
                }
            }
            _ => {}
        }
    }
}

fn merged(global: &Defs, reach: &Defs) -> Defs {
    // (a reaching value of a single-definition temp is its definition in terms of the values
    // that reached it: see `update`)
    let mut d = global.clone();
    for (k, v) in reach {
        d.insert(*k, v.clone());
    }
    d
}

fn kill(reach: &mut Defs, vs: &[VarId]) {
    if vs.is_empty() {
        return;
    }
    reach.retain(|k, e| !vs.contains(k) && !vs.iter().any(|v| e.uses_var(*v)));
}

/// Update the reaching map with the effects of statement `s`.
fn update(reach: &mut Defs, s: &Stmt, cx: &Ctx) {
    let mut asg = vec![];
    assigned_deep(std::slice::from_ref(s), &mut asg);
    kill(reach, &asg);
    // memory-reading values clobbered by the statement
    if !reach.is_empty() {
        let defs = merged(cx.global, reach);
        let env = Env { db: cx.db, vars: cx.vars, defs: &defs, lib: cx.lib, objects: &cx.idx.objects };
        let dead: Vec<VarId> = reach
            .iter()
            .filter(|(_, e)| {
                let mut rd = vec![];
                crate::safety::reads(&expand(e, &defs), &env, &mut rd);
                !rd.is_empty() && crate::safety::clobbers(s, &rd, &env)
            })
            .map(|(k, _)| *k)
            .collect();
        for k in dead {
            reach.remove(&k);
        }
    }
    if let Stmt::Assign { dst: Expr::Var(v), src } = s {
        // a single-definition temp computed from a reassigned one's current value (`end = begin +
        // n` before the walk moves `begin`): its value in terms of what reached it
        if cx.global.contains_key(v) && !src.has_call() {
            let mut uses_reached = false;
            src.walk(&mut |x| {
                if let Expr::Var(w) = x {
                    uses_reached |= reach.contains_key(w) && !cx.global.contains_key(w);
                }
            });
            if uses_reached {
                let mut val = src.clone();
                val.rewrite(&mut |x| {
                    if let Expr::Var(w) = x {
                        if let Some(d) = reach.get(w).filter(|_| !cx.global.contains_key(w)) {
                            *x = d.clone();
                        }
                    }
                });
                reach.insert(*v, val);
            }
        }
        if matches!(cx.vars[*v].kind, VarKind::Local) && !cx.global.contains_key(v) && !src.has_call() && !src.uses_var(*v) {
            // no cycles through other known values
            let defs = merged(cx.global, reach);
            if !expand(src, &defs).uses_var(*v) {
                reach.insert(*v, src.clone());
            }
        }
    }
}

pub fn walk(b: &mut Vec<Stmt>, reach: &mut Defs, cx: &Ctx) -> usize {
    let mut n = 0;
    // object store groups of this list: values known at the list start that the list doesn't
    // redefine
    {
        let mut asg = vec![];
        assigned_deep(b, &mut asg);
        let mut r2 = reach.clone();
        kill(&mut r2, &asg);
        let defs = merged(cx.global, &r2);
        let env = Env { db: cx.db, vars: cx.vars, defs: &defs, lib: cx.lib, objects: &cx.idx.objects };
        n += crate::util::prof::time(2, || crate::groups::rewrite_groups(b, &env, cx.idx));
    }
    let mut i = 0;
    while i < b.len() {
        {
            // a loop's condition also runs after its body: values the body reassigns are unknown
            let mut in_loop = vec![];
            match &b[i] {
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => assigned_deep(body, &mut in_loop),
                Stmt::For { init, step, body, .. } => {
                    assigned_deep(init, &mut in_loop);
                    assigned_deep(step, &mut in_loop);
                    assigned_deep(body, &mut in_loop);
                }
                _ => {}
            }
            let mut here = reach.clone();
            kill(&mut here, &in_loop);
            let defs = merged(cx.global, &here);
            let env = Env { db: cx.db, vars: cx.vars, defs: &defs, lib: cx.lib, objects: &cx.idx.objects };
            if crate::util::prof::time(3, || crate::stmts::try_stmts_at(b, i, cx.whole, &env, cx.idx)) {
                n += 1;
            }
            if i < b.len() && crate::util::prof::time(4, || crate::cflow::try_region_at(b, i, cx.whole, &env, cx.idx)) {
                n += 1;
            }
            if i >= b.len() {
                break;
            }
            n += crate::util::prof::time(5, || scalar_shallow(&mut b[i], &env, cx.idx));
        }
        match &mut b[i] {
            Stmt::If { then, els, .. } => {
                n += walk(then, &mut reach.clone(), cx);
                n += walk(els, &mut reach.clone(), cx);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                let mut asg = vec![];
                assigned_deep(body, &mut asg);
                let mut r = reach.clone();
                kill(&mut r, &asg);
                n += walk(body, &mut r, cx);
            }
            Stmt::For { init, step, body, .. } => {
                let mut asg = vec![];
                assigned_deep(init, &mut asg);
                assigned_deep(step, &mut asg);
                assigned_deep(body, &mut asg);
                let mut r = reach.clone();
                kill(&mut r, &asg);
                n += walk(init, &mut r.clone(), cx);
                n += walk(body, &mut r.clone(), cx);
                n += walk(step, &mut r, cx);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    n += walk(&mut c.body, &mut reach.clone(), cx);
                }
            }
            _ => {}
        }
        let s = b[i].clone();
        crate::util::prof::time(6, || update(reach, &s, cx));
        i += 1;
    }
    n
}
