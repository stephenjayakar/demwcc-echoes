//! Look-through safety: a temp whose definition reads memory may only be looked through (its
//! read moved to the use) when nothing between the definition and the use may change that
//! memory: no call, no store to an overlapping location of the same canonical object.

use crate::addr::access;
use crate::matcher::{expand, teq, Defs, Env};
use mwdec_lift::{Expr, Stmt, VarId};
use std::collections::HashMap;

/// Canonical memory reads of an expression: (pointer, offset, size).
pub fn reads(e: &Expr, env: &Env, out: &mut Vec<(Expr, i32, u32)>) {
    e.walk(&mut |x| {
        if let Expr::Load { ty, .. } | Expr::Member { ty, .. } = x {
            if let Some((p, o)) = access(x, env) {
                let sz = mwdec_lift::types::size_of(Some(env.db), ty).unwrap_or(4).max(1);
                out.push((p, o, sz));
            }
        }
    });
}

fn stmt_uses(s: &Stmt, t: VarId) -> usize {
    let mut n = 0usize;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if matches!(e, Expr::Var(v) if *v == t) {
            n += 1;
        }
    });
    if let Stmt::Assign { dst: Expr::Var(v), .. } = s {
        if *v == t {
            n = n.saturating_sub(1);
        }
    }
    n
}

fn walk_stores(b: &[Stmt], f: &mut dyn FnMut(&Expr), calls: &mut bool) {
    for s in b {
        match s {
            Stmt::Assign { dst, src } => {
                f(dst);
                *calls |= real_call(src) || real_call(dst);
            }
            Stmt::Expr(e) | Stmt::Return(Some(e)) => *calls |= real_call(e),
            Stmt::If { cond, then, els } => {
                *calls |= real_call(cond);
                walk_stores(then, f, calls);
                walk_stores(els, f, calls);
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                *calls |= real_call(cond);
                walk_stores(body, f, calls);
            }
            Stmt::For { init, cond, step, body } => {
                *calls |= real_call(cond);
                walk_stores(init, f, calls);
                walk_stores(step, f, calls);
                walk_stores(body, f, calls);
            }
            Stmt::Switch { e, cases } => {
                *calls |= real_call(e);
                for c in cases {
                    walk_stores(&c.body, f, calls);
                }
            }
            Stmt::Goto(_) | Stmt::Label(_) => *calls = true,
            _ => {}
        }
    }
}

/// Does statement `s` possibly change any of `rd`? (calls, or stores overlapping a read)
pub fn clobbers(s: &Stmt, rd: &[(Expr, i32, u32)], env: &Env) -> bool {
    let mut hit = false;
    let mut visit_store = |dst: &Expr| {
        if let Expr::Var(_) = dst {
            return;
        }
        let dty = mwdec_lift::types::ty_of(dst, env.vars);
        let sz = mwdec_lift::types::size_of(Some(env.db), &dty).unwrap_or(4).max(1);
        match access(dst, env) {
            Some((p, o)) => {
                for (rp, ro, rs) in rd {
                    if o < ro + *rs as i32 && *ro < o + sz as i32 && teq(&p, rp, env.defs) {
                        hit = true;
                    }
                }
            }
            None => hit = true,
        }
    };
    let mut calls = false;
    walk_stores(std::slice::from_ref(s), &mut visit_store, &mut calls);
    hit || calls
}

fn prune(body: &[Stmt], env: &Env, total_uses: &HashMap<VarId, usize>, bad_out: &mut Vec<VarId>) {
    for (i, s) in body.iter().enumerate() {
        match s {
            Stmt::If { then, els, .. } => {
                prune(then, env, total_uses, bad_out);
                prune(els, env, total_uses, bad_out);
            }
            Stmt::While { body: b, .. } | Stmt::DoWhile { body: b, .. } => prune(b, env, total_uses, bad_out),
            Stmt::For { init, step, body: b, .. } => {
                prune(init, env, total_uses, bad_out);
                prune(step, env, total_uses, bad_out);
                prune(b, env, total_uses, bad_out);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    prune(&c.body, env, total_uses, bad_out);
                }
            }
            _ => {}
        }
        let Stmt::Assign { dst: Expr::Var(t), .. } = s else { continue };
        let Some(def) = env.defs.get(t) else { continue };
        let mut rd = vec![];
        reads(&expand(def, env.defs), env, &mut rd);
        let total = total_uses.get(t).copied().unwrap_or(0);
        let mut seen = 0;
        let mut clobbered = false;
        let mut bad = false;
        for s2 in &body[i + 1..] {
            if seen >= total {
                break;
            }
            let u = stmt_uses(s2, *t);
            if u > 0 {
                if clobbered && !rd.is_empty() {
                    bad = true;
                    break;
                }
                seen += u;
                let compound = !matches!(s2, Stmt::Assign { .. } | Stmt::Expr(_) | Stmt::Return(_));
                if compound && !rd.is_empty() && clobbers(s2, &rd, env) {
                    bad = true;
                    break;
                }
            }
            if !rd.is_empty() && !clobbered && clobbers(s2, &rd, env) {
                clobbered = true;
            }
        }
        if bad || seen < total {
            bad_out.push(*t);
        }
    }
}

/// `env.defs` without the temps whose look-through would move a memory read past a clobber.
pub fn safe_defs(body: &[Stmt], env: &Env) -> Defs {
    let mut total: HashMap<VarId, usize> = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            *total.entry(*v).or_default() += 1;
        }
    });
    // the single definition's destination is not a use
    for t in env.defs.keys() {
        if let Some(n) = total.get_mut(t) {
            *n = n.saturating_sub(1);
        }
    }
    let mut bad = vec![];
    prune(body, env, &total, &mut bad);
    let mut d = env.defs.clone();
    for t in bad {
        d.remove(&t);
    }
    d
}

/// Does `e` contain a call that may write memory? Compiler intrinsics (`__fabs`, `__frsqrte`,
/// `__cntlzw`...) don't.
pub fn real_call(e: &Expr) -> bool {
    let mut found = false;
    e.walk(&mut |x| match x {
        Expr::Call { callee: mwdec_lift::Callee::Direct { symbol, .. }, .. } => {
            if !(symbol.starts_with("__") && mwdec_lift::sig::demangle(symbol).is_none()) {
                found = true;
            }
        }
        Expr::Call { .. } | Expr::New { .. } => found = true,
        _ => {}
    });
    found
}
