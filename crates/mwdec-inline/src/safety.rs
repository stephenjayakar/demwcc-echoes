//! Look-through safety: a temp whose definition reads memory may only be looked through (its
//! read moved to the use) when nothing between the definition and the use may change that
//! memory: no call, no store to an overlapping location of the same canonical object.

use crate::addr::access;
use crate::matcher::{expand, teq, Defs, Env};
use mwdec_lift::{Expr, Stmt, VarId};
use std::collections::HashMap;

/// Canonical memory reads of an expression: (pointer, offset, size).
pub fn reads(e: &Expr, env: &Env, out: &mut Vec<(Expr, i32, u32)>) {
    reads_d(e, env, out, 0);
}

fn reads_d(e: &Expr, env: &Env, out: &mut Vec<(Expr, i32, u32)>, depth: u32) {
    e.walk(&mut |x| {
        if let Expr::Load { ty, .. } | Expr::Member { ty, .. } = x {
            if let Some((p, o)) = access(x, env) {
                let sz = mwdec_lift::types::size_of(Some(env.db), ty).unwrap_or(4).max(1);
                out.push((p, o, sz));
            }
        }
        // a folded inline call reads what its expansion reads (an expansion that can't be
        // recovered reads anything: the whole of its object arguments, conservatively)
        if let Expr::Call { callee: mwdec_lift::Callee::Direct { sig, symbol } | mwdec_lift::Callee::Method { sig, symbol, .. }, .. } = x {
            let inline = sig.mangled.is_none() && (symbol.is_empty() || *symbol == sig.qualified_name);
            if inline && depth < 4 {
                match crate::matcher::unfold(x, env) {
                    Some(u) => reads_d(&u, env, out, depth + 1),
                    None => out.push((Expr::Int { value: 0, ty: mwdec_core::Type::Int { size: 4, signed: true } }, i32::MIN / 2, u32::MAX / 2)),
                }
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

fn walk_stores(b: &[Stmt], f: &mut dyn FnMut(&Expr), calls: &mut bool, lib: Option<&crate::InlineLib>) {
    let real_call = |e: &Expr| match lib {
        Some(l) => effect_call(e, l),
        None => real_call(e),
    };
    for s in b {
        match s {
            Stmt::Assign { dst, src } => {
                f(dst);
                *calls |= real_call(src) || real_call(dst);
            }
            Stmt::Expr(e) | Stmt::Return(Some(e)) => *calls |= real_call(e),
            Stmt::If { cond, then, els } => {
                *calls |= real_call(cond);
                walk_stores(then, f, calls, lib);
                walk_stores(els, f, calls, lib);
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                *calls |= real_call(cond);
                walk_stores(body, f, calls, lib);
            }
            Stmt::For { init, cond, step, body } => {
                *calls |= real_call(cond);
                walk_stores(init, f, calls, lib);
                walk_stores(step, f, calls, lib);
                walk_stores(body, f, calls, lib);
            }
            Stmt::Switch { e, cases } => {
                *calls |= real_call(e);
                for c in cases {
                    walk_stores(&c.body, f, calls, lib);
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
    walk_stores(std::slice::from_ref(s), &mut visit_store, &mut calls, Some(env.lib));
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
    // temps holding folded inline calls: only when every call's expansion is known (their
    // reads are then checked like any other)
    d.retain(|_, e| !e.has_call() || unfoldable(e, env, 0));
    d
}

fn unfoldable(e: &Expr, env: &Env, depth: u32) -> bool {
    if depth > 4 {
        return false;
    }
    let mut ok = true;
    e.walk(&mut |x| {
        if ok && matches!(x, Expr::Call { .. }) {
            match crate::matcher::unfold(x, env) {
                Some(u) => ok &= unfoldable(&u, env, depth + 1),
                None => ok = false,
            }
        }
    });
    ok
}

/// Does `e` contain a call that may write memory? Compiler intrinsics (`__fabs`, `__frsqrte`,
/// `__cntlzw`...) don't.
/// A call that may write memory: real calls, and folded inlines with side effects.
pub fn effect_call(e: &Expr, lib: &crate::InlineLib) -> bool {
    let mut found = false;
    e.walk(&mut |x| match x {
        Expr::Call { callee: mwdec_lift::Callee::Direct { sig, symbol }, .. } => {
            if sig.mangled.is_none() && symbol == &sig.qualified_name {
                found |= lib.effectful.contains(&sig.qualified_name);
            } else if !(symbol.starts_with("__") && mwdec_lift::sig::demangle(symbol).is_none()) {
                found = true;
            }
        }
        Expr::Call { callee: mwdec_lift::Callee::Method { sig, symbol, .. }, .. } if symbol.is_empty() && sig.mangled.is_none() => {
            found |= lib.effectful.contains(&sig.qualified_name);
        }
        Expr::Call { .. } | Expr::New { .. } => found = true,
        _ => {}
    });
    found
}

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
