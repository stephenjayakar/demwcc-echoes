//! Statement-level inlines: bodies with stores, calls and control flow (`CInputStream::ReadInt32`
//! = `p = in.ptr; in.ptr = p + 4; return *p;`, auto_ptr resets, token caching...). The probe's
//! statements are matched against consecutive target statements; the probe's own locals are
//! pattern variables (bound to the target's locals, or folded when the target computed the value
//! in place). A returned value is looked for in the statement that follows.

use crate::matcher::{make_call, Bind, Env, Index, M};
use crate::probe::Probe;
use crate::template::{hole_kind, HoleKind, Shape, Template};
use mwdec_core::TypeDb;
use mwdec_lift::{Expr, IrFunction, Stmt, VarId, VarKind};
use std::collections::HashMap;

fn rename(e: &Expr, map: &HashMap<VarId, usize>) -> Option<Expr> {
    let mut ok = true;
    let mut e = e.clone();
    e.rewrite(&mut |x| {
        if let Expr::Var(v) = x {
            match map.get(v) {
                Some(h) => *x = Expr::Var(*h),
                None => ok = false,
            }
        }
    });
    ok.then_some(e)
}

fn rename_stmts(b: &[Stmt], map: &HashMap<VarId, usize>) -> Option<Vec<Stmt>> {
    let mut out = vec![];
    for s in b {
        out.push(match s {
            Stmt::Assign { dst, src } => Stmt::Assign { dst: rename(dst, map)?, src: rename(src, map)? },
            Stmt::Expr(e) => Stmt::Expr(rename(e, map)?),
            Stmt::If { cond, then, els } => Stmt::If { cond: rename(cond, map)?, then: rename_stmts(then, map)?, els: rename_stmts(els, map)? },
            Stmt::Comment(_) => continue,
            _ => return None,
        });
    }
    Some(out)
}

fn has_effect(b: &[Stmt]) -> bool {
    b.iter().any(|s| match s {
        Stmt::Assign { dst, src } => !matches!(dst, Expr::Var(_)) || src.has_call(),
        Stmt::Expr(_) => true,
        Stmt::If { then, els, .. } => has_effect(then) || has_effect(els),
        _ => false,
    })
}

fn count_nodes(b: &[Stmt]) -> usize {
    let mut n = 0;
    Stmt::walk_exprs(b, &mut |e| {
        if matches!(e, Expr::Binary { .. } | Expr::Unary { .. } | Expr::Call { .. } | Expr::Load { .. } | Expr::Member { .. }) {
            n += 1;
        }
    });
    n + b.len()
}

/// Template from a probe body with statements.
pub fn from_probe(p: &Probe, ir: &IrFunction, db: &TypeDb) -> Result<Template, String> {
    let mut body: Vec<Stmt> = ir.body.clone();
    while matches!(body.last(), Some(Stmt::Return(None))) {
        body.pop();
    }
    let mut result_raw = None;
    if contains_return(&body[..body.len().saturating_sub(1)]) || matches!(body.last(), Some(Stmt::If { .. })) && contains_return(&body) {
        // returns inside control flow: every path assigns a result variable instead
        body = returns_to_assign(&body, RESULT_VAR).ok_or("returns")?;
        result_raw = Some(Expr::Var(RESULT_VAR));
    } else if let Some(Stmt::Return(Some(e))) = body.last() {
        result_raw = Some(e.clone());
        body.pop();
    }
    canon_blocks(&mut body);
    if body.is_empty() || !has_effect(&body) {
        return Err("no statements".into());
    }
    if p.fn_template {
        return Err("statement template of a guessed instantiation".into());
    }
    // a plain forwarder (`void f() { g(); }`) is too ambiguous to recognise
    if body.len() == 1 && result_raw.is_none() && matches!(body[0], Stmt::Expr(_)) {
        return Err("forwarder".into());
    }
    let mut map: HashMap<VarId, usize> = HashMap::new();
    for (i, v) in ir.params.iter().enumerate() {
        map.insert(*v, i);
    }
    if ir.params.len() != p.params.len() {
        return Err("params".into());
    }
    let mut holes: Vec<HoleKind> = p.params.iter().map(|t| hole_kind(t, db)).collect();
    // the probe's locals become pattern variables
    let mut locals = vec![];
    Stmt::walk_exprs(&body, &mut |e| {
        if let Expr::Var(v) = e {
            if !map.contains_key(v) && !locals.contains(v) {
                locals.push(*v);
            }
        }
    });
    if result_raw.as_ref() == Some(&Expr::Var(RESULT_VAR)) && !locals.contains(&RESULT_VAR) {
        locals.push(RESULT_VAR);
    }
    for v in &locals {
        if *v != RESULT_VAR && !matches!(ir.vars[*v].kind, VarKind::Local) {
            return Err("non-local var".into());
        }
        map.insert(*v, holes.len());
        holes.push(HoleKind::Local);
    }
    let stmts = rename_stmts(&body, &map).ok_or("statement kind")?;
    let mut result = match &result_raw {
        Some(e) => Some(rename(e, &map).ok_or("result vars")?),
        None => None,
    };
    // `return *this` of a mutator: no value
    if p.ret_ref {
        if let Some(Expr::Var(h)) = &result {
            if *h < p.params.len() {
                result = None;
            }
        }
    }
    let ops = count_nodes(&stmts) + result.as_ref().map_or(0, crate::template::count_ops);
    Ok(Template {
        name: p.sig.qualified_name.clone(),
        kind: p.kind.clone(),
        sig: p.sig.clone(),
        class: p.class.clone(),
        holes,
        shape: Shape::Stmts { stmts, result },
        ops,
        ret_ref: p.ret_ref,
    })
}

/// Pseudo variable holding the result of a probe with returns inside control flow.
const RESULT_VAR: VarId = usize::MAX - 7;

fn contains_return(b: &[Stmt]) -> bool {
    b.iter().any(|s| match s {
        Stmt::Return(_) => true,
        Stmt::If { then, els, .. } => contains_return(then) || contains_return(els),
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => contains_return(body),
        Stmt::For { init, step, body, .. } => contains_return(init) || contains_return(step) || contains_return(body),
        Stmt::Switch { cases, .. } => cases.iter().any(|c| contains_return(&c.body)),
        _ => false,
    })
}

fn always_returns(b: &[Stmt]) -> bool {
    match b.last() {
        Some(Stmt::Return(Some(_))) => true,
        Some(Stmt::If { then, els, .. }) => always_returns(then) && always_returns(els),
        _ => false,
    }
}

/// `if (c) return a; ...; return b;` -> `if (c) { r = a; } else { ...; r = b; }`.
fn returns_to_assign(b: &[Stmt], r: VarId) -> Option<Vec<Stmt>> {
    let mut out = vec![];
    for (k, s) in b.iter().enumerate() {
        match s {
            Stmt::Return(Some(e)) => {
                out.push(Stmt::Assign { dst: Expr::Var(r), src: e.clone() });
                return Some(out);
            }
            Stmt::Return(None) => return None,
            Stmt::If { cond, then, els } if contains_return(then) || contains_return(els) => {
                let rest = &b[k + 1..];
                let branch = |x: &Vec<Stmt>| -> Option<Vec<Stmt>> {
                    if always_returns(x) {
                        returns_to_assign(x, r)
                    } else {
                        let mut v = x.clone();
                        v.extend_from_slice(rest);
                        returns_to_assign(&v, r)
                    }
                };
                out.push(Stmt::If { cond: cond.clone(), then: branch(then)?, els: branch(els)? });
                return Some(out);
            }
            other => {
                if contains_return(std::slice::from_ref(other)) {
                    return None;
                }
                out.push(other.clone());
            }
        }
    }
    None
}

/// Canonical statement order inside conditional blocks: constant assignments to locals (the
/// result flag of an inline) last, so `r = 1; p->m = x;` and `p->m = x; r = 1;` compare equal.
pub fn canon_blocks(b: &mut Vec<Stmt>) {
    for s in b.iter_mut() {
        if let Stmt::If { then, els, .. } = s {
            for blk in [then, els] {
                canon_blocks(blk);
                let original = blk.clone();
                let n = blk.len();
                let mut consts = vec![];
                let mut others = vec![];
                for (k, st) in blk.drain(..).enumerate() {
                    let movable = matches!(&st, Stmt::Assign { dst: Expr::Var(_), src: Expr::Int { .. } | Expr::Float { .. } }) && k + 1 < n;
                    if movable {
                        consts.push(st);
                    } else {
                        others.push(st);
                    }
                }
                // a moved constant must not be read by the statements it moves past
                let read_later = consts.iter().any(|c| match c {
                    Stmt::Assign { dst: Expr::Var(v), .. } => others.iter().any(|o| {
                        let mut hit = false;
                        Stmt::walk_exprs(std::slice::from_ref(o), &mut |e| {
                            if matches!(e, Expr::Var(x) if x == v) {
                                hit = true;
                            }
                        });
                        hit
                    }),
                    _ => false,
                });
                if read_later {
                    *blk = original;
                    continue;
                }
                others.extend(consts);
                *blk = others;
            }
        }
    }
}

// ---------------------------------------------------------------- matching

fn subst_local(b: &[Stmt], h: usize, with: &Expr) -> Vec<Stmt> {
    let mut out = b.to_vec();
    Stmt::rewrite_exprs(&mut out, &mut |e| {
        if matches!(e, Expr::Var(x) if *x == h) {
            *e = with.clone();
        }
    });
    out
}

fn is_local_hole(t: &Template, h: usize) -> bool {
    matches!(t.holes.get(h), Some(HoleKind::Local))
}

fn match_stmt(m: &mut M, p: &Stmt, t: &Stmt) -> bool {
    match (p, t) {
        (Stmt::Assign { dst: Expr::Var(l), src }, Stmt::Assign { dst: Expr::Var(tv), src: src2 }) if is_local_hole(m.t, *l) => {
            if !matches!(m.env.vars[*tv].kind, VarKind::Local) {
                return false;
            }
            match &m.b[*l] {
                None => m.b[*l] = Some(Bind::Val(Expr::Var(*tv))),
                Some(Bind::Val(Expr::Var(x))) if x == tv => {}
                _ => return false,
            }
            m.m(src, src2)
        }
        (Stmt::Assign { dst, src }, Stmt::Assign { dst: d2, src: s2 }) => !matches!(dst, Expr::Var(_)) && m.m(dst, d2) && m.m(src, s2),
        (Stmt::Expr(e), Stmt::Expr(e2)) => m.m(e, e2),
        (Stmt::If { cond, then, els }, Stmt::If { cond: c2, .. }) => {
            let mut tt = vec![t.clone()];
            canon_blocks(&mut tt);
            let Some(Stmt::If { then: t2, els: e2, .. }) = tt.pop() else { return false };
            let (t2, e2) = (&t2, &e2);
            let snap = m.b.clone();
            if m.m(cond, c2) && match_seq(m, then, t2, 0) == Some(t2.len()) && match_seq(m, els, e2, 0) == Some(e2.len()) {
                return true;
            }
            // other polarity
            m.b = snap;
            let mut nc = Expr::not(cond.clone());
            crate::cflow::canon_cond(&mut nc);
            let mut c2c = c2.clone();
            crate::cflow::canon_cond(&mut c2c);
            m.m(&nc, &c2c) && match_seq(m, els, t2, 0) == Some(t2.len()) && match_seq(m, then, e2, 0) == Some(e2.len())
        }
        _ => false,
    }
}

/// Match pattern statements `p` against target statements `t[ti..]`; returns the end index.
fn match_seq(m: &mut M, p: &[Stmt], t: &[Stmt], ti: usize) -> Option<usize> {
    let Some(first) = p.first() else { return Some(ti) };
    let snap = m.b.clone();
    if ti < t.len() && match_stmt(m, first, &t[ti]) {
        if let Some(e) = match_seq(m, &p[1..], t, ti + 1) {
            return Some(e);
        }
    }
    m.b = snap.clone();
    // a pattern local computed in place by the target (folded into its uses)
    if let Stmt::Assign { dst: Expr::Var(l), src } = first {
        if is_local_hole(m.t, *l) && m.b[*l].is_none() && !src.has_call() {
            let rest = subst_local(&p[1..], *l, src);
            m.folded.push((*l, src.clone()));
            if let Some(e) = match_seq(m, &rest, t, ti) {
                return Some(e);
            }
            m.folded.pop();
            m.b = snap;
        }
    }
    None
}

fn uses_in(b: &[Stmt], v: VarId) -> usize {
    let mut n = 0;
    Stmt::walk_exprs(b, &mut |e| {
        if matches!(e, Expr::Var(x) if *x == v) {
            n += 1;
        }
    });
    n
}

/// Find and replace (pre-order) the first subexpression of `e` matching `r`.
fn replace_result(e: &mut Expr, r: &Expr, m: &mut M, call: &dyn Fn(&M) -> Option<Expr>) -> bool {
    let snap = m.b.clone();
    if m.m(r, e) {
        if let Some(c) = call(m) {
            *e = c;
            return true;
        }
    }
    m.b = snap;
    let mut done = false;
    let mut kids: Vec<&mut Expr> = vec![];
    match e {
        Expr::AddrOf(x) | Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } => kids.push(x),
        Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => kids.push(base),
        Expr::Index { base, index, .. } => {
            kids.push(base);
            kids.push(index);
        }
        Expr::Binary { l, r, .. } => {
            kids.push(l);
            kids.push(r);
        }
        Expr::Ternary { c, t, f, .. } => {
            kids.push(c);
            kids.push(t);
            kids.push(f);
        }
        Expr::Call { callee, args, .. } => {
            match callee {
                mwdec_lift::Callee::Method { this, .. } | mwdec_lift::Callee::Virtual { this, .. } => kids.push(this),
                mwdec_lift::Callee::Indirect(x) => kids.push(x),
                _ => {}
            }
            for a in args.iter_mut() {
                kids.push(a);
            }
        }
        Expr::Construct { args, .. } | Expr::New { args, .. } => {
            for a in args.iter_mut() {
                kids.push(a);
            }
        }
        _ => {}
    }
    for k in kids {
        if !done && replace_result(k, r, m, call) {
            done = true;
        }
    }
    done
}

/// Try statement templates at `b[i]`. True if rewritten.
pub fn try_stmts_at(b: &mut Vec<Stmt>, i: usize, whole: &[Stmt], env: &Env, idx: &Index) -> bool {
    for &ti in &idx.stmts {
        let t = &env.lib.templates[ti];
        let Shape::Stmts { stmts, result } = &t.shape else { continue };
        let mut m = M::new(env, t);
        let Some(end) = match_seq(&mut m, stmts, b, i) else { continue };
        if end == i {
            continue;
        }
        let nparams = t.holes.iter().filter(|h| !matches!(h, HoleKind::Local)).count();
        let _ = nparams;
        // locals of the window must not be needed elsewhere (except the result site)
        let window: Vec<Stmt> = b[i..end].to_vec();
        let mut bound_locals: Vec<VarId> = vec![];
        for (h, k) in t.holes.iter().enumerate() {
            if let (HoleKind::Local, Some(Bind::Val(Expr::Var(v)))) = (k, &m.b[h]) {
                bound_locals.push(*v);
            }
        }
        let mk = |m: &M| -> Option<Expr> {
            let (args, _) = m.finalize(0)?;
            Some(make_call(t, args))
        };
        // result: the next statement uses the value
        let mut replaced_next: Option<Stmt> = None;
        if let Some(r) = result {
            // folded pattern locals are part of the value
            let mut r = r.clone();
            for (h, v) in m.folded.iter().rev() {
                r.rewrite(&mut |x| {
                    if matches!(x, Expr::Var(y) if *y == *h) {
                        *x = v.clone();
                    }
                });
            }
            let r = &r;
            if end < b.len() {
                let mut next = b[end].clone();
                let mut mm = M { env: m.env, t: m.t, b: m.b.clone(), folded: vec![] };
                let hit = match &mut next {
                    Stmt::Assign { src, .. } => replace_result(src, r, &mut mm, &mk),
                    Stmt::Expr(e) | Stmt::Return(Some(e)) => replace_result(e, r, &mut mm, &mk),
                    Stmt::If { cond, .. } => replace_result(cond, r, &mut mm, &mk),
                    _ => false,
                };
                if hit {
                    replaced_next = Some(next);
                }
            }
        }
        let consumed = |v: VarId| match &replaced_next {
            Some(s) => uses_in(std::slice::from_ref(&b[end]), v).saturating_sub(uses_in(std::slice::from_ref(s), v)),
            None => 0,
        };
        let ok = bound_locals.iter().all(|v| uses_in(whole, *v) == uses_in(&window, *v) + consumed(*v));
        if !ok {
            continue;
        }
        match replaced_next {
            Some(next) => {
                b[end] = next;
                b.drain(i..end);
            }
            None => {
                // a value-returning inline whose value is unused: only when its statements are
                // distinctive on their own
                if result.is_some() && stmts.len() < 2 {
                    continue;
                }
                let Some(call) = mk(&m) else { continue };
                b.splice(i..end, [Stmt::Expr(call)]);
            }
        }
        return true;
    }
    false
}
