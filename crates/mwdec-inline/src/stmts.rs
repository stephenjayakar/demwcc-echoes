//! Statement-level inlines: bodies with stores, calls and control flow (`CInputStream::ReadInt32`
//! = `p = in.ptr; in.ptr = p + 4; return *p;`, auto_ptr resets, token caching...). The probe's
//! statements are matched against consecutive target statements; the probe's own locals are
//! pattern variables (bound to the target's locals, or folded when the target computed the value
//! in place). A returned value is looked for in the statement that follows.

use crate::matcher::{make_call, Bind, Env, Index, M};
use crate::probe::Probe;
use crate::template::{hole_kind, HoleKind, Shape, Template};
use mwdec_core::TypeDb;
use mwdec_lift::{Callee, Expr, IrFunction, Stmt, VarId, VarKind};
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
            Stmt::While { cond, body } => Stmt::While { cond: rename(cond, map)?, body: rename_stmts(body, map)? },
            Stmt::DoWhile { body, cond } => Stmt::DoWhile { body: rename_stmts(body, map)?, cond: rename(cond, map)? },
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
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => has_effect(body),
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
    // (a member function template instantiated with a guessed scalar: the call deduces its
    // type from the argument, so only a method's is safe to name)
    if p.fn_template && !matches!(p.kind, crate::probe::CallKind::Method) {
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
        // a constructor's object (the probe's return slot): the destination hole
        if *v != RESULT_VAR && ir.vars[*v].kind == VarKind::StructRet && matches!(p.kind, crate::probe::CallKind::Ctor) {
            let Some(c) = p.class.clone() else { return Err("constructor without class".into()) };
            map.insert(*v, holes.len());
            holes.push(HoleKind::Obj { class: c, ptr: true, temp_ok: false });
            continue;
        }
        // (a stack object of the inline: `const float value = t; Put(&value, 4)`)
        if *v != RESULT_VAR && !matches!(ir.vars[*v].kind, VarKind::Local | VarKind::Stack { .. }) {
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
        dead: vec![],
        guessed: p.fn_template,
        fixed: vec![],
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
            if !matches!(m.env.vars[*tv].kind, VarKind::Local | VarKind::Stack { .. }) {
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
        // a guarded constructor call on a pointer (`if (p) p->T(a)`) is the placement new the
        // lifter made of the same code (`new (p) T(a)`)
        (Stmt::If { cond, then, els }, Stmt::Expr(Expr::New { placement, ctor: Some(c2), args: a2, .. })) if els.is_empty() && placement.len() == 1 => {
            let [Stmt::Expr(Expr::Call { callee: Callee::Method { sig, this, .. }, args, .. })] = then.as_slice() else { return false };
            if **this != *cond || !mwdec_lift::sig::is_ctor(sig) || args.len() != a2.len() || mwdec_lift::sig::norm_name(&sig.qualified_name) != mwdec_lift::sig::norm_name(&c2.qualified_name) {
                return false;
            }
            if !m.m(cond, &placement[0]) {
                return false;
            }
            args.iter().zip(a2).all(|(p, t)| {
                let snap = m.b.clone();
                if m.m(p, t) {
                    return true;
                }
                m.b = snap;
                m.m(p, &Expr::AddrOf(Box::new(t.clone())))
            })
        }
        // a loop of the inline (a constructor counting its input): the same loop
        (Stmt::While { cond, body }, Stmt::While { cond: c2, body: b2 }) | (Stmt::DoWhile { body, cond }, Stmt::DoWhile { body: b2, cond: c2 }) if std::mem::discriminant(p) == std::mem::discriminant(t) => {
            let snap = m.b.clone();
            if match_seq(m, body, b2, 0) == Some(b2.len()) && m.m(cond, c2) {
                return true;
            }
            m.b = snap;
            false
        }
        (Stmt::If { cond, then, els }, Stmt::If { cond: c2, .. }) => {
            let mut tt = vec![t.clone()];
            canon_blocks(&mut tt);
            let Some(Stmt::If { then: t2, els: e2, .. }) = tt.pop() else { return false };
            let (t2, e2) = (&t2, &e2);
            let snap = m.b.clone();
            if m.m(cond, c2) && match_seq(m, then, t2, 0) == Some(t2.len()) && match_seq(m, els, e2, 0) == Some(e2.len()) {
                return true;
            }
            // truth tests spelled differently (`!p` / `(unsigned int)p == 0`)
            m.b = snap.clone();
            let (mut pc, mut tc) = (cond.clone(), c2.clone());
            crate::cflow::truth_canon(&mut pc);
            crate::cflow::truth_canon(&mut tc);
            if (pc != *cond || tc != *c2) && m.m(&pc, &tc) && match_seq(m, then, t2, 0) == Some(t2.len()) && match_seq(m, els, e2, 0) == Some(e2.len()) {
                return true;
            }
            static TR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
            if let Some(f) = TR.get_or_init(|| std::env::var("MWDI_TRACE_IF").ok()) {
                if m.t.name.contains(f.as_str()) {
                    m.b = snap.clone();
                    let cm = m.m(&pc, &tc);
                    eprintln!("IF {}: cond {cm}
  {pc:?}
  {tc:?}
  then {then:?}
  vs {t2:?}", m.t.name);
                }
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
            m.b = snap.clone();
        }
    }
    // an `if` decided by constant arguments: the compiler kept only one arm (constfold.rs)
    if let Stmt::If { cond, then, els } = first {
        if std::env::var("MWDI_NO_CONSTIF").is_err() && crate::constfold::decidable(cond, &m.t.holes) {
            for (arm, want) in [(then, true), (els, false)] {
                let mut rest: Vec<Stmt> = arm.clone();
                rest.extend_from_slice(&p[1..]);
                if !has_effect(&rest) {
                    continue;
                }
                m.b = snap.clone();
                let nf = m.folded.len();
                if let Some(e) = match_seq(m, &rest, t, ti) {
                    if crate::constfold::solve(cond, want, m) {
                        return Some(e);
                    }
                }
                m.folded.truncate(nf);
            }
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
/// Destination hole of a constructor statement template (the object it builds), if any.
pub fn ctor_dest(t: &Template) -> Option<usize> {
    if !matches!(t.kind, crate::probe::CallKind::Ctor) {
        return None;
    }
    t.holes.iter().enumerate().skip(t.sig.params.len()).find(|(_, h)| matches!(h, HoleKind::Obj { .. })).map(|(k, _)| k)
}

/// A whole-object copy of a small class as member-wise copies (how a constructor's probe
/// stores it), or None.
fn split_copy(s: &Stmt, env: &Env) -> Option<Vec<Stmt>> {
    let Stmt::Assign { dst, src } = s else { return None };
    let ty = mwdec_lift::types::ty_of(dst, env.vars);
    let cls = crate::util::class_name(&ty, env.db)?;
    let fields = crate::template::flat_fields(env.db, &cls)?;
    if fields.len() < 2 || fields.len() > 4 {
        return None;
    }
    let at = |e: &Expr, o: i32, t: &mwdec_core::Type| -> Option<Expr> {
        Some(match e {
            Expr::Load { base, offset, .. } => Expr::Load { base: base.clone(), offset: offset + o, ty: t.clone() },
            Expr::Member { base, offset, .. } => Expr::Member { base: base.clone(), offset: offset + o, ty: t.clone() },
            Expr::Var(_) | Expr::Global { .. } => Expr::Member { base: Box::new(e.clone()), offset: o, ty: t.clone() },
            _ => return None,
        })
    };
    fields.iter().map(|(o, t)| Some(Stmt::Assign { dst: at(dst, *o, t)?, src: at(src, *o, t)? })).collect()
}

pub fn try_stmts_at(b: &mut Vec<Stmt>, i: usize, whole: &[Stmt], env: &Env, idx: &Index) -> bool {
    for &ti in &idx.stmts {
        let t = &env.lib.templates[ti];
        let Shape::Stmts { stmts, result } = &t.shape else { continue };
        if let Some(dh) = ctor_dest(t) {
            if try_ctor_at(b, i, whole, env, t, stmts, dh) {
                return true;
            }
            continue;
        }
        let mut m = M::new(env, t);
        static TRS: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        if let Some(f) = TRS.get_or_init(|| std::env::var("MWDI_TRACE_STMT").ok()) {
            if t.name.contains(f.as_str()) && b.len() >= i + stmts.len() {
                let mut m2 = M::new(env, t);
                let mut k = 0;
                while k < stmts.len() && match_stmt(&mut m2, &stmts[k], &b[i + k]) {
                    k += 1;
                }
                if k > 0 {
                    eprintln!("STMT {} at {i}: {k} of {} match; next {:?}
  vs {:?}", t.name, stmts.len(), stmts.get(k), b.get(i + k));
                }
            }
        }
        let Some(end) = match_seq(&mut m, stmts, b, i) else {
            // independent statements the compiler scheduled between the inline's own
            if try_interleaved_at(b, i, whole, env, t, stmts, result) {
                return true;
            }
            continue;
        };
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
        // the value read later (`p = in.ptr; in.ptr = p + 4; ...; x = *p;`: the compiler
        // scheduled the load past other statements): `x = in.ReadInt32();` at the window
        let mut later: Option<(usize, VarId, Expr)> = None;
        if let (Some(r), None) = (result, &replaced_next) {
            let mut r = r.clone();
            for (h, v) in m.folded.iter().rev() {
                r.rewrite(&mut |x| {
                    if matches!(x, Expr::Var(y) if *y == *h) {
                        *x = v.clone();
                    }
                });
            }
            let first_use = (end..(end + 8).min(b.len())).find(|&k| bound_locals.iter().any(|v| uses_in(std::slice::from_ref(&b[k]), *v) > 0));
            if let Some(k) = first_use {
                if let Stmt::Assign { dst: Expr::Var(dv), src } = &b[k] {
                    let mut mm = M { env: m.env, t: m.t, b: m.b.clone(), folded: vec![] };
                    let local = matches!(env.vars[*dv].kind, VarKind::Local);
                    let untouched = uses_in(&b[i..k], *dv) == 0;
                    if local && untouched && mm.m(&r, src) {
                        // nothing in between may write what the value reads
                        let mut rd = vec![];
                        crate::safety::reads(&crate::matcher::expand(src, env.defs), env, &mut rd);
                        let clean = (end..k).all(|x| !crate::safety::clobbers(&b[x], &rd, env));
                        if clean {
                            if let Some(call) = mk(&m) {
                                later = Some((k, *dv, call));
                            }
                        }
                    }
                }
            }
        }
        if let Some((k, dv, call)) = later {
            let ok = bound_locals.iter().all(|v| uses_in(whole, *v) == uses_in(&window, *v) + uses_in(std::slice::from_ref(&b[k]), *v));
            if ok {
                b.remove(k);
                b.splice(i..end, [Stmt::Assign { dst: Expr::Var(dv), src: call }]);
                return true;
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
                // ... nor when a pattern local was computed in place: the target read the value
                // itself (`p = in.ptr; in.ptr = p + 1; ... *p`), the inline didn't return it
                // (unless every store of the window is to a member the function can't name: then
                // the source can't have written it, `in.ReadUint16();` with its value unused)
                if result.is_some() && !m.folded.is_empty() && !(std::env::var("MWDI_NO_HIDDEN_STORES").is_err() && hidden_stores_only(&b[i..end], env)) {
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

/// A pattern statement copying a whole object hole (`*dest = other`) as member-wise copies.
fn split_copy_pattern(s: &Stmt, t: &Template, env: &Env) -> Option<Vec<Stmt>> {
    let Stmt::Assign { dst: Expr::Load { base, offset, ty }, src: Expr::Var(h) } = s else { return None };
    let Some(HoleKind::Obj { class, ptr: false, .. }) = t.holes.get(*h) else { return None };
    let fields = crate::template::flat_fields(env.db, class)?;
    if crate::util::class_name(ty, env.db).as_deref() != Some(class.as_str()) || fields.len() < 2 {
        return None;
    }
    Some(
        fields
            .iter()
            .map(|(o, ft)| Stmt::Assign {
                dst: Expr::Load { base: base.clone(), offset: offset + o, ty: ft.clone() },
                src: Expr::Member { base: Box::new(Expr::Var(*h)), offset: *o, ty: ft.clone() },
            })
            .collect(),
    )
}

/// A constructor statement template at `b[i]`: its statements build the object at the
/// destination hole (a whole-object copy in the target counts as the member-wise copies the
/// probe made); the window becomes `*dest = T(args);`.
fn try_ctor_at(b: &mut Vec<Stmt>, i: usize, whole: &[Stmt], env: &Env, t: &Template, stmts: &[Stmt], dh: usize) -> bool {
    // the target with a leading whole-object copy split (index map back to `b`)
    let mut exp: Vec<Stmt> = b[..i].to_vec();
    let mut back: Vec<usize> = (0..i).collect();
    for (k, s) in b.iter().enumerate().skip(i) {
        match (k == i).then(|| split_copy(s, env)).flatten() {
            Some(parts) => {
                for p in parts {
                    exp.push(p);
                    back.push(k);
                }
            }
            None => {
                exp.push(s.clone());
                back.push(k);
            }
        }
    }
    let mut m = M::new(env, t);
    if std::env::var("MWDI_TRACE_CTOR").is_ok_and(|f| t.name.contains(f.as_str())) && b.len() > i {
        let mut m2 = M::new(env, t);
        let mut k = 0;
        while k < stmts.len() && i + k < b.len() && match_stmt(&mut m2, &stmts[k], &b[i + k]) {
            k += 1;
        }
        eprintln!("CTOR {} at {i}: {k} of {}; next {:?}
  vs {:?}", t.name, stmts.len(), stmts.get(k), b.get(i + k));
    }
    // as is first, then with the copy split
    let (exp, back, end) = match match_seq(&mut m, stmts, b, i) {
        Some(e) => (b.clone(), (0..b.len()).collect::<Vec<_>>(), e),
        None => {
            m = M::new(env, t);
            match match_seq(&mut m, stmts, &exp, i) {
                Some(e) => (exp, back, e),
                None => {
                    // the probe's whole-object copy against member-wise stores in the target
                    let split: Vec<Stmt> = stmts.iter().flat_map(|s| split_copy_pattern(s, t, env).unwrap_or_else(|| vec![s.clone()])).collect();
                    if split.len() == stmts.len() {
                        return false;
                    }
                    m = M::new(env, t);
                    if std::env::var("MWDI_TRACE_CTOR").is_ok_and(|f| t.name.contains(f.as_str())) {
                        eprintln!("CTOR {} split {:?}
  vs {:?}", t.name, split, &b[i..(i + split.len()).min(b.len())]);
                    }
                    match match_seq(&mut m, &split, b, i) {
                        Some(e) => (b.clone(), (0..b.len()).collect::<Vec<_>>(), e),
                        None => return false,
                    }
                }
            }
        }
    };
    if end == i {
        return false;
    }
    // the window must cover whole original statements
    let oend = if end < exp.len() { back[end] } else { b.len() };
    if end < exp.len() && back[end - 1] == back[end] {
        return false;
    }
    // pattern locals used nowhere else
    let window = &b[i..oend];
    for (h, k) in t.holes.iter().enumerate() {
        if let (HoleKind::Local, Some(Bind::Val(Expr::Var(v)))) = (k, &m.b[h]) {
            if uses_in(whole, *v) != uses_in(window, *v) {
                return false;
            }
        }
    }
    let Some((mut args, _)) = m.finalize(0) else {
        if std::env::var("MWDI_TRACE_CTOR").is_ok_and(|f| t.name.contains(f.as_str())) {
            eprintln!("CTOR {} finalize failed {:?}", t.name, m.b);
        }
        return false;
    };
    // the destination is the last non-local hole
    let nonlocal_before = t.holes[..dh].iter().filter(|h| !matches!(h, HoleKind::Local)).count();
    if nonlocal_before >= args.len() {
        return false;
    }
    let mut dest = args.remove(nonlocal_before);
    // the destination is written: a `const T*` cast made for a constructor's pointer argument
    // (finalize) would render `*(const T*)&x = T(..)`, which doesn't compile
    if let Expr::Cast { ty: mwdec_core::Type::Ptr(inner), e } = &dest {
        if let mwdec_core::Type::Const(x) = &**inner {
            dest = Expr::Cast { ty: mwdec_core::Type::Ptr(x.clone()), e: e.clone() };
        }
    }
    let lv = match dest {
        Expr::AddrOf(x) => *x,
        p => Expr::Load { base: Box::new(p), offset: 0, ty: mwdec_core::Type::Named(t.class.clone().unwrap_or_default()) },
    };
    let call = make_call(t, args);
    b.splice(i..oend, [Stmt::Assign { dst: lv, src: call }]);
    true
}

/// A pure definition of a single-assignment local (`t = p->x + 1`): may be moved before the
/// statements of an inline expansion it was scheduled into.
fn movable_def(s: &Stmt, whole: &[Stmt], env: &Env) -> Option<VarId> {
    let Stmt::Assign { dst: Expr::Var(v), src } = s else { return None };
    if !matches!(env.vars[*v].kind, VarKind::Local) || src.has_call() || src.uses_var(*v) {
        return None;
    }
    fn defs_of(b: &[Stmt], v: VarId) -> usize {
        b.iter()
            .map(|s| match s {
                Stmt::Assign { dst: Expr::Var(w), .. } => (*w == v) as usize,
                Stmt::If { then, els, .. } => defs_of(then, v) + defs_of(els, v),
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => defs_of(body, v),
                Stmt::For { init, step, body, .. } => defs_of(init, v) + defs_of(step, v) + defs_of(body, v),
                Stmt::Switch { cases, .. } => cases.iter().map(|c| defs_of(&c.body, v)).sum(),
                _ => 0,
            })
            .sum()
    }
    (defs_of(whole, *v) == 1).then_some(*v)
}

/// Statement template `t` at `b[i]` with independent definitions interleaved: the expansion's
/// statements are matched with those set aside, which then go before the folded call.
fn try_interleaved_at(b: &mut Vec<Stmt>, i: usize, whole: &[Stmt], env: &Env, t: &Template, stmts: &[Stmt], result: &Option<Expr>) -> bool {
    if result.is_some() || stmts.len() < 2 || i >= b.len() {
        return false;
    }
    let wend = (i + stmts.len() + 3).min(b.len());
    let movable: Vec<usize> = (i + 1..wend).filter(|&k| movable_def(&b[k], whole, env).is_some()).take(6).collect();
    if movable.is_empty() {
        return false;
    }
    // set-aside choices: one, then two, then three of the movable definitions
    let mut choices: Vec<Vec<usize>> = movable.iter().map(|&k| vec![k]).collect();
    for x in 0..movable.len() {
        for y in x + 1..movable.len() {
            choices.push(vec![movable[x], movable[y]]);
            for z in y + 1..movable.len() {
                choices.push(vec![movable[x], movable[y], movable[z]]);
            }
        }
    }
    for aside in choices {
        let kept: Vec<usize> = (i..wend).filter(|k| !aside.contains(k)).collect();
        let view: Vec<Stmt> = kept.iter().map(|&w| b[w].clone()).collect();
        let mut m = M::new(env, t);
        static TRI: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        if let Some(f) = TRI.get_or_init(|| std::env::var("MWDI_TRACE_STMT").ok()) {
            if t.name.contains(f.as_str()) {
                let mut m2 = M::new(env, t);
                let mut k = 0;
                while k < stmts.len() && k < view.len() && match_stmt(&mut m2, &stmts[k], &view[k]) {
                    k += 1;
                }
                eprintln!("INTER {} at {i} aside {aside:?}: {k} of {} match; next {:?}
  vs {:?}", t.name, stmts.len(), stmts.get(k), view.get(k));
            }
        }
        let Some(n) = match_seq(&mut m, stmts, &view, 0) else { continue };
        if n < 2 {
            continue;
        }
        let end = kept[n - 1] + 1;
        // every set-aside statement lies inside the expansion
        if aside.iter().any(|&a| a >= end) {
            if TRI.get().cloned().flatten().is_some_and(|f| t.name.contains(f.as_str())) {
                eprintln!("INTER aside past end {n}");
            }
            continue;
        }
        let used: Vec<usize> = kept[..n].to_vec();
        // a set-aside statement moves before the expansion's earlier statements: it must not
        // read what they write, and they must not use what it defines
        let ok = aside.iter().all(|&a| {
            let Stmt::Assign { dst: Expr::Var(v), src } = &b[a] else { return false };
            let mut rd = vec![];
            crate::safety::reads(&crate::matcher::expand(src, env.defs), env, &mut rd);
            used.iter().filter(|&&w| w < a).all(|&w| !crate::safety::clobbers(&b[w], &rd, env) && uses_in(std::slice::from_ref(&b[w]), *v) == 0)
        });
        let tr = TRI.get().cloned().flatten().is_some_and(|f| t.name.contains(f.as_str()));
        if !ok {
            if tr {
                eprintln!("INTER dependence");
            }
            continue;
        }
        let window: Vec<Stmt> = used.iter().map(|&w| b[w].clone()).collect();
        let locals_ok = t.holes.iter().enumerate().all(|(h, kd)| match (kd, &m.b[h]) {
            (HoleKind::Local, Some(Bind::Val(Expr::Var(v)))) => uses_in(whole, *v) == uses_in(&window, *v),
            _ => true,
        });
        if !locals_ok {
            if tr {
                eprintln!("INTER locals");
            }
            continue;
        }
        let Some((args, _)) = m.finalize(0) else {
            if tr {
                eprintln!("INTER finalize {:?}", m.b);
            }
            continue;
        };
        let call = make_call(t, args);
        let mut repl: Vec<Stmt> = aside.iter().map(|&a| b[a].clone()).collect();
        repl.push(Stmt::Expr(call));
        b.splice(i..end, repl);
        return true;
    }
    false
}

/// Does every statement of `w` store to a member the function being rewritten can't name (and
/// is there at least one store)?
fn hidden_stores_only(w: &[Stmt], env: &Env) -> bool {
    let mut n = 0;
    for s in w {
        let Stmt::Assign { dst, .. } = s else { return false };
        if matches!(dst, Expr::Var(_)) {
            continue;
        }
        let ty = match dst {
            Expr::Load { ty, .. } | Expr::Member { ty, .. } => ty.clone(),
            _ => return false,
        };
        let Some((p, off)) = crate::addr::access(dst, env) else { return false };
        let Some(outer) = crate::addr::outer_class(&p, env) else { return false };
        let size = mwdec_lift::scalar_size(crate::util::strip(&ty)).unwrap_or(0);
        let Some((path, _)) = mwdec_lift::types::field_path(env.db, &outer, off, size) else { return false };
        let hidden = path.iter().any(|pe| matches!(pe, mwdec_lift::types::PathElem::Field(n, owner) if !crate::matcher::member_accessible(env.db, owner, n)));
        if !hidden {
            return false;
        }
        n += 1;
    }
    n > 0
}
