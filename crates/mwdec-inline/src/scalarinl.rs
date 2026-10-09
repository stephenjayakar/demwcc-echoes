//! Scalar helper inlines in forms the template matcher doesn't see directly.
//!
//! Helpers without an object parameter (`MaskAndShiftLeft(v, m, s)`, `Clamp(lo, v, hi)`,
//! `max_val(a, b)`) are matched by the scalar templates at every expression node, but only in
//! the template's own shape. The compiler folds constants and the structurer picks its own
//! orientation, so the target often has an equivalent form:
//!
//! * `(v << s) & M` for `(v & m) << s` (M = m << s), and `(v & (m << s)) >> s` for
//!   `(v >> s) & m`;
//! * a comparison with its operands swapped (`b > a` for `a < b`) in a ternary;
//! * `if (c1) x = a; else if (c2) x = b;` (or `if (c) x = a; else x = b;`) for `x = c1 ? a : c2 ?
//!   b : x`.
//!
//! Each candidate form is handed to the matcher; a form whose whole value becomes one call is
//! kept. Whether the call compiles like the original is the compiler's decision, so the rewrite
//! is a draft variant (`stmtinl`'s variant switch): otherwise only counted.

use crate::matcher::{scalar_shallow, Env, Index};
use crate::template::{HoleKind, Shape};
use crate::InlineLib;
use mwdec_core::Type;
use mwdec_lift::{BinOp, Callee, Expr, Stmt};

/// Indexes of 3-scalar-hole templates `(v & m) << s` and `(v >> s) & m`.
struct ShiftMask {
    and_shl: Vec<usize>,
    shr_and: Vec<usize>,
}

fn shift_mask_templates(lib: &InlineLib) -> ShiftMask {
    let mut out = ShiftMask { and_shl: vec![], shr_and: vec![] };
    for (i, t) in lib.templates.iter().enumerate() {
        if t.holes.len() != 3 || !t.holes.iter().all(|h| matches!(h, HoleKind::Scalar(_))) {
            continue;
        }
        let Shape::Scalar(p) = &t.shape else { continue };
        let var = |e: &Expr, k: usize| matches!(e, Expr::Var(v) if *v == k);
        match p {
            Expr::Binary { op: BinOp::Shl, l, r, .. } if var(r, 2) => {
                if let Expr::Binary { op: BinOp::And, l: a, r: b, .. } = &**l {
                    if var(a, 0) && var(b, 1) {
                        out.and_shl.push(i);
                    }
                }
            }
            Expr::Binary { op: BinOp::And, l, r, .. } if var(r, 1) => {
                if let Expr::Binary { op: BinOp::Shr, l: a, r: b, .. } = &**l {
                    if var(a, 0) && var(b, 2) {
                        out.shr_and.push(i);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

fn int(e: &Expr) -> Option<i64> {
    match e {
        Expr::Int { value, .. } => Some(*value),
        Expr::Cast { e, .. } => int(e),
        _ => None,
    }
}

fn uint_lit(v: i64) -> Expr {
    Expr::Int { value: v, ty: Type::Int { size: 4, signed: false } }
}

fn call_of(lib: &InlineLib, ti: usize, args: Vec<Expr>) -> Expr {
    crate::matcher::make_call(&lib.templates[ti], args)
}

/// `(v << s) & M` -> `T(v, M >> s, s)`; `(v & M) >> s` -> `T(v, M >> s, s)`.
fn shift_masks(e: &mut Expr, sm: &ShiftMask, lib: &InlineLib) -> usize {
    let mut n = 0;
    e.rewrite(&mut |x| {
        let Expr::Binary { op, l, r, .. } = &*x else { return };
        match op {
            BinOp::And if !sm.and_shl.is_empty() => {
                // (either operand order of `&`)
                let (shl, m) = match (&**l, int(r), int(l)) {
                    (Expr::Binary { op: BinOp::Shl, .. }, Some(m), _) => (&**l, m),
                    (_, _, Some(m)) if matches!(&**r, Expr::Binary { op: BinOp::Shl, .. }) => (&**r, m),
                    _ => return,
                };
                let Expr::Binary { l: v, r: s, .. } = shl else { return };
                let Some(s) = int(s) else { return };
                if !(1..32).contains(&s) || m & ((1 << s) - 1) != 0 || (m as u32 >> s) == 0 {
                    return;
                }
                let v = (**v).clone();
                *x = call_of(lib, sm.and_shl[0], vec![v, uint_lit((m as u32 >> s) as i64), uint_lit(s)]);
                n += 1;
            }
            BinOp::Shr if !sm.shr_and.is_empty() => {
                let Some(s) = int(r) else { return };
                let Expr::Binary { op: BinOp::And, l: v, r: m, .. } = &**l else { return };
                let Some(m) = int(m) else { return };
                if !(1..32).contains(&s) || m & ((1 << s) - 1) != 0 || (m as u32 >> s) == 0 {
                    return;
                }
                let v = (**v).clone();
                *x = call_of(lib, sm.shr_and[0], vec![v, uint_lit((m as u32 >> s) as i64), uint_lit(s)]);
                n += 1;
            }
            _ => {}
        }
    });
    n
}

/// The comparisons of a nested ternary (condition, then the arms' own), up to `max`.
fn cmp_sites(e: &Expr, out: &mut Vec<Vec<u8>>, path: Vec<u8>, max: usize) {
    if out.len() >= max {
        return;
    }
    if let Expr::Ternary { c, t, f, .. } = e {
        if matches!(&**c, Expr::Binary { op, .. } if op.is_cmp()) {
            out.push(path.clone());
        }
        let mut p = path.clone();
        p.push(1);
        cmp_sites(t, out, p, max);
        let mut p = path;
        p.push(2);
        cmp_sites(f, out, p, max);
    }
}

fn flip_at(e: &mut Expr, path: &[u8]) {
    let Expr::Ternary { c, t, f, .. } = e else { return };
    match path.first() {
        None => {
            if let Expr::Binary { op, l, r, .. } = &mut **c {
                *op = op.swap_cmp();
                std::mem::swap(l, r);
            }
        }
        Some(1) => flip_at(t, &path[1..]),
        Some(_) => flip_at(f, &path[1..]),
    }
}

/// Arguments converted to the helper's parameter types where they differ (a function template
/// instance deduces its parameter type from every argument).
fn typed_args(e: &mut Expr, env: &Env) {
    let mut x = &mut *e;
    while let Expr::Cast { e: inner, .. } = x {
        x = inner;
    }
    let Expr::Call { callee: Callee::Direct { sig, .. } | Callee::Method { sig, .. }, args, .. } = x else { return };
    for (a, p) in args.iter_mut().zip(&sig.params) {
        let pt = crate::util::strip(&p.ty).clone();
        let at = mwdec_lift::types::ty_of(a, env.vars);
        let rp = mwdec_lift::types::resolve(Some(env.db), &pt).into_owned();
        let differs = match (crate::util::strip(&rp).int_info(), crate::util::strip(&at).int_info()) {
            (Some(x), Some(y)) => x != y,
            (Some(_), None) => matches!(at, Type::Float { .. }),
            (None, Some(_)) => matches!(rp, Type::Float { .. }),
            _ => matches!((&rp, &at), (Type::Float { size: a }, Type::Float { size: b }) if a != b),
        };
        if differs {
            let inner = std::mem::replace(a, Expr::Int { value: 0, ty: Type::Int { size: 4, signed: true } });
            // (a literal is spelled without its type: cast it too)
            *a = Expr::Cast { ty: pt, e: Box::new(inner) };
        }
    }
}

fn is_helper_call(e: &Expr) -> bool {
    let mut e = e;
    while let Expr::Cast { e: x, .. } = e {
        e = x;
    }
    matches!(e, Expr::Call { callee: Callee::Direct { .. } | Callee::Method { .. }, .. })
}

/// A ternary matched as one call in some orientation of its comparisons.
/// The operand type the comparisons of `e` are made in: float, unsigned or signed int.
fn compare_kind(e: &Expr, env: &Env) -> Option<Type> {
    let mut kind: Option<Type> = None;
    e.walk(&mut |x| {
        let Expr::Binary { op, l, r, .. } = x else { return };
        if !op.is_cmp() {
            return;
        }
        for o in [l, r] {
            let t = mwdec_lift::types::resolve(Some(env.db), crate::util::strip(&mwdec_lift::types::ty_of(o, env.vars))).into_owned();
            match crate::util::strip(&t) {
                Type::Float { size } => kind = Some(Type::Float { size: *size }),
                t => {
                    if let Some((4, false)) = t.int_info() {
                        if !matches!(kind, Some(Type::Float { .. })) && !matches!(o.as_ref(), Expr::Int { .. }) {
                            kind = Some(Type::Int { size: 4, signed: false });
                        }
                    } else if t.int_info().is_some() && kind.is_none() {
                        kind = Some(Type::Int { size: 4, signed: true });
                    }
                }
            }
        }
    });
    kind
}

/// Of a function template's instances (same name and pattern, other parameter types), the one
/// whose parameters are the type the target compares in (`Clamp<unsigned int>` for an unsigned
/// compare).
fn instance_for(call: Expr, want: &Option<Type>, env: &Env) -> Expr {
    let Some(want) = want else { return call };
    let Expr::Call { callee: Callee::Direct { sig, .. } | Callee::Method { sig, .. }, args, .. } = &call else { return call };
    let hole_ty = |t: &crate::template::Template| match t.holes.first() {
        Some(HoleKind::Scalar(h)) => Some(mwdec_lift::types::resolve(Some(env.db), crate::util::strip(h)).into_owned()),
        _ => None,
    };
    let same = |a: &Type, b: &Type| match (crate::util::strip(a), crate::util::strip(b)) {
        (Type::Float { size: x }, Type::Float { size: y }) => x == y,
        (x, y) => x.int_info().is_some() && x.int_info() == y.int_info(),
    };
    if sig.params.first().is_some_and(|p| same(&mwdec_lift::types::resolve(Some(env.db), crate::util::strip(&p.ty)), want)) {
        return call;
    }
    let shape = env.lib.templates.iter().find(|t| t.name == sig.qualified_name && t.sig.params.len() == sig.params.len()).map(|t| format!("{:?}", t.shape));
    for t in env.lib.templates.iter().filter(|t| t.name == sig.qualified_name && t.holes.len() == args.len()) {
        if hole_ty(t).is_some_and(|h| same(&h, want)) && shape.as_deref() == Some(format!("{:?}", t.shape).as_str()) {
            return crate::matcher::make_call(t, args.clone());
        }
    }
    call
}

fn match_ternary(e: &Expr, env: &Env, idx: &Index) -> Option<Expr> {
    let want = compare_kind(e, env);
    // (signedness casts between same-size integers: the helper's parameters convert anyway, and
    // the same operand must read the same in every position)
    let mut e = e.clone();
    e.rewrite(&mut |x| {
        // (one literal spelled with two signednesses: `index < 0u ? 0 : ...`)
        if let Expr::Int { ty: t @ Type::Int { size: 4, .. }, .. } = x {
            *t = Type::Int { size: 4, signed: true };
            return;
        }
        if let Expr::Cast { ty, e: inner } = x {
            let it = mwdec_lift::types::ty_of(inner, env.vars);
            if let (Some((a, _)), Some((b, _))) = (crate::util::strip(ty).int_info(), crate::util::strip(&it).int_info()) {
                if a == b && a == 4 {
                    *x = (**inner).clone();
                }
            }
        }
    });
    let e = &e;
    let mut sites = vec![];
    cmp_sites(e, &mut sites, vec![], 3);
    for mask in 0u32..(1 << sites.len()) {
        let mut cand = e.clone();
        for (k, p) in sites.iter().enumerate() {
            if mask & (1 << k) != 0 {
                flip_at(&mut cand, p);
            }
        }
        if std::env::var_os("MWDI_TRACE_SCALARINL").is_some() {
            eprintln!("scalarinl: try {cand:?}");
        }
        let mut s = Stmt::Expr(cand);
        if scalar_shallow(&mut s, env, idx) > 0 {
            if let Stmt::Expr(mut x) = s {
                if is_helper_call(&x) {
                    let mut x = instance_for(x, &want, env);
                    typed_args(&mut x, env);
                    return Some(x);
                }
            }
        }
    }
    None
}

/// Outermost ternaries first (a clamp contains a min/max).
fn ternaries(e: &mut Expr, env: &Env, idx: &Index) -> usize {
    if matches!(e, Expr::Ternary { .. }) {
        if let Some(c) = match_ternary(e, env, idx) {
            *e = c;
            return 1;
        }
    }
    let mut n = 0;
    match e {
        Expr::Ternary { c, t, f, .. } => {
            n += ternaries(c, env, idx) + ternaries(t, env, idx) + ternaries(f, env, idx);
        }
        Expr::Binary { l, r, .. } => n += ternaries(l, env, idx) + ternaries(r, env, idx),
        Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } | Expr::AddrOf(x) => n += ternaries(x, env, idx),
        Expr::Load { base, .. } | Expr::Member { base, .. } => n += ternaries(base, env, idx),
        Expr::Index { base, index, .. } => n += ternaries(base, env, idx) + ternaries(index, env, idx),
        Expr::Call { args, .. } | Expr::Construct { args, .. } => {
            for a in args {
                n += ternaries(a, env, idx);
            }
        }
        _ => {}
    }
    n
}

/// `if (c1) x = a; else if (c2) x = b;` / `if (c) x = a; else x = b;` as a ternary assignment
/// matched by a helper.
fn if_chains(b: &mut Vec<Stmt>, env: &Env, idx: &Index) -> usize {
    let mut n = 0;
    for s in b.iter_mut() {
        let Stmt::If { cond, then, els } = s else { continue };
        let ([Stmt::Assign { dst: x1, src: a }], rest) = (then.as_slice(), els.as_slice()) else { continue };
        let tern = match rest {
            [Stmt::Assign { dst: x2, src: b2 }] if x2 == x1 => Expr::Ternary { c: Box::new(cond.clone()), t: Box::new(a.clone()), f: Box::new(b2.clone()), ty: Type::Unknown { size: 4 } },
            [Stmt::If { cond: c2, then: t2, els: e2 }] if e2.is_empty() => {
                let [Stmt::Assign { dst: x2, src: b2 }] = t2.as_slice() else { continue };
                if x2 != x1 {
                    continue;
                }
                let inner = Expr::Ternary { c: Box::new(c2.clone()), t: Box::new(b2.clone()), f: Box::new(x1.clone()), ty: Type::Unknown { size: 4 } };
                Expr::Ternary { c: Box::new(cond.clone()), t: Box::new(a.clone()), f: Box::new(inner), ty: Type::Unknown { size: 4 } }
            }
            _ => continue,
        };
        if let Some(c) = match_ternary(&tern, env, idx) {
            *s = Stmt::Assign { dst: x1.clone(), src: c };
            n += 1;
        }
    }
    n
}

/// A float converted to a narrow integer: the lifter reads a quantized paired-single store
/// (`psq_st` + reload, the fast-cast helpers' expansion) as a plain cast, so a cast of a float
/// to the result type of a one-float-parameter helper that is exactly that cast is the helper
/// (`CCast::ToUint8(255.f * r)`). Only in the variant: a source cast reads the same.
fn quantized_casts(e: &mut Expr, lib: &InlineLib, env: &Env) -> usize {
    let mut by_type: Vec<((u8, bool), usize)> = vec![];
    for (i, t) in lib.templates.iter().enumerate() {
        let ([HoleKind::Scalar(h)], Shape::Scalar(Expr::Cast { ty, e: x })) = (t.holes.as_slice(), &t.shape) else { continue };
        if !matches!(crate::util::strip(h), Type::Float { size: 4 }) || !matches!(**x, Expr::Var(0)) {
            continue;
        }
        let (Some(ci), Some(ri)) = (crate::util::strip(ty).int_info(), mwdec_lift::types::resolve(Some(env.db), crate::util::strip(&t.sig.ret)).int_info()) else { continue };
        if ci == ri && ci.0 < 4 && !by_type.iter().any(|(k, _)| *k == ci) {
            by_type.push((ci, i));
        }
    }
    if by_type.is_empty() {
        return 0;
    }
    let mut n = 0;
    e.rewrite(&mut |x| {
        let Expr::Cast { ty, e: inner } = &*x else { return };
        let Some(ci) = crate::util::strip(ty).int_info() else { return };
        if !matches!(mwdec_lift::types::ty_of(inner, env.vars), Type::Float { .. }) {
            return;
        }
        let Some((_, ti)) = by_type.iter().find(|(k, _)| *k == ci) else { return };
        let inner = (**inner).clone();
        *x = call_of(lib, *ti, vec![inner]);
        n += 1;
    });
    n
}

/// Helpers whose value is a conversion of a call (`float cosf(float x) { return (float)cos(x); }`):
/// the matcher's scalar index leaves cast-rooted patterns out (a bare conversion is too common),
/// but one around a call is specific.
fn cast_calls(e: &mut Expr, lib: &InlineLib, env: &Env) -> usize {
    let cands: Vec<usize> = lib
        .templates
        .iter()
        .enumerate()
        .filter(|(_, t)| {
            t.holes.iter().all(|h| matches!(h, HoleKind::Scalar(_)))
                && matches!(&t.shape, Shape::Scalar(Expr::Cast { e, .. }) if e.has_call())
        })
        .map(|(i, _)| i)
        .collect();
    if cands.is_empty() {
        return 0;
    }
    let mut n = 0;
    e.rewrite(&mut |x| {
        if !matches!(x, Expr::Cast { .. }) || !x.has_call() {
            return;
        }
        for &ti in &cands {
            let t = &lib.templates[ti];
            let Shape::Scalar(p) = &t.shape else { continue };
            let mut m = crate::matcher::M::new(env, t);
            if !m.m(p, x) {
                continue;
            }
            let Some((args, _)) = m.finalize(0) else { continue };
            *x = call_of(lib, ti, args);
            n += 1;
            return;
        }
    });
    n
}

/// The conversion-of-a-call helpers over `body` (same code either way, so not a variant).
pub fn conversions(body: &mut [Stmt], env: &Env) -> usize {
    let mut n = 0;
    Stmt::rewrite_exprs(body, &mut |e| n += cast_calls(e, env.lib, env));
    n
}

/// All the rewrites over `body`; returns how many applied.
pub fn rewrite(body: &mut Vec<Stmt>, env: &Env, idx: &Index) -> usize {
    let sm = shift_mask_templates(env.lib);
    let mut n = 0;
    Stmt::rewrite_exprs(body, &mut |e| n += shift_masks(e, &sm, env.lib));
    Stmt::rewrite_exprs(body, &mut |e| n += quantized_casts(e, env.lib, env));
    Stmt::rewrite_exprs(body, &mut |e| n += cast_calls(e, env.lib, env));
    Stmt::for_each_block_mut(body, &mut |b| n += if_chains(b, env, idx));
    fn top(b: &mut Vec<Stmt>, env: &Env, idx: &Index, n: &mut usize) {
        for s in b.iter_mut() {
            match s {
                Stmt::Expr(e) | Stmt::Return(Some(e)) => *n += ternaries(e, env, idx),
                Stmt::Assign { src, .. } => *n += ternaries(src, env, idx),
                Stmt::If { cond, then, els } => {
                    *n += ternaries(cond, env, idx);
                    top(then, env, idx, n);
                    top(els, env, idx, n);
                }
                Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                    *n += ternaries(cond, env, idx);
                    top(body, env, idx, n);
                }
                Stmt::For { init, cond, step, body } => {
                    *n += ternaries(cond, env, idx);
                    top(init, env, idx, n);
                    top(step, env, idx, n);
                    top(body, env, idx, n);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        top(&mut c.body, env, idx, n);
                    }
                }
                _ => {}
            }
        }
    }
    top(body, env, idx, &mut n);
    n
}
