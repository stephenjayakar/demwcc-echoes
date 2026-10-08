//! Control-flow inlines (`Clamp`, `FastSqrtF`, `min_val`, `IsNonZero`...): a structured region
//! that only assigns locals and returns is folded into one value expression with ternaries
//! (`v = c ? a : b`), in canonical form, so templates and targets compare as expressions.

use mwdec_core::Type;
use mwdec_lift::{BinOp, Expr, Stmt, UnOp, VarId};
use std::collections::HashMap;

pub type Env = HashMap<VarId, Expr>;

pub fn subst(e: &Expr, env: &Env) -> Expr {
    let mut x = e.clone();
    x.rewrite(&mut |n| {
        if let Expr::Var(v) = n {
            if let Some(d) = env.get(v) {
                *n = d.clone();
            }
        }
    });
    x
}

fn tern(c: Expr, a: Expr, b: Expr, ty: Type) -> Expr {
    Expr::Ternary { c: Box::new(c), t: Box::new(a), f: Box::new(b), ty }
}

fn ty_hint(e: &Expr) -> Type {
    match e {
        Expr::Float { double, .. } => Type::Float { size: if *double { 8 } else { 4 } },
        Expr::Binary { ty, .. } | Expr::Unary { ty, .. } | Expr::Cast { ty, .. } | Expr::Ternary { ty, .. } | Expr::Load { ty, .. } | Expr::Member { ty, .. } | Expr::Int { ty, .. } => ty.clone(),
        Expr::Call { ret, .. } => ret.clone(),
        _ => Type::Unknown { size: 0 },
    }
}

/// Fold a statement list. Returns Ok(Some(value)) if every path returns, Ok(None) if control
/// falls through (with `env` updated), Err if the region does something else.
pub fn fold_list(stmts: &[Stmt], env: &mut Env, allow_ret: bool) -> Result<Option<Expr>, ()> {
    for (k, s) in stmts.iter().enumerate() {
        match s {
            Stmt::Assign { dst: Expr::Var(v), src } => {
                let e = subst(src, env);
                env.insert(*v, e);
            }
            Stmt::Return(Some(e)) if allow_ret => return Ok(Some(subst(e, env))),
            Stmt::Comment(_) => {}
            Stmt::If { cond, then, els } => {
                let c = subst(cond, env);
                let mut e1 = env.clone();
                let r1 = fold_list(then, &mut e1, allow_ret)?;
                let mut e2 = env.clone();
                let r2 = fold_list(els, &mut e2, allow_ret)?;
                match (r1, r2) {
                    (Some(a), Some(b)) => {
                        let t = ty_hint(&a);
                        return Ok(Some(tern(c, a, b, t)));
                    }
                    (Some(a), None) => {
                        let rest = fold_list(&stmts[k + 1..], &mut e2, allow_ret)?.ok_or(())?;
                        let t = ty_hint(&a);
                        return Ok(Some(tern(c, a, rest, t)));
                    }
                    (None, Some(b)) => {
                        let rest = fold_list(&stmts[k + 1..], &mut e1, allow_ret)?.ok_or(())?;
                        let t = ty_hint(&b);
                        return Ok(Some(tern(c, rest, b, t)));
                    }
                    (None, None) => {
                        let mut keys: Vec<VarId> = e1.keys().chain(e2.keys()).copied().collect();
                        keys.sort_unstable();
                        keys.dedup();
                        for v in keys {
                            let a = e1.get(&v).cloned().unwrap_or(Expr::Var(v));
                            let b = e2.get(&v).cloned().unwrap_or(Expr::Var(v));
                            if a == b {
                                env.insert(v, a);
                            } else {
                                let t = ty_hint(&a);
                                env.insert(v, tern(c.clone(), a, b, t));
                            }
                        }
                    }
                }
            }
            _ => return Err(()),
        }
    }
    Ok(None)
}

fn is_float_cmp(l: &Expr, r: &Expr) -> bool {
    let f = |e: &Expr| matches!(e, Expr::Float { .. }) || matches!(ty_hint(e), Type::Float { .. });
    f(l) || f(r)
}

fn is_bool_lit(e: &Expr) -> Option<bool> {
    match e {
        Expr::Int { value, ty: Type::Bool } => match value {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        },
        _ => None,
    }
}

/// Canonical form of ternaries/conditions (applied bottom-up).
pub fn canon(e: &mut Expr) {
    e.rewrite(&mut |n| loop {
        let changed = match n {
            Expr::Ternary { c, t, f, .. } => {
                if let Expr::Unary { op: UnOp::Not, e: inner, .. } = &**c {
                    let inner = (**inner).clone();
                    *c = Box::new(inner);
                    std::mem::swap(t, f);
                    true
                } else if let Expr::Binary { op, l, r, .. } = &mut **c {
                    // integer compares: canonical Lt/Gt/Eq (exact negation)
                    if matches!(op, BinOp::Ge | BinOp::Le | BinOp::Ne) && !is_float_cmp(l, r) {
                        *op = op.negate_cmp().unwrap();
                        std::mem::swap(t, f);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            _ => false,
        };
        if changed {
            continue;
        }
        // c ? x : x -> x ; bool arms -> && / ||
        let repl = match n {
            Expr::Ternary { t, f, .. } if t == f => Some((**t).clone()),
            Expr::Ternary { c, t, f, .. } => match (is_bool_lit(t), is_bool_lit(f)) {
                (Some(true), Some(false)) => Some((**c).clone()),
                (Some(true), None) => Some(Expr::Binary { op: BinOp::LogOr, l: c.clone(), r: f.clone(), ty: Type::Bool }),
                (None, Some(false)) => Some(Expr::Binary { op: BinOp::LogAnd, l: c.clone(), r: t.clone(), ty: Type::Bool }),
                _ => None,
            },
            _ => None,
        };
        match repl {
            Some(r) => *n = r,
            None => break,
        }
    });
}

/// Canonical condition: `!!x` -> `x`, `!(a < b)` -> `a >= b` for integer compares.
pub fn canon_cond(e: &mut Expr) {
    loop {
        let r = match e {
            Expr::Unary { op: UnOp::Not, e: inner, .. } => match &**inner {
                Expr::Unary { op: UnOp::Not, e: x, .. } => Some((**x).clone()),
                Expr::Binary { op, l, r, ty } if op.is_cmp() && (!is_float_cmp(l, r) || matches!(op, BinOp::Eq | BinOp::Ne)) => {
                    Some(Expr::Binary { op: op.negate_cmp().unwrap(), l: l.clone(), r: r.clone(), ty: ty.clone() })
                }
                _ => None,
            },
            _ => None,
        };
        match r {
            Some(x) => *e = x,
            None => break,
        }
    }
}

/// A bool inline's value as a condition: ternaries over 0/1 arms (`c ? (a ? 1 : b != 0) : 0`,
/// the form a returned `bool` folds to) as `&&` / `||` chains (`c && (a || b != 0)`), the form
/// the inline takes inside an `if` condition. None if `e` has no such ternary.
pub fn bool_form(e: &Expr) -> Option<Expr> {
    fn lit(e: &Expr) -> Option<bool> {
        match e {
            Expr::Int { value: 0, .. } => Some(false),
            Expr::Int { value: 1, .. } => Some(true),
            _ => None,
        }
    }
    fn b(e: &Expr) -> Option<Expr> {
        match e {
            Expr::Ternary { .. } => conv(e),
            Expr::Int { .. } => None,
            e => Some(e.clone()),
        }
    }
    fn n(e: &Expr) -> Option<Expr> {
        let mut x = Expr::Unary { op: UnOp::Not, e: Box::new(b(e)?), ty: Type::Bool };
        canon_cond(&mut x);
        Some(x)
    }
    fn and(l: Expr, r: Expr) -> Expr {
        Expr::Binary { op: BinOp::LogAnd, l: Box::new(l), r: Box::new(r), ty: Type::Bool }
    }
    fn or(l: Expr, r: Expr) -> Expr {
        Expr::Binary { op: BinOp::LogOr, l: Box::new(l), r: Box::new(r), ty: Type::Bool }
    }
    fn conv(e: &Expr) -> Option<Expr> {
        let Expr::Ternary { c, t, f, .. } = e else { return None };
        Some(match (lit(t), lit(f)) {
            (Some(true), Some(false)) => b(c)?,
            (Some(false), Some(true)) => n(c)?,
            (None, Some(false)) => and(b(c)?, b(t)?),
            (Some(false), None) => and(n(c)?, b(f)?),
            (Some(true), None) => or(b(c)?, b(f)?),
            (None, Some(true)) => or(n(c)?, b(t)?),
            _ => return None,
        })
    }
    let mut has = false;
    e.walk(&mut |x| has |= matches!(x, Expr::Ternary { .. }));
    if !has {
        return None;
    }
    let out = conv(e)?;
    let mut left = false;
    out.walk(&mut |x| left |= matches!(x, Expr::Ternary { .. }));
    (!left).then_some(out)
}

/// The condition form of a `bool` inline with a control-flow value (see [`bool_form`]).
pub fn bool_template(t: &crate::Template) -> Option<crate::Template> {
    let Shape::Scalar(p) = &t.shape else { return None };
    if !matches!(crate::util::strip(&t.sig.ret), Type::Bool) || t.ret_ref {
        return None;
    }
    let q = bool_form(p)?;
    if !matches!(q, Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, .. }) {
        return None;
    }
    let mut d = t.clone();
    d.ops = crate::template::count_ops(&q);
    d.shape = Shape::Scalar(q);
    Some(d)
}

/// Does `e` contain a ternary or logical operator (a control-flow value)?
pub fn has_cflow(e: &Expr) -> bool {
    let mut f = false;
    e.walk(&mut |x| {
        if matches!(x, Expr::Ternary { .. } | Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, .. }) {
            f = true;
        }
    });
    f
}

// ---------------------------------------------------------------- target regions

use crate::matcher::{make_call, use_score, Env as MEnv, Index, M};
use crate::template::Shape;

fn uses_in(b: &[Stmt], v: VarId) -> usize {
    let mut n = 0;
    Stmt::walk_exprs(b, &mut |e| {
        if matches!(e, Expr::Var(x) if *x == v) {
            n += 1;
        }
    });
    n
}

fn assigned(b: &[Stmt], out: &mut Vec<VarId>) {
    for s in b {
        match s {
            Stmt::Assign { dst: Expr::Var(v), .. } => out.push(*v),
            Stmt::If { then, els, .. } => {
                assigned(then, out);
                assigned(els, out);
            }
            _ => {}
        }
    }
}

fn has_if(b: &[Stmt]) -> bool {
    b.iter().any(|s| matches!(s, Stmt::If { .. }))
}

/// `x != 0` operands of `&&`/`||` as plain truth tests `x`.
fn truth_operands(e: &mut Expr) {
    if let Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, l, r, .. } = e {
        for x in [l, r] {
            truth_operands(x);
            if let Expr::Binary { op: BinOp::Ne, l: a, r: z, .. } = &**x {
                if matches!(**z, Expr::Int { value: 0, .. }) && !matches!(**a, Expr::Cast { .. }) {
                    let a = (**a).clone();
                    **x = a;
                }
            }
        }
    }
}

/// Fold if-regions that compute one value into a call of a control-flow inline.
pub fn rewrite_regions(b: &mut Vec<Stmt>, whole: &[Stmt], env: &MEnv, idx: &Index) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < b.len() {
        if try_region_at(b, i, whole, env, idx) {
            n += 1;
        }
        i += 1;
    }
    n
}

/// Try to fold a region starting at `b[i]` (largest window first). True if rewritten.
pub fn try_region_at(b: &mut Vec<Stmt>, i: usize, whole: &[Stmt], env: &MEnv, idx: &Index) -> bool {
    if idx.cflow.is_empty() || !matches!(b.get(i), Some(Stmt::Assign { dst: Expr::Var(_), .. }) | Some(Stmt::If { .. })) {
        return false;
    }
    let jmax = (i + 6).min(b.len());
    for j in (i + 1..=jmax).rev() {
        let w = &b[i..j];
        // a value-context `&&`/`||` chain the lifter already materialised (`v = a && (b || c)`)
        let chain = j == i + 1 && matches!(&w[0], Stmt::Assign { dst: Expr::Var(_), src: Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, .. } });
        if !has_if(w) && !chain {
            continue;
        }
        // values computed inside the window must be single-use and call-free (else folding
        // would duplicate work): such temps stay outside, before the window
        let bad_temp = w.iter().any(|s| match s {
            Stmt::Assign { dst: Expr::Var(v), src } => {
                let calls = {
                    let mut c = false;
                    src.walk(&mut |x| {
                        if let Expr::Call { callee, .. } = x {
                            let intrinsic = matches!(callee, mwdec_lift::Callee::Direct { symbol, .. } if symbol.starts_with("__") && !symbol.contains("__F"));
                            if !intrinsic {
                                c = true;
                            }
                        }
                    });
                    c
                };
                calls || uses_in(w, *v) > 1 && !matches!(src, Expr::Int { .. } | Expr::Float { .. } | Expr::Var(_))
            }
            _ => false,
        });
        if bad_temp {
            continue;
        }
        let mut fenv = Env::new();
        let r = if chain {
            let Stmt::Assign { dst: Expr::Var(v), src } = &w[0] else { continue };
            // operands tested for truth as the condition form spells them (`p` for `p != 0`)
            let mut src = src.clone();
            truth_operands(&mut src);
            fenv.insert(*v, src);
            None
        } else {
            let Ok(r) = fold_list(w, &mut fenv, true) else { continue };
            r
        };
        let mut cands: Vec<(Option<VarId>, Expr)> = vec![];
        match r {
            Some(v) => {
                // a region that returns on every path must end the list
                if j != b.len() {
                    continue;
                }
                cands.push((None, v));
            }
            None => {
                for (v, e) in &fenv {
                    if has_cflow(e) {
                        cands.push((Some(*v), e.clone()));
                    }
                }
            }
        }
        let mut asg = vec![];
        assigned(w, &mut asg);
        asg.sort_unstable();
        asg.dedup();
        let mut best: Option<(i32, Option<VarId>, Expr)> = None;
        for (tv, mut val) in cands {
            // other values computed in the window must not be needed after it
            let ok = asg.iter().all(|v| Some(*v) == tv || uses_in(whole, *v) == uses_in(w, *v));
            if !ok {
                continue;
            }
            canon(&mut val);
            let trace = std::env::var("MWDI_TRACE").is_ok();
            if trace {
                eprintln!("REGION value {val:?}");
            }
            for &ti in &idx.cflow {
                let t = &env.lib.templates[ti];
                let Shape::Scalar(p) = &t.shape else { continue };
                let mut m = M::new(env, t);
                if !m.m(p, &val) {
                    if trace && std::env::var("MWDI_TRACE").map_or(false, |f| t.name.contains(f.as_str())) {
                        eprintln!("  no match {}: {p:?}", t.name);
                    }
                    continue;
                }
                let Some((args, extra)) = m.finalize(0) else { continue };
                let sc = use_score(t, extra, false, &args);
                if sc >= crate::matcher::MIN_SCORE && best.as_ref().map_or(true, |(b, _, _)| sc > *b) {
                    let mut call = make_call(t, args);
                    if t.ret_ref {
                        // the value is the address of the referenced object
                        call = Expr::AddrOf(Box::new(call));
                    }
                    best = Some((sc, tv, call));
                }
            }
        }
        if let Some((_, tv, call)) = best {
            let st = match tv {
                Some(v) => Stmt::Assign { dst: Expr::Var(v), src: call },
                None => Stmt::Return(Some(call)),
            };
            b.splice(i..j, [st]);
            return true;
        }
    }
    false
}

/// Truth tests in one spelling: `x != 0` -> `x`, `x == 0` -> `!x` (through integer casts of
/// the tested value), `!!x` -> `x`, inside `&&` / `||` / `!`.
pub fn truth_canon(e: &mut Expr) {
    fn bare(x: &Expr) -> Expr {
        match x {
            Expr::Cast { ty, e } if matches!(crate::util::strip(ty), Type::Int { .. } | Type::Bool) && !matches!(**e, Expr::Binary { .. }) => (**e).clone(),
            x => x.clone(),
        }
    }
    match e {
        Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, l, r, .. } => {
            truth_canon(l);
            truth_canon(r);
        }
        Expr::Binary { op: BinOp::Ne, l, r, .. } if matches!(**r, Expr::Int { value: 0, .. }) => *e = bare(l),
        Expr::Binary { op: BinOp::Eq, l, r, .. } if matches!(**r, Expr::Int { value: 0, .. }) => *e = Expr::Unary { op: UnOp::Not, e: Box::new(bare(l)), ty: Type::Bool },
        Expr::Unary { op: UnOp::Not, e: inner, .. } => {
            truth_canon(inner);
            if let Expr::Unary { op: UnOp::Not, e: x, .. } = &**inner {
                *e = (**x).clone();
            }
        }
        _ => {}
    }
}
