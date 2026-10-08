//! A constant selected by a condition and passed straight to a call (`v = 40; if (alpha) v = 39;
//! f(.., v, ..)`) was a conditional expression of the parameter's type in the source
//! (`f(.., alpha ? GX_CTF_A8 : GX_CTF_R8, ..)`): the compiler orders the code around a select of
//! typed constants differently from an `int` select converted afterwards (draft variant
//! [`crate::variants::SELECT_TYPED_ARG`]).

use crate::ir::*;
use mwdec_core::Type;

fn count_var(stmts: &[Stmt], v: VarId) -> usize {
    let mut n = 0;
    Stmt::walk_exprs(stmts, &mut |e| n += matches!(e, Expr::Var(x) if *x == v) as usize);
    n
}

/// Candidates: (index of `v = K1`, index of the `if`, v, K1, K2, cond).
fn find(b: &[Stmt]) -> Option<(usize, usize, VarId, i64, i64, Expr)> {
    for i in 0..b.len() {
        let Stmt::Assign { dst: Expr::Var(v), src } = &b[i] else { continue };
        let Some(k1) = src.as_int() else { continue };
        let v = *v;
        for j in i + 1..b.len() {
            match &b[j] {
                Stmt::If { cond, then, els } if els.is_empty() && then.len() == 1 => {
                    if let Stmt::Assign { dst: Expr::Var(w), src: s2 } = &then[0] {
                        if *w == v && !cond.uses_var(v) {
                            if let Some(k2) = s2.as_int() {
                                return Some((i, j, v, k1, k2, cond.clone()));
                            }
                        }
                    }
                    break;
                }
                // independent register work in between
                Stmt::Assign { dst: Expr::Var(w), src: s } if *w != v && !s.uses_var(v) && !s.has_call() => {}
                _ => break,
            }
        }
    }
    None
}

pub fn typed_select_args(body: &mut Vec<Stmt>) {
    let Some((i, j, v, k1, k2, cond)) = find(body) else {
            return;
    };
    // the single later use: an argument of a call with a typed (non-int) parameter
    if count_var(body, v) != 3 {
        return;
    }
    let mut param_ty: Option<Type> = None;
    Stmt::walk_exprs(&body[j + 1..], &mut |e| {
        if let Expr::Call { callee: Callee::Direct { sig, .. } | Callee::Method { sig, .. }, args, .. } = e {
            for (n, a) in args.iter().enumerate() {
                let inner = match a {
                    Expr::Cast { e, .. } => &**e,
                    a => a,
                };
                if matches!(inner, Expr::Var(x) if *x == v) {
                    if let Some(p) = sig.params.get(n) {
                        if matches!(strip_cv(&p.ty), Type::Named(_)) {
                            param_ty = Some(p.ty.clone());
                        }
                    }
                }
            }
        }
    });
    let Some(pt) = param_ty else { return };
    if !crate::variants::alt(crate::variants::SELECT_TYPED_ARG) {
        return;
    }
    let sel = Expr::Ternary {
        c: Box::new(cond),
        t: Box::new(Expr::Cast { ty: pt.clone(), e: Box::new(Expr::int(k2)) }),
        f: Box::new(Expr::Cast { ty: pt.clone(), e: Box::new(Expr::int(k1)) }),
        ty: pt,
    };
    Stmt::rewrite_exprs(&mut body[j + 1..], &mut |e| {
        let hit = match e {
            Expr::Cast { e: inner, .. } => matches!(**inner, Expr::Var(x) if x == v),
            Expr::Var(x) => *x == v,
            _ => false,
        };
        if hit {
            *e = sel.clone();
        }
    });
    body.remove(j);
    body.remove(i);
}
