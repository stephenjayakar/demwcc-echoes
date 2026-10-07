//! Array indexing recovery: MWCC turns `a[i].f` into byte arithmetic (`mulli`/`slwi` + `lwzx`) and,
//! in loops, into strength-reduced induction variables (`off += sizeof(T)`). Rebuild `a[i].f`.

use crate::ir::*;
use crate::types;
use mwdec_core::{Type, TypeDb};

fn uncast(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => uncast(e),
        e => e,
    }
}

/// `i * k` / `i << s` -> (i, k)
fn scaled(e: &Expr) -> Option<(Expr, i64)> {
    match uncast(e) {
        Expr::Binary { op: BinOp::Mul, l, r, .. } => match (r.as_int(), l.as_int()) {
            (Some(k), _) => Some(((**l).clone(), k)),
            (None, Some(k)) => Some(((**r).clone(), k)),
            _ => None,
        },
        Expr::Binary { op: BinOp::Shl, l, r, .. } => r.as_int().filter(|s| *s < 31).map(|s| ((**l).clone(), 1i64 << s)),
        _ => None,
    }
}

/// byte offset expression -> (index, scale, constant)
fn split_offset(e: &Expr) -> Option<(Expr, i64, i64)> {
    if let Some((i, k)) = scaled(e) {
        return Some((i, k, 0));
    }
    match uncast(e) {
        Expr::Binary { op: BinOp::Add, l, r, .. } => {
            if let Some(c) = r.as_int() {
                let (i, k, c2) = split_offset(l)?;
                return Some((i, k, c2 + c));
            }
            if let Some(c) = l.as_int() {
                let (i, k, c2) = split_offset(r)?;
                return Some((i, k, c2 + c));
            }
            None
        }
        _ => None,
    }
}

/// `(u8*)p + off` with `p: T*` and `off = i*sizeof(T) + c` -> (`p[i]` lvalue, c)
fn element(p: &Expr, off: &Expr, vars: &[Var], db: Option<&TypeDb>) -> Option<(Expr, i64)> {
    let pt = types::ty_of(p, vars);
    let t = pointee(&pt)?.clone();
    if matches!(strip_cv(&t), Type::Void | Type::Unknown { .. }) {
        return None;
    }
    let s = types::size_of(db, &t)? as i64;
    if s <= 1 {
        return None;
    }
    let (i, k, c) = split_offset(off)?;
    if k != s || c < 0 || c >= s {
        return None;
    }
    Some((Expr::Index { base: Box::new(p.clone()), index: Box::new(i), ty: t }, c))
}

/// Byte-pointer arithmetic `(u8*)p + off` in its two IR spellings.
fn byte_add_parts(e: &Expr) -> Option<(&Expr, &Expr)> {
    match e {
        Expr::Binary { op: BinOp::Add, l, r, .. } => match &**l {
            Expr::Cast { ty, e: p } if matches!(pointee(ty), Some(t) if scalar_size(t) == Some(1)) => Some((p, r)),
            _ => None,
        },
        Expr::AddrOf(inner) => match &**inner {
            Expr::Index { base, index, ty } if scalar_size(ty) == Some(1) => match &**base {
                Expr::Cast { ty: ct, e: p } if matches!(pointee(ct), Some(t) if scalar_size(t) == Some(1)) => Some((p, index)),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

fn rewrite(e: &mut Expr, vars: &[Var], db: Option<&TypeDb>) {
    let new = match &*e {
        Expr::Load { base, offset, ty } => byte_add_parts(base).and_then(|(p, off)| {
            let (el, c) = element(p, off, vars, db)?;
            let o = c as i32 + *offset;
            if o == 0 && types::size_of(db, &types::ty_of(&el, vars)) == scalar_size(ty) {
                return Some(el);
            }
            Some(Expr::Member { base: Box::new(el), offset: o, ty: ty.clone() })
        }),
        e2 @ (Expr::Binary { .. } | Expr::AddrOf(_)) => byte_add_parts(e2).and_then(|(p, off)| {
            let (el, c) = element(p, off, vars, db)?;
            if c == 0 {
                Some(Expr::AddrOf(Box::new(el)))
            } else {
                Some(Expr::AddrOf(Box::new(Expr::Member { base: Box::new(el), offset: c as i32, ty: Type::Unknown { size: 0 } })))
            }
        }),
        _ => None,
    };
    if let Some(n) = new {
        *e = n;
    }
}

pub fn recover(body: &mut [Stmt], vars: &[Var], db: Option<&TypeDb>) {
    Stmt::rewrite_exprs(body, &mut |e| rewrite(e, vars, db));
}

/// In `for (i = 0; i < n; i++)`: an offset variable `v = 0; ... v = v + K;` advancing in lockstep
/// with `i` is `i * K` (strength reduction undone).
pub fn undo_strength_reduction(body: &mut Vec<Stmt>, vars: &[Var]) {
    Stmt::for_each_block_mut(body, &mut |b| {
        for k in 0..b.len() {
            let Stmt::For { init, step, body: lb, .. } = &b[k] else { continue };
            let i = match (init.as_slice(), step.as_slice()) {
                ([Stmt::Assign { dst: Expr::Var(i), src }], [Stmt::Expr(Expr::IncDec { e, delta: 1, .. })]) if src.as_int() == Some(0) && matches!(**e, Expr::Var(x) if x == *i) => *i,
                _ => continue,
            };
            // candidate: last statement of the body (or of the trailing arm) `v = v + K`
            let mut found: Option<(VarId, i64)> = None;
            fn last_inc(b: &[Stmt]) -> Option<(VarId, i64)> {
                match b.last()? {
                    Stmt::Assign { dst: Expr::Var(v), src: Expr::Binary { op: BinOp::Add, l, r, .. } } if matches!(**l, Expr::Var(x) if x == *v) => r.as_int().map(|k| (*v, k)),
                    Stmt::If { els, then, .. } => last_inc(els).or_else(|| last_inc(then)),
                    _ => None,
                }
            }
            if let Some((v, kk)) = last_inc(lb) {
                if vars[v].kind == VarKind::Local && !is_ptr(&vars[v].ty) {
                    found = Some((v, kk));
                }
            }
            let Some((v, kk)) = found else { continue };
            // init `v = 0` earlier in this list, v not used after the loop
            let Some(ini) = b[..k].iter().rposition(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src } if *x == v && src.as_int() == Some(0))) else { continue };
            let used_after = b[k + 1..].iter().any(|s| {
                let mut f = false;
                Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| f |= matches!(e, Expr::Var(x) if *x == v));
                f
            });
            if used_after {
                continue;
            }
            let mut assigns = 0;
            fn count(b: &[Stmt], v: VarId, n: &mut usize) {
                for s in b {
                    match s {
                        Stmt::Assign { dst: Expr::Var(x), .. } if *x == v => *n += 1,
                        Stmt::If { then, els, .. } => {
                            count(then, v, n);
                            count(els, v, n);
                        }
                        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => count(body, v, n),
                        _ => {}
                    }
                }
            }
            count(lb, v, &mut assigns);
            if assigns != 1 {
                continue;
            }
            // rewrite: drop the increment, replace v by i * K
            let repl = Expr::bin(BinOp::Mul, Expr::Var(i), Expr::int(kk), t_s32());
            if let Stmt::For { body: lb, .. } = &mut b[k] {
                fn drop_inc(b: &mut Vec<Stmt>, v: VarId) -> bool {
                    match b.last_mut() {
                        Some(Stmt::Assign { dst: Expr::Var(x), .. }) if *x == v => {
                            b.pop();
                            true
                        }
                        Some(Stmt::If { then, els, .. }) => drop_inc(els, v) || drop_inc(then, v),
                        _ => false,
                    }
                }
                drop_inc(lb, v);
                Stmt::rewrite_exprs(lb, &mut |e| {
                    if matches!(e, Expr::Var(x) if *x == v) {
                        *e = repl.clone();
                    }
                });
            }
            b.remove(ini);
            break;
        }
    });
}
