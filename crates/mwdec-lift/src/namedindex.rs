//! A scaled array index kept in a named local (`int index = table * 2; p = &a[index];`): the
//! compiler computes the local into a register at its assignment, which fixes when the
//! multiplication happens (draft variant [`crate::variants::INDEX_NAMED_SCALED`]).

use crate::ir::*;
use mwdec_core::Type;

fn scaled(e: &Expr) -> Option<Expr> {
    match e {
        Expr::Binary { op: BinOp::Mul, l, r, .. } if matches!(**l, Expr::Var(_)) && r.as_int().is_some_and(|k| k > 1) => Some(e.clone()),
        Expr::Binary { op: BinOp::Add, l, r, .. } if r.as_int().is_some() => scaled(l),
        _ => None,
    }
}

/// The first array element accessed with an index `x * k` (+ constant) gets the scaled index from
/// a new local assigned right before its statement.
pub fn name_scaled_index(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
    let mut found: Option<Expr> = None;
    Stmt::walk_exprs(body, &mut |e| {
        if found.is_none() {
            if let Expr::Index { index, .. } = e {
                found = scaled(index);
            }
        }
    });
    let Some(sc) = found else { return };
    if !crate::variants::alt(crate::variants::INDEX_NAMED_SCALED) {
        return;
    }
    let v = vars.len();
    vars.push(Var { name: "index".into(), ty: Type::Int { size: 4, signed: true }, kind: VarKind::Local });
    is_temp.push(false);
    let mut done = false;
    Stmt::for_each_block_mut(body, &mut |b| {
        if done {
            return;
        }
        for i in 0..b.len() {
            let mut hit = false;
            if let Stmt::Assign { .. } | Stmt::Expr(_) | Stmt::Return(Some(_)) = &b[i] {
                Stmt::rewrite_exprs(std::slice::from_mut(&mut b[i]), &mut |e| {
                    if !hit {
                        if let Expr::Index { index, .. } = e {
                            if scaled(index).as_ref() == Some(&sc) {
                                index.rewrite(&mut |x| {
                                    if !hit && *x == sc {
                                        *x = Expr::Var(v);
                                        hit = true;
                                    }
                                });
                            }
                        }
                    }
                });
            }
            if hit {
                b.insert(i, Stmt::Assign { dst: Expr::Var(v), src: sc.clone() });
                done = true;
                return;
            }
        }
    });
}
