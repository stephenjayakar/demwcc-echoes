//! Frame objects that only carry a call's by-value result to a single read.
//!
//! `x = obj->VGetTime().GetSeconds();` with a virtual (or otherwise unsized) by-value return:
//! MWCC passes a frame temporary as the struct-return pointer and reads one member back. The
//! lifter sees an untyped frame region written whole by the call and read once; declaring the
//! region as a named object adds a copy (`T t = f();` is a temporary plus a copy), so the read
//! moves to the call: `tmp = *(float*)&obj->VGetTime();`. Moving the read earlier is safe: only
//! the call writes the region and its address goes nowhere else.

use crate::ir::*;
use mwdec_core::Type;

pub fn fold_single_reads(ir: &mut IrFunction) {
    let n = ir.vars.len();
    for v in 0..n {
        let VarKind::Stack { offset, .. } = ir.vars[v].kind else { continue };
        if !matches!(ir.vars[v].ty, Type::Unknown { .. }) {
            continue;
        }
        // every mention: one whole-object call store, one scalar member read, nothing else
        let (mut total, mut reads, mut addressed) = (0usize, 0usize, false);
        let mut read_ty: Option<(i32, Type)> = None;
        Stmt::walk_exprs(&ir.body, &mut |e| match e {
            Expr::Var(x) if *x == v => total += 1,
            Expr::Member { base, offset: o, ty } if matches!(&**base, Expr::Var(x) if *x == v) => {
                reads += 1;
                read_ty = Some((*o, ty.clone()));
            }
            Expr::AddrOf(x) if matches!(&**x, Expr::Var(w) if *w == v) => addressed = true,
            _ => {}
        });
        let Some((roff, rty)) = read_ty else { continue };
        if addressed || reads != 1 || total != 2 || matches!(rty, Type::Named(_) | Type::Array(..) | Type::Unknown { .. }) {
            continue;
        }
        // the store and the read in the same statement list, store first
        let t = ir.vars.len();
        let mut done = false;
        Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
            if done {
                return;
            }
            let Some(si) = b.iter().position(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src: Expr::Call { .. } } if *x == v)) else { return };
            let read_at = b.iter().enumerate().skip(si + 1).find(|(_, s)| {
                let mut hit = false;
                Stmt::walk_exprs(std::slice::from_ref(*s), &mut |e| {
                    if matches!(e, Expr::Member { base, .. } if matches!(&**base, Expr::Var(x) if *x == v)) {
                        hit = true;
                    }
                });
                hit
            });
            let Some((ri, _)) = read_at else { return };
            let Stmt::Assign { src: call, .. } = b[si].clone() else { return };
            b[si] = Stmt::Assign { dst: Expr::Var(t), src: Expr::Member { base: Box::new(call), offset: roff, ty: rty.clone() } };
            Stmt::rewrite_exprs(&mut b[ri..=ri], &mut |e| {
                if matches!(e, Expr::Member { base, .. } if matches!(&**base, Expr::Var(x) if *x == v)) {
                    *e = Expr::Var(t);
                }
            });
            done = true;
        });
        if done {
            ir.vars.push(Var { name: format!("result_{offset:x}"), ty: rty, kind: VarKind::Local });
        }
    }
}
