//! `va_start` in variadic functions. MWCC's prologue spills the argument registers into a save
//! area (`r3..r10`, and `f1..f8` when CR bit 6 says floats were passed) and `va_start(ap, fmt)`
//! (`__builtin_va_info`) fills the `va_list` struct: counts of named GPR/FPR arguments, the
//! caller's overflow area and the save area. Those stores become `va_start(ap, last)`.

use crate::ir::*;
use mwdec_core::Type;

fn is_spill_src(e: &Expr, vars: &[Var]) -> bool {
    match e {
        Expr::Unknown { text, .. } => text.starts_with("uninit r") || text.starts_with("uninit f"),
        Expr::Var(v) => matches!(vars[*v].kind, VarKind::Param { .. } | VarKind::This),
        Expr::AddrOf(inner) => matches!(**inner, Expr::Var(v) if matches!(vars[v].kind, VarKind::Param { .. })),
        _ => false,
    }
}

fn stack_var(e: &Expr, vars: &[Var]) -> Option<(VarId, i32)> {
    match e {
        Expr::Member { base, offset, .. } => match **base {
            Expr::Var(v) if matches!(vars[v].kind, VarKind::Stack { .. }) => Some((v, *offset)),
            _ => None,
        },
        Expr::Var(v) if matches!(vars[*v].kind, VarKind::Stack { .. }) => Some((*v, 0)),
        _ => None,
    }
}

fn addr_of_stack(e: &Expr, vars: &[Var]) -> Option<VarId> {
    let e = match e {
        Expr::Cast { e, .. } => &**e,
        e => e,
    };
    match e {
        Expr::AddrOf(inner) => stack_var(inner, vars).map(|x| x.0),
        Expr::Var(v) if matches!(vars[*v].kind, VarKind::Stack { .. }) => Some(*v),
        _ => None,
    }
}

pub fn recover(body: &mut Vec<Stmt>, vars: &mut [Var], last_param: Option<VarId>) {
    // the va_list: a stack object whose word at +8 is set to the address of another stack object
    // (the save area) and whose word at 0 is the packed register counts
    let mut list: Option<(VarId, VarId)> = None;
    for s in body.iter() {
        if let Stmt::Assign { dst, src } = s {
            if let (Some((v, 8)), Some(area)) = (stack_var(dst, vars), addr_of_stack(src, vars)) {
                if area != v {
                    list = Some((v, area));
                }
            }
        }
    }
    let Some((ap, area)) = list else { return };
    let mut first = None;
    let mut i = 0;
    while i < body.len() {
        let drop = match &body[i] {
            // the float spills
            Stmt::If { cond: Expr::Unknown { text, .. }, then, els } if text == "cr bit 6" && els.is_empty() => {
                then.iter().all(|t| matches!(t, Stmt::Assign { dst, src } if stack_var(dst, vars).map(|x| x.0) == Some(area) && is_spill_src(src, vars)))
            }
            // the register spills
            Stmt::Assign { dst, src } if stack_var(dst, vars).map(|x| x.0) == Some(area) && is_spill_src(src, vars) => true,
            // the va_list fields
            Stmt::Assign { dst, src } => match stack_var(dst, vars) {
                Some((v, 0)) if v == ap && src.as_int().is_some() => true,
                Some((v, 4)) if v == ap && addr_of_stack(src, vars).is_some() => true,
                Some((v, 8)) if v == ap && addr_of_stack(src, vars) == Some(area) => true,
                _ => false,
            },
            _ => false,
        };
        if drop {
            if first.is_none() && matches!(&body[i], Stmt::Assign { dst, .. } if stack_var(dst, vars).map(|x| x.0) == Some(ap)) {
                first = Some(i);
            }
            body.remove(i);
        } else {
            i += 1;
        }
    }
    let at = first.unwrap_or(0).min(body.len());
    let mut sig = crate::translate::builtin_sig("va_start", 2);
    sig.ret = Type::Void;
    sig.params[0].ty = Type::Named("va_list".into());
    let mut args = vec![Expr::Var(ap)];
    match last_param {
        Some(p) => {
            sig.params[1].ty = vars[p].ty.clone();
            args.push(Expr::Var(p));
        }
        None => {
            sig.params.pop();
        }
    }
    body.insert(at, Stmt::Expr(Expr::Call { callee: Callee::Direct { symbol: "va_start".into(), sig }, args, ret: Type::Void }));
    vars[ap].ty = Type::Named("va_list".into());
    // the list is passed by name (an array type)
    Stmt::rewrite_exprs(body, &mut |x| {
        if matches!(x, Expr::AddrOf(inner) if matches!(**inner, Expr::Var(v) if v == ap)) {
            *x = Expr::Var(ap);
        }
    });
}
