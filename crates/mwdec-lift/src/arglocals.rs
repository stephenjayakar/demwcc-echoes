//! Call arguments the source computed into named locals first, in argument order
//! (`GXColorSrc amb = (GXColorSrc)(f >> 1 & 1); ...; GXSetChanCtrl(chan, enable, amb, ...)`): the
//! locals fix the order the compiler computes them in (draft variant
//! [`crate::variants::ARGS_NAMED_LOCALS`]). A local takes the parameter's type when that needs an
//! explicit conversion (enums, `bool`), else the value's own type.

use crate::ir::*;
use crate::types;
use mwdec_core::{Type, TypeDb};

fn trivial(e: &Expr) -> bool {
    match e {
        Expr::Var(_) | Expr::Int { .. } | Expr::Float { .. } | Expr::Str { .. } | Expr::Global { .. } | Expr::FuncAddr { .. } => true,
        Expr::Cast { e, .. } => trivial(e),
        _ => false,
    }
}

fn strip(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip(e),
        e => e,
    }
}

pub fn args_to_locals(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>, db: Option<&TypeDb>) {
    // the first top-level call statement with at least three computed arguments
    let pos = body.iter().position(|s| {
        let call = match s {
            Stmt::Expr(c) => Some(c),
            Stmt::Assign { src, .. } | Stmt::Return(Some(src)) => Some(src),
            _ => None,
        };
        matches!(call, Some(Expr::Call { callee: Callee::Direct { .. } | Callee::Method { .. }, args, .. }) if args.iter().filter(|a| !trivial(a)).count() >= 3)
    });
    let Some(i) = pos else { return };
    if !crate::variants::alt(crate::variants::ARGS_NAMED_LOCALS) {
        return;
    }
    let mut new: Vec<Stmt> = vec![];
    let call = match &mut body[i] {
        Stmt::Expr(c) => c,
        Stmt::Assign { src, .. } | Stmt::Return(Some(src)) => src,
        _ => return,
    };
    let Expr::Call { callee, args, .. } = call else { return };
    let sig = match callee {
        Callee::Direct { sig, .. } | Callee::Method { sig, .. } => sig.clone(),
        _ => return,
    };
    for (n, a) in args.iter_mut().enumerate() {
        if trivial(a) {
            continue;
        }
        let pt = sig.params.get(n).map(|p| p.ty.clone());
        let needs_conv = pt.as_ref().is_some_and(|t| {
            let r = types::resolve(db, t).into_owned();
            matches!(strip_cv(&r), Type::Named(_) | Type::Bool) && !types::is_aggregate(db, &r) && !matches!(strip_cv(t), Type::Ptr(_) | Type::Ref(_))
        });
        let (ty, val) = if needs_conv {
            let t = pt.clone().unwrap();
            (t.clone(), Expr::Cast { ty: t, e: Box::new(strip(a).clone()) })
        } else {
            let vt = types::ty_of(strip(a), vars);
            if matches!(vt, Type::Unknown { .. } | Type::Void) {
                continue;
            }
            (vt, strip(a).clone())
        };
        let v = vars.len();
        vars.push(Var { name: format!("arg{n}"), ty, kind: VarKind::Local });
        is_temp.push(false);
        new.push(Stmt::Assign { dst: Expr::Var(v), src: val });
        *a = Expr::Var(v);
    }
    let k = new.len();
    body.splice(i..i, new);
    let _ = k;
}
