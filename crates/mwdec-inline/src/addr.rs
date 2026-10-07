//! Canonical addresses: every target access is reduced to (pointer expression, byte offset), so
//! `r.x`, `*(float*)&r`, `temp->x` with `temp = &r`, and `((char*)p + 4)` agree.

use crate::matcher::{expand, Env};
use crate::util::*;
use mwdec_core::Type;
use mwdec_lift::types::ty_of;
use mwdec_lift::{BinOp, Expr, VarKind};

fn strip_ptr_casts(mut e: Expr, env: &Env) -> Expr {
    loop {
        match e {
            Expr::Cast { ty, e: inner } if matches!(strip(&ty), Type::Ptr(_)) && matches!(strip(&ty_of(&inner, env.vars)), Type::Ptr(_) | Type::Ref(_)) => e = *inner,
            Expr::Cast { ty, e: inner } if matches!(strip(&ty), Type::Ptr(_)) && matches!(&*inner, Expr::AddrOf(_)) => e = *inner,
            other => return other,
        }
    }
}

/// Canonical (pointer, offset) for a pointer-valued expression.
pub fn canon_ptr(b: &Expr, env: &Env) -> (Expr, i32) {
    let b = strip_ptr_casts(expand(b, env.defs), env);
    match &b {
        Expr::AddrOf(x) => match lvalue_addr(x, env) {
            Some(r) => r,
            None => (b.clone(), 0),
        },
        Expr::Binary { op: BinOp::Add, l, r, .. } => match (&**r, matches!(strip(&ty_of(l, env.vars)), Type::Ptr(_) | Type::Ref(_))) {
            (Expr::Int { value, .. }, true) => {
                // byte arithmetic only (`(char*)p + k`)
                let elem = mwdec_lift::pointee(&ty_of(l, env.vars)).and_then(mwdec_lift::scalar_size).unwrap_or(0);
                if elem == 1 {
                    let (p, k) = canon_ptr(l, env);
                    (p, k + *value as i32)
                } else {
                    (b.clone(), 0)
                }
            }
            _ => (b.clone(), 0),
        },
        _ => (b.clone(), 0),
    }
}

/// Canonical address of the object an lvalue designates.
pub fn lvalue_addr(x: &Expr, env: &Env) -> Option<(Expr, i32)> {
    match x {
        Expr::Var(v) => {
            let var = &env.vars[*v];
            match strip(&var.ty) {
                Type::Ref(_) => Some((x.clone(), 0)),
                _ if matches!(var.kind, VarKind::Stack { .. } | VarKind::Param { .. } | VarKind::Local) => Some((Expr::AddrOf(Box::new(x.clone())), 0)),
                _ => None,
            }
        }
        Expr::Load { base, offset, .. } => {
            let (p, k) = canon_ptr(base, env);
            Some((p, k + offset))
        }
        Expr::Member { base, offset, .. } => {
            let (p, k) = lvalue_addr(base, env)?;
            Some((p, k + offset))
        }
        Expr::Global { .. } => Some((Expr::AddrOf(Box::new(x.clone())), 0)),
        _ => None,
    }
}

/// Canonical (pointer, offset) of a scalar access expression.
pub fn access(e: &Expr, env: &Env) -> Option<(Expr, i32)> {
    match e {
        Expr::Load { .. } | Expr::Member { .. } => lvalue_addr(e, env),
        _ => None,
    }
}

/// The class the pointer `p` points at (or the object `X` of `&X`).
pub fn outer_class(p: &Expr, env: &Env) -> Option<String> {
    match p {
        Expr::AddrOf(x) => class_name(&ty_of(x, env.vars), env.db),
        _ => {
            let t = ty_of(p, env.vars);
            class_name(mwdec_lift::pointee(&t)?, env.db)
        }
    }
}

/// (address, lvalue) of the object of class `cls` at `p + off`, if the types say one lives there.
pub fn object_at(p: &Expr, off: i32, cls: &str, env: &Env) -> Option<(Expr, Expr)> {
    if p.has_call() {
        return None;
    }
    let outer = outer_class(p, env)?;
    if !crate::matcher::class_at_pub(env.db, &outer, off, cls) {
        return None;
    }
    let same = off == 0 && is_base_or_same(env.db, cls, &outer);
    let ct = Type::Named(cls.to_string());
    Some(match p {
        Expr::AddrOf(x) => {
            if same {
                ((*p).clone(), (**x).clone())
            } else {
                let lv = Expr::Member { base: x.clone(), offset: off, ty: ct };
                (Expr::AddrOf(Box::new(lv.clone())), lv)
            }
        }
        _ => {
            let is_ref = matches!(strip(&ty_of(p, env.vars)), Type::Ref(_));
            if same {
                let lv = if is_ref { p.clone() } else { Expr::Load { base: Box::new(p.clone()), offset: 0, ty: Type::Named(outer) } };
                (p.clone(), lv)
            } else {
                let lv = if is_ref { Expr::Member { base: Box::new(p.clone()), offset: off, ty: ct } } else { Expr::Load { base: Box::new(p.clone()), offset: off, ty: ct } };
                (Expr::AddrOf(Box::new(lv.clone())), lv)
            }
        }
    })
}

/// Aggregates (class-typed members, the outer class itself) of the object at `p` that contain
/// byte `off`: (start offset, class, size), largest first.
pub fn aggregates_containing(p: &Expr, off: i32, env: &Env) -> Vec<(i32, String, u32)> {
    let mut out = vec![];
    let Some(outer) = outer_class(p, env) else { return out };
    fn go(db: &mwdec_core::TypeDb, cls: &str, base: i32, off: i32, out: &mut Vec<(i32, String, u32)>, d: u32) {
        if d > 8 {
            return;
        }
        let Some(c) = mwdec_lift::sig::find_class(db, cls) else { return };
        if off < base || off >= base + c.size as i32 {
            return;
        }
        out.push((base, c.name.clone(), c.size));
        for b in &c.bases {
            go(db, &b.name, base + b.offset as i32, off, out, d + 1);
        }
        for f in &c.fields {
            if let Some(fc) = class_name(&f.ty, db) {
                go(db, &fc, base + f.offset as i32, off, out, d + 1);
            }
        }
    }
    go(env.db, &outer, 0, off, &mut out, 0);
    out.sort_by_key(|(_, _, s)| std::cmp::Reverse(*s));
    out
}
