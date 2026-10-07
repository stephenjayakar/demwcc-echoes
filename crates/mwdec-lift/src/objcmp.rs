//! Comparisons of small objects through their operators.

use crate::ir::*;
use mwdec_core::{Type, TypeDb};

/// Does class `c` declare `operator==` / `operator!=` (member or free, on `const C&`)?
fn has_cmp_operator(db: &TypeDb, c: &str, op: BinOp) -> bool {
    let name = if op == BinOp::Eq { "operator==" } else { "operator!=" };
    if db.decls.contains_key(&format!("{c}::{name}")) {
        return true;
    }
    db.decls.get(name).map_or(false, |ds| {
        ds.iter().any(|d| d.params.first().map_or(false, |p| match strip_cv(&p.ty) {
            Type::Ref(t) => named(strip_cv(t)).map_or(false, |n| crate::sig::norm_name(n) == crate::sig::norm_name(c)),
            _ => false,
        }))
    })
}

fn strip_casts(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip_casts(e),
        e => e,
    }
}

/// The object whose only scalar `e` reads (`obj.value` of a one-member class with a comparison
/// operator), as an expression of the class type.
fn whole_object(e: &Expr, vars: &[Var], db: &TypeDb, op: BinOp) -> Option<Expr> {
    let cmp_class = |t: &Type, size: u32| -> Option<String> {
        let n = named(strip_cv(t))?.to_string();
        let ty = Type::Named(n.clone());
        (crate::types::size_of(Some(db), &ty) == Some(size) && has_cmp_operator(db, &n, op)).then_some(n)
    };
    match strip_casts(e) {
        Expr::Member { base, offset: 0, ty } if is_int(ty) => {
            let size = crate::types::size_of(Some(db), ty)?;
            let bt = crate::types::ty_of(base, vars);
            cmp_class(&bt, size)?;
            Some((**base).clone())
        }
        Expr::Load { base, offset, ty } if is_int(ty) => {
            let size = crate::types::size_of(Some(db), ty)?;
            let owner = match strip_cv(&crate::types::ty_of(base, vars)) {
                Type::Ptr(p) => named(strip_cv(p))?.to_string(),
                _ => return None,
            };
            // the member at that offset whose type is a comparable class of the scalar's size
            let c = sig_class_at(db, &owner, *offset, size, op)?;
            Some(Expr::Load { base: base.clone(), offset: *offset, ty: Type::Named(c) })
        }
        _ => None,
    }
}

fn is_int(t: &Type) -> bool {
    matches!(strip_cv(t), Type::Int { .. })
}

fn sig_class_at(db: &TypeDb, owner: &str, off: i32, size: u32, op: BinOp) -> Option<String> {
    let (path, _) = crate::types::field_path(db, owner, off, size)?;
    // the scalar is the first member of a member object: [.., Field(obj, owner'), Field(value, C)]
    let n = path.len();
    if n < 2 {
        return None;
    }
    let crate::types::PathElem::Field(_, c) = &path[n - 1] else { return None };
    let ty = Type::Named(c.clone());
    if crate::types::size_of(Some(db), &ty) != Some(size) || !has_cmp_operator(db, c, op) {
        return None;
    }
    crate::types::field_path_of_type(db, owner, off, &ty)?;
    Some(c.clone())
}

/// `a.value != b.value` on one-member classes with comparison operators is `a != b`: the
/// operator's reference parameters keep the compiler from reusing the loaded member later
/// (a by-value copy of `a` passed after the test reloads it).
pub fn object_compares(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>) {
    let Some(db) = db else { return };
    Stmt::rewrite_exprs(body, &mut |e| {
        let Expr::Binary { op, l, r, ty } = e else { return };
        if !matches!(op, BinOp::Eq | BinOp::Ne) {
            return;
        }
        let (Some(a), Some(b)) = (whole_object(l, vars, db, *op), whole_object(r, vars, db, *op)) else { return };
        let (ta, tb) = (crate::types::ty_of(&a, vars), crate::types::ty_of(&b, vars));
        if named(strip_cv(&ta)).map(crate::sig::norm_name) != named(strip_cv(&tb)).map(crate::sig::norm_name) {
            return;
        }
        *e = Expr::Binary { op: *op, l: Box::new(a), r: Box::new(b), ty: ty.clone() };
    });
}
