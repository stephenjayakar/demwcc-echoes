//! Untyped pointers that walk objects of a class: a local `void*` whose every typed view is the
//! address an inline accessor of one class computes from its `this` (`(const H*)((char*)p + 1)`
//! for `const H& C::GetH() const { return *reinterpret_cast<const H*>(this + 1); }`) is a
//! pointer to that class, so the accessors (and the inlines built on them) fold on it.

use crate::probe::CallKind;
use crate::template::{HoleKind, Shape};
use crate::InlineLib;
use mwdec_lift::ir::*;
use mwdec_core::{Type, TypeDb};

fn strip_casts(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip_casts(e),
        e => e,
    }
}

fn untyped_ptr(t: &Type) -> bool {
    match strip_cv(t) {
        Type::Ptr(p) => matches!(strip_cv(p), Type::Void | Type::Unknown { .. } | Type::Int { size: 1, .. } | Type::Char),
        _ => false,
    }
}

/// (var, byte offset) of `p`, `(char*)p + k`, `&p->[k]`.
fn var_offset(e: &Expr) -> Option<(VarId, i64)> {
    match strip_casts(e) {
        Expr::AddrOf(x) => match &**x {
            Expr::Load { base, offset, ty: Type::Unknown { size: 0 } } => match strip_casts(base) {
                Expr::Var(v) => Some((*v, *offset as i64)),
                _ => None,
            },
            _ => None,
        },
        Expr::Var(v) => Some((*v, 0)),
        Expr::Binary { op: BinOp::Add, l, r, .. } => match strip_casts(l) {
            Expr::Var(v) => Some((*v, r.as_int()?)),
            _ => None,
        },
        _ => None,
    }
}

/// Retype the walking pointers of `ir`; returns how many.
pub fn apply(ir: &mut IrFunction, lib: &InlineLib, db: &TypeDb) -> usize {
    let cands: Vec<usize> = (0..ir.vars.len()).filter(|&v| matches!(ir.vars[v].kind, VarKind::Local) && untyped_ptr(&ir.vars[v].ty)).collect();
    if cands.is_empty() {
        return 0;
    }
    // accessors returning the object at a constant offset from `this`: (class, offset, returned class)
    let mut accessors: Vec<(String, i64, String)> = vec![];
    for t in &lib.templates {
        if !matches!(t.kind, CallKind::Method) || !t.ret_ref || t.holes.len() != 1 {
            continue;
        }
        let Some(HoleKind::Obj { class, ptr: true, .. }) = t.holes.first() else { continue };
        let Shape::Scalar(Expr::AddrOf(x)) = &t.shape else { continue };
        let Expr::Load { base, offset, .. } = &**x else { continue };
        if !matches!(**base, Expr::Var(0)) || *offset <= 0 {
            continue;
        }
        let ret = match strip_cv(&t.sig.ret) {
            Type::Ref(r) | Type::Ptr(r) => strip_cv(r).clone(),
            _ => continue,
        };
        let Some(rc) = crate::util::class_name(&ret, db) else { continue };
        accessors.push((class.clone(), *offset as i64, mwdec_lift::sig::norm_name(&rc)));
    }
    if accessors.is_empty() {
        return 0;
    }
    let mut n = 0;
    for v in cands {
        // every typed view of the pointer: (offset, viewed class)
        let mut views: Vec<(i64, String)> = vec![];
        let mut other = false;
        Stmt::walk_exprs(&ir.body, &mut |e| {
            // the object a method runs on
            if let Expr::Call { callee: Callee::Method { this, sig, .. }, .. } = e {
                if let (Some((w, k)), Some(c)) = (var_offset(this), sig.this_class.as_ref()) {
                    if w == v {
                        views.push((k, mwdec_lift::sig::norm_name(c)));
                    }
                }
            }
            if let Expr::Cast { ty, e: inner } = e {
                if let Some((w, k)) = var_offset(inner) {
                    if w == v {
                        if let Type::Ptr(p) = strip_cv(ty) {
                            match crate::util::class_name(strip_cv(p), db) {
                                Some(c) => views.push((k, mwdec_lift::sig::norm_name(&c))),
                                None if k != 0 => other = true,
                                None => {}
                            }
                        }
                    }
                }
            }
        });
        if other || views.is_empty() {
            continue;
        }
        let mut class: Option<&String> = None;
        let mut ok = true;
        for (k, vc) in &views {
            let cs: Vec<&String> = accessors.iter().filter(|(_, o, rc)| o == k && rc == vc).map(|(c, _, _)| c).collect();
            match (cs.as_slice(), class) {
                ([c], None) => class = Some(c),
                ([c], Some(prev)) if *c == prev => {}
                _ => ok = false,
            }
        }
        if let (true, Some(c)) = (ok, class) {
            ir.vars[v].ty = Type::Ptr(Box::new(Type::Const(Box::new(Type::Named(c.clone())))));
            n += 1;
        }
    }
    n
}

/// A frame buffer the lifter left untyped that a constructor builds whole (`buf = C(args)`),
/// where `C` can't be assigned (a const or reference member): the buffer is that object, built
/// in its declaration. Returns how many.
pub fn constructed_buffers(ir: &mut IrFunction, db: &TypeDb) -> usize {
    // the buffer a statement builds whole: `buf = C(..)`, `*(C*)&buf = C(..)`
    fn built(dst: &Expr) -> Option<VarId> {
        match dst {
            Expr::Var(v) => Some(*v),
            Expr::Load { base, offset: 0, .. } => match strip_casts(base) {
                Expr::AddrOf(x) => match &**x {
                    Expr::Var(v) => Some(*v),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }
    let mut found: Vec<(VarId, Type)> = vec![];
    for s in &ir.body {
        let Stmt::Assign { dst, src: Expr::Construct { class, .. } } = s else { continue };
        let Some(v) = built(dst) else { continue };
        let v = &v;
        // (untyped, or typed from a smaller first member)
        let VarKind::Stack { size, .. } = ir.vars[*v].kind else { continue };
        let typed_smaller = match &ir.vars[*v].ty {
            Type::Unknown { .. } => true,
            t @ Type::Named(_) => mwdec_lift::types::size_of(Some(db), t).is_some_and(|z| z < size),
            _ => false,
        };
        if !typed_smaller || found.iter().any(|(w, _)| w == v) {
            continue;
        }
        let Some(cn) = crate::util::class_name(class, db) else { continue };
        let Some(c) = mwdec_lift::sig::find_class(db, &cn) else { continue };
        let unassignable = c.fields.iter().any(|f| matches!(f.ty, Type::Const(_) | Type::Ref(_)));
        if unassignable && mwdec_lift::types::size_of(Some(db), class) == Some(size) {
            found.push((*v, Type::Named(cn)));
        }
    }
    for (v, t) in &found {
        ir.vars[*v].ty = t.clone();
    }
    if !found.is_empty() {
        // (the buffer itself is now the object)
        for s in ir.body.iter_mut() {
            if let Stmt::Assign { dst, src: Expr::Construct { .. } } = s {
                if let Some(v) = built(dst).filter(|v| found.iter().any(|(w, _)| w == v)) {
                    *dst = Expr::Var(v);
                }
            }
        }
    }
    found.len()
}
