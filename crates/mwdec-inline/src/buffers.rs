//! Untyped stack buffers that are objects: a stack slot the lifter could only type by its size
//! whose address is passed as a `const C&` / `C*` argument (or stored into a `const C*`
//! member of such an object) with `sizeof(C)` equal to the slot is an object of class `C`, so
//! the member-wise stores that build it can be folded into its constructor.

use crate::util::{class_name, strip};
use mwdec_core::{Type, TypeDb};
use mwdec_lift::{Callee, Expr, IrFunction, Stmt, Var, VarId, VarKind};
use std::collections::HashMap;

fn untyped_slot(vars: &[Var], v: VarId) -> Option<u32> {
    let var = vars.get(v)?;
    match (&var.kind, strip(&var.ty)) {
        (VarKind::Stack { .. }, Type::Unknown { size }) if *size > 0 => Some(*size),
        _ => None,
    }
}

/// A stack object the lifter typed as a class `B` that the call passes as a derived class `C` of
/// the same size (`CToken` slot passed as `const TToken<CRasterFont>&`): the object is a `C`
/// (identical layouts; the constructors that built it are `C`'s).
fn base_typed_slot(vars: &[Var], v: VarId, c: &str, db: &TypeDb) -> Option<u32> {
    if std::env::var("MWDI_NO_DERIVED_SLOTS").is_ok() {
        return None;
    }
    let var = vars.get(v)?;
    if !matches!(var.kind, VarKind::Stack { .. }) {
        return None;
    }
    let b = class_name(&var.ty, db)?;
    if mwdec_lift::sig::norm_name(&b) == mwdec_lift::sig::norm_name(c) || !derives(db, &b, c, 0) {
        return None;
    }
    let bs = mwdec_lift::sig::find_class(db, &b)?.size;
    Some(bs)
}

fn derives(db: &TypeDb, base: &str, derived: &str, d: u32) -> bool {
    if d > 8 {
        return false;
    }
    let Some(k) = mwdec_lift::sig::find_class(db, derived) else { return false };
    k.bases.iter().any(|x| x.offset == 0 && (mwdec_lift::sig::norm_name(&x.name) == mwdec_lift::sig::norm_name(base) || derives(db, base, &x.name, d + 1)))
}

/// Class pointed to by a pointer/reference parameter type.
fn pointee_class(t: &Type, db: &TypeDb) -> Option<String> {
    match strip(t) {
        Type::Ptr(inner) | Type::Ref(inner) => class_name(inner, db),
        _ => None,
    }
}

/// Retype untyped stack slots as the class objects their uses say they are; returns the
/// number of retyped slots.
pub fn type_object_slots(ir: &mut IrFunction, db: &TypeDb) -> usize {
    let mut total = 0;
    for _round in 0..2 {
        // slot -> candidate classes (a slot used as two different classes stays untyped)
        let mut cand: HashMap<VarId, Vec<String>> = HashMap::new();
        let vars = ir.vars.clone();
        let note = |e: &Expr, t: &Type, cand: &mut HashMap<VarId, Vec<String>>| {
            // (through pointer casts: `*(const C*)&slot` for a reference argument)
            let mut e = e;
            loop {
                match e {
                    Expr::Cast { ty, e: x } if matches!(strip(ty), Type::Ptr(_)) => e = x,
                    Expr::Load { base, offset: 0, .. } if matches!(&**base, Expr::Cast { .. }) => e = base,
                    _ => break,
                }
            }
            if let Expr::AddrOf(x) = e {
                if let Expr::Var(v) = &**x {
                    let pc = pointee_class(t, db);
                    let slot = untyped_slot(&vars, *v).or_else(|| pc.as_deref().and_then(|c| base_typed_slot(&vars, *v, c, db)));
                    if let (Some(size), Some(c)) = (slot, pc) {
                        if mwdec_lift::sig::find_class(db, &c).is_some_and(|k| k.size == size && !k.is_declaration) {
                            cand.entry(*v).or_default().push(c);
                        }
                    }
                }
            }
        };
        Stmt::walk_exprs(&ir.body, &mut |e| {
            if let Expr::Call { callee, args, .. } = e {
                let sig = match callee {
                    Callee::Direct { sig, .. } | Callee::Method { sig, .. } => Some(sig),
                    Callee::Virtual { sig, .. } => sig.as_ref(),
                    _ => None,
                };
                if let Some(sig) = sig {
                    for (a, p) in args.iter().zip(sig.params.iter()) {
                        note(a, &p.ty, &mut cand);
                    }
                }
            }
        });
        // `obj.ptr_member = &slot` with obj an object of known class
        let mut assigns: Vec<(&Expr, &Expr)> = vec![];
        fn collect<'a>(b: &'a [Stmt], out: &mut Vec<(&'a Expr, &'a Expr)>) {
            for s in b {
                match s {
                    Stmt::Assign { dst, src } => out.push((dst, src)),
                    Stmt::If { then, els, .. } => {
                        collect(then, out);
                        collect(els, out);
                    }
                    Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect(body, out),
                    Stmt::For { init, step, body, .. } => {
                        collect(init, out);
                        collect(step, out);
                        collect(body, out);
                    }
                    Stmt::Switch { cases, .. } => {
                        for c in cases {
                            collect(&c.body, out);
                        }
                    }
                    _ => {}
                }
            }
        }
        collect(&ir.body, &mut assigns);
        // an untyped slot copied member by member into a typed object of the same size (a
        // by-value result's temporary copied into the named local): the same class
        if std::env::var("MWDI_NO_COPIED_SLOTS").is_err() {
            fn slot_at(e: &Expr) -> Option<(VarId, i32)> {
                match e {
                    Expr::Member { base, offset, .. } => match &**base {
                        Expr::Var(v) => Some((*v, *offset)),
                        _ => None,
                    },
                    Expr::Load { base, offset, .. } => {
                        let mut b: &Expr = base;
                        while let Expr::Cast { e, .. } = b {
                            b = e;
                        }
                        match b {
                            Expr::AddrOf(x) => match &**x {
                                Expr::Var(v) => Some((*v, *offset)),
                                _ => None,
                            },
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            let mut copies: HashMap<VarId, Vec<Option<String>>> = HashMap::new();
            for (dst, src) in &assigns {
                if let (Some((t, to)), Some((sv, so))) = (slot_at(dst), slot_at(src)) {
                    if to == so && t != sv && untyped_slot(&vars, sv).is_some() {
                        let tc = class_name(&vars[t].ty, db).filter(|c| mwdec_lift::sig::find_class(db, c).is_some_and(|k| Some(k.size) == untyped_slot(&vars, sv)));
                        copies.entry(sv).or_default().push(tc);
                    }
                }
            }
            for (sv, cs) in copies {
                if let Some(Some(c)) = cs.first() {
                    if cs.iter().all(|x| x.as_deref() == Some(c.as_str())) {
                        cand.entry(sv).or_default().push(c.clone());
                    }
                }
            }
        }
        for (dst, src) in assigns {
            if let Expr::Member { base, offset, .. } = dst {
                if let Expr::Var(o) = &**base {
                    if let Some(oc) = class_name(&vars[*o].ty, db) {
                        if let Some((_, ft)) = crate::template::flat_fields(db, &oc).and_then(|f| f.into_iter().find(|(fo, _)| fo == offset)) {
                            note(src, &ft, &mut cand);
                        }
                    }
                }
            }
        }
        let mut n = 0;
        for (v, cs) in cand {
            if cs.iter().all(|c| c == &cs[0]) {
                ir.vars[v].ty = Type::Named(cs[0].clone());
                n += 1;
            }
        }
        total += n;
        if n == 0 {
            break;
        }
    }
    total
}
