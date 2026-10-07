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
            if let Expr::AddrOf(x) = e {
                if let Expr::Var(v) = &**x {
                    if let (Some(size), Some(c)) = (untyped_slot(&vars, *v), pointee_class(t, db)) {
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
