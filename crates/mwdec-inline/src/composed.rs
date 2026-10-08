//! Accessor chains whose constant offsets the compiler folded into one access.
//!
//! A reference-returning inline that dereferences a member pointer plus a constant
//! (`T& front() { return *mStart->get_value(); }`: `&*(*(this + 4) + 8)`) followed by a member
//! read of the referenced object (`front().get()`: `+ 4`) compiles to one load at the summed
//! offset (`*(*(list + 4) + 0xc)`). No template matches that load as a whole and the
//! referenced type is often private to the container (`list::node`), so the draft falls back to
//! raw offsets. Here such a load is split: when its offset is at least the inline's constant
//! and the address `*(base) + constant` matches the inline, the load becomes a read at the
//! remaining offset from the inline's result (`*(&list.front() + 4)`), which the emitter spells
//! through the element type (and its accessors).

use crate::matcher::{make_call, res, Env, M};
use crate::template::{HoleKind, Shape, Template};
use mwdec_core::Type;
use mwdec_lift::{Expr, VarKind};

/// `(hole member offset, constant)` of a dereferencing accessor: `&*(*(h + a) + c)`.
fn deref_accessor(t: &Template) -> Option<(i32, i32)> {
    if !t.ret_ref || t.holes.len() != 1 || !matches!(t.holes[0], HoleKind::Obj { .. }) {
        return None;
    }
    let Shape::Scalar(Expr::AddrOf(inner)) = &t.shape else { return None };
    let Expr::Load { base, offset: c, .. } = &**inner else { return None };
    let Expr::Load { base: hb, offset: a, .. } = &**base else { return None };
    matches!(&**hb, Expr::Var(0)).then_some((*a, *c))
}

/// Split a load whose offset folds a dereferencing accessor's constant (see the module docs).
/// Returns whether `e` was rewritten.
pub fn try_split(e: &mut Expr, env: &Env) -> bool {
    if std::env::var_os("MWDI_NO_COMPOSED").is_some() {
        return false;
    }
    let Expr::Load { base, offset: k, ty } = &*e else { return false };
    let k = *k;
    if k <= 0 {
        return false;
    }
    // the pointer the accessor dereferences, read from the object
    if !matches!(res(base, env.defs), Expr::Load { .. }) {
        return false;
    }
    let trace = std::env::var_os("MWDI_TRACE_COMPOSED").is_some();
    if trace {
        eprintln!("composed: load +{k:#x} of {:?}", res(base, env.defs));
    }
    let mut best: Option<(Expr, i32, i32)> = None;
    for t in &env.lib.templates {
        let Some((_, c)) = deref_accessor(t) else { continue };
        if c <= 0 || c > k {
            continue;
        }
        let Shape::Scalar(pat) = &t.shape else { continue };
        let probe = Expr::AddrOf(Box::new(Expr::Load { base: base.clone(), offset: c, ty: Type::Unknown { size: 0 } }));
        let mut m = M::new(env, t);
        if !m.m(pat, &probe) {
            if trace && t.name.contains("front") {
                eprintln!("composed:   {} (+{c:#x}) no match", t.name);
            }
            continue;
        }
        if trace {
            eprintln!("composed:   {} (+{c:#x}) matched", t.name);
        }
        let Some((args, extra)) = m.finalize(0) else { continue };
        // the class's own members use its fields directly
        if args.first().is_some_and(|a| matches!(res(a, env.defs), Expr::Var(v) if env.vars[*v].kind == VarKind::This)) {
            continue;
        }
        // (an accessor on the object itself before one on a temporary built from it:
        // `list.front()`, not `*list.begin()`)
        let nested = args.iter().any(|a| {
            let mut c = false;
            a.walk(&mut |x| c |= matches!(x, Expr::Call { .. }));
            c
        });
        let sc = crate::matcher::use_score(t, extra, false, &args) - if nested { 100 } else { 0 };
        if trace {
            eprintln!("composed:   {} finalized score {sc} args {:?}", t.name, args);
        }
        if best.as_ref().map_or(true, |(_, b, _)| sc > *b) {
            let mut call = make_call(t, args);
            if let (Expr::Call { ret, .. }, Type::Ref(inner)) = (&mut call, crate::util::strip(&t.sig.ret)) {
                *ret = (**inner).clone();
            }
            best = Some((call, sc, c));
        }
    }
    let Some((call, sc, c)) = best else { return false };
    if sc < -50 {
        return false;
    }
    let ty = ty.clone();
    *e = Expr::Load { base: Box::new(Expr::AddrOf(Box::new(call))), offset: k - c, ty };
    true
}
