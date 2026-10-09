//! By-value results bound to a reference local (`const CVector3f& pos = xf.GetTranslation();`):
//! the compiler materializes the object (stores nobody reads, kept by the lifter as
//! `IrFunction::dead_stores`) and then uses its components from registers. A run of dead stores
//! whose values are the components of an object-returning accessor template becomes that local,
//! and the function's reads of the same values read its members.

use crate::matcher::{build_defs, make_call, teq, Env, M};
use crate::template::{HoleKind, Shape};
use crate::InlineLib;
use mwdec_core::{Type, TypeDb};
use mwdec_lift::{Expr, IrFunction, Stmt, Var, VarKind};

pub fn apply(ir: &mut IrFunction, lib: &InlineLib, db: &TypeDb) -> usize {
    if ir.dead_stores.len() < 2 {
        return 0;
    }
    let mut dead = ir.dead_stores.clone();
    dead.sort_by_key(|d| d.offset);
    // runs of adjacent stores
    let mut runs: Vec<Vec<mwdec_lift::DeadStackStore>> = vec![];
    for d in dead {
        match runs.last_mut() {
            Some(r) if r.last().is_some_and(|l| l.offset + l.size as i32 == d.offset) => r.push(d),
            _ => runs.push(vec![d]),
        }
    }
    let idx = crate::matcher::index(lib);
    let mut n = 0;
    for run in runs.into_iter().filter(|r| r.len() >= 2) {
        let defs = build_defs(&ir.body, &ir.vars);
        let vars = ir.vars.clone();
        let env = Env { db, vars: &vars, defs: &defs, lib, objects: &idx.objects };
        let start = run[0].offset;
        let mut found: Option<(Expr, String, Vec<(i32, Type, Expr)>)> = None;
        for t in &lib.templates {
            let Shape::Object { class, comps } = &t.shape else { continue };
            // an accessor of the object itself (no other arguments)
            if t.holes.len() != 1 || !matches!(t.holes[0], HoleKind::Obj { ptr: true, .. }) || comps.len() != run.len() {
                continue;
            }
            let mut m = M::new(&env, t);
            let ok = comps.iter().zip(&run).all(|(c, d)| c.off == d.offset - start && mwdec_lift::scalar_size(&c.ty) == Some(d.size) && m.m(&c.pat, &d.value));
            if !ok {
                continue;
            }
            let Some((args, _)) = m.finalize(0) else { continue };
            let call = make_call(t, args);
            let parts = comps.iter().zip(&run).map(|(c, d)| (c.off, c.ty.clone(), d.value.clone())).collect();
            found = Some((call, class.clone(), parts));
            break;
        }
        let Some((call, class, parts)) = found else { continue };
        let size = run.iter().map(|d| d.size).sum();
        let v = ir.vars.len();
        ir.vars.push(Var { name: format!("local_{start:x}"), ty: Type::Ref(Box::new(Type::Const(Box::new(Type::Named(class.clone()))))), kind: VarKind::Stack { offset: start, size } });
        // reads of the component values read the local's members
        let mut hits = 0;
        let mut first: Option<usize> = None;
        for (k, s) in ir.body.iter_mut().enumerate() {
            let before = hits;
            Stmt::rewrite_exprs(std::slice::from_mut(s), &mut |e| {
                if !matches!(e, Expr::Load { .. } | Expr::Member { .. } | Expr::Call { .. } | Expr::Var(_)) {
                    return;
                }
                let u = crate::matcher::unfold_pub(crate::matcher::res(e, &defs), &env).unwrap_or_else(|| crate::matcher::res(e, &defs).clone());
                for (off, ty, val) in &parts {
                    if teq(&u, val, &defs) && !matches!(e, Expr::Var(_)) {
                        *e = Expr::Member { base: Box::new(Expr::Var(v)), offset: *off, ty: ty.clone() };
                        hits += 1;
                        return;
                    }
                }
            });
            if hits > before && first.is_none() {
                first = Some(k);
            }
        }
        let Some(at) = first else {
            ir.vars.pop();
            continue;
        };
        ir.body.insert(at, Stmt::Assign { dst: Expr::Var(v), src: call });
        n += 1;
    }
    n
}

/// A reference local of [`apply`] whose only use is one argument of the next statement's call
/// (`operator+(pos, offset)`): the dead stores were the temporary that binds the argument, so
/// the source passed the accessor's result directly (`GetTranslation() + offset`).
pub fn unbind_single_use(ir: &mut IrFunction) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i + 1 < ir.body.len() {
        let v = match &ir.body[i] {
            Stmt::Assign { dst: Expr::Var(v), src: Expr::Call { .. } }
                if matches!(ir.vars[*v].kind, VarKind::Stack { .. }) && matches!(ir.vars[*v].ty, Type::Ref(_)) && ir.vars[*v].name.starts_with("local_") =>
            {
                *v
            }
            _ => {
                i += 1;
                continue;
            }
        };
        let mut uses = 0;
        Stmt::walk_exprs(&ir.body[i + 1..], &mut |x| {
            if matches!(x, Expr::Var(w) if *w == v) {
                uses += 1;
            }
        });
        let Stmt::Assign { src: init, .. } = ir.body[i].clone() else { unreachable!() };
        let mut done = false;
        if uses == 1 {
            let mut next = ir.body[i + 1].clone();
            let mut k = 0;
            Stmt::rewrite_exprs(std::slice::from_mut(&mut next), &mut |x| {
                if let Expr::Call { args, .. } | Expr::Construct { args, .. } = x {
                    for a in args.iter_mut() {
                        let is_v = match a {
                            Expr::Var(w) => *w == v,
                            Expr::AddrOf(y) => matches!(&**y, Expr::Var(w) if *w == v),
                            _ => false,
                        };
                        if is_v {
                            *a = init.clone();
                            k += 1;
                        }
                    }
                }
            });
            if k == 1 {
                ir.body[i + 1] = next;
                ir.body.remove(i);
                n += 1;
                done = true;
            }
        }
        if !done {
            i += 1;
        }
    }
    n
}
