//! Function-local static objects built on first use. MWCC guards a local static's construction
//! with a flag (`if (init == 0) { v.x = 1.f; v.y = 1.f; v.z = 1.f; init = 1; }`); the lifter sees
//! a byte buffer filled member-wise. When the function uses the buffer as an object of a class
//! of its size (copied whole into a member of that class), the guarded stores are that class's
//! construction: `static CVector3f v(1.f, 1.f, 1.f);` (`IrFunction::static_ctors`).

use crate::matcher::{build_defs, explain_object, Env, Index};
use crate::util::*;
use crate::InlineLib;
use mwdec_core::{Type, TypeDb};
use mwdec_lift::{Expr, IrFunction, Stmt, Var};
use std::collections::BTreeMap;

/// A function-local static's symbol (`init$709`, `unitScale$708`).
fn local_static(sym: &str) -> bool {
    let base = mwdec_lift::sig::strip_dtk_suffix(sym);
    base.split_once('$').is_some_and(|(n, rest)| !n.is_empty() && rest.chars().next().is_some_and(|c| c.is_ascii_digit()))
}

fn global_of(e: &Expr) -> Option<&str> {
    let e = match e {
        Expr::Cast { e, .. } => &**e,
        e => e,
    };
    match e {
        Expr::Global { symbol, .. } => Some(symbol),
        _ => None,
    }
}

/// `if (flag == 0) { obj.m = v; ...; flag = 1; }`: (flag, object, size, member values).
fn guard(s: &Stmt) -> Option<(String, String, u32, BTreeMap<i32, Expr>)> {
    let Stmt::If { cond, then, els } = s else { return None };
    if !els.is_empty() || then.len() < 2 {
        return None;
    }
    let Expr::Binary { op: mwdec_lift::BinOp::Eq, l, r, .. } = cond else { return None };
    let flag = global_of(l).filter(|_| r.as_int() == Some(0))?.to_string();
    if !local_static(&flag) {
        return None;
    }
    let (last, stores) = then.split_last()?;
    match last {
        Stmt::Assign { dst, src } if global_of(dst) == Some(flag.as_str()) && src.as_int() == Some(1) => {}
        _ => return None,
    }
    let mut obj: Option<(String, u32)> = None;
    let mut m = BTreeMap::new();
    for st in stores {
        let Stmt::Assign { dst: Expr::Member { base, offset, .. }, src } = st else { return None };
        let Expr::Global { symbol, ty } = &**base else { return None };
        let size = match strip(ty) {
            Type::Unknown { size } => *size as u32,
            _ => return None,
        };
        match &obj {
            None => obj = Some((symbol.clone(), size)),
            Some((o, _)) if o == symbol => {}
            _ => return None,
        }
        if src.has_call() || m.insert(*offset, src.clone()).is_some() {
            return None;
        }
    }
    let (obj, size) = obj?;
    local_static(&obj).then_some((flag, obj, size, m))
}

fn each_stmt(b: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for s in b {
        f(s);
        match s {
            Stmt::If { then, els, .. } => {
                each_stmt(then, f);
                each_stmt(els, f);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => each_stmt(body, f),
            Stmt::For { init, step, body, .. } => {
                each_stmt(init, f);
                each_stmt(step, f);
                each_stmt(body, f);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    each_stmt(&c.body, f);
                }
            }
            _ => {}
        }
    }
}

/// A class-typed field (at any depth) of `cls` starting at `off` with a class of `size` bytes.
fn field_class_at(db: &TypeDb, cls: &str, off: i32, size: u32, depth: u32) -> Option<String> {
    if depth > 6 {
        return None;
    }
    let c = mwdec_lift::sig::find_class(db, cls)?;
    for f in &c.fields {
        let fo = f.offset as i32;
        let Some(fc) = class_name(&f.ty, db) else { continue };
        let fsize = mwdec_lift::types::size_of(Some(db), &f.ty).unwrap_or(0);
        if fo == off && fsize == size {
            return Some(fc);
        }
        if fo <= off && off < fo + fsize as i32 {
            if let Some(x) = field_class_at(db, &fc, off - fo, size, depth + 1) {
                return Some(x);
            }
        }
    }
    for b in &c.bases {
        let bo = b.offset as i32;
        if let Some(x) = field_class_at(db, &b.name, off - bo, size, depth + 1) {
            return Some(x);
        }
    }
    None
}

/// The class the function copies the static object into: a member of that class (of the
/// object's size) receiving the object's members at their own offsets.
fn class_of_use(body: &[Stmt], obj: &str, size: u32, vars: &[Var], db: &TypeDb) -> Option<String> {
    let mut found: Option<String> = None;
    each_stmt(body, &mut |s| {
        if found.is_some() {
            return;
        }
        let Stmt::Assign { dst, src: Expr::Member { base, offset: so, .. } } = s else { return };
        if global_of(base) != Some(obj) {
            return;
        }
        let (b, k) = match dst {
            Expr::Load { base, offset, .. } => (&**base, *offset),
            _ => return,
        };
        let bt = match b {
            Expr::Var(v) => vars[*v].ty.clone(),
            _ => return,
        };
        let Some(outer) = mwdec_lift::pointee(strip(&bt)).and_then(|t| class_name(t, db)) else { return };
        found = field_class_at(db, &outer, k - so, size, 0);
    });
    let cls = found?;
    (mwdec_lift::types::size_of(Some(db), &Type::Named(cls.clone())) == Some(size)).then_some(cls)
}

pub fn apply(ir: &mut IrFunction, lib: &InlineLib, db: &TypeDb, idx: &Index) -> usize {
    if std::env::var_os("MWDI_NO_STATIC_CTORS").is_some() {
        return 0;
    }
    let mut n = 0;
    let mut i = 0;
    while i < ir.body.len() {
        let Some((_flag, obj, size, m)) = guard(&ir.body[i]) else {
            i += 1;
            continue;
        };
        let Some(cls) = class_of_use(&ir.body, &obj, size, &ir.vars, db) else {
            i += 1;
            continue;
        };
        // every member set by the guarded stores
        let whole = crate::template::flat_fields(db, &cls).is_some_and(|f| f.len() == m.len() && f.iter().all(|(o, _)| m.contains_key(o)));
        let defs = build_defs(&ir.body, &ir.vars);
        let vars = ir.vars.clone();
        let env = Env { db, vars: &vars, defs: &defs, lib, objects: &idx.objects };
        let call = if whole { explain_object(&env, &cls, &m, 0).map(|(c, _)| c) } else { None };
        let Some(call) = call else {
            i += 1;
            continue;
        };
        ir.body.remove(i);
        let ct = Type::Named(cls.clone());
        Stmt::rewrite_exprs(&mut ir.body, &mut |e| {
            if let Expr::Global { symbol, ty } = e {
                if *symbol == obj {
                    *ty = ct.clone();
                }
            }
        });
        for g in ir.globals.iter_mut().filter(|g| g.symbol == obj) {
            g.ty = ct.clone();
        }
        ir.static_ctors.push((obj, call));
        n += 1;
    }
    n
}
