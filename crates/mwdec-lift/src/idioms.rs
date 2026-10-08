//! C++ idioms MWCC lowers into plain code: `new` expressions, constructor initializer lists and
//! implicit vtable-pointer stores, destructor wrappers (null check, delete flag, base/member
//! destructor calls).

use crate::ir::*;
use crate::sig;
use crate::types;
use mwdec_core::{Type, TypeDb};
use std::collections::{HashMap, HashSet};

/// Every statement of a body, nested ones included.
fn each_stmt(body: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for s in body {
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

/// Does the function construct an object of another class into its first parameter, with an
/// unknown return type? Then that "parameter" is the hidden struct-return pointer (the mangled
/// name has no return type and the headers don't declare the function).
pub fn constructs_into_param0(ir: &IrFunction) -> bool {
    if !sig::ret_unknown(&ir.sig) && !matches!(ir.sig.ret, Type::Void | Type::Unknown { .. }) {
        return false;
    }
    if ir.vars.iter().any(|v| v.kind == VarKind::StructRet) || sig::is_ctor(&ir.sig) || sig::is_dtor(&ir.sig) {
        return false;
    }
    let Some(&p0) = ir.params.first() else { return false };
    let p0_class = named(&ir.vars[p0].ty).map(sig::norm_name).or_else(|| pointee(&ir.vars[p0].ty).and_then(named).map(sig::norm_name));
    let mut found = false;
    // a store through a pointer/reference-to-const first parameter: C++ can't write there, so
    // that register is the hidden struct-return pointer
    let p0_const = ir.sig.params.first().is_some_and(|p| matches!(strip_cv(&p.ty), Type::Ref(x) | Type::Ptr(x) if matches!(**x, Type::Const(_))));
    if p0_const {
        let mut stores = false;
        each_stmt(&ir.body, &mut |s| {
            if let Stmt::Assign { dst, .. } = s {
                let into = match dst {
                    Expr::Var(v) => *v == p0,
                    Expr::Load { base, .. } | Expr::Member { base, .. } => matches!(&**base, Expr::Var(v) if *v == p0),
                    _ => false,
                };
                stores |= into;
            }
        });
        if stores {
            return true;
        }
    }
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::Call { callee: Callee::Method { sig: s, this, .. }, .. } = e {
            if sig::is_ctor(s) {
                let on_p0 = match &**this {
                    Expr::Var(v) => *v == p0,
                    Expr::AddrOf(x) => matches!(&**x, Expr::Var(v) if *v == p0),
                    _ => false,
                };
                if on_p0 && s.this_class.as_deref().map(sig::norm_name) != p0_class {
                    found = true;
                }
            }
        }
    });
    found
}

/// Name prefix of the stand-in class a guessed struct return gets when no known class fits
/// (the emitter defines it from the constructor signature on the `return`).
pub const STANDIN_RET: &str = "__mwdec_ret";

/// A guessed struct return of unknown class filled member by member at the end of the function
/// (`__return->x0 = a; __return->x4 = b; ...; return;`, stores tiling the object from offset 0)
/// becomes `return S(a, b, ...)` of a stand-in class whose constructor stores each member: the
/// compiler builds such a temporary directly in the caller's object, as the target does.
pub fn standin_sret(ir: &mut IrFunction, db: Option<&TypeDb>) -> bool {
    let Some(sret) = ir.vars.iter().position(|v| v.kind == VarKind::StructRet) else { return false };
    if !matches!(pointee(&ir.vars[sret].ty), Some(Type::Unknown { .. })) {
        return false;
    }
    let mut body = ir.body.clone();
    if matches!(body.last(), Some(Stmt::Return(None))) {
        body.pop();
    }
    let first = body.iter().position(|s| stmt_mentions(s, sret)).unwrap_or(body.len());
    let mut fields: Vec<(i32, Type, Expr)> = vec![];
    // a single word stored whole (already folded into `return x;`): a one-member object
    if first == body.len() {
        if let Some(Stmt::Return(Some(e))) = body.last() {
            let t = types::ty_of(e, &ir.vars);
            let t = match strip_cv(&t) {
                Type::Unknown { size: 4 } => Type::Int { size: 4, signed: true },
                Type::Int { size: 4, .. } | Type::Float { size: 4 } | Type::Ptr(_) => strip_cv(&t).clone(),
                _ => return false,
            };
            if !stmt_mentions(body.last().unwrap(), sret) && !e.has_call() {
                let e = e.clone();
                body.pop();
                fields.push((0, t, e));
                let n = body.len();
                return finish_standin(ir, body, n, fields);
            }
        }
        return false;
    }
    for s in &body[first..] {
        match s {
            Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } if matches!(&**base, Expr::Var(v) if *v == sret) && !src.uses_var(sret) && !src.has_call() => {
                if fields.iter().any(|f| f.0 == *offset) {
                    return false;
                }
                // (an untyped word store: an `int` member)
                let ty = if matches!(strip_cv(ty), Type::Unknown { size: 4 }) { Type::Int { size: 4, signed: true } } else { ty.clone() };
                fields.push((*offset, ty, src.clone()));
            }
            _ => return false,
        }
    }
    if fields.is_empty() {
        return false;
    }
    fields.sort_by_key(|f| f.0);
    let mut at = 0i32;
    for (o, t, _) in &fields {
        let Some(sz) = types::size_of(db, t).filter(|&z| z > 0) else { return false };
        if *o != at || !matches!(strip_cv(t), Type::Int { .. } | Type::Float { .. } | Type::Ptr(_)) {
            return false;
        }
        at += sz as i32;
    }
    finish_standin(ir, body, first, fields)
}

fn finish_standin(ir: &mut IrFunction, mut body: Vec<Stmt>, first: usize, fields: Vec<(i32, Type, Expr)>) -> bool {
    let Some(sret) = ir.vars.iter().position(|v| v.kind == VarKind::StructRet) else { return false };
    let name = format!("{STANDIN_RET}_{}", ir.symbol.split("__").next().unwrap_or("").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_').collect::<String>());
    let class = Type::Named(name.clone());
    let ctor = mwdec_core::FuncSig {
        qualified_name: format!("{name}::{name}"),
        mangled: None,
        ret: Type::Void,
        params: fields.iter().map(|(_, t, _)| mwdec_core::Param { name: None, ty: t.clone() }).collect(),
        this_class: Some(name.clone()),
        is_const: false,
        is_static: false,
        is_virtual: false,
        variadic: false,
    };
    body.truncate(first);
    body.push(Stmt::Return(Some(Expr::Construct { class: class.clone(), ctor: Some(ctor), args: fields.into_iter().map(|f| f.2).collect() })));
    ir.body = body;
    ir.vars[sret].ty = t_ptr(class.clone());
    ir.sig.ret = class;
    true
}

/// A guessed struct return (`StructRet` pointing at an unknown type) takes the class of the
/// object constructed into it.
fn type_guessed_sret(ir: &mut IrFunction, db: Option<&TypeDb>) {
    let Some(sret) = ir.vars.iter().position(|v| v.kind == VarKind::StructRet) else { return };
    if !matches!(pointee(&ir.vars[sret].ty), Some(Type::Unknown { .. })) {
        return;
    }
    let mut cls = None;
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::Call { callee: Callee::Method { sig: s, this, .. }, .. } = e {
            if cls.is_none() && sig::is_ctor(s) && matches!(&**this, Expr::Var(v) if *v == sret) {
                cls = s.this_class.clone();
            }
        }
    });
    // or of an object stored there whole
    if cls.is_none() {
        each_stmt(&ir.body, &mut |s| {
            if let Stmt::Assign { dst: Expr::Load { base, offset: 0, .. }, src: Expr::Construct { class, .. } } = s {
                if cls.is_none() && matches!(&**base, Expr::Var(v) if *v == sret) {
                    cls = named(class).map(|n| n.to_string());
                }
            }
        });
    }
    // or, filled member by member, the class of a by-reference parameter that spans the stores
    // (`T f(const T& a, const T& b)` helpers)
    if let (None, Some(db)) = (&cls, db) {
        let mut extent = 0i32;
        let mut any = false;
        each_stmt(&ir.body, &mut |s| {
            if let Stmt::Assign { dst: Expr::Load { base, offset, ty } | Expr::Member { base, offset, ty }, .. } = s {
                if matches!(&**base, Expr::Var(v) if *v == sret) {
                    any = true;
                    extent = extent.max(offset + types::size_of(Some(db), ty).unwrap_or(4) as i32);
                }
            }
        });
        if any {
            cls = ir.sig.params.iter().find_map(|p| {
                let t = match strip_cv(&p.ty) {
                    Type::Ref(x) => strip_cv(x).clone(),
                    _ => return None,
                };
                let n = named(&t)?.to_string();
                // (the stores fill exactly such an object)
                (types::size_of(Some(db), &t)? as i32 == extent && types::is_aggregate(Some(db), &t)).then_some(n)
            });
        }
    }
    // or the one class of the context whose layout and constructor fit the stores (a variant:
    // nothing in the code names it)
    if let (None, Some(db)) = (&cls, db) {
        let mut stores: Vec<(i32, Expr)> = vec![];
        let mut widths: Vec<(i32, Option<u32>)> = vec![];
        each_stmt(&ir.body, &mut |s| {
            if let Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } = s {
                if matches!(&**base, Expr::Var(v) if *v == sret) {
                    stores.push((*offset, src.clone()));
                    widths.push((*offset, scalar_size(ty)));
                }
            }
        });
        let vars = ir.vars.clone();
        if !stores.is_empty() && stores.len() <= 8 {
            let mut found: Vec<String> = vec![];
            for (name, c) in &db.classes {
                if c.is_declaration || name.contains('<') || found.len() > 1 {
                    continue;
                }
                let mut fields = vec![];
                if !flat_fields(db, name, 0, &mut fields, 0) || fields.len() != stores.len() {
                    continue;
                }
                // each member is one store of its own width and kind
                let fits = fields.iter().all(|(o, ft)| {
                    stores.iter().any(|(so, e)| so == o && is_float(&types::ty_of(e, &vars)) == is_float(ft))
                        && widths.iter().any(|(wo, w)| wo == o && *w == types::size_of(Some(db), ft))
                });
                if fits && construct_from_stores(&stores, &Type::Named(name.clone()), Some(db)).is_some() {
                    found.push(name.clone());
                }
            }
            if found.len() == 1 && crate::variants::alt(crate::variants::SRET_CLASS_BY_LAYOUT) {
                cls = found.pop();
            }
        }
    }
    if let Some(c) = cls {
        ir.vars[sret].ty = t_ptr(Type::Named(c.clone()));
        ir.sig.ret = Type::Named(c);
    }
}

pub fn apply(ir: &mut IrFunction, db: Option<&TypeDb>) {
    type_guessed_sret(ir, db);
    let vars = ir.vars.clone();
    fold_new(&mut ir.body, &vars);
    let this = ir.this_var;
    let is_ctor = sig::is_ctor(&ir.sig);
    let is_dtor = sig::is_dtor(&ir.sig);
    if is_ctor || is_dtor {
        drop_vptr_stores(&mut ir.body);
    } else {
        // vtable pointer stores into stack objects come from inlined ctors/dtors: implicit
        let stack: Vec<bool> = ir.vars.iter().map(|v| matches!(v.kind, VarKind::Stack { .. })).collect();
        Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
            b.retain(|s| match s {
                Stmt::Assign { dst, src } if is_vtable_addr(src) => {
                    let base = match dst {
                        Expr::Var(v) => Some(*v),
                        Expr::Member { base, .. } => match &**base {
                            Expr::Var(v) => Some(*v),
                            _ => None,
                        },
                        _ => None,
                    };
                    !base.map_or(false, |v| stack[v])
                }
                _ => true,
            })
        });
    }
    if let (true, Some(this)) = (is_ctor, this) {
        ctor_init_list(ir, this, db);
    }
    if let (true, Some(this)) = (is_dtor, this) {
        dtor_unwrap(ir, this);
    }
    let vars = ir.vars.clone();
    if let Some(db) = db {
        drop_frame_object_dtor_calls(&mut ir.body, &vars, db);
        // a value kept in a register across the destructor call is returned directly
        return_kept_values(&mut ir.body, &vars);
    }
    let kept = forward_stack_objects(&mut ir.body, &vars, db);
    for &v in &kept {
        // bound to a reference instead: its temporary is created before the objects of the
        // statement that uses it (see `forward_breaks_layout`)
        let t = ir.vars[v].ty.clone();
        if named(&t).is_some() {
            ir.vars[v].ty = Type::Ref(Box::new(Type::Const(Box::new(t))));
        }
    }
    let more = drop_dead_stack_stores_kept(&mut ir.body, &vars);
    ir.dead_stores.extend(more);
    for (n, d) in ir.dead_stores.iter_mut().enumerate() {
        d.order = n;
    }
    if let Some(sret) = ir.vars.iter().position(|v| v.kind == VarKind::StructRet) {
        let rt = ir.sig.ret.clone();
        fold_struct_return(&mut ir.body, sret, &rt, db);
    }
    let body = ir.body.clone();
    untype_undeclarable(&body, &mut ir.vars, db);
}

/// Scalar leaf members of a class in layout order: (offset, type).
pub(crate) fn flat_fields(db: &TypeDb, cls: &str, base: i32, out: &mut Vec<(i32, Type)>, depth: u32) -> bool {
    let Some(c) = sig::find_class(db, cls) else { return false };
    if depth > 8 {
        return false;
    }
    for b in &c.bases {
        if !flat_fields(db, &b.name, base + b.offset as i32, out, depth + 1) {
            return false;
        }
    }
    for f in &c.fields {
        if f.bitfield.is_some() {
            return false;
        }
        let rt = types::resolve(Some(db), &f.ty).into_owned();
        if types::is_aggregate(Some(db), &rt) {
            match named(&rt) {
                Some(n) => {
                    if !flat_fields(db, n, base + f.offset as i32, out, depth + 1) {
                        return false;
                    }
                }
                None => return false,
            }
        } else {
            out.push((base + f.offset as i32, rt));
        }
    }
    true
}

/// `return T(args)` for member-wise stores into the returned object when T has a constructor
/// taking exactly one scalar per member (CVector3f(x, y, z) and friends).
fn construct_from_stores(stores: &[(i32, Expr)], ret: &Type, db: Option<&TypeDb>) -> Option<Expr> {
    let db = db?;
    let cls = named(&types::resolve(Some(db), ret)).map(|s| s.to_string())?;
    // only zero/false stores: what a default constructor does (`optional_object<T>()`, ...)
    if stores.iter().all(|(_, e)| e.as_int() == Some(0)) {
        let last = sig::split_scope(&cls).1;
        let last = last.split('<').next().unwrap_or(last);
        let key = format!("{}::{}", strip_tmpl(&cls), last);
        if db.decls.get(&key).map_or(false, |ds| ds.iter().any(|d| d.params.is_empty())) {
            let mut fields = vec![];
            let full = flat_fields(db, &cls, 0, &mut fields, 0) && fields.len() == stores.len();
            if !full {
                return Some(Expr::Construct { class: Type::Named(cls), ctor: None, args: vec![] });
            }
        }
    }
    let mut fields = vec![];
    if !flat_fields(db, &cls, 0, &mut fields, 0) || fields.len() != stores.len() || fields.is_empty() {
        return None;
    }
    let mut args = vec![];
    for (off, _) in &fields {
        args.push(stores.iter().find(|(o, _)| o == off)?.1.clone());
    }
    let last = sig::split_scope(&cls).1;
    let last = last.split('<').next().unwrap_or(last);
    let key = format!("{}::{}", strip_tmpl(&cls), last);
    let decls = db.decls.get(&key)?;
    let ok = decls.iter().any(|d| {
        d.params.len() == fields.len()
            && d.params.iter().zip(&fields).all(|(p, (_, ft))| {
                let pt = types::resolve(Some(db), &p.ty).into_owned();
                is_float(&pt) == is_float(ft) && !types::is_aggregate(Some(db), &pt) && !is_ptr(&pt)
            })
    });
    if !ok {
        return None;
    }
    Some(Expr::Construct { class: Type::Named(cls), ctor: None, args })
}

/// `return obj` for member-wise stores that copy every member of another object of the
/// returned type (an inlined trivial copy constructor).
fn copy_from_stores(stores: &[(i32, Expr)], ret: &Type, db: Option<&TypeDb>, temps: &std::collections::HashMap<VarId, Expr>) -> Option<Expr> {
    let db = db?;
    let cls = named(&types::resolve(Some(db), ret)).map(|s| s.to_string())?;
    // whole members copied (`r.mList = o.mList`) cover their leaves; single-definition temps
    // read from the source object count as reads of it
    let mut leaves: Vec<(i32, Expr)> = vec![];
    for (off, v) in stores {
        let v = match v {
            Expr::Var(t) => temps.get(t).cloned().unwrap_or_else(|| v.clone()),
            v => v.clone(),
        };
        let (base, boff, ptr, ty) = match &v {
            Expr::Load { base, offset, ty } => ((**base).clone(), *offset, true, ty.clone()),
            Expr::Member { base, offset, ty } => ((**base).clone(), *offset, false, ty.clone()),
            _ => {
                leaves.push((*off, v.clone()));
                continue;
            }
        };
        let rt = types::resolve(Some(db), &ty).into_owned();
        let mut sub = vec![];
        match named(&rt) {
            Some(c) if types::is_aggregate(Some(db), &rt) && flat_fields(db, c, 0, &mut sub, 0) => {
                for (lo, lt) in sub {
                    let e = if ptr {
                        Expr::Load { base: Box::new(base.clone()), offset: boff + lo, ty: lt }
                    } else {
                        Expr::Member { base: Box::new(base.clone()), offset: boff + lo, ty: lt }
                    };
                    leaves.push((*off + lo, e));
                }
            }
            _ => leaves.push((*off, v.clone())),
        }
    }
    let stores = &leaves[..];
    // every byte of the object copied once, from one source object at the same relative offset
    let size = types::size_of(Some(db), &Type::Named(cls.clone()))? as usize;
    if size == 0 || size > 0x400 {
        return None;
    }
    let mut covered = vec![false; size];
    let mut src: Option<(&Expr, i32, bool)> = None;
    for (off, v) in stores {
        let (base, boff, ptr, ty) = match v {
            Expr::Load { base, offset, ty } => (&**base, *offset, true, ty),
            Expr::Member { base, offset, ty } => (&**base, *offset, false, ty),
            _ => return None,
        };
        let n = types::size_of(Some(db), ty).filter(|n| *n > 0)? as usize;
        let start = boff - off;
        match src {
            None => src = Some((base, start, ptr)),
            Some((b, s0, p)) if b == base && s0 == start && p == ptr => {}
            _ => return None,
        }
        let o = usize::try_from(*off).ok()?;
        if o + n > size || covered[o..o + n].iter().any(|c| *c) {
            return None;
        }
        covered[o..o + n].iter_mut().for_each(|c| *c = true);
    }
    // padding bytes need no copy: only the members' bytes must be covered
    let mut fields = vec![];
    if flat_fields(db, &cls, 0, &mut fields, 0) {
        for (fo, ft) in &fields {
            let n = types::size_of(Some(db), ft).unwrap_or(0) as usize;
            let o = *fo as usize;
            if o + n > size || covered[o..o + n].iter().any(|c| !*c) {
                return None;
            }
        }
    } else if covered.iter().any(|c| !*c) {
        return None;
    }
    let (base, start, ptr) = src?;
    Some(if ptr {
        Expr::Load { base: Box::new(base.clone()), offset: start, ty: ret.clone() }
    } else {
        Expr::Member { base: Box::new(base.clone()), offset: start, ty: ret.clone() }
    })
}

fn strip_tmpl(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

fn is_sret_deref(e: &Expr, sret: VarId) -> bool {
    matches!(e, Expr::Load { base, offset: 0, .. } if matches!(**base, Expr::Var(v) if v == sret))
}

/// `*__return = e; return;` -> `return e;`, `__return->T(args); return;` -> `return T(args);`,
/// member-wise stores + return -> `return T(fields...)`.
fn fold_struct_return(body: &mut Vec<Stmt>, sret: VarId, ret: &Type, db: Option<&TypeDb>) {
    // falling off the end returns the object too
    if !matches!(body.last(), Some(Stmt::Return(_))) {
        body.push(Stmt::Return(None));
    }
    // `if (a) { R = x; } else { R = y; } return;`: each branch returns its own object
    Stmt::for_each_block_mut(body, &mut |b| distribute_sret_return(b, sret));
    // single-definition temps holding a member read (`t = o->m; ... R.m = t;`)
    let temps: std::collections::HashMap<VarId, Expr> = {
        let mut defs: std::collections::HashMap<VarId, (usize, Expr)> = Default::default();
        let mut snap = body.clone();
        Stmt::for_each_block_mut(&mut snap, &mut |b| {
            for s in b.iter() {
                if let Stmt::Assign { dst: Expr::Var(v), src } = s {
                    let e = defs.entry(*v).or_insert((0, src.clone()));
                    e.0 += 1;
                }
            }
        });
        defs.into_iter().filter(|(_, (n, e))| *n == 1 && matches!(e, Expr::Load { .. } | Expr::Member { .. }) && !e.has_call()).map(|(v, (_, e))| (v, e)).collect()
    };
    Stmt::for_each_block_mut(body, &mut |b| {
        // constructor call on the return slot
        let mut i = 0;
        while i + 1 < b.len() {
            let c = match (&b[i], &b[i + 1]) {
                (Stmt::Expr(Expr::Call { callee: Callee::Method { sig: s, this, .. }, args, .. }), Stmt::Return(None))
                    if sig::is_ctor(s) && matches!(&**this, Expr::Var(v) if *v == sret) =>
                {
                    Some(Expr::Construct { class: Type::Named(s.this_class.clone().unwrap_or_default()), ctor: Some(s.clone()), args: args.clone() })
                }
                _ => None,
            };
            if let Some(e) = c {
                b.splice(i..=i + 1, [Stmt::Return(Some(e))]);
            }
            i += 1;
        }
        // member-wise stores right before `return;`
        if let Some(rp) = b.iter().position(|s| matches!(s, Stmt::Return(None))) {
            let mut k = rp;
            let mut stores = vec![];
            // temps loaded in between (`t = o->a; R.b = o->b; R.a = t;`): part of the copy
            let mut temp_defs: Vec<VarId> = vec![];
            while k > 0 {
                match &b[k - 1] {
                    Stmt::Assign { dst: Expr::Load { base, offset, .. }, src } if matches!(**base, Expr::Var(v) if v == sret) && !src.uses_var(sret) => {
                        stores.push((*offset, src.clone()));
                        k -= 1;
                    }
                    Stmt::Assign { dst: Expr::Var(t), .. } if temps.contains_key(t) && !stores.is_empty() => {
                        temp_defs.push(*t);
                        k -= 1;
                    }
                    _ => break,
                }
            }
            // (a window must start with a store; leading temp defs belong to other code)
            while k < rp && matches!(&b[k], Stmt::Assign { dst: Expr::Var(t), .. } if temp_defs.contains(t)) {
                if let Stmt::Assign { dst: Expr::Var(t), .. } = &b[k] {
                    let t = *t;
                    temp_defs.retain(|x| *x != t);
                }
                k += 1;
            }
            // temps used outside the window can't be folded away
            let outside = temp_defs.iter().any(|t| b.iter().enumerate().any(|(j, s)| (j < k || j > rp) && stmt_mentions(s, *t)));
            // a whole object stored (a call returning it) is no member-wise construction
            let whole = stores.iter().any(|(_, e)| matches!(e, Expr::Call { ret: r, .. } if types::is_aggregate(db, r)));
            if !stores.is_empty() && !whole && !outside {
                let folded = if temp_defs.is_empty() { construct_from_stores(&stores, ret, db) } else { None };
                if let Some(e) = folded.or_else(|| copy_from_stores(&stores, ret, db, &temps)) {
                    b.splice(k..=rp, [Stmt::Return(Some(e))]);
                }
            }
        }
        let mut i = 0;
        while i < b.len() {
            let fold = match &b[i] {
                Stmt::Assign { dst, src } if is_sret_deref(dst, sret) => {
                    let next_ret = matches!(b.get(i + 1), Some(Stmt::Return(None)));
                    if next_ret {
                        Some(src.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(e) = fold {
                b.splice(i..=i + 1, [Stmt::Return(Some(e))]);
            }
            i += 1;
        }
    });
}

/// Does the statement (list) end by defining the whole returned object on every path?
fn ends_with_sret_def(b: &[Stmt], sret: VarId) -> bool {
    match b.last() {
        // (member-wise stores into it count: folded into a construction per branch)
        Some(Stmt::Assign { dst: Expr::Load { base, .. }, src }) => matches!(**base, Expr::Var(v) if v == sret) && !src.uses_var(sret),
        Some(Stmt::Expr(Expr::Call { callee: Callee::Method { sig: s, this, .. }, .. })) => sig::is_ctor(s) && matches!(&**this, Expr::Var(v) if *v == sret),
        Some(Stmt::If { then, els, .. }) => ends_with_sret_def(then, sret) && ends_with_sret_def(els, sret),
        _ => false,
    }
}

fn push_return(b: &mut Vec<Stmt>) {
    match b.last_mut() {
        Some(Stmt::If { then, els, .. }) => {
            push_return(then);
            push_return(els);
        }
        _ => b.push(Stmt::Return(None)),
    }
}

fn distribute_sret_return(b: &mut Vec<Stmt>, sret: VarId) {
    let n = b.len();
    if n >= 2 && matches!(b[n - 1], Stmt::Return(None)) && matches!(b[n - 2], Stmt::If { .. }) && ends_with_sret_def(&b[n - 2..n - 1], sret) {
        b.pop();
        push_return(b);
        return;
    }
    // `switch (x) { case a: R = ..; break; ... default: R = ..; break; } return;`
    if n >= 2 && matches!(b[n - 1], Stmt::Return(None)) {
        if let Stmt::Switch { cases, .. } = &b[n - 2] {
            let all = cases.iter().any(|c| c.is_default)
                && cases.iter().all(|c| {
                    let body = &c.body;
                    match body.last() {
                        Some(Stmt::Break) => ends_with_sret_def(&body[..body.len() - 1], sret),
                        _ => false,
                    }
                });
            if all {
                if let Stmt::Switch { cases, .. } = &mut b[n - 2] {
                    for c in cases.iter_mut() {
                        if let Some(last) = c.body.last_mut() {
                            *last = Stmt::Return(None);
                        }
                    }
                }
                b.pop();
            }
        }
    }
}

fn is_vtable_addr(e: &Expr) -> bool {
    match e {
        Expr::AddrOf(g) => matches!(&**g, Expr::Global { symbol, .. } if symbol.starts_with("__vt__")),
        Expr::Cast { e, .. } => is_vtable_addr(e),
        _ => false,
    }
}

fn drop_vptr_stores(body: &mut Vec<Stmt>) {
    Stmt::for_each_block_mut(body, &mut |b| b.retain(|s| !matches!(s, Stmt::Assign { src, .. } if is_vtable_addr(src))));
    // `if (this) {}` left behind by inlined base destructors/constructors
    Stmt::for_each_block_mut(body, &mut |b| {
        b.retain(|s| !matches!(s, Stmt::If { cond, then, els } if then.is_empty() && els.is_empty() && !cond.has_call()))
    });
}

fn is_this(e: &Expr, this: VarId) -> bool {
    match e {
        Expr::Var(v) => *v == this,
        Expr::Cast { e, .. } => is_this(e, this),
        _ => false,
    }
}

/// `t = operator new(size, args...); [v = t;] if (t) { t->Ctor(cargs); [v = t;] }` ->
/// `v = new (args...) Class(cargs);`
fn fold_new(body: &mut Vec<Stmt>, _vars: &[Var]) {
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let Some((t, placement)) = (match &b[i] {
                Stmt::Assign { dst: Expr::Var(t), src: Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. } }
                    if symbol.starts_with("__nw__") =>
                {
                    Some((*t, args.iter().skip(1).cloned().collect::<Vec<_>>()))
                }
                _ => None,
            }) else {
                i += 1;
                continue;
            };
            let mut j = i + 1;
            let mut v = t;
            if let Some(Stmt::Assign { dst: Expr::Var(x), src: Expr::Var(y) }) = b.get(j) {
                if *y == t {
                    v = *x;
                    j += 1;
                }
            }
            let ok = match b.get(j) {
                Some(Stmt::If { cond, then, els }) if els.is_empty() => {
                    let cv = match cond {
                        Expr::Var(c) => Some(*c),
                        Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => match &**l {
                            Expr::Var(c) => Some(*c),
                            _ => None,
                        },
                        _ => None,
                    };
                    let ctor = match then.first() {
                        Some(Stmt::Expr(Expr::Call { callee: Callee::Method { sig: s, this, .. }, args, .. }))
                            if sig::is_ctor(s) && matches!(&**this, Expr::Var(x) if *x == t || *x == v) =>
                        {
                            Some((s.clone(), args.clone()))
                        }
                        _ => None,
                    };
                    let rest_ok = then.iter().skip(1).all(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src: Expr::Var(y) } if (*x == v || *x == t) && (*y == t || *y == v)));
                    match (cv, ctor) {
                        (Some(c), Some(ct)) if (c == t || c == v) && rest_ok => Some(ct),
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some((s, args)) = ok {
                let class = Type::Named(s.this_class.clone().unwrap_or_default());
                let new = Expr::New { class, placement, ctor: Some(s), args };
                b.splice(i..=j, [Stmt::Assign { dst: Expr::Var(v), src: new }]);
            }
            i += 1;
        }
    });
}

/// Initializer-list entry for a constructor call on a base subobject or direct member of `this`.
fn ctor_init_of(st: &Stmt, this: VarId, own: &str, db: Option<&TypeDb>) -> Option<Init> {
    let Stmt::Expr(Expr::Call { callee: Callee::Method { sig: s, this: obj, .. }, args, .. }) = st else { return None };
    if !sig::is_ctor(s) {
        return None;
    }
    // the member (or a base subobject of it, built by its inline constructor) a path names
    let member_of = |path: &[types::PathElem]| -> Option<String> {
        match path {
            [types::PathElem::Field(n, owner), rest @ ..] if sig::norm_name(owner) == sig::norm_name(own) && rest.iter().all(|p| matches!(p, types::PathElem::Base(_))) => Some(n.clone()),
            _ => None,
        }
    };
    let cls = s.this_class.clone().unwrap_or_default();
    let direct_base = db.and_then(|db| sig::find_class(db, own)).map_or(true, |c| c.bases.iter().any(|b| sig::norm_name(&b.name) == sig::norm_name(&cls)));
    let init = |target| Some(Init { target, ctor: Some(s.clone()), args: args.clone(), member_ty: None });
    if is_this(obj, this) && sig::norm_name(&cls) != sig::norm_name(own) && direct_base {
        init(InitTarget::Base(cls))
    } else if is_this(obj, this) && sig::norm_name(&cls) != sig::norm_name(own) {
        // the member at offset 0
        db.and_then(|db| types::field_path_of_type(db, own, 0, &Type::Named(cls.clone()))).and_then(|path| member_of(&path)).and_then(|n| init(InitTarget::Member(n)))
    } else if let Expr::AddrOf(inner) = &**obj {
        member_of_this(inner, this).and_then(|off| {
            let db = db?;
            // a base subobject at a non-zero offset?
            if let Some(c) = sig::find_class(db, own) {
                if c.bases.iter().any(|b| b.offset as i32 == off && sig::norm_name(&b.name) == sig::norm_name(&cls)) {
                    return init(InitTarget::Base(cls.clone()));
                }
            }
            let path = types::field_path_of_type(db, own, off, &Type::Named(cls.clone())).or_else(|| types::field_path(db, own, off, 0).map(|x| x.0))?;
            member_of(&path).and_then(|n| init(InitTarget::Member(n)))
        })
    } else {
        None
    }
}

fn ctor_init_list(ir: &mut IrFunction, this: VarId, db: Option<&TypeDb>) {
    let own = ir.sig.this_class.clone().unwrap_or_default();
    let mut inits: Vec<Init> = vec![];
    let k = 0;
    // offset of the last member built by a constructor call: a plain store to a member
    // declared before it ran in the body (members are built in declaration order)
    let field_off = |n: &str| db.and_then(|db| sig::find_class(db, &own)).and_then(|c| c.fields.iter().find(|f| f.name == n).map(|f| f.offset as i32));
    let mut last_call: i32 = -1;
    while k < ir.body.len() {
        let is_call = matches!(&ir.body[k], Stmt::Expr(_));
        // `*this = other` at the start of a constructor (member-wise copy of an object of the
        // same class): each member initialized from the other's (`operator=` may not exist or do
        // something else)
        if inits.is_empty() {
            if let Some(es) = whole_self_copy(&ir.body[k], this, &own, db, &ir.vars) {
                ir.body.remove(k);
                inits.extend(es);
                continue;
            }
        }
        let take = match &ir.body[k] {
            st @ Stmt::Expr(_) => ctor_init_of(st, this, &own, db),
            // leading bitfield stores: initializer-list entries (written through its whole word,
            // a field within one byte is one: a body assignment accesses just that byte)
            Stmt::Assign { dst: Expr::BitField { base, shift, width, .. }, src } if !src.uses_var(this) && !src.has_call() => db.and_then(|db| {
                let Expr::Load { base: b, offset, ty } = &**base else { return None };
                let unit = types::size_of(Some(db), ty)?;
                if !is_this(b, this) || !matches!(unit, 1 | 2 | 4) {
                    return None;
                }
                let mask = (((1u64 << *width) - 1) << *shift) as u32;
                let (path, ft) = types::bitfield_at(db, &own, *offset, unit, mask)?;
                match path.as_slice() {
                    [types::PathElem::Field(n, owner)] if sig::norm_name(owner) == sig::norm_name(&own) && !inits.iter().any(|i: &Init| i.target == InitTarget::Member(n.clone())) => {
                        Some(Init { target: InitTarget::Member(n.clone()), ctor: None, args: vec![src.clone()], member_ty: Some(ft) })
                    }
                    _ => None,
                }
            }),
            // leading plain member stores: initializer-list entries (member order is the store
            // order MWCC emits for an init list)
            Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } if is_this(base, this) && !src.uses_var(this) && !src.has_call() => db.and_then(|db| {
                let size = types::size_of(Some(db), ty)?;
                let (path, ft) = types::field_path(db, &own, *offset, size)?;
                match path.as_slice() {
                    [types::PathElem::Field(n, owner)] if sig::norm_name(owner) == sig::norm_name(&own) && types::size_of(Some(db), &ft) == Some(size) => {
                        if inits.iter().any(|i: &Init| i.target == InitTarget::Member(n.clone())) {
                            return None;
                        }
                        Some(Init { target: InitTarget::Member(n.clone()), ctor: None, args: vec![src.clone()], member_ty: Some(ft.clone()) })
                    }
                    _ => None,
                }
            }),
            _ => None,
        };
        let off = match take.as_ref().map(|i| &i.target) {
            Some(InitTarget::Member(n)) => field_off(n),
            _ => None,
        };
        let take = match (take, off) {
            (Some(_), Some(o)) if !is_call && o < last_call => None,
            (t, o) => {
                if let (true, Some(o), true) = (is_call, o, t.is_some()) {
                    last_call = last_call.max(o);
                }
                t
            }
        };
        match take {
            Some(init) => {
                ir.body.remove(k);
                // default constructors are implicit
                if !init.args.is_empty() {
                    inits.push(init);
                }
            }
            None => break,
        }
    }
    late_ctor_inits(ir, this, &own, db, &mut inits);
    memberwise_inits(ir, this, &own, db, &mut inits);
    ir.init_list = inits;
}

/// Members (and bases) without default constructors whose inline constructor the compiler
/// expanded into stores in the body can only be built in the initializer list: `m(args)` from
/// the stores of the members that constructor sets from its parameters (its other stores are
/// its own constants and go with it).
fn memberwise_inits(ir: &mut IrFunction, this: VarId, own: &str, db: Option<&TypeDb>, inits: &mut Vec<Init>) {
    let Some(db) = db else { return };
    let Some(c) = sig::find_class(db, own).cloned() else { return };
    let vars = ir.vars.clone();
    let mut targets: Vec<(InitTarget, Type, i32)> = vec![];
    for b in &c.bases {
        targets.push((InitTarget::Base(b.name.clone()), Type::Named(b.name.clone()), b.offset as i32));
    }
    for f in &c.fields {
        if f.bitfield.is_none() {
            targets.push((InitTarget::Member(f.name.clone()), f.ty.clone(), f.offset as i32));
        }
    }
    fn nested_write(s: &Stmt, hit: &dyn Fn(&Expr) -> bool) -> bool {
        let mut found = false;
        let mut visit = |b: &[Stmt]| {
            for x in b {
                if let Stmt::Assign { dst, .. } = x {
                    if hit(dst) {
                        found = true;
                    }
                }
                if nested_write(x, hit) {
                    found = true;
                }
            }
        };
        match s {
            Stmt::If { then, els, .. } => {
                visit(then);
                visit(els);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => visit(body),
            Stmt::Switch { cases, .. } => {
                for cs in cases {
                    visit(&cs.body);
                }
            }
            _ => {}
        }
        found
    }
    for (target, ty, off) in targets {
        let ft = types::resolve(Some(db), strip_cv(&ty)).into_owned();
        let Some(fc) = named(&ft).map(|s| s.to_string()) else { continue };
        if !types::is_aggregate(Some(db), &ft) || default_constructible(Some(db), &ft) || inits.iter().any(|i| i.target == target) {
            continue;
        }
        let Some(size) = types::size_of(Some(db), &ft) else { continue };
        let (lo, hi) = (off, off + size as i32);
        let hit = |e: &Expr| matches!(e, Expr::Load { base, offset, .. } if is_this(base, this) && *offset >= lo && *offset < hi);
        if ir.body.iter().any(|s| nested_write(s, &hit)) {
            continue;
        }
        let stores: Vec<(usize, i32, Expr)> = ir
            .body
            .iter()
            .enumerate()
            .filter_map(|(k, s)| match s {
                Stmt::Assign { dst: Expr::Load { base, offset, .. }, src } if is_this(base, this) && *offset >= lo && *offset < hi => Some((k, offset - lo, src.clone())),
                _ => None,
            })
            .collect();
        if stores.is_empty() {
            continue;
        }
        for (cs, offs) in crate::construct::member_ctors(db, &fc) {
            let mut args = vec![];
            for (i, o) in offs.iter().enumerate() {
                let Some((k, _, v)) = stores.iter().find(|(_, so, _)| so == o) else { break };
                let pty = cs.params.get(i).map(|p| match strip_cv(&p.ty) {
                    Type::Ref(x) => strip_cv(x).clone(),
                    t => t.clone(),
                });
                match spellable_in_init(v, false, pty.as_ref(), &ir.body[..*k], &vars, this, Some(db), false) {
                    Some(e) => args.push(e),
                    None => break,
                }
            }
            if args.len() != offs.len() {
                continue;
            }
            for (k, _, _) in stores.iter().rev() {
                ir.body.remove(*k);
            }
            inits.push(Init { target: target.clone(), ctor: Some(cs), args, member_ty: None });
            break;
        }
    }
}

/// Base/member constructor calls left in the body after other statements (their arguments are
/// temporaries the body sets up first) can't stay there: a constructor can't be called on a
/// subobject. Move them into the initializer list when every argument can be spelled there:
/// parameters, globals and constants, temporaries assigned once from such expressions (inlined),
/// and default-constructed temporaries (spelled `T()` by the emitter).
fn late_ctor_inits(ir: &mut IrFunction, this: VarId, own: &str, db: Option<&TypeDb>, inits: &mut Vec<Init>) {
    let vars = ir.vars.clone();
    let mut k = 0;
    while k < ir.body.len() {
        let Some(mut init) = ctor_init_of(&ir.body[k], this, own, db).or_else(|| no_default_member_store(&ir.body[k], this, own, db)) else {
            k += 1;
            continue;
        };
        let mut ok = true;
        let orig_args = init.args.clone();
        let refs: Vec<bool> = (0..init.args.len())
            .map(|i| init.ctor.as_ref().and_then(|s| s.params.get(i)).map_or(init.ctor.is_none(), |p| matches!(p.ty.unqualified(), Type::Ref(_))))
            .collect();
        let members_ok = matches!(init.target, InitTarget::Member(_));
        let ptys: Vec<Option<Type>> = (0..init.args.len())
            .map(|i| init.ctor.as_ref().and_then(|s| s.params.get(i)).map(|p| match strip_cv(&p.ty) {
                Type::Ref(x) => strip_cv(x).clone(),
                t => t.clone(),
            }))
            .collect();
        for ((a, by_ref), pty) in init.args.iter_mut().zip(refs).zip(ptys) {
            match spellable_in_init(a, by_ref, pty.as_ref(), &ir.body[..k], &vars, this, db, members_ok) {
                Some(e) => *a = e,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            k += 1;
            continue;
        }
        ir.body.remove(k);
        k -= drop_consumed_temps(&mut ir.body, k, &orig_args, &vars);
        // where the construction ran, for members set before it whose values are still
        // expanded inlines (folded later, see `INIT_MARK`)
        if let (true, InitTarget::Member(n)) = (k > 0, &init.target) {
            ir.body.insert(k, Stmt::Comment(format!("{INIT_MARK}{n}")));
        }
        // members declared before this one that the body set before its construction: their
        // initializers ran first (`w(in.ReadFloat()), v(in)`)
        let off = match (&init.target, db.and_then(|db| sig::find_class(db, own))) {
            (InitTarget::Member(n), Some(c)) => c.fields.iter().find(|f| &f.name == n).map(|f| f.offset as i32),
            _ => None,
        };
        if let Some(o) = off {
            let mut j = 0;
            while j < k {
                if let Some(e) = earlier_member_init(&ir.body[j], this, own, db, o, inits) {
                    // (only parameters: a register local may hold a value read before a later
                    // store changed its source)
                    let mut params_only = true;
                    e.args[0].walk(&mut |x| params_only &= !matches!(x, Expr::Var(v) if !matches!(vars[*v].kind, VarKind::Param { .. })));
                    if let Some(a) = spellable_in_init(&e.args[0], false, None, &ir.body[..j], &vars, this, db, true).filter(|_| params_only) {
                        ir.body.remove(j);
                        k -= 1;
                        inits.push(Init { args: vec![a], ..e });
                        continue;
                    }
                }
                j += 1;
            }
        }
        if !init.args.is_empty() {
            inits.push(init);
        }
    }
}

/// Initializer-list entries for `*this = other` with `other` an object of the constructor's own
/// class (no bases, no vtable): one per member.
fn whole_self_copy(st: &Stmt, this: VarId, own: &str, db: Option<&TypeDb>, vars: &[Var]) -> Option<Vec<Init>> {
    let Stmt::Assign { dst: Expr::Load { base, offset: 0, ty }, src } = st else { return None };
    if !is_this(base, this) {
        return None;
    }
    let db = db?;
    let c = sig::find_class(db, own)?;
    let same = |t: &Type| types::class_of(Some(db), t).is_some_and(|k| sig::norm_name(&k.name) == sig::norm_name(own));
    if !same(ty) || !c.bases.is_empty() || c.vptr_offset.is_some() || c.is_union || c.fields.is_empty() || c.fields.iter().any(|f| f.bitfield.is_some()) {
        return None;
    }
    // the other object's member at `off`
    let member: Box<dyn Fn(i32, &Type) -> Expr> = match src {
        Expr::Load { base, offset, ty } if same(ty) && !base.uses_var(this) => {
            let (b, o) = ((**base).clone(), *offset);
            Box::new(move |off, t| Expr::Load { base: Box::new(b.clone()), offset: o + off, ty: t.clone() })
        }
        // a reference parameter (the object itself)
        Expr::Var(v) if matches!(vars[*v].kind, VarKind::Param { .. }) && same(strip_cv(&vars[*v].ty)) => {
            let v = *v;
            Box::new(move |off, t| Expr::Member { base: Box::new(Expr::Var(v)), offset: off, ty: t.clone() })
        }
        _ => return None,
    };
    Some(c.fields.iter().map(|f| Init { target: InitTarget::Member(f.name.clone()), ctor: None, args: vec![member(f.offset as i32, &f.ty)], member_ty: Some(f.ty.clone()) }).collect())
}

/// Comment marking where a member's construction ran in a constructor body (moved to the
/// initializer list); consumed after inline folding, never emitted.
pub const INIT_MARK: &str = "mwdec init order: ";

/// `this->m = v` for a direct member declared before offset `before` and not initialized yet.
fn earlier_member_init(st: &Stmt, this: VarId, own: &str, db: Option<&TypeDb>, before: i32, inits: &[Init]) -> Option<Init> {
    let Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } = st else { return None };
    if !is_this(base, this) || *offset >= before || src.uses_var(this) {
        return None;
    }
    let db = db?;
    let size = types::size_of(Some(db), ty)?;
    let (path, ft) = types::field_path(db, own, *offset, size)?;
    match path.as_slice() {
        [types::PathElem::Field(n, owner)] if sig::norm_name(owner) == sig::norm_name(own) && types::size_of(Some(db), &ft) == Some(size) && types::class_of(Some(db), &ft).is_none() => {
            if inits.iter().any(|i| i.target == InitTarget::Member(n.clone())) {
                return None;
            }
            Some(Init { target: InitTarget::Member(n.clone()), ctor: None, args: vec![src.clone()], member_ty: Some(ft.clone()) })
        }
        _ => None,
    }
}

/// The temporaries an initializer-list entry now spells inline (`T(args)`, `f(x)`): their
/// definitions, in-place constructions and destructions before `end` go when nothing else
/// mentions them. Returns how many statements before `end` were removed.
fn drop_consumed_temps(body: &mut Vec<Stmt>, end: usize, args: &[Expr], vars: &[Var]) -> usize {
    let mut cand: Vec<VarId> = vec![];
    for a in args {
        a.walk(&mut |x| {
            if let Expr::Var(v) = x {
                if matches!(vars[*v].kind, VarKind::Stack { .. } | VarKind::Local) && !cand.contains(v) {
                    cand.push(*v);
                }
            }
        });
    }
    let on_var = |s: &Stmt, v: VarId| -> bool {
        match s {
            Stmt::Assign { dst: Expr::Var(x), src } => *x == v && !src.uses_var(v),
            Stmt::Assign { dst: Expr::Member { base, .. }, src } => matches!(&**base, Expr::Var(x) if *x == v) && !src.uses_var(v),
            Stmt::Expr(Expr::Call { callee: Callee::Method { this, .. }, args, .. }) => {
                matches!(&**this, Expr::AddrOf(x) if matches!(&**x, Expr::Var(y) if *y == v)) && !args.iter().any(|a| a.uses_var(v))
            }
            _ => false,
        }
    };
    let mut removed = 0;
    let mut i = 0;
    while i < cand.len() {
        let v = cand[i];
        i += 1;
        let own: Vec<usize> = (0..body.len()).filter(|&j| on_var(&body[j], v)).collect();
        let others = (0..body.len()).any(|j| !own.contains(&j) && stmt_mentions(&body[j], v));
        if others || own.is_empty() || own.iter().any(|&j| j >= end - removed && !matches!(body[j], Stmt::Expr(_))) {
            continue;
        }
        for &j in own.iter().rev() {
            // what the removed statement read may now be dead too
            Stmt::walk_exprs(std::slice::from_ref(&body[j]), &mut |x| {
                if let Expr::Var(w) = x {
                    if *w != v && matches!(vars[*w].kind, VarKind::Stack { .. } | VarKind::Local) && !cand.contains(w) {
                        cand.push(*w);
                    }
                }
            });
            body.remove(j);
            if j < end - removed {
                removed += 1;
            }
        }
    }
    removed
}

/// `this->m = v` for a member whose class has no default constructor: it can only be
/// initialized in the initializer list (`m(v)`).
fn no_default_member_store(st: &Stmt, this: VarId, own: &str, db: Option<&TypeDb>) -> Option<Init> {
    let Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } = st else { return None };
    if !is_this(base, this) {
        return None;
    }
    let db = db?;
    let size = types::size_of(Some(db), ty)?;
    let c = sig::find_class(db, own)?;
    // the direct member at that offset, stored as a whole (`*(u16*)&mId = kInvalid.value`
    // copies a one-member object)
    let f = c.fields.iter().find(|f| f.offset as i32 == *offset && f.bitfield.is_none())?;
    let ft = f.ty.clone();
    // reference and const members can only be initialized in the list either
    if matches!(ft, Type::Ref(_) | Type::Const(_)) && !src.uses_var(this) {
        let fs = match &ft {
            Type::Ref(_) => Some(4),
            t => types::size_of(Some(db), t),
        };
        if fs == Some(size) || matches!(ft, Type::Ref(_)) {
            return Some(Init { target: InitTarget::Member(f.name.clone()), ctor: None, args: vec![src.clone()], member_ty: Some(ft) });
        }
    }
    if !types::is_aggregate(Some(db), &ft) || default_constructible(Some(db), &ft) || types::size_of(Some(db), &ft) != Some(size) {
        return None;
    }
    let whole = match src {
        Expr::Member { base, offset: 0, .. } | Expr::Load { base, offset: 0, .. } => {
            let bt = match src {
                Expr::Member { .. } => types::ty_of(base, &[]),
                _ => pointee(&types::ty_of(base, &[])).cloned().unwrap_or(Type::Void),
            };
            if types::is_aggregate(Some(db), &bt) {
                match src {
                    Expr::Member { .. } => (**base).clone(),
                    _ => Expr::Load { base: base.clone(), offset: 0, ty: ft.clone() },
                }
            } else {
                src.clone()
            }
        }
        _ => src.clone(),
    };
    Some(Init { target: InitTarget::Member(f.name.clone()), ctor: None, args: vec![whole], member_ty: Some(ft) })
}

/// The only store into stack object `v` is one whole-object-sized store at offset 0 from a
/// member at offset 0 of another object (`*(u16*)&tmp = kInvalidUniqueId.value`): that object.
fn single_store(before: &[Stmt], vs: &[VarId], vars: &[Var], db: Option<&TypeDb>) -> Option<Expr> {
    let mut found = None;
    for s in before {
        if let Stmt::Assign { dst, src } = s {
            let into_v = match dst {
                Expr::Member { base, .. } => matches!(&**base, Expr::Var(x) if vs.contains(x)),
                Expr::Var(x) => vs.contains(x),
                _ => false,
            };
            if into_v {
                if found.is_some() {
                    return None;
                }
                let Expr::Member { offset: 0, .. } = dst else { return None };
                found = Some(src.clone());
            }
        } else if vs.iter().any(|&v| stmt_mentions(s, v)) {
            return None;
        }
    }
    // (through a register local holding the value)
    let src = match found? {
        x @ Expr::Var(_) => inline_locals(&x, before, vars, db, 0)?,
        x => x,
    };
    match src {
        Expr::Member { base, offset: 0, .. } if matches!(&*base, Expr::Global { .. }) => Some(*base),
        // a copy of an object parameter (`TId owner` passed on)
        Expr::Member { base, offset: 0, .. } if matches!(&*base, Expr::Var(p) if matches!(vars[*p].kind, VarKind::Param { .. }) && named(&types::resolve(db, strip_cv(&vars[*p].ty))).is_some()) => Some(*base),
        _ => None,
    }
}

/// Stack variables at the same frame slot as `v` (the lifter types one slot several ways).
fn slot_aliases(vars: &[Var], v: VarId) -> Vec<VarId> {
    match vars[v].kind {
        VarKind::Stack { offset, .. } => vars.iter().enumerate().filter(|(_, x)| matches!(x.kind, VarKind::Stack { offset: o, .. } if o == offset)).map(|(i, _)| i).collect(),
        _ => vec![v],
    }
}

/// Can `T()` be written (a default constructor, or no user-declared constructors)?
fn default_constructible(db: Option<&TypeDb>, t: &Type) -> bool {
    let Some(db) = db else { return true };
    let r = types::resolve(Some(db), t).into_owned();
    let Some(cls) = named(&r) else { return true };
    let base = strip_tmpl(cls);
    let last = sig::split_scope(&base).1.to_string();
    match db.decls.get(&format!("{base}::{last}")) {
        Some(ds) => ds.iter().any(|d| d.params.is_empty()),
        None => true,
    }
}

/// `e` rewritten for an initializer list (see `late_ctor_inits`), or None.
fn spellable_in_init(e: &Expr, by_ref: bool, pty: Option<&Type>, before: &[Stmt], vars: &[Var], this: VarId, db: Option<&TypeDb>, members_ok: bool) -> Option<Expr> {
    let stack_var = |e: &Expr| -> Option<VarId> {
        match e {
            Expr::Var(v) if matches!(vars[*v].kind, VarKind::Stack { .. }) => Some(*v),
            _ => None,
        }
    };
    let def_of = |v: VarId| -> Vec<&Expr> {
        let mut defs = vec![];
        for s in before {
            if let Stmt::Assign { dst: Expr::Var(x), src } = s {
                if *x == v {
                    defs.push(src);
                }
            }
        }
        defs
    };
    let pure = |e: &Expr| -> bool {
        let mut ok = true;
        e.walk(&mut |x| {
            if let Expr::Var(v) = x {
                // (a member's initializer may read members built before it)
                if !matches!(vars[*v].kind, VarKind::Param { .. } | VarKind::This) || (*v == this && !members_ok) {
                    ok = false;
                }
            }
        });
        ok
    };
    // a temporary object (by value or by address; `*(const T*)&tmp` reads it as its declared
    // class through a typedef'd spelling)
    let inner = match e {
        Expr::AddrOf(x) => stack_var(x),
        Expr::Load { base, offset: 0, .. } if matches!(&**base, Expr::AddrOf(x) if stack_var(x).is_some()) => match &**base {
            Expr::AddrOf(x) => stack_var(x),
            _ => None,
        },
        x => stack_var(x),
    };
    if let Some(v) = inner {
        let defs = def_of(v);
        // constructed in place by a constructor call: the temporary `T(args)`
        let ctor = before.iter().find_map(|s| match s {
            Stmt::Expr(Expr::Call { callee: Callee::Method { sig: cs, this: obj, .. }, args, .. })
                if sig::is_ctor(cs) && matches!(&**obj, Expr::AddrOf(x) if matches!(&**x, Expr::Var(y) if *y == v)) =>
            {
                Some((cs.clone(), args.clone()))
            }
            _ => None,
        });
        return match (defs.as_slice(), ctor) {
            // (a temporary binds to a reference parameter)
            ([src], None) if pure(src) && (by_ref || !matches!(e, Expr::AddrOf(_))) => Some((*src).clone()),
            ([src], None) if by_ref || !matches!(e, Expr::AddrOf(_)) => inline_locals(src, before, vars, db, 0).filter(|x| pure(x)),
            ([], Some((cs, args))) if by_ref || !matches!(e, Expr::AddrOf(_)) => {
                // (its arguments may be temporaries themselves: copies of globals, nested objects)
                let args: Option<Vec<Expr>> = args.iter().map(|a| if pure(a) { Some(a.clone()) } else { inline_locals(a, before, vars, db, 0).filter(|x| pure(x)) }).collect();
                Some(Expr::Construct { class: Type::Named(cs.this_class.clone().unwrap_or_default()), ctor: Some(cs), args: args? })
            }
            // a single-member object set by one store: a copy of the object the value came
            // from, or the member's value
            ([], None) if single_store(before, &slot_aliases(vars, v), vars, db).is_some() => single_store(before, &slot_aliases(vars, v), vars, db),
            // filled member by member (an expanded inline constructor): that constructor
            ([], None)
                if (by_ref || !matches!(e, Expr::AddrOf(_)))
                    && object_from_member_stores(e, v, pty, before, vars, db).is_some_and(|x| matches!(&x, Expr::Construct { args, .. } if args.iter().all(pure))) =>
            {
                object_from_member_stores(e, v, pty, before, vars, db)
            }
            // never assigned as a whole: a default-constructed temporary (member-wise setup
            // stays in the body); an untyped one is spelled as the parameter's class
            ([], None) if default_constructible(db, &vars[v].ty) && (named(&vars[v].ty).is_some() || pty.map_or(true, |t| default_constructible(db, t))) => Some(e.clone()),
            _ => None,
        };
    }
    if pure(e) {
        return Some(e.clone());
    }
    // register locals computed by the body before the call: their (single) definitions, when
    // those only use parameters, globals and constants
    let inlined = inline_locals(e, before, vars, db, 0)?;
    if pure(&inlined) {
        Some(inlined)
    } else {
        None
    }
}

/// The class object a stack temporary holds when `before` fills it member by member (top-level
/// stores only): built with a constructor that sets those members from its parameters, nested
/// objects likewise.
fn object_from_member_stores(_e: &Expr, v: VarId, pty: Option<&Type>, before: &[Stmt], vars: &[Var], db: Option<&TypeDb>) -> Option<Expr> {
    let db = db?;
    let aliases = slot_aliases(vars, v);
    let cls_t = match named(&vars[v].ty) {
        Some(_) => vars[v].ty.clone(),
        None => pty?.clone(),
    };
    let cls = named(&types::resolve(Some(db), strip_cv(&cls_t)).into_owned())?.to_string();
    let mut stores: Vec<(i32, Expr)> = vec![];
    for s in before {
        match s {
            Stmt::Assign { dst: Expr::Member { base, offset, .. }, src } if matches!(&**base, Expr::Var(x) if aliases.contains(x)) => {
                let src = inline_locals(src, before, vars, Some(db), 0)?;
                stores.push((*offset, src));
            }
            s if aliases.iter().any(|&x| stmt_mentions(s, x)) && !matches!(s, Stmt::Expr(_)) => return None,
            _ => {}
        }
    }
    if stores.is_empty() {
        return None;
    }
    build_object(db, &cls, 0, &stores, 0)
}

fn build_object(db: &TypeDb, cls: &str, base: i32, stores: &[(i32, Expr)], depth: u32) -> Option<Expr> {
    if depth > 4 {
        return None;
    }
    'ctor: for (cs, offs) in crate::construct::member_ctors(db, cls) {
        let mut args = vec![];
        for (p, o) in cs.params.iter().zip(&offs) {
            let pt = match strip_cv(&p.ty) {
                Type::Ref(x) => strip_cv(x).clone(),
                t => t.clone(),
            };
            let pr = types::resolve(Some(db), &pt).into_owned();
            if types::is_aggregate(Some(db), &pr) {
                let Some(pc) = named(&pr) else { continue 'ctor };
                match build_object(db, pc, base + o, stores, depth + 1) {
                    Some(x) => args.push(x),
                    None => continue 'ctor,
                }
            } else {
                match stores.iter().find(|(so, _)| *so == base + o) {
                    Some((_, x)) => args.push(x.clone()),
                    None => continue 'ctor,
                }
            }
        }
        return Some(Expr::Construct { class: Type::Named(cls.to_string()), ctor: Some(cs), args });
    }
    None
}

/// `e` with every register local replaced by its single top-level definition in `before`.
fn inline_locals(e: &Expr, before: &[Stmt], vars: &[Var], db: Option<&TypeDb>, depth: u32) -> Option<Expr> {
    if depth > 4 {
        return None;
    }
    let mut out = e.clone();
    let mut ok = true;
    out.rewrite(&mut |x| {
        if let Expr::Var(v) = x {
            if matches!(vars[*v].kind, VarKind::Local) {
                let defs: Vec<&Expr> = before
                    .iter()
                    .filter_map(|s| match s {
                        Stmt::Assign { dst: Expr::Var(d), src } if d == v => Some(src),
                        _ => None,
                    })
                    .collect();
                // also no writes nested in control flow
                let nested = before.iter().any(|s| !matches!(s, Stmt::Assign { .. } | Stmt::Expr(_)) && stmt_mentions(s, *v));
                // a location it read that a later statement overwrites (a stream read before
                // the stream pointer advances): the value is gone
                let stale = defs.len() == 1 && {
                    let d = before.iter().position(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if x == v)).unwrap_or(0);
                    before[d + 1..].iter().any(|s| match s {
                        Stmt::Assign { dst, .. } if !matches!(dst, Expr::Var(_)) => {
                            let mut hit = false;
                            defs[0].walk(&mut |y| hit |= y == dst);
                            hit
                        }
                        _ => false,
                    })
                };
                match (defs.as_slice(), nested || stale) {
                    ([src], false) => match inline_locals(src, before, vars, db, depth + 1) {
                        Some(r) => *x = r,
                        None => ok = false,
                    },
                    _ => ok = false,
                }
            } else if matches!(vars[*v].kind, VarKind::Stack { .. }) {
                // a stack temporary: its single whole-object definition, or a default-constructed
                // object (`CActorParameters().WithAlphaSorting(true)`)
                let defs: Vec<&Expr> = before
                    .iter()
                    .filter_map(|s| match s {
                        Stmt::Assign { dst: Expr::Var(d), src } if d == v => Some(src),
                        _ => None,
                    })
                    .collect();
                let ctor = before.iter().find_map(|s| match s {
                    Stmt::Expr(Expr::Call { callee: Callee::Method { sig: cs, this: o, .. }, args, .. })
                        if sig::is_ctor(cs) && matches!(&**o, Expr::AddrOf(y) if matches!(&**y, Expr::Var(z) if z == v)) =>
                    {
                        Some((cs.clone(), args.clone()))
                    }
                    _ => None,
                });
                let constructed = ctor.is_some();
                match (defs.as_slice(), constructed) {
                    ([src], false) => match inline_locals(src, before, vars, db, depth + 1) {
                        Some(r) => *x = r,
                        None => ok = false,
                    },
                    ([], true) => {
                        let (cs, args) = ctor.unwrap();
                        let a: Option<Vec<Expr>> = args.iter().map(|a| inline_locals(a, before, vars, db, depth + 1)).collect();
                        match a {
                            Some(a) => *x = Expr::Construct { class: Type::Named(cs.this_class.clone().unwrap_or_default()), ctor: Some(cs), args: a },
                            None => ok = false,
                        }
                    }
                    ([], false) if named(&vars[*v].ty).is_some() && default_constructible(db, &vars[*v].ty) => {
                        *x = Expr::Construct { class: vars[*v].ty.clone(), ctor: None, args: vec![] };
                    }
                    _ => ok = false,
                }
            }
        }
    });
    ok.then_some(out)
}

/// `&this->m != 0` (or `(char*)this + K != 0`) as an inlined destructor's null test.
fn is_member_addr_test(c: &Expr, this: VarId) -> bool {
    let strip = |e: &Expr| -> Expr {
        let mut e = e.clone();
        while let Expr::Cast { e: x, .. } = e {
            e = *x;
        }
        e
    };
    let addr = match c {
        Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => strip(l),
        other => strip(other),
    };
    match &addr {
        Expr::AddrOf(inner) => member_of_this(inner, this).is_some(),
        Expr::Binary { op: BinOp::Add, l, r, .. } => is_this(&strip(l), this) && r.as_int().is_some(),
        _ => false,
    }
}

/// Offset of `this->member` lvalues.
fn member_of_this(e: &Expr, this: VarId) -> Option<i32> {
    match e {
        Expr::Load { base, offset, .. } if is_this(base, this) => Some(*offset),
        _ => None,
    }
}

/// `this` / `this != 0`.
fn is_this_test(cond: &Expr, this: VarId) -> bool {
    is_this(cond, this) || matches!(cond, Expr::Binary { op: BinOp::Ne, l, r, .. } if is_this(l, this) && r.as_int() == Some(0))
}

fn dtor_unwrap(ir: &mut IrFunction, this: VarId) {
    let hidden: Vec<VarId> = ir.vars.iter().enumerate().filter(|(_, v)| v.kind == VarKind::Hidden).map(|(i, _)| i).collect();
    // `if (this) { ... }` wrapper
    let mut unwrapped = false;
    if ir.body.len() == 1 {
        if let Stmt::If { cond, then, els } = &ir.body[0] {
            if els.is_empty() && is_this_test(cond, this) {
                ir.body = then.clone();
                unwrapped = true;
            }
        }
    }
    let uses_hidden = |e: &Expr| hidden.iter().any(|h| e.uses_var(*h));
    // another class's destructor run on `this`: a base's
    let own = ir.sig.this_class.as_deref().map(sig::norm_name);
    let mut bases = vec![];
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::Call { callee: Callee::Method { sig: s, this: obj, .. }, .. } = e {
            if let (true, true, Some(c)) = (sig::is_dtor(s), is_this(obj, this), &s.this_class) {
                if own.as_deref() != Some(sig::norm_name(c).as_str()) && !bases.contains(c) {
                    bases.push(c.clone());
                }
            }
        }
    });
    ir.implicit_bases.extend(bases);
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        b.retain(|s| match s {
            // the `delete this` branch of the deleting destructor
            Stmt::If { cond, .. } if uses_hidden(cond) => false,
            // inlined member destructors start with `if (&this->member != 0)`: implicit
            Stmt::If { cond, .. } if is_member_addr_test(cond, this) => false,
            // (the member at offset 0 tests `this` itself, already known non-null here)
            Stmt::If { cond, els, .. } if unwrapped && els.is_empty() && is_this_test(cond, this) => false,
            // (the null test merged with the member destructor's own first test:
            // `if (&this->m && m.mOwn)`)
            Stmt::If { cond: Expr::Binary { op: BinOp::LogAnd, l, .. }, els, .. } if unwrapped && els.is_empty() && (is_this_test(l, this) || is_member_addr_test(l, this)) => false,
            // implicit base/member destructor calls
            Stmt::Expr(Expr::Call { callee: Callee::Method { sig: s, this: obj, .. }, .. }) if sig::is_dtor(s) => {
                !(is_this(obj, this) || matches!(&**obj, Expr::AddrOf(inner) if member_of_this(inner, this).is_some()))
            }
            _ => true,
        })
    });
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        b.retain(|s| !matches!(s, Stmt::If { cond, then, els } if then.is_empty() && els.is_empty() && !cond.has_call()))
    });
}

/// Replace `Var(v)` uses in `e` by `src` where a temporary object is acceptable: by-value use,
/// member read, receiver of a member call, or argument bound to a reference parameter.
/// (`ref_src`: what a reference argument `&v` becomes, `src` itself or an explicit copy of it)
fn forward_into(e: &mut Expr, v: VarId, src: &Expr, ref_src: &Expr, ok: &mut bool) {
    let is_v = |x: &Expr| matches!(x, Expr::Var(y) if *y == v);
    match e {
        Expr::Var(_) if is_v(e) => *e = src.clone(),
        Expr::Member { base, .. } if is_v(base) => **base = src.clone(),
        Expr::AddrOf(inner) if is_v(inner) => *ok = false,
        Expr::Call { callee, args, .. } => {
            let sig = match callee {
                Callee::Method { sig, this, .. } => {
                    if matches!(&**this, Expr::AddrOf(i) if is_v(i)) {
                        **this = Expr::AddrOf(Box::new(ref_src.clone()));
                    } else {
                        forward_into(this, v, src, ref_src, ok);
                    }
                    Some(sig.clone())
                }
                Callee::Direct { sig, .. } => Some(sig.clone()),
                Callee::Virtual { this, sig, .. } => {
                    forward_into(this, v, src, ref_src, ok);
                    sig.clone()
                }
                Callee::Indirect(f) => {
                    forward_into(f, v, src, ref_src, ok);
                    None
                }
            };
            for (n, a) in args.iter_mut().enumerate() {
                let is_ref = sig.as_ref().and_then(|s| s.params.get(n)).map_or(false, |p| matches!(strip_cv(&p.ty), Type::Ref(_)));
                if is_ref && matches!(a, Expr::AddrOf(i) if is_v(i)) {
                    *a = Expr::AddrOf(Box::new(ref_src.clone()));
                } else {
                    forward_into(a, v, src, ref_src, ok);
                }
            }
        }
        Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => forward_into(base, v, src, ref_src, ok),
        Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } | Expr::AddrOf(x) => forward_into(x, v, src, ref_src, ok),
        Expr::Index { base, index, .. } => {
            forward_into(base, v, src, ref_src, ok);
            forward_into(index, v, src, ref_src, ok);
        }
        Expr::Binary { l, r, .. } => {
            forward_into(l, v, src, ref_src, ok);
            forward_into(r, v, src, ref_src, ok);
        }
        Expr::Ternary { c, t, f, .. } => {
            forward_into(c, v, src, ref_src, ok);
            forward_into(t, v, src, ref_src, ok);
            forward_into(f, v, src, ref_src, ok);
        }
        Expr::New { placement, args, .. } => {
            for a in placement.iter_mut().chain(args.iter_mut()) {
                forward_into(a, v, src, ref_src, ok);
            }
        }
        Expr::Construct { args, ctor, .. } => {
            // arguments are evaluated right to left: a call moved into argument n must not
            // overtake calls in the arguments after it
            let pos = args.iter().position(|a| a.uses_var(v));
            if let Some(p) = pos {
                if src.has_call() && args[p + 1..].iter().any(|a| a.has_call()) {
                    *ok = false;
                    return;
                }
            }
            for (n, a) in args.iter_mut().enumerate() {
                let is_ref = ctor.as_ref().and_then(|s| s.params.get(n)).map_or(false, |p| matches!(strip_cv(&p.ty), Type::Ref(_)));
                if is_ref && matches!(a, Expr::AddrOf(i) if is_v(i)) {
                    *a = Expr::AddrOf(Box::new(ref_src.clone()));
                } else {
                    forward_into(a, v, src, ref_src, ok);
                }
            }
        }
        _ => {}
    }
}

fn mentions(s: &Stmt, v: VarId) -> usize {
    let mut n = 0;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if matches!(e, Expr::Var(x) if *x == v) {
            n += 1;
        }
    });
    n
}

/// Methods the inline destructor of `cls` calls (`~basic_string() { internal_dereference(); }`
/// -> `internal_dereference`).
fn inline_dtor_callees(db: &TypeDb, cls: &str) -> Vec<String> {
    let base = strip_tmpl(cls);
    let last = sig::split_scope(&base).1.to_string();
    let key = format!("{base}::~{last}");
    let mut out = vec![];
    for d in db.decls.get(&key).map(|v| v.as_slice()).unwrap_or(&[]) {
        let Some(body) = &d.inline_body else { continue };
        let toks: Vec<&str> = body.split_whitespace().collect();
        for w in toks.windows(2) {
            if w[1] == "(" && w[0].chars().next().is_some_and(|c| c.is_alphabetic() || c == '_') && !matches!(w[0], "if" | "while" | "for" | "return" | "sizeof") {
                out.push(w[0].to_string());
            }
        }
    }
    out
}

/// The end of a frame object's life written out: its destructor (or what its inline destructor
/// calls), possibly behind a null test of its address, as the last statement mentioning it.
/// C++ destroys objects implicitly (temporaries at the end of their full expression, named
/// objects at the end of their scope), so the call is no statement of the source; dropping it
/// also lets a call result used once become the temporary it was.
fn drop_frame_object_dtor_calls(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb) {
    let total = |body: &Vec<Stmt>, v: VarId| -> usize { body.iter().map(|s| mentions(s, v)).sum() };
    let snapshot = body.clone();
    let on_object = |e: &Expr, v: VarId| -> bool {
        match e {
            Expr::AddrOf(x) => matches!(&**x, Expr::Var(w) if *w == v),
            Expr::Var(w) => *w == v && matches!(vars[v].ty, Type::Ptr(_)),
            _ => false,
        }
    };
    let dtor_call = |s: &Stmt| -> Option<VarId> {
        let call = match s {
            Stmt::Expr(e) => e,
            Stmt::If { cond, then, els } if els.is_empty() && then.len() == 1 => match &then[0] {
                Stmt::Expr(e) if matches!(cond, Expr::AddrOf(_)) => e,
                _ => return None,
            },
            _ => return None,
        };
        let Expr::Call { callee: Callee::Method { sig: sg, this, .. }, args, .. } = call else { return None };
        let v = match &**this {
            Expr::AddrOf(x) => match &**x {
                Expr::Var(w) => *w,
                _ => return None,
            },
            _ => return None,
        };
        if !on_object(this, v) {
            return None;
        }
        let name = sg.qualified_name.rsplit("::").next().unwrap_or("");
        let is_dtor = sig::is_dtor(sg) && args.iter().all(|a| a.as_int().is_some());
        // what the receiver class's inline destructor calls, on a frame object (the class of
        // the object is the method's: the receiver is its start)
        let inline_part = args.is_empty()
            && matches!(vars[v].kind, VarKind::Stack { .. })
            && sg.this_class.as_deref().is_some_and(|c| inline_dtor_callees(db, c).iter().any(|n| n == name));
        (is_dtor || inline_part).then_some(v)
    };
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            if let Some(v) = dtor_call(&b[i]) {
                let before: usize = b[..=i].iter().map(|s| mentions(s, v)).sum();
                let here = mentions(&b[i], v);
                // the last mention, after the object was used, nothing elsewhere; or a member
                // object mentioned nowhere else (destroyed with the object around it)
                // (an object left an untyped buffer keeps its destructor call: nothing else
                // would destroy it; one defined whole by a call result is that call's type)
                let typed = named(&vars[v].ty).is_some()
                    || b[..i].iter().any(|s| matches!(s, Stmt::Assign { dst: Expr::Var(w), src } if *w == v && matches!(src, Expr::Call { .. })));
                if before == total(&snapshot, v) && ((before >= 3 && typed) || here == before) {
                    b.remove(i);
                    continue;
                }
            }
            i += 1;
        }
    });
}

/// Would forwarding the call result held in stack object `v` into statement `s` put its
/// frame temporary below an object `s` builds that the target has below it?
///
/// MWCC lays out frame objects from one list: named locals in declaration order, then C++
/// temporaries in creation order (an object built in an expression is created before the
/// temporaries of its arguments); the list is reversed and sorted stably by size rounded to the
/// alignment, then allocated upwards. Forwarded into `s`, `v` becomes a temporary created after
/// every other object of `s`, so among objects of its size class it gets the lowest offset. If
/// the target has one of them below `v`, `v` is a temporary created first instead: a
/// reference bound to the call result (`const T& r = f(); x += U(r);`).
fn forward_breaks_layout(v: VarId, s: &Stmt, vars: &[Var], folded: &[(Type, i32, u32)], forwardable: &HashSet<VarId>) -> bool {
    let VarKind::Stack { offset: ov, size: sv } = vars[v].kind else { return false };
    let key = |n: u32| (n + 3) & !3;
    let mut breaks = false;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        // an object built in place (`U(...)`) that was a frame object of the target
        if let Expr::Construct { class, .. } = e {
            if folded.iter().any(|(t, oc, sc)| t == class && key(*sc) == key(sv) && *oc < ov) {
                breaks = true;
            }
        }
        if let Expr::AddrOf(x) = e {
            if let Expr::Var(c) = &**x {
                // (another call result about to be forwarded too: no fixed place in the list)
                if *c != v && !forwardable.contains(c) {
                    if let VarKind::Stack { offset: oc, size: sc } = vars[*c].kind {
                        if key(sc) == key(sv) && oc < ov {
                            breaks = true;
                        }
                    }
                }
            }
        }
    });
    breaks
}

/// `T v = f(); g(v);` (v a stack object used once) -> `g(f());`: MWCC copies a returned object
/// into a named local but builds a temporary in place. Returns the call results kept apart
/// because forwarding them would change the frame layout ([`forward_breaks_layout`]).
pub fn forward_stack_objects(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>) -> Vec<VarId> {
    let mut kept: Vec<VarId> = vec![];
    // call results held in a stack object used once: candidates for forwarding
    let mut forwardable: HashSet<VarId> = HashSet::new();
    Stmt::for_each_block_mut(&mut body.clone(), &mut |b| {
        for st in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(w), src } = st {
                if src.has_call() && matches!(vars[*w].kind, VarKind::Stack { .. }) && body.iter().map(|x| mentions(x, *w)).sum::<usize>() == 2 {
                    forwardable.insert(*w);
                }
            }
        }
    });
    // frame objects no longer mentioned: built in place as `U(...)` by an earlier pass
    let folded: Vec<(Type, i32, u32)> = vars
        .iter()
        .enumerate()
        .filter_map(|(v, var)| match var.kind {
            VarKind::Stack { offset, size } if named(&var.ty).is_some() && !body.iter().any(|s| stmt_mentions(s, v)) => Some((var.ty.clone(), offset, size)),
            _ => None,
        })
        .collect();
    let total = |body: &Vec<Stmt>, v: VarId| -> usize { body.iter().map(|s| mentions(s, v)).sum() };
    let snapshot = body.clone();
    let mut counts: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    for (v, var) in vars.iter().enumerate() {
        if matches!(var.kind, VarKind::Stack { .. }) {
            counts.insert(v, total(&snapshot, v));
        }
    }
    // stack objects only ever written whole and read once right after (`tmp = kInvalidId;
    // f(tmp);` in several places): each pair is the value passed directly (MWCC makes the
    // by-value argument copy itself)
    let def_lv = |s: &Stmt| -> Option<(VarId, Expr, Expr)> {
        match s {
            Stmt::Assign { dst: dst @ Expr::Var(v), src } if matches!(vars[*v].kind, VarKind::Stack { .. }) && !src.uses_var(*v) => Some((*v, dst.clone(), src.clone())),
            Stmt::Assign { dst: dst @ Expr::Member { base, offset: 0, .. }, src } => match &**base {
                Expr::Var(v) if matches!(vars[*v].kind, VarKind::Stack { .. }) && !src.uses_var(*v) && !src.has_call() => Some((*v, dst.clone(), src.clone())),
                _ => None,
            },
            _ => None,
        }
    };
    let mut pairs: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    {
        let mut snap = body.clone();
        Stmt::for_each_block_mut(&mut snap, &mut |b| {
            for i in 0..b.len().saturating_sub(1) {
                if let Some((v, dst, _)) = def_lv(&b[i]) {
                    // (a redefinition is no read: `v = a; v = b;`)
                    let redef = matches!(&b[i + 1], Stmt::Assign { dst: d, .. } if d.uses_var(v));
                    if !redef && mentions(&b[i + 1], v) == 1 && matches!(b[i + 1], Stmt::Expr(_) | Stmt::Assign { .. } | Stmt::Return(_)) {
                        let mut found = false;
                        Stmt::walk_exprs(std::slice::from_ref(&b[i + 1]), &mut |e| {
                            if *e == dst {
                                found = true;
                            }
                        });
                        if found {
                            *pairs.entry(v).or_default() += 1;
                        }
                    }
                }
            }
        });
    }
    let pair_ok: HashSet<VarId> = pairs.iter().filter(|(v, n)| **n >= 2 && counts.get(v) == Some(&(2 * **n))).map(|(v, _)| *v).collect();
    if !pair_ok.is_empty() {
        Stmt::for_each_block_mut(body, &mut |b| {
            let mut i = 0;
            while i + 1 < b.len() {
                if let Some((v, dst, src)) = def_lv(&b[i]) {
                    if pair_ok.contains(&v) {
                        let mut s = b[i + 1].clone();
                        Stmt::rewrite_exprs(std::slice::from_mut(&mut s), &mut |e| {
                            if *e == dst {
                                *e = src.clone();
                            }
                        });
                        if mentions(&s, v) == 0 {
                            b[i + 1] = s;
                            b.remove(i);
                            continue;
                        }
                    }
                }
                i += 1;
            }
        });
    }
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            // `w = f(); v = w;` is `T v = f();` (MWCC: temporary + copy); v stays a named object
            if let (Stmt::Assign { dst: Expr::Var(w), src: call }, Stmt::Assign { dst: Expr::Var(v), src: Expr::Var(w2) }) = (&b[i], &b[i + 1]) {
                if w == w2 && w != v && counts.get(w) == Some(&2) && call.has_call() && matches!(vars[*v].kind, VarKind::Stack { .. }) {
                    let (v, call) = (*v, call.clone());
                    b[i + 1] = Stmt::Assign { dst: Expr::Var(v), src: call };
                    b.remove(i);
                    if let Some(c) = counts.get_mut(&v) {
                        *c = usize::MAX; // never forward a copied object
                    }
                    continue;
                }
            }
            let cand = match &b[i] {
                Stmt::Assign { dst: Expr::Var(v), src } if counts.get(v) == Some(&2) && !src.uses_var(*v) && !matches!(src, Expr::Var(w) if matches!(vars[*w].kind, VarKind::Stack { .. })) => Some((*v, src.clone())),
                _ => None,
            };
            // a narrower value stored than read back (`u8` written, the word read): no forwarding
            let wider_read = |v: VarId, src: &Expr, s: &Stmt| -> bool {
                let Some(have) = scalar_size(&types::ty_of(src, vars)).filter(|n| *n > 0) else { return false };
                let mut wide = false;
                Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
                    if let Expr::Member { base, offset, ty } = e {
                        if matches!(**base, Expr::Var(x) if x == v) && scalar_size(ty).map_or(false, |n| *offset as i64 + n as i64 > have as i64) {
                            wide = true;
                        }
                    }
                });
                wide
            };
            if let Some((v, src)) = cand {
                if src.has_call() && mentions(&b[i + 1], v) == 1 && forward_breaks_layout(v, &b[i + 1], vars, &folded, &forwardable) {
                    kept.push(v);
                    i += 1;
                    continue;
                }
                if mentions(&b[i + 1], v) == 1 && !wider_read(v, &src, &b[i + 1]) && !matches!(b[i + 1], Stmt::If { .. } | Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::Switch { .. }) {
                    let mut s = b[i + 1].clone();
                    let mut ok = true;
                    // an object copied into the frame from another object, then passed by
                    // reference: the copy is the source's (`f(T(x))`), not `f(x)`
                    let sty = types::ty_of(&src, vars);
                    let ref_src = if src.is_lvalue() && named(strip_cv(&sty)).is_some() && db.is_some() && types::is_aggregate(db, strip_cv(&sty)) {
                        Expr::Construct { class: strip_cv(&sty).clone(), ctor: None, args: vec![src.clone()] }
                    } else {
                        src.clone()
                    };
                    match &mut s {
                        Stmt::Assign { dst, src: s2 } => {
                            if matches!(dst, Expr::Var(x) if *x == v) {
                                ok = false;
                            }
                            forward_into(dst, v, &src, &ref_src, &mut ok);
                            forward_into(s2, v, &src, &ref_src, &mut ok);
                        }
                        Stmt::Expr(e) | Stmt::Return(Some(e)) => forward_into(e, v, &src, &ref_src, &mut ok),
                        _ => ok = false,
                    }
                    if ok && mentions(&s, v) == 0 {
                        b[i + 1] = s;
                        b.remove(i);
                        // the previous statement may feed the merged one too (`T a = f(); T b = g();
                        // h(b, a);`, arguments evaluated right to left)
                        i = i.saturating_sub(1);
                        continue;
                    }
                }
            }
            i += 1;
        }
    });
    kept
}

/// Does statement `s` define all of stack object `v` (constructor call on `&v`, or `v = e`)?
pub fn defines_object(s: &Stmt, v: VarId) -> bool {
    match s {
        Stmt::Expr(Expr::Call { callee: Callee::Method { sig: sg, this, .. }, .. }) if sig::is_ctor(sg) => {
            matches!(&**this, Expr::AddrOf(inner) if matches!(&**inner, Expr::Var(x) if *x == v))
        }
        Stmt::Assign { dst: Expr::Var(x), src } => *x == v && !src.uses_var(v),
        _ => false,
    }
}

pub fn stmt_mentions(s: &Stmt, v: VarId) -> bool {
    let mut found = false;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if matches!(e, Expr::Var(x) if *x == v) {
            found = true;
        }
    });
    found
}

/// Stack objects (class-typed) that can be declared where they are first defined: the first
/// statement mentioning them defines them entirely, and every later mention is in the same
/// statement list (so the declaration's scope covers all uses).
pub fn decl_at_first_def(body: &[Stmt], vars: &[Var]) -> HashSet<VarId> {
    fn lists<'a>(b: &'a [Stmt], out: &mut Vec<&'a [Stmt]>) {
        out.push(b);
        for s in b {
            match s {
                Stmt::If { then, els, .. } => {
                    lists(then, out);
                    lists(els, out);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => lists(body, out),
                Stmt::For { init, step, body, .. } => {
                    lists(init, out);
                    lists(step, out);
                    lists(body, out);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        lists(&c.body, out);
                    }
                }
                _ => {}
            }
        }
    }
    let mut all = vec![];
    lists(body, &mut all);
    let mut out = HashSet::new();
    for (v, var) in vars.iter().enumerate() {
        // stack objects, and register locals holding a small object (`TUniqueId id = ...`)
        let small_object = matches!(var.kind, VarKind::Local)
            && named(&var.ty).map_or(false, |n| n.starts_with(|c: char| c.is_ascii_uppercase()) && crate::types::is_aggregate(None, &var.ty));
        if !matches!(var.kind, VarKind::Stack { .. }) && !small_object {
            continue;
        }
        // find the list whose statement first mentions v (pre-order over lists is fine: a list's
        // own statements are scanned before nested lists, and a mention inside a nested statement
        // also counts as a mention by its parent statement)
        for l in &all {
            let Some(i) = l.iter().position(|s| stmt_mentions(s, v)) else { continue };
            // a compound statement whose own expression doesn't mention v: look inside it
            let own_mention = match &l[i] {
                Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => Some(cond.uses_var(v)),
                Stmt::For { cond, .. } => Some(cond.uses_var(v)),
                Stmt::Switch { e, .. } => Some(e.uses_var(v)),
                _ => None,
            };
            if own_mention == Some(false) {
                continue;
            }
            if defines_object(&l[i], v) {
                // all mentions of v must be inside l[i..]
                let total: usize = body.iter().map(|s| count_mentions(s, v)).sum();
                let inside: usize = l[i..].iter().map(|s| count_mentions(s, v)).sum();
                if total == inside {
                    out.insert(v);
                }
            }
            break;
        }
    }
    out
}

pub fn count_mentions(s: &Stmt, v: VarId) -> usize {
    let mut n = 0;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if matches!(e, Expr::Var(x) if *x == v) {
            n += 1;
        }
    });
    n
}


/// Stack slots that are written but never read (compiler temporaries the source never named,
/// e.g. the extra copy MWCC makes when passing a struct by value): drop the stores.
pub fn drop_dead_stack_stores(body: &mut Vec<Stmt>, vars: &[Var]) {
    drop_dead_stack_stores_kept(body, vars);
}

/// [`drop_dead_stack_stores`], returning what was dropped (in frame-offset order, `order` unset).
pub fn drop_dead_stack_stores_kept(body: &mut Vec<Stmt>, vars: &[Var]) -> Vec<DeadStackStore> {
    let mut dropped: Vec<DeadStackStore> = vec![];
    for (v, var) in vars.iter().enumerate() {
        if !matches!(var.kind, VarKind::Stack { .. }) {
            continue;
        }
        let total: usize = body.iter().map(|s| count_mentions(s, v)).sum();
        if total == 0 {
            continue;
        }
        let mut stores = 0;
        fn count_stores(b: &[Stmt], v: VarId, n: &mut usize) {
            for s in b {
                match s {
                    Stmt::Assign { dst: Expr::Var(x), src } if *x == v && !src.uses_var(v) => *n += 1,
                    Stmt::If { then, els, .. } => {
                        count_stores(then, v, n);
                        count_stores(els, v, n);
                    }
                    Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => count_stores(body, v, n),
                    Stmt::Switch { cases, .. } => {
                        for c in cases {
                            count_stores(&c.body, v, n);
                        }
                    }
                    _ => {}
                }
            }
        }
        count_stores(body, v, &mut stores);
        if stores != total {
            continue;
        }
        let VarKind::Stack { offset, size } = var.kind else { continue };
        Stmt::for_each_block_mut(body, &mut |b| {
            let mut out: Vec<Stmt> = Vec::with_capacity(b.len());
            for s in b.drain(..) {
                match s {
                    Stmt::Assign { dst: Expr::Var(x), src } if x == v => {
                        // the value: a register temp defined earlier in the block stands for
                        // its (side-effect free) definition
                        let mut value = src.clone();
                        if let Expr::Var(t) = &src {
                            if let Some(Stmt::Assign { src: def, .. }) = out.iter().rev().find(|p| matches!(p, Stmt::Assign { dst: Expr::Var(d), .. } if d == t)) {
                                if !def.has_call() {
                                    value = def.clone();
                                }
                            }
                        }
                        dropped.push(DeadStackStore { offset, size, value, order: 0 });
                        if src.has_call() {
                            out.push(Stmt::Expr(src));
                        }
                    }
                    s => out.push(s),
                }
            }
            *b = out;
        });
    }
    dropped.sort_by_key(|d| d.offset);
    dropped
}

/// A class-typed stack object without a default constructor that is not defined whole at its
/// first use can't be declared: keep it an untyped byte buffer (accesses stay raw).
/// Does class `cls` (or a base) declare a pure virtual method that no class on the way down
/// overrides?
pub fn is_abstract(db: &TypeDb, cls: &str) -> bool {
    fn walk(db: &TypeDb, cls: &str, depth: u32, concrete: &mut HashSet<(String, usize)>) -> bool {
        if depth > 12 {
            return false;
        }
        let key = strip_tmpl(cls);
        let prefix = format!("{key}::");
        let mut pure = vec![];
        for (name, ds) in db.decls.range(prefix.clone()..) {
            if !name.starts_with(&prefix) {
                break;
            }
            let m = &name[prefix.len()..];
            if m.contains("::") {
                continue;
            }
            for d in ds {
                if d.is_pure {
                    pure.push((m.to_string(), d.params.len()));
                } else if d.is_virtual || !d.is_static {
                    concrete.insert((m.to_string(), d.params.len()));
                }
            }
        }
        if pure.iter().any(|p| !concrete.contains(p)) {
            return true;
        }
        let Some(c) = sig::find_class(db, cls) else { return false };
        let bases: Vec<String> = c.bases.iter().map(|b| b.name.clone()).collect();
        bases.iter().any(|b| walk(db, b, depth + 1, concrete))
    }
    walk(db, cls, 0, &mut HashSet::new())
}

pub fn untype_undeclarable(body: &[Stmt], vars: &mut [Var], db: Option<&TypeDb>) {
    let Some(db) = db else { return };
    let at_def = decl_at_first_def(body, vars);
    for (v, var) in vars.iter_mut().enumerate() {
        let VarKind::Stack { size, .. } = var.kind else { continue };
        let Some(cls) = named(&var.ty).map(|s| s.to_string()) else { continue };
        // an object of an abstract class can't be declared at all (the real object is of a
        // derived class whose constructor was inlined)
        if is_abstract(db, &cls) {
            var.ty = Type::Unknown { size };
            continue;
        }
        if at_def.contains(&v) {
            continue;
        }
        let key = format!("{}::{}", strip_tmpl(&cls), strip_tmpl(sig::split_scope(&cls).1));
        let has_default = match db.decls.get(&key) {
            Some(ds) => ds.iter().any(|d| d.params.is_empty()),
            // (an instance of a class template the context never instantiates: nothing says
            // it can be default-constructed)
            None => !(cls.contains('<') && sig::find_class(db, &cls).is_none()),
        };
        if !has_default {
            var.ty = Type::Unknown { size };
        }
    }
}

/// `t = e; return t;` with `t` used nowhere else -> `return e;` (left behind when the destructor
/// call that separated them is dropped as implicit).
fn return_kept_values(body: &mut Vec<Stmt>, vars: &[Var]) {
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            *uses.entry(*v).or_default() += 1;
        }
    });
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            let hit = match (&b[i], &b[i + 1]) {
                (Stmt::Assign { dst: Expr::Var(t), .. }, Stmt::Return(Some(Expr::Var(r)))) => t == r && matches!(vars[*t].kind, VarKind::Local) && uses.get(t) == Some(&2),
                _ => false,
            };
            if hit {
                if let Stmt::Assign { src, .. } = b.remove(i) {
                    b[i] = Stmt::Return(Some(src));
                }
                continue;
            }
            i += 1;
        }
    });
}
