//! C++ idioms MWCC lowers into plain code: `new` expressions, constructor initializer lists and
//! implicit vtable-pointer stores, destructor wrappers (null check, delete flag, base/member
//! destructor calls).

use crate::ir::*;
use crate::sig;
use crate::types;
use mwdec_core::{Type, TypeDb};
use std::collections::HashSet;

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

/// A guessed struct return (`StructRet` pointing at an unknown type) takes the class of the
/// object constructed into it.
fn type_guessed_sret(ir: &mut IrFunction) {
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
    if let Some(c) = cls {
        ir.vars[sret].ty = t_ptr(Type::Named(c.clone()));
        ir.sig.ret = Type::Named(c);
    }
}

pub fn apply(ir: &mut IrFunction, db: Option<&TypeDb>) {
    type_guessed_sret(ir);
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
    forward_stack_objects(&mut ir.body, &vars);
    drop_dead_stack_stores(&mut ir.body, &vars);
    if let Some(sret) = ir.vars.iter().position(|v| v.kind == VarKind::StructRet) {
        let rt = ir.sig.ret.clone();
        fold_struct_return(&mut ir.body, sret, &rt, db);
    }
    let body = ir.body.clone();
    untype_undeclarable(&body, &mut ir.vars, db);
}

/// Scalar leaf members of a class in layout order: (offset, type).
fn flat_fields(db: &TypeDb, cls: &str, base: i32, out: &mut Vec<(i32, Type)>, depth: u32) -> bool {
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
fn copy_from_stores(stores: &[(i32, Expr)], ret: &Type, db: Option<&TypeDb>) -> Option<Expr> {
    let db = db?;
    let cls = named(&types::resolve(Some(db), ret)).map(|s| s.to_string())?;
    let mut fields = vec![];
    if !flat_fields(db, &cls, 0, &mut fields, 0) || fields.len() != stores.len() || fields.is_empty() {
        return None;
    }
    let mut src: Option<(&Expr, i32, bool)> = None;
    for (off, _) in &fields {
        let (_, v) = stores.iter().find(|(o, _)| o == off)?;
        let (base, boff, ptr) = match v {
            Expr::Load { base, offset, .. } => (&**base, *offset, true),
            Expr::Member { base, offset, .. } => (&**base, *offset, false),
            _ => return None,
        };
        let start = boff - off;
        match src {
            None => src = Some((base, start, ptr)),
            Some((b, s0, p)) if b == base && s0 == start && p == ptr => {}
            _ => return None,
        }
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
            while k > 0 {
                match &b[k - 1] {
                    Stmt::Assign { dst: Expr::Load { base, offset, .. }, src } if matches!(**base, Expr::Var(v) if v == sret) && !src.uses_var(sret) => {
                        stores.push((*offset, src.clone()));
                        k -= 1;
                    }
                    _ => break,
                }
            }
            if !stores.is_empty() {
                if let Some(e) = construct_from_stores(&stores, ret, db).or_else(|| copy_from_stores(&stores, ret, db)) {
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
    while k < ir.body.len() {
        let take = match &ir.body[k] {
            st @ Stmt::Expr(_) => ctor_init_of(st, this, &own, db),
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
    ir.init_list = inits;
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
        let refs: Vec<bool> = (0..init.args.len())
            .map(|i| init.ctor.as_ref().and_then(|s| s.params.get(i)).map_or(init.ctor.is_none(), |p| matches!(p.ty.unqualified(), Type::Ref(_))))
            .collect();
        for (a, by_ref) in init.args.iter_mut().zip(refs) {
            match spellable_in_init(a, by_ref, &ir.body[..k], &vars, this, db) {
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
        if !init.args.is_empty() {
            inits.push(init);
        }
    }
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
fn single_store(before: &[Stmt], vs: &[VarId]) -> Option<Expr> {
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
    match found? {
        Expr::Member { base, offset: 0, .. } if matches!(&*base, Expr::Global { .. }) => Some(*base),
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
fn spellable_in_init(e: &Expr, by_ref: bool, before: &[Stmt], vars: &[Var], this: VarId, db: Option<&TypeDb>) -> Option<Expr> {
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
                if !matches!(vars[*v].kind, VarKind::Param { .. } | VarKind::This) || *v == this {
                    ok = false;
                }
            }
        });
        ok
    };
    // a temporary object (by value or by address)
    let inner = match e {
        Expr::AddrOf(x) => stack_var(x),
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
            ([], Some((cs, args))) if (by_ref || !matches!(e, Expr::AddrOf(_))) && args.iter().all(pure) => {
                Some(Expr::Construct { class: Type::Named(cs.this_class.clone().unwrap_or_default()), ctor: Some(cs), args })
            }
            // a single-member object set by one store: a copy of the object the value came
            // from, or the member's value
            ([], None) if single_store(before, &slot_aliases(vars, v)).is_some() => single_store(before, &slot_aliases(vars, v)),
            // never assigned as a whole: a default-constructed temporary (member-wise setup
            // stays in the body)
            ([], None) if default_constructible(db, &vars[v].ty) => Some(e.clone()),
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
                match (defs.as_slice(), nested) {
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

fn dtor_unwrap(ir: &mut IrFunction, this: VarId) {
    let hidden: Vec<VarId> = ir.vars.iter().enumerate().filter(|(_, v)| v.kind == VarKind::Hidden).map(|(i, _)| i).collect();
    // `if (this) { ... }` wrapper
    if ir.body.len() == 1 {
        if let Stmt::If { cond, then, els } = &ir.body[0] {
            if els.is_empty() && (is_this(cond, this) || matches!(cond, Expr::Binary { op: BinOp::Ne, l, r, .. } if is_this(l, this) && r.as_int() == Some(0))) {
                ir.body = then.clone();
            }
        }
    }
    let uses_hidden = |e: &Expr| hidden.iter().any(|h| e.uses_var(*h));
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        b.retain(|s| match s {
            // the `delete this` branch of the deleting destructor
            Stmt::If { cond, .. } if uses_hidden(cond) => false,
            // inlined member destructors start with `if (&this->member != 0)`: implicit
            Stmt::If { cond, .. } if is_member_addr_test(cond, this) => false,
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
fn forward_into(e: &mut Expr, v: VarId, src: &Expr, ok: &mut bool) {
    let is_v = |x: &Expr| matches!(x, Expr::Var(y) if *y == v);
    match e {
        Expr::Var(_) if is_v(e) => *e = src.clone(),
        Expr::Member { base, .. } if is_v(base) => **base = src.clone(),
        Expr::AddrOf(inner) if is_v(inner) => *ok = false,
        Expr::Call { callee, args, .. } => {
            let sig = match callee {
                Callee::Method { sig, this, .. } => {
                    if matches!(&**this, Expr::AddrOf(i) if is_v(i)) {
                        **this = Expr::AddrOf(Box::new(src.clone()));
                    } else {
                        forward_into(this, v, src, ok);
                    }
                    Some(sig.clone())
                }
                Callee::Direct { sig, .. } => Some(sig.clone()),
                Callee::Virtual { this, sig, .. } => {
                    forward_into(this, v, src, ok);
                    sig.clone()
                }
                Callee::Indirect(f) => {
                    forward_into(f, v, src, ok);
                    None
                }
            };
            for (n, a) in args.iter_mut().enumerate() {
                let is_ref = sig.as_ref().and_then(|s| s.params.get(n)).map_or(false, |p| matches!(strip_cv(&p.ty), Type::Ref(_)));
                if is_ref && matches!(a, Expr::AddrOf(i) if is_v(i)) {
                    *a = Expr::AddrOf(Box::new(src.clone()));
                } else {
                    forward_into(a, v, src, ok);
                }
            }
        }
        Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => forward_into(base, v, src, ok),
        Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } | Expr::AddrOf(x) => forward_into(x, v, src, ok),
        Expr::Index { base, index, .. } => {
            forward_into(base, v, src, ok);
            forward_into(index, v, src, ok);
        }
        Expr::Binary { l, r, .. } => {
            forward_into(l, v, src, ok);
            forward_into(r, v, src, ok);
        }
        Expr::Ternary { c, t, f, .. } => {
            forward_into(c, v, src, ok);
            forward_into(t, v, src, ok);
            forward_into(f, v, src, ok);
        }
        Expr::New { placement, args, .. } => {
            for a in placement.iter_mut().chain(args.iter_mut()) {
                forward_into(a, v, src, ok);
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
                    *a = Expr::AddrOf(Box::new(src.clone()));
                } else {
                    forward_into(a, v, src, ok);
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

/// `T v = f(); g(v);` (v a stack object used once) -> `g(f());`: MWCC copies a returned object
/// into a named local but builds a temporary in place.
pub fn forward_stack_objects(body: &mut Vec<Stmt>, vars: &[Var]) {
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
                    if mentions(&b[i + 1], v) == 1 && matches!(b[i + 1], Stmt::Expr(_) | Stmt::Assign { .. } | Stmt::Return(_)) {
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
            if let Some((v, src)) = cand {
                if mentions(&b[i + 1], v) == 1 && !matches!(b[i + 1], Stmt::If { .. } | Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::Switch { .. }) {
                    let mut s = b[i + 1].clone();
                    let mut ok = true;
                    match &mut s {
                        Stmt::Assign { dst, src: s2 } => {
                            if matches!(dst, Expr::Var(x) if *x == v) {
                                ok = false;
                            }
                            forward_into(dst, v, &src, &mut ok);
                            forward_into(s2, v, &src, &mut ok);
                        }
                        Stmt::Expr(e) | Stmt::Return(Some(e)) => forward_into(e, v, &src, &mut ok),
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
        Stmt::for_each_block_mut(body, &mut |b| {
            let mut out = Vec::with_capacity(b.len());
            for s in b.drain(..) {
                match s {
                    Stmt::Assign { dst: Expr::Var(x), src } if x == v => {
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
}

/// A class-typed stack object without a default constructor that is not defined whole at its
/// first use can't be declared: keep it an untyped byte buffer (accesses stay raw).
pub fn untype_undeclarable(body: &[Stmt], vars: &mut [Var], db: Option<&TypeDb>) {
    let Some(db) = db else { return };
    let at_def = decl_at_first_def(body, vars);
    for (v, var) in vars.iter_mut().enumerate() {
        let VarKind::Stack { size, .. } = var.kind else { continue };
        let Some(cls) = named(&var.ty).map(|s| s.to_string()) else { continue };
        if at_def.contains(&v) {
            continue;
        }
        let key = format!("{}::{}", strip_tmpl(&cls), strip_tmpl(sig::split_scope(&cls).1));
        let has_default = match db.decls.get(&key) {
            Some(ds) => ds.iter().any(|d| d.params.is_empty()),
            None => true,
        };
        if !has_default {
            var.ty = Type::Unknown { size };
        }
    }
}
