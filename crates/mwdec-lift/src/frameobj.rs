//! Frame objects that only carry a call's by-value result to a single read.
//!
//! `x = obj->VGetTime().GetSeconds();` with a virtual (or otherwise unsized) by-value return:
//! MWCC passes a frame temporary as the struct-return pointer and reads one member back. The
//! lifter sees an untyped frame region written whole by the call and read once; declaring the
//! region as a named object adds a copy (`T t = f();` is a temporary plus a copy), so the read
//! moves to the call: `tmp = *(float*)&obj->VGetTime();`. Moving the read earlier is safe: only
//! the call writes the region and its address goes nowhere else.

use crate::ir::*;
use mwdec_core::Type;

pub fn fold_single_reads(ir: &mut IrFunction) {
    let n = ir.vars.len();
    for v in 0..n {
        let VarKind::Stack { offset, .. } = ir.vars[v].kind else { continue };
        if !matches!(ir.vars[v].ty, Type::Unknown { .. }) {
            continue;
        }
        // every mention: one whole-object call store, one scalar member read, nothing else
        let (mut total, mut reads, mut addressed) = (0usize, 0usize, false);
        let mut read_ty: Option<(i32, Type)> = None;
        Stmt::walk_exprs(&ir.body, &mut |e| match e {
            Expr::Var(x) if *x == v => total += 1,
            Expr::Member { base, offset: o, ty } if matches!(&**base, Expr::Var(x) if *x == v) => {
                reads += 1;
                read_ty = Some((*o, ty.clone()));
            }
            Expr::AddrOf(x) if matches!(&**x, Expr::Var(w) if *w == v) => addressed = true,
            _ => {}
        });
        let Some((roff, rty)) = read_ty else { continue };
        if addressed || reads != 1 || total != 2 || matches!(rty, Type::Named(_) | Type::Array(..) | Type::Unknown { .. }) {
            continue;
        }
        // the store and the read in the same statement list, store first
        let t = ir.vars.len();
        let mut done = false;
        Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
            if done {
                return;
            }
            let Some(si) = b.iter().position(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src: Expr::Call { .. } } if *x == v)) else { return };
            let read_at = b.iter().enumerate().skip(si + 1).find(|(_, s)| {
                let mut hit = false;
                Stmt::walk_exprs(std::slice::from_ref(*s), &mut |e| {
                    if matches!(e, Expr::Member { base, .. } if matches!(&**base, Expr::Var(x) if *x == v)) {
                        hit = true;
                    }
                });
                hit
            });
            let Some((ri, _)) = read_at else { return };
            let Stmt::Assign { src: call, .. } = b[si].clone() else { return };
            b[si] = Stmt::Assign { dst: Expr::Var(t), src: Expr::Member { base: Box::new(call), offset: roff, ty: rty.clone() } };
            Stmt::rewrite_exprs(&mut b[ri..=ri], &mut |e| {
                if matches!(e, Expr::Member { base, .. } if matches!(&**base, Expr::Var(x) if *x == v)) {
                    *e = Expr::Var(t);
                }
            });
            done = true;
        });
        if done {
            ir.vars.push(Var { name: format!("result_{offset:x}"), ty: rty, kind: VarKind::Local });
        }
    }
}

/// `p = &dst - 4; q = &src - 4; for (i = 0; i < n; i++) { t = q[2]; p[1] = q[1]; p[2] = t;
/// p += 2; q += 2; } [p[1] = q[1];]`: MWCC's copy loop for an object of more than 64 bytes
/// (`lwzu`/`stwu` with a pre-decremented pointer and a counter). It is the assignment
/// `dst = src` of an object of `8n` (or `8n + 4`) bytes.
pub fn fold_block_copies(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let vars = ir.vars.clone();
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut k = 0;
        while k + 2 < b.len() {
            if let Some((n, s)) = block_copy_at(b, k, &vars, db) {
                b.splice(k..k + n, std::iter::once(s));
            }
            k += 1;
        }
    });
}

/// (base object, pointer var) of `p = &obj.@-4`.
fn pre_decremented(s: &Stmt) -> Option<(VarId, Expr)> {
    let Stmt::Assign { dst: Expr::Var(p), src: Expr::AddrOf(x) } = s else { return None };
    let Expr::Member { base, offset: -4, .. } = &**x else { return None };
    Some((*p, (**base).clone()))
}

fn ptr_load(e: &Expr, p: VarId) -> Option<i32> {
    match e {
        Expr::Load { base, offset, .. } if matches!(&**base, Expr::Var(x) if *x == p) => Some(*offset),
        _ => None,
    }
}

fn block_copy_at(b: &[Stmt], k: usize, vars: &[Var], db: Option<&mwdec_core::TypeDb>) -> Option<(usize, Stmt)> {
    let (a, ea) = pre_decremented(&b[k])?;
    let (c, ec) = pre_decremented(&b[k + 1])?;
    let Stmt::For { init, cond, step, body } = &b[k + 2] else { return None };
    let Expr::Binary { op: BinOp::Lt, l, r, .. } = cond else { return None };
    let (Expr::Var(i), Some(n)) = (&**l, r.as_int()) else { return None };
    if !matches!(init.as_slice(), [Stmt::Assign { dst: Expr::Var(x), src }] if x == i && src.as_int() == Some(0)) || step.len() != 1 || n <= 0 {
        return None;
    }
    // which pointer is written: the destination
    let mut stores: Vec<(VarId, i32)> = vec![];
    let mut temps: std::collections::HashMap<VarId, (VarId, i32)> = std::collections::HashMap::new();
    let mut incs = 0;
    let mut copies: Vec<(VarId, i32, VarId, i32)> = vec![];
    for s in body {
        let Stmt::Assign { dst, src } = s else { return None };
        match (dst, src) {
            (Expr::Var(t), x) if ptr_load(x, a).is_some() || ptr_load(x, c).is_some() => {
                let (pv, o) = if let Some(o) = ptr_load(x, a) { (a, o) } else { (c, ptr_load(x, c)?) };
                temps.insert(*t, (pv, o));
            }
            (Expr::Var(p), Expr::AddrOf(x)) if (*p == a || *p == c) && ptr_load(x, *p) == Some(8) => incs += 1,
            (d, x) => {
                let (dp, doff) = if let Some(o) = ptr_load(d, a) { (a, o) } else { (c, ptr_load(d, c)?) };
                let (sp, soff) = match x {
                    Expr::Var(t) => *temps.get(t)?,
                    x => {
                        if let Some(o) = ptr_load(x, a) {
                            (a, o)
                        } else {
                            (c, ptr_load(x, c)?)
                        }
                    }
                };
                stores.push((dp, doff));
                copies.push((dp, doff, sp, soff));
            }
        }
    }
    if incs != 2 || copies.len() != 2 || !copies.iter().all(|&(dp, doff, sp, soff)| dp != sp && doff == soff && (doff == 4 || doff == 8)) || copies[0].1 == copies[1].1 || copies[0].0 != copies[1].0 {
        return None;
    }
    let (dst_p, src_p) = (copies[0].0, copies[0].2);
    let (dst_obj, src_obj) = if dst_p == a { (ea, ec) } else { (ec, ea) };
    // the tail word
    let mut size = 8 * n as u32;
    let mut used = 3;
    if let Some(Stmt::Assign { dst, src }) = b.get(k + 3) {
        if ptr_load(dst, dst_p) == Some(4) && ptr_load(src, src_p) == Some(4) {
            size += 4;
            used = 4;
        }
    }
    // the pointers and the counter are dead afterwards
    let mut later = false;
    Stmt::walk_exprs(&b[k + used..], &mut |e| {
        if matches!(e, Expr::Var(x) if *x == a || *x == c || x == i) {
            later = true;
        }
    });
    if later {
        return None;
    }
    // the copied object: the destination's type when it has that size (its assignment then
    // copies one block; a struct of many members would call its `operator=`), else the source's
    let obj_ty = |e: &Expr| -> Type {
        let t = crate::types::ty_of(e, vars);
        match strip_cv(&t) {
            Type::Ref(x) => strip_cv(x).clone(),
            t => t.clone(),
        }
    };
    let (dty, sty) = (obj_ty(&dst_obj), obj_ty(&src_obj));
    let fits = |t: &Type| named(t).is_some() && crate::types::size_of(db, t) == Some(size);
    let t = if fits(&dty) {
        dty.clone()
    } else if fits(&sty) {
        sty.clone()
    } else {
        return None;
    };
    let whole = |e: Expr, et: &Type| if *et == t { e } else { Expr::Member { base: Box::new(e), offset: 0, ty: t.clone() } };
    Some((used, Stmt::Assign { dst: whole(dst_obj, &dty), src: whole(src_obj, &sty) }))
}

/// A frame object of a class whose whole size is one scalar store (an empty tag struct, a
/// one-word class) written from another object's bytes: the object copied whole,
/// `t = *(T*)&src`, so that it can become a copy `T(src)` where it is passed.
pub fn whole_object_copies(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let vars = ir.vars.clone();
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    crate::inline::count_uses(&ir.body, &mut uses);
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let Stmt::Assign { dst: Expr::Member { base, offset: 0, ty: dt }, src } = &b[i] else {
                i += 1;
                continue;
            };
            let Expr::Var(v) = **base else {
                i += 1;
                continue;
            };
            let t = vars[v].ty.clone();
            let whole = matches!(vars[v].kind, VarKind::Stack { .. })
                && named(&t).is_some()
                && crate::types::is_aggregate(Some(db), &t)
                && crate::types::size_of(Some(db), &t).is_some_and(|n| scalar_size(dt) == Some(n));
            if !whole {
                i += 1;
                continue;
            }
            // the bytes come from a global (directly or through a register loaded just before)
            let (from, def_at) = match src {
                Expr::Var(r) if uses.get(r) == Some(&1) => match b[..i].iter().rposition(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if x == r)) {
                    Some(k) => match &b[k] {
                        Stmt::Assign { src: g @ Expr::Global { .. }, .. } => (g.clone(), Some(k)),
                        _ => {
                            i += 1;
                            continue;
                        }
                    },
                    None => {
                        i += 1;
                        continue;
                    }
                },
                g @ Expr::Global { .. } => (g.clone(), None),
                _ => {
                    i += 1;
                    continue;
                }
            };
            if scalar_size(&crate::types::ty_of(&from, &vars)) != scalar_size(dt) {
                i += 1;
                continue;
            }
            b[i] = Stmt::Assign { dst: Expr::Var(v), src: Expr::Member { base: Box::new(from), offset: 0, ty: t } };
            if let Some(k) = def_at {
                b.remove(k);
                continue;
            }
            i += 1;
        }
    });
}

/// A call to a function without a known signature (no header declaration, no mangled name)
/// passing the address of a frame object that was just filled whole from a class member
/// (`stack = *(this+8); fn_80037944(&mgr, &stack)`): the callee takes that class by value and
/// the frame object is the compiler's argument copy. The member is passed (`fn(mgr, this->mId)`)
/// and the stand-in declaration takes the class; when a dead frame store held the same value
/// first (an inline accessor returning the class by value), an explicit copy `T(x)`.
pub fn unknown_callee_byval(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let vars = ir.vars.clone();
    // mentions of every variable (the store's destination included)
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::Var(v) = e {
            *uses.entry(*v).or_default() += 1;
        }
    });
    let mut dead = std::mem::take(&mut ir.dead_stores);
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            let Some((v, obj, t)) = whole_member_store(&b[i], &vars, db) else {
                i += 1;
                continue;
            };
            if uses.get(&v) != Some(&2) {
                i += 1;
                continue;
            }
            let mut hit = false;
            if !copyable(db, &t) {
                i += 1;
                continue;
            }
            let size = crate::types::size_of(Some(db), &t).unwrap_or(0);
            let copy = match dead.iter().position(|d| d.size == size && same_place(&d.value, &obj)) {
                Some(k) => {
                    dead.remove(k);
                    Expr::Construct { class: t.clone(), ctor: None, args: vec![obj.clone()] }
                }
                None => obj.clone(),
            };
            let mut next = b[i + 1].clone();
            Stmt::rewrite_exprs(std::slice::from_mut(&mut next), &mut |e| {
                let Expr::Call { callee: Callee::Direct { symbol, sig }, args, .. } = e else { return };
                // (a C name the headers don't declare: the signature is the call site's guess)
                if crate::sig::demangle(symbol).is_some() || db.decls.contains_key(symbol.as_str()) || db.decls.contains_key(&format!("::{symbol}")) || db.functions.contains_key(symbol.as_str()) {
                    return;
                }
                for (n, a) in args.iter_mut().enumerate() {
                    if matches!(a, Expr::AddrOf(x) if matches!(**x, Expr::Var(w) if w == v)) {
                        *a = copy.clone();
                        if let Some(p) = sig.params.get_mut(n) {
                            p.ty = t.clone();
                        }
                        hit = true;
                    }
                }
            });
            if hit {
                b[i + 1] = next;
                b.remove(i);
                continue;
            }
            i += 1;
        }
    });
    ir.dead_stores = dead;
}

/// `v.@0 = <member>` where the member is a class object of exactly the stored size and `v` a
/// frame object: (v, the member as that class, the class).
fn whole_member_store(s: &Stmt, vars: &[Var], db: &mwdec_core::TypeDb) -> Option<(VarId, Expr, Type)> {
    let Stmt::Assign { dst, src } = s else { return None };
    let (v, dt) = match dst {
        Expr::Member { base, offset: 0, ty } => match **base {
            Expr::Var(v) => (v, ty.clone()),
            _ => return None,
        },
        _ => return None,
    };
    if !matches!(vars[v].kind, VarKind::Stack { .. }) {
        return None;
    }
    let n = scalar_size(&dt)?;
    let Expr::Load { base, offset, .. } = src else { return None };
    let bt = crate::types::ty_of(base, vars);
    let cls = named(pointee(&bt)?)?.to_string();
    let t = crate::aggregates::aggregate_at(db, &cls, *offset).into_iter().find(|t| crate::types::size_of(Some(db), t) == Some(n))?;
    Some((v, Expr::Load { base: base.clone(), offset: *offset, ty: t.clone() }, t))
}

/// Does `a` read the same memory as `b` (same base and offset, access type aside)?
fn same_place(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Member { base: x, offset: o, .. }, Expr::Member { base: y, offset: p, .. }) | (Expr::Load { base: x, offset: o, .. }, Expr::Load { base: y, offset: p, .. }) => o == p && x == y,
        (x, y) => x == y,
    }
}

/// A returned object built from a call's by-value result copied whole into it (word by word)
/// plus constant stores of the returned class's own members, the class having a constructor
/// taking that result's class: `return R(f());` (`return optional_object<CAABox>(
/// GetBoundingBox())`: the valid flag set, the box copied into the item storage).
pub fn fold_converting_return(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let Some(rv) = ir.vars.iter().position(|v| v.kind == VarKind::StructRet) else { return };
    let rt = strip_cv(&ir.sig.ret).clone();
    let Some(rcls) = named(&rt).map(|s| s.to_string()) else { return };
    let b = &ir.body;
    let Some(Stmt::Return(None)) = b.last() else { return };
    let end = b.len() - 1;
    // the result object: `s = f()` with every later statement a store into the returned object
    // or a register temp read from `s`
    let Some(si) = (0..end).rev().find(|&k| matches!(&b[k], Stmt::Assign { dst: Expr::Var(s), src: Expr::Call { .. } } if matches!(ir.vars[*s].kind, VarKind::Stack { .. }))) else { return };
    let Stmt::Assign { dst: Expr::Var(s), src: call } = &b[si] else { return };
    let (s, call) = (*s, call.clone());
    let t = strip_cv(&ir.vars[s].ty).clone();
    if named(&t).is_none() {
        return;
    }
    let Some(tsize) = crate::types::size_of(Some(db), &t) else { return };
    let mut temps: std::collections::HashMap<VarId, i32> = std::collections::HashMap::new();
    let mut copied: Vec<(i32, i32, u32)> = vec![]; // (dst offset, src offset, size)
    for st in &b[si + 1..end] {
        match st {
            Stmt::Assign { dst: Expr::Var(x), src: Expr::Member { base, offset, .. } } if matches!(**base, Expr::Var(w) if w == s) => {
                temps.insert(*x, *offset);
            }
            Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } if matches!(**base, Expr::Var(w) if w == rv) => {
                let n = scalar_size(ty).unwrap_or(0);
                match src {
                    Expr::Member { base: sb, offset: so, .. } if matches!(**sb, Expr::Var(w) if w == s) => copied.push((*offset, *so, n)),
                    Expr::Var(x) if temps.contains_key(x) => copied.push((*offset, temps[x], n)),
                    Expr::Int { .. } | Expr::Float { .. } => {}
                    _ => return,
                }
            }
            _ => return,
        }
    }
    // the whole object copied, word by word, to one place
    copied.sort_by_key(|c| c.1);
    let Some(&(d0, s0, _)) = copied.first() else { return };
    let mut next = 0i32;
    for &(d, so, n) in &copied {
        if so != next || d - so != d0 - s0 || n == 0 {
            return;
        }
        next += n as i32;
    }
    if next as u32 != tsize || s0 != 0 {
        return;
    }
    // the returned class's constructor taking that class (or its template parameter)
    let base = strip_template_args(&rcls);
    let last = crate::sig::split_scope(&base).1.to_string();
    let key = format!("{base}::{last}");
    let takes = |d: &mwdec_core::DeclInfo| -> bool {
        if d.params.len() != 1 || !d.is_inline_defined || d.access != mwdec_core::Access::Public {
            return false;
        }
        let pt = match strip_cv(&d.params[0].ty) {
            Type::Ref(x) => strip_cv(x).clone(),
            x => x.clone(),
        };
        pt == t || matches!(&pt, Type::Named(n) if d.template_params.contains(n))
    };
    let Some(_) = db.decls.get(&key).and_then(|ds| ds.iter().find(|d| takes(d))) else { return };
    let ctor = mwdec_core::FuncSig {
        qualified_name: key,
        mangled: None,
        ret: Type::Void,
        params: vec![mwdec_core::Param { name: None, ty: Type::Ref(Box::new(Type::Const(Box::new(t.clone())))) }],
        this_class: Some(rcls.clone()),
        is_const: false,
        is_static: false,
        is_virtual: false,
        variadic: false,
        runs_code: false,
    };
    let ret = Stmt::Return(Some(Expr::Construct { class: rt, ctor: Some(ctor), args: vec![call] }));
    ir.body.truncate(si);
    ir.body.push(ret);
}

/// `rstl::optional_object<CAABox>` -> `rstl::optional_object` (decl keys carry no arguments).
fn strip_template_args(s: &str) -> String {
    let mut out = String::new();
    let mut d = 0;
    for c in s.chars() {
        match c {
            '<' => d += 1,
            '>' => d -= 1,
            _ if d == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Can an object of class `t` be copied where the draft writes `T(x)` or passes it by value: no
/// copy constructor declared non-public (an implicit one is public).
pub fn copyable(db: &mwdec_core::TypeDb, t: &Type) -> bool {
    let Some(cls) = named(strip_cv(t)) else { return false };
    let base = strip_template_args(cls);
    let last = crate::sig::split_scope(&base).1.to_string();
    let key = format!("{base}::{last}");
    let copy_ctor = |d: &&mwdec_core::DeclInfo| {
        d.params.len() == 1 && matches!(strip_cv(&d.params[0].ty), Type::Ref(x) if named(strip_cv(x)).is_some_and(|n| strip_template_args(n) == base))
    };
    !db.decls.get(&key).is_some_and(|ds| ds.iter().filter(copy_ctor).any(|d| d.access != mwdec_core::Access::Public))
}

/// A frame object declared as a class with an inline default constructor whose initializer list
/// sets members to constants (`reserved_vector() : mCount(0) {}`): the constant member stores
/// before its first use are that construction, implicit in the declaration.
pub fn drop_default_construction_stores(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let vars = ir.vars.clone();
    for (v, var) in vars.iter().enumerate() {
        if !matches!(var.kind, VarKind::Stack { .. }) {
            continue;
        }
        let Some(cls) = named(&var.ty).map(|s| s.to_string()) else { continue };
        let Some(c) = crate::sig::find_class(db, &cls) else { continue };
        let base = strip_template_args(&cls);
        let last = crate::sig::split_scope(&base).1.to_string();
        let Some(init) = db.decls.get(&format!("{base}::{last}")).and_then(|ds| ds.iter().find(|d| d.params.is_empty() && d.is_inline_defined).and_then(|d| d.init_list.clone())) else { continue };
        // member offset -> constant literal
        let mut consts: Vec<(i32, String)> = vec![];
        for part in crate::sig::split_top(&init, ',') {
            let toks: Vec<&str> = part.split_whitespace().collect();
            if toks.len() < 4 || toks[1] != "(" || toks.last() != Some(&")") {
                continue;
            }
            if let Some(f) = c.fields.iter().find(|f| f.name == toks[0]) {
                consts.push((f.offset as i32, toks[2..toks.len() - 1].join(" ")));
            }
        }
        if consts.is_empty() {
            continue;
        }
        // the leading statements of the top-level body that mention v
        let mut k = 0;
        while k < ir.body.len() {
            let mut mentions = false;
            Stmt::walk_exprs(std::slice::from_ref(&ir.body[k]), &mut |e| {
                if matches!(e, Expr::Var(w) if *w == v) {
                    mentions = true;
                }
            });
            if !mentions {
                k += 1;
                continue;
            }
            let is_ctor_store = match &ir.body[k] {
                Stmt::Assign { dst: Expr::Member { base, offset, .. }, src } if matches!(**base, Expr::Var(w) if w == v) => {
                    consts.iter().any(|(o, lit)| o == offset && literal_value_is(lit, src))
                }
                _ => false,
            };
            if !is_ctor_store {
                break;
            }
            ir.body.remove(k);
        }
    }
}

fn literal_value_is(lit: &str, e: &Expr) -> bool {
    let t: String = lit.split_whitespace().collect();
    match e {
        Expr::Int { value, .. } => t.parse::<i64>().ok() == Some(*value) || (t == "false" && *value == 0) || (t == "true" && *value == 1) || ((t == "nullptr" || t == "NULL") && *value == 0),
        Expr::Float { bits, double } => t.trim_end_matches(|c| c == 'f' || c == 'F').parse::<f64>().ok().is_some_and(|x| if *double { x.to_bits() == *bits } else { (x as f32).to_bits() as u64 == *bits }),
        _ => false,
    }
}

/// Constant stores into the returned object right before `return;` that are exactly what an
/// inline constructor callable without arguments sets (`vector(const Alloc& a = Alloc()) :
/// mAllocator(a), mCount(0), mCapacity(0), mItems(nullptr)`): `return R();`.
pub fn fold_default_constructed_return(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let Some(rv) = ir.vars.iter().position(|v| v.kind == VarKind::StructRet) else { return };
    let rt = strip_cv(&ir.sig.ret).clone();
    let resolved = crate::types::resolve(Some(db), &rt).into_owned();
    let Some(rcls) = named(strip_cv(&resolved)).map(|s| s.to_string()) else { return };
    let Some(c) = crate::sig::find_class(db, &rcls) else { return };
    let base = strip_template_args(&rcls);
    let last = crate::sig::split_scope(&base).1.to_string();
    let Some(decls) = db.decls.get(&format!("{base}::{last}")) else { return };
    // (member offset, constant) of the first argument-free inline constructor with constants
    let mut consts: Option<Vec<(i32, String)>> = None;
    'd: for d in decls {
        let callable = d.params.is_empty() || (d.defaults.len() == d.params.len() && d.defaults.iter().all(|x| x.is_some()));
        if !callable || !d.is_inline_defined || d.access != mwdec_core::Access::Public {
            continue;
        }
        let Some(init) = &d.init_list else { continue };
        let names: Vec<String> = d.params.iter().map(|p| p.name.clone().unwrap_or_default()).collect();
        let mut cs = vec![];
        for part in crate::sig::split_top(init, ',') {
            let toks: Vec<&str> = part.split_whitespace().collect();
            if toks.len() < 4 || toks[1] != "(" || toks.last() != Some(&")") {
                continue 'd;
            }
            let Some(f) = c.fields.iter().find(|f| f.name == toks[0]) else { continue 'd };
            let inner = toks[2..toks.len() - 1].join(" ");
            if names.iter().any(|n| *n == inner) {
                // a parameter (its default) into an empty member: no store
                if crate::types::size_of(Some(db), &f.ty).unwrap_or(0) > 1 || crate::types::is_aggregate(Some(db), &f.ty) && !c.fields.is_empty() && crate::types::size_of(Some(db), &f.ty) != Some(1) {
                    continue 'd;
                }
                continue;
            }
            cs.push((f.offset as i32, inner));
        }
        if !cs.is_empty() {
            consts = Some(cs);
            break;
        }
    }
    let Some(consts) = consts else { return };
    let ctor = mwdec_core::FuncSig {
        qualified_name: format!("{base}::{last}"),
        mangled: None,
        ret: Type::Void,
        params: vec![],
        this_class: Some(rcls.clone()),
        is_const: false,
        is_static: false,
        is_virtual: false,
        variadic: false,
        runs_code: false,
    };
    let matches_all = |stores: &[Stmt]| -> bool {
        let mut seen: Vec<i32> = vec![];
        for st in stores {
            match st {
                Stmt::Assign { dst: Expr::Load { base, offset, .. }, src } if matches!(**base, Expr::Var(w) if w == rv) && consts.iter().any(|(o, lit)| o == offset && literal_value_is(lit, src)) => seen.push(*offset),
                _ => return false,
            }
        }
        seen.sort_unstable();
        seen.dedup();
        seen.len() == consts.len() && stores.len() == seen.len()
    };
    // an `if` arm made of exactly those stores, the function returning right after the `if`
    let n = ir.body.len();
    if n >= 2 && matches!(ir.body[n - 1], Stmt::Return(None)) {
        if let Stmt::If { then, els, .. } = &mut ir.body[n - 2] {
            // one arm the constants, the other constructing the returned object in place
            // (`__return->R(args)`): both become returns; anything else is left alone
            let in_place = |arm: &[Stmt]| -> Option<Expr> {
                if let [Stmt::Expr(Expr::Call { callee: Callee::Method { sig: cs, this, .. }, args, .. })] = arm {
                    if crate::sig::is_ctor(cs) && matches!(**this, Expr::Var(w) if w == rv) {
                        return Some(Expr::Construct { class: rt.clone(), ctor: Some(cs.clone()), args: args.clone() });
                    }
                }
                None
            };
            let dflt = Stmt::Return(Some(Expr::Construct { class: rt.clone(), ctor: Some(ctor.clone()), args: vec![] }));
            if !then.is_empty() && matches_all(then) {
                if let Some(c2) = in_place(els) {
                    *then = vec![dflt];
                    *els = vec![Stmt::Return(Some(c2))];
                }
            } else if !els.is_empty() && matches_all(els) {
                if let Some(c2) = in_place(then) {
                    *els = vec![dflt];
                    *then = vec![Stmt::Return(Some(c2))];
                }
            }
        }
        // both arms return now: the trailing `return;` is unreachable
        if matches!(&ir.body[n - 2], Stmt::If { then, els, .. } if matches!(then.last(), Some(Stmt::Return(Some(_)))) && matches!(els.last(), Some(Stmt::Return(Some(_))))) {
            ir.body.pop();
        }
    }
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let Some(r) = b.iter().position(|s| matches!(s, Stmt::Return(None))) else { return };
        let mut k = r;
        let mut seen: Vec<i32> = vec![];
        while k > 0 {
            match &b[k - 1] {
                Stmt::Assign { dst: Expr::Load { base, offset, .. }, src } if matches!(**base, Expr::Var(w) if w == rv) && consts.iter().any(|(o, lit)| o == offset && literal_value_is(lit, src)) => {
                    seen.push(*offset);
                    k -= 1;
                }
                _ => break,
            }
        }
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != consts.len() || r - k != seen.len() {
            return;
        }
        b.splice(k..=r, std::iter::once(Stmt::Return(Some(Expr::Construct { class: rt.clone(), ctor: Some(ctor.clone()), args: vec![] }))));
    });
}
