//! Array element accesses. MWCC computes `p->arr[i].f` as `slwi t,i,s ; add t,p,t ; lwz x,(A+f)(t)`
//! (element address first, member offset in the displacement). The lifter sees a byte offset
//! `(u8*)p + (i << s)` accessed at `A + f`; with the pointee's array field at `A` (element size
//! `1 << s`) this becomes `p->arr[i].f`, or `p[i].f` when `p` points to elements of that size.
//! Writing the raw sum instead lets the compiler fold `A` into the index (`addi ; lwzx`).

use crate::ir::*;
use crate::types;
use mwdec_core::{Type, TypeDb};
use std::collections::HashMap;

/// `x << s`, `x * n`, `(x << s) & M` (+ constant) -> (x, element size, constant byte offset).
fn split_index(e: &Expr, vars: &[Var], db: Option<&TypeDb>) -> Option<(Expr, u32, i64)> {
    match e {
        Expr::Binary { op: BinOp::Add, l, r, .. } => {
            if let Some(c) = r.as_int() {
                let (x, n, k) = split_index(l, vars, db)?;
                return Some((x, n, k + c));
            }
            if let Some(c) = l.as_int() {
                let (x, n, k) = split_index(r, vars, db)?;
                return Some((x, n, k + c));
            }
            None
        }
        Expr::Binary { op: BinOp::Shl, l, r, .. } => {
            let s = r.as_int()?;
            if !(1..=12).contains(&s) {
                return None;
            }
            Some(((**l).clone(), 1u32 << s, 0))
        }
        Expr::Binary { op: BinOp::Mul, l, r, .. } => {
            let (x, n) = match (l.as_int(), r.as_int()) {
                (_, Some(n)) => ((**l).clone(), n),
                (Some(n), _) => ((**r).clone(), n),
                _ => return None,
            };
            if n < 2 || n > 0x10000 {
                return None;
            }
            Some((x, n as u32, 0))
        }
        Expr::Binary { op: BinOp::And, l, r, ty } => {
            let m = r.as_int()? as u32;
            let Expr::Binary { op: BinOp::Shl, l: x, r: s, .. } = &**l else { return None };
            let s = s.as_int()? as u32;
            if !(1..=12).contains(&s) || m & ((1 << s) - 1) != 0 {
                return None;
            }
            // `(u8)x << s` masks with the type's own range: the plain index
            let xt = types::resolve(db, &types::ty_of(x, vars)).into_owned();
            let own = match strip_cv(&xt) {
                Type::Int { size: 1, signed: false } | Type::Bool => Some(0xffu32),
                Type::Int { size: 2, signed: false } => Some(0xffff),
                _ => None,
            };
            if own == Some(m >> s) {
                return Some(((**x).clone(), 1u32 << s, 0));
            }
            let inner = Expr::bin(BinOp::And, (**x).clone(), Expr::uint((m >> s) as i64), ty.clone());
            Some((inner, 1u32 << s, 0))
        }
        _ => None,
    }
}

/// Array field of `cls` whose element size is `esz` containing byte `off`: (array offset,
/// element type, element count). Nested aggregates and bases are searched (offset relative to
/// `cls`).
fn array_field(db: &TypeDb, cls: &str, off: i32, esz: u32, depth: u32) -> Option<(i32, Type, u32)> {
    if depth > 8 {
        return None;
    }
    let c = crate::sig::find_class(db, cls)?;
    for f in &c.fields {
        if f.bitfield.is_some() {
            continue;
        }
        let fo = f.offset as i32;
        let rt = types::resolve(Some(db), &f.ty).into_owned();
        let fs = types::size_of(Some(db), &rt).unwrap_or(0) as i32;
        if off < fo || off >= fo + fs.max(1) {
            continue;
        }
        match strip_cv(&rt) {
            Type::Array(e, n) => {
                if types::size_of(Some(db), e) == Some(esz) {
                    return Some((fo, (**e).clone(), *n));
                }
            }
            Type::Named(n) => {
                if let Some((a, t, k)) = array_field(db, n, off - fo, esz, depth + 1) {
                    return Some((fo + a, t, k));
                }
            }
            _ => {}
        }
    }
    for b in &c.bases {
        let bo = b.offset as i32;
        if let Some((a, t, k)) = array_field(db, &b.name, off - bo, esz, depth + 1) {
            return Some((bo + a, t, k));
        }
    }
    None
}

fn add_index(x: Expr, j: i64) -> Expr {
    if j == 0 {
        x
    } else {
        Expr::bin(BinOp::Add, x, Expr::int(j), t_s32())
    }
}

/// The element lvalue plus the access at byte `inner` of it with type `ty` (the DB's member
/// type when it agrees with the access width).
fn finish(elem: Expr, et: &Type, inner: i32, ty: &Type, db: Option<&TypeDb>) -> Option<Expr> {
    let es = types::size_of(db, et)?;
    if inner == 0 && matches!(ty, Type::Unknown { size: 0 }) {
        return Some(elem);
    }
    if !types::is_aggregate(db, et) {
        // scalar element: the access must be the whole element
        let ts = scalar_size(ty)?;
        if inner != 0 || ts != es || is_float(et) != is_float(ty) {
            return None;
        }
        return Some(elem);
    }
    if inner < 0 || inner as u32 >= es {
        return None;
    }
    let mut ty = ty.clone();
    if let (Some(db), Some(cls)) = (db, named(&types::resolve(db, et)).map(|s| s.to_string())) {
        if let Some((_, ft)) = types::field_path(db, &cls, inner, scalar_size(&ty).unwrap_or(0)) {
            if crate::translate::compatible_scalar(Some(db), &ft, &ty) {
                ty = ft;
            }
        }
    } else {
        return None;
    }
    Some(Expr::Member { base: Box::new(elem), offset: inner, ty })
}

/// `(u8*)p + idx` (byte_add form) or `(int)p + idx`: (p, idx).
fn byte_sum(base: &Expr, vars: &[Var]) -> Option<(Expr, Expr)> {
    match base {
        Expr::AddrOf(a) => match &**a {
            Expr::Index { base: cp, index, ty: bt } if scalar_size(bt) == Some(1) => match &**cp {
                Expr::Cast { e: p, .. } => Some(((**p).clone(), (**index).clone())),
                p if pointee(&types::ty_of(p, vars)).map_or(false, |t| scalar_size(t) == Some(1)) => Some((p.clone(), (**index).clone())),
                _ => None,
            },
            _ => None,
        },
        Expr::Binary { op: BinOp::Add, l, r, .. } => match &**l {
            Expr::Cast { e: p, ty } if !is_ptr(ty) && (is_ptr(&types::ty_of(p, vars)) || matches!(**p, Expr::AddrOf(_))) => Some(((**p).clone(), (**r).clone())),
            // `(u8*)p + idx` (indexed loads/stores)
            Expr::Cast { e: p, ty } if pointee(ty).map_or(false, |t| scalar_size(t) == Some(1)) && (is_ptr(&types::ty_of(p, vars)) || matches!(**p, Expr::AddrOf(_))) => {
                Some(((**p).clone(), (**r).clone()))
            }
            p if pointee(&types::ty_of(p, vars)).map_or(false, |t| scalar_size(t) == Some(1)) => Some((p.clone(), (**r).clone())),
            _ => None,
        },
        _ => None,
    }
}

fn try_array(e: &Expr, vars: &[Var], db: Option<&TypeDb>, defs: &HashMap<VarId, Expr>) -> Option<Expr> {
    let Expr::Load { base, offset, ty } = e else { return None };
    let (p, index) = byte_sum(base, vars)?;
    let index = &index;
    let p = &Box::new(p);
    // a byte offset computed once into a temp (shared by a load and a store)
    let index = match index {
        Expr::Var(v) if defs.contains_key(v) => &defs[v],
        i => i,
    };
    let (x, esz, k) = split_index(index, vars, db)?;
    let off = *offset as i64 + k;
    if off < 0 || off > 0x100000 {
        return None;
    }
    let off = off as i32;
    // the object the offset is relative to: *p (pointer) or the lvalue whose address p is
    let (obj, ptr, ot) = match &**p {
        // the address of an element itself (multi-dimensional arrays)
        Expr::AddrOf(_) if byte_sum(p, vars).is_some() => {
            let lv = try_array(&Expr::Load { base: p.clone(), offset: 0, ty: t_unk(0) }, vars, db, defs)?;
            let t = types::ty_of(&lv, vars);
            (lv, false, t)
        }
        Expr::AddrOf(lv) => ((**lv).clone(), false, types::ty_of(lv, vars)),
        other => {
            let pt = types::ty_of(other, vars);
            let t = pointee(&pt)?.clone();
            (other.clone(), true, t)
        }
    };
    let otr = types::resolve(db, &ot).into_owned();
    // array member of a class
    if let (Some(db), Some(cls)) = (db, named(&otr)) {
        if let Some((a, et, n)) = array_field(db, cls, off, esz, 0) {
            let rel = off - a;
            let j = (rel as u32 / esz) as i64;
            let inner = rel - (j as i32) * esz as i32;
            let at = Type::Array(Box::new(et.clone()), n);
            let arr = if ptr {
                Expr::Load { base: Box::new(obj), offset: a, ty: at }
            } else {
                Expr::Member { base: Box::new(obj), offset: a, ty: at }
            };
            let elem = Expr::Index { base: Box::new(arr), index: Box::new(add_index(x, j)), ty: et.clone() };
            return finish(elem, &et, inner, ty, Some(db));
        }
    }
    // an array object itself (global/stack array lvalue)
    if let Type::Array(et, _) = strip_cv(&otr) {
        if !ptr && types::size_of(db, et) == Some(esz) {
            let j = (off as u32 / esz) as i64;
            let inner = off - (j as i32) * esz as i32;
            let elem = Expr::Index { base: Box::new(obj), index: Box::new(add_index(x, j)), ty: (**et).clone() };
            return finish(elem, et, inner, ty, db);
        }
    }
    // pointer arithmetic over elements of the pointee's size
    if ptr && esz > 1 && !matches!(strip_cv(&otr), Type::Void | Type::Unknown { .. }) && types::size_of(db, &otr) == Some(esz) {
        let j = (off as u32 / esz) as i64;
        let inner = off - (j as i32) * esz as i32;
        let elem = Expr::Index { base: Box::new(obj), index: Box::new(add_index(x, j)), ty: ot.clone() };
        return finish(elem, &ot, inner, ty, db);
    }
    None
}

/// Rewrite byte-offset element accesses into array indexing everywhere in `body`: accesses
/// first, then remaining element addresses (`&p->arr[i]`) unless they feed byte arithmetic.
pub fn recover(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>) {
    // single-assignment register locals holding a scaled index
    let mut defs: HashMap<VarId, Expr> = HashMap::new();
    {
        let mut n: HashMap<VarId, usize> = HashMap::new();
        Stmt::for_each_block_mut(body, &mut |b| {
            for s in b.iter() {
                if let Stmt::Assign { dst: Expr::Var(v), src } = s {
                    if matches!(vars[*v].kind, VarKind::Local) {
                        *n.entry(*v).or_default() += 1;
                        if split_index(src, vars, db).is_some() && !src.has_call() {
                            defs.insert(*v, src.clone());
                        }
                    }
                }
            }
        });
        defs.retain(|v, _| n.get(v) == Some(&1));
    }
    recover_with(body, vars, db, &defs);
    if defs.is_empty() {
        return;
    }
    // index temps no longer read
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            *uses.entry(*v).or_default() += 1;
        }
    });
    Stmt::for_each_block_mut(body, &mut |b| {
        b.retain(|s| !matches!(s, Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) && uses.get(v) == Some(&1)));
    });
}

fn recover_with(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>, defs: &HashMap<VarId, Expr>) {
    let defs = defs.clone();
    Stmt::rewrite_exprs(body, &mut |e| {
        if let Some(n) = try_array(e, vars, db, &defs) {
            *e = n;
        }
    });
    let mut protected = vec![];
    Stmt::walk_exprs(body, &mut |e| match e {
        Expr::Cast { e, .. } | Expr::Index { base: e, .. } if matches!(**e, Expr::AddrOf(_)) => protected.push((**e).clone()),
        Expr::Binary { l, r, .. } => {
            for x in [l, r] {
                if matches!(**x, Expr::AddrOf(_)) {
                    protected.push((**x).clone());
                }
            }
        }
        _ => {}
    });
    Stmt::rewrite_exprs(body, &mut |e| {
        if matches!(e, Expr::AddrOf(a) if matches!(**a, Expr::Index { .. })) && !protected.contains(e) {
            let as_load = Expr::Load { base: Box::new(e.clone()), offset: 0, ty: t_unk(0) };
            if let Some(n) = try_array(&as_load, vars, db, &defs) {
                *e = Expr::AddrOf(Box::new(n));
            }
        }
    });
}
