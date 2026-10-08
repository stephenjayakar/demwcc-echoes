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

/// Element type of an indexable class template instance: the class declares `operator[]`
/// returning a reference to one of its template parameters (`T& operator[](int)`), whose
/// argument in `cls` is the element type.
fn container_elem(db: &TypeDb, cls: &str) -> Option<Type> {
    let lt = cls.find('<')?;
    let base = &cls[..lt];
    let ds = db.decls.get(&format!("{base}::operator[]"))?;
    let args = crate::sig::split_top(&cls[lt + 1..cls.len().checked_sub(1)?], ',');
    for d in ds {
        let Type::Ref(inner) = strip_cv(&d.ret) else { continue };
        let Some(p) = named(inner) else { continue };
        if let Some(k) = d.template_params.iter().position(|x| x == p) {
            return Some(crate::sig::parse_type(args.get(k)?));
        }
    }
    None
}

/// A member container (`rstl::reserved_vector<T, N>`: indexable, elements stored inline in an
/// array member) of `cls` holding byte `off`, with elements of size `esz`: (container offset,
/// container type, offset of element 0, element type), offsets relative to `cls`.
fn container_field(db: &TypeDb, cls: &str, off: i32, esz: u32, depth: u32) -> Option<(i32, Type, i32, Type)> {
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
        let Type::Named(n) = strip_cv(&rt) else { continue };
        if let Some(et) = container_elem(db, n) {
            if types::size_of(Some(db), &et) == Some(esz) {
                // the inline storage: an array member covering the offset
                if let Some(cc) = crate::sig::find_class(db, n) {
                    for g in &cc.fields {
                        let go = fo + g.offset as i32;
                        let gt = types::resolve(Some(db), &g.ty).into_owned();
                        let gs = types::size_of(Some(db), &gt).unwrap_or(0) as i32;
                        if matches!(strip_cv(&gt), Type::Array(..)) && gs > 0 && gs as u32 % esz == 0 && off >= go && off < go + gs {
                            return Some((fo, f.ty.clone(), go, et));
                        }
                    }
                }
            }
            continue;
        }
        if let Some((a, t, d, e)) = container_field(db, n, off - fo, esz, depth + 1) {
            return Some((fo + a, t, fo + d, e));
        }
    }
    for b in &c.bases {
        let bo = b.offset as i32;
        if let Some((a, t, d, e)) = container_field(db, &b.name, off - bo, esz, depth + 1) {
            return Some((bo + a, t, bo + d, e));
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
    if let Type::Array(e2, _) = strip_cv(et) {
        // an element that is itself an array (`g[i][k]`): the access is one of its elements
        let s2 = types::size_of(db, e2)?;
        let ts = scalar_size(ty)?;
        if s2 == 0 || inner < 0 || inner as u32 % s2 != 0 || ts != s2 || is_float(e2) != is_float(ty) || types::is_aggregate(db, e2) {
            return None;
        }
        return Some(Expr::Index { base: Box::new(elem), index: Box::new(Expr::int((inner as u32 / s2) as i64)), ty: (**e2).clone() });
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
    let (x, esz, k) = match split_index(index, vars, db) {
        Some(r) => r,
        // a byte element indexed by a plain integer (`p->mFlags[i]`)
        None if scalar_size(ty) == Some(1) && !is_ptr(&types::ty_of(index, vars)) && index.as_int().is_none() => (index.clone(), 1, 0),
        None => return two_level(p, index, *offset, ty, vars, db, defs),
    };
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
    // element of a member container (`p->mList[i].f` through its `operator[]`)
    if let (Some(db), Some(cls)) = (db, named(&otr)) {
        if let Some((co, ct, d0, et)) = container_field(db, cls, off, esz, 0) {
            let rel = off - d0;
            let j = (rel as u32 / esz) as i64;
            let inner = rel - (j as i32) * esz as i32;
            let cont = if ptr {
                Expr::Load { base: Box::new(obj.clone()), offset: co, ty: ct }
            } else {
                Expr::Member { base: Box::new(obj.clone()), offset: co, ty: ct }
            };
            let elem = Expr::Index { base: Box::new(cont), index: Box::new(add_index(x.clone(), j)), ty: et.clone() };
            if let Some(r) = finish(elem, &et, inner, ty, Some(db)) {
                return Some(r);
            }
        }
    }
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
    // array member with elements smaller than the index scale: `&p->a[i * 2]` (the scale is
    // a multiple of the element size; the index keeps the factor)
    if let (Some(db), Some(cls)) = (db, named(&otr)) {
        for f in [2u32, 4, 8] {
            if esz % f != 0 || esz / f < 1 {
                continue;
            }
            let e = esz / f;
            if let Some((a, et, n)) = array_field(db, cls, off, e, 0) {
                let rel = off - a;
                let j = (rel as u32 / e) as i64;
                let inner = rel - (j as i32) * e as i32;
                let at = Type::Array(Box::new(et.clone()), n);
                let arr = if ptr {
                    Expr::Load { base: Box::new(obj), offset: a, ty: at }
                } else {
                    Expr::Member { base: Box::new(obj), offset: a, ty: at }
                };
                let scaled = Expr::bin(BinOp::Mul, x, Expr::int(f as i64), Type::Int { size: 4, signed: true });
                let elem = Expr::Index { base: Box::new(arr), index: Box::new(add_index(scaled, j)), ty: et.clone() };
                return finish(elem, &et, inner, ty, Some(db));
            }
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

/// `p + (i * S + j * s + k)` with two scaled indices (S > s): an element of an array inside an
/// element of an outer array (`p->a[i].b[j]`), resolved one level at a time.
fn two_level(p: &Expr, index: &Expr, offset: i32, ty: &Type, vars: &[Var], db: Option<&TypeDb>, defs: &HashMap<VarId, Expr>) -> Option<Expr> {
    fn terms(e: &Expr, out: &mut Vec<Expr>, k: &mut i64) {
        match e {
            Expr::Binary { op: BinOp::Add, l, r, .. } => {
                terms(l, out, k);
                terms(r, out, k);
            }
            e => match e.as_int() {
                Some(c) => *k += c,
                None => out.push(e.clone()),
            },
        }
    }
    let (mut ts, mut k) = (vec![], 0i64);
    terms(index, &mut ts, &mut k);
    if ts.len() != 2 {
        return None;
    }
    let (a, b) = (split_index(&ts[0], vars, db)?, split_index(&ts[1], vars, db)?);
    if a.2 != 0 || b.2 != 0 || a.1 == b.1 {
        return None;
    }
    let (outer, inner_term) = if a.1 > b.1 { (&ts[0], &ts[1]) } else { (&ts[1], &ts[0]) };
    let byte = |base: Expr, idx: Expr| Expr::AddrOf(Box::new(Expr::Index { base: Box::new(Expr::cast(t_ptr(t_int(1, false)), base)), index: Box::new(idx), ty: t_int(1, false) }));
    let off = offset as i64 + k;
    if !(0..=0x100000).contains(&off) {
        return None;
    }
    let first = try_array(&Expr::Load { base: Box::new(byte(p.clone(), outer.clone())), offset: off as i32, ty: t_unk(0) }, vars, db, defs)?;
    let (elem, rest) = match first {
        Expr::Member { base, offset, ty: Type::Unknown { size: 0 } } if matches!(*base, Expr::Index { .. }) => (*base, offset),
        e @ Expr::Index { .. } => (e, 0),
        _ => return None,
    };
    try_array(&Expr::Load { base: Box::new(byte(Expr::AddrOf(Box::new(elem)), inner_term.clone())), offset: rest, ty: ty.clone() }, vars, db, defs)
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
        // an index computed before one of its variables is reassigned (`t = i * 36; i++;
        // a[t]`) can't be re-read at its use
        let mut stale: Vec<VarId> = vec![];
        Stmt::for_each_block_mut(body, &mut |b| {
            for (k, s) in b.iter().enumerate() {
                let Stmt::Assign { dst: Expr::Var(t), src } = s else { continue };
                if !defs.contains_key(t) {
                    continue;
                }
                let reassigned = b[k + 1..].iter().any(|x| {
                    let mut hit = false;
                    fn walk(x: &Stmt, src: &Expr, hit: &mut bool) {
                        match x {
                            Stmt::Assign { dst: Expr::Var(w), .. } if src.uses_var(*w) => *hit = true,
                            Stmt::If { then, els, .. } => {
                                then.iter().for_each(|y| walk(y, src, hit));
                                els.iter().for_each(|y| walk(y, src, hit));
                            }
                            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => body.iter().for_each(|y| walk(y, src, hit)),
                            _ => {}
                        }
                    }
                    walk(x, src, &mut hit);
                    if let Stmt::Expr(e) | Stmt::Assign { src: e, .. } = x {
                        e.walk(&mut |y| if let Expr::IncDec { e: z, .. } = y { if let Expr::Var(w) = &**z { if src.uses_var(*w) { hit = true; } } });
                    }
                    hit
                });
                if reassigned {
                    stale.push(*t);
                }
            }
        });
        for t in stale {
            defs.remove(&t);
        }
    }
    forward_address_temps(body, vars, db, &defs);
    type_indexed_globals(body, vars, db, &defs);
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

/// An element address computed once into a temp and used as the base of the next statement's
/// accesses (`t = (u8*)&a[i] + (j << 2); *(t + 0x24) |= 64`): forward it into each access when
/// every one of them then becomes an array element (`a[i].f[j] |= 64`; MWCC computes the shared
/// address once either way).
fn forward_address_temps(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>, defs: &HashMap<VarId, Expr>) {
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    let mut ndefs: HashMap<VarId, usize> = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(v), .. } = s {
                *ndefs.entry(*v).or_default() += 1;
            }
        }
    });
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            let (t, def) = match &b[i] {
                Stmt::Assign { dst: Expr::Var(t), src } if matches!(vars[*t].kind, VarKind::Local) && ndefs.get(t) == Some(&1) && byte_sum(src, vars).is_some() && !src.has_call() => (*t, src.clone()),
                _ => {
                    i += 1;
                    continue;
                }
            };
            let total = uses.get(&t).copied().unwrap_or(0);
            // uses in the following statements up to the first call (a register word read,
            // updated through other stores and written back)
            let mut def_vars = vec![];
            let mut def_globals = vec![];
            def.walk(&mut |e| match e {
                Expr::Var(v) => def_vars.push(*v),
                Expr::Global { symbol, .. } => def_globals.push(symbol.clone()),
                _ => {}
            });
            // (a pointer the source kept, `u32* p = &a[i]`, is dereferenced at offset 0: its uses
            // must follow right away; a shared partial address may reach further)
            let mut at_zero = false;
            Stmt::walk_exprs(&b[i + 1..(i + 12).min(b.len())], &mut |e| {
                if let Expr::Load { base, offset: 0, .. } = e {
                    at_zero |= matches!(**base, Expr::Var(v) if v == t);
                }
            });
            let reach = if at_zero { 3 } else { 12 };
            let mut span = 0;
            let mut seen = 0;
            for j in i + 1..(i + reach).min(b.len()) {
                if !matches!(b[j], Stmt::Assign { .. } | Stmt::Expr(_)) {
                    break;
                }
                // (the address's inputs must stay unchanged up to each use)
                if j > i + 1 {
                    if let Stmt::Assign { dst, .. } = &b[j - 1] {
                        let clobbers = match dst {
                            Expr::Var(v) => def_vars.contains(v),
                            Expr::Global { symbol, .. } => def_globals.contains(symbol),
                            _ => false,
                        };
                        if clobbers {
                            break;
                        }
                    }
                }
                let mut n = HashMap::new();
                crate::inline::count_uses(std::slice::from_ref(&b[j]), &mut n);
                seen += n.get(&t).copied().unwrap_or(0);
                span = j;
                if seen >= total || stmt_has_call(&b[j]) {
                    break;
                }
            }
            // (beyond the next two statements only across stores of values computed without
            // reading memory: field inserts, constants)
            let reads_memory = |s: &Stmt| -> bool {
                let src = match s {
                    Stmt::Assign { src, .. } => src,
                    Stmt::Expr(e) => e,
                    _ => return true,
                };
                let mut m = false;
                src.walk(&mut |e| {
                    m |= matches!(e, Expr::Load { .. } | Expr::Member { .. } | Expr::Index { .. } | Expr::Global { .. } | Expr::BitField { .. } | Expr::New { .. } | Expr::IncDec { .. })
                        || (matches!(e, Expr::Call { .. }) && !e.is_pure_call());
                });
                m
            };
            if total == 0 || seen != total || (span > i + 2 && (b[i + 1..=span].iter().any(stmt_has_call) || b[i + 2..span].iter().any(|s| reads_memory(s)))) {
                i += 1;
                continue;
            }
            let mut cand: Vec<Stmt> = b[i + 1..=span].to_vec();
            let mut ok = true;
            let mut converted = 0;
            // one location read and written back (`x = x | 64`): every access is the same
            let mut accesses: Vec<(i32, Type)> = vec![];
            Stmt::walk_exprs(&cand, &mut |e| {
                if let Expr::Load { base, offset, ty } = e {
                    if matches!(**base, Expr::Var(v) if v == t) && !accesses.contains(&(*offset, ty.clone())) {
                        accesses.push((*offset, ty.clone()));
                    }
                }
            });
            if accesses.len() != 1 {
                i += 1;
                continue;
            }
            Stmt::rewrite_exprs(&mut cand, &mut |e| {
                if let Expr::Load { base, offset, ty } = e {
                    if matches!(**base, Expr::Var(v) if v == t) {
                        let fwd = Expr::Load { base: Box::new(def.clone()), offset: *offset, ty: ty.clone() };
                        match try_array(&fwd, vars, db, defs) {
                            Some(n) => {
                                *e = n;
                                converted += 1;
                            }
                            None => ok = false,
                        }
                    }
                }
            });
            if ok && converted == total && !cand.iter().any(|s| stmt_uses(s, t)) {
                b.splice(i..=span, cand);
                continue;
            }
            i += 1;
        }
    });
}

fn stmt_has_call(s: &Stmt) -> bool {
    let mut c = false;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if matches!(e, Expr::Call { .. } | Expr::New { .. }) && !e.is_pure_call() {
            c = true;
        }
    });
    c
}

fn stmt_uses(s: &Stmt, v: VarId) -> bool {
    let mut n = HashMap::new();
    crate::inline::count_uses(std::slice::from_ref(s), &mut n);
    n.contains_key(&v)
}

/// Globals the context doesn't declare (unit statics, other units' tables) are typed by their
/// access width only. When every indexed access of one agrees on the element shape (`lhzx` of
/// `(i << 1)` -> `u16`, `lbz` at `+1` of `(i << 2)` -> `u8[4]`) and its plain accesses fit it, it
/// becomes an array of that element (`static u16 sTable[N]`, size from the symbol), so the
/// accesses index it and the declaration has the right size class (no small-data for big
/// objects).
pub(crate) fn type_indexed_globals(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>, defs: &HashMap<VarId, Expr>) {
    #[derive(Default)]
    struct Acc {
        // (element size, byte offset within the global, access type)
        indexed: Vec<(u32, i64, Type)>,
        plain: Vec<(i64, Type)>,
        bad: bool,
        size: u32,
    }
    fn global_of(p: &Expr) -> Option<(String, u32, i64)> {
        match p {
            Expr::AddrOf(g) => match &**g {
                Expr::Global { symbol, ty: Type::Unknown { size } } => Some((symbol.clone(), *size, 0)),
                Expr::Member { base, offset, ty: Type::Unknown { size: 0 } } => match &**base {
                    Expr::Global { symbol, ty: Type::Unknown { size } } => Some((symbol.clone(), *size, *offset as i64)),
                    _ => None,
                },
                _ => None,
            },
            Expr::Cast { e, .. } => global_of(e),
            _ => None,
        }
    }
    let declared = |sym: &str| db.map_or(false, |d| d.globals.contains_key(sym));
    let mut acc: HashMap<String, Acc> = HashMap::new();
    // indexed accesses first (their bases are skipped by the scan below)
    let mut handled: Vec<*const Expr> = vec![];
    Stmt::walk_exprs(body, &mut |e| {
        // an element-indexed untyped global (`g[i]` from scaled indexed loads)
        if let Expr::Index { base, ty, .. } = e {
            // (the address of the global: a bare global base is its pointer value; a cast one
            // is the byte arithmetic of an address computation)
            let g = match &**base {
                Expr::AddrOf(g) => Some(&**g),
                _ => None,
            };
            if let Some(g @ Expr::Global { symbol, ty: Type::Unknown { size } }) = g {
                if let Some(s) = scalar_size(ty).filter(|s| *s > 0) {
                    let a = acc.entry(symbol.clone()).or_default();
                    a.size = *size;
                    a.indexed.push((s, 0, ty.clone()));
                    handled.push(&**base as *const Expr);
                    handled.push(g as *const Expr);
                }
            }
            return;
        }
        let Expr::Load { base, offset, ty } = e else { return };
        let Some((p, idx)) = byte_sum(base, vars) else { return };
        let Some((sym, size, addend)) = global_of(&p) else { return };
        let idx = match &idx {
            Expr::Var(v) if defs.contains_key(v) => defs[v].clone(),
            i => i.clone(),
        };
        let a = acc.entry(sym).or_default();
        a.size = size;
        match split_index(&idx, vars, db) {
            Some((_, esz, k)) if scalar_size(ty).map_or(false, |s| s > 0) => a.indexed.push((esz, addend + *offset as i64 + k, ty.clone())),
            // a byte element indexed by a plain integer
            None if scalar_size(ty) == Some(1) && idx.as_int().is_none() && !is_ptr(&types::ty_of(&idx, vars)) => a.indexed.push((1, addend + *offset as i64, ty.clone())),
            _ => a.bad = true,
        }
        handled.push(&**base as *const Expr);
    });
    // an element address kept in a temp that is only dereferenced whole (`t = &g[i]; .. *t ..
    // *t`: the compiler's shared address of two `g[i]` reads)
    {
        let mut uses: HashMap<VarId, Vec<(i32, Type)>> = HashMap::new();
        let mut mentions: HashMap<VarId, usize> = HashMap::new();
        let mut ndefs: HashMap<VarId, usize> = HashMap::new();
        Stmt::walk_exprs(body, &mut |e| {
            if let Expr::Var(v) = e {
                *mentions.entry(*v).or_default() += 1;
            }
            if let Expr::Load { base, offset, ty } = e {
                if let Expr::Var(v) = **base {
                    uses.entry(v).or_default().push((*offset, ty.clone()));
                }
            }
        });
        let mut pairs: Vec<(&Expr, &Expr)> = vec![];
        assigns(body, &mut pairs);
        for (d, _) in &pairs {
            if let Expr::Var(v) = d {
                *ndefs.entry(*v).or_default() += 1;
            }
        }
        for (d, src) in &pairs {
            let Expr::Var(t) = d else { continue };
            if !matches!(vars[*t].kind, VarKind::Local) || ndefs.get(t) != Some(&1) {
                continue;
            }
            let Some((p, idx)) = byte_sum(src, vars) else { continue };
            let Some((sym, size, addend)) = global_of(&p) else { continue };
            let Some((_, esz, k)) = split_index(&idx, vars, db) else { continue };
            let us = uses.get(t).cloned().unwrap_or_default();
            // (the def itself mentions the temp once as its destination)
            if us.is_empty() || us.len() + 1 != mentions.get(t).copied().unwrap_or(0) || !us.iter().all(|(o, ty)| *o == 0 && scalar_size(ty) == Some(esz)) {
                continue;
            }
            let a = acc.entry(sym).or_default();
            a.size = size;
            for (_, ty) in &us {
                a.indexed.push((esz, addend + k, ty.clone()));
            }
            handled.push(*src as *const Expr);
        }
    }
    if acc.is_empty() {
        return;
    }
    // every other mention of those globals
    fn scan(e: &Expr, vars: &[Var], handled: &[*const Expr], acc: &mut HashMap<String, Acc>) {
        if handled.contains(&(e as *const Expr)) {
            return;
        }
        // an element address that escapes (into a temp, an argument): its uses are unknown
        if let Some((sym, ..)) = byte_sum(e, vars).and_then(|(p, _)| global_of(&p)) {
            if let Some(a) = acc.get_mut(&sym) {
                a.bad = true;
            }
            return;
        }
        match e {
            Expr::Global { symbol, .. } => {
                if let Some(a) = acc.get_mut(symbol) {
                    a.bad = true;
                }
            }
            Expr::Member { base, offset, ty } if matches!(**base, Expr::Global { .. }) => {
                let Expr::Global { symbol, .. } = &**base else { return };
                if let Some(a) = acc.get_mut(symbol) {
                    if matches!(ty, Type::Unknown { size: 0 }) {
                        a.bad = true;
                    } else {
                        a.plain.push((*offset as i64, ty.clone()));
                    }
                }
            }
            // the address itself (array decay, passed along)
            Expr::AddrOf(g) if matches!(**g, Expr::Global { .. }) => {}
            _ => {
                for k in direct_children(e) {
                    scan(k, vars, handled, acc);
                }
            }
        }
    }
    let mut roots: Vec<&Expr> = vec![];
    collect_roots(body, &mut roots);
    for r in roots {
        scan(r, vars, &handled, &mut acc);
    }
    let mut types_for: HashMap<String, Type> = HashMap::new();
    for (sym, a) in &acc {
        if a.bad || a.indexed.is_empty() || declared(sym) || sym.starts_with('@') || sym.starts_with("lbl_") {
            continue;
        }
        let (esz, _, t0) = a.indexed[0].clone();
        let s0 = scalar_size(&t0).unwrap_or(0);
        let same = |t: &Type| scalar_size(t) == Some(s0) && is_float(t) == is_float(&t0);
        if s0 == 0 || esz % s0 != 0 || !a.indexed.iter().all(|(e, o, t)| *e == esz && same(t) && *o >= 0 && *o % s0 as i64 == 0) {
            continue;
        }
        if !a.plain.iter().all(|(o, t)| same(t) && *o >= 0 && *o % s0 as i64 == 0) {
            continue;
        }
        let elem = if esz == s0 { t0.clone() } else { Type::Array(Box::new(t0.clone()), esz / s0) };
        let n = if a.size > 0 && a.size % esz == 0 {
            a.size / esz
        } else if a.size > 0 {
            continue;
        } else {
            0
        };
        types_for.insert(sym.clone(), Type::Array(Box::new(elem), n));
    }
    if types_for.is_empty() {
        return;
    }
    Stmt::rewrite_exprs(body, &mut |e| {
        if let Expr::Global { symbol, ty } = e {
            if let Some(t) = types_for.get(symbol) {
                *ty = t.clone();
            }
        }
        // `(&g)[i]` -> `g[i]` (the array itself; a cast base is byte arithmetic, left to the
        // element recovery)
        if let Expr::Index { base, .. } = e {
            let inner = match &**base {
                Expr::AddrOf(g) => Some((**g).clone()),
                _ => None,
            };
            if let Some(g @ Expr::Global { .. }) = inner {
                if let Expr::Global { symbol, .. } = &g {
                    if let Some(t) = types_for.get(symbol) {
                        **base = Expr::Global { symbol: symbol.clone(), ty: t.clone() };
                    }
                }
            }
        }
    });
    // plain accesses become constant-index elements
    Stmt::rewrite_exprs(body, &mut |e| {
        if let Expr::Member { base, offset, ty } = e {
            if let Expr::Global { ty: Type::Array(et, _), .. } = &**base {
                let et = (**et).clone();
                let es = types::size_of(db, &et).unwrap_or(0) as i32;
                if es > 0 {
                    let j = *offset / es;
                    let inner = *offset - j * es;
                    let elem = Expr::Index { base: base.clone(), index: Box::new(Expr::int(j as i64)), ty: et.clone() };
                    if let Some(n) = finish(elem, &et, inner, ty, db) {
                        *e = n;
                    }
                }
            }
        }
    });
}

fn direct_children(e: &Expr) -> Vec<&Expr> {
    let mut v: Vec<&Expr> = vec![];
    match e {
        Expr::AddrOf(x) | Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } | Expr::IncDec { e: x, .. } => v.push(x),
        Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => v.push(base),
        Expr::Index { base, index, .. } => {
            v.push(base);
            v.push(index);
        }
        Expr::Binary { l, r, .. } => {
            v.push(l);
            v.push(r);
        }
        Expr::Ternary { c, t, f, .. } => {
            v.push(c);
            v.push(t);
            v.push(f);
        }
        Expr::Call { callee, args, .. } => {
            match callee {
                Callee::Method { this, .. } | Callee::Virtual { this, .. } => v.push(this),
                Callee::Indirect(x) => v.push(x),
                Callee::Direct { .. } => {}
            }
            v.extend(args.iter());
        }
        Expr::New { placement, args, .. } => v.extend(placement.iter().chain(args.iter())),
        Expr::Construct { args, .. } => v.extend(args.iter()),
        _ => {}
    }
    v
}

fn collect_roots<'a>(b: &'a [Stmt], out: &mut Vec<&'a Expr>) {
    for s in b {
        match s {
            Stmt::Expr(e) | Stmt::Return(Some(e)) => out.push(e),
            Stmt::Assign { dst, src } => {
                out.push(dst);
                out.push(src);
            }
            Stmt::If { cond, then, els } => {
                out.push(cond);
                collect_roots(then, out);
                collect_roots(els, out);
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                out.push(cond);
                collect_roots(body, out);
            }
            Stmt::For { init, cond, step, body } => {
                collect_roots(init, out);
                out.push(cond);
                collect_roots(step, out);
                collect_roots(body, out);
            }
            Stmt::Switch { e, cases } => {
                out.push(e);
                for c in cases {
                    collect_roots(&c.body, out);
                }
            }
            _ => {}
        }
    }
}

/// The member container of `cls` whose inline storage holds byte `off`, whatever its element
/// size: (container offset, container type, offset of element 0, element type).
fn container_at(db: &TypeDb, cls: &str, off: i32, depth: u32) -> Option<(i32, Type, i32, Type)> {
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
        let Type::Named(n) = strip_cv(&rt) else { continue };
        if let Some(et) = container_elem(db, n) {
            let esz = types::size_of(Some(db), &et)?;
            if esz > 0 {
                return container_field(db, cls, off, esz, depth);
            }
            return None;
        }
        if let Some((a, t, d, e)) = container_at(db, n, off - fo, depth + 1) {
            return Some((fo + a, t, fo + d, e));
        }
    }
    for b in &c.bases {
        let bo = b.offset as i32;
        if let Some((a, t, d, e)) = container_at(db, &b.name, off - bo, depth + 1) {
            return Some((bo + a, t, bo + d, e));
        }
    }
    None
}

/// Constant-offset accesses into a member container's inline storage (`*(u8*)(this + 0x40)`
/// inside a `reserved_vector<S, 4> mList`) are element accesses: `this->mList[1].f`
/// (then bitfields and members resolve inside the element type).
pub fn container_members(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb) {
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter_mut() {
            match s {
                // writes go through the non-const operator[]: not for const containers
                Stmt::Assign { dst, src } => {
                    dst.rewrite(&mut |e| container_rewrite(e, vars, db, false));
                    src.rewrite(&mut |e| container_rewrite(e, vars, db, true));
                }
                Stmt::Expr(e) | Stmt::Return(Some(e)) => e.rewrite(&mut |x| container_rewrite(x, vars, db, true)),
                Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } | Stmt::For { cond, .. } => cond.rewrite(&mut |x| container_rewrite(x, vars, db, true)),
                Stmt::Switch { e, .. } => e.rewrite(&mut |x| container_rewrite(x, vars, db, true)),
                _ => {}
            }
        }
    });
}

fn container_rewrite(e: &mut Expr, vars: &[Var], db: &TypeDb, allow_const: bool) {
    {
        let (base, offset, ty, ptr) = match e {
            Expr::Load { base, offset, ty } => (base.clone(), *offset, ty.clone(), true),
            Expr::Member { base, offset, ty } => (base.clone(), *offset, ty.clone(), false),
            _ => return,
        };
        if matches!(ty, Type::Unknown { size: 0 }) {
            return;
        }
        let bt = types::ty_of(&base, vars);
        let ot = if ptr {
            pointee(&bt).cloned()
        } else {
            match strip_cv(&bt) {
                Type::Ref(t) => Some((**t).clone()),
                t => Some(t.clone()),
            }
        };
        let Some(ot) = ot else { return };
        let is_const = matches!(ot, Type::Const(_)) || (!ptr && matches!(bt, Type::Const(_)));
        if !allow_const && is_const {
            return;
        }
        let otr = types::resolve(Some(db), &ot).into_owned();
        let Some(cls) = named(&otr).map(|s| s.to_string()) else { return };
        // the object is a container itself, or holds one
        let own = container_elem(db, &cls).and_then(|et| {
            let esz = types::size_of(Some(db), &et)?;
            let c = crate::sig::find_class(db, &cls)?;
            if c.fields.is_empty() {
                // an instance without members in the context: the storage offset from another
                // instance of the template (inline byte storage, same place for every T)
                let go = template_storage_offset(db, &cls)?;
                let n = template_count(&cls)?;
                return (offset >= go && offset < go + (n * esz) as i32).then(|| (go, et.clone()));
            }
            c.fields.iter().find_map(|g| {
                let go = g.offset as i32;
                let gt = types::resolve(Some(db), &g.ty).into_owned();
                let gs = types::size_of(Some(db), &gt).unwrap_or(0) as i32;
                (matches!(strip_cv(&gt), Type::Array(..)) && gs > 0 && esz > 0 && gs as u32 % esz == 0 && offset >= go && offset < go + gs).then(|| (go, et.clone()))
            })
        });
        let whole_obj = own.is_some();
        let (co, ct, d0, et) = match own {
            Some((go, et)) => (0, ot.clone(), go, et),
            None => match container_at(db, &cls, offset, 0) {
                Some(x) => x,
                None => return,
            },
        };
        let Some(esz) = types::size_of(Some(db), &et).filter(|s| *s > 0) else { return };
        let rel = offset - d0;
        if rel < 0 {
            return;
        }
        let j = rel / esz as i32;
        let inner = rel - j * esz as i32;
        let cont = if whole_obj && !ptr {
            (*base).clone()
        } else if ptr {
            Expr::Load { base: base.clone(), offset: co, ty: ct }
        } else {
            Expr::Member { base: base.clone(), offset: co, ty: ct }
        };
        let elem = Expr::Index { base: Box::new(cont), index: Box::new(Expr::int(j as i64)), ty: et.clone() };
        if !types::is_aggregate(Some(db), &et) {
            if let Some(n) = finish(elem, &et, inner, &ty, Some(db)) {
                *e = n;
            }
            return;
        }
        // an aggregate element: the member at `inner` (bitfield units stay raw members of it)
        *e = Expr::Member { base: Box::new(elem), offset: inner, ty };
    }
}

/// Offset of the inline byte storage (`uchar mData[N * sizeof(T)]`) in the instances of
/// `cls`'s class template that the context does lay out.
fn template_storage_offset(db: &TypeDb, cls: &str) -> Option<i32> {
    let base = &cls[..cls.find('<')?];
    let prefix = format!("{base}<");
    let mut found: Option<i32> = None;
    for (name, c) in db.classes.range(prefix.clone()..) {
        if !name.starts_with(&prefix) {
            break;
        }
        let arr = c.fields.iter().find(|f| matches!(strip_cv(&f.ty), Type::Array(e, _) if scalar_size(e) == Some(1)));
        if let Some(f) = arr {
            match found {
                None => found = Some(f.offset as i32),
                Some(o) if o == f.offset as i32 => {}
                Some(_) => return None,
            }
        }
    }
    found
}

/// The last integer template argument of `cls` (`reserved_vector<T, 16>` -> 16).
fn template_count(cls: &str) -> Option<u32> {
    let lt = cls.find('<')?;
    let args = crate::sig::split_top(&cls[lt + 1..cls.len().checked_sub(1)?], ',');
    args.last()?.trim().parse().ok()
}

fn assigns<'a>(b: &'a [Stmt], out: &mut Vec<(&'a Expr, &'a Expr)>) {
    for s in b {
        match s {
            Stmt::Assign { dst, src } => out.push((dst, src)),
            Stmt::If { then, els, .. } => {
                assigns(then, out);
                assigns(els, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => assigns(body, out),
            Stmt::For { init, step, body, .. } => {
                assigns(init, out);
                assigns(step, out);
                assigns(body, out);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    assigns(&c.body, out);
                }
            }
            _ => {}
        }
    }
}

/// Hardware register blocks of the console (`0xCC000000`..`0xCC008000`): accesses at constant
/// addresses there that no context declaration covers are elements of a register array the
/// draft declares itself at the block's address (`volatile u32 X[N] : 0xCC006400;`, how the SDK
/// declares them): the compiler then materializes the array's address as a symbol, which differs
/// from a cast constant (scheduling, `lis`/`addi` + indexed access for computed indices).
/// Blocks are 0x400-aligned; every access to a block must have the same size.
pub fn synth_hw_arrays(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>) -> Vec<GlobalRef> {
    const LO: u32 = 0xCC00_0000;
    const HI: u32 = 0xCC00_8000;
    fn konst(e: &Expr) -> Option<u32> {
        match e {
            Expr::Int { value, .. } => Some(*value as u32),
            Expr::Cast { e, .. } => konst(e),
            _ => None,
        }
    }
    // (x, n, address) of `K + x*n` / K
    let split = |base: &Expr, offset: i32| -> Option<(Option<Expr>, u32, u32)> {
        if let Some(k) = konst(base) {
            return Some((None, 1, k.wrapping_add(offset as u32)));
        }
        let mut e = base;
        while let Expr::Cast { e: inner, .. } = e {
            e = inner;
        }
        let Expr::Binary { op: BinOp::Add, l, r, .. } = e else { return None };
        let (k, rest) = match (konst(l), konst(r)) {
            (Some(k), None) => (k, &**r),
            (None, Some(k)) => (k, &**l),
            _ => return None,
        };
        let (x, n, c) = split_index(rest, vars, db).unwrap_or((rest.clone(), 1, 0));
        Some((Some(x), n, (k as i64).wrapping_add(c).wrapping_add(offset as i64) as u32))
    };
    let declared = |a: u32| {
        db.is_some_and(|db| {
            db.abs_addrs.iter().any(|(n, start)| {
                let size = db.globals.get(n).and_then(|(_, t)| types::size_of(Some(db), t)).unwrap_or(4);
                a >= *start && a < start.wrapping_add(size.max(4))
            })
        })
    };
    // block -> access size (0 = mixed)
    let mut blocks: HashMap<u32, u32> = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Load { base, offset, ty } = e {
            if let (Some((x, n, a)), Some(es)) = (split(base, *offset), scalar_size(ty)) {
                if (LO..HI).contains(&a) && !declared(a) && matches!(es, 1 | 2 | 4) && a % es == 0 && (x.is_none() || n % es == 0) {
                    let b = blocks.entry(a & !0x3ff).or_insert(es);
                    if *b != es {
                        *b = 0;
                    }
                }
            }
        }
    });
    blocks.retain(|_, es| *es != 0);
    if blocks.is_empty() {
        return vec![];
    }
    let name = |b: u32| format!("__hwregs_{b:08X}");
    let arr = |es: u32| Type::Array(Box::new(Type::Volatile(Box::new(Type::Int { size: es as u8, signed: false }))), 0x400 / es);
    Stmt::rewrite_exprs(body, &mut |e| {
        let Expr::Load { base, offset, ty } = e else { return };
        let Some((x, n, a)) = split(base, *offset) else { return };
        let Some(&es) = blocks.get(&(a & !0x3ff)) else { return };
        if !(LO..HI).contains(&a) || scalar_size(ty) != Some(es) || a % es != 0 || (x.is_some() && n % es != 0) {
            return;
        }
        let b = a & !0x3ff;
        let it = Type::Int { size: 4, signed: true };
        let k = Expr::int(((a - b) / es) as i64);
        let idx = match x {
            None => k,
            Some(x) => {
                let xi = if n / es == 1 { x } else { Expr::bin(BinOp::Mul, x, Expr::int((n / es) as i64), it.clone()) };
                if a > b {
                    Expr::bin(BinOp::Add, xi, k, it)
                } else {
                    xi
                }
            }
        };
        // (the element type itself: a cast would drop the volatile access)
        let et = Type::Int { size: es as u8, signed: false };
        *e = Expr::Index { base: Box::new(Expr::Global { symbol: name(b), ty: arr(es) }), index: Box::new(idx), ty: et };
    });
    blocks
        .into_iter()
        .map(|(b, es)| GlobalRef { symbol: name(b), ty: arr(es), is_function: false, section: None, local_def: true, init: None, abs_addr: Some(b) })
        .collect()
}

/// Accesses at constant addresses inside a variable the context declares at an absolute address
/// (hardware register arrays, `vu16 __DSPRegs[32] : 0xCC005000;`) are elements of it:
/// `__DSPRegs[3]` (the compiler then materializes the array's base and indexes from it).
pub fn absolute_globals(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb) {
    if db.abs_addrs.is_empty() {
        return;
    }
    // (name, start, element type, element size, count)
    let mut regions: Vec<(String, u32, Type, u32, u32)> = vec![];
    // (name, start, type, size) of aggregates (`PPCWGPipe GXWGFifo : 0xCC008000;`)
    let mut objects: Vec<(String, u32, Type, u32)> = vec![];
    for (n, a) in &db.abs_addrs {
        let Some((_, t)) = db.globals.get(n).or_else(|| db.globals.values().find(|(q, _)| q == n)) else { continue };
        let r = types::resolve(Some(db), t).into_owned();
        if let Type::Array(e, cnt) = strip_cv(&r) {
            if let Some(es) = types::size_of(Some(db), e).filter(|s| matches!(s, 1 | 2 | 4)) {
                regions.push((n.clone(), *a, (**e).clone(), es, *cnt));
            }
        } else if let Some(s) = types::size_of(Some(db), &r).filter(|s| *s > 0) {
            // aggregates, and scalars (`u32 __OSBusClock : 0x800000F8`)
            objects.push((n.clone(), *a, t.clone(), s));
        }
    }
    if regions.is_empty() && objects.is_empty() {
        return;
    }
    fn konst(e: &Expr) -> Option<u32> {
        match e {
            Expr::Int { value, .. } => Some(*value as u32),
            Expr::Cast { e, .. } => konst(e),
            _ => None,
        }
    }
    // a register-array element with a computed index, `K + i*n` (`__EXIRegs[chan * 5]`): the
    // compiler materializes the array's address and adds the scaled index. A single-assignment
    // variable holding such an address (the compiler's CSE of the repeated element) is
    // substituted back into its accesses first.
    // `K + x*n (+ c)` with a constant base K (possibly behind a cast) -> (x, n, K + c)
    let split_abs = |e: &Expr| -> Option<(Expr, u32, i64)> {
        let mut e = e;
        while let Expr::Cast { e: inner, .. } = e {
            e = inner;
        }
        let Expr::Binary { op: BinOp::Add, l, r, .. } = e else { return None };
        let (k, rest) = match (konst(l), konst(r)) {
            (Some(k), None) => (k, &**r),
            (None, Some(k)) => (k, &**l),
            _ => return None,
        };
        let (x, n, c) = split_index(rest, vars, Some(db)).or_else(|| Some((rest.clone(), 1, 0)))?;
        Some((x, n, (k as i64).wrapping_add(c)))
    };
    let in_region = |a: u32| regions.iter().any(|(_, start, _, es, cnt)| a >= *start && a < start.wrapping_add(es * cnt));
    let dyn_addr = |e: &Expr| -> bool {
        matches!(split_abs(e), Some((_, _, k)) if in_region(k as u32))
    };
    let mut nassign: HashMap<VarId, usize> = HashMap::new();
    let mut value: HashMap<VarId, Expr> = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| {
        for st in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(v), src } = st {
                *nassign.entry(*v).or_default() += 1;
                value.insert(*v, src.clone());
            }
        }
    });
    let subst: HashMap<VarId, Expr> = value
        .into_iter()
        .filter(|(v, src)| {
            nassign.get(v) == Some(&1) && dyn_addr(src) && {
                // operands never reassigned (parameters, single-assignment temps)
                let mut ok = true;
                src.walk(&mut |x| {
                    if let Expr::Var(w) = x {
                        ok &= nassign.get(w).copied().unwrap_or(0) <= 1 && w != v;
                    }
                });
                ok
            }
        })
        .collect();
    if !subst.is_empty() {
        Stmt::rewrite_exprs(body, &mut |e| {
            if let Expr::Load { base, .. } = e {
                if let Expr::Var(v) = &**base {
                    if let Some(src) = subst.get(v) {
                        *base = Box::new(src.clone());
                    }
                }
            }
        });
    }
    Stmt::rewrite_exprs(body, &mut |e| {
        let Expr::Load { base, offset, ty } = e else { return };
        if konst(base).is_none() {
            let Some((x, n, k)) = split_abs(base) else { return };
            let a = (k as u32).wrapping_add(*offset as u32);
            for (name, start, et, es, cnt) in &regions {
                let end = start.wrapping_add(es * cnt);
                if a >= *start && a < end && (a - start) % es == 0 && n % es == 0 && scalar_size(ty) == Some(*es) {
                    let it = Type::Int { size: 4, signed: true };
                    let mut idx = if n / es == 1 { x.clone() } else { Expr::bin(BinOp::Mul, x.clone(), Expr::int((n / es) as i64), it.clone()) };
                    if a > *start {
                        idx = Expr::bin(BinOp::Add, idx, Expr::int(((a - start) / es) as i64), it);
                    }
                    let arr = Type::Array(Box::new(et.clone()), *cnt);
                    *e = Expr::Index { base: Box::new(Expr::Global { symbol: name.clone(), ty: arr }), index: Box::new(idx), ty: et.clone() };
                    return;
                }
            }
            return;
        }
        let Some(k) = konst(base) else { return };
        let a = k.wrapping_add(*offset as u32);
        for (n, start, et, es, cnt) in &regions {
            let end = start.wrapping_add(es * cnt);
            if a >= *start && a < end && (a - start) % es == 0 && scalar_size(ty) == Some(*es) {
                let arr = Type::Array(Box::new(et.clone()), *cnt);
                *e = Expr::Index { base: Box::new(Expr::Global { symbol: n.clone(), ty: arr }), index: Box::new(Expr::int(((a - start) / es) as i64)), ty: et.clone() };
                return;
            }
        }
        for (n, start, t, s) in &objects {
            if a >= *start && a < start.wrapping_add(*s) && scalar_size(ty).is_some() {
                let g = Expr::Global { symbol: n.clone(), ty: t.clone() };
                if types::is_aggregate(Some(db), t) {
                    *e = Expr::Member { base: Box::new(g), offset: (a - start) as i32, ty: ty.clone() };
                } else if a == *start && scalar_size(ty) == Some(*s) && matches!(strip_cv(ty), Type::Float { .. }) == matches!(strip_cv(&types::resolve(Some(db), t)), Type::Float { .. }) {
                    // a scalar read whole (as its own kind) is the variable itself
                    *e = g;
                }
                return;
            }
        }
    });
}
