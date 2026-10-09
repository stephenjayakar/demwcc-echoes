//! Aggregate copies: MWCC copies small structs (CVector3f, TUniqueId pairs, ...) member by member
//! (`lfs/stfs` x3). With a TypeDb, a window of member-wise copies covering every scalar member of
//! a struct, from the same struct type, becomes one assignment `dst = src`.

use crate::ir::*;
use crate::sig;
use crate::types;
use mwdec_core::{Type, TypeDb};
use std::collections::HashMap;

/// (base expression, base offset, is_pointer_base) of an lvalue access.
fn split_access(e: &Expr) -> Option<(&Expr, i32, bool, &Type)> {
    match e {
        Expr::Load { base, offset, ty } => Some((base, *offset, true, ty)),
        Expr::Member { base, offset, ty } => Some((base, *offset, false, ty)),
        Expr::Var(_) => None,
        _ => None,
    }
}

/// The class an access base refers to: pointee for pointer bases, the type itself for members.
fn base_class(base: &Expr, ptr: bool, vars: &[Var], db: &TypeDb) -> Option<String> {
    let t = types::ty_of(base, vars);
    let t = if ptr { pointee(&t)?.clone() } else { t };
    let r = types::resolve(Some(db), &t).into_owned();
    named(&r).map(|s| s.to_string())
}

/// Aggregate-typed member starting exactly at `off` of `cls` (outermost), with its type.
pub(crate) fn aggregate_at(db: &TypeDb, cls: &str, off: i32) -> Vec<Type> {
    let mut out = vec![];
    // the class itself at offset 0
    if off == 0 {
        out.push(Type::Named(cls.to_string()));
    }
    // walk nested aggregates starting at off
    fn walk(db: &TypeDb, c: &mwdec_core::Class, off: i32, out: &mut Vec<Type>, depth: u32) {
        if depth > 10 {
            return;
        }
        for b in &c.bases {
            if let Some(bc) = sig::find_class(db, &b.name) {
                let bo = b.offset as i32;
                if off >= bo && off < bo + bc.size as i32 {
                    if off == bo {
                        out.push(Type::Named(b.name.clone()));
                    }
                    walk(db, bc, off - bo, out, depth + 1);
                }
            }
        }
        for f in &c.fields {
            let fs = types::size_of(Some(db), &f.ty).unwrap_or(0) as i32;
            let fo = f.offset as i32;
            if off < fo || off >= fo + fs.max(1) {
                continue;
            }
            let rt = types::resolve(Some(db), &f.ty).into_owned();
            if let Some(fc) = types::class_of(Some(db), &rt) {
                if off == fo {
                    out.push(strip_cv(&rt).clone());
                }
                walk(db, fc, off - fo, out, depth + 1);
            }
        }
    }
    if let Some(c) = sig::find_class(db, cls) {
        walk(db, c, off, &mut out, 0);
    }
    out
}

fn flat(db: &TypeDb, t: &Type) -> Option<Vec<(i32, u32, bool)>> {
    let cls = named(&types::resolve(Some(db), t)).map(|s| s.to_string())?;
    let mut v = vec![];
    flat_into(db, &cls, 0, &mut v, 0)?;
    Some(v)
}

fn flat_into(db: &TypeDb, cls: &str, base: i32, out: &mut Vec<(i32, u32, bool)>, depth: u32) -> Option<()> {
    if depth > 8 {
        return None;
    }
    let c = sig::find_class(db, cls)?;
    if c.vptr_offset.is_some() || c.is_union {
        return None;
    }
    for b in &c.bases {
        flat_into(db, &b.name, base + b.offset as i32, out, depth + 1)?;
    }
    for f in &c.fields {
        if f.bitfield.is_some() {
            return None;
        }
        let rt = types::resolve(Some(db), &f.ty).into_owned();
        match strip_cv(&rt) {
            Type::Array(e, n) => {
                let es = types::size_of(Some(db), e)?;
                if types::is_aggregate(Some(db), e) {
                    return None;
                }
                for k in 0..*n {
                    out.push((base + f.offset as i32 + (k * es) as i32, es, is_float(e)));
                }
            }
            _ if types::is_aggregate(Some(db), &rt) => {
                flat_into(db, named(&rt)?, base + f.offset as i32, out, depth + 1)?;
            }
            _ => out.push((base + f.offset as i32, types::size_of(Some(db), &rt)?, is_float(&rt))),
        }
    }
    Some(())
}

/// Offsets of the scalar leaves of `t` that belong to a member object (not a direct scalar).
fn nested_leaves(db: &TypeDb, t: &Type) -> std::collections::HashSet<i32> {
    let mut out = std::collections::HashSet::new();
    let Some(cls) = named(&types::resolve(Some(db), t)).map(|s| s.to_string()) else { return out };
    let Some(c) = sig::find_class(db, &cls) else { return out };
    for f in &c.fields {
        let ft = types::resolve(Some(db), &f.ty).into_owned();
        if types::is_aggregate(Some(db), &ft) {
            for (o, _, _) in flat(db, &ft).unwrap_or_default() {
                out.insert(f.offset as i32 + o);
            }
        }
    }
    out
}

struct Copy {
    stmt: usize,
    dst_base: Expr,
    dst_off: i32,
    dst_ptr: bool,
    src_base: Expr,
    src_off: i32,
    src_ptr: bool,
    size: u32,
    float: bool,
    /// temp whose def is folded into this copy
    temp_def: Option<usize>,
}

pub fn merge_copies(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &dyn Fn(VarId) -> bool, db: &TypeDb) {
    let mut retype = vec![];
    let none = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| merge_in_list(b, vars, is_temp, db, &mut retype, &none));
}

/// `merge_copies`, also typing untyped stack regions that receive a whole-object copy at their
/// start (`T v = gSomeT;` copied member-wise into a buffer).
pub fn merge_copies_typing(body: &mut Vec<Stmt>, vars: &mut [Var], is_temp: &dyn Fn(VarId) -> bool, db: &TypeDb) {
    // a region that is a by-value argument's copy stays as it is (the by-value forwarding
    // turns `S = x; f(S)` into `f(x)`)
    let mut byval_arg: Vec<VarId> = vec![];
    let mut rvalue_arg: Vec<VarId> = vec![];
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Call { callee, args, .. } = e {
            let sig = match callee {
                Callee::Direct { sig, .. } | Callee::Method { sig, .. } => Some(sig),
                Callee::Virtual { sig, .. } => sig.as_ref(),
                Callee::Indirect(_) => None,
            };
            for (n, a) in args.iter().enumerate() {
                if let Expr::AddrOf(x) = a {
                    if let Expr::Var(v) = **x {
                        let by_ref = sig.and_then(|s| s.params.get(n)).map_or(true, |p| is_ptr(&p.ty));
                        if !by_ref {
                            byval_arg.push(v);
                        }
                    }
                }
            }
        }
    });
    // accesses beyond the object's end: the region is a bigger object that starts with one
    let mut extent: HashMap<VarId, i64> = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        let (v, off, ty) = match e {
            Expr::Member { base, offset, ty } => match **base {
                Expr::Var(v) => (v, *offset, ty),
                _ => return,
            },
            Expr::Load { base, offset, ty } => match &**base {
                Expr::AddrOf(x) => match **x {
                    Expr::Var(v) => (v, *offset, ty),
                    _ => return,
                },
                _ => return,
            },
            _ => return,
        };
        let end = *offset_end(off, ty);
        let x = extent.entry(v).or_insert(0);
        *x = (*x).max(end);
    });
    // a region whose address is passed to one call and that is otherwise only written is the
    // temporary of an rvalue argument (`f(T(x))` for a `const T&`): construction folding keeps it
    {
        let mut addr_args: HashMap<VarId, usize> = HashMap::new();
        let mut mentions: HashMap<VarId, usize> = HashMap::new();
        let mut dst_mentions: HashMap<VarId, usize> = HashMap::new();
        Stmt::walk_exprs(body, &mut |e| {
            if let Expr::Var(v) = e {
                *mentions.entry(*v).or_default() += 1;
            }
            if let Expr::Call { args, .. } = e {
                for a in args {
                    if let Expr::AddrOf(x) = a {
                        if let Expr::Var(v) = **x {
                            *addr_args.entry(v).or_default() += 1;
                        }
                    }
                }
            }
        });
        let mut snap = body.clone();
        Stmt::for_each_block_mut(&mut snap, &mut |b| {
            for s in b.iter() {
                if let Stmt::Assign { dst, .. } = s {
                    let v = match dst {
                        Expr::Var(v) => Some(*v),
                        Expr::Member { base, .. } => match **base {
                            Expr::Var(v) => Some(v),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(v) = v {
                        *dst_mentions.entry(v).or_default() += 1;
                    }
                }
            }
        });
        for (v, n) in addr_args {
            if n == 1 && mentions.get(&v).copied().unwrap_or(0) == 1 + dst_mentions.get(&v).copied().unwrap_or(0) {
                rvalue_arg.push(v);
            }
        }
    }
    let mut retype: Vec<(VarId, Type)> = vec![];
    // (by-value copies are left to the by-value forwarding: no typing for them; a whole copy
    // into an rvalue temporary still merges, `f(T(x))`, but the region stays untyped)
    let mut snapshot: Vec<Var> = vars.to_vec();
    for v in &byval_arg {
        if matches!(snapshot[*v].ty, Type::Unknown { .. }) {
            snapshot[*v].ty = Type::Void;
        }
    }
    Stmt::for_each_block_mut(body, &mut |b| merge_in_list(b, &snapshot, is_temp, db, &mut retype, &extent));
    // one type per region (the first whole copy decides; conflicting copies keep it untyped)
    let mut seen: HashMap<VarId, Option<Type>> = HashMap::new();
    for (v, t) in retype {
        let e = seen.entry(v).or_insert(Some(t.clone()));
        if e.as_ref().map(|x| sig::norm_name(&format!("{x:?}"))) != Some(sig::norm_name(&format!("{t:?}"))) {
            *e = None;
        }
    }
    for (v, t) in seen {
        if let Some(t) = t {
            let size = types::size_of(Some(db), &t).unwrap_or(0) as i64;
            if !byval_arg.contains(&v) && !rvalue_arg.contains(&v) && extent.get(&v).map_or(true, |e| *e <= size) {
                vars[v].ty = t;
            }
        }
    }
}

fn merge_in_list(b: &mut Vec<Stmt>, vars: &[Var], is_temp: &dyn Fn(VarId) -> bool, db: &TypeDb, retype: &mut Vec<(VarId, Type)>, extent: &HashMap<VarId, i64>) {
    let mut i = 0;
    while i < b.len() {
        // collect a window of temp loads + stores
        let mut temps: HashMap<VarId, (usize, Expr)> = HashMap::new();
        let mut copies: Vec<Copy> = vec![];
        let mut j = i;
        while j < b.len() {
            match &b[j] {
                Stmt::Assign { dst: Expr::Var(t), src } if is_temp(*t) && split_access(src).is_some() => {
                    temps.insert(*t, (j, src.clone()));
                }
                Stmt::Assign { dst, src } if split_access(dst).is_some() => {
                    let (s_expr, tdef) = match src {
                        Expr::Var(t) if temps.contains_key(t) => (temps[t].1.clone(), Some(temps[t].0)),
                        e if split_access(e).is_some() => (e.clone(), None),
                        _ => break,
                    };
                    let (db_, doff, dptr, dty) = split_access(dst).unwrap();
                    let (sb, soff, sptr, sty) = split_access(&s_expr).unwrap();
                    // an enum member copied into an untyped word of the destination buffer
                    let enum_size = |t: &Type| -> Option<u32> {
                        match t {
                            Type::Named(n) if db.enums.contains_key(n.as_str()) => types::size_of(Some(db), t),
                            _ => None,
                        }
                    };
                    let (dsize, ssize) = match (dty, enum_size(sty)) {
                        (Type::Unknown { size }, Some(n)) => (Some(*size), Some(n)),
                        _ => (scalar_size(dty), scalar_size(sty).or_else(|| enum_size(sty))),
                    };
                    let size = dsize.unwrap_or(0);
                    if size == 0 || ssize != Some(size) {
                        break;
                    }
                    copies.push(Copy {
                        stmt: j,
                        dst_base: db_.clone(),
                        dst_off: doff,
                        dst_ptr: dptr,
                        src_base: sb.clone(),
                        src_off: soff,
                        src_ptr: sptr,
                        size,
                        float: is_float(dty),
                        temp_def: tdef,
                    });
                }
                _ => break,
            }
            j += 1;
        }
        let mut merged = false;
        if !copies.is_empty() {
            // try the longest prefix-consistent group starting at the first copy
            let c0 = &copies[0];
            let group: Vec<&Copy> = copies
                .iter()
                .filter(|c| c.dst_base == c0.dst_base && c.src_base == c0.src_base && c.dst_ptr == c0.dst_ptr && c.src_ptr == c0.src_ptr && (c.dst_off - c.src_off) == (c0.dst_off - c0.src_off))
                .collect();
            let dmin = group.iter().map(|c| c.dst_off).min().unwrap();
            let smin = dmin - (c0.dst_off - c0.src_off);
            // an untyped stack region written whole from its start takes the source's type
            let untyped_dst = match &c0.dst_base {
                Expr::Var(v) if !c0.dst_ptr && dmin == 0 && matches!(vars[*v].ty, Type::Unknown { .. }) => match vars[*v].kind {
                    VarKind::Stack { size, .. } => Some((*v, size)),
                    _ => None,
                },
                _ => None,
            };
            let scls_opt = base_class(&c0.src_base, c0.src_ptr, vars, db);
            // a byte pointer (an element of a strided array) receiving a whole object
            // (several members: a one-scalar object stored through a byte pointer stays a scalar
            // store, the shape inline templates such as `push_back` expand to)
            let raw_ptr_dst = group.len() >= 2 && c0.dst_ptr && base_class(&c0.dst_base, true, vars, db).is_none() && pointee(&types::ty_of(&c0.dst_base, vars)).is_some_and(|t| scalar_size(t) == Some(1));
            let dcls_opt = base_class(&c0.dst_base, c0.dst_ptr, vars, db).or_else(|| (untyped_dst.is_some() || raw_ptr_dst).then(|| scls_opt.clone()).flatten().and_then(|s| {
                let ts = aggregate_at(db, &s, smin);
                ts.iter().filter_map(|t| named(t).map(|n| n.to_string())).next()
            }));
            if let (Some(dcls), Some(scls)) = (dcls_opt, scls_opt) {
                let dts = if untyped_dst.is_some() && base_class(&c0.dst_base, c0.dst_ptr, vars, db).is_none() {
                    let (uv, usize_) = untyped_dst.unwrap();
                    let need = extent.get(&uv).copied().unwrap_or(0);
                    aggregate_at(db, &scls, smin).into_iter().filter(|t| types::size_of(Some(db), t).map_or(false, |s| s <= usize_ && s as i64 >= need)).collect::<Vec<_>>()
                } else if raw_ptr_dst {
                    aggregate_at(db, &scls, smin)
                } else {
                    aggregate_at(db, &dcls, dmin)
                };
                let sts = aggregate_at(db, &scls, smin);
                for t in dts.iter().filter(|t| sts.iter().any(|s| sig::norm_name(&format!("{s:?}")) == sig::norm_name(&format!("{t:?}")))) {
                    let Some(fields) = flat(db, t) else { continue };
                    // a 64-bit integer member is copied as two words
                    let fields: Vec<_> = fields.into_iter().flat_map(|(o, s, f)| if s == 8 && !f { vec![(o, 4, f), (o + 4, 4, f)] } else { vec![(o, s, f)] }).collect();
                    if fields.len() != group.len() {
                        continue;
                    }
                    // an all-float object copied word by word through integer registers (a POD
                    // copy-construction: `new (p) T(x)`, a by-value copy) is a whole copy too
                    let words = group.iter().all(|c| !c.float && c.size == 4);
                    // a member object is one block region of the copy: its floats move as words
                    let nested = nested_leaves(db, t);
                    let ok = fields.iter().all(|(o, s, f)| {
                        group.iter().any(|c| c.dst_off - dmin == *o && c.size == *s && (c.float == *f || ((words || nested.contains(o)) && *f && !c.float && *s == 4)))
                    });
                    if !ok {
                        continue;
                    }
                    // temps folded must not be used outside the window
                    let window: Vec<usize> = group.iter().map(|c| c.stmt).chain(group.iter().filter_map(|c| c.temp_def)).collect();
                    let temp_ids: Vec<VarId> = temps.iter().filter(|(_, (k, _))| window.contains(k)).map(|(t, _)| *t).collect();
                    let used_outside = b.iter().enumerate().any(|(k, s)| !window.contains(&k) && temp_ids.iter().any(|t| stmt_uses(s, *t)));
                    if used_outside {
                        continue;
                    }
                    let mk = |base: &Expr, off: i32, ptr: bool| {
                        if ptr {
                            Expr::Load { base: Box::new(base.clone()), offset: off, ty: t.clone() }
                        } else if off == 0 && named(&types::ty_of(base, vars)).map(sig::norm_name) == named(t).map(sig::norm_name) {
                            base.clone()
                        } else {
                            Expr::Member { base: Box::new(base.clone()), offset: off, ty: t.clone() }
                        }
                    };
                    let new_dst = match untyped_dst {
                        Some((v, _)) if base_class(&c0.dst_base, c0.dst_ptr, vars, db).is_none() => {
                            retype.push((v, t.clone()));
                            Expr::Var(v)
                        }
                        _ => mk(&c0.dst_base, dmin, c0.dst_ptr),
                    };
                    let new = Stmt::Assign { dst: new_dst, src: mk(&c0.src_base, smin, c0.src_ptr) };
                    let first = *window.iter().min().unwrap();
                    let mut rm: Vec<usize> = window.clone();
                    rm.sort_unstable();
                    rm.dedup();
                    for k in rm.iter().rev() {
                        b.remove(*k);
                    }
                    b.insert(first, new);
                    merged = true;
                    break;
                }
            }
        }
        if !merged {
            i += 1;
        }
    }
}

fn stmt_uses(s: &Stmt, t: VarId) -> bool {
    let mut found = false;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if matches!(e, Expr::Var(v) if *v == t) {
            found = true;
        }
    });
    found
}

fn is_literal_sym(n: &str) -> bool {
    n.starts_with('@') || n.starts_with("...") || n.starts_with("$$")
}

/// A local aggregate initialized from constants: MWCC copies the whole object word by word from
/// a literal data object (`lwz r0,@10 ; stw r0,0x14(r1)`). Every member's value is in that
/// object's bytes, so the copies become the initializer (`GXColor c = {0, 0, 0, 255};`, which
/// makes the compiler emit the same literal object).
pub fn literal_inits(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb, obj: &mwdec_core::ObjectFile) {
    // (stack var, offset, literal symbol, literal offset, size) of one word copy
    let copy = |s: &Stmt| -> Option<(VarId, i32, String, i32, u32)> {
        let Stmt::Assign { dst, src } = s else { return None };
        let (v, off, t) = match dst {
            Expr::Member { base, offset, ty } => match **base {
                Expr::Var(v) => (v, *offset, ty.clone()),
                _ => return None,
            },
            _ => return None,
        };
        if !matches!(vars[v].kind, VarKind::Stack { .. }) {
            return None;
        }
        let (sym, so) = match src {
            Expr::Global { symbol, .. } => (symbol.clone(), 0),
            Expr::Member { base, offset, .. } => match &**base {
                Expr::Global { symbol, .. } => (symbol.clone(), *offset),
                _ => return None,
            },
            _ => return None,
        };
        if !is_literal_sym(&sym) {
            return None;
        }
        let n = scalar_size(&t).filter(|n| matches!(n, 1 | 2 | 4))?;
        Some((v, off, sym, so, n))
    };
    let mut writes: HashMap<VarId, usize> = HashMap::new();
    {
        let mut snap = body.clone();
        Stmt::for_each_block_mut(&mut snap, &mut |b| {
            for s in b.iter() {
                if let Stmt::Assign { dst, .. } = s {
                    let v = match dst {
                        Expr::Var(v) => Some(*v),
                        Expr::Member { base, .. } => match **base {
                            Expr::Var(v) => Some(v),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(v) = v {
                        *writes.entry(v).or_default() += 1;
                    }
                }
            }
        });
    }
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let Some((v, 0, sym, 0, n0)) = copy(&b[i]) else {
                i += 1;
                continue;
            };
            let t = types::resolve(Some(db), &vars[v].ty).into_owned();
            let Some(cls) = named(&t).map(|s| s.to_string()) else {
                i += 1;
                continue;
            };
            let size = types::size_of(Some(db), &t).unwrap_or(0);
            // the run of word copies covering the object
            let mut covered = n0;
            let mut j = i + 1;
            while covered < size && j < b.len() {
                match copy(&b[j]) {
                    Some((v2, o, s2, so, n)) if v2 == v && s2 == sym && o as u32 == covered && so == o => {
                        covered += n;
                        j += 1;
                    }
                    _ => break,
                }
            }
            // the only writes of the object
            if size == 0 || covered != size || writes.get(&v).copied().unwrap_or(0) != j - i {
                i += 1;
                continue;
            }
            let Some(bytes) = mwdec_obj::data_bytes(obj, &sym, 0, size as usize).filter(|x| x.len() == size as usize) else {
                i += 1;
                continue;
            };
            let mut fields = vec![];
            if !crate::idioms::flat_fields(db, &cls, 0, &mut fields, 0) || fields.is_empty() {
                i += 1;
                continue;
            }
            let mut args = vec![];
            let mut ok = true;
            for (off, ft) in &fields {
                let o = *off as usize;
                let rd = |k: usize| -> Option<u64> { bytes.get(o..o + k).map(|s| s.iter().fold(0u64, |a, x| (a << 8) | *x as u64)) };
                let e = match strip_cv(ft) {
                    Type::Float { size: 4 } => rd(4).map(|x| Expr::Float { bits: x, double: false }),
                    Type::Float { size: 8 } => rd(8).map(|x| Expr::Float { bits: x, double: true }),
                    Type::Bool => rd(1).filter(|x| *x <= 1).map(|x| Expr::Int { value: x as i64, ty: Type::Bool }),
                    Type::Int { size, signed } => {
                        let sz = *size as usize;
                        let sg = *signed;
                        rd(sz).map(|x| Expr::Int { value: if sg { ((x << (64 - 8 * sz)) as i64) >> (64 - 8 * sz) } else { x as i64 }, ty: strip_cv(ft).clone() })
                    }
                    Type::Long { signed } => {
                        let sg = *signed;
                        rd(4).map(|x| Expr::Int { value: if sg { x as u32 as i32 as i64 } else { x as i64 }, ty: strip_cv(ft).clone() })
                    }
                    Type::Char => rd(1).map(|x| Expr::Int { value: x as i8 as i64, ty: Type::Char }),
                    Type::Ptr(_) => rd(4).filter(|x| *x == 0).map(|_| Expr::Int { value: 0, ty: strip_cv(ft).clone() }),
                    Type::Named(_) if types::is_enum(Some(db), ft) => {
                        let es = types::size_of(Some(db), ft).unwrap_or(4) as usize;
                        rd(es).map(|x| Expr::cast(ft.clone(), Expr::int(x as u32 as i32 as i64)))
                    }
                    _ => None,
                };
                match e {
                    Some(e) => args.push(e),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                i += 1;
                continue;
            }
            let init = Stmt::Assign { dst: Expr::Var(v), src: Expr::Construct { class: vars[v].ty.clone(), ctor: None, args } };
            b.splice(i..j, [init]);
            i += 1;
        }
    });
}

fn offset_end(off: i32, ty: &Type) -> Box<i64> {
    Box::new(off as i64 + scalar_size(ty).unwrap_or(1).max(1) as i64)
}

/// A local array initialized with values the function computes (`const T* v[2] = {&a, &b};`):
/// MWCC copies a literal template object word by word into the frame and then stores the
/// computed elements over it. The copies and the stores become the array's initializer (the
/// template's words stand for the elements nothing overwrites); the elements are pointers,
/// ints or floats.
pub fn patched_literal_arrays(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, obj: &mwdec_core::ObjectFile) {
    fn is_word(t: &Type) -> bool {
        scalar_size(t) == Some(4) || matches!(t, Type::Unknown { size: 4 })
    }
    // (stack var, offset) a word store writes
    fn word_dst(e: &Expr, vars: &[Var]) -> Option<(VarId, i32)> {
        let (v, offset, ty) = match e {
            Expr::Member { base, offset, ty } => match **base {
                Expr::Var(v) => (v, *offset, ty),
                _ => return None,
            },
            Expr::Load { base, offset, ty } => match &**base {
                Expr::AddrOf(x) => match **x {
                    Expr::Var(v) => (v, *offset, ty),
                    _ => return None,
                },
                _ => return None,
            },
            _ => return None,
        };
        (matches!(vars[v].kind, VarKind::Stack { .. }) && is_word(ty)).then_some((v, offset))
    }
    // (literal, offset) a word load reads
    fn lit_word(e: &Expr) -> Option<(String, i32)> {
        let e = match e {
            Expr::Cast { e, .. } => e,
            e => e,
        };
        let glob = |b: &Expr| -> Option<String> {
            let b = match b {
                Expr::AddrOf(x) => &**x,
                b => b,
            };
            match b {
                Expr::Global { symbol, .. } if is_literal_sym(symbol) => Some(symbol.clone()),
                _ => None,
            }
        };
        match e {
            Expr::Global { symbol, ty } if is_literal_sym(symbol) && is_word(ty) => Some((symbol.clone(), 0)),
            Expr::Member { base, offset, ty } | Expr::Load { base, offset, ty } if is_word(ty) => glob(base).map(|g| (g, *offset)),
            _ => None,
        }
    }
    let mut retyped: Vec<(VarId, Type)> = vec![];
    let snapshot = vars.clone();
    Stmt::for_each_block_mut(body, &mut |b| {
        let vars = &snapshot;
        let mut i = 0;
        while i < b.len() {
            // a literal word copied to offset 0 of a frame object
            let start = match &b[i] {
                Stmt::Assign { dst, src } => match (word_dst(dst, vars), lit_word(src)) {
                    (Some((v, 0)), Some((g, 0))) => Some((v, g)),
                    _ => None,
                },
                _ => None,
            };
            let Some((v, g)) = start else {
                i += 1;
                continue;
            };
            let VarKind::Stack { size, .. } = vars[v].kind else {
                i += 1;
                continue;
            };
            let size = size as i32;
            if size < 8 || size % 4 != 0 || size > 64 || named(&vars[v].ty).is_some() {
                i += 1;
                continue;
            }
            // the window: temps holding template words, copies, then the computed stores, up to
            // the first other use of the object
            let mut temps: HashMap<VarId, i32> = HashMap::new();
            let mut copied: Vec<i32> = vec![];
            let mut patched: Vec<(i32, Expr, usize)> = vec![];
            let mut removable: Vec<usize> = vec![];
            let mut ok = true;
            // temp definitions of template words just before the start
            let mut k0 = i;
            while k0 > 0 {
                match &b[k0 - 1] {
                    Stmt::Assign { dst: Expr::Var(t), src } if lit_word(src).is_some_and(|(s, _)| s == g) => {
                        temps.insert(*t, lit_word(src).unwrap().1);
                        removable.push(k0 - 1);
                        k0 -= 1;
                    }
                    _ => break,
                }
            }
            for k in i..b.len() {
                let s = &b[k];
                let as_word_store = match s {
                    Stmt::Assign { dst, src } => word_dst(dst, vars).filter(|(w, _)| *w == v).map(|(_, o)| (o, src)),
                    _ => None,
                };
                if let Some((o, src)) = as_word_store {
                    if o < 0 || o >= size || o % 4 != 0 {
                        ok = false;
                        break;
                    }
                    let from_tpl = match lit_word(src) {
                        Some((s2, so)) => s2 == g && so == o,
                        None => matches!(src, Expr::Var(t) if temps.get(t) == Some(&o)),
                    };
                    if from_tpl && !copied.contains(&o) && !patched.iter().any(|p| p.0 == o) {
                        copied.push(o);
                    } else if copied.contains(&o) && !patched.iter().any(|p| p.0 == o) && !stmt_uses_var(src, v) {
                        // (a pointer stored as a word: the pointer)
                        let val = match src {
                            Expr::Cast { e, .. } if is_ptr(&types::ty_of(e, vars)) => (**e).clone(),
                            e => e.clone(),
                        };
                        patched.push((o, val, k));
                    } else {
                        ok = false;
                        break;
                    }
                    removable.push(k);
                    continue;
                }
                // a temp holding a template word
                if let Stmt::Assign { dst: Expr::Var(t), src } = s {
                    if let Some((s2, so)) = lit_word(src) {
                        if s2 == g {
                            temps.insert(*t, so);
                            removable.push(k);
                            continue;
                        }
                    }
                }
                if stmt_uses(s, v) || !matches!(s, Stmt::Assign { dst: Expr::Var(_), .. }) {
                    break;
                }
            }
            if !ok || copied.len() as i32 * 4 != size || patched.is_empty() {
                i += 1;
                continue;
            }
            // the temps only serve the copies
            let temp_used_elsewhere = temps.keys().any(|t| b.iter().enumerate().any(|(k, s)| !removable.contains(&k) && stmt_uses(s, *t)));
            if temp_used_elsewhere {
                i += 1;
                continue;
            }
            // (a template in a zero-initialized section has no bytes: zeros)
            let Some(bytes) = mwdec_obj::data_bytes(obj, &g, 0, size as usize).map(|x| if x.is_empty() { vec![0u8; size as usize] } else { x.to_vec() }).filter(|x| x.len() == size as usize) else {
                i += 1;
                continue;
            };
            let ptr = patched.iter().all(|(_, e, _)| is_ptr(&types::ty_of(e, vars)));
            let float = patched.iter().all(|(_, e, _)| matches!(types::ty_of(e, vars), Type::Float { size: 4 }));
            let elem = if ptr {
                Type::Ptr(Box::new(Type::Const(Box::new(Type::Void))))
            } else if float {
                Type::Float { size: 4 }
            } else if patched.iter().all(|(_, e, _)| matches!(types::ty_of(e, vars), Type::Int { .. } | Type::Unknown { size: 4 })) {
                Type::Int { size: 4, signed: true }
            } else {
                i += 1;
                continue;
            };
            let mut args = vec![];
            let mut good = true;
            for w in 0..size / 4 {
                let o = w * 4;
                if let Some((_, e, _)) = patched.iter().find(|p| p.0 == o) {
                    args.push(e.clone());
                    continue;
                }
                let raw = bytes[o as usize..o as usize + 4].iter().fold(0u64, |a, x| (a << 8) | *x as u64);
                match &elem {
                    Type::Ptr(_) if raw == 0 => args.push(Expr::Int { value: 0, ty: elem.clone() }),
                    Type::Float { .. } => args.push(Expr::Float { bits: raw, double: false }),
                    Type::Int { .. } => args.push(Expr::int(raw as u32 as i32 as i64)),
                    _ => {
                        good = false;
                        break;
                    }
                }
            }
            if !good {
                i += 1;
                continue;
            }
            let at = patched.iter().map(|p| p.2).max().unwrap_or(i);
            let arr = Type::Array(Box::new(elem), (size / 4) as u32);
            let init = Stmt::Assign { dst: Expr::Var(v), src: Expr::Construct { class: arr.clone(), ctor: None, args } };
            removable.sort_unstable();
            removable.dedup();
            let insert_at = at - removable.iter().filter(|k| **k < at).count();
            for k in removable.iter().rev() {
                b.remove(*k);
            }
            b.insert(insert_at, init);
            retyped.push((v, arr));
            i = insert_at + 1;
        }
    });
    for (v, t) in retyped {
        vars[v].ty = t;
    }
}

fn stmt_uses_var(e: &Expr, v: VarId) -> bool {
    let mut found = false;
    e.walk(&mut |x| {
        if matches!(x, Expr::Var(w) if *w == v) {
            found = true;
        }
    });
    found
}
