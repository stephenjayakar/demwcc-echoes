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
    Stmt::for_each_block_mut(body, &mut |b| merge_in_list(b, vars, is_temp, db));
}

fn merge_in_list(b: &mut Vec<Stmt>, vars: &[Var], is_temp: &dyn Fn(VarId) -> bool, db: &TypeDb) {
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
                    let size = scalar_size(dty).unwrap_or(0);
                    if size == 0 || scalar_size(sty) != Some(size) {
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
            if let (Some(dcls), Some(scls)) = (base_class(&c0.dst_base, c0.dst_ptr, vars, db), base_class(&c0.src_base, c0.src_ptr, vars, db)) {
                let dts = aggregate_at(db, &dcls, dmin);
                let sts = aggregate_at(db, &scls, smin);
                for t in dts.iter().filter(|t| sts.iter().any(|s| sig::norm_name(&format!("{s:?}")) == sig::norm_name(&format!("{t:?}")))) {
                    let Some(fields) = flat(db, t) else { continue };
                    if fields.len() != group.len() {
                        continue;
                    }
                    let ok = fields.iter().all(|(o, s, f)| group.iter().any(|c| c.dst_off - dmin == *o && c.size == *s && c.float == *f));
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
                    let new = Stmt::Assign { dst: mk(&c0.dst_base, dmin, c0.dst_ptr), src: mk(&c0.src_base, smin, c0.src_ptr) };
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
