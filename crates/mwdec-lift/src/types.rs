//! TypeDb queries shared by the lifter and the emitter: sizes, field paths at offsets, vtables.

use crate::ir::*;
use crate::sig::{find_class, norm_name};
use mwdec_core::{Class, Type, TypeDb};

/// Resolve typedefs (by name) to their underlying type, through cv.
pub fn resolve<'a>(db: Option<&'a TypeDb>, t: &'a Type) -> std::borrow::Cow<'a, Type> {
    use std::borrow::Cow;
    let mut cur: Cow<Type> = Cow::Borrowed(t);
    for _ in 0..16 {
        let next = match strip_cv(&cur) {
            Type::Named(n) => db.and_then(|db| {
                if find_class(db, n).is_some() || db.enums.contains_key(n) {
                    None
                } else {
                    db.typedefs.get(n).or_else(|| {
                        let k = norm_name(n);
                        db.typedefs.iter().find(|(x, _)| norm_name(x) == k).map(|(_, t)| t)
                    })
                }
            }),
            _ => None,
        };
        match next {
            Some(t) => cur = Cow::Owned(t.clone()),
            None => break,
        }
    }
    cur
}

pub fn class_of<'a>(db: Option<&'a TypeDb>, t: &Type) -> Option<&'a Class> {
    let db = db?;
    let r = resolve(Some(db), t);
    match strip_cv(&r) {
        Type::Named(n) => find_class(db, n),
        _ => None,
    }
}

pub fn is_enum(db: Option<&TypeDb>, t: &Type) -> bool {
    let r = resolve(db, t);
    match strip_cv(&r) {
        Type::Named(n) => db.map_or(false, |db| db.enums.contains_key(n.as_str())),
        _ => false,
    }
}

pub fn size_of(db: Option<&TypeDb>, t: &Type) -> Option<u32> {
    let r = resolve(db, t);
    let t = strip_cv(&r);
    if let Some(s) = scalar_size(t) {
        return Some(s);
    }
    match t {
        Type::Named(n) => {
            let db = db?;
            if let Some(c) = find_class(db, n) {
                return Some(c.size);
            }
            db.enums.get(n.as_str()).map(|e| e.size)
        }
        Type::Array(e, k) => size_of(db, e).map(|s| s * k),
        _ => None,
    }
}

/// Is this a class/struct/union passed by hidden reference (aggregate)?
pub fn is_aggregate(db: Option<&TypeDb>, t: &Type) -> bool {
    let r = resolve(db, t);
    match strip_cv(&r) {
        Type::Named(n) => {
            if let Some(db) = db {
                if db.enums.contains_key(n.as_str()) {
                    return false;
                }
                if find_class(db, n).is_some() {
                    return true;
                }
            }
            // Unknown named type: enums are `E...`-prefixed by project convention.
            let last = crate::sig::split_scope(n).1;
            !(last.starts_with('E') && last.chars().nth(1).map_or(false, |c| c.is_ascii_uppercase()))
        }
        // arrays are never passed by value (they decay to pointers)
        Type::Array(..) => false,
        _ => false,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PathElem {
    /// (member name, owning class)
    Field(String, String),
    /// Array index (element size known).
    Index(u32),
    /// Implicit base-class subobject (no syntax needed for member access).
    Base(String),
}

/// Find the member path of an access of `size` bytes (0 = any) at byte `off` inside `class`.
/// Returns the path and the type of the final member. Exact scalar matches only (the final
/// member must start at `off`; nested aggregates are descended).
pub fn field_path(db: &TypeDb, class: &str, off: i32, size: u32) -> Option<(Vec<PathElem>, Type)> {
    let c = find_class(db, class)?;
    field_path_in(db, c, off, size, 0)
}

fn field_path_in(db: &TypeDb, c: &Class, off: i32, size: u32, depth: u32) -> Option<(Vec<PathElem>, Type)> {
    if depth > 12 || off < 0 {
        return None;
    }
    // direct fields first
    for f in &c.fields {
        if f.bitfield.is_some() {
            continue;
        }
        let fs = size_of(Some(db), &f.ty).unwrap_or(0);
        let fo = f.offset as i32;
        if off < fo || (fs > 0 && off >= fo + fs as i32) || (fs == 0 && off != fo) {
            continue;
        }
        let rel = off - fo;
        let rt = resolve(Some(db), &f.ty).into_owned();
        if rel == 0 && (size == 0 || fs == size) && !is_aggregate(Some(db), &rt) {
            return Some((vec![PathElem::Field(f.name.clone(), c.name.clone())], f.ty.clone()));
        }
        if rel == 0 && size == 0 {
            return Some((vec![PathElem::Field(f.name.clone(), c.name.clone())], f.ty.clone()));
        }
        match strip_cv(&rt) {
            Type::Array(e, n) => {
                let es = size_of(Some(db), e).unwrap_or(0);
                if es > 0 {
                    let idx = rel as u32 / es;
                    if idx < *n {
                        let inner = rel - (idx * es) as i32;
                        let mut path = vec![PathElem::Field(f.name.clone(), c.name.clone()), PathElem::Index(idx)];
                        if inner == 0 && (size == 0 || es == size) && !is_aggregate(Some(db), e) {
                            return Some((path, (**e).clone()));
                        }
                        if let Some(ec) = class_of(Some(db), e) {
                            if let Some((p, t)) = field_path_in(db, ec, inner, size, depth + 1) {
                                path.extend(p);
                                return Some((path, t));
                            }
                        }
                    }
                }
            }
            Type::Named(_) => {
                if let Some(fc) = class_of(Some(db), &rt) {
                    if let Some((p, t)) = field_path_in(db, fc, rel, size, depth + 1) {
                        let mut path = vec![PathElem::Field(f.name.clone(), c.name.clone())];
                        path.extend(p);
                        return Some((path, t));
                    }
                    if rel == 0 && fs == size {
                        return Some((vec![PathElem::Field(f.name.clone(), c.name.clone())], f.ty.clone()));
                    }
                }
            }
            _ => {}
        }
    }
    // then base classes
    for b in &c.bases {
        let bc = find_class(db, &b.name)?;
        let bo = b.offset as i32;
        if off >= bo && off < bo + bc.size as i32 {
            if let Some((p, t)) = field_path_in(db, bc, off - bo, size, depth + 1) {
                let mut path = vec![PathElem::Base(b.name.clone())];
                path.extend(p);
                return Some((path, t));
            }
        }
    }
    None
}

/// Member path at `off` whose type is `target` (descending through nested aggregates and bases),
/// e.g. the `CColor` inside a struct whose first member is a `CColor`.
pub fn field_path_of_type(db: &TypeDb, class: &str, off: i32, target: &Type) -> Option<Vec<PathElem>> {
    let c = find_class(db, class)?;
    let want = norm_name(&format!("{:?}", resolve(Some(db), strip_cv(target)).into_owned()));
    path_of_type_in(db, c, off, &want, 0)
}

fn path_of_type_in(db: &TypeDb, c: &Class, off: i32, want: &str, depth: u32) -> Option<Vec<PathElem>> {
    if depth > 12 {
        return None;
    }
    for f in &c.fields {
        if f.bitfield.is_some() {
            continue;
        }
        let fs = size_of(Some(db), &f.ty).unwrap_or(0) as i32;
        let fo = f.offset as i32;
        if off < fo || off >= fo + fs.max(1) {
            continue;
        }
        let rt = resolve(Some(db), strip_cv(&f.ty)).into_owned();
        let rt = strip_cv(&rt).clone();
        if off == fo && norm_name(&format!("{rt:?}")) == want {
            return Some(vec![PathElem::Field(f.name.clone(), c.name.clone())]);
        }
        match &rt {
            Type::Array(e, n) => {
                let es = size_of(Some(db), e).unwrap_or(0) as i32;
                if es > 0 {
                    let idx = (off - fo) / es;
                    if (idx as u32) < *n {
                        let inner = off - fo - idx * es;
                        let et = resolve(Some(db), strip_cv(e)).into_owned();
                        if inner == 0 && norm_name(&format!("{:?}", strip_cv(&et))) == want {
                            return Some(vec![PathElem::Field(f.name.clone(), c.name.clone()), PathElem::Index(idx as u32)]);
                        }
                        if let Some(ec) = class_of(Some(db), e) {
                            if let Some(p) = path_of_type_in(db, ec, inner, want, depth + 1) {
                                let mut v = vec![PathElem::Field(f.name.clone(), c.name.clone()), PathElem::Index(idx as u32)];
                                v.extend(p);
                                return Some(v);
                            }
                        }
                    }
                }
            }
            _ => {
                if let Some(fc) = class_of(Some(db), &rt) {
                    if let Some(p) = path_of_type_in(db, fc, off - fo, want, depth + 1) {
                        let mut v = vec![PathElem::Field(f.name.clone(), c.name.clone())];
                        v.extend(p);
                        return Some(v);
                    }
                }
            }
        }
    }
    for b in &c.bases {
        let bc = find_class(db, &b.name)?;
        let bo = b.offset as i32;
        if off >= bo && off < bo + bc.size as i32 {
            if off == bo && norm_name(&format!("{:?}", Type::Named(b.name.clone()))) == want {
                return Some(vec![PathElem::Base(b.name.clone())]);
            }
            if let Some(p) = path_of_type_in(db, bc, off - bo, want, depth + 1) {
                let mut v = vec![PathElem::Base(b.name.clone())];
                v.extend(p);
                return Some(v);
            }
        }
    }
    None
}

/// Bitfield member covering bits of the byte/half/word at `off` (storage unit `size`) with mask.
pub fn bitfield_at(db: &TypeDb, class: &str, off: i32, size: u32, mask: u32) -> Option<(Vec<PathElem>, Type)> {
    let c = find_class(db, class)?;
    bitfield_in(db, c, off, size, mask, 0)
}

fn bitfield_in(db: &TypeDb, c: &Class, off: i32, size: u32, mask: u32, depth: u32) -> Option<(Vec<PathElem>, Type)> {
    if depth > 12 {
        return None;
    }
    for f in &c.fields {
        let Some((bit_off, bit_size)) = f.bitfield else { continue };
        let unit = if f.size > 0 { f.size } else { size_of(Some(db), &f.ty).unwrap_or(4) };
        // storage unit [f.offset, f.offset+unit); bit_off counted from the MSB of the unit.
        let unit_lo = f.offset as i32;
        if off < unit_lo || off + size as i32 > unit_lo + unit as i32 {
            continue;
        }
        let shift_in_unit = unit * 8 - bit_off as u32 - bit_size as u32;
        let fmask_unit: u64 = (((1u64 << bit_size) - 1) << shift_in_unit) as u64;
        // project to the access window
        let win_shift = (unit_lo + unit as i32 - (off + size as i32)) * 8;
        if win_shift < 0 {
            continue;
        }
        let m = (fmask_unit >> win_shift) as u32 & if size >= 4 { u32::MAX } else { (1u32 << (size * 8)) - 1 };
        if m == mask {
            return Some((vec![PathElem::Field(f.name.clone(), c.name.clone())], f.ty.clone()));
        }
    }
    for f in &c.fields {
        if f.bitfield.is_some() {
            continue;
        }
        if let Some(fc) = class_of(Some(db), &f.ty) {
            let fo = f.offset as i32;
            if off >= fo && off < fo + fc.size as i32 {
                if let Some((p, t)) = bitfield_in(db, fc, off - fo, size, mask, depth + 1) {
                    let mut path = vec![PathElem::Field(f.name.clone(), c.name.clone())];
                    path.extend(p);
                    return Some((path, t));
                }
            }
        }
    }
    for b in &c.bases {
        let bc = find_class(db, &b.name)?;
        let bo = b.offset as i32;
        if off >= bo && off < bo + bc.size as i32 {
            if let Some((p, t)) = bitfield_in(db, bc, off - bo, size, mask, depth + 1) {
                let mut path = vec![PathElem::Base(b.name.clone())];
                path.extend(p);
                return Some((path, t));
            }
        }
    }
    None
}

/// Virtual method at a vtable byte offset of `class` (searching bases).
pub fn vmethod<'a>(db: &'a TypeDb, class: &str, vtable_offset: u32) -> Option<&'a mwdec_core::VirtualMethod> {
    let c = find_class(db, class)?;
    if let Some(m) = c.vtable.iter().find(|m| m.vtable_offset == vtable_offset) {
        if !crate::sig::split_scope(&m.sig.qualified_name).1.is_empty() {
            return Some(m);
        }
        // pure virtual slot in an abstract class: name it from a derived class's override
        if let Some(d) = derived_vmethod(db, &c.name, vtable_offset) {
            return Some(d);
        }
        return Some(m);
    }
    for b in &c.bases {
        if b.offset == 0 {
            if let Some(m) = vmethod(db, &b.name, vtable_offset) {
                return Some(m);
            }
        }
    }
    None
}

fn derives_from(db: &TypeDb, c: &str, base: &str, depth: u32) -> bool {
    if depth > 16 {
        return false;
    }
    let Some(cls) = find_class(db, c) else { return false };
    cls.bases.iter().any(|b| b.offset == 0 && (norm_name(&b.name) == norm_name(base) || derives_from(db, &b.name, base, depth + 1)))
}

fn derived_vmethod<'a>(db: &'a TypeDb, base: &str, off: u32) -> Option<&'a mwdec_core::VirtualMethod> {
    for (n, c) in &db.classes {
        if !derives_from(db, n, base, 0) {
            continue;
        }
        if let Some(m) = c.vtable.iter().find(|m| m.vtable_offset == off) {
            if !crate::sig::split_scope(&m.sig.qualified_name).1.is_empty() {
                return Some(m);
            }
        }
    }
    None
}

/// Type of the value produced by an expression.
pub fn ty_of(e: &Expr, vars: &[Var]) -> Type {
    match e {
        Expr::Var(v) => vars.get(*v).map(|v| v.ty.clone()).unwrap_or(t_unk(4)),
        Expr::Int { ty, .. } => ty.clone(),
        Expr::Float { double, .. } => {
            if *double {
                t_f64()
            } else {
                t_f32()
            }
        }
        Expr::Str { .. } => t_ptr(Type::Const(Box::new(Type::Int { size: 1, signed: true }))),
        Expr::Global { ty, .. } => ty.clone(),
        Expr::FuncAddr { .. } => t_ptr(Type::Void),
        Expr::AddrOf(e) => {
            let t = ty_of(e, vars);
            match t {
                Type::Unknown { size: 0 } => t_ptr(Type::Void),
                // a call returning `T&` is an lvalue of type T
                Type::Ref(inner) => t_ptr(*inner),
                t => t_ptr(t),
            }
        }
        Expr::Load { ty, .. } | Expr::Index { ty, .. } | Expr::Member { ty, .. } => ty.clone(),
        Expr::Unary { ty, .. } | Expr::Binary { ty, .. } | Expr::Ternary { ty, .. } => ty.clone(),
        Expr::Cast { ty, .. } => ty.clone(),
        Expr::Call { ret, .. } => ret.clone(),
        Expr::Unknown { ty, .. } => ty.clone(),
        Expr::New { class, .. } => t_ptr(class.clone()),
        Expr::Construct { class, .. } => class.clone(),
        Expr::BitField { ty, .. } => ty.clone(),
        Expr::IncDec { e, .. } => ty_of(e, vars),
    }
}
