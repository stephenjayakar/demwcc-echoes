//! Sizes and typed member lookup (`field_at`) over a TypeDb.
use crate::resolve::canonical;
use mwdec_core::*;

/// sizeof(t) in bytes, if known.
pub fn size_of(db: &TypeDb, t: &Type) -> Option<u32> {
    size_rec(db, t, 0)
}

fn size_rec(db: &TypeDb, t: &Type, depth: u32) -> Option<u32> {
    if depth > 32 {
        return None;
    }
    Some(match t {
        Type::Void => 0,
        Type::Bool | Type::Char => 1,
        Type::WChar => 2,
        Type::Long { .. } => 4,
        Type::Int { size, .. } => *size as u32,
        Type::Float { size } => *size as u32,
        Type::Ptr(_) | Type::Ref(_) | Type::FuncPtr(_) => 4,
        Type::Const(x) | Type::Volatile(x) => size_rec(db, x, depth + 1)?,
        Type::Array(x, n) => size_rec(db, x, depth + 1)? * n,
        Type::MemberPtr { size, .. } => *size,
        Type::Unknown { size } => *size,
        Type::Named(n) => {
            if let Some(c) = db.classes.get(n) {
                if c.is_declaration {
                    return None;
                }
                c.size
            } else if let Some(e) = db.enums.get(n) {
                e.size
            } else if let Some(u) = db.typedefs.get(n) {
                size_rec(db, u, depth + 1)?
            } else {
                return None;
            }
        }
    })
}

/// Class definition behind a type (through typedefs and cv), if it is a class/struct/union.
pub fn class_of<'a>(db: &'a TypeDb, t: &Type) -> Option<&'a Class> {
    match canonical(db, t).unqualified() {
        Type::Named(n) => db.classes.get(n),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PathStep {
    /// Implicit conversion to a base class subobject (no syntax in C++ member access).
    Base(String),
    /// `.name` (empty name = anonymous struct/union member, no syntax).
    Field(String),
    /// `[i]`
    Index(u32),
}

#[derive(Clone, Debug)]
pub struct FieldAccess {
    /// C++ member access path from the object, e.g. `mTransform.m[1][3]` / `x34_xf.m03`.
    pub path: String,
    pub steps: Vec<PathStep>,
    /// Type of the leaf member.
    pub ty: Type,
    /// Byte offset of the access inside the leaf (non-zero for partial accesses).
    pub offset_in_leaf: u32,
    /// Leaf size equals the access size (or the access size was 0 = don't care).
    pub exact: bool,
    /// Leaf is a bitfield: (bit offset from msb of storage unit, bit size).
    pub bitfield: Option<(u8, u8)>,
}

/// Every member access path that covers `[offset, offset+access_size)` of `class`.
/// Paths descend through bases, nested structs and arrays down to scalar leaves; unions
/// return one result per alternative. Exact matches come first, then deeper paths.
/// `access_size == 0` means any size (e.g. address-of / aggregate access).
pub fn field_at(db: &TypeDb, class: &str, offset: u32, access_size: u32) -> Vec<FieldAccess> {
    let mut out = Vec::new();
    let Some(c) = db.classes.get(class) else { return out };
    walk_class(db, c, offset, access_size, &mut Vec::new(), &mut out, 0);
    out.sort_by(|a, b| {
        b.exact
            .cmp(&a.exact)
            .then(a.offset_in_leaf.cmp(&b.offset_in_leaf))
            .then(b.steps.len().cmp(&a.steps.len()))
    });
    out
}

fn path_string(steps: &[PathStep]) -> String {
    let mut s = String::new();
    for st in steps {
        match st {
            PathStep::Base(_) => {}
            PathStep::Field(n) if n.is_empty() => {}
            PathStep::Field(n) => {
                if !s.is_empty() {
                    s.push('.');
                }
                s.push_str(n);
            }
            PathStep::Index(i) => s.push_str(&format!("[{i}]")),
        }
    }
    s
}

fn walk_class(db: &TypeDb, c: &Class, off: u32, size: u32, steps: &mut Vec<PathStep>, out: &mut Vec<FieldAccess>, depth: u32) {
    if depth > 24 {
        return;
    }
    for b in &c.bases {
        let Some(bc) = db.classes.get(&b.name) else { continue };
        if off >= b.offset && off < b.offset + bc.size.max(1) {
            steps.push(PathStep::Base(b.name.clone()));
            walk_class(db, bc, off - b.offset, size, steps, out, depth + 1);
            steps.pop();
        }
    }
    for f in &c.fields {
        let fsize = if f.bitfield.is_some() && f.size > 0 {
            f.size
        } else {
            size_of(db, &f.ty).unwrap_or(f.size)
        };
        if fsize == 0 {
            continue;
        }
        if off < f.offset || off >= f.offset + fsize {
            continue;
        }
        steps.push(PathStep::Field(f.name.clone()));
        if let Some(bf) = f.bitfield {
            out.push(FieldAccess {
                path: path_string(steps),
                steps: steps.clone(),
                ty: f.ty.clone(),
                offset_in_leaf: off - f.offset,
                exact: size == 0 || (off == f.offset && size == fsize),
                bitfield: Some(bf),
            });
        } else {
            walk_type(db, &f.ty, off - f.offset, size, steps, out, depth + 1);
        }
        steps.pop();
    }
}

fn walk_type(db: &TypeDb, t: &Type, off: u32, size: u32, steps: &mut Vec<PathStep>, out: &mut Vec<FieldAccess>, depth: u32) {
    let ct = canonical(db, t);
    let leaf_size = size_of(db, &ct).unwrap_or(0);
    match ct.unqualified() {
        Type::Array(elem, n) => {
            let esz = size_of(db, elem).unwrap_or(0);
            if esz > 0 && *n > 0 {
                let idx = off / esz;
                if idx < *n {
                    steps.push(PathStep::Index(idx));
                    walk_type(db, elem, off % esz, size, steps, out, depth + 1);
                    steps.pop();
                    // whole-array access (e.g. address of the array)
                    if off == 0 && (size == 0 || size == leaf_size) {
                        push_leaf(t, 0, size, leaf_size, steps, out);
                    }
                    return;
                }
            }
            push_leaf(t, off, size, leaf_size, steps, out);
        }
        Type::Named(n) if db.classes.get(n).is_some_and(|c| !c.is_declaration) => {
            let c = &db.classes[n];
            let before = out.len();
            walk_class(db, c, off, size, steps, out, depth + 1);
            // the aggregate itself (struct copy / address-of / no member matched)
            if (off == 0 && (size == 0 || size == leaf_size)) || out.len() == before {
                push_leaf(t, off, size, leaf_size, steps, out);
            }
        }
        _ => push_leaf(t, off, size, leaf_size, steps, out),
    }
}

fn push_leaf(t: &Type, off: u32, size: u32, leaf_size: u32, steps: &[PathStep], out: &mut Vec<FieldAccess>) {
    out.push(FieldAccess {
        path: path_string(steps),
        steps: steps.to_vec(),
        ty: t.clone(),
        offset_in_leaf: off,
        exact: off == 0 && (size == 0 || size == leaf_size),
        bitfield: None,
    });
}

/// Best single access (first of `field_at`), if exact.
pub fn field_exact(db: &TypeDb, class: &str, offset: u32, access_size: u32) -> Option<FieldAccess> {
    field_at(db, class, offset, access_size).into_iter().find(|a| a.exact)
}

/// `field_at` for an object reached through a value of type `t` (class, or pointer/reference
/// to class, through typedefs and cv): the lifter's `*(T*)(p + off)` lookup.
pub fn field_at_type(db: &TypeDb, t: &Type, offset: u32, access_size: u32) -> Vec<FieldAccess> {
    let ct = canonical(db, t);
    let inner = match ct.unqualified() {
        Type::Ptr(x) | Type::Ref(x) => x.unqualified().clone(),
        other => other.clone(),
    };
    match inner {
        Type::Named(n) => field_at(db, &n, offset, access_size),
        _ => Vec::new(),
    }
}

