//! Constructors: constant stores at the start of a constructor body that the members' own
//! inline constructors make (a default-constructed optional wrapper clearing its flag, an
//! owning pointer set to null, the literal members of the constructor an initializer-list
//! entry names) are implicit in the member's construction, not body statements.
//!
//! The members' constructors come from the header declarations: an inline constructor whose
//! body is empty and whose initializer list sets members to literals (recursively for members
//! of class type that are not in its list).

use crate::util::strip;
use mwdec_core::{DeclInfo, Type, TypeDb};
use mwdec_lift::{Expr, InitTarget, IrFunction, Stmt, VarId, VarKind};
use std::collections::HashMap;

/// (offset, size, value) of a constant store; floats as their f32 bit pattern.
type Store = (i32, u32, i64);

/// `rstl::vector<T, A>` -> `rstl::vector` (every template argument list removed).
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
    out.trim().to_string()
}

/// Constructor declarations of class `cls`.
fn ctor_decls<'a>(db: &'a TypeDb, cls: &str) -> Option<&'a Vec<DeclInfo>> {
    let base = strip_tmpl(cls);
    let last = mwdec_lift::sig::split_scope(&base).1.to_string();
    db.decls.get(&format!("{base}::{last}"))
}

/// Value of a literal token list (`false`, `nullptr`, `- 1`, `0.f`) stored into a field of
/// type `ty`.
fn literal(toks: &[&str], ty: &Type) -> Option<i64> {
    let (neg, toks) = match toks {
        ["-", rest @ ..] => (true, rest),
        _ => (false, toks),
    };
    let [t] = toks else { return None };
    let is_float = matches!(strip(ty), Type::Float { .. });
    let v: i64 = match *t {
        "false" | "nullptr" | "NULL" => 0,
        "true" => 1,
        _ => {
            let s = t.trim_end_matches(|c| matches!(c, 'u' | 'U' | 'l' | 'L'));
            if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                i64::from_str_radix(h, 16).ok()?
            } else if let Ok(v) = s.parse::<i64>() {
                v
            } else if is_float {
                let f: f32 = t.trim_end_matches(|c| matches!(c, 'f' | 'F')).parse().ok()?;
                let f = if neg { -f } else { f };
                return Some(f.to_bits() as i64);
            } else {
                return None;
            }
        }
    };
    let v = if neg { -v } else { v };
    if is_float {
        // an integer literal converted to float
        return (*strip(ty) == Type::Float { size: 4 }).then(|| (v as f32).to_bits() as i64);
    }
    Some(v)
}

/// Entries of an initializer list (`a ( 0 ) , b ( x , y )`): (name tokens, argument tokens).
fn init_entries(list: &str) -> Vec<(String, Vec<&str>)> {
    let mut out = vec![];
    let toks: Vec<&str> = list.split_whitespace().collect();
    let mut i = 0;
    while i < toks.len() {
        let start = i;
        while i < toks.len() && toks[i] != "(" {
            i += 1;
        }
        let name = toks[start..i].join("");
        i += 1;
        let mut depth = 1;
        let args_start = i;
        while i < toks.len() && depth > 0 {
            match toks[i] {
                "(" => depth += 1,
                ")" => depth -= 1,
                _ => {}
            }
            i += 1;
        }
        let args = toks[args_start..i.saturating_sub(1).max(args_start)].to_vec();
        out.push((name, args));
        if i < toks.len() && toks[i] == "," {
            i += 1;
        }
    }
    out
}

/// Constant stores the inline constructor `d` of `cls` makes (offsets inside `cls`), counting
/// only literal-initialised members when `literal_only`; None when the constructor is not a
/// plain inline initializer list (a body, or non-literal entries without `literal_only`).
fn ctor_stores(db: &TypeDb, cls: &str, d: &DeclInfo, literal_only: bool, depth: u32) -> Option<Vec<Store>> {
    if depth > 6 || !d.is_inline_defined || d.inline_body.as_deref().is_some_and(|b| b.contains(';')) {
        return None;
    }
    let c = mwdec_lift::sig::find_class(db, cls)?;
    if c.vptr_offset.is_some() || c.is_union {
        return None;
    }
    let entries = d.init_list.as_deref().map(init_entries).unwrap_or_default();
    let mut out = vec![];
    for f in &c.fields {
        if f.bitfield.is_some() {
            if entries.iter().any(|(n, _)| *n == f.name) {
                return None;
            }
            continue;
        }
        let fty = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
        match entries.iter().find(|(n, _)| *n == f.name) {
            // an empty member (an allocator) copied from anything stores nothing
            Some(_) if empty_class(db, &fty) => {}
            Some((_, args)) => {
                let size = mwdec_lift::scalar_size(strip(&fty));
                match (size, literal(args, &fty)) {
                    (Some(s @ (1 | 2 | 4)), Some(v)) => out.push((f.offset as i32, s, v)),
                    _ if literal_only => {}
                    _ => return None,
                }
            }
            None => {
                // members of class type not in the list: their default constructors
                if let Some(mc) = crate::util::class_name(&fty, db) {
                    for (o, s, v) in default_stores(db, &mc, depth + 1)? {
                        out.push((f.offset as i32 + o, s, v));
                    }
                }
            }
        }
    }
    // base classes named in the list are not handled (their constructors may do anything)
    if c.bases.iter().any(|b| entries.iter().any(|(n, _)| strip_tmpl(n) == strip_tmpl(mwdec_lift::sig::split_scope(&b.name).1))) && !literal_only {
        return None;
    }
    Some(out)
}

/// A class without data (allocators, tags).
fn empty_class(db: &TypeDb, t: &Type) -> bool {
    let Some(c) = crate::util::class_name(t, db).and_then(|n| mwdec_lift::sig::find_class(db, &n).cloned()) else { return false };
    c.fields.is_empty() && c.vptr_offset.is_none() && !c.is_declaration && c.bases.iter().all(|b| empty_class(db, &Type::Named(b.name.clone())))
}

/// `const A&` with `A` an allocator: a class template parameter named like one, or an empty
/// class.
fn allocator_param(db: &TypeDb, d: &DeclInfo, t: &Type) -> bool {
    let Type::Ref(inner) = strip(t) else { return false };
    match strip(inner) {
        Type::Named(n) if d.template_params.iter().any(|p| p == n) => n.contains("Alloc"),
        inner => empty_class(db, inner),
    }
}

/// Constant stores of the default construction of `cls`: the inline default constructor's,
/// or for a class without declared constructors its members'. Empty when the construction
/// stores nothing (or is an out-of-line call); None when unknown.
fn default_stores(db: &TypeDb, cls: &str, depth: u32) -> Option<Vec<Store>> {
    let c = mwdec_lift::sig::find_class(db, cls)?;
    if c.vptr_offset.is_some() || depth > 6 {
        return None;
    }
    match ctor_decls(db, cls) {
        Some(ds) => {
            // (or a constructor taking only an allocator, defaulted: `C(const Alloc& a = Alloc())`)
            let d = ds.iter().find(|d| d.params.is_empty()).or_else(|| ds.iter().find(|d| d.params.len() == 1 && allocator_param(db, d, &d.params[0].ty)))?;
            if !d.is_inline_defined {
                return Some(vec![]);
            }
            ctor_stores(db, cls, d, false, depth)
        }
        None => {
            if !c.bases.is_empty() {
                return None;
            }
            let mut out = vec![];
            for f in &c.fields {
                let fty = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
                if let Some(mc) = crate::util::class_name(&fty, db) {
                    for (o, s, v) in default_stores(db, &mc, depth + 1)? {
                        out.push((f.offset as i32 + o, s, v));
                    }
                }
            }
            Some(out)
        }
    }
}

/// Offset from `this` of the object a pointer expression designates.
fn ptr_off(e: &Expr, this: VarId) -> Option<i32> {
    match e {
        Expr::Var(v) if *v == this => Some(0),
        Expr::Cast { e, .. } => ptr_off(e, this),
        Expr::AddrOf(x) => lv_off(x, this),
        _ => None,
    }
}

fn lv_off(e: &Expr, this: VarId) -> Option<i32> {
    match e {
        Expr::Load { base, offset, .. } => Some(ptr_off(base, this)? + offset),
        Expr::Member { base, offset, .. } => Some(lv_off(base, this)? + offset),
        _ => None,
    }
}

fn const_value(e: &Expr) -> Option<i64> {
    match e {
        Expr::Int { value, .. } => Some(*value),
        // single floats carry their f32 bit pattern
        Expr::Float { bits, double: false } => Some(*bits as i64),
        Expr::Cast { e, .. } => const_value(e),
        _ => None,
    }
}

/// Remove the leading body stores of a constructor that its members' inline constructors
/// make; returns the number of removed statements.
pub fn strip_member_ctor_stores(ir: &mut IrFunction, db: &TypeDb) -> usize {
    if !ir.symbol.starts_with("__ct__") {
        return 0;
    }
    let Some(this) = ir.vars.iter().position(|v| v.kind == VarKind::This) else { return 0 };
    let Some(own) = ir.sig.this_class.clone() else { return 0 };
    let Some(c) = mwdec_lift::sig::find_class(db, &own).cloned() else { return 0 };
    // leading member stores: constants by offset (first store wins), other pure stores skipped
    let mut consts: HashMap<i32, (usize, u32, i64)> = HashMap::new();
    for (i, s) in ir.body.iter().enumerate() {
        let Stmt::Assign { dst, src } = s else { break };
        if src.has_call() {
            break;
        }
        let Some(off) = lv_off(dst, this) else { break };
        let size = match dst {
            Expr::Load { ty, .. } | Expr::Member { ty, .. } => mwdec_lift::scalar_size(strip(ty)),
            _ => None,
        };
        if let (Some(size), Some(v)) = (size, const_value(src)) {
            consts.entry(off).or_insert((i, size, v));
        }
    }
    if consts.is_empty() {
        return 0;
    }
    // (offset, stores) of each subobject's own construction: non-virtual bases not named in the
    // initializer list (default-constructed), then the members
    let mut parts: Vec<(i32, Vec<Store>)> = vec![];
    for b in c.bases.iter().filter(|b| !b.is_virtual) {
        let named = ir.init_list.iter().any(|i| matches!(&i.target, InitTarget::Base(n) if mwdec_lift::sig::norm_name(n) == mwdec_lift::sig::norm_name(&b.name)));
        if !named {
            if let Some(st) = default_stores(db, &b.name, 0) {
                parts.push((b.offset as i32, st));
            }
        }
    }
    for f in &c.fields {
        if f.bitfield.is_some() {
            continue;
        }
        let fty = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
        let Some(mc) = crate::util::class_name(&fty, db) else { continue };
        let init = ir.init_list.iter().find(|i| i.target == InitTarget::Member(f.name.clone()));
        let stores = match init {
            None => default_stores(db, &mc, 0),
            Some(i) => {
                // the inline constructor with that many parameters, if only one
                let cands: Vec<&DeclInfo> = ctor_decls(db, &mc).map(|ds| ds.iter().filter(|d| d.params.len() == i.args.len() && d.is_inline_defined).collect()).unwrap_or_default();
                match cands.as_slice() {
                    [d] if !i.args.is_empty() => ctor_stores(db, &mc, d, true, 0),
                    _ => None,
                }
            }
        };
        if let Some(st) = stores {
            parts.push((f.offset as i32, st));
        }
    }
    let mut remove: Vec<usize> = vec![];
    for (base_off, stores) in parts {
        if stores.is_empty() {
            continue;
        }
        let hits: Option<Vec<usize>> = stores
            .iter()
            .map(|(o, s, v)| {
                let (i, size, val) = consts.get(&(base_off + o))?;
                let mask = if *s >= 8 { -1i64 } else { (1i64 << (s * 8)) - 1 };
                (size == s && (val & mask) == (v & mask)).then_some(*i)
            })
            .collect();
        if let Some(h) = hits {
            remove.extend(h);
        }
    }
    remove.sort_unstable();
    remove.dedup();
    for &i in remove.iter().rev() {
        ir.body.remove(i);
    }
    remove.len()
}

