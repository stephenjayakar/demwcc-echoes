//! DWARF 1.1 DIEs (MWCC flavour) -> `TypeDb`.
use crate::dwarf::{at, tag, AttrVal, Die, DwarfInfo};
use crate::mangle::{demangle_qualified, split_scope};
use mwdec_core::*;
use std::collections::HashMap;

/// Prefix of the forcing declarations we inject into the context TU; never exported.
pub const DUMMY_PREFIX: &str = "__mwdec_";

pub struct Converter<'a> {
    info: &'a DwarfInfo,
    /// DIE offset -> qualified type name (classes, unions, enums, typedefs).
    names: HashMap<u32, String>,
    /// typedef DIEs that just name an anonymous struct (`typedef struct {..} X;`)
    identity_typedefs: std::collections::HashSet<u32>,
    c_lang: bool,
    /// (class, field) whose DIE had no access attribute (filled from the header scan)
    pub missing_access: std::cell::RefCell<std::collections::HashSet<(String, String)>>,
}

/// Access attribute of a member/inheritance DIE, if any.
pub fn die_access(d: &Die) -> Option<Access> {
    if d.has(at::PRIVATE) {
        Some(Access::Private)
    } else if d.has(at::PROTECTED) {
        Some(Access::Protected)
    } else if d.has(at::PUBLIC) {
        Some(Access::Public)
    } else {
        None
    }
}

fn is_class_tag(t: u16) -> bool {
    matches!(t, tag::CLASS_TYPE | tag::STRUCTURE_TYPE | tag::UNION_TYPE)
}

fn fund_type(v: u16) -> Type {
    // MWCC's `long long` codes are vendor values whose low byte collides with FT_integer
    match v {
        0x8008 | 0x8108 => return Type::Int { size: 8, signed: true },
        0x8208 => return Type::Int { size: 8, signed: false },
        _ => {}
    }
    match v & 0xff {
        0x00 => Type::WChar,
        0x01 => Type::Char,
        0x02 => Type::Int { size: 1, signed: true },
        0x03 => Type::Int { size: 1, signed: false },
        0x04 | 0x05 => Type::Int { size: 2, signed: true },
        0x06 => Type::Int { size: 2, signed: false },
        0x07 | 0x08 => Type::Int { size: 4, signed: true },
        0x09 => Type::Int { size: 4, signed: false },
        0x0a | 0x0b => Type::Long { signed: true },
        0x0c => Type::Long { signed: false },
        0x0d => Type::Ptr(Box::new(Type::Void)),
        0x0e => Type::Float { size: 4 },
        0x0f | 0x10 => Type::Float { size: 8 },
        0x14 => Type::Void,
        0x15 => Type::Bool,
        _ => match v {
            0x8008 | 0x8108 => Type::Int { size: 8, signed: true },
            0x8208 => Type::Int { size: 8, signed: false },
            _ => Type::Unknown { size: 4 },
        },
    }
}

/// Offset from a member/inheritance location block: OP_CONST <u32> OP_ADD.
pub fn location_offset(b: &[u8]) -> Option<u32> {
    if b.len() == 6 && b[0] == 0x04 && b[5] == 0x07 {
        Some(u32::from_be_bytes([b[1], b[2], b[3], b[4]]))
    } else {
        None
    }
}

impl<'a> Converter<'a> {
    pub fn new(info: &'a DwarfInfo) -> Self {
        // C (DW_LANG 1/2): one tag namespace, nested struct names are global
        let c_lang = info.dies.iter().find(|d| d.tag == tag::COMPILE_UNIT).and_then(|d| d.data4(0x0136)).is_some_and(|l| l != 4);
        let mut c = Converter { info, names: HashMap::new(), identity_typedefs: Default::default(), c_lang, missing_access: Default::default() };
        c.assign_names();
        c
    }

    fn assign_names(&mut self) {
        let roots = self.info.roots();
        let mut anon = 0u32;
        for &r in &roots {
            self.name_rec(r, None, &mut anon);
        }
        // `typedef struct { ... } Name;` -> name the anonymous struct after the typedef.
        for &r in &roots {
            let d = &self.info.dies[r];
            if d.tag != tag::TYPEDEF {
                continue;
            }
            let (Some(n), Some(target)) = (d.name(), d.reference(at::USER_DEF_TYPE)) else { continue };
            if let Some(td) = self.info.get(target) {
                if is_class_tag(td.tag) || td.tag == tag::ENUMERATION_TYPE {
                    let cur = self.names.get(&target).cloned().unwrap_or_default();
                    if cur.split("::").last().is_some_and(|l| l.starts_with('@')) {
                        self.names.insert(target, n.to_string());
                        self.identity_typedefs.insert(d.off);
                    }
                }
            }
        }
    }

    fn name_rec(&mut self, i: usize, scope: Option<&str>, anon: &mut u32) {
        let d = &self.info.dies[i];
        let named_kind = is_class_tag(d.tag) || d.tag == tag::ENUMERATION_TYPE || d.tag == tag::TYPEDEF;
        if !named_kind {
            return;
        }
        // MWCC names anonymous classes `@class$<n><file>`: treat as anonymous
        let q = match d.name().filter(|n| !n.starts_with('@')) {
            Some(simple) => {
                if let Some(sym) = self.info.sym_names.get(&d.off) {
                    // "<qualified type>::<simple>" for classes; enums/typedefs: just qualified
                    let parts = split_scope(sym);
                    let q = if parts.len() >= 2 && parts.last() == Some(&simple) && is_class_tag(d.tag) {
                        parts[..parts.len() - 1].join("::")
                    } else {
                        sym.clone()
                    };
                    demangle_qualified(&q)
                } else {
                    match scope {
                        Some(s) if !self.c_lang => format!("{s}::{simple}"),
                        _ => simple.to_string(),
                    }
                }
            }
            None => {
                *anon += 1;
                match scope {
                    Some(s) => format!("{s}::@anon{anon}"),
                    None => format!("@anon{anon}"),
                }
            }
        };
        self.names.insert(d.off, q.clone());
        if is_class_tag(d.tag) {
            for c in self.info.children(i) {
                self.name_rec(c, Some(&q), anon);
            }
        }
    }

    /// Type of a DIE's type attribute (FundType/ModFundType/UserDefType/ModUDType).
    pub fn die_type(&self, d: &Die) -> Option<Type> {
        for (k, v) in &d.attrs {
            match (*k, v) {
                (at::FUND_TYPE, AttrVal::Data2(f)) => return Some(fund_type(*f)),
                (at::MOD_FUND_TYPE, AttrVal::Block(b)) if b.len() >= 2 => {
                    let f = u16::from_be_bytes([b[b.len() - 2], b[b.len() - 1]]);
                    return Some(apply_mods(&b[..b.len() - 2], fund_type(f)));
                }
                (at::USER_DEF_TYPE, AttrVal::Ref(r)) => return Some(self.ref_type(*r)),
                (at::MOD_UD_TYPE, AttrVal::Block(b)) if b.len() >= 4 => {
                    let n = b.len();
                    let r = u32::from_be_bytes([b[n - 4], b[n - 3], b[n - 2], b[n - 1]]);
                    return Some(apply_mods(&b[..n - 4], self.ref_type(r)));
                }
                _ => {}
            }
        }
        None
    }

    fn ref_type(&self, off: u32) -> Type {
        let Some(&idx) = self.info.index.get(&off) else { return Type::Unknown { size: 4 } };
        let d = &self.info.dies[idx];
        match d.tag {
            t if is_class_tag(t) || t == tag::ENUMERATION_TYPE => {
                Type::Named(self.names.get(&off).cloned().unwrap_or_else(|| format!("@die{off:x}")))
            }
            tag::TYPEDEF => {
                if self.identity_typedefs.contains(&off) {
                    if let Some(t) = d.reference(at::USER_DEF_TYPE) {
                        return self.ref_type(t);
                    }
                }
                Type::Named(self.names.get(&off).cloned().unwrap_or_else(|| d.name().unwrap_or("?").to_string()))
            }
            tag::ARRAY_TYPE => self.array_type(d).unwrap_or(Type::Unknown { size: 0 }),
            tag::SUBROUTINE_TYPE | tag::GLOBAL_SUBROUTINE | tag::SUBROUTINE => {
                Type::FuncPtr(Box::new(self.subroutine_sig(idx, String::new())))
            }
            tag::PTR_TO_MEMBER_TYPE => {
                let class = d
                    .reference(at::CONTAINING_TYPE)
                    .and_then(|c| self.names.get(&c).cloned())
                    .unwrap_or_default();
                let is_fn = matches!(self.die_type(d), Some(Type::FuncPtr(_)));
                Type::MemberPtr { class, size: if is_fn { 12 } else { 4 } }
            }
            _ => Type::Unknown { size: 4 },
        }
    }

    fn array_type(&self, d: &Die) -> Option<Type> {
        let b = d.block(at::SUBSCR_DATA)?;
        let mut dims: Vec<u32> = Vec::new();
        let mut elem = None;
        let mut p = 0usize;
        while p < b.len() {
            let fmt = b[p];
            p += 1;
            match fmt {
                0x0 => {
                    // FT_C_C: fund index type, lo, hi
                    let hi = u32::from_be_bytes(b.get(p + 6..p + 10)?.try_into().ok()?);
                    p += 10;
                    dims.push(hi.wrapping_add(1));
                }
                0x1 => {
                    // FT_C_X: fund, lo, location block (unknown bound)
                    let l = u16::from_be_bytes(b.get(p + 6..p + 8)?.try_into().ok()?) as usize;
                    p += 8 + l;
                    dims.push(0);
                }
                0x8 => {
                    // FMT_ET: element type attribute
                    let a = u16::from_be_bytes(b.get(p..p + 2)?.try_into().ok()?);
                    p += 2;
                    let (val, len) = match a & 0xf {
                        0x2 => (AttrVal::Ref(u32::from_be_bytes(b.get(p..p + 4)?.try_into().ok()?)), 4),
                        0x5 => (AttrVal::Data2(u16::from_be_bytes(b.get(p..p + 2)?.try_into().ok()?)), 2),
                        0x3 => {
                            let l = u16::from_be_bytes(b.get(p..p + 2)?.try_into().ok()?) as usize;
                            (AttrVal::Block(b.get(p + 2..p + 2 + l)?.to_vec()), 2 + l)
                        }
                        _ => return None,
                    };
                    p += len;
                    let fake = Die { off: 0, size: 0, tag: 0, attrs: vec![(a, val)] };
                    elem = self.die_type(&fake);
                }
                _ => return None,
            }
        }
        let mut t = elem?;
        for &n in dims.iter().rev() {
            t = Type::Array(Box::new(t), n);
        }
        Some(t)
    }

    fn subroutine_sig(&self, idx: usize, qualified_name: String) -> FuncSig {
        let d = &self.info.dies[idx];
        let ret = self.die_type(d).unwrap_or(Type::Void);
        let mut params = Vec::new();
        let mut variadic = false;
        for c in self.info.children(idx) {
            let cd = &self.info.dies[c];
            match cd.tag {
                tag::FORMAL_PARAMETER => params.push(Param {
                    name: cd.name().map(|s| s.to_string()),
                    ty: self.die_type(cd).unwrap_or(Type::Unknown { size: 4 }),
                }),
                tag::UNSPECIFIED_PARAMETERS => variadic = true,
                _ => {}
            }
        }
        let mangled = d.string(at::MW_MANGLED).map(|s| s.to_string());
        let this_class = d.reference(at::MEMBER).and_then(|m| self.names.get(&m).cloned());
        // drop explicit `this` parameter
        if params.first().and_then(|p| p.name.as_deref()) == Some("this") {
            params.remove(0);
        }
        FuncSig {
            qualified_name,
            mangled,
            ret,
            params,
            this_class,
            is_const: false,
            is_static: false,
            is_virtual: d.has(at::VIRTUAL),
            variadic,
            runs_code: false,
        }
    }

    /// Convert everything into a TypeDb.
    pub fn build(&self) -> TypeDb {
        self.build_with_dummies().0
    }

    /// Like `build`, also returning the types of our injected `__mwdec_*` globals.
    pub fn build_with_dummies(&self) -> (TypeDb, std::collections::BTreeMap<String, Type>) {
        let mut db = TypeDb::default();
        let mut dummies = std::collections::BTreeMap::new();
        let roots = self.info.roots();
        for &r in &roots {
            let d = &self.info.dies[r];
            if matches!(d.tag, tag::GLOBAL_VARIABLE | tag::LOCAL_VARIABLE) {
                if let Some(n) = d.name().filter(|n| n.starts_with(DUMMY_PREFIX)) {
                    if let Some(t) = self.die_type(d) {
                        dummies.insert(n.to_string(), t);
                    }
                    continue;
                }
            }
            self.visit(r, &mut db);
        }
        compute_vptr_offsets(&mut db);
        (db, dummies)
    }

    fn visit(&self, i: usize, db: &mut TypeDb) {
        let d = &self.info.dies[i];
        match d.tag {
            t if is_class_tag(t) => {
                let c = self.class(i);
                for ch in self.info.children(i) {
                    let cd = &self.info.dies[ch];
                    if is_class_tag(cd.tag) || cd.tag == tag::ENUMERATION_TYPE {
                        self.visit(ch, db);
                    }
                }
                let keep_old = db.classes.get(&c.name).is_some_and(|o| !o.is_declaration && c.is_declaration);
                if !keep_old {
                    db.classes.insert(c.name.clone(), c);
                }
            }
            tag::ENUMERATION_TYPE => {
                let name = self.names.get(&d.off).cloned().unwrap_or_default();
                let size = d.data4(at::BYTE_SIZE).unwrap_or(4);
                let mut values = Vec::new();
                if let Some(b) = d.block(at::ELEMENT_LIST) {
                    let mut p = 0usize;
                    let sz = size as usize;
                    while p + sz <= b.len() {
                        let v: i64 = match sz {
                            1 => b[p] as i8 as i64,
                            2 => i16::from_be_bytes([b[p], b[p + 1]]) as i64,
                            _ => i32::from_be_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]) as i64,
                        };
                        p += sz;
                        let e = b[p..].iter().position(|&x| x == 0).map(|x| p + x).unwrap_or(b.len());
                        let n: String = b[p..e].iter().map(|&x| x as char).collect();
                        p = e + 1;
                        values.push((n, v));
                    }
                }
                db.enums.insert(name.clone(), Enum { name, size, values });
            }
            tag::TYPEDEF => {
                if self.identity_typedefs.contains(&d.off) {
                    return;
                }
                let name = self.names.get(&d.off).cloned().unwrap_or_else(|| d.name().unwrap_or("").to_string());
                if name.starts_with(DUMMY_PREFIX) {
                    return;
                }
                if let Some(t) = self.die_type(d) {
                    db.typedefs.insert(name, t);
                }
            }
            tag::GLOBAL_VARIABLE | tag::LOCAL_VARIABLE => {
                let Some(name) = d.name() else { return };
                if name.starts_with(DUMMY_PREFIX) {
                    return;
                }
                let key = d.string(at::MW_MANGLED).unwrap_or(name).to_string();
                if let Some(t) = self.die_type(d) {
                    db.globals.insert(key, (name.to_string(), t));
                }
            }
            tag::GLOBAL_SUBROUTINE | tag::SUBROUTINE => {
                let Some(name) = d.name() else { return };
                if name.starts_with(DUMMY_PREFIX) {
                    return;
                }
                let mut sig = self.subroutine_sig(i, String::new());
                sig.qualified_name = match &sig.this_class {
                    Some(c) => format!("{c}::{name}"),
                    None => name.to_string(),
                };
                let key = sig.mangled.clone().unwrap_or_else(|| sig.qualified_name.clone());
                db.functions.insert(key, sig);
            }
            _ => {}
        }
    }

    fn class(&self, i: usize) -> Class {
        let d = &self.info.dies[i];
        let name = self.names.get(&d.off).cloned().unwrap_or_default();
        let size = d.data4(at::BYTE_SIZE).unwrap_or(0);
        let mut c = Class {
            name,
            size,
            is_union: d.tag == tag::UNION_TYPE,
            is_struct: d.tag == tag::STRUCTURE_TYPE,
            ..Default::default()
        };
        let children = self.info.children(i);
        for ch in &children {
            let cd = &self.info.dies[*ch];
            match cd.tag {
                tag::INHERITANCE => {
                    let bname = match self.die_type(cd) {
                        Some(Type::Named(n)) => n,
                        Some(t) => crate::mangle::type_to_string(&t),
                        None => String::new(),
                    };
                    let offset = cd.block(at::LOCATION).and_then(location_offset).unwrap_or(0);
                    // C++ default for a base without an attribute: private for `class`, public for `struct`
                    let default = if d.tag == tag::CLASS_TYPE { Access::Private } else { Access::Public };
                    let access = die_access(cd).unwrap_or(default);
                    c.bases.push(BaseClass { name: bname, offset, is_virtual: cd.has(at::VIRTUAL), access });
                }
                tag::MEMBER => {
                    let fname = cd.name().unwrap_or("").to_string();
                    let ty = self.die_type(cd).unwrap_or(Type::Unknown { size: 0 });
                    let offset = cd.block(at::LOCATION).and_then(location_offset).unwrap_or(0);
                    let bitfield = match (cd.data2(at::BIT_OFFSET), cd.data4(at::BIT_SIZE)) {
                        (Some(o), Some(s)) => Some((o as u8, s as u8)),
                        _ => None,
                    };
                    let fsize = cd.data4(at::BYTE_SIZE).unwrap_or(0);
                    if fname == "__vptr$" {
                        c.vptr_offset = Some(offset);
                    }
                    let access = match die_access(cd) {
                        Some(a) => a,
                        None => {
                            self.missing_access.borrow_mut().insert((c.name.clone(), fname.clone()));
                            Access::Public
                        }
                    };
                    c.fields.push(Field { name: fname, offset, ty, bitfield, size: fsize, access });
                }
                tag::TYPEDEF => {
                    // MWCC: static data members are emitted as typedef-tagged children
                    if let (Some(n), Some(t)) = (cd.name(), self.die_type(cd)) {
                        c.statics.push(StaticMember { name: n.to_string(), ty: t, access: die_access(cd).unwrap_or_default() });
                    }
                }
                _ => {}
            }
        }
        c.is_declaration = size == 0 && c.fields.is_empty() && c.bases.is_empty();
        c
    }
}

fn apply_mods(mods: &[u8], base: Type) -> Type {
    let mut t = base;
    for &m in mods.iter().rev() {
        t = match m & 0x7f {
            0x00 | 0x01 => match t {
                // pointer to function type = function pointer
                Type::FuncPtr(_) => t,
                _ => Type::Ptr(Box::new(t)),
            },
            0x02 => Type::Ref(Box::new(t)),
            0x03 => Type::Const(Box::new(t)),
            0x04 => Type::Volatile(Box::new(t)),
            _ => t,
        };
    }
    t
}

/// Inherit vptr offsets: a class without its own `__vptr$` uses its first polymorphic base's.
pub fn compute_vptr_offsets(db: &mut TypeDb) {
    let names: Vec<String> = db.classes.keys().cloned().collect();
    fn get(db: &TypeDb, n: &str, depth: u32) -> Option<u32> {
        if depth > 32 {
            return None;
        }
        let c = db.classes.get(n)?;
        if let Some(f) = c.fields.iter().find(|f| f.name == "__vptr$") {
            return Some(f.offset);
        }
        for b in &c.bases {
            if let Some(o) = get(db, &b.name, depth + 1) {
                return Some(b.offset + o);
            }
        }
        None
    }
    for n in names {
        let v = get(db, &n, 0);
        if let Some(c) = db.classes.get_mut(&n) {
            c.vptr_offset = v;
        }
    }
}

/// Rename types (and nested `Old::...` names) throughout the db.
pub fn rename_types(db: &mut TypeDb, map: &std::collections::BTreeMap<String, String>) {
    let ren = |n: &str| -> Option<String> {
        if let Some(v) = map.get(n) {
            return Some(v.clone());
        }
        for (k, v) in map {
            if let Some(rest) = n.strip_prefix(k.as_str()) {
                if rest.starts_with("::") {
                    return Some(format!("{v}{rest}"));
                }
            }
        }
        None
    };
    fn fix(t: &mut Type, ren: &dyn Fn(&str) -> Option<String>) {
        match t {
            Type::Named(n) => {
                if let Some(r) = ren(n) {
                    *n = r;
                }
            }
            Type::Ptr(x) | Type::Ref(x) | Type::Const(x) | Type::Volatile(x) | Type::Array(x, _) => fix(x, ren),
            Type::FuncPtr(s) => {
                fix(&mut s.ret, ren);
                for p in &mut s.params {
                    fix(&mut p.ty, ren);
                }
            }
            Type::MemberPtr { class, .. } => {
                if let Some(r) = ren(class) {
                    *class = r;
                }
            }
            _ => {}
        }
    }
    let classes = std::mem::take(&mut db.classes);
    for (k, mut c) in classes {
        let nk = ren(&k).unwrap_or(k);
        c.name = nk.clone();
        for f in &mut c.fields {
            fix(&mut f.ty, &ren);
        }
        for s in &mut c.statics {
            fix(&mut s.ty, &ren);
        }
        for b in &mut c.bases {
            if let Some(r) = ren(&b.name) {
                b.name = r;
            }
        }
        db.classes.insert(nk, c);
    }
    let enums = std::mem::take(&mut db.enums);
    for (k, mut e) in enums {
        let nk = ren(&k).unwrap_or(k);
        e.name = nk.clone();
        db.enums.insert(nk, e);
    }
    for t in db.typedefs.values_mut() {
        fix(t, &ren);
    }
    for (_, t) in db.globals.values_mut() {
        fix(t, &ren);
    }
}
