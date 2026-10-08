//! Virtual tables from the target object's data: `__vt__<class>` symbols with ADDR32 relocs.
//!
//! MWCC GC layout (verified on CEntity/CActor/CCubeRenderer target objects): an 8-byte header
//! (RTTI pointer + this-delta, both 0 with `-RTTI off`) followed by one 4-byte slot per virtual
//! function in declaration order (base class slots first, the destructor in slot 0 of the
//! root). With multiple inheritance, secondary tables follow in the same symbol, each with its
//! own 8-byte (zero) header; their slots point to `@<delta>@<mangled>` this-adjusting thunks.
//! dtk sizes symbols up to the next symbol, so trailing zero words are padding.
use crate::mangle::{demangle_qualified, P};
use crate::sig_from_mangled;
use mwdec_core::*;
use std::collections::BTreeMap;

/// Class name of a vtable symbol (`__vt__6CActor` -> `CActor`).
pub fn vtable_class(sym: &str) -> Option<String> {
    let rest = sym.strip_prefix("__vt__")?;
    let mut p = P::new(rest);
    let q = p.qualified_name()?;
    if !p.at_end() {
        return None;
    }
    Some(demangle_qualified(&q))
}

/// Split `@4@__dt__13CCubeRendererFv` -> (4, `__dt__13CCubeRendererFv`).
pub fn split_thunk(sym: &str) -> Option<(i32, &str)> {
    let rest = sym.strip_prefix('@')?;
    let at = rest.find('@')?;
    let n: i32 = rest[..at].parse().ok()?;
    Some((n, &rest[at + 1..]))
}

fn placeholder_sig(name: &str) -> FuncSig {
    FuncSig {
        qualified_name: name.to_string(),
        mangled: None,
        ret: Type::Unknown { size: 0 },
        params: Vec::new(),
        this_class: None,
        is_const: false,
        is_static: false,
        is_virtual: true,
        variadic: false,
    }
}

/// Every vtable defined in `obj`: class -> slots (vtable_offset = byte offset in the symbol).
/// Empty interior slots of the primary table are reported with an empty `symbol`
/// (pure virtual / unknown).
pub fn vtables_from_object(obj: &ObjectFile, db: &TypeDb) -> BTreeMap<String, Vec<VirtualMethod>> {
    let mut out = BTreeMap::new();
    for (name, d) in &obj.data {
        let Some(class) = vtable_class(name) else { continue };
        let mut slot_targets: BTreeMap<u32, &Reloc> = BTreeMap::new();
        for r in &d.relocs {
            if r.kind == RelocKind::Addr32 && r.offset % 4 == 0 {
                slot_targets.insert(r.offset, r);
            }
        }
        let last = slot_targets.keys().next_back().copied().unwrap_or(0);
        let mut slots = Vec::new();
        let mut off = 8;
        while off <= last {
            match slot_targets.get(&off) {
                Some(r) => {
                    let (adj, real) = split_thunk(&r.target).unwrap_or((0, r.target.as_str()));
                    let mut sig = sig_from_mangled(real, db).unwrap_or_else(|| placeholder_sig(real));
                    sig.is_virtual = true;
                    if sig.mangled.is_none() {
                        sig.mangled = Some(real.to_string());
                    }
                    slots.push(VirtualMethod { vtable_offset: off, sig, symbol: r.target.clone(), this_adjust: adj });
                }
                None => {
                    // secondary-table header (two empty words followed by thunks) or empty slot
                    let next_is_thunk_header = !slot_targets.contains_key(&(off + 4))
                        && slot_targets.get(&(off + 8)).is_some_and(|r| split_thunk(&r.target).is_some());
                    if next_is_thunk_header {
                        off += 8;
                        continue;
                    }
                    slots.push(VirtualMethod { vtable_offset: off, sig: placeholder_sig(""), symbol: String::new(), this_adjust: 0 });
                }
            }
            off += 4;
        }
        name_pure_slot(&class, &mut slots, d.size, db);
        out.insert(class, slots);
    }
    out
}

/// A pure virtual function has no entry (a zero word) in its class's own vtable. When the header
/// declares exactly one virtual of `class` that no slot accounts for, it is the empty interior
/// slot, or else the first zero word after the last entry (within the symbol's size).
fn name_pure_slot(class: &str, slots: &mut Vec<VirtualMethod>, size: u32, db: &TypeDb) {
    let last = |q: &str| q.rsplit("::").next().unwrap_or(q).to_string();
    let present: Vec<(String, usize)> = slots.iter().filter(|s| !s.symbol.is_empty()).map(|s| (last(&s.sig.qualified_name), s.sig.params.len())).collect();
    let prefix = format!("{class}::");
    let mut unassigned: Vec<&DeclInfo> = vec![];
    for (q, list) in db.decls.range(prefix.clone()..) {
        if !q.starts_with(&prefix) {
            break;
        }
        let name = &q[prefix.len()..];
        if name.contains("::") || name.starts_with('~') {
            continue;
        }
        for d in list {
            if (d.is_virtual || d.is_pure) && d.template_params.is_empty() && !present.iter().any(|(n, k)| n == name && *k == d.params.len()) {
                unassigned.push(d);
            }
        }
    }
    if unassigned.len() != 1 {
        return;
    }
    let d = unassigned[0];
    let sig = FuncSig {
        qualified_name: d.qualified_name.clone(),
        mangled: None,
        ret: d.ret.clone(),
        params: d.params.clone(),
        this_class: Some(class.to_string()),
        is_const: d.is_const,
        is_static: false,
        is_virtual: true,
        variadic: d.variadic,
    };
    let empties: Vec<usize> = slots.iter().enumerate().filter(|(_, s)| s.symbol.is_empty()).map(|(i, _)| i).collect();
    match empties.len() {
        1 => slots[empties[0]].sig = sig,
        0 => {
            let next = slots.last().map_or(8, |s| s.vtable_offset + 4);
            let primary_only = slots.iter().all(|s| s.this_adjust == 0);
            if primary_only && next + 4 <= size {
                slots.push(VirtualMethod { vtable_offset: next, sig, symbol: String::new(), this_adjust: 0 });
            }
        }
        _ => {}
    }
}

/// Load an object with `mwdec_obj` and extract its vtables.
pub fn vtables_from_path(path: &str, db: &TypeDb) -> anyhow::Result<BTreeMap<String, Vec<VirtualMethod>>> {
    let obj = mwdec_obj::load_object(path)?;
    Ok(vtables_from_object(&obj, db))
}

/// Store vtables into `db.classes[..].vtable` and mark the corresponding methods/decls virtual.
pub fn apply_vtables(db: &mut TypeDb, vts: &BTreeMap<String, Vec<VirtualMethod>>) {
    for (class, slots) in vts {
        let mut virtual_names = Vec::new();
        for s in slots {
            if !s.sig.qualified_name.is_empty() {
                virtual_names.push(s.sig.qualified_name.clone());
            }
        }
        if let Some(c) = db.classes.get_mut(class) {
            c.vtable = slots.clone();
        }
        for q in virtual_names {
            if let Some(list) = db.decls.get_mut(&q) {
                for d in list {
                    d.is_virtual = true;
                }
            }
        }
    }
    crate::resolve::fill_methods(db);
}

/// Vtables of polymorphic classes no object of the module has a vtable for (abstract
/// interfaces, classes whose vtable lives in another module): the slots follow the header's
/// declaration order, after the primary base's slots, an override taking its base slot (MWCC GC
/// layout: 8 bytes of header, then one word per virtual function, one for the destructor).
/// Returns the number of classes given a vtable.
pub fn declared_vtables(db: &mut TypeDb) -> usize {
    let names: Vec<String> = db.classes.iter().filter(|(n, c)| c.vptr_offset == Some(0) && c.vtable.is_empty() && !c.is_declaration && !n.contains('<')).map(|(n, _)| n.clone()).collect();
    let mut memo: BTreeMap<String, Option<Vec<VirtualMethod>>> = BTreeMap::new();
    if std::env::var_os("MWDEC_VT_VALIDATE").is_some() {
        // agreement of the declaration-order layout with the vtables the objects have
        let have: Vec<String> = db.classes.iter().filter(|(n, c)| c.vptr_offset == Some(0) && !c.vtable.is_empty() && !n.contains('<')).map(|(n, _)| n.clone()).collect();
        let (mut ok, mut bad) = (0, 0);
        for c in have {
            let mut copy = db.clone();
            copy.classes.get_mut(&c).unwrap().vtable.clear();
            let mut m = BTreeMap::new();
            let Some(slots) = declared_slots(&copy, &c, &mut m, 0) else { continue };
            let real = &db.classes[&c].vtable;
            let same = slots.len() == real.len() && slots.iter().zip(real).all(|(a, b)| b.sig.qualified_name.is_empty() || simple_name(&a.sig.qualified_name) == simple_name(&b.sig.qualified_name));
            if same {
                ok += 1;
            } else {
                bad += 1;
                eprintln!("vtable mismatch {c}: declared {:?} vs object {:?}", slots.iter().map(|s| simple_name(&s.sig.qualified_name).to_string()).collect::<Vec<_>>(), real.iter().map(|s| simple_name(&s.sig.qualified_name).to_string()).collect::<Vec<_>>());
            }
        }
        eprintln!("declared vtables: {ok} agree, {bad} differ");
    }
    let mut n = 0;
    // objects' vtables of abstract classes lack the names of pure slots and stop at the last
    // non-pure one: completed from the declarations where those agree
    let have: Vec<String> = db.classes.iter().filter(|(n, c)| c.vptr_offset == Some(0) && !c.vtable.is_empty() && !n.contains('<')).map(|(n, _)| n.clone()).collect();
    for c in have {
        let real = db.classes[&c].vtable.clone();
        let v = completed(db, &c, &real, &mut memo, 0);
        if v.len() != real.len() || v.iter().zip(&real).any(|(a, b)| a.sig.qualified_name != b.sig.qualified_name) {
            db.classes.get_mut(&c).unwrap().vtable = v;
            n += 1;
        }
    }
    for c in names {
        if let Some(slots) = declared_slots(db, &c, &mut memo, 0) {
            if !slots.is_empty() {
                db.classes.get_mut(&c).unwrap().vtable = slots;
                n += 1;
            }
        }
    }
    n
}

/// A type spelling for comparing parameter lists (typedefs resolved, no spaces).
fn norm_ty(db: &TypeDb, t: &Type) -> String {
    fn res<'a>(db: &'a TypeDb, t: &'a Type, d: u32) -> std::borrow::Cow<'a, Type> {
        match t {
            Type::Named(n) if d < 8 => match db.typedefs.get(n) {
                Some(u) => std::borrow::Cow::Owned(res(db, u, d + 1).into_owned()),
                None => std::borrow::Cow::Borrowed(t),
            },
            Type::Ptr(x) => std::borrow::Cow::Owned(Type::Ptr(Box::new(res(db, x, d + 1).into_owned()))),
            Type::Ref(x) => std::borrow::Cow::Owned(Type::Ref(Box::new(res(db, x, d + 1).into_owned()))),
            Type::Const(x) => std::borrow::Cow::Owned(Type::Const(Box::new(res(db, x, d + 1).into_owned()))),
            _ => std::borrow::Cow::Borrowed(t),
        }
    }
    // (top-level const of a by-value parameter is not part of the signature)
    let t = match t {
        Type::Const(x) => &**x,
        t => t,
    };
    let r = res(db, t, 0);
    let s: String = format!("{:?}", r).chars().filter(|c| !c.is_whitespace()).collect();
    // (a class spelled relative to its scope in one place and qualified in the other)
    s.split("Named(\"").map(|p| match p.find('"') {
        Some(e) => format!("{}{}", simple_name(&p[..e]), &p[e..]),
        None => p.to_string(),
    }).collect::<Vec<_>>().join("Named(\"")
}

/// An object's vtable with the names of its unnamed (pure) slots and its trailing slots taken
/// from the declaration-order layout, where the two agree on every named slot.
fn completed(db: &TypeDb, cls: &str, real: &[VirtualMethod], memo: &mut BTreeMap<String, Option<Vec<VirtualMethod>>>, depth: u32) -> Vec<VirtualMethod> {
    let mut copy_db = None;
    let decl = {
        let key = format!("#decl#{cls}");
        if let Some(v) = memo.get(&key) {
            v.clone()
        } else {
            let d = copy_db.get_or_insert_with(|| {
                let mut c = db.clone();
                if let Some(k) = c.classes.get_mut(cls) {
                    k.vtable.clear();
                }
                c
            });
            let mut m = BTreeMap::new();
            let v = declared_slots(d, cls, &mut m, depth + 1);
            memo.insert(key, v.clone());
            v
        }
    };
    let Some(decl) = decl else { return real.to_vec() };
    if real.iter().any(|r| r.this_adjust != 0) {
        return real.to_vec();
    }
    if decl.len() < real.len() {
        return real.to_vec();
    }
    let agree = real.iter().zip(&decl).all(|(r, d)| r.sig.qualified_name.is_empty() || simple_name(&r.sig.qualified_name) == simple_name(&d.sig.qualified_name));
    if !agree {
        return real.to_vec();
    }
    let mut out = real.to_vec();
    for (o, d) in out.iter_mut().zip(&decl) {
        if o.sig.qualified_name.is_empty() {
            o.sig = d.sig.clone();
        }
    }
    out.extend(decl[real.len()..].iter().cloned());
    out
}

fn simple_name(q: &str) -> &str {
    q.rsplit("::").next().unwrap_or(q)
}

fn declared_slots(db: &TypeDb, cls: &str, memo: &mut BTreeMap<String, Option<Vec<VirtualMethod>>>, depth: u32) -> Option<Vec<VirtualMethod>> {
    if let Some(v) = memo.get(cls) {
        return v.clone();
    }
    if depth > 12 {
        return None;
    }
    let c = db.classes.get(cls)?;
    let r = (|| {
        // the primary base (at offset 0, polymorphic): its slots first
        let mut slots: Vec<VirtualMethod> = match c.bases.iter().find(|b| b.offset == 0 && !b.is_virtual && db.classes.get(&b.name).is_some_and(|bc| bc.vptr_offset.is_some())) {
            Some(b) => {
                let bc = db.classes.get(&b.name)?;
                if bc.vtable.is_empty() {
                    declared_slots(db, &b.name, memo, depth + 1)?
                } else {
                    completed(db, &b.name, &bc.vtable, memo, depth)
                }
            }
            None => vec![],
        };
        // a secondary base's slots can't be laid out from declarations
        if c.bases.iter().any(|b| b.offset != 0 && db.classes.get(&b.name).is_some_and(|bc| bc.vptr_offset.is_some())) {
            return None;
        }
        let prefix = format!("{cls}::");
        let mut own: Vec<&DeclInfo> = db
            .decls
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .filter(|(k, _)| !k[prefix.len()..].contains("::"))
            .flat_map(|(_, ds)| ds.iter())
            .filter(|d| !d.is_static && d.template_params.is_empty())
            .collect();
        own.sort_by_key(|d| d.order);
        for d in own {
            let name = simple_name(&d.qualified_name);
            let is_dtor = name.starts_with('~');
            let pkey = |ps: &[Param]| ps.iter().map(|p| norm_ty(db, &p.ty)).collect::<Vec<_>>();
            let overrides = slots.iter().position(|s| {
                let sn = simple_name(&s.sig.qualified_name);
                if is_dtor {
                    sn.starts_with('~')
                } else {
                    sn == name && s.sig.params.len() == d.params.len() && s.sig.is_const == d.is_const && pkey(&s.sig.params) == pkey(&d.params)
                }
            });
            let sig = FuncSig {
                qualified_name: d.qualified_name.clone(),
                mangled: None,
                ret: d.ret.clone(),
                params: d.params.clone(),
                this_class: Some(cls.to_string()),
                is_const: d.is_const,
                is_static: false,
                is_virtual: true,
                variadic: d.variadic,
            };
            match overrides {
                Some(i) => {
                    slots[i].sig = sig;
                    slots[i].symbol.clear();
                }
                None if d.is_virtual => {
                    let off = 8 + 4 * slots.len() as u32;
                    slots.push(VirtualMethod { vtable_offset: off, sig, symbol: String::new(), this_adjust: 0 });
                }
                None => {}
            }
        }
        // a derived class without a declared destructor has its implicit one in the slot
        if let Some(s) = slots.iter_mut().find(|s| simple_name(&s.sig.qualified_name).starts_with('~')) {
            if s.sig.this_class.as_deref() != Some(cls) {
                let last = simple_name(cls).to_string();
                s.sig.qualified_name = format!("{cls}::~{last}");
                s.sig.this_class = Some(cls.to_string());
                s.symbol.clear();
            }
        }
        Some(slots)
    })();
    memo.insert(cls.to_string(), r.clone());
    r
}
