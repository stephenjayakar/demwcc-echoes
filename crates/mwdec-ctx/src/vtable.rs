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
