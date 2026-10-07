//! Which classes matter for a unit: the classes named by the target object's symbols (function
//! scopes and parameter types, data symbols), closed over member and base types. Probing only
//! their inlines keeps the probe TU small.

use mwdec_core::{ObjectFile, Type, TypeDb};
use std::collections::HashSet;

fn classes_in_type(t: &Type, db: &TypeDb, out: &mut Vec<String>) {
    match t {
        Type::Ptr(x) | Type::Ref(x) | Type::Const(x) | Type::Volatile(x) | Type::Array(x, _) => classes_in_type(x, db, out),
        Type::Named(_) => {
            if let Some(c) = crate::util::class_name(t, db) {
                out.push(c);
            }
        }
        _ => {}
    }
}

fn add_scope_classes(q: &str, db: &TypeDb, out: &mut Vec<String>) {
    let mut cur = q.to_string();
    loop {
        let (scope, _) = mwdec_lift::sig::split_scope(&cur);
        let Some(s) = scope else { break };
        if let Some(c) = mwdec_lift::sig::find_class(db, s) {
            out.push(c.name.clone());
        }
        cur = s.to_string();
    }
}

fn seeds_of(syms: &mut dyn Iterator<Item = &str>, db: &TypeDb) -> Vec<String> {
    let mut seeds: Vec<String> = vec![];
    for sym in syms {
        let sig = mwdec_lift::sig::sig_of(sym, Some(db));
        add_scope_classes(&sig.qualified_name, db, &mut seeds);
        for p in &sig.params {
            classes_in_type(&p.ty, db, &mut seeds);
        }
        classes_in_type(&sig.ret, db, &mut seeds);
        if let Some((_, t)) = db.globals.get(sym) {
            classes_in_type(t, db, &mut seeds);
        }
    }
    seeds
}

/// Seeds closed over bases and member types, breadth-first up to `depth`.
fn close(seeds: Vec<String>, db: &TypeDb, depth: u32) -> HashSet<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut frontier: std::collections::VecDeque<(String, u32)> = seeds.into_iter().map(|c| (c, 0)).collect();
    while let Some((c, d)) = frontier.pop_front() {
        if !seen.insert(mwdec_lift::sig::norm_name(&c)) || d >= depth {
            continue;
        }
        let Some(k) = mwdec_lift::sig::find_class(db, &c) else { continue };
        for b in &k.bases {
            frontier.push_back((b.name.clone(), d + 1));
        }
        let mut fs = vec![];
        for f in &k.fields {
            classes_in_type(&f.ty, db, &mut fs);
        }
        for m in &k.methods {
            classes_in_type(&m.ret, db, &mut fs);
        }
        frontier.extend(fs.into_iter().map(|x| (x, d + 1)));
    }
    seen
}

/// Classes named by every symbol of the target object, closed over members (depth 3).
pub fn relevant_classes(obj: &ObjectFile, db: &TypeDb) -> HashSet<String> {
    let seeds = seeds_of(&mut obj.all_symbols.iter().map(|s| s.as_str()), db);
    close(seeds, db, 2)
}

/// Classes one function can touch: its own signature and the symbols it references (callees,
/// globals), closed over members (depth 2). Probing per function keeps probe TUs small.
pub fn relevant_for_function(f: &mwdec_core::Function, db: &TypeDb) -> HashSet<String> {
    let mut syms: Vec<&str> = vec![f.name.as_str()];
    syms.extend(f.relocs.iter().map(|r| r.target.as_str()));
    let seeds = seeds_of(&mut syms.into_iter(), db);
    close(seeds, db, 2)
}

/// Is a declaration with this class / parameter types relevant?
pub fn decl_relevant(class: Option<&str>, params: &[mwdec_core::Param], ret: &Type, db: &TypeDb, rel: &HashSet<String>) -> bool {
    if let Some(c) = class {
        return rel.contains(&mwdec_lift::sig::norm_name(c));
    }
    // free functions: any class-typed parameter must be relevant; scalar-only helpers are kept
    let mut cs = vec![];
    for p in params {
        classes_in_type(&p.ty, db, &mut cs);
    }
    classes_in_type(ret, db, &mut cs);
    cs.iter().all(|c| rel.contains(&mwdec_lift::sig::norm_name(c)))
}

/// Simple accessors/setters the emitter already renders (`return mX ;`, `mX = x ;`).
pub fn trivial_body(body: &str) -> bool {
    let t: Vec<&str> = body.split_whitespace().collect();
    let ident = |s: &str| s.chars().next().map_or(false, |c| c.is_alphabetic() || c == '_') && s.chars().all(|c| c.is_alphanumeric() || c == '_');
    match t.as_slice() {
        ["return", a, ";"] => ident(a),
        ["return", "&", a, ";"] => ident(a),
        ["return", a, ".", b, ";"] => ident(a) && ident(b),
        ["return", "this", "->", a, ";"] => ident(a),
        [a, "=", b, ";"] => ident(a) && ident(b),
        _ => false,
    }
}
