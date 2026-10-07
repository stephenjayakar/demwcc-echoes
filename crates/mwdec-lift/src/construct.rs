//! Temporaries constructed for an argument. `f(SObjectTag('ANIM', id))` compiles to member stores
//! into a stack temporary whose address is passed (const reference) or which is the by-value
//! argument copy itself. When the class has an inline constructor whose initializer list maps
//! each parameter to one member (`SObjectTag(FourCC type, CAssetId id) : type(type), id(id)`), and
//! the stores right before the call fill exactly those members, the stores become `T(args)`.

use crate::byval::whole_object;
use crate::ir::*;
use crate::sig;
use crate::types;
use mwdec_core::{FuncSig, Param, Type, TypeDb};
use std::collections::HashMap;

fn strip_tmpl(s: &str) -> String {
    let mut out = String::new();
    let mut d = 0;
    for c in s.chars() {
        match c {
            '<' => d += 1,
            '>' => d -= 1,
            _ if d == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// A constructor usable for member-wise construction: (signature, member name per parameter).
fn ctors(db: &TypeDb, cls: &str) -> Vec<(FuncSig, Vec<String>)> {
    let last = sig::split_scope(cls).1;
    let key = format!("{}::{}", strip_tmpl(cls), strip_tmpl(last));
    let mut out = vec![];
    for d in db.decls.get(&key).map(|v| v.as_slice()).unwrap_or(&[]) {
        if !d.is_inline_defined || d.params.is_empty() || !d.inline_body.as_deref().map_or(false, |b| b.trim().is_empty()) {
            continue;
        }
        // callable from anywhere, and not in terms of template parameters
        if d.access != mwdec_core::Access::Public || !d.template_params.is_empty() {
            continue;
        }
        let Some(init) = &d.init_list else { continue };
        let names: Vec<String> = d.params.iter().map(|p| p.name.clone().unwrap_or_default()).collect();
        if names.iter().any(|n| n.is_empty()) {
            continue;
        }
        // `field ( name ) , field ( name )`
        let mut fields: Vec<Option<String>> = vec![None; names.len()];
        let mut ok = true;
        for part in sig::split_top(init, ',') {
            let toks: Vec<&str> = part.split_whitespace().collect();
            if toks.len() != 4 || toks[1] != "(" || toks[3] != ")" {
                ok = false;
                break;
            }
            match names.iter().position(|n| n == toks[2]) {
                Some(i) if fields[i].is_none() => fields[i] = Some(toks[0].to_string()),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok || fields.iter().any(|f| f.is_none()) {
            continue;
        }
        let s = FuncSig {
            qualified_name: key.clone(),
            mangled: None,
            ret: Type::Void,
            params: d.params.iter().map(|p| Param { name: p.name.clone(), ty: p.ty.clone() }).collect(),
            this_class: Some(cls.to_string()),
            is_const: false,
            is_static: false,
            is_virtual: false,
            variadic: false,
        };
        out.push((s, fields.into_iter().map(|f| f.unwrap()).collect()));
    }
    out
}

/// Constructors of `cls` that set members directly from parameters, for objects whose inline
/// constructor the compiler expanded into member stores: (signature, member offset per
/// parameter). Initializer-list entries `m(param)` and base classes built from parameters
/// (`Base(a, b)`) map parameters; other entries (constants, expressions) are the constructor's own.
pub fn member_ctors(db: &TypeDb, cls: &str) -> Vec<(FuncSig, Vec<i32>)> {
    member_ctors_d(db, cls, 0)
}

fn member_ctors_d(db: &TypeDb, cls: &str, depth: u32) -> Vec<(FuncSig, Vec<i32>)> {
    let Some(c) = sig::find_class(db, cls) else { return vec![] };
    let last = sig::split_scope(cls).1;
    let key = format!("{}::{}", strip_tmpl(cls), strip_tmpl(last));
    let mut out = vec![];
    for d in db.decls.get(&key).map(|v| v.as_slice()).unwrap_or(&[]) {
        if !d.is_inline_defined || d.params.is_empty() || d.access != mwdec_core::Access::Public || !d.template_params.is_empty() {
            continue;
        }
        let Some(init) = &d.init_list else { continue };
        let names: Vec<String> = d.params.iter().map(|p| p.name.clone().unwrap_or_default()).collect();
        let mut offs: Vec<Option<i32>> = vec![None; names.len()];
        for part in sig::split_top(init, ',') {
            let toks: Vec<&str> = part.split_whitespace().collect();
            if toks.len() < 3 || toks[1] != "(" || toks.last() != Some(&")") {
                continue;
            }
            if toks.len() == 4 {
                if let (Some(i), Some(f)) = (names.iter().position(|n| n == toks[2]), c.fields.iter().find(|f| f.name == toks[0] && f.bitfield.is_none())) {
                    if offs[i].is_none() {
                        offs[i] = Some(f.offset as i32);
                    }
                    continue;
                }
            }
            if depth >= 4 {
                continue;
            }
            // a base class or a member object built from parameters
            let sub = match c.bases.iter().find(|b| sig::split_scope(&strip_tmpl(&b.name)).1 == toks[0]) {
                Some(b) => Some((b.name.clone(), b.offset as i32)),
                None => c.fields.iter().find(|f| f.name == toks[0] && f.bitfield.is_none()).and_then(|f| {
                    let ft = types::resolve(Some(db), strip_cv(&f.ty)).into_owned();
                    named(&ft).map(|n| (n.to_string(), f.offset as i32))
                }),
            };
            let Some((sub_cls, sub_off)) = sub else { continue };
            let inner = toks[2..toks.len() - 1].join(" ");
            let bargs: Vec<String> = sig::split_top(&inner, ',').into_iter().map(|a| a.trim().to_string()).collect();
            if let Some((_, boffs)) = member_ctors_d(db, &sub_cls, depth + 1).into_iter().find(|(s, _)| s.params.len() == bargs.len()) {
                for (j, a) in bargs.iter().enumerate() {
                    if let Some(i) = names.iter().position(|n| n == a) {
                        if offs[i].is_none() {
                            offs[i] = Some(sub_off + boffs[j]);
                        }
                    }
                }
            }
        }
        if offs.iter().any(|o| o.is_none()) {
            continue;
        }
        let s = FuncSig {
            qualified_name: key.clone(),
            mangled: None,
            ret: Type::Void,
            params: d.params.iter().map(|p| Param { name: p.name.clone(), ty: p.ty.clone() }).collect(),
            this_class: Some(cls.to_string()),
            is_const: false,
            is_static: false,
            is_virtual: false,
            variadic: false,
        };
        out.push((s, offs.into_iter().map(|o| o.unwrap()).collect()));
    }
    out
}

/// An object of `cls` built from member stores (`(offset, value)` relative to the object) with a
/// constructor that sets members from its parameters (`member_ctors`), nested class parameters
/// built the same way. Returns the construction and the store offsets it used.
pub fn build_from_stores(db: &TypeDb, cls: &str, stores: &[(i32, Expr)]) -> Option<(Expr, Vec<i32>)> {
    let mut used = vec![];
    let e = build_at(db, cls, 0, stores, 0, &mut used)?;
    Some((e, used))
}

fn build_at(db: &TypeDb, cls: &str, base: i32, stores: &[(i32, Expr)], depth: u32, used: &mut Vec<i32>) -> Option<Expr> {
    if depth > 4 {
        return None;
    }
    'ctor: for (cs, offs) in member_ctors(db, cls) {
        let mark = used.len();
        let mut args = vec![];
        for (p, o) in cs.params.iter().zip(&offs) {
            let pt = match strip_cv(&p.ty) {
                Type::Ref(x) => strip_cv(x).clone(),
                t => t.clone(),
            };
            let pr = types::resolve(Some(db), &pt).into_owned();
            let arg = if types::is_aggregate(Some(db), &pr) {
                named(&pr).and_then(|pc| build_at(db, pc, base + o, stores, depth + 1, used))
            } else {
                stores.iter().find(|(so, _)| *so == base + o).map(|(so, x)| {
                    used.push(*so);
                    x.clone()
                })
            };
            match arg {
                Some(a) => args.push(a),
                None => {
                    used.truncate(mark);
                    continue 'ctor;
                }
            }
        }
        return Some(Expr::Construct { class: Type::Named(cls.to_string()), ctor: Some(cs), args });
    }
    None
}

/// The stored value as a value of member type `ft` (scalar as is; a one-member object from its
/// member's value: whole-object read or a one-argument constructor).
fn as_member(v: &Expr, ft: &Type, vars: &[Var], db: &TypeDb, defs: &HashMap<VarId, Expr>) -> Option<Expr> {
    let r = types::resolve(Some(db), ft).into_owned();
    if !types::is_aggregate(Some(db), &r) {
        return Some(v.clone());
    }
    if let Some(w) = whole_object(v, strip_cv(ft), vars, db) {
        return Some(w);
    }
    // a register value loaded as the whole object
    if let Expr::Var(t) = v {
        if let Some(w) = defs.get(t).and_then(|d| whole_object(d, strip_cv(ft), vars, db)) {
            return Some(w);
        }
    }
    let cls = named(&r)?.to_string();
    let (s, _) = ctors(db, &cls).into_iter().find(|(s, _)| s.params.len() == 1)?;
    Some(Expr::Construct { class: strip_cv(ft).clone(), ctor: Some(s), args: vec![v.clone()] })
}

/// (offset, size, float) of each direct member by name.
fn member_slots(db: &TypeDb, cls: &str) -> HashMap<String, (i32, u32, Type)> {
    let mut m = HashMap::new();
    if let Some(c) = sig::find_class(db, cls) {
        for f in &c.fields {
            if f.bitfield.is_some() {
                continue;
            }
            if let Some(s) = types::size_of(Some(db), &f.ty) {
                m.insert(f.name.clone(), (f.offset as i32, s, f.ty.clone()));
            }
        }
    }
    m
}

fn count_var(b: &[Stmt], v: VarId) -> usize {
    let mut n = 0;
    Stmt::walk_exprs(b, &mut |e| {
        if matches!(e, Expr::Var(x) if *x == v) {
            n += 1
        }
    });
    n
}

/// (stack object, class, the arg expression to replace) for constructible arguments of a call.
fn temp_args(e: &Expr, vars: &[Var], db: &TypeDb) -> Vec<(VarId, String)> {
    let mut out = vec![];
    e.walk(&mut |x| {
        let Expr::Call { callee, args, .. } = x else { return };
        let sig = match callee {
            Callee::Direct { sig, .. } | Callee::Method { sig, .. } => Some(sig),
            Callee::Virtual { sig, .. } => sig.as_ref(),
            Callee::Indirect(_) => None,
        };
        let Some(sig) = sig else { return };
        for (n, a) in args.iter().enumerate() {
            let Some(p) = sig.params.get(n) else { continue };
            let (v, t) = match (a, strip_cv(&p.ty)) {
                // const reference
                (Expr::AddrOf(inner), Type::Ref(t)) if matches!(**t, Type::Const(_)) => match **inner {
                    Expr::Var(v) => (v, (**t).clone()),
                    _ => continue,
                },
                // by-value argument copy
                (Expr::Var(v), t) if !is_ptr(t) && types::is_aggregate(Some(db), t) => (*v, t.clone()),
                _ => continue,
            };
            if !matches!(vars[v].kind, VarKind::Stack { .. }) {
                continue;
            }
            let r = types::resolve(Some(db), strip_cv(&t)).into_owned();
            if let Some(c) = named(&r) {
                out.push((v, c.to_string()));
            }
        }
    });
    out
}

pub fn fold(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb) {
    // single-assignment register locals and their values
    let mut defs: HashMap<VarId, Expr> = HashMap::new();
    {
        let mut n: HashMap<VarId, usize> = HashMap::new();
        Stmt::for_each_block_mut(body, &mut |b| {
            for s in b.iter() {
                if let Stmt::Assign { dst: Expr::Var(v), src } = s {
                    if matches!(vars[*v].kind, VarKind::Local) {
                        *n.entry(*v).or_default() += 1;
                        if !src.has_call() {
                            defs.insert(*v, src.clone());
                        }
                    }
                }
            }
        });
        defs.retain(|v, _| n.get(v) == Some(&1));
    }
    fold_with(body, vars, db, &defs);
    // register locals the constructions made unused
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

fn fold_with(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb, defs: &HashMap<VarId, Expr>) {
    let defs = defs.clone();
    let total = {
        let mut m: HashMap<VarId, usize> = HashMap::new();
        Stmt::walk_exprs(body, &mut |e| {
            if let Expr::Var(v) = e {
                *m.entry(*v).or_default() += 1;
            }
        });
        m
    };
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut j = 0;
        while j < b.len() {
            let heads: Vec<Expr> = match &b[j] {
                Stmt::Expr(e) | Stmt::Return(Some(e)) => vec![e.clone()],
                Stmt::Assign { src, .. } => vec![src.clone()],
                Stmt::If { cond, .. } => vec![cond.clone()],
                _ => vec![],
            };
            let mut done = false;
            for h in &heads {
                for (v, cls) in temp_args(h, vars, db) {
                    // the stores right before the call
                    let mut stores: Vec<(usize, i32, u32, Expr)> = vec![];
                    let mut k = j;
                    while k > 0 {
                        k -= 1;
                        match &b[k] {
                            Stmt::Assign { dst: Expr::Member { base, offset, ty }, src } if matches!(**base, Expr::Var(x) if x == v) => {
                                stores.push((k, *offset, scalar_size(ty).unwrap_or(0), src.clone()));
                            }
                            Stmt::Assign { dst: Expr::Var(x), src } if *x == v => {
                                stores.push((k, 0, scalar_size(&vars[v].ty).unwrap_or(0), src.clone()));
                            }
                            Stmt::Assign { dst: Expr::Var(x), src } if !src.has_call() && !src.uses_var(v) && matches!(vars[*x].kind, VarKind::Local) => {}
                            _ => break,
                        }
                    }
                    if stores.is_empty() || total.get(&v).copied().unwrap_or(0) != stores.len() + count_var(std::slice::from_ref(&b[j]), v) || count_var(std::slice::from_ref(&b[j]), v) != 1 {
                        continue;
                    }
                    let slots = member_slots(db, &cls);
                    let mut built = None;
                    for (s, fields) in ctors(db, &cls) {
                        if fields.len() != stores.len() {
                            continue;
                        }
                        let mut args = vec![];
                        let mut ok = true;
                        for f in &fields {
                            let Some((off, size, ft)) = slots.get(f) else {
                                ok = false;
                                break;
                            };
                            let Some(st) = stores.iter().find(|s| s.1 == *off) else {
                                ok = false;
                                break;
                            };
                            if st.2 != *size {
                                ok = false;
                                break;
                            }
                            match as_member(&st.3, ft, vars, db, &defs) {
                                Some(a) => args.push(a),
                                None => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        // stores to distinct members only
                        let mut offs: Vec<i32> = stores.iter().map(|s| s.1).collect();
                        offs.sort_unstable();
                        offs.dedup();
                        if ok && offs.len() == stores.len() {
                            built = Some(Expr::Construct { class: Type::Named(cls.clone()), ctor: Some(s), args });
                            break;
                        }
                    }
                    // otherwise a constructor setting some members from parameters (nested
                    // objects too); the other stores must be its own constants
                    if built.is_none() {
                        let flat: Vec<(i32, Expr)> = stores.iter().map(|s| (s.1, s.3.clone())).collect();
                        let mut offs: Vec<i32> = flat.iter().map(|s| s.0).collect();
                        offs.sort_unstable();
                        offs.dedup();
                        if offs.len() == flat.len() {
                            if let Some((c, used)) = build_from_stores(db, &cls, &flat) {
                                let rest_const = flat.iter().filter(|(o, _)| !used.contains(o)).all(|(_, x)| matches!(x, Expr::Int { .. } | Expr::Float { .. }));
                                if rest_const {
                                    built = Some(c);
                                }
                            }
                        }
                    }
                    let Some(c) = built else { continue };
                    // replace the argument, drop the stores
                    let rewrite = |x: &mut Expr| {
                        if let Expr::Call { args, .. } = x {
                            for a in args.iter_mut() {
                                let hit = match a {
                                    Expr::AddrOf(inner) => matches!(**inner, Expr::Var(y) if y == v),
                                    Expr::Var(y) => *y == v,
                                    _ => false,
                                };
                                if hit {
                                    *a = c.clone();
                                }
                            }
                        }
                    };
                    let mut rw = rewrite;
                    match &mut b[j] {
                        Stmt::Expr(e) | Stmt::Return(Some(e)) => e.rewrite(&mut rw),
                        Stmt::Assign { src, .. } => src.rewrite(&mut rw),
                        Stmt::If { cond, .. } => cond.rewrite(&mut rw),
                        _ => {}
                    }
                    let mut rm: Vec<usize> = stores.iter().map(|s| s.0).collect();
                    rm.sort_unstable();
                    for i in rm.into_iter().rev() {
                        b.remove(i);
                        j -= 1;
                    }
                    done = true;
                    break;
                }
                if done {
                    break;
                }
            }
            j += 1;
        }
    });
}
