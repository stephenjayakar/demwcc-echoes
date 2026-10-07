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
