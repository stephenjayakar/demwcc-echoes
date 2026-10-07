//! By-value aggregate arguments. MWCC passes a class argument by value as the address of a
//! temporary copy made right before the call:
//!
//! ```text
//! lhz r0, kInvalidUniqueId ; sth r0, 8(r1) ; addi r4, r1, 8 ; bl ApplyTime
//! ```
//!
//! The lifter sees a stack object `S` written from `X` and then passed. When `S` exists only for
//! such copies, `S = X; f(S)` becomes `f(X)` (the compiler re-creates the copy).

use crate::aggregates::aggregate_at;
use crate::ir::*;
use crate::sig;
use crate::types;
use mwdec_core::{FuncSig, Type, TypeDb};
use std::collections::HashMap;

fn type_key(db: &TypeDb, t: &Type) -> Option<String> {
    let r = types::resolve(Some(db), t).into_owned();
    named(&r).map(sig::norm_name)
}

fn strip_casts(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip_casts(e),
        e => e,
    }
}

/// `src` (a scalar read covering all of `t`) as a whole-object lvalue of type `t`.
pub(crate) fn whole_object(src: &Expr, t: &Type, vars: &[Var], db: &TypeDb) -> Option<Expr> {
    let want = type_key(db, t)?;
    let has = |cls: &str, off: i32| aggregate_at(db, cls, off).iter().any(|a| type_key(db, a).as_deref() == Some(want.as_str()));
    match strip_casts(src) {
        Expr::Load { base, offset, .. } => {
            let bt = types::ty_of(base, vars);
            let cls = type_key(db, pointee(&bt)?)?;
            let cname = named(&types::resolve(Some(db), pointee(&bt)?)).map(|s| s.to_string())?;
            let _ = cls;
            if has(&cname, *offset) {
                return Some(Expr::Load { base: base.clone(), offset: *offset, ty: t.clone() });
            }
            None
        }
        Expr::Member { base, offset, .. } => {
            let bt = types::ty_of(base, vars);
            let cname = named(&types::resolve(Some(db), &bt)).map(|s| s.to_string())?;
            if *offset == 0 && type_key(db, &bt).as_deref() == Some(want.as_str()) {
                return Some((**base).clone());
            }
            if has(&cname, *offset) {
                return Some(Expr::Member { base: base.clone(), offset: *offset, ty: t.clone() });
            }
            None
        }
        _ => None,
    }
}

fn is_byval(db: &TypeDb, t: &Type) -> bool {
    !is_ptr(t) && types::is_aggregate(Some(db), t) && types::class_of(Some(db), t).is_some()
}

/// The signature parameter types of a call, aligned with its argument list.
fn call_params(callee: &Callee) -> Option<&FuncSig> {
    match callee {
        Callee::Direct { sig, .. } | Callee::Method { sig, .. } => Some(sig),
        Callee::Virtual { sig, .. } => sig.as_ref(),
        Callee::Indirect(_) => None,
    }
}

/// Number of occurrences of `Var(v)` in an expression.
fn count_in(e: &Expr, v: VarId) -> usize {
    let mut n = 0;
    e.walk(&mut |x| {
        if matches!(x, Expr::Var(y) if *y == v) {
            n += 1
        }
    });
    n
}

/// The expressions a statement evaluates itself (not those of nested statement lists).
fn head_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => vec![e],
        Stmt::Assign { dst, src } => vec![dst, src],
        Stmt::If { cond, .. } => vec![cond],
        Stmt::Switch { e, .. } => vec![e],
        _ => vec![],
    }
}

fn head_exprs_mut(s: &mut Stmt) -> Vec<&mut Expr> {
    match s {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => vec![e],
        Stmt::Assign { dst, src } => vec![dst, src],
        Stmt::If { cond, .. } => vec![cond],
        Stmt::Switch { e, .. } => vec![e],
        _ => vec![],
    }
}

/// Is `Var(v)` (exactly once in `s`'s head) a by-value aggregate call argument?
fn byval_arg_use(s: &Stmt, v: VarId, db: &TypeDb) -> Option<Type> {
    let heads = head_exprs(s);
    if heads.iter().map(|e| count_in(e, v)).sum::<usize>() != 1 {
        return None;
    }
    let mut ok = None;
    for e in heads {
        e.walk(&mut |x| {
            if let Expr::Call { callee, args, .. } = x {
                if let Some(sig) = call_params(callee) {
                    for (n, a) in args.iter().enumerate() {
                        if matches!(a, Expr::Var(y) if *y == v) {
                            if let Some(p) = sig.params.get(n) {
                                if is_byval(db, &p.ty) {
                                    ok = Some(strip_cv(&p.ty).clone());
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    ok
}

fn replace_var(s: &mut Stmt, v: VarId, with: &Expr) {
    for e in head_exprs_mut(s) {
        e.rewrite(&mut |x| {
            if matches!(x, Expr::Var(y) if *y == v) {
                *x = with.clone();
            }
        });
    }
}

fn vars_in(e: &Expr) -> Vec<VarId> {
    let mut out = vec![];
    e.walk(&mut |x| {
        if let Expr::Var(v) = x {
            out.push(*v)
        }
    });
    out
}

/// Candidate (definition, use) pair in one list: (def index, use index, stack var, source).
type Pair = (usize, usize, VarId, Expr);

fn pairs_in(b: &[Stmt], vars: &[Var], db: &TypeDb, defs: &HashMap<VarId, Expr>) -> Vec<Pair> {
    let mut out = vec![];
    for (i, s) in b.iter().enumerate() {
        let Stmt::Assign { dst, src } = s else { continue };
        let (sv, whole) = match dst {
            Expr::Var(v) => (*v, true),
            Expr::Member { base, offset: 0, .. } => match **base {
                Expr::Var(v) => (v, false),
                _ => continue,
            },
            _ => continue,
        };
        if !matches!(vars[sv].kind, VarKind::Stack { .. }) || src.uses_var(sv) {
            continue;
        }
        // the next statement mentioning the object must pass it by value; statements in between
        // only compute register temps that don't feed the source
        let deps = vars_in(src);
        let movable_call = src.has_call();
        let mut j = i + 1;
        let mut use_ty = None;
        while j < b.len() {
            let s2 = &b[j];
            let mut mentions = 0;
            Stmt::walk_exprs(std::slice::from_ref(s2), &mut |x| {
                if matches!(x, Expr::Var(y) if *y == sv) {
                    mentions += 1
                }
            });
            if mentions > 0 {
                use_ty = byval_arg_use(s2, sv, db);
                break;
            }
            match s2 {
                Stmt::Assign { dst: Expr::Var(x), src: s2src }
                    if !s2src.has_call() && !deps.contains(x) && !(movable_call && crate::translate::reads_mem(s2src)) => {}
                Stmt::Comment(_) => {}
                _ => break,
            }
            j += 1;
        }
        let Some(t) = use_ty else { continue };
        let tsize = types::size_of(Some(db), &t).unwrap_or(0);
        if tsize == 0 {
            continue;
        }
        let whole = whole && type_key(db, &vars[sv].ty) == type_key(db, &t);
        let l = if whole {
            match src {
                Expr::Call { .. } | Expr::Construct { .. } if type_key(db, &types::ty_of(src, vars)) == type_key(db, &t) => src.clone(),
                e if e.is_lvalue() && type_key(db, &types::ty_of(e, vars)) == type_key(db, &t) => e.clone(),
                _ => continue,
            }
        } else {
            let dty = types::ty_of(dst, vars);
            if scalar_size(&dty) != Some(tsize) {
                continue;
            }
            let src_whole = src.is_lvalue() && type_key(db, &types::ty_of(src, vars)) == type_key(db, &t);
            match if src_whole { Some(src.clone()) } else { whole_object(src, &t, vars, db) } {
                Some(l) => l,
                // a register value loaded as the whole object (the compiler reused the load
                // for the argument copy): pass that object
                None if matches!(src, Expr::Var(l) if defs.get(l).and_then(|d| whole_object(d, &t, vars, db)).map_or(false, |x| stable(&x, vars))) => {
                    let Expr::Var(l) = src else { continue };
                    whole_object(&defs[l], &t, vars, db).unwrap()
                }
                // a scalar stack local holding the object's only member (checked in `forward`)
                None => match src {
                    Expr::Var(l) if matches!(vars[*l].kind, VarKind::Stack { .. } | VarKind::Local) && scalar_size(&vars[*l].ty) == Some(tsize) && *l != sv => src.clone(),
                    _ => continue,
                },
            }
        };
        out.push((i, j, sv, l));
    }
    out
}

/// Forward by-value argument copies through stack temporaries.
/// An object lvalue whose value can't have changed between a read and a later copy: a
/// parameter (or reference parameter) or `this`'s member, or a global.
fn stable(x: &Expr, vars: &[Var]) -> bool {
    match x {
        Expr::Var(v) => matches!(vars[*v].kind, VarKind::Param { .. }),
        Expr::Member { base, .. } | Expr::Load { base, .. } => stable(base, vars),
        Expr::Global { .. } => true,
        Expr::AddrOf(inner) => stable(inner, vars),
        _ => false,
    }
}

pub fn forward(body: &mut Vec<Stmt>, vars: &mut [Var], db: &TypeDb) {
    // total mentions of every stack var
    let mut total: HashMap<VarId, usize> = HashMap::new();
    Stmt::walk_exprs(body, &mut |x| {
        if let Expr::Var(v) = x {
            *total.entry(*v).or_default() += 1;
        }
    });
    // single-assignment register locals and their values
    let mut defs: HashMap<VarId, Expr> = HashMap::new();
    {
        let mut n: HashMap<VarId, usize> = HashMap::new();
        Stmt::for_each_block_mut(body, &mut |b| {
            for s in b.iter() {
                if let Stmt::Assign { dst: Expr::Var(v), src } = s {
                    if matches!(vars[*v].kind, VarKind::Local) {
                        *n.entry(*v).or_default() += 1;
                        defs.insert(*v, src.clone());
                    }
                }
            }
        });
        defs.retain(|v, _| n.get(v) == Some(&1));
    }
    // collect pairs everywhere
    let mut found: HashMap<VarId, usize> = HashMap::new();
    // scalar locals used as whole-object sources: (local, object type, uses as source)
    let mut locals: HashMap<VarId, (Type, usize)> = HashMap::new();
    let mut local_bad: Vec<VarId> = vec![];
    {
        let vars_ro: &[Var] = vars;
        Stmt::for_each_block_mut(body, &mut |b| {
            for (_, j, v, l) in pairs_in(b, vars_ro, db, &defs) {
                *found.entry(v).or_default() += 2;
                if let Expr::Var(lv) = l {
                    if matches!(vars_ro[v].kind, VarKind::Stack { .. }) && !types::is_aggregate(Some(db), &vars_ro[lv].ty) {
                        let t = byval_arg_use(&b[j], v, db).unwrap_or(Type::Void);
                        let e = locals.entry(lv).or_insert((t.clone(), 0));
                        if type_key(db, &e.0) != type_key(db, &t) {
                            local_bad.push(lv);
                        }
                        e.1 += 1;
                    }
                }
            }
        });
    }
    // such a local must be written only by whole-object member copies and read only by the pairs
    let mut local_defs: Vec<(VarId, Type)> = vec![];
    let mut member_reads: Vec<(VarId, Type, Type)> = vec![];
    for (&lv, (t, n)) in &locals {
        if local_bad.contains(&lv) {
            continue;
        }
        let mut defs = 0;
        let mut ok = true;
        Stmt::for_each_block_mut(body, &mut |b| {
            for s in b.iter() {
                if let Stmt::Assign { dst: Expr::Var(x), src } = s {
                    if *x == lv {
                        defs += 1;
                        if whole_object(src, t, vars, db).is_none() {
                            ok = false;
                        }
                    }
                }
            }
        });
        let tot = total.get(&lv).copied().unwrap_or(0);
        // other reads of the local become reads of the object's only member
        let member_ok = tot == defs + n || types::field_path(db, &named(&types::resolve(Some(db), t)).map(|s| s.to_string()).unwrap_or_default(), 0, scalar_size(&vars[lv].ty).unwrap_or(0)).is_some();
        if ok && defs > 0 && tot >= defs + n && member_ok {
            if tot > defs + n {
                member_reads.push((lv, vars[lv].ty.clone(), t.clone()));
            }
            local_defs.push((lv, t.clone()));
        } else {
            local_bad.push(lv);
        }
    }
    for (lv, t) in &local_defs {
        let lv = *lv;
        let t = t.clone();
        let vars_ro: Vec<Var> = vars.to_vec();
        Stmt::for_each_block_mut(body, &mut |b| {
            for s in b.iter_mut() {
                if let Stmt::Assign { dst: Expr::Var(x), src } = s {
                    if *x == lv {
                        if let Some(w) = whole_object(src, &t, &vars_ro, db) {
                            *src = w;
                        }
                    }
                }
            }
        });
        vars[lv].ty = t;
    }
    let vars: &[Var] = vars;
    // pairs whose source is a local that can't take the object type are dropped
    let mut drop_src: Vec<VarId> = vec![];
    Stmt::for_each_block_mut(body, &mut |b| {
        for (_, _, v, l) in pairs_in(b, vars, db, &defs) {
            if let Expr::Var(lv) = l {
                if local_bad.contains(&lv) {
                    drop_src.push(v);
                }
            }
        }
    });
    let ok: Vec<VarId> = found.iter().filter(|(v, n)| total.get(v) == Some(n) && !drop_src.contains(v)).map(|(v, _)| *v).collect();
    if ok.is_empty() {
        retype_args(body, vars, db);
        return;
    }
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut ps = pairs_in(b, vars, db, &defs);
        ps.retain(|p| ok.contains(&p.2));
        // substitute first (indices stable), then drop the copies back to front
        for (_, j, v, l) in &ps {
            replace_var(&mut b[*j], *v, l);
        }
        let mut rm: Vec<usize> = ps.iter().map(|p| p.0).collect();
        rm.sort_unstable();
        rm.dedup();
        for i in rm.into_iter().rev() {
            b.remove(i);
        }
    });
    for (lv, sty, t) in &member_reads {
        member_rewrite(body, *lv, sty, t, db);
    }
    retype_args(body, vars, db);
}

/// Reads of object local `lv` outside definitions and by-value arguments of its class: the
/// object's single member (`uid.value`).
fn member_rewrite(body: &mut Vec<Stmt>, lv: VarId, sty: &Type, t: &Type, db: &TypeDb) {
    const HOLD: VarId = usize::MAX;
    let tk = type_key(db, t);
    // protect by-value argument uses and definition targets
    Stmt::rewrite_exprs(body, &mut |x| {
        if let Expr::Call { callee, args, .. } = x {
            let ptys: Vec<Type> = call_params(callee).map(|s| s.params.iter().map(|p| p.ty.clone()).collect()).unwrap_or_default();
            for (n, a) in args.iter_mut().enumerate() {
                if matches!(a, Expr::Var(y) if *y == lv) && ptys.get(n).map_or(false, |p| is_byval(db, p) && type_key(db, p) == tk) {
                    *a = Expr::Var(HOLD);
                }
            }
        }
    });
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter_mut() {
            if let Stmt::Assign { dst, .. } = s {
                if matches!(dst, Expr::Var(y) if *y == lv) {
                    *dst = Expr::Var(HOLD);
                }
            }
        }
    });
    Stmt::rewrite_exprs(body, &mut |x| {
        if matches!(x, Expr::Var(y) if *y == lv) {
            *x = Expr::Member { base: Box::new(Expr::Var(lv)), offset: 0, ty: sty.clone() };
        }
    });
    Stmt::rewrite_exprs(body, &mut |x| {
        if matches!(x, Expr::Var(y) if *y == HOLD) {
            *x = Expr::Var(lv);
        }
    });
}

/// A stack object passed by value whose declared type differs from the parameter's (untyped
/// region, no default constructor): pass `*(T*)&S`.
fn retype_args(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb) {
    Stmt::rewrite_exprs(body, &mut |x| {
        let Expr::Call { callee, args, .. } = x else { return };
        let Some(sig) = call_params(callee) else { return };
        let ptys: Vec<Type> = sig.params.iter().map(|p| p.ty.clone()).collect();
        for (n, a) in args.iter_mut().enumerate() {
            let Expr::Var(v) = a else { continue };
            let v = *v;
            if !matches!(vars[v].kind, VarKind::Stack { .. }) {
                continue;
            }
            let Some(pt) = ptys.get(n) else { continue };
            let key = |t: &Type| type_key(db, t).unwrap_or_else(|| sig::norm_name(&format!("{:?}", strip_cv(t))));
            if is_ptr(pt) || !types::is_aggregate(Some(db), pt) || key(&vars[v].ty) == key(pt) {
                continue;
            }
            let t = strip_cv(pt).clone();
            *a = Expr::Load { base: Box::new(Expr::AddrOf(Box::new(Expr::Var(v)))), offset: 0, ty: t };
        }
    });
}
