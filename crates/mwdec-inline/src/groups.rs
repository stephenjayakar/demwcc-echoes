//! Object-valued templates on groups of member-wise stores: `d.x = a.x - b.x; d.y = ...;`
//! becomes `d = a - b;`, `p->x += v.x; ...` becomes `*p += v;`.

use crate::addr::{access, lvalue_addr, object_at};
use crate::matcher::{res, teq, Bind, Env, Index, M};
use crate::template::{flat_fields, HoleKind, Shape};
use crate::util::*;
use mwdec_core::Type;
use mwdec_lift::types::ty_of;
use mwdec_lift::{Expr, Stmt};

#[derive(Clone, Debug)]
struct CStore {
    stmt: usize,
    /// address class: stores with equal ids have the same canonical pointer (modulo temps)
    aid: usize,
    /// canonical pointer to the stored object and byte offset
    addr: Expr,
    off: i32,
    ty: Type,
    src: Expr,
}

fn scalar_vc(t: &Type) -> bool {
    matches!(strip(t), Type::Float { .. } | Type::Int { .. } | Type::Long { .. } | Type::Char | Type::Bool | Type::WChar | Type::Ptr(_) | Type::FuncPtr(_) | Type::Unknown { size: 1 | 2 | 4 | 8 })
}

fn store_size(t: &Type) -> u32 {
    mwdec_lift::scalar_size(t).unwrap_or(4)
}

/// Word-wise copies of a whole object (`lwz/stw` block moves of a class with class members or
/// arrays, POD copy-construction) become one assignment `dst = src`.
fn try_copy(b: &[Stmt], start: usize, end: usize, stores: &[CStore], env: &Env) -> Option<(Vec<Stmt>, usize)> {
    // source pointers' address classes (by expanded form)
    let mut reps: Vec<Expr> = vec![];
    let src_acc: Vec<Option<(usize, i32)>> = stores
        .iter()
        .map(|s| {
            let (p, o) = access(res(&s.src, env.defs), env)?;
            let x = crate::matcher::expand(&p, env.defs);
            let id = match reps.iter().position(|r| *r == x) {
                Some(k) => k,
                None => {
                    reps.push(x);
                    reps.len() - 1
                }
            };
            Some((id, o))
        })
        .collect();
    for (anchor_ix, anchor) in stores.iter().enumerate() {
        let Some((sp, so)) = access(res(&anchor.src, env.defs), env) else { continue };
        let Some((sid, _)) = src_acc[anchor_ix] else { continue };
        let dsrc = so - anchor.off;
        for (fo, cls, size) in crate::addr::aggregates_containing(&anchor.addr, anchor.off, env) {
            if size < 8 {
                continue;
            }
            let lo = fo;
            let hi = fo + size as i32;
            let mut pick: Vec<(usize, &CStore)> = stores.iter().enumerate().filter(|(_, s)| s.off >= lo && s.off < hi && s.aid == anchor.aid).collect();
            pick.sort_by_key(|(_, s)| s.off);
            // exact tiling of the object, every piece copied from the same source layout
            let mut cur = lo;
            let mut ok = true;
            for (k, s) in &pick {
                if s.off != cur {
                    ok = false;
                    break;
                }
                cur += store_size(&s.ty) as i32;
                match src_acc[*k] {
                    Some((id2, o2)) if o2 - s.off == dsrc && id2 == sid && store_size(&s.ty) == mwdec_lift::scalar_size(&mwdec_lift::types::ty_of(res(&s.src, env.defs), env.vars)).unwrap_or(store_size(&s.ty)) => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            let pick: Vec<&CStore> = pick.into_iter().map(|(_, s)| s).collect();
            if !ok || cur != hi || pick.len() < 2 {
                continue;
            }
            // (a class declaring its own out-of-line assignment: `dst = src` would call it)
            if out_of_line_assign(&cls, env) {
                continue;
            }
            // (nor one whose inline assignment is no member-wise copy, `rc_ptr` / `single_ptr`,
            // copied into a member of the method's own object outside a constructor)
            let own_member = matches!(&anchor.addr, Expr::Var(v) if matches!(env.vars[*v].kind, mwdec_lift::ir::VarKind::This));
            if own_member && !IN_CTOR.with(|c| c.get()) && mwdec_lift::aggregates::transfers_on_assign(env.db, &Type::Named(cls.clone())) {
                continue;
            }
            // already member-wise float copies of a small class are handled by the lifter
            let Some((_, dlv)) = object_at(&anchor.addr, lo, &cls, env) else { continue };
            let Some((_, slv)) = object_at(&sp, lo + dsrc, &cls, env) else { continue };
            let mut stmts: Vec<usize> = pick.iter().map(|s| s.stmt).collect();
            stmts.sort_unstable();
            stmts.dedup();
            let covered = stmts.iter().all(|&si| stores.iter().filter(|s| s.stmt == si).all(|s| pick.iter().any(|p| p.stmt == s.stmt && p.off == s.off)));
            if !covered || stmts.len() < 2 {
                // (a single statement is already an object assignment)
                continue;
            }
            let first = *stmts.first().unwrap();
            let last = *stmts.last().unwrap();
            let bad = (first..=last).filter(|k| !stmts.contains(k)).any(|k| {
                let pure_local = matches!(&b[k], Stmt::Assign { dst: Expr::Var(_), src } if !src.has_call());
                touches(&b[k], &anchor.addr, lo, hi, env) || (!pure_local && touches(&b[k], &sp, lo + dsrc, hi + dsrc, env))
            });
            if bad {
                continue;
            }
            let new_stmt = Stmt::Assign { dst: dlv, src: slv };
            let mut out = vec![];
            for k in start..end {
                if k == first {
                    out.push(new_stmt.clone());
                }
                if !stmts.contains(&k) {
                    out.push(b[k].clone());
                }
            }
            return Some((out, first));
        }
    }
    None
}

thread_local! {
    /// Whether the function being folded is a constructor (its members are under construction).
    static IN_CTOR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// An lvalue that is a member of the method's own object (`this->m`, at any depth), written
/// outside a constructor: an assignment to an existing object.
pub fn own_member_assigned(lv: &Expr, env: &Env) -> bool {
    fn root(e: &Expr) -> Option<&Expr> {
        match e {
            Expr::Load { base, .. } | Expr::Member { base, .. } => match &**base {
                b @ Expr::Var(_) => Some(b),
                b => root(b),
            },
            Expr::AddrOf(x) | Expr::Cast { e: x, .. } => root(x),
            _ => None,
        }
    }
    !IN_CTOR.with(|c| c.get()) && matches!(root(lv), Some(Expr::Var(v)) if matches!(env.vars[*v].kind, mwdec_lift::ir::VarKind::This))
}

/// Set for the function about to be folded (see `IN_CTOR`).
pub fn set_in_ctor(v: bool) {
    IN_CTOR.with(|c| c.set(v));
}

/// Does `cls` declare a copy assignment defined out of line?
fn out_of_line_assign(cls: &str, env: &Env) -> bool {
    env.db.decls.get(&format!("{cls}::operator=")).is_some_and(|ds| ds.iter().any(|d| d.params.len() == 1 && !d.is_inline_defined))
}

/// Component stores of one statement.
fn stores_of(i: usize, s: &Stmt, env: &Env) -> Vec<CStore> {
    let Stmt::Assign { dst, src } = s else { return vec![] };
    if src.has_call() {
        // an object member set from a folded inline's value (`r.mPos = Lerp(a, b, t)`): its
        // components are that value's members
        if crate::post::is_folded_value(src) {
            let dty = ty_of(dst, env.vars);
            if let (Some(cls), Some((dp, doff))) = (class_name(&dty, env.db), lvalue_addr(dst, env)) {
                if let Some(fields) = flat_fields(env.db, &cls) {
                    return fields
                        .iter()
                        .map(|(o, t)| CStore { stmt: i, aid: 0, addr: dp.clone(), off: doff + o, ty: t.clone(), src: Expr::Member { base: Box::new(src.clone()), offset: *o, ty: t.clone() } })
                        .collect();
                }
            }
        }
        return vec![];
    }
    let dty = ty_of(dst, env.vars);
    if scalar_vc(&dty) || mwdec_lift::types::is_enum(Some(env.db), &dty) {
        if matches!(dst, Expr::Var(_)) {
            return vec![];
        }
        if let Some((p, o)) = access(dst, env) {
            return vec![CStore { stmt: i, aid: 0, addr: p, off: o, ty: dty, src: src.clone() }];
        }
        return vec![];
    }
    // object copy: expand into components
    let Some(cls) = class_name(&dty, env.db) else { return vec![] };
    let Some(fields) = flat_fields(env.db, &cls) else { return vec![] };
    let Some((dp, doff)) = lvalue_addr(dst, env) else { return vec![] };
    let Some((sp, soff)) = lvalue_addr(res(src, env.defs), env) else { return vec![] };
    fields
        .iter()
        .map(|(o, t)| CStore { stmt: i, aid: 0, addr: dp.clone(), off: doff + o, ty: t.clone(), src: Expr::Load { base: Box::new(sp.clone()), offset: soff + o, ty: t.clone() } })
        .collect()
}

fn is_barrier(s: &Stmt, env: &Env) -> bool {
    match s {
        // folded value inlines are pure (`r.mPos = Lerp(a, b, t)` belongs to a group)
        Stmt::Assign { src, dst } => crate::safety::effect_call(src, env.lib) || crate::safety::effect_call(dst, env.lib),
        Stmt::Comment(_) => false,
        _ => true,
    }
}

/// Does `s` access (read or write) `[lo, hi)` of the object at `addr`?
fn touches(s: &Stmt, addr: &Expr, lo: i32, hi: i32, env: &Env) -> bool {
    let mut hit = false;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if hit {
            return;
        }
        if let Some((p, o)) = access(e, env) {
            if o < hi && o + 4 > lo && teq(&p, addr, env.defs) {
                hit = true;
            }
        }
        if let (Expr::Var(v), Expr::AddrOf(x)) = (e, addr) {
            if matches!(&**x, Expr::Var(w) if w == v) {
                // the whole object read/written as a value
                hit = true;
            }
        }
    });
    hit
}

pub fn rewrite_groups(b: &mut Vec<Stmt>, env: &Env, idx: &Index) -> usize {
    let mut n = 0;
    let mut start = 0;
    let mut seg_limit: Option<usize> = None;
    // (bounded: every rewrite folds stores into fewer statements, but stay safe)
    while start < b.len() && n < 64 {
        // segment [start, end)
        let mut end = start;
        while end < b.len() && !is_barrier(&b[end], env) {
            end += 1;
        }
        if end > start {
            // (retries must not grow the segment without bound: a value rebuilt around itself
            // on every retry grows geometrically)
            let nodes = |ss: &[Stmt]| {
                let mut k = 0usize;
                Stmt::walk_exprs(ss, &mut |_| k += 1);
                k
            };
            let limit = *seg_limit.get_or_insert_with(|| 2 * nodes(&b[start..end]) + 32);
            if let Some((new, rm_lo)) = try_segment(b, start, end, env, idx).filter(|(new, _)| nodes(new) <= limit) {
                if std::env::var("MWDI_TRACE_REWRITE").is_ok() {
                    eprintln!("GROUP rewrite {start}..{end} -> {} stmts: {}", new.len(), format!("{:?}", new).chars().take(300).collect::<String>());
                }
                // replace: new statement list for the segment
                let tail: Vec<Stmt> = b.drain(start..end).collect();
                let _ = tail;
                for (k, s) in new.into_iter().enumerate() {
                    b.insert(start + k, s);
                }
                let _ = rm_lo;
                n += 1;
                continue; // retry the same segment
            }
        }
        start = end + 1;
        seg_limit = None;
    }
    n
}

/// Find one rewritable group in `b[start..end]`; returns the rewritten segment.
fn try_segment(b: &[Stmt], start: usize, end: usize, env: &Env, idx: &Index) -> Option<(Vec<Stmt>, usize)> {
    let mut stores: Vec<CStore> = vec![];
    crate::util::prof::time(8, || {
        for i in start..end {
            stores.extend(stores_of(i, &b[i], env));
        }
    });
    if stores.is_empty() {
        return None;
    }
    // address classes (teq on the canonical pointers = equality of their expansions)
    let mut reps: Vec<Expr> = vec![];
    for s in stores.iter_mut() {
        let x = crate::matcher::expand(&s.addr, env.defs);
        s.aid = match reps.iter().position(|r| *r == x) {
            Some(k) => k,
            None => {
                reps.push(x);
                reps.len() - 1
            }
        };
    }
    if std::env::var("MWDI_TRACE_G").is_ok() {
        for s in &stores {
            eprintln!("STORE stmt {} aid {} +{} {:?} addr {:?}", s.stmt, s.aid, s.off, s.ty, s.addr);
        }
    }
    let mut by_loc: std::collections::HashMap<(usize, i32), Vec<usize>> = std::collections::HashMap::new();
    for (k, s) in stores.iter().enumerate() {
        by_loc.entry((s.aid, s.off)).or_default().push(k);
    }
    if let Some(r) = crate::util::prof::time(9, || try_copy(b, start, end, &stores, env)) {
        return Some(r);
    }
    let mut best: Option<(i32, Vec<Stmt>, usize, bool, Vec<usize>)> = None;
    // the best in-place mutator (for the `x = x op v` / `x op= v` variant point)
    let mut best_mut: Option<(i32, Vec<Stmt>, usize, Vec<usize>)> = None;
    // object_at per (anchor store, offset, class): many templates share a class
    let mut objs: std::collections::HashMap<(usize, i32, String), Option<(Expr, Expr)>> = std::collections::HashMap::new();
    for &ti in &idx.groups {
        let t = &env.lib.templates[ti];
        let (comps, mutate, cls) = match &t.shape {
            Shape::Object { class, comps } => (comps, None, class.clone()),
            Shape::Mutate { hole, comps } => match &t.holes[*hole] {
                HoleKind::Obj { class, .. } => (comps, Some(*hole), class.clone()),
                _ => continue,
            },
            _ => continue,
        };
        let tr = std::env::var("MWDI_TRACE_G").is_ok_and(|f| t.name.contains(f.as_str()));
        // a mutator of plain member stores (`SetTranslation`): only for private members of
        // another class, which the caller could not have written itself
        if mutate.is_some() && t.ops == 0 && t.dead.is_empty() && !foreign_private(&cls, env) {
            continue;
        }
        let min_off = comps.iter().map(|c| c.off).min()?;
        let max_end = comps.iter().map(|c| c.off + 4).max()?;
        let first = comps.iter().find(|c| c.off == min_off)?;
        // (an object built into a member of the method's own object outside a constructor is
        // assigned: no `m = T(...)` for a class whose assignment is no member-wise copy)
        let guarded = mutate.is_none() && !IN_CTOR.with(|c| c.get()) && mwdec_lift::aggregates::transfers_on_assign(env.db, &Type::Named(cls.clone()));
        for (anchor_ix, anchor) in stores.iter().enumerate().filter(|(_, s)| scalar_compat(&s.ty, &first.ty)) {
            if guarded && matches!(&anchor.addr, Expr::Var(v) if matches!(env.vars[*v].kind, mwdec_lift::ir::VarKind::This)) {
                continue;
            }
            let delta = anchor.off - first.off;
            // window: between the neighbouring stores to the anchor's own location
            let same: &[usize] = by_loc.get(&(anchor.aid, anchor.off)).map_or(&[], |v| v.as_slice());
            let wlo = same.iter().map(|&k| &stores[k]).filter(|s| s.stmt < anchor.stmt).map(|s| s.stmt + 1).max().unwrap_or(0);
            let whi = same.iter().map(|&k| &stores[k]).filter(|s| s.stmt > anchor.stmt).map(|s| s.stmt).min().unwrap_or(usize::MAX);
            // every component stored exactly once in the segment, same base
            let mut pick: Vec<&CStore> = vec![];
            let mut ok = true;
            for c in comps {
                let at: &[usize] = by_loc.get(&(anchor.aid, delta + c.off)).map_or(&[], |v| v.as_slice());
                let cands: Vec<&CStore> = at.iter().map(|&k| &stores[k]).filter(|s| s.stmt >= wlo && s.stmt < whi && scalar_compat(&s.ty, &c.ty)).collect();
                if cands.len() != 1 {
                    if tr {
                        eprintln!("G {}: anchor stmt {} comp +{} has {} candidates", t.name, anchor.stmt, c.off, cands.len());
                    }
                    ok = false;
                    break;
                }
                pick.push(cands[0]);
            }
            if !ok {
                continue;
            }
            // statements covered must be covered entirely (object copies expand to several)
            let mut stmts: Vec<usize> = pick.iter().map(|s| s.stmt).collect();
            stmts.sort_unstable();
            stmts.dedup();
            let covered = stmts.iter().all(|&si| stores.iter().filter(|s| s.stmt == si).all(|s| pick.iter().any(|p| p.stmt == s.stmt && p.off == s.off)));
            if !covered {
                continue;
            }
            // one statement that already assigns a whole object of this class (or of a class
            // derived from it: rebuilding `CUnitVector3f u = -v;` member-wise as a `CVector3f`
            // would wrap the value again on every retry)
            if stmts.len() == 1 && matches!(&b[stmts[0]], Stmt::Assign { dst, .. } if class_name(&ty_of(dst, env.vars), env.db).is_some_and(|c| c == cls || crate::util::is_base_or_same(env.db, &cls, &c))) {
                continue;
            }
            if t.ops == 0 && is_copy(&pick, delta, &cls, env) {
                // a plain member-wise copy stays an assignment (but a mutator copying part of an
                // object whose members the function can't name is the call:
                // `xf.SetTranslation(other.GetTranslation())`)
                let whole = flat_fields(env.db, &cls).map_or(true, |f| f.len() == comps.len());
                if mutate.is_none() || whole {
                    continue;
                }
            }
            if matches!(t.kind, crate::probe::CallKind::Ctor) && t.ops == 0 {
                // constructor fallback: only for consecutive stores of a class whose members
                // the function couldn't name anyway (or a stack temporary)
                let mut ss: Vec<usize> = pick.iter().map(|s| s.stmt).collect();
                ss.sort_unstable();
                ss.dedup();
                // (temps loaded in between don't count: `t = src.y; d.x = src.x; d.y = t;`)
                let consecutive = ss.windows(2).all(|w| (w[0] + 1..w[1]).all(|k| matches!(&b[k], Stmt::Assign { dst: Expr::Var(_), src } if !src.has_call())));
                let stack_dst = matches!(&anchor.addr, Expr::AddrOf(x) if matches!(&**x, Expr::Var(_))) || matches!(&anchor.addr, Expr::Var(v) if env.vars[*v].kind == mwdec_lift::VarKind::StructRet);
                let hidden = mwdec_lift::sig::find_class(env.db, &cls).map_or(false, |c| c.fields.iter().any(|f| f.access != mwdec_core::Access::Public));
                // assigning a constructed temporary to a member needs the class's copy
                // assignment; a class declaring its own operator= (smart pointers) is assigned
                // through that instead
                let own_assign = env.db.decls.keys().any(|k| {
                    let base = cls.split('<').next().unwrap_or(&cls);
                    k == &format!("{cls}::operator=") || k == &format!("{base}::operator=")
                });
                if !consecutive || !(stack_dst || hidden) || (!stack_dst && own_assign) {
                    continue;
                }
            }
            let Some((addr, lv)) = crate::util::prof::time(11, || objs.entry((anchor_ix, delta, cls.clone())).or_insert_with(|| object_at(&anchor.addr, delta, &cls, env)).clone()) else {
                if tr {
                    eprintln!("G {}: no object at anchor {:?}+{delta}", t.name, anchor.addr);
                }
                continue;
            };
            let mut m = M::new(env, t);
            if let Some(h) = mutate {
                m.b[h] = Some(Bind::Val(addr.clone()));
            }
            let mut ok = true;
            crate::util::prof::time(10, || {
                for (c, s) in comps.iter().zip(&pick) {
                    if !m.m(&c.pat, &s.src) {
                        ok = false;
                        break;
                    }
                }
            });
            if !ok {
                if tr {
                    eprintln!("G {}: no match {:?} vs {:?}", t.name, comps.iter().map(|c| &c.pat).collect::<Vec<_>>(), pick.iter().map(|s| &s.src).collect::<Vec<_>>());
                }
                continue;
            }
            let Some((args, extra)) = m.finalize(0) else {
                if tr {
                    eprintln!("G {}: finalize failed {:?}", t.name, m.b);
                }
                continue;
            };
            let score = crate::matcher::use_score(t, extra, false, &args);
            if tr {
                eprintln!("G {}: match {:?} from {:?}", t.name, args, pick.iter().map(|s| (s.off, &s.src)).collect::<Vec<_>>());
            }
            // intervening statements must not touch the destination
            let lo = *stmts.first().unwrap();
            let hi = *stmts.last().unwrap();
            let bad = (lo..=hi).filter(|k| !stmts.contains(k)).any(|k| touches(&b[k], &anchor.addr, delta + min_off, delta + max_end, env));
            if bad {
                continue;
            }
            // the sources must not read the destination before all stores are done (aliasing
            // reads of the object being written are fine only for mutators)
            let call = crate::matcher::make_call(t, args);
            let new_stmt = if mutate.is_some() { Stmt::Expr(call) } else { Stmt::Assign { dst: lv, src: call } };
            // the folded statement goes where its last store was when a value it uses is
            // computed between the stores (`d.x = a; t = f(); d.y = t;`)
            let mut defined_between = vec![];
            for k in lo..=hi {
                if !stmts.contains(&k) {
                    if let Stmt::Assign { dst: Expr::Var(v), .. } = &b[k] {
                        defined_between.push(*v);
                    }
                }
            }
            let at = if pick.iter().any(|s| defined_between.iter().any(|v| s.src.uses_var(*v))) { hi } else { lo };
            let mut out = vec![];
            for k in start..end {
                if k == at && at == lo {
                    out.push(new_stmt.clone());
                }
                if !stmts.contains(&k) {
                    out.push(b[k].clone());
                }
                if k == at && at != lo {
                    out.push(new_stmt.clone());
                }
            }
            if mutate.is_some() && best_mut.as_ref().map_or(true, |(b, ..)| score > *b) {
                best_mut = Some((score, out.clone(), lo, stmts.clone()));
            }
            // (a stack object built from nothing: of a mutator and a constructor with the same
            // expansion, the constructor (`CColor c(r, g, b, a)`, not `CColor c; c.Set(...)`))
            let fresh_stack = matches!(&anchor.addr, Expr::AddrOf(x) if matches!(&**x, Expr::Var(v) if matches!(env.vars[*v].kind, mwdec_lift::VarKind::Stack { .. })));
            let ctor_tie = std::env::var("MWDI_NO_CTOR_TIE").is_err() && fresh_stack && mutate.is_none() && matches!(t.kind, crate::probe::CallKind::Ctor) && best.as_ref().is_some_and(|(b, _, _, was_mut, _)| score == *b && *was_mut);
            if best.as_ref().map_or(true, |(b, ..)| score > *b) || ctor_tie {
                best = Some((score, out, lo, mutate.is_some(), stmts));
            }
        }
    }
    let (_, o, l, is_mut, ss) = best?;
    if !is_mut {
        if let Some((_, mo, ml, _)) = best_mut.filter(|(.., mss)| *mss == ss) {
            if mwdec_lift::variants::alt(mwdec_lift::variants::INPLACE_MUTATOR) {
                return Some((mo, ml));
            }
        }
    }
    Some((o, l))
}

/// Does `cls` have non-public members and the function isn't one of its own?
fn foreign_private(cls: &str, env: &Env) -> bool {
    let hidden = mwdec_lift::sig::find_class(env.db, cls).map_or(false, |c| c.fields.iter().any(|f| f.access != mwdec_core::Access::Public));
    let own = env.vars.iter().any(|v| v.kind == mwdec_lift::VarKind::This && mwdec_lift::pointee(&v.ty).and_then(|t| class_name(t, env.db)).as_deref() == Some(cls));
    hidden && !own
}

/// Are the stored values the same-layout components of one other object (a copy)?
fn is_copy(pick: &[&CStore], delta: i32, cls: &str, env: &Env) -> bool {
    let mut base: Option<(Expr, i32)> = None;
    for s in pick {
        let Some((p, o)) = access(res(&s.src, env.defs), env) else { return false };
        let d = o - s.off;
        match &base {
            None => base = Some((p, d)),
            Some((p0, d0)) => {
                if *d0 != d || !teq(p0, &p, env.defs) {
                    return false;
                }
            }
        }
    }
    // an object of the same class lives at the source (not `begin()` = `iterator(mItems)`)
    match &base {
        Some((p0, d0)) => crate::addr::object_at(p0, d0 + delta, cls, env).is_some(),
        None => true,
    }
}

fn scalar_compat(a: &Type, b: &Type) -> bool {
    let fa = matches!(strip(a), Type::Float { .. });
    let fb = matches!(strip(b), Type::Float { .. });
    fa == fb && mwdec_lift::scalar_size(a).unwrap_or(4) == mwdec_lift::scalar_size(b).unwrap_or(4)
}
