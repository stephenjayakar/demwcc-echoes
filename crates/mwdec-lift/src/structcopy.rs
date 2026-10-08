//! Struct copies written member by member in the draft but as one copy in the source.
//!
//! MWCC copies a small class with scalar members (`CVector3f`, ...) member by member, as
//! load/store pairs in member order. Two draft shapes come out of that code when the copy is not
//! recognised before inline folding, and both compile differently from the copy (the source
//! loads every member before storing any, or keeps the members in other registers):
//!
//! - a temporary built from consecutive members of one object:
//!   `CVector3f(p->x, *(float*)((char*)p + 0x40), *(float*)((char*)p + 0x44))` -> `p->pos`
//!   (rendered as the member, or as `*(CVector3f*)((char*)p + 0x3c)`);
//! - a returned object (hidden struct-return pointer) copied from one object and then given the
//!   reference bump of a counted handle's copy constructor (`*rv = this->mPtr; t = rv->GetRefCountPtr();
//!   *t += 1;`) -> `return this->mPtr;`;
//! - a returned object built member by member from values that don't depend on it, for a class
//!   with a constructor taking one argument per member -> `return T(a, b);`.
//!
//! - a run of member stores into a local object, one per member, from one object's members
//!   (rendered `v.SetX(o.GetX()); v.SetY(o.GetY()); ...` for private members) -> `v = o;`. Drafts
//!   built member by member are often exact as they are, so this one is a draft variant
//!   ([`crate::variants::STRUCTCOPY_SETTERS`]), and so is leaving each of the other two alone.
//!
//! Runs on the final body, after inline expansions were folded back into calls.
use crate::ir::*;
use crate::types;
use crate::variants;
use mwdec_core::{Type, TypeDb};

/// Apply both rewrites; returns how many copies were formed.
pub fn apply(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>) -> usize {
    if std::env::var_os("MWDEC_NO_STRUCTCOPY").is_some() {
        return 0;
    }
    let mut n = 0;
    // each rewrite is tried on a copy and asks its variant point only when it applies
    let mut b = body.clone();
    if returned_copy(&mut b, vars, db) && !variants::alt(variants::STRUCTCOPY_NO_RETURN) {
        *body = b;
        n += 1;
    }
    // the returned object copied from one object through its copy constructor's branches
    // (`optional_object`: flag, then the value only when set): a variant
    let mut b = body.clone();
    if returned_whole(&mut b, vars) && variants::alt(variants::STRUCTCOPY_RETURN_WHOLE) {
        *body = b;
        n += 1;
    }
    let mut b = body.clone();
    let mut k = 0;
    for s in b.iter_mut() {
        stmt_exprs(s, &mut |e| k += construct_copies(e, vars, db));
    }
    if k > 0 && !variants::alt(variants::STRUCTCOPY_NO_TEMP) {
        *body = b;
        n += k;
    }
    let mut b = body.clone();
    let mut k = 0;
    Stmt::for_each_block_mut(&mut b, &mut |l| k += setters(l, vars, db));
    if k > 0 && variants::alt(variants::STRUCTCOPY_SETTERS) {
        *body = b;
        n += k;
    }
    n
}

// ------------------------------------------------------------------ temporaries

/// Scalar members (offset, size) of a class in declaration order, if it is a plain class of
/// scalars (no vtable, no bases, no bitfields, no nested aggregates).
fn scalar_members(db: Option<&TypeDb>, t: &Type) -> Option<Vec<(i32, u32)>> {
    let r = types::resolve(db, t).into_owned();
    let c = types::class_of(db, &r)?;
    if c.vptr_offset.is_some() || c.is_union || !c.bases.is_empty() || c.fields.is_empty() {
        return None;
    }
    let mut v = vec![];
    for f in &c.fields {
        if f.bitfield.is_some() {
            return None;
        }
        let ft = types::resolve(db, &f.ty).into_owned();
        if types::is_aggregate(db, &ft) || matches!(strip_cv(&ft), Type::Array(..)) {
            return None;
        }
        v.push((f.offset as i32, types::size_of(db, &ft)?));
    }
    Some(v)
}

/// `base + offset` of a scalar memory access (through a pointer: `Load`).
fn member_access(e: &Expr) -> Option<(&Expr, i32)> {
    match e {
        Expr::Load { base, offset, ty } if scalar_size(ty).is_some() && !matches!(strip_cv(ty), Type::Named(_)) => Some((base, *offset)),
        Expr::Cast { e, .. } => member_access(e),
        _ => None,
    }
}

fn constructed(e: &Expr) -> Option<(&Type, &[Expr])> {
    match e {
        Expr::Construct { class, args, .. } => Some((class, args)),
        _ => None,
    }
}

/// `T(o.a, o.b, o.c)` from every member of `T`, in order, of one object -> that object's `T`.
fn construct_copies(e: &mut Expr, vars: &[Var], db: Option<&TypeDb>) -> usize {
    let mut n = 0;
    for_each_subexpr_mut(e, &mut |x| {
        let Some((class, args)) = constructed(x) else { return };
        let Some(members) = scalar_members(db, class) else { return };
        if members.len() != args.len() || members.len() < 2 {
            return;
        }
        let acc: Vec<Option<(&Expr, i32)>> = args.iter().map(member_access).collect();
        let Some(Some((base, off0))) = acc.first() else { return };
        let start = off0 - members[0].0;
        let ok = acc.iter().zip(&members).all(|(a, (mo, _))| a.is_some_and(|(b, o)| b == *base && o == start + mo));
        if !ok || start < 0 {
            return;
        }
        // sibling members the object's class declares one by one (`CVector2f(mWidth, mHeight)`)
        // are a construction, unless an object of that class sits there (members of different
        // parents mean the declared layout doesn't describe this data: a copy)
        let base_cls = pointee(&types::ty_of(base, vars)).and_then(|t| named(&types::resolve(db, t)).map(String::from));
        if let (Some(db), Some(cls)) = (db, base_cls) {
            let parents: Option<Vec<Vec<types::PathElem>>> = members
                .iter()
                .map(|(mo, sz)| {
                    types::field_path(db, &cls, start + mo, *sz).map(|(mut p, _)| {
                        p.pop();
                        p
                    })
                })
                .collect();
            let siblings = parents.is_some_and(|ps| ps.windows(2).all(|w| w[0] == w[1]));
            if siblings && types::field_path_of_type(db, &cls, start, class).is_none() {
                return;
            }
        }
        let rep = Expr::Load { base: Box::new((*base).clone()), offset: start, ty: class.clone() };
        *x = rep;
        n += 1;
    });
    n
}

// ------------------------------------------------------------------ returned objects

fn mentions_var(e: &Expr, v: VarId) -> bool {
    let mut f = false;
    e.walk(&mut |x| f |= matches!(x, Expr::Var(y) if *y == v));
    f
}

fn stmt_mentions(s: &Stmt, v: VarId) -> bool {
    let mut f = false;
    let mut s = s.clone();
    stmt_exprs(&mut s, &mut |e| f |= mentions_var(e, v));
    f
}

fn has_call(e: &Expr) -> bool {
    let mut f = false;
    e.walk(&mut |x| f |= matches!(x, Expr::Call { .. }) && !crate::postinline::is_inline_call(x));
    f
}

/// The bump of a counted handle's copy: `t = <reads of rv>;` / `*t += 1;` / `++*t`.
fn handle_bump(s: &Stmt, rv: VarId, temps: &mut Vec<VarId>) -> bool {
    let through_temp = |e: &Expr, temps: &[VarId]| matches!(e, Expr::Load { base, .. } if matches!(**base, Expr::Var(t) if temps.contains(&t)));
    match s {
        Stmt::Assign { dst: Expr::Var(t), src } if mentions_var(src, rv) && !has_call(src) => {
            temps.push(*t);
            true
        }
        Stmt::Assign { dst, src: Expr::Binary { op: BinOp::Add, l, r, .. } } => through_temp(dst, temps) && **l == *dst && matches!(**r, Expr::Int { value: 1, .. }),
        Stmt::Expr(Expr::IncDec { e, delta: 1, .. }) => through_temp(e, temps),
        _ => false,
    }
}

/// `a::b<c, d<e>>::f` -> `a::b::f`.
fn strip_args(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for ch in s.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

fn count_mentions(body: &[Stmt], v: VarId) -> usize {
    let mut n = 0;
    let mut b = body.to_vec();
    Stmt::for_each_block_mut(&mut b, &mut |l| n += l.iter().filter(|s| !matches!(s, Stmt::If { .. } | Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::Switch { .. }) && stmt_mentions(s, v)).count());
    n
}

fn ends_in_return(l: &[Stmt]) -> bool {
    match l.last() {
        Some(Stmt::Return(Some(_))) => true,
        Some(Stmt::If { then, els, .. }) => ends_in_return(then) && ends_in_return(els),
        _ => false,
    }
}

fn any_bare_return(body: &[Stmt]) -> bool {
    let mut f = false;
    let mut b = body.to_vec();
    Stmt::for_each_block_mut(&mut b, &mut |l| f |= l.iter().any(|s| matches!(s, Stmt::Return(None))));
    f
}

/// Returned objects (module docs): the statement list that builds the returned object and
/// returns it must hold every use of it; other paths return values of their own.
fn returned_copy(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>) -> bool {
    in_returning_list(body, vars, false, &mut |l, rv| returned_copy_in(l, rv, vars, db))
}

/// Apply `f` to the statement list that holds every use of the returned object (straight-line,
/// or with `if`s when `ifs`), then drop the dead trailing `return __return_value;`; undone if any
/// use or bare `return` remains.
fn in_returning_list(body: &mut Vec<Stmt>, vars: &[Var], ifs: bool, f: &mut dyn FnMut(&mut Vec<Stmt>, VarId) -> bool) -> bool {
    let Some(rv) = vars.iter().position(|v| v.kind == VarKind::StructRet) else { return false };
    let total = count_mentions(body, rv);
    if total == 0 {
        return false;
    }
    let orig = body.clone();
    let mut done = false;
    Stmt::for_each_block_mut(body, &mut |l| {
        if !done && count_mentions(l, rv) == total && l.iter().all(|s| matches!(s, Stmt::Assign { .. } | Stmt::Expr(_) | Stmt::Return(_) | Stmt::Comment(_)) || (ifs && matches!(s, Stmt::If { .. }))) {
            done = f(l, rv);
        }
    });
    if !done {
        return false;
    }
    // the trailing `return __return_value;` after branches that all return is dead
    if matches!(body.last(), Some(Stmt::Return(None))) && body.len() >= 2 && ends_in_return(&body[..body.len() - 1]) {
        body.pop();
    }
    if count_mentions(body, rv) > 0 || any_bare_return(body) {
        *body = orig;
        return false;
    }
    true
}

fn returned_copy_in(body: &mut Vec<Stmt>, rv: VarId, vars: &[Var], db: Option<&TypeDb>) -> bool {
    let Some(class) = pointee(&vars[rv].ty).cloned() else { return false };
    if !matches!(body.last(), Some(Stmt::Return(None))) {
        return false;
    }
    let last = body.len() - 1;
    let Some(first) = body.iter().position(|s| stmt_mentions(s, rv)) else { return false };
    // nothing before the first use may refer to the returned object; it is all straight-line
    if body[first..last].iter().any(|s| !matches!(s, Stmt::Assign { .. } | Stmt::Expr(_))) {
        return false;
    }
    let whole_dst = |e: &Expr| matches!(e, Expr::Load { base, offset: 0, ty } if matches!(**base, Expr::Var(x) if x == rv) && types::resolve(db, strip_cv(ty)).into_owned() == types::resolve(db, strip_cv(&class)).into_owned());
    // (1) whole copy + counted-handle bump
    if let Stmt::Assign { dst, src } = &body[first] {
        if whole_dst(dst) && !mentions_var(src, rv) {
            let mut temps = vec![];
            if body[first + 1..last].iter().all(|s| handle_bump(s, rv, &mut temps)) && !temps.is_empty() {
                let src = src.clone();
                body.splice(first..=last, [Stmt::Return(Some(src))]);
                return true;
            }
        }
    }
    // (2) member-wise construction through a constructor with one parameter per member
    let Some(members) = scalar_members(db, &class) else { return false };
    if last - first != members.len() {
        return false;
    }
    let mut args: Vec<Option<Expr>> = vec![None; members.len()];
    for s in &body[first..last] {
        let Stmt::Assign { dst: Expr::Load { base, offset, .. }, src } = s else { return false };
        if !matches!(**base, Expr::Var(x) if x == rv) || mentions_var(src, rv) {
            return false;
        }
        let Some(k) = members.iter().position(|(o, _)| o == offset) else { return false };
        if args[k].is_some() {
            return false;
        }
        args[k] = Some(src.clone());
    }
    let Some(db) = db else { return false };
    let Some(cls) = named(&class) else { return false };
    let last_name = cls.rsplit("::").next().unwrap_or(cls).split('<').next().unwrap_or(cls);
    // (a template's constructors are declared on the template: `rstl::pair::pair`)
    let generic = strip_args(cls);
    let has_ctor = [format!("{cls}::{last_name}"), format!("{generic}::{last_name}")]
        .iter()
        .any(|k| db.decls.get(k).is_some_and(|ds| ds.iter().any(|d| d.params.len() == members.len())));
    if !has_ctor {
        return false;
    }
    let args: Vec<Expr> = args.into_iter().map(|a| a.unwrap()).collect();
    body.splice(first..=last, [Stmt::Return(Some(Expr::Construct { class: class.clone(), ctor: None, args }))]);
    true
}

// ------------------------------------------------------------------ whole returned object

/// `return x;` for a function whose straight-line body only fills the struct-return object from
/// one object `x` (loads at one fixed distance from the stores, directly or through temporaries),
/// with early `return`s on conditions that don't read the returned object, and values that read
/// nothing from it (a copied flag). The copy constructor's code it stands for is not checked:
/// this is a variant, compiled and kept only when better.
fn returned_whole(body: &mut Vec<Stmt>, vars: &[Var]) -> bool {
    in_returning_list(body, vars, true, &mut |l, rv| returned_whole_in(l, rv, vars))
}

fn returned_whole_in(body: &mut Vec<Stmt>, rv: VarId, vars: &[Var]) -> bool {
    let Some(class) = pointee(&vars[rv].ty).cloned() else { return false };
    if !matches!(body.last(), Some(Stmt::Return(None))) {
        return false;
    }
    let is_rv = |e: &Expr| matches!(e, Expr::Var(x) if *x == rv);
    let mut temps: std::collections::HashMap<VarId, (Expr, i32)> = std::collections::HashMap::new();
    let mut source: Option<(Expr, i32)> = None; // (base, distance src - dst)
    let mut copies = 0;
    for s in &body[..body.len() - 1] {
        match s {
            Stmt::Assign { dst: Expr::Var(t), src } if !mentions_var(src, rv) && !has_call(src) => {
                if let Some((b, o)) = member_access(src) {
                    temps.insert(*t, (b.clone(), o));
                }
            }
            Stmt::Assign { dst: Expr::Load { base, offset, .. }, src } if is_rv(base) => {
                if mentions_var(src, rv) {
                    return false;
                }
                let from = match src {
                    Expr::Var(t) => temps.get(t).cloned(),
                    e => member_access(e).map(|(b, o)| (b.clone(), o)),
                };
                if let Some((b, o)) = from {
                    let here = (b, o - offset);
                    match &source {
                        Some(s0) if *s0 != here => return false,
                        _ => source = Some(here),
                    }
                    copies += 1;
                }
            }
            Stmt::If { cond, then, els } if els.is_empty() && matches!(then.as_slice(), [Stmt::Return(None)]) && !mentions_var(cond, rv) && !has_call(cond) => {}
            _ => return false,
        }
    }
    let Some((base, dist)) = source.filter(|_| copies >= 2) else { return false };
    if dist < 0 {
        return false;
    }
    *body = vec![Stmt::Return(Some(Expr::Load { base: Box::new(base), offset: dist, ty: class }))];
    true
}

// ------------------------------------------------------------------ member runs

/// `v.m = <src>` into a member of local object `v`: (v, member offset, value).
fn member_store(s: &Stmt) -> Option<(VarId, i32, &Expr)> {
    match s {
        Stmt::Assign { dst: Expr::Member { base, offset, ty }, src } if scalar_size(ty).is_some() => match **base {
            Expr::Var(v) => Some((v, *offset, src)),
            _ => None,
        },
        _ => None,
    }
}

/// A run of member stores into local object `v`, one per scalar member of its class, from the
/// members at the same relative offsets of one object behind a pointer (the value read directly
/// or through a temporary defined just before): `v = *(T*)(p + s)`. Temporaries stay, read back
/// from `v`. (Rendered with setters/getters when the members are private: `v.SetX(o.GetX())`.)
fn setters(b: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < b.len() {
        // value of a store: a load, or a temp defined by the previous statement as a load
        let value_at = |b: &[Stmt], k: usize| -> Option<(VarId, i32, Expr, i32, Option<usize>)> {
            let (v, off, src) = member_store(&b[k])?;
            if let Some((base, so)) = member_access(src) {
                return Some((v, off, base.clone(), so, None));
            }
            let Expr::Var(t) = src else { return None };
            let Some(Stmt::Assign { dst: Expr::Var(td), src: ld }) = k.checked_sub(1).map(|j| &b[j]) else { return None };
            if td != t {
                return None;
            }
            let (base, so) = member_access(ld)?;
            Some((v, off, base.clone(), so, Some(k - 1)))
        };
        let Some((v, off0, base, so0, td0)) = value_at(b, i) else {
            i += 1;
            continue;
        };
        let Some(members) = scalar_members(db, &vars[v].ty) else {
            i += 1;
            continue;
        };
        let delta = so0 - off0;
        let mut run: Vec<(usize, i32, Option<usize>)> = vec![(i, off0, td0)];
        let mut j = i + 1;
        while j < b.len() && run.len() < members.len() {
            let k = if value_at(b, j).is_none() && j + 1 < b.len() { j + 1 } else { j };
            match value_at(b, k) {
                Some((v2, o2, b2, s2, td)) if v2 == v && b2 == base && s2 - o2 == delta && (k == j || td == Some(j)) => {
                    run.push((k, o2, td));
                    j = k + 1;
                }
                _ => break,
            }
        }
        let mut offs: Vec<i32> = run.iter().map(|r| r.1).collect();
        offs.sort();
        let want: Vec<i32> = members.iter().map(|m| m.0).collect();
        if offs != want || delta + want[0] < 0 {
            i = j.max(i + 1);
            continue;
        }
        let first_stmt = run.iter().map(|(k, _, td)| td.unwrap_or(*k)).min().unwrap();
        let class = vars[v].ty.clone();
        let mut keep: Vec<Stmt> = vec![Stmt::Assign { dst: Expr::Var(v), src: Expr::Load { base: Box::new(base.clone()), offset: delta, ty: class } }];
        for k in first_stmt..j {
            if run.iter().any(|(r, _, _)| *r == k) {
                continue;
            }
            let mut s = b[k].clone();
            // a temporary feeding the run re-reads the member from the copy
            if let Some((_, o, _)) = run.iter().find(|(_, _, td)| *td == Some(k)) {
                if let Stmt::Assign { src, .. } = &mut s {
                    if let Expr::Load { ty, .. } = src {
                        *src = Expr::Member { base: Box::new(Expr::Var(v)), offset: *o, ty: ty.clone() };
                    }
                }
            }
            keep.push(s);
        }
        let len = keep.len();
        b.splice(first_stmt..j, keep);
        n += 1;
        i = first_stmt + len;
    }
    n
}

// ------------------------------------------------------------------ traversal

fn stmt_exprs(s: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match s {
        Stmt::Expr(e) => f(e),
        Stmt::Assign { dst, src } => {
            f(dst);
            f(src);
        }
        Stmt::If { cond, then, els } => {
            f(cond);
            then.iter_mut().for_each(|s| stmt_exprs(s, f));
            els.iter_mut().for_each(|s| stmt_exprs(s, f));
        }
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            f(cond);
            body.iter_mut().for_each(|s| stmt_exprs(s, f));
        }
        Stmt::For { init, cond, step, body } => {
            f(cond);
            init.iter_mut().chain(step.iter_mut()).chain(body.iter_mut()).for_each(|s| stmt_exprs(s, f));
        }
        Stmt::Switch { e, cases } => {
            f(e);
            for c in cases {
                c.body.iter_mut().for_each(|s| stmt_exprs(s, f));
            }
        }
        Stmt::Return(Some(e)) => f(e),
        _ => {}
    }
}

/// Post-order over `e` and its subexpressions.
fn for_each_subexpr_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    match e {
        Expr::AddrOf(x) | Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } | Expr::IncDec { e: x, .. } => for_each_subexpr_mut(x, f),
        Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => for_each_subexpr_mut(base, f),
        Expr::Index { base, index, .. } => {
            for_each_subexpr_mut(base, f);
            for_each_subexpr_mut(index, f);
        }
        Expr::Binary { l, r, .. } => {
            for_each_subexpr_mut(l, f);
            for_each_subexpr_mut(r, f);
        }
        Expr::Ternary { c, t, f: g, .. } => {
            for_each_subexpr_mut(c, f);
            for_each_subexpr_mut(t, f);
            for_each_subexpr_mut(g, f);
        }
        Expr::Call { callee, args, .. } => {
            match callee {
                Callee::Method { this, .. } | Callee::Virtual { this, .. } => for_each_subexpr_mut(this, f),
                Callee::Indirect(x) => for_each_subexpr_mut(x, f),
                Callee::Direct { .. } => {}
            }
            args.iter_mut().for_each(|a| for_each_subexpr_mut(a, f));
        }
        Expr::New { placement, args, .. } => placement.iter_mut().chain(args.iter_mut()).for_each(|a| for_each_subexpr_mut(a, f)),
        Expr::Construct { args, .. } => args.iter_mut().for_each(|a| for_each_subexpr_mut(a, f)),
        _ => {}
    }
    f(e);
}
