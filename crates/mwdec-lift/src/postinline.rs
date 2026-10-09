//! Clean-ups for bodies after recognised inline expansions were folded back into calls.

use crate::ir::*;
use std::collections::HashMap;

/// A call that stands for an inline expansion (built after lifting: no symbol of its own).
pub fn is_inline_call(e: &Expr) -> bool {
    match e {
        Expr::Call { callee: Callee::Direct { symbol, sig }, .. } => *symbol == sig.qualified_name && sig.mangled.as_deref().map_or(true, |m| m != symbol),
        Expr::Call { callee: Callee::Method { symbol, .. }, .. } => symbol.is_empty(),
        _ => false,
    }
}

/// An inline call written as a call (operators render infix, where a forwarded argument would
/// need its own parentheses).
fn is_plain_inline_call(e: &Expr) -> bool {
    match e {
        Expr::Call { callee: Callee::Direct { sig, .. } | Callee::Method { sig, .. }, .. } => is_inline_call(e) && !sig.qualified_name.contains("operator"),
        _ => false,
    }
}

fn count_defs(body: &mut Vec<Stmt>) -> HashMap<VarId, usize> {
    let mut n: HashMap<VarId, usize> = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(v), .. } = s {
                *n.entry(*v).or_default() += 1;
            }
        }
    });
    n
}

/// Does `e` contain a call other than inline expansions?
fn has_real_call(e: &Expr) -> bool {
    let mut found = false;
    e.walk(&mut |x| match x {
        Expr::Call { .. } if !is_inline_call(x) => found = true,
        Expr::New { .. } | Expr::IncDec { .. } => found = true,
        _ => {}
    });
    found
}

fn stmt_has_real_call(s: &Stmt) -> bool {
    match s {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => has_real_call(e),
        Stmt::Assign { dst, src } => has_real_call(dst) || has_real_call(src),
        _ => true,
    }
}

/// Reads memory (or anything a call could change)?
fn reads_memory(e: &Expr) -> bool {
    let mut found = false;
    e.walk(&mut |x| {
        if matches!(x, Expr::Load { .. } | Expr::Member { .. } | Expr::Index { .. } | Expr::BitField { .. } | Expr::Global { .. } | Expr::Call { .. } | Expr::New { .. } | Expr::IncDec { .. }) {
            found = true;
        }
    });
    found
}

/// Replace the single argument `Var(v)` of an inline call in `e` by `val`. True if replaced.
fn subst_inline_arg(e: &mut Expr, v: VarId, val: &Expr) -> bool {
    let mut done = false;
    e.rewrite(&mut |x| {
        if done || !is_plain_inline_call(x) {
            return;
        }
        if let Expr::Call { args, .. } = x {
            for a in args.iter_mut() {
                if matches!(a, Expr::Var(y) if *y == v) {
                    *a = val.clone();
                    done = true;
                    return;
                }
            }
        }
    });
    done
}

/// `v = e; x = inl(.., v, ..);` with `v` a local defined and read only there is
/// `x = inl(.., e, ..)`: a named local bound to the inline's (reference) parameter is a
/// different object from the argument temporary and changes the expansion's registers and
/// branch layout (`b` over an empty arm of a min/max select). Assignments of other locals in
/// between (no calls, not read by `e`) stay before the use.
/// A pointer walking a container's elements in step with a counted loop (`p = v.data(); for (i =
/// 0; i < n; i++) { ... *p ...; p = p + sizeof(T); }`) is the container indexed by the counter:
/// `v[i]` (its `operator[]`). Written as a pointer walk, MWCC unrolls the loop; the source indexed
/// the container. Returns the number of loops rewritten.
pub fn container_index_walks(body: &mut Vec<Stmt>, _vars: &[Var], db: Option<&mwdec_core::TypeDb>) -> usize {
    fn uncast(e: &Expr) -> &Expr {
        match e {
            Expr::Cast { e, .. } => uncast(e),
            e => e,
        }
    }
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    let mut n = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut k = 1;
        while k < b.len() {
            let found = (|| {
                let Stmt::For { init, cond, step, body: lb } = &b[k] else { return None };
                let [Stmt::Assign { dst: Expr::Var(i), src: i0 }] = init.as_slice() else { return None };
                let i = *i;
                if i0.as_int() != Some(0) {
                    return None;
                }
                let Expr::Binary { op: BinOp::Lt, l, .. } = cond else { return None };
                if !matches!(uncast(l), Expr::Var(x) if *x == i) || step.len() != 1 {
                    return None;
                }
                // the walk's start right before the loop: `p = (cast)container.data()`
                let Stmt::Assign { dst: Expr::Var(p), src: ps } = &b[k - 1] else { return None };
                let p = *p;
                let Expr::Call { callee: Callee::Method { this, sig, .. }, args, ret } = uncast(ps) else { return None };
                if !args.is_empty() || !sig.qualified_name.ends_with("::data") {
                    return None;
                }
                let elem = strip_cv(pointee(ret)?).clone();
                let size = crate::types::size_of(db, &elem).filter(|&z| z > 0)? as i64;
                // the container object (`this` is its address; without the const view the
                // accessor was called through)
                let mut obj = (**this).clone();
                while let Expr::Cast { e, .. } = obj {
                    obj = *e;
                }
                let Expr::AddrOf(obj) = obj else { return None };
                let obj = *obj;
                // its only step, last in the body: `p = p + size`
                let (last, rest) = lb.split_last()?;
                let Stmt::Assign { dst: Expr::Var(q), src: qs } = last else { return None };
                let stepped = match uncast(qs) {
                    Expr::Binary { op: BinOp::Add, l, r, .. } => matches!(uncast(l), Expr::Var(x) if *x == p) && r.as_int() == Some(size),
                    Expr::AddrOf(x) => matches!(&**x, Expr::Load { base, offset, .. } if matches!(uncast(base), Expr::Var(y) if *y == p) && *offset as i64 == size),
                    _ => false,
                };
                if *q != p || !stepped {
                    return None;
                }
                // p read only in the body (plus its start and its step)
                let assigned = {
                    let mut a = false;
                    let mut rc = rest.to_vec();
                    Stmt::for_each_block_mut(&mut rc, &mut |bb| a |= bb.iter().any(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if *x == p)));
                    a
                };
                let mut local: HashMap<VarId, usize> = HashMap::new();
                crate::inline::count_uses(&b[k - 1..=k], &mut local);
                if uses.get(&p) != local.get(&p) || assigned {
                    return None;
                }
                let element = Expr::Index { base: Box::new(obj), index: Box::new(Expr::Var(i)), ty: elem.clone() };
                let elem_at = Expr::AddrOf(Box::new(element.clone()));
                let mut nb: Vec<Stmt> = rest.to_vec();
                Stmt::rewrite_exprs(&mut nb, &mut |e| {
                    if matches!(e, Expr::Var(x) if *x == p) {
                        *e = elem_at.clone();
                    }
                });
                // the element written member by member from one object at the same offsets: a
                // whole-element copy (`v[i] = *src;`)
                let word = |s: &Stmt| -> Option<(i32, i64, Expr)> {
                    let Stmt::Assign { dst: Expr::Load { base: db_, offset: o, ty: t }, src: Expr::Load { base: sb, offset: so, ty: st } } = s else { return None };
                    if **db_ != elem_at || o != so || strip_cv(t) != strip_cv(st) {
                        return None;
                    }
                    Some((*o, crate::types::size_of(db, t)? as i64, (**sb).clone()))
                };
                let mut j = 0;
                while j < nb.len() {
                    let mut covered = 0i64;
                    let mut src: Option<Expr> = None;
                    let mut m = j;
                    while m < nb.len() {
                        let Some((o, z, s)) = word(&nb[m]) else { break };
                        if o as i64 != covered || src.as_ref().is_some_and(|x| *x != s) {
                            break;
                        }
                        src = Some(s);
                        covered += z;
                        m += 1;
                    }
                    if covered == size && m > j {
                        let mut s = src.unwrap();
                        // (a source address computed right before, read only here, folds in)
                        let mut at = j;
                        if let Expr::Var(t) = s {
                            if j > 0 && uses.get(&t) == Some(&(m - j)) {
                                if let Stmt::Assign { dst: Expr::Var(x), src: ts } = &nb[j - 1] {
                                    let mut only_inline = true;
                                    ts.walk(&mut |c| only_inline &= !matches!(c, Expr::Call { .. }) || is_inline_call(c));
                                    if *x == t && only_inline {
                                        s = ts.clone();
                                        at = j - 1;
                                    }
                                }
                            }
                        }
                        nb.splice(at..m, [Stmt::Assign { dst: element.clone(), src: Expr::Load { base: Box::new(s), offset: 0, ty: elem.clone() } }]);
                        j = at;
                    }
                    j += 1;
                }
                Some(nb)
            })();
            match found {
                Some(nb) => {
                    if let Stmt::For { body: lb, .. } = &mut b[k] {
                        *lb = nb;
                    }
                    b.remove(k - 1);
                    n += 1;
                }
                None => k += 1,
            }
        }
    });
    n
}

pub fn forward_inline_args(body: &mut Vec<Stmt>, vars: &[Var]) -> usize {
    let defs = count_defs(body);
    let mut uses = HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    let mut n = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            let mut target = None;
            if let Stmt::Assign { dst: Expr::Var(v), src } = &b[i] {
                // the next statement reading v, past assignments that can't change `e`
                let mut j = i + 1;
                while j < b.len() && j <= i + 4 && !crate::idioms::stmt_mentions(&b[j], *v) {
                    match &b[j] {
                        Stmt::Assign { dst: Expr::Var(w), src: s } if !src.uses_var(*w) && !has_real_call(s) => j += 1,
                        _ => break,
                    }
                }
                if j < b.len() && crate::idioms::stmt_mentions(&b[j], *v) {
                    // read only there: the only def and use, or that statement redefines it from
                    // its single read (`v = e; v = inl(.., v, ..)`)
                    let only_here = defs.get(v) == Some(&1) && uses.get(v) == Some(&1);
                    let redefined = matches!(&b[j], Stmt::Assign { dst: Expr::Var(w), src: s2 } if w == v && {
                        let mut u = HashMap::new();
                        crate::inline::count_uses(std::slice::from_ref(&b[j]), &mut u);
                        u.get(v) == Some(&1) && s2.uses_var(*v)
                    });
                    // a value merged with the inline's result (several definitions when lifted,
                    // `var_`) or the variable the result goes back to; a single-definition temp
                    // the lifter kept apart was a variable of its own
                    let merged = vars.get(*v).map_or(false, |x| x.name.starts_with("var_"));
                    let ok = matches!(vars.get(*v).map(|x| &x.kind), Some(VarKind::Local))
                        // (past other assignments only when the use redefines it)
                        && ((only_here && merged && j == i + 1) || redefined)
                        && !has_real_call(src)
                        && !src.uses_var(*v)
                        // nothing evaluated before the argument may change what `e` reads
                        && (!reads_memory(src) || !stmt_has_real_call(&b[j]));
                    if ok {
                        target = Some(j);
                    }
                }
            }
            if let Some(j) = target {
                let Stmt::Assign { dst: Expr::Var(v), src } = b[i].clone() else { unreachable!() };
                let replaced = match &mut b[j] {
                    Stmt::Assign { dst, src: s2 } => (matches!(dst, Expr::Var(w) if *w == v) || !dst.uses_var(v)) && subst_inline_arg(s2, v, &src),
                    Stmt::Expr(e) | Stmt::Return(Some(e)) => subst_inline_arg(e, v, &src),
                    _ => false,
                };
                if replaced {
                    b.remove(i);
                    n += 1;
                    continue;
                }
            }
            i += 1;
        }
    });
    n
}

/// Variant point [`crate::variants::ORDER_ADDRESS_FIRST`]: the first `v = X; t = &obj...;` pair of
/// adjacent independent temp assignments (no calls or stores, `t` a pointer, `v` not) swapped.
pub fn address_temps_first(body: &mut Vec<Stmt>, vars: &[Var]) -> usize {
    fn pure(e: &Expr) -> bool {
        !e.has_call()
    }
    let mut at = None;
    let mut seen = false;
    Stmt::for_each_block_mut(body, &mut |b| {
        if seen {
            return;
        }
        for k in 0..b.len().saturating_sub(1) {
            let (Stmt::Assign { dst: Expr::Var(v), src: x }, Stmt::Assign { dst: Expr::Var(t), src: y }) = (&b[k], &b[k + 1]) else { continue };
            if v == t || !is_ptr(&vars[*t].ty) || is_ptr(&vars[*v].ty) || !pure(x) || !pure(y) {
                continue;
            }
            if !matches!(vars[*v].kind, VarKind::Local) || !matches!(vars[*t].kind, VarKind::Local) {
                continue;
            }
            if y.uses_var(*v) || x.uses_var(*t) || !matches!(y, Expr::AddrOf(_)) {
                continue;
            }
            seen = true;
            if crate::variants::alt(crate::variants::ORDER_ADDRESS_FIRST) {
                at = Some(k);
                b.swap(k, k + 1);
            }
            return;
        }
    });
    at.map_or(0, |_| 1)
}

/// Variant point [`crate::variants::ORDER_SPLIT_LAST_FIELD`]: the first `v = a | b | c | (x..);`
/// (at least four or-ed field inserts into a local) becomes `v = a | b | c; v = v | (x..);`.
pub fn split_last_field(body: &mut Vec<Stmt>, vars: &[Var]) -> usize {
    fn field(e: &Expr) -> bool {
        matches!(e, Expr::Binary { op: BinOp::Shl | BinOp::And, .. })
    }
    fn ors(e: &Expr) -> usize {
        match e {
            Expr::Binary { op: BinOp::Or, l, r, .. } => ors(l) + ors(r),
            _ => 1,
        }
    }
    let mut done = false;
    let mut n = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        if done {
            return;
        }
        for k in 0..b.len() {
            let Stmt::Assign { dst: Expr::Var(v), src: Expr::Binary { op: BinOp::Or, l, r, ty } } = &b[k] else { continue };
            if !matches!(vars[*v].kind, VarKind::Local) || ors(l) < 3 || !field(r) || l.uses_var(*v) || r.uses_var(*v) {
                continue;
            }
            done = true;
            if crate::variants::alt(crate::variants::ORDER_SPLIT_LAST_FIELD) {
                let (v, l, mut r, ty) = (*v, (**l).clone(), (**r).clone(), ty.clone());
                // the field masked before it is shifted (`(x & m) << s`, as an `|=` statement is written)
                if let Expr::Binary { op: BinOp::And, l: sx, r: mm, ty: aty } = &r {
                    if let (Expr::Binary { op: BinOp::Shl, l: x, r: sh, .. }, Some(m)) = (&**sx, mm.as_int()) {
                        if let Some(k) = sh.as_int().filter(|k| (1..32).contains(k)) {
                            if m & ((1 << k) - 1) == 0 {
                                r = Expr::bin(BinOp::Shl, Expr::bin(BinOp::And, (**x).clone(), Expr::uint(m >> k), aty.clone()), Expr::int(k), aty.clone());
                            }
                        }
                    }
                }
                b[k] = Stmt::Assign { dst: Expr::Var(v), src: l };
                b.insert(k + 1, Stmt::Assign { dst: Expr::Var(v), src: Expr::bin(BinOp::Or, Expr::Var(v), r, ty) });
                n = 1;
            }
            return;
        }
    });
    n
}

/// Variant point [`crate::variants::BOOL_CONST_LOCAL`]: bool locals defined once (where they are
/// declared) become `const bool`, and a returned `&&`/`||` chain is returned through one
/// (`const bool r = a || b; return r;`). The compiler re-extends a const bool at each use.
pub fn const_bool_locals(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, ret_bool: bool) -> usize {
    let defs = count_defs(body);
    let decl_here = crate::idioms::decl_at_first_def_scalars(body, vars);
    let locals: Vec<VarId> = (0..vars.len())
        .filter(|&v| {
            matches!(vars[v].kind, VarKind::Local) && matches!(vars[v].ty, mwdec_core::Type::Bool) && defs.get(&v) == Some(&1) && decl_here.contains(&v)
        })
        .collect();
    let mut chains = 0;
    if ret_bool {
        Stmt::for_each_block_mut(body, &mut |b| {
            chains += b.iter().filter(|s| matches!(s, Stmt::Return(Some(Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, .. })))).count();
        });
    }
    if (locals.is_empty() && chains == 0) || !crate::variants::alt(crate::variants::BOOL_CONST_LOCAL) {
        return 0;
    }
    let cb = mwdec_core::Type::Const(Box::new(mwdec_core::Type::Bool));
    for &v in &locals {
        vars[v].ty = cb.clone();
    }
    let mut n = locals.len();
    if chains > 0 {
        let mut fresh = vec![];
        Stmt::for_each_block_mut(body, &mut |b| {
            let mut k = 0;
            while k < b.len() {
                if let Stmt::Return(Some(e @ Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, .. })) = &b[k] {
                    let e = e.clone();
                    let v = vars.len() + fresh.len();
                    fresh.push(v);
                    b[k] = Stmt::Assign { dst: Expr::Var(v), src: e };
                    b.insert(k + 1, Stmt::Return(Some(Expr::Var(v))));
                    k += 1;
                }
                k += 1;
            }
        });
        for (i, _) in fresh.iter().enumerate() {
            let name = if i == 0 { "result".to_string() } else { format!("result{i}") };
            vars.push(Var { name, ty: cb.clone(), kind: VarKind::Local });
            n += 1;
        }
    }
    n
}

/// `Member { base: r }` with `r` a reference (a folded inline's object reached through a reference
/// member or local, `mPlayer.GetTranslation()`) is the access through it: `Load { base: r }`, which
/// renders through the referent's members and accessors like any pointer access.
pub fn reference_members(body: &mut Vec<Stmt>, vars: &[Var]) -> usize {
    let mut n = 0;
    Stmt::rewrite_exprs(body, &mut |e| {
        if let Expr::Member { base, offset, ty } = &*e {
            if matches!(crate::types::ty_of(base, vars), mwdec_core::Type::Ref(_)) {
                *e = Expr::Load { base: base.clone(), offset: *offset, ty: ty.clone() };
                n += 1;
            }
        }
    });
    n
}

/// Variant point [`crate::variants::LOCALS_NARROW_BY_DEFS`]: an `int` local every write of which
/// is a narrowing cast to one small type (`v = (u8)(v | x)`) or a constant of that type, with at
/// least one cast, is a local of that type (`u8 v; v |= x;`: the compiler's `clrlwi` into the
/// variable's register). Its writes lose the casts; returns the retyped locals (their updates
/// from themselves are written as compound assignments).
pub fn narrow_by_defs(body: &mut Vec<Stmt>, vars: &mut [Var]) -> Vec<VarId> {
    use mwdec_core::Type;
    let n = vars.len();
    let mut ty: Vec<Option<Type>> = vec![None; n];
    let mut bad = vec![false; n];
    let mut casts = vec![0usize; n];
    let mut consts: Vec<Vec<i64>> = vec![vec![]; n];
    let fits = |c: i64, t: &Type| match t {
        Type::Int { size: 1, signed: false } => (0..=0xff).contains(&c),
        Type::Int { size: 2, signed: false } => (0..=0xffff).contains(&c),
        Type::Int { size: 1, signed: true } => (-0x80..=0x7f).contains(&c),
        Type::Int { size: 2, signed: true } => (-0x8000..=0x7fff).contains(&c),
        _ => false,
    };
    {
        let mut snap = body.clone();
        Stmt::for_each_block_mut(&mut snap, &mut |b| {
            for s in b.iter() {
                let Stmt::Assign { dst: Expr::Var(v), src } = s else { continue };
                let v = *v;
                if !matches!(vars[v].kind, VarKind::Local) || !matches!(strip_cv(&vars[v].ty), Type::Int { size: 4, .. } | Type::Unknown { size: 4 }) {
                    bad[v] = true;
                    continue;
                }
                match src {
                    Expr::Int { value, .. } => consts[v].push(*value),
                    Expr::Cast { ty: ct, e } if matches!(strip_cv(ct), Type::Int { size: 1 | 2, .. }) && !matches!(**e, Expr::Int { .. }) => {
                        let t = strip_cv(ct).clone();
                        casts[v] += 1;
                        match &ty[v] {
                            None => ty[v] = Some(t),
                            Some(c) if *c == t => {}
                            Some(_) => bad[v] = true,
                        }
                    }
                    _ => bad[v] = true,
                }
            }
        });
    }
    let chosen: Vec<VarId> = (0..n).filter(|&v| matches!(&ty[v], Some(t) if !bad[v] && casts[v] > 0 && consts[v].iter().all(|&c| fits(c, t)))).collect();
    if chosen.is_empty() || !crate::variants::alt(crate::variants::LOCALS_NARROW_BY_DEFS) {
        return vec![];
    }
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter_mut() {
            if let Stmt::Assign { dst: Expr::Var(v), src } = s {
                if chosen.contains(v) {
                    if let Expr::Cast { e, .. } = src {
                        *src = (**e).clone();
                    }
                }
            }
        }
    });
    for &v in &chosen {
        vars[v].ty = ty[v].clone().unwrap();
    }
    chosen
}
