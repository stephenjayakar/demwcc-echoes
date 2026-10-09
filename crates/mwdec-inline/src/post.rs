//! Clean-ups after folding: stack objects that only carried a folded value into one call become
//! the call's argument (`CAABox box(pos - d, pos + d)`), and one object value used twice (the
//! target shares its components) is named once (`CVector3f delta = a - b;`).

use mwdec_core::Type;
use mwdec_lift::{Callee, Expr, Stmt, Var, VarId, VarKind};
use std::collections::HashMap;

fn mentions_body(body: &[Stmt]) -> HashMap<VarId, usize> {
    let mut m = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            *m.entry(*v).or_default() += 1;
        }
    });
    m
}

fn mentions(s: &Stmt, v: VarId) -> usize {
    let mut n = 0;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if matches!(e, Expr::Var(x) if *x == v) {
            n += 1;
        }
    });
    n
}

pub fn is_folded_value(e: &Expr) -> bool {
    folded_value(e)
}

/// A value our pass produced: a folded inline call or constructor (no side effects).
fn folded_value(e: &Expr) -> bool {
    match e {
        Expr::Call { callee: Callee::Direct { sig, .. } | Callee::Method { sig, .. }, .. } => sig.mangled.is_none(),
        Expr::Construct { .. } => true,
        _ => false,
    }
}

fn pure_stmt(s: &Stmt, vars: &[Var]) -> bool {
    match s {
        Stmt::Assign { dst: Expr::Var(v), src } => {
            matches!(vars[*v].kind, VarKind::Local | VarKind::Stack { .. }) && (!src.has_call() || folded_value(src))
        }
        _ => false,
    }
}

/// Replace the argument `&v` / `v` of a call or construction in `e` by `with`. Returns the
/// number of replacements.
/// `real_call`: `with` is a real call's by-value result (also the object of a const inline and
/// the base of a member read).
fn replace_arg(e: &mut Expr, v: VarId, with: &Expr, only_folded: bool, real_call: bool) -> usize {
    let mut n = 0;
    let is_v = |a: &Expr| match a {
        Expr::Var(x) => *x == v,
        Expr::AddrOf(x) => matches!(&**x, Expr::Var(y) if *y == v),
        _ => false,
    };
    e.rewrite(&mut |x| {
        // a member read of the object: of the value itself (`GetAverage().m_valid`, read
        // through its accessor)
        if let Expr::Member { base, .. } = &mut *x {
            if real_call && matches!(&**base, Expr::Var(w) if *w == v) {
                **base = with.clone();
                n += 1;
            }
            return;
        }
        let sig = match &*x {
            Expr::Call { callee: Callee::Direct { sig, .. } | Callee::Method { sig, .. }, .. } => Some(sig.clone()),
            Expr::Construct { ctor, .. } | Expr::New { ctor, .. } => ctor.clone(),
            Expr::Call { .. } => None,
            _ => return,
        };
        // (a stack object built by a real constructor call moves only into folded inlines)
        if only_folded && !sig.as_ref().is_some_and(|s| s.mangled.is_none()) {
            return;
        }
        // the object of a const member function call: `(a - b).MagSquared()` (a real call's
        // result too, into a folded inline: `GetAverage().valid()`)
        if let (true, Expr::Call { callee: Callee::Method { this, sig: msig, .. }, .. }) = (!only_folded || real_call, &mut *x) {
            if msig.is_const && matches!(&**this, Expr::AddrOf(y) if matches!(&**y, Expr::Var(w) if *w == v)) {
                *this = Box::new(Expr::AddrOf(Box::new(with.clone())));
                n += 1;
            }
        }
        let args = match x {
            Expr::Call { args, .. } | Expr::Construct { args, .. } | Expr::New { args, .. } => args,
            _ => return,
        };
        for (k, a) in args.iter_mut().enumerate() {
            // only where the parameter takes the object itself (reference or by value), not
            // its address
            let by_ref = sig.as_ref().and_then(|s| s.params.get(k)).map_or(false, |p| matches!(crate::util::strip(&p.ty), Type::Ref(_) | Type::Named(_)));
            if is_v(a) && by_ref {
                *a = with.clone();
                n += 1;
            }
        }
    });
    n
}

fn stmt_replace_arg(s: &mut Stmt, v: VarId, with: &Expr, only_folded: bool, real_call: bool) -> usize {
    match s {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => replace_arg(e, v, with, only_folded, real_call),
        Stmt::Assign { dst, src } => replace_arg(src, v, with, only_folded, real_call) + replace_arg(dst, v, with, only_folded, real_call),
        // (a condition, evaluated before the branches)
        Stmt::If { cond, .. } => replace_arg(cond, v, with, only_folded, real_call),
        _ => 0,
    }
}

pub fn forward_stack_temps(body: &mut Vec<Stmt>, vars: &[Var], dead: &[mwdec_lift::DeadStackStore]) -> usize {
    let counts = mentions_body(body);
    // a by-value argument copy with the same value stored, dead, just above it: the copy of a
    // named local the source declared before the call (`T id = f(); g(id);`), not a temporary
    let named_copy = |v: VarId| match vars[v].kind {
        VarKind::Stack { offset, size } => dead.iter().any(|d| d.offset > offset && d.offset <= offset + 16 && d.size == size),
        _ => false,
    };
    let mut n = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let (v, val, only_folded, real_call) = match &b[i] {
                Stmt::Assign { dst: Expr::Var(v), src }
                    if matches!(vars[*v].kind, VarKind::Stack { .. } | VarKind::Local) && matches!(crate::util::strip(&vars[*v].ty), Type::Named(_)) && folded_value(src) && counts.get(v) == Some(&2) && !named_copy(*v) =>
                {
                    (*v, src.clone(), false, false)
                }
                // a stack object holding a real call's by-value result, passed on once into a
                // folded inline (`Deltas(Lerp(..), Slerp(..))`)
                Stmt::Assign { dst: Expr::Var(v), src: src @ Expr::Call { callee: Callee::Direct { sig, .. } | Callee::Method { sig, .. } | Callee::Virtual { sig: Some(sig), .. }, .. } }
                    if sig.mangled.is_some() && matches!(vars[*v].kind, VarKind::Stack { .. }) && matches!(crate::util::strip(&vars[*v].ty), Type::Named(_)) && counts.get(v) == Some(&2) =>
                {
                    (*v, src.clone(), true, true)
                }
                // a stack object built by a constructor call, passed on once (`X(CAABox(a, b))`)
                Stmt::Expr(Expr::Call { callee: Callee::Method { symbol, sig, this, .. }, args, .. })
                    if symbol.starts_with("__ct__") && matches!(&**this, Expr::AddrOf(x) if matches!(&**x, Expr::Var(v) if matches!(vars[*v].kind, VarKind::Stack { .. }) && counts.get(v) == Some(&2))) =>
                {
                    let Expr::AddrOf(x) = &**this else { unreachable!() };
                    let Expr::Var(v) = &**x else { unreachable!() };
                    (*v, Expr::Construct { class: vars[*v].ty.clone(), ctor: Some(sig.clone()), args: args.clone() }, true, false)
                }
                _ => {
                    i += 1;
                    continue;
                }
            };
            let Some(j) = (i + 1..b.len()).find(|&k| mentions(&b[k], v) > 0) else {
                i += 1;
                continue;
            };
            if !(i + 1..j).all(|k| pure_stmt(&b[k], vars)) {
                i += 1;
                continue;
            }
            let mut s = b[j].clone();
            if stmt_replace_arg(&mut s, v, &val, only_folded, real_call) == 1 && mentions(&s, v) == 0 {
                b[j] = s;
                b.remove(i);
                n += 1;
                continue;
            }
            i += 1;
        }
    });
    n
}

fn collect_folded<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    e.walk(&mut |x| {
        // (a non-const method has effects: two `in.Read()` are two reads, whatever the
        // typedef'd result type looks like)
        let mutator = matches!(x, Expr::Call { callee: Callee::Method { sig, .. }, .. } if !sig.is_const);
        // (`l.end()` / `l.begin()`: the source calls the accessors again)
        let bounds = matches!(x, Expr::Call { callee: Callee::Method { sig, .. }, args, .. } if args.is_empty() && matches!(sig.qualified_name.rsplit("::").next(), Some("begin" | "end")));
        if folded_value(x) && !mutator && !bounds && matches!(x, Expr::Call { ret, .. } if matches!(crate::util::strip(ret), Type::Named(_))) {
            out.push(x);
        }
    });
}

fn stmt_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => vec![e],
        Stmt::Assign { dst, src } => vec![dst, src],
        Stmt::If { cond, .. } | Stmt::While { cond, .. } => vec![cond],
        _ => vec![],
    }
}

/// Is `s` free of stores to memory and real calls?
fn no_effects(s: &Stmt) -> bool {
    match s {
        Stmt::Assign { dst: Expr::Var(_), src } => {
            let mut ok = true;
            src.walk(&mut |x| {
                if let Expr::Call { .. } = x {
                    if !folded_value(x) {
                        ok = false;
                    }
                }
            });
            ok
        }
        _ => false,
    }
}

pub fn name_shared_objects(body: &mut Vec<Stmt>, vars: &mut Vec<Var>) -> usize {
    let mut n = 0;
    let mut new_vars: Vec<Var> = vec![];
    let base = vars.len();
    Stmt::for_each_block_mut(body, &mut |b| {
        loop {
            // first duplicated folded object value in this list
            let mut found: Option<(Expr, usize, usize)> = None;
            'outer: for i in 0..b.len() {
                let mut here = vec![];
                for e in stmt_exprs(&b[i]) {
                    collect_folded(e, &mut here);
                }
                for e in here {
                    for k in i..b.len() {
                        let mut there = vec![];
                        for x in stmt_exprs(&b[k]) {
                            collect_folded(x, &mut there);
                        }
                        let cnt = there.iter().filter(|x| ***x == *e).count();
                        if (k == i && cnt >= 2) || (k > i && cnt >= 1) {
                            found = Some(((*e).clone(), i, k));
                            break 'outer;
                        }
                    }
                }
            }
            let Some((e, i, k)) = found else { break };
            // nothing in between may change the operands
            let last = (i..b.len()).rev().find(|&x| stmt_exprs(&b[x]).iter().any(|y| {
                let mut v = vec![];
                collect_folded(y, &mut v);
                v.iter().any(|z| **z == e)
            })).unwrap_or(k);
            if !(i..last).all(|x| x == i || no_effects(&b[x])) || !(i..=last).all(|x| x == i || x == last || no_effects(&b[x])) {
                break;
            }
            let Expr::Call { ret, .. } = &e else { break };
            let id = base + new_vars.len();
            new_vars.push(Var { name: format!("vec{}", new_vars.len()), ty: ret.clone(), kind: VarKind::Local });
            for x in i..=last {
                let mut rw = |y: &mut Expr| {
                    if *y == e {
                        *y = Expr::Var(id);
                    }
                };
                Stmt::rewrite_exprs(std::slice::from_mut(&mut b[x]), &mut rw);
            }
            b.insert(i, Stmt::Assign { dst: Expr::Var(id), src: e });
            n += 1;
        }
    });
    vars.extend(new_vars);
    n
}

/// A local holding a folded inline result that the next statement's condition (or returned
/// value) uses once: `b = (it == end()); if (b)` becomes `if (it == end())`. MWCC materialises
/// a control-flow inline's value and re-tests it either way.
pub fn forward_cond_temps(body: &mut Vec<Stmt>, vars: &[Var]) -> usize {
    let counts = mentions_body(body);
    let mut n = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            let fwd = match &b[i] {
                Stmt::Assign { dst: Expr::Var(v), src } if matches!(vars[*v].kind, VarKind::Local) && folded_value(src) && counts.get(v) == Some(&2) && mentions(&b[i + 1], *v) == 1 => {
                    let ok = match &b[i + 1] {
                        Stmt::If { cond, .. } => cond.uses_var(*v),
                        Stmt::Return(Some(e)) => e.uses_var(*v),
                        _ => false,
                    };
                    // (an algorithm's iterator may have been a named local: a variant)
                    let algorithm = matches!(src, Expr::Call { callee: Callee::Direct { sig, .. }, .. } if sig.qualified_name.starts_with("rstl::") && sig.this_class.is_none());
                    let ok = ok && !(algorithm && mwdec_lift::variants::alt(mwdec_lift::variants::NAMED_ALGORITHM_RESULT));
                    ok.then(|| (*v, src.clone()))
                }
                _ => None,
            };
            if let Some((v, src)) = fwd {
                let sub = |e: &mut Expr| {
                    e.rewrite(&mut |x| {
                        if matches!(x, Expr::Var(y) if *y == v) {
                            *x = src.clone();
                        }
                    })
                };
                match &mut b[i + 1] {
                    Stmt::If { cond, .. } => sub(cond),
                    Stmt::Return(Some(e)) => sub(e),
                    _ => {}
                }
                b.remove(i);
                n += 1;
                continue;
            }
            i += 1;
        }
    });
    n
}

/// `*__return = value; return;` -> `return value;` (the returned object built by a folded
/// inline or constructor).
pub fn return_values(body: &mut Vec<Stmt>, vars: &[Var]) -> usize {
    let sret = |e: &Expr| matches!(e, Expr::Load { base, offset: 0, .. } if matches!(&**base, Expr::Var(v) if vars[*v].kind == VarKind::StructRet));
    let counts = mentions_body(body);
    let Some(rv) = vars.iter().position(|v| v.kind == VarKind::StructRet) else { return 0 };
    let mut n = 0;
    let mut writes = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        writes += b.iter().filter(|s| matches!(s, Stmt::Assign { dst, .. } if sret(dst))).count();
    });
    // every mention of the return slot is one of these writes
    if counts.get(&rv).copied().unwrap_or(0) != writes {
        return 0;
    }
    let value = |s: &Stmt| matches!(s, Stmt::Assign { dst, src } if sret(dst) && (folded_value(src) || matches!(src, Expr::Construct { .. })));
    // `if (c) { *ret = a; } else { *ret = b; } return;` -> returns in both arms
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            let arms = match (&b[i], &b[i + 1]) {
                (Stmt::If { then, els, .. }, Stmt::Return(None)) => then.last().is_some_and(&value) && els.last().is_some_and(&value),
                _ => false,
            };
            if arms {
                if let Stmt::If { then, els, .. } = &mut b[i] {
                    for arm in [then, els] {
                        arm.push(Stmt::Return(None));
                    }
                }
                b.remove(i + 1);
                n += 1;
            }
            i += 1;
        }
    });
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            if let (Stmt::Assign { dst, src }, Stmt::Return(None)) = (&b[i], &b[i + 1]) {
                if sret(dst) && (folded_value(src) || matches!(src, Expr::Construct { .. })) {
                    let v = src.clone();
                    b[i + 1] = Stmt::Return(Some(v));
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

/// Single-use temporaries loaded early (the compiler scheduled the loads ahead of the stores a
/// folded inline replaced) move into the folded value that consumes them:
/// `t1 = p->y; t2 = p->z; return T(p->x, t1, t2);` -> `return T(p->x, p->y, p->z);`. Only
/// pure reads move, past pure temp definitions that don't redefine what they read.
pub fn forward_temps_into_folded(body: &mut Vec<Stmt>, vars: &[Var]) -> usize {
    let counts = mentions_body(body);
    let mut n = 0;
    let is_temp = |v: VarId| matches!(vars[v].kind, VarKind::Local) && vars[v].name.starts_with("temp_");
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = b.len();
        while i > 0 {
            i -= 1;
            let Stmt::Assign { dst: Expr::Var(v), src } = &b[i] else { continue };
            let (v, src) = (*v, src.clone());
            if !is_temp(v) || counts.get(&v) != Some(&2) || src.has_call() || matches!(src, Expr::IncDec { .. }) {
                continue;
            }
            let Some(j) = (i + 1..b.len()).find(|&k| mentions(&b[k], v) > 0) else { continue };
            // the consumer: a folded value taking the temp as an argument
            let mut in_folded = false;
            for e in stmt_exprs(&b[j]) {
                e.walk(&mut |x| {
                    if folded_value(x) {
                        let args = match x {
                            Expr::Call { args, .. } | Expr::Construct { args, .. } => args,
                            _ => return,
                        };
                        if args.iter().any(|a| a.uses_var(v)) {
                            in_folded = true;
                        }
                    }
                });
            }
            if !in_folded {
                continue;
            }
            let mut read = vec![];
            src.walk(&mut |x| {
                if let Expr::Var(w) = x {
                    read.push(*w);
                }
            });
            let between_ok = (i + 1..j).all(|k| match &b[k] {
                Stmt::Assign { dst: Expr::Var(w), src: s2 } => is_temp(*w) && !s2.has_call() && !read.contains(w),
                _ => false,
            });
            if !between_ok {
                continue;
            }
            let mut s = b[j].clone();
            let mut hits = 0;
            let sub = &mut |x: &mut Expr| {
                if matches!(x, Expr::Var(y) if *y == v) {
                    *x = src.clone();
                    hits += 1;
                }
            };
            match &mut s {
                Stmt::Expr(e) | Stmt::Return(Some(e)) => e.rewrite(sub),
                Stmt::Assign { dst, src: s2 } => {
                    s2.rewrite(sub);
                    dst.rewrite(sub);
                }
                _ => continue,
            }
            if hits == 1 {
                b[j] = s;
                b.remove(i);
                n += 1;
            }
        }
    });
    n
}

/// A flag built by nested tests of folded inlines (`v = false; if (a()) { t = x; if (b(t))
/// v = true; }`) is the `&&` chain of the tests: `v = a() && b(x)`, and a flag read once
/// right after is replaced by its value (`return !(a() && b(x));`).
pub fn fold_flag_chains(body: &mut Vec<Stmt>, vars: &[Var]) -> usize {
    let counts = mentions_body(body);
    let mut n = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            let v = match &b[i] {
                Stmt::Assign { dst: Expr::Var(v), src: Expr::Int { value: 0, .. } } if matches!(vars[*v].kind, VarKind::Local) => *v,
                _ => {
                    i += 1;
                    continue;
                }
            };
            let Some(chain) = flag_chain(&b[i + 1], v, &counts, vars) else {
                i += 1;
                continue;
            };
            if !has_folded(&chain) {
                i += 1;
                continue;
            }
            let mut chain = chain;
            // pure single-use temps computed right before the flag go into the chain too
            while i > 0 {
                let Stmt::Assign { dst: Expr::Var(t), src } = &b[i - 1] else { break };
                let (t, src) = (*t, src.clone());
                if !matches!(vars[t].kind, VarKind::Local) || src.has_call() || counts.get(&t) != Some(&2) {
                    break;
                }
                let mut hits = 0;
                let mut c2 = chain.clone();
                c2.rewrite(&mut |x| {
                    if matches!(x, Expr::Var(y) if *y == t) {
                        *x = src.clone();
                        hits += 1;
                    }
                });
                if hits != 1 {
                    break;
                }
                chain = c2;
                b.remove(i - 1);
                i -= 1;
            }
            // the flag read once by the next statement: its value goes there
            let uses = counts.get(&v).copied().unwrap_or(0);
            let next_ok = i + 2 < b.len() && uses == 3 && mentions(&b[i + 2], v) == 1 && matches!(&b[i + 2], Stmt::Return(Some(_)) | Stmt::If { .. });
            if next_ok {
                let mut s = b[i + 2].clone();
                let sub = &mut |x: &mut Expr| {
                    if matches!(x, Expr::Var(y) if *y == v) {
                        *x = chain.clone();
                    }
                };
                match &mut s {
                    Stmt::Return(Some(e)) => e.rewrite(sub),
                    Stmt::If { cond, .. } => cond.rewrite(sub),
                    _ => {}
                }
                b[i + 2] = s;
                b.drain(i..i + 2);
            } else {
                b.splice(i..i + 2, [Stmt::Assign { dst: Expr::Var(v), src: chain }]);
            }
            n += 1;
            i += 1;
        }
    });
    n
}

fn has_folded(e: &Expr) -> bool {
    let mut f = false;
    e.walk(&mut |x| f |= folded_value(x));
    f
}

/// `if (c) { pure temps; <inner> }` where the innermost statement is `v = true`: the chain.
fn flag_chain(s: &Stmt, v: VarId, counts: &HashMap<VarId, usize>, vars: &[Var]) -> Option<Expr> {
    let Stmt::If { cond, then, els } = s else { return None };
    if !els.is_empty() || then.is_empty() {
        return None;
    }
    let (last, temps) = then.split_last()?;
    let mut rest = match last {
        Stmt::Assign { dst: Expr::Var(x), src: Expr::Int { value: 1, .. } } if *x == v && temps.is_empty() => return Some(cond.clone()),
        Stmt::If { .. } => flag_chain(last, v, counts, vars)?,
        _ => return None,
    };
    // single-use pure temps computed between the tests move into the later test
    for t in temps.iter().rev() {
        let Stmt::Assign { dst: Expr::Var(tv), src } = t else { return None };
        if !matches!(vars[*tv].kind, VarKind::Local) || src.has_call() || counts.get(tv) != Some(&2) {
            return None;
        }
        let mut hits = 0;
        rest.rewrite(&mut |x| {
            if matches!(x, Expr::Var(y) if y == tv) {
                *x = src.clone();
                hits += 1;
            }
        });
        if hits != 1 {
            return None;
        }
    }
    Some(Expr::Binary { op: mwdec_lift::BinOp::LogAnd, l: Box::new(cond.clone()), r: Box::new(rest), ty: Type::Bool })
}
