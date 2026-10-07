//! Expression and statement clean-ups on the structured IR (semantics-preserving).

use crate::ir::*;
use crate::types::ty_of;
use mwdec_core::Type;
use std::collections::HashMap;

fn is_boolish(e: &Expr, vars: &[Var]) -> bool {
    match e {
        Expr::BitField { width: 1, .. } => true,
        Expr::Binary { op, .. } if op.is_bool() => true,
        Expr::Unary { op: UnOp::Not, .. } => true,
        _ => matches!(strip_cv(&ty_of(e, vars)), Type::Bool),
    }
}

/// One bottom-up rewrite step.
fn uncast(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => uncast(e),
        e => e,
    }
}

fn is_neg_of(e: &Expr, x: &Expr) -> bool {
    matches!(uncast(e), Expr::Unary { op: UnOp::Neg, e: y, .. } if uncast(y) == uncast(x))
}

/// MWCC's branchless 0/1 materialisations (value-context comparisons) back to comparisons.
fn bool_idiom(e: &Expr) -> Option<Expr> {
    let Expr::Binary { op: BinOp::Shr, l, r, .. } = e else { return None };
    if r.as_int() != Some(31) {
        return None;
    }
    match uncast(l) {
        // (-a | a) >> 31  ==  a != 0
        Expr::Binary { op: BinOp::Or, l: p, r: q, .. } if is_neg_of(p, q) => Some(Expr::cmp(BinOp::Ne, uncast(q).clone(), Expr::int(0))),
        Expr::Binary { op: BinOp::Or, l: p, r: q, .. } if is_neg_of(q, p) => Some(Expr::cmp(BinOp::Ne, uncast(p).clone(), Expr::int(0))),
        // (-a & ~a) >> 31  ==  a > 0
        Expr::Binary { op: BinOp::And, l: p, r: q, .. } if is_neg_of(p, q) => None,
        Expr::Binary { op: BinOp::And, l: p, r: q, .. } => match uncast(q) {
            Expr::Unary { op: UnOp::BitNot, e: a, .. } if is_neg_of(p, a) => Some(Expr::cmp(BinOp::Gt, uncast(a).clone(), Expr::int(0))),
            _ => None,
        },
        // ((b - a) | (a - b)) >> 31  ==  a != b
        Expr::Binary { op: BinOp::Or, l: p, r: q, .. } => match (uncast(p), uncast(q)) {
            (Expr::Binary { op: BinOp::Sub, l: b1, r: a1, .. }, Expr::Binary { op: BinOp::Sub, l: a2, r: b2, .. }) if a1 == a2 && b1 == b2 => {
                Some(Expr::cmp(BinOp::Ne, (**a1).clone(), (**b1).clone()))
            }
            _ => None,
        },
        // ((b ^ a) >>s 1) - ((b ^ a) & b) >> 31  ==  a < b (signed)
        Expr::Binary { op: BinOp::Sub, l: p, r: q, .. } => match (uncast(p), uncast(q)) {
            (Expr::Binary { op: BinOp::Shr, l: x1, r: one, .. }, Expr::Binary { op: BinOp::And, l: x2, r: b, .. }) if one.as_int() == Some(1) && uncast(x1) == uncast(x2) => {
                if let Expr::Binary { op: BinOp::Xor, l: b1, r: a1, .. } = uncast(x1) {
                    if uncast(b1) == uncast(b) {
                        return Some(Expr::cmp(BinOp::Lt, uncast(a1).clone(), uncast(b1).clone()));
                    }
                }
                None
            }
            _ => None,
        },
        // (b << cntlzw(b ^ a)) >> 31  ==  a < b (unsigned)
        Expr::Binary { op: BinOp::Shl, l: b, r: c, .. } => match uncast(c) {
            Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. } if symbol == "__cntlzw" => match args.first().map(uncast) {
                Some(Expr::Binary { op: BinOp::Xor, l: b1, r: a1, .. }) if uncast(b1) == uncast(b) => {
                    Some(Expr::cmp(BinOp::Lt, Expr::cast(t_u32(), uncast(a1).clone()), Expr::cast(t_u32(), uncast(b1).clone())))
                }
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

fn simp(e: &mut Expr, vars: &[Var]) {
    if let Some(n) = bool_idiom(e) {
        *e = n;
        return;
    }
    // (x & a) & b  ->  x & (a & b)
    if let Expr::Binary { op: BinOp::And, l, r, ty } = e {
        if let (Expr::Binary { op: BinOp::And, l: x, r: a, .. }, Some(b)) = (&**l, r.as_int()) {
            if let Some(a) = a.as_int() {
                let m = (a as u32 & b as u32) as i64;
                *e = Expr::Binary { op: BinOp::And, l: x.clone(), r: Box::new(Expr::uint(m)), ty: ty.clone() };
                return;
            }
        }
    }
    // (signed char)c where c is a plain char (signed in MWCC): no-op
    if let Expr::Cast { ty: Type::Int { size: 1, signed: true }, e: x } = e {
        if matches!(strip_cv(&ty_of(x, vars)), Type::Char | Type::Int { size: 1, signed: true }) {
            *e = (**x).clone();
            return;
        }
    }
    // (a >> 31) ^ 1  ==  a >= 0
    if let Expr::Binary { op: BinOp::Xor, l, r, .. } = e {
        if r.as_int() == Some(1) {
            if let Expr::Binary { op: BinOp::Shr, l: a, r: s, .. } = uncast(l) {
                if s.as_int() == Some(31) {
                    *e = Expr::cmp(BinOp::Ge, Expr::cast(t_s32(), uncast(a).clone()), Expr::int(0));
                    return;
                }
            }
        }
    }
    // (double)x + M - M  ==  x  for the int->float conversion magic (after temps are folded)
    if let Expr::Binary { op: BinOp::Sub, l, r, ty } = e {
        if let (Expr::Binary { op: BinOp::Add, l: x, r: m1, .. }, Expr::Float { bits, double: true }) = (&**l, &**r) {
            if matches!(**m1, Expr::Float { bits: b1, double: true } if b1 == *bits) && (*bits == 0x4330_0000_8000_0000 || *bits == 0x4330_0000_0000_0000) {
                let inner = match &**x {
                    Expr::Cast { e, .. } => (**e).clone(),
                    o => o.clone(),
                };
                *e = Expr::cast(ty.clone(), inner);
                return;
            }
        }
    }
    // a - a / b * b  ==  a % b  (MWCC lowers % to divw/mullw/subf)
    if let Expr::Binary { op: BinOp::Sub, l: a, r, ty } = e {
        if let Expr::Binary { op: BinOp::Mul, l: q, r: b, .. } = &**r {
            if let Expr::Binary { op: BinOp::Div, l: a2, r: b2, .. } = &**q {
                let strip = |x: &Expr| -> Expr {
                    match x {
                        Expr::Cast { e, .. } => (**e).clone(),
                        x => x.clone(),
                    }
                };
                if strip(a2) == strip(a) && strip(b2) == strip(b) {
                    *e = Expr::Binary { op: BinOp::Rem, l: a2.clone(), r: b2.clone(), ty: ty.clone() };
                    return;
                }
            }
        }
    }
    // single-bit bitfield used as a truth value / masked with 1
    if let Expr::Binary { op: BinOp::And, l, r, .. } = e {
        if r.as_int() == Some(1) && matches!(**l, Expr::BitField { width: 1, .. }) {
            *e = (**l).clone();
            return;
        }
    }
    // __cntlzw(x) >> 5  ==  (x == 0)
    if let Expr::Binary { op: BinOp::Shr, l, r, .. } = e {
        if r.as_int() == Some(5) {
            let inner = match &**l {
                Expr::Cast { e: x, .. } => &**x,
                x => x,
            };
            if let Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. } = inner {
                if symbol == "__cntlzw" && args.len() == 1 {
                    // value-context `a == b` is `subf ; cntlzw ; srwi 5`
                    if let Expr::Binary { op: BinOp::Sub, l: x, r: y, .. } = uncast(&args[0]) {
                        if y.as_int().is_none() {
                            *e = Expr::cmp(BinOp::Eq, (**y).clone(), (**x).clone());
                            return;
                        }
                    }
                    *e = Expr::cmp(BinOp::Eq, args[0].clone(), Expr::int(0));
                    return;
                }
            }
        }
    }
    // (u8)b where b is bool -> b
    if let Expr::Cast { ty, e: inner } = e {
        if matches!(ty, Type::Int { size: 1, signed: false }) && is_boolish(inner, vars) {
            *e = (**inner).clone();
            return;
        }
        // cast to the type it already has (in C terms too: `int & 0x7f800000` is int in C even
        // when the IR typed it unsigned)
        let c_agrees = match ty {
            Type::Int { size: 4, signed } => crate::translate::c_unsigned(inner, vars) == !*signed,
            _ => true,
        };
        if ty_of(inner, vars) == *ty && !matches!(ty, Type::Unknown { .. }) && c_agrees {
            *e = (**inner).clone();
            return;
        }
        // (T)(T)x
        if let Expr::Cast { ty: t2, e: x } = &**inner {
            if t2 == ty {
                *e = Expr::Cast { ty: ty.clone(), e: x.clone() };
                return;
            }
        }
        // (s32)(u32)x where x is s32 / (u32)(s32)x
        if let (Type::Int { size: 4, .. }, Expr::Cast { ty: Type::Int { size: 4, .. }, e: x }) = (&*ty, &**inner) {
            if ty_of(x, vars) == *ty {
                *e = (**x).clone();
                return;
            }
        }
        // casting an int literal
        if let Expr::Int { value, .. } = **inner {
            match ty {
                Type::Int { size: 4, signed } => {
                    *e = Expr::Int { value: if *signed { value as i32 as i64 } else { value as u32 as i64 }, ty: ty.clone() };
                    return;
                }
                _ => {}
            }
        }
    }
    if let Expr::Binary { op, l, r, ty } = e {
        // cmp(bool, 0)
        if matches!(op, BinOp::Eq | BinOp::Ne) && r.as_int() == Some(0) && is_boolish(l, vars) {
            *e = if *op == BinOp::Ne { (**l).clone() } else { (**l).clone().negate(vars) };
            return;
        }
        if matches!(op, BinOp::Eq | BinOp::Ne) && r.as_int() == Some(1) && is_boolish(l, vars) {
            *e = if *op == BinOp::Eq { (**l).clone() } else { (**l).clone().negate(vars) };
            return;
        }
        // const on the left of a compare
        if op.is_cmp() && l.as_int().is_some() && r.as_int().is_none() {
            let nop = op.swap_cmp();
            *e = Expr::Binary { op: nop, l: r.clone(), r: l.clone(), ty: ty.clone() };
            return;
        }
        // x + (-k) -> x - k
        if *op == BinOp::Add {
            if let Some(k) = r.as_int() {
                if k < 0 && k > i32::MIN as i64 && matches!(**r, Expr::Int { ty: Type::Int { signed: true, .. }, .. }) {
                    *op = BinOp::Sub;
                    **r = Expr::int(-k);
                    return;
                }
            }
        }
        // (bool cmp) == 0 handled; LogAnd with constant true
        if *op == BinOp::LogAnd {
            if matches!(**l, Expr::Int { value: 1, ty: Type::Bool }) {
                *e = (**r).clone();
                return;
            }
        }
    }
    if let Expr::Unary { op: UnOp::Not, e: inner, .. } = e {
        match &**inner {
            Expr::Unary { op: UnOp::Not, e: x, .. } if is_boolish(x, vars) => {
                *e = (**x).clone();
            }
            Expr::Binary { op, l, r, .. } if op.is_cmp() => {
                let float = is_float(&ty_of(l, vars)) || is_float(&ty_of(r, vars));
                if !float || matches!(op, BinOp::Eq | BinOp::Ne) {
                    *e = (**inner).clone().negate(vars);
                }
            }
            _ => {}
        }
    }
}

pub fn simplify_body(body: &mut Vec<Stmt>, vars: &[Var]) {
    for _ in 0..3 {
        Stmt::rewrite_exprs(body, &mut |e| simp(e, vars));
    }
    // conditions: `x != 0` for ints stays explicit only when x isn't boolish
    Stmt::for_each_block_mut(body, &mut |blk| {
        for s in blk.iter_mut() {
            match s {
                Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
                    simp_cond(cond, vars);
                }
                _ => {}
            }
        }
    });
}

fn simp_cond(c: &mut Expr, vars: &[Var]) {
    // `(u8)x != 0` with a bool-ish int: keep. `p != 0` -> `p` for pointers.
    if let Expr::Binary { op: BinOp::Ne, l, r, .. } = c {
        if r.as_int() == Some(0) && is_ptr(&ty_of(l, vars)) {
            *c = (**l).clone();
        }
    } else if let Expr::Binary { op: BinOp::Eq, l, r, .. } = c {
        if r.as_int() == Some(0) && is_ptr(&ty_of(l, vars)) {
            *c = Expr::not((**l).clone());
        }
    }
    if let Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, l, r, .. } = c {
        simp_cond(l, vars);
        simp_cond(r, vars);
    }
}

/// `return;` at the very end of a void function is implicit (also at the end of the branches
/// of a trailing `if`).
pub fn drop_trailing_return(body: &mut Vec<Stmt>) {
    if matches!(body.last(), Some(Stmt::Return(None))) {
        body.pop();
    }
    if let Some(Stmt::If { then, els, .. }) = body.last_mut() {
        drop_trailing_return(then);
        drop_trailing_return(els);
    }
}

/// Locals only ever assigned 0/1 constants or boolean expressions become `bool`.
pub fn refine_bool_vars(body: &[Stmt], vars: &mut [Var]) {
    let mut ok: HashMap<VarId, bool> = HashMap::new();
    fn walk(b: &[Stmt], vars: &[Var], ok: &mut HashMap<VarId, bool>) {
        for s in b {
            match s {
                Stmt::Assign { dst: Expr::Var(v), src } => {
                    let good = matches!(src, Expr::Int { value: 0 | 1, .. }) || is_boolish(src, vars) || matches!(src, Expr::Var(w) if matches!(vars[*w].ty, Type::Bool));
                    let e = ok.entry(*v).or_insert(true);
                    *e &= good;
                }
                Stmt::If { then, els, .. } => {
                    walk(then, vars, ok);
                    walk(els, vars, ok);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => walk(body, vars, ok),
                Stmt::For { init, step, body, .. } => {
                    walk(init, vars, ok);
                    walk(step, vars, ok);
                    walk(body, vars, ok);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        walk(&c.body, vars, ok);
                    }
                }
                _ => {}
            }
        }
    }
    walk(body, vars, &mut ok);
    // `v++` anywhere (e.g. in a loop condition) makes it a counter
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::IncDec { e: x, .. } = e {
            if let Expr::Var(v) = &**x {
                ok.insert(*v, false);
            }
        }
    });
    for (v, good) in ok {
        if good && matches!(vars[v].kind, VarKind::Local) && matches!(vars[v].ty, Type::Unknown { .. } | Type::Int { .. }) {
            vars[v].ty = Type::Bool;
        }
    }
}

/// `i = a; while (c(i)) { ...; i = i + k; }` -> `for (i = a; c(i); i = i + k) { ... }`.
pub fn form_for_loops(body: &mut Vec<Stmt>) {
    Stmt::for_each_block_mut(body, &mut |blk| {
        let mut k = 1;
        while k < blk.len() {
            let is_loop = matches!(blk[k], Stmt::While { .. });
            if is_loop {
                if let (Stmt::Assign { dst: Expr::Var(iv), .. }, Stmt::While { cond, body: lb }) = (&blk[k - 1], &blk[k]) {
                    let iv = *iv;
                    let step_ok = matches!(lb.last(), Some(Stmt::Assign { dst: Expr::Var(v), src }) if *v == iv && src.uses_var(iv));
                    let body_has_continue = contains_continue(lb);
                    if cond.uses_var(iv) && step_ok && !body_has_continue {
                        let init = blk[k - 1].clone();
                        if let Stmt::While { cond, mut body } = blk.remove(k) {
                            let step = body.pop().unwrap();
                            blk[k - 1] = Stmt::For { init: vec![init], cond, step: vec![step], body };
                            continue;
                        }
                    }
                }
            }
            k += 1;
        }
    });
}

fn contains_continue(b: &[Stmt]) -> bool {
    b.iter().any(|s| match s {
        Stmt::Continue => true,
        Stmt::If { then, els, .. } => contains_continue(then) || contains_continue(els),
        Stmt::Switch { cases, .. } => cases.iter().any(|c| contains_continue(&c.body)),
        _ => false,
    })
}

/// Give used variables unique, readable names.
pub fn name_vars(body: &[Stmt], vars: &mut [Var]) {
    let mut used: HashMap<VarId, ()> = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            used.insert(*v, ());
        }
    });
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (i, v) in vars.iter().enumerate() {
        if matches!(v.kind, VarKind::This | VarKind::Param { .. } | VarKind::StructRet) {
            seen.insert(v.name.clone(), 1);
            let _ = i;
        }
    }
    let mut ids: Vec<VarId> = used.keys().copied().collect();
    ids.sort();
    for id in ids {
        if matches!(vars[id].kind, VarKind::This | VarKind::Param { .. } | VarKind::StructRet) {
            continue;
        }
        let base = vars[id].name.clone();
        let n = seen.entry(base.clone()).or_insert(0);
        *n += 1;
        if *n > 1 {
            vars[id].name = format!("{}_{}", base, n);
        }
    }
}

/// Element step of `x = x + k` style updates of variable `x` (pointer: k bytes / pointee size).
fn step_of(src: &Expr, x: VarId, vars: &[Var], db: Option<&mwdec_core::TypeDb>) -> Option<i64> {
    let xt = vars[x].ty.clone();
    let is_x = |e: &Expr| matches!(uncast(e), Expr::Var(v) if *v == x);
    let bytes = match src {
        Expr::Binary { op: BinOp::Add, l, r, .. } if is_x(l) => r.as_int()?,
        Expr::Binary { op: BinOp::Sub, l, r, .. } if is_x(l) => -r.as_int()?,
        Expr::Cast { e, .. } => return step_of(e, x, vars, db),
        Expr::AddrOf(inner) => match &**inner {
            Expr::Load { base, offset, .. } if is_x(base) => *offset as i64,
            Expr::Index { base, index, ty } if is_x(base) && scalar_size(ty) == Some(1) => index.as_int()?,
            _ => return None,
        },
        _ => return None,
    };
    let elem = match strip_cv(&xt) {
        Type::Ptr(p) => crate::types::size_of(db, p)? as i64,
        Type::Int { .. } | Type::Long { .. } | Type::Char => 1,
        _ => return None,
    };
    if elem == 0 || bytes % elem != 0 {
        return None;
    }
    let d = bytes / elem;
    if d == 1 || d == -1 {
        Some(d)
    } else {
        None
    }
}

fn mentions_var(s: &Stmt, v: VarId) -> usize {
    let mut n = 0;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if matches!(e, Expr::Var(x) if *x == v) {
            n += 1;
        }
    });
    n
}

fn subst_var(s: &mut Stmt, v: VarId, with: &Expr) {
    let mut f = |e: &mut Expr| {
        if matches!(e, Expr::Var(x) if *x == v) {
            *e = with.clone();
        }
    };
    Stmt::rewrite_exprs(std::slice::from_mut(s), &mut f);
}

/// `t = *p; p = p + 1; use(t)` -> `use(*p++)`, `t = x + 1; x = t; use(t)` -> `use(++x)`.
pub fn form_incdec(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &[bool], db: Option<&mwdec_core::TypeDb>) {
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    form_incdec_with(body, vars, is_temp, db, &uses);
}

/// `form_incdec` with read counts computed over the whole function (for one block's list).
pub fn form_incdec_with(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &[bool], db: Option<&mwdec_core::TypeDb>, uses: &HashMap<VarId, usize>) {
    let temp = |v: VarId| is_temp.get(v).copied().unwrap_or(false);
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 2 < b.len() {
            // post-increment of a variable read by a temp
            if let (Stmt::Assign { dst: Expr::Var(t), src: e }, Stmt::Assign { dst: Expr::Var(x), src: step }) = (&b[i], &b[i + 1]) {
                let (t, x) = (*t, *x);
                if temp(t) && !temp(x) && uses.get(&t) == Some(&1) && e.uses_var(x) && !e.has_call() {
                    if let Some(d) = step_of(step, x, vars, db) {
                        if mentions_var(&b[i + 2], t) == 1 && mentions_var(&b[i + 2], x) == 0 {
                            let inc = Expr::IncDec { e: Box::new(Expr::Var(x)), delta: d, post: true };
                            let mut e2 = e.clone();
                            e2.rewrite(&mut |y| {
                                if matches!(y, Expr::Var(v) if *v == x) {
                                    *y = inc.clone();
                                }
                            });
                            let mut s = b[i + 2].clone();
                            subst_var(&mut s, t, &e2);
                            b[i + 2] = s;
                            b.drain(i..i + 2);
                            continue;
                        }
                    }
                }
            }
            // pre-increment of an lvalue whose new value is reused
            if let (Stmt::Assign { dst: Expr::Var(t), src: e }, Stmt::Assign { dst: l, src: Expr::Var(t2) }) = (&b[i], &b[i + 1]) {
                let t = *t;
                if *t2 == t && temp(t) && uses.get(&t) == Some(&2) && !matches!(l, Expr::Var(v) if temp(*v)) {
                    let d = match e {
                        Expr::Binary { op: BinOp::Add, l: a, r, .. } if a.as_ref() == l => r.as_int().filter(|k| *k == 1),
                        Expr::Binary { op: BinOp::Sub, l: a, r, .. } if a.as_ref() == l => r.as_int().filter(|k| *k == 1).map(|_| -1),
                        _ => None,
                    };
                    if let Some(d) = d {
                        if mentions_var(&b[i + 2], t) == 1 {
                            let inc = Expr::IncDec { e: Box::new(l.clone()), delta: d, post: false };
                            let mut s = b[i + 2].clone();
                            subst_var(&mut s, t, &inc);
                            b[i + 2] = s;
                            b.drain(i..i + 2);
                            continue;
                        }
                    }
                }
            }
            i += 1;
        }
    });
}

/// `v = c ? a : b; x = v;` (v not used afterwards) -> `x = c ? a : b;`
pub fn inline_ternary_results(body: &mut Vec<Stmt>) {
    let mut total: HashMap<VarId, usize> = HashMap::new();
    crate::inline::count_uses(body, &mut total);
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            if let Stmt::Assign { dst: Expr::Var(v), src: t @ Expr::Ternary { .. } } = &b[i] {
                let v = *v;
                let in_t = {
                    let mut n = 0;
                    t.walk(&mut |e| if matches!(e, Expr::Var(x) if *x == v) { n += 1 });
                    n
                };
                // a ternary reading the previous value of v is fine when that value was set by the
                // statement right before (nothing else reassigns v in between)
                let prev_def = (0..i).rev().find_map(|k| match &b[k] {
                    Stmt::Assign { dst: Expr::Var(x), src } if *x == v => Some(!src.uses_var(v)),
                    s if matches!(s, Stmt::Assign { .. } | Stmt::Expr(_)) => None,
                    _ => Some(false),
                }) == Some(true);
                let later: usize = b[i + 1..].iter().map(|s| mentions_var(s, v)).sum();
                let _ = &total;
                let only_here = later == 1 && !is_used_outside_list(&total, v, b);
                if only_here && mentions_var(&b[i + 1], v) == 1 && (in_t == 0 || prev_def) {
                    let t = t.clone();
                    subst_var(&mut b[i + 1], v, &t);
                    b.remove(i);
                    continue;
                }
            }
            i += 1;
        }
    });
}

/// Does `v` have uses outside statement list `b` (function-wide count vs this list's count)?
fn is_used_outside_list(total: &HashMap<VarId, usize>, v: VarId, b: &[Stmt]) -> bool {
    let mut here: HashMap<VarId, usize> = HashMap::new();
    crate::inline::count_uses(b, &mut here);
    total.get(&v).copied().unwrap_or(0) != here.get(&v).copied().unwrap_or(0)
}

/// `v = K; ... v = c ? v : e;` -> `v = c[K/v] ? K : e;` when v holds the constant K there, and
/// the first store becomes dead.
pub fn fold_ternary_constants(body: &mut Vec<Stmt>) {
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let (v, k) = match &b[i] {
                Stmt::Assign { dst: Expr::Var(v), src } if matches!(src, Expr::Int { .. } | Expr::Float { .. }) => (*v, src.clone()),
                _ => {
                    i += 1;
                    continue;
                }
            };
            // the next statement that mentions v must be the ternary re-assignment
            let mut j = i + 1;
            while j < b.len() && mentions_var(&b[j], v) == 0 {
                j += 1;
            }
            let ok = matches!(b.get(j), Some(Stmt::Assign { dst: Expr::Var(x), src: Expr::Ternary { .. } }) if *x == v)
                && b[i + 1..j].iter().all(|s| matches!(s, Stmt::Assign { dst: Expr::Var(_), .. }));
            if ok {
                if let Stmt::Assign { src, .. } = &mut b[j] {
                    src.rewrite(&mut |e| {
                        if matches!(e, Expr::Var(x) if *x == v) {
                            *e = k.clone();
                        }
                    });
                }
                b.remove(i);
                continue;
            }
            i += 1;
        }
    });
}

fn is_ctr_dec(s: &Stmt, ctr: VarId) -> bool {
    matches!(s, Stmt::Assign { dst: Expr::Var(x), src: Expr::Binary { op: BinOp::Sub, l, r, .. } }
        if *x == ctr && matches!(**l, Expr::Var(y) if y == ctr) && r.as_int() == Some(1))
}

fn ctr_loop_var(cond: &Expr) -> Option<VarId> {
    match cond {
        Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => match uncast(l) {
            Expr::Var(v) => Some(*v),
            _ => None,
        },
        _ => None,
    }
}

/// MWCC turns counted loops into CTR loops (`mtctr n ; ... bdnz`) behind an entry guard. With the
/// guard's condition being the loop test, `if (!c) exit; do { B; ctr--; } while (ctr);` is
/// `while (c) { B }`, and the trip-count computation disappears.
pub fn recover_ctr_loops(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
    recover_counted_ctr_loops(body, vars, is_temp);
    let vars: &[Var] = vars;
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    let is_ctr = |v: VarId| vars[v].name.starts_with("var_ctr");
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut j = 0;
        while j < b.len() {
            let ctr = match &b[j] {
                Stmt::DoWhile { body: lb, cond } => match ctr_loop_var(cond) {
                    Some(c) if is_ctr(c) && lb.last().map_or(false, |s| is_ctr_dec(s, c)) => Some(c),
                    _ => None,
                },
                _ => None,
            };
            let Some(ctr) = ctr else {
                j += 1;
                continue;
            };
            // the counter: one init, the decrement, the test
            if uses.get(&ctr).copied().unwrap_or(0) != 2 {
                j += 1;
                continue;
            }
            // entry guard right before: `if (!c) return R;` followed after the loop by `return R;`
            let guard = if j > 0 {
                match (&b[j - 1], b.get(j + 1)) {
                    (Stmt::If { cond, then, els }, after) if els.is_empty() && then.len() == 1 && matches!(then[0], Stmt::Return(_)) && after == Some(&then[0]) => Some(cond.clone()),
                    _ => None,
                }
            } else {
                None
            };
            let Some(g) = guard else {
                j += 1;
                continue;
            };
            let init = (0..j - 1).rev().find(|&k| matches!(&b[k], Stmt::Assign { dst: Expr::Var(x), .. } if *x == ctr));
            let Some(init) = init else {
                j += 1;
                continue;
            };
            if let Stmt::DoWhile { body: lb, .. } = &mut b[j] {
                lb.pop();
            }
            let lb = match b.remove(j) {
                Stmt::DoWhile { body, .. } => body,
                _ => unreachable!(),
            };
            b[j - 1] = Stmt::While { cond: g.negate(vars), body: lb };
            b.remove(init);
            j = j.saturating_sub(1);
        }
    });
}

/// `ctr = n; if (n > 0) { do { B; ctr--; } while (ctr); }` (the index the source counted with was
/// optimised away) -> `for (i = 0; i < n; i++) { B }` with a fresh counter; MWCC rebuilds the
/// same CTR loop from it. `n != 0` guards mean an unsigned count.
/// Remove the trailing `ctr--; if (ctr == 0) EXIT; else continue;` from a loop body's arms;
/// true if found (with EXIT equal to `exit`).
/// Strip the CTR latch; returns the path (then=0 / else=1 through trailing ifs) of the
/// statement list that held it.
fn strip_ctr_latch_path(b: &mut Vec<Stmt>, ctr: VarId, exit: &Stmt) -> Option<Vec<u8>> {
    let n = b.len();
    if n >= 2 && is_ctr_dec(&b[n - 2], ctr) {
        if let Stmt::If { cond, then, els } = &b[n - 1] {
            let tests_zero = matches!(cond, Expr::Binary { op: BinOp::Eq, l, r, .. } if r.as_int() == Some(0) && matches!(uncast(l), Expr::Var(v) if *v == ctr));
            let ok_els = els.is_empty() || els.as_slice() == [Stmt::Continue];
            if tests_zero && then.as_slice() == std::slice::from_ref(exit) && ok_els {
                b.truncate(n - 2);
                return Some(vec![]);
            }
        }
    }
    if let Some(Stmt::If { then, els, .. }) = b.last_mut() {
        if let Some(mut p) = strip_ctr_latch_path(then, ctr, exit) {
            p.insert(0, 0);
            return Some(p);
        }
        if let Some(mut p) = strip_ctr_latch_path(els, ctr, exit) {
            p.insert(0, 1);
            return Some(p);
        }
    }
    None
}

fn list_at<'a>(b: &'a mut Vec<Stmt>, path: &[u8]) -> Option<&'a mut Vec<Stmt>> {
    if path.is_empty() {
        return Some(b);
    }
    match b.last_mut() {
        Some(Stmt::If { then, els, .. }) => list_at(if path[0] == 0 { then } else { els }, &path[1..]),
        _ => None,
    }
}

/// Every other arm along the path leaves the loop (return/break), so the latch list runs on
/// each continuing iteration.
fn other_arms_exit(b: &[Stmt], path: &[u8]) -> bool {
    if path.is_empty() {
        return true;
    }
    match b.last() {
        Some(Stmt::If { then, els, .. }) => {
            let (go, other) = if path[0] == 0 { (then, els) } else { (els, then) };
            let exits = matches!(other.last(), Some(Stmt::Return(_) | Stmt::Break));
            exits && other_arms_exit(go, &path[1..])
        }
        _ => false,
    }
}

fn count_assigns(b: &[Stmt], v: VarId, n: &mut usize) {
    for s in b {
        match s {
            Stmt::Assign { dst: Expr::Var(x), .. } if *x == v => *n += 1,
            Stmt::If { then, els, .. } => {
                count_assigns(then, v, n);
                count_assigns(els, v, n);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => count_assigns(body, v, n),
            _ => {}
        }
    }
}

fn recover_counted_ctr_loops(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
    // search loops with an early exit: `ctr = n; if (n <= 0) R; while (true) { ...; ctr--; if (!ctr) R; }`
    {
        let mut uses: HashMap<VarId, usize> = HashMap::new();
        crate::inline::count_uses(body, &mut uses);
        let nvars = vars.len();
        let mut new_vars: Vec<Var> = vec![];
        Stmt::for_each_block_mut(body, &mut |b| {
            let mut j = 1;
            while j < b.len() {
                let ok = matches!(&b[j], Stmt::While { cond: Expr::Int { value: 1, .. }, .. });
                if !ok {
                    j += 1;
                    continue;
                }
                let Stmt::If { cond: g, then, els } = &b[j - 1] else {
                    j += 1;
                    continue;
                };
                if !els.is_empty() || then.len() != 1 || !matches!(then[0], Stmt::Return(_)) {
                    j += 1;
                    continue;
                }
                let exit = then[0].clone();
                let g = g.clone();
                // the counter init before the guard, guard on the same count
                let found = b[..j - 1].iter().rposition(|x| matches!(x, Stmt::Assign { dst: Expr::Var(c), .. } if vars[*c].name.starts_with("var_ctr")));
                let Some(ip) = found else {
                    j += 1;
                    continue;
                };
                let (ctr, n) = match &b[ip] {
                    Stmt::Assign { dst: Expr::Var(c), src } => (*c, src.clone()),
                    _ => unreachable!(),
                };
                let signed = match &g {
                    Expr::Binary { op: BinOp::Le, l, r, .. } if r.as_int() == Some(0) && uncast(l) == uncast(&n) => Some(true),
                    Expr::Binary { op: BinOp::Eq, l, r, .. } if r.as_int() == Some(0) && uncast(l) == uncast(&n) => Some(false),
                    _ => None,
                };
                let Some(signed) = signed else {
                    j += 1;
                    continue;
                };
                // after the loop the function returns the same way (or the loop is last)
                let after_ok = match b.get(j + 1) {
                    None => matches!(exit, Stmt::Return(None)),
                    Some(x) => *x == exit,
                };
                if !after_ok || uses.get(&ctr) != Some(&2) {
                    j += 1;
                    continue;
                }
                let Stmt::While { body: lb, .. } = &mut b[j] else { unreachable!() };
                let mut lb2 = lb.clone();
                let Some(path) = strip_ctr_latch_path(&mut lb2, ctr, &exit) else {
                    j += 1;
                    continue;
                };
                // adopt an existing index (`v = 0` before, `v = v + 1` closing each iteration)
                let mut adopted: Option<(VarId, usize)> = None;
                if other_arms_exit(&lb2, &path) {
                    let mut cand: Option<VarId> = None;
                    if let Some(l) = list_at(&mut lb2, &path) {
                        if let Some(Stmt::Assign { dst: Expr::Var(v), src: Expr::Binary { op: BinOp::Add, l: x, r: one, .. } }) = l.last() {
                            if matches!(**x, Expr::Var(y) if y == *v) && one.as_int() == Some(1) {
                                cand = Some(*v);
                            }
                        }
                    }
                    if let Some(v) = cand {
                        let init = b[..j - 1].iter().rposition(|s| matches!(s, Stmt::Assign { dst: Expr::Var(y), src } if *y == v && src.as_int() == Some(0)));
                        let mut n = 0;
                        count_assigns(&lb2, v, &mut n);
                        if let (Some(ini), 1) = (init, n) {
                            if ini != ip && b[ini + 1..j - 1].iter().all(|s| mentions_var(s, v) == 0) {
                                if let Some(l) = list_at(&mut lb2, &path) {
                                    l.pop();
                                }
                                adopted = Some((v, ini));
                            }
                        }
                    }
                }
                let ty = Type::Int { size: 4, signed };
                let i = match adopted {
                    Some((v, _)) => v,
                    None => {
                        let i = nvars + new_vars.len();
                        new_vars.push(Var { name: "i".into(), ty: ty.clone(), kind: VarKind::Local });
                        i
                    }
                };
                let cmp_n = if signed { n.clone() } else { Expr::cast(ty.clone(), n.clone()) };
                b[j] = Stmt::For {
                    init: vec![Stmt::Assign { dst: Expr::Var(i), src: Expr::Int { value: 0, ty: ty.clone() } }],
                    cond: Expr::cmp(BinOp::Lt, Expr::Var(i), cmp_n),
                    step: vec![Stmt::Expr(Expr::IncDec { e: Box::new(Expr::Var(i)), delta: 1, post: true })],
                    body: lb2,
                };
                b.remove(j - 1);
                let mut rm = vec![ip];
                if let Some((_, ini)) = adopted {
                    rm.push(ini);
                }
                rm.sort_unstable();
                for r in rm.iter().rev() {
                    b.remove(*r);
                }
                j = j.saturating_sub(1);
            }
        });
        for v in new_vars {
            vars.push(v);
            is_temp.push(false);
        }
    }
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    let mut todo: Vec<(VarId, Expr, bool)> = vec![];
    // find candidates first (immutable walk), then rewrite with fresh vars
    fn scan(b: &[Stmt], vars: &[Var], uses: &HashMap<VarId, usize>, out: &mut Vec<(VarId, Expr, bool)>) {
        for (k, s) in b.iter().enumerate() {
            if let Stmt::If { cond, then, els } = s {
                if els.is_empty() && then.len() == 1 {
                    if let Stmt::DoWhile { body: lb, cond: lc } = &then[0] {
                        if let Some(c) = ctr_loop_var(lc) {
                            if vars[c].name.starts_with("var_ctr") && lb.last().map_or(false, |x| is_ctr_dec(x, c)) && uses.get(&c) == Some(&2) {
                                // init `ctr = n` earlier in this list, guard on n
                                let init = b[..k].iter().rev().find_map(|x| match x {
                                    Stmt::Assign { dst: Expr::Var(y), src } if *y == c => Some(src.clone()),
                                    _ => None,
                                });
                                if let Some(n) = init {
                                    let signed = match cond {
                                        Expr::Binary { op: BinOp::Gt, l, r, .. } if r.as_int() == Some(0) && uncast(l) == uncast(&n) => Some(true),
                                        Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) && uncast(l) == uncast(&n) => Some(false),
                                        _ => None,
                                    };
                                    if let Some(sg) = signed {
                                        out.push((c, n, sg));
                                    }
                                }
                            }
                        }
                    }
                }
                scan(then, vars, uses, out);
                scan(els, vars, uses, out);
            }
            match s {
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => scan(body, vars, uses, out),
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        scan(&c.body, vars, uses, out);
                    }
                }
                _ => {}
            }
        }
    }
    scan(body, vars, &uses, &mut todo);
    for (c, n, signed) in todo {
        let i = vars.len();
        vars.push(Var { name: "i".into(), ty: Type::Int { size: 4, signed }, kind: VarKind::Local });
        is_temp.push(false);
        Stmt::for_each_block_mut(body, &mut |b| {
            let mut k = 0;
            while k < b.len() {
                let hit = matches!(&b[k], Stmt::If { then, .. } if then.len() == 1 && matches!(&then[0], Stmt::DoWhile { cond, .. } if ctr_loop_var(cond) == Some(c)));
                if hit {
                    let Stmt::If { mut then, .. } = b.remove(k) else { unreachable!() };
                    let Stmt::DoWhile { body: mut lb, .. } = then.remove(0) else { unreachable!() };
                    lb.pop();
                    let ty = Type::Int { size: 4, signed };
                    let cmp_n = if signed { n.clone() } else { Expr::cast(ty.clone(), n.clone()) };
                    let f = Stmt::For {
                        init: vec![Stmt::Assign { dst: Expr::Var(i), src: Expr::Int { value: 0, ty: ty.clone() } }],
                        cond: Expr::cmp(BinOp::Lt, Expr::Var(i), cmp_n),
                        step: vec![Stmt::Expr(Expr::IncDec { e: Box::new(Expr::Var(i)), delta: 1, post: true })],
                        body: lb,
                    };
                    b.insert(k, f);
                    // drop `ctr = n`
                    if let Some(p) = b[..k].iter().rposition(|x| matches!(x, Stmt::Assign { dst: Expr::Var(y), .. } if *y == c)) {
                        b.remove(p);
                    }
                    return;
                }
                k += 1;
            }
        });
    }
}

/// `if (p) delete p;` where the delete goes through a virtual destructor: MWCC's virtual
/// delete-expression already contains the null test (`cmplwi p,0; beq; ...; bctrl`), so a single
/// test in the target is a plain `delete p;` (an explicit test would compile to two).
pub fn fold_virtual_delete_checks(body: &mut Vec<Stmt>) {
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter_mut() {
            let Stmt::If { cond, then, els } = s else { continue };
            if !els.is_empty() {
                continue;
            }
            let real: Vec<&Stmt> = then.iter().filter(|x| !matches!(x, Stmt::Label(_))).collect();
            let [Stmt::Expr(call)] = real.as_slice() else { continue };
            let Expr::Call { callee: Callee::Virtual { this, sig: Some(sg), .. }, args, .. } = call else { continue };
            if !crate::sig::is_dtor(sg) || args.len() != 1 || args[0].as_int() != Some(1) {
                continue;
            }
            let tests_this = match uncast(cond) {
                Expr::Binary { op: BinOp::Ne, l, r, .. } => uncast(l) == uncast(this) && r.as_int() == Some(0),
                c => c == uncast(this),
            };
            if tests_this {
                let call = (*call).clone();
                *s = Stmt::Expr(call);
            }
        }
    });
}

/// `v = e; return v;` -> `return e;` (a register local assigned right before returning it).
pub fn fold_return_values(body: &mut Vec<Stmt>, vars: &[Var]) {
    Stmt::for_each_block_mut(body, &mut |b| {
        let n = b.len();
        if n < 2 {
            return;
        }
        let ok = match (&b[n - 2], &b[n - 1]) {
            (Stmt::Assign { dst: Expr::Var(v), src }, Stmt::Return(Some(Expr::Var(w)))) => {
                v == w && matches!(vars[*v].kind, VarKind::Local) && !src.uses_var(*v)
            }
            _ => false,
        };
        if ok {
            if let Stmt::Assign { src, .. } = b.remove(n - 2) {
                b[n - 2] = Stmt::Return(Some(src));
            }
        }
    });
}

/// MWCC's value-context `&&`/`||` (`li d,0 ; tests ; li d,1`, `gen_LOGICAL`), structured as
/// `v = 0; ...; if (a && b) v = 1;`, back to `v = a && b;`. A single comparison is never
/// materialised with branches (it would be branchless), so only logical chains are folded; the
/// zero may also arrive as a copy of another zeroed local (`li r5,0 ; mr r0,r5`). A folded
/// value read once, at the start of the next statement, is substituted there (`a && b && c`).
pub fn fold_logical_values(body: &mut Vec<Stmt>, vars: &[Var]) {
    let mut changed = false;
    Stmt::for_each_block_mut(body, &mut |b| changed |= fold_logical_list(b, vars));
    if !changed {
        return;
    }
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    Stmt::for_each_block_mut(body, &mut |b| subst_logical_list(b, vars, &uses));
}

fn is_logical(e: &Expr) -> bool {
    matches!(e, Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, .. })
}

fn mentions(s: &Stmt, v: VarId) -> bool {
    let mut found = false;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if matches!(e, Expr::Var(x) if *x == v) {
            found = true;
        }
    });
    found
}

fn fold_logical_list(b: &mut Vec<Stmt>, vars: &[Var]) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i < b.len() {
        let hit = match &b[i] {
            Stmt::If { cond, then, els } if els.is_empty() && is_logical(cond) => {
                let t: Vec<&Stmt> = then.iter().filter(|s| !matches!(s, Stmt::Label(_))).collect();
                match t.as_slice() {
                    [Stmt::Assign { dst: Expr::Var(v), src: Expr::Int { value: 1, .. } }]
                        if matches!(vars[*v].kind, VarKind::Local) && !cond.uses_var(*v) =>
                    {
                        Some(*v)
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        let Some(v) = hit else {
            i += 1;
            continue;
        };
        // the zeroing store: the last statement before that mentions v
        let Some(j) = (0..i).rev().find(|&j| mentions(&b[j], v)) else {
            i += 1;
            continue;
        };
        let zero = |e: &Expr, upto: usize, b: &Vec<Stmt>| -> bool {
            match e {
                Expr::Int { value: 0, .. } => true,
                Expr::Var(w) => match (0..upto).rev().find(|&k| mentions(&b[k], *w)) {
                    Some(k) => matches!(&b[k], Stmt::Assign { dst: Expr::Var(x), src: Expr::Int { value: 0, .. } } if x == w),
                    None => false,
                },
                _ => false,
            }
        };
        let ok = matches!(&b[j], Stmt::Assign { dst: Expr::Var(x), src } if *x == v && zero(src, j, b));
        if !ok {
            i += 1;
            continue;
        }
        let Stmt::If { cond, .. } = b[i].clone() else { unreachable!() };
        b[i] = Stmt::Assign { dst: Expr::Var(v), src: cond };
        b.remove(j);
        i -= 1;
        changed = true;
        // a zeroed local whose zero is now overwritten before any read (`r = 0; x = r;` became
        // `r = 0; x = a && b; r = x && c;`) loses the dead store
        let mut k = 0;
        while k < b.len() {
            let dead = match &b[k] {
                Stmt::Assign { dst: Expr::Var(w), src: Expr::Int { value: 0, .. } } if matches!(vars[*w].kind, VarKind::Local) => {
                    let w = *w;
                    match (k + 1..b.len()).find(|&n| mentions(&b[n], w)) {
                        Some(n) => matches!(&b[n], Stmt::Assign { dst: Expr::Var(x), src } if *x == w && !src.uses_var(w)) && n <= i,
                        None => false,
                    }
                }
                _ => false,
            };
            if dead {
                b.remove(k);
                if k < i {
                    i -= 1;
                }
            } else {
                k += 1;
            }
        }
        i += 1;
    }
    changed
}

/// The operand at the far left of a `&&`/`||` chain (evaluated first).
fn leftmost_logical_operand(e: &mut Expr) -> &mut Expr {
    match e {
        Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, l, .. } => leftmost_logical_operand(l),
        other => other,
    }
}

fn is_var_test(e: &Expr, v: VarId) -> bool {
    match e {
        Expr::Var(x) => *x == v,
        Expr::Cast { e, .. } => is_var_test(e, v),
        Expr::Binary { op: BinOp::Ne, l, r, .. } => r.as_int() == Some(0) && is_var_test(l, v),
        _ => false,
    }
}

fn subst_logical_list(b: &mut Vec<Stmt>, vars: &[Var], uses: &HashMap<VarId, usize>) {
    let mut i = 0;
    while i + 1 < b.len() {
        let v = match &b[i] {
            Stmt::Assign { dst: Expr::Var(v), src } if is_logical(src) && matches!(vars[*v].kind, VarKind::Local) && uses.get(v) == Some(&1) => *v,
            _ => {
                i += 1;
                continue;
            }
        };
        let Stmt::Assign { src: val, .. } = b[i].clone() else { unreachable!() };
        let target: Option<&mut Expr> = match &mut b[i + 1] {
            Stmt::Assign { dst: Expr::Var(_), src } if is_logical(src) => Some(leftmost_logical_operand(src)),
            Stmt::If { cond, .. } if is_logical(cond) => Some(leftmost_logical_operand(cond)),
            Stmt::Return(Some(e)) if is_logical(e) => Some(leftmost_logical_operand(e)),
            _ => None,
        };
        match target {
            Some(t) if is_var_test(t, v) => {
                *t = val;
                b.remove(i);
            }
            _ => i += 1,
        }
    }
}
