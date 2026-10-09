//! Re-rolling MWCC's unrolled counted loops. The front end unrolls `for (i = 0; i < n; i++)`
//! with a straight-line body eight times ("shape A", `IroUnrollLoop.c`):
//!
//! ```text
//! if (n > 0) {
//!     t = n - 8;
//!     if (n > 8) { ctr = (t + 7) >> 3; if (t > 0) do { i += 8; BODY x8 } while (--ctr); }
//!     ctr2 = n - i;
//!     if (i < n) do { BODY } while (--ctr2);
//! }
//! ```
//!
//! The source had one loop; MWCC makes the copies again, so the draft keeps only the remainder
//! loop's body under the original loop header.

use crate::ir::*;

fn strip(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip(e),
        e => e,
    }
}

fn is_int(e: &Expr, k: i64) -> bool {
    strip(e).as_int() == Some(k)
}

/// `v = v - 1` / `v = v + -1`
fn is_dec(s: &Stmt, v: VarId) -> bool {
    match s {
        Stmt::Assign { dst: Expr::Var(x), src } if *x == v => match strip(src) {
            Expr::Binary { op: BinOp::Sub, l, r, .. } => matches!(strip(l), Expr::Var(y) if *y == v) && is_int(r, 1),
            Expr::Binary { op: BinOp::Add, l, r, .. } => matches!(strip(l), Expr::Var(y) if *y == v) && is_int(r, -1),
            _ => false,
        },
        _ => false,
    }
}

/// `do { body; c = c - 1; } while (c != 0)` -> (c, body)
fn ctr_loop(s: &Stmt) -> Option<(VarId, Vec<Stmt>)> {
    let Stmt::DoWhile { body, cond } = s else { return None };
    let Expr::Binary { op: BinOp::Ne, l, r, .. } = strip(cond) else { return None };
    let Expr::Var(c) = strip(l) else { return None };
    if !is_int(r, 0) {
        return None;
    }
    let real: Vec<Stmt> = body.iter().filter(|s| !matches!(s, Stmt::Label(_))).cloned().collect();
    let (last, rest) = real.split_last()?;
    if !is_dec(last, *c) || rest.iter().any(|s| crate::idioms::stmt_mentions(s, *c)) {
        return None;
    }
    Some((*c, rest.to_vec()))
}

fn real(b: &[Stmt]) -> Vec<&Stmt> {
    b.iter().filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))).collect()
}

/// `a op b` with operands compared after stripping casts.
fn cmp_parts(e: &Expr) -> Option<(BinOp, &Expr, &Expr)> {
    match strip(e) {
        Expr::Binary { op, l, r, .. } if op.is_cmp() => Some((*op, strip(l), strip(r))),
        _ => None,
    }
}

fn straight(b: &[Stmt]) -> bool {
    b.iter().all(|s| matches!(s, Stmt::Assign { .. } | Stmt::Expr(_)))
}

/// Shape A at `b[k]`: returns (index var, bound, single body).
fn shape_a(b: &[Stmt], k: usize) -> Option<(VarId, Expr, Vec<Stmt>)> {
    let Stmt::If { cond, then, els } = &b[k] else { return None };
    if !els.is_empty() {
        return None;
    }
    let (op, n, zero) = cmp_parts(cond)?;
    if op != BinOp::Gt || !is_int(zero, 0) {
        return None;
    }
    let t = real(then);
    // [temps...] if (n > 8) {...} ctr2 = n - i; if (i < n) do {...}
    let m = t.len();
    if m >= 2 {
        if let Some(r) = shape_a_formed(&t, n) {
            return Some(r);
        }
    }
    if m < 3 {
        return None;
    }
    let Stmt::If { cond: rc, then: rthen, els: rels } = t[m - 1] else { return None };
    let Stmt::Assign { dst: Expr::Var(c2), src: c2src } = t[m - 2] else { return None };
    // the remainder may walk a pointer set up from the index (`p = base + i`)
    let mut q = m - 3;
    let mut ptrs: Vec<(VarId, Expr)> = vec![];
    while let Stmt::Assign { dst: Expr::Var(p), src } = t[q] {
        ptrs.push((*p, src.clone()));
        if q == 0 {
            return None;
        }
        q -= 1;
    }
    let Stmt::If { cond: c8, then: then8, els: els8 } = t[q] else { return None };
    if !rels.is_empty() || !els8.is_empty() {
        return None;
    }
    // remainder: if (i < n) do { BODY; c2--; } while (c2)
    let (rop, ri, rn) = cmp_parts(rc)?;
    let Expr::Var(i) = ri else { return None };
    if rop != BinOp::Lt || rn != n {
        return None;
    }
    let rl = real(rthen);
    let [rloop] = rl.as_slice() else { return None };
    let (rc2, mut body1) = ctr_loop(rloop)?;
    // (an empty body is a loop over elements with trivial inline destructors, `clear()`)
    if rc2 != *c2 || !straight(&body1) {
        return None;
    }
    // pointer walks of the remainder are the index again
    subst_pointer_walks(&mut body1, &ptrs, *i)?;
    // ctr2 = n - i
    match strip(c2src) {
        Expr::Binary { op: BinOp::Sub, l, r, .. } if strip(l) == n && matches!(strip(r), Expr::Var(x) if x == i) => {}
        _ => return None,
    }
    // unrolled part: if (n > 8) { ctr = ...; if (t > 0) do { i += 8; BODYx8; ctr--; } while (ctr); }
    let (op8, n8, eight) = cmp_parts(c8)?;
    if op8 != BinOp::Gt || n8 != n || !is_int(eight, 8) {
        return None;
    }
    let t8 = real(then8);
    let [Stmt::Assign { dst: Expr::Var(_), .. }, Stmt::If { then: inner, els: inner_els, .. }] = t8.as_slice() else { return None };
    if !inner_els.is_empty() {
        return None;
    }
    let il = real(inner);
    let [uloop] = il.as_slice() else { return None };
    let (_, body8) = ctr_loop(uloop)?;
    let steps8 = body8
        .iter()
        .any(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src } if x == i && matches!(strip(src), Expr::Binary { op: BinOp::Add, l, r, .. } if matches!(strip(l), Expr::Var(y) if y == i) && is_int(r, 8))));
    if !steps8 {
        return None;
    }
    // the leading statements are temps for `n - 8`
    if t[..q].iter().any(|s| !matches!(s, Stmt::Assign { dst: Expr::Var(_), .. })) {
        return None;
    }
    Some((*i, n.clone(), body1))
}

/// Shape A whose remainder loop is already a `for (; i < n; i++) BODY` (the counted-loop
/// recovery ran first): `[temps...] if (n > 8) {...} for (; i < n; i++) BODY`.
fn shape_a_formed(t: &[&Stmt], n: &Expr) -> Option<(VarId, Expr, Vec<Stmt>)> {
    let m = t.len();
    let Stmt::For { init, cond, step, body } = t[m - 1] else { return None };
    if !init.is_empty() {
        return None;
    }
    let (rop, ri, rn) = cmp_parts(cond)?;
    let Expr::Var(i) = ri else { return None };
    if rop != BinOp::Lt || rn != n || !straight(body) {
        return None;
    }
    let steps1 = match step.as_slice() {
        [Stmt::Expr(Expr::IncDec { e, delta: 1, .. })] => matches!(&**e, Expr::Var(x) if x == i),
        [s] => matches!(s, Stmt::Assign { dst: Expr::Var(x), src } if x == i && matches!(strip(src), Expr::Binary { op: BinOp::Add, l, r, .. } if matches!(strip(l), Expr::Var(y) if y == i) && is_int(r, 1))),
        _ => false,
    };
    if !steps1 {
        return None;
    }
    let Stmt::If { cond: c8, then: then8, els: els8 } = t[m - 2] else { return None };
    if !els8.is_empty() {
        return None;
    }
    let (op8, n8, eight) = cmp_parts(c8)?;
    if op8 != BinOp::Gt || n8 != n || !is_int(eight, 8) {
        return None;
    }
    let t8 = real(then8);
    let [Stmt::Assign { dst: Expr::Var(_), .. }, Stmt::If { then: inner, els: inner_els, .. }] = t8.as_slice() else { return None };
    if !inner_els.is_empty() {
        return None;
    }
    let il = real(inner);
    let [uloop] = il.as_slice() else { return None };
    let (_, body8) = ctr_loop(uloop)?;
    let steps8 = body8
        .iter()
        .any(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src } if x == i && matches!(strip(src), Expr::Binary { op: BinOp::Add, l, r, .. } if matches!(strip(l), Expr::Var(y) if y == i) && is_int(r, 8))));
    if !steps8 || t[..m - 2].iter().any(|s| !matches!(s, Stmt::Assign { dst: Expr::Var(_), .. })) {
        return None;
    }
    Some((*i, n.clone(), body.clone()))
}

/// `p + k` written as an add or as `&p->field` (`&*(p + k)`).
fn step_of(src: &Expr, p: VarId) -> Option<i64> {
    match strip(src) {
        Expr::Binary { op: BinOp::Add, l, r, .. } if matches!(strip(l), Expr::Var(y) if *y == p) => r.as_int(),
        Expr::AddrOf(x) => match &**x {
            Expr::Load { base, offset, .. } if matches!(strip(base), Expr::Var(y) if *y == p) => Some(*offset as i64),
            _ => None,
        },
        _ => None,
    }
}

/// Bytes per unit of `i` in an index expression (`i`, `i * c`, `i << s`).
fn index_coef(e: &Expr, i: VarId) -> Option<i64> {
    match strip(e) {
        Expr::Var(x) if *x == i => Some(1),
        Expr::Binary { op: BinOp::Mul, l, r, .. } if matches!(strip(l), Expr::Var(x) if *x == i) => r.as_int(),
        Expr::Binary { op: BinOp::Shl, l, r, .. } if matches!(strip(l), Expr::Var(x) if *x == i) => r.as_int().filter(|s| (0..16).contains(s)).map(|s| 1 << s),
        _ => None,
    }
}

/// The remainder loop of an unrolled loop may walk pointers set up from the index (`p = base +
/// i * k; ... *p = x; p = p + k;`): they are the index again (`*(base + i * k) = x`).
fn subst_pointer_walks(body1: &mut Vec<Stmt>, ptrs: &[(VarId, Expr)], i: VarId) -> Option<()> {
    for (p, init) in ptrs {
        // bytes the pointer moves per unit of i
        let coef = match strip(init) {
            Expr::Binary { op: BinOp::Add, r, .. } => index_coef(r, i)?,
            Expr::AddrOf(x) => match &**x {
                Expr::Index { index, ty, .. } => index_coef(index, i)? * scalar_size(ty)? as i64,
                _ => return None,
            },
            _ => return None,
        };
        let steps: Vec<usize> = (0..body1.len())
            .filter(|&k| matches!(&body1[k], Stmt::Assign { dst: Expr::Var(x), src } if x == p && step_of(src, *p).is_some()))
            .collect();
        let [st] = steps.as_slice() else { return None };
        let Stmt::Assign { src, .. } = &body1[*st] else { unreachable!() };
        if step_of(src, *p)? != coef {
            return None;
        }
        // the step is the iteration's last use of p
        if body1[*st + 1..].iter().any(|s| crate::idioms::stmt_mentions(s, *p)) {
            return None;
        }
        body1.remove(*st);
        let addr = init.clone();
        Stmt::rewrite_exprs(body1, &mut |x| {
            if matches!(x, Expr::Var(y) if y == p) {
                *x = addr.clone();
            }
        });
    }
    Some(())
}

/// Shape A with constant bounds (`for (i = a; i < N; i++)`, N - a > 8): the guards fold away,
/// leaving `ctr = K; do { BODY x8; i += 8; } while (--ctr); ctr2 = N - i; if (i < N) do { BODY }
/// while (--ctr2);`. Returns (counter init, index init, last statement, i, a, N, body).
fn shape_a_const(b: &[Stmt], k: usize) -> Option<(usize, usize, usize, VarId, i64, i64, Vec<Stmt>)> {
    let (c, body8) = ctr_loop(b.get(k)?)?;
    // the main loop's counter starts at a constant
    let ic = (0..k).rev().find(|&j| crate::idioms::stmt_mentions(&b[j], c))?;
    let kk = match &b[ic] {
        Stmt::Assign { dst: Expr::Var(x), src } if *x == c => strip(src).as_int()?,
        _ => return None,
    };
    // pointer setups, ctr2 = N - i, if (i < N) do { BODY } while (ctr2)
    let mut q = k + 1;
    let mut ptrs: Vec<(VarId, Expr)> = vec![];
    while let Some(Stmt::Assign { dst: Expr::Var(p), src }) = b.get(q) {
        if matches!(strip(src), Expr::Binary { op: BinOp::Sub, l, .. } if strip(l).as_int().is_some()) {
            break;
        }
        ptrs.push((*p, src.clone()));
        q += 1;
    }
    let Stmt::Assign { dst: Expr::Var(c2), src: c2src } = b.get(q)? else { return None };
    let Expr::Binary { op: BinOp::Sub, l: nn, r: ii, .. } = strip(c2src) else { return None };
    let n = strip(nn).as_int()?;
    let Expr::Var(i) = strip(ii) else { return None };
    let Stmt::If { cond, then, els } = b.get(q + 1)? else { return None };
    if !els.is_empty() {
        return None;
    }
    let (op, ci, cn) = cmp_parts(cond)?;
    if op != BinOp::Lt || !matches!(ci, Expr::Var(x) if x == i) || cn.as_int() != Some(n) {
        return None;
    }
    let rl = real(then);
    let [rloop] = rl.as_slice() else { return None };
    let (rc2, mut body1) = ctr_loop(rloop)?;
    if rc2 != *c2 || !straight(&body1) || body1.is_empty() {
        return None;
    }
    // the main loop steps i by 8
    let steps8 = body8.iter().any(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src } if x == i && matches!(strip(src), Expr::Binary { op: BinOp::Add, l, r, .. } if matches!(strip(l), Expr::Var(y) if y == i) && is_int(r, 8))));
    if !steps8 {
        return None;
    }
    // i starts at a constant a with N - a = 8 K + r, 0 < r < 8
    let ii0 = (0..k).rev().find(|&j| j != ic && crate::idioms::stmt_mentions(&b[j], *i))?;
    let a = match &b[ii0] {
        Stmt::Assign { dst: Expr::Var(x), src } if x == i => strip(src).as_int()?,
        _ => return None,
    };
    let r = n - a - 8 * kk;
    if !(1..8).contains(&r) {
        return None;
    }
    subst_pointer_walks(&mut body1, &ptrs, *i)?;
    // the remainder steps i itself (it is read after the loop): the for's step does
    let inc: Vec<usize> = (0..body1.len()).filter(|&j| matches!(&body1[j], Stmt::Assign { dst: Expr::Var(x), src } if x == i && matches!(strip(src), Expr::Binary { op: BinOp::Add, l, r, .. } if matches!(strip(l), Expr::Var(y) if y == i) && is_int(r, 1)))).collect();
    match inc.as_slice() {
        [j] if !body1[j + 1..].iter().any(|s| crate::idioms::stmt_mentions(s, *i)) => {
            body1.remove(*j);
        }
        [] => {}
        _ => return None,
    }
    if body1.iter().any(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if x == i)) {
        return None;
    }
    Some((ic, ii0, q + 1, *i, a, n, body1))
}

/// Constant-bound shape A loops back to `for (i = a; i < N; i++) BODY`.
pub fn reroll_const(body: &mut Vec<Stmt>) {
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut k = 0;
        while k < b.len() {
            if let Some((ic, ii0, last, i, a, n, body1)) = shape_a_const(b, k) {
                let ty = t_s32();
                let f = Stmt::For {
                    init: vec![Stmt::Assign { dst: Expr::Var(i), src: Expr::Int { value: a, ty: ty.clone() } }],
                    cond: Expr::cmp(BinOp::Lt, Expr::Var(i), Expr::Int { value: n, ty: ty.clone() }),
                    step: vec![Stmt::Assign { dst: Expr::Var(i), src: Expr::bin(BinOp::Add, Expr::Var(i), Expr::int(1), ty) }],
                    body: body1,
                };
                // statements between the inits and the main loop stay before the for
                let first = ic.min(ii0);
                // (the main loop's own pointers, now unused, go too)
                let mentioned_after = |x: VarId, b: &Vec<Stmt>| b[last + 1..].iter().any(|s| crate::idioms::stmt_mentions(s, x)) || crate::idioms::stmt_mentions(&f, x);
                let keep: Vec<Stmt> = (first..k)
                    .filter(|&j| j != ic && j != ii0)
                    .filter(|&j| !matches!(&b[j], Stmt::Assign { dst: Expr::Var(x), src } if !src.has_call() && !mentioned_after(*x, b) && (first..k).all(|o| o == j || !crate::idioms::stmt_mentions(&b[o], *x))))
                    .map(|j| b[j].clone())
                    .collect();
                // variables only the dropped main loop read (its pointer walks)
                let mut walked: Vec<VarId> = vec![];
                if let Stmt::DoWhile { body: lb, .. } = &b[k] {
                    for s in lb {
                        if let Stmt::Assign { dst: Expr::Var(x), .. } = s {
                            if *x != i && !walked.contains(x) {
                                walked.push(*x);
                            }
                        }
                    }
                }
                let mut repl = keep;
                repl.push(f);
                let nrepl = repl.len();
                b.splice(first..=last, repl);
                k = first + nrepl;
                for x in walked {
                    let users: Vec<usize> = (0..b.len()).filter(|&j| crate::idioms::stmt_mentions(&b[j], x)).collect();
                    if let [j] = users.as_slice() {
                        if matches!(&b[*j], Stmt::Assign { dst: Expr::Var(y), src } if *y == x && !src.has_call()) {
                            b.remove(*j);
                            if *j < k {
                                k -= 1;
                            }
                        }
                    }
                }
                continue;
            }
            k += 1;
        }
    });
}

/// Replace unrolled loops by the loop the source had.
pub fn reroll(body: &mut Vec<Stmt>) {
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut k = 0;
        while k < b.len() {
            if let Some((i, n, body1)) = shape_a(b, k) {
                // `i = 0` right before becomes the init (its only role)
                let mut init = vec![];
                if let Some(j) = (0..k).rev().find(|&j| crate::idioms::stmt_mentions(&b[j], i)) {
                    if matches!(&b[j], Stmt::Assign { dst: Expr::Var(x), src } if *x == i && is_int(src, 0)) {
                        init.push(b.remove(j));
                        k -= 1;
                    }
                }
                b[k] = Stmt::For {
                    init,
                    cond: Expr::cmp(BinOp::Lt, Expr::Var(i), n),
                    step: vec![Stmt::Assign { dst: Expr::Var(i), src: Expr::bin(BinOp::Add, Expr::Var(i), Expr::int(1), t_s32()) }],
                    body: body1,
                };
            }
            k += 1;
        }
    });
}
