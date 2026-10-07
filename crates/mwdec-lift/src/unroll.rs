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
    if m < 3 {
        return None;
    }
    let Stmt::If { cond: rc, then: rthen, els: rels } = t[m - 1] else { return None };
    let Stmt::Assign { dst: Expr::Var(c2), src: c2src } = t[m - 2] else { return None };
    let Stmt::If { cond: c8, then: then8, els: els8 } = t[m - 3] else { return None };
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
    let (rc2, body1) = ctr_loop(rloop)?;
    if rc2 != *c2 || !straight(&body1) || body1.is_empty() {
        return None;
    }
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
    if t[..m - 3].iter().any(|s| !matches!(s, Stmt::Assign { dst: Expr::Var(_), .. })) {
        return None;
    }
    Some((*i, n.clone(), body1))
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
