//! A value built step by step in one callee-saved register (`addi r31, r5, 0x84; add r31, r31,
//! r4; add r31, r31, r0`): every intermediate result lives in the variable's own register, so the
//! source updated one variable statement by statement (`s = a + 0x84; s += b; s += c;`). Folded
//! into one expression the compiler would build the sum in volatile registers (and reassociate
//! it), assigning only the result. Before temporaries are folded, such chains (two updates or
//! more) become one named variable updated in place.

use crate::ir::*;
use std::collections::HashMap;

/// Callee-saved register a temporary is named after (`temp_r31` -> 31).
fn callee_saved(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("temp_r")?;
    let n: u32 = rest.split('_').next()?.parse().ok()?;
    (14..=31).contains(&n).then_some(n)
}

fn is_update(op: BinOp) -> bool {
    matches!(op, BinOp::Add | BinOp::Sub | BinOp::Or | BinOp::And | BinOp::Xor)
}

/// Merge each block's update chains of one callee-saved register into a named variable.
pub fn name_accumulators(lists: &mut [(Vec<Stmt>, u8)], vars: &[Var], is_temp: &mut [bool]) {
    name_accumulators_with(lists, vars, is_temp, 3, false);
}

/// `name_accumulators` for chains of at least `min_len` values; with `either` the previous value
/// may be either operand (`s = x & ~s`), not only the left one. Returns whether anything merged.
pub fn name_accumulators_with(lists: &mut [(Vec<Stmt>, u8)], vars: &[Var], is_temp: &mut [bool], min_len: usize, either: bool) -> bool {
    let mut merged = false;
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    for (items, _) in lists.iter() {
        crate::inline::count_uses(items, &mut uses);
    }
    for (items, _) in lists.iter_mut() {
        // next[a] = b when `b = a op x` (a's only use) in the same register
        let mut next: HashMap<VarId, VarId> = HashMap::new();
        let mut has_prev: HashMap<VarId, bool> = HashMap::new();
        for s in items.iter() {
            let Stmt::Assign { dst: Expr::Var(b), src: Expr::Binary { op, l, r, .. } } = s else { continue };
            let strip_not = |e: &Expr| -> Option<VarId> {
                match e {
                    Expr::Var(v) => Some(*v),
                    Expr::Unary { op: UnOp::BitNot, e, .. } => match &**e {
                        Expr::Var(v) if either => Some(*v),
                        _ => None,
                    },
                    _ => None,
                }
            };
            let b = *b;
            let same_reg = |v: VarId| callee_saved(&vars[v].name).is_some() && callee_saved(&vars[v].name) == callee_saved(&vars[b].name);
            let (a, r) = match (strip_not(l), strip_not(r)) {
                (Some(a), _) if matches!(&**l, Expr::Var(_)) && same_reg(a) => (a, &**r),
                (_, Some(a)) if either && !matches!(op, BinOp::Sub) && same_reg(a) => (a, &**l),
                _ => continue,
            };
            let (Some(ra), Some(rb)) = (callee_saved(&vars[a].name), callee_saved(&vars[b].name)) else { continue };
            let temp = |v: VarId| is_temp.get(v).copied().unwrap_or(false);
            if ra != rb || a == b || !is_update(*op) || !temp(a) || !temp(b) || uses.get(&a).copied() != Some(1) || r.uses_var(a) {
                continue;
            }
            next.insert(a, b);
            has_prev.insert(b, true);
        }
        let mut rename: HashMap<VarId, VarId> = HashMap::new();
        for (&a, _) in next.iter() {
            if has_prev.contains_key(&a) {
                continue;
            }
            // chain head: a -> b -> c ...
            let mut chain = vec![a];
            let mut cur = a;
            while let Some(&n) = next.get(&cur) {
                chain.push(n);
                cur = n;
                if chain.len() > 64 {
                    break;
                }
            }
            if chain.len() < min_len {
                continue;
            }
            // (variant mode: a head computed from a parameter used nowhere else updates that
            // parameter in place, `poll = poll >> 24 & 0xf0; poll = x & ~poll;`)
            // (only a parameter kept across a call before the head: then it lives in that
            // callee-saved register; with no call before, it's still in its argument register)
            let param = if either {
                let hi = items.iter().position(|s| matches!(s, Stmt::Assign { dst: Expr::Var(d), .. } if *d == a));
                hi.and_then(|hi| {
                    let Stmt::Assign { src, .. } = &items[hi] else { return None };
                    let mut ps = vec![];
                    src.walk(&mut |e| {
                        if let Expr::Var(v) = e {
                            if matches!(vars[*v].kind, VarKind::Param { .. }) {
                                ps.push(*v);
                            }
                        }
                    });
                    let p = match ps.as_slice() {
                        [p, rest @ ..] if rest.iter().all(|q| q == p) => *p,
                        _ => return None,
                    };
                    let call_before = items[..hi].iter().any(|s| {
                        let mut c = false;
                        Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| c |= matches!(e, Expr::Call { .. }));
                        c
                    });
                    let used_after = items[hi + 1..].iter().any(|s| {
                        let mut u = false;
                        Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| u |= matches!(e, Expr::Var(v) if *v == p));
                        u
                    });
                    (call_before && !used_after).then_some(p)
                })
            } else {
                None
            };
            let to = param.unwrap_or(a);
            for &v in &chain[1..] {
                rename.insert(v, to);
            }
            if let Some(p) = param {
                rename.insert(a, p);
            } else if let Some(t) = is_temp.get_mut(a) {
                *t = false;
            }
        }
        if rename.is_empty() {
            continue;
        }
        merged = true;
        Stmt::rewrite_exprs(items, &mut |x| {
            if let Expr::Var(v) = x {
                if let Some(&to) = rename.get(v) {
                    *v = to;
                }
            }
        });
    }
    merged
}
