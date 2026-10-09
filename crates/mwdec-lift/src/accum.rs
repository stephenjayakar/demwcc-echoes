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
            let Expr::Var(a) = &**l else { continue };
            let (a, b) = (*a, *b);
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
            if chain.len() < 3 {
                continue;
            }
            for &v in &chain[1..] {
                rename.insert(v, a);
            }
            if let Some(t) = is_temp.get_mut(a) {
                *t = false;
            }
        }
        if rename.is_empty() {
            continue;
        }
        Stmt::rewrite_exprs(items, &mut |x| {
            if let Expr::Var(v) = x {
                if let Some(&to) = rename.get(v) {
                    *v = to;
                }
            }
        });
    }
}
