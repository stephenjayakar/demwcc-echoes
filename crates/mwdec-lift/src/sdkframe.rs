//! GC/1.2.5n frame slots of scalar locals. This compiler gives a stack slot (below the
//! address-taken locals, above the parameter slots) to every declared local that ends up in a
//! volatile register (and to unused ones); locals in callee-saved registers and locals copy
//! propagation removes take none. A draft local the source didn't have (a value the target keeps
//! in a volatile register between its computation and its single use) grows the frame; folding
//! it back into its use restores the source's layout.

use crate::ir::*;
use std::collections::HashMap;

/// A volatile-register local of the draft (named after its target register: `temp_r0`,
/// `var_r5`, `temp_f2`, ...).
pub fn volatile_name(name: &str) -> bool {
    let reg = name.rsplit('_').next().unwrap_or("");
    let (class, num) = reg.split_at(reg.len().min(1));
    matches!(class, "r" | "f") && num.parse::<u32>().is_ok_and(|n| n < 14) && (name.starts_with("temp_") || name.starts_with("var_"))
}

/// Volatile-register locals the draft declares (each costs this compiler a frame slot).
pub fn volatile_locals(body: &[Stmt], vars: &[Var]) -> usize {
    let mut seen = std::collections::BTreeSet::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            if vars[*v].kind == VarKind::Local && volatile_name(&vars[*v].name) {
                seen.insert(*v);
            }
        }
    });
    seen.len()
}

/// Memory objects a value reads: named globals, or `None` if it reads through pointers / calls.
fn reads(e: &Expr) -> Option<Vec<String>> {
    let mut out = vec![];
    let mut ok = true;
    e.walk(&mut |x| match x {
        Expr::Global { symbol, .. } => out.push(symbol.clone()),
        Expr::Load { .. } | Expr::Index { .. } | Expr::BitField { .. } | Expr::New { .. } | Expr::IncDec { .. } => ok = false,
        Expr::Member { .. } => {}
        Expr::Call { .. } if !x.is_pure_call() => ok = false,
        _ => {}
    });
    ok.then_some(out)
}

fn global_root(e: &Expr) -> Option<&str> {
    match e {
        Expr::Global { symbol, .. } => Some(symbol),
        Expr::Member { base, .. } => global_root(base),
        _ => None,
    }
}

/// Fold up to `limit` single-assignment, single-use volatile-register locals into their use when
/// the value reads only named globals and the statements in between only store to other named
/// globals and don't reassign the value's operands. Returns how many were folded.
pub fn fold_volatile_locals(body: &mut Vec<Stmt>, vars: &[Var], limit: usize) -> usize {
    let mut nassign: HashMap<VarId, usize> = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| {
        for st in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(v), .. } = st {
                *nassign.entry(*v).or_default() += 1;
            }
        }
    });
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    let mut done = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() && done < limit {
            let cand = match &b[i] {
                Stmt::Assign { dst: Expr::Var(v), src }
                    if vars[*v].kind == VarKind::Local && volatile_name(&vars[*v].name) && nassign.get(v) == Some(&1) && uses.get(v) == Some(&1) && !src.uses_var(*v) =>
                {
                    reads(src).map(|r| (*v, src.clone(), r))
                }
                _ => None,
            };
            let Some((v, src, rd)) = cand else {
                i += 1;
                continue;
            };
            let mut ops: Vec<VarId> = vec![];
            src.walk(&mut |x| {
                if let Expr::Var(w) = x {
                    ops.push(*w);
                }
            });
            // the statement using v, after only stores to other named globals
            let mut j = i + 1;
            let mut ok = false;
            while j < b.len() {
                let mentions = {
                    let mut m = false;
                    Stmt::walk_exprs(std::slice::from_ref(&b[j]), &mut |e| m |= matches!(e, Expr::Var(x) if *x == v));
                    m
                };
                if mentions {
                    ok = matches!(&b[j], Stmt::If { .. } | Stmt::Assign { .. } | Stmt::Expr(_) | Stmt::Return(_));
                    break;
                }
                match &b[j] {
                    Stmt::Assign { dst, src: s2 } if !dst.has_call() && !s2.has_call() => {
                        let other = match global_root(dst) {
                            Some(s) => !rd.iter().any(|r| r == s),
                            None => matches!(dst, Expr::Var(w) if !ops.contains(w) && *w != v),
                        };
                        if !other {
                            break;
                        }
                    }
                    _ => break,
                }
                j += 1;
            }
            if !ok {
                i += 1;
                continue;
            }
            let mut sub = |e: &mut Expr| {
                if matches!(e, Expr::Var(x) if *x == v) {
                    *e = src.clone();
                }
            };
            match &mut b[j] {
                Stmt::If { cond, .. } => cond.rewrite(&mut sub),
                st => Stmt::rewrite_exprs(std::slice::from_mut(st), &mut sub),
            }
            b.remove(i);
            done += 1;
        }
    });
    done
}
