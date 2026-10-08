//! One source variable reused for successive values (`poll = x << 16; ...; poll = Si.poll;`):
//! the target keeps it in one callee-saved register, the lift splits it into one variable per
//! web (`temp_r31`, `temp_r31_2`). Merging such webs back (draft variant
//! [`crate::variants::MERGE_REGISTER_WEBS`]) keeps the compiler from propagating the first value
//! into its use.

use crate::ir::*;

/// Register a variable is named after (`temp_r31_2` -> `r31`).
fn reg_of(name: &str) -> Option<&str> {
    let rest = name.strip_prefix("temp_").or_else(|| name.strip_prefix("var_"))?;
    let reg = rest.split('_').next()?;
    let n: u32 = reg.strip_prefix('r')?.parse().ok()?;
    (14..=31).contains(&n).then_some(reg)
}

/// Merge top-level, sequential (non-overlapping) webs of the same callee-saved register.
pub fn merge_register_webs(body: &mut Vec<Stmt>, vars: &mut [Var]) -> bool {
    // first/last top-level statement index mentioning each var; vars mentioned in nested
    // statements are left alone
    let mut first: Vec<Option<usize>> = vec![None; vars.len()];
    let mut last: Vec<Option<usize>> = vec![None; vars.len()];
    let mut nested = vec![false; vars.len()];
    for (i, st) in body.iter().enumerate() {
        let top = matches!(st, Stmt::Assign { .. } | Stmt::Expr(_) | Stmt::Return(_));
        Stmt::walk_exprs(std::slice::from_ref(st), &mut |e| {
            if let Expr::Var(v) = e {
                if !top {
                    nested[*v] = true;
                }
                first[*v].get_or_insert(i);
                last[*v] = Some(i);
            }
        });
    }
    let mut changed = false;
    let mut renames: Vec<(VarId, VarId)> = vec![];
    for a in 0..vars.len() {
        let (Some(ra), Some(la)) = (reg_of(&vars[a].name), last[a]) else { continue };
        if nested[a] || vars[a].kind != VarKind::Local {
            continue;
        }
        for b in 0..vars.len() {
            if a == b || nested[b] || vars[b].kind != VarKind::Local || reg_of(&vars[b].name) != Some(ra) {
                continue;
            }
            let Some(fb) = first[b] else { continue };
            // b starts with its own assignment after a's last use
            let starts_with_def = matches!(&body[fb], Stmt::Assign { dst: Expr::Var(v), src } if *v == b && !src.uses_var(b));
            if fb > la && starts_with_def && !renames.iter().any(|(x, _)| *x == b) {
                renames.push((b, a));
                changed = true;
                break;
            }
        }
    }
    if changed {
        Stmt::rewrite_exprs(body, &mut |e| {
            if let Expr::Var(v) = e {
                let mut t = *v;
                let mut fuel = crate::fuel::Fuel::new("samereg.renames", renames.len() + 1);
                while let Some((_, to)) = renames.iter().find(|(f, _)| *f == t) {
                    if !fuel.burn() {
                        break;
                    }
                    t = *to;
                }
                *v = t;
            }
        });
    }
    changed
}
