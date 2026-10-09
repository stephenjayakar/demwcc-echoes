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

/// Register a variable is named after, any GPR (`temp_r0` -> `r0`).
fn any_reg_of(name: &str) -> Option<&str> {
    let rest = name.strip_prefix("temp_").or_else(|| name.strip_prefix("var_"))?;
    let reg = rest.split('_').next()?;
    let n: u32 = reg.strip_prefix('r')?.parse().ok()?;
    (n <= 31).then_some(reg)
}

/// `t = x; v = t & m; ...; return t;` with `t` and `v` in one register: the register was updated
/// in place after a copy of its old value was taken (`mr rB, rA` right after the load), so the
/// source updated one variable and kept the old value in another (`cpr = x; prev = cpr; cpr &=
/// m; ... return prev;`). Draft variant [`crate::variants::UPDATE_AFTER_COPY`].
pub fn update_after_copy(body: &mut Vec<Stmt>, vars: &mut Vec<Var>) -> bool {
    let n = body.len();
    for i in 0..n {
        let Stmt::Assign { dst: Expr::Var(t), src: _ } = &body[i] else { continue };
        let t = *t;
        if vars[t].kind != VarKind::Local {
            continue;
        }
        let Some(rt) = any_reg_of(&vars[t].name).map(str::to_string) else { continue };
        // the next statement: `v = <expr of t>` with v in the same register, a new variable
        let Some(Stmt::Assign { dst: Expr::Var(v), src: vsrc }) = body.get(i + 1) else { continue };
        let v = *v;
        if v == t || vars[v].kind != VarKind::Local || any_reg_of(&vars[v].name) != Some(rt.as_str()) || !vsrc.uses_var(t) || vsrc.uses_var(v) {
            continue;
        }
        // t is read again after v's assignment (at top level, outside nested blocks: kept simple)
        let later: Vec<usize> = (i + 2..n).filter(|&k| stmt_uses(&body[k], t)).collect();
        if later.is_empty() || later.iter().any(|&k| !matches!(body[k], Stmt::Assign { .. } | Stmt::Expr(_) | Stmt::Return(_))) {
            continue;
        }
        // v isn't defined before i (it starts here)
        if (0..=i).any(|k| stmt_uses(&body[k], v)) {
            continue;
        }
        let p = vars.len();
        vars.push(Var { name: format!("{}_old", vars[t].name), ty: vars[t].ty.clone(), kind: VarKind::Local });
        for k in later {
            Stmt::rewrite_exprs(std::slice::from_mut(&mut body[k]), &mut |e| {
                if matches!(e, Expr::Var(x) if *x == t) {
                    *e = Expr::Var(p);
                }
            });
        }
        // v becomes t (one variable updated in place)
        Stmt::rewrite_exprs(&mut body[i + 1..], &mut |e| {
            if matches!(e, Expr::Var(x) if *x == v) {
                *e = Expr::Var(t);
            }
        });
        body.insert(i + 1, Stmt::Assign { dst: Expr::Var(p), src: Expr::Var(t) });
        return true;
    }
    false
}

fn stmt_uses(s: &Stmt, v: VarId) -> bool {
    let mut found = false;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| found |= matches!(e, Expr::Var(x) if *x == v));
    found
}
