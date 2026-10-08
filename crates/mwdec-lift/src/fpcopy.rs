//! Runs of floating-point copies between two objects (`lfd f0; lfd f1; stfd f0; lfd f2; ...`,
//! a copy constructor or assignment of a float-only class). Written with every loaded value
//! folded into its store, the compiler picks its own registers for the copies; the target's
//! rotation (three registers instead of two) comes from values that were named locals. Offered
//! as a draft variant: the run's temporaries stay named locals, which the emitter declares by
//! their target register (lowest first), and MWCC's colouring in declaration order then gives
//! each value the register the target used.
use crate::ir::*;
use crate::variants;

/// Draft variant: keep the temporaries of floating-point copy runs as named locals.
pub const NAMED_FP_COPIES: &str = variants::NAMED_FP_COPIES;

fn reg_of(name: &str) -> Option<u8> {
    let r = name.strip_prefix("temp_f")?;
    let digits: String = r.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// The temporaries of copy runs in `items`: a floating-point temp loaded from memory and stored,
/// unchanged, to memory by a later statement of the list, its only use.
fn run_temps(items: &[Stmt], vars: &[Var], is_temp: &[bool]) -> Vec<VarId> {
    let mut out = vec![];
    for (i, s) in items.iter().enumerate() {
        let Stmt::Assign { dst: Expr::Var(t), src } = s else { continue };
        let t = *t;
        // (an FPR temp: its type may still be unknown at this point)
        if !is_temp.get(t).copied().unwrap_or(false) || reg_of(&vars[t].name).is_none() {
            continue;
        }
        if !matches!(src, Expr::Load { .. } | Expr::Member { .. }) || src.has_call() {
            continue;
        }
        let uses: Vec<usize> = items.iter().enumerate().filter(|(k, x)| *k != i && stmt_reads(x, t)).map(|(k, _)| k).collect();
        let [u] = uses.as_slice() else { continue };
        if *u <= i || !matches!(&items[*u], Stmt::Assign { dst, src: Expr::Var(x) } if *x == t && !matches!(dst, Expr::Var(_))) {
            continue;
        }
        out.push(t);
    }
    out
}

fn stmt_reads(s: &Stmt, v: VarId) -> bool {
    let mut hit = false;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| hit |= matches!(e, Expr::Var(x) if *x == v));
    hit
}

/// Before temporaries are folded: keep the temporaries of floating-point copy runs (at least
/// four values over at least three registers) named when the draft variant is taken.
pub fn keep_copy_runs_named(lists: &[(Vec<Stmt>, u8)], vars: &[Var], is_temp: &mut [bool]) {
    let mut keep = vec![];
    for (items, _) in lists {
        let ts = run_temps(items, vars, is_temp);
        let mut regs: Vec<u8> = ts.iter().filter_map(|&t| reg_of(&vars[t].name)).collect();
        regs.sort_unstable();
        regs.dedup();
        if ts.len() >= 4 && regs.len() >= 3 {
            keep.extend(ts);
        }
    }
    if keep.is_empty() || !variants::alt(NAMED_FP_COPIES) {
        return;
    }
    for t in keep {
        is_temp[t] = false;
    }
}
