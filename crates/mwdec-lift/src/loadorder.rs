//! Values read from the parameters at the start of a function in source order (`lX = l.x;
//! lY = l.y; lZ = l.z; rX = r.x; ...`): the lift orders them by the registers the target used,
//! the compiler's list scheduler then ties on program order (draft variant
//! [`crate::variants::ORDER_PARAM_LOADS`] sorts the leading run by parameter, then offset).

use crate::ir::*;

fn key(e: &Expr, vars: &[Var]) -> Option<(usize, i32)> {
    match e {
        Expr::Load { base, offset, .. } => match &**base {
            Expr::Var(v) => match vars[*v].kind {
                VarKind::Param { index } => Some((index + 1, *offset)),
                VarKind::This => Some((0, *offset)),
                _ => None,
            },
            _ => None,
        },
        Expr::Member { base, offset, .. } => match &**base {
            Expr::Var(v) => match vars[*v].kind {
                VarKind::Param { index } => Some((index + 1, *offset)),
                VarKind::This => Some((0, *offset)),
                _ => None,
            },
            b => key(b, vars).map(|(p, o)| (p, o + *offset)),
        },
        Expr::Cast { e, .. } => key(e, vars),
        _ => None,
    }
}

pub fn order_param_loads(body: &mut Vec<Stmt>, vars: &[Var]) {
    let mut n = 0;
    let mut keys = vec![];
    while n < body.len() {
        match &body[n] {
            Stmt::Assign { dst: Expr::Var(v), src } if vars[*v].kind == VarKind::Local => match key(src, vars) {
                Some(k) => keys.push(k),
                None => break,
            },
            _ => break,
        }
        n += 1;
    }
    if n < 3 {
        return;
    }
    let mut sorted = keys.clone();
    sorted.sort();
    if sorted == keys || !crate::variants::alt(crate::variants::ORDER_PARAM_LOADS) {
        return;
    }
    let mut run: Vec<(usize, (usize, i32))> = keys.iter().copied().enumerate().collect();
    run.sort_by_key(|(_, k)| *k);
    let old: Vec<Stmt> = body[..n].to_vec();
    for (dst, (src, _)) in run.into_iter().enumerate() {
        body[dst] = old[src].clone();
    }
}
