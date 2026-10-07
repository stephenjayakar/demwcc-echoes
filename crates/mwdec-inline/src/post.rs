//! Clean-ups after folding: stack objects that only carried a folded value into one call become
//! the call's argument (`CAABox box(pos - d, pos + d)`), and one object value used twice (the
//! target shares its components) is named once (`CVector3f delta = a - b;`).

use mwdec_core::Type;
use mwdec_lift::{Callee, Expr, Stmt, Var, VarId, VarKind};
use std::collections::HashMap;

fn mentions_body(body: &[Stmt]) -> HashMap<VarId, usize> {
    let mut m = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            *m.entry(*v).or_default() += 1;
        }
    });
    m
}

fn mentions(s: &Stmt, v: VarId) -> usize {
    let mut n = 0;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
        if matches!(e, Expr::Var(x) if *x == v) {
            n += 1;
        }
    });
    n
}

/// A value our pass produced: a folded inline call or constructor (no side effects).
fn folded_value(e: &Expr) -> bool {
    match e {
        Expr::Call { callee: Callee::Direct { sig, .. } | Callee::Method { sig, .. }, .. } => sig.mangled.is_none(),
        Expr::Construct { .. } => true,
        _ => false,
    }
}

fn pure_stmt(s: &Stmt, vars: &[Var]) -> bool {
    match s {
        Stmt::Assign { dst: Expr::Var(v), src } => {
            matches!(vars[*v].kind, VarKind::Local | VarKind::Stack { .. }) && (!src.has_call() || folded_value(src))
        }
        _ => false,
    }
}

/// Replace the argument `&v` / `v` of a call or construction in `e` by `with`. Returns the
/// number of replacements.
fn replace_arg(e: &mut Expr, v: VarId, with: &Expr) -> usize {
    let mut n = 0;
    let is_v = |a: &Expr| match a {
        Expr::Var(x) => *x == v,
        Expr::AddrOf(x) => matches!(&**x, Expr::Var(y) if *y == v),
        _ => false,
    };
    e.rewrite(&mut |x| {
        let args = match x {
            Expr::Call { args, .. } | Expr::Construct { args, .. } | Expr::New { args, .. } => args,
            _ => return,
        };
        for a in args.iter_mut() {
            if is_v(a) {
                *a = with.clone();
                n += 1;
            }
        }
    });
    n
}

fn stmt_replace_arg(s: &mut Stmt, v: VarId, with: &Expr) -> usize {
    match s {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => replace_arg(e, v, with),
        Stmt::Assign { dst, src } => replace_arg(src, v, with) + replace_arg(dst, v, with),
        _ => 0,
    }
}

pub fn forward_stack_temps(body: &mut Vec<Stmt>, vars: &[Var]) -> usize {
    let counts = mentions_body(body);
    let mut n = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let (v, val) = match &b[i] {
                Stmt::Assign { dst: Expr::Var(v), src }
                    if matches!(vars[*v].kind, VarKind::Stack { .. }) && matches!(crate::util::strip(&vars[*v].ty), Type::Named(_)) && folded_value(src) && counts.get(v) == Some(&2) =>
                {
                    (*v, src.clone())
                }
                _ => {
                    i += 1;
                    continue;
                }
            };
            let Some(j) = (i + 1..b.len()).find(|&k| mentions(&b[k], v) > 0) else {
                i += 1;
                continue;
            };
            if !(i + 1..j).all(|k| pure_stmt(&b[k], vars)) {
                i += 1;
                continue;
            }
            let mut s = b[j].clone();
            if stmt_replace_arg(&mut s, v, &val) == 1 && mentions(&s, v) == 0 {
                b[j] = s;
                b.remove(i);
                n += 1;
                continue;
            }
            i += 1;
        }
    });
    n
}

fn collect_folded<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    e.walk(&mut |x| {
        if folded_value(x) && matches!(x, Expr::Call { ret, .. } if matches!(crate::util::strip(ret), Type::Named(_))) {
            out.push(x);
        }
    });
}

fn stmt_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => vec![e],
        Stmt::Assign { dst, src } => vec![dst, src],
        Stmt::If { cond, .. } | Stmt::While { cond, .. } => vec![cond],
        _ => vec![],
    }
}

/// Is `s` free of stores to memory and real calls?
fn no_effects(s: &Stmt) -> bool {
    match s {
        Stmt::Assign { dst: Expr::Var(_), src } => {
            let mut ok = true;
            src.walk(&mut |x| {
                if let Expr::Call { .. } = x {
                    if !folded_value(x) {
                        ok = false;
                    }
                }
            });
            ok
        }
        _ => false,
    }
}

pub fn name_shared_objects(body: &mut Vec<Stmt>, vars: &mut Vec<Var>) -> usize {
    let mut n = 0;
    let mut new_vars: Vec<Var> = vec![];
    let base = vars.len();
    Stmt::for_each_block_mut(body, &mut |b| {
        loop {
            // first duplicated folded object value in this list
            let mut found: Option<(Expr, usize, usize)> = None;
            'outer: for i in 0..b.len() {
                let mut here = vec![];
                for e in stmt_exprs(&b[i]) {
                    collect_folded(e, &mut here);
                }
                for e in here {
                    for k in i..b.len() {
                        let mut there = vec![];
                        for x in stmt_exprs(&b[k]) {
                            collect_folded(x, &mut there);
                        }
                        let cnt = there.iter().filter(|x| ***x == *e).count();
                        if (k == i && cnt >= 2) || (k > i && cnt >= 1) {
                            found = Some(((*e).clone(), i, k));
                            break 'outer;
                        }
                    }
                }
            }
            let Some((e, i, k)) = found else { break };
            // nothing in between may change the operands
            let last = (i..b.len()).rev().find(|&x| stmt_exprs(&b[x]).iter().any(|y| {
                let mut v = vec![];
                collect_folded(y, &mut v);
                v.iter().any(|z| **z == e)
            })).unwrap_or(k);
            if !(i..last).all(|x| x == i || no_effects(&b[x])) || !(i..=last).all(|x| x == i || x == last || no_effects(&b[x])) {
                break;
            }
            let Expr::Call { ret, .. } = &e else { break };
            let id = base + new_vars.len();
            new_vars.push(Var { name: format!("vec{}", new_vars.len()), ty: ret.clone(), kind: VarKind::Local });
            for x in i..=last {
                let mut rw = |y: &mut Expr| {
                    if *y == e {
                        *y = Expr::Var(id);
                    }
                };
                Stmt::rewrite_exprs(std::slice::from_mut(&mut b[x]), &mut rw);
            }
            b.insert(i, Stmt::Assign { dst: Expr::Var(id), src: e });
            n += 1;
        }
    });
    vars.extend(new_vars);
    n
}
