//! Temp folding: substitute single-use temporaries into their use when no side effect in
//! between could change the value (m2c's EvalOnce idea, done as an IR pass so the search stage
//! can re-run or undo it), plus dead temp elimination.

use crate::ir::*;
use std::collections::{HashMap, HashSet};

/// Count reads of every variable in a statement list (assignment destinations that are plain
/// vars are not reads).
pub fn count_uses(stmts: &[Stmt], uses: &mut HashMap<VarId, usize>) {
    for s in stmts {
        count_stmt(s, uses);
    }
}

fn count_expr(e: &Expr, uses: &mut HashMap<VarId, usize>) {
    e.walk(&mut |x| {
        if let Expr::Var(v) = x {
            *uses.entry(*v).or_default() += 1;
        }
    });
}

fn count_lvalue(e: &Expr, uses: &mut HashMap<VarId, usize>) {
    match e {
        Expr::Var(_) => {}
        Expr::Member { base, .. } => count_lvalue(base, uses),
        other => count_expr(other, uses),
    }
}

fn count_stmt(s: &Stmt, uses: &mut HashMap<VarId, usize>) {
    match s {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => count_expr(e, uses),
        Stmt::Assign { dst, src } => {
            count_lvalue(dst, uses);
            count_expr(src, uses);
        }
        Stmt::If { cond, then, els } => {
            count_expr(cond, uses);
            count_uses(then, uses);
            count_uses(els, uses);
        }
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            count_expr(cond, uses);
            count_uses(body, uses);
        }
        Stmt::For { init, cond, step, body } => {
            count_uses(init, uses);
            count_expr(cond, uses);
            count_uses(step, uses);
            count_uses(body, uses);
        }
        Stmt::Switch { e, cases } => {
            count_expr(e, uses);
            for c in cases {
                count_uses(&c.body, uses);
            }
        }
        _ => {}
    }
}

/// Effects summary of an expression.
#[derive(Default, Debug, Clone)]
pub struct Effects {
    pub reads_mem: bool,
    pub calls: bool,
    /// members of reference parameters read (ordered with calls only)
    pub reads_ref: bool,
    /// non-temp vars read
    pub vars: HashSet<VarId>,
}

pub fn effects(e: &Expr, is_temp: &[bool], vars: &[Var]) -> Effects {
    let mut fx = Effects::default();
    effects_into(e, is_temp, vars, &mut fx, false);
    fx
}

fn effects_into(e: &Expr, is_temp: &[bool], vars: &[Var], fx: &mut Effects, addr: bool) {
    match e {
        Expr::Var(v) => {
            if !is_temp.get(*v).copied().unwrap_or(false) {
                fx.vars.insert(*v);
                // address-taken stack aggregates behave like memory
                if !addr && matches!(vars[*v].kind, VarKind::Stack { .. }) && matches!(vars[*v].ty, mwdec_core::Type::Named(_) | mwdec_core::Type::Unknown { .. }) {
                    fx.reads_mem = true;
                }
            }
        }
        Expr::AddrOf(inner) => effects_into(inner, is_temp, vars, fx, true),
        Expr::Load { base, .. } => {
            if !addr {
                fx.reads_mem = true;
            }
            effects_into(base, is_temp, vars, fx, false);
        }
        Expr::Index { base, index, .. } => {
            if !addr {
                fx.reads_mem = true;
            }
            effects_into(base, is_temp, vars, fx, false);
            effects_into(index, is_temp, vars, fx, false);
        }
        Expr::Member { base, .. } => {
            if !addr {
                if let Expr::Var(v) = &**base {
                    if matches!(vars[*v].kind, VarKind::Stack { .. }) {
                        fx.reads_mem = true;
                    }
                    // reference / by-value aggregate parameters are vars of the object type: their
                    // members are memory a call may change (stores are left unordered: MWCC
                    // hoists such loads above `this->x` stores)
                    if matches!(vars[*v].kind, VarKind::Param { .. }) {
                        fx.reads_ref = true;
                    }
                } else {
                    fx.reads_mem = true;
                }
            }
            effects_into(base, is_temp, vars, fx, addr);
        }
        Expr::Global { .. } => {
            if !addr {
                fx.reads_mem = true;
            }
        }
        Expr::Call { args, .. } if e.is_pure_call() => {
            for a in args {
                effects_into(a, is_temp, vars, fx, false);
            }
        }
        Expr::Call { callee, args, .. } => {
            fx.calls = true;
            fx.reads_mem = true;
            match callee {
                Callee::Method { this, .. } | Callee::Virtual { this, .. } => effects_into(this, is_temp, vars, fx, false),
                Callee::Indirect(e) => effects_into(e, is_temp, vars, fx, false),
                _ => {}
            }
            for a in args {
                effects_into(a, is_temp, vars, fx, false);
            }
        }
        Expr::IncDec { e, .. } => {
            // a write: order it like a call
            fx.calls = true;
            fx.reads_mem = true;
            effects_into(e, is_temp, vars, fx, false);
        }
        Expr::BitField { base, .. } => {
            fx.reads_mem = true;
            effects_into(base, is_temp, vars, fx, true);
        }
        Expr::Construct { args, .. } => {
            fx.calls = true;
            fx.reads_mem = true;
            for a in args {
                effects_into(a, is_temp, vars, fx, false);
            }
        }
        Expr::New { placement, args, .. } => {
            fx.calls = true;
            fx.reads_mem = true;
            for a in placement.iter().chain(args.iter()) {
                effects_into(a, is_temp, vars, fx, false);
            }
        }
        Expr::Unary { e, .. } | Expr::Cast { e, .. } => effects_into(e, is_temp, vars, fx, false),
        Expr::Binary { l, r, .. } => {
            effects_into(l, is_temp, vars, fx, false);
            effects_into(r, is_temp, vars, fx, false);
        }
        Expr::Ternary { c, t, f, .. } => {
            effects_into(c, is_temp, vars, fx, false);
            effects_into(t, is_temp, vars, fx, false);
            effects_into(f, is_temp, vars, fx, false);
        }
        _ => {}
    }
}

/// What a statement does: writes (vars / memory), calls.
fn stmt_writes(s: &Stmt, is_temp: &[bool], vars: &[Var]) -> (HashSet<VarId>, bool, Effects) {
    let mut wv = HashSet::new();
    let mut wmem = false;
    let mut fx = Effects::default();
    match s {
        Stmt::Assign { dst, src } => {
            match dst {
                Expr::Var(v) => {
                    wv.insert(*v);
                }
                Expr::Member { base, .. } if matches!(**base, Expr::Var(_)) => {
                    if let Expr::Var(v) = **base {
                        wv.insert(v);
                        wmem = true;
                    }
                }
                other => {
                    wmem = true;
                    if let Expr::Global { .. } = other {}
                    let f2 = effects(other, is_temp, vars);
                    fx.vars.extend(f2.vars);
                    fx.calls |= f2.calls;
                }
            }
            let f1 = effects(src, is_temp, vars);
            fx.reads_mem |= f1.reads_mem;
            fx.reads_ref |= f1.reads_ref;
            fx.calls |= f1.calls;
            fx.vars.extend(f1.vars);
        }
        Stmt::Expr(e) | Stmt::Return(Some(e)) => {
            fx = effects(e, is_temp, vars);
        }
        Stmt::Switch { e, .. } => fx = effects(e, is_temp, vars),
        _ => {
            // structured statements: be conservative
            wmem = true;
            fx.calls = true;
            fx.reads_mem = true;
        }
    }
    if fx.calls {
        wmem = true;
    }
    (wv, wmem, fx)
}

fn conflicts(fx_e: &Effects, s: &Stmt, is_temp: &[bool], vars: &[Var]) -> bool {
    let (wv, wmem, fx_s) = stmt_writes(s, is_temp, vars);
    if wv.iter().any(|v| fx_e.vars.contains(v)) {
        return true;
    }
    if wmem && (fx_e.reads_mem || fx_e.calls) {
        return true;
    }
    if fx_e.reads_ref && fx_s.calls {
        return true;
    }
    // moving a call later past a memory read
    if fx_e.calls && (fx_s.reads_mem || fx_s.reads_ref || fx_s.calls) {
        return true;
    }
    // a call may modify address-taken stack vars read by e
    if fx_s.calls && fx_e.vars.iter().any(|v| matches!(vars[*v].kind, VarKind::Stack { .. })) {
        return true;
    }
    false
}

/// Number of calls in `e` that do not contain `Var(t)` (siblings evaluated in unspecified order).
fn sibling_calls(e: &Expr, t: VarId) -> usize {
    let mut n = 0;
    e.walk(&mut |x| {
        if let Expr::Call { .. } = x {
            if !x.uses_var(t) && !x.is_pure_call() {
                n += 1;
            }
        }
    });
    n
}

/// Same, but memory reads that don't contain t (for moving a call into an expression that reads memory).
fn sibling_reads(e: &Expr, t: VarId) -> bool {
    let mut r = false;
    e.walk(&mut |x| {
        if matches!(x, Expr::Load { .. } | Expr::Index { .. } | Expr::Global { .. }) && !x.uses_var(t) {
            r = true;
        }
    });
    r
}

fn stmt_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Assign { dst, src } => vec![dst, src],
        Stmt::Expr(e) | Stmt::Return(Some(e)) => vec![e],
        Stmt::Switch { e, .. } => vec![e],
        Stmt::If { cond, .. } => vec![cond],
        _ => vec![],
    }
}

fn substitute(s: &mut Stmt, t: VarId, with: &Expr) {
    let mut f = |e: &mut Expr| {
        if let Expr::Var(v) = e {
            if *v == t {
                *e = with.clone();
            }
        }
    };
    match s {
        Stmt::Assign { dst, src } => {
            dst.rewrite(&mut f);
            src.rewrite(&mut f);
        }
        Stmt::Expr(e) | Stmt::Return(Some(e)) => e.rewrite(&mut f),
        Stmt::Switch { e, .. } => e.rewrite(&mut f),
        Stmt::If { cond, .. } => cond.rewrite(&mut f),
        _ => {}
    }
}

fn uses_in_stmt(s: &Stmt, t: VarId) -> usize {
    let mut n = 0;
    for e in stmt_exprs(s) {
        e.walk(&mut |x| {
            if matches!(x, Expr::Var(v) if *v == t) {
                n += 1;
            }
        });
    }
    n
}

/// Pure, cheap expressions that may be duplicated into several uses.
pub fn duplicable(e: &Expr, is_temp: &[bool], vars: &[Var]) -> bool {
    match e {
        Expr::Int { .. } | Expr::Float { .. } | Expr::Var(_) | Expr::FuncAddr { .. } => true,
        Expr::AddrOf(inner) => {
            let fx = effects(e, is_temp, vars);
            !fx.reads_mem && !fx.calls && matches!(**inner, Expr::Load { .. } | Expr::Member { .. } | Expr::Global { .. } | Expr::Var(_))
        }
        Expr::Cast { e, .. } => matches!(**e, Expr::Var(_)),
        _ => false,
    }
}

fn is_temp_assign(s: &Stmt, is_temp: &[bool]) -> bool {
    matches!(s, Stmt::Assign { dst: Expr::Var(v), .. } if is_temp.get(*v).copied().unwrap_or(false))
}

fn stmt_has_call(s: &Stmt) -> bool {
    stmt_exprs(s).iter().any(|e| e.has_call())
}

/// MWCC never schedules an instruction across a call, so a statement whose code follows the call
/// in the target came after it in the source. Folding a call into a later statement `j` across
/// such a statement would move the call after it. When those statements are pure register
/// assignments independent of `j` (and `j` is a plain statement without other calls), they can
/// move after `j` instead (`r = f() > 0; p = q;` rather than `p = q; r = f() > 0;`).
fn movable_after(items: &[Stmt], blockers: &[usize], j: usize, fixed_tail: usize, is_temp: &[bool], vars: &[Var]) -> bool {
    if j + fixed_tail >= items.len() || !matches!(items[j], Stmt::Assign { .. } | Stmt::Expr(_)) || stmt_has_call(&items[j]) {
        return false;
    }
    let (jw, jmem, jfx) = stmt_writes(&items[j], is_temp, vars);
    blockers.iter().all(|&k| match &items[k] {
        Stmt::Assign { dst: Expr::Var(x), src } if !src.has_call() => {
            let fx = effects(src, is_temp, vars);
            !jfx.vars.contains(x) && !jw.contains(x) && !jw.iter().any(|w| fx.vars.contains(w)) && !(jmem && fx.reads_mem) && uses_in_stmt(&items[j], *x) == 0
        }
        _ => false,
    })
}

/// Fold temps within one straight-line statement list. `uses` are function-wide read counts.
/// The last `fixed_tail` items are block terminators (branch condition, return value) that must
/// stay last.
pub fn inline_list(items: &mut Vec<Stmt>, uses: &mut HashMap<VarId, usize>, is_temp: &[bool], vars: &[Var], fixed_tail: usize) {
    let mut i = 0;
    while i < items.len() {
        let (t, src) = match &items[i] {
            Stmt::Assign { dst: Expr::Var(t), src } if is_temp.get(*t).copied().unwrap_or(false) => (*t, src.clone()),
            _ => {
                i += 1;
                continue;
            }
        };
        let total = uses.get(&t).copied().unwrap_or(0);
        if total == 0 {
            i += 1;
            continue;
        }
        // uses within this list after i
        let mut local = 0;
        let mut first_j = None;
        for j in i + 1..items.len() {
            let n = uses_in_stmt(&items[j], t);
            if n > 0 && first_j.is_none() {
                first_j = Some(j);
            }
            local += n;
            // a later reassignment of a var the temp reads doesn't matter for SSA temps
        }
        let Some(j) = first_j else {
            i += 1;
            continue;
        };
        let fx = effects(&src, is_temp, vars);
        if total == 1 && local == 1 {
            let mut ok = (i + 1..j).all(|k| !conflicts(&fx, &items[k], is_temp, vars));
            // MWCC never moves code across a call: a value computed before a call statement and
            // read after it was computed there in the source (a local), even when pure
            if ok && !fx.calls && (i + 1..j).any(|k| stmt_has_call(&items[k])) {
                ok = false;
            }
            // nor across a branch or loop: the value was computed in an earlier block
            if ok && (i + 1..j).any(|k| matches!(items[k], Stmt::If { .. } | Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::Switch { .. })) {
                ok = false;
            }
            if ok && (fx.calls || fx.reads_mem) {
                // unspecified evaluation order among siblings inside the use statement
                for e in stmt_exprs(&items[j]) {
                    if e.uses_var(t) {
                        if sibling_calls(e, t) > 0 && (fx.calls || fx.reads_mem) {
                            // allowed if the sibling call is the consumer (an ancestor): sibling_calls
                            // counts only calls that don't contain t, so any is a real sibling
                            ok = false;
                        }
                        if fx.calls && sibling_reads(e, t) {
                            ok = false;
                        }
                    }
                }
                // the destination of an assignment is evaluated too
                if let Stmt::Assign { dst, src: s2 } = &items[j] {
                    if s2.uses_var(t) && fx.calls {
                        let dfx = effects(dst, is_temp, vars);
                        if dfx.reads_mem && !matches!(dst, Expr::Var(_)) {
                            // `p->x = f()` is fine; reading p happens after either way in MWCC
                        }
                    }
                }
            }
            let mut move_after: Vec<usize> = vec![];
            if ok && fx.calls {
                // temps the use doesn't read keep their place after the call too (`t = f(); p =
                // this + 0x30; v = t > 0;`: p was computed after the call)
                let blockers: Vec<usize> = (i + 1..j)
                    .filter(|&k| !is_temp_assign(&items[k], is_temp) || matches!(&items[k], Stmt::Assign { dst: Expr::Var(x), .. } if uses_in_stmt(&items[j], *x) == 0))
                    .collect();
                if !blockers.is_empty() {
                    if movable_after(items, &blockers, j, fixed_tail, is_temp, vars) {
                        move_after = blockers;
                    } else {
                        ok = false;
                    }
                }
            }
            if ok {
                substitute(&mut items[j], t, &src);
                if !move_after.is_empty() {
                    let moved: Vec<Stmt> = move_after.iter().map(|&k| items[k].clone()).collect();
                    let mut jj = j;
                    for &k in move_after.iter().rev() {
                        items.remove(k);
                        jj -= 1;
                    }
                    for (n, m) in moved.into_iter().enumerate() {
                        items.insert(jj + 1 + n, m);
                    }
                }
                items.remove(i);
                uses.insert(t, 0);
                continue;
            }
        } else if total == local && duplicable(&src, is_temp, vars) {
            // all uses are here: check no var it reads is written before the last use
            let last = (i + 1..items.len()).rev().find(|&k| uses_in_stmt(&items[k], t) > 0).unwrap();
            let ok = (i + 1..last).all(|k| {
                let (wv, _, _) = stmt_writes(&items[k], is_temp, vars);
                !wv.iter().any(|v| fx.vars.contains(v))
            });
            if ok {
                for k in i + 1..=last {
                    substitute(&mut items[k], t, &src);
                }
                items.remove(i);
                uses.insert(t, 0);
                continue;
            }
        }
        i += 1;
    }
}

/// Remove assignments to unused temps (keeping calls as expression statements). Returns true if
/// anything changed.
pub fn dce(stmts: &mut Vec<Stmt>, uses: &HashMap<VarId, usize>, is_temp: &[bool]) -> bool {
    let mut changed = false;
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts.drain(..) {
        match s {
            Stmt::Assign { dst: Expr::Var(t), src } if is_temp.get(t).copied().unwrap_or(false) && uses.get(&t).copied().unwrap_or(0) == 0 => {
                changed = true;
                if src.has_call() {
                    // keep the call (drop the result)
                    if let Expr::Call { .. } = src {
                        out.push(Stmt::Expr(src));
                    } else {
                        let mut calls = vec![];
                        collect_calls(&src, &mut calls);
                        out.extend(calls.into_iter().map(Stmt::Expr));
                    }
                }
            }
            s => out.push(s),
        }
    }
    *stmts = out;
    changed
}

fn collect_calls(e: &Expr, out: &mut Vec<Expr>) {
    if let Expr::Call { .. } = e {
        out.push(e.clone());
        return;
    }
    match e {
        Expr::AddrOf(x) | Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } => collect_calls(x, out),
        Expr::Load { base, .. } | Expr::Member { base, .. } => collect_calls(base, out),
        Expr::Index { base, index, .. } => {
            collect_calls(base, out);
            collect_calls(index, out);
        }
        Expr::Binary { l, r, .. } => {
            collect_calls(l, out);
            collect_calls(r, out);
        }
        Expr::Ternary { c, t, f, .. } => {
            collect_calls(c, out);
            collect_calls(t, out);
            collect_calls(f, out);
        }
        _ => {}
    }
}
