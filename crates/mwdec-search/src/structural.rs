//! Structural operators that statement permutation cannot reach (catalog rows 6, 7, 10, 12, 13,
//! 16, 21): switch <-> if-chain and case order, compare-constant forms, bool return forms,
//! guard splitting/merging, loop exit forms, float literal forms. Same contract as [`crate::ops`]:
//! best-effort semantics preserving rewrites; the strict comparator is the judge.
use crate::cst::{strip_parens, Cst, Edit};
use crate::func::{self, infer_type, is_int_type};
use crate::ops::M;

fn indent_of(cst: &Cst, n: usize) -> String {
    let s = cst.nodes[n].start;
    let ls = cst.src[..s].rfind('\n').map(|i| i + 1).unwrap_or(0);
    cst.src[ls..s].chars().take_while(|c| c.is_whitespace()).collect()
}

fn cond_value(c: &Cst, stmt: usize) -> Option<usize> {
    let cc = c.child(stmt, "condition")?;
    match c.kind(cc) {
        "condition_clause" => {
            let v = c.child(cc, "value")?;
            (c.kind(v) != "declaration").then_some(v)
        }
        _ => None,
    }
}

fn pure_simple(c: &Cst, n: usize) -> bool {
    !c.descendants(n).into_iter().any(|d| {
        matches!(c.kind(d), "call_expression" | "assignment_expression" | "update_expression" | "new_expression" | "delete_expression")
    })
}

/// Case label text: integer literal, enum constant or qualified enum constant (char literal too).
fn is_case_const(c: &Cst, n: usize) -> bool {
    match c.kind(n) {
        "number_literal" | "identifier" | "qualified_identifier" | "char_literal" => true,
        "unary_expression" => c.op(n) == Some("-") && c.child(n, "argument").is_some_and(|a| c.kind(a) == "number_literal"),
        "parenthesized_expression" => c.named(n).first().is_some_and(|&i| is_case_const(c, i)),
        _ => false,
    }
}

/// Statements (named, non-comment) of a case_statement after the label.
fn case_body(c: &Cst, case: usize) -> Vec<usize> {
    let v = c.child(case, "value");
    c.named(case).into_iter().filter(|&s| Some(s) != v && func::is_stmt(c.kind(s))).collect()
}

/// `break` statements that belong to switch/loop `owner` inside `n`.
fn own_breaks(c: &Cst, owner: usize, n: usize) -> Vec<usize> {
    c.descendants(n)
        .into_iter()
        .filter(|&d| c.kind(d) == "break_statement")
        .filter(|&d| c.ancestors(d).into_iter().find(|&a| matches!(c.kind(a), "switch_statement" | "for_statement" | "while_statement" | "do_statement")) == Some(owner))
        .collect()
}

fn ends_in_jump(c: &Cst, s: usize) -> bool {
    match c.kind(s) {
        "return_statement" | "break_statement" | "continue_statement" | "goto_statement" => true,
        "compound_statement" => c.named(s).into_iter().filter(|&x| func::is_stmt(c.kind(x))).last().is_some_and(|l| ends_in_jump(c, l)),
        _ => false,
    }
}

struct CaseGroup {
    /// label value nodes (None = default)
    labels: Vec<Option<usize>>,
    /// body statements (break at the end stripped)
    body: Vec<usize>,
    /// ends with break (stripped), return etc. (kept), or falls through
    terminated: bool,
    /// case_statement nodes covered
    nodes: Vec<usize>,
}

/// Case groups of a switch body; `None` if some group falls through into the next or uses
/// `break` other than as its last statement.
fn case_groups(c: &Cst, sw: usize) -> Option<Vec<CaseGroup>> {
    let body = c.child(sw, "body")?;
    let cases: Vec<usize> = c.named(body).into_iter().filter(|&x| c.kind(x) != "comment").collect();
    if cases.iter().any(|&x| c.kind(x) != "case_statement") {
        return None;
    }
    let mut groups: Vec<CaseGroup> = Vec::new();
    let mut pending: Vec<Option<usize>> = Vec::new();
    let mut pend_nodes = Vec::new();
    for (k, &cs) in cases.iter().enumerate() {
        pending.push(c.child(cs, "value"));
        pend_nodes.push(cs);
        let mut b = case_body(c, cs);
        if b.is_empty() {
            if k + 1 == cases.len() {
                groups.push(CaseGroup { labels: std::mem::take(&mut pending), body: vec![], terminated: true, nodes: std::mem::take(&mut pend_nodes) });
            }
            continue;
        }
        let mut terminated = false;
        if let Some(&l) = b.last() {
            if c.kind(l) == "break_statement" {
                b.pop();
                terminated = true;
            } else if ends_in_jump(c, l) {
                terminated = true;
            }
        }
        // no other breaks of this switch inside
        if b.iter().any(|&s| !own_breaks(c, sw, s).is_empty()) {
            return None;
        }
        if !terminated && k + 1 != cases.len() {
            return None;
        }
        groups.push(CaseGroup { labels: std::mem::take(&mut pending), body: b, terminated: true, nodes: std::mem::take(&mut pend_nodes) });
    }
    Some(groups)
}

fn stmts_text(c: &Cst, v: &[usize], ind: &str) -> String {
    v.iter().map(|&s| format!("{ind}    {}\n", c.text(s))).collect()
}

/// switch -> if/else-if chain (default last).
pub fn op_switch_to_if(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let sws: Vec<usize> = m.nodes.iter().copied().filter(|&n| c.kind(n) == "switch_statement").collect();
    let sw = m.pick_one(&sws)?;
    let x = cond_value(c, sw)?;
    if !pure_simple(c, x) {
        return None;
    }
    let groups = case_groups(c, sw)?;
    let ind = indent_of(c, sw);
    let xt = c.text(x);
    let mut arms: Vec<(String, String)> = Vec::new();
    let mut default: Option<String> = None;
    for g in &groups {
        let body = stmts_text(c, &g.body, &ind);
        if g.labels.iter().any(|l| l.is_none()) {
            default = Some(body);
            continue;
        }
        let conds: Vec<String> = g.labels.iter().map(|l| format!("{xt} == {}", c.text(l.unwrap()))).collect();
        arms.push((conds.join(" || "), body));
    }
    if arms.is_empty() {
        return None;
    }
    let mut out = String::new();
    for (k, (cond, body)) in arms.iter().enumerate() {
        if k > 0 {
            out.push_str(" else ");
        }
        out.push_str(&format!("if ({cond}) {{\n{body}{ind}}}"));
    }
    if let Some(d) = default {
        if !d.trim().is_empty() {
            out.push_str(&format!(" else {{\n{d}{ind}}}"));
        }
    }
    Some(vec![Edit::replace(c, sw, out)])
}

/// `x == K1 || x == K2` -> (x, [K1, K2]) for a pure x and case constants.
fn eq_tests(c: &Cst, e: usize) -> Option<(String, Vec<String>)> {
    let e = match c.kind(e) {
        "parenthesized_expression" => *c.named(e).first()?,
        _ => e,
    };
    if c.kind(e) != "binary_expression" {
        return None;
    }
    match c.op(e)? {
        "==" => {
            let (l, r) = (c.child(e, "left")?, c.child(e, "right")?);
            let (x, k) = if is_case_const(c, r) && !is_case_const(c, l) {
                (l, r)
            } else if is_case_const(c, l) && !is_case_const(c, r) {
                (r, l)
            } else if is_case_const(c, r) {
                (l, r)
            } else {
                return None;
            };
            if !pure_simple(c, x) || c.kind(k) == "identifier" && c.kind(x) == "number_literal" {
                return None;
            }
            Some((strip_parens(c.text(x)).to_string(), vec![c.text(k).to_string()]))
        }
        "||" => {
            let (a, ka) = eq_tests(c, c.child(e, "left")?)?;
            let (b, kb) = eq_tests(c, c.child(e, "right")?)?;
            (a == b).then(|| (a, [ka, kb].concat()))
        }
        _ => None,
    }
}

fn body_lines(c: &Cst, s: usize, ind: &str) -> (String, bool) {
    let stmts: Vec<usize> = if c.kind(s) == "compound_statement" { c.named(s).into_iter().filter(|&x| func::is_stmt(c.kind(x))).collect() } else { vec![s] };
    let jumps = stmts.last().is_some_and(|&l| ends_in_jump(c, l));
    (stmts.iter().map(|&x| format!("{ind}    {}\n", c.text(x))).collect(), jumps)
}

/// if/else-if chain (or a run of `if (x == K) {...; return;}` statements) on one value -> switch.
pub fn op_if_to_switch(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) != "if_statement" {
            continue;
        }
        // chain heads only
        if c.parent(n).is_some_and(|p| c.kind(p) == "else_clause") {
            continue;
        }
        let ind = indent_of(c, n);
        // (a) else-if chain
        let mut arms: Vec<(Vec<String>, String, bool)> = Vec::new();
        let mut x: Option<String> = None;
        let mut cur = n;
        let mut default: Option<(String, bool)> = None;
        let ok = loop {
            let Some(cv) = cond_value(c, cur) else { break false };
            let Some((xv, ks)) = eq_tests(c, cv) else { break false };
            if x.as_ref().is_some_and(|x0| *x0 != xv) {
                break false;
            }
            x = Some(xv);
            let Some(cons) = c.child(cur, "consequence") else { break false };
            if !own_breaks_any(c, cons) {
                let (b, j) = body_lines(c, cons, &ind);
                arms.push((ks, b, j));
            } else {
                break false;
            }
            match c.child(cur, "alternative") {
                None => break true,
                Some(alt) => {
                    let inner = c.named(alt).into_iter().find(|&s| func::is_stmt(c.kind(s)));
                    match inner {
                        Some(s) if c.kind(s) == "if_statement" && cond_value(c, s).and_then(|v| eq_tests(c, v)).is_some_and(|(xv, _)| Some(&xv) == x.as_ref()) => cur = s,
                        Some(s) => {
                            if own_breaks_any(c, s) {
                                break false;
                            }
                            default = Some(body_lines(c, s, &ind));
                            break true;
                        }
                        None => break false,
                    }
                }
            }
        };
        if ok && arms.len() >= 2 {
            let mut t = format!("switch ({}) {{\n", x.clone().unwrap());
            for (ks, b, j) in &arms {
                for k in ks {
                    t.push_str(&format!("{ind}case {k}:\n"));
                }
                t.push_str(b);
                if !j {
                    t.push_str(&format!("{ind}    break;\n"));
                }
            }
            if let Some((b, j)) = &default {
                t.push_str(&format!("{ind}default:\n{b}"));
                if !j {
                    t.push_str(&format!("{ind}    break;\n"));
                }
            }
            t.push_str(&format!("{ind}}}"));
            cands.push((n, n, t));
        }
        // (b) run of sibling `if (x == K) { ...; return; }` without else
        let Some(p) = c.parent(n) else { continue };
        if c.kind(p) != "compound_statement" || c.child(n, "alternative").is_some() {
            continue;
        }
        if c.parent(n).is_some_and(|pp| c.kind(pp) == "else_clause") {
            continue;
        }
        let sibs: Vec<usize> = c.named(p).into_iter().filter(|&s| func::is_stmt(c.kind(s))).collect();
        let Some(i0) = sibs.iter().position(|&s| s == n) else { continue };
        if i0 > 0 && c.kind(sibs[i0 - 1]) == "if_statement" {
            // only start at the first if of a run
            let prev = sibs[i0 - 1];
            if c.child(prev, "alternative").is_none() && cond_value(c, prev).and_then(|v| eq_tests(c, v)).is_some() {
                continue;
            }
        }
        let mut run = Vec::new();
        let mut xr: Option<String> = None;
        for &s in &sibs[i0..] {
            if c.kind(s) != "if_statement" || c.child(s, "alternative").is_some() {
                break;
            }
            let Some((xv, ks)) = cond_value(c, s).and_then(|v| eq_tests(c, v)) else { break };
            if xr.as_ref().is_some_and(|x0| *x0 != xv) {
                break;
            }
            let Some(cons) = c.child(s, "consequence") else { break };
            if !ends_in_jump(c, cons) || own_breaks_any(c, cons) {
                break;
            }
            xr = Some(xv);
            run.push((s, ks, cons));
        }
        if run.len() >= 2 {
            let mut t = format!("switch ({}) {{\n", xr.unwrap());
            for (_, ks, cons) in &run {
                for k in ks {
                    t.push_str(&format!("{ind}case {k}:\n"));
                }
                t.push_str(&body_lines(c, *cons, &ind).0);
            }
            t.push_str(&format!("{ind}}}"));
            cands.push((run[0].0, run.last().unwrap().0, t));
        }
    }
    let k = m.rng.below(cands.len().max(1));
    let (a, b, t) = cands.get(k)?.clone();
    Some(vec![Edit { start: c.nodes[a].start, end: c.nodes[b].end, text: t }])
}

/// Any `break` inside `n` that is not owned by a loop/switch inside `n` (it would bind to a new
/// switch).
fn own_breaks_any(c: &Cst, n: usize) -> bool {
    c.descendants(n).into_iter().filter(|&d| c.kind(d) == "break_statement").any(|d| {
        !c.ancestors(d)
            .into_iter()
            .take_while(|&a| a != n)
            .any(|a| matches!(c.kind(a), "switch_statement" | "for_statement" | "while_statement" | "do_statement"))
    })
}

/// Swap two adjacent case groups (both terminated) of a switch: body order follows label order.
pub fn op_case_order(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for sw in m.nodes.iter().copied().filter(|&n| c.kind(n) == "switch_statement") {
        let Some(groups) = case_groups(c, sw) else { continue };
        for k in 0..groups.len().saturating_sub(1) {
            let (a, b) = (&groups[k], &groups[k + 1]);
            // the last group may lack a trailing break: only swap when it ends in a jump
            let last_ok = |g: &CaseGroup| g.terminated && (g.body.last().is_some_and(|&l| ends_in_jump(c, l)) || has_break_after(c, g));
            if !last_ok(a) || !last_ok(b) {
                continue;
            }
            cands.push((a.nodes[0], *a.nodes.last().unwrap(), b.nodes[0], *b.nodes.last().unwrap()));
        }
    }
    let (a0, a1, b0, b1) = m.pick_one(&cands)?;
    let ta = &c.src[c.nodes[a0].start..c.nodes[a1].end];
    let tb = &c.src[c.nodes[b0].start..c.nodes[b1].end];
    let gap = &c.src[c.nodes[a1].end..c.nodes[b0].start];
    Some(vec![Edit { start: c.nodes[a0].start, end: c.nodes[b1].end, text: format!("{tb}{gap}{ta}") }])
}

fn has_break_after(c: &Cst, g: &CaseGroup) -> bool {
    let last = *g.nodes.last().unwrap();
    case_body(c, last).last().is_some_and(|&l| c.kind(l) == "break_statement")
}

/// Integer compare against a literal in the neighbouring form: `x < K` <-> `x <= K-1`,
/// `x > K` <-> `x >= K+1`; unsigned `x != 0` <-> `x > 0`.
pub fn op_cmp_const(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for b in m.nodes.iter().copied().filter(|&n| c.kind(n) == "binary_expression") {
        let Some(op) = c.op(b) else { continue };
        let (Some(l), Some(r)) = (c.child(b, "left"), c.child(b, "right")) else { continue };
        let Some(k) = int_lit(c, r) else { continue };
        let lt = infer_type(c, m.info, l);
        if lt.as_deref().is_some_and(|t| !is_int_type(t)) {
            continue;
        }
        let unsigned = lt.as_deref().is_some_and(|t| t.contains("unsigned") || t.starts_with('u'));
        let lx = c.text(l);
        let alt = match op {
            "<" => Some(format!("{lx} <= {}", k - 1)),
            "<=" => Some(format!("{lx} < {}", k + 1)),
            ">" => Some(format!("{lx} >= {}", k + 1)),
            ">=" => Some(format!("{lx} > {}", k - 1)),
            "!=" if k == 0 && unsigned => Some(format!("{lx} > 0")),
            "==" if k == 0 && unsigned => Some(format!("{lx} < 1")),
            _ => None,
        };
        if let Some(a) = alt {
            if !a.contains("-1") || !unsigned {
                cands.push((b, a));
            }
        }
    }
    let (n, t) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, t)])
}

fn int_lit(c: &Cst, n: usize) -> Option<i64> {
    match c.kind(n) {
        "number_literal" => {
            let t = c.text(n).trim_end_matches(['u', 'U', 'l', 'L']).to_ascii_lowercase();
            if t.contains('.') || t.ends_with('f') && !t.starts_with("0x") {
                return None;
            }
            match t.strip_prefix("0x") {
                Some(h) => i64::from_str_radix(h, 16).ok(),
                None => t.parse().ok(),
            }
        }
        "unary_expression" if c.op(n) == Some("-") => int_lit(c, c.child(n, "argument")?).map(|v| -v),
        _ => None,
    }
}

fn is_condish(c: &Cst, e: usize) -> bool {
    match c.kind(e) {
        "binary_expression" => matches!(c.op(e), Some("<" | ">" | "<=" | ">=" | "==" | "!=" | "&&" | "||")),
        "unary_expression" => c.op(e) == Some("!"),
        "parenthesized_expression" => c.named(e).first().is_some_and(|&i| is_condish(c, i)),
        _ => false,
    }
}

/// Bool return forms: `return c;` <-> `if (c) { return T; } return F;` <-> `return c ? T : F;`
/// and a flag variable `R r = F; if (c) { r = T; } return r;`.
pub fn op_bool_return(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let ret = m.info.ret_ty.trim().to_string();
    let (t, f) = if ret == "bool" { ("true", "false") } else if is_int_type(&ret) { ("1", "0") } else { return None };
    let mut cands: Vec<(usize, usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) == "return_statement" {
            let Some(&e) = c.named(n).first() else { continue };
            if is_condish(c, e) && c.parent(n).is_some_and(|p| c.kind(p) == "compound_statement") {
                let ind = indent_of(c, n);
                let et = strip_parens(c.text(e));
                cands.push((n, n, format!("if ({et}) {{\n{ind}    return {t};\n{ind}}}\n{ind}return {f};")));
                cands.push((n, n, format!("return ({et}) ? {t} : {f};")));
                let v = m.info.fresh_name(c, "ret_");
                cands.push((n, n, format!("{ret} {v} = {f};\n{ind}if ({et}) {{\n{ind}    {v} = {t};\n{ind}}}\n{ind}return {v};")));
            }
            // `return c ? T : F;` -> `return c;`
            if c.kind(e) == "conditional_expression" {
                let (Some(cc), Some(a), Some(b)) = (c.child(e, "condition"), c.child(e, "consequence"), c.child(e, "alternative")) else { continue };
                let (at, bt) = (c.text(a), c.text(b));
                if (at == t || at == "true" || at == "1") && (bt == f || bt == "false" || bt == "0") {
                    cands.push((n, n, format!("return {};", strip_parens(c.text(cc)))));
                }
            }
        }
        // `if (c) { return T; } return F;` -> `return c;`
        if c.kind(n) == "if_statement" && c.child(n, "alternative").is_none() {
            let Some(p) = c.parent(n) else { continue };
            if c.kind(p) != "compound_statement" {
                continue;
            }
            let sibs: Vec<usize> = c.named(p).into_iter().filter(|&s| func::is_stmt(c.kind(s))).collect();
            let Some(i) = sibs.iter().position(|&s| s == n) else { continue };
            let Some(&next) = sibs.get(i + 1) else { continue };
            let (Some(cv), Some(cons)) = (cond_value(c, n), c.child(n, "consequence")) else { continue };
            let inner = if c.kind(cons) == "compound_statement" {
                let v: Vec<usize> = c.named(cons).into_iter().filter(|&s| func::is_stmt(c.kind(s))).collect();
                if v.len() != 1 {
                    continue;
                }
                v[0]
            } else {
                cons
            };
            let rv = |s: usize| -> Option<String> {
                (c.kind(s) == "return_statement").then(|| c.named(s).first().map(|&e| c.text(e).to_string())).flatten()
            };
            let (Some(a), Some(b)) = (rv(inner), rv(next)) else { continue };
            let is_t = |x: &str| x == "true" || x == "1";
            let is_f = |x: &str| x == "false" || x == "0";
            let ct = strip_parens(c.text(cv));
            if is_t(&a) && is_f(&b) {
                let e = if is_condish(c, cv) { ct.to_string() } else { format!("{ct} != 0") };
                cands.push((n, next, format!("return {e};")));
            } else if is_f(&a) && is_t(&b) {
                cands.push((n, next, format!("return !({ct});")));
            }
        }
    }
    let k = m.rng.below(cands.len().max(1));
    let (a, b, txt) = cands.get(k)?.clone();
    Some(vec![Edit { start: c.nodes[a].start, end: c.nodes[b].end, text: txt }])
}

/// `if (a || b) S` (S a jump) <-> `if (a) S if (b) S`; and `if (a && b) S` <-> nested ifs is
/// `nested_if`. Catalog row 21: the `||` guard adds a `bne; b` trampoline.
pub fn op_guard_split(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) != "if_statement" || c.child(n, "alternative").is_some() {
            continue;
        }
        let Some(p) = c.parent(n) else { continue };
        if c.kind(p) != "compound_statement" {
            continue;
        }
        let (Some(cv), Some(cons)) = (cond_value(c, n), c.child(n, "consequence")) else { continue };
        let ind = indent_of(c, n);
        let cv0 = if c.kind(cv) == "parenthesized_expression" { c.named(cv).first().copied().unwrap_or(cv) } else { cv };
        if c.kind(cv0) == "binary_expression" && c.op(cv0) == Some("||") && ends_in_jump(c, cons) {
            let (Some(l), Some(r)) = (c.child(cv0, "left"), c.child(cv0, "right")) else { continue };
            let ct = c.text(cons);
            cands.push((n, n, format!("if ({}) {ct}\n{ind}if ({}) {ct}", strip_parens(c.text(l)), strip_parens(c.text(r)))));
        }
        // merge with the next sibling guard that has the same body
        let sibs: Vec<usize> = c.named(p).into_iter().filter(|&s| func::is_stmt(c.kind(s))).collect();
        let Some(i) = sibs.iter().position(|&s| s == n) else { continue };
        let Some(&nx) = sibs.get(i + 1) else { continue };
        if c.kind(nx) != "if_statement" || c.child(nx, "alternative").is_some() || !ends_in_jump(c, cons) {
            continue;
        }
        let (Some(cv2), Some(cons2)) = (cond_value(c, nx), c.child(nx, "consequence")) else { continue };
        if crate::cst::normalize(c.text(cons)) == crate::cst::normalize(c.text(cons2)) {
            let paren = |e: usize| if matches!(c.kind(e), "binary_expression" | "conditional_expression" | "assignment_expression") { format!("({})", c.text(e)) } else { c.text(e).to_string() };
            cands.push((n, nx, format!("if ({} || {}) {}", paren(cv), paren(cv2), c.text(cons))));
        }
    }
    let k = m.rng.below(cands.len().max(1));
    let (a, b, txt) = cands.get(k)?.clone();
    Some(vec![Edit { start: c.nodes[a].start, end: c.nodes[b].end, text: txt }])
}

/// Loop exit forms: `do { B } while (c);` <-> `while (1) { B if (!c) break; }`, and
/// `while (c) { B }` -> `if (c) { do { B } while (c); }` is `while_to_do`; here also
/// `for (init; c; step) { B }` -> `init; while (c) { B step; }` when B has no continue.
pub fn op_loop_exit(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        let ind = indent_of(c, n);
        match c.kind(n) {
            "do_statement" => {
                let (Some(body), Some(cc)) = (c.child(n, "body"), c.child(n, "condition")) else { continue };
                if c.descendants(body).into_iter().any(|d| c.kind(d) == "continue_statement") {
                    continue;
                }
                let cv = c.named(cc).first().copied().unwrap_or(cc);
                let inner = if c.kind(body) == "compound_statement" { let t = c.text(body); t[1..t.len() - 1].trim().to_string() } else { c.text(body).to_string() };
                cands.push((n, format!("while (1) {{\n{ind}    {inner}\n{ind}    if (!({})) {{\n{ind}        break;\n{ind}    }}\n{ind}}}", strip_parens(c.text(cv)))));
            }
            "while_statement" => {
                // `while (1) { B if (!c) break; }` -> `do { B } while (c);`
                let (Some(cv), Some(body)) = (cond_value(c, n), c.child(n, "body")) else { continue };
                if !matches!(c.text(cv), "1" | "true") || c.kind(body) != "compound_statement" {
                    continue;
                }
                let st: Vec<usize> = c.named(body).into_iter().filter(|&s| func::is_stmt(c.kind(s))).collect();
                let Some(&last) = st.last() else { continue };
                if c.kind(last) != "if_statement" || c.child(last, "alternative").is_some() {
                    continue;
                }
                let (Some(lc), Some(lcons)) = (cond_value(c, last), c.child(last, "consequence")) else { continue };
                let is_break = c.kind(lcons) == "break_statement"
                    || (c.kind(lcons) == "compound_statement" && c.named(lcons).iter().filter(|&&s| func::is_stmt(c.kind(s))).count() == 1 && c.named(lcons).iter().any(|&s| c.kind(s) == "break_statement"));
                if !is_break {
                    continue;
                }
                let rest: Vec<usize> = st[..st.len() - 1].to_vec();
                if rest.iter().any(|&s| own_breaks_any(c, s) || c.descendants(s).into_iter().any(|d| c.kind(d) == "continue_statement")) {
                    continue;
                }
                let neg = negate_text(c, lc);
                let body_t: String = rest.iter().map(|&s| format!("\n{ind}    {}", c.text(s))).collect();
                cands.push((n, format!("do {{{body_t}\n{ind}}} while ({neg});")));
            }
            "for_statement" => {
                let Some(body) = c.child(n, "body") else { continue };
                if c.descendants(body).into_iter().any(|d| c.kind(d) == "continue_statement") {
                    continue;
                }
                let init = c.child(n, "initializer").map(|i| c.text(i).trim_end_matches(';').to_string());
                let cond = c.child(n, "condition").map(|i| c.text(i).to_string()).unwrap_or_else(|| "1".into());
                let step = c.child(n, "update").map(|i| c.text(i).to_string());
                let inner = if c.kind(body) == "compound_statement" { let t = c.text(body); t[1..t.len() - 1].trim().to_string() } else { c.text(body).to_string() };
                let mut t = String::new();
                if let Some(i) = &init {
                    t.push_str(&format!("{i};\n{ind}"));
                }
                t.push_str(&format!("while ({cond}) {{\n{ind}    {inner}"));
                if let Some(s) = &step {
                    t.push_str(&format!("\n{ind}    {s};"));
                }
                t.push_str(&format!("\n{ind}}}"));
                if init.as_deref().is_some_and(|i| i.contains(' ') && !i.contains('=')) {
                    continue;
                }
                cands.push((n, if init.is_some() { format!("{{\n{ind}{t}\n{ind}}}") } else { t }));
            }
            _ => {}
        }
    }
    let (n, t) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, t)])
}

fn negate_text(c: &Cst, e: usize) -> String {
    match c.kind(e) {
        "parenthesized_expression" => c.named(e).first().map(|&i| negate_text(c, i)).unwrap_or_else(|| format!("!{}", c.text(e))),
        "unary_expression" if c.op(e) == Some("!") => strip_parens(c.text(c.child(e, "argument").unwrap())).to_string(),
        _ => format!("!({})", c.text(e)),
    }
}

/// Float literal forms (catalog row 16): `x / K` <-> `x * (1/K)` for powers of two, and the
/// `f` suffix (a double literal adds lfd+frsp and an 8-byte pool entry).
pub fn op_float_literal(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        match c.kind(n) {
            "number_literal" => {
                let t = c.text(n);
                let lt = t.to_ascii_lowercase();
                if lt.starts_with("0x") || !(lt.contains('.') || lt.contains('e')) {
                    continue;
                }
                if let Some(s) = t.strip_suffix(['f', 'F']) {
                    cands.push((n, s.to_string()));
                } else {
                    cands.push((n, format!("{t}f")));
                }
            }
            "binary_expression" if matches!(c.op(n), Some("/" | "*")) => {
                let (Some(l), Some(r)) = (c.child(n, "left"), c.child(n, "right")) else { continue };
                if c.kind(r) != "number_literal" {
                    continue;
                }
                let rt = c.text(r);
                let suf = if rt.ends_with(['f', 'F']) { "f" } else { "" };
                let Ok(v) = rt.trim_end_matches(['f', 'F']).parse::<f64>() else { continue };
                if v == 0.0 || !v.is_finite() {
                    continue;
                }
                let inv = 1.0 / v;
                // exact reciprocal only (power of two)
                if inv.log2().fract() != 0.0 && v.log2().fract() != 0.0 {
                    continue;
                }
                let lit = format_float(inv, suf);
                let op = if c.op(n) == Some("/") { "*" } else { "/" };
                cands.push((n, format!("{} {op} {lit}", c.text(l))));
            }
            _ => {}
        }
    }
    let (n, t) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, t)])
}

fn format_float(v: f64, suf: &str) -> String {
    let mut s = format!("{v}");
    if !s.contains('.') && !s.contains('e') {
        s.push_str(".0");
    }
    format!("{s}{suf}")
}


// ------------------------------------------------------------------ residue-driven operators
// (from the train-split analysis of insertion/deletion residue)

const INT_TYPE_ALTS: &[(&str, &[&str])] = &[
    ("int", &["bool", "unsigned int", "unsigned char", "short", "unsigned short", "float"]),
    ("unsigned int", &["int", "bool", "unsigned char", "unsigned short"]),
    ("bool", &["int", "unsigned char", "unsigned int"]),
    ("unsigned char", &["bool", "int", "signed char", "unsigned int"]),
    ("signed char", &["unsigned char", "int"]),
    ("char", &["unsigned char", "signed char", "int"]),
    ("short", &["unsigned short", "int"]),
    ("unsigned short", &["short", "unsigned int", "int"]),
    ("float", &["double", "int"]),
    ("double", &["float"]),
    ("void", &["int", "bool"]),
];

/// Types of the draft's own declarations outside the function (stand-in virtual slots
/// `struct __mwdec_vt_N { virtual int _5(); }` and `extern int sym;` objects) that the body uses:
/// a virtual returning `int` where the real one returns `bool` adds `neg/or/srwi` at the use.
pub fn op_decl_type(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let body = m.info.body;
    let body_text = c.text(body);
    let mut cands: Vec<(usize, &'static str)> = Vec::new();
    for n in 0..c.nodes.len() {
        let k = c.kind(n);
        if k != "field_declaration" && k != "declaration" {
            continue;
        }
        if c.contains(m.info.def, n) {
            continue;
        }
        let Some(t) = c.child(n, "type") else { continue };
        let Some(d) = c.child(n, "declarator") else { continue };
        let name = match c.kind(d) {
            "function_declarator" => c.child(d, "declarator").map(|x| c.text(x).to_string()),
            "identifier" | "array_declarator" => func::declarator_name(c, d).map(|x| x.0),
            _ => None,
        };
        let Some(name) = name else { continue };
        if k == "field_declaration" {
            // only stand-in polymorphic structs
            let in_vt = c
                .ancestors(n)
                .into_iter()
                .any(|a| c.kind(a) == "struct_specifier" && c.child(a, "name").is_some_and(|nm| c.text(nm).starts_with("__mwdec_vt_")));
            if !in_vt || c.kind(d) != "function_declarator" {
                continue;
            }
            if !body_text.contains(&format!("->{name}(")) && !body_text.contains(&format!(".{name}(")) {
                continue;
            }
        } else {
            if !c.text(n).starts_with("extern ") || c.kind(d) == "function_declarator" {
                continue;
            }
            if !mentions_word(body_text, &name) {
                continue;
            }
        }
        let tt = c.text(t).trim();
        if let Some((_, alts)) = INT_TYPE_ALTS.iter().find(|(x, _)| *x == tt) {
            for a in alts.iter() {
                if c.kind(d) == "array_declarator" && *a == "bool" {
                    continue;
                }
                cands.push((t, a));
            }
        }
    }
    let (t, a) = m.pick_one(&cands)?;
    let mut edits = vec![Edit::replace(c, t, a)];
    // A slot made bool in a bool function: also make returned 0/1 ternary arms bool (otherwise
    // the int-typed ternary still converts), as one step half of the time.
    if a == "bool" && m.info.ret_ty.trim() == "bool" && m.rng.chance(0.5) {
        for n in m.nodes.clone() {
            if c.kind(n) != "number_literal" || !matches!(c.text(n), "0" | "1") {
                continue;
            }
            if c.parent(n).is_some_and(|p| c.kind(p) == "conditional_expression" && c.nodes[n].field != Some("condition")) {
                edits.push(Edit::replace(c, n, if c.text(n) == "1" { "true" } else { "false" }));
            }
        }
    }
    Some(edits)
}

fn mentions_word(text: &str, w: &str) -> bool {
    let b = text.as_bytes();
    let word = |x: u8| x.is_ascii_alphanumeric() || x == b'_';
    let mut from = 0;
    while let Some(i) = text[from..].find(w) {
        let s = from + i;
        let e = s + w.len();
        let ok_l = s == 0 || !word(b[s - 1]);
        let ok_r = e >= b.len() || !word(b[e]);
        if ok_l && ok_r {
            return true;
        }
        from = e;
    }
    false
}

fn single_assign(c: &Cst, s: usize) -> Option<(String, String)> {
    let s = if c.kind(s) == "compound_statement" {
        let v: Vec<usize> = c.named(s).into_iter().filter(|&x| func::is_stmt(c.kind(x))).collect();
        if v.len() != 1 {
            return None;
        }
        v[0]
    } else {
        s
    };
    if c.kind(s) != "expression_statement" {
        return None;
    }
    let a = *c.named(s).first()?;
    if c.kind(a) != "assignment_expression" || c.op(a) != Some("=") {
        return None;
    }
    Some((c.text(c.child(a, "left")?).to_string(), c.text(c.child(a, "right")?).to_string()))
}

fn is_empty_block(c: &Cst, s: usize) -> bool {
    c.kind(s) == "compound_statement" && !c.named(s).into_iter().any(|x| func::is_stmt(c.kind(x)))
}

/// Value-select forms (min/max idioms): `if (c) {} else { v = e; }` -> `v = c ? v : e;`,
/// `if (c) { v = e; }` -> `v = c ? e : v;`, and an empty then-arm -> negated condition. A float
/// `x = a < b ? a : b` compiles to `bge; b; fmr` where the if-form gives `blt; fmr`.
pub fn op_ternary_self(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) != "if_statement" {
            continue;
        }
        let (Some(cv), Some(cons)) = (cond_value(c, n), c.child(n, "consequence")) else { continue };
        let ct = c.text(cv);
        let alt = c.child(n, "alternative").and_then(|a| c.named(a).into_iter().find(|&s| func::is_stmt(c.kind(s))));
        match alt {
            Some(a) if is_empty_block(c, cons) => {
                if let Some((v, e)) = single_assign(c, a) {
                    cands.push((n, format!("{v} = ({ct}) ? {v} : {e};")));
                }
                cands.push((n, format!("if ({}) {}", negate_text(c, cv), c.text(a))));
            }
            Some(a) if is_empty_block(c, a) => {
                cands.push((n, format!("if ({ct}) {}", c.text(cons))));
            }
            None => {
                if let Some((v, e)) = single_assign(c, cons) {
                    cands.push((n, format!("{v} = ({ct}) ? {e} : {v};")));
                }
            }
            _ => {}
        }
    }
    let (n, t) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, t)])
}

fn num_val(c: &Cst, n: usize) -> Option<f64> {
    let n = if c.kind(n) == "parenthesized_expression" { *c.named(n).first()? } else { n };
    if c.kind(n) != "number_literal" {
        return None;
    }
    c.text(n).trim_end_matches(['f', 'F']).parse::<f64>().ok()
}

/// Int->float conversion magic left in the draft (`(double)(x ^ 0x80000000) + 2^52 - (2^52+2^31)`)
/// -> `(float)x` (unsigned form: `(double)x + 2^52 - 2^52` -> `(float)(unsigned int)x`).
pub fn op_fold_magic(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    const B: f64 = 4503599627370496.0;
    const BS: f64 = 4503601774854144.0;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) != "binary_expression" || c.op(n) != Some("-") {
            continue;
        }
        let (Some(l), Some(r)) = (c.child(n, "left"), c.child(n, "right")) else { continue };
        let Some(rv) = num_val(c, r) else { continue };
        let l = if c.kind(l) == "parenthesized_expression" { c.named(l).first().copied().unwrap_or(l) } else { l };
        if c.kind(l) != "binary_expression" || c.op(l) != Some("+") {
            continue;
        }
        let (Some(a), Some(b)) = (c.child(l, "left"), c.child(l, "right")) else { continue };
        let mut x = if num_val(c, b) == Some(B) {
            a
        } else if num_val(c, a) == Some(B) {
            b
        } else {
            continue;
        };
        // strip parentheses and (double)
        loop {
            match c.kind(x) {
                "parenthesized_expression" => match c.named(x).first() {
                    Some(&i) => x = i,
                    None => break,
                },
                "cast_expression" if c.child(x, "type").is_some_and(|t| c.text(t) == "double") => match c.child(x, "value") {
                    Some(v) => x = v,
                    None => break,
                },
                _ => break,
            }
        }
        if rv == BS {
            if c.kind(x) == "binary_expression" && c.op(x) == Some("^") {
                let (Some(xl), Some(xr)) = (c.child(x, "left"), c.child(x, "right")) else { continue };
                if c.text(xr).to_ascii_lowercase().trim_end_matches('u') == "0x80000000" {
                    cands.push((n, format!("(float)({})", c.text(xl))));
                    cands.push((n, format!("(double)({})", c.text(xl))));
                }
            }
        } else if rv == B {
            cands.push((n, format!("(float)(unsigned int)({})", c.text(x))));
            cands.push((n, format!("(double)(unsigned int)({})", c.text(x))));
        }
    }
    let (n, t) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, t)])
}



// ------------------------------------------------------------------ address forms

fn scalar_size(t: &str) -> Option<i64> {
    let t = t.trim().trim_start_matches("const ").trim();
    if t.ends_with('*') {
        return Some(4);
    }
    Some(match t {
        "float" | "int" | "unsigned int" | "s32" | "u32" | "long" | "unsigned long" | "f32" | "uint" => 4,
        "short" | "unsigned short" | "s16" | "u16" | "ushort" => 2,
        "char" | "unsigned char" | "signed char" | "s8" | "u8" | "bool" | "uchar" => 1,
        "double" | "f64" | "long long" | "unsigned long long" | "s64" | "u64" => 8,
        _ => return None,
    })
}

fn strip_byte_casts(c: &Cst, mut n: usize) -> usize {
    loop {
        match c.kind(n) {
            "parenthesized_expression" => match c.named(n).first() {
                Some(&i) => n = i,
                None => return n,
            },
            "cast_expression" => {
                let t = c.child(n, "type").map(|t| c.text(t).replace(' ', "")).unwrap_or_default();
                let d = c.child(n, "type").and_then(|_| {
                    // `(char*)x`: the abstract declarator holds the `*`
                    let full = c.text(n);
                    let close = full.find(')')?;
                    Some(full[1..close].replace(' ', ""))
                });
                let ty = d.unwrap_or(t);
                if matches!(ty.as_str(), "char*" | "unsignedchar*" | "u8*" | "signedchar*" | "s8*") {
                    match c.child(n, "value") {
                        Some(v) => n = v,
                        None => return n,
                    }
                } else {
                    return n;
                }
            }
            _ => return n,
        }
    }
}

/// Is `n` byte-pointer arithmetic: a `(char*)`-like cast, or a sum whose pointer operand is one?
fn byte_chain(c: &Cst, n: usize) -> bool {
    let mut n = n;
    while c.kind(n) == "parenthesized_expression" {
        match c.named(n).first() {
            Some(&i) => n = i,
            None => return false,
        }
    }
    if strip_byte_casts(c, n) != n {
        return true;
    }
    if c.kind(n) == "binary_expression" && c.op(n) == Some("+") {
        let (Some(l), Some(r)) = (c.child(n, "left"), c.child(n, "right")) else { return false };
        return byte_chain(c, l) || byte_chain(c, r);
    }
    false
}

/// Additive terms of a byte-pointer expression (through `(char*)` casts and parentheses).
fn add_terms(c: &Cst, n: usize, out: &mut Vec<usize>) {
    let mut n = n;
    while c.kind(n) == "parenthesized_expression" {
        match c.named(n).first() {
            Some(&i) => n = i,
            None => break,
        }
    }
    let inner = strip_byte_casts(c, n);
    if inner != n {
        // `(char*)E`: E's own `+` is byte arithmetic only if it is a byte chain itself.
        if c.kind(inner) == "binary_expression" && c.op(inner) == Some("+") && byte_chain(c, inner) {
            n = inner;
        } else {
            out.push(n);
            return;
        }
    }
    if c.kind(n) == "binary_expression" && c.op(n) == Some("+") {
        if let (Some(l), Some(r)) = (c.child(n, "left"), c.child(n, "right")) {
            add_terms(c, l, out);
            add_terms(c, r, out);
            return;
        }
    }
    out.push(n);
}

/// Element index of a term scaled by `size` (`i << 2`, `i * 4`), as text.
fn index_of(c: &Cst, n: usize, size: i64) -> Option<String> {
    let n = if c.kind(n) == "parenthesized_expression" { *c.named(n).first()? } else { n };
    if size == 1 && !matches!(c.kind(n), "number_literal") {
        return None;
    }
    if c.kind(n) != "binary_expression" {
        return None;
    }
    let (l, r) = (c.child(n, "left")?, c.child(n, "right")?);
    let k = int_lit(c, r)?;
    match c.op(n)? {
        "<<" if (1i64 << k) == size => Some(c.text(l).to_string()),
        "*" if k == size => Some(c.text(l).to_string()),
        _ => None,
    }
}

/// Address arithmetic forms of `*(T*)((char*)base + off + (i << k))`: array indexing
/// `((T*)((char*)base + off))[i]` and the other groupings of the byte offsets (they decide
/// whether the constant folds into the load displacement or into an `addi`).
pub fn op_addr_form(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) != "pointer_expression" || c.op(n) != Some("*") {
            continue;
        }
        let Some(arg) = c.child(n, "argument") else { continue };
        let arg = if c.kind(arg) == "parenthesized_expression" { c.named(arg).first().copied().unwrap_or(arg) } else { arg };
        if c.kind(arg) != "cast_expression" {
            continue;
        }
        let full = c.text(arg);
        let Some(close) = full.find(')') else { continue };
        let pty = full[1..close].trim().to_string();
        let Some(elem) = pty.strip_suffix('*') else { continue };
        let Some(size) = scalar_size(elem) else { continue };
        let Some(v) = c.child(arg, "value") else { continue };
        // Only byte arithmetic: typed pointer `+` scales its offset.
        if !byte_chain(c, v) {
            continue;
        }
        let mut terms = Vec::new();
        add_terms(c, v, &mut terms);
        if terms.len() < 2 {
            continue;
        }
        // The base pointer: the first term that is neither a literal nor a scaled index.
        let Some(bi) = terms.iter().position(|&t| {
            let t0 = strip_any_casts(c, t);
            !(c.kind(t0) == "number_literal" || int_lit(c, t0).is_some() || (c.kind(t0) == "binary_expression" && matches!(c.op(t0), Some("<<" | "*"))))
        }) else {
            continue;
        };
        let base = terms[bi];
        let rest: Vec<usize> = terms.iter().enumerate().filter(|&(k, _)| k != bi).map(|(_, &t)| t).collect();
        let bt = c.text(strip_byte_casts(c, base));
        let tt = |x: usize| -> String {
            if matches!(c.kind(x), "binary_expression" | "conditional_expression") {
                format!("({})", c.text(x))
            } else {
                c.text(x).to_string()
            }
        };
        // array form
        for (k, &t) in rest.iter().enumerate() {
            if let Some(ix) = index_of(c, t, size) {
                let others: Vec<String> = rest.iter().enumerate().filter(|&(j, _)| j != k).map(|(_, &x)| tt(x)).collect();
                let basep = if others.is_empty() {
                    format!("(({pty})({bt}))")
                } else {
                    format!("(({pty})((char*)({bt}) + {}))", others.join(" + "))
                };
                cands.push((n, format!("{basep}[{ix}]")));
            }
        }
        // regroupings / orderings of the offsets
        if rest.len() >= 2 {
            let rt: Vec<String> = rest.iter().map(|&x| tt(x)).collect();
            let mut rev = rt.clone();
            rev.reverse();
            cands.push((n, format!("*({pty})((char*)({bt}) + {})", rev.join(" + "))));
            cands.push((n, format!("*({pty})((char*)({bt}) + ({}))", rt.join(" + "))));
            cands.push((n, format!("*({pty})((char*)((char*)({bt}) + {}) + {})", rt[1..].join(" + "), rt[0])));
        } else {
            cands.push((n, format!("*({pty})({} + (char*)({bt}))", tt(rest[0]))));
        }
    }
    let (n, t) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, t)])
}

/// In a bool function, integer literals `0`/`1` returned (directly or as ternary arms) -> `false`
/// /`true` and back: the ternary's type decides whether a bool conversion (`neg/or/srwi`) follows.
pub fn op_bool_literal(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    if m.info.ret_ty.trim() != "bool" {
        return None;
    }
    let mut cands: Vec<(usize, &'static str)> = Vec::new();
    for n in m.nodes.clone() {
        let lit = match c.text(n) {
            "0" if c.kind(n) == "number_literal" => "false",
            "1" if c.kind(n) == "number_literal" => "true",
            "false" => "0",
            "true" => "1",
            _ => continue,
        };
        let mut p = c.parent(n);
        while let Some(x) = p {
            if c.kind(x) == "parenthesized_expression" {
                p = c.parent(x);
            } else {
                break;
            }
        }
        let Some(p) = p else { continue };
        let ok = match c.kind(p) {
            "return_statement" => true,
            "conditional_expression" => c.nodes[n].field != Some("condition"),
            _ => false,
        };
        if ok {
            cands.push((n, lit));
        }
    }
    let (n, t) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, t)])
}


// ------------------------------------------------------------------ induction variables

/// `x = x + 1;` / `x += 1;` / `x++;` / `++x;` statement incrementing identifier `x` by `k`.
fn incr_of(c: &Cst, s: usize) -> Option<(String, String)> {
    if c.kind(s) != "expression_statement" {
        return None;
    }
    let e = *c.named(s).first()?;
    match c.kind(e) {
        "update_expression" => {
            let a = c.child(e, "argument")?;
            (c.kind(a) == "identifier" && c.text(e).contains("++")).then(|| (c.text(a).to_string(), "1".to_string()))
        }
        "assignment_expression" => {
            let l = c.child(e, "left")?;
            if c.kind(l) != "identifier" {
                return None;
            }
            let name = c.text(l).to_string();
            let r = c.child(e, "right")?;
            match c.op(e)? {
                "+=" => Some((name, c.text(r).to_string())),
                "=" => {
                    // x = x + k, x = (T)((char*)x + k)
                    let mut terms = Vec::new();
                    let r0 = strip_any_casts(c, r);
                    add_terms(c, r0, &mut terms);
                    if terms.len() == 2 && strip_any_casts(c, terms[0]) != terms[0] || terms.len() == 2 && c.kind(terms[0]) == "identifier" {
                        let b = strip_any_casts(c, terms[0]);
                        if c.text(b) == name {
                            return Some((name, c.text(terms[1]).to_string()));
                        }
                    }
                    None
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn strip_any_casts(c: &Cst, mut n: usize) -> usize {
    loop {
        match c.kind(n) {
            "parenthesized_expression" => match c.named(n).first() {
                Some(&i) => n = i,
                None => return n,
            },
            "cast_expression" => match c.child(n, "value") {
                Some(v) => n = v,
                None => return n,
            },
            _ => return n,
        }
    }
}

/// Strength-reduced pointer loops back to indexing: a pointer `p` set to `B` before the loop and
/// stepped by `K` bytes next to a counter `i` (0, +1) -> uses `*(T*)p` become `((T*)(B))[i]` (or
/// `(char*)(B) + i * K`), the pointer and its step disappear. MWCC strength-reduces `a[i]` itself.
pub fn op_index_loop(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<Vec<Edit>> = Vec::new();
    for lp in m.nodes.clone() {
        if !matches!(c.kind(lp), "do_statement" | "while_statement" | "for_statement") {
            continue;
        }
        let Some(body) = c.child(lp, "body") else { continue };
        if c.kind(body) != "compound_statement" {
            continue;
        }
        let stmts: Vec<usize> = c.named(body).into_iter().filter(|&s| func::is_stmt(c.kind(s))).collect();
        let incs: Vec<(usize, String, String)> = stmts.iter().filter_map(|&s| incr_of(c, s).map(|(n, k)| (s, n, k))).collect();
        let Some((_, ivar, _)) = incs.iter().find(|x| x.2 == "1" && m.info.vars.get(&x.1).is_some_and(|v| crate::func::is_int_type(&v.ty))) else { continue };
        for (ps, pvar, k) in incs.iter().filter(|x| m.info.vars.get(&x.1).is_some_and(|v| v.ty.ends_with('*'))) {
            if pvar == ivar {
                continue;
            }
            let kt = k.trim().trim_end_matches(['u', 'U']).to_ascii_lowercase();
            let Some(kv) = kt.parse::<i64>().ok().or_else(|| kt.strip_prefix("0x").and_then(|h| i64::from_str_radix(h, 16).ok())) else { continue };
            // the initialisation `p = B;` right before the loop (same block, earlier sibling)
            let Some(par) = c.parent(lp) else { continue };
            let sibs: Vec<usize> = c.named(par).into_iter().filter(|&s| func::is_stmt(c.kind(s))).collect();
            let Some(li) = sibs.iter().position(|&s| s == lp) else { continue };
            let mut init: Option<(usize, String)> = None;
            for &s in sibs[..li].iter().rev() {
                if let Some((l, r)) = single_assign(c, s) {
                    if l == *pvar {
                        init = Some((s, r));
                        break;
                    }
                }
                if crate::ops::mentions(c, s, pvar) {
                    break;
                }
            }
            let Some((is, bexpr)) = init else { continue };
            // every other use of p: inside the loop body, before the increments
            let uses = crate::ops::uses_of(c, m.info.body, pvar);
            let ps_end = c.nodes[*ps].end;
            let bad = uses.iter().any(|&u| {
                let inside = c.contains(body, u);
                let in_inc = c.contains(*ps, u);
                let in_init = c.contains(is, u);
                !(in_init || in_inc || (inside && c.nodes[u].start < ps_end && incs.iter().all(|x| c.nodes[u].start < c.nodes[x.0].start || c.contains(x.0, u))))
            });
            if bad {
                continue;
            }
            let mut edits = vec![remove_stmt_edit(c, *ps), remove_stmt_edit(c, is)];
            for &u in &uses {
                if c.contains(*ps, u) || c.contains(is, u) {
                    continue;
                }
                // `*(T*)p` -> `((T*)(B))[i]` when sizeof(T) == K
                let mut top = u;
                let mut cast_ty: Option<String> = None;
                if let Some(p1) = c.parent(u) {
                    if c.kind(p1) == "cast_expression" {
                        let full = c.text(p1);
                        if let Some(close) = full.find(')') {
                            cast_ty = Some(full[1..close].trim().to_string());
                        }
                        top = p1;
                    }
                }
                let deref = c.parent(top).filter(|&p2| c.kind(p2) == "pointer_expression" && c.op(p2) == Some("*"));
                match (deref, &cast_ty) {
                    (Some(d), Some(t)) if t.strip_suffix('*').and_then(scalar_size) == Some(kv) => {
                        edits.push(Edit::replace(c, d, format!("(({t})({bexpr}))[{ivar}]")));
                    }
                    _ => {
                        let vt = m.info.vars.get(pvar).map(|v| v.ty.clone()).unwrap_or_else(|| "char*".into());
                        edits.push(Edit::replace(c, u, format!("(({vt})((char*)({bexpr}) + {ivar} * {kv}))")));
                    }
                }
            }
            cands.push(edits);
        }
    }
    let k = m.rng.below(cands.len().max(1));
    cands.into_iter().nth(k)
}

fn remove_stmt_edit(c: &Cst, s: usize) -> Edit {
    let (st, en) = (c.nodes[s].start, c.nodes[s].end);
    let ls = c.src[..st].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let le = c.src[en..].find('\n').map(|i| en + i + 1).unwrap_or(en);
    if c.src[ls..st].trim().is_empty() && c.src[en..le].trim().is_empty() {
        Edit { start: ls, end: le, text: String::new() }
    } else {
        Edit { start: st, end: en, text: String::new() }
    }
}


// ------------------------------------------------------------------ stack copies

/// A stack object filled by one reinterpreting store and used once by value
/// (`T s; *(int*)&s = E; f(s);`) -> `f(*(T*)&E)` when `E` is an lvalue: removes the extra frame
/// copy (`stw r0,0x10(r1)` + the argument copy). Also `T s = E2; f(s);` -> `f(E2)` for class types.
pub fn op_forward_stack(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<Vec<Edit>> = Vec::new();
    for (name, var) in m.info.vars.iter() {
        if var.is_param || func::is_scalar_type(&var.ty) || var.ty.contains('&') || var.ty.contains('[') {
            continue;
        }
        let decl = var.decl;
        let ds: Vec<usize> = c.children_by_field(decl, "declarator").collect();
        if ds.len() != 1 {
            continue;
        }
        let uses = crate::ops::uses_of(c, m.info.body, name);
        // `*(X*)&s = E;` stores and value uses
        let mut stores = Vec::new();
        let mut others = Vec::new();
        for &u in &uses {
            let Some(p) = c.parent(u) else { continue };
            if c.kind(p) == "pointer_expression" && c.op(p) == Some("&") {
                // &s inside a cast inside a deref on an assignment's left
                let mut a = p;
                while let Some(q) = c.parent(a) {
                    if matches!(c.kind(q), "cast_expression" | "parenthesized_expression") {
                        a = q;
                    } else {
                        break;
                    }
                }
                let st = c.parent(a).filter(|&q| c.kind(q) == "pointer_expression" && c.op(q) == Some("*"));
                let asg = st.and_then(|q| c.parent(q)).filter(|&q| c.kind(q) == "assignment_expression" && c.op(q) == Some("=") && c.child(q, "left") == st);
                let stmt = asg.and_then(|q| c.parent(q)).filter(|&q| c.kind(q) == "expression_statement");
                match (asg, stmt) {
                    (Some(a), Some(s)) => stores.push((s, c.child(a, "right").unwrap())),
                    _ => others.push(u),
                }
            } else {
                others.push(u);
            }
        }
        let ty = c.child(decl, "type").map(|t| c.text(t).to_string()).unwrap_or_default();
        if ty.is_empty() {
            continue;
        }
        if stores.len() == 1 && others.len() == 1 {
            let (st, rhs) = stores[0];
            let u = others[0];
            if !c.parent(u).is_some_and(|p| c.kind(p) == "argument_list") || c.nodes[u].start < c.nodes[st].end {
                continue;
            }
            let r = strip_any_casts(c, rhs);
            let lv = matches!(c.kind(r), "pointer_expression" | "field_expression" | "subscript_expression") && (c.kind(r) != "pointer_expression" || c.op(r) == Some("*"));
            if !lv {
                continue;
            }
            let mut e = vec![remove_stmt_edit(c, st), Edit::replace(c, u, format!("*({ty}*)&{}", if c.kind(r) == "field_expression" || c.kind(r) == "subscript_expression" { c.text(r).to_string() } else { format!("({})", c.text(r)) }))];
            if c.kind(ds[0]) != "init_declarator" {
                e.push(remove_stmt_edit(c, decl));
            } else {
                continue;
            }
            cands.push(e);
        } else if stores.is_empty() && others.len() == 1 && c.kind(ds[0]) == "init_declarator" {
            let u = others[0];
            if !c.parent(u).is_some_and(|p| c.kind(p) == "argument_list") {
                continue;
            }
            let Some(v) = c.child(ds[0], "value") else { continue };
            let vt = if c.kind(v) == "argument_list" { format!("{ty}{}", c.text(v)) } else { c.text(v).to_string() };
            cands.push(vec![remove_stmt_edit(c, decl), Edit::replace(c, u, vt)]);
        }
    }
    let k = m.rng.below(cands.len().max(1));
    cands.into_iter().nth(k)
}


// ------------------------------------------------------------------ destructor calls

/// `p->~T();` <-> `delete p;` (the deleting destructor gets flag 1 instead of -1/0), and
/// `if (p) { delete p; }` -> `delete p;` (delete tests for null itself).
pub fn op_dtor_delete(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        match c.kind(n) {
            "expression_statement" => {
                let Some(&e) = c.named(n).first() else { continue };
                if c.kind(e) != "call_expression" || c.child(e, "arguments").is_some_and(|a| !c.named(a).is_empty()) {
                    continue;
                }
                let Some(f) = c.child(e, "function") else { continue };
                if c.kind(f) != "field_expression" || c.op(f) != Some("->") {
                    continue;
                }
                let Some(fld) = c.child(f, "field") else { continue };
                if !c.text(fld).starts_with('~') {
                    continue;
                }
                let Some(obj) = c.child(f, "argument") else { continue };
                cands.push((n, format!("delete {};", c.text(obj))));
            }
            "delete_expression" => {
                let Some(st) = c.parent(n).filter(|&p| c.kind(p) == "expression_statement") else { continue };
                // guard removal
                let mut g = c.parent(st);
                if g.is_some_and(|x| c.kind(x) == "compound_statement" && c.named(x).iter().filter(|&&s| func::is_stmt(c.kind(s))).count() == 1) {
                    g = g.and_then(|x| c.parent(x));
                }
                if let Some(ifs) = g.filter(|&x| c.kind(x) == "if_statement" && c.child(x, "alternative").is_none()) {
                    let arg = c.named(n).last().map(|&a| c.text(a).to_string()).unwrap_or_default();
                    if cond_value(c, ifs).is_some_and(|v| strip_parens(c.text(v)) == arg || strip_parens(c.text(v)) == format!("{arg} != 0")) {
                        cands.push((ifs, c.text(st).to_string()));
                    }
                }
            }
            _ => {}
        }
    }
    let (n, t) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, t)])
}


// ------------------------------------------------------------------ split/merge updates

/// `v = S op B;` where `S` uses `v` -> `v = S; v = v op B;` (catalog row 11: the intermediate is
/// written back to v's register: `clrrwi r30,r30,1; srawi r30,r30,1` instead of a temp), and the
/// reverse merge of two consecutive updates of the same local.
pub fn op_split_update(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, usize, String)> = Vec::new();
    for s in m.nodes.clone() {
        if c.kind(s) != "expression_statement" {
            continue;
        }
        let Some(&a) = c.named(s).first() else { continue };
        if c.kind(a) != "assignment_expression" {
            continue;
        }
        let (Some(l), Some(r)) = (c.child(a, "left"), c.child(a, "right")) else { continue };
        if c.kind(l) != "identifier" || !m.info.is_private(c.text(l)) {
            continue;
        }
        let v = c.text(l);
        let ind = indent_of(c, s);
        if c.op(a) == Some("=") {
            let r0 = if c.kind(r) == "parenthesized_expression" { c.named(r).first().copied().unwrap_or(r) } else { r };
            if c.kind(r0) == "binary_expression" {
                let (Some(x), Some(y)) = (c.child(r0, "left"), c.child(r0, "right")) else { continue };
                let op = c.op(r0).unwrap_or("+");
                if matches!(op, "&&" | "||" | "==" | "!=" | "<" | ">" | "<=" | ">=") {
                    continue;
                }
                let xs = strip_any_casts(c, x);
                if c.kind(xs) != "identifier" && crate::ops::mentions(c, x, v) {
                    cands.push((s, s, format!("{v} = {};\n{ind}{v} = {v} {op} {};", c.text(x), c.text(y))));
                }
                let ys = strip_any_casts(c, y);
                if c.kind(ys) != "identifier" && crate::ops::mentions(c, y, v) && matches!(op, "+" | "*" | "&" | "|" | "^") {
                    cands.push((s, s, format!("{v} = {};\n{ind}{v} = {} {op} {v};", c.text(y), c.text(x))));
                }
            }
        }
        // merge with the next statement `v = <expr using v once>;`
        let Some(p) = c.parent(s).filter(|&p| c.kind(p) == "compound_statement") else { continue };
        let sibs: Vec<usize> = c.named(p).into_iter().filter(|&x| func::is_stmt(c.kind(x))).collect();
        let Some(i) = sibs.iter().position(|&x| x == s) else { continue };
        let Some(&nx) = sibs.get(i + 1) else { continue };
        let Some((l2, r2)) = single_assign(c, nx) else { continue };
        if l2 != v || c.op(a) != Some("=") {
            continue;
        }
        let Some(&a2) = c.named(nx).first() else { continue };
        let Some(r2n) = c.child(a2, "right") else { continue };
        let us = crate::ops::uses_of(c, r2n, v);
        if us.len() != 1 {
            continue;
        }
        let u = us[0];
        let rs = c.nodes[r2n].start;
        let rel = (c.nodes[u].start - rs, c.nodes[u].end - rs);
        let mut t = r2.clone();
        t.replace_range(rel.0..rel.1, &format!("({})", c.text(r)));
        cands.push((s, nx, format!("{v} = {t};")));
    }
    let k = m.rng.below(cands.len().max(1));
    let (a, b, t) = cands.get(k)?.clone();
    Some(vec![Edit { start: c.nodes[a].start, end: c.nodes[b].end, text: t }])
}


// ------------------------------------------------------------------ vector component code

/// `recv.SetC(e);` with C in X/Y/Z: (receiver text, component index, statement, argument).
fn set_component(c: &Cst, s: usize) -> Option<(String, usize, usize)> {
    if c.kind(s) != "expression_statement" {
        return None;
    }
    let e = *c.named(s).first()?;
    if c.kind(e) != "call_expression" {
        return None;
    }
    let f = c.child(e, "function")?;
    if c.kind(f) != "field_expression" || c.op(f) != Some(".") {
        return None;
    }
    let comp = match c.text(c.child(f, "field")?) {
        "SetX" => 0,
        "SetY" => 1,
        "SetZ" => 2,
        _ => return None,
    };
    let args = c.named(c.child(e, "arguments")?);
    if args.len() != 1 {
        return None;
    }
    Some((c.text(c.child(f, "argument")?).to_string(), comp, args[0]))
}

/// `A.GetC()` -> (A text, component).
fn get_component(c: &Cst, n: usize) -> Option<(String, usize)> {
    let n = if c.kind(n) == "parenthesized_expression" { *c.named(n).first()? } else { n };
    if c.kind(n) != "call_expression" || c.child(n, "arguments").is_some_and(|a| !c.named(a).is_empty()) {
        return None;
    }
    let f = c.child(n, "function")?;
    if c.kind(f) != "field_expression" || c.op(f) != Some(".") {
        return None;
    }
    let comp = match c.text(c.child(f, "field")?) {
        "GetX" => 0,
        "GetY" => 1,
        "GetZ" => 2,
        _ => return None,
    };
    Some((c.text(c.child(f, "argument")?).to_string(), comp))
}

/// Shape of one component expression: `A.GetC() op B.GetC()`, `s * A.GetC()`, `A.GetC() * s`,
/// or `A.GetC()` (operands may be single-use temps, resolved through `res`, whose definitions are
/// appended to `used`).
fn comp_shape(c: &Cst, n: usize, comp: usize, res: &dyn Fn(usize) -> Option<(usize, usize)>, used: &mut Vec<usize>) -> Option<String> {
    let mut n = if c.kind(n) == "parenthesized_expression" { *c.named(n).first()? } else { n };
    if let Some((v, d)) = res(n) {
        used.push(d);
        n = v;
    }
    if let Some((a, k)) = get_component(c, n) {
        return (k == comp).then(|| a);
    }
    if c.kind(n) != "binary_expression" {
        return None;
    }
    let op = c.op(n)?;
    if !matches!(op, "+" | "-" | "*") {
        return None;
    }
    let (mut l, mut r) = (c.child(n, "left")?, c.child(n, "right")?);
    if let Some((v, d)) = res(l) {
        if get_component(c, v).is_some() {
            used.push(d);
            l = v;
        }
    }
    if let Some((v, d)) = res(r) {
        if get_component(c, v).is_some() {
            used.push(d);
            r = v;
        }
    }
    match (get_component(c, l), get_component(c, r)) {
        (Some((a, ka)), Some((b, kb))) if ka == comp && kb == comp => Some(format!("{a} {op} {b}")),
        (Some((a, ka)), None) if ka == comp && op == "*" && c.kind(r) != "call_expression" => Some(format!("{a} * {}", c.text(r))),
        (None, Some((b, kb))) if kb == comp && op == "*" && c.kind(l) != "call_expression" => Some(format!("{b} * {}", c.text(l))),
        _ => None,
    }
}

/// Component-wise vector code back to vector operators: three `D.SetX/Y/Z(...)` whose arguments
/// (after inlining single-use temps) are `A.GetC() op B.GetC()` -> `D = A op B;` (or `D = A;`).
pub fn op_vec_op(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<Vec<Edit>> = Vec::new();
    for (_, l) in m.lists(3) {
        // temp definitions in this block: name -> (stmt, value node)
        let mut defs: std::collections::HashMap<String, (usize, usize)> = std::collections::HashMap::new();
        for &s in &l {
            if c.kind(s) == "expression_statement" {
                if let Some(&a) = c.named(s).first() {
                    if c.kind(a) == "assignment_expression" && c.op(a) == Some("=") {
                        if let (Some(lh), Some(rh)) = (c.child(a, "left"), c.child(a, "right")) {
                            if c.kind(lh) == "identifier" {
                                defs.insert(c.text(lh).to_string(), (s, rh));
                            }
                        }
                    }
                }
            }
        }
        let sets: Vec<(usize, String, usize, usize)> = l.iter().filter_map(|&s| set_component(c, s).map(|(r, k, a)| (s, r, k, a))).collect();
        let mut recvs: Vec<String> = sets.iter().map(|x| x.1.clone()).collect();
        recvs.dedup();
        for recv in recvs {
            let group: Vec<&(usize, String, usize, usize)> = sets.iter().filter(|x| x.1 == recv).collect();
            let mut by: [Option<&(usize, String, usize, usize)>; 3] = [None, None, None];
            for g in &group {
                if by[g.2].is_none() {
                    by[g.2] = Some(g);
                }
            }
            let Some(parts) = by.iter().copied().collect::<Option<Vec<_>>>() else { continue };
            let mut shapes = Vec::new();
            let mut removed: Vec<usize> = parts.iter().map(|p| p.0).collect();
            let mut ok = true;
            let res = |n: usize| -> Option<(usize, usize)> {
                if c.kind(n) != "identifier" {
                    return None;
                }
                let name = c.text(n);
                let &(ds, rv) = defs.get(name)?;
                (crate::ops::uses_of(c, m.info.body, name).len() == 2).then_some((rv, ds))
            };
            for p in &parts {
                match comp_shape(c, p.3, p.2, &res, &mut removed) {
                    Some(sh) => shapes.push(sh),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || shapes.iter().any(|s| *s != shapes[0]) {
                continue;
            }
            removed.sort();
            removed.dedup();
            let first = *removed.iter().min_by_key(|&&s| c.nodes[s].start).unwrap();
            let first_set = parts.iter().map(|p| p.0).min_by_key(|&s| c.nodes[s].start).unwrap();
            for at in [first_set, first] {
                let mut e: Vec<Edit> = removed.iter().filter(|&&s| s != at).map(|&s| remove_stmt_edit(c, s)).collect();
                e.push(Edit::replace(c, at, format!("{recv} = {};", shapes[0])));
                cands.push(e);
            }
        }
    }
    let k = m.rng.below(cands.len().max(1));
    cands.into_iter().nth(k)
}

