//! Operators for near misses left by the first search rounds (TRAIN-split analysis, eval
//! train s7): argument order of symmetric helpers, single-exit result variables, and dropping a
//! statement whose code the target doesn't have (an explicit store an implicit member constructor
//! already does). Same contract as [`crate::ops`]; the strict comparator is the judge.
use crate::cst::{strip_parens, Cst, Edit};
use crate::func;
use crate::ops::M;

fn indent_of(cst: &Cst, n: usize) -> String {
    let s = cst.nodes[n].start;
    let ls = cst.src[..s].rfind('\n').map(|i| i + 1).unwrap_or(0);
    cst.src[ls..s].chars().take_while(|c| c.is_whitespace()).collect()
}

fn stmts(c: &Cst, block: usize) -> Vec<usize> {
    c.named(block).into_iter().filter(|&s| func::is_stmt(c.kind(s))).collect()
}

/// Last name component of a call's function expression (`rstl::min_val<float>` -> `min_val`).
fn callee_name(c: &Cst, call: usize) -> Option<String> {
    let f = c.child(call, "function")?;
    let t = c.text(f);
    let t = t.split('<').next().unwrap_or(t);
    let t = t.rsplit(['.', '>', ':']).next().unwrap_or(t);
    Some(t.trim().to_string())
}

/// Helpers whose two arguments may be given in either order (min/max select the other operand
/// only on ties or NaN, which the comparator settles).
const SYMMETRIC: &[&str] = &["min_val", "max_val", "min", "max", "Min", "Max", "fmin", "fmax", "__fmin", "__fmax", "Dot", "close_enough"];

/// `min_val(a, b)` -> `min_val(b, a)` (inline expansions compare in operand order).
pub fn op_swap_args(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) != "call_expression" {
            continue;
        }
        let Some(name) = callee_name(c, n) else { continue };
        if !SYMMETRIC.contains(&name.as_str()) {
            continue;
        }
        let Some(args) = c.child(n, "arguments") else { continue };
        let a: Vec<usize> = c.named(args).into_iter().filter(|&x| c.kind(x) != "comment").collect();
        if a.len() == 2 {
            cands.push((a[0], a[1]));
        }
    }
    let (x, y) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, x, c.text(y).to_string()), Edit::replace(c, y, c.text(x).to_string())])
}

fn ret_value(c: &Cst, s: usize) -> Option<usize> {
    let s = single(c, s)?;
    (c.kind(s) == "return_statement").then(|| c.named(s).first().copied()).flatten()
}

fn single(c: &Cst, n: usize) -> Option<usize> {
    if c.kind(n) == "compound_statement" {
        let v = stmts(c, n);
        (v.len() == 1).then(|| v[0])
    } else {
        Some(n)
    }
}

/// Single exit through a result variable (the target branches to one shared epilogue / `blr`):
/// `return c ? a : b;`, `if (c) return a; else return b;` and `if (c) return a; return b;` at the
/// end of the body become `T r; if (c) { r = a; } else { r = b; } return r;` (and back:
/// `T r; if (c) r = a; else r = b; return r;` -> `return c ? a : b;`).
pub fn op_ret_var(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let ret = m.info.ret_ty.trim().to_string();
    if ret.is_empty() || ret == "void" || ret.ends_with('&') {
        return None;
    }
    let body = m.info.body;
    let l = stmts(c, body);
    let mut cands: Vec<(usize, usize, String, String, String)> = Vec::new(); // (first, last, cond, a, b)
    if let Some(&last) = l.last() {
        // return c ? a : b;
        if let Some(e) = ret_value(c, last).filter(|&e| c.kind(e) == "conditional_expression") {
            if let (Some(cc), Some(a), Some(b)) = (c.child(e, "condition"), c.child(e, "consequence"), c.child(e, "alternative")) {
                cands.push((last, last, strip_parens(c.text(cc)).to_string(), c.text(a).to_string(), c.text(b).to_string()));
            }
        }
        // if (c) return a; else return b;
        if c.kind(last) == "if_statement" {
            if let (Some(cv), Some(cons), Some(alt)) = (cond_value(c, last), c.child(last, "consequence"), c.child(last, "alternative")) {
                let alt = if c.kind(alt) == "else_clause" { c.named(alt).first().copied().unwrap_or(alt) } else { alt };
                if let (Some(a), Some(b)) = (ret_value(c, cons), ret_value(c, alt)) {
                    cands.push((last, last, strip_parens(c.text(cv)).to_string(), c.text(a).to_string(), c.text(b).to_string()));
                }
            }
        }
        // if (c) return a; return b;
        if l.len() >= 2 {
            let prev = l[l.len() - 2];
            if c.kind(prev) == "if_statement" && c.child(prev, "alternative").is_none() {
                if let (Some(cv), Some(cons), Some(b)) = (cond_value(c, prev), c.child(prev, "consequence"), ret_value(c, last)) {
                    if let Some(a) = ret_value(c, cons) {
                        cands.push((prev, last, strip_parens(c.text(cv)).to_string(), c.text(a).to_string(), c.text(b).to_string()));
                    }
                }
            }
        }
    }
    let (first, last, ct, a, b) = m.pick_one(&cands)?;
    let ind = indent_of(c, first);
    let v = m.info.fresh_name(c, "result");
    let text = if m.rng.chance(0.5) {
        format!("{ret} {v};\n{ind}if ({ct}) {{\n{ind}    {v} = {a};\n{ind}}} else {{\n{ind}    {v} = {b};\n{ind}}}\n{ind}return {v};")
    } else {
        format!("{ret} {v} = {b};\n{ind}if ({ct}) {{\n{ind}    {v} = {a};\n{ind}}}\n{ind}return {v};")
    };
    Some(vec![Edit { start: c.nodes[first].start, end: c.nodes[last].end, text }])
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

fn is_member_lvalue(c: &Cst, n: usize) -> bool {
    match c.kind(n) {
        "field_expression" | "subscript_expression" => true,
        "pointer_expression" => c.op(n) == Some("*"),
        "parenthesized_expression" => c.named(n).first().is_some_and(|&i| is_member_lvalue(c, i)),
        _ => false,
    }
}

/// Drop a store to memory (`this->x = 0;`, `*(int*)((char*)this + 4) = 0;`) or a call statement:
/// for candidates with instructions the target lacks, e.g. an explicit store that an implicit
/// member constructor already performs, or an explicit call the compiler emits by itself.
pub fn op_delete_stmt(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) != "expression_statement" || c.parent(n).map_or(true, |p| c.kind(p) != "compound_statement") {
            continue;
        }
        let Some(&e) = c.named(n).first() else { continue };
        match c.kind(e) {
            "assignment_expression" if c.op(e) == Some("=") => {
                let (Some(l), Some(r)) = (c.child(e, "left"), c.child(e, "right")) else { continue };
                // chained `a = b = 0`: keep the inner assignment
                if c.kind(r) == "assignment_expression" {
                    cands.push((n, Some(c.text(r).to_string())));
                } else if is_member_lvalue(c, l) && pure(c, r) {
                    cands.push((n, None));
                }
            }
            "call_expression" => cands.push((n, None)),
            _ => {}
        }
    }
    let (s, keep) = m.pick_one(&cands)?;
    Some(vec![match keep {
        Some(k) => Edit::replace(c, s, format!("{k};")),
        None => remove_stmt(c, s),
    }])
}

fn pure(c: &Cst, n: usize) -> bool {
    !c.descendants(n).into_iter().any(|d| matches!(c.kind(d), "call_expression" | "assignment_expression" | "update_expression" | "new_expression" | "delete_expression"))
}

fn remove_stmt(c: &Cst, s: usize) -> Edit {
    let mut start = c.nodes[s].start;
    let mut end = c.nodes[s].end;
    let line_start = c.src[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let rest = &c.src[end..];
    let ws = rest.len() - rest.trim_start_matches([' ', '\t']).len();
    if c.src[line_start..start].trim().is_empty() && rest[ws..].starts_with('\n') {
        start = line_start;
        end += ws + 1;
    }
    Edit { start, end, text: String::new() }
}


/// Split `s` at its top-level binary `+` (outside parentheses/brackets): `(a, b)`.
fn split_plus(s: &str) -> Option<(&str, &str)> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    let mut at = None;
    for (k, &ch) in b.iter().enumerate() {
        match ch {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b'+' if depth == 0 && k > 0 && b.get(k + 1) != Some(&b'+') && b[k - 1] != b'+' => at = Some(k),
            _ => {}
        }
    }
    let k = at?;
    Some((s[..k].trim(), s[k + 1..].trim()))
}

fn int_lit(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(h, 16).ok()
    } else {
        s.parse().ok()
    }
}

/// `*(T*)((char*)E + k)` (either operand order) -> (T, E, k).
fn raw_load(s: &str) -> Option<(String, String, i64)> {
    let s = strip_parens(s);
    let rest = s.strip_prefix('*')?.trim_start();
    let rest = rest.strip_prefix('(')?;
    let close = rest.find(')')?;
    let ty = rest[..close].trim();
    let ty = ty.strip_suffix('*')?.trim().to_string();
    let addr = strip_parens(&rest[close + 1..]);
    let (a, b) = split_plus(addr)?;
    let (base, k) = match (int_lit(a), int_lit(b)) {
        (None, Some(k)) => (a, k),
        (Some(k), None) => (b, k),
        _ => return None,
    };
    let base = strip_parens(base);
    let e = base.strip_prefix("(char*)").or_else(|| base.strip_prefix("(char *)"))?;
    Some((ty, strip_parens(e).to_string(), k))
}

fn type_size(t: &str) -> Option<i64> {
    match t {
        "float" | "int" | "unsigned int" | "s32" | "u32" | "f32" | "long" | "unsigned long" => Some(4),
        "double" | "f64" | "long long" | "unsigned long long" => Some(8),
        "short" | "unsigned short" | "s16" | "u16" => Some(2),
        "char" | "unsigned char" | "signed char" | "s8" | "u8" | "bool" => Some(1),
        _ => None,
    }
}

/// Member-wise construction from consecutive members of one object back to a struct copy:
/// `T(*(float*)((char*)p + 0x3c), *(float*)((char*)p + 0x40), p->z)` -> `*(T*)((char*)p + 0x3c)`
/// (the target copies member by member, load/store pairs, instead of loading every member first).
/// Arguments may be raw loads at consecutive offsets or member accesses of the same object (their
/// offsets inferred from the raw ones).
pub fn op_ctor_copy(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) != "call_expression" {
            continue;
        }
        let Some(f) = c.child(n, "function") else { continue };
        let ty = c.text(f).trim().to_string();
        let last = ty.rsplit("::").next().unwrap_or(&ty);
        if !last.chars().next().is_some_and(|ch| ch.is_ascii_uppercase()) || !matches!(c.kind(f), "identifier" | "qualified_identifier" | "template_function") {
            continue;
        }
        let Some(args) = c.child(n, "arguments") else { continue };
        let a: Vec<usize> = c.named(args).into_iter().filter(|&x| c.kind(x) != "comment").collect();
        if a.len() < 2 {
            continue;
        }
        let parsed: Vec<Option<(String, String, i64)>> = a.iter().map(|&x| raw_load(c.text(x))).collect();
        let Some((ety, base, _)) = parsed.iter().flatten().next().cloned() else { continue };
        let Some(sz) = type_size(&ety) else { continue };
        // start offset implied by every raw argument must agree
        let mut start = None;
        let mut ok = true;
        for (i, p) in parsed.iter().enumerate() {
            match p {
                Some((t, b, k)) => {
                    if *t != ety || *b != base {
                        ok = false;
                        break;
                    }
                    let s0 = k - sz * i as i64;
                    if start.is_some_and(|s| s != s0) {
                        ok = false;
                        break;
                    }
                    start = Some(s0);
                }
                None => {
                    // a member access of the same object
                    let t = strip_parens(c.text(a[i]));
                    let pre_arrow = format!("{base}->");
                    let pre_paren = format!("({base})->");
                    if !(t.starts_with(&pre_arrow) || t.starts_with(&pre_paren)) || t[pre_arrow.len().min(t.len())..].contains(['(', '-', '+', '*', '[']) {
                        ok = false;
                        break;
                    }
                }
            }
        }
        let Some(s0) = start.filter(|_| ok) else { continue };
        if s0 < 0 {
            continue;
        }
        cands.push((n, format!("*({ty}*)((char*){} + 0x{s0:x})", if base.chars().all(|ch| ch.is_alphanumeric() || ch == '_' || ch == ':') { base.clone() } else { format!("({base})") })));
    }
    let (n, text) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, text)])
}

/// `return T(x);` with `T` the function's return type -> `return x;` (the converting constructor
/// called implicitly), and back. The two forms build the returned object in a different order
/// against the epilogue (the explicit temporary is a separate object until it is copied).
pub fn op_return_implicit(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let ret = m.info.ret_ty.split_whitespace().collect::<String>();
    if ret.is_empty() || ret == "void" || ret.ends_with('&') || ret.ends_with('*') {
        return None;
    }
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) != "return_statement" {
            continue;
        }
        let Some(&e) = c.named(n).first() else { continue };
        let explicit = c.kind(e) == "call_expression" && c.child(e, "function").is_some_and(|f| c.text(f).split_whitespace().collect::<String>() == ret);
        if explicit {
            let Some(args) = c.child(e, "arguments") else { continue };
            let a: Vec<usize> = c.named(args).into_iter().filter(|&x| c.kind(x) != "comment").collect();
            if a.len() == 1 {
                cands.push((e, c.text(a[0]).to_string()));
            }
        } else if !matches!(c.kind(e), "number_literal" | "true" | "false" | "null" | "nullptr") {
            cands.push((e, format!("{}({})", m.info.ret_ty.trim(), c.text(e))));
        }
    }
    let (e, text) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, e, text)])
}

/// Call the `const` overload of a member function: `E->M(a)` -> `((const __typeof__(*E)*)(E))->M(a)`
/// (`E.M(a)` -> `((const __typeof__(E)&)(E)).M(a)`); a result whose address is taken is cast
/// back to the non-const pointer type (`(__typeof__(&E->M(a)))&...`). For targets that call
/// `M() const` where the draft resolves the non-const overload (a relocation to `M__1CFv` instead
/// of `M__1CCFv`).
pub fn op_const_call(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        if c.kind(n) != "call_expression" {
            continue;
        }
        let Some(f) = c.child(n, "function") else { continue };
        if c.kind(f) != "field_expression" {
            continue;
        }
        let (Some(obj), Some(field)) = (c.child(f, "argument"), c.child(f, "field")) else { continue };
        let Some(args) = c.child(n, "arguments") else { continue };
        let o = c.text(obj);
        if o.contains("__typeof__") {
            continue;
        }
        let arrow = c.text(f)[c.nodes[obj].end - c.nodes[f].start..].trim_start().starts_with("->");
        let new_call = if arrow {
            format!("((const __typeof__(*{o})*)({o}))->{}{}", c.text(field), c.text(args))
        } else {
            format!("((const __typeof__({o})&)({o})).{}{}", c.text(field), c.text(args))
        };
        // `&call`: back to the non-const pointer type
        match c.parent(n).filter(|&p| c.kind(p) == "pointer_expression" && c.text(p).trim_start().starts_with('&')) {
            Some(p) => cands.push((p, format!("(__typeof__({}))&{new_call}", c.text(p)))),
            None => cands.push((n, new_call)),
        }
    }
    let (n, text) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, text)])
}

/// A comparison used as a number (`(a == b) + 1`, `x = a < b;`) <-> a select of the constants
/// (`(a == b ? 1 : 0) + 1`): the compiler extracts the comparison bit with a byte (bool) width
/// for the first and an int width for the second.
pub fn op_cmp_select(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        match c.kind(n) {
            "binary_expression" if matches!(c.op(n), Some("==" | "!=" | "<" | ">" | "<=" | ">=")) => {
                // used as a value: an arithmetic operand, an initializer / assignment source, or returned
                let Some(p) = c.parent(n).map(|p| {
                    let mut p = p;
                    while c.kind(p) == "parenthesized_expression" {
                        match c.parent(p) {
                            Some(q) => p = q,
                            None => break,
                        }
                    }
                    p
                }) else { continue };
                let value_use = match c.kind(p) {
                    "binary_expression" => matches!(c.op(p), Some("+" | "-" | "*" | "&" | "|" | "^" | "<<" | ">>")),
                    "init_declarator" | "assignment_expression" | "return_statement" | "cast_expression" => true,
                    _ => false,
                };
                if value_use {
                    cands.push((n, format!("({} ? 1 : 0)", c.text(n))));
                }
            }
            "conditional_expression" => {
                let (Some(cond), Some(t), Some(f)) = (c.child(n, "condition"), c.child(n, "consequence"), c.child(n, "alternative")) else { continue };
                match (c.text(t).trim(), c.text(f).trim()) {
                    ("1", "0") => cands.push((n, format!("({})", c.text(cond)))),
                    ("0", "1") => cands.push((n, format!("!({})", c.text(cond)))),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    let (n, t) = m.pick_one(&cands)?;
    Some(vec![Edit::replace(c, n, t)])
}

/// The value of a flag-if chain folded back into one logical expression:
/// `T v = false; if (A) { t = E; if (B) { v = true; } } return !v;` -> `return !(A && B[t := E]);`
/// (also `return v;`, and a direct `if (A && B)`). Inline expansions of `&&` in value context
/// come out of the lifter as such chains.
pub fn op_flag_and(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, usize, String)> = Vec::new(); // (first stmt, last stmt, replacement)
    for b in m.nodes.clone() {
        if c.kind(b) != "compound_statement" {
            continue;
        }
        let l = stmts(c, b);
        for i in 0..l.len().saturating_sub(2) {
            // `T v = false;` / `T v = 0;`
            let d = l[i];
            if c.kind(d) != "declaration" {
                continue;
            }
            let Some(decl) = c.child(d, "declarator").filter(|&x| c.kind(x) == "init_declarator") else { continue };
            let (Some(name), Some(val)) = (c.child(decl, "declarator"), c.child(decl, "value")) else { continue };
            let v = c.text(name).trim().to_string();
            if !matches!(c.text(val).trim(), "false" | "0") {
                continue;
            }
            // `if (A) { [t = E;]* if (B) { v = true; } }`
            let outer = l[i + 1];
            if c.kind(outer) != "if_statement" || c.child(outer, "alternative").is_some() {
                continue;
            }
            let (Some(ca), Some(body)) = (c.child(outer, "condition"), c.child(outer, "consequence")) else { continue };
            let inner_list = if c.kind(body) == "compound_statement" { stmts(c, body) } else { vec![body] };
            let Some((&inner, defs)) = inner_list.split_last() else { continue };
            let set_true = |s: usize| {
                let s = single(c, s);
                s.is_some_and(|s| c.kind(s) == "expression_statement" && {
                    let t = c.text(s).split_whitespace().collect::<String>();
                    t == format!("{v}=true;") || t == format!("{v}=1;")
                })
            };
            let (cond, ok) = if c.kind(inner) == "if_statement" && c.child(inner, "alternative").is_none() {
                let (Some(cb), Some(ib)) = (c.child(inner, "condition"), c.child(inner, "consequence")) else { continue };
                // inline the temporaries defined before the inner if into its condition
                let mut cb_text = strip_parens(c.text(cb)).to_string();
                let mut ok = set_true(ib);
                for &s in defs.iter().rev() {
                    let t = c.text(s).trim().trim_end_matches(';').to_string();
                    let Some((lhs, rhs)) = t.split_once('=') else {
                        ok = false;
                        break;
                    };
                    // (`T t = E` declares it here)
                    let lhs = lhs.trim().rsplit([' ', '*', '&']).next().unwrap_or("");
                    if lhs.is_empty() || !lhs.chars().all(|ch| ch.is_alphanumeric() || ch == '_') || rhs.starts_with('=') {
                        ok = false;
                        break;
                    }
                    cb_text = replace_ident(&cb_text, lhs, &format!("({})", rhs.trim()));
                }
                (format!("({}) && ({cb_text})", strip_parens(c.text(ca))), ok)
            } else {
                (strip_parens(c.text(ca)).to_string(), defs.is_empty() && set_true(inner))
            };
            if !ok {
                continue;
            }
            // `return v;` / `return !v;` right after
            let ret = l[i + 2];
            if c.kind(ret) != "return_statement" {
                continue;
            }
            let rt = c.text(ret).split_whitespace().collect::<String>();
            let text = if rt == format!("return{v};") {
                format!("return {cond};")
            } else if rt == format!("return!{v};") || rt == format!("return!({v});") {
                format!("return !({cond});")
            } else {
                continue;
            };
            // `v` used nowhere else
            if l.iter().enumerate().any(|(k, &s)| (k < i || k > i + 2) && !crate::ops::uses_of(c, s, &v).is_empty()) {
                continue;
            }
            cands.push((d, ret, text));
        }
    }
    let (a, b, t) = m.pick_one(&cands)?;
    Some(vec![Edit { start: c.nodes[a].start, end: c.nodes[b].end, text: t }])
}

/// `s` with identifier `name` replaced by `with` (whole words only).
fn replace_ident(s: &str, name: &str, with: &str) -> String {
    let mut out = String::new();
    let b = s.as_bytes();
    let mut i = 0;
    let is_id = |ch: u8| ch.is_ascii_alphanumeric() || ch == b'_';
    while i < s.len() {
        if s[i..].starts_with(name) && (i == 0 || !is_id(b[i - 1])) && b.get(i + name.len()).is_none_or(|&ch| !is_id(ch)) {
            out.push_str(with);
            i += name.len();
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}
