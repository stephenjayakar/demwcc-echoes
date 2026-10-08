//! Register-only repair: a bounded, deterministic neighbourhood search for candidates whose code
//! differs from the target only in register numbers (same instructions otherwise).
//!
//! Register choice follows from the variable structure of the source (regalloc.md): which values
//! are named locals and which are temporaries, declaration order, evaluation order of operands
//! and the live ranges the statements give each value. Those are exactly the knobs a small set of
//! operators turns: naming / inlining a temporary, reordering declarations, swapping commutative
//! operands, moving a statement, `const` on a by-value parameter, the type of a local. Instead of
//! sampling them at random (the main search), every site of every such operator is enumerated
//! (level 1), the real compiler scores each neighbour, and the best few neighbours are expanded
//! once more (level 2), until an exact match or the compile budget. Directed edits from the
//! compiler tracer (the real colouring of the candidate, [`crate::trace`]) go first when a tracer
//! is available.
//!
//! The result is deterministic for a given source and target (no random sampling beyond fixed
//! enumeration seeds), so it can run as part of drafting.
use crate::cst::normalize;
use crate::hints::RegHints;
use crate::ops::{self, Parsed};
use crate::rng::Rng;
use crate::score::{DiffProfile, Eval, Fitness, Scorer};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Operators that change register choice without changing the instruction mix much, in
/// priority order (most often the deciding edit first; train-split eval statistics).
pub const REG_OPS: &[&str] = &[
    "extract_temp",
    "commutative",
    "inline_var_all",
    "reorder_decls",
    "hoist_decl",
    "param_const",
    "local_type",
    "inline_temp",
    "swap_stmts",
    "move_stmt",
    "cse_temp",
    "merge_decl",
    "flip_compare",
    "compound_assign",
    "swap_stores",
    "chain_assign",
    "refer_to_var",
    "ref_local",
    "split_update",
    "swap_args",
    "vec_op",
    "associative",
    "if_to_ternary",
    "ternary_to_if",
    "select_init",
    "hint_order",
    "hint_temp",
];

#[derive(Clone, Debug)]
pub struct RepairConfig {
    /// Real compiles at most (cache hits don't count).
    pub max_compiles: usize,
    /// Compile budget and extra levels when compiles turn out cheap (persistent compilers: level 1
    /// averaged under `cheap_ms` of wall time per compile).
    pub cheap_max_compiles: usize,
    pub cheap_levels: usize,
    pub cheap_ms: f64,
    /// Neighbours of level 1 expanded at level 2.
    pub beam: usize,
    /// Concurrent compiles.
    pub threads: usize,
    /// Enumeration seeds tried per operator (each picks a site at random; duplicates dropped).
    pub seeds: u64,
    /// Wall-clock cap (a safety net for slow contexts; the compile cap normally ends first).
    pub max_time: std::time::Duration,
}

impl Default for RepairConfig {
    fn default() -> Self {
        RepairConfig { max_compiles: 160, cheap_max_compiles: 480, cheap_levels: 1, cheap_ms: 12.0, beam: 4, threads: 4, seeds: 48, max_time: std::time::Duration::from_secs(20) }
    }
}

#[derive(Clone, Debug)]
pub struct Repair {
    pub src: String,
    pub fitness: Fitness,
    /// Operators applied, in order.
    pub ops: Vec<&'static str>,
    pub compiles: usize,
}

/// The diff is register-only or order-only (same instructions, other registers or positions),
/// so variable-structure and evaluation-order edits are the right tools.
pub fn register_only(f: &Fitness) -> bool {
    let p: &DiffProfile = &f.profile;
    !f.exact && f.size_delta == 0 && p.inserted + p.deleted + p.substituted == 0 && p.stack == 0 && p.branch == 0 && p.reloc == 0 && p.reg + p.reorder > 0
}

/// Every distinct neighbour of `src` under [`REG_OPS`] (one operator application each), in
/// operator priority order. Deduplicated by normalized text against `seen`.
pub fn neighbours(src: &str, symbol: &str, hints: Option<&RegHints>, seeds: u64, seen: &mut HashSet<String>) -> Vec<(String, &'static str)> {
    let Some(p) = Parsed::new(src, symbol) else { return vec![] };
    let mut out = vec![];
    for c in select_else_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "select_else"));
        }
    }
    for c in sink_load_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "sink_load"));
        }
    }
    for c in return_construct_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "return_construct"));
        }
    }
    for c in construct_copy_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "construct_copy"));
        }
    }
    for c in lvalue_ref_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "lvalue_ref"));
        }
    }
    for c in vec_compound_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "vec_compound"));
        }
    }
    for c in swap_defs_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "swap_defs"));
        }
    }
    for c in inline_getter_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "inline_getter"));
        }
    }
    for c in obj_copy_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "obj_copy"));
        }
    }
    for c in vec_local_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "vec_local"));
        }
    }
    for c in arg_order_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "arg_order"));
        }
    }
    for name in REG_OPS {
        let Some(op) = ops::op_index(name) else { continue };
        if (*name == "hint_order" || *name == "hint_temp") && hints.is_none() {
            continue;
        }
        let mut dry = 0;
        for s in 0..seeds {
            let mut rng = Rng::new(0x5eed ^ (s * 0x9e37));
            let Some(c) = p.apply(op, &mut rng, hints) else {
                dry += 1;
                if dry > 6 && s >= 8 {
                    break;
                }
                continue;
            };
            if seen.insert(normalize(&c)) {
                out.push((c, ops::OPS[op].name));
                dry = 0;
            } else {
                dry += 1;
                if dry > 12 {
                    break;
                }
            }
        }
    }
    out
}

/// `src` with every split declaration (`T x; ... x = v;`) merged into its first assignment
/// (`T x = v;`): the drafts' style, which keeps the temp-inlining operators from applying.
pub fn merge_all_decls(src: &str, symbol: &str) -> Option<String> {
    let op = ops::op_index("merge_decl")?;
    let mut cur = src.to_string();
    for i in 0..64u64 {
        let Some(p) = Parsed::new(&cur, symbol) else { break };
        let mut rng = Rng::new(i);
        match p.apply(op, &mut rng, None) {
            Some(n) => cur = n,
            None => break,
        }
    }
    (cur != src).then_some(cur)
}

/// Runs of consecutive assignments `L1 = R1; L2 = R2; ... Lk = Rk;` rewritten the way an
/// inlined call with arguments evaluates them (right to left, into front-end temporaries):
/// `T tk = Rk; ...; T t2 = R2; L1 = R1; L2 = t2; ...`. Front-end temporaries are numbered before
/// codegen temporaries, so this changes which values are coloured first (e.g. three FIFO writes
/// `GXPosition3f32(v.x, v.y, v.z)` expanded by the compiler).
pub fn arg_order_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    for blk in c.descendants(info.body) {
        if c.kind(blk) != "compound_statement" {
            continue;
        }
        let sibs: Vec<usize> = c.named(blk).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
        // (statement, right-hand side) of plain assignment statements
        let rhs = |t: usize| -> Option<usize> {
            if c.kind(t) != "expression_statement" {
                return None;
            }
            let a = *c.named(t).first()?;
            if c.kind(a) != "assignment_expression" || c.op(a) != Some("=") {
                return None;
            }
            let r = c.child(a, "right")?;
            (!matches!(c.kind(r), "number_literal" | "identifier" | "true" | "false" | "null" | "nullptr")).then_some(r)
        };
        for i in 0..sibs.len() {
            for k in 2..=4usize {
                if i + k > sibs.len() {
                    break;
                }
                let rs: Option<Vec<usize>> = (i..i + k).map(|j| rhs(sibs[j])).collect();
                let Some(rs) = rs else { break };
                let at = c.nodes[sibs[i]].start;
                let line_start = src[..at].rfind('\n').map(|x| x + 1).unwrap_or(0);
                let ind = &src[line_start..at];
                let mut decls = String::new();
                let mut edits = vec![];
                for j in (1..k).rev() {
                    let name = info.fresh_name(&c, &format!("arg{j}_"));
                    let ty = crate::func::temp_type(&c, &info, rs[j]);
                    decls.push_str(&format!("{ty} {name} = {};\n{ind}", c.text(rs[j])));
                    edits.push(Edit::replace(&c, rs[j], name));
                }
                edits.push(Edit::insert(at, decls));
                if let Some(s) = apply(src, &edits) {
                    if Cst::parse(&s).errors <= c.errors {
                        out.push(s);
                    }
                }
            }
        }
    }
    out
}

/// `A.GetC()` accessor call: (object text, component).
fn component(c: &crate::cst::Cst, n: usize) -> Option<(String, usize)> {
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

/// Component-wise differences / sums kept in float locals (`x = A.GetX() - B.GetX(); y = ...;
/// z = ...;`) computed as one vector local first (`__typeof__(A - B) d = A - B; x = d.GetX();
/// ...`): the expanded vector operator creates the components in its own order (front-end
/// temporaries), which decides their registers.
pub fn vec_local_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    for blk in c.descendants(info.body) {
        if c.kind(blk) != "compound_statement" {
            continue;
        }
        let sibs: Vec<usize> = c.named(blk).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
        // (statement index, value node, A, B, op, component)
        let mut items: Vec<(usize, usize, String, String, String, usize)> = vec![];
        for (i, &t) in sibs.iter().enumerate() {
            let val = match c.kind(t) {
                "expression_statement" => c.named(t).first().copied().filter(|&a| c.kind(a) == "assignment_expression" && c.op(a) == Some("=")).and_then(|a| c.child(a, "right")),
                "declaration" => {
                    let ds: Vec<usize> = c.children_by_field(t, "declarator").collect();
                    (ds.len() == 1 && c.kind(ds[0]) == "init_declarator").then(|| c.child(ds[0], "value")).flatten()
                }
                _ => None,
            };
            let Some(v) = val else { continue };
            let v0 = if c.kind(v) == "parenthesized_expression" { c.named(v).first().copied().unwrap_or(v) } else { v };
            if c.kind(v0) != "binary_expression" {
                continue;
            }
            let Some(op) = c.op(v0).filter(|o| *o == "-" || *o == "+").map(String::from) else { continue };
            let (Some(l), Some(r)) = (c.child(v0, "left"), c.child(v0, "right")) else { continue };
            if let (Some((a, ka)), Some((b, kb))) = (component(&c, l), component(&c, r)) {
                if ka == kb {
                    items.push((i, v, a, b, op, ka));
                }
            }
        }
        let mut done: Vec<(String, String, String)> = vec![];
        for it in &items {
            let key = (it.2.clone(), it.3.clone(), it.4.clone());
            if done.contains(&key) {
                continue;
            }
            done.push(key.clone());
            let group: Vec<&(usize, usize, String, String, String, usize)> = items.iter().filter(|x| x.2 == key.0 && x.3 == key.1 && x.4 == key.2).collect();
            let mut comps: Vec<usize> = group.iter().map(|x| x.5).collect();
            comps.sort();
            if comps != [0, 1, 2] {
                continue;
            }
            let first = group.iter().map(|x| x.0).min().unwrap();
            let at = c.nodes[sibs[first]].start;
            let line_start = src[..at].rfind('\n').map(|x| x + 1).unwrap_or(0);
            let ind = &src[line_start..at];
            let name = info.fresh_name(&c, "vec");
            let e = format!("{} {} {}", key.0, key.2, key.1);
            let mut edits = vec![Edit::insert(at, format!("__typeof__({e}) {name} = {e};\n{ind}"))];
            for x in &group {
                edits.push(Edit::replace(&c, x.1, format!("{name}.{}()", ["GetX", "GetY", "GetZ"][x.5])));
            }
            if let Some(s) = apply(src, &edits) {
                if Cst::parse(&s).errors <= c.errors {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// An object rebuilt from its own components (`x = A.GetX(); y = A.GetY(); z = A.GetZ(); ...
/// B = T(x, y, z);`) copied whole instead (`__typeof__(A) cp = A; ... B = cp;`, components read
/// from the copy): a struct copy loads every member at once, in its own order. The copy goes
/// before the first component read in the block of the rebuild (and, as a second candidate,
/// before the first component read at all).
pub fn obj_copy_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    // single assignments `v = A.GetC()` (statement, value node, A, component) by variable
    let mut defs: std::collections::HashMap<String, Vec<(usize, usize, String, usize)>> = Default::default();
    for t in c.descendants(info.body) {
        let (name, val) = match c.kind(t) {
            "expression_statement" => {
                let Some(a) = c.named(t).first().copied().filter(|&a| c.kind(a) == "assignment_expression" && c.op(a) == Some("=")) else { continue };
                let Some(l) = c.child(a, "left").filter(|&l| c.kind(l) == "identifier") else { continue };
                (c.text(l).to_string(), c.child(a, "right"))
            }
            "declaration" => {
                let ds: Vec<usize> = c.children_by_field(t, "declarator").collect();
                if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
                    continue;
                }
                let Some((n, _)) = crate::func::declarator_name(&c, ds[0]) else { continue };
                (n, c.child(ds[0], "value"))
            }
            _ => continue,
        };
        let Some(v) = val else { continue };
        let e = defs.entry(name).or_default();
        match component(&c, v) {
            Some((a, k)) => e.push((t, v, a, k)),
            None => e.push((t, v, String::new(), 9)),
        }
    }
    let mut out = vec![];
    for n in c.descendants(info.body) {
        if c.kind(n) != "call_expression" {
            continue;
        }
        let (Some(f), Some(args)) = (c.child(n, "function"), c.child(n, "arguments")) else { continue };
        if !matches!(c.kind(f), "identifier" | "qualified_identifier" | "type_identifier") || info.vars.contains_key(c.text(f)) {
            continue;
        }
        let a: Vec<usize> = c.named(args);
        if a.len() != 3 || !a.iter().all(|&x| c.kind(x) == "identifier") {
            continue;
        }
        let ds: Option<Vec<(usize, usize, String, usize)>> = a
            .iter()
            .enumerate()
            .map(|(k, &x)| match defs.get(c.text(x)).map(|v| v.as_slice()) {
                Some([d]) if d.3 == k => Some(d.clone()),
                _ => None,
            })
            .collect();
        let Some(ds) = ds else { continue };
        if ds.iter().any(|d| d.2 != ds[0].2) {
            continue;
        }
        let obj = ds[0].2.clone();
        // the statement of `n` and its enclosing block
        let Some(stmt) = c.ancestors(n).into_iter().find(|&x| crate::func::is_stmt(c.kind(x)) && c.parent(x).is_some_and(|p| c.kind(p) == "compound_statement")) else { continue };
        let blk = c.parent(stmt).unwrap();
        let name = info.fresh_name(&c, "copy");
        let mut spots: Vec<Vec<usize>> = vec![];
        let in_blk: Vec<usize> = ds.iter().filter(|d| c.parent(d.0) == Some(blk)).map(|d| d.0).collect();
        if !in_blk.is_empty() {
            spots.push(in_blk);
        }
        let all_same_parent = ds.iter().all(|d| c.parent(d.0) == c.parent(ds[0].0));
        if all_same_parent {
            spots.push(ds.iter().map(|d| d.0).collect());
        }
        let mut seen_at = vec![];
        for sp in spots {
            let first = *sp.iter().min_by_key(|&&t| c.nodes[t].start).unwrap();
            if seen_at.contains(&first) {
                continue;
            }
            seen_at.push(first);
            let at = c.nodes[first].start;
            if c.nodes[stmt].start < at {
                continue;
            }
            let line_start = src[..at].rfind('\n').map(|x| x + 1).unwrap_or(0);
            let ind = &src[line_start..at];
            let mut edits = vec![Edit::insert(at, format!("__typeof__({obj}) {name} = {obj};\n{ind}")), Edit::replace(&c, n, name.clone())];
            for d in ds.iter().filter(|d| sp.contains(&d.0)) {
                edits.push(Edit::replace(&c, d.1, format!("{name}.{}()", ["GetX", "GetY", "GetZ"][d.3])));
            }
            if let Some(s) = apply(src, &edits) {
                if Cst::parse(&s).errors <= c.errors {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// A local holding an accessor result (`T t = obj.GetC();`) read again at each use instead
/// (`obj.GetC()`): the compiler CSEs or reloads it, whichever the target did, and the value's
/// register follows from where it is created. Only while `obj` is not written before the last
/// use's own statement (a `Set*` / assignment there happens after its operands are read).
pub fn inline_getter_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    for blk in c.descendants(info.body) {
        if c.kind(blk) != "compound_statement" {
            continue;
        }
        let sibs: Vec<usize> = c.named(blk).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
        for (i, &t) in sibs.iter().enumerate() {
            let (name, val) = match c.kind(t) {
                "declaration" => {
                    let ds: Vec<usize> = c.children_by_field(t, "declarator").collect();
                    if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
                        continue;
                    }
                    let Some((n, suf)) = crate::func::declarator_name(&c, ds[0]) else { continue };
                    if !suf.is_empty() {
                        continue;
                    }
                    (n, c.child(ds[0], "value"))
                }
                _ => continue,
            };
            let Some(v) = val else { continue };
            if c.kind(v) != "call_expression" || c.child(v, "arguments").is_some_and(|a| !c.named(a).is_empty()) {
                continue;
            }
            let Some(f) = c.child(v, "function").filter(|&f| c.kind(f) == "field_expression") else { continue };
            if !c.child(f, "field").is_some_and(|x| c.text(x).starts_with("Get")) {
                continue;
            }
            let Some(obj) = c.child(f, "argument").map(|o| c.text(o).to_string()) else { continue };
            let root: String = obj.chars().take_while(|ch| ch.is_alphanumeric() || *ch == '_').collect();
            // uses after the declaration, in this block's later statements
            let uses: Vec<(usize, usize)> = sibs[i + 1..]
                .iter()
                .enumerate()
                .flat_map(|(k, &s2)| {
                    c.descendants(s2).into_iter().filter(|&d| c.kind(d) == "identifier" && c.text(d) == name && c.parent(d).is_some_and(|p| c.kind(p) != "init_declarator")).map(move |d| (i + 1 + k, d))
                })
                .collect();
            let Some(&(last, _)) = uses.last() else { continue };
            // the variable is never reassigned, the object never written before the last use's statement
            let writes_obj = |s2: usize| -> bool {
                c.descendants(s2).into_iter().any(|d| match c.kind(d) {
                    "assignment_expression" | "update_expression" => c.descendants(d).into_iter().take(3).any(|x| c.kind(x) == "identifier" && (c.text(x) == root || c.text(x) == name)),
                    "call_expression" => c.child(d, "function").is_some_and(|ff| c.kind(ff) != "field_expression" || !c.child(ff, "field").is_some_and(|x| c.text(x).starts_with("Get"))),
                    _ => false,
                })
            };
            if sibs[i + 1..last].iter().any(|&s2| writes_obj(s2)) || !matches!(c.kind(sibs[last]), "expression_statement" | "declaration") {
                continue;
            }
            if sibs[last..].iter().any(|&s2| c.descendants(s2).into_iter().any(|d| c.kind(d) == "assignment_expression" && c.child(d, "left").is_some_and(|l| c.text(l) == name))) {
                continue;
            }
            let mut edits: Vec<Edit> = uses.iter().map(|&(_, d)| Edit::replace(&c, d, c.text(v).to_string())).collect();
            edits.push(Edit::replace(&c, t, String::new()));
            if let Some(s) = apply(src, &edits) {
                if Cst::parse(&s).errors <= c.errors {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// Adjacent independent local definitions swapped (`T a = e1; U b = e2;` -> `U b = e2; T a = e1;`)
/// where the values only read memory, possibly through `Get*` accessors (which the general
/// statement swap treats as calls with side effects): their creation order decides registers.
pub fn swap_defs_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    // (defined local, value node) of a local definition reading memory only
    let local_def = |t: usize| -> Option<(String, usize)> {
        let (name, v) = match c.kind(t) {
            "declaration" => {
                let ds: Vec<usize> = c.children_by_field(t, "declarator").collect();
                if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
                    return None;
                }
                (crate::func::declarator_name(&c, ds[0])?.0, c.child(ds[0], "value")?)
            }
            "expression_statement" => {
                let a = *c.named(t).first()?;
                if c.kind(a) != "assignment_expression" || c.op(a) != Some("=") {
                    return None;
                }
                let l = c.child(a, "left").filter(|&l| c.kind(l) == "identifier")?;
                (c.text(l).to_string(), c.child(a, "right")?)
            }
            _ => return None,
        };
        let ok = c.descendants(v).into_iter().all(|d| match c.kind(d) {
            "assignment_expression" | "update_expression" | "new_expression" | "delete_expression" => false,
            "call_expression" => c.child(d, "function").is_some_and(|f| c.kind(f) == "field_expression" && c.child(f, "field").is_some_and(|x| c.text(x).starts_with("Get"))),
            _ => true,
        });
        ok.then_some((name, v))
    };
    let reads = |v: usize, n: &str| c.descendants(v).into_iter().any(|d| c.kind(d) == "identifier" && c.text(d) == n);
    let mut out = vec![];
    for blk in c.descendants(info.body) {
        if c.kind(blk) != "compound_statement" {
            continue;
        }
        let sibs: Vec<usize> = c.named(blk).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
        for w in sibs.windows(2) {
            let (Some((na, va)), Some((nb, vb))) = (local_def(w[0]), local_def(w[1])) else { continue };
            if na == nb || reads(vb, &na) || reads(va, &nb) {
                continue;
            }
            let e = vec![Edit::replace(&c, w[0], c.text(w[1]).to_string()), Edit::replace(&c, w[1], c.text(w[0]).to_string())];
            if let Some(s) = apply(src, &e) {
                if Cst::parse(&s).errors <= c.errors {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// Three component updates of one object by the same scalar (`D.SetX(D.GetX() * s);` for X, Y,
/// Z, in order) as the compound vector operator (`D *= s;`), whose expansion evaluates the
/// operands in its own order.
pub fn vec_compound_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    // `D.SetC(<D.GetC() op s | s op D.GetC()>)` -> (D, component, op, s)
    let shape = |t: usize| -> Option<(String, usize, String, String)> {
        if c.kind(t) != "expression_statement" {
            return None;
        }
        let call = *c.named(t).first()?;
        if c.kind(call) != "call_expression" {
            return None;
        }
        let f = c.child(call, "function").filter(|&f| c.kind(f) == "field_expression" && c.op(f) == Some("."))?;
        let k = ["SetX", "SetY", "SetZ"].iter().position(|n| c.child(f, "field").is_some_and(|x| c.text(x) == *n))?;
        let d = c.text(c.child(f, "argument")?).to_string();
        let args = c.named(c.child(call, "arguments")?);
        let [v] = args.as_slice() else { return None };
        let v = if c.kind(*v) == "parenthesized_expression" { *c.named(*v).first()? } else { *v };
        if c.kind(v) != "binary_expression" {
            return None;
        }
        let op = c.op(v)?.to_string();
        let (l, r) = (c.child(v, "left")?, c.child(v, "right")?);
        match (component(&c, l), component(&c, r)) {
            (Some((a, ka)), None) if a == d && ka == k => Some((d, k, op, c.text(r).to_string())),
            (None, Some((b, kb))) if b == d && kb == k && op == "*" => Some((d, k, op, c.text(l).to_string())),
            _ => None,
        }
    };
    let mut out = vec![];
    for blk in c.descendants(info.body) {
        if c.kind(blk) != "compound_statement" {
            continue;
        }
        let sibs: Vec<usize> = c.named(blk).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
        for (wi, w) in sibs.windows(3).enumerate() {
            let sh: Option<Vec<_>> = w.iter().map(|&t| shape(t)).collect();
            let Some(sh) = sh else { continue };
            if !(0..3).all(|k| sh[k].1 == k && sh[k].0 == sh[0].0 && sh[k].2 == sh[0].2 && sh[k].3 == sh[0].3) || !matches!(sh[0].2.as_str(), "*" | "/" | "+" | "-") {
                continue;
            }
            let text = format!("{} {}= {};", sh[0].0, sh[0].2, sh[0].3);
            let mut variants = vec![vec![Edit { start: c.nodes[w[0]].start, end: c.nodes[w[2]].end, text }]];
            // the scalar a local defined just before and read nowhere else: inlined too
            if wi > 0 {
                let p = sibs[wi - 1];
                let scalar = &sh[0].3;
                let val = match c.kind(p) {
                    "declaration" => {
                        let ds: Vec<usize> = c.children_by_field(p, "declarator").collect();
                        (ds.len() == 1 && c.kind(ds[0]) == "init_declarator" && crate::func::declarator_name(&c, ds[0]).is_some_and(|x| &x.0 == scalar)).then(|| c.child(ds[0], "value")).flatten()
                    }
                    "expression_statement" => c.named(p).first().copied().filter(|&a| c.kind(a) == "assignment_expression" && c.op(a) == Some("=") && c.child(a, "left").is_some_and(|l| c.text(l) == scalar)).and_then(|a| c.child(a, "right")),
                    _ => None,
                };
                let uses = c.descendants(info.body).into_iter().filter(|&d| c.kind(d) == "identifier" && c.text(d) == scalar).count();
                if let (Some(v), true) = (val, uses <= 5) {
                    let text = format!("{} {}= {};", sh[0].0, sh[0].2, c.text(v));
                    variants.push(vec![Edit { start: c.nodes[p].start, end: c.nodes[w[2]].end, text }]);
                }
            }
            for e in variants {
                if let Some(s) = apply(src, &e) {
                    if Cst::parse(&s).errors <= c.errors {
                        out.push(s);
                    }
                }
            }
        }
    }
    out
}

/// A store to a computed location (`a[i] = v;`, `p->a[i].f = v;`) through a reference bound
/// first (`__typeof__(a[i])& ref = a[i]; ref = v;`): the address is then computed before the
/// stored value, which changes their creation order and so their registers.
pub fn lvalue_ref_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    for t in c.descendants(info.body) {
        if c.kind(t) != "expression_statement" || !c.parent(t).is_some_and(|p| c.kind(p) == "compound_statement") {
            continue;
        }
        let Some(a) = c.named(t).first().copied().filter(|&a| c.kind(a) == "assignment_expression") else { continue };
        let (Some(l), Some(r)) = (c.child(a, "left"), c.child(a, "right")) else { continue };
        if !c.descendants(l).into_iter().any(|d| c.kind(d) == "subscript_expression") {
            continue;
        }
        let op = c.op(a).unwrap_or("=");
        let name = info.fresh_name(&c, "ref");
        let lt = c.text(l);
        let at = c.nodes[t].start;
        let line_start = src[..at].rfind('\n').map(|x| x + 1).unwrap_or(0);
        let ind = &src[line_start..at];
        let text = format!("__typeof__({lt})& {name} = {lt};\n{ind}{name} {op} {};", c.text(r));
        if let Some(s) = apply(src, &[Edit::replace(&c, t, text)]) {
            if Cst::parse(&s).errors <= c.errors {
                out.push(s);
            }
        }
    }
    out
}

/// `(base text, byte offset, element type text)` of a raw load `*(T*)((char*)B + off)`,
/// `*(T*)B` or `B[k]`.
fn raw_load(c: &crate::cst::Cst, n: usize) -> Option<(String, i64, String)> {
    let n = if c.kind(n) == "parenthesized_expression" { *c.named(n).first()? } else { n };
    if c.kind(n) == "subscript_expression" {
        let a = c.child(n, "argument")?;
        let idx = c.named(c.child(n, "indices")?);
        let [i] = idx.as_slice() else { return None };
        let k = parse_int(c.text(*i))?;
        return Some((c.text(a).to_string(), k * 4, String::new()));
    }
    if c.kind(n) != "pointer_expression" || c.op(n) != Some("*") {
        return None;
    }
    let cast = c.child(n, "argument")?;
    if c.kind(cast) != "cast_expression" {
        // `*B`: the object at the base itself
        return (!matches!(c.kind(cast), "pointer_expression" | "binary_expression")).then(|| (c.text(cast).to_string(), 0, String::new()));
    }
    let ty = c.text(c.child(cast, "type")?).trim().trim_end_matches('*').trim().to_string();
    let v = c.child(cast, "value")?;
    let v = if c.kind(v) == "parenthesized_expression" { *c.named(v).first()? } else { v };
    let (base, off) = if c.kind(v) == "binary_expression" && c.op(v) == Some("+") {
        (c.child(v, "left")?, parse_int(c.text(c.child(v, "right")?))?)
    } else {
        (v, 0)
    };
    let b = if c.kind(base) == "cast_expression" && c.text(c.child(base, "type")?).replace(' ', "") == "char*" { c.child(base, "value")? } else { base };
    Some((c.text(b).to_string(), off, ty))
}

fn parse_int(t: &str) -> Option<i64> {
    let t = t.trim();
    match t.strip_prefix("0x") {
        Some(h) => i64::from_str_radix(h, 16).ok(),
        None => t.parse().ok(),
    }
}

/// An object built from consecutive raw loads of one base (`T(*(float*)(B + 0x18),
/// *(float*)(B + 0x1c), ...)`) as a copy of the object there (`*(T*)((char*)B + 0x18)`): a
/// struct copy moves member by member (load/store pairs) instead of loading every member first.
pub fn construct_copy_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    for n in c.descendants(info.body) {
        if c.kind(n) != "call_expression" {
            continue;
        }
        let (Some(f), Some(args)) = (c.child(n, "function"), c.child(n, "arguments")) else { continue };
        if !matches!(c.kind(f), "identifier" | "qualified_identifier" | "type_identifier") || info.vars.contains_key(c.text(f)) {
            continue;
        }
        let a = c.named(args);
        if a.len() < 2 {
            continue;
        }
        let loads: Option<Vec<(String, i64, String)>> = a.iter().map(|&x| raw_load(&c, x)).collect();
        let Some(loads) = loads else { continue };
        if !loads.iter().enumerate().all(|(k, l)| l.0 == loads[0].0 && l.1 == loads[0].1 + 4 * k as i64) {
            continue;
        }
        let t = c.text(f);
        let text = format!("*({t}*)((char*){} + 0x{:x})", loads[0].0, loads[0].1);
        if let Some(s) = apply(src, &[Edit::replace(&c, n, text)]) {
            if Cst::parse(&s).errors <= c.errors {
                out.push(s);
            }
        }
    }
    out
}

/// A returned local filled member by member right before the return (`(&r)->a = x; (&r)->b = y;
/// return r;`) returned as a constructor call instead (`return T(x, y);`): the temporary is
/// built in place with the constructor's evaluation order.
pub fn return_construct_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    // object name of a member store `(&r)->f = e` / `r.f = e`
    let store_obj = |t: usize| -> Option<(String, usize)> {
        if c.kind(t) != "expression_statement" {
            return None;
        }
        let a = c.named(t).first().copied().filter(|&a| c.kind(a) == "assignment_expression" && c.op(a) == Some("="))?;
        let l = c.child(a, "left").filter(|&l| c.kind(l) == "field_expression")?;
        let arg = c.child(l, "argument")?;
        let arg = if c.kind(arg) == "parenthesized_expression" { *c.named(arg).first()? } else { arg };
        let name = match (c.kind(arg), c.op(l)) {
            ("identifier", Some(".")) => c.text(arg).to_string(),
            ("pointer_expression", Some("->")) if c.op(arg) == Some("&") => c.text(c.child(arg, "argument")?).to_string(),
            _ => return None,
        };
        Some((name, c.child(a, "right")?))
    };
    for blk in c.descendants(info.body) {
        if c.kind(blk) != "compound_statement" {
            continue;
        }
        let sibs: Vec<usize> = c.named(blk).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
        for (ri, &r) in sibs.iter().enumerate() {
            if c.kind(r) != "return_statement" {
                continue;
            }
            let Some(rv) = c.named(r).first().copied().filter(|&x| c.kind(x) == "identifier") else { continue };
            let obj = c.text(rv).to_string();
            let mut vals = vec![];
            let mut k = ri;
            while k > 0 {
                match store_obj(sibs[k - 1]) {
                    Some((o, v)) if o == obj => {
                        vals.insert(0, v);
                        k -= 1;
                    }
                    _ => break,
                }
            }
            if vals.len() < 2 {
                continue;
            }
            let Some(ty) = info.vars.get(&obj).filter(|v| !v.is_param).map(|v| v.ty.clone()) else { continue };
            let args: Vec<&str> = vals.iter().map(|&v| c.text(v)).collect();
            let text = format!("return {ty}({});", args.join(", "));
            let e = vec![Edit { start: c.nodes[sibs[k]].start, end: c.nodes[r].end, text }];
            if let Some(s) = apply(src, &e) {
                if Cst::parse(&s).errors <= c.errors {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// A single-use local read from memory (`x = *p; ...; *q = 0; *r = x;`) folded into its use
/// past intervening stores: the drafter keeps such loads where the target's schedule put them
/// (conservatively assuming the stores may alias), but the compiler's alias analysis may hoist
/// the load itself, and where the value is created decides its register.
pub fn sink_load_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    let idents = |n: usize| -> Vec<String> { c.descendants(n).into_iter().filter(|&d| c.kind(d) == "identifier").map(|d| c.text(d).to_string()).collect() };
    let has_call = |n: usize| c.descendants(n).into_iter().any(|d| matches!(c.kind(d), "call_expression" | "new_expression" | "delete_expression"));
    for blk in c.descendants(info.body) {
        if c.kind(blk) != "compound_statement" {
            continue;
        }
        let sibs: Vec<usize> = c.named(blk).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
        for (i, &t) in sibs.iter().enumerate() {
            let (name, val, decl) = match c.kind(t) {
                "expression_statement" => {
                    let Some(a) = c.named(t).first().copied().filter(|&a| c.kind(a) == "assignment_expression" && c.op(a) == Some("=")) else { continue };
                    let Some(l) = c.child(a, "left").filter(|&l| c.kind(l) == "identifier") else { continue };
                    let Some(r) = c.child(a, "right") else { continue };
                    (c.text(l).to_string(), r, false)
                }
                "declaration" => {
                    let ds: Vec<usize> = c.children_by_field(t, "declarator").collect();
                    if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
                        continue;
                    }
                    let Some((n, suf)) = crate::func::declarator_name(&c, ds[0]) else { continue };
                    let Some(v) = c.child(ds[0], "value") else { continue };
                    if !suf.is_empty() {
                        continue;
                    }
                    (n, v, true)
                }
                _ => continue,
            };
            if !info.vars.get(&name).is_some_and(|v| !v.is_param) || info.aliased.contains(&name) || has_call(val) {
                continue;
            }
            let reads = idents(val);
            // all mentions of the variable in the function
            let all: Vec<usize> = c.descendants(info.body).into_iter().filter(|&d| c.kind(d) == "identifier" && c.text(d) == name).collect();
            let later: Vec<(usize, usize)> = sibs[i + 1..]
                .iter()
                .enumerate()
                .flat_map(|(k, &s2)| c.descendants(s2).into_iter().filter(|&d| c.kind(d) == "identifier" && c.text(d) == name).map(move |d| (i + 1 + k, d)))
                .collect();
            let decl_elsewhere = if decl { 0 } else { 1 };
            if later.is_empty() || all.len() != later.len() + 1 + decl_elsewhere {
                continue;
            }
            let j = later.last().unwrap().0;
            let multi = later.len() > 1;
            if !multi && j == i + 1 {
                continue;
            }
            // nothing in between calls out or writes what the value reads (several uses: no
            // memory write before the last use's statement either, so every read sees the same
            // value); no use inside a loop
            let writes = |s2: usize, mem: bool| {
                c.descendants(s2).into_iter().any(|d| {
                    matches!(c.kind(d), "assignment_expression" | "update_expression")
                        && c.descendants(d).into_iter().nth(1).is_some_and(|x| (mem && c.kind(x) != "identifier") || (c.kind(x) == "identifier" && reads.contains(&c.text(x).to_string())))
                })
            };
            let upto = if multi { j + 1 } else { j };
            let blocked = sibs[i + 1..j].iter().any(|&s2| has_call(s2) || writes(s2, multi))
                || (multi && sibs[i + 1..upto].iter().any(|&s2| c.descendants(s2).into_iter().any(|d| matches!(c.kind(d), "for_statement" | "while_statement" | "do_statement"))));
            if blocked {
                continue;
            }
            let mut e: Vec<Edit> = later.iter().map(|&(_, u)| Edit::replace(&c, u, format!("({})", c.text(val)))).collect();
            e.push(Edit::replace(&c, t, String::new()));
            if let Some(s) = apply(src, &e) {
                if Cst::parse(&s).errors <= c.errors {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// `T x = a; if (c) { x = b; }` (or `x = a; ...`) as `T x; if (c) { x = b; } else { x = a; }`:
/// the default value computed only on its own path.
pub fn select_else_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mentions = |n: usize, x: &str| c.descendants(n).into_iter().any(|d| c.kind(d) == "identifier" && c.text(d) == x);
    let mut out = vec![];
    for blk in c.descendants(info.body) {
        if c.kind(blk) != "compound_statement" {
            continue;
        }
        let sibs: Vec<usize> = c.named(blk).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
        for w in sibs.windows(2) {
            let (p, s2) = (w[0], w[1]);
            if c.kind(s2) != "if_statement" || c.child(s2, "alternative").is_some() {
                continue;
            }
            // the default: (variable, value, declaration prefix if a declaration)
            let (x, a, prefix) = match c.kind(p) {
                "declaration" => {
                    let ds: Vec<usize> = c.children_by_field(p, "declarator").collect();
                    if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
                        continue;
                    }
                    let Some((n, _)) = crate::func::declarator_name(&c, ds[0]) else { continue };
                    let (Some(v), Some(d)) = (c.child(ds[0], "value"), c.child(ds[0], "declarator")) else { continue };
                    let pre = format!("{}{};", &src[c.nodes[p].start..c.nodes[ds[0]].start], c.text(d));
                    (n, v, Some(pre))
                }
                "expression_statement" => {
                    let Some(asg) = c.named(p).first().copied().filter(|&q| c.kind(q) == "assignment_expression" && c.op(q) == Some("=")) else { continue };
                    let Some(l) = c.child(asg, "left").filter(|&l| c.kind(l) == "identifier") else { continue };
                    let Some(r) = c.child(asg, "right") else { continue };
                    (c.text(l).to_string(), r, None)
                }
                _ => continue,
            };
            if !info.is_private(&x) || mentions(a, &x) {
                continue;
            }
            let Some(cons) = c.child(s2, "consequence") else { continue };
            let inner: Vec<usize> = if c.kind(cons) == "compound_statement" { c.named(cons).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect() } else { vec![cons] };
            let [only] = inner.as_slice() else { continue };
            let ok = c.kind(*only) == "expression_statement"
                && c.named(*only).first().is_some_and(|&q| c.kind(q) == "assignment_expression" && c.op(q) == Some("=") && c.child(q, "left").is_some_and(|l| c.text(l) == x));
            let Some(cond) = c.child(s2, "condition") else { continue };
            if !ok || mentions(cond, &x) {
                continue;
            }
            let at = c.nodes[p].start;
            let line_start = src[..at].rfind('\n').map(|q| q + 1).unwrap_or(0);
            let ind = &src[line_start..at];
            let head = match &prefix {
                Some(pre) => format!("{pre}\n{ind}"),
                None => String::new(),
            };
            let text = format!("{head}if {} {{\n{ind}    {}\n{ind}}} else {{\n{ind}    {x} = {};\n{ind}}}", c.text(cond), c.text(*only), c.text(a));
            if let Some(s) = apply(src, &[Edit { start: at, end: c.nodes[s2].end, text }]) {
                if Cst::parse(&s).errors <= c.errors {
                    out.push(s);
                }
            }
        }
    }
    out
}

struct Scored {
    src: String,
    fit: Fitness,
    ops: Vec<&'static str>,
}

fn interleave<T: Clone>(lists: Vec<Vec<T>>) -> Vec<T> {
    let longest = lists.iter().map(|l| l.len()).max().unwrap_or(0);
    let mut out = vec![];
    for i in 0..longest {
        for l in &lists {
            if let Some(c) = l.get(i) {
                out.push(c.clone());
            }
        }
    }
    out
}

/// Compile `cands` (in order) with `threads` workers until one is exact or the budget runs out.
fn score_all(scorer: &Scorer, cands: Vec<(String, Vec<&'static str>)>, threads: usize, compiles: &AtomicUsize, max: usize, deadline: std::time::Instant) -> Vec<Scored> {
    let next = AtomicUsize::new(0);
    let done = AtomicBool::new(false);
    let out = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                if done.load(Ordering::Relaxed) || compiles.load(Ordering::Relaxed) >= max || std::time::Instant::now() >= deadline {
                    break;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((src, ops)) = cands.get(i) else { break };
                let (e, ran) = scorer.eval(src);
                if ran {
                    compiles.fetch_add(1, Ordering::Relaxed);
                }
                if let Eval::Ok(f) = e {
                    if f.exact {
                        done.store(true, Ordering::Relaxed);
                    }
                    out.lock().unwrap().push((i, Scored { src: src.clone(), fit: f, ops: ops.clone() }));
                }
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort_by_key(|x| x.0);
    v.into_iter().map(|x| x.1).collect()
}

/// Repair a register-only mismatch of `src` (already scored as `fit`). Returns the best
/// candidate found if it is better than `src` (exact when possible); `None` otherwise or when the
/// diff is not register-only.
pub fn repair(scorer: &Scorer, src: &str, fit: &Fitness, tracer: Option<&crate::trace::Tracer>, cfg: &RepairConfig) -> Option<Repair> {
    if !register_only(fit) {
        return None;
    }
    let symbol = scorer.symbol.clone();
    let hints = crate::hints::target_hints(scorer.tf);
    let hints = (!hints.is_empty()).then_some(&hints);
    let compiles = AtomicUsize::new(0);
    let deadline = std::time::Instant::now() + cfg.max_time;
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(normalize(src));
    // Level 1: tracer-directed edits first, then every operator site.
    let mut cands: Vec<(String, Vec<&'static str>)> = vec![];
    if let Some(tr) = tracer {
        if let Ok(fixes) = tr.fixes(src, &symbol, scorer.tf) {
            for fx in &fixes {
                for c in crate::trace::apply_fix(src, &symbol, fx) {
                    if seen.insert(normalize(&c)) {
                        cands.push((c, vec!["trace_fix"]));
                    }
                }
            }
        }
    }
    // The draft with merged declarations compiles the same in the usual case and lets the
    // temp-inlining operators apply: then it is the only root; otherwise both are.
    let merged = merge_all_decls(src, &symbol).filter(|m| seen.insert(normalize(m)));
    let merged_same = merged.as_ref().is_some_and(|m| {
        let (e, ran) = scorer.eval(m);
        if ran {
            compiles.fetch_add(1, Ordering::Relaxed);
        }
        e.fitness().is_some_and(|f| f.penalty == fit.penalty && f.profile == fit.profile)
    });
    if let (Some(m), false) = (&merged, merged_same) {
        cands.push((m.clone(), vec!["merge_decl"]));
    }
    let a: Vec<(String, Vec<&'static str>)> = if merged_same { vec![] } else { neighbours(src, &symbol, hints, cfg.seeds, &mut seen).into_iter().map(|(c, o)| (c, vec![o])).collect() };
    let b: Vec<(String, Vec<&'static str>)> = match &merged {
        Some(m) => neighbours(m, &symbol, hints, cfg.seeds, &mut seen).into_iter().map(|(c, o)| (c, vec!["merge_decl", o])).collect(),
        None => vec![],
    };
    cands.extend(interleave(vec![a, b]));
    let verbose = std::env::var("MWDEC_REGFIX_VERBOSE").is_ok();
    if verbose {
        eprintln!("regfix: level 1: {} candidates", cands.len());
    }
    let t1 = std::time::Instant::now();
    let mut level = score_all(scorer, cands, cfg.threads, &compiles, cfg.max_compiles, deadline);
    let n1 = compiles.load(Ordering::Relaxed);
    let cheap_now = |t: std::time::Instant, n: usize| n >= 8 && t.elapsed().as_secs_f64() * 1000.0 / (n as f64) < cfg.cheap_ms;
    let mut cheap = cheap_now(t1, n1);
    let mut max_compiles = if cheap { cfg.cheap_max_compiles } else { cfg.max_compiles };
    if verbose {
        for s in &level {
            eprintln!("  {:?} penalty {} {:?}", s.ops, s.fit.penalty, s.fit.profile);
        }
    }
    let mut best: Option<Scored> = None;
    let consider = |best: &mut Option<Scored>, v: &[Scored]| {
        for s in v {
            if best.as_ref().map_or(true, |b| s.fit.better_than(&b.fit)) {
                *best = Some(Scored { src: s.src.clone(), fit: s.fit.clone(), ops: s.ops.clone() });
            }
        }
    };
    consider(&mut best, &level);
    // Level 2 (and 3 while budget remains): expand the best few register-only neighbours.
    if verbose {
        eprintln!("regfix: level 1: {n1} compiles in {:.0} ms (cheap: {cheap})", t1.elapsed().as_secs_f64() * 1000.0);
    }
    // progress: a strictly better neighbour earns the next level a budget of its own
    if best.as_ref().is_some_and(|b| b.fit.better_than(fit)) {
        max_compiles = max_compiles.max(n1 + cfg.max_compiles);
    }
    let mut depth = 0;
    while depth < 2 || (cheap && depth < 2 + cfg.cheap_levels) {
        depth += 1;
        if best.as_ref().is_some_and(|b| b.fit.exact) || compiles.load(Ordering::Relaxed) >= max_compiles {
            break;
        }
        level.retain(|s| register_only(&s.fit) && s.fit.penalty <= fit.penalty);
        level.sort_by(|a, b| a.fit.cmp_better(&b.fit));
        // Best first, one parent per last operator (equal-fitness neighbours of one operator are
        // usually the same kind of change at different sites).
        let mut parents: Vec<Scored> = vec![];
        let mut rest = vec![];
        for s in level.drain(..) {
            if parents.len() < cfg.beam && !parents.iter().any(|p| p.ops.last() == s.ops.last()) {
                parents.push(s);
            } else {
                rest.push(s);
            }
        }
        for s in rest {
            if parents.len() >= cfg.beam {
                break;
            }
            parents.push(s);
        }
        if parents.is_empty() {
            break;
        }
        let mut cands = vec![];
        // Interleave the parents' neighbours so each gets a share of the budget.
        let lists: Vec<Vec<(String, Vec<&'static str>)>> = parents
            .iter()
            .map(|p| {
                neighbours(&p.src, &symbol, hints, cfg.seeds, &mut seen)
                    .into_iter()
                    .map(|(c, o)| {
                        let mut ops = p.ops.clone();
                        ops.push(o);
                        (c, ops)
                    })
                    .collect()
            })
            .collect();
        // Neighbours of parents that improved on the start first (a hill-climb step), then the
        // plateau's.
        let (mut up, mut flat) = (vec![], vec![]);
        for (p, l) in parents.iter().zip(lists) {
            if p.fit.better_than(fit) {
                up.push(l);
            } else {
                flat.push(l);
            }
        }
        cands.extend(interleave(up));
        cands.extend(interleave(flat));
        if verbose {
            eprintln!("regfix: next level: {} candidates from {} parents, {} compiles so far", cands.len(), parents.len(), compiles.load(Ordering::Relaxed));
        }
        let (t, n0) = (std::time::Instant::now(), compiles.load(Ordering::Relaxed));
        level = score_all(scorer, cands, cfg.threads, &compiles, max_compiles, deadline);
        // (persistent compilers start during the first levels: judge each level on its own)
        if !cheap && cheap_now(t, compiles.load(Ordering::Relaxed) - n0) {
            cheap = true;
            max_compiles = cfg.cheap_max_compiles;
        }
        if verbose {
            eprintln!("regfix: level {}: {} compiles in {:.0} ms (cheap: {cheap})", depth + 1, compiles.load(Ordering::Relaxed) - n0, t.elapsed().as_secs_f64() * 1000.0);
        }
        consider(&mut best, &level);
    }
    let b = best?;
    b.fit.better_than(fit).then(|| Repair { src: b.src, fitness: b.fit, ops: b.ops, compiles: compiles.load(Ordering::Relaxed) })
}
