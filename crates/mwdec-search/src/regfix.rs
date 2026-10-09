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
    // macro: every local of one integer type flips signedness (last: displaces nothing above)
    "retype_all",
];

#[derive(Clone, Debug)]
pub struct RepairConfig {
    /// Real compiles at most (cache hits don't count).
    pub max_compiles: usize,
    /// Compile budget and extra levels when compiles turn out cheap (level 1 averaged under
    /// `cheap_ms` of wall time per compile). The default treats every context as cheap: a
    /// wall-clock rule made results depend on machine load.
    pub cheap_max_compiles: usize,
    pub cheap_levels: usize,
    pub cheap_ms: f64,
    /// Neighbours of level 1 expanded at level 2.
    pub beam: usize,
    /// Concurrent compiles.
    pub threads: usize,
    /// Enumeration seeds tried per operator (each picks a site at random; duplicates dropped).
    pub seeds: u64,
    /// Wall-clock safety net against a hanging compiler only: the compile budget decides the
    /// result (a tighter wall-clock cap made results depend on machine load).
    pub max_time: std::time::Duration,
}

impl Default for RepairConfig {
    fn default() -> Self {
        RepairConfig { max_compiles: 320, cheap_max_compiles: 320, cheap_levels: 1, cheap_ms: f64::INFINITY, beam: 4, threads: 4, seeds: 48, max_time: std::time::Duration::from_secs(120) }
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
    // (instructions moved far are aligned as deleted + inserted ones: a few of those in pairs too)
    !f.exact && f.size_delta == 0 && p.inserted == p.deleted && p.inserted + p.substituted <= 4 && p.stack == 0 && p.branch <= 2 && p.reloc == 0 && p.reg + p.reorder + p.other + p.inserted + p.branch > 0
}

/// Every distinct neighbour of `src` under [`REG_OPS`] (one operator application each), in
/// operator priority order. Deduplicated by normalized text against `seen`.
pub fn neighbours(src: &str, symbol: &str, hints: Option<&RegHints>, seeds: u64, seen: &mut HashSet<String>) -> Vec<(String, &'static str)> {
    let Some(p) = Parsed::new(src, symbol) else { return vec![] };
    let mut out = vec![];
    for c in const_param_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "const_param"));
        }
    }
    for c in cond_local_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "cond_local"));
        }
    }
    for c in param_copy_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "param_copy"));
        }
    }
    for c in hoist_loads_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "hoist_loads"));
        }
    }
    for c in loop_bound_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "loop_bound"));
        }
    }
    // every such value read in the loop at once
    let mut cur = src.to_string();
    for _ in 0..8 {
        match loop_bound_variants(&cur, symbol).into_iter().next() {
            Some(n) => cur = n,
            None => break,
        }
    }
    if cur != src && seen.insert(normalize(&cur)) {
        out.push((cur, "loop_bound"));
    }
    for c in store_reuse_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "store_reuse"));
        }
    }
    for c in volatile_poll_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "volatile_poll"));
        }
    }
    for c in hoist_common_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "hoist_common"));
        }
    }
    for c in accumulate_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "accumulate"));
        }
    }
    for c in member_ref_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "member_ref"));
        }
    }
    for c in mask_type_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "mask_type"));
        }
    }
    for c in fold_call_up_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "fold_call_up"));
        }
    }
    for c in narrow_ret_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "narrow_ret"));
        }
    }
    for c in ret_flag_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "ret_flag"));
        }
    }
    for c in widen_local_variants(src, symbol) {
        if seen.insert(normalize(&c)) {
            out.push((c, "widen_local"));
        }
    }
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
    // every such load sunk at once (one step each would need as many levels)
    let mut cur = src.to_string();
    for _ in 0..8 {
        match sink_load_variants(&cur, symbol).into_iter().next() {
            Some(n) => cur = n,
            None => break,
        }
    }
    if cur != src {
        if let Some(cc) = compound_all(&cur, symbol) {
            if seen.insert(normalize(&cc)) {
                out.push((cc, "sink_load"));
            }
        }
        if seen.insert(normalize(&cur)) {
            out.push((cur, "sink_load"));
        }
    }
    if let Some(cc) = compound_all(src, symbol) {
        if seen.insert(normalize(&cc)) {
            out.push((cc, "compound_assign"));
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
        if *name == "retype_all" && std::env::var_os("MWDEC_NO_MACROS").is_some() {
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
                // a hoisted right-hand side must not read what an earlier assignment of the run
                // writes (`p = p + 4; m = p;`: `m` would get the old `p`)
                let lhs: Vec<usize> = (i..i + k).filter_map(|j| c.named(sibs[j]).first().and_then(|&a| c.child(a, "left"))).collect();
                let clash = (1..k).any(|j| {
                    (0..j).any(|e| {
                        let l = lhs.get(e).copied();
                        l.is_some_and(|l| {
                            let lt = c.text(l);
                            let ident_read = c.kind(l) == "identifier" && c.descendants(rs[j]).into_iter().any(|n| c.kind(n) == "identifier" && c.text(n) == lt);
                            ident_read || c.text(rs[j]).contains(lt)
                        })
                    })
                });
                if clash {
                    continue;
                }
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
                    // (zero-argument methods other than setters are taken as reads)
                    "call_expression" => {
                        c.child(d, "function").is_some_and(|ff| c.kind(ff) != "field_expression" || c.child(ff, "field").is_some_and(|x| c.text(x).starts_with("Set")))
                            || c.child(d, "arguments").is_some_and(|a| !c.named(a).is_empty())
                    }
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
            "call_expression" => c.child(d, "function").is_some_and(|f| {
                (c.kind(f) == "field_expression" && c.child(f, "field").is_some_and(|x| c.text(x).starts_with("Get")))
                    || ["const_cast", "static_cast", "reinterpret_cast"].iter().any(|k| c.text(f).starts_with(k))
            }),
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
    // (zero-argument `Get*` accessors read, they don't write)
    let getter = |d: usize| {
        c.kind(d) == "call_expression"
            && c.child(d, "arguments").is_some_and(|a| c.named(a).is_empty())
            && c.child(d, "function").is_some_and(|f| c.kind(f) == "field_expression" && c.child(f, "field").is_some_and(|x| c.text(x).starts_with("Get")))
    };
    let has_call = |n: usize| c.descendants(n).into_iter().any(|d| matches!(c.kind(d), "call_expression" | "new_expression" | "delete_expression") && !getter(d));
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

/// Narrow integer locals (`unsigned short x;`) declared `int` / `unsigned int` instead: a value
/// the compiler knows is already extended (loaded with `lhz`, a constant) is not narrowed again
/// at each use, which changes both instructions and registers.
pub fn widen_local_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    const NARROW: &[&str] = &["unsigned short", "short", "unsigned char", "signed char", "char", "u8", "u16", "s8", "s16", "uchar", "ushort"];
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    for t in c.descendants(info.body) {
        if c.kind(t) != "declaration" {
            continue;
        }
        let Some(ty) = c.child(t, "type") else { continue };
        let tt = c.text(ty).split_whitespace().collect::<Vec<_>>().join(" ");
        if !NARROW.contains(&tt.as_str()) || c.children_by_field(t, "declarator").count() != 1 {
            continue;
        }
        let d = c.child(t, "declarator").unwrap();
        if crate::func::declarator_name(&c, d).is_none_or(|(_, suf)| !suf.is_empty()) {
            continue;
        }
        let signed = !(tt.starts_with("unsigned") || tt.starts_with('u'));
        for w in if signed { ["int", "unsigned int"] } else { ["unsigned int", "int"] } {
            if let Some(s) = apply(src, &[Edit::replace(&c, ty, w.to_string())]) {
                if Cst::parse(&s).errors <= c.errors {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// Every `L = L op e;` written `L op= e;` (one candidate): MWCC evaluates the compound form's
/// operands in another order, which matters for read-modify-writes of memory.
pub fn compound_all(src: &str, symbol: &str) -> Option<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let def = crate::func::find_target(&c, symbol)?;
    let body = c.child(def, "body")?;
    let mut edits = vec![];
    for a in c.descendants(body) {
        if c.kind(a) != "assignment_expression" || c.op(a) != Some("=") {
            continue;
        }
        let (Some(l), Some(r)) = (c.child(a, "left"), c.child(a, "right")) else { continue };
        let r = if c.kind(r) == "parenthesized_expression" { c.named(r).first().copied().unwrap_or(r) } else { r };
        if c.kind(r) != "binary_expression" {
            continue;
        }
        let Some(op) = c.op(r).filter(|o| matches!(*o, "|" | "&" | "^" | "+" | "-" | "*" | "<<" | ">>")) else { continue };
        let (Some(rl), Some(rr)) = (c.child(r, "left"), c.child(r, "right")) else { continue };
        let rl = if c.kind(rl) == "parenthesized_expression" { c.named(rl).first().copied().unwrap_or(rl) } else { rl };
        if c.text(rl).replace(' ', "") != c.text(l).replace(' ', "") {
            continue;
        }
        let rr = if c.kind(rr) == "parenthesized_expression" { c.named(rr).first().copied().unwrap_or(rr) } else { rr };
        edits.push(Edit::replace(&c, a, format!("{} {op}= {}", c.text(l), c.text(rr))));
    }
    if edits.is_empty() {
        return None;
    }
    let s = apply(src, &edits)?;
    (Cst::parse(&s).errors <= c.errors).then_some(s)
}

/// A boolean result kept in a local flag set before everything else (`bool r = K; ...; return r;`):
/// `return a && b;` -> `bool r = false; ... if (a && b) { r = true; } return r;` and
/// `if (c) { ...; return e; } return K;` -> `bool r = K; ... if (c) { ...; r = e; } return r;`.
/// The flag's constant is materialized up front (often in a callee-saved register).
pub fn ret_flag_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let ret_bool = c.child(def, "type").is_some_and(|t| matches!(c.text(t).trim(), "bool" | "BOOL" | "const bool"));
    if !ret_bool {
        return vec![];
    }
    let sibs: Vec<usize> = c.named(info.body).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
    let (Some(&first), Some(&last)) = (sibs.first(), sibs.last()) else { return vec![] };
    if c.kind(last) != "return_statement" {
        return vec![];
    }
    let Some(e) = c.named(last).first().copied() else { return vec![] };
    let name = info.fresh_name(&c, "result");
    let at = c.nodes[first].start;
    let line_start = src[..at].rfind('\n').map(|q| q + 1).unwrap_or(0);
    let ind = src[line_start..at].to_string();
    let lit = |n: usize| matches!(c.text(n).trim(), "true" | "false" | "0" | "1");
    let mut out = vec![];
    let mut push = |edits: Vec<Edit>| {
        if let Some(s) = apply(src, &edits) {
            if Cst::parse(&s).errors <= c.errors {
                out.push(s);
            }
        }
    };
    if !lit(e) {
        push(vec![
            Edit::insert(at, format!("bool {name} = false;\n{ind}")),
            Edit::replace(&c, last, format!("if ({}) {{\n{ind}    {name} = true;\n{ind}}}\n{ind}return {name};", c.text(e))),
        ]);
    } else if sibs.len() >= 2 {
        let prev = sibs[sibs.len() - 2];
        if c.kind(prev) == "if_statement" && c.child(prev, "alternative").is_none() {
            if let Some(cons) = c.child(prev, "consequence").filter(|&x| c.kind(x) == "compound_statement") {
                let inner: Vec<usize> = c.named(cons).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
                let rets = c.descendants(cons).into_iter().filter(|&d| c.kind(d) == "return_statement").count();
                if let Some(&r) = inner.last() {
                    if c.kind(r) == "return_statement" && rets == 1 {
                        if let Some(re) = c.named(r).first().copied() {
                            push(vec![
                                Edit::insert(at, format!("bool {name} = {};\n{ind}", c.text(e))),
                                Edit::replace(&c, r, format!("{name} = {};", c.text(re))),
                                Edit::replace(&c, last, format!("return {name};")),
                            ]);
                        }
                    }
                }
            }
        }
    }
    out
}

/// A returned comparison narrowed explicitly (`return (unsigned char)(a == b);`): the 0/1 is
/// then truncated to a byte (`extrwi`), as when the source returned it through a byte-sized type.
pub fn narrow_ret_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(body) = c.child(def, "body") else { return vec![] };
    let mut edits = vec![];
    for r in c.descendants(body) {
        if c.kind(r) != "return_statement" {
            continue;
        }
        let Some(e) = c.named(r).first().copied() else { continue };
        let e0 = if c.kind(e) == "parenthesized_expression" { c.named(e).first().copied().unwrap_or(e) } else { e };
        let cmp = c.kind(e0) == "binary_expression" && matches!(c.op(e0), Some("==" | "!=" | "<" | ">" | "<=" | ">="));
        if cmp {
            edits.push(Edit::replace(&c, e, format!("(unsigned char)({})", c.text(e0))));
        }
    }
    if edits.is_empty() {
        return vec![];
    }
    match apply(src, &edits) {
        Some(s) if Cst::parse(&s).errors <= c.errors => vec![s],
        _ => vec![],
    }
}

/// A call result kept in a local and used once inside arithmetic later (`t = f(); ...; *p = a *
/// t;`) computed at the call instead (`z = a * f(); ...; *p = z;`): the product is then created
/// right after the call, before whatever the statements in between create.
pub fn fold_call_up_variants(src: &str, symbol: &str) -> Vec<String> {
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
                    let Some(v) = c.child(ds[0], "value") else { continue };
                    if !suf.is_empty() {
                        continue;
                    }
                    (n, v)
                }
                "expression_statement" => {
                    let Some(a) = c.named(t).first().copied().filter(|&a| c.kind(a) == "assignment_expression" && c.op(a) == Some("=")) else { continue };
                    let Some(l) = c.child(a, "left").filter(|&l| c.kind(l) == "identifier") else { continue };
                    let Some(r) = c.child(a, "right") else { continue };
                    (c.text(l).to_string(), r)
                }
                _ => continue,
            };
            if c.kind(val) != "call_expression" || !info.vars.get(&name).is_some_and(|v| !v.is_param) {
                continue;
            }
            let uses: Vec<usize> = c.descendants(info.body).into_iter().filter(|&d| c.kind(d) == "identifier" && c.text(d) == name && c.nodes[d].start > c.nodes[val].end).collect();
            let [u] = uses.as_slice() else { continue };
            let Some(j) = sibs.iter().position(|&x| c.contains(x, *u)) else { continue };
            if j <= i + 1 {
                continue;
            }
            // the arithmetic around the use: binary operators over locals only
            let mut e = *u;
            while let Some(p) = c.parent(e) {
                if matches!(c.kind(p), "binary_expression" | "parenthesized_expression") {
                    e = p;
                } else {
                    break;
                }
            }
            if e == *u {
                continue;
            }
            let ids: Vec<String> = c.descendants(e).into_iter().filter(|&d| c.kind(d) == "identifier").map(|d| c.text(d).to_string()).collect();
            let pure = c.descendants(e).into_iter().all(|d| !matches!(c.kind(d), "call_expression" | "field_expression" | "pointer_expression" | "subscript_expression" | "assignment_expression" | "update_expression"));
            if !pure || ids.iter().any(|x| !info.vars.contains_key(x)) {
                continue;
            }
            // the other operands are not written between the call and the use
            let written = sibs[i..j].iter().any(|&s2| {
                c.descendants(s2).into_iter().any(|d| {
                    (matches!(c.kind(d), "assignment_expression" | "update_expression") && c.child(d, "left").or(c.child(d, "argument")).is_some_and(|l| ids.contains(&c.text(l).to_string()) && c.text(l) != name))
                        || (c.kind(d) == "init_declarator" && crate::func::declarator_name(&c, d).is_some_and(|x| ids.contains(&x.0) && x.0 != name))
                })
            });
            if written {
                continue;
            }
            let z = info.fresh_name(&c, "value");
            let etext = {
                let es = c.nodes[e].start;
                let us = c.nodes[*u].start - es;
                let ue = c.nodes[*u].end - es;
                let et = c.text(e);
                format!("{}{}{}", &et[..us], c.text(val), &et[ue..])
            };
            let at = c.nodes[t].start;
            let line_start = src[..at].rfind('\n').map(|q| q + 1).unwrap_or(0);
            let _ = line_start;
            let edits = vec![Edit::replace(&c, t, format!("__typeof__({etext}) {z} = {etext};")), Edit::replace(&c, e, z)];
            if let Some(s) = apply(src, &edits) {
                if Cst::parse(&s).errors <= c.errors {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// A 32-bit local only ever read through a 16/8-bit mask (`x & 0xffff`) declared that narrow
/// instead (`unsigned short x;`, masks dropped), and unsigned casts before a masked shift dropped
/// (`(unsigned int)x >> 8 & 255` -> `x >> 8 & 255`, the mask clears the sign bits anyway): the
/// drafts spell out extensions the source left to the types (one candidate each).
pub fn mask_type_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    let lit = |n: usize| parse_int(c.text(n));
    // narrow locals
    let mut edits = vec![];
    for t in c.descendants(info.body) {
        if c.kind(t) != "declaration" {
            continue;
        }
        let Some(ty) = c.child(t, "type") else { continue };
        let tt = c.text(ty).split_whitespace().collect::<Vec<_>>().join(" ");
        if !matches!(tt.as_str(), "int" | "unsigned int" | "u32" | "s32" | "uint") || c.children_by_field(t, "declarator").count() != 1 {
            continue;
        }
        let Some((name, suf)) = c.child(t, "declarator").and_then(|d| crate::func::declarator_name(&c, d)) else { continue };
        if !suf.is_empty() {
            continue;
        }
        // reads of the variable: all masked the same way?
        let reads: Vec<usize> = c
            .descendants(info.body)
            .into_iter()
            .filter(|&d| c.kind(d) == "identifier" && c.text(d) == name && c.parent(d).is_some_and(|p| !(c.kind(p) == "assignment_expression" && c.child(p, "left") == Some(d)) && c.kind(p) != "init_declarator" && c.kind(p) != "declaration"))
            .collect();
        let masks: Vec<Option<(usize, i64)>> = reads
            .iter()
            .map(|&d| {
                let p = c.parent(d)?;
                if c.kind(p) == "binary_expression" && c.op(p) == Some("&") {
                    let other = if c.child(p, "left") == Some(d) { c.child(p, "right")? } else { c.child(p, "left")? };
                    lit(other).map(|m| (p, m))
                } else {
                    None
                }
            })
            .collect();
        let masked: Vec<(usize, i64)> = masks.iter().flatten().copied().collect();
        if masked.is_empty() {
            continue;
        }
        let m = masked[0].1;
        let nt = match m {
            0xffff => "unsigned short",
            0xff => "unsigned char",
            _ => continue,
        };
        if !masked.iter().all(|x| x.1 == m) {
            continue;
        }
        edits.push(Edit::replace(&c, ty, nt.to_string()));
        for (p, _) in &masked {
            edits.push(Edit::replace(&c, *p, name.clone()));
        }
    }
    if !edits.is_empty() {
        if let Some(s) = apply(src, &edits) {
            if Cst::parse(&s).errors <= c.errors {
                out.push(s);
            }
        }
    }
    // unsigned casts before masked shifts
    let mut edits = vec![];
    for n in c.descendants(info.body) {
        if c.kind(n) != "binary_expression" || c.op(n) != Some("&") {
            continue;
        }
        let (Some(l), Some(r)) = (c.child(n, "left"), c.child(n, "right")) else { continue };
        let Some(m) = lit(r) else { continue };
        if c.kind(l) != "binary_expression" || c.op(l) != Some(">>") {
            continue;
        }
        let (Some(x), Some(k)) = (c.child(l, "left"), c.child(l, "right")) else { continue };
        let Some(k) = lit(k) else { continue };
        if c.kind(x) != "cast_expression" || !c.child(x, "type").is_some_and(|t| matches!(c.text(t).trim(), "unsigned int" | "u32" | "uint")) {
            continue;
        }
        if k <= 0 || k >= 32 || m < 0 || m >= (1i64 << (32 - k)) {
            continue;
        }
        let Some(v) = c.child(x, "value") else { continue };
        edits.push(Edit::replace(&c, x, c.text(v).to_string()));
    }
    if !edits.is_empty() {
        if let Some(s) = apply(src, &edits) {
            if Cst::parse(&s).errors <= c.errors {
                out.push(s);
            }
        }
    }
    out
}

/// A member object used several times (`this->mBounds.GetMax()`, `this->mBounds.GetMin()`)
/// through a reference bound at the top (`__typeof__(this->mBounds)& ref = this->mBounds;`): its
/// address is then computed once, into a register of its own.
pub fn member_ref_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let sibs: Vec<usize> = c.named(info.body).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
    let Some(&first) = sibs.first() else { return vec![] };
    // object-valued member accesses: `X` in `X.y` / `X.f()` with `X` a `->` member access of `this`
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for n in c.descendants(info.body) {
        if c.kind(n) != "field_expression" || c.op(n) != Some("->") {
            continue;
        }
        if !c.child(n, "argument").is_some_and(|a| c.kind(a) == "this") {
            continue;
        }
        if c.parent(n).is_some_and(|p| c.kind(p) == "field_expression" && c.op(p) == Some(".") && c.child(p, "argument") == Some(n)) {
            groups.entry(c.text(n).to_string()).or_default().push(n);
        }
    }
    let mut out = vec![];
    let at = c.nodes[first].start;
    let line_start = src[..at].rfind('\n').map(|q| q + 1).unwrap_or(0);
    let ind = &src[line_start..at];
    for (k, (x, ns)) in groups.iter().enumerate() {
        if ns.len() < 2 {
            continue;
        }
        let name = info.fresh_name(&c, &format!("ref{k}_"));
        let mut edits = vec![Edit::insert(at, format!("__typeof__({x})& {name} = {x};\n{ind}"))];
        for &n in ns {
            edits.push(Edit::replace(&c, n, name.clone()));
        }
        if let Some(s) = apply(src, &edits) {
            if Cst::parse(&s).errors <= c.errors {
                out.push(s);
            }
        }
    }
    out
}

/// A sum of three or more terms (`a + b + c`, or a vector's `v.MagSquared()` written out with its
/// `GetX/Y/Z` accessors) accumulated in a local before the statement (`T s = a; s += b; s += c;`):
/// the front end keeps the accumulation order instead of reassociating the sum.
pub fn accumulate_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    fn terms(c: &Cst, n: usize, out: &mut Vec<usize>) {
        let n0 = if c.kind(n) == "parenthesized_expression" { c.named(n).first().copied().unwrap_or(n) } else { n };
        if c.kind(n0) == "binary_expression" && c.op(n0) == Some("+") {
            if let (Some(l), Some(r)) = (c.child(n0, "left"), c.child(n0, "right")) {
                terms(c, l, out);
                terms(c, r, out);
                return;
            }
        }
        out.push(n0);
    }
    let mut out = vec![];
    let mut k = 0;
    for n in c.descendants(info.body) {
        // (sum expression, its terms as text)
        let ts: Vec<String> = if c.kind(n) == "binary_expression" && c.op(n) == Some("+") && !c.parent(n).is_some_and(|p| (c.kind(p) == "binary_expression" && c.op(p) == Some("+")) || c.kind(p) == "parenthesized_expression") {
            let mut t = vec![];
            terms(&c, n, &mut t);
            if t.len() < 3 {
                continue;
            }
            t.iter().map(|&x| c.text(x).to_string()).collect()
        } else if c.kind(n) == "call_expression" && c.child(n, "arguments").is_some_and(|a| c.named(a).is_empty()) {
            let Some(f) = c.child(n, "function").filter(|&f| c.kind(f) == "field_expression") else { continue };
            if !c.child(f, "field").is_some_and(|x| c.text(x) == "MagSquared") {
                continue;
            }
            let Some(o) = c.child(f, "argument") else { continue };
            if !matches!(c.kind(o), "identifier" | "this" | "field_expression" | "parenthesized_expression") {
                continue;
            }
            let acc = format!("{}{}", c.text(o), c.op(f).unwrap_or("."));
            ["GetX", "GetY", "GetZ"].iter().map(|g| format!("{acc}{g}() * {acc}{g}()")).collect()
        } else {
            continue;
        };
        let Some(stmt) = c.ancestors(n).into_iter().find(|&x| crate::func::is_stmt(c.kind(x)) && c.parent(x).is_some_and(|p| c.kind(p) == "compound_statement")) else { continue };
        if matches!(c.kind(stmt), "while_statement" | "for_statement" | "do_statement") {
            continue;
        }
        let at = c.nodes[stmt].start;
        let line_start = src[..at].rfind('\n').map(|q| q + 1).unwrap_or(0);
        let ind = &src[line_start..at];
        let name = info.fresh_name(&c, &format!("sum{k}_"));
        k += 1;
        let mut decl = format!("__typeof__({}) {name} = {};\n{ind}", c.text(n), ts[0]);
        for t in &ts[1..] {
            decl.push_str(&format!("{name} += {t};\n{ind}"));
        }
        let edits = vec![Edit::insert(at, decl), Edit::replace(&c, n, name)];
        if let Some(s) = apply(src, &edits) {
            if Cst::parse(&s).errors <= c.errors {
                out.push(s);
            }
        }
    }
    out
}

/// An expression both arms of an `if`/`else` compute (`a[(u8)x] = (u8)x;` / `b[..] = (u8)x;`)
/// named once before the `if` (`__typeof__((u8)x) t = (u8)x;`): the value is then created ahead
/// of the branch, as when the source kept it in a local.
pub fn hoist_common_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    let candidate = |n: usize| -> bool {
        matches!(c.kind(n), "cast_expression" | "binary_expression")
            && c.descendants(n).into_iter().all(|d| !matches!(c.kind(d), "call_expression" | "assignment_expression" | "update_expression" | "pointer_expression" | "subscript_expression" | "field_expression"))
            && c.descendants(n).into_iter().any(|d| c.kind(d) == "identifier")
            && !c.parent(n).is_some_and(|p| c.kind(p) == "assignment_expression" && c.child(p, "left") == Some(n))
    };
    for st in c.descendants(info.body) {
        if c.kind(st) != "if_statement" || !c.parent(st).is_some_and(|p| c.kind(p) == "compound_statement") {
            continue;
        }
        let (Some(cons), Some(alt)) = (c.child(st, "consequence"), c.child(st, "alternative")) else { continue };
        let in_cons: Vec<usize> = c.descendants(cons).into_iter().filter(|&n| candidate(n)).collect();
        let in_alt: Vec<usize> = c.descendants(alt).into_iter().filter(|&n| candidate(n)).collect();
        let mut done: Vec<String> = vec![];
        for &a in &in_cons {
            let t = c.text(a).to_string();
            if done.contains(&t) || !in_alt.iter().any(|&b| c.text(b) == t) {
                continue;
            }
            // no identifier of it written inside the if
            let ids: Vec<String> = c.descendants(a).into_iter().filter(|&d| c.kind(d) == "identifier").map(|d| c.text(d).to_string()).collect();
            let written = c.descendants(st).into_iter().any(|d| {
                matches!(c.kind(d), "assignment_expression" | "update_expression") && c.child(d, "left").or(c.child(d, "argument")).is_some_and(|l| ids.contains(&c.text(l).to_string()))
            });
            if written {
                continue;
            }
            done.push(t.clone());
            // outermost occurrences only (inside the if)
            let occ: Vec<usize> = c.descendants(st).into_iter().filter(|&n| c.text(n) == t && candidate(n)).collect();
            let occ: Vec<usize> = occ.iter().copied().filter(|&n| !occ.iter().any(|&m| m != n && c.contains(m, n))).collect();
            let name = info.fresh_name(&c, "common");
            let at = c.nodes[st].start;
            let line_start = src[..at].rfind('\n').map(|q| q + 1).unwrap_or(0);
            let ind = &src[line_start..at];
            let mut edits = vec![Edit::insert(at, format!("__typeof__({t}) {name} = {t};\n{ind}"))];
            for n in occ {
                edits.push(Edit::replace(&c, n, name.clone()));
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

/// C89 form of a candidate: declarations after the first statement of a block move to the
/// block's start (`T x = v;` becomes `T x;` there and `x = v;` in place), as C units require.
pub fn c89_decls(src: &str, symbol: &str) -> Option<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let def = crate::func::find_target(&c, symbol)?;
    let body = c.child(def, "body")?;
    let mut edits = vec![];
    let mut blocks = c.descendants(body);
    if !blocks.contains(&body) {
        blocks.insert(0, body);
    }
    for blk in blocks {
        if c.kind(blk) != "compound_statement" {
            continue;
        }
        let sibs: Vec<usize> = c.named(blk).into_iter().filter(|&n| c.kind(n) != "comment").collect();
        let Some(first_stmt) = sibs.iter().position(|&n| c.kind(n) != "declaration") else { continue };
        let insert_at = c.nodes[sibs[first_stmt]].start;
        let line_start = src[..insert_at].rfind('\n').map(|q| q + 1).unwrap_or(0);
        let ind = &src[line_start..insert_at];
        let mut hoisted = String::new();
        for &d in &sibs[first_stmt + 1..] {
            if c.kind(d) != "declaration" {
                continue;
            }
            let ds: Vec<usize> = c.children_by_field(d, "declarator").collect();
            if ds.len() != 1 {
                return None;
            }
            let prefix = &src[c.nodes[d].start..c.nodes[ds[0]].start];
            if c.kind(ds[0]) == "init_declarator" {
                let (Some(dd), Some(v)) = (c.child(ds[0], "declarator"), c.child(ds[0], "value")) else { return None };
                if c.kind(dd) == "reference_declarator" || matches!(c.kind(v), "initializer_list" | "argument_list") {
                    return None;
                }
                let (name, _) = crate::func::declarator_name(&c, ds[0])?;
                hoisted.push_str(&format!("{prefix}{};\n{ind}", c.text(dd)));
                edits.push(Edit::replace(&c, d, format!("{name} = {};", c.text(v))));
            } else {
                hoisted.push_str(&format!("{}\n{ind}", c.text(d)));
                edits.push(Edit::replace(&c, d, String::new()));
            }
        }
        if !hoisted.is_empty() {
            edits.push(Edit::insert(insert_at, hoisted));
        }
    }
    if edits.is_empty() {
        return Some(src.to_string());
    }
    let s = apply(src, &edits)?;
    (Cst::parse(&s).errors <= c.errors).then_some(s)
}

/// A loop that only polls memory (`while (g.flag != 0) {}`) reading it through a volatile lvalue
/// (`*(volatile __typeof__(g.flag)*)&g.flag`): the target re-reads it every iteration.
pub fn volatile_poll_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut edits = vec![];
    for l in c.descendants(info.body) {
        if !matches!(c.kind(l), "while_statement" | "do_statement") {
            continue;
        }
        let Some(b) = c.child(l, "body") else { continue };
        let empty = c.kind(b) == "compound_statement" && c.named(b).iter().all(|&n| c.kind(n) == "comment") || c.kind(b) == "expression_statement" && c.named(b).is_empty();
        let Some(cond) = c.child(l, "condition") else { continue };
        if !empty {
            continue;
        }
        let reads: Vec<usize> = c.descendants(cond).into_iter().filter(|&n| matches!(c.kind(n), "field_expression" | "subscript_expression" | "pointer_expression") && !c.text(n).contains("volatile")).collect();
        let outer: Vec<usize> = reads.iter().copied().filter(|&n| !reads.iter().any(|&m| m != n && c.contains(m, n))).collect();
        for n in outer {
            if c.kind(n) == "pointer_expression" && c.op(n) != Some("*") {
                continue;
            }
            let t = c.text(n);
            edits.push(Edit::replace(&c, n, format!("(*(volatile __typeof__({t})*)&{t})")));
        }
    }
    if edits.is_empty() {
        return vec![];
    }
    match apply(src, &edits) {
        Some(s) if Cst::parse(&s).errors <= c.errors => vec![s],
        _ => vec![],
    }
}

/// A value stored to memory right after it is computed and used again later (`t = f(); g = t;
/// h(t);`) read back from where it was stored instead (`g = f(); h(g);`): the compiler forwards
/// the stored value, so the call result stays where the store put it.
pub fn store_reuse_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    let assign = |t: usize| -> Option<(usize, usize)> {
        if c.kind(t) != "expression_statement" {
            return None;
        }
        let a = c.named(t).first().copied().filter(|&a| c.kind(a) == "assignment_expression" && c.op(a) == Some("="))?;
        Some((c.child(a, "left")?, c.child(a, "right")?))
    };
    for blk in c.descendants(info.body) {
        if c.kind(blk) != "compound_statement" {
            continue;
        }
        let sibs: Vec<usize> = c.named(blk).into_iter().filter(|&n| crate::func::is_stmt(c.kind(n))).collect();
        for i in 0..sibs.len().saturating_sub(2) {
            // `x = E;` (or `T x = E;`) then `L = x;`
            let (name, e, decl) = match c.kind(sibs[i]) {
                "declaration" => {
                    let ds: Vec<usize> = c.children_by_field(sibs[i], "declarator").collect();
                    if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
                        continue;
                    }
                    let (Some((n, _)), Some(v)) = (crate::func::declarator_name(&c, ds[0]), c.child(ds[0], "value")) else { continue };
                    (n, v, true)
                }
                _ => {
                    let Some((l, r)) = assign(sibs[i]) else { continue };
                    if c.kind(l) != "identifier" {
                        continue;
                    }
                    (c.text(l).to_string(), r, false)
                }
            };
            let Some((l2, r2)) = assign(sibs[i + 1]) else { continue };
            let r2 = if c.kind(r2) == "parenthesized_expression" { c.named(r2).first().copied().unwrap_or(r2) } else { r2 };
            if c.text(r2) != name || (c.kind(l2) == "identifier" && info.vars.contains_key(c.text(l2))) || c.descendants(l2).into_iter().any(|d| matches!(c.kind(d), "call_expression")) {
                continue;
            }
            if !info.vars.get(&name).is_some_and(|v| !v.is_param) {
                continue;
            }
            // later uses (all in one later statement, nothing in between writing memory or calling)
            let uses: Vec<usize> = c.descendants(info.body).into_iter().filter(|&d| c.kind(d) == "identifier" && c.text(d) == name && c.nodes[d].start > c.nodes[sibs[i + 1]].end).collect();
            if uses.is_empty() {
                continue;
            }
            let Some(j) = sibs.iter().position(|&x| c.contains(x, uses[uses.len() - 1])) else { continue };
            if uses.iter().any(|&u| !c.contains(sibs[j], u)) {
                continue;
            }
            let between = &sibs[i + 2..j];
            if between.iter().any(|&s2| c.descendants(s2).into_iter().any(|d| matches!(c.kind(d), "call_expression" | "assignment_expression" | "update_expression"))) {
                continue;
            }
            let lt = c.text(l2).to_string();
            let mut edits = vec![Edit::replace(&c, sibs[i], String::new()), Edit::replace(&c, sibs[i + 1], format!("{lt} = {};", c.text(e)))];
            for &u in &uses {
                edits.push(Edit::replace(&c, u, lt.clone()));
            }
            let _ = decl;
            if let Some(s) = apply(src, &edits) {
                if Cst::parse(&s).errors <= c.errors {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// A loop bound read into a local before the loop and used only in the loop condition
/// (`n = v.size(); for (i = 0; i < n; i++)`) read in the condition itself (`i < v.size()`): the
/// compiler hoists it anyway, but creates the value at another point.
pub fn loop_bound_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    for t in c.descendants(info.body) {
        let (name, val) = match c.kind(t) {
            "declaration" => {
                let ds: Vec<usize> = c.children_by_field(t, "declarator").collect();
                if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
                    continue;
                }
                let (Some((n, _)), Some(v)) = (crate::func::declarator_name(&c, ds[0]), c.child(ds[0], "value")) else { continue };
                (n, v)
            }
            "expression_statement" => {
                let Some(a) = c.named(t).first().copied().filter(|&a| c.kind(a) == "assignment_expression" && c.op(a) == Some("=")) else { continue };
                let (Some(l), Some(r)) = (c.child(a, "left"), c.child(a, "right")) else { continue };
                if c.kind(l) != "identifier" {
                    continue;
                }
                (c.text(l).to_string(), r)
            }
            _ => continue,
        };
        if !info.vars.get(&name).is_some_and(|v| !v.is_param) || c.descendants(val).into_iter().any(|d| matches!(c.kind(d), "assignment_expression" | "update_expression")) {
            continue;
        }
        // all reads inside one loop that writes no memory and calls nothing but accessors (the
        // value is loop-invariant), or a single read in its condition
        let reads: Vec<usize> = c.descendants(info.body).into_iter().filter(|&d| c.kind(d) == "identifier" && c.text(d) == name && c.nodes[d].start > c.nodes[val].end).collect();
        let Some(&u0) = reads.first() else { continue };
        let Some(lp) = c.ancestors(u0).into_iter().find(|&a| matches!(c.kind(a), "for_statement" | "while_statement" | "do_statement")) else { continue };
        if reads.iter().any(|&u| !c.contains(lp, u)) || c.nodes[t].end > c.nodes[lp].start {
            continue;
        }
        let in_cond = reads.len() == 1 && c.child(lp, "condition").is_some_and(|cd| c.contains(cd, u0));
        let pure_loop = c.descendants(lp).into_iter().all(|d| match c.kind(d) {
            "assignment_expression" | "update_expression" => c.descendants(d).into_iter().nth(1).is_some_and(|x| c.kind(x) == "identifier" && info.vars.contains_key(c.text(x)) && c.text(x) != name),
            "call_expression" => c.child(d, "function").is_some_and(|f| c.kind(f) == "field_expression" && c.child(f, "arguments").is_none() && c.child(d, "arguments").is_some_and(|a| c.named(a).is_empty())),
            _ => true,
        });
        let val_ids: Vec<String> = c.descendants(val).into_iter().filter(|&d| c.kind(d) == "identifier").map(|d| c.text(d).to_string()).collect();
        let ids_written = c.descendants(lp).into_iter().any(|d| matches!(c.kind(d), "assignment_expression" | "update_expression") && c.descendants(d).into_iter().nth(1).is_some_and(|x| val_ids.contains(&c.text(x).to_string())));
        if !(in_cond || pure_loop) || ids_written {
            continue;
        }
        let mut edits = vec![Edit::replace(&c, t, String::new())];
        for &u in &reads {
            edits.push(Edit::replace(&c, u, c.text(val).to_string()));
        }
        if let Some(s) = apply(src, &edits) {
            if Cst::parse(&s).errors <= c.errors {
                out.push(s);
            }
        }
    }
    out
}

/// The loads of a short-circuit condition (`a->x == 0 || a->y == 0 || ...`) read into locals
/// before it (`T l0 = a->x; T l1 = a->y; ...`): every value is then loaded up front, as when the
/// source copied the fields first.
pub fn hoist_loads_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    for st in c.descendants(info.body) {
        let cond = match c.kind(st) {
            "if_statement" => c.child(st, "condition"),
            "return_statement" => c.named(st).first().copied(),
            _ => None,
        };
        let Some(cond) = cond else { continue };
        if !c.parent(st).is_some_and(|p| c.kind(p) == "compound_statement") {
            continue;
        }
        let logical = c.descendants(cond).into_iter().any(|d| c.kind(d) == "binary_expression" && matches!(c.op(d), Some("||" | "&&")));
        if !logical || c.descendants(cond).into_iter().any(|d| matches!(c.kind(d), "call_expression" | "assignment_expression" | "update_expression")) {
            continue;
        }
        let loads: Vec<usize> = c
            .descendants(cond)
            .into_iter()
            .filter(|&d| (c.kind(d) == "pointer_expression" && c.op(d) == Some("*")) || c.kind(d) == "field_expression" || c.kind(d) == "subscript_expression")
            .collect();
        let outer: Vec<usize> = loads.iter().copied().filter(|&n| !loads.iter().any(|&m| m != n && c.contains(m, n))).collect();
        if outer.len() < 2 {
            continue;
        }
        let at = c.nodes[st].start;
        let line_start = src[..at].rfind('\n').map(|q| q + 1).unwrap_or(0);
        let ind = &src[line_start..at];
        let mut decls = String::new();
        let mut edits = vec![];
        let mut names: Vec<(String, String)> = vec![];
        for n in outer {
            let t = c.text(n).to_string();
            let name = match names.iter().find(|x| x.0 == t) {
                Some(x) => x.1.clone(),
                None => {
                    let nm = info.fresh_name(&c, &format!("load{}_", names.len()));
                    decls.push_str(&format!("__typeof__({t}) {nm} = {t};\n{ind}"));
                    names.push((t.clone(), nm.clone()));
                    nm
                }
            };
            edits.push(Edit::replace(&c, n, name));
        }
        edits.push(Edit::insert(at, decls));
        if let Some(s) = apply(src, &edits) {
            if Cst::parse(&s).errors <= c.errors {
                out.push(s);
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

/// Whether `symbol` is a C++-mangled name (`name__<class>F...`, `name__F...`): its parameter types
/// are part of the symbol and cannot change.
fn is_mangled(symbol: &str) -> bool {
    let s = symbol.strip_prefix("__").unwrap_or(symbol);
    match s.find("__") {
        Some(i) => s[i + 2..].starts_with(|c: char| c.is_ascii_digit() || c == 'F' || c == 'Q' || c == 'C'),
        None => false,
    }
}

/// Parameters of a function whose symbol does not encode them (`extern "C"`, C units) declared
/// pointer-to-const when they are only read through: the pointee of a pointer-to-const parameter
/// is a known object to the compiler's alias analysis, so loads through it are no longer ordered
/// after stores through other pointers or after the prologue's register saves (they move up into
/// the first cycles). (A `const T*` whose class has a `mutable` non-pointer member gets no such
/// benefit; the compile decides.) Complements the lifter's rule, which declares integer
/// parameters used only as load bases `const char*` (variant `param.const_pointers`: all of
/// them): here an integer parameter used as an address *and* as a value becomes `const void*`
/// (the value uses read `(int)p`), a `T*` parameter `const T*`; one variant with all of those,
/// and one per parameter (address-only integers included). A parameter that is stored through
/// or passed on as a pointer is left alone.
pub fn const_param_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    if is_mangled(symbol) {
        return vec![];
    }
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut params: Vec<&crate::func::Var> = info.vars.values().filter(|v| v.is_param).collect();
    params.sort_by_key(|v| v.decl);
    let mut per_param: Vec<Vec<Edit>> = vec![];
    // per parameter: also in the combined variant (not covered by the lifter's rule)
    let mut combined: Vec<bool> = vec![];
    for v in params {
        let name = v.name.as_str();
        let ty = v.ty.trim();
        let int_like = matches!(ty, "int" | "unsigned int" | "s32" | "u32" | "long" | "unsigned long");
        let ptr = ty.ends_with('*') && !ty.starts_with("const ") && !ty.contains("**") && !ty.contains('(');
        if !int_like && !ptr {
            continue;
        }
        let uses: Vec<usize> = c.descendants(info.body).into_iter().filter(|&n| c.kind(n) == "identifier" && c.text(n) == name).collect();
        // never assigned and its address never taken
        let written = uses.iter().any(|&n| {
            c.parent(n).is_some_and(|p| match c.kind(p) {
                "assignment_expression" => c.child(p, "left") == Some(n),
                "update_expression" => true,
                "pointer_expression" => c.op(p) == Some("&"),
                _ => false,
            })
        });
        // ... nor stored through, nor passed on as a pointer
        let stored = uses.iter().any(|&n| {
            c.ancestors(n).into_iter().any(|a| c.kind(a) == "assignment_expression" && c.child(a, "left").is_some_and(|l| c.contains(l, n)))
        });
        let passed = ptr && uses.iter().any(|&n| c.parent(n).is_some_and(|p| c.kind(p) == "argument_list"));
        if written || stored || passed {
            continue;
        }
        let cast_parent = |n: usize| c.parent(n).filter(|&p| c.kind(p) == "cast_expression");
        let pointer_cast = |n: usize| cast_parent(n).and_then(|p| c.child(p, "type")).is_some_and(|t| c.text(t).trim_end().ends_with('*'));
        let mut edits = vec![];
        let mut in_combined = true;
        if int_like {
            if !uses.iter().any(|&n| pointer_cast(n)) {
                continue;
            }
            for &n in &uses {
                if cast_parent(n).is_none() {
                    edits.push(Edit::replace(&c, n, format!("(int){name}")));
                }
            }
            // address-only: the lifter's rule (and its variant) already declares it
            in_combined = !edits.is_empty();
            let Some(d) = c.child(v.decl, "type") else { continue };
            edits.push(Edit::replace(&c, d, "const void*"));
        } else {
            let reads = uses.iter().any(|&n| {
                c.parent(n).is_some_and(|p| matches!(c.kind(p), "field_expression" | "subscript_expression" | "pointer_expression"))
            });
            if !reads {
                continue;
            }
            let Some(d) = c.child(v.decl, "type") else { continue };
            edits.push(Edit::replace(&c, d, format!("const {}", c.text(d))));
        }
        per_param.push(edits);
        combined.push(in_combined);
    }
    if per_param.is_empty() {
        return vec![];
    }
    let mut out = vec![];
    let ok = |s: Option<String>| s.filter(|s| Cst::parse(s).errors <= c.errors);
    let all: Vec<Edit> = per_param.iter().zip(&combined).filter(|(_, &k)| k).flat_map(|(e, _)| e.clone()).collect();
    if !all.is_empty() {
        if let Some(s) = ok(apply(src, &all)) {
            out.push(s);
        }
    }
    if per_param.len() > 1 || out.is_empty() {
        for e in per_param.iter().take(4) {
            if let Some(s) = ok(apply(src, e)) {
                out.push(s);
            }
        }
    }
    out
}

/// A parameter the function updates (`p = p + 36` in a loop) read through a local copy instead
/// (`T it = p; ... it = it + 36`): the copy is a named local with a later virtual register than
/// every parameter, so it is coloured first (takes the highest callee-saved register) where the
/// updated parameter would come after the other parameters.
pub fn param_copy_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut params: Vec<&crate::func::Var> = info.vars.values().filter(|v| v.is_param && crate::func::is_scalar_type(&v.ty)).collect();
    params.sort_by_key(|v| v.decl);
    let mut out = vec![];
    for v in params {
        let name = v.name.as_str();
        let uses: Vec<usize> = c.descendants(info.body).into_iter().filter(|&n| c.kind(n) == "identifier" && c.text(n) == name).collect();
        let updated = uses.iter().any(|&n| {
            c.parent(n).is_some_and(|p| match c.kind(p) {
                "assignment_expression" => c.child(p, "left") == Some(n),
                "update_expression" => true,
                _ => false,
            })
        });
        let address_taken = uses.iter().any(|&n| c.parent(n).is_some_and(|p| c.kind(p) == "pointer_expression" && c.op(p) == Some("&")));
        if !updated || address_taken {
            continue;
        }
        let local = info.fresh_name(&c, &format!("{name}_it"));
        let mut edits: Vec<Edit> = uses.iter().map(|&n| Edit::replace(&c, n, local.clone())).collect();
        edits.push(Edit::insert(c.nodes[info.body].start + 1, format!("\n    {} {local} = {name};", v.ty)));
        if let Some(s) = apply(src, &edits).filter(|s| Cst::parse(s).errors <= c.errors) {
            out.push(s);
        }
    }
    out
}

/// A pointer tested by an `if` and used again in its branch (`if (!p.null()) f(p.get());`,
/// `if (o->m) o->m->g();`) held in a named local (`__typeof__(p.get()) t = p.get(); if (t)
/// f(t);`): the named value is coloured with the locals instead of as a common-subexpression
/// temporary. C++ only (a declaration before the statement).
pub fn cond_local_variants(src: &str, symbol: &str) -> Vec<String> {
    use crate::cst::{apply, Cst, Edit};
    let c = Cst::parse(src);
    let Some(def) = crate::func::find_target(&c, symbol) else { return vec![] };
    let Some(info) = crate::func::FuncInfo::analyze(&c, def) else { return vec![] };
    let mut out = vec![];
    for st in c.descendants(info.body) {
        if c.kind(st) != "if_statement" {
            continue;
        }
        let (Some(cond), Some(cons)) = (c.child(st, "condition"), c.child(st, "consequence")) else { continue };
        let ct = crate::cst::strip_parens(c.text(cond).trim()).trim().to_string();
        // the pointer the condition tests
        let key = if let Some(x) = ct.strip_prefix('!').and_then(|x| x.trim().strip_suffix(".null()")) {
            format!("{x}.get()")
        } else {
            let p = ct.strip_suffix(" != 0").or_else(|| ct.strip_suffix(" != nullptr")).unwrap_or(&ct).trim().to_string();
            let simple = p.ends_with(".get()") || p.chars().all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '.' | '-' | '>'));
            if !simple || !(p.contains("->") || p.contains('.')) {
                continue;
            }
            p
        };
        if key.contains('(') && !key.ends_with(".get()") || key.matches('(').count() > 1 {
            continue;
        }
        let body = c.text(cons);
        if !body.contains(&key) {
            continue;
        }
        let name = info.fresh_name(&c, "ptr");
        let at = c.nodes[st].start;
        let line_start = src[..at].rfind('\n').map(|q| q + 1).unwrap_or(0);
        let ind = &src[line_start..at];
        // (its own type, or `const void*` when the branch reads through it with casts only: the
        // pointee type changes how the reads are scheduled)
        let cast_only = body.matches(key.as_str()).count() == body.matches(format!("*){key}").as_str()).count();
        let mut types = vec![format!("__typeof__({key})")];
        if cast_only {
            types.push("const void*".to_string());
        }
        for ty in types {
            let edits = vec![
                Edit { start: at, end: at, text: format!("{ty} {name} = {key};\n{ind}") },
                Edit::replace(&c, cond, format!("({name})")),
                Edit::replace(&c, cons, body.replace(&key, &name)),
            ];
            if let Some(s) = apply(src, &edits).filter(|s| Cst::parse(s).errors <= c.errors) {
                out.push(s);
            }
        }
    }
    out
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
    // C units: generated declarations go to the start of their block
    let c_mode = scorer.ctx.cflags.iter().any(|f| f == "-lang=c" || f == "-lang=c99") || scorer.ctx.tu_name.as_deref().is_some_and(|n| n.ends_with(".c"));
    let c89 = |v: Vec<(String, &'static str)>| -> Vec<(String, &'static str)> {
        if !c_mode {
            return v;
        }
        v.into_iter().filter_map(|(c, o)| c89_decls(&c, &scorer.symbol).map(|x| (x, o))).collect()
    };
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
    let a: Vec<(String, Vec<&'static str>)> = if merged_same { vec![] } else { c89(neighbours(src, &symbol, hints, cfg.seeds, &mut seen)).into_iter().map(|(c, o)| (c, vec![o])).collect() };
    let b: Vec<(String, Vec<&'static str>)> = match &merged {
        Some(m) => c89(neighbours(m, &symbol, hints, cfg.seeds, &mut seen)).into_iter().map(|(c, o)| (c, vec!["merge_decl", o])).collect(),
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
    let cheap_now = |t: std::time::Instant, n: usize| cfg.cheap_ms.is_infinite() || n >= 8 && t.elapsed().as_secs_f64() * 1000.0 / (n as f64) < cfg.cheap_ms;
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
                c89(neighbours(&p.src, &symbol, hints, cfg.seeds, &mut seen))
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
