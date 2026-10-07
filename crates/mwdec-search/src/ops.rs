//! Source mutation operators.
//!
//! Every operator looks for applicable sites inside the target function body, picks one at
//! random and returns text edits. Operators aim to preserve semantics (best effort, with a
//! conservative side-effect model from [`crate::func::effects`]); the strict comparator is the
//! final judge, since an exact match with the original object is correct by construction.
use crate::cst::{apply, normalize, strip_parens, Cst, Edit};
use crate::func::{self, effects, infer_type, is_int_type, is_loop, is_scalar_type, is_stmt, temp_type, FuncInfo};
use crate::hints::{HintClass, RegHints};
use crate::rng::Rng;

/// Operator families, used for diff-guided weighting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Cat {
    /// Statement / declaration order (register allocation, scheduling).
    Order,
    /// Temporaries: extract / inline / refer-to-var.
    Temp,
    /// Types: casts, signedness, const, references.
    Types,
    /// Control flow forms: if/else, loops, ternaries, conditions.
    Control,
    /// Expression forms: operand order, compound assignment, increments.
    Expr,
}

pub type OpFn = fn(&mut M) -> Option<Vec<Edit>>;

pub struct OpDef {
    pub name: &'static str,
    pub cat: Cat,
    /// Base weight (prior). Rescaled by each operator's improvement rate measured over two
    /// TRAIN-split evals (train256 seed 7, v2+v3: improved/tries, smoothed with 200 pseudo-tries
    /// at the global rate, factor clamped to [0.4, 3]). Original provenance (anti-cheat): rank-decreasing priors from the compiler-RE notes'
    /// ranked operator list (SwapDecl > NameTemp/InlineTemp > ToggleCache > SwapOperands >
    /// FlipNegation > ChangeSignedness > SelectForm > RefValLocal > ParamConst > NarrowType >
    /// SplitMergeUpdate > LoopForm), plus ~0 for knobs verified by compiler experiments to have no
    /// codegen effect (kept for cleanup / enabling moves). No frequency counts from project
    /// history are used. Per-function adaptation happens online in the search.
    pub weight: f64,
    /// Diff classes this knob is known to affect (catalog effect classes): [`R`] register
    /// choice, [`S`] instruction order, [`L`] block layout, [`C`] instructions added/changed,
    /// [`M`] stack slots/frame.
    pub aff: u8,
    pub f: OpFn,
}

pub const R: u8 = 1;
pub const S: u8 = 2;
pub const L: u8 = 4;
pub const C: u8 = 8;
pub const M_: u8 = 16;

/// Mutation context for one operator application.
pub struct M<'a> {
    pub cst: &'a Cst,
    pub info: &'a FuncInfo,
    pub rng: &'a mut Rng,
    /// Body nodes, preorder.
    pub nodes: Vec<usize>,
    /// Target register-allocation hints (if available).
    pub hints: Option<&'a RegHints>,
    /// Restrict variable-centric operators to this local.
    pub focus: Option<String>,
}

pub const OPS: &[OpDef] = &[
    // SwapDecl (rank 1)
    OpDef { name: "hoist_decl", cat: Cat::Order, weight: 8.53, aff: R | M_, f: op_hoist_decl },
    OpDef { name: "reorder_decls", cat: Cat::Order, weight: 5.88, aff: R | M_, f: op_reorder_decls },
    // NameTemp / InlineTemp (rank 2)
    OpDef { name: "extract_temp", cat: Cat::Temp, weight: 6.47, aff: C | S | R, f: op_extract_temp },
    OpDef { name: "inline_temp", cat: Cat::Temp, weight: 6.12, aff: C | S | R, f: op_inline_temp },
    // ToggleCache (rank 3)
    OpDef { name: "cse_temp", cat: Cat::Temp, weight: 7.09, aff: C | R, f: op_cse_temp },
    OpDef { name: "inline_var_all", cat: Cat::Temp, weight: 4.37, aff: C | R, f: op_inline_var_all },
    // SwapOperands (rank 4)
    OpDef { name: "commutative", cat: Cat::Expr, weight: 7.75, aff: S | R, f: op_commutative },
    OpDef { name: "flip_compare", cat: Cat::Expr, weight: 4.54, aff: S | L, f: op_flip_compare },
    // FlipNegation (rank 5)
    OpDef { name: "negate_if", cat: Cat::Control, weight: 6.44, aff: L, f: op_negate_if },
    OpDef { name: "push_not", cat: Cat::Control, weight: 3.12, aff: L, f: op_push_not },
    OpDef { name: "cond_zero", cat: Cat::Control, weight: 1.2, aff: L, f: op_cond_zero },
    // ChangeSignedness / NarrowType (ranks 6, 10)
    OpDef { name: "local_type", cat: Cat::Types, weight: 5.24, aff: C | R, f: op_local_type },
    OpDef { name: "sign_cast", cat: Cat::Types, weight: 1.56, aff: C, f: op_sign_cast },
    // SelectForm (rank 7)
    OpDef { name: "ternary_to_if", cat: Cat::Control, weight: 4.91, aff: L | C, f: op_ternary_to_if },
    OpDef { name: "if_to_ternary", cat: Cat::Control, weight: 4.29, aff: L | C, f: op_if_to_ternary },
    OpDef { name: "select_init", cat: Cat::Control, weight: 6.2, aff: L | C, f: op_select_init },
    // RefValLocal (rank 8), ParamConst (rank 9)
    OpDef { name: "ref_local", cat: Cat::Types, weight: 6.06, aff: C | R | M_, f: op_ref_local },
    OpDef { name: "param_const", cat: Cat::Types, weight: 1.87, aff: R, f: op_param_const },
    // SplitMergeUpdate (rank 11)
    OpDef { name: "compound_assign", cat: Cat::Expr, weight: 1.8, aff: C | S, f: op_compound_assign },
    // LoopForm (rank 12)
    OpDef { name: "loop_break", cat: Cat::Control, weight: 4.02, aff: L, f: op_loop_break },
    OpDef { name: "while_to_do", cat: Cat::Control, weight: 3.21, aff: L | C, f: op_while_to_do },
    OpDef { name: "do_to_while", cat: Cat::Control, weight: 1.5, aff: L | C, f: op_do_to_while },
    OpDef { name: "loop_cond_ne", cat: Cat::Control, weight: 1.49, aff: L | C, f: op_loop_cond_ne },
    // Generic permuter moves (decomp-permuter set) not in the ranked list.
    OpDef { name: "swap_stmts", cat: Cat::Order, weight: 4.85, aff: S | R, f: op_swap_stmts },
    OpDef { name: "move_stmt", cat: Cat::Order, weight: 4.15, aff: S | R, f: op_move_stmt },
    OpDef { name: "swap_stores", cat: Cat::Order, weight: 8.04, aff: S, f: op_swap_stores },
    OpDef { name: "split_decl", cat: Cat::Order, weight: 0.8, aff: R, f: op_split_decl },
    OpDef { name: "merge_decl", cat: Cat::Order, weight: 2.21, aff: R, f: op_merge_decl },
    OpDef { name: "refer_to_var", cat: Cat::Temp, weight: 5.37, aff: C | R, f: op_refer_to_var },
    OpDef { name: "chain_assign", cat: Cat::Temp, weight: 3.54, aff: R | C, f: op_chain_assign },
    OpDef { name: "split_assign", cat: Cat::Temp, weight: 0.8, aff: R | C, f: op_split_assign },
    OpDef { name: "add_cast", cat: Cat::Types, weight: 2.33, aff: C, f: op_add_cast },
    OpDef { name: "remove_cast", cat: Cat::Types, weight: 5.07, aff: C, f: op_remove_cast },
    OpDef { name: "pointerize_local", cat: Cat::Types, weight: 1.2, aff: C, f: op_pointerize_local },
    OpDef { name: "nested_if", cat: Cat::Control, weight: 1.05, aff: L, f: op_nested_if },
    OpDef { name: "associative", cat: Cat::Expr, weight: 4.5, aff: S | C, f: op_associative },
    OpDef { name: "demorgan", cat: Cat::Expr, weight: 1.0, aff: L, f: op_demorgan },
    OpDef { name: "const_local", cat: Cat::Types, weight: 0.33, aff: M_, f: op_const_local },
    // Structural forms permutation cannot reach (crate::structural).
    OpDef { name: "switch_to_if", cat: Cat::Control, weight: 4.78, aff: L | C, f: crate::structural::op_switch_to_if },
    OpDef { name: "if_to_switch", cat: Cat::Control, weight: 1.92, aff: L | C, f: crate::structural::op_if_to_switch },
    OpDef { name: "case_order", cat: Cat::Control, weight: 1.17, aff: L, f: crate::structural::op_case_order },
    OpDef { name: "cmp_const", cat: Cat::Expr, weight: 1.52, aff: C | L, f: crate::structural::op_cmp_const },
    OpDef { name: "bool_return", cat: Cat::Control, weight: 1.93, aff: C | L, f: crate::structural::op_bool_return },
    OpDef { name: "guard_split", cat: Cat::Control, weight: 1.92, aff: L, f: crate::structural::op_guard_split },
    OpDef { name: "loop_exit", cat: Cat::Control, weight: 0.85, aff: L | C, f: crate::structural::op_loop_exit },
    OpDef { name: "float_literal", cat: Cat::Expr, weight: 2.34, aff: C | S, f: crate::structural::op_float_literal },
    OpDef { name: "decl_type", cat: Cat::Types, weight: 2.26, aff: C, f: crate::structural::op_decl_type },
    OpDef { name: "ternary_self", cat: Cat::Control, weight: 1.45, aff: L | C, f: crate::structural::op_ternary_self },
    OpDef { name: "fold_magic", cat: Cat::Expr, weight: 12.0, aff: C, f: crate::structural::op_fold_magic },
    OpDef { name: "addr_form", cat: Cat::Expr, weight: 4.01, aff: C | S | R, f: crate::structural::op_addr_form },
    OpDef { name: "bool_literal", cat: Cat::Types, weight: 1.58, aff: C, f: crate::structural::op_bool_literal },
    OpDef { name: "index_loop", cat: Cat::Control, weight: 9.0, aff: C | R | L, f: crate::structural::op_index_loop },
    OpDef { name: "forward_stack", cat: Cat::Temp, weight: 7.5, aff: C | M_, f: crate::structural::op_forward_stack },
    OpDef { name: "dtor_delete", cat: Cat::Expr, weight: 3.0, aff: C, f: crate::structural::op_dtor_delete },
    OpDef { name: "vec_op", cat: Cat::Expr, weight: 3.0, aff: C | S | R, f: crate::structural::op_vec_op },
    OpDef { name: "split_update", cat: Cat::Temp, weight: 2.5, aff: R | C | S, f: crate::structural::op_split_update },
    // Verified no codegen effect: near-zero weight (cleanup / stepping stones only).
    OpDef { name: "unwrap_block", cat: Cat::Order, weight: 0.46, aff: 0, f: op_unwrap_block },
    OpDef { name: "wrap_block", cat: Cat::Order, weight: 0.1, aff: 0, f: op_wrap_block },
    OpDef { name: "early_return", cat: Cat::Control, weight: 0.25, aff: L, f: op_early_return },
    OpDef { name: "for_to_while", cat: Cat::Control, weight: 0.29, aff: L, f: op_for_to_while },
    OpDef { name: "while_to_for", cat: Cat::Control, weight: 0.23, aff: L, f: op_while_to_for },
    OpDef { name: "incr_form", cat: Cat::Expr, weight: 0.19, aff: 0, f: op_incr_form },
    OpDef { name: "struct_ref", cat: Cat::Expr, weight: 0.09, aff: 0, f: op_struct_ref },
    // Register-hint guided (enabled by the search only when target hints exist).
    OpDef { name: "hint_order", cat: Cat::Order, weight: 0.0, aff: R | M_, f: op_hint_order },
    OpDef { name: "hint_temp", cat: Cat::Temp, weight: 0.0, aff: R | C, f: op_hint_temp },
    // Directed edits from the compiler tracer (never sampled; stats only).
    OpDef { name: "trace_fix", cat: Cat::Order, weight: 0.0, aff: R, f: op_never },
    // Directed statement moves from mwdec-oracle schedcheck (never sampled; stats only).
    OpDef { name: "sched_move", cat: Cat::Order, weight: 0.0, aff: S, f: op_never },
    OpDef { name: "sched_swap", cat: Cat::Order, weight: 0.0, aff: S, f: op_never },
    // Cleanup only (weight 0 during search; used by `search::polish`).
    OpDef { name: "strip_parens", cat: Cat::Expr, weight: 0.0, aff: 0, f: op_strip_parens },
];

/// Operators enabled (with this weight) when target register hints are available.
pub const HINT_OPS: &[(&str, f64)] = &[("hint_order", 8.0), ("hint_temp", 6.0)];

/// Operators used by the post-match readability pass.
pub const POLISH_OPS: &[&str] = &["strip_parens", "unwrap_block", "remove_cast", "inline_var_all", "inline_temp", "merge_decl", "cond_zero"];

pub fn op_index(name: &str) -> Option<usize> {
    OPS.iter().position(|o| o.name == name)
}

// ------------------------------------------------------------------ helpers

const PRIMARY: &[&str] = &[
    "identifier",
    "number_literal",
    "string_literal",
    "char_literal",
    "true",
    "false",
    "this",
    "null",
    "nullptr",
    "field_expression",
    "call_expression",
    "subscript_expression",
    "parenthesized_expression",
    "qualified_identifier",
];

fn is_primary(k: &str) -> bool {
    PRIMARY.contains(&k)
}

/// Text of `n`, parenthesized unless it is a primary expression.
fn ptext(cst: &Cst, n: usize) -> String {
    if is_primary(cst.kind(n)) {
        cst.text(n).to_string()
    } else {
        format!("({})", cst.text(n))
    }
}

/// Parenthesize unary-level operands (`!x`, `-x`, casts, `*p`) only when needed for binary contexts.
fn ptext_bin(cst: &Cst, n: usize) -> String {
    match cst.kind(n) {
        k if is_primary(k) => cst.text(n).to_string(),
        "unary_expression" | "pointer_expression" | "cast_expression" | "update_expression" | "sizeof_expression" => {
            cst.text(n).to_string()
        }
        _ => format!("({})", cst.text(n)),
    }
}

fn indent_of(cst: &Cst, n: usize) -> String {
    let s = cst.nodes[n].start;
    let line_start = cst.src[..s].rfind('\n').map(|i| i + 1).unwrap_or(0);
    cst.src[line_start..s].chars().take_while(|c| c.is_whitespace()).collect()
}

fn inner_block(cst: &Cst, n: usize) -> String {
    if cst.kind(n) == "compound_statement" {
        let t = cst.text(n);
        t[1..t.len() - 1].trim().to_string()
    } else {
        cst.text(n).to_string()
    }
}

fn braced(cst: &Cst, n: usize) -> String {
    if cst.kind(n) == "compound_statement" {
        cst.text(n).to_string()
    } else {
        format!("{{ {} }}", cst.text(n))
    }
}

/// Replace statement `s` with several statements, adding braces when `s` is not directly in a block.
fn replace_stmt_many(cst: &Cst, s: usize, parts: &[String]) -> Edit {
    let ind = indent_of(cst, s);
    let joined = parts.join(&format!("\n{ind}"));
    let in_block = cst.parent(s).is_some_and(|p| cst.kind(p) == "compound_statement");
    if in_block {
        Edit::replace(cst, s, joined)
    } else {
        Edit::replace(cst, s, format!("{{\n{ind}{joined}\n{ind}}}"))
    }
}

/// Statements of a compound statement (direct children).
fn stmts_of(cst: &Cst, block: usize) -> Vec<usize> {
    cst.named(block).into_iter().filter(|&c| is_stmt(cst.kind(c))).collect()
}

/// The expression of an if/while condition clause (no declarations).
fn cond_expr(cst: &Cst, stmt: usize) -> Option<usize> {
    let c = cst.child(stmt, "condition")?;
    match cst.kind(c) {
        "condition_clause" => {
            let v = cst.child(c, "value")?;
            (cst.kind(v) != "declaration").then_some(v)
        }
        "parenthesized_expression" => cst.named(c).first().copied(),
        _ => Some(c),
    }
}

/// Text of the logical negation of expression `e`.
fn negate(cst: &Cst, e: usize) -> String {
    match cst.kind(e) {
        "parenthesized_expression" => match cst.named(e).first() {
            Some(&i) => negate(cst, i),
            None => format!("!{}", cst.text(e)),
        },
        "unary_expression" if cst.op(e) == Some("!") => {
            let a = cst.child(e, "argument").unwrap();
            strip_parens(cst.text(a)).to_string()
        }
        "binary_expression" if matches!(cst.op(e), Some("==") | Some("!=")) => {
            let l = cst.child(e, "left").unwrap();
            let r = cst.child(e, "right").unwrap();
            let op = if cst.op(e) == Some("==") { "!=" } else { "==" };
            format!("{} {op} {}", cst.text(l), cst.text(r))
        }
        k if is_primary(k) => format!("!{}", cst.text(e)),
        _ => format!("!({})", cst.text(e)),
    }
}

fn ends_with_return(cst: &Cst, s: usize) -> bool {
    match cst.kind(s) {
        "return_statement" => true,
        "compound_statement" => stmts_of(cst, s).last().is_some_and(|&l| cst.kind(l) == "return_statement"),
        _ => false,
    }
}

/// `continue` statements inside `body` that belong to loop `lp`.
fn has_own_continue(cst: &Cst, lp: usize, body: usize) -> bool {
    cst.descendants(body).into_iter().any(|n| {
        cst.kind(n) == "continue_statement" && cst.ancestors(n).into_iter().find(|&a| is_loop(cst.kind(a))) == Some(lp)
    })
}

fn has_kind(cst: &Cst, n: usize, kinds: &[&str]) -> bool {
    cst.descendants(n).into_iter().any(|d| kinds.contains(&cst.kind(d)))
}

/// Expression is side-effect free (no calls, assignments, increments).
fn pure(cst: &Cst, n: usize) -> bool {
    !has_kind(cst, n, &["call_expression", "assignment_expression", "update_expression", "new_expression", "delete_expression"])
}

fn names_declared(cst: &Cst, decl: usize) -> Vec<String> {
    cst.children_by_field(decl, "declarator").filter_map(|d| func::declarator_name(cst, d).map(|x| x.0)).collect()
}

/// Identifier uses of `name` inside `n` (excluding declarator positions).
pub(crate) fn uses_of(cst: &Cst, n: usize, name: &str) -> Vec<usize> {
    cst.descendants(n)
        .into_iter()
        .filter(|&d| cst.kind(d) == "identifier" && cst.text(d) == name)
        .filter(|&d| {
            let p = cst.parent(d);
            !p.is_some_and(|p| {
                let pk = cst.kind(p);
                (pk == "init_declarator" && cst.nodes[d].field == Some("declarator"))
                    || (pk == "declaration" && cst.nodes[d].field == Some("declarator"))
                    || matches!(pk, "pointer_declarator" | "reference_declarator" | "array_declarator")
            })
        })
        .collect()
}

pub(crate) fn mentions(cst: &Cst, n: usize, name: &str) -> bool {
    !uses_of(cst, n, name).is_empty()
}

impl M<'_> {
    fn of_kind(&self, kinds: &[&str]) -> Vec<usize> {
        self.nodes.iter().copied().filter(|&n| kinds.contains(&self.cst.kind(n))).collect()
    }

    /// Blocks with their statement lists (the body and nested compound statements).
    pub(crate) fn lists(&self, min_len: usize) -> Vec<(usize, Vec<usize>)> {
        self.of_kind(&["compound_statement"])
            .into_iter()
            .map(|b| (b, stmts_of(self.cst, b)))
            .filter(|(_, v)| v.len() >= min_len)
            .collect()
    }

    pub fn pick_one<T: Clone>(&mut self, v: &[T]) -> Option<T> {
        if v.is_empty() {
            None
        } else {
            Some(v[self.rng.below(v.len())].clone())
        }
    }

    fn pick<T: Copy>(&mut self, v: &[T]) -> Option<T> {
        if v.is_empty() {
            None
        } else {
            Some(v[self.rng.below(v.len())])
        }
    }

    fn eff(&self, n: usize) -> func::Effects {
        effects(self.cst, self.info, n)
    }
}

// ------------------------------------------------------------------ Order

fn op_swap_stmts(m: &mut M) -> Option<Vec<Edit>> {
    let lists = m.lists(2);
    let mut cands = Vec::new();
    for (_, l) in &lists {
        for w in l.windows(2) {
            if !m.eff(w[0]).conflicts(&m.eff(w[1])) {
                cands.push((w[0], w[1]));
            }
        }
    }
    let (a, b) = m.pick(&cands)?;
    let c = m.cst;
    Some(vec![Edit::replace(c, a, c.text(b)), Edit::replace(c, b, c.text(a))])
}

/// Reassemble statements `l[lo..=hi]` in a new order, keeping the original gaps.
fn permute_region(cst: &Cst, l: &[usize], lo: usize, hi: usize, order: &[usize]) -> Edit {
    let mut out = String::new();
    for (k, &idx) in order.iter().enumerate() {
        out.push_str(cst.text(l[idx]));
        if lo + k < hi {
            out.push_str(&cst.src[cst.nodes[l[lo + k]].end..cst.nodes[l[lo + k + 1]].start]);
        }
    }
    Edit { start: cst.nodes[l[lo]].start, end: cst.nodes[l[hi]].end, text: out }
}

fn op_move_stmt(m: &mut M) -> Option<Vec<Edit>> {
    let lists = m.lists(3);
    let (_, l) = m.pick(&lists.iter().map(|x| x.0).zip(0..).collect::<Vec<_>>()).map(|(_, i)| lists[i].clone())?;
    let i = m.rng.below(l.len());
    let ei = m.eff(l[i]);
    // Farthest independent positions up and down.
    let mut targets = Vec::new();
    let mut j = i;
    while j > 0 && !ei.conflicts(&m.eff(l[j - 1])) {
        j -= 1;
        targets.push(j);
    }
    let mut j = i;
    while j + 1 < l.len() && !ei.conflicts(&m.eff(l[j + 1])) {
        j += 1;
        targets.push(j);
    }
    targets.retain(|&t| t.abs_diff(i) >= 2);
    let t = m.pick(&targets)?;
    let (lo, hi) = (i.min(t), i.max(t));
    let mut order: Vec<usize> = (lo..=hi).collect();
    if t < i {
        order.remove(i - lo);
        order.insert(0, i);
    } else {
        order.remove(0);
        order.push(i);
    }
    Some(vec![permute_region(m.cst, &l, lo, hi, &order)])
}

fn op_reorder_decls(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    // (a) swap declarators inside a multi-declarator declaration without initializers.
    let multi: Vec<usize> = m
        .of_kind(&["declaration"])
        .into_iter()
        .filter(|&d| {
            let ds: Vec<usize> = c.children_by_field(d, "declarator").collect();
            ds.len() >= 2 && ds.iter().all(|&x| c.kind(x) != "init_declarator")
        })
        .collect();
    // (b) move an initializer-less declaration to another position before its first use.
    let mut moves = Vec::new();
    for (_, l) in m.lists(2) {
        for (i, &s) in l.iter().enumerate() {
            if c.kind(s) != "declaration" || c.children_by_field(s, "declarator").any(|x| c.kind(x) == "init_declarator") {
                continue;
            }
            let names = names_declared(c, s);
            if names.is_empty() || c.child(s, "type").is_some_and(|t| !is_scalar_type(c.text(t))) {
                continue;
            }
            // Earliest: just after previous statement; any position up to first use.
            let first_use = (i + 1..l.len()).find(|&j| names.iter().any(|n| mentions(c, l[j], n))).unwrap_or(l.len());
            for t in 0..first_use {
                if t != i && t + 1 != i + 1 {
                    moves.push((l.clone(), i, t));
                }
            }
        }
    }
    if !multi.is_empty() && (moves.is_empty() || m.rng.chance(0.3)) {
        let d = m.pick(&multi)?;
        let ds: Vec<usize> = c.children_by_field(d, "declarator").collect();
        let a = m.rng.below(ds.len());
        let mut b = m.rng.below(ds.len() - 1);
        if b >= a {
            b += 1;
        }
        return Some(vec![Edit::replace(c, ds[a], c.text(ds[b])), Edit::replace(c, ds[b], c.text(ds[a]))]);
    }
    if moves.is_empty() {
        return None;
    }
    let (l, i, t) = moves[m.rng.below(moves.len())].clone();
    let (lo, hi) = (i.min(t), i.max(t));
    let mut order: Vec<usize> = (lo..=hi).collect();
    if t < i {
        order.remove(i - lo);
        order.insert(0, i);
    } else {
        order.remove(0);
        order.push(i);
    }
    Some(vec![permute_region(c, &l, lo, hi, &order)])
}

/// `T x = e;` -> `T x;\n x = e;` (scalar types; one declarator).
fn op_split_decl(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let cands: Vec<usize> = m
        .of_kind(&["declaration"])
        .into_iter()
        .filter(|&d| c.parent(d).is_some_and(|p| c.kind(p) == "compound_statement"))
        .filter(|&d| {
            let ds: Vec<usize> = c.children_by_field(d, "declarator").collect();
            let t = c.child(d, "type").map(|t| c.text(t)).unwrap_or("");
            ds.len() == 1
                && c.kind(ds[0]) == "init_declarator"
                && c.child(ds[0], "value").is_some_and(|v| c.kind(v) != "initializer_list" && c.kind(v) != "argument_list")
                && !c.text(d).trim_start().starts_with("const")
                && !c.text(d).trim_start().starts_with("static")
                && func::declarator_name(c, ds[0]).is_some_and(|(_, suf)| !suf.contains('&') && !suf.contains('['))
                && (is_scalar_type(t) || func::declarator_name(c, ds[0]).is_some_and(|(_, s)| s.contains('*')))
        })
        .collect();
    let d = m.pick(&cands)?;
    let id = c.children_by_field(d, "declarator").next()?;
    let decl = c.child(id, "declarator")?;
    let val = c.child(id, "value")?;
    let (name, _) = func::declarator_name(c, id)?;
    let prefix = &c.src[c.nodes[d].start..c.nodes[id].start];
    let ind = indent_of(c, d);
    Some(vec![Edit::replace(c, d, format!("{prefix}{};\n{ind}{name} = {};", c.text(decl), c.text(val)))])
}

/// `T x; ... x = e;` -> `... T x = e;` when x is unused in between.
fn op_merge_decl(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for (_, l) in m.lists(2) {
        for (i, &s) in l.iter().enumerate() {
            if c.kind(s) != "declaration" {
                continue;
            }
            let ds: Vec<usize> = c.children_by_field(s, "declarator").collect();
            if ds.len() != 1 || c.kind(ds[0]) == "init_declarator" {
                continue;
            }
            let Some((name, suf)) = func::declarator_name(c, ds[0]) else { continue };
            if suf.contains('[') || suf.contains('&') {
                continue;
            }
            for (j, &t) in l.iter().enumerate().skip(i + 1) {
                if c.kind(t) == "expression_statement" {
                    if let Some(&a) = c.named(t).first() {
                        if c.kind(a) == "assignment_expression"
                            && c.op(a) == Some("=")
                            && c.child(a, "left").is_some_and(|x| c.kind(x) == "identifier" && c.text(x) == name)
                            && !c.child(a, "right").is_some_and(|r| mentions(c, r, &name))
                        {
                            cands.push((s, t, ds[0], a));
                            break;
                        }
                    }
                }
                let _ = j;
                if mentions(c, t, &name) {
                    break;
                }
            }
        }
    }
    let (s, t, d, a) = m.pick(&cands)?;
    let prefix = &c.src[c.nodes[s].start..c.nodes[d].start];
    let rhs = c.child(a, "right")?;
    // Remove the declaration (and its line if it is alone on it).
    let mut end = c.nodes[s].end;
    let rest = &c.src[end..];
    let ws = rest.len() - rest.trim_start_matches([' ', '\t']).len();
    if rest[ws..].starts_with('\n') {
        end += ws + 1;
    }
    let mut start = c.nodes[s].start;
    let line_start = c.src[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
    if c.src[line_start..start].trim().is_empty() && end > c.nodes[s].end {
        start = line_start;
    }
    Some(vec![
        Edit { start, end, text: String::new() },
        Edit::replace(c, t, format!("{prefix}{} = {};", c.text(d), c.text(rhs))),
    ])
}

fn op_wrap_block(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let lists = m.lists(1);
    let (_, l) = lists.get(m.rng.below(lists.len().max(1)))?.clone();
    let i = m.rng.below(l.len());
    let n = 1 + m.rng.below(3.min(l.len() - i));
    let j = i + n - 1;
    // Declarations in the run must not be used after it.
    for &s in &l[i..=j] {
        if c.kind(s) == "declaration" {
            for name in names_declared(c, s) {
                if l[j + 1..].iter().any(|&t| mentions(c, t, &name)) {
                    return None;
                }
            }
        }
        if c.kind(s) == "labeled_statement" {
            return None;
        }
    }
    let ind = indent_of(c, l[i]);
    let body = &c.src[c.nodes[l[i]].start..c.nodes[l[j]].end];
    Some(vec![Edit { start: c.nodes[l[i]].start, end: c.nodes[l[j]].end, text: format!("{{\n{ind}{body}\n{ind}}}") }])
}

fn op_unwrap_block(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let cands: Vec<usize> = m
        .of_kind(&["compound_statement"])
        .into_iter()
        .filter(|&b| b != m.info.body && c.parent(b).is_some_and(|p| c.kind(p) == "compound_statement"))
        .filter(|&b| {
            // Declared names must not clash with other declarations in the function.
            stmts_of(c, b).into_iter().filter(|&s| c.kind(s) == "declaration").flat_map(|s| names_declared(c, s)).all(|n| {
                m.of_kind(&["declaration"]).into_iter().filter(|&d| names_declared(c, d).contains(&n)).count() <= 1
                    && m.info.vars.get(&n).is_some_and(|v| !v.is_param)
            })
        })
        .collect();
    let b = m.pick(&cands)?;
    Some(vec![Edit::replace(c, b, inner_block(c, b))])
}

// ------------------------------------------------------------------ Temp

/// Enclosing statement of expression `e` and whether extraction before it is safe.
fn extraction_site(cst: &Cst, e: usize) -> Option<usize> {
    let mut child = e;
    for a in cst.ancestors(e) {
        let k = cst.kind(a);
        match k {
            "binary_expression" if matches!(cst.op(a), Some("&&") | Some("||")) => {
                if cst.nodes[child].field == Some("right") {
                    return None;
                }
            }
            "conditional_expression" => {
                if cst.nodes[child].field != Some("condition") {
                    return None;
                }
            }
            "sizeof_expression" | "template_argument_list" | "type_descriptor" | "case_statement" | "lambda_expression"
            | "initializer_list" | "field_initializer_list" | "comma_expression" => {
                if k == "case_statement" && cst.nodes[child].field != Some("value") {
                    // statement inside a case: fine, handled at statement level
                } else {
                    return None;
                }
            }
            "for_statement" => {
                if matches!(cst.nodes[child].field, Some("condition") | Some("update")) {
                    return None;
                }
            }
            "while_statement" | "do_statement" => {
                if cst.nodes[child].field == Some("condition") {
                    return None;
                }
            }
            _ => {}
        }
        if is_stmt(k) && k != "compound_statement" {
            return Some(a);
        }
        child = a;
    }
    None
}

/// Is `e` in an lvalue / address-taken / call-target position?
fn in_lvalue_position(cst: &Cst, e: usize) -> bool {
    let mut n = e;
    loop {
        let Some(p) = cst.parent(n) else { return false };
        let pk = cst.kind(p);
        let f = cst.nodes[n].field;
        match pk {
            "assignment_expression" if f == Some("left") => return true,
            "update_expression" => return true,
            "pointer_expression" if cst.op(p) == Some("&") => return true,
            "call_expression" if f == Some("function") => return true,
            "field_expression" if f == Some("argument") && cst.op(p) == Some(".") => {
                // `e.m = ...` or `e.method()`: keep walking up.
                n = p;
            }
            "field_expression" if f == Some("argument") => {
                // `e->m`: e is a pointer value, fine unless it is a method call target on a copy.
                return false;
            }
            "subscript_expression" if f == Some("argument") => {
                // array lvalue: only a problem for arrays (not pointers); be conservative for locals.
                return cst.kind(n) != "identifier" || true;
            }
            "parenthesized_expression" => n = p,
            _ => return false,
        }
    }
}

fn op_extract_temp(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let kinds = [
        "binary_expression",
        "field_expression",
        "subscript_expression",
        "pointer_expression",
        "call_expression",
        "cast_expression",
        "unary_expression",
        "identifier",
        "number_literal",
        "conditional_expression",
    ];
    let mut cands: Vec<(usize, usize)> = Vec::new();
    for n in m.of_kind(&kinds) {
        let k = c.kind(n);
        if k == "pointer_expression" && c.op(n) == Some("&") {
            continue;
        }
        if k == "identifier" && (!m.info.vars.contains_key(c.text(n)) || c.parent(n).is_some_and(|p| c.kind(p) != "argument_list" && c.kind(p) != "binary_expression")) {
            continue;
        }
        if k == "number_literal" && c.parent(n).is_some_and(|p| matches!(c.kind(p), "subscript_argument_list" | "case_statement")) {
            continue;
        }
        if k == "field_expression" && c.parent(n).is_some_and(|p| c.kind(p) == "call_expression" && c.nodes[n].field == Some("function")) {
            continue;
        }
        if in_lvalue_position(c, n) {
            continue;
        }
        // A call whose value is discarded may be void ("illegal use of 'void'").
        if k == "call_expression" && c.parent(n).is_some_and(|p| c.kind(p) == "expression_statement") {
            continue;
        }
        // Not the direct value of a declaration or the whole RHS of a plain assignment statement
        // (that would just rename), except calls/literals which are useful to pull out.
        let Some(s) = extraction_site(c, n) else { continue };
        if c.kind(s) == "declaration" && c.nodes[n].field == Some("value") && k == "identifier" {
            continue;
        }
        if c.kind(s) == "labeled_statement" {
            continue;
        }
        // Calls: only when the statement has no other side effects evaluated around it.
        if !pure(c, n) {
            let others = c.descendants(s).into_iter().filter(|&d| c.kind(d) == "call_expression" && !c.contains(n, d) && !c.contains(d, n)).count();
            if others > 0 {
                continue;
            }
        } else if c.descendants(s).into_iter().any(|d| c.kind(d) == "call_expression" && !c.contains(n, d) && !c.contains(d, n))
            && !m.eff(n).reads.is_empty()
            && m.eff(n).mem_read
        {
            // pure memory read moved before a call in the same statement: unsafe
            continue;
        }
        cands.push((n, s));
    }
    let (e, s) = m.pick(&cands)?;
    let ty = temp_type(c, m.info, e);
    let name = m.info.fresh_name(c, "temp_");
    let decl = format!("{ty} {name} = {};", c.text(e));
    let ind = indent_of(c, s);
    let in_block = c.parent(s).is_some_and(|p| c.kind(p) == "compound_statement");
    if in_block {
        Some(vec![Edit::insert(c.nodes[s].start, format!("{decl}\n{ind}")), Edit::replace(c, e, name)])
    } else {
        // Rebuild the statement text with the replacement, wrapped in a block.
        let st = c.nodes[s].start;
        let stmt_text = format!("{}{}{}", &c.src[st..c.nodes[e].start], name, &c.src[c.nodes[e].end..c.nodes[s].end]);
        Some(vec![Edit::replace(c, s, format!("{{\n{ind}    {decl}\n{ind}    {stmt_text}\n{ind}}}"))])
    }
}

fn op_inline_temp(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for (_, l) in m.lists(2) {
        for (i, &s) in l.iter().enumerate() {
            if c.kind(s) != "declaration" || c.text(s).trim_start().starts_with("static") {
                continue;
            }
            let ds: Vec<usize> = c.children_by_field(s, "declarator").collect();
            if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
                continue;
            }
            let Some((name, suf)) = func::declarator_name(c, ds[0]) else { continue };
            if suf.contains('[') || m.info.aliased.contains(&name) && !suf.contains('&') {
                continue;
            }
            if m.focus.as_ref().is_some_and(|f| *f != name) {
                continue;
            }
            let Some(val) = c.child(ds[0], "value") else { continue };
            if matches!(c.kind(val), "initializer_list" | "argument_list") {
                continue;
            }
            // Exactly one use, a read, in the remaining statements.
            let uses: Vec<usize> = l[i + 1..].iter().flat_map(|&t| uses_of(c, t, &name)).collect();
            if uses.len() != 1 {
                continue;
            }
            let u = uses[0];
            if in_lvalue_position(c, u) && c.parent(u).is_some_and(|p| c.kind(p) != "field_expression") {
                continue;
            }
            let j = (i + 1..l.len()).find(|&j| c.contains(l[j], u))?;
            // Not into a loop that does not contain the declaration (would re-evaluate).
            if c.ancestors(u).into_iter().take_while(|&a| a != l[j]).any(|a| is_loop(c.kind(a))) || is_loop(c.kind(l[j])) && !pure(c, val) {
                continue;
            }
            let ev = m.eff(val);
            if !pure(c, val) && (j != i + 1) {
                continue;
            }
            if l[i + 1..j].iter().any(|&t| ev.conflicts(&m.eff(t)) || m.eff(t).control) {
                continue;
            }
            cands.push((s, ds[0], val, u, name));
        }
    }
    let (s, d, val, u, name) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    let ty = m.info.vars.get(&name).map(|v| v.ty.clone()).unwrap_or_default();
    let vt = infer_type(c, m.info, val);
    let base_ty = ty.trim_end_matches('&').trim().trim_start_matches("const ").trim().to_string();
    let _ = d;
    let repl = if vt.as_deref() == Some(base_ty.as_str()) || ty.ends_with('&') || !is_scalar_type(&base_ty) {
        ptext(c, val)
    } else {
        format!("({base_ty}){}", ptext(c, val))
    };
    // Remove the declaration line.
    let mut start = c.nodes[s].start;
    let mut end = c.nodes[s].end;
    let line_start = c.src[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let rest = &c.src[end..];
    let ws = rest.len() - rest.trim_start_matches([' ', '\t']).len();
    if c.src[line_start..start].trim().is_empty() && rest[ws..].starts_with('\n') {
        start = line_start;
        end += ws + 1;
    }
    Some(vec![Edit { start, end, text: String::new() }, Edit::replace(c, u, repl)])
}

/// Is local `name` ever written after its declaration (assignment, ++/--, address taken)?
fn reassigned(c: &Cst, scope: usize, name: &str) -> bool {
    uses_of(c, scope, name).into_iter().any(|u| {
        let mut n = u;
        loop {
            let Some(p) = c.parent(n) else { return false };
            match c.kind(p) {
                "assignment_expression" => return c.nodes[n].field == Some("left"),
                "update_expression" => return true,
                "pointer_expression" => return c.op(p) == Some("&"),
                "field_expression" if c.op(p) == Some(".") && c.nodes[n].field == Some("argument") => n = p,
                "parenthesized_expression" => n = p,
                _ => return false,
            }
        }
    })
}

/// Text replacing a use of a local of type `ty` initialized with `val`.
fn inline_text(c: &Cst, info: &FuncInfo, ty: &str, val: usize) -> String {
    let vt = infer_type(c, info, val);
    let base_ty = ty.trim_end_matches('&').trim().trim_start_matches("const ").trim().to_string();
    if vt.as_deref() == Some(base_ty.as_str()) || ty.ends_with('&') || !is_scalar_type(&base_ty) || vt.is_none() && base_ty.ends_with('*') {
        ptext(c, val)
    } else {
        format!("({base_ty}){}", ptext(c, val))
    }
}

/// Remove a whole statement, including its line when it is alone on it.
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

/// `T x = E; ... x ... x ...` -> replace every use of `x` with `E` and drop the declaration
/// (E pure, x never reassigned, E's inputs not modified before the last use). This undoes the
/// lifter's register temporaries (`temp_r4`).
fn op_inline_var_all(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for (_, l) in m.lists(2) {
        for (i, &s) in l.iter().enumerate() {
            if c.kind(s) != "declaration" || c.text(s).trim_start().starts_with("static") {
                continue;
            }
            let ds: Vec<usize> = c.children_by_field(s, "declarator").collect();
            if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
                continue;
            }
            let Some((name, suf)) = func::declarator_name(c, ds[0]) else { continue };
            if suf.contains('[') || suf.contains('&') || m.info.aliased.contains(&name) {
                continue;
            }
            if m.focus.as_ref().is_some_and(|f| *f != name) {
                continue;
            }
            let Some(val) = c.child(ds[0], "value") else { continue };
            if matches!(c.kind(val), "initializer_list" | "argument_list") || !pure(c, val) {
                continue;
            }
            let uses: Vec<usize> = l[i + 1..].iter().flat_map(|&t| uses_of(c, t, &name)).collect();
            if uses.len() < 2 || l[i + 1..].iter().any(|&t| reassigned(c, t, &name)) {
                continue;
            }
            let last = *uses.last().unwrap();
            let Some(j) = (i + 1..l.len()).find(|&j| c.contains(l[j], last)) else { continue };
            // No loops around uses (re-evaluation), no writes to E's locals, no calls before
            // the last-use statement when E reads memory.
            if uses.iter().any(|&u| c.ancestors(u).into_iter().take_while(|&a| a != l[i]).any(|a| is_loop(c.kind(a)))) {
                continue;
            }
            let ev = m.eff(val);
            let ok = l[i + 1..=j].iter().enumerate().all(|(k, &t)| {
                let et = m.eff(t);
                et.writes.iter().all(|w| !ev.reads.contains(w))
                    && (!ev.mem_read || i + 1 + k == j || !has_kind(c, t, &["call_expression", "new_expression", "delete_expression"]))
            });
            if !ok {
                continue;
            }
            cands.push((s, val, uses, name));
        }
    }
    let (s, val, uses, name) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    let ty = m.info.vars.get(&name).map(|v| v.ty.clone()).unwrap_or_default();
    let repl = inline_text(c, m.info, &ty, val);
    let mut edits: Vec<Edit> = uses.iter().map(|&u| Edit::replace(c, u, repl.clone())).collect();
    edits.push(remove_stmt(c, s));
    Some(edits)
}

/// Common subexpression -> temporary: every occurrence of a pure expression (2+ times) in one
/// block is replaced by a new local declared before the first occurrence.
fn op_cse_temp(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> = std::collections::BTreeMap::new();
    for n in m.nodes.clone() {
        let k = c.kind(n);
        let in_call_pos = c.parent(n).is_some_and(|p| c.kind(p) == "call_expression" && c.nodes[n].field == Some("function"));
        let ok_kind = match k {
            "field_expression" | "subscript_expression" | "binary_expression" | "cast_expression" => true,
            "pointer_expression" => c.op(n) == Some("*"),
            "qualified_identifier" | "identifier" => !m.info.vars.contains_key(c.text(n)) && !in_call_pos,
            _ => false,
        };
        if !ok_kind || in_call_pos || !pure(c, n) || in_lvalue_position(c, n) {
            continue;
        }
        if c.parent(n).is_some_and(|p| c.kind(p) == "qualified_identifier") {
            continue;
        }
        groups.entry(normalize(strip_parens(c.text(n)))).or_default().push(n);
    }
    let mut cands = Vec::new();
    for (_, occ) in groups {
        if occ.len() < 2 {
            continue;
        }
        // Statement in the innermost common block containing the first occurrence.
        let first = occ[0];
        let Some(site) = extraction_site(c, first) else { continue };
        let mut blk = None;
        for a in c.ancestors(site) {
            if c.kind(a) == "compound_statement" && occ.iter().all(|&o| c.contains(a, o)) {
                blk = Some(a);
                break;
            }
        }
        let Some(b) = blk else { continue };
        let l = stmts_of(c, b);
        let Some(k) = l.iter().position(|&t| c.contains(t, first)) else { continue };
        let Some(j) = l.iter().position(|&t| c.contains(t, *occ.last().unwrap())) else { continue };
        if occ.iter().any(|&o| c.ancestors(o).into_iter().take_while(|&a| a != b).any(|a| is_loop(c.kind(a)))) {
            continue;
        }
        let ev = m.eff(first);
        let ok = l[k..=j].iter().enumerate().all(|(q, &t)| {
            let et = m.eff(t);
            et.writes.iter().all(|w| !ev.reads.contains(w))
                && (!ev.mem_read || k + q == j || !has_kind(c, t, &["call_expression", "new_expression", "delete_expression"]))
        });
        if ok {
            cands.push((occ, l[k]));
        }
    }
    let (occ, at) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    let ty = temp_type(c, m.info, occ[0]);
    let name = m.info.fresh_name(c, "temp_");
    let ind = indent_of(c, at);
    let mut edits = vec![Edit::insert(c.nodes[at].start, format!("{ty} {name} = {};\n{ind}", strip_parens(c.text(occ[0]))))];
    // Nested occurrences can't both be replaced.
    let mut taken: Vec<usize> = Vec::new();
    for &o in &occ {
        if taken.iter().any(|&t| c.contains(t, o) || c.contains(o, t)) {
            continue;
        }
        taken.push(o);
        edits.push(Edit::replace(c, o, name.clone()));
    }
    Some(edits)
}

/// After `v = E;` / `T v = E;`, replace a later identical pure `E` with `v`.
fn op_refer_to_var(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for (_, l) in m.lists(2) {
        for (i, &s) in l.iter().enumerate() {
            let (name, val) = match c.kind(s) {
                "declaration" => {
                    let ds: Vec<usize> = c.children_by_field(s, "declarator").collect();
                    if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
                        continue;
                    }
                    let Some((n, suf)) = func::declarator_name(c, ds[0]) else { continue };
                    if !suf.is_empty() && suf != "*" {
                        continue;
                    }
                    (n, c.child(ds[0], "value"))
                }
                "expression_statement" => {
                    let Some(&a) = c.named(s).first() else { continue };
                    if c.kind(a) != "assignment_expression" || c.op(a) != Some("=") {
                        continue;
                    }
                    let Some(lhs) = c.child(a, "left") else { continue };
                    if c.kind(lhs) != "identifier" || !m.info.is_private(c.text(lhs)) {
                        continue;
                    }
                    (c.text(lhs).to_string(), c.child(a, "right"))
                }
                _ => continue,
            };
            let Some(val) = val else { continue };
            if !pure(c, val) || matches!(c.kind(val), "identifier" | "number_literal" | "initializer_list") {
                continue;
            }
            let key = normalize(strip_parens(c.text(val)));
            let ev = m.eff(val);
            for &t in &l[i + 1..] {
                for d in c.descendants(t) {
                    if d != val && normalize(strip_parens(c.text(d))) == key && c.nodes[d].named && !in_lvalue_position(c, d) {
                        cands.push((d, name.clone()));
                    }
                }
                let et = m.eff(t);
                if ev.conflicts(&et) || et.writes.contains(&name) || et.control {
                    break;
                }
            }
        }
    }
    let (d, name) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit::replace(c, d, name)])
}

fn assign_parts(c: &Cst, s: usize) -> Option<(usize, usize, usize)> {
    if c.kind(s) != "expression_statement" {
        return None;
    }
    let a = *c.named(s).first()?;
    if c.kind(a) != "assignment_expression" || c.op(a) != Some("=") {
        return None;
    }
    Some((a, c.child(a, "left")?, c.child(a, "right")?))
}

/// `a = x; b = a;` -> `b = a = x;` and `a = x; b = x;` -> `a = b = x;` (permuter's chain_assignment).
fn op_chain_assign(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for (_, l) in m.lists(2) {
        for i in 0..l.len() {
            let Some((_, l1, r1)) = assign_parts(c, l[i]) else { continue };
            for j in i + 1..(i + 5).min(l.len()) {
                let Some((_, l2, r2)) = assign_parts(c, l[j]) else { break };
                let r2n = normalize(c.text(r2));
                let same_l = r2n == normalize(c.text(l1));
                let same_r = r2n == normalize(c.text(r1));
                if (same_l || same_r) && pure(c, r2) && pure(c, l2) {
                    let ej = m.eff(l[j]);
                    if l[i + 1..j].iter().all(|&t| !ej.conflicts(&m.eff(t))) {
                        cands.push((l[i], l[j], l1, r1, l2, same_l));
                    }
                }
            }
        }
    }
    let (si, sj, l1, r1, l2, same_l) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    let text = if same_l {
        format!("{} = {} = {};", c.text(l2), c.text(l1), c.text(r1))
    } else {
        format!("{} = {} = {};", c.text(l1), c.text(l2), c.text(r1))
    };
    let mut start = c.nodes[sj].start;
    let line_start = c.src[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
    if c.src[line_start..start].trim().is_empty() {
        start = line_start;
    }
    let _ = same_l;
    Some(vec![Edit::replace(c, si, text), Edit { start, end: c.nodes[sj].end, text: String::new() }])
}

/// `a = b = e;` -> `b = e;\n a = b;`
fn op_split_assign(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let cands: Vec<usize> = m
        .of_kind(&["expression_statement"])
        .into_iter()
        .filter(|&s| assign_parts(c, s).is_some_and(|(_, l, r)| c.kind(r) == "assignment_expression" && c.op(r) == Some("=") && pure(c, l)))
        .collect();
    let s = m.pick(&cands)?;
    let (_, l, r) = assign_parts(c, s)?;
    let inner_l = c.child(r, "left")?;
    Some(vec![replace_stmt_many(c, s, &[format!("{};", c.text(r)), format!("{} = {};", c.text(l), c.text(inner_l))])])
}

// ------------------------------------------------------------------ Types

fn op_add_cast(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, String)> = Vec::new();
    for n in m.nodes.clone() {
        match c.kind(n) {
            "assignment_expression" if c.op(n) == Some("=") => {
                let (Some(l), Some(r)) = (c.child(n, "left"), c.child(n, "right")) else { continue };
                if let Some(t) = infer_type(c, m.info, l) {
                    if is_scalar_type(&t) && c.kind(r) != "cast_expression" {
                        cands.push((r, t));
                    }
                }
            }
            "init_declarator" => {
                let Some(v) = c.child(n, "value") else { continue };
                let Some((name, _)) = func::declarator_name(c, n) else { continue };
                if let Some(var) = m.info.vars.get(&name) {
                    let t = var.ty.trim_start_matches("const ").to_string();
                    if is_scalar_type(&t) && c.kind(v) != "cast_expression" && c.kind(v) != "initializer_list" {
                        cands.push((v, t));
                    }
                }
            }
            "return_statement" => {
                let Some(&v) = c.named(n).first() else { continue };
                let t = m.info.ret_ty.clone();
                if is_scalar_type(&t) && c.kind(v) != "cast_expression" {
                    cands.push((v, t));
                }
            }
            "identifier" | "binary_expression" | "field_expression" => {
                if in_lvalue_position(c, n) || c.parent(n).is_some_and(|p| c.kind(p) == "cast_expression") {
                    continue;
                }
                if let Some(t) = infer_type(c, m.info, n) {
                    if is_int_type(&t) && t != "bool" && extraction_site(c, n).is_some() {
                        cands.push((n, t));
                    }
                }
            }
            _ => {}
        }
    }
    let (e, t) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit::replace(c, e, format!("({t}){}", ptext(c, e)))])
}

fn op_remove_cast(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    // A pointer cast that is dereferenced (`*(T*)x`, `((T*)x)->m`, `((T*)x)[i]`) is needed:
    // removing it mostly gives "pointer/array required" (train-split compile-error log).
    let derefd = |n: usize| -> bool {
        let mut p = c.parent(n);
        while let Some(x) = p {
            if c.kind(x) == "parenthesized_expression" {
                p = c.parent(x);
            } else {
                break;
            }
        }
        p.is_some_and(|x| match c.kind(x) {
            "pointer_expression" => c.op(x) == Some("*"),
            "field_expression" => c.op(x) == Some("->"),
            "subscript_expression" => true,
            _ => false,
        })
    };
    let cands: Vec<usize> = m
        .of_kind(&["cast_expression"])
        .into_iter()
        .filter(|&n| !(derefd(n) && c.child(n, "type").is_some() && c.text(n).split(')').next().is_some_and(|t| t.contains('*'))))
        .collect();
    let n = m.pick(&cands)?;
    let v = c.child(n, "value")?;
    // Keep precedence: cast binds tighter than any binary operator.
    let need_paren = !is_primary(c.kind(v))
        && c.parent(n).is_some_and(|p| !matches!(c.kind(p), "argument_list" | "init_declarator" | "return_statement" | "condition_clause" | "expression_statement" | "parenthesized_expression"))
        && !(c.parent(n).is_some_and(|p| c.kind(p) == "assignment_expression" && c.nodes[n].field == Some("right")));
    let t = if need_paren { format!("({})", c.text(v)) } else { c.text(v).to_string() };
    Some(vec![Edit::replace(c, n, t)])
}

const SIGN_PAIRS: &[(&str, &str)] = &[
    ("int", "unsigned int"),
    ("s32", "u32"),
    ("short", "unsigned short"),
    ("s16", "u16"),
    ("char", "unsigned char"),
    ("signed char", "unsigned char"),
    ("s8", "u8"),
    ("long", "unsigned long"),
];

fn flip_sign(t: &str) -> Option<&'static str> {
    SIGN_PAIRS.iter().find_map(|&(a, b)| if t == a { Some(b) } else if t == b { Some(a) } else { None })
}

/// Cast an operand of a comparison / arithmetic to the other signedness (unsigned compares etc.).
fn op_sign_cast(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for b in m.of_kind(&["binary_expression"]) {
        for side in ["left", "right"] {
            let Some(o) = c.child(b, side) else { continue };
            if c.kind(o) == "cast_expression" {
                continue;
            }
            if let Some(t) = infer_type(c, m.info, o) {
                if let Some(f) = flip_sign(&t) {
                    cands.push((o, f));
                }
            }
        }
    }
    let (o, f) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit::replace(c, o, format!("({f}){}", ptext(c, o)))])
}

const TYPE_ALTS: &[(&str, &[&str])] = &[
    ("int", &["unsigned int", "bool", "short", "s32", "u32"]),
    ("unsigned int", &["int", "u32", "unsigned short"]),
    ("s32", &["u32", "int", "bool", "s16"]),
    ("u32", &["s32", "unsigned int", "u16", "u8"]),
    ("short", &["unsigned short", "int"]),
    ("unsigned short", &["short", "unsigned int"]),
    ("s16", &["u16", "s32"]),
    ("u16", &["s16", "u32"]),
    ("char", &["unsigned char", "int"]),
    ("unsigned char", &["char", "unsigned int", "bool"]),
    ("s8", &["u8", "s32"]),
    ("u8", &["s8", "u32", "bool"]),
    ("bool", &["int", "u8", "s32", "unsigned char"]),
    ("long", &["unsigned long", "int"]),
    ("unsigned long", &["long", "unsigned int"]),
];

/// Value range of an integer type name.
fn int_range(t: &str) -> Option<(i64, i64)> {
    Some(match t {
        "bool" => (0, 1),
        "u8" | "unsigned char" => (0, 255),
        "s8" | "signed char" | "char" => (-128, 127),
        "s16" | "short" => (-32768, 32767),
        "u16" | "unsigned short" => (0, 65535),
        "u32" | "unsigned int" | "unsigned long" => (0, u32::MAX as i64),
        "s32" | "int" | "long" => (i32::MIN as i64, i32::MAX as i64),
        _ => return None,
    })
}

/// Integer literal value of `n` (`5`, `-3`, `0x10`, `true`).
fn literal_value(c: &Cst, n: usize) -> Option<i64> {
    match c.kind(n) {
        "number_literal" => {
            let t = c.text(n).trim_end_matches(['u', 'U', 'l', 'L']).to_ascii_lowercase();
            match t.strip_prefix("0x") {
                Some(h) => i64::from_str_radix(h, 16).ok(),
                None => t.parse().ok(),
            }
        }
        "true" => Some(1),
        "false" => Some(0),
        "unary_expression" if c.op(n) == Some("-") => literal_value(c, c.child(n, "argument")?).map(|v| -v),
        "parenthesized_expression" => literal_value(c, *c.named(n).first()?),
        _ => None,
    }
}

/// Every integer literal assigned to / initializing `name` fits type `t`.
fn literals_fit(c: &Cst, info: &FuncInfo, name: &str, t: &str) -> bool {
    let Some((lo, hi)) = int_range(t) else { return true };
    for u in uses_of(c, info.body, name) {
        let Some(p) = c.parent(u) else { continue };
        let val = match c.kind(p) {
            "assignment_expression" if c.nodes[u].field == Some("left") => c.child(p, "right"),
            _ => None,
        };
        if let Some(v) = val.and_then(|v| literal_value(c, v)) {
            if v < lo || v > hi {
                return false;
            }
        }
    }
    for d in c.descendants(info.body) {
        if c.kind(d) == "init_declarator" && func::declarator_name(c, d).is_some_and(|(n, _)| n == name) {
            if let Some(v) = c.child(d, "value").and_then(|v| literal_value(c, v)) {
                if v < lo || v > hi {
                    return false;
                }
            }
        }
    }
    true
}

fn op_local_type(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for d in m.of_kind(&["declaration"]) {
        let Some(t) = c.child(d, "type") else { continue };
        if let Some((_, alts)) = TYPE_ALTS.iter().find(|(k, _)| *k == c.text(t)) {
            let names = names_declared(c, d);
            for a in alts.iter() {
                if names.iter().all(|n| literals_fit(c, m.info, n, a)) {
                    cands.push((t, *a));
                }
            }
        }
    }
    let (t, a) = m.pick(&cands)?;
    Some(vec![Edit::replace(c, t, a)])
}

fn op_const_local(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let cands: Vec<usize> = m
        .of_kind(&["declaration"])
        .into_iter()
        .filter(|&d| c.children_by_field(d, "declarator").all(|x| c.kind(x) == "init_declarator") && !c.text(d).starts_with("static"))
        .collect();
    let d = m.pick(&cands)?;
    let t = c.text(d);
    if let Some(rest) = t.strip_prefix("const ") {
        Some(vec![Edit::replace(c, d, rest.trim_start())])
    } else {
        Some(vec![Edit::insert(c.nodes[d].start, "const ")])
    }
}

fn op_ref_local(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for d in m.of_kind(&["declaration"]) {
        let ds: Vec<usize> = c.children_by_field(d, "declarator").collect();
        if ds.len() != 1 || c.kind(ds[0]) != "init_declarator" {
            continue;
        }
        let Some(t) = c.child(d, "type") else { continue };
        if is_scalar_type(c.text(t)) {
            continue;
        }
        let Some(inner) = c.child(ds[0], "declarator") else { continue };
        let Some(v) = c.child(ds[0], "value") else { continue };
        match c.kind(inner) {
            "identifier" if matches!(c.kind(v), "identifier" | "field_expression" | "subscript_expression" | "pointer_expression" | "call_expression") => {
                cands.push((d, t, inner, true))
            }
            "reference_declarator" => cands.push((d, t, inner, false)),
            _ => {}
        }
    }
    let (d, _t, inner, to_ref) = m.pick(&cands)?;
    if to_ref {
        let mut e = vec![Edit::replace(c, inner, format!("& {}", c.text(inner)))];
        if !c.text(d).starts_with("const") && m.rng.chance(0.7) {
            e.push(Edit::insert(c.nodes[d].start, "const "));
        }
        Some(e)
    } else {
        let name = *c.named(inner).last()?;
        Some(vec![Edit::replace(c, inner, c.text(name).to_string())])
    }
}

// ------------------------------------------------------------------ Control

fn op_negate_if(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let cands: Vec<usize> = m.of_kind(&["if_statement"]).into_iter().filter(|&i| cond_expr(c, i).is_some()).collect();
    let s = m.pick(&cands)?;
    let ce = cond_expr(c, s)?;
    let cons = c.child(s, "consequence")?;
    let neg = negate(c, ce);
    match c.child(s, "alternative") {
        Some(alt) => {
            let alt_body = *c.named(alt).first()?;
            let new_cons = braced(c, alt_body);
            let new_alt = if c.kind(cons) == "compound_statement" { c.text(cons).to_string() } else { format!("{{ {} }}", c.text(cons)) };
            Some(vec![Edit::replace(c, s, format!("if ({neg}) {new_cons} else {new_alt}"))])
        }
        None => {
            if !m.rng.chance(0.3) {
                return None;
            }
            Some(vec![Edit::replace(c, s, format!("if ({neg}) {{ }} else {}", braced(c, cons)))])
        }
    }
}

fn op_early_return(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut fwd = Vec::new(); // remove else
    let mut back = Vec::new(); // add else around the rest
    for (_, l) in m.lists(1) {
        for (i, &s) in l.iter().enumerate() {
            if c.kind(s) != "if_statement" || cond_expr(c, s).is_none() {
                continue;
            }
            let Some(cons) = c.child(s, "consequence") else { continue };
            if !ends_with_return(c, cons) {
                continue;
            }
            match c.child(s, "alternative") {
                Some(_) => fwd.push(s),
                None => {
                    if i + 1 < l.len() && l[i + 1..].iter().all(|&t| c.kind(t) != "labeled_statement") {
                        back.push((s, *l.last().unwrap()));
                    }
                }
            }
        }
    }
    if !fwd.is_empty() && (back.is_empty() || m.rng.chance(0.5)) {
        let s = m.pick(&fwd)?;
        let alt = c.child(s, "alternative")?;
        let body = *c.named(alt).first()?;
        let ind = indent_of(c, s);
        let head = &c.src[c.nodes[s].start..c.nodes[alt].start];
        let has_decl = c.kind(body) == "compound_statement" && stmts_of(c, body).iter().any(|&x| c.kind(x) == "declaration");
        let rest = if has_decl { c.text(body).to_string() } else { inner_block(c, body) };
        return Some(vec![Edit::replace(c, s, format!("{}\n{ind}{rest}", head.trim_end()))]);
    }
    let (s, last) = back.get(m.rng.below(back.len().max(1)))?.clone();
    let ind = indent_of(c, s);
    let next_start = c.src[c.nodes[s].end..].find(|ch: char| !ch.is_whitespace()).map(|k| c.nodes[s].end + k)?;
    let rest = &c.src[next_start..c.nodes[last].end];
    Some(vec![Edit {
        start: c.nodes[s].start,
        end: c.nodes[last].end,
        text: format!("{} else {{\n{ind}    {rest}\n{ind}}}", c.text(s)),
    }])
}

fn op_for_to_while(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let cands: Vec<usize> = m
        .of_kind(&["for_statement"])
        .into_iter()
        .filter(|&f| c.child(f, "body").is_some_and(|b| !has_own_continue(c, f, b)) && c.child(f, "right").is_none())
        .collect();
    let f = m.pick(&cands)?;
    let body = c.child(f, "body")?;
    let init = c.child(f, "initializer");
    let cond = c.child(f, "condition").map(|x| c.text(x).to_string()).unwrap_or_else(|| "1".into());
    let upd = c.child(f, "update").map(|x| format!("{};", c.text(x)));
    let ind = indent_of(c, f);
    let mut b = inner_block(c, body);
    if let Some(u) = upd {
        b = format!("{b}\n{ind}    {u}");
    }
    let wl = format!("while ({cond}) {{\n{ind}    {b}\n{ind}}}");
    let mut parts = Vec::new();
    let mut is_decl = false;
    if let Some(i) = init {
        let t = c.text(i).trim().to_string();
        is_decl = c.kind(i) == "declaration";
        parts.push(if t.ends_with(';') { t } else { format!("{t};") });
    }
    parts.push(wl);
    if is_decl {
        Some(vec![Edit::replace(c, f, format!("{{\n{ind}{}\n{ind}}}", parts.join(&format!("\n{ind}"))))])
    } else {
        Some(vec![replace_stmt_many(c, f, &parts)])
    }
}

fn update_of(c: &Cst, s: usize, var: &str) -> bool {
    let Some(&e) = c.named(s).first() else { return false };
    c.kind(s) == "expression_statement"
        && match c.kind(e) {
            "update_expression" => c.child(e, "argument").is_some_and(|a| c.text(a) == var),
            "assignment_expression" => c.child(e, "left").is_some_and(|a| c.text(a) == var) && c.op(e) != Some("="),
            _ => false,
        }
}

fn op_while_to_for(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut full = Vec::new();
    let mut simple = Vec::new();
    for (_, l) in m.lists(1) {
        for (i, &s) in l.iter().enumerate() {
            if c.kind(s) != "while_statement" || cond_expr(c, s).is_none() {
                continue;
            }
            simple.push(s);
            let Some(body) = c.child(s, "body") else { continue };
            if i == 0 || c.kind(body) != "compound_statement" || has_own_continue(c, s, body) {
                continue;
            }
            let Some((_, lhs, _)) = assign_parts(c, l[i - 1]) else { continue };
            if c.kind(lhs) != "identifier" {
                continue;
            }
            let bs = stmts_of(c, body);
            if bs.len() >= 2 && update_of(c, *bs.last().unwrap(), c.text(lhs)) {
                full.push((l[i - 1], s, body, *bs.last().unwrap()));
            }
        }
    }
    if !full.is_empty() && m.rng.chance(0.8) {
        let (init, w, body, last) = full[m.rng.below(full.len())];
        let ind = indent_of(c, w);
        let ce = cond_expr(c, w)?;
        let init_t = c.text(init).trim_end_matches(';');
        let upd_t = c.text(last).trim_end_matches(';');
        let bs = stmts_of(c, body);
        let inner = &c.src[c.nodes[bs[0]].start..c.nodes[bs[bs.len() - 2]].end];
        return Some(vec![Edit {
            start: c.nodes[init].start,
            end: c.nodes[w].end,
            text: format!("for ({init_t}; {}; {upd_t}) {{\n{ind}    {inner}\n{ind}}}", c.text(ce)),
        }]);
    }
    let w = m.pick(&simple)?;
    let ce = cond_expr(c, w)?;
    let body = c.child(w, "body")?;
    Some(vec![Edit::replace(c, w, format!("for (; {}; ) {}", c.text(ce), c.text(body)))])
}

fn op_while_to_do(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let cands: Vec<usize> = m.of_kind(&["while_statement"]).into_iter().filter(|&w| cond_expr(c, w).is_some()).collect();
    let w = m.pick(&cands)?;
    let ce = c.text(cond_expr(c, w)?).to_string();
    let body = c.child(w, "body")?;
    let ind = indent_of(c, w);
    Some(vec![Edit::replace(c, w, format!("if ({ce}) {{\n{ind}    do {} while ({ce});\n{ind}}}", braced(c, body)))])
}

fn op_do_to_while(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for s in m.of_kind(&["if_statement"]) {
        if c.child(s, "alternative").is_some() {
            continue;
        }
        let (Some(ce), Some(cons)) = (cond_expr(c, s), c.child(s, "consequence")) else { continue };
        let d = if c.kind(cons) == "do_statement" {
            cons
        } else if c.kind(cons) == "compound_statement" {
            let ss = stmts_of(c, cons);
            if ss.len() != 1 || c.kind(ss[0]) != "do_statement" {
                continue;
            }
            ss[0]
        } else {
            continue;
        };
        let Some(dc) = cond_expr(c, d) else { continue };
        if normalize(strip_parens(c.text(dc))) == normalize(strip_parens(c.text(ce))) {
            cands.push((s, d, ce));
        }
    }
    let (s, d, ce) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    let body = c.child(d, "body")?;
    Some(vec![Edit::replace(c, s, format!("while ({}) {}", c.text(ce), c.text(body)))])
}

fn ternary_branch(c: &Cst, n: usize) -> String {
    match c.kind(n) {
        "conditional_expression" | "assignment_expression" | "comma_expression" => format!("({})", c.text(n)),
        _ => c.text(n).to_string(),
    }
}

fn op_ternary_to_if(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for n in m.of_kind(&["conditional_expression"]) {
        let Some(p) = c.parent(n) else { continue };
        match c.kind(p) {
            "assignment_expression" if c.op(p) == Some("=") && c.nodes[n].field == Some("right") => {
                if let Some(s) = c.parent(p).filter(|&s| c.kind(s) == "expression_statement") {
                    cands.push((n, s, 0));
                }
            }
            "return_statement" => cands.push((n, p, 1)),
            "init_declarator" => {
                if let Some(d) = c.parent(p).filter(|&d| c.kind(d) == "declaration" && c.children_by_field(d, "declarator").count() == 1) {
                    if c.parent(d).is_some_and(|b| c.kind(b) == "compound_statement") && !c.text(d).starts_with("const") {
                        cands.push((n, d, 2));
                    }
                }
            }
            _ => {}
        }
    }
    let (n, s, kind) = m.pick(&cands)?;
    let cond = c.child(n, "condition")?;
    let a = c.child(n, "consequence")?;
    let b = c.child(n, "alternative")?;
    let ct = strip_parens(c.text(cond)).to_string();
    let ind = indent_of(c, s);
    match kind {
        0 => {
            let a_ = c.parent(n)?;
            let lhs = c.text(c.child(a_, "left")?);
            Some(vec![Edit::replace(c, s, format!("if ({ct}) {{\n{ind}    {lhs} = {};\n{ind}}} else {{\n{ind}    {lhs} = {};\n{ind}}}", c.text(a), c.text(b)))])
        }
        1 => {
            if m.rng.chance(0.5) {
                Some(vec![Edit::replace(c, s, format!("if ({ct}) {{\n{ind}    return {};\n{ind}}} else {{\n{ind}    return {};\n{ind}}}", c.text(a), c.text(b)))])
            } else {
                Some(vec![replace_stmt_many(c, s, &[format!("if ({ct}) {{\n{ind}    return {};\n{ind}}}", c.text(a)), format!("return {};", c.text(b))])])
            }
        }
        _ => {
            let id = c.parent(n)?;
            let decl = c.child(id, "declarator")?;
            let (name, _) = func::declarator_name(c, id)?;
            let prefix = &c.src[c.nodes[s].start..c.nodes[id].start];
            Some(vec![Edit::replace(
                c,
                s,
                format!("{prefix}{};\n{ind}if ({ct}) {{\n{ind}    {name} = {};\n{ind}}} else {{\n{ind}    {name} = {};\n{ind}}}", c.text(decl), c.text(a), c.text(b)),
            )])
        }
    }
}

/// Single statement of a branch (unwrapping a one-statement block).
fn single(c: &Cst, n: usize) -> Option<usize> {
    if c.kind(n) == "compound_statement" {
        let s = stmts_of(c, n);
        (s.len() == 1).then(|| s[0])
    } else {
        Some(n)
    }
}

fn op_if_to_ternary(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands: Vec<(usize, usize, String)> = Vec::new();
    for (_, l) in m.lists(1) {
        for (i, &s) in l.iter().enumerate() {
            if c.kind(s) != "if_statement" {
                continue;
            }
            let (Some(ce), Some(cons)) = (cond_expr(c, s), c.child(s, "consequence")) else { continue };
            let Some(a) = single(c, cons) else { continue };
            let ct = ptext_bin(c, ce);
            match c.child(s, "alternative") {
                Some(alt) => {
                    let Some(b) = c.named(alt).first().and_then(|&x| single(c, x)) else { continue };
                    if let (Some((_, la, ra)), Some((_, lb, rb))) = (assign_parts(c, a), assign_parts(c, b)) {
                        if normalize(c.text(la)) == normalize(c.text(lb)) {
                            cands.push((s, s, format!("{} = {ct} ? {} : {};", c.text(la), ternary_branch(c, ra), ternary_branch(c, rb))));
                        }
                    } else if c.kind(a) == "return_statement" && c.kind(b) == "return_statement" {
                        if let (Some(&ra), Some(&rb)) = (c.named(a).first(), c.named(b).first()) {
                            cands.push((s, s, format!("return {ct} ? {} : {};", ternary_branch(c, ra), ternary_branch(c, rb))));
                        }
                    }
                }
                None => {
                    if c.kind(a) == "return_statement" && i + 1 < l.len() && c.kind(l[i + 1]) == "return_statement" {
                        if let (Some(&ra), Some(&rb)) = (c.named(a).first(), c.named(l[i + 1]).first()) {
                            cands.push((s, l[i + 1], format!("return {ct} ? {} : {};", ternary_branch(c, ra), ternary_branch(c, rb))));
                        }
                    }
                }
            }
        }
    }
    let (s, e, t) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit { start: c.nodes[s].start, end: c.nodes[e].end, text: t }])
}

fn and_operand(c: &Cst, n: usize) -> String {
    match c.kind(n) {
        "binary_expression" if c.op(n) == Some("||") => format!("({})", c.text(n)),
        "conditional_expression" | "assignment_expression" | "comma_expression" => format!("({})", c.text(n)),
        _ => c.text(n).to_string(),
    }
}

fn op_nested_if(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut split = Vec::new();
    let mut join = Vec::new();
    for s in m.of_kind(&["if_statement"]) {
        if c.child(s, "alternative").is_some() {
            continue;
        }
        let (Some(ce), Some(cons)) = (cond_expr(c, s), c.child(s, "consequence")) else { continue };
        let ce0 = if c.kind(ce) == "parenthesized_expression" { c.named(ce).first().copied().unwrap_or(ce) } else { ce };
        if c.kind(ce0) == "binary_expression" && c.op(ce0) == Some("&&") {
            split.push((s, ce0, cons));
        }
        if let Some(inner) = single(c, cons) {
            if c.kind(inner) == "if_statement" && c.child(inner, "alternative").is_none() && cond_expr(c, inner).is_some() {
                join.push((s, ce, inner));
            }
        }
    }
    if !split.is_empty() && (join.is_empty() || m.rng.chance(0.5)) {
        let (s, ce, cons) = split[m.rng.below(split.len())];
        let a = c.child(ce, "left")?;
        let b = c.child(ce, "right")?;
        let ind = indent_of(c, s);
        return Some(vec![Edit::replace(
            c,
            s,
            format!("if ({}) {{\n{ind}    if ({}) {}\n{ind}}}", strip_parens(c.text(a)), strip_parens(c.text(b)), c.text(cons)),
        )]);
    }
    let (s, ce, inner) = join.get(m.rng.below(join.len().max(1)))?.clone();
    let ice = cond_expr(c, inner)?;
    let icons = c.child(inner, "consequence")?;
    Some(vec![Edit::replace(c, s, format!("if ({} && {}) {}", and_operand(c, ce), and_operand(c, ice), c.text(icons)))])
}

fn is_zero(t: &str) -> bool {
    matches!(t.trim(), "0" | "NULL" | "nullptr" | "0.0f" | "0.f" | "0.0")
}

fn in_cond_context(c: &Cst, n: usize) -> bool {
    let Some(p) = c.parent(n) else { return false };
    match c.kind(p) {
        "condition_clause" => c.nodes[n].field == Some("value"),
        "parenthesized_expression" => c.parent(p).is_some_and(|g| c.kind(g) == "do_statement"),
        "for_statement" => c.nodes[n].field == Some("condition"),
        "unary_expression" => c.op(p) == Some("!"),
        "binary_expression" => matches!(c.op(p), Some("&&") | Some("||")),
        "conditional_expression" => c.nodes[n].field == Some("condition"),
        _ => false,
    }
}

fn op_cond_zero(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for n in m.nodes.clone() {
        if !c.nodes[n].named || !in_cond_context(c, n) {
            continue;
        }
        let k = c.kind(n);
        let wrap = c.parent(n).is_some_and(|p| c.kind(p) == "unary_expression");
        let repl = match k {
            "binary_expression" => {
                let op = c.op(n)?;
                let (l, r) = (c.child(n, "left")?, c.child(n, "right")?);
                if op == "!=" && is_zero(c.text(r)) {
                    if !m.rng.chance(0.15) {
                        continue;
                    }
                    if wrap { ptext(c, l) } else { c.text(l).to_string() }
                } else if op == "==" && is_zero(c.text(r)) {
                    format!("!{}", ptext(c, l))
                } else {
                    continue;
                }
            }
            "unary_expression" if c.op(n) == Some("!") => {
                let a = c.child(n, "argument")?;
                let t = format!("{} == 0", ptext_bin(c, a));
                if wrap { format!("({t})") } else { t }
            }
            _ => continue,
        };
        cands.push((n, repl));
    }
    let (n, r) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit::replace(c, n, r)])
}

// ------------------------------------------------------------------ Expr

fn op_flip_compare(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let cands: Vec<usize> = m
        .of_kind(&["binary_expression"])
        .into_iter()
        .filter(|&b| matches!(c.op(b), Some("<") | Some(">") | Some("<=") | Some(">=") | Some("==") | Some("!=")))
        .filter(|&b| !int_literal_operand(c, b))
        .collect();
    let b = m.pick(&cands)?;
    let op = match c.op(b)? {
        "<" => ">",
        ">" => "<",
        "<=" => ">=",
        ">=" => "<=",
        o => o,
    };
    let (l, r) = (c.child(b, "left")?, c.child(b, "right")?);
    Some(vec![Edit::replace(c, b, format!("{} {op} {}", c.text(r), c.text(l)))])
}

fn op_commutative(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let cands: Vec<usize> = m
        .of_kind(&["binary_expression"])
        .into_iter()
        .filter(|&b| matches!(c.op(b), Some("+") | Some("*") | Some("&") | Some("|") | Some("^") | Some("==") | Some("!=")))
        .filter(|&b| !int_literal_operand(c, b))
        .collect();
    let b = m.pick(&cands)?;
    let (l, r) = (c.child(b, "left")?, c.child(b, "right")?);
    let op = c.op(b)?;
    // `a + b + c` parses as (a + b) + c; swapping must keep grouping explicit.
    let lt = if c.kind(l) == "binary_expression" && c.op(l) != Some(op) { format!("({})", c.text(l)) } else { c.text(l).to_string() };
    let rt = if c.kind(r) == "binary_expression" { format!("({})", c.text(r)) } else { c.text(r).to_string() };
    Some(vec![Edit::replace(c, b, format!("{rt} {op} {lt}"))])
}

fn op_associative(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for b in m.of_kind(&["binary_expression"]) {
        let Some(op) = c.op(b) else { continue };
        if !matches!(op, "+" | "*" | "&" | "|" | "^") {
            continue;
        }
        let (Some(l), Some(r)) = (c.child(b, "left"), c.child(b, "right")) else { continue };
        if c.kind(l) == "binary_expression" && c.op(l) == Some(op) {
            cands.push((b, true));
        }
        let r0 = if c.kind(r) == "parenthesized_expression" { c.named(r).first().copied().unwrap_or(r) } else { r };
        if c.kind(r0) == "binary_expression" && c.op(r0) == Some(op) {
            cands.push((b, false));
        }
    }
    let (b, left) = m.pick(&cands)?;
    let op = c.op(b)?;
    let (l, r) = (c.child(b, "left")?, c.child(b, "right")?);
    let t = if left {
        let (a, bb) = (c.child(l, "left")?, c.child(l, "right")?);
        format!("{} {op} ({} {op} {})", c.text(a), c.text(bb), c.text(r))
    } else {
        let r0 = if c.kind(r) == "parenthesized_expression" { c.named(r).first().copied().unwrap_or(r) } else { r };
        let (bb, cc) = (c.child(r0, "left")?, c.child(r0, "right")?);
        format!("{} {op} {} {op} {}", c.text(l), c.text(bb), ptext_bin(c, cc))
    };
    Some(vec![Edit::replace(c, b, t)])
}

const COMPOUND_OPS: &[&str] = &["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"];

fn op_compound_assign(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for a in m.of_kind(&["assignment_expression"]) {
        let (Some(l), Some(r)) = (c.child(a, "left"), c.child(a, "right")) else { continue };
        if !pure(c, l) {
            continue;
        }
        let op = c.op(a).unwrap_or("=");
        if op == "=" {
            if c.kind(r) == "binary_expression" && COMPOUND_OPS.contains(&c.op(r).unwrap_or("")) {
                if let Some(rl) = c.child(r, "left") {
                    if normalize(c.text(rl)) == normalize(c.text(l)) {
                        let rr = c.child(r, "right").unwrap();
                        cands.push((a, format!("{} {}= {}", c.text(l), c.op(r).unwrap(), c.text(rr))));
                    }
                }
            }
        } else if let Some(bop) = op.strip_suffix('=') {
            if COMPOUND_OPS.contains(&bop) {
                cands.push((a, format!("{} = {} {bop} {}", c.text(l), c.text(l), ptext_bin(c, r))));
            }
        }
    }
    let (a, t) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit::replace(c, a, t)])
}

fn op_incr_form(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for n in m.of_kind(&["update_expression", "assignment_expression"]) {
        let p = c.parent(n)?;
        let stmt_level = c.kind(p) == "expression_statement" || (c.kind(p) == "for_statement" && c.nodes[n].field == Some("update"));
        if !stmt_level {
            continue;
        }
        let (v, plus) = match c.kind(n) {
            "update_expression" => {
                let a = c.child(n, "argument")?;
                (c.text(a).to_string(), c.op(n) == Some("++"))
            }
            _ => {
                let (l, r) = (c.child(n, "left")?, c.child(n, "right")?);
                let op = c.op(n)?;
                if (op == "+=" || op == "-=") && c.text(r) == "1" {
                    (c.text(l).to_string(), op == "+=")
                } else if op == "=" && c.kind(r) == "binary_expression" && matches!(c.op(r), Some("+") | Some("-")) {
                    let (rl, rr) = (c.child(r, "left")?, c.child(r, "right")?);
                    if normalize(c.text(rl)) == normalize(c.text(l)) && c.text(rr) == "1" {
                        (c.text(l).to_string(), c.op(r) == Some("+"))
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
        };
        if !pure(c, c.child(n, if c.kind(n) == "update_expression" { "argument" } else { "left" })?) {
            continue;
        }
        let (o, oo) = if plus { ("++", "+") } else { ("--", "-") };
        let forms = [format!("{v}{o}"), format!("{o}{v}"), format!("{v} {oo}= 1"), format!("{v} = {v} {oo} 1")];
        let cur = normalize(c.text(n));
        for f in forms {
            if normalize(&f) != cur {
                cands.push((n, f));
            }
        }
    }
    let (n, t) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit::replace(c, n, t)])
}

fn op_demorgan(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for n in m.of_kind(&["unary_expression"]) {
        if c.op(n) != Some("!") {
            continue;
        }
        let a = c.child(n, "argument")?;
        if c.kind(a) != "parenthesized_expression" {
            continue;
        }
        let Some(&b) = c.named(a).first() else { continue };
        if c.kind(b) == "binary_expression" && matches!(c.op(b), Some("&&") | Some("||")) {
            let op = if c.op(b) == Some("&&") { "||" } else { "&&" };
            let (l, r) = (c.child(b, "left")?, c.child(b, "right")?);
            let wrap = |s: String| if s.contains("&&") || s.contains("||") { format!("({s})") } else { s };
            cands.push((n, format!("({} {op} {})", wrap(negate(c, l)), wrap(negate(c, r)))));
        }
    }
    let (n, t) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit::replace(c, n, t)])
}

fn op_struct_ref(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for f in m.of_kind(&["field_expression"]) {
        let a = c.child(f, "argument")?;
        let fld = c.child(f, "field")?;
        if c.op(f) == Some("->") && !c.text(a).starts_with("this") {
            cands.push((f, format!("(*{}).{}", ptext(c, a), c.text(fld))));
        } else if c.op(f) == Some(".") && c.kind(a) == "parenthesized_expression" {
            if let Some(&p) = c.named(a).first() {
                if c.kind(p) == "pointer_expression" && c.op(p) == Some("*") {
                    let pa = c.child(p, "argument")?;
                    cands.push((f, format!("{}->{}", c.text(pa), c.text(fld))));
                }
            }
        }
    }
    let (f, t) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit::replace(c, f, t)])
}


/// An integer local only ever used as a pointer (`(T*)v`) becomes `T* v`: assignments get a
/// `(T*)` cast, the casts at uses are dropped. Pointer vs int changes compares (`cmplwi`).
fn op_pointerize_local(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for (name, var) in &m.info.vars {
        if var.is_param || !is_int_type(&var.ty) && var.ty != "void*" {
            continue;
        }
        let d = var.decl;
        let ds: Vec<usize> = c.children_by_field(d, "declarator").collect();
        if ds.len() != 1 || c.text(d).trim_start().starts_with("const") || c.text(d).trim_start().starts_with("static") {
            continue;
        }
        let uses = uses_of(c, m.info.body, name);
        let mut casts: Vec<(usize, String)> = Vec::new();
        let mut assigns = Vec::new();
        let mut ok = true;
        for &u in &uses {
            let p = c.parent(u).unwrap_or(u);
            match c.kind(p) {
                "cast_expression" => {
                    let t = c.child(p, "type").map(|t| c.text(t).replace(' ', "")).unwrap_or_default();
                    if t.ends_with('*') && !t.starts_with("char") && !t.starts_with("void") && !t.starts_with("unsignedchar") {
                        casts.push((p, t));
                    }
                }
                "assignment_expression" if c.nodes[u].field == Some("left") => {
                    if c.op(p) != Some("=") {
                        ok = false;
                    } else {
                        assigns.push(c.child(p, "right").unwrap());
                    }
                }
                "binary_expression" if matches!(c.op(p), Some("+") | Some("-") | Some("*") | Some("/") | Some("<<") | Some(">>") | Some("&") | Some("|") | Some("^") | Some("%")) => ok = false,
                "update_expression" => ok = false,
                "assignment_expression" if c.op(p) != Some("=") => ok = false,
                _ => {}
            }
        }
        if !ok || casts.is_empty() {
            continue;
        }
        let ty = casts[0].1.clone();
        if !casts.iter().all(|x| x.1 == ty) {
            continue;
        }
        cands.push((d, ds[0], name.clone(), ty, casts.into_iter().map(|x| x.0).collect::<Vec<_>>(), assigns));
    }
    cands.sort_by(|a, b| a.2.cmp(&b.2));
    let (d, decl, name, ty, casts, assigns) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    let base = ty.trim_end_matches('*');
    let mut edits = Vec::new();
    match c.child(decl, "value") {
        Some(v) if c.kind(decl) == "init_declarator" => edits.push(Edit::replace(c, d, format!("{base}* {name} = ({ty}){};", ptext(c, v)))),
        _ => edits.push(Edit::replace(c, d, format!("{base}* {name};"))),
    }
    for a in assigns {
        if c.kind(a) == "number_literal" && c.text(a) == "0" {
            continue;
        }
        edits.push(Edit::replace(c, a, format!("({ty}){}", ptext_bin(c, a))));
    }
    for k in casts {
        let paren = c.parent(k).is_some_and(|p| matches!(c.kind(p), "field_expression" | "subscript_expression" | "call_expression"));
        let _ = paren;
        edits.push(Edit::replace(c, k, name.clone()));
    }
    Some(edits)
}

/// `(x)` -> `x` for primary expressions (no codegen effect; cleanup).
fn op_strip_parens(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let cands: Vec<usize> = m
        .of_kind(&["parenthesized_expression"])
        .into_iter()
        .filter(|&p| c.named(p).first().is_some_and(|&i| is_primary(c.kind(i)) || c.parent(p).is_some_and(|g| matches!(c.kind(g), "init_declarator" | "return_statement" | "expression_statement" | "argument_list"))))
        .filter(|&p| c.parent(p).is_some_and(|g| !matches!(c.kind(g), "condition_clause" | "do_statement" | "sizeof_expression")))
        .collect();
    let p = m.pick(&cands)?;
    let i = *c.named(p).first()?;
    Some(vec![Edit::replace(c, p, c.text(i).to_string())])
}

/// Declare named-class locals in the target's declaration order (regalloc.md rule 3: the local
/// in the highest callee-saved register is declared first). Uses the draft's register-named
/// locals (`var_r31`) and the target hints; moves one out-of-order declaration up.
fn op_hint_order(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let hints = m.hints?;
    let mut cands = Vec::new();
    for (_, l) in m.lists(2) {
        // (list index, rank, decl node, declarator, name)
        let mut named: Vec<(usize, u32, usize, usize, String)> = Vec::new();
        for (i, &s) in l.iter().enumerate() {
            if c.kind(s) != "declaration" {
                continue;
            }
            let ds: Vec<usize> = c.children_by_field(s, "declarator").collect();
            if ds.len() != 1 {
                continue;
            }
            let Some((name, _)) = func::declarator_name(c, ds[0]) else { continue };
            if let Some(HintClass::Named(rank)) = hints.for_var(&name) {
                named.push((i, rank, s, ds[0], name));
            }
        }
        for a in &named {
            for b in &named {
                // a must come before b but is declared after it.
                if a.1 < b.1 && a.0 > b.0 && !l[b.0..a.0].iter().any(|&t| t != a.2 && mentions(c, t, &a.4)) {
                    cands.push((a.clone(), l[b.0]));
                }
            }
        }
    }
    let ((_, _, s, d, name), at) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    let ind = indent_of(c, at);
    let prefix = &c.src[c.nodes[s].start..c.nodes[d].start];
    if c.kind(d) == "init_declarator" {
        let decl = c.child(d, "declarator")?;
        let val = c.child(d, "value")?;
        Some(vec![
            Edit::insert(c.nodes[at].start, format!("{prefix}{};\n{ind}", c.text(decl))),
            Edit::replace(c, s, format!("{name} = {};", c.text(val))),
        ])
    } else {
        Some(vec![Edit::insert(c.nodes[at].start, format!("{}\n{ind}", c.text(s))), remove_stmt(c, s)])
    }
}

/// A local whose target register holds a TEMP-class value: inline it (regalloc.md rule 4,
/// named -> temp: remove its move-uses by computing it at the use site).
fn op_hint_temp(m: &mut M) -> Option<Vec<Edit>> {
    let hints = m.hints?;
    let mut names: Vec<String> = m
        .info
        .vars
        .keys()
        .filter(|n| !m.info.vars[*n].is_param && matches!(hints.for_var(n), Some(HintClass::Temp(_))))
        .cloned()
        .collect();
    names.sort();
    if names.is_empty() {
        return None;
    }
    let pick = names[m.rng.below(names.len())].clone();
    m.focus = Some(pick);
    let r = if m.rng.chance(0.5) { op_inline_var_all(m).or_else(|| op_inline_temp(m)) } else { op_inline_temp(m).or_else(|| op_inline_var_all(m)) };
    m.focus = None;
    r
}

fn op_never(_m: &mut M) -> Option<Vec<Edit>> {
    None
}

fn int_literal_operand(c: &Cst, b: usize) -> bool {
    [c.child(b, "left"), c.child(b, "right")].into_iter().flatten().any(|o| {
        c.kind(o) == "number_literal" && {
            let t = c.text(o).to_ascii_lowercase();
            t.starts_with("0x") || !(t.contains('.') || t.ends_with('f') || t.contains('e'))
        }
    })
}

/// Declare first, assign later: `... T x = e;` -> `T x; ... x = e;` with the declaration moved
/// up (before other declarations/statements that don't mention x). MWCC assigns saved registers
/// and stack slots by declaration order (catalog row 2).
fn op_hoist_decl(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for (_, l) in m.lists(2) {
        for (i, &s) in l.iter().enumerate() {
            if c.kind(s) != "declaration" || i == 0 {
                continue;
            }
            let t = c.text(s).trim_start();
            if t.starts_with("const") || t.starts_with("static") {
                continue;
            }
            let ds: Vec<usize> = c.children_by_field(s, "declarator").collect();
            if ds.len() != 1 {
                continue;
            }
            let Some((name, suf)) = func::declarator_name(c, ds[0]) else { continue };
            if suf.contains('&') || suf.contains('[') {
                continue;
            }
            let ty = c.child(s, "type").map(|x| c.text(x)).unwrap_or("");
            let init = c.kind(ds[0]) == "init_declarator";
            if init && !(is_scalar_type(ty) || suf.contains('*')) {
                continue;
            }
            if init && c.child(ds[0], "value").is_some_and(|v| matches!(c.kind(v), "initializer_list" | "argument_list")) {
                continue;
            }
            let mut t = i;
            while t > 0 && !mentions(c, l[t - 1], &name) && c.kind(l[t - 1]) != "labeled_statement" {
                t -= 1;
                cands.push((s, ds[0], l[t], init, name.clone()));
            }
        }
    }
    let (s, d, at, init, name) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    let ind = indent_of(c, at);
    let prefix = &c.src[c.nodes[s].start..c.nodes[d].start];
    if init {
        let decl = c.child(d, "declarator")?;
        let val = c.child(d, "value")?;
        Some(vec![
            Edit::insert(c.nodes[at].start, format!("{prefix}{};\n{ind}", c.text(decl))),
            Edit::replace(c, s, format!("{name} = {};", c.text(val))),
        ])
    } else {
        Some(vec![Edit::insert(c.nodes[at].start, format!("{}\n{ind}", c.text(s))), remove_stmt(c, s)])
    }
}

fn store_parts(c: &Cst, s: usize) -> Option<(usize, usize)> {
    if c.kind(s) != "expression_statement" {
        return None;
    }
    let a = *c.named(s).first()?;
    if c.kind(a) != "assignment_expression" {
        return None;
    }
    let l = c.child(a, "left")?;
    matches!(c.kind(l), "field_expression" | "subscript_expression" | "pointer_expression").then_some((a, l))
}

/// Swap adjacent call-free stores to different lvalues (stores are emitted in source order,
/// catalog row 18). Aliasing between the stored objects is not checked: the target decides.
fn op_swap_stores(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for (_, l) in m.lists(2) {
        for w in l.windows(2) {
            let (Some((a, la)), Some((b, lb))) = (store_parts(c, w[0]), store_parts(c, w[1])) else { continue };
            if has_kind(c, a, &["call_expression"]) || has_kind(c, b, &["call_expression"]) {
                continue;
            }
            let (na, nb) = (normalize(c.text(la)), normalize(c.text(lb)));
            if na == nb || normalize(c.text(b)).contains(&na) || normalize(c.text(a)).contains(&nb) {
                continue;
            }
            let (ea, eb) = (m.eff(w[0]), m.eff(w[1]));
            if ea.writes.iter().any(|x| eb.reads.contains(x) || eb.writes.contains(x)) || eb.writes.iter().any(|x| ea.reads.contains(x)) {
                continue;
            }
            cands.push((w[0], w[1]));
        }
    }
    let (a, b) = m.pick(&cands)?;
    Some(vec![Edit::replace(c, a, c.text(b)), Edit::replace(c, b, c.text(a))])
}

/// Toggle top-level `const` on a by-value scalar parameter (mangling unchanged; catalog row 9a).
fn op_param_const(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let fd = c.descendants(c.child(m.info.def, "declarator")?).into_iter().find(|&n| c.kind(n) == "function_declarator")?;
    let pl = c.child(fd, "parameters")?;
    let cands: Vec<usize> = c
        .named(pl)
        .into_iter()
        .filter(|&p| c.kind(p) == "parameter_declaration")
        .filter(|&p| c.child(p, "declarator").is_some_and(|d| c.kind(d) == "identifier"))
        .filter(|&p| c.child(p, "type").is_some_and(|t| is_scalar_type(c.text(t))))
        .collect();
    let p = m.pick(&cands)?;
    let t = c.text(p);
    match t.strip_prefix("const ") {
        Some(rest) => Some(vec![Edit::replace(c, p, rest.trim_start())]),
        None => Some(vec![Edit::insert(c.nodes[p].start, "const ")]),
    }
}

fn complement(op: &str) -> Option<&'static str> {
    Some(match op {
        "<" => ">=",
        ">=" => "<",
        ">" => "<=",
        "<=" => ">",
        _ => return None,
    })
}

/// `!(a < b)` <-> `a >= b` (block order differs, catalog row 6; for floats NaN semantics differ
/// and so does codegen, the target decides).
fn op_push_not(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for n in m.of_kind(&["unary_expression"]) {
        if c.op(n) != Some("!") {
            continue;
        }
        let Some(a) = c.child(n, "argument") else { continue };
        if c.kind(a) != "parenthesized_expression" {
            continue;
        }
        let Some(&b) = c.named(a).first() else { continue };
        if c.kind(b) != "binary_expression" {
            continue;
        }
        if let Some(op) = c.op(b).and_then(complement) {
            let (l, r) = (c.child(b, "left")?, c.child(b, "right")?);
            cands.push((n, format!("{} {op} {}", c.text(l), c.text(r))));
        }
    }
    for b in m.of_kind(&["binary_expression"]) {
        if let Some(op) = c.op(b).and_then(complement) {
            if c.parent(b).is_some_and(|p| c.kind(p) == "parenthesized_expression" && c.parent(p).is_some_and(|g| c.kind(g) == "unary_expression")) {
                continue;
            }
            let (l, r) = (c.child(b, "left")?, c.child(b, "right")?);
            cands.push((b, format!("!({} {op} {})", c.text(l), c.text(r))));
        }
    }
    let (n, t) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit::replace(c, n, t)])
}

/// `if (c) x = a; else x = b;` <-> `x = b; if (c) x = a;` for locals (init-then-overwrite hoists
/// the constant above the branch, catalog row 7).
fn op_select_init(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut fwd = Vec::new();
    let mut back = Vec::new();
    for (_, l) in m.lists(1) {
        for (i, &s) in l.iter().enumerate() {
            if c.kind(s) != "if_statement" {
                continue;
            }
            let (Some(ce), Some(cons)) = (cond_expr(c, s), c.child(s, "consequence")) else { continue };
            let Some(a) = single(c, cons) else { continue };
            let Some((_, la, ra)) = assign_parts(c, a) else { continue };
            if c.kind(la) != "identifier" || !m.info.is_private(c.text(la)) {
                continue;
            }
            let x = c.text(la);
            if mentions(c, ce, x) || mentions(c, ra, x) {
                continue;
            }
            match c.child(s, "alternative") {
                Some(alt) => {
                    let Some(b) = c.named(alt).first().and_then(|&y| single(c, y)) else { continue };
                    let Some((_, lb, rb)) = assign_parts(c, b) else { continue };
                    if c.text(lb) == x && !mentions(c, rb, x) {
                        fwd.push((s, x.to_string(), ce, ra, rb));
                    }
                }
                None => {
                    if i > 0 {
                        if let Some((_, lp, rp)) = assign_parts(c, l[i - 1]) {
                            if c.text(lp) == x && !mentions(c, rp, x) {
                                back.push((l[i - 1], s, x.to_string(), ce, ra, rp));
                            }
                        }
                    }
                }
            }
        }
    }
    if !fwd.is_empty() && (back.is_empty() || m.rng.chance(0.5)) {
        let (s, x, ce, ra, rb) = fwd[m.rng.below(fwd.len())].clone();
        let ind = indent_of(c, s);
        return Some(vec![replace_stmt_many(
            c,
            s,
            &[format!("{x} = {};", c.text(rb)), format!("if ({}) {{\n{ind}    {x} = {};\n{ind}}}", c.text(ce), c.text(ra))],
        )]);
    }
    let (p, s, x, ce, ra, rp) = back.get(m.rng.below(back.len().max(1)))?.clone();
    let ind = indent_of(c, s);
    Some(vec![Edit {
        start: c.nodes[p].start,
        end: c.nodes[s].end,
        text: format!("if ({}) {{\n{ind}    {x} = {};\n{ind}}} else {{\n{ind}    {x} = {};\n{ind}}}", c.text(ce), c.text(ra), c.text(rp)),
    }])
}

/// `while (c) B` <-> `while (1) { if (!c) break; B }` (test at the top, catalog row 12); also
/// `for (i; c; u)` -> `for (i;; u) { if (!c) break; ... }`.
fn op_loop_break(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut fwd = Vec::new();
    let mut back = Vec::new();
    for w in m.of_kind(&["while_statement", "for_statement"]) {
        let body = c.child(w, "body")?;
        let ce = if c.kind(w) == "while_statement" { cond_expr(c, w) } else { c.child(w, "condition") };
        match ce {
            Some(ce) if !matches!(c.text(ce).trim(), "1" | "true") => fwd.push((w, ce, body)),
            _ => {
                // Infinite loop whose body starts with `if (X) break;`.
                if c.kind(w) == "while_statement" && c.kind(body) == "compound_statement" {
                    let ss = stmts_of(c, body);
                    if let Some(&first) = ss.first() {
                        if c.kind(first) == "if_statement" && c.child(first, "alternative").is_none() {
                            if let (Some(x), Some(cons)) = (cond_expr(c, first), c.child(first, "consequence")) {
                                if single(c, cons).is_some_and(|b| c.kind(b) == "break_statement") {
                                    back.push((w, x, body, first));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if !fwd.is_empty() && (back.is_empty() || m.rng.chance(0.5)) {
        let (w, ce, body) = fwd[m.rng.below(fwd.len())];
        let ind = indent_of(c, w);
        let inner = inner_block(c, body);
        let neg = negate(c, ce);
        let head = if c.kind(w) == "while_statement" {
            "while (1)".to_string()
        } else {
            let init = c.child(w, "initializer").map(|x| c.text(x).trim_end_matches(';').to_string()).unwrap_or_default();
            let upd = c.child(w, "update").map(|x| c.text(x).to_string()).unwrap_or_default();
            format!("for ({init};; {upd})")
        };
        return Some(vec![Edit::replace(c, w, format!("{head} {{\n{ind}    if ({neg}) {{\n{ind}        break;\n{ind}    }}\n{ind}    {inner}\n{ind}}}"))]);
    }
    let (w, x, body, first) = back.get(m.rng.below(back.len().max(1)))?.clone();
    let ind = indent_of(c, w);
    let rest_start = c.src[c.nodes[first].end..].find(|ch: char| !ch.is_whitespace()).map(|k| c.nodes[first].end + k)?;
    let close = c.nodes[body].end - 1;
    let rest = if rest_start < close { c.src[rest_start..close].trim_end() } else { "" };
    Some(vec![Edit::replace(c, w, format!("while ({}) {{\n{ind}    {rest}\n{ind}}}", negate(c, x)))])
}

/// `for (...; i < n; i++)` <-> `i != n`.
fn op_loop_cond_ne(m: &mut M) -> Option<Vec<Edit>> {
    let c = m.cst;
    let mut cands = Vec::new();
    for f in m.of_kind(&["for_statement"]) {
        let (Some(ce), Some(up)) = (c.child(f, "condition"), c.child(f, "update")) else { continue };
        if c.kind(ce) != "binary_expression" {
            continue;
        }
        let Some(l) = c.child(ce, "left") else { continue };
        let ut = normalize(c.text(up));
        let v = c.text(l);
        if !(ut == format!("{v}++") || ut == format!("++{v}") || ut == format!("{v}+=1")) {
            continue;
        }
        let new = match c.op(ce) {
            Some("<") => "!=",
            Some("!=") => "<",
            _ => continue,
        };
        let r = c.child(ce, "right")?;
        cands.push((ce, format!("{v} {new} {}", c.text(r))));
    }
    let (n, t) = cands.get(m.rng.below(cands.len().max(1)))?.clone();
    Some(vec![Edit::replace(c, n, t)])
}

// ------------------------------------------------------------------ driver

/// Result of one successful mutation.
#[derive(Clone, Debug)]
pub struct Mutation {
    pub src: String,
    pub op: usize,
}

/// Parse `src`, locate the function for `symbol`, and apply operator `op` once.
pub fn apply_op(src: &str, symbol: &str, op: usize, rng: &mut Rng) -> Option<String> {
    apply_op_with(src, symbol, op, rng, None)
}

/// [`apply_op`] with target register hints.
pub fn apply_op_with(src: &str, symbol: &str, op: usize, rng: &mut Rng, hints: Option<&RegHints>) -> Option<String> {
    let p = Parsed::new(src, symbol)?;
    p.apply(op, rng, hints)
}

/// Apply a variable-centric operator (inline_temp, inline_var_all) to local `focus` only.
pub fn apply_op_focused(src: &str, symbol: &str, op: usize, rng: &mut Rng, focus: &str) -> Option<String> {
    Parsed::new(src, symbol)?.apply_focus(op, rng, None, Some(focus))
}

/// A parsed candidate (CST + analysis of the target function), reusable across operator tries.
pub struct Parsed<'s> {
    src: &'s str,
    symbol: &'s str,
    cst: Cst,
    info: FuncInfo,
    nodes: Vec<usize>,
}

impl<'s> Parsed<'s> {
    pub fn new(src: &'s str, symbol: &'s str) -> Option<Parsed<'s>> {
        let cst = Cst::parse(src);
        let def = func::find_target(&cst, symbol)?;
        let info = FuncInfo::analyze(&cst, def)?;
        let nodes = cst.descendants(info.body);
        Some(Parsed { src, symbol, cst, info, nodes })
    }

    pub fn apply(&self, op: usize, rng: &mut Rng, hints: Option<&RegHints>) -> Option<String> {
        self.apply_focus(op, rng, hints, None)
    }

    /// [`Parsed::apply`] restricted to one local for variable-centric operators.
    pub fn apply_focus(&self, op: usize, rng: &mut Rng, hints: Option<&RegHints>, focus: Option<&str>) -> Option<String> {
        self.apply_in(op, rng, hints, focus, None)
    }

    /// Apply `op` with sites restricted to nodes overlapping `ranges` (byte ranges of the source,
    /// e.g. the lines that produce differing instructions). Enclosing blocks overlap too, so
    /// statement-list operators still see the surrounding statements.
    pub fn apply_in(&self, op: usize, rng: &mut Rng, hints: Option<&RegHints>, focus: Option<&str>, ranges: Option<&[(usize, usize)]>) -> Option<String> {
        let nodes = match ranges {
            Some(rs) if !rs.is_empty() => self
                .nodes
                .iter()
                .copied()
                .filter(|&n| {
                    let (s, e) = (self.cst.nodes[n].start, self.cst.nodes[n].end);
                    rs.iter().any(|&(a, b)| s < b && a < e)
                })
                .collect(),
            _ => self.nodes.clone(),
        };
        let mut m = M { cst: &self.cst, info: &self.info, rng, nodes, hints, focus: focus.map(String::from) };
        let edits = (OPS[op].f)(&mut m)?;
        let out = apply(self.src, &edits)?;
        if out == self.src {
            return None;
        }
        let re = Cst::parse(&out);
        if re.errors > self.cst.errors || func::find_target(&re, self.symbol).is_none() {
            return None;
        }
        Some(out)
    }
}

/// Pick an operator by weight and apply it; retries other operators when one has no site.
pub fn mutate(src: &str, symbol: &str, weights: &[f64], rng: &mut Rng) -> Option<Mutation> {
    mutate_with(src, symbol, weights, rng, None)
}

/// [`mutate`] with target register hints.
pub fn mutate_with(src: &str, symbol: &str, weights: &[f64], rng: &mut Rng, hints: Option<&RegHints>) -> Option<Mutation> {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let p = Parsed::new(src, symbol)?;
    for _ in 0..12 {
        let mut x = rng.f64() * total;
        let mut op = 0;
        for (i, w) in weights.iter().enumerate() {
            if x < *w {
                op = i;
                break;
            }
            x -= w;
            op = i;
        }
        if let Some(s) = p.apply(op, rng, hints) {
            return Some(Mutation { src: s, op });
        }
    }
    None
}

/// [`mutate_with`] with sites restricted to byte `ranges` of `src` (see [`Parsed::apply_in`]).
pub fn mutate_in(src: &str, symbol: &str, weights: &[f64], rng: &mut Rng, hints: Option<&RegHints>, ranges: &[(usize, usize)]) -> Option<Mutation> {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let p = Parsed::new(src, symbol)?;
    for _ in 0..12 {
        let mut x = rng.f64() * total;
        let mut op = 0;
        for (i, w) in weights.iter().enumerate() {
            if x < *w {
                op = i;
                break;
            }
            x -= w;
            op = i;
        }
        if let Some(s) = p.apply_in(op, rng, hints, None, Some(ranges)) {
            return Some(Mutation { src: s, op });
        }
    }
    None
}

pub fn base_weights() -> Vec<f64> {
    OPS.iter().map(|o| o.weight).collect()
}

