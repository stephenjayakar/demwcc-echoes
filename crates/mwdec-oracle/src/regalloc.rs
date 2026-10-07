//! Model of MWCC GC/2.7's register colouring (BackEnd/PowerPC/RegisterAllocator/Coloring.c,
//! InterferenceGraph.c, RegisterInfo.c, CodeGen.c allocate_local_vregs, CopyPropagation.c), applied
//! to an abstract interference graph of values ("nodes").
//!
//! Virtual register numbering (allocate_local_vregs + codegen):
//!   params (in parameter order, `this` first) < named locals (REVERSE declaration order: the
//!   `locals` list is built by prepending) < front-end temps < codegen temps (creation order).
//! A named local whose definition is a copy `x = <temp>` and whose every use is a non-move
//! instruction is copy-propagated away (CopyPropagation.c: copies never propagate into moves), so
//! the value lives in the temp: classify it as `Temp`, not `Named`.
//!
//! Colouring: simplify pushes nodes with degree < K in ascending vreg order (repeated passes;
//! pushing decrements neighbours' degrees); select pops (so normally the HIGHEST vreg is coloured
//! first) and gives each node the lowest-numbered register in (volatiles + nonvolatiles obtained so
//! far) minus registers of already-coloured neighbours; when empty, it obtains the next
//! nonvolatile: r31, r30, ... r14 (f31 ... f14). Values live across a call interfere with every
//! volatile register, so they only ever get nonvolatiles.

use crate::webs::RegId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// parameter index (this = 0)
    Param(u32),
    /// declaration index (0 = first declared in the function)
    Named(u32),
    /// front-end temp (CSE / inline-expansion temps), creation index
    FeTemp(u32),
    /// codegen temp, creation index (evaluation order)
    Temp(u32),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub class: Class,
    pub float: bool,
    pub crosses_call: bool,
    /// indices of interfering nodes (same register class only are considered)
    pub interferes: Vec<usize>,
    /// extra degree from values we don't model (short temps); usually 0
    pub extra_degree: u32,
}

/// vreg numbers for nodes (higher = coloured earlier in the normal case).
pub fn numbering(nodes: &[Node]) -> Vec<u32> {
    let mut idx: Vec<usize> = (0..nodes.len()).collect();
    let key = |c: Class| -> (u32, i64) {
        match c {
            Class::Param(i) => (0, i as i64),
            Class::Named(d) => (1, -(d as i64)), // reverse declaration order
            Class::FeTemp(i) => (2, -(i as i64)), // FE temps live in the locals list too (prepended)
            Class::Temp(i) => (3, i as i64),
        }
    };
    idx.sort_by_key(|&i| key(nodes[i].class));
    let mut num = vec![0u32; nodes.len()];
    for (k, &i) in idx.iter().enumerate() {
        num[i] = 32 + k as u32;
    }
    num
}

pub const GPR_K: u32 = 29; // 32 minus r1, r2, r13 (reserved on EABI)
pub const FPR_K: u32 = 32;
const GPR_VOLATILE_COUNT: u32 = 11; // r0, r3..r12
const FPR_VOLATILE_COUNT: u32 = 14; // f0..f13

/// Colour nodes with the MWCC algorithm. Returns register per node (r14..r31 / f14..f31 as RegId,
/// volatile picks as their RegId, or None when spilled).
pub fn color(nodes: &[Node], num: &[u32]) -> Vec<Option<RegId>> {
    let mut res = vec![None; nodes.len()];
    for float in [false, true] {
        let ids: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].float == float).collect();
        if ids.is_empty() {
            continue;
        }
        let k = if float { FPR_K } else { GPR_K };
        let vol_count = if float { FPR_VOLATILE_COUNT } else { GPR_VOLATILE_COUNT };
        // simplify order: ascending vreg
        let mut order = ids.clone();
        order.sort_by_key(|&i| num[i]);
        let mut degree: Vec<i64> = vec![0; nodes.len()];
        for &i in &ids {
            let n = &nodes[i];
            degree[i] = n.interferes.iter().filter(|&&j| nodes[j].float == float).count() as i64
                + if n.crosses_call { vol_count as i64 } else { 0 }
                + n.extra_degree as i64;
        }
        let mut pushed = vec![false; nodes.len()];
        let mut stack: Vec<usize> = Vec::new();
        loop {
            let mut progress = true;
            let mut spill_cands: Vec<usize> = vec![];
            while progress {
                progress = false;
                spill_cands.clear();
                for &i in &order {
                    if pushed[i] {
                        continue;
                    }
                    if degree[i] < k as i64 {
                        for &j in &nodes[i].interferes {
                            degree[j] -= 1;
                        }
                        pushed[i] = true;
                        stack.push(i);
                        progress = true;
                    } else {
                        spill_cands.push(i);
                    }
                }
            }
            if spill_cands.is_empty() {
                break;
            }
            // spill heuristic needs spill costs we don't have: push the highest-degree node
            let best = *spill_cands.iter().max_by_key(|&&i| degree[i]).unwrap();
            for &j in &nodes[best].interferes {
                degree[j] -= 1;
            }
            pushed[best] = true;
            stack.push(best);
        }
        // select
        let nonvol: Vec<RegId> = if float { (46u8..=63).rev().collect() } else { (14u8..=31).rev().collect() };
        let mut obtained: Vec<RegId> = Vec::new();
        let volatile: Vec<RegId> = if float { (32u8..46).collect() } else { [0u8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].to_vec() };
        while let Some(i) = stack.pop() {
            let n = &nodes[i];
            let taken: Vec<RegId> = n.interferes.iter().filter_map(|&j| res[j]).collect();
            let mut cands: Vec<RegId> = Vec::new();
            if !n.crosses_call {
                cands.extend(volatile.iter().copied());
            }
            cands.extend(obtained.iter().copied());
            cands.sort();
            let pick = cands.into_iter().find(|r| !taken.contains(r));
            res[i] = match pick {
                Some(r) => Some(r),
                None => {
                    if obtained.len() < nonvol.len() {
                        let r = nonvol[obtained.len()];
                        obtained.push(r);
                        Some(r)
                    } else {
                        None
                    }
                }
            };
        }
    }
    res
}

/// Convenience: classes -> predicted registers.
pub fn predict(nodes: &[Node]) -> Vec<Option<RegId>> {
    color(nodes, &numbering(nodes))
}

/// Like [`solve_order`] but every pair `(a, b)` in `before` must be coloured a-then-b.
pub fn solve_order_constrained(nodes: &[Node], observed: &[RegId], before: &[(usize, usize)]) -> Option<Vec<usize>> {
    fn rec(
        nodes: &[Node],
        observed: &[RegId],
        before: &[(usize, usize)],
        assigned: &mut Vec<Option<RegId>>,
        obtained: &mut Vec<RegId>,
        order: &mut Vec<usize>,
        nonvol: &[RegId],
        budget: &mut u64,
    ) -> bool {
        if order.len() == nodes.len() {
            return true;
        }
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        for i in 0..nodes.len() {
            if assigned[i].is_some() {
                continue;
            }
            if before.iter().any(|&(a, b)| b == i && assigned[a].is_none()) {
                continue;
            }
            let taken: Vec<RegId> = nodes[i].interferes.iter().filter_map(|&j| assigned[j]).collect();
            let mut free: Vec<RegId> = obtained.iter().copied().filter(|r| !taken.contains(r)).collect();
            free.sort();
            let (pick, new) = match free.first() {
                Some(&r) => (r, false),
                None => match nonvol.get(obtained.len()) {
                    Some(&r) => (r, true),
                    None => continue,
                },
            };
            if pick != observed[i] {
                continue;
            }
            assigned[i] = Some(pick);
            if new {
                obtained.push(pick);
            }
            order.push(i);
            if rec(nodes, observed, before, assigned, obtained, order, nonvol, budget) {
                return true;
            }
            order.pop();
            if new {
                obtained.pop();
            }
            assigned[i] = None;
        }
        false
    }
    let float = nodes.first().map_or(false, |n| n.float);
    let nonvol: Vec<RegId> = if float { (46u8..=63).rev().collect() } else { (14u8..=31).rev().collect() };
    let mut assigned = vec![None; nodes.len()];
    let mut obtained = vec![];
    let mut order = vec![];
    let mut budget = 2_000_000u64;
    if rec(nodes, observed, before, &mut assigned, &mut obtained, &mut order, &nonvol, &mut budget) {
        Some(order)
    } else {
        None
    }
}

/// Inverse problem: find a colouring priority order (highest vreg first) that reproduces the
/// observed registers, ignoring the K-degree effect. Returns all node indices in a valid
/// colouring order (first = coloured first) if one exists. Uses backtracking with the greedy
/// "lowest free obtained register, else next new" rule.
pub fn solve_order(nodes: &[Node], observed: &[RegId]) -> Option<Vec<usize>> {
    fn rec(
        nodes: &[Node],
        observed: &[RegId],
        assigned: &mut Vec<Option<RegId>>,
        obtained: &mut Vec<RegId>,
        order: &mut Vec<usize>,
        nonvol: &[RegId],
    ) -> bool {
        if order.len() == nodes.len() {
            return true;
        }
        for i in 0..nodes.len() {
            if assigned[i].is_some() {
                continue;
            }
            let taken: Vec<RegId> = nodes[i].interferes.iter().filter_map(|&j| assigned[j]).collect();
            let mut free: Vec<RegId> = obtained.iter().copied().filter(|r| !taken.contains(r)).collect();
            free.sort();
            let (pick, new) = match free.first() {
                Some(&r) => (r, false),
                None => match nonvol.get(obtained.len()) {
                    Some(&r) => (r, true),
                    None => continue,
                },
            };
            if pick != observed[i] {
                continue;
            }
            assigned[i] = Some(pick);
            if new {
                obtained.push(pick);
            }
            order.push(i);
            if rec(nodes, observed, assigned, obtained, order, nonvol) {
                return true;
            }
            order.pop();
            if new {
                obtained.pop();
            }
            assigned[i] = None;
        }
        false
    }
    if nodes.is_empty() {
        return Some(vec![]);
    }
    let float = nodes[0].float;
    let nonvol: Vec<RegId> = if float { (46u8..=63).rev().collect() } else { (14u8..=31).rev().collect() };
    let mut assigned = vec![None; nodes.len()];
    let mut obtained = vec![];
    let mut order = vec![];
    if rec(nodes, observed, &mut assigned, &mut obtained, &mut order, &nonvol) {
        Some(order)
    } else {
        None
    }
}

/// Pairwise ordering constraints implied by an observed colouring: (a, b) means "a must be
/// coloured before b" in EVERY valid order (computed by trying to put b before a). Exponential in
/// the worst case; intended for the handful of callee-saved values of one function.
pub fn forced_pairs(nodes: &[Node], observed: &[RegId]) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let base = match solve_order(nodes, observed) {
        Some(o) => o,
        None => return out,
    };
    let pos: Vec<usize> = {
        let mut p = vec![0; nodes.len()];
        for (k, &i) in base.iter().enumerate() {
            p[i] = k;
        }
        p
    };
    for a in 0..nodes.len() {
        for b in 0..nodes.len() {
            if a == b || pos[a] > pos[b] {
                continue;
            }
            // is there a valid order with b before a? brute force via constrained search
            if !exists_order_with(nodes, observed, b, a) {
                out.push((a, b));
            }
        }
    }
    out
}

fn exists_order_with(nodes: &[Node], observed: &[RegId], first: usize, second: usize) -> bool {
    // same backtracking but forbid colouring `second` before `first`
    fn rec(
        nodes: &[Node],
        observed: &[RegId],
        assigned: &mut Vec<Option<RegId>>,
        obtained: &mut Vec<RegId>,
        count: usize,
        nonvol: &[RegId],
        first: usize,
        second: usize,
    ) -> bool {
        if count == nodes.len() {
            return true;
        }
        for i in 0..nodes.len() {
            if assigned[i].is_some() || (i == second && assigned[first].is_none()) {
                continue;
            }
            let taken: Vec<RegId> = nodes[i].interferes.iter().filter_map(|&j| assigned[j]).collect();
            let mut free: Vec<RegId> = obtained.iter().copied().filter(|r| !taken.contains(r)).collect();
            free.sort();
            let (pick, new) = match free.first() {
                Some(&r) => (r, false),
                None => match nonvol.get(obtained.len()) {
                    Some(&r) => (r, true),
                    None => continue,
                },
            };
            if pick != observed[i] {
                continue;
            }
            assigned[i] = Some(pick);
            if new {
                obtained.push(pick);
            }
            if rec(nodes, observed, assigned, obtained, count + 1, nonvol, first, second) {
                return true;
            }
            if new {
                obtained.pop();
            }
            assigned[i] = None;
        }
        false
    }
    let float = nodes.first().map_or(false, |n| n.float);
    let nonvol: Vec<RegId> = if float { (46u8..=63).rev().collect() } else { (14u8..=31).rev().collect() };
    let mut assigned = vec![None; nodes.len()];
    let mut obtained = vec![];
    rec(nodes, observed, &mut assigned, &mut obtained, 0, &nonvol, first, second)
}


// ------------------------------------------------------------------ class inference (emit hints)

/// What the emitter should do with one callee-saved value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InferredClass {
    Param(u32),
    /// unnamed / used only in non-move contexts; `rank` = creation order position among temps
    Temp { rank: u32 },
    /// named local with a move-use; `decl` = declaration order position among named locals
    Named { decl: u32 },
    /// coloured first because its degree reaches K (not steerable by declarations)
    Blocked,
}

#[derive(Clone, Debug)]
pub struct InferInput {
    pub observed: u8,
    /// Some(i) for parameter webs (i = parameter index in its register class order)
    pub param: Option<u32>,
    /// creation position (definition order in the code) for temp hypotheses
    pub def_pos: u32,
    /// prior: true if the code shape says "named" (r0 hop), false if it says "temp" (direct copy)
    pub prior_named: bool,
    /// estimated degree might reach K (blocked possible)
    pub maybe_blocked: bool,
    pub interferes: Vec<usize>,
    pub float: bool,
}

/// Infer temp/named classes and the declaration order of named locals that reproduce the observed
/// registers under the colouring model. Tries the prior first, then single and double flips of the
/// ambiguous values, then allows "blocked" for values flagged `maybe_blocked`.
pub fn infer_classes(vals: &[InferInput]) -> Option<Vec<InferredClass>> {
    let n = vals.len();
    let nodes: Vec<Node> = vals
        .iter()
        .map(|v| Node { class: Class::Temp(0), float: v.float, crosses_call: true, interferes: v.interferes.clone(), extra_degree: 0 })
        .collect();
    let observed: Vec<u8> = vals.iter().map(|v| v.observed).collect();
    let free: Vec<usize> = (0..n).filter(|&i| vals[i].param.is_none()).collect();
    let try_assign = |named: &[bool], blocked: &[bool]| -> Option<Vec<InferredClass>> {
        // precedence: blocked < temps (desc def_pos) < named (free order) < params (desc index)
        let mut before = vec![];
        let group = |i: usize| -> u8 {
            if blocked[i] {
                0
            } else if vals[i].param.is_some() {
                3
            } else if named[i] {
                2
            } else {
                1
            }
        };
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let (ga, gb) = (group(a), group(b));
                if ga < gb {
                    before.push((a, b));
                } else if ga == gb {
                    match ga {
                        1 if vals[a].def_pos > vals[b].def_pos => before.push((a, b)),
                        3 if vals[a].param > vals[b].param => before.push((a, b)),
                        _ => {}
                    }
                }
            }
        }
        let order = solve_order_constrained(&nodes, &observed, &before)?;
        let mut out = vec![InferredClass::Blocked; n];
        let (mut tr, mut nd) = (0u32, 0u32);
        // temps ranked by creation order (ascending def_pos), named by colouring order (= decl order)
        let mut temps: Vec<usize> = (0..n).filter(|&i| group(i) == 1).collect();
        temps.sort_by_key(|&i| vals[i].def_pos);
        for i in temps {
            out[i] = InferredClass::Temp { rank: tr };
            tr += 1;
        }
        for i in 0..n {
            if group(i) == 0 {
                if let Some(pi) = vals[i].param {
                    out[i] = InferredClass::Param(pi);
                }
            }
        }
        for &i in &order {
            match group(i) {
                2 => {
                    out[i] = InferredClass::Named { decl: nd };
                    nd += 1;
                }
                3 => out[i] = InferredClass::Param(vals[i].param.unwrap()),
                _ => {}
            }
        }
        Some(out)
    };
    let prior: Vec<bool> = (0..n).map(|i| vals[i].prior_named).collect();
    let no_block = vec![false; n];
    if let Some(r) = try_assign(&prior, &no_block) {
        return Some(r);
    }
    // single flips, then double flips
    for &i in &free {
        let mut c = prior.clone();
        c[i] = !c[i];
        if let Some(r) = try_assign(&c, &no_block) {
            return Some(r);
        }
    }
    for (ai, &i) in free.iter().enumerate() {
        for &j in &free[ai + 1..] {
            let mut c = prior.clone();
            c[i] = !c[i];
            c[j] = !c[j];
            if let Some(r) = try_assign(&c, &no_block) {
                return Some(r);
            }
        }
    }
    // allow blocked values (K-degree effect)
    let blocked: Vec<bool> = (0..n).map(|i| vals[i].maybe_blocked).collect();
    if blocked.iter().any(|&b| b) {
        if let Some(r) = try_assign(&prior, &blocked) {
            return Some(r);
        }
        for &i in &free {
            let mut c = prior.clone();
            c[i] = !c[i];
            if let Some(r) = try_assign(&c, &blocked) {
                return Some(r);
            }
        }
    }
    None
}
