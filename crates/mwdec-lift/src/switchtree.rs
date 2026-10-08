//! MWCC's switch compare trees (`Switch.c build_case_ranges` / `treecompare`), simulated so the
//! structurer can tell which case set a target tree came from. Case labels with an empty body
//! (`case 1: case 2: break;`) branch to the switch's end like the default does, so the target's
//! leaves alone can't show them, but they change the ranges and so the tree: the simulation
//! finds them.

use std::collections::HashMap;

/// A branch destination in the simulated tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lab {
    /// The switch's default label (the end of the switch when there is no `default:`).
    Default,
    /// A case body (identified by its block).
    Case(usize),
    /// An extra case label with an empty body (`case k: break;`), numbered.
    Extra(u32),
    /// A label inside the tree (`makepclabel`).
    Internal(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    Eq,
    Ge,
    Lt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Cmp(i64),
    Bc(Cond, Lab),
    B(Lab),
    Def(Lab),
}

#[derive(Clone, Copy, Debug)]
struct Range {
    min: i64,
    range: i64,
    label: Lab,
}

const MIN: i64 = i32::MIN as i64;
const MAX: i64 = i32::MAX as i64;

fn build_ranges(cases: &[(i64, Lab)]) -> Vec<Range> {
    let mut cs: Vec<(i64, Lab)> = cases.to_vec();
    cs.sort_by_key(|c| c.0);
    let mut r: Vec<Range> = vec![Range { min: MIN, range: MAX - MIN, label: Lab::Default }];
    for &(v, lab) in &cs {
        if !(MIN..=MAX).contains(&v) {
            continue;
        }
        let cur = r.len() - 1;
        if v > r[cur].min {
            r[cur].range = v - r[cur].min - 1;
            r.push(Range { min: v, range: 0, label: lab });
        } else if r[cur].min > MIN && cur > 0 && r[cur - 1].label == lab {
            r[cur - 1].range += 1;
            if r[cur].range == 0 {
                r.pop();
            } else {
                r[cur].min += 1;
                r[cur].range -= 1;
            }
            continue;
        }
        let cur = r.len() - 1;
        r[cur].range = 0;
        r[cur].label = lab;
        if v < MAX {
            r.push(Range { min: v + 1, range: MAX - (v + 1), label: Lab::Default });
        }
    }
    r
}

/// Jump table instead of a tree (`switchstatement`): N >= 8 && 2N >= span/2 + 4.
pub fn uses_table(cases: &[(i64, Lab)]) -> bool {
    let r = build_ranges(cases);
    let n = r.len() as i64 - 1;
    let vals: Vec<i64> = cases.iter().map(|c| c.0).collect();
    let (Some(lo), Some(hi)) = (vals.iter().min(), vals.iter().max()) else { return false };
    let span = hi - lo;
    n >= 8 && 2 * n >= span / 2 + 4
}

fn tree(r: &[Range], start: usize, end: usize, ops: &mut Vec<Op>, next_internal: &mut u32) {
    let count = end - start;
    let mut r29 = start + (count >> 1) + 1;
    if r[r29 - 1].range == 0 && ((count & 1) == 0 || (r[r29].range != 0 && count > 1)) {
        r29 -= 1;
    }
    let cur = r29;
    let r30 = r29 - 1;
    ops.push(Op::Cmp(r[cur].min));
    if r[cur].range == 0 && r29 < end {
        ops.push(Op::Bc(Cond::Eq, r[cur].label));
        r29 += 1;
    }
    if r29 == end {
        if start == r30 {
            if r[start].label == r[end].label {
                ops.push(Op::B(r[start].label));
            } else {
                ops.push(Op::Bc(Cond::Ge, r[end].label));
                ops.push(Op::B(r[start].label));
            }
        } else {
            ops.push(Op::Bc(Cond::Ge, r[end].label));
            tree(r, start, r30, ops, next_internal);
        }
    } else if start == r30 {
        ops.push(Op::Bc(Cond::Lt, r[start].label));
        tree(r, r29, end, ops, next_internal);
    } else {
        let l = Lab::Internal(*next_internal);
        *next_internal += 1;
        ops.push(Op::Bc(Cond::Ge, l));
        tree(r, start, r30, ops, next_internal);
        ops.push(Op::Def(l));
        tree(r, r29, end, ops, next_internal);
    }
}

/// The compare/branch sequence MWCC emits for a 4-byte selector, after the branch clean-ups that
/// labels sharing an address allow (`bge X; b X` -> `b X`, then the unused compare goes).
pub fn simulate(cases: &[(i64, Lab)], addr: &dyn Fn(Lab) -> Option<u32>) -> Vec<Op> {
    let r = build_ranges(cases);
    if r.len() < 2 {
        return vec![];
    }
    let mut ops = vec![];
    let mut n = 0;
    tree(&r, 0, r.len() - 1, &mut ops, &mut n);
    let same = |a: Lab, b: Lab| a == b || matches!((addr(a), addr(b)), (Some(x), Some(y)) if x == y);
    // one pass: a conditional branch to where the next unconditional branch goes disappears
    // (only the pair as emitted: `blt X; cmpwi; bge X; b X` keeps the `blt`)
    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        if let (Op::Bc(_, a), Some(Op::B(b))) = (op, ops.get(i + 1)) {
            if same(*a, *b) {
                continue;
            }
        }
        out.push(*op);
    }
    // then compares no branch reads
    let mut ops = vec![];
    for (i, op) in out.iter().enumerate() {
        if let Op::Cmp(_) = op {
            if !out[i + 1..].iter().take_while(|o| !matches!(o, Op::Cmp(_))).any(|o| matches!(o, Op::Bc(..))) {
                continue;
            }
        }
        ops.push(*op);
    }
    ops
}

/// One instruction of the target's tree region.
#[derive(Clone, Copy, Debug)]
pub enum TItem {
    Cmp(i64),
    Bc(Cond, u32),
    B(u32),
}

/// Does the simulated sequence reproduce the target items (`(offset, item)` in address order,
/// unrelated scheduled-in instructions already dropped)? Internal labels must sit right before
/// the next tree instruction.
pub fn matches(ops: &[Op], items: &[(u32, TItem)], addr: &dyn Fn(Lab) -> Option<u32>) -> bool {
    let mut map: HashMap<Lab, u32> = HashMap::new();
    let mut p = 0;
    let mut prev_off: Option<u32> = None;
    let bind = |l: Lab, a: u32, map: &mut HashMap<Lab, u32>| -> bool {
        if let Some(x) = addr(l) {
            return x == a;
        }
        match map.get(&l) {
            Some(&x) => x == a,
            None => {
                map.insert(l, a);
                true
            }
        }
    };
    for (oi, op) in ops.iter().enumerate() {
        match *op {
            Op::Def(l) => {
                let Some(&(off, _)) = items.get(p) else { return false };
                // the label sits after the previous tree instruction, at or before this one
                let lo = prev_off.map_or(0, |o| o + 4);
                match map.get(&l) {
                    Some(&x) => {
                        if x < lo || x > off {
                            return false;
                        }
                    }
                    None => {
                        map.insert(l, off);
                    }
                }
            }
            Op::Cmp(k) => {
                let Some(&(off, TItem::Cmp(v))) = items.get(p) else { return false };
                if v != k {
                    return false;
                }
                prev_off = Some(off);
                p += 1;
            }
            Op::Bc(c, l) => {
                let Some(&(off, TItem::Bc(tc, a))) = items.get(p) else { return false };
                if tc != c || !bind(l, a, &mut map) {
                    return false;
                }
                prev_off = Some(off);
                p += 1;
            }
            Op::B(l) => {
                let Some(&(off, TItem::B(a))) = items.get(p) else {
                    // the tree's last jump goes to the case body laid out right after it: no `b`
                    // (not in a one-compare tree: that is an `if`)
                    let last = ops[oi + 1..].iter().all(|o| matches!(o, Op::Def(_))) && ops.iter().filter(|o| matches!(o, Op::Cmp(_))).count() >= 2;
                    if let (true, Some(o)) = (last, prev_off) {
                        if bind(l, o + 4, &mut map) {
                            continue;
                        }
                    }
                    return false;
                };
                if !bind(l, a, &mut map) {
                    return false;
                }
                prev_off = Some(off);
                p += 1;
            }
        }
    }
    p > 0
}

