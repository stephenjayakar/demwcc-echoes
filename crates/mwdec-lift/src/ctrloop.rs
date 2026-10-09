//! Constant-count CTR loops and their unrolled bodies back to the source's `for` loop.
//!
//! MWCC turns `for (i = 0; i < K; i++)` with a call-free body into a CTR loop (`li r0,K ; mtctr ;
//! ... bdnz`), and the back end unrolls known-count loops by a divisor M of K
//! (`LoopOptimization.c`): the target holds M copies of the body per iteration and `ctr = K/M`.
//! Each copy reads the induction variables at its own offset (`lbz 0x3c(r5)`, `lbz 0x3d(r5)`, ...)
//! or steps them itself, and a search loop's copies each leave the loop (`li r0,0 ; b out`).
//!
//! The lifted loop is `c = K/M; do { COPY_0 ... COPY_{M-1}; v += M*S; c--; } while (c);` or,
//! with exits, `while (1) { ...; c--; if (!c) { DONE; break; } }`. We linearise one iteration
//! (exits become `if (t) { ...; break; }` items), express every value through the induction
//! variables at the iteration start (`v + 4`), split the items into M copies that are equal up to
//! a shift of the induction variables, and keep copy 0 in `for (i = 0; i < K; i++)`.

use crate::ir::*;
use std::collections::HashMap;

fn strip(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip(e),
        e => e,
    }
}

fn is_ctr(vars: &[Var], v: VarId) -> bool {
    vars[v].name.starts_with("var_ctr")
}

fn diverges(b: &[Stmt]) -> bool {
    matches!(b.iter().rev().find(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))), Some(Stmt::Return(_) | Stmt::Break | Stmt::Goto(_) | Stmt::Continue))
}

fn real(b: &[Stmt]) -> Vec<Stmt> {
    b.iter().filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))).cloned().collect()
}

/// One step of a linearised iteration.
#[derive(Clone, Debug, PartialEq)]
enum Item {
    S(Stmt),
    /// `if (c) { stmts }` leaving the loop (stmts end in break/return/goto)
    Exit(Expr, Vec<Stmt>),
}

/// `if (c) {A} else {B}; D` with D diverging: push D into the arms that fall out.
fn push_divergent_tail(b: &mut Vec<Stmt>) {
    let b2 = real(b);
    *b = b2;
    let n = b.len();
    if n < 2 || !matches!(b[n - 1], Stmt::Break | Stmt::Return(_)) {
        return;
    }
    if !matches!(b[n - 2], Stmt::If { .. }) {
        return;
    }
    let d = b.pop().unwrap();
    fn push(arm: &mut Vec<Stmt>, d: &Stmt) {
        let r = real(arm);
        *arm = r;
        if diverges(arm) {
            return;
        }
        if let Some(Stmt::If { then, els, .. }) = arm.last_mut() {
            if !then.is_empty() && !els.is_empty() {
                push(then, d);
                push(els, d);
                return;
            }
        }
        arm.push(d.clone());
    }
    if let Some(Stmt::If { then, els, .. }) = b.last_mut() {
        push(then, &d);
        if els.is_empty() {
            // `if (c) {A} D` -> the fall-through path also runs D
            b.push(d);
        } else {
            push(els, &d);
        }
    }
}

/// Linearise a loop iteration: early-exit ifs become `Exit` items, the rest of the iteration
/// continues with the arm that stays. `stay` marks a statement that ends the iteration normally
/// (`continue`, or the end of the list).
fn linearize(b: &[Stmt], out: &mut Vec<Item>) -> Option<()> {
    let mut b = b.to_vec();
    push_divergent_tail(&mut b);
    let mut k = 0;
    while k < b.len() {
        let s = b[k].clone();
        match s {
            Stmt::If { cond, then, els } => {
                let (t_div, e_div) = (diverges(&then), !els.is_empty() && diverges(&els));
                let rest = &b[k + 1..];
                let ends_continue = |x: &[Stmt]| matches!(real(x).last(), Some(Stmt::Continue));
                if t_div && !ends_continue(&then) {
                    out.push(Item::Exit(cond, real(&then)));
                    let mut cont = real(&els);
                    cont.extend_from_slice(rest);
                    return linearize(&cont, out);
                }
                if e_div && !ends_continue(&els) {
                    out.push(Item::Exit(cond.negated(), real(&els)));
                    let mut cont = real(&then);
                    cont.extend_from_slice(rest);
                    return linearize(&cont, out);
                }
                if t_div || e_div {
                    // a `continue` arm: the other arm must leave or the iteration ends here
                    return None;
                }
                out.push(Item::S(Stmt::If { cond, then, els }));
            }
            Stmt::Continue => {
                return if k + 1 == b.len() { Some(()) } else { None };
            }
            Stmt::Break | Stmt::Return(_) | Stmt::Goto(_) => return None,
            Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::Switch { .. } => return None,
            other => out.push(Item::S(other)),
        }
        k += 1;
    }
    Some(())
}

/// Canonical arithmetic: `(x + a) + b` -> `x + (a+b)`, constants folded into load offsets.
fn canon_expr(e: &Expr) -> Expr {
    let mut e = e.clone();
    e.rewrite(&mut |x| {
        // bottom-up rewrite: children are already canonical
        match x {
            Expr::Binary { op: op @ (BinOp::Add | BinOp::Sub), l, r, ty } => {
                if let Some(k) = r.as_int() {
                    let k = if *op == BinOp::Sub { -k } else { k };
                    if let Expr::Binary { op: BinOp::Add, l: l2, r: r2, .. } = &**l {
                        if let Some(k2) = r2.as_int() {
                            let s = k + k2;
                            *x = if s == 0 { (**l2).clone() } else { Expr::Binary { op: BinOp::Add, l: l2.clone(), r: Box::new(Expr::Int { value: s, ty: t_s32() }), ty: ty.clone() } };
                            return;
                        }
                    }
                    if k == 0 {
                        *x = (**l).clone();
                        return;
                    }
                    *x = Expr::Binary { op: BinOp::Add, l: l.clone(), r: Box::new(Expr::Int { value: k, ty: t_s32() }), ty: ty.clone() };
                }
            }
            // `&*(p + k)` is `p + k` (byte arithmetic)
            Expr::AddrOf(inner) if matches!(&**inner, Expr::Load { .. }) => {
                if let Expr::Load { base, offset, .. } = &**inner {
                    let b = (**base).clone();
                    *x = match &b {
                        Expr::Binary { op: BinOp::Add, l, r, ty } if r.as_int().is_some() => {
                            let s = r.as_int().unwrap() + *offset as i64;
                            if s == 0 { (**l).clone() } else { Expr::Binary { op: BinOp::Add, l: l.clone(), r: Box::new(Expr::Int { value: s, ty: t_s32() }), ty: ty.clone() } }
                        }
                        _ if *offset == 0 => b.clone(),
                        _ => Expr::Binary { op: BinOp::Add, l: Box::new(b.clone()), r: Box::new(Expr::Int { value: *offset as i64, ty: t_s32() }), ty: t_unk(4) },
                    };
                }
            }
            Expr::Load { base, offset, ty } => {
                if let Expr::Binary { op: BinOp::Add, l, r, .. } = &**base {
                    if let Some(k) = r.as_int() {
                        *x = Expr::Load { base: l.clone(), offset: *offset + k as i32, ty: ty.clone() };
                    }
                }
            }
            _ => {}
        }
    });
    e
}

/// `Var(v) + k` (or `Var(v)`): (v, k)
fn affine(e: &Expr) -> Option<(VarId, i64)> {
    match strip(e) {
        Expr::Var(v) => Some((*v, 0)),
        Expr::Binary { op: BinOp::Add, l, r, .. } => match (strip(l), r.as_int()) {
            (Expr::Var(v), Some(k)) => Some((*v, k)),
            _ => None,
        },
        _ => None,
    }
}

struct Canon<'a> {
    vars: &'a [Var],
    is_temp: &'a [bool],
    /// temp -> canonical value (affine in an induction variable, or a temp reference)
    env: HashMap<VarId, Expr>,
    /// induction variable -> offset at this point of the iteration
    ivs: HashMap<VarId, i64>,
    /// temps defined so far, by definition order
    ndefs: usize,
}

impl<'a> Canon<'a> {
    fn temp(&self, v: VarId) -> bool {
        self.is_temp.get(v).copied().unwrap_or(false)
    }

    /// Substitute known values (affine temps, IV offsets, temp references) and canonicalise.
    fn value(&self, e: &Expr) -> Expr {
        let mut e = e.clone();
        e.rewrite(&mut |x| {
            if let Expr::Var(v) = x {
                if let Some(val) = self.env.get(v) {
                    *x = val.clone();
                } else if let Some(&o) = self.ivs.get(v) {
                    if o != 0 {
                        *x = Expr::Binary { op: BinOp::Add, l: Box::new(Expr::Var(*v)), r: Box::new(Expr::int(o)), ty: self.vars[*v].ty.clone() };
                    }
                }
            }
        });
        canon_expr(&e)
    }

    fn temp_ref(&self, k: usize) -> Expr {
        Expr::Unknown { text: format!("@T{k}"), ty: t_unk(4) }
    }

    /// Canonical form of a statement list (arms of ifs, exit bodies); temps defined inside are
    /// numbered from `ndefs` on.
    fn stmts(&mut self, b: &[Stmt]) -> Option<Vec<Stmt>> {
        let mut out = vec![];
        for s in b {
            out.push(self.stmt(s)?);
        }
        Some(out)
    }

    fn stmt(&mut self, s: &Stmt) -> Option<Stmt> {
        Some(match s {
            Stmt::Assign { dst: Expr::Var(t), src } if self.temp(*t) => {
                let cs = self.value(src);
                if let Some((v, _)) = affine(&cs) {
                    if self.ivs.contains_key(&v) || !self.temp(v) {
                        self.env.insert(*t, cs);
                        return Some(Stmt::Comment(String::new()));
                    }
                }
                let k = self.ndefs;
                self.ndefs += 1;
                self.env.insert(*t, self.temp_ref(k));
                Stmt::Assign { dst: self.temp_ref(k), src: cs }
            }
            Stmt::Assign { dst, src } => Stmt::Assign { dst: self.value(dst), src: self.value(src) },
            Stmt::Expr(e) => Stmt::Expr(self.value(e)),
            Stmt::Return(e) => Stmt::Return(e.as_ref().map(|e| self.value(e))),
            Stmt::If { cond, then, els } => {
                let c = self.value(cond);
                Stmt::If { cond: c, then: self.stmts(then)?, els: self.stmts(els)? }
            }
            Stmt::Break | Stmt::Continue | Stmt::Goto(_) | Stmt::Comment(_) | Stmt::Label(_) => s.clone(),
            _ => return None,
        })
    }
}

/// Shift induction variables of a canonical item by `d[v]`.
fn shift_item(it: &Item, d: &HashMap<VarId, i64>, vars: &[Var]) -> Item {
    let f = |e: &Expr| -> Expr {
        let mut e = e.clone();
        e.rewrite(&mut |x| {
            if let Expr::Var(v) = x {
                if let Some(&o) = d.get(v) {
                    if o != 0 {
                        *x = Expr::Binary { op: BinOp::Add, l: Box::new(Expr::Var(*v)), r: Box::new(Expr::int(o)), ty: vars[*v].ty.clone() };
                    }
                }
            }
        });
        canon_expr(&e)
    };
    fn on_stmts(b: &[Stmt], f: &dyn Fn(&Expr) -> Expr) -> Vec<Stmt> {
        b.iter().map(|s| on_stmt(s, f)).collect()
    }
    fn on_stmt(s: &Stmt, f: &dyn Fn(&Expr) -> Expr) -> Stmt {
        match s {
            Stmt::Assign { dst, src } => Stmt::Assign { dst: f(dst), src: f(src) },
            Stmt::Expr(e) => Stmt::Expr(f(e)),
            Stmt::Return(e) => Stmt::Return(e.as_ref().map(|e| f(e))),
            Stmt::If { cond, then, els } => Stmt::If { cond: f(cond), then: on_stmts(then, f), els: on_stmts(els, f) },
            other => other.clone(),
        }
    }
    match it {
        Item::S(s) => Item::S(on_stmt(s, &f)),
        Item::Exit(c, b) => Item::Exit(f(c), on_stmts(b, &f)),
    }
}

/// Temp references are numbered by definition order over the whole iteration: renumber them
/// relative to the first definition of the copy so copies compare equal.
fn renumber(it: &Item, base: usize) -> Item {
    let f = |e: &Expr| -> Expr {
        let mut e = e.clone();
        e.rewrite(&mut |x| {
            if let Expr::Unknown { text, .. } = x {
                if let Some(k) = text.strip_prefix("@T").and_then(|k| k.parse::<usize>().ok()) {
                    let r = k as i64 - base as i64;
                    *text = format!("@R{r}");
                }
            }
        });
        e
    };
    fn on_stmts(b: &[Stmt], f: &dyn Fn(&Expr) -> Expr) -> Vec<Stmt> {
        b.iter().map(|s| on_stmt(s, f)).collect()
    }
    fn on_stmt(s: &Stmt, f: &dyn Fn(&Expr) -> Expr) -> Stmt {
        match s {
            Stmt::Assign { dst, src } => Stmt::Assign { dst: f(dst), src: f(src) },
            Stmt::Expr(e) => Stmt::Expr(f(e)),
            Stmt::Return(e) => Stmt::Return(e.as_ref().map(|e| f(e))),
            Stmt::If { cond, then, els } => Stmt::If { cond: f(cond), then: on_stmts(then, f), els: on_stmts(els, f) },
            other => other.clone(),
        }
    }
    match it {
        Item::S(s) => Item::S(on_stmt(s, &f)),
        Item::Exit(c, b) => Item::Exit(f(c), on_stmts(b, &f)),
    }
}

fn count_reads(body: &[Stmt], v: VarId) -> usize {
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    uses.get(&v).copied().unwrap_or(0)
}

/// What one loop statement looks like: (ctr var, iteration items, exit run when the count is
/// exhausted (None for do-while), the loop's own leading test turned into an exit).
struct Shape {
    ctr: VarId,
    items: Vec<Item>,
    done: Option<Vec<Stmt>>,
}

fn loop_shape(s: &Stmt, vars: &[Var], after: Option<&Stmt>) -> Option<Shape> {
    match s {
        Stmt::DoWhile { body, cond } => {
            let Expr::Binary { op: BinOp::Ne, l, r, .. } = strip(cond) else { return None };
            let Expr::Var(c) = strip(l) else { return None };
            if r.as_int() != Some(0) || !is_ctr(vars, *c) {
                return None;
            }
            let mut b = real(body);
            let last = b.pop()?;
            if !is_dec(&last, *c) {
                return None;
            }
            let mut items = vec![];
            linearize(&b, &mut items)?;
            Some(Shape { ctr: *c, items, done: None })
        }
        Stmt::While { cond, body } => {
            let mut items = vec![];
            let infinite = matches!(cond, Expr::Int { value: 1, .. });
            if !infinite {
                // the first copy's test became the loop condition: leaving through it runs the
                // statement after the loop (a return)
                let after = after?;
                if !matches!(after, Stmt::Return(_)) {
                    return None;
                }
                items.push(Item::Exit(cond.clone().negated(), vec![after.clone()]));
            }
            let mut b = real(body);
            push_divergent_tail(&mut b);
            // the latch: `c--; if (c == 0) { DONE } [else continue]` at the end of the path
            // that continues; find it at the end of the linearised items
            let mut lin = vec![];
            linearize(&b, &mut lin)?;
            // [..., c = c - 1, Exit(c == 0, DONE)]
            let n = lin.len();
            if n < 2 {
                return None;
            }
            let Item::Exit(tc, done) = lin[n - 1].clone() else { return None };
            let Item::S(dec) = lin[n - 2].clone() else { return None };
            let c = match strip(&tc) {
                Expr::Binary { op: BinOp::Eq, l, r, .. } if r.as_int() == Some(0) => match strip(l) {
                    Expr::Var(c) => *c,
                    _ => return None,
                },
                _ => return None,
            };
            if !is_ctr(vars, c) || !is_dec(&dec, c) {
                return None;
            }
            lin.truncate(n - 2);
            items.extend(lin);
            Some(Shape { ctr: c, items, done: Some(done) })
        }
        _ => None,
    }
}

fn is_dec(s: &Stmt, v: VarId) -> bool {
    match s {
        Stmt::Assign { dst: Expr::Var(x), src } if *x == v => match strip(src) {
            Expr::Binary { op: BinOp::Sub, l, r, .. } => matches!(strip(l), Expr::Var(y) if *y == v) && r.as_int() == Some(1),
            Expr::Binary { op: BinOp::Add, l, r, .. } => matches!(strip(l), Expr::Var(y) if *y == v) && r.as_int() == Some(-1),
            _ => false,
        },
        _ => false,
    }
}

/// Rebuild one copy's items as statements (affine temps substituted, IV updates appended).
fn rebuild(items: &[Item], subst: &HashMap<VarId, Expr>, vars: &[Var]) -> Vec<Stmt> {
    let f = |e: &Expr| -> Expr {
        let mut e = e.clone();
        e.rewrite(&mut |x| {
            if let Expr::Var(v) = x {
                if let Some(val) = subst.get(v) {
                    *x = val.clone();
                }
            }
        });
        // pointer IVs at an offset are written the lifter's way: `*(p + k)` as a load at
        // offset k, `p + k` as `&p->field`
        e.rewrite(&mut |x| match x {
            Expr::Load { base, offset, ty } => {
                if let Expr::Binary { op: BinOp::Add, l, r, .. } = &**base {
                    if let (Expr::Var(v), Some(k)) = (&**l, r.as_int()) {
                        if is_ptr(&vars[*v].ty) {
                            *x = Expr::Load { base: Box::new(Expr::Var(*v)), offset: *offset + k as i32, ty: ty.clone() };
                        }
                    }
                }
            }
            Expr::Binary { op: BinOp::Add, l, r, .. } => {
                if let (Expr::Var(v), Some(k)) = (&**l, r.as_int()) {
                    if is_ptr(&vars[*v].ty) {
                        *x = ptr_add(*v, k);
                    }
                }
            }
            _ => {}
        });
        e
    };
    let drop = |v: VarId| subst.contains_key(&v);
    fn on_stmts(b: &[Stmt], f: &dyn Fn(&Expr) -> Expr, drop: &dyn Fn(VarId) -> bool) -> Vec<Stmt> {
        b.iter().filter_map(|s| on_stmt(s, f, drop)).collect()
    }
    fn on_stmt(s: &Stmt, f: &dyn Fn(&Expr) -> Expr, drop: &dyn Fn(VarId) -> bool) -> Option<Stmt> {
        Some(match s {
            // the affine temps' own definitions disappear (their uses read the IV directly)
            Stmt::Assign { dst: Expr::Var(t), .. } if drop(*t) => return None,
            Stmt::Assign { dst, src } => Stmt::Assign { dst: f(dst), src: f(src) },
            Stmt::Expr(e) => Stmt::Expr(f(e)),
            Stmt::Return(e) => Stmt::Return(e.as_ref().map(|e| f(e))),
            Stmt::If { cond, then, els } => Stmt::If { cond: f(cond), then: on_stmts(then, f, drop), els: on_stmts(els, f, drop) },
            Stmt::Comment(c) if c.is_empty() => return None,
            other => other.clone(),
        })
    }
    let mut out = vec![];
    for it in items {
        match it {
            Item::S(s) => out.extend(on_stmt(s, &f, &drop)),
            Item::Exit(c, b) => out.push(Stmt::If { cond: f(c), then: on_stmts(b, &f, &drop), els: vec![] }),
        }
    }
    out
}

/// `p + k` bytes for a pointer variable, as the lifter writes it (`&p->field`).
fn ptr_add(v: VarId, k: i64) -> Expr {
    Expr::AddrOf(Box::new(Expr::Load { base: Box::new(Expr::Var(v)), offset: k as i32, ty: t_unk(0) }))
}

static NEXT_LABEL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(900_000);

/// Reroll constant-count CTR loops (unrolled or not) into `for (i = 0; i < K; i++)`.
pub fn recover(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
    let mut new_vars: Vec<Var> = vec![];
    let nvars = vars.len();
    let snapshot_body = body.clone();
    {
        let vars_ro: &[Var] = vars;
        let is_temp_ro: &[bool] = is_temp;
        Stmt::for_each_block_mut(body, &mut |b| {
            let mut j = 0;
            while j < b.len() {
                if let Some(n) = try_reroll(b, j, vars_ro, is_temp_ro, &snapshot_body, nvars + new_vars.len()) {
                    if let Some(v) = n {
                        new_vars.push(v);
                    }
                }
                j += 1;
            }
        });
    }
    for v in new_vars {
        vars.push(v);
        is_temp.push(false);
    }
}

/// `continue`/`break` of this loop level (not of nested loops; `break` in a switch is the
/// switch's own).
fn has_own_jump(b: &[Stmt], in_switch: bool) -> bool {
    b.iter().any(|s| match s {
        Stmt::Continue => true,
        Stmt::Break => !in_switch,
        Stmt::If { then, els, .. } => has_own_jump(then, in_switch) || has_own_jump(els, in_switch),
        Stmt::Switch { cases, .. } => cases.iter().any(|c| has_own_jump(&c.body, true)),
        _ => false,
    })
}

/// Does `part` contain a label that some `goto` of `whole` jumps to?
fn has_targeted_label(part: &[Stmt], whole: &[Stmt]) -> bool {
    fn labels(b: &[Stmt], gotos: bool, out: &mut Vec<LabelId>) {
        for s in b {
            match s {
                Stmt::Label(l) if !gotos => out.push(*l),
                Stmt::Goto(l) if gotos => out.push(*l),
                Stmt::If { then, els, .. } => {
                    labels(then, gotos, out);
                    labels(els, gotos, out);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => labels(body, gotos, out),
                Stmt::For { init, step, body, .. } => {
                    labels(init, gotos, out);
                    labels(step, gotos, out);
                    labels(body, gotos, out);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        labels(&c.body, gotos, out);
                    }
                }
                _ => {}
            }
        }
    }
    let mut ls = vec![];
    labels(part, false, &mut ls);
    if ls.is_empty() {
        return false;
    }
    let mut gs = vec![];
    labels(whole, true, &mut gs);
    ls.iter().any(|l| gs.contains(l))
}

/// Returns Some(new var to add (if any)) when the loop at b[j] was rerolled.
fn try_reroll(b: &mut Vec<Stmt>, j: usize, vars: &[Var], is_temp: &[bool], whole: &[Stmt], next_var: VarId) -> Option<Option<Var>> {
    // a label inside the loop that a goto targets would be lost with the copies dropped
    if has_targeted_label(std::slice::from_ref(&b[j]), whole) {
        return None;
    }
    let shape = loop_shape(&b[j], vars, b.get(j + 1))?;
    let c = shape.ctr;
    // the counter: `c = K` earlier in this list, read only by the loop's decrement and test
    let ip = (0..j).rev().find(|&k| crate::idioms::stmt_mentions(&b[k], c))?;
    let Stmt::Assign { dst: Expr::Var(x), src } = &b[ip] else { return None };
    if *x != c {
        return None;
    }
    let k_iters = strip(src).as_int()?;
    if !(1..=4096).contains(&k_iters) || count_reads(whole, c) != 2 {
        return None;
    }
    // canonical iteration: induction variables are the non-temp vars updated `v = v + k`
    let mut cn = Canon { vars, is_temp, env: HashMap::new(), ivs: HashMap::new(), ndefs: 0 };
    // pre-scan: IV candidates are non-temp locals assigned in the iteration
    let mut canon_items: Vec<Item> = vec![];
    let mut orig_items: Vec<Item> = vec![];
    let mut defs_before: Vec<usize> = vec![];
    let mut updated: Vec<VarId> = vec![];
    let item_mentions = |it: &Item, v: VarId| match it {
        Item::S(s) => crate::idioms::stmt_mentions(s, v),
        Item::Exit(c, b) => c.uses_var(v) || b.iter().any(|s| crate::idioms::stmt_mentions(s, v)),
    };
    for it in &shape.items {
        // an induction variable is not read after its update (MWCC updates at the end of
        // the iteration; intermediate values live in temps)
        if updated.iter().any(|&v| item_mentions(it, v)) {
            let is_update = matches!(it, Item::S(Stmt::Assign { dst: Expr::Var(v), .. }) if updated.contains(v));
            if !is_update {
                return None;
            }
        }
        let before = cn.ndefs;
        match it {
            Item::S(Stmt::Assign { dst: Expr::Var(v), src }) if !cn.temp(*v) && !matches!(vars[*v].kind, VarKind::Stack { .. }) => {
                let cs = cn.value(src);
                if let Some((w, k)) = affine(&cs) {
                    if w == *v {
                        cn.ivs.insert(*v, k);
                        if !updated.contains(v) {
                            updated.push(*v);
                        }
                        continue;
                    }
                }
                let s = cn.stmt(&Stmt::Assign { dst: Expr::Var(*v), src: src.clone() })?;
                canon_items.push(Item::S(s));
            }
            Item::S(s) => {
                let cs = cn.stmt(s)?;
                canon_items.push(Item::S(cs));
            }
            Item::Exit(cond, stmts) => {
                let cc = cn.value(cond);
                let save = cn.ndefs;
                let cb = cn.stmts(stmts)?;
                cn.ndefs = save.max(cn.ndefs);
                canon_items.push(Item::Exit(cc, cb));
            }
        }
        orig_items.push(it.clone());
        defs_before.push(before);
    }
    // affine temps over a variable the iteration assigns other than by an IV update can't be
    // rewritten through it
    let deep_assigns = |v: VarId| {
        fn walk(b: &[Stmt], v: VarId) -> bool {
            b.iter().any(|s| match s {
                Stmt::Assign { dst: Expr::Var(x), .. } => *x == v,
                Stmt::If { then, els, .. } => walk(then, v) || walk(els, v),
                _ => false,
            })
        }
        orig_items.iter().any(|it| match it {
            Item::S(s) => walk(std::slice::from_ref(s), v),
            Item::Exit(_, b) => walk(b, v),
        })
    };
    for e in cn.env.values() {
        if let Some((v, _)) = affine(e) {
            if !cn.ivs.contains_key(&v) && deep_assigns(v) {
                return None;
            }
        }
    }
    // an IV read before its update in a way we can't follow (not affine) isn't an IV: the
    // updates must come after all other items (MWCC places them at the end of each copy, the
    // last one at the end of the iteration)
    let mut total: HashMap<VarId, i64> = cn.ivs.clone();
    // a dead counter (read only by its own update: MWCC's leftover of the source index, e.g.
    // `addi r8,r8,3` per unrolled iteration) is the source's index
    let dead_iv: Option<VarId> = {
        let mut d: Vec<VarId> = total.keys().copied().filter(|&v| count_reads(whole, v) == 1).collect();
        d.sort();
        d.first().copied()
    };
    if let Some(v) = dead_iv {
        total.remove(&v);
    }
    // drop the affine-temp placeholders for the copy split (keep them in orig for rebuild)
    let idx: Vec<usize> = (0..canon_items.len()).filter(|&i| !matches!(&canon_items[i], Item::S(Stmt::Comment(t)) if t.is_empty())).collect();
    let n = idx.len();
    if n == 0 {
        return None;
    }
    // try the largest number of copies first
    let mut chosen: Option<(usize, HashMap<VarId, i64>)> = None;
    for m in (1..=n.min(16)).rev() {
        if n % m != 0 {
            continue;
        }
        let p = n / m;
        let mut stride: HashMap<VarId, i64> = HashMap::new();
        let mut ok = true;
        for (&v, &t) in &total {
            if t % m as i64 != 0 {
                ok = false;
                break;
            }
            stride.insert(v, t / m as i64);
        }
        if !ok {
            continue;
        }
        if m > 1 {
            'copies: for jj in 1..m {
                let d: HashMap<VarId, i64> = stride.iter().map(|(&v, &s)| (v, s * jj as i64)).collect();
                for q in 0..p {
                    let a = &canon_items[idx[q]];
                    let bq = &canon_items[idx[jj * p + q]];
                    let a2 = renumber(&shift_item(a, &d, vars), defs_before[idx[0]]);
                    let b2 = renumber(bq, defs_before[idx[jj * p]]);
                    if a2 != b2 {
                        if std::env::var_os("MWDEC_DBG").is_some() {
                            eprintln!("ctrloop m={m} copy {jj} item {q}:
  {:?}
  {:?}", a2, b2);
                        }
                        ok = false;
                        break 'copies;
                    }
                }
            }
        }
        if ok {
            chosen = Some((m, stride));
            break;
        }
    }
    let (m, stride) = chosen?;
    let p = n / m;
    // copy 0: the original items up to the first item of copy 1 (affine temps included)
    let end0 = if m > 1 { idx[p] } else { orig_items.len() };
    // affine temps of copy 0 are written through the IVs (`temp = v + 4` -> `v + 4`)
    let subst: HashMap<VarId, Expr> = cn
        .env
        .iter()
        .filter(|(_, e)| affine(e).map_or(false, |(v, _)| total.contains_key(&v) || !is_temp.get(v).copied().unwrap_or(false)))
        .map(|(&t, e)| (t, e.clone()))
        .collect();
    // copy 0's affine temps are relative to the iteration start; later copies' temps don't
    // occur in copy 0
    let mut body0 = rebuild(&orig_items[..end0], &subst, vars);
    let count = k_iters * m as i64;
    // the index: an IV starting at 0 with stride 1, else a fresh `i`
    let init_of = |v: VarId, b: &Vec<Stmt>| -> Option<usize> {
        let k = (0..j).rev().find(|&k| crate::idioms::stmt_mentions(&b[k], v))?;
        match &b[k] {
            Stmt::Assign { dst: Expr::Var(x), src } if *x == v && strip(src).as_int() == Some(0) => Some(k),
            _ => None,
        }
    };
    let mut index: Option<(VarId, usize)> = None;
    if let Some(v) = dead_iv {
        if matches!(vars[v].kind, VarKind::Local) {
            if let Some(k) = init_of(v, b) {
                if k != ip {
                    index = Some((v, k));
                }
            }
        }
    }
    for (&v, &s) in &stride {
        if index.is_some() {
            break;
        }
        if s == 1 && matches!(vars[v].kind, VarKind::Local) {
            if let Some(k) = init_of(v, b) {
                if k != ip {
                    index = Some((v, k));
                    break;
                }
            }
        }
    }
    let mut ivs: Vec<(VarId, i64)> = stride.iter().map(|(&v, &s)| (v, s)).filter(|&(v, s)| s != 0 && index.map_or(true, |(i, _)| i != v)).collect();
    ivs.sort();
    for (v, s) in &ivs {
        let src = if is_ptr(&vars[*v].ty) { ptr_add(*v, *s) } else { Expr::bin(BinOp::Add, Expr::Var(*v), Expr::int(*s), vars[*v].ty.clone()) };
        body0.push(Stmt::Assign { dst: Expr::Var(*v), src });
    }
    let (ivar, new_var) = match index {
        Some((v, _)) => (v, None),
        None => (next_var, Some(Var { name: "i".into(), ty: t_s32(), kind: VarKind::Local })),
    };
    let ity = match &new_var {
        Some(v) => v.ty.clone(),
        None => vars[ivar].ty.clone(),
    };
    // after the loop: what ran when the count ran out
    let mut after: Vec<Stmt> = vec![];
    let mut label: Option<usize> = None;
    if let Some(done) = shape.done {
        let mut d = done.clone();
        let ends_break = matches!(d.last(), Some(Stmt::Break));
        if ends_break {
            d.pop();
        }
        // early exits that break out skip DONE: they jump past it
        let breaks_out = body0.iter().any(|s| matches!(s, Stmt::If { then, .. } if matches!(then.last(), Some(Stmt::Break))));
        if ends_break && breaks_out && !d.is_empty() {
            let l = NEXT_LABEL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for s in body0.iter_mut() {
                if let Stmt::If { then, .. } = s {
                    if matches!(then.last(), Some(Stmt::Break)) {
                        then.pop();
                        then.push(Stmt::Goto(l));
                    }
                }
            }
            label = Some(l);
        }
        if !ends_break && !matches!(d.last(), Some(Stmt::Return(_) | Stmt::Goto(_))) {
            return None;
        }
        after = d;
    }
    let for_stmt = Stmt::For {
        init: vec![Stmt::Assign { dst: Expr::Var(ivar), src: Expr::Int { value: 0, ty: ity.clone() } }],
        cond: Expr::cmp(BinOp::Lt, Expr::Var(ivar), Expr::Int { value: count, ty: ity.clone() }),
        step: vec![Stmt::Expr(Expr::IncDec { e: Box::new(Expr::Var(ivar)), delta: 1, post: true })],
        body: body0,
    };
    // a While whose condition was the first exit: the statement after it is now unreachable
    let drop_after = matches!(&b[j], Stmt::While { cond, .. } if !matches!(cond, Expr::Int { value: 1, .. }));
    b[j] = for_stmt;
    let mut at = j + 1;
    if drop_after && at < b.len() && !after.is_empty() {
        b.remove(at);
    }
    for s in after {
        b.insert(at, s);
        at += 1;
    }
    if let Some(l) = label {
        b.insert(at, Stmt::Label(l));
    }
    // drop `c = K` and the index's `= 0`
    let mut rm = vec![ip];
    if let Some((_, k)) = index {
        rm.push(k);
    }
    rm.sort_unstable();
    for r in rm.iter().rev() {
        b.remove(*r);
    }
    Some(new_var)
}

/// `x = K; t = x;` (a temp copying a local that holds a constant, MWCC's CSE of the constant:
/// `li r7,0 ; mr r4,r7`) -> `t = K`, so the temp doesn't keep reading the local.
pub fn forward_constant_copies(body: &mut Vec<Stmt>, is_temp: &[bool]) {
    Stmt::for_each_block_mut(body, &mut |b| {
        for i in 0..b.len() {
            let x = match &b[i] {
                Stmt::Assign { dst: Expr::Var(t), src: Expr::Var(x) } if is_temp.get(*t).copied().unwrap_or(false) && !is_temp.get(*x).copied().unwrap_or(false) => *x,
                _ => continue,
            };
            let Some(k) = (0..i).rev().find(|&k| crate::idioms::stmt_mentions(&b[k], x)) else { continue };
            let val = match &b[k] {
                Stmt::Assign { dst: Expr::Var(y), src: v @ Expr::Int { .. } } if *y == x => v.clone(),
                _ => continue,
            };
            if let Stmt::Assign { src, .. } = &mut b[i] {
                *src = val;
            }
        }
    });
}

/// Temps holding one integer constant everywhere (`t = 0;` read inside a loop's if) are the
/// constant itself: MWCC materialises and hoists constants on its own.
pub fn propagate_constant_temps(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &[bool]) {
    // a constant kept in a callee-saved register across calls was a variable in the source
    let volatile_home = |v: VarId| {
        let n = &vars[v].name;
        n.strip_prefix("temp_r").or_else(|| n.strip_prefix("temp_f")).and_then(|r| r.split('_').next()).and_then(|d| d.parse::<u8>().ok()).map_or(false, |d| d < 14)
    };
    let mut defs: HashMap<VarId, (usize, Option<Expr>)> = HashMap::new();
    fn scan(b: &[Stmt], is_temp: &[bool], defs: &mut HashMap<VarId, (usize, Option<Expr>)>) {
        for s in b {
            match s {
                Stmt::Assign { dst: Expr::Var(t), src } if is_temp.get(*t).copied().unwrap_or(false) => {
                    let e = defs.entry(*t).or_insert((0, None));
                    e.0 += 1;
                    e.1 = matches!(src, Expr::Int { .. }).then(|| src.clone());
                }
                Stmt::If { then, els, .. } => {
                    scan(then, is_temp, defs);
                    scan(els, is_temp, defs);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => scan(body, is_temp, defs),
                Stmt::For { init, step, body, .. } => {
                    scan(init, is_temp, defs);
                    scan(step, is_temp, defs);
                    scan(body, is_temp, defs);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        scan(&c.body, is_temp, defs);
                    }
                }
                _ => {}
            }
        }
    }
    scan(body, is_temp, &mut defs);
    let consts: HashMap<VarId, Expr> = defs.into_iter().filter_map(|(t, (n, e))| if n == 1 { e.map(|e| (t, e)) } else { None }).collect();
    if consts.is_empty() {
        return;
    }
    // IncDec/address-taken temps keep their variable
    let mut bad: Vec<VarId> = vec![];
    Stmt::walk_exprs(body, &mut |e| match e {
        Expr::IncDec { e, .. } | Expr::AddrOf(e) => {
            if let Expr::Var(v) = &**e {
                bad.push(*v);
            }
        }
        _ => {}
    });
    let consts: HashMap<VarId, Expr> = consts.into_iter().filter(|(t, _)| !bad.contains(t) && volatile_home(*t)).collect();
    Stmt::for_each_block_mut(body, &mut |b| {
        b.retain(|s| !matches!(s, Stmt::Assign { dst: Expr::Var(t), .. } if consts.contains_key(t)));
    });
    Stmt::rewrite_exprs(body, &mut |x| {
        if let Expr::Var(v) = x {
            if let Some(c) = consts.get(v) {
                *x = c.clone();
            }
        }
    });
}

/// `t = G; if (c(t)) { *(t + k) = x; }` -> the global read again in each use: the source read
/// `G` twice and MWCC CSE'd the second read (nothing in between may store to it). Naming it
/// gives the value its own variable (different registers).
pub fn rematerialize_global_temps(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &[bool]) {
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            // a global, or a member read through a variable (`t = p->cb; if (t) t(0);`)
            let simple_read = |g: &Expr| match g {
                Expr::Global { .. } => true,
                // (not stack objects: their single read lets the object fold away)
                Expr::Load { base, .. } | Expr::Member { base, .. } => matches!(&**base, Expr::Var(v) if !matches!(vars[*v].kind, VarKind::Stack { .. })),
                _ => false,
            };
            let (t, g) = match &b[i] {
                Stmt::Assign { dst: Expr::Var(t), src: g } if is_temp.get(*t).copied().unwrap_or(false) && simple_read(g) => (*t, g.clone()),
                _ => {
                    i += 1;
                    continue;
                }
            };
            let base_var = match &g {
                Expr::Load { base, .. } | Expr::Member { base, .. } => match &**base {
                    Expr::Var(v) => Some(*v),
                    _ => None,
                },
                _ => None,
            };
            let total = uses.get(&t).copied().unwrap_or(0);
            let ok = match &b[i + 1] {
                Stmt::If { cond, then, els } => {
                    let mut n = 0;
                    cond.walk(&mut |e| if matches!(e, Expr::Var(x) if *x == t) { n += 1 });
                    let in_cond = n;
                    // uses in the first statement of each arm, which is a plain store or
                    // assignment without calls
                    let mut arm_ok = true;
                    for arm in [then, els] {
                        for (k, s) in arm.iter().enumerate() {
                            let m = crate::idioms::stmt_mentions(s, t);
                            if !m {
                                continue;
                            }
                            // a call whose callee or plain arguments read it (evaluated before
                            // the call)
                            let call_ok = |c: &Expr| match c {
                                Expr::Call { callee, args, .. } => {
                                    let callee_ok = match callee {
                                        Callee::Indirect(f) => !f.has_call(),
                                        Callee::Direct { .. } => true,
                                        _ => false,
                                    };
                                    callee_ok && args.iter().all(|a| !a.has_call())
                                }
                                _ => false,
                            };
                            let assigns_base = base_var.map_or(false, |v| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if *x == v));
                            let plain = !assigns_base
                                && match s {
                                    Stmt::Assign { src, .. } => !src.has_call() || call_ok(src),
                                    Stmt::Expr(c) => call_ok(c),
                                    _ => false,
                                };
                            if k != 0 || !plain {
                                arm_ok = false;
                            }
                            Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| if matches!(e, Expr::Var(x) if *x == t) { n += 1 });
                        }
                    }
                    // read in the test and again in an arm (a value only tested keeps its one
                    // read: re-reading it there would let MWCC CSE a later read too)
                    arm_ok && n == total && in_cond >= 1 && n > in_cond && !cond.has_call()
                }
                _ => false,
            };
            if !ok {
                i += 1;
                continue;
            }
            b.remove(i);
            Stmt::rewrite_exprs(&mut b[i..i + 1], &mut |x| {
                if matches!(x, Expr::Var(v) if *v == t) {
                    *x = g.clone();
                }
            });
            i += 1;
        }
    });
}

/// MWCC's back-end unrolling of a loop with an unknown count n ("shape B",
/// `LoopOptimization.c unrollunknownBDNZ`):
///
/// ```text
/// t = n >> s; ctr = t;
/// if (t != 0) { do { BODY x F; ctr--; } while (ctr); n = n & (F-1); if (n != 0) goto R; }
/// else { R: ctr2 = n; do { BODY; ctr2--; } while (ctr2); }
/// ```
///
/// The source had one loop over n (`while (n--)`, `for (; p < e; p++)`): keep the remainder
/// loop's body. A guard `if (n != 0)` / `if (p < e)` around it is the compiler's own test.
pub fn recover_shape_b(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &[bool]) {
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut k = 0;
        while k < b.len() {
            if let Some((start, repl)) = shape_b_at(b, k, vars, is_temp) {
                b.splice(start..=k, repl);
                k = start + 1;
                continue;
            }
            k += 1;
        }
    });
    // a guard that only repeats the loop's test: `if (n != 0) { while (n != 0) ... }`
    Stmt::for_each_block_mut(body, &mut |b| {
        for i in 0..b.len() {
            let prev = if i > 0 { Some(b[i - 1].clone()) } else { None };
            let Stmt::If { cond, then, els } = &mut b[i] else { continue };
            if !els.is_empty() || then.len() != 1 {
                continue;
            }
            let Stmt::While { cond: wc, .. } = &then[0] else { continue };
            let (g, w) = (strip_ne0(cond), strip_ne0(wc));
            // `n = t; if (t != 0) while (n != 0)`
            let copied = matches!((&prev, g, w), (Some(Stmt::Assign { dst: Expr::Var(n), src }), Expr::Var(t), Expr::Var(n2)) if n == n2 && matches!(strip(src), Expr::Var(t2) if t2 == t));
            if g == w || copied {
                let w = then.remove(0);
                b[i] = w;
            }
        }
    });
}

/// `X + 1 - A` (inclusive) or `X - A` (exclusive) trip counts: (A, X, inclusive).
fn trip_bounds(src: &Expr) -> Option<(Expr, Expr, bool)> {
    match strip(src) {
        Expr::Binary { op: BinOp::Sub, l, r, .. } => match strip(l) {
            Expr::Binary { op: BinOp::Add, l: x, r: one, .. } if one.as_int() == Some(1) => Some(((**r).clone(), (**x).clone(), true)),
            _ => Some(((**r).clone(), (**l).clone(), false)),
        },
        Expr::Binary { op: BinOp::Add, l, r: one, .. } if one.as_int() == Some(1) => match strip(l) {
            Expr::Binary { op: BinOp::Sub, l: x, r: a, .. } => Some(((**a).clone(), (**x).clone(), true)),
            _ => None,
        },
        _ => None,
    }
}

fn assigns_any(b: &[Stmt], vs: &[VarId]) -> bool {
    let mut found = false;
    let mut v = b.to_vec();
    Stmt::for_each_block_mut(&mut v, &mut |blk| {
        for s in blk.iter() {
            if let Stmt::Assign { dst: Expr::Var(x), .. } = s {
                if vs.contains(x) {
                    found = true;
                }
            }
        }
    });
    let mut inc = false;
    for s in b {
        if let Stmt::Expr(e) | Stmt::Assign { src: e, .. } = s {
            e.walk(&mut |x| {
                if let Expr::IncDec { e, .. } = x {
                    if matches!(strip(e), Expr::Var(y) if vs.contains(y)) {
                        inc = true;
                    }
                }
            });
        }
    }
    found || inc
}

/// The unknown-count loop recovered from shape B under its own range guard:
/// `n = X + 1 - A; if (A <= X) { while (n != 0) { B; n--; } }` is
/// `for (i = A; i <= X; i++) { B }` (the compiler's count `X - A` for `<`). The guard is the
/// for's first test; a `while (n != 0)` would add a test of its own.
pub fn counted_for(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
    let snapshot = body.clone();
    let mut new_vars: Vec<Var> = vec![];
    let nvars = vars.len();
    {
        let vars_ro: &[Var] = vars;
        Stmt::for_each_block_mut(body, &mut |b| {
            let mut i = 0;
            while i + 1 < b.len() {
                let found = (|| {
                    let Stmt::Assign { dst: Expr::Var(n), src } = &b[i] else { return None };
                    let n = *n;
                    if !matches!(vars_ro[n].kind, VarKind::Local) || count_reads(&snapshot, n) != 2 {
                        return None;
                    }
                    let (a, x, incl) = trip_bounds(src)?;
                    let Stmt::If { cond, then, els } = &b[i + 1] else { return None };
                    if !els.is_empty() || then.len() != 1 {
                        return None;
                    }
                    let Expr::Binary { op, l, r, .. } = strip(cond) else { return None };
                    let (sa, sx) = (strip(&a), strip(&x));
                    let ok = match op {
                        BinOp::Le => incl && strip(l) == sa && strip(r) == sx,
                        BinOp::Ge => incl && strip(l) == sx && strip(r) == sa,
                        BinOp::Lt => !incl && strip(l) == sa && strip(r) == sx,
                        BinOp::Gt => !incl && strip(l) == sx && strip(r) == sa,
                        _ => false,
                    };
                    if !ok {
                        return None;
                    }
                    let Stmt::While { cond: wc, body: wb } = &then[0] else { return None };
                    if !matches!(strip_ne0(wc), Expr::Var(y) if *y == n) {
                        return None;
                    }
                    let last = wb.iter().rposition(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_)))?;
                    if !is_dec(&wb[last], n) {
                        return None;
                    }
                    let mut lb = wb.clone();
                    lb.remove(last);
                    if lb.iter().any(|s| crate::idioms::stmt_mentions(s, n)) || has_own_jump(&lb, false) {
                        return None;
                    }
                    // the bounds are read once, before the loop: nothing in it may change them
                    let mut bvars = vec![];
                    for e in [&a, &x] {
                        let mut pure = true;
                        e.walk(&mut |y| match y {
                            Expr::Var(v) => bvars.push(*v),
                            Expr::Int { .. } | Expr::Cast { .. } | Expr::Binary { .. } | Expr::Unary { .. } => {}
                            _ => pure = false,
                        });
                        if !pure {
                            return None;
                        }
                    }
                    if assigns_any(&lb, &bvars) {
                        return None;
                    }
                    Some((a, x, incl, lb))
                })();
                if let Some((a, x, incl, lb)) = found {
                    let iv = nvars + new_vars.len();
                    let ty = t_s32();
                    new_vars.push(Var { name: "i".into(), ty: ty.clone(), kind: VarKind::Local });
                    let f = Stmt::For {
                        init: vec![Stmt::Assign { dst: Expr::Var(iv), src: a }],
                        cond: Expr::cmp(if incl { BinOp::Le } else { BinOp::Lt }, Expr::Var(iv), x),
                        step: vec![Stmt::Expr(Expr::IncDec { e: Box::new(Expr::Var(iv)), delta: 1, post: true })],
                        body: lb,
                    };
                    b.splice(i..=i + 1, [f]);
                }
                i += 1;
            }
        });
    }
    for v in new_vars {
        vars.push(v);
        is_temp.push(false);
    }
}

fn strip_ne0(e: &Expr) -> &Expr {
    match strip(e) {
        Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => strip(l),
        x => x,
    }
}

/// `do { B; c = c - 1; } while (c != 0)` with c a CTR var: (c, B)
fn ctr_do(s: &Stmt, vars: &[Var]) -> Option<(VarId, Vec<Stmt>)> {
    let Stmt::DoWhile { body, cond } = s else { return None };
    let Expr::Binary { op: BinOp::Ne, l, r, .. } = strip(cond) else { return None };
    let Expr::Var(c) = strip(l) else { return None };
    if r.as_int() != Some(0) || !is_ctr(vars, *c) {
        return None;
    }
    let mut b = real(body);
    let last = b.pop()?;
    if !is_dec(&last, *c) || b.iter().any(|s| crate::idioms::stmt_mentions(s, *c)) {
        return None;
    }
    Some((*c, b))
}

fn shape_b_at(b: &[Stmt], k: usize, vars: &[Var], is_temp: &[bool]) -> Option<(usize, Vec<Stmt>)> {
    let Stmt::If { cond: tc, then, els } = &b[k] else { return None };
    // `t != 0`
    let t = strip_ne0(tc).clone();
    if matches!(strip(tc), Expr::Binary { op: BinOp::Ne, .. }) == false {
        return None;
    }
    // ctr = t right before (and t = n >> s before that, or t inline)
    if k == 0 {
        return None;
    }
    let Stmt::Assign { dst: Expr::Var(ctr), src: ci } = &b[k - 1] else { return None };
    if !is_ctr(vars, *ctr) || strip(ci) != &t && !matches!((strip(ci), &t), (Expr::Var(a), Expr::Var(b)) if a == b) {
        return None;
    }
    let mut start = k - 1;
    let shifted = match &t {
        Expr::Var(tv) if is_temp.get(*tv).copied().unwrap_or(false) => {
            let Stmt::Assign { dst: Expr::Var(x), src } = b.get(k.checked_sub(2)?)? else { return None };
            if x != tv {
                return None;
            }
            start = k - 2;
            src.clone()
        }
        e => e.clone(),
    };
    let Expr::Binary { op: BinOp::Shr, l: nexp, r: sh, .. } = strip(&shifted) else { return None };
    let s = sh.as_int()?;
    if !(1..=3).contains(&s) {
        return None;
    }
    let f = 1i64 << s;
    let Expr::Var(nx) = strip(nexp) else { return None };
    let nx = *nx;
    // then: [main loop, n = n & (F-1), if (n != 0) goto R]
    let then = real(then);
    let els = real(els);
    if then.len() != 3 || els.len() < 2 {
        return None;
    }
    let n = match &then[1] {
        Stmt::Assign { dst: Expr::Var(x), .. } => *x,
        _ => return None,
    };
    // the shift may read the value n was copied from (`n = t; ... t >> 3`)
    if nx != n {
        // the copy is in this list, or (when n isn't touched here) before an enclosing guard
        if let Some(j) = (0..start).rev().find(|&j| crate::idioms::stmt_mentions(&b[j], n)) {
            if !matches!(&b[j], Stmt::Assign { dst: Expr::Var(x), src } if *x == n && matches!(strip(src), Expr::Var(y) if *y == nx)) {
                return None;
            }
        }
    }
    let (c1, _main) = ctr_do(&then[0], vars)?;
    if c1 != *ctr {
        return None;
    }
    match &then[1] {
        Stmt::Assign { dst: Expr::Var(x), src: Expr::Binary { op: BinOp::And, l, r, .. } } if *x == n && matches!(strip(l), Expr::Var(y) if *y == n) && r.as_int() == Some(f - 1) => {}
        _ => return None,
    }
    let Stmt::If { cond: gc, then: gt, els: ge } = &then[2] else { return None };
    if !ge.is_empty() || !matches!(strip_ne0(gc), Expr::Var(y) if *y == n) || gt.len() != 1 || !matches!(gt[0], Stmt::Goto(_)) {
        return None;
    }
    // else: [ctr2 = n, remainder loop] (the label was stripped by real())
    let Stmt::Assign { dst: Expr::Var(c2), src: c2i } = &els[0] else { return None };
    if !matches!(strip(c2i), Expr::Var(y) if *y == n) {
        return None;
    }
    let (c2b, rem) = ctr_do(&els[1], vars)?;
    if c2b != *c2 || els.len() != 2 {
        return None;
    }
    let ty = vars[n].ty.clone();
    let mut lb = rem;
    lb.push(Stmt::Assign { dst: Expr::Var(n), src: Expr::bin(BinOp::Sub, Expr::Var(n), Expr::int(1), ty.clone()) });
    let w = Stmt::While { cond: Expr::cmp(BinOp::Ne, Expr::Var(n), Expr::Int { value: 0, ty }), body: lb };
    // the count from the pointer loop's bounds: `n = ((e + (S-1)) - p) >> log S` -> `while (p < e)`
    Some((start, vec![w]))
}
