//! Template matching in target IR.
//!
//! Target trees are matched modulo single-definition temporaries (a `Var` defined once by a
//! pure expression is looked through), so the compiler's CSE, scheduling and register choice
//! don't matter. Commutative operators match in either order. An object hole (`const C&`
//! parameter) binds to the scalar components the pattern reads; afterwards the components must
//! be one consistent lvalue of class `C` (`p->mPos`), or be explained again as the result of an
//! object-returning template (a nested inline, `Dot(a - b, c)`) or a constructor.

use crate::probe::CallKind;
use crate::template::{HoleKind, Shape, Template};
use crate::util::*;
use crate::InlineLib;
use mwdec_core::{Type, TypeDb};
use mwdec_lift::types::ty_of;
use mwdec_lift::{BinOp, Callee, Expr, IrFunction, Stmt, Var, VarId, VarKind};
use std::collections::{BTreeMap, HashMap};

pub type Defs = HashMap<VarId, Expr>;

pub struct Env<'a> {
    pub db: &'a TypeDb,
    pub vars: &'a [Var],
    pub defs: &'a Defs,
    pub lib: &'a InlineLib,
    /// object templates by class (most specific first)
    pub objects: &'a HashMap<String, Vec<usize>>,
}

/// Resolve single-definition temps.
pub fn res<'e>(mut e: &'e Expr, defs: &'e Defs) -> &'e Expr {
    for _ in 0..64 {
        match e {
            Expr::Var(v) => match defs.get(v) {
                Some(d) => e = d,
                None => return e,
            },
            _ => return e,
        }
    }
    e
}

/// Fully expanded copy (temps substituted), bounded: shared temps would otherwise expand
/// exponentially, so past a node budget the remaining temps stay variables.
pub fn expand(e: &Expr, defs: &Defs) -> Expr {
    let mut budget: i64 = 4000;
    expand_b(e, defs, &mut budget, 0)
}

fn expand_b(e: &Expr, defs: &Defs, budget: &mut i64, depth: u32) -> Expr {
    let r = res(e, defs);
    let mut x = r.clone();
    let mut size = 0i64;
    x.walk(&mut |_| size += 1);
    *budget -= size;
    if *budget <= 0 || depth > 24 {
        return x;
    }
    x.rewrite(&mut |n| {
        if *budget <= 0 {
            return;
        }
        if let Expr::Var(_) = n {
            let r = res(n, defs);
            if !matches!(r, Expr::Var(_)) {
                *n = expand_b(r, defs, budget, depth + 1);
            } else {
                *n = r.clone();
            }
        }
    });
    x
}

pub fn teq(a: &Expr, b: &Expr, defs: &Defs) -> bool {
    a == b || expand(a, defs) == expand(b, defs)
}

fn commutative(op: BinOp) -> bool {
    matches!(op, BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor | BinOp::Eq | BinOp::Ne)
}

/// Coarse value class for type checks: 1 float, 2 pointer, 3 integer, 4 class, 0 unknown.
fn vclass(t: &Type, db: &TypeDb) -> u8 {
    let r = mwdec_lift::types::resolve(Some(db), strip(t)).into_owned();
    match strip(&r) {
        Type::Float { .. } => 1,
        Type::Ptr(_) | Type::Ref(_) | Type::FuncPtr(_) => 2,
        Type::Int { .. } | Type::Long { .. } | Type::Char | Type::WChar | Type::Bool => 3,
        Type::Named(_) if mwdec_lift::types::is_enum(Some(db), &r) => 3,
        Type::Named(_) => 4,
        _ => 0,
    }
}

fn compat(a: &Type, b: &Type, db: &TypeDb) -> bool {
    let (x, y) = (vclass(a, db), vclass(b, db));
    x == y || x == 0 || y == 0 || (x == 2 && y == 3) || (x == 3 && y == 2)
}

/// Is the cast to `to` a value-preserving widening of an integer of type `from` (zero
/// extension of an unsigned narrow type, or sign extension to a signed one)?
fn widening(to: &Type, from: &Type) -> bool {
    match (strip(to), strip(from)) {
        (Type::Int { size: 4, signed: s }, Type::Int { size: n, signed: f }) => *n < 4 && (!*f || *s),
        (Type::Int { size: 4, .. }, Type::Bool | Type::Char) => true,
        _ => false,
    }
}

/// `(hole, offset, type)` of a component read `h->m` / `h.m` in a pattern.
fn comp_of<'p>(p: &'p Expr, holes: &[HoleKind]) -> Option<(usize, i32, &'p Type)> {
    match p {
        Expr::Load { base, offset, ty } | Expr::Member { base, offset, ty } => match &**base {
            Expr::Var(h) if matches!(holes.get(*h), Some(HoleKind::Obj { .. })) => Some((*h, *offset, ty)),
            _ => None,
        },
        _ => None,
    }
}

/// Pattern address of (a member of) an object hole: `h`, `&h->m`, `(char*)h + k` -> (h, k).
fn pat_canon(p: &Expr, holes: &[HoleKind]) -> Option<(usize, i32)> {
    let is_obj = |h: &usize| matches!(holes.get(*h), Some(HoleKind::Obj { .. }));
    match p {
        Expr::Var(h) if is_obj(h) => Some((*h, 0)),
        Expr::Cast { ty, e } if matches!(strip(ty), Type::Ptr(_)) => pat_canon(e, holes),
        Expr::AddrOf(x) => match &**x {
            Expr::Load { base, offset, .. } => pat_canon(base, holes).map(|(h, k)| (h, k + offset)),
            Expr::Member { base, offset, .. } => match &**base {
                Expr::Var(h) if is_obj(h) => Some((*h, *offset)),
                b => pat_canon(&Expr::AddrOf(Box::new(b.clone())), holes).map(|(h, k)| (h, k + offset)),
            },
            Expr::Var(h) if is_obj(h) => Some((*h, 0)),
            _ => None,
        },
        Expr::Binary { op: BinOp::Add, l, r, .. } => match (&**l, &**r) {
            (Expr::Cast { ty, e }, Expr::Int { value, .. }) if matches!(strip(ty), Type::Ptr(inner) if mwdec_lift::scalar_size(inner) == Some(1)) => {
                pat_canon(e, holes).map(|(h, k)| (h, k + *value as i32))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Address expression `base + off` for an object of class `cls`.
fn mk_addr(base: Expr, off: i32, cls: &str) -> Expr {
    if off == 0 {
        return base;
    }
    let ct = Type::Named(cls.to_string());
    match base {
        Expr::AddrOf(x) => Expr::AddrOf(Box::new(Expr::Member { base: x, offset: off, ty: ct })),
        b => Expr::AddrOf(Box::new(Expr::Load { base: Box::new(b), offset: off, ty: ct })),
    }
}

#[derive(Clone, Debug)]
pub enum Bind {
    Val(Expr),
    Comps(BTreeMap<i32, Expr>),
}

pub struct M<'a, 'e> {
    pub env: &'a Env<'e>,
    pub t: &'a Template,
    pub b: Vec<Option<Bind>>,
    /// statement templates: pattern locals folded into their uses (hole, pattern value)
    pub folded: Vec<(usize, Expr)>,
}

impl<'a, 'e> M<'a, 'e> {
    pub fn new(env: &'a Env<'e>, t: &'a Template) -> Self {
        M { env, t, b: vec![None; t.holes.len()], folded: vec![] }
    }

    fn defs(&self) -> &'e Defs {
        self.env.defs
    }

    /// Every dead-store pattern of the template has a distinct target dead store of its size
    /// whose value matches under the current bindings; the number matched.
    fn dead_stores_match(&self) -> Option<usize> {
        TARGET_DEAD.with(|d| {
            let d = d.borrow();
            let mut used: Vec<usize> = vec![];
            for (size, pat) in &self.t.dead {
                let found = d.iter().enumerate().find(|(k, (sz, v))| {
                    if used.contains(k) || sz != size {
                        return false;
                    }
                    let mut m2 = M { env: self.env, t: self.t, b: self.b.clone(), folded: vec![] };
                    if m2.m(pat, v) {
                        return true;
                    }
                    // a member of a named object read through its address, as stores see it
                    let mut v2 = v.clone();
                    v2.rewrite(&mut |x| {
                        if let Expr::Member { base, offset, ty } = x {
                            if matches!(**base, Expr::Global { .. } | Expr::Var(_)) {
                                *x = Expr::Load { base: Box::new(Expr::AddrOf(base.clone())), offset: *offset, ty: ty.clone() };
                            }
                        }
                    });
                    let mut m3 = M { env: self.env, t: self.t, b: self.b.clone(), folded: vec![] };
                    v2 != *v && m3.m(pat, &v2)
                });
                used.push(found?.0);
            }
            Some(used.len())
        })
    }

    /// Bind object hole `h` to an address (unifying with components bound before).
    fn bind_addr(&mut self, h: usize, addr: Expr) -> bool {
        let defs = self.defs();
        match &self.b[h] {
            None => {
                self.b[h] = Some(Bind::Val(addr));
                true
            }
            Some(Bind::Val(x)) => {
                if teq(x, &addr, defs) {
                    return true;
                }
                let (a, ao) = crate::addr::canon_ptr(x, self.env);
                let (b, bo) = crate::addr::canon_ptr(&addr, self.env);
                ao == bo && teq(&a, &b, defs)
            }
            Some(Bind::Comps(m)) => {
                let (lp, lo) = crate::addr::canon_ptr(&addr, self.env);
                let ok = m.iter().all(|(off, e)| match crate::addr::access(res(e, defs), self.env) {
                    Some((p2, o2)) => o2 == lo + off && teq(&p2, &lp, defs),
                    None => false,
                });
                if ok {
                    self.b[h] = Some(Bind::Val(addr));
                }
                ok
            }
        }
    }

    pub fn m(&mut self, p: &Expr, t: &Expr) -> bool {
        let defs = self.defs();
        // a value an earlier fold already named (`size()`): matched in its expanded form
        if !matches!(p, Expr::Var(_) | Expr::Call { .. }) {
            if let Some(u) = unfold(res(t, defs), self.env) {
                return self.m(p, &u);
            }
        }
        // the value of a scalar reference parameter: bound to the lvalue read
        if let Expr::Load { base, offset: 0, .. } | Expr::Member { base, offset: 0, .. } = p {
            if let Expr::Var(h) = &**base {
                if matches!(self.t.holes.get(*h), Some(HoleKind::ScalarRef(_))) {
                    let rt = res(t, defs);
                    if !matches!(rt, Expr::Load { .. } | Expr::Member { .. } | Expr::Global { .. }) {
                        return false;
                    }
                    return match &self.b[*h] {
                        None => {
                            self.b[*h] = Some(Bind::Val(rt.clone()));
                            true
                        }
                        Some(Bind::Val(x)) => teq(x, rt, defs),
                        _ => false,
                    };
                }
            }
        }
        if let Expr::Var(h) = p {
            if matches!(self.t.holes.get(*h), Some(HoleKind::ScalarRef(_))) {
                // the referenced value (how the probe's reference parameter reads): bound to the
                // lvalue read, or to the literal the compiler made for a constant argument
                let rt = res(t, defs);
                // (or a variable: a parameter or local passed by reference)
                let ok = matches!(rt, Expr::Load { .. } | Expr::Member { .. } | Expr::Global { .. }) || matches!(t, Expr::Var(v) if matches!(self.env.vars[*v].kind, VarKind::Param { .. } | VarKind::Local | VarKind::Stack { .. }));
                let rt = if matches!(t, Expr::Var(_)) && !ok_lvalue(rt) { t } else { rt };
                if !ok {
                    return false;
                }
                return match &self.b[*h] {
                    None => {
                        self.b[*h] = Some(Bind::Val(rt.clone()));
                        true
                    }
                    Some(Bind::Val(x)) => teq(x, rt, defs),
                    _ => false,
                };
            }
        }
        // the address of (a member of) an object hole: compare canonical addresses
        if !matches!(p, Expr::Var(_)) {
            if let Some((h, k)) = pat_canon(p, &self.t.holes) {
                let rt = res(t, defs);
                if !matches!(vclass(&ty_of(rt, self.env.vars), self.env.db), 2 | 0) {
                    return false;
                }
                let (tb, to) = crate::addr::canon_ptr(t, self.env);
                let cls = match &self.t.holes[h] {
                    HoleKind::Obj { class, .. } => class.clone(),
                    _ => return false,
                };
                return self.bind_addr(h, mk_addr(tb, to - k, &cls));
            }
        }
        if let Some((h, off, ty)) = comp_of(p, &self.t.holes) {
            // a pointer component read as a word (`(unsigned int)it.mNode == 0`)
            let t = match res(t, defs) {
                Expr::Cast { ty: ct, e: inner } if matches!(strip(ct), Type::Int { size: 4, .. }) && vclass(ty, self.env.db) == 2 && vclass(&ty_of(res(inner, defs), self.env.vars), self.env.db) == 2 => &**inner,
                // a narrow member widened for the compare (`(unsigned int)a.m == ...`)
                Expr::Cast { ty: ct, e: inner } if widening(ct, ty) && widening(ct, &ty_of(res(inner, defs), self.env.vars)) => &**inner,
                _ => t,
            };
            let rt = res(t, defs);
            if !compat(ty, &ty_of(rt, self.env.vars), self.env.db) {
                return false;
            }
            match &mut self.b[h] {
                None => {
                    let mut m = BTreeMap::new();
                    m.insert(off, t.clone());
                    self.b[h] = Some(Bind::Comps(m));
                    true
                }
                Some(Bind::Comps(m)) => match m.get(&off) {
                    // (the same member reached through another pointer to the object)
                    Some(x) => {
                        let x = x.clone();
                        teq(&x, t, defs) || {
                            let a = crate::addr::access(res(&x, defs), self.env);
                            let b = crate::addr::access(rt, self.env);
                            matches!((a, b), (Some((p1, o1)), Some((p2, o2))) if o1 == o2 && teq(&p1, &p2, defs))
                        }
                    }
                    None => {
                        m.insert(off, t.clone());
                        true
                    }
                },
                Some(Bind::Val(addr)) => {
                    let (lp, lo) = crate::addr::canon_ptr(addr, self.env);
                    match crate::addr::access(rt, self.env) {
                        Some((p2, o2)) => o2 == lo + off && teq(&p2, &lp, defs),
                        None => false,
                    }
                }
            }
        } else if let Expr::Var(h) = p {
            let rt = res(t, defs);
            if let Some(HoleKind::Scalar(ty)) = self.t.holes.get(*h) {
                if !compat(ty, &ty_of(rt, self.env.vars), self.env.db) {
                    return false;
                }
            }
            if let Some(HoleKind::Obj { ptr: false, .. }) = self.t.holes.get(*h) {
                // a reference parameter used as the object itself (`*this = other`): bound to
                // the object's address
                if matches!(rt, Expr::Load { .. } | Expr::Member { .. } | Expr::Global { .. }) && matches!(strip(&ty_of(rt, self.env.vars)), Type::Named(_)) {
                    return self.bind_addr(*h, Expr::AddrOf(Box::new(rt.clone())));
                }
            }
            if matches!(self.t.holes.get(*h), Some(HoleKind::Obj { .. })) {
                return self.bind_addr(*h, t.clone());
            }
            match &self.b[*h] {
                None => {
                    self.b[*h] = Some(Bind::Val(t.clone()));
                    true
                }
                // (a pointer value used through a cast: `((int*)p)[i]` for `p[i]`)
                Some(Bind::Val(x)) => teq(x, t, defs) || matches!(t, Expr::Cast { ty, e } if matches!(strip(ty), Type::Ptr(_)) && teq(x, e, defs)),
                Some(Bind::Comps(_)) => false,
            }
        } else {
            let t = res(t, defs);
            // pointer casts don't change values
            if let Expr::Cast { ty, e: inner } = t {
                if matches!(strip(ty), Type::Ptr(_)) && !matches!(p, Expr::Cast { .. }) {
                    return self.m(p, inner);
                }
            }
            if let Expr::Cast { ty, e: inner } = p {
                if matches!(strip(ty), Type::Ptr(_)) && !matches!(t, Expr::Cast { .. }) {
                    return self.m(inner, t);
                }
            }
            // a pointer read as a word (`(unsigned int)p == 0`: the lifter's view of `cmplwi`)
            if let Expr::Cast { ty, e: inner } = t {
                if matches!(strip(ty), Type::Int { size: 4, .. }) && !matches!(p, Expr::Cast { .. }) && vclass(&ty_of(res(inner, defs), self.env.vars), self.env.db) == 2 {
                    return self.m(p, inner);
                }
            }
            match (p, t) {
                (Expr::Binary { op, l, r, ty }, Expr::Binary { op: op2, l: l2, r: r2, ty: ty2 }) => {
                    if op != op2 || (vclass(ty, self.env.db) == 1) != (vclass(ty2, self.env.db) == 1) {
                        return false;
                    }
                    let snap = self.b.clone();
                    if self.m(l, l2) && self.m(r, r2) {
                        return true;
                    }
                    if commutative(*op) {
                        self.b = snap;
                        if self.m(l, r2) && self.m(r, l2) {
                            return true;
                        }
                    }
                    false
                }
                (Expr::Unary { op, e, .. }, Expr::Unary { op: op2, e: e2, .. }) => op == op2 && self.m(e, e2),
                (Expr::Cast { ty, e }, Expr::Cast { ty: ty2, e: e2 }) => vclass(ty, self.env.db) == vclass(ty2, self.env.db) && self.m(e, e2),
                (Expr::Int { value, .. }, Expr::Int { value: v2, .. }) => value == v2,
                (Expr::Float { bits, double }, Expr::Float { bits: b2, double: d2 }) => bits == b2 && double == d2,
                (Expr::Global { symbol, .. }, Expr::Global { symbol: s2, .. }) => symbol == s2,
                (Expr::FuncAddr { symbol }, Expr::FuncAddr { symbol: s2 }) => symbol == s2,
                (Expr::Str { bytes }, Expr::Str { bytes: b2 }) => bytes == b2,
                (Expr::AddrOf(a), Expr::AddrOf(b)) => self.m(a, b),
                (Expr::Load { base, offset, ty }, Expr::Load { base: b2, offset: o2, ty: t2 }) | (Expr::Member { base, offset, ty }, Expr::Member { base: b2, offset: o2, ty: t2 }) => {
                    if !compat(ty, t2, self.env.db) {
                        return false;
                    }
                    let snap = self.b.clone();
                    if offset == o2 && self.m(base, b2) {
                        return true;
                    }
                    self.b = snap;
                    // the same address summed differently (`p->a[i]` vs `(q + 0x10000) + i*2 + 0x3f34`)
                    matches!(p, Expr::Load { .. }) && self.m_linear(base, *offset as i64, b2, *o2 as i64)
                }
                (Expr::Index { base, index, ty }, Expr::Index { base: b2, index: i2, ty: t2 }) => {
                    let snap = self.b.clone();
                    if self.m(base, b2) && self.m(index, i2) {
                        return true;
                    }
                    self.b = snap;
                    compat(ty, t2, self.env.db) && self.m_linear(&Expr::AddrOf(Box::new(p.clone())), 0, &Expr::AddrOf(Box::new(t.clone())), 0)
                }
                (Expr::Index { ty, .. }, Expr::Load { base: b2, offset: o2, ty: t2 }) if compat(ty, t2, self.env.db) => self.m_linear(&Expr::AddrOf(Box::new(p.clone())), 0, b2, *o2 as i64),
                // an element access spelled as indexing on one side and as an address sum on the other
                (Expr::Load { base, offset, ty }, Expr::Index { ty: t2, .. }) if compat(ty, t2, self.env.db) => self.m_linear(base, *offset as i64, &Expr::AddrOf(Box::new(t.clone())), 0),
                (Expr::Ternary { c, t: a, f, .. }, Expr::Ternary { c: c2, t: a2, f: f2, .. }) => self.m(c, c2) && self.m(a, a2) && self.m(f, f2),
                (Expr::Call { callee, args, .. }, Expr::Call { callee: c2, args: a2, .. }) => {
                    if args.len() != a2.len() {
                        return false;
                    }
                    let ok = match (callee, c2) {
                        (Callee::Direct { symbol, .. }, Callee::Direct { symbol: s2, .. }) => symbol == s2,
                        (Callee::Method { symbol, this, .. }, Callee::Method { symbol: s2, this: t2, .. }) => symbol == s2 && self.m(this, t2),
                        (Callee::Virtual { this, vtable_offset, .. }, Callee::Virtual { this: t2, vtable_offset: v2, .. }) => vtable_offset == v2 && self.m(this, t2),
                        _ => false,
                    };
                    ok && args.iter().zip(a2).all(|(x, y)| self.m(x, y))
                }
                _ => false,
            }
        }
    }

    /// Match two addresses as sums of terms plus a constant: one unbound pointer hole of the
    /// pattern takes the matching target pointer term plus the constant difference; the other
    /// terms pair up with equal scales.
    fn m_linear(&mut self, pb: &Expr, po: i64, tb: &Expr, to: i64) -> bool {
        let mut pt = vec![];
        let mut pc = po;
        if !linear(pb, 1, &mut pt, &mut pc, None, 0) {
            return false;
        }
        let mut tt = vec![];
        let mut tc = to;
        if !linear(tb, 1, &mut tt, &mut tc, Some(self.defs()), 0) {
            return false;
        }
        if pt.len() != tt.len() || pt.len() < 2 || pt.len() > 4 {
            return false;
        }
        // the pattern's pointer hole
        let Some(hi) = pt.iter().position(|(e, s)| *s == 1 && matches!(e, Expr::Var(h) if matches!(self.t.holes.get(*h), Some(HoleKind::Obj { ptr: true, .. })) && self.b[*h].is_none())) else { return false };
        let Expr::Var(h) = pt[hi].0.clone() else { return false };
        let HoleKind::Obj { class, .. } = self.t.holes[h].clone() else { return false };
        let snap = self.b.clone();
        for ti in 0..tt.len() {
            if tt[ti].1 != 1 || vclass(&ty_of(res(&tt[ti].0, self.defs()), self.env.vars), self.env.db) != 2 {
                continue;
            }
            self.b = snap.clone();
            // an object of the hole's class must live there (instantiations differing only in
            // a size parameter expand alike)
            let (cb, co) = crate::addr::canon_ptr(&tt[ti].0, self.env);
            let Some((addr, _)) = crate::addr::object_at(&cb, co + (tc - pc) as i32, &class, self.env) else { continue };
            if !self.bind_addr(h, addr) {
                continue;
            }
            let rest_p: Vec<&(Expr, i64)> = pt.iter().enumerate().filter(|(k, _)| *k != hi).map(|(_, x)| x).collect();
            let rest_t: Vec<&(Expr, i64)> = tt.iter().enumerate().filter(|(k, _)| *k != ti).map(|(_, x)| x).collect();
            if self.pair_terms(&rest_p, &rest_t) {
                return true;
            }
        }
        self.b = snap;
        false
    }

    fn pair_terms(&mut self, p: &[&(Expr, i64)], t: &[&(Expr, i64)]) -> bool {
        let Some((first, rest)) = p.split_first() else { return t.is_empty() };
        for k in 0..t.len() {
            if t[k].1 != first.1 {
                continue;
            }
            let snap = self.b.clone();
            if self.m(&first.0, &t[k].0) {
                let others: Vec<&(Expr, i64)> = t.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, x)| *x).collect();
                if self.pair_terms(rest, &others) {
                    return true;
                }
            }
            self.b = snap;
        }
        false
    }

    /// Turn bindings into call arguments (hole order) and the score of nested explanations.
    /// `depth` bounds nested explanations.
    pub fn finalize(&self, depth: u32) -> Option<(Vec<Expr>, i32)> {
        let mut out = vec![];
        let mut score = 0;
        // the dead frame stores the expansion leaves must be in the target too
        if !self.t.dead.is_empty() {
            score += self.dead_stores_match()? as i32 * 2;
        }
        for (h, k) in self.t.holes.iter().enumerate() {
            if matches!(k, HoleKind::Local) {
                continue;
            }
            let b = self.b[h].as_ref()?;
            match (k, b) {
                (HoleKind::Scalar(_), Bind::Val(e)) => out.push(e.clone()),
                (HoleKind::ScalarRef(t), Bind::Val(e)) => match literal_value(e, t, self.env.db) {
                    // a literal the compiler made for a constant argument: the constant
                    Some(v) => out.push(v),
                    None => out.push(Expr::AddrOf(Box::new(e.clone()))),
                },
                // a null pointer argument (`T(id, nullptr, ...)`), not `this`
                (HoleKind::Obj { class, ptr: true, .. }, Bind::Val(Expr::Int { value: 0, .. })) if !(h == 0 && matches!(self.t.kind, CallKind::Method)) => {
                    out.push(Expr::Int { value: 0, ty: Type::Ptr(Box::new(Type::Named(class.clone()))) });
                }
                (HoleKind::Obj { class, ptr, .. }, Bind::Val(e)) => {
                    // the address must hold an object of the hole's class; prefer the typed
                    // spelling of the canonical address over an untyped register
                    let (b, o) = crate::addr::canon_ptr(e, self.env);
                    match crate::addr::object_at(&b, o, class, self.env) {
                        Some((a, _)) => {
                            score -= class_depth(self.env.db, &b, o, class, self.env).unwrap_or(0) as i32;
                            out.push(if !typed_ptr_to(e, class, self.env) { a } else { e.clone() });
                        }
                        None => {
                            if typed_ptr_to(e, class, self.env) {
                                out.push(e.clone());
                            } else if !*ptr && matches!(k, HoleKind::Obj { temp_ok: true, .. }) && matches!(strip(&ty_of(res(e, self.env.defs), self.env.vars)), Type::Ptr(x) if matches!(strip(x), Type::Int { size: 1, .. } | Type::Void | Type::Unknown { .. })) {
                                // a const reference argument reached through a byte pointer
                                // (`(char*)items + i * 8`): the object there, cast
                                out.push(Expr::Cast { ty: Type::Ptr(Box::new(Type::Const(Box::new(Type::Named(class.clone()))))), e: Box::new(e.clone()) });
                            } else if *ptr && matches!(self.t.kind, CallKind::Ctor) && matches!(e, Expr::AddrOf(_)) {
                                // a pointer argument of a constructor is only a value: an address
                                // the types can't name (a member of a derived class reached
                                // through a base pointer) is passed cast
                                out.push(Expr::Cast { ty: Type::Ptr(Box::new(Type::Const(Box::new(Type::Named(class.clone()))))), e: Box::new(e.clone()) });
                            } else {
                                return None;
                            }
                        }
                    }
                }
                (HoleKind::Obj { class, temp_ok, ptr }, Bind::Comps(m)) => {
                    if let Some(v) = whole_value(self.env, m, class).filter(|_| *temp_ok) {
                        // every member of one object value (`r.mPos` set from `Lerp(...)`)
                        out.push(if *ptr { Expr::AddrOf(Box::new(v)) } else { v });
                    } else if let Some(a) = lvalue_addr(self.env, m, class) {
                        out.push(a);
                    } else if *temp_ok && depth < 3 && !matches!(self.t.shape, Shape::Stmts { .. }) {
                        let (v, sc) = explain_object(self.env, class, m, depth + 1)?;
                        score += sc;
                        // pointer holes (`this` of a const method) take the temporary's address
                        out.push(if *ptr { Expr::AddrOf(Box::new(v)) } else { v });
                    } else {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        Some((out, score))
    }
}

/// Penalty for explaining a value as a plain constructor call (last resort).
pub const CTOR_PENALTY: i32 = 100;

/// Minimum score of a rewrite (one operator, small tie-break penalties allowed).
pub const MIN_SCORE: i32 = 5;

/// Score of using template `t` with nested explanation score `extra`: 10 per pattern
/// operator, small negative tie-breaks (nesting depth of the bound objects).
pub fn use_score(t: &Template, extra: i32, nested: bool, args: &[Expr]) -> i32 {
    let mut s = t.ops as i32 * 10 + extra;
    if nested && matches!(t.kind, CallKind::Ctor) {
        // `CVector3f(1.f, 1.f, 1.f)` constants are a natural operand; other values built by a
        // constructor are a last resort
        let lit = |a: &Expr| matches!(a, Expr::Int { .. } | Expr::Float { .. });
        // (some literal members, `CVector3f(0.f, 0.f, h)`, are natural too)
        // (an object of inline results, `CVector3f(FastFSel(..), FastFSel(..), ..)`, is a value
        // the source named too)
        let inline_call = |a: &Expr| matches!(a, Expr::Call { callee: mwdec_lift::Callee::Direct { sig, .. } | mwdec_lift::Callee::Method { sig, .. }, .. } if sig.mangled.is_none());
        s -= if args.iter().all(lit) {
            5
        } else if args.iter().any(lit) {
            25
        } else if args.iter().all(inline_call) {
            40
        } else {
            CTOR_PENALTY
        };
    }
    s
}

/// Is `e` typed as a pointer/reference to `cls` (or a class derived from it)?
fn typed_ptr_to(e: &Expr, cls: &str, env: &Env) -> bool {
    let t = ty_of(e, env.vars);
    let inner = match strip(&t) {
        Type::Ptr(x) | Type::Ref(x) => x.clone(),
        _ => return false,
    };
    class_name(&inner, env.db).map_or(false, |c| is_base_or_same(env.db, cls, &c))
}

/// Nesting depth at which an object of `cls` lives at `p + off` (0: the pointee itself).
fn class_depth(db: &TypeDb, p: &Expr, off: i32, cls: &str, env: &Env) -> Option<u32> {
    let outer = match p {
        Expr::AddrOf(x) => class_name(&ty_of(x, env.vars), db)?,
        _ => class_name(mwdec_lift::pointee(&ty_of(p, env.vars))?, db)?,
    };
    fn go(db: &TypeDb, outer: &str, off: i32, cls: &str, d: u32) -> Option<u32> {
        if d > 10 {
            return None;
        }
        if off == 0 && mwdec_lift::sig::norm_name(outer) == mwdec_lift::sig::norm_name(cls) {
            return Some(d);
        }
        let c = mwdec_lift::sig::find_class(db, outer)?;
        for b in &c.bases {
            if let Some(bc) = mwdec_lift::sig::find_class(db, &b.name) {
                let bo = b.offset as i32;
                if off >= bo && off < bo + bc.size as i32 {
                    if let Some(x) = go(db, &b.name, off - bo, cls, d) {
                        return Some(x);
                    }
                }
            }
        }
        for f in &c.fields {
            if let Some(fc) = class_name(&f.ty, db) {
                let fs = mwdec_lift::types::size_of(Some(db), &f.ty).unwrap_or(0) as i32;
                let fo = f.offset as i32;
                if off >= fo && off < fo + fs.max(1) {
                    if let Some(x) = go(db, &fc, off - fo, cls, d + 1) {
                        return Some(x);
                    }
                }
            }
        }
        if off == 0 && is_base_or_same(db, cls, outer) {
            return Some(d);
        }
        None
    }
    go(db, &outer, off, cls, 0)
}

thread_local! {
    /// Memo of [`class_at_pub`] (big classes make it a scan of hundreds of fields, asked again
    /// for every template of a class). Keyed by the TypeDb's address; bounded.
    static CLASS_AT: std::cell::RefCell<(usize, HashMap<(String, i32, String), bool>)> = std::cell::RefCell::new((0, HashMap::new()));
}

/// Forget the [`class_at_pub`] memo. Its key is the TypeDb's address, which a later unit's
/// TypeDb can reuse once the earlier one is dropped: a stale answer then made the same function
/// fold differently depending on which units were drafted before it in the process.
pub fn reset_memos() {
    CLASS_AT.with(|c| {
        let mut c = c.borrow_mut();
        c.0 = 0;
        c.1.clear();
    });
}

pub fn class_at_pub(db: &TypeDb, outer: &str, off: i32, cls: &str) -> bool {
    let id = db as *const TypeDb as usize;
    let key = (outer.to_string(), off, cls.to_string());
    if let Some(r) = CLASS_AT.with(|c| {
        let c = c.borrow();
        if c.0 == id {
            c.1.get(&key).copied()
        } else {
            None
        }
    }) {
        return r;
    }
    let r = class_at(db, outer, off, cls, 0);
    CLASS_AT.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != id || c.1.len() > 100_000 {
            c.0 = id;
            c.1.clear();
        }
        c.1.insert(key, r);
    });
    r
}

/// The class (or a class derived from `cls` at offset 0) at `off` inside `outer`.
fn class_at(db: &TypeDb, outer: &str, off: i32, cls: &str, depth: u32) -> bool {
    if depth > 10 {
        return false;
    }
    if off == 0 && is_base_or_same(db, cls, outer) {
        return true;
    }
    let Some(c) = mwdec_lift::sig::find_class(db, outer) else { return false };
    for b in &c.bases {
        let bo = b.offset as i32;
        if let Some(bc) = mwdec_lift::sig::find_class(db, &b.name) {
            if off >= bo && off < bo + bc.size as i32 && class_at(db, &b.name, off - bo, cls, depth + 1) {
                return true;
            }
        }
    }
    for f in &c.fields {
        let fo = f.offset as i32;
        if fo > off {
            continue;
        }
        let r = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
        match strip(&r) {
            // raw storage of a class template holding its argument (`optional_object<T>`'s
            // `uchar m_data[sizeof(T)]`): the object starts there
            Type::Array(e, n) if off == fo && mwdec_lift::scalar_size(strip(e)) == Some(1) && template_arg_is(outer, cls) && mwdec_lift::types::size_of(Some(db), &Type::Named(cls.to_string())).is_some_and(|s| s as u64 <= *n as u64) => return true,
            Type::Array(e, n) => {
                if let (Some(ec), Some(es)) = (class_name(e, db), mwdec_lift::types::size_of(Some(db), e)) {
                    let es = es as i32;
                    if es > 0 && off >= fo && off < fo + es * *n as i32 && class_at(db, &ec, (off - fo) % es, cls, depth + 1) {
                        return true;
                    }
                }
            }
            t => {
                if let Some(fc) = class_name(t, db) {
                    let fs = mwdec_lift::types::size_of(Some(db), t).unwrap_or(0) as i32;
                    if off >= fo && off < fo + fs.max(1) && class_at(db, &fc, off - fo, cls, depth + 1) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Is `cls` a template argument of class template instance `outer`?
fn template_arg_is(outer: &str, cls: &str) -> bool {
    let Some(lt) = outer.find('<') else { return false };
    let inner = &outer[lt + 1..outer.len().saturating_sub(1)];
    let n = mwdec_lift::sig::norm_name(cls);
    mwdec_lift::sig::split_top(inner, ',').iter().any(|a| mwdec_lift::sig::norm_name(a.trim()) == n)
}

/// Address of the object of class `cls` whose components are `m`, if they are one lvalue.
pub fn lvalue_addr(env: &Env, m: &BTreeMap<i32, Expr>, cls: &str) -> Option<Expr> {
    let defs = env.defs;
    let mut base: Option<(Expr, i32)> = None;
    for (off, e) in m {
        let (p, o) = crate::addr::access(res(e, defs), env)?;
        let d = o - off;
        match &base {
            None => base = Some((p, d)),
            Some((p0, d0)) => {
                if *d0 != d || !teq(p0, &p, defs) {
                    return None;
                }
            }
        }
    }
    let (p, d) = base?;
    crate::addr::object_at(&p, d, cls, env).map(|(a, _)| a)
}

/// Explain the virtual object `m` (components of a `cls` value) as an object template call.
/// The object value whose members are exactly `m` (`Member(X, o)` for every flat field `o` of
/// `cls`), if one.
fn whole_value(env: &Env, m: &BTreeMap<i32, Expr>, cls: &str) -> Option<Expr> {
    let fields = crate::template::flat_fields(env.db, cls)?;
    if fields.len() != m.len() {
        return None;
    }
    let mut base: Option<&Expr> = None;
    for (o, _) in &fields {
        match m.get(o)? {
            Expr::Member { base: b, offset, .. } if offset == o => {
                if base.is_some_and(|x| x != &**b) {
                    return None;
                }
                base = Some(b);
            }
            _ => return None,
        }
    }
    let b = base?;
    let bty = match b {
        Expr::Construct { class, .. } => class.clone(),
        _ => ty_of(b, env.vars),
    };
    (class_name(&bty, env.db).as_deref() == Some(cls)).then(|| b.clone())
}

/// Is the object at address `a` const (reached from a pointer/reference to const)?
fn const_object(a: &Expr, env: &Env) -> bool {
    let mut e = a;
    loop {
        match e {
            Expr::AddrOf(x) => e = x,
            Expr::Member { base, .. } | Expr::Load { base, .. } => {
                if let Some(Type::Const(_)) = mwdec_lift::pointee(&ty_of(base, env.vars)).map(|t| t.clone()) {
                    return true;
                }
                e = base;
            }
            Expr::Cast { e: x, .. } => e = x,
            Expr::Var(v) => {
                return match strip(&env.vars[*v].ty) {
                    Type::Ptr(x) | Type::Ref(x) => matches!(&**x, Type::Const(_)),
                    _ => false,
                };
            }
            _ => return false,
        }
    }
}

pub fn explain_object(env: &Env, cls: &str, m: &BTreeMap<i32, Expr>, depth: u32) -> Option<(Expr, i32)> {
    let list = env.objects.get(&mwdec_lift::sig::norm_name(cls))?;
    let mut best: Option<(Expr, i32)> = None;
    for &ti in list {
        let t = &env.lib.templates[ti];
        let Shape::Object { comps, .. } = &t.shape else { continue };
        // every bound component must be produced by the template
        if !m.keys().all(|o| comps.iter().any(|c| c.off == *o)) {
            continue;
        }
        let mut mm = M::new(env, t);
        let ok = comps.iter().filter(|c| m.contains_key(&c.off)).all(|c| mm.m(&c.pat, &m[&c.off]));
        if !ok {
            continue;
        }
        if let Some((args, extra)) = mm.finalize(depth) {
            let sc = use_score(t, extra, true, &args);
            if best.as_ref().map_or(true, |(_, b)| sc > *b) {
                best = Some((make_call(t, args), sc));
            }
        }
    }
    best
}

/// The call expression of template `t` with `args` (hole order).
pub fn make_call(t: &Template, mut args: Vec<Expr>) -> Expr {
    let ret = t.sig.ret.clone();
    match t.kind {
        CallKind::Method => {
            let this = args.remove(0);
            Expr::Call { callee: Callee::Method { symbol: String::new(), sig: t.sig.clone(), this: Box::new(this), qualified: false }, args, ret }
        }
        CallKind::Free => Expr::Call { callee: Callee::Direct { symbol: t.sig.qualified_name.clone(), sig: t.sig.clone() }, args, ret },
        CallKind::Ctor => {
            let class = Type::Named(t.class.clone().unwrap_or_default());
            let mut sig = t.sig.clone();
            sig.ret = Type::Void;
            Expr::Construct { class, ctor: Some(sig), args }
        }
    }
}

// ---------------------------------------------------------------- the pass

fn count_defs(body: &[Stmt], n: &mut HashMap<VarId, usize>) {
    for s in body {
        match s {
            Stmt::Assign { dst: Expr::Var(v), .. } => *n.entry(*v).or_default() += 1,
            Stmt::If { then, els, .. } => {
                count_defs(then, n);
                count_defs(els, n);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => count_defs(body, n),
            Stmt::For { init, step, body, .. } => {
                count_defs(init, n);
                count_defs(step, n);
                count_defs(body, n);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    count_defs(&c.body, n);
                }
            }
            _ => {}
        }
    }
}

fn collect_defs(body: &[Stmt], n: &HashMap<VarId, usize>, vars: &[Var], out: &mut Defs) {
    for s in body {
        match s {
            Stmt::Assign { dst: Expr::Var(v), src } => {
                if n.get(v) == Some(&1) && matches!(vars[*v].kind, VarKind::Local) && !src.has_call() && !src.uses_var(*v) {
                    out.insert(*v, src.clone());
                }
            }
            Stmt::If { then, els, .. } => {
                collect_defs(then, n, vars, out);
                collect_defs(els, n, vars, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect_defs(body, n, vars, out),
            Stmt::For { init, step, body, .. } => {
                collect_defs(init, n, vars, out);
                collect_defs(step, n, vars, out);
                collect_defs(body, n, vars, out);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    collect_defs(&c.body, n, vars, out);
                }
            }
            _ => {}
        }
    }
}

pub fn build_defs(body: &[Stmt], vars: &[Var]) -> Defs {
    let mut n = HashMap::new();
    count_defs(body, &mut n);
    let mut d = Defs::new();
    collect_defs(body, &n, vars, &mut d);
    d
}

/// Index of usable templates.
pub struct Index {
    /// statement templates, most specific first
    pub stmts: Vec<usize>,
    /// reference-returning inlines with a computed address (`vec[i]`)
    pub refs: Vec<usize>,
    pub scalars: Vec<usize>,
    /// scalar templates whose pattern is a control-flow value (ternary / && / ||)
    pub cflow: Vec<usize>,
    pub objects: HashMap<String, Vec<usize>>,
    /// object + mutate templates usable on store groups, most specific first
    pub groups: Vec<usize>,
}

fn is_identity_copy(t: &Template) -> bool {
    match &t.shape {
        Shape::Object { comps, .. } => {
            comps.iter().all(|c| matches!(comp_of(&c.pat, &t.holes), Some((_, o, _)) if o == c.off)) && {
                let hs: std::collections::BTreeSet<usize> = comps.iter().filter_map(|c| comp_of(&c.pat, &t.holes).map(|x| x.0)).collect();
                hs.len() == 1
            }
        }
        _ => false,
    }
}

pub fn index(lib: &InlineLib) -> Index {
    let mut scalars = vec![];
    let mut stmts = vec![];
    let mut refs = vec![];
    let mut cflow = vec![];
    let mut objects: HashMap<String, Vec<usize>> = HashMap::new();
    let mut groups = vec![];
    for (i, t) in lib.templates.iter().enumerate() {
        match &t.shape {
            Shape::Scalar(Expr::AddrOf(inner)) if t.ret_ref && matches!(&**inner, Expr::Index { .. }) => {
                refs.push(i);
            }
            Shape::Scalar(p) => {
                // (a plain read only with the dead stores of a by-value class to tell it apart)
                if (t.ops >= 1 && !matches!(p, Expr::Var(_) | Expr::Load { .. } | Expr::Member { .. } | Expr::AddrOf(_) | Expr::Cast { .. })) || !t.dead.is_empty() {
                    scalars.push(i);
                    if crate::cflow::has_cflow(p) {
                        cflow.push(i);
                    }
                }
            }
            Shape::Object { class, comps } => {
                if comps.is_empty() || is_identity_copy(t) {
                    continue;
                }
                objects.entry(mwdec_lift::sig::norm_name(class)).or_default().push(i);
                groups.push(i);
            }
            Shape::Mutate { comps, .. } => {
                if !comps.is_empty() && (t.ops >= 1 || !t.dead.is_empty()) {
                    groups.push(i);
                }
            }
            Shape::Stmts { .. } => stmts.push(i),
        }
    }
    let key = |i: &usize| {
        let t = &lib.templates[*i];
        (t.guessed, std::cmp::Reverse(t.ops), matches!(t.kind, CallKind::Ctor) as u8, t.holes.len(), !t.name.contains("operator"), *i)
    };
    scalars.sort_by_key(key);
    cflow.sort_by_key(key);
    stmts.sort_by_key(key);
    refs.sort_by_key(key);
    groups.sort_by_key(key);
    for v in objects.values_mut() {
        v.sort_by_key(key);
    }
    Index { stmts, refs, scalars, cflow, objects, groups }
}

pub fn apply(ir: &mut IrFunction, lib: &InlineLib, db: &TypeDb) -> usize {
    // (per function: memos keyed by a TypeDb's address must not outlive it)
    reset_memos();
    TARGET_DEAD.with(|d| *d.borrow_mut() = ir.dead_stores.iter().map(|x| (x.size, x.value.clone())).collect());
    LITERALS.with(|d| *d.borrow_mut() = ir.literal_bytes.iter().cloned().collect());
    crate::walkptr::apply(ir, lib, db);
    let n = apply_inner(ir, lib, db);
    TARGET_DEAD.with(|d| d.borrow_mut().clear());
    LITERALS.with(|d| d.borrow_mut().clear());
    n
}

thread_local! {
    /// Bytes of the function's literal symbols while it is being rewritten.
    static LITERALS: std::cell::RefCell<HashMap<String, Vec<u8>>> = std::cell::RefCell::new(HashMap::new());
}

/// The value of literal symbol read `e` (`@N`, a splitter-named pool word) as type `t`.
fn literal_value(e: &Expr, t: &Type, db: &TypeDb) -> Option<Expr> {
    let sym = match e {
        Expr::Global { symbol, .. } => symbol,
        Expr::Load { base, offset: 0, .. } => match &**base {
            Expr::AddrOf(g) => match &**g {
                Expr::Global { symbol, .. } => symbol,
                _ => return None,
            },
            _ => return None,
        },
        _ => return None,
    };
    let b = LITERALS.with(|l| l.borrow().get(sym).cloned())?;
    let r = mwdec_lift::types::resolve(Some(db), strip(t)).into_owned();
    match (strip(&r), b.len()) {
        (Type::Float { size: 4 }, 4) => Some(Expr::Float { bits: u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as u64, double: false }),
        (Type::Float { size: 8 }, 8) => Some(Expr::Float { bits: u64::from_be_bytes(b[..8].try_into().ok()?), double: true }),
        (Type::Int { size: 4, .. } | Type::Long { .. } | Type::Named(_), 4) => Some(Expr::Int { value: i32::from_be_bytes([b[0], b[1], b[2], b[3]]) as i64, ty: strip(t).clone() }),
        _ => None,
    }
}

thread_local! {
    /// The function's dead frame stores (size, value) while it is being rewritten.
    static TARGET_DEAD: std::cell::RefCell<Vec<(u32, Expr)>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn apply_inner(ir: &mut IrFunction, lib: &InlineLib, db: &TypeDb) -> usize {
    // members' own constructors' stores at the start of a constructor body are implicit
    let (stripped, pending) = if std::env::var("MWDI_NO_CTORS").is_ok() { (0, Default::default()) } else { crate::defctor::strip(ir, db, &lib.default_ctors) };
    let stripped = stripped + crate::defctor::copy_ctor_inits(ir, db, &lib.copy_ctors);
    // stack buffers filled as a default construction and passed by reference: `T()`
    let stripped = stripped + crate::defctor::default_temps(ir, db, &lib.default_ctors);
    // stack buffers built as a default construction and used as a named object: `T v;`
    let stripped = stripped + crate::defctor::default_locals(ir, db, &lib.default_ctors);
    // list loops, before folding can merge the walking pointer with its first value
    let stash = if std::env::var("MWDI_NO_ITERLOOPS").is_err() { crate::iterloops::lists_prefold(ir, db) } else { vec![] };
    let stripped = stripped + stash.len();
    if lib.templates.is_empty() {
        crate::iterloops::unstash(ir, stash);
        return stripped + crate::defctor::finish(ir, db, &pending);
    }
    let idx = index(lib);
    let mut total = stripped + if std::env::var("MWDI_NO_CTORS").is_ok() { 0 } else { crate::ctors::strip_member_ctor_stores(ir, db) };
    // stack slots typed only by size that are objects passed to calls
    if std::env::var("MWDI_NO_BUFFERS").is_err() {
        total += crate::buffers::type_object_slots(ir, db);
    }
    total += crate::util::prof::time(0, || crate::objlocals::group(ir, lib, &idx, db));
    for _round in 0..4 {
        let raw = build_defs(&ir.body, &ir.vars);
        let vars = ir.vars.clone();
        let defs = crate::util::prof::time(1, || {
            let env0 = Env { db, vars: &vars, defs: &raw, lib, objects: &idx.objects };
            crate::safety::safe_defs(&ir.body, &env0)
        });
        let whole = ir.body.clone();
        let cx = crate::walk::Ctx { db, vars: &vars, global: &defs, lib, idx: &idx, whole: &whole };
        let n = crate::walk::walk(&mut ir.body, &mut Defs::new(), &cx);
        if n == 0 {
            break;
        }
        total += n;
        drop_write_only_stack(&mut ir.body, &ir.vars);
        dce(&mut ir.body, &ir.vars);
    }
    // constructor initializer lists (`mNormal(Cross(b - a, c - a))`)
    if !ir.init_list.is_empty() {
        let raw = build_defs(&ir.body, &ir.vars);
        let vars = ir.vars.clone();
        let env = Env { db, vars: &vars, defs: &raw, lib, objects: &idx.objects };
        for init in ir.init_list.iter_mut() {
            for a in init.args.iter_mut() {
                total += scalar_expr(a, &env, &idx);
            }
        }
    }
    crate::iterloops::unstash(ir, stash);
    // loops over pointer-iterated containers, as iterator loops
    if std::env::var("MWDI_NO_ITERLOOPS").is_err() {
        total += crate::iterloops::apply(ir, db);
    }
    // container forwarders and negated predicates (`stmtinl`)
    total += crate::stmtinl::apply(ir, lib, db);
    // members built explicitly, now that their values are folded calls
    total += crate::defctor::finish(ir, db, &pending);
    if total > 0 {
        crate::post::forward_stack_temps(&mut ir.body, &ir.vars);
        crate::post::name_shared_objects(&mut ir.body, &mut ir.vars);
        dce(&mut ir.body, &ir.vars);
        crate::post::forward_stack_temps(&mut ir.body, &ir.vars);
        crate::post::forward_cond_temps(&mut ir.body, &ir.vars);
        crate::post::return_values(&mut ir.body, &ir.vars);
        crate::post::forward_temps_into_folded(&mut ir.body, &ir.vars);
        // explicit calls of inline destructors test their objects themselves
        crate::defctor::inline_destructor_tests(&mut ir.body);
        // (a destructor folded only now anchors a default-constructed local)
        crate::defctor::default_locals(ir, db, &lib.default_ctors);
        // temporaries whose construction only folding made one expression (`f(T(a, b))`)
        let vars = ir.vars.clone();
        mwdec_lift::idioms::constructed_arg_temporaries(&mut ir.body, &vars);
        crate::post::fold_flag_chains(&mut ir.body, &ir.vars);
    }
    crate::stmtinl::finish(ir);
    total += crate::walkptr::constructed_buffers(ir, db);
    total
}

/// Stores to scalar stack slots that are never read (the dead component stores MWCC leaves
/// for a temporary whose address an inline took, e.g. `(a - b).MagSquared()`): the folded call
/// recreates them, and as plain locals they would be dead code anyway.
pub fn drop_write_only_stack(body: &mut Vec<Stmt>, vars: &[Var]) {
    let mut mentions: HashMap<VarId, usize> = HashMap::new();
    let mut stores: HashMap<VarId, usize> = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            *mentions.entry(*v).or_default() += 1;
        }
    });
    fn count(b: &[Stmt], vars: &[Var], stores: &mut HashMap<VarId, usize>) {
        for s in b {
            match s {
                Stmt::Assign { dst: Expr::Var(v), .. } if matches!(vars[*v].kind, VarKind::Stack { .. }) => *stores.entry(*v).or_default() += 1,
                Stmt::If { then, els, .. } => {
                    count(then, vars, stores);
                    count(els, vars, stores);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => count(body, vars, stores),
                Stmt::For { init, step, body, .. } => {
                    count(init, vars, stores);
                    count(step, vars, stores);
                    count(body, vars, stores);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        count(&c.body, vars, stores);
                    }
                }
                _ => {}
            }
        }
    }
    count(body, vars, &mut stores);
    let dead: Vec<VarId> = stores
        .iter()
        .filter(|(v, n)| mentions.get(v) == Some(n) && matches!(strip(&vars[**v].ty), Type::Float { .. } | Type::Int { .. }))
        .map(|(v, _)| *v)
        .collect();
    if dead.is_empty() {
        return;
    }
    Stmt::for_each_block_mut(body, &mut |b| {
        b.retain(|s| !matches!(s, Stmt::Assign { dst: Expr::Var(v), src } if dead.contains(v) && !src.has_call()));
    });
}

/// Scalar templates in the statement's own expressions (not its nested statement lists).
pub fn scalar_shallow(s: &mut Stmt, env: &Env, idx: &Index) -> usize {
    match s {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => scalar_expr(e, env, idx),
        Stmt::Assign { dst, src } => {
            let mut n = scalar_expr(src, env, idx);
            // inside the destination's address only (not the stored-to lvalue itself)
            n += match dst {
                Expr::Load { base, .. } | Expr::Member { base, .. } => scalar_expr(base, env, idx),
                Expr::Index { base, index, .. } => scalar_expr(base, env, idx) + scalar_expr(index, env, idx),
                _ => 0,
            };
            n
        }
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } | Stmt::For { cond, .. } => scalar_expr(cond, env, idx),
        Stmt::Switch { e, .. } => scalar_expr(e, env, idx),
        _ => 0,
    }
}

pub fn scalar_stmt(s: &mut Stmt, env: &Env, idx: &Index) -> usize {
    let mut n = 0;
    match s {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => n += scalar_expr(e, env, idx),
        Stmt::Assign { dst, src } => {
            n += scalar_expr(src, env, idx);
            if !matches!(dst, Expr::Var(_)) {
                n += scalar_expr(dst, env, idx);
            }
        }
        Stmt::If { cond, then, els } => {
            n += scalar_expr(cond, env, idx);
            for s in then.iter_mut().chain(els.iter_mut()) {
                n += scalar_stmt(s, env, idx);
            }
        }
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            n += scalar_expr(cond, env, idx);
            for s in body {
                n += scalar_stmt(s, env, idx);
            }
        }
        Stmt::For { init, cond, step, body } => {
            n += scalar_expr(cond, env, idx);
            for s in init.iter_mut().chain(step.iter_mut()).chain(body.iter_mut()) {
                n += scalar_stmt(s, env, idx);
            }
        }
        Stmt::Switch { e, cases } => {
            n += scalar_expr(e, env, idx);
            for c in cases {
                for s in c.body.iter_mut() {
                    n += scalar_stmt(s, env, idx);
                }
            }
        }
        _ => {}
    }
    n
}

fn root_ok(p: &Expr, t: &Expr) -> bool {
    match (p, t) {
        (Expr::Binary { op, .. }, Expr::Binary { op: o2, .. }) => op == o2,
        (Expr::Unary { op, .. }, Expr::Unary { op: o2, .. }) => op == o2,
        (Expr::Call { .. }, Expr::Call { .. }) | (Expr::Ternary { .. }, Expr::Ternary { .. }) | (Expr::Index { .. }, Expr::Index { .. }) => true,
        _ => false,
    }
}

fn scalar_expr(e: &mut Expr, env: &Env, idx: &Index) -> usize {
    // an accessor chain folded into one load (`front().get()`: `composed`)
    if crate::composed::try_split(e, env) {
        if let Expr::Load { base, .. } = e {
            return 1 + scalar_expr(base, env, idx);
        }
    }
    // a one-member object rebuilt from the member of another one (`TUniqueId(ids[i].value)`,
    // a by-value argument in a register): that object
    if let Expr::Construct { class, args, .. } = &*e {
        if let (Some(cn), [a]) = (class_name(class, env.db), args.as_slice()) {
            if crate::template::flat_fields(env.db, &cn).is_some_and(|f| f.len() == 1 && f[0].0 == 0) {
                let mut a = a;
                while let Expr::Cast { e: inner, .. } = a {
                    a = inner;
                }
                let obj = match a {
                    Expr::Member { base, offset: 0, .. } if class_name(&ty_of(base, env.vars), env.db).as_deref() == Some(cn.as_str()) => Some((**base).clone()),
                    Expr::Load { base, offset: 0, .. } if mwdec_lift::pointee(&ty_of(base, env.vars)).and_then(|t| class_name(t, env.db)).as_deref() == Some(cn.as_str()) => {
                        Some(Expr::Load { base: base.clone(), offset: 0, ty: Type::Named(cn.clone()) })
                    }
                    _ => None,
                };
                if let Some(o) = obj {
                    *e = o;
                    return 1;
                }
            }
        }
    }
    // an object built member-wise (`CVector3f(a.x * s + b.x, ...)`): the value of object-valued
    // inlines (`a * s + b`)
    if let Expr::Construct { class, args, .. } = &*e {
        if let Some(cn) = class_name(class, env.db) {
            if let Some(fields) = crate::template::flat_fields(env.db, &cn) {
                if fields.len() == args.len() && args.len() > 1 && args.iter().any(|a| !matches!(a, Expr::Int { .. } | Expr::Float { .. })) {
                    let m: BTreeMap<i32, Expr> = fields.iter().zip(args.iter()).map(|((o, _), a)| (*o, a.clone())).collect();
                    if let Some((call, sc)) = explain_object(env, &cn, &m, 0) {
                        if sc >= MIN_SCORE && !matches!(call, Expr::Construct { .. }) {
                            *e = call;
                            return 1;
                        }
                    }
                }
            }
        }
    }
    // try at this node (the node itself, not through a temp: the temp's def is visited where it
    // is defined)
    if !matches!(e, Expr::Var(_)) {
        let mut best: Option<(Expr, i32)> = None;
        for &ti in &idx.scalars {
            let t = &env.lib.templates[ti];
            let Shape::Scalar(p) = &t.shape else { continue };
            if t.ret_ref || !root_ok(p, e) {
                continue;
            }
            // the inline's result type must be the expression's (no float/double mixups)
            if let (Type::Float { size: a }, Type::Float { size: b }) = (strip(&t.sig.ret), strip(&ty_of(e, env.vars))) {
                if a != b {
                    continue;
                }
            }
            if best.as_ref().map_or(false, |(_, b)| (t.ops as i32) * 10 + 9 < *b) {
                continue;
            }
            let mut m = M::new(env, t);
            static TRS: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
            let tr = TRS.get_or_init(|| std::env::var("MWDI_TRACE_SCALAR").ok()).as_ref().is_some_and(|f| t.name.contains(f.as_str()));
            if !m.m(p, e) {
                if tr {
                    eprintln!("SCALAR {} no match
  {p:?}
  {e:?}", t.name);
                }
                continue;
            }
            let Some((args, extra)) = m.finalize(0) else {
                if tr {
                    eprintln!("SCALAR {} finalize {:?}", t.name, m.b);
                }
                continue;
            };
            let sc = use_score(t, extra, false, &args);
            if tr {
                eprintln!("SCALAR {} score {sc}", t.name);
            }
            if best.as_ref().map_or(true, |(_, b)| sc > *b) {
                best = Some((make_call(t, args), sc));
            }
        }
        // reference-returning inlines: the referenced lvalue (`vec[i]`), or its address
        if best.is_none() {
            for &ti in &idx.refs {
                let t = &env.lib.templates[ti];
                let Shape::Scalar(p @ Expr::AddrOf(inner)) = &t.shape else { continue };
                let (pat, addr) = match e {
                    // the address of a const element (`&cvec[i]` is `const T*`) can't stand where
                    // the lifter typed a plain pointer
                    Expr::AddrOf(_) if matches!(strip(&t.sig.ret), Type::Ref(inner) if matches!(&**inner, Type::Const(_))) => continue,
                    Expr::AddrOf(_) => (p, true),
                    Expr::Index { .. } => (&**inner, false),
                    _ => continue,
                };
                let mut m = M::new(env, t);
                if !m.m(pat, e) {
                    continue;
                }
                let Some((args, _)) = m.finalize(0) else { continue };
                // an element address of a const container is a `const T*` (the const overload)
                if addr && args.first().is_some_and(|a| const_object(a, env)) {
                    continue;
                }
                // `&v.items[v.count]` is the end pointer (`end()`, `data() + size()`), not an
                // element
                if addr && args.len() == 2 {
                    let (ob, oo) = crate::addr::canon_ptr(&args[0], env);
                    if let Some((ib, io)) = crate::addr::access(res(&args[1], env.defs), env) {
                        if teq(&ib, &ob, env.defs) && io >= oo {
                            continue;
                        }
                    }
                }
                // the class's own members use its fields directly
                if args.first().is_some_and(|a| matches!(res(a, env.defs), Expr::Var(v) if env.vars[*v].kind == VarKind::This)) {
                    continue;
                }
                let mut call = make_call(t, args);
                // the referenced object itself (an lvalue of the element type)
                if let (Expr::Call { ret, .. }, Type::Ref(inner)) = (&mut call, strip(&t.sig.ret)) {
                    *ret = (**inner).clone();
                }
                // a scalar read of a class element is its first member (`ids[i].value`)
                if let (false, Expr::Index { ty: ity, .. }, Type::Ref(inner)) = (addr, &*e, strip(&t.sig.ret)) {
                    if class_name(inner, env.db).is_some() && class_name(ity, env.db).is_none() {
                        call = Expr::Member { base: Box::new(call), offset: 0, ty: ity.clone() };
                    }
                }
                best = Some((if addr { Expr::AddrOf(Box::new(call)) } else { call }, MIN_SCORE));
                break;
            }
        }
        if let Some((call, _)) = best.filter(|(_, sc)| *sc >= MIN_SCORE) {
            *e = call;
            // operands may contain further occurrences
            let mut n = 1;
            let e = match e {
                Expr::AddrOf(x) => &mut **x,
                e => e,
            };
            if let Expr::Call { args, callee, .. } = e {
                for a in args.iter_mut() {
                    n += scalar_expr(a, env, idx);
                }
                if let Callee::Method { this, .. } = callee {
                    n += scalar_expr(this, env, idx);
                }
            }
            return n;
        }
    }
    let mut n = 0;
    match e {
        // `&v[i]` was tried as a whole above (an element address); its Index alone would give
        // `&call` with the wrong constness
        Expr::AddrOf(x) if matches!(&**x, Expr::Index { .. }) => {
            if let Expr::Index { base, index, .. } = &mut **x {
                n += scalar_expr(base, env, idx);
                n += scalar_expr(index, env, idx);
            }
        }
        Expr::AddrOf(x) | Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } => n += scalar_expr(x, env, idx),
        Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => n += scalar_expr(base, env, idx),
        Expr::Index { base, index, .. } => {
            n += scalar_expr(base, env, idx);
            n += scalar_expr(index, env, idx);
        }
        Expr::Binary { l, r, .. } => {
            n += scalar_expr(l, env, idx);
            n += scalar_expr(r, env, idx);
        }
        Expr::Ternary { c, t, f, .. } => {
            n += scalar_expr(c, env, idx);
            n += scalar_expr(t, env, idx);
            n += scalar_expr(f, env, idx);
        }
        Expr::Call { callee, args, .. } => {
            match callee {
                Callee::Method { this, .. } | Callee::Virtual { this, .. } => n += scalar_expr(this, env, idx),
                Callee::Indirect(x) => n += scalar_expr(x, env, idx),
                _ => {}
            }
            for a in args {
                n += scalar_expr(a, env, idx);
            }
        }
        Expr::Construct { args, .. } | Expr::New { args, .. } => {
            for a in args {
                n += scalar_expr(a, env, idx);
            }
        }
        _ => {}
    }
    n
}

/// Remove definitions of local temps that are no longer read.
pub fn dce(body: &mut Vec<Stmt>, vars: &[Var]) {
    loop {
        let mut uses: HashMap<VarId, usize> = HashMap::new();
        Stmt::walk_exprs(body, &mut |e| {
            if let Expr::Var(v) = e {
                *uses.entry(*v).or_default() += 1;
            }
        });
        // assignment destinations are not reads
        let mut dsts: HashMap<VarId, usize> = HashMap::new();
        count_defs(body, &mut dsts);
        let mut changed = false;
        Stmt::for_each_block_mut(body, &mut |b| {
            b.retain(|s| match s {
                Stmt::Assign { dst: Expr::Var(v), src } if matches!(vars[*v].kind, VarKind::Local) && !src.has_call() => {
                    let reads = uses.get(v).copied().unwrap_or(0) - dsts.get(v).copied().unwrap_or(0).min(uses.get(v).copied().unwrap_or(0));
                    if reads == 0 {
                        changed = true;
                        false
                    } else {
                        true
                    }
                }
                _ => true,
            })
        });
        if !changed {
            break;
        }
    }
}

/// Decompose an address into scaled terms and a constant (false when it isn't a sum).
/// Target temps are looked through (`defs`) when their definition is itself a sum.
fn linear(e: &Expr, scale: i64, terms: &mut Vec<(Expr, i64)>, c: &mut i64, defs: Option<&Defs>, depth: u32) -> bool {
    if depth > 12 || terms.len() > 6 {
        return false;
    }
    match e {
        Expr::Int { value, .. } => {
            *c += value * scale;
            true
        }
        Expr::Binary { op: BinOp::Add, l, r, .. } => linear(l, scale, terms, c, defs, depth + 1) && linear(r, scale, terms, c, defs, depth + 1),
        Expr::Binary { op: BinOp::Sub, l, r, .. } if matches!(**r, Expr::Int { .. }) => linear(l, scale, terms, c, defs, depth + 1) && linear(r, -scale, terms, c, defs, depth + 1),
        // constant scaling folds into the term's scale (`i << 1`, `i * 12`)
        Expr::Binary { op: BinOp::Shl, l, r, .. } if matches!(**r, Expr::Int { value: 0..=8, .. }) => {
            let Expr::Int { value: k, .. } = **r else { return false };
            linear(l, scale << k, terms, c, defs, depth + 1)
        }
        Expr::Binary { op: BinOp::Mul, l, r, .. } if matches!(**r, Expr::Int { .. }) => {
            let Expr::Int { value: k, .. } = **r else { return false };
            linear(l, scale * k, terms, c, defs, depth + 1)
        }
        Expr::Cast { ty, e: inner } if matches!(strip(ty), Type::Ptr(_) | Type::Int { size: 4, .. }) => linear(inner, scale, terms, c, defs, depth + 1),
        Expr::AddrOf(x) => match &**x {
            Expr::Load { base, offset, .. } => {
                *c += *offset as i64 * scale;
                linear(base, scale, terms, c, defs, depth + 1)
            }
            Expr::Index { base, index, ty } => {
                let Some(sz) = mwdec_lift::scalar_size(ty) else { return false };
                linear(base, scale, terms, c, defs, depth + 1) && linear_index(index, scale * sz as i64, terms, c, depth)
            }
            _ => {
                terms.push((e.clone(), scale));
                true
            }
        },
        Expr::Var(_) => {
            if let Some(d) = defs {
                let r = res(e, d);
                if r != e && matches!(r, Expr::Binary { op: BinOp::Add | BinOp::Sub, .. } | Expr::Cast { .. }) {
                    return linear(r, scale, terms, c, defs, depth + 1);
                }
            }
            terms.push((e.clone(), scale));
            true
        }
        _ => {
            terms.push((e.clone(), scale));
            true
        }
    }
}

/// An index term: a constant shift or multiplier folds into the scale.
fn linear_index(e: &Expr, scale: i64, terms: &mut Vec<(Expr, i64)>, c: &mut i64, _depth: u32) -> bool {
    match e {
        Expr::Int { value, .. } => *c += value * scale,
        Expr::Binary { op: BinOp::Shl, l, r, .. } if matches!(**r, Expr::Int { value: 0..=8, .. }) => {
            let Expr::Int { value: k, .. } = **r else { return false };
            return linear_index(l, scale << k, terms, c, _depth);
        }
        Expr::Binary { op: BinOp::Mul, l, r, .. } if matches!(**r, Expr::Int { .. }) => {
            let Expr::Int { value: k, .. } = **r else { return false };
            return linear_index(l, scale * k, terms, c, _depth);
        }
        _ => terms.push((e.clone(), scale)),
    }
    true
}

fn ok_lvalue(e: &Expr) -> bool {
    matches!(e, Expr::Load { .. } | Expr::Member { .. } | Expr::Global { .. })
}

/// The expansion of a folded scalar inline call (`v.size()` -> `v.mCount`), from its template,
/// when every hole is a plain value or object pointer.
fn unfold(e: &Expr, env: &Env) -> Option<Expr> {
    let Expr::Call { callee, args, .. } = e else { return None };
    let (sig, this) = match callee {
        Callee::Method { sig, this, .. } if sig.mangled.is_none() => (sig, Some(&**this)),
        Callee::Direct { sig, .. } if sig.mangled.is_none() => (sig, None),
        _ => return None,
    };
    let t = env.lib.templates.iter().find(|t| t.name == sig.qualified_name && matches!(t.shape, Shape::Scalar(_)) && !t.ret_ref && t.holes.len() == args.len() + this.is_some() as usize)?;
    let Shape::Scalar(p) = &t.shape else { return None };
    let vals: Vec<&Expr> = this.into_iter().chain(args.iter()).collect();
    for (h, k) in t.holes.iter().enumerate() {
        match k {
            HoleKind::Scalar(_) | HoleKind::Obj { ptr: true, .. } => {}
            _ => return None,
        }
        let _ = h;
    }
    let mut out = p.clone();
    let mut ok = true;
    out.rewrite(&mut |x| {
        if let Expr::Var(h) = x {
            match vals.get(*h) {
                Some(v) => *x = (*v).clone(),
                None => ok = false,
            }
        }
    });
    ok.then_some(out)
}
