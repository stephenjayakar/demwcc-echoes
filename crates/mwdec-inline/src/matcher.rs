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
        // (a bounded amount of matching per function: backtracking over commutative operands
        // and nested candidates can otherwise grow without limit)
        if !step() {
            return false;
        }
        let defs = self.defs();
        // a value an earlier fold already named (`size()`): matched in its expanded form
        if !matches!(p, Expr::Var(_) | Expr::Call { .. }) {
            if let Some(u) = unfold(res(t, defs), self.env) {
                return self.m(p, &u);
            }
        }
        // a call the pattern makes that an earlier fold named as a forwarding inline
        // (`ldexp(x, n)` folded as `scalbn(x, n)` before `ldexpf`'s `(float)ldexp(..)` is tried):
        // inline expansion is transparent, so the pattern call matches the fold's expansion
        if let (Expr::Call { callee: pc, .. }, Expr::Call { callee: tc, .. }) = (p, res(t, defs)) {
            let name = |c: &Callee| match c {
                Callee::Direct { sig, .. } | Callee::Method { sig, .. } => Some(sig.qualified_name.clone()),
                _ => None,
            };
            let folded = matches!(tc, Callee::Direct { sig, .. } | Callee::Method { sig, .. } if sig.mangled.is_none());
            if folded && name(pc) != name(tc) && std::env::var("MWDI_UNFOLD_CALL").is_ok() {
                if let Some(u) = unfold(res(t, defs), self.env) {
                    if matches!(u, Expr::Call { .. }) {
                        let snap = self.b.clone();
                        if self.m(p, &u) {
                            return true;
                        }
                        self.b = snap;
                    }
                }
            }
        }
        // the value of a scalar reference parameter: bound to the lvalue read
        if let Expr::Load { base, offset: 0, .. } | Expr::Member { base, offset: 0, .. } = p {
            if let Expr::Var(h) = &**base {
                if matches!(self.t.holes.get(*h), Some(HoleKind::ScalarRef(_))) {
                    let rt = res(t, defs);
                    // (or a variable: a parameter or local passed by reference)
                    let var = matches!(t, Expr::Var(v) if matches!(self.env.vars[*v].kind, VarKind::Param { .. } | VarKind::Local | VarKind::Stack { .. }));
                    if !matches!(rt, Expr::Load { .. } | Expr::Member { .. } | Expr::Global { .. }) && !var {
                        return false;
                    }
                    let rt = if matches!(t, Expr::Var(_)) && !ok_lvalue(rt) { t } else { rt };
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
                // (an object starting before the address read: a same-shaped inline of another
                // member, `GetAlpha()` at `p - 3` for `GetRed()` at `p`)
                if to < k {
                    return false;
                }
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
                // a float parameter can't take a double value (only a double literal the compiler
                // widened from a float one)
                if matches!(strip(ty), Type::Float { size: 4 }) && matches!(strip(&ty_of(rt, self.env.vars)), Type::Float { size: 8 }) && std::env::var("MWDI_NO_F64CHECK").is_err() {
                    let narrow_lit = matches!(rt, Expr::Float { bits, double: true } if (f64::from_bits(*bits) as f32) as f64 == f64::from_bits(*bits));
                    if !narrow_lit {
                        return false;
                    }
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
            // a fast-cast helper call in the pattern (the probe's own expansion, named by the
            // lifter) against the target's opened conversion (see `apply`)
            if let (Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. }, Expr::Cast { ty: Type::Volatile(_), e: inner }) = (p, t) {
                if args.len() == 1 && FAST_CASTS.iter().any(|(_, s)| s == symbol) {
                    return self.m(&args[0], inner);
                }
                if args.len() == 1 && FAST_LOADS.iter().any(|(_, s)| s == symbol) {
                    return self.m(&args[0], &Expr::AddrOf(inner.clone()));
                }
            }
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
            // a constant argument the compiler folded into the expansion (`Lerp(a, b, 3.14f)`:
            // `(1.0f - t)` became `-2.1415927f`): solve the pattern for its free hole
            if matches!(t, Expr::Float { .. } | Expr::Int { .. }) && matches!(p, Expr::Binary { .. } | Expr::Unary { .. } | Expr::Cast { .. }) && std::env::var("MWDI_NO_SOLVE").is_err() {
                if let Some(c) = Num::of(t) {
                    let snap = self.b.clone();
                    if std::env::var("MWDI_TRACE_SOLVE").is_ok() {
                        eprintln!("SOLVE {} {p:?} = {c:?}", self.t.name);
                    }
                    if self.solve(p, c, 0) {
                        return true;
                    }
                    self.b = snap;
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
                (Expr::Cast { ty, e }, Expr::Cast { ty: ty2, e: e2 }) => is_opened_cast(p) == is_opened_cast(t) && vclass(ty, self.env.db) == vclass(ty2, self.env.db) && self.m(e, e2),
                (Expr::Int { value, .. }, Expr::Int { value: v2, .. }) => value == v2,
                (Expr::Float { bits, double }, Expr::Float { bits: b2, double: d2 }) => bits == b2 && double == d2,
                (Expr::Global { symbol, .. }, Expr::Global { symbol: s2, .. }) => symbol == s2 || symbol.strip_prefix(crate::template::LITERAL_PREFIX).is_some_and(|h| literal_hex(s2).as_deref() == Some(h)),
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
                // a ternary decided by constant arguments: the compiler kept one arm (constfold.rs)
                (Expr::Ternary { c, t: a, f, .. }, _) if std::env::var("MWDI_NO_CONSTIF").is_err() && crate::constfold::decidable(c, &self.t.holes) => {
                    let snap = self.b.clone();
                    if self.m(a, t) && crate::constfold::solve(c, true, self) {
                        return true;
                    }
                    self.b = snap.clone();
                    if self.m(f, t) && crate::constfold::solve(c, false, self) {
                        return true;
                    }
                    self.b = snap;
                    false
                }
                _ => false,
            }
        }
    }

    /// The constant value of pattern `p` under the current bindings (literals and holes bound to
    /// literals, through + - * / and negation), if it has one.
    fn const_val(&self, p: &Expr, depth: u32) -> Option<Num> {
        if depth > 8 {
            return None;
        }
        match p {
            Expr::Float { .. } | Expr::Int { .. } => Num::of(p),
            Expr::Var(h) => match (self.t.holes.get(*h), &self.b[*h]) {
                (Some(HoleKind::Scalar(_)), Some(Bind::Val(v))) => Num::of(res(v, self.defs())),
                _ => None,
            },
            Expr::Unary { op: mwdec_lift::UnOp::Neg, e, .. } => self.const_val(e, depth + 1).map(|v| v.neg()),
            Expr::Unary { op: mwdec_lift::UnOp::BitNot, e, ty } => match self.const_val(e, depth + 1)? {
                Num::I(v, s, g) => Num::I(!v, s, g).cast(ty).or(Some(Num::I(!v, s, g))),
                _ => None,
            },
            Expr::Cast { ty, e } => self.const_val(e, depth + 1).and_then(|v| v.cast(ty)),
            Expr::Binary { op, l, r, ty } => Num::bin(*op, self.const_val(l, depth + 1)?, self.const_val(r, depth + 1)?, ty),
            _ => None,
        }
    }

    /// Make pattern `p` (arithmetic over scalar holes and literals) equal the constant `c`,
    /// binding its one free scalar hole. The value found is checked by evaluating `p` again
    /// (the compiler folded the constant expression with the same rounding).
    fn solve(&mut self, p: &Expr, c: Num, depth: u32) -> bool {
        if depth > 8 {
            return false;
        }
        if let Some(v) = self.const_val(p, 0) {
            return v.same(&c);
        }
        let ok = match p {
            Expr::Var(h) => match self.t.holes.get(*h) {
                Some(HoleKind::Scalar(ty)) if self.b[*h].is_none() => match c.cast(ty) {
                    Some(v) if v.cast(&c.ty()).is_some_and(|w| w.same(&c)) => {
                        self.b[*h] = Some(Bind::Val(v.expr()));
                        true
                    }
                    _ => false,
                },
                _ => false,
            },
            Expr::Unary { op: mwdec_lift::UnOp::Neg, e, .. } => self.solve(e, c.neg(), depth + 1),
            Expr::Cast { ty, e } => {
                // (an int or narrower value converted: solve in the operand's type)
                let inner_ty = self.pat_ty(e);
                match inner_ty.and_then(|it| c.cast(&it)) {
                    Some(v) if v.cast(ty).is_some_and(|w| w.same(&c)) => self.solve(e, v, depth + 1),
                    _ => false,
                }
            }
            Expr::Binary { op, l, r, ty } => {
                let (kl, kr) = (self.const_val(l, 0), self.const_val(r, 0));
                let inv = |op: BinOp, k: Num, left_known: bool| -> Option<Num> {
                    match (op, left_known) {
                        (BinOp::Add, _) => Num::bin(BinOp::Sub, c, k, ty),
                        (BinOp::Sub, false) => Num::bin(BinOp::Add, c, k, ty),
                        (BinOp::Sub, true) => Num::bin(BinOp::Sub, k, c, ty),
                        (BinOp::Mul, _) if !k.is_zero() => Num::bin(BinOp::Div, c, k, ty),
                        (BinOp::Div, false) => Num::bin(BinOp::Mul, c, k, ty),
                        _ => None,
                    }
                };
                match (kl, kr) {
                    (Some(k), None) => inv(*op, k, true).is_some_and(|v| self.solve(r, v, depth + 1)),
                    (None, Some(k)) => inv(*op, k, false).is_some_and(|v| self.solve(l, v, depth + 1)),
                    _ => false,
                }
            }
            _ => false,
        };
        // the solution must reproduce the folded constant exactly
        ok && self.const_val(p, 0).is_some_and(|v| v.same(&c))
    }

    /// Type of a pattern expression over holes (scalar holes and literals only).
    fn pat_ty(&self, p: &Expr) -> Option<Type> {
        match p {
            Expr::Var(h) => match self.t.holes.get(*h) {
                Some(HoleKind::Scalar(t)) => Some(t.clone()),
                _ => None,
            },
            Expr::Float { double, .. } => Some(Type::Float { size: if *double { 8 } else { 4 } }),
            Expr::Int { ty, .. } | Expr::Cast { ty, .. } | Expr::Binary { ty, .. } => Some(ty.clone()),
            Expr::Unary { e, .. } => self.pat_ty(e),
            _ => None,
        }
    }

    /// Match two addresses as sums of terms plus a constant: one unbound pointer hole of the
    /// pattern takes the matching target pointer term plus the constant difference; the other
    /// terms pair up with equal scales.
    fn m_linear(&mut self, pb: &Expr, po: i64, tb: &Expr, to: i64) -> bool {
        // (an address an earlier fold named, `v.end()`: its expansion)
        let unfolded = unfold(res(tb, self.defs()), self.env);
        let tb = unfolded.as_ref().unwrap_or(tb);
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
            // a constant-argument specialisation: the call passes the constant
            if let Some((_, c)) = self.t.fixed.iter().find(|(fh, _)| *fh == h) {
                out.push(c.clone());
                continue;
            }
            // a scalar parameter the expansion never reads (an overload's tag enum, `SetColumn(kDZ,
            // v)`): any value compiles the same; the enum's first value / zero
            if self.b[h].is_none() && std::env::var("MWDI_NO_UNUSED_ARG").is_err() {
                if let HoleKind::Scalar(ty) = k {
                    let rt = mwdec_lift::types::resolve(Some(self.env.db), strip(ty)).into_owned();
                    let zero = if mwdec_lift::types::is_enum(Some(self.env.db), &rt) {
                        Some(Expr::Cast { ty: strip(ty).clone(), e: Box::new(Expr::Int { value: 0, ty: Type::Int { size: 4, signed: true } }) })
                    } else {
                        // (only tags: a constructor's unread scalar would explain an object by
                        // zeros, `CAABox(x, y, 0, 0, 0, z).GetPointE()`)
                        None
                    };
                    if let Some(z) = zero {
                        out.push(z);
                        continue;
                    }
                }
            }
            let b = self.b[h].as_ref()?;
            match (k, b) {
                // a float parameter given a double literal the compiler widened at compile time
                // (`fmodf(3.1415927f, ..)` -> `fmod(3.1415927410125732, ..)`): the float literal
                (HoleKind::Scalar(ty), Bind::Val(Expr::Float { bits, double: true })) if std::env::var("MWDI_NO_SOLVE").is_err() && matches!(strip(ty), Type::Float { size: 4 }) && (f64::from_bits(*bits) as f32) as f64 == f64::from_bits(*bits) => {
                    out.push(Expr::Float { bits: (f64::from_bits(*bits) as f32).to_bits() as u64, double: false })
                }
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
                            let a = if !typed_ptr_to(e, class, self.env) { a } else { e.clone() };
                            // (a reference local is the object: its address is `&r`)
                            let a = match a {
                                Expr::Var(v) if matches!(self.env.vars[v].ty, Type::Ref(_)) => Expr::AddrOf(Box::new(Expr::Var(v))),
                                a => a,
                            };
                            out.push(a);
                        }
                        None => {
                            if typed_ptr_to(e, class, self.env) {
                                out.push(e.clone());
                            } else if !*ptr && matches!(e, Expr::Var(_)) && class_name(&ty_of(e, self.env.vars), self.env.db).is_some_and(|c| is_base_or_same(self.env.db, class, &c)) {
                                // a variable holding the object itself (a parameter the lifter
                                // keeps as a value)
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
                        // (a one-member object built around a scalar, `sine(CRelAngle::FromRadians(x))`
                        // for `sinf(x)`, explains nothing more than the scalar: no bonus, a tie-break
                        // penalty)
                        if m.len() == 1 && std::env::var("MWDI_NO_SINGLE_EXPLAIN_PENALTY").is_err() {
                            // (keep its penalties: a constructor fallback stays a last resort)
                            score += sc.min(0) - 1;
                        } else {
                            score += sc;
                        }
                        // pointer holes (`this` of a const method) take the temporary's address
                        out.push(if *ptr { Expr::AddrOf(Box::new(v)) } else { v });
                    } else if *temp_ok && !*ptr && depth < 3 && whole_object(self.env, class, m) {
                        // a statement inline's argument built in place (`push_back(pair(a, b))`):
                        // only as its class's constructor
                        let (v, sc) = explain_object(self.env, class, m, depth + 1)?;
                        if !matches!(v, Expr::Construct { .. }) {
                            return None;
                        }
                        score += sc;
                        out.push(v);
                    } else {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        // an operand the source named (a reference local bound to an accessor's result): the
        // value is an object expression, not member-wise arithmetic
        if out.iter().any(|a| matches!(a, Expr::AddrOf(x) if matches!(&**x, Expr::Var(v) if matches!(self.env.vars[*v].ty, Type::Ref(_)) && matches!(self.env.vars[*v].kind, VarKind::Stack { .. })))) {
            score += CTOR_PENALTY;
        }
        Some((out, score))
    }
}

/// A literal value with its type, for folding constant arguments.
#[derive(Clone, Copy, Debug)]
pub enum Num {
    F32(f32),
    F64(f64),
    I(i64, u8, bool),
}

impl Num {
    fn of(e: &Expr) -> Option<Num> {
        match e {
            Expr::Float { bits, double: false } => Some(Num::F32(f32::from_bits(*bits as u32))),
            Expr::Float { bits, double: true } => Some(Num::F64(f64::from_bits(*bits))),
            Expr::Int { value, ty } => match strip(ty) {
                Type::Int { size, signed } => Some(Num::I(*value, *size, *signed)),
                Type::Bool | Type::Char => Some(Num::I(*value, 1, false)),
                _ => Some(Num::I(*value, 4, true)),
            },
            _ => None,
        }
    }
    fn ty(&self) -> Type {
        match self {
            Num::F32(_) => Type::Float { size: 4 },
            Num::F64(_) => Type::Float { size: 8 },
            Num::I(_, size, signed) => Type::Int { size: *size, signed: *signed },
        }
    }
    fn expr(&self) -> Expr {
        match self {
            Num::F32(v) => Expr::Float { bits: v.to_bits() as u64, double: false },
            Num::F64(v) => Expr::Float { bits: v.to_bits(), double: true },
            Num::I(v, _, _) => Expr::Int { value: *v, ty: self.ty() },
        }
    }
    fn same(&self, o: &Num) -> bool {
        match (self, o) {
            (Num::F32(a), Num::F32(b)) => a.to_bits() == b.to_bits(),
            (Num::F64(a), Num::F64(b)) => a.to_bits() == b.to_bits(),
            (Num::F32(a), Num::F64(b)) | (Num::F64(b), Num::F32(a)) => (*a as f64).to_bits() == b.to_bits(),
            (Num::I(a, ..), Num::I(b, ..)) => a == b,
            _ => false,
        }
    }
    fn is_zero(&self) -> bool {
        match self {
            Num::F32(v) => *v == 0.0,
            Num::F64(v) => *v == 0.0,
            Num::I(v, ..) => *v == 0,
        }
    }
    fn neg(&self) -> Num {
        match self {
            Num::F32(v) => Num::F32(-v),
            Num::F64(v) => Num::F64(-v),
            Num::I(v, s, g) => Num::I(-v, *s, *g),
        }
    }
    fn cast(&self, ty: &Type) -> Option<Num> {
        Some(match (strip(ty), self) {
            (Type::Float { size: 4 }, Num::F32(v)) => Num::F32(*v),
            (Type::Float { size: 4 }, Num::F64(v)) => Num::F32(*v as f32),
            (Type::Float { size: 4 }, Num::I(v, ..)) => Num::F32(*v as f32),
            (Type::Float { size: 8 }, Num::F32(v)) => Num::F64(*v as f64),
            (Type::Float { size: 8 }, Num::F64(v)) => Num::F64(*v),
            (Type::Float { size: 8 }, Num::I(v, ..)) => Num::F64(*v as f64),
            (Type::Int { size, signed }, n) => {
                let x: i64 = match n {
                    Num::F32(v) if v.fract() == 0.0 && v.abs() < 2e9 => *v as i64,
                    Num::F64(v) if v.fract() == 0.0 && v.abs() < 2e9 => *v as i64,
                    Num::I(v, ..) => *v,
                    _ => return None,
                };
                let bits = (*size as u32) * 8;
                let m = if bits >= 64 { x } else if *signed { (x << (64 - bits)) >> (64 - bits) } else { x & ((1i64 << bits) - 1) };
                Num::I(m, *size, *signed)
            }
            _ => return None,
        })
    }
    fn bin(op: BinOp, a: Num, b: Num, ty: &Type) -> Option<Num> {
        let a = a.cast(ty)?;
        let b = b.cast(ty)?;
        Some(match (a, b) {
            (Num::F32(x), Num::F32(y)) => Num::F32(match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div if y != 0.0 => x / y,
                _ => return None,
            }),
            (Num::F64(x), Num::F64(y)) => Num::F64(match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div if y != 0.0 => x / y,
                _ => return None,
            }),
            (Num::I(x, s, g), Num::I(y, ..)) => {
                let v = match op {
                    BinOp::Add => x.wrapping_add(y),
                    BinOp::Sub => x.wrapping_sub(y),
                    BinOp::Mul => x.wrapping_mul(y),
                    BinOp::Shl if (0..32).contains(&y) => x << y,
                    BinOp::And => x & y,
                    BinOp::Or => x | y,
                    BinOp::Xor => x ^ y,
                    _ => return None,
                };
                Num::I(v, s, g).cast(&Type::Int { size: s, signed: g })?
            }
            _ => return None,
        })
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
    ACCESSORS.with(|a| *a.borrow_mut() = (0, std::rc::Rc::new(vec![])));
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

/// Does `m` set every flat field of `cls`?
fn whole_object(env: &Env, cls: &str, m: &BTreeMap<i32, Expr>) -> bool {
    crate::template::flat_fields(env.db, cls).is_some_and(|f| f.len() == m.len() && f.iter().all(|(o, _)| m.contains_key(o)))
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
    if std::env::var("MWDI_TRACE_SOLVE").is_ok() {
        eprintln!("EXPLAIN {cls} {:?} {m:?}", env.objects.get(&mwdec_lift::sig::norm_name(cls)).map(|l| l.iter().map(|i| env.lib.templates[*i].name.clone()).collect::<Vec<_>>()));
    }
    let list = env.objects.get(&mwdec_lift::sig::norm_name(cls))?;
    let mut best: Option<(Expr, i32)> = None;
    // one value in every component, computed once (`CVector3f(r, r, r)`): the member-wise
    // constructor, not an operator that would compute it per component
    let splat = m.len() > 1 && m.values().all(|v| teq(v, m.values().next().unwrap(), env.defs)) && !matches!(m.values().next(), Some(Expr::Int { .. } | Expr::Float { .. }));
    for &ti in list {
        let t = &env.lib.templates[ti];
        let Shape::Object { comps, .. } = &t.shape else { continue };
        if splat && !matches!(t.kind, CallKind::Ctor) {
            continue;
        }
        // every bound component must be produced by the template
        if !m.keys().all(|o| comps.iter().any(|c| c.off == *o)) {
            continue;
        }
        let mut mm = M::new(env, t);
        let ok = comps.iter().filter(|c| m.contains_key(&c.off)).all(|c| mm.m(&c.pat, &m[&c.off]));
        if !ok {
            continue;
        }
        // a template that hands one object hole's components back unchanged explains nothing
        // (`DepthCompareUpdate(false, false)` specialised to a copy): its nested explanation
        // would be the same object again
        if std::env::var("MWDI_NO_IDENTITY_GUARD").is_err() && mm.b.iter().any(|b| matches!(b, Some(Bind::Comps(c)) if c.len() == m.len() && c.iter().all(|(o, v)| m.get(o).is_some_and(|w| teq(v, w, env.defs))))) {
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
                if n.get(v) == Some(&1) && matches!(vars[*v].kind, VarKind::Local) && only_inline_calls(src) && !src.uses_var(*v) {
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
    /// member accessors returning a member's address or a reference to it (`T& GetM() { return
    /// m; }`, `T* GetM() { return &m; }`): (template, member offset)
    pub member_refs: Vec<(usize, i32)>,
    /// member accessors returning a member's value (`T GetM() const { return m; }`, also through
    /// nested trivial inlines: `int GetNum() const { return x18_list.size(); }`): (template,
    /// member offset, value type)
    pub member_reads: Vec<(usize, i32, Type)>,
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
    let mut member_refs = vec![];
    let mut member_reads = vec![];
    for (i, t) in lib.templates.iter().enumerate() {
        let op_name = mwdec_lift::sig::split_scope(&t.name).1.starts_with("operator");
        if let (Shape::Scalar(p), CallKind::Method, 1, false) = (&t.shape, &t.kind, t.holes.len(), op_name) {
            // (a by-value class return is matched with its dead copy stores elsewhere)
            if matches!(p, Expr::AddrOf(_)) && t.dead.is_empty() && matches!(strip(&t.sig.ret), Type::Ref(_) | Type::Ptr(_)) {
                if let Some((0, k)) = pat_canon(p, &t.holes) {
                    member_refs.push((i, k));
                }
            }
        }
        // the same with tag parameters the expansion never reads (`GetRow(EDimY)`), or the
        // object itself reinterpreted (`GetRow(EDimX)`: `*reinterpret_cast<const CVector3f*>(&m00)`)
        if let (Shape::Scalar(p), CallKind::Method, false) = (&t.shape, &t.kind, op_name) {
            let tags = t.holes.len() > 1 && t.holes[1..].iter().enumerate().all(|(k, h)| matches!(h, HoleKind::Scalar(_)) && !p.uses_var(k + 1));
            let reinterp = matches!(p, Expr::Var(0)) && ref_pointee_class(t).is_some_and(|c| t.class.as_deref().is_some_and(|tc| mwdec_lift::sig::norm_name(tc) != mwdec_lift::sig::norm_name(&c)));
            if (tags || (t.holes.len() == 1 && reinterp)) && (matches!(p, Expr::AddrOf(_)) || reinterp) && t.dead.is_empty() && std::env::var_os("MWDI_TAGGED_REFS").is_some() {
                if let Some((0, k)) = pat_canon(p, &t.holes) {
                    member_refs.push((i, k));
                }
            }
        }
        if let (Shape::Scalar(p), CallKind::Method, 1, false) = (&t.shape, &t.kind, t.holes.len(), op_name) {
            // (a scalar result: by-value class returns are object templates)
            let mut scalar_ret = !matches!(strip(&t.sig.ret), Type::Named(_)) && mwdec_lift::scalar_size(strip(&t.sig.ret)).is_some();
            // (through a conversion to the declared return type: `EType GetType() const { return
            // static_cast<EType>(mType); }`)
            let p = match p {
                Expr::Cast { e, ty } if std::env::var("MWDI_NO_CAST_READS").is_err() && strip(ty) == strip(&t.sig.ret) && matches!(&**e, Expr::Load { .. } | Expr::Member { .. }) => {
                    scalar_ret = true;
                    &**e
                }
                p => p,
            };
            // (an enum result too: checked against the TypeDb where the template is used)
            let named_ret = matches!(strip(&t.sig.ret), Type::Named(_)) && std::env::var("MWDI_NO_CAST_READS").is_err();
            if !t.ret_ref && t.dead.is_empty() && (scalar_ret || named_ret) {
                if let Some((0, k, ty)) = comp_of(p, &t.holes) {
                    member_reads.push((i, k, ty.clone()));
                }
            }
        }
        match &t.shape {
            Shape::Scalar(Expr::AddrOf(inner)) if t.ret_ref && matches!(&**inner, Expr::Index { .. }) => {
                refs.push(i);
            }
            Shape::Scalar(p) => {
                // (a plain read only with the dead stores of a by-value class to tell it apart)
                // (a converted call result, `(float)atan2((double)y, (double)x)`, is specific: the
                // library forwarders of math.h; an element read `(&mX)[i]` too)
                let conv_call = matches!(p, Expr::Cast { e, .. } if matches!(&**e, Expr::Call { .. }) || (t.ops >= 1 && !matches!(&**e, Expr::Var(_) | Expr::Load { .. } | Expr::Member { .. } | Expr::Cast { .. }))) && std::env::var("MWDI_NO_CONVCALL").is_err();
                let elem = matches!(p, Expr::Index { .. }) && std::env::var("MWDI_NO_ELEMREAD").is_err();
                if (t.ops >= 1 && !matches!(p, Expr::Var(_) | Expr::Load { .. } | Expr::Member { .. } | Expr::AddrOf(_) | Expr::Cast { .. })) || !t.dead.is_empty() || conv_call || elem {
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
                // (plain member stores only for members the caller may not name: see
                // `groups::try_segment`)
                if !comps.is_empty() && (t.ops >= 1 || !t.dead.is_empty() || comps.len() >= 2) {
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
    Index { stmts, refs, scalars, cflow, objects, groups, member_refs, member_reads }
}

/// Pattern-match steps one function may take ([`M::m`] calls; train functions take at most ~10^5);
/// past it every match fails.
const MAX_STEPS: u64 = 5_000_000;

thread_local! {
    static STEPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Count one matching step; false once the function's budget is spent.
fn step() -> bool {
    STEPS.with(|s| {
        let n = s.get() + 1;
        s.set(n);
        n <= MAX_STEPS
    })
}

/// Matching steps taken for the current function.
pub fn steps() -> u64 {
    STEPS.with(|s| s.get())
}

pub fn apply(ir: &mut IrFunction, lib: &InlineLib, db: &TypeDb) -> usize {
    STEPS.with(|s| s.set(0));
    OWN.with(|o| *o.borrow_mut() = (ir.sig.this_class.clone(), ir.sig.qualified_name.clone()));
    // (per function: memos keyed by a TypeDb's address must not outlive it)
    reset_memos();
    TARGET_DEAD.with(|d| *d.borrow_mut() = ir.dead_stores.iter().map(|x| (x.size, x.value.clone())).collect());
    LITERALS.with(|d| *d.borrow_mut() = ir.literal_bytes.iter().cloned().collect());
    TEMPS.with(|d| *d.borrow_mut() = ir.temp_bytes.iter().cloned().collect());
    crate::walkptr::apply(ir, lib, db);
    PURE_OK.with(|p| *p.borrow_mut() = Some(lib.effectful.clone()));
    NAMED.with(|n| n.borrow_mut().clear());
    // the lifter names the fast-cast helpers' expansions (`psq_st` / `psq_l` through the
    // quantization registers) as their calls (`CCast::ToUint8(f)`, `CCast::ToReal32(b)`); inline
    // bodies that convert through them (`CColor::WithAlphaOf`: `(mRgba & ~0xff) | ToUint8(a *
    // 255.f)`) see the conversion, so for matching they are opened into a tagged cast, and the
    // ones no fold consumed become the call again
    let fast = std::env::var("MWDI_NO_FASTCAST_OPEN").is_err();
    if fast {
        open_fast_casts(&mut ir.body);
    }
    let n = apply_inner(ir, lib, db);
    if fast {
        restore_fast_casts(ir, db);
    }
    PURE_OK.with(|p| *p.borrow_mut() = None);
    TARGET_DEAD.with(|d| d.borrow_mut().clear());
    LITERALS.with(|d| d.borrow_mut().clear());
    TEMPS.with(|d| d.borrow_mut().clear());
    n
}

/// Only folded inline calls (a template's or a trivial accessor's call, no real call): the
/// value of such a temp is pure, so it can be looked through like any expression (`t =
/// fabs(x); ... (float)t` is `fabsf(x)`) and dropped when dead.
pub fn only_inline_calls(e: &Expr) -> bool {
    if std::env::var("MWDI_NO_PURE_CALL_DEFS").is_ok() {
        return !e.has_call();
    }
    let mut ok = true;
    e.walk(&mut |x| match x {
        Expr::Call { callee: Callee::Direct { sig, symbol }, .. } => ok &= sig.mangled.is_none() && *symbol == sig.qualified_name,
        Expr::Call { callee: Callee::Method { sig, symbol, .. }, .. } => ok &= symbol.is_empty() && sig.mangled.is_none(),
        Expr::Call { .. } | Expr::New { .. } => ok = false,
        _ => {}
    });
    ok
}

thread_local! {
    /// The function being rewritten: (its class, its qualified name), for member access checks.
    static OWN: std::cell::RefCell<(Option<String>, String)> = const { std::cell::RefCell::new((None, String::new())) };
}

/// Can the function being rewritten name member `field` of `owner` (C++98 access: its own
/// class, a friend, protected members from a derived class)?
pub fn member_accessible(db: &TypeDb, owner: &str, field: &str) -> bool {
    let (own, fname) = OWN.with(|o| o.borrow().clone());
    let norm = mwdec_lift::sig::norm_name;
    if own.as_deref().is_some_and(|o| norm(o) == norm(owner)) {
        return true;
    }
    let befriended = || {
        let strip_t = |s: &str| s.split('<').next().unwrap_or(s).to_string();
        let fr = db.friends.get(owner).or_else(|| db.friends.get(&strip_t(owner)));
        let last = |s: &str| mwdec_lift::sig::split_scope(&strip_t(s)).1.to_string();
        let short = mwdec_lift::sig::split_scope(&fname).1.to_string();
        fr.is_some_and(|fr| fr.iter().any(|f| own.as_deref().is_some_and(|o| norm(o) == norm(f) || last(o) == last(f)) || (own.is_none() && (short == *f || fname == *f))))
    };
    let access = mwdec_lift::sig::find_class(db, owner).and_then(|c| c.fields.iter().find(|f| f.name == field)).map(|f| f.access.clone());
    match access {
        Some(mwdec_core::Access::Public) | None => true,
        Some(mwdec_core::Access::Protected) => befriended() || own.as_deref().is_some_and(|o| is_base_or_same_any(db, owner, o)),
        Some(mwdec_core::Access::Private) => befriended(),
    }
}

/// Class a reference/pointer-returning template returns (`const CVector3f&` -> CVector3f).
fn ref_pointee_class(t: &Template) -> Option<String> {
    match strip(&t.sig.ret) {
        Type::Ref(x) | Type::Ptr(x) => match strip(x) {
            Type::Named(n) => Some(n.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Default arguments for the holes after the object (tags the expansion never reads).
fn tag_defaults(t: &Template) -> Vec<Expr> {
    t.holes[1..]
        .iter()
        .map(|h| match h {
            HoleKind::Scalar(ty) => match strip(ty) {
                Type::Named(_) => Expr::Cast { ty: strip(ty).clone(), e: Box::new(Expr::Int { value: 0, ty: Type::Int { size: 4, signed: true } }) },
                Type::Float { size: 4 } => Expr::Float { bits: 0, double: false },
                t => Expr::Int { value: 0, ty: t.clone() },
            },
            _ => Expr::Int { value: 0, ty: Type::Int { size: 4, signed: true } },
        })
        .collect()
}

/// Can the function name the object at `delta` inside what `b` points at (every member on the
/// way accessible)? An accessor of an object it can't reach by name doesn't help.
fn object_path_ok(db: &TypeDb, b: &Expr, delta: i32, env: &Env) -> bool {
    if delta == 0 {
        return true;
    }
    let Some(outer) = crate::addr::outer_class(b, env) else { return false };
    match mwdec_lift::types::field_path(db, &outer, delta, 0) {
        Some((path, _)) => path.iter().all(|pe| !matches!(pe, mwdec_lift::types::PathElem::Field(n, owner) if !member_accessible(db, owner, n))),
        None => true,
    }
}

/// Is `derived` `base` or derived from it (at any offset)?
fn is_base_or_same_any(db: &TypeDb, base: &str, derived: &str) -> bool {
    if mwdec_lift::sig::norm_name(base) == mwdec_lift::sig::norm_name(derived) {
        return true;
    }
    let Some(c) = mwdec_lift::sig::find_class(db, derived) else { return false };
    c.bases.iter().any(|b| is_base_or_same_any(db, base, &b.name))
}

/// A trivial member accessor of the headers (`T GetM() const { return m; }`, `const T& M() const
/// { return m; }`, `T* GetM() { return &m; }`, `return a.b;`): probes skip them (the emitter
/// names them for plain reads), so they are read from the declarations.
#[derive(Clone, Debug)]
pub struct Accessor {
    pub sig: mwdec_core::FuncSig,
    pub class: String,
    /// offset of the member in `class`, its type
    pub off: i32,
    pub ty: Type,
    /// 0 value, 1 reference, 2 pointer (`return &m;`), 3 reference to a pointer member's
    /// pointee (`return *m;`: the member's value is the result's address)
    pub how: u8,
    /// (field, owner) of every member named on the way (`a`, then `b` of `a`'s class)
    pub fields: Vec<(String, String)>,
}

/// Field `name` of `class` or of one of its bases: (offset in `class`, type, owner).
fn field_named(db: &TypeDb, class: &str, name: &str, depth: u32) -> Option<(i32, Type, String)> {
    if depth > 8 {
        return None;
    }
    let c = mwdec_lift::sig::find_class(db, class)?;
    if let Some(f) = c.fields.iter().find(|f| f.name == name && f.bitfield.is_none()) {
        return Some((f.offset as i32, f.ty.clone(), c.name.clone()));
    }
    for b in &c.bases {
        if let Some((o, t, ow)) = field_named(db, &b.name, name, depth + 1) {
            return Some((o + b.offset as i32, t, ow));
        }
    }
    None
}

thread_local! {
    static ACCESSORS: std::cell::RefCell<(usize, std::rc::Rc<Vec<Accessor>>)> = std::cell::RefCell::new((0, std::rc::Rc::new(vec![])));
}

/// The trivial accessors of the context's classes (memoised per TypeDb).
pub fn accessors(db: &TypeDb) -> std::rc::Rc<Vec<Accessor>> {
    let id = db as *const TypeDb as usize ^ db.decls.len();
    if let Some(v) = ACCESSORS.with(|a| (a.borrow().0 == id).then(|| a.borrow().1.clone())) {
        return v;
    }
    let mut out = vec![];
    let ident = |s: &str| s.chars().next().map_or(false, |c| c.is_alphabetic() || c == '_') && s.chars().all(|c| c.is_alphanumeric() || c == '_');
    for (key, ds) in &db.decls {
        let qname = crate::probe::norm_op_name(key);
        let (scope, last) = mwdec_lift::sig::split_scope(&qname);
        let Some(scope) = scope else { continue };
        if scope.contains('<') || last.starts_with("operator") {
            continue;
        }
        let Some(cls) = mwdec_lift::sig::find_class(db, scope) else { continue };
        let class = cls.name.clone();
        for d in ds {
            if !d.is_inline_defined || d.is_static || d.is_virtual || !d.params.is_empty() || !d.template_params.is_empty() || d.access != mwdec_core::Access::Public {
                continue;
            }
            let Some(body) = d.inline_body.as_deref() else { continue };
            let t: Vec<&str> = body.split_whitespace().collect();
            let (amp, names): (bool, Vec<&str>) = match t.as_slice() {
                ["return", a, ";"] if ident(a) => (false, vec![a]),
                ["return", "this", "->", a, ";"] if ident(a) => (false, vec![a]),
                ["return", "&", a, ";"] if ident(a) => (true, vec![a]),
                ["return", a, ".", b, ";"] if ident(a) && ident(b) => (false, vec![a, b]),
                // a reference to what a pointer member points at (`T& GetOwner() { return *mOwner; }`)
                ["return", "*", a, ";"] if ident(a) => (false, vec![a]),
                _ => continue,
            };
            let deref = t.get(1) == Some(&"*");
            let mut off = 0;
            let mut cur = class.clone();
            let mut ty = Type::Void;
            let mut fields = vec![];
            let mut ok = true;
            for n in &names {
                match field_named(db, &cur, n, 0) {
                    Some((o, t, owner)) => {
                        off += o;
                        fields.push((n.to_string(), owner));
                        ty = t;
                        if let Some(c) = class_name(&ty, db) {
                            cur = c;
                        }
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || matches!(strip(&ty), Type::Ref(_)) || (deref && !matches!(strip(&ty), Type::Ptr(_))) {
                continue;
            }
            let how = match (strip(&d.ret), amp) {
                (Type::Ref(_), false) if deref => 3,
                (Type::Ref(_), false) => 1,
                (Type::Ptr(_), true) => 2,
                (r, false) if mwdec_lift::scalar_size(strip(&mwdec_lift::types::resolve(Some(db), r).into_owned())).is_some() || mwdec_lift::types::is_enum(Some(db), r) => 0,
                _ => continue,
            };
            let sig = mwdec_core::FuncSig {
                qualified_name: format!("{class}::{last}"),
                mangled: None,
                ret: d.ret.clone(),
                params: vec![],
                this_class: Some(class.clone()),
                is_const: d.is_const,
                is_static: false,
                is_virtual: false,
                variadic: false,
                runs_code: false,
            };
            if std::env::var("MWDI_TRACE_ACC").is_ok_and(|f| sig.qualified_name.contains(f.as_str())) {
                eprintln!("ACC {} how {how} off {off:#x} ty {ty:?} ret {:?} key {key} body {body}", sig.qualified_name, d.ret);
            }
            out.push(Accessor { sig, class: class.clone(), off, ty, how, fields });
        }
    }
    let rc = std::rc::Rc::new(out);
    ACCESSORS.with(|a| *a.borrow_mut() = (id, rc.clone()));
    rc
}

/// Final naming pass: member reads / member addresses the function can't name become the
/// class's accessor calls. Runs after every other fold, so templates that read the same members
/// (`ALAUp()` = `mAnaLeftY > 0.f ? mAnaLeftY : 0.f`) see them first.
fn accessor_stmts(b: &mut [Stmt], env: &Env, idx: &Index) -> usize {
    let mut n = 0;
    for s in b.iter_mut() {
        match s {
            Stmt::Assign { dst, src } => {
                n += accessor_expr(src, env, idx);
                match dst {
                    Expr::Load { base, .. } | Expr::Member { base, .. } => n += accessor_expr(base, env, idx),
                    Expr::Index { base, index, .. } => n += accessor_expr(base, env, idx) + accessor_expr(index, env, idx),
                    _ => {}
                }
            }
            Stmt::Expr(e) | Stmt::Return(Some(e)) => n += accessor_expr(e, env, idx),
            Stmt::If { cond, then, els } => {
                n += accessor_expr(cond, env, idx);
                n += accessor_stmts(then, env, idx) + accessor_stmts(els, env, idx);
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                n += accessor_expr(cond, env, idx);
                n += accessor_stmts(body, env, idx);
            }
            Stmt::For { init, cond, step, body } => {
                n += accessor_expr(cond, env, idx);
                n += accessor_stmts(init, env, idx) + accessor_stmts(step, env, idx) + accessor_stmts(body, env, idx);
            }
            Stmt::Switch { e, cases } => {
                n += accessor_expr(e, env, idx);
                for c in cases {
                    n += accessor_stmts(&mut c.body, env, idx);
                }
            }
            _ => {}
        }
    }
    n
}

fn accessor_expr(e: &mut Expr, env: &Env, idx: &Index) -> usize {
    // (then the accessor's own object: `bc.GetOwner().GetModelData()`)
    if std::env::var("MWDI_NO_MEMBER_REFS").is_err() {
        if let Some(call) = member_ref_call(e, env, idx) {
            if std::env::var("MWDI_TRACE_ACC").is_ok() {
                eprintln!("MREF {e:?} -> {call:?}");
            }
            return 1 + name_inside(e, call, env, idx);
        }
    }
    if std::env::var("MWDI_NO_MEMBER_READS").is_err() {
        if let Some(call) = member_read_call(e, env, idx) {
            return 1 + name_inside(e, call, env, idx);
        }
    }
    // an object lvalue the function can't name (passed by reference / copied): the accessor
    // returning a reference to it
    if let Expr::Load { ty, .. } | Expr::Member { ty, .. } = &*e {
        if class_name(ty, env.db).is_some() && std::env::var_os("MWDI_TAGGED_REFS").is_some() {
            if let Some(Expr::AddrOf(c)) = member_ref_call(&Expr::AddrOf(Box::new(e.clone())), env, idx) {
                note_named(&c);
                *e = *c;
                return 1 + accessor_expr(e, env, idx);
            }
        }
    }
    let mut n = 0;
    match e {
        // the address of a member lvalue: its base only (a value accessor's call is no lvalue)
        Expr::AddrOf(x) if matches!(&**x, Expr::Load { .. } | Expr::Member { .. }) => {
            if let Expr::Load { base, .. } | Expr::Member { base, .. } = &mut **x {
                n += accessor_expr(base, env, idx);
            }
        }
        Expr::AddrOf(x) | Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } => n += accessor_expr(x, env, idx),
        Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => n += accessor_expr(base, env, idx),
        Expr::Index { base, index, .. } => n += accessor_expr(base, env, idx) + accessor_expr(index, env, idx),
        Expr::Binary { l, r, .. } => n += accessor_expr(l, env, idx) + accessor_expr(r, env, idx),
        Expr::Ternary { c, t, f, .. } => n += accessor_expr(c, env, idx) + accessor_expr(t, env, idx) + accessor_expr(f, env, idx),
        Expr::Call { callee, args, .. } => {
            let params: Vec<Type> = match &*callee {
                Callee::Direct { sig, .. } | Callee::Method { sig, .. } => sig.params.iter().map(|p| p.ty.clone()).collect(),
                Callee::Virtual { sig: Some(sig), .. } => sig.params.iter().map(|p| p.ty.clone()).collect(),
                _ => vec![],
            };
            match callee {
                Callee::Method { this, .. } | Callee::Virtual { this, .. } => n += accessor_expr(this, env, idx),
                Callee::Indirect(x) => n += accessor_expr(x, env, idx),
                _ => {}
            }
            for (i, a) in args.iter_mut().enumerate() {
                // an object pointer passed where another class is expected (`const CVector3f&`
                // given a `CTransform4f*`): the object's accessor returning that class
                // (`xf.GetRow(kDX)` = `*reinterpret_cast<const CVector3f*>(&m00)`)
                if let Some(pt) = params.get(i) {
                    if let (Type::Ref(x) | Type::Ptr(x), true) = (strip(pt), std::env::var_os("MWDI_TAGGED_REFS").is_some()) {
                        if class_name(x, env.db).is_some() && matches!(vclass(&ty_of(a, env.vars), env.db), 2) {
                            let cast = Expr::Cast { ty: Type::Ptr(x.clone()), e: Box::new(a.clone()) };
                            if let Some(Expr::AddrOf(c)) = member_ref_call(&cast, env, idx) {
                                note_named(&c);
                                *a = Expr::AddrOf(c);
                                n += 1;
                                continue;
                            }
                        }
                    }
                }
                n += accessor_expr(a, env, idx);
            }
        }
        // (a member-wise copy of an object, `T(p->a, p->b, p->c)`, stays one: the emitter
        // spells it as the object read)
        Expr::Construct { class, args, .. } if copied_object(class, args, env) => {}
        Expr::Construct { args, .. } | Expr::New { args, .. } => {
            for a in args {
                n += accessor_expr(a, env, idx);
            }
        }
        _ => {}
    }
    n
}

/// Are `args` the flat fields of `class`, read in layout from one address?
fn copied_object(class: &Type, args: &[Expr], env: &Env) -> bool {
    let Some(cn) = class_name(class, env.db) else { return false };
    let Some(fields) = crate::template::flat_fields(env.db, &cn) else { return false };
    if fields.len() != args.len() || args.len() < 2 {
        return false;
    }
    let mut base: Option<(Expr, i32)> = None;
    for ((off, _), a) in fields.iter().zip(args) {
        let a = match a {
            Expr::Cast { e, .. } => &**e,
            a => a,
        };
        if !matches!(a, Expr::Load { .. } | Expr::Member { .. }) {
            return false;
        }
        let Some((p, o)) = crate::addr::access(res(a, env.defs), env) else { return false };
        match &base {
            None => base = Some((p, o - off)),
            Some((p0, d0)) => {
                if *d0 != o - off || !teq(p0, &p, env.defs) {
                    return false;
                }
            }
        }
    }
    true
}

/// The address of the `cls` object a member access `e` reads, spelled from `e`'s own base where
/// it can be (`t->m` keeps the temp `t`: resolving it would drop its definition's other uses),
/// else from the canonical base `b` + `off`.
fn object_as_written(e: &Expr, b: &Expr, off: i32, cls: &str, env: &Env) -> Option<Expr> {
    let canon = crate::addr::object_at(b, off, cls, env)?.0;
    let raw = match e {
        Expr::AddrOf(x) => match &**x {
            Expr::Load { base, offset, .. } => Some(((**base).clone(), *offset)),
            Expr::Member { base, offset, .. } => Some((Expr::AddrOf(base.clone()), *offset)),
            _ => None,
        },
        Expr::Load { base, offset, .. } => Some(((**base).clone(), *offset)),
        Expr::Member { base, offset, .. } => Some((Expr::AddrOf(base.clone()), *offset)),
        _ => None,
    };
    if let Some((rb, _)) = raw {
        // the raw base's canonical form: the same pointer as `b`, `co` bytes on
        let (cb, co) = crate::addr::canon_ptr(&rb, env);
        if teq(&cb, b, env.defs) && off >= co {
            if let Some((a, _)) = crate::addr::object_at(&rb, off - co, cls, env) {
                return Some(a);
            }
        }
    }
    Some(canon)
}

thread_local! {
    /// Calls the accessor naming pass made (qualified names), for measurement: a naming of a
    /// member the function can't access is not an inline recovered from its expansion.
    static NAMED: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn note_named(call: &Expr) {
    let c = match call {
        Expr::AddrOf(x) => &**x,
        c => c,
    };
    if let Expr::Call { callee: Callee::Direct { sig, .. } | Callee::Method { sig, .. }, .. } = c {
        NAMED.with(|n| n.borrow_mut().push(sig.qualified_name.clone()));
    }
}

/// The accessor-naming calls of the last [`apply`] on this thread (cleared by the call).
pub fn take_named() -> Vec<String> {
    NAMED.with(|n| std::mem::take(&mut *n.borrow_mut()))
}

/// Replace `e` by its accessor call, then name what the call's own object is reached through
/// (`bc.GetOwner().GetModelData()`): not that object's address itself, which an accessor of an
/// enclosing object at the same address would name again, around and around.
fn name_inside(e: &mut Expr, call: Expr, env: &Env, idx: &Index) -> usize {
    note_named(&call);
    *e = call;
    let c = match e {
        Expr::AddrOf(x) => &mut **x,
        c => c,
    };
    let Expr::Call { callee: Callee::Method { this, .. }, .. } = c else { return 0 };
    match &mut **this {
        Expr::AddrOf(x) => match &mut **x {
            Expr::Load { base, .. } | Expr::Member { base, .. } => accessor_expr(base, env, idx),
            _ => 0,
        },
        Expr::Load { base, .. } | Expr::Member { base, .. } => accessor_expr(base, env, idx),
        other => accessor_expr(other, env, idx),
    }
}

/// Rank of an accessor choice (lower is better): named const getters first, as the emitter.
fn acc_rank(name: &str, is_const: bool) -> u8 {
    if name.starts_with("operator") {
        4
    } else if is_const {
        0
    } else {
        1
    }
}

/// `&obj.m` / `(char*)obj + k` for a member the function can't name: the class's accessor
/// returning a reference to it or its address (`obj.GetM()`), when the headers have one.
fn member_ref_call(e: &Expr, env: &Env, idx: &Index) -> Option<Expr> {
    // explicit member addresses only (a plain object pointer is not the address of its first
    // member)
    // (or an object pointer reinterpreted as another class: `(const CVector3f*)xf`)
    let cast_to: Option<String> = match e {
        Expr::Cast { ty: Type::Ptr(x), e: inner } if std::env::var_os("MWDI_TAGGED_REFS").is_some() => {
            let to = class_name(x, env.db);
            let from = mwdec_lift::pointee(&ty_of(inner, env.vars)).and_then(|t| class_name(t, env.db));
            to.filter(|t| from.is_some_and(|f| mwdec_lift::sig::norm_name(&f) != mwdec_lift::sig::norm_name(t)))
        }
        // `*(const T*)p` read as an object at the start of another class's object
        Expr::AddrOf(x) if std::env::var_os("MWDI_TAGGED_REFS").is_some() => match &**x {
            Expr::Load { base, offset: 0, ty } => {
                let to = class_name(ty, env.db);
                let from = mwdec_lift::pointee(&ty_of(base, env.vars)).and_then(|t| class_name(t, env.db));
                to.filter(|t| from.is_some_and(|f| mwdec_lift::sig::norm_name(&f) != mwdec_lift::sig::norm_name(t) && !is_base_or_same(env.db, t, &f)))
            }
            _ => None,
        },
        _ => None,
    };
    if std::env::var("MWDI_TRACE_ACC").is_ok() && matches!(e, Expr::Cast { .. } | Expr::AddrOf(_)) {
        eprintln!("MREF cast {e:?} -> {cast_to:?}");
    }
    let explicit = cast_to.is_some()
        || match e {
            Expr::AddrOf(x) => matches!(&**x, Expr::Load { .. } | Expr::Member { .. }),
            Expr::Binary { op: BinOp::Add, r, .. } => matches!(&**r, Expr::Int { value, .. } if *value > 0),
            _ => false,
        };
    if !explicit || !matches!(vclass(&ty_of(e, env.vars), env.db), 2) {
        return None;
    }
    let (b, o) = crate::addr::canon_ptr(e, env);
    // the object the address is used as (`Dot(v, plane.mNormal)` wants a CVector3f): an accessor
    // returning another class (a derived `const CUnitVector3f&`) would need a cast
    let want = mwdec_lift::pointee(&ty_of(e, env.vars)).and_then(|t| class_name(t, env.db));
    let fits = |ret: &Type| match (&want, mwdec_lift::pointee(strip(ret)).and_then(|t| class_name(t, env.db))) {
        (Some(w), Some(r)) => mwdec_lift::sig::norm_name(w) == mwdec_lift::sig::norm_name(&r),
        _ => true,
    };
    // (rank, template, object address)
    let mut best: Option<(u8, usize, Expr)> = None;
    for &(ti, k) in &idx.member_refs {
        let t = &env.lib.templates[ti];
        let HoleKind::Obj { class, .. } = &t.holes[0] else { continue };
        if o < k || !fits(&t.sig.ret) {
            continue;
        }
        let Some(addr) = object_as_written(e, &b, o - k, class, env) else { continue };
        if !object_path_ok(env.db, &b, o - k, env) {
            continue;
        }
        // reinterpreted: the accessor must return that class
        if let Some(to) = &cast_to {
            if !ref_pointee_class(t).is_some_and(|c| mwdec_lift::sig::norm_name(&c) == mwdec_lift::sig::norm_name(to)) {
                continue;
            }
        } else if matches!(t.shape, Shape::Scalar(Expr::Var(_))) {
            continue;
        }
        if k == 0 && cast_to.is_none() {
            // the object's own address: only through an explicit member lvalue of another type
            let Expr::AddrOf(x) = e else { continue };
            if class_name(&ty_of(x, env.vars), env.db).map_or(true, |c| is_base_or_same(env.db, class, &c)) {
                continue;
            }
        }
        // the member must be one this function can't name itself
        let Some((path, _)) = mwdec_lift::types::field_path(env.db, class, k, 0) else { continue };
        // (the first member on the way the function can't name must be the accessor class's own:
        // otherwise the inner object's accessor names it, `mPtr.get()`, not the outer one's)
        let first_hidden = path.iter().find_map(|pe| match pe {
            mwdec_lift::types::PathElem::Field(n, owner) if !member_accessible(env.db, owner, n) => Some(owner.clone()),
            _ => None,
        });
        let hidden = first_hidden.is_some_and(|ow| is_base_or_same_any(env.db, &ow, class));
        if !hidden {
            continue;
        }
        // a const object only has its const accessors
        let cobj = const_object(&addr, env);
        if cobj && !t.sig.is_const {
            continue;
        }
        let name = mwdec_lift::sig::split_scope(&t.name).1.to_string();
        // named const getters first (as the emitter's accessor choice); of an overload pair the
        // non-const one on a non-const object
        let rank = if name.starts_with("operator") { 4 } else if t.sig.is_const { 0 } else { 1 } + if t.guessed { 8 } else { 0 };
        let better = match &best {
            None => true,
            Some((r, bi, _)) => {
                let bt = &env.lib.templates[*bi];
                let bn = mwdec_lift::sig::split_scope(&bt.name).1.to_string();
                if bn == name && !cobj {
                    !t.sig.is_const && bt.sig.is_const
                } else {
                    rank < *r
                }
            }
        };
        if better {
            best = Some((rank, ti, addr));
        }
    }
    // trivial accessors from the declarations
    let accs = accessors(env.db);
    let mut best_d: Option<(u8, String, usize, Expr)> = None;
    for (ai, a) in accs.iter().enumerate() {
        if matches!(a.how, 0 | 3) || o < a.off || !fits(&a.sig.ret) {
            continue;
        }
        let Some(addr) = object_as_written(e, &b, o - a.off, &a.class, env) else { continue };
        if !object_path_ok(env.db, &b, o - a.off, env) {
            continue;
        }
        if a.off == 0 {
            let Expr::AddrOf(x) = e else { continue };
            if class_name(&ty_of(x, env.vars), env.db).map_or(true, |c| is_base_or_same(env.db, &a.class, &c)) {
                continue;
            }
        }
        if !a.fields.iter().find(|(n, owner)| !member_accessible(env.db, owner, n)).is_some_and(|(_, ow)| is_base_or_same_any(env.db, ow, &a.class)) {
            continue;
        }
        let cobj = const_object(&addr, env);
        if cobj && !a.sig.is_const {
            continue;
        }
        let name = mwdec_lift::sig::split_scope(&a.sig.qualified_name).1.to_string();
        let rank = acc_rank(&name, a.sig.is_const);
        let better = match &best_d {
            None => true,
            Some((r, bn, bi, _)) => {
                if *bn == name && !cobj {
                    !a.sig.is_const && accs[*bi].sig.is_const
                } else {
                    rank < *r || (rank == *r && name < *bn)
                }
            }
        };
        if better {
            best_d = Some((rank, name, ai, addr));
        }
    }
    if best.is_none() || best_d.is_some() {
        if let Some((_, _, ai, addr)) = best_d {
            let a = &accs[ai];
            let ret = match (a.how, strip(&a.sig.ret)) {
                (1, Type::Ref(inner)) => (**inner).clone(),
                _ => a.sig.ret.clone(),
            };
            let call = Expr::Call { callee: Callee::Method { symbol: String::new(), sig: a.sig.clone(), this: Box::new(addr), qualified: false }, args: vec![], ret };
            return Some(if a.how == 1 { Expr::AddrOf(Box::new(call)) } else { call });
        }
    }
    let (_, ti, addr) = best?;
    let t = &env.lib.templates[ti];
    let mut args = vec![addr];
    args.extend(tag_defaults(t));
    let mut call = make_call(t, args);
    if t.ret_ref {
        if let (Expr::Call { ret, .. }, Type::Ref(inner)) = (&mut call, strip(&t.sig.ret)) {
            *ret = (**inner).clone();
        }
        return Some(Expr::AddrOf(Box::new(call)));
    }
    Some(call)
}

/// A read `obj.m` / `*(T*)((char*)obj + k)` of a member the function can't name: the class's
/// accessor returning its value (`obj.GetM()`), when the headers have one.
fn member_read_call(e: &Expr, env: &Env, idx: &Index) -> Option<Expr> {
    let ety = match e {
        Expr::Load { ty, .. } | Expr::Member { ty, .. } => ty.clone(),
        _ => return None,
    };
    let esize = mwdec_lift::scalar_size(strip(&ety))?;
    let (b, o) = crate::addr::access(e, env)?;
    let mut best: Option<(u8, String, usize, Expr)> = None;
    for (ti, k, ty) in &idx.member_reads {
        let t = &env.lib.templates[*ti];
        if matches!(strip(&t.sig.ret), Type::Named(_)) && !mwdec_lift::types::is_enum(Some(env.db), strip(&t.sig.ret)) {
            continue;
        }
        if *k != o && o < *k {
            continue;
        }
        if mwdec_lift::scalar_size(strip(ty)) != Some(esize) || vclass(ty, env.db) != vclass(&ety, env.db) {
            continue;
        }
        let HoleKind::Obj { class, .. } = &t.holes[0] else { continue };
        let Some(addr) = object_as_written(e, &b, o - k, class, env) else { continue };
        if !object_path_ok(env.db, &b, o - k, env) {
            continue;
        }
        let Some((path, _)) = mwdec_lift::types::field_path(env.db, class, *k, esize) else { continue };
        // (the first member on the way the function can't name must be the accessor class's own:
        // otherwise the inner object's accessor names it, `mPtr.get()`, not the outer one's)
        let first_hidden = path.iter().find_map(|pe| match pe {
            mwdec_lift::types::PathElem::Field(n, owner) if !member_accessible(env.db, owner, n) => Some(owner.clone()),
            _ => None,
        });
        let hidden = first_hidden.is_some_and(|ow| is_base_or_same_any(env.db, &ow, class));
        if !hidden {
            continue;
        }
        let cobj = const_object(&addr, env);
        if cobj && !t.sig.is_const {
            continue;
        }
        let name = mwdec_lift::sig::split_scope(&t.name).1.to_string();
        // the accessor of the innermost object first (the function names the way to it)
        let depth = path.iter().filter(|pe| matches!(pe, mwdec_lift::types::PathElem::Field(..))).count() as u8;
        let rank = depth * 16 + if name.starts_with("operator") { 4 } else if t.sig.is_const { 0 } else { 1 } + if t.guessed { 8 } else { 0 };
        let better = match &best {
            None => true,
            Some((r, bn, _, _)) => rank < *r || (rank == *r && name < *bn),
        };
        if better {
            best = Some((rank, name, *ti, addr));
        }
    }
    // trivial accessors from the declarations (ranked together with the templates)
    let accs = accessors(env.db);
    let mut best_d: Option<(u8, String, usize, Expr)> = None;
    for (ai, a) in accs.iter().enumerate() {
        if !matches!(a.how, 0 | 3) || o < a.off {
            continue;
        }
        let rt = mwdec_lift::types::resolve(Some(env.db), strip(&a.ty)).into_owned();
        if mwdec_lift::scalar_size(strip(&rt)).or_else(|| mwdec_lift::types::is_enum(Some(env.db), &rt).then_some(4)) != Some(esize) || vclass(&a.ty, env.db) != vclass(&ety, env.db) {
            continue;
        }
        let Some(addr) = object_as_written(e, &b, o - a.off, &a.class, env) else { continue };
        if !object_path_ok(env.db, &b, o - a.off, env) {
            continue;
        }
        if !a.fields.iter().find(|(n, owner)| !member_accessible(env.db, owner, n)).is_some_and(|(_, ow)| is_base_or_same_any(env.db, ow, &a.class)) {
            continue;
        }
        let cobj = const_object(&addr, env);
        if cobj && !a.sig.is_const {
            continue;
        }
        let name = mwdec_lift::sig::split_scope(&a.sig.qualified_name).1.to_string();
        let rank = a.fields.len() as u8 * 16 + acc_rank(&name, a.sig.is_const);
        if best_d.as_ref().map_or(true, |(r, bn, _, _)| rank < *r || (rank == *r && name < *bn)) {
            best_d = Some((rank, name, ai, addr));
        }
    }
    let use_t = match (&best, &best_d) {
        (Some((rt, nt, _, _)), Some((rd, nd, _, _))) => (*rt, nt) <= (*rd, nd),
        (Some(_), None) => true,
        _ => false,
    };
    if std::env::var("MWDI_TRACE_ACC").is_ok() {
        eprintln!("MREAD {e:?} -> t {:?} d {:?}", best.as_ref().map(|b| env.lib.templates[b.2].name.clone()), best_d.as_ref().map(|b| accs[b.2].sig.qualified_name.clone()));
    }
    if use_t {
        let (_, _, ti, addr) = best?;
        return Some(make_call(&env.lib.templates[ti], vec![addr]));
    }
    let (_, _, ai, addr) = best_d?;
    let a = &accs[ai];
    if a.how == 3 {
        // the pointer value is the address of the returned reference's object
        let Type::Ref(inner) = strip(&a.sig.ret) else { return None };
        let call = Expr::Call { callee: Callee::Method { symbol: String::new(), sig: a.sig.clone(), this: Box::new(addr), qualified: false }, args: vec![], ret: (**inner).clone() };
        return Some(Expr::AddrOf(Box::new(call)));
    }
    Some(Expr::Call { callee: Callee::Method { symbol: String::new(), sig: a.sig.clone(), this: Box::new(addr), qualified: false }, args: vec![], ret: a.sig.ret.clone() })
}

/// The lifter's fast-cast helpers for quantized stores (float -> u8, u16, s8, s16).
const FAST_CASTS: [(&str, &str); 4] = [("CCast::ToUint8", "ToUint8__5CCastFf"), ("CCast::FtoUS", "FtoUS__5CCastFf"), ("CCast::ToInt8", "ToInt8__5CCastFf"), ("CCast::FtoS", "FtoS__5CCastFf")];
/// ... and for quantized loads (u8, s16 -> float), taking the integer by reference.
const FAST_LOADS: [(&str, &str); 2] = [("CCast::ToReal32", "ToReal32__5CCastFRCUc"), ("CCast::StoF", "StoF__5CCastFRCs")];

/// A conversion opened from a fast-cast helper call: the cast's type is wrapped in `Volatile`.
fn is_opened_cast(e: &Expr) -> bool {
    matches!(e, Expr::Cast { ty: Type::Volatile(_), .. })
}

fn open_fast_casts(body: &mut [Stmt]) {
    Stmt::rewrite_exprs(body, &mut |e| {
        let Expr::Call { callee: Callee::Direct { symbol, .. }, args, ret } = e else { return };
        if args.len() != 1 {
            return;
        }
        if FAST_CASTS.iter().any(|(_, s)| s == symbol) {
            *e = Expr::Cast { ty: Type::Volatile(Box::new(ret.clone())), e: Box::new(args[0].clone()) };
        } else if FAST_LOADS.iter().any(|(_, s)| s == symbol) {
            if let Expr::AddrOf(x) = &args[0] {
                *e = Expr::Cast { ty: Type::Volatile(Box::new(ret.clone())), e: x.clone() };
            }
        }
    });
}

fn restore_fast_casts(ir: &mut IrFunction, db: &TypeDb) {
    let vars = ir.vars.clone();
    let mut restore = |e: &mut Expr| {
        let Expr::Cast { ty: Type::Volatile(qt), e: x } = e else { return };
        let call = |sym: &str, arg: Expr, ret: Type| Expr::Call { callee: Callee::Direct { symbol: sym.to_string(), sig: mwdec_lift::sig::sig_of(sym, Some(db)) }, args: vec![arg], ret };
        let store = match strip(qt) {
            Type::Int { size: 1, signed: false } => Some(0),
            Type::Int { size: 2, signed: false } => Some(1),
            Type::Int { size: 1, signed: true } => Some(2),
            Type::Int { size: 2, signed: true } => Some(3),
            _ => None,
        };
        if let Some(k) = store {
            *e = call(FAST_CASTS[k].1, (**x).clone(), (**qt).clone());
            return;
        }
        let load = match strip(&ty_of(x, &vars)) {
            Type::Int { size: 1, signed: false } => Some(0),
            Type::Int { size: 2, signed: true } => Some(1),
            _ => None,
        };
        match load {
            Some(k) => *e = call(FAST_LOADS[k].1, Expr::AddrOf(x.clone()), (**qt).clone()),
            None => *e = Expr::Cast { ty: (**qt).clone(), e: x.clone() },
        }
    };
    Stmt::rewrite_exprs(&mut ir.body, &mut restore);
    for init in ir.init_list.iter_mut() {
        for a in init.args.iter_mut() {
            a.rewrite(&mut restore);
        }
    }
}

thread_local! {
    /// Bytes of the function's literal symbols while it is being rewritten.
    static LITERALS: std::cell::RefCell<HashMap<String, Vec<u8>>> = std::cell::RefCell::new(HashMap::new());
    /// Bytes of its writable words that may be compiler temporaries (compared, never folded).
    static TEMPS: std::cell::RefCell<HashMap<String, Vec<u8>>> = std::cell::RefCell::new(HashMap::new());
}

/// The bytes of the function's literal symbol `sym` in hex (see [`crate::template::LITERAL_PREFIX`]).
fn literal_hex(sym: &str) -> Option<String> {
    let hex = |b: &Vec<u8>| b.iter().map(|x| format!("{x:02x}")).collect();
    LITERALS.with(|l| l.borrow().get(sym).map(hex)).or_else(|| TEMPS.with(|l| l.borrow().get(sym).map(hex)))
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
    // `if (dst) <copy construction>`: `new (dst) T(src)`
    let stripped = stripped + crate::defctor::placement_copies(ir, db, &lib.copy_ctors);
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
    if std::env::var("MWDI_DUMP_IR").is_ok_and(|f| ir.symbol.contains(f.as_str())) {
        for st in &ir.body {
            eprintln!("IR {st:?}");
        }
        for (k, v) in ir.vars.iter().enumerate() {
            eprintln!("VAR {k} {:?} {:?}", v.kind, v.ty);
        }
    }
    // by-value accessor results bound to a reference local (their dead stores), before folding
    // so the folds read the local's members
    total += crate::reflocal::apply(ir, lib, db);
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
    // reference locals bound only to pass the accessor's result on once
    total += crate::reflocal::unbind_single_use(ir);
    // members the function can't name, through their accessors (last: a naming step after the
    // temp-forwarding passes, which would treat the accessor calls as values to share)
    let named = {
        let raw = build_defs(&ir.body, &ir.vars);
        let vars = ir.vars.clone();
        let env0 = Env { db, vars: &vars, defs: &raw, lib, objects: &idx.objects };
        let defs = crate::safety::safe_defs(&ir.body, &env0);
        let env = Env { db, vars: &vars, defs: &defs, lib, objects: &idx.objects };
        accessor_stmts(&mut ir.body, &env, &idx)
    };
    crate::stmtinl::finish(ir);
    total += crate::walkptr::constructed_buffers(ir, db);
    total + named
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
    // an object built in place by its (out-of-line) member-wise constructor: the value of an
    // object-valued inline (`CVector2f::Lerp(a, b, t)` returning `CVector2f(x, y)`)
    if std::env::var("MWDI_NO_CTOR_EXPLAIN").is_err() {
        if let Stmt::Expr(Expr::Call { callee: Callee::Method { sig, this, .. }, args, .. }) = &*s {
            if let (Some(cls), Expr::AddrOf(slot)) = (sig.this_class.as_deref(), &**this) {
                let (_, last) = mwdec_lift::sig::split_scope(&sig.qualified_name);
                let ctor = mwdec_lift::sig::split_scope(cls).1.split('<').next() == Some(last);
                if ctor && matches!(&**slot, Expr::Var(_)) && args.len() > 1 && args.iter().any(|a| !matches!(a, Expr::Int { .. } | Expr::Float { .. })) {
                    if let Some(fields) = crate::template::flat_fields(env.db, cls).filter(|f| f.len() == args.len()) {
                        let m: BTreeMap<i32, Expr> = fields.iter().zip(args.iter()).map(|((o, _), a)| (*o, a.clone())).collect();
                        if let Some((call, sc)) = explain_object(env, cls, &m, 0) {
                            if sc >= MIN_SCORE && !matches!(call, Expr::Construct { .. }) {
                                *s = Stmt::Assign { dst: (**slot).clone(), src: call };
                                return 1 + scalar_shallow(s, env, idx);
                            }
                        }
                    }
                }
            }
        }
    }
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

/// The member values of `cls(args...)` by constructor `sig`: each member its constructor template
/// sets from one parameter (`mTime(time)`), when every argument sets one.
fn ctor_members(env: &Env, cls: &str, sig: &mwdec_core::FuncSig, args: &[Expr]) -> Option<BTreeMap<i32, Expr>> {
    let list = env.objects.get(&mwdec_lift::sig::norm_name(cls))?;
    for &ti in list {
        let t = &env.lib.templates[ti];
        let Shape::Object { comps, .. } = &t.shape else { continue };
        if !matches!(t.kind, CallKind::Ctor) || t.holes.len() != args.len() || t.sig.params.len() != sig.params.len() {
            continue;
        }
        if t.sig.params.iter().zip(&sig.params).any(|(a, b)| strip(&a.ty) != strip(&b.ty)) {
            continue;
        }
        let mut m = BTreeMap::new();
        for c in comps {
            let h = match &c.pat {
                Expr::Var(h) => *h,
                Expr::Load { base, offset: 0, .. } | Expr::Member { base, offset: 0, .. } => match &**base {
                    Expr::Var(h) => *h,
                    _ => return None,
                },
                Expr::Int { .. } | Expr::Float { .. } => continue,
                _ => return None,
            };
            m.insert(c.off, args.get(h)?.clone());
        }
        return (m.len() == args.len()).then_some(m);
    }
    None
}

fn root_ok(p: &Expr, t: &Expr) -> bool {
    match (p, t) {
        (Expr::Binary { op, .. }, Expr::Binary { op: o2, .. }) => op == o2,
        (Expr::Unary { op, .. }, Expr::Unary { op: o2, .. }) => op == o2,
        (Expr::Call { .. }, Expr::Call { .. }) | (Expr::Ternary { .. }, Expr::Ternary { .. }) | (Expr::Index { .. }, Expr::Index { .. }) => true,
        (Expr::Cast { e: a, .. }, Expr::Cast { e: b, .. }) => matches!((&**a, &**b), (Expr::Call { .. }, Expr::Call { .. })) || (!matches!(&**a, Expr::Cast { .. }) && root_ok(a, b)),
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
    if let Expr::Construct { class, args, ctor } = &*e {
        if let Some(cn) = class_name(class, env.db) {
            if let Some(fields) = crate::template::flat_fields(env.db, &cn) {
                if fields.len() == args.len() && args.len() > 1 && args.iter().any(|a| !matches!(a, Expr::Int { .. } | Expr::Float { .. })) {
                    // (a declared constructor's arguments are in its parameter order: the
                    // members they initialise, per its template)
                    let by_ctor = ctor.as_ref().and_then(|s| ctor_members(env, &cn, s, args));
                    let m: BTreeMap<i32, Expr> = by_ctor.unwrap_or_else(|| fields.iter().zip(args.iter()).map(|((o, _), a)| (*o, a.clone())).collect());
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
            let mut sc = use_score(t, extra, false, &args);
            // an element read of an object (`v[i]` = `(&v.mX)[i]`): the object binding is the
            // evidence (no operator nodes to count)
            if matches!(p, Expr::Index { .. }) && t.ops == 0 && std::env::var("MWDI_NO_ELEMREAD").is_err() {
                // (a const object only has the const overload)
                if matches!(t.kind, CallKind::Method) && !t.sig.is_const && args.first().is_some_and(|a| const_object(a, env)) {
                    continue;
                }
                sc = sc.max(MIN_SCORE);
            }
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
        Expr::AddrOf(x) if matches!(&**x, Expr::Load { .. } | Expr::Member { .. }) => {
            if let Expr::Load { base, .. } | Expr::Member { base, .. } = &mut **x {
                n += scalar_expr(base, env, idx);
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
/// A value dce may drop: no call, or (`PURE_NAMES` set) only folded calls of side-effect-free
/// inlines.
fn pure_value(src: &Expr) -> bool {
    if !src.has_call() {
        return true;
    }
    if !only_inline_calls(src) || std::env::var("MWDI_NO_PURE_CALL_DEFS").is_ok() {
        return false;
    }
    let mut ok = true;
    src.walk(&mut |x| {
        if let Expr::Call { callee: Callee::Direct { sig, .. } | Callee::Method { sig, .. }, .. } = x {
            ok &= PURE_OK.with(|p| p.borrow().as_ref().is_some_and(|eff| !eff.contains(&sig.qualified_name)));
        }
    });
    ok
}

thread_local! {
    /// Names of effectful inlines of the library in use (None outside `apply`).
    static PURE_OK: std::cell::RefCell<Option<std::collections::HashSet<String>>> = const { std::cell::RefCell::new(None) };
}

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
                Stmt::Assign { dst: Expr::Var(v), src } if matches!(vars[*v].kind, VarKind::Local) && pure_value(src) => {
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

pub fn unfold_pub(e: &Expr, env: &Env) -> Option<Expr> {
    unfold(e, env)
}

/// The expansion of a folded scalar inline call (`v.size()` -> `v.mCount`), from its template,
/// when every hole is a plain value or object pointer.
pub(crate) fn unfold(e: &Expr, env: &Env) -> Option<Expr> {
    let Expr::Call { callee, args, .. } = e else { return None };
    let (sig, this) = match callee {
        Callee::Method { sig, this, .. } if sig.mangled.is_none() => (sig, Some(&**this)),
        Callee::Direct { sig, .. } if sig.mangled.is_none() => (sig, None),
        _ => return None,
    };
    let Some(t) = env.lib.templates.iter().find(|t| t.name == sig.qualified_name && matches!(t.shape, Shape::Scalar(_)) && !t.ret_ref && t.holes.len() == args.len() + this.is_some() as usize) else {
        // a trivial accessor folded from the declarations: its member read / address again
        let this = this?;
        if !args.is_empty() {
            return None;
        }
        let accs = accessors(env.db);
        let a = accs.iter().find(|a| matches!(a.how, 0 | 2) && a.sig.qualified_name == sig.qualified_name && a.sig.is_const == sig.is_const)?;
        let load = Expr::Load { base: Box::new(this.clone()), offset: a.off, ty: a.ty.clone() };
        return Some(if a.how == 2 { Expr::AddrOf(Box::new(load)) } else { load });
    };
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
