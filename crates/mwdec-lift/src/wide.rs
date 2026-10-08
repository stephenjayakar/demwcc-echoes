//! 64-bit integers: MWCC keeps a `long long` in a register pair (high word in the lower
//! register), passes it in r3:r4 / r5:r6 ..., returns it in r3:r4, and calls runtime helpers for
//! shifts, division and modulo (`__shl2i(hi, lo, n)` = `x << n`). The lifter models a 64-bit
//! value as one expression and its halves as `(u32)(x >> 32)` / `(u32)x`; these helpers build and
//! take apart those forms, and the late pass merges half-word stores back into 64-bit stores.

use crate::ir::*;
use crate::types::ty_of;
use mwdec_core::Type;
use std::collections::HashMap;

pub fn is_wide(t: &Type) -> bool {
    matches!(strip_cv(t), Type::Int { size: 8, .. } | Type::Unknown { size: 8 })
}

fn wide_ty(signed: bool) -> Type {
    Type::Int { size: 8, signed }
}

/// High word of a 64-bit value.
pub fn hi32(e: Expr) -> Expr {
    let ty = match ty_of_plain(&e) {
        Some(t) if is_wide(&t) => t,
        _ => wide_ty(true),
    };
    Expr::Cast { ty: t_u32(), e: Box::new(Expr::bin(BinOp::Shr, e, Expr::int(32), ty)) }
}

/// Low word of a 64-bit value.
pub fn lo32(e: Expr) -> Expr {
    Expr::Cast { ty: t_u32(), e: Box::new(e) }
}

fn ty_of_plain(e: &Expr) -> Option<Type> {
    match e {
        Expr::Int { ty, .. } | Expr::Cast { ty, .. } | Expr::Binary { ty, .. } | Expr::Load { ty, .. } | Expr::Member { ty, .. } => Some(ty.clone()),
        Expr::Call { ret, .. } => Some(ret.clone()),
        _ => None,
    }
}

/// `(u32)(x >> 32)` -> x
pub fn as_hi(e: &Expr, vars: &[Var]) -> Option<Expr> {
    if let Expr::Cast { ty, e } = e {
        if matches!(strip_cv(ty), Type::Int { size: 4, .. }) {
            if let Expr::Binary { op: BinOp::Shr, l, r, .. } = &**e {
                if r.as_int() == Some(32) && is_wide(&ty_of(l, vars)) {
                    return Some((**l).clone());
                }
            }
        }
    }
    None
}

/// `(u32)x` with x 64-bit -> x
pub fn as_lo(e: &Expr, vars: &[Var]) -> Option<Expr> {
    if let Expr::Cast { ty, e } = e {
        if matches!(strip_cv(ty), Type::Int { size: 4, .. }) && is_wide(&ty_of(e, vars)) && as_hi(&Expr::Cast { ty: ty.clone(), e: e.clone() }, vars).is_none() {
            return Some((**e).clone());
        }
    }
    None
}

/// The 64-bit value whose halves are `hi` and `lo`.
pub fn pair(hi: Expr, lo: Expr, signed: bool, vars: &[Var]) -> Expr {
    if let (Some(a), Some(b)) = (as_hi(&hi, vars), as_lo(&lo, vars)) {
        if a == b {
            return a;
        }
    }
    if let (Some(h), Some(l)) = (hi.as_int(), lo.as_int()) {
        return Expr::Int { value: (h << 32) | (l & 0xffff_ffff), ty: wide_ty(signed) };
    }
    // adjacent words of one object
    match (&hi, &lo) {
        (Expr::Load { base: b1, offset: o1, ty: t1 }, Expr::Load { base: b2, offset: o2, ty: t2 })
            if b1 == b2 && *o2 == *o1 + 4 && scalar_size(t1) == Some(4) && scalar_size(t2) == Some(4) =>
        {
            return Expr::Load { base: b1.clone(), offset: *o1, ty: wide_ty(signed) };
        }
        (Expr::Member { base: b1, offset: o1, ty: t1 }, Expr::Member { base: b2, offset: o2, ty: t2 })
            if b1 == b2 && *o2 == *o1 + 4 && scalar_size(t1) == Some(4) && scalar_size(t2) == Some(4) =>
        {
            return Expr::Member { base: b1.clone(), offset: *o1, ty: wide_ty(signed) };
        }
        _ => {}
    }
    // zero / sign extension of a 32-bit value
    if hi.as_int() == Some(0) {
        return Expr::Cast { ty: wide_ty(false), e: Box::new(lo) };
    }
    if let Expr::Binary { op: BinOp::Shr, l, r, .. } = &hi {
        if r.as_int() == Some(31) && **l == lo {
            return Expr::Cast { ty: wide_ty(true), e: Box::new(lo) };
        }
    }
    let ty = wide_ty(signed);
    Expr::bin(
        BinOp::Or,
        Expr::bin(BinOp::Shl, Expr::Cast { ty: ty.clone(), e: Box::new(hi) }, Expr::int(32), ty.clone()),
        Expr::Cast { ty: wide_ty(false), e: Box::new(lo) },
        ty,
    )
}

/// Did [`pair`] recognize the halves (rather than building `(hi << 32) | lo`)?
pub fn is_merged_pair(e: &Expr) -> bool {
    !matches!(e, Expr::Binary { op: BinOp::Or, l, .. } if matches!(&**l, Expr::Binary { op: BinOp::Shl, r, .. } if r.as_int() == Some(32)))
}

/// Runtime helpers for 64-bit operations: (operator, signed, second operand is a pair).
pub fn helper(sym: &str) -> Option<(BinOp, bool, bool)> {
    Some(match sym {
        "__shl2i" => (BinOp::Shl, true, false),
        "__shr2i" => (BinOp::Shr, true, false),
        "__shr2u" => (BinOp::Shr, false, false),
        "__div2i" => (BinOp::Div, true, true),
        "__div2u" => (BinOp::Div, false, true),
        "__mod2i" => (BinOp::Rem, true, true),
        "__mod2u" => (BinOp::Rem, false, true),
        _ => return None,
    })
}

fn split_lvalue(e: &Expr) -> Option<(&Expr, i32, bool, &Type)> {
    match e {
        Expr::Load { base, offset, ty } => Some((base, *offset, true, ty)),
        Expr::Member { base, offset, ty } => Some((base, *offset, false, ty)),
        // a global word is its own base
        Expr::Global { ty, .. } => Some((e, 0, false, ty)),
        _ => None,
    }
}

fn same_base(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Global { symbol: x, .. }, Expr::Global { symbol: y, .. }) => x == y,
        _ => a == b,
    }
}

fn with_offset(e: &Expr, off: i32, ty: Type) -> Expr {
    match e {
        Expr::Load { base, .. } => Expr::Load { base: base.clone(), offset: off, ty },
        Expr::Member { base, .. } => Expr::Member { base: base.clone(), offset: off, ty },
        g @ Expr::Global { .. } => Expr::Member { base: Box::new(g.clone()), offset: off, ty },
        other => other.clone(),
    }
}

/// One half of a 64-bit masked equality: `((w & m) ^ c)` with the `& m` / `^ c` optional.
fn half_parts(e: &Expr) -> (Expr, Option<Expr>, Option<Expr>) {
    let (x, c) = match e {
        Expr::Binary { op: BinOp::Xor, l, r, .. } if r.as_int().is_some() => ((**l).clone(), Some((**r).clone())),
        _ => (e.clone(), None),
    };
    match x {
        Expr::Binary { op: BinOp::And, l, r, .. } => (*l, Some(*r), c),
        x => (x, None, c),
    }
}

/// MWCC compares 64-bit values for (in)equality by xoring each half with the other operand's
/// half and or-ing the results: `((hi & mh) ^ ch) | ((lo & ml) ^ cl)` tested against 0 is
/// `(w & m) == c` on the 64-bit word `w` (`CMaterialList`-style bit sets). Rebuild it when the two
/// halves are the adjacent words of one object.
pub fn merge_compares(body: &mut Vec<Stmt>, vars: &[Var]) {
    Stmt::rewrite_exprs(body, &mut |e| {
        let Expr::Binary { op: op @ (BinOp::Eq | BinOp::Ne), l, r, .. } = &*e else { return };
        if r.as_int() != Some(0) {
            return;
        }
        let Expr::Binary { op: BinOp::Or, l: a, r: b, .. } = &**l else { return };
        let (wa, ma, ca) = half_parts(a);
        let (wb, mb, cb) = half_parts(b);
        let off = |w: &Expr| split_lvalue(w).filter(|(_, _, _, t)| scalar_size(t) == Some(4)).map(|(base, o, _, _)| (base.clone(), o));
        let (Some((ba, oa)), Some((bb, ob))) = (off(&wa), off(&wb)) else { return };
        if !same_base(&ba, &bb) || (oa - ob).abs() != 4 {
            return;
        }
        // at least one half must really be part of a 64-bit test (a mask or a constant)
        if ma.is_none() && mb.is_none() && ca.is_none() && cb.is_none() {
            return;
        }
        let ((wh, mh, ch), (wl, ml, cl)) = if oa < ob { ((wa, ma, ca), (wb, mb, cb)) } else { ((wb, mb, cb), (wa, ma, ca)) };
        if mh.is_some() != ml.is_some() {
            return;
        }
        let w = pair(wh, wl, false, vars);
        if !matches!(w, Expr::Load { .. } | Expr::Member { .. }) {
            return;
        }
        let w = match w {
            Expr::Load { base, offset, .. } => Expr::Load { base, offset, ty: wide_ty(false) },
            Expr::Member { base, offset, .. } => Expr::Member { base, offset, ty: wide_ty(false) },
            o => o,
        };
        let masked = match (mh, ml) {
            (Some(h), Some(l)) => Expr::bin(BinOp::And, w, pair(h, l, true, vars), wide_ty(false)),
            _ => w,
        };
        let c = pair(ch.unwrap_or(Expr::int(0)), cl.unwrap_or(Expr::int(0)), false, vars);
        *e = Expr::cmp(*op, masked, c);
    });
}

/// `LO = LO | (u32)x` / `HI = HI | (u32)(x >> 32)` on the two words of one object: the word's
/// value or-ed with x (in place or through a temp `t = LO | (u32)x; HI = ..; LO = t;`).
fn or_half<'a>(src: &'a Expr, dst: &Expr, vars: &[Var]) -> Option<(Expr, bool)> {
    let Expr::Binary { op: BinOp::Or, l, r, .. } = src else { return None };
    if !same_lvalue(l, dst) {
        return None;
    }
    if let Some(x) = as_hi(r, vars) {
        return Some((x, true));
    }
    as_lo(r, vars).map(|x| (x, false))
}

fn same_lvalue(a: &Expr, b: &Expr) -> bool {
    match (split_lvalue(a), split_lvalue(b)) {
        (Some((ba, oa, _, _)), Some((bb, ob, _, _))) => oa == ob && same_base(ba, bb),
        _ => false,
    }
}

/// The 64-bit lvalue at the lower of two adjacent word lvalues.
fn wide_at(d1: &Expr, o1: i32, d2: &Expr, o2: i32, signed: bool) -> Expr {
    let d = if o1 <= o2 { d1 } else { d2 };
    let d = match d {
        Expr::Member { base, .. } if matches!(**base, Expr::Global { .. }) => (**base).clone(),
        other => other.clone(),
    };
    // a global whose 64-bit word starts it is that word (declared `unsigned long long`)
    if let (Expr::Global { symbol, .. }, 0) = (&d, o1.min(o2)) {
        return Expr::Global { symbol: symbol.clone(), ty: Type::Int { size: 8, signed: false } };
    }
    with_offset(&d, o1.min(o2), Type::Int { size: 8, signed })
}

/// `w |= x` and `w = 0` on a 64-bit object kept as two words (`CMaterialList` bit sets built
/// with `1LL << n`): merge the half-word statements back into 64-bit ones.
pub fn merge_or_assigns(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &[bool]) {
    let temp = |v: VarId| is_temp.get(v).copied().unwrap_or(false);
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        let mut merged: Vec<(usize, Expr)> = vec![];
        while i < b.len() {
            // forms: [t = LO | lo(x);] HI = HI | hi(x); LO = t|LO | lo(x)  (any order of the two stores)
            let found = (|| {
                let (mut t, mut start) = (None, i);
                let (s0, s1, s2) = (b.get(i)?, b.get(i + 1), b.get(i + 2));
                let mut stores: Vec<(&Expr, &Expr)> = vec![];
                if let Stmt::Assign { dst: Expr::Var(v), src } = s0 {
                    if temp(*v) {
                        t = Some((*v, src));
                        start = i + 1;
                    }
                }
                let _ = (s1, s2);
                for k in start..(start + 2).min(b.len()) {
                    let Stmt::Assign { dst, src } = &b[k] else { return None };
                    stores.push((dst, src));
                }
                if stores.len() != 2 {
                    return None;
                }
                let mut x_hi = None;
                let mut x_lo = None;
                let mut lv: Vec<(&Expr, i32)> = vec![];
                for (dst, src) in &stores {
                    let (_, o, _, ty) = split_lvalue(dst)?;
                    // (a global's own type may be the whole object: its word access is the op's)
                    if scalar_size(ty) != Some(4) && !matches!(dst, Expr::Global { .. }) {
                        return None;
                    }
                    let src = match (src, t) {
                        (Expr::Var(v), Some((tv, ts))) if *v == tv => ts,
                        (s, _) => *s,
                    };
                    let (x, hi) = or_half(src, dst, vars)?;
                    if hi {
                        x_hi = Some((x, o));
                    } else {
                        x_lo = Some((x, o));
                    }
                    lv.push((dst, o));
                }
                let ((xh, oh), (xl, ol)) = (x_hi?, x_lo?);
                if xh != xl || ol != oh + 4 || !same_base(split_lvalue(lv[0].0)?.0, split_lvalue(lv[1].0)?.0) {
                    return None;
                }
                // a temp must be the one consumed by the stores
                if let Some((tv, _)) = t {
                    let used = stores.iter().filter(|(_, s)| matches!(s, Expr::Var(v) if *v == tv)).count();
                    if used != 1 {
                        return None;
                    }
                }
                let signed = matches!(strip_cv(&ty_of(&xh, vars)), Type::Int { signed: true, .. });
                let w = wide_at(lv[0].0, lv[0].1, lv[1].0, lv[1].1, signed);
                Some((start + 2, w.clone(), Expr::bin(BinOp::Or, w, xh, Type::Int { size: 8, signed })))
            })();
            if let Some((end, w, v)) = found {
                b.splice(i..end, [Stmt::Assign { dst: w.clone(), src: v }]);
                merged.push((i, w));
            }
            i += 1;
        }
        // `LO = 0; HI = 0;` before the first merged `w |= x` (only temps computed between)
        for (at, w) in merged.into_iter().rev() {
            let Some((wb, wo, _, _)) = split_lvalue(&w).map(|(b, o, p, t)| (b.clone(), o, p, t.clone())) else { continue };
            let mut k = at;
            let mut zeros: Vec<usize> = vec![];
            while k > 0 && zeros.len() < 2 {
                k -= 1;
                match &b[k] {
                    Stmt::Assign { dst: Expr::Var(v), src } if temp(*v) && !src.has_call() && !matches!(src, Expr::Global { .. } if false) => {
                        let mut reads = false;
                        src.walk(&mut |e| reads |= split_lvalue(e).is_some_and(|(bb, _, _, _)| same_base(bb, &wb)));
                        if reads {
                            break;
                        }
                    }
                    Stmt::Assign { dst, src } if src.as_int() == Some(0) => match split_lvalue(dst) {
                        Some((bb, o, _, t)) if same_base(bb, &wb) && (scalar_size(t) == Some(4) || matches!(dst, Expr::Global { .. })) && (o == wo || o == wo + 4) => zeros.push(k),
                        _ => break,
                    },
                    _ => break,
                }
            }
            if zeros.len() == 2 {
                let offs: Vec<i32> = zeros.iter().map(|&z| if let Stmt::Assign { dst, .. } = &b[z] { split_lvalue(dst).unwrap().1 } else { 0 }).collect();
                if offs[0] != offs[1] {
                    let (z0, z1) = (zeros[0].max(zeros[1]), zeros[0].min(zeros[1]));
                    b.remove(z0);
                    b[z1] = Stmt::Assign { dst: w.clone(), src: Expr::Int { value: 0, ty: Type::Int { size: 8, signed: false } } };
                }
            }
        }
    });
}

/// Late clean-up: locals assigned once from a half of a 64-bit temp read the half directly,
/// and `*(u32*)p = (u32)(x >> 32); *(u32*)(p + 4) = (u32)x;` become one 64-bit store.
pub fn merge_halves(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &[bool]) {
    // single-assignment locals holding a half of a temp
    let mut assigns: HashMap<VarId, usize> = HashMap::new();
    let mut src_of: HashMap<VarId, Expr> = HashMap::new();
    fn walk(b: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
        for s in b {
            f(s);
            match s {
                Stmt::If { then, els, .. } => {
                    walk(then, f);
                    walk(els, f);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => walk(body, f),
                Stmt::For { init, step, body, .. } => {
                    walk(init, f);
                    walk(step, f);
                    walk(body, f);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        walk(&c.body, f);
                    }
                }
                _ => {}
            }
        }
    }
    walk(body, &mut |s| {
        if let Stmt::Assign { dst: Expr::Var(v), src } = s {
            *assigns.entry(*v).or_default() += 1;
            src_of.insert(*v, src.clone());
        }
    });
    let halves: HashMap<VarId, Expr> = src_of
        .into_iter()
        .filter(|(v, e)| {
            assigns.get(v) == Some(&1)
                && matches!(vars[*v].kind, VarKind::Local)
                && (as_hi(e, vars).or_else(|| as_lo(e, vars))).map_or(false, |x| {
                    // a half of a temp, or of a 64-bit parameter never reassigned
                    matches!(x, Expr::Var(t) if is_temp.get(t).copied().unwrap_or(false) || (matches!(vars[t].kind, VarKind::Param { .. }) && !assigns.contains_key(&t)))
                })
        })
        .collect();
    if !halves.is_empty() {
        Stmt::for_each_block_mut(body, &mut |b| b.retain(|s| !matches!(s, Stmt::Assign { dst: Expr::Var(v), .. } if halves.contains_key(v))));
        let mut sub = |e: &mut Expr| {
            if let Expr::Var(v) = e {
                if let Some(h) = halves.get(v) {
                    *e = h.clone();
                }
            }
        };
        Stmt::rewrite_exprs(body, &mut sub);
    }
    merge_rmw_halves(body, vars);
    // store pairs
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let found = (|| {
                let Stmt::Assign { dst: d1, src: s1 } = &b[i] else { return None };
                let (base1, o1, _, t1) = split_lvalue(d1)?;
                if scalar_size(t1) != Some(4) {
                    return None;
                }
                let (x1, is_hi) = match (as_hi(s1, vars), as_lo(s1, vars)) {
                    (Some(x), _) => (x, true),
                    (None, Some(x)) => (x, false),
                    _ => return None,
                };
                for j in i + 1..(i + 6).min(b.len()) {
                    if let Stmt::Assign { dst: d2, src: s2 } = &b[j] {
                        if let Some((base2, o2, _, t2)) = split_lvalue(d2) {
                            if same_base(base2, base1) && scalar_size(t2) == Some(4) {
                                let other = if is_hi { as_lo(s2, vars) } else { as_hi(s2, vars) };
                                let want = if is_hi { o1 + 4 } else { o1 - 4 };
                                if o2 == want && other.as_ref() == Some(&x1) {
                                    let signed = matches!(strip_cv(&ty_of(&x1, vars)), Type::Int { signed: true, .. });
                                    let lo_off = o1.min(o2);
                                    // build from the lower-addressed store's lvalue (its base is the object)
                                    let d = if o1 <= o2 { d1 } else { d2 };
                                    let d = match d {
                                        Expr::Member { base, .. } if matches!(**base, Expr::Global { .. }) => (**base).clone(),
                                        other => other.clone(),
                                    };
                                    return Some((j, with_offset(&d, lo_off, Type::Int { size: 8, signed }), x1));
                                }
                            }
                        }
                    }
                    // stop at anything that might read the memory or change x
                    if !matches!(b[j], Stmt::Assign { .. }) {
                        break;
                    }
                }
                None
            })();
            if let Some((j, dst, x)) = found {
                b.remove(j);
                b[i] = Stmt::Assign { dst, src: x };
            }
            i += 1;
        }
    });
}

/// `lo = lo OP v; hi = hi OP (v >> 31);` (OP one of `|`, `&`, `^`; `hi` possibly read into a
/// temp first) is a 64-bit `x OP= (long long)v` of a sign-extended word: one statement on the
/// 64-bit object (`x |= 1 << n` on a `u64` member).
fn merge_rmw_halves(body: &mut Vec<Stmt>, vars: &[Var]) {
    fn uses_of(b: &[Stmt], t: VarId) -> usize {
        let mut n = 0;
        Stmt::walk_exprs(b, &mut |e| {
            if matches!(e, Expr::Var(x) if *x == t) {
                n += 1;
            }
        });
        n
    }
    let total: HashMap<VarId, usize> = {
        let mut m = HashMap::new();
        Stmt::walk_exprs(body, &mut |e| {
            if let Expr::Var(x) = e {
                *m.entry(*x).or_default() += 1;
            }
        });
        m
    };
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let found = (|| {
                let Stmt::Assign { dst: d1, src: Expr::Binary { op, l, r: v, .. } } = &b[i] else { return None };
                if !matches!(op, BinOp::Or | BinOp::And | BinOp::Xor) || **l != *d1 {
                    return None;
                }
                let (base1, o1, _, t1) = split_lvalue(d1)?;
                if scalar_size(t1) != Some(4) || !matches!(strip_cv(&ty_of(v, vars)), Type::Int { size: 4, .. }) {
                    return None;
                }
                for j in i + 1..(i + 4).min(b.len()) {
                    let Stmt::Assign { dst: d2, src: s2 } = &b[j] else { return None };
                    let Some((base2, o2, _, t2)) = split_lvalue(d2) else { continue };
                    if !same_base(base2, base1) || o2 != o1 - 4 || scalar_size(t2) != Some(4) {
                        continue;
                    }
                    let Expr::Binary { op: op2, l: h, r: sx, .. } = s2 else { return None };
                    if op2 != op {
                        return None;
                    }
                    let sign = matches!(&**sx, Expr::Binary { op: BinOp::Shr, l: y, r: k, .. } if k.as_int() == Some(31) && **y == **v);
                    if !sign {
                        return None;
                    }
                    // the high word read directly, or through a temp read from it before
                    let temp = match &**h {
                        x if *x == *d2 => None,
                        Expr::Var(t) => {
                            let k = (0..i).rev().find(|&k| matches!(&b[k], Stmt::Assign { dst: Expr::Var(x), src } if x == t && *src == *d2))?;
                            if total.get(t).copied().unwrap_or(0) != 2 || uses_of(&b[k + 1..j], *t) != 0 || b[k + 1..j].iter().any(|s| matches!(s, Stmt::Assign { dst, .. } if dst == d2)) {
                                return None;
                            }
                            Some(k)
                        }
                        _ => return None,
                    };
                    let signed = false;
                    let dst = with_offset(d2, o2, Type::Int { size: 8, signed });
                    let src = Expr::bin(*op, dst.clone(), Expr::Cast { ty: Type::Int { size: 8, signed: true }, e: v.clone() }, Type::Int { size: 8, signed });
                    return Some((j, temp, Stmt::Assign { dst, src }));
                }
                None
            })();
            if let Some((j, temp, st)) = found {
                b[i] = st;
                b.remove(j);
                if let Some(k) = temp {
                    b.remove(k);
                    i -= 1;
                }
            }
            i += 1;
        }
    });
}
