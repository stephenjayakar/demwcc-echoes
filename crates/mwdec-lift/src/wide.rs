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
                && (as_hi(e, vars).or_else(|| as_lo(e, vars))).map_or(false, |x| matches!(x, Expr::Var(t) if is_temp.get(t).copied().unwrap_or(false)))
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
