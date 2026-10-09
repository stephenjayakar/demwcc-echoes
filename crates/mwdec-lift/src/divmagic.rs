//! Division by constants. MWCC divides by a constant with a multiply-high by a magic number:
//!
//! ```text
//! unsigned:  q = mulhwu(x, M) >> s                       (x / d)
//!            t = mulhwu(x, M); q = (((x - t) >> 1) + t) >> s
//! signed:    t = mulhw(M, x) >> s; q = t + ((u32)t >> 31) (also with `+ x` before the shift)
//! ```
//!
//! The lifter writes `mulhw[u]` as the pure `(u32)((u64)a * (u64)b >> 32)`; this pass recognises
//! the shapes above (looking through single-assignment temps), verifies the divisor numerically
//! and rewrites them to `x / d`.

use crate::ir::*;
use crate::types;
use mwdec_core::Type;
use std::collections::HashMap;

/// `mulhwu(a, b)` / `mulhw(a, b)` as a pure expression.
pub fn mulh(a: Expr, b: Expr, signed: bool) -> Expr {
    let (t64, t32) = if signed { (t_int(8, true), t_s32()) } else { (t_int(8, false), t_u32()) };
    let prod = Expr::bin(BinOp::Mul, Expr::cast(t64.clone(), a), Expr::cast(t64.clone(), b), t64.clone());
    Expr::cast(t32, Expr::bin(BinOp::Shr, prod, Expr::int(32), t64))
}

fn strip(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, ty } if matches!(strip_cv(ty), Type::Int { size: 4, .. }) => strip(e),
        e => e,
    }
}

struct Ctx<'a> {
    defs: &'a HashMap<VarId, Expr>,
}

impl Ctx<'_> {
    fn res<'b>(&'b self, e: &'b Expr) -> &'b Expr {
        let e = strip(e);
        if let Expr::Var(v) = e {
            if let Some(d) = self.defs.get(v) {
                return strip(d);
            }
        }
        e
    }

    /// mulh(x, M) -> (x, M, signed)
    fn mulh_of(&self, e: &Expr) -> Option<(Expr, u32, bool)> {
        let Expr::Cast { e: inner, ty } = self.res_keep_cast(e) else { return None };
        let signed = is_signed(ty) == Some(true);
        let Expr::Binary { op: BinOp::Shr, l, r, .. } = &**inner else { return None };
        if r.as_int() != Some(32) {
            return None;
        }
        let Expr::Binary { op: BinOp::Mul, l: a, r: b, .. } = &**l else { return None };
        let (a, b) = (strip_any(a), strip_any(b));
        let (x, m) = match (self.res(a).as_int(), self.res(b).as_int()) {
            (_, Some(m)) => (a.clone(), m),
            (Some(m), _) => (b.clone(), m),
            _ => return None,
        };
        Some((x, m as u32, signed))
    }

    /// Like `res` but keeps the outer 32-bit cast of a mulh.
    fn res_keep_cast<'b>(&'b self, e: &'b Expr) -> &'b Expr {
        let mut e = e;
        let mut fuel = crate::fuel::Fuel::new("divmagic.defs", crate::fuel::CAP_WALK);
        loop {
            if !fuel.burn() {
                return e;
            }
            match e {
                Expr::Var(v) => match self.defs.get(v) {
                    Some(d) => e = d,
                    None => return e,
                },
                Expr::Cast { e: inner, .. } if matches!(&**inner, Expr::Cast { .. } | Expr::Var(_)) => e = inner,
                _ => return e,
            }
        }
    }
}

fn strip_any(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip_any(e),
        e => e,
    }
}

fn same(c: &Ctx, a: &Expr, b: &Expr) -> bool {
    strip(a) == strip(b) || c.res(a) == c.res(b)
}

const SAMPLES: [u32; 16] = [0, 1, 7, 99, 100, 1000, 4095, 65535, 65536, 999_999, 0x0123_4567, 0x7fff_ffff, 0x8000_0000, 0x9abc_def0, 0xffff_fff0, 0xffff_ffff];

fn find_divisor(est: f64, check: &dyn Fn(u32) -> bool) -> Option<i64> {
    let base = est.round() as i64;
    for d in [base, base - 1, base + 1] {
        if d >= 2 && check(d as u32) {
            return Some(d);
        }
    }
    None
}

/// Unsigned `mulhwu(x, M) >> s` -> d
fn udiv_simple(m: u32, s: u32) -> Option<i64> {
    let est = 2f64.powi(32 + s as i32) / m as f64;
    find_divisor(est, &|d| SAMPLES.iter().all(|&x| (((x as u64 * m as u64) >> 32) >> s) as u32 == x / d))
}

/// Unsigned add form -> d
fn udiv_add(m: u32, s: u32) -> Option<i64> {
    let est = 2f64.powi(33 + s as i32) / (2f64.powi(32) + m as f64);
    find_divisor(est, &|d| {
        SAMPLES.iter().all(|&x| {
            let t = ((x as u64 * m as u64) >> 32) as u32;
            ((((x - t) >> 1) + t) >> s) == x / d
        })
    })
}

/// Signed `t = mulhw(M, x) [+ x] >> s; t + (t >>> 31)` -> d
fn sdiv(m: u32, s: u32, plus_x: bool) -> Option<i64> {
    let mi = m as i32 as i64;
    let est = if plus_x { 2f64.powi(32 + s as i32) / (mi as f64 + 2f64.powi(32)) } else { 2f64.powi(32 + s as i32) / mi as f64 };
    find_divisor(est, &|d| {
        SAMPLES.iter().all(|&ux| {
            let x = ux as i32;
            let mut t = ((x as i64 * mi) >> 32) as i32;
            if plus_x {
                t = t.wrapping_add(x);
            }
            let t = t >> s;
            let q = t.wrapping_add(((t as u32) >> 31) as i32);
            q == x.wrapping_div(d as i32)
        })
    })
}

fn try_fold(e: &Expr, c: &Ctx, vars: &[Var]) -> Option<Expr> {
    match e {
        // `x / d * 2^k`: the quotient's final shift and the multiply fold into one mask
        // (`(q >> k) << k` is `q & -(1 << k)`)
        Expr::Binary { op: BinOp::And, l, r, ty } if r.as_int().is_some_and(|m| {
            let m = m as u32;
            m != 0 && m != u32::MAX && (!m).wrapping_add(1).is_power_of_two() && !m & (!m).wrapping_add(1) == 0
        }) => {
            let m = r.as_int()? as u32;
            let k = (!m).wrapping_add(1).trailing_zeros();
            let q = try_fold(&Expr::bin(BinOp::Shr, (**l).clone(), Expr::int(k as i64), ty.clone()), c, vars)?;
            let qt = types::ty_of(&q, vars);
            Some(Expr::bin(BinOp::Mul, q, Expr::int(1i64 << k), qt))
        }
        Expr::Binary { op: BinOp::Shr, l, r, .. } => {
            let s = r.as_int()? as u32;
            // unsigned simple
            if let Some((x, m, false)) = c.mulh_of(l) {
                let d = udiv_simple(m, s)?;
                return Some(Expr::bin(BinOp::Div, Expr::cast(t_u32(), x), Expr::uint(d), t_u32()));
            }
            // unsigned add form: ((x - t) >> 1) + t
            if let Expr::Binary { op: BinOp::Add, l: a, r: t, .. } = c.res(l) {
                let (tm, sub) = if c.mulh_of(t).is_some() { (t, a) } else { (a, t) };
                let (x, m, false) = c.mulh_of(tm)? else { return None };
                let Expr::Binary { op: BinOp::Shr, l: sl, r: one, .. } = c.res(sub) else { return None };
                if one.as_int() != Some(1) {
                    return None;
                }
                let Expr::Binary { op: BinOp::Sub, l: x2, r: t2, .. } = c.res(sl) else { return None };
                if !same(c, x2, &x) || !same(c, t2, tm) {
                    return None;
                }
                let d = udiv_add(m, s)?;
                return Some(Expr::bin(BinOp::Div, Expr::cast(t_u32(), x), Expr::uint(d), t_u32()));
            }
            None
        }
        Expr::Binary { op: BinOp::Add, l, r, .. } => {
            // t + ((u32)t >> 31) with t = mulhw(M, x) [+ x] >> s (arithmetic)
            let (t, sign) = match (c.res(l), c.res(r)) {
                (_, Expr::Binary { op: BinOp::Shr, l: tt, r: k, .. }) if k.as_int() == Some(31) && same(c, tt, l) => (l, ()),
                (Expr::Binary { op: BinOp::Shr, l: tt, r: k, .. }, _) if k.as_int() == Some(31) && same(c, tt, r) => (r, ()),
                _ => return None,
            };
            let _ = sign;
            let Expr::Binary { op: BinOp::Shr, l: inner, r: s, .. } = c.res(t) else { return None };
            let s = s.as_int()? as u32;
            if let Some((x, m, true)) = c.mulh_of(inner) {
                let d = sdiv(m, s, false)?;
                return Some(Expr::bin(BinOp::Div, Expr::cast(t_s32(), x), Expr::int(d), t_s32()));
            }
            if let Expr::Binary { op: BinOp::Add, l: a, r: b, .. } = c.res(inner) {
                let (mh, x2) = if c.mulh_of(a).is_some() { (a, b) } else { (b, a) };
                let (x, m, true) = c.mulh_of(mh)? else { return None };
                if !same(c, x2, &x) {
                    return None;
                }
                let d = sdiv(m, s, true)?;
                return Some(Expr::bin(BinOp::Div, Expr::cast(t_s32(), x), Expr::int(d), t_s32()));
            }
            let _ = vars;
            None
        }
        _ => None,
    }
}

fn collect_defs(b: &[Stmt], is_temp: &[bool], out: &mut HashMap<VarId, Expr>, count: &mut HashMap<VarId, usize>) {
    for s in b {
        if let Stmt::Assign { dst: Expr::Var(v), src } = s {
            if is_temp.get(*v).copied().unwrap_or(false) {
                out.insert(*v, src.clone());
                *count.entry(*v).or_default() += 1;
            }
        }
        match s {
            Stmt::If { then, els, .. } => {
                collect_defs(then, is_temp, out, count);
                collect_defs(els, is_temp, out, count);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect_defs(body, is_temp, out, count),
            Stmt::For { init, step, body, .. } => {
                collect_defs(init, is_temp, out, count);
                collect_defs(step, is_temp, out, count);
                collect_defs(body, is_temp, out, count);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    collect_defs(&c.body, is_temp, out, count);
                }
            }
            _ => {}
        }
    }
}

/// Signed remainder by 2^k: `t = (u32)x >> 31; r = rotl((x << (32 - k)) - t, k) + t` is `x % 2^k`.
fn rem_pow2(e: &Expr, c: &Ctx, vars: &[Var]) -> Option<Expr> {
    let Expr::Binary { op: BinOp::Add, l, r, .. } = e else { return None };
    // the sign term: x >> 31 (logical)
    let sign_of = |t: &Expr| -> Option<Expr> {
        match c.res(t) {
            Expr::Binary { op: BinOp::Shr, l: x, r: k, .. } if k.as_int() == Some(31) => Some((**x).clone()),
            _ => None,
        }
    };
    // rotl(s, k) as `(s << k | s >> (32 - k)) [& 0xffffffff]`
    let rot = |e: &Expr| -> Option<(Expr, i64)> {
        let e = match c.res(e) {
            Expr::Binary { op: BinOp::And, l, r, .. } if r.as_int().map(|m| m as u32) == Some(u32::MAX) => c.res(l),
            e => e,
        };
        let Expr::Binary { op: BinOp::Or, l, r, .. } = e else { return None };
        let (Expr::Binary { op: BinOp::Shl, l: a, r: k1, .. }, Expr::Binary { op: BinOp::Shr, l: b, r: k2, .. }) = (strip_any(l), strip_any(r)) else { return None };
        let (k1, k2) = (k1.as_int()?, k2.as_int()?);
        (k1 + k2 == 32 && same(c, a, b)).then(|| ((**a).clone(), k1))
    };
    for (rp, tp) in [(l, r), (r, l)] {
        let Some(x) = sign_of(tp) else { continue };
        let Some((s, k)) = rot(rp) else { continue };
        if !(1..=16).contains(&k) {
            continue;
        }
        let Expr::Binary { op: BinOp::Sub, l: sh, r: t2, .. } = c.res(&s) else { continue };
        if !same(c, t2, tp) {
            continue;
        }
        let Expr::Binary { op: BinOp::Shl, l: x2, r: k3, .. } = c.res(sh) else { continue };
        if k3.as_int() != Some(32 - k) || !same(c, x2, &x) {
            continue;
        }
        let _ = vars;
        return Some(Expr::bin(BinOp::Rem, Expr::cast(t_s32(), x), Expr::int(1i64 << k), t_s32()));
    }
    None
}

pub fn fold(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &[bool]) {
    {
        let mut defs = HashMap::new();
        let mut count = HashMap::new();
        collect_defs(body, is_temp, &mut defs, &mut count);
        defs.retain(|v, _| count.get(v) == Some(&1));
        let c = Ctx { defs: &defs };
        Stmt::rewrite_exprs(body, &mut |e| {
            if let Some(n) = rem_pow2(e, &c, vars) {
                *e = n;
            } else if let Some(n) = half(e, &c, vars) {
                *e = n;
            }
        });
    }
    let mut has = false;
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Binary { op: BinOp::Mul, ty, .. } = e {
            if matches!(ty, Type::Int { size: 8, .. }) {
                has = true;
            }
        }
    });
    if !has {
        return;
    }
    let mut defs = HashMap::new();
    let mut count = HashMap::new();
    collect_defs(body, is_temp, &mut defs, &mut count);
    defs.retain(|v, _| count.get(v) == Some(&1));
    let c = Ctx { defs: &defs };
    let mut changed = false;
    Stmt::rewrite_exprs(body, &mut |e| {
        if let Some(n) = try_fold(e, &c, vars) {
            *e = n;
            changed = true;
        }
    });
    if !changed {
        return;
    }
    // temps left unused by the folding (pure mulh / shift values)
    loop {
        let mut uses: HashMap<VarId, usize> = HashMap::new();
        Stmt::walk_exprs(body, &mut |e| {
            if let Expr::Var(v) = e {
                *uses.entry(*v).or_default() += 1;
            }
        });
        let mut removed = false;
        Stmt::for_each_block_mut(body, &mut |b| {
            b.retain(|s| match s {
                Stmt::Assign { dst: Expr::Var(v), src } if defs.contains_key(v) && uses.get(v).copied().unwrap_or(0) == 1 && !src.has_call() && !types::ty_of(src, vars).eq(&Type::Void) => {
                    removed = true;
                    false
                }
                _ => true,
            });
        });
        if !removed {
            break;
        }
    }
}

/// Signed halving: `(x + ((u32)x >> 31)) >> 1` (arithmetic) is `x / 2` (`srwi 31; add; srawi 1`).
fn half(e: &Expr, c: &Ctx, vars: &[Var]) -> Option<Expr> {
    let Expr::Binary { op: BinOp::Shr, l, r, .. } = e else { return None };
    if r.as_int() != Some(1) || is_signed(&types::ty_of(l, vars)) != Some(true) {
        return None;
    }
    let Expr::Binary { op: BinOp::Add, l: a, r: b, .. } = c.res(l) else { return None };
    let sign_of = |t: &Expr| -> Option<Expr> {
        match c.res(t) {
            Expr::Binary { op: BinOp::Shr, l: x, r: k, .. } if k.as_int() == Some(31) && is_signed(&types::ty_of(x, vars)) != Some(true) => Some((**x).clone()),
            _ => None,
        }
    };
    let x = match (sign_of(a), sign_of(b)) {
        (Some(x), _) if same(c, &x, b) => (**b).clone(),
        (_, Some(x)) if same(c, &x, a) => (**a).clone(),
        _ => return None,
    };
    Some(Expr::bin(BinOp::Div, Expr::cast(t_s32(), strip_any(&x).clone()), Expr::int(2), t_s32()))
}
