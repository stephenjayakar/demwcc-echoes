//! Constant-argument branch elimination. MWCC substitutes a constant argument into the inline
//! body at the call (learnings/mwcc-codegen/inlining.md 3.1) and its constant propagation then
//! deletes the branches the constant decides (`if (normalize == kN_Yes) Normalize();` called with
//! `kN_No` leaves nothing; `if (idx != -1) return a[idx]; return -1;` called with 4 leaves
//! `a[4]`). A statement template can match such an expansion with one arm of a constant-decidable
//! `if` in place of the `if`, provided the holes of the condition are (or can be bound to)
//! constants that select that arm.

use crate::matcher::{res, Bind, M};
use crate::template::HoleKind;
use mwdec_core::Type;
use mwdec_lift::{BinOp, Expr, UnOp};

#[derive(Clone, Copy, Debug, PartialEq)]
enum V {
    I(i64),
    F(f64),
}

impl V {
    fn truth(self) -> bool {
        match self {
            V::I(x) => x != 0,
            V::F(x) => x != 0.0,
        }
    }
    fn f(self) -> f64 {
        match self {
            V::I(x) => x as f64,
            V::F(x) => x,
        }
    }
}

/// Can `cond` be decided by constants alone (no memory, no calls; only scalar holes, literals
/// and operators)? `holes` of the template.
pub fn decidable(cond: &Expr, holes: &[HoleKind]) -> bool {
    let mut ok = true;
    let mut any_hole = false;
    cond.walk(&mut |x| match x {
        Expr::Var(h) => match holes.get(*h) {
            Some(HoleKind::Scalar(_)) => any_hole = true,
            _ => ok = false,
        },
        Expr::Int { .. } | Expr::Float { .. } | Expr::Binary { .. } | Expr::Unary { .. } | Expr::Cast { .. } => {}
        _ => ok = false,
    });
    ok && any_hole
}

fn val_of(e: &Expr, m: &M) -> Option<V> {
    match e {
        Expr::Int { value, .. } => Some(V::I(*value)),
        Expr::Float { bits, double } => Some(V::F(if *double { f64::from_bits(*bits) } else { f32::from_bits(*bits as u32) as f64 })),
        Expr::Cast { e, ty } => {
            let v = val_of(e, m)?;
            Some(match (ty, v) {
                (Type::Float { .. }, V::I(x)) => V::F(x as f64),
                (Type::Int { .. } | Type::Long { .. } | Type::Char, V::F(x)) => V::I(x as i64),
                (Type::Bool, v) => V::I(v.truth() as i64),
                (_, v) => v,
            })
        }
        Expr::Var(h) => match m.b.get(*h)? {
            Some(Bind::Val(t)) => {
                let t = res(t, m.env.defs);
                if matches!(t, Expr::Var(_)) {
                    return None;
                }
                val_of_target(t)
            }
            _ => None,
        },
        Expr::Unary { op, e, .. } => {
            let v = val_of(e, m)?;
            Some(match (op, v) {
                (UnOp::Neg, V::I(x)) => V::I(x.wrapping_neg()),
                (UnOp::Neg, V::F(x)) => V::F(-x),
                (UnOp::BitNot, V::I(x)) => V::I(!x),
                (UnOp::Not, v) => V::I(!v.truth() as i64),
                _ => return None,
            })
        }
        Expr::Binary { op, l, r, .. } => {
            let (a, b) = (val_of(l, m)?, val_of(r, m)?);
            binop(*op, a, b)
        }
        _ => None,
    }
}

/// A target constant (literal, possibly cast).
fn val_of_target(t: &Expr) -> Option<V> {
    match t {
        Expr::Int { value, .. } => Some(V::I(*value)),
        Expr::Float { bits, double } => Some(V::F(if *double { f64::from_bits(*bits) } else { f32::from_bits(*bits as u32) as f64 })),
        Expr::Cast { e, .. } => val_of_target(e),
        _ => None,
    }
}

fn binop(op: BinOp, a: V, b: V) -> Option<V> {
    let ints = matches!((a, b), (V::I(_), V::I(_)));
    let (x, y) = (a.f(), b.f());
    let bi = |c: bool| Some(V::I(c as i64));
    match op {
        BinOp::Eq => bi(x == y),
        BinOp::Ne => bi(x != y),
        BinOp::Lt => bi(x < y),
        BinOp::Le => bi(x <= y),
        BinOp::Gt => bi(x > y),
        BinOp::Ge => bi(x >= y),
        BinOp::LogAnd => bi(a.truth() && b.truth()),
        BinOp::LogOr => bi(a.truth() || b.truth()),
        _ if !ints => match op {
            BinOp::Add => Some(V::F(x + y)),
            BinOp::Sub => Some(V::F(x - y)),
            BinOp::Mul => Some(V::F(x * y)),
            BinOp::Div if y != 0.0 => Some(V::F(x / y)),
            _ => None,
        },
        _ => {
            let (V::I(p), V::I(q)) = (a, b) else { return None };
            Some(V::I(match op {
                BinOp::Add => p.wrapping_add(q),
                BinOp::Sub => p.wrapping_sub(q),
                BinOp::Mul => p.wrapping_mul(q),
                BinOp::Div if q != 0 => p / q,
                BinOp::Rem if q != 0 => p % q,
                BinOp::And => p & q,
                BinOp::Or => p | q,
                BinOp::Xor => p ^ q,
                BinOp::Shl if (0..64).contains(&q) => p << q,
                BinOp::Shr if (0..64).contains(&q) => p >> q,
                _ => return None,
            }))
        }
    }
}

/// Bind unbound hole `h` to constant `v` (typed as the hole).
fn bind_const(m: &mut M, h: usize, v: i64) -> bool {
    if m.b.get(h).map_or(true, |b| b.is_some()) {
        return false;
    }
    let Some(HoleKind::Scalar(ty)) = m.t.holes.get(h) else { return false };
    let base = match ty {
        Type::Const(t) => (**t).clone(),
        t => t.clone(),
    };
    let lit = match &base {
        Type::Bool | Type::Int { .. } | Type::Long { .. } | Type::Char => Expr::Int { value: v, ty: base.clone() },
        // an enum parameter: the enumerator's value, cast
        Type::Named(_) => Expr::Cast { ty: base.clone(), e: Box::new(Expr::Int { value: v, ty: Type::Int { size: 4, signed: true } }) },
        _ => return false,
    };
    m.b[h] = Some(Bind::Val(lit));
    true
}

/// Make `cond` evaluate to `want` under the bindings of `m`: evaluate it when its holes are
/// bound to constants, else bind a free hole of a simple test (`h`, `!h`, `h == K`, `h != K`).
pub fn solve(cond: &Expr, want: bool, m: &mut M) -> bool {
    if let Some(v) = val_of(cond, m) {
        return v.truth() == want;
    }
    let lit = |e: &Expr| match e {
        Expr::Int { value, .. } => Some(*value),
        Expr::Cast { e, .. } => match &**e {
            Expr::Int { value, .. } => Some(*value),
            _ => None,
        },
        _ => None,
    };
    let hole = |e: &Expr| match e {
        Expr::Var(h) => Some(*h),
        Expr::Cast { e, .. } => match &**e {
            Expr::Var(h) => Some(*h),
            _ => None,
        },
        _ => None,
    };
    match cond {
        Expr::Var(h) => bind_const(m, *h, want as i64),
        Expr::Unary { op: UnOp::Not, e, .. } => solve(e, !want, m),
        Expr::Binary { op, l, r, .. } if op.is_cmp() => {
            // hole on the left: `h op K`; on the right: mirror the operator
            let (h, k, op) = match (hole(l), lit(r), hole(r), lit(l)) {
                (Some(h), Some(k), _, _) => (h, k, *op),
                (_, _, Some(h), Some(k)) => (h, k, match op {
                    BinOp::Lt => BinOp::Gt,
                    BinOp::Le => BinOp::Ge,
                    BinOp::Gt => BinOp::Lt,
                    BinOp::Ge => BinOp::Le,
                    o => *o,
                }),
                _ => return false,
            };
            // a value of the free hole that gives the wanted outcome (the hole appears nowhere
            // else in the matched code, so any such value compiles the same)
            let op = if want { op } else { op.negate_cmp().unwrap_or(op) };
            let is_bool = matches!(m.t.holes.get(h), Some(HoleKind::Scalar(Type::Bool)));
            let v = match op {
                BinOp::Eq => k,
                BinOp::Ne if is_bool => 1 - k.clamp(0, 1),
                BinOp::Ne => if k == 0 { 1 } else { 0 },
                BinOp::Lt => k - 1,
                BinOp::Le => k,
                BinOp::Gt => k + 1,
                BinOp::Ge => k,
                _ => return false,
            };
            bind_const(m, h, v) && val_of(cond, m).is_some_and(|x| x.truth() == want)
        }
        _ => false,
    }
}
