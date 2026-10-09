//! Integer sums written left to right.
//!
//! The lifter rebuilds a sum from the `add` tree the compiler emitted, which its reassociation
//! has reshaped: a source `a + b + c` can come out as `a + (b + c)`. Compiled again, the nested
//! form evaluates (and schedules) its operands in another order. Variant point
//! [`crate::variants::ARITH_LEFT_ASSOC`]: right-nested integer sums are flattened into
//! `a + b + c` (float sums keep their shape: the compiler does not reassociate them).
use crate::ir::{BinOp, Expr, Stmt};
use mwdec_core::Type;

fn int_type(t: &Type) -> bool {
    matches!(t, Type::Int { .. } | Type::Unknown { size: 4 })
}

fn right_nested(e: &Expr) -> bool {
    matches!(e, Expr::Binary { op: BinOp::Add, r, ty, .. } if int_type(ty) && matches!(&**r, Expr::Binary { op: BinOp::Add, ty: t2, .. } if t2 == ty))
}

/// Flatten right-nested integer sums `a + (b + c)` into `a + b + c` when the variant point is
/// flipped (asked only where such a sum exists).
pub fn left_assoc_sums(body: &mut [Stmt]) {
    let mut found = false;
    Stmt::rewrite_exprs(body, &mut |e| found |= right_nested(e));
    if !found || !crate::variants::alt(crate::variants::ARITH_LEFT_ASSOC) {
        return;
    }
    Stmt::rewrite_exprs(body, &mut |e| {
        // (post-order: the operands are flat already; rotate this node until its right operand is
        // no longer a sum of the same type)
        while right_nested(e) {
            let Expr::Binary { l, r, ty, .. } = std::mem::replace(e, Expr::Int { value: 0, ty: Type::Unknown { size: 4 } }) else { unreachable!() };
            let Expr::Binary { l: b, r: c, .. } = *r else { unreachable!() };
            let inner = Expr::Binary { op: BinOp::Add, l, r: b, ty: ty.clone() };
            *e = Expr::Binary { op: BinOp::Add, l: Box::new(inner), r: c, ty };
        }
    });
}

