//! Source shapes the scheduler is sensitive to (from `mwcc-oracle why` on train functions):
//! - byte/halfword fields packed into a word are written as narrowing conversions
//!   (`(uchar)x << 16`), not masks (`x << 16 & 0xff0000`) (variant
//!   [`crate::variants::EXPR_BYTE_FIELDS`]);
//! - an extern only read is a constant (`static const uchar kHasColor = 2;` in the source): its
//!   loads carry no memory dependence on stores, which changes the scheduler's deadlines (variant
//!   [`crate::variants::CONST_READ_ONLY_EXTERNS`]).

use crate::ir::*;
use mwdec_core::Type;
use std::collections::BTreeMap;

fn byte_field(e: &Expr) -> Option<Expr> {
    let Expr::Binary { op: BinOp::And, l, r, .. } = e else { return None };
    let m = r.as_int()? as u32;
    let Expr::Binary { op: BinOp::Shl, l: x, r: s, ty } = &**l else { return None };
    let s = s.as_int()?;
    if !(1..=24).contains(&s) {
        return None;
    }
    let size = if m == 0xff << s { 1 } else if m == 0xffff << s && s <= 16 { 2 } else { return None };
    Some(Expr::bin(BinOp::Shl, Expr::cast(Type::Int { size, signed: false }, (**x).clone()), Expr::int(s), ty.clone()))
}

pub fn byte_fields(body: &mut Vec<Stmt>) {
    let mut any = false;
    Stmt::walk_exprs(body, &mut |e| any |= byte_field(e).is_some());
    if !any || !crate::variants::alt(crate::variants::EXPR_BYTE_FIELDS) {
        return;
    }
    Stmt::rewrite_exprs(body, &mut |e| {
        if let Some(n) = byte_field(e) {
            *e = n;
        }
    });
    // the top byte of such a pack needs no mask (`x << 24`): a conversion like its siblings
    fn has_byte(e: &Expr) -> bool {
        match e {
            Expr::Binary { op: BinOp::Or, l, r, .. } => has_byte(l) || has_byte(r),
            Expr::Binary { op: BinOp::Shl, l, .. } => matches!(&**l, Expr::Cast { ty: Type::Int { size: 1, signed: false }, .. }),
            Expr::Cast { ty: Type::Int { size: 1, signed: false }, .. } => true,
            _ => false,
        }
    }
    fn top_byte(e: &mut Expr) {
        match e {
            Expr::Binary { op: BinOp::Or, l, r, .. } => {
                top_byte(l);
                top_byte(r);
            }
            Expr::Binary { op: BinOp::Shl, l, r, .. } if r.as_int() == Some(24) && !matches!(&**l, Expr::Cast { ty: Type::Int { size: 1, .. }, .. }) => {
                let x = std::mem::replace(&mut **l, Expr::int(0));
                **l = Expr::cast(Type::Int { size: 1, signed: false }, x);
            }
            _ => {}
        }
    }
    Stmt::rewrite_exprs(body, &mut |e| {
        if matches!(e, Expr::Binary { op: BinOp::Or, .. }) && has_byte(e) {
            top_byte(e);
        }
    });
}

pub fn const_read_only_externs(body: &mut Vec<Stmt>, globals: &mut BTreeMap<String, GlobalRef>) {
    // externs (not defined here) of scalar type that the function never writes
    let mut written: Vec<String> = vec![];
    fn root(e: &Expr) -> Option<&str> {
        match e {
            Expr::Global { symbol, .. } => Some(symbol),
            Expr::Member { base, .. } | Expr::Index { base, .. } => root(base),
            _ => None,
        }
    }
    Stmt::for_each_block_mut(body, &mut |b| {
        for st in b.iter() {
            if let Stmt::Assign { dst, .. } = st {
                if let Some(s) = root(dst) {
                    written.push(s.to_string());
                }
            }
        }
    });
    let mut addressed: Vec<String> = vec![];
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::AddrOf(inner) = e {
            if let Some(s) = root(inner) {
                addressed.push(s.to_string());
            }
        }
    });
    // the scalar type each global is read as
    let mut read_ty: BTreeMap<String, Type> = BTreeMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Global { symbol, ty } = e {
            if matches!(strip_cv(ty), Type::Int { .. } | Type::Float { .. } | Type::Bool | Type::Char) {
                read_ty.entry(symbol.clone()).or_insert(ty.clone());
            }
        }
    });
    let cands: Vec<String> = globals
        .values()
        .filter(|g| !g.is_function && !g.local_def && g.abs_addr.is_none() && !written.contains(&g.symbol) && !addressed.contains(&g.symbol))
        .filter(|g| read_ty.contains_key(&g.symbol) && matches!(strip_cv(&g.ty), Type::Int { .. } | Type::Float { .. } | Type::Bool | Type::Char | Type::Unknown { .. }) && !matches!(g.ty, Type::Const(_)))
        .map(|g| g.symbol.clone())
        .collect();
    let mut used = false;
    Stmt::walk_exprs(body, &mut |e| used |= matches!(e, Expr::Global { symbol, .. } if cands.contains(symbol)));
    if !used || !crate::variants::alt(crate::variants::CONST_READ_ONLY_EXTERNS) {
        return;
    }
    for s in &cands {
        if let Some(g) = globals.get_mut(s) {
            g.ty = Type::Const(Box::new(read_ty[s].clone()));
        }
    }
    Stmt::rewrite_exprs(body, &mut |e| {
        if let Expr::Global { symbol, ty } = e {
            if cands.contains(symbol) && !matches!(ty, Type::Const(_)) {
                *ty = Type::Const(Box::new(ty.clone()));
            }
        }
    });
}

/// `k & ~(c ? -1 : 0)` is `c ? 0 : k`, `k & (c ? -1 : 0)` is `c ? k : 0` (the compiler's
/// branchless select of a value or zero; variant [`crate::variants::EXPR_MASK_SELECT`]).
fn mask_select(e: &Expr) -> Option<Expr> {
    let Expr::Binary { op: BinOp::And, l, r, ty } = e else { return None };
    let all_ones = |t: &Expr| t.as_int() == Some(-1) || t.as_int().is_some_and(|v| v as u32 == u32::MAX);
    let sel = |m: &Expr| -> Option<(Expr, bool)> {
        let (neg, m) = match m {
            Expr::Unary { op: UnOp::BitNot, e, .. } => (true, &**e),
            m => (false, m),
        };
        match m {
            Expr::Ternary { c, t, f, .. } if all_ones(t) && f.as_int() == Some(0) => Some(((**c).clone(), neg)),
            _ => None,
        }
    };
    for (k, m) in [(l, r), (r, l)] {
        if let Some((c, neg)) = sel(m) {
            let zero = Expr::Int { value: 0, ty: ty.clone() };
            let (t, f) = if neg { (zero, (**k).clone()) } else { ((**k).clone(), zero) };
            return Some(Expr::Ternary { c: Box::new(c), t: Box::new(t), f: Box::new(f), ty: ty.clone() });
        }
    }
    None
}

pub fn mask_selects(body: &mut Vec<Stmt>) {
    let mut any = false;
    Stmt::walk_exprs(body, &mut |e| any |= mask_select(e).is_some());
    if !any || !crate::variants::alt(crate::variants::EXPR_MASK_SELECT) {
        return;
    }
    Stmt::rewrite_exprs(body, &mut |e| {
        if let Some(n) = mask_select(e) {
            *e = n;
        }
    });
}
