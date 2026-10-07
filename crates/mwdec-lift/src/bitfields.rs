//! Bitfield recovery (TypeDb-confirmed): MWCC reads bitfields with `lbz` + `rlwinm`/`extrwi` and
//! writes them with `lbz` + `rlwimi` + `stb`. Shift/mask expressions over a storage-unit access
//! become `Expr::BitField`, read-modify-write stores become assignments to it.

use crate::ir::*;
use crate::types;
use mwdec_core::{Type, TypeDb};

fn unit_bits(e: &Expr) -> Option<u32> {
    match e {
        Expr::Load { ty, .. } | Expr::Member { ty, .. } => match strip_cv(ty) {
            Type::Int { size, .. } if *size <= 4 => Some(*size as u32 * 8),
            Type::Unknown { size } if *size <= 4 && *size > 0 => Some(size * 8),
            _ => None,
        },
        _ => None,
    }
}

/// (class, byte offset, unit size) of a storage-unit access.
fn unit_of(e: &Expr, vars: &[Var], db: &TypeDb) -> Option<(String, i32, u32)> {
    let bits = unit_bits(e)?;
    let (base, off, ptr) = match e {
        Expr::Load { base, offset, .. } => (&**base, *offset, true),
        Expr::Member { base, offset, .. } => (&**base, *offset, false),
        _ => return None,
    };
    let t = types::ty_of(base, vars);
    let t = if ptr { pointee(&t)?.clone() } else { t };
    let r = types::resolve(Some(db), &t).into_owned();
    Some((named(&r)?.to_string(), off, bits / 8))
}

fn int_of(e: &Expr) -> Option<u32> {
    e.as_int().map(|v| v as u32)
}

/// Contiguous mask -> (shift, width).
fn mask_range(m: u32) -> Option<(u8, u8)> {
    if m == 0 {
        return None;
    }
    let s = m.trailing_zeros();
    let w = (m >> s).trailing_ones();
    if (m >> s) >> w != 0 {
        return None;
    }
    Some((s as u8, w as u8))
}

fn make(l: &Expr, shift: u8, width: u8, vars: &[Var], db: &TypeDb) -> Option<Expr> {
    let (cls, off, size) = unit_of(l, vars, db)?;
    let mask = (((1u64 << width) - 1) << shift) as u32;
    let (_, ty) = types::bitfield_at(db, &cls, off, size, mask)?;
    Some(Expr::BitField { base: Box::new(l.clone()), shift, width, ty })
}

fn strip_cast(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip_cast(e),
        e => e,
    }
}

/// Rewrite a read pattern into a BitField if the DB has one there.
fn read(e: &Expr, vars: &[Var], db: &TypeDb) -> Option<Expr> {
    match e {
        Expr::Binary { op: BinOp::And, l, r, .. } => {
            let m = int_of(r)?;
            let (_, w) = mask_range(m)?;
            if m >> m.trailing_zeros() != m >> m.trailing_zeros() {
                return None;
            }
            match strip_cast(l) {
                Expr::Binary { op: BinOp::Shr, l: x, r: s, .. } => {
                    let s = int_of(s)? as u8;
                    if m.trailing_zeros() != 0 {
                        return None;
                    }
                    make(strip_cast(x), s, w, vars, db)
                }
                x if m.trailing_zeros() == 0 => make(x, 0, w, vars, db),
                _ => None,
            }
        }
        Expr::Binary { op: BinOp::Shr, l, r, .. } => {
            let x = strip_cast(l);
            let bits = unit_bits(x)?;
            let s = int_of(r)? as u8;
            if (s as u32) >= bits {
                return None;
            }
            make(x, s, bits as u8 - s, vars, db)
        }
        _ => None,
    }
}

/// `(x & M) == 0` with M a single field's mask: test the field.
fn masked_test(e: &Expr, vars: &[Var], db: &TypeDb) -> Option<Expr> {
    if let Expr::Binary { op: op @ (BinOp::Eq | BinOp::Ne), l, r, .. } = e {
        if r.as_int() != Some(0) {
            return None;
        }
        if let Expr::Binary { op: BinOp::And, l: x, r: m, .. } = strip_cast(l) {
            let m = int_of(m)?;
            let (s, w) = mask_range(m)?;
            if s == 0 {
                return None;
            }
            let bf = make(strip_cast(x), s, w, vars, db)?;
            return Some(Expr::cmp(*op, bf, Expr::int(0)));
        }
    }
    None
}

/// The value stored into the field by `L = (L & ~M) | Y`.
fn write_value(l: &Expr, y: &Expr, shift: u8, width: u8, vars: &[Var], db: &TypeDb) -> Option<Expr> {
    let m = (((1u64 << width) - 1) << shift) as u32;
    match y {
        Expr::Int { value, .. } => {
            let v = *value as u32;
            if v & !m != 0 {
                return None;
            }
            Some(Expr::int((v >> shift) as i64))
        }
        Expr::Binary { op: BinOp::And, l: e, r: mm, .. } if int_of(mm) == Some(m) => {
            match strip_cast(e) {
                Expr::Binary { op: BinOp::Shl, l: v, r: s, .. } if int_of(s) == Some(shift as u32) => Some((**v).clone()),
                other if shift == 0 => {
                    // value computed from the old field (e.g. `f | flags`)
                    let mut v = other.clone();
                    let bf = make(l, 0, width, vars, db)?;
                    v.rewrite(&mut |x| {
                        if let Expr::Binary { op: BinOp::And, l: a, r: b, .. } = x {
                            if **a == *l && int_of(b) == Some(m) {
                                *x = bf.clone();
                            }
                        }
                    });
                    Some(v)
                }
                _ => None,
            }
        }
        Expr::Binary { op: BinOp::Shl, l: v, r: s, .. } if int_of(s) == Some(shift as u32) && shift as u32 + width as u32 == unit_bits(l).unwrap_or(32) => {
            Some((**v).clone())
        }
        _ => None,
    }
}

pub fn recover(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb) {
    // writes
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter_mut() {
            let Stmt::Assign { dst, src } = s else { continue };
            if unit_bits(dst).is_none() {
                continue;
            }
            let Expr::Binary { op: BinOp::Or, l: keep, r: y, .. } = src else { continue };
            let Expr::Binary { op: BinOp::And, l: old, r: nm, .. } = &**keep else { continue };
            if strip_cast(old) != dst {
                continue;
            }
            let bits = unit_bits(dst).unwrap();
            let unit_mask = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
            let Some(nm) = int_of(nm) else { continue };
            let m = !nm & unit_mask;
            let Some((shift, width)) = mask_range(m) else { continue };
            let Some(bf) = make(dst, shift, width, vars, db) else { continue };
            let Some(v) = write_value(dst, y, shift, width, vars, db) else { continue };
            *s = Stmt::Assign { dst: bf, src: v };
        }
    });
    // reads (top-down so the widest pattern wins)
    let mut f = |e: &mut Expr| top_down(e, vars, db);
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter_mut() {
            match s {
                Stmt::Assign { dst, src } => {
                    if !matches!(dst, Expr::BitField { .. }) {
                        sub_reads(dst, &mut f);
                    } else if let Expr::BitField { base, .. } = dst {
                        sub_reads(base, &mut f);
                    }
                    f(src);
                }
                Stmt::Expr(e) | Stmt::Return(Some(e)) => f(e),
                Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } | Stmt::For { cond, .. } => f(cond),
                Stmt::Switch { e, .. } => f(e),
                _ => {}
            }
        }
    });
}

/// Rewrite the operands of an lvalue (not the lvalue itself).
fn sub_reads(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    match e {
        Expr::Load { base, .. } | Expr::Member { base, .. } => f(base),
        Expr::Index { base, index, .. } => {
            f(base);
            f(index);
        }
        _ => {}
    }
}

fn top_down(e: &mut Expr, vars: &[Var], db: &TypeDb) {
    if let Some(n) = masked_test(e, vars, db).or_else(|| read(e, vars, db)) {
        *e = n;
        if let Expr::Binary { l, .. } = e {
            if let Expr::BitField { base, .. } = &mut **l {
                sub_reads(base, &mut |x| top_down(x, vars, db));
            }
        } else if let Expr::BitField { base, .. } = e {
            sub_reads(base, &mut |x| top_down(x, vars, db));
        }
        return;
    }
    match e {
        Expr::AddrOf(x) | Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } => top_down(x, vars, db),
        Expr::Load { base, .. } | Expr::Member { base, .. } => top_down(base, vars, db),
        Expr::Index { base, index, .. } => {
            top_down(base, vars, db);
            top_down(index, vars, db);
        }
        Expr::Binary { l, r, .. } => {
            top_down(l, vars, db);
            top_down(r, vars, db);
        }
        Expr::Ternary { c, t, f, .. } => {
            top_down(c, vars, db);
            top_down(t, vars, db);
            top_down(f, vars, db);
        }
        Expr::Call { callee, args, .. } => {
            match callee {
                Callee::Method { this, .. } | Callee::Virtual { this, .. } => top_down(this, vars, db),
                Callee::Indirect(x) => top_down(x, vars, db),
                Callee::Direct { .. } => {}
            }
            for a in args {
                top_down(a, vars, db);
            }
        }
        Expr::New { placement, args, .. } => {
            for a in placement.iter_mut().chain(args.iter_mut()) {
                top_down(a, vars, db);
            }
        }
        Expr::Construct { args, .. } => {
            for a in args {
                top_down(a, vars, db);
            }
        }
        _ => {}
    }
}
