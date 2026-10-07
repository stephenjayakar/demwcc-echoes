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
    forward_unit_temps(body, vars, db);
    recover_in(body, vars, db);
}

/// A storage unit loaded once into a temp, tested and written back (`if (mDirty) { mDirty =
/// false; ...}`: MWCC keeps the loaded byte for the read-modify-write): the temp's uses in the
/// following test and the statements before any call or other store read the unit itself, when
/// every one of them then is a bitfield access.
fn forward_unit_temps(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb) {
    let mut ndefs: std::collections::HashMap<VarId, usize> = Default::default();
    let mut uses: std::collections::HashMap<VarId, usize> = Default::default();
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(v), .. } = s {
                *ndefs.entry(*v).or_default() += 1;
            }
        }
    });
    crate::inline::count_uses(body, &mut uses);
    fn subst(e: &mut Expr, t: VarId, unit: &Expr, n: &mut usize) {
        e.rewrite(&mut |x| {
            if matches!(x, Expr::Var(v) if *v == t) {
                *x = unit.clone();
                *n += 1;
            }
        });
    }
    fn has_call(s: &Stmt) -> bool {
        let mut c = false;
        Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
            match e {
                // the insert intrinsic of a write-back
                Expr::Call { callee: Callee::Direct { symbol, .. }, .. } if symbol == "__rlwimi" => {}
                Expr::Call { .. } | Expr::New { .. } => c = true,
                _ => {}
            }
        });
        c
    }
    // statements of a block evaluated before anything could change the unit: plain variable
    // assignments, then at most the unit's own write-back
    fn prefix(b: &mut [Stmt], t: VarId, unit: &Expr, n: &mut usize) {
        for s in b.iter_mut() {
            if has_call(s) {
                return;
            }
            match s {
                Stmt::Assign { dst: Expr::Var(_), src } => subst(src, t, unit, n),
                Stmt::Assign { dst, src } if dst == unit => {
                    subst(src, t, unit, n);
                    return;
                }
                _ => return,
            }
        }
    }
    let raw_left = |s: &Stmt, unit: &Expr| -> bool {
        let mut all = 0;
        let mut ok = 0;
        Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
            if e == unit {
                all += 1;
            }
            if let Expr::BitField { base, .. } = e {
                if **base == *unit {
                    ok += 1;
                }
            }
        });
        all != ok
    };
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            let (t, unit) = match &b[i] {
                Stmt::Assign { dst: Expr::Var(t), src } if matches!(vars[*t].kind, VarKind::Local) && ndefs.get(t) == Some(&1) && unit_of(src, vars, db).is_some() && !src.has_call() => (*t, src.clone()),
                _ => {
                    i += 1;
                    continue;
                }
            };
            let mut next = b[i + 1].clone();
            let mut n = 0;
            match &mut next {
                Stmt::If { cond, then, els } => {
                    subst(cond, t, &unit, &mut n);
                    prefix(then, t, &unit, &mut n);
                    prefix(els, t, &unit, &mut n);
                }
                _ => {}
            }
            if n == 0 || n != uses.get(&t).copied().unwrap_or(0) {
                i += 1;
                continue;
            }
            let mut v = vec![next];
            recover_in(&mut v, vars, db);
            if raw_left(&v[0], &unit) {
                i += 1;
                continue;
            }
            b[i + 1] = v.pop().unwrap();
            b.remove(i);
        }
    });
}

/// `__rlwimi(dst, v, sh, mb, me)` spelled as `(dst & ~M) | ((v << sh) & M)`.
fn rlwimi_form(src: &Expr, dst: &Expr) -> Option<Expr> {
    let Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. } = src else { return None };
    if symbol != "__rlwimi" || args.len() != 5 || strip_cast(&args[0]) != dst {
        return None;
    }
    let (sh, mb, me) = (int_of(&args[2])?, int_of(&args[3])?, int_of(&args[4])?);
    if mb > me || me > 31 || sh > 31 {
        return None;
    }
    let m = (u32::MAX >> mb) & (u32::MAX << (31 - me));
    let u = || Type::Int { size: 4, signed: false };
    let keep = Expr::bin(BinOp::And, args[0].clone(), Expr::uint((!m) as i64), u());
    let y = match args[1].as_int() {
        Some(v) => Expr::uint((((v as u32) << sh) & m) as i64),
        None => {
            let shifted = if sh == 0 { args[1].clone() } else { Expr::bin(BinOp::Shl, args[1].clone(), Expr::int(sh as i64), u()) };
            Expr::bin(BinOp::And, shifted, Expr::uint(m as i64), u())
        }
    };
    Some(Expr::bin(BinOp::Or, keep, y, u()))
}

fn recover_in(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb) {
    // writes
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter_mut() {
            let Stmt::Assign { dst, src } = s else { continue };
            if unit_bits(dst).is_none() {
                continue;
            }
            let spelled;
            let src = match rlwimi_form(src, dst) {
                Some(e) => {
                    spelled = e;
                    &spelled
                }
                None => &*src,
            };
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

/// Field inserts (`__rlwimi`) chained through temps are one register variable updated in place,
/// the SDK's `reg = gx->cmode0; SET_REG_FIELD(reg, ...); SET_REG_FIELD(reg, ...);`:
/// `t1 = __rlwimi(x, a, ..); t2 = __rlwimi(__rlwimi(t1, b, ..), c, ..);` becomes
/// `t1 = x; t1 = __rlwimi(t1, a, ..); t1 = __rlwimi(t1, b, ..); t1 = __rlwimi(t1, c, ..);`.
pub fn insert_chains(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
    stored_chains(body, vars, is_temp);
    fn is_insert(e: &Expr) -> bool {
        matches!(e, Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. } if symbol == "__rlwimi" && args.len() == 5)
    }
    // (base, inserts innermost first)
    fn flatten(e: &Expr) -> (Expr, Vec<Expr>) {
        let mut inserts = vec![];
        let mut cur = e;
        while is_insert(cur) {
            inserts.push(cur.clone());
            let Expr::Call { args, .. } = cur else { unreachable!() };
            cur = &args[0];
        }
        inserts.reverse();
        (cur.clone(), inserts)
    }
    let mut uses = std::collections::HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    let mut chain: std::collections::HashMap<VarId, VarId> = Default::default();
    let mut renames: Vec<(VarId, VarId)> = vec![];
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let Stmt::Assign { dst: Expr::Var(t), src } = &b[i] else {
                i += 1;
                continue;
            };
            let t = *t;
            if !is_temp.get(t).copied().unwrap_or(false) || !is_insert(src) || chain.contains_key(&t) {
                i += 1;
                continue;
            }
            let (base, inserts) = flatten(src);
            let mut out = vec![];
            let r = match &base {
                Expr::Var(u) if chain.contains_key(u) && uses.get(u) == Some(&1) => chain[u],
                _ => {
                    out.push(Stmt::Assign { dst: Expr::Var(t), src: base.clone() });
                    t
                }
            };
            for ins in inserts {
                let Expr::Call { callee, mut args, ret } = ins else { unreachable!() };
                args[0] = Expr::Var(r);
                out.push(Stmt::Assign { dst: Expr::Var(r), src: Expr::Call { callee, args, ret } });
            }
            chain.insert(t, r);
            if r != t {
                renames.push((t, r));
            }
            let n = out.len();
            b.splice(i..i + 1, out);
            i += n;
        }
    });
    for (_, r) in &chain {
        if let Some(x) = is_temp.get_mut(*r) {
            *x = false;
        }
    }
    if !renames.is_empty() {
        let map: std::collections::HashMap<VarId, VarId> = renames.into_iter().collect();
        let resolve = |mut v: VarId| {
            while let Some(&n) = map.get(&v) {
                v = n;
            }
            v
        };
        Stmt::rewrite_exprs(body, &mut |e| {
            if let Expr::Var(v) = e {
                *v = resolve(*v);
            }
        });
    }
}

/// A register word built from a constant or with several inserts and stored right away
/// (`GXWGFifo.u32 = __rlwimi(__rlwimi(0, a, ..), b, ..)`) is a variable set up field by field
/// first (`reg = 0; reg = __rlwimi(reg, a, ..); ..; GXWGFifo.u32 = reg;`), computed before the
/// stores just ahead of it when it reads no memory.
fn stored_chains(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
    fn is_insert(e: &Expr) -> bool {
        matches!(e, Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. } if symbol == "__rlwimi" && args.len() == 5)
    }
    fn reads_memory(e: &Expr) -> bool {
        let mut m = false;
        e.walk(&mut |x| {
            if matches!(x, Expr::Load { .. } | Expr::Member { .. } | Expr::Index { .. } | Expr::Global { .. } | Expr::BitField { .. } | Expr::New { .. } | Expr::IncDec { .. })
                || (matches!(x, Expr::Call { .. }) && !x.is_pure_call())
            {
                m = true;
            }
        });
        m
    }
    let mut new_vars: Vec<Var> = vec![];
    let base_id = vars.len();
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let Stmt::Assign { dst, src } = &b[i] else {
                i += 1;
                continue;
            };
            if matches!(dst, Expr::Var(_)) {
                i += 1;
                continue;
            }
            let (outer_cast, chain) = match src {
                Expr::Cast { ty, e } if is_insert(e) => (Some(ty.clone()), (**e).clone()),
                e if is_insert(e) => (None, e.clone()),
                _ => {
                    i += 1;
                    continue;
                }
            };
            let mut inserts = vec![];
            let mut cur = &chain;
            while is_insert(cur) {
                inserts.push(cur.clone());
                let Expr::Call { args, .. } = cur else { unreachable!() };
                cur = &args[0];
            }
            let base = cur.clone();
            if !(inserts.len() >= 2 || base.as_int().is_some()) {
                i += 1;
                continue;
            }
            inserts.reverse();
            let r = base_id + new_vars.len();
            new_vars.push(Var { name: if new_vars.is_empty() { "reg".into() } else { format!("reg{}", new_vars.len() + 1) }, ty: Type::Int { size: 4, signed: false }, kind: VarKind::Local });
            let mut setup = vec![Stmt::Assign { dst: Expr::Var(r), src: base }];
            for ins in inserts {
                let Expr::Call { callee, mut args, ret } = ins else { unreachable!() };
                args[0] = Expr::Var(r);
                setup.push(Stmt::Assign { dst: Expr::Var(r), src: Expr::Call { callee, args, ret } });
            }
            let dst = dst.clone();
            let v = match outer_cast {
                Some(ty) => Expr::Cast { ty, e: Box::new(Expr::Var(r)) },
                None => Expr::Var(r),
            };
            b[i] = Stmt::Assign { dst, src: v };
            // ahead of the plain stores before it
            let mut at = i;
            if !reads_memory(&chain) {
                while at > 0 && matches!(&b[at - 1], Stmt::Assign { dst, src } if !matches!(dst, Expr::Var(_)) && !src.has_call() && !dst.has_call()) {
                    at -= 1;
                }
            }
            let n = setup.len();
            b.splice(at..at, setup);
            i += n + 1;
        }
    });
    is_temp.resize(base_id + new_vars.len(), false);
    vars.extend(new_vars);
}
