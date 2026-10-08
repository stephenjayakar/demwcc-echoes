//! Bitfield recovery (TypeDb-confirmed): MWCC reads bitfields with `lbz` + `rlwinm`/`extrwi` and
//! writes them with `lbz` + `rlwimi` + `stb`. Shift/mask expressions over a storage-unit access
//! become `Expr::BitField`, read-modify-write stores become assignments to it.

use crate::ir::*;
use crate::types;
use mwdec_core::{Type, TypeDb};
use std::collections::HashMap;

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
        // the rotate wraps: a mask inside the low `sh` bits takes the value's top bits
        None if sh > 0 && (m as u64) < (1u64 << sh) => {
            let v = Expr::cast(u(), args[1].clone());
            Expr::bin(BinOp::And, Expr::bin(BinOp::Shr, v, Expr::int(32 - sh as i64), u()), Expr::uint(m as i64), u())
        }
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
/// `sdk`: the SDK compiler's unit (GC/1.2.5n): also words kept across branches, values split
/// into bit pieces and words inserted whole (the SDK's `SET_REG_FIELD` sequences).
pub fn insert_chains(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>, sdk: bool) {
    insert_chains_inner(body, vars, is_temp);
    if sdk {
        hoist_inserted_words(body, vars, is_temp);
        join_chains(body, is_temp);
        split_field_values(body);
        setups_before_stores(body);
        shared_word_starts(body);
    }
}

/// See [`crate::variants::SDK_WORD_SHARED_INLINE`]: a single-assignment pure value whose every use
/// is in the start of a register word (`r = f(t)` followed by inserts into r), used twice or more.
fn shared_word_starts(body: &mut Vec<Stmt>) {
    let mut defs: HashMap<VarId, (usize, Expr)> = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| {
        for st in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(v), src } = st {
                let e = defs.entry(*v).or_insert((0, src.clone()));
                e.0 += 1;
            }
        }
    });
    let pure_val = |e: &Expr| {
        let mut ok = true;
        e.walk(&mut |x| ok &= matches!(x, Expr::Var(_) | Expr::Int { .. } | Expr::Binary { .. } | Expr::Cast { .. } | Expr::Unary { .. }));
        ok
    };
    // uses: (in word starts, elsewhere)
    let mut uses: HashMap<VarId, (usize, usize)> = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| {
        for (i, st) in b.iter().enumerate() {
            let start = matches!(st, Stmt::Assign { dst: Expr::Var(r), src } if !is_rlwimi(src)
                && matches!(b.get(i + 1), Some(Stmt::Assign { dst: Expr::Var(r2), src: s2 }) if r2 == r && is_rlwimi(s2)));
            let mut count = |e: &Expr, word: bool| {
                e.walk(&mut |x| {
                    if let Expr::Var(v) = x {
                        let u = uses.entry(*v).or_default();
                        if word { u.0 += 1 } else { u.1 += 1 }
                    }
                })
            };
            match st {
                Stmt::Assign { dst: Expr::Var(_), src } => count(src, start),
                Stmt::Assign { dst, src } => {
                    count(dst, false);
                    count(src, false)
                }
                Stmt::Expr(e) | Stmt::Return(Some(e)) => count(e, false),
                Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } | Stmt::For { cond, .. } => count(cond, false),
                Stmt::Switch { e, .. } => count(e, false),
                _ => {}
            }
        }
    });
    let cand: HashMap<VarId, Expr> = defs
        .into_iter()
        .filter(|(v, (n, src))| *n == 1 && pure_val(src) && !src.uses_var(*v) && uses.get(v).is_some_and(|(w, o)| *w >= 2 && *o == 0))
        .map(|(v, (_, src))| (v, src))
        .collect();
    if cand.is_empty() || !crate::variants::alt(crate::variants::SDK_WORD_SHARED_INLINE) {
        return;
    }
    Stmt::for_each_block_mut(body, &mut |b| b.retain(|st| !matches!(st, Stmt::Assign { dst: Expr::Var(v), .. } if cand.contains_key(v))));
    Stmt::rewrite_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            if let Some(d) = cand.get(v) {
                *e = d.clone();
            }
        }
    });
}

/// Register words are computed before they are written out: a pure setup run (`r = x;
/// r = __rlwimi(r, ..); ..`, no memory reads) moves up past the stores just before it (the
/// FIFO writes of the previous word) when they don't involve it.
fn setups_before_stores(body: &mut Vec<Stmt>) {
    fn pure(e: &Expr) -> bool {
        let mut m = true;
        e.walk(&mut |x| {
            if matches!(x, Expr::Load { .. } | Expr::Member { .. } | Expr::Index { .. } | Expr::Global { .. } | Expr::BitField { .. } | Expr::New { .. } | Expr::IncDec { .. }) || (matches!(x, Expr::Call { .. }) && !x.is_pure_call()) {
                m = false;
            }
        });
        m
    }
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            // a run: r = pure; r = __rlwimi(r, pure..) ..
            let Stmt::Assign { dst: Expr::Var(r), src } = &b[i] else {
                i += 1;
                continue;
            };
            let r = *r;
            if is_rlwimi(src) || !pure(src) || src.uses_var(r) {
                i += 1;
                continue;
            }
            let mut end = i + 1;
            while let Some(Stmt::Assign { dst: Expr::Var(r2), src }) = b.get(end) {
                if *r2 != r || !is_rlwimi(src) || !pure(src) || !matches!(chain_base(src), Expr::Var(x) if *x == r) {
                    break;
                }
                end += 1;
            }
            if end - i < 2 {
                i += 1;
                continue;
            }
            // operands of the run
            let mut ops: Vec<VarId> = vec![];
            for st in &b[i..end] {
                if let Stmt::Assign { src, .. } = st {
                    src.walk(&mut |x| {
                        if let Expr::Var(v) = x {
                            ops.push(*v);
                        }
                    });
                }
            }
            let mut at = i;
            while at > 0 {
                match &b[at - 1] {
                    Stmt::Assign { dst, src } if !matches!(dst, Expr::Var(_)) && !dst.uses_var(r) && !src.uses_var(r) && !src.has_call() && !dst.has_call() => at -= 1,
                    _ => break,
                }
            }
            // (nothing the run reads may be assigned in between: only stores were passed)
            let _ = &ops;
            if at < i {
                let run: Vec<Stmt> = b.drain(i..end).collect();
                let n = run.len();
                b.splice(at..at, run);
                i = at + n;
            } else {
                i = end;
            }
        }
    });
}

/// One variable whose different bits are inserted into several fields is split into its bit
/// pieces in the source
/// (`SET_REG_FIELD(reg, 1, 18, op & 1); SET_REG_FIELD(reg, 2, 20, (op >> 1) & 3);`): the masks
/// leave the instructions alone but not the register allocation.
fn split_field_values(body: &mut Vec<Stmt>) {
    fn var_of(e: &Expr) -> Option<VarId> {
        match e {
            Expr::Var(v) => Some(*v),
            Expr::Cast { e, ty } if matches!(strip_cv(ty), Type::Int { size: 4, .. }) => var_of(e),
            _ => None,
        }
    }
    // source bit of the field's lowest bit (the value's bits rotate left by `sh` into place)
    let low = |sh: i64, me: i64| ((31 - me) - sh).rem_euclid(32);
    let mut count: HashMap<VarId, Vec<(i64, i64, i64)>> = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Call { args, .. } = e {
            if is_rlwimi(e) {
                if let (Some(v), Some(sh), Some(mb), Some(me)) = (var_of(&args[1]), args[2].as_int(), args[3].as_int(), args[4].as_int()) {
                    let f = count.entry(v).or_default();
                    if !f.contains(&(sh, mb, me)) {
                        f.push((sh, mb, me));
                    }
                }
            }
        }
    });
    // pieces: different bits of the value (the same bits at different places are alternatives)
    count.retain(|_, f| {
        let mut lows: Vec<i64> = f.iter().map(|(sh, _, me)| low(*sh, *me)).collect();
        lows.sort();
        lows.dedup();
        lows.len() >= 2
    });
    if count.is_empty() {
        return;
    }
    Stmt::rewrite_exprs(body, &mut |e| {
        if !is_rlwimi(e) {
            return;
        }
        let Expr::Call { args, .. } = e else { return };
        let (Some(v), Some(sh), Some(mb), Some(me)) = (var_of(&args[1]), args[2].as_int(), args[3].as_int(), args[4].as_int()) else { return };
        if !count.contains_key(&v) || !(0..=31).contains(&mb) || !(mb..=31).contains(&me) {
            return;
        }
        let fs = 31 - me;
        let width = me - mb + 1;
        let d = low(sh, me);
        if width >= 32 || d + width > 32 {
            return;
        }
        // (the value's own signedness: a logical shift of an unsigned word)
        let unsigned = matches!(&args[1], Expr::Cast { ty, .. } if matches!(strip_cv(ty), Type::Int { signed: false, .. }));
        let ty = Type::Int { size: 4, signed: !unsigned };
        let mut x = args[1].clone();
        if d > 0 {
            x = Expr::bin(BinOp::Shr, x, Expr::int(d), ty.clone());
        }
        // (a shift that leaves only the field's bits needs no mask)
        if d + width < 32 {
            x = Expr::bin(BinOp::And, x, Expr::int((1i64 << width) - 1), ty);
        }
        args[1] = x;
        args[2] = Expr::int(fs);
    });
}

/// A register word built by inserts and then itself inserted into another word
/// (`t = (mode >> 1) & 1; __rlwimi(t, mode, 1, 30, 30); SET_REG_FIELD(gx->genMode, 2, 14, t);`)
/// is a variable set up first: `t = base; t = __rlwimi(t, ..); .. __rlwimi(w, t, ..)`.
fn hoist_inserted_words(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
    let base_id = vars.len();
    let mut new_vars: Vec<Var> = vec![];
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            // the first insert chain used as an inserted value in this statement
            let mut found: Option<Expr> = None;
            if let Stmt::Assign { src, .. } | Stmt::Expr(src) = &b[i] {
                src.walk(&mut |e| {
                    if found.is_none() && is_rlwimi(e) {
                        if let Expr::Call { args, .. } = e {
                            let v = match &args[1] {
                                Expr::Cast { e, .. } => &**e,
                                v => v,
                            };
                            if is_rlwimi(v) {
                                found = Some(v.clone());
                            }
                        }
                    }
                });
            }
            let Some(chain) = found else {
                i += 1;
                continue;
            };
            let r = base_id + new_vars.len();
            new_vars.push(Var { name: format!("field{}", new_vars.len() + 1), ty: Type::Int { size: 4, signed: false }, kind: VarKind::Local });
            let mut inserts = vec![];
            let mut cur = chain.clone();
            while is_rlwimi(&cur) {
                let Expr::Call { callee, mut args, ret } = cur else { unreachable!() };
                let inner = std::mem::replace(&mut args[0], Expr::Var(r));
                inserts.push(Expr::Call { callee, args, ret });
                cur = inner;
            }
            let mut setup = vec![Stmt::Assign { dst: Expr::Var(r), src: cur }];
            for ins in inserts.into_iter().rev() {
                setup.push(Stmt::Assign { dst: Expr::Var(r), src: ins });
            }
            if let Stmt::Assign { src, .. } | Stmt::Expr(src) = &mut b[i] {
                src.rewrite(&mut |e| {
                    if *e == chain {
                        *e = Expr::Var(r);
                    }
                });
            }
            let n = setup.len();
            b.splice(i..i, setup);
            i += n;
        }
    });
    is_temp.resize(base_id + new_vars.len(), false);
    vars.extend(new_vars);
}

fn is_rlwimi(e: &Expr) -> bool {
    matches!(e, Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. } if symbol == "__rlwimi" && args.len() == 5)
}

/// Innermost base of an insert chain.
fn chain_base(e: &Expr) -> &Expr {
    let mut cur = e;
    while let Expr::Call { args, .. } = cur {
        if !is_rlwimi(cur) {
            break;
        }
        cur = &args[0];
    }
    cur
}

/// The register word kept in one variable across a branch, as the SDK writes it
/// (`reg = gx->tevc[i]; SET_REG_FIELD(reg, ..); if (c) { SET_REG_FIELD(reg, ..); } else
/// { SET_REG_FIELD(reg, ..); } SET_REG_FIELD(reg, ..);`): the compiler gives each branch's
/// result its own register, so the lift sees `w = ins(ins(r, ..), ..)` in both arms and a copy
/// `t = w` before the next inserts. Both arms become in-place inserts on `r`, and `w` and the
/// copy `t` are `r`.
fn join_chains(body: &mut Vec<Stmt>, is_temp: &mut Vec<bool>) {
    let mut defs: HashMap<VarId, usize> = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| {
        for st in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(v), .. } = st {
                *defs.entry(*v).or_default() += 1;
            }
        }
    });
    let mut renames: HashMap<VarId, VarId> = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| {
        for i in 0..b.len() {
            let Stmt::If { then, els, .. } = &mut b[i] else { continue };
            let arm = |a: &Vec<Stmt>| -> Option<(VarId, VarId)> {
                let Some(Stmt::Assign { dst: Expr::Var(w), src }) = a.last() else { return None };
                if !is_rlwimi(src) {
                    return None;
                }
                let Expr::Var(r) = chain_base(src) else { return None };
                Some((*w, *r))
            };
            let (Some((w1, r1)), Some((w2, r2))) = (arm(then), arm(els)) else { continue };
            if w1 != w2 || r1 != r2 || w1 == r1 || defs.get(&w1) != Some(&2) {
                continue;
            }
            let (w, r) = (w1, r1);
            for a in [&mut *then, &mut *els] {
                let Some(Stmt::Assign { src, .. }) = a.pop() else { unreachable!() };
                let mut inserts = vec![];
                let mut cur = src;
                while is_rlwimi(&cur) {
                    let Expr::Call { callee, args, ret } = cur else { unreachable!() };
                    let mut args = args;
                    let inner = std::mem::replace(&mut args[0], Expr::Var(r));
                    inserts.push(Expr::Call { callee, args, ret });
                    cur = inner;
                }
                for ins in inserts.into_iter().rev() {
                    a.push(Stmt::Assign { dst: Expr::Var(r), src: ins });
                }
            }
            renames.insert(w, r);
        }
    });
    if renames.is_empty() {
        return;
    }
    let resolve = |mut v: VarId| {
        while let Some(&n) = renames.get(&v) {
            v = n;
        }
        v
    };
    Stmt::rewrite_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            *v = resolve(*v);
        }
    });
    // `t = r` followed by inserts into t, with r dead afterwards: t is r
    let roots: Vec<VarId> = renames.values().copied().collect();
    let mut more: HashMap<VarId, VarId> = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let copy = match &b[i] {
                Stmt::Assign { dst: Expr::Var(t), src: Expr::Var(r) } if roots.contains(r) && t != r => Some((*t, *r)),
                _ => None,
            };
            if let Some((t, r)) = copy {
                let mut later = HashMap::new();
                crate::inline::count_uses(&b[i + 1..], &mut later);
                let next_insert = matches!(b.get(i + 1), Some(Stmt::Assign { dst: Expr::Var(t2), src }) if *t2 == t && is_rlwimi(src) && matches!(chain_base(src), Expr::Var(x) if *x == t));
                if next_insert && later.get(&r).copied().unwrap_or(0) == 0 {
                    b.remove(i);
                    more.insert(t, r);
                    continue;
                }
            }
            i += 1;
        }
    });
    if !more.is_empty() {
        Stmt::rewrite_exprs(body, &mut |e| {
            if let Expr::Var(v) = e {
                if let Some(&n) = more.get(v) {
                    *v = n;
                }
            }
        });
    }
    for r in roots {
        if let Some(x) = is_temp.get_mut(r) {
            *x = false;
        }
    }
}

fn insert_chains_inner(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
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
