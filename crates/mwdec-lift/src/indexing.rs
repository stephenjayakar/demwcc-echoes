//! Array indexing recovery: MWCC turns `a[i].f` into byte arithmetic (`mulli`/`slwi` + `lwzx`) and,
//! in loops, into strength-reduced induction variables (`off += sizeof(T)`). Rebuild `a[i].f`.

use crate::ir::*;
use crate::types;
use mwdec_core::{Type, TypeDb};

fn uncast(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => uncast(e),
        e => e,
    }
}

/// `i * k` / `i << s` -> (i, k)
fn scaled(e: &Expr) -> Option<(Expr, i64)> {
    match uncast(e) {
        Expr::Binary { op: BinOp::Mul, l, r, .. } => match (r.as_int(), l.as_int()) {
            (Some(k), _) => Some(((**l).clone(), k)),
            (None, Some(k)) => Some(((**r).clone(), k)),
            _ => None,
        },
        Expr::Binary { op: BinOp::Shl, l, r, .. } => r.as_int().filter(|s| *s < 31).map(|s| ((**l).clone(), 1i64 << s)),
        _ => None,
    }
}

/// byte offset expression -> (index, scale, constant)
fn split_offset(e: &Expr) -> Option<(Expr, i64, i64)> {
    if let Some((i, k)) = scaled(e) {
        return Some((i, k, 0));
    }
    match uncast(e) {
        Expr::Binary { op: BinOp::Add, l, r, .. } => {
            if let Some(c) = r.as_int() {
                let (i, k, c2) = split_offset(l)?;
                return Some((i, k, c2 + c));
            }
            if let Some(c) = l.as_int() {
                let (i, k, c2) = split_offset(r)?;
                return Some((i, k, c2 + c));
            }
            None
        }
        _ => None,
    }
}

/// `(u8*)p + off` with `p: T*` and `off = i*sizeof(T) + c` -> (`p[i]` lvalue, c)
fn element(p: &Expr, off: &Expr, vars: &[Var], db: Option<&TypeDb>) -> Option<(Expr, i64)> {
    let pt = types::ty_of(p, vars);
    let t = pointee(&pt)?.clone();
    if matches!(strip_cv(&t), Type::Void | Type::Unknown { .. }) {
        return None;
    }
    let s = types::size_of(db, &t)? as i64;
    if s <= 1 {
        return None;
    }
    let (i, k, c) = split_offset(off)?;
    if k != s || c < 0 || c >= s {
        return None;
    }
    Some((Expr::Index { base: Box::new(p.clone()), index: Box::new(i), ty: t }, c))
}

/// Byte-pointer arithmetic `(u8*)p + off` in its two IR spellings.
fn byte_add_parts(e: &Expr) -> Option<(&Expr, &Expr)> {
    match e {
        Expr::Binary { op: BinOp::Add, l, r, .. } => match &**l {
            Expr::Cast { ty, e: p } if matches!(pointee(ty), Some(t) if scalar_size(t) == Some(1)) => Some((p, r)),
            _ => None,
        },
        Expr::AddrOf(inner) => match &**inner {
            Expr::Index { base, index, ty } if scalar_size(ty) == Some(1) => match &**base {
                Expr::Cast { ty: ct, e: p } if matches!(pointee(ct), Some(t) if scalar_size(t) == Some(1)) => Some((p, index)),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

fn rewrite(e: &mut Expr, vars: &[Var], db: Option<&TypeDb>) {
    let new = match &*e {
        Expr::Load { base, offset, ty } => byte_add_parts(base).and_then(|(p, off)| {
            let (el, c) = element(p, off, vars, db)?;
            let o = c as i32 + *offset;
            if o == 0 && types::size_of(db, &types::ty_of(&el, vars)) == scalar_size(ty) {
                return Some(el);
            }
            Some(Expr::Member { base: Box::new(el), offset: o, ty: ty.clone() })
        }),
        e2 @ (Expr::Binary { .. } | Expr::AddrOf(_)) => byte_add_parts(e2).and_then(|(p, off)| {
            let (el, c) = element(p, off, vars, db)?;
            if c == 0 {
                Some(Expr::AddrOf(Box::new(el)))
            } else {
                Some(Expr::AddrOf(Box::new(Expr::Member { base: Box::new(el), offset: c as i32, ty: Type::Unknown { size: 0 } })))
            }
        }),
        _ => None,
    };
    if let Some(n) = new {
        *e = n;
    }
}

pub fn recover(body: &mut [Stmt], vars: &[Var], db: Option<&TypeDb>) {
    Stmt::rewrite_exprs(body, &mut |e| rewrite(e, vars, db));
}

/// Scaled index of a raw byte offset: `i << s`, `i * 2^s`, `(i << s) & (0xff << s)` (an 8/16-bit
/// index), or a plain byte index (`k = 1`).
fn raw_scaled(e: &Expr) -> (Expr, i64) {
    match uncast(e) {
        Expr::Binary { op: BinOp::Shl, l, r, .. } => {
            if let Some(s) = r.as_int().filter(|s| (1..4).contains(s)) {
                return ((**l).clone(), 1i64 << s);
            }
        }
        Expr::Binary { op: BinOp::Mul, l, r, .. } => {
            if let Some(k) = r.as_int().filter(|k| matches!(k, 2 | 4 | 8)) {
                return ((**l).clone(), k);
            }
        }
        Expr::Binary { op: BinOp::And, l, r, .. } => {
            if let (Expr::Binary { op: BinOp::Shl, l: x, r: sh, .. }, Some(m)) = (uncast(l), r.as_int()) {
                if let Some(s) = sh.as_int().filter(|s| (1..4).contains(s)) {
                    for (w, sz) in [(0xffi64, 1u8), (0xffff, 2)] {
                        if m == w << s {
                            return (Expr::cast(t_int(sz, false), (**x).clone()), 1i64 << s);
                        }
                    }
                }
            }
        }
        _ => {}
    }
    (e.clone(), 1)
}

/// A raw access `*(T*)(((u8*)p + (i << s)) + c)` (the target adds the scaled index first and
/// folds `c` into the load displacement: `slwi; add; lwz c(rX)`) is spelled as an array element
/// `((T*)((u8*)p + c))[i]`: MWCC re-associates the byte-pointer spelling into `p + ((i << s) + c)`
/// (`addi; lwzx`). Element types the access does not match go through an integer array of the
/// scale's width (`*(T*)&((u16*)((u8*)p + c))[i]`).
pub fn raw_index(body: &mut [Stmt], vars: &[Var]) {
    Stmt::rewrite_exprs(body, &mut |e| {
        let Expr::Load { base, offset, ty } = &*e else { return };
        if *offset == 0 {
            // `slwi; addi c; lwzx` (index and constant summed before the indexed access): an
            // element of an array of the access type, `((T*)p)[(i << (s - log2 sz)) + c / sz]`
            let Some((p, off)) = byte_add_parts(base) else { return };
            let Some((i, k, c)) = split_offset(off) else { return };
            let Some(sz) = scalar_size(ty).filter(|z| matches!(z, 2 | 4 | 8)).map(|z| z as i64) else { return };
            if c <= 0 || c % sz != 0 || k <= sz || k % sz != 0 || !((k / sz) as u64).is_power_of_two() {
                return;
            }
            let ety = if matches!(strip_cv(ty), Type::Unknown { .. }) { t_int(sz as u8, true) } else { ty.clone() };
            let pt = types::ty_of(p, vars);
            if !is_ptr(&pt) && !matches!(strip_cv(&pt), Type::Int { size: 4, .. } | Type::Unknown { size: 4 }) {
                return;
            }
            let sh = (k / sz).trailing_zeros() as i64;
            let idx = Expr::bin(BinOp::Add, Expr::bin(BinOp::Shl, i, Expr::int(sh), t_s32()), Expr::int(c / sz), t_s32());
            *e = Expr::Index { base: Box::new(Expr::cast(t_ptr(ety.clone()), p.clone())), index: Box::new(idx), ty: ety };
            return;
        }
        let Some((p, off)) = byte_add_parts(base) else { return };
        if off.as_int().is_some() {
            return;
        }
        let pt = types::ty_of(p, vars);
        if !is_ptr(&pt) && !matches!(strip_cv(&pt), Type::Int { size: 4, .. } | Type::Unknown { size: 4 }) {
            return;
        }
        let (i, k) = raw_scaled(off);
        if i.as_int().is_some() {
            return;
        }
        let at = scalar_size(ty);
        let (ety, wrap) = if at == Some(k as u32) {
            (if matches!(strip_cv(ty), Type::Unknown { .. }) { t_int(k as u8, k == 4) } else { ty.clone() }, false)
        } else {
            (t_int(k as u8, false), true)
        };
        let row = Expr::AddrOf(Box::new(Expr::Index {
            base: Box::new(Expr::cast(t_ptr(t_int(1, false)), p.clone())),
            index: Box::new(Expr::int(*offset as i64)),
            ty: t_int(1, false),
        }));
        let el = Expr::Index { base: Box::new(Expr::cast(t_ptr(ety.clone()), row)), index: Box::new(i), ty: ety };
        *e = if wrap { Expr::Load { base: Box::new(Expr::AddrOf(Box::new(el))), offset: 0, ty: ty.clone() } } else { el };
    });
}

/// In `for (i = 0; i < n; i++)`: an offset variable `v = 0; ... v = v + K;` advancing in lockstep
/// with `i` is `i * K` (strength reduction undone).
pub fn undo_strength_reduction(body: &mut Vec<Stmt>, vars: &[Var]) {
    // (one offset per pass: a loop may step several, the counter's old copy among them)
    for _ in 0..4 {
        let mut changed = false;
        Stmt::for_each_block_mut(body, &mut |b| changed |= undo_one(b, vars));
        if !changed {
            break;
        }
    }
}

fn undo_one(b: &mut Vec<Stmt>, vars: &[Var]) -> bool {
    {
        for k in 0..b.len() {
            let Stmt::For { init, step, body: lb, .. } = &b[k] else { continue };
            let i = match (init.as_slice(), step.as_slice()) {
                ([Stmt::Assign { dst: Expr::Var(i), src }], [st]) if src.as_int() == Some(0) && is_inc(st, *i) => *i,
                // (the counter set to 0 before the loop)
                ([], [st @ (Stmt::Assign { dst: Expr::Var(_), .. } | Stmt::Expr(_))]) => {
                    let i = match st {
                        Stmt::Assign { dst: Expr::Var(i), .. } => *i,
                        Stmt::Expr(Expr::IncDec { e, .. }) => match &**e {
                            Expr::Var(i) => *i,
                            _ => continue,
                        },
                        _ => continue,
                    };
                    let zero = b[..k].iter().rposition(|s| crate::idioms::stmt_mentions(s, i)).is_some_and(|z| matches!(&b[z], Stmt::Assign { dst: Expr::Var(x), src } if *x == i && src.as_int() == Some(0)));
                    if !is_inc(st, i) || !zero {
                        continue;
                    }
                    i
                }
                _ => continue,
            };
            // candidates: the trailing steps of the body (or of its trailing arm) `v = v + K`
            fn step_of(s: &Stmt) -> Option<(VarId, i64)> {
                match s {
                    Stmt::Assign { dst: Expr::Var(v), src } => match uncast(src) {
                        Expr::Binary { op: BinOp::Add, l, r, .. } if matches!(uncast(l), Expr::Var(x) if x == v) => r.as_int().map(|k| (*v, k)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            fn trailing_steps(b: &[Stmt], out: &mut Vec<(VarId, i64)>) {
                let real: Vec<&Stmt> = b.iter().filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))).collect();
                for s in real.iter().rev() {
                    match step_of(s) {
                        Some(c) => out.push(c),
                        None => {
                            if out.is_empty() {
                                if let Stmt::If { then, els, .. } = s {
                                    trailing_steps(els, out);
                                    if out.is_empty() {
                                        trailing_steps(then, out);
                                    }
                                }
                            }
                            return;
                        }
                    }
                }
            }
            let mut cands = vec![];
            trailing_steps(lb, &mut cands);
            let found = cands.into_iter().find(|&(v, _)| {
                v != i
                    && vars[v].kind == VarKind::Local
                    && !is_ptr(&vars[v].ty)
                    && b[..k].iter().any(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src } if *x == v && src.as_int() == Some(0)))
            });
            let Some((v, kk)) = found else { continue };
            // init `v = 0` earlier in this list, v not used after the loop
            let Some(ini) = b[..k].iter().rposition(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src } if *x == v && src.as_int() == Some(0))) else { continue };
            let used_after = b[k + 1..].iter().any(|s| {
                let mut f = false;
                Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| f |= matches!(e, Expr::Var(x) if *x == v));
                f
            });
            if used_after {
                continue;
            }
            let mut assigns = 0;
            fn count(b: &[Stmt], v: VarId, n: &mut usize) {
                for s in b {
                    match s {
                        Stmt::Assign { dst: Expr::Var(x), .. } if *x == v => *n += 1,
                        Stmt::If { then, els, .. } => {
                            count(then, v, n);
                            count(els, v, n);
                        }
                        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => count(body, v, n),
                        _ => {}
                    }
                }
            }
            count(lb, v, &mut assigns);
            if assigns != 1 {
                continue;
            }
            // rewrite: drop the increment, replace v by i * K
            let repl = if kk == 1 { Expr::Var(i) } else { Expr::bin(BinOp::Mul, Expr::Var(i), Expr::int(kk), t_s32()) };
            if let Stmt::For { body: lb, .. } = &mut b[k] {
                fn drop_inc(b: &mut Vec<Stmt>, v: VarId) -> bool {
                    if let Some(at) = b.iter().rposition(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if *x == v)) {
                        b.remove(at);
                        return true;
                    }
                    let Some(at) = b.iter().rposition(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_))) else { return false };
                    match &mut b[at] {
                        Stmt::If { then, els, .. } => drop_inc(els, v) || drop_inc(then, v),
                        _ => false,
                    }
                }
                drop_inc(lb, v);
                Stmt::rewrite_exprs(lb, &mut |e| {
                    if matches!(e, Expr::Var(x) if *x == v) {
                        *e = repl.clone();
                    }
                });
            }
            b.remove(ini);
            return true;
        }
    }
    false
}

/// `p = p + K` (through casts, or `&*(p + K)`): K.
fn walk_step(src: &Expr, p: VarId) -> Option<i64> {
    match uncast(src) {
        Expr::Binary { op: BinOp::Add, l, r, .. } if matches!(uncast(l), Expr::Var(y) if *y == p) => r.as_int(),
        // `&p->field` (`&*(p + k)`)
        Expr::AddrOf(x) => match &**x {
            Expr::Load { base, offset, .. } if matches!(uncast(base), Expr::Var(y) if *y == p) => Some(*offset as i64),
            _ => None,
        },
        _ => None,
    }
}

/// `i = i + 1` / `i++`
fn is_inc(s: &Stmt, i: VarId) -> bool {
    match s {
        Stmt::Assign { dst: Expr::Var(x), src } if *x == i => matches!(uncast(src), Expr::Binary { op: BinOp::Add, l, r, .. } if matches!(uncast(l), Expr::Var(y) if *y == i) && r.as_int() == Some(1)),
        Stmt::Expr(Expr::IncDec { e, delta: 1, .. }) => matches!(**e, Expr::Var(x) if x == i),
        _ => false,
    }
}

fn assigns_var(b: &[Stmt], v: VarId) -> usize {
    let mut n = 0;
    for s in b {
        match s {
            Stmt::Assign { dst: Expr::Var(x), .. } if *x == v => n += 1,
            Stmt::Expr(Expr::IncDec { e, .. }) if matches!(**e, Expr::Var(x) if x == v) => n += 1,
            Stmt::If { then, els, .. } => n += assigns_var(then, v) + assigns_var(els, v),
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => n += assigns_var(body, v),
            Stmt::For { init, step, body, .. } => n += assigns_var(init, v) + assigns_var(step, v) + assigns_var(body, v),
            Stmt::Switch { cases, .. } => n += cases.iter().map(|c| assigns_var(&c.body, v)).sum::<usize>(),
            _ => {}
        }
    }
    n
}

/// A pointer walking an array in lockstep with a loop counter that starts at 0 (`p = a; i = 0;
/// do { .. *p ..; p = p + K; i = i + 1; } while (i < n);`, the compiler's strength reduction of
/// `a[i]`): its element reads are `a[i]` again (`((T*)a)[i]`, or members `a[i].f`), and the
/// pointer goes. Returns the walks rewritten as member accesses (for the array recovery).
pub fn pointer_walks(body: &mut Vec<Stmt>, vars: &[Var]) -> usize {
    let mut members = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        for k in 0..b.len() {
            // the loop: (counter, its body, the counter's step inside the body if any)
            let (i, lb, inc_at) = match &b[k] {
                Stmt::DoWhile { body: lb, cond } => {
                    let Expr::Binary { op: BinOp::Lt, l, .. } = uncast(cond) else { continue };
                    let Expr::Var(i) = uncast(l) else { continue };
                    let Some(at) = lb.iter().position(|s| is_inc(s, *i)) else { continue };
                    (*i, lb, Some(at))
                }
                Stmt::For { init, cond, step, body: lb } => {
                    let Expr::Binary { op: BinOp::Lt, l, .. } = uncast(cond) else { continue };
                    let Expr::Var(i) = uncast(l) else { continue };
                    if !matches!(step.as_slice(), [s] if is_inc(s, *i)) || !matches!(init.as_slice(), [Stmt::Assign { dst: Expr::Var(x), src }] if x == i && src.as_int() == Some(0)) {
                        continue;
                    }
                    (*i, lb, None)
                }
                _ => continue,
            };
            if assigns_var(lb, i) != inc_at.is_some() as usize {
                continue;
            }
            if inc_at.is_some() {
                let Some(ii) = b[..k].iter().rposition(|s| crate::idioms::stmt_mentions(s, i)) else { continue };
                if !matches!(&b[ii], Stmt::Assign { dst: Expr::Var(x), src } if *x == i && src.as_int() == Some(0)) {
                    continue;
                }
            }
            // a local stepped once at the top level of the body, after its last read
            let mut found = None;
            for (at, s) in lb.iter().enumerate() {
                let Stmt::Assign { dst: Expr::Var(p), src } = s else { continue };
                let Some(kk) = walk_step(src, *p) else { continue };
                let p = *p;
                if p == i || !matches!(vars[p].kind, VarKind::Local) || kk <= 0 || assigns_var(lb, p) != 1 || lb[at + 1..].iter().any(|s| crate::idioms::stmt_mentions(s, p)) {
                    continue;
                }
                // the counter steps after the pointer's reads
                if inc_at.map_or(false, |ia| lb[ia..].iter().any(|s| crate::idioms::stmt_mentions(s, p) && !matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if *x == p))) {
                    continue;
                }
                found = Some((p, at, kk));
                break;
            }
            let Some((p, at, kk)) = found else { continue };
            // set up from a base before the loop, not read after it
            let Some(pi) = b[..k].iter().rposition(|s| crate::idioms::stmt_mentions(s, p)) else { continue };
            let Stmt::Assign { dst: Expr::Var(x), src: base } = &b[pi] else { continue };
            if *x != p || base.uses_var(p) || b[k + 1..].iter().any(|s| crate::idioms::stmt_mentions(s, p)) {
                continue;
            }
            let mut base_vars = vec![];
            base.walk(&mut |e| {
                if let Expr::Var(v) = e {
                    base_vars.push(*v);
                }
            });
            if base.has_call() || base_vars.iter().any(|&v| assigns_var(&b[pi + 1..=k], v) > 0) {
                continue;
            }
            // every read of p is a load through it: K-byte elements read whole, or members of
            // K-byte elements (`p->field`)
            let mut reads = 0;
            let mut loads = 0;
            let mut elem: Option<Type> = None;
            let mut whole = true;
            let mut inside = true;
            Stmt::walk_exprs(&lb[..at], &mut |e| {
                if matches!(e, Expr::Var(y) if *y == p) {
                    reads += 1;
                }
                if let Expr::Load { base: lbase, offset, ty } = e {
                    if matches!(uncast(lbase), Expr::Var(y) if *y == p) {
                        loads += 1;
                        let sz = scalar_size(ty).unwrap_or(0) as i64;
                        if *offset != 0 || sz != kk {
                            whole = false;
                        }
                        if *offset < 0 || sz == 0 || *offset as i64 + sz > kk {
                            inside = false;
                        }
                        match &elem {
                            Some(t) if t != ty => whole = false,
                            _ => elem = Some(ty.clone()),
                        }
                    }
                }
            });
            let Some(ety) = elem else { continue };
            // (several members of one element per iteration: the source walked it through a
            // pointer or reference of its own, which the walk spells better)
            if reads != loads || !(whole || (inside && loads == 1)) {
                continue;
            }
            let base = base.clone();
            let (Stmt::DoWhile { body: lb, .. } | Stmt::For { body: lb, .. }) = &mut b[k] else { unreachable!() };
            lb.remove(at);
            if whole {
                let ety = if matches!(strip_cv(&ety), Type::Unknown { .. }) { t_int(kk as u8, true) } else { ety };
                let ebase = match &base {
                    Expr::AddrOf(g) if matches!(&**g, Expr::Global { ty: Type::Unknown { .. }, .. }) => base.clone(),
                    Expr::Global { ty: Type::Unknown { .. }, .. } => Expr::AddrOf(Box::new(base.clone())),
                    _ => Expr::cast(t_ptr(ety.clone()), base.clone()),
                };
                Stmt::rewrite_exprs(lb, &mut |e| {
                    if let Expr::Load { base: lbase, offset: 0, .. } = &*e {
                        if matches!(uncast(lbase), Expr::Var(y) if *y == p) {
                            *e = Expr::Index { base: Box::new(ebase.clone()), index: Box::new(Expr::Var(i)), ty: ety.clone() };
                        }
                    }
                });
            } else {
                // members: byte arithmetic from the object the walk starts in (`(u8*)P + i * K`
                // at the start's offset), which the array recovery turns into `P->arr[i].f`
                let (obj, c) = walk_origin(&base);
                let u8p = t_ptr(t_int(1, false));
                let at_i = Expr::bin(BinOp::Add, Expr::cast(u8p.clone(), obj), Expr::bin(BinOp::Mul, Expr::Var(i), Expr::int(kk), t_s32()), u8p);
                Stmt::rewrite_exprs(lb, &mut |e| {
                    if let Expr::Load { base: lbase, offset, ty } = &*e {
                        if matches!(uncast(lbase), Expr::Var(y) if *y == p) {
                            *e = Expr::Load { base: Box::new(at_i.clone()), offset: *offset + c, ty: ty.clone() };
                        }
                    }
                });
                members += 1;
            }
            b.remove(pi);
            return;
        }
    });
    members
}

/// The object and byte offset a walk starts at: `&P->x` (raw), `(u8*)P + c`, else the start itself.
fn walk_origin(base: &Expr) -> (Expr, i32) {
    match base {
        Expr::Cast { e, .. } => walk_origin(e),
        Expr::AddrOf(x) => match &**x {
            Expr::Load { base: p, offset, ty: Type::Unknown { size: 0 } } => ((**p).clone(), *offset),
            _ => (base.clone(), 0),
        },
        Expr::Binary { op: BinOp::Add, l, r, .. } => match (r.as_int(), &**l) {
            (Some(c), Expr::Cast { ty, e: p }) if matches!(pointee(ty), Some(t) if scalar_size(t) == Some(1)) => ((**p).clone(), c as i32),
            _ => (base.clone(), 0),
        },
        _ => (base.clone(), 0),
    }
}

/// Variant point [`crate::variants::LOOP_INVARIANT_READS`]: temps assigned just before a loop
/// from memory reads the loop doesn't change (`n = v.size(); x = id.value; for (..; i < n; ..)
/// if (a[i].id == x)`) and read once inside it are read there (`i < v.size()`, `a[i].id ==
/// id`): the compiler hoisted them. Returns the reads moved.
pub fn invariant_reads(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &[bool]) -> usize {
    let mut uses = std::collections::HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    let mut defs: std::collections::HashMap<VarId, usize> = Default::default();
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(v), .. } = s {
                *defs.entry(*v).or_default() += 1;
            }
        }
    });
    let pure = |e: &Expr| {
        let mut ok = !e.has_call();
        // (globals stay: a read of one can't be shown loop-invariant like a member's)
        e.walk(&mut |x| ok &= !matches!(x, Expr::IncDec { .. } | Expr::New { .. } | Expr::Global { .. }));
        ok && { let mut m = false; e.walk(&mut |x| m |= matches!(x, Expr::Load { .. } | Expr::Member { .. })); m }
    };
    let mut asked = false;
    let mut apply = false;
    let mut n = 0;
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut k = 0;
        while k < b.len() {
            if !matches!(&b[k], Stmt::For { .. }) {
                k += 1;
                continue;
            }
            let mut cands = vec![];
            let mut j = k;
            while j > 0 {
                let Stmt::Assign { dst: Expr::Var(t), src } = &b[j - 1] else { break };
                let t = *t;
                let mut src_vars = vec![];
                src.walk(&mut |x| {
                    if let Expr::Var(v) = x {
                        src_vars.push(*v);
                    }
                });
                let ok = is_temp.get(t).copied().unwrap_or(false)
                    && matches!(vars[t].kind, VarKind::Local)
                    && defs.get(&t) == Some(&1)
                    && uses.get(&t) == Some(&1)
                    && crate::idioms::stmt_mentions(&b[k], t)
                    && pure(src)
                    && !src.uses_var(t)
                    // nothing it reads changes between it and the loop or inside the loop
                    && src_vars.iter().all(|&v| assigns_var(&b[j..k], v) == 0 && assigns_var(std::slice::from_ref(&b[k]), v) == 0);
                if ok {
                    cands.push(j - 1);
                }
                j -= 1;
            }
            if cands.is_empty() {
                k += 1;
                continue;
            }
            if !asked {
                asked = true;
                apply = crate::variants::alt(crate::variants::LOOP_INVARIANT_READS);
            }
            if !apply {
                return;
            }
            // later temps first (indices stay valid)
            cands.sort_unstable();
            for &c in cands.iter().rev() {
                let Stmt::Assign { dst: Expr::Var(t), src } = b[c].clone() else { unreachable!() };
                b.remove(c);
                k -= 1;
                Stmt::rewrite_exprs(&mut b[k..k + 1], &mut |x| {
                    if matches!(x, Expr::Var(y) if *y == t) {
                        *x = src.clone();
                    }
                });
                n += 1;
            }
            k += 1;
        }
    });
    n
}

/// Undeclared word globals holding a pointer to rows of words (`FstStart`: a table of 12-byte
/// entries read as `*(int*)((u8*)g + i * 12 + 4)`): the global becomes `u32 (*g)[K / 4]` and the
/// reads `g[i][k]` (a byte-offset temp `t = i * 12` read only there goes). Every use of the global
/// must be such a word read inside a row.
pub fn pointer_global_rows(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&TypeDb>) {
    let declared = |s: &str| db.map_or(false, |d| d.globals.contains_key(s));
    // single-definition locals defined as `i * K` (the shared byte offset)
    let mut defs: std::collections::HashMap<VarId, (usize, Option<Expr>)> = Default::default();
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(v), src } = s {
                let e = defs.entry(*v).or_insert((0, None));
                e.0 += 1;
                e.1 = Some(src.clone());
            }
        }
    });
    let scaled_of = |e: &Expr| -> Option<(Expr, i64)> {
        match uncast(e) {
            Expr::Var(v) if matches!(vars[*v].kind, VarKind::Local) => match defs.get(v) {
                Some((1, Some(d))) => scaled(d),
                _ => None,
            },
            x => scaled(x),
        }
    };
    // (global, index, row size) of a row address `g + off` / `(u8*)g + off`
    let row = |b: &Expr| -> Option<(String, Expr, i64)> {
        let Expr::Binary { op: BinOp::Add, l, r, .. } = uncast(b) else { return None };
        let Expr::Global { symbol, ty } = uncast(l) else { return None };
        if declared(symbol) || scalar_size(ty) != Some(4) || is_ptr(ty) {
            return None;
        }
        let (i, k) = scaled_of(r)?;
        Some((symbol.clone(), i, k))
    };
    let mut rows: std::collections::HashMap<String, (i64, bool)> = Default::default();
    let mut reads: std::collections::HashMap<String, usize> = Default::default();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Global { symbol, .. } = e {
            *reads.entry(symbol.clone()).or_default() += 1;
        }
        if let Expr::Load { base, offset, ty } = e {
            if let Some((g, _, k)) = row(base) {
                let ok = k >= 8 && k % 4 == 0 && k <= 64 && *offset >= 0 && offset % 4 == 0 && (*offset as i64) + 4 <= k && scalar_size(ty) == Some(4) && !matches!(strip_cv(ty), Type::Float { .. });
                let r = rows.entry(g).or_insert((k, true));
                r.1 &= ok && r.0 == k;
            }
        }
    });
    // every mention of the global is the base of such a read
    let mut counted: std::collections::HashMap<String, usize> = Default::default();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Load { base, .. } = e {
            if let Some((g, _, _)) = row(base) {
                *counted.entry(g).or_default() += 1;
            }
        }
    });
    let chosen: std::collections::HashMap<String, i64> = rows.into_iter().filter(|(g, (_, ok))| *ok && counted.get(g) == reads.get(g)).map(|(g, (k, _))| (g, k)).collect();
    if chosen.is_empty() {
        return;
    }
    let u32t = Type::Int { size: 4, signed: false };
    let mut offset_temps = std::collections::HashSet::new();
    Stmt::rewrite_exprs(body, &mut |e| {
        let Expr::Load { base, offset, ty } = &*e else { return };
        let Some((g, i, k)) = row(base) else { return };
        if chosen.get(&g) != Some(&k) {
            return;
        }
        if let Expr::Binary { r, .. } = uncast(base) {
            if let Expr::Var(t) = uncast(r) {
                offset_temps.insert(*t);
            }
        }
        let rt = Type::Array(Box::new(u32t.clone()), (k / 4) as u32);
        let gl = Expr::Global { symbol: g, ty: Type::Ptr(Box::new(rt.clone())) };
        let r = Expr::Index { base: Box::new(gl), index: Box::new(i), ty: rt };
        let ety = if matches!(strip_cv(ty), Type::Unknown { .. }) { u32t.clone() } else { ty.clone() };
        *e = Expr::Index { base: Box::new(r), index: Box::new(Expr::int((*offset / 4) as i64)), ty: ety };
    });
    // byte-offset temps no longer read
    let mut uses = std::collections::HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    Stmt::for_each_block_mut(body, &mut |b| {
        b.retain(|s| match s {
            Stmt::Assign { dst: Expr::Var(v), src } => !(offset_temps.contains(v) && uses.get(v).copied().unwrap_or(0) == 0 && !src.has_call()),
            _ => true,
        });
    });
}
