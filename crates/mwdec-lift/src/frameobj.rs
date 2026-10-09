//! Frame objects that only carry a call's by-value result to a single read.
//!
//! `x = obj->VGetTime().GetSeconds();` with a virtual (or otherwise unsized) by-value return:
//! MWCC passes a frame temporary as the struct-return pointer and reads one member back. The
//! lifter sees an untyped frame region written whole by the call and read once; declaring the
//! region as a named object adds a copy (`T t = f();` is a temporary plus a copy), so the read
//! moves to the call: `tmp = *(float*)&obj->VGetTime();`. Moving the read earlier is safe: only
//! the call writes the region and its address goes nowhere else.

use crate::ir::*;
use mwdec_core::Type;

pub fn fold_single_reads(ir: &mut IrFunction) {
    let n = ir.vars.len();
    for v in 0..n {
        let VarKind::Stack { offset, .. } = ir.vars[v].kind else { continue };
        if !matches!(ir.vars[v].ty, Type::Unknown { .. }) {
            continue;
        }
        // every mention: one whole-object call store, one scalar member read, nothing else
        let (mut total, mut reads, mut addressed) = (0usize, 0usize, false);
        let mut read_ty: Option<(i32, Type)> = None;
        Stmt::walk_exprs(&ir.body, &mut |e| match e {
            Expr::Var(x) if *x == v => total += 1,
            Expr::Member { base, offset: o, ty } if matches!(&**base, Expr::Var(x) if *x == v) => {
                reads += 1;
                read_ty = Some((*o, ty.clone()));
            }
            Expr::AddrOf(x) if matches!(&**x, Expr::Var(w) if *w == v) => addressed = true,
            _ => {}
        });
        let Some((roff, rty)) = read_ty else { continue };
        if addressed || reads != 1 || total != 2 || matches!(rty, Type::Named(_) | Type::Array(..) | Type::Unknown { .. }) {
            continue;
        }
        // the store and the read in the same statement list, store first
        let t = ir.vars.len();
        let mut done = false;
        Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
            if done {
                return;
            }
            let Some(si) = b.iter().position(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src: Expr::Call { .. } } if *x == v)) else { return };
            let read_at = b.iter().enumerate().skip(si + 1).find(|(_, s)| {
                let mut hit = false;
                Stmt::walk_exprs(std::slice::from_ref(*s), &mut |e| {
                    if matches!(e, Expr::Member { base, .. } if matches!(&**base, Expr::Var(x) if *x == v)) {
                        hit = true;
                    }
                });
                hit
            });
            let Some((ri, _)) = read_at else { return };
            let Stmt::Assign { src: call, .. } = b[si].clone() else { return };
            b[si] = Stmt::Assign { dst: Expr::Var(t), src: Expr::Member { base: Box::new(call), offset: roff, ty: rty.clone() } };
            Stmt::rewrite_exprs(&mut b[ri..=ri], &mut |e| {
                if matches!(e, Expr::Member { base, .. } if matches!(&**base, Expr::Var(x) if *x == v)) {
                    *e = Expr::Var(t);
                }
            });
            done = true;
        });
        if done {
            ir.vars.push(Var { name: format!("result_{offset:x}"), ty: rty, kind: VarKind::Local });
        }
    }
}

/// `p = &dst - 4; q = &src - 4; for (i = 0; i < n; i++) { t = q[2]; p[1] = q[1]; p[2] = t;
/// p += 2; q += 2; } [p[1] = q[1];]`: MWCC's copy loop for an object of more than 64 bytes
/// (`lwzu`/`stwu` with a pre-decremented pointer and a counter). It is the assignment
/// `dst = src` of an object of `8n` (or `8n + 4`) bytes.
pub fn fold_block_copies(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let vars = ir.vars.clone();
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut k = 0;
        while k + 2 < b.len() {
            if let Some((n, s)) = block_copy_at(b, k, &vars, db) {
                b.splice(k..k + n, std::iter::once(s));
            }
            k += 1;
        }
    });
}

/// (base object, pointer var) of `p = &obj.@-4`.
fn pre_decremented(s: &Stmt) -> Option<(VarId, Expr)> {
    let Stmt::Assign { dst: Expr::Var(p), src: Expr::AddrOf(x) } = s else { return None };
    let Expr::Member { base, offset: -4, .. } = &**x else { return None };
    Some((*p, (**base).clone()))
}

fn ptr_load(e: &Expr, p: VarId) -> Option<i32> {
    match e {
        Expr::Load { base, offset, .. } if matches!(&**base, Expr::Var(x) if *x == p) => Some(*offset),
        _ => None,
    }
}

fn block_copy_at(b: &[Stmt], k: usize, vars: &[Var], db: Option<&mwdec_core::TypeDb>) -> Option<(usize, Stmt)> {
    let (a, ea) = pre_decremented(&b[k])?;
    let (c, ec) = pre_decremented(&b[k + 1])?;
    let Stmt::For { init, cond, step, body } = &b[k + 2] else { return None };
    let Expr::Binary { op: BinOp::Lt, l, r, .. } = cond else { return None };
    let (Expr::Var(i), Some(n)) = (&**l, r.as_int()) else { return None };
    if !matches!(init.as_slice(), [Stmt::Assign { dst: Expr::Var(x), src }] if x == i && src.as_int() == Some(0)) || step.len() != 1 || n <= 0 {
        return None;
    }
    // which pointer is written: the destination
    let mut stores: Vec<(VarId, i32)> = vec![];
    let mut temps: std::collections::HashMap<VarId, (VarId, i32)> = std::collections::HashMap::new();
    let mut incs = 0;
    let mut copies: Vec<(VarId, i32, VarId, i32)> = vec![];
    for s in body {
        let Stmt::Assign { dst, src } = s else { return None };
        match (dst, src) {
            (Expr::Var(t), x) if ptr_load(x, a).is_some() || ptr_load(x, c).is_some() => {
                let (pv, o) = if let Some(o) = ptr_load(x, a) { (a, o) } else { (c, ptr_load(x, c)?) };
                temps.insert(*t, (pv, o));
            }
            (Expr::Var(p), Expr::AddrOf(x)) if (*p == a || *p == c) && ptr_load(x, *p) == Some(8) => incs += 1,
            (d, x) => {
                let (dp, doff) = if let Some(o) = ptr_load(d, a) { (a, o) } else { (c, ptr_load(d, c)?) };
                let (sp, soff) = match x {
                    Expr::Var(t) => *temps.get(t)?,
                    x => {
                        if let Some(o) = ptr_load(x, a) {
                            (a, o)
                        } else {
                            (c, ptr_load(x, c)?)
                        }
                    }
                };
                stores.push((dp, doff));
                copies.push((dp, doff, sp, soff));
            }
        }
    }
    if incs != 2 || copies.len() != 2 || !copies.iter().all(|&(dp, doff, sp, soff)| dp != sp && doff == soff && (doff == 4 || doff == 8)) || copies[0].1 == copies[1].1 || copies[0].0 != copies[1].0 {
        return None;
    }
    let (dst_p, src_p) = (copies[0].0, copies[0].2);
    let (dst_obj, src_obj) = if dst_p == a { (ea, ec) } else { (ec, ea) };
    // the tail word
    let mut size = 8 * n as u32;
    let mut used = 3;
    if let Some(Stmt::Assign { dst, src }) = b.get(k + 3) {
        if ptr_load(dst, dst_p) == Some(4) && ptr_load(src, src_p) == Some(4) {
            size += 4;
            used = 4;
        }
    }
    // the pointers and the counter are dead afterwards
    let mut later = false;
    Stmt::walk_exprs(&b[k + used..], &mut |e| {
        if matches!(e, Expr::Var(x) if *x == a || *x == c || x == i) {
            later = true;
        }
    });
    if later {
        return None;
    }
    // the copied object: the destination's type when it has that size (its assignment then
    // copies one block; a struct of many members would call its `operator=`), else the source's
    let obj_ty = |e: &Expr| -> Type {
        let t = crate::types::ty_of(e, vars);
        match strip_cv(&t) {
            Type::Ref(x) => strip_cv(x).clone(),
            t => t.clone(),
        }
    };
    let (dty, sty) = (obj_ty(&dst_obj), obj_ty(&src_obj));
    let fits = |t: &Type| named(t).is_some() && crate::types::size_of(db, t) == Some(size);
    let t = if fits(&dty) {
        dty.clone()
    } else if fits(&sty) {
        sty.clone()
    } else {
        return None;
    };
    let whole = |e: Expr, et: &Type| if *et == t { e } else { Expr::Member { base: Box::new(e), offset: 0, ty: t.clone() } };
    Some((used, Stmt::Assign { dst: whole(dst_obj, &dty), src: whole(src_obj, &sty) }))
}

/// A frame object of a class whose whole size is one scalar store (an empty tag struct, a
/// one-word class) written from another object's bytes: the object copied whole,
/// `t = *(T*)&src`, so that it can become a copy `T(src)` where it is passed.
pub fn whole_object_copies(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let vars = ir.vars.clone();
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    crate::inline::count_uses(&ir.body, &mut uses);
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let Stmt::Assign { dst: Expr::Member { base, offset: 0, ty: dt }, src } = &b[i] else {
                i += 1;
                continue;
            };
            let Expr::Var(v) = **base else {
                i += 1;
                continue;
            };
            let t = vars[v].ty.clone();
            let whole = matches!(vars[v].kind, VarKind::Stack { .. })
                && named(&t).is_some()
                && crate::types::is_aggregate(Some(db), &t)
                && crate::types::size_of(Some(db), &t).is_some_and(|n| scalar_size(dt) == Some(n));
            if !whole {
                i += 1;
                continue;
            }
            // the bytes come from a global (directly or through a register loaded just before)
            let (from, def_at) = match src {
                Expr::Var(r) if uses.get(r) == Some(&1) => match b[..i].iter().rposition(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if x == r)) {
                    Some(k) => match &b[k] {
                        Stmt::Assign { src: g @ Expr::Global { .. }, .. } => (g.clone(), Some(k)),
                        _ => {
                            i += 1;
                            continue;
                        }
                    },
                    None => {
                        i += 1;
                        continue;
                    }
                },
                g @ Expr::Global { .. } => (g.clone(), None),
                _ => {
                    i += 1;
                    continue;
                }
            };
            if scalar_size(&crate::types::ty_of(&from, &vars)) != scalar_size(dt) {
                i += 1;
                continue;
            }
            b[i] = Stmt::Assign { dst: Expr::Var(v), src: Expr::Member { base: Box::new(from), offset: 0, ty: t } };
            if let Some(k) = def_at {
                b.remove(k);
                continue;
            }
            i += 1;
        }
    });
}

/// A call to a function without a known signature (no header declaration, no mangled name)
/// passing the address of a frame object that was just filled whole from a class member
/// (`stack = *(this+8); fn_80037944(&mgr, &stack)`): the callee takes that class by value and
/// the frame object is the compiler's argument copy. The member is passed (`fn(mgr, this->mId)`)
/// and the stand-in declaration takes the class; when a dead frame store held the same value
/// first (an inline accessor returning the class by value), an explicit copy `T(x)`.
pub fn unknown_callee_byval(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let vars = ir.vars.clone();
    // mentions of every variable (the store's destination included)
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::Var(v) = e {
            *uses.entry(*v).or_default() += 1;
        }
    });
    let mut dead = std::mem::take(&mut ir.dead_stores);
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            let Some((v, obj, t)) = whole_member_store(&b[i], &vars, db) else {
                i += 1;
                continue;
            };
            if uses.get(&v) != Some(&2) {
                i += 1;
                continue;
            }
            let mut hit = false;
            if !copyable(db, &t) {
                i += 1;
                continue;
            }
            let size = crate::types::size_of(Some(db), &t).unwrap_or(0);
            let copy = match dead.iter().position(|d| d.size == size && same_place(&d.value, &obj)) {
                Some(k) => {
                    dead.remove(k);
                    Expr::Construct { class: t.clone(), ctor: None, args: vec![obj.clone()] }
                }
                None => obj.clone(),
            };
            let mut next = b[i + 1].clone();
            Stmt::rewrite_exprs(std::slice::from_mut(&mut next), &mut |e| {
                let Expr::Call { callee: Callee::Direct { symbol, sig }, args, .. } = e else { return };
                // (a C name the headers don't declare: the signature is the call site's guess)
                if crate::sig::demangle(symbol).is_some() || db.decls.contains_key(symbol.as_str()) || db.decls.contains_key(&format!("::{symbol}")) || db.functions.contains_key(symbol.as_str()) {
                    return;
                }
                for (n, a) in args.iter_mut().enumerate() {
                    if matches!(a, Expr::AddrOf(x) if matches!(**x, Expr::Var(w) if w == v)) {
                        *a = copy.clone();
                        if let Some(p) = sig.params.get_mut(n) {
                            p.ty = t.clone();
                        }
                        hit = true;
                    }
                }
            });
            if hit {
                b[i + 1] = next;
                b.remove(i);
                continue;
            }
            i += 1;
        }
    });
    ir.dead_stores = dead;
}

/// A function no declaration describes that only copies whole word objects its parameters point
/// at into frame objects and passes those to a call no declaration describes either
/// (`stack_c = *arg0; stack_8 = *arg1; fn(&stack_c, &stack_8);`): both take the objects by value
/// (MWCC passes a class by value as a pointer to a copy), `fn(a, b)` with word-object parameters.
/// A declared callee taking a one-word class by value gives the parameters that class.
pub fn forwarded_byval_params(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let declared = |sym: &str| crate::sig::demangle(sym).is_some() || db.is_some_and(|db| db.decls.contains_key(sym) || db.decls.contains_key(&format!("::{sym}")) || db.functions.contains_key(sym));
    if declared(&ir.symbol) || ir.this_var.is_some() {
        return;
    }
    let real: Vec<&Stmt> = ir.body.iter().filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_) | Stmt::Return(None))).collect();
    let Some((Stmt::Expr(call), copies)) = real.split_last() else { return };
    let Expr::Call { callee: Callee::Direct { symbol, sig: csig }, args, .. } = call else { return };
    if copies.is_empty() {
        return;
    }
    // a declared callee: its by-value one-word class parameters
    let callee_declared = declared(symbol);
    let class_param = |n: usize| -> Option<Type> {
        let t = &csig.params.get(n)?.ty;
        // (a template instance the context hasn't sized: the 4-byte frame copy says one word)
        (named(strip_cv(t)).is_some() && matches!(crate::types::size_of(db, t), Some(0 | 4) | None)).then(|| t.clone())
    };
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::Var(v) = e {
            *uses.entry(*v).or_default() += 1;
        }
    });
    let is_param = |v: VarId| matches!(ir.vars[v].kind, VarKind::Param { .. });
    // temps holding a parameter's word, frame objects filled with one: frame -> parameter
    let mut temps: std::collections::HashMap<VarId, VarId> = std::collections::HashMap::new();
    let mut frames: std::collections::HashMap<VarId, VarId> = std::collections::HashMap::new();
    let word_of = |e: &Expr, temps: &std::collections::HashMap<VarId, VarId>| -> Option<VarId> {
        let e = match e {
            Expr::Cast { e, .. } => &**e,
            e => e,
        };
        match e {
            Expr::Load { base, offset: 0, ty } if scalar_size(ty) == Some(4) => match **base {
                Expr::Var(p) if is_param(p) && uses.get(&p) == Some(&1) => Some(p),
                _ => None,
            },
            Expr::Var(t) => temps.get(t).copied(),
            _ => None,
        }
    };
    for s in copies {
        let Stmt::Assign { dst, src } = s else { return };
        let Some(p) = word_of(src, &temps) else { return };
        match dst {
            Expr::Var(t) if matches!(ir.vars[*t].kind, VarKind::Local) && ir.vars[*t].name.starts_with("temp_") && uses.get(t) == Some(&2) => {
                temps.insert(*t, p);
            }
            Expr::Member { base, offset: 0, ty } if scalar_size(ty) == Some(4) => match **base {
                Expr::Var(f) if matches!(ir.vars[f].kind, VarKind::Stack { size: 4, .. }) && uses.get(&f) == Some(&2) => {
                    frames.insert(f, p);
                }
                _ => return,
            },
            _ => return,
        }
    }
    if frames.is_empty() || frames.len() + temps.len() != copies.len() {
        return;
    }
    let words = crate::helpers::words(4);
    let mut call = call.clone();
    let mut ptys: Vec<(VarId, Type)> = vec![];
    let Expr::Call { callee: Callee::Direct { sig, .. }, args: cargs, .. } = &mut call else { return };
    let mut n_hit = 0;
    for (n, a) in cargs.iter_mut().enumerate() {
        // (the frame object itself when a declared callee takes it by value)
        let f = match a {
            Expr::AddrOf(x) => match **x {
                Expr::Var(f) => f,
                _ => continue,
            },
            Expr::Var(f) if callee_declared => *f,
            _ => continue,
        };
        let Some(&p) = frames.get(&f) else { continue };
        let ty = if callee_declared {
            match class_param(n) {
                Some(t) => t,
                None => return,
            }
        } else {
            words.clone()
        };
        *a = Expr::Var(p);
        if !callee_declared {
            if let Some(sp) = sig.params.get_mut(n) {
                sp.ty = ty.clone();
            }
        }
        ptys.push((p, ty));
        n_hit += 1;
    }
    if n_hit != frames.len() || args.len() != cargs.len() {
        return;
    }
    for (p, ty) in ptys {
        ir.vars[p].ty = ty.clone();
        if let VarKind::Param { index } = ir.vars[p].kind {
            if let Some(sp) = ir.sig.params.get_mut(index) {
                sp.ty = ty.clone();
            }
        }
    }
    ir.body = vec![Stmt::Expr(call)];
}

/// Name of the helper of [`inline_by_value_helper`].
pub const INLINE_HELPER: &str = "mwdec_inline_body";

/// An undeclared function whose every parameter is read once, as a word (`*arg0`, `*arg1`),
/// leaving frame stores of exactly those words that nothing reads: the source passed its by-value
/// one-word class parameters on to an inline function by value (`destroy<It>(It b, It e) {
/// destroy_impl(b, e); }`: MWCC copies them into the inline's parameter objects and then works
/// on registers). The body moves into a helper `inline` function taking one-word objects
/// (`w[0]` read once each), which the function calls with its parameters.
pub fn inline_by_value_helper(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let declared = crate::sig::demangle(&ir.symbol).is_some() || db.is_some_and(|db| db.decls.contains_key(ir.symbol.as_str()) || db.decls.contains_key(&format!("::{}", ir.symbol)) || db.functions.contains_key(ir.symbol.as_str()));
    if declared || ir.this_var.is_some() || ir.params.is_empty() || ir.inline_helper.is_some() || !matches!(ir.sig.ret, Type::Void) || ir.dead_stores.is_empty() {
        return;
    }
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::Var(v) = e {
            *uses.entry(*v).or_default() += 1;
        }
    });
    let word = |p: VarId| Expr::Load { base: Box::new(Expr::Var(p)), offset: 0, ty: Type::Int { size: 4, signed: true } };
    let is_word = |e: &Expr, p: VarId| matches!(e, Expr::Load { base, offset: 0, ty } if matches!(**base, Expr::Var(v) if v == p) && scalar_size(ty) == Some(4));
    for &p in &ir.params {
        if uses.get(&p) != Some(&1) {
            return;
        }
        let mut read = 0;
        Stmt::walk_exprs(&ir.body, &mut |e| read += is_word(e, p) as usize);
        // (the frame copy of the word: the inline's parameter object)
        if read != 1 || !ir.dead_stores.iter().any(|d| d.size == 4 && is_word(&d.value, p)) {
            return;
        }
    }
    let _ = word;
    let words = crate::helpers::words(4);
    let mut h = ir.clone();
    h.symbol = INLINE_HELPER.to_string();
    h.sig.qualified_name = INLINE_HELPER.to_string();
    h.sig.mangled = None;
    h.dead_stores = vec![];
    for &p in &ir.params {
        h.vars[p].ty = words.clone();
        if let VarKind::Param { index } = h.vars[p].kind {
            if let Some(sp) = h.sig.params.get_mut(index) {
                sp.ty = words.clone();
            }
        }
    }
    let params = ir.params.clone();
    Stmt::rewrite_exprs(&mut h.body, &mut |e| {
        if let Some(&p) = params.iter().find(|&&p| is_word(e, p)) {
            *e = Expr::Member { base: Box::new(Expr::Var(p)), offset: 0, ty: Type::Int { size: 4, signed: true } };
        }
    });
    for &p in &ir.params {
        ir.vars[p].ty = words.clone();
        if let VarKind::Param { index } = ir.vars[p].kind {
            if let Some(sp) = ir.sig.params.get_mut(index) {
                sp.ty = words.clone();
            }
        }
    }
    let call = Expr::Call { callee: Callee::Direct { symbol: INLINE_HELPER.to_string(), sig: h.sig.clone() }, args: ir.params.iter().map(|&p| Expr::Var(p)).collect(), ret: Type::Void };
    ir.body = vec![Stmt::Expr(call)];
    ir.inline_helper = Some(Box::new(h));
}

/// A loop `v = X; e = Y; ... while (v != e) {...}` whose begin and end words were also stored to
/// frame slots nothing reads: the source called an inline function taking the begin and end
/// iterators by value (`uninitialized_copy(begin(), end(), out)`; MWCC copies each argument into
/// the parameter object and runs the loop on registers). The loop and its setup move into a
/// helper `inline` function taking two one-word iterator objects (made by value, as a
/// container's inline `begin()`/`end()` returns them) and the loop's other inputs.
pub fn inline_loop_helper(ir: &mut IrFunction) {
    if ir.inline_helper.is_some() || ir.dead_stores.iter().filter(|d| d.size == 4).count() < 2 {
        return;
    }
    let snapshot = ir.clone();
    let whole = ir.body.clone();
    let mut made: Option<IrFunction> = None;
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        if made.is_some() {
            return;
        }
        for k in 0..b.len() {
            if let Some((h, call, range)) = loop_helper_at(&snapshot, &whole, b, k) {
                b.splice(range, [call]);
                made = Some(h);
                return;
            }
        }
    });
    if let Some(h) = made {
        ir.inline_helper = Some(Box::new(h));
    }
}

/// The helper for the loop at `b[k]`: (helper function, the call replacing the run, the run).
fn loop_helper_at(ir: &IrFunction, whole: &[Stmt], b: &[Stmt], k: usize) -> Option<(IrFunction, Stmt, std::ops::Range<usize>)> {
    let Stmt::While { cond, body: lb } = &b[k] else { return None };
    fn strip(e: &Expr) -> Expr {
        let mut e = e;
        while let Expr::Cast { e: x, .. } = e {
            e = x;
        }
        e.clone()
    }
    let Expr::Binary { op: BinOp::Ne, l, r, .. } = cond else { return None };
    let (Expr::Var(v), Expr::Var(e)) = (strip(l), strip(r)) else { return None };
    // the setup run right before the loop: `x = src` assignments
    let mut j = k;
    while j > 0 && matches!(&b[j - 1], Stmt::Assign { dst: Expr::Var(_), src } if !src.has_call()) {
        j -= 1;
    }
    let inits: Vec<(VarId, Expr)> = b[j..k]
        .iter()
        .filter_map(|s| match s {
            Stmt::Assign { dst: Expr::Var(x), src } => Some((*x, src.clone())),
            _ => None,
        })
        .collect();
    let def = |x: VarId| inits.iter().find(|(y, _)| *y == x).map(|(_, s)| s.clone());
    let x_src = def(v)?;
    let y_src = def(e)?;
    // Y in terms of what the run started from (`e = v + n*k` reads v's start)
    let mut y_full = y_src.clone();
    y_full.rewrite(&mut |t| {
        if matches!(t, Expr::Var(w) if *w == v) {
            *t = x_src.clone();
        }
    });
    let stored = |want: &Expr| ir.dead_stores.iter().filter(|d| d.size == 4 && strip(&d.value) == strip(want)).count();
    // (the stored end may still read the begin variable)
    if stored(&x_src) == 0 || stored(&y_full) + stored(&y_src) == 0 {
        return None;
    }
    // inputs: variables the loop and the setup read that the setup doesn't define
    let mut reads: Vec<VarId> = vec![];
    {
        let mut note = |t: &Expr| {
            if let Expr::Var(w) = t {
                if !reads.contains(w) {
                    reads.push(*w);
                }
            }
        };
        cond.walk(&mut note);
        Stmt::walk_exprs(lb, &mut note);
        // (begin and end are computed at the call)
        for (x, s) in &inits {
            if *x != v && *x != e {
                s.walk(&mut note);
            }
        }
    }
    let defined: Vec<VarId> = inits.iter().map(|(x, _)| *x).collect();
    let mut inputs: Vec<VarId> = reads.iter().copied().filter(|w| !defined.contains(w) && *w != v && *w != e).collect();
    inputs.sort_unstable();
    // nothing the loop or setup writes is read anywhere else
    let mut written: Vec<VarId> = defined.clone();
    let mut lbc = lb.clone();
    Stmt::for_each_block_mut(&mut lbc, &mut |bb| {
        for s in bb.iter() {
            if let Stmt::Assign { dst: Expr::Var(w), .. } = s {
                written.push(*w);
            }
        }
    });
    let count = |stmts: &[Stmt]| {
        let mut n = 0usize;
        Stmt::walk_exprs(stmts, &mut |t| {
            if let Expr::Var(w) = t {
                if written.contains(w) {
                    n += 1;
                }
            }
        });
        n
    };
    if count(whole) != count(&b[j..=k]) || inputs.iter().any(|w| matches!(ir.vars[*w].kind, VarKind::Stack { .. })) {
        return None;
    }
    // the helper: (begin, end, inputs...)
    let iter = crate::helpers::iter();
    let int = Type::Int { size: 4, signed: true };
    let mut h = ir.clone();
    h.symbol = INLINE_LOOP.to_string();
    h.sig.qualified_name = INLINE_LOOP.to_string();
    h.sig.mangled = None;
    h.sig.ret = Type::Void;
    h.sig.this_class = None;
    h.sig.is_const = false;
    h.this_var = None;
    h.dead_stores = vec![];
    h.inline_helper = None;
    for var in h.vars.iter_mut() {
        if matches!(var.kind, VarKind::Param { .. } | VarKind::This | VarKind::StructRet) {
            var.kind = VarKind::Local;
        }
    }
    let bv = h.vars.len();
    let ev = bv + 1;
    h.vars.push(Var { name: "begin".into(), ty: iter.clone(), kind: VarKind::Param { index: 0 } });
    h.vars.push(Var { name: "end".into(), ty: iter.clone(), kind: VarKind::Param { index: 1 } });
    let mut params = vec![bv, ev];
    let mut sp = vec![mwdec_core::Param { name: Some("begin".into()), ty: iter.clone() }, mwdec_core::Param { name: Some("end".into()), ty: iter.clone() }];
    for (n, &w) in inputs.iter().enumerate() {
        h.vars[w].kind = VarKind::Param { index: 2 + n };
        params.push(w);
        sp.push(mwdec_core::Param { name: Some(h.vars[w].name.clone()), ty: h.vars[w].ty.clone() });
    }
    h.params = params;
    h.decl_params = vec![];
    h.sig.params = sp;
    let word = |p: VarId| Expr::Member { base: Box::new(Expr::Var(p)), offset: 0, ty: int.clone() };
    let mut hb: Vec<Stmt> = vec![];
    for (x, s) in &inits {
        if *x == e {
            continue;
        }
        let src = if *x == v { word(bv) } else { s.clone() };
        hb.push(Stmt::Assign { dst: Expr::Var(*x), src });
    }
    let mut hcond = cond.clone();
    hcond.rewrite(&mut |t| {
        if matches!(t, Expr::Var(w) if *w == e) {
            *t = word(ev);
        }
    });
    hb.push(Stmt::While { cond: hcond, body: lb.clone() });
    h.body = hb;
    // the call: (iter_at(X), iter_at(Y), inputs...)
    let at_sig = mwdec_core::FuncSig { qualified_name: crate::helpers::ITER_AT.into(), mangled: None, ret: iter.clone(), params: vec![mwdec_core::Param { name: None, ty: int.clone() }], this_class: None, is_const: false, is_static: false, is_virtual: false, variadic: false, runs_code: false };
    let at = |x: Expr| Expr::Call { callee: Callee::Direct { symbol: crate::helpers::ITER_AT.into(), sig: at_sig.clone() }, args: vec![Expr::cast(int.clone(), x)], ret: iter.clone() };
    let mut args = vec![at(x_src), at(y_full)];
    args.extend(inputs.iter().map(|&w| Expr::Var(w)));
    let call = Stmt::Expr(Expr::Call { callee: Callee::Direct { symbol: INLINE_LOOP.to_string(), sig: h.sig.clone() }, args, ret: Type::Void });
    Some((h, call, j..k + 1))
}

/// Name of the helper of [`inline_loop_helper`].
pub const INLINE_LOOP: &str = "mwdec_inline_loop";

/// `v.@0 = <member>` where the member is a class object of exactly the stored size and `v` a
/// frame object: (v, the member as that class, the class).
fn whole_member_store(s: &Stmt, vars: &[Var], db: &mwdec_core::TypeDb) -> Option<(VarId, Expr, Type)> {
    let Stmt::Assign { dst, src } = s else { return None };
    let (v, dt) = match dst {
        Expr::Member { base, offset: 0, ty } => match **base {
            Expr::Var(v) => (v, ty.clone()),
            _ => return None,
        },
        _ => return None,
    };
    if !matches!(vars[v].kind, VarKind::Stack { .. }) {
        return None;
    }
    let n = scalar_size(&dt)?;
    let Expr::Load { base, offset, .. } = src else { return None };
    let bt = crate::types::ty_of(base, vars);
    let cls = named(pointee(&bt)?)?.to_string();
    let t = crate::aggregates::aggregate_at(db, &cls, *offset).into_iter().find(|t| crate::types::size_of(Some(db), t) == Some(n))?;
    Some((v, Expr::Load { base: base.clone(), offset: *offset, ty: t.clone() }, t))
}

/// Does `a` read the same memory as `b` (same base and offset, access type aside)?
fn same_place(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Member { base: x, offset: o, .. }, Expr::Member { base: y, offset: p, .. }) | (Expr::Load { base: x, offset: o, .. }, Expr::Load { base: y, offset: p, .. }) => o == p && x == y,
        (x, y) => x == y,
    }
}

/// A returned object built from a call's by-value result copied whole into it (word by word)
/// plus constant stores of the returned class's own members, the class having a constructor
/// taking that result's class: `return R(f());` (`return optional_object<CAABox>(
/// GetBoundingBox())`: the valid flag set, the box copied into the item storage).
pub fn fold_converting_return(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let Some(rv) = ir.vars.iter().position(|v| v.kind == VarKind::StructRet) else { return };
    let rt = strip_cv(&ir.sig.ret).clone();
    let Some(rcls) = named(&rt).map(|s| s.to_string()) else { return };
    let b = &ir.body;
    let Some(Stmt::Return(None)) = b.last() else { return };
    let end = b.len() - 1;
    // the result object: `s = f()` with every later statement a store into the returned object
    // or a register temp read from `s`
    let Some(si) = (0..end).rev().find(|&k| matches!(&b[k], Stmt::Assign { dst: Expr::Var(s), src: Expr::Call { .. } } if matches!(ir.vars[*s].kind, VarKind::Stack { .. }))) else { return };
    let Stmt::Assign { dst: Expr::Var(s), src: call } = &b[si] else { return };
    let (s, call) = (*s, call.clone());
    let t = strip_cv(&ir.vars[s].ty).clone();
    if named(&t).is_none() {
        return;
    }
    let Some(tsize) = crate::types::size_of(Some(db), &t) else { return };
    let mut temps: std::collections::HashMap<VarId, i32> = std::collections::HashMap::new();
    let mut copied: Vec<(i32, i32, u32)> = vec![]; // (dst offset, src offset, size)
    for st in &b[si + 1..end] {
        match st {
            Stmt::Assign { dst: Expr::Var(x), src: Expr::Member { base, offset, .. } } if matches!(**base, Expr::Var(w) if w == s) => {
                temps.insert(*x, *offset);
            }
            Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } if matches!(**base, Expr::Var(w) if w == rv) => {
                let n = scalar_size(ty).unwrap_or(0);
                match src {
                    Expr::Member { base: sb, offset: so, .. } if matches!(**sb, Expr::Var(w) if w == s) => copied.push((*offset, *so, n)),
                    Expr::Var(x) if temps.contains_key(x) => copied.push((*offset, temps[x], n)),
                    Expr::Int { .. } | Expr::Float { .. } => {}
                    _ => return,
                }
            }
            _ => return,
        }
    }
    // the whole object copied, word by word, to one place
    copied.sort_by_key(|c| c.1);
    let Some(&(d0, s0, _)) = copied.first() else { return };
    let mut next = 0i32;
    for &(d, so, n) in &copied {
        if so != next || d - so != d0 - s0 || n == 0 {
            return;
        }
        next += n as i32;
    }
    if next as u32 != tsize || s0 != 0 {
        return;
    }
    // the returned class's constructor taking that class (or its template parameter)
    let base = strip_template_args(&rcls);
    let last = crate::sig::split_scope(&base).1.to_string();
    let key = format!("{base}::{last}");
    let takes = |d: &mwdec_core::DeclInfo| -> bool {
        if d.params.len() != 1 || !d.is_inline_defined || d.access != mwdec_core::Access::Public {
            return false;
        }
        let pt = match strip_cv(&d.params[0].ty) {
            Type::Ref(x) => strip_cv(x).clone(),
            x => x.clone(),
        };
        pt == t || matches!(&pt, Type::Named(n) if d.template_params.contains(n))
    };
    let Some(_) = db.decls.get(&key).and_then(|ds| ds.iter().find(|d| takes(d))) else { return };
    let ctor = mwdec_core::FuncSig {
        qualified_name: key,
        mangled: None,
        ret: Type::Void,
        params: vec![mwdec_core::Param { name: None, ty: Type::Ref(Box::new(Type::Const(Box::new(t.clone())))) }],
        this_class: Some(rcls.clone()),
        is_const: false,
        is_static: false,
        is_virtual: false,
        variadic: false,
        runs_code: false,
    };
    let ret = Stmt::Return(Some(Expr::Construct { class: rt, ctor: Some(ctor), args: vec![call] }));
    ir.body.truncate(si);
    ir.body.push(ret);
}

/// `rstl::optional_object<CAABox>` -> `rstl::optional_object` (decl keys carry no arguments).
pub(crate) fn strip_template_args(s: &str) -> String {
    let mut out = String::new();
    let mut d = 0;
    for c in s.chars() {
        match c {
            '<' => d += 1,
            '>' => d -= 1,
            _ if d == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Can an object of class `t` be copied where the draft writes `T(x)` or passes it by value: no
/// copy constructor declared non-public (an implicit one is public).
pub fn copyable(db: &mwdec_core::TypeDb, t: &Type) -> bool {
    let Some(cls) = named(strip_cv(t)) else { return false };
    let base = strip_template_args(cls);
    let last = crate::sig::split_scope(&base).1.to_string();
    let key = format!("{base}::{last}");
    let copy_ctor = |d: &&mwdec_core::DeclInfo| {
        d.params.len() == 1 && matches!(strip_cv(&d.params[0].ty), Type::Ref(x) if named(strip_cv(x)).is_some_and(|n| strip_template_args(n) == base))
    };
    !db.decls.get(&key).is_some_and(|ds| ds.iter().filter(copy_ctor).any(|d| d.access != mwdec_core::Access::Public))
}

/// A frame object declared as a class with an inline default constructor whose initializer list
/// sets members to constants (`reserved_vector() : mCount(0) {}`): the constant member stores
/// before its first use are that construction, implicit in the declaration.
pub fn drop_default_construction_stores(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let vars = ir.vars.clone();
    for (v, var) in vars.iter().enumerate() {
        if !matches!(var.kind, VarKind::Stack { .. }) {
            continue;
        }
        let Some(cls) = named(&var.ty).map(|s| s.to_string()) else { continue };
        let Some(c) = crate::sig::find_class(db, &cls) else { continue };
        let base = strip_template_args(&cls);
        let last = crate::sig::split_scope(&base).1.to_string();
        let Some(init) = db.decls.get(&format!("{base}::{last}")).and_then(|ds| ds.iter().find(|d| d.params.is_empty() && d.is_inline_defined).and_then(|d| d.init_list.clone())) else { continue };
        // member offset -> constant literal
        let mut consts: Vec<(i32, String)> = vec![];
        for part in crate::sig::split_top(&init, ',') {
            let toks: Vec<&str> = part.split_whitespace().collect();
            if toks.len() < 4 || toks[1] != "(" || toks.last() != Some(&")") {
                continue;
            }
            if let Some(f) = c.fields.iter().find(|f| f.name == toks[0]) {
                consts.push((f.offset as i32, toks[2..toks.len() - 1].join(" ")));
            }
        }
        if consts.is_empty() {
            continue;
        }
        // the leading statements of the top-level body that mention v
        let mut k = 0;
        while k < ir.body.len() {
            let mut mentions = false;
            Stmt::walk_exprs(std::slice::from_ref(&ir.body[k]), &mut |e| {
                if matches!(e, Expr::Var(w) if *w == v) {
                    mentions = true;
                }
            });
            if !mentions {
                k += 1;
                continue;
            }
            let is_ctor_store = match &ir.body[k] {
                Stmt::Assign { dst: Expr::Member { base, offset, .. }, src } if matches!(**base, Expr::Var(w) if w == v) => {
                    consts.iter().any(|(o, lit)| o == offset && literal_value_is(lit, src))
                }
                _ => false,
            };
            if !is_ctor_store {
                break;
            }
            ir.body.remove(k);
        }
    }
}

fn literal_value_is(lit: &str, e: &Expr) -> bool {
    let t: String = lit.split_whitespace().collect();
    match e {
        Expr::Int { value, .. } => t.parse::<i64>().ok() == Some(*value) || (t == "false" && *value == 0) || (t == "true" && *value == 1) || ((t == "nullptr" || t == "NULL") && *value == 0),
        Expr::Float { bits, double } => t.trim_end_matches(|c| c == 'f' || c == 'F').parse::<f64>().ok().is_some_and(|x| if *double { x.to_bits() == *bits } else { (x as f32).to_bits() as u64 == *bits }),
        _ => false,
    }
}

/// Constant stores into the returned object right before `return;` that are exactly what an
/// inline constructor callable without arguments sets (`vector(const Alloc& a = Alloc()) :
/// mAllocator(a), mCount(0), mCapacity(0), mItems(nullptr)`): `return R();`.
pub fn fold_default_constructed_return(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let Some(rv) = ir.vars.iter().position(|v| v.kind == VarKind::StructRet) else { return };
    let rt = strip_cv(&ir.sig.ret).clone();
    let resolved = crate::types::resolve(Some(db), &rt).into_owned();
    let Some(rcls) = named(strip_cv(&resolved)).map(|s| s.to_string()) else { return };
    let Some(c) = crate::sig::find_class(db, &rcls) else { return };
    let base = strip_template_args(&rcls);
    let last = crate::sig::split_scope(&base).1.to_string();
    let Some(decls) = db.decls.get(&format!("{base}::{last}")) else { return };
    // (member offset, constant) of the first argument-free inline constructor with constants
    let mut consts: Option<Vec<(i32, String)>> = None;
    'd: for d in decls {
        let callable = d.params.is_empty() || (d.defaults.len() == d.params.len() && d.defaults.iter().all(|x| x.is_some()));
        if !callable || !d.is_inline_defined || d.access != mwdec_core::Access::Public {
            continue;
        }
        let Some(init) = &d.init_list else { continue };
        let names: Vec<String> = d.params.iter().map(|p| p.name.clone().unwrap_or_default()).collect();
        let mut cs = vec![];
        for part in crate::sig::split_top(init, ',') {
            let toks: Vec<&str> = part.split_whitespace().collect();
            if toks.len() < 4 || toks[1] != "(" || toks.last() != Some(&")") {
                continue 'd;
            }
            let Some(f) = c.fields.iter().find(|f| f.name == toks[0]) else { continue 'd };
            let inner = toks[2..toks.len() - 1].join(" ");
            if names.iter().any(|n| *n == inner) {
                // a parameter (its default) into an empty member: no store
                if crate::types::size_of(Some(db), &f.ty).unwrap_or(0) > 1 || crate::types::is_aggregate(Some(db), &f.ty) && !c.fields.is_empty() && crate::types::size_of(Some(db), &f.ty) != Some(1) {
                    continue 'd;
                }
                continue;
            }
            cs.push((f.offset as i32, inner));
        }
        if !cs.is_empty() {
            consts = Some(cs);
            break;
        }
    }
    let Some(consts) = consts else { return };
    let ctor = mwdec_core::FuncSig {
        qualified_name: format!("{base}::{last}"),
        mangled: None,
        ret: Type::Void,
        params: vec![],
        this_class: Some(rcls.clone()),
        is_const: false,
        is_static: false,
        is_virtual: false,
        variadic: false,
        runs_code: false,
    };
    let matches_all = |stores: &[Stmt]| -> bool {
        let mut seen: Vec<i32> = vec![];
        for st in stores {
            match st {
                Stmt::Assign { dst: Expr::Load { base, offset, .. }, src } if matches!(**base, Expr::Var(w) if w == rv) && consts.iter().any(|(o, lit)| o == offset && literal_value_is(lit, src)) => seen.push(*offset),
                _ => return false,
            }
        }
        seen.sort_unstable();
        seen.dedup();
        seen.len() == consts.len() && stores.len() == seen.len()
    };
    // an `if` arm made of exactly those stores, the function returning right after the `if`
    let n = ir.body.len();
    if n >= 2 && matches!(ir.body[n - 1], Stmt::Return(None)) {
        if let Stmt::If { then, els, .. } = &mut ir.body[n - 2] {
            // one arm the constants, the other constructing the returned object in place
            // (`__return->R(args)`): both become returns; anything else is left alone
            let in_place = |arm: &[Stmt]| -> Option<Expr> {
                if let [Stmt::Expr(Expr::Call { callee: Callee::Method { sig: cs, this, .. }, args, .. })] = arm {
                    if crate::sig::is_ctor(cs) && matches!(**this, Expr::Var(w) if w == rv) {
                        return Some(Expr::Construct { class: rt.clone(), ctor: Some(cs.clone()), args: args.clone() });
                    }
                }
                None
            };
            let dflt = Stmt::Return(Some(Expr::Construct { class: rt.clone(), ctor: Some(ctor.clone()), args: vec![] }));
            if !then.is_empty() && matches_all(then) {
                if let Some(c2) = in_place(els) {
                    *then = vec![dflt];
                    *els = vec![Stmt::Return(Some(c2))];
                }
            } else if !els.is_empty() && matches_all(els) {
                if let Some(c2) = in_place(then) {
                    *els = vec![dflt];
                    *then = vec![Stmt::Return(Some(c2))];
                }
            }
        }
        // both arms return now: the trailing `return;` is unreachable
        if matches!(&ir.body[n - 2], Stmt::If { then, els, .. } if matches!(then.last(), Some(Stmt::Return(Some(_)))) && matches!(els.last(), Some(Stmt::Return(Some(_))))) {
            ir.body.pop();
        }
    }
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let Some(r) = b.iter().position(|s| matches!(s, Stmt::Return(None))) else { return };
        let mut k = r;
        let mut seen: Vec<i32> = vec![];
        while k > 0 {
            match &b[k - 1] {
                Stmt::Assign { dst: Expr::Load { base, offset, .. }, src } if matches!(**base, Expr::Var(w) if w == rv) && consts.iter().any(|(o, lit)| o == offset && literal_value_is(lit, src)) => {
                    seen.push(*offset);
                    k -= 1;
                }
                _ => break,
            }
        }
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != consts.len() || r - k != seen.len() {
            return;
        }
        b.splice(k..=r, std::iter::once(Stmt::Return(Some(Expr::Construct { class: rt.clone(), ctor: Some(ctor.clone()), args: vec![] }))));
    });
}

/// The member a class's inline copy constructor increments through (`rc_ptr(const rc_ptr& o)
/// : mPtr(o.mPtr), mRefCount(o.mRefCount) { ++*mRefCount; }`): its offset.
fn copy_ctor_increment(db: &mwdec_core::TypeDb, cls: &str) -> Option<i32> {
    let base = strip_template_args(cls);
    let last = crate::sig::split_scope(&base).1.to_string();
    let ds = db.decls.get(&format!("{base}::{last}"))?;
    let d = ds.iter().find(|d| {
        d.is_inline_defined && d.params.len() == 1 && matches!(strip_cv(&d.params[0].ty), Type::Ref(x) if named(strip_cv(x)).is_some_and(|n| strip_template_args(n) == base))
    })?;
    let body = d.inline_body.as_deref()?.replace("( ( void ) 0 ) ;", "");
    let toks: Vec<&str> = body.split_whitespace().collect();
    let name = match toks.as_slice() {
        ["++", "*", n, ";"] | ["++", "(", "*", n, ")", ";"] | ["*", n, "+=", "1", ";"] => *n,
        _ => return None,
    };
    let c = crate::sig::find_class(db, cls)?;
    c.fields.iter().find(|f| f.name == name).map(|f| f.offset as i32)
}

/// A frame object built as a copy of an object of a class whose inline copy constructor
/// increments a counter through a member (`rc_ptr`): the increment right after the member
/// stores (`stack.mRefCount = p; *p += 1;`) is the copy constructor's body, implicit in the
/// copy-initialization the stores fold to.
pub fn drop_copy_ctor_increments(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let vars = ir.vars.clone();
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    crate::inline::count_uses(&ir.body, &mut uses);
    let mut untype: Vec<VarId> = vec![];
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut j = 0;
        while j < b.len() {
            let Stmt::Assign { dst: Expr::Load { base, offset: 0, .. }, src: Expr::Binary { op: BinOp::Add, l, r, .. } } = &b[j] else {
                j += 1;
                continue;
            };
            if !(matches!(&**l, Expr::Load { base: lb, offset: 0, .. } if lb == base) && matches!(&**r, Expr::Int { value: 1, .. })) {
                j += 1;
                continue;
            }
            // the counter pointer just stored into a frame object's counter member
            let p = (**base).clone();
            let stored = b[j.saturating_sub(3)..j].iter().find_map(|s| match s {
                Stmt::Assign { dst: Expr::Member { base: ob, offset, .. }, src } if *src == p => match &**ob {
                    Expr::Var(d) if matches!(vars[*d].kind, VarKind::Stack { .. }) && named(&vars[*d].ty).and_then(|c| copy_ctor_increment(db, c)) == Some(*offset) => Some(*d),
                    _ => None,
                },
                _ => None,
            });
            if let Some(d) = stored {
                // (a register temp used for nothing else: the store and the increment's read
                // and write; otherwise the object stays raw storage, its copy explicit)
                if matches!(p, Expr::Var(t) if uses.get(&t).copied().unwrap_or(0) != 3) {
                    untype.push(d);
                } else {
                    b.remove(j);
                    continue;
                }
            }
            j += 1;
        }
    });
    for d in untype {
        if let VarKind::Stack { size, .. } = ir.vars[d].kind {
            ir.vars[d].ty = Type::Unknown { size };
        }
    }
}

/// The 4-byte stack local a word copy reads (`L`, `L.@0`, through casts).
fn word_local(e: &Expr, vars: &[Var]) -> Option<VarId> {
    let v = match e {
        Expr::Cast { e, .. } => return word_local(e, vars),
        Expr::Var(v) => *v,
        Expr::Member { base, offset: 0, .. } => match **base {
            Expr::Var(v) => v,
            _ => return None,
        },
        _ => return None,
    };
    matches!(vars[v].kind, VarKind::Stack { size: 4, .. }).then_some(v)
}

fn strip_casts(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip_casts(e),
        e => e,
    }
}

/// Word slots of the frame filled one by one and copied out as words into consecutive members
/// of an object, the next member stored from a register whose value a dead frame store right
/// after the slots holds too (`CSphere(center, r)` built in the frame and assigned: the radius
/// store into the temporary is dead, the assignment takes it from the register): one frame
/// object of the destination member's class, built member-wise and assigned whole.
pub fn frame_object_copied_back(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    if ir.dead_stores.is_empty() {
        return;
    }
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    crate::inline::count_uses(&ir.body, &mut uses);
    let vars = ir.vars.clone();
    let off_of = |v: VarId| match vars[v].kind {
        VarKind::Stack { offset, .. } => offset,
        _ => i32::MIN,
    };
    let mut new_vars: Vec<Var> = vec![];
    let mut consumed: Vec<usize> = vec![];
    let dead = ir.dead_stores.clone();
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        // word copies out of stack locals, directly or through a register temp read once:
        // (index, destination base, offset, local, the temp's definition)
        let temp_def = |t: VarId| -> Option<(usize, VarId)> {
            if !matches!(vars[t].kind, VarKind::Local) || uses.get(&t).copied().unwrap_or(0) != 1 {
                return None;
            }
            let ds: Vec<(usize, &Expr)> = b.iter().enumerate().filter_map(|(i, s)| match s {
                Stmt::Assign { dst: Expr::Var(x), src } if *x == t => Some((i, src)),
                _ => None,
            }).collect();
            match ds.as_slice() {
                [(i, src)] => word_local(src, &vars).map(|l| (*i, l)),
                _ => None,
            }
        };
        let copies: Vec<(usize, Expr, i32, VarId, Option<usize>)> = b
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } if scalar_size(ty) == Some(4) => match strip_casts(src) {
                    Expr::Var(t) if matches!(vars[*t].kind, VarKind::Local) => temp_def(*t).map(|(ti, l)| (i, (**base).clone(), *offset, l, Some(ti))),
                    _ => word_local(src, &vars).map(|l| (i, (**base).clone(), *offset, l, None)),
                },
                _ => None,
            })
            .collect();
        for &(i0, ref d, d0, l0, _) in &copies {
            // a run starting here: consecutive destination offsets and slots
            let mut run = vec![(i0, l0)];
            let mut temps: Vec<usize> = copies.iter().filter(|c| c.0 == i0).filter_map(|c| c.4).collect();
            loop {
                let k = run.len() as i32;
                match copies.iter().find(|c| c.1 == *d && c.2 == d0 + 4 * k && off_of(c.3) == off_of(l0) + 4 * k) {
                    Some(c) => {
                        run.push((c.0, c.3));
                        temps.extend(c.4);
                    }
                    None => break,
                }
            }
            let n = run.len() as i32;
            if n < 2 || copies.iter().any(|c| c.1 == *d && c.2 == d0 - 4 && off_of(c.3) == off_of(l0) - 4) {
                continue;
            }
            let last = run.iter().map(|r| r.0).max().unwrap();
            // the next member from the register a dead store at the next slot holds
            let Some((iv, vty, val)) = b.iter().enumerate().skip(last + 1).take(3).find_map(|(i, s)| match s {
                Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } if **base == *d && *offset == d0 + 4 * n => Some((i, ty.clone(), src.clone())),
                _ => None,
            }) else {
                continue;
            };
            let vsize = scalar_size(&vty).unwrap_or(0);
            let Some(di) = dead.iter().position(|x| x.offset == off_of(l0) + 4 * n && x.size == vsize && strip_casts(&x.value) == strip_casts(&val)) else { continue };
            if consumed.contains(&di) {
                continue;
            }
            // the destination member's class spans exactly the slots and that member
            let size = (4 * n) as u32 + vsize;
            let Some(cls) = (match &crate::types::ty_of(d, &vars) {
                t if is_ptr(t) => pointee(t).and_then(|p| named(&crate::types::resolve(Some(db), p)).map(|s| s.to_string())),
                _ => None,
            }) else {
                continue;
            };
            let Some(t) = crate::aggregates::aggregate_at(db, &cls, d0).into_iter().find(|t| crate::types::size_of(Some(db), t) == Some(size)) else { continue };
            // each slot: one definition in this list before its copy, read only by it
            let mut defs = vec![];
            for &(ci, l) in &run {
                let ds: Vec<usize> = b.iter().enumerate().filter(|(_, s)| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if *x == l)).map(|(i, _)| i).collect();
                if ds.len() != 1 || ds[0] > ci || uses.get(&l).copied().unwrap_or(0) != 1 {
                    break;
                }
                defs.push(ds[0]);
            }
            if defs.len() != run.len() {
                continue;
            }
            let id = vars.len() + new_vars.len();
            new_vars.push(Var { name: format!("stack_{:x}", off_of(l0)), ty: t.clone(), kind: VarKind::Stack { offset: off_of(l0), size } });
            for (k, &di2) in defs.iter().enumerate() {
                if let Stmt::Assign { dst, .. } = &mut b[di2] {
                    let lt = vars[run[k].1].ty.clone();
                    *dst = Expr::Member { base: Box::new(Expr::Var(id)), offset: 4 * k as i32, ty: lt };
                }
            }
            // the register member and the whole assignment in place of the last copy
            let mut out: Vec<Stmt> = Vec::with_capacity(b.len());
            for (i, s) in std::mem::take(b).into_iter().enumerate() {
                if i == last {
                    out.push(Stmt::Assign { dst: Expr::Member { base: Box::new(Expr::Var(id)), offset: 4 * n, ty: vty.clone() }, src: val.clone() });
                    out.push(Stmt::Assign { dst: Expr::Load { base: Box::new(d.clone()), offset: d0, ty: t.clone() }, src: Expr::Var(id) });
                } else if i == iv || run.iter().any(|r| r.0 == i) || temps.contains(&i) {
                    continue;
                } else {
                    out.push(s);
                }
            }
            *b = out;
            consumed.push(di);
            // (one object per list and pass: indices changed)
            break;
        }
    });
    ir.vars.extend(new_vars);
    consumed.sort_unstable();
    for di in consumed.into_iter().rev() {
        ir.dead_stores.remove(di);
    }
    for (i, d) in ir.dead_stores.iter_mut().enumerate() {
        d.order = i;
    }
}

/// `if (c) { A = f(); p = &A; } else { B = g(); p = &B; } use(p)`: each arm's by-value call
/// result in its own frame temporary, the joined pointer read once: the conditional expression
/// `use(&(c ? f() : g()))` (the compiler makes the arm temporaries again).
pub fn conditional_temporaries(ir: &mut IrFunction) {
    let mut total: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::Var(v) = e {
            *total.entry(*v).or_default() += 1;
        }
    });
    let vars = ir.vars.clone();
    let arm = |b: &[Stmt]| -> Option<(VarId, Expr, VarId)> {
        let [Stmt::Assign { dst: Expr::Var(a), src: call @ Expr::Call { .. } }, Stmt::Assign { dst: Expr::Var(p), src: Expr::AddrOf(x) }] = b else { return None };
        (matches!(**x, Expr::Var(y) if y == *a) && matches!(vars[*a].kind, VarKind::Stack { .. }) && named(&vars[*a].ty).is_some() && matches!(vars[*p].kind, VarKind::Local)).then(|| (*a, call.clone(), *p))
    };
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut i = 0;
        while i + 1 < b.len() {
            let Stmt::If { cond, then, els } = &b[i] else {
                i += 1;
                continue;
            };
            let (Some((a, ca, p)), Some((bv, cb, p2))) = (arm(then), arm(els)) else {
                i += 1;
                continue;
            };
            let same_ty = types_eq(&vars[a].ty, &vars[bv].ty);
            if p != p2 || !same_ty || total.get(&p) != Some(&3) || total.get(&a) != Some(&2) || total.get(&bv) != Some(&2) {
                i += 1;
                continue;
            }
            let t = Expr::Ternary { c: Box::new(cond.clone()), t: Box::new(ca), f: Box::new(cb), ty: vars[a].ty.clone() };
            let with = Expr::AddrOf(Box::new(t));
            let mut n = 0;
            Stmt::rewrite_exprs(std::slice::from_mut(&mut b[i + 1]), &mut |e| {
                if matches!(e, Expr::Var(y) if *y == p) {
                    *e = with.clone();
                    n += 1;
                }
            });
            if n == 1 {
                b.remove(i);
            }
            i += 1;
        }
    });
}

fn types_eq(a: &Type, b: &Type) -> bool {
    named(a).map(crate::sig::norm_name) == named(b).map(crate::sig::norm_name)
}

/// The class a vtable symbol belongs to (`__vt__29CValidCameraWaypointPredicate`).
fn vtable_class(sym: &str) -> Option<String> {
    let rest = sym.strip_prefix("__vt__")?;
    let n: usize = rest.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()?;
    let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    let name = rest.get(digits..digits + n)?;
    (rest.len() == digits + n && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')).then(|| name.to_string())
}

fn vtable_store(s: &Stmt, v: VarId) -> Option<String> {
    let Stmt::Assign { dst: Expr::Member { base, offset: 0, .. }, src: Expr::AddrOf(g) } = s else { return None };
    if !matches!(**base, Expr::Var(x) if x == v) {
        return None;
    }
    match &**g {
        Expr::Global { symbol, .. } => vtable_class(symbol),
        _ => None,
    }
}

/// A frame object built by an inline constructor of a class the context lacks (its vtable
/// stored over the base's), passed by reference to one call and destroyed right after
/// through the base's destructor: a temporary of that class of the unit's own source,
/// `f(..., D())` (`D : B`; the emitter defines it from `local_class_bases`).
pub fn derived_temporary_args(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let mut total: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::Var(v) = e {
            *total.entry(*v).or_default() += 1;
        }
    });
    let vars = ir.vars.clone();
    let mut found: Vec<(String, String, VarId)> = vec![];
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut j = 0;
        while j < b.len() {
            // a call taking `&S` of a stack object S
            let mut s_var = None;
            Stmt::walk_exprs(std::slice::from_ref(&b[j]), &mut |e| {
                if let Expr::Call { args, .. } = e {
                    for a in args {
                        if let Expr::AddrOf(x) = a {
                            if let Expr::Var(v) = **x {
                                if matches!(vars[v].kind, VarKind::Stack { .. }) {
                                    s_var.get_or_insert(v);
                                }
                            }
                        }
                    }
                }
            });
            let Some(v) = s_var else {
                j += 1;
                continue;
            };
            // vtable stores before it (the last names the class)
            let mut pre: Vec<usize> = vec![];
            let mut k = j;
            while k > 0 && vtable_store(&b[k - 1], v).is_some() {
                k -= 1;
                pre.push(k);
            }
            let Some(d) = pre.iter().min().map(|_| vtable_store(&b[j - 1], v).unwrap()) else {
                j += 1;
                continue;
            };
            // after it: the derived vtable again and the base destructor on S
            let mut post: Vec<usize> = vec![];
            let mut q = j + 1;
            while q < b.len() && vtable_store(&b[q], v).is_some() {
                post.push(q);
                q += 1;
            }
            let base = match b.get(q) {
                Some(Stmt::Expr(Expr::Call { callee: Callee::Method { sig: s, this, .. }, .. })) if crate::sig::is_dtor(s) && matches!(&**this, Expr::AddrOf(x) if matches!(**x, Expr::Var(y) if y == v)) => s.this_class.clone(),
                _ => None,
            };
            let Some(base) = base else {
                j += 1;
                continue;
            };
            let known = crate::sig::find_class(db, &d).is_some_and(|c| !c.is_declaration);
            let uses = pre.len() + post.len() + 2;
            if known || crate::sig::norm_name(&d) == crate::sig::norm_name(&base) || total.get(&v) != Some(&uses) || crate::sig::find_class(db, &base).map_or(true, |c| c.vtable.is_empty()) {
                j += 1;
                continue;
            }
            // the temporary in place of the argument
            let built = Expr::AddrOf(Box::new(Expr::Construct { class: Type::Named(d.clone()), ctor: None, args: vec![] }));
            Stmt::rewrite_exprs(std::slice::from_mut(&mut b[j]), &mut |e| {
                if matches!(e, Expr::AddrOf(x) if matches!(**x, Expr::Var(y) if y == v)) {
                    *e = built.clone();
                }
            });
            let mut rm: Vec<usize> = pre.iter().chain(post.iter()).copied().collect();
            rm.push(q);
            rm.sort_unstable();
            let first = *rm.first().unwrap();
            for i in rm.into_iter().rev() {
                b.remove(i);
            }
            found.push((d, base, v));
            let _ = first;
            j = j - pre.len() + 1;
        }
    });
    for (d, base, v) in found {
        ir.vars[v].ty = Type::Named(d.clone());
        if !ir.local_class_bases.iter().any(|x| x.0 == d) {
            ir.local_class_bases.push((d, base));
        }
    }
}

/// Scope-guard classes of the context: one member, an inline default constructor initializing
/// it from an argument-free call (`mEnabled(OSDisableInterrupts())`) and an inline destructor
/// passing it to another function (`OSRestoreInterrupts(mEnabled);`): (class, member size,
/// acquire function, release function).
fn scope_guard_classes(db: &mwdec_core::TypeDb) -> Vec<(String, u32, String, String)> {
    let mut out = vec![];
    for (name, c) in &db.classes {
        if c.is_declaration || c.fields.len() != 1 || !c.bases.is_empty() || c.vptr_offset.is_some() || name.contains('<') {
            continue;
        }
        let f = &c.fields[0];
        let last = crate::sig::split_scope(name).1;
        let Some(ctor) = db.decls.get(&format!("{name}::{last}")).and_then(|ds| ds.iter().find(|d| d.params.is_empty() && d.is_inline_defined)) else { continue };
        let Some(dtor) = db.decls.get(&format!("{name}::~{last}")).and_then(|ds| ds.iter().find(|d| d.is_inline_defined)) else { continue };
        let init: Vec<&str> = ctor.init_list.as_deref().unwrap_or("").split_whitespace().collect();
        let body: Vec<&str> = dtor.inline_body.as_deref().unwrap_or("").split_whitespace().collect();
        let (acq, rel) = match (init.as_slice(), body.as_slice()) {
            ([m, "(", a, "(", ")", ")"], [r, "(", m2, ")", ";"]) if *m == f.name && *m2 == f.name => (a.to_string(), r.to_string()),
            _ => continue,
        };
        let Some(sz) = crate::types::size_of(Some(db), &f.ty) else { continue };
        out.push((name.clone(), sz, acq, rel));
    }
    out
}

/// The callee of an argument-free direct call, through a comparison with zero / casts.
fn acquire_call(e: &Expr) -> Option<&str> {
    match e {
        Expr::Cast { e, .. } => acquire_call(e),
        Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => acquire_call(l),
        Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. } if args.is_empty() => Some(symbol),
        _ => None,
    }
}

/// `t = acquire(); ... release(t);` with `t` also stored, dead, into a frame slot of the
/// member's size: a scope guard object (`CInterruptGuard guard;`) whose inline constructor
/// and destructor are those calls.
pub fn scope_guards(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    if ir.dead_stores.is_empty() {
        return;
    }
    let guards = scope_guard_classes(db);
    if guards.is_empty() {
        return;
    }
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    crate::inline::count_uses(&ir.body, &mut uses);
    let vars = ir.vars.clone();
    let dead = ir.dead_stores.clone();
    let mut consumed: Vec<usize> = vec![];
    let mut new_vars: Vec<Var> = vec![];
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let Stmt::Assign { dst: Expr::Var(t), src } = &b[i] else {
                i += 1;
                continue;
            };
            let (t, src) = (*t, src.clone());
            let Some(acq) = acquire_call(&src).map(|s| s.to_string()) else {
                i += 1;
                continue;
            };
            let Some((cls, sz, _, rel)) = guards.iter().find(|g| g.2 == acq).cloned() else {
                i += 1;
                continue;
            };
            // every use of the flag a release call (one on each path out of the scope)
            let is_release = |s: &Stmt| matches!(s, Stmt::Expr(Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. }) if *symbol == rel && args.len() == 1 && matches!(strip_casts(&args[0]), Expr::Var(x) if *x == t));
            let mut releases = 0usize;
            fn count_rel(b: &[Stmt], f: &dyn Fn(&Stmt) -> bool, n: &mut usize) {
                for s in b {
                    if f(s) {
                        *n += 1;
                    }
                    match s {
                        Stmt::If { then, els, .. } => {
                            count_rel(then, f, n);
                            count_rel(els, f, n);
                        }
                        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => count_rel(body, f, n),
                        _ => {}
                    }
                }
            }
            count_rel(&b[i + 1..], &is_release, &mut releases);
            if !matches!(vars[t].kind, VarKind::Local) || releases == 0 || uses.get(&t) != Some(&releases) {
                i += 1;
                continue;
            }
            let k = usize::MAX;
            // (the call's register result, tested against zero: its temp folded away since)
            let reg_flag = |e: &Expr| match strip_casts(e) {
                Expr::Binary { op: BinOp::Ne, l, r, .. } => r.as_int() == Some(0) && matches!(strip_casts(l), Expr::Var(_)),
                Expr::Var(_) => true,
                _ => false,
            };
            let calls_acq = |e: &Expr| {
                let mut hit = false;
                e.walk(&mut |x| hit |= matches!(x, Expr::Call { callee: Callee::Direct { symbol, .. }, .. } if *symbol == acq));
                hit
            };
            let Some(di) = dead.iter().position(|d| d.size == sz && (d.value == src || d.value == Expr::Var(t) || calls_acq(&d.value) || reg_flag(&d.value))) else {
                i += 1;
                continue;
            };
            if consumed.contains(&di) {
                i += 1;
                continue;
            }
            let id = vars.len() + new_vars.len();
            new_vars.push(Var { name: "guard".into(), ty: Type::Named(cls.clone()), kind: VarKind::Stack { offset: dead[di].offset, size: sz } });
            let last = crate::sig::split_scope(&cls).1.to_string();
            let ctor = mwdec_core::FuncSig {
                qualified_name: format!("{cls}::{last}"),
                mangled: None,
                ret: Type::Void,
                params: vec![],
                this_class: Some(cls.clone()),
                is_const: false,
                is_static: false,
                is_virtual: false,
                variadic: false,
                runs_code: true,
            };
            let _ = k;
            fn drop_rel(b: &mut Vec<Stmt>, f: &dyn Fn(&Stmt) -> bool) {
                b.retain(|s| !f(s));
                for s in b.iter_mut() {
                    match s {
                        Stmt::If { then, els, .. } => {
                            drop_rel(then, f);
                            drop_rel(els, f);
                        }
                        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => drop_rel(body, f),
                        _ => {}
                    }
                }
            }
            let mut rest = b.split_off(i + 1);
            drop_rel(&mut rest, &is_release);
            b.extend(rest);
            b[i] = Stmt::Expr(Expr::Call { callee: Callee::Method { symbol: String::new(), sig: ctor, this: Box::new(Expr::AddrOf(Box::new(Expr::Var(id)))), qualified: false }, args: vec![], ret: Type::Void });
            consumed.push(di);
            i += 1;
        }
    });
    ir.vars.extend(new_vars);
    consumed.sort_unstable();
    for di in consumed.into_iter().rev() {
        ir.dead_stores.remove(di);
    }
    for (n, d) in ir.dead_stores.iter_mut().enumerate() {
        d.order = n;
    }
}

/// The struct return filled member by member at the end, each member read from a frame slot
/// of one frame object (`&result.mCardSize` passed to a call) or taken from a register whose
/// value a dead store into that object's slot holds: a named local of the returned class,
/// `R result; result.m = f(&result.n); return result;` (no return-value optimization).
pub fn returned_frame_object(ir: &mut IrFunction, db: Option<&mwdec_core::TypeDb>) {
    let Some(db) = db else { return };
    let Some(sret) = ir.vars.iter().position(|v| v.kind == VarKind::StructRet) else { return };
    let Some(cls) = pointee(&ir.vars[sret].ty).and_then(|t| named(&crate::types::resolve(Some(db), t)).map(|s| s.to_string())) else { return };
    let mut fields = vec![];
    if !crate::idioms::flat_fields(db, &cls, 0, &mut fields, 0) || fields.len() < 2 || fields.len() > 8 {
        return;
    }
    let Some(size) = crate::types::size_of(Some(db), &Type::Named(cls.clone())) else { return };
    let body = &ir.body;
    let mut end = body.len();
    if matches!(body.last(), Some(Stmt::Return(None))) {
        end -= 1;
    }
    // the trailing member stores
    let mut stores: Vec<(usize, i32, Type, Expr)> = vec![];
    let mut k = end;
    while k > 0 {
        match &body[k - 1] {
            Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } if matches!(**base, Expr::Var(v) if v == sret) => {
                stores.push((k - 1, *offset, ty.clone(), src.clone()));
                k -= 1;
            }
            Stmt::Assign { dst: Expr::Var(_), src } if !src.has_call() => k -= 1,
            _ => break,
        }
    }
    if stores.len() != fields.len() || !fields.iter().all(|(o, _)| stores.iter().filter(|s| s.1 == *o).count() == 1) {
        return;
    }
    let vars = ir.vars.clone();
    let mut uses: std::collections::HashMap<VarId, usize> = std::collections::HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    let def_of = |t: VarId| -> Option<(usize, Expr)> {
        let ds: Vec<(usize, &Expr)> = body.iter().enumerate().filter_map(|(i, s)| match s {
            Stmt::Assign { dst: Expr::Var(x), src } if *x == t => Some((i, src)),
            _ => None,
        }).collect();
        match ds.as_slice() {
            [(i, e)] => Some((*i, (*e).clone())),
            _ => None,
        }
    };
    let slot_of = |e: &Expr| -> Option<VarId> {
        let v = match strip_casts(e) {
            Expr::Var(v) => *v,
            Expr::Member { base, offset: 0, .. } => match **base {
                Expr::Var(v) => v,
                _ => return None,
            },
            _ => return None,
        };
        matches!(vars[v].kind, VarKind::Stack { .. }).then_some(v)
    };
    let off_of = |v: VarId| match vars[v].kind {
        VarKind::Stack { offset, .. } => offset,
        _ => i32::MIN,
    };
    // classify: (field offset, slot local) or (field offset, dead store index)
    let mut slots: Vec<(i32, VarId, Option<usize>)> = vec![];
    let mut deads: Vec<(i32, usize)> = vec![];
    let mut base: Option<i32> = None;
    let field_size = |off: i32| fields.iter().find(|f| f.0 == off).and_then(|f| crate::types::size_of(Some(db), &f.1)).unwrap_or(0);
    for (_, off, _, src) in &stores {
        let fsz = field_size(*off);
        let via = match strip_casts(src) {
            Expr::Var(t) if matches!(vars[*t].kind, VarKind::Local) && uses.get(t) == Some(&1) => def_of(*t),
            _ => None,
        };
        let slot = slot_of(src).or_else(|| via.as_ref().and_then(|(_, e)| slot_of(e)));
        if let Some(l) = slot {
            if !matches!(vars[l].kind, VarKind::Stack { size, .. } if size == fsz) {
                return;
            }
            let o = off_of(l) - off;
            if base.is_some_and(|b| b != o) {
                return;
            }
            base = Some(o);
            slots.push((*off, l, via.map(|v| v.0).filter(|_| slot_of(src).is_none())));
            continue;
        }
        deads.push((*off, usize::MAX));
        let _ = fsz;
    }
    let Some(o) = base else { return };
    for (off, di) in deads.iter_mut() {
        let st = stores.iter().find(|s| s.1 == *off).unwrap();
        let fsz = field_size(*off);
        let Some(i) = ir.dead_stores.iter().position(|d| d.offset == o + *off && d.size == fsz && strip_casts(&d.value) == strip_casts(&st.3)) else { return };
        *di = i;
    }
    // register temps of the register members defined by a call, used only by their store:
    // (field offset, definition index)
    let mut folds: Vec<(i32, usize)> = vec![];
    for (idx, off, _, src) in &stores {
        if !deads.iter().any(|d| d.0 == *off) {
            continue;
        }
        if let Expr::Var(t) = strip_casts(src) {
            if matches!(vars[*t].kind, VarKind::Local) && uses.get(t) == Some(&1) {
                if let Some((di, _)) = def_of(*t) {
                    if (di + 1..*idx).all(|q| matches!(&body[q], Stmt::Assign { dst: Expr::Var(_) | Expr::Load { .. }, src } if !src.has_call())) {
                        folds.push((*off, di));
                    }
                }
            }
        }
    }
    let id = ir.vars.len();
    ir.vars.push(Var { name: "result".into(), ty: Type::Named(cls.clone()), kind: VarKind::Stack { offset: o, size } });
    let fty = |off: i32| fields.iter().find(|f| f.0 == off).map(|f| f.1.clone()).unwrap();
    for (off, l, _) in &slots {
        let m = Expr::Member { base: Box::new(Expr::Var(id)), offset: *off, ty: fty(*off) };
        Stmt::rewrite_exprs(&mut ir.body, &mut |e| {
            match e {
                Expr::Member { base, offset: 0, .. } if matches!(**base, Expr::Var(v) if v == *l) => *e = m.clone(),
                Expr::Var(v) if *v == *l => *e = m.clone(),
                _ => {}
            }
        });
    }
    // drop the member stores (and the temps reading the slots); the register members go into
    // the local where they were stored; then the whole object returned
    let mut rm: Vec<usize> = slots.iter().filter_map(|s| s.2).collect();
    for (idx, off, _, src) in &stores {
        if deads.iter().any(|d| d.0 == *off) {
            // (a register temp holding a call's result, used only here: the call itself)
            let mut val = src.clone();
            if let Some(&(_, di)) = folds.iter().find(|f| f.0 == *off) {
                if let Stmt::Assign { src: e, .. } = &ir.body[di] {
                    val = e.clone();
                    rm.push(di);
                }
            }
            ir.body[*idx] = Stmt::Assign { dst: Expr::Member { base: Box::new(Expr::Var(id)), offset: *off, ty: fty(*off) }, src: val };
        } else {
            rm.push(*idx);
        }
    }
    let ret_at = end;
    ir.body.insert(ret_at, Stmt::Assign { dst: Expr::Load { base: Box::new(Expr::Var(sret)), offset: 0, ty: Type::Named(cls.clone()) }, src: Expr::Var(id) });
    rm.sort_unstable();
    rm.dedup();
    for i in rm.into_iter().rev() {
        ir.body.remove(i);
    }
    let mut dis: Vec<usize> = deads.iter().map(|d| d.1).collect();
    dis.sort_unstable();
    for i in dis.into_iter().rev() {
        ir.dead_stores.remove(i);
    }
    for (n, d) in ir.dead_stores.iter_mut().enumerate() {
        d.order = n;
    }
}
