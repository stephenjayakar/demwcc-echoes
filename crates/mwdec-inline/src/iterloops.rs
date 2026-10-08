//! Loops over containers whose iterators are plain pointers (`rstl::reserved_vector<T, N>`):
//! the header's inline `begin()`/`end()` expand to address arithmetic on the container
//! (`(char*)&v + 4`, `+ v.size() * sizeof(T)`) and `++it` to a byte step, which as written
//! out schedule differently from the source's iterator loop. The loop is rewritten as the source
//! wrote it: `for (const T* it = v.begin(); it != v.end(); ++it)` (as a `while`), the pointer
//! typed `T*` so element accesses read as members.

use crate::util::strip;
use mwdec_core::{FuncSig, Type, TypeDb};
use mwdec_lift::{BinOp, Callee, Expr, IrFunction, Stmt, VarId, VarKind};

/// `rstl::reserved_vector<T, N>`: (T, element size, offset of the data).
fn pointer_container(db: &TypeDb, cls: &str) -> Option<(Type, i64, i32)> {
    let base = cls.split('<').next()?.trim();
    if base != "rstl::reserved_vector" {
        return None;
    }
    let inner = &cls[cls.find('<')? + 1..cls.rfind('>')?];
    let args = mwdec_lift::sig::split_top(inner, ',');
    let t = mwdec_lift::sig::parse_type(args.first()?.trim());
    let n: i64 = args.get(1)?.trim().parse().ok()?;
    let c = mwdec_lift::sig::find_class(db, cls)?;
    let data = c.fields.iter().find(|f| matches!(strip(&mwdec_lift::types::resolve(Some(db), &f.ty)), Type::Array(..)))?;
    // (the storage is `uchar mData[N * sizeof(T)]`: the element size without T's layout)
    let bytes = mwdec_lift::types::size_of(Some(db), &data.ty)? as i64;
    let size = if n > 0 { bytes / n } else { 0 };
    (size > 0).then_some((t, size, data.offset as i32))
}

fn strip_casts(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip_casts(e),
        e => e,
    }
}

/// `(char*)base + k` / `&base->field` forms: (base, k).
fn base_plus(e: &Expr) -> Option<(&Expr, i64)> {
    match strip_casts(e) {
        Expr::Binary { op: BinOp::Add, l, r, .. } => Some((strip_casts(l), r.as_int()?)),
        Expr::AddrOf(x) => match &**x {
            Expr::Load { base, offset, .. } => Some((strip_casts(base), *offset as i64)),
            Expr::Member { base, offset, .. } => match &**base {
                Expr::Var(_) => Some((&**base, *offset as i64)),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

/// The container a data pointer `begin` points into: (container lvalue, class, element type,
/// element size, const).
fn container_of(ir: &IrFunction, db: &TypeDb, begin: &Expr) -> Option<(Expr, String, Type, i64, bool)> {
    let (b, k) = base_plus(begin)?;
    let Expr::Var(v) = b else { return None };
    // a member of `this`
    if Some(*v) == ir.this_var {
        let own = ir.sig.this_class.as_deref()?;
        let c = mwdec_lift::sig::find_class(db, own)?;
        for f in &c.fields {
            let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
            let Some(cls) = crate::util::class_name(&ft, db) else { continue };
            let Some((t, size, data)) = pointer_container(db, &cls) else { continue };
            if f.offset as i64 + data as i64 == k {
                let lv = Expr::Load { base: Box::new(Expr::Var(*v)), offset: f.offset as i32, ty: ft.clone() };
                return Some((lv, cls, t, size, ir.sig.is_const));
            }
        }
        return None;
    }
    // a parameter (the object itself, passed by reference)
    if matches!(ir.vars[*v].kind, VarKind::Param { .. }) {
        let pt = ir.vars[*v].ty.clone();
        let (inner, cst) = match &pt {
            Type::Const(x) => ((**x).clone(), true),
            Type::Ref(x) | Type::Ptr(x) => (strip(x).clone(), matches!(&**x, Type::Const(_))),
            t => (t.clone(), false),
        };
        let cls = crate::util::class_name(&inner, db)?;
        let (t, size, data) = pointer_container(db, &cls)?;
        if data as i64 == k {
            return Some((Expr::Var(*v), cls, t, size, cst));
        }
    }
    None
}

fn is_step(s: &Stmt, p: VarId, size: i64) -> bool {
    match s {
        Stmt::Assign { dst: Expr::Var(x), src } if *x == p => matches!(base_plus(src), Some((Expr::Var(y), k)) if *y == p && k == size),
        _ => false,
    }
}

/// Replace every step of `p` in `body` by `++p`; false if `p` is assigned otherwise.
fn rewrite_steps(body: &mut [Stmt], p: VarId, size: i64) -> bool {
    let mut ok = true;
    for s in body.iter_mut() {
        if is_step(s, p, size) {
            *s = Stmt::Expr(Expr::IncDec { e: Box::new(Expr::Var(p)), delta: 1, post: false });
            continue;
        }
        match s {
            Stmt::Assign { dst: Expr::Var(x), .. } if *x == p => ok = false,
            Stmt::If { then, els, .. } => ok &= rewrite_steps(then, p, size) && rewrite_steps(els, p, size),
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => ok &= rewrite_steps(body, p, size),
            Stmt::For { init, step, body, .. } => ok &= rewrite_steps(init, p, size) && rewrite_steps(step, p, size) && rewrite_steps(body, p, size),
            Stmt::Switch { cases, .. } => {
                for c in cases.iter_mut() {
                    ok &= rewrite_steps(&mut c.body, p, size);
                }
            }
            _ => {}
        }
    }
    ok
}

fn mentions(s: &Stmt, v: VarId) -> bool {
    let mut hit = false;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| hit |= matches!(e, Expr::Var(x) if *x == v));
    hit
}

fn method(cls: &str, name: &str, ret: Type, is_const: bool) -> FuncSig {
    FuncSig { qualified_name: format!("{cls}::{name}"), mangled: None, ret, params: vec![], this_class: Some(cls.to_string()), is_const, is_static: false, is_virtual: false, variadic: false }
}

/// Rewrite pointer-walk loops over `reserved_vector`s; returns the number rewritten.
pub fn apply(ir: &mut IrFunction, db: &TypeDb) -> usize {
    let mut n = 0;
    let mut k = 0;
    while k < ir.body.len() {
        if let Some(()) = try_loop(ir, db, k) {
            n += 1;
        }
        k += 1;
    }
    if n > 0 {
        unsigned_locals(ir);
    }
    n
}

/// Signed locals defined once and only read as unsigned (`(unsigned
/// int)t`): unsigned, as the source declared them (a container element of the loop's search).
fn unsigned_locals(ir: &mut IrFunction) {
    let vars = ir.vars.clone();
    for v in 0..vars.len() {
        if vars[v].kind != VarKind::Local || !matches!(vars[v].ty, Type::Int { size: 4, signed: true } | Type::Unknown { size: 4 }) {
            continue;
        }
        let mut defs = vec![];
        fn collect(body: &[Stmt], v: VarId, out: &mut Vec<Expr>) {
            for s in body {
                match s {
                    Stmt::Assign { dst: Expr::Var(x), src } if *x == v => out.push(src.clone()),
                    Stmt::If { then, els, .. } => {
                        collect(then, v, out);
                        collect(els, v, out);
                    }
                    Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect(body, v, out),
                    _ => {}
                }
            }
        }
        collect(&ir.body, v, &mut defs);
        // (a 32-bit value: the same bits either way)
        let [_src] = defs.as_slice() else { continue };
        // every read is a cast to unsigned int
        let mut reads = 0usize;
        let mut casts = 0usize;
        Stmt::walk_exprs(&ir.body, &mut |e| {
            match e {
                Expr::Var(x) if *x == v => reads += 1,
                Expr::Cast { ty: Type::Int { size: 4, signed: false }, e } if matches!(&**e, Expr::Var(x) if *x == v) => casts += 1,
                _ => {}
            }
        });
        // (reads count the definition's destination too)
        if casts == 0 || casts + 1 != reads {
            continue;
        }
        ir.vars[v].ty = Type::Int { size: 4, signed: false };
        Stmt::rewrite_exprs(&mut ir.body, &mut |e| {
            if let Expr::Cast { ty: Type::Int { size: 4, signed: false }, e: x } = e {
                if matches!(&**x, Expr::Var(y) if *y == v) {
                    *e = Expr::Var(v);
                }
            }
        });
    }
}

fn try_loop(ir: &mut IrFunction, db: &TypeDb, w: usize) -> Option<()> {
    let Stmt::While { cond, .. } = &ir.body[w] else { return None };
    // (`p != end && more`: the search condition folded into the loop's)
    let (test, more) = match cond {
        Expr::Binary { op: BinOp::LogAnd, l, r, .. } => (&**l, Some((**r).clone())),
        c => (c, None),
    };
    let Expr::Binary { op: BinOp::Ne, l, r, .. } = test else { return None };
    // `p != end` (either order): p a register local
    let (p, end) = match (strip_casts(l), strip_casts(r)) {
        (Expr::Var(a), e) if matches!(ir.vars[*a].kind, VarKind::Local) && !matches!(e, Expr::Var(b) if b == a) => (*a, e.clone()),
        (e, Expr::Var(b)) if matches!(ir.vars[*b].kind, VarKind::Local) => (*b, e.clone()),
        _ => return None,
    };
    // its single definition before the loop: the container's data
    let defs: Vec<usize> = (0..w).filter(|&i| matches!(&ir.body[i], Stmt::Assign { dst: Expr::Var(x), .. } if *x == p)).collect();
    let [d] = defs.as_slice() else { return None };
    let Stmt::Assign { src: begin, .. } = &ir.body[*d] else { return None };
    let (cont, cls, t, size, cst) = container_of(ir, db, begin)?;
    // the end: `p`'s start plus the count (directly, or a local defined so)
    let (end_var, end_expr) = match &end {
        Expr::Var(e) => {
            let ds: Vec<usize> = (0..w).filter(|&i| matches!(&ir.body[i], Stmt::Assign { dst: Expr::Var(x), .. } if x == e)).collect();
            let [de] = ds.as_slice() else { return None };
            let Stmt::Assign { src, .. } = &ir.body[*de] else { return None };
            (Some((*e, *de)), src.clone())
        }
        e => (None, e.clone()),
    };
    // (it reads the container's size: a `size()` call or its count)
    let mut reads_count = false;
    end_expr.walk(&mut |x| {
        reads_count |= matches!(x, Expr::Call { callee: Callee::Method { sig, .. }, .. } if sig.qualified_name.ends_with("::size"));
        reads_count |= matches!(x, Expr::Load { .. } | Expr::Member { .. });
    });
    if !reads_count {
        return None;
    }
    // the end local is used by nothing else before the loop (after it, it is `v.end()` again)
    if let Some((e, de)) = end_var {
        if ir.body.iter().enumerate().any(|(i, s)| i != de && i < w && mentions(s, e)) {
            return None;
        }
        if let Stmt::While { body, .. } = &ir.body[w] {
            if body.iter().any(|s| mentions(s, e)) {
                return None;
            }
        }
    }
    // p: defined once before, stepped by the element size in the loop, not used before it
    if (d + 1..w).any(|i| Some(i) != end_var.map(|x| x.1) && mentions(&ir.body[i], p)) {
        return None;
    }
    let mut body = match &ir.body[w] {
        Stmt::While { body, .. } => body.clone(),
        _ => return None,
    };
    if !rewrite_steps(&mut body, p, size) {
        return None;
    }
    // (stepped at least once)
    let mut stepped = false;
    Stmt::walk_exprs(&body, &mut |e| stepped |= matches!(e, Expr::IncDec { e, .. } if matches!(&**e, Expr::Var(x) if *x == p)));
    if !stepped {
        return None;
    }
    let pt = Type::Ptr(Box::new(if cst { Type::Const(Box::new(t)) } else { t }));
    let obj = Box::new(Expr::AddrOf(Box::new(cont)));
    let call = |name: &str| Expr::Call { callee: Callee::Method { symbol: String::new(), sig: method(&cls, name, pt.clone(), cst), this: obj.clone(), qualified: false }, args: vec![], ret: pt.clone() };
    let mut new_cond = Expr::Binary { op: BinOp::Ne, l: Box::new(Expr::Var(p)), r: Box::new(call("end")), ty: Type::Bool };
    if let Some(m) = more {
        if let Some((e, _)) = end_var {
            if m.uses_var(e) {
                return None;
            }
        }
        new_cond = Expr::Binary { op: BinOp::LogAnd, l: Box::new(new_cond), r: Box::new(m), ty: Type::Bool };
    }
    ir.vars[p].ty = pt.clone();
    ir.body[w] = Stmt::While { cond: new_cond, body };
    if let Some((e, _)) = end_var {
        let end_call = call("end");
        Stmt::rewrite_exprs(&mut ir.body[w + 1..], &mut |x| {
            if matches!(x, Expr::Var(y) if *y == e) {
                *x = end_call.clone();
            }
        });
    }
    // `p = v.begin()` right before the loop
    let init = Stmt::Assign { dst: Expr::Var(p), src: call("begin") };
    let mut gone = vec![*d];
    if let Some((_, de)) = end_var {
        gone.push(de);
    }
    gone.sort_unstable();
    let mut w = w;
    for &i in gone.iter().rev() {
        ir.body.remove(i);
        w -= 1;
    }
    ir.body.insert(w, init);
    Some(())
}
