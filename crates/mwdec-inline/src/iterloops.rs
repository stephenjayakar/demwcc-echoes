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
    FuncSig { qualified_name: format!("{cls}::{name}"), mangled: None, ret, params: vec![], this_class: Some(cls.to_string()), is_const, is_static: false, is_virtual: false, variadic: false, runs_code: false }
}

/// Rewrite pointer-walk loops over `reserved_vector`s; returns the number rewritten.
pub fn apply(ir: &mut IrFunction, db: &TypeDb) -> usize {
    let mut n = iterator_steps(ir, db);
    let mut k = 0;
    while k < ir.body.len() {
        if try_loop(ir, db, k).or_else(|| try_index_loop(ir, db, k)).is_some() {
            n += 1;
        }
        k += 1;
    }
    if n > 0 {
        unsigned_locals(ir);
        object_pointer_locals(ir);
        n += push_backs(ir, db);
    }
    n
}

/// `it.current = it.current->mPrev` (`mNext`) on a list iterator member: `--it` (`++it`).
fn iterator_steps(ir: &mut IrFunction, db: &TypeDb) -> usize {
    let Some(this) = ir.this_var else { return 0 };
    let Some(own) = ir.sig.this_class.clone() else { return 0 };
    let Some(c) = mwdec_lift::sig::find_class(db, &own) else { return 0 };
    // list iterator members: offset -> (type, mPrev offset, mNext offset)
    let mut members: Vec<(i32, Type, i32, i32)> = vec![];
    for f in &c.fields {
        let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
        let Some(cls) = crate::util::class_name(&ft, db) else { continue };
        let Some(list) = cls.strip_suffix("::iterator").or_else(|| cls.strip_suffix("::const_iterator")) else { continue };
        if list_layout(db, list).is_none() {
            continue;
        }
        let (prev, next) = match mwdec_lift::sig::find_class(db, &format!("{list}::node")) {
            Some(n) => {
                let o = |x: &str| n.fields.iter().find(|f| f.name == x).map(|f| f.offset as i32);
                (o("mPrev").unwrap_or(0), o("mNext").unwrap_or(4))
            }
            None => (0, 4),
        };
        members.push((f.offset as i32, ft, prev, next));
    }
    if members.is_empty() {
        return 0;
    }
    let mut n = 0;
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        for s in b.iter_mut() {
            let Stmt::Assign { dst: Expr::Load { base, offset: k, .. }, src } = s else { continue };
            if !matches!(strip_casts(base), Expr::Var(v) if *v == this) {
                continue;
            }
            let Some((_, ty, prev, next)) = members.iter().find(|m| m.0 == *k) else { continue };
            let Expr::Load { base: cur, offset: o, .. } = strip_casts(src) else { continue };
            let same = matches!(strip_casts(cur), Expr::Load { base: b2, offset: k2, .. } if k2 == k && matches!(strip_casts(b2), Expr::Var(v) if *v == this));
            let delta = if *o == *prev { -1 } else if *o == *next { 1 } else { continue };
            if !same || prev == next {
                continue;
            }
            let lv = Expr::Load { base: Box::new(Expr::Var(this)), offset: *k, ty: ty.clone() };
            *s = Stmt::Expr(Expr::IncDec { e: Box::new(lv), delta, post: false });
            n += 1;
        }
    });
    n
}

/// The location an lvalue reads (`o.m`, `*(T*)((char*)p + k)`, `p->m`): (root, byte offset).
fn word_at(e: &Expr) -> Option<(Expr, i64)> {
    fn ptr(e: &Expr) -> Option<(Expr, i64)> {
        match strip_casts(e) {
            Expr::Binary { op: BinOp::Add, l, r, .. } => {
                let (b, k) = ptr(l)?;
                Some((b, k + r.as_int()?))
            }
            Expr::AddrOf(x) => word_at(x),
            x @ Expr::Var(_) => Some((x.clone(), 0)),
            _ => None,
        }
    }
    match strip_casts(e) {
        Expr::Load { base, offset, .. } => {
            let (b, k) = ptr(base)?;
            Some((b, k + *offset as i64))
        }
        Expr::Member { base, offset, .. } => {
            let (b, k) = word_at(base)?;
            Some((b, k + *offset as i64))
        }
        // (an object variable: its own address)
        x @ Expr::Var(_) => Some((Expr::AddrOf(Box::new(x.clone())), 0)),
        _ => None,
    }
}

/// `*v.end() = x; ++v.mCount;` (the expansion of `reserved_vector::push_back`): `v.push_back(x)`,
/// with `x` the object the element was copied from (a by-value or reference parameter).
fn push_backs(ir: &mut IrFunction, db: &TypeDb) -> usize {
    let mut n = 0;
    let mut k = 0;
    while k + 1 < ir.body.len() {
        let hit = (|| {
            let Stmt::Assign { dst: Expr::Load { base, offset: 0, ty }, src } = &ir.body[k] else { return None };
            let Expr::Call { callee: Callee::Method { sig, this: obj, .. }, .. } = &**base else { return None };
            let cls = sig.this_class.clone()?;
            if !sig.qualified_name.ends_with("::end") {
                return None;
            }
            let (t, size, _) = pointer_container(db, &cls)?;
            if mwdec_lift::types::size_of(Some(db), ty).map(|s| s as i64) != Some(size) {
                return None;
            }
            // the count increment right after: `v.mCount += 1` on the same container
            let Expr::AddrOf(cont) = &**obj else { return None };
            let at = word_at(cont)?;
            let Stmt::Assign { dst: d2, src: inc } = &ir.body[k + 1] else { return None };
            if word_at(d2) != Some(at.clone()) {
                return None;
            }
            let Expr::Binary { op: BinOp::Add, l, r, .. } = strip_casts(inc) else { return None };
            if r.as_int() != Some(1) || word_at(strip_casts(l)) != Some(at) {
                return None;
            }
            // the copied object: a parameter's whole value
            let arg = match strip_casts(src) {
                Expr::Member { base: pb, offset: 0, .. } => match &**pb {
                    Expr::Var(p) if matches!(ir.vars[*p].kind, VarKind::Param { .. }) => Expr::Var(*p),
                    _ => return None,
                },
                // (a by-value parameter of the element's class, whole)
                Expr::Var(p) if matches!(ir.vars[*p].kind, VarKind::Param { .. }) && crate::util::class_name(&ir.vars[*p].ty, db).is_some_and(|c| Some(c) == crate::util::class_name(&t, db)) => Expr::Var(*p),
                e if crate::util::class_name(&t, db).is_none() => e.clone(),
                _ => return None,
            };
            let ps = vec![mwdec_core::Param { name: None, ty: Type::Ref(Box::new(Type::Const(Box::new(t.clone())))) }];
            let sig = FuncSig { qualified_name: format!("{cls}::push_back"), mangled: None, ret: Type::Void, params: ps, this_class: Some(cls.clone()), is_const: false, is_static: false, is_virtual: false, variadic: false, runs_code: false };
            Some(Stmt::Expr(Expr::Call { callee: Callee::Method { symbol: String::new(), sig, this: obj.clone(), qualified: false }, args: vec![arg], ret: Type::Void }))
        })();
        if let Some(st) = hit {
            ir.body[k] = st;
            ir.body.remove(k + 1);
            n += 1;
        }
        k += 1;
    }
    n
}

/// An untyped local holding the address of an element's member object (`t = &it->second`) and
/// used as a pointer to its class: typed so, and calls on that object go through it (the
/// source's `CAdditiveAnimPlayback& anim = it->second;`).
fn object_pointer_locals(ir: &mut IrFunction) {
    let vars = ir.vars.clone();
    for v in 0..vars.len() {
        if vars[v].kind != VarKind::Local || !matches!(strip(&vars[v].ty), Type::Ptr(x) if matches!(strip(x), Type::Void | Type::Unknown { .. } | Type::Int { size: 1, .. } | Type::Char)) {
            continue;
        }
        let defs: Vec<Expr> = ir.body.iter().flat_map(|s| collect_defs(s, v)).collect();
        let [src] = defs.as_slice() else { continue };
        let Expr::AddrOf(lv) = strip_casts(src) else { continue };
        let lt = mwdec_lift::types::ty_of(lv, &vars);
        // the class: the object's type, or that of the methods called on it
        let mut recv: Option<String> = None;
        Stmt::walk_exprs(&ir.body, &mut |e| {
            if let Expr::Call { callee: Callee::Method { this, sig, .. }, .. } = e {
                if matches!(&**this, Expr::Var(x) if *x == v) {
                    recv = recv.clone().or(sig.this_class.clone());
                }
            }
        });
        let cls = match strip(&lt) {
            Type::Named(c) => c.clone(),
            _ => match recv {
                Some(c) => c,
                None => continue,
            },
        };
        let pt = Type::Ptr(Box::new(Type::Named(cls.clone())));
        // every use: a cast to that pointer type
        let (mut uses, mut casts) = (0usize, 0usize);
        let same_cls = |c: &Option<String>| c.as_deref().is_some_and(|c| mwdec_lift::sig::norm_name(c) == mwdec_lift::sig::norm_name(&cls));
        Stmt::walk_exprs(&ir.body, &mut |e| match e {
            Expr::Var(x) if *x == v => uses += 1,
            Expr::Cast { ty, e } if *ty == pt && matches!(&**e, Expr::Var(x) if *x == v) => casts += 1,
            // (a method of the class called on it)
            Expr::Call { callee: Callee::Method { this, sig, .. }, .. } if matches!(&**this, Expr::Var(x) if *x == v) && same_cls(&sig.this_class) => casts += 1,
            _ => {}
        });
        if casts == 0 || casts + 1 != uses {
            continue;
        }
        ir.vars[v].ty = pt.clone();
        let lv = (**lv).clone();
        // reads of the object's members through the element pointer go through it too
        let (obase, ooff) = match &lv {
            Expr::Load { base, offset, .. } => ((**base).clone(), *offset),
            _ => (Expr::Unknown { text: String::new(), ty: Type::Void }, i32::MIN),
        };
        let osize = mwdec_lift::types::size_of(None, &Type::Named(cls.clone())).unwrap_or(0) as i32;
        let def_at = ir.body.iter().position(|s| !collect_defs(s, v).is_empty());
        Stmt::rewrite_exprs(&mut ir.body, &mut |e| {
            if matches!(e, Expr::Cast { ty, e: x } if *ty == pt && matches!(&**x, Expr::Var(y) if *y == v)) {
                *e = Expr::Var(v);
            }
            if let Expr::Call { callee: Callee::Method { this, .. }, .. } = e {
                if matches!(&**this, Expr::AddrOf(x) if **x == lv) {
                    **this = Expr::Var(v);
                }
            }
        });
        if ooff != i32::MIN {
            // (after the definition, in the same statement list)
            let at = def_at.unwrap_or(0);
            fn rebase(body: &mut [Stmt], v: VarId, obase: &Expr, ooff: i32, size: i32) {
                Stmt::rewrite_exprs(body, &mut |e| {
                    if let Expr::Load { base, offset, ty } = e {
                        if **base == *obase && *offset > ooff && (size <= 0 || *offset < ooff + size) {
                            *e = Expr::Load { base: Box::new(Expr::Var(v)), offset: *offset - ooff, ty: ty.clone() };
                        }
                    }
                });
            }
            for s in ir.body.iter_mut().skip(at + 1) {
                rebase(std::slice::from_mut(s), v, &obase, ooff, osize);
            }
            if let Some(Stmt::If { then, .. }) = ir.body.get_mut(at) {
                // the definition inside an `if`: its siblings after it
                if let Some(k) = then.iter().position(|s| !collect_defs(s, v).is_empty()) {
                    rebase(&mut then[k + 1..], v, &obase, ooff, osize);
                }
            }
        }
    }
}

fn collect_defs(s: &Stmt, v: VarId) -> Vec<Expr> {
    let mut out = vec![];
    match s {
        Stmt::Assign { dst: Expr::Var(x), src } if *x == v => out.push(src.clone()),
        Stmt::If { then, els, .. } => {
            for t in then.iter().chain(els) {
                out.extend(collect_defs(t, v));
            }
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
            for t in body {
                out.extend(collect_defs(t, v));
            }
        }
        _ => {}
    }
    out
}

/// Marker of a rewritten list loop kept out of inline folding (see [`lists_prefold`]).
const STASH: &str = "mwdec iterloop ";

/// Rewrite list walks before inline folding (which would merge the walking pointer with its
/// first value), keeping the rewritten loops out of the folding: each `it = l.begin(); while
/// (...) {...}` pair is replaced by a marker until [`unstash`] puts it back.
pub fn lists_prefold(ir: &mut IrFunction, db: &TypeDb) -> Vec<Vec<Stmt>> {
    let mut stash = vec![];
    let mut k = 0;
    while k < ir.body.len() {
        if let Some(w) = try_list_loop(ir, db, k) {
            // (its init just before it)
            if w >= 1 && matches!(ir.body[w], Stmt::While { .. }) {
                k = w;
                let pair: Vec<Stmt> = ir.body.drain(w - 1..=w).collect();
                ir.body.insert(w - 1, Stmt::Comment(format!("{STASH}{}", stash.len())));
                stash.push(pair);
            }
        } else if let Some(w) = try_map_loop(ir, db, k) {
            let one = std::mem::replace(&mut ir.body[w], Stmt::Comment(format!("{STASH}{}", stash.len())));
            stash.push(vec![one]);
            k = w;
        } else if let Some(at) = try_list_find(ir, db, k) {
            let one = std::mem::replace(&mut ir.body[at], Stmt::Comment(format!("{STASH}{}", stash.len())));
            stash.push(vec![one]);
            k = at;
        }
        k += 1;
    }
    if !stash.is_empty() {
        unsigned_locals(ir);
    }
    stash
}

/// Put the loops [`lists_prefold`] kept aside back.
pub fn unstash(ir: &mut IrFunction, stash: Vec<Vec<Stmt>>) {
    for (i, pair) in stash.into_iter().enumerate().rev() {
        let mark = format!("{STASH}{i}");
        if let Some(at) = ir.body.iter().position(|s| matches!(s, Stmt::Comment(c) if *c == mark)) {
            ir.body.splice(at..=at, pair);
        }
    }
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
    let w = sink_param_reads(ir, w + 1);
    as_find(ir, db, w, p, &cls, size);
    Some(())
}

/// `it = v.begin(); while (it != v.end() && *it != x) ++it;` is `rstl::find`'s expansion:
/// `it = rstl::find(v.begin(), v.end(), x);`.
fn as_find(ir: &mut IrFunction, db: &TypeDb, w: usize, p: VarId, cls: &str, size: i64) -> Option<()> {
    let Stmt::While { cond, body } = &ir.body[w] else { return None };
    if !matches!(body.as_slice(), [Stmt::Expr(Expr::IncDec { e, .. })] if matches!(&**e, Expr::Var(x) if *x == p)) {
        return None;
    }
    let Expr::Binary { op: BinOp::LogAnd, l: test, r: cmp, .. } = cond else { return None };
    let Expr::Binary { op: BinOp::Ne, r: end, .. } = &**test else { return None };
    let Expr::Binary { op: BinOp::Ne, l: a, r: b, .. } = strip_casts(cmp) else { return None };
    // one side the element (its only scalar, the element's whole size), the other the value
    let elem = |e: &Expr| matches!(strip_casts(e), Expr::Load { base, offset: 0, ty } if matches!(&**base, Expr::Var(x) if *x == p) && mwdec_lift::types::size_of(Some(db), ty).map(|s| s as i64) == Some(size));
    let value = |e: &Expr| -> Option<Expr> {
        if e.uses_var(p) {
            return None;
        }
        match strip_casts(e) {
            // a parameter object's only member: the object
            Expr::Member { base, offset: 0, ty } if matches!(&**base, Expr::Var(v) if matches!(ir.vars[*v].kind, VarKind::Param { .. })) && mwdec_lift::types::size_of(Some(db), ty).map(|s| s as i64) == Some(size) => Some((**base).clone()),
            Expr::Var(v) if matches!(ir.vars[*v].kind, VarKind::Param { .. }) => Some(e.clone()),
            _ => None,
        }
    };
    let val = if elem(a) { value(b)? } else if elem(b) { value(a)? } else { return None };
    let Stmt::Assign { dst: Expr::Var(x), src: begin } = &ir.body[w - 1] else { return None };
    if *x != p {
        return None;
    }
    let pt = ir.vars[p].ty.clone();
    let find = FuncSig { qualified_name: "rstl::find".into(), mangled: None, ret: pt.clone(), params: vec![], this_class: None, is_const: false, is_static: false, is_virtual: false, variadic: false, runs_code: false };
    let call = Expr::Call { callee: Callee::Direct { symbol: "rstl::find".into(), sig: find }, args: vec![begin.clone(), (**end).clone(), val], ret: pt };
    let _ = cls;
    ir.body[w - 1] = Stmt::Assign { dst: Expr::Var(p), src: call };
    ir.body.remove(w);
    Some(())
}

// ---------------------------------------------------------------- rstl::list

/// `rstl::list<T, A>`: offsets of `mStart` and `mEnd`, and the node's `mNext` / item offsets.
fn list_layout(db: &TypeDb, cls: &str) -> Option<(i32, i32, i32, i32)> {
    if cls.split('<').next()?.trim() != "rstl::list" {
        return None;
    }
    let c = mwdec_lift::sig::find_class(db, cls)?;
    let off = |n: &str| c.fields.iter().find(|f| f.name == n).map(|f| f.offset as i32);
    let (start, end) = (off("mStart")?, off("mEnd")?);
    let (next, item) = match mwdec_lift::sig::find_class(db, &format!("{cls}::node")) {
        Some(n) => {
            let o = |x: &str| n.fields.iter().find(|f| f.name == x).map(|f| f.offset as i32);
            (o("mNext").unwrap_or(4), o("mItem").unwrap_or(8))
        }
        None => (4, 8),
    };
    Some((start, end, next, item))
}

/// The list a node pointer `begin` (`l.mStart`) starts: (list lvalue, class, layout, const).
fn list_of(ir: &IrFunction, db: &TypeDb, begin: &Expr) -> Option<(Expr, String, (i32, i32, i32, i32), bool)> {
    let Expr::Load { base, offset, .. } = strip_casts(begin) else { return None };
    let Expr::Var(v) = strip_casts(base) else { return None };
    let k = *offset;
    if Some(*v) == ir.this_var {
        let own = ir.sig.this_class.as_deref()?;
        let c = mwdec_lift::sig::find_class(db, own)?;
        for f in &c.fields {
            let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
            let Some(cls) = crate::util::class_name(&ft, db) else { continue };
            let Some(lay) = list_layout(db, &cls) else { continue };
            if f.offset as i32 + lay.0 == k {
                let lv = Expr::Load { base: Box::new(Expr::Var(*v)), offset: f.offset as i32, ty: ft.clone() };
                return Some((lv, cls, lay, ir.sig.is_const));
            }
        }
        return None;
    }
    if matches!(ir.vars[*v].kind, VarKind::Param { .. }) {
        let pt = ir.vars[*v].ty.clone();
        let (inner, cst) = match &pt {
            Type::Ref(x) | Type::Ptr(x) => (strip(x).clone(), matches!(&**x, Type::Const(_))),
            Type::Const(x) => ((**x).clone(), true),
            t => (t.clone(), false),
        };
        let cls = crate::util::class_name(&inner, db)?;
        let lay = list_layout(db, &cls)?;
        if lay.0 == k {
            let lv = match &pt {
                Type::Ptr(_) => Expr::Load { base: Box::new(Expr::Var(*v)), offset: 0, ty: inner.clone() },
                _ => Expr::Var(*v),
            };
            return Some((lv, cls, lay, cst));
        }
    }
    None
}

/// `(base var, offset)` of a container lvalue built by `list_of`.
fn lv_loc(e: &Expr) -> Option<(VarId, i32)> {
    match e {
        Expr::Var(v) => Some((*v, 0)),
        Expr::Load { base, offset, .. } => match strip_casts(base) {
            Expr::Var(v) => Some((*v, *offset)),
            _ => None,
        },
        _ => None,
    }
}

/// A walk over a list's nodes as an iterator loop:
/// `for (iterator it = l.begin(); it != l.end(); ++it) { ... it->m ... return it; } return l.end();`
fn try_list_loop(ir: &mut IrFunction, db: &TypeDb, w: usize) -> Option<usize> {
    let Stmt::While { cond, body } = &ir.body[w] else { return None };
    let Expr::Binary { op: BinOp::Ne, l, r, .. } = cond else { return None };
    let (a, b) = (strip_casts(l).clone(), strip_casts(r).clone());
    // the node pointer: the local defined from the list's start
    let mut found = None;
    for (pv, other) in [(&a, &b), (&b, &a)] {
        let Expr::Var(p) = pv else { continue };
        if !matches!(ir.vars[*p].kind, VarKind::Local) {
            continue;
        }
        let defs: Vec<usize> = (0..w).filter(|&i| matches!(&ir.body[i], Stmt::Assign { dst: Expr::Var(x), .. } if x == p)).collect();
        let [d] = defs.as_slice() else { continue };
        let Stmt::Assign { src, .. } = &ir.body[*d] else { continue };
        if let Some(l) = list_of(ir, db, src) {
            found = Some((*p, *d, other.clone(), l));
            break;
        }
    }
    let (p, d, end, (cont, cls, (_start, endo, next, item), cst)) = found?;
    let (bv, boff) = lv_loc(&cont)?;
    // the end: `l.mEnd`, directly or through a local
    let is_end_load = |e: &Expr| matches!(strip_casts(e), Expr::Load { base, offset, .. } if matches!(strip_casts(base), Expr::Var(v) if *v == bv) && *offset == boff + endo);
    let end_var = match &end {
        Expr::Var(e) => {
            let ds: Vec<usize> = (0..w).filter(|&i| matches!(&ir.body[i], Stmt::Assign { dst: Expr::Var(x), .. } if x == e)).collect();
            let [de] = ds.as_slice() else { return None };
            let Stmt::Assign { src, .. } = &ir.body[*de] else { return None };
            if !is_end_load(src) {
                return None;
            }
            Some((*e, *de))
        }
        e if is_end_load(e) => None,
        _ => return None,
    };
    // not used between its definition and the loop
    if (d + 1..w).any(|i| Some(i) != end_var.map(|x| x.1) && mentions(&ir.body[i], p)) {
        return None;
    }
    let iter_cls = format!("{cls}::{}", if cst { "const_iterator" } else { "iterator" });
    let it_ty = Type::Named(iter_cls.clone());
    // the element type: the list's first template argument
    let elem = cls.find('<').and_then(|i| cls.rfind('>').map(|j| &cls[i + 1..j])).and_then(|inner| mwdec_lift::sig::split_top(inner, ',').first().map(|t| mwdec_lift::sig::parse_type(t.trim()))).unwrap_or(Type::Unknown { size: 0 });
    let elem = if cst { Type::Const(Box::new(elem)) } else { elem };
    let elem_ptr = Type::Ptr(Box::new(elem));
    let obj = Box::new(Expr::AddrOf(Box::new(cont.clone())));
    let call = |name: &str| Expr::Call { callee: Callee::Method { symbol: String::new(), sig: method(&cls, name, it_ty.clone(), cst), this: obj.clone(), qualified: false }, args: vec![], ret: it_ty.clone() };
    let arrow = Expr::Call {
        callee: Callee::Method { symbol: String::new(), sig: FuncSig { qualified_name: format!("{iter_cls}::operator->"), mangled: None, ret: elem_ptr.clone(), params: vec![], this_class: Some(iter_cls.clone()), is_const: true, is_static: false, is_virtual: false, variadic: false, runs_code: false }, this: Box::new(Expr::AddrOf(Box::new(Expr::Var(p)))), qualified: false },
        args: vec![],
        ret: elem_ptr,
    };
    // the loop body: `p = p->mNext` steps become `++it`, element reads `it->m`, `iterator(p)`
    // `it`; any other use of the node pointer and the rewrite is off
    let mut new_body = body.clone();
    let mut ok = true;
    let mut stepped = false;
    fn walk_steps(b: &mut [Stmt], p: VarId, next: i32, stepped: &mut bool, ok: &mut bool) {
        for s in b.iter_mut() {
            let is_step = matches!(s, Stmt::Assign { dst: Expr::Var(x), src } if *x == p && matches!(strip_casts(src), Expr::Load { base, offset, .. } if matches!(strip_casts(base), Expr::Var(y) if *y == p) && *offset == next));
            if is_step {
                *s = Stmt::Expr(Expr::IncDec { e: Box::new(Expr::Var(p)), delta: 1, post: false });
                *stepped = true;
                continue;
            }
            match s {
                Stmt::Assign { dst: Expr::Var(x), .. } if *x == p => *ok = false,
                Stmt::If { then, els, .. } => {
                    walk_steps(then, p, next, stepped, ok);
                    walk_steps(els, p, next, stepped, ok);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => walk_steps(body, p, next, stepped, ok),
                _ => {}
            }
        }
    }
    walk_steps(&mut new_body, p, next, &mut stepped, &mut ok);
    if !ok || !stepped {
        return None;
    }
    let rewrite = |e: &mut Expr, ok: &mut bool| {
        e.rewrite(&mut |x| {
            let repl = match x {
                Expr::Load { base, offset, ty } if matches!(strip_casts(base), Expr::Var(y) if *y == p) && *offset >= item => Some(Expr::Load { base: Box::new(arrow.clone()), offset: *offset - item, ty: ty.clone() }),
                Expr::Construct { args, .. } if matches!(args.as_slice(), [a] if matches!(strip_casts(a), Expr::Var(y) if *y == p)) => Some(Expr::Var(p)),
                _ => None,
            };
            if let Some(r) = repl {
                *x = r;
            }
        });
        // (the node pointer left anywhere but as the iterator itself)
        e.walk(&mut |x| {
            if let Expr::Load { base, offset, .. } = x {
                if matches!(strip_casts(base), Expr::Var(y) if *y == p) && *offset < item {
                    *ok = false;
                }
            }
        });
    };
    Stmt::rewrite_exprs(&mut new_body, &mut |e| rewrite(e, &mut ok));
    if !ok {
        return None;
    }
    // after the loop: the end local and `*(iterator*)&l.mEnd` are `l.end()`, element reads too
    let end_call = call("end");
    let mut tail: Vec<Stmt> = ir.body[w + 1..].to_vec();
    Stmt::rewrite_exprs(&mut tail, &mut |x| {
        x.rewrite(&mut |y| {
            let is_end = match &*y {
                Expr::Var(e) => end_var.is_some_and(|(v, _)| v == *e),
                Expr::Load { base, offset, ty } => matches!(strip_casts(base), Expr::Var(v) if *v == bv) && *offset == boff + endo && crate::util::class_name(ty, db).is_some(),
                _ => false,
            };
            if is_end {
                *y = end_call.clone();
            }
        });
    });
    Stmt::rewrite_exprs(&mut tail, &mut |e| rewrite(e, &mut ok));
    if !ok {
        return None;
    }
    let mut cond_ok = true;
    // (the body may not read the end local)
    if let Some((e, _)) = end_var {
        cond_ok = !new_body.iter().any(|s| mentions(s, e));
    }
    if !cond_ok {
        return None;
    }
    ir.vars[p].ty = it_ty.clone();
    ir.body.truncate(w + 1);
    ir.body.extend(tail);
    ir.body[w] = Stmt::While { cond: Expr::Binary { op: BinOp::Ne, l: Box::new(Expr::Var(p)), r: Box::new(call("end")), ty: Type::Bool }, body: new_body };
    let mut gone = vec![d];
    if let Some((_, de)) = end_var {
        gone.push(de);
    }
    gone.sort_unstable();
    let mut w = w;
    for &i in gone.iter().rev() {
        ir.body.remove(i);
        w -= 1;
    }
    ir.body.insert(w, Stmt::Assign { dst: Expr::Var(p), src: call("begin") });
    // (returns where the loop ends up)
    Some(sink_param_reads(ir, w + 1))
}

// ---------------------------------------------------------------- rstl::red_black_tree

/// A class that is (or derives at offset 0 from) an `rstl::red_black_tree`: (tree class, offset
/// of its header, offset of a node's value, value type).
fn tree_layout(db: &TypeDb, cls: &str) -> Option<(String, i32, i32, Type)> {
    let c = mwdec_lift::sig::find_class(db, cls)?;
    if cls.split('<').next()?.trim() != "rstl::red_black_tree" {
        let b = c.bases.iter().find(|b| b.offset == 0)?;
        return tree_layout(db, &b.name);
    }
    let header = c.fields.iter().find(|f| f.name == "mHeader")?.offset as i32;
    let value = mwdec_lift::sig::find_class(db, &format!("{cls}::node")).and_then(|n| n.fields.iter().find(|f| f.name == "mValue").map(|f| f.offset as i32)).unwrap_or(16);
    let inner = &cls[cls.find('<')? + 1..cls.rfind('>')?];
    let args = mwdec_lift::sig::split_top(inner, ',');
    let p = mwdec_lift::sig::parse_type(args.get(1)?.trim());
    Some((cls.to_string(), header, value, p))
}

/// A walk over a tree's nodes (`rstl::map`, `rstl::set`) as an iterator loop:
/// `t = &m.mHeader; for (p = t->mLeftmost; p || t != t; p = rbtree_traverse_forward(t, p)) {...}`
/// (the expansion of `it != m.end()` and `++it`) is
/// `for (const_iterator it = m.begin(); it != m.end(); ++it) { ... it->second ... }`.
/// Returns the index of the loop.
fn try_map_loop(ir: &mut IrFunction, db: &TypeDb, w: usize) -> Option<usize> {
    if let Stmt::For { init, cond, step, body } = &ir.body[w] {
        let ([i @ Stmt::Assign { .. }], [s]) = (init.as_slice(), step.as_slice()) else { return None };
        let mut t = ir.clone();
        let (i, c, s, b) = (i.clone(), cond.clone(), s.clone(), body.clone());
        let mut nb = b;
        nb.push(s);
        t.body[w] = Stmt::While { cond: c, body: nb };
        t.body.insert(w, i);
        let r = try_map_loop(&mut t, db, w + 1)?;
        *ir = t;
        return Some(r);
    }
    let Stmt::While { cond, .. } = &ir.body[w] else { return None };
    let Expr::Binary { op: BinOp::LogOr, l, r, .. } = cond else { return None };
    let p = match strip_casts(l) {
        Expr::Var(p) => *p,
        Expr::Binary { op: BinOp::Ne, l: a, r: z, .. } if z.as_int() == Some(0) => match strip_casts(a) {
            Expr::Var(p) => *p,
            _ => return None,
        },
        _ => return None,
    };
    let t = match strip_casts(r) {
        Expr::Binary { op: BinOp::Ne, l: a, r: b, .. } => match (strip_casts(a), strip_casts(b)) {
            (Expr::Var(x), Expr::Var(y)) if x == y => *x,
            _ => return None,
        },
        _ => return None,
    };
    if !matches!(ir.vars[p].kind, VarKind::Local | VarKind::Stack { .. }) || !matches!(ir.vars[t].kind, VarKind::Local) {
        return None;
    }
    // the init of the walk just before the loop, the header address before that
    let Stmt::Assign { dst: Expr::Var(x), src: first } = &ir.body[w.checked_sub(1)?] else { return None };
    if *x != p {
        return None;
    }
    let tdefs: Vec<usize> = (0..w).filter(|&i| matches!(&ir.body[i], Stmt::Assign { dst: Expr::Var(x), .. } if *x == t)).collect();
    let [td] = tdefs.as_slice() else { return None };
    let td = *td;
    let Stmt::Assign { src: hdr, .. } = &ir.body[td] else { return None };
    let (cbase, k) = base_plus(hdr)?;
    // the walk starts at the header's leftmost node: `t->mLeftmost`, or the same word read
    // from the container (`m.mHeader.mLeftmost`)
    let starts = match strip_casts(first) {
        Expr::Load { base: fb, offset: 0, .. } | Expr::Member { base: fb, offset: 0, .. } if matches!(strip_casts(fb), Expr::Var(y) if *y == t) => true,
        Expr::Load { base: fb, offset, .. } | Expr::Member { base: fb, offset, .. } => *offset as i64 == k && strip_casts(fb) == cbase,
        _ => false,
    };
    if !starts {
        return None;
    }
    let cb = cbase.clone();
    map_loop_rewrite(ir, db, w, p, t, td, cb, k)
}

#[allow(clippy::too_many_arguments)]
fn map_loop_rewrite(ir: &mut IrFunction, db: &TypeDb, w: usize, p: VarId, t: VarId, td: usize, cbase: Expr, k: i64) -> Option<usize> {
    // the container: a reference parameter, or a member of this
    let param_obj = |v: VarId| -> Option<(Expr, String, bool)> {
        let cls = crate::util::class_name(strip(&ir.vars[v].ty), db)?;
        let index = match ir.vars[v].kind {
            VarKind::Param { index } => index,
            _ => return None,
        };
        let cst = matches!(&ir.vars[v].ty, Type::Const(_)) || ir.sig.params.get(index).is_some_and(|q| matches!(&q.ty, Type::Ref(r) if matches!(&**r, Type::Const(_))));
        Some((Expr::Var(v), cls, cst))
    };
    let (cont, cls, cst) = match &cbase {
        // a reference parameter's object
        Expr::Var(v) if matches!(ir.vars[*v].kind, VarKind::Param { .. }) && !matches!(strip(&ir.vars[*v].ty), Type::Ptr(_)) => param_obj(*v)?,
        Expr::AddrOf(x) => match &**x {
            Expr::Var(v) if matches!(ir.vars[*v].kind, VarKind::Param { .. }) => {
                let cls = crate::util::class_name(strip(&ir.vars[*v].ty), db)?;
                let cst = matches!(&ir.vars[*v].ty, Type::Const(_)) || ir.sig.params.get(match ir.vars[*v].kind {
                    VarKind::Param { index } => index,
                    _ => usize::MAX,
                }).is_some_and(|q| matches!(&q.ty, Type::Ref(r) if matches!(&**r, Type::Const(_))));
                (Expr::Var(*v), cls, cst)
            }
            _ => return None,
        },
        Expr::Var(v) if Some(*v) == ir.this_var => {
            // `(char*)this + k`: the member holding the header
            let own = ir.sig.this_class.as_deref()?;
            let c = mwdec_lift::sig::find_class(db, own)?;
            let mut hit = None;
            for f in &c.fields {
                let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
                let Some(fc) = crate::util::class_name(&ft, db) else { continue };
                if let Some((_, h, _, _)) = tree_layout(db, &fc) {
                    if f.offset as i64 + h as i64 == k {
                        hit = Some((Expr::Load { base: Box::new(Expr::Var(*v)), offset: f.offset as i32, ty: ft.clone() }, fc, ir.sig.is_const));
                    }
                }
            }
            let (lv, fc, cst) = hit?;
            let (_, h, _, _) = tree_layout(db, &fc)?;
            return map_loop_finish(ir, db, w, p, t, td, lv, fc, cst, h as i64);
        }
        _ => return None,
    };
    let (_, h, _, _) = tree_layout(db, &cls)?;
    if h as i64 != k {
        return None;
    }
    map_loop_finish(ir, db, w, p, t, td, cont, cls, cst, k)
}

#[allow(clippy::too_many_arguments)]
fn map_loop_finish(ir: &mut IrFunction, db: &TypeDb, w: usize, p: VarId, t: VarId, td: usize, cont: Expr, cls: String, cst: bool, _k: i64) -> Option<usize> {
    let (_, _, value, vty) = tree_layout(db, &cls)?;
    let Stmt::While { body, .. } = &ir.body[w] else { return None; };
    // the step last: `p = rbtree_traverse_forward(t, p)`
    let (last, rest) = body.split_last()?;
    let Stmt::Assign { dst: Expr::Var(x), src } = last else { return None; };
    let Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. } = strip_casts(src) else { return None; };
    if *x != p || !symbol.starts_with("rbtree_traverse_forward") || !matches!(args.as_slice(), [a, b] if matches!(strip_casts(a), Expr::Var(y) if *y == t) && matches!(strip_casts(b), Expr::Var(y) if *y == p)) {
        return None;
    }
    // the header local and the walk used nowhere else (but as the element's address)
    if rest.iter().any(|s| mentions(s, t)) || ir.body[w + 1..].iter().any(|s| mentions(s, t) || mentions(s, p)) || (td + 1..w - 1).any(|i| mentions(&ir.body[i], t) || mentions(&ir.body[i], p)) {
        return None;
    }
    let iter_cls = format!("{cls}::{}", if cst { "const_iterator" } else { "iterator" });
    let it_ty = Type::Named(iter_cls.clone());
    let vty = if cst { Type::Const(Box::new(vty)) } else { vty };
    let elem_ptr = Type::Ptr(Box::new(vty));
    let obj = Box::new(Expr::AddrOf(Box::new(cont.clone())));
    let call = |name: &str| Expr::Call { callee: Callee::Method { symbol: String::new(), sig: method(&cls, name, it_ty.clone(), cst), this: obj.clone(), qualified: false }, args: vec![], ret: it_ty.clone() };
    let arrow = Expr::Call {
        callee: Callee::Method { symbol: String::new(), sig: FuncSig { qualified_name: format!("{iter_cls}::operator->"), mangled: None, ret: elem_ptr.clone(), params: vec![], this_class: Some(iter_cls.clone()), is_const: true, is_static: false, is_virtual: false, variadic: false, runs_code: false }, this: Box::new(Expr::AddrOf(Box::new(Expr::Var(p)))), qualified: false },
        args: vec![],
        ret: elem_ptr,
    };
    let mut new_body: Vec<Stmt> = rest.to_vec();
    Stmt::rewrite_exprs(&mut new_body, &mut |e| {
        e.rewrite(&mut |x| {
            if let Expr::Load { base, offset, ty } = x {
                if matches!(strip_casts(base), Expr::Var(y) if *y == p) && *offset >= value {
                    *x = Expr::Load { base: Box::new(arrow.clone()), offset: *offset - value, ty: ty.clone() };
                }
            }
        })
    });
    // the walk left only inside the element reads (`&it` of each `it->`)
    let (mut bare, mut arrows) = (0, 0);
    Stmt::walk_exprs(&new_body, &mut |e| match e {
        Expr::Var(y) if *y == p => bare += 1,
        Expr::Call { callee: Callee::Method { this, .. }, .. } if matches!(&**this, Expr::AddrOf(z) if matches!(&**z, Expr::Var(y) if *y == p)) => arrows += 1,
        _ => {}
    });
    if bare != arrows {
        return None;
    }
    ir.vars[p].ty = it_ty.clone();
    // (a tree iterator has no default constructor: declared by the loop's init)
    ir.body[w] = Stmt::For {
        init: vec![Stmt::Assign { dst: Expr::Var(p), src: call("begin") }],
        cond: Expr::Binary { op: BinOp::Ne, l: Box::new(Expr::Var(p)), r: Box::new(call("end")), ty: Type::Bool },
        step: vec![Stmt::Expr(Expr::IncDec { e: Box::new(Expr::Var(p)), delta: 1, post: false })],
        body: new_body,
    };
    ir.body.remove(w - 1);
    ir.body.remove(td);
    Some(w - 2)
}

/// `p = l.mStart; while (p != l.mEnd && p->mItem != x) p = p->mNext;` is `rstl::find`'s
/// expansion over a list: `it = rstl::find(l.begin(), l.end(), x)`, the node pointer an
/// iterator, `l.end()` for the end after the loop and `l.erase(it)` for `l.do_erase(p)`.
/// Returns the index of the `find` assignment.
fn try_list_find(ir: &mut IrFunction, db: &TypeDb, w: usize) -> Option<usize> {
    // (`for (p = l.mStart; ...; p = p->mNext) {}` the same)
    if let Stmt::For { init, cond, step, body } = &ir.body[w] {
        let ([i @ Stmt::Assign { .. }], [s], []) = (init.as_slice(), step.as_slice(), body.as_slice()) else { return None };
        let mut t = ir.clone();
        let (i, c, s) = (i.clone(), cond.clone(), s.clone());
        t.body[w] = Stmt::While { cond: c, body: vec![s] };
        t.body.insert(w, i);
        let r = try_list_find(&mut t, db, w + 1)?;
        *ir = t;
        return Some(r);
    }
    let Stmt::While { cond, body } = &ir.body[w] else { return None };
    let Expr::Binary { op: BinOp::LogAnd, l: test, r: cmp, .. } = cond else { return None };
    let Expr::Binary { op: BinOp::Ne, l: ta, r: tb, .. } = strip_casts(test) else { return None };
    let (p, end) = match (strip_casts(ta), strip_casts(tb)) {
        (Expr::Var(p), e) => (*p, e.clone()),
        _ => return None,
    };
    if !matches!(ir.vars[p].kind, VarKind::Local | VarKind::Stack { .. }) {
        return None;
    }
    let defs: Vec<usize> = (0..w).filter(|&i| matches!(&ir.body[i], Stmt::Assign { dst: Expr::Var(x), .. } if *x == p)).collect();
    let [d] = defs.as_slice() else { return None };
    let d = *d;
    let Stmt::Assign { src, .. } = &ir.body[d] else { return None };
    let (cont, cls, (_start, endo, next, item), cst) = list_of(ir, db, src)?;
    let (bv, boff) = lv_loc(&cont)?;
    let is_end_load = |e: &Expr| matches!(strip_casts(e), Expr::Load { base, offset, .. } if matches!(strip_casts(base), Expr::Var(v) if *v == bv) && *offset == boff + endo);
    let end_var = match &end {
        Expr::Var(e) => {
            let ds: Vec<usize> = (0..w).filter(|&i| matches!(&ir.body[i], Stmt::Assign { dst: Expr::Var(x), .. } if x == e)).collect();
            let [de] = ds.as_slice() else { return None };
            let Stmt::Assign { src, .. } = &ir.body[*de] else { return None };
            if !is_end_load(src) {
                return None;
            }
            Some((*e, *de))
        }
        e if is_end_load(e) => None,
        _ => return None,
    };
    // the body: only the step
    let [Stmt::Assign { dst: Expr::Var(x), src: step }] = body.as_slice() else { return None };
    let at_p = |e: &Expr, k: i32| matches!(strip_casts(e), Expr::Load { base, offset, .. } if matches!(strip_casts(base), Expr::Var(y) if *y == p) && *offset == k);
    if *x != p || !at_p(step, next) {
        return None;
    }
    // the item against a value free of the walk
    let Expr::Binary { op: BinOp::Ne, l: a, r: b, .. } = strip_casts(cmp) else { return None };
    let val = if at_p(a, item) {
        b
    } else if at_p(b, item) {
        a
    } else {
        return None;
    };
    let val = strip_casts(val).clone();
    if val.uses_var(p) || val.has_call() || end_var.is_some_and(|(e, _)| val.uses_var(e)) {
        return None;
    }
    // nothing else between uses the walk or the end
    if (d + 1..w).any(|i| Some(i) != end_var.map(|x| x.1) && (mentions(&ir.body[i], p) || end_var.is_some_and(|(e, _)| mentions(&ir.body[i], e)))) {
        return None;
    }
    let iter_cls = format!("{cls}::{}", if cst { "const_iterator" } else { "iterator" });
    let it_ty = Type::Named(iter_cls);
    let obj = Box::new(Expr::AddrOf(Box::new(cont.clone())));
    let call = |name: &str, args: Vec<Expr>| Expr::Call { callee: Callee::Method { symbol: String::new(), sig: method(&cls, name, it_ty.clone(), cst), this: obj.clone(), qualified: false }, args, ret: it_ty.clone() };
    // after the loop: copies of the walk are the walk, the end `l.end()`, comparisons without
    // the pointer casts, `do_erase(p)` `erase(it)`; any other use and the rewrite is off
    let mut tail: Vec<Stmt> = ir.body[w + 1..].to_vec();
    let mut k = 0;
    while k < tail.len() {
        if let Stmt::Assign { dst: Expr::Var(t), src } = &tail[k] {
            let t = *t;
            if matches!(strip_casts(src), Expr::Var(y) if *y == p) && matches!(ir.vars[t].kind, VarKind::Local) && !ir.body[..=w].iter().any(|s| mentions(s, t)) {
                let others = tail.iter().enumerate().any(|(i, s)| i != k && matches!(s, Stmt::Assign { dst: Expr::Var(y), .. } if *y == t || *y == p));
                if !others {
                    tail.remove(k);
                    Stmt::rewrite_exprs(&mut tail, &mut |e| {
                        e.rewrite(&mut |y| {
                            if matches!(y, Expr::Var(v) if *v == t) {
                                *y = Expr::Var(p);
                            }
                        })
                    });
                    continue;
                }
            }
        }
        k += 1;
    }
    let is_p = |e: &Expr| matches!(strip_casts(e), Expr::Var(y) if *y == p);
    let end_call = call("end", vec![]);
    let mut ok = true;
    Stmt::rewrite_exprs(&mut tail, &mut |e| {
        e.rewrite(&mut |y| {
            let is_end = match &*y {
                Expr::Var(v) => end_var.is_some_and(|(ev, _)| ev == *v),
                Expr::Load { base, offset, ty } => matches!(strip_casts(base), Expr::Var(v) if *v == bv) && *offset == boff + endo && crate::util::class_name(ty, db).is_some(),
                _ => false,
            };
            if is_end {
                *y = end_call.clone();
            }
        });
        e.rewrite(&mut |y| match y {
            Expr::Binary { op: BinOp::Ne | BinOp::Eq, l, r, .. } if is_p(l) || is_p(r) => {
                let (a, b) = (strip_casts(l).clone(), strip_casts(r).clone());
                **l = a;
                **r = b;
            }
            Expr::Call { callee: Callee::Method { sig, this, .. }, args, .. } if sig.qualified_name.ends_with("::do_erase") && matches!(args.as_slice(), [a] if is_p(a)) => {
                if let Expr::AddrOf(l) = &**this {
                    if lv_loc(l) == Some((bv, boff)) {
                        *y = Expr::Call { callee: Callee::Method { symbol: String::new(), sig: method(&cls, "erase", it_ty.clone(), false), this: this.clone(), qualified: false }, args: vec![Expr::Var(p)], ret: it_ty.clone() };
                    }
                }
            }
            _ => {}
        });
    });
    // (the walk left only as the iterator itself)
    Stmt::walk_exprs(&tail, &mut |y| match y {
        Expr::Cast { e, .. } if matches!(&**e, Expr::Var(v) if *v == p) => ok = false,
        Expr::Load { base, .. } if is_p(base) => ok = false,
        _ => {}
    });
    if !ok || end_var.is_some_and(|(e, _)| tail.iter().any(|s| mentions(s, e))) {
        return None;
    }
    let find = FuncSig { qualified_name: "rstl::find".into(), mangled: None, ret: it_ty.clone(), params: vec![], this_class: None, is_const: false, is_static: false, is_virtual: false, variadic: false, runs_code: false };
    let found = Expr::Call { callee: Callee::Direct { symbol: "rstl::find".into(), sig: find }, args: vec![call("begin", vec![]), end_call, val], ret: it_ty.clone() };
    ir.vars[p].ty = it_ty;
    ir.body.truncate(w + 1);
    ir.body.extend(tail);
    ir.body[w] = Stmt::Assign { dst: Expr::Var(p), src: found };
    let mut gone = vec![d];
    if let Some((_, de)) = end_var {
        gone.push(de);
    }
    gone.sort_unstable();
    let mut w = w;
    for &i in gone.iter().rev() {
        ir.body.remove(i);
        w -= 1;
    }
    Some(w)
}

/// Locals set before the loop at `w` from a by-value parameter's member (`t = id.value`) and used
/// only in the loop: read where used, as the source's `it->m == id` does.
fn sink_param_reads(ir: &mut IrFunction, w: usize) -> usize {
    let mut i = 0;
    let mut w = w;
    while i < w {
        let Stmt::Assign { dst: Expr::Var(t), src } = &ir.body[i] else {
            i += 1;
            continue;
        };
        let (t, src) = (*t, src.clone());
        let param_read = match &src {
            Expr::Member { base, .. } => matches!(&**base, Expr::Var(v) if matches!(ir.vars[*v].kind, VarKind::Param { .. })),
            _ => false,
        };
        let only_loop = ir.vars[t].kind == VarKind::Local && ir.body.iter().enumerate().all(|(k, s)| k == i || k == w || !mentions(s, t));
        let single = {
            let mut n = 0;
            Stmt::walk_exprs(&ir.body, &mut |e| n += matches!(e, Expr::Var(x) if *x == t) as usize);
            n >= 2
        };
        if !(param_read && only_loop && single) {
            i += 1;
            continue;
        }
        ir.body.remove(i);
        w -= 1;
        Stmt::rewrite_exprs(std::slice::from_mut(&mut ir.body[w]), &mut |e| {
            if matches!(e, Expr::Var(x) if *x == t) {
                *e = src.clone();
            }
        });
    }
    w
}

// ---------------------------------------------------------------- index loops over vectors

/// `rstl::vector<T, A>`: offsets of `mCount` and `mItems`, and the element size.
fn vector_layout(db: &TypeDb, cls: &str) -> Option<(i32, i32, Type)> {
    if cls.split('<').next()?.trim() != "rstl::vector" {
        return None;
    }
    let c = mwdec_lift::sig::find_class(db, cls)?;
    let off = |n: &str| c.fields.iter().find(|f| f.name == n).map(|f| f.offset as i32);
    let inner = &cls[cls.find('<')? + 1..cls.rfind('>')?];
    let t = mwdec_lift::sig::parse_type(mwdec_lift::sig::split_top(inner, ',').first()?.trim());
    Some((off("mCount")?, off("mItems")?, t))
}

/// `(this, member offset, class)` of the vector member whose field at `k` (from `this`) is read.
fn vector_member_at(ir: &IrFunction, db: &TypeDb, e: &Expr, field: fn(&(i32, i32, Type)) -> i32) -> Option<(Expr, String, (i32, i32, Type))> {
    let Expr::Load { base, offset, .. } = strip_casts(e) else { return None };
    let Expr::Var(v) = strip_casts(base) else { return None };
    if Some(*v) != ir.this_var {
        return None;
    }
    let own = ir.sig.this_class.as_deref()?;
    let c = mwdec_lift::sig::find_class(db, own)?;
    for f in &c.fields {
        let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
        let Some(cls) = crate::util::class_name(&ft, db) else { continue };
        let Some(lay) = vector_layout(db, &cls) else { continue };
        if f.offset as i32 + field(&lay) == *offset {
            return Some((Expr::Load { base: Box::new(Expr::Var(*v)), offset: f.offset as i32, ty: ft.clone() }, cls, lay));
        }
    }
    None
}

fn def_of(ir: &IrFunction, upto: usize, v: VarId) -> Option<(usize, Expr)> {
    let ds: Vec<usize> = (0..upto).filter(|&i| matches!(&ir.body[i], Stmt::Assign { dst: Expr::Var(x), .. } if *x == v)).collect();
    let [d] = ds.as_slice() else { return None };
    let Stmt::Assign { src, .. } = &ir.body[*d] else { return None };
    Some((*d, src.clone()))
}

/// An index loop over a vector member with the element address strength-reduced to a byte
/// offset (`o += sizeof(T)` beside `++i`): `v[i]`, `i < v.size()`, the offset gone.
fn try_index_loop(ir: &mut IrFunction, db: &TypeDb, w: usize) -> Option<()> {
    let Stmt::While { cond, body } = &ir.body[w] else { return None };
    let Expr::Binary { op: BinOp::Lt, l, r, .. } = cond else { return None };
    let Expr::Var(i) = strip_casts(l) else { return None };
    let i = *i;
    // the bound: the vector's count (directly or through a local)
    let (n_def, bound) = match strip_casts(r) {
        Expr::Var(n) => {
            let (d, src) = def_of(ir, w, *n)?;
            (Some((*n, d)), src)
        }
        e => (None, e.clone()),
    };
    let (cont, cls, (_, _, t)) = vector_member_at(ir, db, &bound, |l| l.0)?;
    let size = mwdec_lift::types::size_of(Some(db), &t)? as i64;
    // the steps at the end of the body: `o += size` and `i += 1` (either order)
    let step_of = |s: &Stmt| match s {
        Stmt::Assign { dst: Expr::Var(x), src } => match strip_casts(src) {
            Expr::Binary { op: BinOp::Add, l, r, .. } if matches!(strip_casts(l), Expr::Var(y) if y == x) => Some((*x, r.as_int()?)),
            _ => None,
        },
        _ => None,
    };
    let k = body.len();
    if k < 2 {
        return None;
    }
    let (s1, s2) = (step_of(&body[k - 2])?, step_of(&body[k - 1])?);
    let (o, so) = if s1.0 == i { s2 } else { s1 };
    if (s1.0 != i && s2.0 != i) || so != size || o == i {
        return None;
    }
    if [s1, s2].iter().find(|s| s.0 == i)?.1 != 1 {
        return None;
    }
    // i and o start at 0; the data pointer is the vector's
    let (di, si) = def_of(ir, w, i)?;
    let (dof, so0) = def_of(ir, w, o)?;
    if si.as_int() != Some(0) || so0.as_int() != Some(0) {
        return None;
    }
    // the element pointer `(char*)base + o` / `&((char*)base)[o]` with base the data pointer
    let mut base_var = None;
    for s in &body[..k - 2] {
        Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
            if let Expr::Index { base, index, .. } = e {
                if matches!(strip_casts(index), Expr::Var(x) if *x == o) {
                    if let Expr::Var(b) = strip_casts(base) {
                        base_var = Some(*b);
                    }
                }
            }
        });
    }
    let b = base_var?;
    let (db_, bsrc) = def_of(ir, w, b)?;
    let (bcont, _, _) = vector_member_at(ir, db, &bsrc, |l| l.1)?;
    if bcont != cont {
        return None;
    }
    // nothing else uses o or the data pointer
    let mut body2: Vec<Stmt> = body[..k - 2].to_vec();
    let istep = if s1.0 == i { body[k - 2].clone() } else { body[k - 1].clone() };
    let obj = Box::new(Expr::AddrOf(Box::new(cont.clone())));
    let elem = Expr::Call {
        callee: Callee::Method { symbol: String::new(), sig: FuncSig { qualified_name: format!("{cls}::operator[]"), mangled: None, ret: Type::Ref(Box::new(t.clone())), params: vec![mwdec_core::Param { name: None, ty: Type::Int { size: 4, signed: true } }], this_class: Some(cls.clone()), is_const: ir.sig.is_const, is_static: false, is_virtual: false, variadic: false, runs_code: false }, this: obj.clone(), qualified: false },
        args: vec![Expr::Var(i)],
        ret: Type::Ref(Box::new(t.clone())),
    };
    let mut ok = true;
    Stmt::rewrite_exprs(&mut body2, &mut |e| {
        e.rewrite(&mut |x| {
            let hit = matches!(x, Expr::AddrOf(y) if matches!(&**y, Expr::Index { base, index, .. } if matches!(strip_casts(base), Expr::Var(v) if *v == b) && matches!(strip_casts(index), Expr::Var(v) if *v == o)));
            if hit {
                *x = Expr::AddrOf(Box::new(elem.clone()));
            }
        });
    });
    Stmt::walk_exprs(&body2, &mut |e| ok &= !matches!(e, Expr::Var(v) if *v == o || *v == b));
    if !ok {
        return None;
    }
    // (o, the data pointer and the bound local used nowhere else)
    let others = |v: VarId, skip: &[usize]| ir.body.iter().enumerate().any(|(j, s)| j != w && !skip.contains(&j) && mentions(s, v));
    if others(o, &[dof]) || others(b, &[db_]) || n_def.is_some_and(|(n, d)| others(n, &[d])) {
        return None;
    }
    body2.push(istep);
    let size_call = Expr::Call { callee: Callee::Method { symbol: String::new(), sig: method(&cls, "size", Type::Int { size: 4, signed: true }, true), this: obj, qualified: false }, args: vec![], ret: Type::Int { size: 4, signed: true } };
    ir.body[w] = Stmt::While { cond: Expr::Binary { op: BinOp::Lt, l: Box::new(Expr::Var(i)), r: Box::new(size_call), ty: Type::Bool }, body: body2 };
    // the counter starts right before the loop; the offset, data pointer and bound go
    let mut gone = vec![dof, db_, di];
    if let Some((_, d)) = n_def {
        gone.push(d);
    }
    gone.sort_unstable();
    gone.dedup();
    let mut w = w;
    for &j in gone.iter().rev() {
        ir.body.remove(j);
        w -= 1;
    }
    ir.body.insert(w, Stmt::Assign { dst: Expr::Var(i), src: Expr::int(0) });
    Some(())
}
