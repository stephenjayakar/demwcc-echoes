//! Late local variable typing: register locals are typed from their first definition while
//! lifting, before later passes (array recovery, struct copies) give the assigned values their
//! real types. A local still untyped (`Unknown`, declared `int`) whose every assignment now has
//! the same pointer type, and which is only used where a pointer is valid, takes that type.

use crate::ir::*;
use crate::types;
use mwdec_core::Type;
use std::collections::HashMap;

fn key(t: &Type) -> String {
    crate::sig::norm_name(&format!("{:?}", strip_cv(t)))
}

/// Is every occurrence of `Var(v)` in `e` in a pointer-safe position (not an arithmetic operand
/// or an index)?
fn uses_ok(e: &Expr, v: VarId) -> bool {
    let is_v = |x: &Expr| matches!(x, Expr::Var(y) if *y == v);
    let mut ok = true;
    e.walk(&mut |x| match x {
        Expr::Binary { op, l, r, .. } if !op.is_bool() => {
            if is_v(l) || is_v(r) {
                ok = false;
            }
        }
        Expr::Unary { op: UnOp::Neg | UnOp::BitNot, e, .. } if is_v(e) => ok = false,
        Expr::Index { base, index, .. } if is_v(base) || is_v(index) => ok = false,
        _ => {}
    });
    ok
}

/// Is the object an lvalue designates const (reached through a pointer/reference to const)?
fn lvalue_is_const(e: &Expr, vars: &[Var]) -> bool {
    match e {
        Expr::Load { base, .. } => {
            let bt = types::ty_of(base, vars);
            matches!(pointee(&bt), Some(Type::Const(_))) || matches!(&**base, Expr::AddrOf(inner) if lvalue_is_const(inner, vars))
        }
        Expr::Member { base, .. } | Expr::Index { base, .. } => matches!(types::ty_of(base, vars), Type::Const(_)) || lvalue_is_const(base, vars),
        Expr::Var(v) => matches!(vars[*v].ty, Type::Const(_)),
        Expr::Global { ty, .. } => matches!(ty, Type::Const(_)),
        _ => false,
    }
}

pub fn retype(body: &[Stmt], vars: &mut [Var]) {
    let mut srcs: HashMap<VarId, Vec<Type>> = HashMap::new();
    let mut bad: Vec<VarId> = vec![];
    fn visit(b: &[Stmt], vars: &[Var], srcs: &mut HashMap<VarId, Vec<Type>>) {
        for s in b {
            if let Stmt::Assign { dst: Expr::Var(v), src } = s {
                if !matches!(src, Expr::Int { .. }) {
                    let mut t = types::ty_of(src, vars);
                    // the address of a member of a const object points to const
                    if let (Expr::AddrOf(lv), Type::Ptr(p)) = (src, &t) {
                        if lvalue_is_const(lv, vars) && !matches!(**p, Type::Const(_)) {
                            t = t_ptr(Type::Const(p.clone()));
                        }
                    }
                    srcs.entry(*v).or_default().push(t);
                }
            }
            match s {
                Stmt::If { then, els, .. } => {
                    visit(then, vars, srcs);
                    visit(els, vars, srcs);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => visit(body, vars, srcs),
                Stmt::For { init, step, body, .. } => {
                    visit(init, vars, srcs);
                    visit(step, vars, srcs);
                    visit(body, vars, srcs);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        visit(&c.body, vars, srcs);
                    }
                }
                _ => {}
            }
        }
    }
    visit(body, vars, &mut srcs);
    let cands: Vec<VarId> = srcs
        .iter()
        .filter(|(v, ts)| {
            matches!(vars[**v].kind, VarKind::Local)
                // untyped, or the byte pointer of raw address arithmetic
                && (matches!(vars[**v].ty, Type::Unknown { size: 4 }) || vars[**v].ty == t_ptr(t_int(1, false)) || matches!(pointee(&vars[**v].ty).map(strip_cv), Some(Type::Void)))
                && !ts.is_empty()
                && ts.iter().all(|t| matches!(strip_cv(t), Type::Ptr(_)) && key(t) == key(&ts[0]))
                && key(&ts[0]) != key(&vars[**v].ty)
        })
        .map(|(v, _)| *v)
        .collect();
    if cands.is_empty() {
        return;
    }
    fn roots<'a>(b: &'a [Stmt], out: &mut Vec<&'a Expr>) {
        for s in b {
            match s {
                Stmt::Expr(e) | Stmt::Return(Some(e)) => out.push(e),
                Stmt::Assign { dst, src } => {
                    out.push(dst);
                    out.push(src);
                }
                Stmt::If { cond, then, els } => {
                    out.push(cond);
                    roots(then, out);
                    roots(els, out);
                }
                Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                    out.push(cond);
                    roots(body, out);
                }
                Stmt::For { init, cond, step, body } => {
                    out.push(cond);
                    roots(init, out);
                    roots(step, out);
                    roots(body, out);
                }
                Stmt::Switch { e, cases } => {
                    out.push(e);
                    for c in cases {
                        roots(&c.body, out);
                    }
                }
                _ => {}
            }
        }
    }
    let mut rs = vec![];
    roots(body, &mut rs);
    for e in rs {
        for &v in &cands {
            if !bad.contains(&v) && !uses_ok(e, v) {
                bad.push(v);
            }
        }
    }
    for v in cands {
        if !bad.contains(&v) {
            let mut t = srcs[&v][0].clone();
            // element addresses only read through: pointer to const (the element may be reached
            // through a const accessor in the source spelling)
            if let Type::Ptr(p) = &t {
                if !matches!(**p, Type::Const(_)) && types::is_aggregate(None, p) && read_only(body, v) {
                    t = t_ptr(Type::Const(p.clone()));
                }
            }
            vars[v].ty = t;
        }
    }
}

/// Is pointer local `v` only dereferenced for reading (never stored through, passed on, returned
/// or used as a receiver)?
fn read_only(body: &[Stmt], v: VarId) -> bool {
    let mut ok = true;
    fn check(e: &Expr, v: VarId, ok: &mut bool) {
        e.walk(&mut |x| match x {
            Expr::Call { callee, args, .. } => {
                if args.iter().any(|a| a.uses_var(v)) {
                    *ok = false;
                }
                if let Callee::Method { this, .. } | Callee::Virtual { this, .. } = callee {
                    if this.uses_var(v) {
                        *ok = false;
                    }
                }
            }
            Expr::New { .. } | Expr::Construct { .. } | Expr::IncDec { .. } | Expr::AddrOf(_) if x.uses_var(v) => *ok = false,
            _ => {}
        });
    }
    fn visit(b: &[Stmt], v: VarId, ok: &mut bool) {
        for s in b {
            match s {
                Stmt::Assign { dst, src } => {
                    if !matches!(dst, Expr::Var(_)) && dst.uses_var(v) {
                        *ok = false;
                    }
                    if matches!(dst, Expr::Var(w) if *w != v) && matches!(src, Expr::Var(w) if *w == v) {
                        *ok = false;
                    }
                    check(src, v, ok);
                }
                Stmt::Expr(e) => check(e, v, ok),
                Stmt::Return(Some(e)) => {
                    if e.uses_var(v) {
                        *ok = false;
                    }
                }
                Stmt::If { cond, then, els } => {
                    check(cond, v, ok);
                    visit(then, v, ok);
                    visit(els, v, ok);
                }
                Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                    check(cond, v, ok);
                    visit(body, v, ok);
                }
                Stmt::For { init, cond, step, body } => {
                    visit(init, v, ok);
                    check(cond, v, ok);
                    visit(step, v, ok);
                    visit(body, v, ok);
                }
                Stmt::Switch { e, cases } => {
                    check(e, v, ok);
                    for c in cases {
                        visit(&c.body, v, ok);
                    }
                }
                _ => {}
            }
        }
    }
    visit(body, v, &mut ok);
    ok
}

/// `(v << s) & (0xff << s)` / `v & 0xff` (and the 0xffff forms): a masked read of local `v` as
/// an unsigned byte/halfword -> (v, type, shift).
fn masked_read(e: &Expr) -> Option<(VarId, Type, Option<i64>)> {
    let Expr::Binary { op: BinOp::And, l, r, .. } = e else { return None };
    let m = r.as_int()? as u32;
    let (v, s) = match &**l {
        Expr::Var(v) => (*v, None),
        Expr::Binary { op: BinOp::Shl, l: x, r: s, .. } => match (&**x, s.as_int()) {
            (Expr::Var(v), Some(s)) if (1..24).contains(&s) => (*v, Some(s)),
            _ => return None,
        },
        _ => return None,
    };
    let sh = s.unwrap_or(0) as u32;
    let t = if m == 0xff << sh {
        Type::Int { size: 1, signed: false }
    } else if m == 0xffff << sh {
        Type::Int { size: 2, signed: false }
    } else {
        return None;
    };
    Some((v, t, s))
}

fn narrow_ty(t: &Type) -> Option<Type> {
    match strip_cv(t) {
        Type::Int { size: 1 | 2, .. } | Type::Char | Type::Bool => Some(strip_cv(t).clone()),
        _ => None,
    }
}

fn all_stmt_lists<'a>(b: &'a [Stmt], out: &mut Vec<&'a Stmt>) {
    for s in b {
        out.push(s);
        match s {
            Stmt::If { then, els, .. } => {
                all_stmt_lists(then, out);
                all_stmt_lists(els, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => all_stmt_lists(body, out),
            Stmt::For { init, step, body, .. } => {
                all_stmt_lists(init, out);
                all_stmt_lists(step, out);
                all_stmt_lists(body, out);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    all_stmt_lists(&c.body, out);
                }
            }
            _ => {}
        }
    }
}

/// Narrow register locals: an `int` local every read of which is narrowed to the same small type
/// (`(u8)i`, `(u16)i`; the compiler's `clrlwi`/`extsb` on each use of a `u8`/`short` variable) and
/// every write of which is an integer, takes that type and loses the casts. Locals assigned only
/// booleans/0/1 and read as `(u8)x` become `bool`.
pub fn narrow(body: &mut Vec<Stmt>, vars: &mut [Var], db: Option<&mwdec_core::TypeDb>) {
    let n = vars.len();
    // per var: Some(type) while all narrowing reads agree, plus "bad" when a plain read exists
    let mut cast_ty: Vec<Option<Type>> = vec![None; n];
    let mut bad = vec![false; n];
    let mut reads = vec![0usize; n];
    let mut cast_reads = vec![0usize; n];
    let is_cand = |v: VarId, vars: &[Var]| matches!(vars[v].kind, VarKind::Local) && matches!(strip_cv(&vars[v].ty), Type::Unknown { size: 4 } | Type::Int { size: 4, .. });
    let mut stmts = vec![];
    all_stmt_lists(body, &mut stmts);
    // defs
    let mut def_bool = vec![true; n];
    let mut has_def = vec![false; n];
    for s in &stmts {
        if let Stmt::Assign { dst: Expr::Var(v), src } = s {
            let v = *v;
            has_def[v] = true;
            let st = types::resolve(db, &types::ty_of(src, vars)).into_owned();
            let int_like = matches!(strip_cv(&st), Type::Int { .. } | Type::Long { .. } | Type::Char | Type::Bool | Type::Unknown { size: 1 | 2 | 4 }) || types::is_enum(db, &st);
            if !int_like {
                bad[v] = true;
            }
            let b = matches!(src, Expr::Int { value: 0 | 1, .. }) || matches!(strip_cv(&st), Type::Bool) || matches!(src, Expr::Binary { op, .. } if op.is_bool()) || matches!(src, Expr::Unary { op: UnOp::Not, .. });
            def_bool[v] &= b;
        }
    }
    // reads: walk every expression; casts of a var count as narrowing reads
    let mut roots: Vec<&Expr> = vec![];
    for s in &stmts {
        match s {
            Stmt::Expr(e) | Stmt::Return(Some(e)) => roots.push(e),
            Stmt::Assign { dst, src } => {
                if !matches!(dst, Expr::Var(_)) {
                    roots.push(dst);
                }
                roots.push(src);
            }
            Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } | Stmt::For { cond, .. } => roots.push(cond),
            Stmt::Switch { e, .. } => roots.push(e),
            _ => {}
        }
    }
    for e in &roots {
        e.walk(&mut |x| match x {
            Expr::Var(v) => reads[*v] += 1,
            Expr::Binary { .. } if masked_read(x).is_some() => {
                let (v, t, _) = masked_read(x).unwrap();
                cast_reads[v] += 1;
                match &cast_ty[v] {
                    None => cast_ty[v] = Some(t),
                    Some(c) if key(c) == key(&t) => {}
                    Some(_) => bad[v] = true,
                }
            }
            Expr::Cast { ty, e } => {
                if let Expr::Var(v) = &**e {
                    cast_reads[*v] += 1;
                    match narrow_ty(&types::resolve(db, ty)) {
                        Some(t) => match &cast_ty[*v] {
                            None => cast_ty[*v] = Some(t),
                            Some(c) if key(c) == key(&t) => {}
                            Some(_) => bad[*v] = true,
                        },
                        None => bad[*v] = true,
                    }
                }
            }
            _ => {}
        });
    }
    // a var's own increment (`i = i + 1`) is a plain read that's fine to keep
    let mut self_reads = vec![0usize; n];
    for s in &stmts {
        if let Stmt::Assign { dst: Expr::Var(v), src } = s {
            if let Expr::Binary { op: BinOp::Add | BinOp::Sub, l, r, .. } = src {
                if matches!(&**l, Expr::Var(w) if w == v) && r.as_int().is_some() {
                    self_reads[*v] += 1;
                }
            }
        }
    }
    let mut chosen: HashMap<VarId, Type> = HashMap::new();
    for v in 0..n {
        if !is_cand(v, vars) || bad[v] || !has_def[v] || cast_reads[v] == 0 {
            continue;
        }
        if reads[v] != cast_reads[v] + self_reads[v] {
            continue;
        }
        let Some(t) = cast_ty[v].clone() else { continue };
        let t = if def_bool[v] && self_reads[v] == 0 && matches!(t, Type::Int { size: 1, signed: false } | Type::Bool) { Type::Bool } else { t };
        chosen.insert(v, t);
    }
    if chosen.is_empty() {
        return;
    }
    for (v, t) in &chosen {
        vars[*v].ty = t.clone();
    }
    Stmt::rewrite_exprs(body, &mut |x| {
        if let Some((v, _, s)) = masked_read(x) {
            if chosen.contains_key(&v) {
                *x = match s {
                    None => Expr::Var(v),
                    Some(s) => Expr::bin(BinOp::Shl, Expr::Var(v), Expr::int(s), t_s32()),
                };
                return;
            }
        }
        if let Expr::Cast { e, .. } = x {
            if let Expr::Var(v) = &**e {
                if chosen.contains_key(v) {
                    *x = Expr::Var(*v);
                }
            }
        }
    });
}

/// Types of globals the context doesn't declare (statics of the unit): the lifter types each
/// access by its width (`unsigned char` for `lbz`). When every typed context the global's value
/// flows into or comes from agrees on one scalar type of that width (returned from a `bool`
/// function, passed as a `bool`/pointer parameter, assigned to/from a typed lvalue), every
/// access takes that type, so the extern declaration matches (`bool`, `CFoo*`).
pub fn global_types(body: &mut Vec<Stmt>, vars: &[Var], ret: &Type, db: Option<&mwdec_core::TypeDb>) {
    let undeclared = |s: &str| db.map_or(true, |db| !db.globals.contains_key(s));
    let mut expect: HashMap<String, Vec<Type>> = HashMap::new();
    let mut bad: Vec<String> = vec![];
    let gsym = |e: &Expr| -> Option<String> {
        match e {
            Expr::Global { symbol, ty } if scalar_size(ty).map_or(false, |s| s > 0) && undeclared(symbol) => Some(symbol.clone()),
            _ => None,
        }
    };
    let mut stmts = vec![];
    all_stmt_lists(body, &mut stmts);
    let mut roots: Vec<&Expr> = vec![];
    for s in &stmts {
        match s {
            Stmt::Return(Some(e)) => {
                if let Some(g) = gsym(e) {
                    expect.entry(g).or_default().push(ret.clone());
                }
                roots.push(e);
            }
            Stmt::Assign { dst, src } => {
                if let Some(g) = gsym(src) {
                    expect.entry(g).or_default().push(types::ty_of(dst, vars));
                }
                if let Some(g) = gsym(dst) {
                    if !matches!(src, Expr::Int { .. }) {
                        expect.entry(g).or_default().push(types::ty_of(src, vars));
                    }
                }
                roots.push(dst);
                roots.push(src);
            }
            Stmt::Expr(e) => roots.push(e),
            Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } | Stmt::For { cond, .. } => roots.push(cond),
            Stmt::Switch { e, .. } => roots.push(e),
            _ => {}
        }
    }
    for e in &roots {
        e.walk(&mut |x| match x {
            Expr::Call { callee, args, .. } => {
                let sig = match callee {
                    Callee::Direct { sig, .. } | Callee::Method { sig, .. } => Some(sig),
                    Callee::Virtual { sig, .. } => sig.as_ref(),
                    Callee::Indirect(_) => None,
                };
                for (n, a) in args.iter().enumerate() {
                    if let Some(g) = gsym(a) {
                        match sig.and_then(|s| s.params.get(n)) {
                            Some(p) => expect.entry(g).or_default().push(p.ty.clone()),
                            None => bad.push(g),
                        }
                    }
                }
            }
            Expr::Binary { op, l, r, .. } if !op.is_bool() => {
                for o in [l, r] {
                    if let Some(g) = gsym(o) {
                        bad.push(g);
                    }
                }
            }
            Expr::Unary { op: UnOp::Neg | UnOp::BitNot, e, .. } => {
                if let Some(g) = gsym(e) {
                    bad.push(g);
                }
            }
            Expr::Cast { e, .. } | Expr::AddrOf(e) => {
                if let Some(g) = gsym(e) {
                    bad.push(g);
                }
            }
            _ => {}
        });
    }
    let mut chosen: HashMap<String, Type> = HashMap::new();
    for (g, ts) in &expect {
        if bad.contains(g) {
            continue;
        }
        let ts: Vec<Type> = ts.iter().map(|t| strip_cv(&types::resolve(db, t)).clone()).filter(|t| !matches!(t, Type::Unknown { .. } | Type::Void)).collect();
        if ts.is_empty() || !ts.iter().all(|t| key(t) == key(&ts[0])) {
            continue;
        }
        let t = &ts[0];
        if types::is_aggregate(db, t) || matches!(t, Type::Ref(_)) {
            continue;
        }
        chosen.insert(g.clone(), t.clone());
    }
    if chosen.is_empty() {
        return;
    }
    // all accesses must have the chosen type's width
    let mut widths_ok: HashMap<String, bool> = HashMap::new();
    Stmt::walk_exprs(body, &mut |x| {
        if let Expr::Global { symbol, ty } = x {
            if let Some(t) = chosen.get(symbol) {
                let ok = scalar_size(ty) == scalar_size(t) || matches!(ty, Type::Unknown { size: 0 });
                *widths_ok.entry(symbol.clone()).or_insert(true) &= ok;
            }
        }
    });
    Stmt::rewrite_exprs(body, &mut |x| {
        if let Expr::Global { symbol, ty } = x {
            if let Some(t) = chosen.get(symbol.as_str()) {
                if widths_ok.get(symbol.as_str()).copied().unwrap_or(false) && !matches!(ty, Type::Unknown { size: 0 }) {
                    *ty = t.clone();
                }
            }
        }
    });
}

/// `x & 0xff` where `x` is already an unsigned byte (`(u8)f`, a `u8` local), likewise 0xffff for
/// unsigned halfwords: the mask is the compiler's `rlwimi`/`clrlwi` of the narrow value.
pub fn drop_redundant_masks(body: &mut Vec<Stmt>, vars: &[Var], db: Option<&mwdec_core::TypeDb>) {
    Stmt::rewrite_exprs(body, &mut |x| {
        let Expr::Binary { op: BinOp::And, l, r, .. } = x else { return };
        let Some(m) = r.as_int() else { return };
        let lt = types::resolve(db, &types::ty_of(l, vars)).into_owned();
        let full = match strip_cv(&lt) {
            Type::Int { size: 1, signed: false } | Type::Bool => 0xff,
            Type::Int { size: 2, signed: false } => 0xffff,
            _ => return,
        };
        if m as u32 == full {
            *x = (**l).clone();
        }
    });
}

/// Integer intrinsics (`__rlwimi`) take their operands as integers: pointers are converted.
pub fn cast_intrinsic_args(body: &mut Vec<Stmt>, vars: &[Var]) {
    Stmt::rewrite_exprs(body, &mut |x| {
        let Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. } = x else { return };
        if symbol != "__rlwimi" {
            return;
        }
        for a in args.iter_mut().take(2) {
            let t = types::ty_of(a, vars);
            if is_ptr(&t) || matches!(strip_cv(&t), Type::Ref(_) | Type::Unknown { .. }) {
                *a = Expr::cast(t_u32(), a.clone());
            }
        }
    });
}

/// A pointer global copied into a temp read through until the next call is the global itself at
/// each use (`__GXData->x = ..; __GXData->y = ..;`): the compiler keeps its single load in a
/// register of its own choosing; a named local is allocated differently.
pub fn forward_global_pointers(body: &mut Vec<Stmt>, vars: &[Var], is_temp: &[bool]) {
    let mut uses = HashMap::new();
    crate::inline::count_uses(body, &mut uses);
    let mut defs: HashMap<VarId, usize> = HashMap::new();
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter() {
            if let Stmt::Assign { dst: Expr::Var(v), .. } = s {
                *defs.entry(*v).or_default() += 1;
            }
        }
    });
    fn count(s: &Stmt, t: VarId) -> usize {
        let mut n = 0;
        Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
            if matches!(e, Expr::Var(v) if *v == t) {
                n += 1;
            }
        });
        n
    }
    fn writes_global(s: &Stmt, g: &str) -> bool {
        let mut w = false;
        Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
            if let Expr::AddrOf(x) = e {
                if matches!(&**x, Expr::Global { symbol, .. } if symbol == g) {
                    w = true;
                }
            }
        });
        fn assigns(s: &Stmt, g: &str) -> bool {
            match s {
                Stmt::Assign { dst: Expr::Global { symbol, .. }, .. } => symbol == g,
                Stmt::If { then, els, .. } => then.iter().chain(els.iter()).any(|s| assigns(s, g)),
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => body.iter().any(|s| assigns(s, g)),
                Stmt::For { init, step, body, .. } => init.iter().chain(step.iter()).chain(body.iter()).any(|s| assigns(s, g)),
                Stmt::Switch { cases, .. } => cases.iter().any(|c| c.body.iter().any(|s| assigns(s, g))),
                _ => false,
            }
        }
        w || assigns(s, g)
    }
    fn has_call(s: &Stmt) -> bool {
        let mut c = false;
        Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| {
            if matches!(e, Expr::Call { .. } | Expr::New { .. } | Expr::IncDec { .. }) && !e.is_pure_call() {
                c = true;
            }
        });
        c
    }
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let (t, g) = match &b[i] {
                Stmt::Assign { dst: Expr::Var(t), src: g @ Expr::Global { ty, .. } }
                    if is_temp.get(*t).copied().unwrap_or(false) && defs.get(t) == Some(&1) && is_ptr(strip_cv(ty)) && matches!(vars[*t].kind, VarKind::Local) =>
                {
                    (*t, g.clone())
                }
                _ => {
                    i += 1;
                    continue;
                }
            };
            let Expr::Global { symbol, .. } = &g else { unreachable!() };
            let total = uses.get(&t).copied().unwrap_or(0);
            let mut seen = 0;
            let mut end = i + 1;
            while end < b.len() && seen < total {
                if has_call(&b[end]) || writes_global(&b[end], symbol) {
                    break;
                }
                seen += count(&b[end], t);
                end += 1;
            }
            if total == 0 || seen != total {
                i += 1;
                continue;
            }
            for s in &mut b[i + 1..end] {
                Stmt::rewrite_exprs(std::slice::from_mut(s), &mut |e| {
                    if matches!(e, Expr::Var(v) if *v == t) {
                        *e = g.clone();
                    }
                });
            }
            b.remove(i);
        }
    });
}

/// `if (p) delete p;` -> `delete p;` (the delete expression has its own null check).
pub fn fold_delete_checks(body: &mut Vec<Stmt>) {
    fn strip(e: &Expr) -> &Expr {
        match e {
            Expr::Cast { e, .. } => strip(e),
            e => e,
        }
    }
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter_mut() {
            let Stmt::If { cond, then, els } = s else { continue };
            if !els.is_empty() || then.len() != 1 {
                continue;
            }
            let Stmt::Expr(Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. }) = &then[0] else { continue };
            if symbol != "__delete" || args.len() != 1 {
                continue;
            }
            let p = strip(&args[0]);
            let tested = match strip(cond) {
                Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => strip(l),
                other => other,
            };
            if tested == p {
                *s = then[0].clone();
            }
        }
    });
}

/// Untyped scalar accesses (`Unknown` width-only types from raw loads) whose base got its class
/// type after the access was built: take the member's type from the DB when it agrees with the
/// access width (so locals assigned from them can be typed, `int* rc = p.x4_refCount`).
pub fn refresh_access_types(body: &mut Vec<Stmt>, vars: &[Var], db: &mwdec_core::TypeDb) {
    Stmt::rewrite_exprs(body, &mut |x| {
        let (base, off, ty, ptr) = match x {
            Expr::Load { base, offset, ty } => (&**base, *offset, ty.clone(), true),
            Expr::Member { base, offset, ty } => (&**base, *offset, ty.clone(), false),
            _ => return,
        };
        if !matches!(ty, Type::Unknown { size: 4 | 2 | 1 }) {
            return;
        }
        let bt = types::ty_of(base, vars);
        let ct = if ptr {
            match pointee(&bt) {
                Some(p) => p.clone(),
                None => return,
            }
        } else {
            bt
        };
        let Some(cls) = named(&types::resolve(Some(db), &ct)).map(|s| s.to_string()) else { return };
        let Some((_, ft)) = types::field_path(db, &cls, off, scalar_size(&ty).unwrap_or(0)) else { return };
        if !crate::translate::compatible_scalar(Some(db), &ft, &ty) {
            return;
        }
        match x {
            Expr::Load { ty, .. } | Expr::Member { ty, .. } => *ty = ft,
            _ => {}
        }
    });
}

/// Return types of functions the context doesn't declare (unit statics): the lifter types their
/// result as a word. When every use of a callee's result narrows it to one type (`(u16)f(..)`,
/// returned from a `u16` function), that is the callee's return type (it returns the value
/// already extended, so the caller doesn't extend it again).
pub fn undeclared_returns(body: &mut Vec<Stmt>, ret: &Type, db: Option<&mwdec_core::TypeDb>) {
    let undeclared = |sym: &str, s: &mwdec_core::FuncSig| -> bool {
        crate::sig::ret_unknown(s) && crate::sig::demangle(sym).is_none() && !db.map_or(false, |d| d.decls.contains_key(sym) || d.functions.contains_key(sym))
    };
    let narrow = |t: &Type| -> Option<Type> {
        let r = types::resolve(db, t).into_owned();
        match strip_cv(&r) {
            Type::Int { size: 1 | 2, .. } | Type::Bool => Some(strip_cv(t).clone()),
            _ => None,
        }
    };
    // per callee: the narrow type of every use (None = a plain use)
    let mut uses: HashMap<String, Vec<Option<Type>>> = HashMap::new();
    fn visit(e: &Expr, ctx: Option<Type>, uses: &mut HashMap<String, Vec<Option<Type>>>, und: &dyn Fn(&str, &mwdec_core::FuncSig) -> bool, narrow: &dyn Fn(&Type) -> Option<Type>) {
        match e {
            Expr::Cast { ty, e: inner } if matches!(**inner, Expr::Call { .. }) => {
                visit(inner, narrow(ty), uses, und, narrow);
            }
            Expr::Call { callee, args, .. } => {
                if let Callee::Direct { symbol, sig } = callee {
                    if und(symbol, sig) {
                        uses.entry(symbol.clone()).or_default().push(ctx.clone());
                    }
                }
                if let Callee::Method { this, .. } | Callee::Virtual { this, .. } = callee {
                    visit(this, None, uses, und, narrow);
                }
                if let Callee::Indirect(f) = callee {
                    visit(f, None, uses, und, narrow);
                }
                for a in args {
                    visit(a, None, uses, und, narrow);
                }
            }
            _ => {
                for k in direct_kids(e) {
                    visit(k, None, uses, und, narrow);
                }
            }
        }
    }
    let ret_narrow = narrow(ret);
    let mut stmts = vec![];
    all_stmt_lists(body, &mut stmts);
    for s in &stmts {
        match s {
            Stmt::Return(Some(e)) => {
                if matches!(e, Expr::Call { .. }) {
                    visit(e, ret_narrow.clone(), &mut uses, &undeclared, &narrow);
                } else {
                    visit(e, None, &mut uses, &undeclared, &narrow);
                }
            }
            Stmt::Expr(e) => {
                // a result never used doesn't decide
                if let Expr::Call { callee, args, .. } = e {
                    for a in args {
                        visit(a, None, &mut uses, &undeclared, &narrow);
                    }
                    if let Callee::Method { this, .. } | Callee::Virtual { this, .. } = callee {
                        visit(this, None, &mut uses, &undeclared, &narrow);
                    }
                } else {
                    visit(e, None, &mut uses, &undeclared, &narrow);
                }
            }
            Stmt::Assign { dst, src } => {
                visit(dst, None, &mut uses, &undeclared, &narrow);
                visit(src, None, &mut uses, &undeclared, &narrow);
            }
            Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } | Stmt::For { cond, .. } => visit(cond, None, &mut uses, &undeclared, &narrow),
            Stmt::Switch { e, .. } => visit(e, None, &mut uses, &undeclared, &narrow),
            _ => {}
        }
    }
    let mut chosen: HashMap<String, Type> = HashMap::new();
    for (sym, us) in uses {
        let Some(Some(t)) = us.first().cloned() else { continue };
        if us.iter().all(|u| u.as_ref().map(key) == Some(key(&t))) {
            chosen.insert(sym, t);
        }
    }
    if chosen.is_empty() {
        return;
    }
    Stmt::rewrite_exprs(body, &mut |e| {
        let replace = match e {
            Expr::Cast { e: inner, .. } => match &**inner {
                Expr::Call { callee: Callee::Direct { symbol, .. }, .. } => chosen.get(symbol).cloned(),
                _ => None,
            },
            _ => None,
        };
        if let Some(t) = replace {
            if let Expr::Cast { e: inner, .. } = e {
                let mut c = (**inner).clone();
                if let Expr::Call { ret, .. } = &mut c {
                    *ret = t;
                }
                *e = c;
            }
        } else if let Expr::Call { callee: Callee::Direct { symbol, .. }, ret, .. } = e {
            if let Some(t) = chosen.get(symbol) {
                *ret = t.clone();
            }
        }
    });
}

fn direct_kids(e: &Expr) -> Vec<&Expr> {
    let mut v: Vec<&Expr> = vec![];
    match e {
        Expr::AddrOf(x) | Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } | Expr::IncDec { e: x, .. } => v.push(x),
        Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => v.push(base),
        Expr::Index { base, index, .. } => {
            v.push(base);
            v.push(index);
        }
        Expr::Binary { l, r, .. } => {
            v.push(l);
            v.push(r);
        }
        Expr::Ternary { c, t, f, .. } => {
            v.push(c);
            v.push(t);
            v.push(f);
        }
        Expr::New { placement, args, .. } => v.extend(placement.iter().chain(args.iter())),
        Expr::Construct { args, .. } => v.extend(args.iter()),
        _ => {}
    }
    v
}
