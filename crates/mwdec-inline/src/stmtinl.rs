//! Statement-level container inlines the template library can't express directly.
//!
//! * Negated predicates: `count != 0` in a condition is `!empty()` when a bool inline of the
//!   object's class is exactly `count == 0` (the library matches `== 0` itself; the negation has
//!   no template of its own).
//! * Forwarders: a public inline member whose body is one call (`void pop_front() {
//!   erase(mStart); }`, `void push_back(const T& v) { do_insert_before(mEnd, v); }`) compiles to
//!   the call it ends in, which the library rejects as ambiguous. A call statement to that
//!   callee whose arguments are the forwarder's member arguments (and its parameters) is the
//!   forwarder. The member arguments come from the header declaration's body, the callee from
//!   the declaration itself or, one level deeper, from the intermediate inline's template
//!   (`erase(it)` = `do_erase(it.node)`).

use crate::matcher::{res, Defs, Env};
use crate::template::Shape;
use crate::InlineLib;
use mwdec_core::{DeclInfo, FuncSig, Type, TypeDb};
use mwdec_lift::{BinOp, Callee, Expr, IrFunction, Stmt, UnOp, Var};

thread_local! {
    /// Draft variant: a forwarder applied to a container that is a member of another object
    /// goes through a local holding the container's address (`T& l = x->list; l.pop_front();`):
    /// the compiler then computes the address first and reads the member through it.
    static CONTAINER_LOCALS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Forwarders applied to such containers since the last `take_forwarded`.
    static FORWARDED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Switch the container-local variant on or off (this thread).
pub fn set_container_locals(on: bool) {
    CONTAINER_LOCALS.with(|c| c.set(on));
}

/// Number of forwarders applied to member containers since the last call (this thread).
pub fn take_forwarded() -> usize {
    FORWARDED.with(|c| c.replace(0))
}

/// The forwarder name marker on calls this pass made (`Callee::Method::symbol`).
const FWD_MARK: &str = "@stmtinl";

fn norm(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Class (TypeDb key) of the object `this` points to.
fn class_of(this: &Expr, vars: &[Var], db: &TypeDb) -> Option<String> {
    let t = mwdec_lift::types::ty_of(this, vars);
    let p = mwdec_lift::pointee(&t)?;
    crate::util::class_name(p, db)
}

/// `(base pointer, offset)` of a member read `*(p + off)`.
fn member_read(e: &Expr) -> Option<(&Expr, i32)> {
    match e {
        Expr::Load { base, offset, .. } => Some((base, *offset)),
        Expr::Cast { e, .. } => member_read(e),
        _ => None,
    }
}

fn same(a: &Expr, b: &Expr, defs: &Defs) -> bool {
    crate::matcher::teq(a, b, defs)
}

// ---------------------------------------------------------------- negated predicates

/// `(template index, member offset)` of bool inlines `return member == 0;` per class.
fn zero_predicates(lib: &InlineLib) -> Vec<(usize, String, i32)> {
    let mut out = vec![];
    for (i, t) in lib.templates.iter().enumerate() {
        let (Some(c), Shape::Scalar(Expr::Binary { op: BinOp::Eq, l, r, .. })) = (&t.class, &t.shape) else { continue };
        if t.holes.len() != 1 || !matches!(crate::util::strip(&t.sig.ret), Type::Bool) || r.as_int() != Some(0) {
            continue;
        }
        if let Expr::Load { base, offset, .. } = &**l {
            if matches!(&**base, Expr::Var(0)) {
                out.push((i, norm(c), *offset));
            }
        }
    }
    out
}

/// The object of a class with a predicate on the member at `off` of the object `p` points to:
/// (its address, the member's offset inside it), innermost first.
fn holder(p: &Expr, off: i32, env: &Env) -> Vec<(Expr, String, i32)> {
    let mut v = crate::addr::aggregates_containing(p, off, env);
    v.reverse();
    let outer = crate::addr::outer_class(p, env);
    v.into_iter()
        .map(|(start, cls, _)| {
            let addr = if start == 0 && outer.as_deref() == Some(cls.as_str()) {
                p.clone()
            } else {
                let lv = match p {
                    Expr::AddrOf(x) => Expr::Member { base: x.clone(), offset: start, ty: Type::Named(cls.clone()) },
                    _ => Expr::Load { base: Box::new(p.clone()), offset: start, ty: Type::Named(cls.clone()) },
                };
                Expr::AddrOf(Box::new(lv))
            };
            (addr, cls, off - start)
        })
        .collect()
}

fn negate_in_cond(e: &mut Expr, preds: &[(usize, String, i32)], env: &Env) -> usize {
    match e {
        Expr::Binary { op: BinOp::LogAnd | BinOp::LogOr, l, r, .. } => negate_in_cond(l, preds, env) + negate_in_cond(r, preds, env),
        Expr::Unary { op: UnOp::Not, e: x, .. } => negate_in_cond(x, preds, env),
        Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => {
            let Some((base, off)) = member_read(l) else { return 0 };
            for (addr, cls, rel) in holder(base, off, env) {
                if let Some((ti, _, _)) = preds.iter().find(|(_, c, o)| *c == norm(&cls) && *o == rel) {
                    let call = crate::matcher::make_call(&env.lib.templates[*ti], vec![addr]);
                    *e = Expr::Unary { op: UnOp::Not, e: Box::new(call), ty: Type::Bool };
                    return 1;
                }
            }
            0
        }
        _ => 0,
    }
}

fn negated_predicates(body: &mut Vec<Stmt>, env: &Env) -> usize {
    let preds = zero_predicates(env.lib);
    if preds.is_empty() {
        return 0;
    }
    let mut n = 0;
    fn conds(b: &mut Vec<Stmt>, f: &mut dyn FnMut(&mut Expr)) {
        for s in b.iter_mut() {
            match s {
                Stmt::If { cond, then, els } => {
                    f(cond);
                    conds(then, f);
                    conds(els, f);
                }
                Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                    f(cond);
                    conds(body, f);
                }
                Stmt::For { init, cond, step, body } => {
                    f(cond);
                    conds(init, f);
                    conds(step, f);
                    conds(body, f);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        conds(&mut c.body, f);
                    }
                }
                _ => {}
            }
        }
    }
    conds(body, &mut |c| n += negate_in_cond(c, &preds, env));
    n
}

// ---------------------------------------------------------------- forwarders

/// A forwarder: public inline member of `class` whose body is `name ( args ) ;`.
struct Fwd {
    sig: FuncSig,
    callee: String,
    /// per argument of the call: a member offset or a parameter index
    args: Vec<Arg>,
}

#[derive(Clone, Copy, PartialEq)]
enum Arg {
    Member(i32),
    Param(usize),
}

fn forwarders(class: &str, db: &TypeDb) -> Vec<Fwd> {
    let base = mwdec_lift::sig::split_scope(class).1.to_string();
    let key = strip_targs(class);
    let Some(c) = mwdec_lift::sig::find_class(db, class) else { return vec![] };
    let prefix = format!("{key}::");
    let mut out = vec![];
    for (k, ds) in db.decls.range(prefix.clone()..) {
        let Some(m) = k.strip_prefix(&prefix) else { break };
        if m.contains("::") || m.starts_with('~') || m == base.split('<').next().unwrap_or("") {
            continue;
        }
        for d in ds {
            let Some(f) = forwarder(d, m, class, c, db) else { continue };
            out.push(f);
        }
    }
    out
}

fn strip_targs(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for ch in s.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

fn forwarder(d: &DeclInfo, name: &str, class: &str, c: &mwdec_core::Class, _db: &TypeDb) -> Option<Fwd> {
    if !d.is_inline_defined || d.is_static || d.is_virtual || d.access != mwdec_core::Access::Public || !matches!(d.ret, Type::Void) {
        return None;
    }
    let body = d.inline_body.as_deref()?;
    let t: Vec<&str> = body.split_whitespace().collect();
    // `callee ( a , b ) ;`
    if t.len() < 4 || t[1] != "(" || t[t.len() - 2] != ")" || t[t.len() - 1] != ";" {
        return None;
    }
    let callee = t[0];
    if !callee.chars().all(|ch| ch.is_alphanumeric() || ch == '_') || callee == name {
        return None;
    }
    let inner = &t[2..t.len() - 2];
    let mut args = vec![];
    if !inner.is_empty() {
        for a in inner.split(|x| *x == ",") {
            let [one] = a else { return None };
            if let Some(i) = d.params.iter().position(|p| p.name.as_deref() == Some(*one)) {
                args.push(Arg::Param(i));
            } else if let Some(f) = c.fields.iter().find(|f| f.name == *one && f.bitfield.is_none()) {
                args.push(Arg::Member(f.offset as i32));
            } else {
                return None;
            }
        }
    }
    // at least one member argument (else the forwarder is a plain wrapper the emitter already
    // knows: `Routed::Wrapper`)
    if !args.iter().any(|a| matches!(a, Arg::Member(_))) {
        return None;
    }
    let sig = FuncSig {
        qualified_name: format!("{class}::{name}"),
        mangled: None,
        ret: Type::Void,
        params: d.params.clone(),
        this_class: Some(class.to_string()),
        is_const: d.is_const,
        is_static: false,
        is_virtual: false,
        variadic: false,
    };
    Some(Fwd { sig, callee: callee.to_string(), args })
}

/// Is `name` (a member or an accessor returning one) the first word of class `cls` (through
/// bases at offset 0)?
fn first_word(cls: &str, name: &str, db: &TypeDb, depth: u32) -> bool {
    let Some(c) = mwdec_lift::sig::find_class(db, cls) else { return false };
    if let Some(f0) = c.fields.iter().find(|f| f.offset == 0 && f.bitfield.is_none()) {
        if f0.name == name {
            return true;
        }
        let body = format!("return {} ;", f0.name);
        let body2 = format!("return this -> {} ;", f0.name);
        let k = format!("{}::{name}", strip_targs(cls));
        if db.decls.get(&k).is_some_and(|ds| ds.iter().any(|x| x.params.is_empty() && matches!(x.inline_body.as_deref(), Some(b) if b == body || b == body2))) {
            return true;
        }
    }
    depth < 4 && c.bases.iter().any(|b| b.offset == 0 && first_word(&b.name, name, db, depth + 1))
}

/// How the forwarded call appears compiled: (compiled callee, per compiled argument the
/// forwarder argument it comes from and the offset read from it, if any).
fn compiled_chain(f: &Fwd, class: &str, lib: &InlineLib, db: &TypeDb) -> Vec<(String, Vec<(usize, Option<i32>)>)> {
    let mut out = vec![(f.callee.clone(), (0..f.args.len()).map(|i| (i, None)).collect())];
    // one inline level from the declaration: `iterator erase(const iterator& item) { return
    // do_erase(item.get_node()); }` (an accessor of the argument's first member)
    let key = format!("{}::{}", strip_targs(class), f.callee);
    for d in db.decls.get(&key).map(|v| v.as_slice()).unwrap_or(&[]) {
        if d.params.len() != f.args.len() || !d.is_inline_defined {
            continue;
        }
        let Some(body) = d.inline_body.as_deref() else { continue };
        let mut t: Vec<&str> = body.split_whitespace().collect();
        if t.first() == Some(&"return") {
            t.remove(0);
        }
        if t.len() < 4 || t[1] != "(" || t[t.len() - 2] != ")" || t[t.len() - 1] != ";" {
            continue;
        }
        let cal = t[0];
        let inner = &t[2..t.len() - 2];
        let mut m = vec![];
        for a in inner.split(|x| *x == ",") {
            let pi = |n: &str| d.params.iter().position(|p| p.name.as_deref() == Some(n));
            let first_member = |pidx: usize, mname: &str| -> bool {
                // the parameter's class (spelled relative to the container: `iterator`)
                let mut t = &d.params[pidx].ty;
                loop {
                    match t {
                        Type::Ref(x) | Type::Ptr(x) | Type::Const(x) => t = x,
                        _ => break,
                    }
                }
                let Type::Named(n) = t else { return false };
                let pc = mwdec_lift::sig::find_class(db, &format!("{class}::{n}")).or_else(|| mwdec_lift::sig::find_class(db, n)).map(|c| c.name.clone());
                pc.is_some_and(|pc| first_word(&pc, mname, db, 0))
            };
            match a {
                [p] => match pi(p) {
                    Some(i) => m.push((i, None)),
                    None => break,
                },
                [p, ".", n, "(", ")"] | [p, ".", n] => match pi(p) {
                    Some(i) if first_member(i, n) => m.push((i, Some(0))),
                    _ => break,
                },
                _ => break,
            }
        }
        if !inner.is_empty() && m.len() == inner.split(|x| *x == ",").count() {
            out.push((cal.to_string(), m));
        }
    }
    // one inline level: `erase(it)` whose template is `do_erase(*(it + 0))`
    let want = norm(&format!("{class}::{}", f.callee));
    for t in &lib.templates {
        if norm(&t.name) != want || t.holes.len() != f.args.len() + 1 {
            continue;
        }
        // (a call returned as is, or as the one member of a returned object: an iterator)
        let call = match &t.shape {
            Shape::Scalar(c) => c,
            Shape::Object { comps, .. } if comps.len() == 1 && comps[0].off == 0 => &comps[0].pat,
            _ => continue,
        };
        let Expr::Call { callee: Callee::Method { sig, this, .. }, args, .. } = call else { continue };
        if !matches!(&**this, Expr::Var(0)) {
            continue;
        }
        let mut m = vec![];
        for a in args {
            match a {
                Expr::Var(h) if *h >= 1 => m.push((*h - 1, None)),
                Expr::Load { base, offset, .. } => match &**base {
                    Expr::Var(h) if *h >= 1 => m.push((*h - 1, Some(*offset))),
                    _ => break,
                },
                _ => break,
            }
        }
        if m.len() == args.len() {
            out.push((mwdec_lift::sig::split_scope(&sig.qualified_name).1.to_string(), m));
        }
    }
    out
}

fn forward_calls(body: &mut Vec<Stmt>, env: &Env) -> usize {
    let (lib, vars, db, defs) = (env.lib, env.vars, env.db, env.defs);
    let mut n = 0;
    let mut cache: std::collections::HashMap<String, Vec<(Fwd, Vec<(String, Vec<(usize, Option<i32>)>)>)>> = Default::default();
    Stmt::for_each_block_mut(body, &mut |b| {
        for s in b.iter_mut() {
            let Stmt::Expr(Expr::Call { callee: Callee::Method { sig, this, .. }, args, .. }) = s else { continue };
            // (the called member's class: the object may be a raw address)
            let Some(cls) = sig.this_class.as_deref().and_then(|c| mwdec_lift::sig::find_class(db, c)).map(|c| c.name.clone()).or_else(|| class_of(this, vars, db)) else { continue };
            let fwds = cache.entry(cls.clone()).or_insert_with(|| {
                forwarders(&cls, db).into_iter().map(|f| {
                    let ch = compiled_chain(&f, &cls, lib, db);
                    (f, ch)
                }).collect()
            });
            let called = mwdec_lift::sig::split_scope(&sig.qualified_name).1.to_string();
            if std::env::var_os("MWDI_TRACE_STMTINL").is_some() {
                eprintln!("stmtinl: call {called} on {cls}: {} forwarders {:?}", fwds.len(), fwds.iter().map(|(f, ch)| (f.sig.qualified_name.clone(), ch.iter().map(|c| c.0.clone()).collect::<Vec<_>>())).collect::<Vec<_>>());
            }
            let (tb, tk) = crate::addr::canon_ptr(this, env);
            let mut hit: Option<(FuncSig, Vec<Expr>)> = None;
            'f: for (f, chains) in fwds.iter() {
                for (cal, map) in chains {
                    if *cal != called || map.len() != args.len() {
                        continue;
                    }
                    let mut params: Vec<Option<Expr>> = vec![None; f.sig.params.len()];
                    for ((fi, sub), a) in map.iter().zip(args.iter()) {
                        let a = res(a, defs);
                        match (f.args[*fi], sub) {
                            // (an iterator object built from the member: its first word)
                            (Arg::Member(off), None | Some(0)) => {
                                let Some((pb, o)) = crate::addr::access(a, env) else { continue 'f };
                                if o != tk + off || !same(&pb, &tb, defs) {
                                    continue 'f;
                                }
                            }
                            (Arg::Param(pi), None) => params[pi] = Some(a.clone()),
                            _ => continue 'f,
                        }
                    }
                    if params.iter().all(|p| p.is_some()) {
                        hit = Some((f.sig.clone(), params.into_iter().map(|p| p.unwrap()).collect()));
                        break 'f;
                    }
                }
            }
            if let Some((fsig, fargs)) = hit {
                let this = this.clone();
                if !matches!(res(&this, defs), Expr::Var(_)) {
                    FORWARDED.with(|c| c.set(c.get() + 1));
                }
                *s = Stmt::Expr(Expr::Call { callee: Callee::Method { symbol: FWD_MARK.into(), sig: fsig, this, qualified: false }, args: fargs, ret: Type::Void });
                n += 1;
            }
        }
    });
    n
}

/// Both rewrites over a function body (after the template passes).
pub fn apply(ir: &mut IrFunction, lib: &InlineLib, db: &TypeDb) -> usize {
    if std::env::var_os("MWDI_NO_STMTINL").is_some() {
        return 0;
    }
    let defs = crate::matcher::build_defs(&ir.body, &ir.vars);
    let vars = ir.vars.clone();
    let idx = crate::matcher::index(lib);
    let env = Env { db, vars: &vars, defs: &defs, lib, objects: &idx.objects };
    let mut n = forward_calls(&mut ir.body, &env);
    n += negated_predicates(&mut ir.body, &env);
    let on = CONTAINER_LOCALS.with(|c| c.get());
    let sc = stream_ctors(&mut ir.body, lib, db, on);
    if sc > 0 {
        FORWARDED.with(|c| c.set(c.get() + sc));
        if on {
            n += sc;
        }
    }
    n
}

/// `C(x.f(), x.g())` where a constructor of `C` taking `x` reads exactly those members in that
/// order (`CAdditiveAnimationInfo(CInputStream& in) : a(in.ReadFloat()), b(in.ReadFloat())`):
/// in the variant, that constructor (the stores then follow each read, as in the target when the
/// source used it); otherwise only counted as an opportunity.
fn stream_ctors(body: &mut Vec<Stmt>, lib: &InlineLib, db: &TypeDb, apply: bool) -> usize {
    let mut n = 0;
    Stmt::rewrite_exprs(body, &mut |e| {
        let Expr::Construct { class, args, .. } = &*e else { return };
        if args.len() < 2 {
            return;
        }
        let Some(cn) = crate::util::class_name(class, db) else { return };
        // the reads: method calls without arguments on one object
        let mut obj: Option<&Expr> = None;
        let mut sigs = vec![];
        for a in args {
            let mut a = a;
            while let Expr::Cast { e: x, .. } = a {
                a = x;
            }
            let Expr::Call { callee: Callee::Method { sig, this, .. }, args: ca, .. } = a else { return };
            if !ca.is_empty() || obj.is_some_and(|o| o != &**this) {
                return;
            }
            obj = Some(this);
            sigs.push(norm(&sig.qualified_name));
        }
        let Some(obj) = obj else { return };
        for t in &lib.templates {
            if !matches!(t.kind, crate::probe::CallKind::Ctor) || t.holes.len() != 1 || t.class.as_deref().map(norm) != Some(norm(&cn)) {
                continue;
            }
            let Shape::Object { comps, .. } = &t.shape else { continue };
            if comps.len() != sigs.len() {
                continue;
            }
            let mut cs: Vec<&crate::template::Comp> = comps.iter().collect();
            cs.sort_by_key(|c| c.off);
            let same = cs.iter().zip(&sigs).all(|(c, sg)| match &c.pat {
                Expr::Call { callee: Callee::Method { sig, this, .. }, args, .. } => args.is_empty() && norm(&sig.qualified_name) == *sg && matches!(&**this, Expr::AddrOf(x) if matches!(**x, Expr::Var(0))) ,
                _ => false,
            });
            if !same {
                continue;
            }
            n += 1;
            if apply {
                // the object (the reads went through its address)
                let o = match obj {
                    Expr::AddrOf(x) => (**x).clone(),
                    o => Expr::Load { base: Box::new(o.clone()), offset: 0, ty: Type::Named(t.class.clone().unwrap_or_default()) },
                };
                let mut sig = t.sig.clone();
                sig.ret = Type::Void;
                *e = Expr::Construct { class: class.clone(), ctor: Some(sig), args: vec![o] };
            }
            return;
        }
    });
    n
}

/// After the template passes: drop the forwarder marks; in the container-local variant, a
/// forwarder on a member container goes through a local holding its address.
pub fn finish(ir: &mut IrFunction) {
    let on = CONTAINER_LOCALS.with(|c| c.get());
    let mut new_vars: Vec<Var> = vec![];
    let base = ir.vars.len();
    Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            let mut insert = None;
            if let Stmt::Expr(Expr::Call { callee: Callee::Method { symbol, sig, this, .. }, .. }) = &mut b[i] {
                if symbol == FWD_MARK {
                    symbol.clear();
                    if on && !matches!(**this, Expr::Var(_)) {
                        if let Some(c) = sig.this_class.clone() {
                            let last = mwdec_lift::sig::split_scope(&c).1.split('<').next().unwrap_or("c").to_lowercase();
                            let v = base + new_vars.len();
                            new_vars.push(Var { name: last, ty: Type::Ptr(Box::new(Type::Named(c))), kind: mwdec_lift::VarKind::Local });
                            let addr = std::mem::replace(&mut **this, Expr::Var(v));
                            insert = Some(Stmt::Assign { dst: Expr::Var(v), src: addr });
                        }
                    }
                }
            }
            if let Some(st) = insert {
                b.insert(i, st);
                i += 1;
            }
            i += 1;
        }
    });
    ir.vars.extend(new_vars);
}
