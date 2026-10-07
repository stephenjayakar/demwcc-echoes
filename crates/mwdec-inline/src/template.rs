//! Templates: lifted probe bodies with the probe parameters as holes.

use crate::probe::{CallKind, Probe};
use crate::util::*;
use mwdec_core::{FuncSig, Type, TypeDb};
use mwdec_lift::{Expr, IrFunction, Stmt, VarId, VarKind};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub enum HoleKind {
    /// A scalar value of this type.
    Scalar(Type),
    /// An object passed by pointer/reference: `class`, whether the parameter is a pointer (vs
    /// reference / by value), and whether it may bind to a temporary (const ref or by value).
    Obj { class: String, ptr: bool, temp_ok: bool },
    /// A local variable of a statement template (not an argument).
    Local,
    /// A reference to a scalar (`const int&`): binds to the referenced lvalue's value.
    ScalarRef(Type),
}

/// One scalar component of an object value: `*(obj + off)` of type `ty` equals `pat`.
#[derive(Clone, Debug)]
pub struct Comp {
    pub off: i32,
    pub ty: Type,
    pub pat: Expr,
}

#[derive(Clone, Debug)]
pub enum Shape {
    /// `return pat;` (scalar or pointer result)
    Scalar(Expr),
    /// Returns an object of `class` by value, member-wise.
    Object { class: String, comps: Vec<Comp> },
    /// Stores through object hole `hole` (`*a += b`), member-wise.
    Mutate { hole: usize, comps: Vec<Comp> },
    /// Statements (stores, calls, control flow) and the returned value, if any.
    Stmts { stmts: Vec<Stmt>, result: Option<Expr> },
}

#[derive(Clone, Debug)]
pub struct Template {
    /// Display name (qualified).
    pub name: String,
    pub kind: CallKind,
    pub sig: FuncSig,
    pub class: Option<String>,
    pub holes: Vec<HoleKind>,
    pub shape: Shape,
    /// Operator/call nodes in the pattern (specificity).
    pub ops: usize,
    pub ret_ref: bool,
}

pub fn hole_kind(t: &Type, db: &TypeDb) -> HoleKind {
    let s = strip(t);
    match s {
        Type::Ptr(inner) | Type::Ref(inner) => {
            let is_ptr = matches!(s, Type::Ptr(_));
            let is_const = matches!(&**inner, Type::Const(_));
            if let Some(c) = class_name(inner, db) {
                return HoleKind::Obj { class: c, ptr: is_ptr, temp_ok: !is_ptr && is_const };
            }
            if !is_ptr && mwdec_lift::scalar_size(strip(inner)).is_some() && !matches!(strip(inner), Type::Unknown { .. }) {
                return HoleKind::ScalarRef((**inner).clone());
            }
            HoleKind::Scalar(t.clone())
        }
        _ => {
            if let Some(c) = class_name(s, db) {
                return HoleKind::Obj { class: c, ptr: false, temp_ok: true };
            }
            HoleKind::Scalar(t.clone())
        }
    }
}

/// Substitute single-definition locals into their uses (pure trees), dropping their defs.
fn expand_temps(ir: &IrFunction) -> Option<Vec<Stmt>> {
    let mut defs: HashMap<VarId, Expr> = HashMap::new();
    let mut count: HashMap<VarId, usize> = HashMap::new();
    for s in &ir.body {
        if let Stmt::Assign { dst: Expr::Var(v), .. } = s {
            *count.entry(*v).or_default() += 1;
        }
    }
    let mut out = vec![];
    for s in &ir.body {
        match s {
            Stmt::Assign { dst: Expr::Var(v), src } if count[v] == 1 && matches!(ir.vars[*v].kind, VarKind::Local) && !src.has_call() => {
                let mut e = src.clone();
                subst(&mut e, &defs);
                defs.insert(*v, e);
            }
            Stmt::If { .. } | Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::Switch { .. } | Stmt::Goto(_) | Stmt::Label(_) => return None,
            s => {
                let mut s = s.clone();
                Stmt::rewrite_exprs(std::slice::from_mut(&mut s), &mut |e| {
                    if let Expr::Var(v) = e {
                        if let Some(d) = defs.get(v) {
                            *e = d.clone();
                        }
                    }
                });
                out.push(s);
            }
        }
    }
    Some(out)
}

fn subst(e: &mut Expr, defs: &HashMap<VarId, Expr>) {
    e.rewrite(&mut |x| {
        if let Expr::Var(v) = x {
            if let Some(d) = defs.get(v) {
                *x = d.clone();
            }
        }
    });
}

/// Rename probe parameters to hole indices; None if other variables remain.
fn to_holes(e: &Expr, map: &HashMap<VarId, usize>) -> Option<Expr> {
    let mut ok = true;
    let mut e = e.clone();
    e.rewrite(&mut |x| {
        if let Expr::Var(v) = x {
            match map.get(v) {
                Some(h) => *x = Expr::Var(*h),
                None => ok = false,
            }
        }
    });
    ok.then_some(e)
}

/// Flat scalar fields (offset, type) of a class, in layout order.
pub fn flat_fields(db: &TypeDb, cls: &str) -> Option<Vec<(i32, Type)>> {
    let mut v = vec![];
    flat_into(db, cls, 0, &mut v, 0)?;
    Some(v)
}

fn flat_into(db: &TypeDb, cls: &str, base: i32, out: &mut Vec<(i32, Type)>, depth: u32) -> Option<()> {
    if depth > 8 {
        return None;
    }
    let c = mwdec_lift::sig::find_class(db, cls)?;
    if c.is_union || c.vptr_offset.is_some() {
        return None;
    }
    for b in &c.bases {
        flat_into(db, &b.name, base + b.offset as i32, out, depth + 1)?;
    }
    for f in &c.fields {
        if f.bitfield.is_some() {
            return None;
        }
        let rt = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
        match strip(&rt) {
            Type::Array(e, n) => {
                let es = mwdec_lift::types::size_of(Some(db), e)?;
                if mwdec_lift::types::is_aggregate(Some(db), e) {
                    return None;
                }
                for k in 0..*n {
                    out.push((base + f.offset as i32 + (k * es) as i32, (**e).clone()));
                }
            }
            _ if mwdec_lift::types::is_aggregate(Some(db), &rt) => {
                flat_into(db, mwdec_lift::named(&rt)?, base + f.offset as i32, out, depth + 1)?;
            }
            t => out.push((base + f.offset as i32, t.clone())),
        }
    }
    Some(())
}

/// Components of an object value expression `e` (an lvalue of `class`, or a construction).
fn object_comps(e: &Expr, class: &str, db: &TypeDb) -> Option<Vec<Comp>> {
    let fields = flat_fields(db, class)?;
    match e {
        Expr::Construct { args, .. } if args.len() == fields.len() => Some(fields.iter().zip(args).map(|((o, t), a)| Comp { off: *o, ty: t.clone(), pat: a.clone() }).collect()),
        Expr::Load { base, offset, .. } => Some(fields.iter().map(|(o, t)| Comp { off: *o, ty: t.clone(), pat: Expr::Load { base: base.clone(), offset: offset + o, ty: t.clone() } }).collect()),
        _ => None,
    }
}

pub fn count_ops(e: &Expr) -> usize {
    let mut n = 0;
    e.walk(&mut |x| {
        if matches!(x, Expr::Binary { .. } | Expr::Unary { .. } | Expr::Call { .. } | Expr::Ternary { .. }) {
            n += 1;
        }
    });
    n
}

/// Build the template of one lifted probe. `Err` carries the reason it is unsupported.
pub fn from_probe(p: &Probe, ir: &IrFunction, db: &TypeDb) -> Result<Template, String> {
    match from_probe_expr(p, ir, db) {
        Ok(t) => Ok(t),
        Err(e) if e == "virtual dispatch" || e == "not inlined" => Err(e),
        Err(e) => crate::stmts::from_probe(p, ir, db).map_err(|e2| format!("{e}; {e2}")),
    }
}

fn from_probe_expr(p: &Probe, ir: &IrFunction, db: &TypeDb) -> Result<Template, String> {
    let Some(body) = expand_temps(ir) else {
        return from_cflow_probe(p, ir, db);
    };
    let mut map: HashMap<VarId, usize> = HashMap::new();
    for (i, v) in ir.params.iter().enumerate() {
        map.insert(*v, i);
    }
    if ir.params.len() != p.params.len() {
        return Err(format!("param count {} vs {}", ir.params.len(), p.params.len()));
    }
    let mut holes: Vec<HoleKind> = p.params.iter().map(|t| hole_kind(t, db)).collect();
    if matches!(p.kind, CallKind::Method) && p.decl.is_const {
        // a const member function may be called on a temporary: `(a - b).MagSquared()`
        if let Some(HoleKind::Obj { temp_ok, .. }) = holes.get_mut(0) {
            *temp_ok = true;
        }
    }
    let sret = ir.vars.iter().position(|v| v.kind == VarKind::StructRet);
    let ret_class = class_name(&p.ret, db);
    let shape = match (&body[..], &ret_class) {
        ([Stmt::Return(Some(e))], Some(c)) if !matches!(strip(&p.ret), Type::Ptr(_)) => {
            let comps = object_comps(e, c, db).ok_or("object return form")?;
            let comps = comps.into_iter().map(|c| to_holes(&c.pat, &map).map(|pat| Comp { pat, ..c })).collect::<Option<Vec<_>>>().ok_or("free vars")?;
            Shape::Object { class: c.clone(), comps }
        }
        ([Stmt::Return(Some(e))], _) => Shape::Scalar(to_holes(e, &map).ok_or("free vars")?),
        _ => {
            // member-wise stores (to the return slot or through one object parameter)
            let mut comps = vec![];
            let mut target: Option<Option<usize>> = None;
            for s in &body {
                match s {
                    Stmt::Return(None) => {}
                    // `return *this;` of a mutator returning a reference
                    Stmt::Return(Some(Expr::Var(v))) if p.ret_ref && map.contains_key(v) => {}
                    Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } => {
                        let t = match &**base {
                            Expr::Var(v) if Some(*v) == sret => None,
                            Expr::Var(v) if map.contains_key(v) => Some(map[v]),
                            _ => return Err("store base".into()),
                        };
                        if target.map_or(false, |x| x != t) {
                            return Err("stores to two objects".into());
                        }
                        target = Some(t);
                        comps.push(Comp { off: *offset, ty: ty.clone(), pat: to_holes(src, &map).ok_or("free vars")? });
                    }
                    Stmt::Assign { dst, src } if matches!(dst, Expr::Var(v) if Some(*v) == sret) || matches!(dst, Expr::Load{base,offset:0,..} if matches!(**base, Expr::Var(v) if Some(v)==sret)) => {
                        let c = ret_class.as_ref().ok_or("sret without class")?;
                        let oc = object_comps(src, c, db).ok_or("object copy form")?;
                        for c in oc {
                            comps.push(Comp { pat: to_holes(&c.pat, &map).ok_or("free vars")?, ..c });
                        }
                        target = Some(None);
                    }
                    _ => return Err(format!("stmt {:?}", std::mem::discriminant(s))),
                }
            }
            match target {
                Some(None) => Shape::Object { class: ret_class.clone().ok_or("no class")?, comps },
                Some(Some(h)) => Shape::Mutate { hole: h, comps },
                None => return Err("empty".into()),
            }
        }
    };
    // a virtual function called through the object: the probe is a plain virtual call
    let virt = |e: &Expr| matches!(e, Expr::Call { callee: mwdec_lift::Callee::Virtual { .. }, .. });
    // not inlined (too big, template without `inline`): the probe just calls it
    let base = |q: &str| q.split('<').next().unwrap_or(q).to_string();
    let itself = |e: &Expr| match e {
        Expr::Call { callee: mwdec_lift::Callee::Direct { sig, .. } | mwdec_lift::Callee::Method { sig, .. }, .. } => base(&sig.qualified_name) == base(&p.sig.qualified_name),
        _ => false,
    };
    match &shape {
        Shape::Scalar(e) if virt(e) => return Err("virtual dispatch".into()),
        Shape::Scalar(e) if itself(e) => return Err("not inlined".into()),
        _ => {}
    }
    let ops = match &shape {
        Shape::Scalar(e) => count_ops(e),
        Shape::Object { comps, .. } | Shape::Mutate { comps, .. } => comps.iter().map(|c| count_ops(&c.pat)).sum(),
        Shape::Stmts { .. } => 0,
    };
    Ok(Template { name: p.sig.qualified_name.clone(), kind: p.kind.clone(), sig: p.sig.clone(), class: p.class.clone(), holes, shape, ops, ret_ref: p.ret_ref })
}

/// A probe whose body has control flow: fold it into one value expression (ternaries).
fn from_cflow_probe(p: &Probe, ir: &IrFunction, db: &TypeDb) -> Result<Template, String> {
    let mut env = crate::cflow::Env::new();
    let ret = crate::cflow::fold_list(&ir.body, &mut env, true).map_err(|_| "control flow")?.ok_or("control flow (no value)")?;
    let mut map: HashMap<VarId, usize> = HashMap::new();
    for (i, v) in ir.params.iter().enumerate() {
        map.insert(*v, i);
    }
    if ir.params.len() != p.params.len() || class_name(&p.ret, db).is_some() && !matches!(strip(&p.ret), Type::Ptr(_)) {
        return Err("control flow object".into());
    }
    let mut e = to_holes(&ret, &map).ok_or("free vars")?;
    crate::cflow::canon(&mut e);
    let mut holes: Vec<HoleKind> = p.params.iter().map(|t| hole_kind(t, db)).collect();
    if matches!(p.kind, CallKind::Method) && p.decl.is_const {
        if let Some(HoleKind::Obj { temp_ok, .. }) = holes.get_mut(0) {
            *temp_ok = true;
        }
    }
    let ops = count_ops(&e);
    Ok(Template { name: p.sig.qualified_name.clone(), kind: p.kind.clone(), sig: p.sig.clone(), class: p.class.clone(), holes, shape: Shape::Scalar(e), ops, ret_ref: p.ret_ref })
}
