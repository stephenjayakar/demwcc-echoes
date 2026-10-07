//! Struct locals MWCC kept in registers. `CVector3f r = a; if (c) r = f(); return r;` compiles to
//! three float registers that are always assigned together from the members of one object and
//! only read to build the result; the draft's `x = a.x; y = a.y; z = a.z; ... return
//! CVector3f(x, y, z);` is not the same code (the epilogue schedules differently). Such register
//! groups become one object local again.

use crate::ir::*;
use mwdec_core::{Type, TypeDb};
use std::collections::{HashMap, HashSet};

fn member_access(e: &Expr) -> Option<(Expr, i32)> {
    match e {
        Expr::Member { base, offset, ty } if matches!(ty, Type::Float { size: 4 }) => Some(((**base).clone(), *offset)),
        Expr::Load { base, offset, ty } if matches!(ty, Type::Float { size: 4 }) => Some((Expr::Load { base: base.clone(), offset: 0, ty: Type::Unknown { size: 0 } }, *offset)),
        _ => None,
    }
}

/// The object expression whose members are read at `base` + `off`.
fn object_at(base: &Expr, off: i32, ty: &Type) -> Expr {
    match base {
        Expr::Load { base: b, .. } => Expr::Load { base: b.clone(), offset: off, ty: ty.clone() },
        Expr::Var(_) | Expr::Global { .. } if off == 0 => base.clone(),
        other => Expr::Member { base: Box::new(other.clone()), offset: off, ty: ty.clone() },
    }
}

/// Group assignments `x = B.@o; y = B.@(o+4); z = B.@(o+8)` (consecutive statements).
fn group_at(b: &[Stmt], i: usize, trio: &[VarId]) -> Option<(Expr, i32)> {
    let n = trio.len();
    if i + n > b.len() {
        return None;
    }
    let mut base: Option<(Expr, i32)> = None;
    for k in 0..n {
        let Stmt::Assign { dst: Expr::Var(v), src } = &b[i + k] else { return None };
        if *v != trio[k] {
            return None;
        }
        let (bb, off) = member_access(src)?;
        match &base {
            None => base = Some((bb, off)),
            Some((b0, o0)) => {
                if *b0 != bb || off != o0 + 4 * k as i32 {
                    return None;
                }
            }
        }
    }
    base
}

/// Classes that are just N floats (`CVector3f`, `CVector2f`, ...): name -> N.
fn float_classes(db: &TypeDb) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for (name, c) in &db.classes {
        if !c.bases.is_empty() || c.vptr_offset.is_some() || c.is_union || c.fields.len() < 2 || c.fields.len() > 4 {
            continue;
        }
        let ok = c.fields.iter().enumerate().all(|(k, f)| f.offset == 4 * k as u32 && matches!(strip_cv(&f.ty), Type::Float { size: 4 }));
        if ok && c.size == 4 * c.fields.len() as u32 {
            out.insert(name.clone(), c.fields.len());
        }
    }
    out
}

pub fn regroup(ir: &mut IrFunction, db: Option<&TypeDb>) {
    let Some(db) = db else { return };
    let classes = float_classes(db);
    // candidate groups: the arguments of `CVector3f(x, y, z)`-style constructions
    let mut trios: Vec<(String, Vec<VarId>)> = vec![];
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::Construct { class: Type::Named(c), args, .. } = e {
            if classes.get(c) == Some(&args.len()) {
                let vs: Vec<VarId> = args.iter().filter_map(|a| if let Expr::Var(v) = a { Some(*v) } else { None }).collect();
                if vs.len() == args.len() {
                    trios.push((c.clone(), vs));
                }
            }
        }
    });
    for (cls, trio) in trios {
        let trio = trio.as_slice();
        let set: HashSet<VarId> = trio.iter().copied().collect();
        if set.len() != trio.len() || trio.iter().any(|v| !matches!(ir.vars[*v].kind, VarKind::Local) || !matches!(ir.vars[*v].ty, Type::Float { size: 4 })) {
            continue;
        }
        // every write is a group; every read is the construction
        let mut writes = 0usize;
        let mut groups = 0usize;
        let mut ok = true;
        fn scan(b: &[Stmt], trio: &[VarId], set: &HashSet<VarId>, writes: &mut usize, groups: &mut usize, ok: &mut bool) {
            let mut i = 0;
            while i < b.len() {
                if group_at(b, i, trio).is_some() {
                    *groups += 1;
                    *writes += trio.len();
                    i += trio.len();
                    continue;
                }
                match &b[i] {
                    Stmt::Assign { dst: Expr::Var(v), .. } if set.contains(v) => *ok = false,
                    Stmt::If { then, els, .. } => {
                        scan(then, trio, set, writes, groups, ok);
                        scan(els, trio, set, writes, groups, ok);
                    }
                    Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => scan(body, trio, set, writes, groups, ok),
                    Stmt::For { init, step, body, .. } => {
                        scan(init, trio, set, writes, groups, ok);
                        scan(step, trio, set, writes, groups, ok);
                        scan(body, trio, set, writes, groups, ok);
                    }
                    Stmt::Switch { cases, .. } => {
                        for c in cases {
                            scan(&c.body, trio, set, writes, groups, ok);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        }
        scan(&ir.body, trio, &set, &mut writes, &mut groups, &mut ok);
        if !ok || groups < 2 {
            continue;
        }
        // reads: only inside the construction (3 reads) besides the group sources (none)
        let mut reads: HashMap<VarId, usize> = HashMap::new();
        Stmt::walk_exprs(&ir.body, &mut |e| {
            if let Expr::Var(v) = e {
                if set.contains(v) {
                    *reads.entry(*v).or_default() += 1;
                }
            }
        });
        // walk_exprs also visits assignment destinations: each var is written `groups` times
        if trio.iter().any(|v| reads.get(v).copied().unwrap_or(0) != groups + 1) {
            continue;
        }
        let ty = Type::Named(cls.clone());
        let r = ir.vars.len();
        ir.vars.push(Var { name: format!("{}_obj", ir.vars[trio[0]].name), ty: ty.clone(), kind: VarKind::Local });
        // rewrite groups
        Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
            let mut i = 0;
            while i < b.len() {
                if let Some((base, off)) = group_at(b, i, trio) {
                    let obj = object_at(&base, off, &ty);
                    b[i] = Stmt::Assign { dst: Expr::Var(r), src: obj };
                    b.drain(i + 1..i + trio.len());
                }
                i += 1;
            }
        });
        // `w = f(); r = w;` (w used nowhere else) -> `r = f();`
        let mut mentions: HashMap<VarId, usize> = HashMap::new();
        Stmt::walk_exprs(&ir.body, &mut |e| {
            if let Expr::Var(v) = e {
                *mentions.entry(*v).or_default() += 1;
            }
        });
        Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
            let mut i = 1;
            while i < b.len() {
                let merged = match (&b[i - 1], &b[i]) {
                    (Stmt::Assign { dst: Expr::Var(w), src: call }, Stmt::Assign { dst: Expr::Var(x), src: Expr::Var(w2) })
                        if *x == r && w == w2 && call.has_call() && mentions.get(w) == Some(&2) =>
                    {
                        Some(call.clone())
                    }
                    _ => None,
                };
                if let Some(call) = merged {
                    b[i] = Stmt::Assign { dst: Expr::Var(r), src: call };
                    b.remove(i - 1);
                    continue;
                }
                i += 1;
            }
        });
        // the construction is the object
        Stmt::rewrite_exprs(&mut ir.body, &mut |e| {
            if let Expr::Construct { class: Type::Named(c), args, .. } = e {
                if *c == cls && args.len() == trio.len() && args.iter().zip(trio.iter()).all(|(a, v)| matches!(a, Expr::Var(x) if x == v)) {
                    *e = Expr::Var(r);
                }
            }
        });
    }
}
