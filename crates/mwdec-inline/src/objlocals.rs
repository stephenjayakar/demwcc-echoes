//! Named object locals: register temps that hold the components of one object value and are
//! used more than once (`dx = c.x - p.x; dy = ...; a = dx*ax + ...; b = dx*bx + ...`) were a
//! named local in the source (`CVector3f d = c - p;`, scalarised by the compiler). They become
//! one class-typed local, so the inlines reading it (`CVector3f::Dot(a, d)`) see an lvalue.

use crate::matcher::{make_call, Env, Index, M};
use crate::template::Shape;
use mwdec_core::Type;
use mwdec_lift::{Expr, IrFunction, Stmt, Var, VarId, VarKind};
use std::collections::HashMap;

fn uses(body: &[Stmt]) -> HashMap<VarId, usize> {
    let mut m = HashMap::new();
    Stmt::walk_exprs(body, &mut |e| {
        if let Expr::Var(v) = e {
            *m.entry(*v).or_default() += 1;
        }
    });
    m
}

/// Find one group in list `b`: (statement indices of the defs, temp per component offset, the
/// object value, its class).
fn find_group(b: &[Stmt], env: &Env, idx: &Index, uses: &HashMap<VarId, usize>) -> Option<(Vec<usize>, Vec<(i32, VarId, Type)>, Expr, String)> {
    // candidate temps: single-definition locals, pure value, read at least twice
    let mut cands: Vec<(usize, VarId, &Expr)> = vec![];
    for (i, s) in b.iter().enumerate() {
        if let Stmt::Assign { dst: Expr::Var(v), src } = s {
            if env.defs.contains_key(v) && matches!(env.vars[*v].kind, VarKind::Local) && uses.get(v).copied().unwrap_or(0) >= 3 && !src.has_call() {
                cands.push((i, *v, src));
            }
        }
    }
    if cands.len() < 2 {
        return None;
    }
    let mut best: Option<(i32, Vec<usize>, Vec<(i32, VarId, Type)>, Expr, String)> = None;
    for list in idx.objects.values() {
        for &ti in list {
            let t = &env.lib.templates[ti];
            let Shape::Object { comps, class } = &t.shape else { continue };
            if t.ops < 2 || comps.len() < 2 || comps.len() > cands.len() {
                continue;
            }
            // every assignment of distinct candidates to the components
            fn go(comps: &[crate::template::Comp], cands: &[(usize, VarId, &Expr)], m: &mut M, pick: &mut Vec<usize>, out: &mut Vec<(Vec<usize>, Vec<Option<crate::matcher::Bind>>)>) {
                let k = pick.len();
                if k == comps.len() {
                    out.push((pick.clone(), m.b.clone()));
                    return;
                }
                for (ci, (_, _, src)) in cands.iter().enumerate() {
                    if pick.contains(&ci) || out.len() > 64 {
                        continue;
                    }
                    let snap = m.b.clone();
                    if m.m(&comps[k].pat, src) {
                        pick.push(ci);
                        go(comps, cands, m, pick, out);
                        pick.pop();
                    }
                    m.b = snap;
                }
            }
            let mut m = M::new(env, t);
            let mut sols = vec![];
            go(comps, &cands, &mut m, &mut vec![], &mut sols);
            for (pick, binds) in sols {
                let mut m = M::new(env, t);
                m.b = binds;
                let Some((args, extra)) = m.finalize(0) else { continue };
                let sc = crate::matcher::use_score(t, extra, false, &args);
                if sc < crate::matcher::MIN_SCORE || best.as_ref().is_some_and(|b| b.0 >= sc) {
                    continue;
                }
                let call = make_call(t, args);
                let stmts: Vec<usize> = pick.iter().map(|&ci| cands[ci].0).collect();
                let temps: Vec<(i32, VarId, Type)> = comps.iter().zip(&pick).map(|(c, &ci)| (c.off, cands[ci].1, c.ty.clone())).collect();
                best = Some((sc, stmts, temps, call, class.clone()));
            }
        }
    }
    best.map(|(_, s, t, c, k)| (s, t, c, k))
}

pub fn group(ir: &mut IrFunction, lib: &crate::InlineLib, idx: &Index, db: &mwdec_core::TypeDb) -> usize {
    let mut n = 0;
    for _ in 0..8 {
        let u = uses(&ir.body);
        let raw = crate::matcher::build_defs(&ir.body, &ir.vars);
        let vars = ir.vars.clone();
        let defs = {
            let env0 = Env { db, vars: &vars, defs: &raw, lib, objects: &idx.objects };
            crate::safety::safe_defs(&ir.body, &env0)
        };
        let env = Env { db, vars: &vars, defs: &defs, lib, objects: &idx.objects };
        let mut found: Option<(Vec<usize>, Vec<(i32, VarId, Type)>, Expr, String, usize)> = None;
        let mut path_id = 0usize;
        let mut target_id = usize::MAX;
        Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
            if found.is_none() {
                if let Some((s, t, c, cls)) = find_group(b, &env, idx, &u) {
                    found = Some((s, t, c, cls, path_id));
                    target_id = path_id;
                }
            }
            path_id += 1;
        });
        let Some((stmts, temps, call, cls, _)) = found else { break };
        let id = ir.vars.len();
        ir.vars.push(Var { name: format!("vec{id}"), ty: Type::Named(cls), kind: VarKind::Local });
        let first = *stmts.iter().min().unwrap();
        let mut pid = 0usize;
        Stmt::for_each_block_mut(&mut ir.body, &mut |b| {
            if pid == target_id {
                let mut idxs = stmts.clone();
                idxs.sort_unstable();
                for k in idxs.iter().rev() {
                    b.remove(*k);
                }
                b.insert(first, Stmt::Assign { dst: Expr::Var(id), src: call.clone() });
            }
            pid += 1;
        });
        // every read of a component temp reads the object's member
        Stmt::rewrite_exprs(&mut ir.body, &mut |e| {
            if let Expr::Var(v) = e {
                if let Some((off, _, ty)) = temps.iter().find(|(_, t, _)| t == v) {
                    *e = Expr::Member { base: Box::new(Expr::Var(id)), offset: *off, ty: ty.clone() };
                }
            }
        });
        n += 1;
    }
    n
}
