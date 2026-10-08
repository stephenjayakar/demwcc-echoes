//! A whole global object copied word by word to memory behind a pointer
//! (`*(p + 0) = g.@0; *(p + 4) = g.@4; ...` over the object's full size) was one struct
//! assignment in the source (`*(CQuaternion*)p = CQuaternion::sNoRotation;`; draft variant
//! [`crate::variants::GLOBAL_STRUCT_COPY`]).

use crate::ir::*;
use crate::types;
use mwdec_core::{Type, TypeDb};
use std::collections::HashMap;

/// `g.@o` -> (symbol, type of g, o)
fn global_part(e: &Expr) -> Option<(&str, &Type, i32)> {
    match e {
        Expr::Member { base, offset, .. } => match &**base {
            Expr::Global { symbol, ty } => Some((symbol, ty, *offset)),
            _ => None,
        },
        Expr::Global { symbol, ty } => Some((symbol, ty, 0)),
        _ => None,
    }
}

fn copy_run(b: &[Stmt], start: usize, vars: &[Var], db: &TypeDb) -> Option<(usize, VarId, String, Type, Vec<VarId>)> {
    let mut temps: HashMap<VarId, (String, i32)> = HashMap::new();
    let mut covered: Vec<(i32, u32)> = vec![];
    let (mut ptr, mut sym, mut gty): (Option<VarId>, Option<String>, Option<Type>) = (None, None, None);
    let mut used_temps = vec![];
    let mut end = start;
    for (k, st) in b.iter().enumerate().skip(start) {
        match st {
            Stmt::Assign { dst: Expr::Var(t), src } if global_part(src).is_some() && vars[*t].kind == VarKind::Local => {
                let (s, ty, o) = global_part(src)?;
                if sym.as_deref().is_some_and(|x| x != s) {
                    break;
                }
                sym = Some(s.to_string());
                gty = Some(ty.clone());
                temps.insert(*t, (s.to_string(), o));
            }
            Stmt::Assign { dst: Expr::Load { base, offset, ty }, src } => {
                let Expr::Var(p) = &**base else { break };
                let (s, o) = match src {
                    Expr::Var(t) => match temps.get(t) {
                        Some((s, o)) => {
                            used_temps.push(*t);
                            (s.clone(), *o)
                        }
                        None => break,
                    },
                    e => match global_part(e) {
                        Some((s, gt, o)) => {
                            gty.get_or_insert(gt.clone());
                            (s.to_string(), o)
                        }
                        None => break,
                    },
                };
                if o != *offset || ptr.is_some_and(|x| x != *p) || sym.as_deref().is_some_and(|x| x != s) {
                    break;
                }
                ptr = Some(*p);
                sym = Some(s);
                covered.push((o, scalar_size(ty)?));
            }
            _ => break,
        }
        end = k + 1;
    }
    let (p, s, t) = (ptr?, sym?, strip_cv(&gty?).clone());
    let size = types::size_of(Some(db), &t)?;
    if !types::is_aggregate(Some(db), &t) || covered.len() < 2 {
        return None;
    }
    covered.sort();
    let mut at = 0u32;
    for (o, sz) in &covered {
        if *o as u32 != at {
            return None;
        }
        at += sz;
    }
    (at == size).then_some((end, p, s, t, used_temps))
}

pub fn global_struct_copies(body: &mut Vec<Stmt>, vars: &[Var], db: &TypeDb) {
    // find one run first (ask the variant only when there is one)
    let mut found = false;
    Stmt::for_each_block_mut(body, &mut |b| {
        for i in 0..b.len() {
            if !found && copy_run(b, i, vars, db).is_some() {
                found = true;
            }
        }
    });
    if !found || !crate::variants::alt(crate::variants::GLOBAL_STRUCT_COPY) {
        return;
    }
    Stmt::for_each_block_mut(body, &mut |b| {
        let mut i = 0;
        while i < b.len() {
            if let Some((end, p, s, t, _)) = copy_run(b, i, vars, db) {
                let st = Stmt::Assign { dst: Expr::Load { base: Box::new(Expr::Var(p)), offset: 0, ty: t.clone() }, src: Expr::Global { symbol: s, ty: t } };
                b.splice(i..end, [st]);
            }
            i += 1;
        }
    });
}
