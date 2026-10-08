//! MWCC fully unrolls small constant-count loops (N <= 8): `for (i = 0; i < 2; i++) a[i] = 1;`
//! becomes `stb 1, 0(a); stb 1, 1(a)`. Rerolled as a draft variant (the stores compile
//! differently from the loop: constant materialisation order, registers).
use crate::ir::*;
use mwdec_core::Type;

/// (index of the first store, run length, rebuilt loop) for a run starting at `k`.
fn run_at(b: &[Stmt], k: usize) -> Option<(usize, Expr, i32, Type, Expr)> {
    let Stmt::Assign { dst, src } = &b[k] else { return None };
    let c = src.as_int()?;
    let (base, off, ty) = match dst {
        Expr::Member { base, offset, ty } if matches!(**base, Expr::Global { .. }) => (Expr::AddrOf(base.clone()), *offset, ty.clone()),
        Expr::Load { base, offset, ty } if matches!(**base, Expr::Var(_) | Expr::AddrOf(_) | Expr::Global { .. }) => ((**base).clone(), *offset, ty.clone()),
        _ => return None,
    };
    let size = scalar_size(&ty).filter(|s| matches!(s, 1 | 2 | 4))? as i32;
    let mut n = 1;
    while k + n < b.len() && n < 8 {
        let Stmt::Assign { dst: d2, src: s2 } = &b[k + n] else { break };
        if s2.as_int() != Some(c) {
            break;
        }
        let same = match d2 {
            Expr::Member { base: b2, offset, ty: t2 } => Expr::AddrOf(b2.clone()) == base && *offset == off + n as i32 * size && *t2 == ty,
            Expr::Load { base: b2, offset, ty: t2 } => **b2 == base && *offset == off + n as i32 * size && *t2 == ty,
            _ => false,
        };
        if !same {
            break;
        }
        n += 1;
    }
    (n >= 2).then(|| (n, base, off, ty, src.clone()))
}

fn has_runs(body: &[Stmt]) -> bool {
    let mut found = false;
    let mut b = body.to_vec();
    Stmt::for_each_block_mut(&mut b, &mut |blk| {
        for k in 0..blk.len() {
            found |= run_at(blk, k).is_some();
        }
    });
    found
}

/// Ask the variant points when a run exists; reroll when one is flipped.
pub fn reroll_const_stores(body: &mut Vec<Stmt>, vars: &mut Vec<Var>, is_temp: &mut Vec<bool>) {
    if !has_runs(body) {
        return;
    }
    let plain = crate::variants::alt(crate::variants::REROLL_CONST_STORES);
    let early = crate::variants::alt(crate::variants::REROLL_CONST_STORES_EARLY);
    if !plain && !early {
        return;
    }
    let mut new_vars: Vec<Var> = vec![];
    let base_id = vars.len();
    Stmt::for_each_block_mut(body, &mut |blk| {
        let mut k = 0;
        while k < blk.len() {
            let Some((n, base, off, ty, c)) = run_at(blk, k) else {
                k += 1;
                continue;
            };
            let i = base_id + new_vars.len();
            new_vars.push(Var { name: "i".into(), ty: t_s32(), kind: VarKind::Local });
            let ptr = Expr::cast(t_ptr(ty.clone()), base.clone());
            let ptr = if off != 0 { Expr::bin(BinOp::Add, ptr, Expr::int(off as i64 / types_size(&ty)), t_ptr(ty.clone())) } else { ptr };
            let lp = Stmt::For {
                init: vec![Stmt::Assign { dst: Expr::Var(i), src: Expr::int(0) }],
                cond: Expr::cmp(BinOp::Lt, Expr::Var(i), Expr::int(n as i64)),
                step: vec![Stmt::Expr(Expr::IncDec { e: Box::new(Expr::Var(i)), delta: 1, post: false })],
                body: vec![Stmt::Assign { dst: Expr::Index { base: Box::new(ptr), index: Box::new(Expr::Var(i)), ty: ty.clone() }, src: c }],
            };
            blk.splice(k..k + n, [lp]);
            let mut at = k;
            if early {
                // move before the constant stores to other objects right before it
                while at > 0 {
                    match &blk[at - 1] {
                        Stmt::Assign { dst, src } if src.as_int().is_some() && !dst.uses_var_any() && !matches!(dst, Expr::Var(_)) && !mentions_base(dst, &base) => at -= 1,
                        _ => break,
                    }
                }
                if at != k {
                    let lp = blk.remove(k);
                    blk.insert(at, lp);
                }
            }
            k = k.max(at) + 1;
        }
    });
    for v in new_vars {
        vars.push(v);
        is_temp.push(false);
    }
}

fn types_size(t: &Type) -> i64 {
    scalar_size(t).unwrap_or(1).max(1) as i64
}

fn mentions_base(dst: &Expr, base: &Expr) -> bool {
    let mut f = false;
    let inner = match base {
        Expr::AddrOf(g) => (**g).clone(),
        e => e.clone(),
    };
    dst.walk(&mut |e| f |= *e == inner || e == base);
    f
}

trait UsesAnyVar {
    fn uses_var_any(&self) -> bool;
}
impl UsesAnyVar for Expr {
    fn uses_var_any(&self) -> bool {
        let mut f = false;
        self.walk(&mut |e| f |= matches!(e, Expr::Var(_)));
        f
    }
}
