//! mwdec-lift: PowerPC (Gekko, MWCC GC/2.7 output) -> structured IR.
//!
//! Pipeline (see DESIGN.md):
//! 1. decode (`ppc750cl`) + relocations -> [`cfg::Cfg`] (incl. jump tables via data-symbol relocs)
//! 2. [`frame`]: prologue/epilogue recognition (stwu/mflr/stmw/_savegpr/stfd+psq_st saves)
//! 3. [`translate`]: reaching definitions, register webs (phi variables), symbolic execution of
//!    every block into SSA-like temporaries, EABI calls (REL24 + cwdemangle signatures), virtual
//!    calls, literals from the target object's data, stack slots and conversion scratch
//! 4. [`inline`]: fold single-use temps (hazard-checked), dead temp elimination
//! 5. [`structure`]: if/else (&&/|| chains), loops, switches, gotos as fallback
//! 6. [`simplify`]: expression clean-up, `for` loops, naming
//!
//! The resulting [`ir::IrFunction`] is plain owned data the search stage can rewrite.

pub mod aggregates;
pub mod arrays;
pub mod bitfields;
pub mod byval;
pub mod cfg;
pub mod construct;
pub mod ctrloop;
pub mod divmagic;
pub mod debug;
pub mod frame;
pub mod idioms;
pub mod indexing;
pub mod inline;
pub mod localtypes;
pub mod insn;
pub mod ir;
pub mod sig;
pub mod scalars;
pub mod simplify;
pub mod structure;
pub mod switchtree;
pub mod translate;
pub mod varargs;
pub mod types;
pub mod unroll;
pub mod wide;

pub use ir::*;
pub use mwdec_core::{Function, ObjectFile, TypeDb};

use std::collections::HashMap;
use translate::Lifter;

/// Options for the late, optional passes (all on by default).
#[derive(Clone, Debug)]
pub struct LiftOptions {
    pub inline_temps: bool,
    pub for_loops: bool,
}

impl Default for LiftOptions {
    fn default() -> Self {
        LiftOptions { inline_temps: true, for_loops: true }
    }
}

/// Lift one function of a target object to IR.
pub fn lift_function(obj: &ObjectFile, f: &Function, db: Option<&TypeDb>) -> anyhow::Result<IrFunction> {
    lift_function_with(obj, f, db, &LiftOptions::default())
}

pub fn lift_function_with(obj: &ObjectFile, f: &Function, db: Option<&TypeDb>, opts: &LiftOptions) -> anyhow::Result<IrFunction> {
    let ir = lift_once(obj, f, db, opts, Some(false))?;
    if idioms::constructs_into_param0(&ir) {
        // the "first parameter" is really the hidden struct-return pointer
        if let Ok(ir2) = lift_once(obj, f, db, opts, Some(true)) {
            return Ok(ir2);
        }
    }
    // a guessed struct return whose class couldn't be found: no declarable return type
    if ir.vars.iter().any(|v| v.kind == VarKind::StructRet && matches!(pointee(&v.ty), Some(mwdec_core::Type::Unknown { .. }))) {
        if let Ok(ir2) = lift_once(obj, f, db, opts, None) {
            return Ok(ir2);
        }
    }
    Ok(ir)
}

/// `force_sret`: Some(true) = r3 is the struct-return pointer, Some(false) = guess, None = never.
fn lift_once(obj: &ObjectFile, f: &Function, db: Option<&TypeDb>, opts: &LiftOptions, force_sret: Option<bool>) -> anyhow::Result<IrFunction> {
    if f.code.is_empty() {
        anyhow::bail!("empty function {}", f.name);
    }
    let mut l = Lifter::new(obj, f, db);
    l.force_sret = force_sret == Some(true);
    l.no_sret_guess = force_sret.is_none();
    l.run()?;
    let nb = l.cfg.blocks.len();

    // Terminator expressions join their block's statement list for folding.
    let mut lists: Vec<(Vec<Stmt>, u8)> = Vec::with_capacity(nb);
    for b in 0..nb {
        let bo = &mut l.blocks_out[b];
        let mut items = std::mem::take(&mut bo.stmts);
        let mut mask = 0u8;
        if let Some(e) = bo.switch.take() {
            items.push(Stmt::Switch { e, cases: vec![] });
            mask |= 1;
        }
        if let Some(c) = bo.cond.take() {
            items.push(Stmt::Expr(c));
            mask |= 2;
        }
        if let Some(r) = bo.ret.take() {
            items.push(Stmt::Return(Some(r)));
            mask |= 4;
        }
        lists.push((items, mask));
    }
    let count_all = |lists: &Vec<(Vec<Stmt>, u8)>| {
        let mut uses: HashMap<VarId, usize> = HashMap::new();
        for (items, _) in lists {
            inline::count_uses(items, &mut uses);
        }
        uses
    };
    if std::env::var_os("MWDEC_DUMP").is_some() {
        for (b, (items, _)) in lists.iter().enumerate() {
            debug::stage(&format!("block {b} (translated)"), items, &l.vars);
        }
    }
    // fold, drop dead temps, and fold again: a dead read (the vtable load of a virtual call)
    // can keep a temp at two uses in the first round
    for round in 0..2 {
        if opts.inline_temps {
            let mut uses = count_all(&lists);
            for (items, mask) in lists.iter_mut() {
                inline::inline_list(items, &mut uses, &l.is_temp, &l.vars, mask.count_ones() as usize);
            }
        }
        let mut any = false;
        loop {
            let uses = count_all(&lists);
            let mut changed = false;
            for (items, _) in lists.iter_mut() {
                changed |= inline::dce(items, &uses, &l.is_temp);
            }
            if !changed {
                break;
            }
            any = true;
        }
        if !any || round == 1 {
            break;
        }
    }
    // `i++ < n` in a branch condition (loop tests chained with &&)
    {
        let uses = count_all(&lists);
        for (items, mask) in lists.iter_mut() {
            if *mask & 2 != 0 {
                simplify::form_incdec_with(items, &l.vars, &l.is_temp, db, &uses);
            }
        }
    }
    for (b, (mut items, mask)) in lists.into_iter().enumerate() {
        let bo = &mut l.blocks_out[b];
        if mask & 4 != 0 {
            if let Some(Stmt::Return(Some(r))) = items.pop() {
                bo.ret = Some(r);
            }
        }
        if mask & 2 != 0 {
            if let Some(Stmt::Expr(c)) = items.pop() {
                bo.cond = Some(c);
            }
        }
        if mask & 1 != 0 {
            if let Some(Stmt::Switch { e, .. }) = items.pop() {
                bo.switch = Some(e);
            }
        }
        bo.stmts = items;
    }

    rematerialize_loop_headers(&mut l);
    early_returns(&mut l);
    let ret_void = matches!(strip_cv(&l.ret_ty), mwdec_core::Type::Void);
    let mut body = {
        let vars = l.vars.clone();
        let s = structure::Structurer::new(&l.cfg, &mut l.blocks_out, &vars, ret_void).with_insns(&l.insns);
        s.run()
    };
    debug::stage("structure", &body, &l.vars);
    simplify::fold_logical_values(&mut body, &l.vars);
    ctrloop::forward_constant_copies(&mut body, &l.is_temp);
    ctrloop::propagate_constant_temps(&mut body, &l.vars, &l.is_temp);
    simplify::refine_bool_vars(&body, &mut l.vars);
    simplify::simplify_body(&mut body, &l.vars);
    simplify::fold_virtual_delete_checks(&mut body);
    simplify::fold_return_values(&mut body, &l.vars);
    wide::merge_halves(&mut body, &l.vars, &l.is_temp);
    // compiler-made stack copies the source never names, then fold the temps they kept alive
    idioms::drop_dead_stack_stores(&mut body, &l.vars);
    localtypes::fold_delete_checks(&mut body);
    if l.sig.variadic {
        varargs::recover(&mut body, &mut l.vars, l.params.last().copied());
    }
    divmagic::fold(&mut body, &l.vars, &l.is_temp);
    debug::stage("simplify+dead stack", &body, &l.vars);
    if opts.inline_temps {
        reinline(&mut body, &l.is_temp, &l.vars);
    }
    debug::stage("reinline", &body, &l.vars);
    if let Some(db) = db {
        let is_temp = l.is_temp.clone();
        aggregates::merge_copies_typing(&mut body, &mut l.vars, &|v| is_temp.get(v).copied().unwrap_or(false), db);
        arrays::container_members(&mut body, &l.vars, db);
        bitfields::recover(&mut body, &l.vars, db);
        byval::forward(&mut body, &mut l.vars, db);
        aggregates::literal_inits(&mut body, &l.vars, db, obj);
        byval::forward_ptmf_args(&mut body, &l.vars, db);
        construct::fold(&mut body, &l.vars, db);
        arrays::recover(&mut body, &l.vars, Some(db));
        localtypes::refresh_access_types(&mut body, &l.vars, db);
        localtypes::retype(&body, &mut l.vars);
        localtypes::refresh_access_types(&mut body, &l.vars, db);
        simplify::simplify_body(&mut body, &l.vars);
    }
    localtypes::narrow(&mut body, &mut l.vars, db);
    localtypes::global_types(&mut body, &l.vars, &l.ret_ty, db);
    localtypes::drop_redundant_masks(&mut body, &l.vars, db);
    debug::stage("aggregates/bitfields", &body, &l.vars);
    ctrloop::recover(&mut body, &mut l.vars, &mut l.is_temp);
    ctrloop::recover_shape_b(&mut body, &l.vars, &l.is_temp);
    ctrloop::rematerialize_global_temps(&mut body, &l.is_temp);
    debug::stage("ctrloop", &body, &l.vars);
    simplify::recover_ctr_loops(&mut body, &mut l.vars, &mut l.is_temp);
    unroll::reroll(&mut body);
    if opts.inline_temps {
        reinline(&mut body, &l.is_temp, &l.vars);
    }
    indexing::undo_strength_reduction(&mut body, &l.vars);
    indexing::recover(&mut body, &l.vars, db);
    arrays::type_indexed_globals(&mut body, &l.vars, db, &Default::default());
    simplify::form_incdec(&mut body, &l.vars, &l.is_temp, db);
    simplify::fold_ternary_constants(&mut body);
    simplify::inline_ternary_results(&mut body);
    if ret_void {
        simplify::drop_trailing_return(&mut body);
    }

    let mut sig = l.sig.clone();
    sig.ret = l.ret_ty.clone();
    let mut ir = IrFunction {
        symbol: f.name.clone(),
        sig,
        vars: l.vars,
        params: l.params,
        this_var: l.this_var,
        body,
        init_list: vec![],
        globals: l.globals.into_values().collect(),
        frame: l.frame.info,
        warnings: l.warnings,
        decl_params: l.decl_params,
    };
    debug::stage("late simplify", &ir.body, &ir.vars);
    idioms::apply(&mut ir, db);
    scalars::regroup(&mut ir, db);
    debug::stage("idioms", &ir.body, &ir.vars);
    if opts.for_loops {
        simplify::form_for_loops(&mut ir.body);
    }
    simplify::name_vars(&ir.body, &mut ir.vars);
    Ok(ir)
}

/// A loop header that only loads a value tested by its condition and reused at the top of the
/// body (`while (*p) { (*p)(); ... }`, the compiler CSEs the two reads) gets the read duplicated
/// into its uses, so the header is a plain condition and structures as `while`.
fn rematerialize_loop_headers(l: &mut Lifter) {
    for lp in l.cfg.loops() {
        let h = lp.header;
        let Some((t, f)) = (match l.cfg.blocks[h].term {
            cfg::Term::Cond { taken, fall } if taken != fall => Some((taken, fall)),
            _ => None,
        }) else {
            continue;
        };
        let (body, out) = if lp.body.contains(&t) && !lp.body.contains(&f) {
            (t, f)
        } else if lp.body.contains(&f) && !lp.body.contains(&t) {
            (f, t)
        } else {
            continue;
        };
        if l.cfg.blocks[body].preds.len() != 1 {
            continue;
        }
        // the block the test falls out to may reuse the value too (`n` re-read after the loop)
        let out_ok = out < l.cfg.blocks.len() && l.cfg.blocks[out].preds.len() == 1;
        let stmts = l.blocks_out[h].stmts.clone();
        if stmts.is_empty() {
            continue;
        }
        // every header statement: temp = pure read
        let mut defs = vec![];
        let ok = stmts.iter().all(|s| match s {
            Stmt::Assign { dst: Expr::Var(v), src } if l.is_temp.get(*v).copied().unwrap_or(false) && !src.has_call() => {
                defs.push((*v, src.clone()));
                true
            }
            _ => false,
        });
        if !ok {
            continue;
        }
        // other uses must be in the body block before any store/call
        let mut uses: HashMap<VarId, usize> = HashMap::new();
        for b in 0..l.blocks_out.len() {
            let bo = &l.blocks_out[b];
            let mut items: Vec<Stmt> = bo.stmts.clone();
            if let Some(c) = &bo.cond {
                items.push(Stmt::Expr(c.clone()));
            }
            if let Some(r) = &bo.ret {
                items.push(Stmt::Return(Some(r.clone())));
            }
            if let Some(sw) = &bo.switch {
                items.push(Stmt::Expr(sw.clone()));
            }
            inline::count_uses(&items, &mut uses);
        }
        let in_cond = |v: VarId, l: &Lifter| l.blocks_out[h].cond.as_ref().map_or(0, |c| {
            let mut n = 0;
            c.walk(&mut |e| if matches!(e, Expr::Var(x) if *x == v) { n += 1 });
            n
        });
        // reads of v in a block's statements up to (and including) the first effectful one, plus
        // its branch condition when no statement had effects
        let leading_uses = |blk: usize, v: VarId, l: &Lifter| -> usize {
            let mut seen = 0;
            let mut clean = true;
            for s in &l.blocks_out[blk].stmts {
                let mut n = 0;
                Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| if matches!(e, Expr::Var(x) if *x == v) { n += 1 });
                seen += n;
                let effectful = match s {
                    Stmt::Assign { dst, src } => !matches!(dst, Expr::Var(_)) || src.has_call(),
                    _ => true,
                };
                if effectful {
                    clean = false;
                    break;
                }
            }
            if clean {
                if let Some(c) = &l.blocks_out[blk].cond {
                    c.walk(&mut |e| if matches!(e, Expr::Var(x) if *x == v) { seen += 1 });
                }
            }
            seen
        };
        let mut body_ok = true;
        let mut use_out = false;
        for (v, _) in &defs {
            let mut seen = in_cond(*v, l);
            seen += leading_uses(body, *v, l);
            let total = uses.get(v).copied().unwrap_or(0);
            if seen != total && out_ok {
                let o = leading_uses(out, *v, l);
                if o > 0 {
                    use_out = true;
                }
                seen += o;
            }
            if seen != total {
                body_ok = false;
            }
        }
        if !body_ok {
            continue;
        }
        for (v, e) in &defs {
            let sub = |x: &mut Expr| {
                if matches!(x, Expr::Var(y) if *y == *v) {
                    *x = e.clone();
                }
            };
            if let Some(c) = l.blocks_out[h].cond.as_mut() {
                c.rewrite(&mut { sub });
            }
            Stmt::rewrite_exprs(&mut l.blocks_out[body].stmts, &mut { sub });
            if let Some(c) = l.blocks_out[body].cond.as_mut() {
                c.rewrite(&mut { sub });
            }
            if use_out {
                Stmt::rewrite_exprs(&mut l.blocks_out[out].stmts, &mut { sub });
                if let Some(c) = l.blocks_out[out].cond.as_mut() {
                    c.rewrite(&mut { sub });
                }
            }
        }
        l.blocks_out[h].stmts.clear();
    }
}

/// `if (a) { if (b) return f(); } return false;`: MWCC lays out the shared tail (`li r3,0`) once,
/// right before the return block, and the early return jumps over it (`b epilogue`). Structured
/// as one if/else the tail would be duplicated into every arm, so such jumps become returns.
fn early_returns(l: &mut Lifter) {
    // split return blocks: every predecessor returns its own value
    let mut splits: Vec<usize> = l.split_returns.iter().copied().collect();
    splits.sort();
    for r in splits {
        let preds = l.cfg.blocks[r].preds.clone();
        // the predecessor that falls into the return block is the function's natural end
        let tail = preds.iter().copied().find(|&p| matches!(l.cfg.blocks[p].term, cfg::Term::Fall(_)));
        // a tail shared by several tests (`return false;` laid out once) keeps falling into the
        // return block, which returns its value
        let shared_tail = tail.filter(|&t| l.cfg.blocks[t].preds.len() >= 2);
        if let Some(t) = shared_tail {
            l.blocks_out[r].ret = l.blocks_out[t].ret.take();
        }
        for p in preds {
            if Some(p) == shared_tail {
                continue;
            }
            let cont = match l.cfg.blocks[p].preds.as_slice() {
                [c] if Some(p) != tail => match l.cfg.blocks[*c].term {
                    cfg::Term::Cond { taken, fall } if fall == p && taken != p => Some(taken),
                    cfg::Term::Cond { taken, fall } if taken == p && fall != p => Some(fall),
                    _ => tail,
                },
                _ if Some(p) != tail => tail,
                _ => None,
            };
            match cont {
                Some(c) => make_return_at(l, p, c),
                None => l.cfg.make_return_plain(p),
            }
        }
    }
    loop_returns(l);
    let nb = l.cfg.blocks.len();
    for r in 0..nb {
        if !matches!(l.cfg.blocks[r].term, cfg::Term::Return) || !l.blocks_out[r].stmts.is_empty() {
            continue;
        }
        // `if (c) return x;` with x already in place: a branch over an empty block that only
        // jumps to the return block (`bge skip; b epilogue`). Not in void functions, where such
        // blocks are mostly switch-tree leaves (`bge case; b end`) and breaks.
        let has_value = l.blocks_out[r].ret.is_some();
        let bare: Vec<usize> = if !has_value { vec![] } else { l.cfg.blocks[r]
            .preds
            .iter()
            .copied()
            .filter(|&p| {
                if !matches!(l.cfg.blocks[p].term, cfg::Term::Jump(t) if t == r) || !l.blocks_out[p].stmts.is_empty() {
                    return false;
                }
                let [c] = l.cfg.blocks[p].preds.as_slice() else { return false };
                let cfg::Term::Cond { taken, fall } = l.cfg.blocks[*c].term else { return false };
                // not the `v = c ? v : e` layout (the other arm assigns once and goes on)
                let ternary_arm = matches!(l.cfg.blocks[taken].term, cfg::Term::Fall(t) | cfg::Term::Jump(t) if t == r) && l.blocks_out[taken].stmts.len() == 1;
                // switch-tree leaves (`bge case; b default`) come from statement-free tests
                let computes = !l.blocks_out[*c].stmts.is_empty();
                fall == p && taken != r && !ternary_arm && computes
            })
            .collect() };
        for p in bare {
            l.blocks_out[p].ret = l.blocks_out[r].ret.clone();
            let cont = match l.cfg.blocks[l.cfg.blocks[p].preds[0]].term {
                cfg::Term::Cond { taken, .. } => taken,
                _ => r,
            };
            make_return_at(l, p, cont);
        }
        // the tail that falls into the return block, shared by several paths
        let Some(tail) = l.cfg.blocks[r].preds.iter().copied().find(|&p| matches!(l.cfg.blocks[p].term, cfg::Term::Fall(t) if t == r)) else {
            continue;
        };
        if l.cfg.blocks[tail].preds.len() < 2 {
            continue;
        }
        // the tail only sets the returned value (`return false;`): in void functions a jump to
        // the epilogue is a `break` or the end of an if/else as often as a return
        let Some(Expr::Var(rv)) = l.blocks_out[r].ret.clone() else { continue };
        let sets_ret = |b: usize, l: &Lifter| matches!(l.blocks_out[b].stmts.as_slice(), [Stmt::Assign { dst: Expr::Var(v), .. }] if *v == rv);
        if !sets_ret(tail, l) {
            continue;
        }
        let jumps: Vec<usize> = l.cfg.blocks[r].preds.iter().copied().filter(|&p| p != tail && matches!(l.cfg.blocks[p].term, cfg::Term::Jump(t) if t == r)).collect();
        for p in jumps {
            l.blocks_out[p].ret = l.blocks_out[r].ret.clone();
            // for structuring, the return continues where its guarding test's other arm goes
            let cont = match l.cfg.blocks[p].preds.as_slice() {
                [c] => match l.cfg.blocks[*c].term {
                    cfg::Term::Cond { taken, fall } if fall == p && taken != p => taken,
                    cfg::Term::Cond { taken, fall } if taken == p && fall != p => fall,
                    _ => tail,
                },
                _ => tail,
            };
            make_return_at(l, p, cont);
        }
    }
}

/// Does every path from `b` return within a small region (blocks already turned into returns,
/// jumps to the epilogue, tests whose arms all return)?
fn returns_only(l: &Lifter, b: usize) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut work = vec![b];
    while let Some(x) = work.pop() {
        if x >= l.cfg.blocks.len() {
            continue;
        }
        if !seen.insert(x) || seen.len() > 12 {
            return false;
        }
        match &l.cfg.blocks[x].term {
            cfg::Term::Return | cfg::Term::TailCall => {}
            cfg::Term::Fall(t) | cfg::Term::Jump(t) => work.push(*t),
            cfg::Term::Cond { taken, fall } => {
                work.push(*taken);
                work.push(*fall);
            }
            _ => return false,
        }
    }
    true
}

/// `make_return(p, cont)`, where a continuation that itself only returns (`if (c) { f(); return
/// true; } else return false;`) moves up to the enclosing guard's other arm, so the code after
/// the whole returning region (a loop latch, the rest of the function) stays the join.
fn make_return_at(l: &mut Lifter, p: usize, cont: usize) {
    let mut x = p;
    let mut c = cont;
    for _ in 0..8 {
        if c == p || !returns_only(l, c) {
            break;
        }
        let [g] = l.cfg.blocks[x].preds.as_slice() else { break };
        let g = *g;
        let arms_ok = matches!(l.cfg.blocks[g].term, cfg::Term::Cond { taken, fall } if (taken == x && fall == c) || (fall == x && taken == c));
        if !arms_ok {
            break;
        }
        let [gg] = l.cfg.blocks[g].preds.as_slice() else { break };
        let gg = *gg;
        let other = match l.cfg.blocks[gg].term {
            cfg::Term::Cond { taken, fall } if fall == g && taken != g => taken,
            cfg::Term::Cond { taken, fall } if taken == g && fall != g => fall,
            _ => break,
        };
        x = g;
        c = other;
    }
    if c != cont && returns_only(l, c) {
        c = cont;
    }
    l.cfg.make_return(p, c);
}

/// A jump from inside a loop straight to the epilogue (`for (...) { if (f(x)) { ...; return; } }`)
/// is a `return`, unless the loop's own tests also leave to the epilogue (then nothing follows
/// the loop and it may be a `break`). As a return it continues (for post-dominance) at its
/// guard's other arm, so the loop's latch stays one shared block instead of a copy per arm.
fn loop_returns(l: &mut Lifter) {
    let loops = l.cfg.loops();
    let nb = l.cfg.blocks.len();
    let mut todo: Vec<(usize, usize, usize)> = vec![];
    for p in 0..nb {
        let cfg::Term::Jump(r) = l.cfg.blocks[p].term else { continue };
        if !matches!(l.cfg.blocks[r].term, cfg::Term::Return) || !l.blocks_out[r].stmts.is_empty() {
            continue;
        }
        // the outermost loop holding p or its guard (an exit block of the loop)
        let in_loop = |lp: &cfg::Loop| lp.body.contains(&p) || l.cfg.blocks[p].preds.iter().any(|q| lp.body.contains(q));
        let Some(lp) = loops.iter().filter(|lp| in_loop(lp)).max_by_key(|lp| lp.body.len()) else { continue };
        if lp.body.contains(&r) {
            continue;
        }
        let tests_leave_to_r = lp.body.iter().any(|&b| matches!(l.cfg.blocks[b].term, cfg::Term::Cond { taken, fall } if taken == r || fall == r));
        if tests_leave_to_r {
            continue;
        }
        let cont = match l.cfg.blocks[p].preds.as_slice() {
            // the loop test's own exit (header or latch) is the end of the loop, not a return
            [c] if *c == lp.header || lp.latches.contains(c) => continue,
            [c] => match l.cfg.blocks[*c].term {
                cfg::Term::Cond { taken, fall } if fall == p && taken != p => taken,
                cfg::Term::Cond { taken, fall } if taken == p && fall != p => fall,
                _ => continue,
            },
            _ => continue,
        };
        todo.push((p, r, cont));
    }
    for (p, r, cont) in todo {
        l.blocks_out[p].ret = l.blocks_out[r].ret.clone();
        make_return_at(l, p, cont);
    }
}

/// Temp folding + dead temp elimination over a structured body (each statement list on its own;
/// compound statements are barriers).
pub fn reinline(body: &mut Vec<Stmt>, is_temp: &[bool], vars: &[Var]) {
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    inline::count_uses(body, &mut uses);
    Stmt::for_each_block_mut(body, &mut |b| inline::inline_list(b, &mut uses, is_temp, vars, 0));
    loop {
        let mut uses: HashMap<VarId, usize> = HashMap::new();
        inline::count_uses(body, &mut uses);
        let mut changed = false;
        Stmt::for_each_block_mut(body, &mut |b| changed |= inline::dce(b, &uses, is_temp));
        if !changed {
            break;
        }
    }
}
