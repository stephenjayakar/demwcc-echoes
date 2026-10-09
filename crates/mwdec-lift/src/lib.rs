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
pub mod asmonly;
pub mod bitfields;
pub mod byval;
pub mod cfg;
pub mod construct;
pub mod ctrloop;
pub mod divmagic;
pub mod debug;
pub mod frame;
pub mod frameobj;
pub mod fpcopy;
pub mod fuel;
pub mod idioms;
pub mod indexing;
pub mod inline;
pub mod localtypes;
pub mod objcmp;
pub mod postinline;
pub mod reread;
pub mod reroll;
pub mod insn;
pub mod ir;
pub mod sig;
pub mod scalars;
pub mod sdkframe;
pub mod shapes;
pub mod globalcopy;
pub mod loadorder;
pub mod arglocals;
pub mod selects;
pub mod samereg;
pub mod namedindex;
pub mod simplify;
pub mod helpers;
pub mod structcopy;
pub mod variants;
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

/// Name prefix of locals the draft declares but never uses (frame slots of the SDK compiler);
/// the emitter declares them although nothing refers to them.
pub const UNUSED_LOCAL_PREFIX: &str = "unused";

/// Options for the late, optional passes (all on by default).
#[derive(Clone, Debug)]
pub struct LiftOptions {
    pub inline_temps: bool,
    pub for_loops: bool,
    /// The unit's compiler version from the build configuration (`GC/2.7`, `GC/1.2.5n`, ...),
    /// for idioms only one compiler generation has. `None` = the game compiler.
    pub compiler: Option<String>,
}

impl LiftOptions {
    /// The SDK compiler generation (GC/1.2.x): parameter home slots in the frame.
    pub fn sdk_compiler(&self) -> bool {
        self.compiler.as_deref().is_some_and(|c| c.replace('\\', "/").contains("GC/1.2"))
    }
}

impl Default for LiftOptions {
    fn default() -> Self {
        LiftOptions { inline_temps: true, for_loops: true, compiler: None }
    }
}

/// Lift one function of a target object to IR.
pub fn lift_function(obj: &ObjectFile, f: &Function, db: Option<&TypeDb>) -> anyhow::Result<IrFunction> {
    lift_function_with(obj, f, db, &LiftOptions::default())
}

pub fn lift_function_with(obj: &ObjectFile, f: &Function, db: Option<&TypeDb>, opts: &LiftOptions) -> anyhow::Result<IrFunction> {
    // memos keyed by a TypeDb's address must not outlive it (a later unit's TypeDb can reuse
    // the address): a draft must not depend on what was drafted before it in the process
    sig::reset_memos();
    let ir = lift_once(obj, f, db, opts, Some(false))?;
    if idioms::constructs_into_param0(&ir) {
        // the "first parameter" is really the hidden struct-return pointer
        if let Ok(ir2) = lift_once(obj, f, db, opts, Some(true)) {
            return Ok(ir2);
        }
    }
    // a guessed struct return whose class couldn't be found: no declarable return type
    let mut ir = ir;
    if idioms::standin_sret(&mut ir, db) {
        return Ok(ir);
    }
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
    let fuel_mark = fuel::mark();
    let mut l = Lifter::new(obj, f, db);
    l.force_sret = force_sret == Some(true);
    l.no_sret_guess = force_sret.is_none();
    l.param_home_slots = opts.sdk_compiler();
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
    fpcopy::keep_copy_runs_named(&lists, &l.vars, &mut l.is_temp);
    for round in 0..2 {
        if opts.inline_temps {
            let mut uses = count_all(&lists);
            for (items, mask) in lists.iter_mut() {
                inline::inline_list(items, &mut uses, &l.is_temp, &l.vars, mask.count_ones() as usize);
            }
        }
        let mut any = false;
        let mut fuel = fuel::Fuel::new("lift.dce", fuel::CAP_FIXPOINT);
        while fuel.burn() {
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
        let (body, extra) = s.run_with_vars();
        for v in extra {
            l.vars.push(v);
            l.is_temp.push(false);
        }
        body
    };
    debug::stage("structure", &body, &l.vars);
    structure::volatile_spin_loads(&mut body);
    simplify::fold_logical_values(&mut body, &l.vars);
    ctrloop::forward_constant_copies(&mut body, &l.is_temp);
    ctrloop::propagate_constant_temps(&mut body, &l.vars, &l.is_temp);
    simplify::refine_bool_vars(&body, &mut l.vars);
    simplify::simplify_body(&mut body, &l.vars);
    simplify::fold_virtual_delete_checks(&mut body);
    simplify::fold_return_values(&mut body, &l.vars);
    if sig::demangle(&f.name).is_some() {
        bool_return(&body, &l.vars, &mut l.ret_ty);
    }
    wide::merge_halves(&mut body, &l.vars, &l.is_temp, l.param_home_slots);
    wide::set_adjacent_labels(obj);
    wide::merge_or_assigns(&mut body, &l.vars, &l.is_temp);
    wide::merge_compares(&mut body, &l.vars);
    // compiler-made stack copies the source never names, then fold the temps they kept alive
    let mut dead_stores = idioms::drop_dead_stack_stores_kept(&mut body, &l.vars);
    localtypes::fold_delete_checks(&mut body);
    simplify::fold_double_delete_checks(&mut body);
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
        arrays::absolute_globals(&mut body, &l.vars, db);
        bitfields::recover(&mut body, &l.vars, db);
        byval::forward(&mut body, &mut l.vars, db);
        byval::copy_temporaries(&mut body, &mut dead_stores, db);
        aggregates::literal_inits(&mut body, &l.vars, db, obj);
        byval::forward_ptmf_args(&mut body, &l.vars, db);
        construct::fold(&mut body, &l.vars, db);
        arrays::recover(&mut body, &l.vars, Some(db));
        // fields of array elements: the element accesses exist only now
        bitfields::recover(&mut body, &l.vars, db);
        localtypes::refresh_access_types(&mut body, &l.vars, db);
        localtypes::retype(&body, &mut l.vars);
        localtypes::refresh_access_types(&mut body, &l.vars, db);
        simplify::simplify_body(&mut body, &l.vars);
    }
    construct::fold_returned_temps(&mut body, &l.vars, &mut l.is_temp, &l.ctor_ret_used);
    if l.param_home_slots {
        for g in arrays::synth_hw_arrays(&mut body, &l.vars, db) {
            l.globals.insert(g.symbol.clone(), g);
        }
    }
    localtypes::narrow(&mut body, &mut l.vars, db);
    localtypes::global_types(&mut body, &l.vars, &l.ret_ty, db);
    localtypes::undeclared_returns(&mut body, &l.ret_ty, db);
    localtypes::bool_call_results(&mut body, &l.vars, &l.ret_ty, db);
    localtypes::bit_copies(&mut body, &l.vars);
    localtypes::drop_redundant_masks(&mut body, &l.vars, db);
    localtypes::cast_intrinsic_args(&mut body, &l.vars);
    debug::stage("aggregates/bitfields", &body, &l.vars);
    objcmp::object_compares(&mut body, &l.vars, db);
    unroll::reroll_const(&mut body);
    ctrloop::recover(&mut body, &mut l.vars, &mut l.is_temp);
    ctrloop::recover_shape_b(&mut body, &l.vars, &l.is_temp);
    ctrloop::counted_for(&mut body, &mut l.vars, &mut l.is_temp);
    ctrloop::rematerialize_global_temps(&mut body, &l.vars, &l.is_temp);
    debug::stage("ctrloop", &body, &l.vars);
    simplify::recover_ctr_loops(&mut body, &mut l.vars, &mut l.is_temp);
    structure::offset_ctr_loops(&mut body, &l.vars);
    structure::ctr_break_loops(&mut body, &mut l.vars, &mut l.is_temp);
    unroll::reroll(&mut body);
    if opts.inline_temps {
        reinline(&mut body, &l.is_temp, &l.vars);
    }
    if indexing::pointer_walks(&mut body, &l.vars) > 0 {
        if let Some(db) = db {
            arrays::recover(&mut body, &l.vars, Some(db));
        }
    }
    if indexing::invariant_reads(&mut body, &l.vars, &l.is_temp) > 0 {
        objcmp::object_compares(&mut body, &l.vars, db);
    }
    indexing::undo_strength_reduction(&mut body, &l.vars);
    indexing::recover(&mut body, &l.vars, db);
    indexing::raw_index(&mut body, &l.vars);
    arrays::type_indexed_globals(&mut body, &l.vars, db, &Default::default());
    indexing::pointer_global_rows(&mut body, &l.vars, db);
    simplify::form_incdec(&mut body, &l.vars, &l.is_temp, db);
    simplify::fold_ternary_constants(&mut body);
    simplify::inline_ternary_results(&mut body);
    bitfields::insert_chains(&mut body, &mut l.vars, &mut l.is_temp, l.param_home_slots);
    namedindex::name_scaled_index(&mut body, &mut l.vars, &mut l.is_temp);
    selects::typed_select_args(&mut body);
    arglocals::args_to_locals(&mut body, &mut l.vars, &mut l.is_temp, db);
    loadorder::order_param_loads(&mut body, &l.vars);
    shapes::byte_fields(&mut body);
    shapes::mask_selects(&mut body);
    shapes::const_read_only_externs(&mut body, &mut l.globals);
    if let Some(db) = db {
        globalcopy::global_struct_copies(&mut body, &l.vars, db);
    }
    localtypes::forward_global_pointers(&mut body, &l.vars, &l.is_temp);
    if ret_void {
        simplify::drop_trailing_return(&mut body);
    }
    reread::reread_temps(&mut body, &l.vars);
    reroll::reroll_const_stores(&mut body, &mut l.vars, &mut l.is_temp);
    simplify::drop_garbage_return(&mut body);

    // GC/1.2.5n reserves a frame slot for every declared local once the frame has a local area,
    // used or not (and for the parameters and locals of inlined helpers): a frame larger than
    // this draft's slots make it gets unused locals (functions without stack objects only)
    if l.param_home_slots && l.frame.info.size > 0 && !l.has_stack_objects() {
        let slots = l.sig.params.iter().map(|p| if crate::types::size_of(db, &p.ty).unwrap_or(4) > 4 { 2 } else { 1 }).sum::<u32>();
        if l.sdk_frame_size(slots) < l.frame.info.size {
            if let Some(k) = (1..=6).find(|&k| l.sdk_frame_size(slots + k) == l.frame.info.size) {
                for n in 0..k {
                    l.vars.push(Var { name: format!("{UNUSED_LOCAL_PREFIX}{}", n + 1), ty: mwdec_core::Type::Int { size: 4, signed: true }, kind: VarKind::Local });
                    l.is_temp.push(false);
                }
            }
        }
    }

    // more volatile-register locals in the draft than the target's frame has scalar slots for
    // (each costs this compiler a slot): fold the extra ones into their uses (variant)
    if l.param_home_slots && l.frame.info.size > 0 {
        let np = l.sig.params.iter().map(|p| if crate::types::size_of(db, &p.ty).unwrap_or(4) > 4 { 2 } else { 1 }).sum::<i32>();
        let vol = sdkframe::volatile_locals(&body, &l.vars) as i32;
        let target_slots = match l.lowest_stack_offset() {
            Some(lo) => Some((lo - 8) / 4 - np),
            None => (0..=vol).find(|&k| l.sdk_frame_size((np + k) as u32) == l.frame.info.size),
        };
        if let Some(t) = target_slots.filter(|t| *t >= 0 && vol > *t) {
            let mut probe = body.clone();
            if sdkframe::fold_volatile_locals(&mut probe, &l.vars, (vol - t) as usize) > 0 && variants::alt(variants::SDK_FOLD_SLOT_LOCALS) {
                body = probe;
            }
        }
    }

    {
        let mut probe = body.clone();
        if samereg::merge_register_webs(&mut probe, &mut l.vars) && variants::alt(variants::MERGE_REGISTER_WEBS) {
            body = probe;
        }
    }

    let mut sig = l.sig.clone();
    sig.ret = l.ret_ty.clone();
    let string_pool = l.string_pool_prefix();
    let literal_bytes = l.literal_bytes();
    let mut ir = IrFunction {
        symbol: f.name.clone(),
        sig,
        vars: l.vars,
        params: l.params,
        this_var: l.this_var,
        body,
        init_list: vec![],
        implicit_bases: vec![],
        globals: l.globals.into_values().collect(),
        frame: l.frame.info,
        string_pool,
        literal_bytes,
        warnings: l.warnings,
        decl_params: l.decl_params,
        dead_stores,
    };
    debug::stage("late simplify", &ir.body, &ir.vars);
    frameobj::fold_single_reads(&mut ir);
    frameobj::fold_block_copies(&mut ir, db);
    frameobj::whole_object_copies(&mut ir, db);
    frameobj::unknown_callee_byval(&mut ir, db);
    frameobj::fold_converting_return(&mut ir, db);
    frameobj::drop_default_construction_stores(&mut ir, db);
    localtypes::float_word_copies(&mut ir.body, &ir.vars);
    idioms::apply(&mut ir, db);
    idioms::narrow_float_stores(&mut ir, db);
    scalars::regroup(&mut ir, db);
    debug::stage("idioms", &ir.body, &ir.vars);
    if opts.for_loops {
        simplify::form_for_loops(&mut ir.body);
        simplify::narrow_counter_steps(&mut ir.body, &ir.vars);
    }
    simplify::name_vars(&ir.body, &mut ir.vars);
    structure::guard_not_swap(&mut ir.body);
    for w in structure::invariant_loop_conditions(&ir.body) {
        ir.warnings.push(w);
    }
    ir.warnings.extend(fuel::since(fuel_mark));
    Ok(ir)
}

/// An inferred word return (no declaration) whose every returned value is a truth value
/// (`a && b`, comparisons, bool locals, 0/1 constants) is `bool`: returning it as `int`
/// re-extends the byte (`clrlwi`) where the target just copies it.
fn bool_return(body: &[Stmt], vars: &[ir::Var], ret: &mut mwdec_core::Type) {
    use ir::*;
    if !matches!(ret, mwdec_core::Type::Unknown { size: 4 }) {
        return;
    }
    fn returns<'a>(b: &'a [Stmt], out: &mut Vec<&'a Expr>) {
        for s in b {
            match s {
                Stmt::Return(Some(e)) => out.push(e),
                Stmt::If { then, els, .. } => {
                    returns(then, out);
                    returns(els, out);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => returns(body, out),
                Stmt::Switch { cases, .. } => cases.iter().for_each(|c| returns(&c.body, out)),
                _ => {}
            }
        }
    }
    let mut rs = vec![];
    returns(body, &mut rs);
    let (mut real, mut all) = (false, true);
    for e in &rs {
        let t = types::ty_of(e, vars);
        let is_b = matches!(e, Expr::Binary { op, .. } if op.is_bool()) || matches!(e, Expr::Unary { op: UnOp::Not, .. }) || matches!(strip_cv(&t), mwdec_core::Type::Bool);
        if is_b {
            real = true;
        } else if !matches!(e.as_int(), Some(0 | 1)) {
            all = false;
        }
    }
    if real && all {
        *ret = mwdec_core::Type::Bool;
    }
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
        // variables the reads depend on (`*p`): reassigning one ends the stretch where the read
        // may be repeated
        let mut deps: Vec<VarId> = vec![];
        for (_, e) in &defs {
            e.walk(&mut |x| {
                if let Expr::Var(y) = x {
                    deps.push(*y);
                }
            });
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
                    Stmt::Assign { dst: Expr::Var(x), src } => src.has_call() || deps.contains(x),
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
        // loop blocks dominated by the body's first block through effect-free blocks only (the
        // arms of an `if` at the top of the body re-reading the value)
        let effect_free = |blk: usize, l: &Lifter| {
            l.blocks_out[blk].stmts.iter().all(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), src } if !src.has_call() && !deps.contains(x)) || matches!(s, Stmt::Label(_) | Stmt::Comment(_)))
        };
        let mut deeper: Vec<usize> = vec![];
        for &u in &lp.body {
            if u == body || u == h {
                continue;
            }
            let mut x = l.cfg.idom[u];
            let mut ok = true;
            let mut guard = 0;
            while x != body {
                if x == usize::MAX || x == h || !lp.body.contains(&x) || !effect_free(x, l) || guard > 32 {
                    ok = false;
                    break;
                }
                x = l.cfg.idom[x];
                guard += 1;
            }
            if ok && effect_free(body, l) {
                deeper.push(u);
            }
        }
        let mut body_ok = true;
        let mut use_out = false;
        let mut used_deeper: Vec<usize> = vec![];
        for (v, _) in &defs {
            let mut seen = in_cond(*v, l);
            seen += leading_uses(body, *v, l);
            for &u in &deeper {
                let n = leading_uses(u, *v, l);
                if n > 0 && !used_deeper.contains(&u) {
                    used_deeper.push(u);
                }
                seen += n;
            }
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
            // the body stores before reusing the value (`while ((p = head) != 0) { head = p->next;
            // f(p); }`): the test reads it, the body's first statement reads it again (MWCC
            // CSEs the two reads, nothing runs in between)
            let all_in_loop = defs.iter().all(|(v, _)| {
                let mut n = in_cond(*v, l);
                for &b in &lp.body {
                    if b == h {
                        continue;
                    }
                    let bo = &l.blocks_out[b];
                    let mut items: Vec<Stmt> = bo.stmts.clone();
                    if let Some(c) = &bo.cond {
                        items.push(Stmt::Expr(c.clone()));
                    }
                    if let Some(r) = &bo.ret {
                        items.push(Stmt::Return(Some(r.clone())));
                    }
                    let mut m: HashMap<VarId, usize> = HashMap::new();
                    inline::count_uses(&items, &mut m);
                    n += m.get(v).copied().unwrap_or(0);
                }
                n == uses.get(v).copied().unwrap_or(0) && in_cond(*v, l) > 0
            });
            if !all_in_loop {
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
            }
            let moved = std::mem::take(&mut l.blocks_out[h].stmts);
            let rest = std::mem::take(&mut l.blocks_out[body].stmts);
            l.blocks_out[body].stmts = moved.into_iter().chain(rest).collect();
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
            for &u in &used_deeper {
                Stmt::rewrite_exprs(&mut l.blocks_out[u].stmts, &mut { sub });
                if let Some(c) = l.blocks_out[u].cond.as_mut() {
                    c.rewrite(&mut { sub });
                }
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
                    // jumped to forward over the other arm: `if (c) { .. } else { return x; }`,
                    // the arms meet at the function's last return
                    cfg::Term::Cond { taken, fall } if taken == p && fall != p && tail.is_some() && l.cfg.blocks[p].start > l.cfg.blocks[fall].start && real_arm(l, fall) => tail,
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
        // the value already in place: the block is only the epilogue (its own computations
        // folded into the returned expression don't count)
        let only_epilogue = {
            let blk = &l.cfg.blocks[r];
            (blk.start..blk.end).all(|k| l.frame.skip.contains(&k) || l.insns[k].is_blr())
        };
        let has_value = l.blocks_out[r].ret.is_some() && only_epilogue;
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
        // a tail with one way in: only returns the tests jump forward to over their other arm
        // (`if (p) { ... } else { return false; } return true;`) meet it
        let forward_only = l.cfg.blocks[tail].preds.len() < 2;
        // the tail only sets the returned value (`return false;`): in void functions a jump to
        // the epilogue is a `break` or the end of an if/else as often as a return
        let Some(Expr::Var(rv)) = l.blocks_out[r].ret.clone() else { continue };
        let sets_ret = |b: usize, l: &Lifter| matches!(l.blocks_out[b].stmts.as_slice(), [Stmt::Assign { dst: Expr::Var(v), .. }] if *v == rv);
        if !sets_ret(tail, l) {
            continue;
        }
        let jumps: Vec<usize> = l.cfg.blocks[r].preds.iter().copied().filter(|&p| p != tail && matches!(l.cfg.blocks[p].term, cfg::Term::Jump(t) if t == r)).collect();
        let forward = |p: usize, l: &Lifter| match l.cfg.blocks[p].preds.as_slice() {
            [c] => matches!(l.cfg.blocks[*c].term, cfg::Term::Cond { taken, fall } if taken == p && fall != p && l.cfg.blocks[p].start > l.cfg.blocks[fall].start && real_arm(l, fall)),
            _ => false,
        };
        let jumps: Vec<usize> = jumps.into_iter().filter(|&p| !forward_only || (forward(p, l) && sets_ret(p, l))).collect();
        for p in jumps {
            l.blocks_out[p].ret = l.blocks_out[r].ret.clone();
            // for structuring, the return continues where its guarding test's other arm goes
            let cont = match l.cfg.blocks[p].preds.as_slice() {
                [c] => match l.cfg.blocks[*c].term {
                    cfg::Term::Cond { taken, fall } if fall == p && taken != p => taken,
                    // a returning arm the test jumps forward to, over the other arm: the source
                    // had the other arm first (`if (!c) { B } else { A; return x; }`), the two
                    // meet at the shared tail
                    cfg::Term::Cond { taken, fall } if taken == p && fall != p && l.cfg.blocks[p].start > l.cfg.blocks[fall].start && real_arm(l, fall) => tail,
                    cfg::Term::Cond { taken, fall } if taken == p && fall != p => fall,
                    _ => tail,
                },
                _ => tail,
            };
            make_return_at(l, p, cont);
        }
    }
    shared_return_tail(l);
    cond_returns_to_epilogue(l);
}

/// Void functions: a test branching straight to the epilogue is `if (c) return;` when, as an
/// edge, it would make the epilogue the join of an enclosing test whose arms otherwise meet
/// earlier (`if (a) { if (c) return; x = 1; } f();`, the two ways into `f()` shared).
fn cond_returns_to_epilogue(l: &mut Lifter) {
    if !matches!(strip_cv(&l.ret_ty), mwdec_core::Type::Void) {
        return;
    }
    let nb = l.cfg.blocks.len();
    let ends: Vec<usize> = (0..nb)
        .filter(|&b| matches!(l.cfg.blocks[b].term, cfg::Term::Return) && l.blocks_out[b].stmts.is_empty() && l.cfg.blocks[b].preds.len() >= 2 && !l.cfg.pd_extra.iter().any(|e| e.0 == b))
        .collect();
    let [end] = ends.as_slice() else { return };
    let end = *end;
    let exit = l.cfg.exit();
    let cands: Vec<usize> = l.cfg.blocks[end].preds.iter().copied().filter(|&c| matches!(l.cfg.blocks[c].term, cfg::Term::Cond { taken, fall } if taken == end && fall != end)).collect();
    for c in cands {
        let mut trial = l.cfg.clone();
        trial.make_cond_return(c);
        let opens = (0..nb).any(|x| x != c && l.cfg.idom[x] != usize::MAX && l.cfg.ipdom[x] == end && trial.ipdom[x] != end && trial.ipdom[x] != exit && trial.ipdom[x] != usize::MAX);
        if opens {
            l.cfg = trial;
        }
    }
}

/// Leaf functions return with a `blr` per `return` statement, the function's last return laid
/// out at the end. When that last return block is shared by several paths (`if (p && i < n)
/// return &a[i]; return 0;`), the earlier return blocks are early returns continuing (for
/// structuring) at their guard's other arm, so the shared tail becomes the join instead of a
/// copy in every arm.
fn shared_return_tail(l: &mut Lifter) {
    let nb = l.cfg.blocks.len();
    let Some(last) = (0..nb).filter(|&b| l.cfg.idom[b] != usize::MAX).max_by_key(|&b| l.cfg.blocks[b].start) else { return };
    if !matches!(l.cfg.blocks[last].term, cfg::Term::Return) || l.cfg.blocks[last].preds.len() < 2 || l.cfg.pd_extra.iter().any(|e| e.0 == last) {
        return;
    }
    let mut todo: Vec<(usize, usize)> = vec![];
    for p in 0..nb {
        if p == last || !matches!(l.cfg.blocks[p].term, cfg::Term::Return) || l.cfg.pd_extra.iter().any(|e| e.0 == p) {
            continue;
        }
        let [c] = l.cfg.blocks[p].preds.as_slice() else { continue };
        let cont = match l.cfg.blocks[*c].term {
            cfg::Term::Cond { taken, fall } if fall == p && taken != p => taken,
            cfg::Term::Cond { taken, fall } if taken == p && fall != p => fall,
            _ => continue,
        };
        todo.push((p, cont));
    }
    for (p, cont) in todo {
        make_return_at(l, p, cont);
    }
}

/// An arm with code of its own (not a lone `b x`, as in a compare tree's leaves): a test jumping
/// over it to a return had that arm first in the source.
fn real_arm(l: &Lifter, b: usize) -> bool {
    !(l.blocks_out[b].stmts.is_empty() && matches!(l.cfg.blocks[b].term, cfg::Term::Jump(_)))
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
    // returns inside a loop that are their own blocks (`li r3,1 ; blr` in a leaf): the same,
    // they continue at their guard's other arm
    let loops = l.cfg.loops();
    let mut todo2: Vec<(usize, usize)> = vec![];
    for p in 0..l.cfg.blocks.len() {
        if !matches!(l.cfg.blocks[p].term, cfg::Term::Return) || l.cfg.pd_extra.iter().any(|e| e.0 == p) {
            continue;
        }
        let [c] = l.cfg.blocks[p].preds.as_slice() else { continue };
        let c = *c;
        let Some(lp) = loops.iter().filter(|lp| lp.body.contains(&c)).max_by_key(|lp| lp.body.len()) else { continue };
        // a CTR loop's `bdnz` latch is its test: a return from the header is an early return
        let ctr_latch = lp.latches.iter().any(|&lt| {
            let mut ctr = false;
            if let Some(cond) = &l.blocks_out[lt].cond {
                cond.walk(&mut |e| ctr |= matches!(e, Expr::Var(v) if l.vars[*v].name.starts_with("var_ctr")));
            }
            ctr
        });
        if (c == lp.header && !ctr_latch) || lp.latches.contains(&c) {
            continue;
        }
        let cont = match l.cfg.blocks[c].term {
            cfg::Term::Cond { taken, fall } if fall == p && taken != p => taken,
            cfg::Term::Cond { taken, fall } if taken == p && fall != p => fall,
            _ => continue,
        };
        todo2.push((p, cont));
    }
    for (p, cont) in todo2 {
        make_return_at(l, p, cont);
    }
}

/// Temp folding + dead temp elimination over a structured body (each statement list on its own;
/// compound statements are barriers).
pub fn reinline(body: &mut Vec<Stmt>, is_temp: &[bool], vars: &[Var]) {
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    inline::count_uses(body, &mut uses);
    Stmt::for_each_block_mut(body, &mut |b| inline::inline_list(b, &mut uses, is_temp, vars, 0));
    let mut fuel = fuel::Fuel::new("lift.reinline_dce", fuel::CAP_FIXPOINT);
    while fuel.burn() {
        let mut uses: HashMap<VarId, usize> = HashMap::new();
        inline::count_uses(body, &mut uses);
        let mut changed = false;
        Stmt::for_each_block_mut(body, &mut |b| changed |= inline::dce(b, &uses, is_temp));
        if !changed {
            break;
        }
    }
}
