//! Draft variants: decisions of the lifter and the emitter that the machine code doesn't settle
//! (a setter run or a whole-object copy, a folded inline or the plain expansion, a named local or
//! a temporary, ...). Instead of guessing, a decision site asks [`alt`] whether to take its
//! alternative. The answer is "no" (the default) unless the drafting driver redrafts with that
//! point flipped; every point asked during a draft is recorded, so the driver knows which
//! variants exist for this function and compiles each of them, keeping the best (the compiler
//! settles the ambiguity).
//!
//! Decision site (lift or emit, anything that runs inside [`draft`]):
//!
//! ```ignore
//! if variants::alt(variants::STRUCTCOPY_SETTERS) { /* the alternative */ } else { /* default */ }
//! ```
//!
//! Driver: `let (out, points) = variants::draft(&[], || lift_and_emit());` gives the default
//! draft and the points it asked; `variants::draft(&[p], ...)` drafts with point `p` flipped.
//! Outside [`draft`] every point takes its default and nothing is recorded.
//!
//! Points are plain `&'static str` names; list new ones in [`POINTS`] with a line of
//! documentation. Ask only where the alternative actually differs (a point asked but without
//! effect costs one redraft and is deduplicated before compiling).
use std::cell::RefCell;
use std::collections::BTreeSet;

/// Whole-object copy instead of a run of member setters `v.SetX(o.GetX()); v.SetY(...)...`.
pub const STRUCTCOPY_SETTERS: &str = "structcopy.setters";
/// Keep a temporary built from every member of one object as a construction (no copy).
pub const STRUCTCOPY_NO_TEMP: &str = "structcopy.no_temp";
/// Keep a returned object built in the struct-return storage (no `return x;` / `return T(..);`).
pub const STRUCTCOPY_NO_RETURN: &str = "structcopy.no_return";

/// Pass trailing arguments equal to the declaration's default arguments explicitly (by default
/// they are omitted: MWCC reuses one temporary for an omitted by-value class default across
/// calls, but makes a fresh one for each explicit argument).
pub const EXPLICIT_DEFAULT_ARGS: &str = "emit.explicit_default_args";
/// A returned object filled from one object (through flag checks and early returns, as an
/// `optional_object` copy does) becomes `return x;`.
pub const STRUCTCOPY_RETURN_WHOLE: &str = "structcopy.return_whole";
/// Values read from memory once and kept in a local read again at every use (MWCC CSEs the reads).
/// (named to sort after the other points: the driver tries the first `MAX_VARIANT_POINTS` asked)
pub const REREAD_TEMPS: &str = "temps.reread";
/// An address computed before a branch and used only in its arms spelled again at each use
/// (by default it is a local assigned before the branch).
pub const ARMS_ADDRESS_INLINE: &str = "temps.arms_address_inline";
/// A call result held in a scratch register across another argument's setup folds into its use
/// (by default it is a named local).
pub const HELD_CALL_RESULT_INLINE: &str = "temps.held_call_result_inline";
/// A counted loop entered under a signed `n != 0` guard counts `for (i = 0; i != n; i++)`
/// (by default `for (i = 0; i < (unsigned)n; i++)`).
pub const LOOP_NE_COUNT: &str = "loops.ne_count";
/// `while (1) { if (c) { S; return x; } B }` written `while (!c) { B } S; return x;` (the exit
/// code after the loop).
pub const LOOP_EXIT_AFTER: &str = "loops.exit_after";
/// Runs of one constant stored to consecutive array elements (`a[0] = 1; a[1] = 1;`, MWCC's full
/// unroll of a small constant loop) rerolled into `for (i = 0; i < n; i++) a[i] = 1;`.
pub const REROLL_CONST_STORES: &str = "unroll.const_stores";
/// The same, the loop placed before the constant stores to other objects right before it (the
/// scheduler moved those first).
pub const REROLL_CONST_STORES_EARLY: &str = "unroll.const_stores_early";

/// A value computed once and used to start several register words (`t = id << 1; ra = (t +
/// 224) << 24; bg = (t + 225) << 24;`) is written out at each start, as the SDK does
/// (`(0xE0 + id * 2) << 24`); the compiler's CSE then decides the registers.
pub const SDK_WORD_SHARED_INLINE: &str = "sdk.word_shared_inline";

/// An address temp computed right after an independent value temp moves before it (the source
/// took the object's address first: `T& s = a[i]; u32 f = ...;`).
pub const ORDER_ADDRESS_FIRST: &str = "order.address_first";

/// Right-nested integer sums `a + (b + c)` (the compiler's reassociation of the source sum)
/// written left to right `a + b + c` (`mwdec_lift::assoc`).
pub const ARITH_LEFT_ASSOC: &str = "arith.left_assoc";

/// A guessed struct return whose class nothing names takes the only class of the context whose
/// layout and constructor fit the stores into it (`return T(args);`).
pub const SRET_CLASS_BY_LAYOUT: &str = "sret.class_by_layout";

/// A run of word copies (`*(int*)(d + 4k) = *(int*)(s + 4k)`) becomes 64-bit copies, one per pair.
pub const STRUCTCOPY_WORDS_LL: &str = "structcopy.words_ll";
/// A run of word copies becomes one block copy through a helper struct (`mwdec_lift::helpers`).
pub const STRUCTCOPY_WORDS_BLOCK: &str = "structcopy.words_block";

/// A packed word built from field inserts `v = a | b | c | d;` gets its last field in a statement
/// of its own (`v = a | b | c; v |= d;`: the compiler copies the partial word before the insert).
pub const ORDER_SPLIT_LAST_FIELD: &str = "order.split_last_field";

/// Bool locals defined once become `const bool` (a returned `&&`/`||` chain goes through one): the
/// compiler re-extends (`clrlwi`) a const bool where it is used.
pub const BOOL_CONST_LOCAL: &str = "bool.const_local";

/// Temps read from memory just before a loop and used once inside it are read there (the
/// compiler hoisted the loop-invariant reads: `i < v.size()`, `a[i].id == id`).
pub const LOOP_INVARIANT_READS: &str = "loop.invariant_reads";

/// A float product read by an add or subtract that the compiler didn't fuse (separate fmuls /
/// fadds) stays a local of its own.
pub const FLOAT_UNFUSED_PRODUCTS: &str = "float.unfused_products";

/// A narrow (char / short) destination updated from itself is written as a compound assignment
/// (`v |= x`, not `v = (u8)(v | x)`).
pub const ASSIGN_COMPOUND_NARROW: &str = "assign.compound_narrow";

/// Int locals written only narrowed to one small type (or small constants) take that type, their
/// self-updates written as compound assignments.
pub const LOCALS_NARROW_BY_DEFS: &str = "locals.narrow_by_defs";

/// GC/1.2.5n: draft locals in volatile registers holding a global read are folded into their
/// single use when the target's frame shows no scalar-local slots (`mwdec_lift::sdkframe`).
pub const SDK_FOLD_SLOT_LOCALS: &str = "sdk.fold_slot_locals";

/// A scaled array index (`a[x * 2]`) comes from a named local assigned right before its
/// statement (`int index = x * 2;`), which fixes when the compiler computes it.
pub const INDEX_NAMED_SCALED: &str = "index.named_scaled";

/// A free algorithm's result (`it = rstl::find(...)`) used once by the next condition stays a named
/// local instead of being written into the condition.
pub const NAMED_ALGORITHM_RESULT: &str = "inline.named_algorithm_result";
/// Member stores explained both as `x = x op v` and as `x op= v`: the in-place mutator.
pub const INPLACE_MUTATOR: &str = "inline.inplace_mutator";

/// Successive webs of one callee-saved register (`temp_r31`, `temp_r31_2`) are one variable.
pub const MERGE_REGISTER_WEBS: &str = "regs.merge_webs";

/// A `clrlslwi` (mask ending where the shift starts) is `(x & m) << s` instead of `x << s & M`.
pub const EXPR_MASK_THEN_SHIFT: &str = "expr.mask_then_shift";

/// A condition-selected constant passed to a call becomes `c ? (T)K2 : (T)K1` of the parameter's type.
pub const SELECT_TYPED_ARG: &str = "expr.select_typed_arg";

/// Computed call arguments become named locals assigned in argument order before the call.
pub const ARGS_NAMED_LOCALS: &str = "expr.args_named_locals";

/// The leading run of parameter loads ordered by parameter, then offset (source order).
pub const ORDER_PARAM_LOADS: &str = "order.param_loads";

/// A global object copied word by word behind a pointer becomes one struct assignment.
pub const GLOBAL_STRUCT_COPY: &str = "structcopy.global_whole";

/// Temporaries of a run of floating-point copies between two objects stay named locals (declared
/// by target register), so the compiler's colouring follows the target's register rotation.
pub const NAMED_FP_COPIES: &str = "copies.named_fp";

/// A pointer step after a read the next statement uses (`t = *p; p += 1; f(t);`) is a statement
/// of its own after it (`f(*p); ++p;`) instead of a post-increment inside it (`f(*p++)`).
pub const INCDEC_STEP_AFTER: &str = "incdec.step_after";

/// Byte/halfword fields packed into a word as narrowing conversions (`(uchar)x << 16`), not masks.
pub const EXPR_BYTE_FIELDS: &str = "expr.byte_fields";
/// Externs the function only reads (scalars) are declared `const`.
pub const CONST_READ_ONLY_EXTERNS: &str = "types.const_read_only_externs";

/// `k & ~(c ? -1 : 0)` written as the select `c ? 0 : k`.
pub const EXPR_MASK_SELECT: &str = "expr.mask_select";

/// Read-only pointer parameters of a function without a prototype declared pointer-to-const.
pub const PARAM_CONST_POINTERS: &str = "param.const_pointers";

/// Hardware registers accessed at folded addresses through a constant pointer instead of an
/// array at the address.
pub const HW_CONST_POINTER: &str = "hw.const_pointer";

/// A value updated in place after a copy of its old value was taken: one variable updated, the
/// old value in another (`cpr = x; prev = cpr; cpr &= m; ... return prev;`).
pub const UPDATE_AFTER_COPY: &str = "vars.update_after_copy";

/// (older compiler) Memory reached through a named local copy of a pointer parameter
/// (`__GXFifoObj* realFifo = (__GXFifoObj*)fifo;`).
pub const SDK_PARAM_VIEW: &str = "sdk.param_view";

/// A single in-place update of a callee-saved register is a variable updated in place
/// (`s = x << 16; s |= y << 8;`, `s = x >> 24 & 0xf0; s = g & ~s;`).
pub const ACCUM_SINGLE_UPDATE: &str = "vars.accum_single_update";

/// The values of a run of register-field inserts are named locals computed before the run.
pub const INSERT_VALUES_FIRST: &str = "bitfield.insert_values_first";

/// OS low-memory words (0x80000000..0x80004000) accessed through arrays declared at their
/// addresses (`u32 lomem[256] : 0x80003000;`).
pub const LOWMEM_ARRAYS: &str = "sdk.lowmem_arrays";

/// Registered decision points: (name, what the alternative does).
pub const POINTS: &[(&str, &str)] = &[
    (EXPLICIT_DEFAULT_ARGS, "trailing arguments equal to their declared defaults are passed explicitly"),
    (STRUCTCOPY_SETTERS, "a run of member setters from one object's getters becomes a whole-object copy"),
    (STRUCTCOPY_NO_TEMP, "a temporary built from every member of one object stays a construction"),
    (STRUCTCOPY_NO_RETURN, "a returned object stays built in the struct-return storage"),
    (REROLL_CONST_STORES, "runs of one constant stored to consecutive array elements become a for loop"),
    (REROLL_CONST_STORES_EARLY, "the same, the loop moved before the constant stores right before it"),
    (REREAD_TEMPS, "single-assignment locals of pure memory reads are re-read where they are used"),
    (LOOP_EXIT_AFTER, "a loop whose only exit returns is written with the exit code after it"),
    (LOOP_NE_COUNT, "a counted loop under a signed `n != 0` guard tests `i != n`"),
    (ARMS_ADDRESS_INLINE, "an address computed before a branch and read only in its arms is spelled at each use"),
    (HELD_CALL_RESULT_INLINE, "a call result held in a scratch register across another argument's setup folds into its use"),
    (STRUCTCOPY_WORDS_LL, "a run of word copies between two objects becomes 64-bit copies (one per word pair)"),
    (STRUCTCOPY_WORDS_BLOCK, "a run of word copies between two objects becomes one block copy (helper struct)"),
    (STRUCTCOPY_RETURN_WHOLE, "a returned object filled from one object behind flag checks becomes `return x;`"),
    (SDK_WORD_SHARED_INLINE, "a value shared by several register-word starts is written out at each start"),
    (SDK_FOLD_SLOT_LOCALS, "volatile-register locals holding a global read fold into their use (no frame slots)"),
    (INDEX_NAMED_SCALED, "a scaled array index comes from a named local assigned before its statement"),
    (MERGE_REGISTER_WEBS, "successive webs of one callee-saved register are one variable"),
    (EXPR_MASK_THEN_SHIFT, "a clrlslwi becomes (x & m) << s (mask first) instead of x << s & M"),
    (SELECT_TYPED_ARG, "a condition-selected constant passed to a call is a select of typed constants in the call"),
    (ARGS_NAMED_LOCALS, "computed call arguments become named locals assigned in argument order"),
    (ORDER_PARAM_LOADS, "the leading parameter loads are ordered by parameter, then offset"),
    (GLOBAL_STRUCT_COPY, "a global object copied word by word behind a pointer becomes one struct assignment"),
    (NAMED_FP_COPIES, "temporaries of a floating-point copy run stay named locals, declared by target register"),
    (INCDEC_STEP_AFTER, "a step after a read the next statement uses is a statement after it, not a post-increment"),
    (EXPR_BYTE_FIELDS, "byte/halfword fields packed into a word are narrowing conversions, not masks"),
    (CONST_READ_ONLY_EXTERNS, "scalar externs the function only reads are declared const"),
    (EXPR_MASK_SELECT, "a value masked by a 0/-1 select is a select between the value and zero"),
    (HW_CONST_POINTER, "hardware registers at folded addresses are read through a constant pointer, not an array at the address"),
    (UPDATE_AFTER_COPY, "a register updated in place after a copy of its old value: one variable updated, the copy in another"),
    (SDK_PARAM_VIEW, "accesses through a pointer parameter go through a named local copy of it"),
    (ACCUM_SINGLE_UPDATE, "a value updated once in its callee-saved register is one variable updated in place"),
    (INSERT_VALUES_FIRST, "values inserted by a run of register-field inserts are named locals computed before it"),
    (LOWMEM_ARRAYS, "OS low-memory words are elements of arrays declared at their addresses"),
    (PARAM_CONST_POINTERS, "parameters only loaded through are declared pointer-to-const (their loads ignore stores)"),
    (ORDER_ADDRESS_FIRST, "an address temp computed after an independent value temp moves before it"),
    (NAMED_ALGORITHM_RESULT, "a free algorithm's result used by the next condition stays a named local"),
    (INPLACE_MUTATOR, "member stores explained as `x = x op v` and as `x op= v` take the in-place mutator"),
    (ORDER_SPLIT_LAST_FIELD, "a packed word built from field inserts gets its last field in a statement of its own"),
    (BOOL_CONST_LOCAL, "bool locals defined once are const (re-extended at their uses)"),
    (LOOP_INVARIANT_READS, "temps read before a loop and used once inside it are read in the loop"),
    (FLOAT_UNFUSED_PRODUCTS, "float products the compiler did not fuse into an add stay locals"),
    (ASSIGN_COMPOUND_NARROW, "narrow destinations updated from themselves use compound assignment"),
    (LOCALS_NARROW_BY_DEFS, "int locals written only narrowed take the narrow type"),
    (ARITH_LEFT_ASSOC, "right-nested integer sums are written left to right (a + b + c)"),
    (SRET_CLASS_BY_LAYOUT, "an unnamed struct return takes the one context class whose layout and constructor fit its stores"),
];

#[derive(Default)]
struct State {
    active: bool,
    flipped: Vec<&'static str>,
    asked: BTreeSet<&'static str>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Whether decision point `name` takes its alternative in the current draft (recording that the
/// point was asked).
pub fn alt(name: &'static str) -> bool {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if !s.active {
            return false;
        }
        s.asked.insert(name);
        s.flipped.contains(&name)
    })
}

/// Run `f` (a lift + emit) with the points in `flipped` taking their alternatives; returns its
/// result and the decision points it asked, in name order. Not reentrant.
pub fn draft<T>(flipped: &[&'static str], f: impl FnOnce() -> T) -> (T, Vec<&'static str>) {
    STATE.with(|s| *s.borrow_mut() = State { active: true, flipped: flipped.to_vec(), asked: BTreeSet::new() });
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            STATE.with(|s| *s.borrow_mut() = State::default());
        }
    }
    let _reset = Reset;
    let out = f();
    let asked = STATE.with(|s| s.borrow().asked.iter().copied().collect());
    (out, asked)
}

