# mwdec: a matching decompiler for Metroid Prime 2 (MWCC GC/2.7)

Goal: given the machine code of a function from the original game (main.dol / RELs, as split by dtk),
produce C++ source that, compiled with the project's own compiler and flags, is **byte-identical**
(instructions, relocations and referenced literal values) to the target. "Matches" is decided by
the real compiler plus a strict comparator, never by the decompiler itself.

mwdec works in two halves. A deterministic **drafter** lifts the target object's PowerPC code
through a chain of intermediate representations (instructions, CFG, per-block statements, a
structured statement tree) to C++ text, using types recovered from the project's headers. A
**searcher** then compiles that text with the real compiler, compares it strictly with the
target, and applies source-level mutations until the object is identical or the budget runs out.

This document describes the representations and the layering. How the crates fit together, the
command data flows, caches and the memory cap are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Anti-cheat rules (hard)

The decompiler (every crate except the eval harness's bookkeeping) may read ONLY:

1. The **target object** of the unit: `build/G2ME01/obj/<unit>.o` (dtk split of the original
   binary: code bytes, relocations, symbol table, data sections), and the other target objects of
   the same module for literal values and vtables they define.
2. The **symbol table** names (mangled names give signatures, like debug symbols would).
   Names are information about *what* is called/loaded, never a key into known source.
3. **Headers** under `include/`, `libc/` etc. as type context, compiled with MWCC `-g` to get
   DWARF type info and/or parsed. A context TU is the list of `#include` lines of the unit
   (standard decomp "context"); the unit's own `.cpp` body is never read. Units without a source
   file get an automatic context of header includes chosen from the module's symbol names.
4. The **compiler** (`build/compilers/<version>/mwcceppc.exe`, the unit's version) as an oracle it
   may run as often as it likes, and anything learned by reverse engineering it (experiments with
   the compiler, observing the unmodified compiler under a debugger).

Forbidden: reading `src/**/*.cpp` bodies, git history, report.json source paths for content,
any table keyed by function name/address that maps to source text, `asm { }` / `asm` functions,
`#pragma` codegen hacks not used by the project (e.g. `#pragma optimization_level` changes
are allowed only if the project's real source uses them for that unit, which we don't know, so: no),
emitting raw bytes, `__declspec(section)` games. Output must be ordinary C/C++. Diagnostic
compiles may use extra flags or pragmas (`-sym on`, `#pragma scheduling off` for line
attribution) but a candidate is only ever judged by the plain compile, and such pragmas never
appear in output.

The decompiler binary must never open files under `src/`. The single sanctioned read of a source
file is `mwdec_project::harness::context_tu`, which returns only its `#include` lines; it is eval
harness bookkeeping. report.json is read only by the harness (dataset ground truth, harvest
candidate lists), never for content.

## Strict comparator (what "100%" means)

For the function symbol in our compiled object vs the target object:

- same size, same instruction words after masking relocated fields;
- same relocation offsets and kinds (per instruction word);
- relocation targets: same symbol name and addend, OR both are compiler/splitter-local names
  (`@123`, `...data.0`, `lbl_`, `@stringBase0`, `init$12` style) whose referenced **bytes are
  equal** (float/double/string values; for string pools the NUL-terminated string at
  symbol+addend), in the same section, with equivalent relocations inside the referenced data
  (jump tables, pointer tables); the target-to-ours mapping of such names must be one-to-one
  within the function;
- branch targets inside the function equal;
- placeholders (`fn_<address>`, functions dtk named by address because nobody has named them
  yet) against one of our named functions (a template instance, an inline emitted out of line):
  equal only when **proven by bytes**: our function is compiled in the same context (from the
  sources that make the compiler emit it, `mwdec_emit::instantiate`) and compares exact with the
  placeholder's code under these same rules, recursively up to depth 2 (nested placeholders need
  their own proof), results cached per unit. Without the proof the relocation differs. Module
  `mwdec_mwcc::placeholder` (the proof) and `mwdec::placeholders` (the compiles);
  `MWDEC_NO_PLACEHOLDER=1` turns it off.

This is stricter than objdiff (which ignores literal values and reloc targets). Implementation:
`mwdec_mwcc::compare` / `compare_indexed`, which also classify a mismatch (`DiffClass`: size,
code, reloc-layout, target-name, unresolved, literal). Literals the target only references (dtk
put them in another split) are resolved through an `ExternIndex` of the module's other target
objects. An exact match found with a split precompiled header (a compiler-crash workaround) only
counts once the plain-context compile confirms it.

## Train/test split

Learned rules: any statistics/patterns mined from existing matched (asm, source) pairs or from
eval results must come from the **train split** only. Evaluation is reported on the **test
split** (held-out units).

- Split: FNV-1a 32 of the unit name (the objdiff unit path, e.g. `main/Dir/CFoo`):
  `hash % 10 < 7` train, else test (`mwdec_project::split_of`).
- Dataset (`Project::dataset`): functions at 100% in report.json (ground truth that a match
  exists), non-weak in the target object, not compiler-generated (`name$123`), in units with
  compiler flags and a built object.
- The eval harness builds the decompiler's inputs itself and passes only paths in `build/`,
  `include/`, `libc/`-like header dirs and `config/`.

## Paths

The decomp checkout is `$MWDEC_ROOT` (read-only input) and scratch output goes to
`$MWDEC_WORK_BASE` (defaults: current directory and `./mwdec-work`; build-time defaults can be set
with `MWDEC_DEFAULT_ROOT` / `MWDEC_DEFAULT_WORK_BASE` in a local `.cargo/config.toml` `[env]`
table). See `mwdec_core::paths`.

## Representations

Every stage produces plain owned Rust data that the next stage consumes. The running example is
a made-up class (not game code), compiled with the project's game flags; the outputs shown are
what mwdec produces for it:

```cpp
class Counter { public: int Bump(int n); int mPad; int mCount; };
int Counter::Bump(int n) { if (n > 0) mCount = n; return mCount; }
```

### 1. Object model (`mwdec_core::ObjectFile`, built by `mwdec_obj::load_object`)

`ObjectFile { functions: Vec<Function>, data: BTreeMap<String, DataSymbol>, sections:
Vec<Section>, symbols: Vec<SymbolDef>, all_symbols }`. A `Function` is `{ name (mangled),
binding, address, code: Vec<u8> (big-endian), relocs: Vec<Reloc> }`; a `Reloc` is `{ offset
(function-relative), kind: RelocKind, target: String, addend }`.

- Invariants: relocations always name a symbol. Target objects (dtk) already do; on our side
  (MWCC output) section-symbol relocations are resolved to the containing named symbol + addend,
  so both sides compare by name.
- Added: function boundaries, symbol names, literal bytes reachable at any symbol+addend.
- Lost: ELF layout details irrelevant to matching.
- Example: `Bump__7CounterFi`, 20 bytes of code, no relocations (it touches only `this`).

### 2. Decoded instructions (`mwdec_lift::insn::Insn`, from `mwdec_lift::cfg::decode`)

`Insn { off, ins: ppc750cl::Ins, reloc: Option<Reloc> }`, one per 4-byte word. The register model
(`insn::Reg`) numbers GPRs 0-31, FPRs 32-63, CR fields 64-71, then CTR, LR and XER.CA;
`insn::defs_uses` gives each instruction's defined and used registers.

- Invariants: halfword relocations (`@ha`/`@l`/SDA21 at offset+2) are normalized to the
  instruction start, so `reloc` is the relocation of that instruction.
- Added: opcodes, operands, per-instruction def/use sets. Lost: nothing (a 1:1 view of the
  words).

```
0: cmpwi r4, 0x0
4: ble   0xc
8: stw   r4, 0x4(r3)
c: lwz   r3, 0x4(r3)
10: blr
```

### 3. CFG and frame (`mwdec_lift::cfg::Cfg`, `mwdec_lift::frame::Frame`)

`Cfg { blocks: Vec<Block>, block_of, rpo, idom, ipdom, pd_extra }` with `Block { start, end
(instruction range), term: Term, succs, preds }`. `Term` is `Fall`, `Jump`, `Cond { taken, fall
}`, `Return`, `CondReturn` (`beqlr`), `Switch { targets, table }` (jump tables resolved through
the table's data relocations), `TailCall` or `Stop`. `Cfg::loops` gives natural loops.
`frame::analyze` recognizes the prologue/epilogue (stack size, LR save, `stmw`/`_savegpr`,
FPR and paired-single saves) into `FrameInfo` and marks those instructions skipped.

- Invariants: every instruction belongs to exactly one block (`block_of`); post-dominators are
  computed on the CFG plus the `pd_extra` edges (an early return continues, for structuring, at
  the shared tail it jumped over).
- Added: control flow, loop nesting, which instructions are frame bookkeeping.
- Example: `b0 [0,2) Cond { taken: 2, fall: 1 }`, `b1 [2,3) Fall(2)`, `b2 [3,5) Return`; leaf,
  `FrameInfo { size: 0, .. }`.

### 4. Per-block statements (`mwdec_lift::translate::Lifter` -> `BlockOut`)

The register-free value IR. `Lifter::run` computes the signature (`sig::sig_of`: cwdemangle of
the symbol, refined by the `TypeDb`; return types are not mangled, so they come from header
declarations or are inferred), the EABI argument layout (`translate::layout`), reaching
definitions and register **webs** (union of def sites that reach a common use: these are the
phi-variables), stack slots and address-taken stack objects (`StackModel`), then symbolically
executes every block in reverse post-order. Each block yields `BlockOut { stmts: Vec<Stmt>, cond,
switch, ret }` over the same `Stmt`/`Expr` types as stage 5 (`mwdec_lift::ir`).

- Values: `Var`s indexed by `VarId` with a `VarKind` (`This`, `Param`, `StructRet`, `Local`,
  `Stack { offset, size }`, `Hidden`). `Lifter::is_temp` marks SSA-like temporaries (single
  assignment, one per def site); webs become mutable `Local`s. Both are named after the target
  register they live in (`temp_r3`, `var_r31`), which later lets register hints map onto them.
- Memory accesses keep base, byte offset and access type: `Expr::Load { base, offset, ty }`.
  Calls are `Expr::Call` with a `Callee` (`Direct`, `Method`, `Virtual { vtable_offset, .. }`,
  `Indirect`); literals are read from the object bytes into `Expr::Float` / `Expr::Str`.
- Then `inline::inline_list` folds single-use temps into their use when no intervening side
  effect could change the value (m2c's EvalOnce idea, as a pass), and `inline::dce` drops dead
  temps.
- Lost: register names (kept only as variable names), instruction order within the constraints
  of the effects model.
- Example (before folding):

```
b0: cond (arg0 <= 0)
b1: *(this+0x4) = arg0;
b2: temp_r3 = *(this+0x4);   ret temp_r3
```

### 5. Structured statement tree (`mwdec_lift::ir::IrFunction`)

`structure::Structurer` turns the CFG of `BlockOut`s into `Vec<Stmt>`: `If` (with `&&`/`||`
chains, m2c's reduction), `While`, `DoWhile`, `For`, `Switch` (jump tables and simulated compare
trees, `switchtree`), `Return`, `Break`, `Continue`, with `Goto`/`Label` as the fallback.
The result is `IrFunction { symbol, sig: FuncSig, vars, params, decl_params, this_var, body,
init_list, globals: Vec<GlobalRef>, frame, warnings }`.

- Invariants: plain owned trees (`Box`/`Vec`, `Clone`, no arenas or interior mutability), so
  later passes and tools can rewrite freely; variables are function-scoped; every expression
  carries a `mwdec_core::Type`; lvalues are `Var`, `Load`, `Index`, `Member`, `Global`,
  `BitField`.
- Added: source-level control flow, then C++ idioms, recovered types and loop forms (the
  recovery passes listed under "Pass order" below), names.
- Lost: block layout (only what structuring needs: e.g. which arm of a branch fell through is
  forgotten, the search can re-explore it).
- Example (`IrFunction::body`, TypeDb applied):

```
[If { cond: Binary { op: Gt, l: Var(1), r: Int { value: 0, .. }, .. },
      then: [Assign { dst: Load { base: Var(0), offset: 4, ty: Int { size: 4, signed: true } },
                      src: Var(1) }],
      els: [] },
 Return(Some(Load { base: Var(0), offset: 4, ty: Int { size: 4, signed: true } }))]
```

### 6. Types (`mwdec_core::TypeDb`, built by `mwdec_ctx::build_typedb`)

`Type` is a small enum (`Int { size, signed }`, `Float`, `Ptr`, `Ref`, `Named(qualified name)`,
`Array`, `Const`, `FuncPtr`, `MemberPtr`, `Unknown { size }`, plus `Char`/`Long`/`WChar` kept
distinct for mangling). `TypeDb` holds `classes` (`Class { size, bases, fields: Vec<Field {
name, offset, ty, bitfield, access }>, vtable, methods, vptr_offset, .. }`), `enums`,
`typedefs`, `functions`, `globals`, `decls` (header declarations as `DeclInfo`, including
inline bodies as token lists), `namespaces`, `templates`, `friends`, `abs_addrs`.

Built per context TU: preprocess (`-E`), `scan` the preprocessed headers for declarations,
compile the context plus forcing declarations with `-g` (MWCC only emits DWARF for used types),
parse DWARF 1.1 (`dwarf`, `convert`), resolve header declarations against it (`resolve`). Then
the drafter adds vtables from target objects (`vtables_from_object` / `apply_vtables`) and
layouts of template instances the context only declares
(`mwdec_inline::complete::complete_in`). Queries go through `mwdec_ctx::layout` (`field_at`,
`size_of`) and `mwdec_lift::types` (`field_path`, `ty_of`).

- Invariants: class keys are normalized qualified names (`rstl::auto_ptr<CFoo>` whether they
  came from DWARF or from a mangled symbol); a forward-declared class has `is_declaration`.
- Added: field names and types, signatures, return types, inline bodies. Lost: what neither
  DWARF 1.1 nor the header scan carries (MWCC emits no typedef DIEs and encodes `bool` as
  `unsigned char`; both are recovered from the scan where possible).
- Example: `Counter { size: 8, fields: [mPad @0: int, mCount @4: int] }` turns `Load { base:
  this, offset: 4 }` into `this->mCount` at emission. The TypeDb is optional everywhere: without
  it the draft uses raw offsets.

### 7. Inline templates (`mwdec_inline::InlineLib`, `Template`)

Header inline functions are expanded by MWCC, so their bodies appear inside target functions.
`mwdec_inline::probe` writes one probe function per inline of the context (parameters as
operands), compiles the probes in the unit context, and lifts them with mwdec-lift: each lifted
body becomes a `Template` (IR with `HoleKind` holes for parameters). `mwdec_inline::apply`
matches templates in a target `IrFunction` modulo single-definition temps (so scheduling and
register choice don't matter) and replaces the expansion with the call (`v.MagSquared()`,
`CVector3f::Dot(a, b)`, accessors).

- Invariant: a template is exactly what this compiler with these flags generates for the
  inline, so recognition is by construction, not by name.
- Added: calls to header inlines. Lost: the expanded form (the drafter keeps both variants and
  lets the compiler pick, `choose_draft`).

### 8. Emitted source (`mwdec_emit::Emitted`)

`emit_function(&IrFunction, Option<&TypeDb>, &EmitOptions) -> Emitted { preamble, body }`.
The body is the function definition spelled from the demangled signature (`decl_params` keeps
the header's exact parameter spellings); member accesses become field paths when the TypeDb
knows them, else `*(T*)((char*)p + off)`; virtual calls become method calls; float literals
round-trip exactly (`float::format_float`). The preamble declares globals, file-local functions
and minimal stand-in types the context lacks. `EmitOptions` selects C mode (`-lang=c` units) or
raw offsets (a fallback when field access doesn't compile, e.g. private members).

- Invariant: text only; no AST survives emission. Whatever comes next re-parses.
- Example (with and without TypeDb; both compile to the target's 5 words exactly):

```cpp
int Counter::Bump(int arg0) {
    if (arg0 > 0) {
        this->mCount = arg0;          // without TypeDb: *(int*)((char*)this + 0x4) = arg0;
    }
    return this->mCount;
}
```

(`arg0`: this example has no header declaration to take a parameter name from.)

### 9. Candidates, objects and fitness (`mwdec_search`, `mwdec_mwcc`)

The searcher works on source text. `cst::Cst` is an owned arena over a tree-sitter-cpp parse;
an operator (`ops`, `structural`, `near`) returns byte-range `Edit`s, the text is re-parsed and
deduplicated by `cst::normalize`. `Scorer::eval` compiles the candidate with
`Mwcc::compile_in(&UnitContext, code) -> Compiled { obj, .. }`, loads it with mwdec-obj, and
scores it: `Fitness { exact, penalty, score, class: DiffClass, size_delta, profile: DiffProfile
}`, where exactness is the strict comparator and the penalty/profile (register, stack, branch,
reorder, inserted, deleted, reloc differences over an opcode alignment) rank non-exact
candidates and steer operator weights.

- Invariant: the comparator is the only judge; operators are best-effort semantics preserving.
- Example: the draft above is already exact, so the search loop never runs; only the polish
  pass (readability clean-ups that must keep the match) does. Otherwise operators such as
  `flip_compare` (`arg0 > 0` -> `0 < arg0`) or `negate_if` would be tried, weighted by what
  differs.

## Layering and pass order

```
target .o --mwdec-obj--> ObjectFile/Function      context TU --mwdec-ctx--> TypeDb
   --mwdec-lift-->  Insn -> Cfg/Frame -> BlockOut (temps, webs) -> structured IrFunction
                    -> idiom / type / loop recovery passes            (uses TypeDb if present)
   --mwdec-inline--> header inline expansions folded back into calls (compiler-made templates)
   --mwdec-emit-->  C++ text (preamble + definition)
   --mwdec-mwcc-->  compile in the unit context (PCH) --> strict compare --> exact? done
   --mwdec-search-> mutate text, guided by diff profile, localisation and compiler models; repeat
```

### Pass order inside `mwdec_lift::lift_function`

1. `cfg::decode`, `Cfg::build`, `frame::analyze`, `sig::sig_of` (`Lifter::new`).
2. `Lifter::run`: C parameter inference, parameter setup, call layouts, stack arguments,
   reaching definitions, return inference, split returns, webs, stack scan, block translation.
3. Per block: `inline::inline_list` + `inline::dce` (two rounds), `simplify::form_incdec_with`
   for branch conditions; `rematerialize_loop_headers`, `early_returns`.
4. `structure::Structurer::run`.
5. Value clean-ups: `simplify::*` (logical values, bool vars, virtual delete checks, return
   values), `ctrloop::forward_constant_copies`, `wide::merge_halves` (64-bit pairs),
   `idioms::drop_dead_stack_stores`, `varargs::recover`, `divmagic::fold` (magic-number
   division), `reinline`.
6. TypeDb-driven recovery (only with a TypeDb): `aggregates` (member-wise copies -> struct
   assignment, literal initializers), `arrays` (container members, absolute globals, element
   accesses), `bitfields::recover`, `byval` (by-value argument copies), `construct::fold`
   (argument temporaries -> `T(args)`), `localtypes` (retyping).
7. Typing that works with or without a TypeDb: `localtypes::narrow`, `global_types`,
   `undeclared_returns`, `drop_redundant_masks`.
8. Loops: `ctrloop::recover` (CTR loops and unrolled copies back to one `for`),
   `simplify::recover_ctr_loops`, `unroll::reroll`, `indexing` (undo strength reduction, `a[i].f`).
9. Expression forms: `form_incdec`, ternaries, trailing `return;`.
10. `IrFunction` built; `idioms::apply` (`new`, constructor initializer lists, vtable pointer
    stores, destructor wrappers), `scalars::regroup` (struct locals kept in registers),
    `simplify::form_for_loops`, `simplify::name_vars`.

If the result looks like a constructor into parameter 0, the function is lifted again with r3
as the hidden struct-return pointer. Set `MWDEC_DUMP=1` to print the body after each stage
(`mwdec_lift::debug`).

The draft path in the CLI (`search_cmds::draft`) is: `mwdec_project::standalone` check (header
inlines and compiler-generated members have no standalone source) -> `lift_function` ->
`mwdec_inline::apply` -> `emit_function` -> `extern "C"` for unmangled placeholder symbols.

### Dependency rules

| layer | may use | must not |
|---|---|---|
| `mwdec-core` | serde | depend on any mwdec crate; carry behaviour (types only; add fields/variants, never rename/remove) |
| `mwdec-obj` | core | run the compiler; interpret code beyond disassembly |
| `mwdec-project` | core, obj | read source bodies (`harness::context_tu` returns include lines only, for the harness) |
| `mwdec-mwcc` | core, obj | know about IR or C++ structure; it compiles text and compares objects |
| `mwdec-ctx` | core, obj, mwcc | read source bodies; its inputs are the context TU, the compiler's output and target objects (vtables) |
| `mwdec-lift` | core, obj, ctx (signature lookup only) | run the compiler; read anything but the target `ObjectFile` and an optional `TypeDb` |
| `mwdec-emit` | core, lift | run the compiler; look at objects; change IR semantics (rendering only) |
| `mwdec-inline` | core, obj, lift, mwcc, ctx, project | match by function name; templates come from compiling probes |
| `mwdec-oracle` | core | be required by the draft path; it is a model/experiment library |
| `mwdec-search` | core, obj, mwcc, project, oracle | depend on lift/emit: it searches any C++ text (a draft or a hand-written `--init`) |
| `mwdec` (CLI) | everything | contain decompiler logic beyond orchestration |

Lift is a pure function of (target object, function, optional TypeDb): deterministic, no file I/O,
no compiler. All compiler use is concentrated in mwdec-mwcc (candidate compiles, PCH), mwdec-ctx
(type extraction), mwdec-inline (probe templates) and mwdec-oracle (experiments, tracer).

### Where compiler knowledge plugs in

- **Inverse lowering in mwdec-lift.** Passes undo specific MWCC code generation: CTR loops and
  unrolling (`ctrloop`, `unroll`), switch compare trees simulated to recover case sets
  (`switchtree`), magic-number division (`divmagic`), 64-bit register pairs and runtime helpers
  (`wide`), by-value copies (`byval`), struct locals split into registers (`scalars`), idiom
  wrappers for `new`/constructors/destructors (`idioms`). Each module documents the shape it
  inverts.
- **Compiler as oracle at draft time.** mwdec-ctx gets types from `-g` DWARF; mwdec-inline gets
  inline expansions by compiling probes; the CLI picks between draft variants by compiling both.
- **Compiler models in mwdec-oracle.** `regalloc` (model of MWCC's register colouring:
  virtual-register numbering of params, named locals and temps, simplify/select order),
  `webs` (callee-saved webs recovered from target code), `hints` (inverse colouring: which
  target register holds a param, temp or named local), `sched` / `schedcheck` / `explain`
  (list scheduler pick rule; whether an order difference is forced by a dependence, by register
  reuse, or by statement order), `iro` (the front-end optimizer's own dump), `tracer` (the
  unmodified compiler under a minimal debugger: inliner decisions, colouring, PCode). The models
  are validated against the real compiler on generated functions (`ra-fuzz`, `ra-simplify`,
  `sched-alias-check`) and, source-free, on train-split target code (`ra-scan`,
  `ra-spill-scan`); the reasoning behind them lives in those modules' docs.
- **Consumers in mwdec-search.** `hints` drives the `hint_order` / `hint_temp` operators (draft
  locals are named after their target register, so hints map onto them); `trace` turns
  register-only diffs into directed edits (`trace_fix`); `locate` attributes differing
  instructions to source lines with diagnostic `-sym on` compiles and proposes `schedcheck`
  statement moves (`sched_move`, `sched_swap`).

## CLI contract

Global options: `--root <dir>` (project, read-only; default `$MWDEC_ROOT`), `--work <dir>`
(scratch). Every binary installs the memory cap first (`mwdec_core::memcap::install`).

- `mwdec dump <unit|file.o> <symbol> [--base]`: disassembly with relocations (target object, or our
  built object with `--base`).
- `mwdec check <unit> <symbol> <file.cpp> [--no-context] [--no-pch] [--diff]`: compile the file in
  the unit context with the unit's flags and strictly compare the symbol. Exit 0 exact, 1
  mismatch/missing, 2 compile error, 3 compiler crash.
- `mwdec match <unit> <symbol> [--budget-secs N] [--init file.cpp] [--max-compiles N] [--workers N]
  [--seed S] [--no-db] [--out dir]`: first draft (or `--init`) + search; prints the best source,
  writes `best.cpp`/`result.json`; exit 0 only if exact. `--budget-secs 0` scores the draft
  without searching.
- `mwdec eval [--split test|train] [--max-size N] [--limit K] [--jobs J] [--budget-secs N]
  [--list rows.jsonl] [--mem-report] [--exclude-implicit]`: draft + search over a seeded sample of the dataset of one
  split (default test); JSONL rows plus an exact-match table by size bucket and by kind (header
  inlines, template instances and implicit members are drafted as instantiations and counted). Never reads source bodies.
  `--budget-secs 0` = first drafts only. `--jobs` at most 3.
- `mwdec harvest [--max-size N] [--scope sourced|auto|all] [--tag T] [--summary]`: draft + search
  over functions not yet matched in report.json, smallest first; keeps exact results (resumable,
  supervised).
- `mwdec verify-units <unit>...`: strictly compare every function of built objects with targets.
- Harness/maintenance: `selftest` (comparator over our built objects), `dataset [--stats]`,
  `context <unit>`, `ctx-check`, `bench-compile <unit>`; `draft-server` is internal (eval's
  capped drafting child).
