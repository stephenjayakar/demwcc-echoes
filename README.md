# demwcc-echoes

`mwdec`, a matching decompiler for the Metroid Prime 2 (Echoes) decompilation project ([PrimeDecomp/echoes](https://github.com/PrimeDecomp/echoes)), targeting Metrowerks CodeWarrior (MWCC GC/2.7) for PowerPC Gekko.

This repository holds the decompiler code only. It contains no game code or data; to run it you need your own built checkout of the decomp project (with its compilers) and point `MWDEC_ROOT` at it. Memory: every binary caps itself at `MWDEC_MEM_MB` (default 3072).

Build: `cargo build --release`, binary `mwdec`.

---

# mwdec: a matching decompiler for Metroid Prime 2 (MWCC GC/2.7)

Goal: given the machine code of a function from the original game (main.dol / RELs, as split by dtk),
produce C++ source that, compiled with the project's own compiler and flags, is **byte-identical**
(instructions, relocations and referenced literal values) to the target. "Matches" is decided by
the real compiler plus a strict comparator, never by the decompiler itself.

## Anti-cheat rules (hard)

The decompiler (every crate except the eval harness's bookkeeping) may read ONLY:

1. The **target object** of the unit: `build/G2ME01/obj/<unit>.o` (dtk split of the original
   binary: code bytes, relocations, symbol table, data sections).
2. The **symbol table** names (mangled names give signatures, like debug symbols would).
   Names are information about *what* is called/loaded, never a key into known source.
3. **Headers** under `include/` and `libc` etc. as type context, compiled with MWCC `-g` to get
   DWARF type info and/or parsed. A context TU is the list of `#include` lines of the unit
   (standard decomp "context"); the unit's own `.cpp` body is never read.
4. The **compiler** (`build/compilers/GC/2.7/mwcceppc.exe`) as an oracle it may run as often as it
   likes, and anything we learn by reverse engineering it (experiments with the compiler).

Forbidden: reading `src/**/*.cpp` bodies, git history, report.json source paths for content,
any table keyed by function name/address that maps to source text, `asm { }` / `asm` functions,
`#pragma` codegen hacks not used by the project (e.g. `#pragma optimization_level` changes
are allowed only if the project's real source uses them for that unit, which we don't know, so: no),
emitting raw bytes, `__declspec(section)` games. Output must be ordinary C/C++.

Learned rules: any statistics/patterns mined from existing matched (asm, source) pairs must come
from the **train split** only. Evaluation is reported on the **test split** (held-out units).
Split: unit name hashed (FNV-1a 32 of the unit path) — `hash % 10 < 7` train, else test.
The decompiler binary must never open files under `src/`; the eval harness enforces this by
building the decompiler's inputs itself and passing only paths in `build/`, `include/`,
`libc/`-like header dirs and `config/`.

## Strict comparator (what "100%" means)

For the function symbol in our compiled object vs the target object:
- same size, same instruction words after masking relocated fields;
- same relocation offsets and kinds;
- relocation targets: same symbol name, OR both are compiler-local literals (`@123`, `...data.0`,
  `lbl_`/`@stringBase0` style) whose referenced **bytes are equal** (float/double/string values);
- branch targets inside the function equal.
This is stricter than objdiff (which ignores literal values and reloc targets).

## Workspace layout

| crate | role |
|---|---|
| `mwdec-core` | shared data types (objects, functions, relocs, types DB, IR handles) |
| `mwdec-obj` | load ELF objects (target + ours), extract functions/data, disassemble (ppc750cl) |
| `mwdec-project` | parse `build.ninja` (units, cflags, paths), `report.json`, dataset + split |
| `mwdec-mwcc` | compile a TU with unit flags, strict compare, PCH/caching, parallel compile pool |
| `mwdec-ctx` | type context: compile context TU with `-g`, parse DWARF 1.1 into `TypeDb` |
| `mwdec-lift` | PPC -> IR: CFG, stack frame, calling convention, SSA/dataflow, expressions, structuring |
| `mwdec-emit` | IR -> C++ text with project idioms (members, vtable calls, CVector3f, etc.) |
| `mwdec-search` | match loop: variant generators guided by asm diff, compiler-in-the-loop search |
| `mwdec` (cli) | `dump`, `decomp`, `check`, `eval`, `match` subcommands |

Paths: the decomp checkout is `$MWDEC_ROOT` (read-only input) and scratch output goes to
`$MWDEC_WORK_BASE` (defaults: current directory and `./mwdec-work`; build-time defaults can be set
with `MWDEC_DEFAULT_ROOT` / `MWDEC_DEFAULT_WORK_BASE` in a local `.cargo/config.toml` `[env]` table).

## Pipeline

```
target .o --obj--> Function{code,relocs}  +  TypeDb (ctx)  +  symbol signatures (cwdemangle)
   --lift--> IR (CFG of statements/expressions with typed values)
   --emit--> candidate C++ function text
   --mwcc--> compile in context TU --compare--> exact? done : diff
   --search--> rewrite variants (statement order, temps, types, inline accessors, loop forms,
               casts, comparison forms, register-pressure shaping...) --> repeat
```

## CLI contract

- `mwdec dump <unit> <symbol>`: disassembly with relocs.
- `mwdec decomp <unit> <symbol>`: print first-draft C++.
- `mwdec check <unit> <symbol> <file.cpp>`: compile file (body for that symbol) in unit context, strict compare.
- `mwdec match <unit> <symbol> [--budget N]`: decomp + search, prints matching source or best.
- `mwdec eval [--split test] [--max-size N] [--limit K] [--jobs J]`: run `match` over the dataset of
  functions currently at 100% in report.json (ground truth that a match exists), report exact-match
  rate by size bucket. Never reads source.
