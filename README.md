# demwcc-echoes

`mwdec`, a matching decompiler for the Metroid Prime 2 (Echoes) decompilation project ([PrimeDecomp/echoes](https://github.com/PrimeDecomp/echoes)), targeting Metrowerks CodeWarrior (MWCC GC/2.7) for PowerPC Gekko.

This repository holds the decompiler code only. It contains no game code or data; to run it you need your own built checkout of the decomp project (with its compilers) and point `MWDEC_ROOT` at it. Memory: every binary caps itself at `MWDEC_MEM_MB` (default 3072).

Build: `cargo build --release`, binary `mwdec`.

## How it works

Goal: given the machine code of a function from the original game (main.dol / RELs, as split by dtk),
produce C++ source that, compiled with the project's own compiler and flags, is **byte-identical**
(instructions, relocations and referenced literal values) to the target. "Matches" is decided by
the real compiler plus a strict comparator, never by the decompiler itself.

mwdec works in two halves. A deterministic **drafter** lifts the target object's PowerPC code
through a chain of intermediate representations (instructions, CFG, per-block statements, a
structured statement tree) to C++ text, using types recovered from the project's headers. A
**searcher** then compiles that text with the real compiler, compares it strictly with the
target, and applies source-level mutations until the object is identical or the budget runs out.

## Documentation

- [DESIGN.md](DESIGN.md): the representations from object bytes to C++, pass order and layering, anti-cheat rules, the strict comparator (what a match means), the train/test split, the CLI.
- [ARCHITECTURE.md](ARCHITECTURE.md): what each crate does, the dependency graph, data flow of the main commands, caches and the memory cap.
