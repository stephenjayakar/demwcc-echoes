//! mwdec CLI. Each subcommand is a separate `cmd_*` function; new subcommands (decomp, match,
//! eval) plug in by adding a `Cmd` variant and a function that calls into their crate.
use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use mwdec_core::*;
use mwdec_mwcc::{DiffClass, ExternIndex, Mwcc, ObjIndex};
use mwdec_project::{harness, size_bucket, Project, SIZE_BUCKETS};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

mod search_cmds;

#[derive(Parser)]
#[command(name = "mwdec", about = "Matching decompiler for Metroid Prime 2 (MWCC GC/2.7)")]
struct Cli {
    /// Project root (read-only). Default: $MWDEC_ROOT or the frozen c1 worktree.
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    /// Scratch dir for compiles/PCH/cache. Default: $MWDEC_WORK or $MWDEC_WORK_BASE/mwdec-mwcc.
    #[arg(long, global = true)]
    work: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Disassemble a function (with relocations) from the unit's target object
    /// (or from an object file, if <unit> ends in ".o").
    Dump {
        unit: String,
        symbol: String,
        /// Dump our compiled base object (build/G2ME01/src/...) instead of the target.
        #[arg(long)]
        base: bool,
    },
    /// Compile <file.cpp> in the unit's context with the unit's flags and strictly compare <symbol>.
    Check {
        unit: String,
        symbol: String,
        file: PathBuf,
        /// Compile the file standalone (no unit context prepended).
        #[arg(long)]
        no_context: bool,
        /// Prepend the context as text instead of using a precompiled header.
        #[arg(long)]
        no_pch: bool,
        /// Also print both disassemblies.
        #[arg(long)]
        diff: bool,
    },
    /// Strictly compare our already-built base objects against the targets over the dataset.
    Selftest {
        #[arg(long)]
        limit: Option<usize>,
        /// Only units whose name contains this string.
        #[arg(long)]
        unit: Option<String>,
        /// Print every mismatch (default: first 40).
        #[arg(long)]
        all: bool,
    },
    /// The dataset of currently-100% non-weak functions.
    Dataset {
        /// Counts by split and size bucket.
        #[arg(long)]
        stats: bool,
        /// Only this split ("train" / "test") when listing.
        #[arg(long)]
        split: Option<String>,
    },
    /// Print the harness context TU (include lines only) of a unit.
    Context { unit: String },
    /// Compile every unit's context TU (no PCH, no cache) and list the ones that fail.
    CtxCheck {
        /// Only units whose name contains this string.
        #[arg(long)]
        unit: Option<String>,
        #[arg(long, default_value_t = 6)]
        jobs: usize,
    },
    /// Measure compile times for a unit: plain context vs PCH vs cache hit vs parallel pool.
    BenchCompile {
        unit: String,
        #[arg(long, default_value_t = 5)]
        n: usize,
        #[arg(long, default_value_t = 6)]
        jobs: usize,
    },
    /// First draft (lift + emit, or --init) followed by the compiler-in-the-loop permuter.
    Match {
        unit: String,
        symbol: String,
        #[arg(long, default_value_t = 60)]
        budget_secs: u64,
        /// Start from this candidate source instead of the decompiler's draft.
        #[arg(long)]
        init: Option<PathBuf>,
        /// Stop after this many real compiles.
        #[arg(long)]
        max_compiles: Option<usize>,
        /// Concurrent candidate evaluations (compiles in flight are capped at 6).
        #[arg(long, default_value_t = 6)]
        workers: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Draft without the header TypeDb.
        #[arg(long)]
        no_db: bool,
        /// Print every improvement.
        #[arg(short, long)]
        verbose: bool,
        /// Directory for best.cpp/result.json (default $MWDEC_WORK_BASE/search/<unit>/<symbol>).
        #[arg(long)]
        out: Option<PathBuf>,        /// Search without diff localisation (focus / schedcheck moves).
        #[arg(long)]
        no_locate: bool,
        /// Comma-separated operator names to disable (ablations).
        #[arg(long)]
        disable_ops: Option<String>,
    },
    /// Decompile + search a sample of dataset functions from one split; JSONL + table by size.
    Eval {
        #[arg(long, default_value = "test")]
        split: String,
        #[arg(long)]
        max_size: Option<u32>,
        #[arg(long, default_value_t = 0)]
        min_size: u32,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Search budget per function (0 = first draft only).
        #[arg(long, default_value_t = 30)]
        budget_secs: u64,
        #[arg(long)]
        max_compiles: Option<usize>,
        /// Functions processed in parallel.
        #[arg(long, default_value_t = 3)]
        jobs: usize,
        /// Search workers per function (default ceil(6 / jobs)).
        #[arg(long)]
        workers: Option<usize>,
        #[arg(long)]
        no_db: bool,
        /// Only units whose name contains this string.
        #[arg(long)]
        unit: Option<String>,
        /// Output JSONL path (default $MWDEC_WORK_BASE/eval/eval_<split>_...jsonl).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Also evaluate functions that can't exist as standalone source (compiler-generated
        /// special members, header-defined template members); default: an `impl` column.
        #[arg(long)]
        include_implicit: bool,
        /// Search without diff localisation (focus / schedcheck moves).
        #[arg(long)]
        no_locate: bool,
        /// Comma-separated operator names to disable (ablations).
        #[arg(long)]
        disable_ops: Option<String>,
    },
}

fn main() -> Result<()> {
    mwdec_core::memcap::install();
    // Deep IR recursion on large functions needs more than the default 1 MB stack.
    std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(real_main)
        .expect("spawn main thread")
        .join()
        .unwrap_or_else(|e| std::panic::resume_unwind(e))
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();
    let root = cli.root.clone().unwrap_or_else(mwdec_mwcc::default_root);
    let work = cli.work.clone().unwrap_or_else(mwdec_mwcc::default_work);
    match cli.cmd {
        Cmd::Dump { unit, symbol, base } => cmd_dump(&root, &unit, &symbol, base),
        Cmd::Check { unit, symbol, file, no_context, no_pch, diff } => {
            cmd_check(&root, &work, &unit, &symbol, &file, no_context, no_pch, diff)
        }
        Cmd::Selftest { limit, unit, all } => cmd_selftest(&root, limit, unit.as_deref(), all),
        Cmd::Dataset { stats, split } => cmd_dataset(&root, stats, split.as_deref()),
        Cmd::Context { unit } => cmd_context(&root, &unit),
        Cmd::CtxCheck { unit, jobs } => cmd_ctx_check(&root, &work, unit.as_deref(), jobs),
        Cmd::BenchCompile { unit, n, jobs } => cmd_bench_compile(&root, &work, &unit, n, jobs),
        Cmd::Match { unit, symbol, budget_secs, init, max_compiles, workers, seed, no_db, verbose, out, no_locate, disable_ops } => search_cmds::cmd_match(
            &root,
            &cli.work.clone().unwrap_or_else(|| search_cmds::search_work()),
            search_cmds::MatchArgs { unit, symbol, budget_secs, init, max_compiles, workers, seed, no_db, verbose, out, no_locate, disable_ops },
        ),
        Cmd::Eval { split, max_size, min_size, limit, seed, budget_secs, max_compiles, jobs, workers, no_db, unit, out, no_locate, disable_ops, include_implicit } => {
            search_cmds::cmd_eval(
                &root,
                &cli.work.clone().unwrap_or_else(|| search_cmds::search_work()),
                search_cmds::EvalArgs { split, max_size, min_size, limit, seed, budget_secs, max_compiles, jobs, workers, no_db, unit, out, no_locate, disable_ops, include_implicit },
            )
        }
    }
}

fn load_project(root: &Path) -> Result<Project> {
    Project::load(root).with_context(|| format!("loading project at {}", root.display()))
}

fn find_unit<'p>(p: &'p Project, name: &str) -> Result<&'p Unit> {
    if let Some(u) = p.unit(name) {
        return Ok(u);
    }
    // Convenience: accept a unique suffix (e.g. "MetroidPrime/CActor").
    let m: Vec<&Unit> = p.units.iter().filter(|u| u.name.ends_with(name)).collect();
    match m.len() {
        1 => Ok(m[0]),
        0 => bail!("unknown unit {name}"),
        _ => bail!("ambiguous unit {name}: {}", m.iter().map(|u| u.name.as_str()).collect::<Vec<_>>().join(", ")),
    }
}

fn load_obj(p: &Project, rel: &str) -> Result<ObjectFile> {
    mwdec_obj::load_object(&p.path(rel).to_string_lossy())
}

fn cmd_dump(root: &Path, unit: &str, symbol: &str, base: bool) -> Result<()> {
    let p = load_project(root)?;
    if unit.ends_with(".o") {
        // Direct object path (absolute or project-relative).
        let o = mwdec_obj::load_object(&p.path(unit).to_string_lossy())?;
        let f = mwdec_obj::find_function(&o, symbol).ok_or_else(|| anyhow!("{symbol} not in {unit}"))?;
        println!("# {} ({:?}, {:#x} bytes) in {unit}", f.name, f.binding, f.code.len());
        for l in mwdec_obj::disassemble(f) {
            println!("{l}");
        }
        return Ok(());
    }
    let u = find_unit(&p, unit)?;
    let rel = if base { u.base_obj.clone().ok_or_else(|| anyhow!("unit has no base object"))? } else { u.target_obj.clone() };
    let o = load_obj(&p, &rel)?;
    let f = mwdec_obj::find_function(&o, symbol).ok_or_else(|| anyhow!("{symbol} not in {rel}"))?;
    println!("# {} {} ({:?}, {:#x} bytes) in {rel}", u.name, f.name, f.binding, f.code.len());
    for l in mwdec_obj::disassemble(f) {
        println!("{l}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_check(
    root: &Path,
    work: &Path,
    unit: &str,
    symbol: &str,
    file: &Path,
    no_context: bool,
    no_pch: bool,
    diff: bool,
) -> Result<()> {
    let p = load_project(root)?;
    let u = find_unit(&p, unit)?;
    if u.cflags.is_empty() {
        bail!("unit {} has no compiler flags in build.ninja", u.name);
    }
    let code = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let m = Mwcc::new(root, work, 6);
    let context = if no_context { String::new() } else { harness::context_tu(&p, u)? };
    let t = Instant::now();
    let plain = m.plain_context(&context, &u.cflags);
    let ctx = if no_pch || context.is_empty() {
        plain.clone()
    } else {
        // A context that does not precompile (or crashes the compiler) still works as plain text.
        m.precompile(&context, &u.cflags).unwrap_or_else(|e| {
            eprintln!("warning: precompiling the context failed ({}); using the plain context", e.to_string().lines().next().unwrap_or(""));
            plain.clone()
        })
    };
    let t_ctx = t.elapsed();
    let t = Instant::now();
    // Crash fallback: PCH crashes already retry with the plain context inside `compile_in`; a
    // crash with the plain context is retried once (crashes are never cached).
    let mut res = m.compile_in(&ctx, &code);
    if matches!(res, Err(mwdec_mwcc::MwccError::Crash { .. })) {
        eprintln!("warning: compiler crashed; retrying with the plain context");
        res = m.compile_in(&plain, &code);
    }
    let compiled = match res {
        Ok(c) => c,
        Err(e @ mwdec_mwcc::MwccError::Crash { .. }) => {
            println!("COMPILER CRASH\n{e}");
            std::process::exit(3);
        }
        Err(e) => {
            println!("COMPILE ERROR\n{e}");
            std::process::exit(2);
        }
    };
    let t_cc = t.elapsed();
    let ours = mwdec_obj::load_object_bytes(&compiled.obj_path.to_string_lossy(), &compiled.obj)?;
    let target = load_obj(&p, &u.target_obj)?;
    let tf = mwdec_obj::find_function(&target, symbol).ok_or_else(|| anyhow!("{symbol} not in target {}", u.target_obj))?;
    let Some(of) = mwdec_obj::find_function(&ours, symbol) else {
        println!("MISSING: {symbol} not defined by the compiled file; it defines:");
        for f in &ours.functions {
            println!("  {}", f.name);
        }
        std::process::exit(1);
    };
    let ext = module_externs(&p, std::iter::once(u.name.as_str()));
    let (text, oext) = &ext[Project::module_of(&u.name)];
    let d = mwdec_mwcc::compare_indexed(&ObjIndex::with_externs(&target, text), tf, &ObjIndex::with_externs(&ours, oext), of);
    let r = &d.result;
    println!(
        "{} {symbol}: score {:.1} ({}; target {:#x} bytes, ours {:#x}; ctx {:.0} ms, compile {:.0} ms{})",
        if r.exact { "EXACT" } else { "MISMATCH" },
        r.score,
        d.class.label(),
        r.target_len,
        r.ours_len,
        t_ctx.as_secs_f64() * 1e3,
        t_cc.as_secs_f64() * 1e3,
        if compiled.cache_hit { ", cached" } else { "" }
    );
    for n in &r.notes {
        println!("  {n}");
    }
    if !compiled.messages.trim().is_empty() {
        println!("compiler messages:\n{}", compiled.messages.trim_end());
    }
    if diff {
        let a = mwdec_obj::disassemble(tf);
        let b = mwdec_obj::disassemble(of);
        println!("--- target{:>40}--- ours", "");
        for i in 0..a.len().max(b.len()) {
            let l = a.get(i).map(String::as_str).unwrap_or("");
            let rr = b.get(i).map(String::as_str).unwrap_or("");
            let mark = if l.get(7..) == rr.get(7..) { ' ' } else { '|' };
            println!("{l:<70} {mark} {rr}");
        }
    }
    if !r.exact {
        std::process::exit(1);
    }
    Ok(())
}

struct SelfRow {
    unit: String,
    symbol: String,
    size: u32,
    class: DiffClass,
    notes: Vec<String>,
}

fn cmd_selftest(root: &Path, limit: Option<usize>, unit_filter: Option<&str>, all: bool) -> Result<()> {
    let t0 = Instant::now();
    let p = load_project(root)?;
    let mut ds = p.dataset()?;
    if let Some(f) = unit_filter {
        ds.retain(|e| e.unit.contains(f));
    }
    if let Some(n) = limit {
        ds.truncate(n);
    }
    let t_ds = t0.elapsed();
    let mut by_unit: BTreeMap<&str, Vec<&DatasetEntry>> = BTreeMap::new();
    for e in &ds {
        by_unit.entry(e.unit.as_str()).or_default().push(e);
    }
    let units: HashMap<&str, &Unit> = p.units.iter().map(|u| (u.name.as_str(), u)).collect();
    let externs = module_externs(&p, by_unit.keys().copied());
    let rows: Vec<SelfRow> = by_unit
        .par_iter()
        .flat_map_iter(|(uname, entries)| {
            let u = units[uname];
            let (ext, ours_ext) = &externs[Project::module_of(uname)];
            let row = |e: &DatasetEntry, class, notes| SelfRow {
                unit: e.unit.clone(),
                symbol: e.symbol.clone(),
                size: e.size,
                class,
                notes,
            };
            let objs = load_obj(&p, &u.target_obj).and_then(|t| Ok((t, load_obj(&p, u.base_obj.as_deref().unwrap_or(""))?)));
            let out: Vec<SelfRow> = match objs {
                Err(err) => entries.iter().map(|e| row(e, DiffClass::Unresolved, vec![format!("load: {err}")])).collect(),
                Ok((t, o)) => {
                    let ti = ObjIndex::with_externs(&t, ext);
                    let oi = ObjIndex::with_externs(&o, ours_ext);
                    entries
                        .iter()
                        .map(|e| match (ti.function(&e.symbol), oi.function(&e.symbol)) {
                            (Some(tf), Some(of)) => {
                                let d = mwdec_mwcc::compare_indexed(&ti, tf, &oi, of);
                                row(e, d.class, d.result.notes)
                            }
                            (a, b) => row(
                                e,
                                DiffClass::Unresolved,
                                vec![format!("missing in {}", if a.is_none() { "target" } else if b.is_none() { "base" } else { "?" })],
                            ),
                        })
                        .collect()
                }
            };
            out
        })
        .collect();
    let exact = rows.iter().filter(|r| r.class == DiffClass::Exact).count();
    let mut by_class: BTreeMap<DiffClass, usize> = BTreeMap::new();
    for r in &rows {
        *by_class.entry(r.class).or_default() += 1;
    }
    let bad: Vec<&SelfRow> = rows.iter().filter(|r| r.class != DiffClass::Exact).collect();
    for r in bad.iter().take(if all { usize::MAX } else { 40 }) {
        println!("[{}] {} {} ({:#x})", r.class.label(), r.unit, r.symbol, r.size);
        for n in r.notes.iter().take(4) {
            println!("    {n}");
        }
    }
    if !all && bad.len() > 40 {
        println!("... {} more mismatches (use --all)", bad.len() - 40);
    }
    println!(
        "selftest: {exact}/{} exact ({:.2}%), dataset {:.2}s, total {:.2}s",
        rows.len(),
        100.0 * exact as f64 / rows.len().max(1) as f64,
        t_ds.as_secs_f64(),
        t0.elapsed().as_secs_f64()
    );
    for (c, n) in by_class {
        let why = match c {
            DiffClass::Exact => "",
            DiffClass::Size | DiffClass::Code => "  (instruction differences: stale base object or relocs dtk did not emit)",
            DiffClass::RelocLayout => "  (relocation positions/kinds differ)",
            DiffClass::TargetName => "  (real: base build references differently named symbols)",
            DiffClass::Literal => "  (real: literal values/strings/sections differ, or literal sharing differs)",
            DiffClass::Unresolved => "  (undecidable: placeholder name on one side, other side not visible)",
        };
        println!("  {:<13} {n}{why}", c.label());
    }
    Ok(())
}

/// Extern indexes per module, (target side, our side): main + the unit's REL, so literals and
/// functions living in other splits/objects can be resolved. Our side is "as linked": base
/// objects where they exist, else the target split.
fn module_externs<'u>(p: &Project, units: impl Iterator<Item = &'u str>) -> HashMap<String, (ExternIndex, ExternIndex)> {
    let mut mods: Vec<&str> = units.map(Project::module_of).collect();
    mods.sort();
    mods.dedup();
    let main_t = p.load_module_data("main");
    let main_o = p.load_module_linked("main");
    mods.par_iter()
        .map(|m| {
            let mut t = main_t.clone();
            let mut o = main_o.clone();
            if *m != "main" {
                t.extend(p.load_module_data(m));
                o.extend(p.load_module_linked(m));
            }
            (m.to_string(), (ExternIndex::new(t), ExternIndex::new(o)))
        })
        .collect()
}

fn cmd_dataset(root: &Path, stats: bool, split: Option<&str>) -> Result<()> {
    let p = load_project(root)?;
    let ds = p.dataset()?;
    if !stats {
        for e in ds.iter().filter(|e| split.map_or(true, |s| e.split == s)) {
            println!("{}\t{}\t{}\t{}", e.split, e.size, e.unit, e.symbol);
        }
        return Ok(());
    }
    let mut counts: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut units: BTreeMap<&str, std::collections::BTreeSet<&str>> = BTreeMap::new();
    for e in &ds {
        *counts.entry((e.split.as_str(), size_bucket(e.size))).or_default() += 1;
        units.entry(e.split.as_str()).or_default().insert(e.unit.as_str());
    }
    print!("{:<8}", "split");
    for b in SIZE_BUCKETS {
        print!("{b:>8}");
    }
    println!("{:>8}{:>8}", "total", "units");
    for s in ["train", "test"] {
        print!("{s:<8}");
        let mut tot = 0;
        for b in SIZE_BUCKETS {
            let n = counts.get(&(s, b)).copied().unwrap_or(0);
            tot += n;
            print!("{n:>8}");
        }
        println!("{tot:>8}{:>8}", units.get(s).map_or(0, |u| u.len()));
    }
    print!("{:<8}", "all");
    for b in SIZE_BUCKETS {
        let n: usize = ["train", "test"].iter().map(|s| counts.get(&(*s, b)).copied().unwrap_or(0)).sum();
        print!("{n:>8}");
    }
    println!("{:>8}{:>8}", ds.len(), units.values().map(|u| u.len()).sum::<usize>());
    Ok(())
}

fn cmd_context(root: &Path, unit: &str) -> Result<()> {
    let p = load_project(root)?;
    let u = find_unit(&p, unit)?;
    print!("{}", harness::context_tu(&p, u)?);
    Ok(())
}

fn cmd_ctx_check(root: &Path, work: &Path, filter: Option<&str>, jobs: usize) -> Result<()> {
    let p = load_project(root)?;
    let mut m = Mwcc::new(root, work, jobs);
    m.disk_cache = None;
    let units: Vec<&Unit> = p
        .units
        .iter()
        .filter(|u| u.source.is_some() && !u.cflags.is_empty() && filter.map_or(true, |f| u.name.contains(f)))
        .collect();
    let t = Instant::now();
    let bad: Vec<(String, String)> = units
        .par_iter()
        .with_max_len(1)
        .filter_map(|u| {
            let ctx = match harness::context_tu(&p, u) {
                Ok(c) => c,
                Err(e) => return Some((u.name.clone(), format!("context: {e}"))),
            };
            match m.compile_tu(&ctx, &u.cflags, None, &work.join("tmp")) {
                Ok(c) => {
                    let _ = std::fs::remove_file(&c.obj_path);
                    None
                }
                Err(e) => {
                    let msg = e.messages();
                    let line = msg.lines().filter(|l| l.contains("Error") || l.contains("cannot be opened") || l.contains("not found")).last()
                        .or_else(|| msg.lines().find(|l| !l.trim().is_empty())).unwrap_or("").trim().to_string();
                    let detail = msg.lines().skip_while(|l| !l.contains("Error")).nth(1).unwrap_or("").trim().to_string();
                    Some((u.name.clone(), format!("{} {line} {detail}", if matches!(e, mwdec_mwcc::MwccError::Crash { .. }) { "CRASH" } else { "" })))
                }
            }
        })
        .collect();
    for (u, e) in &bad {
        println!("FAIL {u}: {e}");
    }
    println!("ctx-check: {}/{} contexts compile ({:.1}s)", units.len() - bad.len(), units.len(), t.elapsed().as_secs_f64());
    Ok(())
}

/// A probe exercising floats, doubles, strings, a switch and a call; independent of headers.
fn probe(i: usize) -> String {
    format!(
        "#ifdef __cplusplus\nextern \"C\"\n#endif\nvoid mwdec_sink(const char*, float, double);\n\
         int mwdec_probe_{i}(int a, float f) {{\n\
           switch (a) {{ case 0: mwdec_sink(\"zero\", f * 1.5f, 2.25); break;\n\
           case 1: mwdec_sink(\"one\", f + 3.0f, 0.5); break; case 2: return 7; case 3: return {i}; }}\n\
           return a * {i} + 1;\n}}\n"
    )
}

/// A probe using header inlines (CVector3f), for checking PCH equivalence where available.
fn header_probe(i: usize) -> String {
    format!(
        "float mwdec_hprobe_{i}(const CVector3f& a, const CVector3f& b, float t) {{
           CVector3f v = CVector3f::Lerp(a, b, t);
           v += CVector3f::Zero();
           v *= 2.5f;
           return CVector3f::Dot(v, a) + v.MagSquared() + v.Magnitude() + {i}.0f;
}}
"
    )
}

fn cmd_bench_compile(root: &Path, work: &Path, unit: &str, n: usize, jobs: usize) -> Result<()> {
    let p = load_project(root)?;
    let u = find_unit(&p, unit)?;
    let context = harness::context_tu(&p, u)?;
    // Fresh driver without disk cache so every compile really runs.
    let mut m = Mwcc::new(root, work, jobs);
    m.disk_cache = None;
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;

    let plain = m.plain_context(&context, &u.cflags);
    let mut v = Vec::new();
    for i in 0..n {
        let t = Instant::now();
        m.compile_in(&plain, &probe(1000 + i)).map_err(|e| anyhow!("{e}"))?;
        v.push(ms(t));
    }
    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!("plain context (no PCH): {:.0} ms/compile  {:?}", avg(&v), v.iter().map(|x| x.round()).collect::<Vec<_>>());

    let t = Instant::now();
    let pch = m.precompile(&format!("{context}// bench {}\n", std::process::id()), &u.cflags).map_err(|e| anyhow!("{e}"))?;
    println!("precompile context (.mch): {:.0} ms (once per unit, then cached on disk)", ms(t));
    let mut w = Vec::new();
    for i in 0..n {
        let t = Instant::now();
        m.compile_in(&pch, &probe(2000 + i)).map_err(|e| anyhow!("{e}"))?;
        w.push(ms(t));
    }
    println!("with PCH (-prefix .mch): {:.0} ms/compile  {:?}", avg(&w), w.iter().map(|x| x.round()).collect::<Vec<_>>());

    let t = Instant::now();
    m.compile_in(&pch, &probe(2000)).map_err(|e| anyhow!("{e}"))?;
    println!("cache hit: {:.3} ms", ms(t));

    // Equivalence: same probe, plain vs PCH, must give identical functions (including weak
    // header-inline instantiations) and identical data.
    for code in [probe(3000), header_probe(3001)] {
        let a = match m.compile_in(&plain, &code) {
            Ok(a) => a,
            Err(e) => {
                println!("plain vs PCH: probe skipped (does not compile in this context: {})", e.messages().lines().last().unwrap_or(""));
                continue;
            }
        };
        let b = m.compile_in(&pch, &code).map_err(|e| anyhow!("{e}"))?;
        let oa = mwdec_obj::load_object_bytes("plain", &a.obj)?;
        let ob = mwdec_obj::load_object_bytes("pch", &b.obj)?;
        let (mut same, mut diff) = (0, Vec::new());
        for f in &oa.functions {
            match mwdec_obj::find_function(&ob, &f.name) {
                Some(g) if mwdec_mwcc::compare(&oa, f, &ob, g).exact && f.binding == g.binding => same += 1,
                _ => diff.push(f.name.clone()),
            }
        }
        if ob.functions.len() != oa.functions.len() {
            diff.push(format!("function count {} vs {}", oa.functions.len(), ob.functions.len()));
        }
        let data_a: Vec<(&String, &Vec<u8>)> = oa.data.iter().map(|(k, v)| (k, &v.bytes)).collect();
        let data_b: Vec<(&String, &Vec<u8>)> = ob.data.iter().map(|(k, v)| (k, &v.bytes)).collect();
        if data_a != data_b {
            diff.push("data symbols differ".into());
        }
        println!("plain vs PCH: {same}/{} functions identical{}", oa.functions.len(),
            if diff.is_empty() { ", data identical".to_string() } else { format!("; DIFFERENT: {diff:?}") });
    }

    let k = jobs * 4;
    let codes: Vec<String> = (0..k).map(|i| probe(4000 + i)).collect();
    let t = Instant::now();
    let res = m.compile_many(&pch, &codes);
    let total = ms(t);
    let ok = res.iter().filter(|r| r.is_ok()).count();
    println!(
        "parallel pool ({jobs} jobs): {k} compiles in {total:.0} ms = {:.1} ms/compile effective ({ok} ok)",
        total / k as f64
    );
    // Clean up: the bench-only PCH and the uncached objects left in work/tmp.
    if let Some(mch) = &pch.mch {
        let _ = std::fs::remove_file(mch);
    }
    if let Ok(rd) = std::fs::read_dir(work.join("tmp")) {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with(&format!("tu_{}_", std::process::id())) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    Ok(())
}
