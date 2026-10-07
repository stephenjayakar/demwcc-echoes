//! Synthetic validation of the permuter: for each invented target function in
//! `synth/corpus.cpp`, compile it, scramble it with K random operators (keeping only scrambles
//! that compile to a different object), then search from the scramble and check whether the
//! exact target object is recovered.
//!
//! permuter-bench [--corpus FILE] [--cases a,b] [--seeds N] [--k K] [--budget-secs S]
//!                [--max-evals N] [--workers W] [--no-guided] [--no-adaptive] [--ops a,b]
//!                [--scramble-ops a,b] [--json FILE]
use anyhow::{anyhow, bail, Context, Result};
use mwdec_mwcc::{Mwcc, ObjIndex};
use mwdec_search::ops::{self, OPS};
use mwdec_search::rng::Rng;
use mwdec_search::score::{Eval, Scorer};
use mwdec_search::search::{search, SearchConfig};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

struct Args {
    corpus: PathBuf,
    cases: Vec<String>,
    seeds: u64,
    k: usize,
    budget: u64,
    max_evals: Option<usize>,
    workers: usize,
    guided: bool,
    adaptive: bool,
    ops: Vec<String>,
    json: Option<PathBuf>,
    scramble_ops: Vec<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        corpus: PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/synth/corpus.cpp")),
        cases: vec![],
        seeds: 3,
        k: 2,
        budget: 30,
        max_evals: None,
        workers: 6,
        guided: true,
        adaptive: true,
        ops: vec![],
        json: None,
        scramble_ops: vec![],
    };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        let mut v = || it.next().ok_or_else(|| anyhow!("{x} needs a value"));
        match x.as_str() {
            "--corpus" => a.corpus = v()?.into(),
            "--cases" => a.cases = v()?.split(',').map(String::from).collect(),
            "--seeds" => a.seeds = v()?.parse()?,
            "--k" => a.k = v()?.parse()?,
            "--budget-secs" => a.budget = v()?.parse()?,
            "--max-evals" => a.max_evals = Some(v()?.parse()?),
            "--workers" => a.workers = v()?.parse()?,
            "--no-guided" => a.guided = false,
            "--no-adaptive" => a.adaptive = false,
            "--ops" => a.ops = v()?.split(',').map(String::from).collect(),
            "--scramble-ops" => a.scramble_ops = v()?.split(',').map(String::from).collect(),
            "--json" => a.json = Some(v()?.into()),
            _ => bail!("unknown argument {x}"),
        }
    }
    Ok(a)
}

fn parse_corpus(text: &str) -> (String, Vec<(String, String)>) {
    let mut pre = String::new();
    let mut cases: Vec<(String, String)> = vec![];
    for line in text.lines() {
        if let Some(n) = line.strip_prefix("//@ case ") {
            cases.push((n.trim().to_string(), String::new()));
        } else if let Some(c) = cases.last_mut() {
            c.1.push_str(line);
            c.1.push('\n');
        } else {
            pre.push_str(line);
            pre.push('\n');
        }
    }
    (pre, cases)
}

fn main() -> Result<()> {
    mwdec_core::memcap::install();
    let args = parse_args()?;
    let root = mwdec_mwcc::default_root();
    let project = mwdec_project::Project::load(&root)?;
    // Game profile flags: any main.dol C++ unit.
    let unit = project
        .units
        .iter()
        .find(|u| u.name.starts_with("main/MetroidPrime/") && !u.cflags.is_empty())
        .ok_or_else(|| anyhow!("no main game unit with flags"))?;
    let work = mwdec_core::paths::work_dir("mwdec-search/bench");
    let mut mwcc = Mwcc::new(&root, &work, args.workers.min(6));
    mwcc.disk_cache = None;
    let text = std::fs::read_to_string(&args.corpus).with_context(|| format!("reading {}", args.corpus.display()))?;
    let (pre, cases) = parse_corpus(&text);
    let ctx = mwcc.precompile(&pre, &unit.cflags).map_err(|e| anyhow!("{e}"))?;
    let mut rows = Vec::new();
    let mut agg: BTreeMap<&str, (u64, u64, u64)> = BTreeMap::new(); // tries, improved, new_best
    let mut scramble_w = ops::base_weights();
    if !args.scramble_ops.is_empty() {
        for (i, o) in OPS.iter().enumerate() {
            if !args.scramble_ops.iter().any(|n| n == o.name) {
                scramble_w[i] = 0.0;
            }
        }
    }
    for (name, src) in &cases {
        if !args.cases.is_empty() && !args.cases.contains(name) {
            continue;
        }
        let tobj_c = mwcc.compile_in(&ctx, src).map_err(|e| anyhow!("{name}: {e}"))?;
        let tobj = mwdec_obj::load_object_bytes(name, &tobj_c.obj)?;
        let tf = tobj
            .functions
            .iter()
            .find(|f| f.name.starts_with(&format!("{name}__")))
            .ok_or_else(|| anyhow!("{name}: function not found in object"))?;
        let ti = ObjIndex::new(&tobj);
        let scorer = Scorer::new(&mwcc, &ctx, None, &ti, tf, None, &tf.name);
        for seed in 0..args.seeds {
            // Scramble: K random mutations, must compile and differ from the target.
            let mut rng = Rng::new(seed * 1000 + 17);
            let mut start = None;
            for _attempt in 0..60 {
                let mut s = src.clone();
                let mut used = vec![];
                for _ in 0..args.k {
                    if let Some(m) = ops::mutate(&s, &tf.name, &scramble_w, &mut rng) {
                        s = m.src;
                        used.push(OPS[m.op].name);
                    }
                }
                if used.is_empty() {
                    continue;
                }
                if let (Eval::Ok(f), _) = scorer.eval(&s) {
                    if !f.exact {
                        start = Some((s, used, f));
                        break;
                    }
                }
            }
            let Some((start, used, f0)) = start else {
                println!("{name:<12} seed {seed}: no codegen-changing scramble found");
                continue;
            };
            let cfg = SearchConfig {
                budget: Duration::from_secs(args.budget),
                max_compiles: args.max_evals,
                workers: args.workers,
                seed: seed + 1,
                guided: args.guided,
                adaptive: args.adaptive,
                only_ops: args.ops.clone(),
                out_dir: Some(work.join("runs").join(format!("{name}_{seed}"))),
                ..Default::default()
            };
            let r = search(&scorer, &start, &cfg);
            for s in &r.op_stats {
                let e = agg.entry(s.name).or_default();
                e.0 += s.tries;
                e.1 += s.improved;
                e.2 += s.new_best;
            }
            println!(
                "{name:<12} seed {seed}: {} start penalty {:>4} -> {:>4}  evals {:>5} compiles {:>5} errs {:>4} {:>6.1}s  scramble {:?}",
                if r.exact { "EXACT" } else { "-----" },
                f0.penalty,
                r.best.as_ref().map_or(0, |b| b.penalty),
                r.evals,
                r.compiles,
                r.compile_errors,
                r.seconds,
                used
            );
            rows.push(serde_json::json!({
                "case": name, "seed": seed, "scramble": used, "start_penalty": f0.penalty,
                "exact": r.exact, "best_penalty": r.best.as_ref().map(|b| b.penalty),
                "evals": r.evals, "compiles": r.compiles, "compile_errors": r.compile_errors,
                "seconds": r.seconds, "history": r.history,
            }));
        }
    }
    let n = rows.len();
    let ex = rows.iter().filter(|r| r["exact"].as_bool() == Some(true)).count();
    let evals: u64 = rows.iter().map(|r| r["evals"].as_u64().unwrap_or(0)).sum();
    let secs: f64 = rows.iter().map(|r| r["seconds"].as_f64().unwrap_or(0.0)).sum();
    println!(
        "\nrecovered {ex}/{n} ({:.0}%), {evals} evals in {secs:.0}s = {:.1} candidates/s",
        100.0 * ex as f64 / n.max(1) as f64,
        evals as f64 / secs.max(1e-9)
    );
    println!("{:<16} {:>7} {:>8} {:>8} {:>8}", "operator", "tries", "improved", "new_best", "rate%");
    for (k, (t, i, b)) in &agg {
        if *t > 0 {
            println!("{k:<16} {t:>7} {i:>8} {b:>8} {:>8.2}", 100.0 * *i as f64 / *t as f64);
        }
    }
    if let Some(p) = &args.json {
        std::fs::write(p, rows.iter().map(|r| r.to_string() + "\n").collect::<String>())?;
    }
    Ok(())
}
