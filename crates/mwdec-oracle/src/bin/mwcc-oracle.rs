//! One-command MWCC experiment harness.
//!
//! mwcc-oracle asm <file.cpp|-e CODE> [--fn PAT]... [--profile P] [--offsets] [--data] [-- extra flags]
//!     compile with project flags, print annotated disassembly (functions in emission order)
//! mwcc-oracle trace <file.cpp|-e CODE> [--fn PAT] [--all] [--json] [--pcode initial,presched,precolor] [--fix r30:r31,...]
//!     REAL compiler internals (GC/2.7, debugger): per function/class every value's vreg, name, kind
//!     (param/named/FE temp/temp), IG degree, register, colouring order, K-blocked; --fix explains how
//!     to move candidate registers to target registers
//! mwcc-oracle inline <file.cpp|-e CODE> [--fn CALLER] [--all] [--full] [--json]
//!     inliner decisions per call: cost vs inline_max_size, pass, reason; body expansion order
//! mwcc-oracle iro <file.cpp|-e CODE> [--fn PAT] [--all] [--full] [--json]
//!     the front-end optimizer's own dump, rendered as statements per flowgraph block: the final
//!     form handed to code generation (reassociation, operand order, CSE temps, cond. assignments);
//!     --all: every IRO pass; --full: the raw dump text
//! mwcc-oracle picks <file.cpp|-e CODE> [--fn PAT] [--all] [--full] [--json]
//!     REAL list scheduler per basic block: issue order with the reason of every pick (program
//!     order = statement order decides; urgent/uncovers/height/opcode rank/unit busy = it does not);
//!     --all: also the pre-RA pass; --full: the DAG (successors with edge kind and latency)
//! mwcc-oracle copyprop <file.cpp|-e CODE> [--fn PAT] [--all]
//!     REAL PCode copy propagation: for each copy `mr vX, vY` the uses that refused propagation and
//!     why (a move use keeps a named local named); --all: accepted uses and removed copies too
//! mwcc-oracle peephole <file.cpp|-e CODE> [--fn PAT]
//!     REAL post-RA peephole: every rule that changed or removed an instruction
//! mwcc-oracle why <cand.cpp|-e CODE> --target <target.o> --fn NAME [--json]
//!     every order difference against the target, explained by the REAL scheduler: forced by a
//!     dependence, register reuse, program order (=> which statement to move), waiting for an
//!     operand (=> which computation to move), or the scheduler's priority (statement order irrelevant)
//! mwcc-oracle sched <cand.cpp|-e CODE|cand.o> --target <target.o> --fn NAME [--json]
//!     scheduling dependence check: per block, which instruction pairs are in a different order and
//!     whether a data/memory dependence forces it (=> source statements must move; lines via -sym on)
//! mwcc-oracle memclass <file.cpp|-e CODE|file.o> [--fn PAT]
//!     the scheduler alias model per memory access: stack (range, escaped), global, pointer
//! mwcc-oracle hints <file.cpp|-e CODE|file.o> [--fn PAT]...
//!     emit hints from the inverted colouring model: which callee-saved values are temps, named
//!     locals (with declaration order), params, or K-blocked
//! mwcc-oracle webs <file.cpp|-e CODE|file.o> [--fn PAT]...
//!     callee-saved register webs: origin (param / call result / hop...), call crossing,
//!     estimated interference degree (K-effect when >= 29 GPR / 32 FPR), interference
//! mwcc-oracle obj <file.o> [--fn PAT]... [--data]
//!     annotated disassembly of an existing object (e.g. a target object)
//! mwcc-oracle var <exp.cpp> [--full] [--data] [-j N]
//!     compile every variant; print the first listing and diffs of the others against it
//! mwcc-oracle check <exp.cpp|dir>... [-j N] [--full]
//!     run the //@ expect lines of experiment files; exit 1 on any failure (--full: print the
//!     listings of failing variants)
//! Profiles: game (default), rel, relpool, sdkc, sdk125, musyx.
//! Env: MWDEC_ROOT (project root, read-only), MWDEC_ORACLE_WORK (scratch dir).
use anyhow::{anyhow, bail, Result};
use mwdec_oracle::asm::{self, AsmOpts};
use mwdec_oracle::compile::Compiler;
use mwdec_oracle::flags::Profile;
use mwdec_oracle::variants::{self, ExpFile};
use std::path::{Path, PathBuf};

struct Args {
    cmd: String,
    inputs: Vec<String>,
    code: Option<String>,
    fns: Vec<String>,
    profile: Option<Profile>,
    version: Option<String>,
    offsets: bool,
    data: bool,
    full: bool,
    json: bool,
    all: bool,
    pcode: Vec<String>,
    fixes: Vec<(u8, u8)>,
    target: Option<String>,
    jobs: usize,
    extra: Vec<String>,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().ok_or_else(|| anyhow!("usage: mwcc-oracle asm|var|check ... (see source header)"))?;
    let mut a = Args {
        cmd,
        inputs: vec![],
        code: None,
        fns: vec![],
        profile: None,
        version: None,
        offsets: false,
        data: false,
        full: false,
        json: false,
        all: false,
        pcode: vec![],
        fixes: vec![],
        target: None,
        jobs: 6,
        extra: vec![],
    };
    while let Some(x) = it.next() {
        match x.as_str() {
            "-e" => a.code = Some(it.next().ok_or_else(|| anyhow!("-e needs code"))?),
            "--fn" => a.fns.push(it.next().ok_or_else(|| anyhow!("--fn needs pattern"))?),
            "--profile" | "-p" => {
                let p = it.next().unwrap_or_default();
                a.profile = Some(Profile::parse(&p).ok_or_else(|| anyhow!("unknown profile {p}"))?)
            }
            "--rel" => a.profile = Some(Profile::Rel),
            "--ver" => a.version = it.next(),
            "--offsets" => a.offsets = true,
            "--data" => a.data = true,
            "--full" => a.full = true,
            "--json" => a.json = true,
            "--target" => a.target = it.next(),
            "--fix" => {
                // e.g. --fix r30:r31,r31:r30  (candidate register : target register)
                for p in it.next().unwrap_or_default().split(',') {
                    let t: Vec<u8> = p.split(':').filter_map(|x| x.trim_start_matches(['r', 'f']).parse().ok()).collect();
                    if t.len() == 2 {
                        a.fixes.push((t[0], t[1]));
                    }
                }
            }
            "--all" => a.all = true,
            "--pcode" => a.pcode = it.next().unwrap_or_default().split(',').map(String::from).collect(),
            "-j" => a.jobs = it.next().and_then(|s| s.parse().ok()).unwrap_or(6).min(6),
            "--" => {
                a.extra.extend(it.by_ref());
            }
            _ => a.inputs.push(x),
        }
    }
    Ok(a)
}

fn compiler(a: &Args) -> Compiler {
    let mut c = Compiler::default();
    if let Some(p) = a.profile {
        c.profile = p;
    }
    c.version = a.version.clone();
    c.extra = a.extra.clone();
    c
}

/// Object for webs/hints: an existing `.o`, or a compiled source file / `-e` snippet.
fn load_obj(a: &Args) -> Result<asm::Obj> {
    if a.inputs.first().map_or(false, |f| f.ends_with(".o")) {
        return asm::parse(&std::fs::read(&a.inputs[0])?);
    }
    let src = match (&a.code, a.inputs.first()) {
        (Some(c), _) => c.clone(),
        (None, Some(f)) => std::fs::read_to_string(f)?,
        _ => bail!("need a file or -e CODE"),
    };
    asm::parse(&compiler(a).compile(&src)?.object)
}

fn collect_files(inputs: &[String]) -> Vec<PathBuf> {
    let mut v = vec![];
    for i in inputs {
        let p = Path::new(i);
        if p.is_dir() {
            let mut fs: Vec<PathBuf> = std::fs::read_dir(p)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| matches!(p.extension().and_then(|s| s.to_str()), Some("cpp") | Some("c")))
                .collect();
            fs.sort();
            v.extend(fs);
        } else {
            v.push(p.to_path_buf());
        }
    }
    v
}

fn main() -> Result<()> {
    mwdec_core::memcap::install();
    let a = parse_args()?;
    let opts = AsmOpts { offsets: a.offsets, literals: true };
    match a.cmd.as_str() {
        "asm" => {
            let src = match (&a.code, a.inputs.first()) {
                (Some(c), _) => c.clone(),
                (None, Some(f)) => std::fs::read_to_string(f)?,
                _ => bail!("asm: need a file or -e CODE"),
            };
            let c = compiler(&a);
            let out = c.compile(&src)?;
            if !out.messages.trim().is_empty() {
                eprintln!("{}", out.messages.trim());
            }
            let obj = asm::parse(&out.object)?;
            for l in variants::listing_for(&obj, &a.fns, opts) {
                println!("{l}");
            }
            if a.data {
                println!("# data");
                for l in asm::data_listing(&obj) {
                    println!("{l}");
                }
            }
        }
        "why" => {
            // scheduling diff explained by the real scheduler (post-RA and pre-RA picks)
            let tpath = a.target.clone().ok_or_else(|| anyhow!("why: need --target <target.o>"))?;
            let fname = a.fns.first().cloned().ok_or_else(|| anyhow!("why: need --fn <mangled name>"))?;
            let tobj = asm::parse(&std::fs::read(&tpath)?)?;
            let tf = tobj
                .funcs
                .iter()
                .find(|f| f.name == fname)
                .or_else(|| tobj.funcs.iter().find(|f| f.name.contains(fname.as_str())))
                .cloned()
                .ok_or_else(|| anyhow!("function {fname} not in target"))?;
            let src = match (&a.code, a.inputs.first()) {
                (Some(c), _) => c.clone(),
                (None, Some(f)) => std::fs::read_to_string(f)?,
                _ => bail!("why: need a candidate file or -e CODE"),
            };
            let adv = mwdec_oracle::explain::explain_sched_diff(&compiler(&a), &src, &tf.name, &tobj, &tf)?;
            if a.json {
                println!("{}", serde_json::to_string_pretty(&adv)?);
                return Ok(());
            }
            if adv.is_empty() {
                println!("no instruction-order differences");
            }
            for x in &adv {
                println!("target wants [{}] before [{}]: {}", x.second.text, x.first.text, x.text);
            }
        }
        "sched" => {
            // scheduling dependence check: candidate (source compiled with -sym on, or .o) vs target .o
            let tpath = a.target.clone().ok_or_else(|| anyhow!("sched: need --target <target.o>"))?;
            let fname = a.fns.first().cloned().ok_or_else(|| anyhow!("sched: need --fn <mangled name or substring>"))?;
            let tobj = asm::parse(&std::fs::read(&tpath)?)?;
            let mut uns: Option<asm::Obj> = None;
            let cobj = if a.inputs.first().map_or(false, |f| f.ends_with(".o")) {
                asm::parse(&std::fs::read(&a.inputs[0])?)?
            } else {
                let src = match (&a.code, a.inputs.first()) {
                    (Some(c), _) => c.clone(),
                    (None, Some(f)) => std::fs::read_to_string(f)?,
                    _ => bail!("sched: need a candidate file or -e CODE"),
                };
                // unscheduled compile with line info for statement attribution
                let mut c2 = compiler(&a);
                c2.extra.push("-sym".into());
                c2.extra.push("on".into());
                if let Ok(o) = c2.compile(&format!("#pragma scheduling off
{src}")) {
                    uns = asm::parse(&o.object).ok();
                }
                asm::parse(&compiler(&a).compile(&src)?.object)?
            };
            let pick = |o: &asm::Obj| -> Option<asm::Func> {
                o.funcs.iter().find(|f| f.name == fname).or_else(|| o.funcs.iter().find(|f| f.name.contains(fname.as_str()))).cloned()
            };
            let tf = pick(&tobj).ok_or_else(|| anyhow!("function {fname} not in target"))?;
            let cf = pick(&cobj).ok_or_else(|| anyhow!("function {fname} not in candidate"))?;
            let lines = uns.as_ref().and_then(|u| pick(u).map(|uf| mwdec_oracle::schedcheck::attribute_lines(&cobj, &cf, u, &uf, 1)));
            // exact stack-local extents from the unscheduled -sym on compile (same frame layout)
            let objs = uns.as_ref().and_then(|u| pick(u).map(|uf| mwdec_oracle::schedcheck::stack_objects(u, &uf.name)));
            let objs = objs.filter(|o| !o.is_empty());
            let rep = mwdec_oracle::schedcheck::check_with(
                &tobj,
                &tf,
                &cobj,
                &cf,
                &mwdec_oracle::schedcheck::CheckOptions {
                    lines: lines.as_deref(),
                    cand_stack: objs.as_deref(),
                    target_stack: objs.as_deref(),
                },
            );
            if a.json {
                println!("{}", serde_json::to_string_pretty(&rep)?);
                return Ok(());
            }
            if rep.structure_differs {
                println!("WARNING: basic-block structure differs (not a scheduling-only diff); comparing blocks pairwise");
            }
            for b in &rep.blocks {
                if b.inversions.is_empty() && b.unmatched_cand.is_empty() && b.unmatched_target.is_empty() {
                    continue;
                }
                println!("== block {} (candidate {:x}..{:x}, target {:x}..{:x})", b.index, b.cand_range.0, b.cand_range.1, b.target_range.0, b.target_range.1);
                for u in &b.unmatched_target {
                    println!("   only in target:    {u}");
                }
                for u in &b.unmatched_cand {
                    println!("   only in candidate: {u}");
                }
                for inv in &b.inversions {
                    let l = |r: &mwdec_oracle::schedcheck::InstrRef| r.line.map(|x| format!(" (line {x})")).unwrap_or_default();
                    let kind = match (&inv.forced, &inv.forced_in) {
                        (Some(k), Some(w)) => format!("FORCED by {k:?} dependence in {w}"),
                        _ => "priority only (independent)".into(),
                    };
                    println!(
                        "   target puts [{}]{} before [{}]{}: {}",
                        inv.cand_second.text,
                        l(&inv.cand_second),
                        inv.cand_first.text,
                        l(&inv.cand_first),
                        kind
                    );
                }
            }
            if rep.blocks.iter().all(|b| b.inversions.is_empty() && b.unmatched_cand.is_empty() && b.unmatched_target.is_empty()) {
                println!("no instruction-order differences (diff is registers/operands only)");
            }
            if !rep.statement_moves.is_empty() {
                println!("== source statements that must move (dependence-forced):");
                for (mv, before, k) in &rep.statement_moves {
                    println!("   move line {mv} before line {before} ({k:?} dependence)");
                }
            }
        }
        "inline" | "trace" | "iro" | "picks" | "copyprop" | "peephole" => {
            use mwdec_oracle::tracer::{trace_source, TraceOptions};
            let src = match (&a.code, a.inputs.first()) {
                (Some(c), _) => c.clone(),
                (None, Some(f)) => std::fs::read_to_string(f)?,
                _ => bail!("{}: need a file or -e CODE", a.cmd),
            };
            let opts = TraceOptions {
                inline: a.cmd == "inline",
                coloring: a.cmd == "trace",
                pcode: a.pcode.clone(),
                iro: a.cmd == "iro",
                iro_all_stages: a.all,
                sched: a.cmd == "picks",
                sched_filter: if a.cmd == "picks" { a.fns.first().cloned() } else { None },
                copyprop: a.cmd == "copyprop",
                peephole: a.cmd == "peephole",
                ..Default::default()
            };
            let t = trace_source(&compiler(&a), &src, &opts)?;
            let want = |f: &str| a.fns.is_empty() || a.fns.iter().any(|p| f.contains(p.as_str()));
            if a.cmd == "copyprop" {
                // per copy: every use decision; rejected uses explain why a named local stays named
                let mut last = String::new();
                for e in &t.copyprop {
                    if !want(&e.function) || (!a.all && e.accepted) {
                        continue;
                    }
                    let head = format!("{}: {}", e.function, e.copy);
                    if head != last {
                        println!("== {head}");
                        last = head;
                    }
                    println!("   {} [{}]: {}", if e.accepted { "ok  " } else { "KEEP" }, e.use_text, e.reason);
                }
                for r in &t.copyprop_removals {
                    if want(&r.function) && a.all {
                        println!("-- {}: {} {}", r.function, r.copy, if r.removed { "removed" } else { "kept (not propagated)" });
                    }
                }
                return Ok(());
            }
            if a.cmd == "peephole" {
                for h in &t.peephole {
                    if !want(&h.function) {
                        continue;
                    }
                    println!(
                        "{}: {:<32} [{}] -> {}",
                        h.function,
                        h.rule,
                        h.before,
                        h.after.as_deref().map(|x| format!("[{x}]")).unwrap_or_else(|| "removed".into())
                    );
                }
                return Ok(());
            }
            if a.cmd == "picks" {
                // list scheduler: DAG + picks with reasons (post-RA pass only unless --all)
                for b in &t.sched {
                    if !want(&b.function) || (!a.all && b.pre_ra) {
                        continue;
                    }
                    println!(
                        "== {} block {} ({} pass)",
                        b.function,
                        b.block,
                        if b.pre_ra { "pre-RA" } else { "post-RA" }
                    );
                    if a.full {
                        for (i, n) in b.nodes.iter().enumerate() {
                            let succ: Vec<String> =
                                n.succs.iter().map(|e| format!("{}:{:?}/{}", e.to, e.kind, e.latency)).collect();
                            println!(
                                "   n{i:<3} {:<34} h{:<3} dl{:<3} rank{:<3} -> {}",
                                n.text,
                                n.height,
                                n.deadline,
                                n.opcode_rank,
                                succ.join(" ")
                            );
                        }
                    }
                    for p in b.analyze() {
                        let over = p.over.map(|o| format!(" over n{o} [{}]", b.nodes[o].text)).unwrap_or_default();
                        println!("   c{:<3} n{:<3} {:<34} {:?}{over}", p.cycle, p.node, b.nodes[p.node].text, p.reason);
                    }
                }
                return Ok(());
            }
            if a.cmd == "iro" {
                // --full: the raw dump; else the rendered statements of each (filtered) function
                if a.full {
                    print!("{}", t.iro);
                    return Ok(());
                }
                for st in t.iro_stages() {
                    if !want(&st.function) || (!a.all && st.stage != mwdec_oracle::iro::FINAL_STAGE) {
                        continue;
                    }
                    println!("== {} after {}", st.function, st.stage);
                    for l in st.render() {
                        println!("{l}");
                    }
                }
                return Ok(());
            }
            if a.json {
                println!("{}", serde_json::to_string_pretty(&t)?);
                return Ok(());
            }
            if a.cmd == "inline" {
                println!("# body expansion / code generation order: {}", t.expand_order.join(", "));
                println!("# caller -> callee: decision (cost vs inline_max_size, expansion passes, reason) [checks]");
                // collapse repeated checks of the same call (one per expansion pass) unless --full
                let mut rows: Vec<(String, String, bool, Option<i32>, i32, String, i16, i16, usize)> = vec![];
                for e in &t.inline {
                    if !want(&e.caller) || (!a.all && e.reason == "not_inline_candidate") {
                        continue;
                    }
                    if !a.full {
                        if let Some(r) = rows.iter_mut().find(|r| {
                            r.0 == e.caller && r.1 == e.callee && r.2 == e.inlined && r.3 == e.cost && r.5 == e.reason
                        }) {
                            r.7 = r.7.max(e.pass);
                            r.6 = r.6.min(e.pass);
                            r.8 += 1;
                            continue;
                        }
                    }
                    rows.push((
                        e.caller.clone(),
                        e.callee.clone(),
                        e.inlined,
                        e.cost,
                        e.max,
                        e.reason.clone(),
                        e.pass,
                        e.pass,
                        1,
                    ));
                }
                for r in rows {
                    println!(
                        "{} -> {}: {} (cost {} vs {}, pass {}{}, {}) [{}]",
                        r.0,
                        r.1,
                        if r.2 { "INLINED" } else { "called" },
                        r.3.map(|c| c.to_string()).unwrap_or("-".into()),
                        r.4,
                        r.6,
                        if r.7 != r.6 { format!("-{}", r.7) } else { String::new() },
                        r.5,
                        r.8
                    );
                }
            } else {
                for r in &t.rounds {
                    if !want(&r.function) {
                        continue;
                    }
                    println!(
                        "== {} {}: vregs {}..{} (named < {}, FE temps < {}, codegen temps after)",
                        r.function,
                        r.class,
                        r.nreal,
                        r.nreal as usize + r.nodes.len(),
                        r.first_fe_temp,
                        r.first_temp
                    );
                    if !a.fixes.is_empty() {
                        for f in mwdec_oracle::tracer::register_fixes(r, &a.fixes) {
                            println!("  FIX r{} -> r{}: {}", f.from, f.to, f.suggestion);
                        }
                    }
                    let mut ns: Vec<&mwdec_oracle::tracer::IgNode> = r.nodes.iter().collect();
                    ns.sort_by_key(|n| (n.order.is_none(), n.order.unwrap_or(0), n.vreg));
                    let k = if r.class == "FPR" { 32 } else { 29 };
                    for n in ns {
                        if !a.all && n.degree == 0 {
                            continue;
                        }
                        let regn = match n.reg {
                            Some(x) => format!("{}{}", if r.class == "FPR" { "f" } else { "r" }, x),
                            None => "-".into(),
                        };
                        println!(
                            "  {:>4} v{:<4} {:<7?} {:<16} deg {:<3}{} -> {}{}{}",
                            n.order.map(|o| format!("#{o}")).unwrap_or("".into()),
                            n.vreg,
                            n.kind,
                            n.name.clone().unwrap_or_default(),
                            n.degree,
                            if n.degree as u32 >= k { "(>=K)" } else { "     " },
                            regn,
                            if n.blocked { "  [blocked: coloured early]" } else { "" },
                            n.coalesced_into.map(|c| format!("  [coalesced into {c}]")).unwrap_or_default()
                        );
                    }
                }
            }
            for p in &t.pcode {
                if !want(&p.function) {
                    continue;
                }
                println!("== PCode {} ({})", p.stage, p.function);
                for l in &p.lines {
                    println!("{l}");
                }
            }
        }
        "hints" => {
            let obj = load_obj(&a)?;
            for f in &obj.funcs {
                if !a.fns.is_empty() && !a.fns.iter().any(|p| f.name.contains(p.as_str())) {
                    continue;
                }
                let calls: std::collections::BTreeMap<u32, String> =
                    f.relocs.iter().filter(|r| r.r_type == 10).map(|r| (r.offset, r.target.clone())).collect();
                let ins = mwdec_oracle::webs::decode(&f.code, &calls);
                let (ws, per_class) = mwdec_oracle::hints::hints(&ins);
                println!("{}", asm::func_header(f));
                for (k, h) in per_class.iter().enumerate() {
                    let cls = if k == 0 { "GPR" } else { "FPR" };
                    match h {
                        None => println!("  {cls}: no consistent class assignment (spill or unmodelled effect)"),
                        Some(v) => {
                            let mut v = v.clone();
                            v.sort_by_key(|h| std::cmp::Reverse(h.reg));
                            for h in v {
                                let call = match &h.origin {
                                    mwdec_oracle::webs::Origin::CallResult { call }
                                    | mwdec_oracle::webs::Origin::CallResultHop { call }
                                    | mwdec_oracle::webs::Origin::FromCallResult { call } => {
                                        format!(" <- {}", ins[*call].call_target.clone().unwrap_or_else(|| "?".into()))
                                    }
                                    _ => String::new(),
                                };
                                let origin: String = format!("{:?}{}", h.origin, call).chars().take(40).collect();
                                println!(
                                    "  {:4} def@{:<5x} {:<40} deg~{:<3} {}",
                                    mwdec_oracle::webs::reg_name(h.reg),
                                    ins[ws[h.web].defs[0]].off,
                                    origin,
                                    h.est_degree,
                                    mwdec_oracle::hints::describe(&h.class)
                                );
                            }
                        }
                    }
                }
            }
        }
        "memclass" => {
            // the scheduler alias model's class of every memory access (source compiled with -sym on
            // so the DWARF local extents are used, or an object)
            let (obj, dwarf) = if a.inputs.first().map_or(false, |f| f.ends_with(".o")) {
                (asm::parse(&std::fs::read(&a.inputs[0])?)?, true)
            } else {
                let src = match (&a.code, a.inputs.first()) {
                    (Some(c), _) => c.clone(),
                    (None, Some(f)) => std::fs::read_to_string(f)?,
                    _ => bail!("memclass: need a file or -e CODE"),
                };
                let mut c = compiler(&a);
                c.extra.extend(["-sym".to_string(), "on".to_string()]);
                (asm::parse(&c.compile(&src)?.object)?, true)
            };
            for f in &obj.funcs {
                if !a.fns.is_empty() && !a.fns.iter().any(|p| f.name.contains(p.as_str())) {
                    continue;
                }
                let objs = if dwarf { mwdec_oracle::schedcheck::stack_objects(&obj, &f.name) } else { vec![] };
                println!("{}", asm::func_header(f));
                if !objs.is_empty() {
                    let o: Vec<String> = objs.iter().map(|o| format!("{}@0x{:x}+{}", o.name, o.off, o.size)).collect();
                    println!("    ; stack objects: {}", o.join(" "));
                }
                let cls: std::collections::BTreeMap<u32, mwdec_oracle::schedcheck::Mem> =
                    mwdec_oracle::schedcheck::memory_classes(f, if objs.is_empty() { None } else { Some(&objs) })
                        .into_iter()
                        .collect();
                for l in asm::disasm_func(&obj, f, AsmOpts { offsets: true, literals: false }) {
                    let off = l.trim().split_once(": ").and_then(|(o, _)| u32::from_str_radix(o.trim(), 16).ok());
                    match off.and_then(|o| cls.get(&o)) {
                        Some(m) => println!("{l:<48} ; {m:?}"),
                        None => println!("{l}"),
                    }
                }
            }
        }
        "webs" => {
            // callee-saved webs of each function (from a source file/-e snippet, or an object with --obj)
            let obj = if a.inputs.first().map_or(false, |f| f.ends_with(".o")) {
                asm::parse(&std::fs::read(&a.inputs[0])?)?
            } else {
                let src = match (&a.code, a.inputs.first()) {
                    (Some(c), _) => c.clone(),
                    (None, Some(f)) => std::fs::read_to_string(f)?,
                    _ => bail!("webs: need a file or -e CODE"),
                };
                asm::parse(&compiler(&a).compile(&src)?.object)?
            };
            for f in &obj.funcs {
                if !a.fns.is_empty() && !a.fns.iter().any(|p| f.name.contains(p.as_str())) {
                    continue;
                }
                let calls: std::collections::BTreeMap<u32, String> =
                    f.relocs.iter().filter(|r| r.r_type == 10).map(|r| (r.offset, r.target.clone())).collect();
                let ins = mwdec_oracle::webs::decode(&f.code, &calls);
                let ws = mwdec_oracle::webs::webs(&ins);
                let wall = mwdec_oracle::webs::webs_ext(&ins, true);
                let deg = mwdec_oracle::webs::estimate_degrees(&ins, &wall);
                println!("{}", asm::func_header(f));
                for w in &ws {
                    let d = wall.iter().position(|x| x.reg == w.reg && x.defs == w.defs).map(|k| deg[k]);
                    println!(
                        "  {:4} def {:<12} {:?} cross={} deg~{} inter=[{}]",
                        mwdec_oracle::webs::reg_name(w.reg),
                        w.defs.iter().map(|&d| format!("{:x}", ins[d].off)).collect::<Vec<_>>().join(","),
                        w.origin,
                        w.crosses_call,
                        d.map(|x| x.to_string()).unwrap_or("?".into()),
                        w.interferes.iter().map(|&j| mwdec_oracle::webs::reg_name(ws[j].reg)).collect::<Vec<_>>().join(" ")
                    );
                }
            }
        }
        "obj" => {
            // disassemble an existing object (e.g. a target object from build/G2ME01/obj)
            let f = a.inputs.first().ok_or_else(|| anyhow!("obj: need an object file"))?;
            let obj = asm::parse(&std::fs::read(f)?)?;
            for l in variants::listing_for(&obj, &a.fns, opts) {
                println!("{l}");
            }
            if a.data {
                println!("# data");
                for l in asm::data_listing(&obj) {
                    println!("{l}");
                }
            }
        }
        "var" => {
            let f = a.inputs.first().ok_or_else(|| anyhow!("var: need an experiment file"))?;
            let mut exp: ExpFile = variants::parse_exp(&std::fs::read_to_string(f)?)?;
            exp.fn_filters.extend(a.fns.iter().cloned());
            let res = variants::run_file(&compiler(&a), &exp, opts, a.jobs);
            let base = &res[0];
            for (k, r) in res.iter().enumerate() {
                println!("==================== variant {}", r.name);
                match &r.listing {
                    Err(e) => println!("COMPILE ERROR: {e}"),
                    Ok(l) => {
                        let same = res[..k].iter().find(|o| o.listing.as_ref().ok().map(|x| variants::strip_names(x)) == Some(variants::strip_names(l)));
                        if let Some(o) = same {
                            println!("(identical to {})", o.name);
                        } else if k == 0 || a.full || base.listing.is_err() {
                            for x in l {
                                println!("{x}");
                            }
                        } else {
                            println!("(diff vs {})", base.name);
                            for x in variants::diff(base.listing.as_ref().unwrap(), l) {
                                println!("{x}");
                            }
                        }
                    }
                }
                if a.data {
                    println!("# data");
                    for x in &r.data {
                        println!("{x}");
                    }
                }
            }
            let rep = variants::check(&exp, &res);
            if rep.passed + rep.failed.len() > 0 {
                println!("==================== expectations: {} passed, {} failed", rep.passed, rep.failed.len());
                for x in &rep.failed {
                    println!("{x}");
                }
            }
        }
        "check" => {
            let files = collect_files(&a.inputs);
            let (mut tp, mut tf) = (0, 0);
            for f in files {
                let exp = match variants::parse_exp(&std::fs::read_to_string(&f)?) {
                    Ok(e) => e,
                    Err(e) => {
                        println!("{}: parse error {e}", f.display());
                        tf += 1;
                        continue;
                    }
                };
                let res = variants::run_file(&compiler(&a), &exp, opts, a.jobs);
                let rep = variants::check(&exp, &res);
                if a.full && !rep.failed.is_empty() {
                    for r in &res {
                        if rep.failed.iter().any(|f| f.starts_with(&format!("[{}]", r.name))) {
                            println!("---- listing of failing variant {} ({})", r.name, f.display());
                            match &r.listing {
                                Ok(l) => l.iter().for_each(|x| println!("{x}")),
                                Err(e) => println!("{e}"),
                            }
                        }
                    }
                }
                println!(
                    "{:<52} {:>3} passed {:>3} failed",
                    f.file_name().unwrap().to_string_lossy(),
                    rep.passed,
                    rep.failed.len()
                );
                for x in &rep.failed {
                    println!("    {x}");
                }
                tp += rep.passed;
                tf += rep.failed.len();
            }
            println!("TOTAL {tp} passed, {tf} failed");
            if tf > 0 {
                std::process::exit(1);
            }
        }
        _ => bail!("unknown command {}", a.cmd),
    }
    Ok(())
}
