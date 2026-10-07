//! Run the search API (`advice`) over a list of real candidates and measure whether its top
//! statement move helps: for each candidate, the order diagnosis and peephole hits against the
//! target, then the candidate with the top move applied, compiled normally and compared.
//!
//! Input: a TSV with one candidate per line:
//! `id <TAB> symbol <TAB> candidate.cpp <TAB> target.o <TAB> cflags file (one flag per line) <TAB> context file`
//!
//! usage: advice-scan list.tsv [--root PROJECT_ROOT]
use mwdec_oracle::advice::{peepholes_fired, statement_moves, Candidate};
use mwdec_oracle::asm;
use mwdec_oracle::compile::Compiler;
use std::time::Instant;

fn listing(obj: &[u8], sym: &str) -> Option<Vec<String>> {
    let o = asm::parse(obj).ok()?;
    let f = o.funcs.iter().find(|f| f.name == sym)?;
    Some(asm::disasm_func(&o, f, asm::AsmOpts { offsets: false, literals: true }))
}

/// Differing instruction positions (plus the length difference).
fn distance(a: &[String], b: &[String]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// `src` with 1-based line `line` moved to just before line `before`.
fn move_line(src: &str, line: u32, before: u32) -> Option<String> {
    let mut lines: Vec<&str> = src.lines().collect();
    let (l, b) = (line as usize - 1, before as usize - 1);
    if l >= lines.len() || b >= lines.len() || l == b {
        return None;
    }
    let x = lines.remove(l);
    let b = if b > l { b - 1 } else { b };
    lines.insert(b, x);
    Some(lines.join("\n") + "\n")
}

fn main() -> anyhow::Result<()> {
    mwdec_core::memcap::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let list = args.first().ok_or_else(|| anyhow::anyhow!("usage: advice-scan list.tsv [--root R]"))?;
    let mut comp0 = Compiler::default();
    if let Some(r) = args.iter().position(|a| a == "--root").and_then(|i| args.get(i + 1)) {
        comp0.root = r.into();
    }
    let (mut n, mut with_diff, mut with_moves, mut helped, mut same, mut worse, mut failed) = (0, 0, 0, 0, 0, 0, 0);
    let (mut not_steerable, mut peep_missed, mut errors) = (0, 0, 0);
    let mut t_moves = 0.0f64;
    for line in std::fs::read_to_string(list)?.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 6 {
            continue;
        }
        let (id, sym) = (f[0], f[1]);
        let (Ok(src), Ok(target), Ok(cflags), Ok(context)) =
            (std::fs::read_to_string(f[2]), std::fs::read(f[3]), std::fs::read_to_string(f[4]), std::fs::read_to_string(f[5]))
        else {
            println!("{id} {sym}: input missing");
            continue;
        };
        let cflags: Vec<String> = cflags.lines().map(str::to_string).collect();
        let comp = Compiler { cflags: Some(cflags), ..comp0.clone() };
        n += 1;
        let cand = Candidate { context: &context, src: &src, symbol: sym };
        let Some(tl) = listing(&target, sym) else {
            println!("{id} {sym}: not in the target");
            continue;
        };
        let base = match comp.compile(&cand.tu()).map(|o| listing(&o.object, sym)) {
            Ok(Some(l)) => l,
            _ => {
                println!("{id} {sym}: candidate does not compile");
                errors += 1;
                continue;
            }
        };
        let d0 = distance(&base, &tl);
        let t = Instant::now();
        let diag = match statement_moves(&comp, &cand, &target, sym) {
            Ok(d) => d,
            Err(e) => {
                println!("{id} {sym}: statement_moves failed: {e:#}");
                errors += 1;
                continue;
            }
        };
        t_moves += t.elapsed().as_secs_f64();
        let peep = peepholes_fired(&comp, &cand, Some((&target, sym))).unwrap_or_default();
        let missed = peep.iter().filter(|p| p.missed_in_target()).count();
        if missed > 0 {
            peep_missed += 1;
        }
        if !diag.in_order() {
            with_diff += 1;
        }
        if diag.order_not_steerable() {
            not_steerable += 1;
        }
        let mut verdict = String::from("-");
        if let Some(m) = diag.moves.first() {
            with_moves += 1;
            verdict = match move_line(&src, m.line, m.before).map(|s| comp.compile(&format!("{context}\n{s}"))) {
                Some(Ok(o)) => match listing(&o.object, sym) {
                    Some(l) => {
                        let d1 = distance(&l, &tl);
                        if d1 < d0 {
                            helped += 1;
                        } else if d1 == d0 {
                            same += 1;
                        } else {
                            worse += 1;
                        }
                        format!("{d0} -> {d1}")
                    }
                    None => {
                        failed += 1;
                        "moved: function missing".into()
                    }
                },
                _ => {
                    failed += 1;
                    "moved: does not compile".into()
                }
            };
        }
        println!(
            "{id} {sym}: diff {d0}; order diffs {} (moves {}, data {}, regalloc {}, priority {}, unexplained {}); top move {:?}: {verdict}; peephole hits {} (missed in target {missed})",
            diag.details.len(),
            diag.moves.len(),
            diag.data_dependent,
            diag.regalloc,
            diag.priority,
            diag.unexplained,
            diag.moves.first().map(|m| (m.line, m.before, format!("{:?}", m.reason))),
            peep.len()
        );
    }
    println!(
        "{n} candidates: {with_diff} with order differences, {with_moves} with a statement move (top move: {helped} closer, {same} same, {worse} further, {failed} broke), {not_steerable} order-not-steerable, {peep_missed} with a peephole missed in the target, {errors} errors; statement_moves {:.2} s avg",
        t_moves / (n.max(1) as f64)
    );
    Ok(())
}

