//! Validate the simplify / spill-candidate model against the real compiler: generate random
//! functions with high register pressure (so simplify blocks and has to push spill candidates),
//! trace them, and replay `regalloc::simplify_order` on the traced interference graph with
//! (a) uniform costs (highest current degree first) and (b) the compiler's spill costs
//! (lowest cost/degree first, highest vreg on ties). Reports how often each rule reproduces the
//! exact colouring order of the real compiler.
//!
//! usage: ra-simplify [N=100] [seed=1] [--gpr] [-v]
use mwdec_oracle::compile::Compiler;
use mwdec_oracle::regalloc::{simplify_order, FPR_K, GPR_K};
use mwdec_oracle::tracer::{trace_source, ColorRound, TraceOptions};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

fn gen(r: &mut Rng, gpr: bool) -> String {
    let t = if gpr { "int" } else { "float" };
    let get = if gpr { "get" } else { "getf" };
    let nlong = 6 + r.below(if gpr { 14 } else { 12 }) as usize;
    let np = r.below(4) as usize;
    let mut s = format!("{t} {get}(int);\nvoid sink({t}, {t}, {t});\nvoid sinkp({t}*);\n");
    let params: Vec<String> = (0..np).map(|i| format!("{t} p{i}")).collect();
    s += &format!("void F({}) {{\n", params.join(", "));
    let mut vals: Vec<String> = (0..np).map(|i| format!("p{i}")).collect();
    for i in 0..nlong {
        s += &format!("    {t} x{i} = {get}({});\n", 100 + i);
        vals.push(format!("x{i}"));
    }
    // a few expressions with many temps while everything is live, then calls using all values
    let nexpr = 2 + r.below(5) as usize;
    for e in 0..nexpr {
        let terms = 3 + r.below(8) as usize;
        let mut ex = String::new();
        for k in 0..terms {
            let a = &vals[r.below(vals.len() as u64) as usize];
            let b = &vals[r.below(vals.len() as u64) as usize];
            if k > 0 {
                ex += if r.below(2) == 0 { " + " } else { " - " };
            }
            ex += &format!("{a} * {b}");
        }
        s += &format!("    {t} e{e} = {ex};\n");
        vals.push(format!("e{e}"));
        if r.below(3) == 0 {
            s += &format!("    sinkp(&e{e});\n");
        }
    }
    // keep everything alive across a final sequence of calls
    let mut i = 0;
    while i < vals.len() {
        let a = &vals[i];
        let b = vals.get(i + 1).unwrap_or(a);
        let c = vals.get(i + 2).unwrap_or(a);
        s += &format!("    sink({a}, {b}, {c});\n");
        i += 3;
    }
    let a = &vals[r.below(vals.len() as u64) as usize];
    s += &format!("    sink({a}, {}, {});\n", vals[0], vals[vals.len() - 1]);
    s += "}\n";
    s
}

/// Replay simplify on a traced round; returns colouring order (vregs) for uniform / real costs.
fn replay(round: &ColorRound, k: i64, use_costs: bool) -> Vec<u16> {
    let nodes: Vec<_> = round.nodes.iter().filter(|n| n.coalesced_into.is_none() && n.order.is_some()).collect();
    let idx: std::collections::HashMap<u16, usize> = nodes.iter().enumerate().map(|(i, n)| (n.vreg, i)).collect();
    let adj: Vec<Vec<usize>> = nodes
        .iter()
        .map(|n| n.neighbours.iter().filter_map(|&v| if v >= 0 { idx.get(&(v as u16)).copied() } else { None }).collect())
        .collect();
    let deg: Vec<i64> = nodes.iter().map(|n| n.simplify_degree as i64).collect();
    let num: Vec<u32> = nodes.iter().map(|n| n.vreg as u32).collect();
    let costs: Vec<f64> = nodes.iter().map(|n| n.spill_cost as f64).collect();
    let push = simplify_order(&adj, &deg, &num, k, if use_costs { Some(&costs) } else { None });
    push.iter().rev().map(|&i| nodes[i].vreg).collect()
}

fn main() -> anyhow::Result<()> {
    mwdec_core::memcap::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    let n: usize = pos.first().and_then(|s| s.parse().ok()).unwrap_or(100);
    let seed: u64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let gpr = args.iter().any(|a| a == "--gpr");
    let verbose = args.iter().any(|a| a == "-v");
    let comp = Compiler::default();
    let mut r = Rng(0x9E3779B97F4A7C15 ^ seed.wrapping_mul(0x2545F4914F6CDD1D));
    let (mut rounds, mut blocked_rounds, mut ok_uniform, mut ok_cost, mut skipped) = (0, 0, 0, 0, 0);
    let (mut ok_uniform_all, mut ok_cost_all) = (0, 0);
    for case in 0..n {
        let src = gen(&mut r, gpr);
        let t = match trace_source(&comp, &src, &TraceOptions { coloring: true, ..Default::default() }) {
            Ok(t) => t,
            Err(e) => {
                if verbose {
                    eprintln!("case {case}: {e}");
                }
                skipped += 1;
                continue;
            }
        };
        for round in &t.rounds {
            let k = if round.class == "FPR" { FPR_K } else { GPR_K } as i64;
            let mut real: Vec<(usize, u16)> = round.nodes.iter().filter_map(|n| Some((n.order?, n.vreg))).collect();
            real.sort();
            let real: Vec<u16> = real.into_iter().map(|x| x.1).collect();
            if real.is_empty() {
                continue;
            }
            rounds += 1;
            let blocked = round.nodes.iter().any(|n| n.blocked);
            let u = replay(round, k, false);
            let c = replay(round, k, true);
            if u == real {
                ok_uniform_all += 1;
            }
            if c == real {
                ok_cost_all += 1;
            }
            if !blocked {
                continue;
            }
            blocked_rounds += 1;
            if u == real {
                ok_uniform += 1;
            }
            if c == real {
                ok_cost += 1;
            }
            if verbose && c != real {
                println!("---- case {case} {} round mismatch\n{src}", round.class);
                println!("  real    {real:?}\n  cost    {c:?}\n  uniform {u:?}");
                for nd in &round.nodes {
                    if nd.order.is_some() {
                        println!(
                            "   v{} deg {} sdeg {} cost {} remat {} blocked {} order {:?}",
                            nd.vreg, nd.degree, nd.simplify_degree, nd.spill_cost, nd.remat, nd.blocked, nd.order
                        );
                    }
                }
            }
        }
    }
    println!(
        "{rounds} colouring rounds ({skipped} cases skipped); exact colouring order: uniform-cost rule {ok_uniform_all}, cost/degree rule {ok_cost_all}"
    );
    println!(
        "rounds with blocked nodes: {blocked_rounds}; exact order: uniform-cost rule {ok_uniform} ({:.1}%), cost/degree rule {ok_cost} ({:.1}%)",
        100.0 * ok_uniform as f64 / blocked_rounds.max(1) as f64,
        100.0 * ok_cost as f64 / blocked_rounds.max(1) as f64
    );
    Ok(())
}
