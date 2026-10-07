//! Test variants of the list scheduler's pick rule against the real compiler: generate random
//! straight-line functions (integer and float arithmetic, loads, stores, constants), record every
//! pick of both scheduling passes with the scheduler probe, and count per pass how often the rule
//! with and without the opcode-rank tie-break agrees with the recorded pick, or is contradicted
//! (the node the variant prefers issued later in the same cycle, so the unit model accepted it).
//!
//! usage: sched-rule-check [N=200] [seed=1] [--show]
use mwdec_oracle::compile::Compiler;
use mwdec_oracle::sched::RuleEvidence;
use mwdec_oracle::tracer::{trace_source, TraceOptions};

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

const PRELUDE: &str = "struct S { int a, b, c, d; float f, g, h; short s; unsigned char u; };\nint use4(int, int, int, float);\nint g1; float gf;\n";

fn gen(r: &mut Rng) -> String {
    let mut body = String::new();
    let mut ints = vec!["x".to_string(), "y".to_string()];
    let mut flts = vec!["k".to_string()];
    let n = 6 + r.below(14);
    for i in 0..n {
        let pick = |r: &mut Rng, v: &Vec<String>| v[r.below(v.len() as u64) as usize].clone();
        match r.below(10) {
            0 | 1 => {
                let f = ["a", "b", "c", "d", "s", "u"][r.below(6) as usize];
                let p = if r.below(2) == 0 { "p" } else { "q" };
                body.push_str(&format!("    int i{i} = {p}->{f};\n"));
                ints.push(format!("i{i}"));
            }
            2 | 3 => {
                let op = ["+", "-", "*", "&", "|", "^", "<<", ">>"][r.below(8) as usize];
                let a = pick(r, &ints);
                let b = if r.below(3) == 0 { format!("{}", 1 + r.below(30)) } else { pick(r, &ints) };
                let b = if op == "<<" || op == ">>" { format!("({b} & 7)") } else { b };
                body.push_str(&format!("    int i{i} = {a} {op} {b};\n"));
                ints.push(format!("i{i}"));
            }
            4 => {
                body.push_str(&format!("    int i{i} = {};\n", r.below(2000) as i64 - 1000));
                ints.push(format!("i{i}"));
            }
            5 => {
                let f = ["f", "g", "h"][r.below(3) as usize];
                let p = if r.below(2) == 0 { "p" } else { "q" };
                body.push_str(&format!("    float f{i} = {p}->{f};\n"));
                flts.push(format!("f{i}"));
            }
            6 => {
                let op = ["+", "-", "*"][r.below(3) as usize];
                let a = pick(r, &flts);
                let b = if r.below(3) == 0 { format!("{}.5f", r.below(9)) } else { pick(r, &flts) };
                body.push_str(&format!("    float f{i} = {a} {op} {b};\n"));
                flts.push(format!("f{i}"));
            }
            7 | 8 => {
                let f = ["a", "b", "c", "d"][r.below(4) as usize];
                let p = if r.below(2) == 0 { "p" } else { "q" };
                body.push_str(&format!("    {p}->{f} = {};\n", pick(r, &ints)));
            }
            _ => {
                let f = ["f", "g", "h"][r.below(3) as usize];
                let p = if r.below(2) == 0 { "p" } else { "q" };
                body.push_str(&format!("    {p}->{f} = {};\n", pick(r, &flts)));
            }
        }
    }
    // a call after the straight-line code gives the function a frame: the code before it is in
    // the block merged with the prologue, which the post-RA pass schedules
    if r.below(2) == 0 {
        let a: Vec<String> = ints.iter().rev().take(3).cloned().collect();
        body.push_str(&format!("    x = use4({}, {}, {}, {});\n", a[0], a.get(1).unwrap_or(&a[0]), a.get(2).unwrap_or(&a[0]), flts.last().unwrap()));
        ints.push("x".into());
    }
    let ri = ints.iter().rev().take(3).cloned().collect::<Vec<_>>().join(" + ");
    let rf = flts.last().unwrap().clone();
    format!("{PRELUDE}int F(S* p, S* q, int x, int y, float k) {{\n{body}    gf = {rf};\n    return {ri};\n}}\n")
}

fn main() -> anyhow::Result<()> {
    mwdec_core::memcap::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let show = args.iter().any(|a| a == "--show");
    let comp = Compiler::default();
    let mut r = Rng(0x9E3779B97F4A7C15 ^ seed.wrapping_mul(0x2545F4914F6CDD1D));
    // [pass][rank]: pass 0 = pre-RA, 1 = post-RA; rank 0 = off, 1 = on
    let mut ev = [[RuleEvidence::default(); 2]; 2];
    let mut traced = 0;
    for _ in 0..n {
        let src = gen(&mut r);
        let opts = TraceOptions { sched: true, sched_filter: Some("F".into()), ..Default::default() };
        let Ok(t) = trace_source(&comp, &src, &opts) else { continue };
        traced += 1;
        let mut contra = [[0u32; 2]; 2];
        for b in &t.sched {
            let pass = if b.pre_ra { 0 } else { 1 };
            for rank in 0..2 {
                let e = b.rule_evidence(rank == 1);
                contra[pass][rank] += e.contradict;
                ev[pass][rank] += e;
            }
        }
        if show && (contra[0][0] > 0 || contra[1][1] > 0) {
            println!("--- contradicts pre-RA without rank: {}, post-RA with rank: {}\n{src}", contra[0][0], contra[1][1]);
        }
    }
    println!("{traced} functions traced");
    for (pass, name) in [(0, "pre-RA"), (1, "post-RA")] {
        for (rank, rn) in [(0, "without opcode rank"), (1, "with opcode rank")] {
            let e = ev[pass][rank];
            println!("{name:8} {rn:20}: agree {:6}  contradicted {:5}  other {:5}", e.agree, e.contradict, e.other);
        }
    }
    Ok(())
}
