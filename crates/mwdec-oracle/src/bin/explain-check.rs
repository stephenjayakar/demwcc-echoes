//! Robustness check of `explain::explain_sched_diff`: random target functions, candidates that
//! swap two adjacent statements; tally the verdicts and how many order differences could not be
//! located in the recorded schedules.
//!
//! usage: explain-check [N=50] [seed=1] [-v]
use mwdec_oracle::asm;
use mwdec_oracle::compile::Compiler;
use mwdec_oracle::explain::{explain_sched_diff, Verdict};
use std::collections::BTreeMap;

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

fn stmt(r: &mut Rng) -> String {
    let f = ["a", "b", "c"][r.below(3) as usize];
    let k = r.below(4);
    let v = match r.below(4) {
        0 => "x * y".to_string(),
        1 => "y".to_string(),
        2 => format!("x + {}", r.below(9)),
        _ => "v".to_string(),
    };
    match r.below(12) {
        0 => format!("*p = {v};"),
        1 => format!("p[{k}] = {v};"),
        2 => format!("s->{f} = {v};"),
        3 => format!("ls.{f} = {v};"),
        4 => format!("arr[{k}] = {v};"),
        5 => format!("g1 = {v};"),
        6 => "v += *q;".into(),
        7 => format!("v += s->{f};"),
        8 => format!("v += ls.{f};"),
        9 => format!("v += arr[{k}];"),
        10 => format!("v = v * {} + x;", 2 + r.below(5)),
        _ => format!("v += q[{k}];"),
    }
}

fn main() -> anyhow::Result<()> {
    mwdec_core::memcap::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    let n: usize = pos.first().and_then(|s| s.parse().ok()).unwrap_or(50);
    let seed: u64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let verbose = args.iter().any(|a| a == "-v");
    let comp = Compiler::default();
    let mut r = Rng(0x9E3779B97F4A7C15 ^ seed.wrapping_mul(0x2545F4914F6CDD1D));
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    let (mut cases, mut same) = (0, 0);
    let pre = "struct S { int a; int b; int c; };\nvoid use(S*);\nint g1;\n";
    for _ in 0..n {
        let k = 4 + r.below(6) as usize;
        let body: Vec<String> = (0..k).map(|_| stmt(&mut r)).collect();
        let sw = r.below(k as u64 - 1) as usize;
        let mut cbody = body.clone();
        cbody.swap(sw, sw + 1);
        let tail = if r.below(2) == 0 { "    use(&ls);\n" } else { "" };
        let mk = |b: &[String]| {
            format!(
                "{pre}int F(int* p, int* q, S* s, int x, int y) {{\n    S ls; int arr[4]; int v = 0;\n    ls.a = 0; ls.b = 0; ls.c = 0; arr[0] = 0; arr[1] = 0; arr[2] = 0; arr[3] = 0;\n{}\n{tail}    return v + ls.a + ls.b + ls.c + arr[0] + arr[1] + arr[2] + arr[3];\n}}\n",
                b.iter().map(|s| format!("    {s}")).collect::<Vec<_>>().join("\n")
            )
        };
        let (tsrc, csrc) = (mk(&body), mk(&cbody));
        let Ok(to) = comp.compile(&tsrc) else { continue };
        let Ok(co) = comp.compile(&csrc) else { continue };
        let to = asm::parse(&to.object)?;
        let co = asm::parse(&co.object)?;
        let tf = to.funcs.iter().find(|f| f.name.starts_with("F__")).unwrap().clone();
        let cf = co.funcs.iter().find(|f| f.name.starts_with("F__")).unwrap();
        if tf.code == cf.code {
            same += 1;
            continue;
        }
        cases += 1;
        let adv = explain_sched_diff(&comp, &csrc, &tf.name, &to, &tf)?;
        for a in &adv {
            let key = match &a.verdict {
                Verdict::Forced(_) => "forced".to_string(),
                Verdict::RegAlloc => "regalloc".into(),
                Verdict::SwapStatements(..) => "swap statements".into(),
                Verdict::WaitFor(..) => "wait for".into(),
                Verdict::Priority(p) => format!("priority {p:?}"),
                Verdict::Unknown => format!("unknown: {}", a.text),
            };
            *tally.entry(key).or_default() += 1;
        }
        if verbose {
            println!("---- swapped statements {} and {}:\n{csrc}", sw + 1, sw + 2);
            for a in &adv {
                println!("   [{}] / [{}]: {}", a.first.text, a.second.text, a.text);
            }
        }
    }
    println!("{cases} cases with a code difference ({same} swaps changed nothing); verdicts per order difference: {tally:#?}");
    Ok(())
}
