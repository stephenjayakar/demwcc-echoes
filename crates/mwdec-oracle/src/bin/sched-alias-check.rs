//! Validate `schedcheck`'s memory model against the real scheduler's dependence DAG: generate
//! random memory-heavy functions (pointer, struct-local, array, global, escaped/non-escaped
//! accesses), trace the real compiler's post-RA scheduler, and compare for every pair of memory
//! instructions in a block (at least one store, no register dependence between them) whether the
//! compiler put a memory edge between them with whether the asm-level model says they may alias.
//!
//! usage: sched-alias-check [N=100] [seed=1] [-v] [--dwarf]
use mwdec_oracle::asm;
use mwdec_oracle::compile::Compiler;
use mwdec_oracle::sched::EdgeKind;
use mwdec_oracle::schedcheck::{frame_offsets, memory_dependent, stack_objects};
use mwdec_oracle::tracer::{trace_source, TraceOptions};
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

const PRELUDE: &str = "struct S { int a; int b; int c; };\nvoid use(S*);\nvoid usei(int*);\nint g1; int g2; S gs;\n";

fn gen(r: &mut Rng) -> String {
    let mut body = String::new();
    let n = 4 + r.below(8);
    let val = |r: &mut Rng| -> String {
        match r.below(4) {
            0 => "x * y".into(),
            1 => "y".into(),
            2 => format!("x + {}", r.below(9)),
            _ => "v".into(),
        }
    };
    for _ in 0..n {
        let f = ["a", "b", "c"][r.below(3) as usize];
        let k = r.below(4);
        let v = val(r);
        let st = match r.below(17) {
            0 => format!("*p = {v};"),
            1 => format!("p[{k}] = {v};"),
            2 => format!("s->{f} = {v};"),
            3 => format!("ls.{f} = {v};"),
            4 => format!("arr[{k}] = {v};"),
            5 => format!("arr[i] = {v};"),
            6 => format!("g1 = {v};"),
            7 => format!("gs.{f} = {v};"),
            8 => format!("t = {v};"),
            9 => "v += *q;".into(),
            10 => format!("v += s->{f};"),
            11 => format!("v += ls.{f};"),
            12 => format!("v += arr[{k}];"),
            13 => "v += arr[i];".into(),
            14 => "v += g2;".into(),
            15 => "v += t;".into(),
            _ => format!("v += q[{k}];"),
        };
        body += "    ";
        body += &st;
        body += "\n";
    }
    let mut tail = String::new();
    if r.below(2) == 0 {
        tail += "    use(&ls);\n";
    }
    if r.below(2) == 0 {
        tail += "    usei(arr);\n";
    }
    if r.below(3) == 0 {
        tail += "    usei(&t);\n";
    }
    if r.below(3) == 0 {
        // escape in a later block
        tail = format!("    if (x) {{\n{tail}    }}\n");
    }
    format!(
        "{PRELUDE}int F(int* p, int* q, S* s, int x, int y, int i) {{\n    S ls; int arr[4]; int t; int v = 0;\n    ls.a = 0; ls.b = 0; ls.c = 0; arr[0] = 0; arr[1] = 0; arr[2] = 0; arr[3] = 0; t = 0;\n{body}{tail}    return v + ls.a + ls.b + ls.c + arr[0] + arr[1] + arr[2] + arr[3] + t;\n}}\n"
    )
}

fn is_mem(m: &str) -> bool {
    (m.starts_with('l') && !matches!(m, "li" | "lis")) || m.starts_with("st") || m.starts_with("psq_")
}

fn main() -> anyhow::Result<()> {
    mwdec_core::memcap::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    let n: usize = pos.first().and_then(|s| s.parse().ok()).unwrap_or(100);
    let seed: u64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let verbose = args.iter().any(|a| a == "-v");
    let use_dwarf = args.iter().any(|a| a == "--dwarf");
    let comp = Compiler::default();
    let mut csym = comp.clone();
    csym.extra.extend(["-sym".to_string(), "on".to_string()]);
    let mut r = Rng(0x9E3779B97F4A7C15 ^ seed.wrapping_mul(0x2545F4914F6CDD1D));
    let mut tally: BTreeMap<(bool, bool), usize> = BTreeMap::new();
    let mut examples: BTreeMap<(bool, bool), Vec<String>> = BTreeMap::new();
    let mut unaligned = 0;
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for _case in 0..n {
        let src = gen(&mut r);
        let opts = TraceOptions { sched: true, sched_filter: Some("F".into()), ..Default::default() };
        let Ok(t) = trace_source(&comp, &src, &opts) else { continue };
        let obj = asm::parse(&t.object)?;
        let Some(f) = obj.funcs.iter().find(|f| f.name.starts_with("F__")) else { continue };
        let objs = if use_dwarf {
            asm::parse(&csym.compile(&src)?.object).ok().map(|o| stack_objects(&o, &f.name))
        } else {
            None
        };
        // final memory instructions in order: (offset, mnemonic)
        let lines = asm::disasm_func(&obj, f, asm::AsmOpts { offsets: true, literals: false });
        let mut fin: Vec<(u32, String, String)> = vec![];
        for l in &lines {
            if let Some((o, txt)) = l.trim().split_once(": ") {
                let Ok(off) = u32::from_str_radix(o.trim(), 16) else { continue };
                let m = txt.split_whitespace().next().unwrap_or("").to_string();
                if is_mem(&m) {
                    fin.push((off, m, txt.trim().to_string()));
                }
            }
        }
        let frames = frame_offsets(f);
        for b in &t.sched {
            for p in b.analyze() {
                *reasons.entry(format!("{}{:?}", if b.pre_ra { "pre-RA " } else { "post-RA " }, p.reason)).or_default() += 1;
            }
        }
        let mut cursor = 0usize;
        for b in t.sched.iter().filter(|b| !b.pre_ra) {
            // map memory picks (issue order = final order) to final memory instructions
            let mut off_of: BTreeMap<usize, (u32, String)> = BTreeMap::new();
            for pk in &b.picks {
                let m = b.nodes[pk.node].text.split_whitespace().next().unwrap_or("").to_string();
                if !is_mem(&m) {
                    continue;
                }
                match fin[cursor..].iter().position(|x| x.1 == m) {
                    Some(k) => {
                        off_of.insert(pk.node, (fin[cursor + k].0, fin[cursor + k].2.clone()));
                        cursor += k + 1;
                    }
                    None => unaligned += 1,
                }
            }
            for a in 0..b.nodes.len() {
                for c in a + 1..b.nodes.len() {
                    let (Some((oa, ta)), Some((oc, tc))) = (off_of.get(&a), off_of.get(&c)) else { continue };
                    let ma = b.nodes[a].text.split_whitespace().next().unwrap_or("");
                    let mc = b.nodes[c].text.split_whitespace().next().unwrap_or("");
                    if !(ma.starts_with("st") || mc.starts_with("st") || ma.starts_with("psq_st") || mc.starts_with("psq_st")) {
                        continue;
                    }
                    // frame code (stack pointer update, LR / callee-saved saves) is not modelled
                    // frame code: post-RA instructions with worst-case alias, not part of the model
                    if frames.contains(oa) || frames.contains(oc) {
                        continue;
                    }
                    let edge = b.nodes[a].succs.iter().find(|e| e.to == c).map(|e| e.kind);
                    if matches!(edge, Some(EdgeKind::Data | EdgeKind::Anti | EdgeKind::Output)) {
                        continue; // a register relation exists; the memory part is not observable
                    }
                    let real = edge == Some(EdgeKind::Memory);
                    let model = memory_dependent(f, *oa, *oc, objs.as_deref());
                    *tally.entry((real, model)).or_default() += 1;
                    let ex = examples.entry((real, model)).or_default();
                    if real != model && ex.len() < 12 {
                        ex.push(format!("[{ta}] / [{tc}]   ({} / {})\n{src}", b.nodes[a].text, b.nodes[c].text));
                    }
                }
            }
        }
    }
    let total: usize = tally.values().sum();
    let agree = tally.get(&(true, true)).copied().unwrap_or(0) + tally.get(&(false, false)).copied().unwrap_or(0);
    println!(
        "{total} memory pairs: agree {agree} ({:.1}%); real edge & model alias {}, real edge only {}, model only {}, neither {}; unaligned picks {unaligned}",
        100.0 * agree as f64 / total.max(1) as f64,
        tally.get(&(true, true)).unwrap_or(&0),
        tally.get(&(true, false)).unwrap_or(&0),
        tally.get(&(false, true)).unwrap_or(&0),
        tally.get(&(false, false)).unwrap_or(&0),
    );
    println!("pick reasons (replay of the exact rule over the recorded DAGs): {reasons:?}");
    if verbose {
        for ((real, model), ex) in &examples {
            for e in ex {
                println!("---- real {real} model {model}: {e}");
            }
        }
    }
    Ok(())
}
