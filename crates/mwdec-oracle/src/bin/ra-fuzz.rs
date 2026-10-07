//! Fuzz-validate the register-colouring model: generate random straight-line functions whose value
//! classes are known from the source (params / named locals / temps), compile them with the real
//! compiler, recover callee-saved webs from the object code, predict registers with
//! `regalloc::predict` (using the observed interference graph) and compare.
//!
//! usage: ra-fuzz [N=200] [seed=1] [--float] [-v]
use mwdec_oracle::asm;
use mwdec_oracle::compile::Compiler;
use mwdec_oracle::regalloc::{self, Class, Node};
use mwdec_oracle::webs::{self, reg_name, Origin};
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
    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}

#[derive(Clone, Debug)]
enum Val {
    Param(usize),
    Named(usize), // x index
    Unnamed(usize), // creation index among unnamed temps (imm 300+k)
}

struct Gen {
    src: String,
    /// class per value key
    classes: BTreeMap<String, Class>,
}

/// Generate one function. Returns source and the expected class of each value key
/// ("p0", "x3", "u1").
fn gen(r: &mut Rng, float: bool) -> Gen {
    let t = if float { "float" } else { "int" };
    let get = if float { "getf" } else { "get" };
    let np = r.below(4) as usize;
    let nx = 2 + r.below(7) as usize;
    let mut body = String::new();
    // per x: uses (move?)
    let mut x_move_use = vec![false; nx];
    let mut x_used = vec![false; nx];
    let mut defined = 0usize;
    let mut temp_counter = 0u32; // codegen temp creation order
    let mut x_temp_idx = vec![0u32; nx];
    let mut unnamed: Vec<u32> = vec![]; // creation idx per unnamed
    let mut p_used_late = vec![false; np];
    let mut cnum = 1;
    let mut stmts = 0;
    // x defined by the previous statement (its call result is still in r3/f1)
    let mut just_defined: Option<usize> = None;
    while defined < nx || stmts < nx + 3 {
        stmts += 1;
        if stmts > 40 {
            break;
        }
        let choice = r.below(10);
        if defined < nx && (choice < 4 || stmts > nx + 6) {
            body.push_str(&format!("    {t} x{defined} = {get}({});\n", 100 + defined));
            x_temp_idx[defined] = temp_counter;
            temp_counter += 1;
            just_defined = Some(defined);
            defined += 1;
            continue;
        } else if choice < 6 {
            body.push_str(&format!("    {get}({});\n", 200 + stmts));
            temp_counter += 1; // discarded result still allocates a temp? (not compared)
            just_defined = None;
            continue;
        } else if defined > 0 || np > 0 {
            // a sink call with 3 args from live values; args evaluated right to left
            let mut args: Vec<String> = vec![String::new(); 3];
            let mut kinds: Vec<Option<Val>> = vec![None; 3];
            for a in 0..3 {
                let pick = r.below(10);
                if pick < 6 && defined > 0 {
                    kinds[a] = Some(Val::Named(r.below(defined as u64) as usize));
                } else if pick < 8 && np > 0 {
                    kinds[a] = Some(Val::Param(r.below(np as u64) as usize));
                } else if pick < 9 {
                    kinds[a] = Some(Val::Unnamed(0));
                }
            }
            // evaluation order right to left: unnamed call temps get creation indices then
            for a in (0..3).rev() {
                if let Some(Val::Unnamed(_)) = kinds[a] {
                    let k = unnamed.len();
                    unnamed.push(temp_counter);
                    temp_counter += 1;
                    kinds[a] = Some(Val::Unnamed(k));
                }
            }
            let has_call_arg = kinds.iter().any(|k| matches!(k, Some(Val::Unnamed(_))));
            for a in 0..3 {
                args[a] = match &kinds[a] {
                    None => format!("{cnum}"),
                    Some(Val::Named(i)) => {
                        x_used[*i] = true;
                        if r.chance(50) {
                            // Value numbering deletes `mr r3, x` when r3 still holds x (x was the
                            // previous call's result and nothing clobbered r3): not a move-use.
                            let redundant = a == 0 && just_defined == Some(*i) && !has_call_arg;
                            if !redundant {
                                x_move_use[*i] = true;
                            }
                            format!("x{i}")
                        } else {
                            cnum += 1;
                            format!("x{i} + {cnum}")
                        }
                    }
                    Some(Val::Param(j)) => {
                        p_used_late[*j] = true;
                        if r.chance(50) {
                            format!("p{j}")
                        } else {
                            cnum += 1;
                            format!("p{j} + {cnum}")
                        }
                    }
                    Some(Val::Unnamed(k)) => {
                        cnum += 1;
                        format!("{get}({}) + {cnum}", 300 + k)
                    }
                };
                cnum += 1;
            }
            body.push_str(&format!("    sink{t}({}, {}, {});\n", args[0], args[1], args[2]));
            just_defined = None;
        }
    }
    // final use of everything not yet used, after one more call
    body.push_str(&format!("    {get}(999);\n"));
    let mut tail = vec![];
    for i in 0..nx {
        if !x_used[i] {
            x_used[i] = true;
            if r.chance(50) {
                x_move_use[i] = true;
                tail.push(format!("x{i}"));
            } else {
                cnum += 1;
                tail.push(format!("x{i} + {cnum}"));
            }
        }
    }
    while tail.len() % 3 != 0 {
        cnum += 1;
        tail.push(format!("{cnum}"));
    }
    for c in tail.chunks(3) {
        body.push_str(&format!("    sink{t}({}, {}, {});\n", c[0], c[1], c[2]));
    }
    let params: Vec<String> = (0..np).map(|j| format!("{t} p{j}")).collect();
    let src = format!(
        "{t} {get}(int); void sink{t}({t}, {t}, {t});\nvoid F({}) {{\n{body}}}\n",
        params.join(", ")
    );
    let mut classes = BTreeMap::new();
    for j in 0..np {
        classes.insert(format!("p{j}"), Class::Param(j as u32));
    }
    for i in 0..nx {
        let c = if x_move_use[i] { Class::Named(i as u32) } else { Class::Temp(x_temp_idx[i]) };
        classes.insert(format!("x{i}"), c);
    }
    for (k, &ci) in unnamed.iter().enumerate() {
        classes.insert(format!("u{k}"), Class::Temp(ci));
    }
    Gen { src, classes }
}

fn main() -> anyhow::Result<()> {
    mwdec_core::memcap::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    let n: usize = pos.first().and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let float = args.iter().any(|a| a == "--float");
    let verbose = args.iter().any(|a| a == "-v");
    let comp = Compiler::default();
    let mut r = Rng(0x9E3779B97F4A7C15 ^ seed.wrapping_mul(0x2545F4914F6CDD1D));
    let (mut fn_ok, mut fn_bad, mut node_ok, mut node_bad, mut skipped) = (0, 0, 0, 0, 0);
    let (mut inf_found, mut inf_none, mut inf_cls_ok, mut inf_cls_bad, mut inf_ord_ok, mut inf_ord_bad) = (0, 0, 0, 0, 0, 0);
    for case in 0..n {
        let g = gen(&mut r, float);
        let out = match comp.compile(&g.src) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("compile error: {e}\n{}", g.src);
                skipped += 1;
                continue;
            }
        };
        let obj = asm::parse(&out.object)?;
        let f = obj.funcs.iter().find(|f| f.name.starts_with("F__")).unwrap();
        let calls: BTreeMap<u32, String> =
            f.relocs.iter().filter(|r| r.r_type == 10).map(|r| (r.offset, r.target.clone())).collect();
        let ins = webs::decode(&f.code, &calls);
        let ws = webs::webs(&ins);
        // map webs to value keys
        let mut key_of: BTreeMap<usize, String> = BTreeMap::new();
        for w in &ws {
            let key = match &w.origin {
                Origin::Param { arg_reg } => {
                    let j = if float { *arg_reg as i32 - 33 } else { *arg_reg as i32 - 3 };
                    Some(format!("p{j}"))
                }
                Origin::CallResult { call } | Origin::CallResultHop { call } | Origin::FromCallResult { call } => {
                    match webs::call_imm_arg(&ins, *call) {
                        Some(v) if (100..200).contains(&v) => Some(format!("x{}", v - 100)),
                        Some(v) if (300..400).contains(&v) => Some(format!("u{}", v - 300)),
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(k) = key {
                key_of.insert(w.id, k);
            }
        }
        // nodes = mapped webs that are callee-saved
        let mapped: Vec<usize> = ws.iter().filter(|w| key_of.contains_key(&w.id)).map(|w| w.id).collect();
        if mapped.len() != ws.len() || mapped.is_empty() {
            if verbose {
                eprintln!("case {case}: {} webs, {} mapped; skipping", ws.len(), mapped.len());
                for w in ws.iter().filter(|w| !key_of.contains_key(&w.id)) {
                    eprintln!("   unmapped {} {:?} def {}", reg_name(w.reg), w.origin, ins[w.defs[0]].text);
                }
            }
            skipped += 1;
            continue;
        }
        let local: BTreeMap<usize, usize> = mapped.iter().enumerate().map(|(k, &w)| (w, k)).collect();
        // K-degree effect: estimate each value's full IG degree from the all-register web graph
        let wall = webs::webs_ext(&ins, true);
        let deg_all = webs::estimate_degrees(&ins, &wall);
        let use_deg = !args.iter().any(|a| a == "--no-degree");
        let mut nodes = vec![];
        let mut bad_key = false;
        for &w in &mapped {
            let Some(&class) = g.classes.get(&key_of[&w]) else {
                bad_key = true;
                break;
            };
            let interferes: Vec<usize> = ws[w].interferes.iter().filter_map(|j| local.get(j).copied()).collect();
            let est = wall
                .iter()
                .position(|x| x.reg == ws[w].reg && x.defs == ws[w].defs)
                .map(|k| deg_all[k])
                .unwrap_or(0);
            let base = interferes.len() as u32 + if ws[w].crosses_call { if float { 14 } else { 11 } } else { 0 };
            nodes.push(Node {
                class,
                float,
                crosses_call: ws[w].crosses_call,
                interferes,
                extra_degree: if use_deg { est.saturating_sub(base) } else { 0 },
            });
        }
        if bad_key {
            skipped += 1;
            continue;
        }
        // inverse check: do the emit hints recover the true classes / declaration order?
        {
            let (hws, per_class) = mwdec_oracle::hints::hints(&ins);
            let idx = if float { 1 } else { 0 };
            match &per_class[idx] {
                None => inf_none += 1,
                Some(hs) => {
                    inf_found += 1;
                    let mut named_pairs: Vec<(u32, u32)> = vec![]; // (true decl, inferred decl)
                    for h in hs {
                        let Some(key) = key_of.get(&h.web) else { continue };
                        let _ = &hws;
                        let Some(truth) = g.classes.get(key) else { continue };
                        let ok = matches!(
                            (truth, &h.class),
                            (Class::Temp(_), regalloc::InferredClass::Temp { .. })
                                | (Class::Named(_), regalloc::InferredClass::Named { .. })
                                | (Class::Param(_), regalloc::InferredClass::Param(_))
                        );
                        if ok {
                            inf_cls_ok += 1;
                        } else {
                            inf_cls_bad += 1;
                        }
                        if let (Class::Named(td), regalloc::InferredClass::Named { decl }) = (truth, &h.class) {
                            named_pairs.push((*td, *decl));
                        }
                    }
                    for a in 0..named_pairs.len() {
                        for b in a + 1..named_pairs.len() {
                            let (ta, ia) = named_pairs[a];
                            let (tb, ib) = named_pairs[b];
                            if (ta < tb) == (ia < ib) {
                                inf_ord_ok += 1;
                            } else {
                                inf_ord_bad += 1;
                            }
                        }
                    }
                }
            }
        }
        let pred = regalloc::predict(&nodes);
        let mut all = true;
        let mut lines = vec![];
        for (k, &w) in mapped.iter().enumerate() {
            let ok = pred[k] == Some(ws[w].reg);
            if ok {
                node_ok += 1;
            } else {
                node_bad += 1;
                all = false;
            }
            lines.push(format!(
                "  {:4} {:?} actual {} predicted {}{}",
                key_of[&w],
                nodes[k].class,
                reg_name(ws[w].reg),
                pred[k].map(reg_name).unwrap_or("-".into()),
                if ok { "" } else { "   <-- MISMATCH" }
            ));
        }
        if all {
            fn_ok += 1;
        } else {
            fn_bad += 1;
        }
        if verbose || !all {
            println!("---- case {case} {}", if all { "OK" } else { "MISMATCH" });
            if !all {
                print!("{}", g.src);
                for l in asm::disasm_func(&obj, f, Default::default()) {
                    println!("{l}");
                }
            }
            for l in lines {
                println!("{l}");
            }
        }
    }
    println!(
        "functions: {fn_ok} ok, {fn_bad} mismatched, {skipped} skipped; values: {node_ok} ok, {node_bad} mismatched"
    );
    println!(
        "inverse (hints): consistent assignment found for {inf_found} functions, none for {inf_none}; \
         class agrees with source truth {inf_cls_ok}, differs {inf_cls_bad}; named decl-order pairs agree {inf_ord_ok}, differ {inf_ord_bad}"
    );
    Ok(())
}
