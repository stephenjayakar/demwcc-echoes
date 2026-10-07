//! Source-free check of the simplify spill-candidate rule on the ORIGINAL game code (target objects
//! of TRAIN-split C++ game units): for every function where the estimated interference graph
//! blocks simplify (some call-crossing callee-saved value keeps degree >= K after all low-degree
//! nodes are pushed), predict the colouring order of the blocked values with
//!   (a) the uniform-cost rule (highest current degree pushed first), and
//!   (b) GC/2.7's rule (lowest spill cost / current degree pushed first, highest vreg on ties),
//! with spill costs estimated from the code (uses 2w, defs w, loop weight 8^depth; rematerialisable
//! and param copy-in definitions subtract), and check whether some colouring order that respects
//! the prediction reproduces the observed registers.
//! usage: ra-spill-scan [--fpr] [--limit N] [-v]
use mwdec_oracle::asm;
use mwdec_oracle::regalloc::{simplify_order, solve_order_constrained, Class, Node, FPR_K, GPR_K};
use mwdec_oracle::webs::{self, reg_name, Origin};
use std::collections::{BTreeMap, BTreeSet};

fn split_of(unit: &str) -> &'static str {
    let mut h: u32 = 0x811c9dc5;
    for b in unit.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    if h % 10 < 7 {
        "train"
    } else {
        "test"
    }
}

fn train_units() -> anyhow::Result<Vec<(String, String)>> {
    let text = std::fs::read_to_string(mwdec_core::paths::project_root().join("objdiff.json"))?;
    let mut units = vec![];
    let mut rest = text.as_str();
    while let Some(p) = rest.find("\"name\": \"") {
        rest = &rest[p + 9..];
        let name = &rest[..rest.find('"').unwrap()];
        let Some(tp) = rest.find("\"target_path\": \"") else { break };
        let next_name = rest.find("\"name\": \"").unwrap_or(usize::MAX);
        if tp > next_name {
            continue;
        }
        let r2 = &rest[tp + 16..];
        let target = &r2[..r2.find('"').unwrap()];
        let block = &rest[..next_name.min(rest.len())];
        if block.contains("-lang=c++") && block.contains("deferred,noauto") && split_of(name) == "train" {
            units.push((name.to_string(), target.to_string()));
        }
    }
    Ok(units)
}

/// Loop nesting depth of each instruction: number of backward-branch ranges containing it.
fn loop_depth(ins: &[webs::Instr]) -> Vec<u32> {
    let mut ranges = vec![];
    for (i, x) in ins.iter().enumerate() {
        if x.is_call || !x.ins.is_branch() {
            continue;
        }
        if let Some(d) = x.ins.branch_dest(x.off) {
            let t = d as usize / 4;
            if t <= i {
                ranges.push((t, i));
            }
        }
    }
    (0..ins.len()).map(|i| ranges.iter().filter(|&&(s, e)| s <= i && i <= e).count() as u32).collect()
}

fn main() -> anyhow::Result<()> {
    mwdec_core::memcap::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fpr = args.iter().any(|a| a == "--fpr");
    let verbose = args.iter().any(|a| a == "-v");
    let limit: usize =
        args.iter().position(|a| a == "--limit").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let k: i64 = if fpr { FPR_K } else { GPR_K } as i64;
    let units = train_units()?;
    eprintln!("{} train C++ game units", units.len());
    let (mut nfn, mut blocked_fns, mut differ, mut ok_u, mut ok_c, mut ok_u_d, mut ok_c_d) = (0, 0, 0, 0, 0, 0, 0);
    for (name, target) in units.iter().take(limit) {
        let Ok(bytes) = std::fs::read(mwdec_core::paths::project_root().join(target)) else { continue };
        let Ok(obj) = asm::parse(&bytes) else { continue };
        for f in &obj.funcs {
            if f.code.len() < 8 {
                continue;
            }
            let calls: BTreeMap<u32, String> =
                f.relocs.iter().filter(|r| r.r_type == 10).map(|r| (r.offset, r.target.clone())).collect();
            let ins = webs::decode(&f.code, &calls);
            let wall = webs::webs_ext(&ins, true);
            let deg_all = webs::estimate_degrees(&ins, &wall);
            // simplify nodes: class webs not pinned to a physical register by a call result
            let cls: Vec<usize> = (0..wall.len())
                .filter(|&i| (wall[i].reg >= 32) == fpr && !wall[i].defs.iter().any(|&d| ins[d].is_call))
                .collect();
            let sel: Vec<usize> =
                cls.iter().copied().filter(|&i| wall[i].crosses_call && webs::is_callee_saved(wall[i].reg)).collect();
            if sel.len() < 2 {
                continue;
            }
            nfn += 1;
            let local: BTreeMap<usize, usize> = cls.iter().enumerate().map(|(x, &i)| (i, x)).collect();
            let adj: Vec<Vec<usize>> =
                cls.iter().map(|&i| wall[i].interferes.iter().filter_map(|j| local.get(j).copied()).collect()).collect();
            let deg: Vec<i64> = cls.iter().map(|&i| deg_all[i] as i64).collect();
            // blocked set: what remains after repeatedly pushing degree < K nodes
            let mut d = deg.clone();
            let mut gone = vec![false; cls.len()];
            loop {
                let mut any = false;
                for x in 0..cls.len() {
                    if !gone[x] && d[x] < k {
                        gone[x] = true;
                        any = true;
                        for &y in &adj[x] {
                            d[y] -= 1;
                        }
                    }
                }
                if !any {
                    break;
                }
            }
            let blocked: BTreeSet<usize> = (0..cls.len()).filter(|&x| !gone[x]).collect();
            if !sel.iter().any(|i| blocked.contains(&local[i])) {
                continue;
            }
            blocked_fns += 1;
            // approximate vreg numbering: params by argument register, then by first definition
            let num: Vec<u32> = cls
                .iter()
                .map(|&i| match wall[i].origin {
                    Origin::Param { arg_reg } => arg_reg as u32,
                    _ => 100 + wall[i].defs.iter().copied().min().unwrap_or(0) as u32,
                })
                .collect();
            let depth = loop_depth(&ins);
            let w = |i: usize| 8i64.pow(depth[i].min(4));
            let costs: Vec<f64> = cls
                .iter()
                .map(|&i| {
                    let web = &wall[i];
                    let remat = web.defs.len() == 1 && {
                        let t = &ins[web.defs[0]].text;
                        t.starts_with("li ") || t.starts_with("lis ") || t.contains(", r1, ")
                    };
                    let arg_init = matches!(web.origin, Origin::Param { .. });
                    let u: i64 = web.uses.iter().map(|&x| if remat { w(x) } else { 2 * w(x) }).sum();
                    let dd: i64 = web.defs.iter().map(|&x| if remat || arg_init { -w(x) } else { w(x) }).sum();
                    (u + dd) as f64
                })
                .collect();
            let chain = |use_costs: bool| -> Vec<usize> {
                let push = simplify_order(&adj, &deg, &num, k, if use_costs { Some(&costs) } else { None });
                // colouring order (reverse push) of the blocked callee-saved values
                push.iter().rev().copied().filter(|x| blocked.contains(x) && sel.contains(&cls[*x])).collect()
            };
            let (cu, cc) = (chain(false), chain(true));
            // solve on the callee-saved webs
            let sl: BTreeMap<usize, usize> = sel.iter().enumerate().map(|(x, &i)| (i, x)).collect();
            let nodes: Vec<Node> = sel
                .iter()
                .map(|&i| Node {
                    class: Class::Temp(0),
                    float: fpr,
                    crosses_call: true,
                    interferes: wall[i].interferes.iter().filter_map(|j| sl.get(j).copied()).collect(),
                    extra_degree: 0,
                })
                .collect();
            let observed: Vec<u8> = sel.iter().map(|&i| wall[i].reg).collect();
            let consistent = |ch: &[usize]| -> bool {
                let ids: Vec<usize> = ch.iter().map(|&x| sl[&cls[x]]).collect();
                let mut before = vec![];
                for p in 0..ids.len() {
                    if p + 1 < ids.len() {
                        before.push((ids[p], ids[p + 1]));
                    }
                }
                for &b in &ids {
                    for o in 0..sel.len() {
                        if !ids.contains(&o) {
                            before.push((b, o));
                        }
                    }
                }
                solve_order_constrained(&nodes, &observed, &before).is_some()
            };
            if solve_order_constrained(&nodes, &observed, &[]).is_none() {
                continue; // not greedy-consistent at all (spills etc.)
            }
            let (gu, gc) = (consistent(&cu), consistent(&cc));
            ok_u += gu as usize;
            ok_c += gc as usize;
            if cu != cc {
                differ += 1;
                ok_u_d += gu as usize;
                ok_c_d += gc as usize;
                if verbose {
                    let show = |ch: &[usize]| ch.iter().map(|&x| reg_name(wall[cls[x]].reg)).collect::<Vec<_>>().join(" ");
                    println!(
                        "{name} {}: uniform [{}] {} / cost [{}] {}",
                        f.name,
                        show(&cu),
                        if gu { "ok" } else { "FAIL" },
                        show(&cc),
                        if gc { "ok" } else { "FAIL" }
                    );
                }
            }
        }
    }
    println!(
        "{nfn} functions with >=2 call-crossing {} webs; {blocked_fns} with an estimated K-blocked value",
        if fpr { "FPR" } else { "GPR" }
    );
    println!(
        "blocked-order prediction consistent with observed registers: uniform-cost rule {ok_u}, cost/degree rule {ok_c} (of {blocked_fns}, greedy-consistent only)"
    );
    println!("functions where the two rules predict different orders: {differ}; consistent: uniform {ok_u_d}, cost/degree {ok_c_d}");
    Ok(())
}
