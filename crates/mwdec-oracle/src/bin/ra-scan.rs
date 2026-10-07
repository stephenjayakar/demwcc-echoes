//! Source-free validation of the colouring model on the ORIGINAL game code: for every function of
//! the target objects (dtk split of the retail binary) of TRAIN-split game units, recover the
//! callee-saved webs and check
//!   (a) greedy-consistency: some colouring order reproduces the observed registers under MWCC's
//!       rule (lowest free obtained nonvolatile, else next of r31..r14);
//!   (b) param-consistency: such an order exists with all parameter webs coloured after every
//!       other web, in decreasing parameter order (params have the lowest vregs).
//! usage: ra-scan [--fpr] [--limit N] [-v]
use mwdec_oracle::asm;
use mwdec_oracle::regalloc::{solve_order_constrained, Class, Node};
use mwdec_oracle::webs::{self, reg_name, Origin};
use std::collections::BTreeMap;



fn split_of(unit: &str) -> &'static str {
    let mut h: u32 = 0x811c9dc5;
    for b in unit.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    if h % 10 < 7 { "train" } else { "test" }
}

fn main() -> anyhow::Result<()> {
    mwdec_core::memcap::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fpr = args.iter().any(|a| a == "--fpr");
    let verbose = args.iter().any(|a| a == "-v");
    let limit: usize = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    // units from objdiff.json (minimal JSON scan to avoid a serde dep)
    let text = std::fs::read_to_string(mwdec_core::paths::project_root().join("objdiff.json"))?;
    let mut units: Vec<(String, String)> = vec![];
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
        // only C++ game units compiled with 2.7 deferred,noauto (main + REL game code)
        let block_end = next_name.min(rest.len());
        let block = &rest[..block_end];
        if block.contains("-lang=c++") && block.contains("deferred,noauto") && split_of(name) == "train" {
            units.push((name.to_string(), target.to_string()));
        }
    }
    eprintln!("{} train C++ game units", units.len());
    let (mut nfn, mut greedy_ok, mut param_ok, mut trivial, mut unknown) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut fails: BTreeMap<&'static str, usize> = BTreeMap::new();
    let (mut blocked_ok, mut blocked_fail) = (0usize, 0usize);
    let mut param_ok_deg = 0usize;
    for (name, target) in units.iter().take(limit) {
        let path = mwdec_core::paths::project_root().join(&target);
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let Ok(obj) = asm::parse(&bytes) else { continue };
        for f in &obj.funcs {
            if f.code.len() < 8 {
                continue;
            }
            let calls: BTreeMap<u32, String> =
                f.relocs.iter().filter(|r| r.r_type == 10).map(|r| (r.offset, r.target.clone())).collect();
            let ins = webs::decode(&f.code, &calls);
            let ws = webs::webs(&ins);
            let sel: Vec<usize> =
                ws.iter().filter(|w| (w.reg >= 32) == fpr && w.crosses_call).map(|w| w.id).collect();
            if sel.len() < 2 {
                trivial += 1;
                continue;
            }
            nfn += 1;
            let local: BTreeMap<usize, usize> = sel.iter().enumerate().map(|(k, &w)| (w, k)).collect();
            let nodes: Vec<Node> = sel
                .iter()
                .map(|&w| Node {
                    class: Class::Temp(0),
                    float: fpr,
                    crosses_call: true,
                    interferes: ws[w].interferes.iter().filter_map(|j| local.get(j).copied()).collect(),
                    extra_degree: 0,
                })
                .collect();
            let observed: Vec<u8> = sel.iter().map(|&w| ws[w].reg).collect();
            // degree estimate from the all-register web graph (map callee-saved webs by def+reg)
            let wall = webs::webs_ext(&ins, true);
            let deg_all = webs::estimate_degrees(&ins, &wall);
            let deg_of = |w: usize| -> u32 {
                wall.iter()
                    .position(|x| x.reg == ws[w].reg && x.defs == ws[w].defs)
                    .map(|k| deg_all[k])
                    .unwrap_or(0)
            };
            let param_blocked = sel.iter().any(|&w| matches!(ws[w].origin, Origin::Param { .. }) && deg_of(w) >= 29);
            match solve_order_constrained(&nodes, &observed, &[]) {
                None => {
                    *fails.entry("greedy-inconsistent").or_default() += 1;
                    if verbose {
                        println!("GREEDY FAIL {name} {}", f.name);
                        for &w in &sel {
                            println!(
                                "   {} {:?} defs {:?} inter {:?}",
                                reg_name(ws[w].reg),
                                ws[w].origin,
                                ws[w].defs.iter().map(|&d| format!("{:x}", ins[d].off)).collect::<Vec<_>>(),
                                ws[w].interferes.iter().map(|&j| reg_name(ws[j].reg)).collect::<Vec<_>>()
                            );
                        }
                    }
                    continue;
                }
                Some(_) => greedy_ok += 1,
            }
            // params: arg register order -> param order
            let params: Vec<(usize, u8)> = sel
                .iter()
                .enumerate()
                .filter_map(|(k, &w)| match ws[w].origin {
                    Origin::Param { arg_reg } => Some((k, arg_reg)),
                    _ => None,
                })
                .collect();
            if params.is_empty() {
                param_ok += 1;
                param_ok_deg += 1;
                unknown += 1;
                continue;
            }
            let mut before = vec![];
            for k in 0..sel.len() {
                if params.iter().any(|&(p, _)| p == k) {
                    continue;
                }
                for &(p, _) in &params {
                    before.push((k, p));
                }
            }
            for &(p, ra) in &params {
                for &(q, rb) in &params {
                    if ra > rb {
                        before.push((p, q)); // later param coloured first
                    }
                }
            }
            // degree-aware variant: params blocked in simplify pass 1 (degree >= K at their visit)
            // are pushed in a later pass, hence coloured BEFORE every pass-1 node.
            let mut ps: Vec<(usize, u8)> = params.clone();
            ps.sort_by_key(|&(_, ra)| ra);
            let mut pushed1: Vec<usize> = vec![];
            let mut blocked: Vec<usize> = vec![];
            for &(p, _) in &ps {
                let w = sel[p];
                let dec = pushed1.iter().filter(|&&q| ws[w].interferes.contains(&sel[q])).count() as u32;
                if deg_of(w).saturating_sub(dec) >= 29 {
                    blocked.push(p);
                } else {
                    pushed1.push(p);
                }
            }
            let mut before2 = vec![];
            for k in 0..sel.len() {
                if params.iter().any(|&(p, _)| p == k) {
                    continue;
                }
                for &p in &pushed1 {
                    before2.push((k, p));
                }
            }
            for &(p, ra) in &params {
                for &(q, rb) in &params {
                    let pb = blocked.contains(&p);
                    let qb = blocked.contains(&q);
                    if (pb == qb && ra > rb) || (pb && !qb) {
                        before2.push((p, q));
                    }
                }
            }
            if solve_order_constrained(&nodes, &observed, &before2).is_some() {
                param_ok_deg += 1;
            } else if verbose {
                println!("PARAM(deg) FAIL {name} {} blocked={:?}", f.name, blocked.iter().map(|&p| reg_name(ws[sel[p]].reg)).collect::<Vec<_>>());
                for &w in &sel {
                    println!("   {} {:?} deg {} inter {:?}", reg_name(ws[w].reg), ws[w].origin, deg_of(w),
                        ws[w].interferes.iter().map(|&j| reg_name(ws[j].reg)).collect::<Vec<_>>());
                }
            }
            match solve_order_constrained(&nodes, &observed, &before) {
                Some(_) => {
                    param_ok += 1;
                    if param_blocked {
                        blocked_ok += 1;
                    }
                }
                None => {
                    if param_blocked {
                        blocked_fail += 1;
                    }
                    *fails.entry("param-order-inconsistent").or_default() += 1;
                    if verbose {
                        println!("PARAM FAIL {name} {}", f.name);
                        for &w in &sel {
                            println!(
                                "   {} {:?} inter {:?}",
                                reg_name(ws[w].reg),
                                ws[w].origin,
                                ws[w].interferes.iter().map(|&j| reg_name(ws[j].reg)).collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
        }
    }
    println!(
        "{} functions with >=2 call-crossing {} webs ({} trivial skipped): greedy-consistent {} ({:.1}%), param-consistent {} ({:.1}%) [{} without params]",
        nfn,
        if fpr { "FPR" } else { "GPR" },
        trivial,
        greedy_ok,
        100.0 * greedy_ok as f64 / nfn.max(1) as f64,
        param_ok,
        100.0 * param_ok as f64 / nfn.max(1) as f64,
        unknown
    );
    println!("failures: {fails:?}");
    println!("param-consistent with degree-aware simplify model: {} ({:.1}%)", param_ok_deg, 100.0 * param_ok_deg as f64 / nfn.max(1) as f64);
    println!(
        "functions with a param of estimated degree >= 29: {} among param-consistent, {} among param-inconsistent",
        blocked_ok, blocked_fail
    );
    Ok(())
}
