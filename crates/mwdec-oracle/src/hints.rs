//! Emit hints for one target function: classify its callee-saved values (param / temp / named /
//! blocked) and the declaration order of named locals, by inverting the colouring model.

use crate::regalloc::{infer_classes, InferInput, InferredClass};
use crate::webs::{self, Instr, Origin, Web};

#[derive(Clone, Debug)]
pub struct Hint {
    pub web: usize,
    pub reg: u8,
    pub origin: Origin,
    pub class: InferredClass,
    pub est_degree: u32,
}

/// Per register class (GPR then FPR). `None` for a class = no consistent assignment found.
pub fn hints(ins: &[Instr]) -> (Vec<Web>, Vec<Option<Vec<Hint>>>) {
    let ws = webs::webs(ins);
    let wall = webs::webs_ext(ins, true);
    let deg = webs::estimate_degrees(ins, &wall);
    let mut out = vec![];
    for float in [false, true] {
        let sel: Vec<usize> =
            ws.iter().filter(|w| (w.reg >= 32) == float && w.crosses_call).map(|w| w.id).collect();
        if sel.is_empty() {
            out.push(Some(vec![]));
            continue;
        }
        let local: std::collections::BTreeMap<usize, usize> = sel.iter().enumerate().map(|(k, &w)| (w, k)).collect();
        let k = if float { 32 } else { 29 };
        let est: Vec<u32> = sel
            .iter()
            .map(|&w| {
                wall.iter().position(|x| x.reg == ws[w].reg && x.defs == ws[w].defs).map(|i| deg[i]).unwrap_or(0)
            })
            .collect();
        let vals: Vec<InferInput> = sel
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let param = match ws[w].origin {
                    Origin::Param { arg_reg } => {
                        Some(if arg_reg >= 32 { arg_reg as u32 - 33 } else { arg_reg as u32 - 3 })
                    }
                    _ => None,
                };
                InferInput {
                    observed: ws[w].reg,
                    param,
                    def_pos: ws[w].defs[0] as u32,
                    prior_named: matches!(ws[w].origin, Origin::CallResultHop { .. }),
                    maybe_blocked: est[i] + 6 >= k,
                    interferes: ws[w].interferes.iter().filter_map(|j| local.get(j).copied()).collect(),
                    float,
                }
            })
            .collect();
        out.push(infer_classes(&vals).map(|cls| {
            sel.iter()
                .enumerate()
                .map(|(i, &w)| Hint {
                    web: w,
                    reg: ws[w].reg,
                    origin: ws[w].origin.clone(),
                    class: cls[i].clone(),
                    est_degree: est[i],
                })
                .collect()
        }));
    }
    (ws, out)
}

/// Human-readable description of a hint's class.
pub fn describe(c: &InferredClass) -> String {
    match c {
        InferredClass::Param(i) => format!("param #{i}"),
        InferredClass::Temp { rank } => format!("TEMP (unnamed, or non-move uses only), creation rank {rank}"),
        InferredClass::Named { decl } => format!("NAMED local (has a move-use), declare #{decl}"),
        InferredClass::Blocked => "blocked by IG degree (K-effect), coloured first".into(),
    }
}
