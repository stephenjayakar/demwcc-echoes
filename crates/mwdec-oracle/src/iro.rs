//! The front-end optimizer's (IRO) own dump, switched on by the tracer (`TraceOptions::iro`):
//! parse it into per-function stages / flowgraph blocks / linear nodes, and render each block as
//! C-like statements. The last stage of a function ("After IRO_Optimizer") is the expression form
//! that code generation consumes, so it shows reassociation, operand order (which side is evaluated
//! first), common subexpressions, conditional assignments and strength reduction as they reach
//! instruction selection.
//!
//! Dump line forms (one linear node per line, children listed by node index):
//! `N: Operand x <flags>` (an object reference `&x`, or an integer constant), `N: EINDIRECT c`,
//! `N: EADD a b`, `N: EASS dst src`, `N: Funccall f(a,b)`, `N: If c @L`, `N: IfNot c @L`,
//! `N: Goto @L`, `N: Label @L`, `N: Return [c]`, `N: Switch c`, `N: Nop`, `N: End`.

use serde::Serialize;
use std::collections::HashMap;

/// Name of the final stage (the form handed to code generation).
pub const FINAL_STAGE: &str = "After IRO_Optimizer";

#[derive(Clone, Debug, Serialize)]
pub struct IroNode {
    pub index: u32,
    /// `Operand`, `EINDIRECT`, `EADD`, ..., `Funccall`, `If`, `Goto`, `Label`, `Return`, `Nop`
    pub op: String,
    /// child node indices (for `Funccall`: function first, then the arguments)
    pub args: Vec<u32>,
    /// operand text (`Operand x`) or label (`@15`)
    pub text: Option<String>,
    /// `<...>` flags (`assigned`, `used`, `ind`, `reffed`, ...)
    pub flags: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct IroBlock {
    pub index: u32,
    pub succ: Vec<u32>,
    pub pred: Vec<u32>,
    pub loop_depth: u32,
    pub nodes: Vec<IroNode>,
}

#[derive(Clone, Debug, Serialize)]
pub struct IroStage {
    pub function: String,
    pub stage: String,
    pub blocks: Vec<IroBlock>,
}

fn nums(s: &str) -> Vec<u32> {
    s.split_whitespace().filter_map(|x| x.parse().ok()).collect()
}

fn parse_node(line: &str) -> Option<IroNode> {
    let (idx, rest) = line.trim().split_once(": ")?;
    let index: u32 = idx.trim().parse().ok()?;
    let mut flags = vec![];
    let mut body = rest.to_string();
    while let (Some(a), Some(b)) = (body.rfind('<'), body.rfind('>')) {
        if a > b || b != body.trim_end().len() - 1 {
            break;
        }
        flags.insert(0, body[a + 1..b].to_string());
        body = body[..a].trim_end().to_string();
    }
    let (op, tail) = match body.split_once(' ') {
        Some((o, t)) => (o.to_string(), t.trim().to_string()),
        None => (body.clone(), String::new()),
    };
    let mut args = vec![];
    let mut text = None;
    match op.as_str() {
        "Operand" => text = Some(tail),
        "Funccall" => {
            // `f(a,b)`
            if let Some((f, a)) = tail.split_once('(') {
                args.extend(nums(f));
                args.extend(a.trim_end_matches(')').split(',').filter_map(|x| x.trim().parse::<u32>().ok()));
            }
        }
        _ => {
            for tok in tail.split_whitespace() {
                if tok.starts_with('@') {
                    text = Some(tok.to_string());
                } else if let Ok(n) = tok.parse() {
                    args.push(n);
                } else {
                    text = Some(tok.to_string());
                }
            }
        }
    }
    Some(IroNode { index, op, args, text, flags })
}

/// Parse a whole dump (all functions, all stages present in it).
pub fn parse(dump: &str) -> Vec<IroStage> {
    let mut out: Vec<IroStage> = vec![];
    let mut open = false;
    for line in dump.lines() {
        let t = line.trim_end();
        if let Some(rest) = t.strip_prefix("Dumping function ") {
            if let Some((f, st)) = rest.split_once(" after ") {
                out.push(IroStage { function: f.to_string(), stage: st.trim().to_string(), blocks: vec![] });
                open = true;
            }
            continue;
        }
        if !t.is_empty() && !t.starts_with(' ') && !is_dump_header(t) {
            open = false; // a pass message: the flowgraph listing ended
        }
        if !open {
            continue;
        }
        let Some(stage) = out.last_mut() else { continue };
        if let Some(rest) = t.strip_prefix("Flowgraph node ") {
            let index = rest.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0);
            stage.blocks.push(IroBlock { index, succ: vec![], pred: vec![], loop_depth: 0, nodes: vec![] });
            continue;
        }
        let Some(b) = stage.blocks.last_mut() else { continue };
        if let Some(r) = t.strip_prefix("Succ = ") {
            b.succ = nums(r);
        } else if let Some(r) = t.strip_prefix("Pred = ") {
            b.pred = nums(r);
        } else if let Some(r) = t.strip_prefix("LoopDepth = ") {
            b.loop_depth = r.trim().parse().unwrap_or(0);
        } else if t.starts_with("Succ =") || t.starts_with("Pred =") {
            // empty lists
        } else if t.starts_with(' ') {
            if let Some(n) = parse_node(t) {
                b.nodes.push(n);
            }
        }
    }
    out
}

fn is_dump_header(t: &str) -> bool {
    ["Flowgraph", "Succ =", "Pred =", "MustReach", "LoopDepth", "Dom:", "----"].iter().any(|p| t.starts_with(p))
}

/// The optimizer's pass messages per function (loop unrolling decisions, propagations, ...): every
/// non-listing line, grouped under the `Starting function X` line that precedes it.
pub fn messages(dump: &str) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = vec![];
    let mut in_listing = false;
    for line in dump.lines() {
        let t = line.trim_end();
        if let Some(f) = t.strip_prefix("Starting function ") {
            out.push((f.trim().to_string(), vec![]));
            in_listing = false;
            continue;
        }
        if t.starts_with("Dumping function ") {
            in_listing = true;
            continue;
        }
        if t.is_empty() || t.starts_with(' ') || is_dump_header(t) || t.starts_with("*****") {
            continue;
        }
        if in_listing && !t.starts_with(' ') {
            in_listing = false;
        }
        if let Some(last) = out.last_mut() {
            last.1.push(t.to_string());
        }
    }
    out
}

fn binop(op: &str) -> Option<&'static str> {
    Some(match op {
        "EMUL" => "*",
        "EDIV" => "/",
        "EMODULO" => "%",
        "EADD" => "+",
        "ESUB" => "-",
        "ESHL" => "<<",
        "ESHR" => ">>",
        "ELESS" => "<",
        "EGREATER" => ">",
        "ELESSEQU" => "<=",
        "EGREATEREQU" => ">=",
        "EEQU" => "==",
        "ENOTEQU" => "!=",
        "EAND" => "&",
        "EXOR" => "^",
        "EOR" => "|",
        "ELAND" => "&&",
        "ELOR" => "||",
        "EASS" => "=",
        "EMULASS" => "*=",
        "EDIVASS" => "/=",
        "EMODASS" => "%=",
        "EADDASS" => "+=",
        "ESUBASS" => "-=",
        "ESHLASS" => "<<=",
        "ESHRASS" => ">>=",
        "EANDASS" => "&=",
        "EXORASS" => "^=",
        "EORASS" => "|=",
        "ECOMMA" => ",",
        _ => return None,
    })
}

impl IroBlock {
    /// Render every root expression (nodes no later node uses) as a C-like statement, in order.
    /// `Operand x` is the address of `x`, so `EINDIRECT(Operand x)` prints as `x`.
    pub fn statements(&self) -> Vec<String> {
        // index -> position of the latest node with that index (indices can be reused)
        let mut used = vec![false; self.nodes.len()];
        let mut child_pos: Vec<Vec<usize>> = vec![vec![]; self.nodes.len()];
        let mut latest: HashMap<u32, usize> = HashMap::new();
        for (p, n) in self.nodes.iter().enumerate() {
            for a in &n.args {
                if let Some(&q) = latest.get(a) {
                    used[q] = true;
                    child_pos[p].push(q);
                }
            }
            latest.insert(n.index, p);
        }
        fn render(nodes: &[IroNode], kids: &[Vec<usize>], p: usize, depth: usize) -> String {
            if depth > 64 {
                return "...".into();
            }
            let n = &nodes[p];
            let k = |i: usize| -> String {
                kids[p].get(i).map(|&q| render(nodes, kids, q, depth + 1)).unwrap_or_else(|| "?".into())
            };
            let is_addr = |i: usize| kids[p].get(i).map_or(false, |&q| nodes[q].op == "Operand");
            match n.op.as_str() {
                "Operand" => {
                    let t = n.text.clone().unwrap_or_default();
                    if t.parse::<f64>().is_ok() {
                        t
                    } else {
                        format!("&{t}")
                    }
                }
                "EINDIRECT" => {
                    if is_addr(0) {
                        k(0).trim_start_matches('&').to_string()
                    } else {
                        format!("*({})", k(0))
                    }
                }
                "EMONMIN" => format!("-({})", k(0)),
                "EBINNOT" => format!("~({})", k(0)),
                "ELOGNOT" => format!("!({})", k(0)),
                "EPOSTINC" => format!("{}++", k(0)),
                "EPOSTDEC" => format!("{}--", k(0)),
                "EPREINC" => format!("++{}", k(0)),
                "EPREDEC" => format!("--{}", k(0)),
                "ETYPCON" => format!("(cast)({})", k(0)),
                "EBITFIELD" => format!("bitfield({})", k(0)),
                "ECOND" => format!("({} ? {} : {})", k(0), k(1), k(2)),
                "ECONDASS" => format!("if ({}) {} = {}", k(0), k(1), k(2)),
                "Funccall" => {
                    let args: Vec<String> = (1..kids[p].len()).map(k).collect();
                    format!("{}({})", k(0).trim_start_matches('&'), args.join(", "))
                }
                "If" => format!("if ({}) goto {}", k(0), n.text.clone().unwrap_or_default()),
                "IfNot" => format!("if (!({})) goto {}", k(0), n.text.clone().unwrap_or_default()),
                "Goto" => format!("goto {}", n.text.clone().unwrap_or_default()),
                "Label" => format!("{}:", n.text.clone().unwrap_or_default()),
                "Return" => {
                    if kids[p].is_empty() {
                        "return".into()
                    } else {
                        format!("return {}", k(0))
                    }
                }
                "Switch" => format!("switch ({})", k(0)),
                op => match binop(op) {
                    Some(sym) if kids[p].len() == 2 => {
                        let top = matches!(sym, "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "<<=" | ">>=" | "&=" | "^=" | "|=");
                        if top && depth == 0 {
                            format!("{} {sym} {}", k(0), k(1))
                        } else {
                            format!("({} {sym} {})", k(0), k(1))
                        }
                    }
                    _ => {
                        let args: Vec<String> = (0..kids[p].len()).map(k).collect();
                        format!("{op}({})", args.join(", "))
                    }
                },
            }
        }
        (0..self.nodes.len())
            .filter(|&p| !used[p] && !matches!(self.nodes[p].op.as_str(), "Nop" | "End"))
            .map(|p| render(&self.nodes, &child_pos, p, 0))
            .collect()
    }
}

impl IroStage {
    /// All statements of the stage, block by block (`B<n>:` headers with loop depth).
    pub fn render(&self) -> Vec<String> {
        let mut out = vec![];
        for b in &self.blocks {
            out.push(format!("B{} (loop depth {}, succ {:?}):", b.index, b.loop_depth, b.succ));
            out.extend(b.statements().into_iter().map(|s| format!("    {s}")));
        }
        out
    }
}

/// The final IRO form of the first function whose name contains `func`.
pub fn final_stage<'a>(stages: &'a [IroStage], func: &str) -> Option<&'a IroStage> {
    stages.iter().filter(|s| s.function.contains(func) && s.stage == FINAL_STAGE).last()
}

