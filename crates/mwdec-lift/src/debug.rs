//! Compact C-like rendering of the IR for debugging the lift pipeline (`MWDEC_DUMP=1` prints the
//! body after every stage to stderr). Not used for output: `mwdec-emit` renders the real source.

use crate::ir::*;

pub fn expr(e: &Expr, vars: &[Var]) -> String {
    match e {
        Expr::Var(v) => vars.get(*v).map_or(format!("v{v}"), |x| format!("{}#{v}", x.name)),
        Expr::Int { value, .. } => format!("{value}"),
        Expr::Float { bits, double } => {
            if *double {
                format!("{}", f64::from_bits(*bits))
            } else {
                format!("{}f", f32::from_bits(*bits as u32))
            }
        }
        Expr::Str { bytes } => format!("{:?}", String::from_utf8_lossy(bytes)),
        Expr::Global { symbol, .. } => symbol.clone(),
        Expr::FuncAddr { symbol } => format!("&{symbol}"),
        Expr::AddrOf(x) => format!("&({})", expr(x, vars)),
        Expr::Load { base, offset, .. } => format!("*({}+{offset:#x})", expr(base, vars)),
        Expr::Index { base, index, .. } => format!("{}[{}]", expr(base, vars), expr(index, vars)),
        Expr::Member { base, offset, .. } => format!("{}.@{offset:#x}", expr(base, vars)),
        Expr::Unary { op, e, .. } => {
            let o = match op {
                UnOp::Neg => "-",
                UnOp::BitNot => "~",
                UnOp::Not => "!",
            };
            format!("{o}({})", expr(e, vars))
        }
        Expr::Binary { op, l, r, .. } => format!("({} {} {})", expr(l, vars), op.c_str(), expr(r, vars)),
        Expr::Cast { e, .. } => format!("(cast){}", expr(e, vars)),
        Expr::Call { callee, args, .. } => {
            let a: Vec<String> = args.iter().map(|a| expr(a, vars)).collect();
            let c = match callee {
                Callee::Direct { symbol, .. } => symbol.clone(),
                Callee::Method { symbol, this, .. } => format!("{}->{}", expr(this, vars), symbol),
                Callee::Virtual { this, vtable_offset, .. } => format!("{}->vt[{vtable_offset:#x}]", expr(this, vars)),
                Callee::Indirect(f) => format!("({})", expr(f, vars)),
            };
            format!("{c}({})", a.join(", "))
        }
        Expr::Ternary { c, t, f, .. } => format!("({} ? {} : {})", expr(c, vars), expr(t, vars), expr(f, vars)),
        Expr::Unknown { text, .. } => format!("?{text}?"),
        Expr::New { args, .. } => format!("new(..{})", args.len()),
        Expr::Construct { args, .. } => {
            let a: Vec<String> = args.iter().map(|a| expr(a, vars)).collect();
            format!("T({})", a.join(", "))
        }
        Expr::BitField { base, shift, width, .. } => format!("{}:[{shift},{width}]", expr(base, vars)),
        Expr::IncDec { e, delta, post } => {
            let o = if *delta > 0 { "++" } else { "--" };
            if *post {
                format!("{}{o}", expr(e, vars))
            } else {
                format!("{o}{}", expr(e, vars))
            }
        }
    }
}

pub fn body(b: &[Stmt], vars: &[Var]) -> String {
    let mut s = String::new();
    stmts(b, vars, 1, &mut s);
    s
}

fn stmts(b: &[Stmt], vars: &[Var], ind: usize, out: &mut String) {
    for st in b {
        stmt(st, vars, ind, out);
    }
}

fn stmt(st: &Stmt, vars: &[Var], ind: usize, out: &mut String) {
    let pad = "  ".repeat(ind);
    match st {
        Stmt::Expr(e) => out.push_str(&format!("{pad}{};\n", expr(e, vars))),
        Stmt::Assign { dst, src } => out.push_str(&format!("{pad}{} = {};\n", expr(dst, vars), expr(src, vars))),
        Stmt::If { cond, then, els } => {
            out.push_str(&format!("{pad}if {} {{\n", expr(cond, vars)));
            stmts(then, vars, ind + 1, out);
            if !els.is_empty() {
                out.push_str(&format!("{pad}}} else {{\n"));
                stmts(els, vars, ind + 1, out);
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::While { cond, body } => {
            out.push_str(&format!("{pad}while {} {{\n", expr(cond, vars)));
            stmts(body, vars, ind + 1, out);
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::DoWhile { body, cond } => {
            out.push_str(&format!("{pad}do {{\n"));
            stmts(body, vars, ind + 1, out);
            out.push_str(&format!("{pad}}} while {};\n", expr(cond, vars)));
        }
        Stmt::For { init, cond, step, body } => {
            out.push_str(&format!("{pad}for (init {} ; {} ; step {}) {{\n", init.len(), expr(cond, vars), step.len()));
            stmts(init, vars, ind + 2, out);
            stmts(step, vars, ind + 2, out);
            stmts(body, vars, ind + 1, out);
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::Switch { e, cases } => {
            out.push_str(&format!("{pad}switch {} {{\n", expr(e, vars)));
            for c in cases {
                out.push_str(&format!("{pad}case {:?}{}:\n", c.values, if c.is_default { " default" } else { "" }));
                stmts(&c.body, vars, ind + 1, out);
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Stmt::Return(e) => out.push_str(&format!("{pad}return {};\n", e.as_ref().map_or(String::new(), |e| expr(e, vars)))),
        Stmt::Break => out.push_str(&format!("{pad}break;\n")),
        Stmt::Continue => out.push_str(&format!("{pad}continue;\n")),
        Stmt::Goto(l) => out.push_str(&format!("{pad}goto L{l};\n")),
        Stmt::Label(l) => out.push_str(&format!("{pad}L{l}:\n")),
        Stmt::Comment(c) => out.push_str(&format!("{pad}// {c}\n")),
    }
}

/// Print the body under a stage name when `MWDEC_DUMP` is set.
pub fn stage(name: &str, b: &[Stmt], vars: &[Var]) {
    if std::env::var_os("MWDEC_DUMP").is_some() {
        eprintln!("=== {name}\n{}", body(b, vars));
    }
}
