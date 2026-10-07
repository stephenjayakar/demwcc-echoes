//! Per-function analysis for the permuter: which function to mutate, local/param types, a
//! cheap side-effect model for statement reordering, and best-effort expression types.
use crate::cst::Cst;
use std::collections::{HashMap, HashSet};

pub const STMT_KINDS: &[&str] = &[
    "expression_statement",
    "declaration",
    "if_statement",
    "for_statement",
    "while_statement",
    "do_statement",
    "return_statement",
    "compound_statement",
    "switch_statement",
    "break_statement",
    "continue_statement",
    "goto_statement",
    "labeled_statement",
];

pub fn is_stmt(k: &str) -> bool {
    STMT_KINDS.contains(&k)
}

pub fn is_loop(k: &str) -> bool {
    matches!(k, "for_statement" | "while_statement" | "do_statement")
}

/// Declared name of a declarator (innermost identifier) and the pointer/reference suffix.
pub fn declarator_name(cst: &Cst, d: usize) -> Option<(String, String)> {
    let mut suffix = String::new();
    let mut n = d;
    loop {
        match cst.kind(n) {
            "identifier" | "field_identifier" => return Some((cst.text(n).to_string(), suffix)),
            "init_declarator" => n = cst.child(n, "declarator")?,
            "pointer_declarator" => {
                suffix.push('*');
                // `T* const p`
                n = cst.child(n, "declarator")?;
            }
            "reference_declarator" => {
                suffix.push('&');
                n = *cst.named(n).last()?;
            }
            "array_declarator" => {
                suffix.push_str("[]");
                n = cst.child(n, "declarator")?;
            }
            "parenthesized_declarator" => n = *cst.named(n).first()?,
            _ => return None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Var {
    pub name: String,
    /// Full type text, e.g. `const CVector3f&`, `u32*`, `int`.
    pub ty: String,
    pub is_param: bool,
    /// Declaration node (locals) or parameter_declaration.
    pub decl: usize,
}

/// Analysis of the function being permuted.
#[derive(Clone, Debug)]
pub struct FuncInfo {
    pub def: usize,
    pub body: usize,
    pub ret_ty: String,
    pub vars: HashMap<String, Var>,
    /// Locals whose address is taken (`&x`) or that are references/arrays: treat as memory.
    pub aliased: HashSet<String>,
}

/// Mangled-name base: `Foo__3BarFi` -> `Foo`, `__ct__3BarFv` -> `__ct`.
fn mangled_base(sym: &str) -> &str {
    let s = sym.strip_prefix("__").map(|r| (r, 2)).unwrap_or((sym, 0));
    match s.0.find("__") {
        Some(i) => &sym[..i + s.1],
        None => sym,
    }
}

/// Last component of a function definition's declarator name, e.g. `Bar` for `CFoo::Bar`,
/// `~CFoo` for a destructor, and the class name for `CFoo::CFoo`.
pub fn def_name(cst: &Cst, def: usize) -> Option<(String, Option<String>)> {
    let mut d = cst.child(def, "declarator")?;
    loop {
        match cst.kind(d) {
            "function_declarator" => d = cst.child(d, "declarator")?,
            "pointer_declarator" | "reference_declarator" => d = *cst.named(d).last()?,
            "qualified_identifier" => {
                let scope = cst.child(d, "scope").map(|s| cst.text(s).to_string());
                let name = cst.child(d, "name")?;
                if cst.kind(name) == "qualified_identifier" {
                    d = name;
                    continue;
                }
                return Some((cst.text(name).to_string(), scope));
            }
            _ => return Some((cst.text(d).to_string(), None)),
        }
    }
}

/// All function definitions with a body in the file.
pub fn function_defs(cst: &Cst) -> Vec<usize> {
    (0..cst.nodes.len()).filter(|&i| cst.kind(i) == "function_definition" && cst.child(i, "body").is_some()).collect()
}

/// The function definition to permute for mangled `symbol`: the definition whose name matches
/// the mangled base name (constructors/destructors by class name), else the last definition.
pub fn find_target(cst: &Cst, symbol: &str) -> Option<usize> {
    let defs = function_defs(cst);
    let base = mangled_base(symbol);
    let pick = defs.iter().copied().filter(|&d| {
        let Some((name, scope)) = def_name(cst, d) else { return false };
        let scope_last = scope.as_deref().map(|s| s.rsplit("::").next().unwrap_or(s).to_string());
        match base {
            "__ct" => scope_last.as_deref() == Some(name.as_str()),
            "__dt" => name.starts_with('~'),
            b if b.starts_with("__") => name.starts_with("operator") || name == b,
            b => name == b,
        }
    });
    let v: Vec<usize> = pick.collect();
    v.last().copied().or_else(|| defs.last().copied())
}

fn type_prefix(cst: &Cst, decl: usize, first_declarator: usize) -> String {
    let s = cst.nodes[decl].start;
    let e = cst.nodes[first_declarator].start;
    cst.src[s..e].split_whitespace().collect::<Vec<_>>().join(" ")
}

impl FuncInfo {
    pub fn analyze(cst: &Cst, def: usize) -> Option<FuncInfo> {
        let body = cst.child(def, "body")?;
        let mut vars = HashMap::new();
        let mut aliased = HashSet::new();
        let ret_ty = cst.child(def, "type").map(|t| cst.text(t).to_string()).unwrap_or_default();
        // Parameters.
        if let Some(fd) = cst.descendants(cst.child(def, "declarator")?).into_iter().find(|&n| cst.kind(n) == "function_declarator") {
            if let Some(pl) = cst.child(fd, "parameters") {
                for p in cst.named(pl) {
                    if cst.kind(p) != "parameter_declaration" {
                        continue;
                    }
                    let Some(d) = cst.child(p, "declarator") else { continue };
                    let Some((name, suf)) = declarator_name(cst, d) else { continue };
                    let ty = format!("{}{}", type_prefix(cst, p, d), suf);
                    vars.insert(name.clone(), Var { name, ty, is_param: true, decl: p });
                }
            }
        }
        for n in cst.descendants(body) {
            match cst.kind(n) {
                "declaration" => {
                    let ds: Vec<usize> = cst.children_by_field(n, "declarator").collect();
                    let Some(&first) = ds.first() else { continue };
                    let prefix = type_prefix(cst, n, first);
                    for d in ds {
                        if let Some((name, suf)) = declarator_name(cst, d) {
                            if suf.contains('&') || suf.contains('[') {
                                aliased.insert(name.clone());
                            }
                            vars.insert(name.clone(), Var { name, ty: format!("{prefix}{suf}"), is_param: false, decl: n });
                        }
                    }
                }
                "pointer_expression" if cst.op(n) == Some("&") => {
                    if let Some(a) = cst.child(n, "argument") {
                        let mut a = a;
                        while cst.kind(a) == "field_expression" && cst.op(a) == Some(".") {
                            a = cst.child(a, "argument").unwrap_or(a);
                            if cst.kind(a) != "field_expression" {
                                break;
                            }
                        }
                        if cst.kind(a) == "identifier" {
                            aliased.insert(cst.text(a).to_string());
                        }
                    }
                }
                _ => {}
            }
        }
        Some(FuncInfo { def, body, ret_ty, vars, aliased })
    }

    /// Is `name` a non-aliased local or parameter (a pure register-like value)?
    pub fn is_private(&self, name: &str) -> bool {
        self.vars.contains_key(name) && !self.aliased.contains(name)
    }

    pub fn fresh_name(&self, cst: &Cst, stem: &str) -> String {
        for k in 0.. {
            let n = format!("{stem}{k}");
            if !self.vars.contains_key(&n) && !cst.src.contains(&n) {
                return n;
            }
        }
        unreachable!()
    }
}

/// What a piece of code reads/writes, conservatively.
#[derive(Clone, Debug, Default)]
pub struct Effects {
    pub reads: HashSet<String>,
    pub writes: HashSet<String>,
    pub mem_read: bool,
    pub mem_write: bool,
    /// return/break/continue/goto/labels/switch case labels: never reorder across.
    pub control: bool,
}

impl Effects {
    pub fn has_side_effects(&self) -> bool {
        self.mem_write || !self.writes.is_empty() || self.control
    }
    pub fn conflicts(&self, o: &Effects) -> bool {
        if self.control || o.control {
            return true;
        }
        if (self.mem_write && (o.mem_read || o.mem_write)) || (o.mem_write && self.mem_read) {
            return true;
        }
        self.writes.iter().any(|w| o.reads.contains(w) || o.writes.contains(w)) || o.writes.iter().any(|w| self.reads.contains(w))
    }
    pub fn merge(&mut self, o: &Effects) {
        self.reads.extend(o.reads.iter().cloned());
        self.writes.extend(o.writes.iter().cloned());
        self.mem_read |= o.mem_read;
        self.mem_write |= o.mem_write;
        self.control |= o.control;
    }
}

/// Root identifier of an lvalue if it is a private local accessed directly (`x`, `x.a.b`).
fn lvalue_local<'a>(cst: &'a Cst, info: &FuncInfo, mut n: usize) -> Option<&'a str> {
    loop {
        match cst.kind(n) {
            "identifier" => {
                let t = cst.text(n);
                return info.is_private(t).then_some(t);
            }
            "parenthesized_expression" => n = *cst.named(n).first()?,
            "field_expression" if cst.op(n) == Some(".") => n = cst.child(n, "argument")?,
            _ => return None,
        }
    }
}

pub fn effects(cst: &Cst, info: &FuncInfo, root: usize) -> Effects {
    let mut e = Effects::default();
    let mut st = vec![root];
    while let Some(n) = st.pop() {
        let k = cst.kind(n);
        match k {
            "return_statement" | "break_statement" | "continue_statement" | "goto_statement" | "labeled_statement"
            | "case_statement" | "throw_statement" => e.control = true,
            "call_expression" | "new_expression" | "delete_expression" => {
                e.mem_read = true;
                e.mem_write = true;
            }
            "assignment_expression" | "update_expression" => {
                let lhs = cst.child(n, if k == "update_expression" { "argument" } else { "left" });
                if let Some(l) = lhs {
                    match lvalue_local(cst, info, l) {
                        Some(v) => {
                            e.writes.insert(v.to_string());
                            if k == "update_expression" || cst.op(n) != Some("=") {
                                e.reads.insert(v.to_string());
                            }
                        }
                        None => e.mem_write = true,
                    }
                }
            }
            "field_expression" => {
                if cst.op(n) == Some("->") {
                    e.mem_read = true;
                }
            }
            "subscript_expression" => e.mem_read = true,
            "pointer_expression" => {
                if cst.op(n) == Some("*") {
                    e.mem_read = true;
                }
            }
            "this" => e.mem_read = true,
            "identifier" => {
                // Skip names in call position / qualified names / declarator names.
                let t = cst.text(n);
                let par = cst.parent(n);
                let in_decl = par.is_some_and(|p| {
                    let pk = cst.kind(p);
                    (pk == "init_declarator" && cst.nodes[n].field == Some("declarator"))
                        || (pk == "declaration" && cst.nodes[n].field == Some("declarator"))
                        || matches!(pk, "pointer_declarator" | "reference_declarator" | "array_declarator")
                });
                if in_decl {
                    e.writes.insert(t.to_string());
                } else if info.is_private(t) {
                    e.reads.insert(t.to_string());
                } else if par.is_some_and(|p| cst.kind(p) == "call_expression" && cst.nodes[n].field == Some("function")) {
                    // function name: covered by the call
                } else {
                    // Globals, members accessed implicitly, aliased locals: memory.
                    e.mem_read = true;
                    if info.vars.contains_key(t) {
                        e.reads.insert(t.to_string());
                    }
                }
            }
            _ => {}
        }
        // Writes to aliased locals are memory writes too (handled: lvalue_local None => mem_write).
        st.extend(cst.nodes[n].children.iter().copied());
    }
    // A declaration with a class type may run a constructor: treat as a call.
    if cst.kind(root) == "declaration" {
        if let Some(t) = cst.child(root, "type") {
            if !is_scalar_type(cst.text(t)) {
                e.mem_read = true;
                e.mem_write = true;
            }
        }
    }
    e
}

/// Builtin arithmetic / pointer types (and the project's common typedefs).
pub fn is_scalar_type(t: &str) -> bool {
    let t = t.trim();
    if t.ends_with('*') {
        return true;
    }
    let t = t.trim_start_matches("const ").trim_start_matches("volatile ").trim();
    matches!(
        t,
        "int" | "unsigned int" | "unsigned" | "signed int" | "short" | "unsigned short" | "signed short" | "char"
            | "unsigned char" | "signed char" | "long" | "unsigned long" | "long long" | "unsigned long long"
            | "float" | "double" | "bool" | "s8" | "u8" | "s16" | "u16" | "s32" | "u32" | "s64" | "u64" | "f32" | "f64"
            | "uint" | "ushort" | "uchar" | "size_t" | "BOOL" | "int8_t" | "uint8_t" | "int16_t" | "uint16_t"
            | "int32_t" | "uint32_t"
    )
}

pub fn is_int_type(t: &str) -> bool {
    is_scalar_type(t) && !t.ends_with('*') && !matches!(t.trim_start_matches("const ").trim(), "float" | "double" | "f32" | "f64")
}

/// Best-effort static type of an expression (text), or `None`.
pub fn infer_type(cst: &Cst, info: &FuncInfo, n: usize) -> Option<String> {
    match cst.kind(n) {
        "identifier" => info.vars.get(cst.text(n)).map(|v| v.ty.trim_end_matches('&').trim().trim_start_matches("const ").to_string()),
        "number_literal" => {
            let t = cst.text(n).to_ascii_lowercase();
            if t.starts_with("0x") {
                Some(if t.ends_with('u') { "unsigned int" } else { "int" }.into())
            } else if t.ends_with('f') {
                Some("float".into())
            } else if t.contains('.') || t.contains('e') {
                Some("double".into())
            } else if t.ends_with('u') {
                Some("unsigned int".into())
            } else {
                Some("int".into())
            }
        }
        "true" | "false" => Some("bool".into()),
        "cast_expression" => cst.child(n, "type").map(|t| cst.text(t).to_string()),
        "parenthesized_expression" => cst.named(n).first().and_then(|&c| infer_type(cst, info, c)),
        "binary_expression" => {
            let op = cst.op(n)?;
            if matches!(op, "<" | ">" | "<=" | ">=" | "==" | "!=" | "&&" | "||") {
                return Some("bool".into());
            }
            let l = infer_type(cst, info, cst.child(n, "left")?)?;
            let r = infer_type(cst, info, cst.child(n, "right")?)?;
            if l == r && (is_scalar_type(&l)) {
                let narrow = matches!(l.as_str(), "short" | "unsigned short" | "char" | "unsigned char" | "signed char" | "s8" | "u8" | "s16" | "u16" | "bool");
                if narrow { Some("int".into()) } else { Some(l) }
            } else if l.ends_with('*') && is_int_type(&r) && matches!(op, "+" | "-") {
                Some(l)
            } else {
                None
            }
        }
        "unary_expression" => {
            let op = cst.op(n)?;
            if op == "!" {
                return Some("bool".into());
            }
            infer_type(cst, info, cst.child(n, "argument")?)
        }
        _ => None,
    }
}

/// Type to declare a temporary holding expression `n`: inferred if possible, else `__typeof__`.
pub fn temp_type(cst: &Cst, info: &FuncInfo, n: usize) -> String {
    match infer_type(cst, info, n) {
        Some(t) if !t.is_empty() => t,
        _ => format!("__typeof__({})", cst.text(n)),
    }
}

