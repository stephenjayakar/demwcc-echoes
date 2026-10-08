//! Sources for functions the compiler emits on demand rather than from a definition in the unit:
//! members of class templates and function templates defined in headers (instantiated), inline
//! functions of the context headers (emitted out of line when a use is not inlined) and special
//! members the class never declares (generated on use: destructor, copy constructor, assignment).
//!
//! A definition of such a function in the unit is a redefinition or an explicit specialization;
//! the source that produces it is code that makes the compiler emit it: an explicit
//! instantiation, a use (call, `new`, destructor call, assignment) or its address. `triggers`
//! lists candidate sources, most natural first; the caller compiles them and keeps the one whose
//! object has the symbol with the target's bytes. Everything is spelled from the demangled symbol
//! (and an optional return type for the forms that must name the function's type).

use crate::types::{split_closers, type_str};
use mwdec_core::Type;
use mwdec_lift::sig::{demangle, split_scope, split_top};

/// The demangled pieces of a member/free function symbol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parts {
    /// Enclosing class or namespace (`rstl::vector<float, rstl::rmemory_allocator>`), if any.
    pub scope: Option<String>,
    /// Unqualified name with its template arguments (`destroy<CFoo>`, `operator=`, `~vector`).
    pub name: String,
    /// Parameter type spellings (no `void`).
    pub params: Vec<String>,
    pub is_const: bool,
    pub variadic: bool,
}

/// Split a symbol's demangled text into `Parts` (None for unmangled names).
pub fn parts(symbol: &str) -> Option<Parts> {
    let text = demangle(symbol)?;
    let mut t = text.trim();
    let mut is_const = false;
    if let Some(r) = t.strip_suffix(" const") {
        is_const = true;
        t = r;
    }
    if !t.ends_with(')') {
        return None;
    }
    let b = t.as_bytes();
    let (mut depth, mut open) = (0i32, None);
    for i in (0..b.len()).rev() {
        match b[i] {
            b')' => depth += 1,
            b'(' => {
                depth -= 1;
                if depth == 0 {
                    open = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let o = open?;
    let (q, ps) = (&t[..o], &t[o + 1..t.len() - 1]);
    let mut variadic = false;
    let params = split_top(ps, ',')
        .into_iter()
        .filter(|p| !p.is_empty() && p != "void")
        .filter(|p| {
            if p == "..." {
                variadic = true;
                false
            } else {
                true
            }
        })
        .map(|p| split_closers(&p))
        .collect();
    let (scope, name) = split_scope(q);
    Some(Parts { scope: scope.map(split_closers), name: name.to_string(), params, is_const, variadic })
}

/// Does the symbol name an instance of a template (class template member or function template)?
pub fn is_template_instance(symbol: &str) -> bool {
    parts(symbol).is_some_and(|p| has_template_args(&p.name) || p.scope.as_deref().is_some_and(|s| s.contains('<')))
}

/// `T` + `name` with declarator syntax for function / member / array pointers.
fn named(ty: &str, name: &str) -> String {
    // the innermost declarator group: `(*)`, `(**)`, `(&)`, `(*&)`, `(C::*)`
    let b = ty.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if c != b'(' {
            continue;
        }
        let rest = &ty[i + 1..];
        let Some(close) = rest.find(')') else { break };
        let inner = &rest[..close];
        let head = inner.trim_end_matches(['*', '&']);
        if head.len() < inner.len() && (head.is_empty() || head.ends_with("::")) {
            let at = i + 1 + close;
            return format!("{}{}{}", &ty[..at], name, &ty[at..]);
        }
    }
    format!("{ty} {name}")
}

/// `rstl::vector<int, A>` -> `vector` (the class's own name, for constructor/destructor syntax).
fn class_base_name(cls: &str) -> &str {
    let last = split_scope(cls).1;
    last.split('<').next().unwrap_or(last).trim()
}

fn has_template_args(s: &str) -> bool {
    s.contains('<') && !s.starts_with("operator")
}

/// Is `scope` a class (not a namespace)? Template classes always are; otherwise `is_class` decides.
fn scope_is_class(scope: &str, is_class: &dyn Fn(&str) -> bool) -> bool {
    has_template_args(split_scope(scope).1) || is_class(scope)
}

/// Candidate sources (most natural first) that make the compiler emit `symbol`. `ret` is the
/// function's return type when known (needed only for the explicit-instantiation and address
/// forms); `is_class` says whether a qualifier names a class (vs a namespace).
pub fn triggers(symbol: &str, ret: Option<&Type>, is_class: &dyn Fn(&str) -> bool) -> Vec<String> {
    match parts(symbol) {
        Some(p) => triggers_for(&p, ret, is_class),
        None => Vec::new(),
    }
}

/// `triggers` for a function with C linkage (an unmangled symbol), from its declared signature.
pub fn triggers_c(sig: &mwdec_core::FuncSig) -> Vec<String> {
    if mwdec_lift::sig::ret_unknown(sig) || sig.qualified_name.contains("::") {
        return Vec::new();
    }
    let p = Parts {
        scope: None,
        name: sig.qualified_name.clone(),
        params: sig.params.iter().map(|q| type_str(&q.ty)).collect(),
        is_const: false,
        variadic: sig.variadic,
    };
    triggers_for(&p, Some(&sig.ret), &|_| false)
}

fn triggers_for(p: &Parts, ret: Option<&Type>, is_class: &dyn Fn(&str) -> bool) -> Vec<String> {
    if p.variadic {
        return Vec::new();
    }
    let args: Vec<String> = (0..p.params.len()).map(|i| format!("a{i}")).collect();
    let decls: Vec<String> = p.params.iter().zip(&args).map(|(t, a)| named(t, a)).collect();
    let types = p.params.join(", ");
    let call = args.join(", ");
    let ret_s = ret.filter(|t| !matches!(t, Type::Unknown { size: 0 })).map(type_str);
    let cv = if p.is_const { " const" } else { "" };
    let mut out = Vec::new();
    let with_params = |extra: &str| {
        let mut v: Vec<String> = Vec::new();
        if !extra.is_empty() {
            v.push(extra.to_string());
        }
        v.extend(decls.iter().cloned());
        v.join(", ")
    };
    let class = p.scope.as_deref().filter(|s| scope_is_class(s, is_class));
    match class {
        Some(cls) => {
            let base = class_base_name(cls);
            let templ = has_template_args(cls);
            if p.name == format!("~{base}") {
                out.push(format!("void Instantiate({cls}* obj) {{\n    obj->~{base}();\n}}\n"));
                out.push(format!("void Instantiate({cls}* obj) {{\n    delete obj;\n}}\n"));
                // (array deletion passes the destructor's address: never inlined)
                out.push(format!("void Instantiate({cls}* obj) {{\n    delete[] obj;\n}}\n"));
            } else if p.name == base {
                out.push(format!("void Instantiate({}) {{\n    new {cls}({call});\n}}\n", with_params("")));
                out.push(format!("void Instantiate({}) {{\n    {cls} obj({call});\n}}\n", with_params("")).replace(" obj();", " obj;"));
                if p.params.is_empty() {
                    // (array construction passes the constructor's address: never inlined)
                    out.push(format!("void Instantiate() {{\n    new {cls}[1];\n}}\n"));
                }
            } else {
                if templ {
                    out.push(format!("template class {cls};\n"));
                }
                let obj = format!("{}{cls}& obj", if p.is_const { "const " } else { "" });
                if has_template_args(&p.name) {
                    // a member template's arguments are deduced from the call
                    let plain = p.name.split('<').next().unwrap_or(&p.name);
                    out.push(format!("void Instantiate({}) {{\n    obj.{plain}({call});\n}}\n", with_params(&obj)));
                }
                out.push(format!("void Instantiate({}) {{\n    obj.{}({call});\n}}\n", with_params(&obj), p.name));
                if let Some(r) = &ret_s {
                    out.push(format!("{} = &{cls}::{};\n", named(&format!("{r} ({cls}::*)({types}){cv}"), "Instantiate"), p.name));
                    out.push(format!("{} = &{cls}::{};\n", named(&format!("{r} (*)({types})"), "Instantiate"), p.name));
                    if templ {
                        out.push(format!("template {r} {cls}::{}({types}){cv};\n", p.name));
                    }
                }
            }
        }
        None => {
            let q = match &p.scope {
                Some(s) => format!("{s}::{}", p.name),
                None => p.name.clone(),
            };
            if let Some(lt) = p.name.find('<').filter(|_| has_template_args(&p.name)) {
                // a function template's arguments are deduced from the call
                let plain = match &p.scope {
                    Some(s) => format!("{s}::{}", &p.name[..lt]),
                    None => p.name[..lt].to_string(),
                };
                out.push(format!("void Instantiate({}) {{\n    {plain}({call});\n}}\n", with_params("")));
            }
            out.push(format!("void Instantiate({}) {{\n    {q}({call});\n}}\n", with_params("")));
            if let Some(r) = &ret_s {
                if has_template_args(&p.name) {
                    out.push(format!("template {r} {q}({types});\n"));
                }
                out.push(format!("{} = &{q};\n", named(&format!("{r} (*)({types})"), "Instantiate")));
            }
        }
    }
    out
}
