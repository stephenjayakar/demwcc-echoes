//! Eval-harness classification: can a dataset function exist as standalone source at all?
//!
//! Some target functions are emitted by the compiler as a side effect of other code: inline
//! functions of the context headers, special members the class never declares (MWCC generates
//! the destructor / copy constructor / assignment on use), and members of class templates (or
//! function templates) defined in a header, which are instantiated rather than written. A
//! definition of those in the unit is an error (`redefined`, `illegal explicit template
//! specialization`) or a different (explicitly specialized) function. Bench and eval report them
//! in their own columns instead of counting them as failures; both can include them on request.
//!
//! Inputs are the function's signature (from its mangled name) and the context's `TypeDb`;
//! nothing here reads source.

use mwdec_core::{FuncSig, Type, TypeDb};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Standalone {
    /// A definition can be written in the unit.
    Yes,
    /// Defined inline in a context header (the same declaration with a body).
    HeaderInline,
    /// Compiler-generated / template-instantiated: no standalone definition exists. The string
    /// says which kind (`implicit-dtor`, `implicit-copy-ctor`, `implicit-default-ctor`,
    /// `implicit-assign`, `template-header-member`).
    Implicit(&'static str),
}

impl Standalone {
    /// Short status label used in bench/eval output.
    pub fn label(&self) -> &'static str {
        match self {
            Standalone::Yes => "standalone",
            Standalone::HeaderInline => "hdr-inline",
            Standalone::Implicit(_) => "implicit",
        }
    }
}

/// `rstl::vector<int, A>::reserve` -> `rstl::vector::reserve` (decl keys carry no template args).
fn strip_template_args(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// (scope, last component) at the last top-level `::`.
fn split_scope(q: &str) -> (Option<&str>, &str) {
    let b = q.as_bytes();
    let (mut depth, mut last) = (0i32, None);
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'<' | b'(' => depth += 1,
            b'>' | b')' => depth -= 1,
            b':' if depth == 0 && i + 1 < b.len() && b[i + 1] == b':' => {
                last = Some(i);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    match last {
        Some(i) => (Some(&q[..i]), &q[i + 2..]),
        None => (None, q),
    }
}

fn norm(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Does `t` name class `cls` (through cv/references)?
fn names_class(t: &Type, cls: &str) -> bool {
    match t {
        Type::Named(n) => {
            // (decls name the class as written inside it: `vector`, `const T&`-free spellings)
            let a = strip_template_args(n);
            let b = strip_template_args(cls);
            norm(n) == norm(cls) || norm(&a) == norm(&b) || split_scope(&a).1 == split_scope(&b).1
        }
        Type::Const(x) | Type::Volatile(x) | Type::Ref(x) => names_class(x, cls),
        _ => false,
    }
}

/// Classify a dataset function (see the module docs).
pub fn standalone(sig: &FuncSig, db: &TypeDb) -> Standalone {
    let q = &sig.qualified_name;
    let np = sig.params.len();
    if db.decls.get(q).is_some_and(|v| v.iter().any(|d| d.is_inline_defined && d.params.len() == np)) {
        return Standalone::HeaderInline;
    }
    let key = strip_template_args(q);
    // a member of a class template / a function template instance whose definition is in a header
    if q.contains('<') && db.decls.get(&key).is_some_and(|v| v.iter().any(|d| d.is_inline_defined && d.params.len() == np)) {
        return Standalone::Implicit("template-header-member");
    }
    let Some(cls) = sig.this_class.clone().or_else(|| split_scope(q).0.map(|s| s.to_string())) else { return Standalone::Yes };
    let ckey = strip_template_args(&cls);
    let last = split_scope(&ckey).1.to_string();
    let name = split_scope(&key).1.to_string();
    // only classes the context declares: a class of the unit's own source may declare anything
    let prefix = format!("{ckey}::");
    let known = db.classes.keys().any(|k| norm(k) == norm(&cls)) || db.decls.range(prefix.clone()..).next().is_some_and(|(k, _)| k.starts_with(&prefix));
    if !known {
        return Standalone::Yes;
    }
    let decls = |n: &str| db.decls.get(&format!("{ckey}::{n}")).cloned().unwrap_or_default();
    if name == format!("~{last}") {
        if decls(&name).is_empty() {
            return Standalone::Implicit("implicit-dtor");
        }
    } else if name == last {
        let ctors = decls(&name);
        if np == 0 && ctors.is_empty() {
            return Standalone::Implicit("implicit-default-ctor");
        }
        if np == 1 && names_class(&sig.params[0].ty, &cls) && matches!(sig.params[0].ty, Type::Ref(_)) && !ctors.iter().any(|d| d.params.len() == 1 && names_class(&d.params[0].ty, &cls)) {
            return Standalone::Implicit("implicit-copy-ctor");
        }
    } else if name == "operator=" && np == 1 && names_class(&sig.params[0].ty, &cls) && !decls("operator=").iter().any(|d| d.params.len() == 1 && names_class(&d.params[0].ty, &cls)) {
        return Standalone::Implicit("implicit-assign");
    }
    Standalone::Yes
}
