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

use mwdec_core::{DeclInfo, FuncSig, Type, TypeDb};

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
    // an operator's name (`operator->`, `operator<=`, `operator<<`) is not a template argument list
    if let Some(i) = s.match_indices("operator").map(|(i, _)| i).find(|&i| i == 0 || s[..i].ends_with("::")) {
        return format!("{}{}", if i > 0 { strip_template_args(&s[..i]) } else { String::new() }, &s[i..]);
    }
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

/// `t` with typedefs resolved and cv-qualifiers on by-value parameters dropped, as a comparable key.
fn type_key(t: &Type, db: &TypeDb, depth: u32) -> String {
    match t {
        Type::Named(n) if depth < 8 => match db.typedefs.get(n) {
            Some(u) => type_key(u, db, depth + 1),
            // (headers spell names relative to the enclosing scope: compare the last component)
            None => norm(split_scope(&strip_template_args(n)).1),
        },
        Type::Ptr(x) => format!("{}*", type_key(x, db, depth + 1)),
        Type::Ref(x) => format!("{}&", type_key(x, db, depth + 1)),
        Type::Const(x) => format!("const {}", type_key(x, db, depth + 1)),
        Type::Volatile(x) => format!("volatile {}", type_key(x, db, depth + 1)),
        // (`long` and `int`, `char` and `signed char` differ in mangling but a typedef may hide
        // either spelling: compare by size and signedness)
        t => match t.int_info() {
            Some((sz, sg)) if !matches!(t, Type::Bool) => format!("i{sz}{sg}"),
            _ => norm(&format!("{t:?}")),
        },
    }
}

fn param_key(t: &Type, db: &TypeDb) -> String {
    match t {
        Type::Const(x) | Type::Volatile(x) => type_key(x, db, 0),
        t => type_key(t, db, 0),
    }
}

/// The header declarations of `q` that can be the symbol's: same parameter count and, among
/// overloads, the ones whose parameter types match the symbol's (all same-count ones when none
/// matches by type, e.g. template parameters spelled `T`).
fn candidates<'a>(decls: &'a [DeclInfo], sig: &FuncSig, db: &TypeDb) -> Vec<&'a DeclInfo> {
    let np = sig.params.len();
    let same: Vec<&DeclInfo> = decls.iter().filter(|d| d.params.len() == np && d.is_const == sig.is_const).collect();
    let same = if same.is_empty() { decls.iter().filter(|d| d.params.len() == np).collect() } else { same };
    let want: Vec<String> = sig.params.iter().map(|p| param_key(&p.ty, db)).collect();
    let typed: Vec<&DeclInfo> = same.iter().copied().filter(|d| d.params.iter().map(|p| param_key(&p.ty, db)).collect::<Vec<_>>() == want).collect();
    if typed.is_empty() {
        same
    } else {
        typed
    }
}

/// Classify a dataset function (see the module docs).
pub fn standalone(sig: &FuncSig, db: &TypeDb) -> Standalone {
    let q = &sig.qualified_name;
    let np = sig.params.len();
    // `this`-adjusting thunks of virtual overrides reached through a secondary base
    // (`@4@Method__5CFooFv`): generated with the override
    let thunk = |s: &str| s.strip_prefix('@').is_some_and(|r| r.split('@').next().is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit())));
    if sig.mangled.as_deref().is_some_and(thunk) || thunk(split_scope(q).1) {
        return Standalone::Implicit("thunk");
    }
    // (an overload declared without a body is defined in the unit even when another overload
    // with as many parameters is inline)
    if db.decls.get(q).is_some_and(|v| {
        let c = candidates(v, sig, db);
        !c.is_empty() && c.iter().all(|d| d.is_inline_defined)
    }) {
        return Standalone::HeaderInline;
    }
    let key = strip_template_args(q);
    // a member of a class template / a function template instance whose definition is in a header
    if q.contains('<') && db.decls.get(&key).is_some_and(|v| candidates(v, sig, db).iter().any(|d| d.is_inline_defined)) {
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
