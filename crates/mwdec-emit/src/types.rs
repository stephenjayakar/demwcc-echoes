//! C++ spelling of IR types and declarators.

use mwdec_core::Type;
use std::cell::Cell;

thread_local! {
    /// Emitting C (not C++): no `bool` type.
    pub static C_MODE: Cell<bool> = const { Cell::new(false) };
    /// C mode: the context typedefs `bool` (its declarations spell it), so it can be spelled.
    pub static C_BOOL: Cell<bool> = const { Cell::new(false) };
    /// C mode: struct/union/enum tags (name -> keyword) that need an elaborated specifier.
    pub static C_TAGS: std::cell::RefCell<std::collections::HashMap<String, String>> = std::cell::RefCell::new(Default::default());
}

/// Abstract type spelling (`const CVector3f*`).
pub fn type_str(t: &Type) -> String {
    decl(t, "").trim_end().to_string()
}

/// Declaration `T name` with correct declarator syntax for pointers to arrays/functions.
pub fn decl(t: &Type, name: &str) -> String {
    let (base, d) = declarator(t, name.to_string());
    if d.is_empty() {
        base
    } else if d.starts_with('*') || d.starts_with('&') {
        // `T* name` / `T& name`
        let n = d.find(|c: char| c != '*' && c != '&').unwrap_or(d.len());
        let (ptrs, rest) = d.split_at(n);
        if rest.is_empty() {
            format!("{base}{ptrs}")
        } else if rest.starts_with("const") {
            format!("{base}{ptrs} {rest}")
        } else {
            format!("{base}{ptrs} {rest}")
        }
    } else if d.starts_with('(') {
        format!("{base} {d}")
    } else {
        format!("{base} {d}")
    }
}

fn base_name(t: &Type) -> String {
    match t {
        Type::Void => "void".into(),
        Type::Bool => (if C_MODE.with(|c| c.get()) && !C_BOOL.with(|c| c.get()) { "int" } else { "bool" }).into(),
        Type::Int { size: 1, signed: true } => "signed char".into(),
        Type::Int { size: 1, signed: false } => "unsigned char".into(),
        Type::Int { size: 2, signed: true } => "short".into(),
        Type::Int { size: 2, signed: false } => "unsigned short".into(),
        Type::Int { size: 4, signed: true } => "int".into(),
        Type::Int { size: 4, signed: false } => "unsigned int".into(),
        Type::Int { size: 8, signed: true } => "long long".into(),
        Type::Int { size: 8, signed: false } => "unsigned long long".into(),
        Type::Int { .. } => "int".into(),
        Type::Char => "char".into(),
        Type::WChar => "wchar_t".into(),
        Type::Long { signed: true } => "long".into(),
        Type::Long { signed: false } => "unsigned long".into(),
        Type::Float { size: 4 } => "float".into(),
        Type::Float { .. } => "double".into(),
        Type::Named(n) if n.contains("@unnamed@") && crate::strip_unnamed_ns(n) != *n => base_name(&Type::Named(crate::strip_unnamed_ns(n))),
        Type::Named(n) => {
            if C_MODE.with(|c| c.get()) {
                if let Some(kw) = C_TAGS.with(|t| t.borrow().get(n).cloned()) {
                    return format!("{kw} {n}");
                }
            }
            split_closers(n)
        }
        Type::Unknown { size: 1 } => "unsigned char".into(),
        Type::Unknown { size: 2 } => "unsigned short".into(),
        Type::Unknown { size: 8 } => "long long".into(),
        Type::Unknown { size: 0 } => "void".into(),
        Type::Unknown { size: 4 } => "int".into(),
        // odd-sized unknown objects are byte buffers
        Type::Unknown { .. } => "unsigned char".into(),
        Type::MemberPtr { .. } => "int".into(),
        _ => "int".into(),
    }
}

/// Returns (base specifier, declarator) for `t` applied to inner declarator `inner`.
fn declarator(t: &Type, inner: String) -> (String, String) {
    match t {
        Type::Const(x) => {
            match &**x {
                // `T* const` -> const applies to the pointer
                Type::Ptr(_) => {
                    let (b, d) = declarator(x, format!("const {inner}").trim_end().to_string());
                    (b, d)
                }
                _ => {
                    let (b, d) = declarator(x, inner);
                    (format!("const {b}"), d)
                }
            }
        }
        Type::Volatile(x) => {
            let (b, d) = declarator(x, inner);
            (format!("volatile {b}"), d)
        }
        Type::Ptr(x) => {
            let d = format!("*{inner}");
            // (through a cv-qualified array: `const T (*p)[n]`)
            let d = if matches!(unqualified(x), Type::Array(..) | Type::FuncPtr(_)) { format!("({d})") } else { d };
            declarator(x, d)
        }
        Type::Ref(x) => {
            let d = format!("&{inner}");
            let d = if matches!(unqualified(x), Type::Array(..)) { format!("({d})") } else { d };
            declarator(x, d)
        }
        Type::Array(x, n) if *n == 0 => declarator(x, format!("{inner}[]")),
        Type::Array(x, n) => declarator(x, format!("{inner}[{n}]")),
        Type::FuncPtr(sig) => {
            let params: Vec<String> = sig.params.iter().map(|p| type_str(&p.ty)).collect();
            let ret = type_str(&sig.ret);
            (ret, format!("(*{inner})({})", params.join(", ")))
        }
        Type::MemberPtr { size, .. } if *size > 4 => ("int".into(), format!("{inner}[{}]", size / 4)),
        t => (base_name(t), inner),
    }
}

/// `A<B<C>>` -> `A<B<C> >`: the compiler reads `>>` as a shift (C++98), so nested template
/// argument lists close with a space (operator names keep theirs).
pub fn split_closers(s: &str) -> String {
    if !s.contains(">>") {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 4);
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'>' && i + 1 < b.len() && b[i + 1] == b'>' && !out.ends_with("operator") && !out.ends_with("operator>") {
            out.push_str("> ");
            i += 1;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// `t` without its outer const/volatile qualifiers.
fn unqualified(t: &Type) -> &Type {
    match t {
        Type::Const(x) | Type::Volatile(x) => unqualified(x),
        t => t,
    }
}
