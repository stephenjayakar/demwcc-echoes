//! Function signatures from CodeWarrior mangled names (via `cwdemangle`), with C++ type parsing
//! of the demangled text. Return types are not part of CW mangling (except templates), so they
//! come from the `TypeDb` when available, else they are inferred by the lifter.

use mwdec_core::{FuncSig, Param, Type, TypeDb};

/// Demangle to text (None for C symbols / non-mangled names).
pub fn demangle(sym: &str) -> Option<String> {
    // dtk disambiguates duplicate local names with an address suffix (`f__Fv_80412345`); the
    // suffixed name may still demangle (into a bogus trailing parameter type), so the stripped
    // name wins whenever it demangles
    let s = strip_dtk_suffix(sym);
    if s.len() < sym.len() {
        if let Some(d) = cwdemangle::demangle(s, &cwdemangle::DemangleOptions::default()) {
            return Some(d);
        }
    }
    cwdemangle::demangle(sym, &cwdemangle::DemangleOptions::default())
}

/// `name_80412345` -> `name` (dtk's address suffix on duplicate local symbols).
pub fn strip_dtk_suffix(sym: &str) -> &str {
    let b = sym.as_bytes();
    if b.len() > 9 && b[b.len() - 9] == b'_' && b[b.len() - 8..].iter().all(|c| c.is_ascii_digit() || (b'A'..=b'F').contains(c)) && b[b.len() - 8] == b'8' {
        &sym[..sym.len() - 9]
    } else {
        sym
    }
}

/// Split `s` at top-level occurrences of `sep` (not inside <>, (), []).
pub fn split_top(s: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            _ => {}
        }
        if c == sep && depth == 0 {
            out.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    if !cur.trim().is_empty() || !out.is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Split a qualified name at the last top-level `::` -> (scope, last).
pub fn split_scope(q: &str) -> (Option<&str>, &str) {
    let b = q.as_bytes();
    let mut depth = 0i32;
    let mut last = None;
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

/// Parse a C++ type spelling as produced by cwdemangle (`const CVector3f&`, `unsigned char*`,
/// `rstl::vector<int, rstl::rmemory_allocator>&`, `void (*)(int)`).
pub fn parse_type(s: &str) -> Type {
    let s = s.trim();
    if s.is_empty() {
        return Type::Unknown { size: 4 };
    }
    // function pointer: R (*)(args) / R (C::*)(args)
    // pointer to array: T (*)[N]
    if let Some(p) = s.find("(*)[") {
        let rest = &s[p + 3..];
        if let Some(n) = rest.strip_prefix('[').and_then(|r| r.split(']').next()).and_then(|n| n.parse::<u32>().ok()) {
            return Type::Ptr(Box::new(Type::Array(Box::new(parse_type(&s[..p])), n)));
        }
    }
    if let Some(p) = s.find("(*)") {
        let ret = parse_type(&s[..p]);
        let args = &s[p + 3..];
        let args = args.trim().trim_start_matches('(');
        let args = args.strip_suffix(')').unwrap_or(args);
        let params = split_top(args, ',')
            .into_iter()
            .filter(|a| !a.is_empty() && a != "void")
            .map(|a| Param { name: None, ty: parse_type(&a) })
            .collect();
        return Type::FuncPtr(Box::new(FuncSig {
            qualified_name: String::new(),
            mangled: None,
            ret,
            params,
            this_class: None,
            is_const: false,
            is_static: false,
            is_virtual: false,
            variadic: false,
            runs_code: false,
        }));
    }
    if s.contains("::*") {
        let class = s.split("::*").next().unwrap_or("").rsplit(['(', ' ']).next().unwrap_or("").to_string();
        let size = if s.contains(")(") { 12 } else { 4 };
        return Type::MemberPtr { class, size };
    }
    if let Some(r) = s.strip_suffix('&') {
        return Type::Ref(Box::new(parse_type(r)));
    }
    if let Some(r) = s.strip_suffix('*') {
        return Type::Ptr(Box::new(parse_type(r)));
    }
    if let Some(r) = s.strip_suffix(" const") {
        return Type::Const(Box::new(parse_type(r)));
    }
    if let Some(r) = s.strip_suffix(" volatile") {
        return Type::Volatile(Box::new(parse_type(r)));
    }
    if let Some(r) = s.strip_prefix("const ") {
        return Type::Const(Box::new(parse_type(r)));
    }
    if let Some(r) = s.strip_prefix("volatile ") {
        return Type::Volatile(Box::new(parse_type(r)));
    }
    // array `T[4]`
    if s.ends_with(']') {
        if let Some(lb) = s.rfind('[') {
            if let Ok(n) = s[lb + 1..s.len() - 1].parse::<u32>() {
                return Type::Array(Box::new(parse_type(&s[..lb])), n);
            }
        }
    }
    match s {
        "void" => Type::Void,
        "bool" => Type::Bool,
        "char" => Type::Char,
        "signed char" => Type::Int { size: 1, signed: true },
        "unsigned char" => Type::Int { size: 1, signed: false },
        "short" | "signed short" => Type::Int { size: 2, signed: true },
        "unsigned short" => Type::Int { size: 2, signed: false },
        "wchar_t" => Type::WChar,
        "int" | "signed int" => Type::Int { size: 4, signed: true },
        "long" | "signed long" => Type::Long { signed: true },
        "unsigned int" => Type::Int { size: 4, signed: false },
        "unsigned long" => Type::Long { signed: false },
        "long long" | "signed long long" => Type::Int { size: 8, signed: true },
        "unsigned long long" => Type::Int { size: 8, signed: false },
        "float" => Type::Float { size: 4 },
        "double" | "long double" => Type::Float { size: 8 },
        _ => Type::Named(s.to_string()),
    }
}

/// Normalize a type/class name for TypeDb lookups (remove spaces).
pub fn norm_name(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Known namespaces (not classes) used by the project; everything else with a scope is assumed
/// to be a class when the TypeDb doesn't say otherwise.
fn is_namespace_guess(scope: &str, db: Option<&TypeDb>) -> bool {
    if let Some(db) = db {
        if find_class(db, scope).is_some() {
            return false;
        }
    }
    let last = split_scope(scope).1;
    // the anonymous namespace (`@unnamed@CFoo_cpp@`)
    if last.starts_with("@unnamed@") {
        return true;
    }
    // Project namespaces are lower-case (`rstl`, `std`, `nl`); classes are `CFoo`/`SFoo`/`TFoo`.
    last.chars().next().map_or(false, |c| c.is_ascii_lowercase()) && !last.contains('<')
}

/// Find a class in the TypeDb by (qualified) name, tolerating spacing differences.
pub fn find_class<'a>(db: &'a TypeDb, name: &str) -> Option<&'a mwdec_core::Class> {
    if let Some(c) = db.classes.get(name) {
        return Some(c);
    }
    let n = norm_name(name);
    // only keys with whitespace can match a different spelling: look them up in a per-thread
    // index of this TypeDb (misses were a scan over every class with an allocation each)
    let id = (db as *const TypeDb as usize, db.classes.len());
    let key = SPACED.with(|s| {
        let mut s = s.borrow_mut();
        if s.0 != id {
            s.0 = id;
            s.1 = db.classes.keys().filter(|k| k.chars().any(char::is_whitespace)).map(|k| (norm_name(k), k.clone())).collect();
        }
        s.1.get(&n).cloned()
    });
    key.and_then(|k| db.classes.get(&k))
}

/// Forget the per-thread class-name index of [`find_class`] (keyed by the TypeDb's address and
/// size, which a later TypeDb can reuse once the earlier one is dropped).
pub fn reset_memos() {
    SPACED.with(|s| {
        let mut s = s.borrow_mut();
        s.0 = (0, 0);
        s.1.clear();
    });
}

thread_local! {
    static SPACED: std::cell::RefCell<((usize, usize), std::collections::HashMap<String, String>)> = std::cell::RefCell::new(((0, 0), std::collections::HashMap::new()));
}

thread_local! {
    static CALLEE_SIGS: std::cell::RefCell<Option<std::sync::Arc<std::collections::HashMap<String, FuncSig>>>> = const { std::cell::RefCell::new(None) };
}

/// Run `f` with the signatures of C functions the context doesn't declare that the drafter
/// lifted from their own code (placeholder-named functions of the unit): calls of them pass
/// those parameters and the draft declares them so (every draft of the unit agrees with the
/// callee's own definition).
pub fn with_callee_sigs<R>(sigs: Option<std::sync::Arc<std::collections::HashMap<String, FuncSig>>>, f: impl FnOnce() -> R) -> R {
    let prev = CALLEE_SIGS.with(|c| std::mem::replace(&mut *c.borrow_mut(), sigs));
    let r = f();
    CALLEE_SIGS.with(|c| *c.borrow_mut() = prev);
    r
}

/// The lifted signature of an undeclared C function (see [`with_callee_sigs`]).
pub fn callee_sig(name: &str) -> Option<FuncSig> {
    CALLEE_SIGS.with(|c| c.borrow().as_ref().and_then(|m| m.get(name).cloned()))
}

/// Signature of a mangled function symbol. The return type is `Unknown{0}` ("not known") unless
/// the TypeDb has the function; callers infer it.
pub fn sig_of(mangled: &str, db: Option<&TypeDb>) -> FuncSig {
    if let Some(db) = db {
        if let Some(mut s) = mwdec_ctx::sig_from_mangled(mangled, db) {
            if s.mangled.is_none() {
                s.mangled = Some(mangled.to_string());
            }
            return s;
        }
        if let Some(s) = db.functions.get(mangled) {
            let mut s = s.clone();
            if s.mangled.is_none() {
                s.mangled = Some(mangled.to_string());
            }
            return s;
        }
    }
    let Some(text) = demangle(mangled) else {
        // C function
        return FuncSig {
            qualified_name: mangled.to_string(),
            mangled: Some(mangled.to_string()),
            ret: Type::Unknown { size: 0 },
            params: Vec::new(),
            this_class: None,
            is_const: false,
            is_static: false,
            is_virtual: false,
            variadic: false,
            runs_code: false,
        };
    };
    let mut sig = parse_demangled(&text, db);
    sig.mangled = Some(mangled.to_string());
    if let Some(db) = db {
        // A DB entry keyed by qualified name with matching param count.
        if let Some(s) = db.functions.get(&sig.qualified_name) {
            if s.params.len() == sig.params.len() {
                sig.ret = s.ret.clone();
                sig.is_static = s.is_static;
                sig.is_virtual = s.is_virtual;
            }
        }
        // Method list of the class.
        if let Some(cls) = sig.this_class.as_deref().and_then(|c| find_class(db, c)) {
            let last = split_scope(&sig.qualified_name).1.to_string();
            for m in cls.methods.iter().chain(cls.vtable.iter().map(|v| &v.sig)) {
                if m.mangled.as_deref() == Some(mangled)
                    || (split_scope(&m.qualified_name).1 == last
                        && m.params.len() == sig.params.len()
                        && m.is_const == sig.is_const
                        && params_compatible(&m.params, &sig.params))
                {
                    sig.ret = m.ret.clone();
                    sig.is_static = m.is_static;
                    sig.is_virtual |= m.is_virtual;
                    for (p, q) in sig.params.iter_mut().zip(&m.params) {
                        if p.name.is_none() {
                            p.name = q.name.clone();
                        }
                    }
                    break;
                }
            }
        }
    }
    sig
}

fn params_compatible(a: &[Param], b: &[Param]) -> bool {
    a.iter().zip(b).all(|(x, y)| type_key(&x.ty) == type_key(&y.ty) || matches!(x.ty, Type::Unknown { .. }))
}

fn type_key(t: &Type) -> String {
    norm_name(&format!("{t:?}"))
}

/// Parse `Scope::Name(params) const` text.
pub fn parse_demangled(text: &str, db: Option<&TypeDb>) -> FuncSig {
    let mut t = text.trim();
    let mut is_const = false;
    if let Some(r) = t.strip_suffix(" const") {
        is_const = true;
        t = r;
    }
    // find the matching '(' of the final ')'
    let (name, params) = if t.ends_with(')') {
        let b = t.as_bytes();
        let mut depth = 0;
        let mut open = None;
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
        match open {
            Some(o) => (&t[..o], &t[o + 1..t.len() - 1]),
            None => (t, ""),
        }
    } else {
        (t, "")
    };
    let mut variadic = false;
    let params: Vec<Param> = split_top(params, ',')
        .into_iter()
        .filter(|p| !p.is_empty() && p != "void")
        .filter_map(|p| {
            if p == "..." {
                variadic = true;
                None
            } else {
                Some(Param { name: None, ty: parse_type(&p) })
            }
        })
        .collect();
    let (scope, _last) = split_scope(name);
    let this_class = match scope {
        Some(s) if !is_namespace_guess(s, db) => Some(s.to_string()),
        _ => None,
    };
    FuncSig {
        qualified_name: name.to_string(),
        mangled: None,
        ret: Type::Unknown { size: 0 },
        params,
        this_class,
        is_const,
        is_static: false,
        is_virtual: false,
        variadic,
        runs_code: false,
    }
}

/// True when the return type is "not known" (placeholder from `sig_of`).
pub fn ret_unknown(sig: &FuncSig) -> bool {
    matches!(sig.ret, Type::Unknown { size: 0 })
}

pub fn is_ctor(sig: &FuncSig) -> bool {
    let (scope, last) = split_scope(&sig.qualified_name);
    match scope {
        Some(s) => {
            let cls_last = split_scope(s).1;
            let cls_base = cls_last.split('<').next().unwrap_or(cls_last);
            last == cls_last || last == cls_base
        }
        None => false,
    }
}

pub fn is_dtor(sig: &FuncSig) -> bool {
    split_scope(&sig.qualified_name).1.starts_with('~')
}

