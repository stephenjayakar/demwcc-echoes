//! Structured CodeWarrior (GC) demangling into `mwdec_core::Type`, and C++ spelling of types.
//!
//! The textual demangler is `cwdemangle`; this module mirrors its grammar but produces types,
//! so that class names derived from DWARF symbol names (`rstl::auto_ptr<9CAnimData>`) and from
//! function symbols (`__ct__Q24rstl20auto_ptr<9CAnimData>Fv`) normalize to the same spelling
//! (`rstl::auto_ptr<CAnimData>`), which is the key of `TypeDb::classes`.
use mwdec_core::*;

/// Parser state over a mangled string.
pub struct P<'a> {
    s: &'a [u8],
    pub i: usize,
    /// raw (still mangled) text of the last name component parsed
    pub last_raw: String,
}

impl<'a> P<'a> {
    pub fn new(s: &'a str) -> Self {
        P { s: s.as_bytes(), i: 0, last_raw: String::new() }
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    pub fn at_end(&self) -> bool {
        self.i >= self.s.len()
    }
    pub fn rest(&self) -> &'a str {
        std::str::from_utf8(&self.s[self.i.min(self.s.len())..]).unwrap_or("")
    }
    fn digits(&mut self) -> Option<usize> {
        let st = self.i;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if st == self.i {
            return None;
        }
        std::str::from_utf8(&self.s[st..self.i]).ok()?.parse().ok()
    }

    /// `<len><name[<args>]>` -> demangled component (template args demangled).
    fn component(&mut self) -> Option<String> {
        let n = self.digits()?;
        if self.i + n > self.s.len() {
            return None;
        }
        let raw = std::str::from_utf8(&self.s[self.i..self.i + n]).ok()?;
        self.last_raw = raw.to_string();
        self.i += n;
        demangle_component(raw)
    }

    /// `Q<n>` + n components, or a single component.
    pub fn qualified_name(&mut self) -> Option<String> {
        if self.peek() == Some(b'Q') {
            self.i += 1;
            let c = self.peek()?;
            if !c.is_ascii_digit() {
                return None;
            }
            self.i += 1;
            let count = (c - b'0') as usize;
            let mut parts = Vec::new();
            for _ in 0..count {
                parts.push(self.component()?);
            }
            Some(parts.join("::"))
        } else {
            self.component()
        }
    }

    /// One mangled type.
    pub fn ty(&mut self) -> Option<Type> {
        // qualifiers are prefix operators; collect then apply innermost-last
        let mut quals: Vec<u8> = Vec::new();
        loop {
            match self.peek()? {
                q @ (b'P' | b'R' | b'C' | b'V' | b'U' | b'S') => {
                    quals.push(q);
                    self.i += 1;
                }
                _ => break,
            }
        }
        let mut unsigned = None;
        // U/S apply to the base type; strip them from the qualifier list
        quals.retain(|&q| match q {
            b'U' => {
                unsigned = Some(true);
                false
            }
            b'S' => {
                unsigned = Some(false);
                false
            }
            _ => true,
        });
        let base = match self.peek()? {
            b'0'..=b'9' | b'Q' => Type::Named(self.qualified_name()?),
            b'i' => {
                self.i += 1;
                Type::Int { size: 4, signed: unsigned != Some(true) }
            }
            b'l' => {
                self.i += 1;
                Type::Long { signed: unsigned != Some(true) }
            }
            b's' => {
                self.i += 1;
                Type::Int { size: 2, signed: unsigned != Some(true) }
            }
            b'x' => {
                self.i += 1;
                Type::Int { size: 8, signed: unsigned != Some(true) }
            }
            b'c' => {
                self.i += 1;
                match unsigned {
                    None => Type::Char,
                    Some(true) => Type::Int { size: 1, signed: false },
                    Some(false) => Type::Int { size: 1, signed: true },
                }
            }
            b'b' => {
                self.i += 1;
                Type::Bool
            }
            b'f' => {
                self.i += 1;
                Type::Float { size: 4 }
            }
            b'd' | b'r' => {
                self.i += 1;
                Type::Float { size: 8 }
            }
            b'w' => {
                self.i += 1;
                Type::WChar
            }
            b'v' => {
                self.i += 1;
                Type::Void
            }
            b'F' => {
                self.i += 1;
                let (params, variadic) = self.params()?;
                if self.peek() != Some(b'_') {
                    return None;
                }
                self.i += 1;
                let ret = self.ty()?;
                let sig = FuncSig {
                    qualified_name: String::new(),
                    mangled: None,
                    ret,
                    params,
                    this_class: None,
                    is_const: false,
                    is_static: false,
                    is_virtual: false,
                    variadic,
                };
                // A function type only appears under a pointer: P F... -> FuncPtr
                if quals.last() == Some(&b'P') {
                    quals.pop();
                }
                Type::FuncPtr(Box::new(sig))
            }
            b'M' => {
                self.i += 1;
                let class = self.qualified_name()?;
                // member function pointer: M<class>F PCvPv|PCvPCv <args> _ <ret>; data: M<class><type>
                if self.peek() == Some(b'F') {
                    self.i += 1;
                    if self.rest().starts_with("PCvPCv") {
                        self.i += 6;
                    } else if self.rest().starts_with("PCvPv") {
                        self.i += 5;
                    }
                    let _ = self.params()?;
                    if self.peek() == Some(b'_') {
                        self.i += 1;
                        let _ = self.ty()?;
                    }
                    Type::MemberPtr { class, size: 12 }
                } else {
                    let _ = self.ty()?;
                    Type::MemberPtr { class, size: 4 }
                }
            }
            b'A' => {
                self.i += 1;
                let n = self.digits()? as u32;
                if self.peek() != Some(b'_') {
                    return None;
                }
                self.i += 1;
                let elem = self.ty()?;
                Type::Array(Box::new(elem), n)
            }
            b'e' => {
                // variadic marker handled by params()
                return None;
            }
            _ => return None,
        };
        let mut t = base;
        for &q in quals.iter().rev() {
            t = match q {
                b'P' => Type::Ptr(Box::new(t)),
                b'R' => Type::Ref(Box::new(t)),
                b'C' => Type::Const(Box::new(t)),
                b'V' => Type::Volatile(Box::new(t)),
                _ => t,
            };
        }
        Some(t)
    }

    /// Function parameter list up to `_` or end. `v` alone = no params; `e` = variadic.
    pub fn params(&mut self) -> Option<(Vec<Param>, bool)> {
        let mut out = Vec::new();
        let mut variadic = false;
        if self.peek() == Some(b'v') && matches!(self.s.get(self.i + 1), None | Some(b'_')) {
            self.i += 1;
            return Some((out, false));
        }
        while !self.at_end() && self.peek() != Some(b'_') {
            if self.peek() == Some(b'e') {
                self.i += 1;
                variadic = true;
                continue;
            }
            let t = self.ty()?;
            out.push(Param { name: None, ty: t });
        }
        Some((out, variadic))
    }
}

/// Demangle a name component that may carry CW-mangled template args: `auto_ptr<9CAnimData>`.
pub fn demangle_component(raw: &str) -> Option<String> {
    let Some(lt) = raw.find('<') else { return Some(raw.to_string()) };
    if !raw.ends_with('>') {
        return Some(raw.to_string());
    }
    let base = &raw[..lt];
    let args = &raw[lt + 1..raw.len() - 1];
    let mut out = String::from(base);
    out.push('<');
    let mut p = P::new(args);
    let mut first = true;
    while !p.at_end() {
        if !first {
            out.push_str(", ");
        }
        first = false;
        // literal (possibly negative) followed by ',' or end
        let rest = p.rest();
        let lit_len = {
            let b = rest.as_bytes();
            let mut k = if b.first() == Some(&b'-') { 1 } else { 0 };
            let st = k;
            while k < b.len() && b[k].is_ascii_digit() {
                k += 1;
            }
            if k > st && (k == b.len() || b[k] == b',') {
                Some(k)
            } else {
                None
            }
        };
        if let Some(k) = lit_len {
            out.push_str(&rest[..k]);
            p.i += k;
        } else {
            let t = p.ty()?;
            out.push_str(&type_to_string(&t));
        }
        if p.peek() == Some(b',') {
            p.i += 1;
        } else if !p.at_end() {
            return None;
        }
    }
    if out.ends_with('>') {
        out.push(' ');
    }
    out.push('>');
    Some(out)
}

/// Normalize a qualified name whose components may carry mangled template args, e.g. the
/// DWARF symbol form `rstl::vector<11SConnection,Q24rstl17rmemory_allocator>`.
pub fn demangle_qualified(q: &str) -> String {
    split_scope(q)
        .iter()
        .map(|c| demangle_component(c).unwrap_or_else(|| c.to_string()))
        .collect::<Vec<_>>()
        .join("::")
}

/// Split at `::` outside template brackets.
pub fn split_scope(q: &str) -> Vec<&str> {
    let b = q.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut st = 0;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'<' => depth += 1,
            b'>' => depth -= 1,
            b':' if depth == 0 && i + 1 < b.len() && b[i + 1] == b':' => {
                out.push(&q[st..i]);
                i += 2;
                st = i;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    out.push(&q[st..]);
    out
}

/// C++ spelling of a type (cwdemangle style: `const CVector3f&`, `rstl::vector<int, ...>`).
pub fn type_to_string(t: &Type) -> String {
    declarator(t, "")
}

/// C++ declaration of `name` with type `t` (handles arrays and function pointers).
pub fn declarator(t: &Type, name: &str) -> String {
    let mut prefix = String::new();
    let mut suffix = String::new();
    build_decl(t, &mut prefix, &mut suffix, name);
    let s = format!("{prefix}{suffix}");
    s.trim().to_string()
}

fn base_name(t: &Type) -> Option<String> {
    Some(match t {
        Type::Void => "void".into(),
        Type::Bool => "bool".into(),
        Type::Char => "char".into(),
        Type::WChar => "wchar_t".into(),
        Type::Long { signed: true } => "long".into(),
        Type::Long { signed: false } => "unsigned long".into(),
        Type::Int { size, signed } => match (size, signed) {
            (1, true) => "signed char",
            (1, false) => "unsigned char",
            (2, true) => "short",
            (2, false) => "unsigned short",
            (4, true) => "int",
            (4, false) => "unsigned int",
            (8, true) => "long long",
            (8, false) => "unsigned long long",
            _ => "int",
        }
        .into(),
        Type::Float { size: 4 } => "float".into(),
        Type::Float { .. } => "double".into(),
        Type::Named(n) => n.clone(),
        Type::Unknown { size } => format!("__unknown{size}"),
        _ => return None,
    })
}

fn build_decl(t: &Type, prefix: &mut String, _suffix: &mut String, name: &str) {
    // `inner` is the declarator built so far; a non-empty name starts with a space so that
    // pointers read "T* name" and plain types "T name".
    fn go(t: &Type, inner: String) -> String {
        match t {
            Type::Const(x) | Type::Volatile(x) => {
                let kw = if matches!(t, Type::Const(_)) { "const" } else { "volatile" };
                match x.as_ref() {
                    // cv on a base type is written as a prefix: "const int x"
                    b if base_name(b).is_some() => join(&format!("{kw} {}", base_name(b).unwrap()), &inner),
                    // cv on a pointer: "T* const x"
                    _ => go(x, format!(" {kw}{inner}")),
                }
            }
            Type::Ptr(x) => go(x, format!("*{inner}")),
            Type::Ref(x) => go(x, format!("&{inner}")),
            Type::Array(x, n) => {
                let i = if inner.starts_with('*') || inner.starts_with('&') { format!(" ({})", inner) } else { inner };
                go(x, format!("{i}[{n}]"))
            }
            Type::FuncPtr(sig) => {
                let params: Vec<String> = sig.params.iter().map(|p| type_to_string(&p.ty)).collect();
                let mut ps = params.join(", ");
                if sig.variadic {
                    ps = if ps.is_empty() { "...".into() } else { format!("{ps}, ...") };
                }
                go(&sig.ret, format!(" (*{})({ps})", inner.trim_start()))
            }
            Type::MemberPtr { class, .. } => format!("void ({class}::*{})()", inner.trim_start()),
            b => join(&base_name(b).unwrap_or_else(|| "?".into()), &inner),
        }
    }
    fn join(bn: &str, inner: &str) -> String {
        format!("{bn}{inner}")
    }
    let n = if name.is_empty() { String::new() } else { format!(" {name}") };
    prefix.push_str(&go(t, n));
}

/// Split a mangled function symbol into (function-name part, rest after `__`) like cwdemangle.
fn find_split(s: &str, special: bool) -> Option<usize> {
    let b = s.as_bytes();
    let mut start = 0;
    if special && s.starts_with("op") {
        // conversion operator: __op<type>__<class>...
        let mut p = P::new(&s[2..]);
        p.ty()?;
        start = 2 + p.i;
    }
    let mut depth = 0i32;
    let mut i = start;
    while i + 1 < b.len() {
        match b[i] {
            b'<' => depth += 1,
            b'>' => depth -= 1,
            b'_' if b[i + 1] == b'_' && depth == 0 => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

/// Parsed pieces of a mangled function name.
#[derive(Clone, Debug)]
pub struct MangledFn {
    /// Simple function name as written in C++ (`SetActive`, `CActor`, `~CActor`, `operator=`).
    pub name: String,
    /// Owning class or namespace (demangled), if any.
    pub scope: Option<String>,
    pub is_const: bool,
    pub params: Vec<Param>,
    pub variadic: bool,
    /// Return type (only encoded for template functions).
    pub ret: Option<Type>,
    /// ct / dt / op...
    pub special: Option<String>,
    /// Template arguments of the owning class (`vector<9TUniqueId,...>`): (type, spelling).
    /// Integer literals are `Type::Named("4")`.
    pub scope_targs: Vec<(Type, String)>,
    /// Template arguments of a function template (`TCastToPtr<6CActor>`).
    pub fn_targs: Vec<(Type, String)>,
}

/// Template arguments of a raw CW name component (`auto_ptr<9CAnimData>`).
pub fn raw_template_args(raw: &str) -> Vec<(Type, String)> {
    let mut out = Vec::new();
    let Some(lt) = raw.find('<') else { return out };
    if !raw.ends_with('>') {
        return out;
    }
    let args = &raw[lt + 1..raw.len() - 1];
    let mut p = P::new(args);
    while !p.at_end() {
        let rest = p.rest();
        let b = rest.as_bytes();
        let mut k = if b.first() == Some(&b'-') { 1 } else { 0 };
        let st = k;
        while k < b.len() && b[k].is_ascii_digit() {
            k += 1;
        }
        if k > st && (k == b.len() || b[k] == b',') {
            out.push((Type::Named(rest[..k].to_string()), rest[..k].to_string()));
            p.i += k;
        } else {
            match p.ty() {
                Some(t) => {
                    let sp = type_to_string(&t);
                    out.push((t, sp));
                }
                None => return Vec::new(),
            }
        }
        if p.peek() == Some(b',') {
            p.i += 1;
        } else if !p.at_end() {
            return Vec::new();
        }
    }
    out
}

pub fn parse_mangled_fn(sym: &str) -> Option<MangledFn> {
    if !sym.is_ascii() {
        return None;
    }
    let mut s = sym;
    let special = s.starts_with("__");
    if special {
        s = &s[2..];
    }
    let mut idx = find_split(s, special)?;
    while s.as_bytes().get(idx + 2) == Some(&b'_') {
        idx += 1;
    }
    let (fname, rest) = s.split_at(idx);
    let mut p = P::new(&rest[2..]);
    let mut scope = None;
    let mut scope_raw_last = String::new();
    if p.peek() != Some(b'F') && !p.at_end() {
        scope = Some(p.qualified_name()?);
        scope_raw_last = p.last_raw.clone();
    }
    let mut is_const = false;
    if p.peek() == Some(b'C') {
        p.i += 1;
        is_const = true;
    }
    let (params, variadic) = if p.peek() == Some(b'F') {
        p.i += 1;
        p.params()?
    } else {
        // data symbol (static member / namespaced variable): not a function
        return None;
    };
    let mut ret = None;
    if p.peek() == Some(b'_') {
        p.i += 1;
        ret = Some(p.ty()?);
    }
    if !p.at_end() {
        return None;
    }
    let class_simple = scope.as_deref().map(|q| {
        let last = split_scope(q).last().copied().unwrap_or("").to_string();
        match last.find('<') {
            Some(i) => last[..i].to_string(),
            None => last,
        }
    });
    let (name, special_kind) = if special {
        let (op, _) = match fname.find('<') {
            Some(i) => (&fname[..i], &fname[i..]),
            None => (fname, ""),
        };
        let n = match op {
            "ct" => class_simple.clone().unwrap_or_default(),
            "dt" => format!("~{}", class_simple.clone().unwrap_or_default()),
            o if o.starts_with("op") => {
                let mut tp = P::new(&o[2..]);
                let t = tp.ty()?;
                format!("operator {}", type_to_string(&t))
            }
            o => match operator_name(o) {
                Some(n) => n.to_string(),
                None => format!("__{o}"),
            },
        };
        (n, Some(op.to_string()))
    } else {
        (demangle_component(fname)?, None)
    };
    let scope_targs = raw_template_args(&scope_raw_last);
    let fn_targs = if special { Vec::new() } else { raw_template_args(fname) };
    Some(MangledFn { name, scope, is_const, params, variadic, ret, special: special_kind, scope_targs, fn_targs })
}

pub fn operator_name(op: &str) -> Option<&'static str> {
    Some(match op {
        "nw" => "operator new",
        "nwa" => "operator new[]",
        "dl" => "operator delete",
        "dla" => "operator delete[]",
        "pl" => "operator+",
        "mi" => "operator-",
        "ml" => "operator*",
        "dv" => "operator/",
        "md" => "operator%",
        "er" => "operator^",
        "ad" => "operator&",
        "or" => "operator|",
        "co" => "operator~",
        "nt" => "operator!",
        "as" => "operator=",
        "lt" => "operator<",
        "gt" => "operator>",
        "apl" => "operator+=",
        "ami" => "operator-=",
        "amu" => "operator*=",
        "adv" => "operator/=",
        "amd" => "operator%=",
        "aer" => "operator^=",
        "aad" => "operator&=",
        "aor" => "operator|=",
        "ls" => "operator<<",
        "rs" => "operator>>",
        "ars" => "operator>>=",
        "als" => "operator<<=",
        "eq" => "operator==",
        "ne" => "operator!=",
        "le" => "operator<=",
        "ge" => "operator>=",
        "aa" => "operator&&",
        "oo" => "operator||",
        "pp" => "operator++",
        "mm" => "operator--",
        "cm" => "operator,",
        "rm" => "operator->*",
        "rf" => "operator->",
        "cl" => "operator()",
        "vc" => "operator[]",
        _ => return None,
    })
}

