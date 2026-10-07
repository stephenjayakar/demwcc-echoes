//! Naturalness of drafts: a metric of decompiler-style constructs, and a polish pass that rewrites
//! an already exact source towards ordinary hand-written C++ while keeping it exact.
//!
//! The metric counts what code review rejects: raw `*(T*)((char*)p + 0x10)` accesses, casts,
//! register-named locals, temporaries used once, `goto`, `extern "C" fn_X` stubs, unnamed
//! parameters (`arg0`), `this->` prefixes and negated comparisons.
//!
//! The polish pass works on the source text. Renames and `this->` removal don't change what the
//! compiler sees; every other rewrite (dropping a cast, inlining a temporary, flipping a negated
//! comparison) is kept only when the caller's checker says the result still matches.

use mwdec_core::TypeDb;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum K {
    Id,
    Num,
    Str,
    Punct,
    /// whitespace, comments and preprocessor lines
    Ws,
}

#[derive(Clone, Debug)]
struct Tok {
    k: K,
    s: String,
}

fn lex(src: &str) -> Vec<Tok> {
    const P3: [&str; 3] = ["<<=", ">>=", "..."];
    const P2: [&str; 19] = ["->", "::", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|="];
    let b = src.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    let mut line_start = true;
    while i < b.len() {
        let c = b[i];
        let start = i;
        let k;
        if c == b'#' && line_start {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            k = K::Ws;
        } else if c.is_ascii_whitespace() {
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            k = K::Ws;
        } else if src[i..].starts_with("//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            k = K::Ws;
        } else if src[i..].starts_with("/*") {
            i = src[i + 2..].find("*/").map_or(b.len(), |p| i + 2 + p + 2);
            k = K::Ws;
        } else if c == b'"' || c == b'\'' {
            i += 1;
            while i < b.len() && b[i] != c {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i = (i + 1).min(b.len());
            k = K::Str;
        } else if c.is_ascii_alphabetic() || c == b'_' || c == b'$' {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'$') {
                i += 1;
            }
            k = K::Id;
        } else if c.is_ascii_digit() || (c == b'.' && b.get(i + 1).map_or(false, |d| d.is_ascii_digit())) {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'.' || ((b[i] == b'-' || b[i] == b'+') && matches!(b[i - 1], b'e' | b'E') && !src[start..i].starts_with("0x"))) {
                i += 1;
            }
            k = K::Num;
        } else {
            let rest = &src[i..];
            i += P3.iter().chain(P2.iter()).find(|p| rest.starts_with(**p)).map_or(1, |p| p.len());
            // keep i on a char boundary for non-ASCII input
            while !src.is_char_boundary(i) {
                i += 1;
            }
            k = K::Punct;
        }
        let s = &src[start..i];
        line_start = if k == K::Ws { s.contains('\n') || (line_start && !s.starts_with('#')) } else { false };
        out.push(Tok { k, s: s.to_string() });
    }
    out
}

fn join(t: &[Tok]) -> String {
    t.iter().map(|t| t.s.as_str()).collect()
}

/// Indices of the non-whitespace tokens.
fn sig(t: &[Tok]) -> Vec<usize> {
    (0..t.len()).filter(|&i| t[i].k != K::Ws).collect()
}

const BUILTIN: &[&str] = &[
    "int", "unsigned", "signed", "char", "short", "long", "float", "double", "bool", "void", "const", "volatile", "u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64", "f32",
    "f64", "uint", "size_t", "wchar_t",
];

const KEYWORDS: &[&str] = &[
    "if", "else", "while", "for", "do", "switch", "case", "default", "return", "break", "continue", "goto", "sizeof", "new", "delete", "this", "true", "false", "nullptr",
    "static", "extern", "struct", "class", "union", "enum", "typedef", "operator", "template", "typename", "virtual", "inline", "namespace", "using", "throw", "static_cast",
    "reinterpret_cast", "const_cast", "dynamic_cast", "NULL",
];

fn is_builtin(s: &str) -> bool {
    BUILTIN.contains(&s)
}

/// Token range `[a, b)` (significant indices into `sg`) spells a type: `unsigned int`, `CFoo*`,
/// `const CBar&`, `rstl::vector< int >*`, `CState2::EType`.
fn looks_like_type(t: &[Tok], sg: &[usize], a: usize, b: usize) -> bool {
    if a >= b {
        return false;
    }
    let mut saw_name = false;
    let mut builtin = false;
    let mut upper = false;
    let mut ptr = false;
    for &i in &sg[a..b] {
        let s = t[i].s.as_str();
        match t[i].k {
            K::Id => {
                if KEYWORDS.contains(&s) && s != "struct" && s != "class" && s != "union" && s != "enum" {
                    return false;
                }
                if ptr {
                    // `T* const` only
                    if s != "const" {
                        return false;
                    }
                }
                builtin |= is_builtin(s);
                upper |= s.chars().next().map_or(false, |c| c.is_ascii_uppercase());
                saw_name = true;
            }
            K::Punct if s == "*" || s == "&" => ptr = true,
            K::Punct if s == "::" || s == "<" || s == ">" || s == "," || s == ">>" => {}
            K::Num => {}
            _ => return false,
        }
    }
    saw_name && (builtin || upper || ptr)
}

/// Matching close of the bracket at significant position `p`.
fn close_of(t: &[Tok], sg: &[usize], p: usize) -> Option<usize> {
    let (o, c) = match t[sg[p]].s.as_str() {
        "(" => ("(", ")"),
        "[" => ("[", "]"),
        "{" => ("{", "}"),
        _ => return None,
    };
    let mut d = 0i32;
    for q in p..sg.len() {
        let s = t[sg[q]].s.as_str();
        if s == o {
            d += 1;
        } else if s == c {
            d -= 1;
            if d == 0 {
                return Some(q);
            }
        }
    }
    None
}

/// A C-style cast found in the token stream.
#[derive(Clone, Debug)]
struct Cast {
    /// significant positions of `(` and `)`
    open: usize,
    close: usize,
    /// end (exclusive, significant position) of the cast's operand (a unary/postfix expression)
    operand_end: usize,
    ptr: bool,
    /// cast to a byte pointer (`(char*)`, `(u8*)`, `(unsigned char*)`)
    byte_ptr: bool,
}

/// End (exclusive) of the unary/postfix expression starting at significant position `p`.
fn operand_end(t: &[Tok], sg: &[usize], mut p: usize) -> usize {
    // prefix operators and nested casts
    loop {
        if p >= sg.len() {
            return p;
        }
        let s = t[sg[p]].s.as_str();
        if matches!(s, "*" | "&" | "-" | "!" | "~" | "++" | "--" | "+") {
            p += 1;
            continue;
        }
        if s == "(" {
            if let Some(c) = close_of(t, sg, p) {
                if looks_like_type(t, sg, p + 1, c) && starts_operand(t, sg, c + 1) {
                    p = c + 1;
                    continue;
                }
            }
        }
        break;
    }
    // primary
    let s = t[sg[p]].s.as_str();
    if s == "(" {
        p = close_of(t, sg, p).map_or(sg.len(), |c| c + 1);
    } else {
        p += 1;
        // qualified names
        while p + 1 < sg.len() && t[sg[p]].s == "::" && t[sg[p + 1]].k == K::Id {
            p += 2;
        }
    }
    // postfix
    loop {
        if p >= sg.len() {
            return p;
        }
        let s = t[sg[p]].s.as_str();
        match s {
            "(" | "[" => p = close_of(t, sg, p).map_or(sg.len(), |c| c + 1),
            "." | "->" => {
                p += 1;
                while p < sg.len() && (t[sg[p]].k == K::Id || t[sg[p]].s == "::" || t[sg[p]].s == "~") {
                    p += 1;
                }
            }
            "++" | "--" => p += 1,
            _ => return p,
        }
    }
}

fn starts_operand(t: &[Tok], sg: &[usize], p: usize) -> bool {
    let Some(&i) = sg.get(p) else { return false };
    match t[i].k {
        K::Id => !matches!(t[i].s.as_str(), "else" | "return" | "case"),
        K::Num | K::Str => true,
        K::Punct => matches!(t[i].s.as_str(), "(" | "*" | "&" | "-" | "!" | "~" | "++" | "--"),
        K::Ws => false,
    }
}

fn find_casts(t: &[Tok], sg: &[usize]) -> Vec<Cast> {
    let mut out = vec![];
    for p in 0..sg.len() {
        if t[sg[p]].s != "(" {
            continue;
        }
        // a call, a declarator or a keyword's parenthesis
        if p > 0 {
            let prev = &t[sg[p - 1]];
            if prev.k == K::Id && !matches!(prev.s.as_str(), "return" | "case" | "else" | "delete" | "throw") {
                continue;
            }
            if prev.k == K::Num || prev.k == K::Str || prev.s == ")" || prev.s == "]" || prev.s == ">" {
                continue;
            }
        }
        let Some(c) = close_of(t, sg, p) else { continue };
        if !looks_like_type(t, sg, p + 1, c) || !starts_operand(t, sg, c + 1) {
            continue;
        }
        // the cast is written tight against its operand by the emitter; `(a) b` isn't a cast
        if sg[c] + 1 != sg[c + 1] {
            continue;
        }
        let inner: Vec<&str> = sg[p + 1..c].iter().map(|&i| t[i].s.as_str()).collect();
        let ptr = inner.last().map_or(false, |s| *s == "*");
        let base: Vec<&str> = inner.iter().copied().filter(|s| *s != "const" && *s != "volatile" && *s != "*").collect();
        let byte_ptr = ptr && inner.iter().filter(|s| **s == "*").count() == 1 && matches!(base.as_slice(), ["char"] | ["unsigned", "char"] | ["signed", "char"] | ["u8"] | ["s8"]);
        out.push(Cast { open: p, close: c, operand_end: operand_end(t, sg, c + 1), ptr, byte_ptr });
    }
    out
}

pub const N_FIELDS: usize = 14;

/// Counts of decompiler-style constructs in one draft.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Naturalness {
    /// pointer arithmetic with a literal byte offset, `*(T*)this` reinterpretations
    pub raw_offsets: u32,
    /// C-style casts (other than the raw-offset ones)
    pub casts: u32,
    /// `static_cast` / `reinterpret_cast` / `const_cast`
    pub named_casts: u32,
    /// locals named after registers or stack slots (`temp_r3`, `var_r31`, `local_14`, `stack_50`)
    pub reg_locals: u32,
    /// locals initialized and then read exactly once
    pub single_use: u32,
    pub gotos: u32,
    /// `extern "C"` definitions of placeholder-named functions
    pub extern_c: u32,
    /// unnamed parameters (`arg0`)
    pub unnamed_params: u32,
    /// explicit `this->`
    pub this_arrows: u32,
    /// `!(a < b)` style negated comparisons
    pub neg_compares: u32,
    /// synthesized stand-in types (`__mwdec_*`)
    pub synth_types: u32,
    /// locals declared bare and assigned by their first use (`int x; ... x = f();`)
    pub split_decls: u32,
    /// `x = x + e;` instead of `x += e;` / `x++;`
    pub self_ops: u32,
    /// `*&x`
    pub deref_addr: u32,
}

impl Naturalness {
    pub const FIELDS: [&'static str; N_FIELDS] = ["raw", "cast", "ncast", "reg", "once", "goto", "externc", "argN", "this", "negcmp", "synth", "split", "selfop", "*&"];

    pub fn values(&self) -> [u32; N_FIELDS] {
        [
            self.raw_offsets,
            self.casts,
            self.named_casts,
            self.reg_locals,
            self.single_use,
            self.gotos,
            self.extern_c,
            self.unnamed_params,
            self.this_arrows,
            self.neg_compares,
            self.synth_types,
            self.split_decls,
            self.self_ops,
            self.deref_addr,
        ]
    }

    /// Weighted total: 0 = nothing a reviewer would flag by these counts.
    pub fn penalty(&self) -> u32 {
        const W: [u32; N_FIELDS] = [3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 3, 1, 1, 1];
        self.values().iter().zip(W).map(|(v, w)| v * w).sum()
    }

    pub fn add(&mut self, o: &Naturalness) {
        let v = o.values();
        let s = self.values();
        let mut r = [0u32; N_FIELDS];
        for i in 0..N_FIELDS {
            r[i] = s[i] + v[i];
        }
        *self = Naturalness::from_values(r);
    }

    pub fn from_values(v: [u32; N_FIELDS]) -> Self {
        Naturalness {
            raw_offsets: v[0],
            casts: v[1],
            named_casts: v[2],
            reg_locals: v[3],
            single_use: v[4],
            gotos: v[5],
            extern_c: v[6],
            unnamed_params: v[7],
            this_arrows: v[8],
            neg_compares: v[9],
            synth_types: v[10],
            split_decls: v[11],
            self_ops: v[12],
            deref_addr: v[13],
        }
    }

    /// `raw=1 cast=3 ...` with the non-zero counts only, then the weighted total.
    pub fn summary(&self) -> String {
        let mut s: Vec<String> = Self::FIELDS.iter().zip(self.values()).filter(|(_, v)| *v > 0).map(|(n, v)| format!("{n}={v}")).collect();
        s.push(format!("pen={}", self.penalty()));
        s.join(" ")
    }
}

fn is_reg_name(s: &str) -> bool {
    ["temp_", "var_", "local_", "stack_", "unk_"].iter().any(|p| s.starts_with(p) && s.len() > p.len())
        || (s.starts_with("sp") && s.len() > 2 && s[2..].chars().all(|c| c.is_ascii_hexdigit()))
}

fn is_arg_name(s: &str) -> bool {
    s.len() > 3 && s.starts_with("arg") && s[3..].chars().all(|c| c.is_ascii_digit())
}

/// The function definition: significant positions of its parameter list `(`..`)` and body
/// `{`..`}` (the last top-level body in the text, after any preamble).
struct Def {
    params: (usize, usize),
    body: (usize, usize),
}

fn find_def(t: &[Tok], sg: &[usize]) -> Option<Def> {
    let mut best = None;
    let mut p = 0;
    while p < sg.len() {
        let s = t[sg[p]].s.as_str();
        if s == "{" {
            let c = close_of(t, sg, p)?;
            // `) {` or `) const {`
            let mut q = p;
            if q > 0 && t[sg[q - 1]].s == "const" {
                q -= 1;
            }
            if q > 0 && t[sg[q - 1]].s == ")" {
                // matching open paren
                let mut d = 0;
                let mut o = q - 1;
                loop {
                    match t[sg[o]].s.as_str() {
                        ")" => d += 1,
                        "(" => {
                            d -= 1;
                            if d == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    if o == 0 {
                        break;
                    }
                    o -= 1;
                }
                best = Some(Def { params: (o, q - 1), body: (p, c) });
            }
            p = c + 1;
            continue;
        }
        p += 1;
    }
    best
}

/// Parameter names (in order) of the definition.
fn param_names(t: &[Tok], sg: &[usize], d: &Def) -> Vec<(usize, String)> {
    let (a, b) = d.params;
    let mut out = vec![];
    let mut depth = 0;
    let mut last_id: Option<usize> = None;
    for p in a + 1..=b {
        let s = t[sg[p]].s.as_str();
        match s {
            "(" | "<" | "[" => depth += 1,
            ")" | ">" | "]" if p < b => depth -= 1,
            _ => {}
        }
        if (s == "," && depth == 0) || p == b {
            if let Some(l) = last_id.take() {
                out.push((l, t[sg[l]].s.clone()));
            }
            continue;
        }
        if t[sg[p]].k == K::Id && depth == 0 && !is_builtin(&t[sg[p]].s) && t[sg[p]].s != "const" {
            // the declarator name follows the type: the last identifier not followed by `::`
            if t.get(sg.get(p + 1).copied().unwrap_or(usize::MAX)).map_or(true, |n| n.s != "::") && p > a + 1 && t[sg[p - 1]].s != "::" {
                last_id = Some(p);
            } else if p > a + 1 && t[sg[p - 1]].s != "::" {
                last_id = None;
            }
        }
    }
    // a lone type (`void`, `CFoo&`) has no name: drop entries that are the first token
    out.retain(|(p, n)| *p > a + 1 && !is_builtin(n) && n.chars().next().map_or(false, |c| c.is_ascii_lowercase() || c == '_'));
    out
}

/// Locals declared in the body: name -> (significant position of the declarator, type text,
/// initializer range `[from, to)` when declared `T x = init;`).
#[derive(Clone, Debug)]
struct Local {
    pos: usize,
    ty: String,
    init: Option<(usize, usize)>,
    /// significant position where the declaration statement starts
    stmt: usize,
    /// end of the declaration statement (the `;`)
    end: usize,
}

fn locals(t: &[Tok], sg: &[usize], d: &Def) -> HashMap<String, Local> {
    let mut out = HashMap::new();
    let (a, b) = d.body;
    let mut p = a + 1;
    while p < b {
        // statement start: after `{`, `}`, `;`
        let prev = t[sg[p - 1]].s.as_str();
        if !(prev == "{" || prev == "}" || prev == ";") {
            p += 1;
            continue;
        }
        // type tokens then a name then `=`, `;`, `(`, `[`
        let mut q = p;
        while q < b && (t[sg[q]].k == K::Id || matches!(t[sg[q]].s.as_str(), "*" | "&" | "::" | "<" | ">" | ",")) {
            if t[sg[q]].s == "," && !t[sg[p..q].iter().copied().next().unwrap_or(sg[p])].s.is_empty() {
                // `,` only inside template args
                let opens = sg[p..q].iter().filter(|&&i| t[i].s == "<").count();
                let closes = sg[p..q].iter().filter(|&&i| t[i].s == ">").count();
                if opens <= closes {
                    break;
                }
            }
            q += 1;
        }
        if q > p + 1 && q < b && t[sg[q - 1]].k == K::Id && matches!(t[sg[q]].s.as_str(), "=" | ";" | "(" | "[") {
            let name = t[sg[q - 1]].s.clone();
            if !KEYWORDS.contains(&name.as_str()) && looks_like_type(t, sg, p, q - 1) && !KEYWORDS.contains(&t[sg[p]].s.as_str()) {
                let mut e = q;
                let mut depth = 0;
                while e < b {
                    match t[sg[e]].s.as_str() {
                        "(" | "[" | "{" => depth += 1,
                        ")" | "]" | "}" => depth -= 1,
                        ";" if depth == 0 => break,
                        _ => {}
                    }
                    e += 1;
                }
                let init = if t[sg[q]].s == "=" { Some((q + 1, e)) } else { None };
                let ty = sg[p..q - 1].iter().map(|&i| t[i].s.as_str()).collect::<Vec<_>>().join(" ").replace(" *", "*").replace(" &", "&").replace(" :: ", "::");
                out.entry(name).or_insert(Local { pos: q - 1, ty, init, stmt: p, end: e });
            }
        }
        p += 1;
    }
    out
}

fn ident_uses(t: &[Tok], sg: &[usize], from: usize, to: usize, name: &str) -> Vec<usize> {
    (from..to).filter(|&p| t[sg[p]].k == K::Id && t[sg[p]].s == name && (p == 0 || !matches!(t[sg[p - 1]].s.as_str(), "." | "->" | "::"))).collect()
}

/// Measure one draft (preamble + definition).
pub fn measure(code: &str) -> Naturalness {
    // (text heuristics: malformed input yields no counts rather than a panic)
    std::panic::catch_unwind(|| measure_text(code)).unwrap_or_default()
}

fn measure_text(code: &str) -> Naturalness {
    let t = lex(code);
    let sg = sig(&t);
    let mut n = Naturalness::default();
    let casts = find_casts(&t, &sg);
    let mut counted_nums: HashSet<usize> = HashSet::new();
    let mut raw_casts: HashSet<usize> = HashSet::new();
    for c in &casts {
        if !c.ptr {
            continue;
        }
        // offset arithmetic inside the operand, or a byte pointer the arithmetic is done on
        let mut hit = false;
        for q in c.close + 1..c.operand_end.min(sg.len()) {
            if matches!(t[sg[q]].s.as_str(), "+" | "-") && sg.get(q + 1).map_or(false, |&i| t[i].k == K::Num) && !counted_nums.contains(&(q + 1)) {
                counted_nums.insert(q + 1);
                hit = true;
            }
        }
        if c.byte_ptr && sg.get(c.operand_end).map_or(false, |&i| matches!(t[i].s.as_str(), "+" | "-")) && sg.get(c.operand_end + 1).map_or(false, |&i| t[i].k == K::Num) {
            if counted_nums.insert(c.operand_end + 1) {
                hit = true;
            }
        }
        // `*(T*)this`
        if c.open > 0 && t[sg[c.open - 1]].s == "*" && t[sg[c.close + 1]].s == "this" {
            hit = true;
        }
        if hit {
            n.raw_offsets += 1;
        }
        if hit || c.byte_ptr {
            raw_casts.insert(c.open);
        }
    }
    n.casts = casts.iter().filter(|c| !raw_casts.contains(&c.open)).count() as u32;
    let mut reg: HashSet<&str> = HashSet::new();
    for (p, &i) in sg.iter().enumerate() {
        let s = t[i].s.as_str();
        match t[i].k {
            K::Id => match s {
                "static_cast" | "reinterpret_cast" | "const_cast" => n.named_casts += 1,
                "goto" => n.gotos += 1,
                "this" if sg.get(p + 1).map_or(false, |&j| t[j].s == "->") => n.this_arrows += 1,
                _ if s.starts_with("__mwdec") => n.synth_types += 1,
                _ if is_reg_name(s) && (p == 0 || !matches!(t[sg[p - 1]].s.as_str(), "." | "->" | "::")) => {
                    reg.insert(s);
                }
                _ => {}
            },
            K::Punct if s == "*" && sg.get(p + 1).map_or(false, |&j| t[j].s == "&") && p > 0 && !matches!(t[sg[p - 1]].k, K::Id | K::Num) && !matches!(t[sg[p - 1]].s.as_str(), ")" | "]") => {
                n.deref_addr += 1;
            }
            K::Punct if s == "!" && sg.get(p + 1).map_or(false, |&j| t[j].s == "(") => {
                if let Some(c) = close_of(&t, &sg, p + 1) {
                    if top_level_compare(&t, &sg, p + 2, c).is_some() {
                        n.neg_compares += 1;
                    }
                }
            }
            _ => {}
        }
    }
    n.reg_locals = reg.len() as u32;
    if let Some(d) = find_def(&t, &sg) {
        let params = param_names(&t, &sg, &d);
        n.unnamed_params = params.iter().filter(|(_, s)| is_arg_name(s)).count() as u32;
        // `extern "C" ... fn_X(` in the definition's declarator
        let decl_start = (0..d.params.0).rev().find(|&p| matches!(t[sg[p]].s.as_str(), ";" | "}")).map_or(0, |p| p + 1);
        let header: Vec<&str> = sg[decl_start..d.params.0].iter().map(|&i| t[i].s.as_str()).collect();
        if header.contains(&"extern") && header.last().map_or(false, |s| s.starts_with("fn_")) {
            n.extern_c += 1;
        }
        for (name, l) in locals(&t, &sg, &d) {
            let uses = ident_uses(&t, &sg, l.end, d.body.1, &name);
            if l.init.is_some() && uses.len() == 1 {
                n.single_use += 1;
            }
            if l.init.is_none() && t[sg[l.pos + 1]].s == ";" && uses.first().map_or(false, |&u| is_plain_assign(&t, &sg, u)) {
                n.split_decls += 1;
            }
        }
        n.self_ops = self_op_sites(&t, &sg, &d).len() as u32;
    }
    n
}

/// `x = ...;` with `x` at a statement start (or a `for (` init).
fn is_plain_assign(t: &[Tok], sg: &[usize], u: usize) -> bool {
    u > 0 && sg.get(u + 1).map_or(false, |&i| t[i].s == "=") && (matches!(t[sg[u - 1]].s.as_str(), ";" | "{" | "}") || (t[sg[u - 1]].s == "(" && u > 1 && t[sg[u - 2]].s == "for"))
}

/// `x = x op e;` statements: (position of the first `x`, operator position, end `;`).
fn self_op_sites(t: &[Tok], sg: &[usize], d: &Def) -> Vec<(usize, usize, usize)> {
    let mut out = vec![];
    for p in d.body.0 + 1..d.body.1.saturating_sub(4) {
        if t[sg[p]].k != K::Id || !matches!(t[sg[p - 1]].s.as_str(), ";" | "{" | "}") || t[sg[p + 1]].s != "=" || t[sg[p + 2]].s != t[sg[p]].s {
            continue;
        }
        let op = t[sg[p + 3]].s.as_str();
        if !matches!(op, "+" | "-" | "*" | "/" | "|" | "&" | "^" | "<<" | ">>" | "%") {
            continue;
        }
        // the rest up to `;` must be one operand of the same precedence level or tighter
        let mut e = p + 4;
        let mut depth = 0;
        let mut ok = true;
        while e < d.body.1 {
            let s = t[sg[e]].s.as_str();
            match s {
                "(" | "[" => depth += 1,
                ")" | "]" if depth == 0 => break,
                ")" | "]" => depth -= 1,
                ";" if depth == 0 => break,
                "+" | "-" | "*" | "/" | "|" | "&" | "^" | "<<" | ">>" | "%" | "?" | "&&" | "||" | "<" | ">" | "==" | "!=" | "<=" | ">=" if depth == 0 => {
                    // a unary minus right after the operator is fine
                    if !(e == p + 4 && s == "-") {
                        ok = false;
                    }
                }
                _ => {}
            }
            e += 1;
        }
        if ok && e > p + 4 {
            out.push((p, p + 3, e));
        }
    }
    out
}

/// Position of a comparison operator at the top level of `[a, b)`, if it's the only binary
/// operator there (`x < y`, not `a + b < c && d`).
fn top_level_compare(t: &[Tok], sg: &[usize], a: usize, b: usize) -> Option<usize> {
    let mut depth = 0;
    let mut found = None;
    for p in a..b {
        let s = t[sg[p]].s.as_str();
        match s {
            "(" | "[" => depth += 1,
            ")" | "]" => depth -= 1,
            "<" | ">" | "<=" | ">=" | "==" | "!=" if depth == 0 => {
                if found.is_some() {
                    return None;
                }
                found = Some(p);
            }
            "&&" | "||" | "?" | "=" if depth == 0 => return None,
            _ => {}
        }
    }
    found
}

// ---------------------------------------------------------------------------------------------
// Polish

/// Rewrites that keep the compiler's view of the function identical: `this->` removal (when no
/// local or parameter shadows the member) and register-named locals / `argN` parameters renamed
/// after what they hold.
pub fn tidy_names(code: &str) -> String {
    tidy_names_db(code, None)
}

/// `tidy_names` with a type context: locals passed as arguments are named after the callee's
/// parameters.
pub fn tidy_names_db(code: &str, db: Option<&TypeDb>) -> String {
    let t = drop_parens(&drop_this(code));
    rename_locals_db(&t, db)
}

/// The declared name of the parameter that the argument at significant position `u` is passed
/// to (`c.Get(r, g, b)` -> `r`, `g`, `b` from `void Get(float& r, float& g, float& b)`), when
/// every declaration of that name agrees.
fn name_from_call_arg(t: &[Tok], sg: &[usize], u: usize, db: &TypeDb, d: &Def, decls: &HashMap<String, Local>) -> Option<String> {
    // the argument must be the whole argument (`f(x)`, `f(&x)`)
    let mut a = u;
    if a > 0 && t[sg[a - 1]].s == "&" {
        a -= 1;
    }
    let next = sg.get(u + 1).map(|&i| t[i].s.as_str())?;
    if !matches!(next, "," | ")") {
        return None;
    }
    // walk back to the call's `(`, counting arguments
    let mut depth = 0;
    let mut idx = 0;
    let mut q = a;
    loop {
        if q == 0 {
            return None;
        }
        q -= 1;
        match t[sg[q]].s.as_str() {
            ")" | "]" => depth += 1,
            "(" | "[" if depth > 0 => depth -= 1,
            "(" => break,
            "," if depth == 0 => idx += 1,
            ";" | "{" | "}" => return None,
            _ => {}
        }
    }
    if q == 0 || t[sg[q - 1]].k != K::Id {
        return None;
    }
    let callee = t[sg[q - 1]].s.as_str();
    // the callee's class: of the object (`obj.f(` / `obj->f(`), else the function's own class
    let own = {
        let decl_start = (0..d.params.0).rev().find(|&p| matches!(t[sg[p]].s.as_str(), ";" | "}")).map_or(0, |p| p + 1);
        let q: Vec<&str> = sg[decl_start..d.params.0].iter().map(|&i| t[i].s.as_str()).collect();
        q.iter().rposition(|s| *s == "::").map(|i| {
            let mut a = i;
            while a >= 2 && q[a - 2] == "::" {
                a -= 2;
            }
            q[a.saturating_sub(1)..i].concat()
        })
    };
    let field_type = |cls: &str, field: &str| -> Option<String> {
        let mut c = mwdec_lift::sig::find_class(db, cls)?;
        for _ in 0..8 {
            if let Some(f) = c.fields.iter().find(|f| f.name == field) {
                return mwdec_lift::ir::named(mwdec_lift::ir::strip_cv(&f.ty)).map(|s| s.to_string());
            }
            c = mwdec_lift::sig::find_class(db, &c.bases.first()?.name)?;
        }
        None
    };
    let class: Option<String> = if q >= 3 && matches!(t[sg[q - 2]].s.as_str(), "." | "->") && t[sg[q - 3]].k == K::Id {
        let obj = t[sg[q - 3]].s.as_str();
        let before = if q >= 4 { t[sg[q - 4]].s.as_str() } else { "" };
        if matches!(before, "." | "->" | "::") {
            None
        } else if let Some(l) = decls.get(obj) {
            Some(l.ty.replace("const ", "").trim_end_matches(['*', '&']).trim().to_string())
        } else {
            own.as_deref().and_then(|o| field_type(o, obj))
        }
    } else if q >= 2 && matches!(t[sg[q - 2]].s.as_str(), "." | "->") {
        return None;
    } else {
        own.clone()
    };
    let suffix = format!("::{callee}");
    let exact = class.as_ref().map(|c| format!("{c}::{callee}"));
    let have_exact = exact.as_ref().map_or(false, |e| db.decls.contains_key(e));
    let mut names: HashSet<&str> = HashSet::new();
    for (k, ds) in &db.decls {
        if have_exact {
            if Some(k) != exact.as_ref() {
                continue;
            }
        } else if k != callee && !k.ends_with(&suffix) {
            continue;
        }
        for d in ds {
            if let Some(n) = d.params.get(idx).and_then(|p| p.name.as_deref()) {
                names.insert(n);
            }
        }
    }
    let n = (names.len() == 1).then(|| names.into_iter().next())??;
    if n.is_empty() || is_arg_name(n) || is_reg_name(n) || !n.starts_with(|c: char| c.is_ascii_alphabetic()) || KEYWORDS.contains(&n) {
        return None;
    }
    Some(n.to_string())
}

/// `(mItems)[i]` / `(p->mFoo).x` -> `mItems[i]` / `p->mFoo.x`: parentheses around a plain
/// postfix chain before a postfix operator.
fn drop_parens(code: &str) -> String {
    let t = lex(code);
    let sg = sig(&t);
    let mut drop: HashSet<usize> = HashSet::new();
    for p in 0..sg.len() {
        if t[sg[p]].s != "(" {
            continue;
        }
        if p > 0 {
            let prev = &t[sg[p - 1]];
            if prev.k == K::Id || prev.k == K::Num || matches!(prev.s.as_str(), ")" | "]" | ">") {
                continue;
            }
        }
        let Some(c) = close_of(&t, &sg, p) else { continue };
        let next = sg.get(c + 1).map_or("", |&i| t[i].s.as_str());
        let prev = if p > 0 { t[sg[p - 1]].s.as_str() } else { "" };
        // a whole statement `(a = b);` or a whole argument `f((a / b))`
        let whole = (matches!(prev, ";" | "{" | "}") && next == ";") || (matches!(prev, "(" | ",") && matches!(next, ")" | ",") && p > 1 && t[sg[p - 2]].k == K::Id && prev == "(" || prev == "," && matches!(next, ")" | ","));
        if whole && !looks_like_type(&t, &sg, p + 1, c) && c > p + 1 {
            drop.insert(sg[p]);
            drop.insert(sg[c]);
            continue;
        }
        if !matches!(next, "[" | "." | "->") {
            continue;
        }
        if t[sg[p + 1]].k != K::Id || looks_like_type(&t, &sg, p + 1, c) || operand_end(&t, &sg, p + 1) != c {
            continue;
        }
        drop.insert(sg[p]);
        drop.insert(sg[c]);
    }
    if drop.is_empty() {
        return code.to_string();
    }
    t.iter().enumerate().filter(|(i, _)| !drop.contains(i)).map(|(_, t)| t.s.as_str()).collect()
}

fn drop_this(code: &str) -> String {
    let t = lex(code);
    let sg = sig(&t);
    let Some(d) = find_def(&t, &sg) else { return code.to_string() };
    let mut shadow: HashSet<String> = param_names(&t, &sg, &d).into_iter().map(|(_, n)| n).collect();
    shadow.extend(locals(&t, &sg, &d).into_keys());
    // also any identifier declared in a nested scope we didn't parse: be conservative and treat
    // every identifier that appears right after a type-looking token as declared
    let mut drop: HashSet<usize> = HashSet::new();
    for p in d.body.0..d.body.1 {
        if t[sg[p]].s == "this" && sg.get(p + 2).is_some() && t[sg[p + 1]].s == "->" && t[sg[p + 2]].k == K::Id {
            let m = &t[sg[p + 2]].s;
            if shadow.contains(m) || m == "operator" || (p > 0 && matches!(t[sg[p - 1]].s.as_str(), "." | "->" | "::")) {
                continue;
            }
            drop.insert(sg[p]);
            drop.insert(sg[p + 1]);
        }
    }
    if drop.is_empty() {
        return code.to_string();
    }
    t.iter().enumerate().filter(|(i, _)| !drop.contains(i)).map(|(_, t)| t.s.as_str()).collect()
}

/// Name for a value of type `ty` (type text as declared).
fn name_from_type(ty: &str) -> Option<String> {
    let base: String = ty.replace("const ", "").replace(['*', '&'], "").trim().to_string();
    let last = base.rsplit("::").next().unwrap_or(&base).trim();
    let known: &[(&str, &str)] = &[
        ("CStateManager", "mgr"),
        ("CTransform4f", "xf"),
        ("CVector3f", "vec"),
        ("CVector2f", "vec"),
        ("CQuaternion", "quat"),
        ("CMatrix3f", "mtx"),
        ("CAABox", "box"),
        ("CColor", "color"),
        ("CInputStream", "in"),
        ("COutputStream", "out"),
        ("CRelAngle", "angle"),
        ("CPlayer", "player"),
        ("CPlayerState", "playerState"),
        ("CActor", "actor"),
        ("CEntity", "ent"),
        ("CScriptMsg", "msg"),
        ("TUniqueId", "uid"),
        ("CFinalInput", "input"),
        ("CMaterialList", "materials"),
        ("CMaterialFilter", "filter"),
        ("CDamageInfo", "info"),
        ("CAnimData", "animData"),
        ("CModelData", "modelData"),
        ("CBodyController", "bodyCtrl"),
        ("CGameArea", "area"),
        ("CWorld", "world"),
        ("SObjectTag", "tag"),
        ("CVParamTransfer", "xfer"),
        ("CArchitectureMessage", "msg"),
        ("CArchitectureQueue", "queue"),
        ("CFrustumPlanes", "frustum"),
        ("CCollisionInfoList", "list"),
        ("CRayCastResult", "result"),
        ("CMRay", "ray"),
        ("CPlane", "plane"),
        ("CSphere", "sphere"),
        ("CTimeRef", "time"),
        ("CRandom16", "rand"),
    ];
    if let Some((_, n)) = known.iter().find(|(k, _)| *k == last) {
        return Some(n.to_string());
    }
    match last {
        "float" | "f32" | "double" | "f64" => return None,
        "bool" => return Some("flag".into()),
        _ => {}
    }
    if is_builtin(last.split_whitespace().last().unwrap_or("")) {
        return None;
    }
    // CFooBar -> fooBar, EThing -> thing, SData -> data, TToken<..> -> skip
    if last.contains('<') || last.is_empty() {
        return None;
    }
    Some(class_var_name(last))
}

/// `DVDFileInfo` -> `dvdFileInfo`, `CRC` -> `crc`, `Foo` -> `foo`.
fn lower_camel(s: &str) -> String {
    let c: Vec<char> = s.chars().collect();
    let run = c.iter().take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit()).count();
    if run == 0 {
        return s.to_string();
    }
    // keep the last capital of a run when a lowercase letter follows (`DVDFile` -> `dvdFile`)
    let k = if run > 1 && run < c.len() && c[run].is_ascii_lowercase() { run - 1 } else { run };
    c[..k].iter().map(|c| c.to_ascii_lowercase()).chain(c[k..].iter().copied()).collect()
}

/// A class/enum name as a variable name: `CTexture` -> `texture`, `EState` -> `state`.
fn class_var_name(s: &str) -> String {
    let b = s.as_bytes();
    let stripped = if b.len() > 2 && matches!(b[0], b'C' | b'E' | b'S' | b'I' | b'T') && b[1].is_ascii_uppercase() && (b[2].is_ascii_lowercase() || b.len() > 3 && b[3].is_ascii_lowercase()) { &s[1..] } else { s };
    lower_camel(stripped)
}

/// Accessors whose name says nothing about the value (`mAnimData.get()` is named after
/// `mAnimData`).
const GENERIC_CALLS: &[&str] = &["get", "GetPtr", "GetObj", "data", "front", "back", "begin", "end", "c_str", "operator", "first", "second", "GetData", "Get"];

/// Name for a value initialized from `init` (getter/member/call name).
fn name_from_init(t: &[Tok], sg: &[usize], a: usize, b: usize) -> Option<String> {
    if a >= b {
        return None;
    }
    // `new (...) CFoo(...)`: the class
    if t[sg[a]].s == "new" {
        let mut p = a + 1;
        if t[sg[p]].s == "(" {
            p = close_of(t, sg, p)? + 1;
        }
        let mut last = None;
        while p < b && (t[sg[p]].k == K::Id || t[sg[p]].s == "::") {
            if t[sg[p]].k == K::Id {
                last = Some(t[sg[p]].s.clone());
            }
            p += 1;
        }
        return last.map(|n| class_var_name(&n));
    }
    // the names of a postfix chain at the top level: `mgr.GetPlayer()` -> player,
    // `this->mFoo` -> foo, `x->GetBar(1)` -> bar
    let mut chain: Vec<(String, bool)> = vec![];
    let mut depth = 0;
    for p in a..b {
        let s = t[sg[p]].s.as_str();
        match s {
            "(" | "[" => depth += 1,
            ")" | "]" => depth -= 1,
            _ => {}
        }
        if depth == 0 && t[sg[p]].k == K::Id && !KEYWORDS.contains(&s) {
            let call = sg.get(p + 1).map_or(false, |&i| t[i].s == "(");
            chain.push((s.to_string(), call));
        }
        if depth == 0 && matches!(s, "+" | "-" | "*" | "/" | "<" | ">" | "==" | "!=" | "&&" | "||" | "?" | "&" | "|" | "^" | "%" | "<<" | ">>") && p > a {
            return None;
        }
    }
    while chain.len() > 1 && GENERIC_CALLS.contains(&chain.last()?.0.as_str()) {
        chain.pop();
    }
    let (n, call) = chain.pop()?;
    // offset-named placeholders (`x4_`, `unk8`) say nothing either
    let offset_named = (n.starts_with('x') || n.starts_with("unk")) && n.trim_start_matches("unk").trim_start_matches('x').trim_end_matches('_').chars().all(|c| c.is_ascii_hexdigit()) && n.len() > 1;
    // placeholder symbols (`lbl_8041D118`, `fn_80012345`) neither
    let placeholder = ["lbl_", "fn_", "data_", "gap_", "jumptable_", "func_"].iter().any(|p| n.starts_with(p));
    if is_reg_name(&n) || is_arg_name(&n) || is_builtin(&n) || GENERIC_CALLS.contains(&n.as_str()) || offset_named || placeholder {
        return None;
    }
    // SDK idiom
    if n == "OSDisableInterrupts" {
        return Some("enabled".into());
    }
    let n = if call {
        // getters name their value; other calls (actions) don't
        if let Some(r) = n.strip_prefix("Get").filter(|r| r.starts_with(|c: char| c.is_ascii_uppercase())) {
            r.to_string()
        } else if (n.starts_with("Is") || n.starts_with("Has")) && n.len() > 3 {
            n.clone()
        } else if let Some(r) = ["Find", "Calc", "Compute", "Make", "Build", "Create", "Load", "Read", "Alloc", "Query", "Fetch"]
            .iter()
            .find_map(|v| n.strip_prefix(v))
            .filter(|r| r.starts_with(|c: char| c.is_ascii_uppercase()))
        {
            r.to_string()
        } else if n.starts_with(|c: char| c.is_ascii_uppercase()) && chain.is_empty() && !n.contains('_') {
            // a constructor-style temporary `CFoo(...)`
            if n.starts_with('C') && n.len() > 1 && n[1..].starts_with(|c: char| c.is_ascii_uppercase()) {
                return Some(class_var_name(&n));
            }
            return None;
        } else {
            return None;
        }
    } else if n.len() > 1 && (n.starts_with('m') || n.starts_with('s')) && n[1..].starts_with(|c: char| c.is_ascii_uppercase()) {
        n[1..].to_string()
    } else {
        n
    };
    if !n.starts_with(|c: char| c.is_ascii_alphabetic()) {
        return None;
    }
    Some(lower_camel(&n))
}

/// An integer local tested in a loop condition and stepped by a constant: a loop counter.
fn is_counter(t: &[Tok], sg: &[usize], d: &Def, name: &str, l: &Local) -> bool {
    if l.ty.contains('*') || !is_builtin(l.ty.split_whitespace().last().unwrap_or("")) || l.ty.contains("float") || l.ty.contains("double") || l.ty.contains("bool") {
        return false;
    }
    let uses = ident_uses(t, sg, l.end, d.body.1, name);
    let in_cond = uses.iter().any(|&u| {
        // inside the parentheses of a `while (` / `for (`
        let mut depth = 0;
        let mut q = u;
        while q > d.body.0 {
            q -= 1;
            match t[sg[q]].s.as_str() {
                ")" => depth += 1,
                "(" if depth == 0 => return q > 0 && matches!(t[sg[q - 1]].s.as_str(), "while" | "for"),
                "(" => depth -= 1,
                ";" | "{" | "}" if depth == 0 && t[sg[q]].s != ";" => return false,
                _ => {}
            }
        }
        false
    });
    let stepped = uses.iter().any(|&u| {
        let next = sg.get(u + 1).map(|&i| t[i].s.as_str()).unwrap_or("");
        let prev = t[sg[u - 1]].s.as_str();
        matches!(next, "++" | "--" | "+=" | "-=") || matches!(prev, "++" | "--") || (next == "=" && sg.get(u + 2).map_or(false, |&i| t[i].s == name) && sg.get(u + 4).map_or(false, |&i| t[i].k == K::Num))
    });
    in_cond && stepped
}

fn rename_locals_db(code: &str, db: Option<&TypeDb>) -> String {
    let t = lex(code);
    let sg = sig(&t);
    let Some(d) = find_def(&t, &sg) else { return code.to_string() };
    let params = param_names(&t, &sg, &d);
    let locs = locals(&t, &sg, &d);
    // every identifier in use: new names must not collide with any of them
    let mut taken: HashSet<String> = t.iter().filter(|x| x.k == K::Id).map(|x| x.s.clone()).collect();
    taken.extend(KEYWORDS.iter().map(|s| s.to_string()));
    let mut map: HashMap<String, String> = HashMap::new();
    let pick = |base: String, taken: &mut HashSet<String>| -> String {
        let base = if KEYWORDS.contains(&base.as_str()) || is_builtin(&base) { format!("{base}_") } else { base };
        let mut n = base.clone();
        let mut k = 2;
        while taken.contains(&n) {
            n = format!("{base}{k}");
            k += 1;
        }
        taken.insert(n.clone());
        n
    };
    // parameters: from their type
    let (pa, pb) = d.params;
    for (pos, name) in &params {
        if !is_arg_name(name) {
            continue;
        }
        // type text: tokens from the previous `,`/`(` to the name
        let mut q = *pos;
        while q > pa + 1 && !matches!(t[sg[q - 1]].s.as_str(), "," | "(") {
            q -= 1;
        }
        let ty = sg[q..*pos].iter().map(|&i| t[i].s.as_str()).collect::<Vec<_>>().join(" ");
        // an integer used as an object address (`*(T*)(arg0 + 0x10)`, `(T*)arg0`)
        let as_address = !ty.contains('*')
            && ident_uses(&t, &sg, d.body.0, d.body.1, name).iter().any(|&u| {
                let prev = t[sg[u - 1]].s.as_str();
                let next = sg.get(u + 1).map_or("", |&i| t[i].s.as_str());
                (prev == ")" && u > 1 && t[sg[u - 2]].s == "*") || (prev == "(" && u > 2 && t[sg[u - 2]].s == ")" && t[sg[u - 3]].s == "*" && (next == "+" || next == "-"))
            });
        let base = if as_address {
            "obj".to_string()
        } else {
            name_from_type(&ty).unwrap_or_else(|| if ty.contains('*') { "ptr".into() } else if ty.contains("float") { "f".into() } else { "val".into() })
        };
        let _ = pb;
        map.insert(name.clone(), pick(base, &mut taken));
    }
    // locals in declaration order
    let mut ls: Vec<(&String, &Local)> = locs.iter().filter(|(n, _)| is_reg_name(n)).collect();
    ls.sort_by_key(|(_, l)| l.pos);
    for (name, l) in ls {
        // loop counters
        let in_for = (l.stmt > 1 && t[sg[l.stmt - 1]].s == "(" && t[sg[l.stmt - 2]].s == "for") || is_counter(&t, &sg, &d, name, l);
        let from_init = l.init.and_then(|(a, b)| name_from_init(&t, &sg, a, b));
        // a local assigned later: name after its first assignment
        let from_assign = || {
            let uses = ident_uses(&t, &sg, l.end, d.body.1, name);
            uses.iter().find_map(|&u| {
                if sg.get(u + 1).map_or(false, |&i| t[i].s == "=") {
                    let mut e = u + 2;
                    let mut depth = 0;
                    while e < d.body.1 {
                        match t[sg[e]].s.as_str() {
                            "(" | "[" => depth += 1,
                            ")" | "]" => depth -= 1,
                            ";" if depth == 0 => break,
                            _ => {}
                        }
                        if depth < 0 {
                            break;
                        }
                        e += 1;
                    }
                    name_from_init(&t, &sg, u + 2, e)
                } else {
                    None
                }
            })
        };
        let base = if in_for && !l.ty.contains('*') {
            "i".to_string()
        } else {
            from_init
                .or_else(from_assign)
                .or_else(|| {
                    let db = db?;
                    let first = *ident_uses(&t, &sg, l.end, d.body.1, name).first()?;
                    name_from_call_arg(&t, &sg, first, db, &d, &locs)
                })
                // where the value goes: a member (`mFoo = x;`), a parameter, the return value
                .or_else(|| {
                    let uses = ident_uses(&t, &sg, l.end, d.body.1, name);
                    let member = uses.iter().find_map(|&u| {
                        let (prev, next) = (t[sg[u - 1]].s.as_str(), sg.get(u + 1).map_or("", |&i| t[i].s.as_str()));
                        if prev != "=" || next != ";" || u < 2 {
                            return None;
                        }
                        let m = t[sg[u - 2]].s.as_str();
                        (t[sg[u - 2]].k == K::Id && m.len() > 1 && m.starts_with('m') && m[1..].starts_with(|c: char| c.is_ascii_uppercase())).then(|| lower_camel(&m[1..]))
                    });
                    member
                        .or_else(|| db.and_then(|db| uses.iter().skip(1).find_map(|&u| name_from_call_arg(&t, &sg, u, db, &d, &locs))))
                        .or_else(|| uses.iter().any(|&u| t[sg[u - 1]].s == "return" && sg.get(u + 1).map_or(false, |&i| t[i].s == ";")).then(|| "result".to_string()))
                })
                .or_else(|| name_from_type(&l.ty))
                .unwrap_or_else(|| if l.ty.contains('*') { "ptr".into() } else if l.ty.contains("float") || l.ty.contains("double") { "f".into() } else { "val".into() })
        };
        map.insert(name.clone(), pick(base, &mut taken));
    }
    if map.is_empty() {
        return code.to_string();
    }
    t.iter()
        .enumerate()
        .map(|(i, x)| {
            if x.k == K::Id {
                let prev = (0..i).rev().find(|&j| t[j].k != K::Ws).map(|j| t[j].s.as_str());
                if !matches!(prev, Some("." | "->" | "::")) {
                    if let Some(n) = map.get(&x.s) {
                        return n.clone();
                    }
                }
            }
            x.s.clone()
        })
        .collect()
}

/// Integer constants cast to an enum the context declares: the enumerator (`(CFoo::EState)1` ->
/// `CFoo::kS_Moving`). Values with no enumerator (flag combinations) keep their cast.
pub fn enum_names(code: &str, db: &TypeDb) -> String {
    let t = lex(code);
    let sg = sig(&t);
    let mut nt = t.clone();
    let mut changed = false;
    for c in find_casts(&t, &sg) {
        if c.ptr {
            continue;
        }
        let ty: String = sg[c.open + 1..c.close].iter().map(|&i| t[i].s.as_str()).collect();
        let Some(e) = db.enums.get(&ty) else { continue };
        // the operand: a literal, optionally negated
        let (neg, lit) = match t[sg[c.close + 1]].s.as_str() {
            "-" => (true, c.close + 2),
            _ => (false, c.close + 1),
        };
        if c.operand_end != lit + 1 || t.get(sg.get(lit).copied().unwrap_or(usize::MAX)).map_or(true, |x| x.k != K::Num) {
            continue;
        }
        let txt = t[sg[lit]].s.trim_end_matches(|c: char| matches!(c, 'u' | 'U' | 'l' | 'L'));
        let v = if let Some(h) = txt.strip_prefix("0x").or_else(|| txt.strip_prefix("0X")) { i64::from_str_radix(h, 16).ok() } else { txt.parse::<i64>().ok() };
        let Some(mut v) = v else { continue };
        if neg {
            v = -v;
        }
        let hits: Vec<&String> = e.values.iter().filter(|(_, x)| *x == v).map(|(n, _)| n).collect();
        let Some(name) = hits.first() else { continue };
        if name.is_empty() || name.starts_with('@') {
            continue;
        }
        // enumerators live in the scope enclosing the enum
        let scope = ty.rfind("::").map_or("", |i| &ty[..i + 2]);
        for q in c.open..c.operand_end {
            nt[sg[q]].s.clear();
            // whitespace inside
        }
        for i in sg[c.open]..sg[c.operand_end - 1] {
            if nt[i].k == K::Ws {
                nt[i].s.clear();
            }
        }
        nt[sg[c.open]].s = format!("{scope}{name}");
        changed = true;
    }
    // `case N:` of a switch on an enum-typed member, local or parameter
    if let Some(d) = find_def(&t, &sg) {
        let locs = locals(&t, &sg, &d);
        let params = param_names(&t, &sg, &d);
        let own = own_class(&t, &sg, &d);
        let mut p = d.body.0;
        while p + 3 < d.body.1 {
            if t[sg[p]].s != "switch" || t[sg[p + 1]].s != "(" || t[sg[p + 2]].k != K::Id || t[sg[p + 3]].s != ")" || t[sg[p + 4]].s != "{" {
                p += 1;
                continue;
            }
            let x = t[sg[p + 2]].s.as_str();
            let ty = if let Some(l) = locs.get(x) {
                Some(l.ty.replace("const ", ""))
            } else if let Some((pos, _)) = params.iter().find(|(_, n)| n == x) {
                let mut q = *pos;
                while q > d.params.0 + 1 && !matches!(t[sg[q - 1]].s.as_str(), "," | "(") {
                    q -= 1;
                }
                Some(squash_tokens(&t, &sg, q, *pos).replace("const", ""))
            } else {
                own.as_deref().and_then(|o| member_type(db, o, x))
            };
            let Some(e) = ty.as_deref().and_then(|ty| db.enums.get(ty.trim()).map(|e| (ty.trim().to_string(), e))) else {
                p += 1;
                continue;
            };
            let Some(end) = close_of(&t, &sg, p + 4) else { break };
            let scope = e.0.rfind("::").map_or("", |i| &e.0[..i + 2]).to_string();
            let mut q = p + 5;
            while q < end {
                // nested switches have their own labels
                if t[sg[q]].s == "switch" {
                    if let Some(b) = (q..end).find(|&k| t[sg[k]].s == "{") {
                        q = close_of(&t, &sg, b).unwrap_or(end) + 1;
                        continue;
                    }
                }
                if t[sg[q]].s == "case" && t[sg[q + 1]].k == K::Num && t[sg[q + 2]].s == ":" {
                    if let Ok(v) = t[sg[q + 1]].s.parse::<i64>() {
                        if let Some((n, _)) = e.1.values.iter().find(|(n, x)| *x == v && !n.is_empty() && !n.starts_with('@')) {
                            nt[sg[q + 1]].s = format!("{scope}{n}");
                            changed = true;
                        }
                    }
                }
                q += 1;
            }
            p = end;
        }
    }
    if changed {
        join(&nt)
    } else {
        code.to_string()
    }
}

/// Class of the function's definition (`void CFoo::Bar()` -> `CFoo`).
fn own_class(t: &[Tok], sg: &[usize], d: &Def) -> Option<String> {
    let decl_start = (0..d.params.0).rev().find(|&p| matches!(t[sg[p]].s.as_str(), ";" | "}")).map_or(0, |p| p + 1);
    let q: Vec<&str> = sg[decl_start..d.params.0].iter().map(|&i| t[i].s.as_str()).collect();
    q.iter().rposition(|s| *s == "::").map(|i| {
        let mut a = i;
        while a >= 2 && q[a - 2] == "::" {
            a -= 2;
        }
        q[a.saturating_sub(1)..i].concat()
    })
}

/// Declared class/enum type name of member `field` of `cls` (or its first bases).
fn member_type(db: &TypeDb, cls: &str, field: &str) -> Option<String> {
    let mut c = mwdec_lift::sig::find_class(db, cls)?;
    for _ in 0..8 {
        if let Some(f) = c.fields.iter().find(|f| f.name == field) {
            return mwdec_lift::ir::named(mwdec_lift::ir::strip_cv(&f.ty)).map(|s| s.to_string());
        }
        c = mwdec_lift::sig::find_class(db, &c.bases.first()?.name)?;
    }
    None
}

/// One-step rewrites whose effect on code generation is unknown: each candidate must be
/// checked. Ordered roughly by how much they help readability.
pub fn candidates(code: &str) -> Vec<String> {
    candidates_db(code, None)
}

/// `candidates`, with the layouts of a type context (pointer walks need element sizes).
pub fn candidates_db(code: &str, db: Option<&TypeDb>) -> Vec<String> {
    let mut out = vec![];
    let t = lex(code);
    let sg = sig(&t);
    let Some(d) = find_def(&t, &sg) else { return out };
    out.extend(ptr_walks(&t, &sg, &d, db));
    // `*&x` -> `x`
    {
        let mut nt = t.clone();
        let mut any = false;
        for p in d.body.0..d.body.1.saturating_sub(2) {
            if t[sg[p]].s == "*" && t[sg[p + 1]].s == "&" && t[sg[p + 2]].k == K::Id && !matches!(t[sg[p - 1]].k, K::Id | K::Num) && !matches!(t[sg[p - 1]].s.as_str(), ")" | "]") && operand_end(&t, &sg, p + 2) == p + 3 {
                nt[sg[p]].s.clear();
                nt[sg[p + 1]].s.clear();
                any = true;
            }
        }
        if any {
            out.push(join(&nt));
        }
    }
    // `x = x + 1;` -> `x++;`, `x = x op e;` -> `x op= e;`
    for (p, op, e) in self_op_sites(&t, &sg, &d) {
        let mut nt = t.clone();
        let o = t[sg[op]].s.clone();
        let one = e == op + 2 && t[sg[op + 1]].s == "1";
        // `x = x` + ` op ` -> `x op=`
        for i in sg[p] + 1..sg[op + 1] {
            nt[i].s.clear();
        }
        if one && (o == "+" || o == "-") {
            for i in sg[op + 1]..sg[e] {
                nt[i].s.clear();
            }
            nt[sg[p]].s = format!("{}{o}{o}", t[sg[p]].s);
        } else {
            nt[sg[p]].s = format!("{} {o}= ", t[sg[p]].s);
        }
        out.push(join(&nt));
    }
    // `T x; ... x = e;` -> `T x = e;` at the first use
    for (name, l) in locals(&t, &sg, &d) {
        if l.init.is_some() || t[sg[l.pos + 1]].s != ";" {
            continue;
        }
        let uses = ident_uses(&t, &sg, l.end, d.body.1, &name);
        let Some(&u) = uses.first() else { continue };
        if !is_plain_assign(&t, &sg, u) {
            continue;
        }
        let mut nt = t.clone();
        nt[sg[u]].s = format!("{} {}", l.ty, name);
        let from = sg[l.stmt];
        let to = sg[l.end];
        let ws_from = if from > 0 && t[from - 1].k == K::Ws { from - 1 } else { from };
        for i in ws_from..=to {
            nt[i].s.clear();
        }
        out.push(join(&nt));
    }
    // `if (!(c)) { return; } rest` at the end of a void function -> `if (c) { rest }`
    if let Some(c) = invert_guard(&t, &sg, &d) {
        out.push(c);
    }
    // a `void*` local only ever used through one pointer cast: declared with that type
    let casts = find_casts(&t, &sg);
    for (name, l) in locals(&t, &sg, &d) {
        if l.ty != "void*" {
            continue;
        }
        let uses = ident_uses(&t, &sg, l.end, d.body.1, &name);
        // the object a `new` expression stores in it decides the type
        let new_class = |a: usize| -> Option<String> {
            if t[sg[a]].s != "new" {
                return None;
            }
            let mut p = a + 1;
            if t[sg[p]].s == "(" {
                p = close_of(&t, &sg, p)? + 1;
            }
            let from = p;
            while p < d.body.1 && (t[sg[p]].k == K::Id || t[sg[p]].s == "::") {
                p += 1;
            }
            (p > from).then(|| squash_tokens(&t, &sg, from, p) + "*")
        };
        let newed = l.init.and_then(|(a, _)| new_class(a)).or_else(|| uses.iter().find_map(|&u| if sg.get(u + 1).map_or(false, |&i| t[i].s == "=") { new_class(u + 2) } else { None }));
        let mut target: Option<String> = newed.clone();
        let mut cast_pos = vec![];
        let mut ok = !uses.is_empty();
        for &u in &uses {
            if let Some(c) = casts.iter().find(|c| c.close + 1 == u && c.operand_end == u + 1) {
                let ty: String = sg[c.open + 1..c.close].iter().map(|&i| t[i].s.as_str()).collect::<Vec<_>>().join(" ").replace(" *", "*");
                if ty == "void*" {
                    cast_pos.push(c.clone());
                    continue;
                }
                // casts to other types of a new'd object stay (bases: dropped later if unneeded)
                if newed.is_some() && target.as_ref() != Some(&ty) {
                    continue;
                }
                if target.as_ref().map_or(false, |x| *x != ty) || !c.ptr || c.byte_ptr {
                    ok = false;
                    break;
                }
                target = Some(ty);
                cast_pos.push(c.clone());
            } else if !matches!(sg.get(u + 1).map(|&i| t[i].s.as_str()), Some("=")) {
                ok = false;
                break;
            }
        }
        let (true, Some(ty)) = (ok, target) else { continue };
        let mut nt = t.clone();
        // the declared type: tokens from the statement start to the name
        for q in l.stmt..l.pos {
            nt[sg[q]].s.clear();
        }
        for i in sg[l.stmt]..sg[l.pos] {
            if nt[i].k == K::Ws {
                nt[i].s.clear();
            }
        }
        nt[sg[l.stmt]].s = format!("{ty} ");
        for c in cast_pos {
            for q in c.open..=c.close {
                nt[sg[q]].s.clear();
            }
        }
        out.push(join(&nt));
    }
    // negated comparisons of integers / pointers: `!(a < b)` -> `a >= b`
    for p in d.body.0..d.body.1 {
        if t[sg[p]].s == "!" && t[sg[p + 1]].s == "(" {
            if let Some(c) = close_of(&t, &sg, p + 1) {
                if let Some(op) = top_level_compare(&t, &sg, p + 2, c) {
                    let flip = match t[sg[op]].s.as_str() {
                        "<" => ">=",
                        ">" => "<=",
                        "<=" => ">",
                        ">=" => "<",
                        "==" => "!=",
                        "!=" => "==",
                        _ => continue,
                    };
                    let mut nt = t.clone();
                    nt[sg[op]].s = flip.to_string();
                    nt[sg[p]].s.clear();
                    // keep the parentheses when the context needs them (operand of `&&` etc.):
                    // drop them only in `if (!(...))` / `return !(...);` shapes
                    let before = t[sg[p - 1]].s.as_str();
                    let after = t[sg[c + 1]].s.as_str();
                    if (before == "(" && after == ")") || (before == "return" && after == ";") || (before == "=" && after == ";") {
                        nt[sg[p + 1]].s.clear();
                        nt[sg[c]].s.clear();
                    }
                    out.push(join(&nt));
                }
            }
        }
    }
    // inline locals read once
    let locs = locals(&t, &sg, &d);
    let mut ls: Vec<(&String, &Local)> = locs.iter().collect();
    ls.sort_by_key(|(_, l)| l.pos);
    for (name, l) in ls {
        let Some((ia, ib)) = l.init else { continue };
        let uses = ident_uses(&t, &sg, l.end, d.body.1, name);
        if uses.len() != 1 {
            continue;
        }
        let u = uses[0];
        // not an assignment target / address taken / incremented
        let next = sg.get(u + 1).map(|&i| t[i].s.as_str()).unwrap_or("");
        let prev = t[sg[u - 1]].s.as_str();
        if matches!(next, "=" | "+=" | "-=" | "*=" | "/=" | "|=" | "&=" | "^=" | "<<=" | ">>=" | "++" | "--") || matches!(prev, "&" | "++" | "--") {
            continue;
        }
        // a reference or array or object-constructing declaration: skip
        if l.ty.ends_with('&') || t[sg[l.pos + 1]].s != "=" {
            continue;
        }
        let init = join(&t[sg[ia]..=sg[ib - 1]]);
        let simple = ib - ia == 1 || {
            let e = operand_end(&t, &sg, ia);
            e == ib && !matches!(t[sg[ia]].s.as_str(), "*" | "&" | "-" | "!" | "~" | "(")
        };
        // the declared type may convert: keep it as a cast when it's a scalar type
        let rep = if simple { init.clone() } else { format!("({init})") };
        let mut nt = t.clone();
        nt[sg[u]].s = rep.clone();
        // remove the declaration statement (and the whitespace before it)
        let from = sg[l.stmt];
        let to = sg[l.end];
        let ws_from = if from > 0 && t[from - 1].k == K::Ws { from - 1 } else { from };
        for i in ws_from..=to {
            nt[i].s.clear();
        }
        out.push(join(&nt));
        // and the variant that keeps the conversion of the declared type
        if !l.ty.contains('*') && is_builtin(l.ty.split_whitespace().last().unwrap_or("")) {
            let mut nt2 = nt.clone();
            nt2[sg[u]].s = format!("({}){}", l.ty, if simple { init.clone() } else { format!("({init})") });
            out.push(join(&nt2));
        }
    }
    // a scalar local always read through one conversion (`(unsigned int)x`): declared with that type
    for (name, l) in locals(&t, &sg, &d) {
        if l.ty.contains('*') || !is_builtin(l.ty.split_whitespace().last().unwrap_or("")) {
            continue;
        }
        let uses = ident_uses(&t, &sg, l.end, d.body.1, &name);
        let mut conv: Option<String> = None;
        let mut hits = vec![];
        for &u in &uses {
            if let Some(c) = casts.iter().find(|c| c.close + 1 == u && c.operand_end == u + 1 && !c.ptr) {
                let ty = sg[c.open + 1..c.close].iter().map(|&i| t[i].s.as_str()).collect::<Vec<_>>().join(" ");
                if conv.as_ref().map_or(false, |x| *x != ty) || !is_builtin(ty.split_whitespace().last().unwrap_or("")) || size_of(&ty, None) != size_of(&l.ty, None) || ty.contains("float") != l.ty.contains("float") {
                    conv = None;
                    hits.clear();
                    break;
                }
                conv = Some(ty);
                hits.push(c.clone());
            }
        }
        let Some(ty) = conv else { continue };
        if ty == l.ty {
            continue;
        }
        let mut nt = t.clone();
        for q in l.stmt..l.pos {
            nt[sg[q]].s.clear();
        }
        for i in sg[l.stmt]..sg[l.pos] {
            if nt[i].k == K::Ws {
                nt[i].s.clear();
            }
        }
        nt[sg[l.stmt]].s = format!("{ty} ");
        for c in hits {
            for i in sg[c.open]..=sg[c.close] {
                nt[i].s.clear();
            }
        }
        out.push(join(&nt));
    }
    // drop casts: every cast to one type at once, then one at a time (raw-offset casts stay)
    let all = find_casts(&t, &sg);
    let mut groups: HashMap<String, Vec<&Cast>> = HashMap::new();
    for c in all.iter().filter(|c| !c.byte_ptr) {
        groups.entry(join(&t[sg[c.open]..=sg[c.close]])).or_default().push(c);
    }
    let mut keys: Vec<&String> = groups.keys().filter(|k| groups[*k].len() > 1).collect();
    keys.sort();
    // `*(T*)&x` without the cast is `x`
    let collapse = |nt: &mut Vec<Tok>, c: &Cast| {
        if c.open > 0 && t[sg[c.open - 1]].s == "*" && t[sg[c.close + 1]].s == "&" && sg.get(c.close + 2).map_or(false, |&i| t[i].k == K::Id) && operand_end(&t, &sg, c.close + 2) == c.close + 3 {
            nt[sg[c.open - 1]].s.clear();
            nt[sg[c.close + 1]].s.clear();
        }
    };
    for k in keys {
        let mut nt = t.clone();
        for c in &groups[k] {
            for i in sg[c.open]..=sg[c.close] {
                nt[i].s.clear();
            }
            collapse(&mut nt, c);
        }
        out.push(join(&nt));
    }
    for c in all {
        if c.byte_ptr {
            continue;
        }
        let mut nt = t.clone();
        for q in c.open..=c.close {
            nt[sg[q]].s.clear();
        }
        // whitespace inside the parentheses
        for i in sg[c.open]..=sg[c.close] {
            if nt[i].k == K::Ws {
                nt[i].s.clear();
            }
        }
        let plain = join(&nt);
        collapse(&mut nt, &c);
        let collapsed = join(&nt);
        if collapsed != plain {
            out.push(collapsed);
        }
        out.push(plain);
    }
    out
}

/// The tokens `[a, b)` (significant positions) without whitespace.
fn squash_tokens(t: &[Tok], sg: &[usize], a: usize, b: usize) -> String {
    sg[a..b].iter().map(|&i| t[i].s.as_str()).collect()
}

/// Byte size of a type spelled `ty` (builtins, pointers, classes the context lays out).
fn size_of(ty: &str, db: Option<&TypeDb>) -> Option<u32> {
    let ty = ty.replace("const ", "").replace("volatile ", "");
    let ty = ty.trim();
    if ty.ends_with('*') {
        return Some(4);
    }
    let words: Vec<&str> = ty.split_whitespace().collect();
    let last = *words.last()?;
    let long_long = words.iter().filter(|w| **w == "long").count() == 2;
    Some(match last {
        _ if long_long => 8,
        "char" | "u8" | "s8" | "bool" | "uchar" => 1,
        "short" | "u16" | "s16" | "ushort" => 2,
        "int" | "long" | "u32" | "s32" | "float" | "f32" | "uint" | "unsigned" | "signed" => 4,
        "double" | "f64" | "u64" | "s64" => 8,
        _ => {
            let c = db?.classes.get(ty)?;
            if c.is_declaration || c.size == 0 {
                return None;
            }
            c.size
        }
    })
}

/// Pointer walks written as byte arithmetic: `p = (char*)p + 0x18` over elements of size 0x18
/// (with `(T*)p` uses) -> `T* p` stepped by `p + 1`.
fn ptr_walks(t: &[Tok], sg: &[usize], d: &Def, db: Option<&TypeDb>) -> Vec<String> {
    let mut out = vec![];
    let casts = find_casts(t, sg);
    let squash = |a: usize, b: usize| -> String { sg[a..b].iter().map(|&i| t[i].s.as_str()).collect() };
    for (name, l) in locals(t, sg, d) {
        if !l.ty.ends_with('*') || l.ty.ends_with("**") {
            continue;
        }
        let uses = ident_uses(t, sg, l.end, d.body.1, &name);
        // element type: the declared pointee, or the one pointer type a void* is cast to
        let mut elem: Option<String> = if l.ty == "void*" { None } else { Some(l.ty.trim_end_matches('*').trim().to_string()) };
        let mut elem_casts = vec![];
        if l.ty == "void*" {
            for &u in &uses {
                if let Some(c) = casts.iter().find(|c| c.close + 1 == u && c.operand_end == u + 1 && c.ptr && !c.byte_ptr) {
                    let ty = sg[c.open + 1..c.close].iter().map(|&i| t[i].s.as_str()).collect::<Vec<_>>().join(" ").replace(" *", "*").replace(" :: ", "::");
                    if ty == "void*" {
                        continue;
                    }
                    let e = ty.trim_end_matches('*').trim().to_string();
                    if elem.as_ref().map_or(false, |x| *x != e) {
                        elem = None;
                        elem_casts.clear();
                        break;
                    }
                    elem = Some(e);
                    elem_casts.push(c.clone());
                }
            }
        }
        let Some(elem) = elem else { continue };
        let Some(size) = size_of(&elem, db) else { continue };
        // steps: `name = <cast>(char*)name + N` / `name = (char*)name + N` up to `;` or `)`
        let mut steps = vec![];
        for &u in &uses {
            if !sg.get(u + 1).map_or(false, |&i| t[i].s == "=") {
                continue;
            }
            let mut e = u + 2;
            let mut depth = 0;
            while e < d.body.1 {
                match t[sg[e]].s.as_str() {
                    "(" | "[" => depth += 1,
                    ")" | "]" if depth == 0 => break,
                    ")" | "]" => depth -= 1,
                    ";" if depth == 0 => break,
                    _ => {}
                }
                e += 1;
            }
            let rhs = squash(u + 2, e);
            let n = rhs
                .strip_prefix(&format!("(char*){name}+"))
                .or_else(|| rhs.strip_prefix(&format!("(unsignedchar*){name}+")))
                .map(|s| s.to_string())
                .or_else(|| {
                    let inner = rhs.strip_prefix(&format!("({elem}*)((char*){name}+")).or_else(|| rhs.strip_prefix(&format!("({}*)((char*){name}+", elem.replace(' ', ""))))?;
                    inner.strip_suffix(')').map(|s| s.to_string())
                });
            let Some(n) = n else { continue };
            let v = if let Some(h) = n.strip_prefix("0x") { u32::from_str_radix(h, 16).ok() } else { n.parse().ok() };
            if v == Some(size) {
                steps.push((u + 2, e));
            }
        }
        if steps.is_empty() {
            continue;
        }
        for wrap in [false, true] {
            let mut nt = t.to_vec();
            if l.ty == "void*" {
                for q in l.stmt..l.pos {
                    nt[sg[q]].s.clear();
                }
                for i in sg[l.stmt]..sg[l.pos] {
                    if nt[i].k == K::Ws {
                        nt[i].s.clear();
                    }
                }
                nt[sg[l.stmt]].s = format!("{elem}* ");
                for c in &elem_casts {
                    for q in c.open..=c.close {
                        nt[sg[q]].s.clear();
                    }
                }
                if wrap {
                    // other assignments of byte-pointer values keep a conversion
                    let assigns: Vec<(usize, usize)> = uses
                        .iter()
                        .filter(|&&u| sg.get(u + 1).map_or(false, |&i| t[i].s == "=") && !steps.iter().any(|s| s.0 == u + 2))
                        .map(|&u| {
                            let mut e = u + 2;
                            let mut depth = 0;
                            while e < d.body.1 {
                                match t[sg[e]].s.as_str() {
                                    "(" | "[" => depth += 1,
                                    ")" | "]" if depth == 0 => break,
                                    ")" | "]" => depth -= 1,
                                    ";" if depth == 0 => break,
                                    _ => {}
                                }
                                e += 1;
                            }
                            (u + 2, e)
                        })
                        .collect();
                    let mut any = false;
                    for (a, b) in assigns {
                        if squash(a, b).contains("(char*)") {
                            nt[sg[a]].s = format!("({elem}*)({}", nt[sg[a]].s);
                            nt[sg[b - 1]].s = format!("{})", nt[sg[b - 1]].s);
                            any = true;
                        }
                    }
                    if !any {
                        continue;
                    }
                }
            } else if wrap {
                continue;
            }
            for &(a, b) in &steps {
                for q in a..b {
                    nt[sg[q]].s.clear();
                }
                for i in sg[a]..sg[b - 1] {
                    if nt[i].k == K::Ws {
                        nt[i].s.clear();
                    }
                }
                nt[sg[a]].s = format!("{name} + 1");
            }
            out.push(join(&nt));
        }
    }
    out
}

fn invert_guard(t: &[Tok], sg: &[usize], d: &Def) -> Option<String> {
    // a void function: `void` right before the declarator name
    let decl_start = (0..d.params.0).rev().find(|&p| matches!(t[sg[p]].s.as_str(), ";" | "}")).map_or(0, |p| p + 1);
    if !sg[decl_start..d.params.0].iter().any(|&i| t[i].s == "void") || sg[decl_start..d.params.0].iter().any(|&i| t[i].s == "*") {
        return None;
    }
    let (a, b) = d.body;
    let mut p = a + 1;
    while p < b {
        // only statements at the top level of the body
        if t[sg[p]].s == "{" {
            p = close_of(t, sg, p)? + 1;
            continue;
        }
        let prev = t[sg[p - 1]].s.as_str();
        if t[sg[p]].s == "if" && matches!(prev, "{" | "}" | ";") && t[sg[p + 1]].s == "(" && t[sg[p + 2]].s == "!" && t[sg[p + 3]].s == "(" {
            let cond_close = close_of(t, sg, p + 1)?;
            let inner_close = close_of(t, sg, p + 3)?;
            if inner_close + 1 == cond_close
                && cond_close + 5 < b
                && t[sg[cond_close + 1]].s == "{"
                && t[sg[cond_close + 2]].s == "return"
                && t[sg[cond_close + 3]].s == ";"
                && t[sg[cond_close + 4]].s == "}"
                && t[sg[cond_close + 5]].s != "else"
            {
                let cond = join(&t[sg[p + 4]..sg[inner_close]]);
                let rest_from = sg[cond_close + 4] + 1;
                let rest_to = sg[b];
                // the rest, without the whitespace before the body's `}`
                let mut rest = join(&t[rest_from..rest_to]);
                let trail = rest.len() - rest.trim_end().len();
                let trailing = rest[rest.len() - trail..].to_string();
                rest.truncate(rest.len() - trail);
                if rest.trim().is_empty() {
                    return None;
                }
                let before = join(&t[..sg[p]]);
                let indent: String = before.rsplit('\n').next().unwrap_or("").chars().take_while(|c| *c == ' ').collect();
                let rest = rest.replace('\n', "\n    ");
                let mut out = before;
                out.push_str(&format!("if ({cond}) {{{rest}\n{indent}}}"));
                out.push_str(&trailing);
                out.push_str(&join(&t[sg[b]..]));
                return Some(out);
            }
        }
        p += 1;
    }
    None
}

/// Polish an exact source: apply the name-only rewrites, then greedily every candidate rewrite
/// that `still_exact` accepts, up to `budget` checks. Returns the polished source (the input when
/// nothing could be applied) and the number of checks made.
pub fn polish(code: &str, db: Option<&TypeDb>, budget: usize, still_exact: &mut dyn FnMut(&str) -> bool) -> (String, usize) {
    // a rewrite that trips over unexpected text gives up (the input is exact as it is)
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| polish_text(code, db, budget, still_exact))).unwrap_or_else(|_| (code.to_string(), 0))
}

fn polish_text(code: &str, db: Option<&TypeDb>, budget: usize, still_exact: &mut dyn FnMut(&str) -> bool) -> (String, usize) {
    let mut checks = 0;
    let mut cur = code.to_string();
    let enums = |c: &str| db.map_or_else(|| c.to_string(), |db| enum_names(c, db));
    let named = enums(&tidy_names_db(&cur, db));
    if named != cur {
        checks += 1;
        if still_exact(&named) {
            cur = named;
        } else {
            // each part separately
            let rename = |c: &str| rename_locals_db(c, db);
            let parts: [&dyn Fn(&str) -> String; 3] = [&drop_this, &rename, &enums];
            for f in parts {
                let n = f(&cur);
                if n != cur && checks < budget {
                    checks += 1;
                    if still_exact(&n) {
                        cur = n;
                    }
                }
            }
        }
    }
    let mut rejected: HashSet<String> = HashSet::new();
    loop {
        let mut progressed = false;
        let base_pen = measure(&cur).penalty();
        for cand in candidates_db(&cur, db) {
            if checks >= budget {
                return (cur, checks);
            }
            if rejected.contains(&cand) || measure(&cand).penalty() >= base_pen {
                continue;
            }
            checks += 1;
            if still_exact(&cand) {
                cur = cand;
                progressed = true;
                break;
            }
            rejected.insert(cand);
        }
        if !progressed {
            // casts dropped above can leave `(p)->x`
            let fin = drop_parens(&cur);
            if fin != cur {
                checks += 1;
                if still_exact(&fin) {
                    cur = fin;
                }
            }
            return (cur, checks);
        }
    }
}


// ---------------------------------------------------------------------------------------------
// Emitter helpers

impl crate::Em<'_> {
    /// A scalar member read or written as another scalar type of the same size (`*(int*)&mX`
    /// for the bits of a float) instead of a raw byte offset from the object.
    pub(crate) fn punned_member(&mut self, base: &mwdec_lift::ir::Expr, cls: &str, path: &[mwdec_lift::types::PathElem], ft: &mwdec_core::Type, ty: &mwdec_core::Type, read: bool) -> Option<String> {
        use mwdec_core::Type;
        let fs = mwdec_lift::ir::scalar_size(mwdec_lift::ir::strip_cv(ft))?;
        let ts = mwdec_lift::ir::scalar_size(ty)?;
        if fs != ts || matches!(mwdec_lift::ir::strip_cv(ft), Type::Ref(_)) || matches!(ty, Type::Unknown { size: 0 }) || path.iter().any(|p| matches!(p, mwdec_lift::types::PathElem::Field(n, _) if n.starts_with("__"))) {
            return None;
        }
        // every member on the path nameable directly (no getters: the address is taken)
        for p in path {
            if let mwdec_lift::types::PathElem::Field(n, owner) = p {
                if !self.field_accessible(owner, n) {
                    return None;
                }
            }
        }
        if self.protected_through_other(cls, path) {
            return None;
        }
        // bitfields have no address
        let db = self.db?;
        if let Some(mwdec_lift::types::PathElem::Field(n, owner)) = path.last() {
            if mwdec_lift::sig::find_class(db, owner).and_then(|c| c.fields.iter().find(|f| f.name == *n)).map_or(true, |f| f.bitfield.is_some()) {
                return None;
            }
        }
        let b = self.expr(base, 15);
        let m = format!("{}->{}", b, self.path_str(path, read));
        let ptr = crate::type_str(&Type::Ptr(Box::new(crate::access_type(ty))));
        Some(format!("*({ptr})&{m}"))
    }

    /// A member path read only through public members and inline getters (`GetWorkerId()`), for
    /// members a direct access can't name (protected members of another object).
    pub(crate) fn getter_path_str(&self, path: &[mwdec_lift::types::PathElem]) -> Option<String> {
        use mwdec_lift::types::PathElem;
        let db = self.db?;
        let mut s = String::new();
        for p in path {
            match p {
                PathElem::Field(n, owner) => {
                    if !s.is_empty() {
                        s.push('.');
                    }
                    let public = mwdec_lift::sig::find_class(db, owner).and_then(|c| c.fields.iter().find(|f| f.name == *n)).map_or(false, |f| matches!(f.access, mwdec_core::Access::Public));
                    if public {
                        s.push_str(n);
                    } else {
                        s.push_str(&self.usable_getter(owner, n, true)?);
                        s.push_str("()");
                    }
                }
                PathElem::Index(k) => s.push_str(&format!("[{k}]")),
                PathElem::Base(_) => {}
            }
        }
        (!s.is_empty()).then_some(s)
    }
}
