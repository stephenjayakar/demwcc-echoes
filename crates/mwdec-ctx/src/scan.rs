//! Lightweight scanner over the *preprocessed* context headers.
//!
//! It recovers what MWCC's DWARF lacks or only emits on use:
//! - names of every non-template class/struct/union/enum/typedef/global declared in the
//!   context, so we can force the compiler to emit DWARF for them (see `force_tu`);
//! - member/free function declarations (return type, params, const/static/virtual/pure,
//!   inline bodies) as token lists, resolved to types later against the TypeDb;
//! - `bool` member fields (MWCC DWARF encodes bool as unsigned char).
//!
//! This is not a C++ parser: it walks declarations at namespace/class scope, skipping
//! templates, function bodies and initializers by bracket matching. Anything it cannot
//! understand is ignored.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokKind {
    Ident,
    Num,
    Punct,
    Str,
}

#[derive(Clone, Debug)]
pub struct Tok {
    pub s: String,
    pub kind: TokKind,
}

impl Tok {
    fn is(&self, s: &str) -> bool {
        self.s == s
    }
    fn ident(&self) -> bool {
        self.kind == TokKind::Ident
    }
}

pub fn tokenize(src: &str) -> Vec<Tok> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let mut line_start = true;
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            line_start = true;
            i += 1;
            continue;
        }
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if line_start && c == b'#' {
            // preprocessor residue (#pragma ...): skip line
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        line_start = false;
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' || c == b'$' {
            let st = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'$') {
                i += 1;
            }
            out.push(Tok { s: src[st..i].to_string(), kind: TokKind::Ident });
            continue;
        }
        if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let st = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'.' || b[i] == b'_') {
                // exponent sign
                if (b[i] == b'e' || b[i] == b'E') && i + 1 < b.len() && (b[i + 1] == b'-' || b[i + 1] == b'+') && !src[st..i].starts_with("0x") {
                    i += 2;
                    continue;
                }
                i += 1;
            }
            out.push(Tok { s: src[st..i].to_string(), kind: TokKind::Num });
            continue;
        }
        if c == b'"' || c == b'\'' {
            let st = i;
            i += 1;
            while i < b.len() && b[i] != c {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            let e = i.min(b.len());
            out.push(Tok { s: String::from_utf8_lossy(&b[st..e]).into_owned(), kind: TokKind::Str });
            continue;
        }
        // punctuation
        let two = if i + 1 < b.len() { &src[i..i + 2] } else { "" };
        if two == "::" || two == "->" || two == "==" || two == "!=" || two == "<=" || two == ">=" || two == "&&" || two == "||" || two == "++" || two == "--" {
            out.push(Tok { s: two.to_string(), kind: TokKind::Punct });
            i += 2;
            continue;
        }
        if i + 2 < b.len() && &src[i..i + 3] == "..." {
            out.push(Tok { s: "...".into(), kind: TokKind::Punct });
            i += 3;
            continue;
        }
        let ch_len = src[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
        out.push(Tok { s: src[i..i + ch_len].to_string(), kind: TokKind::Punct });
        i += ch_len;
    }
    out
}

/// A function declaration as written (tokens), plus scope.
#[derive(Clone, Debug)]
pub struct RawDecl {
    /// Enclosing scope (namespace/class), "" for global.
    pub scope: String,
    /// Possibly qualified name as written (out-of-line definitions: `CVector3f::Dot`).
    pub name: String,
    pub ret: Vec<String>,
    pub params: Vec<Vec<String>>,
    pub variadic: bool,
    pub is_const: bool,
    pub is_static: bool,
    pub is_virtual: bool,
    pub is_pure: bool,
    pub body: Option<String>,
    /// declared inside a class body
    pub in_class: bool,
    /// Template parameters in scope (enclosing class template's, then the function's own).
    pub template_params: Vec<String>,
    /// Access section the declaration appeared in (Public outside classes).
    pub access: mwdec_core::Access,
    /// Constructor initializer list tokens (definitions only).
    pub init_list: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ScanResult {
    /// qualified names of non-template class/struct/union definitions
    pub classes: Vec<String>,
    /// class -> is "struct"/"union" keyword (public default)
    pub enums: Vec<String>,
    pub typedefs: Vec<String>,
    /// qualified names of namespace-scope variables and static data members
    pub globals: Vec<String>,
    pub decls: Vec<RawDecl>,
    /// (class, field name, declared type tokens) for non-static data members
    pub fields: Vec<(String, String, Vec<String>, mwdec_core::Access)>,
    /// class-key / `enum` of each recorded class/enum name (for elaborated specifiers in C)
    pub tag_keyword: std::collections::HashMap<String, String>,
    /// qualified names of class templates (for resolving `rstl::vector<..>` spellings)
    pub templates: Vec<String>,
    pub namespaces: Vec<String>,
    /// member typedefs of class templates: (template, name, type tokens)
    pub template_typedefs: Vec<(String, String, Vec<String>)>,
    /// class template -> its parameter names
    pub template_params: std::collections::HashMap<String, Vec<String>>,
    /// (class, befriended class or function name as written)
    pub friends: Vec<(String, String)>,
    /// (class template, field name, declared type tokens, access) for data members of templates
    pub template_fields: Vec<(String, String, Vec<String>, mwdec_core::Access)>,
    /// namespace-scope variables declared at an absolute address (`T name[N] : 0xCC005000;`)
    pub abs_addrs: Vec<(String, u32)>,
}

const SPECIFIERS: &[&str] = &["virtual", "static", "inline", "explicit", "extern", "friend", "mutable", "register", "__inline", "__declspec"];

struct Scanner<'t> {
    t: &'t [Tok],
    i: usize,
    out: ScanResult,
    /// active template parameters (non-empty inside a class template body)
    tparams: Vec<String>,
    c_mode: bool,
    /// current access section and the default for the next class body entered
    access: mwdec_core::Access,
    next_default: mwdec_core::Access,
}

fn qual(scope: &str, name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{scope}::{name}")
    }
}

pub fn scan(src: &str) -> ScanResult {
    scan_lang(src, false)
}

/// `c_mode`: C has a single tag namespace, so structs/enums nested in a struct are global.
pub fn scan_lang(src: &str, c_mode: bool) -> ScanResult {
    let toks = tokenize(src);
    let mut s = Scanner { t: &toks, i: 0, out: ScanResult::default(), tparams: Vec::new(), c_mode, access: mwdec_core::Access::Public, next_default: mwdec_core::Access::Public };
    s.scope("", false);
    let mut out = s.out;
    out.classes.dedup();
    out
}

impl<'t> Scanner<'t> {
    fn peek(&self, k: usize) -> Option<&'t Tok> {
        self.t.get(self.i + k)
    }

    /// Skip a balanced group starting at the current open token.
    fn skip_balanced(&mut self) {
        let open = &self.t[self.i].s;
        let close = match open.as_str() {
            "{" => "}",
            "(" => ")",
            "[" => "]",
            "<" => ">",
            _ => {
                self.i += 1;
                return;
            }
        };
        let open = open.clone();
        let mut depth = 0;
        while self.i < self.t.len() {
            let s = &self.t[self.i].s;
            if *s == open {
                depth += 1;
            } else if s == close {
                depth -= 1;
                if depth == 0 {
                    self.i += 1;
                    return;
                }
            }
            self.i += 1;
        }
    }

    /// Text of a balanced `{...}` group (exclusive of braces), consuming it.
    fn take_body(&mut self) -> String {
        let st = self.i;
        self.skip_balanced();
        let en = self.i;
        let inner = if en > st + 1 { &self.t[st + 1..en - 1] } else { &[][..] };
        inner.iter().map(|t| t.s.as_str()).collect::<Vec<_>>().join(" ")
    }

    fn skip_to_semicolon(&mut self) {
        while self.i < self.t.len() {
            match self.t[self.i].s.as_str() {
                ";" => {
                    self.i += 1;
                    return;
                }
                "{" | "(" | "[" => self.skip_balanced(),
                "}" => return,
                _ => self.i += 1,
            }
        }
    }

    /// Collect a declaration's tokens up to a top-level `;` or `{` (not consumed).
    fn collect(&mut self) -> Vec<&'t Tok> {
        let mut out = Vec::new();
        let mut depth = 0i32;
        while self.i < self.t.len() {
            let t = &self.t[self.i];
            match t.s.as_str() {
                "(" | "[" => depth += 1,
                ")" | "]" => depth -= 1,
                ";" | "{" | "}" if depth <= 0 => return out,
                _ => {}
            }
            out.push(t);
            self.i += 1;
        }
        out
    }

    fn scope(&mut self, scope: &str, is_class: bool) {
        let saved = self.access;
        self.access = if is_class { self.next_default } else { mwdec_core::Access::Public };
        self.scope_inner(scope, is_class);
        self.access = saved;
    }

    fn scope_inner(&mut self, scope: &str, is_class: bool) {
        while self.i < self.t.len() {
            let t = &self.t[self.i];
            match t.s.as_str() {
                "}" => {
                    self.i += 1;
                    return;
                }
                ";" => {
                    self.i += 1;
                    continue;
                }
                "public" | "private" | "protected" if self.peek(1).is_some_and(|n| n.is(":")) => {
                    self.access = match t.s.as_str() {
                        "public" => mwdec_core::Access::Public,
                        "protected" => mwdec_core::Access::Protected,
                        _ => mwdec_core::Access::Private,
                    };
                    self.i += 2;
                    continue;
                }
                "template" => {
                    self.template(scope, is_class);
                    continue;
                }
                "namespace" => {
                    self.i += 1;
                    let mut name = String::new();
                    if let Some(n) = self.peek(0).filter(|n| n.ident()) {
                        name = n.s.clone();
                        self.i += 1;
                    }
                    if self.peek(0).is_some_and(|n| n.is("{")) {
                        self.i += 1;
                        let q = if name.is_empty() { scope.to_string() } else { qual(scope, &name) };
                        if !name.is_empty() && !self.out.namespaces.contains(&q) {
                            self.out.namespaces.push(q.clone());
                        }
                        self.scope(&q, false);
                    } else {
                        self.skip_to_semicolon();
                    }
                    continue;
                }
                "extern" if self.peek(1).is_some_and(|n| n.kind == TokKind::Str) => {
                    self.i += 2;
                    if self.peek(0).is_some_and(|n| n.is("{")) {
                        self.i += 1;
                        self.scope(scope, is_class);
                    }
                    continue;
                }
                "friend" if is_class => {
                    // `friend class X;` / `friend R f(...);`: X / f may use the class's privates
                    let st = self.i + 1;
                    self.skip_to_semicolon();
                    let toks: Vec<&str> = self.t[st..self.i].iter().map(|t| t.s.as_str()).collect();
                    let name = match toks.iter().position(|t| *t == "(") {
                        Some(p) => {
                            // the declarator name before `(`, with `operator` joined
                            match toks[..p].iter().rposition(|t| *t == "operator") {
                                Some(o) => toks[o..p].concat().replacen("operator", "operator", 1),
                                None => toks[..p].last().map(|s| s.to_string()).unwrap_or_default(),
                            }
                        }
                        None => toks.iter().rev().find(|t| t.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')).map(|s| s.to_string()).unwrap_or_default(),
                    };
                    if !name.is_empty() && !matches!(name.as_str(), "class" | "struct") {
                        self.out.friends.push((scope.to_string(), name));
                    }
                    continue;
                }
                "using" | "friend" | "static_assert" | "__static_assert" => {
                    self.skip_to_semicolon();
                    continue;
                }
                _ => {}
            }
            let start = self.i;
            let decl = self.collect();
            if self.i >= self.t.len() {
                return;
            }
            let term = self.t[self.i].s.as_str();
            if term == "}" {
                // stray tokens before scope end
                continue;
            }
            if term == "{" {
                self.braced_decl(scope, is_class, &decl);
            } else {
                self.i += 1; // ';'
                self.plain_decl(scope, is_class, &decl);
            }
            if self.i == start {
                self.i += 1;
            }
        }
    }

    fn template(&mut self, scope: &str, is_class: bool) {
        // template < params > declaration
        self.i += 1;
        let mut params = Vec::new();
        let mut explicit_spec = true;
        if self.peek(0).is_some_and(|t| t.is("<")) {
            let st = self.i;
            self.skip_balanced();
            let inner: Vec<&Tok> = self.t[st + 1..self.i.saturating_sub(1)].iter().collect();
            explicit_spec = inner.is_empty();
            for part in split_top(&inner, ",") {
                let cut = part.iter().position(|t| t.is("=")).unwrap_or(part.len());
                if let Some(n) = part[..cut]
                    .iter()
                    .rev()
                    .find(|t| t.ident() && !matches!(t.s.as_str(), "class" | "typename") && !is_type_keyword(&t.s))
                {
                    params.push(n.s.clone());
                }
            }
        }
        if self.peek(0).is_some_and(|t| t.is("template")) {
            // nested template header (member template of a class template defined out of line)
            let save = std::mem::take(&mut self.tparams);
            self.tparams = save.iter().cloned().chain(params.iter().cloned()).collect();
            self.template(scope, is_class);
            self.tparams = save;
            return;
        }
        let decl = self.collect();
        let kw = decl.iter().position(|t| t.is("class") || t.is("struct") || t.is("union"));
        let paren = decl.iter().position(|t| t.is("("));
        let classlike = kw.is_some() && (paren.is_none() || paren > kw) && !decl.iter().any(|t| t.is("typedef"));
        if classlike {
            let k = kw.unwrap();
            let name = decl.get(k + 1).filter(|n| n.ident()).map(|n| n.s.clone());
            let specialization = decl.get(k + 2).is_some_and(|t| t.is("<"));
            let q = name.as_ref().map(|n| qual(scope, n));
            if let (Some(q), false) = (&q, specialization) {
                if !self.out.templates.contains(q) {
                    self.out.templates.push(q.clone());
                }
            }
            if self.peek(0).is_some_and(|t| t.is("{")) {
                match (&q, specialization || explicit_spec || !self.tparams.is_empty()) {
                    (Some(q), false) => {
                        self.out.template_params.insert(q.clone(), params.clone());
                        self.i += 1;
                        self.tparams = params;
                        self.next_default = if decl[k].is("class") { mwdec_core::Access::Private } else { mwdec_core::Access::Public };
                        self.scope(q, true);
                        self.tparams = Vec::new();
                        self.skip_to_semicolon();
                    }
                    _ => {
                        self.skip_balanced();
                        self.skip_to_semicolon();
                    }
                }
            } else if self.peek(0).is_some_and(|t| t.is(";")) {
                self.i += 1;
            }
            return;
        }
        let all_params: Vec<String> = self.tparams.iter().cloned().chain(params.iter().cloned()).collect();
        if paren.is_some() && !explicit_spec {
            let body = if self.peek(0).is_some_and(|t| t.is("{")) {
                Some(self.take_body())
            } else {
                if self.peek(0).is_some_and(|t| t.is(";")) {
                    self.i += 1;
                }
                None
            };
            if let Some(mut d) = parse_fn_decl(scope, is_class, &decl, body) {
                d.template_params = all_params;
                d.access = self.access;
                self.out.decls.push(d);
            }
            return;
        }
        if self.peek(0).is_some_and(|t| t.is("{")) {
            self.skip_balanced();
        }
        if self.peek(0).is_some_and(|t| t.is(";")) {
            self.i += 1;
        }
    }

    fn braced_decl(&mut self, scope: &str, is_class: bool, decl: &[&'t Tok]) {
        let words: Vec<&str> = decl.iter().map(|t| t.s.as_str()).collect();
        let is_typedef = words.first() == Some(&"typedef");
        let kw = words.iter().position(|w| matches!(*w, "class" | "struct" | "union" | "enum"));
        let has_paren_before_kw = kw.is_some_and(|k| words[..k].contains(&"("));
        if let (Some(k), false, false) = (kw, has_paren_before_kw, self.tparams.is_empty()) {
            // nested type inside a class template: not a concrete type, but its member functions
            // are recorded (`red_black_tree::const_iterator::operator==`, with the enclosing
            // template's parameters) for tools that instantiate them
            let simple = matches!(words[k], "class" | "struct") && k == 0 && words.len() >= 2 && (words.len() == 2 || words[2] == ":") && decl[1].ident() && self.peek(0).is_some_and(|t| t.is("{"));
            if simple && !self.c_mode {
                let name = words[k + 1].to_string();
                self.i += 1; // '{'
                let mut sub = Scanner {
                    t: self.t,
                    i: self.i,
                    out: ScanResult::default(),
                    tparams: self.tparams.clone(),
                    c_mode: self.c_mode,
                    access: mwdec_core::Access::Public,
                    next_default: if words[k] == "class" { mwdec_core::Access::Private } else { mwdec_core::Access::Public },
                };
                sub.scope(&qual(scope, &name), true);
                self.i = sub.i;
                self.out.decls.extend(sub.out.decls);
                self.skip_to_semicolon();
                return;
            }
            self.skip_balanced();
            self.skip_to_semicolon();
            return;
        }
        if let (Some(k), false) = (kw, has_paren_before_kw) {
            let is_enum = words[k] == "enum";
            // name: identifiers (possibly qualified) after keyword, until ':' or end
            let mut name = String::new();
            let mut j = k + 1;
            let mut specialization = false;
            while j < words.len() {
                let w = words[j];
                if decl[j].ident() || w == "::" {
                    name.push_str(w);
                    j += 1;
                } else {
                    if w == "<" {
                        specialization = true;
                    }
                    break;
                }
            }
            if is_enum {
                if !name.is_empty() {
                    let q = if self.c_mode { name.clone() } else { qual(scope, &name) };
                    self.out.enums.push(q.clone());
                    self.out.tag_keyword.insert(q, "enum".into());
                }
                self.skip_balanced();
                self.trailing_declarators(scope, is_typedef, if name.is_empty() { None } else { Some(qual(scope, &name)) });
                return;
            }
            if specialization {
                self.skip_balanced();
                self.skip_to_semicolon();
                return;
            }
            self.i += 1; // '{'
            let q = if name.is_empty() {
                None
            } else if self.c_mode {
                Some(name.clone())
            } else {
                Some(qual(scope, &name))
            };
            match &q {
                Some(q) => {
                    self.out.classes.push(q.clone());
                    self.out.tag_keyword.insert(q.clone(), words[k].to_string());
                    self.next_default = if words[k] == "class" { mwdec_core::Access::Private } else { mwdec_core::Access::Public };
                    self.scope(q, true);
                }
                None => {
                    // anonymous struct/union: members belong to the enclosing class for access;
                    // scan but don't record (no stable name)
                    let mut sub = Scanner { t: self.t, i: self.i, out: ScanResult::default(), tparams: Vec::new(), c_mode: self.c_mode, access: mwdec_core::Access::Public, next_default: mwdec_core::Access::Public };
                    sub.scope("@anon", true);
                    self.i = sub.i;
                }
            }
            self.trailing_declarators(scope, is_typedef, q);
            return;
        }
        // function definition?
        if words.contains(&"(") {
            let body = self.take_body();
            if let Some(mut d) = parse_fn_decl(scope, is_class, decl, Some(body)) {
                d.template_params = self.tparams.clone();
                d.access = self.access;
                self.out.decls.push(d);
            }
            return;
        }
        // initializer `= { ... }` or something else
        self.skip_balanced();
        self.skip_to_semicolon();
        let _ = is_class;
    }

    /// After `}` of a class/enum: `} a, *b;` or `} Name;` for typedefs.
    fn trailing_declarators(&mut self, scope: &str, is_typedef: bool, _ty: Option<String>) {
        let decl = self.collect();
        if self.peek(0).is_some_and(|t| t.is(";")) {
            self.i += 1;
        }
        if is_typedef {
            for part in decl.split(|t| t.is(",")) {
                if let Some(n) = part.iter().rev().find(|t| t.ident()) {
                    self.out.typedefs.push(qual(scope, &n.s));
                }
            }
        }
    }

    fn plain_decl(&mut self, scope: &str, is_class: bool, decl: &[&'t Tok]) {
        if decl.is_empty() {
            return;
        }
        let first = decl[0].s.as_str();
        if first == "typedef" {
            if let Some(n) = declarator_name(&decl[1..]) {
                if self.tparams.is_empty() {
                    self.out.typedefs.push(qual(scope, &n));
                } else {
                    let mut toks: Vec<String> = decl[1..].iter().map(|t| t.s.clone()).collect();
                    if let Some(p) = toks.iter().rposition(|w| *w == n) {
                        toks.remove(p);
                    }
                    self.out.template_typedefs.push((scope.to_string(), n, toks));
                }
            }
            return;
        }
        if matches!(first, "friend" | "using" | "template" | "namespace" | "asm" | "__asm") {
            return;
        }
        // forward declarations: `class X;`
        if matches!(first, "class" | "struct" | "union" | "enum") && decl.len() <= 3 && !decl.iter().any(|t| t.is("(")) {
            return;
        }
        let is_fn = is_function_decl(decl);
        if is_fn {
            if let Some(mut d) = parse_fn_decl(scope, is_class, decl, None) {
                d.template_params = self.tparams.clone();
                d.access = self.access;
                self.out.decls.push(d);
            }
            return;
        }
        if !self.tparams.is_empty() {
            // data members of class templates: no concrete layout here; their declared types are
            // kept apart (`T* mItems`) for what DWARF can't say (see resolve::patch_void_pointers)
            if is_class && !decl.iter().any(|t| t.is("static")) {
                if let Some(name) = declarator_name(decl) {
                    let toks: Vec<String> = decl.iter().map(|t| t.s.clone()).filter(|w| *w != name).collect();
                    let cut = toks.iter().position(|w| w == ":" || w == "=").unwrap_or(toks.len());
                    self.out.template_fields.push((scope.to_string(), name, toks[..cut].to_vec(), self.access));
                }
            }
            return;
        }
        // variable(s)
        let is_static = decl.iter().any(|t| t.is("static"));
        // split declarators at top-level commas
        let parts = split_top(decl, ",");
        let mut base: Vec<String> = Vec::new();
        for (pi, part) in parts.iter().enumerate() {
            let Some(name) = declarator_name(part) else { continue };
            let toks: Vec<String> = part.iter().map(|t| t.s.clone()).collect();
            let ty_toks = if pi == 0 {
                // everything except the name and bitfield/initializer
                let cut = toks.iter().position(|w| w == ":" || w == "=").unwrap_or(toks.len());
                let mut v: Vec<String> = toks[..cut].to_vec();
                if let Some(p) = v.iter().rposition(|w| *w == name) {
                    v.remove(p);
                }
                base = v.iter().filter(|w| !matches!(w.as_str(), "*" | "&")).cloned().collect();
                v
            } else {
                let mut v = base.clone();
                for w in &toks {
                    if w == "*" || w == "&" {
                        v.push(w.clone());
                    }
                }
                v
            };
            if is_class && !is_static {
                if scope != "@anon" {
                    self.out.fields.push((scope.to_string(), name, ty_toks, self.access));
                }
            } else if scope != "@anon" && !decl.iter().any(|t| t.is("operator")) {
                self.out.globals.push(qual(scope, &name));
                // `T name[N] : 0xADDR` (CodeWarrior absolute-address variable)
                if let Some(c) = toks.iter().position(|w| w == ":") {
                    let lit = toks.get(c + 1).map(|s| s.trim_end_matches(|ch: char| matches!(ch, 'u' | 'U' | 'l' | 'L')));
                    let addr = lit.and_then(|s| match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                        Some(h) => u32::from_str_radix(h, 16).ok(),
                        None => s.parse::<u32>().ok(),
                    });
                    if let (Some(a), true) = (addr, toks.len() == c + 2) {
                        self.out.abs_addrs.push((qual(scope, &name), a));
                    }
                }
            }
        }
    }
}

/// Split tokens at a top-level separator (outside (), [], <>, {}).
pub fn split_top<'a, T: AsRef<str>>(toks: &'a [T], sep: &str) -> Vec<&'a [T]> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut st = 0;
    for (i, t) in toks.iter().enumerate() {
        match t.as_ref() {
            "(" | "[" | "<" | "{" => depth += 1,
            ")" | "]" | ">" | "}" => depth -= 1,
            s if s == sep && depth == 0 => {
                out.push(&toks[st..i]);
                st = i + 1;
            }
            _ => {}
        }
    }
    out.push(&toks[st..]);
    out
}

impl AsRef<str> for Tok {
    fn as_ref(&self) -> &str {
        &self.s
    }
}

/// Name declared by a (variable/typedef) declaration: `(*name)(...)` or the last top-level
/// identifier before `[`, `:`, `=`.
fn declarator_name(decl: &[&Tok]) -> Option<String> {
    // function pointer: ( * name )
    for w in decl.windows(3) {
        if w[0].is("(") && w[1].is("*") && w[2].ident() {
            return Some(w[2].s.clone());
        }
    }
    let mut depth = 0i32;
    let mut last = None;
    for t in decl {
        match t.s.as_str() {
            "(" | "[" | "<" => depth += 1,
            ")" | "]" | ">" => depth -= 1,
            ":" | "=" if depth == 0 => break,
            _ if depth == 0 && t.ident() && !is_type_keyword(&t.s) => last = Some(t.s.clone()),
            _ => {}
        }
    }
    last
}

fn is_type_keyword(s: &str) -> bool {
    matches!(
        s,
        "const" | "volatile" | "unsigned" | "signed" | "short" | "long" | "int" | "char" | "float" | "double" | "void" | "bool" | "wchar_t"
            | "static" | "extern" | "inline" | "mutable" | "struct" | "class" | "union" | "enum" | "typename" | "virtual" | "register"
    )
}

/// Does this `;`-terminated declaration declare a function (vs a variable / function pointer)?
fn is_function_decl(decl: &[&Tok]) -> bool {
    let Some(p) = decl.iter().position(|t| t.is("(")) else { return false };
    if p == 0 {
        return false;
    }
    // function pointer variable: `R (*name)(...)`
    if decl.get(p + 1).is_some_and(|t| t.is("*")) {
        return false;
    }
    let prev = &decl[p - 1];
    // `int x(5);` direct-init is rare in headers; treat ident( as function
    prev.ident() || prev.kind == TokKind::Punct && decl.iter().any(|t| t.is("operator"))
}

fn parse_fn_decl(scope: &str, in_class: bool, decl: &[&Tok], body: Option<String>) -> Option<RawDecl> {
    let mut toks: Vec<&Tok> = decl.to_vec();
    let mut is_static = false;
    let mut is_virtual = false;
    // strip leading specifiers
    let mut k = 0;
    while k < toks.len() {
        match toks[k].s.as_str() {
            "static" => is_static = true,
            "virtual" => is_virtual = true,
            "__declspec" => {
                // __declspec(...)
                toks.remove(k);
                if k < toks.len() && toks[k].is("(") {
                    let mut d = 0;
                    while k < toks.len() {
                        let s = toks[k].s.clone();
                        toks.remove(k);
                        if s == "(" {
                            d += 1;
                        } else if s == ")" {
                            d -= 1;
                            if d == 0 {
                                break;
                            }
                        }
                    }
                }
                continue;
            }
            s if SPECIFIERS.contains(&s) => {}
            _ => {
                k += 1;
                continue;
            }
        }
        toks.remove(k);
    }
    // locate name + param list
    let (name_start, lparen) = if let Some(op) = toks.iter().position(|t| t.is("operator")) {
        // operator X ( ... )  /  operator ( ) ( ... )
        let mut j = op + 1;
        if toks.get(j).is_some_and(|t| t.is("(")) && toks.get(j + 1).is_some_and(|t| t.is(")")) {
            j += 2;
        }
        while j < toks.len() && !toks[j].is("(") {
            j += 1;
        }
        // include qualification before `operator`
        let mut ns = op;
        while ns >= 2 && toks[ns - 1].is("::") && toks[ns - 2].ident() {
            ns -= 2;
        }
        (ns, j)
    } else {
        let mut p = toks.iter().position(|t| t.is("("))?;
        if p == 0 || !toks[p - 1].ident() {
            return None;
        }
        let mut ns = p - 1;
        if ns >= 1 && toks[ns - 1].is("~") {
            ns -= 1;
        }
        let mut targs_skipped: Vec<(usize, usize)> = Vec::new();
        while ns >= 2 && toks[ns - 1].is("::") && (toks[ns - 2].ident() || toks[ns - 2].is(">")) {
            if toks[ns - 2].is(">") {
                // templated qualifier `vector<T>::reserve`: drop the argument list
                let mut d = 0;
                let mut j = ns - 2;
                loop {
                    if toks[j].is(">") {
                        d += 1;
                    } else if toks[j].is("<") {
                        d -= 1;
                        if d == 0 {
                            break;
                        }
                    }
                    if j == 0 {
                        return None;
                    }
                    j -= 1;
                }
                if j == 0 || !toks[j - 1].ident() {
                    return None;
                }
                targs_skipped.push((j, ns - 2));
                ns = j - 1;
                continue;
            }
            ns -= 2;
        }
        for (a, b) in targs_skipped {
            toks.drain(a..=b);
            p -= b - a + 1;
        }
        (ns, p)
    };
    if lparen >= toks.len() {
        return None;
    }
    let name: String = match toks[name_start..lparen].iter().position(|t| t.is("operator")) {
        Some(k) => {
            // `operator+=`, `operator()`, `operator new`, conversion `operator bool`: punctuators
            // join without spaces, identifiers keep one
            let k = name_start + k;
            let q: String = toks[name_start..k].iter().map(|t| t.s.as_str()).collect();
            let mut op = String::from("operator");
            for t in &toks[k + 1..lparen] {
                if t.s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
                    op.push(' ');
                }
                op.push_str(&t.s);
            }
            format!("{q}{op}")
        }
        None => {
            let name: String = toks[name_start..lparen].iter().map(|t| t.s.as_str()).collect::<Vec<_>>().join(" ");
            name.replace(" :: ", "::").replace("~ ", "~")
        }
    };
    let ret: Vec<String> = toks[..name_start].iter().map(|t| t.s.clone()).collect();
    // params
    let mut depth = 0;
    let mut rparen = lparen;
    for (j, t) in toks.iter().enumerate().skip(lparen) {
        if t.is("(") {
            depth += 1;
        } else if t.is(")") {
            depth -= 1;
            if depth == 0 {
                rparen = j;
                break;
            }
        }
    }
    let inner: Vec<&Tok> = toks[lparen + 1..rparen].to_vec();
    let mut params = Vec::new();
    let mut variadic = false;
    if !(inner.is_empty() || (inner.len() == 1 && inner[0].is("void"))) {
        for part in split_top(&inner, ",") {
            let p: Vec<String> = part.iter().map(|t| t.s.clone()).collect();
            if p.len() == 1 && p[0] == "..." {
                variadic = true;
                continue;
            }
            // strip default argument
            let cut = {
                let mut d = 0i32;
                let mut c = p.len();
                for (j, w) in p.iter().enumerate() {
                    match w.as_str() {
                        "(" | "<" | "[" => d += 1,
                        ")" | ">" | "]" => d -= 1,
                        "=" if d == 0 => {
                            c = j;
                            break;
                        }
                        _ => {}
                    }
                }
                c
            };
            params.push(p[..cut].to_vec());
        }
    }
    let tail: Vec<&str> = toks[rparen + 1..].iter().map(|t| t.s.as_str()).collect();
    // tail up to ctor-initializer ':' (definitions) is const/throw/=0
    let tail_end = tail.iter().position(|w| *w == ":").unwrap_or(tail.len());
    let init_list = if tail_end < tail.len() && body.is_some() { Some(tail[tail_end + 1..].join(" ")) } else { None };
    let tail = &tail[..tail_end];
    let is_const = tail.first() == Some(&"const");
    let is_pure = tail.windows(2).any(|w| w[0] == "=" && w[1] == "0");
    Some(RawDecl {
        scope: scope.to_string(),
        name,
        ret,
        params,
        variadic,
        is_const,
        is_static,
        is_virtual,
        is_pure,
        body,
        in_class,
        template_params: Vec::new(),
        access: mwdec_core::Access::Public,
        init_list,
    })
}



