//! Resolve type spellings from the headers (token lists) into `Type`s using the TypeDb,
//! build `DeclInfo`s from scanned declarations, and canonicalize types for comparison.
use crate::mangle::{split_scope, type_to_string};
use crate::scan::{split_top, ScanResult};
use mwdec_core::*;
use std::collections::{BTreeMap, HashMap, HashSet};

pub fn qual(scope: &str, name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{scope}::{name}")
    }
}

/// Expand typedefs (recursively) so two spellings of the same type compare equal.
pub fn canonical(db: &TypeDb, t: &Type) -> Type {
    canon_rec(db, t, 0)
}

fn canon_rec(db: &TypeDb, t: &Type, depth: u32) -> Type {
    if depth > 24 {
        return t.clone();
    }
    match t {
        Type::Named(n) => match db.typedefs.get(n) {
            Some(u) => canon_rec(db, u, depth + 1),
            None => t.clone(),
        },
        Type::Ptr(x) => Type::Ptr(Box::new(canon_rec(db, x, depth + 1))),
        Type::Ref(x) => Type::Ref(Box::new(canon_rec(db, x, depth + 1))),
        Type::Const(x) => match canon_rec(db, x, depth + 1) {
            // const const T == const T (typedef of const)
            Type::Const(y) => Type::Const(y),
            y => Type::Const(Box::new(y)),
        },
        Type::Volatile(x) => Type::Volatile(Box::new(canon_rec(db, x, depth + 1))),
        Type::Array(x, n) => Type::Array(Box::new(canon_rec(db, x, depth + 1)), *n),
        Type::FuncPtr(sig) => {
            let mut s = (**sig).clone();
            s.ret = canon_rec(db, &s.ret, depth + 1);
            for p in &mut s.params {
                p.ty = canon_rec(db, &p.ty, depth + 1);
                p.name = None;
            }
            Type::FuncPtr(Box::new(s))
        }
        _ => t.clone(),
    }
}

/// Canonical form for parameter matching: typedefs expanded, top-level cv dropped.
pub fn param_key(db: &TypeDb, t: &Type) -> Type {
    canonical(db, t).unqualified().clone()
}

pub struct Resolver<'a> {
    pub db: &'a TypeDb,
    known: HashSet<String>,
    templates: HashSet<String>,
    /// template base ("rstl::vector") -> instantiated class names in db
    instances: HashMap<String, Vec<String>>,
    /// Template parameter names in effect (resolve to themselves, never looked up).
    pub tparams: Vec<String>,
}

fn template_base(name: &str) -> Option<String> {
    let parts = split_scope(name);
    let last = parts.last()?;
    let lt = last.find('<')?;
    let mut v: Vec<&str> = parts[..parts.len() - 1].to_vec();
    v.push(&last[..lt]);
    Some(v.join("::"))
}

impl<'a> Resolver<'a> {
    pub fn new(db: &'a TypeDb, templates: &[String], extra_known: &[String]) -> Self {
        let mut known: HashSet<String> = HashSet::new();
        known.extend(db.classes.keys().cloned());
        known.extend(db.enums.keys().cloned());
        known.extend(db.typedefs.keys().cloned());
        known.extend(extra_known.iter().cloned());
        let mut instances: HashMap<String, Vec<String>> = HashMap::new();
        for n in db.classes.keys() {
            if let Some(b) = template_base(n) {
                instances.entry(b).or_default().push(n.clone());
            }
        }
        let mut templates: HashSet<String> = templates.iter().cloned().collect();
        templates.extend(instances.keys().cloned());
        templates.extend(db.templates.keys().cloned());
        Resolver { db, known, templates, instances, tparams: Vec::new() }
    }

    /// Scopes to search for a name written inside `scope`: the scope chain outward, plus base
    /// classes of each class scope.
    fn search_scopes(&self, scope: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut cur: Option<String> = Some(scope.to_string());
        while let Some(s) = cur {
            self.push_with_bases(&s, &mut out, &mut seen, 0);
            cur = if s.is_empty() {
                None
            } else {
                let parts = split_scope(&s);
                Some(parts[..parts.len() - 1].join("::"))
            };
        }
        out
    }

    fn push_with_bases(&self, s: &str, out: &mut Vec<String>, seen: &mut HashSet<String>, depth: u32) {
        if !seen.insert(s.to_string()) || depth > 16 {
            return;
        }
        out.push(s.to_string());
        if let Some(c) = self.db.classes.get(s) {
            for b in &c.bases {
                self.push_with_bases(&b.name, out, seen, depth + 1);
            }
        }
    }

    /// Qualified name of a (class/enum/typedef) name written in `scope`.
    pub fn lookup(&self, written: &str, scope: &str) -> Option<String> {
        let written = written.trim_start_matches("::");
        for s in self.search_scopes(scope) {
            let q = qual(&s, written);
            if self.known.contains(&q) || self.templates.contains(&q) {
                return Some(q);
            }
        }
        None
    }

    /// Find the db instantiation of `base<args...>` (args canonical), allowing defaulted
    /// trailing arguments.
    fn instantiation(&self, base: &str, args: &[String]) -> String {
        let written = format!("{base}<{}>", args.join(", "));
        let written = written.replace(">>", "> >");
        if self.db.classes.contains_key(&written) {
            return written;
        }
        if let Some(list) = self.instances.get(base) {
            let prefix = format!("{base}<{}", args.join(", "));
            let mut best: Option<&String> = None;
            for n in list {
                if n.starts_with(&prefix) && n[prefix.len()..].starts_with(", ") {
                    if best.is_none_or(|b| n.len() < b.len()) {
                        best = Some(n);
                    }
                }
            }
            if let Some(b) = best {
                return b.clone();
            }
        }
        written
    }

    /// Parse a type spelling (tokens of a declaration minus specifiers). Returns the type and
    /// the declarator name, if any. Arrays are kept as arrays.
    pub fn parse(&self, toks: &[String], scope: &str) -> Option<(Type, Option<String>)> {
        let toks: Vec<&str> = toks.iter().map(|s| s.as_str()).filter(|s| !matches!(*s, "struct" | "class" | "enum" | "typename" | "union" | "register" | "mutable" | "static" | "inline" | "virtual" | "extern" | "explicit")).collect();
        if toks.is_empty() {
            return None;
        }
        // function pointer: R ( * name ) ( params )   /  R ( * ) ( params )
        if let Some(p) = toks.iter().position(|t| *t == "(") {
            if toks.get(p + 1) == Some(&"*") {
                let close = (p..toks.len()).find(|&j| toks[j] == ")")?;
                let name = toks[p + 2..close].iter().find(|t| is_ident(t)).map(|s| s.to_string());
                let ret_toks: Vec<String> = toks[..p].iter().map(|s| s.to_string()).collect();
                let (ret, _) = self.parse(&ret_toks, scope)?;
                let lp = close + 1;
                if toks.get(lp) != Some(&"(") {
                    return None;
                }
                let mut depth = 0;
                let mut rp = lp;
                for (j, t) in toks.iter().enumerate().skip(lp) {
                    if *t == "(" {
                        depth += 1;
                    } else if *t == ")" {
                        depth -= 1;
                        if depth == 0 {
                            rp = j;
                            break;
                        }
                    }
                }
                let inner: Vec<String> = toks[lp + 1..rp].iter().map(|s| s.to_string()).collect();
                let mut params = Vec::new();
                let mut variadic = false;
                if !(inner.is_empty() || inner == ["void"]) {
                    for part in split_top(&inner, ",") {
                        if part.len() == 1 && part[0] == "..." {
                            variadic = true;
                            continue;
                        }
                        let (t, n) = self.parse(part, scope)?;
                        params.push(Param { name: n, ty: decay(t) });
                    }
                }
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
                return Some((Type::FuncPtr(Box::new(sig)), name));
            }
        }
        let mut i = 0;
        let mut base_const = false;
        let mut base_volatile = false;
        let mut n_long = 0;
        let mut unsigned: Option<bool> = None;
        let mut builtin: Option<&str> = None;
        let mut named: Option<Type> = None;
        // specifiers / base type
        while i < toks.len() {
            let t = toks[i];
            match t {
                "const" => base_const = true,
                "volatile" => base_volatile = true,
                "unsigned" => unsigned = Some(true),
                "signed" => unsigned = Some(false),
                "long" => n_long += 1,
                "short" | "int" | "char" | "float" | "double" | "void" | "bool" | "wchar_t" => {
                    if builtin.is_none() || builtin == Some("int") {
                        builtin = Some(t);
                    }
                }
                _ if (is_ident(t) || t == "::") && named.is_none() && builtin.is_none() && unsigned.is_none() && n_long == 0 => {
                    let (ty, ni) = self.parse_named(&toks, i, scope)?;
                    named = Some(ty);
                    i = ni;
                    continue;
                }
                _ => break,
            }
            i += 1;
        }
        let mut ty = if let Some(n) = named {
            n
        } else {
            let signed = unsigned != Some(true);
            match (builtin, n_long) {
                (Some("void"), _) => Type::Void,
                (Some("bool"), _) => Type::Bool,
                (Some("float"), _) => Type::Float { size: 4 },
                (Some("double"), _) => Type::Float { size: 8 },
                (Some("wchar_t"), _) => Type::WChar,
                (Some("char"), _) => match unsigned {
                    None => Type::Char,
                    Some(true) => Type::Int { size: 1, signed: false },
                    Some(false) => Type::Int { size: 1, signed: true },
                },
                (Some("short"), _) => Type::Int { size: 2, signed },
                (_, 1) => Type::Long { signed },
                (_, n) if n >= 2 => Type::Int { size: 8, signed },
                (Some("int"), _) | (None, 0) if builtin.is_some() || unsigned.is_some() => Type::Int { size: 4, signed },
                _ => return None,
            }
        };
        if base_volatile {
            ty = Type::Volatile(Box::new(ty));
        }
        if base_const {
            ty = Type::Const(Box::new(ty));
        }
        // declarator: * & const, name, [N]
        let mut name = None;
        let mut dims: Vec<u32> = Vec::new();
        while i < toks.len() {
            match toks[i] {
                "*" => ty = Type::Ptr(Box::new(ty)),
                "&" => ty = Type::Ref(Box::new(ty)),
                "const" => {
                    if !matches!(ty, Type::Const(_)) {
                        ty = Type::Const(Box::new(ty))
                    }
                }
                "volatile" => ty = Type::Volatile(Box::new(ty)),
                "[" => {
                    let n = toks.get(i + 1).and_then(|s| parse_int(s)).unwrap_or(0);
                    dims.push(n);
                    while i < toks.len() && toks[i] != "]" {
                        i += 1;
                    }
                }
                ":" | "=" => break,
                t if is_ident(t) && name.is_none() => name = Some(t.to_string()),
                _ => {}
            }
            i += 1;
        }
        for &n in dims.iter().rev() {
            ty = Type::Array(Box::new(ty), n);
        }
        Some((ty, name))
    }

    /// Parse a (possibly qualified, possibly templated) name starting at `i`.
    fn parse_named(&self, toks: &[&str], mut i: usize, scope: &str) -> Option<(Type, usize)> {
        let mut parts: Vec<String> = Vec::new();
        let mut templated_last: Option<Vec<String>>;
        if toks.get(i) == Some(&"::") {
            i += 1;
        }
        loop {
            let id = *toks.get(i)?;
            if !is_ident(id) {
                return None;
            }
            i += 1;
            let mut comp = id.to_string();
            templated_last = None;
            if toks.get(i) == Some(&"<") {
                // template args
                let st = i + 1;
                let mut depth = 0;
                let mut en = i;
                for (j, t) in toks.iter().enumerate().skip(i) {
                    match *t {
                        "<" => depth += 1,
                        ">" => {
                            depth -= 1;
                            if depth == 0 {
                                en = j;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if en <= i {
                    return None;
                }
                let inner: Vec<String> = toks[st..en].iter().map(|s| s.to_string()).collect();
                let mut args = Vec::new();
                for part in split_top(&inner, ",") {
                    if part.len() == 1 && parse_int(&part[0]).is_some() {
                        args.push(parse_int(&part[0]).unwrap().to_string());
                    } else if part.len() == 2 && part[0] == "-" && parse_int(&part[1]).is_some() {
                        args.push(format!("-{}", parse_int(&part[1]).unwrap()));
                    } else {
                        let (t, _) = self.parse(part, scope)?;
                        args.push(type_to_string(&canonical(self.db, &t)));
                    }
                }
                templated_last = Some(args);
                i = en + 1;
                comp = id.to_string();
            }
            parts.push(comp);
            if toks.get(i) == Some(&"::") && toks.get(i + 1).is_some_and(|t| is_ident(t)) {
                if let Some(args) = &templated_last {
                    // member of a template instantiation (`rstl::list<T>::iterator`): the
                    // instance's name, then the nested names
                    let written = parts.join("::");
                    let q = self.lookup(&written, scope).unwrap_or(written.clone());
                    let mut name = self.instantiation(&q, args);
                    while toks.get(i) == Some(&"::") && toks.get(i + 1).is_some_and(|t| is_ident(t)) {
                        if toks.get(i + 2) == Some(&"<") {
                            return Some((Type::Named(written), i));
                        }
                        name.push_str("::");
                        name.push_str(toks[i + 1]);
                        i += 2;
                    }
                    return Some((Type::Named(name), i));
                }
                i += 1;
                continue;
            }
            break;
        }
        let written = parts.join("::");
        if parts.len() == 1 && templated_last.is_none() && self.tparams.contains(&written) {
            return Some((Type::Named(written), i));
        }
        let q = self.lookup(&written, scope).unwrap_or(written);
        let name = match templated_last {
            Some(args) => self.instantiation(&q, &args),
            None => q,
        };
        Some((Type::Named(name), i))
    }
}

fn is_ident(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty() && (b[0].is_ascii_alphabetic() || b[0] == b'_') && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_' || *c == b'$')
}

fn parse_int(s: &str) -> Option<u32> {
    let s = s.trim_end_matches(['u', 'U', 'l', 'L']);
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(h, 16).ok()
    } else {
        s.parse().ok()
    }
}

/// Array parameters decay to pointers.
pub fn decay(t: Type) -> Type {
    match t {
        Type::Array(e, _) => Type::Ptr(e),
        t => t,
    }
}

/// Resolve scanned declarations into `TypeDb::decls` and `Class::methods`.
pub fn apply_decls(db: &mut TypeDb, sr: &ScanResult) {
    for (t, ps) in &sr.template_params {
        db.templates.insert(t.clone(), ps.clone());
    }
    for t in &sr.templates {
        db.templates.entry(t.clone()).or_default();
    }
    let mut known: Vec<String> = sr.classes.clone();
    known.extend(sr.enums.iter().cloned());
    known.extend(sr.typedefs.iter().cloned());
    known.extend(sr.template_typedefs.iter().map(|(s, n, _)| qual(s, n)));
    let decls = &sr.decls;
    let mut out: BTreeMap<String, Vec<DeclInfo>> = BTreeMap::new();
    let mut ttds: Vec<(String, Type)> = Vec::new();
    {
        let mut r = Resolver::new(db, &sr.templates, &known);
        for (scope, name, toks) in &sr.template_typedefs {
            r.tparams = sr.template_params.get(scope).cloned().unwrap_or_default();
            if let Some((t, _)) = r.parse(toks, scope) {
                ttds.push((qual(scope, name), t));
            }
        }
        for d in decls {
            r.tparams = d.template_params.clone();
            // out-of-line definitions `A::B::f` inside scope S: class scope = S::A::B
            let parts = split_scope(&d.name);
            let simple = parts.last().copied().unwrap_or("").to_string();
            let outer = parts[..parts.len() - 1].join("::");
            let mut type_scope = d.scope.clone();
            if !outer.is_empty() {
                type_scope = r.lookup(&outer, &d.scope).unwrap_or_else(|| qual(&d.scope, &outer));
            }
            if type_scope.starts_with('@') {
                continue;
            }
            let qualified = qual(&type_scope, &simple);
            let ret = if d.ret.is_empty() {
                if let Some(conv) = simple.strip_prefix("operator ") {
                    let toks: Vec<String> = conv.split_whitespace().map(|s| s.to_string()).collect();
                    r.parse(&toks, &type_scope).map(|x| x.0).unwrap_or(Type::Unknown { size: 0 })
                } else {
                    Type::Void
                }
            } else {
                match r.parse(&d.ret, &type_scope) {
                    Some((t, _)) => t,
                    None => Type::Unknown { size: 0 },
                }
            };
            let mut params = Vec::new();
            let mut ok = true;
            for p in &d.params {
                match r.parse(p, &type_scope) {
                    Some((t, n)) => params.push(Param { name: n, ty: decay(t) }),
                    None => {
                        ok = false;
                        params.push(Param { name: None, ty: Type::Unknown { size: 4 } });
                    }
                }
            }
            let _ = ok;
            let info = DeclInfo {
                qualified_name: qualified.clone(),
                ret,
                params,
                is_const: d.is_const,
                is_static: d.is_static,
                is_virtual: d.is_virtual,
                is_pure: d.is_pure,
                is_inline_defined: d.body.is_some(),
                variadic: d.variadic,
                inline_body: d.body.clone(),
                template_params: d.template_params.clone(),
                access: d.access,
                init_list: d.init_list.clone(),
            };
            let list = out.entry(qualified).or_default();
            // merge an out-of-line definition into the matching in-class declaration
            let key: Vec<Type> = info.params.iter().map(|p| param_key(db, &p.ty)).collect();
            if let Some(ex) = list.iter_mut().find(|e| {
                e.is_const == info.is_const && e.params.iter().map(|p| param_key(db, &p.ty)).collect::<Vec<_>>() == key
            }) {
                if ex.inline_body.is_none() && info.inline_body.is_some() {
                    ex.inline_body = info.inline_body.clone();
                    ex.init_list = info.init_list.clone();
                    ex.is_inline_defined = true;
                }
                ex.is_static |= info.is_static;
                ex.is_virtual |= info.is_virtual;
            } else {
                list.push(info);
            }
        }
    }
    db.template_typedefs.extend(ttds);
    for (k, v) in out {
        db.decls.entry(k).or_default().extend(v);
    }
    fill_methods(db);
}

/// (Re)build `Class::methods` from the non-template `decls` of each class.
pub fn fill_methods(db: &mut TypeDb) {
    for c in db.classes.values_mut() {
        c.methods.clear();
    }
    for (q, list) in &db.decls {
        let parts = split_scope(q);
        if parts.len() < 2 {
            continue;
        }
        let class = parts[..parts.len() - 1].join("::");
        if let Some(c) = db.classes.get_mut(&class) {
            for d in list.iter().filter(|d| d.template_params.is_empty()) {
                c.methods.push(FuncSig {
                    qualified_name: d.qualified_name.clone(),
                    mangled: None,
                    ret: d.ret.clone(),
                    params: d.params.clone(),
                    this_class: if d.is_static { None } else { Some(class.clone()) },
                    is_const: d.is_const,
                    is_static: d.is_static,
                    is_virtual: d.is_virtual,
                    variadic: d.variadic,
                });
            }
        }
    }
}

/// Field access from the header's access sections, for fields whose DIE had none.
pub fn patch_field_access(db: &mut TypeDb, fields: &[(String, String, Vec<String>, Access)], missing: &std::collections::HashSet<(String, String)>) {
    for (class, name, _, access) in fields {
        if !missing.contains(&(class.clone(), name.clone())) {
            continue;
        }
        if let Some(c) = db.classes.get_mut(class) {
            for f in c.fields.iter_mut().filter(|f| &f.name == name) {
                f.access = *access;
            }
        }
    }
}

/// MWCC DWARF encodes `bool` as `unsigned char`; restore it from the header spelling.
pub fn patch_bool_fields(db: &mut TypeDb, fields: &[(String, String, Vec<String>, Access)]) {
    for (class, name, toks, _) in fields {
        let is_bool = toks.iter().any(|t| t == "bool") && !toks.iter().any(|t| t == "*" || t == "&");
        if !is_bool {
            continue;
        }
        if let Some(c) = db.classes.get_mut(class) {
            for f in c.fields.iter_mut().filter(|f| &f.name == name) {
                match &f.ty {
                    Type::Int { size: 1, signed: false } => f.ty = Type::Bool,
                    Type::Array(e, n) if **e == (Type::Int { size: 1, signed: false }) => f.ty = Type::Array(Box::new(Type::Bool), *n),
                    _ => {}
                }
            }
        }
    }
}

/// MWCC's DWARF spells `void**` like `void*` (one FT_pointer, no pointer modifier): restore the
/// second level from the header's declaration, directly (`void** p`) or through a template
/// parameter bound to `void*` (`T* mItems` in `vector<void*>`).
pub fn patch_void_pointers(db: &mut TypeDb, fields: &[(String, String, Vec<String>, Access)], template_params: &HashMap<String, Vec<String>>) {
    let is_void_ptr = |t: &Type| matches!(t.unqualified(), Type::Ptr(x) if matches!(x.unqualified(), Type::Void));
    for (scope, name, toks, _) in fields {
        let toks: Vec<&str> = toks.iter().map(|s| s.as_str()).filter(|t| *t != "const" && *t != "volatile").collect();
        // direct: void * * name
        if toks.len() >= 3 && toks[toks.len() - 3..] == ["void", "*", "*"] {
            if let Some(c) = db.classes.get_mut(scope) {
                for f in c.fields.iter_mut().filter(|f| &f.name == name && is_void_ptr(&f.ty)) {
                    f.ty = Type::Ptr(Box::new(f.ty.clone()));
                }
            }
            continue;
        }
        // through a template parameter: P *
        let Some(params) = template_params.get(scope) else { continue };
        if toks.len() != 2 || toks[1] != "*" {
            continue;
        }
        let Some(idx) = params.iter().position(|p| p == toks[0]) else { continue };
        let prefix = format!("{scope}<");
        let names: Vec<String> = db.classes.keys().filter(|k| k.starts_with(&prefix)).cloned().collect();
        for cn in names {
            let inner = &cn[prefix.len()..cn.len().saturating_sub(1)];
            let args = split_top_level(inner);
            let bound_void_ptr = args.get(idx).is_some_and(|a| a.replace(' ', "") == "void*");
            if !bound_void_ptr {
                continue;
            }
            if let Some(c) = db.classes.get_mut(&cn) {
                for f in c.fields.iter_mut().filter(|f| &f.name == name && is_void_ptr(&f.ty)) {
                    f.ty = Type::Ptr(Box::new(f.ty.clone()));
                }
            }
        }
    }
}

/// `const T* m` in a header where DWARF says `T*` (the pointee's `const` is lost for some
/// types, e.g. `const void*`): the member's declared type.
pub fn patch_const_pointees(db: &mut TypeDb, fields: &[(String, String, Vec<String>, Access)]) {
    for (scope, name, toks, _) in fields {
        // `const T m;` (a const member must be initialized in constructor lists)
        if toks.first().map(|s| s.as_str()) == Some("const") && !toks.iter().any(|t| t == "*" || t == "&" || t == "(" || t == "[" || t == "<") {
            if let Some(c) = db.classes.get_mut(scope) {
                for f in c.fields.iter_mut().filter(|f| &f.name == name && f.bitfield.is_none()) {
                    if !matches!(f.ty, Type::Const(_)) {
                        f.ty = Type::Const(Box::new(f.ty.clone()));
                    }
                }
            }
            continue;
        }
        // `const T *` / `T const *` with a single pointer level, no arrays or function pointers
        if toks.iter().filter(|t| *t == "*").count() != 1 || toks.last().map(|s| s.as_str()) != Some("*") || toks.iter().any(|t| t == "(" || t == "[" || t == "<") {
            continue;
        }
        let star = toks.len() - 1;
        if !toks[..star].iter().any(|t| t == "const") {
            continue;
        }
        if let Some(c) = db.classes.get_mut(scope) {
            for f in c.fields.iter_mut().filter(|f| &f.name == name) {
                let cv_outer = matches!(f.ty, Type::Const(_));
                let inner = match f.ty.unqualified() {
                    Type::Ptr(x) if !matches!(**x, Type::Const(_)) => Some((**x).clone()),
                    _ => None,
                };
                if let Some(x) = inner {
                    let p = Type::Ptr(Box::new(Type::Const(Box::new(x))));
                    f.ty = if cv_outer { Type::Const(Box::new(p)) } else { p };
                }
            }
        }
    }
}

/// Split template arguments at top-level commas.
fn split_top_level(s: &str) -> Vec<String> {
    let mut out = vec![];
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '<' | '(' => {
                depth += 1;
                cur.push(c);
            }
            '>' | ')' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => out.push(std::mem::take(&mut cur).trim().to_string()),
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Find the db class for an instantiation spelling, allowing defaulted trailing arguments.
pub fn normalize_instance(db: &TypeDb, name: &str) -> String {
    let name = name.replace(">>", "> >");
    if db.classes.contains_key(&name) || !name.ends_with('>') {
        return name;
    }
    let mut stem = name[..name.len() - 1].to_string();
    while stem.ends_with(' ') {
        stem.pop();
    }
    let prefix = format!("{stem}, ");
    db.classes
        .keys()
        .filter(|k| k.starts_with(&prefix))
        .min_by_key(|k| k.len())
        .cloned()
        .unwrap_or(name)
}

/// Replace whole identifiers `params[i]` by `args[i]` in a spelled name.
fn subst_text(s: &str, bind: &HashMap<String, (Type, String)>) -> String {
    let mut out = String::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let st = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let w = &s[st..i];
            let prev_colon = st >= 2 && &s[st - 2..st] == "::";
            match bind.get(w) {
                Some((_, sp)) if !prev_colon => out.push_str(sp),
                _ => out.push_str(w),
            }
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// Substitute template parameters in a type from a template declaration.
/// `injected` = (template base, instantiation) for the injected class name.
pub fn substitute(db: &TypeDb, t: &Type, bind: &HashMap<String, (Type, String)>, injected: Option<(&str, &str)>) -> Type {
    subst_rec(db, t, bind, injected, 0)
}

fn subst_rec(db: &TypeDb, t: &Type, bind: &HashMap<String, (Type, String)>, inj: Option<(&str, &str)>, depth: u32) -> Type {
    if depth > 16 {
        return t.clone();
    }
    let rec = |x: &Type| subst_rec(db, x, bind, inj, depth + 1);
    match t {
        Type::Named(n) => {
            if let Some((ty, _)) = bind.get(n) {
                return ty.clone();
            }
            // a member type of a template parameter: `T::const_iterator`
            if let Some((p, rest)) = n.split_once("::") {
                if let Some((Type::Named(a), _)) = bind.get(p) {
                    return Type::Named(format!("{a}::{rest}"));
                }
            }
            if let Some((base, inst)) = inj {
                if n == base {
                    return Type::Named(inst.to_string());
                }
            }
            if let Some(td) = db.template_typedefs.get(n) {
                // member typedef of the template being instantiated (or of another template)
                return rec(td);
            }
            if n.contains('<') {
                return Type::Named(normalize_instance(db, &subst_text(n, bind)));
            }
            t.clone()
        }
        Type::Ptr(x) => Type::Ptr(Box::new(rec(x))),
        Type::Ref(x) => Type::Ref(Box::new(rec(x))),
        Type::Const(x) => Type::Const(Box::new(rec(x))),
        Type::Volatile(x) => Type::Volatile(Box::new(rec(x))),
        Type::Array(x, n) => Type::Array(Box::new(rec(x)), *n),
        Type::FuncPtr(sig) => {
            let mut s = (**sig).clone();
            s.ret = rec(&s.ret);
            for p in &mut s.params {
                p.ty = rec(&p.ty);
            }
            Type::FuncPtr(Box::new(s))
        }
        _ => t.clone(),
    }
}

/// Does a type still mention an unbound template parameter (or unresolved template member)?
pub fn has_template_residue(t: &Type, params: &[String]) -> bool {
    match t {
        // `T`, `T::const_iterator`, `pair<T, int>`
        Type::Named(n) => params.iter().any(|p| {
            p == n || n.starts_with(&format!("{p}::")) || {
                let toks: Vec<&str> = n.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).collect();
                n.contains('<') && toks.iter().any(|t| t == p)
            }
        }),
        Type::Ptr(x) | Type::Ref(x) | Type::Const(x) | Type::Volatile(x) | Type::Array(x, _) => has_template_residue(x, params),
        Type::FuncPtr(s) => has_template_residue(&s.ret, params) || s.params.iter().any(|p| has_template_residue(&p.ty, params)),
        _ => false,
    }
}
