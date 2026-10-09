//! Probe generation: one tiny function per header inline that calls it with its parameters as
//! operands, compiled in the unit context with the unit flags. Compile errors (inaccessible or
//! unusable declarations) drop the offending probes and the TU is recompiled.

use mwdec_core::{DeclInfo, FuncSig, ObjectFile, Param, Type, TypeDb};
use std::collections::BTreeSet;

/// How the inline is called (and how a recognised occurrence is rendered back as IR).
#[derive(Clone, Debug)]
pub enum CallKind {
    /// Non-static member function: probe param 0 is the object pointer.
    Method,
    /// Static member / free function / free operator.
    Free,
    /// Constructor: the probe returns `C(args)` by value.
    Ctor,
}

#[derive(Clone, Debug)]
pub struct Probe {
    /// Probe function name (unmangled); the object symbol is `{name}__F...`.
    pub name: String,
    pub decl: DeclInfo,
    /// Class for members/constructors.
    pub class: Option<String>,
    pub kind: CallKind,
    /// Probe parameter types in order (object pointer first for methods).
    pub params: Vec<Type>,
    /// Probe return type (references become pointers).
    pub ret: Type,
    /// The inline returns a reference (probe returns its address).
    pub ret_ref: bool,
    /// Signature of the inline itself (for building calls).
    pub sig: FuncSig,
    pub line: usize,
    /// Instantiated from a function template with a guessed scalar type.
    pub fn_template: bool,
    /// A trivial accessor (`m = x;`) probed only for the dead stores of its by-value class
    /// parameter or return: its template needs them.
    pub needs_dead: bool,
    /// Constant-argument specialisation: (probe parameter index, constant) pairs the probe
    /// passes instead of the parameter (the compiler folds constant arguments into the body
    /// before optimising, so the expansion differs from the generic one).
    pub fixed: Vec<(usize, mwdec_lift::Expr)>,
}

/// C++ spelling of a type, or None when the probe can't name it.
pub fn spell(t: &Type, db: &TypeDb, tparams: &[String]) -> Option<String> {
    Some(match t {
        Type::Void => "void".into(),
        Type::Bool => "bool".into(),
        Type::Char => "char".into(),
        Type::WChar => "wchar_t".into(),
        Type::Long { signed: true } => "long".into(),
        Type::Long { signed: false } => "unsigned long".into(),
        Type::Int { size, signed } => {
            let base = match size {
                1 => "char",
                2 => "short",
                4 => "int",
                8 => "long long",
                _ => return None,
            };
            if *signed {
                if *size == 1 {
                    "signed char".into()
                } else {
                    base.into()
                }
            } else {
                format!("unsigned {base}")
            }
        }
        Type::Float { size: 4 } => "float".into(),
        Type::Float { size: 8 } => "double".into(),
        Type::Ptr(inner) => format!("{}*", spell(inner, db, tparams)?),
        Type::Ref(inner) => format!("{}&", spell(inner, db, tparams)?),
        Type::Const(inner) => match &**inner {
            Type::Ptr(_) => format!("{} const", spell(inner, db, tparams)?),
            _ => format!("const {}", spell(inner, db, tparams)?),
        },
        Type::Volatile(inner) => format!("volatile {}", spell(inner, db, tparams)?),
        Type::Named(n) => {
            if tparams.iter().any(|p| p == n) {
                return None;
            }
            let known = db.classes.contains_key(n) || db.enums.contains_key(n) || db.typedefs.contains_key(n) || mwdec_lift::sig::find_class(db, n).is_some();
            if !known {
                return None;
            }
            n.clone()
        }
        _ => return None,
    })
}

/// `operator+ =` (token-joined by the header scanner) -> `operator+=`.
pub fn norm_op_name(n: &str) -> String {
    if let Some(i) = n.find("operator") {
        let (a, b) = n.split_at(i + "operator".len());
        let b = b.trim();
        if b.chars().next().map_or(false, |c| c.is_alphabetic()) {
            // conversion / new / delete: keep one space
            return format!("{a} {}", b.split_whitespace().collect::<Vec<_>>().join(" "));
        }
        return format!("{a}{}", b.replace(' ', ""));
    }
    n.to_string()
}

fn is_class_type(t: &Type, db: &TypeDb) -> bool {
    let r = mwdec_lift::types::resolve(Some(db), t);
    mwdec_lift::types::class_of(Some(db), &r).is_some()
}

fn sig_of_decl(d: &DeclInfo, class: Option<&str>, qname: &str) -> FuncSig {
    FuncSig {
        qualified_name: qname.to_string(),
        mangled: None,
        ret: d.ret.clone(),
        params: d.params.clone(),
        this_class: if d.is_static { None } else { class.map(|c| c.to_string()) },
        is_const: d.is_const,
        is_static: d.is_static,
        is_virtual: d.is_virtual,
        variadic: d.variadic,
        runs_code: false,
    }
}

/// Declarations to probe: (qualified name, class, member name, declaration with concrete types).
fn work_list(db: &TypeDb) -> Vec<(String, Option<String>, String, DeclInfo, bool)> {
    let mut work = vec![];
    for (key, ds) in &db.decls {
        let qname = norm_op_name(key);
        let (scope, last) = mwdec_lift::sig::split_scope(&qname);
        // the scope is a class (not a namespace)?
        let class = scope.filter(|s| mwdec_lift::sig::find_class(db, s).is_some()).map(|s| s.to_string());
        for d in ds {
            if d.template_params.is_empty() {
                work.push((qname.clone(), class.clone(), last.to_string(), d.clone(), false));
                continue;
            }
            if class.as_deref().map_or(false, |c| db.templates.contains_key(c)) || scope.map_or(false, |s| db.templates.contains_key(s)) {
                continue;
            }
            // function templates: instantiate with the common scalar types (every template
            // parameter must be deducible from the arguments)
            let mentions = |t: &Type, n: &str| format!("{t:?}").contains(&format!("Named(\"{n}\")"));
            if !d.template_params.iter().all(|tp| d.params.iter().any(|p| mentions(&p.ty, tp))) {
                continue;
            }
            // iterator algorithms can't be instantiated with scalars
            if d.template_params.iter().any(|tp| tp.starts_with("It") || tp.starts_with("Iter") || tp.ends_with("It")) {
                continue;
            }
            for ty in [Type::Float { size: 4 }, Type::Int { size: 4, signed: true }, Type::Int { size: 4, signed: false }] {
                let mut d2 = d.clone();
                let sub = |t: &Type| subst_tparams(t, &d.template_params, &ty);
                d2.params = d.params.iter().map(|p| Param { name: p.name.clone(), ty: sub(&p.ty) }).collect();
                d2.ret = sub(&d.ret);
                d2.template_params = vec![];
                work.push((qname.clone(), class.clone(), last.to_string(), d2, true));
            }
        }
    }
    // members of instantiated class templates (`rstl::vector<int, ...>::size`) and of classes
    // nested in them (`rstl::red_black_tree<...>::const_iterator::operator==`)
    for (cname, c) in &db.classes {
        if !cname.contains('<') || c.is_declaration {
            continue;
        }
        let parts = mwdec_ctx::mangle::split_scope(cname);
        let Some(ti) = parts.iter().rposition(|p| p.contains('<')) else { continue };
        let tpart = parts[ti];
        let Some(lt) = tpart.find('<') else { continue };
        if !tpart.ends_with('>') {
            continue;
        }
        let mut bparts: Vec<&str> = parts[..ti].to_vec();
        bparts.push(&tpart[..lt]);
        let base = bparts.join("::");
        let inst = parts[..=ti].join("::");
        let nested: Vec<&str> = parts[ti + 1..].to_vec();
        let Some(tps) = db.templates.get(&base) else { continue };
        let arg_text = mwdec_lift::sig::split_top(&tpart[lt + 1..tpart.len() - 1], ',');
        let args: Vec<Type> = arg_text.iter().map(|a| mwdec_lift::sig::parse_type(a.trim())).collect();
        if args.len() != tps.len() {
            continue;
        }
        let bind: std::collections::HashMap<String, (Type, String)> = tps.iter().cloned().zip(args.iter().cloned().zip(arg_text.iter().map(|a| a.trim().to_string()))).collect();
        let decl_scope = if nested.is_empty() { base.clone() } else { format!("{base}::{}", nested.join("::")) };
        let own_name = nested.last().copied().unwrap_or(mwdec_lift::sig::split_scope(&base).1);
        let prefix = format!("{decl_scope}::");
        for (key, ds) in db.decls.range(prefix.clone()..) {
            if !key.starts_with(&prefix) {
                break;
            }
            let member = norm_op_name(&key[prefix.len()..]);
            if member.contains("::") {
                continue;
            }
            for d in ds {
                if d.template_params != *tps {
                    continue;
                }
                let sub = |t: &Type| {
                    let t = if nested.is_empty() { subst_class(t, tps, &args, &base, cname) } else { mwdec_ctx::resolve::substitute(db, t, &bind, Some((&base, &inst))) };
                    let t = mwdec_ctx::resolve::substitute(db, &t, &bind, Some((&base, &inst)));
                    // member types named unqualified (`const_iterator end()`, `rstl::vector::iterator`)
                    let t = mwdec_ctx::qualify_nested(db, &t, cname);
                    if nested.is_empty() {
                        t
                    } else {
                        mwdec_ctx::qualify_nested(db, &t, &inst)
                    }
                };
                let mut d2 = d.clone();
                d2.params = d.params.iter().map(|p| Param { name: p.name.clone(), ty: sub(&p.ty) }).collect();
                d2.ret = sub(&d.ret);
                d2.template_params = vec![];
                let last = if member == own_name { own_name.to_string() } else { member.clone() };
                work.push((format!("{cname}::{member}"), Some(cname.clone()), last, d2, false));
            }
        }
    }
    work
}

fn subst_class(t: &Type, tps: &[String], args: &[Type], base: &str, cname: &str) -> Type {
    match t {
        Type::Named(n) => {
            if let Some(i) = tps.iter().position(|p| p == n) {
                return args[i].clone();
            }
            if n == base {
                return Type::Named(cname.to_string());
            }
            t.clone()
        }
        Type::Ptr(x) => Type::Ptr(Box::new(subst_class(x, tps, args, base, cname))),
        Type::Ref(x) => Type::Ref(Box::new(subst_class(x, tps, args, base, cname))),
        Type::Const(x) => Type::Const(Box::new(subst_class(x, tps, args, base, cname))),
        Type::Volatile(x) => Type::Volatile(Box::new(subst_class(x, tps, args, base, cname))),
        t => t.clone(),
    }
}

/// Generate probes for the inline declarations of the context.
pub fn generate(db: &TypeDb, rel: Option<&std::collections::HashSet<String>>) -> Vec<Probe> {
    let mut out = vec![];
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (qname, class, last, d, fn_template) in work_list(db) {
        let last = last.as_str();
        let d = &d;
        if std::env::var("MWDI_TRACE_PROBE").is_ok_and(|f| qname.contains(f.as_str())) {
            eprintln!("PROBE {qname} class {class:?} params {:?} inline {} body {:?}", d.params.iter().map(|p| &p.ty).collect::<Vec<_>>(), d.is_inline_defined, d.inline_body);
        }
        {
            if !d.is_inline_defined || d.variadic || d.is_virtual {
                continue;
            }
            // trivial accessors only with a by-value class parameter or return (their copy
            // leaves a dead frame store that tells the call apart from a plain member access)
            let by_value_class = |t: &Type| !matches!(strip_cv(t), Type::Ptr(_) | Type::Ref(_)) && is_class_type(t, db);
            let needs_dead = d.inline_body.as_deref().map_or(false, crate::relevance::trivial_body);
            if needs_dead && !(d.params.iter().any(|p| by_value_class(&p.ty)) || by_value_class(&d.ret)) {
                continue;
            }
            // loops never fold into one value, and their instantiations are the ones that fail
            // (a constructor's loop is a statement template building its object in place:
            // `basic_string(literal_t, const char*)` counting the length)
            let ctor_like = class.as_deref().is_some_and(|c| {
                let own = mwdec_lift::sig::split_scope(c).1;
                own == last || own.split('<').next() == Some(last)
            });
            let loop_ok = ctor_like && std::env::var_os("MWDI_NO_LOOP_CTORS").is_none();
            if d.inline_body.as_deref().map_or(false, |b| b.split_whitespace().any(|t| matches!(t, "for" | "while" | "do" | "goto") && !(loop_ok && t != "goto"))) {
                continue;
            }
            // constructors of class template instances with stream or count parameters
            // (containers reading streams, sized vectors: loops and allocations)
            if class.as_deref().map_or(false, |c| c.contains('<'))
                && class.as_deref().map(|c| mwdec_lift::sig::split_scope(c).1.split('<').next().unwrap_or("").to_string()).as_deref() == Some(last)
                && d.params.iter().any(|p| matches!(strip_cv(&p.ty), Type::Int { .. } | Type::Long { .. }) || format!("{:?}", p.ty).contains("CInputStream"))
            {
                continue;
            }
            if let Some(rel) = rel {
                if !crate::relevance::decl_relevant(if d.is_static { None } else { class.as_deref() }, &d.params, &d.ret, db, rel) {
                    continue;
                }
            }
            if d.access != mwdec_core::Access::Public {
                continue;
            }
            if last.starts_with('~') || last.starts_with("operator new") || last.starts_with("operator delete") {
                continue;
            }
            if let Some(c) = &class {
                if mwdec_lift::sig::find_class(db, c).map_or(true, |k| k.is_declaration) {
                    continue;
                }
            }
            let is_ctor = class.as_deref().map_or(false, |c| {
                let own = mwdec_lift::sig::split_scope(c).1;
                own == last || own.split('<').next() == Some(last)
            });
            let tp: &[String] = &[];
            let skips = std::env::var("MWDI_SKIPS").is_ok();
            let Some(pspell) = d.params.iter().map(|p| spell(&p.ty, db, tp)).collect::<Option<Vec<_>>>() else {
                if skips {
                    eprintln!("SKIP {qname}: params {:?}", d.params.iter().map(|p| &p.ty).collect::<Vec<_>>());
                }
                continue;
            };
            let mut params = vec![];
            let mut decl_params = vec![];
            let kind;
            let call_args: Vec<String> = (0..d.params.len()).map(|i| format!("a{i}")).collect();
            let mut ret_ref = false;
            let ret: Type;
            let body: String;
            // a non-const method's value can be discarded: its expansion without the result
            // (the compiler drops the result's computation) is probed too
            let mut discard: Option<String> = None;
            if is_ctor {
                let c = class.clone().unwrap();
                // default constructors only for class template instances (`optional_object<T>()`:
                // an empty optional); elsewhere zero stores are too common to name
                if (d.params.is_empty() && !c.contains('<')) || mwdec_lift::sig::find_class(db, &c).map_or(true, |k| k.vptr_offset.is_some()) {
                    continue;
                }
                kind = CallKind::Ctor;
                ret = Type::Named(c.clone());
                body = format!("return {c}({});", call_args.join(", "));
            } else {
                let callee = if let (Some(c), false) = (&class, d.is_static) {
                    let self_t = if d.is_const { Type::Ptr(Box::new(Type::Const(Box::new(Type::Named(c.clone()))))) } else { Type::Ptr(Box::new(Type::Named(c.clone()))) };
                    params.push(self_t);
                    kind = CallKind::Method;
                    format!("self->{last}")
                } else {
                    kind = CallKind::Free;
                    qname.clone()
                };
                let call = format!("{callee}({})", call_args.join(", "));
                match strip_cv(&d.ret) {
                    Type::Void => {
                        ret = Type::Void;
                        body = format!("{call};");
                    }
                    Type::Ref(inner) => {
                        ret_ref = true;
                        ret = Type::Ptr(inner.clone());
                        body = format!("return &{call};");
                    }
                    Type::Unknown { .. } => continue,
                    _ => {
                        ret = d.ret.clone();
                        body = format!("return {call};");
                        if matches!(kind, CallKind::Method) && !d.is_const {
                            discard = Some(call.clone());
                        }
                    }
                }
            }
            let Some(rspell) = spell(&ret, db, tp) else {
                if skips {
                    eprintln!("SKIP {qname}: ret {ret:?}");
                }
                continue;
            };
            for (i, p) in d.params.iter().enumerate() {
                params.push(p.ty.clone());
                decl_params.push(format!("{} a{i}", pspell[i]));
            }
            let name = format!("__mwdi_p{}", out.len());
            let mut all_params = vec![];
            if matches!(kind, CallKind::Method) {
                let st = spell(&params[0], db, tp).unwrap();
                all_params.push(format!("{st} self"));
            }
            all_params.extend(decl_params);
            let text = format!("{rspell} {name}({}) {{ {body} }}", all_params.join(", "));
            if !seen.insert(text.replace(&name, "")) {
                continue;
            }
            let sig = sig_of_decl(d, class.as_deref(), &qname);
            let _ = is_class_type;
            out.push(Probe { name, decl: d.clone(), class: class.clone(), kind: kind.clone(), params: params.clone(), ret: ret.clone(), ret_ref, sig: sig.clone(), line: 0, fn_template, needs_dead, fixed: vec![] });
            // stash the text in the decl body slot for rendering
            out.last_mut().unwrap().decl.inline_body = Some(text);
            // constant-argument specialisations (see `specialisations`)
            // (constructors only on request: their folds into differently typed stack objects
            // render casts that don't compile)
            if std::env::var("MWDI_NO_CONSTSPEC").is_err() && (!matches!(kind, CallKind::Ctor) || std::env::var("MWDI_CONSTSPEC_CTOR").is_ok()) {
                let off = if matches!(kind, CallKind::Method) { 1 } else { 0 };
                for spec in specialisations(d, db) {
                    let mut args = call_args.clone();
                    let mut fixed = vec![];
                    for (i, lit, e) in &spec {
                        args[*i] = lit.clone();
                        fixed.push((*i + off, e.clone()));
                    }
                    let name = format!("__mwdi_p{}", out.len());
                    let callee = if matches!(kind, CallKind::Method) { format!("self->{last}") } else if matches!(kind, CallKind::Ctor) { class.clone().unwrap_or_default() } else { qname.clone() };
                    let call = format!("{callee}({})", args.join(", "));
                    let body = if matches!(kind, CallKind::Ctor) { format!("return {call};") } else if ret_ref { format!("return &{call};") } else if matches!(strip_cv(&d.ret), Type::Void) { format!("{call};") } else { format!("return {call};") };
                    let text = format!("{rspell} {name}({}) {{ {body} }}", all_params.join(", "));
                    if !seen.insert(text.replace(&name, "")) {
                        continue;
                    }
                    out.push(Probe { name, decl: d.clone(), class: class.clone(), kind: kind.clone(), params: params.clone(), ret: ret.clone(), ret_ref, sig: sig.clone(), line: 0, fn_template, needs_dead, fixed });
                    out.last_mut().unwrap().decl.inline_body = Some(text);
                }
            }
            if let Some(call) = discard {
                let name = format!("__mwdi_p{}", out.len());
                let text = format!("void {name}({}) {{ {call}; }}", all_params.join(", "));
                if seen.insert(text.replace(&name, "")) {
                    out.push(Probe { name, decl: d.clone(), class: class.clone(), kind, params, ret: Type::Void, ret_ref: false, sig, line: 0, fn_template, needs_dead: false, fixed: vec![] });
                    out.last_mut().unwrap().decl.inline_body = Some(text);
                }
            }
        }
    }
    out
}

/// Constant-argument specialisations worth probing for one inline: each is a list of
/// (parameter index, literal text, literal expression). MWCC substitutes a constant argument
/// into the body and folds before optimising, so `f(x, true)` loses the branch on the flag and
/// `clamp(x, 0, 1)` folds compares: the generic template (constant in a hole) can't match. Only
/// parameters whose constant changes code are specialised: `bool` parameters (true/false), and
/// integer / enum parameters with the literals they are compared or switched on.
pub fn specialisations(d: &DeclInfo, db: &TypeDb) -> Vec<Vec<(usize, String, mwdec_lift::Expr)>> {
    use mwdec_lift::Expr;
    let Some(body) = d.inline_body.as_deref() else { return vec![] };
    let toks: Vec<&str> = body.split_whitespace().collect();
    let words: Vec<&str> = body.split(|c: char| !(c.is_alphanumeric() || c == '_')).filter(|t| !t.is_empty()).collect();
    // tokens inside conditions: `if ( .. )`, `while ( .. )`, `switch ( .. )`, and a ternary's
    // condition (from the statement start to `?`)
    let mut in_cond = vec![false; toks.len()];
    for k in 0..toks.len() {
        if matches!(toks[k], "if" | "while" | "switch") && toks.get(k + 1) == Some(&"(") {
            let mut d = 0;
            for j in k + 1..toks.len() {
                match toks[j] {
                    "(" => d += 1,
                    ")" => {
                        d -= 1;
                        if d == 0 {
                            break;
                        }
                    }
                    _ => in_cond[j] = true,
                }
            }
        }
        if toks[k] == "?" {
            let mut j = k;
            while j > 0 && !matches!(toks[j - 1], ";" | "{" | "}" | "return" | "=" | ":") {
                j -= 1;
                in_cond[j] = true;
            }
        }
    }
    let conditioned = |n: &str| toks.iter().zip(&in_cond).any(|(t, c)| *c && *t == n);
    // literals a parameter is compared with or switched on: `n == K`, `K != n`, `switch ( n ) { case K :`
    let compared = |n: &str| -> Vec<String> {
        let mut out: Vec<String> = vec![];
        for k in 0..toks.len() {
            if toks[k] != n {
                continue;
            }
            for (a, b) in [(k + 1, k + 2), (k.wrapping_sub(1), k.wrapping_sub(2))] {
                if matches!(toks.get(a), Some(&"==") | Some(&"!=")) {
                    if let Some(v) = toks.get(b) {
                        let v = if *v == "-" { toks.get(if b > k { b + 1 } else { b.wrapping_sub(1) }).map(|x| format!("-{x}")).unwrap_or_default() } else { v.to_string() };
                        out.push(v);
                    }
                }
            }
            // switch ( n ) { case K : ... }
            if k >= 2 && toks[k - 1] == "(" && toks[k - 2] == "switch" {
                let mut depth = 0;
                for t in &toks[k + 1..] {
                    match *t {
                        "{" => depth += 1,
                        "}" => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let mut it = toks[k + 1..].iter().peekable();
                let mut d = 0;
                while let Some(t) = it.next() {
                    match *t {
                        "{" => d += 1,
                        "}" => {
                            d -= 1;
                            if d == 0 {
                                break;
                            }
                        }
                        "case" if d == 1 => {
                            if let Some(v) = it.next() {
                                let v = if *v == "-" { it.next().map(|x| format!("-{x}")).unwrap_or_default() } else { v.to_string() };
                                out.push(v);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        out.retain(|v| !v.is_empty());
        out.dedup();
        out
    };
    let all = std::env::var("MWDI_CONSTSPEC_ALL").is_ok();
    let mut per: Vec<Vec<(usize, String, Expr)>> = vec![];
    for (i, p) in d.params.iter().enumerate() {
        let Some(n) = p.name.as_deref() else { continue };
        if !words.iter().any(|t| *t == n) {
            continue;
        }
        let pt = strip_cv(&p.ty);
        let int_lit = |v: &str, t: &Type| -> Option<(String, Expr)> {
            let x = if let Some(h) = v.strip_prefix("0x") { i64::from_str_radix(h.trim_end_matches(['u', 'U']), 16).ok()? } else { v.trim_end_matches(['u', 'U']).parse::<i64>().ok()? };
            Some((v.to_string(), Expr::Int { value: x, ty: t.clone() }))
        };
        let mut vals: Vec<(String, Expr)> = match pt {
            Type::Bool => vec![("true".into(), Expr::Int { value: 1, ty: Type::Bool }), ("false".into(), Expr::Int { value: 0, ty: Type::Bool })],
            // (the literals it is tested against; other constants in conditions only with
            // `MWDI_CONSTSPEC_ALL`)
            t @ (Type::Int { .. } | Type::Long { .. } | Type::Char) => {
                let mut v: Vec<(String, Expr)> = compared(n).iter().filter_map(|x| int_lit(x, t)).collect();
                if conditioned(n) && all {
                    for x in ["0", "1"] {
                        if !v.iter().any(|(l, _)| l == x) {
                            v.extend(int_lit(x, t));
                        }
                    }
                }
                v
            }
            Type::Float { size } if conditioned(n) && all => {
                let f = |x: f64| Expr::Float { bits: if *size == 4 { (x as f32).to_bits() as u64 } else { x.to_bits() }, double: *size == 8 };
                vec![("0.0f".into(), f(0.0)), ("1.0f".into(), f(1.0))]
            }
            Type::Named(en) if db.enums.contains_key(en.as_str()) => {
                let e = &db.enums[en.as_str()];
                let mut lbls = compared(n);
                // the other side of a two-way test (`normalize == kN_Yes`): another enumerator
                if lbls.len() == 1 {
                    let short = lbls[0].rsplit("::").next().unwrap_or("").to_string();
                    if let Some((nm, _)) = e.values.iter().find(|(nm, _)| nm.rsplit("::").next() != Some(short.as_str())) {
                        lbls.push(nm.clone());
                    }
                }
                lbls
                    .iter()
                    .filter_map(|lbl| {
                        let short = lbl.rsplit("::").next().unwrap_or(lbl);
                        let (_, v) = e.values.iter().find(|(nm, _)| nm == short || nm.rsplit("::").next() == Some(short))?;
                        Some((format!("({en}){v}"), Expr::Cast { ty: pt.clone(), e: Box::new(Expr::Int { value: *v, ty: Type::Int { size: 4, signed: true } }) }))
                    })
                    .collect()
            }
            _ => vec![],
        };
        vals.truncate(8);
        if !vals.is_empty() {
            per.push(vals.into_iter().map(|(l, e)| (i, l, e)).collect());
        }
    }
    // one parameter at a time (at most 3 parameters), plus every combination of two bools
    let mut out = vec![];
    for vals in per.iter().take(3) {
        for v in vals {
            out.push(vec![v.clone()]);
        }
    }
    let bools: Vec<&Vec<(usize, String, Expr)>> = per.iter().filter(|v| matches!(v[0].2, Expr::Int { ty: Type::Bool, .. })).collect();
    if bools.len() == 2 {
        for a in bools[0] {
            for b in bools[1] {
                out.push(vec![a.clone(), b.clone()]);
            }
        }
    }
    out
}

fn subst_tparams(t: &Type, tps: &[String], with: &Type) -> Type {
    match t {
        Type::Named(n) if tps.iter().any(|p| p == n) => with.clone(),
        Type::Ptr(x) => Type::Ptr(Box::new(subst_tparams(x, tps, with))),
        Type::Ref(x) => Type::Ref(Box::new(subst_tparams(x, tps, with))),
        Type::Const(x) => Type::Const(Box::new(subst_tparams(x, tps, with))),
        Type::Volatile(x) => Type::Volatile(Box::new(subst_tparams(x, tps, with))),
        t => t.clone(),
    }
}

fn strip_cv(t: &Type) -> &Type {
    match t {
        Type::Const(t) | Type::Volatile(t) => strip_cv(t),
        t => t,
    }
}

/// Source text of a probe TU (one probe per line); sets `line` (1-based).
pub fn render(probes: &mut [Probe], keep: &[bool]) -> String {
    let mut s = String::new();
    let mut line = 1;
    for (p, k) in probes.iter_mut().zip(keep) {
        if !*k {
            continue;
        }
        p.line = line;
        s.push_str(p.decl.inline_body.as_deref().unwrap_or(""));
        s.push('\n');
        line += 1;
    }
    s
}

/// Line numbers (in the candidate file) of compiler errors.
pub fn error_lines(messages: &str) -> Vec<usize> {
    let mut out = vec![];
    let mut cur_file_is_tu = false;
    for l in messages.lines() {
        let t = l.trim_start_matches('#').trim();
        if let Some(f) = t.strip_prefix("File:").or_else(|| t.strip_prefix("In:")) {
            let f = f.trim().replace('\\', "/");
            cur_file_is_tu = f.contains("/tmp/tu_");
            continue;
        }
        if cur_file_is_tu {
            if let Some((n, _)) = t.split_once(':') {
                if let Ok(n) = n.trim().parse::<usize>() {
                    out.push(n);
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Names of templates being instantiated when an error happened (`(instantiating: 'f<T>(...)')`).
fn instantiating(messages: &str) -> Vec<String> {
    let mut out = vec![];
    // undo the compiler's line wrapping
    let flat: String = messages.lines().map(|l| l.trim_start_matches('#').trim()).collect::<Vec<_>>().join(" ");
    for part in flat.split("(instantiating: '").skip(1) {
        // name up to the parameter list's '(' (outside template brackets)
        let mut depth = 0;
        let mut name = String::new();
        let mut it = part.chars().peekable();
        while let Some(c) = it.next() {
            match c {
                '<' => depth += 1,
                '>' => depth -= 1,
                '(' if depth == 0 && name.ends_with("operator") && it.peek() == Some(&')') => {
                    it.next();
                    name.push_str("()");
                    continue;
                }
                '(' | '\'' if depth == 0 => break,
                _ => {}
            }
            name.push(c);
        }
        let full: String = name.chars().filter(|c| !c.is_whitespace()).collect();
        // a function template instance: also its name without the trailing arguments
        if full.ends_with('>') {
            let mut d = 0;
            for (k, c) in full.char_indices().rev() {
                match c {
                    '>' => d += 1,
                    '<' => {
                        d -= 1;
                        if d == 0 {
                            out.push(full[..k].to_string());
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
        out.push(full);
    }
    out
}

/// Compile one set of probes; failing probes are dropped (by error line or instantiation name,
/// else by bisection). Returns compiled chunks.
/// Key shared by the instantiations of one member of a class template (`rstl::vector::PutTo`),
/// or the function itself.
fn poison_key(p: &Probe) -> String {
    let (scope, last) = mwdec_lift::sig::split_scope(&p.sig.qualified_name);
    match scope {
        Some(s) => format!("{}::{last}", s.split('<').next().unwrap_or(s)),
        None => p.sig.qualified_name.split('<').next().unwrap_or("").to_string(),
    }
}

fn compile_set(
    probes: Vec<Probe>,
    compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync),
    depth: u32,
    poison: &std::sync::Mutex<BTreeSet<String>>,
    learn: bool,
    out: &mut Vec<(ObjectFile, Vec<Probe>)>,
) {
    let mut probes = probes;
    for _round in 0..12 {
        probes.retain(|p| !poison.lock().unwrap().contains(&poison_key(p)));
        if probes.is_empty() {
            return;
        }
        let keep = vec![true; probes.len()];
        let src = render(&mut probes, &keep);
        match compile(&src) {
            Ok(o) => {
                out.push((o, probes));
                return;
            }
            Err(msg) => {
                let lines = error_lines(&msg);
                let names = instantiating(&msg);
                let before = probes.len();
                let nospace = |q: &str| q.chars().filter(|c| !c.is_whitespace()).collect::<String>();
                probes.retain(|p| !lines.contains(&p.line) && !names.iter().any(|n| !n.is_empty() && *n == nospace(&p.sig.qualified_name)));
                if probes.len() == before {
                    if probes.len() == 1 {
                        // the culprit: its siblings (other instantiations) will fail too
                        if learn {
                            poison.lock().unwrap().insert(poison_key(&probes[0]));
                        }
                        return;
                    }
                    if depth > 14 {
                        return;
                    }
                    let second = probes.split_off(probes.len() / 2);
                    compile_set(probes, compile, depth + 1, poison, learn, out);
                    compile_set(second, compile, depth + 1, poison, learn, out);
                    return;
                }
            }
        }
    }
}

/// Compile the probes in chunks (concurrently), dropping failing ones.
pub fn compile(probes: Vec<Probe>, compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync)) -> Vec<(ObjectFile, Vec<Probe>)> {
    let out = std::sync::Mutex::new(vec![]);
    let poison = std::sync::Mutex::new(BTreeSet::new());
    // one instantiation of every class-template member first: failures poison the rest
    let mut firsts = vec![];
    let mut rest = vec![];
    let mut seen = BTreeSet::new();
    for p in probes {
        if seen.insert(poison_key(&p)) {
            firsts.push(p);
        } else {
            rest.push(p);
        }
    }
    for (phase, list) in [firsts, rest].into_iter().enumerate() {
        let n = list.len();
        let nchunks = n.div_ceil(250).clamp(1, 4).min(n.max(1));
        let size = n.div_ceil(nchunks.max(1)).max(1);
        let mut chunks: Vec<Vec<Probe>> = vec![];
        let mut it = list.into_iter().peekable();
        while it.peek().is_some() {
            chunks.push(it.by_ref().take(size).collect());
        }
        std::thread::scope(|sc| {
            for c in chunks {
                let (poison, out) = (&poison, &out);
                sc.spawn(move || {
                    let mut local = vec![];
                    // only the first phase (one instance per member) learns which members
                    // fail; its chunks are disjoint in keys, so the result is deterministic
                    compile_set(c, compile, 0, poison, phase == 0, &mut local);
                    out.lock().unwrap().extend(local);
                });
            }
        });
    }
    let mut out = out.into_inner().unwrap();
    // deterministic order (probe names are numbered)
    out.sort_by_key(|(_, ps)| ps.first().map(|p| p.name[8..].parse::<usize>().unwrap_or(0)).unwrap_or(0));
    out
}

/// Declarations of the probe functions so the lifter knows their return types.
pub fn inject_decls(db: &mut TypeDb, probes: &[Probe]) {
    for p in probes {
        let params: Vec<Param> = p.params.iter().map(|t| Param { name: None, ty: t.clone() }).collect();
        db.decls.insert(
            p.name.clone(),
            vec![DeclInfo {
                qualified_name: p.name.clone(),
                ret: p.ret.clone(),
                params,
                is_const: false,
                is_static: false,
                is_virtual: false,
                is_pure: false,
                is_inline_defined: false,
                variadic: false,
                inline_body: None,
                template_params: vec![],
                access: mwdec_core::Access::Public,
                init_list: None,
                order: 0,
                defaults: vec![],
            }],
        );
    }
}

