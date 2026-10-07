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
    // members of instantiated class templates (`rstl::vector<int, ...>::size`)
    for (cname, c) in &db.classes {
        let Some(lt) = cname.find('<') else { continue };
        if !cname.ends_with('>') || c.is_declaration {
            continue;
        }
        let base = &cname[..lt];
        let Some(tps) = db.templates.get(base) else { continue };
        let args: Vec<Type> = mwdec_lift::sig::split_top(&cname[lt + 1..cname.len() - 1], ',').iter().map(|a| mwdec_lift::sig::parse_type(a.trim())).collect();
        if args.len() != tps.len() {
            continue;
        }
        let prefix = format!("{base}::");
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
                let sub = |t: &Type| subst_class(t, tps, &args, base, cname);
                let mut d2 = d.clone();
                d2.params = d.params.iter().map(|p| Param { name: p.name.clone(), ty: sub(&p.ty) }).collect();
                d2.ret = sub(&d.ret);
                d2.template_params = vec![];
                let last = if member == mwdec_lift::sig::split_scope(base).1 { mwdec_lift::sig::split_scope(base).1.to_string() } else { member.clone() };
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
        {
            if !d.is_inline_defined || d.variadic || d.is_virtual {
                continue;
            }
            if d.inline_body.as_deref().map_or(false, crate::relevance::trivial_body) {
                continue;
            }
            // loops never fold into one value, and their instantiations are the ones that fail
            if d.inline_body.as_deref().map_or(false, |b| b.split_whitespace().any(|t| matches!(t, "for" | "while" | "do" | "goto"))) {
                continue;
            }
            // constructors of class template instances (containers reading streams etc.)
            if class.as_deref().map_or(false, |c| c.contains('<')) && class.as_deref().map(|c| mwdec_lift::sig::split_scope(c).1.split('<').next().unwrap_or("").to_string()).as_deref() == Some(last) {
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
            let is_ctor = class.as_deref().map_or(false, |c| mwdec_lift::sig::split_scope(c).1 == last);
            let tp: &[String] = &[];
            let Some(pspell) = d.params.iter().map(|p| spell(&p.ty, db, tp)).collect::<Option<Vec<_>>>() else { continue };
            let mut params = vec![];
            let mut decl_params = vec![];
            let kind;
            let call_args: Vec<String> = (0..d.params.len()).map(|i| format!("a{i}")).collect();
            let mut ret_ref = false;
            let ret: Type;
            let body: String;
            if is_ctor {
                let c = class.clone().unwrap();
                if d.params.is_empty() || mwdec_lift::sig::find_class(db, &c).map_or(true, |k| k.vptr_offset.is_some()) {
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
                    }
                }
            }
            let Some(rspell) = spell(&ret, db, tp) else { continue };
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
            out.push(Probe { name, decl: d.clone(), class: class.clone(), kind, params, ret, ret_ref, sig, line: 0, fn_template });
            // stash the text in the decl body slot for rendering
            out.last_mut().unwrap().decl.inline_body = Some(text);
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
            }],
        );
    }
}

