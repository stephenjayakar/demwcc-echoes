//! Complete the layouts of class template instances the context only declares.
//!
//! The context's DWARF describes a class template instance (or a class nested in one, like
//! `rstl::red_black_tree<...>::const_iterator` or `rstl::pointer_iterator<T, vector<T>, A>`)
//! only when the context itself instantiated it; types that merely appear in signatures stay
//! forward declarations without members. Header inlines on them (iterator compares, `end()`,
//! `operator*`) can then be neither probed nor matched. Here every such class is instantiated
//! by a dummy function taking it by value, compiled with `-g` in the unit context, and the
//! layouts the compiler reports are merged into the unit's TypeDb.

use mwdec_core::{ObjectFile, TypeDb};

/// At most this many classes per context (one line each in the instantiation TU).
const MAX_CLASSES: usize = 400;

/// Template-instance classes (or classes nested in one) without a layout.
pub fn incomplete_instances(db: &TypeDb) -> Vec<String> {
    let mut out: Vec<String> = db
        .classes
        .iter()
        .filter(|(n, c)| c.is_declaration && c.fields.is_empty() && n.contains('<'))
        .map(|(n, _)| n.clone())
        // names the instantiation TU can't spell
        .filter(|n| !n.contains('(') && !n.contains('@') && !n.contains('$'))
        .collect();
    // member typedefs of complete instances naming instances the context never mentioned
    // (`rstl::vector<X>::iterator` = `rstl::pointer_iterator<X, rstl::vector<X>, A>`)
    for (cname, c) in &db.classes {
        if c.is_declaration || !cname.ends_with('>') {
            continue;
        }
        let Some(lt) = cname.find('<') else { continue };
        let base = &cname[..lt];
        if base.contains("::") && mwdec_lift::sig::find_class(db, mwdec_lift::sig::split_scope(base).0.unwrap_or("")).is_some() {
            continue;
        }
        let Some(tps) = db.templates.get(base) else { continue };
        let arg_text = mwdec_lift::sig::split_top(&cname[lt + 1..cname.len() - 1], ',');
        if arg_text.len() != tps.len() {
            continue;
        }
        let bind: std::collections::HashMap<String, (mwdec_core::Type, String)> =
            tps.iter().cloned().zip(arg_text.iter().map(|a| (mwdec_lift::sig::parse_type(a.trim()), a.trim().to_string()))).collect();
        let prefix = format!("{base}::");
        for (k, t) in db.template_typedefs.range(prefix.clone()..) {
            if !k.starts_with(&prefix) {
                break;
            }
            if let mwdec_core::Type::Named(n) = mwdec_ctx::resolve::substitute(db, t, &bind, Some((base, cname))) {
                if n.contains('<') && !db.classes.contains_key(&n) && !out.contains(&n) && !n.contains('(') && !bare_template(&n, db) {
                    out.push(n);
                }
            }
        }
    }
    // instances only named as template arguments (`rstl::pair<E, int>` of
    // `rstl::reserved_vector<rstl::pair<E, int>, 8>`): the elements pushed into containers
    let mut args: Vec<String> = vec![];
    for cname in db.classes.keys() {
        if cname.contains('<') {
            instance_args(cname, &mut args);
        }
    }
    for n in args {
        let base = &n[..n.find('<').unwrap_or(n.len())];
        if db.templates.contains_key(base) && !db.classes.contains_key(&n) && !out.contains(&n) && !n.contains('(') && !n.contains('@') && !n.contains('$') && !bare_template(&n, db) {
            out.push(n);
        }
    }
    out.truncate(MAX_CLASSES);
    out
}

/// The type arguments of class template instance `name` held by value (`T` of
/// `rstl::vector<T, A>`; not pointers or references).
pub(crate) fn value_args(name: &str) -> Vec<String> {
    let (Some(lt), true) = (name.find('<'), name.ends_with('>')) else { return vec![] };
    mwdec_lift::sig::split_top(&name[lt + 1..name.len() - 1], ',')
        .into_iter()
        .map(|a| a.trim().strip_prefix("const ").unwrap_or(a.trim()).to_string())
        .filter(|a| !a.contains('*') && !a.contains('&'))
        .collect()
}

/// Template-instance spellings among the template arguments of `name`, at any depth.
fn instance_args(name: &str, out: &mut Vec<String>) {
    for a in value_args(name) {
        if a.contains('<') && a.ends_with('>') {
            if !out.contains(&a) {
                out.push(a.clone());
            }
            instance_args(&a, out);
        }
    }
}

/// Does the spelling still name a class template without arguments (`rstl::basic_string`,
/// `rstl::map::value_type`: an unresolved member typedef)?
fn bare_template(name: &str, db: &TypeDb) -> bool {
    let b = name.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let st = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || (b[i] == b':' && i + 1 < b.len() && b[i + 1] == b':')) {
                i += if b[i] == b':' { 2 } else { 1 };
            }
            let path = &name[st..i];
            let next_lt = b.get(i) == Some(&b'<');
            if db.templates.contains_key(path) && !next_lt {
                return true;
            }
            // a scope prefix that is a template without arguments
            let mut p = path;
            while let Some(k) = p.rfind("::") {
                p = &p[..k];
                if db.templates.contains_key(p) {
                    return true;
                }
            }
        } else {
            i += 1;
        }
    }
    false
}

/// Instantiate the incomplete template-instance classes of `db` and merge their layouts.
/// `compile` compiles a TU body in the unit context (used to drop lines that don't compile
/// before the `-g` build). Returns the number of classes completed.
pub fn complete_instances(db: &mut TypeDb, context: &str, cflags: &[String], compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync)) -> usize {
    let names = incomplete_instances(db);
    if names.is_empty() {
        return 0;
    }
    let t0 = std::time::Instant::now();
    let mut keep = vec![true; names.len()];
    let render = |keep: &[bool]| -> String {
        let mut s = String::new();
        for (i, (n, k)) in names.iter().zip(keep).enumerate() {
            if *k {
                s.push_str(&format!("void __mwdi_inst{i}({n} a) {{ (void)&a; }}"));
            }
            s.push('\n');
        }
        s
    };
    // drop lines the compiler points at; errors elsewhere (inside an instantiated header
    // template) are narrowed down by bisection
    fn settle(set: Vec<usize>, keep: &mut [bool], render: &dyn Fn(&[bool]) -> String, compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync), depth: u32) {
        let mut set = set;
        for _ in 0..24 {
            if set.is_empty() {
                return;
            }
            let mut k = vec![false; keep.len()];
            for &i in &set {
                k[i] = true;
            }
            match compile(&render(&k)) {
                Ok(_) => {
                    for &i in &set {
                        keep[i] = true;
                    }
                    return;
                }
                Err(msg) => {
                    let lines = crate::probe::error_lines(&msg);
                    if std::env::var("MWDI_TRACE_COMPLETE").is_ok() {
                        eprintln!("complete: {} lines, error lines {lines:?}: {}", set.len(), msg.chars().take(600).collect::<String>());
                    }
                    let before = set.len();
                    set.retain(|i| !lines.contains(&(i + 1)));
                    if set.len() == before {
                        if set.len() == 1 || depth > 10 {
                            return;
                        }
                        let second = set.split_off(set.len() / 2);
                        settle(set, keep, render, compile, depth + 1);
                        settle(second, keep, render, compile, depth + 1);
                        return;
                    }
                }
            }
        }
    }
    let all: Vec<usize> = (0..names.len()).collect();
    keep.iter_mut().for_each(|k| *k = false);
    settle(all, &mut keep, &render, compile, 0);
    let ok = keep.iter().any(|k| *k);
    if !ok {
        if std::env::var("MWDI_DEBUG").is_ok() {
            eprintln!("complete instances: {} incomplete, instantiation TU does not compile", names.len());
        }
        return 0;
    }
    let extra = render(&keep);
    // (the salt names the scanner generation: members of classes nested in templates)
    let full = format!("{context}\n// mwdec-inline instances v2\n{extra}");
    let db2 = match mwdec_ctx::build_typedb(&full, cflags, &mwdec_ctx::default_work_dir()) {
        Ok(d) => d,
        Err(e) => {
            if std::env::var("MWDI_DEBUG").is_ok() {
                eprintln!("complete instances: -g build failed: {e}");
            }
            return 0;
        }
    };
    let mut n = 0;
    for (name, k) in names.iter().zip(&keep) {
        if !*k {
            continue;
        }
        let Some(c2) = db2.classes.get(name) else { continue };
        if c2.is_declaration || (c2.fields.is_empty() && c2.bases.is_empty()) {
            continue;
        }
        let mut c = c2.clone();
        // keep what the unit's own build attached (vtables from the module objects)
        if let Some(old) = db.classes.get(name) {
            if c.vtable.is_empty() {
                c.vtable = old.vtable.clone();
            }
        }
        db.classes.insert(name.clone(), c);
        n += 1;
    }
    // classes the instantiation pulled in that the unit db lacks entirely (nested node types,
    // bases of the instances)
    for (name, c2) in &db2.classes {
        if !db.classes.contains_key(name) && !c2.is_declaration && name.contains('<') {
            db.classes.insert(name.clone(), c2.clone());
        }
    }
    // declarations the unit's (possibly older, cached) build lacks: members of classes nested
    // in class templates (`red_black_tree::const_iterator::operator==`)
    for (k, ds) in &db2.decls {
        if !db.decls.contains_key(k) && !k.starts_with("__mwdi") {
            db.decls.insert(k.clone(), ds.clone());
        }
    }
    mwdec_ctx::resolve::fill_methods(db);
    if std::env::var("MWDI_DEBUG").is_ok() {
        eprintln!("complete instances: {} incomplete, {} compiled, {n} completed, {:.1}s", names.len(), keep.iter().filter(|k| **k).count(), t0.elapsed().as_secs_f64());
    }
    n
}

/// [`complete_instances`] compiling with `m` in the unit context `ctx` (plain context on a
/// compiler crash), for callers that hold the unit's compiler setup.
pub fn complete_in(db: &mut TypeDb, context: &str, cflags: &[String], m: &mwdec_mwcc::Mwcc, ctx: &mwdec_mwcc::UnitContext, plain: &mwdec_mwcc::UnitContext) -> usize {
    complete_instances(db, context, cflags, &|code| {
        let mut r = m.compile_in(ctx, code);
        if let Err(mwdec_mwcc::MwccError::Compile { status, messages }) = &r {
            if status.is_some_and(|s| s < 0) || messages.contains("Unhandled exception") {
                r = m.compile_in(plain, code);
            }
        }
        match r {
            Err(e) => Err(e.messages().to_string()),
            Ok(c) => mwdec_obj::load_object_bytes(&c.obj_path.to_string_lossy(), &c.obj).map_err(|e| e.to_string()),
        }
    })
}

