//! Type context: compile a context TU (just `#include` lines) with `-g`, parse MWCC's
//! DWARF 1.1 into a `TypeDb`, and complement it with a light scan of the preprocessed headers.
//!
//! Pipeline (`build_typedb`):
//! 1. preprocess the context TU with the unit's flags (`-E`);
//! 2. scan the preprocessed text for every non-template class/struct/union/enum/typedef,
//!    namespace-scope variable, static data member and function declaration (`scan`);
//! 3. compile `context + forcing declarations` with `-g`: one `T* __mwdec_cN;` global per
//!    class/enum/typedef and `__typeof__(var)* __mwdec_vN;` per variable, because MWCC only
//!    emits DWARF for types reachable from emitted objects and never emits typedef DIEs;
//!    lines the compiler rejects are dropped and the compile retried;
//! 4. parse `.debug` (`dwarf`) -> classes/enums/... (`convert`), typedef targets and variable
//!    types from the forcing globals, `bool` fields restored, declarations resolved (`resolve`);
//! 5. cache the TypeDb as JSON under `work_dir/ctxcache/<hash>.json`.
pub mod convert;
pub mod dwarf;
pub mod layout;
pub mod mangle;
pub mod mwcc;
pub mod resolve;
pub mod scan;
pub mod vtable;

pub use layout::{class_of, field_at, field_at_type, field_exact, size_of, FieldAccess, PathStep};
pub use vtable::{apply_vtables, vtables_from_object, vtables_from_path};

use anyhow::Result;
use mwdec_core::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Bump when the TypeDb produced for the same inputs changes (invalidates caches).
const CACHE_VERSION: &str = "mwdec-ctx-17-abs-addrs-errs";

/// Parse the DWARF of an already-compiled MWCC object into a TypeDb (no header scan).
pub fn typedb_from_object_bytes(elf: &[u8]) -> Result<TypeDb> {
    let info = dwarf::read_dwarf(elf)?;
    Ok(convert::Converter::new(&info).build())
}

fn fnv1a64(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for p in parts {
        for &b in *p {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Cache key of a context build.
pub fn context_hash(context_tu: &str, cflags: &[String], root: &Path) -> String {
    let flags = cflags.join("\u{1}");
    let root = root.to_string_lossy();
    format!(
        "{:016x}",
        fnv1a64(&[CACHE_VERSION.as_bytes(), context_tu.as_bytes(), flags.as_bytes(), root.as_bytes()])
    )
}

/// Statistics of the last build (for reports/tests).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct BuildStats {
    pub cached: bool,
    pub forced_total: usize,
    pub forced_rejected: usize,
    pub compile_attempts: usize,
    /// `#include` lines of the context that failed to preprocess and were dropped.
    #[serde(default)]
    pub dropped_includes: Vec<String>,
    pub preprocess_ms: u128,
    pub compile_ms: u128,
    pub total_ms: u128,
}

/// `build_typedb` with the project root from `$MWDEC_PROJECT_ROOT` (default: the frozen tree).
pub fn build_typedb(context_tu: &str, cflags: &[String], work_dir: &Path) -> Result<TypeDb> {
    build_typedb_in(&mwcc::project_root(), context_tu, cflags, work_dir).map(|(db, _)| db)
}

/// Build (or load from cache) the TypeDb of a context TU, compiling in `root`.
pub fn build_typedb_in(root: &Path, context_tu: &str, cflags: &[String], work_dir: &Path) -> Result<(TypeDb, BuildStats)> {
    let t0 = std::time::Instant::now();
    let key = context_hash(context_tu, cflags, root);
    let cache_dir = work_dir.join("ctxcache");
    let cache_file = cache_dir.join(format!("{key}.json.gz"));
    if let Some(mut db) = load_cache(&cache_file) {
        {
            resolve::fill_methods(&mut db);
            let stats = BuildStats { cached: true, total_ms: t0.elapsed().as_millis(), ..Default::default() };
            return Ok((db, stats));
        }
    }
    std::fs::create_dir_all(&cache_dir)?;
    // unique per build: concurrent builds of the same context must not share files
    static BUILD_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let seq = BUILD_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = work_dir.join("ctxbuild").join(format!("{key}.{}.{seq}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let (db, mut stats) = build_uncached(root, context_tu, cflags, &dir)?;
    // atomic-ish publish (several processes may build the same context)
    let tmp = cache_dir.join(format!("{key}.{}.tmp", std::process::id()));
    save_cache(&tmp, &db)?;
    let _ = std::fs::rename(&tmp, &cache_file);
    let _ = std::fs::remove_file(&tmp);
    if std::env::var_os("MWDEC_CTX_KEEP").is_none() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    stats.total_ms = t0.elapsed().as_millis();
    Ok((db, stats))
}

/// Cache format: gzip'd JSON of the TypeDb without `Class::methods` (rebuilt from `decls`).
fn save_cache(path: &Path, db: &TypeDb) -> Result<()> {
    use std::io::Write;
    let mut slim = db.clone();
    for c in slim.classes.values_mut() {
        c.methods.clear();
    }
    let f = std::fs::File::create(path)?;
    let mut enc = flate2::write::GzEncoder::new(std::io::BufWriter::new(f), flate2::Compression::fast());
    serde_json::to_writer(&mut enc, &slim)?;
    enc.finish()?.flush()?;
    Ok(())
}

fn load_cache(path: &Path) -> Option<TypeDb> {
    let bytes = std::fs::read(path).ok()?;
    let mut json = Vec::new();
    use std::io::Read;
    flate2::read::GzDecoder::new(&bytes[..]).read_to_end(&mut json).ok()?;
    serde_json::from_slice(&json).ok()
}

#[derive(Clone, Debug)]
enum Force {
    Type,
    Typedef(String),
    Var(String),
}

fn mangle_scope(scope: &str) -> Option<String> {
    let parts = mangle::split_scope(scope);
    if parts.iter().any(|p| p.contains('<') || p.is_empty()) {
        return None;
    }
    if parts.len() == 1 {
        Some(format!("{}{}", parts[0].len(), parts[0]))
    } else if parts.len() < 10 {
        Some(format!("Q{}{}", parts.len(), parts.iter().map(|p| format!("{}{}", p.len(), p)).collect::<String>()))
    } else {
        None
    }
}

/// Symbol name MWCC gives a variable `qualified` (`gpMain`, `sZeroVector__9CVector3f`).
pub fn mangle_variable(qualified: &str) -> Option<String> {
    let parts = mangle::split_scope(qualified);
    let name = parts.last()?;
    if parts.len() == 1 {
        return Some(name.to_string());
    }
    let scope = parts[..parts.len() - 1].join("::");
    Some(format!("{name}__{}", mangle_scope(&scope)?))
}

fn strip_ptr(t: Type) -> Type {
    match t {
        Type::Ptr(x) => *x,
        t => t,
    }
}

/// Preprocess `text` (`-E`); Ok(Err(messages)) when the compiler rejects it.
fn try_preprocess(root: &Path, text: &str, flags: &[String], dir: &Path) -> Result<std::result::Result<String, String>> {
    let mut args = flags.to_vec();
    args.push("-E".into());
    Ok(mwcc::compile_text(root, text, &args, dir)?.map(|b| String::from_utf8_lossy(&b).into_owned()))
}

fn preprocess_text(root: &Path, text: &str, flags: &[String], dir: &Path) -> Result<String> {
    match try_preprocess(root, text, flags, dir)? {
        Ok(p) => Ok(p),
        Err(m) => anyhow::bail!("preprocessing the context failed:\n{m}"),
    }
}

/// Remove context lines that make preprocessing fail: lines the compiler points at directly,
/// otherwise the first failing line found by bisecting on prefixes.
fn drop_bad_includes(root: &Path, ctx: &str, flags: &[String], dir: &Path) -> Result<(String, Vec<String>)> {
    let mut lines: Vec<String> = ctx.lines().map(|l| l.to_string()).collect();
    let mut dropped = Vec::new();
    let try_lines = |ls: &[String]| -> Result<(bool, String)> {
        let mut t = ls.join("\n");
        t.push('\n');
        Ok(match try_preprocess(root, &t, flags, dir)? {
            Ok(_) => (true, String::new()),
            Err(m) => (false, m),
        })
    };
    for _ in 0..16 {
        let (ok, out) = try_lines(&lines)?;
        if ok {
            return Ok((lines.join("\n") + "\n", dropped));
        }
        let text = lines.join("\n") + "\n";
        let bad = mwcc::error_lines(&out, &text);
        let bad_line = if let Some(&l) = bad.first() {
            l - 1
        } else {
            // bisect: smallest prefix that fails
            let (mut lo, mut hi) = (0usize, lines.len());
            while lo + 1 < hi {
                let mid = (lo + hi) / 2;
                if try_lines(&lines[..mid])?.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            hi - 1
        };
        if bad_line >= lines.len() || lines[bad_line].trim().is_empty() {
            break;
        }
        dropped.push(lines[bad_line].clone());
        lines[bad_line] = String::new();
    }
    anyhow::bail!("could not make the context preprocess (dropped {dropped:?})")
}

/// Preprocess a context TU with the unit's flags (`-E`), returning the text.
pub fn preprocess(root: &Path, context_tu: &str, cflags: &[String], dir: &Path) -> Result<String> {
    std::fs::create_dir_all(dir)?;
    preprocess_text(root, context_tu, &mwcc::sanitize_flags(cflags), dir)
}

fn build_uncached(root: &Path, context_tu: &str, cflags: &[String], dir: &Path) -> Result<(TypeDb, BuildStats)> {
    let mut stats = BuildStats::default();
    let flags = mwcc::sanitize_flags(cflags);
    let ctx_cpp = dir.join("ctx.cpp");
    let mut ctx_text = context_tu.to_string();
    if !ctx_text.ends_with('\n') {
        ctx_text.push('\n');
    }
    std::fs::write(&ctx_cpp, &ctx_text)?;

    // 1. preprocess (dropping #include lines that cannot be preprocessed, e.g. headers that
    // live next to the unit's source and are not on the include path)
    let t = std::time::Instant::now();
    let pre = match try_preprocess(root, &ctx_text, &flags, dir)? {
        Ok(p) => p,
        Err(_) => {
            let (text, dropped) = drop_bad_includes(root, &ctx_text, &flags, dir)?;
            stats.dropped_includes = dropped;
            ctx_text = text;
            std::fs::write(&ctx_cpp, &ctx_text)?;
            preprocess_text(root, &ctx_text, &flags, dir)?
        }
    };
    stats.preprocess_ms = t.elapsed().as_millis();

    // 2. scan
    let c_mode = flags.iter().any(|f| f == "-lang=c" || f == "-lang=c99") || (flags.windows(2).any(|w| w[0] == "-lang" && (w[1] == "c" || w[1] == "c99")));
    let sr = scan::scan_lang(&pre, c_mode);

    // 3. forcing TU
    let mut forces: Vec<(Force, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for c in sr.classes.iter().chain(sr.enums.iter()) {
        if seen.insert(c.clone()) {
            let i = forces.len();
            // elaborated specifier: required for struct tags in C, harmless in C++
            let kw = sr.tag_keyword.get(c).map(|s| s.as_str()).unwrap_or("struct");
            forces.push((Force::Type, format!("{kw} {c}* {}c{i};", convert::DUMMY_PREFIX)));
        }
    }
    for td in &sr.typedefs {
        if seen.insert(td.clone()) {
            let i = forces.len();
            forces.push((Force::Typedef(td.clone()), format!("{td}* {}t{i};", convert::DUMMY_PREFIX)));
        }
    }
    // variables live in their own namespace (C: `struct synthInfo synthInfo;`)
    let mut seen_vars = std::collections::HashSet::new();
    for v in &sr.globals {
        if seen_vars.insert(v.clone()) {
            let i = forces.len();
            forces.push((Force::Var(v.clone()), format!("__typeof__({v})* {}v{i};", convert::DUMMY_PREFIX)));
        }
    }
    stats.forced_total = forces.len();
    let ctx_lines = ctx_text.lines().count();
    let mut alive: Vec<bool> = vec![true; forces.len()];
    let force_cpp = dir.join("force.cpp");
    let force_o = dir.join("force.o");
    let t = std::time::Instant::now();
    let mut elf: Option<Vec<u8>> = None;
    let mut last_output = String::new();
    for attempt in 0..8 {
        stats.compile_attempts = attempt + 1;
        let mut text = ctx_text.clone();
        let mut line_of: BTreeMap<usize, usize> = BTreeMap::new();
        for (k, (_, line)) in forces.iter().enumerate() {
            // one line per force (empty when dropped) keeps line numbers stable
            line_of.insert(ctx_lines + 1 + k, k);
            if alive[k] {
                text.push_str(line);
            }
            text.push('\n');
        }
        text.push_str("void __mwdec_anchor(void) {}\n");
        if std::env::var_os("MWDEC_CTX_KEEP").is_some() {
            std::fs::write(&force_cpp, &text)?;
        }
        let mut args = flags.clone();
        args.extend(["-g".into(), "-maxerrors".into(), "1000".into()]);
        let messages = match mwcc::compile_text(root, &text, &args, dir)? {
            Ok(bytes) => {
                elf = Some(bytes);
                break;
            }
            Err(m) => m,
        };
        let bad = mwcc::error_lines(&messages, &text);
        let mut removed = 0;
        for l in bad {
            if let Some(&k) = line_of.get(&l) {
                if alive[k] {
                    alive[k] = false;
                    removed += 1;
                }
            }
        }
        last_output = messages;
        if removed == 0 {
            // errors we cannot attribute: drop all forcing (degraded, but still a TypeDb)
            if alive.iter().any(|a| *a) {
                alive.iter_mut().for_each(|a| *a = false);
            } else {
                break;
            }
        }
    }
    stats.compile_ms = t.elapsed().as_millis();
    stats.forced_rejected = alive.iter().filter(|a| !**a).count();
    if std::env::var_os("MWDEC_CTX_KEEP").is_some() {
        let rej: Vec<&str> = forces.iter().zip(&alive).filter(|(_, a)| !**a).map(|((_, l), _)| l.as_str()).collect();
        let _ = std::fs::write(dir.join("rejected.txt"), rej.join("\n"));
        let _ = std::fs::write(dir.join("last_output.txt"), &last_output);
    }
    let Some(elf) = elf else {
        anyhow::bail!("compiling the context with -g failed:\n{last_output}");
    };
    if std::env::var_os("MWDEC_CTX_KEEP").is_some() {
        let _ = std::fs::write(&force_o, &elf);
    }

    // 4. DWARF -> TypeDb
    let info = dwarf::read_dwarf(&elf)?;
    let conv = convert::Converter::new(&info);
    let (mut db, dummies) = conv.build_with_dummies();
    let missing_access = conv.missing_access.take();
    for (k, (f, _)) in forces.iter().enumerate() {
        if !alive[k] {
            continue;
        }
        let dname = match f {
            Force::Type => format!("{}c{k}", convert::DUMMY_PREFIX),
            Force::Typedef(_) => format!("{}t{k}", convert::DUMMY_PREFIX),
            Force::Var(_) => format!("{}v{k}", convert::DUMMY_PREFIX),
        };
        let Some(t) = dummies.get(&dname) else { continue };
        match f {
            Force::Type => {}
            Force::Typedef(n) => {
                let u = strip_ptr(t.clone());
                if u != Type::Named(n.clone()) {
                    db.typedefs.insert(n.clone(), u);
                }
            }
            Force::Var(n) => {
                let key = mangle_variable(n).unwrap_or_else(|| n.clone());
                db.globals.insert(key, (n.clone(), strip_ptr(t.clone())));
            }
        }
    }
    // `typedef struct {..} X;` / `typedef enum {..} X;`: MWCC emits no typedef DIEs, so the
    // anonymous type only shows up as `@anonN`; give it the typedef's name.
    let mut renames: BTreeMap<String, String> = BTreeMap::new();
    for (n, t) in &db.typedefs {
        if let Type::Named(a) = t {
            if a.starts_with('@') && !renames.contains_key(a) {
                renames.insert(a.clone(), n.clone());
            }
        }
    }
    if !renames.is_empty() {
        convert::rename_types(&mut db, &renames);
        for n in renames.values() {
            db.typedefs.remove(n);
        }
    }
    db.namespaces.extend(sr.namespaces.iter().cloned());
    for (n, a) in &sr.abs_addrs {
        db.abs_addrs.insert(n.clone(), *a);
    }
    for (n, kw) in &sr.tag_keyword {
        db.tag_keywords.insert(n.clone(), kw.clone());
    }
    for (c, f) in &sr.friends {
        db.friends.entry(c.clone()).or_default().push(f.clone());
    }
    resolve::patch_bool_fields(&mut db, &sr.fields);
    resolve::patch_void_pointers(&mut db, &sr.fields, &sr.template_params);
    resolve::patch_const_pointees(&mut db, &sr.fields);
    resolve::patch_void_pointers(&mut db, &sr.template_fields, &sr.template_params);
    resolve::patch_field_access(&mut db, &sr.fields, &missing_access);
    resolve::apply_decls(&mut db, &sr);
    Ok((db, stats))
}

fn sig_from_decl(d: &DeclInfo, mangled: Option<&str>, db: &TypeDb) -> FuncSig {
    let parts = mangle::split_scope(&d.qualified_name);
    let scope = parts[..parts.len() - 1].join("::");
    let is_class = !scope.is_empty() && is_class_scope(&scope, db);
    FuncSig {
        qualified_name: d.qualified_name.clone(),
        mangled: mangled.map(|s| s.to_string()),
        ret: d.ret.clone(),
        params: d.params.clone(),
        this_class: if is_class && !d.is_static { Some(scope) } else { None },
        is_const: d.is_const,
        is_static: d.is_static,
        is_virtual: d.is_virtual,
        variadic: d.variadic,
    }
}

/// Is `scope` (from a mangled name) a class rather than a namespace?
pub fn is_class_scope(scope: &str, db: &TypeDb) -> bool {
    // the anonymous namespace (`@unnamed@CFoo_cpp@`)
    if scope.starts_with("@unnamed@") && !scope.contains("::") {
        return false;
    }
    if db.classes.contains_key(scope) {
        return true;
    }
    // project namespaces are lower-case (`rstl`, `std`, `nl`); classes `CFoo`/`SFoo`/`TFoo`: an
    // unknown lower-case scope is a namespace the context doesn't declare
    let last = scope.rsplit("::").next().unwrap_or(scope);
    if last.chars().next().is_some_and(|c| c.is_ascii_lowercase()) && !scope.contains('<') {
        return false;
    }
    !db.namespaces.contains(scope)
}

/// Signature from a mangled CodeWarrior name: parameter types and constness come from the
/// mangling (exact), return type / static / virtual / parameter names from the scanned header
/// declaration when one matches. Unknown return type = `Type::Unknown { size: 0 }`.
/// Unmangled (C) names are looked up in the declarations.
pub fn sig_from_mangled(mangled: &str, db: &TypeDb) -> Option<FuncSig> {
    let Some(m) = mangle::parse_mangled_fn(mangled) else {
        if let Some(d) = db.decls.get(mangled).and_then(|v| v.first()) {
            return Some(sig_from_decl(d, Some(mangled), db));
        }
        return db.functions.get(mangled).cloned().or_else(|| runtime_helper(mangled));
    };
    let qualified = match &m.scope {
        Some(s) => format!("{s}::{}", m.name),
        None => m.name.clone(),
    };
    let keys: Vec<Type> = m.params.iter().map(|p| resolve::param_key(db, &p.ty)).collect();
    let pick = |list: &[DeclInfo]| -> Option<DeclInfo> {
        let same_shape = |d: &&DeclInfo| d.params.len() == m.params.len() && d.is_const == m.is_const && d.variadic == m.variadic;
        list.iter()
            .filter(same_shape)
            .find(|d| d.params.iter().map(|p| resolve::param_key(db, &p.ty)).collect::<Vec<_>>() == keys)
            .or_else(|| {
                let cands: Vec<&DeclInfo> = list.iter().filter(same_shape).collect();
                if cands.len() == 1 {
                    Some(cands[0])
                } else {
                    None
                }
            })
            .cloned()
    };
    let mut decl = db.decls.get(&qualified).and_then(|l| pick(l));
    if decl.is_none() {
        decl = template_decl(&m, db).and_then(|l| pick(&l));
    }
    let decl = decl.as_ref();
    let is_ctor_dtor = matches!(m.special.as_deref(), Some("ct") | Some("dt"));
    let ret = m
        .ret
        .clone()
        .or_else(|| decl.map(|d| d.ret.clone()))
        .unwrap_or(match (is_ctor_dtor, m.name.as_str()) {
            (true, _) | (_, "operator delete") | (_, "operator delete[]") => Type::Void,
            (_, "operator new") | (_, "operator new[]") => Type::Ptr(Box::new(Type::Void)),
            _ => Type::Unknown { size: 0 },
        });
    let is_static = decl.is_some_and(|d| d.is_static);
    let class_scope = m.scope.as_deref().filter(|s| is_class_scope(s, db));
    // member types named unqualified inside the class (`const_iterator find(...)`)
    let ret = match class_scope {
        Some(c) => qualify_nested(db, &ret, c),
        None => ret,
    };
    let in_vtable = class_scope
        .and_then(|c| db.classes.get(c))
        .is_some_and(|c| c.vtable.iter().any(|v| v.sig.mangled.as_deref() == Some(mangled)));
    let mut params = m.params.clone();
    if let Some(d) = decl {
        for (p, dp) in params.iter_mut().zip(&d.params) {
            p.name = dp.name.clone();
        }
    }
    Some(FuncSig {
        qualified_name: qualified,
        mangled: Some(mangled.to_string()),
        ret,
        params,
        this_class: if is_static { None } else { class_scope.map(|s| s.to_string()) },
        is_const: m.is_const,
        is_static,
        is_virtual: in_vtable || decl.is_some_and(|d| d.is_virtual),
        variadic: m.variadic,
    })
}

/// Qualify member type names used unqualified in a member declaration of `scope`
/// (`iterator` -> `rstl::vector<int, ...>::iterator`), also inside template arguments.
pub fn qualify_nested(db: &TypeDb, t: &Type, scope: &str) -> Type {
    match t {
        Type::Named(n) => Type::Named(qualify_nested_str(db, n, scope)),
        Type::Ptr(x) => Type::Ptr(Box::new(qualify_nested(db, x, scope))),
        Type::Ref(x) => Type::Ref(Box::new(qualify_nested(db, x, scope))),
        Type::Const(x) => Type::Const(Box::new(qualify_nested(db, x, scope))),
        Type::Volatile(x) => Type::Volatile(Box::new(qualify_nested(db, x, scope))),
        Type::Array(x, k) => Type::Array(Box::new(qualify_nested(db, x, scope)), *k),
        t => t.clone(),
    }
}

fn nested_in(db: &TypeDb, scope: &str, name: &str, depth: u32) -> bool {
    if depth > 8 {
        return false;
    }
    let q = format!("{scope}::{name}");
    if db.classes.contains_key(&q) || db.typedefs.contains_key(&q) || db.enums.contains_key(&q) {
        return true;
    }
    let base = strip_last_targs(scope);
    if db.template_typedefs.contains_key(&format!("{base}::{name}")) || db.classes.keys().any(|k| k.starts_with(&q) && k[q.len()..].starts_with('<')) {
        return true;
    }
    match db.classes.get(scope) {
        Some(c) => c.bases.iter().any(|b| nested_in(db, &b.name, name, depth + 1)),
        None => false,
    }
}

fn qualify_nested_str(db: &TypeDb, n: &str, scope: &str) -> String {
    let b = n.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let id = &n[st..i];
            let after_scope = out.ends_with("::");
            let known = db.classes.contains_key(id) || db.typedefs.contains_key(id) || db.enums.contains_key(id) || db.templates.contains_key(id) || db.namespaces.contains(id);
            let keyword = matches!(id, "const" | "volatile" | "unsigned" | "signed" | "int" | "char" | "short" | "long" | "float" | "double" | "bool" | "void" | "wchar_t");
            if !after_scope && !known && !keyword && nested_in(db, scope, id, 0) {
                out.push_str(scope);
                out.push_str("::");
            }
            out.push_str(id);
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// Signatures of MWCC runtime helpers the compiler calls implicitly (64-bit arithmetic,
/// conversions) and of static-initializer functions; not declared in any header.
pub fn runtime_helper(name: &str) -> Option<FuncSig> {
    let ll = Type::Int { size: 8, signed: true };
    let ull = Type::Int { size: 8, signed: false };
    let int = Type::Int { size: 4, signed: true };
    let dbl = Type::Float { size: 8 };
    let (ret, params): (Type, Vec<Type>) = match name {
        "__div2i" | "__mod2i" => (ll.clone(), vec![ll.clone(), ll]),
        "__div2u" | "__mod2u" => (ull.clone(), vec![ull.clone(), ull]),
        "__shl2i" | "__shr2i" => (ll.clone(), vec![ll, int]),
        "__shr2u" => (ull.clone(), vec![ull, int]),
        "__cvt_fp2unsigned" => (Type::Int { size: 4, signed: false }, vec![dbl]),
        "__cvt_dbl_ull" => (ull, vec![dbl]),
        "__cvt_dbl_ll" => (ll, vec![dbl]),
        "__cvt_ll_dbl" => (dbl, vec![ll]),
        "__cvt_ull_dbl" => (dbl, vec![ull]),
        "__cvt_sll_flt" => (Type::Float { size: 4 }, vec![ll]),
        "__cvt_ull_flt" => (Type::Float { size: 4 }, vec![ull]),
        n if n.starts_with("__sinit_") || n.starts_with("__sterm_") => (Type::Void, vec![]),
        _ => return None,
    };
    Some(FuncSig {
        qualified_name: name.to_string(),
        mangled: Some(name.to_string()),
        ret,
        params: params.into_iter().map(|ty| Param { name: None, ty }).collect(),
        this_class: None,
        is_const: false,
        is_static: false,
        is_virtual: false,
        variadic: false,
    })
}

fn strip_last_targs(q: &str) -> String {
    let parts = mangle::split_scope(q);
    let mut v: Vec<String> = parts.iter().map(|p| p.to_string()).collect();
    if let Some(last) = v.last_mut() {
        if let Some(i) = last.find('<') {
            last.truncate(i);
        }
    }
    v.join("::")
}

/// Declarations of the template a mangled member/function instantiates, with the template
/// parameters substituted by the mangled template arguments.
fn template_decl(m: &mangle::MangledFn, db: &TypeDb) -> Option<Vec<DeclInfo>> {
    if m.scope_targs.is_empty() && m.fn_targs.is_empty() {
        return None;
    }
    let scope_base = m.scope.as_deref().map(strip_last_targs);
    let name_base = strip_last_targs(&m.name);
    let key = match &scope_base {
        Some(s) => format!("{s}::{name_base}"),
        None => name_base.clone(),
    };
    let list = db.decls.get(&key)?;
    let class_params: Vec<String> = scope_base
        .as_ref()
        .and_then(|s| db.templates.get(s).cloned())
        .unwrap_or_default();
    let mut out = Vec::new();
    for d in list.iter().filter(|d| !d.template_params.is_empty()) {
        let mut bind: std::collections::HashMap<String, (Type, String)> = Default::default();
        let ncls = if m.scope_targs.is_empty() { 0 } else { class_params.len().min(d.template_params.len()) };
        for (p, a) in d.template_params[..ncls].iter().zip(&m.scope_targs) {
            bind.insert(p.clone(), a.clone());
        }
        for (p, a) in d.template_params[ncls..].iter().zip(&m.fn_targs) {
            bind.insert(p.clone(), a.clone());
        }
        let inj = match (&scope_base, &m.scope) {
            (Some(b), Some(s)) if !m.scope_targs.is_empty() => Some((b.as_str(), s.as_str())),
            _ => None,
        };
        let mut nd = d.clone();
        nd.ret = resolve::substitute(db, &d.ret, &bind, inj);
        for p in &mut nd.params {
            p.ty = resolve::substitute(db, &p.ty, &bind, inj);
        }
        if resolve::has_template_residue(&nd.ret, &d.template_params) {
            nd.ret = Type::Unknown { size: 0 };
        }
        out.push(nd);
    }
    Some(out)
}

/// Default scratch dir for context builds.
pub fn default_work_dir() -> PathBuf {
    mwdec_core::paths::work_dir("mwdec-ctx")
}
