//! Split PCH contexts: a workaround for MWCC crashing (access violation) when a candidate is
//! compiled against some units' precompiled header.
//!
//! The crash depends on a few context headers being *in* the PCH (e.g. a call to an out-of-line
//! member of another class crashes once a particular header is precompiled). Without a fix, every
//! candidate of such a unit falls back to the plain context (the whole header set parsed as text,
//! 1.5-2 s per compile instead of ~60 ms), which starves the search.
//!
//! [`Mwcc::repair`] finds a small set of context lines (delta debugging over the include lines)
//! that, left out of the PCH and included as text after it, stops the crash; the result is checked
//! against the plain context (identical object bytes for the crashing candidate) and cached on disk
//! next to the PCH, so it is found once per unit context. `compile_in` then uses the split context
//! for every candidate of that context. Callers that need certainty (an exact match) re-check with
//! the plain context ([`Mwcc::uses_split`]).
use super::*;

/// The include lines of a context TU (non-empty lines, in order).
fn lines_of(context: &str) -> Vec<&str> {
    context.lines().filter(|l| !l.trim().is_empty()).collect()
}

type ObjectFileRef = mwdec_core::ObjectFile;

/// Same functions (strict comparison, binding) and same data in two compiled objects. Byte
/// equality is too strict: the object records the (unique) temporary source file name.
pub(crate) fn same_code(a: &[u8], b: &[u8]) -> bool {
    let (Ok(oa), Ok(ob)) = (mwdec_obj::load_object_bytes("a.o", a), mwdec_obj::load_object_bytes("b.o", b)) else { return false };
    if oa.functions.len() != ob.functions.len() {
        return false;
    }
    let fns = oa.functions.iter().all(|f| {
        mwdec_obj::find_function(&ob, &f.name).is_some_and(|g| f.binding == g.binding && crate::compare(&oa, f, &ob, g).exact)
    });
    // Compiler-local data (`@123`, header statics) is numbered and ordered by declaration order,
    // which moving headers out of the PCH changes: compare as a multiset of (section, bytes).
    let data = |o: &ObjectFileRef| {
        let mut v: Vec<(String, Vec<u8>)> = o.data.values().map(|d| (if d.name.starts_with('@') { String::new() } else { d.name.clone() }, d.bytes.clone())).collect();
        v.sort();
        v
    };
    fns && data(&oa) == data(&ob)
}

/// At most this many trial precompiles per repair.
const MAX_TRIALS: usize = 24;

impl Mwcc {
    fn split_key(&self, ctx: &UnitContext) -> u128 {
        // Not keyed by flags: the diagnostic `-sym on` variant of a context reuses the base
        // context's split (re-verified on use).
        content_hash(&[CACHE_SALT.as_bytes(), b"split", self.compiler.as_bytes(), ctx.context.as_bytes()])
    }

    fn split_file(&self, ctx: &UnitContext) -> Option<PathBuf> {
        let dir = ctx.mch.as_ref()?.parent()?.to_path_buf();
        Some(dir.join(format!("{:032x}.split", self.split_key(ctx))))
    }

    /// The split replacement of a PCH context, if one has been established for it.
    pub fn known_split(&self, ctx: &UnitContext) -> Option<UnitContext> {
        let mut s = self.splits.lock().unwrap().get(&ctx.hash).and_then(|c| c.get().cloned()).flatten()?;
        s.tu_name = ctx.tu_name.clone();
        Some(s)
    }

    /// Whether candidates in `ctx` are compiled against a split PCH (exact results should be
    /// confirmed with the plain context).
    pub fn uses_split(&self, ctx: &UnitContext) -> bool {
        self.known_split(ctx).is_some()
    }

    /// Context whose PCH holds the context lines except `excluded`, which are included as text
    /// in front of each candidate (followed by `#line 1`, so candidate line numbers are unchanged).
    pub fn precompile_split(&self, ctx: &UnitContext, excluded: &[usize]) -> Result<UnitContext, MwccError> {
        self.split_context(ctx, excluded, false)
    }

    /// [`Mwcc::precompile_split`] into a private `.mch` that only the caller uses (and deletes).
    fn precompile_split_trial(&self, ctx: &UnitContext, excluded: &[usize]) -> Result<UnitContext, MwccError> {
        self.split_context(ctx, excluded, true)
    }

    fn split_context(&self, ctx: &UnitContext, excluded: &[usize], private: bool) -> Result<UnitContext, MwccError> {
        let lines = lines_of(&ctx.context);
        let pch: Vec<&str> = lines.iter().enumerate().filter(|(i, _)| !excluded.contains(i)).map(|(_, l)| *l).collect();
        let text: Vec<&str> = lines.iter().enumerate().filter(|(i, _)| excluded.contains(i)).map(|(_, l)| *l).collect();
        let mut pctx = format!("{}\n", pch.join("\n"));
        if pch.is_empty() {
            pctx = "/* empty */\n".into();
        }
        let built = if private { self.precompile_private(&pctx, &ctx.cflags)? } else { self.precompile_mch(&pctx, &ctx.cflags)? };
        Ok(UnitContext {
            cflags: ctx.cflags.clone(),
            context: ctx.context.clone(),
            mch: built.mch,
            // Same identity as the full context: the split is verified to compile identically.
            hash: ctx.hash,
            text: format!("{}\n#line 1\n", text.join("\n")),
            excluded: excluded.to_vec(),
            tu_name: ctx.tu_name.clone(),
        })
    }

    /// Compile with a (possibly split) PCH context, no caching and no fallback.
    pub(crate) fn compile_pch(&self, ctx: &UnitContext, code: &str) -> Result<Compiled, MwccError> {
        let tmp = self.work.join("tmp");
        let m = ctx.mch.as_deref();
        let name = ctx.tu_name.as_deref();
        if ctx.text.is_empty() {
            self.compile_tu_as(code, &ctx.cflags, m, &tmp, name)
        } else {
            self.compile_tu_as(&format!("{}{code}", ctx.text), &ctx.cflags, m, &tmp, name)
        }
    }

    pub(crate) fn plain_compile(&self, ctx: &UnitContext, code: &str) -> Result<Compiled, MwccError> {
        let mut tu = ctx.context.clone();
        if !tu.is_empty() && !tu.ends_with('\n') {
            tu.push('\n');
        }
        tu.push_str(code);
        self.compile_tu_as(&tu, &ctx.cflags, None, &self.work.join("tmp"), ctx.tu_name.as_deref())
    }

    /// The split context for `ctx` given a candidate `code` that crashes its PCH: established once
    /// per context (concurrent callers wait for the first), cached on disk.
    pub fn repair(&self, ctx: &UnitContext, code: &str) -> Option<UnitContext> {
        ctx.mch.as_ref()?;
        if std::env::var_os("MWDEC_NO_SPLIT").is_some() {
            return None;
        }
        let cell = self.splits.lock().unwrap().entry(ctx.hash).or_default().clone();
        let mut s = cell.get_or_init(|| self.find_split(ctx, code)).clone()?;
        s.tu_name = ctx.tu_name.clone();
        Some(s)
    }

    fn find_split(&self, ctx: &UnitContext, code: &str) -> Option<UnitContext> {
        let log = std::env::var_os("MWDEC_MWCC_LOG").is_some();
        let file = self.split_file(ctx);
        if let Some(f) = &file {
            if let Ok(s) = std::fs::read_to_string(f) {
                let s = s.trim();
                if s == "none" {
                    return None;
                }
                let ex: Vec<usize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if let Ok(sp) = self.precompile_split(ctx, &ex) {
                    match (self.compile_pch(&sp, code), self.plain_compile(ctx, code)) {
                        (Ok(a), Ok(b)) if same_code(&a.obj, &b.obj) => return Some(sp),
                        (Err(MwccError::Compile { .. }), Err(MwccError::Compile { .. })) => return Some(sp),
                        _ => {}
                    }
                }
            }
        }
        let t0 = Instant::now();
        let n = lines_of(&ctx.context).len();
        let trials = std::cell::Cell::new(0usize);
        // crash-free with these lines out of the PCH?
        let mut ok = |ex: &[usize]| -> Option<bool> {
            trials.set(trials.get() + 1);
            if trials.get() > MAX_TRIALS {
                return None;
            }
            // a private PCH: the shared content-addressed one of the same lines may be in use by
            // another process (deleting it made their compiles fail with "cannot be opened")
            let sp = self.precompile_split_trial(ctx, ex).ok()?;
            let r = !matches!(self.compile_pch(&sp, code), Err(MwccError::Crash { .. }));
            if let Some(m) = sp.mch.as_ref() {
                let _ = std::fs::remove_file(m);
            }
            Some(r)
        };
        // Culprit search. A crash needs every line of some crashing set in the PCH, so leaving out
        // one line of each set is enough. The last line of the shortest crashing PCH prefix (the
        // remaining lines as text) belongs to a crashing set: binary search for it, leave it out,
        // and repeat while the PCH still crashes. The unit's own header (line 0) is tried first.
        let mut ex: Vec<usize> = Vec::new();
        let mut found = false;
        match ok(&[0]) {
            Some(true) => {
                ex = vec![0];
                found = true;
            }
            Some(false) => {}
            None => {}
        }
        let mut rounds = 0;
        while !found && rounds < 3 && trials.get() < MAX_TRIALS {
            rounds += 1;
            // the PCH holds lines[..k] minus `ex`; crash(k) is monotone in k
            let crash = |k: usize, ok: &mut dyn FnMut(&[usize]) -> Option<bool>| -> Option<bool> {
                let mut e: Vec<usize> = ex.clone();
                e.extend(k..n);
                e.sort();
                e.dedup();
                ok(&e).map(|b| !b)
            };
            let (mut lo, mut hi) = (0usize, n); // crash(lo) false, crash(hi) true (hi = full minus ex)
            let mut failed = false;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                match crash(mid, &mut ok) {
                    Some(true) => hi = mid,
                    Some(false) => lo = mid,
                    None => {
                        failed = true;
                        break;
                    }
                }
            }
            if failed || ex.contains(&(hi - 1)) {
                break;
            }
            ex.push(hi - 1);
            ex.sort();
            match ok(&ex) {
                Some(true) => found = true,
                Some(false) => {}
                None => break,
            }
        }
        // A split must leave most of the context precompiled to be worth it, and compile the
        // crashing candidate exactly like the plain context.
        let mut why = String::new();
        let result = (|| {
            if !found || ex.len() * 2 > n.max(1) {
                why = format!("found {found}, {} of {n} lines excluded", ex.len());
                return None;
            }
            let sp = self.precompile_split(ctx, &ex).ok()?;
            match (self.compile_pch(&sp, code), self.plain_compile(ctx, code)) {
                (Ok(a), Ok(b)) if same_code(&a.obj, &b.obj) => Some(sp),
                (Err(MwccError::Compile { .. }), Err(MwccError::Compile { .. })) => Some(sp),
                (a, b) => {
                    why = format!("lines {ex:?}: split {} vs plain {}", a.is_ok(), b.is_ok());
                    if let (Ok(a), Ok(b)) = (&a, &b) {
                        why += &format!(" (objects differ: {} vs {} bytes)", a.obj.len(), b.obj.len());
                    }
                    None
                }
            }
        })();
        if log {
            eprintln!(
                "mwcc: split PCH for context {:032x}: {} ({} trials, {:.1}s)",
                ctx.hash,
                match &result {
                    Some(s) => format!("lines {:?} as text", s.excluded),
                    None => format!("none ({why})"),
                },
                trials.get(),
                t0.elapsed().as_secs_f64()
            );
        }
        if let Some(f) = &file {
            let body = match &result {
                Some(s) => s.excluded.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
                None => "none".into(),
            };
            // written whole (readers never see a partial file)
            let tmp = f.with_extension(format!("split.{}", self.unique("w")));
            if std::fs::write(&tmp, body).is_ok() && std::fs::rename(&tmp, f).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        result
    }
}
