//! Strict function comparator (DESIGN.md "Strict comparator").
//!
//! Exact means:
//! - same size and same instruction words after masking relocated fields
//!   (branch targets inside the function compare as plain displacement bits; a relocated
//!   branch to a label inside the same function, as dtk emits for some asm, is first turned
//!   back into a plain branch);
//! - same relocations per instruction (instruction index + kind; the byte offset convention for
//!   `EMB_SDA21` differs between dtk (+0) and MWCC (+2), so offsets compare at word granularity);
//! - relocation targets equivalent: same symbol name (and addend) for ordinary symbols; for
//!   compiler-generated / splitter-generated names (`@123`, `@stringBase0`, `lbl_...`,
//!   `...data.0`, `fn_...`, `init$12`, bare sections) the referenced **bytes** must be equal
//!   (string pools: the NUL-terminated string at symbol+addend), in the same section, with
//!   equivalent relocations inside the referenced data (jump tables, pointer tables); bss
//!   targets only need the same section; unnamed code targets compare by masked code, except a
//!   target placeholder (`fn_<addr>`) against one of our named functions, which needs a proof by
//!   bytes ([`crate::placeholder`]);
//! - the mapping between target and our generated targets is one-to-one within the function
//!   (two different target literals can't both be one of ours, and vice versa).
//!
//! Literals the target object only references (dtk put them in another split, e.g. the
//! auto-generated `.sdata2`/`.rodata` units) are resolved through an [`ExternIndex`] built
//! from the other target objects of the module.
use mwdec_core::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Global symbols defined by a set of objects (e.g. every target object of a module), for
/// resolving the values of literals that a target object only references.
///
/// An index can be layered on a shared base ([`ExternIndex::layered`]): a REL module's index
/// reuses the main module's (built once) instead of re-indexing every main object per module.
#[derive(Default)]
pub struct ExternIndex {
    /// Every indexed object: the base's objects first, then this layer's.
    pub objs: Vec<Arc<ObjectFile>>,
    /// name -> (object, symbol, weak); `None` = defined globally more than once (ambiguous).
    by_name: HashMap<String, Option<(u32, u32, bool)>>,
    /// name -> (object, function); `None` = ambiguous.
    funcs: HashMap<String, Option<(u32, u32)>>,
    base: Option<Arc<ExternIndex>>,
}

impl ExternIndex {
    /// Index the non-local symbols of `objs` (names defined in several objects are dropped as ambiguous).
    pub fn new(objs: Vec<Arc<ObjectFile>>) -> Self {
        Self::build(objs, None)
    }

    /// `base`'s objects followed by `extra`, with the same lookups as `ExternIndex::new` over the
    /// concatenation, without re-indexing `base`.
    pub fn layered(base: Arc<ExternIndex>, extra: Vec<Arc<ObjectFile>>) -> Self {
        Self::build(extra, Some(base))
    }

    fn build(extra: Vec<Arc<ObjectFile>>, base: Option<Arc<ExternIndex>>) -> Self {
        let start = base.as_ref().map_or(0, |b| b.objs.len());
        let mut objs: Vec<Arc<ObjectFile>> = base.as_ref().map(|b| b.objs.clone()).unwrap_or_default();
        objs.extend(extra);
        // Weak duplicates (inline/template copies) are interchangeable: keep the first. A name
        // defined globally in several objects is ambiguous and dropped.
        let mut by_name: HashMap<String, Option<(u32, u32, bool)>> = HashMap::new();
        let mut funcs: HashMap<String, Option<(u32, u32)>> = HashMap::new();
        for (oi, o) in objs.iter().enumerate().skip(start) {
            for (si, s) in o.symbols.iter().enumerate() {
                if s.binding == SymBinding::Local {
                    continue;
                }
                let weak = s.binding == SymBinding::Weak;
                let here = Some((oi as u32, si as u32, weak));
                let prev = match by_name.get(&s.name) {
                    Some(v) => Some(*v),
                    None => base.as_ref().and_then(|b| b.raw_symbol(&s.name)),
                };
                let next = match prev {
                    None => here,
                    Some(Some((_, _, true))) if weak => continue,
                    Some(Some((_, _, true))) => here,
                    Some(_) if weak => continue,
                    Some(_) => None,
                };
                by_name.insert(s.name.clone(), next);
            }
            for (fi, f) in o.functions.iter().enumerate() {
                let here = Some((oi as u32, fi as u32));
                let prev = match funcs.get(&f.name) {
                    Some(v) => Some(*v),
                    None => base.as_ref().and_then(|b| b.raw_func(&f.name)),
                };
                let next = match (f.binding, prev) {
                    (SymBinding::Local, _) => continue,
                    (SymBinding::Weak, Some(_)) => continue,
                    (SymBinding::Weak, None) => here,
                    (SymBinding::Global, Some(_)) => None,
                    (SymBinding::Global, None) => here,
                };
                funcs.insert(f.name.clone(), next);
            }
        }
        by_name.shrink_to_fit();
        funcs.shrink_to_fit();
        ExternIndex { objs, by_name, funcs, base }
    }

    fn raw_symbol(&self, name: &str) -> Option<Option<(u32, u32, bool)>> {
        match self.by_name.get(name) {
            Some(v) => Some(*v),
            None => self.base.as_ref()?.raw_symbol(name),
        }
    }

    fn raw_func(&self, name: &str) -> Option<Option<(u32, u32)>> {
        match self.funcs.get(name) {
            Some(v) => Some(*v),
            None => self.base.as_ref()?.raw_func(name),
        }
    }

    /// A non-local function defined in one of the indexed objects.
    pub fn function(&self, name: &str) -> Option<&Function> {
        let (oi, fi) = self.raw_func(name)??;
        Some(&self.objs[oi as usize].functions[fi as usize])
    }

    /// Unambiguous names in this layer (plus the base's).
    pub fn len(&self) -> usize {
        self.by_name.values().filter(|v| v.is_some()).count() + self.base.as_ref().map_or(0, |b| b.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A non-local symbol defined in one of the indexed objects: its symbol entry, containing
    /// section, and the section bytes from the symbol on (at most `max`).
    pub fn defined(&self, name: &str, max: usize) -> Option<(&SymbolDef, &Section, &[u8])> {
        let (sec, addr, _, d) = self.locate(name)?;
        let start = (addr as usize).min(sec.bytes.len());
        let end = start.saturating_add(max).min(sec.bytes.len());
        Some((d?, sec, &sec.bytes[start..end]))
    }

    fn locate(&self, name: &str) -> Option<(&Section, u32, u32, Option<&SymbolDef>)> {
        let (oi, si, _) = self.raw_symbol(name)??;
        let o = &self.objs[oi as usize];
        let d = &o.symbols[si as usize];
        let sec = o.sections.iter().find(|s| s.name == d.section)?;
        Some((sec, d.address, d.size, Some(d)))
    }
}

/// Name lookups over one object, built once and reused across many comparisons.
pub struct ObjIndex<'a> {
    pub obj: &'a ObjectFile,
    by_name: HashMap<&'a str, &'a SymbolDef>,
    sections: HashMap<&'a str, &'a Section>,
    funcs_by_sec: HashMap<&'a str, Vec<&'a SymbolDef>>,
    functions: HashMap<&'a str, &'a Function>,
    externs: Option<&'a ExternIndex>,
    /// Our side only: compiles definitions of functions a target placeholder may be.
    prover: Option<&'a dyn crate::placeholder::PlaceholderProver>,
}

type Loc<'a> = (&'a Section, u32, u32, Option<&'a SymbolDef>);

impl<'a> ObjIndex<'a> {
    pub fn new(obj: &'a ObjectFile) -> Self {
        let mut by_name = HashMap::new();
        let mut funcs_by_sec: HashMap<&str, Vec<&SymbolDef>> = HashMap::new();
        for s in &obj.symbols {
            by_name.entry(s.name.as_str()).or_insert(s);
            if s.is_func {
                funcs_by_sec.entry(s.section.as_str()).or_default().push(s);
            }
        }
        for v in funcs_by_sec.values_mut() {
            v.sort_by_key(|s| s.address);
        }
        let sections = obj.sections.iter().map(|s| (s.name.as_str(), s)).collect();
        let functions = obj.functions.iter().map(|f| (f.name.as_str(), f)).collect();
        ObjIndex { obj, by_name, sections, funcs_by_sec, functions, externs: None, prover: None }
    }

    /// Resolve target placeholders against our named functions through `p` (see
    /// [`crate::placeholder`]).
    pub fn with_prover(mut self, p: Option<&'a dyn crate::placeholder::PlaceholderProver>) -> Self {
        self.prover = p;
        self
    }

    /// Like [`ObjIndex::new`], resolving symbols undefined here through `externs`.
    pub fn with_externs(obj: &'a ObjectFile, externs: &'a ExternIndex) -> Self {
        let mut i = Self::new(obj);
        i.externs = Some(externs);
        i
    }

    pub fn function(&self, name: &str) -> Option<&'a Function> {
        self.functions.get(name).copied()
    }

    /// (section, section-relative address, declared size, symbol) of a defined symbol, a bare
    /// section name, or (through the extern index) a symbol defined in another object.
    fn locate(&self, name: &str) -> Option<Loc<'a>> {
        if let Some(d) = self.by_name.get(name) {
            let sec = self.sections.get(d.section.as_str())?;
            return Some((sec, d.address, d.size, Some(d)));
        }
        if let Some(s) = self.sections.get(name) {
            return Some((*s, 0, s.size, None));
        }
        self.externs.and_then(|e| e.locate(name))
    }

    /// If `name+addend` lies in code of this object, the containing function and offset in it.
    fn code_location(&self, name: &str, addend: i64) -> Option<(&'a str, i64)> {
        let d = self.by_name.get(name).copied();
        let (sec, addr) = match d {
            Some(d) => (*self.sections.get(d.section.as_str())?, d.address),
            None => (*self.sections.get(name)?, 0),
        };
        if !sec.executable {
            return None;
        }
        let at = addr as i64 + addend;
        let funcs = self.funcs_by_sec.get(sec.name.as_str())?;
        let i = funcs.partition_point(|f| (f.address as i64) <= at);
        let f = funcs.get(i.checked_sub(1)?)?;
        if at < f.address as i64 + f.size.max(1) as i64 {
            Some((f.name.as_str(), at - f.address as i64))
        } else {
            None
        }
    }
}

impl<'a> ObjIndex<'a> {
    /// The function containing `name+addend` (here, or an external one via the extern index).
    fn func_view(&self, name: &str, addend: i64) -> Option<(&'a Function, i64)> {
        if let Some((f, off)) = self.code_location(name, addend) {
            return Some((self.function(f)?, off));
        }
        if self.by_name.contains_key(name) {
            return None;
        }
        Some((self.externs?.function(name)?, addend))
    }

    /// Canonical identity of a reloc target for the one-to-one mapping check.
    fn canon(&self, name: &str, addend: i64) -> (String, i64) {
        match self.code_location(name, addend) {
            Some((f, off)) => (strip_dtk_suffix(f).to_string(), off),
            None => (strip_dtk_suffix(name).to_string(), addend),
        }
    }
}

/// Names the compiler/splitter invents; equality of such names says nothing about the value.
pub fn is_generated_name(n: &str) -> bool {
    n.starts_with('@')
        || n.starts_with("lbl_")
        || n.starts_with("fn_")
        || n.starts_with("...")
        || n.starts_with('.')
        || n.starts_with("jumptable_")
        || n.starts_with("gap_")
        || n.starts_with("switch_")
        || has_numbered_suffix(n)
}

/// `name$123` (function-local statics, compiler helpers), optionally followed by dtk's
/// `_80412345` address disambiguation suffix.
fn has_numbered_suffix(n: &str) -> bool {
    let n = strip_dtk_suffix(n);
    match n.rfind('$') {
        Some(i) => i + 1 < n.len() && n[i + 1..].bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

/// dtk appends `_XXXXXXXX` (address) to disambiguate duplicate local names.
fn strip_dtk_suffix(n: &str) -> &str {
    if n.len() > 9 {
        let (a, b) = n.split_at(n.len() - 9);
        let generated_base = matches!(a, "lbl" | "fn" | "func" | "jumptable" | "switch" | "gap");
        if !generated_base && b.starts_with('_') && b[1..].bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_lowercase()) {
            return a;
        }
    }
    n
}

fn is_string_pool(n: &str) -> bool {
    n.starts_with("@stringBase")
}

fn is_wstring_pool(n: &str) -> bool {
    n.starts_with("@wstringBase")
}

/// Bits of an instruction word covered by a relocation of this kind.
pub fn reloc_mask(kind: RelocKind) -> u32 {
    match kind {
        RelocKind::Addr16Lo | RelocKind::Addr16Hi | RelocKind::Addr16Ha => 0x0000_ffff,
        // rA (r0/r2/r13 chosen by the linker) + 16-bit displacement
        RelocKind::EmbSda21 => 0x001f_ffff,
        RelocKind::Rel24 => 0x03ff_fffc,
        RelocKind::Rel14 => 0x0000_fffc,
        RelocKind::Addr32 | RelocKind::Other(_) => 0xffff_ffff,
    }
}

/// Instruction words with relocated fields zeroed.
pub fn masked_words(f: &Function) -> Vec<u32> {
    masked(&f.words().collect::<Vec<_>>(), &f.relocs)
}

fn masked(words: &[u32], relocs: &[Reloc]) -> Vec<u32> {
    let mut w = words.to_vec();
    for r in relocs {
        let i = (r.offset / 4) as usize;
        if i < w.len() {
            w[i] &= !reloc_mask(r.kind);
        }
    }
    w
}

/// Turn relocated branches to labels inside the function itself back into plain branches.
fn internalize(idx: &ObjIndex, f: &Function) -> (Vec<u32>, Vec<Reloc>) {
    let mut words: Vec<u32> = f.words().collect();
    let mut relocs = Vec::with_capacity(f.relocs.len());
    for r in &f.relocs {
        if matches!(r.kind, RelocKind::Rel24 | RelocKind::Rel14) {
            if let Some((fname, dest)) = idx.code_location(&r.target, r.addend) {
                let i = (r.offset / 4) as usize;
                if fname == f.name && i < words.len() {
                    let disp = (dest - (i as i64) * 4) as u32;
                    let m = reloc_mask(r.kind);
                    words[i] = (words[i] & !m) | (disp & m);
                    continue;
                }
            }
        }
        relocs.push(r.clone());
    }
    (words, relocs)
}

/// Why a comparison is not exact. Ordered from most to least fundamental.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiffClass {
    Exact,
    /// Function sizes differ.
    Size,
    /// Masked instruction words differ.
    Code,
    /// Relocation positions or kinds differ.
    RelocLayout,
    /// A relocation targets a differently named (non-generated) symbol, or a different addend.
    TargetName,
    /// A generated target could not be resolved to bytes on one side (comparator limitation
    /// or a genuinely different kind of target).
    Unresolved,
    /// Generated targets resolved on both sides but their bytes/section differ, or the
    /// target<->ours mapping is not one-to-one.
    Literal,
}

impl DiffClass {
    pub fn label(self) -> &'static str {
        match self {
            DiffClass::Exact => "exact",
            DiffClass::Size => "size",
            DiffClass::Code => "code",
            DiffClass::RelocLayout => "reloc-layout",
            DiffClass::TargetName => "target-name",
            DiffClass::Unresolved => "unresolved",
            DiffClass::Literal => "literal",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Detailed {
    pub result: CompareResult,
    /// Most fundamental difference found (`Exact` iff `result.exact`).
    pub class: DiffClass,
    /// Every difference class seen.
    pub classes: Vec<DiffClass>,
}

enum TEq {
    Same,
    Diff(DiffClass, String),
}

fn hex(b: &[u8]) -> String {
    let mut s: String = b.iter().take(16).map(|x| format!("{x:02x}")).collect();
    if b.len() > 16 {
        s.push_str("..");
    }
    s
}

fn fmt_target(n: &str, a: i64) -> String {
    match a {
        0 => n.to_string(),
        a if a > 0 => format!("{n}+0x{a:x}"),
        a => format!("{n}-0x{:x}", -a),
    }
}

fn read(sec: &Section, start: i64, len: usize) -> Option<&[u8]> {
    if start < 0 || sec.bytes.is_empty() {
        return None;
    }
    let s = start as usize;
    let e = s.checked_add(len)?;
    sec.bytes.get(s..e)
}

/// NUL-terminated string at `start` (units of `width` bytes), without the terminator.
fn c_str(sec: &Section, start: i64, width: usize) -> Option<&[u8]> {
    if start < 0 {
        return None;
    }
    let b = sec.bytes.get(start as usize..)?;
    let n = b.chunks_exact(width).position(|c| c.iter().all(|&x| x == 0))?;
    Some(&b[..n * width])
}

fn relocs_in(sec: &Section, start: i64, len: usize) -> Vec<&Reloc> {
    let lo = sec.relocs.partition_point(|r| (r.offset as i64) < start);
    sec.relocs[lo..].iter().take_while(|r| (r.offset as i64) < start + len as i64).collect()
}

fn targets_equiv(t: &ObjIndex, tn: &str, ta: i64, o: &ObjIndex, on: &str, oa: i64, depth: u32) -> TEq {
    let desc = || format!("target {} vs ours {}", fmt_target(tn, ta), fmt_target(on, oa));
    let generated = is_generated_name(tn) || is_generated_name(on);
    // Code compares as (function, offset); unnamed functions (`fn_...`) by their masked code.
    match (t.func_view(tn, ta), o.func_view(on, oa)) {
        (Some((a, to)), Some((b, oo))) => {
            if to != oo {
                return TEq::Diff(DiffClass::TargetName, format!("code offset {}", desc()));
            }
            if strip_dtk_suffix(&a.name) == strip_dtk_suffix(&b.name) {
                return TEq::Same;
            }
            // a target placeholder vs one of our named functions: proven by compiling ours
            if crate::placeholder::is_placeholder(strip_dtk_suffix(&a.name)) && !is_generated_name(&b.name) {
                return if prove_placeholder(t, a, o, Some(b), &b.name) {
                    TEq::Same
                } else {
                    TEq::Diff(DiffClass::TargetName, format!("placeholder not proven: {}", desc()))
                };
            }
            if is_generated_name(&a.name) || is_generated_name(&b.name) {
                let same = a.code.len() == b.code.len()
                    && masked_words(a) == masked_words(b)
                    && a.relocs.len() == b.relocs.len()
                    && a.relocs.iter().zip(&b.relocs).all(|(x, y)| x.offset / 4 == y.offset / 4 && x.kind == y.kind);
                return if same {
                    TEq::Same
                } else {
                    TEq::Diff(DiffClass::Literal, format!("unnamed function code differs: {}", desc()))
                };
            }
            return TEq::Diff(DiffClass::TargetName, format!("code {}", desc()));
        }
        (Some((a, 0)), None) if oa == 0 && crate::placeholder::is_placeholder(strip_dtk_suffix(&a.name)) && !is_generated_name(on) && o.prover.is_some() => {
            return if prove_placeholder(t, a, o, None, on) {
                TEq::Same
            } else {
                TEq::Diff(DiffClass::TargetName, format!("placeholder not proven: {}", desc()))
            };
        }
        (Some(_), None) | (None, Some(_)) if generated && strip_dtk_suffix(tn) != strip_dtk_suffix(on) => {
            return TEq::Diff(
                DiffClass::Unresolved,
                format!("unnamed function vs function not defined on the other side: {}", desc()),
            );
        }
        _ => {}
    }
    if !generated && strip_dtk_suffix(tn) == strip_dtk_suffix(on) {
        return if ta == oa { TEq::Same } else { TEq::Diff(DiffClass::TargetName, format!("addend {}", desc())) };
    }
    let tl = t.locate(tn);
    let ol = o.locate(on);
    let (Some((tsec, taddr, tsize, tdef)), Some((osec, oaddr, osize, odef))) = (tl, ol) else {
        if tn == on && ta == oa {
            return TEq::Same; // same generated name, undefined on a side: nothing more to check
        }
        return if generated && (tl.is_some() || ol.is_some()) {
            TEq::Diff(DiffClass::Unresolved, format!("undefined on one side: {}", desc()))
        } else {
            TEq::Diff(DiffClass::TargetName, desc())
        };
    };
    // Both defined. Only compare by value if a side is generated, or both are local with the
    // same name modulo dtk's address suffix.
    let local = |d: Option<&SymbolDef>| d.map_or(true, |d| d.binding == SymBinding::Local);
    let same_local = local(tdef) && local(odef) && strip_dtk_suffix(tn) == strip_dtk_suffix(on);
    if !generated && !same_local {
        return TEq::Diff(DiffClass::TargetName, desc());
    }
    if tsec.name != osec.name {
        return TEq::Diff(DiffClass::Literal, format!("section {} vs {}: {}", tsec.name, osec.name, desc()));
    }
    let tstart = taddr as i64 + ta;
    let ostart = oaddr as i64 + oa;
    for (pool, width) in [(is_string_pool as fn(&str) -> bool, 1usize), (is_wstring_pool, 2)] {
        if pool(tn) || pool(on) {
            return match (c_str(tsec, tstart, width), c_str(osec, ostart, width)) {
                (Some(a), Some(b)) if a == b => TEq::Same,
                (Some(a), Some(b)) if width == 1 => TEq::Diff(
                    DiffClass::Literal,
                    format!("string {:?} vs {:?}: {}", String::from_utf8_lossy(a), String::from_utf8_lossy(b), desc()),
                ),
                (Some(a), Some(b)) => {
                    TEq::Diff(DiffClass::Literal, format!("wide string {} vs {}: {}", hex(a), hex(b), desc()))
                }
                _ => TEq::Diff(DiffClass::Unresolved, format!("no string: {}", desc())),
            };
        }
    }
    if tsec.bytes.is_empty() || osec.bytes.is_empty() {
        // bss-like: no values. Same section suffices (dtk's sizes of unnamed bss are guesses);
        // the one-to-one mapping check catches swapped variables.
        return if tsec.bytes.is_empty() && osec.bytes.is_empty() {
            TEq::Same
        } else {
            TEq::Diff(DiffClass::Literal, format!("bss vs data: {}", desc()))
        };
    }
    // Extent of the referenced object: ours first (MWCC sizes literals exactly), else target's.
    let len = if (osize as i64) > oa {
        (osize as i64 - oa) as usize
    } else if (tsize as i64) > ta {
        (tsize as i64 - ta) as usize
    } else {
        4
    };
    let (Some(tb), Some(ob)) = (read(tsec, tstart, len), read(osec, ostart, len)) else {
        return TEq::Diff(DiffClass::Unresolved, format!("cannot read {len} bytes: {}", desc()));
    };
    if tb != ob {
        return TEq::Diff(DiffClass::Literal, format!("bytes {} vs {}: {}", hex(tb), hex(ob), desc()));
    }
    // Relocations inside the referenced data (jump tables, pointer tables). Nested references
    // resolve in the objects that hold the data.
    let trs = relocs_in(tsec, tstart, len);
    let ors = relocs_in(osec, ostart, len);
    if trs.len() != ors.len() {
        return TEq::Diff(DiffClass::Literal, format!("data relocs {} vs {}: {}", trs.len(), ors.len(), desc()));
    }
    for (a, b) in trs.iter().zip(&ors) {
        if a.offset as i64 - tstart != b.offset as i64 - ostart || a.kind != b.kind {
            return TEq::Diff(DiffClass::Literal, format!("data reloc layout: {}", desc()));
        }
        if depth < 2 {
            if let TEq::Diff(c, m) = targets_equiv(t, &a.target, a.addend, o, &b.target, b.addend, depth + 1) {
                return TEq::Diff(c, format!("in {}: {m}", desc()));
            }
        } else if a.target != b.target || a.addend != b.addend {
            return TEq::Diff(DiffClass::Unresolved, format!("nested reloc too deep: {}", desc()));
        }
    }
    TEq::Same
}

/// Is the target placeholder function `tf` our function `symbol`? Ours is `local` (defined in our
/// object) or compiled by the prover; the two must compare exact with the strict comparator.
fn prove_placeholder(t: &ObjIndex, tf: &Function, o: &ObjIndex, local: Option<&Function>, symbol: &str) -> bool {
    if !crate::placeholder::enabled() {
        return false;
    }
    let key = (strip_dtk_suffix(&tf.name).to_string(), symbol.to_string());
    if let Some(p) = o.prover {
        if let Some(v) = crate::placeholder::remembered(p, &key) {
            return v;
        }
    }
    let verdict = match local {
        Some(of) => crate::placeholder::nested(|| {
            let d = compare_indexed(t, tf, o, of);
            if std::env::var("MWDEC_PLACEHOLDER_DEBUG").is_ok() {
                eprintln!("placeholder {} vs local {symbol}: {:?}", tf.name, d.result.notes);
            }
            d.result.exact
        })
        .unwrap_or(false),
        None => {
            let Some(p) = o.prover else { return false };
            let Some(obj) = p.definition(symbol) else {
                if std::env::var("MWDEC_PLACEHOLDER_DEBUG").is_ok() {
                    eprintln!("placeholder {} vs {symbol}: no definition", tf.name);
                }
                crate::placeholder::remember(p, key, false);
                return false;
            };
            let Some(of) = obj.functions.iter().find(|f| f.name == symbol) else { return false };
            let mut oi = ObjIndex::new(&obj).with_prover(Some(p));
            oi.externs = o.externs;
            crate::placeholder::nested(|| {
                let d = compare_indexed(t, tf, &oi, of);
                if std::env::var("MWDEC_PLACEHOLDER_DEBUG").is_ok() {
                    eprintln!("placeholder {} vs {symbol}: {:?}", tf.name, d.result.notes);
                }
                d.result.exact
            })
            .unwrap_or(false)
        }
    };
    if let Some(p) = o.prover {
        crate::placeholder::remember(p, key, verdict);
    }
    verdict
}

/// Longest common subsequence length (two-row DP; positional fallback for huge inputs).
fn lcs(a: &[u32], b: &[u32]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    if a.len() * b.len() > 16_000_000 {
        return a.iter().zip(b).filter(|(x, y)| x == y).count();
    }
    let mut prev = vec![0u32; b.len() + 1];
    let mut cur = vec![0u32; b.len() + 1];
    for &x in a {
        for (j, &y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { cur[j].max(prev[j + 1]) };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] as usize
}

const MAX_NOTES: usize = 12;

fn note(notes: &mut Vec<String>, s: String) {
    if notes.len() < MAX_NOTES {
        notes.push(s);
    }
}

/// Strict comparison with difference classification, using prebuilt indexes.
pub fn compare_indexed(t: &ObjIndex, tf: &Function, o: &ObjIndex, of: &Function) -> Detailed {
    let mut notes = Vec::new();
    let mut classes = Vec::new();
    let (traw, trel) = internalize(t, tf);
    let (oraw, orel) = internalize(o, of);
    let tw = masked(&traw, &trel);
    let ow = masked(&oraw, &orel);
    if tf.code.len() != of.code.len() {
        classes.push(DiffClass::Size);
        note(&mut notes, format!("size: target {:#x} vs ours {:#x}", tf.code.len(), of.code.len()));
    }
    let mut code_diffs = 0usize;
    for (i, (a, b)) in tw.iter().zip(&ow).enumerate() {
        if a != b {
            code_diffs += 1;
            if code_diffs == 1 {
                classes.push(DiffClass::Code);
            }
            if code_diffs <= 4 {
                let off = (i * 4) as u32;
                note(
                    &mut notes,
                    format!(
                        "code @{off:#x}: target `{}` vs ours `{}`",
                        mwdec_obj::disasm_word(traw[i], off),
                        mwdec_obj::disasm_word(oraw[i], off)
                    ),
                );
            }
        }
    }
    if code_diffs > 4 {
        note(&mut notes, format!("... {code_diffs} differing instructions in total"));
    }
    // Relocations, matched per instruction.
    let mut reloc_ok = 0usize;
    let nrel = trel.len().max(orel.len());
    let mut layout_bad = trel.len() != orel.len();
    let mut fwd: HashMap<(String, i64), (String, i64)> = HashMap::new();
    let mut rev: HashMap<(String, i64), (String, i64)> = HashMap::new();
    for (a, b) in trel.iter().zip(&orel) {
        if a.offset / 4 != b.offset / 4 || a.kind != b.kind {
            note(
                &mut notes,
                format!(
                    "reloc layout: target @{:#x} {:?} {} vs ours @{:#x} {:?} {}",
                    a.offset, a.kind, a.target, b.offset, b.kind, b.target
                ),
            );
            layout_bad = true;
            continue;
        }
        let ins = a.offset & !3;
        // One-to-one mapping of targets within the function.
        let tk = t.canon(&a.target, a.addend);
        let ok = o.canon(&b.target, b.addend);
        let f = fwd.entry(tk.clone()).or_insert_with(|| ok.clone()).clone();
        let r = rev.entry(ok.clone()).or_insert_with(|| tk.clone()).clone();
        if f != ok || r != tk {
            classes.push(DiffClass::Literal);
            note(
                &mut notes,
                format!(
                    "reloc @{ins:#x}: mapping not one-to-one: target {} vs ours {} (elsewhere {} <-> {})",
                    fmt_target(&tk.0, tk.1),
                    fmt_target(&ok.0, ok.1),
                    fmt_target(&r.0, r.1),
                    fmt_target(&f.0, f.1)
                ),
            );
            continue;
        }
        match targets_equiv(t, &a.target, a.addend, o, &b.target, b.addend, 0) {
            TEq::Same => reloc_ok += 1,
            TEq::Diff(c, m) => {
                classes.push(c);
                note(&mut notes, format!("reloc @{ins:#x} {:?}: {m}", a.kind));
            }
        }
    }
    if layout_bad {
        classes.push(DiffClass::RelocLayout);
        if trel.len() != orel.len() {
            note(&mut notes, format!("reloc count: target {} vs ours {}", trel.len(), orel.len()));
        }
    }
    classes.sort();
    classes.dedup();
    let exact = classes.is_empty();
    let score = if exact {
        100.0
    } else {
        let l = lcs(&tw, &ow) as f64;
        let base = 2.0 * l / (tw.len() + ow.len()).max(1) as f64;
        let rel = if nrel == 0 { 1.0 } else { reloc_ok as f64 / nrel as f64 };
        (100.0 * base * (0.9 + 0.1 * rel)).min(99.9)
    };
    let class = classes.first().copied().unwrap_or(DiffClass::Exact);
    Detailed {
        result: CompareResult {
            exact,
            score,
            target_len: tf.code.len() as u32,
            ours_len: of.code.len() as u32,
            notes,
        },
        class,
        classes,
    }
}

/// Strict comparison with classification (builds indexes; prefer [`compare_indexed`] in loops,
/// and [`ObjIndex::with_externs`] for target objects whose literals live in other splits).
pub fn compare_detailed(target: &ObjectFile, target_fn: &Function, ours: &ObjectFile, ours_fn: &Function) -> Detailed {
    compare_indexed(&ObjIndex::new(target), target_fn, &ObjIndex::new(ours), ours_fn)
}

/// Strict comparison of one function (ours vs target), resolving literal values via each object's data.
pub fn compare(target: &ObjectFile, target_fn: &Function, ours: &ObjectFile, ours_fn: &Function) -> CompareResult {
    compare_detailed(target, target_fn, ours, ours_fn).result
}

