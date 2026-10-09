//! On-disk template cache: a probe's text (its signature and call) determines its template, so
//! templates are shared by every unit and run (keyed by probe text + compiler + flags). Mirror
//! types make the IR serialisable without touching mwdec-lift.

use crate::probe::CallKind;
use crate::template::{Comp, HoleKind, Shape, Template};
use mwdec_core::{FuncSig, Type};
use mwdec_lift::{BinOp, Callee, Expr, Stmt, UnOp};
use serde::{Deserialize, Serialize};

const BINOPS: [BinOp; 18] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Div,
    BinOp::Rem,
    BinOp::And,
    BinOp::Or,
    BinOp::Xor,
    BinOp::Shl,
    BinOp::Shr,
    BinOp::Eq,
    BinOp::Ne,
    BinOp::Lt,
    BinOp::Le,
    BinOp::Gt,
    BinOp::Ge,
    BinOp::LogAnd,
    BinOp::LogOr,
];
const UNOPS: [UnOp; 3] = [UnOp::Neg, UnOp::BitNot, UnOp::Not];

#[derive(Serialize, Deserialize)]
enum SCallee {
    Direct(String, FuncSig),
    Method(String, FuncSig, Box<SExpr>, bool),
    Virtual(Box<SExpr>, u32, u32, Option<String>, Option<FuncSig>),
    Indirect(Box<SExpr>),
}

#[derive(Serialize, Deserialize)]
enum SExpr {
    Var(usize),
    Int(i64, Type),
    Float(u64, bool),
    Str(Vec<u8>),
    Global(String, Type),
    FuncAddr(String),
    AddrOf(Box<SExpr>),
    Load(Box<SExpr>, i32, Type),
    Index(Box<SExpr>, Box<SExpr>, Type),
    Member(Box<SExpr>, i32, Type),
    Unary(u8, Box<SExpr>, Type),
    Binary(u8, Box<SExpr>, Box<SExpr>, Type),
    Cast(Type, Box<SExpr>),
    Call(SCallee, Vec<SExpr>, Type),
    Ternary(Box<SExpr>, Box<SExpr>, Box<SExpr>, Type),
    Unknown(String, Type),
    New(Type, Vec<SExpr>, Option<FuncSig>, Vec<SExpr>),
    Construct(Type, Option<FuncSig>, Vec<SExpr>),
    BitField(Box<SExpr>, u8, u8, Type),
    IncDec(Box<SExpr>, i64, bool),
}

fn bx(e: &Expr) -> Box<SExpr> {
    Box::new(se(e))
}

fn se(e: &Expr) -> SExpr {
    match e {
        Expr::Var(v) => SExpr::Var(*v),
        Expr::Int { value, ty } => SExpr::Int(*value, ty.clone()),
        Expr::Float { bits, double } => SExpr::Float(*bits, *double),
        Expr::Str { bytes } => SExpr::Str(bytes.clone()),
        Expr::Global { symbol, ty } => SExpr::Global(symbol.clone(), ty.clone()),
        Expr::FuncAddr { symbol } => SExpr::FuncAddr(symbol.clone()),
        Expr::AddrOf(x) => SExpr::AddrOf(bx(x)),
        Expr::Load { base, offset, ty } => SExpr::Load(bx(base), *offset, ty.clone()),
        Expr::Index { base, index, ty } => SExpr::Index(bx(base), bx(index), ty.clone()),
        Expr::Member { base, offset, ty } => SExpr::Member(bx(base), *offset, ty.clone()),
        Expr::Unary { op, e, ty } => SExpr::Unary(UNOPS.iter().position(|o| o == op).unwrap() as u8, bx(e), ty.clone()),
        Expr::Binary { op, l, r, ty } => SExpr::Binary(BINOPS.iter().position(|o| o == op).unwrap() as u8, bx(l), bx(r), ty.clone()),
        Expr::Cast { ty, e } => SExpr::Cast(ty.clone(), bx(e)),
        Expr::Call { callee, args, ret } => SExpr::Call(
            match callee {
                Callee::Direct { symbol, sig } => SCallee::Direct(symbol.clone(), sig.clone()),
                Callee::Method { symbol, sig, this, qualified } => SCallee::Method(symbol.clone(), sig.clone(), bx(this), *qualified),
                Callee::Virtual { this, vtable_offset, vptr_offset, class, sig } => SCallee::Virtual(bx(this), *vtable_offset, *vptr_offset, class.clone(), sig.clone()),
                Callee::Indirect(x) => SCallee::Indirect(bx(x)),
            },
            args.iter().map(se).collect(),
            ret.clone(),
        ),
        Expr::Ternary { c, t, f, ty } => SExpr::Ternary(bx(c), bx(t), bx(f), ty.clone()),
        Expr::Unknown { text, ty } => SExpr::Unknown(text.clone(), ty.clone()),
        Expr::New { class, placement, ctor, args } => SExpr::New(class.clone(), placement.iter().map(se).collect(), ctor.clone(), args.iter().map(se).collect()),
        Expr::Construct { class, ctor, args } => SExpr::Construct(class.clone(), ctor.clone(), args.iter().map(se).collect()),
        Expr::BitField { base, shift, width, ty } => SExpr::BitField(bx(base), *shift, *width, ty.clone()),
        Expr::IncDec { e, delta, post } => SExpr::IncDec(bx(e), *delta, *post),
    }
}

fn db(e: &SExpr) -> Box<Expr> {
    Box::new(de(e))
}

fn de(e: &SExpr) -> Expr {
    match e {
        SExpr::Var(v) => Expr::Var(*v),
        SExpr::Int(value, ty) => Expr::Int { value: *value, ty: ty.clone() },
        SExpr::Float(bits, double) => Expr::Float { bits: *bits, double: *double },
        SExpr::Str(b) => Expr::Str { bytes: b.clone() },
        SExpr::Global(s, ty) => Expr::Global { symbol: s.clone(), ty: ty.clone() },
        SExpr::FuncAddr(s) => Expr::FuncAddr { symbol: s.clone() },
        SExpr::AddrOf(x) => Expr::AddrOf(db(x)),
        SExpr::Load(b, o, ty) => Expr::Load { base: db(b), offset: *o, ty: ty.clone() },
        SExpr::Index(b, i, ty) => Expr::Index { base: db(b), index: db(i), ty: ty.clone() },
        SExpr::Member(b, o, ty) => Expr::Member { base: db(b), offset: *o, ty: ty.clone() },
        SExpr::Unary(op, x, ty) => Expr::Unary { op: UNOPS[*op as usize], e: db(x), ty: ty.clone() },
        SExpr::Binary(op, l, r, ty) => Expr::Binary { op: BINOPS[*op as usize], l: db(l), r: db(r), ty: ty.clone() },
        SExpr::Cast(ty, x) => Expr::Cast { ty: ty.clone(), e: db(x) },
        SExpr::Call(c, args, ret) => Expr::Call {
            callee: match c {
                SCallee::Direct(s, sig) => Callee::Direct { symbol: s.clone(), sig: sig.clone() },
                SCallee::Method(s, sig, this, q) => Callee::Method { symbol: s.clone(), sig: sig.clone(), this: db(this), qualified: *q },
                SCallee::Virtual(this, vo, po, class, sig) => Callee::Virtual { this: db(this), vtable_offset: *vo, vptr_offset: *po, class: class.clone(), sig: sig.clone() },
                SCallee::Indirect(x) => Callee::Indirect(db(x)),
            },
            args: args.iter().map(de).collect(),
            ret: ret.clone(),
        },
        SExpr::Ternary(c, t, f, ty) => Expr::Ternary { c: db(c), t: db(t), f: db(f), ty: ty.clone() },
        SExpr::Unknown(text, ty) => Expr::Unknown { text: text.clone(), ty: ty.clone() },
        SExpr::New(class, p, ctor, a) => Expr::New { class: class.clone(), placement: p.iter().map(de).collect(), ctor: ctor.clone(), args: a.iter().map(de).collect() },
        SExpr::Construct(class, ctor, a) => Expr::Construct { class: class.clone(), ctor: ctor.clone(), args: a.iter().map(de).collect() },
        SExpr::BitField(b, s, w, ty) => Expr::BitField { base: db(b), shift: *s, width: *w, ty: ty.clone() },
        SExpr::IncDec(x, d, p) => Expr::IncDec { e: db(x), delta: *d, post: *p },
    }
}

#[derive(Serialize, Deserialize)]
enum SStmt {
    Assign(SExpr, SExpr),
    Expr(SExpr),
    If(SExpr, Vec<SStmt>, Vec<SStmt>),
    While(SExpr, Vec<SStmt>),
    DoWhile(Vec<SStmt>, SExpr),
}

fn ss(s: &Stmt) -> Option<SStmt> {
    Some(match s {
        Stmt::Assign { dst, src } => SStmt::Assign(se(dst), se(src)),
        Stmt::Expr(e) => SStmt::Expr(se(e)),
        Stmt::If { cond, then, els } => SStmt::If(se(cond), then.iter().map(ss).collect::<Option<_>>()?, els.iter().map(ss).collect::<Option<_>>()?),
        Stmt::While { cond, body } => SStmt::While(se(cond), body.iter().map(ss).collect::<Option<_>>()?),
        Stmt::DoWhile { body, cond } => SStmt::DoWhile(body.iter().map(ss).collect::<Option<_>>()?, se(cond)),
        _ => return None,
    })
}

fn ds(s: &SStmt) -> Stmt {
    match s {
        SStmt::Assign(d, s) => Stmt::Assign { dst: de(d), src: de(s) },
        SStmt::Expr(e) => Stmt::Expr(de(e)),
        SStmt::If(c, t, e) => Stmt::If { cond: de(c), then: t.iter().map(ds).collect(), els: e.iter().map(ds).collect() },
        SStmt::While(c, b) => Stmt::While { cond: de(c), body: b.iter().map(ds).collect() },
        SStmt::DoWhile(b, c) => Stmt::DoWhile { body: b.iter().map(ds).collect(), cond: de(c) },
    }
}

#[derive(Serialize, Deserialize)]
enum SHole {
    Scalar(Type),
    Obj(String, bool, bool),
    Local,
    ScalarRef(Type),
}

#[derive(Serialize, Deserialize)]
struct SComp(i32, Type, SExpr);

#[derive(Serialize, Deserialize)]
enum SShape {
    Scalar(SExpr),
    Object(String, Vec<SComp>),
    Mutate(usize, Vec<SComp>),
    Stmts(Vec<SStmt>, Option<SExpr>),
}

#[derive(Serialize, Deserialize)]
pub struct STemplate {
    name: String,
    kind: u8,
    sig: FuncSig,
    class: Option<String>,
    holes: Vec<SHole>,
    shape: SShape,
    ops: usize,
    ret_ref: bool,
    #[serde(default)]
    dead: Vec<(u32, SExpr)>,
    #[serde(default)]
    guessed: bool,
    #[serde(default)]
    fixed: Vec<(usize, SExpr)>,
}

/// A cache entry: the template, or why the probe gives none.
#[derive(Serialize, Deserialize)]
pub enum Entry {
    Ok(STemplate),
    Rejected(String),
}

pub fn encode(t: &Template) -> Option<Entry> {
    let comps = |cs: &[Comp]| cs.iter().map(|c| SComp(c.off, c.ty.clone(), se(&c.pat))).collect::<Vec<_>>();
    Some(Entry::Ok(STemplate {
        name: t.name.clone(),
        kind: match t.kind {
            CallKind::Method => 0,
            CallKind::Free => 1,
            CallKind::Ctor => 2,
        },
        sig: t.sig.clone(),
        class: t.class.clone(),
        holes: t
            .holes
            .iter()
            .map(|h| match h {
                HoleKind::Scalar(t) => SHole::Scalar(t.clone()),
                HoleKind::Obj { class, ptr, temp_ok } => SHole::Obj(class.clone(), *ptr, *temp_ok),
                HoleKind::Local => SHole::Local,
                HoleKind::ScalarRef(t) => SHole::ScalarRef(t.clone()),
            })
            .collect(),
        shape: match &t.shape {
            Shape::Scalar(e) => SShape::Scalar(se(e)),
            Shape::Object { class, comps: c } => SShape::Object(class.clone(), comps(c)),
            Shape::Mutate { hole, comps: c } => SShape::Mutate(*hole, comps(c)),
            Shape::Stmts { stmts, result } => SShape::Stmts(stmts.iter().map(ss).collect::<Option<_>>()?, result.as_ref().map(se)),
        },
        ops: t.ops,
        ret_ref: t.ret_ref,
        dead: t.dead.iter().map(|(n, e)| (*n, se(e))).collect(),
        guessed: t.guessed,
        fixed: t.fixed.iter().map(|(h, e)| (*h, se(e))).collect(),
    }))
}

pub fn decode(e: &Entry) -> Result<Template, String> {
    let t = match e {
        Entry::Ok(t) => t,
        Entry::Rejected(r) => return Err(r.clone()),
    };
    let comps = |cs: &[SComp]| cs.iter().map(|c| Comp { off: c.0, ty: c.1.clone(), pat: de(&c.2) }).collect::<Vec<_>>();
    Ok(Template {
        name: t.name.clone(),
        kind: match t.kind {
            0 => CallKind::Method,
            1 => CallKind::Free,
            _ => CallKind::Ctor,
        },
        sig: t.sig.clone(),
        class: t.class.clone(),
        holes: t
            .holes
            .iter()
            .map(|h| match h {
                SHole::Scalar(t) => HoleKind::Scalar(t.clone()),
                SHole::Obj(c, p, k) => HoleKind::Obj { class: c.clone(), ptr: *p, temp_ok: *k },
                SHole::Local => HoleKind::Local,
                SHole::ScalarRef(t) => HoleKind::ScalarRef(t.clone()),
            })
            .collect(),
        shape: match &t.shape {
            SShape::Scalar(e) => Shape::Scalar(de(e)),
            SShape::Object(c, cs) => Shape::Object { class: c.clone(), comps: comps(cs) },
            SShape::Mutate(h, cs) => Shape::Mutate { hole: *h, comps: comps(cs) },
            SShape::Stmts(st, r) => Shape::Stmts { stmts: st.iter().map(ds).collect(), result: r.as_ref().map(de) },
        },
        ops: t.ops,
        ret_ref: t.ret_ref,
        dead: t.dead.iter().map(|(n, e)| (*n, de(e))).collect(),
        guessed: t.guessed,
        fixed: t.fixed.iter().map(|(h, e)| (*h, de(e))).collect(),
    })
}

/// Disk cache of probe outcomes, keyed by probe text (without its numbered name) and a salt
/// (compiler + flags + generator version).
pub struct Cache {
    dir: std::path::PathBuf,
    salt: String,
}

impl Cache {
    pub fn new(dir: std::path::PathBuf, salt: String) -> Cache {
        let _ = std::fs::create_dir_all(&dir);
        Cache { dir, salt }
    }

    fn path(&self, probe_text: &str) -> std::path::PathBuf {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in self.salt.bytes().chain([0u8]).chain(probe_text.bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.dir.join(format!("{:02x}", h & 0xff)).join(format!("{h:016x}.json"))
    }

    pub fn get(&self, probe_text: &str) -> Option<Result<Template, String>> {
        let s = std::fs::read_to_string(self.path(probe_text)).ok()?;
        let (text, e): (String, Entry) = serde_json::from_str(&s).ok()?;
        if text != probe_text {
            return None;
        }
        Some(decode(&e))
    }

    pub fn put(&self, probe_text: &str, r: &Result<Template, String>) {
        let e = match r {
            Ok(t) => match encode(t) {
                Some(e) => e,
                None => return,
            },
            Err(why) => Entry::Rejected(why.clone()),
        };
        let p = self.path(probe_text);
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        if let Ok(s) = serde_json::to_string(&(probe_text, e)) {
            // write-then-rename so concurrent readers never see a partial file
            let tmp = p.with_extension(format!("tmp{}", std::process::id()));
            if std::fs::write(&tmp, s).is_ok() {
                let _ = std::fs::rename(&tmp, &p);
            }
        }
    }
}

/// Disk cache of compiled probe functions (a probe's text + compiler + flags decide its code,
/// whatever the lifter): templates are re-lifted from these after a lifter change without
/// compiling again. Each entry is a minimal object: the probe function (named `__P...`) and
/// the data it references.
pub struct ObjCache {
    dir: std::path::PathBuf,
    salt: String,
}

impl ObjCache {
    pub fn new(dir: std::path::PathBuf, salt: String) -> ObjCache {
        let _ = std::fs::create_dir_all(&dir);
        ObjCache { dir, salt }
    }

    fn path(&self, probe_text: &str) -> std::path::PathBuf {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in self.salt.bytes().chain([0u8]).chain(probe_text.bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.dir.join(format!("{:02x}", h & 0xff)).join(format!("{h:016x}.json"))
    }

    /// `Some(None)`: compiled, but the probe function wasn't in the object.
    pub fn get(&self, probe_text: &str) -> Option<Option<mwdec_core::ObjectFile>> {
        let s = std::fs::read_to_string(self.path(probe_text)).ok()?;
        let (text, o): (String, Option<mwdec_core::ObjectFile>) = serde_json::from_str(&s).ok()?;
        (text == probe_text).then_some(o)
    }

    pub fn put(&self, probe_text: &str, o: &Option<mwdec_core::ObjectFile>) {
        let p = self.path(probe_text);
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        if let Ok(s) = serde_json::to_string(&(probe_text, o)) {
            let tmp = p.with_extension(format!("tmp{}", std::process::id()));
            if std::fs::write(&tmp, s).is_ok() {
                let _ = std::fs::rename(&tmp, &p);
            }
        }
    }
}

/// The minimal object of one function of `obj`: the function renamed `prefix` -> `__P`, the
/// data symbols its relocations reach (two levels) laid out in fresh sections.
pub fn minimal_object(obj: &mwdec_core::ObjectFile, f: &mwdec_core::Function, prefix: &str) -> mwdec_core::ObjectFile {
    use mwdec_core::{DataSymbol, ObjectFile, Section, SymbolDef};
    let mut want: Vec<String> = f.relocs.iter().map(|r| r.target.clone()).collect();
    let mut k = 0;
    while k < want.len() && k < 256 {
        if let Some(d) = obj.data.get(&want[k]) {
            for r in &d.relocs {
                if !want.contains(&r.target) {
                    want.push(r.target.clone());
                }
            }
        }
        k += 1;
    }
    let rename = |n: &str| if let Some(rest) = n.strip_prefix(prefix) { format!("__P{rest}") } else { n.to_string() };
    let mut out = ObjectFile { path: obj.path.clone(), ..Default::default() };
    let mut func = f.clone();
    func.name = rename(&f.name);
    func.address = 0;
    for r in func.relocs.iter_mut() {
        r.target = rename(&r.target);
    }
    let mut sections: Vec<Section> = vec![Section { name: ".text".into(), size: f.code.len() as u32, bytes: f.code.clone(), executable: true, relocs: vec![] }];
    out.symbols.push(SymbolDef { name: func.name.clone(), section: ".text".into(), address: 0, size: f.code.len() as u32, binding: f.binding, is_func: true });
    for n in &want {
        let Some(d) = obj.data.get(n) else { continue };
        // the bytes as the section has them (a string pool reaches past its symbol's size)
        let (sec_bytes, base) = match mwdec_obj::symbol_location(obj, n) {
            Some((s, a)) => (s.bytes.clone(), a as usize),
            None => (d.bytes.clone(), 0),
        };
        let size = d.size.max(d.bytes.len() as u32) as usize;
        let mut end = (base + size).min(sec_bytes.len());
        // through the end of the last string that starts inside the symbol
        while end < sec_bytes.len() && end > base && sec_bytes[end - 1] != 0 && d.section != ".bss" && d.section != ".sbss" {
            end += 1;
        }
        let bytes: Vec<u8> = if base <= end && end <= sec_bytes.len() { sec_bytes[base..end].to_vec() } else { d.bytes.clone() };
        let si = match sections.iter().position(|s| s.name == d.section) {
            Some(i) => i,
            None => {
                sections.push(Section { name: d.section.clone(), size: 0, bytes: vec![], executable: false, relocs: vec![] });
                sections.len() - 1
            }
        };
        let sec = &mut sections[si];
        while sec.bytes.len() % 8 != 0 {
            sec.bytes.push(0);
        }
        let addr = sec.bytes.len() as u32;
        sec.bytes.extend_from_slice(&bytes);
        sec.size = sec.bytes.len() as u32;
        let mut nd = DataSymbol { address: addr, ..d.clone() };
        for r in nd.relocs.iter_mut() {
            r.target = rename(&r.target);
        }
        out.symbols.push(SymbolDef { name: n.clone(), section: d.section.clone(), address: addr, size: d.size, binding: d.binding, is_func: false });
        out.data.insert(n.clone(), nd);
    }
    out.all_symbols = std::iter::once(func.name.clone()).chain(want.iter().map(|n| rename(n))).collect();
    out.functions.push(func);
    out.sections = sections;
    out
}

/// Disk cache of whole template libraries (one file per set of probes: units with the same
/// context share it), for one lifter: reading one file instead of one per probe.
pub struct PackCache {
    dir: std::path::PathBuf,
    salt: String,
}

impl PackCache {
    pub fn new(dir: std::path::PathBuf, salt: String) -> PackCache {
        PackCache { dir, salt }
    }

    fn path(&self, keys: &[String]) -> std::path::PathBuf {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in self.salt.bytes().chain([0u8]).chain(keys.iter().flat_map(|k| k.bytes().chain([0u8]))) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.dir.join(format!("{h:016x}.json.gz"))
    }

    /// The outcome of every probe (by key; a key without an entry didn't compile).
    pub fn get(&self, keys: &[String]) -> Option<std::collections::HashMap<String, Result<Template, String>>> {
        let f = std::fs::File::open(self.path(keys)).ok()?;
        let (stored, entries): (Vec<String>, Vec<(String, Entry)>) = serde_json::from_reader(std::io::BufReader::new(flate2::read::GzDecoder::new(f))).ok()?;
        if stored != keys {
            return None;
        }
        Some(entries.iter().map(|(k, e)| (k.clone(), decode(e))).collect())
    }

    pub fn put(&self, keys: &[String], outcomes: &[(String, Result<Template, String>)]) {
        let entries: Vec<(String, Entry)> = outcomes
            .iter()
            .filter_map(|(k, r)| {
                let e = match r {
                    Ok(t) => encode(t)?,
                    Err(w) => Entry::Rejected(w.clone()),
                };
                Some((k.clone(), e))
            })
            .collect();
        let p = self.path(keys);
        let _ = std::fs::create_dir_all(&self.dir);
        if let Ok(s) = serde_json::to_vec(&(keys, entries)) {
            use std::io::Write;
            let mut z = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            if z.write_all(&s).is_err() {
                return;
            }
            let Ok(bytes) = z.finish() else { return };
            let tmp = p.with_extension(format!("tmp{}", std::process::id()));
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &p);
            }
        }
    }
}
