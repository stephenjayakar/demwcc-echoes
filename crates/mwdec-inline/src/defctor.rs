//! Default construction of members and bases, learned from the compiler.
//!
//! A constructor's members (and bases) that are not in its initializer list are default
//! constructed by the compiler before the body runs; when that constructor is inline (an
//! empty string pointing at its shared null buffer, an empty list linking its sentinel to
//! itself, a reference-counted pointer taking a reference on a shared null object) its
//! expansion shows up as stores at the start of the lifted body. Written out again in the
//! draft they would be emitted twice.
//!
//! For each class a constructor of the target object holds as a member or base, a probe
//! (`struct W { C m; W(); }; W::W() {}`) is compiled in the unit context and lifted; its body
//! is the class's default construction, canonicalised to stores relative to the object
//! (`this + k`), symbol addresses and constants. [`strip`] removes those stores from the start
//! of a constructor body for each member/base built implicitly.

use crate::util::strip as strip_cv;
use mwdec_core::{ObjectFile, Type, TypeDb};
use mwdec_lift::{BinOp, Expr, InitTarget, IrFunction, Stmt, VarId, VarKind};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Canonical value / address expression.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CE {
    Int(i64),
    Float(u64),
    /// `(char*)this + k`
    This(i32),
    /// `(char*)&sym + k`
    Sym(String, i64),
    /// `size` bytes read at an address
    Load(Box<CE>, u32),
    /// a local the code never set (an empty object's byte copied from an unset temporary)
    Unset,
    Bin(String, Box<CE>, Box<CE>),
}

/// `*(addr) = val` with `size` bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CStore {
    pub addr: CE,
    pub size: u32,
    pub val: CE,
}

/// Default construction of each class (canonical stores, offsets relative to the object);
/// classes whose default construction is not a sequence of plain stores are absent.
pub type DefCtors = HashMap<String, Vec<CStore>>;

fn add_off(a: CE, k: i64) -> CE {
    match a {
        CE::This(o) => CE::This(o + k as i32),
        CE::Sym(s, o) => CE::Sym(s, o + k),
        CE::Int(v) => CE::Int(v + k),
        other => CE::Bin("+".into(), Box::new(other), Box::new(CE::Int(k))),
    }
}

fn size_of(t: &Type, db: &TypeDb) -> Option<u32> {
    let r = mwdec_lift::types::resolve(Some(db), strip_cv(t)).into_owned();
    if let Some(s) = mwdec_lift::scalar_size(strip_cv(&r)) {
        return (s > 0 && s <= 8).then_some(s);
    }
    if mwdec_lift::types::is_aggregate(Some(db), &r) {
        return None;
    }
    mwdec_lift::types::size_of(Some(db), &r).filter(|s| *s > 0 && *s <= 8)
}

struct Canon<'a> {
    this: Option<VarId>,
    /// locals never assigned
    unset: HashSet<VarId>,
    defs: &'a HashMap<VarId, Expr>,
    db: &'a TypeDb,
}

impl Canon<'_> {
    fn val(&self, e: &Expr, depth: u32) -> Option<CE> {
        if depth > 12 {
            return None;
        }
        match e {
            Expr::Var(v) if Some(*v) == self.this => Some(CE::This(0)),
            Expr::Var(v) => match self.defs.get(v) {
                Some(d) => self.val(d, depth + 1),
                None if self.unset.contains(v) => Some(CE::Unset),
                None => None,
            },
            Expr::Int { value, .. } => Some(CE::Int(*value)),
            Expr::Float { bits, .. } => Some(CE::Float(*bits)),
            Expr::Cast { e, .. } => self.val(e, depth + 1),
            Expr::AddrOf(x) => self.addr(x, depth + 1),
            Expr::Global { ty, .. } | Expr::Load { ty, .. } | Expr::Member { ty, .. } => Some(CE::Load(Box::new(self.addr(e, depth + 1)?), size_of(ty, self.db)?)),
            Expr::Binary { op, l, r, .. } => {
                let (l, r) = (self.val(l, depth + 1)?, self.val(r, depth + 1)?);
                match (op, &l, &r) {
                    (BinOp::Add, _, CE::Int(k)) => Some(add_off(l, *k)),
                    (BinOp::Add, CE::Int(k), _) => Some(add_off(r, *k)),
                    _ => Some(CE::Bin(format!("{op:?}"), Box::new(l), Box::new(r))),
                }
            }
            _ => None,
        }
    }

    fn addr(&self, lv: &Expr, depth: u32) -> Option<CE> {
        match lv {
            Expr::Load { base, offset, .. } => Some(add_off(self.val(base, depth + 1)?, *offset as i64)),
            Expr::Member { base, offset, .. } => Some(add_off(self.addr(base, depth + 1)?, *offset as i64)),
            Expr::Global { symbol, .. } => Some(CE::Sym(symbol.clone(), 0)),
            _ => None,
        }
    }

    /// A store statement; None for anything else.
    fn store(&self, s: &Stmt) -> Option<CStore> {
        match s {
            Stmt::Assign { dst, src } if !matches!(dst, Expr::Var(_)) && !src.has_call() => {
                let ty = match dst {
                    Expr::Load { ty, .. } | Expr::Member { ty, .. } | Expr::Global { ty, .. } => ty,
                    _ => return None,
                };
                Some(CStore { addr: self.addr(dst, 0)?, size: size_of(ty, self.db)?, val: self.val(src, 0)? })
            }
            Stmt::Expr(Expr::IncDec { e, delta, .. }) => {
                let ty = match &**e {
                    Expr::Load { ty, .. } | Expr::Member { ty, .. } | Expr::Global { ty, .. } => ty,
                    _ => return None,
                };
                let size = size_of(ty, self.db)?;
                let addr = self.addr(e, 0)?;
                let val = add_off(CE::Load(Box::new(addr.clone()), size), *delta);
                Some(CStore { addr, size, val })
            }
            _ => None,
        }
    }
}

/// Single-definition call-free locals of a statement list (temporaries) and their values.
fn temp_defs(body: &[Stmt], vars: &[mwdec_lift::Var]) -> HashMap<VarId, Expr> {
    fn count_defs(body: &[Stmt], n: &mut HashMap<VarId, usize>) {
        for s in body {
            match s {
                Stmt::Assign { dst: Expr::Var(v), .. } => *n.entry(*v).or_default() += 1,
                Stmt::If { then, els, .. } => {
                    count_defs(then, n);
                    count_defs(els, n);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => count_defs(body, n),
                Stmt::For { init, step, body, .. } => {
                    count_defs(init, n);
                    count_defs(step, n);
                    count_defs(body, n);
                }
                Stmt::Switch { cases, .. } => cases.iter().for_each(|c| count_defs(&c.body, n)),
                _ => {}
            }
        }
    }
    let mut count: HashMap<VarId, usize> = HashMap::new();
    count_defs(body, &mut count);
    let mut out = HashMap::new();
    for s in body {
        if let Stmt::Assign { dst: Expr::Var(v), src } = s {
            if count[v] == 1 && vars[*v].kind == VarKind::Local && !src.has_call() {
                out.insert(*v, src.clone());
            }
        }
    }
    out
}

/// Locals (and stack slots) nothing assigns or takes the address of.
fn unset_vars(ir: &IrFunction) -> HashSet<VarId> {
    let mut set: HashSet<VarId> = HashSet::new();
    Stmt::walk_exprs(&ir.body, &mut |e| {
        if let Expr::AddrOf(x) = e {
            if let Expr::Var(v) = &**x {
                set.insert(*v);
            }
        }
    });
    fn defs(body: &[Stmt], set: &mut HashSet<VarId>) {
        for s in body {
            match s {
                Stmt::Assign { dst: Expr::Var(v), .. } => {
                    set.insert(*v);
                }
                Stmt::Assign { dst: Expr::Member { base, .. }, .. } => {
                    if let Expr::Var(v) = &**base {
                        set.insert(*v);
                    }
                }
                Stmt::If { then, els, .. } => {
                    defs(then, set);
                    defs(els, set);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => defs(body, set),
                Stmt::For { init, step, body, .. } => {
                    defs(init, set);
                    defs(step, set);
                    defs(body, set);
                }
                Stmt::Switch { cases, .. } => cases.iter().for_each(|c| defs(&c.body, set)),
                _ => {}
            }
        }
    }
    defs(&ir.body, &mut set);
    (0..ir.vars.len()).filter(|v| matches!(ir.vars[*v].kind, VarKind::Local | VarKind::Stack { .. }) && !set.contains(v)).collect()
}

/// The default construction in a lifted probe body (`W::W()` with the member at offset 0).
fn from_probe(ir: &IrFunction, db: &TypeDb) -> Option<Vec<CStore>> {
    if !ir.init_list.is_empty() {
        return None;
    }
    let defs = temp_defs(&ir.body, &ir.vars);
    let c = Canon { this: ir.this_var, unset: unset_vars(ir), defs: &defs, db };
    let mut out = vec![];
    for s in &ir.body {
        match s {
            Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
            Stmt::Return(None) => {}
            s => out.push(c.store(s)?),
        }
    }
    (!out.is_empty()).then_some(out)
}

fn shift(e: &CE, by: i32) -> CE {
    match e {
        CE::This(k) => CE::This(k + by),
        CE::Load(a, s) => CE::Load(Box::new(shift(a, by)), *s),
        CE::Bin(o, a, b) => CE::Bin(o.clone(), Box::new(shift(a, by)), Box::new(shift(b, by))),
        other => other.clone(),
    }
}

/// Classes held (as direct members or non-virtual bases) by the classes whose constructors
/// `symbols` define.
pub fn wanted<'a>(symbols: impl Iterator<Item = &'a str>, db: &TypeDb) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let mut seen = HashSet::new();
    for s in symbols {
        if !s.starts_with("__ct__") {
            continue;
        }
        let sig = mwdec_lift::sig::sig_of(s, Some(db));
        let Some(own) = sig.this_class.as_deref() else { continue };
        let Some(c) = mwdec_lift::sig::find_class(db, own) else { continue };
        let mut add = |n: String| {
            if seen.insert(n.clone()) {
                out.push(n);
            }
        };
        for b in c.bases.iter().filter(|b| !b.is_virtual) {
            if let Some(n) = crate::util::class_name(&Type::Named(b.name.clone()), db) {
                add(n);
            }
        }
        for f in &c.fields {
            if f.bitfield.is_some() {
                continue;
            }
            let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
            if matches!(strip_cv(&ft), Type::Array(..)) {
                continue;
            }
            if let Some(n) = crate::util::class_name(&ft, db) {
                add(n);
            }
        }
    }
    out.retain(|n| mwdec_lift::sig::find_class(db, n).is_some_and(|c| !c.is_declaration && !c.is_union));
    out
}

/// Disk cache of probe outcomes (keyed by probe text and compiler/flags salt).
pub struct DefCache {
    dir: std::path::PathBuf,
    salt: String,
}

impl DefCache {
    pub fn new(dir: std::path::PathBuf, salt: String) -> DefCache {
        DefCache { dir, salt }
    }

    fn path(&self, text: &str) -> std::path::PathBuf {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in VERSION.bytes().chain(self.salt.bytes()).chain([0u8]).chain(text.bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.dir.join(format!("{:02x}", h & 0xff)).join(format!("{h:016x}.json"))
    }

    fn get(&self, text: &str) -> Option<Option<Vec<CStore>>> {
        let s = std::fs::read_to_string(self.path(text)).ok()?;
        let (t, v): (String, Option<Vec<CStore>>) = serde_json::from_str(&s).ok()?;
        (t == text).then_some(v)
    }

    fn put(&self, text: &str, v: &Option<Vec<CStore>>) {
        let p = self.path(text);
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        if let Ok(s) = serde_json::to_string(&(text, v)) {
            let tmp = p.with_extension(format!("tmp{}", std::process::id()));
            if std::fs::write(&tmp, s).is_ok() {
                let _ = std::fs::rename(&tmp, &p);
            }
        }
    }
}

const PROBE: &str = "__mwdec_dc";
/// Version of the canonical form (part of the cache key).
const VERSION: &str = "defctor v2";

fn probe_text(spelled: &str, k: usize) -> String {
    format!("struct {PROBE}{k} {{ {spelled} m; {PROBE}{k}(); }};\n{PROBE}{k}::{PROBE}{k}() {{}}\n")
}

/// Default constructions of `classes`, from the cache or by compiling probes.
pub fn build(db: &TypeDb, classes: &[String], cache: Option<&DefCache>, compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync)) -> DefCtors {
    let mut out = DefCtors::new();
    let mut todo: Vec<(String, String)> = vec![];
    if std::env::var("MWDI_DEFCTOR_DEBUG").is_ok() {
        eprintln!("defctor classes {classes:?}");
    }
    for c in classes {
        let Some(sp) = crate::probe::spell(&Type::Named(c.clone()), db, &[]) else { continue };
        let key = probe_text(&sp, 0);
        match cache.and_then(|ca| ca.get(&key)) {
            Some(Some(v)) => {
                out.insert(c.clone(), v);
            }
            Some(None) => {}
            None => todo.push((c.clone(), sp)),
        }
    }
    if todo.is_empty() {
        return out;
    }
    let lift = |obj: &ObjectFile, k: usize| -> Option<Vec<CStore>> {
        let name = format!("{PROBE}{k}");
        let f = obj.functions.iter().find(|f| f.name.starts_with("__ct__") && f.name.contains(&format!("{}{name}F", name.len())))?;
        let ir = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mwdec_lift::lift_function(obj, f, Some(db)))).ok()?.ok()?;
        from_probe(&ir, db)
    };
    let all: String = todo.iter().enumerate().map(|(k, (_, sp))| probe_text(sp, k)).collect();
    let results: Vec<Option<Vec<CStore>>> = match compile(&all) {
        Ok(obj) => (0..todo.len()).map(|k| lift(&obj, k)).collect(),
        // one class without an accessible default constructor fails the batch: one by one
        Err(_) => todo.iter().map(|(_, sp)| compile(&probe_text(sp, 0)).ok().and_then(|o| lift(&o, 0))).collect(),
    };
    for ((c, sp), r) in todo.iter().zip(results) {
        if let Some(ca) = cache {
            ca.put(&probe_text(sp, 0), &r);
        }
        if let Some(v) = r {
            out.insert(c.clone(), v);
        }
    }
    out
}

/// Remove, from the start of a constructor body, the stores of the default constructions of
/// its members and bases that the initializer list doesn't name. Returns the number of
/// removed statements.
pub fn strip(ir: &mut IrFunction, db: &TypeDb, dc: &DefCtors) -> usize {
    if dc.is_empty() || !ir.symbol.starts_with("__ct__") {
        return 0;
    }
    let Some(this) = ir.this_var else { return 0 };
    let Some(own) = ir.sig.this_class.clone() else { return 0 };
    let Some(c) = mwdec_lift::sig::find_class(db, &own).cloned() else { return 0 };
    // implicit default constructions: (offset, class, member name and type)
    let mut parts: Vec<(i32, String, Option<(String, Type)>)> = vec![];
    for b in c.bases.iter().filter(|b| !b.is_virtual) {
        let named = ir.init_list.iter().any(|i| matches!(&i.target, InitTarget::Base(n) if mwdec_lift::sig::norm_name(n) == mwdec_lift::sig::norm_name(&b.name)));
        if let (false, Some(n)) = (named, crate::util::class_name(&Type::Named(b.name.clone()), db)) {
            parts.push((b.offset as i32, n, None));
        }
    }
    for f in &c.fields {
        if f.bitfield.is_some() || ir.init_list.iter().any(|i| i.target == InitTarget::Member(f.name.clone())) {
            continue;
        }
        let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
        if matches!(strip_cv(&ft), Type::Array(..)) {
            continue;
        }
        if let Some(n) = crate::util::class_name(&ft, db) {
            parts.push((f.offset as i32, n, Some((f.name.clone(), ft.clone()))));
        }
    }
    if std::env::var("MWDI_DEFCTOR_DEBUG").is_ok() {
        eprintln!("defctor {}: parts {:?}; known {:?}", ir.symbol, parts.iter().map(|p| (&p.0, &p.1)).collect::<Vec<_>>(), dc.keys().collect::<Vec<_>>());
    }
    parts.retain(|(_, n, _)| dc.contains_key(n));
    if parts.is_empty() {
        return 0;
    }
    // the leading run of stores (temporaries in between)
    let defs = temp_defs(&ir.body, &ir.vars);
    let cn = Canon { this: Some(this), unset: unset_vars(ir), defs: &defs, db };
    let mut lead: Vec<(usize, CStore)> = vec![];
    for (i, s) in ir.body.iter().enumerate() {
        match s {
            Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
            s => match cn.store(s) {
                Some(st) => lead.push((i, st)),
                None => break,
            },
        }
    }
    let mut used = vec![false; lead.len()];
    let mut remove = vec![];
    let mut unmatched = vec![];
    for (off, cls, member) in parts {
        let want = &dc[&cls];
        let mut pick = vec![];
        for w in want {
            let (a, v) = (shift(&w.addr, off), shift(&w.val, off));
            let hit = (0..lead.len()).find(|&k| !used[k] && !pick.contains(&k) && lead[k].1.addr == a && lead[k].1.size == w.size && lead[k].1.val == v);
            match hit {
                Some(k) => pick.push(k),
                None => {
                    pick.clear();
                    break;
                }
            }
        }
        if pick.is_empty() {
            if let Some((n, t)) = member {
                unmatched.push((off, cls.clone(), n, t));
            }
        }
        for k in pick {
            used[k] = true;
            remove.push(lead[k].0);
        }
    }
    remove.sort_unstable();
    for &i in remove.iter().rev() {
        ir.body.remove(i);
    }
    let moved = explicit_inits(ir, db, this, &unmatched);
    if !remove.is_empty() || moved > 0 {
        let vars = ir.vars.clone();
        drop_unused_temps(&mut ir.body, &vars);
    }
    remove.len() + moved
}

/// Members whose default construction is not at the start of the body were built explicitly
/// in the initializer list: the body's first assignment of the whole member (`m = e`, or the
/// only scalar of a one-member class) becomes the entry `m(e)`.
fn explicit_inits(ir: &mut IrFunction, db: &TypeDb, this: VarId, unmatched: &[(i32, String, String, Type)]) -> usize {
    let mut moved = 0;
    let own = ir.sig.this_class.clone().unwrap_or_default();
    let field_off = |n: &str| mwdec_lift::sig::find_class(db, &own).and_then(|c| c.fields.iter().find(|f| f.name == n).map(|f| f.offset as i32));
    for (off, cls, name, fty) in unmatched {
        let Some(size) = mwdec_lift::types::size_of(Some(db), fty) else { continue };
        let defs = temp_defs(&ir.body, &ir.vars);
        let cn = Canon { this: Some(this), unset: unset_vars(ir), defs: &defs, db };
        // the first top-level statement writing into the member
        let mut found = None;
        for (i, s) in ir.body.iter().enumerate() {
            match s {
                Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
                Stmt::Assign { dst, src } if !matches!(dst, Expr::Var(_)) => {
                    let Some(CE::This(a)) = cn.addr(dst, 0) else { break };
                    if a >= *off && a < off + size as i32 {
                        found = Some((i, a, dst.clone(), src.clone()));
                        break;
                    }
                }
                _ => break,
            }
        }
        if std::env::var("MWDI_DEFCTOR_DEBUG").is_ok() {
            eprintln!("defctor explicit {name}: found {:?}", found.as_ref().map(|f| (f.0, f.1)));
        }
        let Some((i, a, dst, src)) = found else { continue };
        if a != *off {
            continue;
        }
        let dty = match &dst {
            Expr::Load { ty, .. } | Expr::Member { ty, .. } => ty.clone(),
            _ => continue,
        };
        let whole = crate::util::class_name(&dty, db).is_some_and(|c| mwdec_lift::sig::norm_name(&c) == mwdec_lift::sig::norm_name(cls));
        let single = !whole && mwdec_lift::types::size_of(Some(db), &dty) == Some(size) && mwdec_lift::sig::find_class(db, cls).is_some_and(|c| c.vptr_offset.is_none());
        // the whole object read as one scalar from another object of the class: a copy
        let copy_of = match &src {
            Expr::Load { base, offset, ty } if single && mwdec_lift::types::size_of(Some(db), ty) == Some(size) => {
                let bt = mwdec_lift::types::ty_of(base, &ir.vars);
                let pointee = match strip_cv(&bt) {
                    Type::Ptr(x) | Type::Ref(x) => Some((**x).clone()),
                    _ => None,
                };
                pointee
                    .and_then(|p| crate::util::class_name(&p, db))
                    .filter(|c| mwdec_lift::sig::norm_name(c) == mwdec_lift::sig::norm_name(cls))
                    .map(|_| Expr::Load { base: base.clone(), offset: *offset, ty: fty.clone() })
            }
            _ => None,
        };
        // a one-scalar class built from a scalar needs a converting constructor
        let converting = single
            && crate::ctors::ctor_decls(db, cls).is_some_and(|ds| {
                ds.iter().any(|d| d.params.len() == 1 && crate::util::class_name(&d.params[0].ty, db).is_none() && mwdec_lift::scalar_size(strip_cv(&mwdec_lift::types::resolve(Some(db), &d.params[0].ty))).is_some())
            });
        if std::env::var("MWDI_DEFCTOR_DEBUG").is_ok() {
            eprintln!("defctor explicit {name}: whole {whole} single {single} converting {converting} src {src:?}");
        }
        if !(whole || converting || copy_of.is_some()) {
            continue;
        }
        let whole = whole || copy_of.is_some();
        // the value with temporaries substituted: only parameters, `this`, globals, constants
        let mut arg = copy_of.unwrap_or_else(|| src.clone());
        let mut ok = true;
        for _ in 0..6 {
            let mut changed = false;
            arg.rewrite(&mut |x| {
                if let Expr::Var(v) = x {
                    if let Some(d) = defs.get(v) {
                        *x = d.clone();
                        changed = true;
                    }
                }
            });
            if !changed {
                break;
            }
        }
        arg.walk(&mut |x| {
            if let Expr::Var(v) = x {
                ok &= Some(*v) == ir.this_var || ir.params.contains(v) || matches!(ir.vars[*v].kind, VarKind::Param { .. });
            }
        });
        if std::env::var("MWDI_DEFCTOR_DEBUG").is_ok() {
            eprintln!("defctor explicit {name}: arg ok {ok} {arg:?} params {:?}", ir.params);
        }
        if !ok {
            continue;
        }
        ir.body.remove(i);
        let init = mwdec_lift::Init { target: InitTarget::Member(name.clone()), ctor: None, args: vec![arg], member_ty: whole.then(|| fty.clone()) };
        let pos = ir.init_list.iter().position(|e| match &e.target {
            InitTarget::Member(n) => field_off(n).is_some_and(|o| o > *off),
            InitTarget::Base(_) => false,
        });
        match pos {
            Some(p) => ir.init_list.insert(p, init),
            None => ir.init_list.push(init),
        }
        moved += 1;
    }
    moved
}

/// Top-level definitions of temporaries nothing reads any more.
fn drop_unused_temps(body: &mut Vec<Stmt>, vars: &[mwdec_lift::Var]) {
    loop {
        let mut uses: HashMap<VarId, usize> = HashMap::new();
        Stmt::walk_exprs(body, &mut |e| {
            if let Expr::Var(v) = e {
                *uses.entry(*v).or_default() += 1;
            }
        });
        let before = body.len();
        // a definition counts its destination once
        body.retain(|s| !matches!(s, Stmt::Assign { dst: Expr::Var(v), src } if vars[*v].kind == VarKind::Local && uses.get(v).copied().unwrap_or(0) <= 1 && !src.has_call()));
        if body.len() == before {
            break;
        }
    }
}

