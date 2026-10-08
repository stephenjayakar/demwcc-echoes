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
    /// `(char*)p + k` for parameter `p` (a pointer's value, or the address of an object
    /// passed by reference)
    Param(usize, i32),
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
        CE::Param(v, o) => CE::Param(v, o + k as i32),
        CE::Int(v) => CE::Int(v + k),
        other if k == 0 => other,
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
    /// parameters
    params: HashSet<VarId>,
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
                None if self.params.contains(v) => Some(CE::Param(*v, 0)),
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
            Expr::Var(v) if self.params.contains(v) => Some(CE::Param(*v, 0)),
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

fn param_vars(ir: &IrFunction) -> HashSet<VarId> {
    (0..ir.vars.len()).filter(|v| matches!(ir.vars[*v].kind, VarKind::Param { .. })).collect()
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
    let c = Canon { this: ir.this_var, unset: unset_vars(ir), params: param_vars(ir), defs: &defs, db };
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

    fn get<T: serde::de::DeserializeOwned>(&self, text: &str) -> Option<Option<T>> {
        let s = std::fs::read_to_string(self.path(text)).ok()?;
        let (t, v): (String, Option<T>) = serde_json::from_str(&s).ok()?;
        (t == text).then_some(v)
    }

    fn put<T: Serialize>(&self, text: &str, v: &Option<T>) {
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
const VERSION: &str = "defctor v5";

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
        match cache.and_then(|ca| ca.get::<Vec<CStore>>(&key)) {
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
pub fn strip(ir: &mut IrFunction, db: &TypeDb, dc: &DefCtors) -> (usize, Pending) {
    if !ir.symbol.starts_with("__ct__") {
        return (0, Pending::default());
    }
    let Some(this) = ir.this_var else { return (0, Pending::default()) };
    let Some(own) = ir.sig.this_class.clone() else { return (0, Pending::default()) };
    let Some(c) = mwdec_lift::sig::find_class(db, &own).cloned() else { return (0, Pending::default()) };
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
    // members of classes without a known default construction: only a member-wise copy
    let unknown: Vec<(i32, String, String, Type)> = parts.iter().filter(|(_, n, m)| m.is_some() && !dc.contains_key(n)).map(|(o, n, m)| (*o, n.clone(), m.clone().unwrap().0, m.clone().unwrap().1)).collect();
    let copied = memberwise_copies(ir, db, this, &unknown);
    parts.retain(|(_, n, _)| dc.contains_key(n));
    if parts.is_empty() {
        return (copied, Pending::default());
    }
    // the leading run of stores (temporaries in between)
    let defs = temp_defs(&ir.body, &ir.vars);
    let cn = Canon { this: Some(this), unset: unset_vars(ir), params: param_vars(ir), defs: &defs, db };
    let mut lead: Vec<(usize, CStore)> = vec![];
    for (i, s) in ir.body.iter().enumerate() {
        match s {
            Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
            Stmt::Comment(c) if c.starts_with(mwdec_lift::idioms::INIT_MARK) => {}
            s => match cn.store(s) {
                Some(st) => lead.push((i, st)),
                // (a member set from a parameter, a bitfield, an array of members built by the
                // runtime: other members' constructions may follow)
                None if matches!(s, Stmt::Assign { dst, src } if !matches!(dst, Expr::Var(_)) && !src.has_call() && matches!(cn.addr(dst, 0), Some(CE::This(_)))) => {}
                None if matches!(s, Stmt::Assign { dst: Expr::BitField { base, .. }, src } if !src.has_call() && matches!(cn.addr(base, 0), Some(CE::This(_)))) => {}
                None if matches!(s, Stmt::Expr(Expr::Call { callee: mwdec_lift::Callee::Direct { symbol, .. }, args, .. }) if symbol == "__construct_array" && args.first().is_some_and(|a| matches!(cn.val(a, 0), Some(CE::This(_))))) => {}
                None => break,
            },
        }
    }
    let mut used = vec![false; lead.len()];
    let mut remove = vec![];
    let mut unmatched = vec![];
    let mut marks: Vec<(usize, String)> = vec![];
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
            if let Some((n, t)) = &member {
                unmatched.push((off, cls.clone(), n.clone(), t.clone()));
            }
        }
        // where the member's construction ran: scalar members set before it (declared before
        // it) were initialized first (see `finish`)
        if let (Some(first), Some((n, _))) = (pick.iter().map(|&k| lead[k].0).min(), &member) {
            marks.push((first, n.clone()));
        }
        for k in pick {
            used[k] = true;
            remove.push(lead[k].0);
        }
    }
    remove.sort_unstable();
    for &i in remove.iter().rev() {
        match marks.iter().find(|m| m.0 == i) {
            Some((_, n)) => ir.body[i] = Stmt::Comment(format!("{}{n}", mwdec_lift::idioms::INIT_MARK)),
            None => {
                ir.body.remove(i);
            }
        }
    }
    let mut moved = copied + explicit_inits(ir, db, this, &unmatched);
    moved += memberwise_copies(ir, db, this, &unmatched);
    let pending = Pending(unmatched.into_iter().filter(|m| !ir.init_list.iter().any(|i| i.target == InitTarget::Member(m.2.clone()))).collect());
    if !remove.is_empty() || moved > 0 {
        let vars = ir.vars.clone();
        drop_unused_temps(&mut ir.body, &vars);
    }
    (remove.len() + moved, pending)
}

/// Members whose default construction is not at the start of the body were built explicitly
/// in the initializer list: the body's first assignment of the whole member (`m = e`, or the
/// only scalar of a one-member class) becomes the entry `m(e)`.
fn explicit_inits(ir: &mut IrFunction, db: &TypeDb, this: VarId, unmatched: &[(i32, String, String, Type)]) -> usize {
    let mut moved = 0;
    for (off, cls, name, fty) in unmatched {
        if ir.init_list.iter().any(|i| i.target == InitTarget::Member(name.clone())) {
            continue;
        }
        let Some(size) = mwdec_lift::types::size_of(Some(db), fty) else { continue };
        let defs = temp_defs(&ir.body, &ir.vars);
        let cn = Canon { this: Some(this), unset: unset_vars(ir), params: param_vars(ir), defs: &defs, db };
        // the first top-level statement writing into the member
        let mut found = None;
        for (i, s) in ir.body.iter().enumerate() {
            match s {
                Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
                Stmt::Comment(c) if c.starts_with(mwdec_lift::idioms::INIT_MARK) => {}
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
        if !(whole || converting || copy_of.is_some()) {
            continue;
        }
        let whole = whole || copy_of.is_some();
        let Some(arg) = init_arg(&copy_of.unwrap_or_else(|| src.clone()), &defs, ir, true, i) else { continue };
        let mut gone = if whole { copy_residue(ir, &cn, &defs, i + 1, *off, size as i32) } else { vec![] };
        gone.push(i);
        gone.sort_unstable();
        for &k in gone.iter().rev() {
            ir.body.remove(k);
        }
        let n = move_earlier(ir, db, this, i, *off);
        insert_init(ir, db, mwdec_lift::Init { target: InitTarget::Member(name.clone()), ctor: None, args: vec![arg], member_ty: whole.then(|| fty.clone()) }, *off);
        moved += 1 + n;
    }
    moved
}

/// Byte ranges of a class's data (bitfield units whole), relative to the object.
fn data_ranges(db: &TypeDb, cls: &str, base: i32, out: &mut Vec<(i32, i32)>, depth: u32) -> Option<()> {
    if depth > 8 {
        return None;
    }
    let c = mwdec_lift::sig::find_class(db, cls)?;
    if c.vptr_offset.is_some() {
        return None;
    }
    for b in &c.bases {
        data_ranges(db, &b.name, base + b.offset as i32, out, depth + 1)?;
    }
    for f in &c.fields {
        let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
        let o = base + f.offset as i32;
        if f.bitfield.is_none() {
            if let Some(n) = crate::util::class_name(&ft, db) {
                if !matches!(strip_cv(&ft), Type::Array(..)) {
                    data_ranges(db, &n, o, out, depth + 1)?;
                    continue;
                }
            }
        }
        let size = if f.bitfield.is_some() && f.size > 0 { f.size } else { mwdec_lift::types::size_of(Some(db), &ft)? };
        out.push((o, o + size as i32));
    }
    Some(())
}

/// A member whose default construction is absent, filled store by store from the same
/// offsets of one object passed in: copy-constructed from it (`m(other)`).
fn memberwise_copies(ir: &mut IrFunction, db: &TypeDb, this: VarId, unmatched: &[(i32, String, String, Type)]) -> usize {
    let mut moved = 0;
    for (off, cls, name, fty) in unmatched {
        if ir.init_list.iter().any(|i| i.target == InitTarget::Member(name.clone())) {
            continue;
        }
        let Some(size) = mwdec_lift::types::size_of(Some(db), fty) else { continue };
        let mut ranges = vec![];
        if data_ranges(db, cls, 0, &mut ranges, 0).is_none() || ranges.is_empty() {
            continue;
        }
        let defs = temp_defs(&ir.body, &ir.vars);
        let cn = Canon { this: Some(this), unset: unset_vars(ir), params: param_vars(ir), defs: &defs, db };
        let mut src: Option<usize> = None;
        let mut hits: Vec<(usize, i32, i32)> = vec![];
        let mut ok = true;
        for (k, s) in ir.body.iter().enumerate() {
            match s {
                Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
                Stmt::Comment(_) => {}
                s => {
                    let Some(st) = cn.store(s) else { break };
                    let CE::This(a) = st.addr else { break };
                    if a < *off || a >= off + size as i32 {
                        continue;
                    }
                    // the same offset of one parameter's object
                    match &st.val {
                        CE::Load(b, sz) if *sz == st.size => match &**b {
                            CE::Param(p, o) if *o == a - off && src.map_or(true, |x| x == *p) => {
                                src = Some(*p);
                                hits.push((k, a - off, a - off + st.size as i32));
                            }
                            _ => {
                                ok = false;
                                break;
                            }
                        },
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
            }
        }
        let Some(p) = src else { continue };
        // every data byte of the class written
        let covered = |x: i32| hits.iter().any(|(_, lo, hi)| x >= *lo && x < *hi);
        if !ok || !ranges.iter().all(|(lo, hi)| (*lo..*hi).all(covered)) {
            continue;
        }
        let pt = ir.vars[p].ty.clone();
        let same = |t: &Type| crate::util::class_name(t, db).is_some_and(|c| mwdec_lift::sig::norm_name(&c) == mwdec_lift::sig::norm_name(cls));
        let arg = match strip_cv(&pt) {
            Type::Ptr(x) | Type::Ref(x) if same(strip_cv(x)) => Expr::Load { base: Box::new(Expr::Var(p)), offset: 0, ty: fty.clone() },
            t if same(t) => Expr::Var(p),
            _ => continue,
        };
        let mut gone: Vec<usize> = hits.iter().map(|h| h.0).collect();
        gone.sort_unstable();
        gone.dedup();
        let first = gone[0];
        for &k in gone.iter().rev() {
            ir.body.remove(k);
        }
        let n = move_earlier(ir, db, this, first, *off);
        insert_init(ir, db, mwdec_lift::Init { target: InitTarget::Member(name.clone()), ctor: None, args: vec![arg], member_ty: Some(fty.clone()) }, *off);
        moved += 1 + n;
    }
    moved
}

/// `e` with temporaries substituted, when it only reads parameters, globals, constants (and
/// `this` with `this_ok`: members built before).
fn init_arg(e: &Expr, defs: &HashMap<VarId, Expr>, ir: &IrFunction, this_ok: bool, at: usize) -> Option<Expr> {
    let mut arg = e.clone();
    // a temporary holding a memory read can't move past a store or call after its definition
    let stale = |v: VarId| -> bool {
        let Some(d) = ir.body.iter().position(|s| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if *x == v)) else { return false };
        let mut reads = false;
        defs[&v].walk(&mut |x| reads |= matches!(x, Expr::Load { .. } | Expr::Member { .. } | Expr::Global { .. } | Expr::Index { .. }));
        reads && ir.body.iter().take(at).skip(d + 1).any(|s| match s {
            Stmt::Assign { dst: Expr::Var(_), src } => src.has_call(),
            Stmt::Comment(_) => false,
            _ => true,
        })
    };
    let mut bad = false;
    for _ in 0..6 {
        let mut changed = false;
        arg.rewrite(&mut |x| {
            if let Expr::Var(v) = x {
                if let Some(d) = defs.get(v) {
                    bad |= stale(*v);
                    *x = d.clone();
                    changed = true;
                }
            }
        });
        if !changed {
            break;
        }
    }
    let mut ok = true;
    arg.walk(&mut |x| {
        if let Expr::Var(v) = x {
            ok &= (this_ok && Some(*v) == ir.this_var) || ir.params.contains(v) || matches!(ir.vars[*v].kind, VarKind::Param { .. });
        }
    });
    (ok && !bad).then_some(arg)
}

fn field_off(ir: &IrFunction, db: &TypeDb, n: &str) -> Option<i32> {
    let own = ir.sig.this_class.as_deref()?;
    mwdec_lift::sig::find_class(db, own).and_then(|c| c.fields.iter().find(|f| f.name == n).map(|f| f.offset as i32))
}

/// Insert an initializer-list entry in declaration (offset) order.
fn insert_init(ir: &mut IrFunction, db: &TypeDb, init: mwdec_lift::Init, off: i32) {
    let pos = ir.init_list.iter().position(|e| match &e.target {
        InitTarget::Member(n) => field_off(ir, db, n).is_some_and(|o| o > off),
        InitTarget::Base(_) => false,
    });
    match pos {
        Some(p) => ir.init_list.insert(p, init),
        None => ir.init_list.push(init),
    }
}

/// Scalar members declared before offset `before` that the body set ahead of statement `upto`
/// (in its leading run of stores): their initializers ran first, so they are entries too.
fn move_earlier(ir: &mut IrFunction, db: &TypeDb, this: VarId, upto: usize, before: i32) -> usize {
    let Some(own) = ir.sig.this_class.clone() else { return 0 };
    let Some(c) = mwdec_lift::sig::find_class(db, &own).cloned() else { return 0 };
    let defs = temp_defs(&ir.body, &ir.vars);
    let mut take: Vec<(usize, mwdec_lift::Init, i32)> = vec![];
    for (j, s) in ir.body.iter().enumerate().take(upto) {
        match s {
            Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
            Stmt::Comment(c) if c.starts_with(mwdec_lift::idioms::INIT_MARK) => {}
            Stmt::Assign { dst, src } => {
                let fld = match dst {
                    Expr::Load { base, offset, ty } if matches!(&**base, Expr::Var(t) if *t == this) => {
                        c.fields.iter().find(|f| f.offset as i32 == *offset && f.bitfield.is_none() && mwdec_lift::types::size_of(Some(db), &f.ty) == mwdec_lift::types::size_of(Some(db), ty))
                    }
                    // a bitfield of the class
                    Expr::BitField { base, shift, width, .. } => match &**base {
                        Expr::Load { base: b, offset, ty } if matches!(&**b, Expr::Var(t) if *t == this) => {
                            let unit = mwdec_lift::types::size_of(Some(db), ty).unwrap_or(0);
                            let mask = (((1u64 << *width) - 1) << *shift) as u32;
                            match mwdec_lift::types::bitfield_at(db, &own, *offset, unit, mask).map(|x| x.0) {
                                Some(path) => match path.as_slice() {
                                    [mwdec_lift::types::PathElem::Field(n, _)] => c.fields.iter().find(|f| f.name == *n),
                                    _ => None,
                                },
                                None => None,
                            }
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let Some(f) = fld else { break };
                let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
                if f.offset as i32 >= before || crate::util::class_name(&ft, db).is_some() || ir.init_list.iter().any(|i| i.target == InitTarget::Member(f.name.clone())) {
                    break;
                }
                let Some(arg) = init_arg(src, &defs, ir, false, j) else { break };
                take.push((j, mwdec_lift::Init { target: InitTarget::Member(f.name.clone()), ctor: None, args: vec![arg], member_ty: Some(ft) }, f.offset as i32));
            }
            _ => break,
        }
    }
    let n = take.len();
    for (j, init, off) in take.into_iter().rev() {
        ir.body.remove(j);
        insert_init(ir, db, init, off);
    }
    n
}

fn ce_in(c: &CE, off: i32, size: i32) -> bool {
    match c {
        CE::This(k) => *k >= off && *k < off + size,
        CE::Load(a, _) => ce_in(a, off, size),
        CE::Bin(_, a, b) => ce_in(a, off, size) || ce_in(b, off, size),
        _ => false,
    }
}

/// A store through a pointer the member holds (`++*m.mRefCount`): the rest of an inline copy
/// constructor.
fn through_member(addr: &CE, off: i32, size: i32) -> bool {
    matches!(addr, CE::Load(a, _) if ce_in(a, off, size))
}

/// Statements after `b` that finish an inline copy constructor of the member at `off`.
fn copy_residue(ir: &IrFunction, cn: &Canon, defs: &HashMap<VarId, Expr>, from: usize, off: i32, size: i32) -> Vec<usize> {
    let mut out = vec![];
    // locals holding a pointer read from the member through folded accessors
    // (`t = m.GetObj().GetRefCountPtr(); *t += 1;`)
    let mut held: Vec<VarId> = vec![];
    let reads_m = |e: &Expr| {
        let mut hit = false;
        e.walk(&mut |x| {
            if matches!(x, Expr::Load { .. } | Expr::Member { .. }) {
                hit |= matches!(cn.addr(x, 0), Some(CE::This(k)) if k >= off && k < off + size);
            }
        });
        hit
    };
    for k in from..ir.body.len() {
        match &ir.body[k] {
            Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
            Stmt::Assign { dst: Expr::Var(v), src } if ir.vars[*v].kind == VarKind::Local && reads_m(src) => {
                held.push(*v);
                out.push(k);
            }
            Stmt::Assign { dst: Expr::Load { base, offset: 0, .. }, src } if matches!(&**base, Expr::Var(v) if held.contains(v)) && matches!(src, Expr::Binary { .. }) => out.push(k),
            s => match cn.store(s) {
                Some(st) if through_member(&st.addr, off, size) => out.push(k),
                _ => break,
            },
        }
    }
    // a local whose pointer was not stored through is not part of it
    let used: Vec<usize> = out.iter().copied().filter(|&k| !matches!(&ir.body[k], Stmt::Assign { dst: Expr::Var(v), .. } if !out.iter().any(|&j| j != k && mentions(&ir.body[j], *v)))).collect();
    used
}

fn strip_casts(e: &Expr) -> &Expr {
    match e {
        Expr::Cast { e, .. } => strip_casts(e),
        e => e,
    }
}

/// The variable an lvalue lies in (`v.m`, `*(T*)&v`).
fn lvalue_root(e: &Expr) -> Option<VarId> {
    match strip_casts(e) {
        Expr::Var(v) => Some(*v),
        Expr::Member { base, .. } => lvalue_root(base),
        Expr::Load { base, .. } => match strip_casts(base) {
            Expr::AddrOf(x) => lvalue_root(x),
            _ => None,
        },
        _ => None,
    }
}

fn mentions(s: &Stmt, v: VarId) -> bool {
    let mut hit = false;
    Stmt::walk_exprs(std::slice::from_ref(s), &mut |e| hit |= matches!(e, Expr::Var(x) if *x == v));
    hit
}

/// Members copy-constructed from an object (`m(f())`, `m(param)`): the body's whole-member
/// assignment from it, the rest of the copy constructor's expansion (stores inside the member
/// or through pointers it holds) and the temporary's destruction become the entry.
fn copy_inits(ir: &mut IrFunction, db: &TypeDb, this: VarId, members: &[(i32, String, String, Type)], need_residue: bool) -> usize {
    let mut moved = 0;
    for (off, cls, name, fty) in members {
        if ir.init_list.iter().any(|i| i.target == InitTarget::Member(name.clone())) {
            continue;
        }
        let Some(size) = mwdec_lift::types::size_of(Some(db), fty) else { continue };
        let defs = temp_defs(&ir.body, &ir.vars);
        let cn = Canon { this: Some(this), unset: unset_vars(ir), params: param_vars(ir), defs: &defs, db };
        let same = |t: &Type| crate::util::class_name(t, db).is_some_and(|c| mwdec_lift::sig::norm_name(&c) == mwdec_lift::sig::norm_name(cls));
        // `this->m = src` with src an object of the class
        let Some(b) = ir.body.iter().position(|s| {
            matches!(s, Stmt::Assign { dst, src } if !matches!(dst, Expr::Var(_)) && cn.addr(dst, 0) == Some(CE::This(*off)) && same(&mwdec_lift::types::ty_of(dst, &ir.vars)) && same(&mwdec_lift::types::ty_of(src, &ir.vars)))
        }) else {
            continue;
        };
        let Stmt::Assign { src, .. } = &ir.body[b] else { continue };
        let src = src.clone();
        // the source: a stack temporary defined once from a value, or a parameter
        let (arg, temp) = match &src {
            Expr::Var(v) if matches!(ir.vars[*v].kind, VarKind::Stack { .. }) => {
                let defs_of: Vec<usize> = ir.body.iter().enumerate().filter(|(_, s)| matches!(s, Stmt::Assign { dst: Expr::Var(x), .. } if x == v)).map(|(i, _)| i).collect();
                let [a] = defs_of.as_slice() else { continue };
                if *a > b {
                    continue;
                }
                let Stmt::Assign { src: e, .. } = &ir.body[*a] else { continue };
                let Some(e) = init_arg(e, &defs, ir, false, *a) else { continue };
                (e, Some((*v, *a)))
            }
            e => match init_arg(e, &defs, ir, false, b) {
                Some(x) => (x, None),
                None => continue,
            },
        };
        // the rest of the copy: stores into the member or through a pointer read from it, up
        // to the temporary's destruction (a call on it)
        let on_temp = |obj: &Expr, t: VarId| matches!(obj, Expr::AddrOf(x) if matches!(&**x, Expr::Var(y) if *y == t)) || matches!(obj, Expr::Var(y) if *y == t);
        let mut remove = vec![b];
        let mut destroyed = temp.is_none();
        for k in b + 1..ir.body.len() {
            match &ir.body[k] {
                Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
                Stmt::Expr(Expr::Call { callee: mwdec_lift::Callee::Method { this: obj, .. }, args, .. }) if temp.is_some_and(|(t, _)| on_temp(obj, t) && args.iter().all(|a| !a.uses_var(t))) => {
                    remove.push(k);
                    destroyed = true;
                    break;
                }
                s => match cn.store(s) {
                    Some(st) if ce_in(&st.addr, *off, size as i32) || through_member(&st.addr, *off, size as i32) => remove.push(k),
                    _ => break,
                },
            }
        }
        if !destroyed || (need_residue && remove.len() < 2) {
            continue;
        }
        if let Some((t, a)) = temp {
            // nothing else may mention the temporary
            if ir.body.iter().enumerate().any(|(i, s)| !remove.contains(&i) && i != a && mentions(s, t)) {
                continue;
            }
            remove.push(a);
        }
        remove.sort_unstable();
        remove.dedup();
        let first = remove[0];
        for &i in remove.iter().rev() {
            ir.body.remove(i);
        }
        let n = move_earlier(ir, db, this, first, *off);
        insert_init(ir, db, mwdec_lift::Init { target: InitTarget::Member(name.clone()), ctor: None, args: vec![arg], member_ty: Some(fty.clone()) }, *off);
        moved += 1 + n;
    }
    moved
}

/// A member built explicitly whose first use is a call on its address with another object of
/// its class (`f(&this->m, &other)`, an out-of-line copy constructor without a usable name):
/// copy-constructed from that object.
fn call_copy_inits(ir: &mut IrFunction, db: &TypeDb, this: VarId, members: &[(i32, String, String, Type)]) -> usize {
    let mut moved = 0;
    for (off, cls, name, fty) in members {
        if ir.init_list.iter().any(|i| i.target == InitTarget::Member(name.clone())) {
            continue;
        }
        let Some(size) = mwdec_lift::types::size_of(Some(db), fty) else { continue };
        let defs = temp_defs(&ir.body, &ir.vars);
        let cn = Canon { this: Some(this), unset: unset_vars(ir), params: param_vars(ir), defs: &defs, db };
        let same = |t: &Type| crate::util::class_name(t, db).is_some_and(|x| mwdec_lift::sig::norm_name(&x) == mwdec_lift::sig::norm_name(cls));
        let mut found = None;
        for (k, s) in ir.body.iter().enumerate() {
            match s {
                Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
                Stmt::Expr(Expr::Call { callee: mwdec_lift::Callee::Direct { sig, .. }, args, .. }) if args.len() == 2 && !matches!(sig.ret, Type::Ptr(_) | Type::Int { .. } | Type::Float { .. } | Type::Bool) => {
                    if cn.val(&args[0], 0) == Some(CE::This(*off)) {
                        if let Some(CE::Param(p, 0)) = cn.val(&args[1], 0) {
                            let pt = &ir.vars[p].ty;
                            let ok = match strip_cv(pt) {
                                Type::Ptr(x) | Type::Ref(x) => same(strip_cv(x)),
                                t => same(t),
                            };
                            if ok {
                                found = Some((k, p));
                            }
                        }
                    }
                    break;
                }
                s => match cn.store(s) {
                    // other members' stores may come first; none into this one
                    Some(CStore { addr: CE::This(a), .. }) if a < *off || a >= off + size as i32 => {}
                    _ => break,
                },
            }
        }
        let Some((k, p)) = found else { continue };
        let arg = match strip_cv(&ir.vars[p].ty) {
            Type::Ptr(_) => Expr::Load { base: Box::new(Expr::Var(p)), offset: 0, ty: fty.clone() },
            _ => Expr::Var(p),
        };
        ir.body.remove(k);
        moved += 1 + move_earlier(ir, db, this, k, *off);
        insert_init(ir, db, mwdec_lift::Init { target: InitTarget::Member(name.clone()), ctor: None, args: vec![arg], member_ty: Some(fty.clone()) }, *off);
    }
    moved
}

/// After inline folding: members still built explicitly get a second chance (their values are
/// calls now, `m(in.ReadInt32())`), and members copy-constructed from objects.
pub fn finish(ir: &mut IrFunction, db: &TypeDb, pending: &Pending) -> usize {
    let Some(this) = ir.this_var else { return 0 };
    let mut n = 0;
    // members set before a construction the lifter moved to the initializer list
    while let Some(m) = ir.body.iter().position(|s| matches!(s, Stmt::Comment(c) if c.starts_with(mwdec_lift::idioms::INIT_MARK))) {
        let Stmt::Comment(c) = ir.body.remove(m) else { break };
        let name = &c[mwdec_lift::idioms::INIT_MARK.len()..];
        if let Some(off) = field_off(ir, db, name) {
            n += move_earlier(ir, db, this, m, off);
        }
    }
    // member constructor calls left in the body (their arguments were expanded inlines when
    // the lifter built the initializer list): entries now, with the members set before them
    loop {
        let defs = temp_defs(&ir.body, &ir.vars);
        let cn = Canon { this: Some(this), unset: unset_vars(ir), params: param_vars(ir), defs: &defs, db };
        let found = ir.body.iter().enumerate().find_map(|(k, s)| match s {
            Stmt::Expr(Expr::Call { callee: mwdec_lift::Callee::Method { sig, this: obj, .. }, args, .. }) if mwdec_lift::sig::is_ctor(sig) => {
                let Some(CE::This(off)) = cn.val(obj, 0) else { return None };
                let own = ir.sig.this_class.as_deref()?;
                let f = mwdec_lift::sig::find_class(db, own)?.fields.iter().find(|f| f.offset as i32 == off && f.bitfield.is_none())?;
                let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
                let fc = crate::util::class_name(&ft, db)?;
                let sc = sig.this_class.as_deref()?;
                if mwdec_lift::sig::norm_name(sc) != mwdec_lift::sig::norm_name(&fc) || ir.init_list.iter().any(|i| i.target == InitTarget::Member(f.name.clone())) {
                    return None;
                }
                let a: Option<Vec<Expr>> = args.iter().map(|a| init_arg(a, &defs, ir, true, k)).collect();
                Some((k, f.name.clone(), off, sig.clone(), a?))
            }
            _ => None,
        });
        let Some((k, name, off, sig, args)) = found else { break };
        ir.body.remove(k);
        n += 1 + move_earlier(ir, db, this, k, off);
        insert_init(ir, db, mwdec_lift::Init { target: InitTarget::Member(name), ctor: Some(sig), args, member_ty: None }, off);
    }
    // members of classes without a known default construction, copied and then finished by
    // a copy constructor's expansion: copy-constructed too
    let others: Vec<(i32, String, String, Type)> = ir
        .sig
        .this_class
        .as_deref()
        .and_then(|own| mwdec_lift::sig::find_class(db, own))
        .map(|c| {
            c.fields
                .iter()
                .filter(|f| f.bitfield.is_none() && !pending.0.iter().any(|p| p.2 == f.name))
                .filter_map(|f| {
                    let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
                    let cls = crate::util::class_name(&ft, db)?;
                    Some((f.offset as i32, cls, f.name.clone(), ft))
                })
                .collect()
        })
        .unwrap_or_default();
    n += copy_inits(ir, db, this, &others, true);
    if !pending.0.is_empty() {
        n += explicit_inits(ir, db, this, &pending.0);
        n += copy_inits(ir, db, this, &pending.0, false);
        n += call_copy_inits(ir, db, this, &pending.0);
    }
    // members copy-constructed in the initializer list: the rest of their inline copy
    // constructor at the start of the body is implicit
    let copies: Vec<(i32, i32, Option<VarId>)> = ir
        .init_list
        .iter()
        .filter_map(|i| {
            let InitTarget::Member(name) = &i.target else { return None };
            let [arg] = i.args.as_slice() else { return None };
            let off = field_off(ir, db, name)?;
            let own = ir.sig.this_class.as_deref()?;
            let f = mwdec_lift::sig::find_class(db, own)?.fields.iter().find(|f| f.name == *name)?;
            let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
            crate::util::class_name(&ft, db)?;
            // the source object, when a parameter or local: an ownership transfer clears it
            let root = match strip_casts(arg) {
                Expr::Var(v) if matches!(ir.vars[*v].kind, VarKind::Param { .. } | VarKind::Stack { .. }) => Some(*v),
                _ => None,
            };
            Some((off, mwdec_lift::types::size_of(Some(db), &ft)? as i32, root))
        })
        .collect();
    for (off, size, root) in copies {
        let defs = temp_defs(&ir.body, &ir.vars);
        let cn = Canon { this: Some(this), unset: unset_vars(ir), params: param_vars(ir), defs: &defs, db };
        let mut gone = copy_residue(ir, &cn, &defs, 0, off, size);
        if let (Some(r), true) = (root, gone.is_empty()) {
            // `*(bool*)&src = false` (an owning pointer's transfer) right at the start
            for (k, s) in ir.body.iter().enumerate() {
                match s {
                    Stmt::Assign { dst, src } if !src.has_call() && lvalue_root(dst) == Some(r) => gone.push(k),
                    _ => break,
                }
            }
        }
        n += gone.len();
        for &k in gone.iter().rev() {
            ir.body.remove(k);
        }
    }
    // `m(T(args))` for a member of class T: `m(args)` (no temporary to copy from)
    let own = ir.sig.this_class.clone().unwrap_or_default();
    for init in ir.init_list.iter_mut() {
        let InitTarget::Member(name) = &init.target else { continue };
        let Some(ft) = mwdec_lift::sig::find_class(db, &own).and_then(|c| c.fields.iter().find(|f| f.name == *name)).map(|f| mwdec_lift::types::resolve(Some(db), &f.ty).into_owned()) else { continue };
        let Some(fc) = crate::util::class_name(&ft, db) else { continue };
        if let [Expr::Construct { class, ctor, args }] = init.args.as_slice() {
            // (the same template: its arguments may be spelled through typedefs)
            let tbase = |n: &str| mwdec_lift::sig::norm_name(n.split('<').next().unwrap_or(n));
            let same = crate::util::class_name(class, db).is_some_and(|c| mwdec_lift::sig::norm_name(&c) == mwdec_lift::sig::norm_name(&fc))
                || matches!(strip_cv(class), Type::Named(n) if n.contains('<') && tbase(n) == tbase(&fc));
            if same && !args.is_empty() {
                let (cs, args) = (ctor.clone(), args.clone());
                init.ctor = cs;
                init.args = args;
                init.member_ty = None;
                n += 1;
            }
        }
    }
    if n > 0 {
        let vars = ir.vars.clone();
        drop_unused_temps(&mut ir.body, &vars);
    }
    n
}

/// Members whose default construction [`strip`] didn't find: built explicitly.
#[derive(Default, Clone, Debug)]
pub struct Pending(Vec<(i32, String, String, Type)>);

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

/// One step of an inline copy constructor: a store, or a call (by qualified name).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum COp {
    Store(CStore),
    Call { name: String, args: Vec<CE> },
}

/// Copy constructions of classes (canonical steps; the source object is `CE::Param`).
pub type CopyCtors = HashMap<String, Vec<COp>>;

const CPROBE: &str = "__mwdec_cc";

fn copy_probe_text(spelled: &str, k: usize) -> String {
    format!("struct {CPROBE}{k} {{ {spelled} m; {CPROBE}{k}(const {spelled}& o); }};\n{CPROBE}{k}::{CPROBE}{k}(const {spelled}& o) : m(o) {{}}\n")
}

fn call_name(c: &mwdec_lift::Callee) -> Option<String> {
    match c {
        mwdec_lift::Callee::Method { sig, .. } | mwdec_lift::Callee::Direct { sig, .. } => Some(format!("{}/{}", sig.qualified_name, sig.params.len())),
        _ => None,
    }
}

impl Canon<'_> {
    /// A call argument: its value, or for an object passed by reference its address.
    fn arg(&self, x: &Expr) -> Option<CE> {
        self.val(x, 0).or_else(|| match x {
            Expr::Load { .. } | Expr::Member { .. } | Expr::Var(_) => self.addr(x, 0),
            Expr::Cast { e, .. } => self.arg(e),
            _ => None,
        })
    }

    fn op(&self, s: &Stmt) -> Option<COp> {
        match s {
            Stmt::Expr(Expr::Call { callee, args, .. }) => {
                let name = call_name(callee)?;
                let mut a = vec![];
                if let mwdec_lift::Callee::Method { this: obj, .. } = callee {
                    a.push(self.val(obj, 0)?);
                }
                for x in args {
                    a.push(self.arg(x)?);
                }
                Some(COp::Call { name, args: a })
            }
            s => self.store(s).map(COp::Store),
        }
    }
}

fn ops_from_probe(ir: &IrFunction, db: &TypeDb) -> Option<Vec<COp>> {
    let defs = temp_defs(&ir.body, &ir.vars);
    let c = Canon { this: ir.this_var, unset: unset_vars(ir), params: param_vars(ir), defs: &defs, db };
    let mut out = vec![];
    // constructor calls the lifter put in the initializer list (bases of the member)
    let own = ir.sig.this_class.clone().unwrap_or_default();
    for i in &ir.init_list {
        let sig = i.ctor.as_ref()?;
        let off = match &i.target {
            InitTarget::Member(_) => 0,
            // (the probe's own class is unknown to the context: its member is at 0)
            InitTarget::Base(b) => mwdec_lift::sig::find_class(db, &own).and_then(|k| k.bases.iter().find(|x| mwdec_lift::sig::norm_name(&x.name) == mwdec_lift::sig::norm_name(b))).map_or(0, |x| x.offset as i32),
        };
        let mut a = vec![CE::This(off)];
        for x in &i.args {
            a.push(c.arg(x)?);
        }
        out.push(COp::Call { name: format!("{}/{}", sig.qualified_name, sig.params.len()), args: a });
    }
    for s in &ir.body {
        match s {
            Stmt::Assign { dst: Expr::Var(v), .. } if defs.contains_key(v) => {}
            Stmt::Return(None) => {}
            s => out.push(c.op(s)?),
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Copy constructions of `classes`, from the cache or by compiling probes.
pub fn build_copies(db: &TypeDb, classes: &[String], cache: Option<&DefCache>, compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync)) -> CopyCtors {
    let mut out = CopyCtors::new();
    let mut todo: Vec<(String, String)> = vec![];
    for c in classes {
        let Some(sp) = crate::probe::spell(&Type::Named(c.clone()), db, &[]) else { continue };
        match cache.and_then(|ca| ca.get::<Vec<COp>>(&copy_probe_text(&sp, 0))) {
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
    let lift = |obj: &ObjectFile, k: usize| -> Option<Vec<COp>> {
        let name = format!("{CPROBE}{k}");
        let f = obj.functions.iter().find(|f| f.name.starts_with("__ct__") && f.name.contains(&format!("{}{name}F", name.len())))?;
        let ir = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mwdec_lift::lift_function(obj, f, Some(db)))).ok()?.ok()?;
        let r = ops_from_probe(&ir, db);
        if std::env::var("MWDI_DEFCTOR_DEBUG").is_ok() {
            eprintln!("copy probe {name}: {r:?} init {:?} body {:?}", ir.init_list, ir.body);
        }
        r
    };
    let all: String = todo.iter().enumerate().map(|(k, (_, sp))| copy_probe_text(sp, k)).collect();
    let results: Vec<Option<Vec<COp>>> = match compile(&all) {
        Ok(obj) => (0..todo.len()).map(|k| lift(&obj, k)).collect(),
        Err(_) => todo.iter().map(|(_, sp)| compile(&copy_probe_text(sp, 0)).ok().and_then(|o| lift(&o, 0))).collect(),
    };
    for ((c, sp), r) in todo.iter().zip(results) {
        if let Some(ca) = cache {
            ca.put(&copy_probe_text(sp, 0), &r);
        }
        if let Some(v) = r {
            out.insert(c.clone(), v);
        }
    }
    out
}

/// `e` with `This(k)` shifted by `by` and the probe's source parameter bound to `bind`.
fn bind_ce(e: &CE, by: i32, src: usize, bind: usize) -> CE {
    match e {
        CE::This(k) => CE::This(k + by),
        CE::Param(v, k) if *v == src => CE::Param(bind, *k),
        CE::Load(a, s) => CE::Load(Box::new(bind_ce(a, by, src, bind)), *s),
        CE::Bin(o, a, b) => CE::Bin(o.clone(), Box::new(bind_ce(a, by, src, bind)), Box::new(bind_ce(b, by, src, bind))),
        other => other.clone(),
    }
}

fn first_param(ops: &[COp]) -> Option<usize> {
    fn find(e: &CE) -> Option<usize> {
        match e {
            CE::Param(v, _) => Some(*v),
            CE::Load(a, _) => find(a),
            CE::Bin(_, a, b) => find(a).or_else(|| find(b)),
            _ => None,
        }
    }
    ops.iter().find_map(|o| match o {
        COp::Store(s) => find(&s.addr).or_else(|| find(&s.val)),
        COp::Call { args, .. } => args.iter().find_map(find),
    })
}

/// Members whose inline copy constructor's steps appear in the body (in order), copying a
/// parameter: `m(param)`.
pub fn copy_ctor_inits(ir: &mut IrFunction, db: &TypeDb, cc: &CopyCtors) -> usize {
    let Some(this) = ir.this_var else { return 0 };
    let Some(own) = ir.sig.this_class.clone() else { return 0 };
    let Some(c) = mwdec_lift::sig::find_class(db, &own).cloned() else { return 0 };
    if cc.is_empty() || !ir.symbol.starts_with("__ct__") {
        return 0;
    }
    let mut moved = 0;
    for f in &c.fields {
        if f.bitfield.is_some() || ir.init_list.iter().any(|i| i.target == InitTarget::Member(f.name.clone())) {
            continue;
        }
        let ft = mwdec_lift::types::resolve(Some(db), &f.ty).into_owned();
        let Some(cls) = crate::util::class_name(&ft, db) else { continue };
        let Some(ops) = cc.get(&cls) else { continue };
        let Some(src) = first_param(ops) else { continue };
        let off = f.offset as i32;
        let defs = temp_defs(&ir.body, &ir.vars);
        let cn = Canon { this: Some(this), unset: unset_vars(ir), params: param_vars(ir), defs: &defs, db };
        let body_ops: Vec<Option<COp>> = ir.body.iter().map(|s| cn.op(s)).collect();
        let mut params: Vec<VarId> = param_vars(ir).into_iter().collect();
        params.sort_unstable();
        let mut found = None;
        for p in params {
            let mut pick = vec![];
            let mut at = 0;
            for o in ops {
                let want = match o {
                    COp::Store(s) => COp::Store(CStore { addr: bind_ce(&s.addr, off, src, p), size: s.size, val: bind_ce(&s.val, off, src, p) }),
                    COp::Call { name, args } => COp::Call { name: name.clone(), args: args.iter().map(|a| bind_ce(a, off, src, p)).collect() },
                };
                match (at..body_ops.len()).find(|&k| body_ops[k].as_ref() == Some(&want)) {
                    Some(k) => {
                        pick.push(k);
                        at = k + 1;
                    }
                    None => {
                        pick.clear();
                        break;
                    }
                }
            }
            if !pick.is_empty() {
                found = Some((p, pick));
                break;
            }
        }
        let Some((p, pick)) = found else { continue };
        let pt = ir.vars[p].ty.clone();
        let same = |t: &Type| crate::util::class_name(t, db).is_some_and(|x| mwdec_lift::sig::norm_name(&x) == mwdec_lift::sig::norm_name(&cls));
        let arg = match strip_cv(&pt) {
            Type::Ptr(x) if same(strip_cv(x)) => Expr::Load { base: Box::new(Expr::Var(p)), offset: 0, ty: ft.clone() },
            Type::Ref(x) if same(strip_cv(x)) => Expr::Var(p),
            t if same(t) => Expr::Var(p),
            _ => continue,
        };
        let first = pick[0];
        for &k in pick.iter().rev() {
            ir.body.remove(k);
        }
        moved += 1 + move_earlier(ir, db, this, first, off);
        insert_init(ir, db, mwdec_lift::Init { target: InitTarget::Member(f.name.clone()), ctor: None, args: vec![arg], member_ty: Some(ft.clone()) }, off);
        let vars = ir.vars.clone();
        drop_unused_temps(&mut ir.body, &vars);
    }
    moved
}

