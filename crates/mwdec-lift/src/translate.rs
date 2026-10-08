//! PPC -> per-block statements: reaching definitions, register webs (phi variables), symbolic
//! execution of each block with SSA-like temporaries, calls/vcalls, literals, stack slots.

use crate::cfg::{Cfg, Term};
use crate::frame::Frame;
use crate::insn::*;
use crate::ir::*;
use crate::sig::{self, sig_of};
use crate::types::{self, ty_of};
use mwdec_core::{FuncSig, Function, ObjectFile, RelocKind, Type, TypeDb};
use ppc750cl::Opcode;
use std::collections::{BTreeMap, HashMap, HashSet};

pub const ENTRY: u32 = u32::MAX;
pub type Site = (u32, Reg);

#[derive(Clone, Debug, PartialEq)]
pub enum ArgLoc {
    Gpr(u8),
    Fpr(u8),
    GprPair(u8),
    Stack,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub sret: Option<u8>,
    pub this: Option<u8>,
    pub params: Vec<ArgLoc>,
}

/// Argument register layout for a signature under the PPC EABI.
pub fn layout(sig: &FuncSig, has_this: bool, sret: bool, db: Option<&TypeDb>) -> Layout {
    let mut g = 3u8;
    let mut fl = 1u8;
    let mut out = Layout { sret: None, this: None, params: vec![] };
    if sret {
        out.sret = Some(g);
        g += 1;
    }
    if has_this {
        out.this = Some(g);
        g += 1;
    }
    for p in &sig.params {
        let t = types::resolve(db, &p.ty).into_owned();
        let t = strip_cv(&t).clone();
        let loc = match &t {
            Type::Float { .. } => {
                if fl <= 8 {
                    fl += 1;
                    ArgLoc::Fpr(fl - 1)
                } else {
                    ArgLoc::Stack
                }
            }
            Type::Int { size: 8, .. } => {
                if g % 2 == 0 {
                    g += 1;
                }
                if g + 1 <= 10 {
                    g += 2;
                    ArgLoc::GprPair(g - 2)
                } else {
                    ArgLoc::Stack
                }
            }
            _ => {
                if g <= 10 {
                    g += 1;
                    ArgLoc::Gpr(g - 1)
                } else {
                    ArgLoc::Stack
                }
            }
        };
        out.params.push(loc);
    }
    out
}

#[derive(Clone, Debug)]
enum Ca {
    Ne0(Expr),
    Eq0(Expr),
    /// CA from `srawi x, n` (for `addze` signed division by 2^n)
    Srawi(Expr, u8),
    /// CA from `subfc t, a, b` (= b >=u a), for the branchless `a <= b` idiom
    Subfc(Expr, Expr),
    /// CA from `addc t, a, b` (the low-word add of a 64-bit addition)
    Addc(Expr, Expr),
    Unknown,
}

#[derive(Clone)]
struct St {
    regs: Vec<Option<Expr>>,
    /// def site currently held by each register (for snapshot fixups)
    site: Vec<Option<Site>>,
    cr: Vec<Option<Expr>>,
    ca: Ca,
    /// stack conversion scratch: offset -> (size, value)
    mem: BTreeMap<i32, (u32, Expr)>,
    out: Vec<Stmt>,
}

#[derive(Clone, Debug, Default)]
pub struct StackModel {
    /// scalar slots: offset -> var
    pub slots: BTreeMap<i32, VarId>,
    /// address-taken regions: (start, size, var)
    pub regions: Vec<(i32, u32, VarId)>,
    /// conversion scratch offsets (8-byte pairs / psq)
    pub conv: HashSet<i32>,
}

pub struct BlockOut {
    pub stmts: Vec<Stmt>,
    /// condition for taking the branch (Cond / CondReturn terms)
    pub cond: Option<Expr>,
    pub switch: Option<Expr>,
    pub ret: Option<Expr>,
}

/// Declared size of an extern addressed absolutely whose size the object doesn't know: anything
/// above the small-data limit (8 bytes) keeps the compiler off small-data addressing.
const FAR_EXTERN_SIZE: u32 = 16;

pub struct Lifter<'a> {
    pub obj: &'a ObjectFile,
    pub f: &'a Function,
    pub db: Option<&'a TypeDb>,
    pub insns: Vec<Insn>,
    pub cfg: Cfg,
    pub frame: Frame,
    pub sig: FuncSig,
    pub vars: Vec<Var>,
    /// temps (SSA, single assignment) vs mutable vars
    pub is_temp: Vec<bool>,
    pub this_var: Option<VarId>,
    pub sret_var: Option<VarId>,
    /// Treat r3 as a hidden struct-return pointer although the signature doesn't say so.
    pub force_sret: bool,
    /// Never guess a struct return.
    pub no_sret_guess: bool,
    /// The unit's compiler gives every parameter a stack slot below the locals whenever the
    /// frame has a local area (GC/1.2.5n), so local offsets tell the parameter count.
    pub param_home_slots: bool,
    pub params: Vec<VarId>,
    pub decl_params: Vec<String>,
    entry_vals: HashMap<Reg, Expr>,
    rd_in: Vec<Vec<Vec<u32>>>,
    use_count: HashMap<Site, u32>,
    use_blocks: HashMap<Site, HashSet<usize>>,
    web_var: HashMap<Site, VarId>,
    def_value: HashMap<Site, Expr>,
    cr_value: HashMap<u32, [Option<Expr>; 4]>,
    pub temp_def: HashMap<VarId, Expr>,
    call_layouts: HashMap<usize, (FuncSig, Layout, bool)>,
    pub ret_reg: Option<Reg>,
    pub ret_ty: Type,
    pub stack: StackModel,
    pub blocks_out: Vec<BlockOut>,
    pub warnings: Vec<String>,
    pub globals: BTreeMap<String, GlobalRef>,
    switch_index: HashMap<usize, Expr>,
    /// incoming stack parameters (param index, value), in order
    stack_params: Vec<(usize, Expr)>,
    /// incoming stack parameter slots: r1 offset -> value
    in_stack: HashMap<i32, Expr>,
    /// outgoing stack argument slots (r1 offsets), captured as values for the call
    out_stack: HashSet<i32>,
    /// Pure return blocks (epilogue + `blr`) whose return-value phi is split: every predecessor
    /// returns its own value (`return this; ... return w; ... return nullptr;`)
    pub split_returns: HashSet<usize>,
    /// CR snapshots at `mfcr` instructions
    mfcr_vals: HashMap<usize, Vec<Option<Expr>>>,
    /// string literal -> (pool symbol, offset) it was read from
    str_origin: HashMap<Vec<u8>, (String, i64)>,
    /// The one-byte zero literal read as `""` (a `const char&` argument's constant).
    empty_str_origin: Option<(String, i64)>,
    /// Stack objects whose constructor's return value (`this`) is used afterwards.
    pub ctor_ret_used: HashSet<VarId>,
}

fn is_literal_name(n: &str) -> bool {
    n.starts_with("lbl_") || n.starts_with('@') || n.starts_with("...") || n.starts_with("$$")
}

pub fn is_trivial(e: &Expr) -> bool {
    match e {
        Expr::Int { .. } | Expr::Float { .. } | Expr::Str { .. } | Expr::Var(_) | Expr::FuncAddr { .. } => true,
        Expr::AddrOf(inner) => is_addr_trivial(inner),
        Expr::Cast { e, .. } => matches!(**e, Expr::Var(_) | Expr::Int { .. }),
        Expr::Unknown { text, .. } => text.starts_with("ha:") || text.starts_with("mfcr:"),
        _ => false,
    }
}

fn is_addr_trivial(e: &Expr) -> bool {
    match e {
        Expr::Global { .. } | Expr::Var(_) => true,
        Expr::Member { base, .. } => is_addr_trivial(base),
        Expr::Load { base, .. } => matches!(**base, Expr::Var(_)) || matches!(&**base, Expr::AddrOf(b) if is_addr_trivial(b)),
        _ => false,
    }
}

/// Does `e` read the value of `v` (taking `&v` or `&v.member` does not)?
pub fn reads_var(e: &Expr, v: VarId) -> bool {
    match e {
        Expr::Var(x) => *x == v,
        Expr::AddrOf(inner) => match &**inner {
            Expr::Var(_) => false,
            Expr::Member { base, .. } if matches!(**base, Expr::Var(_)) => false,
            other => reads_var_inner(other, v),
        },
        other => reads_var_inner(other, v),
    }
}

fn reads_var_inner(e: &Expr, v: VarId) -> bool {
    let mut found = false;
    let mut f = |x: &Expr| {
        if reads_var_shallow(x, v) {
            found = true;
        }
    };
    match e {
        Expr::AddrOf(_) | Expr::Var(_) => return reads_var(e, v),
        _ => {}
    }
    // check children via recursion on immediate sub-expressions
    match e {
        Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => return reads_var(base, v) || matches!(e, Expr::Member { base, .. } if matches!(**base, Expr::Var(x) if x == v)),
        Expr::Unary { e: x, .. } | Expr::Cast { e: x, .. } => return reads_var(x, v),
        Expr::Binary { l, r, .. } => return reads_var(l, v) || reads_var(r, v),
        Expr::Index { base, index, .. } => return reads_var(base, v) || reads_var(index, v),
        _ => {}
    }
    e.walk(&mut f);
    found
}

fn reads_var_shallow(e: &Expr, v: VarId) -> bool {
    matches!(e, Expr::Var(x) if *x == v)
}

pub fn reads_mem(e: &Expr) -> bool {
    let mut r = false;
    e.walk(&mut |x| match x {
        Expr::Load { .. } | Expr::Index { .. } | Expr::Global { .. } | Expr::Call { .. } => r = true,
        _ => {}
    });
    // &global / &this->x are not reads
    if r {
        if let Expr::AddrOf(inner) = e {
            return addr_reads(inner);
        }
    }
    r
}

fn addr_reads(e: &Expr) -> bool {
    match e {
        Expr::Global { .. } | Expr::Var(_) => false,
        Expr::Load { base, .. } | Expr::Member { base, .. } => reads_mem(base),
        Expr::Index { base, index, .. } => reads_mem(base) || reads_mem(index),
        _ => true,
    }
}

fn mask32(mb: u8, me: u8) -> u32 {
    let a = u32::MAX >> mb;
    let b = u32::MAX << (31 - me);
    if mb <= me {
        a & b
    } else {
        a | b
    }
}

impl<'a> Lifter<'a> {
    pub fn new(obj: &'a ObjectFile, f: &'a Function, db: Option<&'a TypeDb>) -> Lifter<'a> {
        let insns = crate::cfg::decode(f);
        let cfg = Cfg::build(obj, f, &insns);
        let frame = crate::frame::analyze(&insns, &cfg);
        let sig = sig_of(&f.name, db);
        let nb = cfg.blocks.len();
        Lifter {
            obj,
            f,
            db,
            insns,
            cfg,
            frame,
            sig,
            vars: vec![],
            is_temp: vec![],
            this_var: None,
            sret_var: None,
            force_sret: false,
            no_sret_guess: false,
            param_home_slots: false,
            params: vec![],
            decl_params: vec![],
            entry_vals: HashMap::new(),
            rd_in: vec![vec![]; nb],
            use_count: HashMap::new(),
            use_blocks: HashMap::new(),
            web_var: HashMap::new(),
            def_value: HashMap::new(),
            cr_value: HashMap::new(),
            temp_def: HashMap::new(),
            call_layouts: HashMap::new(),
            ret_reg: None,
            split_returns: HashSet::new(),
            ret_ty: Type::Void,
            stack: StackModel::default(),
            blocks_out: vec![],
            warnings: vec![],
            globals: BTreeMap::new(),
            switch_index: HashMap::new(),
            stack_params: vec![],
            in_stack: HashMap::new(),
            out_stack: HashSet::new(),
            mfcr_vals: HashMap::new(),
            str_origin: HashMap::new(),
            empty_str_origin: None,
            ctor_ret_used: HashSet::new(),
        }
    }

    pub fn new_var(&mut self, name: String, ty: Type, kind: VarKind, temp: bool) -> VarId {
        self.vars.push(Var { name, ty, kind });
        self.is_temp.push(temp);
        self.vars.len() - 1
    }

    fn warn(&mut self, s: String) {
        if self.warnings.len() < 64 {
            self.warnings.push(s);
        }
    }

    // ------------------------------------------------------------ setup

    fn setup_params(&mut self) {
        let db = self.db;
        let has_this = self.sig.this_class.is_some() && !self.sig.is_static && !self.guess_static();
        let sret_guess = self.guess_sret(has_this);
        let sret = (types::is_aggregate(db, &self.sig.ret) && !sig::ret_unknown(&self.sig)) || sret_guess;
        let lay = layout(&self.sig, has_this, sret, db);
        if let Some(r) = lay.sret {
            // a guessed struct return gets its class from how the object is built (idioms)
            let ty = t_ptr(if sret_guess { t_unk(0) } else { self.sig.ret.clone() });
            let v = self.new_var("__return".into(), ty, VarKind::StructRet, false);
            self.sret_var = Some(v);
            self.entry_vals.insert(gpr(r), Expr::Var(v));
        }
        if let Some(r) = lay.this {
            let cls = Type::Named(self.sig.this_class.clone().unwrap());
            let pt = if self.sig.is_const { Type::Const(Box::new(cls)) } else { cls };
            let v = self.new_var("this".into(), t_ptr(pt), VarKind::This, false);
            self.this_var = Some(v);
            self.entry_vals.insert(gpr(r), Expr::Var(v));
            if sig::is_dtor(&self.sig) && self.sig.params.is_empty() {
                // deleting-destructor flag (short) in the next register
                let f = self.new_var("__dtor_flag".into(), t_int(2, true), VarKind::Hidden, false);
                self.entry_vals.insert(gpr(r + 1), Expr::Var(f));
            }
        }
        let demangled = sig::demangle(&self.f.name).unwrap_or_default();
        let mut spellings = param_spellings(&demangled);
        // a header's `const` on a by-value scalar parameter (not in the mangled name; MWCC
        // trusts a const bool parameter's value where it would re-truncate a plain one)
        if let Some(db) = db {
            for (i, c) in header_const_params(db, &self.sig).into_iter().enumerate() {
                if let Some(s) = spellings.get_mut(i).filter(|s| c && !s.is_empty() && !s.starts_with("const ") && !s.contains('*') && !s.contains('&')) {
                    *s = format!("const {s}");
                }
            }
        }
        for (i, p) in self.sig.params.clone().iter().enumerate() {
            let name = p.name.clone().filter(|n| !n.is_empty()).unwrap_or_else(|| format!("arg{i}"));
            // References are objects in C++: the register holds their address.
            let resolved = types::resolve(db, &p.ty).into_owned();
            let (vty, by_addr) = match (strip_cv(&p.ty), strip_cv(&resolved)) {
                (Type::Ref(inner), _) => ((**inner).clone(), true),
                // array parameters decay to pointers
                (_, Type::Array(e, _)) => (t_ptr((**e).clone()), false),
                _ => (p.ty.clone(), types::is_aggregate(db, &p.ty) && !is_ptr(&p.ty)),
            };
            let v = self.new_var(name, vty, VarKind::Param { index: i }, false);
            self.params.push(v);
            self.decl_params.push(spellings.get(i).cloned().unwrap_or_default());
            match lay.params[i] {
                ArgLoc::Gpr(r) => {
                    let val = if by_addr { Expr::AddrOf(Box::new(Expr::Var(v))) } else { Expr::Var(v) };
                    self.entry_vals.insert(gpr(r), val);
                }
                ArgLoc::Fpr(r) => {
                    self.entry_vals.insert(fpr(r), Expr::Var(v));
                }
                ArgLoc::GprPair(r) if !self.param_home_slots => {
                    self.entry_vals.insert(gpr(r), Expr::Var(v));
                }
                ArgLoc::GprPair(r) => {
                    // (SDK compiler units) high word in the first register of the pair (typed as
                    // the 64-bit integer a typedef like `OSTime` stands for, so its halves are
                    // recognized)
                    let rt = types::resolve(self.db, &self.vars[v].ty).into_owned();
                    if crate::wide::is_wide(&rt) && !crate::wide::is_wide(&self.vars[v].ty) {
                        self.vars[v].ty = rt;
                    }
                    self.entry_vals.insert(gpr(r), crate::wide::hi32(Expr::Var(v)));
                    self.entry_vals.insert(gpr(r + 1), crate::wide::lo32(Expr::Var(v)));
                }
                ArgLoc::Stack => {
                    let val = if by_addr { Expr::AddrOf(Box::new(Expr::Var(v))) } else { Expr::Var(v) };
                    self.stack_params.push((i, val));
                }
            }
        }
        if self.sig.variadic {
            self.decl_params.push("...".into());
        }
    }

    /// Struct return of a function the TypeDb doesn't declare (mangled names carry no return
    /// type): the register after the last parameter is read before being written, so the
    /// parameters start one register later, after the hidden return-object pointer.
    fn guess_sret(&self, has_this: bool) -> bool {
        if self.force_sret && sig::ret_unknown(&self.sig) {
            return true;
        }
        if self.no_sret_guess {
            return false;
        }
        if !sig::ret_unknown(&self.sig) || sig::is_ctor(&self.sig) || sig::is_dtor(&self.sig) || self.sig.variadic || sig::demangle(&self.f.name).is_none() {
            return false;
        }
        let lay = layout(&self.sig, has_this, false, self.db);
        let mut gprs: Vec<u8> = lay.params.iter().filter_map(|l| if let ArgLoc::Gpr(r) = l { Some(*r) } else { None }).collect();
        if let Some(t) = lay.this {
            gprs.push(t);
        }
        let next = gprs.iter().max().map_or(3, |r| r + 1);
        if next > 10 || lay.params.iter().any(|l| matches!(l, ArgLoc::Stack | ArgLoc::GprPair(_))) {
            return false;
        }
        let mut written: HashSet<Reg> = HashSet::new();
        for (k, i) in self.insns.iter().enumerate().take(64) {
            if self.frame.skip.contains(&k) {
                continue;
            }
            let (d, u) = defs_uses(i);
            if i.is_call() || i.is_bctrl() {
                // a constructor run on the unwritten r3: the object being returned
                if next == 3 && !written.contains(&gpr(3)) && i.is_call() {
                    if let Some(r) = &i.reloc {
                        if sig::is_ctor(&sig_of(&r.target, self.db)) {
                            return true;
                        }
                    }
                }
                break;
            }
            if u.contains(&gpr(next)) && !written.contains(&gpr(next)) {
                return true;
            }
            written.extend(d);
            if i.op() == Opcode::Bc || i.op() == Opcode::B || i.op() == Opcode::Bclr {
                break;
            }
        }
        false
    }

    /// Without a TypeDb we can't tell static members from methods; guess static when the last
    /// GPR parameter register under the "member" layout is never read before being written.
    fn guess_static(&self) -> bool {
        if self.db.map_or(false, |db| crate::sig::find_class(db, self.sig.this_class.as_deref().unwrap_or("")).is_some()) {
            return false; // the DB would have set is_static
        }
        if sig::is_ctor(&self.sig) || sig::is_dtor(&self.sig) || self.sig.is_const {
            return false;
        }
        let name = sig::split_scope(&self.sig.qualified_name).1;
        if name.starts_with("operator") {
            return false;
        }
        let lay = layout(&self.sig, true, false, self.db);
        let gprs: Vec<u8> = lay.params.iter().filter_map(|l| if let ArgLoc::Gpr(r) = l { Some(*r) } else { None }).collect();
        // read-before-write scan of the entry region (linear, first 64 insns)
        let mut written: HashSet<Reg> = HashSet::new();
        let mut read: HashSet<Reg> = HashSet::new();
        for (k, i) in self.insns.iter().enumerate().take(64) {
            if self.frame.skip.contains(&k) {
                continue;
            }
            let (d, u) = defs_uses(i);
            if i.is_call() || i.is_bctrl() {
                break;
            }
            for r in u {
                if !written.contains(&r) {
                    read.insert(r);
                }
            }
            written.extend(d);
            if i.op() == Opcode::Bc || i.op() == Opcode::B || i.op() == Opcode::Bclr {
                break;
            }
        }
        match gprs.last() {
            Some(&last) => !read.contains(&gpr(last)) && read.contains(&gpr(last - 1)) && last > 3,
            None => false,
        }
    }

    // ------------------------------------------------------------ calls

    fn call_layout(&self, k: usize) -> Option<(FuncSig, Layout, bool)> {
        let i = &self.insns[k];
        let r = i.reloc.as_ref()?;
        // a call through a pointer to member function: `__ptmf_scall(this, args...)` with the
        // address of the pointer-to-member object in r12 (arguments: this, &pmf, r4.., f1..)
        if r.target == "__ptmf_scall" || r.target == "__ptmf_scall4" {
            let mut params = vec![ArgLoc::Gpr(3), ArgLoc::Gpr(12)];
            let set = (4..=10u8).rev().find(|q| self.reg_set_for_call(k, gpr(*q)));
            if let Some(g) = set.max(self.passed_through(k).filter(|&r| r >= 4)) {
                params.extend((4..=g).map(ArgLoc::Gpr));
            }
            if let Some(fm) = (1..=8u8).rev().find(|q| self.reg_set_for_call(k, fpr(*q))) {
                params.extend((1..=fm).map(ArgLoc::Fpr));
            }
            let mut s = builtin_sig(&r.target, params.len());
            for (p, l) in s.params.iter_mut().zip(&params) {
                if matches!(l, ArgLoc::Fpr(_)) {
                    p.ty = t_f32();
                }
            }
            return Some((s, Layout { sret: None, this: None, params }, false));
        }
        let mut s = sig_of(&r.target, self.db);
        // 64-bit runtime helpers take (hi, lo, count) / (hi, lo, hi2, lo2) in r3..
        if let Some((_, _, pair)) = crate::wide::helper(&r.target) {
            s.params = (0..if pair { 4 } else { 3 }).map(|_| mwdec_core::Param { name: None, ty: t_u32() }).collect();
            s.this_class = None;
            let lay = layout(&s, false, false, self.db);
            return Some((s, lay, false));
        }
        // C callee without a prototype in the context: arguments are the registers set up
        // for this call (up to the highest one written since the previous call)
        if sig::demangle(&r.target).is_none()
            && s.params.is_empty()
            && !self.db.map_or(false, |db| db.decls.contains_key(&r.target) || db.functions.contains_key(&r.target))
        {
            // words stored to the outgoing parameter area: the GPR arguments are all used, and
            // registers still holding this function's own arguments are passed on
            let outgoing = self.outgoing_stack_words(k);
            let gmax = if outgoing > 0 { Some(10) } else { (3..=10u8).rev().find(|q| self.reg_set_for_call(k, gpr(*q))) };
            if let Some(g) = gmax {
                for _ in 3..=g {
                    s.params.push(mwdec_core::Param { name: None, ty: t_s32() });
                }
            }
            let fset = (1..=8u8).rev().find(|q| self.reg_set_for_call(k, fpr(*q)));
            let fpass = if outgoing > 0 { (1..=8u8).rev().find(|q| self.entry_val_intact(k, fpr(*q))) } else { None };
            if let Some(fm) = fset.max(fpass) {
                for _ in 1..=fm {
                    s.params.push(mwdec_core::Param { name: None, ty: t_f32() });
                }
            }
            for _ in 0..outgoing {
                s.params.push(mwdec_core::Param { name: None, ty: t_s32() });
            }
        }
        let sret = !sig::ret_unknown(&s) && types::is_aggregate(self.db, &s.ret);
        let mut has_this = s.this_class.is_some() && !s.is_static;
        if has_this && self.db.map_or(true, |db| crate::sig::find_class(db, s.this_class.as_deref().unwrap()).is_none()) {
            // guess: member if the register after the last explicit param was set in this block
            let lay = layout(&s, true, sret, self.db);
            let last_g = lay.params.iter().rev().find_map(|l| if let ArgLoc::Gpr(r) = l { Some(*r) } else { None });
            // without explicit GPR parameters the register to test is `this` (r3) itself
            if let Some(last) = last_g.or(lay.this) {
                // (a register the block reads as an operand before the call, never writing it
                // after, holds a value of this function, not the last argument)
                if !sig::is_ctor(&s) && !sig::is_dtor(&s) && !s.is_const && (!self.reg_set_before(k, gpr(last)) || self.reg_read_last_in_block(k, gpr(last))) {
                    has_this = false;
                }
            }
        }
        let mut lay = layout(&s, has_this, sret, self.db);
        if s.variadic && !lay.params.iter().any(|p| *p == ArgLoc::Stack) {
            // the variadic part: argument registers set up for this call after the named ones
            // (marked `...`; dropped from the callee's signature once the arguments are read);
            // float ones only when the caller sets cr1eq (`creqv 6,6,6`)
            let g = lay.params.iter().fold(2u8, |m, l| match l {
                ArgLoc::Gpr(r) => m.max(*r),
                ArgLoc::GprPair(r) => m.max(*r + 1),
                _ => m,
            });
            let g = g.max(lay.this.unwrap_or(0)).max(lay.sret.unwrap_or(0));
            let f = lay.params.iter().fold(0u8, |m, l| if let ArgLoc::Fpr(r) = l { m.max(*r) } else { m });
            let extra = |p: &mut Vec<mwdec_core::Param>| p.push(mwdec_core::Param { name: Some("...".into()), ty: Type::Unknown { size: 0 } });
            let set = (g + 1..=10u8).rev().find(|q| self.reg_set_for_call(k, gpr(*q)));
            // (or incoming parameters passed on unchanged: the allocator kept them free)
            let through = self.passed_through(k).filter(|&r| r > g && self.entry_vals.contains_key(&gpr(r)));
            if let Some(top) = set.max(through) {
                for r in g + 1..=top {
                    lay.params.push(ArgLoc::Gpr(r));
                    extra(&mut s.params);
                }
            }
            let b = self.cfg.block_of[k];
            let floats = (self.cfg.blocks[b].start..k).rev().take_while(|&j| !self.insns[j].is_call()).any(|j| {
                let i = &self.insns[j];
                i.op() == Opcode::Creqv && i.ins.field_crbd() == 6
            });
            if floats {
                if let Some(top) = (f + 1..=8u8).rev().find(|q| self.reg_set_for_call(k, fpr(*q))) {
                    for r in f + 1..=top {
                        lay.params.push(ArgLoc::Fpr(r));
                        extra(&mut s.params);
                    }
                }
            }
        }
        Some((s, lay, has_this))
    }

    /// Highest argument GPR (r3..r10) that still holds its incoming value at the first call `k`
    /// and that the register allocator kept free for it: untouched from the function entry to the
    /// call, while some short-lived value computed before the call took a higher volatile register
    /// (temporaries get the lowest free one). Such a register is an argument passed through to
    /// the call unchanged (`salHooks.malloc(len)`, `(this->*pmf)(mgr)`).
    fn passed_through(&self, k: usize) -> Option<u8> {
        let mut touched: HashSet<Reg> = HashSet::new();
        let mut higher: Vec<u8> = vec![];
        for j in 0..k {
            if self.frame.skip.contains(&j) {
                continue;
            }
            let i = &self.insns[j];
            if i.is_call() || i.is_bctrl() {
                return None;
            }
            let (d, u) = defs_uses(i);
            for r in d.iter().chain(u.iter()) {
                touched.insert(*r);
            }
            for &r in &d {
                if (4..=11).contains(&r) && !self.reg_set_for_call(k, r) {
                    higher.push(r as u8);
                }
            }
        }
        let top = *higher.iter().max()?;
        (3..top.min(11)).rev().find(|&r| r <= 10 && !touched.contains(&gpr(r)) && higher.iter().any(|&d| d > r))
    }

    /// Was `reg` written in the same block before instruction k (after the previous call), or is it
    /// a live-in parameter register?
    /// Walking back from the call `k` in its block, is `reg` read before any write (its value is
    /// consumed by an instruction of the block, not set up for the call)?
    fn reg_read_last_in_block(&self, k: usize, reg: Reg) -> bool {
        let b = self.cfg.block_of[k];
        let start = self.cfg.blocks[b].start;
        let mut j = k;
        while j > start {
            j -= 1;
            let i = &self.insns[j];
            if i.is_call() || i.is_bctrl() || self.frame.skip.contains(&j) {
                return false;
            }
            let (d, u) = defs_uses(i);
            if d.contains(&reg) {
                return false;
            }
            if u.contains(&reg) {
                return true;
            }
        }
        false
    }

    fn reg_set_before(&self, k: usize, reg: Reg) -> bool {
        let b = self.cfg.block_of[k];
        let start = self.cfg.blocks[b].start;
        let mut j = k;
        while j > start {
            j -= 1;
            let i = &self.insns[j];
            if i.is_call() || i.is_bctrl() {
                return false;
            }
            if !self.frame.skip.contains(&j) && defs_uses(i).0.contains(&reg) {
                return true;
            }
        }
        // reached block start: assume set if it reaches from a predecessor
        b != 0 || self.entry_vals.contains_key(&reg)
    }

    /// Argument registers for an indirect call (no signature): r3.. / f1.. written since the
    /// previous call in this block.
    fn indirect_args(&self, k: usize) -> (Vec<Reg>, Vec<Reg>) {
        let mut g = vec![];
        for r in 4..=10u8 {
            if self.reg_written_before_call(k, gpr(r)) {
                g.push(gpr(r));
            } else {
                break;
            }
        }
        let mut fl = vec![];
        for r in 1..=8u8 {
            if self.reg_written_before_call(k, fpr(r)) {
                fl.push(fpr(r));
            } else {
                break;
            }
        }
        (g, fl)
    }

    /// Was `reg` last written in this block (since the previous call) and not read again before
    /// the call at `k` (i.e. set up as an argument)?
    /// Words stored right before call `k` (same block) to the outgoing parameter area
    /// `8(r1)`, `12(r1)`, ... and never read back: stack-passed arguments.
    fn outgoing_stack_words(&self, k: usize) -> usize {
        if self.frame.info.size == 0 {
            return 0;
        }
        let b = self.cfg.block_of[k];
        let mut offs = vec![];
        for j in self.cfg.blocks[b].start..k {
            let i = &self.insns[j];
            if self.frame.skip.contains(&j) || i.reloc.is_some() || i.ra() != 1 {
                continue;
            }
            if i.is_call() || i.is_bctrl() {
                offs.clear();
                continue;
            }
            if matches!(i.op(), ppc750cl::Opcode::Stw) && i.disp() >= 8 && !self.frame.save_slots.contains_key(&i.disp()) {
                offs.push(i.disp());
            }
        }
        offs.sort();
        offs.dedup();
        if offs.is_empty() || offs.iter().enumerate().any(|(n, o)| *o != 8 + 4 * n as i32) {
            return 0;
        }
        // never read back (not a local)
        let read = self.insns.iter().enumerate().any(|(j, i)| {
            !self.frame.skip.contains(&j) && i.ra() == 1 && i.reloc.is_none() && matches!(i.op(), ppc750cl::Opcode::Lwz | ppc750cl::Opcode::Lhz | ppc750cl::Opcode::Lbz | ppc750cl::Opcode::Lha) && offs.contains(&i.disp())
                || (matches!(i.op(), ppc750cl::Opcode::Addi) && i.ra() == 1 && offs.contains(&(i.simm() as i32)))
        });
        if read {
            return 0;
        }
        offs.len()
    }

    /// `reg` still holds this function's incoming argument at instruction `k` (nothing before
    /// `k` writes it, no call in between).
    fn entry_val_intact(&self, k: usize, reg: Reg) -> bool {
        self.entry_vals.contains_key(&reg)
            && !self.insns[..k].iter().enumerate().any(|(j, i)| {
                !self.frame.skip.contains(&j) && (i.is_call() || i.is_bctrl() || defs_uses(i).0.contains(&reg))
            })
    }

    fn reg_set_for_call(&self, k: usize, reg: Reg) -> bool {
        let b = self.cfg.block_of[k];
        let start = self.cfg.blocks[b].start;
        let mut j = k;
        while j > start {
            j -= 1;
            let i = &self.insns[j];
            if i.is_call() || i.is_bctrl() {
                return false;
            }
            if self.frame.skip.contains(&j) {
                continue;
            }
            let (d, u) = defs_uses(i);
            if d.contains(&reg) {
                return true;
            }
            if u.contains(&reg) {
                return false;
            }
        }
        // set up in a dominating block (an incoming value copied before a loop, passed after
        // it): the register's only touch in the function, on the dominator path, no call between
        let mut bb = b;
        let mut fuel = crate::fuel::Fuel::new("translate.dominator_walk", self.cfg.idom.len() + 1);
        loop {
            if !fuel.burn() {
                return false;
            }
            let up = self.cfg.idom[bb];
            if up == usize::MAX || up == bb {
                return false;
            }
            bb = up;
            let blk = &self.cfg.blocks[bb];
            for j in (blk.start..blk.end).rev() {
                let i = &self.insns[j];
                if i.is_call() || i.is_bctrl() {
                    return false;
                }
                if self.frame.skip.contains(&j) {
                    continue;
                }
                let (d, u) = defs_uses(i);
                if u.contains(&reg) {
                    return false;
                }
                if d.contains(&reg) {
                    let touches = (0..self.insns.len()).filter(|&q| !self.frame.skip.contains(&q) && !self.insns[q].is_call() && !self.insns[q].is_bctrl() && {
                        let (d2, u2) = defs_uses(&self.insns[q]);
                        d2.contains(&reg) || u2.contains(&reg)
                    }).count();
                    return touches == 1;
                }
            }
            if bb == 0 {
                return false;
            }
        }
    }

    /// Like `reg_written_in_block`, continuing into single predecessors (an argument loaded before
    /// a guard: `lwz r5,len ; cmplwi r5,0 ; beq ; ... ; bctrl`), up to the previous call.
    fn reg_written_before_call(&self, k: usize, reg: Reg) -> bool {
        let mut b = self.cfg.block_of[k];
        let mut j = k;
        for _ in 0..4 {
            let start = self.cfg.blocks[b].start;
            while j > start {
                j -= 1;
                let i = &self.insns[j];
                if i.is_call() || i.is_bctrl() {
                    return false;
                }
                if !self.frame.skip.contains(&j) && defs_uses(i).0.contains(&reg) {
                    return true;
                }
            }
            let preds = &self.cfg.blocks[b].preds;
            if preds.len() != 1 || preds[0] >= b {
                return false;
            }
            b = preds[0];
            j = self.cfg.blocks[b].end;
        }
        false
    }

    fn reg_written_in_block(&self, k: usize, reg: Reg) -> bool {
        let b = self.cfg.block_of[k];
        let start = self.cfg.blocks[b].start;
        let mut j = k;
        while j > start {
            j -= 1;
            let i = &self.insns[j];
            if i.is_call() || i.is_bctrl() {
                return false;
            }
            if !self.frame.skip.contains(&j) && defs_uses(i).0.contains(&reg) {
                return true;
            }
        }
        false
    }

    /// Precise uses of an instruction (narrowing calls/returns).
    fn uses_of(&self, k: usize) -> Vec<Reg> {
        let i = &self.insns[k];
        if i.is_call() || (i.is_jump() && i.reloc.is_some()) {
            if let Some((_, lay, _)) = self.call_layouts.get(&k) {
                let mut u = vec![];
                if let Some(r) = lay.sret {
                    u.push(gpr(r));
                }
                if let Some(r) = lay.this {
                    u.push(gpr(r));
                }
                for p in &lay.params {
                    match p {
                        ArgLoc::Gpr(r) => u.push(gpr(*r)),
                        ArgLoc::Fpr(r) => u.push(fpr(*r)),
                        ArgLoc::GprPair(r) => {
                            u.push(gpr(*r));
                            u.push(gpr(*r + 1));
                        }
                        ArgLoc::Stack => {}
                    }
                }
                return u;
            }
            return vec![];
        }
        if i.is_bctrl() {
            let (g, f) = self.indirect_args(k);
            let mut u = vec![if i.is_blrl() { LR } else { CTR }];
            if self.reg_written_in_block(k, gpr(3)) {
                u.push(gpr(3));
            }
            if let Some(m) = self.passed_through(k) {
                u.extend((3..=m).map(gpr).filter(|r| !g.contains(r)));
            }
            u.extend(g);
            u.extend(f);
            return u;
        }
        if i.is_blr() || i.is_cond_blr() {
            let mut u = vec![];
            if i.is_cond_blr() {
                u.push(crf(i.ins.field_bi() >> 2));
            }
            if let Some(r) = self.ret_reg {
                u.push(r);
                if r == gpr(3) && crate::wide::is_wide(&types::resolve(self.db, &self.ret_ty)) {
                    u.push(gpr(4));
                }
            }
            return u;
        }
        if i.is_bctr() {
            return vec![CTR];
        }
        defs_uses(i).1
    }

    // ------------------------------------------------------------ dataflow

    fn defs_of(&self, k: usize) -> Vec<Reg> {
        let i = &self.insns[k];
        let (mut d, _) = defs_uses(i);
        // record forms of float ops set cr1; ignore
        d.retain(|r| *r != gpr(1));
        d
    }

    fn reaching_defs(&mut self) {
        let nb = self.cfg.blocks.len();
        // gen per block: last def per reg
        let mut gen: Vec<HashMap<Reg, u32>> = vec![HashMap::new(); nb];
        for b in 0..nb {
            let blk = &self.cfg.blocks[b];
            for k in blk.start..blk.end {
                if self.frame.skip.contains(&k) {
                    continue;
                }
                for r in self.defs_of(k) {
                    gen[b].insert(r, k as u32);
                }
            }
        }
        let mut inn: Vec<Vec<Vec<u32>>> = vec![vec![vec![]; NREGS]; nb];
        let mut out: Vec<Vec<Vec<u32>>> = vec![vec![vec![]; NREGS]; nb];
        for r in 0..NREGS {
            inn[0][r] = vec![ENTRY];
        }
        let mut changed = true;
        let rpo = self.cfg.rpo.clone();
        let mut fuel = crate::fuel::Fuel::new("translate.reaching_defs", crate::fuel::CAP_FIXPOINT);
        while changed && fuel.burn() {
            changed = false;
            for &b in &rpo {
                if b != 0 || !self.cfg.blocks[0].preds.is_empty() {
                    let mut new: Vec<Vec<u32>> = vec![vec![]; NREGS];
                    if b == 0 {
                        for r in 0..NREGS {
                            new[r].push(ENTRY);
                        }
                    }
                    for &p in &self.cfg.blocks[b].preds {
                        for r in 0..NREGS {
                            for &d in &out[p][r] {
                                if !new[r].contains(&d) {
                                    new[r].push(d);
                                }
                            }
                        }
                    }
                    for v in new.iter_mut() {
                        v.sort_unstable();
                    }
                    if new != inn[b] {
                        inn[b] = new;
                        changed = true;
                    }
                }
                let mut o = inn[b].clone();
                for (&r, &k) in &gen[b] {
                    o[r as usize] = vec![k];
                }
                if o != out[b] {
                    out[b] = o;
                    changed = true;
                }
            }
        }
        self.rd_in = inn;
    }

    /// Walk all instructions with current reaching defs, calling `f(k, reg, defs)` per use.
    fn for_each_use(&self, mut f: impl FnMut(usize, Reg, &[u32])) {
        for b in 0..self.cfg.blocks.len() {
            if self.cfg.idom[b] == usize::MAX {
                continue;
            }
            let mut cur = self.rd_in[b].clone();
            let blk = &self.cfg.blocks[b];
            for k in blk.start..blk.end {
                if self.frame.skip.contains(&k) {
                    continue;
                }
                for r in self.uses_of(k) {
                    f(k, r, &cur[r as usize]);
                }
                for r in self.defs_of(k) {
                    cur[r as usize] = vec![k as u32];
                }
            }
        }
    }

    fn infer_return(&mut self) {
        if self.sret_var.is_some() && sig::ret_unknown(&self.sig) {
            // guessed struct return: the object is the result (typed by idioms)
            self.ret_reg = None;
            self.ret_ty = Type::Void;
            return;
        }
        if !sig::ret_unknown(&self.sig) {
            self.ret_ty = self.sig.ret.clone();
            if types::is_aggregate(self.db, &self.ret_ty) {
                self.ret_reg = None;
            } else if is_float(&types::resolve(self.db, &self.ret_ty)) {
                self.ret_reg = Some(fpr(1));
            } else if matches!(strip_cv(&self.ret_ty), Type::Void) {
                self.ret_reg = None;
            } else {
                self.ret_reg = Some(gpr(3));
            }
            return;
        }
        // Heuristic: the value in r3/f1 at every return is produced by a non-call instruction
        // whose result isn't otherwise used.
        let mut other_uses: HashMap<Site, u32> = HashMap::new();
        let mut ret_defs: HashMap<Reg, Vec<u32>> = HashMap::new();
        // defs reaching each use (to see through register copies)
        let mut use_defs: HashMap<(u32, Reg), Vec<u32>> = HashMap::new();
        self.ret_reg = None;
        self.for_each_use(|k, r, defs| {
            for &d in defs {
                *other_uses.entry((d, r)).or_default() += 1;
            }
            use_defs.insert((k as u32, r), defs.to_vec());
        });
        for b in 0..self.cfg.blocks.len() {
            if self.cfg.idom[b] == usize::MAX || !matches!(self.cfg.blocks[b].term, Term::Return) {
                continue;
            }
            let mut cur = self.rd_in[b].clone();
            let blk = &self.cfg.blocks[b];
            for k in blk.start..blk.end {
                if self.frame.skip.contains(&k) {
                    continue;
                }
                for r in self.defs_of(k) {
                    cur[r as usize] = vec![k as u32];
                }
            }
            for r in [gpr(3), fpr(1)] {
                ret_defs.entry(r).or_default().extend(cur[r as usize].iter().copied());
            }
        }
        let name = sig::split_scope(&self.sig.qualified_name).1.to_string();
        let getter_like = ["Get", "Is", "Has", "Can", "Should", "Does", "Are", "Was", "Find", "Check"]
            .iter()
            .any(|p| name.starts_with(p));
        let judge = |r: Reg, this: &Lifter| -> (bool, bool) {
            // (all defs are value-producing, some def is a non-call)
            let defs = ret_defs.get(&r).cloned().unwrap_or_default();
            if defs.is_empty() {
                return (false, false);
            }
            let mut any_noncall = false;
            for &d in &defs {
                if d == ENTRY {
                    if !this.entry_vals.contains_key(&r) {
                        return (false, false);
                    }
                    continue;
                }
                let i = &this.insns[d as usize];
                if i.is_call() || i.is_bctrl() {
                    continue;
                }
                any_noncall = true;
                if other_uses.get(&(d, r)).copied().unwrap_or(0) > 0 {
                    return (false, false);
                }
            }
            (true, any_noncall)
        };
        let (f_ok, f_nc) = judge(fpr(1), self);
        let (g_ok, g_nc) = judge(gpr(3), self);
        // a recursive call whose r3 result is read: the function returns a value there
        let self_value = self.insns.iter().enumerate().any(|(k, i)| {
            i.is_call() && i.reloc.as_ref().is_some_and(|r| r.target == self.f.name) && other_uses.get(&(k as u32, gpr(3))).copied().unwrap_or(0) > 0
        });
        let g_ok = g_ok || self_value;
        let g_nc = g_nc || self_value;
        // a returned call result whose callee's return type is known decides between f1 and r3
        let call_ret = |r: Reg, want_float: bool| {
            ret_defs.get(&r).is_some_and(|defs| {
                defs.iter().any(|&d| {
                    self.call_layouts.get(&(d as usize)).is_some_and(|(s, _, _)| {
                        !sig::ret_unknown(s) && !matches!(strip_cv(&s.ret), Type::Void) && is_float(&types::resolve(self.db, &s.ret)) == want_float && !types::is_aggregate(self.db, &s.ret)
                    })
                })
            })
        };
        let f_strong = f_ok && (f_nc || (getter_like && call_ret(fpr(1), true)));
        let g_strong = g_ok && (g_nc || (getter_like && call_ret(gpr(3), false)));
        let pick_f = if f_strong && !(g_strong && !f_nc) {
            true
        } else if g_strong {
            false
        } else {
            f_ok && getter_like
        };
        if pick_f {
            self.ret_reg = Some(fpr(1));
            self.ret_ty = t_f32();
        } else if g_ok && (g_nc || getter_like) {
            self.ret_reg = Some(gpr(3));
            // bool if every def is li 0/1
            let mut defs = ret_defs.get(&gpr(3)).cloned().unwrap_or_default();
            // through `mr rD, rS` copies to the values they copy
            let mut flat = vec![];
            let mut seen = HashSet::new();
            while let Some(d) = defs.pop() {
                if !seen.insert(d) || seen.len() > 64 {
                    continue;
                }
                if d != ENTRY {
                    let i = &self.insns[d as usize];
                    let copy_of = match i.op() {
                        Opcode::Or if i.rs() == i.rb() => Some(i.rs()),
                        Opcode::Addi if i.ra() != 0 && i.simm() == 0 => Some(i.ra()),
                        _ => None,
                    };
                    if let Some(rs) = copy_of {
                        if let Some(src) = use_defs.get(&(d, gpr(rs))) {
                            defs.extend(src.iter().copied());
                            continue;
                        }
                    }
                }
                flat.push(d);
            }
            let all_bool = !flat.is_empty() && flat.iter().all(|&d| {
                d != ENTRY && {
                    let i = &self.insns[d as usize];
                    (i.op() == Opcode::Addi && i.ra() == 0 && (i.simm() == 0 || i.simm() == 1))
                        || (i.op() == Opcode::Rlwinm && i.ins.field_mb() == 24 && i.ins.field_me() == 31 && i.ins.field_sh() == 0)
                        || (i.op() == Opcode::Rlwinm && i.ins.field_mb() == 31 && i.ins.field_me() == 31)
                }
            });
            self.ret_ty = if all_bool { Type::Bool } else { t_unk(4) };
        } else {
            self.ret_ty = Type::Void;
        }
        self.sig.ret = self.ret_ty.clone();
    }

    fn compute_webs(&mut self) {
        // union-find over sites
        let mut parent: HashMap<Site, Site> = HashMap::new();
        fn find(p: &mut HashMap<Site, Site>, x: Site) -> Site {
            let mut r = x;
            while let Some(&q) = p.get(&r) {
                if q == r {
                    break;
                }
                r = q;
            }
            // path compression
            let mut c = x;
            while let Some(&q) = p.get(&c) {
                if q == c {
                    break;
                }
                p.insert(c, r);
                c = q;
            }
            r
        }
        let mut uses: HashMap<Site, u32> = HashMap::new();
        let mut ublocks: HashMap<Site, HashSet<usize>> = HashMap::new();
        let mut multi: Vec<Vec<Site>> = vec![];
        let block_of = self.cfg.block_of.clone();
        let split = self.split_returns.clone();
        let is_ret: HashSet<usize> = (0..self.insns.len()).filter(|&k| self.insns[k].is_blr()).collect();
        self.for_each_use(|k, r, defs| {
            for &d in defs {
                *uses.entry((d, r)).or_default() += 1;
                ublocks.entry((d, r)).or_default().insert(block_of[k]);
            }
            let split_ret = split.contains(&block_of[k]) && is_ret.contains(&k);
            if defs.len() > 1 && (r < 64 || r == CTR) && !split_ret {
                multi.push(defs.iter().map(|&d| (d, r)).collect());
            }
        });
        for group in &multi {
            for s in group {
                parent.entry(*s).or_insert(*s);
            }
            let a = find(&mut parent, group[0]);
            for s in &group[1..] {
                let b = find(&mut parent, *s);
                if a != b {
                    parent.insert(b, a);
                }
            }
        }
        // assign vars per web root
        let sites: Vec<Site> = parent.keys().copied().collect();
        let mut root_var: HashMap<Site, VarId> = HashMap::new();
        // params first: webs containing an entry def of a param register reuse the param var
        for s in &sites {
            if s.0 == ENTRY {
                if let Some(Expr::Var(v)) = self.entry_vals.get(&s.1) {
                    let root = find(&mut parent, *s);
                    root_var.insert(root, *v);
                }
            }
        }
        // a reassigned parameter (`result = f(); ... use(result)`): a web one of whose defs
        // copies a param register that holds only its entry value and has no other use
        // (`mr. r29, r4`) is the parameter itself, not a local copied from it
        {
            let mut copy_src: HashMap<u32, (Reg, bool)> = HashMap::new();
            let mut entry_users: HashMap<Reg, HashSet<usize>> = HashMap::new();
            self.for_each_use(|k, r, defs| {
                let e = copy_src.entry(k as u32).or_insert((r, true));
                if e.0 != r || defs != [ENTRY] {
                    e.1 = false;
                }
                if defs.contains(&ENTRY) {
                    entry_users.entry(r).or_default().insert(k);
                }
            });
            // params are coloured last (regalloc.md): they take the lowest callee-saved
            // registers, so a copy living above some local's register is a named local
            let copy_of_entry = |k: usize| -> bool {
                let i = &self.insns[k];
                let src = match i.op() {
                    Opcode::Or if i.rs() == i.rb() => gpr(i.rs()),
                    Opcode::Addi if i.simm() == 0 && i.ra() != 0 => gpr(i.ra()),
                    Opcode::Fmr => fpr(i.rb()),
                    _ => return false,
                };
                copy_src.get(&(k as u32)) == Some(&(src, true)) && self.entry_vals.contains_key(&src)
            };
            let mut low_local = [99u8; 2];
            for k in 0..self.insns.len() {
                if self.frame.skip.contains(&k) || copy_of_entry(k) {
                    continue;
                }
                for r in self.defs_of(k) {
                    let (cls, n) = if r < 32 { (0, r) } else if r < 64 { (1, r - 32) } else { continue };
                    if n >= 14 {
                        low_local[cls] = low_local[cls].min(n);
                    }
                }
            }
            let used_params: HashSet<VarId> = root_var.values().copied().collect();
            let mut taken: HashSet<VarId> = used_params;
            let mut sorted = sites.clone();
            sorted.sort();
            for s in &sorted {
                let root = find(&mut parent, *s);
                if root_var.contains_key(&root) || s.0 == ENTRY || s.1 >= 64 {
                    continue;
                }
                let i = &self.insns[s.0 as usize];
                let src = match i.op() {
                    Opcode::Or if i.rs() == i.rb() => gpr(i.rs()),
                    Opcode::Addi if i.simm() == 0 && i.ra() != 0 => gpr(i.ra()),
                    Opcode::Fmr => fpr(i.rb()),
                    _ => continue,
                };
                if src == s.1 || copy_src.get(&s.0) != Some(&(src, true)) {
                    continue;
                }
                let Some(Expr::Var(v)) = self.entry_vals.get(&src).cloned() else { continue };
                if !matches!(self.vars[v].kind, VarKind::Param { .. }) || taken.contains(&v) {
                    continue;
                }
                let (cls, n) = if s.1 < 32 { (0, s.1) } else { (1, s.1 - 32) };
                if n >= 14 && n > low_local[cls] {
                    continue;
                }
                if entry_users.get(&src).map_or(0, |u| u.len()) != 1 {
                    continue;
                }
                taken.insert(v);
                root_var.insert(root, v);
            }
        }
        let mut sorted = sites.clone();
        sorted.sort();
        for s in sorted {
            let root = find(&mut parent, s);
            let v = match root_var.get(&root) {
                Some(&v) => v,
                None => {
                    let ty = if (32..64).contains(&s.1) { t_f32() } else { t_unk(4) };
                    let v = self.new_var(format!("var_{}", reg_name(s.1)), ty, VarKind::Local, false);
                    root_var.insert(root, v);
                    v
                }
            };
            self.web_var.insert(s, v);
        }
        self.use_count = uses;
        self.use_blocks = ublocks;
    }

    // ------------------------------------------------------------ stack model

    /// Offsets (relative to the caller's r1 + 8) of stack-passed parameters of a layout.
    fn stack_arg_offsets(&self, sig: &FuncSig, lay: &Layout) -> Vec<(usize, i32, u32)> {
        let mut out = vec![];
        let mut o = 0i32;
        for (n, p) in lay.params.iter().enumerate() {
            if *p != ArgLoc::Stack {
                continue;
            }
            let t = types::resolve(self.db, &sig.params[n].ty).into_owned();
            let size = match strip_cv(&t) {
                // (a float argument takes one word: `stfs` to consecutive words)
                Type::Float { size: 8 } | Type::Int { size: 8, .. } => 8,
                _ => 4,
            };
            if size == 8 && o % 8 != 0 {
                o += 4;
            }
            out.push((n, o, size));
            o += size as i32;
        }
        out
    }

    fn setup_stack_args(&mut self) {
        let base = self.frame.info.size as i32 + 8;
        let has_this = self.this_var.is_some();
        let sret = self.sret_var.is_some();
        let lay = layout(&self.sig, has_this, sret, self.db);
        let sp = self.stack_params.clone();
        for (n, o, size) in self.stack_arg_offsets(&self.sig.clone(), &lay) {
            if let Some((_, v)) = sp.iter().find(|(i, _)| *i == n) {
                for k in 0..size as i32 {
                    self.in_stack.insert(base + o + k, v.clone());
                }
            }
        }
        let calls: Vec<(FuncSig, Layout)> = self.call_layouts.values().map(|(s, l, _)| (s.clone(), l.clone())).collect();
        for (s, l) in calls {
            for (_, o, size) in self.stack_arg_offsets(&s, &l) {
                for k in 0..size as i32 {
                    self.out_stack.insert(8 + o + k);
                }
            }
        }
        // indirect/virtual calls (signature found later): word stores into the parameter area
        // right before the call to slots nothing reads or takes the address of (an address taken
        // lower in the area may be an object spanning the slot, e.g. a by-reference vector temp)
        let read_or_addressed = |off: i32, me: &Self| {
            me.insns.iter().any(|q| {
                let addr = matches!(q.op(), Opcode::Addi) && q.ra() == 1 && (8..=off).contains(&(q.simm() as i32));
                let load = q.ra() == 1 && q.reloc.is_none() && matches!(q.op(), Opcode::Lwz | Opcode::Lfs | Opcode::Lfd | Opcode::Lhz | Opcode::Lha | Opcode::Lbz) && q.simm() as i32 == off;
                addr || load
            })
        };
        for k in 0..self.insns.len() {
            if !self.insns[k].is_bctrl() {
                continue;
            }
            let b = self.cfg.block_of[k];
            for j in (self.cfg.blocks[b].start..k).rev() {
                let q = &self.insns[j];
                if q.is_call() || q.is_bctrl() {
                    break;
                }
                if self.frame.skip.contains(&j) {
                    continue;
                }
                if matches!(q.op(), Opcode::Stw | Opcode::Stfs | Opcode::Stfd) && q.ra() == 1 && q.reloc.is_none() {
                    let off = q.simm() as i32;
                    if (8..8 + 0x40).contains(&off) && !read_or_addressed(off, self) {
                        let size = if q.op() == Opcode::Stfd { 8 } else { 4 };
                        for x in 0..size {
                            self.out_stack.insert(off + x);
                        }
                    }
                }
            }
        }
    }

    fn scan_stack(&mut self) {
        #[derive(Default, Clone)]
        struct Acc {
            sizes: Vec<(u32, bool, bool)>, // (size, float, store)
        }
        let mut acc: BTreeMap<i32, Acc> = BTreeMap::new();
        let mut addr: Vec<i32> = vec![];
        let mut obj_size: HashMap<i32, u32> = HashMap::new();
        let mut psq: HashSet<i32> = HashSet::new();
        // addresses passed in r3 (a struct return) to calls whose signature is not known yet
        let mut indirect: HashSet<i32> = HashSet::new();
        // addresses receiving a struct return (always an object of their own)
        let mut sret_addr: HashSet<i32> = HashSet::new();
        // addresses that are the receiver of a method call
        let mut receiver_addr: HashSet<i32> = HashSet::new();
        for (k, i) in self.insns.iter().enumerate() {
            if self.frame.skip.contains(&k) {
                continue;
            }
            use Opcode::*;
            let (size, fl, st) = match i.op() {
                Lwz | Lwzu => (4, false, false),
                Lhz | Lha => (2, false, false),
                Lbz => (1, false, false),
                Stw => (4, false, true),
                Sth => (2, false, true),
                Stb => (1, false, true),
                Lfs => (4, true, false),
                Lfd => (8, true, false),
                Stfs => (4, true, true),
                Stfd => (8, true, true),
                PsqL | PsqSt => {
                    if i.ra() == 1 {
                        psq.insert(i.ins.field_ps_offset() as i32);
                    }
                    continue;
                }
                Addi | Addic | Addic_ if i.ra() == 1 && i.reloc.is_none() => {
                    addr.push(i.simm() as i32);
                    if let Some(sz) = self.addr_object_size(k, i.rd()) {
                        let e = obj_size.entry(i.simm() as i32).or_insert(0);
                        *e = (*e).max(sz);
                        if self.addr_is_sret(k, i.rd()) {
                            sret_addr.insert(i.simm() as i32);
                        }
                    } else if i.rd() == 3 && self.addr_to_indirect_call(k, i.rd()) {
                        indirect.insert(i.simm() as i32);
                    }
                    if self.addr_is_receiver(k, i.rd()) {
                        receiver_addr.insert(i.simm() as i32);
                    }
                    continue;
                }
                _ => continue,
            };
            if i.ra() != 1 || i.reloc.is_some() {
                continue;
            }
            if self.in_stack.contains_key(&i.disp()) || self.out_stack.contains(&i.disp()) {
                continue;
            }
            acc.entry(i.disp()).or_default().sizes.push((size, fl, st));
        }
        // conversion scratch
        let mut conv = HashSet::new();
        for (&o, a) in &acc {
            let lfd = a.sizes.iter().any(|s| s.0 == 8 && s.1 && !s.2);
            let stfd = a.sizes.iter().any(|s| s.0 == 8 && s.1 && s.2);
            let lo = acc.get(&(o + 4));
            if lfd && a.sizes.iter().any(|s| s.0 == 4 && !s.1 && s.2) && lo.map_or(false, |l| l.sizes.iter().any(|s| s.2)) {
                conv.insert(o);
                conv.insert(o + 4);
            }
            if stfd && lo.map_or(false, |l| l.sizes.iter().any(|s| !s.2 && !s.1)) {
                conv.insert(o);
                conv.insert(o + 4);
            }
        }
        // a wide access covering narrower ones (bytes stored one by one, the word read back:
        // a small object built member-wise and copied whole) is one object, not separate slots
        for (&o, a) in &acc {
            if conv.contains(&o) || self.frame.save_slots.contains_key(&o) {
                continue;
            }
            let wide = a.sizes.iter().map(|s| s.0).max().unwrap_or(0) as i32;
            if wide > 1 && acc.range(o + 1..o + wide).any(|(q, _)| !conv.contains(q)) {
                addr.push(o);
            }
        }
        addr.sort();
        addr.dedup();
        // the linkage area holds no object: an address there points just below the lowest object
        if addr.iter().any(|&a| a >= 8) {
            addr.retain(|&a| a >= 8);
        }
        // addresses inside a class object whose address is taken too (a member passed by
        // reference) are parts of that object, not objects of their own: unless they are
        // themselves objects reaching past its end, struct returns, or method receivers (an
        // inlined member destructor of a temporary stays apart so the temporary can fold)
        let mut starts: Vec<i32> = vec![];
        let mut covered_to = i32::MIN;
        for &o in &addr {
            if o < covered_to && !sret_addr.contains(&o) && !receiver_addr.contains(&o) && obj_size.get(&o).map_or(true, |&sz| o + sz as i32 <= covered_to) {
                continue;
            }
            starts.push(o);
            covered_to = covered_to.max(obj_size.get(&o).map_or(i32::MIN, |&sz| o + sz as i32));
        }
        let addr = starts;
        let top = self.frame.saves_lo.min(self.frame.info.size as i32);
        let mut regions = vec![];
        for (n, &o) in addr.iter().enumerate() {
            let mut end = addr.get(n + 1).copied().unwrap_or(top).max(o + 1);
            // The topmost object: the gap up to the register save area includes the frame's
            // alignment padding, so size it from what the code shows (the class object it is
            // passed as, the accesses into it), within the sizes that give the frame its size.
            if n + 1 == addr.len() {
                if let Some((floor, max_end)) = self.locals_bounds() {
                    let scratch = |c: i32| conv.contains(&c) || psq.contains(&c);
                    let ext = acc
                        .iter()
                        .filter(|(&a, _)| a >= o && a < end && !(a > o && scratch(a)))
                        .map(|(&a, x)| a - o + x.sizes.iter().map(|s| s.0 as i32).max().unwrap_or(1))
                        .chain(obj_size.get(&o).map(|&s| s as i32))
                        .max()
                        .unwrap_or(0);
                    let ext = (ext + 3) & !3;
                    let first_scratch = conv.iter().chain(psq.iter()).copied().filter(|&c| c >= o + ext.max(1) && c < end).min();
                    if let Some(fs) = first_scratch {
                        // codegen temporaries (8-aligned conversion scratch) follow the locals
                        end = (o + ext).max(fs - 4).min(fs);
                    } else if ext > 0 && !indirect.contains(&o) {
                        end = (o + ext).max(floor).min(max_end.max(o + ext)).min(end);
                    } else {
                        end = end.min(max_end).max(o + 1);
                    }
                }
            }
            // the class object the address is passed as (constructor/method receiver, reference
            // or by-value argument) bounds the region; the rest are separate locals
            if let Some(&sz) = obj_size.get(&o) {
                if sz > 0 && o + (sz as i32) < end {
                    end = o + sz as i32;
                }
            }
            let size = (end - o) as u32;
            let v = self.new_var(format!("stack_{:x}", o), t_unk(size), VarKind::Stack { offset: o, size }, false);
            regions.push((o, size, v));
        }
        // quantized loads/stores of the frame are conversion scratch, unless they read an object
        // whose address is taken (a byte color filled by a callee, read as floats)
        for &o in &psq {
            if !self.frame.save_slots.contains_key(&o) && !regions.iter().any(|&(s, z, _)| o >= s && o < s + z as i32) {
                conv.insert(o);
            }
        }
        let mut slots = BTreeMap::new();
        for (&o, a) in &acc {
            if conv.contains(&o) || regions.iter().any(|&(s, z, _)| o >= s && o < s + z as i32) {
                continue;
            }
            if self.frame.save_slots.contains_key(&o) {
                continue;
            }
            let (size, fl, _) = a.sizes[0];
            let ty = match (size, fl) {
                (8, true) => t_f64(),
                (4, true) => t_f32(),
                (4, false) => t_unk(4),
                (2, false) => t_int(2, false),
                _ => t_int(1, false),
            };
            let v = self.new_var(format!("local_{:x}", o), ty, VarKind::Stack { offset: o, size }, false);
            slots.insert(o, v);
        }
        self.stack = StackModel { slots, regions, conv };
    }

    /// Bounds `(lowest, highest)` for the end offset of the frame's local area that give the
    /// frame its size.
    ///
    /// MWCC lays out the frame as linkage (8) + outgoing arguments + locals and temporaries
    /// (rounded to 8), then pads so that the GPR save area ends on the frame alignment (16 on
    /// GC/2.x, 8 on the older compilers), then the FPR saves (8 bytes each, 16 with a `psq_st`
    /// upper half) rounded to the alignment. `None` if the prologue doesn't fit that model.
    fn locals_bounds(&self) -> Option<(i32, i32)> {
        let fi = &self.frame.info;
        let size = fi.size as i32;
        if size == 0 || fi.uses_savegpr {
            return None;
        }
        let align = if size % 16 == 0 { 16 } else { 8 };
        let fpr_bytes = 8 * (fi.saved_fprs.len() + fi.saved_ps.len()) as i32;
        let fpr_area = (fpr_bytes + align - 1) & !(align - 1);
        let gpr_lo = size - fpr_area - 4 * fi.saved_gprs.len() as i32;
        if let Some(&lo) = self.frame.save_slots.iter().filter(|(_, &r)| r < 32).map(|(o, _)| o).min() {
            if lo != gpr_lo {
                return None;
            }
        }
        // the locals rounded to 8 end in (gpr_lo - align, gpr_lo]; objects are word aligned
        let lo8 = ((gpr_lo - align) / 8 + 1) * 8;
        let hi8 = gpr_lo / 8 * 8;
        if hi8 < lo8 {
            return None;
        }
        Some((lo8 - 4, hi8))
    }

    /// Whether the address `addi rd, r1, X` at instruction `k` is the struct-return pointer of
    /// the next call.
    fn addr_is_sret(&self, k: usize, rd: u8) -> bool {
        self.addr_call_role(k, rd, |_, lay| lay.sret == Some(rd))
    }

    /// Whether the address `addi rd, r1, X` at instruction `k` is the receiver (`this`) of the
    /// next call, other than a constructor (a member constructed in place is part of its object).
    fn addr_is_receiver(&self, k: usize, rd: u8) -> bool {
        self.addr_call_role(k, rd, |sig, lay| lay.this == Some(rd) && !sig::is_ctor(sig))
    }

    fn addr_call_role(&self, k: usize, rd: u8, role: impl Fn(&FuncSig, &Layout) -> bool) -> bool {
        let mut b = self.cfg.block_of[k];
        let mut from = k + 1;
        // through a null test of the address (`addic. rd, r1, X; beq`) into its fall-through
        for _ in 0..2 {
            for j in from..self.cfg.blocks[b].end {
                let i = &self.insns[j];
                if i.is_call() || (i.is_jump() && i.reloc.is_some()) {
                    return self.call_layouts.get(&j).map_or(false, |(sg, lay, _)| role(sg, lay));
                }
                if i.is_bctrl() || defs_uses(i).0.contains(&gpr(rd)) {
                    return false;
                }
            }
            match self.cfg.blocks[b].term {
                Term::Cond { fall, .. } => {
                    b = fall;
                    from = self.cfg.blocks[b].start;
                }
                _ => return false,
            }
        }
        false
    }

    /// Whether the address `addi rd, r1, X` at instruction `k` is passed to an indirect
    /// (virtual) call, whose signature is resolved only later.
    fn addr_to_indirect_call(&self, k: usize, rd: u8) -> bool {
        let b = self.cfg.block_of[k];
        for j in k + 1..self.cfg.blocks[b].end {
            let i = &self.insns[j];
            if i.is_bctrl() {
                return true;
            }
            if i.is_call() || defs_uses(i).0.contains(&gpr(rd)) {
                return false;
            }
        }
        false
    }

    /// Size of the class object whose address `addi rd, r1, X` at instruction `k` computes, from
    /// the first call in the block that receives it (receiver, sret, reference/pointer or
    /// by-value class argument). Scalar pointees are not trusted (arrays decay to them).
    fn addr_object_size(&self, k: usize, rd: u8) -> Option<u32> {
        let b = self.cfg.block_of[k];
        let end = self.cfg.blocks[b].end;
        let class_size = |t: &Type| -> Option<u32> {
            let r = types::resolve(self.db, t).into_owned();
            let r = strip_cv(&r);
            if named(r).is_some() && types::is_aggregate(self.db, r) {
                types::size_of(self.db, r)
            } else {
                None
            }
        };
        for j in k + 1..end {
            if self.frame.skip.contains(&j) {
                continue;
            }
            let i = &self.insns[j];
            if i.is_call() || (i.is_jump() && i.reloc.is_some()) {
                let (sig, lay, _) = self.call_layouts.get(&j)?;
                if lay.this == Some(rd) {
                    return class_size(&Type::Named(sig.this_class.clone()?));
                }
                if lay.sret == Some(rd) {
                    return class_size(&sig.ret);
                }
                for (n, p) in lay.params.iter().enumerate() {
                    if *p == ArgLoc::Gpr(rd) {
                        let pt = &sig.params[n].ty;
                        return match pointee(pt) {
                            Some(t) => class_size(t),
                            None => class_size(pt),
                        };
                    }
                }
                return None;
            }
            if i.is_bctrl() || defs_uses(i).0.contains(&gpr(rd)) {
                return None;
            }
        }
        None
    }

    /// Lvalue for a stack access at r1+off of `size` bytes and type `ty`.
    fn stack_lvalue(&mut self, off: i32, ty: Type) -> Option<Expr> {
        if let Some(&v) = self.stack.slots.get(&off) {
            // the same slot read as another type (float bits as int, ...): reinterpret in memory
            let vt = self.vars[v].ty.clone();
            if (is_float(&vt) != is_float(&ty) && scalar_size(&vt) == scalar_size(&ty)) || (scalar_size(&ty).is_some() && scalar_size(&vt).map_or(false, |s| s > 0) && scalar_size(&vt) != scalar_size(&ty) && !matches!(vt, Type::Unknown { .. })) {
                return Some(Expr::Member { base: Box::new(Expr::Var(v)), offset: 0, ty });
            }
            return Some(Expr::Var(v));
        }
        for &(s, z, v) in &self.stack.regions {
            if off >= s && off < s + z as i32 {
                let rel = off - s;
                return Some(Expr::Member { base: Box::new(Expr::Var(v)), offset: rel, ty });
            }
        }
        None
    }

    fn stack_addr(&mut self, off: i32) -> Expr {
        for &(s, z, v) in &self.stack.regions {
            if off >= s && off < s + z as i32 {
                if off == s {
                    return Expr::AddrOf(Box::new(Expr::Var(v)));
                }
                return Expr::AddrOf(Box::new(Expr::Member { base: Box::new(Expr::Var(v)), offset: off - s, ty: t_unk(0) }));
            }
        }
        if let Some(&v) = self.stack.slots.get(&off) {
            return Expr::AddrOf(Box::new(Expr::Var(v)));
        }
        // below the locals (the linkage area): a block copy's pre-decremented pointer into the
        // lowest object (`addi r5, r1, 4; lwzu r0, 8(r5)`)
        if off < 8 {
            if let Some(&(s, _, v)) = self.stack.regions.iter().min_by_key(|r| r.0) {
                if s - off <= 8 {
                    return Expr::AddrOf(Box::new(Expr::Member { base: Box::new(Expr::Var(v)), offset: off - s, ty: t_unk(0) }));
                }
            }
        }
        Expr::Unknown { text: format!("sp+0x{off:x}"), ty: t_ptr(Type::Void) }
    }

    // ------------------------------------------------------------ globals / literals

    fn note_global(&mut self, sym: &str, ty: &Type, is_fn: bool) {
        if self.globals.contains_key(sym) {
            return;
        }
        let d = self.obj.data.get(sym);
        let local_def = d.is_some() || self.obj.functions.iter().any(|f| f.name == sym);
        let init = d
            .filter(|d| matches!(d.section.as_str(), ".data" | ".sdata" | ".rodata" | ".sdata2") && d.relocs.is_empty() && !d.bytes.is_empty() && !is_literal_name(sym))
            .map(|d| d.bytes.clone());
        self.globals.insert(
            sym.to_string(),
            GlobalRef {
                symbol: sym.to_string(),
                ty: ty.clone(),
                is_function: is_fn,
                section: d.map(|d| d.section.clone()),
                local_def,
                init,
                abs_addr: None,
            },
        );
    }

    fn data_bytes(&self, sym: &str, addend: i64, len: usize) -> Option<Vec<u8>> {
        if let Some(b) = mwdec_obj::data_bytes(self.obj, sym, addend, len) {
            return Some(b.to_vec());
        }
        None
    }

    /// Value of a memory read from symbol+addend with the given access type, if it's a literal.
    fn literal_load(&mut self, sym: &str, addend: i64, ty: &Type) -> Option<Expr> {
        if !is_literal_name(sym) {
            return None;
        }
        let d = self.obj.data.get(sym)?;
        if !matches!(d.section.as_str(), ".sdata2" | ".rodata" | ".sbss2") {
            return None;
        }
        match ty {
            Type::Float { size: 4 } => {
                let b = self.data_bytes(sym, addend, 4)?;
                Some(Expr::Float { bits: u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as u64, double: false })
            }
            Type::Float { size: 8 } => {
                let b = self.data_bytes(sym, addend, 8)?;
                Some(Expr::Float { bits: u64::from_be_bytes(b.try_into().ok()?), double: true })
            }
            _ => None,
        }
    }

    /// Bytes of the word / double-word literal symbols the function references.
    pub fn literal_bytes(&self) -> Vec<(String, Vec<u8>)> {
        self.globals
            .keys()
            .filter(|s| is_literal_name(s))
            .filter_map(|s| {
                let d = self.obj.data.get(s)?;
                // (only constants: read-only pools, or compiler-named literals; a writable
                // splitter-named word may be a variable)
                let ro = matches!(d.section.as_str(), ".sdata2" | ".rodata");
                if !matches!(d.size, 4 | 8) || !d.relocs.is_empty() || !(ro || s.starts_with('@')) {
                    return None;
                }
                Some((s.clone(), self.data_bytes(s, 0, d.size as usize)?))
            })
            .collect()
    }

    /// Strings of the string pool before (and including) the last string the function uses,
    /// when some used string is not at the start of its pool; empty otherwise or when strings
    /// come from more than one pool.
    pub fn string_pool_prefix(&self) -> Vec<Vec<u8>> {
        let pools: HashSet<&str> = self.str_origin.values().map(|(s, _)| s.as_str()).collect();
        if pools.len() != 1 {
            return vec![];
        }
        let sym = *pools.iter().next().unwrap();
        let end = self.str_origin.iter().map(|(b, (_, a))| a + b.len() as i64 + 1).max().unwrap_or(0);
        if self.str_origin.values().all(|(_, a)| *a == 0) || end > 0x4000 {
            return vec![];
        }
        let mut out = vec![];
        let mut off = 0i64;
        while off < end {
            let Some(s) = mwdec_obj::c_string_at(self.obj, sym, off) else { return vec![] };
            off += s.len() as i64 + 1;
            out.push(s.to_vec());
        }
        out
    }

    /// (symbol, addend) of a literal-pool address held in a register: a string or a literal
    /// symbol's address, directly or through a temporary.
    fn literal_addr(&self, e: &Expr, depth: u32) -> Option<(String, i64)> {
        match e {
            Expr::Var(t) if depth < 4 => self.literal_addr(self.temp_def.get(t)?, depth + 1),
            Expr::Str { bytes } => self.str_origin.get(bytes).cloned(),
            Expr::AddrOf(g) => match &**g {
                Expr::Global { symbol, .. } if is_literal_name(symbol) => Some((symbol.clone(), 0)),
                Expr::Member { base, offset, .. } => match &**base {
                    Expr::Global { symbol, .. } if is_literal_name(symbol) => Some((symbol.clone(), *offset as i64)),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// Address of symbol+addend as an expression.
    fn sym_addr(&mut self, sym: &str, addend: i64) -> Expr {
        if self.obj.functions.iter().any(|f| f.name == sym) || (!self.obj.data.contains_key(sym) && sig::demangle(sym).map_or(false, |d| d.contains('('))) {
            self.note_global(sym, &Type::Void, true);
            return Expr::FuncAddr { symbol: sym.to_string() };
        }
        if is_literal_name(sym) {
            if let Some(d) = self.obj.data.get(sym) {
                if matches!(d.section.as_str(), ".rodata" | ".data" | ".sdata" | ".sdata2") {
                    if let Some(s) = mwdec_obj::c_string_at(self.obj, sym, addend) {
                        let printable = !s.is_empty() && s.iter().all(|&c| c >= 0x20 && c < 0x7f || c == b'\n' || c == b'\t');
                        // a small literal object that is exactly one string (`"dvdfs.c"` in .sdata)
                        let lone = addend == 0 && d.size as usize == s.len() + 1;
                        let is_strpool = sym.contains("stringBase") || d.section == ".rodata" || d.section == ".data" || lone;
                        if printable && is_strpool && (addend == 0 || sym.contains("stringBase")) {
                            self.str_origin.insert(s.to_vec(), (sym.to_string(), addend));
                            return Expr::Str { bytes: s.to_vec() };
                        }
                        // an empty string: a lone byte, or a string pool's entry (`""` ahead of
                        // the unit's other literals)
                        if s.is_empty() && (d.size == 1 || sym.contains("stringBase")) {
                            self.empty_str_origin = Some((sym.to_string(), addend));
                            return Expr::Str { bytes: vec![] };
                        }
                    }
                }
            }
        }
        let ty = self.global_type(sym);
        self.note_global(sym, &ty, false);
        let g = Expr::Global { symbol: sym.to_string(), ty: ty.clone() };
        if addend == 0 {
            Expr::AddrOf(Box::new(g))
        } else {
            Expr::AddrOf(Box::new(Expr::Member { base: Box::new(g), offset: addend as i32, ty: t_unk(0) }))
        }
    }

    fn global_type(&self, sym: &str) -> Type {
        if let Some(db) = self.db {
            if let Some((_, t)) = db.globals.get(sym) {
                return t.clone();
            }
        }
        match self.obj.data.get(sym) {
            Some(d) => t_unk(d.size),
            // an extern only ever addressed absolutely (`@ha`/`@l`), never small-data relative:
            // the compiler only does that for an object larger than the small-data limit, so a
            // scalar declaration would turn its accesses into SDA ones
            // (static initializers define their objects from the constructor calls instead)
            // (static initializers define the objects they construct from the constructor calls:
            // only an object written through its address keeps the rule there)
            None if self.far_only(sym) && (!self.f.name.starts_with("__sinit_") || self.addressed_by_memory_ops(sym)) => t_unk(FAR_EXTERN_SIZE),
            None => t_unk(0),
        }
    }

    /// The address of `sym` (`addi rD, rX, sym@l`) is used as the base of loads/stores before
    /// rD is redefined, and never passed to a call.
    fn addressed_by_memory_ops(&self, sym: &str) -> bool {
        let mut based = false;
        for (k, i) in self.insns.iter().enumerate() {
            if !(i.op() == ppc750cl::Opcode::Addi && i.reloc.as_ref().is_some_and(|r| r.target == sym)) {
                continue;
            }
            let rd = i.rd();
            let end = self.cfg.blocks[self.cfg.block_of[k]].end;
            for j in k + 1..end {
                let n = &self.insns[j];
                let (d, u) = defs_uses(n);
                if n.is_call() || n.is_bctrl() {
                    if (3..=10).contains(&rd) {
                        return false;
                    }
                    break;
                }
                if u.contains(&gpr(rd)) && n.ra() == rd && matches!(n.op(), ppc750cl::Opcode::Stw | ppc750cl::Opcode::Sth | ppc750cl::Opcode::Stb | ppc750cl::Opcode::Lwz | ppc750cl::Opcode::Lhz | ppc750cl::Opcode::Lbz | ppc750cl::Opcode::Stfs | ppc750cl::Opcode::Stfd | ppc750cl::Opcode::Lfs | ppc750cl::Opcode::Lfd) {
                    based = true;
                }
                if d.contains(&gpr(rd)) {
                    break;
                }
            }
        }
        based
    }

    fn far_only(&self, sym: &str) -> bool {
        let mut far = false;
        for i in &self.insns {
            if let Some(r) = i.reloc.as_ref().filter(|r| r.target == sym) {
                match r.kind {
                    RelocKind::Addr16Ha | RelocKind::Addr16Lo | RelocKind::Addr16Hi => far = true,
                    _ => return false,
                }
            }
        }
        far
    }

    /// Memory lvalue for symbol+addend accessed with type `ty`.
    fn sym_lvalue(&mut self, sym: &str, addend: i64, ty: Type) -> Expr {
        let gty = self.global_type(sym);
        self.note_global(sym, &gty, false);
        let g = Expr::Global { symbol: sym.to_string(), ty: gty.clone() };
        let gsize = types::size_of(self.db, &gty).unwrap_or(0);
        // (an object larger than the small-data limit stays an object: a scalar declaration
        // would move its accesses to small-data addressing)
        if addend == 0 && (gsize == scalar_size(&ty).unwrap_or(0) || matches!(gty, Type::Unknown { size } if size <= 8)) {
            if let Type::Unknown { .. } = gty {
                return Expr::Global { symbol: sym.to_string(), ty };
            }
            if !types::is_aggregate(self.db, &gty) {
                return g;
            }
        }
        // a scalar access into an aggregate global: resolve the member through the DB
        self.member(g, addend as i32, ty)
    }

    // ------------------------------------------------------------ memory helpers

    /// Build an lvalue for `base + off` accessed with type `ty`.
    fn mem(&mut self, base: Expr, off: i32, ty: Type) -> Expr {
        match base {
            Expr::AddrOf(inner) => match *inner {
                Expr::Load { base: b, offset: o, .. } => self.mem(*b, o + off, ty),
                Expr::Member { base: b, offset: o, .. } => self.member(*b, o + off, ty),
                Expr::Var(v) if matches!(self.vars[v].kind, VarKind::Stack { .. }) => self.member(Expr::Var(v), off, ty),
                Expr::Var(v) if matches!(self.vars[v].kind, VarKind::Param { .. }) => self.member(Expr::Var(v), off, ty),
                Expr::Global { symbol, ty: gty } => self.member(Expr::Global { symbol, ty: gty }, off, ty),
                call @ Expr::Call { .. } => self.member(call, off, ty),
                other => Expr::Load { base: Box::new(Expr::AddrOf(Box::new(other))), offset: off, ty },
            },
            Expr::Binary { op: BinOp::Add, l, r, ty: bt } if r.as_int().is_some() && !is_ptr(&bt) => {
                let k = r.as_int().unwrap() as i32;
                self.mem(*l, off + k, ty)
            }
            base => {
                let ty = self.refine_load_type(&base, off, ty);
                Expr::Load { base: Box::new(base), offset: off, ty }
            }
        }
    }

    fn member(&mut self, base: Expr, off: i32, ty: Type) -> Expr {
        // an extern of unknown size read at its start is a scalar of the access type
        if let Expr::Global { symbol, ty: Type::Unknown { size: 0 } } = &base {
            if off == 0 && scalar_size(&ty).map_or(false, |s| s > 0) {
                return Expr::Global { symbol: symbol.clone(), ty };
            }
        }
        if off == 0 {
            let bt = ty_of(&base, &self.vars);
            if scalar_size(&bt).is_some() && scalar_size(&bt) == scalar_size(&ty) && !matches!(bt, Type::Unknown { .. }) {
                return base;
            }
            if let Type::Unknown { size } = bt {
                if Some(size) == scalar_size(&ty) {
                    // retype the slot var on first scalar use
                    if let Expr::Var(v) = base {
                        if matches!(self.vars[v].kind, VarKind::Stack { .. }) {
                            self.vars[v].ty = ty.clone();
                        }
                    }
                    return base;
                }
            }
        }
        // field type through the DB
        let bt = ty_of(&base, &self.vars);
        let mut ty = ty;
        if let (Some(db), Some(cls)) = (self.db, named(&bt).map(|s| s.to_string())) {
            if let Some((_, ft)) = types::field_path(db, &cls, off, scalar_size(&ty).unwrap_or(0)) {
                if compatible_scalar(self.db, &ft, &ty) {
                    ty = ft;
                }
            }
        }
        Expr::Member { base: Box::new(base), offset: off, ty }
    }

    /// Use the DB's field type when it agrees with the access width/kind.
    fn refine_load_type(&self, base: &Expr, off: i32, ty: Type) -> Type {
        let bt = ty_of(base, &self.vars);
        // `*p` through a typed scalar pointer: the pointee type (char vs unsigned char, ...)
        if off == 0 {
            if let Some(p) = pointee(&bt) {
                let pr = types::resolve(self.db, p).into_owned();
                let ps = strip_cv(&pr);
                if !types::is_aggregate(self.db, ps) && compatible_scalar(self.db, ps, &ty) && !matches!(ps, Type::Void) {
                    return ps.clone();
                }
            }
        }
        let Some(db) = self.db else { return ty };
        let Some(p) = pointee(&bt) else { return ty };
        let Some(cls) = named(&types::resolve(Some(db), p)).map(|s| s.to_string()) else { return ty };
        match types::field_path(db, &cls, off, scalar_size(&ty).unwrap_or(0)) {
            Some((_, ft)) if compatible_scalar(self.db, &ft, &ty) => ft,
            _ => ty,
        }
    }

    // ------------------------------------------------------------ state ops

    fn mutable_var(&self, v: VarId) -> bool {
        if self.is_temp[v] {
            return false;
        }
        match self.vars[v].kind {
            VarKind::This | VarKind::StructRet | VarKind::Hidden => false,
            // (a reference parameter's object is reassigned by stores through it)
            VarKind::Param { .. } => self.web_var.values().any(|&w| w == v) || self.entry_vals.values().any(|e| matches!(e, Expr::AddrOf(x) if matches!(**x, Expr::Var(w) if w == v))),
            _ => true,
        }
    }

    /// Does `e` read a variable that may be reassigned? (`&v` does not read `v`.)
    fn refs_mutable(&self, e: &Expr) -> bool {
        match e {
            Expr::Var(v) => self.mutable_var(*v),
            Expr::AddrOf(inner) => match &**inner {
                Expr::Var(_) => false,
                Expr::Member { base, .. } => !matches!(**base, Expr::Var(_)) && self.refs_mutable(base),
                other => self.refs_mutable_inner(other),
            },
            other => self.refs_mutable_inner(other),
        }
    }

    fn refs_mutable_inner(&self, e: &Expr) -> bool {
        let mut r = false;
        e.walk(&mut |x| {
            if let Expr::Var(v) = x {
                if self.mutable_var(*v) {
                    r = true;
                }
            }
        });
        r
    }

    /// Before assigning mutable var `v`, materialize register/CR values that read it.
    fn snapshot(&mut self, st: &mut St, v: VarId, except: Option<Reg>) {
        for r in 0..NREGS {
            if Some(r as Reg) == except {
                continue;
            }
            let Some(e) = st.regs[r].clone() else { continue };
            if !e.reads_var(v) {
                continue;
            }
            let ty = ty_of(&e, &self.vars);
            let t = self.new_var(format!("temp_{}", reg_name(r as Reg)), ty, VarKind::Local, true);
            st.out.push(Stmt::Assign { dst: Expr::Var(t), src: e.clone() });
            self.temp_def.insert(t, e);
            st.regs[r] = Some(Expr::Var(t));
            if let Some(site) = st.site[r] {
                self.def_value.insert(site, Expr::Var(t));
            }
        }
        for b in 0..32 {
            if let Some(e) = st.cr[b].clone() {
                if e.reads_var(v) {
                    let t = self.new_var("temp_cr".into(), Type::Bool, VarKind::Local, true);
                    st.out.push(Stmt::Assign { dst: Expr::Var(t), src: e.clone() });
                    st.cr[b] = Some(Expr::Var(t));
                }
            }
        }
    }

    /// Define register `reg` at instruction `k` with value `e`.
    fn def(&mut self, st: &mut St, k: usize, reg: Reg, e: Expr) {
        let site = (k as u32, reg);
        st.site[reg as usize] = Some(site);
        // a temp copied into a callee-saved register lives there: name it after that home
        // (the emitter orders declarations by register)
        if let Expr::Var(t) = &e {
            let t = *t;
            let callee_saved = (14..32).contains(&reg) || (32 + 14..64).contains(&reg);
            if self.is_temp.get(t).copied().unwrap_or(false) && callee_saved && !self.web_var.contains_key(&site) {
                let n = &self.vars[t].name;
                let home_is_volatile = n
                    .strip_prefix("temp_")
                    .and_then(|r| r[1..].split('_').next().and_then(|d| d.parse::<u8>().ok()))
                    .map_or(false, |d| d < 14);
                if home_is_volatile {
                    self.vars[t].name = format!("temp_{}", reg_name(reg));
                }
            }
        }
        if let Some(&v) = self.web_var.get(&site) {
            self.snapshot(st, v, Some(reg));
            // type refinement of the web var
            let et = ty_of(&e, &self.vars);
            if matches!(self.vars[v].ty, Type::Unknown { .. }) && !matches!(et, Type::Unknown { .. } | Type::Void) {
                self.vars[v].ty = et;
            }
            if e != Expr::Var(v) {
                st.out.push(Stmt::Assign { dst: Expr::Var(v), src: e });
            }
            st.regs[reg as usize] = Some(Expr::Var(v));
            self.def_value.insert(site, Expr::Var(v));
            return;
        }
        let uses = self.use_count.get(&site).copied().unwrap_or(0);
        let cross = self.use_blocks.get(&site).map_or(false, |s| s.iter().any(|&b| b != self.cfg.block_of[k]));
        // a member address kept in a callee-saved register (`addi r31, r30, 4` read after
        // calls) was a local in the source (`T& m = obj.GetMember();`):
        // re-evaluated at each use it would be recomputed in a volatile register
        let kept_address = (14..32).contains(&reg)
            && uses > 0
            && matches!(&e, Expr::AddrOf(inner) if matches!(&**inner, Expr::Member { base, .. } | Expr::Load { base, .. } if !matches!(&**base, Expr::Var(v) if matches!(self.vars[*v].kind, VarKind::Stack { .. }))));
        if is_trivial(&e) && !(cross && self.refs_mutable(&e)) && !kept_address {
            st.regs[reg as usize] = Some(e.clone());
            self.def_value.insert(site, e);
            return;
        }
        if uses == 0 {
            if e.has_call() {
                st.out.push(Stmt::Expr(e));
                st.regs[reg as usize] = None;
            } else {
                st.regs[reg as usize] = Some(e);
            }
            return;
        }
        let mut ty = ty_of(&e, &self.vars);
        // a floating-point register holds a float whatever the value's guessed type
        // (an untyped literal pool word read with lfs)
        let mut e = e;
        if (32..64).contains(&reg) && matches!(strip_cv(&ty), Type::Unknown { size: 4 } | Type::Int { size: 4, .. }) {
            ty = t_f32();
            // (the word itself is read as a float, not converted)
            if let Expr::Global { ty: gt, .. } | Expr::Load { ty: gt, .. } | Expr::Member { ty: gt, .. } = &mut e {
                *gt = t_f32();
            }
        }
        let t = self.new_var(format!("temp_{}", reg_name(reg)), ty, VarKind::Local, true);
        st.out.push(Stmt::Assign { dst: Expr::Var(t), src: e.clone() });
        self.temp_def.insert(t, e);
        st.regs[reg as usize] = Some(Expr::Var(t));
        self.def_value.insert(site, Expr::Var(t));
    }

    fn get(&mut self, st: &St, r: Reg) -> Expr {
        match &st.regs[r as usize] {
            Some(e) => e.clone(),
            None => Expr::Unknown { text: format!("uninit {}", reg_name(r)), ty: if (32..64).contains(&r) { t_f32() } else { t_unk(4) } },
        }
    }

    fn gpr_or_zero(&mut self, st: &St, r: u8) -> Expr {
        if r == 0 {
            Expr::int(0)
        } else {
            self.get(st, gpr(r))
        }
    }

    fn store(&mut self, st: &mut St, dst: Expr, src: Expr) {
        // writing a mutable var directly (stack slot): snapshot readers
        if let Expr::Var(v) = &dst {
            let v = *v;
            self.snapshot(st, v, None);
        }
        if let Expr::Member { base, .. } = &dst {
            if let Expr::Var(v) = **base {
                self.snapshot(st, v, None);
            }
        }
        st.out.push(Stmt::Assign { dst, src });
    }

    fn set_cr_cmp(&mut self, st: &mut St, field: u8, l: Expr, r: Expr, float: bool) {
        let b = field as usize * 4;
        st.cr[b] = Some(Expr::cmp(BinOp::Lt, l.clone(), r.clone()));
        st.cr[b + 1] = Some(Expr::cmp(BinOp::Gt, l.clone(), r.clone()));
        st.cr[b + 2] = Some(Expr::cmp(BinOp::Eq, l.clone(), r.clone()));
        st.cr[b + 3] = if float { Some(Expr::Unknown { text: "unordered".into(), ty: Type::Bool }) } else { None };
    }

    fn record(&mut self, st: &mut St, k: usize, r: Reg) {
        let v = self.get(st, r);
        let vt = ty_of(&v, &self.vars);
        let z = if is_ptr(&vt) { Expr::Int { value: 0, ty: vt.clone() } } else { Expr::int(0) };
        let sv = as_signed(v.clone(), &self.vars);
        st.cr[0] = Some(Expr::cmp(BinOp::Lt, sv.clone(), Expr::int(0)));
        st.cr[1] = Some(Expr::cmp(BinOp::Gt, sv, Expr::int(0)));
        st.cr[2] = Some(Expr::cmp(BinOp::Eq, v, z));
        st.cr[3] = None;
        self.save_cr(st, k, 0);
    }

    fn save_cr(&mut self, st: &St, k: usize, field: u8) {
        let b = field as usize * 4;
        self.cr_value.insert(k as u32 * 8 + field as u32, [st.cr[b].clone(), st.cr[b + 1].clone(), st.cr[b + 2].clone(), st.cr[b + 3].clone()]);
    }

    // ------------------------------------------------------------ main

    /// C functions whose prototype the context doesn't have: parameters are the argument
    /// registers read before being written.
    /// GC/1.2.5n frame size for a function with `slots` 4-byte parameter/local slots and no
    /// address-taken locals: linkage (8), then (only when the frame has a local area at all:
    /// saved registers or slots) the slots rounded to 8, then the register saves; all rounded to 8.
    pub fn sdk_frame_size(&self, slots: u32) -> u32 {
        let saves = 4 * self.frame.info.saved_gprs.len() as u32 + 8 * self.frame.info.saved_fprs.len() as u32;
        let area = if saves > 0 || slots > 0 { (4 * slots + 7) & !7 } else { 0 };
        (8 + area + saves + 7) & !7
    }

    /// Lowest r1 offset of a stack object (see `lowest_local_offset`).
    pub fn lowest_stack_offset(&self) -> Option<i32> {
        self.lowest_local_offset()
    }

    /// Any r1-relative access outside the prologue/epilogue (a stack object).
    pub fn has_stack_objects(&self) -> bool {
        self.lowest_local_offset().is_some()
    }

    /// Lowest r1 offset of a local stack object (an address taken or a slot accessed), outside
    /// the linkage area and the register save area.
    fn lowest_local_offset(&self) -> Option<i32> {
        if self.frame.info.size == 0 {
            return None;
        }
        let mut lo: Option<i32> = None;
        for (k, i) in self.insns.iter().enumerate() {
            if self.frame.skip.contains(&k) || i.reloc.is_some() || i.ra() != 1 {
                continue;
            }
            use ppc750cl::Opcode::*;
            let off = match i.op() {
                Addi => i.simm() as i32,
                Lwz | Lhz | Lha | Lbz | Stw | Sth | Stb | Lfs | Lfd | Stfs | Stfd => i.disp(),
                _ => continue,
            };
            if off >= 8 && off < self.frame.saves_lo && !self.frame.save_slots.contains_key(&off) {
                lo = Some(lo.map_or(off, |l: i32| l.min(off)));
            }
        }
        lo
    }

    fn infer_c_params(&mut self) {
        if sig::demangle(&self.f.name).is_some() || !self.sig.params.is_empty() {
            return;
        }
        if self.db.map_or(false, |db| db.decls.contains_key(&self.f.name) || db.functions.contains_key(&self.f.name)) {
            return;
        }
        for k in 0..self.insns.len() {
            let i = &self.insns[k];
            if i.is_call() || (i.is_jump() && i.reloc.is_some()) {
                if let Some(cl) = self.call_layout(k) {
                    self.call_layouts.insert(k, cl);
                }
            }
        }
        self.reaching_defs();
        let mut live = HashSet::new();
        self.for_each_use(|_, r, defs| {
            if defs.contains(&ENTRY) {
                live.insert(r);
            }
        });
        let mut gmax = (3..=10u8).rev().find(|r| live.contains(&gpr(*r)));
        let fmax = (1..=8u8).rev().find(|r| live.contains(&fpr(*r)));
        // unused trailing parameters still own a home slot: the lowest local sits above them,
        // or (no stack objects) the frame is larger than the used parameters' slots make it
        if self.param_home_slots && fmax.is_none() && self.frame.info.size > 0 {
            if let Some(lo) = self.lowest_local_offset() {
                let slots = ((lo - 8) / 4) as u8;
                if (1..=8).contains(&slots) && gmax.map_or(true, |g| g - 2 < slots) {
                    gmax = Some(2 + slots);
                }
            } else {
                let used = gmax.map_or(0, |g| (g - 2) as u32);
                if self.sdk_frame_size(used) < self.frame.info.size {
                    if let Some(n) = (used + 1..=8).find(|&n| self.sdk_frame_size(n) == self.frame.info.size) {
                        gmax = Some(2 + n as u8);
                    }
                }
            }
        }
        if let Some(g) = gmax {
            for _ in 3..=g {
                self.sig.params.push(mwdec_core::Param { name: None, ty: t_s32() });
            }
        }
        if let Some(fm) = fmax {
            for _ in 1..=fm {
                self.sig.params.push(mwdec_core::Param { name: None, ty: t_f32() });
            }
        }
        self.call_layouts.clear();
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        self.infer_c_params();
        self.setup_params();
        for k in 0..self.insns.len() {
            if self.frame.skip.contains(&k) {
                continue;
            }
            let i = &self.insns[k];
            if i.is_call() || (i.is_jump() && i.reloc.is_some()) {
                if let Some(cl) = self.call_layout(k) {
                    self.call_layouts.insert(k, cl);
                }
            }
        }
        self.setup_stack_args();
        self.reaching_defs();
        self.infer_return();
        self.find_split_returns();
        self.compute_webs();
        self.scan_stack();
        let nb = self.cfg.blocks.len();
        self.blocks_out = (0..nb).map(|_| BlockOut { stmts: vec![], cond: None, switch: None, ret: None }).collect();
        for &b in &self.cfg.rpo.clone() {
            let out = self.translate_block(b);
            self.blocks_out[b] = out;
        }
        Ok(())
    }

    fn entry_state(&mut self, b: usize) -> St {
        let mut st = St {
            regs: vec![None; NREGS],
            site: vec![None; NREGS],
            cr: vec![None; 32],
            ca: Ca::Unknown,
            mem: BTreeMap::new(),
            out: vec![],
        };
        for r in 0..NREGS {
            let defs = self.rd_in[b][r].clone();
            if defs.is_empty() {
                continue;
            }
            if r >= 64 && r < 72 {
                // CR field: single reaching def
                if defs.len() == 1 && defs[0] != ENTRY {
                    if let Some(vals) = self.cr_value.get(&(defs[0] * 8 + (r as u32 - 64))) {
                        for q in 0..4 {
                            st.cr[(r - 64) * 4 + q] = vals[q].clone();
                        }
                    }
                }
                continue;
            }
            let site = (defs[0], r as Reg);
            if defs.len() > 1 || self.web_var.contains_key(&site) {
                if let Some(&v) = self.web_var.get(&site) {
                    // a web holding an incoming value that isn't the parameter variable itself
                    // (a reference parameter stepped as a pointer: `p = &ref; ... p += 16`)
                    // starts from it
                    if b == 0 && self.cfg.blocks[0].preds.is_empty() && defs == [ENTRY] {
                        if let Some(e) = self.entry_vals.get(&(r as Reg)) {
                            if *e != Expr::Var(v) {
                                let et = ty_of(e, &self.vars);
                                if matches!(self.vars[v].ty, Type::Unknown { .. }) && !matches!(et, Type::Unknown { .. }) {
                                    self.vars[v].ty = et;
                                }
                                st.out.push(Stmt::Assign { dst: Expr::Var(v), src: e.clone() });
                            }
                        }
                    }
                    st.regs[r] = Some(Expr::Var(v));
                }
                continue;
            }
            if defs[0] == ENTRY {
                if let Some(e) = self.entry_vals.get(&(r as Reg)) {
                    st.regs[r] = Some(e.clone());
                }
                continue;
            }
            if let Some(e) = self.def_value.get(&site) {
                st.regs[r] = Some(e.clone());
                st.site[r] = Some(site);
            }
        }
        st
    }

    fn translate_block(&mut self, b: usize) -> BlockOut {
        let mut st = self.entry_state(b);
        let (start, end) = (self.cfg.blocks[b].start, self.cfg.blocks[b].end);
        let mut cond = None;
        let mut ret = None;
        for k in start..end {
            if self.frame.skip.contains(&k) {
                // still process blr/terminators below
                if !(self.insns[k].is_blr() || k == end - 1) {
                    continue;
                }
                if !self.insns[k].is_blr() {
                    continue;
                }
            }
            let i = self.insns[k].clone();
            if k == end - 1 {
                match self.cfg.blocks[b].term.clone() {
                    Term::Cond { .. } => {
                        cond = Some(self.branch_cond_k(&mut st, k, &i));
                        continue;
                    }
                    Term::CondReturn { .. } => {
                        cond = Some(self.branch_cond(&st, &i));
                        if let Some(r) = self.ret_reg {
                            ret = Some(self.ret_value(&st, r));
                        }
                        continue;
                    }
                    Term::Return => {
                        if let Some(r) = self.ret_reg {
                            ret = Some(self.ret_value(&st, r));
                        }
                        continue;
                    }
                    Term::Jump(_) => continue,
                    Term::Switch { .. } => continue,
                    Term::TailCall => {
                        let e = self.do_call(&mut st, k, &i, true);
                        ret = e;
                        continue;
                    }
                    _ => {}
                }
            }
            self.step(&mut st, k, &i);
        }
        if let Some(r) = self.ret_reg {
            if self.cfg.blocks[b].succs.iter().any(|s| self.split_returns.contains(s)) {
                ret = Some(self.ret_value(&st, r));
            }
        }
        let switch = self.switch_index.get(&b).cloned();
        BlockOut { stmts: st.out, cond, switch, ret }
    }

    /// Branch condition including CTR-decrementing forms (`bdnz`): the decrement becomes an
    /// assignment to the CTR variable.
    fn branch_cond_k(&mut self, st: &mut St, k: usize, i: &Insn) -> Expr {
        let bo = i.ins.field_bo();
        if bo & 0x04 != 0 {
            return self.branch_cond(st, i);
        }
        let ctr = self.get(st, CTR);
        let dec = arith(BinOp::Sub, ctr, Expr::int(1), &self.vars);
        self.def(st, k, CTR, dec);
        let c = self.get(st, CTR);
        let test = Expr::cmp(if bo & 0x02 != 0 { BinOp::Eq } else { BinOp::Ne }, c, Expr::int(0));
        if bo & 0x10 == 0 {
            let bit = self.branch_cond(st, i);
            return Expr::cmp(BinOp::LogAnd, test, bit);
        }
        test
    }

    fn branch_cond(&mut self, st: &St, i: &Insn) -> Expr {
        let bo = i.ins.field_bo();
        let bi = i.ins.field_bi() as usize;
        if bo & 0x10 != 0 {
            return Expr::Int { value: 1, ty: Type::Bool };
        }
        let bit = st.cr[bi].clone().unwrap_or(Expr::Unknown { text: format!("cr bit {bi}"), ty: Type::Bool });
        if bo & 0x08 != 0 {
            bit
        } else {
            bit.negate(&self.vars)
        }
    }

    fn do_call(&mut self, st: &mut St, k: usize, i: &Insn, tail: bool) -> Option<Expr> {
        let sym = i.reloc.as_ref().map(|r| r.target.clone()).unwrap_or_default();
        // 64-bit runtime helpers are operators: `__shl2i(hi, lo, n)` is `x << n`
        if let Some((op, signed, pair_b)) = crate::wide::helper(&sym) {
            let a = crate::wide::pair(self.get(st, gpr(3)), self.get(st, gpr(4)), signed, &self.vars);
            let b = if pair_b { crate::wide::pair(self.get(st, gpr(5)), self.get(st, gpr(6)), signed, &self.vars) } else { self.get(st, gpr(5)) };
            let e = Expr::bin(op, a, b, Type::Int { size: 8, signed });
            if tail {
                return Some(e);
            }
            self.clobber(st);
            self.def_wide(st, k, e);
            return None;
        }
        let Some((mut sig, mut lay, has_this)) = self.call_layouts.get(&k).cloned() else {
            self.warn(format!("call without layout at {:#x}", i.off));
            return None;
        };
        self.wide_args_unprototyped(st, &sym, &mut sig, &mut lay);
        let mut args = vec![];
        for (n, p) in lay.params.iter().enumerate() {
            let pt = sig.params[n].ty.clone();
            let a = match p {
                ArgLoc::Gpr(r) => {
                    let v = self.get(st, gpr(*r));
                    if types::is_aggregate(self.db, &pt) && !is_ptr(&pt) {
                        // by-value aggregate: pass the pointed-to object
                        if let Expr::AddrOf(inner) = &v {
                            let inner = (**inner).clone();
                            self.type_stack_object(&inner, &pt, true);
                        }
                        match v {
                            Expr::AddrOf(inner) => match *inner {
                                Expr::Load { base, offset, ty: Type::Unknown { size: 0 } } => Expr::Load { base, offset, ty: pt.clone() },
                                Expr::Member { base, offset, ty: Type::Unknown { size: 0 } } => Expr::Member { base, offset, ty: pt.clone() },
                                other => other,
                            },
                            other => {
                                let t = pt.clone();
                                Expr::Load { base: Box::new(other), offset: 0, ty: t }
                            }
                        }
                    } else {
                        self.coerce_arg(v, &pt)
                    }
                }
                ArgLoc::Fpr(r) => self.get(st, fpr(*r)),
                ArgLoc::GprPair(r) if scalar_size(&types::ty_of(&self.get(st, gpr(*r)), &self.vars)) == Some(8) => self.get(st, gpr(*r)),
                ArgLoc::GprPair(r) => {
                    // 64-bit argument: high word in r, low word in r+1
                    let signed = is_signed(&pt).unwrap_or(true);
                    let (h, l) = (self.get(st, gpr(*r)), self.get(st, gpr(*r + 1)));
                    self.pair_of(h, l, signed)
                }
                ArgLoc::Stack => {
                    let offs = self.stack_arg_offsets(&sig, &lay);
                    let o = offs.iter().find(|x| x.0 == n).map(|x| 8 + x.1).unwrap_or(0);
                    let found = (o..o + 8).find_map(|q| st.mem.get(&q).map(|x| x.1.clone()));
                    found.unwrap_or(Expr::Unknown { text: "stack arg".into(), ty: pt.clone() })
                }
            };
            args.push(a);
        }
        // a constant bound to a `const T&` parameter: the compiler materialised it as an
        // anonymous object (`@N` / a splitter label); the source passed the value
        for (n, a) in args.iter_mut().enumerate() {
            let Some(pt) = sig.params.get(n).map(|p| p.ty.clone()) else { continue };
            let Type::Ref(inner) = strip_cv(&pt) else { continue };
            if !matches!(**inner, Type::Const(_)) {
                continue;
            }
            let t = types::resolve(self.db, strip_cv(inner)).into_owned();
            let Some(sz) = scalar_size(&t).filter(|z| matches!(z, 1 | 2 | 4 | 8)) else { continue };
            let (sym, add) = match &*a {
                Expr::AddrOf(g) => match &**g {
                    Expr::Global { symbol, .. } => (symbol.clone(), 0i64),
                    Expr::Member { base, offset, .. } => match &**base {
                        Expr::Global { symbol, .. } => (symbol.clone(), *offset as i64),
                        _ => continue,
                    },
                    _ => continue,
                },
                Expr::Str { bytes } => match self.str_origin.get(bytes).or(if bytes.is_empty() { self.empty_str_origin.as_ref() } else { None }) {
                    Some((s, o)) => (s.clone(), *o),
                    None => continue,
                },
                _ => continue,
            };
            if !is_literal_name(&sym) || !self.obj.data.get(&sym).is_some_and(|d| matches!(d.section.as_str(), ".sdata" | ".sdata2" | ".rodata") && d.size as i64 >= add + sz as i64) {
                continue;
            }
            let Some(b) = self.data_bytes(&sym, add, sz as usize) else { continue };
            let raw = b.iter().fold(0u64, |acc, &x| (acc << 8) | x as u64);
            let lit = match strip_cv(&t) {
                Type::Float { size: 4 } => Expr::Float { bits: raw, double: false },
                Type::Float { size: 8 } => Expr::Float { bits: raw, double: true },
                tt if types::is_enum(self.db, tt) || matches!(tt, Type::Int { .. } | Type::Char | Type::Bool | Type::WChar | Type::Long { .. }) => {
                    let signed = is_signed(tt).unwrap_or(true);
                    let v = match (sz, signed) {
                        (1, true) => raw as u8 as i8 as i64,
                        (2, true) => raw as u16 as i16 as i64,
                        (4, true) => raw as u32 as i32 as i64,
                        _ => raw as i64,
                    };
                    Expr::Int { value: v, ty: tt.clone() }
                }
                _ => continue,
            };
            *a = lit;
        }
        if has_this && self.deleting_dtor_call(st, k, &sig) {
            args.push(Expr::int(1));
        }
        let ret_ty = if !sig::ret_unknown(&sig) {
            sig.ret.clone()
        } else {
            let r3 = self.use_count.get(&(k as u32, gpr(3))).copied().unwrap_or(0);
            let f1 = self.use_count.get(&(k as u32, fpr(1))).copied().unwrap_or(0);
            if tail {
                self.ret_ty.clone()
            } else if r3 > 0 {
                t_unk(4)
            } else if f1 > 0 {
                t_f32()
            } else {
                Type::Void
            }
        };
        self.note_global(&sym, &Type::Void, true);
        // type stack objects from how they are used: receiver of a method / reference argument
        if has_this {
            if let (Expr::AddrOf(inner), Some(c)) = (self.get(st, gpr(lay.this.unwrap())), sig.this_class.clone()) {
                self.type_stack_object(&inner, &Type::Named(c), sig::is_ctor(&sig));
            }
        }
        for (n, a) in args.iter().enumerate() {
            if let Expr::AddrOf(inner) = a {
                if let Some(t) = pointee(&sig.params[n].ty) {
                    let t = strip_cv(t).clone();
                    if named(&t).is_some() && types::is_aggregate(self.db, &t) {
                        self.type_stack_object(inner, &t, false);
                    }
                }
            }
        }
        self.type_scalar_out_args(&args, &sig);
        let mut sig = sig;
        sig.params.retain(|p| p.name.as_deref() != Some("..."));
        let callee = if has_this {
            let this = self.get(st, gpr(lay.this.unwrap()));
            // deleting destructor call (flag 1 in the next register): `delete p`
            if sig::is_dtor(&sig) && !tail && lay.this == Some(3) && self.reg_written_in_block(k, gpr(4)) && self.get(st, gpr(4)).as_int() == Some(1) {
                let call = delete_call(this, &sig);
                self.clobber(st);
                st.out.push(Stmt::Expr(call));
                return None;
            }
            Callee::Method { symbol: sym.clone(), sig: sig.clone(), this: Box::new(this), qualified: false }
        } else {
            Callee::Direct { symbol: sym.clone(), sig: sig.clone() }
        };
        let call = Expr::Call { callee, args, ret: ret_ty.clone() };
        if tail {
            return Some(call);
        }
        let (call, ret_ty) = match strip_cv(&ret_ty) {
            Type::Ref(inner) => {
                let inner = (**inner).clone();
                (Expr::AddrOf(Box::new(call)), t_ptr(inner))
            }
            _ => (call, ret_ty),
        };
        let sret_dst = lay.sret.map(|r| self.get(st, gpr(r)));
        let this_val = lay.this.map(|r| self.get(st, gpr(r)));
        // clobber
        for r in call_clobbers() {
            st.regs[r as usize] = None;
            st.site[r as usize] = None;
        }
        for q in [0usize, 1, 5, 6, 7] {
            for z in 0..4 {
                st.cr[q * 4 + z] = None;
            }
        }
        st.ca = Ca::Unknown;
        if let Some(dst) = sret_dst {
            // struct return into *dst
            let dst_copy = dst.clone();
            let lv = match dst {
                Expr::AddrOf(inner) => *inner,
                other => Expr::Load { base: Box::new(other), offset: 0, ty: ret_ty.clone() },
            };
            if let Expr::Var(v) = &lv {
                if matches!(self.vars[*v].ty, Type::Unknown { .. }) {
                    self.vars[*v].ty = ret_ty.clone();
                }
            }
            st.out.push(Stmt::Assign { dst: lv, src: call });
            let _ = ret_ty;
            self.def(st, k, gpr(3), dst_copy);
            return None;
        }
        let rreg = if is_float(&types::resolve(self.db, &ret_ty)) { fpr(1) } else { gpr(3) };
        // a callee without a known prototype whose r4 is read afterwards returns a 64-bit value
        let r4_read = self.use_count.get(&(k as u32, gpr(4))).copied().unwrap_or(0) > 0 || self.web_var.contains_key(&(k as u32, gpr(4)));
        let proto_known = !sig::ret_unknown(&sig) && self.db.map_or(false, |db| db.decls.contains_key(&sym) || db.functions.contains_key(&sym) || sig::demangle(&sym).is_some());
        if crate::wide::is_wide(&types::resolve(self.db, &ret_ty)) || (r4_read && !proto_known && !matches!(ret_ty, Type::Void) && rreg == gpr(3)) {
            let call = match call {
                Expr::Call { callee, args, .. } => Expr::Call { callee, args, ret: Type::Int { size: 8, signed: true } },
                other => other,
            };
            self.def_wide(st, k, call);
            return None;
        }
        if matches!(ret_ty, Type::Void) {
            st.out.push(Stmt::Expr(call));
            // MWCC constructors/destructors return `this`
            if sig::is_ctor(&sig) || sig::is_dtor(&sig) {
                if let Some(t) = this_val {
                    if sig::is_ctor(&sig) && self.use_count.get(&(k as u32, gpr(3))).copied().unwrap_or(0) > 0 {
                        if let Expr::AddrOf(x) = &t {
                            if let Expr::Var(v) = &**x {
                                self.ctor_ret_used.insert(*v);
                            }
                        }
                    }
                    self.def(st, k, gpr(3), t);
                }
            }
        } else {
            let uses = self.use_count.get(&(k as u32, rreg)).copied().unwrap_or(0);
            if uses == 0 && !self.web_var.contains_key(&(k as u32, rreg)) {
                st.out.push(Stmt::Expr(call));
            } else {
                self.def(st, k, rreg, call);
            }
        }
        None
    }

    /// Give an untyped stack region the class type it is used as (whole object at its start).
    fn type_stack_object(&mut self, lv: &Expr, t: &Type, constructed: bool) {
        if let Expr::Var(v) = lv {
            let v = *v;
            if let VarKind::Stack { size, .. } = self.vars[v].kind {
                if matches!(self.vars[v].ty, Type::Unknown { .. }) {
                    let ts = types::size_of(self.db, t);
                    // an uninitialised local of class type needs a default constructor
                    let default_ok = self.db.map_or(true, |db| {
                        let cls = named(t).unwrap_or("");
                        let key = format!("{}::{}", strip_tmpl(cls), strip_tmpl(sig::split_scope(cls).1));
                        // a user-provided default constructor with effects (`TAreaId() :
                        // value(-1) {}`) would add code the target doesn't have
                        match db.decls.get(&key) {
                            Some(ds) => ds.iter().any(|d| {
                                d.params.is_empty()
                                    && d.is_inline_defined
                                    && d.inline_body.as_deref().map_or(true, |b| b.trim().is_empty())
                                    && d.init_list.is_none()
                            }),
                            None => true,
                        }
                    });
                    // a constructor call declares the object with its class type whatever the region guess
                    if (constructed || ts.map_or(true, |s| s <= size)) && (default_ok || constructed) {
                        self.vars[v].ty = t.clone();
                    }
                }
            }
        }
    }

    /// `float& out` filled by the callee: the stack slot passed is that scalar (its region may
    /// look bigger because of padding before the register saves).
    fn type_scalar_out_args(&mut self, args: &[Expr], sig: &FuncSig) {
        for (n, a) in args.iter().enumerate() {
            let Expr::AddrOf(inner) = a else { continue };
            let Expr::Var(v) = &**inner else { continue };
            let Some(t) = sig.params.get(n).and_then(|p| pointee(&p.ty)) else { continue };
            let t = types::resolve(self.db, strip_cv(t)).into_owned();
            if matches!(t, Type::Float { size: 4 } | Type::Int { size: 4, .. } | Type::Long { .. })
                && matches!(self.vars[*v].kind, VarKind::Stack { .. })
                && matches!(self.vars[*v].ty, Type::Unknown { size } if size >= 4)
            {
                self.vars[*v].ty = t;
            }
        }
    }

    /// A callee without a prototype passed a 64-bit value: an odd-aligned register pair holding
    /// the halves of one 64-bit value is one argument, and a stale low half in the even register
    /// before it is the alignment gap, not an argument.
    fn wide_args_unprototyped(&mut self, st: &St, sym: &str, sig: &mut FuncSig, lay: &mut Layout) {
        if !self.param_home_slots {
            return;
        }
        if sig::demangle(sym).is_some() || self.db.map_or(false, |db| db.decls.contains_key(sym) || db.functions.contains_key(sym)) || lay.this.is_some() || lay.sret.is_some() {
            return;
        }
        if sig.params.len() != lay.params.len() || sig.params.iter().any(|p| p.name.is_some() || p.ty != t_s32()) && !sig.params.iter().all(|p| matches!(p.ty, Type::Float { .. }) || p.ty == t_s32()) {
            return;
        }
        let mut n = 0;
        while n + 1 < lay.params.len() {
            if let (ArgLoc::Gpr(r), ArgLoc::Gpr(r2)) = (lay.params[n].clone(), lay.params[n + 1].clone()) {
                let (h, l) = (self.get(st, gpr(r)), self.get(st, gpr(r2)));
                // (two constants only with a stale low half in the gap register before them)
                let gap_stale = r >= 5 && n > 0 && lay.params[n - 1] == ArgLoc::Gpr(r - 1) && {
                    let prev = self.get(st, gpr(r - 1));
                    let prev = self.resolve_temps(&prev);
                    crate::wide::as_lo(&prev, &self.vars).is_some()
                };
                if r % 2 == 1 && r2 == r + 1 && (gap_stale || !(h.as_int().is_some() && l.as_int().is_some())) {
                    let x = self.pair_resolved(h.clone(), l.clone()).filter(|_| {
                        // only halves recognizably taken from one 64-bit value
                        let (h, l) = (self.resolve_temps(&h), self.resolve_temps(&l));
                        crate::wide::as_hi(&h, &self.vars).is_some() || (gap_stale && h.as_int().is_some() && l.as_int().is_some()) || (crate::wide::as_lo(&l, &self.vars).is_some() && h.as_int().is_some())
                    });
                    if let Some(x) = x {
                        let xt = types::ty_of(&x, &self.vars);
                        lay.params.splice(n..n + 2, [ArgLoc::GprPair(r)]);
                        sig.params.splice(n..n + 2, [mwdec_core::Param { name: None, ty: if crate::wide::is_wide(&xt) { xt } else { Type::Int { size: 8, signed: true } } }]);
                        if n > 0 && lay.params[n - 1] == ArgLoc::Gpr(r - 1) && r - 1 >= 4 {
                            let prev = self.get(st, gpr(r - 1));
                            let prev = self.resolve_temps(&prev);
                            if crate::wide::as_lo(&prev, &self.vars).is_some() {
                                lay.params.remove(n - 1);
                                sig.params.remove(n - 1);
                                n -= 1;
                            }
                        }
                    }
                }
            }
            n += 1;
        }
    }

    /// Value through single-assignment temps (other than call results).
    fn resolve_temps(&self, e: &Expr) -> Expr {
        let mut e = e.clone();
        for _ in 0..8 {
            let mut changed = false;
            e.rewrite(&mut |x| {
                if let Expr::Var(v) = x {
                    // (never a call: its value is the temp itself)
                    if let Some(d) = self.temp_def.get(v).filter(|d| !d.has_call()) {
                        *x = d.clone();
                        changed = true;
                    }
                }
            });
            if !changed {
                break;
            }
        }
        e
    }

    /// A 64-bit value from its register halves, looking through temps; `None` when the halves
    /// aren't recognizably one value.
    fn pair_resolved(&self, hi: Expr, lo: Expr) -> Option<Expr> {
        let p = crate::wide::pair(hi.clone(), lo.clone(), true, &self.vars);
        if crate::wide::is_merged_pair(&p) {
            return Some(p);
        }
        let (h, l) = (self.resolve_temps(&hi), self.resolve_temps(&lo));
        let p = crate::wide::pair(h.clone(), l.clone(), true, &self.vars);
        if crate::wide::is_merged_pair(&p) {
            return Some(p);
        }
        // the halves of a 64-bit sum/difference: hi32(x op y) and lo32(x) op lo32(y)
        let mut h0 = hi;
        while let Expr::Var(v) = &h0 {
            match self.temp_def.get(v) {
                Some(d) if !d.has_call() => h0 = d.clone(),
                _ => break,
            }
        }
        let s = crate::wide::as_hi(&h0, &self.vars)?;
        let sdef = match &s {
            Expr::Var(v) => self.temp_def.get(v).map(|d| self.resolve_temps(d)).unwrap_or(s.clone()),
            _ => self.resolve_temps(&s),
        };
        let Expr::Binary { op: op @ (BinOp::Add | BinOp::Sub), l: x, r: y, .. } = &sdef else { return None };
        let Expr::Binary { op: lop, l: lx, r: ly, .. } = &l else { return None };
        let lo_of = |e: &Expr, w: &Expr| {
            crate::wide::as_lo(e, &self.vars).as_ref() == Some(w)
                || matches!(e, Expr::Cast { e: inner, .. } if &**inner == w)
                // the low word of a 64-bit object read separately
                || matches!((e, w), (Expr::Load { base: b1, offset: o1, ty: t1 }, Expr::Load { base: b2, offset: o2, .. }) if b1 == b2 && *o1 == *o2 + 4 && scalar_size(t1) == Some(4))
                || matches!((e, w), (Expr::Member { base: b1, offset: o1, ty: t1 }, Expr::Member { base: b2, offset: o2, .. }) if b1 == b2 && *o1 == *o2 + 4 && scalar_size(t1) == Some(4))
        };
        let ok = lop == op && ((lo_of(lx, x) && lo_of(ly, y)) || (*op == BinOp::Add && lo_of(lx, y) && lo_of(ly, x)));
        ok.then_some(s.clone())
    }

    /// [`wide::pair`], also recognizing halves computed through temps.
    fn pair_of(&self, hi: Expr, lo: Expr, signed: bool) -> Expr {
        if !self.param_home_slots {
            return crate::wide::pair(hi, lo, signed, &self.vars);
        }
        match self.pair_resolved(hi.clone(), lo.clone()) {
            Some(p) => p,
            None => crate::wide::pair(hi, lo, signed, &self.vars),
        }
    }

    fn coerce_arg(&self, v: Expr, _pt: &Type) -> Expr {
        v
    }

    /// Argument values at a call for the parameter locations of `lay`.
    fn layout_args(&mut self, st: &mut St, lay: &Layout, sig: &FuncSig) -> Vec<Expr> {
        let mut a2 = vec![];
        let offs = self.stack_arg_offsets(sig, lay);
        for (n, p) in lay.params.iter().enumerate() {
            match p {
                ArgLoc::Gpr(r) => a2.push(self.get(st, gpr(*r))),
                ArgLoc::Fpr(r) => a2.push(self.get(st, fpr(*r))),
                ArgLoc::GprPair(r) => {
                    let (h, l) = (self.get(st, gpr(*r)), self.get(st, gpr(*r + 1)));
                    let p = self.pair_of(h, l, true);
                    a2.push(p)
                }
                ArgLoc::Stack => {
                    // stored to the outgoing parameter area before the call
                    let o = offs.iter().find(|x| x.0 == n).map(|x| 8 + x.1).unwrap_or(0);
                    let found = (o..o + 8).find_map(|q| st.mem.get(&q).map(|x| x.1.clone()));
                    a2.push(found.unwrap_or(Expr::Unknown { text: "stack arg".into(), ty: t_unk(4) }))
                }
            }
        }
        a2
    }

    fn do_vcall(&mut self, st: &mut St, k: usize) {
        let ctr = self.get(st, CTR);
        // look through temps: ctr = Load(Load(obj, vptr_off), slot)
        let resolve = |e: &Expr, td: &HashMap<VarId, Expr>| -> Expr {
            match e {
                Expr::Var(v) => td.get(v).cloned().unwrap_or(e.clone()),
                _ => e.clone(),
            }
        };
        let c = resolve(&ctr, &self.temp_def);
        let (g, fl) = self.indirect_args(k);
        let this = self.get(st, gpr(3));
        let mut args: Vec<Expr> = g.iter().map(|&r| self.get(st, r)).collect();
        args.extend(fl.iter().map(|&r| self.get(st, r)));
        let r3u = self.use_count.get(&(k as u32, gpr(3))).copied().unwrap_or(0);
        let f1u = self.use_count.get(&(k as u32, fpr(1))).copied().unwrap_or(0);
        let mut ret = if r3u > 0 { t_unk(4) } else if f1u > 0 { t_f32() } else { Type::Void };
        let mut callee = Callee::Indirect(Box::new(ctr.clone()));
        if let Expr::Load { base, offset: slot, .. } = &c {
            let vt = resolve(base, &self.temp_def);
            // the vptr of an object lvalue (reference param, stack object) is a Member access
            let vt = match vt {
                Expr::Member { base: ob, offset, ty } => Expr::Load { base: Box::new(Expr::AddrOf(ob)), offset, ty },
                v => v,
            };
            // a virtual call loads the vptr at the class's vptr offset; anything else is a call
            // through a function pointer stored in a struct
            let is_vptr = |obj: &Expr, vo: i32, l: &Self| -> bool {
                let class = pointee(&ty_of(obj, &l.vars)).and_then(|t| named(t).map(|s| s.to_string()));
                match (l.db, class) {
                    (Some(db), Some(c)) => match crate::sig::find_class(db, &c) {
                        Some(cls) if !cls.is_declaration => cls.vptr_offset == Some(vo as u32),
                        _ => vo == 0,
                    },
                    _ => true,
                }
            };
            if let Expr::Load { base: obj, offset: vo, .. } = &vt.clone() {
                // MWCC vtables start with an 8-byte header: slots below 8 are struct fields
                if !is_vptr(obj, *vo, self) || *slot < 8 {
                    let (g, fl) = self.indirect_args(k);
                    let mut args: Vec<Expr> = vec![];
                    if self.reg_written_in_block(k, gpr(3)) {
                        args.push(self.get(st, gpr(3)));
                    }
                    args.extend(g.iter().map(|&r| self.get(st, r)));
                    args.extend(fl.iter().map(|&r| self.get(st, r)));
                    let call = Expr::Call { callee: Callee::Indirect(Box::new(c.clone())), args, ret: ret.clone() };
                    self.clobber(st);
                    if matches!(ret, Type::Void) {
                        st.out.push(Stmt::Expr(call));
                    } else {
                        let rreg = if is_float(&ret) { fpr(1) } else { gpr(3) };
                        self.def(st, k, rreg, call);
                    }
                    return;
                }
            }
            if let Expr::Load { base: obj, offset: vo, .. } = &vt {
                let obj = (**obj).clone();
                let class = pointee(&ty_of(&obj, &self.vars)).and_then(|t| named(t).map(|s| s.to_string()));
                let mut vsig = None;
                if let (Some(db), Some(cls)) = (self.db, class.as_deref()) {
                    // (an unnamed pure slot says nothing about the arguments: keep the ones set up)
                    if let Some(m) = types::vmethod(db, cls, *slot as u32).filter(|m| !sig::split_scope(&m.sig.qualified_name).1.is_empty()) {
                        vsig = Some(m.sig.clone());
                        if !sig::ret_unknown(&m.sig) {
                            ret = m.sig.ret.clone();
                        }
                    }
                }
                // `this` may be the object or the sret pointer may be in r3 (then obj is r4)
                let this_e = if obj == this { this.clone() } else { obj.clone() };
                if obj != this {
                    // struct-return virtual: r3 = result, r4 = this
                    if let Some(Expr::Var(_)) = Some(&this) {}
                }
                // virtual destructor with the delete flag set: `delete p`
                if let Some(s) = vsig.as_ref().filter(|s| sig::is_dtor(s)) {
                    if obj == this && self.get(st, gpr(4)).as_int() == Some(1) {
                        let call = delete_call(this_e.clone(), s);
                        self.clobber(st);
                        st.out.push(Stmt::Expr(call));
                        return;
                    }
                }
                callee = Callee::Virtual {
                    this: Box::new(this_e),
                    vtable_offset: *slot as u32,
                    vptr_offset: *vo as u32,
                    class,
                    sig: vsig.clone(),
                };
                // a class returned by value: r3 is the destination even when `this` (r4) and the
                // arguments were not set up here (forwarded: `return this->GetAimPosition(mgr, 0.f)`)
                let sret_ret = vsig.as_ref().is_some_and(|s| types::is_aggregate(self.db, strip_cv(&s.ret)) && named(strip_cv(&s.ret)).is_some());
                if obj != this && ((!args.is_empty() && args[0] == obj) || sret_ret) {
                    // sret: r3 is the destination
                    if !args.is_empty() && args[0] == obj {
                        args.remove(0);
                    }
                    if let Some(s) = &vsig {
                        // the parameters follow the result and `this` (r5..), whether or not
                        // this function set them up (forwarded incoming parameters)
                        args = self.layout_args(st, &layout(s, true, true, self.db), s);
                        self.type_scalar_out_args(&args, s);
                    }
                    let call = Expr::Call { callee, args, ret: ret.clone() };
                    let lv = match this {
                        Expr::AddrOf(inner) => *inner,
                        other => Expr::Load { base: Box::new(other), offset: 0, ty: ret.clone() },
                    };
                    self.clobber(st);
                    st.out.push(Stmt::Assign { dst: lv, src: call });
                    return;
                }
                if let Some(s) = &vsig {
                    // trim args to the signature's arity
                    args = self.layout_args(st, &layout(s, true, false, self.db), s);
                    self.type_scalar_out_args(&args, s);
                    if self.deleting_dtor_call(st, k, s) {
                        args.push(Expr::int(1));
                    }
                }
            }
        }
        if let Callee::Indirect(_) = callee {
            // plain function-pointer call: arguments are the registers set up for it, or our own
            // incoming parameters still untouched in r3.. (the compiler kept them live)
            let live = |r: u8, l: &Self| -> bool {
                if l.reg_set_for_call(k, gpr(r)) || l.passed_through(k).is_some_and(|m| r <= m && l.entry_vals.contains_key(&gpr(r))) {
                    return true;
                }
                let b = l.cfg.block_of[k];
                let untouched = (l.cfg.blocks[b].start..k).all(|j| l.frame.skip.contains(&j) || !defs_uses(&l.insns[j]).0.contains(&gpr(r)));
                untouched && l.rd_in[b][gpr(r) as usize] == vec![ENTRY] && l.entry_vals.contains_key(&gpr(r))
            };
            let max = (3..=10u8).rev().find(|r| live(*r, self));
            args = match max {
                Some(m) => (3..=m).map(|r| self.get(st, gpr(r))).collect(),
                None => vec![],
            };
            args.extend(fl.iter().map(|&r| self.get(st, r)));
        }
        let call = Expr::Call { callee, args, ret: ret.clone() };
        self.clobber(st);
        if matches!(ret, Type::Void) {
            st.out.push(Stmt::Expr(call));
        } else {
            let rreg = if is_float(&types::resolve(self.db, &ret)) { fpr(1) } else { gpr(3) };
            let call = if let Type::Ref(_) = strip_cv(&ret) { Expr::AddrOf(Box::new(call)) } else { call };
            self.def(st, k, rreg, call);
        }
    }

    /// Return blocks that only hold the epilogue, reached by jumps/falls only, with a value: each
    /// predecessor's value is returned there instead of a variable merging them. Not for two
    /// constant arms (`li r3,0` / `li r3,1`: a ternary/if-else assignment's layout).
    fn find_split_returns(&mut self) {
        let Some(rr) = self.ret_reg else { return };
        for b in 0..self.cfg.blocks.len() {
            let blk = &self.cfg.blocks[b];
            if !matches!(blk.term, Term::Return) || blk.preds.len() < 2 {
                continue;
            }
            if !(blk.start..blk.end).all(|k| self.frame.skip.contains(&k) || self.insns[k].is_blr()) {
                continue;
            }
            if !blk.preds.iter().all(|&p| matches!(self.cfg.blocks[p].term, Term::Jump(t) | Term::Fall(t) if t == b)) {
                continue;
            }
            // the last definition of the return register in each predecessor
            let last_li = |p: usize, me: &Self| -> bool {
                let pb = &me.cfg.blocks[p];
                (pb.start..pb.end).rev().find(|&k| !me.frame.skip.contains(&k) && defs_uses(&me.insns[k]).0.contains(&rr)).map_or(false, |k| {
                    let i = &me.insns[k];
                    matches!(i.op(), Opcode::Addi) && i.ra() == 0 || matches!(i.op(), Opcode::Lfs)
                })
            };
            if blk.preds.len() == 2 && blk.preds.iter().all(|&p| last_li(p, self)) {
                continue;
            }
            // with a tail shared by several tests (`return false;` laid out once) the tests
            // keep jumping to it; split only to keep a returned incoming parameter (`return
            // this;`) out of a variable merging it with the other results
            let shared_tail = blk.preds.iter().any(|&p| matches!(self.cfg.blocks[p].term, Term::Fall(_)) && self.cfg.blocks[p].preds.len() >= 2);
            if shared_tail {
                let k = (blk.start..blk.end).find(|&k| self.insns[k].is_blr()).unwrap_or(blk.start);
                let _ = k;
                let entry_reaches = self.rd_in[b][rr as usize].contains(&ENTRY);
                if !entry_reaches {
                    continue;
                }
            }
            self.split_returns.insert(b);
        }
    }

    /// A 64-bit value produced at instruction `k` into r3:r4: one temp, its halves in the
    /// registers.
    fn def_wide(&mut self, st: &mut St, k: usize, e: Expr) {
        let r3u = self.use_count.get(&(k as u32, gpr(3))).copied().unwrap_or(0);
        let r4u = self.use_count.get(&(k as u32, gpr(4))).copied().unwrap_or(0);
        if r3u == 0 && r4u == 0 && !self.web_var.contains_key(&(k as u32, gpr(3))) && !self.web_var.contains_key(&(k as u32, gpr(4))) {
            st.out.push(Stmt::Expr(e));
            return;
        }
        let ty = match ty_of(&e, &self.vars) {
            t if crate::wide::is_wide(&t) => t,
            _ => Type::Int { size: 8, signed: true },
        };
        let t = self.new_var("temp_r3".into(), ty, VarKind::Local, true);
        st.out.push(Stmt::Assign { dst: Expr::Var(t), src: e.clone() });
        self.temp_def.insert(t, e);
        self.def(st, k, gpr(3), crate::wide::hi32(Expr::Var(t)));
        self.def(st, k, gpr(4), crate::wide::lo32(Expr::Var(t)));
    }

    fn ret_value(&mut self, st: &St, r: Reg) -> Expr {
        if r == gpr(3) && crate::wide::is_wide(&types::resolve(self.db, &self.ret_ty)) {
            let signed = !matches!(strip_cv(&types::resolve(self.db, &self.ret_ty)), Type::Int { signed: false, .. });
            let (h, l) = (self.get(st, gpr(3)), self.get(st, gpr(4)));
            return self.pair_of(h, l, signed);
        }
        // an undeclared return: r3:r4 holding the halves of one computed 64-bit value
        if r == gpr(3) && self.param_home_slots && (sig::ret_unknown(&self.sig) || matches!(self.sig.ret, Type::Unknown { .. })) && matches!(self.ret_ty, Type::Unknown { size: 0 | 4 }) {
            let (h, l) = (self.get(st, gpr(3)), self.get(st, gpr(4)));
            let (hr, lr) = (self.resolve_temps(&h), self.resolve_temps(&l));
            // halves recognizably taken from one 64-bit value (not a garbage r4)
            let recognized = crate::wide::as_hi(&hr, &self.vars).is_some() && !lr.any_unknown();
            if recognized {
                if let Some(x) = self.pair_resolved(h, l) {
                    let xt = types::ty_of(&x, &self.vars);
                    if crate::wide::is_wide(&xt) {
                        self.ret_ty = xt;
                        return x;
                    }
                }
            }
        }
        self.get(st, r)
    }

    /// A destructor call with a positive flag in r4 is `delete obj` (MWCC passes 1; -1 is an
    /// explicit `obj->~T()`, 0 a member/base subobject). The flag becomes a trailing `1` argument
    /// that the emitter renders as a delete-expression.
    fn deleting_dtor_call(&self, st: &St, k: usize, s: &FuncSig) -> bool {
        sig::is_dtor(s)
            && s.params.is_empty()
            && self.reg_set_for_call(k, gpr(4))
            && matches!(&st.regs[gpr(4) as usize], Some(Expr::Int { value, .. }) if *value > 0)
    }

    fn clobber(&mut self, st: &mut St) {
        for r in call_clobbers() {
            st.regs[r as usize] = None;
            st.site[r as usize] = None;
        }
        for q in [0usize, 1, 5, 6, 7] {
            for z in 0..4 {
                st.cr[q * 4 + z] = None;
            }
        }
        st.ca = Ca::Unknown;
    }

    fn load_ty(op: Opcode) -> Type {
        use Opcode::*;
        match op {
            Lwz | Lwzu | Lwzx | Lwzux => t_unk(4),
            Lhz | Lhzu | Lhzx | Lhzux => t_int(2, false),
            Lha | Lhau | Lhax | Lhaux => t_int(2, true),
            Lbz | Lbzu | Lbzx | Lbzux => t_int(1, false),
            Lfs | Lfsu | Lfsx | Lfsux => t_f32(),
            Lfd | Lfdu | Lfdx | Lfdux => t_f64(),
            _ => t_unk(4),
        }
    }

    fn store_ty(op: Opcode) -> Type {
        use Opcode::*;
        match op {
            Stw | Stwu | Stwx | Stwux => t_unk(4),
            Sth | Sthu | Sthx | Sthux => t_int(2, false),
            Stb | Stbu | Stbx | Stbux => t_int(1, false),
            Stfs | Stfsu | Stfsx | Stfsux => t_f32(),
            Stfd | Stfdu | Stfdx | Stfdux => t_f64(),
            _ => t_unk(4),
        }
    }

    /// Address operand (rA + d or symbol) for a D-form access; returns lvalue.
    fn d_lvalue(&mut self, st: &mut St, i: &Insn, ty: Type) -> Expr {
        let ra = i.ra();
        if ra == 1 && i.reloc.is_none() {
            if let Some(v) = self.in_stack.get(&i.disp()) {
                return v.clone();
            }
        }
        if let Some(r) = &i.reloc {
            let (sym, add) = (r.target.clone(), r.addend);
            match r.kind {
                RelocKind::EmbSda21 => return self.sym_lvalue(&sym, add, ty),
                RelocKind::Addr16Lo => return self.sym_lvalue(&sym, add, ty),
                _ => {}
            }
        }
        if ra == 1 {
            let off = i.disp();
            if let Some(lv) = self.stack_lvalue(off, ty.clone()) {
                return lv;
            }
            return Expr::Unknown { text: format!("stack 0x{off:x}"), ty };
        }
        let base = self.gpr_or_zero(st, ra);
        self.mem(base, i.disp(), ty)
    }

    /// Update-form access: (lvalue, new value of rA).
    fn update_access(&mut self, st: &mut St, i: &Insn, ty: Type) -> (Expr, Expr) {
        if let Some(r) = i.reloc.clone() {
            if matches!(r.kind, RelocKind::Addr16Lo | RelocKind::EmbSda21) {
                let lv = self.sym_lvalue(&r.target, r.addend, ty);
                let na = self.sym_addr(&r.target, r.addend);
                return (lv, na);
            }
        }
        let base = self.get(st, gpr(i.ra()));
        let lv = self.mem(base.clone(), i.disp(), ty);
        let na = self.addr_add(base, i.disp() as i64);
        (lv, na)
    }

    fn x_lvalue(&mut self, st: &mut St, i: &Insn, ty: Type) -> Expr {
        let base = self.gpr_or_zero(st, i.ra());
        let idx = self.get(st, gpr(i.rb()));
        let esz = scalar_size(&ty).unwrap_or(1);
        // idx = x << log2(esz) -> base[x]
        let idx_r = match &idx {
            Expr::Var(v) => self.temp_def.get(v).cloned().unwrap_or(idx.clone()),
            e => e.clone(),
        };
        if let Expr::Binary { op: BinOp::Shl, l, r, .. } = &idx_r {
            if let Some(s) = r.as_int() {
                if esz > 1 && (1u32 << s) == esz {
                    return Expr::Index { base: Box::new(base), index: l.clone(), ty };
                }
            }
        }
        if let Expr::Binary { op: BinOp::Mul, l, r, .. } = &idx_r {
            if r.as_int() == Some(esz as i64) {
                return Expr::Index { base: Box::new(base), index: l.clone(), ty };
            }
        }
        if let Some(c) = idx.as_int() {
            return self.mem(base, c as i32, ty);
        }
        let bp = Expr::bin(BinOp::Add, Expr::cast(t_ptr(t_int(1, false)), base), idx, t_ptr(t_int(1, false)));
        Expr::Load { base: Box::new(bp), offset: 0, ty }
    }

    fn step(&mut self, st: &mut St, k: usize, i: &Insn) {
        use Opcode::*;
        let ins = i.ins;
        match ins.op {
            // ---------------- loads
            Lwz | Lhz | Lha | Lbz => {
                let ty = Self::load_ty(ins.op);
                if i.ra() == 1 && self.stack.conv.contains(&i.disp()) {
                    let v = self.conv_read(st, i.disp(), &ty);
                    self.def(st, k, gpr(i.rd()), v);
                    return;
                }
                let lv = self.d_lvalue(st, i, ty);
                self.def(st, k, gpr(i.rd()), lv);
            }
            Lwzu | Lhzu | Lhau | Lbzu => {
                let ty = Self::load_ty(ins.op);
                let (lv, na) = self.update_access(st, i, ty);
                self.def(st, k, gpr(i.rd()), lv);
                self.def(st, k, gpr(i.ra()), na);
            }
            Lwzx | Lhzx | Lhax | Lbzx => {
                let ty = Self::load_ty(ins.op);
                let lv = self.x_lvalue(st, i, ty);
                self.def(st, k, gpr(i.rd()), lv);
            }
            Lfs | Lfd => {
                let ty = Self::load_ty(ins.op);
                if let Some(r) = &i.reloc {
                    let (sym, add) = (r.target.clone(), r.addend);
                    if let Some(c) = self.literal_load(&sym, add, &ty) {
                        self.def(st, k, fpr(ins.field_frd()), c);
                        return;
                    }
                } else if i.ra() != 1 && i.ra() != 0 {
                    // a literal read through its address in a register (`addi rX, rY, lit@l;
                    // lfs f0, 0(rX)`, the scheduler's choice for float constants): the literal
                    let base = self.get(st, gpr(i.ra()));
                    if let Some((sym, add)) = self.literal_addr(&base, 0) {
                        if let Some(c) = self.literal_load(&sym, add + i.disp() as i64, &ty) {
                            self.def(st, k, fpr(ins.field_frd()), c);
                            return;
                        }
                    }
                }
                if i.ra() == 1 && self.stack.conv.contains(&i.disp()) {
                    let v = self.conv_read(st, i.disp(), &ty);
                    self.def(st, k, fpr(ins.field_frd()), v);
                    return;
                }
                let lv = self.d_lvalue(st, i, ty);
                self.def(st, k, fpr(ins.field_frd()), lv);
            }
            Lfsu | Lfdu => {
                let ty = Self::load_ty(ins.op);
                let mut lit = None;
                if let Some(r) = &i.reloc {
                    let (sym, add) = (r.target.clone(), r.addend);
                    lit = self.literal_load(&sym, add, &ty);
                }
                let (lv, na) = self.update_access(st, i, ty);
                self.def(st, k, fpr(ins.field_frd()), lit.unwrap_or(lv));
                self.def(st, k, gpr(i.ra()), na);
            }
            Lfsx | Lfdx => {
                let ty = Self::load_ty(ins.op);
                let lv = self.x_lvalue(st, i, ty);
                self.def(st, k, fpr(ins.field_frd()), lv);
            }
            PsqL => {
                let w = ins.field_ps_w();
                let q = ins.field_ps_i();
                let off = ins.field_ps_offset() as i32;
                if i.ra() == 1 && self.stack.conv.contains(&off) {
                    // int -> float through a quantized load of a value stored for it (as wide as
                    // the quantized type: a wider store is an object's bytes read back)
                    let qsize = quant_type(q).and_then(|t| scalar_size(&t));
                    if let Some((sz, v)) = st.mem.get(&off).cloned().filter(|(sz, _)| qsize.map_or(true, |q| q == *sz)) {
                        let _ = sz;
                        self.def(st, k, fpr(ins.field_frd()), Expr::cast(t_f32(), v));
                        return;
                    }
                    if quant_type(q).is_none() || w != 1 {
                        self.def(st, k, fpr(ins.field_frd()), Expr::cast(t_f32(), Expr::Unknown { text: "psq".into(), ty: t_unk(4) }));
                        return;
                    }
                }
                // a scalar load through the runtime's quantization registers (qr2..qr5 = u8, u16,
                // s8, s16; scale 0): an integer member read as a float
                if w == 1 {
                    if let Some(qt) = quant_type(q) {
                        let lv = if i.ra() == 1 {
                            match self.stack_lvalue(off, qt.clone()) {
                                Some(lv) => lv,
                                None => Expr::Unknown { text: "psq".into(), ty: qt.clone() },
                            }
                        } else {
                            let base = self.gpr_or_zero(st, i.ra());
                            self.mem(base, off, qt)
                        };
                        // the header's fast-cast inline (`CCast::ToReal32(const uchar&)`: `psq_l r,
                        // 0(in), 1, 2`) when the context declares it; a C cast is the int->float magic
                        let fast = match q {
                            2 => Some(("CCast::ToReal32", "ToReal32__5CCastFRCUc")),
                            5 => Some(("CCast::StoF", "StoF__5CCastFRCs")),
                            _ => None,
                        };
                        if let (Some((qn, sym)), Some(db)) = (fast, self.db) {
                            if !matches!(lv, Expr::Unknown { .. }) && db.decls.contains_key(qn) {
                                let sig = sig::sig_of(sym, Some(db));
                                let call = Expr::Call { callee: Callee::Direct { symbol: sym.to_string(), sig }, args: vec![Expr::AddrOf(Box::new(lv))], ret: t_f32() };
                                self.def(st, k, fpr(ins.field_frd()), call);
                                return;
                            }
                        }
                        self.def(st, k, fpr(ins.field_frd()), Expr::cast(t_f32(), lv));
                        return;
                    }
                }
                let ty = if w == 1 && q == 0 { t_f32() } else { t_unk(8) };
                let base = self.gpr_or_zero(st, i.ra());
                let lv = self.mem(base, off, ty);
                self.def(st, k, fpr(ins.field_frd()), lv);
            }
            // ---------------- stores
            Stw | Sth | Stb | Stfs | Stfd => {
                let ty = Self::store_ty(ins.op);
                let src = if matches!(ins.op, Stfs | Stfd) { self.get(st, fpr(ins.field_frs())) } else { self.get(st, gpr(i.rs())) };
                if i.ra() == 1 && i.reloc.is_none() && (self.stack.conv.contains(&i.disp()) || self.out_stack.contains(&i.disp())) {
                    let sz = scalar_size(&ty).unwrap_or(4);
                    st.mem.insert(i.disp(), (sz, src));
                    return;
                }
                let src = narrow_store(src, &ty);
                let ty = const_store_ty(ty, &src);
                let lv = self.d_lvalue(st, i, ty);
                self.store(st, lv, src);
            }
            Stwu | Sthu | Stbu | Stfsu | Stfdu => {
                let ty = Self::store_ty(ins.op);
                let src = if matches!(ins.op, Stfsu | Stfdu) { self.get(st, fpr(ins.field_frs())) } else { self.get(st, gpr(i.rs())) };
                let (lv, na) = self.update_access(st, i, ty);
                self.store(st, lv, src);
                self.def(st, k, gpr(i.ra()), na);
            }
            Stwx | Sthx | Stbx | Stfsx | Stfdx => {
                let ty = Self::store_ty(ins.op);
                let src = if matches!(ins.op, Stfsx | Stfdx) { self.get(st, fpr(ins.field_frs())) } else { self.get(st, gpr(i.rs())) };
                let ty = const_store_ty(ty, &src);
                let lv = self.x_lvalue(st, i, ty);
                self.store(st, lv, src);
            }
            Stfiwx => {
                let src = self.get(st, fpr(ins.field_frs()));
                let lv = self.x_lvalue(st, i, t_s32());
                self.store(st, lv, src);
            }
            PsqSt => {
                let off = ins.field_ps_offset() as i32;
                let src = self.get(st, fpr(ins.field_frs()));
                if i.ra() == 1 && self.stack.conv.contains(&off) {
                    st.mem.insert(off, (4, src));
                    return;
                }
                let w = ins.field_ps_w();
                let base = self.gpr_or_zero(st, i.ra());
                if w == 0 && ins.field_ps_i() == 0 {
                    // a pair of floats: copy both halves (a psq_l pair) or splat a scalar
                    let (a, b) = match &src {
                        Expr::Load { base: sb, offset: so, ty: Type::Unknown { size: 8 } } => (
                            self.mem((**sb).clone(), *so, t_f32()),
                            self.mem((**sb).clone(), *so + 4, t_f32()),
                        ),
                        Expr::Member { base: sb, offset: so, ty: Type::Unknown { size: 8 } } => (
                            self.member((**sb).clone(), *so, t_f32()),
                            self.member((**sb).clone(), *so + 4, t_f32()),
                        ),
                        other => (other.clone(), other.clone()),
                    };
                    let lv0 = self.mem(base.clone(), off, t_f32());
                    self.store(st, lv0, a);
                    let lv1 = self.mem(base, off + 4, t_f32());
                    self.store(st, lv1, b);
                    return;
                }
                let ty = if w == 1 && ins.field_ps_i() == 0 { t_f32() } else { t_unk(8) };
                let lv = self.mem(base, off, ty);
                self.store(st, lv, src);
            }
            // ---------------- integer arithmetic
            Addi | Addis => {
                let imm = if ins.op == Addis { (i.simm() as i64) << 16 } else { i.simm() as i64 };
                let rd = gpr(i.rd());
                if let Some(r) = i.reloc.clone() {
                    match r.kind {
                        RelocKind::EmbSda21 | RelocKind::Addr16Lo => {
                            let a = self.sym_addr(&r.target, r.addend);
                            self.def(st, k, rd, a);
                            return;
                        }
                        RelocKind::Addr16Ha | RelocKind::Addr16Hi => {
                            let e = Expr::Unknown { text: format!("ha:{}", r.target), ty: t_unk(4) };
                            self.def(st, k, rd, e);
                            return;
                        }
                        _ => {}
                    }
                }
                if i.ra() == 0 {
                    let v = imm as i32 as i64;
                    self.def(st, k, rd, Expr::int(v));
                    return;
                }
                if i.ra() == 1 {
                    let a = self.stack_addr(imm as i32);
                    self.def(st, k, rd, a);
                    return;
                }
                let a = self.get(st, gpr(i.ra()));
                let v = self.addr_add(a, imm);
                self.def(st, k, rd, v);
            }
            Addic | Addic_ => {
                let imm = i.simm() as i64;
                // `addic. rD, r1, k`: a stack object's address, tested (inlined placement new)
                let (a, v) = if i.ra() == 1 {
                    let v = self.stack_addr(imm as i32);
                    (v.clone(), v)
                } else {
                    let a = self.get(st, gpr(i.ra()));
                    let v = arith(BinOp::Add, a.clone(), Expr::int(imm), &self.vars);
                    (a, v)
                };
                st.ca = if imm == -1 { Ca::Ne0(a.clone()) } else { Ca::Unknown };
                self.def(st, k, gpr(i.rd()), v);
                if ins.op == Addic_ {
                    self.record(st, k, gpr(i.rd()));
                }
            }
            Subfic => {
                let a = self.get(st, gpr(i.ra()));
                let imm = i.simm() as i64;
                st.ca = if imm == 0 { Ca::Eq0(a.clone()) } else { Ca::Unknown };
                let v = arith(BinOp::Sub, Expr::int(imm), a, &self.vars);
                self.def(st, k, gpr(i.rd()), v);
            }
            Mulli => {
                let a = self.get(st, gpr(i.ra()));
                let v = arith(BinOp::Mul, a, Expr::int(i.simm() as i64), &self.vars);
                self.def(st, k, gpr(i.rd()), v);
            }
            Add | Subf | Mullw | Divw | Divwu | Addc | Subfc => {
                let a = self.get(st, gpr(i.ra()));
                let b = self.get(st, gpr(i.rb()));
                let v = match ins.op {
                    Add | Addc => {
                        if is_ptr(&ty_of(&a, &self.vars)) && !is_ptr(&ty_of(&b, &self.vars)) {
                            byte_add(a, b)
                        } else if is_ptr(&ty_of(&b, &self.vars)) && !is_ptr(&ty_of(&a, &self.vars)) {
                            byte_add(b, a)
                        } else {
                            arith(BinOp::Add, a, b, &self.vars)
                        }
                    }
                    Subf | Subfc => arith(BinOp::Sub, b, a, &self.vars),
                    Mullw => arith(BinOp::Mul, a, b, &self.vars),
                    Divw => arith(BinOp::Div, as_signed(a, &self.vars), as_signed(b, &self.vars), &self.vars),
                    _ => arith(BinOp::Div, as_unsigned(a, &self.vars), as_unsigned(b, &self.vars), &self.vars),
                };
                st.ca = match ins.op {
                    Subfc => Ca::Subfc(self.get(st, gpr(i.ra())), self.get(st, gpr(i.rb()))),
                    Addc => Ca::Addc(self.get(st, gpr(i.ra())), self.get(st, gpr(i.rb()))),
                    _ => std::mem::replace(&mut st.ca, Ca::Unknown),
                };
                self.def(st, k, gpr(i.rd()), v);
                if i.rc() {
                    self.record(st, k, gpr(i.rd()));
                }
            }
            Mulhw | Mulhwu => {
                let a = self.get(st, gpr(i.ra()));
                let b = self.get(st, gpr(i.rb()));
                // high word of the product (a division by a constant, folded by `divmagic`)
                let v = crate::divmagic::mulh(a, b, ins.op == Mulhw);
                self.def(st, k, gpr(i.rd()), v);
            }
            Adde | Subfe | Addze | Addme | Subfze | Subfme => {
                let a = self.get(st, gpr(i.ra()));
                let ca = std::mem::replace(&mut st.ca, Ca::Unknown);
                let v = match (ins.op, &ca) {
                    (Subfe, Ca::Ne0(x)) => {
                        // ~(x-1) + x + CA == CA when ra == x - 1, rb == x
                        let b = self.get(st, gpr(i.rb()));
                        if b == *x {
                            Expr::cmp(BinOp::Ne, x.clone(), Expr::int(0))
                        } else {
                            Expr::Unknown { text: "subfe".into(), ty: t_unk(4) }
                        }
                    }
                    (Adde, Ca::Eq0(x)) => {
                        let b = self.get(st, gpr(i.rb()));
                        if b == *x || a == *x {
                            Expr::cmp(BinOp::Eq, x.clone(), Expr::int(0))
                        } else {
                            Expr::Unknown { text: "adde".into(), ty: t_unk(4) }
                        }
                    }
                    (Adde, Ca::Subfc(x, y)) => {
                        // srawi b,31 + srwi a,31 + CA(b >=u a)  ==  a <= b (signed)
                        let b2 = self.get(st, gpr(i.rb()));
                        let look = |e: &Expr, l: &Self| match e {
                            Expr::Var(v) => l.temp_def.get(v).cloned().unwrap_or(e.clone()),
                            e => e.clone(),
                        };
                        let (ra, rb) = (look(&a, self), look(&b2, self));
                        let is_shr = |e: &Expr, of: &Expr| -> bool {
                            match e {
                                Expr::Binary { op: BinOp::Shr, l, r, .. } => {
                                    r.as_int() == Some(31) && {
                                        let inner = match &**l {
                                            Expr::Cast { e, .. } => &**e,
                                            o => o,
                                        };
                                        inner == of || matches!(of, Expr::Cast { e, .. } if &**e == inner)
                                    }
                                }
                                _ => false,
                            }
                        };
                        if (is_shr(&ra, y) && is_shr(&rb, x)) || (is_shr(&rb, y) && is_shr(&ra, x)) {
                            Expr::cmp(BinOp::Le, as_signed(x.clone(), &self.vars), as_signed(y.clone(), &self.vars))
                        } else {
                            Expr::Unknown { text: "adde".into(), ty: t_unk(4) }
                        }
                    }
                    (Addze, Ca::Srawi(x, n)) => arith(BinOp::Div, as_signed(x.clone(), &self.vars), Expr::int(1i64 << n), &self.vars),
                    // high word of a 64-bit addition / subtraction: the halves of two 64-bit values
                    (Adde, Ca::Addc(alo, blo)) | (Subfe, Ca::Subfc(alo, blo)) if self.param_home_slots => {
                        let b2 = self.get(st, gpr(i.rb()));
                        let (x, y) = (self.pair_resolved(a.clone(), alo.clone()), self.pair_resolved(b2, blo.clone()));
                        match (x, y) {
                            (Some(x), Some(y)) => {
                                let wt = Type::Int { size: 8, signed: true };
                                let s = if ins.op == Adde { Expr::bin(BinOp::Add, x, y, wt.clone()) } else { Expr::bin(BinOp::Sub, y, x, wt.clone()) };
                                // the 64-bit result is computed here (later uses must not
                                // re-evaluate its operands)
                                let t = self.new_var(format!("temp_{}", reg_name(gpr(i.rd()))), wt, VarKind::Local, true);
                                st.out.push(Stmt::Assign { dst: Expr::Var(t), src: s.clone() });
                                self.temp_def.insert(t, s);
                                crate::wide::hi32(Expr::Var(t))
                            }
                            _ => {
                                self.warn(format!("carry op {} at {:#x}", i.text(), i.off));
                                Expr::Unknown { text: i.text(), ty: t_unk(4) }
                            }
                        }
                    }
                    _ => {
                        self.warn(format!("carry op {} at {:#x}", i.text(), i.off));
                        Expr::Unknown { text: i.text(), ty: t_unk(4) }
                    }
                };
                self.def(st, k, gpr(i.rd()), v);
                if i.rc() {
                    self.record(st, k, gpr(i.rd()));
                }
            }
            Neg => {
                let a = self.get(st, gpr(i.ra()));
                let v = Expr::Unary { op: UnOp::Neg, ty: int_ty(&a, &self.vars), e: Box::new(a) };
                self.def(st, k, gpr(i.rd()), v);
                if i.rc() {
                    self.record(st, k, gpr(i.rd()));
                }
            }
            // ---------------- logical
            Or | And | Xor | Andc | Orc | Nor | Nand | Eqv | Slw | Srw | Sraw => {
                let s = self.get(st, gpr(i.rs()));
                let b = self.get(st, gpr(i.rb()));
                let v = if ins.op == Or && i.rs() == i.rb() {
                    s
                } else {
                    match ins.op {
                        Or => arith(BinOp::Or, s, b, &self.vars),
                        And => arith(BinOp::And, s, b, &self.vars),
                        Xor => arith(BinOp::Xor, s, b, &self.vars),
                        Andc => arith(BinOp::And, s, bitnot(b, &self.vars), &self.vars),
                        Orc => arith(BinOp::Or, s, bitnot(b, &self.vars), &self.vars),
                        Nor => bitnot(arith(BinOp::Or, s, b, &self.vars), &self.vars),
                        Nand => bitnot(arith(BinOp::And, s, b, &self.vars), &self.vars),
                        Eqv => bitnot(arith(BinOp::Xor, s, b, &self.vars), &self.vars),
                        Slw => arith(BinOp::Shl, s, b, &self.vars),
                        Srw => shr_kind(arith(BinOp::Shr, as_unsigned(s, &self.vars), b, &self.vars), false),
                        _ => {
                            st.ca = Ca::Unknown;
                            shr_kind(arith(BinOp::Shr, as_signed(s, &self.vars), b, &self.vars), true)
                        }
                    }
                };
                self.def(st, k, gpr(i.ra()), v);
                if i.rc() {
                    self.record(st, k, gpr(i.ra()));
                }
            }
            Ori | Oris | Xori | Xoris | Andi_ | Andis_ => {
                let s = self.get(st, gpr(i.rs()));
                let imm = i.uimm() as i64;
                let imm = if matches!(ins.op, Oris | Xoris | Andis_) { imm << 16 } else { imm };
                let v = match ins.op {
                    Ori | Oris => {
                        if imm == 0 {
                            s
                        } else {
                            arith(BinOp::Or, s, Expr::uint(imm), &self.vars)
                        }
                    }
                    Xori | Xoris => arith(BinOp::Xor, s, Expr::uint(imm), &self.vars),
                    _ => arith(BinOp::And, s, Expr::uint(imm), &self.vars),
                };
                self.def(st, k, gpr(i.ra()), v);
                if matches!(ins.op, Andi_ | Andis_) {
                    self.record(st, k, gpr(i.ra()));
                }
            }
            Srawi => {
                let s = self.get(st, gpr(i.rs()));
                let n = ins.field_sh();
                st.ca = Ca::Srawi(s.clone(), n);
                let v = shr_kind(arith(BinOp::Shr, as_signed(s, &self.vars), Expr::int(n as i64), &self.vars), true);
                self.def(st, k, gpr(i.ra()), v);
                if i.rc() {
                    self.record(st, k, gpr(i.ra()));
                }
            }
            Extsb | Extsh => {
                let s = self.get(st, gpr(i.rs()));
                let t = if ins.op == Extsb { t_int(1, true) } else { t_int(2, true) };
                let v = Expr::cast(t, s);
                self.def(st, k, gpr(i.ra()), v);
                if i.rc() {
                    self.record(st, k, gpr(i.ra()));
                }
            }
            Cntlzw => {
                let s = self.get(st, gpr(i.rs()));
                let v = Expr::Call {
                    callee: Callee::Direct { symbol: "__cntlzw".into(), sig: builtin_sig("__cntlzw", 1) },
                    args: vec![s],
                    ret: t_s32(),
                };
                self.def(st, k, gpr(i.ra()), v);
            }
            Rlwinm => {
                let s = self.get(st, gpr(i.rs()));
                let v = self.rlwinm(s, ins.field_sh(), ins.field_mb(), ins.field_me());
                if self.cfg.blocks[self.cfg.block_of[k]].term.is_switch() && ins.field_sh() == 2 && ins.field_me() == 29 {
                    let s = self.get(st, gpr(i.rs()));
                    self.switch_index.insert(self.cfg.block_of[k], s);
                }
                self.def(st, k, gpr(i.ra()), v);
                if i.rc() {
                    self.record(st, k, gpr(i.ra()));
                }
            }
            Rlwimi => {
                let s = self.get(st, gpr(i.rs()));
                let a = self.get(st, gpr(i.ra()));
                let (mut sh, mb, me) = (ins.field_sh(), ins.field_mb(), ins.field_me());
                // inserting one CR bit copied out with `mfcr` (a compare stored into a one-bit
                // field): the compare's value shifted into place
                let s = match &s {
                    Expr::Unknown { text, .. } if mb == me => match text.strip_prefix("mfcr:").and_then(|x| x.parse::<usize>().ok()) {
                        Some(k) => match self.mfcr_vals.get(&k).and_then(|c| c[(mb as usize + sh as usize) % 32].clone()) {
                            Some(b) => {
                                sh = 31 - mb;
                                b
                            }
                            None => s,
                        },
                        None => s,
                    },
                    _ => s,
                };
                // inserting a compare result (`subf; cntlzw` rotated so its bit 5 lands in a
                // one-bit field): the value is `__cntlzw(x) >> 5` (`a == b`) shifted into place
                let is_clz = |e: &Expr| matches!(e, Expr::Call { callee: Callee::Direct { symbol, .. }, .. } if symbol == "__cntlzw");
                let clz = is_clz(&s) || matches!(&s, Expr::Var(t) if self.temp_def.get(t).is_some_and(|d| is_clz(d)));
                let s = match &s {
                    _ if clz && mb == me && (sh as u32 + 5) % 32 == 31 - mb as u32 => {
                        sh = 31 - mb;
                        arith(BinOp::Shr, s, Expr::int(5), &self.vars)
                    }
                    _ => s,
                };
                let konst = matches!(&s, Expr::Int { .. }) || matches!(&s, Expr::Cast { e, .. } if matches!(**e, Expr::Int { .. }));
                let m = mask32(mb, me);
                let base_konst = matches!(&a, Expr::Int { .. }) || matches!(&a, Expr::Cast { e, .. } if matches!(**e, Expr::Int { .. }));
                // the SDK compiler's `t = (x >> k) & m; __rlwimi(t, y, ..)`: a field extracted into
                // a variable that is then inserted into (`GXSetCullMode`'s hardware mode)
                let extracted = self.param_home_slots && {
                    let ar = self.resolve_temps(&a);
                    let e = match &ar {
                        Expr::Cast { e, .. } => &**e,
                        e => e,
                    };
                    matches!(e, Expr::Binary { op: BinOp::And, l, r, .. } if r.as_int().is_some() && matches!(&**l, Expr::Binary { op: BinOp::Shr, r: k, .. } if k.as_int().is_some_and(|k| k > 0)))
                        // a register word started as a shifted value (`reg = (0xE0 + id * 2) << 24;
                        // SET_REG_FIELD(reg, ..)`)
                        || matches!(e, Expr::Binary { op: BinOp::Shl, r: k, .. } if k.as_int().is_some_and(|k| k >= 16))
                };
                let v = if extracted || konst || base_konst || !known_zero(&a, m, &self.vars, &|t| self.temp_def.get(&t), 0) {
                    // inserting a constant or into a constant (C shift/mask forms would fold to a
                    // plain and/or), or into
                    // a value whose target bits aren't known clear (the older compiler clears them
                    // with a separate `rlwinm` for C forms): the source used the intrinsic
                    // (`SET_REG_FIELD` in the SDK headers). Bitfield recovery turns read-modify-writes
                    // of declared bitfields back into assignments.
                    Expr::Call {
                        callee: Callee::Direct { symbol: "__rlwimi".into(), sig: builtin_sig("__rlwimi", 5) },
                        args: vec![a, s, Expr::int(sh as i64), Expr::int(mb as i64), Expr::int(me as i64)],
                        ret: t_u32(),
                    }
                } else {
                    let ins_part = self.rlwinm(s, sh, mb, me);
                    // (the target bits are known clear here: no mask to keep the rest)
                    let keep = a;
                    // operand order decides which value the compiler builds in place: the right
                    // operand of `|` becomes the insert target (C functions: SDK code built by
                    // the older compiler, which always takes it), allocated r0 when it can be
                    let c_linkage = sig::demangle(sig::strip_dtk_suffix(&self.f.name)).is_none();
                    // (a field inserted into a single shifted/masked value, `a | (b << 8)`: the
                    // compiler builds the shifted right operand and inserts the left one)
                    let leaf = |e: &Expr| {
                        let e = match e {
                            Expr::Var(t) => self.temp_def.get(t).unwrap_or(e),
                            e => e,
                        };
                        matches!(e, Expr::Binary { op: BinOp::Shl | BinOp::And, l, .. } if !matches!(&**l, Expr::Binary { op: BinOp::Or, .. }))
                    };
                    if c_linkage || (i.ra() == 0 && i.rs() != 0) || leaf(&keep) {
                        arith(BinOp::Or, ins_part, keep, &self.vars)
                    } else {
                        arith(BinOp::Or, keep, ins_part, &self.vars)
                    }
                };
                self.def(st, k, gpr(i.ra()), v);
                if i.rc() {
                    self.record(st, k, gpr(i.ra()));
                }
            }
            Rlwnm => {
                let s = self.get(st, gpr(i.rs()));
                let b = self.get(st, gpr(i.rb()));
                let m = mask32(ins.field_mb(), ins.field_me());
                let rot = arith(
                    BinOp::Or,
                    arith(BinOp::Shl, s.clone(), b.clone(), &self.vars),
                    arith(BinOp::Shr, as_unsigned(s, &self.vars), arith(BinOp::Sub, Expr::int(32), b, &self.vars), &self.vars),
                    &self.vars,
                );
                let v = arith(BinOp::And, rot, Expr::uint(m as i64), &self.vars);
                self.def(st, k, gpr(i.ra()), v);
            }
            // ---------------- compares
            Cmp | Cmpl | Cmpi | Cmpli => {
                let a = self.get(st, gpr(i.ra()));
                let signed = matches!(ins.op, Cmp | Cmpi);
                let b = match ins.op {
                    Cmpi => Expr::int(i.simm() as i64),
                    Cmpli => Expr::uint(i.uimm() as i64),
                    _ => self.get(st, gpr(i.rb())),
                };
                // explicit signedness on every non-literal operand; the simplifier drops the casts
                // C typing makes redundant once temps are folded
                let mark = |e: Expr, vars: &[Var]| -> Expr {
                    match e {
                        Expr::Int { .. } => if signed { as_signed(e, vars) } else { as_unsigned(e, vars) },
                        Expr::Cast { .. } => if signed { as_signed(e, vars) } else { as_unsigned(e, vars) },
                        e if is_ptr(&ty_of(&e, vars)) && !signed => e,
                        e => Expr::cast(if signed { t_s32() } else { t_u32() }, e),
                    }
                };
                let (a, b) = (mark(a, &self.vars), mark(b, &self.vars));
                let f = ins.field_crfd();
                self.set_cr_cmp(st, f, a, b, false);
                self.save_cr(st, k, f);
            }
            Fcmpu | Fcmpo => {
                let a = self.get(st, fpr(ins.field_fra()));
                let b = self.get(st, fpr(ins.field_frb()));
                let f = ins.field_crfd();
                self.set_cr_cmp(st, f, a, b, true);
                self.save_cr(st, k, f);
            }
            Cror | Crnor | Crand | Crandc | Crxor | Creqv | Crnand | Crorc => {
                let (bd, ba, bb) = (ins.field_crbd() as usize, ins.field_crba() as usize, ins.field_crbb() as usize);
                let unk = || Expr::Unknown { text: "crbit".into(), ty: Type::Bool };
                let a = st.cr[ba].clone().unwrap_or_else(unk);
                let b = st.cr[bb].clone().unwrap_or_else(unk);
                let v = match ins.op {
                    Cror => combine_or(a, b),
                    Crnor => combine_or(a, b).negate(&self.vars),
                    Crand => Expr::cmp(BinOp::LogAnd, a, b),
                    Crandc => Expr::cmp(BinOp::LogAnd, a, b.negate(&self.vars)),
                    Crxor if ba == bb => Expr::Int { value: 0, ty: Type::Bool },
                    Creqv if ba == bb => Expr::Int { value: 1, ty: Type::Bool },
                    Crorc => combine_or(a, b.negate(&self.vars)),
                    Crnand => Expr::cmp(BinOp::LogAnd, a, b).negate(&self.vars),
                    _ => unk(),
                };
                st.cr[bd] = Some(v);
                self.save_cr(st, k, (bd / 4) as u8);
            }
            Mcrf => {
                let (d, s) = (ins.field_crfd() as usize, ins.field_crfs() as usize);
                for q in 0..4 {
                    st.cr[d * 4 + q] = st.cr[s * 4 + q].clone();
                }
                self.save_cr(st, k, d as u8);
            }
            // ---------------- floats
            Fadds | Fsubs | Fdivs | Fadd | Fsub | Fdiv | Fmuls | Fmul => {
                let a = self.get(st, fpr(ins.field_fra()));
                let b = if matches!(ins.op, Fmuls | Fmul) { self.get(st, fpr(ins.field_frc())) } else { self.get(st, fpr(ins.field_frb())) };
                let single = matches!(ins.op, Fadds | Fsubs | Fdivs | Fmuls);
                let ty = if single { t_f32() } else { t_f64() };
                let mut op = match ins.op {
                    Fadds | Fadd => BinOp::Add,
                    Fsubs | Fsub => BinOp::Sub,
                    Fdivs | Fdiv => BinOp::Div,
                    _ => BinOp::Mul,
                };
                // MWCC turns `x / 2^n` into `x * 2^-n` with the constant on the right; a source
                // `x * K` keeps the constant on the left (commutative operands put it first)
                let mut b = b;
                if op == BinOp::Mul && !matches!(a, Expr::Float { .. }) {
                    if let Expr::Float { bits, double } = b {
                        if let Some(r) = pow2_reciprocal(bits, double) {
                            op = BinOp::Div;
                            b = Expr::Float { bits: r, double };
                        }
                    }
                }
                let v = fold_magic(Expr::bin(op, a, b, ty.clone()), &ty);
                self.def(st, k, fpr(ins.field_frd()), v);
            }
            Fmadds | Fmsubs | Fnmadds | Fnmsubs | Fmadd | Fmsub | Fnmadd | Fnmsub => {
                let a = self.get(st, fpr(ins.field_fra()));
                let b = self.get(st, fpr(ins.field_frb()));
                let c = self.get(st, fpr(ins.field_frc()));
                let single = matches!(ins.op, Fmadds | Fmsubs | Fnmadds | Fnmsubs);
                let ty = if single { t_f32() } else { t_f64() };
                let prod = Expr::bin(BinOp::Mul, a, c, ty.clone());
                let v = match ins.op {
                    Fmadds | Fmadd => Expr::bin(BinOp::Add, prod, b, ty.clone()),
                    Fmsubs | Fmsub => Expr::bin(BinOp::Sub, prod, b, ty.clone()),
                    Fnmadds | Fnmadd => Expr::Unary { op: UnOp::Neg, e: Box::new(Expr::bin(BinOp::Add, prod, b, ty.clone())), ty: ty.clone() },
                    _ => Expr::bin(BinOp::Sub, b, prod, ty.clone()),
                };
                self.def(st, k, fpr(ins.field_frd()), v);
            }
            Fmr | PsMr => {
                let b = self.get(st, fpr(ins.field_frb()));
                self.def(st, k, fpr(ins.field_frd()), b);
            }
            Fneg | PsNeg => {
                let b = self.get(st, fpr(ins.field_frb()));
                let ty = ty_of(&b, &self.vars);
                let ty = if is_float(&ty) { ty } else { t_f32() };
                self.def(st, k, fpr(ins.field_frd()), Expr::Unary { op: UnOp::Neg, e: Box::new(b), ty });
            }
            Fabs | Fnabs | PsAbs => {
                let b = self.get(st, fpr(ins.field_frb()));
                let call = Expr::Call {
                    callee: Callee::Direct { symbol: "__fabs".into(), sig: builtin_sig("__fabs", 1) },
                    args: vec![b],
                    ret: t_f64(),
                };
                let v = if ins.op == Fnabs { Expr::Unary { op: UnOp::Neg, e: Box::new(call), ty: t_f64() } } else { call };
                self.def(st, k, fpr(ins.field_frd()), v);
            }
            Frsp => {
                let b = self.get(st, fpr(ins.field_frb()));
                let b_is_var = matches!(b, Expr::Var(_));
                let v = if is_float(&ty_of(&b, &self.vars)) && ty_of(&b, &self.vars) == t_f32() { b } else { Expr::cast(t_f32(), b) };
                // the rounding happens here: `float s = sin(x);` (a cast of a double variable is
                // otherwise propagated to every use, i.e. converted where it's read)
                let d = fpr(ins.field_frd());
                let site = (k as u32, d);
                if matches!(v, Expr::Cast { .. }) && b_is_var && !self.web_var.contains_key(&site) && self.use_count.get(&site).copied().unwrap_or(0) > 0 {
                    let t = self.new_var(format!("temp_{}", reg_name(d)), t_f32(), VarKind::Local, true);
                    st.out.push(Stmt::Assign { dst: Expr::Var(t), src: v.clone() });
                    self.temp_def.insert(t, v);
                    st.site[d as usize] = Some(site);
                    st.regs[d as usize] = Some(Expr::Var(t));
                    self.def_value.insert(site, Expr::Var(t));
                } else {
                    self.def(st, k, d, v);
                }
            }
            Fctiwz | Fctiw => {
                let b = self.get(st, fpr(ins.field_frb()));
                self.def(st, k, fpr(ins.field_frd()), Expr::cast(t_s32(), b));
            }
            Fres | Frsqrte => {
                let b = self.get(st, fpr(ins.field_frb()));
                let name = if ins.op == Fres { "__fres" } else { "__frsqrte" };
                let v = Expr::Call { callee: Callee::Direct { symbol: name.into(), sig: builtin_sig(name, 1) }, args: vec![b], ret: t_f32() };
                self.def(st, k, fpr(ins.field_frd()), v);
            }
            Fsel => {
                let a = self.get(st, fpr(ins.field_fra()));
                let b = self.get(st, fpr(ins.field_frb()));
                let c = self.get(st, fpr(ins.field_frc()));
                let ty = ty_of(&c, &self.vars);
                let v = Expr::Ternary { c: Box::new(Expr::cmp(BinOp::Ge, a, Expr::Float { bits: 0, double: false })), t: Box::new(c), f: Box::new(b), ty };
                self.def(st, k, fpr(ins.field_frd()), v);
            }
            // ---------------- special registers
            Mfcr => {
                self.mfcr_vals.insert(k, st.cr.clone());
                self.def(st, k, gpr(i.rd()), Expr::Unknown { text: format!("mfcr:{k}"), ty: t_unk(4) });
            }
            Mfspr => {
                let v = match ins.field_spr() {
                    9 => self.get(st, CTR),
                    _ => Expr::Unknown { text: format!("mfspr {}", ins.field_spr()), ty: t_unk(4) },
                };
                self.def(st, k, gpr(i.rd()), v);
            }
            Mtspr => {
                let s = self.get(st, gpr(i.rs()));
                match ins.field_spr() {
                    9 => self.def(st, k, CTR, s),
                    8 => self.def(st, k, LR, s),
                    _ => self.warn(format!("mtspr {} at {:#x}", ins.field_spr(), i.off)),
                }
            }
            // ---------------- branches inside blocks
            B if ins.field_lk() => {
                self.do_call(st, k, i, false);
            }
            Bcctr if ins.field_lk() => {
                self.do_vcall(st, k);
            }
            Bclr if i.is_blrl() => {
                // `mtlr rX; blrl`: a call through a function pointer (older compiler)
                st.regs[CTR as usize] = st.regs[LR as usize].clone();
                self.do_vcall(st, k);
            }
            Bcctr => {}
            Bclr | Bc | B => {}
            Sync | Isync | Eieio | Dcbf | Dcbi | Dcbst | Dcbt | Dcbtst | Dcbz | DcbzL | Icbi => {
                self.warn(format!("cache/sync op {} at {:#x}", i.text(), i.off));
            }
            _ => {
                self.warn(format!("unhandled {} at {:#x}", i.text(), i.off));
                let (d, _) = defs_uses(i);
                for r in d {
                    let ty = if r >= 32 && r < 64 { t_f32() } else { t_unk(4) };
                    self.def(st, k, r, Expr::Unknown { text: i.text(), ty });
                }
            }
        }
    }

    fn conv_read(&mut self, st: &St, off: i32, ty: &Type) -> Expr {
        match ty {
            Type::Float { size: 8 } => {
                // int -> double magic: hi word 0x43300000, lo word x (^0x80000000 for signed)
                let hi = st.mem.get(&off).map(|x| x.1.clone());
                let lo = st.mem.get(&(off + 4)).map(|x| x.1.clone());
                match (hi, lo) {
                    (Some(h), Some(l)) if h.as_int().map(|v| v as u32) == Some(0x4330_0000) => {
                        // the xor may have been computed into a temp
                        let lx = match &l {
                            Expr::Var(t) => self.temp_def.get(t).cloned().unwrap_or(l.clone()),
                            _ => l.clone(),
                        };
                        if let Expr::Binary { op: BinOp::Xor, l: x, r, .. } = &lx {
                            if r.as_int().map(|v| v as u32) == Some(0x8000_0000) {
                                return Expr::bin(
                                    BinOp::Add,
                                    Expr::cast(t_f64(), as_signed((**x).clone(), &self.vars)),
                                    Expr::Float { bits: 0x4330_0000_8000_0000, double: true },
                                    t_f64(),
                                );
                            }
                        }
                        Expr::bin(
                            BinOp::Add,
                            Expr::cast(t_f64(), as_unsigned(l, &self.vars)),
                            Expr::Float { bits: 0x4330_0000_0000_0000, double: true },
                            t_f64(),
                        )
                    }
                    (Some(h), _) => h,
                    _ => Expr::Unknown { text: format!("conv 0x{off:x}"), ty: ty.clone() },
                }
            }
            _ => {
                // fctiwz result: stfd at off-4, lwz at off
                if let Some((8, v)) = st.mem.get(&(off - 4)).cloned() {
                    return v;
                }
                if let Some((_, v)) = st.mem.get(&off).cloned() {
                    // float stored quantized (`psq_st f, off, 1, qrN`) and read back as an
                    // integer: a float -> u8/u16/s16 conversion
                    if is_float(&ty_of(&v, &self.vars)) && !is_float(ty) {
                        return Expr::cast(ty.clone(), v);
                    }
                    return v;
                }
                Expr::Unknown { text: format!("conv 0x{off:x}"), ty: ty.clone() }
            }
        }
    }

    fn addr_add(&mut self, a: Expr, imm: i64) -> Expr {
        if imm == 0 {
            return a;
        }
        // string pool + offset: the string at that offset
        if let Expr::Str { bytes } = &a {
            if let Some((sym, add)) = self.str_origin.get(bytes).cloned() {
                if let Some(s) = mwdec_obj::c_string_at(self.obj, &sym, add + imm) {
                    let s = s.to_vec();
                    self.str_origin.insert(s.clone(), (sym, add + imm));
                    return Expr::Str { bytes: s };
                }
            }
        }
        if let Some(c) = a.as_int() {
            return Expr::int((c + imm) as i32 as i64);
        }
        if let Expr::Unknown { text, .. } = &a {
            if text.starts_with("ha:") {
                return Expr::Unknown { text: format!("{text}+{imm}"), ty: t_unk(4) };
            }
        }
        let at = ty_of(&a, &self.vars);
        if is_ptr(&at) || matches!(a, Expr::AddrOf(_)) {
            // pointer + byte offset -> &base->field
            return match a {
                Expr::AddrOf(inner) => match *inner {
                    Expr::Load { base, offset, .. } => Expr::AddrOf(Box::new(Expr::Load { base, offset: offset + imm as i32, ty: t_unk(0) })),
                    Expr::Member { base, offset, .. } => Expr::AddrOf(Box::new(Expr::Member { base, offset: offset + imm as i32, ty: t_unk(0) })),
                    other => Expr::AddrOf(Box::new(Expr::Member { base: Box::new(other), offset: imm as i32, ty: t_unk(0) })),
                },
                a => {
                    let lv = Expr::Load { base: Box::new(a), offset: imm as i32, ty: t_unk(0) };
                    Expr::AddrOf(Box::new(lv))
                }
            };
        }
        arith(BinOp::Add, a, Expr::int(imm), &self.vars)
    }

    fn rlwinm(&self, s: Expr, sh: u8, mb: u8, me: u8) -> Expr {
        let m = mask32(mb, me);
        let v = &self.vars;
        if let Expr::Unknown { text, .. } = &s {
            if let Some(k) = text.strip_prefix("mfcr:").and_then(|x| x.parse::<usize>().ok()) {
                if mb == 31 && me == 31 {
                    let bit = (sh as usize + 31) % 32;
                    if let Some(Some(b)) = self.mfcr_vals.get(&k).map(|c| c[bit].clone()) {
                        return b;
                    }
                }
            }
        }
        if sh == 0 {
            return match m {
                0xFF => Expr::cast(t_int(1, false), s),
                0xFFFF => Expr::cast(t_int(2, false), s),
                0xFFFF_FFFF => s,
                _ => arith(BinOp::And, s, Expr::uint(m as i64), v),
            };
        }
        if mb == 0 && me == 31 - sh {
            return arith(BinOp::Shl, s, Expr::int(sh as i64), v);
        }
        if me == 31 && sh as u32 + mb as u32 == 32 {
            return arith(BinOp::Shr, as_unsigned(s, v), Expr::int(mb as i64), v);
        }
        if me == 31 && (32 - mb as u32) <= sh as u32 {
            // extract: (s >> (32-sh)) & lowmask(32-mb)
            let shifted = arith(BinOp::Shr, as_unsigned(s, v), Expr::int(32 - sh as i64), v);
            let lm = if mb == 0 { u32::MAX } else { (1u32 << (32 - mb as u32)) - 1 };
            return arith(BinOp::And, shifted, Expr::uint(lm as i64), v);
        }
        // `clrlslwi`: the field ends where the shift starts; the source may have masked first
        // (`(x & 3) << 1`) — a draft variant, the instruction is the same
        if mb <= me && me as u32 == 31 - sh as u32 && crate::variants::alt(crate::variants::EXPR_MASK_THEN_SHIFT) {
            return arith(BinOp::Shl, arith(BinOp::And, s, Expr::uint((m >> sh) as i64), v), Expr::int(sh as i64), v);
        }
        if mb <= me && (me as u32) <= 31 - sh as u32 {
            return arith(BinOp::And, arith(BinOp::Shl, s, Expr::int(sh as i64), v), Expr::uint(m as i64), v);
        }
        if mb <= me && (mb as u32) >= 32 - sh as u32 {
            // only bits that came from the wrap-around: (s >> (32-sh)) & m
            return arith(BinOp::And, arith(BinOp::Shr, as_unsigned(s, v), Expr::int(32 - sh as i64), v), Expr::uint(m as i64), v);
        }
        let rot = arith(
            BinOp::Or,
            arith(BinOp::Shl, s.clone(), Expr::int(sh as i64), v),
            arith(BinOp::Shr, as_unsigned(s, v), Expr::int(32 - sh as i64), v),
            v,
        );
        arith(BinOp::And, rot, Expr::uint(m as i64), v)
    }
}

impl Term {
    pub fn is_switch(&self) -> bool {
        matches!(self, Term::Switch { .. })
    }
}

fn strip_tmpl(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Demangled parameter spellings, exactly as in the symbol.
/// Parameters the header declares with a top-level `const` (by-value scalars), when exactly one
/// declaration fits the signature.
fn header_const_params(db: &TypeDb, s: &FuncSig) -> Vec<bool> {
    let Some(ds) = db.decls.get(&s.qualified_name) else { return vec![] };
    let same = |a: &Type, b: &Type| types::resolve(Some(db), strip_cv(a)).into_owned() == types::resolve(Some(db), strip_cv(b)).into_owned();
    let fits: Vec<&mwdec_core::DeclInfo> = ds.iter().filter(|d| d.params.len() == s.params.len() && d.is_const == s.is_const && d.params.iter().zip(&s.params).all(|(a, b)| same(&a.ty, &b.ty))).collect();
    let [d] = fits.as_slice() else { return vec![] };
    d.params
        .iter()
        .map(|p| match &p.ty {
            Type::Const(x) => matches!(types::resolve(Some(db), x).as_ref(), Type::Bool | Type::Int { .. } | Type::Long { .. } | Type::Float { .. }),
            _ => false,
        })
        .collect()
}

pub fn param_spellings(demangled: &str) -> Vec<String> {
    let mut t = demangled.trim();
    if let Some(r) = t.strip_suffix(" const") {
        t = r;
    }
    if !t.ends_with(')') {
        return vec![];
    }
    let b = t.as_bytes();
    let mut depth = 0;
    for i in (0..b.len()).rev() {
        match b[i] {
            b')' => depth += 1,
            b'(' => {
                depth -= 1;
                if depth == 0 {
                    return sig::split_top(&t[i + 1..t.len() - 1], ',').into_iter().filter(|s| !s.is_empty() && s != "void" && s != "...").collect();
                }
            }
            _ => {}
        }
    }
    vec![]
}

/// `delete p` for a deleting destructor call: `__delete(p)` with the class pointer as parameter
/// type (the emitter spells it `delete (T*)p`).
pub fn delete_call(p: Expr, dtor: &FuncSig) -> Expr {
    let mut s = builtin_sig("__delete", 1);
    if let Some(c) = &dtor.this_class {
        s.params[0].ty = t_ptr(Type::Named(c.clone()));
    }
    s.ret = Type::Void;
    Expr::Call { callee: Callee::Direct { symbol: "__delete".into(), sig: s }, args: vec![p], ret: Type::Void }
}

/// The bits of `m` are known to be clear in `e` (masked, narrow unsigned, shifted out).
pub fn known_zero<'d>(e: &Expr, m: u32, vars: &[Var], def: &dyn Fn(VarId) -> Option<&'d Expr>, depth: u32) -> bool {
    zero_bits(e, vars, def, depth) & m == m
}

/// Bits known to be clear in the value of `e`.
fn zero_bits<'d>(e: &Expr, vars: &[Var], def: &dyn Fn(VarId) -> Option<&'d Expr>, depth: u32) -> u32 {
    if depth > 8 {
        return 0;
    }
    let zb = |x: &Expr| zero_bits(x, vars, def, depth + 1);
    let narrow = |t: &Type| match strip_cv(t) {
        Type::Int { size: 1, signed: false } | Type::Bool => !0xffu32,
        Type::Int { size: 2, signed: false } | Type::WChar => !0xffffu32,
        _ => 0,
    };
    let by_type = narrow(&ty_of(e, vars));
    by_type
        | match e {
            Expr::Int { value, .. } => !(*value as u32),
            Expr::Cast { ty, e: x, .. } => match narrow(ty) {
                0 if scalar_size(ty) == Some(4) => zb(x),
                0 => 0,
                n => n | zb(x),
            },
            Expr::Binary { op: BinOp::And, l, r, .. } => zb(l) | zb(r),
            Expr::Binary { op: BinOp::Or, l, r, .. } => zb(l) & zb(r),
            Expr::Binary { op: BinOp::Shl, l, r, .. } => match r.as_int() {
                Some(n @ 0..=31) => (zb(l) << n) | ((1u32 << n) - 1),
                _ => 0,
            },
            Expr::Binary { op: BinOp::Shr, l, r, .. } => match r.as_int() {
                Some(n @ 1..=31) if !ty_of(l, vars).int_info().is_some_and(|(_, s)| s) => (zb(l) >> n) | !(u32::MAX >> n),
                _ => 0,
            },
            // (the compiler doesn't look through the insert intrinsic)
            Expr::Var(t) => def(*t).map_or(0, |d| zb(d)),
            _ => 0,
        }
}

pub fn builtin_sig(name: &str, n: usize) -> FuncSig {
    FuncSig {
        qualified_name: name.into(),
        mangled: Some(name.into()),
        ret: t_unk(0),
        params: (0..n).map(|_| mwdec_core::Param { name: None, ty: t_unk(4) }).collect(),
        this_class: None,
        is_const: false,
        is_static: false,
        is_virtual: false,
        variadic: false,
        runs_code: false,
    }
}

pub fn compatible_scalar(db: Option<&TypeDb>, field: &Type, access: &Type) -> bool {
    let r = types::resolve(db, field);
    let fs = types::size_of(db, &r);
    let asz = scalar_size(access);
    if fs != asz {
        return false;
    }
    let fl = is_float(&r);
    let al = is_float(access);
    if fl != al {
        return false;
    }
    if types::is_aggregate(db, &r) {
        return false;
    }
    // an array is never a scalar (a byte buffer read as a wider value)
    if matches!(strip_cv(&r), Type::Array(..)) && !matches!(strip_cv(access), Type::Array(..)) {
        return false;
    }
    true
}

fn int_ty(e: &Expr, vars: &[Var]) -> Type {
    let t = norm_int(&ty_of(e, vars));
    match strip_cv(&t) {
        Type::Int { size: 4, .. } => t,
        Type::Unknown { .. } | Type::Ptr(_) | Type::Ref(_) => t_s32(),
        Type::Int { signed, .. } => t_int(4, *signed),
        _ => t_s32(),
    }
}

/// Integer type of a quantization register as the runtime sets them up (`OSInitFastCast`):
/// qr2 u8, qr3 u16, qr4 s8, qr5 s16.
fn quant_type(q: u8) -> Option<Type> {
    match q {
        2 => Some(t_int(1, false)),
        3 => Some(t_int(2, false)),
        4 => Some(t_int(1, true)),
        5 => Some(t_int(2, true)),
        _ => None,
    }
}

/// The shift kind is the type of a right shift (`srw` unsigned, `sraw`/`srawi` signed); a narrow
/// operand promotes to `int`, so the emitter converts it when that disagrees.
fn shr_kind(e: Expr, signed: bool) -> Expr {
    match e {
        Expr::Binary { op: BinOp::Shr, l, r, .. } => Expr::Binary { op: BinOp::Shr, l, r, ty: t_int(4, signed) },
        e => e,
    }
}

/// Integer binary op with C-like result type.
pub fn arith(op: BinOp, a: Expr, b: Expr, vars: &[Var]) -> Expr {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        let (x, y) = (x as i32, y as i32);
        let r = match op {
            BinOp::Add => Some(x.wrapping_add(y)),
            BinOp::Sub => Some(x.wrapping_sub(y)),
            BinOp::Mul => Some(x.wrapping_mul(y)),
            BinOp::And => Some(x & y),
            BinOp::Or => Some(x | y),
            BinOp::Xor => Some(x ^ y),
            BinOp::Shl => Some(((x as u32) << (y & 31)) as i32),
            BinOp::Shr => Some(((x as u32) >> (y & 31)) as i32),
            _ => None,
        };
        if let Some(r) = r {
            let unsigned = matches!(a, Expr::Int { ty: Type::Int { signed: false, .. }, .. })
                || matches!(b, Expr::Int { ty: Type::Int { signed: false, .. }, .. });
            return if unsigned { Expr::uint(r as u32 as i64) } else { Expr::int(r as i64) };
        }
    }
    let ta = int_ty(&a, vars);
    let tb = int_ty(&b, vars);
    let ty = if matches!(op, BinOp::Shl | BinOp::Shr) {
        ta
    } else if is_signed(&ta) == Some(false) || is_signed(&tb) == Some(false) {
        t_u32()
    } else {
        t_s32()
    };
    Expr::bin(op, a, b, ty)
}

fn bitnot(e: Expr, vars: &[Var]) -> Expr {
    if let Some(c) = e.as_int() {
        return Expr::uint((!(c as u32)) as i64);
    }
    let ty = int_ty(&e, vars);
    Expr::Unary { op: UnOp::BitNot, e: Box::new(e), ty }
}

/// Long/Char/WChar -> the equivalent `Int` (for arithmetic typing).
pub fn norm_int(t: &Type) -> Type {
    match strip_cv(t) {
        Type::Long { .. } | Type::Char | Type::WChar => {
            let (size, signed) = strip_cv(t).int_info().unwrap();
            Type::Int { size, signed }
        }
        _ => t.clone(),
    }
}

/// Signedness of an expression as C would type it (unsuffixed literals that fit are `int`,
/// usual arithmetic conversions), which can differ from the IR's instruction-derived type.
pub fn c_unsigned(e: &Expr, vars: &[Var]) -> bool {
    match e {
        Expr::Int { value, .. } => *value > i32::MAX as i64,
        Expr::Binary { op, l, r, .. } if op.is_bool() => {
            let _ = (l, r);
            false
        }
        Expr::Binary { op: BinOp::Shl | BinOp::Shr, l, .. } => c_unsigned(l, vars),
        Expr::Binary { l, r, .. } => c_unsigned(l, vars) || c_unsigned(r, vars),
        Expr::Unary { e, .. } => c_unsigned(e, vars),
        _ => {
            let t = norm_int(&ty_of(e, vars));
            match strip_cv(&t) {
                // narrower unsigned types promote to int
                Type::Int { signed: false, size } => *size >= 4,
                Type::Ptr(_) | Type::Ref(_) => true,
                _ => false,
            }
        }
    }
}

pub fn as_signed(e: Expr, vars: &[Var]) -> Expr {
    if matches!(e, Expr::Binary { .. }) && c_unsigned(&e, vars) && !matches!(ty_of(&e, vars), Type::Unknown { .. }) {
        return Expr::cast(t_s32(), e);
    }
    let t = norm_int(&ty_of(&e, vars));
    match strip_cv(&t) {
        Type::Int { signed: false, size } if *size >= 4 => {
            if let Expr::Int { value, .. } = e {
                return Expr::int(value as i32 as i64);
            }
            Expr::cast(t_s32(), e)
        }
        Type::Ptr(_) | Type::Ref(_) => Expr::cast(t_s32(), e),
        Type::Bool => Expr::cast(t_s32(), e),
        _ => e,
    }
}

pub fn as_unsigned(e: Expr, vars: &[Var]) -> Expr {
    if matches!(e, Expr::Binary { .. }) && !c_unsigned(&e, vars) {
        return Expr::cast(t_u32(), e);
    }
    let t = norm_int(&ty_of(&e, vars));
    match strip_cv(&t) {
        Type::Int { signed: true, size } => {
            if let Expr::Int { value, .. } = e {
                return Expr::uint(value as u32 as i64);
            }
            if *size < 4 {
                // a sign-extended small int compared unsigned
                return Expr::cast(t_u32(), e);
            }
            Expr::cast(t_u32(), e)
        }
        Type::Unknown { .. } => {
            if let Expr::Int { value, .. } = e {
                return Expr::uint(value as u32 as i64);
            }
            // an untyped word compared unsigned (cmplw): make the compare unsigned in C
            Expr::cast(t_u32(), e)
        }
        _ => e,
    }
}

/// A halfword/byte store of a negative constant (`li r0, -1; sth`) stores a signed narrow value:
/// through an unsigned type the constant would be 0xffff, materialised as `lis; subi`.
fn const_store_ty(ty: Type, src: &Expr) -> Type {
    match (&ty, src.as_int()) {
        (Type::Int { size: size @ (1 | 2), signed: false }, Some(c)) if c < 0 && c >= -(1i64 << (8 * *size as i64 - 1)) => Type::Int { size: *size, signed: true },
        _ => ty,
    }
}

fn byte_add(p: Expr, off: Expr) -> Expr {
    let cp = Expr::cast(t_ptr(t_int(1, false)), p);
    Expr::AddrOf(Box::new(Expr::Index { base: Box::new(cp), index: Box::new(off), ty: t_int(1, false) }))
}

/// Narrow a stored value to the store width (`stb` of an int stores its low byte).
fn narrow_store(e: Expr, _ty: &Type) -> Expr {
    e
}

/// `(x + M) - M` -> `x` for the int->double conversion magic constants.
fn fold_magic(e: Expr, ty: &Type) -> Expr {
    if let Expr::Binary { op: BinOp::Sub, l, r, .. } = &e {
        if let (Expr::Binary { op: BinOp::Add, l: x, r: m1, .. }, Expr::Float { bits, double: true }) = (&**l, &**r) {
            if let Expr::Float { bits: b1, double: true } = &**m1 {
                if b1 == bits && (*bits == 0x4330_0000_8000_0000 || *bits == 0x4330_0000_0000_0000) {
                    let inner = match &**x {
                        Expr::Cast { e, .. } => (**e).clone(),
                        other => other.clone(),
                    };
                    return Expr::cast(ty.clone(), inner);
                }
            }
        }
    }
    e
}

/// OR of two CR bits, folding `a<b || a==b` into `a<=b`.
fn combine_or(a: Expr, b: Expr) -> Expr {
    if let (Expr::Binary { op: o1, l: l1, r: r1, .. }, Expr::Binary { op: o2, l: l2, r: r2, .. }) = (&a, &b) {
        if l1 == l2 && r1 == r2 {
            let op = match (o1, o2) {
                (BinOp::Lt, BinOp::Eq) | (BinOp::Eq, BinOp::Lt) => Some(BinOp::Le),
                (BinOp::Gt, BinOp::Eq) | (BinOp::Eq, BinOp::Gt) => Some(BinOp::Ge),
                (BinOp::Lt, BinOp::Gt) | (BinOp::Gt, BinOp::Lt) => Some(BinOp::Ne),
                _ => None,
            };
            if let Some(op) = op {
                return Expr::cmp(op, (**l1).clone(), (**r1).clone());
            }
        }
    }
    Expr::cmp(BinOp::LogOr, a, b)
}


/// Bits of `1 / v` when `v` is a power of two other than 1 (exactly representable).
fn pow2_reciprocal(bits: u64, double: bool) -> Option<u64> {
    if double {
        let (sign, exp, man) = (bits >> 63, (bits >> 52) & 0x7ff, bits & ((1u64 << 52) - 1));
        if man != 0 || exp == 0 || exp == 0x7ff || exp == 1023 {
            return None;
        }
        let re = 2046 - exp as i64;
        if !(1..=2046).contains(&re) {
            return None;
        }
        Some((sign << 63) | ((re as u64) << 52))
    } else {
        let b = bits as u32;
        let (sign, exp, man) = (b >> 31, (b >> 23) & 0xff, b & 0x7f_ffff);
        if man != 0 || exp == 0 || exp == 0xff || exp == 127 {
            return None;
        }
        let re = 254 - exp as i64;
        if !(1..=254).contains(&re) {
            return None;
        }
        Some(((sign << 31) | ((re as u32) << 23)) as u64)
    }
}
