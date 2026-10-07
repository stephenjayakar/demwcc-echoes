//! The decompiler IR: a structured tree of statements over typed expressions.
//!
//! Design goals:
//! - **Rewriteable**: plain owned trees (`Box`/`Vec`), `Clone`, no arenas or interior
//!   mutability, so the search stage can reorder statements, inline/outline temps, change loop
//!   forms and retype variables freely, then hand the result to `mwdec-emit`.
//! - **Lossless enough for matching**: every memory access keeps its base expression, byte offset
//!   and access type; the emitter decides whether it becomes `this->mField`, `p->a.b` or a raw
//!   `*(T*)((char*)p + 0x34)` depending on the `TypeDb`.
//! - Variables are function-scoped (`IrFunction::vars`) and referenced by index.

use mwdec_core::{FuncSig, Type};

pub type VarId = usize;
pub type LabelId = usize;

#[derive(Clone, Debug, PartialEq)]
pub enum VarKind {
    /// Implicit `this` (not declared).
    This,
    /// Declared parameter number `index` (0-based, not counting `this`/hidden return pointer).
    Param { index: usize },
    /// Hidden struct-return pointer (not declared in C++; the emitter renders writes through it as
    /// constructing the return value where it can).
    StructRet,
    /// A register-allocated local (a web of register definitions).
    Local,
    /// A stack-frame local at `r1 + offset` of `size` bytes (address may be taken).
    Stack { offset: i32, size: u32 },
    /// Compiler-internal value with no source spelling (e.g. the destructor's "delete" flag in
    /// r4); idiom passes remove its uses, it is never declared.
    Hidden,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Var {
    pub name: String,
    pub ty: Type,
    pub kind: VarKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    /// Right shift; arithmetic vs logical is decided by the (signedness of the) left operand type.
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LogAnd,
    LogOr,
}

impl BinOp {
    pub fn is_cmp(self) -> bool {
        matches!(self, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)
    }
    pub fn is_bool(self) -> bool {
        self.is_cmp() || matches!(self, BinOp::LogAnd | BinOp::LogOr)
    }
    /// Negation of a comparison (`<` -> `>=`). Only valid for integer compares or ordered floats.
    pub fn negate_cmp(self) -> Option<BinOp> {
        Some(match self {
            BinOp::Eq => BinOp::Ne,
            BinOp::Ne => BinOp::Eq,
            BinOp::Lt => BinOp::Ge,
            BinOp::Ge => BinOp::Lt,
            BinOp::Gt => BinOp::Le,
            BinOp::Le => BinOp::Gt,
            _ => return None,
        })
    }
    /// `a op b` == `b swap(op) a`
    pub fn swap_cmp(self) -> BinOp {
        match self {
            BinOp::Lt => BinOp::Gt,
            BinOp::Gt => BinOp::Lt,
            BinOp::Le => BinOp::Ge,
            BinOp::Ge => BinOp::Le,
            o => o,
        }
    }
    pub fn c_str(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::And => "&",
            BinOp::Or => "|",
            BinOp::Xor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::LogAnd => "&&",
            BinOp::LogOr => "||",
        }
    }
    /// C precedence (higher binds tighter).
    pub fn prec(self) -> u8 {
        match self {
            BinOp::Mul | BinOp::Div | BinOp::Rem => 13,
            BinOp::Add | BinOp::Sub => 12,
            BinOp::Shl | BinOp::Shr => 11,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 10,
            BinOp::Eq | BinOp::Ne => 9,
            BinOp::And => 8,
            BinOp::Xor => 7,
            BinOp::Or => 6,
            BinOp::LogAnd => 5,
            BinOp::LogOr => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Neg,
    /// `~x`
    BitNot,
    /// `!x`
    Not,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Callee {
    /// Free function or static member: `Ns::Func(args)`.
    Direct { symbol: String, sig: FuncSig },
    /// Non-virtual member call `obj->Method(args)` / `obj.Method(args)` (or a qualified base call
    /// `Base::Method(args)` when `qualified` is set, e.g. `CEntity::Think(dt, mgr)` from a
    /// derived override).
    Method { symbol: String, sig: FuncSig, this: Box<Expr>, qualified: bool },
    /// Virtual call through the vtable of `this` (`lwz r12,0(rX); lwz r12,slot(r12); bctrl`).
    Virtual { this: Box<Expr>, vtable_offset: u32, vptr_offset: u32, class: Option<String>, sig: Option<FuncSig> },
    /// Call through a function pointer expression.
    Indirect(Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var(VarId),
    /// Integer constant (`ty` is an int/bool/pointer type: `0` of pointer type renders as `nullptr`/`0`).
    Int { value: i64, ty: Type },
    /// Float constant. `bits` holds the f32 bits if `!double`, else f64 bits. `symbol` is the
    /// literal-pool symbol it came from (bookkeeping).
    Float { bits: u64, double: bool },
    /// String literal (bytes without the trailing NUL).
    Str { bytes: Vec<u8> },
    /// Named global data object (lvalue).
    Global { symbol: String, ty: Type },
    /// Address of a function symbol.
    FuncAddr { symbol: String },
    /// `&e`
    AddrOf(Box<Expr>),
    /// Memory access of type `ty` at `base + offset` where `base` is a pointer value (lvalue).
    /// Renders as a member access when the pointee type is known.
    Load { base: Box<Expr>, offset: i32, ty: Type },
    /// `base[index]` with element type `ty` (from `lwzx`/`slwi` patterns), lvalue.
    Index { base: Box<Expr>, index: Box<Expr>, ty: Type },
    /// Member of an aggregate lvalue (stack struct, global struct) at a byte offset.
    Member { base: Box<Expr>, offset: i32, ty: Type },
    Unary { op: UnOp, e: Box<Expr>, ty: Type },
    Binary { op: BinOp, l: Box<Expr>, r: Box<Expr>, ty: Type },
    Cast { ty: Type, e: Box<Expr> },
    Call { callee: Callee, args: Vec<Expr>, ret: Type },
    Ternary { c: Box<Expr>, t: Box<Expr>, f: Box<Expr>, ty: Type },
    /// Something the lifter could not express; renders as a comment + `0`.
    Unknown { text: String, ty: Type },
    /// `new (placement...) Class(args)`; `ctor` is None for types without a constructor call.
    New { class: Type, placement: Vec<Expr>, ctor: Option<FuncSig>, args: Vec<Expr> },
    /// Temporary object `Class(args)` (constructor call as a value, e.g. a returned object).
    Construct { class: Type, ctor: Option<FuncSig>, args: Vec<Expr> },
    /// Bitfield member stored in the storage-unit access `base` (a Load/Member of an integer
    /// type): value bits `[shift, shift+width)` counted from the LSB of the unit (lvalue).
    BitField { base: Box<Expr>, shift: u8, width: u8, ty: Type },
    /// `x++` / `x--` (post) or `++x` / `--x` (pre) on an lvalue; `delta` is +1/-1 elements.
    IncDec { e: Box<Expr>, delta: i64, post: bool },
}

/// Constructor initializer-list entry.
#[derive(Clone, Debug, PartialEq)]
pub enum InitTarget {
    /// Base class subobject.
    Base(String),
    /// Direct member by name.
    Member(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Init {
    pub target: InitTarget,
    pub ctor: Option<FuncSig>,
    pub args: Vec<Expr>,
    /// Member type for plain (non-constructor) initialization.
    pub member_ty: Option<Type>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwitchCase {
    pub values: Vec<i64>,
    pub is_default: bool,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Expr(Expr),
    Assign { dst: Expr, src: Expr },
    If { cond: Expr, then: Vec<Stmt>, els: Vec<Stmt> },
    While { cond: Expr, body: Vec<Stmt> },
    DoWhile { body: Vec<Stmt>, cond: Expr },
    For { init: Vec<Stmt>, cond: Expr, step: Vec<Stmt>, body: Vec<Stmt> },
    Switch { e: Expr, cases: Vec<SwitchCase> },
    Return(Option<Expr>),
    Break,
    Continue,
    Goto(LabelId),
    Label(LabelId),
    Comment(String),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameInfo {
    /// Stack frame size (0 for leaf functions without a frame).
    pub size: u32,
    pub saves_lr: bool,
    /// Callee-saved GPRs (r14..r31) and FPRs (f14..f31) saved in the prologue.
    pub saved_gprs: Vec<u8>,
    pub saved_fprs: Vec<u8>,
    /// FPRs saved with `psq_st` as well (paired-single upper halves).
    pub saved_ps: Vec<u8>,
    pub uses_stmw: bool,
    pub uses_savegpr: bool,
}

/// A referenced global (data or function) the emitter may need to declare.
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalRef {
    pub symbol: String,
    pub ty: Type,
    pub is_function: bool,
    /// Section the symbol lives in, if defined in the target object.
    pub section: Option<String>,
    /// Defined (not extern) in the target object.
    pub local_def: bool,
    /// Initial bytes of an object defined in an initialized data section of the target object
    /// (without relocations), for a definition the emitter writes itself (function statics).
    pub init: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IrFunction {
    pub symbol: String,
    /// Signature (qualified name, params, const, inferred or DB return type).
    pub sig: FuncSig,
    pub vars: Vec<Var>,
    /// Declared parameters in order (indices into `vars`).
    pub params: Vec<VarId>,
    /// Exact C++ spelling of each declared parameter type (from the demangled symbol), so the
    /// definition matches the header declaration (`long` vs `int`, typedef-free templates).
    pub decl_params: Vec<String>,
    pub this_var: Option<VarId>,
    pub body: Vec<Stmt>,
    /// Constructor initializer list (empty for other functions).
    pub init_list: Vec<Init>,
    pub globals: Vec<GlobalRef>,
    pub frame: FrameInfo,
    /// Lifter diagnostics (unhandled instructions, guesses).
    pub warnings: Vec<String>,
}

impl IrFunction {
    pub fn var(&self, v: VarId) -> &Var {
        &self.vars[v]
    }
}

// ---------------------------------------------------------------- type helpers

pub fn t_int(size: u8, signed: bool) -> Type {
    Type::Int { size, signed }
}
pub fn t_s32() -> Type {
    t_int(4, true)
}
pub fn t_u32() -> Type {
    t_int(4, false)
}
pub fn t_f32() -> Type {
    Type::Float { size: 4 }
}
pub fn t_f64() -> Type {
    Type::Float { size: 8 }
}
pub fn t_unk(size: u32) -> Type {
    Type::Unknown { size }
}
pub fn t_ptr(t: Type) -> Type {
    Type::Ptr(Box::new(t))
}

/// Strip const/volatile.
pub fn strip_cv(t: &Type) -> &Type {
    match t {
        Type::Const(t) | Type::Volatile(t) => strip_cv(t),
        t => t,
    }
}

/// Pointee of a pointer/reference type (cv stripped on the outer level).
pub fn pointee(t: &Type) -> Option<&Type> {
    match strip_cv(t) {
        Type::Ptr(t) | Type::Ref(t) => Some(t),
        Type::Array(t, _) => Some(t),
        _ => None,
    }
}

pub fn is_float(t: &Type) -> bool {
    matches!(strip_cv(t), Type::Float { .. })
}

pub fn is_ptr(t: &Type) -> bool {
    matches!(strip_cv(t), Type::Ptr(_) | Type::Ref(_) | Type::FuncPtr(_))
}

pub fn is_signed(t: &Type) -> Option<bool> {
    match strip_cv(t) {
        Type::Int { signed, .. } | Type::Long { signed } => Some(*signed),
        Type::Char => Some(true),
        Type::Bool | Type::WChar => Some(false),
        Type::Ptr(_) | Type::Ref(_) => Some(false),
        _ => None,
    }
}

/// Size in bytes for scalar types; None for named aggregates (needs the TypeDb).
pub fn scalar_size(t: &Type) -> Option<u32> {
    match strip_cv(t) {
        Type::Void => Some(0),
        Type::Bool | Type::Char => Some(1),
        Type::WChar => Some(2),
        Type::Long { .. } => Some(4),
        Type::Int { size, .. } => Some(*size as u32),
        Type::Float { size } => Some(*size as u32),
        Type::Ptr(_) | Type::Ref(_) | Type::FuncPtr(_) => Some(4),
        Type::Unknown { size } => Some(*size),
        Type::Array(t, n) => scalar_size(t).map(|s| s * n),
        Type::MemberPtr { size, .. } => Some(*size),
        _ => None,
    }
}

/// Class name for `Named`, through cv.
pub fn named(t: &Type) -> Option<&str> {
    match strip_cv(t) {
        Type::Named(n) => Some(n),
        _ => None,
    }
}

// ---------------------------------------------------------------- expr helpers

impl Expr {
    pub fn int(v: i64) -> Expr {
        Expr::Int { value: v, ty: t_s32() }
    }
    pub fn uint(v: i64) -> Expr {
        Expr::Int { value: v, ty: t_u32() }
    }
    pub fn bin(op: BinOp, l: Expr, r: Expr, ty: Type) -> Expr {
        Expr::Binary { op, l: Box::new(l), r: Box::new(r), ty }
    }
    pub fn cmp(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::Binary { op, l: Box::new(l), r: Box::new(r), ty: Type::Bool }
    }
    pub fn not(e: Expr) -> Expr {
        Expr::Unary { op: UnOp::Not, e: Box::new(e), ty: Type::Bool }
    }
    pub fn cast(ty: Type, e: Expr) -> Expr {
        Expr::Cast { ty, e: Box::new(e) }
    }
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Expr::Int { value, .. } => Some(*value),
            _ => None,
        }
    }
    pub fn is_lvalue(&self) -> bool {
        matches!(self, Expr::Var(_) | Expr::Global { .. } | Expr::Load { .. } | Expr::Index { .. } | Expr::Member { .. } | Expr::BitField { .. })
    }

    /// Logical negation with comparison flipping where it is exact.
    pub fn negated(self) -> Expr {
        match self {
            Expr::Unary { op: UnOp::Not, e, .. } => *e,
            Expr::Binary { op, l, r, ty } if op.is_cmp() => {
                let float = is_float_expr_hint(&l) || is_float_expr_hint(&r);
                if float && !matches!(op, BinOp::Eq | BinOp::Ne) {
                    // !(a < b) != (a >= b) for NaN; keep the negation explicit.
                    Expr::not(Expr::Binary { op, l, r, ty })
                } else {
                    Expr::Binary { op: op.negate_cmp().unwrap(), l, r, ty }
                }
            }
            Expr::Binary { op: BinOp::LogAnd, l, r, ty } => {
                Expr::Binary { op: BinOp::LogOr, l: Box::new(l.negated()), r: Box::new(r.negated()), ty }
            }
            Expr::Binary { op: BinOp::LogOr, l, r, ty } => {
                Expr::Binary { op: BinOp::LogAnd, l: Box::new(l.negated()), r: Box::new(r.negated()), ty }
            }
            e => Expr::not(e),
        }
    }

    /// Logical negation using variable types to keep float compares NaN-exact
    /// (`!(a < b)` stays as is for floats; integer compares flip).
    pub fn negate(self, vars: &[Var]) -> Expr {
        match self {
            Expr::Unary { op: UnOp::Not, e, .. } => *e,
            Expr::Int { value, ty: Type::Bool } => Expr::Int { value: (value == 0) as i64, ty: Type::Bool },
            Expr::Binary { op, l, r, ty } if op.is_cmp() => {
                let float = is_float(&crate::types::ty_of(&l, vars)) || is_float(&crate::types::ty_of(&r, vars));
                if float && !matches!(op, BinOp::Eq | BinOp::Ne) {
                    Expr::not(Expr::Binary { op, l, r, ty })
                } else {
                    Expr::Binary { op: op.negate_cmp().unwrap(), l, r, ty }
                }
            }
            Expr::Binary { op: BinOp::LogAnd, l, r, ty } => {
                Expr::Binary { op: BinOp::LogOr, l: Box::new(l.negate(vars)), r: Box::new(r.negate(vars)), ty }
            }
            Expr::Binary { op: BinOp::LogOr, l, r, ty } => {
                Expr::Binary { op: BinOp::LogAnd, l: Box::new(l.negate(vars)), r: Box::new(r.negate(vars)), ty }
            }
            e => Expr::not(e),
        }
    }

    /// Visit all sub-expressions (pre-order), including callee operands.
    pub fn walk<'a>(&'a self, f: &mut dyn FnMut(&'a Expr)) {
        f(self);
        match self {
            Expr::AddrOf(e) | Expr::Unary { e, .. } | Expr::Cast { e, .. } => e.walk(f),
            Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => base.walk(f),
            Expr::IncDec { e, .. } => e.walk(f),
            Expr::Index { base, index, .. } => {
                base.walk(f);
                index.walk(f);
            }
            Expr::Binary { l, r, .. } => {
                l.walk(f);
                r.walk(f);
            }
            Expr::Ternary { c, t, f: e, .. } => {
                c.walk(f);
                t.walk(f);
                e.walk(f);
            }
            Expr::Call { callee, args, .. } => {
                match callee {
                    Callee::Method { this, .. } | Callee::Virtual { this, .. } => this.walk(f),
                    Callee::Indirect(e) => e.walk(f),
                    Callee::Direct { .. } => {}
                }
                for a in args {
                    a.walk(f);
                }
            }
            Expr::New { placement, args, .. } => {
                for a in placement.iter().chain(args.iter()) {
                    a.walk(f);
                }
            }
            Expr::Construct { args, .. } => {
                for a in args {
                    a.walk(f);
                }
            }
            _ => {}
        }
    }

    /// Mutable post-order rewrite.
    pub fn rewrite(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        match self {
            Expr::AddrOf(e) | Expr::Unary { e, .. } | Expr::Cast { e, .. } => e.rewrite(f),
            Expr::Load { base, .. } | Expr::Member { base, .. } | Expr::BitField { base, .. } => base.rewrite(f),
            Expr::IncDec { e, .. } => e.rewrite(f),
            Expr::Index { base, index, .. } => {
                base.rewrite(f);
                index.rewrite(f);
            }
            Expr::Binary { l, r, .. } => {
                l.rewrite(f);
                r.rewrite(f);
            }
            Expr::Ternary { c, t, f: e, .. } => {
                c.rewrite(f);
                t.rewrite(f);
                e.rewrite(f);
            }
            Expr::Call { callee, args, .. } => {
                match callee {
                    Callee::Method { this, .. } | Callee::Virtual { this, .. } => this.rewrite(f),
                    Callee::Indirect(e) => e.rewrite(f),
                    Callee::Direct { .. } => {}
                }
                for a in args {
                    a.rewrite(f);
                }
            }
            Expr::New { placement, args, .. } => {
                for a in placement.iter_mut().chain(args.iter_mut()) {
                    a.rewrite(f);
                }
            }
            Expr::Construct { args, .. } => {
                for a in args.iter_mut() {
                    a.rewrite(f);
                }
            }
            _ => {}
        }
        f(self);
    }

    pub fn uses_var(&self, v: VarId) -> bool {
        let mut found = false;
        self.walk(&mut |e| {
            if matches!(e, Expr::Var(x) if *x == v) {
                found = true
            }
        });
        found
    }

    /// Does evaluating `self` read the value (contents) of variable `v`? Unlike `uses_var`, a
    /// pure address computation (`&v`, `&v.field`) does not read `v`.
    pub fn reads_var(&self, v: VarId) -> bool {
        fn addr_reads(lv: &Expr, v: VarId) -> bool {
            match lv {
                Expr::Var(_) | Expr::Global { .. } => false,
                Expr::Member { base, .. } => addr_reads(base, v),
                Expr::Index { base, index, .. } => addr_reads(base, v) || index.reads_var(v),
                Expr::Load { base, .. } => base.reads_var(v),
                other => other.uses_var(v),
            }
        }
        match self {
            Expr::Var(x) => *x == v,
            Expr::AddrOf(inner) => addr_reads(inner, v),
            Expr::Cast { e, .. } => e.reads_var(v),
            Expr::Binary { op: BinOp::Add | BinOp::Sub, l, r, .. } => l.reads_var(v) || r.reads_var(v),
            _ => self.uses_var(v),
        }
    }

    pub fn has_call(&self) -> bool {
        let mut found = false;
        self.walk(&mut |e| {
            if matches!(e, Expr::Call { .. } | Expr::New { .. } | Expr::IncDec { .. }) && !e.is_pure_call() {
                found = true
            }
        });
        found
    }

    /// A call of a side-effect-free intrinsic on its operands (`__rlwimi`, `__cntlzw`): an operator, not a call.
    pub fn is_pure_call(&self) -> bool {
        matches!(self, Expr::Call { callee: Callee::Direct { symbol, .. }, .. } if symbol == "__rlwimi" || symbol == "__cntlzw")
    }
}

fn is_float_expr_hint(e: &Expr) -> bool {
    match e {
        Expr::Float { .. } => true,
        Expr::Load { ty, .. } | Expr::Member { ty, .. } | Expr::Index { ty, .. } | Expr::Global { ty, .. } => is_float(ty),
        Expr::Binary { ty, .. } | Expr::Unary { ty, .. } | Expr::Cast { ty, .. } | Expr::Ternary { ty, .. } => is_float(ty),
        Expr::Call { ret, .. } => is_float(ret),
        _ => false,
    }
}

impl Stmt {
    /// Visit nested statement lists mutably (pre-order over statements).
    pub fn for_each_block_mut(body: &mut Vec<Stmt>, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
        f(body);
        for s in body.iter_mut() {
            match s {
                Stmt::If { then, els, .. } => {
                    Stmt::for_each_block_mut(then, f);
                    Stmt::for_each_block_mut(els, f);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => Stmt::for_each_block_mut(body, f),
                Stmt::For { init, step, body, .. } => {
                    Stmt::for_each_block_mut(init, f);
                    Stmt::for_each_block_mut(step, f);
                    Stmt::for_each_block_mut(body, f);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        Stmt::for_each_block_mut(&mut c.body, f);
                    }
                }
                _ => {}
            }
        }
    }

    /// Visit every expression in a statement list (including conditions).
    pub fn walk_exprs<'a>(body: &'a [Stmt], f: &mut dyn FnMut(&'a Expr)) {
        for s in body {
            match s {
                Stmt::Expr(e) | Stmt::Return(Some(e)) => e.walk(f),
                Stmt::Assign { dst, src } => {
                    dst.walk(f);
                    src.walk(f);
                }
                Stmt::If { cond, then, els } => {
                    cond.walk(f);
                    Stmt::walk_exprs(then, f);
                    Stmt::walk_exprs(els, f);
                }
                Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                    cond.walk(f);
                    Stmt::walk_exprs(body, f);
                }
                Stmt::For { init, cond, step, body } => {
                    Stmt::walk_exprs(init, f);
                    cond.walk(f);
                    Stmt::walk_exprs(step, f);
                    Stmt::walk_exprs(body, f);
                }
                Stmt::Switch { e, cases } => {
                    e.walk(f);
                    for c in cases {
                        Stmt::walk_exprs(&c.body, f);
                    }
                }
                _ => {}
            }
        }
    }

    /// Rewrite every expression in a statement list (post-order inside each expression).
    pub fn rewrite_exprs(body: &mut [Stmt], f: &mut dyn FnMut(&mut Expr)) {
        for s in body {
            match s {
                Stmt::Expr(e) | Stmt::Return(Some(e)) => e.rewrite(f),
                Stmt::Assign { dst, src } => {
                    dst.rewrite(f);
                    src.rewrite(f);
                }
                Stmt::If { cond, then, els } => {
                    cond.rewrite(f);
                    Stmt::rewrite_exprs(then, f);
                    Stmt::rewrite_exprs(els, f);
                }
                Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                    cond.rewrite(f);
                    Stmt::rewrite_exprs(body, f);
                }
                Stmt::For { init, cond, step, body } => {
                    Stmt::rewrite_exprs(init, f);
                    cond.rewrite(f);
                    Stmt::rewrite_exprs(step, f);
                    Stmt::rewrite_exprs(body, f);
                }
                Stmt::Switch { e, cases } => {
                    e.rewrite(f);
                    for c in cases {
                        Stmt::rewrite_exprs(&mut c.body, f);
                    }
                }
                _ => {}
            }
        }
    }
}
