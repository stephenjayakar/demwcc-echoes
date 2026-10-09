//! mwdec-emit: IR -> C++ function text (MWCC GC/2.7 compatible, C++98 + project idioms).
//!
//! `emit_function` renders a definition `Ret Class::Method(params) const { ... }` from the
//! demangled signature, member accesses as field paths when a `TypeDb` is available (else raw
//! `*(T*)((char*)p + 0x34)`), calls as `obj->Method(args)` / `Ns::Func(args)`, virtual calls as
//! method calls (or vtable function-pointer calls without a DB), float literals that round-trip
//! exactly, and a separate preamble of declarations for referenced globals not in the context.

pub mod float;
pub mod instantiate;
mod sinit;
pub mod tidy;
pub mod types;

use mwdec_core::{Type, TypeDb};
use mwdec_lift::ir::*;
use mwdec_lift::sig;
use mwdec_lift::types::{field_path, ty_of, PathElem};
use std::collections::{BTreeSet, HashSet};
use std::fmt::Write;
pub use types::{decl, type_str};

#[derive(Clone, Debug)]
pub struct EmitOptions {
    pub indent: String,
    /// Render member accesses as raw offsets even when the TypeDb knows the field.
    pub raw_offsets: bool,
    /// Emit lifter warnings as a comment block before the function.
    pub warnings_comment: bool,
    /// Spelling for null pointers.
    pub null: String,
    /// The unit is C (`-lang=c`): no bool/true/false/templates/casts of C++ flavour.
    pub c_mode: bool,
    /// Static initializers: define the globals the context doesn't declare `const` (a const
    /// object's initialization is scheduled differently).
    pub sinit_const: bool,
    /// Static initializers: alternative k (1-based) for objects whose stores match no
    /// constructor initializer list: the k-th constructor taking one scalar per stored member.
    pub sinit_variant: u8,
}

impl Default for EmitOptions {
    fn default() -> Self {
        EmitOptions { indent: "    ".into(), raw_offsets: false, warnings_comment: true, null: "nullptr".into(), c_mode: false, sinit_const: false, sinit_variant: 0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Emitted {
    /// Declarations needed before the function (externs for globals not in the context).
    pub preamble: String,
    /// The function definition.
    pub body: String,
}

struct Em<'a> {
    ir: &'a IrFunction,
    db: Option<&'a TypeDb>,
    opts: &'a EmitOptions,
    out: String,
    externs: BTreeSet<String>,
    /// rendering the destination of an assignment (no accessor substitution at the top level)
    lvalue_ctx: bool,
    /// stack objects constructed in place (`T v(args);` at the constructor call)
    constructed: HashSet<VarId>,
    /// locals initialized once from constants with a brace initializer (`T v = {..};`), and
    /// those whose declaration already carries it
    brace_vars: HashSet<VarId>,
    brace_done: HashSet<VarId>,
    declared: HashSet<VarId>,
    /// locals declared as object arrays (`T a[n];`) whose lifted type is an untyped buffer
    obj_arrays: std::collections::HashMap<VarId, Type>,
    vt_count: usize,
    sret_local: Option<VarId>,
    /// Declared type of each global the emitter declares itself (one per symbol, even when the
    /// IR accesses it with several types).
    gtypes: std::collections::HashMap<String, Type>,
    /// Undeclared globals accessed as several scalar members: (offset, type) per member of the
    /// stand-in struct they are declared as (named-object accesses, which the compiler's alias
    /// analysis tells apart, unlike casts of its address).
    gstructs: std::collections::HashMap<String, Vec<(i32, Type)>>,
    /// Function-local statics (`init$90`) declared at the top of the body.
    local_statics: BTreeSet<String>,
    /// Forward declarations of functions the context doesn't declare (file-local statics,
    /// anonymous-namespace helpers): symbol -> (confidence, declaration).
    fn_decls: std::collections::BTreeMap<String, (u8, String)>,
    /// parameter types of the functions declared from a call's argument types
    fn_param_tys: std::collections::HashMap<String, Vec<Type>>,
    /// Types the context lacks (classes of the unit's own source), synthesized from their uses.
    synth: std::collections::BTreeMap<String, Synth>,
    /// Their rendered definitions (first in the preamble).
    type_defs: Vec<String>,
    /// the struct-return local is declared at its constructor call
    sret_ctor_decl: bool,
    /// type an unknown virtual call's result is used as (its stand-in's return type)
    vt_ret_hint: Option<Type>,
    /// the body assigns to `this`: it is copied into a local `self`
    this_local: bool,
    /// coercing the value of a store into a member (its declared type may be imprecise)
    member_store: bool,
    /// locals narrowed by their writes: self-updates as compound assignments
    narrowed: Vec<VarId>,
}

/// Run `f` (type spellings outside [`emit_function`], e.g. instantiation drafts) with the C-mode
/// spelling switches of a unit (`c_mode`: a `-lang=c` unit), restoring the C++ defaults after.
pub fn with_c_mode<R>(c_mode: bool, db: Option<&TypeDb>, f: impl FnOnce() -> R) -> R {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            types::C_MODE.with(|c| c.set(false));
            types::C_BOOL.with(|c| c.set(false));
        }
    }
    let _reset = Reset;
    types::C_MODE.with(|c| c.set(c_mode));
    types::C_BOOL.with(|c| {
        let uses_bool = |t: &Type| matches!(t.unqualified(), Type::Bool);
        c.set(c_mode && db.map_or(false, |db| db.decls.values().flatten().any(|d| uses_bool(&d.ret) || d.params.iter().any(|p| uses_bool(&p.ty)))));
    });
    f()
}

pub fn emit_function(ir: &IrFunction, db: Option<&TypeDb>, opts: &EmitOptions) -> Emitted {
    // locals typed after lifting (folded inline expansions) whose class can't be declared
    // without constructor arguments stay untyped buffers
    let mut fixed;
    let mut narrowed: Vec<VarId> = vec![];
    let ir = if !opts.c_mode && db.is_some() {
        fixed = ir.clone();
        let body = fixed.body.clone();
        mwdec_lift::idioms::untype_undeclarable(&body, &mut fixed.vars, db);
        let forwarded = mwdec_lift::postinline::forward_inline_args(&mut fixed.body, &fixed.vars);
        let forwarded = forwarded + mwdec_lift::postinline::reference_members(&mut fixed.body, &fixed.vars);
        let copies = mwdec_lift::structcopy::apply(&mut fixed.body, &fixed.vars, db);
        let copies = copies + mwdec_lift::postinline::address_temps_first(&mut fixed.body, &fixed.vars);
        let copies = copies + mwdec_lift::postinline::split_last_field(&mut fixed.body, &fixed.vars);
        let copies = copies + mwdec_lift::postinline::const_bool_locals(&mut fixed.body, &mut fixed.vars, matches!(fixed.sig.ret, Type::Bool));
        narrowed = mwdec_lift::postinline::narrow_by_defs(&mut fixed.body, &mut fixed.vars);
        let copies = copies + narrowed.len();
        if fixed.vars != ir.vars || forwarded > 0 || copies > 0 {
            &fixed
        } else {
            ir
        }
    } else {
        ir
    };
    // C has no nullptr
    let c_opts;
    let opts = if opts.c_mode && opts.null == "nullptr" {
        c_opts = EmitOptions { null: "0".into(), ..opts.clone() };
        &c_opts
    } else {
        opts
    };
    // The C-mode spelling switches are per emit: restore the C++ defaults on the way out, or
    // type spellings made later on this thread (instantiation drafts, other tools) would follow
    // the last function emitted (`bool` spelled `int` after a C unit's draft).
    struct ResetCMode;
    impl Drop for ResetCMode {
        fn drop(&mut self) {
            types::C_MODE.with(|c| c.set(false));
            types::C_BOOL.with(|c| c.set(false));
            types::C_TAGS.with(|t| t.borrow_mut().clear());
        }
    }
    let _reset = ResetCMode;
    types::C_MODE.with(|c| c.set(opts.c_mode));
    // C has no nullptr
    let c_opts;
    let opts = if opts.c_mode && opts.null == "nullptr" {
        c_opts = EmitOptions { null: "0".into(), ..opts.clone() };
        &c_opts
    } else {
        opts
    };
    types::C_BOOL.with(|c| {
        let uses_bool = |t: &Type| matches!(t.unqualified(), Type::Bool);
        c.set(opts.c_mode && db.map_or(false, |db| db.decls.values().flatten().any(|d| uses_bool(&d.ret) || d.params.iter().any(|p| uses_bool(&p.ty)))));
    });
    types::C_TAGS.with(|t| {
        let mut t = t.borrow_mut();
        t.clear();
        if let (true, Some(db)) = (opts.c_mode, db) {
            for (n, kw) in &db.tag_keywords {
                // a typedef of the same name makes the plain name valid too
                if !db.typedefs.contains_key(n) && kw != "class" {
                    t.insert(n.clone(), kw.clone());
                }
            }
        }
    });
    let new_em = || Em { ir, db, opts, out: String::new(), externs: BTreeSet::new(), lvalue_ctx: false, constructed: HashSet::new(), brace_vars: HashSet::new(), brace_done: HashSet::new(), declared: HashSet::new(), obj_arrays: Default::default(), vt_count: 0, sret_local: None, gtypes: Default::default(), gstructs: Default::default(), local_statics: BTreeSet::new(), fn_decls: Default::default(), fn_param_tys: Default::default(), synth: Default::default(), type_defs: vec![], sret_ctor_decl: false, vt_ret_hint: None, this_local: false, member_store: false, narrowed: narrowed.clone() };
    // a static initializer: the global definitions it is generated from
    if ir.symbol.starts_with("__sinit_") {
        let mut em = new_em();
        em.collect_global_types();
        if let Some(defs) = em.sinit_defs() {
            let preamble = em.type_defs.iter().cloned().chain(em.externs.iter().cloned()).chain(em.fn_decls.values().map(|(_, d)| d.clone())).chain(string_pool_decl(ir)).collect::<Vec<_>>().join("\n");
            return Emitted { preamble: if preamble.is_empty() { preamble } else { preamble + "\n" }, body: defs };
        }
    }
    let mut em = new_em();
    em.function();
    em.standin_ret_def();
    // helper types the body uses (mwdec_lift::helpers) come first
    let helpers = mwdec_lift::helpers::definitions(&em.out);
    let preamble = helpers.into_iter().chain(em.type_defs.iter().cloned()).chain(em.externs.iter().cloned()).chain(em.fn_decls.values().map(|(_, d)| d.clone())).chain(string_pool_decl(ir)).collect::<Vec<_>>().join("
");
    Emitted { preamble: if preamble.is_empty() { preamble } else { preamble + "\n" }, body: em.out }
}

/// The unit's string literals that precede this function's in the target's string pool: the
/// compiler pools a unit's literals in order of appearance (sharing equal ones), so listing them
/// first puts the function's strings at their target offsets.
fn string_pool_decl(ir: &IrFunction) -> Option<String> {
    if ir.string_pool.is_empty() {
        return None;
    }
    let strs: Vec<String> = ir.string_pool.iter().map(|b| float::c_string(b)).collect();
    Some(format!("const char* __unit_strings[] = {{ {} }};", strs.join(", ")))
}

/// Demangle a data/function symbol to a C++ qualified name (`sZero__9CVector3f` ->
/// `CVector3f::sZero`); plain names are returned unchanged. Anonymous-namespace scopes
/// (`@unnamed@File_cpp@::x`) are dropped (the name is visible unqualified in the TU).
pub fn symbol_name(sym: &str) -> String {
    let n = if let Some(d) = sig::demangle(sym) {
        // functions: strip the parameter list
        if d.ends_with(')') || d.ends_with(") const") {
            sig::parse_demangled(&d, None).qualified_name
        } else {
            d
        }
    } else {
        sym.to_string()
    };
    let n = types::split_closers(&strip_unnamed_ns(&n));
    // compiler-generated statics (`@123`) have no C++ spelling
    match n.strip_prefix('@') {
        Some(rest) => format!("_at_{}", rest.replace(|c: char| !c.is_ascii_alphanumeric(), "_")),
        None => n,
    }
}

pub(crate) fn strip_unnamed_ns(n: &str) -> String {
    let mut s = n.to_string();
    while let Some(p) = s.find("@unnamed@") {
        match s[p + 9..].find("@::") {
            Some(e) => s.replace_range(p..p + 9 + e + 3, ""),
            // the scope alone (`@unnamed@CFoo_cpp@`): nothing
            None if s[p + 9..].ends_with('@') && !s[p + 9..s.len() - 1].contains(['@', ':', '<', ',', ' ']) => {
                s.replace_range(p.., "");
                break;
            }
            None => break,
        }
    }
    s
}

/// `v->T(args)` constructor call statement on the object `v` points at.
fn is_ctor_on(s: &Stmt, v: VarId) -> bool {
    matches!(s, Stmt::Expr(Expr::Call { callee: Callee::Method { sig: sg, this, .. }, .. }) if sig::is_ctor(sg) && matches!(&**this, Expr::Var(x) if *x == v))
}

fn stmt_mentions(s: &Stmt, v: VarId) -> bool {
    mwdec_lift::idioms::stmt_mentions(s, v)
}

fn prec_of(e: &Expr) -> u8 {
    match e {
        Expr::Binary { op, .. } => op.prec(),
        Expr::Ternary { .. } => 3,
        Expr::AddrOf(inner) => match &**inner {
            Expr::Load { ty: Type::Unknown { size: 0 }, .. } | Expr::Member { ty: Type::Unknown { size: 0 }, .. } => 12,
            Expr::Index { ty: Type::Int { size: 1, .. }, .. } => 12,
            _ => 14,
        },
        Expr::Unary { .. } | Expr::Cast { .. } => 14,
        Expr::IncDec { post: true, .. } => 15,
        Expr::IncDec { post: false, .. } => 14,
        Expr::Load { .. } | Expr::Member { .. } => 15,
        Expr::Int { value, .. } if *value < 0 => 14,
        Expr::Float { .. } => 15,
        _ => 16,
    }
}

impl<'a> Em<'a> {
    fn vars(&self) -> &[Var] {
        &self.ir.vars
    }

    fn function(&mut self) {
        let ir = self.ir;
        if self.opts.warnings_comment && !ir.warnings.is_empty() {
            self.out.push_str("// mwdec warnings:\n");
            for w in ir.warnings.iter().take(12) {
                let _ = writeln!(self.out, "//   {}", w.replace('\n', " "));
            }
        }
        self.collect_global_types();
        self.synth_own_class();
        self.synth_unknown_types();
        if { let l = sig::split_scope(&ir.sig.qualified_name).1; strip_template_args(l) != l } && ir.sig.this_class.is_none() {
            let own = ir.sig.clone();
            self.declare_function(&ir.symbol, Some(&own), None, 1);
        }
        if let Some(t) = ir.this_var {
            let mut assigned = false;
            let mut chk = |b: &[Stmt]| {
                fn walk(b: &[Stmt], t: VarId, out: &mut bool) {
                    for s in b {
                        match s {
                            Stmt::Assign { dst: Expr::Var(x), .. } if *x == t => *out = true,
                            Stmt::If { then, els, .. } => {
                                walk(then, t, out);
                                walk(els, t, out);
                            }
                            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => walk(body, t, out),
                            Stmt::For { init, step, body, .. } => {
                                walk(init, t, out);
                                walk(step, t, out);
                                walk(body, t, out);
                            }
                            Stmt::Switch { cases, .. } => {
                                for c in cases {
                                    walk(&c.body, t, out);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                walk(b, t, &mut assigned);
            };
            chk(&ir.body);
            self.this_local = assigned;
        }
        let sigq = &ir.sig;
        let is_cdtor = sig::is_ctor(sigq) || sig::is_dtor(sigq);
        let mut head = String::new();
        if !is_cdtor {
            let rt = if matches!(sigq.ret, Type::Unknown { size: 0 }) { Type::Void } else { sigq.ret.clone() };
            // a return type nested in the class must be qualified outside the class scope
            let rt = match (&sigq.this_class, self.db) {
                (Some(c), Some(db)) => qualify_nested(&rt, c, db),
                _ => rt,
            };
            head.push_str(&type_str(&rt));
            head.push(' ');
        }
        // explicit specialization of a function template / member of a class template
        if strip_template_args(&sigq.qualified_name) != sigq.qualified_name {
            head.insert_str(0, "template <> ");
        }
        // an unmangled symbol in a C++ unit has C linkage (every C++ function is mangled)
        if !self.opts.c_mode && sigq.this_class.is_none() && sig::demangle(&ir.symbol).is_none() && !ir.symbol.starts_with("__sinit_") && !ir.symbol.contains('@') {
            head.insert_str(0, "extern \"C\" ");
        }
        // a function of a namespace the context doesn't declare it in (`std::terminate` without
        // <exception>): define it inside the namespace
        let qn_clean = strip_unnamed_ns(&sigq.qualified_name);
        let wrap_ns: Option<String> = match (sig::split_scope(&qn_clean), self.db) {
            ((Some(sc), _), Some(db))
                if sigq.this_class.is_none()
                    && !sc.contains('<')
                    && sig::find_class(db, sc).is_none()
                    && !db.templates.contains_key(sc)
                    && !db.decls.contains_key(&qn_clean) =>
            {
                Some(sc.to_string())
            }
            _ => None,
        };
        match &wrap_ns {
            Some(_) => head.push_str(&types::split_closers(sig::split_scope(&qn_clean).1)),
            None => head.push_str(&types::split_closers(&qn_clean)),
        }
        head.push('(');
        let mut ps = vec![];
        for (i, &v) in ir.params.iter().enumerate() {
            let name = &ir.vars[v].name;
            let mut spelled = ir.decl_params.get(i).cloned().unwrap_or_default();
            // (a header's top-level `const` goes when the body writes the parameter)
            if spelled.starts_with("const ") && !spelled.contains('*') && !spelled.contains('&') && param_written(&ir.body, v) {
                spelled = spelled["const ".len()..].to_string();
            }
            // no mangled spelling (C functions): the header declaration's parameter type
            let declared = ir.sig.params.get(i).map(|p| &p.ty).filter(|t| !matches!(t, Type::Unknown { .. }));
            let p = if !spelled.is_empty() {
                // (`>>` closing nested template argument lists is a shift token in C++03)
                spell_param(&types::split_closers(&strip_unnamed_ns(&spelled)), name)
            } else if let Some(t) = declared.filter(|_| sig::demangle(&ir.symbol).is_none()) {
                decl(t, name)
            } else {
                decl(&ir.vars[v].ty, name)
            };
            ps.push(p);
        }
        if ir.decl_params.last().map_or(false, |s| s == "...") {
            ps.push("...".into());
        }
        head.push_str(&ps.join(", "));
        head.push(')');
        if sigq.is_const {
            head.push_str(" const");
        }
        let missing = self.missing_member_inits();
        if !ir.init_list.is_empty() || !missing.is_empty() {
            let mut parts = vec![];
            for init in &ir.init_list {
                // stack objects passed to member constructors are temporaries (`rmemory_allocator()`):
                // the body's locals don't exist yet
                let stack_obj = |a: &Expr| {
                    match a {
                        Expr::Var(v) => Some(*v),
                        Expr::AddrOf(x) => match &**x {
                            Expr::Var(v) => Some(*v),
                            _ => None,
                        },
                        _ => None,
                    }
                    .filter(|v| matches!(ir.vars[*v].kind, VarKind::Stack { .. }))
                };
                let a = match (&init.member_ty, init.args.as_slice()) {
                    (Some(t), [x]) if stack_obj(x).is_none() => self.coerce(x, t),
                    _ => {
                        let mut parts_a = vec![];
                        for (i, a) in init.args.iter().enumerate() {
                            let pt = init.ctor.as_ref().and_then(|s| s.params.get(i)).map(|p| p.ty.clone());
                            let s = match (stack_obj(a), &pt) {
                                (Some(v), _) if named(&ir.vars[v].ty).is_some() => format!("{}()", type_str(strip_cv(&ir.vars[v].ty))),
                                (Some(_), Some(t)) if pointee(t).and_then(|x| named(x)).is_some() => format!("{}()", type_str(strip_cv(pointee(t).unwrap()))),
                                (Some(_), Some(t)) if named(strip_cv(t)).is_some() => format!("{}()", type_str(strip_cv(t))),
                                (_, Some(t)) => self.coerce(a, t),
                                _ => self.expr(a, 0),
                            };
                            parts_a.push(s);
                        }
                        parts_a.join(", ")
                    }
                };
                match &init.target {
                    InitTarget::Base(b) => parts.push(format!("{b}({a})")),
                    InitTarget::Member(m) => parts.push(format!("{m}({a})")),
                }
            }
            parts.extend(missing);
            head.push_str(" : ");
            head.push_str(&parts.join(", "));
        }
        self.out.push_str(&head);
        self.out.push_str(" {\n");
        // local declarations
        let mut used: BTreeSet<VarId> = BTreeSet::new();
        Stmt::walk_exprs(&ir.body, &mut |e| {
            if let Expr::Var(v) = e {
                used.insert(*v);
            }
        });
        // locals declared for their frame slots only
        for (v, var) in ir.vars.iter().enumerate() {
            if var.kind == VarKind::Local && var.name.starts_with(mwdec_lift::UNUSED_LOCAL_PREFIX) {
                used.insert(v);
            }
        }
        // C89 (c_mode) only allows declarations at the start of a block
        self.constructed = if self.opts.c_mode { HashSet::new() } else { mwdec_lift::idioms::decl_at_first_def(&ir.body, &ir.vars) };
        // local arrays of objects are declared where they are constructed
        if !self.opts.c_mode {
            fn walk(me: &mut Em, b: &[Stmt]) {
                for s in b {
                    if let Some((v, _, _)) = me.array_construction(s) {
                        me.constructed.insert(v);
                    }
                    match s {
                        Stmt::If { then, els, .. } => {
                            walk(me, then);
                            walk(me, els);
                        }
                        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => walk(me, body),
                        Stmt::Switch { cases, .. } => {
                            for c in cases {
                                walk(me, &c.body);
                            }
                        }
                        _ => {}
                    }
                }
            }
            walk(self, &ir.body);
        }
        // struct return through the hidden pointer that idioms couldn't fold into `return T(...)`:
        // build the object in a local and return it
        self.sret_local = used.iter().copied().find(|&v| ir.vars[v].kind == VarKind::StructRet);
        if let Some(v) = self.sret_local {
            // constructed by a top-level constructor call before any other use: declare it there
            let ctor_at = ir.body.iter().position(|s| is_ctor_on(s, v));
            let first_use = ir.body.iter().position(|s| stmt_mentions(s, v));
            // (a base class's constructor of an expanded inline constructor builds only part
            // of the returned object: not its declaration)
            // (typedefs resolved: a typedef of a container is the container its constructor builds)
            let ret_cls = pointee(&ir.vars[v].ty).map(|t| mwdec_lift::types::resolve(self.db, strip_cv(t)).into_owned()).and_then(|t| named(strip_cv(&t)).map(sig::norm_name));
            let ctor_cls = ctor_at.and_then(|k| match &ir.body[k] {
                Stmt::Expr(Expr::Call { callee: Callee::Method { sig: sg, .. }, .. }) => sg.this_class.as_deref().map(sig::norm_name),
                _ => None,
            });
            // (or a class the returned one converts from: a base with a converting constructor)
            let converts = |db: &TypeDb, r: &str, c: &str| -> bool {
                let key = strip_template_args(r);
                let last = sig::split_scope(&key).1.to_string();
                db.decls.get(&format!("{key}::{last}")).is_some_and(|ds| {
                    ds.iter().any(|dd| {
                        dd.params.len() == 1 && {
                            let pt = match strip_cv(&dd.params[0].ty) {
                                Type::Ref(x) => strip_cv(x).clone(),
                                t => t.clone(),
                            };
                            named(&pt).is_some_and(|n| sig::norm_name(&strip_template_args(n)) == sig::norm_name(&strip_template_args(c)))
                        }
                    })
                })
            };
            let whole = ret_cls.is_none()
                || ctor_cls == ret_cls
                || matches!((self.db, &ret_cls, &ctor_cls), (Some(db), Some(r), Some(c)) if converts(db, r, c));
            if ctor_at.is_some() && ctor_at == first_use && !self.opts.c_mode && whole {
                self.sret_ctor_decl = true;
            } else if let Some(t) = pointee(&ir.vars[v].ty) {
                let t = local_type(t);
                if self.default_constructible(&t) || self.opts.c_mode {
                    let _ = writeln!(self.out, "{}{};", self.opts.indent, decl(&t, "__return_value"));
                } else {
                    // no default constructor: the object is built member-wise in raw storage
                    let ts = type_str(&t);
                    let _ = writeln!(self.out, "{}unsigned char __return_storage[sizeof({ts})];", self.opts.indent);
                    let _ = writeln!(self.out, "{}{ts}& __return_value = *({ts}*)__return_storage;", self.opts.indent);
                }
            }
        }
        if self.this_local {
            if let Some(t) = ir.this_var {
                let _ = writeln!(self.out, "{}{} = this;", self.opts.indent, decl(&local_type(&ir.vars[t].ty), "self"));
            }
        }
        // MWCC colours named locals in reverse declaration order: the first declared gets r31.
        // Declare register-derived locals by target register, highest first (GPRs, then FPRs).
        let mut order: Vec<VarId> = used.iter().copied().collect();
        // the first declared named local is coloured first and takes the first register in
        // allocation preference: r31, r30, ... for callee-saved, r0, r3, r4, ... for volatile.
        // Stack objects: MWCC assigns frame offsets in reverse declaration order.
        order.sort_by_key(|&v| {
            let (class, reg) = reg_of_name(&ir.vars[v].name);
            if class < 2 && reg >= 14 {
                (0u8, class as i32, -(reg as i32), v)
            } else if class < 2 {
                (1u8, class as i32, reg as i32, v)
            } else if let VarKind::Stack { offset, .. } = ir.vars[v].kind {
                (2u8, 0, -offset, v)
            } else {
                (3u8, 0, 0, v)
            }
        });
        self.brace_vars = self.find_brace_inits();
        for &v in &order {
            let var = &ir.vars[v];
            if matches!(var.kind, VarKind::This | VarKind::Param { .. } | VarKind::StructRet | VarKind::Hidden) {
                continue;
            }
            if self.constructed.contains(&v) {
                continue;
            }
            if self.brace_vars.contains(&v) {
                if let Some(init) = self.brace_init_of(v) {
                    let d = decl(&local_type(&var.ty), &var.name);
                    let _ = writeln!(self.out, "{}{} = {};", self.opts.indent, d, init);
                    self.brace_done.insert(v);
                    continue;
                }
            }
            let d = self.local_decl(v);
            let _ = writeln!(self.out, "{}{};", self.opts.indent, d);
        }
        let body = ir.body.clone();
        let body_start = self.out.len();
        self.block(&body, 1);
        self.out.push_str("}\n");
        if !self.local_statics.is_empty() {
            let decls: String = self.local_statics.iter().map(|d| format!("{}{d}\n", self.opts.indent)).collect();
            // before the local declarations
            let at = self.out[..body_start].find(" {\n").map(|p| p + 3).unwrap_or(body_start);
            self.out.insert_str(at, &decls);
        }
        // a free function of the anonymous namespace is defined inside it
        let anon = sig::split_scope(&sigq.qualified_name).0.is_some_and(|s| s.starts_with("@unnamed@") && strip_unnamed_ns(s).is_empty());
        let wrap = match (&wrap_ns, anon) {
            (Some(ns), _) => Some(ns.split("::").map(|p| format!("namespace {p} {{\n")).collect::<String>()).map(|o| (o, ns.split("::").map(|_| "}\n").collect::<String>())),
            (None, true) => Some(("namespace {\n".to_string(), "}\n".to_string())),
            _ => None,
        };
        if let Some((open, close)) = wrap {
            // (after any warning comment block, before the definition)
            let at = self.out.lines().take_while(|l| l.starts_with("//")).map(|l| l.len() + 1).sum::<usize>().min(self.out.len());
            self.out.insert_str(at, &open);
            self.out.push_str(&close);
        }
    }

    /// The function's own class isn't in the context (a class defined in the unit's source file):
    /// declare it with this method, so the definition compiles. Its base is the class whose
    /// constructor/destructor the function runs on `this`; it is polymorphic when the function
    /// stores its vtable.
    fn synth_own_class(&mut self) {
        let Some(db) = self.db else { return };
        let ir = self.ir;
        // (static members can't be told from namespace functions: only methods)
        let Some(cls) = ir.sig.this_class.clone() else { return };
        // (a class of the anonymous namespace is named without it)
        let cls = if cls.starts_with("@unnamed@") { strip_unnamed_ns(&cls) } else { cls };
        // (a lower-case scope may be a namespace, `std`: not when it has variables)
        let ns_vars = ir.globals.iter().any(|g| sig::demangle(&g.symbol).map_or(false, |d| sig::split_scope(&d).0 == Some(cls.as_str())));
        // (a class its enclosing class only declares can be defined out of line)
        let declared_nested = cls.contains("::") && sig::find_class(db, &cls).is_some_and(|c| c.is_declaration) && sig::split_scope(&cls).0.is_some_and(|sc| sig::find_class(db, sc).is_some_and(|p| !p.is_declaration));
        if (cls.chars().next().map_or(true, |c| c.is_ascii_lowercase()) && ns_vars) || (cls.contains("::") && !declared_nested) || cls.contains('<') || cls.contains('@') || sig::find_class(db, &cls).is_some_and(|c| !c.is_declaration || !declared_nested) || db.namespaces.contains(&cls) || db.templates.contains_key(&cls) {
            return;
        }
        let mut base: Option<String> = None;
        let mut poly = false;
        let this = ir.this_var;
        Stmt::walk_exprs(&ir.body, &mut |e| match e {
            Expr::Call { callee: Callee::Method { sig: s, this: o, .. }, .. } if (sig::is_ctor(s) || sig::is_dtor(s)) && matches!(&**o, Expr::Var(v) if Some(*v) == this) => {
                if let Some(c) = &s.this_class {
                    if sig::norm_name(c) != sig::norm_name(&cls) && sig::find_class(db, c).is_some() {
                        base.get_or_insert(c.clone());
                    }
                }
            }
            // a direct call of another class's method on `this` (`this->Base::Method(...)`):
            // that class is a base
            Expr::Call { callee: Callee::Method { sig: s, this: o, .. }, .. } if matches!(&**o, Expr::Var(v) if Some(*v) == this) => {
                if let Some(c) = &s.this_class {
                    if sig::norm_name(c) != sig::norm_name(&cls) && sig::find_class(db, c).is_some() {
                        base.get_or_insert(c.clone());
                    }
                }
            }
            Expr::Global { symbol, .. } if symbol.starts_with("__vt__") => poly = true,
            _ => {}
        });
        for init in &ir.init_list {
            if let InitTarget::Base(b) = &init.target {
                base.get_or_insert(b.clone());
            }
        }
        for b in ir.implicit_bases.iter().filter(|b| sig::find_class(db, b).is_some()) {
            base.get_or_insert(b.clone());
        }
        if ir.globals.iter().any(|g| g.symbol.starts_with("__vt__") && g.symbol.ends_with(&cls)) {
            poly = true;
        }
        let name = sig::split_scope(&ir.sig.qualified_name).1.to_string();
        let mut ps: Vec<String> = ir.decl_params.iter().filter(|p| p.as_str() != "...").map(|p| strip_unnamed_ns(p)).collect();
        if ps.is_empty() {
            ps = ir.params.iter().map(|&v| type_str(&ir.vars[v].ty)).collect();
        }
        if ir.decl_params.last().map_or(false, |s| s == "...") {
            ps.push("...".into());
        }
        let is_cdtor = sig::is_ctor(&ir.sig) || sig::is_dtor(&ir.sig);
        let ret = if is_cdtor {
            String::new()
        } else {
            let rt = if matches!(ir.sig.ret, Type::Unknown { size: 0 }) { Type::Void } else { ir.sig.ret.clone() };
            format!("{} ", type_str(&rt))
        };
        let mut m = format!("{ret}{name}({})", ps.join(", "));
        if ir.sig.is_const {
            m.push_str(" const");
        }
        if this.is_none() {
            m = format!("static {m}");
        } else if poly && (sig::is_dtor(&ir.sig) || base.is_some() && !sig::is_ctor(&ir.sig)) {
            m = format!("virtual {m}");
        }
        // variables of the same scope the context doesn't declare: its static members (a
        // mangled name doesn't tell a class from a namespace; one spelling must serve both)
        let mut statics = vec![];
        for g in ir.globals.iter().filter(|g| !g.is_function && !db.globals.contains_key(&g.symbol)) {
            let Some(d) = sig::demangle(&g.symbol) else { continue };
            if d.contains('(') {
                continue;
            }
            let (sc, n) = sig::split_scope(&d);
            if sc == Some(cls.as_str()) {
                let t = self.gtypes.get(&g.symbol).cloned().unwrap_or_else(|| extern_type(&g.ty));
                statics.push((n.to_string(), format!("static {}", decl(&t, n))));
            }
        }
        let e = self.synth.entry(cls.clone()).or_default();
        e.base = base;
        e.add_method(&format!("{name}({})", ps.join(", ")), m);
        for (n, d) in statics {
            e.add_method(&format!("static {n}"), d);
        }
    }

    /// Is `n` a type the context doesn't have (so a minimal definition must be synthesized)?
    fn unknown_type(&self, db: &TypeDb, n: &str) -> bool {
        if n.is_empty() || n.contains('<') || n.contains('(') || n.contains('@') || n.contains(' ') {
            return false;
        }
        let last = sig::split_scope(n).1;
        if last.len() <= 1 || last.starts_with('_') || last.starts_with("__") || is_c_keyword(last) {
            return false;
        }
        if sig::find_class(db, n).map_or(false, |c| !c.is_declaration) || db.typedefs.contains_key(n) || db.enums.contains_key(n) || db.templates.contains_key(n) || db.namespaces.contains(n) {
            return false;
        }
        // nested in a class the context defines: can't be added from outside, unless that
        // class declares it (`class Inner;`): then it can be defined out of line
        if let (Some(sc), Some(c)) = (sig::split_scope(n).0, sig::find_class(db, n)) {
            if c.is_declaration && sig::find_class(db, sc).is_some_and(|p| !p.is_declaration) {
                return true;
            }
        }
        if let Some(sc) = sig::split_scope(n).0 {
            if sig::find_class(db, sc).is_some() || db.templates.contains_key(&strip_template_args(sc)) || !(db.namespaces.contains(sc) || self.unknown_type(db, sc) || sc.chars().next().is_some_and(|c| c.is_ascii_lowercase())) {
                return false;
            }
        }
        true
    }

    /// Minimal definitions for types the function uses that the context lacks (types of the
    /// unit's own source: functors, helper structs, local classes): every method it calls on
    /// them, nested types it names, and their size when a local of the type shows it.
    fn synth_unknown_types(&mut self) {
        let Some(db) = self.db else { return };
        if self.opts.c_mode {
            return;
        }
        let ir = self.ir;
        let mut chains: BTreeSet<String> = BTreeSet::new();
        let mut unnamed: BTreeSet<String> = BTreeSet::new();
        let mut note = |s: &str, chains: &mut BTreeSet<String>| {
            for c in type_chains(s) {
                if s.contains("@unnamed@") {
                    unnamed.insert(c.clone());
                }
                chains.insert(c);
            }
        };
        fn tnames(t: &Type, out: &mut Vec<String>) {
            match t {
                Type::Named(n) => out.push(n.clone()),
                Type::Ptr(x) | Type::Ref(x) | Type::Const(x) | Type::Volatile(x) | Type::Array(x, _) => tnames(x, out),
                Type::FuncPtr(s) => {
                    tnames(&s.ret, out);
                    for p in &s.params {
                        tnames(&p.ty, out);
                    }
                }
                _ => {}
            }
        }
        // function names: only their scope and template arguments name types
        let fn_types = |qn: &str| -> Vec<String> {
            let (scope, last) = sig::split_scope(qn);
            let mut v: Vec<String> = scope.map(|s| s.to_string()).into_iter().collect();
            if let Some(lt) = last.find('<') {
                v.push(last[lt..].to_string());
            }
            v
        };
        // (a free function's scope is a namespace)
        let mut names: Vec<String> = fn_types(&ir.sig.qualified_name).into_iter().filter(|n| ir.sig.this_class.is_some() || n.starts_with('<')).collect();
        for v in &ir.vars {
            tnames(&v.ty, &mut names);
        }
        for p in &ir.sig.params {
            tnames(&p.ty, &mut names);
        }
        tnames(&ir.sig.ret, &mut names);
        names.extend(ir.decl_params.iter().cloned());
        // methods called on (or static members of) unknown classes: (class, key, decl)
        let mut calls: Vec<(String, String, String)> = vec![];
        Stmt::walk_exprs(&ir.body, &mut |e| match e {
            Expr::Call { callee: Callee::Method { sig: s, .. } | Callee::Direct { sig: s, .. }, ret, .. } => {
                names.extend(fn_types(&s.qualified_name));
                for p in &s.params {
                    tnames(&p.ty, &mut names);
                }
                tnames(ret, &mut names);
                if let Some(c) = &s.this_class {
                    let is_static = matches!(e, Expr::Call { callee: Callee::Direct { .. }, .. });
                    let name = sig::split_scope(&strip_unnamed_ns(&s.qualified_name)).1.to_string();
                    let ps: Vec<String> = s.params.iter().map(|p| type_str(&p.ty)).collect();
                    let key = format!("{name}({})", ps.join(", "));
                    let mut d = if sig::is_ctor(s) || sig::is_dtor(s) {
                        format!("{name}({})", ps.join(", "))
                    } else {
                        let rt = if !sig::ret_unknown(s) { s.ret.clone() } else { value_type(&local_type(ret)) };
                        format!("{} {name}({})", type_str(&rt), ps.join(", "))
                    };
                    if s.is_const {
                        d.push_str(" const");
                    }
                    if is_static {
                        d = format!("static {d}");
                    }
                    calls.push((strip_unnamed_ns(c), key, d));
                }
            }
            Expr::New { class, .. } | Expr::Construct { class, .. } | Expr::Cast { ty: class, .. } | Expr::Load { ty: class, .. } | Expr::Member { ty: class, .. } => tnames(class, &mut names),
            _ => {}
        });
        for n in &names {
            note(n, &mut chains);
        }
        let mut wanted: BTreeSet<String> = chains.into_iter().filter(|c| self.unknown_type(db, c) && !mwdec_lift::helpers::is_helper(c)).collect();
        for (c, _, _) in &calls {
            if self.unknown_type(db, c) {
                wanted.insert(c.clone());
            }
        }
        // the own class (synth_own_class) is wanted too
        wanted.extend(self.synth.keys().cloned());
        if wanted.is_empty() {
            return;
        }
        for (c, key, d) in calls {
            if wanted.contains(&c) {
                self.synth.entry(c).or_default().add_method(&key, d);
            }
        }
        for v in &ir.vars {
            if let (VarKind::Stack { size, .. }, Type::Named(n)) = (&v.kind, &v.ty) {
                let n = strip_unnamed_ns(n);
                if wanted.contains(&n) {
                    let e = self.synth.entry(n).or_default();
                    e.size = e.size.max(*size);
                }
            }
        }
        // nested names (`A::B` with `A` synthesized) live inside their parent
        let all: Vec<String> = wanted.iter().cloned().collect();
        let is_nested = |n: &str| sig::split_scope(n).0.map_or(false, |sc| all.iter().any(|a| a == sc));
        for n in &all {
            self.synth.entry(n.clone()).or_default();
        }
        let render = |me: &Self, n: &str| -> String { me.render_synth(n, &all) };
        let mut defs = vec![];
        let mut fwd = vec![];
        for n in all.iter().filter(|n| !is_nested(n)) {
            let (scope, last) = sig::split_scope(n);
            let body = render(self, n);
            // a nested class its (defined) enclosing class only declares: defined out of line
            if scope.is_some_and(|sc| sig::find_class(db, sc).is_some_and(|p| !p.is_declaration)) {
                defs.push(body.replacen(&format!("struct {last}"), &format!("struct {n}"), 1));
                continue;
            }
            let mut f = format!("struct {last};");
            let mut d = body;
            if let Some(sc) = scope {
                for part in sc.split("::").collect::<Vec<_>>().into_iter().rev() {
                    f = format!("namespace {part} {{ {f} }}");
                    d = format!("namespace {part} {{ {d} }}");
                }
            }
            if unnamed.contains(n) {
                f = format!("namespace {{ {f} }}");
                d = format!("namespace {{ {d} }}");
            }
            fwd.push(f);
            defs.push(d);
        }
        self.type_defs = fwd.into_iter().chain(defs).collect();
    }

    /// Locals whose only definition is a constant aggregate `Construct` without a constructor
    /// (a POD struct initialized from a literal object).
    /// A scalar frame slot the target stores and reloads although its address is never taken.
    /// MWCC keeps plain locals in registers, so the source forced memory (an inline object such
    /// as a scoped lock, or `volatile`); `volatile` reproduces the store and the reloads.
    fn memory_scalar(&self, v: VarId) -> bool {
        let var = &self.ir.vars[v];
        if !matches!(var.kind, VarKind::Stack { .. }) || !var.name.starts_with("local_") {
            return false;
        }
        let scalar = match &var.ty {
            Type::Unknown { size } => matches!(size, 1 | 2 | 4),
            Type::Int { .. } | Type::Bool | Type::Char | Type::Long { .. } | Type::Ptr(_) => true,
            _ => false,
        };
        if !scalar {
            return false;
        }
        fn assigns(b: &[Stmt], v: VarId) -> usize {
            b.iter()
                .map(|s| match s {
                    Stmt::Assign { dst: Expr::Var(w), .. } if *w == v => 1,
                    Stmt::If { then, els, .. } => assigns(then, v) + assigns(els, v),
                    Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => assigns(body, v),
                    Stmt::For { init, step, body, .. } => assigns(init, v) + assigns(step, v) + assigns(body, v),
                    Stmt::Switch { cases, .. } => cases.iter().map(|c| assigns(&c.body, v)).sum(),
                    _ => 0,
                })
                .sum()
        }
        let (mut uses, mut addressed) = (0usize, false);
        Stmt::walk_exprs(&self.ir.body, &mut |e| match e {
            Expr::Var(w) if *w == v => uses += 1,
            Expr::AddrOf(x) | Expr::Member { base: x, .. } | Expr::BitField { base: x, .. } if matches!(&**x, Expr::Var(w) if *w == v) => addressed = true,
            _ => {}
        });
        let defs = assigns(&self.ir.body, v);
        !addressed && defs > 0 && uses > defs
    }

    /// Size of a local `local_decl` declares as `unsigned char x[N]`.
    fn byte_array_local(&self, v: VarId) -> Option<u32> {
        let var = &self.ir.vars[v];
        match &var.ty {
            Type::Unknown { size } if *size != 4 && *size != 2 && *size != 1 && *size != 8 => Some((*size).max(1)),
            Type::Unknown { size: 8 } if matches!(var.kind, VarKind::Stack { offset, .. } if offset % 8 != 0) => Some(8),
            _ => None,
        }
    }

    /// Declaration of a local (without initializer).
    fn local_decl(&self, v: VarId) -> String {
        let var = &self.ir.vars[v];
        let d = match &var.ty {
            Type::Unknown { size } if *size != 4 && *size != 2 && *size != 1 && *size != 8 => {
                format!("unsigned char {}[{}]", var.name, size.max(&1))
            }
            // an 8-byte frame object off an 8-byte boundary is not a `long long` (that would
            // be realigned): a word-aligned byte array keeps its place
            Type::Unknown { size: 8 } if matches!(var.kind, VarKind::Stack { offset, .. } if offset % 8 != 0) => {
                format!("unsigned char {}[8]", var.name)
            }
            // a const scalar (declared at its definition)
            Type::Const(t) if matches!(**t, Type::Bool) && matches!(var.kind, VarKind::Local) => format!("const {}", decl(t, &var.name)),
            t => decl(&local_type(t), &var.name),
        };
        if self.memory_scalar(v) {
            format!("volatile {d}")
        } else {
            d
        }
    }

    fn find_brace_inits(&self) -> HashSet<VarId> {
        let mut defs: std::collections::HashMap<VarId, (usize, bool)> = Default::default();
        fn visit(b: &[Stmt], defs: &mut std::collections::HashMap<VarId, (usize, bool)>) {
            for s in b {
                match s {
                    Stmt::Assign { dst, src } => {
                        let v = match dst {
                            Expr::Var(v) => Some(*v),
                            Expr::Member { base, .. } => match **base {
                                Expr::Var(v) => Some(v),
                                _ => None,
                            },
                            _ => None,
                        };
                        if let Some(v) = v {
                            let e = defs.entry(v).or_insert((0, false));
                            e.0 += 1;
                            e.1 = matches!(dst, Expr::Var(_)) && matches!(src, Expr::Construct { ctor: None, .. });
                        }
                    }
                    Stmt::If { then, els, .. } => {
                        visit(then, defs);
                        visit(els, defs);
                    }
                    Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => visit(body, defs),
                    Stmt::For { init, step, body, .. } => {
                        visit(init, defs);
                        visit(step, defs);
                        visit(body, defs);
                    }
                    Stmt::Switch { cases, .. } => {
                        for c in cases {
                            visit(&c.body, defs);
                        }
                    }
                    _ => {}
                }
            }
        }
        visit(&self.ir.body, &mut defs);
        defs.into_iter()
            .filter(|(v, (n, c))| *n == 1 && *c && matches!(self.ir.vars[*v].kind, VarKind::Stack { .. }) && self.pod_aggregate(&self.ir.vars[*v].ty))
            .map(|(v, _)| v)
            .collect()
    }

    /// `{a, b, c}` of the constant initializer of `v`.
    fn brace_init_of(&mut self, v: VarId) -> Option<String> {
        fn find(b: &[Stmt], v: VarId) -> Option<Vec<Expr>> {
            for s in b {
                match s {
                    Stmt::Assign { dst: Expr::Var(x), src: Expr::Construct { args, .. } } if *x == v => return Some(args.clone()),
                    Stmt::If { then, els, .. } => {
                        if let Some(a) = find(then, v).or_else(|| find(els, v)) {
                            return Some(a);
                        }
                    }
                    Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::For { body, .. } => {
                        if let Some(a) = find(body, v) {
                            return Some(a);
                        }
                    }
                    Stmt::Switch { cases, .. } => {
                        for c in cases {
                            if let Some(a) = find(&c.body, v) {
                                return Some(a);
                            }
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        let args = find(&self.ir.body, v)?;
        let parts: Vec<String> = args.iter().map(|a| self.expr(a, 0)).collect();
        Some(format!("{{{}}}", parts.join(", ")))
    }

    /// A class that can take a brace initializer: no constructors, virtuals or bases.
    fn pod_aggregate(&self, t: &Type) -> bool {
        let Some(db) = self.db else { return false };
        let r = mwdec_lift::types::resolve(Some(db), t).into_owned();
        let Some(cls) = named(&r) else { return false };
        let Some(c) = sig::find_class(db, cls) else { return false };
        if c.vptr_offset.is_some() || !c.bases.is_empty() || c.is_declaration {
            return false;
        }
        if self.opts.c_mode {
            return true;
        }
        let base = strip_template_args(cls);
        let last = sig::split_scope(&base).1.to_string();
        !db.decls.contains_key(&format!("{base}::{last}"))
    }

    /// `struct Last [: public Base] { nested...; methods...; pad };`
    /// The stand-in class of a guessed struct return (`mwdec_lift::idioms::standin_sret`): one
    /// member per constructor parameter, a constructor storing them.
    fn standin_ret_def(&mut self) {
        let Some(n) = named(&self.ir.sig.ret).filter(|n| n.starts_with(mwdec_lift::idioms::STANDIN_RET)).map(|n| n.to_string()) else { return };
        let mut sig = None;
        Stmt::walk_exprs(&self.ir.body, &mut |e| {
            if let Expr::Construct { class, ctor: Some(c), .. } = e {
                if named(class) == Some(n.as_str()) && sig.is_none() {
                    sig = Some(c.clone());
                }
            }
        });
        let Some(sig) = sig else { return };
        let members: Vec<String> = sig.params.iter().enumerate().map(|(i, p)| format!("{};", decl(&p.ty, &format!("m{i}")))).collect();
        let params: Vec<String> = sig.params.iter().enumerate().map(|(i, p)| decl(&p.ty, &format!("a{i}"))).collect();
        let inits: Vec<String> = (0..sig.params.len()).map(|i| format!("m{i}(a{i})")).collect();
        self.type_defs.push(format!("struct {n} {{ {} {n}({}) : {} {{}} }};", members.join(" "), params.join(", "), inits.join(", ")));
    }

    fn render_synth(&self, n: &str, all: &[String]) -> String {
        let last = sig::split_scope(n).1;
        let s = self.synth.get(n).cloned().unwrap_or_default();
        let mut body = String::new();
        for c in all {
            if sig::split_scope(c).0 == Some(n) {
                body.push(' ');
                body.push_str(&self.render_synth(c, all));
            }
        }
        for (_, d) in &s.methods {
            body.push(' ');
            body.push_str(d);
            body.push(';');
        }
        // a destructor the unit's object defines or calls (`delete` calls it)
        if !s.methods.iter().any(|(k, _)| k.starts_with('~')) && self.db.is_some_and(|db| db.object_dtors.contains(n) || db.object_dtors.iter().any(|d| strip_unnamed_ns(d) == n)) {
            let _ = write!(body, " ~{last}();");
        }
        if s.size > 0 && s.base.is_none() {
            let _ = write!(body, " unsigned char __mwdec_data[{}];", s.size);
        }
        let b = s.base.map(|b| format!(" : public {b}")).unwrap_or_default();
        format!("struct {last}{b} {{{body} }};")
    }

    /// One declared type per global symbol: the single type it's used with, else the largest
    /// (byte buffers for mixed-size uses); other uses reinterpret it.
    fn collect_global_types(&mut self) {
        let mut seen: std::collections::HashMap<String, Vec<Option<Type>>> = Default::default();
        Stmt::walk_exprs(&self.ir.body, &mut |e| {
            if let Expr::Global { symbol, ty } = e {
                let v = seen.entry(symbol.clone()).or_default();
                // unknown-size (address-only) uses don't decide the type unless they're all
                let t = if matches!(ty, Type::Unknown { size: 0 }) { None } else { Some(extern_type(ty)) };
                if !v.contains(&t) {
                    v.push(t);
                }
            }
        });
        for (sym, ts) in seen {
            let mut ts: Vec<Type> = ts.into_iter().flatten().collect();
            if ts.is_empty() {
                ts.push(extern_type(&Type::Unknown { size: 0 }));
            }
            let t = if ts.len() == 1 {
                ts[0].clone()
            } else {
                let size = |t: &Type| mwdec_lift::types::size_of(self.db, t).unwrap_or(0);
                let max = ts.iter().map(size).max().unwrap_or(0);
                let all_max: Vec<&Type> = ts.iter().filter(|t| size(t) == max).collect();
                match all_max.iter().find(|t| matches!(t, Type::Array(..))) {
                    Some(t) => (*t).clone(),
                    None if all_max.len() == 1 => all_max[0].clone(),
                    None => Type::Array(Box::new(Type::Int { size: 1, signed: false }), max.max(1)),
                }
            };
            self.gtypes.insert(sym, t);
        }
        self.collect_global_structs();
    }

    /// Globals the context lacks that are only read and written as scalar members (not just the
    /// one at offset 0; never by address): a stand-in struct with one member per offset.
    fn collect_global_structs(&mut self) {
        // (ordered: the stand-ins are emitted in this order, and drafts must be deterministic)
        let mut fields: std::collections::BTreeMap<String, std::collections::BTreeMap<i32, Vec<Type>>> = Default::default();
        let mut total: std::collections::HashMap<String, usize> = Default::default();
        let mut covered: std::collections::HashMap<String, usize> = Default::default();
        Stmt::walk_exprs(&self.ir.body, &mut |e| match e {
            Expr::Global { symbol, .. } => *total.entry(symbol.clone()).or_default() += 1,
            Expr::Member { base, offset, ty } => {
                if let Expr::Global { symbol, .. } = &**base {
                    *covered.entry(symbol.clone()).or_default() += 1;
                    let v = fields.entry(symbol.clone()).or_default().entry(*offset).or_default();
                    if !v.contains(ty) {
                        v.push(ty.clone());
                    }
                }
            }
            _ => {}
        });
        for (sym, fs) in fields {
            // (one member at offset 0 is the object itself: nothing to name)
            if (fs.len() < 2 && fs.keys().all(|&o| o == 0)) || total.get(&sym) != covered.get(&sym) || !self.self_declared(&sym) || local_static_name(&sym).is_some() {
                continue;
            }
            if !sym.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                continue;
            }
            let mut members = vec![];
            let mut list = vec![];
            let mut at = 0i32;
            let mut ok = true;
            for (off, ts) in &fs {
                let [t] = ts.as_slice() else { ok = false; break };
                let t = extern_type(t);
                let sz = match strip_cv(&t) {
                    Type::Int { size, .. } => *size as i32,
                    Type::Float { size } => *size as i32,
                    Type::Bool | Type::Char => 1,
                    Type::Ptr(_) => 4,
                    _ => 0,
                };
                if sz == 0 || *off < at || off % sz != 0 {
                    ok = false;
                    break;
                }
                if *off > at {
                    members.push(format!("unsigned char _p{:x}[{}];", at, off - at));
                }
                members.push(format!("{};", decl(&t, &format!("f{:x}", off))));
                list.push((*off, t));
                at = off + sz;
            }
            if !ok {
                continue;
            }
            let name = format!("__mwdec_g_{sym}");
            // (keeps the object's size when its other uses showed it)
            if let Some(Type::Array(_, n)) = self.gtypes.get(&sym) {
                if (*n as i32) > at {
                    members.push(format!("unsigned char _p{:x}[{}];", at, *n as i32 - at));
                }
            }
            // (a typedef: the same spelling in C and C++ units)
            self.type_defs.push(format!("typedef struct {{ {} }} {name};", members.join(" ")));
            self.gtypes.insert(sym.clone(), Type::Named(name));
            self.gstructs.insert(sym, list);
        }
    }

    fn ind(&self, depth: usize) -> String {
        self.opts.indent.repeat(depth)
    }

    fn block(&mut self, stmts: &[Stmt], depth: usize) {
        let mut i = 0;
        while i < stmts.len() {
            // `v = f(); <implicit destructors>; return v;` is `return f();`: the value is
            // computed before the locals are destroyed (a named `v` would add a copy)
            if let Stmt::Assign { dst: Expr::Var(v), src } = &stmts[i] {
                let mut j = i + 1;
                while j < stmts.len() && matches!(&stmts[j], Stmt::Expr(x) if self.implicit_destruction(x)) {
                    j += 1;
                }
                if j > i + 1
                    && matches!(stmts.get(j), Some(Stmt::Return(Some(Expr::Var(w)))) if w == v)
                    && matches!(self.ir.vars[*v].kind, VarKind::Local)
                    && !src.uses_var(*v)
                {
                    let mut uses = std::collections::HashMap::new();
                    mwdec_lift::inline::count_uses(&self.ir.body, &mut uses);
                    if uses.get(v).copied().unwrap_or(0) == 1 {
                        for s in &stmts[i + 1..j] {
                            self.stmt(s, depth);
                        }
                        self.stmt(&Stmt::Return(Some(src.clone())), depth);
                        i = j + 1;
                        continue;
                    }
                }
            }
            // `U u(r); x = u;` with `r` a reference bound to a call result: `x = U(r);`. The
            // reference's temporary was kept apart because it lies above the object built from
            // it, which then has to be a temporary created after it (a named `u` comes first
            // in MWCC's frame object list).
            if let (Some((c, sg, args)), Some(Stmt::Assign { dst, src: Expr::Var(w) })) = (self.ctor_from_ref_binding(&stmts[i]), stmts.get(i + 1)) {
                let mut uses = std::collections::HashMap::new();
                mwdec_lift::inline::count_uses(&self.ir.body, &mut uses);
                if *w == c && !dst.uses_var(c) && uses.get(&c).copied().unwrap_or(0) <= 2 {
                    let class = Type::Named(sg.this_class.clone().unwrap_or_default());
                    let st = Stmt::Assign { dst: dst.clone(), src: Expr::Construct { class, ctor: Some(sg.clone()), args: args.to_vec() } };
                    self.stmt(&st, depth);
                    i += 2;
                    continue;
                }
            }
            self.stmt(&stmts[i], depth);
            i += 1;
        }
    }

    /// `&u->U(args)` on a frame object with an argument that is a frame temporary bound to a
    /// reference: (u, constructor, args).
    fn ctor_from_ref_binding<'s>(&self, s: &'s Stmt) -> Option<(VarId, &'s mwdec_core::FuncSig, &'s [Expr])> {
        let Stmt::Expr(Expr::Call { callee: Callee::Method { sig: sg, this, .. }, args, .. }) = s else { return None };
        if !sig::is_ctor(sg) || sg.this_class.is_none() {
            return None;
        }
        let Expr::AddrOf(x) = &**this else { return None };
        let Expr::Var(c) = &**x else { return None };
        if !matches!(self.ir.vars[*c].kind, VarKind::Stack { .. }) {
            return None;
        }
        let bound = |e: &Expr| {
            let mut hit = false;
            e.walk(&mut |y| {
                if let Expr::Var(r) = y {
                    if matches!(self.ir.vars[*r].kind, VarKind::Stack { .. }) && matches!(self.ir.vars[*r].ty, Type::Ref(_)) {
                        hit = true;
                    }
                }
            });
            hit
        };
        args.iter().any(bound).then_some((*c, sg, args.as_slice()))
    }

    /// `if (p) p->T(args);` with `p` not a frame object: the expansion of `new (p) T(args)`.
    /// Returns the object pointer, the arguments and the constructor.
    fn placement_new<'s>(&self, cond: &Expr, s: &'s Stmt) -> Option<(&'s Expr, &'s [Expr], &'s mwdec_core::FuncSig)> {
        let Stmt::Expr(Expr::Call { callee: Callee::Method { sig: sg, this, .. }, args, .. }) = s else { return None };
        if !sig::is_ctor(sg) || sg.this_class.is_none() {
            return None;
        }
        let tested = match cond {
            Expr::Binary { op: BinOp::Ne, l, r, .. } if r.as_int() == Some(0) => &**l,
            e => e,
        };
        fn strip(mut e: &Expr) -> &Expr {
            while let Expr::Cast { e: x, .. } = e {
                e = x;
            }
            e
        }
        // a frame object constructed in place is a declaration, not a placement new
        if strip(tested) != strip(this) || matches!(strip(this), Expr::AddrOf(_)) || pointee(&ty_of(this, self.vars())).is_none() {
            return None;
        }
        Some((&**this, args.as_slice(), sg))
    }

    fn stmt(&mut self, s: &Stmt, depth: usize) {
        let ind = self.ind(depth);
        match s {
            Stmt::Expr(e) if self.sret_ctor_decl && self.sret_local.map_or(false, |v| is_ctor_on(s, v)) => {
                self.sret_ctor_decl = false;
                if let Expr::Call { callee: Callee::Method { sig: sg, .. }, args, .. } = e {
                    let t = Type::Named(sg.this_class.clone().unwrap_or_default());
                    let a = self.args(args, Some(sg));
                    let a = if args.len() == 1 { vexing_parens(a) } else { a };
                    if a.is_empty() {
                        let _ = writeln!(self.out, "{ind}{} __return_value;", type_str(&t));
                    } else {
                        let _ = writeln!(self.out, "{ind}{} __return_value({a});", type_str(&t));
                    }
                }
            }
            // an inlined destructor body run on a local (or a member, in a destructor): implicit
            Stmt::Expr(e) if self.implicit_destruction(e) => {}
            // a local array of objects: `T a[n];` (its construction and destruction are implicit)
            Stmt::Expr(_) if self.array_construction(s).is_some() => {
                let (v, cls, n) = self.array_construction(s).unwrap();
                if !self.declared.contains(&v) {
                    self.declared.insert(v);
                    let name = self.ir.vars[v].name.clone();
                    let _ = writeln!(self.out, "{ind}{} {name}[{n}];", type_str(&Type::Named(cls.clone())));
                    self.obj_arrays.insert(v, Type::Array(Box::new(Type::Named(cls)), n as u32));
                }
            }
            // arrays of member objects in constructors/destructors, local arrays at scope end:
            // built and destroyed implicitly
            Stmt::Expr(Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. })
                if (symbol == "__construct_array" || symbol == "__destroy_arr") && args.first().is_some_and(|a| self.implicit_array(a, symbol == "__construct_array")) => {}
            Stmt::Expr(e) => {
                // in-place construction / implicit destruction of stack objects
                if let Expr::Call { callee: Callee::Method { sig: sg, this, .. }, args, .. } = e {
                    if let Expr::AddrOf(inner) = &**this {
                        if let Expr::Var(v) = &**inner {
                            if self.constructed.contains(v) {
                                if sig::is_ctor(sg) && !self.declared.contains(v) {
                                    self.declared.insert(*v);
                                    let var = &self.ir.vars[*v];
                                    let t = match &var.ty {
                                        Type::Named(_) => var.ty.clone(),
                                        _ => Type::Named(sg.this_class.clone().unwrap_or_default()),
                                    };
                                    let name = var.name.clone();
                                    let a = self.args(args, Some(sg));
                                    // (`T x(U(y));` would declare a function: `T x((U(y)));`)
                                    let a = if args.len() == 1 { vexing_parens(a) } else { a };
                                    if a.is_empty() {
                                        let _ = writeln!(self.out, "{ind}{} {name};", type_str(&t));
                                    } else {
                                        let _ = writeln!(self.out, "{ind}{} {name}({a});", type_str(&t));
                                    }
                                    return;
                                }
                                if sig::is_dtor(sg) {
                                    return;
                                }
                            }
                        }
                    }
                }
                // the address of a call's result (a returned reference) is no effect of its own
                let mut e = e;
                while let Expr::AddrOf(x) = e {
                    if !matches!(&**x, Expr::Call { .. }) {
                        break;
                    }
                    e = x;
                }
                let t = self.expr(e, 0);
                let _ = writeln!(self.out, "{ind}{t};");
            }
            Stmt::Assign { dst: Expr::Var(v), src }
                if matches!(self.ir.vars[*v].ty, Type::Unknown { .. }) && mwdec_lift::types::is_aggregate(self.db, &ty_of(src, self.vars())) && named(&ty_of(src, self.vars())).is_some() =>
            {
                // whole object stored into an untyped buffer
                let st = ty_of(src, self.vars());
                let name = self.ir.vars[*v].name.clone();
                if self.constructed.contains(v) && !self.declared.contains(v) {
                    self.declared.insert(*v);
                    let d = match &self.ir.vars[*v].ty {
                        Type::Unknown { size } if *size != 4 && *size != 2 && *size != 1 && *size != 8 => format!("unsigned char {name}[{size}]"),
                        t => decl(&local_type(t), &name),
                    };
                    let _ = writeln!(self.out, "{ind}{d};");
                }
                let byte_array = matches!(self.ir.vars[*v].ty, Type::Unknown { size } if size != 4 && size != 2 && size != 1 && size != 8);
                let d = if byte_array { format!("*({}){name}", ptr_to(strip_cv(&st))) } else { format!("*({})&{name}", ptr_to(strip_cv(&st))) };
                let val = self.expr(src, 0);
                let _ = writeln!(self.out, "{ind}{d} = {val};");
            }
            Stmt::Assign { dst, src } if self.setter_for(dst).is_some() => {
                // a private member written through its inline setter
                let (obj, m, pt) = self.setter_for(dst).unwrap();
                let v = match &pt {
                    Some(t) => self.coerce(src, t),
                    None => self.expr(src, 0),
                };
                let _ = writeln!(self.out, "{ind}{obj}{m}({v});");
            }
            Stmt::Assign { dst: Expr::Var(v), src: Expr::Construct { .. } } if self.brace_vars.contains(v) => {
                // a constant aggregate initializer: in the declaration
                if self.brace_done.contains(v) {
                    return;
                }
                self.declared.insert(*v);
                self.brace_done.insert(*v);
                let (vt, vn) = (self.ir.vars[*v].ty.clone(), self.ir.vars[*v].name.clone());
                let d = decl(&local_type(&vt), &vn);
                let init = self.brace_init_of(*v).unwrap_or_default();
                let _ = writeln!(self.out, "{ind}{d} = {init};");
            }
            // an object returned by an unknown call into an untyped stack buffer: the buffer
            // assigned as a blob of its size (an array can't be initialized from a call)
            Stmt::Assign { dst: Expr::Var(v), src: src @ Expr::Call { ret, callee: Callee::Virtual { sig: vs, .. }, .. } }
                if matches!(self.ir.vars[*v].ty, Type::Unknown { size } if size > 8 || size == 3 || (size > 4 && size < 8))
                    && matches!(ret, Type::Void | Type::Unknown { .. })
                    && vs.as_ref().map_or(true, |s| self.method_name(s).is_empty()) =>
            {
                let Type::Unknown { size } = self.ir.vars[*v].ty else { unreachable!() };
                let name = self.ir.vars[*v].name.clone();
                if !self.declared.contains(v) && self.constructed.contains(v) {
                    self.declared.insert(*v);
                    let _ = writeln!(self.out, "{ind}unsigned char {name}[{size}];");
                }
                let blob = format!("__mwdec_blob_{size}");
                let def = format!("struct {blob} {{ unsigned char b[{size}]; }};");
                if !self.type_defs.contains(&def) {
                    self.type_defs.push(def);
                }
                let val = self.coerce(src, &Type::Named(blob.clone()));
                let _ = writeln!(self.out, "{ind}*({blob}*){name} = {val};");
            }
            Stmt::Assign { dst: Expr::Var(v), src } if self.constructed.contains(v) && !self.declared.contains(v) => {
                // declaration with initializer at the first definition
                self.declared.insert(*v);
                let var = &self.ir.vars[*v];
                let (t, name) = (var.ty.clone(), var.name.clone());
                // a frame temporary bound to a reference (`const T& r = f();`)
                if let (VarKind::Stack { .. }, Type::Ref(inner)) = (&var.kind, &t) {
                    let d = decl(&t, &name);
                    let val = self.coerce(src, strip_cv(inner));
                    let _ = writeln!(self.out, "{ind}{d} = {val};");
                    return;
                }
                let d = match &t {
                    Type::Unknown { size } if *size != 4 && *size != 2 && *size != 1 && *size != 8 => decl(&Type::Array(Box::new(Type::Int { size: 1, signed: false }), *size), &name),
                    _ => self.local_decl(*v),
                };
                let val = self.coerce(src, &local_type(&t));
                let _ = writeln!(self.out, "{ind}{d} = {val};");
            }
            // vtable pointer stores of an inlined constructor/destructor: implicit in C++
            Stmt::Assign { dst, src } if is_vtable_store(dst, src) => {
                let _ = writeln!(self.out, "{ind}// (vtable pointer store)");
            }
            Stmt::Assign { dst, src } => {
                if let Expr::BitField { base, shift, width, .. } = dst {
                    if self.bitfield_name(base, *shift, *width, false).is_none() {
                        // no member name: explicit read-modify-write of the storage unit
                        let b = self.lvalue(base);
                        let r = self.expr(base, 8);
                        let m = (((1u64 << width) - 1) << shift) as u32;
                        let v = self.expr(src, 11);
                        let _ = writeln!(self.out, "{ind}{b} = {r} & 0x{:x} | ({v} << {shift}) & 0x{m:x};", !m);
                        return;
                    }
                }
                // a whole object stored through a scalar access of its only member
                // (`*(u16*)ret = kNullId`): an object assignment
                let st = strip_cv(&mwdec_lift::types::resolve(self.db, &ty_of(src, self.vars())).into_owned()).clone();
                if let (Some(sc), true) = (named(&st).map(|s| s.to_string()), mwdec_lift::types::is_aggregate(self.db, &st)) {
                    let same = |t: &Type| named(strip_cv(&mwdec_lift::types::resolve(self.db, t).into_owned())).is_some_and(|n| sig::norm_name(n) == sig::norm_name(&sc));
                    let obj = match dst {
                        Expr::Load { base, offset: 0, ty } if !mwdec_lift::types::is_aggregate(self.db, ty) && pointee(&ty_of(base, self.vars())).is_some_and(|p| same(p)) => {
                            Some(Expr::Load { base: base.clone(), offset: 0, ty: st.clone() })
                        }
                        Expr::Member { base, offset: 0, ty } if !mwdec_lift::types::is_aggregate(self.db, ty) && same(&ty_of(base, self.vars())) => Some((**base).clone()),
                        _ => None,
                    };
                    if let Some(o) = obj {
                        let d = self.lvalue(&o);
                        let v = self.expr(src, 0);
                        let _ = writeln!(self.out, "{ind}{d} = {v};");
                        return;
                    }
                }
                let dt = if matches!(dst, Expr::Var(_)) { local_type(&self.decl_type_rw(dst, false)) } else { self.decl_type_rw(dst, false) };
                let d = self.lvalue(dst);
                // read-modify-write of a memory word: `m |= v` (the older compiler computes the
                // address of `m = m | v` twice and folds it into an update-form load)
                if matches!(dst, Expr::Load { .. } | Expr::Member { .. } | Expr::Index { .. } | Expr::Global { .. }) && strip_cv(&ty_of(dst, self.vars())).int_info().is_some_and(|(sz, _)| sz == 4) {
                    let inner = match src {
                        Expr::Cast { e, ty } if strip_cv(ty).int_info().is_some_and(|(sz, _)| sz == 4) => &**e,
                        e => e,
                    };
                    if let Expr::Binary { op: op @ (BinOp::Or | BinOp::And | BinOp::Xor | BinOp::Add | BinOp::Sub), l, r, .. } = inner {
                        if **l == *dst && !is_ptr(&ty_of(r, self.vars())) {
                            let o = match op {
                                BinOp::Or => "|",
                                BinOp::And => "&",
                                BinOp::Xor => "^",
                                BinOp::Add => "+",
                                _ => "-",
                            };
                            let v = self.expr(r, 1);
                            let _ = writeln!(self.out, "{ind}{d} {o}= {v};");
                            return;
                        }
                    }
                }
                // variant point [`mwdec_lift::variants::ASSIGN_COMPOUND_NARROW`]: a narrow (char /
                // short) destination updated from itself, `v = (u8)(v | x)`, as `v |= x` (the compiler
                // computes the two forms differently)
                if strip_cv(&ty_of(dst, self.vars())).int_info().is_some_and(|(sz, _)| sz < 4) {
                    let inner = match src {
                        Expr::Cast { e, ty } if strip_cv(ty).int_info().is_some() => &**e,
                        e => e,
                    };
                    if let Expr::Binary { op: op @ (BinOp::Or | BinOp::And | BinOp::Xor | BinOp::Add | BinOp::Sub | BinOp::Shl | BinOp::Shr), l, r, .. } = inner {
                        let l = match &**l {
                            Expr::Cast { e, .. } => &**e,
                            e => e,
                        };
                        let narrowed = matches!(dst, Expr::Var(v) if self.narrowed.contains(v));
                        if *l == *dst && !is_ptr(&ty_of(r, self.vars())) && (narrowed || mwdec_lift::variants::alt(mwdec_lift::variants::ASSIGN_COMPOUND_NARROW)) {
                            let o = match op {
                                BinOp::Or => "|",
                                BinOp::And => "&",
                                BinOp::Xor => "^",
                                BinOp::Add => "+",
                                BinOp::Sub => "-",
                                BinOp::Shl => "<<",
                                _ => ">>",
                            };
                            let v = self.expr(r, 1);
                            let _ = writeln!(self.out, "{ind}{d} {o}= {v};");
                            return;
                        }
                    }
                }
                self.member_store = !matches!(dst, Expr::Var(_));
                let v = self.coerce(src, &dt);
                self.member_store = false;
                let _ = writeln!(self.out, "{ind}{d} = {v};");
            }
            // `new (p) T(args)`: placement new constructs only after testing `p` for null
            Stmt::If { cond, then, els } if !self.opts.c_mode && els.is_empty() && then.len() == 1 && self.placement_new(cond, &then[0]).is_some() => {
                let (this, args, sg) = self.placement_new(cond, &then[0]).unwrap();
                let cls = type_str(&Type::Named(strip_unnamed_ns(sg.this_class.as_deref().unwrap_or_default())));
                let p = self.expr(this, 0);
                let a = self.args(args, Some(sg));
                let _ = writeln!(self.out, "{ind}new ({p}) {cls}({a});");
            }
            Stmt::If { cond, then, els } => {
                let c = self.cond(cond);
                let _ = writeln!(self.out, "{ind}if ({c}) {{");
                self.block(then, depth + 1);
                if els.is_empty() {
                    let _ = writeln!(self.out, "{ind}}}");
                } else if els.len() == 1 && matches!(els[0], Stmt::If { .. }) {
                    let _ = write!(self.out, "{ind}}} else ");
                    // render `else if` inline
                    let mut tmp = std::mem::take(&mut self.out);
                    self.stmt(&els[0], depth);
                    let rendered = std::mem::take(&mut self.out);
                    tmp.push_str(rendered.trim_start());
                    self.out = tmp;
                } else {
                    let _ = writeln!(self.out, "{ind}}} else {{");
                    self.block(els, depth + 1);
                    let _ = writeln!(self.out, "{ind}}}");
                }
            }
            Stmt::While { cond, body } => {
                let c = self.cond(cond);
                let _ = writeln!(self.out, "{ind}while ({c}) {{");
                self.block(body, depth + 1);
                let _ = writeln!(self.out, "{ind}}}");
            }
            Stmt::DoWhile { body, cond } => {
                let _ = writeln!(self.out, "{ind}do {{");
                self.block(body, depth + 1);
                let c = self.cond(cond);
                let _ = writeln!(self.out, "{ind}}} while ({c});");
            }
            Stmt::For { init, cond, step, body } => {
                let mut parts = vec![];
                for s in init {
                    match s {
                        // declared by the init (`for (T it = c.begin(); ...)`)
                        Stmt::Assign { dst: Expr::Var(v), src } if self.constructed.contains(v) && !self.declared.contains(v) && init.len() == 1 => {
                            self.declared.insert(*v);
                            let t = self.ir.vars[*v].ty.clone();
                            let d = self.local_decl(*v);
                            let val = self.coerce(src, &local_type(&t));
                            parts.push(format!("{d} = {val}"));
                        }
                        s => parts.push(self.inline_stmt(s)),
                    }
                }
                let i = parts.join(", ");
                let c = self.cond(cond);
                let st = step.iter().map(|s| self.inline_stmt(s)).collect::<Vec<_>>().join(", ");
                let _ = writeln!(self.out, "{ind}for ({i}; {c}; {st}) {{");
                self.block(body, depth + 1);
                let _ = writeln!(self.out, "{ind}}}");
            }
            Stmt::Switch { e, cases } => {
                let t = if is_ptr(&ty_of(e, self.vars())) || mwdec_lift::types::is_aggregate(self.db, &ty_of(e, self.vars())) {
                    // (only integers can be switched on)
                    format!("(int){}", self.expr(e, 14))
                } else {
                    self.expr(e, 0)
                };
                let _ = writeln!(self.out, "{ind}switch ({t}) {{");
                for c in cases {
                    if c.is_default {
                        let _ = writeln!(self.out, "{ind}default:");
                    }
                    for v in &c.values {
                        let _ = writeln!(self.out, "{ind}case {v}:");
                    }
                    self.block(&c.body, depth + 1);
                }
                let _ = writeln!(self.out, "{ind}}}");
            }
            Stmt::Return(None) => {
                if self.sret_local.is_some() {
                    let _ = writeln!(self.out, "{ind}return __return_value;");
                } else {
                    let _ = writeln!(self.out, "{ind}return;");
                }
            }
            Stmt::Return(Some(e)) => {
                let rt = self.ir.sig.ret.clone();
                // a returned ternary converts each arm (`c ? (E)2 : (E)0`): converting the
                // whole int-typed ternary would let MWCC use its branchless int forms
                let v = match e {
                    Expr::Ternary { c, t, f, .. } if !matches!(strip_cv(&rt), Type::Ref(_)) => {
                        let cs = self.expr(c, 4);
                        let ts = self.coerce(t, &rt);
                        let fs = self.coerce(f, &rt);
                        format!("{cs} ? {ts} : {fs}")
                    }
                    _ => self.coerce(e, &rt),
                };
                let _ = writeln!(self.out, "{ind}return {v};");
            }
            Stmt::Break => {
                let _ = writeln!(self.out, "{ind}break;");
            }
            Stmt::Continue => {
                let _ = writeln!(self.out, "{ind}continue;");
            }
            Stmt::Goto(l) => {
                let _ = writeln!(self.out, "{ind}goto block_{l};");
            }
            Stmt::Label(l) => {
                let _ = writeln!(self.out, "block_{l}:;");
            }
            Stmt::Comment(c) if c.starts_with(mwdec_lift::idioms::INIT_MARK) => {}
            Stmt::Comment(c) => {
                let _ = writeln!(self.out, "{ind}// {c}");
            }
        }
    }

    fn inline_stmt(&mut self, s: &Stmt) -> String {
        match s {
            Stmt::Assign { dst, src } => {
                let dt = if matches!(dst, Expr::Var(_)) { local_type(&self.decl_type_rw(dst, false)) } else { self.decl_type_rw(dst, false) };
                let d = self.lvalue(dst);
                let v = self.coerce(src, &dt);
                format!("{d} = {v}")
            }
            Stmt::Expr(e) => self.expr(e, 0),
            _ => String::new(),
        }
    }

    /// Render an assignment destination.
    fn lvalue(&mut self, e: &Expr) -> String {
        // a store into a member of a const object (a const-reference parameter written by code
        // the lifter couldn't give a return object): through a cast that drops the const
        if let Expr::Member { base, offset, ty } = e {
            if matches!(**base, Expr::Var(_)) && self.const_lvalue(e) && scalar_size(ty).is_some_and(|z| z > 0) {
                return self.raw_access(base, *offset, ty, false);
            }
        }
        self.lvalue_ctx = true;
        let s = self.expr(e, 2);
        self.lvalue_ctx = false;
        // (a store through a reinterpreting cast: never to a const object; `*(const T*)p`, not
        // `*(const T**)p`, whose object is a pointer)
        if let Some(rest) = s.strip_prefix("*(const ") {
            if let Some(ty) = rest.split(')').next() {
                if !ty.contains('(') && ty.ends_with('*') && !ty[..ty.len() - 1].contains('*') {
                    return format!("*({rest}");
                }
            }
        }
        s
    }

    /// Class whose members the function can access directly.
    fn own_class(&self) -> Option<String> {
        if let Some(c) = &self.ir.sig.this_class {
            return Some(c.clone());
        }
        sig::split_scope(&self.ir.sig.qualified_name).0.map(|s| s.to_string())
    }

    /// Can this function name `owner::field` directly (C++ access rules with the TypeDb's
    /// recorded member access)?
    fn field_accessible(&self, owner: &str, field: &str) -> bool {
        let own = self.own_class();
        if own.as_deref().map(sig::norm_name) == Some(sig::norm_name(owner)) {
            return true;
        }
        let access = self
            .db
            .and_then(|db| sig::find_class(db, owner))
            .and_then(|c| c.fields.iter().find(|f| f.name == field))
            .map(|f| f.access.clone());
        // (MWCC follows C++98: nested classes get no special access to the enclosing class)
        match access {
            Some(mwdec_core::Access::Public) => true,
            Some(mwdec_core::Access::Protected) => {
                self.befriended(owner)
                    || match &own {
                        Some(o) => self.is_base_of(Some(&Type::Named(owner.to_string())), &Type::Named(o.clone())),
                        None => false,
                    }
            }
            Some(mwdec_core::Access::Private) => self.befriended(owner),
            None => self.accessible(owner),
        }
    }

    /// A protected member of a base reached through an object whose class isn't ours (or
    /// derived from ours): C++ only allows `this`-like access to inherited protected members.
    fn protected_through_other(&self, obj_cls: &str, path: &[PathElem]) -> bool {
        let (Some(db), Some(own)) = (self.db, self.own_class()) else { return false };
        let derived_from_own = sig::norm_name(obj_cls) == sig::norm_name(&own) || self.is_base_of(Some(&Type::Named(own.clone())), &Type::Named(obj_cls.to_string()));
        if derived_from_own {
            return false;
        }
        path.iter().any(|p| match p {
            PathElem::Field(n, owner) if sig::norm_name(owner) != sig::norm_name(&own) => {
                let access = sig::find_class(db, owner).and_then(|c| c.fields.iter().find(|f| f.name == *n)).map(|f| f.access.clone());
                matches!(access, Some(mwdec_core::Access::Protected)) && !self.befriended(owner)
            }
            _ => false,
        })
    }

    /// `__construct_array(local, T::T, T::~T, size, n)` on a stack buffer: (variable, class, n).
    fn array_construction(&self, s: &Stmt) -> Option<(VarId, String, i64)> {
        let Stmt::Expr(Expr::Call { callee: Callee::Direct { symbol, .. }, args, .. }) = s else { return None };
        if symbol != "__construct_array" || args.len() != 5 {
            return None;
        }
        let v = stack_var_of(&args[0], self.ir)?;
        fn fsym(e: &Expr) -> Option<&str> {
            match e {
                Expr::FuncAddr { symbol } => Some(symbol),
                Expr::Cast { e, .. } => fsym(e),
                _ => None,
            }
        }
        let ctor = fsym(&args[1])?;
        let cls = sig::sig_of(ctor, self.db).this_class?;
        let n = args[4].as_int()?;
        let size = args[3].as_int()?;
        let VarKind::Stack { size: vs, .. } = self.ir.vars[v].kind else { return None };
        (n > 0 && size > 0 && (size * n) as u32 <= vs.max((size * n) as u32)).then_some((v, cls, n))
    }

    /// The array argument of an array construction/destruction call is a member array of
    /// `this` in a constructor/destructor, or a local array declared by `array_construction`.
    fn implicit_array(&self, a: &Expr, constructing: bool) -> bool {
        if let Some(v) = stack_var_of(a, self.ir) {
            // (the same frame slot may be several variables)
            let slot = |x: VarId| match self.ir.vars[x].kind {
                VarKind::Stack { offset, .. } => Some(offset),
                _ => None,
            };
            return !constructing && self.constructed.iter().any(|&c| c == v || (slot(c).is_some() && slot(c) == slot(v)));
        }
        let in_cdtor = if constructing { sig::is_ctor(&self.ir.sig) } else { sig::is_dtor(&self.ir.sig) };
        if !in_cdtor {
            return false;
        }
        let mut cur = a;
        loop {
            match cur {
                Expr::AddrOf(x) | Expr::Cast { e: x, .. } => cur = x,
                Expr::Load { base, .. } | Expr::Member { base, .. } => cur = base,
                Expr::Binary { op: BinOp::Add, l, .. } => cur = l,
                Expr::Var(v) => return self.ir.this_var == Some(*v),
                _ => return false,
            }
        }
    }

    /// Is the member access `e` rendered through an inline getter that returns a non-const
    /// reference or pointer?
    fn read_via_nonconst_getter(&self, e: &Expr) -> bool {
        let Some(db) = self.db else { return false };
        let (base, off, ptr, ty) = match e {
            Expr::Load { base, offset, ty } => (&**base, *offset, true, ty),
            Expr::Member { base, offset, ty } => (&**base, *offset, false, ty),
            _ => return false,
        };
        let bt = ty_of(base, self.vars());
        let Some(ct) = (if ptr { pointee(&bt).cloned() } else { Some(bt) }) else { return false };
        let ctr = mwdec_lift::types::resolve(Some(db), strip_cv(&ct)).into_owned();
        let Some(cls) = named(&ctr) else { return false };
        let want = if matches!(ty, Type::Unknown { size: 0 }) { 0 } else { scalar_size(ty).unwrap_or(0) };
        let Some((path, _)) = field_path(db, cls, off, want) else { return false };
        match path.last() {
            Some(PathElem::Field(n, owner)) if !self.field_accessible(owner, n) => self.accessor_decl(owner, n).is_some_and(|(_, rt)| match strip_cv(&rt) {
                Type::Ref(x) | Type::Ptr(x) => !matches!(**x, Type::Const(_)),
                _ => false,
            }),
            _ => false,
        }
    }

    /// Does `owner` declare this function or its class a friend?
    fn befriended(&self, owner: &str) -> bool {
        let Some(db) = self.db else { return false };
        let fr = db.friends.get(owner).or_else(|| db.friends.get(&strip_template_args(owner)));
        let Some(fr) = fr else { return false };
        let last = |s: &str| sig::split_scope(&strip_template_args(s)).1.to_string();
        let own = self.own_class();
        let fname = sig::split_scope(&self.ir.sig.qualified_name).1.to_string();
        fr.iter().any(|f| {
            own.as_deref().map_or(false, |o| sig::norm_name(o) == sig::norm_name(f) || last(o) == last(f))
                || (self.ir.sig.this_class.is_none() && (fname == *f || self.ir.sig.qualified_name == *f))
        })
    }

    fn accessible(&self, owner: &str) -> bool {
        let Some(own) = self.own_class() else { return false };
        if sig::norm_name(&own) == sig::norm_name(owner) {
            return true;
        }
        // members of bases (assume protected)
        if self.is_base_of(Some(&Type::Named(owner.to_string())), &Type::Named(own.clone())) {
            return true;
        }
        self.befriended(owner)
    }

    /// An inline getter `T Get() const { return field; }` of `owner`, if the headers have one.
    fn accessor(&self, owner: &str, field: &str) -> Option<String> {
        self.accessor_decl(owner, field).map(|(m, _)| m)
    }

    /// The getter `accessor` picks and its declared return type.
    fn accessor_decl(&self, owner: &str, field: &str) -> Option<(String, Type)> {
        self.accessor_decl_c(owner, field, false)
    }

    /// `accessor_decl` on an object that is const when `obj_const`: of a const/non-const
    /// overload pair, the const one (returning a pointer/reference to const) is what's called.
    fn accessor_decl_c(&self, owner: &str, field: &str, obj_const: bool) -> Option<(String, Type)> {
        self.accessor_decl_k(owner, field, obj_const).map(|(m, t, _)| (m, t))
    }

    /// `accessor_decl_c` with whether the accessor is a const method.
    fn accessor_decl_k(&self, owner: &str, field: &str, obj_const: bool) -> Option<(String, Type, bool)> {
        let db = self.db?;
        let key = strip_template_args(owner);
        let prefix = format!("{key}::");
        let body = format!("return {field} ;");
        // the member's declared type: a "getter" returning something else converts it
        // (`bool HasModel() const { return mModel; }`)
        let ft = sig::find_class(db, owner).and_then(|c| c.fields.iter().find(|f| f.name == field)).map(|f| mwdec_lift::types::resolve(Some(db), strip_cv(&f.ty)).into_owned());
        let compatible = |rt: &Type| -> bool {
            let Some(ft) = &ft else { return true };
            let r = match strip_cv(rt) {
                Type::Ref(x) => strip_cv(x).clone(),
                t => t.clone(),
            };
            let r = mwdec_lift::types::resolve(Some(db), &r).into_owned();
            let agg = |t: &Type| mwdec_lift::types::is_aggregate(Some(db), t);
            match (&r, strip_cv(ft)) {
                // (template parameters and unresolved spellings: unknown)
                (Type::Named(n), _) if sig::find_class(db, n).is_none() && !db.typedefs.contains_key(n.as_str()) && !db.enums.contains_key(n.as_str()) => true,
                (Type::Named(a), Type::Named(b)) => sig::norm_name(a) == sig::norm_name(b) || agg(&r) == agg(ft),
                (a, b) if agg(a) || agg(b) => false,
                (a, b) if is_ptr(a) != (is_ptr(b) || matches!(b, Type::Array(..))) => false,
                _ => true,
            }
        };
        let mut best: Option<(u8, String, Type, bool)> = None;
        for (name, ds) in db.decls.range(prefix.clone()..) {
            if !name.starts_with(&prefix) {
                break;
            }
            let m = &name[prefix.len()..];
            // (conversion operators aren't getters one can name: `x.operator const T&()`)
            if m.contains("::") || m.starts_with("operator ") {
                continue;
            }
            for d in ds {
                if d.params.is_empty() && !d.is_static && d.inline_body.as_deref().map_or(false, |x| x == body || getter_returns(x, field)) && compatible(&d.ret) {
                    // prefer named const getters over operators
                    let rank = if m.starts_with("operator") { 2 } else if d.is_const { 0 } else { 1 };
                    let const_ret = |t: &Type| matches!(pointee(strip_cv(t)), Some(Type::Const(_))) || matches!(strip_cv(t), Type::Ref(x) if matches!(**x, Type::Const(_)));
                    let const_tie = obj_const && const_ret(&d.ret) && best.as_ref().is_some_and(|(r, bm, bt, _)| rank == *r && bm == m && !const_ret(bt));
                    if const_tie || best.as_ref().map_or(true, |(r, _, _, _)| rank < *r) {
                        best = Some((rank, m.to_string(), d.ret.clone(), d.is_const));
                    }
                }
            }
        }
        best.map(|(_, m, t, c)| (m, t, c))
    }

    /// A read of `path` goes through a non-const getter first (an inaccessible member whose only
    /// inline getter is non-const): on a const object, the object is cast non-const.
    fn path_needs_nonconst(&self, path: &[PathElem]) -> bool {
        for p in path {
            if let PathElem::Field(n, owner) = p {
                if self.field_accessible(owner, n) {
                    return false;
                }
                return self.accessor_decl_k(owner, n, false).is_some_and(|(_, _, c)| !c);
            }
        }
        false
    }

    /// `obj.GetX(i)` for `obj.mX[i]` when `mX` is inaccessible here and the class declares an
    /// inline accessor returning `mX[param]`.
    fn element_accessor_call(&mut self, base: &Expr, i: &str) -> Option<String> {
        let db = self.db?;
        if self.opts.raw_offsets {
            return None;
        }
        let (obj, off, ptr) = match base {
            Expr::Load { base, offset, ty: Type::Array(..) } => (&**base, *offset, true),
            Expr::Member { base, offset, ty: Type::Array(..) } => (&**base, *offset, false),
            _ => return None,
        };
        let ot = ty_of(obj, self.vars());
        let ct = if ptr { pointee(&ot)?.clone() } else { ot };
        let cls = named(&mwdec_lift::types::resolve(Some(db), strip_cv(&ct)).into_owned())?.to_string();
        let at = match base {
            Expr::Load { ty, .. } | Expr::Member { ty, .. } => ty.clone(),
            _ => return None,
        };
        let size = mwdec_lift::types::size_of(Some(db), &at)?;
        let (path, _) = field_path(db, &cls, off, size)?;
        let (last, prefix) = path.split_last()?;
        let PathElem::Field(field, owner) = last else { return None };
        if self.field_accessible(owner, field) || prefix.iter().any(|p| !matches!(p, PathElem::Field(..))) {
            return None;
        }
        let key = format!("{}::", strip_template_args(owner));
        let mut found = None;
        for (qn, ds) in db.decls.range(key.clone()..) {
            if !qn.starts_with(&key) {
                break;
            }
            let m = &qn[key.len()..];
            if m.contains("::") || m.starts_with("operator") {
                continue;
            }
            for d in ds {
                let Some(pn) = d.params.first().and_then(|p| p.name.clone()) else { continue };
                let want = format!("return {field} [ {pn} ] ;");
                if d.params.len() == 1 && !d.is_static && d.inline_body.as_deref().map(|b| b.split_whitespace().collect::<Vec<_>>().join(" ")) == Some(want) && matches!(strip_cv(&d.ret), Type::Ref(_)) {
                    found = Some(m.to_string());
                }
            }
        }
        let m = found?;
        let o = self.expr(obj, 15);
        let pre = self.path_str(prefix, true);
        let sep = if ptr { "->" } else { "." };
        Some(if pre.is_empty() { format!("{o}{sep}{m}({i})") } else { format!("{o}{sep}{pre}.{m}({i})") })
    }

    /// A getter for an inaccessible member; for a store through the member (`obj.GetX().y = v`,
    /// `obj.data()[4] = v`) it must return a non-const reference or pointer.
    fn usable_getter(&self, owner: &str, field: &str, read: bool) -> Option<String> {
        let (m, rt) = self.accessor_decl(owner, field)?;
        if read {
            return Some(m);
        }
        match strip_cv(&rt) {
            Type::Ref(x) | Type::Ptr(x) if !matches!(**x, Type::Const(_)) => Some(m),
            _ => None,
        }
    }

    /// Render a member path; inaccessible fields read through inline getters where possible.
    fn path_str(&self, path: &[PathElem], read: bool) -> String {
        let mut s = String::new();
        for (i, p) in path.iter().enumerate() {
            match p {
                PathElem::Field(n, owner) => {
                    if !s.is_empty() {
                        s.push('.');
                    }
                    let last = i + 1 == path.len();
                    // inaccessible members are read through inline getters when the headers
                    // have one
                    if (read || !last) && !self.field_accessible(owner, n) {
                        if let Some(m) = self.usable_getter(owner, n, read) {
                            let _ = write!(s, "{m}()");
                            continue;
                        }
                    }
                    s.push_str(n);
                }
                PathElem::Index(k) => {
                    let _ = write!(s, "[{k}]");
                }
                PathElem::Base(_) => {}
            }
        }
        s
    }

    fn cond(&mut self, e: &Expr) -> String {
        // a bool member tested against zero through its widened byte: the bool itself
        // (`x.valid()`, not `(unsigned int)x.valid() != 0`)
        if let Expr::Binary { op: op @ (BinOp::Ne | BinOp::Eq), l, r, .. } = e {
            if r.as_int() == Some(0) {
                let mut inner = &**l;
                while let Expr::Cast { e: x, .. } = inner {
                    inner = x;
                }
                if !std::ptr::eq(inner, &**l) && matches!(inner, Expr::Member { .. } | Expr::Load { .. }) && matches!(strip_cv(&self.decl_type_rw(inner, true)), Type::Bool) {
                    let x = self.expr(inner, 15);
                    return if *op == BinOp::Ne { x } else { format!("!{x}") };
                }
            }
        }
        self.expr(e, 0)
    }

    /// Forward-declare a function the context doesn't declare (file-local static functions,
    /// anonymous-namespace helpers, function templates of the unit): parameter types from the
    /// mangled name, else from the call (`(arg types, return type)`).
    fn declare_function(&mut self, symbol: &str, s: Option<&mwdec_core::FuncSig>, call: Option<(&[Type], &Type)>, conf: u8) {
        let Some(db) = self.db else { return };
        if self.fn_decls.get(symbol).map_or(false, |(c, _)| *c >= conf) {
            return;
        }
        // (the function itself only when it's an instance of a template the context doesn't
        // declare: its explicit specialization needs a primary template)
        if is_compiler_builtin(symbol) || (symbol == self.ir.symbol && !{ let l = sig::split_scope(&self.ir.sig.qualified_name).1; strip_template_args(l) != l }) {
            return;
        }
        let demangled = sig::demangle(symbol);
        let owned;
        let s = match s {
            Some(s) => s,
            None => {
                owned = if demangled.is_some() { sig::sig_of(symbol, Some(db)) } else { sig::sig_of(symbol, None) };
                &owned
            }
        };
        let qn = strip_unnamed_ns(&s.qualified_name);
        let key = strip_template_args(&qn);
        let declared = db.decls.contains_key(&qn)
            || db.decls.contains_key(&key)
            || db.functions.contains_key(symbol)
            || db.functions.contains_key(&qn)
            || db.globals.contains_key(symbol);
        if declared || s.this_class.is_some() {
            return;
        }
        let (scope, base) = sig::split_scope(&qn);
        // members of classes can't be declared outside the class
        if let Some(sc) = scope {
            if sig::find_class(db, sc).is_some() || db.templates.contains_key(&strip_template_args(sc)) {
                return;
            }
        }
        let c_mode = self.opts.c_mode;
        let params: Vec<String> = if demangled.is_some() {
            s.params
                .iter()
                .enumerate()
                .map(|(i, p)| match (&p.ty, call.and_then(|(tys, _)| tys.get(i))) {
                    (Type::Unknown { .. }, Some(t)) => type_str(t),
                    (t, _) => type_str(t),
                })
                .collect()
        } else if let Some((tys, _)) = call.filter(|(tys, _)| {
            let r = |t: &Type| mwdec_lift::types::resolve(Some(db), t).into_owned();
            // (narrow integer arguments the call passes unextended: a prototype keeps them so)
            !c_mode || conf >= 3 || tys.iter().any(|t| is_float(&r(t))) || tys.iter().any(|t| matches!(strip_cv(&r(t)), Type::Int { size: 1 | 2, .. } | Type::Bool | Type::Char))
        }) {
            // C: unprototyped (K&R) unless float arguments would be promoted to double, narrow
            // integers extended, or the address is converted to a prototyped function pointer
            // type (MWCC checks those)
            self.fn_param_tys.insert(symbol.to_string(), tys.to_vec());
            tys.iter().map(type_str).collect()
        } else {
            vec![]
        };
        let ret = match call {
            Some((_, r)) if !matches!(r, Type::Unknown { size: 0 }) => value_type(r),
            _ if !sig::ret_unknown(s) => s.ret.clone(),
            _ => Type::Void,
        };
        let ret = local_type(&ret);
        let mut plist = params.join(", ");
        if s.variadic {
            plist.push_str(if plist.is_empty() { "..." } else { ", ..." });
        }
        if plist.is_empty() && c_mode && demangled.is_some() {
            plist = "void".into();
        }
        // template function: declare the primary template with the instance's parameter types;
        // the call spells the template arguments explicitly
        let mut tmpl = String::new();
        let mut name = base.to_string();
        if let Some(lt) = base.find('<') {
            let args = sig::split_top(&base[lt + 1..base.len().saturating_sub(1)], ',');
            let ps: Vec<String> = args
                .iter()
                .enumerate()
                .map(|(i, a)| if a.trim().parse::<i64>().is_ok() { format!("int T{i}") } else { format!("class T{i}") })
                .collect();
            tmpl = format!("template <{}> ", ps.join(", "));
            name = base[..lt].to_string();
        }
        let mut d = format!("{tmpl}{}({plist});", decl(&ret, &name));
        if demangled.is_none() && !c_mode {
            d = format!("extern \"C\" {d}");
        }
        if let Some(sc) = scope {
            for part in sc.split("::").collect::<Vec<_>>().into_iter().rev() {
                d = format!("namespace {part} {{ {d} }}");
            }
        }
        if s.qualified_name.contains("@unnamed@") {
            d = format!("namespace {{ {d} }}");
        }
        self.fn_decls.insert(symbol.to_string(), (conf, d));
    }

    /// Render `e` converted to type `to` (casts where C++ wouldn't convert implicitly).
    fn coerce(&mut self, e: &Expr, to: &Type) -> String {
        // the address of an element of a container template's raw storage passed as `T&`/`T*`
        if let (Expr::AddrOf(inner), Some(db), false) = (e, self.db, self.opts.raw_offsets) {
            if let Expr::Member { base, offset, .. } = &**inner {
                let target = match strip_cv(to) {
                    Type::Ref(t) | Type::Ptr(t) => Some(strip_cv(t).clone()),
                    _ => None,
                };
                let bt = mwdec_lift::types::resolve(Some(db), &ty_of(base, self.vars())).into_owned();
                if let (Some(t), Some(cls)) = (target, named(strip_cv(&bt)).map(|s| s.to_string())) {
                    if let Some(i) = named(&t).and_then(|tc| element_index(db, &cls, *offset, tc, &t)) {
                        let b = self.expr(base, 15);
                        return if matches!(strip_cv(to), Type::Ref(_)) { format!("{b}[{i}]") } else { format!("&{b}[{i}]") };
                    }
                }
            }
        }
        // a value the lifter couldn't recover (stack-passed argument, uninitialized register):
        // a placeholder of the wanted type
        if let Expr::Unknown { text, .. } = e {
            let note = format!("/* {} */", text.replace("*/", "* /"));
            let tr = mwdec_lift::types::resolve(self.db, strip_cv(to)).into_owned();
            return match strip_cv(to) {
                Type::Ref(inner) => format!("*({})0 {note}", ptr_to(strip_cv(inner))),
                Type::Void | Type::Unknown { .. } => format!("0 {note}"),
                t if mwdec_lift::types::is_aggregate(self.db, &tr) => format!("*({})0 {note}", ptr_to(t)),
                t => format!("({})0 {note}", type_str(t)),
            };
        }
        // (an unknown virtual method: the stand-in class's slot gets this return type)
        if let Expr::Call { callee: Callee::Virtual { sig: vs, .. }, ret: Type::Void | Type::Unknown { .. }, .. } = e {
            if vs.as_ref().map_or(true, |s| self.method_name(s).is_empty()) && !matches!(strip_cv(to), Type::Void | Type::Unknown { size: 0 } | Type::Ref(_)) {
                self.vt_ret_hint = Some(access_type(to));
                let s = self.expr(e, 0);
                self.vt_ret_hint = None;
                return s;
            }
        }
        if let Expr::FuncAddr { symbol } = e {
            // an undeclared function whose address is taken as a typed function pointer
            if let Type::FuncPtr(fs) = mwdec_lift::types::resolve(self.db, to).as_ref() {
                let fs = (**fs).clone();
                let ptys: Vec<Type> = fs.params.iter().map(|p| p.ty.clone()).collect();
                self.declare_function(symbol, None, Some((&ptys, &fs.ret)), 3);
            }
        }
        // a float object's bits read as a word (`*(int*)&f`, a copy through integer registers)
        // wanted as a float: the float itself (a conversion would change the value)
        if matches!(strip_cv(to), Type::Float { size: 4 }) {
            if let Expr::Load { base, offset: 0, ty } = e {
                if let (Expr::AddrOf(x), true) = (&**base, matches!(strip_cv(ty), Type::Int { size: 4, .. } | Type::Unknown { size: 4 })) {
                    if matches!(strip_cv(&ty_of(x, self.vars())), Type::Float { size: 4 }) {
                        return self.expr(x, 0);
                    }
                }
            }
        }
        let from = self.decl_type_rw(e, true);
        let tos = strip_cv(to);
        // (a reference loaded from memory where a pointer is wanted: the pointer it is stored as)
        let from = match strip_cv(&from) {
            Type::Ref(inner) if matches!(tos, Type::Ptr(_)) && matches!(e, Expr::Load { .. } | Expr::Member { .. }) => Type::Ptr(inner.clone()),
            _ => from,
        };
        let froms = strip_cv(&from);
        if !matches!(tos, Type::Ref(_)) {
            if let Some(s) = self.coerce_extra(e, to, froms) {
                return s;
            }
        }
        if let Type::Ref(inner) = tos {
            // a reference is bound to the object the register points at
            return match e {
                // an unknown virtual call's object result bound to a const reference: the
                // stand-in method returns the class by value
                Expr::AddrOf(x)
                    if matches!(**inner, Type::Const(_))
                        && matches!(&**x, Expr::Call { callee: Callee::Virtual { sig: vs, .. }, ret: Type::Void | Type::Unknown { .. }, .. } if vs.as_ref().map_or(true, |s| self.method_name(s).is_empty())) =>
                {
                    self.vt_ret_hint = Some(strip_cv(inner).clone());
                    let s = self.expr(x, 0);
                    self.vt_ret_hint = None;
                    s
                }
                Expr::AddrOf(x) => {
                    let xt = ty_of(x, self.vars());
                    // a call returning `T&` is already the object
                    let xt = match strip_cv(&xt) {
                        Type::Ref(r) => (**r).clone(),
                        _ => xt,
                    };
                    let same = sig::norm_name(&format!("{:?}", strip_cv(&xt))) == sig::norm_name(&format!("{:?}", strip_cv(inner)));
                    if !same {
                        if let Some(s) = self.typed_member(x, inner) {
                            return s;
                        }
                        // any other mismatch: reinterpret the object at that address
                        let a = self.expr(e, 14);
                        return format!("*({})({a})", ptr_to(inner));
                    }
                    // a const object bound to a non-const reference parameter
                    if !matches!(**inner, Type::Const(_)) && self.const_lvalue(x) {
                        return format!("const_cast<{}&>({})", type_str(inner), self.expr(x, 0));
                    }
                    self.expr(x, 0)
                }
                _ if is_ptr(froms) => {
                    let p = pointee(froms).map(|t| strip_cv(t).clone());
                    let to_const_ptr = matches!(pointee(froms), Some(Type::Const(_)))
                        || matches!(e, Expr::Call { callee: Callee::Method { sig: cs, .. } | Callee::Direct { sig: cs, .. }, args, .. } if self.returns_const_ptr(cs, args.len()));
                    if !matches!(**inner, Type::Const(_)) && to_const_ptr && p.as_ref().map_or(false, |p| named(p).is_some()) {
                        let pt = strip_cv(p.as_ref().unwrap()).clone();
                        format!("*const_cast<{}>({})", ptr_to(&pt), self.expr(e, 0))
                    } else if p.as_ref() == Some(strip_cv(inner)) || self.is_base_of(Some(strip_cv(inner)), p.as_ref().unwrap_or(&Type::Void)) {
                        format!("*{}", self.expr(e, 14))
                    } else if let Some(s) = self.typed_member(&Expr::Load { base: Box::new(e.clone()), offset: 0, ty: Type::Unknown { size: 0 } }, inner) {
                        s
                    } else {
                        format!("*({}){}", ptr_to(inner), self.expr(e, 14))
                    }
                }
                // a word-sized access that is the whole member object (`damage.GetWeaponMode()`)
                Expr::Load { .. } | Expr::Member { .. } if matches!(froms, Type::Unknown { .. }) && self.typed_member(e, inner).is_some() => self.typed_member(e, inner).unwrap(),
                // a constant bound to a const reference (the compiler materialises it): the value
                // for a scalar, a class constructed from it by a one-scalar constructor
                Expr::Int { .. } | Expr::Float { .. } if matches!(**inner, Type::Const(_)) && self.scalar_target(inner) => self.expr(e, 0),
                Expr::Int { .. } if matches!(**inner, Type::Const(_)) && self.scalar_ctor_class(inner).is_some() => {
                    format!("{}({})", self.scalar_ctor_class(inner).unwrap(), self.expr(e, 0))
                }
                _ if matches!(froms, Type::Int { .. } | Type::Unknown { .. }) => format!("*({}){}", ptr_to(inner), self.expr(e, 14)),
                _ => self.expr(e, 0),
            };
        }
        // an array converts to a pointer to its element type only
        let decayed;
        let froms = match froms {
            Type::Array(el, _) if matches!(tos, Type::Ptr(_)) => {
                decayed = Type::Ptr(el.clone());
                &decayed
            }
            f => f,
        };
        let need_cast = match (tos, froms) {
            (Type::Ptr(_), Type::Ptr(a)) => {
                let tp = pointee(tos).map(|t| strip_cv(t).clone());
                let fp = strip_cv(a).clone();
                let is_const = |t: &Type| match t {
                    Type::Const(_) => true,
                    Type::Array(e, _) => matches!(**e, Type::Const(_)),
                    _ => false,
                };
                // (also the address of a member of a const object: its type doesn't say const)
                let const_addr = matches!(e, Expr::AddrOf(x) if self.const_lvalue(x));
                let drops_const = (is_const(a) || const_addr) && !matches!(pointee(tos), Some(Type::Const(_)));
                drops_const || (tp.as_ref() != Some(&fp) && !matches!(tp, Some(Type::Void)) && !self.is_base_of(tp.as_ref(), &fp))
            }
            (Type::Bool, _) if e.as_int().is_some() => {
                let c = types::C_MODE.with(|c| c.get());
                return match (e.as_int() == Some(0), c) {
                    (true, false) => "false".into(),
                    (false, false) => "true".into(),
                    (true, true) => "0".into(),
                    (false, true) => "1".into(),
                };
            }
            (Type::Ptr(_), Type::Int { .. } | Type::Unknown { .. } | Type::Bool) => e.as_int() != Some(0),
            (Type::Ptr(_), Type::Named(_)) => true,
            // an object made from a pointer: its converting constructor (maybe `explicit`);
            // without one, the address of an object passed by value (the object there)
            (Type::Named(n), Type::Ptr(_)) if mwdec_lift::types::is_aggregate(self.db, tos) && e.as_int().is_none() => {
                let key = strip_template_args(n);
                let last = sig::split_scope(&key).1.to_string();
                let ptr_ctor = self.db.and_then(|db| db.decls.get(&format!("{key}::{last}"))).is_some_and(|ds| ds.iter().any(|d| d.params.len() == 1 && matches!(strip_cv(&d.params[0].ty), Type::Ptr(_) | Type::FuncPtr(_))));
                // (or a constructor from a reference to another type, the pointer's object:
                // `T(const Data& d) : mData(&d)`)
                let pointee_ty = pointee(froms).map(|t| strip_cv(t).clone());
                let ref_ctor = self.db.and_then(|db| db.decls.get(&format!("{key}::{last}"))).is_some_and(|ds| {
                    ds.iter().any(|d| {
                        d.params.len() == 1
                            && matches!(strip_cv(&d.params[0].ty), Type::Ref(x) if named(strip_cv(x)).is_some_and(|n| sig::split_scope(&strip_template_args(n)).1 != last) && pointee_ty.as_ref().map_or(true, |p| named(p).map_or(true, |a| sig::split_scope(&strip_template_args(a)).1 != last)))
                    })
                });
                return if ptr_ctor {
                    format!("{}({})", type_str(tos), self.expr(e, 0))
                } else if ref_ctor {
                    match e {
                        Expr::AddrOf(x) => format!("{}({})", type_str(tos), self.expr(x, 0)),
                        _ => format!("{}(*{})", type_str(tos), self.expr(e, 14)),
                    }
                } else {
                    format!("*({}){}", ptr_to(tos), self.expr(e, 14))
                };
            }
            (Type::FuncPtr(_), Type::Int { .. } | Type::Unknown { .. } | Type::Long { .. }) => e.as_int() != Some(0),
            (Type::Int { .. } | Type::Unknown { .. } | Type::Bool, Type::Ptr(_) | Type::FuncPtr(_)) => true,
            (Type::Named(a), Type::Named(b)) => sig::norm_name(a) != sig::norm_name(b) && !self.is_base_of(Some(tos), froms),
            (Type::Named(_), Type::Int { .. } | Type::Unknown { .. }) if mwdec_lift::types::is_aggregate(self.db, tos) && e.is_lvalue() => {
                // an untyped buffer holding the object: reinterpret it
                let byte_array = matches!(e, Expr::Var(v) if matches!(self.ir.vars[*v].ty, Type::Unknown { size } if size != 4 && size != 2 && size != 1 && size != 8));
                let x = self.expr(e, 14);
                return if byte_array { format!("*({}){x}", ptr_to(tos)) } else { format!("*({})&{x}", ptr_to(tos)) };
            }
            (Type::Named(_), Type::Int { .. } | Type::Unknown { .. }) => !mwdec_lift::types::is_aggregate(self.db, tos),
            (Type::Ref(_), _) => false,
            _ => false,
        };
        if matches!(e, Expr::Int { value: 0, .. }) && matches!(tos, Type::Ptr(_)) {
            return self.opts.null.clone();
        }
        if need_cast {
            if let Type::Named(_) = tos {
                if mwdec_lift::types::is_aggregate(self.db, tos) {
                    return self.expr(e, 0);
                }
            }
            format!("({}){}", type_str(to), self.expr(e, 14))
        } else {
            self.expr(e, 0)
        }
    }

    /// Is the member `typed_member` renders for `inner` read through a getter that returns a
    /// reference/pointer to const?
    fn typed_member_via_const_getter(&self, inner: &Expr, target: &Type) -> bool {
        let Some(db) = self.db else { return false };
        let (base, off, ptr) = match inner {
            Expr::Load { base, offset, .. } => (&**base, *offset, true),
            Expr::Member { base, offset, .. } => (&**base, *offset, false),
            _ => return false,
        };
        let bt = ty_of(base, self.vars());
        let Some(ct) = (if ptr { pointee(&bt).cloned() } else { Some(bt) }) else { return false };
        let ctr = mwdec_lift::types::resolve(Some(db), &ct).into_owned();
        let Some(cls) = named(&ctr) else { return false };
        let Some(path) = mwdec_lift::types::field_path_of_type(db, cls, off, target) else { return false };
        path.iter().any(|p| match p {
            PathElem::Field(n, owner) if !self.field_accessible(owner, n) => {
                self.accessor_decl(owner, n).map_or(false, |(_, rt)| matches!(pointee(&rt), Some(Type::Const(_))))
            }
            _ => false,
        })
    }

    /// Does the function's header declaration (or the const one of a const/non-const pair)
    /// return a pointer to const?
    fn returns_const_ptr(&self, s: &mwdec_core::FuncSig, nargs: usize) -> bool {
        if matches!(pointee(&s.ret), Some(Type::Const(_))) && is_ptr(&s.ret) {
            return true;
        }
        let Some(db) = self.db else { return false };
        let key = strip_template_args(&s.qualified_name);
        db.decls.get(&key).map_or(false, |ds| ds.iter().any(|d| d.params.len() == nargs && matches!(strip_cv(&d.ret), Type::Ptr(x) if matches!(**x, Type::Const(_)))))
    }

    /// Does the method's header declaration return a reference to const (or is it the const
    /// overload of a const/non-const pair, which a const object selects)?
    fn returns_const_ref(&self, s: &mwdec_core::FuncSig, nargs: usize) -> bool {
        // (the header declarations win over a signature reconstructed elsewhere)
        let key = strip_template_args(&s.qualified_name);
        if let Some(ds) = self.db.and_then(|db| db.decls.get(&key)).filter(|ds| ds.iter().any(|d| d.params.len() == nargs)) {
            return ds.iter().filter(|d| d.params.len() == nargs).all(|d| matches!(strip_cv(&d.ret), Type::Ref(x) if matches!(**x, Type::Const(_))));
        }
        matches!(strip_cv(&s.ret), Type::Ref(x) if matches!(**x, Type::Const(_)))
    }

    /// Is the lvalue `e` const in C++ (rooted at `this` of a const method, or at a parameter
    /// that is a pointer/reference to const)?
    fn const_lvalue(&self, e: &Expr) -> bool {
        let mut cur = e;
        loop {
            match cur {
                Expr::Load { base, .. } => {
                    // through a pointer: the pointee's constness is the pointer's
                    match &**base {
                        Expr::Var(v) => return self.const_root(*v, true),
                        Expr::AddrOf(x) => cur = x,
                        _ => return false,
                    }
                }
                Expr::Member { base, .. } | Expr::Index { base, .. } => cur = base,
                Expr::Var(v) => return self.const_root(*v, false),
                // a reference returned by a method: const when declared so, or when the object
                // is const and the method has a const overload returning a const reference
                Expr::Call { callee: Callee::Method { this, sig: ms, .. }, args, .. } => return self.const_ref_call(this, ms, args.len()),
                _ => return false,
            }
        }
    }

    fn const_ref_call(&self, this_e: &Expr, ms: &mwdec_core::FuncSig, nargs: usize) -> bool {
        // (only a reference result is an lvalue: a value result is a temporary)
        let key = strip_template_args(&ms.qualified_name);
        let by_ref = matches!(strip_cv(&ms.ret), Type::Ref(_))
            || self.db.and_then(|db| db.decls.get(&key)).is_some_and(|ds| ds.iter().any(|d| d.params.len() == nargs && matches!(strip_cv(&d.ret), Type::Ref(_))));
        if !by_ref {
            return false;
        }
        if self.returns_const_ref(ms, nargs) {
            return true;
        }
        let obj_const = match this_e {
            Expr::AddrOf(x) => self.const_lvalue(x),
            Expr::Var(v) => self.const_root(*v, true) || matches!(pointee(&self.ir.vars[*v].ty), Some(Type::Const(_))),
            e => matches!(pointee(&ty_of(e, self.vars())), Some(Type::Const(_))),
        };
        if !obj_const {
            return false;
        }
        let Some(db) = self.db else { return false };
        let key = strip_template_args(&ms.qualified_name);
        db.decls.get(&key).is_some_and(|ds| ds.iter().any(|d| d.params.len() == nargs && matches!(strip_cv(&d.ret), Type::Ref(x) if matches!(**x, Type::Const(_)))))
    }

    /// Does the method's class declare a non-const overload with the same arity?
    fn has_nonconst_overload(&self, s: &mwdec_core::FuncSig, nargs: usize) -> bool {
        let key = strip_template_args(&s.qualified_name);
        self.db.and_then(|db| db.decls.get(&key)).is_some_and(|ds| ds.iter().any(|d| !d.is_const && d.params.len() == nargs))
    }

    /// Is the object a method is called on (given by its address) const in C++?
    fn const_object(&self, this_e: &Expr) -> bool {
        match this_e {
            Expr::AddrOf(x) => self.const_lvalue(x) || matches!(self.decl_type_rw(x, true), Type::Const(_)),
            Expr::Var(v) => self.const_root(*v, true) || matches!(pointee(&self.ir.vars[*v].ty), Some(Type::Const(_))),
            Expr::Load { .. } | Expr::Member { .. } => matches!(pointee(&self.decl_type_rw(this_e, true)), Some(Type::Const(_))),
            Expr::Call { callee: Callee::Method { sig: cs, .. } | Callee::Direct { sig: cs, .. }, args, .. } => self.returns_const_ptr(cs, args.len()),
            e => matches!(pointee(&ty_of(e, self.vars())), Some(Type::Const(_))),
        }
    }

    fn const_root(&self, v: VarId, through_ptr: bool) -> bool {
        let var = &self.ir.vars[v];
        match var.kind {
            VarKind::This => self.ir.sig.is_const,
            VarKind::Param { index } => {
                let t = self.ir.sig.params.get(index).map(|p| p.ty.clone()).unwrap_or_else(|| var.ty.clone());
                match strip_cv(&t) {
                    Type::Ref(x) => matches!(**x, Type::Const(_)),
                    Type::Ptr(x) if through_ptr => matches!(**x, Type::Const(_)),
                    _ => false,
                }
            }
            // a local pointer to const (as declared)
            _ => through_ptr && matches!(local_type(&var.ty), Type::Ptr(x) if matches!(*x, Type::Const(_))),
        }
    }

    /// Declared C++ type of `e` as rendered: the member's declared type for accesses that render
    /// as a member path (the IR's access type may differ in constness or pointee).
    fn decl_type_rw(&self, e: &Expr, read: bool) -> Type {
        // an object array local (its address renders as the decayed array)
        match e {
            // a string literal is an array of plain `char` (not `signed char`)
            Expr::Str { .. } => return Type::Ptr(Box::new(Type::Const(Box::new(Type::Char)))),
            Expr::Var(v) if self.obj_arrays.contains_key(v) => return self.obj_arrays[v].clone(),
            // (a frame object declared as a byte array: see `local_decl`)
            Expr::AddrOf(x) if matches!(&**x, Expr::Var(v) if self.byte_array_local(*v).is_some()) => {
                let Expr::Var(v) = &**x else { unreachable!() };
                // (its name is its address unless it is word-pair sized: see `expr`)
                let n = self.byte_array_local(*v).unwrap();
                let byte = Type::Int { size: 1, signed: false };
                return Type::Ptr(Box::new(if matches!(self.ir.vars[*v].ty, Type::Unknown { size: 8 }) { Type::Array(Box::new(byte), n) } else { byte }));
            }
            Expr::AddrOf(x) => {
                if let Some(Type::Array(el, _)) = if let Expr::Var(v) = &**x { self.obj_arrays.get(v) } else { None } {
                    return Type::Ptr(el.clone());
                }
                // a function-local static in read-only data is declared const (see `global`)
                if let Expr::Global { symbol, ty } = &**x {
                    let ro = local_static_name(symbol).is_some()
                        && self.ir.globals.iter().any(|g| g.symbol == *symbol && g.init.is_some() && matches!(g.section.as_deref(), Some(".rodata" | ".sdata2")));
                    if ro {
                        let dt = self.gtypes.get(symbol).cloned().unwrap_or_else(|| extern_type(ty));
                        let el = match dt {
                            Type::Array(e, _) => *e,
                            t => t,
                        };
                        return Type::Ptr(Box::new(Type::Const(Box::new(el))));
                    }
                }
            }
            // byte-pointer arithmetic renders as pointer arithmetic (see `expr`): a byte pointer
            Expr::Binary { op: op @ (BinOp::Add | BinOp::Sub), l, r, ty } if !is_ptr(strip_cv(ty)) => {
                let lt = strip_cv(&ty_of(l, self.vars())).clone();
                let rt = strip_cv(&ty_of(r, self.vars())).clone();
                let byte_ptr = |t: &Type| is_ptr(t) && pointee(t).map_or(false, |p| scalar_size(strip_cv(p)) == Some(1));
                if byte_ptr(&lt) && !is_ptr(&rt) {
                    return lt;
                }
                if *op == BinOp::Add && byte_ptr(&rt) && !is_ptr(&lt) {
                    return rt;
                }
            }
            _ => {}
        }
        let it = ty_of(e, self.vars());
        if self.opts.raw_offsets {
            return it;
        }
        let Some(db) = self.db else { return it };
        let (base, off, ty, ptr) = match e {
            Expr::Load { base, offset, ty } => (&**base, *offset, ty, true),
            Expr::Member { base, offset, ty } => (&**base, *offset, ty, false),
            _ => return it,
        };
        if matches!(ty, Type::Unknown { size: 0 }) {
            return it;
        }
        let bt = ty_of(base, self.vars());
        let ct = if ptr { pointee(&bt).cloned() } else { Some(bt) };
        let Some(ct) = ct else { return it };
        let cr = mwdec_lift::types::resolve(Some(db), &ct).into_owned();
        let Some(cls) = named(&cr) else { return it };
        // the whole object (not its first member of the same size)
        if off == 0 && named(strip_cv(&mwdec_lift::types::resolve(Some(db), ty))).is_some_and(|n| sig::norm_name(n) == sig::norm_name(cls)) {
            return ty.clone();
        }
        let want = scalar_size(ty).unwrap_or(0);
        match field_path(db, cls, off, want) {
            Some((path, ft)) if access_ok(self.db, &ft, ty) && self.reachable(&path, read) => {
                // read through a getter: its declared return type (often const-qualified)
                if read && !(matches!(strip_cv(&ft), Type::Ref(_)) && is_ptr(ty)) {
                    if let Some(PathElem::Field(n, owner)) = path.last() {
                        if !self.field_accessible(owner, n) {
                            if let Some((_, rt)) = self.accessor_decl_c(owner, n, self.const_lvalue(e)) {
                                let rt = match strip_cv(&rt) {
                                    Type::Ref(x) => (**x).clone(),
                                    t => t.clone(),
                                };
                                if !matches!(rt, Type::Unknown { .. }) && !unresolved_named(db, &rt) {
                                    return rt;
                                }
                                // (a template's `const T* operator->() const`: the member's own
                                // pointer type, made pointer-to-const)
                                if let (Type::Ptr(rp), Type::Ptr(fp)) = (&rt, strip_cv(&ft)) {
                                    if matches!(**rp, Type::Const(_)) && !matches!(**fp, Type::Const(_)) {
                                        return Type::Ptr(Box::new(Type::Const(fp.clone())));
                                    }
                                }
                            }
                        }
                    }
                }
                match strip_cv(&ft) {
                    // reference members are read as pointers (see ptr_access)
                    Type::Ref(x) if is_ptr(ty) && read => Type::Ptr(x.clone()),
                    _ => ft,
                }
            }
            _ => it,
        }
    }

    /// Conversions the basic rules miss (typedef'd function pointers, enums, aggregate members
    /// accessed as scalars, calls of unknown return type): `Some(rendered)` when handled.
    fn coerce_extra(&mut self, e: &Expr, to: &Type, froms: &Type) -> Option<String> {
        let db = self.db;
        let tr = mwdec_lift::types::resolve(db, to).into_owned();
        let fr = mwdec_lift::types::resolve(db, froms).into_owned();
        let tr = strip_cv(&tr).clone();
        let fr = strip_cv(&fr).clone();
        let scalar = |t: &Type| matches!(t, Type::Int { .. } | Type::Long { .. } | Type::Char | Type::WChar | Type::Bool | Type::Unknown { .. } | Type::Float { .. });
        let intlike = |t: &Type| matches!(t, Type::Int { .. } | Type::Long { .. } | Type::Char | Type::WChar | Type::Bool | Type::Unknown { .. });
        let is_enum = |t: &Type| mwdec_lift::types::is_enum(db, t);
        let to_scalar = scalar(&tr) || is_ptr(&tr) || is_enum(&tr);
        // a negated condition as a (small) enum: the selection of the two constants, which MWCC
        // materializes without narrowing the 0/1 (`(E)!x` truncates to the enum's size)
        if let Expr::Unary { op: UnOp::Not, e: x, .. } = e {
            if is_enum(&tr) {
                let t = type_str(to);
                let c = self.expr(x, 4);
                return Some(format!("({c} ? ({t})0 : ({t})1)"));
            }
        }
        // an aggregate member read where a scalar is wanted: reinterpret its bytes
        if to_scalar && mwdec_lift::types::is_aggregate(db, &fr) && !matches!(tr, Type::Unknown { size: 0 }) {
            if let Expr::Load { base, offset, .. } | Expr::Member { base, offset, .. } = e {
                let ptr = matches!(e, Expr::Load { .. });
                return Some(self.raw_access(base, *offset, &tr, ptr));
            }
        }
        // an object lvalue (a reference returned by a call, a stack object) where a scalar is
        // wanted: its first bytes
        if to_scalar && mwdec_lift::types::is_aggregate(db, &fr) && !matches!(tr, Type::Unknown { size: 0 }) {
            let ref_call = matches!(e, Expr::Call { callee: Callee::Method { sig: cs, .. } | Callee::Direct { sig: cs, .. }, .. } if matches!(strip_cv(&cs.ret), Type::Ref(_)));
            if ref_call || matches!(e, Expr::Var(_) | Expr::Global { .. }) {
                let ps = type_str(&Type::Ptr(Box::new(access_type(&tr))));
                return Some(format!("*({ps})&{}", self.expr(e, 14)));
            }
        }
        // a call whose return type the lifter guessed: say what the value is used as
        if let Expr::Call { ret: Type::Unknown { .. }, .. } = e {
            if to_scalar && !matches!(tr, Type::Unknown { .. } | Type::Bool) {
                return Some(format!("({}){}", type_str(to), self.expr(e, 14)));
            }
        }
        // a method with const and non-const overloads may resolve to the const one (returning a
        // const pointer) on a const object
        // (or one declared to return a pointer/reference to const)
        if let (Expr::Call { callee: Callee::Method { sig: ms, .. } | Callee::Direct { sig: ms, .. }, args, .. }, Type::Ptr(tp)) = (e, &tr) {
            if !matches!(**tp, Type::Const(_)) && is_ptr(&fr) {
                let key = strip_template_args(&ms.qualified_name);
                let const_ret = |t: &Type| matches!(pointee(t), Some(Type::Const(_)));
                let risky = db.and_then(|db| db.decls.get(&key)).map_or(false, |ds| {
                    (ds.iter().any(|d| d.is_const && d.params.len() == args.len()) && ds.iter().any(|d| !d.is_const && d.params.len() == args.len()))
                        || ds.iter().any(|d| d.params.len() == args.len() && const_ret(&d.ret))
                });
                if risky {
                    return Some(format!("({}){}", type_str(to), self.expr(e, 14)));
                }
            }
        }
        // pointer arithmetic on a pointer to const stays a pointer to const
        if let (Expr::Binary { op: BinOp::Add | BinOp::Sub, l, r, .. }, Type::Ptr(tp)) = (e, &tr) {
            if !matches!(**tp, Type::Const(_)) {
                let cp = |x: &Expr, me: &Self| -> bool {
                    let t = match x {
                        Expr::Var(v) => local_type(&me.ir.vars[*v].ty),
                        x => me.decl_type_rw(x, true),
                    };
                    is_ptr(strip_cv(&t)) && matches!(pointee(strip_cv(&t)), Some(Type::Const(_)))
                };
                if cp(l, self) || cp(r, self) {
                    return Some(format!("({}){}", type_str(to), self.expr(e, 14)));
                }
            }
        }
        // the address of an object reached through `this` in a const method or through a
        // pointer/reference-to-const parameter is a pointer to const
        if let (Expr::AddrOf(x), Type::Ptr(tp)) = (e, &tr) {
            if !matches!(**tp, Type::Const(_)) && self.const_lvalue(x) {
                return Some(format!("({}){}", type_str(to), self.expr(e, 14)));
            }
        }
        // an aggregate global read where a scalar is wanted: its bytes
        if to_scalar && mwdec_lift::types::is_aggregate(db, &fr) && !matches!(tr, Type::Unknown { size: 0 }) && matches!(e, Expr::Global { .. }) {
            let ps = type_str(&Type::Ptr(Box::new(access_type(&tr))));
            return Some(format!("*({ps})&{}", self.expr(e, 14)));
        }
        // a pointer where an integer is wanted
        if matches!(tr, Type::Long { .. } | Type::Char | Type::WChar) && is_ptr(&fr) {
            return Some(format!("({}){}", type_str(to), self.expr(e, 14)));
        }
        // array-typed parameters (`va_list`): the object reinterpreted as that array type
        if let Type::Array(..) = tr {
            if !matches!(fr, Type::Array(..)) || fr != tr {
                let addr = if self.byte_array_base(e) || is_ptr(&fr) { self.expr(e, 14) } else { format!("&{}", self.expr(e, 14)) };
                return Some(format!("*({}){addr}", ptr_to(to)));
            }
        }
        let fptr = |t: &Type| matches!(t, Type::FuncPtr(_));
        let need = match (&tr, &fr) {
            // (function pointer types from DWARF are imprecise (`bool` vs `unsigned char`,
            // dropped `const`): leave those conversions to the source types)
            // (a cast between function pointer types changes no code; declared types of
            // members may differ from their DWARF spelling in `const` parameters)
            // (not into members: their DWARF types may have lost the `const` the header has)
            (a, b) if fptr(a) && fptr(b) => {
                !self.member_store
                    && !matches!(e, Expr::FuncAddr { symbol } if sig::sig_of(symbol, self.db).qualified_name.contains("::"))
                    && (type_str(a) != type_str(b) || matches!(e, Expr::Load { .. } | Expr::Member { .. } | Expr::Var(_)))
            }
            // (a function's address converted to the pointer type wanted: its declaration may
            // have other parameter types)
            (a, b) if fptr(a) => {
                !(matches!(e, Expr::Int { value: 0, .. }))
                    && match e {
                        // (C function pointer members are spelled as declared)
                        Expr::FuncAddr { symbol } => (!self.member_store || self.opts.c_mode) && !sig::sig_of(symbol, self.db).qualified_name.contains("::"),
                        _ => intlike(b) || is_ptr(b),
                    }
            }
            (a, b) if fptr(b) => intlike(a) || (is_ptr(a) && !matches!(pointee(a).map(strip_cv), Some(Type::Void))),
            (a, b) if is_enum(a) => (intlike(b) && e.as_int().is_none()) || (is_enum(b) && a != b) || e.as_int().is_some(),
            _ => false,
        };
        if need {
            if matches!(e, Expr::Int { value: 0, .. }) && fptr(&tr) {
                return Some(self.opts.null.clone());
            }
            return Some(format!("({}){}", type_str(to), self.expr(e, 14)));
        }
        None
    }

    fn is_base_of(&self, base: Option<&Type>, derived: &Type) -> bool {
        let (Some(db), Some(b)) = (self.db, base.and_then(named)) else { return false };
        let Some(d) = named(derived) else { return false };
        fn walk(db: &TypeDb, c: &str, target: &str, depth: u32) -> bool {
            if depth > 16 {
                return false;
            }
            let Some(cls) = sig::find_class(db, c) else { return false };
            cls.bases.iter().any(|x| sig::norm_name(&x.name) == sig::norm_name(target) || walk(db, &x.name, target, depth + 1))
        }
        walk(db, d, b, 0)
    }

    fn paren(&mut self, e: &Expr, min_prec: u8) -> String {
        let s = self.expr_inner(e);
        // raw accesses render as a unary deref
        let p = if s.starts_with("*(") && matches!(e, Expr::Load { .. } | Expr::Member { .. } | Expr::Index { .. }) { 14 } else { prec_of(e) };
        if p < min_prec {
            format!("({s})")
        } else {
            s
        }
    }

    fn expr(&mut self, e: &Expr, min_prec: u8) -> String {
        self.paren(e, min_prec)
    }

    /// Can a local of type `t` be declared without arguments (a default constructor, or no
    /// user-declared constructors)?
    /// In a constructor, members of a class without a default constructor that the lifted
    /// initializer list leaves out (the body assigns them): they must be initialized in the
    /// list, so each is copy-initialized from its own storage there (the body's assignment
    /// still sets the value).
    fn missing_member_inits(&self) -> Vec<String> {
        let ir = self.ir;
        let (Some(db), Some(cls)) = (self.db, ir.sig.this_class.as_deref()) else { return vec![] };
        if self.opts.c_mode || !sig::is_ctor(&ir.sig) {
            return vec![];
        }
        let Some(c) = sig::find_class(db, cls) else { return vec![] };
        let mut out = vec![];
        let ctor_declared = |t: &Type| -> bool {
            let r = mwdec_lift::types::resolve(Some(db), t).into_owned();
            let base = strip_template_args(named(strip_cv(&r)).unwrap_or_default());
            let last = sig::split_scope(&base).1.to_string();
            db.decls.get(&format!("{base}::{last}")).is_some_and(|ds| !ds.is_empty())
        };
        // base classes likewise (the body constructs them explicitly), from the object itself
        for b in c.bases.iter().filter(|b| !b.is_virtual) {
            let bt = Type::Named(b.name.clone());
            if ir.init_list.iter().any(|i| matches!(&i.target, InitTarget::Base(x) if sig::norm_name(x) == sig::norm_name(&b.name))) {
                continue;
            }
            if !ctor_declared(&bt) || self.default_constructible(&bt) || b.name.contains('<') {
                continue;
            }
            let ts = types::split_closers(&type_str(&bt));
            out.push(format!("{ts}(*({ts}*)this)"));
        }
        for f in &c.fields {
            if f.bitfield.is_some() || f.name.starts_with("__") {
                continue;
            }
            if ir.init_list.iter().any(|i| matches!(&i.target, InitTarget::Member(m) if *m == f.name)) {
                continue;
            }
            let ft = strip_cv(&f.ty);
            let fr = mwdec_lift::types::resolve(Some(db), ft).into_owned();
            if !matches!(strip_cv(&fr), Type::Named(_)) || !mwdec_lift::types::is_aggregate(Some(db), &fr) {
                continue;
            }
            // (only classes whose constructors are all known to take arguments; template
            // instances' constructors often default all of theirs, which the declarations
            // don't record)
            if !ctor_declared(ft) || self.default_constructible(ft) || named(strip_cv(&fr)).is_some_and(|n| n.contains('<')) {
                continue;
            }
            let ts = types::split_closers(&type_str(ft));
            out.push(format!("{n}(*({ts}*)&{n})", n = f.name));
        }
        out
    }

    fn default_constructible(&self, t: &Type) -> bool {
        let Some(db) = self.db else { return true };
        let r = mwdec_lift::types::resolve(Some(db), t).into_owned();
        let Some(cls) = named(&r) else { return true };
        if !mwdec_lift::types::is_aggregate(Some(db), &r) {
            return true;
        }
        let base = strip_template_args(cls);
        let last = sig::split_scope(&base).1.to_string();
        match db.decls.get(&format!("{base}::{last}")) {
            Some(ds) => ds.iter().any(|d| d.params.is_empty()),
            // no user-declared constructor: only trust that if the class's members were scanned
            // (nested classes of templates aren't)
            None => {
                let prefix = format!("{base}::");
                db.decls.range(prefix.clone()..).next().map_or(false, |(k, _)| k.starts_with(&prefix))
                    || sig::find_class(db, cls).map_or(false, |c| c.vtable.is_empty() && !c.fields.is_empty() && !base.contains("::"))
            }
        }
    }

    /// Does the emitter declare `symbol` itself (not in the context)?
    fn self_declared(&self, symbol: &str) -> bool {
        let known = self.db.map_or(false, |db| db.globals.contains_key(symbol));
        if known {
            return false;
        }
        if !(sig::demangle(symbol).is_none() || symbol.starts_with("lbl_") || symbol.contains("@unnamed@")) {
            // a namespace-scope variable the context doesn't declare (`Ns::sTable`); static
            // members of classes can't be declared outside their class
            return self.undeclared_ns_scope(symbol).is_some();
        }
        // a function declared in the context, referenced as data (its address)
        let name = symbol_name(symbol);
        !self.db.map_or(false, |db| db.decls.contains_key(&name) || db.functions.contains_key(symbol))
    }

    /// (namespace, name) of a demangled variable in a namespace (not a known class) the context
    /// doesn't declare.
    fn undeclared_ns_scope(&self, symbol: &str) -> Option<(String, String)> {
        let db = self.db?;
        let d = sig::demangle(symbol)?;
        if d.contains('(') {
            return None;
        }
        let q = strip_unnamed_ns(&d);
        let (scope, name) = sig::split_scope(&q);
        let scope = scope?;
        if sig::find_class(db, scope).is_some() || db.templates.contains_key(&strip_template_args(scope)) || scope.contains('<') || self.synth.contains_key(scope) {
            return None;
        }
        if db.globals.values().any(|(qn, _)| *qn == q) {
            return None;
        }
        Some((scope.to_string(), name.to_string()))
    }

    /// Is `base` (an aggregate lvalue) spelled as a byte array, i.e. its name is its address?
    fn byte_array_base(&self, base: &Expr) -> bool {
        let arr = |t: &Type| matches!(t, Type::Unknown { size } if *size != 4 && *size != 2 && *size != 1 && *size != 8);
        match base {
            Expr::Global { symbol, ty } if self.self_declared(symbol) => match self.gtypes.get(symbol) {
                Some(dt) => matches!(dt, Type::Array(..)),
                None => arr(ty),
            },
            // declared by the context: its declared type; else the emitter's own declaration
            // (static members of synthesized classes)
            Expr::Global { symbol, ty } if arr(ty) => match self.db.and_then(|db| db.globals.get(symbol)) {
                // (only an arithmetic scalar or an object needs its address taken; pointers and
                // arrays are already addresses)
                Some((_, dt)) => {
                    let r = strip_cv(&mwdec_lift::types::resolve(self.db, dt).into_owned()).clone();
                    !(matches!(r, Type::Float { .. } | Type::Int { .. } | Type::Bool | Type::Char | Type::Long { .. }) || (matches!(r, Type::Named(_)) && mwdec_lift::types::is_aggregate(self.db, &r)))
                }
                // (a static member of a class this draft synthesizes: declared as it's used)
                None => {
                    let synth_static = sig::demangle(symbol).is_some_and(|dm| sig::split_scope(&dm).0.is_some_and(|sc| self.synth.contains_key(sc)));
                    !synth_static || self.gtypes.get(symbol).map_or(true, |dt| matches!(dt, Type::Array(..)))
                }
            },
            _ => arr(&ty_of(base, self.vars())),
        }
    }

    fn global(&mut self, symbol: &str, ty: &Type) -> String {
        let name = symbol_name(symbol);
        if !self.self_declared(symbol) {
            return name;
        }
        let t = extern_type(ty);
        let dt = self.gtypes.get(symbol).cloned().unwrap_or_else(|| t.clone());
        if let Some(st) = local_static_name(symbol) {
            // function-local static (`init$90`): declared in the body, with its initial value
            // when the object has one (initialized data sections)
            let g = self.ir.globals.iter().find(|g| g.symbol == symbol);
            let init = g.and_then(|g| g.init.as_ref().map(|b| (b.clone(), g.section.clone().unwrap_or_default())));
            match init.as_ref().and_then(|(b, sec)| static_init(&dt, b).map(|i| (i, sec.clone()))) {
                Some((i, sec)) => {
                    let cq = if matches!(sec.as_str(), ".rodata" | ".sdata2") { "const " } else { "" };
                    self.local_statics.insert(format!("static {cq}{} = {i};", decl(&dt, &st)));
                }
                None => {
                    self.local_statics.insert(format!("static {};", decl(&dt, &st)));
                }
            }
            return reinterpret_global(st, &t, &dt);
        }
        if let Some(a) = self.ir.globals.iter().find(|g| g.symbol == symbol).and_then(|g| g.abs_addr) {
            // a variable at a fixed address (hardware registers), the project's own syntax
            self.externs.insert(format!("{} : {a:#010X};", decl(&dt, &name)));
            return reinterpret_global(name, &t, &dt);
        }
        if let Some((ns, n)) = self.undeclared_ns_scope(symbol) {
            let mut d = format!("extern {};", decl(&dt, &n));
            for part in ns.split("::").collect::<Vec<_>>().into_iter().rev() {
                d = format!("namespace {part} {{ {d} }}");
            }
            self.externs.insert(d);
        } else if symbol.starts_with("lbl_") || symbol.starts_with('@') {
            self.externs.insert(format!("extern {};", decl(&dt, &name)));
        } else if symbol.contains("@unnamed@") {
            self.externs.insert(format!("namespace {{ extern {}; }}", decl(&dt, &name)));
        } else if self.db.is_some() {
            self.externs.insert(format!("extern {};", decl(&dt, &name)));
        }
        reinterpret_global(name, &t, &dt)
    }

    /// Render an lvalue access of `ty` at byte `off` from pointer expression `base`.
    fn ptr_access(&mut self, base: &Expr, off: i32, ty: &Type) -> String {
        let read = !std::mem::replace(&mut self.lvalue_ctx, false);
        // `*&f()` of a value-returning call: the value (an rvalue has no address)
        if let (0, true, Expr::AddrOf(x)) = (off, read, base) {
            if let Expr::Call { ret, .. } = &**x {
                let rt = strip_cv(ret);
                let named_same = named(rt).is_some() && named(rt).map(sig::norm_name) == named(strip_cv(ty)).map(sig::norm_name);
                if !matches!(rt, Type::Ref(_) | Type::Void | Type::Unknown { .. }) && (same_scalar(rt, ty) || named_same) {
                    return self.expr(x, 14);
                }
                // (a call whose return type the lifter didn't know, read as a float: a float
                // result has no address either)
                if matches!(rt, Type::Unknown { .. }) && is_float(strip_cv(ty)) {
                    return self.expr(x, 14);
                }
            }

        }
        let bt = ty_of(base, self.vars());
        let pt = pointee(&bt).cloned();
        // the whole pointee object
        if off == 0 && named(ty).is_some() {
            if let Some(p) = &pt {
                if sig::norm_name(&format!("{:?}", strip_cv(p))) == sig::norm_name(&format!("{:?}", strip_cv(ty))) {
                    let b = self.expr(base, 14);
                    return format!("*{b}");
                }
            }
        }
        if !self.opts.raw_offsets {
            if let (Some(db), Some(p)) = (self.db, pt.as_ref()) {
                let pr = mwdec_lift::types::resolve(Some(db), p).into_owned();
                if let Some(cls) = named(&pr) {
                    let want = scalar_size(ty).unwrap_or(0);
                    if let Some((path, ft)) = field_path(db, cls, off, if matches!(ty, Type::Unknown { size: 0 }) { 0 } else { want }) {
                        if access_ok(self.db, &ft, ty) {
                            // a store through a pointer the declarations make a pointer to const
                            // (or a pointer-to-const parameter/local written through)
                            let bd = match base {
                                Expr::Load { .. } | Expr::Member { .. } => self.decl_type_rw(base, true),
                                Expr::Var(v) if Some(*v) != self.ir.this_var => local_type(&self.ir.vars[*v].ty),
                                _ => Type::Void,
                            };
                            let b = match (read, pointee(strip_cv(&bd))) {
                                (false, Some(Type::Const(x))) if is_ptr(strip_cv(&bd)) && self.opts.c_mode => format!("(({}){})", ptr_to(x), self.expr(base, 14)),
                                (false, Some(Type::Const(x))) if is_ptr(strip_cv(&bd)) => format!("const_cast<{}>({})", ptr_to(x), self.expr(base, 0)),
                                _ => self.expr(base, 15),
                            };
                            if !self.reachable(&path, read) || self.protected_through_other(cls, &path) {
                                // a protected member of another object: its public getter
                                if let (true, Some(g)) = (read, self.getter_path_str(&path)) {
                                    return format!("{b}->{g}");
                                }
                                return self.raw_access(base, off, ty, true);
                            }
                            // (a non-const getter first on a pointer to const)
                            let b = if read && self.path_needs_nonconst(&path) && (matches!(pointee(strip_cv(&bd)), Some(Type::Const(_))) || matches!(pointee(&ty_of(base, self.vars())), Some(Type::Const(_)))) {
                                format!("const_cast<{}*>({})", types::split_closers(cls), self.expr(base, 0))
                            } else {
                                b
                            };
                            let m = format!("{}->{}", b, self.path_str(&path, read));
                            // a reference member read as the pointer it is stored as
                            if read && matches!(strip_cv(&ft), Type::Ref(_)) && is_ptr(ty) {
                                return format!("(&{m})");
                            }
                            return m;
                        } else if let Some(s) = self.punned_member(base, cls, &path, &ft, ty, read) {
                            return s;
                        }
                    }
                }
            }
        }
        // *base when the pointee matches
        if off == 0 {
            if let Some(p) = &pt {
                if same_scalar(p, ty) {
                    let b = self.expr(base, 14);
                    return format!("*{b}");
                }
            }
        }
        let b = self.expr(base, 14);
        let t = if matches!(ty, Type::Unknown { size: 0 }) { Type::Int { size: 1, signed: false } } else { ty.clone() };
        // (a store's destination is never const)
        let t = if read { t } else { strip_cv(&t).clone() };
        // pointer-to-T spelling (declarator syntax for function pointers)
        let ts = type_str(&Type::Ptr(Box::new(access_type(&t))));
        let bc = self.byte_cast(base);
        if off == 0 {
            format!("*({ts}){b}")
        } else {
            format!("*({ts})({bc}{b} + {})", hex_off(off))
        }
    }

    /// `(char*)` for byte arithmetic on `base`, nothing when it already is a byte pointer: the
    /// older compiler (SDK units) keeps a converted address as a shared subexpression
    /// (`lwzu r0, 0xd8(r3)` / `stw r0, 0(r3)`) where `p + 0xd8` folds into each access.
    fn byte_cast(&self, base: &Expr) -> &'static str {
        let t = ty_of(base, self.vars());
        match pointee(&t).map(strip_cv) {
            Some(Type::Int { size: 1, .. }) | Some(Type::Char) => "",
            _ => "(char*)",
        }
    }

    /// Member of an aggregate lvalue (stack region, global, by-value param).
    fn member_access(&mut self, base: &Expr, off: i32, ty: &Type) -> String {
        let read = !std::mem::replace(&mut self.lvalue_ctx, false);
        if let Expr::Global { symbol, .. } = base {
            if let Some(t) = self.gstructs.get(symbol).and_then(|l| l.iter().find(|f| f.0 == off)).map(|f| f.1.clone()) {
                let dt = self.gtypes.get(symbol).cloned().unwrap_or_else(|| t.clone());
                let g = self.global(symbol, &dt);
                let m = format!("{g}.f{off:x}");
                return if extern_type(ty) == t { m } else { format!("(*({})&{m})", ptr_to(&extern_type(ty))) };
            }
        }
        let bt = ty_of(base, self.vars());
        // an arithmetic value has no address: the value itself (its bits read as another
        // same-sized scalar can't be spelled)
        // (likewise literals and calls returning a scalar by value)
        let rvalue = match base {
            Expr::Binary { .. } | Expr::Unary { .. } | Expr::Int { .. } | Expr::Float { .. } => true,
            Expr::Call { ret, .. } => !matches!(strip_cv(ret), Type::Ref(_) | Type::Void | Type::Unknown { .. }) && !mwdec_lift::types::is_aggregate(self.db, ret),
            _ => false,
        };
        if read && off == 0 && rvalue && scalar_size(ty).is_some() && scalar_size(&bt) == scalar_size(ty) && is_float(strip_cv(&bt)) == is_float(strip_cv(ty)) {
            return format!("({})", self.expr(base, 0));
        }
        if !self.opts.raw_offsets {
            if let Some(db) = self.db {
                let br = mwdec_lift::types::resolve(Some(db), &bt).into_owned();
                // a member of a reference member: of the referenced object
                let br = match strip_cv(&br) {
                    Type::Ref(inner) => mwdec_lift::types::resolve(Some(db), inner).into_owned(),
                    _ => br,
                };
                // the object a class template keeps in raw storage (`optional_object<T>`'s bytes):
                // through its accessor (`opt.data()`)
                if let (Some(cls), Some(tc)) = (named(&br), named(strip_cv(ty))) {
                    if let Some(acc) = storage_accessor(db, cls, off, tc) {
                        let b = self.expr(base, 15);
                        return format!("{b}.{acc}()");
                    }
                    // an element of a container template's raw storage: `v[i]` through its operator[]
                    if let Some(i) = element_index(db, cls, off, tc, ty) {
                        let b = self.expr(base, 15);
                        return format!("{b}[{i}]");
                    }
                }
                if let Some(cls) = named(&br) {
                    let want = scalar_size(ty).unwrap_or(0);
                    if let Some((path, ft)) = field_path(db, cls, off, if matches!(ty, Type::Unknown { size: 0 }) { 0 } else { want }) {
                        if access_ok(self.db, &ft, ty) {
                            // (a reference member as the object, not the pointer it's stored as)
                            if matches!(strip_cv(&bt), Type::Ref(_)) {
                                self.lvalue_ctx = true;
                            }
                            let b = self.expr(base, 15);
                            self.lvalue_ctx = false;
                            if !self.reachable(&path, read) {
                                return self.raw_access(base, off, ty, false);
                            }
                            let b = if read && self.path_needs_nonconst(&path) && self.const_lvalue(base) { format!("const_cast<{}&>({b})", types::split_closers(cls)) } else { b };
                            let m = format!("{}.{}", b, self.path_str(&path, read));
                            // a reference member read as the pointer it is stored as
                            if read && matches!(strip_cv(&ft), Type::Ref(_)) && is_ptr(ty) {
                                return format!("(&{m})");
                            }
                            return m;
                        }
                    }
                }
            }
        }
        let b = self.expr(base, 14);
        let t = if matches!(ty, Type::Unknown { size: 0 }) { Type::Int { size: 1, signed: false } } else { ty.clone() };
        // (a store's destination is never const)
        let t = if read { t } else { strip_cv(&t).clone() };
        // pointer-to-T spelling (declarator syntax for function pointers)
        let ts = type_str(&Type::Ptr(Box::new(access_type(&t))));
        let is_array = self.byte_array_base(base);
        let addr = if is_array { b } else { format!("&{b}") };
        if off == 0 {
            format!("*({ts}){addr}")
        } else {
            format!("*({ts})((char*){addr} + {})", hex_off(off))
        }
    }

    /// Whether a member named `name` of class `obj` (or a class between it and its base `base`)
    /// hides `base::name`, so an unqualified call wouldn't reach the base's method.
    fn name_hidden_below(&self, obj: &str, base: &str, name: &str) -> bool {
        let b = sig::norm_name(base);
        if sig::norm_name(obj) == b {
            return false;
        }
        let cur_cls = self.ir.sig.this_class.as_deref().map(sig::norm_name);
        let cur_name = sig::split_scope(&self.ir.sig.qualified_name).1;
        let mut seen: Vec<String> = vec![];
        let mut stack = vec![obj.to_string()];
        while let Some(c) = stack.pop() {
            let cn = sig::norm_name(&c);
            if cn == b || seen.contains(&cn) || seen.len() > 32 {
                continue;
            }
            if cur_cls.as_ref() == Some(&cn) && cur_name == name {
                return true;
            }
            seen.push(cn.clone());
            let Some(db) = self.db else { continue };
            if db.decls.contains_key(&format!("{c}::{name}")) {
                return true;
            }
            if let Some(k) = sig::find_class(db, &c) {
                let own = |m: &mwdec_core::FuncSig| sig::split_scope(&m.qualified_name).1 == name && m.this_class.as_deref().map(sig::norm_name).as_ref() == Some(&cn);
                if k.methods.iter().any(own) || k.vtable.iter().any(|v| own(&v.sig)) {
                    return true;
                }
                stack.extend(k.bases.iter().map(|x| x.name.clone()));
            }
        }
        false
    }

    fn method_name(&self, s: &mwdec_core::FuncSig) -> String {
        sig::split_scope(&s.qualified_name).1.to_string()
    }

    /// `obj->` / `obj.` prefix for a member call on `this_e`, with a cast when the static type
    /// doesn't know the class.
    /// Can the member path be spelled (every member accessible, or read through a getter)?
    fn reachable(&self, path: &[PathElem], read: bool) -> bool {
        for (i, p) in path.iter().enumerate() {
            if let PathElem::Field(n, owner) = p {
                // (the vtable pointer is no member one can name)
                if n.starts_with("__vptr") || n.starts_with("__vt") {
                    return false;
                }
                if self.field_accessible(owner, n) {
                    continue;
                }
                let last = i + 1 == path.len();
                if (read || !last) && self.usable_getter(owner, n, read).is_some() {
                    continue;
                }
                return false;
            }
        }
        true
    }

    /// A scalar (or enum) type, through cv and typedefs.
    fn scalar_target(&self, t: &Type) -> bool {
        let r = mwdec_lift::types::resolve(self.db, strip_cv(t)).into_owned();
        mwdec_lift::types::is_enum(self.db, &r) || (scalar_size(&r).is_some() && named(&r).is_none())
    }

    /// The class `t` names when it has a constructor taking one scalar by value
    /// (`explicit CMaterialList(u64 value)`).
    fn scalar_ctor_class(&self, t: &Type) -> Option<String> {
        let db = self.db?;
        let c = strip_cv(t);
        let n = named(c)?;
        let base = strip_template_args(n);
        let key = format!("{base}::{}", sig::split_scope(&base).1);
        let ok = db.decls.get(&key)?.iter().any(|d| {
            d.params.len() == 1 && !matches!(strip_cv(&d.params[0].ty), Type::Ref(_) | Type::Ptr(_)) && self.scalar_target(&d.params[0].ty) && !mwdec_lift::types::is_enum(Some(db), strip_cv(&d.params[0].ty))
        });
        ok.then(|| type_str(c))
    }

    /// `*(T*)((char*)p + off)` (pointer base) / `*(T*)((char*)&obj + off)` (aggregate lvalue).
    fn raw_access(&mut self, base: &Expr, off: i32, ty: &Type, ptr: bool) -> String {
        // an arithmetic value has no address: the value itself (its bits read as another
        // same-sized scalar can't be spelled)
        if !ptr && off == 0 && matches!(base, Expr::Binary { .. } | Expr::Unary { .. }) && scalar_size(ty).is_some() && scalar_size(&ty_of(base, self.vars())) == scalar_size(ty) {
            return format!("({})", self.expr(base, 0));
        }
        let b = self.expr(base, 14);
        let t = if matches!(ty, Type::Unknown { size: 0 }) { Type::Int { size: 1, signed: false } } else { ty.clone() };
        // pointer-to-T spelling (declarator syntax for function pointers)
        let ts = type_str(&Type::Ptr(Box::new(access_type(&t))));
        let bc = if ptr { self.byte_cast(base) } else { "(char*)" };
        let addr = if ptr { b } else { format!("&{b}") };
        if off == 0 {
            format!("*({ts}){addr}")
        } else {
            format!("*({ts})({bc}{addr} + {})", hex_off(off))
        }
    }

    /// For a store into an inaccessible member that has an inline setter
    /// (`void SetX(T v) { mX = v; }`): (object prefix ending in `->`/`.`, setter, param type).
    fn setter_for(&mut self, dst: &Expr) -> Option<(String, String, Option<Type>)> {
        let db = self.db?;
        if self.opts.raw_offsets {
            return None;
        }
        // resolve the member path
        let (base, off, ptr, size, mask) = match dst {
            Expr::Load { base, offset, ty } => (&**base, *offset, true, mwdec_lift::types::size_of(Some(db), ty).unwrap_or(0), None),
            Expr::Member { base, offset, ty } => (&**base, *offset, false, mwdec_lift::types::size_of(Some(db), ty).unwrap_or(0), None),
            Expr::BitField { base, shift, width, .. } => match &**base {
                Expr::Load { base: b, offset, ty } => (&**b, *offset, true, scalar_size(ty)?, Some((((1u64 << width) - 1) << shift) as u32)),
                Expr::Member { base: b, offset, ty } => (&**b, *offset, false, scalar_size(ty)?, Some((((1u64 << width) - 1) << shift) as u32)),
                _ => return None,
            },
            _ => return None,
        };
        let bt = ty_of(base, self.vars());
        let ct = if ptr { pointee(&bt)?.clone() } else { bt };
        let cls = named(&mwdec_lift::types::resolve(Some(db), &ct).into_owned())?.to_string();
        let path = match mask {
            Some(m) => mwdec_lift::types::bitfield_at(db, &cls, off, size, m)?.0,
            None => field_path(db, &cls, off, size)?.0,
        };
        let (last, prefix) = path.split_last()?;
        let PathElem::Field(name, owner) = last else { return None };
        // the object path must be directly accessible (a getter would return a const view)
        let direct = prefix.iter().all(|p| match p {
            PathElem::Field(n, o) => self.field_accessible(o, n),
            _ => true,
        });
        if self.field_accessible(owner, name) || !direct {
            return None;
        }
        let key = format!("{}::", strip_template_args(owner));
        let mut found = None;
        for (qn, ds) in db.decls.range(key.clone()..) {
            if !qn.starts_with(&key) {
                break;
            }
            for d in ds {
                if d.params.len() != 1 || d.is_static {
                    continue;
                }
                let pname = d.params[0].name.clone().unwrap_or_default();
                let want = format!("{name} = {pname} ;");
                if !pname.is_empty() && d.inline_body.as_deref() == Some(want.as_str()) {
                    found = Some((qn[key.len()..].to_string(), d.params[0].ty.clone()));
                }
            }
        }
        let (m, pt) = found?;
        let b = self.expr(base, 15);
        let mut p = self.path_str(prefix, true);
        if !p.is_empty() {
            p.push('.');
        }
        let sep = if ptr { "->" } else { "." };
        Some((format!("{b}{sep}{p}"), m, Some(pt)))
    }

    /// `obj->flag` for a bitfield in the storage unit accessed by `base`.
    fn bitfield_name(&mut self, base: &Expr, shift: u8, width: u8, read: bool) -> Option<String> {
        if self.opts.raw_offsets {
            return None;
        }
        let db = self.db?;
        let (b, off, ptr, ty) = match base {
            Expr::Load { base, offset, ty } => (&**base, *offset, true, ty),
            Expr::Member { base, offset, ty } => (&**base, *offset, false, ty),
            _ => return None,
        };
        let size = scalar_size(ty)?;
        let bt = ty_of(b, self.vars());
        let ct = if ptr { pointee(&bt)?.clone() } else { bt };
        let ctr = mwdec_lift::types::resolve(Some(db), &ct).into_owned();
        let cls = named(&ctr)?.to_string();
        let mask = (((1u64 << width) - 1) << shift) as u32;
        let (path, _) = mwdec_lift::types::bitfield_at(db, &cls, off, size, mask)?;
        // members of other classes (incl. bases, which may be private there): read through a
        // getter if there is one, else fall back to explicit shifts/masks (always compiles)
        let own = self.own_class().map(|o| sig::norm_name(&o));
        let _ = &own;
        let foreign = path.iter().any(|p| matches!(p, PathElem::Field(n, owner) if !self.field_accessible(owner, n)));
        if foreign {
            if !read {
                return None;
            }
            if let Some(PathElem::Field(n, owner)) = path.last() {
                self.accessor(owner, n)?;
            }
        }
        let p = self.path_str(&path, read);
        // a store through a pointer to const
        // (not `this` of a const method: members written there are `mutable`)
        let bd = match b {
            Expr::Load { .. } | Expr::Member { .. } => {
                let d = self.decl_type_rw(b, true);
                if is_ptr(strip_cv(&d)) { d } else { ty_of(b, self.vars()) }
            }
            _ => Type::Void,
        };
        let bs = match (read, ptr, pointee(strip_cv(&bd))) {
            (false, true, Some(Type::Const(x))) => format!("const_cast<{}>({})", ptr_to(x), self.expr(b, 0)),
            _ => self.expr(b, 15),
        };
        Some(format!("{bs}{}{p}", if ptr { "->" } else { "." }))
    }

    /// `base->path` for an lvalue `inner` (Load/Member at an offset) whose member type is
    /// `target`, through the TypeDb.
    fn typed_member(&mut self, inner: &Expr, target: &Type) -> Option<String> {
        if self.opts.raw_offsets {
            return None;
        }
        let db = self.db?;
        let (base, off, ptr) = match inner {
            Expr::Load { base, offset, .. } => (&**base, *offset, true),
            Expr::Member { base, offset, .. } => (&**base, *offset, false),
            _ => return None,
        };
        let bt = ty_of(base, self.vars());
        let ct = if ptr { pointee(&bt)?.clone() } else { bt };
        let ctr = mwdec_lift::types::resolve(Some(db), &ct).into_owned();
        let cls = named(&ctr)?.to_string();
        let path = mwdec_lift::types::field_path_of_type(db, &cls, off, target)?;
        if !self.reachable(&path, true) {
            return None;
        }
        let p = self.path_str(&path, true);
        let b = self.expr(base, 15);
        if p.is_empty() {
            return Some(if ptr { format!("*{b}") } else { b });
        }
        Some(format!("{b}{}{p}", if ptr { "->" } else { "." }))
    }

    /// `delete p` for a deleting-destructor call on the object `this_e` points at (cast to the
    /// destructor's class when the pointer isn't typed).
    fn delete_expr(&mut self, this_e: &Expr, class: Option<&str>) -> String {
        let pre = self.object_prefix(this_e, class);
        if let Some(p) = pre.strip_suffix("->") {
            format!("delete {p}")
        } else if let Some(o) = pre.strip_suffix('.') {
            format!("delete &{o}")
        } else {
            format!("delete {}", self.expr(this_e, 14))
        }
    }

    fn object_prefix(&mut self, this_e: &Expr, class: Option<&str>) -> String {
        self.object_prefix_c(this_e, class, true)
    }

    /// Prefix for calling a method on the object `this_e` points at; `is_const_method` false
    /// adds a `const_cast` when the object is const.
    fn object_prefix_c(&mut self, this_e: &Expr, class: Option<&str>, is_const_method: bool) -> String {
        // a method of the member at offset 0 (`this->mPos.IsEqu(...)` called with `this`)
        if let Some(c) = class {
            if let Some(s) = self.offset0_member(this_e, c, is_const_method) {
                return s;
            }
            // an object whose static class doesn't have the method (a base-class pointer to a
            // derived object): downcast
            let static_t = match this_e {
                Expr::AddrOf(inner) => Some(local_type(&ty_of(inner, self.vars()))),
                e => pointee(&ty_of(e, self.vars())).cloned(),
            };
            if let (Some(st), Some(db)) = (static_t, self.db) {
                let sr = mwdec_lift::types::resolve(Some(db), &st).into_owned();
                if let Some(sc) = named(&sr) {
                    // (or a class this draft synthesizes: unrelated to any context class)
                    let known = sig::find_class(db, sc).is_some() && (sig::find_class(db, c).is_some() || self.synth.contains_key(c));
                    if known && sig::norm_name(sc) != sig::norm_name(c) && !self.is_base_of(Some(&Type::Named(c.to_string())), &Type::Named(sc.to_string())) {
                        let o = self.expr(this_e, 14);
                        let cst = if matches!(sr, Type::Const(_)) && is_const_method { "const " } else { "" };
                        return format!("(({cst}{}*){o})->", types::split_closers(c));
                    }
                } else if matches!(this_e, Expr::Call { .. }) && sig::find_class(db, c).is_some() && matches!(strip_cv(&sr), Type::Int { size: 1, .. } | Type::Char | Type::Void) {
                    // a call returning a byte pointer to the object (`AfterEnd()` of a header
                    // followed by the next one): the object there
                    let o = self.expr(this_e, 14);
                    let cst = if matches!(sr, Type::Const(_)) && is_const_method { "const " } else { "" };
                    return format!("(({cst}{}*){o})->", types::split_closers(c));
                }
            }
        }
        if let Expr::AddrOf(inner) = this_e {
            let it = ty_of(inner, self.vars());
            let it = match it {
                Type::Ref(x) => *x,
                t => t,
            };
            if let (Some(c), Type::Unknown { size: 0 }) = (class, &it) {
                let target = Type::Named(c.to_string());
                if let Some(s) = self.typed_member(inner, &target) {
                    // a const view (getter returning a const reference, const `this`): a
                    // non-const method needs the object non-const
                    if !is_const_method && (self.typed_member_via_const_getter(inner, &target) || (self.const_lvalue(inner) && !self.read_via_nonconst_getter(inner))) {
                        return format!("const_cast<{c}&>({s}).");
                    }
                    return format!("{s}.");
                }
            }
            if named(&it).is_some() || class.is_none() {
                if !is_const_method && matches!(&**inner, Expr::Load { .. } | Expr::Member { .. }) {
                    let dt = self.decl_type_rw(inner, true);
                    // (a getter returning a non-const reference gives a modifiable object even
                    // on a const one)
                    if matches!(dt, Type::Const(_)) || (self.const_lvalue(inner) && !self.read_via_nonconst_getter(inner)) {
                        let n = named(strip_cv(&dt)).filter(|_| !matches!(dt, Type::Unknown { .. })).or(named(&it)).map(|s| s.to_string());
                        if let Some(n) = n {
                            return format!("const_cast<{n}&>({}).", self.expr(inner, 0));
                        }
                    }
                }
                // a const object (a reference-to-const parameter, a const local)
                if !is_const_method && matches!(&**inner, Expr::Var(v) if matches!(self.ir.vars[*v].ty, Type::Const(_)) || matches!(&self.ir.vars[*v].ty, Type::Ref(r) if matches!(**r, Type::Const(_)))) {
                    if let Some(n) = named(strip_cv(&it)) {
                        let n = n.to_string();
                        return format!("const_cast<{n}&>({}).", self.expr(inner, 0));
                    }
                }
                if !is_const_method {
                    if let Expr::Call { callee: Callee::Method { sig: cs, .. } | Callee::Direct { sig: cs, .. }, args, .. } = &**inner {
                        if self.returns_const_ref(cs, args.len()) {
                            if let Some(n) = named(&it) {
                                let n = n.to_string();
                                return format!("const_cast<{n}&>({}).", self.expr(inner, 0));
                            }
                        }
                    }
                }
                // (a reference member is the object itself here, not the pointer it's stored as)
                if matches!(ty_of(inner, self.vars()), Type::Ref(_)) {
                    self.lvalue_ctx = true;
                }
                let s = self.expr(inner, 15);
                self.lvalue_ctx = false;
                return format!("{s}.");
            }
        }
        let t = ty_of(this_e, self.vars());
        let has_class = pointee(&t).and_then(named).is_some();
        // (a pointer read through a getter returning a pointer to const, or a call that may
        // resolve to an overload returning one: `const_cast` to the same type is harmless)
        let rendered_const = match this_e {
            Expr::Load { .. } | Expr::Member { .. } => matches!(pointee(&self.decl_type_rw(this_e, true)), Some(Type::Const(_))),
            Expr::Call { callee: Callee::Method { sig: cs, .. } | Callee::Direct { sig: cs, .. }, args, .. } => self.returns_const_ptr(cs, args.len()),
            _ => false,
        };
        let is_const_obj = matches!(pointee(&t), Some(Type::Const(_))) || rendered_const;
        if matches!(this_e, Expr::Var(v) if self.ir.this_var == Some(*v)) && self.this_local {
            return "self->".into();
        }
        if matches!(this_e, Expr::Var(v) if self.ir.this_var == Some(*v)) {
            if is_const_obj && !is_const_method {
                if let Some(c) = pointee(&t).and_then(named) {
                    return format!("const_cast<{c}*>(this)->");
                }
            }
            return "this->".into();
        }
        if has_class || class.is_none() {
            if is_const_obj && !is_const_method {
                if let Some(c) = pointee(&t).and_then(named) {
                    return format!("const_cast<{c}*>({})->", self.expr(this_e, 0));
                }
            }
            format!("{}->", self.expr(this_e, 15))
        } else {
            let c = types::split_closers(class.unwrap());
            format!("(({c}*){})->", self.expr(this_e, 14))
        }
    }

    /// `obj->member.` / `obj.member.` when the object's class is unrelated to the method's class
    /// `c` but has a member of that class (or derived from it) at offset 0.
    fn offset0_member(&mut self, this_e: &Expr, c: &str, is_const_method: bool) -> Option<String> {
        let db = self.db?;
        let (base_s, ptr, objt) = match this_e {
            Expr::AddrOf(inner) => {
                let it = ty_of(inner, self.vars());
                (inner.as_ref().clone(), false, local_type(&it))
            }
            e => {
                let t = ty_of(e, self.vars());
                (e.clone(), true, pointee(&t)?.clone())
            }
        };
        let objr = mwdec_lift::types::resolve(Some(db), &objt).into_owned();
        let oc = named(&objr)?.to_string();
        if sig::norm_name(&oc) == sig::norm_name(c) || self.is_base_of(Some(&Type::Named(c.to_string())), &Type::Named(oc.clone())) {
            return None;
        }
        let path = mwdec_lift::types::field_path_of_type(db, &oc, 0, &Type::Named(c.to_string()))?;
        if path.iter().all(|p| matches!(p, PathElem::Base(_))) || !self.reachable(&path, true) {
            return None;
        }
        let p = self.path_str(&path, true);
        let b = if ptr && matches!(this_e, Expr::Var(v) if self.ir.this_var == Some(*v)) { "this".to_string() } else { self.expr(&base_s, 15) };
        let s = format!("{b}{}{p}", if ptr { "->" } else { "." });
        // a non-const method on a member read through a getter returning a reference to const
        let via_const_getter = path.iter().any(|q| match q {
            PathElem::Field(n, owner) if !self.field_accessible(owner, n) => self.accessor_decl(owner, n).map_or(false, |(_, rt)| matches!(pointee(&rt), Some(Type::Const(_)))),
            _ => false,
        });
        if !is_const_method && via_const_getter {
            return Some(format!("const_cast<{}&>({s}).", types::split_closers(c)));
        }
        Some(format!("{s}."))
    }

    /// Access of a method from its header declaration (None when undeclared).
    fn method_access(&self, s: &mwdec_core::FuncSig, nargs: usize) -> Option<mwdec_core::Access> {
        let db = self.db?;
        let key = strip_template_args(&s.qualified_name);
        let ds = db.decls.get(&key)?;
        ds.iter().find(|d| d.params.len() == nargs && d.is_const == s.is_const).or_else(|| ds.iter().find(|d| d.params.len() == nargs)).map(|d| d.access)
    }

    fn ctor_factory(&self, owner: &str, nargs: usize) -> Option<String> {
        let db = self.db?;
        let key = strip_template_args(owner);
        let last = sig::split_scope(&key).1.to_string();
        // every constructor of that arity is out of reach
        let ctors: Vec<&mwdec_core::DeclInfo> = db.decls.get(&format!("{key}::{last}"))?.iter().filter(|d| d.params.len() == nargs).collect();
        if ctors.is_empty() || ctors.iter().any(|d| self.member_access_ok(owner, d.access)) {
            return None;
        }
        let prefix = format!("{key}::");
        for (name, ds) in db.decls.range(prefix.clone()..) {
            if !name.starts_with(&prefix) {
                break;
            }
            let m = &name[prefix.len()..];
            if m.contains("::") {
                continue;
            }
            for d in ds {
                if !d.is_static || d.access != mwdec_core::Access::Public || d.params.len() != nargs {
                    continue;
                }
                let names: Vec<String> = d.params.iter().map(|p| p.name.clone().unwrap_or_default()).collect();
                let want = format!("return {last} ( {} ) ;", names.join(" , "));
                if d.inline_body.as_deref() == Some(want.as_str()) && names.iter().all(|n| !n.is_empty()) {
                    return Some(m.to_string());
                }
            }
        }
        None
    }

    /// Can this function call a method of `owner` with the given access?
    fn member_access_ok(&self, owner: &str, access: mwdec_core::Access) -> bool {
        let own = self.own_class();
        let n = |s: &str| sig::norm_name(&strip_template_args(s));
        if own.as_deref().map(n) == Some(n(owner)) {
            return true;
        }
        match access {
            mwdec_core::Access::Public => true,
            mwdec_core::Access::Protected => self.befriended(owner) || own.map_or(false, |o| self.is_base_of(Some(&Type::Named(owner.to_string())), &Type::Named(o.clone()))),
            mwdec_core::Access::Private => self.befriended(owner),
        }
    }

    /// How to spell a call of an inaccessible method (see `Routed`).
    fn route_inaccessible(&self, s: &mwdec_core::FuncSig, nargs: usize) -> Option<Routed> {
        let db = self.db?;
        let owner = s.this_class.clone()?;
        let access = self.method_access(s, nargs)?;
        if self.member_access_ok(&owner, access) {
            return None;
        }
        let key = strip_template_args(&owner);
        let name = self.method_name(s);
        let last = sig::split_scope(&key).1.to_string();
        if nargs == 0 {
            let dt = format!("{key}::~{last}");
            if db.decls.get(&dt).map_or(false, |ds| ds.iter().any(|d| d.inline_body.as_deref().map(str::trim) == Some(&format!("{name} ( ) ;")))) {
                return Some(Routed::Dtor(format!("~{last}")));
            }
        }
        let prefix = format!("{key}::");
        for (qn, ds) in db.decls.range(prefix.clone()..) {
            if !qn.starts_with(&prefix) {
                break;
            }
            let m = &qn[prefix.len()..];
            if m.contains("::") || m == name {
                continue;
            }
            for d in ds {
                if d.params.len() != nargs || d.access != mwdec_core::Access::Public || d.is_static {
                    continue;
                }
                let Some(body) = d.inline_body.as_deref() else { continue };
                let pnames: Vec<String> = d.params.iter().map(|p| p.name.clone().unwrap_or_default()).collect();
                if pnames.iter().any(|p| p.is_empty()) {
                    continue;
                }
                let call = format!("{name} ( {} ) ;", pnames.join(" , "));
                let call = call.replace("(  )", "( )");
                let b = body.trim();
                if b == call || b == format!("return {call}") {
                    return Some(Routed::Wrapper(m.to_string()));
                }
            }
        }
        None
    }

    /// Statement `obj.m()` where `m` is the body of the class's inline destructor and `obj` is a
    /// local (destroyed implicitly at scope end) or a member of `this` in a destructor.
    fn implicit_destruction(&self, e: &Expr) -> bool {
        let Expr::Call { callee: Callee::Method { sig: s, this, .. }, args, .. } = e else { return false };
        if !args.is_empty() || !matches!(self.route_inaccessible(s, 0), Some(Routed::Dtor(_))) {
            return false;
        }
        match &**this {
            Expr::AddrOf(x) => match &**x {
                Expr::Var(v) => matches!(self.ir.vars[*v].kind, VarKind::Stack { .. }),
                Expr::Load { base, .. } => sig::is_dtor(&self.ir.sig) && matches!(&**base, Expr::Var(v) if self.ir.this_var == Some(*v)),
                _ => false,
            },
            _ => false,
        }
    }

    /// `(obj->*pmf)(args)` for `__ptmf_scall(obj, &pmf, args...)`. The member pointer is read
    /// through its address with an explicit pointer-to-member type: the enclosing method's own
    /// signature when it forwards its parameters (the usual thunk), else the argument types.
    fn ptmf_call(&mut self, args: &[Expr], ret: &Type) -> Option<String> {
        let [obj, pm, rest @ ..] = args else { return None };
        let ir = self.ir;
        let ot = ty_of(obj, self.vars());
        let cls = pointee(&ot).and_then(named).map(|s| s.to_string()).or_else(|| ir.sig.this_class.clone())?;
        let forwards = ir.this_var.is_some() && rest.len() == ir.params.len() && ir.sig.this_class.as_deref().map(sig::norm_name) == Some(sig::norm_name(&cls));
        let (rt, ps, is_const, rendered): (Type, Vec<String>, bool, Vec<String>) = if forwards {
            let rt = if matches!(ir.sig.ret, Type::Unknown { size: 0 }) { Type::Void } else { ir.sig.ret.clone() };
            let ps: Vec<String> = (0..ir.params.len())
                .map(|i| ir.decl_params.get(i).filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| type_str(&ir.vars[ir.params[i]].ty)))
                .collect();
            let mut out = vec![];
            for (i, a) in rest.iter().enumerate() {
                let pv = ir.params[i];
                let direct = match a {
                    Expr::Var(v) => *v == pv,
                    Expr::AddrOf(x) => matches!(&**x, Expr::Var(v) if *v == pv || matches!(ir.vars[*v].kind, VarKind::Stack { .. })),
                    _ => false,
                };
                if direct {
                    out.push(ir.vars[pv].name.clone());
                } else {
                    let pt = ir.sig.params.get(i).map(|p| p.ty.clone()).unwrap_or(Type::Unknown { size: 4 });
                    out.push(self.coerce(a, &pt));
                }
            }
            (rt, ps, ir.sig.is_const, out)
        } else {
            let ps: Vec<String> = rest.iter().map(|a| type_str(&value_type(&local_type(&ty_of(a, self.vars()))))).collect();
            let out: Vec<String> = rest.iter().map(|a| self.expr(a, 0)).collect();
            (value_type(&local_type(ret)), ps, false, out)
        };
        let cq = if is_const { " const" } else { "" };
        let pmt = format!("{} ({cls}::**)({}){cq}", type_str(&rt), ps.join(", "));
        let o = self.expr(obj, 14);
        let p = self.expr(pm, 14);
        Some(format!("({o}->*(*({pmt}){p}))({})", rendered.join(", ")))
    }

    /// MWCC runtime helpers (64-bit arithmetic, float->unsigned) are what the compiler emits for
    /// plain operators: spell the operator.
    fn runtime_op(&mut self, symbol: &str, args: &[Expr]) -> Option<String> {
        let ll = Type::Int { size: 8, signed: true };
        let ull = Type::Int { size: 8, signed: false };
        let bin = |me: &mut Self, op: &str, t: &Type| -> Option<String> {
            let [a, b] = args else { return None };
            let l = me.expr(a, 14);
            let r = me.expr(b, 14);
            Some(format!("(({}){l} {op} {r})", type_str(t)))
        };
        let cvt = |me: &mut Self, t: &str| -> Option<String> {
            let [a] = args else { return None };
            Some(format!("(({t}){})", me.expr(a, 14)))
        };
        match symbol {
            "__shl2i" => bin(self, "<<", &ll),
            "__shr2i" => bin(self, ">>", &ll),
            "__shr2u" => bin(self, ">>", &ull),
            "__div2i" => bin(self, "/", &ll),
            "__div2u" => bin(self, "/", &ull),
            "__mod2i" => bin(self, "%", &ll),
            "__mod2u" => bin(self, "%", &ull),
            "__cvt_fp2unsigned" => cvt(self, "unsigned int"),
            "__cvt_dbl_ull" | "__cvt_flt_ull" => cvt(self, "unsigned long long"),
            "__cvt_dbl_ll" | "__cvt_flt_ll" => cvt(self, "long long"),
            "__cvt_ll_dbl" | "__cvt_sll_dbl" => cvt(self, "double"),
            "__cvt_ull_dbl" => cvt(self, "double"),
            "__cvt_ll_flt" | "__cvt_sll_flt" => cvt(self, "float"),
            "__cvt_ull_flt" => cvt(self, "float"),
            _ => None,
        }
    }

    /// The object (as an lvalue expression) a member call is made on, for operator syntax.
    fn object_lvalue(&mut self, this_e: &Expr, class: Option<&str>) -> String {
        self.object_lvalue_c(this_e, class, true)
    }

    /// `object_lvalue` for a method that is const or not (a non-const operator on a const view
    /// of the object needs it non-const).
    fn object_lvalue_c(&mut self, this_e: &Expr, class: Option<&str>, is_const_method: bool) -> String {
        let pre = self.object_prefix_c(this_e, class, is_const_method);
        if let Some(p) = pre.strip_suffix("->") {
            if p == "this" {
                return "*this".into();
            }
            return format!("*{p}");
        }
        pre.strip_suffix('.').unwrap_or(&pre).to_string()
    }

    fn args(&mut self, args: &[Expr], sig: Option<&mwdec_core::FuncSig>) -> String {
        let mut v = vec![];
        let keep = sig.map_or(args.len(), |s| self.args_without_defaults(args, s));
        for (i, a) in args.iter().enumerate().take(keep) {
            // unknown parameter types (undeclared functions, declared from these arguments): as is
            let pt = sig.and_then(|s| s.params.get(i)).map(|p| p.ty.clone()).filter(|t| !matches!(t, Type::Unknown { .. }));
            // an argument already of the parameter's integer type spelled through a typedef (`s8`
            // for a `signed char`): an explicit cast would re-extend it (`extsb`), passing it doesn't
            let same_int = pt.as_ref().is_some_and(|t| {
                matches!(strip_cv(t), Type::Named(_))
                    && matches!(mwdec_lift::types::resolve(self.db, strip_cv(t)).as_ref(), r @ Type::Int { .. } if *r == value_type(&ty_of(a, self.vars())))
            });
            // (the count-leading-zeros intrinsic takes an `unsigned int`: a pointer is converted)
            let clz_ptr = sig.is_some_and(|s| s.qualified_name == "__cntlzw") && is_ptr(&ty_of(a, self.vars()));
            let s = match pt {
                Some(_) if same_int => self.expr(a, 0),
                Some(t) => self.coerce(a, &t),
                None if clz_ptr => format!("(unsigned int){}", self.expr(a, 14)),
                None => self.expr(a, 0),
            };
            v.push(s);
        }
        v.join(", ")
    }

    /// How many leading arguments to pass: trailing arguments equal to the declaration's
    /// default arguments are left out, as long as the shorter call still means the same
    /// overload (draft variant [`mwdec_lift::variants::EXPLICIT_DEFAULT_ARGS`] keeps them).
    fn args_without_defaults(&self, args: &[Expr], sig: &mwdec_core::FuncSig) -> usize {
        let Some(db) = self.db else { return args.len() };
        let key = strip_template_args(&sig.qualified_name);
        let Some(decls) = db.decls.get(&key) else { return args.len() };
        let Some(d) = decls.iter().find(|d| d.params.len() == args.len() && d.defaults.len() == args.len() && !d.variadic) else { return args.len() };
        let mut keep = args.len();
        while keep > 0 && d.defaults[keep - 1].as_deref().is_some_and(|def| self.default_matches(def, &args[keep - 1])) {
            keep -= 1;
        }
        // a constructor keeps one argument: `T x();` would declare a function
        if sig::is_ctor(sig) {
            keep = keep.max(1);
        }
        if keep == args.len() {
            return keep;
        }
        // another overload callable with `keep` arguments would change the call's meaning
        let min_arity = |x: &mwdec_core::DeclInfo| x.params.len() - x.defaults.iter().rev().take_while(|d| d.is_some()).count();
        if decls.iter().any(|x| !std::ptr::eq(x, d) && (x.variadic || (min_arity(x) <= keep && keep <= x.params.len()))) {
            return args.len();
        }
        if mwdec_lift::variants::alt(mwdec_lift::variants::EXPLICIT_DEFAULT_ARGS) {
            return args.len();
        }
        keep
    }

    /// Is argument `a` the default argument spelled `def` (space-joined tokens)?
    fn default_matches(&self, def: &str, a: &Expr) -> bool {
        let mut a = a;
        while let Expr::Cast { e, .. } = a {
            a = e;
        }
        let d: String = def.split_whitespace().collect();
        let int = |s: &str| -> Option<i64> {
            let (neg, s) = match s.strip_prefix('-') {
                Some(r) => (true, r),
                None => (false, s),
            };
            let s = s.trim_end_matches(|c| c == 'u' || c == 'U' || c == 'l' || c == 'L');
            let v = if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) { i64::from_str_radix(h, 16).ok()? } else { s.parse::<i64>().ok()? };
            Some(if neg { -v } else { v })
        };
        match d.as_str() {
            "true" => return a.as_int() == Some(1),
            "false" | "nullptr" | "NULL" => return a.as_int() == Some(0),
            _ => {}
        }
        if let Some(v) = int(&d) {
            return a.as_int() == Some(v);
        }
        if let Expr::Float { bits, double } = a {
            let s = d.trim_end_matches(|c| c == 'f' || c == 'F');
            if let Ok(x) = s.parse::<f64>() {
                return if *double { *bits == x.to_bits() } else { *bits == (x as f32).to_bits() as u64 };
            }
            return false;
        }
        if !d.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':') || d.starts_with(|c: char| c.is_ascii_digit()) {
            return false;
        }
        let last = d.rsplit("::").next().unwrap_or(&d);
        // an enumerator
        if let Some(v) = a.as_int() {
            let db = match self.db {
                Some(db) => db,
                None => return false,
            };
            let mut vals = db.enums.values().flat_map(|e| e.values.iter()).filter(|(n, _)| n == last || n.rsplit("::").next() == Some(last));
            return vals.next().is_some_and(|(_, x)| *x == v);
        }
        // a global object or constant
        match a {
            Expr::Global { symbol, .. } => symbol_name(symbol).rsplit("::").next() == Some(last),
            _ => false,
        }
    }

    fn expr_inner(&mut self, e: &Expr) -> String {
        match e {
            Expr::Var(v) if Some(*v) == self.sret_local => "(&__return_value)".into(),
            Expr::Var(v) if self.this_local && Some(*v) == self.ir.this_var => "self".into(),
            Expr::Var(v) => self.ir.vars[*v].name.clone(),
            Expr::Int { value, ty } => int_lit(*value, ty, &self.opts.null),
            Expr::Float { bits, double } => float::format_float(*bits, *double),
            Expr::Str { bytes } => float::c_string(bytes),
            Expr::Global { symbol, ty } => self.global(symbol, ty),
            Expr::FuncAddr { symbol } => {
                let n = symbol_name(symbol);
                self.declare_function(symbol, None, None, 1);
                format!("&{n}")
            }
            Expr::AddrOf(inner) => {
                // &base->field where the field is unknown -> byte pointer arithmetic
                match &**inner {
                    Expr::Load { base, offset, ty: Type::Unknown { size: 0 } } => {
                        if let Some(s) = self.field_addr(base, *offset, true) {
                            return s;
                        }
                        let b = self.expr(base, 14);
                        let bc = self.byte_cast(base);
                        format!("{bc}{b} + {}", hex_off(*offset))
                    }
                    Expr::Member { base, offset, ty: Type::Unknown { size: 0 } } => {
                        if let Some(s) = self.field_addr(base, *offset, false) {
                            return s;
                        }
                        let is_array = self.byte_array_base(base);
                        let b = self.expr(base, 14);
                        if is_array {
                            format!("{b} + {}", hex_off(*offset))
                        } else {
                            format!("(char*)&{b} + {}", hex_off(*offset))
                        }
                    }
                    Expr::Index { base, index, ty } if matches!(ty, Type::Int { size: 1, .. }) => {
                        let bt = ty_of(base, self.vars());
                        let i = self.expr(index, 13);
                        // byte arithmetic on a pointer to wider elements (an element address
                        // recovered inside the byte sum): keep it byte-wise
                        if is_ptr(&bt) && pointee(&bt).map_or(false, |p| !matches!(scalar_size(strip_cv(p)), Some(1))) {
                            let b = self.expr(base, 14);
                            return format!("({}){b} + {i}", type_str(&t_ptr(ty.clone())));
                        }
                        let b = self.expr(base, 12);
                        format!("{b} + {i}")
                    }
                    // untyped stack regions are byte arrays: their name is their address
                    Expr::Var(v) if matches!(self.ir.vars[*v].ty, Type::Unknown { size } if size != 4 && size != 2 && size != 1 && size != 8) => {
                        self.ir.vars[*v].name.clone()
                    }
                    Expr::Global { symbol, ty: ty @ Type::Unknown { size } } if *size != 4 && *size != 2 && *size != 1 && *size != 8 && self.byte_array_base(inner) => {
                        self.global(symbol, ty)
                    }
                    _ => {
                        let s = self.expr(inner, 14);
                        format!("&{s}")
                    }
                }
            }
            Expr::Load { base, offset, ty } => self.ptr_access(base, *offset, ty),
            Expr::Member { base, offset, ty } => self.member_access(base, *offset, ty),
            Expr::Index { base, index, ty } => {
                let bt = ty_of(base, self.vars());
                // (the index is read even in an assignment destination)
                let lv = std::mem::replace(&mut self.lvalue_ctx, false);
                let i = self.expr(index, 0);
                self.lvalue_ctx = lv;
                // an element of an inaccessible member array read through the class's element
                // accessor (`T& GetX(int i) { return mX[i]; }`)
                if let Some(s) = self.element_accessor_call(base, &i) {
                    return s;
                }
                // an indexable class object (its operator[])
                if named(&bt).is_some() {
                    let b = self.expr(base, 15);
                    return format!("{b}[{i}]");
                }
                if pointee(&bt).map_or(false, |p| same_scalar(p, ty)) {
                    let b = self.expr(base, 15);
                    format!("{b}[{i}]")
                } else {
                    let b = self.expr(base, 14);
                    format!("(({}*){b})[{i}]", type_str(&access_type(ty)))
                }
            }
            Expr::Unary { op, e, .. } => {
                let s = if is_ptr(&ty_of(e, self.vars())) && *op != UnOp::Not { format!("(unsigned int){}", self.expr(e, 14)) } else { self.expr(e, 14) };
                match op {
                    UnOp::Neg => format!("-{s}"),
                    UnOp::BitNot => format!("~{s}"),
                    UnOp::Not => format!("!{s}"),
                }
            }
            Expr::Binary { op, l, r, ty }
                if (matches!(op, BinOp::And | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::Shr | BinOp::Rem)
                    || (matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) && matches!(ty, Type::Int { .. }) && (l.is_lvalue() || r.is_lvalue())))
                    && (is_float(&ty_of(l, self.vars())) || is_float(&ty_of(r, self.vars()))) =>
            {
                // integer-only operators on a float operand: an lvalue's bits (`__HI(x)`), or a
                // conversion of a computed value
                let it = if is_float(ty) || matches!(ty, Type::Unknown { .. }) { Type::Int { size: 4, signed: true } } else { ty.clone() };
                let side = |me: &mut Self, x: &Expr, prec: u8| -> String {
                    if !is_float(&ty_of(x, me.vars())) {
                        return me.expr(x, prec);
                    }
                    if x.is_lvalue() {
                        format!("*({})&{}", ptr_to(&it), me.expr(x, 14))
                    } else {
                        format!("({}){}", type_str(&it), me.expr(x, 14))
                    }
                };
                let p = op.prec();
                let ls = side(self, l, p);
                let rs = side(self, r, p + 1);
                format!("{ls} {} {rs}", op.c_str())
            }
            Expr::Binary { op, l, r, ty } => {
                let p = op.prec();
                let mut ls = self.expr(l, p);
                let rs = self.expr(r, p + 1);
                // pointer compares against int literals
                if op.is_cmp() && is_ptr(&ty_of(l, self.vars())) {
                    if let Expr::Int { value, .. } = **r {
                        let rs = if value == 0 { self.opts.null.clone() } else { format!("(void*){value}") };
                        return format!("{ls} {} {rs}", op.c_str());
                    }
                }
                // compares mixing pointers with ints / unrelated pointers: compare as void*
                if op.is_cmp() {
                    let lt = ty_of(l, self.vars());
                    let rt = ty_of(r, self.vars());
                    let lp = is_ptr(&lt);
                    let rp = is_ptr(&rt);
                    let unrelated = lp && rp && {
                        let a = pointee(&lt).map(|t| sig::norm_name(&format!("{:?}", strip_cv(t))));
                        let b = pointee(&rt).map(|t| sig::norm_name(&format!("{:?}", strip_cv(t))));
                        a != b
                    };
                    if (lp != rp && r.as_int().is_none() && l.as_int().is_none()) || unrelated {
                        let ls = self.expr(l, 14);
                        let rs = self.expr(r, 14);
                        return format!("(void*){ls} {} (void*){rs}", op.c_str());
                    }
                }
                // integer arithmetic on pointer values (the IR's byte arithmetic / masks)
                if !op.is_bool() {
                    let lt = ty_of(l, self.vars());
                    let rt = ty_of(r, self.vars());
                    let lp = is_ptr(&lt);
                    let rp = is_ptr(&rt);
                    // char* +/- int is byte arithmetic already (the IR's meaning)
                    let byte_ptr = |t: &Type| pointee(t).map_or(false, |p| scalar_size(strip_cv(p)) == Some(1));
                    let byte_arith = matches!(op, BinOp::Add | BinOp::Sub) && ((lp && !rp && byte_ptr(&lt)) || (rp && !lp && *op == BinOp::Add && byte_ptr(&rt)));
                    if (lp || rp) && !byte_arith {
                        let ls2 = if lp { format!("(unsigned int){}", self.expr(l, 14)) } else { ls.clone() };
                        let rs2 = if rp { format!("(unsigned int){}", self.expr(r, 14)) } else { rs.clone() };
                        return format!("{ls2} {} {rs2}", op.c_str());
                    }
                }
                if *op == BinOp::Shr {
                    // make the shift kind explicit when the operand type disagrees
                    let lt = ty_of(l, self.vars());
                    // narrow operands promote to `int`: an unsigned char/short shifted logically
                    // needs an explicit unsigned conversion (srw, not sraw)
                    // (the C type of the operand after promotion, not the IR's operation type)
                    let promoted = match strip_cv(&lt) {
                        Type::Ptr(_) | Type::Ref(_) | Type::Float { .. } => is_signed(&lt),
                        _ if matches!(**l, Expr::Binary { .. } | Expr::Unary { .. } | Expr::Int { .. }) => Some(!mwdec_lift::translate::c_unsigned(l, self.vars())),
                        Type::Int { size, .. } if *size < 4 => Some(true),
                        Type::Bool | Type::Char | Type::WChar => Some(true),
                        t => is_signed(t),
                    };
                    let want = match strip_cv(ty) {
                        Type::Int { size, signed } if *size >= 4 => Some(*signed),
                        Type::Long { signed } => Some(*signed),
                        _ => None,
                    };
                    if want.is_some() && promoted.is_some() && want != promoted {
                        ls = format!("({}){}", type_str(ty), self.expr(l, 14));
                    }
                }
                format!("{ls} {} {rs}", op.c_str())
            }
            Expr::Cast { ty, e } => {
                let s = self.expr(e, 14);
                format!("({}){s}", type_str(ty))
            }
            Expr::Ternary { c, t, f, .. } => {
                let cs = self.expr(c, 4);
                let ts = self.expr(t, 4);
                let fs = self.expr(f, 3);
                format!("{cs} ? {ts} : {fs}")
            }
            Expr::Call { callee, args, ret } => self.call(callee, args, ret),
            Expr::Unknown { text, .. } => format!("0 /* {} */", text.replace("*/", "* /")),
            Expr::IncDec { e, delta, post } => {
                let op = if *delta > 0 { "++" } else { "--" };
                // (the operand is written: an lvalue spelling, not a getter read)
                self.lvalue_ctx = true;
                let x = if *post { self.expr(e, 15) } else { self.expr(e, 14) };
                self.lvalue_ctx = false;
                if *post {
                    format!("{x}{op}")
                } else {
                    format!("{op}{x}")
                }
            }
            Expr::BitField { base, shift, width, .. } => {
                let read = !std::mem::replace(&mut self.lvalue_ctx, false);
                if let Some(s) = self.bitfield_name(base, *shift, *width, read) {
                    return s;
                }
                let b = self.expr(base, 11);
                let m = (1u64 << width) - 1;
                if *shift == 0 {
                    format!("({b} & {m})")
                } else {
                    format!("({b} >> {shift} & {m})")
                }
            }
            Expr::Construct { class, ctor, args } => {
                // an inaccessible constructor: the class's public static factory that only
                // forwards to it (`static T FromX(float x) { return T(x); }`)
                if let Some(f) = named(strip_cv(class)).and_then(|c| self.ctor_factory(c, args.len())) {
                    let a = self.args(args, ctor.as_ref());
                    return format!("{}::{f}({a})", type_str(class));
                }
                // the implicit copy constructor (no declaration) from a pointer to the class: `T(*p)`
                if ctor.is_none() && args.len() == 1 {
                    let at = ty_of(&args[0], self.vars());
                    if let (Some(pc), Some(c)) = (pointee(&at).map(strip_cv).and_then(named), named(strip_cv(class))) {
                        if sig::norm_name(pc) == sig::norm_name(c) {
                            let r = Type::Ref(Box::new(Type::Const(Box::new(strip_cv(class).clone()))));
                            let a = self.coerce(&args[0], &r);
                            return format!("{}({a})", type_str(class));
                        }
                    }
                }
                // an undeclared constructor signature: the class's only constructor taking as
                // many arguments (a reference parameter then takes `*p`)
                let decl_ctor = if ctor.is_none() { named(strip_cv(class)).and_then(|c| self.only_ctor(c, args.len())) } else { None };
                let a = self.args(args, ctor.as_ref().or(decl_ctor.as_ref()));
                format!("{}({a})", type_str(class))
            }
            Expr::New { class, placement, ctor, args } => {
                let p = if placement.is_empty() {
                    String::new()
                } else {
                    let decl_ps: Option<Vec<Type>> = self.db.and_then(|db| db.decls.get("operator new")).and_then(|ds| {
                        ds.iter().find(|d| d.params.len() == placement.len() + 1).map(|d| d.params[1..].iter().map(|p| p.ty.clone()).collect())
                    });
                    let ps: Vec<String> = placement
                        .iter()
                        .enumerate()
                        .map(|(i, x)| match decl_ps.as_ref().and_then(|v| v.get(i)) {
                            Some(t) => self.coerce(x, t),
                            None => self.expr(x, 0),
                        })
                        .collect();
                    format!("({}) ", ps.join(", "))
                };
                let a = self.args(args, ctor.as_ref());
                format!("new {p}{}({a})", type_str(class))
            }
        }
    }

    /// The virtual function at `vtable_offset` of the static type of `this` (a pointer to a class
    /// whose vtable the TypeDb knows, with the vtable pointer at `vptr_offset`).
    fn slot_of_object(&self, this: &Expr, vtable_offset: u32, vptr_offset: u32) -> Option<mwdec_core::FuncSig> {
        let db = self.db?;
        // (the declared type of a member read: the IR may type a raw read as an integer)
        let declared = self.decl_type_rw(this, true);
        let t = if pointee(&declared).is_some() { declared } else { ty_of(this, self.vars()) };
        let p = pointee(&t)?;
        let r = mwdec_lift::types::resolve(Some(db), strip_cv(p)).into_owned();
        let cls = named(&r)?;
        let c = sig::find_class(db, cls)?;
        if c.vptr_offset != Some(vptr_offset) {
            return None;
        }
        let v = c.vtable.iter().find(|v| v.vtable_offset == vtable_offset && v.this_adjust == 0)?;
        if v.sig.qualified_name.is_empty() || placeholder_name(sig::split_scope(&v.sig.qualified_name).1) {
            return None;
        }
        Some(v.sig.clone())
    }

    /// The signature of the only constructor of `cls` the headers declare with `n` parameters.
    fn only_ctor(&self, cls: &str, n: usize) -> Option<mwdec_core::FuncSig> {
        let db = self.db?;
        let base = strip_template_args(cls);
        let last = sig::split_scope(&base).1.to_string();
        let ds: Vec<&mwdec_core::DeclInfo> = db.decls.get(&format!("{base}::{last}"))?.iter().filter(|d| d.params.len() == n && d.template_params.is_empty()).collect();
        let [d] = ds.as_slice() else { return None };
        Some(mwdec_core::FuncSig {
            qualified_name: format!("{base}::{last}"),
            mangled: None,
            ret: Type::Void,
            params: d.params.iter().map(|p| mwdec_core::Param { name: p.name.clone(), ty: p.ty.clone() }).collect(),
            this_class: Some(cls.to_string()),
            is_const: false,
            is_static: false,
            is_virtual: false,
            variadic: false,
            runs_code: false,
        })
    }

    /// `&base->field` when the DB knows a member starting at `off`.
    fn field_addr(&mut self, base: &Expr, off: i32, ptr: bool) -> Option<String> {
        if self.opts.raw_offsets {
            return None;
        }
        let db = self.db?;
        let bt = ty_of(base, self.vars());
        let t = if ptr { pointee(&bt)?.clone() } else { bt };
        let tr = mwdec_lift::types::resolve(Some(db), &t).into_owned();
        let cls = named(&tr)?;
        let (path, _) = field_path(db, cls, off, 0)?;
        // taking the address needs the member itself to be accessible
        if !self.reachable(&path, false) {
            return None;
        }
        let b = self.expr(base, 15);
        Some(format!("&{}{}{}", b, if ptr { "->" } else { "." }, render_path(&path)))
    }

    fn call(&mut self, callee: &Callee, args: &[Expr], ret: &Type) -> String {
        match callee {
            Callee::Direct { symbol, sig: s } if symbol == "__delete" && args.len() == 1 => {
                let t = s.params[0].ty.clone();
                let p = if matches!(t, Type::Unknown { .. }) { self.expr(&args[0], 14) } else { self.coerce(&args[0], &t) };
                format!("delete {p}")
            }
            Callee::Direct { symbol, sig: s } => {
                // free operator functions (header inlines folded back by mwdec-inline): infix
                if let Some(op) = s.qualified_name.strip_prefix("operator") {
                    let op = op.trim();
                    let binary = ["+", "-", "*", "/", "==", "!=", "<", ">", "<=", ">=", "&", "|", "^", "%"];
                    if args.len() == 2 && binary.contains(&op) {
                        let l = match s.params.first() {
                            Some(p) => self.coerce(&args[0], &p.ty),
                            None => self.expr(&args[0], 0),
                        };
                        let r = match s.params.get(1) {
                            Some(p) => self.coerce(&args[1], &p.ty),
                            None => self.expr(&args[1], 0),
                        };
                        return format!("({l} {op} {r})");
                    }
                    if args.len() == 1 && matches!(op, "-" | "!" | "~") {
                        let x = match s.params.first() {
                            Some(p) => self.coerce(&args[0], &p.ty),
                            None => self.expr(&args[0], 0),
                        };
                        return format!("{op}({x})");
                    }
                }
                if let Some(op) = self.runtime_op(symbol, args) {
                    return op;
                }
                if symbol == "__ptmf_scall" || symbol == "__ptmf_scall4" {
                    if let Some(c) = self.ptmf_call(args, ret) {
                        return c;
                    }
                }
                let name = if symbol.starts_with("__") && sig::demangle(symbol).is_none() { symbol.clone() } else { strip_unnamed_ns(&s.qualified_name) };
                // (an argument explicitly narrowed from a value of that same narrow type: the
                // extension is the conversion to a wider parameter)
                let arg_tys: Vec<Type> = args
                    .iter()
                    .map(|a| {
                        // (the address of a frame object declared as a byte array)
                        if matches!(a, Expr::AddrOf(x) if matches!(&**x, Expr::Var(v) if self.byte_array_local(*v).is_some())) {
                            return self.decl_type_rw(a, true);
                        }
                        if let Expr::Cast { ty, e } = a {
                            let r = |t: &Type| mwdec_lift::types::resolve(self.db, t).into_owned();
                            let (ct, et) = (r(ty), r(&ty_of(e, self.vars())));
                            if matches!(strip_cv(&ct), Type::Int { size: 1 | 2, .. }) && strip_cv(&ct) == strip_cv(&et) {
                                return Type::Int { size: 4, signed: false };
                            }
                        }
                        value_type(&ty_of(a, self.vars()))
                    })
                    .collect();
                self.declare_function(symbol, Some(s), Some((&arg_tys, ret)), 2);
                let self_decl = sig::demangle(symbol).is_none() && self.fn_decls.contains_key(symbol);
                // (later calls with other argument types: converted to the declared ones)
                let declared_tys = if self_decl { self.fn_param_tys.get(symbol).cloned().filter(|t| t.len() == args.len()) } else { None };
                let a = match declared_tys {
                    Some(tys) => args
                        .iter()
                        .zip(&tys)
                        .map(|(x, t)| {
                            // (the address of a member of a const object never passes as a pointer to non-const)
                            let const_addr = matches!(x, Expr::AddrOf(l) if self.const_lvalue(l)) && is_ptr(t) && !matches!(pointee(t), Some(Type::Const(_)));
                            // (a pointer the declarations type as one to another class than the lifter's guess)
                            let dt = self.decl_type_rw(x, true);
                            let other_class = matches!((pointee(&dt).map(strip_cv), pointee(t).map(strip_cv)), (Some(a @ Type::Named(_)), Some(b @ Type::Named(_))) if sig::norm_name(&format!("{a:?}")) != sig::norm_name(&format!("{b:?}")));
                            if !const_addr && !other_class && (value_type(&ty_of(x, self.vars())) == *t || dt == *t) {
                                self.expr(x, 0)
                            } else {
                                self.coerce(x, t)
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", "),
                    None if self_decl => args.iter().map(|x| self.expr(x, 0)).collect::<Vec<_>>().join(", "),
                    None => self.args(args, Some(s)),
                };
                format!("{name}({a})")
            }
            Callee::Method { sig: s, this, qualified, .. } => {
                // (anonymous-namespace scopes inside template arguments too)
                let cls = s.this_class.as_deref().map(strip_unnamed_ns);
                if sig::is_ctor(s) {
                    // placement-style construction isn't expressible; call through the object
                    let pre = self.object_prefix(this, cls.as_deref());
                    let a = self.args(args, Some(s));
                    return format!("{pre}{}({a})", self.method_name(s));
                }
                if sig::is_dtor(s) && args.len() == 1 && args[0].as_int() == Some(1) {
                    return self.delete_expr(this, cls.as_deref());
                }
                if sig::is_dtor(s) {
                    // `obj.~T()` needs an object whose static type is T
                    if let Some(c) = cls.as_deref() {
                        let static_cls = match this.as_ref() {
                            Expr::AddrOf(inner) => named(&ty_of(inner, self.vars())).map(|x| x.to_string()),
                            e => pointee(&ty_of(e, self.vars())).and_then(named).map(|x| x.to_string()),
                        };
                        if static_cls.as_deref().map(sig::norm_name) != Some(sig::norm_name(c)) {
                            let o = self.expr(this, 14);
                            return format!("(({}*){o})->{}()", types::split_closers(c), self.method_name(s));
                        }
                    }
                }
                // a private/protected method of another class: an inline destructor's body is the
                // destructor itself; a public inline wrapper with the same call is used instead
                let mut mname = self.method_name(s);
                match self.route_inaccessible(s, args.len()) {
                    Some(Routed::Dtor(d)) => {
                        let pre = self.object_prefix(this, cls.as_deref());
                        return format!("{pre}{d}()");
                    }
                    Some(Routed::Wrapper(w)) => mname = w,
                    None => {}
                }
                if let Some(op) = mname.strip_prefix("operator") {
                    let op = op.trim();
                    let binary = ["+", "-", "*", "/", "==", "!=", "<", ">", "<=", ">=", "+=", "-=", "*=", "/=", "=", "&", "|", "^", "%", "&=", "|="];
                    if args.len() == 1 && binary.contains(&op) {
                        let l = self.object_lvalue_c(this, cls.as_deref(), s.is_const);
                        let r = self.args(args, Some(s));
                        return format!("({l} {op} {r})");
                    }
                    if args.len() == 1 && op == "[]" {
                        let l = self.object_lvalue_c(this, cls.as_deref(), s.is_const);
                        let r = self.args(args, Some(s));
                        return format!("({l})[{r}]");
                    }
                    // `it->m`: the iterator itself before `->`
                    if args.is_empty() && op == "->" {
                        return self.object_lvalue_c(this, cls.as_deref(), s.is_const);
                    }
                    if args.is_empty() && matches!(op, "-" | "!" | "~") {
                        let l = self.object_lvalue_c(this, cls.as_deref(), s.is_const);
                        return format!("{op}({l})");
                    }
                }
                // the const overload of a const/non-const pair called on a non-const object: a
                // const view of the object selects it
                let pre = match cls.as_deref() {
                    Some(c) if s.is_const && self.has_nonconst_overload(s, args.len()) && !self.const_object(this) => {
                        let c = types::split_closers(c);
                        match this.as_ref() {
                            Expr::AddrOf(inner) => format!("((const {c}&){}).", self.expr(inner, 14)),
                            e => format!("((const {c}*){})->", self.expr(e, 14)),
                        }
                    }
                    _ => self.object_prefix_c(this, cls.as_deref(), s.is_const),
                };
                let obj_cls = {
                    let t = ty_of(this, self.vars());
                    match this.as_ref() {
                        Expr::AddrOf(inner) => named(&ty_of(inner, self.vars())).map(|s| s.to_string()),
                        _ => pointee(&t).and_then(named).map(|s| s.to_string()),
                    }
                };
                let a = self.args(args, Some(s));
                // qualification suppresses virtual dispatch (a direct `bl`); other base methods
                // are found by name lookup
                // a same-named method declared between the object's class and the callee's
                // class would be found first (often the function being written: recursion)
                let hidden = match (obj_cls.as_deref(), cls.as_deref()) {
                    (Some(o), Some(c)) => self.name_hidden_below(o, c, &mname),
                    _ => false,
                };
                let qualify = *qualified || s.is_virtual || hidden;
                if qualify {
                    format!("{pre}{}::{}({a})", cls.unwrap_or_default(), mname)
                } else {
                    format!("{pre}{}({a})", mname)
                }
            }
            Callee::Virtual { this, vtable_offset, vptr_offset, class, sig: s } => {
                // (a slot whose function only has a placeholder name has no member to call)
                if let Some(s) = s.as_ref().filter(|s| !self.method_name(s).is_empty() && !placeholder_name(&self.method_name(s))) {
                    if sig::is_dtor(s) && args.len() == 1 && args[0].as_int() == Some(1) {
                        let c = s.this_class.clone().or_else(|| class.clone());
                        return self.delete_expr(this, c.as_deref());
                    }
                    let pre = self.object_prefix(this, class.as_deref().map(strip_unnamed_ns).as_deref());
                    let n = if s.variadic { args.len() } else { args.len().min(s.params.len()) };
                    let a = self.args(&args[..n], Some(s));
                    return format!("{pre}{}({a})", self.method_name(s));
                }
                // the object's type known only now (e.g. a folded header inline returns it): its
                // class's vtable names the slot
                if let Some(vs) = self.slot_of_object(this, *vtable_offset, *vptr_offset) {
                    if !self.method_name(&vs).is_empty() && (vs.variadic || vs.params.len() <= args.len()) {
                        let pre = self.object_prefix(this, vs.this_class.as_deref());
                        let n = if vs.variadic { args.len() } else { vs.params.len() };
                        let a = self.args(&args[..n], Some(&vs));
                        // (the IR has the returned reference as the address it is)
                        if matches!(strip_cv(&vs.ret), Type::Ref(_)) {
                            return format!("&{pre}{}({a})", self.method_name(&vs));
                        }
                        return format!("{pre}{}({a})", self.method_name(&vs));
                    }
                }
                // method unknown: a stand-in polymorphic class whose slot N has this call's
                // signature, so MWCC emits the real virtual-call sequence (vptr into r12)
                let mut ptys = vec![];
                for a in args {
                    let t = value_type(&ty_of(a, self.vars()));
                    // (the address of a const object: a pointer to const)
                    let t = match (&t, a) {
                        (Type::Ptr(x), Expr::AddrOf(o)) if !matches!(**x, Type::Const(_)) && self.const_lvalue(o) => Type::Ptr(Box::new(Type::Const(x.clone()))),
                        _ => t,
                    };
                    ptys.push(type_str(&t));
                }
                let ret_t = match (ret, self.vt_ret_hint.take()) {
                    (Type::Void | Type::Unknown { .. }, Some(h)) => h,
                    _ => ret.clone(),
                };
                let rt = type_str(&value_type(&local_type(&ret_t)));
                if self.opts.c_mode {
                    // C has no virtual functions: a function pointer read through the object's
                    // first word (a table of functions)
                    let o = self.expr(this, 14);
                    let tbl = if *vptr_offset == 0 { format!("*(char**){o}") } else { format!("*(char**)((char*){o} + {})", hex_off(*vptr_offset as i32)) };
                    let mut ps = vec!["void*".to_string()];
                    ps.extend(ptys.iter().cloned());
                    let a: Vec<String> = std::iter::once(o.clone()).chain(args.iter().map(|x| self.expr(x, 0))).collect();
                    return format!("(*({rt} (**)({}))({tbl} + {}))({})", ps.join(", "), hex_off(*vtable_offset as i32), a.join(", "));
                }
                let slot = (vtable_offset.saturating_sub(8) / 4) as usize;
                let sname = format!("__mwdec_vt_{}", self.vt_count);
                self.vt_count += 1;
                let mut decl = format!("struct {sname} {{");
                for k in 0..slot {
                    let _ = write!(decl, " virtual void _{k}();");
                }
                let _ = write!(decl, " virtual {rt} _{slot}({}); }};", ptys.join(", "));
                self.externs.insert(decl);
                let o = self.expr(this, 14);
                let obj = if *vptr_offset == 0 { format!("(({sname}*){o})") } else { format!("(({sname}*)((char*){o} + {}))", hex_off(*vptr_offset as i32)) };
                let a = args.iter().map(|x| self.expr(x, 0)).collect::<Vec<_>>().join(", ");
                format!("{obj}->_{slot}({a})")
            }
            Callee::Indirect(f) => {
                let mut ptys = vec![];
                for a in args {
                    ptys.push(type_str(&value_type(&ty_of(a, self.vars()))));
                }
                let rt = type_str(&value_type(ret));
                let fs = self.expr(f, 14);
                let a = args.iter().map(|x| self.expr(x, 0)).collect::<Vec<_>>().join(", ");
                format!("(({rt} (*)({})){fs})({a})", ptys.join(", "))
            }
        }
    }
}

/// A lone constructor argument spelled `U(y)` (a functional cast or a call): parenthesized, so
/// that `T x(U(y));` isn't a function declaration.
fn vexing_parens(a: String) -> String {
    let t = a.trim();
    if t.ends_with(')') && t.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        format!("({a})")
    } else {
        a
    }
}

/// A splitter placeholder (`fn_80073C7C`, `lbl_...`): no source name.
fn placeholder_name(n: &str) -> bool {
    (n.starts_with("fn_") || n.starts_with("lbl_")) && sig::demangle(n).is_none()
}

fn hex_off(o: i32) -> String {
    if o < 0 {
        format!("-0x{:x}", -(o as i64))
    } else {
        format!("0x{o:x}")
    }
}

fn int_lit(value: i64, ty: &Type, null: &str) -> String {
    match strip_cv(ty) {
        Type::Bool if types::C_MODE.with(|c| c.get()) => (if value != 0 { "1" } else { "0" }).into(),
        Type::Bool => (if value != 0 { "true" } else { "false" }).into(),
        Type::Ptr(_) if value == 0 => null.into(),
        // 64-bit constants need the suffix (`1LL << n` is a 64-bit shift)
        Type::Int { size: 8, signed } => {
            let sfx = if *signed { "LL" } else { "ULL" };
            if value < -0x10000 || value > 0xffff {
                if value < 0 && *signed {
                    format!("-0x{:x}{sfx}", -(value as i128))
                } else {
                    format!("0x{:x}{sfx}", value as u64)
                }
            } else if value < 0 && !*signed {
                format!("0x{:x}{sfx}", value as u64)
            } else {
                format!("{value}{sfx}")
            }
        }
        Type::Int { signed: false, .. } => {
            let v = value as u64;
            if v >= 0x100 {
                format!("0x{v:x}")
            } else {
                format!("{v}")
            }
        }
        _ => {
            if value < -0x10000 || value > 0xffff {
                if value < 0 {
                    format!("-0x{:x}", -value)
                } else {
                    format!("0x{value:x}")
                }
            } else {
                format!("{value}")
            }
        }
    }
}

/// Inline body `return [cast<...>(] field [)] ;` (tokens space-separated as the TypeDb stores them).
fn getter_returns(body: &str, field: &str) -> bool {
    let toks: Vec<&str> = body.split_whitespace().collect();
    if toks.len() < 3 || toks[0] != "return" || *toks.last().unwrap() != ";" {
        return false;
    }
    let mut rest = vec![];
    let mut i = 1;
    let inner = &toks[1..toks.len() - 1];
    while i - 1 < inner.len() {
        let t = inner[i - 1];
        // (a getter that converts the member to another type, `reinterpret_cast<T*>(m_data)`,
        // doesn't read the member as declared)
        if matches!(t, "static_cast" | "reinterpret_cast") {
            return false;
        }
        if t == "const_cast" {
            // skip `< ... >`
            let mut depth = 0;
            i += 1;
            while i - 1 < inner.len() {
                match inner[i - 1] {
                    "<" => depth += 1,
                    ">" => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    ">>" => {
                        depth -= 2;
                        if depth <= 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        } else if t != "(" && t != ")" {
            rest.push(t);
        }
        i += 1;
    }
    rest == [field] || rest == ["this", "->", field]
}

/// (register class, number) encoded in lifter variable names (`var_r31`, `temp_f30_2`):
/// GPRs sort before FPRs; other names last.
fn reg_of_name(n: &str) -> (u8, u8) {
    let rest = n.strip_prefix("var_").or_else(|| n.strip_prefix("temp_"));
    if let Some(r) = rest {
        let (cls, num) = match r.as_bytes().first() {
            Some(b'r') => (0u8, &r[1..]),
            Some(b'f') => (1u8, &r[1..]),
            _ => return (2, 0),
        };
        let digits: String = num.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(k) = digits.parse::<u8>() {
            return (cls, k);
        }
    }
    (2, 0)
}

/// A type synthesized for the draft (see `synth_unknown_types`).
#[derive(Clone, Debug, Default)]
struct Synth {
    base: Option<String>,
    /// (signature key, declaration)
    methods: Vec<(String, String)>,
    size: u32,
}

impl Synth {
    fn add_method(&mut self, key: &str, decl: String) {
        if !self.methods.iter().any(|(k, _)| k == key) {
            self.methods.push((key.to_string(), decl));
        }
    }
}

/// Qualified identifier chains in a type or function spelling (`rstl::vector<A::B, C>` ->
/// `rstl::vector`, `A::B`, `C`), anonymous-namespace scopes dropped.
fn type_chains(s: &str) -> Vec<String> {
    let s = strip_unnamed_ns(s);
    let b = s.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            loop {
                while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                if i + 2 < b.len() && &s[i..i + 2] == "::" && ((b[i + 2] as char).is_ascii_alphabetic() || b[i + 2] == b'_') {
                    i += 2;
                    continue;
                }
                break;
            }
            // a member of a template instance (`A<T>::node`) is no free-standing name
            if st >= 2 && &s[st - 2..st] == "::" {
                continue;
            }
            // a function name's own last component (`Ns::Func(`) isn't a type
            if b.get(i) != Some(&b'(') {
                out.push(s[st..i].to_string());
            } else if let (Some(sc), _) = sig::split_scope(&s[st..i]) {
                out.push(sc.to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

fn is_c_keyword(s: &str) -> bool {
    matches!(
        s,
        "const" | "volatile" | "unsigned" | "signed" | "int" | "char" | "short" | "long" | "float" | "double" | "bool" | "void" | "wchar_t" | "struct" | "class" | "union" | "enum" | "operator" | "template" | "typename"
    )
}

/// Spelling of a call to a method the function may not name.
enum Routed {
    /// the class's inline destructor consists of this call: `obj.~T()`
    Dtor(String),
    /// a public inline method whose body is exactly this call
    Wrapper(String),
}

/// `obj->__vptr$ = &vtable` / `*(int*)obj = (int)&__vt__X`: a vtable pointer store.
fn is_vtable_store(dst: &Expr, src: &Expr) -> bool {
    let mut vt = false;
    src.walk(&mut |e| {
        if let Expr::Global { symbol, .. } = e {
            if symbol.starts_with("__vt__") {
                vt = true;
            }
        }
    });
    vt && matches!(dst, Expr::Load { .. } | Expr::Member { .. })
}

/// MWCC intrinsics and runtime helpers: never declared (the compiler knows or emits them).
fn is_compiler_builtin(sym: &str) -> bool {
    const B: &[&str] = &[
        "__cntlzw", "__fabs", "__fnabs", "__fabsf", "__frsqrte", "__fres", "__abs", "__labs", "__rlwimi", "__rlwinm", "__rlwnm", "__lhbrx", "__lwbrx", "__sthbrx", "__stwbrx",
        "__dcbf", "__dcbi", "__dcbst", "__dcbt", "__dcbtst", "__dcbz", "__sync", "__eieio", "__isync", "__mfspr", "__mtspr", "__mftb", "__setflm", "__fmadd", "__fmadds",
        "__fmsub", "__fmsubs", "__fnmadd", "__fnmadds", "__fnmsub", "__fnmsubs", "__fsel", "__fsels", "__mulhw", "__mulhwu", "__alloca", "__memcpy", "__va_arg",
        "__shl2i", "__shr2i", "__shr2u", "__div2i", "__div2u", "__mod2i", "__mod2u", "__cvt_fp2unsigned", "__cvt_ull_flt", "__cvt_sll_flt", "__cvt_ull_dbl", "__cvt_sll_dbl",
        "__cvt_dbl_ull", "__cvt_dbl_usll", "__cvt_flt_ull", "__construct_array", "__destroy_arr", "__construct_new_array", "__destroy_new_array", "__destroy_new_array2",
        "__ptmf_test", "__ptmf_cmpr", "__ptmf_scall", "__ptmf_scall4", "__register_global_object", "__save_gpr", "__restore_gpr", "__copy",
        // lifter pseudo-calls spelled as expressions (`va_start(ap, last)`, `delete p`)
        "va_start", "__delete",
    ];
    B.contains(&sym) || sym.starts_with("_savegpr") || sym.starts_with("_restgpr") || sym.starts_with("_savefpr") || sym.starts_with("_restfpr")
}

/// Does `t` name a type the context doesn't know (a template parameter such as `T`)?
fn unresolved_named(db: &TypeDb, t: &Type) -> bool {
    match t {
        Type::Named(n) => !n.contains('<') && sig::find_class(db, n).is_none() && !db.typedefs.contains_key(n) && !db.enums.contains_key(n),
        Type::Ptr(x) | Type::Ref(x) | Type::Const(x) | Type::Volatile(x) | Type::Array(x, _) => unresolved_named(db, x),
        _ => false,
    }
}

/// Spelling of `T*` (declarator syntax for arrays and function pointers: `unsigned char (*)[2]`).
fn ptr_to(t: &Type) -> String {
    let t = match t {
        Type::Array(..) | Type::Ref(_) => access_type(t),
        Type::Const(x) if matches!(**x, Type::Ref(_)) => access_type(t),
        t => t.clone(),
    };
    type_str(&Type::Ptr(Box::new(t)))
}

/// Declared type of a global the emitter declares itself, from an IR access type.
fn extern_type(ty: &Type) -> Type {
    match ty {
        Type::Unknown { size } if *size == 4 || *size == 0 => Type::Int { size: 4, signed: true },
        Type::Unknown { size } if *size == 1 => Type::Int { size: 1, signed: false },
        Type::Unknown { size } if *size == 2 => Type::Int { size: 2, signed: false },
        Type::Unknown { size } => Type::Array(Box::new(Type::Int { size: 1, signed: false }), *size),
        t => t.clone(),
    }
}

/// `name` used with type `t` while declared as `dt`.
fn reinterpret_global(name: String, t: &Type, dt: &Type) -> String {
    // byte-buffer views are address uses: callers check the declared type (byte_array_base)
    if t == dt || matches!(t, Type::Array(..)) {
        return name;
    }
    format!("(*({})&{name})", ptr_to(t))
}

/// Initializer text for an object of type `t` holding `bytes` (big-endian): scalars and arrays
/// of scalars (nested arrays flattened by braces).
fn static_init(t: &Type, bytes: &[u8]) -> Option<String> {
    fn val(t: &Type, b: &[u8]) -> Option<String> {
        let rd = |n: usize| -> Option<u64> { (b.len() >= n).then(|| b[..n].iter().fold(0u64, |a, x| (a << 8) | *x as u64)) };
        Some(match strip_cv(t) {
            Type::Int { size, signed } => {
                let n = *size as usize;
                let x = rd(n)?;
                let v = if *signed && n < 8 { ((x << (64 - 8 * n)) as i64) >> (64 - 8 * n) } else { x as i64 };
                if !*signed && n == 4 && v > i32::MAX as i64 {
                    format!("{v}u")
                } else if n == 8 {
                    format!("{v}LL")
                } else {
                    format!("{v}")
                }
            }
            Type::Long { signed } => {
                let x = rd(4)?;
                if *signed { format!("{}", x as u32 as i32) } else { format!("{x}u") }
            }
            Type::Char => format!("{}", rd(1)? as u8 as i8),
            Type::Bool => match rd(1)? {
                0 => "false".into(),
                1 => "true".into(),
                _ => return None,
            },
            Type::Float { size: 4 } => float::format_float(rd(4)?, false),
            Type::Float { size: 8 } => float::format_float(rd(8)?, true),
            Type::Unknown { size: s @ (1 | 2 | 4) } => format!("{}", rd(*s as usize)?),
            Type::Ptr(_) if rd(4)? == 0 => "0".into(),
            Type::Array(e, n) => {
                let es = mwdec_lift::types::size_of(None, e)? as usize;
                if es == 0 || b.len() < es * *n as usize {
                    return None;
                }
                let parts: Option<Vec<String>> = (0..*n as usize).map(|k| val(e, &b[k * es..])).collect();
                format!("{{{}}}", parts?.join(", "))
            }
            // byte buffers
            Type::Unknown { size } => {
                let n = *size as usize;
                if b.len() < n {
                    return None;
                }
                format!("{{{}}}", b[..n].iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
            }
            _ => return None,
        })
    }
    let s = val(t, bytes)?;
    // a scalar initializer of a byte-array declaration is a brace list too
    Some(s)
}

/// C++ name of a function-local static symbol (`init$90`, `buf$11_803DF590`), if it is one.
fn local_static_name(sym: &str) -> Option<String> {
    let (n, rest) = sig::strip_dtk_suffix(sym).split_once('$')?;
    if n.is_empty() || !rest.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    if !n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    Some(n.to_string())
}

/// `node*` -> `Class::node*` when `node` is a type nested in `class` and not a global name.
fn qualify_nested(t: &Type, class: &str, db: &TypeDb) -> Type {
    match t {
        Type::Ptr(x) => Type::Ptr(Box::new(qualify_nested(x, class, db))),
        Type::Ref(x) => Type::Ref(Box::new(qualify_nested(x, class, db))),
        Type::Const(x) => Type::Const(Box::new(qualify_nested(x, class, db))),
        Type::Named(n) if !n.contains("::") && !n.contains('<') => {
            let global = sig::find_class(db, n).is_some() || db.enums.contains_key(n) || db.typedefs.contains_key(n);
            let nested = format!("{class}::{n}");
            let nested_known = sig::find_class(db, &nested).is_some()
                || db.enums.contains_key(&nested)
                || db.typedefs.contains_key(&nested)
                || db.classes.keys().any(|k| k.starts_with(&strip_template_args(class)) && k.ends_with(&format!("::{n}")));
            if !global && nested_known {
                Type::Named(nested)
            } else {
                t.clone()
            }
        }
        _ => t.clone(),
    }
}

/// Declarable type for a local variable: references become pointers, top-level const goes.
fn local_type(t: &Type) -> Type {
    match strip_cv(t) {
        Type::Ref(x) => Type::Ptr(x.clone()),
        t => t.clone(),
    }
}

/// `rstl::single_ptr<CAnimData>` -> `rstl::single_ptr` (decl keys of templates have no args).
pub fn strip_template_args(s: &str) -> String {
    // an operator's name (`operator->`, `operator<=`, `operator<<`) is not a template argument list
    let op_at = s.match_indices("operator").map(|(i, _)| i).find(|&i| i == 0 || s[..i].ends_with("::"));
    if let Some(i) = op_at {
        if i > 0 {
            return format!("{}{}", strip_template_args(&s[..i]), &s[i..]);
        }
        return s.to_string();
    }
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

fn render_path(path: &[PathElem]) -> String {
    let mut s = String::new();
    for p in path {
        match p {
            PathElem::Field(n, _) => {
                if !s.is_empty() && !s.ends_with(']') {
                    s.push('.');
                } else if s.ends_with(']') {
                    s.push('.');
                }
                s.push_str(n);
            }
            PathElem::Index(i) => {
                let _ = write!(s, "[{i}]");
            }
            PathElem::Base(_) => {}
        }
    }
    s
}

/// Can a field of type `field` be used for an access of IR type `access`?
fn access_ok(db: Option<&TypeDb>, field: &Type, access: &Type) -> bool {
    if matches!(access, Type::Unknown { size: 0 }) {
        return true;
    }
    let fr = mwdec_lift::types::resolve(db, field).into_owned();
    // an array is never a scalar (a byte buffer read as a wider value)
    if matches!(strip_cv(&fr), Type::Array(..)) && !matches!(strip_cv(access), Type::Array(..)) {
        return false;
    }
    let fs = mwdec_lift::types::size_of(db, &fr);
    if fs != mwdec_lift::types::size_of(db, access) && fs.is_some() {
        // aggregates of the same size are copied as wholes
        return false;
    }
    is_float(&fr) == is_float(access)
}

fn same_scalar(a: &Type, b: &Type) -> bool {
    let a = strip_cv(a);
    let b = strip_cv(b);
    if a == b {
        return true;
    }
    match (a, b) {
        (Type::Int { size: x, .. }, Type::Int { size: y, .. }) => x == y && false,
        _ => false,
    }
}

/// Type to use in a raw pointer cast for an access.
fn access_type(t: &Type) -> Type {
    match t {
        Type::Unknown { size: 4 } => Type::Int { size: 4, signed: true },
        Type::Unknown { size: 2 } => Type::Int { size: 2, signed: false },
        Type::Unknown { size: 1 } => Type::Int { size: 1, signed: false },
        Type::Unknown { size: 8 } => Type::Int { size: 8, signed: true },
        Type::Unknown { .. } => Type::Int { size: 1, signed: false },
        // small arrays can't be assigned: their bytes as one integer
        Type::Array(..) if matches!(scalar_size(t), Some(1 | 2 | 4)) => Type::Int { size: scalar_size(t).unwrap() as u8, signed: false },
        // a reference is stored as the pointer it is (`T&*` isn't a type)
        Type::Ref(x) => Type::Ptr(x.clone()),
        Type::Const(x) if matches!(**x, Type::Ref(_)) => access_type(x),
        t => t.clone(),
    }
}

fn value_type(t: &Type) -> Type {
    match t {
        Type::Unknown { size: 0 } => Type::Void,
        t => access_type(t),
    }
}

/// Does the body assign (or take the address of) variable `v`?
fn param_written(body: &[Stmt], v: VarId) -> bool {
    fn assigns(b: &[Stmt], v: VarId) -> bool {
        b.iter().any(|s| match s {
            Stmt::Assign { dst: Expr::Var(x), .. } => *x == v,
            Stmt::If { then, els, .. } => assigns(then, v) || assigns(els, v),
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => assigns(body, v),
            Stmt::For { init, step, body, .. } => assigns(init, v) || assigns(step, v) || assigns(body, v),
            Stmt::Switch { cases, .. } => cases.iter().any(|c| assigns(&c.body, v)),
            _ => false,
        })
    }
    let mut hit = assigns(body, v);
    Stmt::walk_exprs(body, &mut |e| match e {
        Expr::IncDec { e, .. } | Expr::AddrOf(e) if matches!(&**e, Expr::Var(x) if *x == v) => hit = true,
        _ => {}
    });
    hit
}

/// Insert a parameter name into a type spelling (`void (*)(int)` -> `void (*name)(int)`).
fn spell_param(spelled: &str, name: &str) -> String {
    // abstract declarator `(*)`, `(**)`, `(&)`, `(* const)`: the name goes before its `)`
    let b = spelled.as_bytes();
    for i in 0..b.len() {
        if b[i] != b'(' {
            continue;
        }
        let Some(close) = spelled[i + 1..].find(')') else { break };
        let inner = &spelled[i + 1..i + 1 + close];
        let ptrish = !inner.trim().is_empty() && inner.replace("const", "").chars().all(|c| c == '*' || c == '&' || c == ' ') && inner.contains(['*', '&']);
        if ptrish {
            let at = i + 1 + close;
            let sp = if inner.trim_end().ends_with("const") { " " } else { "" };
            return format!("{}{sp}{}{}", &spelled[..at], name, &spelled[at..]);
        }
    }
    if spelled.contains("::*)") {
        return spelled.replacen("::*)", &format!("::*{name})"), 1);
    }
    format!("{spelled} {name}")
}

/// Names of functions/globals referenced by the IR (for callers that build contexts).
pub fn referenced_symbols(ir: &IrFunction) -> HashSet<String> {
    ir.globals.iter().map(|g| g.symbol.clone()).collect()
}

/// The stack variable an address expression points at (`&v`, `v` of an untyped buffer, casts).
fn stack_var_of(e: &Expr, ir: &IrFunction) -> Option<VarId> {
    match e {
        Expr::AddrOf(x) => match &**x {
            Expr::Var(v) if matches!(ir.vars[*v].kind, VarKind::Stack { .. }) => Some(*v),
            _ => None,
        },
        Expr::Cast { e, .. } => stack_var_of(e, ir),
        Expr::Var(v) if matches!(ir.vars[*v].kind, VarKind::Stack { .. }) => Some(*v),
        _ => None,
    }
}

/// The accessor (`data`) of class template instance `cls` returning the `T&` it keeps in raw
/// byte storage at `off`, when `T` is `obj`.
/// `off` is element `i` of the raw byte storage of class template instance `cls` whose element
/// type is the template argument `obj`, and the template has an inline `operator[]` returning
/// that parameter by reference: `i`.
fn element_index(db: &TypeDb, cls: &str, off: i32, obj: &str, ty: &Type) -> Option<i32> {
    let lt = cls.find('<')?;
    let base = &cls[..lt];
    let tps = db.templates.get(base)?;
    let args = sig::split_top(&cls[lt + 1..cls.len().checked_sub(1)?], ',');
    let k = args.iter().position(|a| sig::norm_name(a.trim()) == sig::norm_name(obj))?;
    let tp = tps.get(k)?;
    let esize = mwdec_lift::types::size_of(Some(db), strip_cv(ty))? as i32;
    if esize == 0 {
        return None;
    }
    let c = sig::find_class(db, cls)?;
    let f = c.fields.iter().find(|f| {
        let fsize = mwdec_lift::types::size_of(Some(db), &f.ty).unwrap_or(0) as i32;
        matches!(strip_cv(&f.ty), Type::Array(e, _) if scalar_size(strip_cv(e)) == Some(1)) && f.offset as i32 <= off && off + esize <= f.offset as i32 + fsize
    })?;
    let rel = off - f.offset as i32;
    if rel % esize != 0 {
        return None;
    }
    let indexer = db.decls.get(&format!("{base}::operator[]")).is_some_and(|ds| {
        ds.iter().any(|d| d.params.len() == 1 && d.is_inline_defined && d.access == mwdec_core::Access::Public && matches!(strip_cv(&d.ret), Type::Ref(t) if matches!(strip_cv(t), Type::Named(n) if n == tp)))
    });
    indexer.then_some(rel / esize)
}

fn storage_accessor(db: &TypeDb, cls: &str, off: i32, obj: &str) -> Option<String> {
    let lt = cls.find('<')?;
    let base = &cls[..lt];
    let tps = db.templates.get(base)?;
    let args = sig::split_top(&cls[lt + 1..cls.len().checked_sub(1)?], ',');
    let k = args.iter().position(|a| sig::norm_name(a.trim()) == sig::norm_name(obj))?;
    let tp = tps.get(k)?;
    let c = sig::find_class(db, cls)?;
    let f = c.fields.iter().find(|f| f.offset as i32 == off)?;
    if !matches!(strip_cv(&f.ty), Type::Array(e, _) if scalar_size(strip_cv(e)) == Some(1)) {
        return None;
    }
    let prefix = format!("{base}::");
    let mut found: Option<String> = None;
    for (key, ds) in db.decls.range(prefix.clone()..) {
        if !key.starts_with(&prefix) {
            break;
        }
        let name = &key[prefix.len()..];
        if name.contains("::") || name.starts_with("operator") {
            continue;
        }
        if ds.iter().any(|d| d.params.is_empty() && !d.is_const && d.is_inline_defined && matches!(strip_cv(&d.ret), Type::Ref(t) if matches!(strip_cv(t), Type::Named(n) if n == tp))) {
            found = Some(name.to_string());
            break;
        }
    }
    found
}
