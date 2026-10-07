//! Per-unit setup shared by the tools: context TU, TypeDb (with vtables), compiler context.
//! Mirrors what the bench does, so experiments see the same inputs.

use anyhow::Result;
use mwdec_core::{ObjectFile, TypeDb};
use mwdec_mwcc::{Mwcc, UnitContext};
use mwdec_project::{harness, Project};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

pub struct UnitSession {
    pub unit: String,
    pub cflags: Vec<String>,
    pub context: String,
    pub obj: ObjectFile,
    pub db: Option<TypeDb>,
    pub mwcc: Arc<Mwcc>,
    pub ctx: UnitContext,
    pub plain: UnitContext,
}

pub fn work_dir() -> PathBuf {
    mwdec_core::paths::work_dir("mwdec-inline")
}

/// Add vtables of every class the context knows from the module's objects (as the bench does).
pub fn add_vtables(p: &Project, unit: &str, obj: &ObjectFile, db: &mut TypeDb) {
    let module = Project::module_of(unit);
    let mut objs: Vec<Arc<ObjectFile>> = p.load_module_data("main");
    if module != "main" {
        objs.extend(p.load_module_data(module));
    }
    for o in objs.iter() {
        let relevant = o.data.keys().any(|n| mwdec_ctx::vtable::vtable_class(n).map_or(false, |c| db.classes.contains_key(&c)));
        if relevant {
            let vt = mwdec_ctx::vtables_from_object(o, db);
            let vt: BTreeMap<_, _> = vt.into_iter().filter(|(c, _)| db.classes.contains_key(c)).collect();
            mwdec_ctx::apply_vtables(db, &vt);
        }
    }
    let vt = mwdec_ctx::vtables_from_object(obj, db);
    mwdec_ctx::apply_vtables(db, &vt);
}

impl UnitSession {
    pub fn open(p: &Project, unit: &str, jobs: usize) -> Result<UnitSession> {
        let u = p.unit(unit).ok_or_else(|| anyhow::anyhow!("no unit {unit}"))?;
        let context = harness::context_tu(p, u)?;
        let obj = mwdec_obj::load_object(&p.path(&u.target_obj).to_string_lossy())?;
        let mut db = mwdec_ctx::build_typedb(&context, &u.cflags, &mwdec_ctx::default_work_dir()).ok();
        if let Some(db) = db.as_mut() {
            add_vtables(p, &u.name, &obj, db);
        }
        let root = mwdec_mwcc::default_root();
        let compiler = p.compiler_rel(&u.name);
        let mut m = Mwcc::new(&root, &work_dir().join(compiler.replace(['/', '.'], "_")), jobs);
        m.compiler = compiler;
        let plain = m.plain_context(&context, &u.cflags);
        let ctx = m.precompile(&context, &u.cflags).unwrap_or_else(|_| plain.clone());
        Ok(UnitSession { unit: u.name.clone(), cflags: u.cflags.clone(), context, obj, db, mwcc: Arc::new(m), ctx, plain })
    }

    /// Compile `code` in the unit context (PCH, plain fallback on a crash) and load the object.
    pub fn compile(&self, code: &str) -> Result<ObjectFile, String> {
        let mut r = self.mwcc.compile_in(&self.ctx, code);
        if let Err(mwdec_mwcc::MwccError::Compile { status, messages }) = &r {
            if status.map_or(false, |s| s < 0) || messages.contains("Unhandled exception") {
                r = self.mwcc.compile_in(&self.plain, code);
            }
        }
        match r {
            Err(e) => Err(e.messages().to_string()),
            Ok(c) => mwdec_obj::load_object_bytes(&c.obj_path.to_string_lossy(), &c.obj).map_err(|e| e.to_string()),
        }
    }
}
