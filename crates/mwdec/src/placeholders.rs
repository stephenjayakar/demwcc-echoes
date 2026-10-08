//! The unit-side half of placeholder resolution (`mwdec_mwcc::placeholder`): our definition of
//! a function referenced by name, made by compiling the sources that make the compiler emit it
//! (`mwdec_emit::instantiate`) in the unit's context. Definitions are cached per symbol.
use mwdec_core::{ObjectFile, TypeDb};
use mwdec_mwcc::{Mwcc, UnitContext};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct UnitProver<'a> {
    mwcc: &'a Mwcc,
    ctx: &'a UnitContext,
    db: Option<&'a TypeDb>,
    defs: Mutex<HashMap<String, Option<Arc<ObjectFile>>>>,
    memo: Mutex<HashMap<(String, String), bool>>,
}

impl<'a> UnitProver<'a> {
    pub fn new(mwcc: &'a Mwcc, ctx: &'a UnitContext, db: Option<&'a TypeDb>) -> Self {
        UnitProver { mwcc, ctx, db, defs: Default::default(), memo: Default::default() }
    }

    fn make(&self, symbol: &str) -> Option<Arc<ObjectFile>> {
        let sig = mwdec_lift::sig::sig_of(symbol, self.db);
        let db = self.db;
        let is_class = |s: &str| {
            db.is_some_and(|d| mwdec_lift::sig::find_class(d, s).is_some())
                || mwdec_lift::sig::split_scope(s).1.chars().next().is_some_and(|c| !c.is_ascii_lowercase())
        };
        let sources = if mwdec_lift::sig::demangle(symbol).is_none() {
            mwdec_emit::instantiate::triggers_c(&sig)
        } else {
            mwdec_emit::instantiate::triggers(symbol, Some(&sig.ret), &is_class)
        };
        for src in sources.iter().take(6) {
            let Ok(c) = self.mwcc.compile_in(self.ctx, src) else { continue };
            let Ok(o) = mwdec_obj::load_object_bytes("placeholder.o", &c.obj) else { continue };
            if o.functions.iter().any(|f| f.name == symbol) {
                return Some(Arc::new(o));
            }
        }
        None
    }
}

impl mwdec_mwcc::PlaceholderProver for UnitProver<'_> {
    fn definition(&self, symbol: &str) -> Option<Arc<ObjectFile>> {
        if let Some(v) = self.defs.lock().ok()?.get(symbol) {
            return v.clone();
        }
        let v = self.make(symbol);
        if let Ok(mut m) = self.defs.lock() {
            if m.len() >= 256 {
                m.clear();
            }
            m.insert(symbol.to_string(), v.clone());
        }
        v
    }

    fn memo(&self) -> &Mutex<HashMap<(String, String), bool>> {
        &self.memo
    }
}
