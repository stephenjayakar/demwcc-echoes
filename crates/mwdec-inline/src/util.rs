//! Small type helpers.
use mwdec_core::{Type, TypeDb};

pub fn strip(t: &Type) -> &Type {
    match t {
        Type::Const(t) | Type::Volatile(t) => strip(t),
        t => t,
    }
}

/// Class name of a (cv-qualified, typedef'd) class type.
pub fn class_name(t: &Type, db: &TypeDb) -> Option<String> {
    let r = mwdec_lift::types::resolve(Some(db), strip(t)).into_owned();
    let c = mwdec_lift::types::class_of(Some(db), &r)?;
    Some(c.name.clone())
}

/// Is `derived` the class `base` or derived from it (at offset 0)?
pub fn is_base_or_same(db: &TypeDb, base: &str, derived: &str) -> bool {
    if mwdec_lift::sig::norm_name(base) == mwdec_lift::sig::norm_name(derived) {
        return true;
    }
    let Some(c) = mwdec_lift::sig::find_class(db, derived) else { return false };
    c.bases.iter().any(|b| b.offset == 0 && is_base_or_same(db, base, &b.name))
}

/// Phase timers for `MWDI_DEBUG` profiling (process-wide, nanoseconds).
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub const NAMES: [&str; 12] = ["objlocals", "defs", "groups", "stmts", "cflow", "scalar", "update", "post", "g.stores", "g.copy", "g.match", "g.x"];
    static T: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
    pub fn time<R>(k: usize, f: impl FnOnce() -> R) -> R {
        let t = std::time::Instant::now();
        let r = f();
        T[k].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }
    pub fn report() -> String {
        NAMES.iter().enumerate().map(|(i, n)| format!("{n} {:.2}", T[i].load(Ordering::Relaxed) as f64 * 1e-9)).collect::<Vec<_>>().join(", ")
    }
}
