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
