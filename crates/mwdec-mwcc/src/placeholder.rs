//! Placeholder resolution in the strict comparator (DESIGN.md "Strict comparator").
//!
//! dtk names functions nobody has named yet `fn_<address>`. A target calling such a placeholder
//! can never match by name: our compile references the function's real (mangled) name, e.g. a
//! template instance or an inline function emitted out of line. The two count as the same
//! function only when that is **proven by bytes**: our referenced function is compiled in the
//! same context (by a [`PlaceholderProver`]) and compared with the placeholder's code from the
//! target objects using the same strict comparator, recursively (nested placeholders are
//! resolved the same way) up to [`MAX_DEPTH`]. Without the proof the relocation differs.
//!
//! `MWDEC_NO_PLACEHOLDER=1` turns the resolution off.
use mwdec_core::ObjectFile;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Nesting limit of placeholder proofs (a proof's own relocations may need proofs).
pub const MAX_DEPTH: u32 = 2;

/// Bound on remembered proofs per prover (cleared when full).
const MEMO_CAP: usize = 4096;

/// Makes our definitions of functions referenced by name, compiled in the unit's context.
pub trait PlaceholderProver: Send + Sync {
    /// An object defining `symbol` (a template instance, an out-of-line inline, a generated
    /// special member), if some source can make the compiler emit it.
    fn definition(&self, symbol: &str) -> Option<Arc<ObjectFile>>;
    /// Remembered proofs: (placeholder, our symbol) -> identical.
    fn memo(&self) -> &Mutex<HashMap<(String, String), bool>>;
}

/// Is placeholder resolution on?
pub fn enabled() -> bool {
    std::env::var("MWDEC_NO_PLACEHOLDER").is_err()
}

/// `fn_8004A1C0` (optionally with dtk's `_XXXXXXXX` suffix): a function dtk named by address.
pub fn is_placeholder(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("fn_") else { return false };
    let hex8 = |s: &str| s.len() == 8 && s.bytes().all(|c| c.is_ascii_hexdigit());
    match rest.split_once('_') {
        None => hex8(rest),
        Some((a, b)) => hex8(a) && hex8(b),
    }
}

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Run a nested proof one level deeper; `None` past the limit.
pub(crate) fn nested<R>(f: impl FnOnce() -> R) -> Option<R> {
    let d = DEPTH.with(|d| d.get());
    if d >= MAX_DEPTH {
        return None;
    }
    DEPTH.with(|c| c.set(d + 1));
    let r = f();
    DEPTH.with(|c| c.set(d));
    Some(r)
}

pub(crate) fn remembered(p: &dyn PlaceholderProver, key: &(String, String)) -> Option<bool> {
    p.memo().lock().ok()?.get(key).copied()
}

pub(crate) fn remember(p: &dyn PlaceholderProver, key: (String, String), v: bool) {
    if let Ok(mut m) = p.memo().lock() {
        if m.len() >= MEMO_CAP {
            m.clear();
        }
        m.insert(key, v);
    }
}

