//! mwdec-search: compiler-in-the-loop source permuter (see DESIGN.md).
//!
//! - [`cst`]: tree-sitter-cpp parse into an owned arena + text edits + normalization
//! - [`func`]: target function lookup by mangled symbol, local/param types, effect model
//! - [`ops`]: semantics-preserving (best effort) mutation operators
//! - [`score`]: strict comparator + permuter-style penalty and diff profile
//! - [`search`]: parallel beam / hill-climb driver with diff-guided adaptive operator weights
pub mod cst;
pub mod func;
pub mod hints;
pub mod locate;
pub mod ops;
pub mod rng;
pub mod score;
pub mod search;
pub mod structural;
pub mod trace;

pub use score::{Eval, Fitness, Scorer};
pub use search::{search, SearchConfig, SearchResult};
