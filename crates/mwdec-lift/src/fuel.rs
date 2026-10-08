//! Iteration caps for the lifter's fixpoint, worklist and chain-walking loops. A loop that should
//! settle (a reduction that keeps rewriting, a walk along fall-through edges that comes back to
//! where it started) burns [`Fuel`] each round; once its cap is used up it stops and the site is
//! recorded, and the draft carries a warning instead of the lifter spinning forever.
//!
//! ```ignore
//! let mut fuel = Fuel::new("simplify.fold", CAP_FIXPOINT);
//! while changed && fuel.burn() { ... }
//! ```
use std::cell::RefCell;

/// Rounds of a whole-body fixpoint (each round rewrites at least one statement).
pub const CAP_FIXPOINT: usize = 1 << 14;
/// Steps of a walk along the CFG or a worklist over its blocks (scaled by the block count).
pub const CAP_WALK: usize = 1 << 16;
/// The draft warning for a loop stopped by its cap.
pub const WARN_ITERATION_CAP: &str = "iteration cap reached in";

thread_local! {
    static HIT: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

pub struct Fuel {
    site: &'static str,
    left: usize,
    hit: bool,
}

impl Fuel {
    pub fn new(site: &'static str, cap: usize) -> Self {
        Fuel { site, left: cap, hit: false }
    }

    /// One more round; false (recorded once) when the cap is used up.
    pub fn burn(&mut self) -> bool {
        if self.left == 0 {
            if !self.hit {
                self.hit = true;
                HIT.with(|h| h.borrow_mut().push(self.site));
            }
            return false;
        }
        self.left -= 1;
        true
    }
}

/// Position in the record of stopped loops (to collect the ones of one draft with [`since`]).
pub fn mark() -> usize {
    HIT.with(|h| h.borrow().len())
}

/// The warnings for the loops stopped since `mark` (removed from the record).
pub fn since(mark: usize) -> Vec<String> {
    HIT.with(|h| {
        let mut h = h.borrow_mut();
        let mut sites: Vec<&'static str> = if mark < h.len() { h.drain(mark..).collect() } else { vec![] };
        sites.dedup();
        sites.into_iter().map(|s| format!("{WARN_ITERATION_CAP} {s}")).collect()
    })
}
