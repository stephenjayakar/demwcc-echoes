//! Draft variants: decisions of the lifter and the emitter that the machine code doesn't settle
//! (a setter run or a whole-object copy, a folded inline or the plain expansion, a named local or
//! a temporary, ...). Instead of guessing, a decision site asks [`alt`] whether to take its
//! alternative. The answer is "no" (the default) unless the drafting driver redrafts with that
//! point flipped; every point asked during a draft is recorded, so the driver knows which
//! variants exist for this function and compiles each of them, keeping the best (the compiler
//! settles the ambiguity).
//!
//! Decision site (lift or emit, anything that runs inside [`draft`]):
//!
//! ```ignore
//! if variants::alt(variants::STRUCTCOPY_SETTERS) { /* the alternative */ } else { /* default */ }
//! ```
//!
//! Driver: `let (out, points) = variants::draft(&[], || lift_and_emit());` gives the default
//! draft and the points it asked; `variants::draft(&[p], ...)` drafts with point `p` flipped.
//! Outside [`draft`] every point takes its default and nothing is recorded.
//!
//! Points are plain `&'static str` names; list new ones in [`POINTS`] with a line of
//! documentation. Ask only where the alternative actually differs (a point asked but without
//! effect costs one redraft and is deduplicated before compiling).
use std::cell::RefCell;
use std::collections::BTreeSet;

/// Whole-object copy instead of a run of member setters `v.SetX(o.GetX()); v.SetY(...)...`.
pub const STRUCTCOPY_SETTERS: &str = "structcopy.setters";
/// Keep a temporary built from every member of one object as a construction (no copy).
pub const STRUCTCOPY_NO_TEMP: &str = "structcopy.no_temp";
/// Keep a returned object built in the struct-return storage (no `return x;` / `return T(..);`).
pub const STRUCTCOPY_NO_RETURN: &str = "structcopy.no_return";

/// A returned object filled from one object (through flag checks and early returns, as an
/// `optional_object` copy does) becomes `return x;`.
pub const STRUCTCOPY_RETURN_WHOLE: &str = "structcopy.return_whole";

/// Registered decision points: (name, what the alternative does).
pub const POINTS: &[(&str, &str)] = &[
    (STRUCTCOPY_SETTERS, "a run of member setters from one object's getters becomes a whole-object copy"),
    (STRUCTCOPY_NO_TEMP, "a temporary built from every member of one object stays a construction"),
    (STRUCTCOPY_NO_RETURN, "a returned object stays built in the struct-return storage"),
    (STRUCTCOPY_RETURN_WHOLE, "a returned object filled from one object behind flag checks becomes `return x;`"),
];

#[derive(Default)]
struct State {
    active: bool,
    flipped: Vec<&'static str>,
    asked: BTreeSet<&'static str>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Whether decision point `name` takes its alternative in the current draft (recording that the
/// point was asked).
pub fn alt(name: &'static str) -> bool {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if !s.active {
            return false;
        }
        s.asked.insert(name);
        s.flipped.contains(&name)
    })
}

/// Run `f` (a lift + emit) with the points in `flipped` taking their alternatives; returns its
/// result and the decision points it asked, in name order. Not reentrant.
pub fn draft<T>(flipped: &[&'static str], f: impl FnOnce() -> T) -> (T, Vec<&'static str>) {
    STATE.with(|s| *s.borrow_mut() = State { active: true, flipped: flipped.to_vec(), asked: BTreeSet::new() });
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            STATE.with(|s| *s.borrow_mut() = State::default());
        }
    }
    let _reset = Reset;
    let out = f();
    let asked = STATE.with(|s| s.borrow().asked.iter().copied().collect());
    (out, asked)
}

