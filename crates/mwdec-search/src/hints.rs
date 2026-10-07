//! Register-allocation hints for the target function, from the compiler-RE inverse colouring
//! model (`mwdec_oracle::hints`): for each callee-saved register, whether the value living there
//! is a param, a TEMP (unnamed / non-move uses, ranked by creation) or a NAMED local (ranked by
//! declaration order), or blocked by interference degree.
//!
//! The draft emitter names register-carried locals after their TARGET register (`var_r31`,
//! `temp_r29`, `var_f30_2`), so hints map straight onto draft locals. Only the target object is
//! read (anti-cheat: allowed input).
use mwdec_core::{Function, RelocKind};
use mwdec_oracle::regalloc::InferredClass;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HintClass {
    Param,
    Temp(u32),
    Named(u32),
    Blocked,
}

#[derive(Clone, Debug, Default)]
pub struct RegHints {
    /// "r31" / "f30" -> class.
    pub by_reg: HashMap<String, HintClass>,
}

impl RegHints {
    pub fn is_empty(&self) -> bool {
        self.by_reg.is_empty()
    }

    /// Hint for a draft local named after a register (`var_r31`, `temp_f30_2`).
    pub fn for_var(&self, name: &str) -> Option<HintClass> {
        self.by_reg.get(&reg_of_var(name)?).copied()
    }
}

/// `var_r31` / `temp_r29_3` / `var_f30` -> "r31" / "r29" / "f30".
pub fn reg_of_var(name: &str) -> Option<String> {
    let rest = name.strip_prefix("var_").or_else(|| name.strip_prefix("temp_"))?;
    let kind = rest.chars().next()?;
    if kind != 'r' && kind != 'f' {
        return None;
    }
    let digits: String = rest[1..].chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let tail = &rest[1 + digits.len()..];
    if !(tail.is_empty() || tail.starts_with('_') && tail[1..].chars().all(|c| c.is_ascii_digit())) {
        return None;
    }
    Some(format!("{kind}{digits}"))
}

/// Hints for a target function (empty if the model finds no consistent assignment).
pub fn target_hints(f: &Function) -> RegHints {
    let calls: BTreeMap<u32, String> =
        f.relocs.iter().filter(|r| r.kind == RelocKind::Rel24).map(|r| (r.offset & !3, r.target.clone())).collect();
    let r = std::panic::catch_unwind(|| {
        let ins = mwdec_oracle::webs::decode(&f.code, &calls);
        mwdec_oracle::hints::hints(&ins).1
    });
    let mut out = RegHints::default();
    let Ok(per_class) = r else { return out };
    for v in per_class.into_iter().flatten() {
        for h in v {
            let c = match h.class {
                InferredClass::Param(_) => HintClass::Param,
                InferredClass::Temp { rank } => HintClass::Temp(rank),
                InferredClass::Named { decl } => HintClass::Named(decl),
                InferredClass::Blocked => HintClass::Blocked,
            };
            out.by_reg.insert(mwdec_oracle::webs::reg_name(h.reg), c);
        }
    }
    out
}

