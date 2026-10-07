//! Prologue/epilogue recognition (stack frame, LR save, callee-saved GPR/FPR/paired-single saves).

use crate::cfg::{Cfg, Term};
use crate::insn::Insn;
use crate::ir::FrameInfo;
use ppc750cl::Opcode;
use std::collections::{HashMap, HashSet};

pub struct Frame {
    pub info: FrameInfo,
    /// Instruction indices that belong to the prologue/epilogue (skipped by the translator).
    pub skip: HashSet<usize>,
    /// Save-slot offsets (r1-relative) -> register (gpr n or 32+fpr n), excluded from locals.
    pub save_slots: HashMap<i32, u8>,
    /// Lowest r1 offset used by saves (locals live below this).
    pub saves_lo: i32,
}

fn is_sym(i: &Insn, prefix: &str) -> bool {
    i.reloc.as_ref().map_or(false, |r| r.target.starts_with(prefix))
}

pub fn analyze(insns: &[Insn], cfg: &Cfg) -> Frame {
    let mut info = FrameInfo::default();
    let mut skip = HashSet::new();
    let mut save_slots: HashMap<i32, u8> = HashMap::new();
    let entry = &cfg.blocks[0];
    let mut defined: HashSet<u8> = HashSet::new();
    for k in entry.start..entry.end {
        let i = &insns[k];
        match i.op() {
            Opcode::Stwu if i.rs() == 1 && i.ra() == 1 && info.size == 0 => {
                info.size = (-(i.disp())) as u32;
                skip.insert(k);
            }
            Opcode::Mfspr if i.ins.field_spr() == 8 && i.rd() == 0 => {
                info.saves_lr = true;
                skip.insert(k);
                defined.insert(0);
                // mark r0 as "the LR" so its store is recognized
                defined.remove(&0);
                // find its store
                for k2 in k + 1..entry.end {
                    let j = &insns[k2];
                    if j.op() == Opcode::Stw && j.rs() == 0 && j.ra() == 1 {
                        skip.insert(k2);
                        break;
                    }
                    if crate::insn::defs_uses(j).0.contains(&0) {
                        break;
                    }
                }
            }
            Opcode::Stw if i.ra() == 1 && i.rs() >= 14 && !defined.contains(&i.rs()) => {
                save_slots.insert(i.disp(), i.rs());
                info.saved_gprs.push(i.rs());
                skip.insert(k);
            }
            Opcode::Stmw if i.ra() == 1 => {
                for (n, r) in (i.rs()..32).enumerate() {
                    save_slots.insert(i.disp() + 4 * n as i32, r);
                    info.saved_gprs.push(r);
                }
                info.uses_stmw = true;
                skip.insert(k);
            }
            Opcode::Stfd if i.ra() == 1 && i.ins.field_frs() >= 14 && !defined.contains(&(32 + i.ins.field_frs())) => {
                save_slots.insert(i.disp(), 32 + i.ins.field_frs());
                info.saved_fprs.push(i.ins.field_frs());
                skip.insert(k);
            }
            Opcode::PsqSt
                if i.ra() == 1 && i.ins.field_frs() >= 14 && !defined.contains(&(32 + i.ins.field_frs())) =>
            {
                save_slots.insert(i.ins.field_ps_offset() as i32, 32 + i.ins.field_frs());
                info.saved_ps.push(i.ins.field_frs());
                skip.insert(k);
            }
            Opcode::Addi if i.rd() == 11 && i.ra() == 1 && k + 1 < entry.end && is_sym(&insns[k + 1], "_save") => {
                skip.insert(k);
                skip.insert(k + 1);
                info.uses_savegpr = true;
                let name = &insns[k + 1].reloc.as_ref().unwrap().target;
                if let Some(n) = name.rsplit('_').next().and_then(|s| s.parse::<u8>().ok()) {
                    if name.contains("gpr") {
                        info.saved_gprs.extend(n..32);
                    } else {
                        info.saved_fprs.extend(n..32);
                    }
                }
            }
            _ => {}
        }
        if !skip.contains(&k) {
            for d in crate::insn::defs_uses(i).0 {
                defined.insert(d);
            }
        }
    }
    let size = info.size as i32;
    // epilogues in returning blocks
    for b in &cfg.blocks {
        if !matches!(b.term, Term::Return | Term::TailCall) {
            continue;
        }
        let mut k = b.end;
        while k > b.start {
            k -= 1;
            let i = &insns[k];
            let is_epi = match i.op() {
                Opcode::Addi if i.rd() == 1 && i.ra() == 1 && i.simm() as i32 == size => true,
                Opcode::Lwz if i.rd() == 1 && i.ra() == 1 && i.disp() == 0 => true,
                Opcode::Mtspr if i.ins.field_spr() == 8 && i.rs() == 0 => true,
                Opcode::Lwz if i.rd() == 0 && i.ra() == 1 && i.disp() == size + 4 && info.saves_lr => true,
                Opcode::Lwz if i.ra() == 1 && save_slots.get(&i.disp()) == Some(&i.rd()) => true,
                Opcode::Lmw if i.ra() == 1 => true,
                Opcode::Lfd if i.ra() == 1 && save_slots.get(&i.disp()) == Some(&(32 + i.ins.field_frd())) => true,
                Opcode::PsqL
                    if i.ra() == 1 && save_slots.get(&(i.ins.field_ps_offset() as i32)) == Some(&(32 + i.ins.field_frd())) =>
                {
                    true
                }
                Opcode::B if i.ins.field_lk() && is_sym(i, "_rest") => {
                    if k > b.start && insns[k - 1].op() == Opcode::Addi && insns[k - 1].rd() == 11 {
                        skip.insert(k - 1);
                    }
                    true
                }
                _ => false,
            };
            if is_epi {
                skip.insert(k);
            }
        }
    }
    let saves_lo = save_slots.keys().copied().min().unwrap_or(size);
    Frame { info, skip, save_slots, saves_lo }
}
