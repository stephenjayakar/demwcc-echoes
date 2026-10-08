//! Functions whose machine code no C/C++ source can produce under MWCC: they were written in
//! assembly (`asm void f() { nofralloc ... }`, or an `asm { }` block inside a C function).
//! Ordinary source can't match them, so the drafter reports them instead of drafting.
//!
//! Evidence, from the code alone:
//! - an instruction MWCC never selects for C and offers no intrinsic for: special-purpose
//!   register moves other than LR/CTR (`mfspr HID0`, `mtmsr`, `mftb`), `sc`, `rfi`, cache
//!   management without an intrinsic (`icbi`, `dcbi`), TLB/segment ops, FPSCR moves outside
//!   the `__setflm` pair, paired-single arithmetic, and paired or unquantized paired-single
//!   loads/stores (these compilers have no paired-single intrinsics; they only emit single
//!   quantized loads/stores for int<->float conversions, and prologue/epilogue saves);
//! - a register read before any definition that no calling convention passes in (r0, r11, r12,
//!   CTR, XER.CA, f0, f9-f13, CR fields other than cr1, which varargs prologues test), or a
//!   callee-saved register (r14-r31, f14-f31) read or written without the prologue saving it.
//!   Compiled code saves every callee-saved register it touches.

use crate::cfg::{self, Cfg};
use crate::frame;
use crate::insn::{self, crf, fpr, gpr, Reg, CA, CTR, NREGS};
use mwdec_core::{Function, ObjectFile};
use ppc750cl::Opcode;

/// Why `f` can only come from assembly, if it can.
pub fn requires_asm(obj: &ObjectFile, f: &Function) -> Option<String> {
    if f.code.is_empty() {
        return None;
    }
    let insns = cfg::decode(f);
    let cfg = Cfg::build(obj, f, &insns);
    let fr = frame::analyze(&insns, &cfg);
    // 1. instructions without a C spelling
    for (k, i) in insns.iter().enumerate() {
        if fr.skip.contains(&k) {
            continue;
        }
        use Opcode::*;
        let bad = match i.op() {
            Mfspr | Mtspr => !matches!(i.ins.field_spr(), 8 | 9),
            Mfmsr | Mtmsr | Mftb | Sc | Rfi | Icbi | Dcbi | Tlbie | Tlbsync | Mfsr | Mtsr | Mfsrin | Mtsrin | Eciwx | Ecowx | Mcrxr | Mtfsb0 | Mtfsb1 | Mtfsfi | Mtcrf => true,
            // `__setflm(x)`: `mffs fd` directly followed by `mtfsf 255, fs`
            Mffs => !insns.get(k + 1).is_some_and(|n| n.op() == Mtfsf && n.ins.field_mtfsf_fm() == 0xff),
            Mtfsf => !(k > 0 && insns[k - 1].op() == Mffs && i.ins.field_mtfsf_fm() == 0xff),
            // the compiler's own paired-single use is the scalar int<->float conversion
            // (`psq_l fD, d(rA), 1, qr2..7`); paired (w=0) or identity-quantized ones are asm
            PsqL | PsqLu | PsqLx | PsqLux | PsqSt | PsqStu | PsqStx | PsqStux => {
                let (w, q) = if matches!(i.op(), PsqL | PsqLu | PsqSt | PsqStu) { (i.ins.field_ps_w(), i.ins.field_ps_i()) } else { (i.ins.field_ps_wx(), i.ins.field_ps_ix()) };
                w == 0 || q == 0
            }
            op => is_paired_single(op),
        };
        if bad {
            return Some(format!("{} at {:#x}", i.text(), i.off));
        }
    }
    // 2. reads of registers nothing passed in, and unsaved callee-saved registers
    let saved = |r: Reg| -> bool {
        if (gpr(14)..=gpr(31)).contains(&r) {
            fr.info.saved_gprs.contains(&((r - gpr(0)) as u8))
        } else if (fpr(14)..=fpr(31)).contains(&r) {
            fr.info.saved_fprs.contains(&((r - fpr(0)) as u8))
        } else {
            true
        }
    };
    let callee_saved = |r: Reg| (gpr(14)..=gpr(31)).contains(&r) || (fpr(14)..=fpr(31)).contains(&r);
    let never_passed = |r: Reg| {
        r == gpr(0) || r == gpr(11) || r == gpr(12) || r == CTR || r == CA || r == fpr(0) || (fpr(9)..=fpr(13)).contains(&r) || ((crf(0)..=crf(7)).contains(&r) && r != crf(1))
    };
    for (k, i) in insns.iter().enumerate() {
        if fr.skip.contains(&k) {
            continue;
        }
        let (d, u) = insn::defs_uses(i);
        if i.is_call() || i.is_bctrl() || i.is_blrl() {
            // calls report every clobbered register as defined; only explicit uses matter
            if let Some(r) = d.iter().chain(u.iter()).copied().find(|r| callee_saved(*r) && !saved(*r)) {
                return Some(format!("unsaved {} at {:#x}", insn::reg_name(r), i.off));
            }
            continue;
        }
        if let Some(r) = d.iter().chain(u.iter()).copied().find(|r| callee_saved(*r) && !saved(*r)) {
            return Some(format!("unsaved {} at {:#x}", insn::reg_name(r), i.off));
        }
    }
    // definitely-defined registers at block entry (forward must-analysis)
    let nb = cfg.blocks.len();
    let full = vec![true; NREGS];
    let mut din: Vec<Vec<bool>> = vec![full.clone(); nb];
    let mut entry = vec![false; NREGS];
    for r in 0..NREGS as Reg {
        entry[r as usize] = !never_passed(r);
    }
    din[0] = entry.clone();
    let transfer = |b: usize, mut s: Vec<bool>| -> Vec<bool> {
        for k in cfg.blocks[b].start..cfg.blocks[b].end {
            let (d, _) = insn::defs_uses(&insns[k]);
            for r in d {
                s[r as usize] = true;
            }
        }
        s
    };
    let mut changed = true;
    let mut rounds = 0;
    while changed && rounds < 64 {
        changed = false;
        rounds += 1;
        for &b in &cfg.rpo {
            let mut s = if b == 0 { entry.clone() } else { full.clone() };
            for &p in &cfg.blocks[b].preds {
                if cfg.idom[p] == usize::MAX {
                    continue;
                }
                let o = transfer(p, din[p].clone());
                for r in 0..NREGS {
                    s[r] &= o[r];
                }
            }
            if b == 0 {
                for r in 0..NREGS {
                    s[r] &= entry[r];
                }
            }
            if s != din[b] {
                din[b] = s;
                changed = true;
            }
        }
    }
    for &b in &cfg.rpo {
        let mut s = din[b].clone();
        for k in cfg.blocks[b].start..cfg.blocks[b].end {
            let i = &insns[k];
            let (d, u) = insn::defs_uses(i);
            // (`mfcr` reads every field; the compiler extracts the bits it set)
            if !fr.skip.contains(&k) && !(i.is_call() || i.is_bctrl() || i.is_blrl()) && i.op() != Opcode::Mfcr {
                if let Some(r) = u.iter().copied().find(|r| !s[*r as usize] && never_passed(*r)) {
                    return Some(format!("{} read before any definition at {:#x}", insn::reg_name(r), i.off));
                }
            }
            for r in d {
                s[r as usize] = true;
            }
        }
    }
    None
}

fn is_paired_single(op: Opcode) -> bool {
    format!("{op:?}").starts_with("Ps")
}
