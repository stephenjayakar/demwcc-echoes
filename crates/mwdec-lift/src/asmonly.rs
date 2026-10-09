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
//!   quantized loads/stores for int<->float conversions, and prologue/epilogue saves), and
//!   integer compares into cr2-cr7 (compiled code uses cr0, cr1 for floats),
//!   addresses built with `@h`/`@l` (the compiler uses `@ha`), absolute branches, the stack
//!   pointer written outside the prologue/epilogue, r2/r13 read as values, and a return through LR loaded from a register other
//!   than r0 (calls through pointers use `mtlr r12; blrl`, returns restore LR from r0);
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
            // integer compares into cr2-cr7: compiled code compares into cr0 (cr1 for floats);
            // only hand-written code spreads tests over fields (`cmpwi cr6, r6, 0; bnelr cr6`)
            Cmp | Cmpi | Cmpl | Cmpli => i.ins.field_crfd() >= 2,
            op => is_paired_single(op),
        };
        // an address built with `lis @h` + `ori @l`: the compiler always uses `@ha` + `addi`/`@l`
        let bad = bad || i.reloc.as_ref().is_some_and(|r| r.kind == mwdec_core::RelocKind::Addr16Hi);
        // absolute branches; the stack pointer written outside the prologue/epilogue; LR loaded
        // from anything but r0 (the epilogue's restore) for a computed `blr`
        let bad = bad
            || (matches!(i.op(), B | Bc) && i.ins.field_aa())
            || (insn::defs_uses(i).0.contains(&gpr(1)) && !i.is_call() && !i.is_bctrl() && !i.is_blrl())
            // the small-data base registers read as values (the compiler reaches them only
            // through `@sda21` relocations)
            || (i.reloc.is_none() && !i.is_call() && !i.is_bctrl() && !i.is_blrl() && insn::defs_uses(i).1.iter().any(|r| *r == gpr(2) || *r == gpr(13)))
            || (i.is_blr() && insns[..k].iter().rev().take_while(|p| !p.is_call() && !p.is_blrl() && !p.is_blr()).find(|p| p.op() == Mtspr && p.ins.field_spr() == 8).is_some_and(|p| p.rs() != 0));
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
