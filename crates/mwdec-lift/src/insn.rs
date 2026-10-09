//! Instruction decoding helpers: register model and per-instruction defs/uses.

use mwdec_core::{Reloc, RelocKind};
use ppc750cl::{Ins, Opcode};

/// Register index: GPRs 0..32, FPRs 32..64, CR fields 64..72, CTR 72, LR 73, XER.CA 74.
pub type Reg = u8;
pub const NREGS: usize = 75;
pub const fn gpr(n: u8) -> Reg {
    n
}
pub const fn fpr(n: u8) -> Reg {
    32 + n
}
pub const fn crf(n: u8) -> Reg {
    64 + n
}
pub const CTR: Reg = 72;
pub const LR: Reg = 73;
pub const CA: Reg = 74;

pub fn reg_name(r: Reg) -> String {
    match r {
        0..=31 => format!("r{r}"),
        32..=63 => format!("f{}", r - 32),
        64..=71 => format!("cr{}", r - 64),
        CTR => "ctr".into(),
        LR => "lr".into(),
        _ => "ca".into(),
    }
}

#[derive(Clone, Debug)]
pub struct Insn {
    /// Byte offset in the function.
    pub off: u32,
    pub ins: Ins,
    pub reloc: Option<Reloc>,
}

impl Insn {
    pub fn op(&self) -> Opcode {
        self.ins.op
    }
    pub fn rd(&self) -> u8 {
        self.ins.field_rd()
    }
    pub fn rs(&self) -> u8 {
        self.ins.field_rs()
    }
    pub fn ra(&self) -> u8 {
        self.ins.field_ra()
    }
    pub fn rb(&self) -> u8 {
        self.ins.field_rb()
    }
    pub fn simm(&self) -> i16 {
        self.ins.field_simm()
    }
    pub fn uimm(&self) -> u16 {
        self.ins.field_uimm()
    }
    pub fn rc(&self) -> bool {
        self.ins.field_rc()
    }
    pub fn text(&self) -> String {
        format!("{}", self.ins.simplified())
    }
    pub fn reloc_kind(&self) -> Option<RelocKind> {
        self.reloc.as_ref().map(|r| r.kind)
    }
    /// `bl` (call) with a REL24 relocation.
    pub fn is_call(&self) -> bool {
        self.op() == Opcode::B && self.ins.field_lk()
    }
    /// Indirect call: `bctrl`, or `blrl` (the older compiler calls function pointers via LR).
    pub fn is_bctrl(&self) -> bool {
        (self.op() == Opcode::Bcctr && self.ins.field_lk()) || self.is_blrl()
    }
    pub fn is_blrl(&self) -> bool {
        self.op() == Opcode::Bclr && self.ins.field_lk() && self.ins.field_bo() == 20
    }
    pub fn is_bctr(&self) -> bool {
        self.op() == Opcode::Bcctr && !self.ins.field_lk() && self.ins.field_bo() == 20
    }
    pub fn is_blr(&self) -> bool {
        self.op() == Opcode::Bclr && !self.ins.field_lk() && self.ins.field_bo() == 20
    }
    /// Conditional return `beqlr` etc.
    pub fn is_cond_blr(&self) -> bool {
        self.op() == Opcode::Bclr && !self.ins.field_lk() && self.ins.field_bo() != 20
    }
    /// Unconditional `b` to a local target (no link); `b sym` with a reloc is a tail call.
    pub fn is_jump(&self) -> bool {
        self.op() == Opcode::B && !self.ins.field_lk()
    }
    pub fn is_cond_branch(&self) -> bool {
        self.op() == Opcode::Bc && !self.ins.field_lk() && self.ins.field_bo() != 20
    }
    /// Branch target (function offset) of b/bc.
    pub fn target(&self) -> Option<u32> {
        self.ins.branch_dest(self.off)
    }
    /// Memory displacement of D-form loads/stores.
    pub fn disp(&self) -> i32 {
        self.ins.field_offset() as i32
    }
}

/// Registers clobbered by a call under the EABI (volatile).
pub fn call_clobbers() -> Vec<Reg> {
    let mut v = vec![gpr(0)];
    v.extend((3..=12).map(gpr));
    v.extend((0..=13).map(fpr));
    v.extend([crf(0), crf(1), crf(5), crf(6), crf(7), CTR, LR, CA]);
    v
}

fn push_g(v: &mut Vec<Reg>, r: u8) {
    v.push(gpr(r));
}

/// (defs, uses) of an instruction. Calls report their argument registers as uses conservatively
/// (r3..r10, f1..f8); the translator narrows using signatures.
pub fn defs_uses(i: &Insn) -> (Vec<Reg>, Vec<Reg>) {
    use Opcode::*;
    let mut d = Vec::new();
    let mut u = Vec::new();
    let ins = i.ins;
    let rc = |d: &mut Vec<Reg>| {
        if ins.field_rc() {
            d.push(crf(0))
        }
    };
    match ins.op {
        // D-form integer loads
        Lwz | Lbz | Lhz | Lha => {
            push_g(&mut d, i.rd());
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
        }
        Lwzu | Lbzu | Lhzu | Lhau => {
            push_g(&mut d, i.rd());
            push_g(&mut d, i.ra());
            push_g(&mut u, i.ra());
        }
        Lwzx | Lbzx | Lhzx | Lhax | Lwbrx | Lhbrx => {
            push_g(&mut d, i.rd());
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
            push_g(&mut u, i.rb());
        }
        Lwzux | Lbzux | Lhzux | Lhaux => {
            push_g(&mut d, i.rd());
            push_g(&mut d, i.ra());
            push_g(&mut u, i.ra());
            push_g(&mut u, i.rb());
        }
        Lfs | Lfd => {
            d.push(fpr(ins.field_frd()));
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
        }
        Lfsu | Lfdu => {
            d.push(fpr(ins.field_frd()));
            push_g(&mut d, i.ra());
            push_g(&mut u, i.ra());
        }
        Lfsx | Lfdx => {
            d.push(fpr(ins.field_frd()));
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
            push_g(&mut u, i.rb());
        }
        Lfsux | Lfdux => {
            d.push(fpr(ins.field_frd()));
            push_g(&mut d, i.ra());
            push_g(&mut u, i.ra());
            push_g(&mut u, i.rb());
        }
        PsqL | PsqLu => {
            d.push(fpr(ins.field_frd()));
            if i.ra() != 0 || ins.op == PsqLu {
                push_g(&mut u, i.ra());
            }
            if ins.op == PsqLu {
                push_g(&mut d, i.ra());
            }
        }
        PsqLx | PsqLux => {
            d.push(fpr(ins.field_frd()));
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
            push_g(&mut u, i.rb());
            if ins.op == PsqLux {
                push_g(&mut d, i.ra());
            }
        }
        Lmw => {
            for r in i.rd()..32 {
                push_g(&mut d, r);
            }
            push_g(&mut u, i.ra());
        }
        // stores
        Stw | Stb | Sth => {
            push_g(&mut u, i.rs());
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
        }
        Stwu | Stbu | Sthu => {
            push_g(&mut u, i.rs());
            push_g(&mut u, i.ra());
            push_g(&mut d, i.ra());
        }
        Stwx | Stbx | Sthx | Stwbrx | Sthbrx => {
            push_g(&mut u, i.rs());
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
            push_g(&mut u, i.rb());
        }
        Stwux | Stbux | Sthux => {
            push_g(&mut u, i.rs());
            push_g(&mut u, i.ra());
            push_g(&mut u, i.rb());
            push_g(&mut d, i.ra());
        }
        Stfs | Stfd => {
            u.push(fpr(ins.field_frs()));
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
        }
        Stfsu | Stfdu => {
            u.push(fpr(ins.field_frs()));
            push_g(&mut u, i.ra());
            push_g(&mut d, i.ra());
        }
        Stfsx | Stfdx | Stfiwx => {
            u.push(fpr(ins.field_frs()));
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
            push_g(&mut u, i.rb());
        }
        Stfsux | Stfdux => {
            u.push(fpr(ins.field_frs()));
            push_g(&mut u, i.ra());
            push_g(&mut u, i.rb());
            push_g(&mut d, i.ra());
        }
        PsqSt | PsqStu => {
            u.push(fpr(ins.field_frs()));
            if i.ra() != 0 || ins.op == PsqStu {
                push_g(&mut u, i.ra());
            }
            if ins.op == PsqStu {
                push_g(&mut d, i.ra());
            }
        }
        PsqStx | PsqStux => {
            u.push(fpr(ins.field_frs()));
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
            push_g(&mut u, i.rb());
            if ins.op == PsqStux {
                push_g(&mut d, i.ra());
            }
        }
        Stmw => {
            for r in i.rs()..32 {
                push_g(&mut u, r);
            }
            push_g(&mut u, i.ra());
        }
        // integer arithmetic
        Addi | Addis => {
            push_g(&mut d, i.rd());
            if i.ra() != 0 {
                push_g(&mut u, i.ra());
            }
        }
        Addic | Addic_ => {
            push_g(&mut d, i.rd());
            d.push(CA);
            push_g(&mut u, i.ra());
            if ins.op == Addic_ {
                d.push(crf(0));
            }
        }
        Subfic => {
            push_g(&mut d, i.rd());
            d.push(CA);
            push_g(&mut u, i.ra());
        }
        Mulli => {
            push_g(&mut d, i.rd());
            push_g(&mut u, i.ra());
        }
        Add | Subf | Mullw | Mulhw | Mulhwu | Divw | Divwu => {
            push_g(&mut d, i.rd());
            push_g(&mut u, i.ra());
            push_g(&mut u, i.rb());
            rc(&mut d);
        }
        Addc | Subfc => {
            push_g(&mut d, i.rd());
            d.push(CA);
            push_g(&mut u, i.ra());
            push_g(&mut u, i.rb());
            rc(&mut d);
        }
        Adde | Subfe => {
            push_g(&mut d, i.rd());
            d.push(CA);
            push_g(&mut u, i.ra());
            push_g(&mut u, i.rb());
            u.push(CA);
            rc(&mut d);
        }
        Addze | Addme | Subfze | Subfme => {
            push_g(&mut d, i.rd());
            d.push(CA);
            push_g(&mut u, i.ra());
            u.push(CA);
            rc(&mut d);
        }
        Neg => {
            push_g(&mut d, i.rd());
            push_g(&mut u, i.ra());
            rc(&mut d);
        }
        // logical (rA = rS op rB)
        And | Andc | Or | Orc | Nor | Nand | Xor | Eqv | Slw | Srw => {
            push_g(&mut d, i.ra());
            push_g(&mut u, i.rs());
            push_g(&mut u, i.rb());
            rc(&mut d);
        }
        Sraw => {
            push_g(&mut d, i.ra());
            d.push(CA);
            push_g(&mut u, i.rs());
            push_g(&mut u, i.rb());
            rc(&mut d);
        }
        Srawi => {
            push_g(&mut d, i.ra());
            d.push(CA);
            push_g(&mut u, i.rs());
            rc(&mut d);
        }
        Ori | Oris | Xori | Xoris => {
            push_g(&mut d, i.ra());
            push_g(&mut u, i.rs());
        }
        Andi_ | Andis_ => {
            push_g(&mut d, i.ra());
            d.push(crf(0));
            push_g(&mut u, i.rs());
        }
        Extsb | Extsh | Cntlzw => {
            push_g(&mut d, i.ra());
            push_g(&mut u, i.rs());
            rc(&mut d);
        }
        Rlwinm => {
            push_g(&mut d, i.ra());
            push_g(&mut u, i.rs());
            rc(&mut d);
        }
        Rlwnm => {
            push_g(&mut d, i.ra());
            push_g(&mut u, i.rs());
            push_g(&mut u, i.rb());
            rc(&mut d);
        }
        Rlwimi => {
            push_g(&mut d, i.ra());
            push_g(&mut u, i.rs());
            push_g(&mut u, i.ra());
            rc(&mut d);
        }
        // compares
        Cmp | Cmpl => {
            d.push(crf(ins.field_crfd()));
            push_g(&mut u, i.ra());
            push_g(&mut u, i.rb());
        }
        Cmpi | Cmpli => {
            d.push(crf(ins.field_crfd()));
            push_g(&mut u, i.ra());
        }
        Fcmpu | Fcmpo => {
            d.push(crf(ins.field_crfd()));
            u.push(fpr(ins.field_fra()));
            u.push(fpr(ins.field_frb()));
        }
        PsCmpu0 | PsCmpo0 | PsCmpu1 | PsCmpo1 => {
            d.push(crf(ins.field_crfd()));
            u.push(fpr(ins.field_fra()));
            u.push(fpr(ins.field_frb()));
        }
        Cror | Crnor | Crand | Crandc | Crxor | Creqv | Crnand | Crorc => {
            let bd = ins.field_crbd();
            d.push(crf(bd >> 2));
            u.push(crf(bd >> 2));
            u.push(crf(ins.field_crba() >> 2));
            u.push(crf(ins.field_crbb() >> 2));
        }
        Mcrf => {
            d.push(crf(ins.field_crfd()));
            u.push(crf(ins.field_crfs()));
        }
        // float arithmetic
        Fadds | Fsubs | Fdivs | Fadd | Fsub | Fdiv | PsAdd | PsSub | PsDiv | PsMerge00 | PsMerge01
        | PsMerge10 | PsMerge11 => {
            d.push(fpr(ins.field_frd()));
            u.push(fpr(ins.field_fra()));
            u.push(fpr(ins.field_frb()));
            if ins.field_rc() {
                d.push(crf(1));
            }
        }
        Fmuls | Fmul | PsMul | PsMuls0 | PsMuls1 => {
            d.push(fpr(ins.field_frd()));
            u.push(fpr(ins.field_fra()));
            u.push(fpr(ins.field_frc()));
        }
        Fmadds | Fmsubs | Fnmadds | Fnmsubs | Fmadd | Fmsub | Fnmadd | Fnmsub | Fsel | PsMadd | PsMsub
        | PsNmadd | PsNmsub | PsMadds0 | PsMadds1 | PsSum0 | PsSum1 | PsSel => {
            d.push(fpr(ins.field_frd()));
            u.push(fpr(ins.field_fra()));
            u.push(fpr(ins.field_frb()));
            u.push(fpr(ins.field_frc()));
        }
        Fmr | Fneg | Fabs | Fnabs | Frsp | Fctiw | Fctiwz | Fres | Frsqrte | PsMr | PsNeg | PsAbs
        | PsNabs | PsRes | PsRsqrte => {
            d.push(fpr(ins.field_frd()));
            u.push(fpr(ins.field_frb()));
        }
        Mffs => d.push(fpr(ins.field_frd())),
        Mtfsf => u.push(fpr(ins.field_frb())),
        Mtfsb0 | Mtfsb1 | Mtfsfi => {}
        // special registers
        Mftb => push_g(&mut d, i.rd()),
        Mfspr => {
            push_g(&mut d, i.rd());
            match ins.field_spr() {
                8 => u.push(LR),
                9 => u.push(CTR),
                _ => {}
            }
        }
        Mtspr => {
            push_g(&mut u, i.rs());
            match ins.field_spr() {
                8 => d.push(LR),
                9 => d.push(CTR),
                _ => {}
            }
        }
        Mfcr => {
            push_g(&mut d, i.rd());
            u.extend((0..8).map(crf));
        }
        Mtcrf => {
            push_g(&mut u, i.rs());
            u.extend((0..8).map(crf));
        }
        // branches
        B => {
            if ins.field_lk() {
                u.extend((3..=10).map(gpr));
                u.extend((1..=8).map(fpr));
                d.extend(call_clobbers());
            }
        }
        Bc => {
            let bo = ins.field_bo();
            if bo & 0x10 == 0 {
                u.push(crf(ins.field_bi() >> 2));
            }
            if bo & 0x04 == 0 {
                u.push(CTR);
                d.push(CTR);
            }
        }
        Bcctr => {
            u.push(CTR);
            let bo = ins.field_bo();
            if bo & 0x10 == 0 {
                u.push(crf(ins.field_bi() >> 2));
            }
            if ins.field_lk() {
                u.extend((3..=10).map(gpr));
                u.extend((1..=8).map(fpr));
                d.extend(call_clobbers());
            }
        }
        Bclr => {
            u.push(LR);
            let bo = ins.field_bo();
            if bo & 0x10 == 0 {
                u.push(crf(ins.field_bi() >> 2));
            }
            if ins.field_lk() {
                u.extend((3..=10).map(gpr));
                u.extend((1..=8).map(fpr));
                d.extend(call_clobbers());
            }
        }
        _ => {}
    }
    (d, u)
}
