//! Minimal DWARF 1.1 reader for MWCC (GC/2.x) relocatable objects.
//!
//! MWCC writes one compile-unit DIE in `.debug` and refers to other DIEs through
//! `R_PPC_UADDR32`/`R_PPC_ADDR32` relocations against per-DIE local/weak symbols named
//! `.dwarf.<tag>.<qualified name>` (named types) or `.dwarf.<tag>..<n>` (anonymous). We apply those
//! relocations to a copy of the section bytes, then parse DIEs. The weak symbol names are also
//! the only place where the fully qualified (namespace + template arguments, CW-mangled) name of
//! a type appears: the DIE `AT_name` is just the simple identifier.
//!
//! Format reference: DWARF 1.1 spec + decomp-toolkit `src/util/dwarf` (ported in spirit).
use anyhow::{bail, Context, Result};
use object::{Object, ObjectSection, ObjectSymbol, RelocationTarget};
use std::collections::HashMap;

pub mod tag {
    pub const PADDING: u16 = 0x00;
    pub const ARRAY_TYPE: u16 = 0x01;
    pub const CLASS_TYPE: u16 = 0x02;
    pub const ENUMERATION_TYPE: u16 = 0x04;
    pub const FORMAL_PARAMETER: u16 = 0x05;
    pub const GLOBAL_SUBROUTINE: u16 = 0x06;
    pub const GLOBAL_VARIABLE: u16 = 0x07;
    pub const LOCAL_VARIABLE: u16 = 0x0c;
    pub const MEMBER: u16 = 0x0d;
    pub const COMPILE_UNIT: u16 = 0x11;
    pub const STRUCTURE_TYPE: u16 = 0x13;
    pub const SUBROUTINE: u16 = 0x14;
    pub const SUBROUTINE_TYPE: u16 = 0x15;
    pub const TYPEDEF: u16 = 0x16;
    pub const UNION_TYPE: u16 = 0x17;
    pub const UNSPECIFIED_PARAMETERS: u16 = 0x18;
    pub const INHERITANCE: u16 = 0x1c;
    pub const INLINED_SUBROUTINE: u16 = 0x1d;
    pub const PTR_TO_MEMBER_TYPE: u16 = 0x1f;
}

pub mod at {
    pub const SIBLING: u16 = 0x0012;
    pub const LOCATION: u16 = 0x0023;
    pub const NAME: u16 = 0x0038;
    pub const FUND_TYPE: u16 = 0x0055;
    pub const MOD_FUND_TYPE: u16 = 0x0063;
    pub const USER_DEF_TYPE: u16 = 0x0072;
    pub const MOD_UD_TYPE: u16 = 0x0083;
    pub const SUBSCR_DATA: u16 = 0x00a3;
    pub const BYTE_SIZE: u16 = 0x00b6;
    pub const BIT_OFFSET: u16 = 0x00c5;
    pub const BIT_SIZE: u16 = 0x00d6;
    pub const ELEMENT_LIST: u16 = 0x00f4;
    pub const LOW_PC: u16 = 0x0111;
    pub const MEMBER: u16 = 0x0142;
    pub const CONTAINING_TYPE: u16 = 0x01d2;
    pub const PRIVATE: u16 = 0x0248;
    pub const PROTECTED: u16 = 0x0268;
    pub const PUBLIC: u16 = 0x0288;
    pub const PURE_VIRTUAL: u16 = 0x0298;
    pub const VIRTUAL: u16 = 0x0308;
    pub const MW_MANGLED: u16 = 0x2008;
}

#[derive(Clone, Debug)]
pub enum AttrVal {
    Addr(u32),
    Ref(u32),
    Data2(u16),
    Data4(u32),
    Data8(u64),
    Block(Vec<u8>),
    Str(String),
}

#[derive(Clone, Debug)]
pub struct Die {
    /// Offset in `.debug` (the DIE's key).
    pub off: u32,
    /// Total size including the 4-byte length.
    pub size: u32,
    /// 0 for null entries.
    pub tag: u16,
    pub attrs: Vec<(u16, AttrVal)>,
}

impl Die {
    pub fn attr(&self, a: u16) -> Option<&AttrVal> {
        self.attrs.iter().find(|(k, _)| *k == a).map(|(_, v)| v)
    }
    pub fn has(&self, a: u16) -> bool {
        self.attrs.iter().any(|(k, _)| *k == a)
    }
    pub fn name(&self) -> Option<&str> {
        match self.attr(at::NAME) {
            Some(AttrVal::Str(s)) => Some(s),
            _ => None,
        }
    }
    pub fn sibling(&self) -> Option<u32> {
        match self.attr(at::SIBLING) {
            Some(AttrVal::Ref(r)) => Some(*r),
            _ => None,
        }
    }
    pub fn data4(&self, a: u16) -> Option<u32> {
        match self.attr(a) {
            Some(AttrVal::Data4(v)) => Some(*v),
            _ => None,
        }
    }
    pub fn data2(&self, a: u16) -> Option<u16> {
        match self.attr(a) {
            Some(AttrVal::Data2(v)) => Some(*v),
            _ => None,
        }
    }
    pub fn reference(&self, a: u16) -> Option<u32> {
        match self.attr(a) {
            Some(AttrVal::Ref(v)) => Some(*v),
            _ => None,
        }
    }
    pub fn block(&self, a: u16) -> Option<&[u8]> {
        match self.attr(a) {
            Some(AttrVal::Block(v)) => Some(v),
            _ => None,
        }
    }
    pub fn string(&self, a: u16) -> Option<&str> {
        match self.attr(a) {
            Some(AttrVal::Str(v)) => Some(v),
            _ => None,
        }
    }
}

/// Parsed `.debug` section.
#[derive(Debug, Default)]
pub struct DwarfInfo {
    pub dies: Vec<Die>,
    pub index: HashMap<u32, usize>,
    /// DIE offset -> name of the `.dwarf.*` symbol defined there (tag prefix stripped),
    /// e.g. `rstl::auto_ptr<9CAnimData>::auto_ptr`.
    pub sym_names: HashMap<u32, String>,
}

impl DwarfInfo {
    pub fn get(&self, off: u32) -> Option<&Die> {
        self.index.get(&off).map(|&i| &self.dies[i])
    }

    /// Children of the DIE at index `i` (DWARF 1: sibling chain starting right after the DIE,
    /// ending at the DIE's own sibling).
    pub fn children(&self, i: usize) -> Vec<usize> {
        let end = self.dies[i].sibling();
        let mut out = Vec::new();
        let mut cur = i + 1;
        while cur < self.dies.len() {
            let d = &self.dies[cur];
            if Some(d.off) == end {
                break;
            }
            if d.tag == tag::PADDING {
                // null entry terminates a sibling chain
                break;
            }
            out.push(cur);
            match d.sibling().and_then(|s| self.index.get(&s).copied()) {
                Some(n) if n > cur => cur = n,
                _ => break,
            }
        }
        out
    }

    /// Top-level DIEs (children of each compile unit).
    pub fn roots(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for (i, d) in self.dies.iter().enumerate() {
            if d.tag == tag::COMPILE_UNIT {
                out.extend(self.children(i));
            }
        }
        out
    }
}

/// Read `.debug` from an ELF object (bytes), applying its relocations.
pub fn read_dwarf(elf: &[u8]) -> Result<DwarfInfo> {
    let file = object::File::parse(elf).context("parsing ELF")?;
    let Some(sec) = file.section_by_name(".debug") else {
        bail!("object has no .debug section (no DWARF emitted)");
    };
    let sec_index = sec.index();
    let mut data = sec.data()?.to_vec();
    for (off, rel) in sec.relocations() {
        let RelocationTarget::Symbol(si) = rel.target() else { continue };
        let sym = file.symbol_by_index(si)?;
        let v = (sym.address() as i64 + rel.addend()) as u32;
        let o = off as usize;
        if o + 4 <= data.len() {
            data[o..o + 4].copy_from_slice(&v.to_be_bytes());
        }
    }
    let mut sym_names = HashMap::new();
    for sym in file.symbols() {
        if sym.section_index() != Some(sec_index) {
            continue;
        }
        let Ok(name) = sym.name() else { continue };
        // ".dwarf.0002.rstl::auto_ptr<Uc>::auto_ptr" / ".dwarf.0015..354"
        if let Some(rest) = name.strip_prefix(".dwarf.") {
            if rest.len() > 5 && rest.as_bytes()[4] == b'.' {
                let q = &rest[5..];
                if !q.is_empty() && !q.starts_with('.') {
                    sym_names.insert(sym.address() as u32, q.to_string());
                }
            }
        }
    }
    let mut info = parse_debug(&data)?;
    info.sym_names = sym_names;
    Ok(info)
}

fn rd_u16(d: &[u8], p: usize) -> Result<u16> {
    d.get(p..p + 2).map(|b| u16::from_be_bytes([b[0], b[1]])).context("truncated DWARF")
}
fn rd_u32(d: &[u8], p: usize) -> Result<u32> {
    d.get(p..p + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]])).context("truncated DWARF")
}

pub fn parse_debug(data: &[u8]) -> Result<DwarfInfo> {
    let mut info = DwarfInfo::default();
    let mut pos = 0usize;
    while pos + 4 <= data.len() {
        let size = rd_u32(data, pos)? as usize;
        if size < 8 {
            info.index.insert(pos as u32, info.dies.len());
            info.dies.push(Die { off: pos as u32, size: size.max(4) as u32, tag: 0, attrs: Vec::new() });
            pos += size.max(4);
            continue;
        }
        let end = pos + size;
        if end > data.len() {
            bail!("DIE at {pos:#x} overruns .debug");
        }
        let tagv = rd_u16(data, pos + 4)?;
        let mut p = pos + 6;
        let mut attrs = Vec::new();
        if tagv != 0 {
            while p < end {
                let a = rd_u16(data, p)?;
                p += 2;
                let v = match a & 0xf {
                    0x1 => {
                        p += 4;
                        AttrVal::Addr(rd_u32(data, p - 4)?)
                    }
                    0x2 => {
                        p += 4;
                        AttrVal::Ref(rd_u32(data, p - 4)?)
                    }
                    0x3 => {
                        let l = rd_u16(data, p)? as usize;
                        p += 2 + l;
                        AttrVal::Block(data.get(p - l..p).context("truncated block")?.to_vec())
                    }
                    0x4 => {
                        let l = rd_u32(data, p)? as usize;
                        p += 4 + l;
                        AttrVal::Block(data.get(p - l..p).context("truncated block")?.to_vec())
                    }
                    0x5 => {
                        p += 2;
                        AttrVal::Data2(rd_u16(data, p - 2)?)
                    }
                    0x6 => {
                        p += 4;
                        AttrVal::Data4(rd_u32(data, p - 4)?)
                    }
                    0x7 => {
                        let hi = rd_u32(data, p)? as u64;
                        let lo = rd_u32(data, p + 4)? as u64;
                        p += 8;
                        AttrVal::Data8(hi << 32 | lo)
                    }
                    0x8 => {
                        let e = data[p..end].iter().position(|&b| b == 0).map(|x| p + x).unwrap_or(end);
                        // Shift-JIS bytes are mapped 1:1 (latin-1 style); identifiers are ASCII.
                        let s: String = data[p..e].iter().map(|&b| b as char).collect();
                        p = e + 1;
                        AttrVal::Str(s)
                    }
                    f => bail!("unknown DWARF form {f:#x} (attr {a:#06x}) in DIE {pos:#x}"),
                };
                attrs.push((a, v));
            }
        }
        info.index.insert(pos as u32, info.dies.len());
        info.dies.push(Die { off: pos as u32, size: size as u32, tag: tagv, attrs });
        pos = end;
    }
    Ok(info)
}
