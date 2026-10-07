//! Minimal ELF reader + annotated disassembly for MWCC objects (self-contained: object + ppc750cl).

use anyhow::{Context, Result};
use object::{
    Object, ObjectSection, ObjectSymbol, RelocationFlags, RelocationTarget, SectionIndex, SymbolKind,
};
use ppc750cl::{Argument, Ins};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug)]
pub struct Rel {
    pub offset: u32,
    pub r_type: u32,
    pub target: String,
    pub addend: i64,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub name: String,
    pub weak: bool,
    pub local: bool,
    pub section: String,
    pub code: Vec<u8>,
    pub relocs: Vec<Rel>,
    /// section-relative address
    pub address: u32,
}

#[derive(Clone, Debug)]
pub struct DataSym {
    pub name: String,
    pub section: String,
    pub address: u32,
    pub size: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Obj {
    pub funcs: Vec<Func>,
    pub data: Vec<DataSym>,
    /// section name -> bytes
    pub sections: HashMap<String, Vec<u8>>,
    /// data symbol name -> (section, address)
    pub sym_loc: HashMap<String, (String, u32)>,
    /// DWARF 1 `.line` table (compiled with `-sym on`): (.text address, source line), sorted
    pub lines: Vec<(u32, u32)>,
}

pub fn parse(bytes: &[u8]) -> Result<Obj> {
    let file = object::File::parse(bytes).context("parsing ELF")?;
    let mut o = Obj::default();
    let mut secname: HashMap<SectionIndex, String> = HashMap::new();
    for s in file.sections() {
        let n = s.name().unwrap_or("").to_string();
        secname.insert(s.index(), n.clone());
        if let Ok(d) = s.data() {
            o.sections.insert(n, d.to_vec());
        }
    }
    // symbols by section
    let mut per: HashMap<SectionIndex, Vec<(u32, u32, String, SymbolKind, bool, bool)>> = HashMap::new();
    for sym in file.symbols() {
        let name = sym.name().unwrap_or("");
        if name.is_empty() || matches!(sym.kind(), SymbolKind::Section | SymbolKind::File) {
            continue;
        }
        let Some(si) = sym.section_index() else { continue };
        per.entry(si).or_default().push((
            sym.address() as u32,
            sym.size() as u32,
            name.to_string(),
            sym.kind(),
            sym.is_weak(),
            sym.is_local(),
        ));
    }
    for v in per.values_mut() {
        v.sort_by_key(|s| (s.0, s.1 == 0));
    }
    let resolve = |si: SectionIndex, off: i64| -> (String, i64) {
        if let Some(v) = per.get(&si) {
            let mut best: Option<&(u32, u32, String, SymbolKind, bool, bool)> = None;
            for s in v {
                if s.0 as i64 > off {
                    break;
                }
                if off < s.0 as i64 + s.1 as i64 || (s.1 == 0 && s.0 as i64 == off) {
                    best = Some(s);
                }
            }
            if let Some(s) = best {
                return (s.2.clone(), off - s.0 as i64);
            }
        }
        (secname.get(&si).cloned().unwrap_or_default(), off)
    };
    let mut relocs_by_sec: HashMap<SectionIndex, Vec<Rel>> = HashMap::new();
    for s in file.sections() {
        let mut v = Vec::new();
        for (off, r) in s.relocations() {
            let r_type = match r.flags() {
                RelocationFlags::Elf { r_type } => r_type,
                _ => 0,
            };
            let (target, addend) = match r.target() {
                RelocationTarget::Symbol(i) => {
                    let sym = file.symbol_by_index(i)?;
                    if sym.kind() == SymbolKind::Section {
                        match sym.section_index() {
                            Some(tsi) => resolve(tsi, sym.address() as i64 + r.addend()),
                            None => ("?".into(), r.addend()),
                        }
                    } else {
                        (sym.name().unwrap_or("?").to_string(), r.addend())
                    }
                }
                RelocationTarget::Section(tsi) => resolve(tsi, r.addend()),
                _ => ("?".into(), r.addend()),
            };
            v.push(Rel { offset: off as u32, r_type, target, addend });
        }
        v.sort_by_key(|r| r.offset);
        relocs_by_sec.insert(s.index(), v);
    }
    // Emit functions in section order, then address order.
    let mut secs: Vec<SectionIndex> = per.keys().copied().collect();
    secs.sort_by_key(|s| s.0);
    for si in secs {
        let sname = secname.get(&si).cloned().unwrap_or_default();
        let sec = file.section_by_index(si)?;
        let sdata = sec.data().unwrap_or(&[]);
        let syms = &per[&si];
        for (k, s) in syms.iter().enumerate() {
            if s.3 == SymbolKind::Text {
                let size = if s.1 != 0 {
                    s.1
                } else {
                    syms.get(k + 1).map(|n| n.0).unwrap_or(sec.size() as u32) - s.0
                };
                let start = s.0 as usize;
                let end = (start + size as usize).min(sdata.len());
                let rels = relocs_by_sec
                    .get(&si)
                    .map(|v| {
                        v.iter()
                            .filter(|r| r.offset >= s.0 && r.offset < s.0 + size)
                            .map(|r| Rel { offset: r.offset - s.0, ..r.clone() })
                            .collect()
                    })
                    .unwrap_or_default();
                o.funcs.push(Func {
                    name: s.2.clone(),
                    weak: s.4,
                    local: s.5,
                    section: sname.clone(),
                    code: sdata.get(start..end).unwrap_or(&[]).to_vec(),
                    relocs: rels,
                    address: s.0,
                });
            } else {
                o.sym_loc.insert(s.2.clone(), (sname.clone(), s.0));
                o.data.push(DataSym { name: s.2.clone(), section: sname.clone(), address: s.0, size: s.1 });
            }
        }
    }
    // DWARF 1.1 .line: chunks of [u32 length][u32 base (relocated)] + entries {u32 line, u16 col, u32 delta}
    if let Some(ls) = file.sections().find(|s| s.name().ok() == Some(".line")) {
        let data = ls.data().unwrap_or(&[]);
        let rels = relocs_by_sec.get(&ls.index()).cloned().unwrap_or_default();
        let text_addr = |name: &str| -> u32 { o.funcs.iter().find(|f| f.name == name).map(|f| f.address).unwrap_or(0) };
        let mut pos = 0usize;
        while pos + 8 <= data.len() {
            let len = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
            if len < 8 || pos + len > data.len() {
                break;
            }
            let raw_base = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
            let base = match rels.iter().find(|r| r.offset as usize == pos + 4) {
                Some(r) => text_addr(&r.target).wrapping_add(r.addend as u32),
                None => raw_base,
            };
            let mut e = pos + 8;
            while e + 10 <= pos + len {
                let line = u32::from_be_bytes([data[e], data[e + 1], data[e + 2], data[e + 3]]);
                let delta = u32::from_be_bytes([data[e + 6], data[e + 7], data[e + 8], data[e + 9]]);
                if line != 0 {
                    o.lines.push((base.wrapping_add(delta), line));
                }
                e += 10;
            }
            pos += len;
        }
        o.lines.sort();
    }
    Ok(o)
}

impl Obj {
    /// Source line of the instruction at section address `addr` (needs `-sym on`).
    pub fn line_at(&self, addr: u32) -> Option<u32> {
        let i = self.lines.partition_point(|&(a, _)| a <= addr);
        if i == 0 {
            None
        } else {
            Some(self.lines[i - 1].1)
        }
    }
}

impl Obj {
    pub fn bytes_at(&self, sym: &str, addend: i64, len: usize) -> Option<&[u8]> {
        let (sec, addr) = self.sym_loc.get(sym)?;
        let b = self.sections.get(sec)?;
        let s = (*addr as i64 + addend) as usize;
        if s + len > b.len() {
            return None;
        }
        Some(&b[s..s + len])
    }
    pub fn cstr_at(&self, sym: &str, addend: i64) -> Option<String> {
        let (sec, addr) = self.sym_loc.get(sym)?;
        let b = self.sections.get(sec)?;
        let s = (*addr as i64 + addend) as usize;
        let rest = b.get(s..)?;
        let e = rest.iter().position(|&c| c == 0)?;
        Some(String::from_utf8_lossy(&rest[..e]).into_owned())
    }
}

fn kind_suffix(r_type: u32) -> &'static str {
    match r_type {
        4 => "@l",
        5 => "@h",
        6 => "@ha",
        109 => "@sda21",
        _ => "",
    }
}

fn fmt_target(r: &Rel) -> String {
    if r.addend == 0 {
        r.target.clone()
    } else if r.addend > 0 {
        format!("{}+0x{:x}", r.target, r.addend)
    } else {
        format!("{}-0x{:x}", r.target, -r.addend)
    }
}

/// Literal value annotation for a data reference made by `mnemonic`.
fn literal(obj: &Obj, mnemonic: &str, r: &Rel) -> Option<String> {
    let fl = |b: &[u8]| f32::from_be_bytes([b[0], b[1], b[2], b[3]]);
    match mnemonic {
        "lfs" | "lfsu" | "lfsx" => obj.bytes_at(&r.target, r.addend, 4).map(|b| format!("{:?}f", fl(b))),
        "lfd" | "lfdu" | "lfdx" => obj.bytes_at(&r.target, r.addend, 8).map(|b| {
            let d = f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            let w = u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            format!("{d:?} (0x{w:016x})")
        }),
        _ => {
            if r.target.contains("stringBase") || r.target.starts_with("@stringBase") {
                obj.cstr_at(&r.target, r.addend).map(|s| format!("{s:?}"))
            } else {
                None
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AsmOpts {
    pub offsets: bool,
    pub literals: bool,
}

impl Default for AsmOpts {
    fn default() -> Self {
        AsmOpts { offsets: false, literals: true }
    }
}

/// Disassemble one function into lines (labels `.L_xx:` for in-function branch targets).
pub fn disasm_func(obj: &Obj, f: &Func, o: AsmOpts) -> Vec<String> {
    let words: Vec<u32> =
        f.code.chunks_exact(4).map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]])).collect();
    let rel_at: HashMap<u32, &Rel> = f.relocs.iter().map(|r| (r.offset & !3, r)).collect();
    // branch targets
    let mut targets = BTreeSet::new();
    for (i, &w) in words.iter().enumerate() {
        let off = (i * 4) as u32;
        let ins = Ins::new(w);
        if ins.is_direct_branch() && !rel_at.contains_key(&off) {
            if let Some(d) = ins.branch_dest(off) {
                targets.insert(d);
            }
        }
    }
    let mut out = Vec::new();
    for (i, &w) in words.iter().enumerate() {
        let off = (i * 4) as u32;
        if targets.contains(&off) {
            out.push(format!(".L_{off:x}:"));
        }
        let ins = Ins::new(w);
        let p = ins.simplified();
        let rel = rel_at.get(&off).copied();
        let mut text = String::from(p.mnemonic);
        let args: Vec<&Argument> = p.args_iter().collect();
        let mut first = true;
        let mut in_off = false;
        let mut substituted = false;
        for a in args.iter() {
            let s = match a {
                Argument::BranchDest(_) => {
                    if let Some(r) = rel {
                        substituted = true;
                        fmt_target(r)
                    } else if let Some(d) = ins.branch_dest(off) {
                        format!(".L_{d:x}")
                    } else {
                        a.to_string()
                    }
                }
                Argument::Simm(_) | Argument::Uimm(_) | Argument::Offset(_) if rel.is_some() && !substituted => {
                    substituted = true;
                    let r = rel.unwrap();
                    format!("{}{}", fmt_target(r), kind_suffix(r.r_type))
                }
                _ => a.to_string(),
            };
            if first {
                text.push(' ');
                first = false;
            } else if !in_off {
                text.push_str(", ");
            }
            text.push_str(&s);
            if let Argument::Offset(_) = a {
                text.push('(');
                in_off = true;
            } else if in_off {
                text.push(')');
                in_off = false;
            }
        }
        if let Some(r) = rel {
            if !substituted {
                text.push_str(&format!("  # {}{}", fmt_target(r), kind_suffix(r.r_type)));
            }
            if o.literals {
                if let Some(v) = literal(obj, p.mnemonic, r) {
                    text.push_str(&format!("  # ={v}"));
                }
            }
        }
        if o.offsets {
            out.push(format!("{off:4x}: {text}"));
        } else {
            out.push(format!("    {text}"));
        }
    }
    out
}

pub fn func_header(f: &Func) -> String {
    let b = if f.weak { "weak" } else if f.local { "local" } else { "global" };
    format!(".fn {} ({}, 0x{:x}){}", f.name, b, f.code.len(), if f.section != ".text" { format!(" [{}]", f.section) } else { String::new() })
}

/// Data symbols with values (pool order matters for matching).
pub fn data_listing(obj: &Obj) -> Vec<String> {
    let mut out = Vec::new();
    for d in &obj.data {
        if d.section == ".text" {
            continue;
        }
        let mut line = format!("{:10} {:#06x} {:4} {}", d.section, d.address, d.size, d.name);
        if let Some(b) = obj.bytes_at(&d.name, 0, d.size as usize) {
            if d.name.contains("stringBase") {
                let parts: Vec<String> = b.split(|&c| c == 0).filter(|s| !s.is_empty()).map(|s| format!("{:?}", String::from_utf8_lossy(s))).collect();
                line.push_str(&format!("  {}", parts.join(" ")));
            } else if d.size == 4 {
                let f = f32::from_be_bytes([b[0], b[1], b[2], b[3]]);
                line.push_str(&format!("  0x{:08x} ({:?}f)", u32::from_be_bytes([b[0], b[1], b[2], b[3]]), f));
            } else if d.size == 8 {
                let w = u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                line.push_str(&format!("  0x{:016x} ({:?})", w, f64::from_bits(w)));
            } else if d.size <= 32 {
                line.push_str(&format!("  {}", b.iter().map(|x| format!("{x:02x}")).collect::<String>()));
            }
        }
        out.push(line);
    }
    out
}
