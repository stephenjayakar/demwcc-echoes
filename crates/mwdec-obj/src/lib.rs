//! Load MWCC/dtk ELF objects into `mwdec_core::ObjectFile`, and disassemble.
//!
//! Both kinds of object are handled the same way:
//! - target objects (dtk split of the original binary, `build/G2ME01/obj/...`), whose relocations
//!   point at named symbols (`fn_...`, `lbl_...`, `@123`, `...data.0`);
//! - our objects (MWCC output, `build/G2ME01/src/...` or a candidate compile), whose relocations
//!   may point at section symbols; those are resolved to the containing named symbol + addend.
use anyhow::{Context, Result};
use mwdec_core::*;
use object::{
    elf, Object, ObjectSection, ObjectSymbol, RelocationFlags, RelocationTarget, SectionFlags,
    SectionIndex, SymbolKind, SymbolScope,
};
use std::collections::HashMap;

/// Map an ELF `R_PPC_*` type number to our reloc kind.
pub fn reloc_kind(r_type: u32) -> RelocKind {
    match r_type {
        1 => RelocKind::Addr32,
        4 => RelocKind::Addr16Lo,
        5 => RelocKind::Addr16Hi,
        6 => RelocKind::Addr16Ha,
        10 => RelocKind::Rel24,
        11 => RelocKind::Rel14,
        109 => RelocKind::EmbSda21,
        n => RelocKind::Other(n),
    }
}

/// Short display name of a reloc kind (as in objdump / asm syntax).
pub fn reloc_kind_name(k: RelocKind) -> String {
    match k {
        RelocKind::Addr32 => "ADDR32".into(),
        RelocKind::Addr16Lo => "@l".into(),
        RelocKind::Addr16Hi => "@h".into(),
        RelocKind::Addr16Ha => "@ha".into(),
        RelocKind::Rel24 => "REL24".into(),
        RelocKind::Rel14 => "REL14".into(),
        RelocKind::EmbSda21 => "@sda21".into(),
        RelocKind::Other(n) => format!("R_PPC_{n}"),
    }
}

/// Load every function (STT_FUNC in executable sections, any binding) and data symbol with relocations.
pub fn load_object(path: &str) -> Result<ObjectFile> {
    let data = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    load_object_bytes(path, &data)
}

#[derive(Clone)]
struct SymInfo {
    name: String,
    kind: SymbolKind,
    address: u32,
    size: u32,
    binding: SymBinding,
}

fn binding_of(sym: &object::Symbol<'_, '_>) -> SymBinding {
    if sym.is_weak() {
        SymBinding::Weak
    } else if sym.scope() == SymbolScope::Compilation || sym.is_local() {
        SymBinding::Local
    } else {
        SymBinding::Global
    }
}

/// Like [`load_object`] but from bytes; `path` is only recorded.
pub fn load_object_bytes(path: &str, data: &[u8]) -> Result<ObjectFile> {
    let file = object::File::parse(data).with_context(|| format!("parsing ELF {path}"))?;

    // Allocated sections.
    let mut sec_map: HashMap<SectionIndex, usize> = HashMap::new();
    let mut sections: Vec<Section> = Vec::new();
    for sec in file.sections() {
        let (alloc, exec) = match sec.flags() {
            SectionFlags::Elf { sh_flags } => (
                sh_flags & elf::SHF_ALLOC as u64 != 0,
                sh_flags & elf::SHF_EXECINSTR as u64 != 0,
            ),
            _ => (false, false),
        };
        if !alloc {
            continue;
        }
        let bytes = if sec.kind() == object::SectionKind::UninitializedData {
            Vec::new()
        } else {
            sec.data().unwrap_or(&[]).to_vec()
        };
        sec_map.insert(sec.index(), sections.len());
        sections.push(Section {
            name: sec.name().unwrap_or("").to_string(),
            size: sec.size() as u32,
            bytes,
            executable: exec,
            relocs: Vec::new(),
        });
    }

    // Defined named symbols per section, sorted by address.
    let mut all_symbols = Vec::new();
    let mut per_sec: Vec<Vec<SymInfo>> = vec![Vec::new(); sections.len()];
    for sym in file.symbols() {
        let name = sym.name().unwrap_or("");
        if !name.is_empty() && sym.kind() != SymbolKind::Section && sym.kind() != SymbolKind::File {
            all_symbols.push(name.to_string());
        }
        if name.is_empty() || matches!(sym.kind(), SymbolKind::Section | SymbolKind::File) {
            continue;
        }
        let Some(si) = sym.section_index().and_then(|i| sec_map.get(&i).copied()) else { continue };
        per_sec[si].push(SymInfo {
            name: name.to_string(),
            kind: sym.kind(),
            address: sym.address() as u32,
            size: sym.size() as u32,
            binding: binding_of(&sym),
        });
    }
    for v in per_sec.iter_mut() {
        // Sized symbols first at equal addresses, then globals before locals.
        v.sort_by_key(|s| (s.address, s.size == 0, s.binding == SymBinding::Local));
    }

    let sec_names: Vec<String> = sections.iter().map(|s| s.name.clone()).collect();
    // Resolve a (section, offset) pair to the containing named symbol.
    let resolve = |si: usize, off: i64| -> (String, i64) {
        let syms = &per_sec[si];
        let mut best: Option<&SymInfo> = None;
        for s in syms {
            if (s.address as i64) > off {
                break;
            }
            let end = s.address as i64 + s.size as i64;
            let contains = off < end || (s.size == 0 && s.address as i64 == off);
            if contains {
                // Prefer the innermost (latest-starting) containing symbol.
                match best {
                    Some(b) if b.address > s.address => {}
                    Some(b) if b.address == s.address => {}
                    _ => best = Some(s),
                }
            }
        }
        match best {
            Some(s) => (s.name.clone(), off - s.address as i64),
            None => (sec_names[si].clone(), off),
        }
    };

    // Relocations, per allocated section.
    let mut all_relocs: Vec<Vec<Reloc>> = vec![Vec::new(); sections.len()];
    for sec in file.sections() {
        let Some(&si) = sec_map.get(&sec.index()) else { continue };
        let mut relocs = Vec::new();
        for (offset, rel) in sec.relocations() {
            let r_type = match rel.flags() {
                RelocationFlags::Elf { r_type } => r_type,
                _ => 0,
            };
            let kind = reloc_kind(r_type);
            let addend = rel.addend();
            let (target, addend) = match rel.target() {
                RelocationTarget::Symbol(idx) => {
                    let sym = file.symbol_by_index(idx)?;
                    if sym.kind() == SymbolKind::Section {
                        match sym.section_index().and_then(|i| sec_map.get(&i).copied()) {
                            Some(tsi) => resolve(tsi, sym.address() as i64 + addend),
                            None => (sym.name().unwrap_or("?").to_string(), addend),
                        }
                    } else {
                        (sym.name().unwrap_or("?").to_string(), addend)
                    }
                }
                RelocationTarget::Section(idx) => match sec_map.get(&idx).copied() {
                    Some(tsi) => resolve(tsi, addend),
                    None => ("?section".to_string(), addend),
                },
                _ => ("?".to_string(), addend),
            };
            relocs.push(Reloc { offset: offset as u32, kind, target, addend });
        }
        relocs.sort_by_key(|r| r.offset);
        all_relocs[si] = relocs;
    }
    for (s, r) in sections.iter_mut().zip(all_relocs) {
        s.relocs = r;
    }

    let relocs_in = |si: usize, start: u32, size: u32| -> Vec<Reloc> {
        let rs = &sections[si].relocs;
        let lo = rs.partition_point(|r| r.offset < start);
        rs[lo..]
            .iter()
            .take_while(|r| r.offset < start + size)
            .map(|r| Reloc { offset: r.offset - start, ..r.clone() })
            .collect()
    };
    let bytes_of = |si: usize, start: u32, size: u32| -> Vec<u8> {
        let b = &sections[si].bytes;
        let s = (start as usize).min(b.len());
        let e = (start as usize + size as usize).min(b.len());
        b[s..e].to_vec()
    };

    let mut symbols = Vec::new();
    for (si, syms) in per_sec.iter().enumerate() {
        for s in syms {
            symbols.push(SymbolDef {
                name: s.name.clone(),
                section: sections[si].name.clone(),
                address: s.address,
                size: s.size,
                binding: s.binding,
                is_func: s.kind == SymbolKind::Text,
            });
        }
    }

    let mut functions = Vec::new();
    let mut data = std::collections::BTreeMap::new();
    for (si, syms) in per_sec.iter().enumerate() {
        let exec = sections[si].executable;
        for s in syms {
            if exec {
                if s.kind != SymbolKind::Text {
                    continue;
                }
                let size = if s.size != 0 {
                    s.size
                } else {
                    // Unsized function: extend to the next symbol or section end.
                    let next = syms.iter().map(|o| o.address).filter(|&a| a > s.address).min();
                    next.unwrap_or(sections[si].size) - s.address
                };
                functions.push(Function {
                    name: s.name.clone(),
                    binding: s.binding,
                    address: s.address,
                    code: bytes_of(si, s.address, size),
                    relocs: relocs_in(si, s.address, size),
                });
            } else {
                if s.kind == SymbolKind::Text {
                    continue;
                }
                data.entry(s.name.clone()).or_insert_with(|| DataSymbol {
                    name: s.name.clone(),
                    binding: s.binding,
                    section: sections[si].name.clone(),
                    size: s.size,
                    bytes: bytes_of(si, s.address, s.size),
                    relocs: relocs_in(si, s.address, s.size),
                    address: s.address,
                });
            }
        }
    }

    Ok(ObjectFile { path: path.to_string(), functions, data, all_symbols, sections, symbols })
}

/// Find a function by mangled name.
pub fn find_function<'a>(obj: &'a ObjectFile, name: &str) -> Option<&'a Function> {
    obj.functions.iter().find(|f| f.name == name)
}

/// Read up to `len` bytes of a data symbol starting at `addend` (may run past the symbol's
/// declared size into the rest of its section, e.g. for string pools). `None` if the symbol is
/// not defined in this object or lies in a bss-like section.
pub fn data_bytes<'a>(obj: &'a ObjectFile, symbol: &str, addend: i64, len: usize) -> Option<&'a [u8]> {
    let (sec, base) = symbol_location(obj, symbol)?;
    let start = base as i64 + addend;
    if start < 0 || start as usize > sec.bytes.len() {
        return None;
    }
    let s = start as usize;
    Some(&sec.bytes[s..(s + len).min(sec.bytes.len())])
}

/// Section and section-relative address of a defined data symbol (or of a section name itself,
/// which is what unresolvable section-relative relocations target).
pub fn symbol_location<'a>(obj: &'a ObjectFile, symbol: &str) -> Option<(&'a Section, u32)> {
    if let Some(d) = obj.symbols.iter().find(|s| s.name == symbol) {
        let sec = obj.sections.iter().find(|s| s.name == d.section)?;
        return Some((sec, d.address));
    }
    obj.sections.iter().find(|s| s.name == symbol).map(|s| (s, 0))
}

/// The NUL-terminated byte string at `symbol + addend` (without the NUL), if defined here.
pub fn c_string_at<'a>(obj: &'a ObjectFile, symbol: &str, addend: i64) -> Option<&'a [u8]> {
    let b = data_bytes(obj, symbol, addend, usize::MAX / 2)?;
    let end = b.iter().position(|&c| c == 0)?;
    Some(&b[..end])
}

fn fmt_reloc(r: &Reloc) -> String {
    let add = if r.addend == 0 {
        String::new()
    } else if r.addend > 0 {
        format!("+0x{:x}", r.addend)
    } else {
        format!("-0x{:x}", -r.addend)
    };
    format!("{} {}{}", reloc_kind_name(r.kind), r.target, add)
}

/// Disassemble a single instruction word at function offset `off` (no reloc info).
pub fn disasm_word(word: u32, off: u32) -> String {
    let ins = ppc750cl::Ins::new(word);
    let parsed = ins.simplified();
    let mut text = parsed.to_string();
    if ins.is_direct_branch() {
        if let Some(dest) = ins.branch_dest(off) {
            // Replace the relative displacement with a function-relative target.
            if let Some(pos) = text.rfind(|c: char| c == ' ' || c == ',') {
                text.truncate(pos + 1);
                text.push_str(&format!("0x{dest:x}"));
            }
        }
    }
    text
}

/// One line per instruction: "offset: word  mnemonic operands  [reloc]".
pub fn disassemble(f: &Function) -> Vec<String> {
    let mut out = Vec::new();
    let mut ri = 0usize;
    for (i, word) in f.words().enumerate() {
        let off = (i * 4) as u32;
        let mut line = format!("{off:5x}: {word:08x}  {}", disasm_word(word, off));
        while ri < f.relocs.len() && f.relocs[ri].offset < off {
            ri += 1;
        }
        let mut rs = Vec::new();
        let mut rj = ri;
        while rj < f.relocs.len() && f.relocs[rj].offset < off + 4 {
            rs.push(fmt_reloc(&f.relocs[rj]));
            rj += 1;
        }
        if !rs.is_empty() {
            let pad = 44usize.saturating_sub(line.len());
            line.push_str(&" ".repeat(pad));
            line.push_str(&format!("  [{}]", rs.join("; ")));
        }
        out.push(line);
    }
    out
}
