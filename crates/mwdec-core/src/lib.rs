//! Shared data types for mwdec. Keep this crate dependency-light; other crates own behaviour.
//! Changing a type here affects every crate: add fields/variants, don't rename or remove.

pub mod memcap;
pub mod paths;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------- objects

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelocKind {
    /// R_PPC_ADDR32
    Addr32,
    /// R_PPC_ADDR16_LO
    Addr16Lo,
    /// R_PPC_ADDR16_HI
    Addr16Hi,
    /// R_PPC_ADDR16_HA
    Addr16Ha,
    /// R_PPC_REL24 (bl/b)
    Rel24,
    /// R_PPC_REL14 (bc)
    Rel14,
    /// R_PPC_EMB_SDA21 (r2/r13 small data)
    EmbSda21,
    Other(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reloc {
    /// Byte offset from the start of the function (or data symbol).
    pub offset: u32,
    pub kind: RelocKind,
    /// Target symbol name exactly as in the object's symbol table.
    pub target: String,
    pub addend: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymBinding {
    Global,
    Local,
    Weak,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Function {
    /// Mangled symbol name.
    pub name: String,
    pub binding: SymBinding,
    /// Section-relative address in the object.
    pub address: u32,
    /// Big-endian machine code.
    pub code: Vec<u8>,
    pub relocs: Vec<Reloc>,
}

impl Function {
    pub fn words(&self) -> impl Iterator<Item = u32> + '_ {
        self.code.chunks_exact(4).map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSymbol {
    pub name: String,
    pub binding: SymBinding,
    /// e.g. ".data", ".rodata", ".sdata", ".sdata2", ".bss", ".sbss"
    pub section: String,
    pub size: u32,
    /// Empty for bss-like sections.
    pub bytes: Vec<u8>,
    pub relocs: Vec<Reloc>,
    /// Section-relative address in the object.
    #[serde(default)]
    pub address: u32,
}

/// A loaded (SHF_ALLOC) section, kept so literal bytes can be read at any symbol+addend
/// (e.g. string pools, where the referenced string extends past a symbol's declared size).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Section {
    pub name: String,
    pub size: u32,
    /// Empty for bss-like (NOBITS) sections.
    pub bytes: Vec<u8>,
    pub executable: bool,
    /// Relocations of the whole section, offsets relative to the section start.
    pub relocs: Vec<Reloc>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ObjectFile {
    pub path: String,
    pub functions: Vec<Function>,
    pub data: BTreeMap<String, DataSymbol>,
    /// Every symbol name in the object (defined or undefined).
    pub all_symbols: Vec<String>,
    /// Allocated sections (code and data), in object order.
    #[serde(default)]
    pub sections: Vec<Section>,
    /// Every defined named symbol (functions, data, local labels), sorted by (section, address).
    #[serde(default)]
    pub symbols: Vec<SymbolDef>,
}

/// A defined symbol, as in the object's symbol table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolDef {
    pub name: String,
    /// Containing section name.
    pub section: String,
    /// Section-relative address.
    pub address: u32,
    pub size: u32,
    pub binding: SymBinding,
    /// STT_FUNC
    pub is_func: bool,
}

// ---------------------------------------------------------------- project

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Unit {
    /// objdiff/report unit name, e.g. "main/MetroidPrime/CActor" or "Ripper/MetroidPrime/Enemies/CRipper"
    pub name: String,
    /// Path of the source file relative to the project root (bookkeeping only; never read by the decompiler).
    pub source: Option<String>,
    /// Target (original) object: build/G2ME01/obj/... or the REL equivalent.
    pub target_obj: String,
    /// Our compiled object, if built.
    pub base_obj: Option<String>,
    /// Full compiler flags for this unit (without -c/-o/input).
    pub cflags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetEntry {
    pub unit: String,
    pub symbol: String,
    pub size: u32,
    /// "train" or "test"
    pub split: String,
}

// ---------------------------------------------------------------- types

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    Void,
    Bool,
    /// size in bytes: 1, 2, 4, 8
    Int { size: u8, signed: bool },
    /// 4 or 8
    Float { size: u8 },
    Ptr(Box<Type>),
    Ref(Box<Type>),
    /// class/struct/union/enum/typedef by fully qualified name
    Named(String),
    Array(Box<Type>, u32),
    Const(Box<Type>),
    Volatile(Box<Type>),
    FuncPtr(Box<FuncSig>),
    /// pointer-to-member (data or function), opaque
    MemberPtr { class: String, size: u32 },
    Unknown { size: u32 },
    /// plain `char` (distinct from `signed char` = `Int{1,true}` for mangling/overloads; MWCC: signed)
    Char,
    /// `long` / `unsigned long` (4 bytes; distinct from `int` for mangling: `l` vs `i`)
    Long { signed: bool },
    /// `wchar_t` (2 bytes, unsigned)
    WChar,
}

impl Type {
    /// (size, signed) for integer-like types (Int/Long/Char/WChar/Bool), else None.
    pub fn int_info(&self) -> Option<(u8, bool)> {
        match self {
            Type::Int { size, signed } => Some((*size, *signed)),
            Type::Long { signed } => Some((4, *signed)),
            Type::Char => Some((1, true)),
            Type::WChar => Some((2, false)),
            Type::Bool => Some((1, false)),
            _ => None,
        }
    }

    /// Strip top-level const/volatile.
    pub fn unqualified(&self) -> &Type {
        match self {
            Type::Const(t) | Type::Volatile(t) => t.unqualified(),
            t => t,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Param {
    pub name: Option<String>,
    pub ty: Type,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FuncSig {
    /// Fully qualified, e.g. "CActor::SetActive" or "rstl::string::size"
    pub qualified_name: String,
    pub mangled: Option<String>,
    pub ret: Type,
    pub params: Vec<Param>,
    /// Class for non-static member functions.
    pub this_class: Option<String>,
    pub is_const: bool,
    pub is_static: bool,
    pub is_virtual: bool,
    pub variadic: bool,
}

/// C++ member access (DWARF 1.1 AT_public/AT_protected/AT_private, or the header's access section).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Access {
    #[default]
    Public,
    Protected,
    Private,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub offset: u32,
    pub ty: Type,
    /// (bit_offset_from_msb_of_storage_unit, bit_size) as MWCC/DWARF report it
    pub bitfield: Option<(u8, u8)>,
    /// Size in bytes of the field (for bitfields: of the storage unit), as DWARF reports it.
    #[serde(default)]
    pub size: u32,
    #[serde(default)]
    pub access: Access,
}

/// Static data member (MWCC DWARF emits these as typedef-tagged children of the class).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StaticMember {
    pub name: String,
    pub ty: Type,
    #[serde(default)]
    pub access: Access,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaseClass {
    pub name: String,
    pub offset: u32,
    pub is_virtual: bool,
    #[serde(default)]
    pub access: Access,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VirtualMethod {
    /// byte offset inside the vtable (MWCC GC: 8 bytes of RTTI/offset header, then 4 per slot)
    pub vtable_offset: u32,
    pub sig: FuncSig,
    /// Relocation target in the vtable data (e.g. `Think__6CActorFfR13CStateManager`, or a
    /// this-adjusting thunk `@4@__dt__13CCubeRendererFv`); empty for an empty (pure?) slot.
    #[serde(default)]
    pub symbol: String,
    /// `this` adjustment of a thunk slot (`@N@...`), 0 otherwise.
    #[serde(default)]
    pub this_adjust: i32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Class {
    pub name: String,
    pub size: u32,
    pub is_union: bool,
    pub bases: Vec<BaseClass>,
    pub fields: Vec<Field>,
    pub vtable: Vec<VirtualMethod>,
    pub methods: Vec<FuncSig>,
    /// offset of the vtable pointer if the class (or a base) is polymorphic
    pub vptr_offset: Option<u32>,
    /// `struct` (vs `class`) keyword in the source, as DWARF reports it (TAG_structure_type).
    #[serde(default)]
    pub is_struct: bool,
    /// Only forward-declared in the context (DWARF byte size 0, no members): layout unknown.
    #[serde(default)]
    pub is_declaration: bool,
    #[serde(default)]
    pub statics: Vec<StaticMember>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Enum {
    pub name: String,
    pub size: u32,
    pub values: Vec<(String, i64)>,
}

/// Source-level declaration of a function/method found by scanning the (preprocessed) context
/// headers. Used to recover what DWARF/mangling lack: return types, `static`, `virtual`, `bool`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeclInfo {
    /// Fully qualified, e.g. "CActor::GetTransform"
    pub qualified_name: String,
    pub ret: Type,
    pub params: Vec<Param>,
    pub is_const: bool,
    pub is_static: bool,
    pub is_virtual: bool,
    pub is_pure: bool,
    /// Body is defined in the header (inline accessor etc.).
    pub is_inline_defined: bool,
    pub variadic: bool,
    /// Header body of an inline definition, as space-joined tokens (`return mTransform ;`).
    #[serde(default)]
    pub inline_body: Option<String>,
    /// Template parameter names (class template's, then the function's own) when the decl
    /// belongs to a template; its types then mention them as `Type::Named("T")`.
    #[serde(default)]
    pub template_params: Vec<String>,
    /// Access of the declaration inside its class (Public for free functions).
    #[serde(default)]
    pub access: Access,
    /// Constructor initializer list of an inline definition, as space-joined tokens after the
    /// `:` (`value ( - 1 )`); None when the definition has none.
    #[serde(default)]
    pub init_list: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TypeDb {
    pub classes: BTreeMap<String, Class>,
    pub enums: BTreeMap<String, Enum>,
    pub typedefs: BTreeMap<String, Type>,
    /// keyed by mangled name when known, else qualified name
    pub functions: BTreeMap<String, FuncSig>,
    /// global variables: mangled name -> (qualified name, type)
    pub globals: BTreeMap<String, (String, Type)>,
    /// Function/method declarations scanned from the headers, keyed by qualified name
    /// (several entries per name when overloaded).
    #[serde(default)]
    pub decls: BTreeMap<String, Vec<DeclInfo>>,
    /// Namespaces declared in the context (to tell `ns::f` from `Class::f` in mangled names).
    #[serde(default)]
    pub namespaces: std::collections::BTreeSet<String>,
    /// Class templates declared in the context -> parameter names.
    #[serde(default)]
    pub templates: BTreeMap<String, Vec<String>>,
    /// Member typedefs of class templates (`rstl::vector::iterator`), in terms of the params.
    #[serde(default)]
    pub template_typedefs: BTreeMap<String, Type>,
    /// Names declared as tags in the context (`struct X`/`union X`/`enum X`) -> keyword; C code
    /// must spell such a type with its keyword unless a typedef of the same name exists.
    #[serde(default)]
    pub tag_keywords: BTreeMap<String, String>,
    /// class -> names it befriends (`friend class X;` -> "X", `friend T f(...);` -> "f"), as
    /// written in the header (unqualified or partially qualified).
    #[serde(default)]
    pub friends: BTreeMap<String, Vec<String>>,
}

// ---------------------------------------------------------------- match results

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CompareResult {
    /// Strictly identical (see DESIGN.md "Strict comparator").
    pub exact: bool,
    /// 0..=100, instruction-level similarity for guiding search.
    pub score: f64,
    pub target_len: u32,
    pub ours_len: u32,
    /// Human-readable first differences (bounded).
    pub notes: Vec<String>,
}
