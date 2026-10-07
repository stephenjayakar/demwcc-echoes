//! mwdec-oracle: experiment harness around the real MWCC compiler.
//!
//! - [`compile::Compiler`]: compile a source string with a project flag profile.
//! - [`asm`]: parse the ELF and print an annotated listing (relocs, literal values, labels).
//! - [`variants`]: experiment files with variants + machine-checked expectations.
//! - [`webs`]: recover callee-saved register webs (live ranges, interference) from object code.
//! - [`regalloc`]: model of MWCC's register colouring (predict / inverse-solve priority orders).
pub mod asm;
pub mod compile;
pub mod flags;
pub mod hints;
pub mod regalloc;
pub mod schedcheck;
pub mod tracer;
pub mod variants;
pub mod webs;
