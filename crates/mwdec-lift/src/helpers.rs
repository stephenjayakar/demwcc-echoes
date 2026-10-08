//! Helper types a draft may use that no header declares (the emitter defines them first in the
//! preamble): passes put `Type::Named(<helper name>)` into the IR, and
//! [`definitions`] gives the definitions of every helper a rendered body mentions.
//!
//! - `mwdec_words_<N>`: `struct { int w[N/4]; }`, an N-byte block copied as one object (MWCC
//!   copies it word by word with rotating registers, as it does any struct of that size).

const WORDS: &str = "mwdec_words_";

/// The helper type of an `nbytes`-byte block of words (`nbytes` a multiple of 4).
pub fn words(nbytes: u32) -> mwdec_core::Type {
    mwdec_core::Type::Named(format!("{WORDS}{nbytes}"))
}

/// Whether `name` is a helper type (defined by [`definitions`], not by any header).
pub fn is_helper(name: &str) -> bool {
    name.strip_prefix(WORDS).is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Definitions of the helper types mentioned in `text`, in first-mention order.
pub fn definitions(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: Vec<u32> = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(WORDS) {
        let tail = &rest[i + WORDS.len()..];
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        let before_ok = i == 0 || !rest[..i].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_');
        if let (true, Ok(n)) = (before_ok, digits.parse::<u32>()) {
            if n > 0 && n % 4 == 0 && !seen.contains(&n) {
                seen.push(n);
                out.push(format!("struct {WORDS}{n} {{ int w[{}]; }};", n / 4));
            }
        }
        rest = &tail[digits.len()..];
    }
    out
}

