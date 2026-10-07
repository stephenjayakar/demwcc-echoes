//! Minimal build.ninja reader: enough to get the `mwcc*` compile edges and their variables.
use anyhow::Result;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct MwccEdge {
    /// Output object, as written in build.ninja (may use backslashes).
    pub output: String,
    /// Source file.
    pub input: String,
    pub rule: String,
    /// Tokenized `cflags` (quotes removed), without -MMD/-c/-o/input.
    pub cflags: Vec<String>,
    /// e.g. `GC\2.7`
    pub mw_version: String,
}

/// Join `$\n` continuations into logical lines, keeping the leading indentation of the first line.
fn logical_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cont = false;
    for raw in text.lines() {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        let piece = if cont { line.trim_start() } else { line };
        // A trailing '$' is a continuation unless it is an escaped "$$".
        let trailing = piece.len() - piece.trim_end_matches('$').len();
        if trailing % 2 == 1 {
            cur.push_str(&piece[..piece.len() - 1]);
            cont = true;
        } else {
            cur.push_str(piece);
            out.push(std::mem::take(&mut cur));
            cont = false;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Evaluate a ninja value: `$$`, `$ `, `$:` escapes and `$var` / `${var}` expansion.
fn eval(value: &str, scopes: &[&HashMap<String, String>]) -> String {
    let mut out = String::new();
    let b = value.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'$' && i + 1 < b.len() {
            let c = b[i + 1];
            match c {
                b'$' | b' ' | b':' => {
                    out.push(c as char);
                    i += 2;
                }
                b'{' => {
                    let end = value[i + 2..].find('}').map(|e| i + 2 + e).unwrap_or(b.len());
                    let name = &value[i + 2..end];
                    out.push_str(&lookup(name, scopes));
                    i = end + 1;
                }
                _ if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' => {
                    let mut e = i + 1;
                    while e < b.len() && (b[e].is_ascii_alphanumeric() || b[e] == b'_' || b[e] == b'-') {
                        e += 1;
                    }
                    out.push_str(&lookup(&value[i + 1..e], scopes));
                    i = e;
                }
                _ => {
                    out.push('$');
                    i += 1;
                }
            }
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn lookup(name: &str, scopes: &[&HashMap<String, String>]) -> String {
    scopes.iter().find_map(|s| s.get(name).cloned()).unwrap_or_default()
}

/// Split a `build` line's path list on unescaped spaces, unescaping `$ `, `$:`, `$$`.
fn split_paths(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '$' => {
                if let Some(n) = it.next() {
                    cur.push(n);
                }
            }
            ' ' => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Find the first unescaped ':' in a build line.
fn find_colon(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'$' {
            i += 2;
            continue;
        }
        if b[i] == b':' {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Tokenize a command-line string shell-style (whitespace separated, double quotes group).
pub fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    let mut have = false;
    for c in s.chars() {
        match c {
            '"' => {
                in_q = !in_q;
                have = true;
            }
            c if c.is_whitespace() && !in_q => {
                if have {
                    out.push(std::mem::take(&mut cur));
                    have = false;
                }
            }
            _ => {
                cur.push(c);
                have = true;
            }
        }
    }
    if have {
        out.push(cur);
    }
    out
}

/// All `build <obj>: mwcc* <src>` edges with evaluated `cflags` / `mw_version`.
pub fn mwcc_edges(text: &str) -> Result<Vec<MwccEdge>> {
    let mut globals: HashMap<String, String> = HashMap::new();
    let mut edges = Vec::new();
    let lines = logical_lines(text);
    let mut i = 0;
    while i < lines.len() {
        let line = &lines[i];
        i += 1;
        if line.trim_start().starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let indented = line.starts_with(' ') || line.starts_with('\t');
        if indented {
            continue; // rule/pool/other-edge body we don't care about
        }
        if let Some(rest) = line.strip_prefix("build ") {
            let Some(colon) = find_colon(rest) else { continue };
            let outs = split_paths(&rest[..colon]);
            let rhs = split_paths(&rest[colon + 1..]);
            // Edge-scoped variables.
            let mut vars: HashMap<String, String> = HashMap::new();
            while i < lines.len() && (lines[i].starts_with(' ') || lines[i].starts_with('\t')) {
                if let Some((k, v)) = lines[i].trim().split_once('=') {
                    let v = eval(v.trim_start(), &[&vars, &globals]);
                    vars.insert(k.trim().to_string(), v);
                }
                i += 1;
            }
            let Some(rule) = rhs.first() else { continue };
            if !rule.starts_with("mwcc") || rule.starts_with("mwcc_pch") {
                continue;
            }
            let Some(input) = rhs.get(1).filter(|s| *s != "|" && *s != "||") else { continue };
            let Some(output) = outs.first() else { continue };
            let cflags = split_args(&lookup("cflags", &[&vars, &globals]));
            let mw_version = lookup("mw_version", &[&vars, &globals]);
            edges.push(MwccEdge {
                output: output.clone(),
                input: input.clone(),
                rule: rule.clone(),
                cflags,
                mw_version,
            });
        } else if let Some((k, v)) = line.split_once('=') {
            let k = k.trim();
            if !k.is_empty() && !k.contains(' ') {
                let v = eval(v.trim_start(), &[&globals]);
                globals.insert(k.to_string(), v);
            }
        }
    }
    Ok(edges)
}

