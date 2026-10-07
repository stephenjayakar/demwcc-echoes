//! Experiment files: a shared preamble plus named variants, with machine-checked expectations.
//!
//! ```text
//! //@ fn A__Fi            (optional, repeatable) only list functions whose name contains this
//! //@ profile rel         (optional) flag profile for the whole file
//! extern int get(int);    preamble shared by every variant
//! //@ variant decl_order
//! void A(int p) { ... }
//! //@ expect bl get__Fi / mr r31, r3      consecutive lines (regex per line, labels skipped)
//! //@ expect stw r31 ... stw r30          " ... " = later, not necessarily adjacent
//! //@ expect-not stmw
//! //@ same-as other_variant               identical listing (selected functions)
//! //@ differs-from other_variant
//! //@ flags -inline off                   extra compiler flags for this variant
//! //@ profile rel                         inside a variant: profile for that variant only
//! //@ expect-data @stringBase0 ... 0x3f000000   regexes over the `--data` listing lines
//! //@ same-as-exact other                 like same-as but compiler label ids (@N) must match too
//! //@ strip-frame                         (file level) expectations ignore prologue/epilogue lines
//! ```
//! `//@ fn` / `//@ expect` lines placed before the first variant apply to every variant.
//! Patterns are regexes matched against TRIMMED instruction lines (so `^bl\s`, not `^\s+bl`).
//! `same-as`/`differs-from` and the `var` "identical" check ignore function names and renumber
//! compiler-local labels (`@123`) by first appearance, so only real differences count.

use crate::asm::{self, AsmOpts};
use crate::compile::Compiler;
use crate::flags::Profile;
use anyhow::{anyhow, Result};
use regex::Regex;

#[derive(Clone, Debug)]
pub enum Expect {
    Seq(String),
    Not(String),
    SameAs(String),
    DiffersFrom(String),
    SameAsExact(String),
    Data(String),
}

#[derive(Clone, Debug, Default)]
pub struct Variant {
    pub name: String,
    pub body: String,
    pub expects: Vec<Expect>,
    pub flags: Vec<String>,
    pub line: usize,
    pub profile: Option<Profile>,
}

#[derive(Clone, Debug, Default)]
pub struct ExpFile {
    pub preamble: String,
    pub fn_filters: Vec<String>,
    pub profile: Option<Profile>,
    pub global_expects: Vec<Expect>,
    pub variants: Vec<Variant>,
    pub strip_frame: bool,
}

pub fn parse_exp(text: &str) -> Result<ExpFile> {
    let mut f = ExpFile::default();
    let mut cur: Option<Variant> = None;
    for (ln, line) in text.lines().enumerate() {
        let t = line.trim_start();
        if let Some(d) = t.strip_prefix("//@") {
            let d = d.trim();
            let (kw, rest) = d.split_once(char::is_whitespace).unwrap_or((d, ""));
            let rest = rest.trim().to_string();
            match kw {
                "variant" => {
                    if let Some(v) = cur.take() {
                        f.variants.push(v);
                    }
                    cur = Some(Variant { name: rest, line: ln + 1, ..Default::default() });
                }
                "fn" => f.fn_filters.push(rest),
                "profile" => {
                    let p = Profile::parse(&rest).ok_or_else(|| anyhow!("bad profile {rest}"))?;
                    match cur.as_mut() {
                        Some(v) => v.profile = Some(p),
                        None => f.profile = Some(p),
                    }
                }
                "strip-frame" => f.strip_frame = true,
                "flags" => match cur.as_mut() {
                    Some(v) => v.flags.extend(rest.split_whitespace().map(String::from)),
                    None => return Err(anyhow!("line {}: //@ flags outside a variant", ln + 1)),
                },
                "expect" | "expect-not" | "same-as" | "differs-from" | "same-as-exact" | "expect-data" => {
                    let e = match kw {
                        "expect" => Expect::Seq(rest),
                        "expect-not" => Expect::Not(rest),
                        "same-as" => Expect::SameAs(rest),
                        "same-as-exact" => Expect::SameAsExact(rest),
                        "expect-data" => Expect::Data(rest),
                        _ => Expect::DiffersFrom(rest),
                    };
                    match cur.as_mut() {
                        Some(v) => v.expects.push(e),
                        None => f.global_expects.push(e),
                    }
                }
                _ => {
                    // unknown directive: keep as comment text
                    push_line(&mut f, &mut cur, line);
                }
            }
            continue;
        }
        push_line(&mut f, &mut cur, line);
    }
    if let Some(v) = cur.take() {
        f.variants.push(v);
    }
    if f.variants.is_empty() {
        f.variants.push(Variant { name: "main".into(), ..Default::default() });
    }
    Ok(f)
}

fn push_line(f: &mut ExpFile, cur: &mut Option<Variant>, line: &str) {
    match cur.as_mut() {
        Some(v) => {
            v.body.push_str(line);
            v.body.push('\n');
        }
        None => {
            f.preamble.push_str(line);
            f.preamble.push('\n');
        }
    }
}

pub struct VariantResult {
    pub name: String,
    /// Listing lines of the selected functions (with `.fn` headers).
    pub listing: Result<Vec<String>, String>,
    pub data: Vec<String>,
}

/// Select functions by filters (substring of mangled name). Empty filters = all.
pub fn listing_for(obj: &asm::Obj, filters: &[String], opts: AsmOpts) -> Vec<String> {
    let mut out = Vec::new();
    for f in &obj.funcs {
        if !filters.is_empty() && !filters.iter().any(|p| f.name.contains(p.as_str())) {
            continue;
        }
        out.push(asm::func_header(f));
        out.extend(asm::disasm_func(obj, f, opts));
    }
    out
}

pub fn run_file(comp: &Compiler, exp: &ExpFile, opts: AsmOpts, jobs: usize) -> Vec<VariantResult> {
    let mut comp = comp.clone();
    if let Some(p) = exp.profile {
        comp.profile = p;
    }
    let work: Vec<(usize, &Variant)> = exp.variants.iter().enumerate().collect();
    let mut results: Vec<Option<VariantResult>> = (0..work.len()).map(|_| None).collect();
    let chunks: Vec<Vec<(usize, &Variant)>> = {
        let j = jobs.max(1);
        let mut c: Vec<Vec<(usize, &Variant)>> = (0..j).map(|_| Vec::new()).collect();
        for (k, w) in work.into_iter().enumerate() {
            c[k % j].push(w);
        }
        c
    };
    std::thread::scope(|s| {
        let hs: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let comp = comp.clone();
                s.spawn(move || {
                    chunk
                        .into_iter()
                        .map(|(i, v)| {
                            let mut c = comp.clone();
                            if let Some(p) = v.profile {
                                c.profile = p;
                            }
                            c.extra.extend(v.flags.iter().cloned());
                            // Keep line numbers meaningful: pad the body to its original line.
                            let pre_lines = exp.preamble.lines().count();
                            let pad = v.line.saturating_sub(pre_lines);
                            let src = format!("{}{}{}", exp.preamble, "\n".repeat(pad), v.body);
                            let r = match c.compile(&src) {
                                Ok(out) => match asm::parse(&out.object) {
                                    Ok(obj) => (Ok(listing_for(&obj, &exp.fn_filters, opts)), asm::data_listing(&obj)),
                                    Err(e) => (Err(format!("{e:#}")), vec![]),
                                },
                                Err(e) => (Err(format!("{e:#}")), vec![]),
                            };
                            (i, VariantResult { name: v.name.clone(), listing: r.0, data: r.1 })
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in hs {
            for (i, r) in h.join().unwrap() {
                results[i] = Some(r);
            }
        }
    });
    results.into_iter().map(|r| r.unwrap()).collect()
}

fn instr_lines(listing: &[String]) -> Vec<String> {
    listing.iter().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
}

/// Check one sequence expectation ("a / b" adjacent, "a ... b" gapped) against a listing.
pub fn seq_matches(listing: &[String], pat: &str) -> Result<bool> {
    let lines: Vec<String> =
        instr_lines(listing).into_iter().filter(|l| !l.ends_with(':') && !l.starts_with(".fn ")).collect();
    // groups separated by " ... ", each group = adjacent regexes separated by " / "
    let groups: Vec<Vec<Regex>> = pat
        .split(" ... ")
        .map(|g| g.split(" / ").map(|p| Regex::new(p.trim())).collect::<Result<Vec<_>, _>>())
        .collect::<Result<Vec<_>, _>>()?;
    let mut pos = 0usize;
    for g in &groups {
        let mut found = None;
        'outer: for start in pos..lines.len() {
            if start + g.len() > lines.len() {
                break;
            }
            for (k, re) in g.iter().enumerate() {
                if !re.is_match(&lines[start + k]) {
                    continue 'outer;
                }
            }
            found = Some(start + g.len());
            break;
        }
        match found {
            Some(p) => pos = p,
            None => return Ok(false),
        }
    }
    Ok(true)
}

pub struct CheckReport {
    pub passed: usize,
    pub failed: Vec<String>,
}

pub fn check(exp: &ExpFile, results: &[VariantResult]) -> CheckReport {
    let mut rep = CheckReport { passed: 0, failed: vec![] };
    let get = |n: &str| results.iter().find(|r| r.name == n);
    for (v, r) in exp.variants.iter().zip(results) {
        let listing = match &r.listing {
            Ok(l) => l,
            Err(e) => {
                rep.failed.push(format!("[{}] compile error: {}", v.name, e.lines().next().unwrap_or("")));
                continue;
            }
        };
        let matched: Vec<String> = if exp.strip_frame { strip_frame(listing) } else { listing.clone() };
        for e in exp.global_expects.iter().chain(v.expects.iter()) {
            let (ok, desc) = match e {
                Expect::Seq(p) => (seq_matches(&matched, p).unwrap_or(false), format!("expect {p}")),
                Expect::Not(p) => (!seq_matches(&matched, p).unwrap_or(true), format!("expect-not {p}")),
                Expect::Data(p) => (seq_matches(&r.data, p).unwrap_or(false), format!("expect-data {p}")),
                Expect::SameAs(o) | Expect::DiffersFrom(o) | Expect::SameAsExact(o) => {
                    let exact = matches!(e, Expect::SameAsExact(_));
                    let same = match get(o).map(|x| &x.listing) {
                        Some(Ok(ol)) => {
                            if exact {
                                strip_names_exact(ol) == strip_names_exact(listing)
                            } else {
                                strip_names(ol) == strip_names(listing)
                            }
                        }
                        _ => {
                            rep.failed.push(format!("[{}] no such variant {o}", v.name));
                            continue;
                        }
                    };
                    let want_same = !matches!(e, Expect::DiffersFrom(_));
                    let kw = match e {
                        Expect::SameAs(_) => "same-as",
                        Expect::SameAsExact(_) => "same-as-exact",
                        _ => "differs-from",
                    };
                    (same == want_same, format!("{kw} {o}"))
                }
            };
            if ok {
                rep.passed += 1;
            } else {
                rep.failed.push(format!("[{}] FAILED {}", v.name, desc));
            }
        }
    }
    rep
}

/// Listing without `.fn` headers' names (so variants may name functions differently), exact labels.
pub fn strip_names_exact(l: &[String]) -> Vec<String> {
    l.iter().map(|x| if x.starts_with(".fn ") { ".fn".to_string() } else { x.trim().to_string() }).collect()
}

/// Like [`strip_names_exact`] but compiler-local labels (`@123`) renumbered by first appearance, so
/// listings that differ only in literal-pool label ids compare equal.
pub fn strip_names(l: &[String]) -> Vec<String> {
    let re = Regex::new(r"@(\d+)").unwrap();
    let mut map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    strip_names_exact(l)
        .into_iter()
        .map(|x| {
            re.replace_all(&x, |c: &regex::Captures| {
                let n = map.len();
                let k = *map.entry(c[1].to_string()).or_insert(n);
                format!("@L{k}")
            })
            .into_owned()
        })
        .collect()
}

/// Drop prologue/epilogue lines (frame setup, LR save/restore, callee-saved saves/restores).
pub fn strip_frame(l: &[String]) -> Vec<String> {
    let re = Regex::new(
        r"^(stwu r1, |mflr r0$|mtlr r0$|addi r1, r1, |(stw|lwz) r0, 0x[0-9a-f]+\(r1\)$|(stw|lwz) r(1[4-9]|2[0-9]|3[01]), 0x[0-9a-f]+\(r1\)$|(stmw|lmw) r[0-9]+, |(stfd|lfd) f(1[4-9]|2[0-9]|3[01]), 0x[0-9a-f]+\(r1\)$|psq_(st|l) f(1[4-9]|2[0-9]|3[01]), 0x[0-9a-f]+\(r1\))",
    )
    .unwrap();
    l.iter().filter(|x| !re.is_match(x.trim())).cloned().collect()
}

fn norm(l: &str) -> &str {
    let t = l.trim();
    if t.starts_with(".fn ") { ".fn" } else { t }
}

/// Line diff (LCS); returns lines prefixed with ' ', '-', '+'.
pub fn diff(a: &[String], b: &[String]) -> Vec<String> {
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if norm(&a[i]) == norm(&b[j]) { dp[i + 1][j + 1] + 1 } else { dp[i + 1][j].max(dp[i][j + 1]) };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < n || j < m {
        if i < n && j < m && norm(&a[i]) == norm(&b[j]) {
            out.push(format!("  {}", a[i]));
            i += 1;
            j += 1;
        } else if j < m && (i == n || dp[i][j + 1] >= dp[i + 1][j]) {
            out.push(format!("+ {}", b[j]));
            j += 1;
        } else {
            out.push(format!("- {}", a[i]));
            i += 1;
        }
    }
    out
}

