//! Owned concrete syntax tree over tree-sitter-cpp, plus text edits.
//!
//! The permuter works on source text: every mutation is a set of byte-range replacements on the
//! original text, followed by a re-parse. The arena form ([`Cst`]) avoids tree-sitter lifetimes and
//! gives cheap parent links.
use std::cell::RefCell;

#[derive(Clone, Debug)]
pub struct N {
    pub kind: &'static str,
    /// Field name in the parent (`left`, `body`, `condition`, ...).
    pub field: Option<&'static str>,
    pub start: usize,
    pub end: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub named: bool,
}

#[derive(Clone, Debug)]
pub struct Cst {
    pub src: String,
    pub nodes: Vec<N>,
    /// ERROR + MISSING nodes.
    pub errors: usize,
}

thread_local! {
    static PARSER: RefCell<tree_sitter::Parser> = RefCell::new({
        let mut p = tree_sitter::Parser::new();
        p.set_language(&tree_sitter_cpp::LANGUAGE.into()).expect("tree-sitter-cpp");
        p
    });
}

impl Cst {
    pub fn parse(src: &str) -> Cst {
        // tree-sitter-cpp does not know `__typeof__(e)` as a type: parse a copy where each such
        // span is masked to an identifier of the same byte length (offsets stay valid; node
        // texts still come from the original source).
        let masked = mask_typeof(src);
        let tree = PARSER.with(|p| p.borrow_mut().parse(masked.as_deref().unwrap_or(src), None)).expect("parse");
        let mut nodes: Vec<N> = Vec::new();
        let mut errors = 0;
        let mut cur = tree.walk();
        // Iterative preorder walk with an explicit parent stack.
        let mut stack: Vec<usize> = Vec::new();
        loop {
            let n = cur.node();
            if n.is_error() || n.is_missing() {
                errors += 1;
            }
            let id = nodes.len();
            nodes.push(N {
                kind: n.kind(),
                field: cur.field_name(),
                start: n.start_byte(),
                end: n.end_byte(),
                parent: stack.last().copied(),
                children: Vec::new(),
                named: n.is_named(),
            });
            if let Some(&p) = stack.last() {
                nodes[p].children.push(id);
            }
            if cur.goto_first_child() {
                stack.push(id);
                continue;
            }
            loop {
                if cur.goto_next_sibling() {
                    break;
                }
                if !cur.goto_parent() {
                    return Cst { src: src.to_string(), nodes, errors };
                }
                stack.pop();
            }
        }
    }

    #[inline]
    pub fn kind(&self, i: usize) -> &'static str {
        self.nodes[i].kind
    }

    #[inline]
    pub fn text(&self, i: usize) -> &str {
        &self.src[self.nodes[i].start..self.nodes[i].end]
    }

    pub fn child(&self, i: usize, field: &str) -> Option<usize> {
        self.nodes[i].children.iter().copied().find(|&c| self.nodes[c].field == Some(field))
    }

    pub fn children_by_field<'a>(&'a self, i: usize, field: &'a str) -> impl Iterator<Item = usize> + 'a {
        self.nodes[i].children.iter().copied().filter(move |&c| self.nodes[c].field == Some(field))
    }

    /// Named children, skipping comments.
    pub fn named(&self, i: usize) -> Vec<usize> {
        self.nodes[i].children.iter().copied().filter(|&c| self.nodes[c].named && self.nodes[c].kind != "comment").collect()
    }

    /// First anonymous child with this text (an operator token).
    pub fn token(&self, i: usize, field: &str) -> Option<usize> {
        self.child(i, field)
    }

    pub fn parent(&self, i: usize) -> Option<usize> {
        self.nodes[i].parent
    }

    /// Preorder descendants including `i`.
    pub fn descendants(&self, i: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut st = vec![i];
        while let Some(n) = st.pop() {
            out.push(n);
            for &c in self.nodes[n].children.iter().rev() {
                st.push(c);
            }
        }
        out
    }

    /// Is `a` an ancestor of (or equal to) `b`?
    pub fn contains(&self, a: usize, b: usize) -> bool {
        self.nodes[a].start <= self.nodes[b].start && self.nodes[b].end <= self.nodes[a].end && {
            let mut x = Some(b);
            while let Some(n) = x {
                if n == a {
                    return true;
                }
                x = self.nodes[n].parent;
            }
            false
        }
    }

    pub fn ancestors(&self, i: usize) -> Vec<usize> {
        let mut v = Vec::new();
        let mut x = self.nodes[i].parent;
        while let Some(n) = x {
            v.push(n);
            x = self.nodes[n].parent;
        }
        v
    }

    /// Operator text of a binary/assignment/unary/update expression.
    pub fn op(&self, i: usize) -> Option<&str> {
        self.child(i, "operator").map(|o| self.text(o))
    }
}

fn mask_typeof(src: &str) -> Option<String> {
    const KW: &str = "__typeof__(";
    if !src.contains(KW) {
        return None;
    }
    let mut b = src.as_bytes().to_vec();
    let mut from = 0;
    while let Some(k) = src[from..].find(KW) {
        let s = from + k;
        let mut depth = 0i32;
        let mut e = s + KW.len() - 1;
        while e < b.len() {
            match src.as_bytes()[e] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            e += 1;
        }
        let e = (e + 1).min(b.len());
        for x in &mut b[s..e] {
            *x = b'_';
        }
        from = e;
    }
    String::from_utf8(b).ok()
}

/// A byte-range replacement.
#[derive(Clone, Debug)]
pub struct Edit {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

impl Edit {
    pub fn replace(cst: &Cst, node: usize, text: impl Into<String>) -> Edit {
        Edit { start: cst.nodes[node].start, end: cst.nodes[node].end, text: text.into() }
    }
    pub fn insert(at: usize, text: impl Into<String>) -> Edit {
        Edit { start: at, end: at, text: text.into() }
    }
}

/// Apply non-overlapping edits. Returns `None` if edits overlap.
pub fn apply(src: &str, edits: &[Edit]) -> Option<String> {
    let mut es: Vec<&Edit> = edits.iter().collect();
    es.sort_by_key(|e| (e.start, e.end));
    let mut out = String::with_capacity(src.len() + 64);
    let mut pos = 0;
    for e in es {
        if e.start < pos || e.end < e.start || e.end > src.len() {
            return None;
        }
        out.push_str(&src[pos..e.start]);
        out.push_str(&e.text);
        pos = e.end;
    }
    out.push_str(&src[pos..]);
    Some(out)
}

/// Whitespace/comment-insensitive normal form used for deduplication.
pub fn normalize(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut pend_space = false;
    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            pend_space = true;
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            pend_space = true;
            continue;
        }
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
            pend_space = true;
            continue;
        }
        if pend_space {
            // Keep a separator only between two word characters.
            if let Some(&l) = out.as_bytes().last() {
                if is_word(l) && is_word(c) {
                    out.push(' ');
                }
            }
            pend_space = false;
        }
        if c == b'"' || c == b'\'' {
            let q = c;
            let s = i;
            i += 1;
            while i < b.len() && b[i] != q {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i = (i + 1).min(b.len());
            out.push_str(&src[s..i]);
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// Strip redundant outer parentheses.
pub fn strip_parens(s: &str) -> &str {
    let mut s = s.trim();
    while s.starts_with('(') && s.ends_with(')') {
        // Only if the first '(' matches the last ')'.
        let b = s.as_bytes();
        let mut depth = 0i32;
        let mut ok = true;
        for (k, &c) in b.iter().enumerate() {
            if c == b'(' {
                depth += 1;
            } else if c == b')' {
                depth -= 1;
                if depth == 0 && k != b.len() - 1 {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            break;
        }
        s = s[1..s.len() - 1].trim();
    }
    s
}


impl Cst {
    /// Debug dump (S-expression-ish with fields).
    pub fn dump(&self, i: usize) -> String {
        let mut s = String::new();
        self.dump_rec(i, 0, &mut s);
        s
    }
    fn dump_rec(&self, i: usize, d: usize, s: &mut String) {
        let n = &self.nodes[i];
        let f = n.field.map(|f| format!("{f}: ")).unwrap_or_default();
        if n.children.is_empty() {
            s.push_str(&format!("{:w$}{f}{} {:?}\n", "", n.kind, self.text(i), w = d * 2));
        } else {
            s.push_str(&format!("{:w$}{f}{}\n", "", n.kind, w = d * 2));
            for &c in &n.children {
                self.dump_rec(c, d + 1, s);
            }
        }
    }
}

