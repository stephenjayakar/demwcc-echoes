//! Literal formatting: floats that round-trip exactly through MWCC, C string literals.

/// Format a float literal (f32 bits if `!double`, else f64 bits) so that both direct
/// decimal->float rounding and decimal->double->float give exactly `bits`.
pub fn format_float(bits: u64, double: bool) -> String {
    if double {
        let v = f64::from_bits(bits);
        if !v.is_finite() {
            return if v.is_nan() { "(0.0 / 0.0)".into() } else if v > 0.0 { "(1.0 / 0.0)".into() } else { "(-1.0 / 0.0)".into() };
        }
        return ensure_point(&shortest_f64(v), "");
    }
    let v = f32::from_bits(bits as u32);
    if !v.is_finite() {
        return if v.is_nan() { "(0.0f / 0.0f)".into() } else if v > 0.0 { "(1.0f / 0.0f)".into() } else { "(-1.0f / 0.0f)".into() };
    }
    let s = shortest_f32(v);
    // check the double-rounding path: parse as f64 then round to f32
    let ok = s.parse::<f64>().map(|d| (d as f32).to_bits() == v.to_bits()).unwrap_or(false);
    let s = if ok { s } else { shortest_f64(v as f64) };
    ensure_point(&s, "f")
}

fn shortest_f32(v: f32) -> String {
    let a = v.abs();
    if a != 0.0 && (a < 1e-4 || a >= 1e9) {
        format!("{v:e}")
    } else {
        format!("{v}")
    }
}

fn shortest_f64(v: f64) -> String {
    let a = v.abs();
    if a != 0.0 && (a < 1e-4 || a >= 1e15) {
        format!("{v:e}")
    } else {
        format!("{v}")
    }
}

fn ensure_point(s: &str, suffix: &str) -> String {
    let mut s = s.to_string();
    if !s.contains('.') && !s.contains('e') && !s.contains("inf") && !s.contains("NaN") {
        s.push_str(".0");
    } else if s.contains('e') && !s.contains('.') {
        // 1e-5 is a valid floating literal already
    }
    s.push_str(suffix);
    s
}

/// C string literal with escapes (octal for non-printables, split after octal escapes).
pub fn c_string(bytes: &[u8]) -> String {
    let mut s = String::from("\"");
    let mut prev_octal = false;
    for &b in bytes {
        let esc = match b {
            b'"' => Some("\\\"".to_string()),
            b'\\' => Some("\\\\".to_string()),
            b'\n' => Some("\\n".to_string()),
            b'\t' => Some("\\t".to_string()),
            b'\r' => Some("\\r".to_string()),
            0x20..=0x7e => None,
            _ => Some(format!("\\{:03o}", b)),
        };
        match esc {
            Some(e) => {
                prev_octal = e.len() == 4 && e.as_bytes()[1].is_ascii_digit();
                s.push_str(&e);
            }
            None => {
                if prev_octal && b.is_ascii_digit() {
                    s.push_str("\" \"");
                }
                prev_octal = false;
                s.push(b as char);
            }
        }
    }
    s.push('"');
    s
}

