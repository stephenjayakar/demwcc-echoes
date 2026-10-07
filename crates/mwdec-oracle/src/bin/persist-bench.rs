//! Benchmark + correctness check of the persistent compiler prototype: compile N candidate
//! function bodies after a shared context both normally and through `persist::PersistentCompiler`,
//! compare the annotated listings, and report the time per compile.
//!
//! usage: persist-bench [N=40] [--mixed] [--context-file TU] [--header H]... (headers relative to the project include path)
use mwdec_oracle::asm;
use mwdec_oracle::compile::Compiler;
use mwdec_oracle::persist::PersistentCompiler;
use std::time::Instant;

fn listing(obj: &[u8]) -> Vec<String> {
    let o = asm::parse(obj).unwrap();
    let mut out = vec![];
    for f in &o.funcs {
        out.push(asm::func_header(f));
        out.extend(asm::disasm_func(&o, f, asm::AsmOpts { offsets: false, literals: true }));
    }
    out.extend(asm::data_listing(&o));
    out
}

fn main() -> anyhow::Result<()> {
    mwdec_core::memcap::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = args.iter().find(|a| !a.starts_with('-') && a.parse::<usize>().is_ok()).and_then(|s| s.parse().ok()).unwrap_or(40);
    let mut headers: Vec<String> = vec![];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--header" {
            if let Some(h) = it.next() {
                headers.push(h.clone());
            }
        }
    }
    let mut context = String::new();
    if let Some(f) = args.iter().position(|a| a == "--context-file").and_then(|i| args.get(i + 1)) {
        context.push_str(&std::fs::read_to_string(f)?);
        context.push('\n');
    }
    for h in &headers {
        context.push_str(&format!("#include \"{h}\"\n"));
    }
    context.push_str("struct PS { int a; int b; float f; };\nint pget(int); float pgetf(int); extern int pg;\n");
    let comp = Compiler::default();
    let mixed = args.iter().any(|a| a == "--mixed");
    if mixed {
        context.push_str("#include \"Kyoto/Math/CVector3f.hpp\"
");
    }
    // --mixed: bodies that use header inlines, static locals, strings, switches, float literals,
    // virtual classes and templates defined after the context (state the compile creates and must
    // not leak into the next one)
    let mixed_body = |k: usize| -> String {
        match k % 6 {
            0 => format!("float M{k}(const CVector3f& a, const CVector3f& b) {{ CVector3f d = a - b * {k}.25f; return d.MagSquared() + CVector3f::Dot(a, d); }}
"),
            1 => format!("const char* M{k}(int x) {{ static int cnt; cnt += x; switch (x) {{ case 0: return \"zero{k}\"; case 1: return \"one\"; case 2: return \"two\"; case 5: return \"five\"; default: return cnt > {k} ? \"big\" : \"small\"; }} }}
"),
            2 => format!("struct V{k} {{ virtual ~V{k}(); virtual int Get() const {{ return {k}; }} int m; }};
V{k}::~V{k}() {{}}
int M{k}(V{k}* v) {{ return v->Get() + v->m; }}
"),
            3 => format!("template <typename T> T Tm{k}(T a, T b) {{ return a > b ? a - b : b - a; }}
float M{k}(float x, int y) {{ return Tm{k}(x, {k}.5f) + (float)Tm{k}(y, {k}); }}
"),
            4 => format!("static double tab{k}[3] = {{ 1.0, {k}.0, 3.5 }};
double M{k}(int i, PS* s) {{ double r = tab{k}[i % 3]; for (int j = 0; j < s->a; j++) r += tab{k}[j & 1] * j; return r; }}
"),
            _ => format!("int F{k}(PS* s, int x) {{ int r = pget({k}) + s->a; if (x > {k}) r += s->b; pg = r; return r ^ x; }}
"),
        }
    };
    let bodies: Vec<String> = (0..n)
        .map(|k| {
            if mixed {
                return mixed_body(k);
            }
            format!(
                "int F{k}(PS* s, int x) {{ int r = pget({k}) + s->a * {m}; if (x > {k}) r += s->b; s->f = pgetf(r) * {k}.5f; pg = r; return r ^ x; }}\n",
                m = k % 7 + 1
            )
        })
        .collect();
    // normal compiles
    let t0 = Instant::now();
    let mut normal = vec![];
    for b in &bodies {
        normal.push(comp.compile(&format!("{context}{b}")).map(|o| listing(&o.object)));
    }
    let t_normal = t0.elapsed();
    // persistent
    let t1 = Instant::now();
    let mut pc = PersistentCompiler::start(&comp, &context, 16 * 1024)?;
    let t_start = t1.elapsed();
    if std::env::var_os("MWDEC_PERSIST_DEBUG").is_some() {
        match pc.compile("int bad( { return ; }
") {
            Ok(o) => println!("bad body compiled?! {} bytes", o.object.len()),
            Err(e) => println!("bad body: {e}"),
        }
        match pc.compile(&bodies[0]) {
            Ok(o) => println!("body0: {} bytes, funcs {:?}", o.object.len(), asm::parse(&o.object).map(|x| x.funcs.iter().map(|f| f.name.clone()).collect::<Vec<_>>()).ok()),
            Err(e) => println!("body0: {e}"),
        }
    }
    let t2 = Instant::now();
    let mut same = 0;
    let mut diff = 0;
    let verbose = std::env::var_os("MWDEC_PERSIST_DEBUG").is_some();
    for (k, b) in bodies.iter().enumerate() {
        let tk = Instant::now();
        let r = pc.compile(b);
        if verbose {
            println!("compile {k}: {:.1} ms", tk.elapsed().as_secs_f64() * 1000.0);
        }
        match r {
            Ok(o) => {
                let l = listing(&o.object);
                match &normal[k] {
                    Ok(nl) if *nl == l => same += 1,
                    Ok(nl) => {
                        diff += 1;
                        if diff <= 2 {
                            println!("DIFF {k}:\n  normal {:?}\n  persist {:?}", &nl[..nl.len().min(12)], &l[..l.len().min(12)]);
                        }
                    }
                    Err(e) => println!("normal failed {k}: {e}"),
                }
            }
            Err(e) => {
                diff += 1;
                println!("persistent compile {k} failed: {e}");
            }
        }
    }
    let t_persist = t2.elapsed();
    println!("restore: {:.1} ms per compile; retries {}", pc.restore_time.as_secs_f64() * 1000.0 / pc.compiles.max(1) as f64, pc.retries);
    if let Some((peak, private)) = pc.memory() {
        println!("compiler process: peak working set {} MB, committed private {} MB", peak >> 20, private >> 20);
    }
    println!(
        "{n} compiles: normal {:.1} ms each; persistent {:.1} ms each (+ {:.0} ms start, {} KB restored per compile); identical {same}, different {diff}",
        t_normal.as_secs_f64() * 1000.0 / n as f64,
        t_persist.as_secs_f64() * 1000.0 / n as f64,
        t_start.as_secs_f64() * 1000.0,
        pc.snapshot_bytes / 1024
    );
    Ok(())
}
