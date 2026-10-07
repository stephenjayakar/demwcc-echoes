//! Compiler flag profiles, copied from the project's build.ninja (the distinct cflags sets).
//! Include paths are made absolute so compiles can run from any working directory.

use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// main.dol game code (762 units): GC/2.7, C++, deferred,noauto, inline_max_size(125).
    Game,
    /// REL game code (27 units): Game + `-sdata 0 -sdata2 0 -pool off`.
    Rel,
    /// REL with pooling (3 units): Game + `-sdata 0 -sdata2 0`.
    RelPool,
    /// SDK / runtime C (57 units): GC/2.7, C, deferred,auto, common off.
    SdkC,
    /// Dolphin SDK C (78 units): GC/1.2.5n.
    Sdk125,
    /// MusyX (31 units): GC/1.3.2.
    Musyx,
}

impl Profile {
    pub fn parse(s: &str) -> Option<Profile> {
        Some(match s {
            "game" | "main" => Profile::Game,
            "rel" => Profile::Rel,
            "relpool" | "rel-pool" => Profile::RelPool,
            "sdkc" | "c" => Profile::SdkC,
            "sdk125" | "1.2.5n" => Profile::Sdk125,
            "musyx" | "1.3.2" => Profile::Musyx,
            _ => return None,
        })
    }

    /// Compiler version directory under build/compilers/.
    pub fn version(self) -> &'static str {
        match self {
            Profile::Sdk125 => "GC/1.2.5n",
            Profile::Musyx => "GC/1.3.2",
            _ => "GC/2.7",
        }
    }

    pub fn is_c(self) -> bool {
        matches!(self, Profile::SdkC | Profile::Sdk125 | Profile::Musyx)
    }
}

fn split(s: &str) -> Vec<String> {
    // Minimal shell-like split honouring double quotes.
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut q = false;
    let mut any = false;
    for c in s.chars() {
        match c {
            '"' => {
                q = !q;
                any = true;
            }
            c if c.is_whitespace() && !q => {
                if any {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            c => {
                cur.push(c);
                any = true;
            }
        }
    }
    if any {
        out.push(cur);
    }
    out
}

/// Full flag list (without `-c`, input, `-o`) for a profile. `root` = project root (for include paths).
pub fn profile_flags(p: Profile, root: &Path) -> Vec<String> {
    let r = root.to_string_lossy().replace('\\', "/");
    let inc = format!("-i {r}/include -i {r}/libc -i {r}/build/G2ME01/include");
    let common = "-nodefaults -proc gekko -align powerpc -enum int -fp hardware -Cpp_exceptions off -O4,p \
        -inline auto -pragma \"cats off\" -pragma \"warn_notinlined off\" -maxerrors 1 -nosyspath -RTTI off \
        -fp_contract on -str reuse";
    let defs = "-DBUILD_VERSION=0 -DVERSION=0 -multibyte -DNDEBUG=1";
    let game = format!(
        "{common} {inc} {defs} -use_lmw_stmw on -str reuse,pool,readonly -gccinc -inline deferred,noauto -common on \
         -i {r}/extern/musyx/include -DMUSY_TARGET=MUSY_TARGET_DOLPHIN -DMUSY_VERSION_MAJOR=2 -DMUSY_VERSION_MINOR=0 \
         -DMUSY_VERSION_PATCH=3 -pragma \"inline_max_size(125)\""
    );
    let s = match p {
        Profile::Game => format!("{game} -lang=c++"),
        Profile::Rel => format!("{game} -sdata 0 -sdata2 0 -pool off -lang=c++"),
        Profile::RelPool => format!("{game} -sdata 0 -sdata2 0 -lang=c++"),
        Profile::SdkC => format!(
            "{common} {inc} {defs} -use_lmw_stmw on -str reuse,pool,readonly -gccinc -common off \
             -inline deferred,auto -DMSL_OLD_FP_CLASSIFY -DMSL_NO_INLINE_SQRT -lang=c"
        ),
        Profile::Sdk125 => format!("{common} {inc} {defs} -multibyte -fp_contract off -lang=c"),
        Profile::Musyx => format!(
            "-proc gekko -nodefaults -nosyspath -i {r}/include -i {r}/libc -i {r}/extern/musyx/include \
             -inline auto,depth=4 -O4,p -fp hard -enum int -sym on -Cpp_exceptions off -str reuse,pool,readonly \
             -fp_contract off -DMUSY_TARGET=MUSY_TARGET_DOLPHIN -DM_PI=3.14159265358979323846 \
             -DMUSY_VERSION_MAJOR=2 -DMUSY_VERSION_MINOR=0 -DMUSY_VERSION_PATCH=3 -lang=c"
        ),
    };
    split(&s)
}

