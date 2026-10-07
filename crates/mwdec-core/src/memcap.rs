//! Hard memory cap for mwdec processes.
//!
//! Every binary calls [`install`] first: it puts the process (and the compilers it spawns, which
//! inherit the job) in a Windows job object with a committed-memory limit, so a runaway allocation
//! fails fast inside mwdec ("memory allocation of N bytes failed") instead of exhausting the
//! system's commit charge.
//!
//! Limit: `MWDEC_MEM_MB` (default 3072). `MWDEC_MEM_MB=0` disables the cap.

pub const DEFAULT_MB: u64 = 3072;

pub fn limit_mb() -> u64 {
    std::env::var("MWDEC_MEM_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(DEFAULT_MB)
}

/// Installs the cap. Never fails the caller; problems are reported on stderr.
pub fn install() {
    let mb = limit_mb();
    if mb == 0 {
        return;
    }
    if let Err(e) = imp::install(mb << 20) {
        eprintln!("mwdec memcap: could not install {mb} MB limit: {e}");
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default)]
    struct BasicLimit {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoCounters {
        counts: [u64; 6],
    }

    #[repr(C)]
    #[derive(Default)]
    struct ExtendedLimit {
        basic: BasicLimit,
        io: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
    const JOB_OBJECT_LIMIT_JOB_MEMORY: u32 = 0x200;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(attrs: *mut c_void, name: *const u16) -> *mut c_void;
        fn SetInformationJobObject(job: *mut c_void, class: i32, info: *mut c_void, len: u32) -> i32;
        fn AssignProcessToJobObject(job: *mut c_void, process: *mut c_void) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
    }

    pub fn install(bytes: u64) -> Result<(), String> {
        unsafe {
            // The handle is deliberately leaked: the job lives as long as the process.
            let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
            if job.is_null() {
                return Err(format!("CreateJobObjectW: {}", std::io::Error::last_os_error()));
            }
            let mut info = ExtendedLimit::default();
            info.basic.limit_flags = JOB_OBJECT_LIMIT_JOB_MEMORY;
            info.job_memory_limit = bytes as usize;
            if SetInformationJobObject(
                job,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                &mut info as *mut ExtendedLimit as *mut c_void,
                std::mem::size_of::<ExtendedLimit>() as u32,
            ) == 0
            {
                return Err(format!("SetInformationJobObject: {}", std::io::Error::last_os_error()));
            }
            if AssignProcessToJobObject(job, GetCurrentProcess()) == 0 {
                return Err(format!("AssignProcessToJobObject: {}", std::io::Error::last_os_error()));
            }
        }
        Ok(())
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn install(_bytes: u64) -> Result<(), String> {
        Ok(())
    }
}
