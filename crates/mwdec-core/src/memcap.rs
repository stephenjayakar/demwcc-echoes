//! Hard memory cap for mwdec processes.
//!
//! Every binary calls [`install`] first: it puts the process (and the compilers it spawns, which
//! inherit the job) in a Windows job object with a committed-memory limit, so a runaway allocation
//! fails fast inside mwdec ("memory allocation of N bytes failed") instead of exhausting the
//! system's commit charge. Processes still in the job when the process ends (normally or
//! killed) are terminated with it.
//!
//! Limit: `MWDEC_MEM_MB` (default 3072). `MWDEC_MEM_MB=0` disables the cap.

pub const DEFAULT_MB: u64 = 3072;

pub fn limit_mb() -> u64 {
    std::env::var("MWDEC_MEM_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(DEFAULT_MB)
}

/// Memory figures of this process (MB): (current commit, peak commit) of the process itself,
/// and the peak of the whole job (process + compilers it spawned). Zero where unknown.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemStats {
    pub commit_mb: u64,
    pub peak_commit_mb: u64,
    pub job_peak_mb: u64,
}

pub fn stats() -> MemStats {
    imp::stats()
}

/// Current committed memory of this process in MB (0 if unknown).
pub fn commit_mb() -> u64 {
    imp::stats().commit_mb
}

/// One-line summary for `--mem-report`.
pub fn report_line() -> String {
    let s = stats();
    format!(
        "mem: process commit {} MB (peak {} MB); job peak incl. compilers {} MB; cap {} MB",
        s.commit_mb,
        s.peak_commit_mb,
        s.job_peak_mb,
        limit_mb()
    )
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

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    /// The job handle (as usize) once installed.
    static JOB: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    pub fn stats() -> super::MemStats {
        let mut s = super::MemStats::default();
        unsafe {
            let mut pmc = ProcessMemoryCounters { cb: std::mem::size_of::<ProcessMemoryCounters>() as u32, ..Default::default() };
            if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
                s.commit_mb = (pmc.pagefile_usage >> 20) as u64;
                s.peak_commit_mb = (pmc.peak_pagefile_usage >> 20) as u64;
            }
            let job = JOB.load(std::sync::atomic::Ordering::Relaxed);
            if job != 0 {
                let mut info = ExtendedLimit::default();
                let mut ret = 0u32;
                if QueryInformationJobObject(
                    job as *mut c_void,
                    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    &mut info as *mut ExtendedLimit as *mut c_void,
                    std::mem::size_of::<ExtendedLimit>() as u32,
                    &mut ret,
                ) != 0
                {
                    s.job_peak_mb = (info.peak_job_memory_used >> 20) as u64;
                }
            }
        }
        s
    }

    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
    const JOB_OBJECT_LIMIT_JOB_MEMORY: u32 = 0x200;
    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x2000;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(attrs: *mut c_void, name: *const u16) -> *mut c_void;
        fn SetInformationJobObject(job: *mut c_void, class: i32, info: *mut c_void, len: u32) -> i32;
        fn AssignProcessToJobObject(job: *mut c_void, process: *mut c_void) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(process: *mut c_void, pmc: *mut ProcessMemoryCounters, cb: u32) -> i32;
        fn QueryInformationJobObject(job: *mut c_void, class: i32, info: *mut c_void, len: u32, ret: *mut u32) -> i32;
    }

    pub fn install(bytes: u64) -> Result<(), String> {
        unsafe {
            // The handle is deliberately leaked: the job lives as long as the process, and its
            // other processes (compilers, draft children) end with it, also when the process is
            // killed (they would otherwise hold a caller's output pipes open indefinitely).
            let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
            if job.is_null() {
                return Err(format!("CreateJobObjectW: {}", std::io::Error::last_os_error()));
            }
            let mut info = ExtendedLimit::default();
            info.basic.limit_flags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
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
            JOB.store(job as usize, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn stats() -> super::MemStats {
        super::MemStats::default()
    }
    pub fn install(_bytes: u64) -> Result<(), String> {
        Ok(())
    }
}
