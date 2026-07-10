//! FFI bindings for Qwen3Guard safety classification.

use crate::model_architectures::generative::Qwen3GuardModel;
use candle_core::Device;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Global Qwen3Guard instance (for safety/jailbreak detection)
static GLOBAL_QWEN3_GUARD: OnceLock<Mutex<Qwen3GuardModel>> = OnceLock::new();

static QWEN3_GUARD_TIMING: Qwen3GuardTimingAccumulator = Qwen3GuardTimingAccumulator::new();

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen3GuardTimingStats {
    pub calls: u64,
    pub errors: u64,
    pub lock_wait_ns_total: u64,
    pub lock_wait_ns_max: u64,
    pub generation_ns_total: u64,
    pub generation_ns_max: u64,
}

struct Qwen3GuardTimingAccumulator {
    calls: AtomicU64,
    errors: AtomicU64,
    lock_wait_ns_total: AtomicU64,
    lock_wait_ns_max: AtomicU64,
    generation_ns_total: AtomicU64,
    generation_ns_max: AtomicU64,
}

impl Qwen3GuardTimingAccumulator {
    const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            lock_wait_ns_total: AtomicU64::new(0),
            lock_wait_ns_max: AtomicU64::new(0),
            generation_ns_total: AtomicU64::new(0),
            generation_ns_max: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
        self.lock_wait_ns_total.store(0, Ordering::Relaxed);
        self.lock_wait_ns_max.store(0, Ordering::Relaxed);
        self.generation_ns_total.store(0, Ordering::Relaxed);
        self.generation_ns_max.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> Qwen3GuardTimingStats {
        Qwen3GuardTimingStats {
            calls: self.calls.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            lock_wait_ns_total: self.lock_wait_ns_total.load(Ordering::Relaxed),
            lock_wait_ns_max: self.lock_wait_ns_max.load(Ordering::Relaxed),
            generation_ns_total: self.generation_ns_total.load(Ordering::Relaxed),
            generation_ns_max: self.generation_ns_max.load(Ordering::Relaxed),
        }
    }

    fn record(&self, lock_wait: Duration, generation: Duration, error: bool) {
        let lock_wait_ns = duration_as_u64_ns(lock_wait);
        let generation_ns = duration_as_u64_ns(generation);

        self.calls.fetch_add(1, Ordering::Relaxed);
        if error {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        self.lock_wait_ns_total
            .fetch_add(lock_wait_ns, Ordering::Relaxed);
        self.generation_ns_total
            .fetch_add(generation_ns, Ordering::Relaxed);
        store_max(&self.lock_wait_ns_max, lock_wait_ns);
        store_max(&self.generation_ns_max, generation_ns);
    }
}

fn duration_as_u64_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn store_max(slot: &AtomicU64, value: u64) {
    let mut current = slot.load(Ordering::Relaxed);
    while value > current {
        match slot.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

/// Guard generation result returned to Go (raw text only)
#[repr(C)]
pub struct GuardResult {
    /// Raw generated output (null-terminated C string)
    pub raw_output: *mut c_char,

    /// Error flag
    pub error: bool,

    /// Error message (null-terminated C string, only set if error=true)
    pub error_message: *mut c_char,
}

impl Default for GuardResult {
    fn default() -> Self {
        Self {
            raw_output: ptr::null_mut(),
            error: true,
            error_message: ptr::null_mut(),
        }
    }
}

fn create_error_message(msg: &str) -> *mut c_char {
    match CString::new(msg) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free guard result
///
/// # Safety
/// - `result` must be a valid pointer to a `GuardResult` initialized by this FFI module.
/// - Must only be called once per result; the owned string pointers inside the result must not
///   be freed elsewhere.
#[no_mangle]
pub unsafe extern "C" fn free_guard_result(result: *mut GuardResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        if !(*result).raw_output.is_null() {
            let _ = CString::from_raw((*result).raw_output);
        }

        if !(*result).error_message.is_null() {
            let _ = CString::from_raw((*result).error_message);
        }
    }
}

/// Initialize Qwen3Guard model
///
/// # Arguments
/// - `model_path`: Path to Qwen3Guard model directory
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn init_qwen3_guard(model_path: *const c_char) -> i32 {
    if model_path.is_null() {
        eprintln!("Error: model_path is null");
        return -1;
    }

    let model_path_str = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_path: {}", e);
                return -1;
            }
        }
    };

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    if GLOBAL_QWEN3_GUARD.get().is_some() {
        println!("Qwen3Guard already initialized, reusing existing instance");
        return 0;
    }

    match Qwen3GuardModel::new(model_path_str, &device, None) {
        Ok(guard) => match GLOBAL_QWEN3_GUARD.set(Mutex::new(guard)) {
            Ok(_) => {
                println!("Qwen3Guard initialized: {}", model_path_str);
                0
            }
            Err(_) => {
                println!("Qwen3Guard already initialized (race condition), reusing");
                0
            }
        },
        Err(e) => {
            eprintln!("Error: failed to load Qwen3Guard: {}", e);
            -1
        }
    }
}

/// Classify text with Qwen3Guard
///
/// # Arguments
/// - `text`: Input text to classify (null-terminated C string)
/// - `mode`: Classification mode ("input" for user prompts, "output" for model responses)
/// - `result`: Pointer to GuardResult struct (allocated by caller)
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - `text` and `mode` must be valid null-terminated C strings.
/// - `result` must be a valid writable pointer for one `GuardResult`.
/// - Caller must later release owned fields with `free_guard_result`.
#[no_mangle]
pub unsafe extern "C" fn classify_with_qwen3_guard(
    text: *const c_char,
    mode: *const c_char,
    result: *mut GuardResult,
) -> i32 {
    if text.is_null() || mode.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to classify_with_qwen3_guard");
        return -1;
    }

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = GuardResult::default();
                (*result).error_message = create_error_message(&format!("Invalid UTF-8: {}", e));
                return -1;
            }
        }
    };

    let mode_str = unsafe {
        match CStr::from_ptr(mode).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in mode: {}", e);
                (*result) = GuardResult::default();
                (*result).error_message = create_error_message(&format!("Invalid UTF-8: {}", e));
                return -1;
            }
        }
    };

    let guard_mutex = match GLOBAL_QWEN3_GUARD.get() {
        Some(g) => g,
        None => {
            eprintln!("Error: Qwen3Guard not initialized");
            unsafe {
                (*result) = GuardResult::default();
                (*result).error_message = create_error_message("Guard not initialized");
            }
            return -1;
        }
    };

    let lock_start = Instant::now();
    match guard_mutex.lock() {
        Ok(mut guard) => {
            let lock_wait = lock_start.elapsed();
            let generation_start = Instant::now();
            let generation_result = guard.generate_guard(text_str, mode_str);
            let generation_elapsed = generation_start.elapsed();
            QWEN3_GUARD_TIMING.record(lock_wait, generation_elapsed, generation_result.is_err());

            match generation_result {
                Ok(guard_result) => {
                    let raw_output_c = match CString::new(guard_result.raw_output.as_str()) {
                        Ok(s) => s.into_raw(),
                        Err(e) => {
                            eprintln!("Error: failed to create raw_output C string: {}", e);
                            unsafe {
                                (*result) = GuardResult::default();
                                (*result).error_message = create_error_message(&format!(
                                    "Failed to create C string: {}",
                                    e
                                ));
                            }
                            return -1;
                        }
                    };

                    unsafe {
                        (*result) = GuardResult {
                            raw_output: raw_output_c,
                            error: false,
                            error_message: ptr::null_mut(),
                        };
                    }

                    0
                }
                Err(e) => {
                    eprintln!("Error: guard classification failed: {}", e);
                    unsafe {
                        (*result) = GuardResult::default();
                        (*result).error_message =
                            create_error_message(&format!("Classification failed: {}", e));
                    }
                    -1
                }
            }
        }
        Err(e) => {
            QWEN3_GUARD_TIMING.record(lock_start.elapsed(), Duration::ZERO, true);
            eprintln!("Error: failed to acquire lock: {}", e);
            unsafe {
                (*result) = GuardResult::default();
                (*result).error_message =
                    create_error_message(&format!("Failed to acquire lock: {}", e));
            }
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn reset_qwen3_guard_timing_stats() {
    QWEN3_GUARD_TIMING.reset();
}

#[no_mangle]
pub unsafe extern "C" fn get_qwen3_guard_timing_stats(stats: *mut Qwen3GuardTimingStats) -> i32 {
    if stats.is_null() {
        return -1;
    }

    unsafe {
        *stats = QWEN3_GUARD_TIMING.snapshot();
    }
    0
}

/// Check if Qwen3Guard is initialized
///
/// # Returns
/// - 1 if initialized
/// - 0 if not initialized
#[no_mangle]
pub extern "C" fn is_qwen3_guard_initialized() -> i32 {
    if GLOBAL_QWEN3_GUARD.get().is_some() {
        1
    } else {
        0
    }
}
