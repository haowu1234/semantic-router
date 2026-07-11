//! FFI bindings for Qwen3Guard safety classification.

use crate::core::{UnifiedError, UnifiedResult};
use crate::model_architectures::generative::{GuardGenerationResult, Qwen3GuardModel};
use candle_core::Device;
use std::env;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{
    mpsc::{channel, Receiver, Sender},
    Mutex, OnceLock,
};
use std::thread;
use std::time::{Duration, Instant};

/// Global Qwen3Guard instance (for safety/jailbreak detection)
static GLOBAL_QWEN3_GUARD: OnceLock<Mutex<Qwen3GuardModel>> = OnceLock::new();
static GLOBAL_QWEN3_GUARD_SCHEDULER: OnceLock<Qwen3GuardDecodeScheduler> = OnceLock::new();

static QWEN3_GUARD_TIMING: Qwen3GuardTimingAccumulator = Qwen3GuardTimingAccumulator::new();
static QWEN3_GUARD_DEVICE_KIND: AtomicU64 = AtomicU64::new(QWEN3_GUARD_DEVICE_UNINITIALIZED);

const QWEN3_GUARD_DEVICE_UNINITIALIZED: u64 = 0;
const QWEN3_GUARD_DEVICE_CPU: u64 = 1;
const QWEN3_GUARD_DEVICE_CUDA: u64 = 2;
const QWEN3_GUARD_DEVICE_METAL: u64 = 3;

struct Qwen3GuardDecodeScheduler {
    request_tx: Sender<GuardSchedulerRequest>,
}

struct GuardSchedulerRequest {
    text: String,
    mode: String,
    response_tx: Sender<UnifiedResult<GuardGenerationResult>>,
}

#[derive(Clone)]
struct Qwen3GuardSchedulerConfig {
    max_batch_size: usize,
    batch_timeout: Duration,
    verbose: bool,
}

impl Qwen3GuardDecodeScheduler {
    fn new(model: Qwen3GuardModel, config: Qwen3GuardSchedulerConfig) -> Self {
        let (request_tx, request_rx) = channel();
        thread::spawn(move || {
            qwen3_guard_scheduler_loop(model, request_rx, config);
        });
        Self { request_tx }
    }

    fn classify(&self, text: String, mode: String) -> UnifiedResult<GuardGenerationResult> {
        let (response_tx, response_rx) = channel();
        self.request_tx
            .send(GuardSchedulerRequest {
                text,
                mode,
                response_tx,
            })
            .map_err(|_| processing_error("submit guard request", "scheduler stopped"))?;
        response_rx
            .recv()
            .map_err(|_| processing_error("receive guard result", "scheduler dropped response"))?
    }
}

fn qwen3_guard_scheduler_loop(
    mut model: Qwen3GuardModel,
    request_rx: Receiver<GuardSchedulerRequest>,
    config: Qwen3GuardSchedulerConfig,
) {
    while let Ok(first) = request_rx.recv() {
        let mut batch = vec![first];
        let deadline = Instant::now() + config.batch_timeout;

        while batch.len() < config.max_batch_size {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            match request_rx.recv_timeout(remaining) {
                Ok(request) => batch.push(request),
                Err(_) => break,
            }
        }

        if config.verbose {
            println!(
                "Qwen3Guard decode scheduler processing batch={}",
                batch.len()
            );
        }

        let inputs: Vec<(String, String)> = batch
            .iter()
            .map(|request| (request.text.clone(), request.mode.clone()))
            .collect();
        let results = model.generate_guard_micro_batch(&inputs);

        for (request, result) in batch.into_iter().zip(results.into_iter()) {
            let _ = request.response_tx.send(result);
        }
    }
}

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

fn qwen3_guard_device_kind(device: &Device) -> u64 {
    match device {
        Device::Cpu => QWEN3_GUARD_DEVICE_CPU,
        Device::Cuda(_) => QWEN3_GUARD_DEVICE_CUDA,
        Device::Metal(_) => QWEN3_GUARD_DEVICE_METAL,
    }
}

fn processing_error(operation: &str, source: impl ToString) -> UnifiedError {
    UnifiedError::Processing {
        operation: operation.to_string(),
        source: source.to_string(),
        input_context: None,
    }
}

fn qwen3_guard_scheduler_enabled() -> bool {
    env::var("QWEN3_GUARD_DECODE_SCHEDULER")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn qwen3_guard_scheduler_config() -> Qwen3GuardSchedulerConfig {
    let max_batch_size = env::var("QWEN3_GUARD_DECODE_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(4);
    let batch_timeout_ms = env::var("QWEN3_GUARD_DECODE_BATCH_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(5);
    let verbose = env::var("QWEN3_GUARD_DECODE_SCHEDULER_VERBOSE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);

    Qwen3GuardSchedulerConfig {
        max_batch_size,
        batch_timeout: Duration::from_millis(batch_timeout_ms),
        verbose,
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

unsafe fn write_guard_generation_result(
    result: *mut GuardResult,
    generation_result: UnifiedResult<GuardGenerationResult>,
) -> i32 {
    match generation_result {
        Ok(guard_result) => {
            let raw_output_c = match CString::new(guard_result.raw_output.as_str()) {
                Ok(s) => s.into_raw(),
                Err(e) => {
                    eprintln!("Error: failed to create raw_output C string: {}", e);
                    unsafe {
                        (*result) = GuardResult::default();
                        (*result).error_message =
                            create_error_message(&format!("Failed to create C string: {}", e));
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

    if GLOBAL_QWEN3_GUARD.get().is_some() || GLOBAL_QWEN3_GUARD_SCHEDULER.get().is_some() {
        println!("Qwen3Guard already initialized, reusing existing instance");
        return 0;
    }

    match Qwen3GuardModel::new(model_path_str, &device, None) {
        Ok(guard) => {
            QWEN3_GUARD_DEVICE_KIND.store(qwen3_guard_device_kind(&device), Ordering::Relaxed);

            if qwen3_guard_scheduler_enabled() {
                let config = qwen3_guard_scheduler_config();
                if config.verbose {
                    println!(
                        "Qwen3Guard decode scheduler enabled: max_batch_size={}, timeout={:?}",
                        config.max_batch_size, config.batch_timeout
                    );
                }
                match GLOBAL_QWEN3_GUARD_SCHEDULER
                    .set(Qwen3GuardDecodeScheduler::new(guard, config))
                {
                    Ok(_) => {
                        println!(
                            "Qwen3Guard initialized with decode scheduler: {}",
                            model_path_str
                        );
                        0
                    }
                    Err(_) => {
                        println!("Qwen3Guard already initialized (race condition), reusing");
                        0
                    }
                }
            } else {
                match GLOBAL_QWEN3_GUARD.set(Mutex::new(guard)) {
                    Ok(_) => {
                        println!("Qwen3Guard initialized: {}", model_path_str);
                        0
                    }
                    Err(_) => {
                        println!("Qwen3Guard already initialized (race condition), reusing");
                        0
                    }
                }
            }
        }
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

    if let Some(scheduler) = GLOBAL_QWEN3_GUARD_SCHEDULER.get() {
        let generation_start = Instant::now();
        let generation_result = scheduler.classify(text_str.to_string(), mode_str.to_string());
        let generation_elapsed = generation_start.elapsed();
        QWEN3_GUARD_TIMING.record(
            Duration::ZERO,
            generation_elapsed,
            generation_result.is_err(),
        );
        return unsafe { write_guard_generation_result(result, generation_result) };
    }

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

            unsafe { write_guard_generation_result(result, generation_result) }
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

#[no_mangle]
pub extern "C" fn get_qwen3_guard_device_kind() -> u64 {
    QWEN3_GUARD_DEVICE_KIND.load(Ordering::Relaxed)
}

/// Check if Qwen3Guard is initialized
///
/// # Returns
/// - 1 if initialized
/// - 0 if not initialized
#[no_mangle]
pub extern "C" fn is_qwen3_guard_initialized() -> i32 {
    if GLOBAL_QWEN3_GUARD.get().is_some() || GLOBAL_QWEN3_GUARD_SCHEDULER.get().is_some() {
        1
    } else {
        0
    }
}
