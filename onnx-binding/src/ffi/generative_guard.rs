//! FFI bindings for Qwen3Guard ONNX safety classification.

use crate::core::unified_error::{errors, UnifiedResult};
use crate::model_architectures::classification::ClassifierExecutionProvider;
use crate::model_architectures::generative::Qwen3GuardOnnxModel;
use parking_lot::Mutex;
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

static QWEN3_GUARD_ONNX: OnceLock<Mutex<Qwen3GuardOnnxModel>> = OnceLock::new();

#[repr(C)]
pub struct GuardResultFFI {
    pub raw_output: *mut c_char,
    pub error: bool,
    pub error_message: *mut c_char,
    pub lock_wait_ns: u64,
    pub generation_ns: u64,
    pub total_ns: u64,
}

impl Default for GuardResultFFI {
    fn default() -> Self {
        Self {
            raw_output: ptr::null_mut(),
            error: true,
            error_message: ptr::null_mut(),
            lock_wait_ns: 0,
            generation_ns: 0,
            total_ns: 0,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct GuardTiming {
    lock_wait_ns: u64,
    generation_ns: u64,
    total_ns: u64,
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn error_cstring(message: &str) -> *mut c_char {
    CString::new(message.replace('\0', " "))
        .unwrap_or_else(|_| CString::new("unknown ffi error").unwrap())
        .into_raw()
}

unsafe fn write_guard_result(
    result: *mut GuardResultFFI,
    output: UnifiedResult<String>,
    timing: GuardTiming,
) -> i32 {
    if result.is_null() {
        return -1;
    }

    match output {
        Ok(raw_output) => match CString::new(raw_output) {
            Ok(raw_output) => {
                unsafe {
                    *result = GuardResultFFI {
                        raw_output: raw_output.into_raw(),
                        error: false,
                        error_message: ptr::null_mut(),
                        lock_wait_ns: timing.lock_wait_ns,
                        generation_ns: timing.generation_ns,
                        total_ns: timing.total_ns,
                    };
                }
                0
            }
            Err(e) => {
                unsafe {
                    *result = GuardResultFFI::default();
                    (*result).error_message =
                        error_cstring(&format!("failed to create C string: {}", e));
                    (*result).lock_wait_ns = timing.lock_wait_ns;
                    (*result).generation_ns = timing.generation_ns;
                    (*result).total_ns = timing.total_ns;
                }
                -1
            }
        },
        Err(e) => {
            unsafe {
                *result = GuardResultFFI::default();
                (*result).error_message = error_cstring(&format!("guard generation failed: {}", e));
                (*result).lock_wait_ns = timing.lock_wait_ns;
                (*result).generation_ns = timing.generation_ns;
                (*result).total_ns = timing.total_ns;
            }
            -1
        }
    }
}

fn provider_from_flags(use_gpu: bool, provider_hint: &str) -> ClassifierExecutionProvider {
    if !use_gpu {
        return ClassifierExecutionProvider::Cpu;
    }

    match provider_hint.trim().to_ascii_lowercase().as_str() {
        "rocm" | "migraphx" | "amd" => ClassifierExecutionProvider::Rocm,
        "cuda" | "nvidia" => ClassifierExecutionProvider::Cuda,
        "openvino" => ClassifierExecutionProvider::OpenVino,
        _ => ClassifierExecutionProvider::Auto,
    }
}

#[no_mangle]
pub extern "C" fn init_qwen3_guard_onnx(
    model_path: *const c_char,
    use_gpu: bool,
    provider_hint: *const c_char,
) -> bool {
    if model_path.is_null() {
        eprintln!("Error: null model_path passed to init_qwen3_guard_onnx");
        return false;
    }

    if QWEN3_GUARD_ONNX.get().is_some() {
        println!("INFO: Qwen3Guard ONNX already initialized, reusing existing instance");
        return true;
    }

    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => {
                eprintln!("Error: invalid model_path in init_qwen3_guard_onnx: {}", e);
                return false;
            }
        }
    };

    let provider_hint = if provider_hint.is_null() {
        String::new()
    } else {
        unsafe {
            CStr::from_ptr(provider_hint)
                .to_str()
                .unwrap_or_default()
                .to_string()
        }
    };
    let provider = provider_from_flags(use_gpu, &provider_hint);

    match Qwen3GuardOnnxModel::load(&model_path, provider, None) {
        Ok(model) => {
            println!("INFO: Loaded {}", model.model_info());
            match QWEN3_GUARD_ONNX.set(Mutex::new(model)) {
                Ok(()) => true,
                Err(_) => {
                    println!("INFO: Qwen3Guard ONNX already initialized by another thread");
                    true
                }
            }
        }
        Err(e) => {
            eprintln!("ERROR: Failed to load Qwen3Guard ONNX model: {}", e);
            false
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn classify_with_qwen3_guard_onnx(
    text: *const c_char,
    mode: *const c_char,
    result: *mut GuardResultFFI,
) -> i32 {
    let total_start = Instant::now();

    if text.is_null() || mode.is_null() || result.is_null() {
        return unsafe {
            write_guard_result(
                result,
                Err(errors::validation_error(
                    "ffi_args",
                    "non-null text, mode, and result pointers",
                    "null pointer",
                )),
                GuardTiming {
                    total_ns: duration_ns(total_start.elapsed()),
                    ..GuardTiming::default()
                },
            )
        };
    }

    let text = match unsafe { CStr::from_ptr(text).to_str() } {
        Ok(s) => s,
        Err(e) => {
            return unsafe {
                write_guard_result(
                    result,
                    Err(errors::tokenization_error(&format!(
                        "invalid text utf8: {}",
                        e
                    ))),
                    GuardTiming {
                        total_ns: duration_ns(total_start.elapsed()),
                        ..GuardTiming::default()
                    },
                )
            }
        }
    };
    let mode = match unsafe { CStr::from_ptr(mode).to_str() } {
        Ok(s) => s,
        Err(e) => {
            return unsafe {
                write_guard_result(
                    result,
                    Err(errors::config_error(
                        "mode",
                        &format!("invalid utf8: {}", e),
                    )),
                    GuardTiming {
                        total_ns: duration_ns(total_start.elapsed()),
                        ..GuardTiming::default()
                    },
                )
            }
        }
    };

    let guard = match QWEN3_GUARD_ONNX.get() {
        Some(guard) => guard,
        None => {
            return unsafe {
                write_guard_result(
                    result,
                    Err(errors::model_load(
                        "qwen3_guard_onnx",
                        "model is not initialized",
                    )),
                    GuardTiming {
                        total_ns: duration_ns(total_start.elapsed()),
                        ..GuardTiming::default()
                    },
                )
            }
        }
    };

    let lock_start = Instant::now();
    let mut guard = guard.lock();
    let lock_wait_ns = duration_ns(lock_start.elapsed());
    let generation_start = Instant::now();
    let output = guard.generate_guard(text, mode);
    let generation_ns = duration_ns(generation_start.elapsed());
    let total_ns = duration_ns(total_start.elapsed());

    unsafe {
        write_guard_result(
            result,
            output,
            GuardTiming {
                lock_wait_ns,
                generation_ns,
                total_ns,
            },
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn free_qwen3_guard_onnx_result(result: *mut GuardResultFFI) {
    if result.is_null() {
        return;
    }

    unsafe {
        if !(*result).raw_output.is_null() {
            let _ = CString::from_raw((*result).raw_output);
            (*result).raw_output = ptr::null_mut();
        }
        if !(*result).error_message.is_null() {
            let _ = CString::from_raw((*result).error_message);
            (*result).error_message = ptr::null_mut();
        }
    }
}

#[no_mangle]
pub extern "C" fn is_qwen3_guard_onnx_initialized() -> bool {
    QWEN3_GUARD_ONNX.get().is_some()
}
