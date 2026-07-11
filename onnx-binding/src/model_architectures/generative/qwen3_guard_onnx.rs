//! Qwen3Guard generative safety model using ONNX Runtime.
//!
//! This backend intentionally follows the ONNX binding execution-provider
//! behavior used by mmBERT: GPU providers must register successfully before they
//! are considered active, and otherwise inference falls back to CPU.

use crate::core::unified_error::{errors, UnifiedResult};
use crate::model_architectures::classification::ClassifierExecutionProvider;
use half::f16;
use ort::memory::Allocator;
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::tensor::TensorElementType;
use ort::value::{DynValue, Tensor};
use std::borrow::Cow;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

const DEFAULT_MAX_TOKENS: usize = 64;
const DEFAULT_REPEAT_LAST_N: usize = 64;
const DEFAULT_REPEAT_PENALTY: f32 = 1.0;

#[derive(Debug, Clone)]
pub struct Qwen3GuardOnnxConfig {
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for Qwen3GuardOnnxConfig {
    fn default() -> Self {
        Self {
            max_tokens: env_usize("QWEN3_GUARD_ONNX_MAX_TOKENS").unwrap_or(DEFAULT_MAX_TOKENS),
            repeat_penalty: env_f32("QWEN3_GUARD_ONNX_REPEAT_PENALTY")
                .unwrap_or(DEFAULT_REPEAT_PENALTY),
            repeat_last_n: env_usize("QWEN3_GUARD_ONNX_REPEAT_LAST_N")
                .unwrap_or(DEFAULT_REPEAT_LAST_N),
        }
    }
}

pub struct Qwen3GuardOnnxModel {
    session: Session,
    tokenizer: Arc<Tokenizer>,
    config: Qwen3GuardOnnxConfig,
    accepts_attention_mask: bool,
    accepts_position_ids: bool,
    past_ios: Vec<PastKeyValueIo>,
    eos_token_id: u32,
    im_end_token_id: Option<u32>,
    model_path: String,
}

#[derive(Debug, Clone, Copy)]
struct ModelDimensions {
    kv_heads: usize,
    head_dim: usize,
}

impl Default for ModelDimensions {
    fn default() -> Self {
        Self {
            kv_heads: 8,
            head_dim: 128,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PastKind {
    Key,
    Value,
}

#[derive(Debug, Clone)]
struct PastKeyValueIo {
    input_name: String,
    output_name: String,
    layer: usize,
    kind: PastKind,
    dtype: TensorElementType,
    kv_heads: usize,
    head_dim: usize,
}

impl Qwen3GuardOnnxModel {
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
        config: Option<Qwen3GuardOnnxConfig>,
    ) -> UnifiedResult<Self> {
        let model_path_str = model_path.as_ref().display().to_string();
        let model_dir = model_path.as_ref();

        let tokenizer_path = find_file(model_dir, &["tokenizer.json", "onnx/tokenizer.json"])?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        let onnx_candidates = find_onnx_models(model_dir)?;
        let (session, selected_path) =
            create_session_with_fallback(onnx_candidates, provider, &model_path_str)?;
        println!(
            "INFO: Selected Qwen3Guard ONNX file: {}",
            selected_path.display()
        );

        let input_names: HashSet<String> = session.inputs.iter().map(|i| i.name.clone()).collect();
        let model_dims = read_model_dimensions(model_dir).unwrap_or_default();
        let (accepts_attention_mask, accepts_position_ids, past_ios) =
            inspect_session_io(&session, &input_names, model_dims)?;

        let eos_token_id =
            read_eos_token_id(model_dir).or_else(|| tokenizer.token_to_id("<|endoftext|>"));
        let eos_token_id = eos_token_id.ok_or_else(|| {
            errors::config_error(
                "eos_token_id",
                "Qwen3Guard ONNX model must provide eos_token_id in config.json or tokenizer",
            )
        })?;
        let im_end_token_id = tokenizer.token_to_id("<|im_end|>");

        Ok(Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config: config.unwrap_or_default(),
            accepts_attention_mask,
            accepts_position_ids,
            past_ios,
            eos_token_id,
            im_end_token_id,
            model_path: model_path_str,
        })
    }

    pub fn generate_guard(&mut self, text: &str, mode: &str) -> UnifiedResult<String> {
        let prompt = format_guard_prompt(text, mode);
        self.generate(&prompt)
    }

    pub fn model_info(&self) -> String {
        format!(
            "Qwen3GuardOnnxModel(path={}, max_tokens={})",
            self.model_path, self.config.max_tokens
        )
    }

    fn generate(&mut self, prompt: &str) -> UnifiedResult<String> {
        if self.past_ios.is_empty() {
            self.generate_no_past(prompt)
        } else {
            self.generate_with_past(prompt)
        }
    }

    fn generate_no_past(&mut self, prompt: &str) -> UnifiedResult<String> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let mut generated_text = String::new();

        for _ in 0..self.config.max_tokens {
            let logits = self.forward_full_context(&tokens)?;
            let logits = apply_repeat_penalty(
                logits,
                &tokens[tokens.len().saturating_sub(self.config.repeat_last_n)..],
                self.config.repeat_penalty,
            );
            let next_token = argmax(&logits) as u32;

            if next_token == self.eos_token_id || Some(next_token) == self.im_end_token_id {
                break;
            }

            tokens.push(next_token);
            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                generated_text.push_str(&piece);
            }
        }

        Ok(generated_text)
    }

    fn generate_with_past(&mut self, prompt: &str) -> UnifiedResult<String> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            return Err(errors::inference_error(
                "generate",
                "empty prompt token sequence",
            ));
        }

        let mut past = self.create_empty_past()?;
        let mut past_len = 0usize;
        let mut prefill = true;
        let mut generated_text = String::new();

        for _ in 0..self.config.max_tokens {
            let input_tokens: Vec<u32> = if prefill {
                tokens.clone()
            } else {
                vec![*tokens
                    .last()
                    .ok_or_else(|| errors::inference_error("generate", "missing decode token"))?]
            };
            let input_len = input_tokens.len();
            let position_start = past_len;
            let attention_len = past_len + input_len;

            let (logits, next_past) =
                self.forward_with_past(&input_tokens, position_start, attention_len, past)?;
            past = next_past;
            past_len += input_len;
            prefill = false;

            let logits = apply_repeat_penalty(
                logits,
                &tokens[tokens.len().saturating_sub(self.config.repeat_last_n)..],
                self.config.repeat_penalty,
            );
            let next_token = argmax(&logits) as u32;

            if next_token == self.eos_token_id || Some(next_token) == self.im_end_token_id {
                break;
            }

            tokens.push(next_token);
            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                generated_text.push_str(&piece);
            }
        }

        Ok(generated_text)
    }

    fn forward_full_context(&mut self, tokens: &[u32]) -> UnifiedResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(errors::inference_error("forward", "empty token sequence"));
        }

        let seq_len = tokens.len();
        let input_ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))
            .map_err(|e: ort::Error| errors::inference_error("create_input_ids", &e.to_string()))?;

        let outputs = if self.accepts_attention_mask {
            let attention_mask = vec![1i64; seq_len];
            let attention_mask_tensor = Tensor::from_array(([1usize, seq_len], attention_mask))
                .map_err(|e: ort::Error| {
                    errors::inference_error("create_attention_mask", &e.to_string())
                })?;
            self.session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                ])
                .map_err(|e: ort::Error| errors::inference_error("session_run", &e.to_string()))?
        } else {
            self.session
                .run(ort::inputs!["input_ids" => input_ids_tensor])
                .map_err(|e: ort::Error| errors::inference_error("session_run", &e.to_string()))?
        };

        extract_last_token_logits(&outputs)
    }

    fn forward_with_past(
        &mut self,
        tokens: &[u32],
        position_start: usize,
        attention_len: usize,
        past: Vec<DynValue>,
    ) -> UnifiedResult<(Vec<f32>, Vec<DynValue>)> {
        if tokens.is_empty() {
            return Err(errors::inference_error("forward", "empty token sequence"));
        }
        if past.len() != self.past_ios.len() {
            return Err(errors::inference_error(
                "forward",
                &format!(
                    "KV cache count mismatch: expected {}, got {}",
                    self.past_ios.len(),
                    past.len()
                ),
            ));
        }

        let input_ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let input_ids_tensor = Tensor::from_array(([1usize, tokens.len()], input_ids))
            .map_err(|e: ort::Error| errors::inference_error("create_input_ids", &e.to_string()))?;

        let mut inputs: Vec<(Cow<'static, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(3 + self.past_ios.len());
        inputs.push((
            Cow::Borrowed("input_ids"),
            SessionInputValue::from(input_ids_tensor),
        ));

        if self.accepts_attention_mask {
            let attention_mask = vec![1i64; attention_len];
            let attention_mask_tensor =
                Tensor::from_array(([1usize, attention_len], attention_mask)).map_err(
                    |e: ort::Error| {
                        errors::inference_error("create_attention_mask", &e.to_string())
                    },
                )?;
            inputs.push((
                Cow::Borrowed("attention_mask"),
                SessionInputValue::from(attention_mask_tensor),
            ));
        }

        if self.accepts_position_ids {
            let position_ids: Vec<i64> = (position_start..position_start + tokens.len())
                .map(|pos| pos as i64)
                .collect();
            let position_ids_tensor = Tensor::from_array(([1usize, tokens.len()], position_ids))
                .map_err(|e: ort::Error| {
                    errors::inference_error("create_position_ids", &e.to_string())
                })?;
            inputs.push((
                Cow::Borrowed("position_ids"),
                SessionInputValue::from(position_ids_tensor),
            ));
        }

        for (io, value) in self.past_ios.iter().zip(past.into_iter()) {
            inputs.push((
                Cow::Owned(io.input_name.clone()),
                SessionInputValue::from(value),
            ));
        }

        let past_ios = self.past_ios.clone();
        let mut outputs = self
            .session
            .run(inputs)
            .map_err(|e: ort::Error| errors::inference_error("session_run", &e.to_string()))?;
        let logits = extract_last_token_logits(&outputs)?;
        let next_past = collect_present_past(&past_ios, &mut outputs)?;
        Ok((logits, next_past))
    }

    fn create_empty_past(&self) -> UnifiedResult<Vec<DynValue>> {
        self.past_ios
            .iter()
            .map(|io| create_empty_past_value(io, self.session.allocator()))
            .collect::<UnifiedResult<Vec<_>>>()
    }
}

fn collect_present_past(
    past_ios: &[PastKeyValueIo],
    outputs: &mut SessionOutputs<'_>,
) -> UnifiedResult<Vec<DynValue>> {
    past_ios
        .iter()
        .map(|io| {
            outputs.remove(&io.output_name).ok_or_else(|| {
                errors::inference_error(
                    "collect_past",
                    &format!(
                        "missing ONNX output '{}' for KV cache input '{}'",
                        io.output_name, io.input_name
                    ),
                )
            })
        })
        .collect()
}

fn create_session_with_fallback(
    onnx_candidates: Vec<PathBuf>,
    provider: ClassifierExecutionProvider,
    model_path: &str,
) -> UnifiedResult<(Session, PathBuf)> {
    let mut last_error: Option<String> = None;
    for onnx_path in onnx_candidates {
        match create_session(&onnx_path, provider) {
            Ok(session) => return Ok((session, onnx_path)),
            Err(e) => {
                let reason = format!("{:?}", e);
                println!(
                    "WARN: Failed to initialize Qwen3Guard ONNX session from {}: {}",
                    onnx_path.display(),
                    reason
                );
                last_error = Some(format!("{}: {}", onnx_path.display(), reason));
            }
        }
    }
    let detail = last_error.unwrap_or_else(|| "no ONNX candidate was loadable".to_string());
    Err(errors::model_load(model_path, &detail))
}

fn create_session<P: AsRef<Path>>(
    onnx_path: P,
    provider: ClassifierExecutionProvider,
) -> UnifiedResult<Session> {
    let onnx_path_str = onnx_path.as_ref().display().to_string();

    match provider {
        ClassifierExecutionProvider::Cpu => create_cpu_session(onnx_path, &onnx_path_str),
        ClassifierExecutionProvider::Rocm | ClassifierExecutionProvider::Auto => {
            #[cfg(feature = "rocm")]
            {
                use crate::core::gpu_memory;
                use ort::execution_providers::{
                    ArenaExtendStrategy, MIGraphXExecutionProvider, ROCmExecutionProvider,
                };

                match Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .with_execution_providers([MIGraphXExecutionProvider::default()
                        .build()
                        .error_on_failure()])
                    .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                {
                    Ok(session) => {
                        println!(
                            "INFO: Using MIGraphX execution provider (AMD GPU) for Qwen3Guard — verified"
                        );
                        return Ok(session);
                    }
                    Err(e) => {
                        println!("INFO: MIGraphX EP failed to register for Qwen3Guard: {}", e)
                    }
                }

                let mem_limit = gpu_memory::get_gpu_mem_limit();
                match Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .with_execution_providers([ROCmExecutionProvider::default()
                        .with_mem_limit(mem_limit)
                        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                        .build()
                        .error_on_failure()])
                    .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                {
                    Ok(session) => {
                        println!(
                            "INFO: Using ROCm execution provider (AMD GPU) for Qwen3Guard — verified"
                        );
                        return Ok(session);
                    }
                    Err(e) => println!("INFO: ROCm EP failed to register for Qwen3Guard: {}", e),
                }

                println!("WARNING: All AMD GPU execution providers failed for Qwen3Guard, falling back to CPU");
            }

            #[cfg(not(feature = "rocm"))]
            {
                if matches!(provider, ClassifierExecutionProvider::Rocm) {
                    println!(
                        "WARNING: ROCm requested for Qwen3Guard but 'rocm' feature not enabled, using CPU"
                    );
                }
            }

            #[cfg(feature = "cuda")]
            {
                if matches!(provider, ClassifierExecutionProvider::Auto) {
                    use crate::core::gpu_memory;
                    use ort::execution_providers::{
                        ArenaExtendStrategy as CudaArenaStrategy, CUDAExecutionProvider,
                    };
                    let mem_limit = gpu_memory::get_gpu_mem_limit();
                    match Session::builder()
                        .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                        .with_execution_providers([CUDAExecutionProvider::default()
                            .with_memory_limit(mem_limit)
                            .with_arena_extend_strategy(CudaArenaStrategy::SameAsRequested)
                            .build()
                            .error_on_failure()])
                        .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                    {
                        Ok(session) => {
                            println!(
                                "INFO: Using CUDA execution provider (NVIDIA GPU) for Qwen3Guard — verified"
                            );
                            return Ok(session);
                        }
                        Err(e) => {
                            println!(
                                "WARNING: CUDA EP failed for Qwen3Guard: {}, falling back to CPU",
                                e
                            );
                        }
                    }
                }
            }

            create_cpu_session(onnx_path, &onnx_path_str)
        }
        ClassifierExecutionProvider::Cuda => {
            #[cfg(feature = "cuda")]
            {
                use crate::core::gpu_memory;
                use ort::execution_providers::{
                    ArenaExtendStrategy as CudaArenaStrategy, CUDAExecutionProvider,
                };
                let mem_limit = gpu_memory::get_gpu_mem_limit();
                match Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .with_execution_providers([CUDAExecutionProvider::default()
                        .with_memory_limit(mem_limit)
                        .with_arena_extend_strategy(CudaArenaStrategy::SameAsRequested)
                        .build()
                        .error_on_failure()])
                    .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                {
                    Ok(session) => {
                        println!(
                            "INFO: Using CUDA execution provider (NVIDIA GPU) for Qwen3Guard — verified"
                        );
                        return Ok(session);
                    }
                    Err(e) => println!("WARNING: CUDA EP failed for Qwen3Guard: {}", e),
                }
            }

            #[cfg(not(feature = "cuda"))]
            println!("WARNING: CUDA requested for Qwen3Guard but 'cuda' feature not enabled");

            create_cpu_session(onnx_path, &onnx_path_str)
        }
        ClassifierExecutionProvider::OpenVino => {
            #[cfg(feature = "openvino")]
            {
                use ort::execution_providers::OpenVINOExecutionProvider;
                match Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .with_execution_providers([OpenVINOExecutionProvider::default()
                        .build()
                        .error_on_failure()])
                    .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                {
                    Ok(session) => {
                        println!(
                            "INFO: Using OpenVINO execution provider (Intel) for Qwen3Guard — verified"
                        );
                        return Ok(session);
                    }
                    Err(e) => println!("WARNING: OpenVINO EP failed for Qwen3Guard: {}", e),
                }
            }

            #[cfg(not(feature = "openvino"))]
            println!(
                "WARNING: OpenVINO requested for Qwen3Guard but 'openvino' feature not enabled"
            );

            create_cpu_session(onnx_path, &onnx_path_str)
        }
    }
}

fn create_cpu_session<P: AsRef<Path>>(onnx_path: P, onnx_path_str: &str) -> UnifiedResult<Session> {
    println!("INFO: Using CPU execution provider for Qwen3Guard");
    Session::builder()
        .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
        .commit_from_file(onnx_path.as_ref())
        .map_err(|e: ort::Error| errors::model_load(onnx_path_str, &e.to_string()))
}

fn find_onnx_models(model_dir: &Path) -> UnifiedResult<Vec<PathBuf>> {
    let candidates = [
        "model.onnx",
        "model_quantized.onnx",
        "decoder_model.onnx",
        "decoder_model_merged.onnx",
        "onnx/model.onnx",
        "onnx/model_quantized.onnx",
        "onnx/decoder_model.onnx",
        "onnx/decoder_model_merged.onnx",
    ];

    let mut results = Vec::new();
    for rel in candidates {
        let path = model_dir.join(rel);
        if path.exists() && !results.iter().any(|p| p == &path) {
            results.push(path);
        }
    }

    for base in [model_dir.to_path_buf(), model_dir.join("onnx")] {
        if let Ok(entries) = std::fs::read_dir(base) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "onnx")
                    && !results.iter().any(|p| p == &path)
                {
                    results.push(path);
                }
            }
        }
    }

    if results.is_empty() {
        return Err(errors::file_not_found(&format!(
            "No Qwen3Guard ONNX model found under {}",
            model_dir.display()
        )));
    }
    Ok(results)
}

fn find_file(model_dir: &Path, candidates: &[&str]) -> UnifiedResult<PathBuf> {
    candidates
        .iter()
        .map(|rel| model_dir.join(rel))
        .find(|path| path.exists())
        .ok_or_else(|| {
            errors::file_not_found(&format!(
                "none of {:?} found under {}",
                candidates,
                model_dir.display()
            ))
        })
}

fn inspect_session_io(
    session: &Session,
    input_names: &HashSet<String>,
    model_dims: ModelDimensions,
) -> UnifiedResult<(bool, bool, Vec<PastKeyValueIo>)> {
    if !input_names.contains("input_ids") {
        return Err(errors::config_error(
            "input_ids",
            "Qwen3Guard ONNX model must expose an input_ids input",
        ));
    }

    let unsupported: Vec<&str> = input_names
        .iter()
        .map(String::as_str)
        .filter(|name| {
            *name != "input_ids"
                && *name != "attention_mask"
                && *name != "position_ids"
                && parse_past_key_value_name(name).is_none()
        })
        .collect();
    if !unsupported.is_empty() {
        return Err(errors::config_error(
            "onnx_inputs",
            &format!(
                "unsupported Qwen3Guard ONNX inputs {:?}; supported inputs are input_ids, attention_mask, position_ids, and past_key_values.*",
                unsupported
            ),
        ));
    }

    let output_names: HashSet<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
    let mut past_ios = Vec::new();

    for input in &session.inputs {
        let Some((layer, kind)) = parse_past_key_value_name(&input.name) else {
            continue;
        };
        let dtype = input.input_type.tensor_type().ok_or_else(|| {
            errors::config_error(
                "past_key_values",
                &format!("{} must be a tensor input", input.name),
            )
        })?;
        if dtype != TensorElementType::Float16 && dtype != TensorElementType::Float32 {
            return Err(errors::config_error(
                "past_key_values",
                &format!(
                    "{} has unsupported dtype {}; expected Float16 or Float32",
                    input.name, dtype
                ),
            ));
        }

        let (kv_heads, head_dim) = infer_kv_dimensions(input.input_type.tensor_shape(), model_dims);
        let output_name =
            find_present_output_name(&output_names, layer, kind).ok_or_else(|| {
                errors::config_error(
                    "past_key_values",
                    &format!(
                        "could not find present KV output for ONNX input '{}'",
                        input.name
                    ),
                )
            })?;

        past_ios.push(PastKeyValueIo {
            input_name: input.name.clone(),
            output_name,
            layer,
            kind,
            dtype,
            kv_heads,
            head_dim,
        });
    }

    past_ios.sort_by_key(|io| (io.layer, io.kind));

    Ok((
        input_names.contains("attention_mask"),
        input_names.contains("position_ids"),
        past_ios,
    ))
}

fn parse_past_key_value_name(name: &str) -> Option<(usize, PastKind)> {
    let rest = name.strip_prefix("past_key_values.")?;
    let mut parts = rest.split('.');
    let layer = parts.next()?.parse::<usize>().ok()?;
    let kind = match parts.next()? {
        "key" => PastKind::Key,
        "value" => PastKind::Value,
        _ => return None,
    };
    if parts.next().is_some() {
        return None;
    }
    Some((layer, kind))
}

fn find_present_output_name(
    output_names: &HashSet<String>,
    layer: usize,
    kind: PastKind,
) -> Option<String> {
    let kind = match kind {
        PastKind::Key => "key",
        PastKind::Value => "value",
    };
    [
        format!("present.{layer}.{kind}"),
        format!("present_key_values.{layer}.{kind}"),
        format!("present.{layer}.decoder.{kind}"),
        format!("present_key_values.{layer}.decoder.{kind}"),
    ]
    .into_iter()
    .find(|name| output_names.contains(name))
}

fn infer_kv_dimensions(
    shape: Option<&ort::tensor::Shape>,
    fallback: ModelDimensions,
) -> (usize, usize) {
    let Some(shape) = shape else {
        return (fallback.kv_heads, fallback.head_dim);
    };
    let dims = shape.as_ref();
    match dims {
        [_, heads, _, head_dim] if *heads > 0 && *head_dim > 0 => {
            (*heads as usize, *head_dim as usize)
        }
        _ => (fallback.kv_heads, fallback.head_dim),
    }
}

fn create_empty_past_value(io: &PastKeyValueIo, allocator: &Allocator) -> UnifiedResult<DynValue> {
    let shape = [1usize, io.kv_heads, 0usize, io.head_dim];
    match io.dtype {
        TensorElementType::Float16 => {
            let tensor = Tensor::<f16>::new(allocator, shape).map_err(|e: ort::Error| {
                errors::inference_error("create_empty_past_f16", &e.to_string())
            })?;
            Ok(DynValue::from(tensor))
        }
        TensorElementType::Float32 => {
            let tensor = Tensor::<f32>::new(allocator, shape).map_err(|e: ort::Error| {
                errors::inference_error("create_empty_past_f32", &e.to_string())
            })?;
            Ok(DynValue::from(tensor))
        }
        other => Err(errors::config_error(
            "past_key_values",
            &format!("unsupported KV cache dtype {other}"),
        )),
    }
}

fn read_eos_token_id(model_dir: &Path) -> Option<u32> {
    let config_path = [
        model_dir.join("config.json"),
        model_dir.join("onnx/config.json"),
    ]
    .into_iter()
    .find(|p| p.exists())?;
    let config = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&config).ok()?;
    match json.get("eos_token_id")? {
        serde_json::Value::Number(n) => n.as_u64().map(|id| id as u32),
        serde_json::Value::Array(values) => {
            values.iter().find_map(|v| v.as_u64().map(|id| id as u32))
        }
        _ => None,
    }
}

fn read_model_dimensions(model_dir: &Path) -> Option<ModelDimensions> {
    let config_path = [
        model_dir.join("config.json"),
        model_dir.join("onnx/config.json"),
    ]
    .into_iter()
    .find(|p| p.exists())?;
    let config = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&config).ok()?;

    let kv_heads = json
        .get("num_key_value_heads")
        .or_else(|| json.get("num_attention_heads"))?
        .as_u64()? as usize;
    let head_dim = json.get("head_dim").and_then(|v| v.as_u64()).or_else(|| {
        let hidden = json.get("hidden_size")?.as_u64()?;
        let heads = json.get("num_attention_heads")?.as_u64()?;
        Some(hidden / heads)
    })? as usize;

    Some(ModelDimensions { kv_heads, head_dim })
}

fn extract_last_token_logits(outputs: &SessionOutputs<'_>) -> UnifiedResult<Vec<f32>> {
    if let Some(output) = outputs.get("logits") {
        if let Ok(logits) = logits_from_value_f32(output) {
            return Ok(logits);
        }
        if let Ok(logits) = logits_from_value_f16(output) {
            return Ok(logits);
        }
    }

    for (_name, output) in outputs {
        if let Ok(logits) = logits_from_value_f32(&output) {
            return Ok(logits);
        }
        if let Ok(logits) = logits_from_value_f16(&output) {
            return Ok(logits);
        }
    }

    Err(errors::inference_error(
        "extract_logits",
        "no f32/f16 logits tensor found in ONNX outputs",
    ))
}

fn logits_from_value_f32(output: &ort::value::DynValue) -> UnifiedResult<Vec<f32>> {
    let (shape, data) = output
        .try_extract_tensor::<f32>()
        .map_err(|e: ort::Error| errors::inference_error("extract_logits_f32", &e.to_string()))?;
    last_token_logits_from_slice(shape.as_ref(), data)
}

fn logits_from_value_f16(output: &ort::value::DynValue) -> UnifiedResult<Vec<f32>> {
    let (shape, data) = output
        .try_extract_tensor::<f16>()
        .map_err(|e: ort::Error| errors::inference_error("extract_logits_f16", &e.to_string()))?;
    let converted: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    last_token_logits_from_slice(shape.as_ref(), &converted)
}

fn last_token_logits_from_slice(shape: &[i64], data: &[f32]) -> UnifiedResult<Vec<f32>> {
    match shape {
        [1, vocab] => Ok(data[..*vocab as usize].to_vec()),
        [1, seq_len, vocab] => {
            let seq_len = *seq_len as usize;
            let vocab = *vocab as usize;
            if seq_len == 0 || vocab == 0 || data.len() < seq_len * vocab {
                return Err(errors::inference_error(
                    "extract_logits",
                    &format!("invalid logits shape {:?}", shape),
                ));
            }
            let start = (seq_len - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }
        _ => Err(errors::inference_error(
            "extract_logits",
            &format!("unsupported logits shape {:?}", shape),
        )),
    }
}

fn apply_repeat_penalty(mut logits: Vec<f32>, recent_tokens: &[u32], penalty: f32) -> Vec<f32> {
    if penalty == 1.0 {
        return logits;
    }
    for &token in recent_tokens {
        if let Some(logit) = logits.get_mut(token as usize) {
            if *logit < 0.0 {
                *logit *= penalty;
            } else {
                *logit /= penalty;
            }
        }
    }
    logits
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn format_guard_prompt(text: &str, mode: &str) -> String {
    let user_label = if mode == "output" {
        "ASSISTANT"
    } else {
        "USER"
    };

    format!(
        "<|im_start|>user\n# Task:\n\
        1) Evaluate whether THE LAST {}'s query belongs to safe or unsafe content based on our safety policy.\n\
        2) Identify the category of unsafe content found.\n\n\
        <BEGIN SAFETY POLICY>\n\
        **Safety: Safe**: The {}'s query does not contain any prohibited content or content that could lead to prohibited content.\n\
        **Safety: Unsafe**: The {}'s query contains prohibited content.\n\
        **Safety: Controversial**: The {}'s query does not explicitly contain prohibited content, but its intent, context, or potential responses could be misinterpreted, misused, or exploited to generate unsafe content under certain conditions.\n\
        <END SAFETY POLICY>\n\n\
        <BEGIN UNSAFE CONTENT CATEGORIES>\n\
        Violent.\n\
        Non-violent Illegal Acts.\n\
        Sexual Content or Sexual Acts.\n\
        PII.\n\
        Suicide & Self-Harm.\n\
        Unethical Acts.\n\
        Politically Sensitive Topics.\n\
        Copyright Violation.\n\
        Jailbreak.\n\
        <END UNSAFE CONTENT CATEGORIES>\n\n\
        <BEGIN CONVERSATION>\n\n\
        {}: {}\n\n\
        <END CONVERSATION>\n\n\
        Provide your safety assessment for ONLY THE LAST **{}'s query** in the above conversation:\n \
        - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n \
        - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>\n\
        <|im_start|>assistant\n\
        <think>\n\n\
        </think>\n\n",
        user_label, user_label, user_label, user_label, user_label, text, user_label
    )
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_uses_assistant_label_for_output_mode() {
        let prompt = format_guard_prompt("hello", "output");
        assert!(prompt.contains("ASSISTANT: hello"));
        assert!(prompt.contains("LAST **ASSISTANT's query**"));
    }

    #[test]
    fn prompt_uses_user_label_for_input_mode() {
        let prompt = format_guard_prompt("hello", "input");
        assert!(prompt.contains("USER: hello"));
        assert!(prompt.contains("LAST **USER's query**"));
    }

    #[test]
    fn argmax_handles_negative_values() {
        assert_eq!(argmax(&[-3.0, -0.5, -1.0]), 1);
    }

    #[test]
    fn parses_past_key_value_names() {
        assert_eq!(
            parse_past_key_value_name("past_key_values.12.key"),
            Some((12, PastKind::Key))
        );
        assert_eq!(
            parse_past_key_value_name("past_key_values.0.value"),
            Some((0, PastKind::Value))
        );
        assert_eq!(parse_past_key_value_name("attention_mask"), None);
        assert_eq!(parse_past_key_value_name("past_key_values.x.key"), None);
    }
}
