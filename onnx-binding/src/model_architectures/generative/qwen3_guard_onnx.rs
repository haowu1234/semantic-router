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
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const DEFAULT_MAX_TOKENS: usize = 64;
const DEFAULT_REPEAT_LAST_N: usize = 64;
const DEFAULT_REPEAT_PENALTY: f32 = 1.0;

#[derive(Debug, Clone)]
pub struct Qwen3GuardOnnxConfig {
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub prefix_cache_enabled: bool,
}

impl Default for Qwen3GuardOnnxConfig {
    fn default() -> Self {
        Self {
            max_tokens: env_usize("QWEN3_GUARD_ONNX_MAX_TOKENS").unwrap_or(DEFAULT_MAX_TOKENS),
            repeat_penalty: env_f32("QWEN3_GUARD_ONNX_REPEAT_PENALTY")
                .unwrap_or(DEFAULT_REPEAT_PENALTY),
            repeat_last_n: env_usize("QWEN3_GUARD_ONNX_REPEAT_LAST_N")
                .unwrap_or(DEFAULT_REPEAT_LAST_N),
            prefix_cache_enabled: env_bool("QWEN3_GUARD_ONNX_PREFIX_CACHE").unwrap_or(true),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3GuardOnnxProfile {
    pub output: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub tokenize_ns: u64,
    pub prefill_ns: u64,
    pub decode_ns: u64,
    pub decode_steps: usize,
    pub total_ns: u64,
    pub stopped_by_complete_assessment: bool,
}

#[derive(Debug, Clone)]
pub struct Qwen3GuardOnnxBatchProfile {
    pub output: String,
    pub batch_size: usize,
    pub prompt_tokens: usize,
    pub generated_tokens_per_request: usize,
    pub tokenize_ns: u64,
    pub prefill_ns: u64,
    pub decode_ns: u64,
    pub decode_steps: usize,
    pub total_ns: u64,
    pub stopped_by_complete_assessment: usize,
}

pub struct Qwen3GuardOnnxBatchPrefixCache {
    mode: String,
    batch_size: usize,
    prefix_tokens: Vec<u32>,
    past: Vec<DynValue>,
    prepare_ns: u64,
}

impl Qwen3GuardOnnxBatchPrefixCache {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn prefix_tokens(&self) -> usize {
        self.prefix_tokens.len()
    }

    pub fn prepare_ns(&self) -> u64 {
        self.prepare_ns
    }
}

pub struct Qwen3GuardOnnxPrefixCache {
    mode: String,
    prefix_tokens: Vec<u32>,
    past: Vec<DynValue>,
    prepare_ns: u64,
}

impl Qwen3GuardOnnxPrefixCache {
    pub fn prefix_tokens(&self) -> usize {
        self.prefix_tokens.len()
    }

    pub fn prepare_ns(&self) -> u64 {
        self.prepare_ns
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
    prefix_cache_input: Option<Qwen3GuardOnnxPrefixCache>,
    prefix_cache_output: Option<Qwen3GuardOnnxPrefixCache>,
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

        let mut model = Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config: config.unwrap_or_default(),
            accepts_attention_mask,
            accepts_position_ids,
            past_ios,
            eos_token_id,
            im_end_token_id,
            model_path: model_path_str,
            prefix_cache_input: None,
            prefix_cache_output: None,
        };

        if model.config.prefix_cache_enabled {
            if let Err(e) = model.initialize_guard_prefix_caches() {
                println!(
                    "WARNING: Qwen3Guard ONNX prefix cache initialization failed; using full prompt path: {}",
                    e
                );
            }
        } else {
            println!(
                "INFO: Qwen3Guard ONNX prefix cache disabled by QWEN3_GUARD_ONNX_PREFIX_CACHE"
            );
        }

        Ok(model)
    }

    pub fn generate_guard(&mut self, text: &str, mode: &str) -> UnifiedResult<String> {
        Ok(self
            .generate_guard_profile_cached_or_full(text, mode)?
            .output)
    }

    pub fn generate_guard_profile(
        &mut self,
        text: &str,
        mode: &str,
    ) -> UnifiedResult<Qwen3GuardOnnxProfile> {
        let prompt = format_guard_prompt(text, mode);
        self.generate_profile(&prompt)
    }

    pub fn generate_guard_batch_profile(
        &mut self,
        text: &str,
        mode: &str,
        batch_size: usize,
    ) -> UnifiedResult<Qwen3GuardOnnxBatchProfile> {
        if batch_size == 0 {
            return Err(errors::config_error(
                "batch_size",
                "Qwen3Guard batch microbench requires batch_size > 0",
            ));
        }
        if self.past_ios.is_empty() {
            return Err(errors::config_error(
                "batch_microbench",
                "Qwen3Guard batch microbench requires a decoder model with past_key_values inputs",
            ));
        }

        let prompt = format_guard_prompt(text, mode);
        self.generate_batch_with_past_profile(&prompt, batch_size)
    }

    pub fn prepare_guard_prefix_cache(
        &mut self,
        mode: &str,
    ) -> UnifiedResult<Qwen3GuardOnnxPrefixCache> {
        if self.past_ios.is_empty() {
            return Err(errors::config_error(
                "prefix_cache",
                "prefix cache requires a decoder model with past_key_values inputs",
            ));
        }

        let (prefix, _tail) = format_guard_prompt_parts("", mode);
        let encoding = self
            .tokenizer
            .encode(prefix.as_str(), true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        let prefix_tokens = encoding.get_ids().to_vec();
        if prefix_tokens.is_empty() {
            return Err(errors::inference_error(
                "prepare_prefix_cache",
                "empty prefix token sequence",
            ));
        }

        let start = Instant::now();
        let empty_past = self.create_empty_past()?;
        let (_logits, past) =
            self.forward_with_past(&prefix_tokens, 0, prefix_tokens.len(), empty_past)?;
        Ok(Qwen3GuardOnnxPrefixCache {
            mode: mode.to_string(),
            prefix_tokens,
            past,
            prepare_ns: duration_ns(start.elapsed()),
        })
    }

    pub fn prepare_guard_batch_prefix_cache(
        &mut self,
        mode: &str,
        batch_size: usize,
    ) -> UnifiedResult<Qwen3GuardOnnxBatchPrefixCache> {
        if batch_size == 0 {
            return Err(errors::config_error(
                "batch_size",
                "Qwen3Guard batch prefix cache requires batch_size > 0",
            ));
        }
        if self.past_ios.is_empty() {
            return Err(errors::config_error(
                "batch_prefix_cache",
                "batch prefix cache requires a decoder model with past_key_values inputs",
            ));
        }

        let (prefix, _tail) = format_guard_prompt_parts("", mode);
        let encoding = self
            .tokenizer
            .encode(prefix.as_str(), true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        let prefix_tokens = encoding.get_ids().to_vec();
        if prefix_tokens.is_empty() {
            return Err(errors::inference_error(
                "prepare_batch_prefix_cache",
                "empty prefix token sequence",
            ));
        }

        let start = Instant::now();
        let empty_past = self.create_empty_past_batch(batch_size)?;
        let (_logits, past) = self.forward_with_past_batch(
            &prefix_tokens,
            batch_size,
            0,
            prefix_tokens.len(),
            empty_past,
        )?;
        Ok(Qwen3GuardOnnxBatchPrefixCache {
            mode: mode.to_string(),
            batch_size,
            prefix_tokens,
            past,
            prepare_ns: duration_ns(start.elapsed()),
        })
    }

    pub fn generate_guard_profile_with_prefix_cache(
        &mut self,
        text: &str,
        mode: &str,
        cache: &Qwen3GuardOnnxPrefixCache,
    ) -> UnifiedResult<Qwen3GuardOnnxProfile> {
        if mode != cache.mode {
            return Err(errors::config_error(
                "prefix_cache",
                &format!(
                    "prefix cache mode '{}' cannot be used for mode '{}'",
                    cache.mode, mode
                ),
            ));
        }

        let total_start = Instant::now();
        let tokenize_start = Instant::now();
        let (prefix, tail) = format_guard_prompt_parts(text, mode);
        let full_prompt = format!("{prefix}{tail}");
        let full_tokens = self
            .tokenizer
            .encode(full_prompt.as_str(), true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?
            .get_ids()
            .to_vec();
        let prefix_tokens = self
            .tokenizer
            .encode(prefix.as_str(), true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?
            .get_ids()
            .to_vec();
        let tail_tokens = self
            .tokenizer
            .encode(tail.as_str(), true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?
            .get_ids()
            .to_vec();
        let tokenize_ns = duration_ns(tokenize_start.elapsed());

        if prefix_tokens != cache.prefix_tokens {
            return Err(errors::validation_error(
                "prefix_cache_tokens",
                "request prefix tokens match cached prefix tokens",
                "token mismatch",
            ));
        }

        let mut joined_tokens = prefix_tokens;
        joined_tokens.extend_from_slice(&tail_tokens);
        if joined_tokens != full_tokens {
            return Err(errors::validation_error(
                "prefix_cache_tokenization",
                "tokenize(prefix) + tokenize(tail) equals tokenize(full_prompt)",
                "token boundary mismatch",
            ));
        }

        self.generate_with_borrowed_prefix_past(
            &tail_tokens,
            full_tokens.len(),
            cache.prefix_tokens.len(),
            &cache.past,
            tokenize_ns,
            total_start,
        )
    }

    pub fn generate_guard_batch_profile_with_prefix_cache(
        &mut self,
        text: &str,
        mode: &str,
        cache: &Qwen3GuardOnnxBatchPrefixCache,
    ) -> UnifiedResult<Qwen3GuardOnnxBatchProfile> {
        if mode != cache.mode {
            return Err(errors::config_error(
                "batch_prefix_cache",
                &format!(
                    "batch prefix cache mode '{}' cannot be used for mode '{}'",
                    cache.mode, mode
                ),
            ));
        }

        let total_start = Instant::now();
        let tokenize_start = Instant::now();
        let (prefix, tail) = format_guard_prompt_parts(text, mode);
        let full_prompt = format!("{prefix}{tail}");
        let full_tokens = self
            .tokenizer
            .encode(full_prompt.as_str(), true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?
            .get_ids()
            .to_vec();
        let prefix_tokens = self
            .tokenizer
            .encode(prefix.as_str(), true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?
            .get_ids()
            .to_vec();
        let tail_tokens = self
            .tokenizer
            .encode(tail.as_str(), true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?
            .get_ids()
            .to_vec();
        let tokenize_ns = duration_ns(tokenize_start.elapsed());

        if prefix_tokens != cache.prefix_tokens {
            return Err(errors::validation_error(
                "batch_prefix_cache_tokens",
                "request prefix tokens match cached prefix tokens",
                "token mismatch",
            ));
        }

        let mut joined_tokens = prefix_tokens;
        joined_tokens.extend_from_slice(&tail_tokens);
        if joined_tokens != full_tokens {
            return Err(errors::validation_error(
                "batch_prefix_cache_tokenization",
                "tokenize(prefix) + tokenize(tail) equals tokenize(full_prompt)",
                "token boundary mismatch",
            ));
        }
        if tail_tokens.is_empty() {
            return Err(errors::inference_error(
                "generate_batch_prefix_cache",
                "empty tail token sequence",
            ));
        }

        let prefill_start = Instant::now();
        let (logits, past) = self.forward_with_past_batch_refs(
            &tail_tokens,
            cache.batch_size,
            cache.prefix_tokens.len(),
            cache.prefix_tokens.len() + tail_tokens.len(),
            &cache.past,
        )?;
        let prefill_ns = duration_ns(prefill_start.elapsed());
        self.finish_batch_generation(
            &tail_tokens,
            full_tokens.len(),
            cache.batch_size,
            logits,
            past,
            cache.prefix_tokens.len() + tail_tokens.len(),
            tokenize_ns,
            prefill_ns,
            total_start,
        )
    }

    pub fn model_info(&self) -> String {
        format!(
            "Qwen3GuardOnnxModel(path={}, max_tokens={}, prefix_cache={})",
            self.model_path,
            self.config.max_tokens,
            self.prefix_cache_status()
        )
    }

    fn initialize_guard_prefix_caches(&mut self) -> UnifiedResult<()> {
        if self.past_ios.is_empty() {
            println!(
                "INFO: Qwen3Guard ONNX prefix cache unavailable: model does not expose past_key_values inputs"
            );
            return Ok(());
        }

        let input_cache = self.prepare_guard_prefix_cache("input")?;
        let output_cache = self.prepare_guard_prefix_cache("output")?;
        println!(
            "INFO: Qwen3Guard ONNX prefix cache initialized: input={} tokens ({:.2}ms), output={} tokens ({:.2}ms)",
            input_cache.prefix_tokens(),
            input_cache.prepare_ns() as f64 / 1e6,
            output_cache.prefix_tokens(),
            output_cache.prepare_ns() as f64 / 1e6
        );
        self.prefix_cache_input = Some(input_cache);
        self.prefix_cache_output = Some(output_cache);
        Ok(())
    }

    fn generate_guard_profile_cached_or_full(
        &mut self,
        text: &str,
        mode: &str,
    ) -> UnifiedResult<Qwen3GuardOnnxProfile> {
        let Some(cache) = self.take_guard_prefix_cache(mode) else {
            return self.generate_guard_profile(text, mode);
        };

        let result = self.generate_guard_profile_with_prefix_cache(text, mode, &cache);
        self.restore_guard_prefix_cache(cache);
        match result {
            Ok(profile) => Ok(profile),
            Err(e) => {
                println!(
                    "WARNING: Qwen3Guard ONNX prefix cache fast path failed; using full prompt path: {}",
                    e
                );
                self.generate_guard_profile(text, mode)
            }
        }
    }

    fn take_guard_prefix_cache(&mut self, mode: &str) -> Option<Qwen3GuardOnnxPrefixCache> {
        match mode {
            "input" => self.prefix_cache_input.take(),
            "output" => self.prefix_cache_output.take(),
            _ => None,
        }
    }

    fn restore_guard_prefix_cache(&mut self, cache: Qwen3GuardOnnxPrefixCache) {
        let mode = cache.mode.clone();
        match mode.as_str() {
            "input" => self.prefix_cache_input = Some(cache),
            "output" => self.prefix_cache_output = Some(cache),
            _ => {}
        }
    }

    fn prefix_cache_status(&self) -> &'static str {
        if !self.config.prefix_cache_enabled {
            "disabled"
        } else if self.prefix_cache_input.is_some() && self.prefix_cache_output.is_some() {
            "ready"
        } else {
            "unavailable"
        }
    }

    fn generate_profile(&mut self, prompt: &str) -> UnifiedResult<Qwen3GuardOnnxProfile> {
        if self.past_ios.is_empty() {
            self.generate_no_past_profile(prompt)
        } else {
            self.generate_with_past_profile(prompt)
        }
    }

    fn generate_no_past_profile(&mut self, prompt: &str) -> UnifiedResult<Qwen3GuardOnnxProfile> {
        let total_start = Instant::now();
        let tokenize_start = Instant::now();
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        let tokenize_ns = duration_ns(tokenize_start.elapsed());
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_tokens = tokens.len();
        let mut generated_text = String::new();
        let mut decode_ns = 0;
        let mut generated_tokens = 0;
        let mut stopped_by_complete_assessment = false;

        for _ in 0..self.config.max_tokens {
            let decode_start = Instant::now();
            let logits = self.forward_full_context(&tokens)?;
            decode_ns += duration_ns(decode_start.elapsed());
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
            generated_tokens += 1;
            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                generated_text.push_str(&piece);
            }
            if guard_assessment_complete(&generated_text) {
                stopped_by_complete_assessment = true;
                break;
            }
        }

        Ok(Qwen3GuardOnnxProfile {
            output: generated_text,
            prompt_tokens,
            generated_tokens,
            tokenize_ns,
            prefill_ns: 0,
            decode_ns,
            decode_steps: generated_tokens,
            total_ns: duration_ns(total_start.elapsed()),
            stopped_by_complete_assessment,
        })
    }

    fn generate_with_past_profile(&mut self, prompt: &str) -> UnifiedResult<Qwen3GuardOnnxProfile> {
        let total_start = Instant::now();
        let tokenize_start = Instant::now();
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        let tokenize_ns = duration_ns(tokenize_start.elapsed());
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            return Err(errors::inference_error(
                "generate",
                "empty prompt token sequence",
            ));
        }
        let prompt_tokens = tokens.len();

        let mut past = self.create_empty_past()?;
        let mut past_len = 0usize;
        let mut prefill = true;
        let mut generated_text = String::new();
        let mut prefill_ns = 0;
        let mut decode_ns = 0;
        let mut decode_steps = 0;
        let mut generated_tokens = 0;
        let mut stopped_by_complete_assessment = false;

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

            let forward_start = Instant::now();
            let (logits, next_past) =
                self.forward_with_past(&input_tokens, position_start, attention_len, past)?;
            let forward_ns = duration_ns(forward_start.elapsed());
            if prefill {
                prefill_ns += forward_ns;
            } else {
                decode_ns += forward_ns;
                decode_steps += 1;
            }
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
            generated_tokens += 1;
            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                generated_text.push_str(&piece);
            }
            if guard_assessment_complete(&generated_text) {
                stopped_by_complete_assessment = true;
                break;
            }
        }

        Ok(Qwen3GuardOnnxProfile {
            output: generated_text,
            prompt_tokens,
            generated_tokens,
            tokenize_ns,
            prefill_ns,
            decode_ns,
            decode_steps,
            total_ns: duration_ns(total_start.elapsed()),
            stopped_by_complete_assessment,
        })
    }

    fn generate_batch_with_past_profile(
        &mut self,
        prompt: &str,
        batch_size: usize,
    ) -> UnifiedResult<Qwen3GuardOnnxBatchProfile> {
        let total_start = Instant::now();
        let tokenize_start = Instant::now();
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        let tokenize_ns = duration_ns(tokenize_start.elapsed());
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_tokens.is_empty() {
            return Err(errors::inference_error(
                "generate_batch",
                "empty prompt token sequence",
            ));
        }
        let prompt_token_count = prompt_tokens.len();

        let mut past = self.create_empty_past_batch(batch_size)?;
        let prefill_start = Instant::now();
        let (logits, next_past) =
            self.forward_with_past_batch(&prompt_tokens, batch_size, 0, prompt_tokens.len(), past)?;
        let prefill_ns = duration_ns(prefill_start.elapsed());
        past = next_past;

        let mut past_len = prompt_tokens.len();
        let mut histories = vec![prompt_tokens.clone(); batch_size];
        let mut generated_texts = vec![String::new(); batch_size];
        let mut done = vec![false; batch_size];
        let mut stopped_by_complete_assessment = vec![false; batch_size];
        let mut generated_tokens_per_request = 0;
        let mut decode_ns = 0;
        let mut decode_steps = 0;

        let mut next_tokens = logits
            .into_iter()
            .enumerate()
            .map(|(idx, logits)| {
                let logits = apply_repeat_penalty(
                    logits,
                    &histories[idx][histories[idx]
                        .len()
                        .saturating_sub(self.config.repeat_last_n)..],
                    self.config.repeat_penalty,
                );
                argmax(&logits) as u32
            })
            .collect::<Vec<_>>();

        loop {
            let mut active_generated = false;
            for idx in 0..batch_size {
                if done[idx] {
                    continue;
                }

                let next_token = next_tokens[idx];
                if next_token == self.eos_token_id || Some(next_token) == self.im_end_token_id {
                    done[idx] = true;
                    continue;
                }

                histories[idx].push(next_token);
                active_generated = true;
                if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                    generated_texts[idx].push_str(&piece);
                }
                if guard_assessment_complete(&generated_texts[idx])
                    || histories[idx].len() - prompt_token_count >= self.config.max_tokens
                {
                    stopped_by_complete_assessment[idx] =
                        guard_assessment_complete(&generated_texts[idx]);
                    done[idx] = true;
                }
            }

            if active_generated {
                generated_tokens_per_request += 1;
            }
            if done.iter().all(|is_done| *is_done)
                || generated_tokens_per_request >= self.config.max_tokens
            {
                break;
            }

            let decode_start = Instant::now();
            let (logits, next_past) = self.forward_with_past_batch(
                &next_tokens,
                batch_size,
                past_len,
                past_len + 1,
                past,
            )?;
            decode_ns += duration_ns(decode_start.elapsed());
            decode_steps += 1;
            past = next_past;
            past_len += 1;

            for (idx, logits) in logits.into_iter().enumerate() {
                if done[idx] {
                    continue;
                }
                let logits = apply_repeat_penalty(
                    logits,
                    &histories[idx][histories[idx]
                        .len()
                        .saturating_sub(self.config.repeat_last_n)..],
                    self.config.repeat_penalty,
                );
                next_tokens[idx] = argmax(&logits) as u32;
            }
        }

        Ok(Qwen3GuardOnnxBatchProfile {
            output: generated_texts.into_iter().next().unwrap_or_default(),
            batch_size,
            prompt_tokens: prompt_token_count,
            generated_tokens_per_request,
            tokenize_ns,
            prefill_ns,
            decode_ns,
            decode_steps,
            total_ns: duration_ns(total_start.elapsed()),
            stopped_by_complete_assessment: stopped_by_complete_assessment
                .into_iter()
                .filter(|stopped| *stopped)
                .count(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_batch_generation(
        &mut self,
        seed_tokens: &[u32],
        prompt_token_count: usize,
        batch_size: usize,
        logits: Vec<Vec<f32>>,
        mut past: Vec<DynValue>,
        mut past_len: usize,
        tokenize_ns: u64,
        prefill_ns: u64,
        total_start: Instant,
    ) -> UnifiedResult<Qwen3GuardOnnxBatchProfile> {
        let mut histories = vec![seed_tokens.to_vec(); batch_size];
        let mut generated_texts = vec![String::new(); batch_size];
        let mut done = vec![false; batch_size];
        let mut stopped_by_complete_assessment = vec![false; batch_size];
        let mut generated_tokens_per_request = 0;
        let mut decode_ns = 0;
        let mut decode_steps = 0;

        let mut next_tokens = logits
            .into_iter()
            .enumerate()
            .map(|(idx, logits)| {
                let logits = apply_repeat_penalty(
                    logits,
                    &histories[idx][histories[idx]
                        .len()
                        .saturating_sub(self.config.repeat_last_n)..],
                    self.config.repeat_penalty,
                );
                argmax(&logits) as u32
            })
            .collect::<Vec<_>>();

        loop {
            let mut active_generated = false;
            for idx in 0..batch_size {
                if done[idx] {
                    continue;
                }

                let next_token = next_tokens[idx];
                if next_token == self.eos_token_id || Some(next_token) == self.im_end_token_id {
                    done[idx] = true;
                    continue;
                }

                histories[idx].push(next_token);
                active_generated = true;
                if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                    generated_texts[idx].push_str(&piece);
                }
                if guard_assessment_complete(&generated_texts[idx])
                    || histories[idx].len().saturating_sub(seed_tokens.len())
                        >= self.config.max_tokens
                {
                    stopped_by_complete_assessment[idx] =
                        guard_assessment_complete(&generated_texts[idx]);
                    done[idx] = true;
                }
            }

            if active_generated {
                generated_tokens_per_request += 1;
            }
            if done.iter().all(|is_done| *is_done)
                || generated_tokens_per_request >= self.config.max_tokens
            {
                break;
            }

            let decode_start = Instant::now();
            let (logits, next_past) = self.forward_with_past_batch(
                &next_tokens,
                batch_size,
                past_len,
                past_len + 1,
                past,
            )?;
            decode_ns += duration_ns(decode_start.elapsed());
            decode_steps += 1;
            past = next_past;
            past_len += 1;

            for (idx, logits) in logits.into_iter().enumerate() {
                if done[idx] {
                    continue;
                }
                let logits = apply_repeat_penalty(
                    logits,
                    &histories[idx][histories[idx]
                        .len()
                        .saturating_sub(self.config.repeat_last_n)..],
                    self.config.repeat_penalty,
                );
                next_tokens[idx] = argmax(&logits) as u32;
            }
        }

        Ok(Qwen3GuardOnnxBatchProfile {
            output: generated_texts.into_iter().next().unwrap_or_default(),
            batch_size,
            prompt_tokens: prompt_token_count,
            generated_tokens_per_request,
            tokenize_ns,
            prefill_ns,
            decode_ns,
            decode_steps,
            total_ns: duration_ns(total_start.elapsed()),
            stopped_by_complete_assessment: stopped_by_complete_assessment
                .into_iter()
                .filter(|stopped| *stopped)
                .count(),
        })
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

    fn forward_with_past_refs(
        &mut self,
        tokens: &[u32],
        position_start: usize,
        attention_len: usize,
        past: &[DynValue],
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

        for (io, value) in self.past_ios.iter().zip(past.iter()) {
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

    fn create_empty_past_batch(&self, batch_size: usize) -> UnifiedResult<Vec<DynValue>> {
        self.past_ios
            .iter()
            .map(|io| create_empty_past_value_batch(io, self.session.allocator(), batch_size))
            .collect::<UnifiedResult<Vec<_>>>()
    }

    fn forward_with_past_batch(
        &mut self,
        tokens: &[u32],
        batch_size: usize,
        position_start: usize,
        attention_len: usize,
        past: Vec<DynValue>,
    ) -> UnifiedResult<(Vec<Vec<f32>>, Vec<DynValue>)> {
        if tokens.is_empty() {
            return Err(errors::inference_error(
                "forward_batch",
                "empty token sequence",
            ));
        }
        if batch_size == 0 {
            return Err(errors::inference_error("forward_batch", "empty batch"));
        }
        if past.len() != self.past_ios.len() {
            return Err(errors::inference_error(
                "forward_batch",
                &format!(
                    "KV cache count mismatch: expected {}, got {}",
                    self.past_ios.len(),
                    past.len()
                ),
            ));
        }
        let is_decode_batch = tokens.len() == batch_size && attention_len == position_start + 1;
        if is_decode_batch && tokens.is_empty() {
            return Err(errors::inference_error(
                "forward_batch",
                "decode batch must contain one token per batch item",
            ));
        }

        let seq_len = if is_decode_batch { 1 } else { tokens.len() };
        let input_ids: Vec<i64> = if is_decode_batch {
            tokens.iter().map(|&token| token as i64).collect()
        } else {
            let mut values = Vec::with_capacity(batch_size * tokens.len());
            for _ in 0..batch_size {
                values.extend(tokens.iter().map(|&token| token as i64));
            }
            values
        };
        let input_ids_tensor =
            Tensor::from_array(([batch_size, seq_len], input_ids)).map_err(|e: ort::Error| {
                errors::inference_error("create_batch_input_ids", &e.to_string())
            })?;

        let mut inputs: Vec<(Cow<'static, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(3 + self.past_ios.len());
        inputs.push((
            Cow::Borrowed("input_ids"),
            SessionInputValue::from(input_ids_tensor),
        ));

        if self.accepts_attention_mask {
            let attention_mask = vec![1i64; batch_size * attention_len];
            let attention_mask_tensor =
                Tensor::from_array(([batch_size, attention_len], attention_mask)).map_err(
                    |e: ort::Error| {
                        errors::inference_error("create_batch_attention_mask", &e.to_string())
                    },
                )?;
            inputs.push((
                Cow::Borrowed("attention_mask"),
                SessionInputValue::from(attention_mask_tensor),
            ));
        }

        if self.accepts_position_ids {
            let mut position_ids = Vec::with_capacity(batch_size * seq_len);
            for _ in 0..batch_size {
                position_ids
                    .extend((position_start..position_start + seq_len).map(|pos| pos as i64));
            }
            let position_ids_tensor = Tensor::from_array(([batch_size, seq_len], position_ids))
                .map_err(|e: ort::Error| {
                    errors::inference_error("create_batch_position_ids", &e.to_string())
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
        let logits = extract_last_token_logits_batch(&outputs, batch_size)?;
        let next_past = collect_present_past(&past_ios, &mut outputs)?;
        Ok((logits, next_past))
    }

    fn forward_with_past_batch_refs(
        &mut self,
        tokens: &[u32],
        batch_size: usize,
        position_start: usize,
        attention_len: usize,
        past: &[DynValue],
    ) -> UnifiedResult<(Vec<Vec<f32>>, Vec<DynValue>)> {
        if tokens.is_empty() {
            return Err(errors::inference_error(
                "forward_batch",
                "empty token sequence",
            ));
        }
        if batch_size == 0 {
            return Err(errors::inference_error("forward_batch", "empty batch"));
        }
        if past.len() != self.past_ios.len() {
            return Err(errors::inference_error(
                "forward_batch",
                &format!(
                    "KV cache count mismatch: expected {}, got {}",
                    self.past_ios.len(),
                    past.len()
                ),
            ));
        }

        let is_decode_batch = tokens.len() == batch_size && attention_len == position_start + 1;
        let seq_len = if is_decode_batch { 1 } else { tokens.len() };
        let input_ids: Vec<i64> = if is_decode_batch {
            tokens.iter().map(|&token| token as i64).collect()
        } else {
            let mut values = Vec::with_capacity(batch_size * tokens.len());
            for _ in 0..batch_size {
                values.extend(tokens.iter().map(|&token| token as i64));
            }
            values
        };
        let input_ids_tensor =
            Tensor::from_array(([batch_size, seq_len], input_ids)).map_err(|e: ort::Error| {
                errors::inference_error("create_batch_input_ids", &e.to_string())
            })?;

        let mut inputs: Vec<(Cow<'static, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(3 + self.past_ios.len());
        inputs.push((
            Cow::Borrowed("input_ids"),
            SessionInputValue::from(input_ids_tensor),
        ));

        if self.accepts_attention_mask {
            let attention_mask = vec![1i64; batch_size * attention_len];
            let attention_mask_tensor =
                Tensor::from_array(([batch_size, attention_len], attention_mask)).map_err(
                    |e: ort::Error| {
                        errors::inference_error("create_batch_attention_mask", &e.to_string())
                    },
                )?;
            inputs.push((
                Cow::Borrowed("attention_mask"),
                SessionInputValue::from(attention_mask_tensor),
            ));
        }

        if self.accepts_position_ids {
            let mut position_ids = Vec::with_capacity(batch_size * seq_len);
            for _ in 0..batch_size {
                position_ids
                    .extend((position_start..position_start + seq_len).map(|pos| pos as i64));
            }
            let position_ids_tensor = Tensor::from_array(([batch_size, seq_len], position_ids))
                .map_err(|e: ort::Error| {
                    errors::inference_error("create_batch_position_ids", &e.to_string())
                })?;
            inputs.push((
                Cow::Borrowed("position_ids"),
                SessionInputValue::from(position_ids_tensor),
            ));
        }

        for (io, value) in self.past_ios.iter().zip(past.iter()) {
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
        let logits = extract_last_token_logits_batch(&outputs, batch_size)?;
        let next_past = collect_present_past(&past_ios, &mut outputs)?;
        Ok((logits, next_past))
    }

    fn generate_with_borrowed_prefix_past(
        &mut self,
        tail_tokens: &[u32],
        prompt_tokens: usize,
        prefix_len: usize,
        prefix_past: &[DynValue],
        tokenize_ns: u64,
        total_start: Instant,
    ) -> UnifiedResult<Qwen3GuardOnnxProfile> {
        if tail_tokens.is_empty() {
            return Err(errors::inference_error(
                "generate_prefix_cache",
                "empty tail token sequence",
            ));
        }

        let tail_start = Instant::now();
        let (logits, mut past) = self.forward_with_past_refs(
            tail_tokens,
            prefix_len,
            prefix_len + tail_tokens.len(),
            prefix_past,
        )?;
        let prefill_ns = duration_ns(tail_start.elapsed());
        let mut past_len = prefix_len + tail_tokens.len();
        let mut tokens: Vec<u32> = tail_tokens.to_vec();
        let mut generated_text = String::new();
        let mut decode_ns = 0;
        let mut decode_steps = 0;
        let mut generated_tokens = 0;
        let mut stopped_by_complete_assessment = false;

        let logits = apply_repeat_penalty(
            logits,
            &tokens[tokens.len().saturating_sub(self.config.repeat_last_n)..],
            self.config.repeat_penalty,
        );
        let mut next_token = argmax(&logits) as u32;

        loop {
            if next_token == self.eos_token_id || Some(next_token) == self.im_end_token_id {
                break;
            }

            tokens.push(next_token);
            generated_tokens += 1;
            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                generated_text.push_str(&piece);
            }
            if guard_assessment_complete(&generated_text)
                || generated_tokens >= self.config.max_tokens
            {
                stopped_by_complete_assessment = guard_assessment_complete(&generated_text);
                break;
            }

            let decode_start = Instant::now();
            let (logits, next_past) =
                self.forward_with_past(&[next_token], past_len, past_len + 1, past)?;
            decode_ns += duration_ns(decode_start.elapsed());
            decode_steps += 1;
            past = next_past;
            past_len += 1;
            let logits = apply_repeat_penalty(
                logits,
                &tokens[tokens.len().saturating_sub(self.config.repeat_last_n)..],
                self.config.repeat_penalty,
            );
            next_token = argmax(&logits) as u32;
        }

        Ok(Qwen3GuardOnnxProfile {
            output: generated_text,
            prompt_tokens,
            generated_tokens,
            tokenize_ns,
            prefill_ns,
            decode_ns,
            decode_steps,
            total_ns: duration_ns(total_start.elapsed()),
            stopped_by_complete_assessment,
        })
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

fn create_empty_past_value_batch(
    io: &PastKeyValueIo,
    allocator: &Allocator,
    batch_size: usize,
) -> UnifiedResult<DynValue> {
    let shape = [batch_size, io.kv_heads, 0usize, io.head_dim];
    match io.dtype {
        TensorElementType::Float16 => {
            let tensor = Tensor::<f16>::new(allocator, shape).map_err(|e: ort::Error| {
                errors::inference_error("create_empty_past_batch_f16", &e.to_string())
            })?;
            Ok(DynValue::from(tensor))
        }
        TensorElementType::Float32 => {
            let tensor = Tensor::<f32>::new(allocator, shape).map_err(|e: ort::Error| {
                errors::inference_error("create_empty_past_batch_f32", &e.to_string())
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

fn extract_last_token_logits_batch(
    outputs: &SessionOutputs<'_>,
    batch_size: usize,
) -> UnifiedResult<Vec<Vec<f32>>> {
    if let Some(output) = outputs.get("logits") {
        if let Ok(logits) = logits_batch_from_value_f32(output, batch_size) {
            return Ok(logits);
        }
        if let Ok(logits) = logits_batch_from_value_f16(output, batch_size) {
            return Ok(logits);
        }
    }

    for (_name, output) in outputs {
        if let Ok(logits) = logits_batch_from_value_f32(&output, batch_size) {
            return Ok(logits);
        }
        if let Ok(logits) = logits_batch_from_value_f16(&output, batch_size) {
            return Ok(logits);
        }
    }

    Err(errors::inference_error(
        "extract_batch_logits",
        "no f32/f16 logits tensor found in ONNX outputs",
    ))
}

fn logits_from_value_f32(output: &ort::value::DynValue) -> UnifiedResult<Vec<f32>> {
    let (shape, data) = output
        .try_extract_tensor::<f32>()
        .map_err(|e: ort::Error| errors::inference_error("extract_logits_f32", &e.to_string()))?;
    last_token_logits_from_slice(shape.as_ref(), data)
}

fn logits_batch_from_value_f32(
    output: &ort::value::DynValue,
    batch_size: usize,
) -> UnifiedResult<Vec<Vec<f32>>> {
    let (shape, data) = output
        .try_extract_tensor::<f32>()
        .map_err(|e: ort::Error| {
            errors::inference_error("extract_batch_logits_f32", &e.to_string())
        })?;
    last_token_logits_batch_from_slice(shape.as_ref(), data, batch_size)
}

fn logits_batch_from_value_f16(
    output: &ort::value::DynValue,
    batch_size: usize,
) -> UnifiedResult<Vec<Vec<f32>>> {
    let (shape, data) = output
        .try_extract_tensor::<f16>()
        .map_err(|e: ort::Error| {
            errors::inference_error("extract_batch_logits_f16", &e.to_string())
        })?;
    let converted: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    last_token_logits_batch_from_slice(shape.as_ref(), &converted, batch_size)
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

fn last_token_logits_batch_from_slice(
    shape: &[i64],
    data: &[f32],
    batch_size: usize,
) -> UnifiedResult<Vec<Vec<f32>>> {
    match shape {
        [batch, vocab] if *batch as usize == batch_size => {
            let vocab = *vocab as usize;
            if vocab == 0 || data.len() < batch_size * vocab {
                return Err(errors::inference_error(
                    "extract_batch_logits",
                    &format!("invalid logits shape {:?}", shape),
                ));
            }
            Ok((0..batch_size)
                .map(|batch_idx| {
                    let start = batch_idx * vocab;
                    data[start..start + vocab].to_vec()
                })
                .collect())
        }
        [batch, seq_len, vocab] if *batch as usize == batch_size => {
            let seq_len = *seq_len as usize;
            let vocab = *vocab as usize;
            if seq_len == 0 || vocab == 0 || data.len() < batch_size * seq_len * vocab {
                return Err(errors::inference_error(
                    "extract_batch_logits",
                    &format!("invalid logits shape {:?}", shape),
                ));
            }
            Ok((0..batch_size)
                .map(|batch_idx| {
                    let start = (batch_idx * seq_len + (seq_len - 1)) * vocab;
                    data[start..start + vocab].to_vec()
                })
                .collect())
        }
        _ => Err(errors::inference_error(
            "extract_batch_logits",
            &format!("unsupported batch logits shape {:?}", shape),
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

fn guard_assessment_complete(text: &str) -> bool {
    if !text.contains("Safety: Safe")
        && !text.contains("Safety: Unsafe")
        && !text.contains("Safety: Controversial")
    {
        return false;
    }

    let Some(categories_start) = text.find("Categories:") else {
        return false;
    };
    let categories = &text[categories_start + "Categories:".len()..];
    let trimmed = categories.trim_start();
    if trimmed.starts_with("None") {
        return true;
    }

    categories.contains('\n') && !trimmed.is_empty()
}

fn format_guard_prompt_parts(text: &str, mode: &str) -> (String, String) {
    let user_label = if mode == "output" {
        "ASSISTANT"
    } else {
        "USER"
    };

    let prefix = format!(
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
        <BEGIN CONVERSATION>\n\n",
        user_label, user_label, user_label, user_label
    );
    let tail = format!(
        "        {}: {}\n\n\
        <END CONVERSATION>\n\n\
        Provide your safety assessment for ONLY THE LAST **{}'s query** in the above conversation:\n \
        - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n \
        - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>\n\
        <|im_start|>assistant\n\
        <think>\n\n\
        </think>\n\n",
        user_label, text, user_label
    );
    (prefix, tail)
}

fn format_guard_prompt(text: &str, mode: &str) -> String {
    let (prefix, tail) = format_guard_prompt_parts(text, mode);
    format!("{prefix}{tail}")
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

fn env_bool(name: &str) -> Option<bool> {
    let raw = std::env::var(name).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
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
    fn guard_assessment_complete_detects_terminal_safe_result() {
        assert!(guard_assessment_complete("Safety: Safe\nCategories: None"));
        assert!(guard_assessment_complete(
            "Safety: Controversial\nCategories: None"
        ));
    }

    #[test]
    fn guard_assessment_complete_waits_for_category_line_end() {
        assert!(!guard_assessment_complete(
            "Safety: Unsafe\nCategories: Violent"
        ));
        assert!(guard_assessment_complete(
            "Safety: Unsafe\nCategories: Violent\n"
        ));
        assert!(!guard_assessment_complete("Safety: Unsafe\nCategories:"));
        assert!(!guard_assessment_complete("Safety: Safe"));
    }

    #[test]
    fn last_token_logits_batch_supports_rank_2_logits() {
        let data = [0.1, 0.2, 0.3, 1.1, 1.2, 1.3];
        let logits = last_token_logits_batch_from_slice(&[2, 3], &data, 2).unwrap();

        assert_eq!(logits, vec![vec![0.1, 0.2, 0.3], vec![1.1, 1.2, 1.3]]);
    }

    #[test]
    fn last_token_logits_batch_supports_rank_3_logits() {
        let data = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, // batch 0, seq 0..1
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, // batch 1, seq 0..1
        ];
        let logits = last_token_logits_batch_from_slice(&[2, 2, 3], &data, 2).unwrap();

        assert_eq!(logits, vec![vec![0.4, 0.5, 0.6], vec![1.4, 1.5, 1.6]]);
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
