use super::qwen3_guard_sampling::{apply_repeat_penalty, sample_argmax, sample_topp};
use super::{GuardGenerationResult, Qwen3GuardModel};
use crate::core::{ConfigErrorType, UnifiedError, UnifiedResult};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::kv_cache::KvCache;
use std::collections::HashMap;

struct GuardDecodeSession {
    result_index: usize,
    tokens: Vec<u32>,
    total_tokens: usize,
    generated_tokens: usize,
    generated_text: String,
    kv_snapshot: Vec<KvCache>,
    done: bool,
}

impl Qwen3GuardModel {
    /// Generate several guard outputs together by batching autoregressive decode
    /// steps for sessions that currently share the same sequence length.
    ///
    /// The fixed prefix and request-specific suffix are still prefetched per
    /// request so that each session has an isolated KV cache. During decode,
    /// sessions with the same `total_tokens` can merge their KV snapshots into
    /// one batch cache, run one `forward([batch, 1])`, and split the updated
    /// cache back into per-request snapshots.
    pub fn generate_guard_micro_batch(
        &mut self,
        requests: &[(String, String)],
    ) -> Vec<UnifiedResult<GuardGenerationResult>> {
        let mut outputs: Vec<Option<UnifiedResult<GuardGenerationResult>>> =
            (0..requests.len()).map(|_| None).collect();
        let mut sessions = Vec::new();

        for (idx, (text, mode)) in requests.iter().enumerate() {
            let use_cache = matches!(mode.as_str(), "input" | "output")
                && match mode.as_str() {
                    "input" => self.prefix_cache_input.is_some(),
                    "output" => self.prefix_cache_output.is_some(),
                    _ => false,
                };

            if !use_cache {
                outputs[idx] = Some(self.generate_guard(text, mode));
                continue;
            }

            match self.prepare_decode_session(idx, text, mode) {
                Ok(session) => sessions.push(session),
                Err(err) => outputs[idx] = Some(Err(err)),
            }
        }

        while sessions.iter().any(|session| !session.done) {
            let groups = active_session_groups(&sessions);
            for group in groups {
                if group.len() <= 1 {
                    if let Some(session_idx) = group.first().copied() {
                        if let Err(err) = self.decode_one_session(&mut sessions[session_idx]) {
                            let result_idx = sessions[session_idx].result_index;
                            outputs[result_idx] = Some(Err(err));
                            sessions[session_idx].done = true;
                        }
                    }
                } else if let Err(err) = self.decode_session_batch(&mut sessions, &group) {
                    for session_idx in group {
                        let result_idx = sessions[session_idx].result_index;
                        outputs[result_idx] = Some(Err(clone_unified_error(&err)));
                        sessions[session_idx].done = true;
                    }
                }
            }
        }

        for session in sessions {
            let result_idx = session.result_index;
            if outputs[result_idx].is_none() {
                outputs[result_idx] = Some(Ok(GuardGenerationResult {
                    raw_output: session.generated_text,
                }));
            }
        }

        outputs
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| {
                    Err(prefix_cache_error(
                        "micro-batch scheduler did not produce a result",
                    ))
                })
            })
            .collect()
    }

    fn prepare_decode_session(
        &mut self,
        result_index: usize,
        text: &str,
        mode: &str,
    ) -> UnifiedResult<GuardDecodeSession> {
        let (_prefix_tokens, prefix_len) = self.cached_prefix_snapshot(mode)?;
        self.restore_kv_snapshot(mode)?;

        let suffix = cached_guard_suffix(text, mode);
        let encoding = self.tokenizer.encode(suffix.as_str(), true).map_err(|e| {
            UnifiedError::Configuration {
                operation: "tokenize suffix".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            }
        })?;
        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        self.process_cached_suffix(&tokens, prefix_len)?;

        Ok(GuardDecodeSession {
            result_index,
            total_tokens: prefix_len + tokens.len(),
            generated_tokens: 0,
            tokens,
            generated_text: String::new(),
            kv_snapshot: self.model.kv_cache_snapshot(),
            done: false,
        })
    }

    fn decode_one_session(&mut self, session: &mut GuardDecodeSession) -> UnifiedResult<()> {
        self.model.kv_cache_restore(&session.kv_snapshot);
        let next_token = self.decode_session_logits(session)?;
        session.kv_snapshot = self.model.kv_cache_snapshot();
        self.apply_decoded_token(session, next_token);
        Ok(())
    }

    fn decode_session_batch(
        &mut self,
        sessions: &mut [GuardDecodeSession],
        group: &[usize],
    ) -> UnifiedResult<()> {
        let first = group
            .first()
            .copied()
            .ok_or_else(|| prefix_cache_error("empty decode batch"))?;
        let total_tokens = sessions[first].total_tokens;
        let start_pos = total_tokens - 1;

        let input_tokens: Vec<u32> = group
            .iter()
            .map(|&idx| {
                sessions[idx]
                    .tokens
                    .last()
                    .copied()
                    .ok_or_else(|| prefix_cache_error("decode session has no tokens"))
            })
            .collect::<UnifiedResult<_>>()?;
        let input = Tensor::new(input_tokens.as_slice(), &self.device)
            .map_err(|e| processing_error("create decode batch tensor", e))?
            .unsqueeze(1)
            .map_err(|e| processing_error("unsqueeze decode batch", e))?;

        let snapshots: Vec<&[KvCache]> = group
            .iter()
            .map(|&idx| sessions[idx].kv_snapshot.as_slice())
            .collect();
        let merged_snapshot = merge_kv_snapshots(&snapshots)?;
        self.model.kv_cache_restore(&merged_snapshot);

        let logits = self
            .model
            .forward(&input, start_pos)
            .map_err(|e| processing_error("forward decode batch", e))?
            .squeeze(1)
            .map_err(|e| processing_error("squeeze decode batch", e))?
            .to_dtype(DType::F32)
            .map_err(|e| processing_error("decode batch to_dtype", e))?;
        let updated_snapshots = split_kv_snapshots(&self.model.kv_cache_snapshot(), group.len())?;

        for (row_idx, &session_idx) in group.iter().enumerate() {
            sessions[session_idx].kv_snapshot = updated_snapshots[row_idx].clone();
            let row_logits = logits
                .i(row_idx)
                .map_err(|e| processing_error("slice decode batch logits", e))?;
            let row_logits =
                self.apply_session_repeat_penalty(&row_logits, &sessions[session_idx])?;
            let next_token = self.sample_next_token(&row_logits)?;
            self.apply_decoded_token(&mut sessions[session_idx], next_token);
        }

        Ok(())
    }

    fn decode_session_logits(&mut self, session: &GuardDecodeSession) -> UnifiedResult<u32> {
        let context_size = 1;
        let start_pos = session.total_tokens - context_size;
        let ctxt = &session.tokens[session.tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &self.device)
            .map_err(|e| UnifiedError::Processing {
                operation: "create tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .unsqueeze(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "unsqueeze".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        let logits = self.last_token_logits(&input, start_pos)?;
        let logits = self.apply_session_repeat_penalty(&logits, session)?;
        self.sample_next_token(&logits)
    }

    fn apply_session_repeat_penalty(
        &self,
        logits: &Tensor,
        session: &GuardDecodeSession,
    ) -> UnifiedResult<Tensor> {
        if self.config.repeat_penalty != 1.0 {
            let start_at = session
                .tokens
                .len()
                .saturating_sub(self.config.repeat_last_n);
            apply_repeat_penalty(
                logits,
                self.config.repeat_penalty,
                &session.tokens[start_at..],
            )
        } else {
            Ok(logits.clone())
        }
    }

    fn apply_decoded_token(&self, session: &mut GuardDecodeSession, next_token: u32) {
        if next_token == self.eos_token_id || next_token == self.im_end_token_id {
            session.done = true;
            return;
        }

        session.tokens.push(next_token);
        session.total_tokens += 1;
        session.generated_tokens += 1;
        if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
            session.generated_text.push_str(&piece);
        }

        if session.generated_tokens >= self.config.max_tokens {
            session.done = true;
        }
    }

    /// Generate with prefix caching (faster tokenization + KV reuse).
    ///
    /// This restores a pre-computed KV cache snapshot instead of re-prefilling
    /// the fixed prefix on every request. The snapshot was taken once during
    /// `initialize_prefix_caches()` and is byte-identical to what
    /// `clear_kv_cache() + process_prefix()` would produce.
    ///
    /// This reduces per-request latency by ~640ms on CPU (eliminating the
    /// redundant prefix prefill) while producing identical output.
    pub(super) fn generate_with_prefix_cache(
        &mut self,
        text: &str,
        mode: &str,
    ) -> UnifiedResult<String> {
        let (_prefix_tokens, prefix_len) = self.cached_prefix_snapshot(mode)?;

        // Restore the pre-computed KV cache snapshot instead of clearing and
        // re-prefilling the prefix on every request. This is the same pattern
        // used by `qwen3_multi_lora_classifier.rs:1118`.
        self.restore_kv_snapshot(mode)?;

        let suffix = cached_guard_suffix(text, mode);
        let encoding = self.tokenizer.encode(suffix.as_str(), true).map_err(|e| {
            UnifiedError::Configuration {
                operation: "tokenize suffix".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            }
        })?;
        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        self.process_cached_suffix(&tokens, prefix_len)?;
        self.generate_cached_suffix_tokens(tokens, prefix_len)
    }

    fn cached_prefix_snapshot(&self, mode: &str) -> UnifiedResult<(Vec<u32>, usize)> {
        let cache = match mode {
            "input" => self
                .prefix_cache_input
                .as_ref()
                .ok_or_else(|| prefix_cache_error("input cache not initialized"))?,
            "output" => self
                .prefix_cache_output
                .as_ref()
                .ok_or_else(|| prefix_cache_error("output cache not initialized"))?,
            _ => return Err(prefix_cache_error(&format!("invalid mode: {}", mode))),
        };
        Ok((cache.prefix_tokens().to_vec(), cache.prefix_length()))
    }

    /// Restore the pre-computed KV cache snapshot for the given mode.
    ///
    /// The snapshot was created during `initialize_prefix_caches()` by running
    /// `process_prefix()` once and calling `kv_cache_snapshot()`. Restoring
    /// it here is ~0.05ms vs ~640ms for re-prefilling on CPU.
    ///
    /// This mirrors the pattern in `qwen3_multi_lora_classifier.rs:1088,1118`.
    fn restore_kv_snapshot(&mut self, mode: &str) -> UnifiedResult<()> {
        let snapshot = match mode {
            "input" => self
                .kv_snapshot_input
                .as_ref()
                .ok_or_else(|| prefix_cache_error("input KV snapshot not initialized"))?,
            "output" => self
                .kv_snapshot_output
                .as_ref()
                .ok_or_else(|| prefix_cache_error("output KV snapshot not initialized"))?,
            _ => return Err(prefix_cache_error(&format!("invalid mode: {}", mode))),
        };

        self.model.kv_cache_restore(snapshot);
        Ok(())
    }

    fn process_cached_suffix(&mut self, tokens: &[u32], prefix_len: usize) -> UnifiedResult<()> {
        let suffix_tensor = Tensor::new(tokens, &self.device)
            .map_err(|e| UnifiedError::Processing {
                operation: "create suffix tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .unsqueeze(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "unsqueeze suffix tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        self.model
            .forward(&suffix_tensor, prefix_len)
            .map_err(|e| UnifiedError::Processing {
                operation: "forward suffix".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        Ok(())
    }

    fn generate_cached_suffix_tokens(
        &mut self,
        mut tokens: Vec<u32>,
        prefix_len: usize,
    ) -> UnifiedResult<String> {
        let mut generated_text = String::new();
        let mut total_tokens = prefix_len + tokens.len();

        for _step in 0..self.config.max_tokens {
            let context_size = 1;
            let start_pos = total_tokens - context_size;
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)
                .map_err(|e| UnifiedError::Processing {
                    operation: "create tensor".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?
                .unsqueeze(0)
                .map_err(|e| UnifiedError::Processing {
                    operation: "unsqueeze".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;
            let logits = self.last_token_logits(&input, start_pos)?;
            let logits = if self.config.repeat_penalty != 1.0 {
                let start_at = tokens.len().saturating_sub(self.config.repeat_last_n);
                apply_repeat_penalty(&logits, self.config.repeat_penalty, &tokens[start_at..])?
            } else {
                logits
            };
            let next_token = self.sample_next_token(&logits)?;
            if next_token == self.eos_token_id || next_token == self.im_end_token_id {
                break;
            }

            tokens.push(next_token);
            total_tokens += 1;
            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                generated_text.push_str(&piece);
            }
        }

        Ok(generated_text)
    }

    fn last_token_logits(&mut self, input: &Tensor, start_pos: usize) -> UnifiedResult<Tensor> {
        self.model
            .forward(input, start_pos)
            .map_err(|e| UnifiedError::Processing {
                operation: "forward pass".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .squeeze(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "squeeze".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .squeeze(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "squeeze".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .to_dtype(DType::F32)
            .map_err(|e| UnifiedError::Processing {
                operation: "to_dtype".to_string(),
                source: e.to_string(),
                input_context: None,
            })
    }

    fn sample_next_token(&self, logits: &Tensor) -> UnifiedResult<u32> {
        if self.config.temperature == 0.0 {
            sample_argmax(logits)
        } else {
            sample_topp(logits, self.config.temperature, self.config.top_p)
        }
    }
}

fn active_session_groups(sessions: &[GuardDecodeSession]) -> Vec<Vec<usize>> {
    let mut by_total_tokens: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, session) in sessions.iter().enumerate() {
        if !session.done {
            by_total_tokens
                .entry(session.total_tokens)
                .or_insert_with(Vec::new)
                .push(idx);
        }
    }
    by_total_tokens.into_values().collect()
}

fn merge_kv_snapshots(snapshots: &[&[KvCache]]) -> UnifiedResult<Vec<KvCache>> {
    let first = snapshots
        .first()
        .ok_or_else(|| prefix_cache_error("empty KV snapshot batch"))?;
    let layer_count = first.len();
    let mut merged = Vec::with_capacity(layer_count);

    for layer_idx in 0..layer_count {
        let mut keys = Vec::with_capacity(snapshots.len());
        let mut values = Vec::with_capacity(snapshots.len());
        let mut expected_seq_len = None;

        for snapshot in snapshots {
            if snapshot.len() != layer_count {
                return Err(prefix_cache_error("mismatched KV snapshot layer count"));
            }
            let key = snapshot[layer_idx]
                .k()
                .map_err(|e| processing_error("read key cache", e))?
                .ok_or_else(|| prefix_cache_error("missing key cache"))?;
            let value = snapshot[layer_idx]
                .v()
                .map_err(|e| processing_error("read value cache", e))?
                .ok_or_else(|| prefix_cache_error("missing value cache"))?;
            let seq_len = key.dim(2).map_err(|e| processing_error("key dim", e))?;
            if expected_seq_len
                .map(|expected| expected != seq_len)
                .unwrap_or(false)
            {
                return Err(prefix_cache_error("mismatched KV sequence length"));
            }
            expected_seq_len = Some(seq_len);
            keys.push(key);
            values.push(value);
        }

        let key_refs: Vec<&Tensor> = keys.iter().collect();
        let value_refs: Vec<&Tensor> = values.iter().collect();
        let merged_key =
            Tensor::cat(&key_refs, 0).map_err(|e| processing_error("merge key cache", e))?;
        let merged_value =
            Tensor::cat(&value_refs, 0).map_err(|e| processing_error("merge value cache", e))?;
        merged.push(kv_cache_from_tensors(&merged_key, &merged_value)?);
    }

    Ok(merged)
}

fn split_kv_snapshots(
    merged_snapshot: &[KvCache],
    batch_size: usize,
) -> UnifiedResult<Vec<Vec<KvCache>>> {
    let mut split = vec![Vec::with_capacity(merged_snapshot.len()); batch_size];

    for layer_cache in merged_snapshot {
        let key = layer_cache
            .k()
            .map_err(|e| processing_error("read merged key cache", e))?
            .ok_or_else(|| prefix_cache_error("missing merged key cache"))?;
        let value = layer_cache
            .v()
            .map_err(|e| processing_error("read merged value cache", e))?
            .ok_or_else(|| prefix_cache_error("missing merged value cache"))?;

        for (batch_idx, split_snapshot) in split.iter_mut().enumerate().take(batch_size) {
            let single_key = key
                .narrow(0, batch_idx, 1)
                .map_err(|e| processing_error("split key cache", e))?;
            let single_value = value
                .narrow(0, batch_idx, 1)
                .map_err(|e| processing_error("split value cache", e))?;
            split_snapshot.push(kv_cache_from_tensors(&single_key, &single_value)?);
        }
    }

    Ok(split)
}

fn kv_cache_from_tensors(key: &Tensor, value: &Tensor) -> UnifiedResult<KvCache> {
    let seq_len = key
        .dim(2)
        .map_err(|e| processing_error("KV cache seq dim", e))?;
    let key = key
        .contiguous()
        .map_err(|e| processing_error("contiguous key cache", e))?;
    let value = value
        .contiguous()
        .map_err(|e| processing_error("contiguous value cache", e))?;
    let mut cache = KvCache::new(2, seq_len.max(1));
    cache
        .append(&key, &value)
        .map_err(|e| processing_error("append KV cache", e))?;
    Ok(cache)
}

fn processing_error<E: ToString>(operation: &str, source: E) -> UnifiedError {
    UnifiedError::Processing {
        operation: operation.to_string(),
        source: source.to_string(),
        input_context: None,
    }
}

fn clone_unified_error(error: &UnifiedError) -> UnifiedError {
    UnifiedError::Processing {
        operation: "micro-batch decode".to_string(),
        source: error.to_string(),
        input_context: None,
    }
}

fn prefix_cache_error(source: &str) -> UnifiedError {
    UnifiedError::Processing {
        operation: "get prefix cache".to_string(),
        source: source.to_string(),
        input_context: None,
    }
}

fn cached_guard_suffix(text: &str, mode: &str) -> String {
    let user_label = if mode == "output" {
        "ASSISTANT"
    } else {
        "USER"
    };
    format!(
        "{}\n\n\
        <END CONVERSATION>\n\n\
        Provide your safety assessment for ONLY THE LAST **{}'s query** in the above conversation:\n \
        - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n \
        - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>\n\
        <|im_start|>assistant\n\
        <think>\n\n\
        </think>\n\n",
        text, user_label
    )
}
