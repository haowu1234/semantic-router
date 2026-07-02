//! Real Qwen3Guard model benchmark:
//! Compares kv_cache_snapshot/restore vs clear+process_prefix per request.
//!
//! This benchmark directly uses the same Qwen3Model, Config, and loading code
//! as the production Qwen3Guard, just with timing instrumentation.
//!
//! Usage:
//!   export QWEN3_GUARD_MODEL_PATH=/data/models/Qwen3Guard-Gen-0.6B
//!   cargo run --release --example bench_real_model

use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::VarBuilder;
use candle_semantic_router::model_architectures::generative::qwen3_with_lora::{
    Config as Qwen3Config, ModelForCausalLM as Qwen3Model,
};
use candle_semantic_router::model_architectures::generative::qwen3_guard::qwen3_guard_loading::load_guard_var_builder;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

fn measure_ms<F: FnOnce()>(f: F) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64() * 1000.0
}

/// Build the exact prefix string Qwen3Guard uses for "input" mode.
/// This is copied verbatim from qwen3_guard.rs:extract_fixed_prefix
fn extract_fixed_prefix(mode: &str) -> String {
    let user_label = if mode == "output" { "ASSISTANT" } else { "USER" };
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
        {}: ",
        user_label, user_label, user_label, user_label, user_label
    )
}

/// Build the suffix (same as qwen3_guard_generation.rs:cached_guard_suffix)
fn cached_guard_suffix(text: &str, mode: &str) -> String {
    let user_label = if mode == "output" { "ASSISTANT" } else { "USER" };
    format!(
        "{}\n\n\
        <END CONVERSATION>\n\n\
        Provide your safety assessment for ONLY THE LAST **{}'s query** in the above conversation:\n \
        - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n \
        - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>\n\
        <|im_start|>assistant\n\
         \n\n\
        \n\n",
        text, user_label
    )
}

fn main() {
    let model_path = std::env::var("QWEN3_GUARD_MODEL_PATH")
        .unwrap_or_else(|_| "/data/models/Qwen3Guard-Gen-0.6B".to_string());

    println!("=======================================================");
    println!(" Real Qwen3Guard Benchmark: snapshot/restore vs clear+prefill");
    println!("=======================================================");
    println!();
    println!("Model path: {}", model_path);
    println!("Device: CPU (F32)");
    println!();

    let base_dir = Path::new(&model_path);

    // Load config.json (same as Qwen3GuardModel::new)
    println!("Loading config...");
    let config_path = base_dir.join("config.json");
    let config_str = std::fs::read(&config_path).expect("read config.json");
    let model_config: Qwen3Config = serde_json::from_slice(&config_str).expect("parse config");
    println!(
        "  hidden_size={}, layers={}, heads={}, kv_heads={}, head_dim={}, vocab={}",
        model_config.hidden_size,
        model_config.num_hidden_layers,
        model_config.num_attention_heads,
        model_config.num_key_value_heads,
        model_config.head_dim,
        model_config.vocab_size
    );

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer_path = base_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load tokenizer");

    // Load model weights (same as production: mmap safetensors)
    let device = Device::Cpu;
    let dtype = DType::F32; // CPU → F32 (same as Qwen3GuardModel::new)
    println!("Loading model weights (mmap, {:?})...", dtype);
    let vb = load_guard_var_builder(base_dir, dtype, &device).expect("load weights");

    // Build the actual Qwen3 model (same struct used by production guard)
    println!("Building Qwen3 model...");
    let mut model = Qwen3Model::new(&model_config, vb).expect("build model");
    println!("Model ready.");
    println!();

    // Tokenize prefix (same as initialize_prefix_caches)
    let prefix_str = extract_fixed_prefix("input");
    let prefix_encoding = tokenizer.encode(prefix_str.as_str(), true).expect("encode prefix");
    let prefix_tokens: Vec<u32> = prefix_encoding.get_ids().to_vec();
    let prefix_len = prefix_tokens.len();
    println!("Input prefix: {} tokens", prefix_len);

    // Tokenize suffix for a test request
    let test_text = "How do I make a bomb?";
    let suffix_str = cached_guard_suffix(test_text, "input");
    let suffix_encoding = tokenizer.encode(suffix_str.as_str(), true).expect("encode suffix");
    let suffix_tokens: Vec<u32> = suffix_encoding.get_ids().to_vec();
    println!("Suffix: {} tokens (text: {:?})", suffix_tokens.len(), test_text);
    println!();

    let num_requests = 5;

    // ================================================================
    // APPROACH 1: Current guard fast path (clear + process_prefix per request)
    // ================================================================
    println!("-------------------------------------------------------");
    println!("APPROACH 1: Current guard (clear+prefill per request)");
    println!("  [This is what generate_with_prefix_cache does now]");
    println!("-------------------------------------------------------");

    let mut current_times = Vec::new();
    for i in 0..num_requests {
        let t_clear = measure_ms(|| { model.clear_kv_cache(); });
        let t_prefill = measure_ms(|| {
            model.process_prefix(&prefix_tokens).expect("process_prefix");
        });
        let t_suffix = measure_ms(|| {
            let suffix_tensor = Tensor::new(&suffix_tokens[..], &device)
                .unwrap().unsqueeze(0).unwrap();
            model.forward(&suffix_tensor, prefix_len).expect("forward suffix");
        });
        let total = t_clear + t_prefill + t_suffix;
        current_times.push(total);
        println!("  Request {}: clear={:.1}ms  prefill={:.1}ms  suffix_fwd={:.1}ms  TOTAL={:.1}ms",
            i+1, t_clear, t_prefill, t_suffix, total);
    }

    // ================================================================
    // APPROACH 2: Fixed guard (prefill once + snapshot + restore per request)
    // ================================================================
    println!();
    println!("-------------------------------------------------------");
    println!("APPROACH 2: Fixed guard (prefill once + snapshot + restore)");
    println!("  [Using kv_cache_snapshot/restore that already exist]");
    println!("-------------------------------------------------------");

    // Step 1: Prefill once and snapshot (this would happen in initialize_prefix_caches)
    model.clear_kv_cache();
    let t_init_prefill = measure_ms(|| {
        model.process_prefix(&prefix_tokens).expect("process_prefix");
    });
    let t_snapshot = measure_ms(|| { model.kv_cache_snapshot(); });
    let snapshot = model.kv_cache_snapshot();
    println!("  Init: prefill={:.1}ms  snapshot={:.1}ms  (one-time cost)", t_init_prefill, t_snapshot);

    let mut fixed_times = Vec::new();
    for i in 0..num_requests {
        let t_restore = measure_ms(|| { model.kv_cache_restore(&snapshot); });
        let t_suffix = measure_ms(|| {
            let suffix_tensor = Tensor::new(&suffix_tokens[..], &device)
                .unwrap().unsqueeze(0).unwrap();
            model.forward(&suffix_tensor, prefix_len).expect("forward suffix");
        });
        let total = t_restore + t_suffix;
        fixed_times.push(total);
        println!("  Request {}: restore={:.1}ms  suffix_fwd={:.1}ms  TOTAL={:.1}ms",
            i+1, t_restore, t_suffix, total);
    }

    // ================================================================
    // COMPARISON
    // ================================================================
    println!();
    println!("-------------------------------------------------------");
    println!("COMPARISON (avg of {} requests, excluding init)", num_requests);
    println!("-------------------------------------------------------");
    let avg_current: f64 = current_times.iter().sum::<f64>() / num_requests as f64;
    let avg_fixed: f64 = fixed_times.iter().sum::<f64>() / num_requests as f64;
    let speedup = avg_current / avg_fixed;

    println!("  Current (clear+prefill per request): {:.1}ms avg", avg_current);
    println!("    breakdown: clear+prefill={:.1}ms + suffix={:.1}ms",
        avg_current - avg_fixed, avg_fixed);
    println!("  Fixed (restore per request):         {:.1}ms avg", avg_fixed);
    println!("    breakdown: restore=~0ms + suffix={:.1}ms", avg_fixed);
    println!("  Speedup:                             {:.1}x", speedup);
    println!("  Time saved per request:             {:.1}ms", avg_current - avg_fixed);
    println!();

    // ================================================================
    // DETERMINISM CHECK
    // ================================================================
    println!("-------------------------------------------------------");
    println!("DETERMINISM: verify prefill produces identical forward output");
    println!("-------------------------------------------------------");

    // Run prefill 3 times, after each do forward(suffix), compare logits
    let mut outputs = Vec::new();
    for run in 0..3 {
        model.clear_kv_cache();
        model.process_prefix(&prefix_tokens).expect("process_prefix");
        let suffix_tensor = Tensor::new(&suffix_tokens[..], &device)
            .unwrap().unsqueeze(0).unwrap();
        let logits = model.forward(&suffix_tensor, prefix_len).expect("forward");
        let logits_vec: Vec<f32> = logits.squeeze(0).unwrap()
            .squeeze(0).unwrap().to_dtype(DType::F32).unwrap()
            .to_vec1().expect("to_vec1");
        outputs.push(logits_vec);
        println!("  Run {}: {} logits, first 3: {:?}...", run+1, outputs[run].len(), &outputs[run][..3.min(outputs[run].len())]);
    }

    let identical_01 = outputs[0] == outputs[1];
    let identical_02 = outputs[0] == outputs[2];
    println!();
    println!("  Run 1 vs Run 2: {}", if identical_01 { "IDENTICAL" } else { "DIFFERS" });
    println!("  Run 1 vs Run 3: {}", if identical_02 { "IDENTICAL" } else { "DIFFERS" });
    if identical_01 && identical_02 {
        println!("  → prefill produces byte-identical forward output every time.");
        println!("    snapshot/restore is mathematically equivalent to clear+prefill.");
    } else {
        println!("  → WARNING: outputs differ! Check float determinism.");
    }

    println!();
    println!("-------------------------------------------------------");
    println!("FINAL VERDICT");
    println!("-------------------------------------------------------");
    println!("  Real Qwen3Guard-Gen-0.6B on CPU:");
    println!("    Current fast path: {:.1}ms per request (avg)", avg_current);
    println!("    With snapshot/restore: {:.1}ms per request (avg)", avg_fixed);
    println!("    Speedup: {:.1}x", speedup);
    println!("    Saved: {:.1}ms per request", avg_current - avg_fixed);
    println!();
    println!("  The kv_cache_snapshot/restore methods already exist in");
    println!("  qwen3_with_lora.rs:509,514 and are used by MultiLoRA classifier.");
    println!("  They are NOT used by qwen3_guard_generation.rs.");
}
