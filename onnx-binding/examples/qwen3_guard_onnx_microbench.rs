//! Qwen3Guard ONNX Runtime internal microbenchmark.
//!
//! This bypasses Go/cgo and the FFI mutex to measure the model path directly:
//! tokenization, prompt prefill, token decode, and total generation time.

use onnx_semantic_router::{
    ClassifierExecutionProvider, Qwen3GuardOnnxBatchProfile, Qwen3GuardOnnxConfig,
    Qwen3GuardOnnxModel, Qwen3GuardOnnxProfile,
};
use std::env;
use std::error::Error;

const DEFAULT_RUNS: usize = 5;

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = env::args()
        .nth(1)
        .or_else(|| env::var("QWEN3_GUARD_ONNX_MODEL_PATH").ok())
        .ok_or("model path argument or QWEN3_GUARD_ONNX_MODEL_PATH is required")?;
    let provider = provider_from_env();
    let runs = env_usize("QWEN3_GUARD_ONNX_RUNS").unwrap_or(DEFAULT_RUNS);
    let batch_sizes =
        env_usize_list("QWEN3_GUARD_ONNX_BATCH_SIZES").unwrap_or_else(|| vec![1, 2, 4, 8, 16]);
    let mode = env::var("QWEN3_GUARD_ONNX_MODE").unwrap_or_else(|_| "input".to_string());
    let text = env::var("QWEN3_GUARD_ONNX_TEXT")
        .unwrap_or_else(|_| "Hello, can you summarize this article?".to_string());

    println!("# Qwen3Guard ONNX Microbench");
    println!();
    println!("- model: `{}`", model_path);
    println!("- provider: `{:?}`", provider);
    println!("- runs: `{}`", runs);
    println!("- batch_sizes: `{:?}`", batch_sizes);
    println!(
        "- max_tokens: `{}`",
        Qwen3GuardOnnxConfig::default().max_tokens
    );
    println!("- mode: `{}`", mode);
    println!();

    let mut model = Qwen3GuardOnnxModel::load(&model_path, provider, None)?;

    let warmup = model.generate_guard_profile(&text, &mode)?;
    println!("Warmup output: {:?}", warmup.output);

    let mut baseline_profiles = Vec::with_capacity(runs);
    for _ in 0..runs {
        baseline_profiles.push(model.generate_guard_profile(&text, &mode)?);
    }
    print_profiles("baseline_full_prompt", &baseline_profiles);

    println!();
    println!("## prefix_cache_probe");
    match model.prepare_guard_prefix_cache(&mode) {
        Ok(cache) => {
            println!();
            println!("- prefix_tokens: `{}`", cache.prefix_tokens());
            println!(
                "- one_time_prepare_ms: `{:.3}`",
                cache.prepare_ns() as f64 / 1_000_000.0
            );
            let warmup = model.generate_guard_profile_with_prefix_cache(&text, &mode, &cache)?;
            println!("- warmup_output: {:?}", warmup.output);
            let mut cached_profiles = Vec::with_capacity(runs);
            for _ in 0..runs {
                cached_profiles
                    .push(model.generate_guard_profile_with_prefix_cache(&text, &mode, &cache)?);
            }
            print_profiles("prefix_cache_tail", &cached_profiles);
        }
        Err(error) => {
            println!();
            println!("- unavailable: `{}`", error);
        }
    }

    println!();
    println!("## batch_probe_full_prompt");
    println!();
    println!(
        "This probe repeats the same prompt across each batch slot and uses one ONNX session run per prefill/decode step."
    );
    for batch_size in batch_sizes {
        match run_batch_profiles(&mut model, &text, &mode, batch_size, runs) {
            Ok(profiles) => print_batch_profiles(&format!("batch_size_{batch_size}"), &profiles),
            Err(error) => {
                println!();
                println!("### batch_size_{batch_size}");
                println!();
                println!("- unavailable: `{}`", error);
            }
        }

        match run_batch_prefix_profiles(&mut model, &text, &mode, batch_size, runs) {
            Ok(profiles) => print_batch_profiles(
                &format!("batch_size_{batch_size}_prefix_cache_tail"),
                &profiles,
            ),
            Err(error) => {
                println!();
                println!("### batch_size_{batch_size}_prefix_cache_tail");
                println!();
                println!("- unavailable: `{}`", error);
            }
        }
    }

    Ok(())
}

fn provider_from_env() -> ClassifierExecutionProvider {
    if env::var("QWEN3_GUARD_ONNX_USE_CPU").ok().as_deref() == Some("1") {
        return ClassifierExecutionProvider::Cpu;
    }

    match env::var("QWEN3_GUARD_ONNX_PROVIDER")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "rocm" | "migraphx" | "amd" => ClassifierExecutionProvider::Rocm,
        "cuda" | "nvidia" => ClassifierExecutionProvider::Cuda,
        "openvino" => ClassifierExecutionProvider::OpenVino,
        "cpu" => ClassifierExecutionProvider::Cpu,
        _ => ClassifierExecutionProvider::Auto,
    }
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn env_usize_list(name: &str) -> Option<Vec<usize>> {
    let values = env::var(name).ok()?;
    let parsed = values
        .split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .collect::<Vec<_>>();
    if parsed.is_empty() {
        None
    } else {
        Some(parsed)
    }
}

fn run_batch_profiles(
    model: &mut Qwen3GuardOnnxModel,
    text: &str,
    mode: &str,
    batch_size: usize,
    runs: usize,
) -> Result<Vec<Qwen3GuardOnnxBatchProfile>, Box<dyn Error>> {
    let warmup = model.generate_guard_batch_profile(text, mode, batch_size)?;
    println!(
        "- batch_size_{}_warmup_output: {:?}",
        batch_size, warmup.output
    );
    let mut profiles = Vec::with_capacity(runs);
    for _ in 0..runs {
        profiles.push(model.generate_guard_batch_profile(text, mode, batch_size)?);
    }
    Ok(profiles)
}

fn run_batch_prefix_profiles(
    model: &mut Qwen3GuardOnnxModel,
    text: &str,
    mode: &str,
    batch_size: usize,
    runs: usize,
) -> Result<Vec<Qwen3GuardOnnxBatchProfile>, Box<dyn Error>> {
    let cache = model.prepare_guard_batch_prefix_cache(mode, batch_size)?;
    println!(
        "- batch_size_{}_prefix_tokens: `{}`",
        batch_size,
        cache.prefix_tokens()
    );
    println!(
        "- batch_size_{}_prefix_prepare_ms: `{:.3}`",
        batch_size,
        cache.prepare_ns() as f64 / 1_000_000.0
    );
    let warmup = model.generate_guard_batch_profile_with_prefix_cache(text, mode, &cache)?;
    println!(
        "- batch_size_{}_prefix_warmup_output: {:?}",
        batch_size, warmup.output
    );
    let mut profiles = Vec::with_capacity(runs);
    for _ in 0..runs {
        profiles.push(model.generate_guard_batch_profile_with_prefix_cache(text, mode, &cache)?);
    }
    Ok(profiles)
}

fn print_duration_row(name: &str, values: impl Iterator<Item = u64>) {
    let values: Vec<f64> = values.map(|ns| ns as f64 / 1_000_000.0).collect();
    print_f64_row(name, &values);
}

fn print_batch_profiles(name: &str, profiles: &[Qwen3GuardOnnxBatchProfile]) {
    println!();
    println!("### {name}");
    println!();
    println!("| metric | avg | min | max |");
    println!("| --- | ---: | ---: | ---: |");
    print_usize_row("batch_size", profiles.iter().map(|p| p.batch_size));
    print_duration_row("tokenize_ms", profiles.iter().map(|p| p.tokenize_ns));
    print_duration_row("prefill_ms", profiles.iter().map(|p| p.prefill_ns));
    print_duration_row("decode_ms", profiles.iter().map(|p| p.decode_ns));
    print_duration_row("batch_total_ms", profiles.iter().map(|p| p.total_ns));
    print_f64_row(
        "per_request_ms",
        &profiles
            .iter()
            .map(|p| (p.total_ns as f64 / 1_000_000.0) / p.batch_size as f64)
            .collect::<Vec<_>>(),
    );
    print_usize_row("prompt_tokens", profiles.iter().map(|p| p.prompt_tokens));
    print_usize_row(
        "generated_tokens_per_request",
        profiles.iter().map(|p| p.generated_tokens_per_request),
    );
    print_usize_row("decode_steps", profiles.iter().map(|p| p.decode_steps));
    println!(
        "| req_per_s | {:.2} | | |",
        avg_batch_requests_per_second(profiles)
    );
    println!(
        "| generated_tokens_per_s | {:.2} | | |",
        avg_batch_generated_tokens_per_second(profiles)
    );
    println!();
    println!(
        "early_stop_requests: {}/{}",
        profiles
            .iter()
            .map(|p| p.stopped_by_complete_assessment)
            .sum::<usize>(),
        profiles.iter().map(|p| p.batch_size).sum::<usize>()
    );
    println!(
        "last_output: {:?}",
        profiles.last().map(|p| p.output.as_str()).unwrap_or("")
    );
}

fn print_profiles(name: &str, profiles: &[Qwen3GuardOnnxProfile]) {
    println!();
    println!("## {name}");
    println!();
    println!("| metric | avg | min | max |");
    println!("| --- | ---: | ---: | ---: |");
    print_duration_row("tokenize_ms", profiles.iter().map(|p| p.tokenize_ns));
    print_duration_row("prefill_ms", profiles.iter().map(|p| p.prefill_ns));
    print_duration_row("decode_ms", profiles.iter().map(|p| p.decode_ns));
    print_duration_row("total_ms", profiles.iter().map(|p| p.total_ns));
    print_usize_row("prompt_tokens", profiles.iter().map(|p| p.prompt_tokens));
    print_usize_row(
        "generated_tokens",
        profiles.iter().map(|p| p.generated_tokens),
    );
    print_usize_row("decode_steps", profiles.iter().map(|p| p.decode_steps));
    println!(
        "| generated_tokens_per_s | {:.2} | | |",
        avg_decode_tokens_per_second(profiles)
    );
    println!();
    println!(
        "early_stop_runs: {}/{}",
        profiles
            .iter()
            .filter(|p| p.stopped_by_complete_assessment)
            .count(),
        profiles.len()
    );
    println!(
        "last_output: {:?}",
        profiles.last().map(|p| p.output.as_str()).unwrap_or("")
    );
}

fn print_usize_row(name: &str, values: impl Iterator<Item = usize>) {
    let values: Vec<f64> = values.map(|value| value as f64).collect();
    print_f64_row(name, &values);
}

fn print_f64_row(name: &str, values: &[f64]) {
    let avg = values.iter().sum::<f64>() / values.len() as f64;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!("| {} | {:.3} | {:.3} | {:.3} |", name, avg, min, max);
}

fn avg_decode_tokens_per_second(profiles: &[Qwen3GuardOnnxProfile]) -> f64 {
    let generated_tokens: usize = profiles.iter().map(|p| p.generated_tokens).sum();
    let decode_ns: u64 = profiles.iter().map(|p| p.prefill_ns + p.decode_ns).sum();
    if decode_ns == 0 {
        return 0.0;
    }
    generated_tokens as f64 / (decode_ns as f64 / 1_000_000_000.0)
}

fn avg_batch_requests_per_second(profiles: &[Qwen3GuardOnnxBatchProfile]) -> f64 {
    let requests: usize = profiles.iter().map(|p| p.batch_size).sum();
    let total_ns: u64 = profiles.iter().map(|p| p.total_ns).sum();
    if total_ns == 0 {
        return 0.0;
    }
    requests as f64 / (total_ns as f64 / 1_000_000_000.0)
}

fn avg_batch_generated_tokens_per_second(profiles: &[Qwen3GuardOnnxBatchProfile]) -> f64 {
    let generated_tokens: usize = profiles
        .iter()
        .map(|p| p.generated_tokens_per_request * p.batch_size)
        .sum();
    let model_ns: u64 = profiles.iter().map(|p| p.prefill_ns + p.decode_ns).sum();
    if model_ns == 0 {
        return 0.0;
    }
    generated_tokens as f64 / (model_ns as f64 / 1_000_000_000.0)
}
