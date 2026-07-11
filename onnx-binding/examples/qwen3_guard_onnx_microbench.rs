//! Qwen3Guard ONNX Runtime internal microbenchmark.
//!
//! This bypasses Go/cgo and the FFI mutex to measure the model path directly:
//! tokenization, prompt prefill, token decode, and total generation time.

use onnx_semantic_router::{
    ClassifierExecutionProvider, Qwen3GuardOnnxConfig, Qwen3GuardOnnxModel, Qwen3GuardOnnxProfile,
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
    let mode = env::var("QWEN3_GUARD_ONNX_MODE").unwrap_or_else(|_| "input".to_string());
    let text = env::var("QWEN3_GUARD_ONNX_TEXT")
        .unwrap_or_else(|_| "Hello, can you summarize this article?".to_string());

    println!("# Qwen3Guard ONNX Microbench");
    println!();
    println!("- model: `{}`", model_path);
    println!("- provider: `{:?}`", provider);
    println!("- runs: `{}`", runs);
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

fn print_duration_row(name: &str, values: impl Iterator<Item = u64>) {
    let values: Vec<f64> = values.map(|ns| ns as f64 / 1_000_000.0).collect();
    print_f64_row(name, &values);
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
