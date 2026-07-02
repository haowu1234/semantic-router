use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use std::time::Instant;

const HIDDEN_SIZE: usize = 1024;
const NUM_LAYERS: usize = 28;
const NUM_HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const PREFIX_TOKENS: usize = 500;
const NUM_CATEGORIES: usize = 10;
const NUM_TRIALS: usize = 3;
const MAX_SEQ: usize = 1024;

struct CachedLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    kv_cache: candle_nn::kv_cache::KvCache,
}

impl CachedLayer {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        let q_proj = candle_nn::linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("o_proj"))?;
        let kv_cache = candle_nn::kv_cache::KvCache::new(HIDDEN_SIZE / NUM_HEADS, MAX_SEQ);
        Ok(Self { q_proj, k_proj, v_proj, o_proj, kv_cache })
    }

    fn forward(&mut self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let q = q.reshape((b, l, NUM_HEADS, HEAD_DIM))?;
        let k = k.reshape((b, l, NUM_HEADS, HEAD_DIM))?;
        let v = v.reshape((b, l, NUM_HEADS, HEAD_DIM))?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, l, HIDDEN_SIZE))?;
        self.o_proj.forward(&out)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = candle_nn::kv_cache::KvCache::new(HIDDEN_SIZE / NUM_HEADS, MAX_SEQ);
    }
}

struct MockModel {
    layers: Vec<CachedLayer>,
    embed: candle_nn::Embedding,
    device: Device,
}

impl MockModel {
    fn new(device: &Device) -> candle_core::Result<Self> {
        let vb = VarBuilder::zeros(candle_core::DType::F32, device);
        let embed = candle_nn::embedding(151936, HIDDEN_SIZE, vb.pp("embed"))?;
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            layers.push(CachedLayer::new(vb.pp(&format!("layer.{}", i)))?);
        }
        Ok(Self { layers, embed, device: device.clone() })
    }

    fn forward(&mut self, token_ids: &[u32]) -> candle_core::Result<Tensor> {
        let ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let mut h = self.embed.forward(&ids)?;
        for layer in &mut self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers { l.clear_cache(); }
    }

    fn kv_cache_snapshot(&self) -> Vec<candle_nn::kv_cache::KvCache> {
        self.layers.iter().map(|l| l.kv_cache.clone()).collect()
    }

    fn kv_cache_restore(&mut self, caches: &[candle_nn::kv_cache::KvCache]) {
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter()) {
            layer.kv_cache = cache.clone();
        }
    }
}

fn gen_tokens(n: usize) -> Vec<u32> {
    (0..n).map(|i| (i as u32 * 7919) % 151936).collect()
}

fn measure_ms<F: FnOnce()>(f: F) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64() * 1000.0
}

fn main() -> candle_core::Result<()> {
    println!("=======================================================");
    println!(" KV Cache Benchmark: snapshot/restore vs clear+prefill");
    println!("=======================================================");
    println!();
    println!("Model: Qwen3-0.6B (hidden={}, layers={}, heads={}, dim={})",
        HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, HEAD_DIM);
    println!("Prefix: {} tokens, Categories: {}, Trials: {}", PREFIX_TOKENS, NUM_CATEGORIES, NUM_TRIALS);
    println!();

    let device = Device::Cpu;
    println!("Initializing mock model (zeros, CPU F32)...");
    let mut model = MockModel::new(&device)?;
    println!("Model ready.");
    println!();

    let prefix_tokens = gen_tokens(PREFIX_TOKENS);
    let cat_tokens: Vec<Vec<u32>> = (0..NUM_CATEGORIES).map(|i| gen_tokens(3 + i % 5)).collect();

    // Warmup
    println!("Warmup...");
    model.clear_kv_cache();
    let _ = model.forward(&prefix_tokens)?;
    let _ = model.kv_cache_snapshot();
    println!("Warmup done.");
    println!();

    // === APPROACH 1: WITH snapshot/restore ===
    println!("-------------------------------------------------------");
    println!("APPROACH 1: WITH snapshot/restore (MultiLoRA pattern)");
    println!("  clear -> prefill(500) -> snapshot -> [restore -> fwd(cat)] x N");
    println!("-------------------------------------------------------");

    let mut with_times = Vec::new();
    for trial in 0..NUM_TRIALS {
        model.clear_kv_cache();
        let t_pre = measure_ms(|| { model.forward(&prefix_tokens).unwrap(); });
        let snap = model.kv_cache_snapshot();
        let t_snap = measure_ms(|| { model.kv_cache_snapshot(); });

        let mut t_restore_total = 0.0;
        let mut t_fwd_total = 0.0;
        for ct in &cat_tokens {
            t_restore_total += measure_ms(|| { model.kv_cache_restore(&snap); });
            t_fwd_total += measure_ms(|| { model.forward(ct).unwrap(); });
        }
        let total = t_pre + t_snap + t_restore_total + t_fwd_total;
        with_times.push(total);
        println!("  Trial {}: prefill={:.1}ms snap={:.1}ms restores={:.1}ms({}x{:.2}ms) fwds={:.1}ms({}x{:.1}ms) TOTAL={:.1}ms",
            trial+1, t_pre, t_snap, t_restore_total, NUM_CATEGORIES, t_restore_total/NUM_CATEGORIES as f64,
            t_fwd_total, NUM_CATEGORIES, t_fwd_total/NUM_CATEGORIES as f64, total);
    }

    // === APPROACH 2: WITHOUT snapshot ===
    println!();
    println!("-------------------------------------------------------");
    println!("APPROACH 2: WITHOUT snapshot (clear+prefill per category)");
    println!("  [clear -> prefill(500) -> fwd(cat)] x N");
    println!("-------------------------------------------------------");

    let mut without_times = Vec::new();
    for trial in 0..NUM_TRIALS {
        let mut t_pre_total = 0.0;
        let mut t_fwd_total = 0.0;
        for ct in &cat_tokens {
            model.clear_kv_cache();
            t_pre_total += measure_ms(|| { model.forward(&prefix_tokens).unwrap(); });
            t_fwd_total += measure_ms(|| { model.forward(ct).unwrap(); });
        }
        let total = t_pre_total + t_fwd_total;
        without_times.push(total);
        println!("  Trial {}: prefills={:.1}ms({}x{:.1}ms) fwds={:.1}ms({}x{:.1}ms) TOTAL={:.1}ms",
            trial+1, t_pre_total, NUM_CATEGORIES, t_pre_total/NUM_CATEGORIES as f64,
            t_fwd_total, NUM_CATEGORIES, t_fwd_total/NUM_CATEGORIES as f64, total);
    }

    // === COMPARISON ===
    println!();
    println!("-------------------------------------------------------");
    println!("COMPARISON ({} categories, avg of {} trials)", NUM_CATEGORIES, NUM_TRIALS);
    println!("-------------------------------------------------------");
    let avg_with = with_times.iter().sum::<f64>() / NUM_TRIALS as f64;
    let avg_without = without_times.iter().sum::<f64>() / NUM_TRIALS as f64;
    println!("  WITH snapshot/restore:     {:.1}ms", avg_with);
    println!("  WITHOUT snapshot/restore:   {:.1}ms", avg_without);
    println!("  Speedup:                    {:.1}x", avg_without / avg_with);
    println!("  Time saved per classification: {:.1}ms", avg_without - avg_with);
    println!();

    // === GUARD SCENARIO ===
    println!("-------------------------------------------------------");
    println!("GUARD SCENARIO: 20 sequential requests (prefill only)");
    println!("-------------------------------------------------------");
    let n_req = 20;

    let mut guard_current = 0.0;
    for _ in 0..n_req {
        model.clear_kv_cache();
        guard_current += measure_ms(|| { model.forward(&prefix_tokens).unwrap(); });
    }

    model.clear_kv_cache();
    let init_pre = measure_ms(|| { model.forward(&prefix_tokens).unwrap(); });
    let snap = model.kv_cache_snapshot();
    let mut guard_fixed = init_pre;
    for _ in 0..n_req {
        guard_fixed += measure_ms(|| { model.kv_cache_restore(&snap); });
    }

    println!("  Current guard ({}x clear+prefill):      {:.1}ms", n_req, guard_current);
    println!("  Fixed guard (1x prefill + {}x restore): {:.1}ms", n_req, guard_fixed);
    println!("  Speedup:                                 {:.1}x", guard_current / guard_fixed);
    println!("  Saved:                                   {:.1}ms ({:.1}s)", guard_current - guard_fixed, (guard_current - guard_fixed)/1000.0);

    Ok(())
}
