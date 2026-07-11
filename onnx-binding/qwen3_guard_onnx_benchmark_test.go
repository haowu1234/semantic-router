//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package onnx_binding

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var qwen3GuardBenchTexts = []string{
	"Hello, can you summarize this article?",
	"How do I reset my account password?",
	"Write a short note thanking a teammate for their help.",
	"Can you explain the difference between precision and recall?",
}

func initQwen3GuardOnnxBenchmark(b *testing.B) {
	b.Helper()

	modelPath := os.Getenv("QWEN3_GUARD_ONNX_MODEL_PATH")
	if modelPath == "" {
		b.Skip("QWEN3_GUARD_ONNX_MODEL_PATH not set")
	}

	provider := os.Getenv("QWEN3_GUARD_ONNX_PROVIDER")
	useCPU := os.Getenv("QWEN3_GUARD_ONNX_USE_CPU") == "1"
	if err := InitQwen3GuardOnnx(modelPath, useCPU, provider); err != nil {
		b.Fatalf("InitQwen3GuardOnnx failed: %v", err)
	}
}

func BenchmarkQwen3GuardOnnxSingle(b *testing.B) {
	initQwen3GuardOnnxBenchmark(b)

	var lockWaitNs int64
	var generationNs int64
	var totalNs int64

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := ClassifyPromptSafetyOnnx(qwen3GuardBenchTexts[i%len(qwen3GuardBenchTexts)])
		if err != nil {
			b.Fatalf("ClassifyPromptSafetyOnnx failed: %v", err)
		}
		lockWaitNs += int64(result.LockWaitTime)
		generationNs += int64(result.GenerationTime)
		totalNs += int64(result.TotalTime)
	}
	b.StopTimer()

	reportQwen3GuardTiming(b, b.N, lockWaitNs, generationNs, totalNs, b.Elapsed())
}

func BenchmarkQwen3GuardOnnxConcurrent(b *testing.B) {
	initQwen3GuardOnnxBenchmark(b)

	for _, concurrency := range qwen3GuardBenchmarkConcurrencies() {
		b.Run(fmt.Sprintf("goroutines_%d", concurrency), func(b *testing.B) {
			var lockWaitNs atomic.Int64
			var generationNs atomic.Int64
			var totalNs atomic.Int64
			var failures atomic.Int64

			jobs := make(chan int)
			var wg sync.WaitGroup

			b.ReportAllocs()
			b.ResetTimer()
			start := time.Now()
			for worker := 0; worker < concurrency; worker++ {
				wg.Add(1)
				go func(workerID int) {
					defer wg.Done()
					for i := range jobs {
						text := qwen3GuardBenchTexts[(workerID+i)%len(qwen3GuardBenchTexts)]
						result, err := ClassifyPromptSafetyOnnx(text)
						if err != nil {
							failures.Add(1)
							continue
						}
						lockWaitNs.Add(int64(result.LockWaitTime))
						generationNs.Add(int64(result.GenerationTime))
						totalNs.Add(int64(result.TotalTime))
					}
				}(worker)
			}
			for i := 0; i < b.N; i++ {
				jobs <- i
			}
			close(jobs)
			wg.Wait()
			elapsed := time.Since(start)
			b.StopTimer()

			if failures.Load() > 0 {
				b.Fatalf("%d classification calls failed", failures.Load())
			}
			reportQwen3GuardTiming(
				b,
				b.N,
				lockWaitNs.Load(),
				generationNs.Load(),
				totalNs.Load(),
				elapsed,
			)
		})
	}
}

func qwen3GuardBenchmarkConcurrencies() []int {
	raw := strings.TrimSpace(os.Getenv("QWEN3_GUARD_ONNX_BENCH_CONCURRENCY"))
	if raw == "" {
		return []int{1, 2, 4, 8}
	}

	var values []int
	for _, part := range strings.Split(raw, ",") {
		value, err := strconv.Atoi(strings.TrimSpace(part))
		if err == nil && value > 0 {
			values = append(values, value)
		}
	}
	if len(values) == 0 {
		return []int{1, 2, 4, 8}
	}
	return values
}

func reportQwen3GuardTiming(
	b *testing.B,
	ops int,
	lockWaitNs int64,
	generationNs int64,
	totalNs int64,
	elapsed time.Duration,
) {
	if ops == 0 {
		return
	}

	opsFloat := float64(ops)
	b.ReportMetric(opsFloat/elapsed.Seconds(), "req/s")
	b.ReportMetric(float64(lockWaitNs)/opsFloat/1e6, "lock_ms/op")
	b.ReportMetric(float64(generationNs)/opsFloat/1e6, "generation_ms/op")
	b.ReportMetric(float64(totalNs)/opsFloat/1e6, "ffi_total_ms/op")
}
