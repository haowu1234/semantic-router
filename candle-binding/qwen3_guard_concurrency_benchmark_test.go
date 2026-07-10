//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package candle_binding

import (
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const qwen3GuardBenchDefaultConcurrency = "1,2,4,8"

var qwen3GuardBenchTexts = []string{
	"What is the weather like today?",
	"My email is john@example.com",
	"How do I make cookies?",
	"Tell me how to harm someone",
	"What is 2 plus 2?",
	"Can you summarize the safety policy in one sentence?",
	"Write a polite refusal for a dangerous request.",
	"Explain why personal data should be handled carefully.",
}

var (
	qwen3GuardBenchInitOnce sync.Once
	qwen3GuardBenchInitErr  error
)

// BenchmarkQwen3GuardConcurrency measures the current global Qwen3Guard FFI
// mutex under concurrent callers.
//
// Recommended full run:
//
//	QWEN3_GUARD_MODEL_PATH=/path/to/Qwen3Guard-Gen-0.6B \
//	QWEN3_GUARD_BENCH_CONCURRENCY=1,2,4,8 \
//	go test -run '^$' -bench BenchmarkQwen3GuardConcurrency -benchtime=30s -count=1
func BenchmarkQwen3GuardConcurrency(b *testing.B) {
	initQwen3GuardForBenchmark(b)

	deviceLabel := os.Getenv("QWEN3_GUARD_BENCH_DEVICE_LABEL")
	if deviceLabel == "" {
		deviceLabel = "auto"
	}
	b.Logf(
		"Qwen3Guard benchmark environment: device_label=%s runtime_device=%s gomaxprocs=%d concurrency=%v",
		deviceLabel,
		Qwen3GuardDeviceKindString(GetQwen3GuardDeviceKind()),
		runtime.GOMAXPROCS(0),
		qwen3GuardBenchConcurrencies(),
	)

	for _, concurrency := range qwen3GuardBenchConcurrencies() {
		b.Run(fmt.Sprintf("concurrency_%d", concurrency), func(b *testing.B) {
			runQwen3GuardConcurrencyBenchmark(b, concurrency)
		})
	}
}

func initQwen3GuardForBenchmark(b *testing.B) {
	b.Helper()

	modelPath := os.Getenv("QWEN3_GUARD_MODEL_PATH")
	if modelPath == "" {
		modelPath = Qwen3GuardModelPath
	}

	qwen3GuardBenchInitOnce.Do(func() {
		qwen3GuardBenchInitErr = InitQwen3Guard(modelPath)
		if qwen3GuardBenchInitErr != nil {
			return
		}

		_, qwen3GuardBenchInitErr = ClassifyPromptSafety(qwen3GuardBenchTexts[0])
	})

	if qwen3GuardBenchInitErr != nil {
		if isModelInitializationError(qwen3GuardBenchInitErr) {
			b.Skipf(
				"skipping Qwen3Guard concurrency benchmark; set QWEN3_GUARD_MODEL_PATH to a local model: %v",
				qwen3GuardBenchInitErr,
			)
		}
		b.Fatalf("failed to initialize Qwen3Guard benchmark: %v", qwen3GuardBenchInitErr)
	}
}

func runQwen3GuardConcurrencyBenchmark(b *testing.B, concurrency int) {
	b.Helper()
	b.ReportAllocs()

	ResetQwen3GuardTimingStats()
	latencies := make([]time.Duration, b.N)
	b.ResetTimer()

	start := time.Now()
	var next atomic.Uint64
	var failures atomic.Uint64
	var wg sync.WaitGroup

	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			for {
				requestID := int(next.Add(1) - 1)
				if requestID >= b.N {
					return
				}

				text := qwen3GuardBenchTexts[(requestID+workerID)%len(qwen3GuardBenchTexts)]
				requestStart := time.Now()
				_, err := ClassifyPromptSafety(text)
				latencies[requestID] = time.Since(requestStart)
				if err != nil {
					failures.Add(1)
				}
			}
		}(worker)
	}

	wg.Wait()
	elapsed := time.Since(start)
	b.StopTimer()

	if failures.Load() > 0 {
		b.Fatalf("Qwen3Guard benchmark saw %d classification failures", failures.Load())
	}

	stats, err := GetQwen3GuardTimingStats()
	if err != nil {
		b.Fatalf("failed to get Qwen3Guard timing stats: %v", err)
	}
	reportQwen3GuardBenchmarkStats(b, concurrency, elapsed, stats, latencies)
}

func reportQwen3GuardBenchmarkStats(
	b *testing.B,
	concurrency int,
	elapsed time.Duration,
	stats Qwen3GuardTimingStats,
	latencies []time.Duration,
) {
	b.Helper()

	calls := stats.Calls
	if calls == 0 {
		b.Fatalf("Qwen3Guard timing stats recorded zero calls")
	}
	if stats.Errors != 0 {
		b.Fatalf("Qwen3Guard timing stats recorded %d errors", stats.Errors)
	}

	callsFloat := float64(calls)
	elapsedSeconds := elapsed.Seconds()
	if elapsedSeconds > 0 {
		b.ReportMetric(callsFloat/elapsedSeconds, "requests/s")
		b.ReportMetric(elapsedSeconds*1000/callsFloat, "avg_wall_ms/op")
	}

	b.ReportMetric(float64(stats.LockWaitTotalNS)/callsFloat/1e6, "avg_lock_wait_ms/op")
	b.ReportMetric(float64(stats.LockWaitMaxNS)/1e6, "max_lock_wait_ms")
	b.ReportMetric(float64(stats.GenerationTotalNS)/callsFloat/1e6, "avg_generation_ms/op")
	b.ReportMetric(float64(stats.GenerationMaxNS)/1e6, "max_generation_ms")
	b.ReportMetric(1, "current_pool_workers")
	b.ReportMetric(float64(GetQwen3GuardDeviceKind()), "runtime_device_kind")
	reportQwen3GuardLatencyPercentiles(b, latencies)

	for _, poolSize := range []int{1, 2, 4, 8} {
		parallelism := poolSize
		if parallelism > concurrency {
			parallelism = concurrency
		}
		if parallelism <= 0 || stats.GenerationTotalNS == 0 {
			continue
		}

		estimatedSeconds := (float64(stats.GenerationTotalNS) / 1e9) / float64(parallelism)
		if estimatedSeconds > 0 {
			b.ReportMetric(
				callsFloat/estimatedSeconds,
				fmt.Sprintf("estimated_pool_%d_requests/s", poolSize),
			)
		}
	}
}

func reportQwen3GuardLatencyPercentiles(b *testing.B, latencies []time.Duration) {
	b.Helper()
	if len(latencies) == 0 {
		return
	}

	sorted := append([]time.Duration(nil), latencies...)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i] < sorted[j]
	})

	b.ReportMetric(durationMS(percentileDuration(sorted, 0.50)), "p50_wall_ms")
	b.ReportMetric(durationMS(percentileDuration(sorted, 0.95)), "p95_wall_ms")
	b.ReportMetric(durationMS(percentileDuration(sorted, 0.99)), "p99_wall_ms")
}

func percentileDuration(sorted []time.Duration, percentile float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}

	index := int(percentile*float64(len(sorted)-1) + 0.5)
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func durationMS(duration time.Duration) float64 {
	return float64(duration.Nanoseconds()) / 1e6
}

func qwen3GuardBenchConcurrencies() []int {
	raw := os.Getenv("QWEN3_GUARD_BENCH_CONCURRENCY")
	if raw == "" {
		raw = qwen3GuardBenchDefaultConcurrency
	}

	seen := map[int]struct{}{}
	var concurrencies []int
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		value, err := strconv.Atoi(part)
		if err != nil || value <= 0 {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}

		seen[value] = struct{}{}
		concurrencies = append(concurrencies, value)
	}

	if len(concurrencies) == 0 {
		return []int{1}
	}
	return concurrencies
}
