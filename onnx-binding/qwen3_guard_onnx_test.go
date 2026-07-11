//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package onnx_binding

import (
	"os"
	"testing"
)

func TestQwen3GuardOnnxSmoke(t *testing.T) {
	modelPath := os.Getenv("QWEN3_GUARD_ONNX_MODEL_PATH")
	if modelPath == "" {
		t.Skip("QWEN3_GUARD_ONNX_MODEL_PATH not set")
	}

	provider := os.Getenv("QWEN3_GUARD_ONNX_PROVIDER")
	useCPU := os.Getenv("QWEN3_GUARD_ONNX_USE_CPU") == "1"
	if err := InitQwen3GuardOnnx(modelPath, useCPU, provider); err != nil {
		t.Fatalf("InitQwen3GuardOnnx failed: %v", err)
	}
	if !IsQwen3GuardOnnxInitialized() {
		t.Fatal("Qwen3Guard ONNX should be initialized")
	}

	result, err := ClassifyPromptSafetyOnnx("Hello, can you summarize this article?")
	if err != nil {
		t.Fatalf("ClassifyPromptSafetyOnnx failed: %v", err)
	}
	if result.RawOutput == "" {
		t.Fatal("expected non-empty raw output")
	}
	t.Logf("safety=%q categories=%v raw=%q", result.SafetyLabel, result.Categories, result.RawOutput)
}
