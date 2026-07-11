package classification

import (
	"reflect"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestQwen3GuardModelDetection(t *testing.T) {
	tests := []struct {
		name    string
		modelID string
		want    bool
	}{
		{name: "official compact name", modelID: "models/Qwen3Guard-0.6B-ONNX-Quantized", want: true},
		{name: "snake case name", modelID: "Qwen/Qwen3_Guard-Gen-0.6B", want: true},
		{name: "underscore guard name", modelID: "local/qwen_guard_policy", want: true},
		{name: "mmbert", modelID: "models/mmbert32k-jailbreak-detector-merged", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isQwen3GuardModel(tt.modelID); got != tt.want {
				t.Fatalf("isQwen3GuardModel(%q) = %v, want %v", tt.modelID, got, tt.want)
			}
		})
	}
}

func TestQwen3GuardClassIndexesFromMapping(t *testing.T) {
	tests := []struct {
		name string
		m    *JailbreakMapping
		want qwen3GuardClassIndexes
	}{
		{
			name: "jailbreak first",
			m: &JailbreakMapping{
				LabelToIdx: map[string]int{"jailbreak": 0, "benign": 1},
				IdxToLabel: map[string]string{"0": "jailbreak", "1": "benign"},
			},
			want: qwen3GuardClassIndexes{unsafeClass: 0, safeClass: 1},
		},
		{
			name: "jailbreak second",
			m: &JailbreakMapping{
				LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1},
				IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak"},
			},
			want: qwen3GuardClassIndexes{unsafeClass: 1, safeClass: 0},
		},
		{
			name: "huggingface id fields",
			m: &JailbreakMapping{
				LabelToID: map[string]int{"safe": 4, "jailbreak": 7},
				IDToLabel: map[string]string{"4": "safe", "7": "jailbreak"},
			},
			want: qwen3GuardClassIndexes{unsafeClass: 7, safeClass: 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := qwen3GuardClassIndexesFromMapping(tt.m)
			if err != nil {
				t.Fatalf("qwen3GuardClassIndexesFromMapping() error = %v", err)
			}
			if got != tt.want {
				t.Fatalf("qwen3GuardClassIndexesFromMapping() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestQwen3GuardSafetyToClassResult(t *testing.T) {
	classes := qwen3GuardClassIndexes{unsafeClass: 1, safeClass: 0}

	tests := []struct {
		name       string
		result     *candle_binding.SafetyClassificationResult
		wantClass  int
		wantConf   float32
		wantCats   []string
		wantErrSub string
	}{
		{
			name:      "safe maps to safe class",
			result:    &candle_binding.SafetyClassificationResult{SafetyLabel: "Safe", Categories: []string{"None"}},
			wantClass: 0,
			wantConf:  0.1,
			wantCats:  []string{"None"},
		},
		{
			name:      "unsafe maps to jailbreak class",
			result:    &candle_binding.SafetyClassificationResult{SafetyLabel: "Unsafe", Categories: []string{"Jailbreak"}},
			wantClass: 1,
			wantConf:  0.9,
			wantCats:  []string{"Jailbreak"},
		},
		{
			name:      "controversial maps to safe class with lower confidence",
			result:    &candle_binding.SafetyClassificationResult{SafetyLabel: "Controversial"},
			wantClass: 0,
			wantConf:  0.6,
		},
		{
			name:       "unknown label fails closed",
			result:     &candle_binding.SafetyClassificationResult{SafetyLabel: "Unknown"},
			wantErrSub: "unknown Qwen3Guard safety label",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := qwen3GuardSafetyToClassResult(tt.result, classes)
			if tt.wantErrSub != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErrSub) {
					t.Fatalf("qwen3GuardSafetyToClassResult() error = %v, want substring %q", err, tt.wantErrSub)
				}
				return
			}
			if err != nil {
				t.Fatalf("qwen3GuardSafetyToClassResult() error = %v", err)
			}
			if got.Class != tt.wantClass || got.Confidence != tt.wantConf || !reflect.DeepEqual(got.Categories, tt.wantCats) {
				t.Fatalf("qwen3GuardSafetyToClassResult() = %+v, want class=%d confidence=%.1f categories=%#v", got, tt.wantClass, tt.wantConf, tt.wantCats)
			}
		})
	}
}

func TestBuildJailbreakDependenciesSelectsQwen3Guard(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.PromptGuard.ModelID = "models/Qwen3Guard-0.6B-ONNX-Quantized"

	mapping := &JailbreakMapping{
		LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1},
		IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak"},
	}

	initializer, inference, err := buildJailbreakDependencies(cfg, mapping)
	if err != nil {
		t.Fatalf("buildJailbreakDependencies() error = %v", err)
	}
	if _, ok := initializer.(*Qwen3GuardInitializerImpl); !ok {
		t.Fatalf("initializer = %T, want *Qwen3GuardInitializerImpl", initializer)
	}
	qwenInference, ok := inference.(*Qwen3GuardJailbreakInferenceImpl)
	if !ok {
		t.Fatalf("inference = %T, want *Qwen3GuardJailbreakInferenceImpl", inference)
	}
	if qwenInference.classes != (qwen3GuardClassIndexes{unsafeClass: 1, safeClass: 0}) {
		t.Fatalf("classes = %+v, want unsafe=1 safe=0", qwenInference.classes)
	}
}
