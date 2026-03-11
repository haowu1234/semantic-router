# DSL GLOBAL Features Restructure Plan (方案 C)

## Overview

将 DSL 中非端点配置（semantic_cache、memory、response_api、embedding_model、provider_profile）从独立的 `BACKEND` 声明迁移到 `GLOBAL` 块内的嵌套子块，使 DSL 语义更清晰。

## ⚠️ 重要说明：两层设计

本方案迁移的是**基础设施配置**层，不涉及 PLUGIN 层。系统中存在两层配置：

| 层级 | 位置 | 职责 | 本方案是否修改 |
|------|------|------|----------------|
| **基础设施层** | `BACKEND semantic_cache` → `GLOBAL { SEMANTIC_CACHE }` | 配置缓存存储后端（Redis/Milvus 地址、最大条目、淘汰策略等） | ✅ 迁移到 GLOBAL |
| **路由级开关** | `ROUTE { PLUGIN semantic_cache }` | 按路由启用/关闭缓存，可覆盖阈值等参数 | ❌ 保持不变 |

**用户需要同时配置两者才能使用语义缓存：**
1. 在 GLOBAL 中声明缓存基础设施（存储后端地址、TTL、淘汰策略等）
2. 在需要缓存的 ROUTE 中添加 `PLUGIN semantic_cache { enabled: true }`

---

## Change Surfaces

根据 `docs/agent/change-surfaces.md`，本次变更涉及：

| Surface | Files | Impact |
|---------|-------|--------|
| `dsl_crd` | compiler.go, decompiler.go, validator.go | DSL 语法解析与生成 |
| `dashboard_config_ui` | dslMutations.ts, builderPageGlobalSettings*.tsx | UI 表单与 DSL 同步 |
| `router_config_contract` | dsl_test.go | 配置行为验证 |

---

## Current vs Target DSL Syntax

### 现有语法 (Deprecated)

```dsl
ROUTER my-router

GLOBAL {
  default_model: "gpt-4"
  strategy: "priority"
}

BACKEND vllm_endpoint my-llm {
  address: "localhost:8080"
}

BACKEND semantic_cache main {
  enabled: true
  backend_type: "redis"
  similarity_threshold: 0.85
  max_entries: 5000
  ttl_seconds: 3600
  eviction_policy: "lru"
}

BACKEND memory mem {
  enabled: true
  auto_store: true
  default_retrieval_limit: 10
  default_similarity_threshold: 0.75
}

BACKEND response_api main {
  enabled: true
  store_backend: "redis"
  ttl_seconds: 3600
  max_responses: 1000
}

BACKEND embedding_model main {
  mmbert_model_path: "/models/mmbert"
  use_cpu: true
}

BACKEND provider_profile openai {
  type: "openai"
  base_url: "https://api.openai.com/v1"
}
```

### 目标语法 (New)

```dsl
ROUTER my-router

GLOBAL {
  default_model: "gpt-4"
  strategy: "priority"

  SEMANTIC_CACHE main {
    enabled: true
    backend_type: "redis"
    similarity_threshold: 0.85
    max_entries: 5000
    ttl_seconds: 3600
    eviction_policy: "lru"
  }

  MEMORY mem {
    enabled: true
    auto_store: true
    default_retrieval_limit: 10
    default_similarity_threshold: 0.75
  }

  RESPONSE_API main {
    enabled: true
    store_backend: "redis"
    ttl_seconds: 3600
    max_responses: 1000
  }

  EMBEDDING_MODEL main {
    mmbert_model_path: "/models/mmbert"
    use_cpu: true
  }

  PROVIDER_PROFILE openai {
    type: "openai"
    base_url: "https://api.openai.com/v1"
  }
}

BACKEND vllm_endpoint my-llm {
  address: "localhost:8080"
}
```

---

## 配置字段映射（与代码实际结构对齐）

### SEMANTIC_CACHE 字段

基于 `config/runtime_config.go` 中 `SemanticCache` 结构：

| DSL 字段 | Go 字段 | 类型 | 说明 |
|----------|---------|------|------|
| `enabled` | `Enabled` | bool | 是否启用 |
| `backend_type` | `BackendType` | string | "redis", "milvus", "memory" |
| `similarity_threshold` | `SimilarityThreshold` | *float32 | 相似度阈值 0-1 |
| `max_entries` | `MaxEntries` | int | 最大缓存条目数 |
| `ttl_seconds` | `TTLSeconds` | int | 缓存过期时间（秒） |
| `eviction_policy` | `EvictionPolicy` | string | 淘汰策略 "lru" 等 |
| `embedding_model` | `EmbeddingModel` | string | 使用的嵌入模型名称 |

### MEMORY 字段

基于 `config/runtime_config.go` 中 `MemoryConfig` 结构：

| DSL 字段 | Go 字段 | 类型 | 说明 |
|----------|---------|------|------|
| `enabled` | `Enabled` | bool | 是否启用 |
| `auto_store` | `AutoStore` | bool | 是否自动存储对话 |
| `embedding_model` | `EmbeddingModel` | string | 使用的嵌入模型 |
| `default_retrieval_limit` | `DefaultRetrievalLimit` | int | 默认检索数量 |
| `default_similarity_threshold` | `DefaultSimilarityThreshold` | float32 | 默认相似度阈值 |
| `hybrid_search` | `HybridSearch` | bool | 是否启用混合搜索 |
| `hybrid_mode` | `HybridMode` | string | 混合搜索模式 |

### RESPONSE_API 字段

基于 `config/runtime_config.go` 中 `ResponseAPIConfig` 结构：

| DSL 字段 | Go 字段 | 类型 | 说明 |
|----------|---------|------|------|
| `enabled` | `Enabled` | bool | 是否启用 |
| `store_backend` | `StoreBackend` | string | 存储后端类型 |
| `ttl_seconds` | `TTLSeconds` | int | 响应过期时间（秒） |
| `max_responses` | `MaxResponses` | int | 最大响应数量 |

### EMBEDDING_MODEL 字段

基于 `config/model_config_types.go` 中 `EmbeddingModels` 结构：

| DSL 字段 | Go 字段 | 类型 | 说明 |
|----------|---------|------|------|
| `mmbert_model_path` | `MmBertModelPath` | string | mmBERT 模型路径 |
| `bert_model_path` | `BertModelPath` | string | BERT 模型路径 |
| `qwen3_model_path` | `Qwen3ModelPath` | string | Qwen3 模型路径 |
| `gemma_model_path` | `GemmaModelPath` | string | Gemma 模型路径 |
| `use_cpu` | `UseCPU` | bool | 是否使用 CPU 推理 |

### PROVIDER_PROFILE 字段

基于 `config/model_config_types.go` 中 `ProviderProfile` 结构：

| DSL 字段 | Go 字段 | 类型 | 说明 |
|----------|---------|------|------|
| `type` | `Type` | string | 提供商类型 (openai, anthropic, azure-openai 等) |
| `base_url` | `BaseURL` | string | API 基础 URL |
| `auth_header` | `AuthHeader` | string | 认证头名称 |
| `auth_prefix` | `AuthPrefix` | string | 认证值前缀 |
| `api_version` | `APIVersion` | string | API 版本（Azure 用） |
| `chat_path` | `ChatPath` | string | Chat 接口路径 |

---

## Implementation Details

### Phase 1: Backend (Go)

#### 1.1 ast.go

**文件路径**: `src/semantic-router/pkg/dsl/ast.go`

**新增 GLOBAL 子块类型常量**:

```go
// GlobalSubBlockType represents nested block types inside GLOBAL
type GlobalSubBlockType string

const (
    GlobalSubBlockSemanticCache  GlobalSubBlockType = "SEMANTIC_CACHE"
    GlobalSubBlockMemory         GlobalSubBlockType = "MEMORY"
    GlobalSubBlockResponseAPI    GlobalSubBlockType = "RESPONSE_API"
    GlobalSubBlockEmbeddingModel GlobalSubBlockType = "EMBEDDING_MODEL"
    GlobalSubBlockProviderProfile GlobalSubBlockType = "PROVIDER_PROFILE"
)

// GlobalSubBlock represents a nested block inside GLOBAL
type GlobalSubBlock struct {
    Type   GlobalSubBlockType
    Name   string
    Fields map[string]Value
    Pos    Position
}
```

**修改 GlobalDecl 结构**:

```go
type GlobalDecl struct {
    Fields    map[string]Value
    SubBlocks []GlobalSubBlock  // 新增：嵌套子块列表
    Pos       Position
}
```

#### 1.2 parser.go

**文件路径**: `src/semantic-router/pkg/dsl/parser.go`

**修改 parseGlobal 函数以支持嵌套块解析**:

```go
func (p *Parser) parseGlobal() *GlobalDecl {
    g := &GlobalDecl{
        Fields:    make(map[string]Value),
        SubBlocks: []GlobalSubBlock{},
        Pos:       p.pos(),
    }

    p.expect(LBRACE)

    for !p.check(RBRACE) && !p.isAtEnd() {
        // 检查是否是嵌套子块
        if p.isGlobalSubBlockType(p.peek().Literal) {
            subBlock := p.parseGlobalSubBlock()
            g.SubBlocks = append(g.SubBlocks, subBlock)
            continue
        }

        // 普通字段解析（现有逻辑）
        key := p.expect(IDENT).Literal
        p.expect(COLON)
        value := p.parseValue()
        g.Fields[key] = value

        // 可选逗号
        p.match(COMMA)
    }

    p.expect(RBRACE)
    return g
}

func (p *Parser) isGlobalSubBlockType(literal string) bool {
    switch strings.ToUpper(literal) {
    case "SEMANTIC_CACHE", "MEMORY", "RESPONSE_API", "EMBEDDING_MODEL", "PROVIDER_PROFILE":
        return true
    }
    return false
}

func (p *Parser) parseGlobalSubBlock() GlobalSubBlock {
    typeTok := p.advance()
    blockType := GlobalSubBlockType(strings.ToUpper(typeTok.Literal))

    // 名称是可选的
    var name string
    if p.check(IDENT) && !p.check(LBRACE) {
        name = p.advance().Literal
    } else {
        name = "default"
    }

    p.expect(LBRACE)

    fields := make(map[string]Value)
    for !p.check(RBRACE) && !p.isAtEnd() {
        key := p.expect(IDENT).Literal
        p.expect(COLON)
        value := p.parseValue()
        fields[key] = value
        p.match(COMMA)
    }

    p.expect(RBRACE)

    return GlobalSubBlock{
        Type:   blockType,
        Name:   name,
        Fields: fields,
        Pos:    p.pos(),
    }
}
```

#### 1.3 compiler.go

**文件路径**: `src/semantic-router/pkg/dsl/compiler.go`

**修改 compileGlobal 函数**:

```go
func (c *Compiler) compileGlobal() {
    g := c.prog.Global.Fields

    // --- 现有字段处理（保持不变） ---
    if obj, ok := g["listeners"]; ok {
        c.config.Listeners = c.compileListeners(obj, c.prog.Global.Pos)
    }
    if v, ok := getStringField(g, "default_model"); ok {
        c.config.DefaultModel = v
    }
    // ... 其他现有字段 ...

    // --- 新增：处理嵌套子块 ---
    for _, sub := range c.prog.Global.SubBlocks {
        switch sub.Type {
        case GlobalSubBlockSemanticCache:
            c.compileSemanticCacheFromGlobal(sub)
        case GlobalSubBlockMemory:
            c.compileMemoryFromGlobal(sub)
        case GlobalSubBlockResponseAPI:
            c.compileResponseAPIFromGlobal(sub)
        case GlobalSubBlockEmbeddingModel:
            c.compileEmbeddingModelFromGlobal(sub)
        case GlobalSubBlockProviderProfile:
            c.compileProviderProfileFromGlobal(sub)
        }
    }
}

// compileSemanticCacheFromGlobal 编译 GLOBAL 内的 SEMANTIC_CACHE 子块
func (c *Compiler) compileSemanticCacheFromGlobal(sub GlobalSubBlock) {
    // 复用现有 compileSemanticCacheBackend 的字段处理逻辑
    if v, ok := getBoolField(sub.Fields, "enabled"); ok {
        c.config.SemanticCache.Enabled = v
    }
    if v, ok := getStringField(sub.Fields, "backend_type"); ok {
        c.config.SemanticCache.BackendType = v
    }
    if v, ok := getFloat32Field(sub.Fields, "similarity_threshold"); ok {
        c.config.SemanticCache.SimilarityThreshold = &v
    }
    if v, ok := getIntField(sub.Fields, "max_entries"); ok {
        c.config.SemanticCache.MaxEntries = v
    }
    if v, ok := getIntField(sub.Fields, "ttl_seconds"); ok {
        c.config.SemanticCache.TTLSeconds = v
    }
    if v, ok := getStringField(sub.Fields, "eviction_policy"); ok {
        c.config.SemanticCache.EvictionPolicy = v
    }
    if v, ok := getStringField(sub.Fields, "embedding_model"); ok {
        c.config.SemanticCache.EmbeddingModel = v
    }
}

// compileMemoryFromGlobal 编译 GLOBAL 内的 MEMORY 子块
func (c *Compiler) compileMemoryFromGlobal(sub GlobalSubBlock) {
    if v, ok := getBoolField(sub.Fields, "enabled"); ok {
        c.config.Memory.Enabled = v
    }
    if v, ok := getBoolField(sub.Fields, "auto_store"); ok {
        c.config.Memory.AutoStore = v
    }
    if v, ok := getIntField(sub.Fields, "default_retrieval_limit"); ok {
        c.config.Memory.DefaultRetrievalLimit = v
    }
    if v, ok := getFloat32Field(sub.Fields, "default_similarity_threshold"); ok {
        c.config.Memory.DefaultSimilarityThreshold = v
    }
    if v, ok := getStringField(sub.Fields, "embedding_model"); ok {
        c.config.Memory.EmbeddingModel = v
    }
    if v, ok := getBoolField(sub.Fields, "hybrid_search"); ok {
        c.config.Memory.HybridSearch = v
    }
    if v, ok := getStringField(sub.Fields, "hybrid_mode"); ok {
        c.config.Memory.HybridMode = v
    }
}

// compileResponseAPIFromGlobal 编译 GLOBAL 内的 RESPONSE_API 子块
func (c *Compiler) compileResponseAPIFromGlobal(sub GlobalSubBlock) {
    if v, ok := getBoolField(sub.Fields, "enabled"); ok {
        c.config.ResponseAPI.Enabled = v
    }
    if v, ok := getStringField(sub.Fields, "store_backend"); ok {
        c.config.ResponseAPI.StoreBackend = v
    }
    if v, ok := getIntField(sub.Fields, "ttl_seconds"); ok {
        c.config.ResponseAPI.TTLSeconds = v
    }
    if v, ok := getIntField(sub.Fields, "max_responses"); ok {
        c.config.ResponseAPI.MaxResponses = v
    }
}

// compileEmbeddingModelFromGlobal 编译 GLOBAL 内的 EMBEDDING_MODEL 子块
func (c *Compiler) compileEmbeddingModelFromGlobal(sub GlobalSubBlock) {
    if v, ok := getStringField(sub.Fields, "mmbert_model_path"); ok {
        c.config.EmbeddingModels.MmBertModelPath = v
    }
    if v, ok := getStringField(sub.Fields, "bert_model_path"); ok {
        c.config.EmbeddingModels.BertModelPath = v
    }
    if v, ok := getStringField(sub.Fields, "qwen3_model_path"); ok {
        c.config.EmbeddingModels.Qwen3ModelPath = v
    }
    if v, ok := getStringField(sub.Fields, "gemma_model_path"); ok {
        c.config.EmbeddingModels.GemmaModelPath = v
    }
    if v, ok := getBoolField(sub.Fields, "use_cpu"); ok {
        c.config.EmbeddingModels.UseCPU = v
    }
}

// compileProviderProfileFromGlobal 编译 GLOBAL 内的 PROVIDER_PROFILE 子块
func (c *Compiler) compileProviderProfileFromGlobal(sub GlobalSubBlock) {
    profile := config.ProviderProfile{}

    if v, ok := getStringField(sub.Fields, "type"); ok {
        profile.Type = v
    }
    if v, ok := getStringField(sub.Fields, "base_url"); ok {
        profile.BaseURL = v
    }
    if v, ok := getStringField(sub.Fields, "auth_header"); ok {
        profile.AuthHeader = v
    }
    if v, ok := getStringField(sub.Fields, "auth_prefix"); ok {
        profile.AuthPrefix = v
    }
    if v, ok := getStringField(sub.Fields, "api_version"); ok {
        profile.APIVersion = v
    }
    if v, ok := getStringField(sub.Fields, "chat_path"); ok {
        profile.ChatPath = v
    }

    if c.config.ProviderProfiles == nil {
        c.config.ProviderProfiles = make(map[string]config.ProviderProfile)
    }
    c.config.ProviderProfiles[sub.Name] = profile
}
```

#### 1.4 decompiler.go

**文件路径**: `src/semantic-router/pkg/dsl/decompiler.go`

**修改 decompile 函数，将特性配置输出到 GLOBAL 内**:

```go
func (d *decompiler) decompile() string {
    d.write("ROUTER %s\n\n", d.cfg.ConfigSource.RouterName)

    // 输出 GLOBAL（包含嵌套子块）
    d.decompileGlobalWithFeatures()

    // 只输出 vllm_endpoint 类型的 BACKEND
    d.decompileVLLMEndpoints()

    // ... 其他块（SIGNAL, ROUTE 等）...

    return d.sb.String()
}

func (d *decompiler) decompileGlobalWithFeatures() {
    d.write("GLOBAL {\n")

    // 基础字段
    if d.cfg.DefaultModel != "" {
        d.write("  default_model: %q\n", d.cfg.DefaultModel)
    }
    if d.cfg.Strategy != "" {
        d.write("  strategy: %q\n", d.cfg.Strategy)
    }
    // ... 其他基础字段 ...

    // 嵌套子块
    d.decompileSemanticCacheSubBlock()
    d.decompileMemorySubBlock()
    d.decompileResponseAPISubBlock()
    d.decompileEmbeddingModelSubBlock()
    d.decompileProviderProfileSubBlocks()

    d.write("}\n\n")
}

func (d *decompiler) decompileSemanticCacheSubBlock() {
    sc := d.cfg.SemanticCache
    if !sc.Enabled && sc.BackendType == "" && sc.MaxEntries == 0 {
        return // 没有配置，不输出
    }

    d.write("\n  SEMANTIC_CACHE main {\n")
    if sc.Enabled {
        d.write("    enabled: true\n")
    }
    if sc.BackendType != "" {
        d.write("    backend_type: %q\n", sc.BackendType)
    }
    if sc.SimilarityThreshold != nil {
        d.write("    similarity_threshold: %v\n", *sc.SimilarityThreshold)
    }
    if sc.MaxEntries != 0 {
        d.write("    max_entries: %d\n", sc.MaxEntries)
    }
    if sc.TTLSeconds != 0 {
        d.write("    ttl_seconds: %d\n", sc.TTLSeconds)
    }
    if sc.EvictionPolicy != "" {
        d.write("    eviction_policy: %q\n", sc.EvictionPolicy)
    }
    if sc.EmbeddingModel != "" {
        d.write("    embedding_model: %q\n", sc.EmbeddingModel)
    }
    d.write("  }\n")
}

func (d *decompiler) decompileMemorySubBlock() {
    mem := d.cfg.Memory
    if !mem.Enabled && !mem.AutoStore && mem.DefaultRetrievalLimit == 0 {
        return
    }

    d.write("\n  MEMORY mem {\n")
    if mem.Enabled {
        d.write("    enabled: true\n")
    }
    if mem.AutoStore {
        d.write("    auto_store: true\n")
    }
    if mem.DefaultRetrievalLimit != 0 {
        d.write("    default_retrieval_limit: %d\n", mem.DefaultRetrievalLimit)
    }
    if mem.DefaultSimilarityThreshold != 0 {
        d.write("    default_similarity_threshold: %v\n", mem.DefaultSimilarityThreshold)
    }
    if mem.EmbeddingModel != "" {
        d.write("    embedding_model: %q\n", mem.EmbeddingModel)
    }
    if mem.HybridSearch {
        d.write("    hybrid_search: true\n")
    }
    if mem.HybridMode != "" {
        d.write("    hybrid_mode: %q\n", mem.HybridMode)
    }
    d.write("  }\n")
}

func (d *decompiler) decompileResponseAPISubBlock() {
    api := d.cfg.ResponseAPI
    if !api.Enabled && api.StoreBackend == "" && api.TTLSeconds == 0 {
        return
    }

    d.write("\n  RESPONSE_API main {\n")
    if api.Enabled {
        d.write("    enabled: true\n")
    }
    if api.StoreBackend != "" {
        d.write("    store_backend: %q\n", api.StoreBackend)
    }
    if api.TTLSeconds != 0 {
        d.write("    ttl_seconds: %d\n", api.TTLSeconds)
    }
    if api.MaxResponses != 0 {
        d.write("    max_responses: %d\n", api.MaxResponses)
    }
    d.write("  }\n")
}

func (d *decompiler) decompileEmbeddingModelSubBlock() {
    em := d.cfg.EmbeddingModels
    if em.MmBertModelPath == "" && em.BertModelPath == "" && !em.UseCPU {
        return
    }

    d.write("\n  EMBEDDING_MODEL main {\n")
    if em.MmBertModelPath != "" {
        d.write("    mmbert_model_path: %q\n", em.MmBertModelPath)
    }
    if em.BertModelPath != "" {
        d.write("    bert_model_path: %q\n", em.BertModelPath)
    }
    if em.Qwen3ModelPath != "" {
        d.write("    qwen3_model_path: %q\n", em.Qwen3ModelPath)
    }
    if em.GemmaModelPath != "" {
        d.write("    gemma_model_path: %q\n", em.GemmaModelPath)
    }
    if em.UseCPU {
        d.write("    use_cpu: true\n")
    }
    d.write("  }\n")
}

func (d *decompiler) decompileProviderProfileSubBlocks() {
    for name, profile := range d.cfg.ProviderProfiles {
        d.write("\n  PROVIDER_PROFILE %s {\n", name)
        if profile.Type != "" {
            d.write("    type: %q\n", profile.Type)
        }
        if profile.BaseURL != "" {
            d.write("    base_url: %q\n", profile.BaseURL)
        }
        if profile.AuthHeader != "" {
            d.write("    auth_header: %q\n", profile.AuthHeader)
        }
        if profile.AuthPrefix != "" {
            d.write("    auth_prefix: %q\n", profile.AuthPrefix)
        }
        if profile.APIVersion != "" {
            d.write("    api_version: %q\n", profile.APIVersion)
        }
        if profile.ChatPath != "" {
            d.write("    chat_path: %q\n", profile.ChatPath)
        }
        d.write("  }\n")
    }
}

func (d *decompiler) decompileVLLMEndpoints() {
    for _, ep := range d.cfg.VLLMEndpoints {
        d.write("BACKEND vllm_endpoint %s {\n", ep.Name)
        if ep.Address != "" {
            d.write("  address: %q\n", ep.Address)
        }
        if ep.Port != 0 {
            d.write("  port: %d\n", ep.Port)
        }
        if ep.Weight != 0 {
            d.write("  weight: %d\n", ep.Weight)
        }
        if ep.Model != "" {
            d.write("  model: %q\n", ep.Model)
        }
        if ep.ProviderProfileName != "" {
            d.write("  provider_profile: %q\n", ep.ProviderProfileName)
        }
        d.write("}\n\n")
    }
}
```

#### 1.5 validator.go

**文件路径**: `src/semantic-router/pkg/dsl/validator.go`

**新增废弃警告**:

```go
// deprecatedBackendTypes 定义已废弃的 BACKEND 类型及其迁移说明
var deprecatedBackendTypes = map[string]string{
    "semantic_cache":   "Use SEMANTIC_CACHE block inside GLOBAL instead",
    "memory":           "Use MEMORY block inside GLOBAL instead",
    "response_api":     "Use RESPONSE_API block inside GLOBAL instead",
    "embedding_model":  "Use EMBEDDING_MODEL block inside GLOBAL instead",
    "provider_profile": "Use PROVIDER_PROFILE block inside GLOBAL instead",
}

func (v *Validator) validateBackends() {
    for _, b := range v.prog.Backends {
        // 检查废弃类型
        if msg, deprecated := deprecatedBackendTypes[b.BackendType]; deprecated {
            v.addWarning(b.Pos, "BACKEND type %q is deprecated: %s", b.BackendType, msg)
        }

        // 其他验证逻辑...
    }
}
```

#### 1.6 dsl_test.go

**文件路径**: `src/semantic-router/pkg/dsl/dsl_test.go`

**新增测试用例**:

```go
// ---------- P1-XX: GLOBAL Nested Feature Blocks ----------

func TestCompileGlobalSemanticCacheSubBlock(t *testing.T) {
    input := `
ROUTER test-router

GLOBAL {
  default_model: "gpt-4"

  SEMANTIC_CACHE main {
    enabled: true
    backend_type: "redis"
    similarity_threshold: 0.85
    max_entries: 5000
    ttl_seconds: 3600
    eviction_policy: "lru"
  }
}`

    cfg, errs := Compile(input)
    if len(errs) > 0 {
        t.Fatalf("compile errors: %v", errs)
    }

    // 验证 semantic_cache 配置
    if !cfg.SemanticCache.Enabled {
        t.Error("expected semantic_cache enabled")
    }
    if cfg.SemanticCache.BackendType != "redis" {
        t.Errorf("backend_type = %q, want redis", cfg.SemanticCache.BackendType)
    }
    if cfg.SemanticCache.SimilarityThreshold == nil || *cfg.SemanticCache.SimilarityThreshold != 0.85 {
        t.Error("expected similarity_threshold = 0.85")
    }
    if cfg.SemanticCache.MaxEntries != 5000 {
        t.Errorf("max_entries = %d, want 5000", cfg.SemanticCache.MaxEntries)
    }
    if cfg.SemanticCache.TTLSeconds != 3600 {
        t.Errorf("ttl_seconds = %d, want 3600", cfg.SemanticCache.TTLSeconds)
    }
    if cfg.SemanticCache.EvictionPolicy != "lru" {
        t.Errorf("eviction_policy = %q, want lru", cfg.SemanticCache.EvictionPolicy)
    }
}

func TestCompileGlobalMemorySubBlock(t *testing.T) {
    input := `
ROUTER test-router

GLOBAL {
  MEMORY mem {
    enabled: true
    auto_store: true
    default_retrieval_limit: 10
    default_similarity_threshold: 0.75
  }
}`

    cfg, errs := Compile(input)
    if len(errs) > 0 {
        t.Fatalf("compile errors: %v", errs)
    }

    if !cfg.Memory.Enabled {
        t.Error("expected memory enabled")
    }
    if !cfg.Memory.AutoStore {
        t.Error("expected auto_store")
    }
    if cfg.Memory.DefaultRetrievalLimit != 10 {
        t.Errorf("default_retrieval_limit = %d", cfg.Memory.DefaultRetrievalLimit)
    }
    if cfg.Memory.DefaultSimilarityThreshold != 0.75 {
        t.Errorf("default_similarity_threshold = %v", cfg.Memory.DefaultSimilarityThreshold)
    }
}

func TestCompileGlobalProviderProfileSubBlock(t *testing.T) {
    input := `
ROUTER test-router

GLOBAL {
  PROVIDER_PROFILE openai {
    type: "openai"
    base_url: "https://api.openai.com/v1"
  }

  PROVIDER_PROFILE azure {
    type: "azure-openai"
    base_url: "https://my-resource.openai.azure.com"
    api_version: "2024-02-01"
  }
}`

    cfg, errs := Compile(input)
    if len(errs) > 0 {
        t.Fatalf("compile errors: %v", errs)
    }

    if len(cfg.ProviderProfiles) != 2 {
        t.Fatalf("expected 2 profiles, got %d", len(cfg.ProviderProfiles))
    }

    openai := cfg.ProviderProfiles["openai"]
    if openai.Type != "openai" {
        t.Errorf("openai.type = %q", openai.Type)
    }

    azure := cfg.ProviderProfiles["azure"]
    if azure.Type != "azure-openai" {
        t.Errorf("azure.type = %q", azure.Type)
    }
    if azure.APIVersion != "2024-02-01" {
        t.Errorf("azure.api_version = %q", azure.APIVersion)
    }
}

func TestBackwardCompatibilityDeprecatedBackend(t *testing.T) {
    // 旧语法仍然可以解析，但会产生废弃警告
    input := `
ROUTER test-router

BACKEND semantic_cache old-cache {
  enabled: true
  backend_type: "redis"
}`

    cfg, errs := Compile(input)
    if len(errs) > 0 {
        t.Fatalf("compile errors: %v", errs)
    }

    // 配置应该正确解析
    if !cfg.SemanticCache.Enabled {
        t.Error("expected semantic_cache enabled")
    }

    // 应该有废弃警告（通过 Validate 函数获取）
    warnings := Validate(input)
    hasDeprecationWarning := false
    for _, w := range warnings {
        if strings.Contains(w.Message, "deprecated") {
            hasDeprecationWarning = true
            break
        }
    }
    if !hasDeprecationWarning {
        t.Error("expected deprecation warning for BACKEND semantic_cache")
    }
}

func TestDecompileGlobalSubBlocks(t *testing.T) {
    cfg := &config.RouterConfig{
        ConfigSource: config.ConfigSource{RouterName: "test-router"},
        DefaultModel: "gpt-4",
        SemanticCache: config.SemanticCache{
            Enabled:     true,
            BackendType: "redis",
            MaxEntries:  5000,
            TTLSeconds:  3600,
        },
        Memory: config.MemoryConfig{
            Enabled:   true,
            AutoStore: true,
        },
    }

    output := Decompile(cfg)

    // 验证输出包含 GLOBAL 嵌套子块
    if !strings.Contains(output, "SEMANTIC_CACHE main {") {
        t.Error("expected SEMANTIC_CACHE sub-block in GLOBAL")
    }
    if !strings.Contains(output, "backend_type: \"redis\"") {
        t.Error("expected backend_type in SEMANTIC_CACHE")
    }
    if !strings.Contains(output, "MEMORY mem {") {
        t.Error("expected MEMORY sub-block in GLOBAL")
    }

    // 验证不再输出独立的 BACKEND semantic_cache
    if strings.Contains(output, "BACKEND semantic_cache") {
        t.Error("should not output BACKEND semantic_cache, use GLOBAL sub-block instead")
    }
}

func TestRoundTripGlobalSubBlocks(t *testing.T) {
    input := `
ROUTER test-router

GLOBAL {
  default_model: "gpt-4"

  SEMANTIC_CACHE main {
    enabled: true
    backend_type: "redis"
    max_entries: 5000
    ttl_seconds: 3600
  }

  MEMORY mem {
    enabled: true
    auto_store: true
  }
}

BACKEND vllm_endpoint my-llm {
  address: "localhost:8080"
}`

    // 编译
    cfg1, errs := Compile(input)
    if len(errs) > 0 {
        t.Fatalf("compile errors: %v", errs)
    }

    // 反编译
    output := Decompile(cfg1)

    // 再次编译
    cfg2, errs := Compile(output)
    if len(errs) > 0 {
        t.Fatalf("recompile errors: %v", errs)
    }

    // 验证配置一致
    if cfg1.SemanticCache.BackendType != cfg2.SemanticCache.BackendType {
        t.Errorf("semantic_cache.backend_type mismatch: %q vs %q",
            cfg1.SemanticCache.BackendType, cfg2.SemanticCache.BackendType)
    }
    if cfg1.SemanticCache.MaxEntries != cfg2.SemanticCache.MaxEntries {
        t.Errorf("semantic_cache.max_entries mismatch: %d vs %d",
            cfg1.SemanticCache.MaxEntries, cfg2.SemanticCache.MaxEntries)
    }
    if cfg1.Memory.Enabled != cfg2.Memory.Enabled {
        t.Errorf("memory.enabled mismatch: %v vs %v",
            cfg1.Memory.Enabled, cfg2.Memory.Enabled)
    }
}
```

---

### Phase 2: Frontend (TypeScript/React)

#### 2.1 dslMutations.ts

**文件路径**: `dashboard/frontend/src/lib/dslMutations.ts`

**修改内容**:

```typescript
// 更新 BACKEND_TYPES，只保留真正的端点类型
export const BACKEND_TYPES = ['vllm_endpoint'] as const;

// 废弃的 BACKEND 类型（用于向后兼容解析）
export const DEPRECATED_BACKEND_TYPES = [
  'semantic_cache',
  'memory',
  'response_api',
  'embedding_model',
  'provider_profile',
] as const;

// GLOBAL 内的特性子块类型
export const GLOBAL_FEATURE_TYPES = [
  'SEMANTIC_CACHE',
  'MEMORY',
  'RESPONSE_API',
  'EMBEDDING_MODEL',
  'PROVIDER_PROFILE',
] as const;

export type GlobalFeatureType = (typeof GLOBAL_FEATURE_TYPES)[number];

// 特性字段 Schema（与 Go 代码对齐）
export interface FieldSchema {
  name: string;
  type: 'string' | 'number' | 'boolean';
  required: boolean;
  description: string;
}

export function getGlobalFeatureFieldSchema(featureType: GlobalFeatureType): FieldSchema[] {
  switch (featureType) {
    case 'SEMANTIC_CACHE':
      return [
        { name: 'enabled', type: 'boolean', required: false, description: 'Enable semantic cache' },
        { name: 'backend_type', type: 'string', required: false, description: 'Cache backend type (redis, milvus, memory)' },
        { name: 'similarity_threshold', type: 'number', required: false, description: 'Similarity threshold (0-1)' },
        { name: 'max_entries', type: 'number', required: false, description: 'Maximum cache entries' },
        { name: 'ttl_seconds', type: 'number', required: false, description: 'Cache TTL in seconds' },
        { name: 'eviction_policy', type: 'string', required: false, description: 'Eviction policy (e.g., lru)' },
        { name: 'embedding_model', type: 'string', required: false, description: 'Embedding model name' },
      ];
    case 'MEMORY':
      return [
        { name: 'enabled', type: 'boolean', required: false, description: 'Enable memory' },
        { name: 'auto_store', type: 'boolean', required: false, description: 'Auto-store conversations' },
        { name: 'default_retrieval_limit', type: 'number', required: false, description: 'Default retrieval limit' },
        { name: 'default_similarity_threshold', type: 'number', required: false, description: 'Default similarity threshold' },
        { name: 'embedding_model', type: 'string', required: false, description: 'Embedding model name' },
        { name: 'hybrid_search', type: 'boolean', required: false, description: 'Enable hybrid search' },
        { name: 'hybrid_mode', type: 'string', required: false, description: 'Hybrid search mode' },
      ];
    case 'RESPONSE_API':
      return [
        { name: 'enabled', type: 'boolean', required: false, description: 'Enable Response API' },
        { name: 'store_backend', type: 'string', required: false, description: 'Storage backend type' },
        { name: 'ttl_seconds', type: 'number', required: false, description: 'Response TTL in seconds' },
        { name: 'max_responses', type: 'number', required: false, description: 'Maximum stored responses' },
      ];
    case 'EMBEDDING_MODEL':
      return [
        { name: 'mmbert_model_path', type: 'string', required: false, description: 'mmBERT model path' },
        { name: 'bert_model_path', type: 'string', required: false, description: 'BERT model path' },
        { name: 'qwen3_model_path', type: 'string', required: false, description: 'Qwen3 model path' },
        { name: 'gemma_model_path', type: 'string', required: false, description: 'Gemma model path' },
        { name: 'use_cpu', type: 'boolean', required: false, description: 'Use CPU for inference' },
      ];
    case 'PROVIDER_PROFILE':
      return [
        { name: 'type', type: 'string', required: true, description: 'Provider type (openai, anthropic, azure-openai, etc.)' },
        { name: 'base_url', type: 'string', required: false, description: 'Base URL override' },
        { name: 'auth_header', type: 'string', required: false, description: 'Auth header name' },
        { name: 'auth_prefix', type: 'string', required: false, description: 'Auth value prefix' },
        { name: 'api_version', type: 'string', required: false, description: 'API version (for Azure)' },
        { name: 'chat_path', type: 'string', required: false, description: 'Chat endpoint path' },
      ];
    default:
      return [];
  }
}

// DSL 生成：GLOBAL 块（包含嵌套子块）
export interface GlobalSettings {
  defaultModel?: string;
  strategy?: string;
  semanticCache?: SemanticCacheConfig;
  memory?: MemoryConfig;
  responseApi?: ResponseAPIConfig;
  embeddingModel?: EmbeddingModelConfig;
  providerProfiles?: ProviderProfileConfig[];
}

export interface SemanticCacheConfig {
  name?: string;
  enabled?: boolean;
  backend_type?: string;
  similarity_threshold?: number;
  max_entries?: number;
  ttl_seconds?: number;
  eviction_policy?: string;
  embedding_model?: string;
}

export interface MemoryConfig {
  name?: string;
  enabled?: boolean;
  auto_store?: boolean;
  default_retrieval_limit?: number;
  default_similarity_threshold?: number;
  embedding_model?: string;
  hybrid_search?: boolean;
  hybrid_mode?: string;
}

export interface ResponseAPIConfig {
  name?: string;
  enabled?: boolean;
  store_backend?: string;
  ttl_seconds?: number;
  max_responses?: number;
}

export interface EmbeddingModelConfig {
  name?: string;
  mmbert_model_path?: string;
  bert_model_path?: string;
  qwen3_model_path?: string;
  gemma_model_path?: string;
  use_cpu?: boolean;
}

export interface ProviderProfileConfig {
  name: string;
  type: string;
  base_url?: string;
  auth_header?: string;
  auth_prefix?: string;
  api_version?: string;
  chat_path?: string;
}

export function generateGlobalBlock(settings: GlobalSettings): string {
  let dsl = 'GLOBAL {\n';

  // 基础设置
  if (settings.defaultModel) {
    dsl += `  default_model: "${settings.defaultModel}"\n`;
  }
  if (settings.strategy) {
    dsl += `  strategy: "${settings.strategy}"\n`;
  }

  // 嵌套子块
  if (settings.semanticCache && hasValues(settings.semanticCache)) {
    dsl += generateFeatureSubBlock('SEMANTIC_CACHE', settings.semanticCache.name || 'main', settings.semanticCache);
  }
  if (settings.memory && hasValues(settings.memory)) {
    dsl += generateFeatureSubBlock('MEMORY', settings.memory.name || 'mem', settings.memory);
  }
  if (settings.responseApi && hasValues(settings.responseApi)) {
    dsl += generateFeatureSubBlock('RESPONSE_API', settings.responseApi.name || 'main', settings.responseApi);
  }
  if (settings.embeddingModel && hasValues(settings.embeddingModel)) {
    dsl += generateFeatureSubBlock('EMBEDDING_MODEL', settings.embeddingModel.name || 'main', settings.embeddingModel);
  }
  if (settings.providerProfiles) {
    for (const profile of settings.providerProfiles) {
      dsl += generateFeatureSubBlock('PROVIDER_PROFILE', profile.name, profile);
    }
  }

  dsl += '}\n';
  return dsl;
}

function generateFeatureSubBlock(type: GlobalFeatureType, name: string, config: Record<string, unknown>): string {
  let block = `\n  ${type} ${name} {\n`;

  const schema = getGlobalFeatureFieldSchema(type);
  for (const field of schema) {
    const value = config[field.name];
    if (value !== undefined && value !== null && value !== '' && field.name !== 'name') {
      if (field.type === 'boolean') {
        block += `    ${field.name}: ${value}\n`;
      } else if (field.type === 'number') {
        block += `    ${field.name}: ${value}\n`;
      } else {
        block += `    ${field.name}: "${value}"\n`;
      }
    }
  }

  block += '  }\n';
  return block;
}

function hasValues(obj: Record<string, unknown>): boolean {
  for (const [key, value] of Object.entries(obj)) {
    if (key !== 'name' && value !== undefined && value !== null && value !== '') {
      return true;
    }
  }
  return false;
}
```

#### 2.2 新建 builderPageGlobalSettingsFeatureSections.tsx

**文件路径**: `dashboard/frontend/src/pages/builderPageGlobalSettingsFeatureSections.tsx`

```tsx
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Plus, Trash2 } from 'lucide-react';
import type {
  SemanticCacheConfig,
  MemoryConfig,
  ResponseAPIConfig,
  EmbeddingModelConfig,
  ProviderProfileConfig,
} from '@/lib/dslMutations';

// Semantic Cache Section
interface SemanticCacheSectionProps {
  value: SemanticCacheConfig;
  onChange: (value: SemanticCacheConfig) => void;
}

export function SemanticCacheSection({ value, onChange }: SemanticCacheSectionProps) {
  const handleChange = <K extends keyof SemanticCacheConfig>(field: K, fieldValue: SemanticCacheConfig[K]) => {
    onChange({ ...value, [field]: fieldValue });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Semantic Cache</CardTitle>
        <p className="text-sm text-muted-foreground">
          Configure the cache infrastructure. Enable per-route via PLUGIN semantic_cache in ROUTE blocks.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center space-x-2">
          <Switch
            id="cache-enabled"
            checked={value.enabled || false}
            onCheckedChange={(checked) => handleChange('enabled', checked)}
          />
          <Label htmlFor="cache-enabled">Enabled</Label>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="cache-backend">Backend Type</Label>
            <Select
              value={value.backend_type || ''}
              onValueChange={(v) => handleChange('backend_type', v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select backend" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="redis">Redis</SelectItem>
                <SelectItem value="milvus">Milvus</SelectItem>
                <SelectItem value="memory">In-Memory</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="cache-threshold">Similarity Threshold</Label>
            <Input
              id="cache-threshold"
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={value.similarity_threshold ?? ''}
              onChange={(e) => handleChange('similarity_threshold', parseFloat(e.target.value) || undefined)}
              placeholder="0.85"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="cache-entries">Max Entries</Label>
            <Input
              id="cache-entries"
              type="number"
              value={value.max_entries ?? ''}
              onChange={(e) => handleChange('max_entries', parseInt(e.target.value) || undefined)}
              placeholder="5000"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="cache-ttl">TTL (seconds)</Label>
            <Input
              id="cache-ttl"
              type="number"
              value={value.ttl_seconds ?? ''}
              onChange={(e) => handleChange('ttl_seconds', parseInt(e.target.value) || undefined)}
              placeholder="3600"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="cache-eviction">Eviction Policy</Label>
            <Select
              value={value.eviction_policy || ''}
              onValueChange={(v) => handleChange('eviction_policy', v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select policy" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="lru">LRU</SelectItem>
                <SelectItem value="lfu">LFU</SelectItem>
                <SelectItem value="fifo">FIFO</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="cache-embedding">Embedding Model</Label>
            <Input
              id="cache-embedding"
              value={value.embedding_model || ''}
              onChange={(e) => handleChange('embedding_model', e.target.value)}
              placeholder="main"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Memory Section
interface MemorySectionProps {
  value: MemoryConfig;
  onChange: (value: MemoryConfig) => void;
}

export function MemorySection({ value, onChange }: MemorySectionProps) {
  const handleChange = <K extends keyof MemoryConfig>(field: K, fieldValue: MemoryConfig[K]) => {
    onChange({ ...value, [field]: fieldValue });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Memory</CardTitle>
        <p className="text-sm text-muted-foreground">
          Configure conversation memory storage. Enable per-route via PLUGIN memory in ROUTE blocks.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Switch
              id="mem-enabled"
              checked={value.enabled || false}
              onCheckedChange={(checked) => handleChange('enabled', checked)}
            />
            <Label htmlFor="mem-enabled">Enabled</Label>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="mem-autostore"
              checked={value.auto_store || false}
              onCheckedChange={(checked) => handleChange('auto_store', checked)}
            />
            <Label htmlFor="mem-autostore">Auto Store</Label>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="mem-hybrid"
              checked={value.hybrid_search || false}
              onCheckedChange={(checked) => handleChange('hybrid_search', checked)}
            />
            <Label htmlFor="mem-hybrid">Hybrid Search</Label>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="mem-limit">Default Retrieval Limit</Label>
            <Input
              id="mem-limit"
              type="number"
              value={value.default_retrieval_limit ?? ''}
              onChange={(e) => handleChange('default_retrieval_limit', parseInt(e.target.value) || undefined)}
              placeholder="10"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="mem-threshold">Default Similarity Threshold</Label>
            <Input
              id="mem-threshold"
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={value.default_similarity_threshold ?? ''}
              onChange={(e) => handleChange('default_similarity_threshold', parseFloat(e.target.value) || undefined)}
              placeholder="0.75"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="mem-embedding">Embedding Model</Label>
            <Input
              id="mem-embedding"
              value={value.embedding_model || ''}
              onChange={(e) => handleChange('embedding_model', e.target.value)}
              placeholder="main"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="mem-hybrid-mode">Hybrid Mode</Label>
            <Input
              id="mem-hybrid-mode"
              value={value.hybrid_mode || ''}
              onChange={(e) => handleChange('hybrid_mode', e.target.value)}
              placeholder="rrf"
              disabled={!value.hybrid_search}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Response API Section
interface ResponseAPISectionProps {
  value: ResponseAPIConfig;
  onChange: (value: ResponseAPIConfig) => void;
}

export function ResponseAPISection({ value, onChange }: ResponseAPISectionProps) {
  const handleChange = <K extends keyof ResponseAPIConfig>(field: K, fieldValue: ResponseAPIConfig[K]) => {
    onChange({ ...value, [field]: fieldValue });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Response API</CardTitle>
        <p className="text-sm text-muted-foreground">
          Configure response storage for async retrieval.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center space-x-2">
          <Switch
            id="api-enabled"
            checked={value.enabled || false}
            onCheckedChange={(checked) => handleChange('enabled', checked)}
          />
          <Label htmlFor="api-enabled">Enabled</Label>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="api-backend">Store Backend</Label>
            <Select
              value={value.store_backend || ''}
              onValueChange={(v) => handleChange('store_backend', v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select backend" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="redis">Redis</SelectItem>
                <SelectItem value="milvus">Milvus</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="api-ttl">TTL (seconds)</Label>
            <Input
              id="api-ttl"
              type="number"
              value={value.ttl_seconds ?? ''}
              onChange={(e) => handleChange('ttl_seconds', parseInt(e.target.value) || undefined)}
              placeholder="3600"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="api-max">Max Responses</Label>
            <Input
              id="api-max"
              type="number"
              value={value.max_responses ?? ''}
              onChange={(e) => handleChange('max_responses', parseInt(e.target.value) || undefined)}
              placeholder="1000"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Embedding Model Section
interface EmbeddingModelSectionProps {
  value: EmbeddingModelConfig;
  onChange: (value: EmbeddingModelConfig) => void;
}

export function EmbeddingModelSection({ value, onChange }: EmbeddingModelSectionProps) {
  const handleChange = <K extends keyof EmbeddingModelConfig>(field: K, fieldValue: EmbeddingModelConfig[K]) => {
    onChange({ ...value, [field]: fieldValue });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Embedding Model</CardTitle>
        <p className="text-sm text-muted-foreground">
          Configure embedding models for semantic features.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center space-x-2">
          <Switch
            id="embed-cpu"
            checked={value.use_cpu || false}
            onCheckedChange={(checked) => handleChange('use_cpu', checked)}
          />
          <Label htmlFor="embed-cpu">Use CPU</Label>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="embed-mmbert">mmBERT Model Path</Label>
            <Input
              id="embed-mmbert"
              value={value.mmbert_model_path || ''}
              onChange={(e) => handleChange('mmbert_model_path', e.target.value)}
              placeholder="/models/mmbert"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="embed-bert">BERT Model Path</Label>
            <Input
              id="embed-bert"
              value={value.bert_model_path || ''}
              onChange={(e) => handleChange('bert_model_path', e.target.value)}
              placeholder="/models/bert"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="embed-qwen3">Qwen3 Model Path</Label>
            <Input
              id="embed-qwen3"
              value={value.qwen3_model_path || ''}
              onChange={(e) => handleChange('qwen3_model_path', e.target.value)}
              placeholder="/models/qwen3"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="embed-gemma">Gemma Model Path</Label>
            <Input
              id="embed-gemma"
              value={value.gemma_model_path || ''}
              onChange={(e) => handleChange('gemma_model_path', e.target.value)}
              placeholder="/models/gemma"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Provider Profiles Section (multiple)
interface ProviderProfilesSectionProps {
  value: ProviderProfileConfig[];
  onChange: (value: ProviderProfileConfig[]) => void;
}

export function ProviderProfilesSection({ value, onChange }: ProviderProfilesSectionProps) {
  const profiles = value || [];

  const handleAdd = () => {
    onChange([...profiles, { name: '', type: 'openai' }]);
  };

  const handleRemove = (index: number) => {
    onChange(profiles.filter((_, i) => i !== index));
  };

  const handleChange = <K extends keyof ProviderProfileConfig>(
    index: number,
    field: K,
    fieldValue: ProviderProfileConfig[K]
  ) => {
    const updated = [...profiles];
    updated[index] = { ...updated[index], [field]: fieldValue };
    onChange(updated);
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-lg">Provider Profiles</CardTitle>
          <p className="text-sm text-muted-foreground">
            Configure API providers for vLLM endpoints.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={handleAdd}>
          <Plus className="h-4 w-4 mr-1" /> Add Profile
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {profiles.map((profile, index) => (
          <div key={index} className="border rounded-lg p-4 space-y-4">
            <div className="flex justify-between items-center">
              <span className="font-medium">Profile {index + 1}</span>
              <Button variant="ghost" size="sm" onClick={() => handleRemove(index)}>
                <Trash2 className="h-4 w-4 text-red-500" />
              </Button>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Name</Label>
                <Input
                  value={profile.name}
                  onChange={(e) => handleChange(index, 'name', e.target.value)}
                  placeholder="openai-profile"
                />
              </div>

              <div className="space-y-2">
                <Label>Type</Label>
                <Select
                  value={profile.type}
                  onValueChange={(v) => handleChange(index, 'type', v)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="anthropic">Anthropic</SelectItem>
                    <SelectItem value="azure-openai">Azure OpenAI</SelectItem>
                    <SelectItem value="bedrock">AWS Bedrock</SelectItem>
                    <SelectItem value="gemini">Google Gemini</SelectItem>
                    <SelectItem value="vertex-ai">Vertex AI</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Base URL</Label>
                <Input
                  value={profile.base_url || ''}
                  onChange={(e) => handleChange(index, 'base_url', e.target.value)}
                  placeholder="https://api.openai.com/v1"
                />
              </div>

              <div className="space-y-2">
                <Label>API Version</Label>
                <Input
                  value={profile.api_version || ''}
                  onChange={(e) => handleChange(index, 'api_version', e.target.value)}
                  placeholder="2024-02-01"
                  disabled={profile.type !== 'azure-openai'}
                />
              </div>
            </div>
          </div>
        ))}

        {profiles.length === 0 && (
          <p className="text-sm text-muted-foreground text-center py-4">
            No provider profiles configured. Click "Add Profile" to add one.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// Combined export
export interface GlobalSettingsFeatureSectionsProps {
  semanticCache: SemanticCacheConfig;
  memory: MemoryConfig;
  responseApi: ResponseAPIConfig;
  embeddingModel: EmbeddingModelConfig;
  providerProfiles: ProviderProfileConfig[];
  onSemanticCacheChange: (value: SemanticCacheConfig) => void;
  onMemoryChange: (value: MemoryConfig) => void;
  onResponseApiChange: (value: ResponseAPIConfig) => void;
  onEmbeddingModelChange: (value: EmbeddingModelConfig) => void;
  onProviderProfilesChange: (value: ProviderProfileConfig[]) => void;
}

export function GlobalSettingsFeatureSections({
  semanticCache,
  memory,
  responseApi,
  embeddingModel,
  providerProfiles,
  onSemanticCacheChange,
  onMemoryChange,
  onResponseApiChange,
  onEmbeddingModelChange,
  onProviderProfilesChange,
}: GlobalSettingsFeatureSectionsProps) {
  return (
    <div className="space-y-6">
      <SemanticCacheSection value={semanticCache} onChange={onSemanticCacheChange} />
      <MemorySection value={memory} onChange={onMemoryChange} />
      <ResponseAPISection value={responseApi} onChange={onResponseApiChange} />
      <EmbeddingModelSection value={embeddingModel} onChange={onEmbeddingModelChange} />
      <ProviderProfilesSection value={providerProfiles} onChange={onProviderProfilesChange} />
    </div>
  );
}
```

---

## Verification Flow

### Step 1: Fast Gate (本地快速验证)

```bash
# Go 测试
cd src/semantic-router
go test ./pkg/dsl/... -run "TestCompileGlobal|TestDecompileGlobal|TestBackwardCompat|TestRoundTrip" -v

# 前端检查
cd dashboard/frontend
npm run typecheck
npm run lint
```

### Step 2: Local Smoke Test

```bash
# 启动本地开发环境
make vllm-sr-dev
vllm-sr serve --image-pull-policy never

# 使用新语法配置文件测试
cat > /tmp/test-global-features.dsl << 'EOF'
ROUTER test-router

GLOBAL {
  default_model: "gpt-4"

  SEMANTIC_CACHE main {
    enabled: true
    backend_type: "redis"
    max_entries: 5000
    ttl_seconds: 3600
  }

  MEMORY mem {
    enabled: true
    auto_store: true
  }

  PROVIDER_PROFILE openai {
    type: "openai"
    base_url: "https://api.openai.com/v1"
  }
}

BACKEND vllm_endpoint my-llm {
  address: "localhost:8080"
  provider_profile: "openai"
}
EOF

# 验证配置可以被正确解析
vllm-sr config validate /tmp/test-global-features.dsl
```

### Step 3: E2E Tests

```bash
# 运行受影响的 E2E 测试
make agent-e2e-affected CHANGED_FILES="src/semantic-router/pkg/dsl/compiler.go src/semantic-router/pkg/dsl/decompiler.go src/semantic-router/pkg/dsl/parser.go"
```

---

## Backward Compatibility

1. **旧语法继续支持**: `BACKEND semantic_cache` 等旧语法仍可解析，产生 deprecation 警告
2. **Decompiler 输出新语法**: 无论输入是旧语法还是新语法，decompiler 都输出新语法格式
3. **配置等效**: 旧语法和新语法编译后产生相同的 `RouterConfig` 结构
4. **Dashboard 双向兼容**: 前端可以解析旧配置，但生成新配置时使用新语法

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `src/semantic-router/pkg/dsl/ast.go` | Modify | 添加 GlobalSubBlock 类型 |
| `src/semantic-router/pkg/dsl/parser.go` | Modify | 解析 GLOBAL 内嵌套子块 |
| `src/semantic-router/pkg/dsl/compiler.go` | Modify | 编译 GLOBAL 嵌套子块 |
| `src/semantic-router/pkg/dsl/decompiler.go` | Modify | 输出特性为 GLOBAL 嵌套子块 |
| `src/semantic-router/pkg/dsl/validator.go` | Modify | 添加废弃 BACKEND 类型警告 |
| `src/semantic-router/pkg/dsl/dsl_test.go` | Modify | 添加新语法测试用例 |
| `dashboard/frontend/src/lib/dslMutations.ts` | Modify | 更新类型定义和 DSL 生成 |
| `dashboard/frontend/src/pages/builderPageGlobalSettingsFeatureSections.tsx` | Create | 新建特性配置 UI 组件 |

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 现有配置文件解析失败 | Low | High | 保留旧语法支持，仅发出警告 |
| Dashboard 与后端不同步 | Medium | Medium | 前后端同时修改，通过 E2E 验证 |
| Round-trip 不一致 | Medium | High | 添加专门的 round-trip 测试 |
| 用户混淆两层设计 | Medium | Low | 在 UI 中添加说明文字 |

---

## Execution Order

1. **Phase 1A**: 修改 ast.go 和 parser.go（添加 AST 支持）
2. **Phase 1B**: 修改 compiler.go（编译新语法）
3. **Phase 1C**: 修改 decompiler.go（输出新语法）
4. **Phase 1D**: 修改 validator.go（废弃警告）
5. **Phase 1E**: 添加测试用例
6. **Phase 2**: 修改前端代码
7. **Phase 3**: 运行完整 E2E 验证

---

## Related Documentation

- [docs/agent/change-surfaces.md](../change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../feature-complete-checklist.md)
- [tools/agent/task-matrix.yaml](../../tools/agent/task-matrix.yaml)
- [tools/agent/e2e-profile-map.yaml](../../tools/agent/e2e-profile-map.yaml)
