# Signal DSL Model Server

本地部署训练好的模型，提供 **OpenAI 兼容 API**，可直接对接 Dashboard NL Mode。

## 🚀 推荐方案：vLLM

vLLM 是最优雅的部署方式，原生支持 OpenAI API 且高性能。

### 1. 安装 vLLM

```bash
# ROCm (AMD GPU)
pip install vllm

# CUDA (NVIDIA GPU)  
pip install vllm
```

### 2. 启动服务

```bash
# 使用启动脚本（自动检测 LoRA）
./start_vllm.sh ../checkpoints/stage2_sft/checkpoint-120

# 或指定端口和模型名
./start_vllm.sh ../checkpoints/stage2_sft/checkpoint-120 8080 dsl-generator
```

**手动启动（LoRA checkpoint）**:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --enable-lora \
  --lora-modules "dsl-generator=../checkpoints/stage2_sft/checkpoint-120" \
  --served-model-name dsl-generator \
  --port 8080 \
  --host 0.0.0.0
```

### 3. 对接 Dashboard

```bash
export NL_LLM_ENDPOINT="http://<server>:8080/v1/chat/completions"
export NL_LLM_MODEL="dsl-generator"
```

---

## 备选方案：自定义 FastAPI Server

如果 vLLM 不可用，可使用自带的 `server.py`：

```bash
pip install -r requirements.txt
python server.py --checkpoint ../checkpoints/stage2_sft/checkpoint-120
```

---

## API 测试

### 健康检查

```bash
curl http://localhost:8080/health      # 自定义 server
curl http://localhost:8080/v1/models   # vLLM
```

### Chat Completion

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dsl-generator",
    "messages": [
      {"role": "system", "content": "You are a Signal DSL generator."},
      {"role": "user", "content": "处理代码相关的问题"}
    ],
    "temperature": 0.1,
    "max_tokens": 512
  }'
```

### 交互式测试

```bash
python test_client.py
```

---

## 方案对比

| 方案 | 优势 | 劣势 |
|------|------|------|
| **vLLM** ⭐ | 高性能、原生 OpenAI API、支持 LoRA 热加载 | 需要安装额外依赖 |
| **server.py** | 轻量、无额外依赖 | 性能一般、需维护 |
| **TGI** | Docker 部署方便 | 镜像较大 |

---

## 文件说明

```
serve/
├── start_vllm.sh     # vLLM 启动脚本 (推荐)
├── server.py         # 自定义 FastAPI 服务器 (备选)
├── test_client.py    # 测试客户端
├── requirements.txt  # Python 依赖
└── README.md         # 本文档
```

---

## 故障排除

### vLLM 启动失败

1. **ROCm 环境问题**：确保 `rocm-smi` 可用
2. **显存不足**：尝试 `--max-model-len 1024`
3. **LoRA 加载失败**：检查 `adapter_config.json` 存在

### Dashboard 连接失败

1. 确保服务监听 `0.0.0.0`
2. 检查防火墙允许端口
3. 验证 `NL_LLM_ENDPOINT` 环境变量
