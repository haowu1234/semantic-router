#!/usr/bin/env python3
"""
Signal DSL Model Server (OpenAI Compatible)
本地部署训练好的模型，提供 OpenAI 兼容的 API

Usage:
    # 启动服务器
    python server.py --checkpoint ./checkpoints/stage2_sft/checkpoint-120
    
    # 指定端口
    python server.py --checkpoint ./checkpoints/stage2_sft/checkpoint-120 --port 8080
    
API Endpoints:
    POST /v1/chat/completions  - OpenAI 兼容的 chat completion API
    GET  /v1/models            - 列出可用模型
    GET  /health               - 健康检查
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, List, Generator

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread


# ============== Configuration ==============

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are a Signal DSL configuration generator. Convert natural language descriptions into valid Signal DSL configurations."
MODEL_NAME = "dsl-generator"  # 模型名称，用于 API 响应


# ============== OpenAI Compatible Models ==============

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion Request"""
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = Field(default=512, alias="max_tokens")
    stream: Optional[bool] = False
    n: Optional[int] = 1
    stop: Optional[List[str]] = None
    
    class Config:
        populate_by_name = True


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completion Response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI Chat Completion Streaming Chunk"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint: str
    device: str
    base_model: str


# ============== Model Manager ==============

class DSLModelServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.checkpoint_path = None
        self.device = None
        self.base_model_name = None
        self.model_loaded_time = None
        
    def load_model(self, checkpoint_path: str, base_model: str = DEFAULT_BASE_MODEL):
        """加载模型，支持 PEFT checkpoint"""
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        self.checkpoint_path = checkpoint_path
        self.base_model_name = base_model
        
        # 检测设备
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            self.device = "cuda"  # ROCm uses cuda API
            print("Using AMD ROCm")
        else:
            self.device = "cpu"
            print("Using CPU (slow)")
        
        # 加载 tokenizer
        print(f"Loading tokenizer from {base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            padding_side="left"  # 推理时用 left padding
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 检查是否是 PEFT checkpoint
        adapter_config_path = Path(checkpoint_path) / "adapter_config.json"
        is_peft = adapter_config_path.exists()
        
        if is_peft:
            print(f"Detected PEFT checkpoint, loading base model + adapter...")
            
            # 先加载基座模型
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"  # ROCm compatible
            )
            
            # 加载 PEFT adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path,
                is_trainable=False
            )
            print("PEFT adapter loaded successfully")
        else:
            print(f"Loading full model from {checkpoint_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"
            )
        
        self.model.eval()
        self.model_loaded_time = int(time.time())
        print(f"Model loaded successfully on {self.device}")
        
        # 打印显存使用
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory allocated: {allocated:.2f} GB")
    
    def _prepare_inputs(self, messages: List[ChatMessage]):
        """准备模型输入"""
        # 转换消息格式，确保 content 不为 None
        chat_messages = [{"role": m.role, "content": m.content or ""} for m in messages]
        
        # 使用 tokenizer 的 chat template
        try:
            full_prompt = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # 确保是字符串
            if isinstance(full_prompt, list):
                full_prompt = full_prompt[0] if full_prompt else ""
            full_prompt = str(full_prompt)
        except Exception as e:
            # Fallback: 手动构建 prompt
            print(f"Warning: apply_chat_template failed: {e}, using fallback")
            system_msg = next((m["content"] for m in chat_messages if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in chat_messages if m["role"] == "user"), "")
            full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize - 显式传入字符串
        inputs = self.tokenizer(
            text=full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        return inputs
    
    def generate(self, 
                 messages: List[ChatMessage],
                 max_tokens: int = 512,
                 temperature: float = 0.1,
                 top_p: float = 0.9) -> str:
        """非流式生成"""
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        inputs = self._prepare_inputs(messages)
        input_length = inputs["input_ids"].shape[1]
        
        # 非流式生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode (only new tokens)
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def generate_stream(self, 
                        messages: List[ChatMessage],
                        max_tokens: int = 512,
                        temperature: float = 0.1,
                        top_p: float = 0.9) -> Generator[str, None, None]:
        """流式生成"""
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        inputs = self._prepare_inputs(messages)
        
        # 流式生成
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # 在后台线程中运行生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 逐个 token 返回
        for text in streamer:
            yield text
            
        thread.join()


# ============== FastAPI App ==============

app = FastAPI(
    title="Signal DSL Model Server",
    description="OpenAI 兼容的 DSL 生成模型 API",
    version="1.0.0"
)

# CORS - 允许 dashboard 跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model server instance
model_server = DSLModelServer()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="ok" if model_server.model is not None else "model_not_loaded",
        model_loaded=model_server.model is not None,
        checkpoint=model_server.checkpoint_path or "none",
        device=model_server.device or "none",
        base_model=model_server.base_model_name or "none"
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """列出可用模型 (OpenAI 兼容)"""
    models = []
    if model_server.model is not None:
        models.append(ModelInfo(
            id=MODEL_NAME,
            created=model_server.model_loaded_time or int(time.time())
        ))
    return ModelList(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI 兼容的 Chat Completion API
    
    Dashboard NL mode 会调用这个接口
    """
    if model_server.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 手动解析 JSON 请求
    try:
        body = await request.json()
        print(f"[DEBUG] Raw request body: {body}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    # 手动提取字段
    model = body.get("model", MODEL_NAME)
    messages_raw = body.get("messages", [])
    temperature = body.get("temperature", 0.1)
    top_p = body.get("top_p", 0.9)
    max_tokens = body.get("max_tokens", 512)
    stream = body.get("stream", False)
    
    # 转换 messages
    messages = []
    for m in messages_raw:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        messages.append(ChatMessage(role=role, content=content))
    
    print(f"[DEBUG] Parsed messages: {[(m.role, m.content[:50] if m.content else '') for m in messages]}")
    
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    try:
        if stream:
            # 流式响应
            async def generate_stream():
                # 发送初始 chunk (role)
                initial_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=MODEL_NAME,
                    choices=[ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(role="assistant"),
                        finish_reason=None
                    )]
                )
                yield f"data: {initial_chunk.model_dump_json()}\n\n"
                
                # 流式生成内容
                full_content = ""
                for token in model_server.generate_stream(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                ):
                    full_content += token
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        created=created,
                        model=MODEL_NAME,
                        choices=[ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=token),
                            finish_reason=None
                        )]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                
                # 发送结束 chunk
                final_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=MODEL_NAME,
                    choices=[ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop"
                    )]
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 非流式响应
            generated_text = model_server.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Debug: 检查生成结果
            print(f"[DEBUG] generated_text type: {type(generated_text)}")
            print(f"[DEBUG] generated_text value: {repr(generated_text)[:200] if generated_text else 'None'}")
            
            # 确保是字符串
            if generated_text is None:
                generated_text = ""
            elif not isinstance(generated_text, str):
                generated_text = str(generated_text)
            
            # 计算 token 数量 (估算)
            prompt_text = " ".join([m.content or "" for m in messages])
            prompt_tokens = len(model_server.tokenizer.encode(prompt_text, add_special_tokens=False)) if prompt_text else 0
            completion_tokens = len(model_server.tokenizer.encode(generated_text, add_special_tokens=False)) if generated_text else 0
            
            return ChatCompletionResponse(
                id=request_id,
                created=created,
                model=MODEL_NAME,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )],
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
    
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in chat_completions:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============== Legacy API (保持向后兼容) ==============

class LegacyGenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    system_prompt: Optional[str] = None


class LegacyGenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    model_info: dict


@app.post("/generate", response_model=LegacyGenerateResponse)
async def legacy_generate(request: LegacyGenerateRequest):
    """旧版 API (向后兼容)"""
    if model_server.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    messages = [
        ChatMessage(role="system", content=request.system_prompt or DEFAULT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=request.prompt)
    ]
    
    generated_text = model_server.generate(
        messages=messages,
        max_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )
    
    return LegacyGenerateResponse(
        generated_text=generated_text,
        prompt=request.prompt,
        model_info={
            "checkpoint": model_server.checkpoint_path,
            "base_model": model_server.base_model_name,
            "device": model_server.device
        }
    )


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Signal DSL Model Server (OpenAI Compatible)")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (PEFT or full model)"
    )
    parser.add_argument(
        "--base-model", "-b",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model name (default: {DEFAULT_BASE_MODEL})"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    # 验证 checkpoint 路径
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # 加载模型
    model_server.load_model(args.checkpoint, args.base_model)
    
    # 启动服务器
    print(f"\n{'='*60}")
    print(f"Signal DSL Model Server (OpenAI Compatible)")
    print(f"{'='*60}")
    print(f"Server:    http://{args.host}:{args.port}")
    print(f"API Docs:  http://{args.host}:{args.port}/docs")
    print(f"")
    print(f"OpenAI Compatible Endpoints:")
    print(f"  POST /v1/chat/completions  - Chat completion")
    print(f"  GET  /v1/models            - List models")
    print(f"")
    print(f"Dashboard NL Mode Config:")
    print(f"  NL_LLM_ENDPOINT=http://<server>:{args.port}/v1/chat/completions")
    print(f"  NL_LLM_MODEL={MODEL_NAME}")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
