#!/usr/bin/env python3
"""
NL 描述自动生成器

对每个有效 DSL 配置，使用 LLM 生成多种风格的自然语言描述。

支持的后端:
- vllm: 本地 vLLM 服务器 (推荐，支持任意开源模型)
- openai: OpenAI API (GPT-4o)
- anthropic: Anthropic API (Claude)
- local: 本地规则生成 (无需 API，用于测试)

生成风格:
- en_formal: 英文正式 (技术文档风格)
- en_casual: 英文口语 (用户对话风格)  
- en_technical: 英文技术 (工程师风格)
- zh_formal: 中文正式
- zh_casual: 中文口语
- ambiguous: 故意模糊/欠指定的描述 (测试模型推理能力)

用法:
    # 使用本地 vLLM 服务器 (推荐)
    python nl_generator.py --input synthetic/ --output nl_pairs/ --api vllm --vllm-url http://localhost:8000
    
    # 使用远程 vLLM 服务器
    python nl_generator.py --input synthetic/ --output nl_pairs/ --api vllm --vllm-url http://192.168.1.100:8000 --model Qwen/Qwen2.5-72B-Instruct
    
    # 使用 OpenAI API
    python nl_generator.py --input synthetic/ --output nl_pairs/ --api openai --api-key $OPENAI_API_KEY
    
    # 本地规则生成 (无需 API)
    python nl_generator.py --input seeds/from_tests.jsonl --output nl_pairs/test_nl.jsonl --dry-run
"""

import argparse
import json
import os
import time
import hashlib
from pathlib import Path
from typing import Generator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


def detect_vllm_model(vllm_url: str) -> str | None:
    """自动检测 vLLM 服务器上加载的模型"""
    import requests
    
    try:
        response = requests.get(
            f"{vllm_url.rstrip('/')}/v1/models",
            timeout=10
        )
        response.raise_for_status()
        models = response.json().get('data', [])
        if models:
            return models[0].get('id')
    except Exception as e:
        print(f"Warning: Failed to detect vLLM model: {e}")
    return None


def get_processed_ids(output_dir: Path) -> set:
    """获取已处理的样本 ID（用于断点续传）"""
    processed = set()
    if not output_dir.exists():
        return processed
    
    for jsonl_file in output_dir.glob('*.jsonl'):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        source_id = data.get('source_id')
                        if source_id:
                            processed.add(source_id)
        except Exception:
            pass
    return processed


# NL 生成提示词模板
NL_GENERATION_PROMPT = '''You are an expert at describing Signal DSL router configurations in natural language.

Given this DSL configuration, generate 6 different natural language descriptions that a user might say to request this configuration. Each description should be self-contained and provide enough detail to recreate the DSL.

DSL Configuration:
```
{dsl}
```

Generate exactly 6 descriptions in JSON format:
{{
  "en_formal": "Formal English - Technical documentation style, precise and complete",
  "en_casual": "Casual English - Conversational style, like talking to a colleague",
  "en_technical": "Technical English - Engineer style, using DSL terminology",
  "zh_formal": "正式中文 - 技术文档风格，精确完整",
  "zh_casual": "口语中文 - 对话风格，像和同事聊天",
  "ambiguous": "Intentionally underspecified - Missing some details that require inference"
}}

Rules:
1. Each description should capture the INTENT of the configuration
2. Include key parameters like model names, thresholds, signal types
3. The ambiguous description should be missing 1-2 key details
4. Chinese descriptions should use natural Chinese expressions, not word-by-word translation
5. Keep descriptions concise but sufficient to recreate the DSL

Output JSON only, no markdown code blocks.'''

# 简化版提示词 (用于 dry-run 或低成本测试)
NL_SIMPLE_PROMPT = '''Describe this DSL configuration in one sentence:
```
{dsl}
```
Output a single English sentence describing what this configuration does.'''


@dataclass
class NLGenerationConfig:
    """NL 生成配置"""
    model: str = 'gpt-4o'
    temperature: float = 0.8
    max_tokens: int = 2000
    timeout: int = 120
    retry_count: int = 3
    retry_delay: float = 2.0
    rate_limit_delay: float = 0.1  # 请求间隔 (vLLM 本地可以更快)
    vllm_url: str = 'http://localhost:8000'  # vLLM 服务器地址


def generate_nl_vllm(dsl: str, config: NLGenerationConfig) -> dict | None:
    """使用本地 vLLM 服务器生成 NL 描述
    
    vLLM 提供 OpenAI 兼容的 API，支持任意开源模型。
    推荐模型: Qwen2.5-72B-Instruct, Qwen2.5-32B-Instruct, DeepSeek-V3
    
    启动 vLLM 服务器:
        python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen2.5-72B-Instruct \
            --tensor-parallel-size 4 \
            --port 8000
    """
    import requests
    
    url = f"{config.vllm_url.rstrip('/')}/v1/chat/completions"
    
    # 构建 chat 消息
    messages = [
        {"role": "user", "content": NL_GENERATION_PROMPT.format(dsl=dsl)}
    ]
    
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }
    
    for attempt in range(config.retry_count):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=config.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 尝试提取 JSON (模型可能返回 markdown 包裹的 JSON)
            content = extract_json_from_response(content)
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error (attempt {attempt + 1}): {e}")
            print(f"Raw response: {content[:500]}...")
        except requests.exceptions.RequestException as e:
            print(f"HTTP error (attempt {attempt + 1}): {e}")
            if attempt < config.retry_count - 1:
                time.sleep(config.retry_delay * (attempt + 1))
        except Exception as e:
            print(f"Unexpected error (attempt {attempt + 1}): {e}")
            if attempt < config.retry_count - 1:
                time.sleep(config.retry_delay * (attempt + 1))
    
    return None


def extract_json_from_response(content: str) -> str:
    """从 LLM 响应中提取 JSON 内容
    
    处理各种情况:
    - 纯 JSON
    - ```json ... ``` 包裹
    - ``` ... ``` 包裹
    - 前后有额外文本
    """
    content = content.strip()
    
    # 尝试直接解析
    if content.startswith('{') and content.endswith('}'):
        return content
    
    # 提取 ```json ... ``` 块
    if '```json' in content:
        start = content.find('```json') + 7
        end = content.find('```', start)
        if end > start:
            return content[start:end].strip()
    
    # 提取 ``` ... ``` 块
    if '```' in content:
        parts = content.split('```')
        if len(parts) >= 3:
            return parts[1].strip()
    
    # 尝试找到 JSON 对象边界
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    if start_idx != -1 and end_idx > start_idx:
        return content[start_idx:end_idx + 1]
    
    return content


def generate_nl_openai(dsl: str, config: NLGenerationConfig) -> dict | None:
    """使用 OpenAI API 生成 NL 描述"""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        return None
    
    client = OpenAI()
    
    for attempt in range(config.retry_count):
        try:
            response = client.chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                messages=[
                    {"role": "user", "content": NL_GENERATION_PROMPT.format(dsl=dsl)}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error (attempt {attempt + 1}): {e}")
        except Exception as e:
            print(f"API error (attempt {attempt + 1}): {e}")
            if attempt < config.retry_count - 1:
                time.sleep(config.retry_delay * (attempt + 1))
    
    return None


def generate_nl_anthropic(dsl: str, config: NLGenerationConfig) -> dict | None:
    """使用 Anthropic API 生成 NL 描述"""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        return None
    
    client = anthropic.Anthropic()
    
    for attempt in range(config.retry_count):
        try:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=config.max_tokens,
                messages=[
                    {"role": "user", "content": NL_GENERATION_PROMPT.format(dsl=dsl)}
                ]
            )
            
            content = response.content[0].text
            # Claude 可能返回带 markdown 的 JSON，需要提取
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error (attempt {attempt + 1}): {e}")
        except Exception as e:
            print(f"API error (attempt {attempt + 1}): {e}")
            if attempt < config.retry_count - 1:
                time.sleep(config.retry_delay * (attempt + 1))
    
    return None


def generate_nl_local(dsl: str, config: NLGenerationConfig) -> dict:
    """本地规则生成 NL 描述 (用于 dry-run 或无 API 场景)
    
    基于 DSL 结构分析生成模板化描述，质量较低但无需 API 调用。
    """
    import re
    
    # 提取关键信息
    signals = re.findall(r'SIGNAL\s+(\w+)\s+(\w+)', dsl)
    routes = re.findall(r'ROUTE\s+(\w+)', dsl)
    models = re.findall(r'MODEL\s+"([^"]+)"', dsl)
    algorithms = re.findall(r'ALGORITHM\s+(\w+)', dsl)
    plugins = re.findall(r'PLUGIN\s+(\w+)', dsl)
    
    # 构建描述
    signal_desc = ', '.join([f'{t} detection for {n}' for t, n in signals[:3]]) if signals else 'no signals'
    model_desc = ', '.join(models[:2]) if models else 'default model'
    route_desc = f'{len(routes)} route(s)' if routes else 'no routes'
    algo_desc = algorithms[0] if algorithms else 'default'
    
    en_formal = f"Configure a router with {signal_desc}. Route traffic using {model_desc} with {algo_desc} algorithm. {route_desc} defined."
    en_casual = f"Set up routing with {signal_desc}. Use {model_desc} for processing."
    en_technical = f"Create {len(signals)} SIGNAL declarations ({', '.join([s[0] for s in signals[:3]])}), {len(routes)} ROUTE blocks using MODEL {model_desc}."
    zh_formal = f"配置路由器，包含{len(signals)}个信号检测和{len(routes)}个路由规则，使用{model_desc}模型。"
    zh_casual = f"设置一下路由，用{model_desc}来处理请求。"
    ambiguous = f"Route requests using {model_desc}."
    
    return {
        'en_formal': en_formal,
        'en_casual': en_casual,
        'en_technical': en_technical,
        'zh_formal': zh_formal,
        'zh_casual': zh_casual,
        'ambiguous': ambiguous,
    }


def load_dsl_samples(input_path: Path) -> Generator[dict, None, None]:
    """加载 DSL 样本"""
    if input_path.is_file():
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif input_path.is_dir():
        for jsonl_file in sorted(input_path.glob('*.jsonl')):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def create_nl_pairs(sample: dict, nl_descriptions: dict, style: str) -> dict:
    """创建 NL-DSL 配对样本"""
    dsl_hash = hashlib.md5(sample['dsl'].encode()).hexdigest()[:8]
    
    return {
        'id': f"{sample.get('id', 'unknown')}_{style}",
        'instruction': "Convert the following natural language description into Signal DSL configuration.",
        'input': nl_descriptions[style],
        'output': sample['dsl'],
        'style': style,
        'complexity': sample.get('complexity', sample.get('metadata', {}).get('complexity', 'unknown')),
        'source_id': sample.get('id'),
        'valid': sample.get('valid', True),
        'metadata': sample.get('metadata', {}),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate NL descriptions for DSL configurations')
    parser.add_argument('--input', type=Path, required=True,
                        help='Input JSONL file or directory with DSL samples')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output directory for NL-DSL pairs')
    parser.add_argument('--api', choices=['vllm', 'openai', 'anthropic', 'local'],
                        default='local', help='API provider to use (default: local)')
    parser.add_argument('--vllm-url', type=str, default='http://localhost:8000',
                        help='vLLM server URL (default: http://localhost:8000)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (auto-detected for vLLM if not specified)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='Use local generation (no API calls)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Save checkpoint every N samples')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    parser.add_argument('--max-tokens', type=int, default=2000,
                        help='Max tokens for generation (default: 2000)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (for vLLM, default: 1)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output files')
    args = parser.parse_args()
    
    if args.dry_run:
        args.api = 'local'
    
    # 自动检测 vLLM 模型
    if args.api == 'vllm' and args.model is None:
        args.model = detect_vllm_model(args.vllm_url)
        if args.model:
            print(f"Auto-detected vLLM model: {args.model}")
        else:
            print("Warning: Could not detect vLLM model, using 'default'")
            args.model = 'default'
    elif args.model is None:
        args.model = 'gpt-4o' if args.api == 'openai' else 'claude-3-sonnet-20240229'
    
    # 配置
    config = NLGenerationConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        vllm_url=args.vllm_url,
        rate_limit_delay=0.05 if args.api == 'vllm' else 0.5,  # vLLM 本地可以更快
    )
    
    # 选择生成函数
    if args.api == 'vllm':
        generate_fn = lambda dsl: generate_nl_vllm(dsl, config)
        print(f"Using vLLM server at {args.vllm_url} with model {args.model}")
    elif args.api == 'openai':
        generate_fn = lambda dsl: generate_nl_openai(dsl, config)
    elif args.api == 'anthropic':
        generate_fn = lambda dsl: generate_nl_anthropic(dsl, config)
    else:
        generate_fn = lambda dsl: generate_nl_local(dsl, config)
    
    # 确保输出目录存在
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 断点续传: 获取已处理的 ID
    processed_ids = set()
    if args.resume:
        processed_ids = get_processed_ids(args.output)
        print(f"Resume mode: found {len(processed_ids)} already processed samples")
    
    # 按风格分类的输出文件 (追加模式用于断点续传)
    file_mode = 'a' if args.resume else 'w'
    style_files = {
        'en_formal': open(args.output / 'en_formal.jsonl', file_mode, encoding='utf-8'),
        'en_casual': open(args.output / 'en_casual.jsonl', file_mode, encoding='utf-8'),
        'en_technical': open(args.output / 'en_technical.jsonl', file_mode, encoding='utf-8'),
        'zh_formal': open(args.output / 'zh_formal.jsonl', file_mode, encoding='utf-8'),
        'zh_casual': open(args.output / 'zh_casual.jsonl', file_mode, encoding='utf-8'),
        'ambiguous': open(args.output / 'ambiguous.jsonl', file_mode, encoding='utf-8'),
    }
    
    # 统计
    processed = 0
    skipped = 0
    failed = 0
    start_time = time.time()
    
    try:
        samples = list(load_dsl_samples(args.input))
        total = min(len(samples), args.limit) if args.limit else len(samples)
        print(f"Total samples to process: {total}")
        
        for i, sample in enumerate(samples):
            if args.limit and processed >= args.limit:
                break
            
            sample_id = sample.get('id', f'sample_{i}')
            dsl = sample.get('dsl', '')
            
            # 跳过无效或已处理的样本
            if not dsl or not sample.get('valid', True):
                continue
            
            if sample_id in processed_ids:
                skipped += 1
                continue
            
            # 生成 NL 描述
            nl_descriptions = generate_fn(dsl)
            
            if nl_descriptions is None:
                failed += 1
                print(f"[{processed + failed}/{total}] FAILED: {sample_id}")
                continue
            
            # 创建并保存配对
            for style in style_files:
                if style in nl_descriptions:
                    pair = create_nl_pairs(sample, nl_descriptions, style)
                    style_files[style].write(json.dumps(pair, ensure_ascii=False) + '\n')
            
            processed += 1
            
            # 进度显示
            if processed % 10 == 0 or processed == 1:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed - skipped) / rate if rate > 0 else 0
                print(f"[{processed}/{total}] Processed: {processed}, Skipped: {skipped}, Failed: {failed}, "
                      f"Rate: {rate:.1f}/s, ETA: {eta/60:.1f}min")
            
            # 定期刷新
            if processed % args.batch_size == 0:
                for f in style_files.values():
                    f.flush()
            
            # Rate limiting (vLLM 本地不需要太多延迟)
            if args.api != 'local':
                time.sleep(config.rate_limit_delay)
    
    finally:
        # 关闭所有文件
        for f in style_files.values():
            f.close()
    
    # 最终统计
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"=== Generation Complete ===")
    print(f"{'='*50}")
    print(f"API:             {args.api}")
    print(f"Model:           {args.model}")
    print(f"Total processed: {processed}")
    print(f"Skipped:         {skipped}")
    print(f"Failed:          {failed}")
    print(f"Time elapsed:    {elapsed/60:.1f} minutes")
    print(f"Average rate:    {processed/elapsed:.2f} samples/sec" if elapsed > 0 else "N/A")
    print(f"Output dir:      {args.output}")
    print(f"Total NL pairs:  {processed * 6}")
    
    # 保存统计
    stats = {
        'total_processed': processed,
        'skipped': skipped,
        'failed': failed,
        'api': args.api,
        'model': args.model,
        'vllm_url': args.vllm_url if args.api == 'vllm' else None,
        'samples_per_style': processed,
        'total_pairs': processed * 6,
        'time_elapsed_seconds': elapsed,
        'rate_per_second': processed / elapsed if elapsed > 0 else 0,
    }
    with open(args.output / 'generation_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
