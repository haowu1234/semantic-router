#!/usr/bin/env python3
"""
Signal DSL Model Test Client
交互式测试客户端

Usage:
    # 交互模式
    python test_client.py
    
    # 单次测试
    python test_client.py --prompt "处理代码相关的问题"
    
    # 批量测试
    python test_client.py --batch test_cases.json
"""

import argparse
import json
import sys
from typing import Optional

import requests

DEFAULT_SERVER = "http://localhost:8080"


def generate(prompt: str, 
             server: str = DEFAULT_SERVER,
             temperature: float = 0.1,
             max_tokens: int = 512,
             system_prompt: str = None) -> dict:
    """调用 OpenAI 兼容的 Chat Completion API"""
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append({
            "role": "system", 
            "content": "You are a Signal DSL configuration generator. Convert natural language descriptions into valid Signal DSL configurations."
        })
    messages.append({"role": "user", "content": prompt})
    
    response = requests.post(
        f"{server}/v1/chat/completions",
        json={
            "model": "dsl-generator",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        },
        timeout=120
    )
    
    if response.status_code != 200:
        return {"error": response.text}
    
    data = response.json()
    # 提取生成的文本
    if "choices" in data and len(data["choices"]) > 0:
        return {
            "generated_text": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {})
        }
    return {"error": "Invalid response format"}


def check_health(server: str = DEFAULT_SERVER) -> dict:
    """检查服务器健康状态"""
    try:
        response = requests.get(f"{server}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def interactive_mode(server: str):
    """交互式测试模式"""
    print("\n" + "=" * 60)
    print("Signal DSL Model - Interactive Test Client")
    print("=" * 60)
    
    # 检查服务器
    health = check_health(server)
    if "error" in health:
        print(f"❌ Server not available: {health['error']}")
        print(f"   Make sure server is running at {server}")
        return
    
    if not health.get("model_loaded"):
        print("❌ Model not loaded on server")
        return
    
    print(f"✅ Connected to server: {server}")
    print(f"   Checkpoint: {health.get('checkpoint', 'unknown')}")
    print(f"   Device: {health.get('device', 'unknown')}")
    print("\nCommands:")
    print("  Type your natural language description to generate DSL")
    print("  /temp <value>  - Set temperature (current: 0.1)")
    print("  /health        - Check server health")
    print("  /quit or /q    - Exit")
    print("-" * 60 + "\n")
    
    temperature = 0.1
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        
        if not user_input:
            continue
        
        # 命令处理
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd in ["/quit", "/q", "/exit"]:
                print("Bye!")
                break
            elif cmd == "/health":
                health = check_health(server)
                print(f"Health: {json.dumps(health, indent=2)}")
            elif cmd == "/temp":
                if len(parts) > 1:
                    try:
                        temperature = float(parts[1])
                        print(f"Temperature set to: {temperature}")
                    except ValueError:
                        print("Invalid temperature value")
                else:
                    print(f"Current temperature: {temperature}")
            else:
                print(f"Unknown command: {cmd}")
            continue
        
        # 生成
        print("Generating...", end=" ", flush=True)
        result = generate(user_input, server, temperature)
        
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
        else:
            print("Done!\n")
            print("-" * 40)
            print("Generated DSL:")
            print("-" * 40)
            print(result.get("generated_text", ""))
            print("-" * 40 + "\n")


def single_test(prompt: str, server: str, temperature: float):
    """单次测试"""
    print(f"Prompt: {prompt}")
    print("Generating...")
    
    result = generate(prompt, server, temperature)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("\nGenerated DSL:")
    print("-" * 40)
    print(result.get("generated_text", ""))
    print("-" * 40)


def batch_test(test_file: str, server: str, output_file: Optional[str] = None):
    """批量测试"""
    with open(test_file, "r") as f:
        test_cases = json.load(f)
    
    results = []
    total = len(test_cases)
    
    print(f"Running {total} test cases...")
    
    for i, case in enumerate(test_cases, 1):
        prompt = case.get("prompt") or case.get("input") or case.get("nl")
        expected = case.get("expected") or case.get("output") or case.get("dsl")
        
        print(f"[{i}/{total}] {prompt[:50]}...", end=" ", flush=True)
        
        result = generate(prompt, server)
        
        if "error" in result:
            print("❌ Error")
            results.append({
                "prompt": prompt,
                "expected": expected,
                "generated": None,
                "error": result["error"],
                "match": False
            })
        else:
            generated = result.get("generated_text", "")
            # 简单匹配检查
            match = expected and (expected.strip() == generated.strip())
            print("✅" if match else "⚠️")
            results.append({
                "prompt": prompt,
                "expected": expected,
                "generated": generated,
                "match": match
            })
    
    # 统计
    total_tests = len(results)
    errors = sum(1 for r in results if r.get("error"))
    matches = sum(1 for r in results if r.get("match"))
    
    print(f"\n{'='*50}")
    print(f"Results: {matches}/{total_tests} exact matches")
    print(f"Errors: {errors}")
    print(f"{'='*50}")
    
    # 保存结果
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Signal DSL Model Test Client")
    parser.add_argument(
        "--server", "-s",
        type=str,
        default=DEFAULT_SERVER,
        help=f"Server URL (default: {DEFAULT_SERVER})"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Single prompt to test"
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="JSON file with batch test cases"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for batch results"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        batch_test(args.batch, args.server, args.output)
    elif args.prompt:
        single_test(args.prompt, args.server, args.temperature)
    else:
        interactive_mode(args.server)


if __name__ == "__main__":
    main()
