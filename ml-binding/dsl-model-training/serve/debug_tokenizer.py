#!/usr/bin/env python3
"""调试 tokenizer 问题"""

from transformers import AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    trust_remote_code=True
)
print(f"Tokenizer type: {type(tokenizer)}")
print(f"Tokenizer class: {tokenizer.__class__.__name__}")

# 测试 1: 简单字符串
print("\n=== Test 1: Simple string ===")
try:
    text = "Hello world"
    print(f"Input type: {type(text)}, value: {repr(text)}")
    result = tokenizer(text, return_tensors="pt")
    print(f"Success! Shape: {result['input_ids'].shape}")
except Exception as e:
    print(f"Error: {e}")

# 测试 2: apply_chat_template
print("\n=== Test 2: apply_chat_template ===")
try:
    messages = [{"role": "user", "content": "处理代码相关的问题"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Output type: {type(prompt)}")
    print(f"Output value: {repr(prompt[:200]) if len(str(prompt)) > 200 else repr(prompt)}")
except Exception as e:
    print(f"Error: {e}")

# 测试 3: tokenize the prompt
print("\n=== Test 3: Tokenize the prompt ===")
try:
    if isinstance(prompt, list):
        prompt = prompt[0]
    prompt = str(prompt)
    print(f"Prompt type after conversion: {type(prompt)}")
    result = tokenizer(prompt, return_tensors="pt")
    print(f"Success! Shape: {result['input_ids'].shape}")
except Exception as e:
    print(f"Error: {e}")

# 测试 4: 使用 encode
print("\n=== Test 4: Using encode ===")
try:
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Success! Shape: {tokens.shape}")
except Exception as e:
    print(f"Error: {e}")

# 测试 5: 检查 tokenizer 的 __call__ 签名
print("\n=== Test 5: Tokenizer signature ===")
import inspect
sig = inspect.signature(tokenizer.__call__)
print(f"Parameters: {list(sig.parameters.keys())[:10]}")

# 测试 6: 不同调用方式
print("\n=== Test 6: Different call methods ===")
test_text = "测试文本"

# 方式 A
try:
    result = tokenizer(test_text)
    print(f"A. tokenizer(text): OK")
except Exception as e:
    print(f"A. tokenizer(text): {e}")

# 方式 B
try:
    result = tokenizer(text=test_text)
    print(f"B. tokenizer(text=text): OK")
except Exception as e:
    print(f"B. tokenizer(text=text): {e}")

# 方式 C
try:
    result = tokenizer([test_text])
    print(f"C. tokenizer([text]): OK")
except Exception as e:
    print(f"C. tokenizer([text]): {e}")

# 方式 D
try:
    result = tokenizer.encode(test_text)
    print(f"D. tokenizer.encode(text): OK, length={len(result)}")
except Exception as e:
    print(f"D. tokenizer.encode(text): {e}")

print("\n=== Done ===")
