#!/usr/bin/env python3
"""
DSL 数据验证与最终数据集构建管线。

职责:
1. 验证正样本 DSL（syntax / semantic / compile）
2. 构建 train/eval/test 的 Stage 2 SFT 数据
3. 基于同一 NL prompt 构建 conditional DPO 数据
4. 输出统一的 final/ 数据集文件供训练直接消费
"""

import argparse
import functools
import json
import random
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """DSL 验证结果。"""

    syntax_valid: bool
    semantic_valid: bool
    compile_valid: bool
    errors: list[str]
    warnings: list[str]


@functools.lru_cache(maxsize=None)
def detect_go_validator_mode(parser_bin: str) -> str:
    """Detect whether the binary is the legacy JSON parser or the newer sr-dsl CLI."""
    try:
        result = subprocess.run(
            [parser_bin, "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        help_text = f"{result.stdout}\n{result.stderr}"
        if "Usage: sr-dsl <command>" in help_text:
            return "sr_dsl_cli"
    except Exception:
        pass
    return "legacy_json"


def parse_sr_dsl_diagnostics(output: str) -> tuple[list[str], list[str], list[str]]:
    """Parse plain-text diagnostics emitted by sr-dsl validate."""
    errors = []
    warnings = []
    constraints = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line == "No issues found." or line.startswith("Summary:"):
            continue

        if "🔴" in line or "Error:" in line:
            errors.append(line.split("Error:", 1)[-1].strip() if "Error:" in line else line)
        elif "🟡" in line or "Warning:" in line:
            warnings.append(line.split("Warning:", 1)[-1].strip() if "Warning:" in line else line)
        elif "🟠" in line or "Constraint:" in line:
            constraints.append(line.split("Constraint:", 1)[-1].strip() if "Constraint:" in line else line)
        else:
            errors.append(line)

    return errors, warnings, constraints


def validate_dsl_with_legacy_json_parser(dsl: str, parser_bin: Path) -> ValidationResult:
    """Validate DSL using the old stdin+JSON parser contract."""
    result = subprocess.run(
        [str(parser_bin), "--validate", "--json"],
        input=dsl,
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode != 0:
        return ValidationResult(
            syntax_valid=False,
            semantic_valid=False,
            compile_valid=False,
            errors=[result.stderr.strip() or result.stdout.strip() or "Parser failed"],
            warnings=[],
        )

    output = json.loads(result.stdout)
    return ValidationResult(
        syntax_valid=output.get("syntax_valid", True),
        semantic_valid=output.get("semantic_valid", True),
        compile_valid=output.get("compile_valid", True),
        errors=output.get("errors", []),
        warnings=output.get("warnings", []),
    )


def validate_dsl_with_sr_dsl_cli(dsl: str, parser_bin: Path) -> ValidationResult:
    """Validate DSL using the current sr-dsl file-based CLI."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dsl", delete=False, encoding="utf-8") as handle:
        handle.write(dsl)
        temp_path = Path(handle.name)

    try:
        validate_result = subprocess.run(
            [str(parser_bin), "validate", str(temp_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        errors, warnings, constraints = parse_sr_dsl_diagnostics(
            f"{validate_result.stdout}\n{validate_result.stderr}"
        )

        syntax_valid = len(errors) == 0
        semantic_valid = syntax_valid and len(warnings) == 0 and len(constraints) == 0
        compile_valid = False

        if syntax_valid:
            compile_result = subprocess.run(
                [str(parser_bin), "compile", str(temp_path), "-o", "-"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            compile_valid = compile_result.returncode == 0
            if compile_result.returncode != 0:
                compile_error = compile_result.stderr.strip() or compile_result.stdout.strip() or "Compile failed"
                errors.append(compile_error)

        return ValidationResult(
            syntax_valid=syntax_valid,
            semantic_valid=semantic_valid,
            compile_valid=compile_valid,
            errors=errors,
            warnings=warnings + constraints,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def validate_dsl_with_go(dsl: str, parser_bin: Path | None) -> ValidationResult:
    """用 Go parser 做完整验证；不可用时退回简单验证。"""
    if parser_bin and parser_bin.exists():
        try:
            mode = detect_go_validator_mode(str(parser_bin))
            if mode == "sr_dsl_cli":
                return validate_dsl_with_sr_dsl_cli(dsl, parser_bin)
            return validate_dsl_with_legacy_json_parser(dsl, parser_bin)
        except subprocess.TimeoutExpired:
            return ValidationResult(
                syntax_valid=False,
                semantic_valid=False,
                compile_valid=False,
                errors=["Validation timeout"],
                warnings=[],
            )
        except json.JSONDecodeError:
            return ValidationResult(
                syntax_valid=result.returncode == 0,
                semantic_valid=False,
                compile_valid=False,
                errors=[],
                warnings=[],
            )
        except Exception as exc:
            return ValidationResult(
                syntax_valid=False,
                semantic_valid=False,
                compile_valid=False,
                errors=[str(exc)],
                warnings=[],
            )

    return validate_dsl_simple(dsl)


def validate_dsl_simple(dsl: str) -> ValidationResult:
    """轻量级 DSL 验证，用于 fallback 或 skip-validation。"""
    errors = []
    warnings = []

    brace_delta = dsl.count("{") - dsl.count("}")
    if brace_delta != 0:
        errors.append(f"Brace mismatch: {brace_delta} unclosed")

    if dsl.count('"') % 2 != 0:
        errors.append("Unclosed string literal")

    keywords = ["SIGNAL", "ROUTE", "PLUGIN", "BACKEND", "GLOBAL", "PRIORITY", "WHEN", "MODEL", "ALGORITHM"]
    for keyword in keywords:
        wrong_patterns = [keyword + "S", keyword[:-1], keyword[:3] + keyword[4:]]
        for wrong in wrong_patterns:
            if re.search(rf"\b{wrong}\b", dsl, re.IGNORECASE) and wrong != keyword:
                errors.append(f"Possible typo: {wrong} (should be {keyword}?)")

    defined_signals = set()
    for match in re.finditer(r"SIGNAL\s+(\w+)\s+(\w+)", dsl):
        defined_signals.add((match.group(1), match.group(2)))

    when_blocks = re.finditer(r"WHEN\s+.*?(?=\n\s*(?:MODEL|PLUGIN|ALGORITHM)|$)", dsl, re.DOTALL)
    for block in when_blocks:
        for match in re.finditer(r"(\w+)\(\"([^\"]+)\"\)", block.group(0)):
            ref = (match.group(1), match.group(2))
            if ref not in defined_signals:
                warnings.append(f'Reference to undefined signal: {match.group(1)}("{match.group(2)}")')

    syntax_valid = len(errors) == 0
    semantic_valid = len([w for w in warnings if "undefined" in w.lower()]) == 0

    return ValidationResult(
        syntax_valid=syntax_valid,
        semantic_valid=semantic_valid,
        compile_valid=syntax_valid and semantic_valid,
        errors=errors,
        warnings=warnings,
    )


def validate_dsl(dsl: str, parser_bin: Path | None, force_simple: bool = False) -> ValidationResult:
    """统一验证入口。"""
    if force_simple:
        return validate_dsl_simple(dsl)
    return validate_dsl_with_go(dsl, parser_bin)


def load_samples(input_path: Path):
    """递归加载目录中的 JSONL 样本。"""
    if input_path.is_file():
        with open(input_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return

    for jsonl_file in sorted(input_path.rglob("*.jsonl")):
        if "final" in str(jsonl_file):
            continue
        with open(jsonl_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def normalize_dsl(dsl: str) -> str:
    """标准化 DSL，用于 pair 去重和等价过滤。"""
    dsl = re.sub(r"//.*?$", "", dsl, flags=re.MULTILINE)
    dsl = re.sub(r"/\*.*?\*/", "", dsl, flags=re.DOTALL)
    dsl = re.sub(r"\s+", " ", dsl)
    return dsl.strip()


def infer_source_id(sample: dict) -> str:
    """统一获取样本的 source id。"""
    if sample.get("source_id"):
        return sample["source_id"]
    if sample.get("original_id"):
        return sample["original_id"]
    return sample.get("id", "")


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """保存 JSONL 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_by_source_id(samples: list[dict], seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, list[dict]]:
    """按 source_id 分桶，保证同一 DSL 家族不会跨 split。"""
    grouped = defaultdict(list)
    for sample in samples:
        grouped[infer_source_id(sample)].append(sample)

    source_ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(source_ids)

    total = len(source_ids)
    train_count = int(total * ratios[0])
    eval_count = int(total * ratios[1])

    train_ids = set(source_ids[:train_count])
    eval_ids = set(source_ids[train_count:train_count + eval_count])
    test_ids = set(source_ids[train_count + eval_count:])

    splits = {"train": [], "eval": [], "test": []}
    for source_id, items in grouped.items():
        if source_id in train_ids:
            splits["train"].extend(items)
        elif source_id in eval_ids:
            splits["eval"].extend(items)
        else:
            splits["test"].extend(items)

    return splits


def validate_stage1_sample(sample: dict, parser_bin: Path | None, force_simple: bool) -> tuple[dict | None, ValidationResult]:
    """验证 Stage 1 正样本。"""
    dsl = sample.get("dsl", "")
    if not dsl:
        return None, ValidationResult(False, False, False, ["Empty DSL"], [])

    result = validate_dsl(dsl, parser_bin, force_simple=force_simple)
    # Stage 1 is syntax pretraining: keep syntactically valid DSL even if it has softer diagnostics.
    if not result.syntax_valid:
        return None, result

    record = {
        "id": sample.get("id", ""),
        "source_id": infer_source_id(sample),
        "dsl": dsl,
        "complexity": sample.get("complexity", sample.get("metadata", {}).get("complexity", "unknown")),
    }
    return record, result


def validate_stage2_sample(sample: dict, parser_bin: Path | None, force_simple: bool) -> tuple[dict | None, ValidationResult]:
    """验证 Stage 2 正样本，chosen 必须三重校验通过。"""
    prompt = (sample.get("input") or "").strip()
    chosen = (sample.get("output") or sample.get("dsl") or "").strip()
    if not prompt or not chosen:
        return None, ValidationResult(False, False, False, ["Missing input/output"], [])

    result = validate_dsl(chosen, parser_bin, force_simple=force_simple)
    if not (result.syntax_valid and result.semantic_valid and result.compile_valid):
        return None, result

    record = {
        "id": sample.get("id", ""),
        "source_id": infer_source_id(sample),
        "instruction": sample.get(
            "instruction",
            "Convert the following natural language description into Signal DSL configuration.",
        ),
        "input": prompt,
        "output": chosen,
        "style": sample.get("style", "unknown"),
        "complexity": sample.get("complexity", "unknown"),
        "chosen_syntax_valid": result.syntax_valid,
        "chosen_semantic_valid": result.semantic_valid,
        "chosen_compile_valid": result.compile_valid,
    }
    return record, result


def run_parallel_validation(samples: list[dict], validator, parser_bin: Path | None, force_simple: bool, workers: int, progress_label: str) -> tuple[list[dict], int]:
    """并行验证样本。"""
    accepted = []
    rejected = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(validator, sample, parser_bin, force_simple)
            for sample in samples
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            record, _ = future.result()
            if record is not None:
                accepted.append(record)
            else:
                rejected += 1

            if index % 500 == 0:
                print(f"  {progress_label}: validated {index}/{len(samples)}")

    return accepted, rejected


def classify_negative_sample(sample: dict) -> tuple[str, str]:
    """归类 negative 的 broad error class 和 difficulty。"""
    error_class = sample.get("error_class")
    if error_class:
        difficulty = sample.get("difficulty")
        if difficulty:
            return error_class, difficulty

    category = sample.get("mutation_category", "")
    if category == "intent_mismatch":
        return "intent_mismatch", "hard"
    if category == "near_miss":
        return "near_miss", "hard"
    if category in {"syntax_error", "encoding_error"}:
        return "legality", "easy"
    return "semantic", "medium"


def negative_priority(sample: dict) -> tuple[int, str]:
    """优先保留 hard negatives。"""
    error_class, _ = classify_negative_sample(sample)
    order = {
        "intent_mismatch": 0,
        "near_miss": 1,
        "semantic": 2,
        "legality": 3,
    }
    return order.get(error_class, 4), sample.get("mutation_type", "")


def build_conditional_dpo_pairs(
    stage2_samples: list[dict],
    negative_samples: list[dict],
    parser_bin: Path | None,
    force_simple: bool,
    max_pairs_per_source: int = 3,
) -> list[dict]:
    """基于同一 NL prompt 构建 conditional DPO 数据。"""
    negatives_by_source = defaultdict(list)
    for sample in negative_samples:
        negatives_by_source[infer_source_id(sample)].append(sample)

    pairs = []
    started_at = time.time()
    total = len(stage2_samples)

    for index, sample in enumerate(stage2_samples, start=1):
        source_id = sample["source_id"]
        chosen = sample["output"]
        chosen_norm = normalize_dsl(chosen)
        prompt = sample["input"]
        chosen_instruction = sample["instruction"]

        kept = 0
        seen = set()
        for negative in sorted(negatives_by_source.get(source_id, []), key=negative_priority):
            rejected = (negative.get("dsl") or "").strip()
            if not rejected:
                continue

            rejected_norm = normalize_dsl(rejected)
            if rejected_norm == chosen_norm:
                continue

            dedupe_key = (rejected_norm, negative.get("mutation_type", ""))
            if dedupe_key in seen:
                continue

            rejected_result = validate_dsl(rejected, parser_bin, force_simple=force_simple)
            error_class, difficulty = classify_negative_sample(negative)

            # Hard negatives 应尽量保持 syntax valid，避免退化成拼写纠错器。
            if error_class in {"intent_mismatch", "near_miss"} and not rejected_result.syntax_valid:
                continue

            pair = {
                "id": f"dpo_{sample['id']}_{negative.get('mutation_type', kept)}",
                "source_id": source_id,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "instruction": chosen_instruction,
                "style": sample.get("style", "unknown"),
                "complexity": sample.get("complexity", "unknown"),
                "error_class": error_class,
                "error_type": negative.get("mutation_type", "unknown"),
                "difficulty": difficulty,
                "chosen_syntax_valid": sample["chosen_syntax_valid"],
                "chosen_semantic_valid": sample["chosen_semantic_valid"],
                "chosen_compile_valid": sample["chosen_compile_valid"],
                "rejected_syntax_valid": rejected_result.syntax_valid,
                "rejected_semantic_valid": rejected_result.semantic_valid,
                "rejected_compile_valid": rejected_result.compile_valid,
                "mutation_category": negative.get("mutation_category", ""),
            }
            pairs.append(pair)
            seen.add(dedupe_key)
            kept += 1

            if kept >= max_pairs_per_source:
                break

        if index % 500 == 0 or index == total:
            elapsed = time.time() - started_at
            rate = index / elapsed if elapsed > 0 else 0.0
            eta_seconds = int((total - index) / rate) if rate > 0 else 0
            print(
                f"  dpo: processed {index}/{total} sources, "
                f"built {len(pairs)} pairs, eta ~{eta_seconds}s"
            )

    return pairs


def build_eval_benchmark(samples: list[dict], size: int = 200, seed: int = 42) -> list[dict]:
    """从 test split 构建 prompt-disjoint benchmark。"""
    rng = random.Random(seed)
    by_complexity = defaultdict(list)
    for sample in samples:
        by_complexity[sample.get("complexity", "unknown")].append(sample)

    allocation = {"L1": 0.05, "L2": 0.20, "L3": 0.40, "L4": 0.25, "L5": 0.10}
    benchmark = []
    for complexity, ratio in allocation.items():
        pool = by_complexity.get(complexity, [])
        rng.shuffle(pool)
        n = min(int(size * ratio), len(pool))
        benchmark.extend(pool[:n])

    if len(benchmark) < min(size, len(samples)):
        remaining_ids = {sample["id"] for sample in benchmark}
        leftovers = [sample for sample in samples if sample["id"] not in remaining_ids]
        rng.shuffle(leftovers)
        benchmark.extend(leftovers[: max(0, min(size, len(samples)) - len(benchmark))])

    return benchmark


def write_split_records(output_dir: Path, prefix: str, splits: dict[str, list[dict]]) -> None:
    """统一输出 split 文件，同时保留 legacy 文件名兼容训练脚本。"""
    save_jsonl(splits["train"], output_dir / f"{prefix}_train.jsonl")
    save_jsonl(splits["eval"], output_dir / f"{prefix}_eval.jsonl")
    save_jsonl(splits["test"], output_dir / f"{prefix}_test.jsonl")

    if prefix == "stage2_sft":
        save_jsonl(splits["train"], output_dir / "stage2_sft.jsonl")
        save_jsonl(splits["eval"], output_dir / "stage2_sft_eval.jsonl")
    elif prefix == "stage3_dpo":
        save_jsonl(splits["train"], output_dir / "stage3_dpo.jsonl")
        save_jsonl(splits["eval"], output_dir / "stage3_dpo_eval.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Validate DSL data and build final training datasets")
    parser.add_argument("--input", type=Path, required=True, help="Input directory containing generated data")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for final datasets")
    parser.add_argument("--parser-bin", type=Path, default=None, help="Path to Go DSL parser binary")
    parser.add_argument("--workers", type=int, default=4, help="Number of validation workers")
    parser.add_argument("--skip-validation", action="store_true", help="Use simple validation instead of Go parser")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility")
    parser.add_argument("--max-dpo-pairs-per-source", type=int, default=3, help="Maximum DPO pairs per source id")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading samples...")
    all_samples = list(load_samples(args.input))
    print(f"Loaded {len(all_samples)} samples")

    raw_dsl_samples = [
        sample for sample in all_samples
        if sample.get("dsl") and not sample.get("input") and sample.get("valid") is not False
    ]
    nl_pair_samples = [sample for sample in all_samples if sample.get("input") and sample.get("output")]
    negative_samples = [sample for sample in all_samples if sample.get("valid") is False and sample.get("dsl")]

    print(f"  Raw DSL positives: {len(raw_dsl_samples)}")
    print(f"  NL-DSL pairs:      {len(nl_pair_samples)}")
    print(f"  Negative samples:  {len(negative_samples)}")

    force_simple = args.skip_validation

    print("\nValidating Stage 1 positives...")
    stage1_records, rejected_stage1 = run_parallel_validation(
        raw_dsl_samples,
        validate_stage1_sample,
        args.parser_bin,
        force_simple,
        args.workers,
        "stage1",
    )
    print(f"  Accepted: {len(stage1_records)}  Rejected: {rejected_stage1}")

    print("\nValidating Stage 2 positives...")
    stage2_records, rejected_stage2 = run_parallel_validation(
        nl_pair_samples,
        validate_stage2_sample,
        args.parser_bin,
        force_simple,
        args.workers,
        "stage2",
    )
    print(f"  Accepted: {len(stage2_records)}  Rejected: {rejected_stage2}")

    print("\nSplitting Stage 1 and Stage 2 by source_id...")
    stage1_splits = split_by_source_id(stage1_records, seed=args.seed)
    stage2_splits = split_by_source_id(stage2_records, seed=args.seed)

    print("\nBuilding conditional DPO splits...")
    stage3_splits = {}
    for split_name, records in stage2_splits.items():
        stage3_splits[split_name] = build_conditional_dpo_pairs(
            records,
            negative_samples,
            args.parser_bin,
            force_simple,
            max_pairs_per_source=args.max_dpo_pairs_per_source,
        )
        print(f"  Stage3 {split_name}: {len(stage3_splits[split_name])}")

    benchmark = build_eval_benchmark(stage2_splits["test"], seed=args.seed)

    print("\nWriting final datasets...")
    save_jsonl(stage1_splits["train"], args.output / "stage1_syntax_pt.jsonl")
    save_jsonl(stage1_splits["eval"], args.output / "stage1_syntax_pt_eval.jsonl")
    save_jsonl(stage1_splits["test"], args.output / "stage1_syntax_pt_test.jsonl")

    write_split_records(args.output, "stage2_sft", stage2_splits)
    write_split_records(args.output, "stage3_dpo", stage3_splits)
    save_jsonl(benchmark, args.output / "eval_benchmark.jsonl")

    stats = {
        "raw_dsl_samples": len(raw_dsl_samples),
        "nl_pair_samples": len(nl_pair_samples),
        "negative_samples": len(negative_samples),
        "stage1_train": len(stage1_splits["train"]),
        "stage1_eval": len(stage1_splits["eval"]),
        "stage1_test": len(stage1_splits["test"]),
        "stage2_train": len(stage2_splits["train"]),
        "stage2_eval": len(stage2_splits["eval"]),
        "stage2_test": len(stage2_splits["test"]),
        "stage3_train": len(stage3_splits["train"]),
        "stage3_eval": len(stage3_splits["eval"]),
        "stage3_test": len(stage3_splits["test"]),
        "benchmark_size": len(benchmark),
        "validation_mode": "simple" if force_simple else "go_parser",
    }

    with open(args.output / "dataset_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    print("\n=== Final Dataset Summary ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
