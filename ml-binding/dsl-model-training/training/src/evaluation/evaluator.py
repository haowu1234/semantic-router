#!/usr/bin/env python3
"""
DSL Model Evaluator.

Comprehensive evaluation of trained DSL models including:
- Generation quality metrics
- Error category analysis
- Complexity-stratified evaluation
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from .metrics import DSLMetrics


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    
    # Overall metrics
    syntax_accuracy: float = 0.0
    semantic_accuracy: float = 0.0
    compile_success: float = 0.0
    exact_match: float = 0.0
    bleu: float = 0.0
    
    # By complexity
    by_complexity: dict = field(default_factory=dict)
    
    # By style
    by_style: dict = field(default_factory=dict)
    
    # Error analysis
    error_categories: dict = field(default_factory=dict)
    
    # Sample predictions
    samples: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'overall': {
                'syntax_accuracy': self.syntax_accuracy,
                'semantic_accuracy': self.semantic_accuracy,
                'compile_success': self.compile_success,
                'exact_match': self.exact_match,
                'bleu': self.bleu,
            },
            'by_complexity': self.by_complexity,
            'by_style': self.by_style,
            'error_categories': self.error_categories,
            'num_samples': len(self.samples),
        }


class DSLEvaluator:
    """
    Comprehensive DSL model evaluator.
    
    Evaluates model on:
    1. Overall generation quality
    2. Stratified by complexity (L1-L5)
    3. Stratified by NL style
    4. Error category analysis
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: dict,
        dsl_parser_path: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.metrics = DSLMetrics(dsl_parser_path=dsl_parser_path)
        
        # Generation config
        eval_config = config.get('evaluation', {})
        self.num_beams = eval_config.get('num_beams', 1)
        self.do_sample = eval_config.get('do_sample', False)
        self.temperature = eval_config.get('temperature', 1.0)
        self.top_p = eval_config.get('top_p', 1.0)
        self.max_new_tokens = eval_config.get('max_new_tokens', 1024)
    
    def evaluate(
        self,
        eval_file: str | Path,
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        save_predictions: bool = True,
        output_file: Optional[str | Path] = None,
    ) -> EvaluationResult:
        """
        Run comprehensive evaluation.
        
        Args:
            eval_file: Path to evaluation JSONL file
            batch_size: Batch size for generation
            max_samples: Maximum samples to evaluate
            save_predictions: Whether to save predictions
            output_file: Path to save results
        
        Returns:
            EvaluationResult with all metrics
        """
        # Load eval data
        samples = self._load_eval_data(eval_file, max_samples)
        
        # Generate predictions
        predictions = self._generate_predictions(samples, batch_size)
        
        # Extract references
        references = [s.get('output', s.get('dsl', '')) for s in samples]
        
        # Compute overall metrics
        overall_metrics = self.metrics.compute_all(predictions, references)
        
        # Stratify by complexity
        by_complexity = self._stratify_metrics(
            samples, predictions, references, key='complexity'
        )
        
        # Stratify by style
        by_style = self._stratify_metrics(
            samples, predictions, references, key='style'
        )
        
        # Error analysis
        error_analysis = self._analyze_errors(samples, predictions, references)
        
        # Build result
        result = EvaluationResult(
            syntax_accuracy=overall_metrics['syntax_accuracy'],
            semantic_accuracy=overall_metrics['semantic_accuracy'],
            compile_success=overall_metrics['compile_success'],
            exact_match=overall_metrics['exact_match'],
            bleu=overall_metrics['bleu'],
            by_complexity=by_complexity,
            by_style=by_style,
            error_categories=error_analysis,
        )
        
        # Add sample predictions
        if save_predictions:
            for i, (sample, pred) in enumerate(zip(samples[:20], predictions[:20])):
                result.samples.append({
                    'id': sample.get('id', f'sample_{i}'),
                    'input': sample.get('input', ''),
                    'reference': references[i],
                    'prediction': pred,
                    'syntax_valid': self.metrics._is_syntax_valid(pred),
                })
        
        # Save results
        if output_file:
            self._save_results(result, output_file)
        
        return result
    
    def _load_eval_data(
        self,
        eval_file: str | Path,
        max_samples: Optional[int] = None,
    ) -> list[dict]:
        """Load evaluation data from JSONL file."""
        samples = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
                    if max_samples and len(samples) >= max_samples:
                        break
        return samples
    
    def _generate_predictions(
        self,
        samples: list[dict],
        batch_size: int,
    ) -> list[str]:
        """Generate predictions for all samples."""
        self.model.eval()
        predictions = []
        
        data_config = self.config.get('data', {})
        system_prompt = data_config.get(
            'system_prompt',
            'You are a Signal DSL configuration generator.'
        )
        default_instruction = data_config.get(
            'default_instruction',
            'Convert the following natural language description into Signal DSL configuration.'
        )
        
        with torch.no_grad():
            for i in tqdm(range(0, len(samples), batch_size), desc="Generating"):
                batch = samples[i:i + batch_size]
                
                # Build prompts
                prompts = []
                for sample in batch:
                    instruction = sample.get('instruction', default_instruction)
                    input_text = sample.get('input', '')
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
                    ]
                    
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompts.append(prompt)
                
                # Tokenize
                inputs = self.tokenizer(
                    prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.get('model', {}).get('max_length', 2048) - self.max_new_tokens,
                ).to(self.model.device)
                
                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    do_sample=self.do_sample,
                    temperature=self.temperature if self.do_sample else 1.0,
                    top_p=self.top_p if self.do_sample else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode (only new tokens)
                for j, output in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    generated = output[input_length:]
                    decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
                    predictions.append(decoded.strip())
        
        return predictions
    
    def _stratify_metrics(
        self,
        samples: list[dict],
        predictions: list[str],
        references: list[str],
        key: str,
    ) -> dict:
        """Compute metrics stratified by a key (complexity, style, etc.)."""
        groups = defaultdict(lambda: {'preds': [], 'refs': []})
        
        for sample, pred, ref in zip(samples, predictions, references):
            group = sample.get(key, 'unknown')
            groups[group]['preds'].append(pred)
            groups[group]['refs'].append(ref)
        
        stratified = {}
        for group, data in groups.items():
            metrics = self.metrics.compute_all(data['preds'], data['refs'])
            metrics['count'] = len(data['preds'])
            stratified[group] = metrics
        
        return stratified
    
    def _analyze_errors(
        self,
        samples: list[dict],
        predictions: list[str],
        references: list[str],
    ) -> dict:
        """Analyze error categories in predictions."""
        errors = defaultdict(int)
        
        for pred in predictions:
            if not self.metrics._is_syntax_valid(pred):
                # Categorize syntax errors
                if '{' in pred and pred.count('{') != pred.count('}'):
                    errors['unbalanced_braces'] += 1
                elif not any(kw in pred for kw in ['SIGNAL', 'ROUTE', 'BACKEND', 'GLOBAL']):
                    errors['missing_keywords'] += 1
                else:
                    errors['other_syntax'] += 1
            elif not self.metrics._is_semantic_valid(pred):
                errors['reference_errors'] += 1
        
        return dict(errors)
    
    def _save_results(self, result: EvaluationResult, output_file: str | Path) -> None:
        """Save evaluation results to file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to {output_file}")
    
    def print_summary(self, result: EvaluationResult) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("DSL Model Evaluation Summary")
        print("=" * 60)
        
        print("\nOverall Metrics:")
        print(f"  Syntax Accuracy:   {result.syntax_accuracy:.2%}")
        print(f"  Semantic Accuracy: {result.semantic_accuracy:.2%}")
        print(f"  Compile Success:   {result.compile_success:.2%}")
        print(f"  Exact Match:       {result.exact_match:.2%}")
        print(f"  BLEU:              {result.bleu:.4f}")
        
        if result.by_complexity:
            print("\nBy Complexity:")
            for level in sorted(result.by_complexity.keys()):
                metrics = result.by_complexity[level]
                print(f"  {level}: syntax={metrics['syntax_accuracy']:.2%}, "
                      f"compile={metrics['compile_success']:.2%}, "
                      f"n={metrics['count']}")
        
        if result.error_categories:
            print("\nError Analysis:")
            for error_type, count in sorted(result.error_categories.items(), key=lambda x: -x[1]):
                print(f"  {error_type}: {count}")
        
        print("=" * 60)
