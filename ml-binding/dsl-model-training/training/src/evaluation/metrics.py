#!/usr/bin/env python3
"""
Evaluation metrics for DSL model training.

Includes:
- Syntax accuracy (does it parse?)
- Semantic accuracy (are references valid?)
- Compile success (can it compile?)
- BLEU score (text similarity)
- Exact match (perfect match rate)
"""

import re
import subprocess
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DSLMetrics:
    """
    DSL-specific evaluation metrics.
    
    Uses the Go DSL parser for accurate syntax/semantic validation.
    """
    
    dsl_parser_path: Optional[str] = None  # Path to dsl-parser binary
    
    # Cached results
    _syntax_cache: dict = field(default_factory=dict)
    
    def syntax_accuracy(self, predictions: list[str], references: list[str]) -> float:
        """
        Calculate syntax accuracy.
        
        Returns percentage of predictions that are syntactically valid.
        """
        valid_count = 0
        for pred in predictions:
            if self._is_syntax_valid(pred):
                valid_count += 1
        return valid_count / len(predictions) if predictions else 0.0
    
    def semantic_accuracy(self, predictions: list[str], references: list[str]) -> float:
        """
        Calculate semantic accuracy.
        
        Returns percentage of predictions that are semantically valid
        (references are resolved, types match, etc.)
        """
        valid_count = 0
        for pred in predictions:
            if self._is_semantic_valid(pred):
                valid_count += 1
        return valid_count / len(predictions) if predictions else 0.0
    
    def compile_success(self, predictions: list[str], references: list[str]) -> float:
        """
        Calculate compile success rate.
        
        Returns percentage of predictions that compile successfully.
        """
        valid_count = 0
        for pred in predictions:
            if self._can_compile(pred):
                valid_count += 1
        return valid_count / len(predictions) if predictions else 0.0
    
    def exact_match(self, predictions: list[str], references: list[str]) -> float:
        """
        Calculate exact match rate.
        
        Returns percentage of predictions that exactly match references
        (after normalization).
        """
        match_count = 0
        for pred, ref in zip(predictions, references):
            if self._normalize(pred) == self._normalize(ref):
                match_count += 1
        return match_count / len(predictions) if predictions else 0.0
    
    def bleu_score(self, predictions: list[str], references: list[str]) -> float:
        """
        Calculate BLEU score.
        
        Uses sentence-level BLEU for DSL text comparison.
        """
        try:
            from sacrebleu.metrics import BLEU
            bleu = BLEU()
            # References need to be wrapped in a list for sacrebleu
            refs = [[r] for r in references]
            result = bleu.corpus_score(predictions, refs)
            return result.score / 100.0  # Normalize to 0-1
        except ImportError:
            # Fallback to simple token overlap
            return self._simple_token_overlap(predictions, references)
    
    def compute_all(self, predictions: list[str], references: list[str]) -> dict:
        """Compute all metrics."""
        return {
            'syntax_accuracy': self.syntax_accuracy(predictions, references),
            'semantic_accuracy': self.semantic_accuracy(predictions, references),
            'compile_success': self.compile_success(predictions, references),
            'exact_match': self.exact_match(predictions, references),
            'bleu': self.bleu_score(predictions, references),
        }
    
    def _is_syntax_valid(self, dsl: str) -> bool:
        """Check if DSL is syntactically valid."""
        # Use Go parser if available
        if self.dsl_parser_path:
            return self._validate_with_go_parser(dsl, 'syntax')
        
        # Fallback to regex-based validation
        return self._regex_syntax_check(dsl)
    
    def _is_semantic_valid(self, dsl: str) -> bool:
        """Check if DSL is semantically valid."""
        if self.dsl_parser_path:
            return self._validate_with_go_parser(dsl, 'semantic')
        
        # Fallback: check basic reference validity
        return self._basic_semantic_check(dsl)
    
    def _can_compile(self, dsl: str) -> bool:
        """Check if DSL can compile."""
        if self.dsl_parser_path:
            return self._validate_with_go_parser(dsl, 'compile')
        
        # Without Go parser, assume semantic valid = compile valid
        return self._is_semantic_valid(dsl)
    
    def _validate_with_go_parser(self, dsl: str, level: str) -> bool:
        """Validate DSL using Go parser binary."""
        try:
            result = subprocess.run(
                [self.dsl_parser_path, '--validate', '--json'],
                input=dsl,
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode != 0:
                return False
            
            import json
            validation = json.loads(result.stdout)
            
            if level == 'syntax':
                return validation.get('syntax_valid', False)
            elif level == 'semantic':
                return validation.get('semantic_valid', False)
            elif level == 'compile':
                return validation.get('compile_valid', False)
            
            return False
        except Exception:
            return False
    
    def _regex_syntax_check(self, dsl: str) -> bool:
        """Basic regex-based syntax check."""
        # Check for basic structure
        has_signal_or_route = bool(re.search(r'\b(SIGNAL|ROUTE|BACKEND|GLOBAL)\b', dsl))
        
        # Check balanced braces
        open_braces = dsl.count('{')
        close_braces = dsl.count('}')
        balanced = open_braces == close_braces
        
        # Check for common syntax errors
        has_typo = bool(re.search(r'\b(SINGAL|ROTUE|BAKEND|PRIRITY)\b', dsl, re.IGNORECASE))
        
        return has_signal_or_route and balanced and not has_typo
    
    def _basic_semantic_check(self, dsl: str) -> bool:
        """Basic semantic check (reference validity)."""
        # Extract defined signals
        defined_signals = set()
        for match in re.finditer(r'SIGNAL\s+(\w+)\s+(\w+)\s*\{', dsl):
            sig_type = match.group(1)
            sig_name = match.group(2)
            defined_signals.add((sig_type, sig_name))
        
        # Extract signal references in WHEN clauses
        referenced = []
        for match in re.finditer(r'WHEN\s+.*?(?=\n\s*\n|\n\s*MODEL)', dsl, re.DOTALL):
            when_clause = match.group(0)
            for ref_match in re.finditer(r'(\w+)\("([^"]+)"\)', when_clause):
                ref_type = ref_match.group(1)
                ref_name = ref_match.group(2)
                referenced.append((ref_type, ref_name))
        
        # Check all references are defined
        for ref_type, ref_name in referenced:
            if (ref_type, ref_name) not in defined_signals:
                return False
        
        return True
    
    def _normalize(self, dsl: str) -> str:
        """Normalize DSL for comparison."""
        # Remove extra whitespace
        dsl = re.sub(r'\s+', ' ', dsl)
        # Remove comments
        dsl = re.sub(r'//.*?\n', '\n', dsl)
        dsl = re.sub(r'/\*.*?\*/', '', dsl, flags=re.DOTALL)
        # Normalize quotes
        dsl = dsl.replace('"', '"').replace('"', '"')
        return dsl.strip()
    
    def _simple_token_overlap(self, predictions: list[str], references: list[str]) -> float:
        """Simple token overlap score as BLEU fallback."""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())
            
            if not ref_tokens:
                scores.append(0.0)
            else:
                overlap = len(pred_tokens & ref_tokens)
                scores.append(overlap / len(ref_tokens))
        
        return sum(scores) / len(scores) if scores else 0.0


def compute_metrics_for_hf_trainer(eval_pred, tokenizer, dsl_metrics: DSLMetrics) -> dict:
    """
    Compute metrics function compatible with HuggingFace Trainer.
    
    Usage:
        trainer = Trainer(
            ...,
            compute_metrics=lambda p: compute_metrics_for_hf_trainer(p, tokenizer, metrics)
        )
    """
    predictions, labels = eval_pred
    
    # Decode predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Replace -100 with pad token id
    labels = [[l if l != -100 else tokenizer.pad_token_id for l in label] for label in labels]
    
    # Decode
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute metrics
    return dsl_metrics.compute_all(decoded_preds, decoded_labels)
