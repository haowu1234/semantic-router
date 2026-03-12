#!/usr/bin/env python3
"""
Early Stopping Callbacks for DPO Training.

DPO training has unique characteristics that require specialized early stopping:
1. Loss converges to 0 (unlike SFT which converges to ~0)
2. Accuracy can quickly reach 100%
3. Model can diverge too far from reference (overfitting)

This module provides multiple early stopping strategies.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


@dataclass
class DPOEarlyStoppingConfig:
    """Configuration for DPO early stopping.
    
    Strategies:
    - accuracy: Stop when accuracy reaches target
    - margin: Stop when margin reaches target (risk of overfit if too high)
    - logps_drift: Stop when logps drifts too far from initial (prevents overfit)
    - loss: Stop when loss is low enough
    - combined: Use multiple criteria together
    
    Recommended settings for typical DPO training:
    - accuracy_threshold: 0.95 (95% is usually good enough)
    - margin_threshold: 2.0 (e^2 ≈ 7x preference, reasonable)
    - max_logps_drift: 100 (prevent severe overfit)
    - min_loss: 0.15 (don't need to go lower)
    """
    
    # Strategy selection
    strategy: str = "combined"  # accuracy, margin, logps_drift, loss, combined
    
    # Accuracy-based stopping
    accuracy_threshold: float = 0.95
    accuracy_patience: int = 3  # Steps to wait after reaching threshold
    
    # Margin-based stopping (prevent excessive margins)
    margin_threshold: float = 2.0  # Stop if margins exceed this
    margin_min: float = 0.5  # Ensure margin is at least this before stopping
    
    # LogPS drift stopping (prevent model diverging too far from reference)
    max_logps_drift: float = 100.0  # Max allowed drift from initial logps
    initial_logps: Optional[float] = None  # Will be set during training
    
    # Loss-based stopping
    min_loss: float = 0.15  # Stop if loss goes below this
    
    # Combined strategy thresholds
    combined_accuracy: float = 0.90
    combined_margin_min: float = 0.5
    combined_margin_max: float = 3.0
    combined_max_logps_drift: float = 80.0
    
    # Minimum steps before early stopping can trigger
    min_steps: int = 50
    
    # Whether to save checkpoint when stopping
    save_on_stop: bool = True
    
    # Logging
    verbose: bool = True


class DPOEarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback specifically designed for DPO training.
    
    Monitors DPO-specific metrics and stops training when criteria are met.
    """
    
    def __init__(self, config: Optional[DPOEarlyStoppingConfig] = None):
        self.config = config or DPOEarlyStoppingConfig()
        self.history: List[Dict[str, Any]] = []
        self.best_step: int = 0
        self.best_metrics: Dict[str, Any] = {}
        self.should_stop: bool = False
        self.stop_reason: str = ""
        
        # Track initial values
        self.initial_logps_chosen: Optional[float] = None
        self.initial_logps_rejected: Optional[float] = None
        
        # Patience counters
        self.accuracy_patience_counter: int = 0
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Check early stopping criteria on each log event."""
        if logs is None or state.global_step < self.config.min_steps:
            return
        
        # Extract DPO metrics
        accuracy = logs.get('rewards/accuracies', None)
        margins = logs.get('rewards/margins', None)
        loss = logs.get('loss', None)
        logps_chosen = logs.get('logps/chosen', None)
        logps_rejected = logs.get('logps/rejected', None)
        
        # Skip if essential metrics missing
        if accuracy is None or margins is None:
            return
        
        # Initialize baseline logps
        if self.initial_logps_chosen is None and logps_chosen is not None:
            self.initial_logps_chosen = logps_chosen
            self.initial_logps_rejected = logps_rejected
            if self.config.verbose:
                print(f"\n[EarlyStop] Initial logps/chosen: {logps_chosen:.2f}")
        
        # Store history
        self.history.append({
            'step': state.global_step,
            'accuracy': accuracy,
            'margins': margins,
            'loss': loss,
            'logps_chosen': logps_chosen,
            'logps_rejected': logps_rejected,
        })
        
        # Calculate drift
        logps_drift = 0
        if self.initial_logps_chosen is not None and logps_chosen is not None:
            logps_drift = abs(logps_chosen - self.initial_logps_chosen)
        
        # Check stopping criteria based on strategy
        should_stop = False
        reason = ""
        
        if self.config.strategy == "accuracy":
            should_stop, reason = self._check_accuracy(accuracy)
            
        elif self.config.strategy == "margin":
            should_stop, reason = self._check_margin(margins)
            
        elif self.config.strategy == "logps_drift":
            should_stop, reason = self._check_logps_drift(logps_drift)
            
        elif self.config.strategy == "loss":
            should_stop, reason = self._check_loss(loss)
            
        elif self.config.strategy == "combined":
            should_stop, reason = self._check_combined(
                accuracy, margins, logps_drift, loss
            )
        
        # Track best metrics (based on accuracy + reasonable margin)
        if accuracy is not None and margins is not None:
            if (accuracy >= self.best_metrics.get('accuracy', 0) and 
                margins >= 0.3 and margins <= 5.0):
                self.best_step = state.global_step
                self.best_metrics = {
                    'accuracy': accuracy,
                    'margins': margins,
                    'loss': loss,
                    'logps_drift': logps_drift,
                }
        
        # Execute stopping
        if should_stop:
            self.should_stop = True
            self.stop_reason = reason
            control.should_training_stop = True
            
            if self.config.verbose:
                print(f"\n" + "="*60)
                print(f"[EarlyStop] Stopping training at step {state.global_step}")
                print(f"[EarlyStop] Reason: {reason}")
                print(f"[EarlyStop] Current metrics:")
                print(f"  - Accuracy: {accuracy:.4f}")
                print(f"  - Margins: {margins:.4f}")
                print(f"  - Loss: {loss:.4f}" if loss else "  - Loss: N/A")
                print(f"  - LogPS Drift: {logps_drift:.2f}")
                print(f"[EarlyStop] Best step: {self.best_step}")
                print(f"[EarlyStop] Best metrics: {self.best_metrics}")
                print("="*60 + "\n")
    
    def _check_accuracy(self, accuracy: float) -> tuple[bool, str]:
        """Check accuracy-based stopping."""
        if accuracy >= self.config.accuracy_threshold:
            self.accuracy_patience_counter += 1
            if self.accuracy_patience_counter >= self.config.accuracy_patience:
                return True, f"Accuracy {accuracy:.4f} >= {self.config.accuracy_threshold} for {self.config.accuracy_patience} steps"
        else:
            self.accuracy_patience_counter = 0
        return False, ""
    
    def _check_margin(self, margins: float) -> tuple[bool, str]:
        """Check margin-based stopping (stop if too high = overfit)."""
        if margins >= self.config.margin_threshold:
            return True, f"Margins {margins:.4f} >= {self.config.margin_threshold} (risk of overfit)"
        return False, ""
    
    def _check_logps_drift(self, drift: float) -> tuple[bool, str]:
        """Check logps drift stopping."""
        if drift >= self.config.max_logps_drift:
            return True, f"LogPS drift {drift:.2f} >= {self.config.max_logps_drift} (model diverged too far)"
        return False, ""
    
    def _check_loss(self, loss: Optional[float]) -> tuple[bool, str]:
        """Check loss-based stopping."""
        if loss is not None and loss <= self.config.min_loss:
            return True, f"Loss {loss:.4f} <= {self.config.min_loss}"
        return False, ""
    
    def _check_combined(
        self, 
        accuracy: float, 
        margins: float, 
        logps_drift: float,
        loss: Optional[float]
    ) -> tuple[bool, str]:
        """
        Combined stopping strategy (RECOMMENDED).
        
        Stop when:
        1. Accuracy is good enough (>= 90%)
        2. Margins are reasonable (0.5 <= m <= 3.0)
        3. Model hasn't diverged too far (drift <= 80)
        
        This prevents:
        - Stopping too early (low accuracy)
        - Overfitting (high margins, high drift)
        """
        reasons = []
        
        # Must have minimum accuracy
        if accuracy < self.config.combined_accuracy:
            return False, ""
        
        # Check margin is in reasonable range
        margin_ok = (self.config.combined_margin_min <= margins <= self.config.combined_margin_max)
        
        # Check drift is acceptable
        drift_ok = (logps_drift <= self.config.combined_max_logps_drift)
        
        # If margins getting too high, stop to prevent overfit
        if margins > self.config.combined_margin_max:
            return True, f"Margins {margins:.2f} exceeded max {self.config.combined_margin_max} with accuracy {accuracy:.2%}"
        
        # If drift getting too high, stop
        if logps_drift > self.config.combined_max_logps_drift:
            return True, f"LogPS drift {logps_drift:.2f} exceeded max {self.config.combined_max_logps_drift}"
        
        # If accuracy high and margins reasonable, we're done
        if accuracy >= 0.95 and margin_ok and drift_ok:
            return True, f"Optimal point: accuracy={accuracy:.2%}, margins={margins:.2f}, drift={logps_drift:.2f}"
        
        return False, ""
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log summary at end of training."""
        if self.config.verbose:
            print(f"\n[EarlyStop] Training ended")
            if self.should_stop:
                print(f"[EarlyStop] Early stopped: {self.stop_reason}")
            print(f"[EarlyStop] Best checkpoint at step {self.best_step}")
            print(f"[EarlyStop] Best metrics: {self.best_metrics}")
    
    def save_history(self, output_dir: str):
        """Save training history for analysis."""
        output_path = Path(output_dir) / "early_stopping_history.json"
        with open(output_path, 'w') as f:
            json.dump({
                'history': self.history,
                'best_step': self.best_step,
                'best_metrics': self.best_metrics,
                'stop_reason': self.stop_reason,
                'config': {
                    'strategy': self.config.strategy,
                    'accuracy_threshold': self.config.accuracy_threshold,
                    'margin_threshold': self.config.margin_threshold,
                    'max_logps_drift': self.config.max_logps_drift,
                }
            }, f, indent=2)


def create_early_stopping_callback(
    strategy: str = "combined",
    accuracy_threshold: float = 0.95,
    margin_threshold: float = 2.0,
    max_logps_drift: float = 100.0,
    min_steps: int = 50,
    verbose: bool = True,
) -> DPOEarlyStoppingCallback:
    """
    Factory function to create early stopping callback.
    
    Recommended configurations:
    
    1. Conservative (for important models):
       strategy="combined"
       accuracy_threshold=0.95
       margin_threshold=2.5
       max_logps_drift=100
    
    2. Aggressive (for quick experiments):
       strategy="combined"
       accuracy_threshold=0.90
       margin_threshold=1.5
       max_logps_drift=60
    
    3. Simple accuracy-based:
       strategy="accuracy"
       accuracy_threshold=0.95
    """
    config = DPOEarlyStoppingConfig(
        strategy=strategy,
        accuracy_threshold=accuracy_threshold,
        margin_threshold=margin_threshold,
        max_logps_drift=max_logps_drift,
        min_steps=min_steps,
        verbose=verbose,
    )
    return DPOEarlyStoppingCallback(config)
