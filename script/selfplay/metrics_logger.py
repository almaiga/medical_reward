"""
Training metrics logging for GRPO self-play training.

This module provides utilities to log and track training metrics over time,
including loss, rewards, gradients, and other GRPO-specific metrics.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class MetricsLogger:
    """Logger for tracking training metrics across rounds."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory to save metrics
            experiment_name: Name of the experiment (used in filename)
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Metrics log file
        self.metrics_path = os.path.join(
            log_dir, 
            f"{experiment_name}_metrics.jsonl"
        )
        
        # Summary log file (for easy viewing)
        self.summary_path = os.path.join(
            log_dir,
            f"{experiment_name}_metrics_summary.json"
        )
        
        # In-memory storage for summary
        self.summary = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "rounds": []
        }
    
    def log_training_metrics(
        self,
        round_num: int,
        phase: str,  # "attacker" or "assessor"
        trainer_state: Any,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log metrics from a training phase.
        
        Args:
            round_num: Current round number
            phase: Training phase ("attacker" or "assessor")
            trainer_state: Trainer state object (from GRPOTrainer)
            additional_metrics: Additional metrics to log
        """
        # Extract metrics from trainer state
        metrics = {
            "round": round_num,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add trainer metrics if available
        if hasattr(trainer_state, "log_history") and trainer_state.log_history:
            # Aggregate metrics from all log entries (GRPO logs different metrics at different steps)
            # We want the most recent value for each metric
            aggregated = {}
            for log_entry in trainer_state.log_history:
                aggregated.update(log_entry)
            
            # Core training metrics
            if "loss" in aggregated:
                metrics["loss"] = aggregated["loss"]
            if "grad_norm" in aggregated:
                metrics["grad_norm"] = aggregated["grad_norm"]
            if "learning_rate" in aggregated:
                metrics["learning_rate"] = aggregated["learning_rate"]
            if "epoch" in aggregated:
                metrics["epoch"] = aggregated["epoch"]
            
            # GRPO-specific metrics
            grpo_keys = [
                "num_tokens",
                "completions/mean_length",
                "completions/min_length",
                "completions/max_length",
                "completions/clipped_ratio",
                "reward",
                "reward_std",
                "frac_reward_zero_std",
                "entropy",
                "clip_ratio/low_mean",
                "clip_ratio/high_mean",
                "clip_ratio/region_mean",
            ]
            
            for key in grpo_keys:
                if key in aggregated:
                    metrics[key] = aggregated[key]
            
            # Reward function specific metrics
            reward_keys = [k for k in aggregated.keys() if k.startswith("rewards/")]
            for key in reward_keys:
                metrics[key] = aggregated[key]
        
        # Add additional metrics
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Write to JSONL file
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        
        # Update summary
        self._update_summary(round_num, phase, metrics)
        
        print(f"📊 Logged {phase} metrics for round {round_num}")
    
    def _update_summary(self, round_num: int, phase: str, metrics: Dict[str, Any]):
        """Update the summary with latest metrics."""
        # Find or create round entry
        round_entry = None
        for r in self.summary["rounds"]:
            if r["round"] == round_num:
                round_entry = r
                break
        
        if round_entry is None:
            round_entry = {"round": round_num}
            self.summary["rounds"].append(round_entry)
        
        # Add phase metrics
        round_entry[phase] = {
            "loss": metrics.get("loss"),
            "reward": metrics.get("reward"),
            "reward_std": metrics.get("reward_std"),
            "grad_norm": metrics.get("grad_norm"),
            "learning_rate": metrics.get("learning_rate"),
            "entropy": metrics.get("entropy"),
            "mean_length": metrics.get("completions/mean_length"),
        }
        
        # Write summary to file
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
    
    def log_round_summary(
        self,
        round_num: int,
        diversity_stats: Dict[str, Any],
        judge_stats: Dict[str, Any]
    ):
        """Log summary statistics for a round.
        
        Args:
            round_num: Current round number
            diversity_stats: Diversity statistics
            judge_stats: Judge validation statistics
        """
        summary = {
            "round": round_num,
            "phase": "round_summary",
            "timestamp": datetime.now().isoformat(),
            "diversity": diversity_stats.copy(),
            "judge": judge_stats.copy(),
        }
        
        # Write to JSONL file
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        
        print(f"📊 Logged round {round_num} summary")
    
    def finalize(self):
        """Finalize logging and write final summary."""
        self.summary["end_time"] = datetime.now().isoformat()
        
        # Calculate overall statistics
        if self.summary["rounds"]:
            # Average metrics across rounds
            attacker_losses = []
            assessor_losses = []
            attacker_rewards = []
            assessor_rewards = []
            
            for round_entry in self.summary["rounds"]:
                if "attacker" in round_entry and round_entry["attacker"].get("loss"):
                    attacker_losses.append(round_entry["attacker"]["loss"])
                    if round_entry["attacker"].get("reward"):
                        attacker_rewards.append(round_entry["attacker"]["reward"])
                
                if "assessor" in round_entry and round_entry["assessor"].get("loss"):
                    assessor_losses.append(round_entry["assessor"]["loss"])
                    if round_entry["assessor"].get("reward"):
                        assessor_rewards.append(round_entry["assessor"]["reward"])
            
            self.summary["overall"] = {
                "total_rounds": len(self.summary["rounds"]),
                "attacker": {
                    "avg_loss": sum(attacker_losses) / len(attacker_losses) if attacker_losses else None,
                    "avg_reward": sum(attacker_rewards) / len(attacker_rewards) if attacker_rewards else None,
                },
                "assessor": {
                    "avg_loss": sum(assessor_losses) / len(assessor_losses) if assessor_losses else None,
                    "avg_reward": sum(assessor_rewards) / len(assessor_rewards) if assessor_rewards else None,
                }
            }
        
        # Write final summary
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("METRICS LOGGING COMPLETE")
        print(f"{'='*60}")
        print(f"Metrics log: {self.metrics_path}")
        print(f"Summary: {self.summary_path}")
        print(f"{'='*60}\n")


def print_metrics_summary(metrics: Dict[str, Any], phase: str):
    """Print a formatted summary of training metrics.
    
    Args:
        metrics: Dictionary of metrics
        phase: Training phase name
    """
    print(f"\n{'='*60}")
    print(f"{phase.upper()} TRAINING METRICS")
    print(f"{'='*60}")
    
    # Core metrics
    if "loss" in metrics:
        print(f"Loss: {metrics['loss']:.4f}")
    if "grad_norm" in metrics:
        print(f"Gradient Norm: {metrics['grad_norm']:.4f}")
    if "learning_rate" in metrics:
        print(f"Learning Rate: {metrics['learning_rate']:.2e}")
    
    # Reward metrics
    if "reward" in metrics:
        print(f"\nReward Statistics:")
        print(f"  Mean: {metrics['reward']:.4f}")
        if "reward_std" in metrics:
            print(f"  Std: {metrics['reward_std']:.4f}")
        if "frac_reward_zero_std" in metrics:
            print(f"  Frac Zero Std: {metrics['frac_reward_zero_std']:.2%}")
    
    # Generation metrics
    if "completions/mean_length" in metrics:
        print(f"\nGeneration Statistics:")
        print(f"  Mean Length: {metrics['completions/mean_length']:.1f} tokens")
        if "completions/min_length" in metrics:
            print(f"  Min Length: {metrics['completions/min_length']:.1f} tokens")
        if "completions/max_length" in metrics:
            print(f"  Max Length: {metrics['completions/max_length']:.1f} tokens")
    
    # Policy metrics
    if "entropy" in metrics:
        print(f"\nPolicy Statistics:")
        print(f"  Entropy: {metrics['entropy']:.4f}")
        if "clip_ratio/region_mean" in metrics:
            print(f"  Clip Ratio: {metrics['clip_ratio/region_mean']:.4f}")
    
    print(f"{'='*60}\n")
