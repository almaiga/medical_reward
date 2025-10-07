#!/usr/bin/env python3
"""
Monitor SFT training progress by watching log files and GPU usage.
Run this in a separate terminal while training.
"""

import os
import time
import subprocess
import argparse
from datetime import datetime
import glob


def get_gpu_info():
    """Get GPU memory usage if available."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 3:
                    used, total, util = parts[0], parts[1], parts[2]
                    gpu_info.append(f"GPU {i}: {used}MB/{total}MB ({util}%)")
            return gpu_info
    except FileNotFoundError:
        pass
    return ["GPU info not available"]


def monitor_training_dir(training_dir: str):
    """Monitor training directory for progress."""
    print(f"🔍 Monitoring training directory: {training_dir}")
    
    while True:
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Training Status")
        print("-" * 60)
        
        # Check for training output directories
        pattern = os.path.join(training_dir, "*")
        training_dirs = glob.glob(pattern)
        training_dirs = [d for d in training_dirs if os.path.isdir(d)]
        
        if training_dirs:
            latest_dir = max(training_dirs, key=os.path.getctime)
            print(f"📁 Latest training: {os.path.basename(latest_dir)}")
            
            # Check for checkpoints
            checkpoint_pattern = os.path.join(latest_dir, "checkpoint-*")
            checkpoints = glob.glob(checkpoint_pattern)
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                checkpoint_num = os.path.basename(latest_checkpoint).split('-')[1]
                print(f"💾 Latest checkpoint: {checkpoint_num}")
            else:
                print("💾 No checkpoints yet")
                
            # Check for trainer_state.json
            trainer_state_file = os.path.join(latest_dir, "trainer_state.json")
            if os.path.exists(trainer_state_file):
                try:
                    import json
                    with open(trainer_state_file, 'r') as f:
                        state = json.load(f)
                    
                    if 'log_history' in state and state['log_history']:
                        latest_log = state['log_history'][-1]
                        if 'train_loss' in latest_log:
                            print(f"📉 Latest loss: {latest_log['train_loss']:.4f}")
                        if 'epoch' in latest_log:
                            print(f"📊 Epoch: {latest_log['epoch']:.2f}")
                        if 'step' in latest_log:
                            print(f"🔢 Step: {latest_log['step']}")
                except Exception as e:
                    print(f"⚠️  Could not read trainer state: {e}")
        else:
            print("📁 No training directories found yet")
        
        # Show GPU info
        gpu_info = get_gpu_info()
        print("🖥️  GPU Status:")
        for info in gpu_info:
            print(f"   {info}")
        
        print("-" * 60)
        time.sleep(30)  # Update every 30 seconds


def monitor_log_file(log_file: str):
    """Monitor a specific log file."""
    print(f"📄 Monitoring log file: {log_file}")
    
    if not os.path.exists(log_file):
        print(f"⚠️  Log file not found: {log_file}")
        return
    
    # Follow the log file
    with open(log_file, 'r') as f:
        # Go to end of file
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if line:
                print(line.strip())
            else:
                time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Monitor SFT training progress")
    parser.add_argument("--training_dir", type=str, default="trainer_output",
                       help="Training output directory to monitor")
    parser.add_argument("--log_file", type=str,
                       help="Specific log file to monitor")
    parser.add_argument("--interval", type=int, default=30,
                       help="Update interval in seconds")
    
    args = parser.parse_args()
    
    print("🔍 SFT Training Monitor")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print("=" * 50)
    
    try:
        if args.log_file:
            monitor_log_file(args.log_file)
        else:
            monitor_training_dir(args.training_dir)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")


if __name__ == "__main__":
    main()