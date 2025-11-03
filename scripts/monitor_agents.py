#!/usr/bin/env python3
"""
Live agent monitoring for NAM pipeline
Displays real-time progress of all 6 agents
"""
import time
import os
import sys
from pathlib import Path
from datetime import datetime

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_current_stage():
    """Extract current pipeline stage from log"""
    log_file = Path('outputs/nam_pipeline.log')

    if not log_file.exists():
        return "Pipeline not started"

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Look for stage markers in last 100 lines
        for line in reversed(lines[-100:]):
            if '[' in line and '/' in line and 'Agent' in line:
                # Extract stage info
                parts = line.split('|')
                if len(parts) > 0:
                    stage_info = parts[-1].strip()
                    return stage_info
    except Exception as e:
        return f"Error reading log: {e}"

    return "Running..."

def get_training_progress():
    """Get current training epoch if available"""
    log_file = Path('outputs/nam_pipeline.log')

    if not log_file.exists():
        return None

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Look for epoch info
        for line in reversed(lines[-50:]):
            if 'Epoch' in line and '/' in line:
                return line.strip()
    except:
        pass

    return None

def monitor():
    """Monitor agents in real-time"""
    print("\nStarting live agent monitoring...")
    print("Press Ctrl+C to stop\n")
    time.sleep(2)

    iteration = 0

    while True:
        try:
            clear_screen()

            print("="*70)
            print("NAM MULTI-AGENT SYSTEM - LIVE STATUS")
            print("="*70)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Refresh: Every 3 seconds | Iteration: {iteration}")
            print("="*70)

            # Show current stage
            current_stage = get_current_stage()
            print(f"\nCurrent Stage: {current_stage}")

            # Show training progress if available
            training_progress = get_training_progress()
            if training_progress:
                print(f"Training: {training_progress}")

            print()

            # Show agent status
            import subprocess
            result = subprocess.run(
                [sys.executable, 'scripts/check_agent_status.py'],
                capture_output=True,
                text=True
            )
            print(result.stdout)

            # Show recent log entries
            log_file = Path('outputs/nam_pipeline.log')
            if log_file.exists():
                print("\n" + "="*70)
                print("RECENT LOG ENTRIES (Last 5 lines):")
                print("="*70)
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for line in lines[-5:]:
                        # Clean up line
                        clean_line = line.strip()
                        if clean_line:
                            # Truncate very long lines
                            if len(clean_line) > 120:
                                clean_line = clean_line[:117] + "..."
                            print(clean_line)

            print("\n" + "="*70)
            print("Press Ctrl+C to stop monitoring")
            print("="*70)

            iteration += 1
            time.sleep(3)

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("Monitoring stopped by user")
            print("="*70)
            break
        except Exception as e:
            print(f"\nError during monitoring: {e}")
            time.sleep(3)

if __name__ == "__main__":
    try:
        monitor()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
