#!/usr/bin/env python3
"""
Check status of all agents
Shows progress and completion status
"""
import json
from pathlib import Path
from datetime import datetime


def load_agent_progress(agent_id, agent_name):
    """Load progress for a specific agent"""
    progress_file = Path(f".agents/agent_{agent_id}_{agent_name}/progress.json")

    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    else:
        return {}


def check_all_agents():
    """Check status of all 6 agents"""
    agents = [
        ('01', 'data'),
        ('02', 'model'),
        ('03', 'training'),
        ('04', 'evaluation'),
        ('05', 'business'),
        ('06', 'testing')
    ]

    print("="*70)
    print("NAM Project - Agent Status Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    total_tasks = 0
    total_completed = 0

    for agent_id, agent_name in agents:
        progress = load_agent_progress(agent_id, agent_name)

        # Count tasks
        if progress:
            tasks = len(progress)
            completed = sum(1 for status in progress.values() if status == 'completed')
        else:
            tasks = 0
            completed = 0

        total_tasks += tasks
        total_completed += completed

        # Progress percentage
        if tasks > 0:
            pct = (completed / tasks) * 100
            progress_bar = create_progress_bar(pct)
        else:
            pct = 0
            progress_bar = create_progress_bar(0)

        # Print agent status
        print(f"\nAgent {agent_id}: {agent_name.upper()}")
        print(f"  {progress_bar} {pct:.0f}% ({completed}/{tasks})")

        # Show incomplete tasks
        if progress:
            incomplete = [task for task, status in progress.items() if status != 'completed']
            if incomplete:
                print(f"  Pending:")
                for task in incomplete[:3]:  # Show first 3
                    print(f"    - {task}")
                if len(incomplete) > 3:
                    print(f"    ... and {len(incomplete) - 3} more")

    # Overall summary
    print("\n" + "="*70)
    if total_tasks > 0:
        overall_pct = (total_completed / total_tasks) * 100
        print(f"Overall Progress: {total_completed}/{total_tasks} tasks ({overall_pct:.1f}%)")
    else:
        print("Overall Progress: No tasks tracked yet")
        print("\nNote: Agents will create progress.json files as they work")

    print("="*70)


def create_progress_bar(percentage, width=30):
    """Create ASCII progress bar"""
    filled = int((percentage / 100) * width)
    bar = '#' * filled + '-' * (width - filled)
    return f"[{bar}]"


if __name__ == "__main__":
    check_all_agents()
