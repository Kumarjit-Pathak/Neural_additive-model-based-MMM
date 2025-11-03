#!/usr/bin/env python3
"""
Setup individual agent environments using uv
Much faster than pip-based setup
"""
import subprocess
import sys
from pathlib import Path


def setup_agent_environment(agent_name, agent_id):
    """
    Setup environment for a specific agent using uv

    Args:
        agent_name: Name of agent (e.g., 'data', 'model')
        agent_id: Agent ID (e.g., '01', '02')
    """
    print(f"\n{'='*60}")
    print(f"Setting up environment for Agent {agent_id}: {agent_name}")
    print(f"{'='*60}")

    # Create virtual environment with uv (very fast!)
    venv_path = f".venv_agent_{agent_id}"
    print(f"Creating virtual environment: {venv_path}")

    try:
        subprocess.run(['uv', 'venv', venv_path], check=True)
    except FileNotFoundError:
        print("ERROR: 'uv' not found. Please install uv first:")
        print("  Windows: irm https://astral.sh/uv/install.ps1 | iex")
        print("  Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

    # Determine activation script
    if sys.platform == 'win32':
        activate_script = f"{venv_path}\\Scripts\\activate"
        python_exe = f"{venv_path}\\Scripts\\python.exe"
    else:
        activate_script = f"source {venv_path}/bin/activate"
        python_exe = f"{venv_path}/bin/python"

    print(f"✓ Virtual environment created")
    print(f"  Activate with: {activate_script}")

    # Install dependencies from lock file (reproducible)
    requirements_lock = Path(f"requirements/agent_{agent_id}_{agent_name}.lock")
    requirements_txt = Path(f"requirements/agent_{agent_id}_{agent_name}.txt")

    if requirements_lock.exists():
        print(f"\nInstalling from lock file: {requirements_lock.name}")
        subprocess.run([
            'uv', 'pip', 'install',
            '--python', python_exe,
            '-r', str(requirements_lock)
        ], check=True)
    elif requirements_txt.exists():
        # Fallback to requirements.txt
        print(f"\nInstalling from requirements: {requirements_txt.name}")
        subprocess.run([
            'uv', 'pip', 'install',
            '--python', python_exe,
            '-r', str(requirements_txt)
        ], check=True)
    else:
        print(f"WARNING: No requirements file found for agent {agent_id}")

    print(f"\n✓ Agent {agent_id} ({agent_name}) environment ready!")
    return venv_path


def setup_all_agents():
    """Setup environments for all 6 agents in parallel"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    agents = [
        ('data', '01'),
        ('model', '02'),
        ('training', '03'),
        ('evaluation', '04'),
        ('business', '05'),
        ('testing', '06')
    ]

    print("\n" + "="*60)
    print("Setting up all agent environments in parallel using uv")
    print("This is much faster than sequential pip install!")
    print("="*60)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(setup_agent_environment, name, id): (name, id)
            for name, id in agents
        }

        for future in as_completed(futures):
            name, id = futures[future]
            try:
                venv_path = future.result()
                print(f"\n✓ Agent {id} ({name}) completed")
            except Exception as e:
                print(f"\n✗ Agent {id} ({name}) failed: {e}")

    print("\n" + "="*60)
    print("✓ All agent environments ready!")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Setup agent environments with uv')
    parser.add_argument('--agent', help='Specific agent name (data, model, training, etc.)')
    parser.add_argument('--id', help='Agent ID (01-06)')
    parser.add_argument('--all', action='store_true', help='Setup all agents in parallel')

    args = parser.parse_args()

    if args.all:
        setup_all_agents()
    elif args.agent and args.id:
        setup_agent_environment(args.agent, args.id)
    else:
        print("Usage:")
        print("  python scripts/setup_agent_environment.py --all")
        print("  python scripts/setup_agent_environment.py --agent data --id 01")
