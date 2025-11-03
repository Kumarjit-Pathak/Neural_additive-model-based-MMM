#!/usr/bin/env python3
"""
Verify project setup is complete
Checks folders, files, and configurations
"""
from pathlib import Path
import sys


def check_directory_structure():
    """Check if all required directories exist"""
    required_dirs = [
        '.agents/orchestrator',
        '.agents/agent_01_data',
        '.agents/agent_02_model',
        '.agents/agent_03_training',
        '.agents/agent_04_evaluation',
        '.agents/agent_05_business',
        '.agents/agent_06_testing',
        'data/raw',
        'data/processed',
        'data/features',
        'data/splits',
        'src/data',
        'src/models',
        'src/training',
        'src/evaluation',
        'src/optimization',
        'src/utils',
        'tests/unit',
        'tests/integration',
        'tests/fixtures',
        'configs',
        'scripts',
        'notebooks',
        'requirements',
        'outputs/models',
        'outputs/figures',
        'outputs/reports',
        'experiments/mlruns'
    ]

    print("Checking directory structure...")
    missing_dirs = []

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"[X] Missing {len(missing_dirs)} directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    else:
        print(f"[OK] All {len(required_dirs)} directories present")
        return True


def check_requirements_files():
    """Check if all requirements files exist"""
    required_files = [
        'requirements/base.txt',
        'requirements/agent_01_data.txt',
        'requirements/agent_02_model.txt',
        'requirements/agent_03_training.txt',
        'requirements/agent_04_evaluation.txt',
        'requirements/agent_05_business.txt',
        'requirements/agent_06_testing.txt'
    ]

    print("\nChecking requirements files...")
    missing_files = []

    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"[X] Missing {len(missing_files)} files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print(f"[OK] All {len(required_files)} requirements files present")
        return True


def check_agent_configs():
    """Check if all agent configurations exist"""
    required_configs = [
        '.agents/orchestrator/config.yaml',
        '.agents/orchestrator/workflow.yaml',
        '.agents/agent_01_data/config.yaml',
        '.agents/agent_01_data/tasks.md',
        '.agents/agent_02_model/config.yaml',
        '.agents/agent_02_model/tasks.md',
        '.agents/agent_03_training/config.yaml',
        '.agents/agent_03_training/tasks.md',
        '.agents/agent_04_evaluation/config.yaml',
        '.agents/agent_04_evaluation/tasks.md',
        '.agents/agent_05_business/config.yaml',
        '.agents/agent_05_business/tasks.md',
        '.agents/agent_06_testing/config.yaml',
        '.agents/agent_06_testing/tasks.md'
    ]

    print("\nChecking agent configurations...")
    missing_configs = []

    for config_path in required_configs:
        if not Path(config_path).exists():
            missing_configs.append(config_path)

    if missing_configs:
        print(f"[X] Missing {len(missing_configs)} configurations:")
        for config_path in missing_configs:
            print(f"  - {config_path}")
        return False
    else:
        print(f"[OK] All {len(required_configs)} agent configurations present")
        return True


def check_project_configs():
    """Check if project configuration files exist"""
    required_configs = [
        'configs/model_config.yaml',
        'configs/training_config.yaml',
        'configs/data_config.yaml',
        'pyproject.toml',
        'pytest.ini'
    ]

    print("\nChecking project configurations...")
    missing_configs = []

    for config_path in required_configs:
        if not Path(config_path).exists():
            missing_configs.append(config_path)

    if missing_configs:
        print(f"[X] Missing {len(missing_configs)} configurations:")
        for config_path in missing_configs:
            print(f"  - {config_path}")
        return False
    else:
        print(f"[OK] All {len(required_configs)} project configurations present")
        return True


def check_uv_installed():
    """Check if uv is installed"""
    import subprocess

    print("\nChecking uv installation...")

    try:
        result = subprocess.run(
            ['uv', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        print(f"[OK] uv is installed: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[X] uv is not installed")
        print("  Install with:")
        print("    Windows: irm https://astral.sh/uv/install.ps1 | iex")
        print("    Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def main():
    """Run all verification checks"""
    print("="*70)
    print("NAM Project Setup Verification")
    print("="*70)

    checks = [
        check_uv_installed(),
        check_directory_structure(),
        check_requirements_files(),
        check_agent_configs(),
        check_project_configs()
    ]

    print("\n" + "="*70)
    if all(checks):
        print("[OK] Setup verification PASSED")
        print("="*70)
        print("\nNext steps:")
        print("  1. Generate lock files: python scripts/generate_lock_files.py")
        print("  2. Setup agent environments: python scripts/setup_agent_environment.py --all")
        print("  3. Place data files in data/raw/")
        print("  4. Review agent tasks in .agents/agent_XX_*/tasks.md")
        return 0
    else:
        print("[X] Setup verification FAILED")
        print("="*70)
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
