#!/usr/bin/env python3
"""
Generate uv lock files for all requirements
Run this once, then commit lock files to git for reproducibility
"""
import subprocess
import sys
from pathlib import Path


def generate_lock_files():
    """Generate lock files for all requirements using uv"""
    requirements_dir = Path("requirements")

    if not requirements_dir.exists():
        print("ERROR: requirements/ directory not found")
        sys.exit(1)

    # Find all .txt requirement files
    requirement_files = sorted(requirements_dir.glob("*.txt"))

    if not requirement_files:
        print("ERROR: No .txt files found in requirements/")
        sys.exit(1)

    print(f"Found {len(requirement_files)} requirement files")
    print("Generating lock files with uv...\n")

    success_count = 0
    fail_count = 0

    for req_file in requirement_files:
        lock_file = req_file.with_suffix('.lock')

        print(f"{'='*60}")
        print(f"Compiling: {req_file.name} → {lock_file.name}")
        print(f"{'='*60}")

        try:
            # Check if uv is available
            subprocess.run(['uv', '--version'], check=True, capture_output=True)
        except FileNotFoundError:
            print("ERROR: 'uv' not found. Please install uv first:")
            print("  Windows: irm https://astral.sh/uv/install.ps1 | iex")
            print("  Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh")
            sys.exit(1)

        try:
            result = subprocess.run([
                'uv', 'pip', 'compile',
                str(req_file),
                '-o', str(lock_file)
            ], check=True, capture_output=True, text=True)

            print(f"✓ Generated {lock_file.name}")
            success_count += 1

        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to generate {lock_file.name}")
            print(f"Error: {e.stderr}")
            fail_count += 1

        print()

    print("="*60)
    print(f"✓ Lock file generation complete!")
    print(f"  Success: {success_count}/{len(requirement_files)}")
    if fail_count > 0:
        print(f"  Failed: {fail_count}/{len(requirement_files)}")
    print("="*60)

    if success_count > 0:
        print("\nNext steps:")
        print("  1. Review the generated .lock files")
        print("  2. Commit to git:")
        print("     git add requirements/*.lock")
        print("     git commit -m 'Add uv lock files for reproducibility'")
        print("  3. Use 'uv pip sync requirements/<file>.lock' for exact reproducibility")


if __name__ == "__main__":
    generate_lock_files()
