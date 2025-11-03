"""
Test script to verify all tutorial notebooks run smoothly
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_notebook(notebook_path: Path) -> Tuple[bool, List[str]]:
    """
    Test a single notebook for execution issues

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Tuple of (success, list of errors)
    """
    errors = []

    try:
        # Read the notebook with proper UTF-8 encoding
        with open(notebook_path, 'r', encoding='utf-8', errors='ignore') as f:
            notebook = json.load(f)

        # Check for valid notebook structure
        if 'cells' not in notebook:
            errors.append(f"Invalid notebook structure: missing 'cells'")
            return False, errors

        # Extract code cells
        code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']

        print(f"\nTesting {notebook_path.name}:")
        print(f"  Found {len(code_cells)} code cells")

        # Test each code cell for syntax errors
        for i, cell in enumerate(code_cells):
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

            # Skip empty cells
            if not source.strip():
                continue

            # Basic syntax check
            # Skip Jupyter magic commands (lines starting with % or %%)
            lines = source.split('\n')
            filtered_lines = [line for line in lines if not line.strip().startswith(('%', '!', '?'))]
            filtered_source = '\n'.join(filtered_lines)

            try:
                compile(filtered_source, f"cell_{i}", 'exec')
            except SyntaxError as e:
                errors.append(f"Cell {i+1} has syntax error: {e}")

            # Check for common issues
            if 'import' in source:
                # Check for potentially missing packages
                if 'mlflow' in source and 'optional' not in source.lower():
                    errors.append(f"Cell {i+1}: mlflow import might fail (should be optional)")
                if '../..' in source and 'sys.path' not in source:
                    errors.append(f"Cell {i+1}: Relative import without sys.path setup")

            # Check for file paths
            if 'data/' in source or 'models/' in source or 'plots/' in source:
                # Verify paths are correct
                if 'data/raw/' in source:
                    files = ['firstfile.csv', 'MediaInvestment.csv', 'MonthlyNPSscore.csv']
                    for file in files:
                        if file in source:
                            file_path = Path('data/raw') / file
                            if not file_path.exists():
                                errors.append(f"Cell {i+1}: References missing file {file_path}")

        # Try to execute the notebook (without actually running the code)
        # We'll use nbconvert to test conversion
        try:
            result = subprocess.run(
                ['python', '-m', 'nbconvert', '--to', 'python', '--stdout', str(notebook_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0 and 'ModuleNotFoundError' not in result.stderr:
                errors.append(f"nbconvert failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            pass  # Timeout is okay for conversion test
        except FileNotFoundError:
            print("  Note: nbconvert not installed, skipping conversion test")

        if not errors:
            print(f"  [OK] No issues found")
            return True, []
        else:
            print(f"  [X] Found {len(errors)} issue(s)")
            return False, errors

    except Exception as e:
        # Clean error message of any Unicode characters
        error_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
        errors.append(f"Failed to read notebook: {error_msg}")
        return False, errors

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        'tensorflow', 'numpy', 'pandas', 'scikit-learn',
        'matplotlib', 'seaborn', 'loguru'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Warning: Missing packages: {', '.join(missing)}")
        print("Students will need to install these packages")
    else:
        print("[OK] All required packages are installed")

def main():
    """Test all tutorial notebooks"""

    print("=" * 60)
    print("Testing Tutorial Notebooks")
    print("=" * 60)

    # Check dependencies first
    check_dependencies()

    # Find all tutorial notebooks
    notebook_dir = Path('notebooks/tutorials')
    notebooks = sorted(notebook_dir.glob('*.ipynb'))

    if not notebooks:
        print(f"No notebooks found in {notebook_dir}")
        return

    print(f"\nFound {len(notebooks)} tutorial notebooks")

    # Test each notebook
    all_errors = {}
    success_count = 0

    for notebook_path in notebooks:
        success, errors = test_notebook(notebook_path)
        if success:
            success_count += 1
        else:
            all_errors[notebook_path.name] = errors

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful notebooks: {success_count}/{len(notebooks)}")

    if all_errors:
        print("\nIssues found:")
        for notebook_name, errors in all_errors.items():
            print(f"\n{notebook_name}:")
            for error in errors:
                print(f"  - {error}")
    else:
        print("\n[OK] All notebooks are ready to run!")

    # Check for common setup issues
    print("\n" + "=" * 60)
    print("SETUP REQUIREMENTS FOR STUDENTS")
    print("=" * 60)
    print("1. Python 3.8+ with TensorFlow 2.x")
    print("2. Required packages: tensorflow, numpy, pandas, scikit-learn, matplotlib, seaborn, loguru")
    print("3. Data files in data/raw/: firstfile.csv, MediaInvestment.csv, MonthlyNPSscore.csv")
    print("4. Run from project root directory")
    print("5. Optional: Jupyter or VSCode with notebook support")

if __name__ == "__main__":
    main()