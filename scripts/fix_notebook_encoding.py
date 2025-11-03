"""
Fix Unicode encoding issues in notebooks
Replaces Unicode characters with ASCII equivalents
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

def replace_unicode_in_text(text: str) -> str:
    """Replace Unicode characters with ASCII equivalents"""
    replacements = {
        '✓': '[OK]',
        '✔': '[OK]',
        '✗': '[X]',
        '✖': '[X]',
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        '•': '*',
        '—': '-',
        '–': '-',
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
        '…': '...',
        '≈': '~',
        '≤': '<=',
        '≥': '>=',
        'α': 'alpha',
        'β': 'beta',
        'γ': 'gamma',
        '∑': 'Sum',
        '∂': 'd',
        '∇': 'gradient',
        '∈': 'in',
        '∞': 'inf',
        '™': '(TM)',
        '®': '(R)',
        '©': '(C)',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        '°': 'deg',
        '²': '^2',
        '³': '^3',
        '½': '1/2',
        '¼': '1/4',
        '¾': '3/4',
    }

    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)

    # Also handle any remaining non-ASCII characters
    # by encoding and decoding with errors='ignore'
    text = text.encode('ascii', errors='ignore').decode('ascii')

    return text

def process_cell_content(content: Any) -> Any:
    """Process cell content to replace Unicode characters"""
    if isinstance(content, str):
        return replace_unicode_in_text(content)
    elif isinstance(content, list):
        return [process_cell_content(item) for item in content]
    elif isinstance(content, dict):
        return {key: process_cell_content(value) for key, value in content.items()}
    else:
        return content

def fix_notebook(notebook_path: Path) -> bool:
    """
    Fix Unicode encoding issues in a notebook

    Args:
        notebook_path: Path to the notebook file

    Returns:
        True if fixed successfully
    """
    try:
        print(f"Processing {notebook_path.name}...")

        # Read the notebook with UTF-8 encoding
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Process all cells
        for cell in notebook.get('cells', []):
            # Process source content
            if 'source' in cell:
                cell['source'] = process_cell_content(cell['source'])

            # Process outputs if present
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        output['text'] = process_cell_content(output['text'])
                    if 'data' in output:
                        for key, value in output['data'].items():
                            output['data'][key] = process_cell_content(value)

        # Process metadata if needed
        if 'metadata' in notebook:
            notebook['metadata'] = process_cell_content(notebook['metadata'])

        # Write the fixed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=True)

        print(f"  Fixed: {notebook_path.name}")
        return True

    except Exception as e:
        print(f"  Error fixing {notebook_path.name}: {e}")
        return False

def main():
    """Fix all tutorial notebooks"""

    print("=" * 60)
    print("Fixing Unicode Encoding in Tutorial Notebooks")
    print("=" * 60)

    # Find all tutorial notebooks
    notebook_dir = Path('notebooks/tutorials')
    notebooks = sorted(notebook_dir.glob('*.ipynb'))

    if not notebooks:
        print(f"No notebooks found in {notebook_dir}")
        return

    print(f"Found {len(notebooks)} notebooks to fix\n")

    # Fix each notebook
    success_count = 0
    for notebook_path in notebooks:
        if fix_notebook(notebook_path):
            success_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Fixed {success_count}/{len(notebooks)} notebooks")

    if success_count == len(notebooks):
        print("\nAll notebooks have been fixed!")
        print("Students should now be able to open and run them without encoding issues.")
    else:
        print(f"\nSome notebooks could not be fixed. Please check manually.")

if __name__ == "__main__":
    main()