"""
Check specific cell in a notebook for syntax errors
"""

import json
import sys

notebook_path = "notebooks/tutorials/01_Data_Foundation.ipynb"

with open(notebook_path, 'r', encoding='utf-8', errors='ignore') as f:
    notebook = json.load(f)

# Get the second cell (index 1, since index 0 is usually markdown)
if len(notebook['cells']) > 1:
    cell = notebook['cells'][1]
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        print("Cell 1 source code:")
        print("=" * 60)
        print(source)
        print("=" * 60)

        # Try to compile it
        try:
            compile(source, "cell_1", 'exec')
            print("\nNo syntax error found in compilation")
        except SyntaxError as e:
            print(f"\nSyntax error found:")
            print(f"  Line {e.lineno}: {e.msg}")
            print(f"  Text: {e.text}")
            print(f"  Position: {' ' * (e.offset-1)}^")