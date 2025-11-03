"""
Pytest configuration and shared fixtures
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import fixtures from fixtures module
pytest_plugins = ['tests.fixtures.sample_data']
