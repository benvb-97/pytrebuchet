"""Path utilities for PyTrebuchet."""

from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

# Define other important paths
SOURCE_DIR = ROOT_DIR / "src" / "pytrebuchet"
