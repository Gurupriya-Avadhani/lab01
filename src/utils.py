#!/usr/bin/env python3
"""
Utility functions for MLOps pipeline.
"""

from typing import Dict, Any
import json
from pathlib import Path
import psutil


def ensure_directory(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary as JSON with nice formatting."""
    ensure_directory(Path(filepath).parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file as dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_section(title: str, length: int = 60) -> None:
    """Print formatted section header."""
    print("\n" + "=" * length)
    print(f" {title}")
    print("=" * length + "\n")


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


if __name__ == '__main__':
    print_section("Pipeline Utilities")
    print("✓ json helpers loaded")
    print("✓ directory utils loaded")
    print("✓ logger helpers loaded")