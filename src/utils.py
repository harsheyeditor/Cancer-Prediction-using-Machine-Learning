"""
Utilities Module - Cancer Prediction Project
=============================================

Shared utility functions used across multiple notebooks and modules.
Contains helpers for display formatting, configuration, and common
operations that don't belong to a specific domain module.

Author: Cancer Prediction ML Project
"""

import os
import sys
import logging
from typing import Any, Optional


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure project-wide logging with a consistent format.

    Why use logging instead of print()?
    ------------------------------------
    1. Log levels (DEBUG, INFO, WARNING, ERROR) let you control verbosity
    2. Timestamps help track execution flow
    3. Module names help locate the source of messages
    4. Production systems use log aggregation tools that expect structured logs

    Args:
        level: Logging level (default: INFO)

    Returns:
        logging.Logger: Configured root logger
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("cancer_prediction")
    logger.info("Logging configured at level: %s", logging.getLevelName(level))
    return logger


def add_src_to_path() -> None:
    """
    Add the src/ directory to Python's import path.

    This allows notebooks in the notebooks/ folder to import
    modules from the src/ folder using standard import syntax:

        from data_loader import load_from_sklearn

    Instead of messy relative imports or sys.path hacks scattered
    throughout every notebook.

    Why is this needed?
    -------------------
    Python only searches for imports in:
    1. The directory of the running script
    2. Directories in PYTHONPATH
    3. The default installation directories

    Since our notebooks are in notebooks/ and our modules are in src/,
    we need to explicitly add src/ to the search path.
    """
    # Navigate from notebooks/ up to project root, then into src/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(project_root, "src")

    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.

    Returns:
        str: Absolute path to project root
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.

    Args:
        value: Decimal value (e.g., 0.9567)
        decimals: Number of decimal places (default: 2)

    Returns:
        str: Formatted percentage (e.g., "95.67%")

    Example:
        >>> format_percentage(0.9567)
        '95.67%'
    """
    return f"{value * 100:.{decimals}f}%"


def print_section_header(title: str, emoji: str = "[*]") -> None:
    """
    Print a visually distinct section header for notebook output.

    Args:
        title: Section title text
        emoji: Leading symbol (default: [*])
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  {emoji} {title}")
    print(f"{separator}\n")


def print_metric(name: str, value: Any, target: Optional[float] = None) -> None:
    """
    Print a metric with optional target comparison.

    Args:
        name: Metric name
        value: Metric value
        target: Optional target value for pass/fail indicator

    Example:
        >>> print_metric("Accuracy", 0.9567, target=0.95)
        [PASS] Accuracy: 95.67% (Target: 95.00%)
    """
    if isinstance(value, float) and value <= 1.0:
        formatted = format_percentage(value)
    else:
        formatted = str(value)

    if target is not None:
        target_str = format_percentage(target)
        status = "[PASS]" if value >= target else "[BELOW TARGET]"
        print(f"  {status} {name}: {formatted} (Target: {target_str})")
    else:
        print(f"  [*] {name}: {formatted}")
