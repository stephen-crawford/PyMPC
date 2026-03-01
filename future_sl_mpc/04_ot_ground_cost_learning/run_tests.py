#!/usr/bin/env python3
"""Run tests for OT ground cost learning (Section 6.4)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_and_cost import run_tests
sys.exit(0 if run_tests() else 1)
