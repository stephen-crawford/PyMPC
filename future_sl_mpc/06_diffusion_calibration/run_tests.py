#!/usr/bin/env python3
"""Run tests for diffusion calibration (Section 6.6)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion_calibration import run_tests
sys.exit(0 if run_tests() else 1)
