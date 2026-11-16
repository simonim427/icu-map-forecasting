"""Regenerate everything from data/raw: clean data, forecast pairs, every model, tables and figures.

Usage:  uv run python run_all.py

Runs the numbered scripts in scripts/ in order, so a new script (for example
scripts/10_gru_d.py) is picked up automatically.
"""
import subprocess
import sys
from pathlib import Path

for script in sorted((Path(__file__).parent / "scripts").glob("[0-9]*.py")):
    print(f"\n=== {script.name} ===", flush=True)
    subprocess.run([sys.executable, str(script)], check=True)
