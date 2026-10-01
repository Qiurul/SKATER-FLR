# -*- coding: utf-8 -*-
"""Unified command-line entry point."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(args):
    cmd = [sys.executable, *args]
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "full"])
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "smoke":
        run(["validate_implementation.py"])
        return
    run(["run_example.py"])
    run(["run_estimation.py", "--reps", "100", "--jobs", str(args.jobs)])
    run(["run_sensitivity.py", "--reps", "100", "--jobs", str(args.jobs)])
    run(["run_joint_test.py", "--reps", "200", "--B", "1000", "--jobs", str(args.jobs)])
    run(["run_empirical.py"])

if __name__ == "__main__":
    main()