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
    run(["-m", "simulation.run_example"])
    run(["-m", "simulation.run_estimation", "--reps", "100", "--jobs", str(args.jobs)])
    run(["-m", "simulation.run_sensitivity", "--reps", "100", "--jobs", str(args.jobs)])
    run(["-m", "simulation.run_joint_test", "--reps", "200", "--B", "1000", "--jobs", str(args.jobs)])
    run(["-m", "empirical.run_empirical"])

if __name__ == "__main__":
    main()
