"""One-command reproduction of the HAR paper's computational results.

Runs, in order: environment check, equivalence tests, Figure 1, Figure 2, the Table 1
benchmark (resumable) + assembly + comparison to the paper, and the reviewer-requested
experiments. Each step shells out to its module so failures are isolated and the heavy
benchmark can resume from checkpoints.

    venv/bin/python -m repro.run_all              # everything
    venv/bin/python -m repro.run_all --skip-benchmark --skip-revision
"""
import argparse
import subprocess
import sys

PY = [sys.executable]


def step(title, args, cwd=None):
    print(f"\n{'='*70}\n# {title}\n{'='*70}", flush=True)
    r = subprocess.run(PY + args, cwd=cwd)
    if r.returncode != 0:
        print(f"!! step failed: {title} (exit {r.returncode})", flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--skip-revision", action="store_true")
    args = ap.parse_args()

    if not args.skip_tests:
        step("equivalence tests", ["-m", "pytest", "tests.py", "-q"])
    step("Figure 1 (fits.pdf)", ["-m", "repro.fig1_fits"])
    step("Figure 2 (convergence.pdf)", ["-m", "repro.fig2_convergence"])
    if not args.skip_benchmark:
        step("Table 1 benchmark (resumable)", ["-m", "repro.table1_benchmark", "--reps", "5"])
        step("Table 1 assemble", ["-m", "repro.table1_assemble"])
        step("Table 1 vs paper", ["-m", "repro.compare_table1"])
    if not args.skip_revision:
        step("reviewer experiments", ["-m", "repro.revision", "all"])
    print("\nDone. Artefacts in HAR/results/repro/.", flush=True)


if __name__ == "__main__":
    main()
