"""Central configuration for the reproduction harness: paths, seeds, grids."""
from pathlib import Path

# Input data (the paper's UCI csvs; features in all-but-last column, target last).
DATA_DIR = Path("/Users/aschuler/Documents/research/projects/csv")

# Output lives under HAR/results/repro so the original committed results stay intact
# for side-by-side comparison.
HAR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = HAR_ROOT / "results"
REPRO_DIR = RESULTS_DIR / "repro"
PLOTS_DIR = REPRO_DIR / "plots"
DATA_OUT = REPRO_DIR / "data"
SIMS_OUT = REPRO_DIR / "sims"

# Reference figures shipped with the manuscript (for visual comparison).
PAPER_DIR = HAR_ROOT.parent / "paper"

# Base seed; per-rep seeds are derived as SEED + rep so every run is repeatable.
SEED = 20240101

# Benchmark (Table 1) settings.
MAX_ROWS = 2000
TEST_FRAC = 0.2
N_REPS = 5

# Kernel-ridge regularization search (paper appendix: 50-point log grid in [0, lambda_0]).
N_ALPHAS = 50
GAMMAS = [0.001, 0.01, 0.1, 1, 10]   # radial-basis bandwidth grid

# Baselines.
RF_TREES = 2000
RIDGE_ALPHA = 1e-3

# Datasets in paper Table 1 order, with expected feature count p after taking
# the first MAX_ROWS rows (asserted at load time).
DATASETS = [
    "power", "yacht", "concrete", "energy", "kin8nm", "protein",
    "wine", "boston", "naval", "yearmsd", "slice",
]
DATASET_P = {
    "power": 4, "yacht": 6, "concrete": 8, "energy": 8, "kin8nm": 8,
    "protein": 9, "wine": 11, "boston": 13, "naval": 17, "yearmsd": 90, "slice": 384,
}
# HAL is only run on the small datasets (it is computationally impractical otherwise).
HAL_DATASETS = ["yacht", "energy", "boston", "concrete"]


def ensure_dirs():
    for d in (REPRO_DIR, PLOTS_DIR, DATA_OUT, SIMS_OUT):
        d.mkdir(parents=True, exist_ok=True)
