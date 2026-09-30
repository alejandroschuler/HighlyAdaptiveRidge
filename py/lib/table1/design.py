"""Settings that every Table 1 cell uses. A change here reruns every cell."""

# A cell's train/test split and its random forest use the seed SEED + rep, so a
# cell's data depends on its name alone and not on which other cells ran.
SEED = 20240101

# The first MAX_ROWS rows of each dataset, split at random into train and test.
MAX_ROWS = 2000
TEST_FRAC = 0.2

# The number of features p after taking the first MAX_ROWS rows, as the paper's
# table prints it. data.load() checks it.
DATASET_P = {
    "power": 4, "yacht": 6, "concrete": 8, "energy": 8, "kin8nm": 8,
    "protein": 9, "wine": 11, "boston": 13, "naval": 17, "yearmsd": 90,
    "slice": 384,
}
