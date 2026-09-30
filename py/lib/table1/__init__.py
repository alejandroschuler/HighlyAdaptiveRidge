"""The Table 1 benchmark (tab:empirical): six methods on eleven UCI datasets.

A cell is one dataset and one repetition. The Snakefile lists the cells and
declares, for each rule, only the files here that its cells use, because the
HAL cells take hours and should not rerun after an unrelated edit.
"""
