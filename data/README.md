# Input data

`uci/` holds the eleven UCI regression datasets of Table 1, one csv file per
dataset: boston, concrete, energy, kin8nm, naval, power, protein, slice, wine,
yacht and yearmsd. In each file the features are every column but the last, and
the target is the last column. The benchmark uses the first 2000 rows.

The files are not in git. On the machine where this project was set up, `uci`
is a symbolic link to `../../csv`, a folder that other projects share:

```
ln -s ../../csv data/uci
```

Elsewhere, put the files in `data/uci/` or point the link at a copy.
