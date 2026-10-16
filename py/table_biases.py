"""Section 4.3: the tables of the inductive-bias simulation, one for each claim
(tab:bias-waves, tab:bias-rules, tab:bias-density), each estimator's mean test
MSE on each DGP, and the numbers file with the ratios that the text quotes, the
edge checks of appendix C and the settings of the simulation, read from the
stored results and from the DGP code that ran."""
import numpy as np
import pandas as pd

import artefacts as art
from biases import dgps, report, summary, variants
from display import tables

frames = [pd.read_csv(art.track_read(p)) for p in snakemake.input.results]
tuning = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.tunings], ignore_index=True)
allres = pd.concat(frames, ignore_index=True)
dgp_list = list(snakemake.params.dgps)
methods = list(snakemake.params.methods)
names = dict(snakemake.params.names)
# The main run, its edge checks, and the frequency check, whose DGPs are the
# waves at other frequencies.
freq_dgps = [d for d in dgps.WAVES if d not in dgp_list]
edges = allres[allres["method"].isin(list(variants.VARIANTS))]
check = allres[allres["dgp"].isin(freq_dgps) & allres["method"].isin(methods)]
results = allres[allres["dgp"].isin(dgp_list) & allres["method"].isin(methods)]

# ---- the tables, one for each claim
for claim, claim_dgps in report.CLAIM_TABLES.items():
    body = report.table(results, claim_dgps, methods, lambda m: tables._header(m, names))
    art.save_table(getattr(snakemake.output, claim), body, label=f"tab:bias-{claim}", escape=False,
                   column_format="l|" + "r" * len(methods))

# ---- the frequency check, on its repetitions, with the main run's waves on the same ones
check_reps = sorted(check["rep"].unique())
waves = pd.concat([check, results[results["dgp"].isin(report.WAVES) & results["rep"].isin(check_reps)]])
by_periods = {}
for name, (k, periods) in dgps.WAVES.items():
    by_periods.setdefault(periods, {})[k] = name
by_periods = dict(sorted(by_periods.items()))
k_values = sorted(next(iter(by_periods.values())))
art.save_table(snakemake.output.frequency, report.frequency_table(waves, by_periods, k_values),
               label="tab:bias-frequency", escape=False, column_format="r|" + "c" * len(k_values))

# ---- the numbers
m = summary.means(results)


def mse(x):
    """A mean test MSE, rounded as in the tables."""
    return rf"\num{{{report.mse_cell(x)}}}"



def rng(values, digits=2):
    """The smallest and the largest of some ratios, as two macros' values."""
    return art.num(min(values), digits), art.num(max(values), digits)


def ratio_ci(r, digits=2):
    return rf"{art.num(r['ratio'], digits)} {art.ci(r['low'], r['high'], digits)}"


numbers = {}

# Claim 1: the wave along k covariates.
one = report.ratios(results, [("wave1", b, "wave1", "rbf") for b in report.BV])
numbers["resBiasWaveOneLow"], numbers["resBiasWaveOneHigh"] = rng([r["ratio"] for r in one])
numbers["resBiasWaveOneWinsLow"] = art.pct(100 * min(r["wins"] for r in one), 0)
for k, word in zip([2, 4], ["Two", "Four"]):
    rs = report.ratios(results, [(f"wave{k}", b, f"wave{k}", "rbf") for b in report.BV])
    numbers[f"resBiasWave{word}Low"], numbers[f"resBiasWave{word}High"] = rng([r["ratio"] for r in rs], 1)
rbf = [m.loc[(d, "rbf"), "mean"] for d in report.WAVES]
numbers["resBiasWaveRbfLow"], numbers["resBiasWaveRbfHigh"] = mse(min(rbf)), mse(max(rbf))
for d, word in zip(report.WAVES, ["One", "Two", "Three", "Four"]):
    bv = [m.loc[(d, b), "mean"] for b in report.BV]
    numbers[f"resBiasWave{word}BvLow"], numbers[f"resBiasWave{word}BvHigh"] = mse(min(bv)), mse(max(bv))
# the depth that the depth-tuned estimators chose on each wave
depth_tuned = results[results["method"].isin(["har", "mixed_sobolev_depth", "hal"])
                      & results["dgp"].isin(report.WAVES)]
k_of = depth_tuned["dgp"].str[4:].astype(int)
numbers["resBiasDepthAtLeastK"] = art.pct(100 * float(np.mean(depth_tuned["chosen_depth"] >= k_of)), 0)
numbers["resBiasDepthOneOnWaveOne"] = art.pct(
    100 * float(np.mean(depth_tuned.loc[k_of == 1, "chosen_depth"] == 1)), 0)

# Claim 2: steps and ramps.
steps = report.ratios(results, [("rules", a, "rules", b) for a in report.STEPS for b in report.SQUARED])
ramps = report.ratios(results, [("ramps", b, "ramps", a) for a in report.STEPS for b in report.SQUARED])
numbers["resBiasStepsLow"], numbers["resBiasStepsHigh"] = rng([r["ratio"] for r in steps])
numbers["resBiasRampsLow"], numbers["resBiasRampsHigh"] = rng([r["ratio"] for r in ramps])

# Claim 3: dense, sparse and flipped.
dense = summary.contrast(results, "dense", "har", "dense", "mixed_sobolev_depth")
sparse = summary.contrast(results, "sparse", "mixed_sobolev_depth", "sparse", "har")
flip_har = summary.contrast(results, "flipped", "har", "dense", "har")
flip_hal = summary.contrast(results, "flipped", "hal", "dense", "hal")
flipped = summary.contrast(results, "flipped", "har", "flipped", "mixed_sobolev_depth")
numbers["resBiasDenseHarMs"] = ratio_ci(dense)
numbers["resBiasDenseHarMsWins"] = art.pct(100 * dense["wins"], 0)
numbers["resBiasSparseMsHar"] = ratio_ci(sparse)
numbers["resBiasSparseMsHarWins"] = art.pct(100 * sparse["wins"], 0)
numbers["resBiasFlipHar"] = ratio_ci(flip_har)
numbers["resBiasFlipHarWorse"] = art.pct(100 * (1 - flip_har["wins"]), 0)
numbers["resBiasFlipHal"] = ratio_ci(flip_hal)
numbers["resBiasFlippedHarMs"] = ratio_ci(flipped)
numbers["resBiasFlipMsMaxRel"] = rf"\num{{{report.flip_max_rel(results, 'mixed_sobolev_depth'):.0e}}}"

# Monte Carlo error: the largest SE of a mean in the table relative to the
# mean, and the smallest |z| of a contrast that a claim promises.
numbers["resBiasMaxRelSe"] = art.pct(100 * float((m["se"] / m["mean"]).max()), 1)
cf = report.contrast_frame(results)
numbers["resBiasMinZ"] = art.int(float(cf["z"].abs().min()))
numbers["resBiasNContrasts"] = art.int(len(cf))

# The edge checks of appendix C.
rbf1 = results[(results["dgp"] == "wave1") & (results["method"] == "rbf")]
numbers["resBiasRbfEdgeShare"] = art.pct(100 * float(np.mean(np.isclose(rbf1["chosen_scale"], 2.0 ** -6))), 0)
numbers["resBiasRbfMseWaveOne"] = mse(m.loc[("wave1", "rbf"), "mean"])
wide = edges[edges["method"] == "rbf_wide"]
numbers["resBiasRbfWideMse"] = mse(wide["mse"].mean())
numbers["resBiasRbfWideScaleLow"], numbers["resBiasRbfWideScaleHigh"] = (
    art.int(np.log2(wide["chosen_scale"]).min()), art.int(np.log2(wide["chosen_scale"]).max()))
bv1 = [m.loc[("wave1", b), "mean"] for b in report.BV]
numbers["resBiasBvMseWaveOneHigh"] = mse(max(bv1))
deep = edges[edges["method"] == "gbt_deep"]
gbt34 = results[(results["method"] == "gbt") & results["dgp"].isin(["wave3", "wave4"])]
numbers["resBiasGbtEdgeShare"] = art.pct(100 * float(np.mean(gbt34["chosen_depth"] == max(variants.gbt.DEPTHS))), 0)
for d, word in zip(["wave3", "wave4"], ["Three", "Four"]):
    numbers[f"resBiasGbtMse{word}"] = mse(m.loc[(d, "gbt"), "mean"])
    numbers[f"resBiasGbtDeepMse{word}"] = mse(deep.loc[deep["dgp"] == d, "mse"].mean())
    numbers[f"resBiasRbfMse{word}"] = mse(m.loc[(d, "rbf"), "mean"])
numbers["mthBiasFreqReps"] = art.int(len(check_reps))
numbers["mthBiasCheckPeriods"] = art.numlist(dgps.CHECK_PERIODS)
numbers["resBiasGbtDeepAddedShare"] = art.pct(
    100 * float(np.mean(deep["chosen_depth"] > max(variants.gbt.DEPTHS))), 0)

# The other edges of the grids in the main run: the final penalty of a kernel
# method or of HAL at either end of its grid, boosting's cap on the trees, and
# its smallest learning rate, with the mean relative drop in CV risk from the
# next learning rate at the same depth, over the fits that chose it.
pen = results[results["method"].isin(["har", "mixed_sobolev_depth", "rbf", "hal"])]
at_end = (pen["chosen_alpha_index"] == 0) | (pen["chosen_alpha_index"] == pen["chosen_n_alphas"] - 1)
numbers["resBiasPenaltyEdgeFits"] = art.int(int(at_end.sum()))
numbers["resBiasPenaltyFits"] = art.int(len(pen))
gb = results[results["method"] == "gbt"]
numbers["resBiasGbtCapShare"] = art.pct(100 * float(np.mean(gb["chosen_n_trees"] >= gb["chosen_max_trees"])), 0)
rates = sorted(variants.gbt.RATES)
low = gb[np.isclose(gb["chosen_learning_rate"], rates[0])]
gt = tuning[(tuning["method"] == "gbt") & tuning["dgp"].isin(dgp_list)].set_index(
    ["dgp", "rep", "depth", "learning_rate"])["cv_risk"]
gains = [gt.loc[(r.dgp, r.rep, r.chosen_depth, rates[1])] / gt.loc[(r.dgp, r.rep, r.chosen_depth, rates[0])] - 1
         for r in low.itertuples()]
numbers["resBiasGbtRateShare"] = art.pct(100 * len(low) / len(gb), 0)
numbers["resBiasGbtRateGain"] = art.pct(100 * float(np.mean(gains)), 1)
numbers["mthBiasGbtRateLow"] = art.num(rates[0], 2)
numbers["mthBiasGbtRateNext"] = art.num(rates[1], 1)

# ---- the settings, from the stored results and the DGP code that ran
def single(column):
    values = results[column].unique()
    if len(values) != 1:
        raise ValueError(f"{column} took the values {list(values)}, not one")
    return values[0]


reps = results.groupby(["dgp", "method"])["rep"].nunique()
if reps.nunique() != 1:
    raise ValueError(f"the cells have different numbers of repetitions: {reps.to_dict()}")
sigma = float(single("sigma"))
numbers.update(
    mthBiasN=art.int(single("n")), mthBiasNTest=art.int(single("n_test")), mthBiasP=art.int(single("p")),
    mthBiasReps=art.int(reps.iloc[0]), mthBiasSigma=art.num(sigma, 1),
    mthBiasRsq=art.pct(100 / (1 + sigma ** 2), 0),
    mthBiasWavePeriods=art.num(dgps.WAVE_PERIODS, 0) if float(dgps.WAVE_PERIODS).is_integer()
    else art.num(dgps.WAVE_PERIODS, 1),
    mthBiasRuleCentres="(" + ", ".join(rf"\num{{{c:g}}}" for c in dgps.RULE_CENTRES) + ")",
    mthBiasRampWidth=art.num(dgps.RAMP_WIDTH, 1),
    mthBiasDenseTop=art.num(dgps.DENSE_TOP, 1),
    mthBiasDenseMass=art.pct(100 * dgps.DENSE_MASS, 0),
    mthBiasDensityHigh=art.num(dgps.DENSE_MASS / dgps.DENSE_TOP, 0),
    mthBiasDensityLow=art.num((1 - dgps.DENSE_MASS) / (1 - dgps.DENSE_TOP), 2),
    mthBiasDenseCentre=art.num(dgps.DENSE_TOP / 2, 1),
    mthBiasSparseCentre=art.num((1 + dgps.DENSE_TOP) / 2, 1),
    mthBiasSparseWidth=art.num(1 - dgps.DENSE_TOP, 1),
    mthBiasWideScaleMin=art.int(np.log2(min(variants.WIDE_SCALES))),
    mthBiasDeepDepths=art.numlist([d for d in variants.DEEP_DEPTHS if d not in variants.gbt.DEPTHS]),
)
art.emit_numbers(snakemake.output.numbers, **numbers)
