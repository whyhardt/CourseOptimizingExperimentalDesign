"""Analysis helpers for the Stroop tutorials, so that the notebooks can focus on design questions."""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def clean_trials(data: pd.DataFrame, min_rt: float = 200) -> pd.DataFrame:
    """Add the previous trial's congruency / ink / word, then keep correct, timely trials (RT >= min_rt)."""
    data = data.sort_values(["participant", "block", "trial"]).copy()
    grouped = data.groupby(["participant", "block"])
    for column in ["congruency", "ink", "word"]:
        data[f"prev_{column}"] = grouped[column].shift(1)
    keep = (data["timed_out"] == 0) & (data["correct"] == 1) & (data["rt_ms"] >= min_rt)
    return data[keep].reset_index(drop=True)


def stroop_effect(rt, congruent, min_trials: int = 2):
    """Mean RT(incongruent) - mean RT(congruent) and its standard error (NaN if a condition has too few trials)."""
    rt = np.asarray(rt, dtype=float)
    congruent = np.asarray(congruent, dtype=bool)
    inc, con = rt[~congruent], rt[congruent]
    if len(inc) < min_trials or len(con) < min_trials:
        return np.nan, np.nan
    effect = inc.mean() - con.mean()
    se = np.sqrt(inc.var(ddof=1) / len(inc) + con.var(ddof=1) / len(con))
    return effect, se


def participant_effects(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: Stroop effect, SE, 95% CI and number of trials."""
    rows = []
    for participant, d in clean.groupby("participant"):
        effect, se = stroop_effect(d["rt_ms"], d["congruency"] == "congruent")
        rows.append({"participant": participant, "effect": effect, "se": se, "n_trials": len(d)})
    effects = pd.DataFrame(rows)
    effects["ci_low"] = effects["effect"] - 1.96 * effects["se"]
    effects["ci_high"] = effects["effect"] + 1.96 * effects["se"]
    return effects


def trials_until_significant(d: pd.DataFrame, alpha: float = 0.05, n_min: int = 6) -> float:
    """First n at which the Stroop effect of the first n trials is significant (one-sided Welch t-test); NaN if never."""
    d = d.sort_values("trial")
    rt = d["rt_ms"].to_numpy()
    congruent = (d["congruency"] == "congruent").to_numpy()
    for n in range(n_min, len(d) + 1):
        c = congruent[:n]
        if c.sum() >= 2 and (~c).sum() >= 2:
            if stats.ttest_ind(rt[:n][~c], rt[:n][c], equal_var=False, alternative="greater").pvalue < alpha:
                return n
    return np.nan


def ci_halfwidth(rt, congruent, n: int, n_boot: int = 1000, rng: Optional[np.random.Generator] = None) -> float:
    """95% CI half-width of the Stroop effect in an experiment with n trials, by resampling (bootstrap)."""
    rng = rng or np.random.default_rng()
    rt = np.asarray(rt, dtype=float)
    congruent = np.asarray(congruent, dtype=bool)
    estimates = []
    while len(estimates) < n_boot:
        idx = rng.integers(0, len(rt), n)
        c = congruent[idx]
        if c.sum() < 2 or (~c).sum() < 2:
            continue
        estimates.append(rt[idx][~c].mean() - rt[idx][c].mean())
    return 1.96 * np.std(estimates)


def probe_sequence_effects(clean: pd.DataFrame) -> pd.DataFrame:
    """Per participant: Stroop effect at the probe targets after CCC and after III, and their difference."""
    targets = clean[clean["probe_position"] == 3].copy()
    targets["history"] = targets["probe"].str[:3]
    cells = targets.groupby(["participant", "history", "congruency"])["rt_ms"].mean().unstack("congruency")
    effects = (cells["incongruent"] - cells["congruent"]).unstack("history")
    effects["sequence effect"] = effects["CCC"] - effects["III"]
    return effects


def random_sequence_effects(clean: pd.DataFrame, exclude_repetitions: bool = False) -> pd.DataFrame:
    """Per participant, random trials only: Stroop effect after a congruent / incongruent previous trial.

    With exclude_repetitions, trials that repeat the previous ink colour or word are dropped
    (feature repetitions alone can produce sequence effects, Mayr et al. 2003).
    """
    d = clean[(clean["probe_position"] == -1) & clean["prev_congruency"].notna()]
    if exclude_repetitions:
        d = d[(d["ink"] != d["prev_ink"]) & (d["word"] != d["prev_word"])]
    rows = []
    for participant, dp in d.groupby("participant"):
        row = {"participant": participant}
        for prev in ["congruent", "incongruent"]:
            dd = dp[dp["prev_congruency"] == prev]
            row[f"after {prev}"] = stroop_effect(dd["rt_ms"], dd["congruency"] == "congruent")[0]
        rows.append(row)
    effects = pd.DataFrame(rows).set_index("participant")
    effects["sequence effect"] = effects["after congruent"] - effects["after incongruent"]
    return effects


def count_by_chance(data: pd.DataFrame, pattern: str) -> pd.Series:
    """Per participant: how often a congruency pattern (e.g. 'CCCI') occurs within the random trials."""
    def count(d):
        d = d.sort_values("trial")
        letters = "".join(np.where(d["congruency"] == "congruent", "C", "I"))
        is_random = (d["probe_position"] == -1).to_numpy()
        return sum(letters[i:i + len(pattern)] == pattern and is_random[i:i + len(pattern)].all()
                   for i in range(len(d) - len(pattern) + 1))
    return data.groupby("participant").apply(count, include_groups=False)


def split_half_reliability(clean: pd.DataFrame):
    """Correlation of odd- vs even-trial Stroop effects across participants, and its Spearman-Brown correction."""
    halves = []
    for _, d in clean.groupby("participant"):
        d = d.sort_values("trial")
        odd, even = d.iloc[1::2], d.iloc[0::2]
        halves.append((stroop_effect(odd["rt_ms"], odd["congruency"] == "congruent")[0],
                       stroop_effect(even["rt_ms"], even["congruency"] == "congruent")[0]))
    halves = np.array(halves)
    halves = halves[~np.isnan(halves).any(axis=1)]
    r_half = np.corrcoef(halves[:, 0], halves[:, 1])[0, 1]
    return r_half, 2 * r_half / (1 + r_half)


def variance_components(clean: pd.DataFrame, effects: pd.DataFrame):
    """SD of the true effects between people and SD of a single trial (within a condition)."""
    var_between = effects["effect"].var(ddof=1) - np.mean(effects["se"] ** 2)
    sd_between = np.sqrt(max(var_between, 0.0))
    sd_trial = clean.groupby(["participant", "congruency"])["rt_ms"].std().mean()
    return sd_between, sd_trial
