"""Tools for designing Stroop sequences: SweetPea conversion, export, and pre-testing a design.

The scoreboard of the design competition: at a fixed budget (96 trials), how precisely does a
design measure
  - the Stroop effect: mean RT(incongruent) - mean RT(congruent)
  - the congruency sequence effect (CSE): Stroop effect after a congruent trial minus after an
    incongruent trial
Precision = the within-player SE of the estimate (smaller is better). A design can also be
biased (e.g. a blocked design confounds congruency with practice), which the pre-test on the
synthetic participant shows because there the true values are known.
"""

import os
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from resources.stroop_analysis import clean_trials, stroop_effect

COLORS = ["red", "green", "blue", "yellow"]


def sweetpea_to_trials(block, experiments, index: int = 0) -> pd.DataFrame:
    """One synthesized SweetPea sequence as a data frame with the columns word and ink (+ all other factors)."""
    from sweetpea import experiments_to_dicts

    sequence = experiments_to_dicts(block, experiments)[index]
    if isinstance(sequence, dict):   # older SweetPea versions: {factor: [levels]}
        sequence = pd.DataFrame(sequence)
    else:
        sequence = pd.DataFrame(list(sequence))
    return sequence


def synthesize(block, n_samples: int = 1, seed: Optional[int] = None) -> list:
    """Sample sequences from a SweetPea block (quietly) with the CMSGen sampler; one data frame per sequence.

    SweetPea's default RandomGen gets very slow as soon as a design has transitions; CMSGen samples
    a 96-trial Stroop design in about a second.
    """
    import contextlib
    import io
    import random

    from sweetpea import CMSGen, synthesize_trials

    if seed is not None:
        random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        experiments = synthesize_trials(block, n_samples, sampling_strategy=CMSGen)
    if not experiments:
        raise ValueError("SweetPea found no sequence that satisfies all constraints")
    return [sweetpea_to_trials(block, experiments, i) for i in range(len(experiments))]


def design_summary(trials: pd.DataFrame) -> pd.Series:
    """How balanced is a sequence? Counts of conditions, transitions, colours and repetitions."""
    congruent = (trials["word"] == trials["ink"]).to_numpy()
    letters = np.where(congruent, "C", "I")
    pairs = pd.Series([a + b for a, b in zip(letters[:-1], letters[1:])]).value_counts()
    runs = max(len(r) for r in "".join(letters).replace("CI", "C I").replace("IC", "I C").split())
    summary = {"trials": len(trials), "congruent": int(congruent.sum()), "incongruent": int((~congruent).sum())}
    summary.update({f"{k[0]}→{k[1]}": int(pairs.get(k, 0)) for k in ["CC", "CI", "IC", "II"]})
    summary.update({f"ink {c}": int((trials["ink"] == c).sum()) for c in COLORS})
    summary["ink repetitions"] = int((trials["ink"].values[1:] == trials["ink"].values[:-1]).sum())
    summary["word repetitions"] = int((trials["word"].values[1:] == trials["word"].values[:-1]).sum())
    summary["longest run"] = runs
    return pd.Series(summary)


def count_sequences(trials: pd.DataFrame, length: int = 4) -> pd.Series:
    """How often each congruency sequence of `length` trials (e.g. CCCI = three congruent, then incongruent) occurs."""
    letters = "".join(np.where(trials["word"] == trials["ink"], "C", "I"))
    sequences = [letters[i - length + 1:i + 1] for i in range(length - 1, len(letters))]
    return pd.Series(sequences).value_counts().sort_index().rename("count")


def check_design(trials: pd.DataFrame, n_trials: int = 96) -> pd.DataFrame:
    """Validate a design for the Stroop experiment and return its word/ink columns."""
    missing = {"word", "ink"} - set(trials.columns)
    if missing:
        raise ValueError(f"the design needs the columns {sorted(missing)}")
    trials = trials[["word", "ink"]].astype(str).apply(lambda c: c.str.strip().str.lower())
    unknown = (set(trials["word"]) | set(trials["ink"])) - set(COLORS)
    if unknown:
        raise ValueError(f"unknown colours {unknown}; allowed: {COLORS}")
    if len(trials) != n_trials:
        raise ValueError(f"the design has {len(trials)} trials, the budget is {n_trials}")
    return trials.reset_index(drop=True)


def save_design(trials: pd.DataFrame, path: str, n_trials: int = 96) -> str:
    """Save a design as CSV that the experiment can play: python -m experiments.stroop --design <path>."""
    trials = check_design(trials, n_trials)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    trials.to_csv(path, index=False)
    return os.path.abspath(path)


def estimates(clean: pd.DataFrame) -> dict:
    """Stroop effect and congruency sequence effect (with SEs) of one player's cleaned data."""
    congruent = clean["congruency"] == "congruent"
    stroop, stroop_se = stroop_effect(clean["rt_ms"], congruent)
    parts = []
    for prev in ["congruent", "incongruent"]:
        d = clean[clean["prev_congruency"] == prev]
        parts.append(stroop_effect(d["rt_ms"], d["congruency"] == "congruent"))
    cse = parts[0][0] - parts[1][0]
    cse_se = np.sqrt(parts[0][1] ** 2 + parts[1][1] ** 2)
    return {"stroop": stroop, "stroop_se": stroop_se, "cse": cse, "cse_se": cse_se}


def pretest_design(design: Union[pd.DataFrame, Callable[[], pd.DataFrame]], n_players: int = 200,
                   seed: int = 0) -> pd.Series:
    """Let synthetic players play a design and summarise precision and bias of both estimates.

    `design` is a fixed sequence (every player gets the same one) or a function that returns a new
    sequence for every player (e.g. a random sampler). In the simulation the true Stroop effect of a
    player is known (averaged over both histories), so the bias of a design can be measured.
    """
    from resources.stroop_data import simulate_participant

    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        trials = design() if callable(design) else design
        data, truth = simulate_participant(trials, f"SIM{p}", rng, return_truth=True)
        row = estimates(clean_trials(data))
        row["stroop_error"] = row["stroop"] - truth["stroop"]
        row["cse_error"] = row["cse"] - truth["cse"]
        rows.append(row)
    r = pd.DataFrame(rows)
    return pd.Series({
        "Stroop: SE": np.sqrt(np.nanmean(r["stroop_se"] ** 2)),
        "Stroop: bias": np.nanmean(r["stroop_error"]),
        "CSE: SE": np.sqrt(np.nanmean(r["cse_se"] ** 2)),
        "CSE: bias": np.nanmean(r["cse_error"]),
        "CSE: not estimable": np.mean(np.isnan(r["cse"])),
    }).round(1)
