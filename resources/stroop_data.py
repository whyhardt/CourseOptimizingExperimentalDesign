"""Load the class Stroop data, or generate an example data set in the same format.

The example data is only a fallback for when no class data is available yet. It uses the
same block design as the experiment (random trials with hidden probe sequences) and a simple
generative model: ex-Gaussian RTs, an individual Stroop effect, a congruency sequence effect
(smaller Stroop effect after incongruent trials), a benefit for repeated ink colours and an extra
benefit for complete repetitions (same word and ink; Mayr et al. 2003), practice and occasional errors.
"""

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "stroop")


def load_stroop_data(data_dir: str = DATA_DIR, block: Optional[str] = None) -> pd.DataFrame:
    """Concatenate all CSV files of the Stroop experiment (one file per participant and block)."""
    files = sorted(glob.glob(os.path.join(data_dir, "stroop_*.csv")))
    if not files:
        return pd.DataFrame()
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    data["participant"] = data["participant"].astype(str)
    data["block"] = data["block"].astype(str)
    data["probe"] = data["probe"].fillna("")
    if block is not None:
        data = data[data["block"] == str(block)].reset_index(drop=True)
    return data


def simulate_participant(trials: pd.DataFrame, participant: str = "SIM1", rng: Optional[np.random.Generator] = None,
                         design: str = "simulated", return_truth: bool = False):
    """Let a synthetic participant play a trial sequence (columns word, ink[, probe, probe_position]).

    Returns data in the format of experiments/stroop.py. The participant's parameters are drawn
    from the class distribution (see module docstring). With return_truth, also returns the true
    Stroop effect (averaged over a congruent and an incongruent previous trial) and the true CSE.
    """
    from experiments.stroop import KEYS

    rng = rng or np.random.default_rng()
    colors = list(KEYS)
    base = rng.normal(640, 80)                 # mean RT on congruent trials
    stroop = rng.normal(80, 25)                # individual Stroop effect
    sequence_effect = rng.normal(30, 10)       # reduction of the Stroop effect after an incongruent trial
    rows, previous = [], None
    for i, t in enumerate(trials.itertuples(index=False)):
        incongruent = t.word != t.ink
        effect = stroop
        if previous is not None and previous.word != previous.ink:
            effect -= sequence_effect
        mu = base + incongruent * effect - 0.4 * i
        if previous is not None and previous.ink == t.ink:
            mu -= 25                           # repeating the ink colour (and response) is faster
            if previous.word == t.word:
                mu -= 40                       # a complete repetition (same word too) is even faster
        rt = rng.normal(mu - 110, 55) + rng.exponential(110)
        p_error = 0.02 + 0.05 * incongruent
        response = t.ink if rng.random() > p_error else rng.choice([c for c in colors if c != t.ink])
        timed_out = rt > 2000
        rows.append({
            "participant": participant, "block": "1", "trial": i, "task": "color",
            "word": t.word, "ink": t.ink, "congruency": "incongruent" if incongruent else "congruent",
            "correct_response": t.ink, "response": None if timed_out else response,
            "correct": int(response == t.ink and not timed_out),
            "rt_ms": None if timed_out else round(max(rt, 150), 1), "timed_out": int(timed_out),
            "probe": getattr(t, "probe", ""), "probe_position": getattr(t, "probe_position", -1), "design": design,
        })
        previous = t
    data = pd.DataFrame(rows)
    if return_truth:
        return data, {"stroop": stroop - sequence_effect / 2, "cse": sequence_effect}
    return data


def example_stroop_data(n_participants: int = 24, n_trials: int = 96, seed: int = 0) -> pd.DataFrame:
    """Simulated class data in the format of experiments/stroop.py (see module docstring)."""
    from experiments.stroop import random_trials_with_probes

    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_participants):
        participant = "".join(rng.choice(list("ABCDEFGHJKMNPQRSTUVWXYZ23456789"), 4))
        trials = random_trials_with_probes(n_trials, seed=int(rng.integers(1e9)))
        data.append(simulate_participant(trials, participant, rng, design="example data (simulated)"))
    return pd.concat(data, ignore_index=True)
