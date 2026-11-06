"""Session 4: information measures on the Stroop designs and data.

All measures are in bits (log base 2):
  entropy               H(X)   = −Σ p(x) · log2 p(x)
  joint entropy         H(X,Y) = −Σ p(x,y) · log2 p(x,y)
  conditional entropy   H(Y|X) = H(X,Y) − H(X)
  mutual information    I(X;Y) = H(Y) − H(Y|X) = H(X) + H(Y) − H(X,Y)
X and Y are independent exactly when I(X;Y) = 0.

Models as measuring instruments: a linear regression with the structure of the simulated Stroop
player is fitted on one block type and scored on held-out blocks as the information gain per trial
over a baseline that only knows each player's mean response time.
"""

import numpy as np
import pandas as pd

from resources.stroop_analysis import clean_trials

# English letter frequencies in % (Lewand, 2000)
LETTER_FREQUENCIES = {
    "e": 12.70, "t": 9.06, "a": 8.17, "o": 7.51, "i": 6.97, "n": 6.75, "s": 6.33, "h": 6.09, "r": 5.99,
    "d": 4.25, "l": 4.03, "c": 2.78, "u": 2.76, "m": 2.41, "w": 2.36, "f": 2.23, "g": 2.02, "y": 1.97,
    "p": 1.93, "b": 1.29, "v": 0.98, "k": 0.77, "j": 0.15, "x": 0.15, "q": 0.10, "z": 0.07,
}


def entropy(counts) -> float:
    """H = −Σ p · log2 p of a list / array / Series of counts (or probabilities)."""
    p = np.asarray(counts, dtype=float).ravel()
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_information(x, y) -> float:
    """I(X;Y) = H(X) + H(Y) − H(X,Y), estimated from paired observations."""
    table = pd.crosstab(pd.Series(np.asarray(x)), pd.Series(np.asarray(y)))
    return entropy(table.sum(axis=1)) + entropy(table.sum(axis=0)) - entropy(table.to_numpy())


def conditional_entropy(y, x) -> float:
    """H(Y|X) = H(X,Y) − H(X)."""
    table = pd.crosstab(pd.Series(np.asarray(x)), pd.Series(np.asarray(y)))
    return entropy(table.to_numpy()) - entropy(table.sum(axis=1))


def oddball_information(k: int, n_balls: int = 12) -> float:
    """Expected information gain of weighing k against k balls (one ball is heavier or lighter).

    The outcome is left heavy / right heavy / balance with probabilities k/n, k/n and (n − 2k)/n,
    and the information gain is the entropy of that outcome.
    """
    return entropy([k, k, n_balls - 2 * k])


# ------------------------------------------------------------------ designs

def letters(trials: pd.DataFrame) -> np.ndarray:
    return np.where(trials["word"].to_numpy() == trials["ink"].to_numpy(), "C", "I")


def design_factors(trials: pd.DataFrame) -> pd.DataFrame:
    """Factors of every trial (from the second one on): congruency, previous congruency, block half, ink repetition."""
    c = letters(trials)
    ink = trials["ink"].to_numpy()
    half = np.where(np.arange(len(c)) < len(c) // 2, "first", "second")
    return pd.DataFrame({"congruency": c[1:], "previous": c[:-1], "half": half[1:],
                         "ink repetition": np.where(ink[1:] == ink[:-1], "repeat", "switch")})


def design_independence(trials: pd.DataFrame) -> pd.Series:
    """Mutual information (bits) between congruency and the other factors of a design: 0 = independent."""
    f = design_factors(trials)
    return pd.Series({f"I(congruency; {other})": round(mutual_information(f["congruency"], f[other]), 3)
                      for other in ["previous", "half", "ink repetition"]})


def predictability(trials, max_history: int = 4) -> pd.Series:
    """H(next C/I | last k trials) for k = 0 .. max_history: how uncertain the next trial is for a player who knows the last k.

    `trials` is one block or a list of blocks of the same design (pooled - one block of 96 trials
    underestimates the entropy for long histories, because each history occurs only a few times).
    """
    blocks = trials if isinstance(trials, list) else [trials]
    history = {k: [] for k in range(max_history + 1)}
    nxt = []
    for block in blocks:
        c = "".join(letters(block))
        for i in range(max_history, len(c)):
            nxt.append(c[i])
            for k in range(max_history + 1):
                history[k].append(c[i - k:i])
    out = {0: round(entropy(pd.Series(nxt).value_counts()), 3)}
    for k in range(1, max_history + 1):
        out[k] = round(conditional_entropy(nxt, history[k]), 3)
    return pd.Series(out, name="H(next | last k) in bits")


# ------------------------------------------------------------------ responses

def rt_information(data: pd.DataFrame, n_bins: int = 4) -> pd.Series:
    """How many bits the factors tell about the response time (RT split into n_bins quantiles per player)."""
    clean = clean_trials(data).dropna(subset=["prev_congruency"]).copy()
    clean["rt bin"] = clean.groupby("participant")["rt_ms"].transform(
        lambda rt: pd.qcut(rt, n_bins, labels=False, duplicates="drop"))
    out = {"H(RT bin)": entropy(clean["rt bin"].value_counts()),
           "I(congruency; RT bin)": mutual_information(clean["congruency"], clean["rt bin"])}
    # sequence effect: what the previous trial adds once the current one is known
    parts = [mutual_information(d["prev_congruency"], d["rt bin"]) * len(d) / len(clean)
             for _, d in clean.groupby("congruency")]
    out["I(previous; RT bin | congruency)"] = sum(parts)
    return pd.Series(out).round(4)


def rt_bin_table(data: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """Share of each response-time bin (quantiles per player) within congruent and incongruent trials, and its entropy.

    H(RT bin | congruency) is the average of the two entropies, weighted by how often each congruency occurs.
    """
    clean = clean_trials(data).dropna(subset=["prev_congruency"]).copy()
    clean["rt bin"] = clean.groupby("participant")["rt_ms"].transform(
        lambda rt: pd.qcut(rt, n_bins, labels=False, duplicates="drop"))
    table = pd.crosstab(clean["congruency"], clean["rt bin"], normalize="index")
    table.columns = ["fastest quarter", "2nd quarter", "3rd quarter", "slowest quarter"][:n_bins]
    table["H (bits)"] = [entropy(row) for row in table.to_numpy()]
    table["share of trials"] = clean["congruency"].value_counts(normalize=True)
    return table.round(3)


# ------------------------------------------------------------------ models as measuring instruments

REGRESSORS = ["incongruent", "after incongruent", "incongruent x after incongruent", "ink repeat",
              "complete repeat", "trial"]


def model_table(block_data: pd.DataFrame) -> pd.DataFrame:
    """Clean trials of one or more blocks with the regressors of the Stroop model."""
    d = block_data.sort_values(["participant", "block", "trial"]).copy()
    g = d.groupby(["participant", "block"])
    d["prev_word"] = g["word"].shift(1)
    d["prev_ink"] = g["ink"].shift(1)
    d["prev_congruency"] = g["congruency"].shift(1)
    keep = (d["timed_out"] == 0) & (d["correct"] == 1) & (d["rt_ms"] >= 200) & d["prev_congruency"].notna()
    d = d[keep].copy()
    d["incongruent"] = (d["congruency"] == "incongruent").astype(float)
    d["after incongruent"] = (d["prev_congruency"] == "incongruent").astype(float)
    d["incongruent x after incongruent"] = d["incongruent"] * d["after incongruent"]
    d["ink repeat"] = (d["ink"] == d["prev_ink"]).astype(float)
    d["complete repeat"] = d["ink repeat"] * (d["word"] == d["prev_word"]).astype(float)
    d["trial"] = d["trial"].astype(float)
    return d


def fit_model(train: pd.DataFrame, exclude_ink_repetitions: bool = False) -> dict:
    """Least squares: RT = player mean + Σ b · regressor. Regressors that never vary in the data get b = 0.

    With exclude_ink_repetitions, trials that repeat the previous ink colour are dropped before fitting.
    """
    d = model_table(train)
    if exclude_ink_repetitions:
        d = d[d["ink repeat"] == 0]
    x = d[REGRESSORS].to_numpy()
    x = x - d.groupby("participant")[REGRESSORS].transform("mean").to_numpy()   # within-player
    y = (d["rt_ms"] - d.groupby("participant")["rt_ms"].transform("mean")).to_numpy()
    usable = x.std(axis=0) > 0
    b = np.zeros(len(REGRESSORS))
    b[usable] = np.linalg.lstsq(x[:, usable], y, rcond=None)[0]
    residual = y - x @ b
    return {"b": pd.Series(b, index=REGRESSORS), "sigma": float(residual.std()),
            "sigma_baseline": float(y.std()), "learned": pd.Series(usable, index=REGRESSORS)}


def information_gain(model: dict, test: pd.DataFrame) -> float:
    """Average bits per held-out trial the model gains over the baseline (each player's mean only).

    gain = mean of log2 N(y | model) − log2 N(y | player mean); the player's mean comes from the
    test block itself, so only the effects are tested, not the player's speed.
    """
    d = model_table(test)
    x = d[REGRESSORS].to_numpy() - d.groupby("participant")[REGRESSORS].transform("mean").to_numpy()
    y = (d["rt_ms"] - d.groupby("participant")["rt_ms"].transform("mean")).to_numpy()

    def log2_density(residual, sigma):
        return (-0.5 * np.log(2 * np.pi * sigma ** 2) - residual ** 2 / (2 * sigma ** 2)) / np.log(2)

    gain = log2_density(y - x @ model["b"].to_numpy(), model["sigma"]) - log2_density(y, model["sigma_baseline"])
    return float(gain.mean())


TRAINING_SOURCES = {"predefined": ("designed", False), "random": ("random", False),
                    "random, repetitions removed": ("random", True)}


def cross_validation(data: pd.DataFrame, test_blocks=("designed", "random"), seed: int = 0) -> pd.DataFrame:
    """Bits per held-out trial gained over the baseline, for models trained on each source and tested on each block type.

    The players are split into two halves: train on one half, test on the other, then swap and
    average - so every test set comes from players the model has never seen.
    """
    players = np.array(sorted(data["participant"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(players)
    halves = [set(players[: len(players) // 2]), set(players[len(players) // 2:])]
    table = {}
    for source, (block, remove) in TRAINING_SOURCES.items():
        gains = {test: [] for test in test_blocks}
        for train_half, test_half in [(halves[0], halves[1]), (halves[1], halves[0])]:
            train = data[data["participant"].isin(train_half) & (data["block"] == block)]
            model = fit_model(train, exclude_ink_repetitions=remove)
            for test in test_blocks:
                held_out = data[data["participant"].isin(test_half) & (data["block"] == test)]
                gains[test].append(information_gain(model, held_out))
        table[source] = {test: round(float(np.mean(g)), 3) for test, g in gains.items()}
    table = pd.DataFrame(table).T.rename(columns={"designed": "predefined"})
    return table.rename_axis(index="trained on", columns="tested on (bits per trial)")


def example_three_blocks(designed_pool: pd.DataFrame, n_players: int = 24, seed: int = 0) -> pd.DataFrame:
    """Simulated stand-in data: every player plays the Session 1 block, a predefined sequence and a random block."""
    from experiments.stroop import random_trials, random_trials_with_probes
    from resources.design_quality import pool_sequences
    from resources.stroop_data import simulate_participant

    rng = np.random.default_rng(seed)
    sequences = pool_sequences(designed_pool)
    data = []
    for i in range(n_players):
        blocks = [("1", random_trials_with_probes(seed=int(rng.integers(1e9)))[["word", "ink"]]),
                  ("designed", sequences[int(rng.integers(len(sequences)))][["word", "ink"]]),
                  ("random", random_trials(seed=int(rng.integers(1e9)))[["word", "ink"]])]
        if i % 2:
            blocks[1], blocks[2] = blocks[2], blocks[1]
        trials = pd.concat([t for _, t in blocks], ignore_index=True)
        d = simulate_participant(trials, f"SIM{i:02d}", rng)
        sizes = [len(t) for _, t in blocks]
        d["block"] = np.repeat([b for b, _ in blocks], sizes)
        d["trial"] = np.concatenate([np.arange(n) for n in sizes])
        data.append(d)
    return pd.concat(data, ignore_index=True)
