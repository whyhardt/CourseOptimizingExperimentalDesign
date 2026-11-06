"""Session 4: information measures on the Stroop designs and data.

All measures are in bits (log base 2):
  entropy               H(X)   = −Σ p(x) · log2 p(x)
  joint entropy         H(X,Y) = −Σ p(x,y) · log2 p(x,y)
  conditional entropy   H(Y|X) = H(X,Y) − H(X)
  mutual information    I(X;Y) = H(Y) − H(Y|X) = H(X) + H(Y) − H(X,Y)
X and Y are independent exactly when I(X;Y) = 0.

The storyline of the session: X = a player's Stroop effect (what we want to know), Y = a response
time (what we observe), and our choice = the condition of the next trial. The design checks
(independence of factors, predictability) apply the same measures to the design itself.

Models as measuring instruments: a linear regression with the structure of the simulated Stroop
player is fitted on one block type and scored on held-out blocks as the information gain per trial
over a baseline that only knows each player's mean response time.
"""

import numpy as np
import pandas as pd


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
    """Factors of every trial (from the third one on): congruency, previous and two-back congruency, ink repetition."""
    c = letters(trials)
    ink = trials["ink"].to_numpy()
    return pd.DataFrame({"congruency": c[2:], "previous": c[1:-1], "two back": c[:-2],
                         "ink repetition": np.where(ink[2:] == ink[1:-1], "repeat", "switch")})


def design_independence(trials) -> pd.Series:
    """Mutual information (bits) between congruency and the other factors of a design: 0 = independent.

    `trials` is one sequence or a list of sequences of the same design (pooled).
    """
    seqs = [trials] if isinstance(trials, pd.DataFrame) else list(trials)
    f = pd.concat([design_factors(t) for t in seqs], ignore_index=True)
    return pd.Series({f"I(congruency; {other})": round(mutual_information(f["congruency"], f[other]), 3)
                      for other in ["previous", "two back", "ink repetition"]})


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


# ------------------------------------------------------------------ responses: bits about the Stroop effect

NORMAL_ENTROPY = 0.5 * np.log2(2 * np.pi * np.e)   # H of a normal distribution = log2(SD) + this, in bits


def bits_gained(sd_before: float, se: float) -> dict:
    """What a measurement with standard error `se` tells about an effect we knew up to `sd_before`.

    Before and after are normal distributions; their precisions (1 / SD²) add:
        SD after = 1 / √(1 / SD²before + 1 / SE²)
        bits     = log2(SD before / SD after)   (halving the SD = 1 bit)
    """
    sd_after = 1 / np.sqrt(1 / sd_before ** 2 + 1 / se ** 2)
    return {"SD before (ms)": sd_before, "SE of the data (ms)": se, "SD after (ms)": sd_after,
            "bits": float(np.log2(sd_before / sd_after))}


def stroop_trial_information(n_trials: int = 400, sigma: float = 200.0, sd_effect: float = 50.0,
                             sd_base: float = 100.0, resolution: float = 10.0) -> pd.DataFrame:
    """Trial by trial: X = a player's Stroop effect, Y = the response time of the next trial.

    Model: RT = base speed + X · incongruent + noise (SD `sigma`). Before the first trial, the base
    speed is known up to `sd_base` and the Stroop effect up to `sd_effect` (normal distributions).
    Every trial is the condition (congruent or incongruent) whose RT tells the most about X.
    Entropies of continuous quantities need a resolution (here `resolution` ms); differences in
    entropy (bits gained) do not depend on it. In this normal model the numbers do not depend on
    the response times actually observed, so they are exact, not averages.

    Returns one row per trial, *before* that trial: H(X), H(Y), I(X;Y), H(Y|X), the chosen condition
    and the trials run so far in each condition.
    """
    cov = np.diag([sd_base ** 2, sd_effect ** 2])          # uncertainty about (base speed, Stroop effect)
    rows, counts = [], {"congruent": 0, "incongruent": 0}
    for trial in range(1, n_trials + 1):
        options = []
        for cell, a in (("congruent", np.array([1.0, 0.0])), ("incongruent", np.array([1.0, 1.0]))):
            var_y = a @ cov @ a + sigma ** 2                 # how uncertain the next RT is
            var_x_after = cov[1, 1] - (cov @ a)[1] ** 2 / var_y
            options.append((0.5 * np.log2(cov[1, 1] / var_x_after), cell, a, var_y))
        info, cell, a, var_y = max(options, key=lambda o: o[0])
        h_x = NORMAL_ENTROPY + np.log2(np.sqrt(cov[1, 1]) / resolution)
        h_y = NORMAL_ENTROPY + np.log2(np.sqrt(var_y) / resolution)
        rows.append({"trial": trial, "H(X)": h_x, "H(Y)": h_y, "I(X;Y)": info, "H(Y|X)": h_y - info,
                     "SD of X (ms)": np.sqrt(cov[1, 1]), "next condition": cell,
                     "congruent so far": counts["congruent"], "incongruent so far": counts["incongruent"]})
        cov = cov - np.outer(cov @ a, cov @ a) / var_y
        counts[cell] += 1
    return pd.DataFrame(rows).set_index("trial")


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
