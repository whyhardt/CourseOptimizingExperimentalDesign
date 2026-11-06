"""Session 6: model predictions, uncertainty and disagreement on the 2AFC task (group level).

The intuitive entry model is the psychometric curve of Session 5, written as a logistic regression:
    P(blue) = 1 / (1 + exp(−z)),  z = X · b
with the columns of X chosen by `features` ("ratio", "ratio+scat" or "ratio*scat"). It is fitted to
the POOLED answers of a group (maximum a posteriori with a weak normal prior, which keeps it stable
with few trials).

Plausible curves (bootstrap): resample the trials with replacement, refit, repeat (default 50 times).
Each refit is one curve the data cannot rule out. Only curves that rise with ratio are kept.

The answer's entropy splits as in Session 4 (X = the curve, Y = the next answer):
    H(Y)          = h(mean over curves of p)        how unpredictable the answer is
    H(Y | X)      = mean over curves of h(p)        noise: left even if we knew the curve
    I(X; Y)       = H(Y) − H(Y | X)                 expected information gain of the trial
with h(p) = −p·log2 p − (1 − p)·log2(1 − p). Uncertainty sampling runs the trials with the largest
I(X; Y); with bootstrapped curves that is the disagreement of a bootstrapped ensemble. Disagreement
between different models uses the same formula with the models in place of the curves.

Simulation (group level, no individual differences): one diffusion-model player (resources/ddm.py)
stands for every group member; a block is 5 members x 100 trials = 500 pooled trials, proposed in
cycles (default 10 trials per cycle).
"""

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from resources.ddm import DDMPlayer

FEATURES = {
    "ratio": lambda r, s: np.c_[np.ones_like(r), r - 0.5],
    "ratio+scat": lambda r, s: np.c_[np.ones_like(r), r - 0.5, s],
    "ratio*scat": lambda r, s: np.c_[np.ones_like(r), r - 0.5, s, (r - 0.5) * s],
}
PRIOR_SD = np.array([2.0, 50.0, 2.0, 50.0])      # weak prior: intercept, ratio, scat, ratio x scat
N_BOOT = 50


def h2(p):
    """Entropy of a yes/no answer with probability p, in bits."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def _prec(d):
    return 1 / PRIOR_SD[:d] ** 2


def fit(ratio, scat, choice_blue, features: str = "ratio") -> dict:
    """Fit P(blue) to the (pooled) answers; keeps the data for the bootstrap.

    Returns b, the covariance from the Fisher information (+ prior) and the data.
    """
    ratio, scat, y = (np.asarray(v, float) for v in (ratio, scat, choice_blue))
    X = FEATURES[features](ratio, scat)
    prec = _prec(X.shape[1])

    def loss(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(prec * b ** 2)

    def grad(b):
        return X.T @ (sigmoid(X @ b) - y) + prec * b

    b = minimize(loss, np.zeros(X.shape[1]), jac=grad, method="BFGS").x
    p = sigmoid(X @ b)
    info = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(prec)
    return {"b": b, "cov": np.linalg.inv(info), "features": features, "X": X, "y": y}


def bootstrap(model: dict, n: int = N_BOOT, rng=None) -> np.ndarray:
    """Parameters of n bootstrap refits (resample the trials, refit), only curves rising with ratio.

    All refits run at once: Newton steps from the full-data fit with bootstrap counts as weights.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    X, y = model["X"], model["y"]
    m, d = X.shape
    w = rng.multinomial(m, np.full(m, 1 / m), size=2 * n).astype(float)     # (B, m) resampling counts
    b = np.tile(model["b"], (2 * n, 1))
    prec = _prec(d)
    for _ in range(12):
        p = sigmoid(b @ X.T)                                                 # (B, m)
        g = (w * (p - y)) @ X + prec * b                                      # (B, d)
        H = np.einsum("bm,mi,mj->bij", w * p * (1 - p), X, X) + np.diag(prec)
        b = b - np.linalg.solve(H, g[..., None])[..., 0]
    return rising(b, model["features"])[:n]


def rising(draws: np.ndarray, features: str) -> np.ndarray:
    """Keep only curves that rise with ratio (more blue tiles -> more blue answers), at every scatteredness."""
    ok = draws[:, 1] > 0
    if features == "ratio*scat":
        ok &= draws[:, 1] + draws[:, 3] > 0
    return draws[ok]


def predict(model: dict, ratio, scat=None) -> np.ndarray:
    ratio = np.asarray(ratio, float)
    scat = np.zeros_like(ratio) if scat is None else np.asarray(scat, float)
    return sigmoid(FEATURES[model["features"]](ratio, scat) @ model["b"])


def plausible_curves(model: dict, ratio, scat=None, n: int = N_BOOT, rng=None) -> np.ndarray:
    """P(blue) of the bootstrapped curves at each candidate trial: shape (curves, trials)."""
    ratio = np.asarray(ratio, float)
    scat = np.zeros_like(ratio) if scat is None else np.asarray(scat, float)
    return sigmoid(bootstrap(model, n, rng) @ FEATURES[model["features"]](ratio, scat).T)


def split_entropy(p_curves: np.ndarray) -> dict:
    """H(Y), H(Y | X) and I(X; Y) at each candidate trial from P(blue) of several curves or models."""
    total = h2(p_curves.mean(axis=0))
    noise = h2(p_curves).mean(axis=0)
    return {"H(Y)": total, "H(Y|X)": noise, "I(X;Y)": total - noise}


def threshold(model_or_b) -> float:
    """Ratio where P(blue) = 0.5 (at scatteredness 0)."""
    b = model_or_b["b"] if isinstance(model_or_b, dict) else model_or_b
    return 0.5 - b[0] / b[1]


def threshold_variance(model: dict, extra=None) -> float:
    """Variance of the threshold from the Fisher information (delta method); `extra` adds information."""
    b = model["b"]
    info = np.linalg.inv(model["cov"]) + (0 if extra is None else extra)
    g = np.zeros(len(b))
    g[0], g[1] = -1 / b[1], b[0] / b[1] ** 2
    return float(g @ np.linalg.solve(info, g))


# ------------------------------------------------------------------ simulated group and samplers

def simulated_player(threshold_: float = 0.5, slope: float = 15.0, slope_scat: float = 0.0, rng=None) -> DDMPlayer:
    """A diffusion-model player (resources/ddm.py) standing for every member of a group.

    Its choices follow the logistic curve with this threshold and slope (slope_scat < 0 weakens the
    drift on scattered grids: slope = slope + slope_scat · scatteredness); RT in player.last_rt_ms.
    """
    return DDMPlayer(threshold=threshold_, slope=slope, slope_scat=slope_scat, rng=rng)


def candidates(ratio_range=(0.1, 0.9), features: str = "ratio") -> pd.DataFrame:
    if features == "ratio":
        return pd.DataFrame({"ratio": np.linspace(*ratio_range, 81), "scatteredness": 0.0})
    return pd.DataFrame([(a, s) for a in np.linspace(*ratio_range, 21) for s in np.linspace(0, 1, 6)],
                        columns=["ratio", "scatteredness"])


def _top(score, cand, n, rng):
    best = np.argsort(-(score + 1e-9 * rng.random(len(score))))[:n]
    return [(cand["ratio"].iat[i], cand["scatteredness"].iat[i]) for i in best]


def propose(sampler: str, data: pd.DataFrame, n: int, features: str, cand: pd.DataFrame, rng,
            ratio_range=(0.1, 0.9), committee=("ratio", "ratio+scat", "ratio*scat")) -> list:
    """The next n trials (ratio, scatteredness) from one of the samplers.

    random       uniform draws
    uncertainty  the n candidates with the largest I(X;Y) of the bootstrapped curves
    committee    the n candidates where the models in `committee` (feature sets) disagree most
    targeted     greedy: each trial shrinks the threshold variance most (Fisher information), ratio only
    """
    if sampler == "random":
        r = rng.uniform(*ratio_range, n)
        s = np.zeros(n) if features == "ratio" else rng.uniform(0, 1, n)
        return list(zip(r, s))
    d = data[data["choice_blue"] >= 0]
    r, s, y = d["ratio"].to_numpy(), d["scatteredness"].to_numpy(), d["choice_blue"].to_numpy()
    cr, cs = cand["ratio"].to_numpy(), cand["scatteredness"].to_numpy()
    if sampler == "uncertainty":
        return _top(split_entropy(plausible_curves(fit(r, s, y, features), cr, cs, rng=rng))["I(X;Y)"], cand, n, rng)
    if sampler == "committee":
        p = np.array([predict(fit(r, s, y, k), cr, cs) for k in committee])
        return _top(split_entropy(p)["I(X;Y)"], cand, n, rng)
    if sampler == "targeted":
        m = fit(r, s, y, "ratio")
        X = FEATURES["ratio"](cr, cs)
        w = sigmoid(X @ m["b"])
        w = w * (1 - w)
        added, out = np.zeros((2, 2)), []
        for _ in range(n):
            var = [threshold_variance(m, added + w[i] * np.outer(X[i], X[i])) for i in range(len(cr))]
            i = int(np.argmin(var))
            added = added + w[i] * np.outer(X[i], X[i])
            out.append((cr[i], cs[i]))
        return out
    raise ValueError(sampler)


def run_group(player: Callable, sampler="random", n_trials: int = 500, cycle: int = 10, n_start: int = 50,
              features: str = "ratio", rng=None, ratio_range=(0.1, 0.9), track: bool = True,
              proposer: Callable = None) -> pd.DataFrame:
    """One group block: n_start random trials, then cycles of `cycle` trials proposed by the sampler.

    `proposer(data, n, rng)` replaces the built-in samplers (e.g. the Gaussian-process theorist).
    Time-outs (choice_blue = -1) stay in the data but are left out of the fits. With `track`, the
    threshold and slope of a ratio-only fit are recorded after every cycle (attrs["estimates"]).
    """
    rng = rng if rng is not None else np.random.default_rng()
    cand = candidates(ratio_range, features)
    data = pd.DataFrame(columns=["ratio", "scatteredness", "choice_blue", "rt_ms"])
    estimates = []
    while len(data) < n_trials:
        first = len(data) < n_start
        n = min(n_start if first else cycle, n_trials - len(data))
        if first or proposer is None:
            trials = propose("random" if first else sampler, data, n, features, cand, rng, ratio_range)
        else:
            trials = proposer(data, n, rng)
        new = []
        for r, s in trials:
            c = int(player(r, s))
            new.append({"ratio": r, "scatteredness": s, "choice_blue": c, "rt_ms": getattr(player, "last_rt_ms", np.nan)})
        data = pd.concat([data, pd.DataFrame(new)], ignore_index=True) if len(data) else pd.DataFrame(new)
        if track:
            d = data[data["choice_blue"] >= 0]
            m = fit(d["ratio"], d["scatteredness"], d["choice_blue"], "ratio")
            estimates.append((len(data), threshold(m), m["b"][1]))
    if track:
        data.attrs["estimates"] = pd.DataFrame(estimates, columns=["trials", "threshold", "slope"])
    return data


def held_out_bits(model: dict, test: pd.DataFrame) -> float:
    """Bits per held-out answer gained over a coin flip: mean of log2 p(answer) − log2 0.5."""
    test = test[test["choice_blue"] >= 0]                     # time-outs give no answer
    p = predict(model, test["ratio"], test["scatteredness"])
    p_answer = np.where(test["choice_blue"] == 1, p, 1 - p)
    return float(np.mean(np.log2(np.clip(p_answer, 1e-12, 1)) + 1))
