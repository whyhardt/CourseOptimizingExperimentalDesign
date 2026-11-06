"""The course theorist for 2AFC: a Gaussian process (GP) group model of choice AND response time.

Fitted to the POOLED trials of a group over the two factors (ratio, scatteredness):
  choice    GP classification (Laplace approximation, logistic link): a latent function f(ratio, scat)
            with P(blue) = sigmoid(f); its posterior gives a mean and a variance of f at every trial.
  RT        GP regression on log RT (standardised): a smooth function plus trial noise.
Kernel: constant x RBF with one length scale per factor; the RT model adds a white-noise term. The
kernel settings are FITTED (maximum marginal likelihood within fixed bounds) every time the model is
fitted, by the same procedure for every data set.

Expected information gain of a trial (bits), as in Sessions 4 and 6:
  choice    I = h(mean P(blue)) − mean h(P(blue)) over draws of the latent function
  RT        I = 1/2 · log2(1 + var_f / var_noise)   (Gaussian: how much the RT would shrink the
            uncertainty about the RT function at that trial)
  total     choice + RT
Held-out score (bits per trial, the project's evaluation):
  choice    log2 p(answer) − log2 0.5            (gain over a coin flip)
  RT        log2 N(log RT | prediction) − log2 N(log RT | training mean and SD)   (gain over "RT is always the same")

The same class is the theorist inside the loop (predictions and uncertainty for the experimentalist)
and the evaluation model after data collection (fit on each training block, score each test block).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from resources.uncertainty import candidates as _candidates, h2, sigmoid, split_entropy

LENGTH_BOUNDS = (0.1, 2.0)


def _X(d):
    return np.c_[np.asarray(d["ratio"], float), np.asarray(d["scatteredness"], float)]


def _kernel(noise: bool):
    k = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=[0.2, 0.5], length_scale_bounds=LENGTH_BOUNDS)
    return k + WhiteKernel(0.5, (1e-3, 10.0)) if noise else k


class GroupGP:
    """Fit with .fit(data); data needs ratio, scatteredness, choice_blue (−1 = time-out) and rt_ms."""

    def fit(self, data: pd.DataFrame, rt: bool = True):
        d = data[data["choice_blue"] >= 0]
        X = _X(d)
        # one thread per fit: for 500 trials this is ~2.5x faster than spreading over all cores
        with warnings.catch_warnings(), threadpool_limits(1):
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.choice = GaussianProcessClassifier(_kernel(False), n_restarts_optimizer=0).fit(X, d["choice_blue"].astype(int))
            self.rt = None
            if rt and d["rt_ms"].notna().sum() > 5:
                y = np.log(d["rt_ms"].to_numpy(float))
                self.rt_mean, self.rt_sd = y.mean(), y.std()
                self.rt = GaussianProcessRegressor(_kernel(True), n_restarts_optimizer=0).fit(X, (y - self.rt_mean) / self.rt_sd)
                self.rt_noise = float(self.rt.kernel_.k2.noise_level)
        return self

    # -------------------------------------------------------------- predictions
    def latent(self, trials: pd.DataFrame):
        """Posterior mean and variance of the latent choice function f (sklearn's Laplace quantities)."""
        g = self.choice.base_estimator_
        X = _X(trials)
        K_star = g.kernel_(g.X_train_, X)
        mean = K_star.T @ (g.y_train_ - g.pi_)
        v = solve_triangular(g.L_, g.W_sr_[:, None] * K_star, lower=True)
        var = g.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
        return mean, np.maximum(var, 1e-12)

    def p_blue(self, trials: pd.DataFrame, n_draws: int = 200, rng=None):
        """P(blue) for each trial, averaged over draws of the latent function; also the draws."""
        rng = rng if rng is not None else np.random.default_rng(0)
        mean, var = self.latent(trials)
        draws = sigmoid(mean + np.sqrt(var) * rng.standard_normal((n_draws, len(mean))))
        return draws.mean(axis=0), draws

    def rt_prediction(self, trials: pd.DataFrame):
        """Predicted log RT (ms) mean, variance of the RT function, and the trial-noise variance (log units)."""
        mean, sd = self.rt.predict(_X(trials), return_std=True)
        var_f = np.maximum(sd ** 2 - self.rt_noise, 1e-12)
        s2 = self.rt_sd ** 2
        return self.rt_mean + self.rt_sd * mean, var_f * s2, self.rt_noise * s2

    def information(self, trials: pd.DataFrame, rng=None) -> pd.DataFrame:
        """Expected information gain (bits) of each candidate trial: choice, RT and total."""
        _, draws = self.p_blue(trials, rng=rng)
        out = pd.DataFrame({"choice": split_entropy(draws)["I(X;Y)"]}, index=trials.index)
        if self.rt is not None:
            _, var_f, var_n = self.rt_prediction(trials)
            out["RT"] = 0.5 * np.log2(1 + var_f / var_n)
        else:
            out["RT"] = 0.0
        out["total"] = out["choice"] + out["RT"]
        return out

    def held_out_bits(self, test: pd.DataFrame) -> dict:
        """Bits per held-out trial over the baselines (coin flip; constant RT), for choice, RT and both."""
        t = test[test["choice_blue"] >= 0]
        p, _ = self.p_blue(t)
        p_answer = np.where(t["choice_blue"] == 1, p, 1 - p)
        choice = float(np.mean(np.log2(np.clip(p_answer, 1e-12, 1)) + 1))
        y = np.log(t["rt_ms"].to_numpy(float))
        mean, var_f, var_n = self.rt_prediction(t)

        def log2_normal(x, m, v):
            return (-0.5 * np.log(2 * np.pi * v) - (x - m) ** 2 / (2 * v)) / np.log(2)

        rt = float(np.mean(log2_normal(y, mean, var_f + var_n) - log2_normal(y, self.rt_mean, self.rt_sd ** 2)))
        return {"choice": choice, "RT": rt, "total": choice + rt}

    def threshold(self, scatteredness: float = 0.5, grid=np.linspace(0.1, 0.9, 801)) -> float:
        """Ratio where the predicted P(blue) crosses 0.5 at a given scatteredness (the group threshold)."""
        mean, _ = self.latent(pd.DataFrame({"ratio": grid, "scatteredness": scatteredness}))
        i = np.flatnonzero(np.diff(np.sign(mean)))
        if not len(i):
            return float("nan")
        i = i[0]
        return float(grid[i] - mean[i] * (grid[i + 1] - grid[i]) / (mean[i + 1] - mean[i]))



GP = GroupGP          # plain name for the tutorials: the GP on choice and RT

def gp_proposer(ratio_range=(0.1, 0.9), use_rt: bool = True):
    """Experimentalist for run_group: fit the GP to all trials so far, run the n candidates with the largest gain."""
    cand = _candidates(ratio_range, "ratio*scat")

    def propose(data, n, rng):
        info = GroupGP().fit(data, rt=use_rt).information(cand, rng=rng)
        score = (info["total"] if use_rt else info["choice"]).to_numpy()
        best = np.argsort(-(score + 1e-9 * rng.random(len(score))))[:n]
        return [(cand["ratio"].iat[i], cand["scatteredness"].iat[i]) for i in best]

    return propose
