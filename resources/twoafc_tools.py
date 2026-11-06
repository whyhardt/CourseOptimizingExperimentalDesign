"""Ready-made 2AFC tools for the tutorials and the project: a simulated group, the random sampler,
the candidate grid, playing a whole block with a sampler, the cross-validation table, and plots.

Everything is group level: the trials of all members are pooled. The simulated group has no
individual differences: one diffusion-model player (resources/ddm.py) answers for every member.

Data format (one row per trial): ratio, scatteredness, choice_blue (1 blue, 0 orange, -1 time-out),
rt_ms (NaN for a time-out), member.
"""

import glob
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resources.ddm import DDMPlayer
from resources.gp import GroupGP

RATIO_RANGE = (0.1, 0.9)            # the task's random sampler (decided 2026-11-06: 0.1 to 0.9 everywhere)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ------------------------------------------------------------------ experiment runner

class SimulatedGroup:
    """A group of `n_members` simulated players without individual differences.

    The curve gets flatter on scattered grids (slope 24 at scatteredness 0, 8 at scatteredness 1),
    threshold 0.5. run(trials) plays the trials in order, the members taking turns.
    """

    def __init__(self, n_members: int = 5, threshold: float = 0.5, slope: float = 24.0,
                 slope_scat: float = -16.0, seed: Optional[int] = None):
        self.n_members = n_members
        self.player = DDMPlayer(threshold=threshold, slope=slope, slope_scat=slope_scat,
                                rng=np.random.default_rng(seed))
        self.played = 0

    def run(self, trials: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for r, s in zip(trials["ratio"], trials["scatteredness"]):
            choice, rt = self.player.trial(float(r), float(s))
            rows.append({"ratio": float(r), "scatteredness": float(s), "choice_blue": choice, "rt_ms": rt,
                         "member": self.played % self.n_members})
            self.played += 1
        return pd.DataFrame(rows)


class SimulatedPlayer(SimulatedGroup):
    """One simulated player (a diffusion model): run(trials) returns ratio, scatteredness, choice_blue, rt_ms."""

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(n_members=1, seed=seed, **kwargs)

    def run(self, trials: pd.DataFrame) -> pd.DataFrame:
        return super().run(trials).drop(columns="member")


def random_trials(n: int, rng=None, ratio_range=RATIO_RANGE) -> pd.DataFrame:
    """The random sampler: ratio uniform in ratio_range, scatteredness uniform in [0, 1]."""
    rng = rng if rng is not None else np.random.default_rng()
    return pd.DataFrame({"ratio": rng.uniform(*ratio_range, n), "scatteredness": rng.uniform(0, 1, n)})


def candidate_grid(ratio_range=RATIO_RANGE, n_ratio: int = 21, n_scat: int = 6) -> pd.DataFrame:
    """All trials the experimentalist may choose from: a grid over ratio x scatteredness."""
    r, s = np.meshgrid(np.linspace(*ratio_range, n_ratio), np.linspace(0, 1, n_scat))
    return pd.DataFrame({"ratio": r.ravel(), "scatteredness": s.ravel()})


def play_block(sampler: Callable, group: SimulatedGroup, n_trials: int = 500, n_start: int = 50,
               cycle: int = 10, rng=None) -> pd.DataFrame:
    """A whole block: n_start random trials, then cycles of `cycle` trials from sampler(data, cycle, rng)."""
    rng = rng if rng is not None else np.random.default_rng()
    data = group.run(random_trials(n_start, rng))
    while len(data) < n_trials:
        n = min(cycle, n_trials - len(data))
        data = pd.concat([data, group.run(sampler(data, n, rng))], ignore_index=True)
    return data


# ------------------------------------------------------------------ evaluation (ready-made, as in the project)

def cross_validate(train: dict, test: dict, part: str = "total") -> pd.DataFrame:
    """The project's table: the GP fitted on each training block, scored on each test block and on all tests together.

    part: "total" (choice + RT), "choice" or "RT".

    Bits per held-out trial over the baselines (coin flip for the choice, 'the RT is always the same'
    for the RT), choice + RT. Rows: training blocks; columns: test blocks.
    """
    tests = dict(test)
    tests["all tests"] = pd.concat(list(test.values()), ignore_index=True)
    table = {}
    for name, block in train.items():
        gp = GroupGP().fit(block)
        table[f"trained on {name}"] = {f"{t} test" if t != "all tests" else t: round(gp.held_out_bits(d)[part], 3)
                                       for t, d in tests.items()}
    return pd.DataFrame(table).T


# ------------------------------------------------------------------ the project: a group's pooled blocks

def to_loop_format(d: pd.DataFrame) -> pd.DataFrame:
    """Task data (data/twoafc/*.csv) in the loop's format: ratio, scatteredness, choice_blue (-1 = time-out), rt_ms, member."""
    return pd.DataFrame({"ratio": d["ratio"], "scatteredness": d["scatteredness"],
                         "choice_blue": d["choice_blue"].fillna(-1).astype(int), "rt_ms": d["rt_ms"],
                         "member": d["participant"]})


def load_group_blocks(ids, blocks=("train", "test", "exp_train", "exp_test"),
                      data_dir: str = os.path.join(ROOT, "data", "twoafc")) -> dict:
    """Each block label -> the pooled trials of all group members (the newest file per member and block).

    Returns {label: DataFrame} and prints which files were found.
    """
    out = {}
    for block in blocks:
        parts = []
        for pid in ids:
            files = sorted(glob.glob(os.path.join(data_dir, f"twoafc_{pid}_{block}_*.csv")))
            if files:
                parts.append(to_loop_format(pd.read_csv(files[-1])))
            else:
                print(f"missing: {pid} {block}")
        if parts:
            out[block] = pd.concat(parts, ignore_index=True)
            print(f"{block}: {len(parts)} members, {len(out[block])} trials")
    return out


# ------------------------------------------------------------------ class data

def load_class_data(block: str, data_dir: str = os.path.join(ROOT, "data", "twoafc")) -> pd.DataFrame:
    """All players' trials of one block ('train' or 'test'), pooled, in the loop's data format.

    With fewer than 5 players' files, a simulated class (25 members, 100 random trials each) stands in.
    """
    files = sorted(glob.glob(os.path.join(data_dir, f"twoafc_*_{block}_*.csv")))
    if len(files) >= 5:
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        out = pd.DataFrame({"ratio": d["ratio"], "scatteredness": d["scatteredness"],
                            "choice_blue": d["choice_blue"].fillna(-1).astype(int), "rt_ms": d["rt_ms"],
                            "member": d["participant"]})
        out.attrs["source"] = f"class data: {len(files)} players"
        return out
    seed = {"train": 101, "test": 202}.get(block, 0)
    out = SimulatedGroup(n_members=25, seed=seed).run(random_trials(2500, np.random.default_rng(seed)))
    out.attrs["source"] = "simulated stand-in (fewer than 5 players' files found)"
    return out


# ------------------------------------------------------------------ plots

def plot_iteration(gp: GroupGP, candidates: pd.DataFrame, gain_choice, gain_rt, next_trials: pd.DataFrame, data=None):
    """Three maps over the candidates: expected gain from the choice, from the RT, and the total; next trials in red."""
    n_r, n_s = candidates["ratio"].nunique(), candidates["scatteredness"].nunique()
    r = candidates["ratio"].to_numpy().reshape(n_s, n_r)
    s = candidates["scatteredness"].to_numpy().reshape(n_s, n_r)
    gains = {"from the choice": np.asarray(gain_choice), "from the RT": np.asarray(gain_rt)}
    gains["choice + RT"] = gains["from the choice"] + gains["from the RT"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (title, g) in zip(axes, gains.items()):
        im = ax.pcolormesh(r, s, g.reshape(n_s, n_r), cmap="Blues", shading="auto")
        if data is not None:
            ax.scatter(data["ratio"], data["scatteredness"], s=4, color="grey", alpha=0.4)
        ax.scatter(next_trials["ratio"], next_trials["scatteredness"], s=40, color="#ae0d36", edgecolor="white", zorder=3)
        ax.set_title(f"{title}: up to {g.max():.3f} bits", fontsize=10)
        ax.set_xlabel("ratio")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("scatteredness")
    fig.tight_layout()
    return fig


def plot_curves(gp: GroupGP, data: Optional[pd.DataFrame] = None):
    """The theorist's predictions: P(blue) and the typical RT along ratio, at scatteredness 0 and 1."""
    grid = np.linspace(*RATIO_RANGE, 100)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for sc, ls in [(0.0, "-"), (1.0, "--")]:
        t = pd.DataFrame({"ratio": grid, "scatteredness": sc})
        p, draws = gp.p_blue(t)
        lo, hi = np.percentile(draws, [5, 95], axis=0)
        axes[0].fill_between(grid, lo, hi, color="#2f7bbf", alpha=0.15)
        axes[0].plot(grid, p, color="#2f7bbf", ls=ls, label=f"scatteredness {sc:g}")
        m, var_f, _ = gp.rt_prediction(t)
        axes[1].fill_between(grid, np.exp(m - 2 * np.sqrt(var_f)), np.exp(m + 2 * np.sqrt(var_f)), color="#ae0d36", alpha=0.15)
        axes[1].plot(grid, np.exp(m), color="#ae0d36", ls=ls, label=f"scatteredness {sc:g}")
    axes[0].set_ylabel("P(answer = blue)")
    axes[1].set_ylabel("typical RT (ms)")
    for ax in axes:
        ax.set_xlabel("ratio")
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig
