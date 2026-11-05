"""Session 3: judging a Stroop design before and after data collection.

An effect is a **contrast** between cells: effect = Σ w_c · mean RT of cell c (weights +1 / −1).
Cells are named by their sequence, the last letter is the current trial (CI = incongruent after congruent):
  - Stroop effect:   RT(I) − RT(C)
  - sequence effect: [RT(I after C) − RT(C after C)] − [RT(I after I) − RT(C after I)]
  - history effect (only computed on paper in S3):
                     [RT(I after CCC) − RT(C after CCC)] − [RT(I after III) − RT(C after III)]

Before data collection (design only, no model):
  expected SE = σ · √(Σ_c w_c² / n_c), σ = SD of a single trial, n_c = trials in cell c
  plus coverage (C/I sequences of four trials) and confound checks.
After data collection (class level): everybody played the predefined design (block "designed",
file designs/designed.csv) and a random block (block "random"); class estimate = mean of the
players' estimates, SE = SD(player estimates) / √n.
"""

import numpy as np
import pandas as pd

from resources.stroop_analysis import clean_trials

DESIGNED_BLOCK = "designed"
RANDOM_BLOCK = "random"

EFFECTS = {
    "Stroop effect": {"I": +1, "C": -1},
    "sequence effect": {"CI": +1, "CC": -1, "II": -1, "IC": +1},
    "history effect": {"CCCI": +1, "CCCC": -1, "IIII": -1, "IIIC": +1},
}


def _letters(trials: pd.DataFrame) -> np.ndarray:
    return np.where(trials["word"].to_numpy() == trials["ink"].to_numpy(), "C", "I")


def label_cells(trials: pd.DataFrame, effect: str) -> pd.Series:
    """The cell of every trial for an effect ('' = not used). `trials`: one block in the order it was played."""
    letters = _letters(trials)
    cells = np.full(len(letters), "", dtype=object)
    if effect == "Stroop effect":
        cells[:] = letters
    elif effect == "sequence effect":
        cells[1:] = [f"{a}{b}" for a, b in zip(letters[:-1], letters[1:])]
    elif effect == "history effect":
        for i in range(3, len(letters)):
            history = "".join(letters[i - 3:i])
            if history in ("CCC", "III"):
                cells[i] = f"{history}{letters[i]}"
    else:
        raise ValueError(f"unknown effect {effect!r}; choose from {list(EFFECTS)}")
    return pd.Series(cells, index=trials.index, name="cell")


def cell_counts(trials: pd.DataFrame, effect: str) -> pd.Series:
    """Trials per cell of an effect (empty cells included)."""
    cells = label_cells(trials, effect)
    return pd.Series({c: int((cells == c).sum()) for c in EFFECTS[effect]}, name="trials")


def expected_se(counts: pd.Series, effect: str, sigma: float) -> float:
    """σ · √(Σ w² / n): the SE a design reaches if every trial has SD σ (inf if a cell is empty)."""
    weights = EFFECTS[effect]
    if any(counts[c] == 0 for c in weights):
        return np.inf
    return float(sigma * np.sqrt(sum(weights[c] ** 2 / counts[c] for c in weights)))


def trials_from_pattern(pattern: str, seed: int = 0) -> pd.DataFrame:
    """A block from a string of C and I (e.g. "CCCCIIII..."): random words and inks, no repetitions."""
    from experiments.stroop import draw_stimulus

    rng = np.random.default_rng(seed)
    rows, previous = [], None
    for letter in pattern:
        previous = draw_stimulus(rng, letter == "C", previous)
        rows.append(previous)
    return pd.DataFrame(rows)[["word", "ink"]]


def coverage(trials: pd.DataFrame, length: int = 4) -> dict:
    """Coverage of the C/I sequences of `length` trials: how many occur, and the entropy of their counts.

    entropy = −Σ p · log2 p over the sequence counts; maximum = length bits (all 2^length sequences equally often).
    """
    letters = "".join(_letters(trials))
    sequences = [letters[i - length + 1:i + 1] for i in range(length - 1, len(letters))]
    counts = pd.Series(sequences).value_counts()
    p = counts / counts.sum()
    return {"sequences hit": f"{len(counts)} of {2 ** length}",
            "entropy (bits)": round(float(-(p * np.log2(p)).sum()), 2)}


def confounds(trials: pd.DataFrame) -> dict:
    """Warning signs for bias: congruency vs. trial position (practice), and repetitions of the previous trial."""
    letters = _letters(trials)
    congruent = (letters == "C").astype(float)
    ink, word = trials["ink"].to_numpy(), trials["word"].to_numpy()
    ink_repeat = np.r_[False, ink[1:] == ink[:-1]]
    complete_repeat = ink_repeat & np.r_[False, word[1:] == word[:-1]]
    return {
        "corr. congruent x trial": round(float(np.corrcoef(congruent, np.arange(len(trials)))[0, 1]), 2),
        "ink repetitions": int(ink_repeat.sum()),
        "complete repetitions": int(complete_repeat.sum()),
        "longest C/I run": max(len(r) for r in "".join(letters).replace("CI", "C I").replace("IC", "I C").split()),
    }


def design_report(trials: pd.DataFrame, sigma: float) -> pd.Series:
    """All before-data metrics of one design: cell counts, expected SEs, coverage, confounds."""
    report = {}
    for effect in ["Stroop effect", "sequence effect"]:
        counts = cell_counts(trials, effect)
        if effect == "sequence effect":
            report.update({f"trials {c}": n for c, n in counts.items()})
        report["SE Stroop (ms)" if effect == "Stroop effect" else "SE sequence (ms)"] = round(expected_se(counts, effect, sigma), 1)
    report.update(coverage(trials))
    report.update(confounds(trials))
    return pd.Series(report)


def scoreboard(designs: dict, sigma: float) -> pd.DataFrame:
    """One column per design (name -> trials), sorted by the expected SE of the sequence effect."""
    board = pd.DataFrame({name: design_report(trials, sigma) for name, trials in designs.items()})
    return board[board.loc["SE sequence (ms)"].astype(float).sort_values().index]


def pool_sequences(trials: pd.DataFrame) -> list:
    """The sequences of a design file: one block, or a pool of blocks (column `sequence`)."""
    if "sequence" not in trials:
        return [trials]
    return [d.reset_index(drop=True) for _, d in trials.groupby("sequence")]


def load_designs(folder: str) -> dict:
    """All designs in a folder (designs/<group>.csv); of a pool, its first sequence."""
    import glob
    import os

    designs = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        sequences = pool_sequences(pd.read_csv(f))
        name = os.path.splitext(os.path.basename(f))[0]
        if len(sequences) > 1:
            name += f" (1 of {len(sequences)} sequences)"
        designs[name] = sequences[0]
    return designs


def trial_sd(data: pd.DataFrame, default: float = 200.0) -> float:
    """SD of a single trial within a condition, from clean data (fallback: ~200 ms, Rouder, Kumar & Haaf 2023)."""
    if data.empty:
        return default
    clean = clean_trials(data)
    return float(clean.groupby(["participant", "block", "congruency"])["rt_ms"].std().mean())


# ------------------------------------------------------------------ after data collection

def player_estimate(block_data: pd.DataFrame, effect: str, exclude_ink_repetitions: bool = False,
                    min_trials: int = 2) -> dict:
    """Effect, SE and trials used for one player's block (cells from the played order, then cleaned)."""
    block_data = block_data.sort_values("trial").copy()
    block_data["cell"] = label_cells(block_data, effect).to_numpy()
    ink = block_data["ink"].to_numpy()
    block_data["ink_repeat"] = np.r_[False, ink[1:] == ink[:-1]]
    clean = clean_trials(block_data)
    if exclude_ink_repetitions:
        clean = clean[~clean["ink_repeat"]]
    estimate, variance, used = 0.0, 0.0, 0
    for c, w in EFFECTS[effect].items():
        rt = clean.loc[clean["cell"] == c, "rt_ms"]
        if len(rt) < min_trials:
            return {"effect": np.nan, "se": np.nan, "trials used": used}
        estimate += w * rt.mean()
        variance += w ** 2 * rt.var(ddof=1) / len(rt)
        used += len(rt)
    return {"effect": estimate, "se": np.sqrt(variance), "trials used": used}


def player_results(data: pd.DataFrame, effect: str, blocks=(DESIGNED_BLOCK, RANDOM_BLOCK),
                   exclude_ink_repetitions: bool = False) -> pd.DataFrame:
    """One row per player who played all `blocks`: estimate, SE and trials used per block."""
    played = data.groupby("participant")["block"].agg(set)
    players = sorted(p for p, b in played.items() if set(blocks) <= b)
    rows = []
    for p in players:
        row = {"participant": p}
        for block in blocks:
            est = player_estimate(data[(data["participant"] == p) & (data["block"] == block)], effect,
                                  exclude_ink_repetitions)
            row.update({f"{block}: {k}": v for k, v in est.items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("participant")


def class_summary(results: pd.DataFrame, blocks=(DESIGNED_BLOCK, RANDOM_BLOCK), paired: bool = False) -> pd.DataFrame:
    """Class level, per block: estimate ± SE (SD of the player estimates / √n); optionally the paired difference."""
    columns = {block: results[f"{block}: effect"] for block in blocks}
    if paired and len(blocks) == 2:
        columns[f"{blocks[0]} − {blocks[1]} (paired)"] = results[f"{blocks[0]}: effect"] - results[f"{blocks[1]}: effect"]
    rows = {}
    for name, x in columns.items():
        x = x.dropna()
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        rows[name] = {"players": len(x), "estimate (ms)": x.mean(), "SE (ms)": se,
                      "95% CI low": x.mean() - 1.96 * se, "95% CI high": x.mean() + 1.96 * se}
    return pd.DataFrame(rows).T.round(1)


def variance_split(results: pd.DataFrame, block: str) -> dict:
    """SE²(class) = (σ²_between + mean SE²_within) / n: which part comes from people, which from trials."""
    est, se = results[f"{block}: effect"].dropna(), results[f"{block}: se"].dropna()
    n = len(est)
    within = float(np.mean(se ** 2))
    between = max(float(est.var(ddof=1)) - within, 0.0)
    return {"players": n,
            "SD between people (ms)": round(np.sqrt(between), 1),
            "mean SE within a player (ms)": round(np.sqrt(within), 1),
            "class SE (ms)": round(np.sqrt((between + within) / n), 1),
            "class SE from trial noise only (ms)": round(np.sqrt(within / n), 1)}


def repetition_check(data: pd.DataFrame, effect: str = "sequence effect", block: str = RANDOM_BLOCK) -> pd.DataFrame:
    """The effect in one block type with all trials vs. without ink repetitions (class level)."""
    rows = {}
    for exclude, name in [(False, "all trials"), (True, "without ink repetitions")]:
        r = player_results(data, effect, blocks=(block,), exclude_ink_repetitions=exclude)
        x = r[f"{block}: effect"].dropna()
        rows[name] = {"players": len(x), "estimate (ms)": x.mean(), "SE (ms)": x.std(ddof=1) / np.sqrt(len(x)),
                      "trials used per player": r[f"{block}: trials used"].mean()}
    return pd.DataFrame(rows).T.round(1)


def example_class_data(design: pd.DataFrame, n_players: int = 24, seed: int = 7) -> pd.DataFrame:
    """Simulated stand-in data: every player plays the predefined design (a random sequence of the pool)
    and a fresh random block (order alternates).

    The default seed gives a typical class of 24 (a class this small scatters a lot - some seeds
    happen to show the random block as the more precise one).
    """
    from experiments.stroop import random_trials
    from resources.stroop_data import simulate_participant

    rng = np.random.default_rng(seed)
    sequences = pool_sequences(design)
    data = []
    for i in range(n_players):
        # one simulated person plays both blocks in one go
        designed = sequences[int(rng.integers(len(sequences)))]
        blocks = [(DESIGNED_BLOCK, designed[["word", "ink"]]),
                  (RANDOM_BLOCK, random_trials(seed=int(rng.integers(1e9)))[["word", "ink"]])]
        if i % 2:
            blocks.reverse()
        trials = pd.concat([t for _, t in blocks], ignore_index=True)
        d = simulate_participant(trials, f"SIM{i:02d}", rng)
        sizes = [len(t) for _, t in blocks]
        d["block"] = np.repeat([b for b, _ in blocks], sizes)
        d["trial"] = np.concatenate([np.arange(n) for n in sizes])
        d["design"] = d["block"]
        data.append(d)
    return pd.concat(data, ignore_index=True)


# ------------------------------------------------------------------ a black box with a known truth

def black_box_study(designs: dict, n_players: int = 500, seed: int = 0) -> pd.DataFrame:
    """Play designs on simulated players whose true effects are known, and compare estimates with the truth.

    `designs` maps a name to "random" (a fresh random block per player), "random, repetitions removed"
    (the same, but trials that repeat the previous ink colour are dropped in the analysis), a pool
    of sequences (column `sequence`, one picked per player) or a single block (every player plays the same).
    Returns, per design and effect: the bias (average estimate − truth), the SD of a single
    player's error, and how often the effect was not estimable.
    """
    from experiments.stroop import random_trials
    from resources.stroop_data import simulate_participant

    rng = np.random.default_rng(seed)
    truth_key = {"Stroop effect": "stroop", "sequence effect": "cse"}
    rows = []
    for name, design in designs.items():
        sequences = None if isinstance(design, str) else pool_sequences(design)
        remove_repetitions = isinstance(design, str) and design == "random, repetitions removed"
        errors = {effect: [] for effect in truth_key}
        for i in range(n_players):
            if sequences is None:
                trials = random_trials(seed=int(rng.integers(1e9)))[["word", "ink"]]
            else:
                trials = sequences[int(rng.integers(len(sequences)))][["word", "ink"]]
            data, truth = simulate_participant(trials.reset_index(drop=True), f"P{i}", rng, return_truth=True)
            for effect, key in truth_key.items():
                estimate = player_estimate(data, effect, exclude_ink_repetitions=remove_repetitions)["effect"]
                errors[effect].append(estimate - truth[key])
        for effect, e in errors.items():
            e = np.array(e, dtype=float)
            ok = e[~np.isnan(e)]
            rows.append({"design": name, "effect": effect, "bias (ms)": ok.mean(),
                         "SD per player (ms)": ok.std(), "not estimable (%)": 100 * np.isnan(e).mean()})
    return pd.DataFrame(rows).set_index(["effect", "design"]).sort_index().round(1)


# ------------------------------------------------------------------ figures

COLORS = ["#2f7bbf", "#ae0d36", "#666666", "#e08a00"]


def plot_estimates(tables: dict, ylabel: str = "class estimate ± 95% CI (ms)"):
    """One panel per table (e.g. per effect): every row's estimate with its 95% CI (estimate ± 1.96 SE)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.6), squeeze=False)
    for ax, (title, table) in zip(axes[0], tables.items()):
        for i, (name, row) in enumerate(table.iterrows()):
            ax.errorbar(i, row["estimate (ms)"], yerr=1.96 * row["SE (ms)"], fmt="o", ms=8, capsize=5, lw=2,
                        color=COLORS[i % len(COLORS)])
        ax.set_xticks(range(len(table)), [str(n).replace(" ", "\n", 1) for n in table.index])
        ax.set_xlim(-0.6, len(table) - 0.4)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_title(title)
    axes[0][0].set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_black_box(table, n_players: int = 24):
    """Where a class average of n players lands relative to the truth: bias ± 2 · SD / √n, per design and effect."""
    import matplotlib.pyplot as plt

    effects = table.index.get_level_values("effect").unique()
    fig, axes = plt.subplots(1, len(effects), figsize=(4.6 * len(effects), 3.8), squeeze=False)
    for ax, effect in zip(axes[0], effects):
        rows = table.loc[effect]
        for i, (name, row) in enumerate(rows.iterrows()):
            spread = 2 * row["SD per player (ms)"] / np.sqrt(n_players)
            ax.errorbar(i, row["bias (ms)"], yerr=spread, fmt="o", ms=8, capsize=5, lw=2, color=COLORS[i % len(COLORS)])
        ax.axhline(0, color="black", lw=1.2, ls="--")
        ax.text(len(rows) - 0.5, 0, " truth", va="center", fontsize=9)
        ax.set_xticks(range(len(rows)), [str(n).replace(", ", ",\n").replace(" ", "\n", 1) if len(str(n)) > 12 else str(n)
                                         for n in rows.index])
        ax.set_xlim(-0.6, len(rows) - 0.1)
        ax.set_title(effect)
    axes[0][0].set_ylabel(f"class of {n_players}: error (ms)\nbias ± 2 · SD / √{n_players}")
    fig.tight_layout()
    return fig
