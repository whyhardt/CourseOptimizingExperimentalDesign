"""2AFC tiles task: are there more BLUE or more ORANGE tiles in the grid?

Examples (run from the repository root):
    python -m experiments.twoafc --practice                 # unrecorded practice block with feedback
    python -m experiments.twoafc --block train              # asks for your ID; Enter assigns a new one - note it!
    python -m experiments.twoafc --id K7QM --block test     # second block with the same ID
    python -m experiments.twoafc --block seq --ratio-hazard 0.1 --ratio-drift 0.02
    python -m experiments.twoafc --block exp --design designs/my_design.csv
    python -m experiments.twoafc --id K7QM --block exp_train --experimentalist uncertainty
    python -m experiments.twoafc --id K7QM --block exp_train --experimentalist my_experimentalist.py:my_experimentalist

Closed loop (--experimentalist): the block starts with --n-start random trials; then, in cycles of --cycle
trials, AutoRA fits the theorist (the GP on choice and RT) to your trials so far and your experimentalist
proposes the next trials (resources/autora_2afc.py describes the experimentalist's form). Without
--experimentalist the block is a plain random block.

Within-trial factors (both continuous, in [0, 1]):
    ratio          proportion of blue tiles (0.5 = balanced). The grid has an odd number of
                   tiles, so there is always a correct answer; the number of blue tiles is
                   ratio * 225, rounded up or down at random (seeded per trial)
    scatteredness  0 = the two colours are segregated by a straight line at a random angle,
                   1 = the tiles are placed completely at random; values in between blur the border

Across-trial factors: each within-trial factor follows its own process over the block.
    hazard   probability that the level jumps to a new random value on a trial (change points)
    drift    SD of the random-walk step of the level from one trial to the next
    jitter   SD of the trial-to-trial noise around the current level
The defaults (hazard 1, drift 0, jitter 0) give the random sampler: every trial is an
independent uniform draw from the range. Values leaving the range are reflected back into it.

A design file is a CSV with one row per trial and the columns `ratio` and `scatteredness`
(optionally `seed` to fix the tile layout). Trials are played in the given order. A file with a
`sequence` column is a pool of sequences; one of them is picked at random.
"""

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from experiments.core import (Abort, Display, ask_participant_id, check_participant_id, new_participant_id,
                              save_data)
import pygame  # imported after experiments.core, which sets the SDL environment first

TASK = "twoafc"
ROWS, COLS = 15, 15  # odd number of tiles -> never a tie

COLORS = {"blue": (47, 123, 191), "orange": (235, 140, 35)}
KEYS = {"blue": "f", "orange": "j"}

FIXATION_DURATION = 0.5
RESPONSE_TIMEOUT = 2.5
FEEDBACK_DURATION = 0.6
ITI = 0.4


# --------------------------------------------------------------------------------------------
# Stimulus
# --------------------------------------------------------------------------------------------

def n_blue_tiles(ratio: float, rng: np.random.Generator, n_tiles: int = ROWS * COLS) -> int:
    """Number of blue tiles shown for a requested ratio.

    Stochastic rounding: ratio * n_tiles is rounded up with probability equal to its fractional
    part, so the expected number of blue tiles is exactly ratio * n_tiles (and ratio 0.5 favours
    neither colour).
    """
    return int(np.clip(np.floor(ratio * n_tiles + rng.random()), 0, n_tiles))


def tile_layout(ratio: float, scatteredness: float, seed: int, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """Boolean (rows, cols) array, True = blue tile.

    Each tile gets a score that mixes its position along a random direction (segregated) with a
    uniform random number (scattered): score = (1 - scatteredness) * position + scatteredness * noise.
    Both parts are ranks scaled to [0, 1]. The n_blue tiles with the lowest score are blue.
    """
    rng = np.random.default_rng(seed)
    n_tiles = rows * cols
    n_blue = n_blue_tiles(ratio, rng, n_tiles)
    y, x = np.mgrid[0:rows, 0:cols]
    angle = rng.uniform(0, 2 * np.pi)
    projection = (x * np.cos(angle) + y * np.sin(angle)).ravel()
    projection = projection + rng.uniform(0, 1e-6, n_tiles)  # break ties at random
    position = np.argsort(np.argsort(projection)) / (n_tiles - 1)
    noise = rng.permutation(n_tiles) / (n_tiles - 1)
    score = (1 - scatteredness) * position + scatteredness * noise
    blue = np.zeros(n_tiles, dtype=bool)
    blue[np.argsort(score, kind="stable")[:n_blue]] = True
    return blue.reshape(rows, cols)


# --------------------------------------------------------------------------------------------
# Designs (trial sequences)
# --------------------------------------------------------------------------------------------

@dataclass
class Process:
    """How one within-trial factor evolves over a block (see module docstring)."""
    low: float = 0.0
    high: float = 1.0
    hazard: float = 1.0
    drift: float = 0.0
    jitter: float = 0.0

    def describe(self) -> str:
        return f"[{self.low:g},{self.high:g}] hazard={self.hazard:g} drift={self.drift:g} jitter={self.jitter:g}"


def _reflect(value: float, low: float, high: float) -> float:
    """Reflect a value back into [low, high]."""
    if high <= low:
        return low
    width = high - low
    value = (value - low) % (2 * width)
    return low + (value if value <= width else 2 * width - value)


def simulate_process(process: Process, n_trials: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Values of one factor over a block, and a 0/1 array marking the trials with a jump."""
    values, jumps = np.empty(n_trials), np.zeros(n_trials, dtype=int)
    level = rng.uniform(process.low, process.high)
    for t in range(n_trials):
        if t > 0:
            if rng.random() < process.hazard:
                level, jumps[t] = rng.uniform(process.low, process.high), 1
            else:
                level = _reflect(level + rng.normal(0, process.drift), process.low, process.high)
        values[t] = _reflect(level + rng.normal(0, process.jitter), process.low, process.high)
    jumps[0] = 1
    return values, jumps


def sequence_trials(n_trials: int = 100, ratio: Process = Process(0.1, 0.9), scatteredness: Process = Process(),
                    seed: Optional[int] = None) -> pd.DataFrame:
    """Trial sequence in which ratio and scatteredness each follow their own process.

    With the default processes (hazard 1, no drift, no jitter) this is the random sampler.
    """
    rng = np.random.default_rng(seed)
    r, r_jump = simulate_process(ratio, n_trials, rng)
    s, s_jump = simulate_process(scatteredness, n_trials, rng)
    return pd.DataFrame({
        "ratio": r.round(4), "scatteredness": s.round(4),
        "ratio_jump": r_jump, "scatteredness_jump": s_jump,
        "seed": rng.integers(0, 2**31 - 1, n_trials),
    })


def random_trials(n_trials: int = 100, ratio_range=(0.1, 0.9), scatteredness_range=(0.0, 1.0),
                  seed: Optional[int] = None) -> pd.DataFrame:
    """Random sampler: every trial is an independent uniform draw of ratio and scatteredness."""
    return sequence_trials(n_trials, Process(*ratio_range), Process(*scatteredness_range), seed)


def load_trials(path: str, seed: Optional[int] = None) -> pd.DataFrame:
    """Load a trial sequence (columns ratio, scatteredness[, seed]) from a CSV file."""
    trials = pd.read_csv(path)
    trials.columns = [c.strip().lower() for c in trials.columns]
    pick = None
    if "sequence" in trials:
        pick = int(random.choice(sorted(trials["sequence"].unique())))
        trials = trials[trials["sequence"] == pick].reset_index(drop=True)
    missing = {"ratio", "scatteredness"} - set(trials.columns)
    if missing:
        raise ValueError(f"Design file needs the columns {sorted(missing)}")
    for column in ["ratio", "scatteredness"]:
        trials[column] = pd.to_numeric(trials[column], errors="raise").astype(float)
        if not trials[column].between(0, 1).all():
            raise ValueError(f"All values of '{column}' must lie in [0, 1]")
    if "seed" not in trials:
        trials["seed"] = np.random.default_rng(seed).integers(0, 2**31 - 1, len(trials))
    trials = trials[["ratio", "scatteredness", "seed"]].assign(ratio_jump=-1, scatteredness_jump=-1)
    if pick is not None:
        trials.attrs["sequence"] = pick
    return trials


# --------------------------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------------------------

def draw_legend(display: Display):
    y = display.height - 50
    display.text(f"{KEYS['blue'].upper()} = more BLUE", (display.width // 2 - 160, y), size=34, color=COLORS["blue"])
    display.text(f"{KEYS['orange'].upper()} = more ORANGE", (display.width // 2 + 160, y), size=34,
                 color=COLORS["orange"])


def draw_grid(display: Display, layout: np.ndarray):
    rows, cols = layout.shape
    size = int(min(display.height - 160, display.width - 100))
    cell = size // max(rows, cols)
    gap = max(1, cell // 12)
    left = (display.width - cell * cols) // 2
    top = (display.height - 100 - cell * rows) // 2
    for i in range(rows):
        for j in range(cols):
            color = COLORS["blue"] if layout[i, j] else COLORS["orange"]
            pygame.draw.rect(display.screen, color, (left + j * cell, top + i * cell, cell - gap, cell - gap))


def run(trials: pd.DataFrame, display: Display, participant: str, block: str,
        practice: bool = False, design: str = "random", next_trials=None) -> pd.DataFrame:
    """Present the trials and return one row of data per trial.

    next_trials (optional): called with all data so far whenever the trials run out; returns the next
    trials (same columns as `trials`) or None to end the block. This is how an experimentalist plugs in.
    """
    key_to_color = {key: color for color, key in KEYS.items()}
    center = (display.width // 2, (display.height - 100) // 2)
    n_tiles = ROWS * COLS

    lines = ["You will see a grid of blue and orange tiles.",
             "Decide which colour covers MORE tiles.",
             "Respond as fast and as accurately as possible.", "",
             "Keys: see the bottom of the screen."]
    if practice:
        lines += ["", "This is a practice block with feedback."]
    else:
        lines += ["", f"Your participant ID:  {participant}   - please write it down!"]
    display.clear()
    display.paragraph(lines + ["", "Press SPACE to start"])
    draw_legend(display)
    display.flip()
    display.wait_for_key(["space"])

    data, queue, index = [], list(trials.itertuples(index=False)), -1
    while True:
        if not queue:
            more = None if next_trials is None else next_trials(pd.DataFrame(data))
            if more is None or len(more) == 0:
                break
            queue = list(more.itertuples(index=False))
        trial = queue.pop(0)
        index += 1
        layout = tile_layout(trial.ratio, trial.scatteredness, int(trial.seed))
        n_blue = int(layout.sum())
        majority = "blue" if n_blue > n_tiles / 2 else "orange"

        display.clear()
        display.text("+", center, size=80)
        draw_legend(display)
        display.flip()
        display.wait(FIXATION_DURATION)

        display.clear()
        draw_grid(display, layout)
        draw_legend(display)
        onset = display.flip()
        key, rt = display.wait_for_key(KEYS.values(), onset=onset, timeout=RESPONSE_TIMEOUT)

        response = key_to_color.get(key)
        correct = response == majority
        data.append({
            "participant": participant,
            "block": block,
            "trial": index,
            "ratio": trial.ratio,
            "scatteredness": trial.scatteredness,
            "n_blue": n_blue,
            "n_tiles": n_tiles,
            "ratio_shown": round(n_blue / n_tiles, 4),
            "majority": majority,
            "response": response,
            "choice_blue": None if response is None else int(response == "blue"),
            "correct": int(correct),
            "rt_ms": None if rt is None else round(rt * 1000, 1),
            "timed_out": int(key is None),
            "ratio_jump": trial.ratio_jump,
            "scatteredness_jump": trial.scatteredness_jump,
            "seed": int(trial.seed),
            "design": design,
        })

        if practice:
            message, color = ("correct", (90, 200, 90)) if correct else ("wrong", (230, 80, 80))
            if key is None:
                message = "too slow"
            display.clear()
            display.text(message, center, size=60, color=color)
            draw_legend(display)
            display.flip()
            display.wait(FEEDBACK_DURATION)

        display.clear()
        draw_legend(display)
        display.flip()
        display.wait(ITI)

    display.clear()
    if practice:
        display.paragraph(["Done, thank you!", "", "Press SPACE to close."])
    else:
        display.paragraph(["Done, thank you!", "", f"Your participant ID:  {participant}",
                           "Write it down - you need it to find your own data.", "", "Press SPACE to close."])
    display.flip()
    display.wait_for_key(["space"])
    return pd.DataFrame(data)


def closed_loop_trials(spec: str, n_trials: int, n_start: int, cycle: int, seed: Optional[int] = None):
    """The first n_start random trials and the callback that runs the AutoRA loop between cycles.

    Between cycles (while the player sees the pause screen): theorist (the GP) fits all data so far,
    the experimentalist proposes the next `cycle` trials.
    """
    from resources.autora_2afc import add_data, load_experimentalist, new_state, random_experimentalist, theorist

    experimentalist = load_experimentalist(spec)
    rng = np.random.default_rng(seed)
    state = {"s": new_state()}

    def as_trials(conditions: pd.DataFrame) -> pd.DataFrame:
        missing = {"ratio", "scatteredness"} - set(conditions.columns)
        if missing:
            raise ValueError(f"the experimentalist must return the columns ratio and scatteredness (missing {sorted(missing)})")
        t = conditions[["ratio", "scatteredness"]].astype(float).clip(0, 1).reset_index(drop=True)
        return t.assign(seed=rng.integers(0, 2**31 - 1, len(t)), ratio_jump=-1, scatteredness_jump=-1)

    first = random_experimentalist(state["s"], num_samples=min(n_start, n_trials), random_state=int(rng.integers(2**31)))

    def next_trials(data: pd.DataFrame):
        if len(data) >= n_trials:
            return None
        seen = len(state["s"].experiment_data)
        s = add_data(state["s"], data.iloc[seen:])
        s = theorist(s)
        s = experimentalist(s, num_samples=min(cycle, n_trials - len(data)), random_state=int(rng.integers(2**31)))
        state["s"] = s
        return as_trials(s.conditions)

    return as_trials(first.conditions), next_trials


def simulated_responder(keys):
    """Random key with a plausible RT; only for testing the pipeline."""
    return random.choice(keys), random.uniform(0.35, 1.4)


def main(argv=None):
    parser = argparse.ArgumentParser(description="2AFC tiles task (data is saved to data/twoafc/)")
    parser.add_argument("--id", help="reuse the participant ID you were given earlier (default: ask, or assign a new one)")
    parser.add_argument("--block", default="train", help="block label stored with the data (default: train)")
    parser.add_argument("--n-trials", type=int, default=100, help="trials in the block (default: 100)")
    parser.add_argument("--design", help="CSV with columns ratio, scatteredness[, seed] to play instead of a generated sequence")
    parser.add_argument("--seed", type=int, help="random seed for the generated sequence")
    parser.add_argument("--practice", action="store_true", help="short practice block with feedback; nothing is saved")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--simulate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--experimentalist", help="run a closed loop: 'random', 'uncertainty' or path/to/file.py:function "
                        "(your experimentalist, see resources/autora_2afc.py); default: a plain random block")
    parser.add_argument("--n-start", type=int, default=20, help="closed loop: random trials before the first cycle (default: 20)")
    parser.add_argument("--cycle", type=int, default=10, help="closed loop: trials per cycle (default: 10)")
    defaults = {"ratio": Process(0.1, 0.9), "scatteredness": Process(0.0, 1.0)}
    for name, short in [("ratio", "ratio"), ("scatteredness", "scat")]:
        d = defaults[name]
        group = parser.add_argument_group(f"sequence of {name}")
        group.add_argument(f"--{short}-range", type=float, nargs=2, default=(d.low, d.high), metavar=("LOW", "HIGH"),
                           help=f"range of {name} (default: {d.low:g} {d.high:g})")
        group.add_argument(f"--{short}-hazard", type=float, default=1.0,
                           help="probability of a jump to a new random level per trial (default: 1 = random sampler)")
        group.add_argument(f"--{short}-drift", type=float, default=0.0, help="SD of the random-walk step of the level (default: 0)")
        group.add_argument(f"--{short}-jitter", type=float, default=0.0, help="SD of the trial-to-trial noise (default: 0)")
    args = parser.parse_args(argv)

    processes = {}
    for name, short in [("ratio", "ratio"), ("scatteredness", "scat")]:
        low, high = getattr(args, f"{short}_range")
        process = Process(low, high, getattr(args, f"{short}_hazard"), getattr(args, f"{short}_drift"),
                          getattr(args, f"{short}_jitter"))
        if not (0 <= process.low <= process.high <= 1 and 0 <= process.hazard <= 1
                and process.drift >= 0 and process.jitter >= 0):
            print(f"Invalid settings for {name}: {process.describe()}")
            return 1
        processes[name] = process

    try:
        if args.id:
            participant = check_participant_id(args.id)
        elif args.simulate or not sys.stdin.isatty():
            participant = new_participant_id()
        else:
            participant = ask_participant_id(TASK)
    except ValueError as error:
        print(error)
        return 1
    except (EOFError, KeyboardInterrupt):
        print()
        return 1

    if args.design:
        try:
            trials = load_trials(args.design, args.seed)
        except (OSError, ValueError, KeyError) as error:
            print(f"Could not load design file {args.design}: {error}")
            return 1
        design = os.path.basename(args.design)
        if "sequence" in trials.attrs:
            design += f" #{trials.attrs['sequence']}"
    elif args.practice:
        trials, design = sequence_trials(20, processes["ratio"], processes["scatteredness"], args.seed), "practice"
    else:
        trials = sequence_trials(args.n_trials, processes["ratio"], processes["scatteredness"], args.seed)
        is_random = all(p.hazard == 1 and p.jitter == 0 for p in processes.values())
        design = ("random" if is_random else "sequence") + \
            f"(ratio {processes['ratio'].describe()}; scatteredness {processes['scatteredness'].describe()})"

    next_trials = None
    if args.experimentalist and not args.practice and not args.design:
        try:
            trials, next_trials = closed_loop_trials(args.experimentalist, args.n_trials, args.n_start, args.cycle, args.seed)
        except Exception as error:                     # e.g. a broken experimentalist file
            print(f"Could not load the experimentalist {args.experimentalist}: {error}")
            return 1
        design = f"experimentalist {args.experimentalist}"

    display = Display(fullscreen=args.fullscreen, simulate=simulated_responder if args.simulate else None)
    try:
        data = run(trials, display, participant, args.block, practice=args.practice, design=design,
                   next_trials=next_trials)
        aborted = False
    except Abort:
        data, aborted = None, True
    finally:
        display.close()

    if aborted:
        print("Aborted with ESC - no data saved.")
        return 1
    if args.practice:
        print(f"Practice done: {data['correct'].mean():.0%} correct. Nothing was saved.")
        return 0
    path = save_data(data, TASK, participant, args.block)
    print(f"Saved {len(data)} trials to {path}")
    print(f"Your participant ID is {participant} - write it down, you need it to find your own data.")
    print("Please upload this file as instructed in the course.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
