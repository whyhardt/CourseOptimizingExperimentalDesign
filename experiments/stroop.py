"""Stroop task: name the ink colour of a colour word (optionally: read the word).

Examples (run from the repository root):
    python -m experiments.stroop --practice
    python -m experiments.stroop                       # asks for your ID; Enter assigns a new one - note it!
    python -m experiments.stroop --id K7QM --block 2   # another block with your ID (no question asked)
    python -m experiments.stroop --id K7QM --block comp1 --design designs/group1.csv

The default block is a random sequence with hidden probe sequences: a run of three congruent
(CCC) or three incongruent (III) trials, followed by a congruent or incongruent target trial.
They show how the Stroop effect depends on the recent history (congruency sequence effect).

A design file is a CSV with one row per trial and the columns `word`, `ink` and optionally
`task` ('color' or 'word'). Trials are played in the given order, e.g. a SweetPea sequence.
"""

import argparse
import os
import random
import sys
from typing import Optional

import numpy as np
import pandas as pd

from experiments.core import (Abort, Display, ask_participant_id, check_participant_id, new_participant_id,
                              save_data)

COLORS = {
    "red": (225, 45, 45),
    "green": (40, 170, 70),
    "blue": (60, 110, 240),
    "yellow": (235, 205, 40),
}
# response key for each colour (same mapping for both tasks)
KEYS = {"red": "d", "green": "f", "blue": "j", "yellow": "k"}

# probe sequences: history of three trials + target trial (C = congruent, I = incongruent)
PROBES = ["CCC-C", "CCC-I", "III-C", "III-I"]

FIXATION_DURATION = 0.5
RESPONSE_TIMEOUT = 2.0
FEEDBACK_DURATION = 0.6
ITI = 0.4


def draw_stimulus(rng, congruent: bool, previous: Optional[dict] = None) -> dict:
    """Draw one trial. If `previous` is given, neither its word nor its ink is repeated
    (avoids the feature-repetition confound of sequence effects, Mayr et al. 2003)."""
    colors = list(COLORS)
    banned = set() if previous is None else {previous["word"], previous["ink"]}
    inks = [c for c in colors if c not in banned]
    ink = rng.choice(inks)
    if congruent:
        word = ink
    else:
        word = rng.choice([c for c in colors if c != ink and c not in banned])
    return {"word": word, "ink": ink, "task": "color"}


def random_trials(n_trials: int = 96, p_congruent: float = 0.5, seed: Optional[int] = None) -> pd.DataFrame:
    """Random sampler: each trial is independently congruent with probability p_congruent.

    The ink colour is drawn uniformly; for incongruent trials the word is drawn uniformly
    from the remaining colours.
    """
    rng = np.random.default_rng(seed)
    trials = [draw_stimulus(rng, rng.random() < p_congruent) for _ in range(n_trials)]
    return pd.DataFrame(trials).assign(probe="", probe_position=-1)


def random_trials_with_probes(n_trials: int = 96, p_congruent: float = 0.5, probe_reps: int = 2,
                              min_gap: int = 3, seed: Optional[int] = None) -> pd.DataFrame:
    """Random sequence with each probe in PROBES hidden `probe_reps` times.

    Probes are placed at random positions with at least `min_gap` random trials before each
    of them. Within a probe (and at its first trial) no word or ink is repeated.
    """
    rng = np.random.default_rng(seed)
    probes = [p for p in PROBES for _ in range(probe_reps)]
    rng.shuffle(probes)
    n_random = n_trials - 4 * len(probes)
    if n_random < min_gap * len(probes):
        raise ValueError("Too many probes for the number of trials")

    # split the random trials into gaps before each probe (each >= min_gap) and a final gap
    extra = n_random - min_gap * len(probes)
    cuts = np.sort(rng.integers(0, extra + 1, size=len(probes)))
    gaps = np.diff(np.concatenate([[0], cuts])) + min_gap
    tail = n_random - gaps.sum()

    trials, previous = [], None

    def add(congruent, probe="", position=-1, constrained=False):
        nonlocal previous
        trial = draw_stimulus(rng, congruent, previous if constrained else None)
        trials.append({**trial, "probe": probe, "probe_position": position})
        previous = trial

    for gap, probe in zip(gaps, probes):
        for _ in range(gap):
            add(rng.random() < p_congruent)
        for position, letter in enumerate(probe.replace("-", "")):
            add(letter == "C", probe, position, constrained=True)
    for _ in range(tail):
        add(rng.random() < p_congruent)
    return pd.DataFrame(trials)


def load_trials(path: str) -> pd.DataFrame:
    """Load a trial sequence (e.g. exported from SweetPea) from a CSV file."""
    trials = pd.read_csv(path)
    trials.columns = [c.strip().lower() for c in trials.columns]
    missing = {"word", "ink"} - set(trials.columns)
    if missing:
        raise ValueError(f"Design file needs the columns {sorted(missing)}")
    if "task" not in trials:
        trials["task"] = "color"
    for column in ["word", "ink", "task"]:
        trials[column] = trials[column].astype(str).str.strip().str.lower()
    unknown = set(trials["word"]) | set(trials["ink"])
    unknown -= set(COLORS)
    if unknown:
        raise ValueError(f"Unknown colours in design file: {unknown}. Allowed: {list(COLORS)}")
    if not set(trials["task"]) <= {"color", "word"}:
        raise ValueError("The task column may only contain 'color' or 'word'")
    return trials[["word", "ink", "task"]].assign(probe="", probe_position=-1)


def draw_legend(display: Display):
    legend = "     ".join(f"{key.upper()} = {color}" for color, key in KEYS.items())
    display.text(legend, (display.width // 2, display.height - 60), size=32)


def run(trials: pd.DataFrame, display: Display, participant: str, block: str,
        practice: bool = False, design: str = "random") -> pd.DataFrame:
    """Present the trials and return one row of data per trial."""
    show_task_cue = (trials["task"] == "word").any()
    key_to_color = {key: color for color, key in KEYS.items()}
    center = (display.width // 2, display.height // 2)

    lines = ["Name the INK COLOUR of the word, ignore what it says.",
             "Respond as fast and as accurately as possible.", ""]
    if show_task_cue:
        lines = ["Above each word a cue tells you what to do:",
                 "COLOR = name the ink colour,  WORD = read the word.",
                 "Respond as fast and as accurately as possible.", ""]
    lines += ["Keys: see the bottom of the screen."]
    if practice:
        lines += ["", "This is a practice block with feedback."]
    else:
        lines += ["", f"Your participant ID:  {participant}   - please write it down!"]
    display.clear()
    display.paragraph(lines + ["", "Press SPACE to start"])
    draw_legend(display)
    display.flip()
    display.wait_for_key(["space"])

    data = []
    for index, trial in enumerate(trials.itertuples(index=False)):
        # fixation
        display.clear()
        display.text("+", center, size=80)
        draw_legend(display)
        display.flip()
        display.wait(FIXATION_DURATION)

        # stimulus
        display.clear()
        if show_task_cue:
            display.text(trial.task.upper(), (center[0], center[1] - 110), size=40)
        display.text(trial.word.upper(), center, size=120, color=COLORS[trial.ink])
        draw_legend(display)
        onset = display.flip()
        key, rt = display.wait_for_key(KEYS.values(), onset=onset, timeout=RESPONSE_TIMEOUT)

        target = trial.ink if trial.task == "color" else trial.word
        response = key_to_color.get(key)
        correct = response == target
        data.append({
            "participant": participant,
            "block": block,
            "trial": index,
            "task": trial.task,
            "word": trial.word,
            "ink": trial.ink,
            "congruency": "congruent" if trial.word == trial.ink else "incongruent",
            "correct_response": target,
            "response": response,
            "correct": int(correct),
            "rt_ms": None if rt is None else round(rt * 1000, 1),
            "timed_out": int(key is None),
            "probe": trial.probe,
            "probe_position": trial.probe_position,
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


def simulated_responder(keys):
    """Random key with a plausible RT; only for testing the pipeline."""
    return random.choice(keys), random.uniform(0.35, 1.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stroop task (data is saved to data/stroop/)")
    parser.add_argument("--id", help="reuse the participant ID you were given earlier (default: ask, or assign a new one)")
    parser.add_argument("--block", default="1", help="block label stored with the data (default: 1)")
    parser.add_argument("--n-trials", type=int, default=96, help="trials for the random design (default: 96)")
    parser.add_argument("--p-congruent", type=float, default=0.5, help="P(congruent) for the random design (default: 0.5)")
    parser.add_argument("--probe-reps", type=int, default=2,
                        help="repetitions of each hidden probe sequence in the random design; 0 = none (default: 2)")
    parser.add_argument("--design", help="CSV with columns word, ink[, task] to play instead of a random sequence")
    parser.add_argument("--seed", type=int, help="random seed for the random design")
    parser.add_argument("--practice", action="store_true", help="short practice block with feedback; nothing is saved")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--simulate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        if args.id:
            participant = check_participant_id(args.id)
        elif args.simulate or not sys.stdin.isatty():
            participant = new_participant_id()
        else:
            participant = ask_participant_id("stroop")
    except ValueError as error:
        print(error)
        return 1
    except (EOFError, KeyboardInterrupt):
        print()
        return 1

    if args.design:
        try:
            trials = load_trials(args.design)
        except (OSError, ValueError, KeyError) as error:
            print(f"Could not load design file {args.design}: {error}")
            return 1
        design = os.path.basename(args.design)
    elif args.practice:
        trials, design = random_trials(16, args.p_congruent, args.seed), "practice"
    elif args.probe_reps > 0:
        try:
            trials = random_trials_with_probes(args.n_trials, args.p_congruent, args.probe_reps, seed=args.seed)
        except ValueError as error:
            print(error)
            return 1
        design = f"random+probes(p_congruent={args.p_congruent}, reps={args.probe_reps})"
    else:
        trials, design = random_trials(args.n_trials, args.p_congruent, args.seed), f"random(p_congruent={args.p_congruent})"

    display = Display(fullscreen=args.fullscreen, simulate=simulated_responder if args.simulate else None)
    try:
        data = run(trials, display, participant, args.block, practice=args.practice, design=design)
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
    path = save_data(data, "stroop", participant, args.block)
    print(f"Saved {len(data)} trials to {path}")
    print(f"Your participant ID is {participant} - write it down, you need it to find your own data.")
    print("Please upload this file as instructed in the course.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
