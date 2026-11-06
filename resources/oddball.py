"""The odd-ball puzzle as a live game: 12 balls, one is heavier or lighter, a balance with three outcomes.

Play in the lecture (the class proposes weighings, you type them in):
    python -m resources.oddball
    weighing> 1 2 3 4 | 5 6 7 8

or in a notebook:
    game = OddBall()
    game.expected_gain([1, 2, 3, 4], [5, 6, 7, 8])
    game.weigh([1, 2, 3, 4], [5, 6, 7, 8])

The game hides one odd ball (and whether it is heavier or lighter) and keeps the list of
possibilities that are still consistent with all weighings so far. Before every weighing it can
show the expected information gain = the entropy of the three outcomes; after it, the bits
actually gained = log2(possibilities before / possibilities after).
"""

import random
from typing import Optional

import numpy as np

OUTCOMES = ["left side heavier", "right side heavier", "balance"]


def outcome(left, right, ball, heavier) -> str:
    """What the balance shows if `ball` is the odd one."""
    if ball in left:
        return OUTCOMES[0] if heavier else OUTCOMES[1]
    if ball in right:
        return OUTCOMES[1] if heavier else OUTCOMES[0]
    return OUTCOMES[2]


class OddBall:
    def __init__(self, n_balls: int = 12, seed: Optional[int] = None):
        self.n_balls = n_balls
        rng = random.Random(seed)
        self.secret = (rng.randint(1, n_balls), rng.random() < 0.5)
        # every possibility: (ball, heavier?)
        self.possible = [(b, h) for b in range(1, n_balls + 1) for h in (True, False)]
        self.history = []

    @property
    def entropy(self) -> float:
        """Remaining uncertainty in bits (all remaining possibilities equally likely)."""
        return float(np.log2(len(self.possible)))

    def _check(self, left, right):
        left, right = list(left), list(right)
        if len(left) != len(right):
            raise ValueError("put the same number of balls on both sides")
        if set(left) & set(right):
            raise ValueError("a ball cannot be on both sides")
        if not set(left + right) <= set(range(1, self.n_balls + 1)):
            raise ValueError(f"balls are numbered 1 to {self.n_balls}")
        return left, right

    def outcome_probabilities(self, left, right) -> dict:
        """How likely each outcome is, given the remaining possibilities."""
        left, right = self._check(left, right)
        counts = {o: 0 for o in OUTCOMES}
        for ball, heavier in self.possible:
            counts[outcome(left, right, ball, heavier)] += 1
        return {o: c / len(self.possible) for o, c in counts.items()}

    def expected_gain(self, left, right) -> float:
        """Expected information gain of a weighing = entropy of its outcome, in bits."""
        p = np.array([q for q in self.outcome_probabilities(left, right).values() if q > 0])
        return float(abs(-(p * np.log2(p)).sum()))

    def weigh(self, left, right) -> dict:
        """Weigh, keep the consistent possibilities, and report the outcome and the bits gained."""
        left, right = self._check(left, right)
        result = outcome(left, right, *self.secret)
        before = len(self.possible)
        expected = self.expected_gain(left, right)
        self.possible = [(b, h) for b, h in self.possible if outcome(left, right, b, h) == result]
        gained = float(np.log2(before / len(self.possible)))
        step = {"weighing": f"{left} vs {right}", "outcome": result, "expected gain (bits)": round(expected, 2),
                "gained (bits)": round(gained, 2), "possibilities left": len(self.possible),
                "uncertainty left (bits)": round(self.entropy, 2)}
        self.history.append(step)
        return step

    def remaining(self) -> list:
        """The possibilities still consistent with all weighings, e.g. '5 heavier'."""
        return [f"{b} {'heavier' if h else 'lighter'}" for b, h in self.possible]


def _parse(line: str):
    left, right = line.split("|")
    return [int(x) for x in left.split()], [int(x) for x in right.split()]


def main():
    game = OddBall()
    print(f"12 balls, one is heavier or lighter: {len(game.possible)} possibilities = {game.entropy:.2f} bits.")
    print("Type a weighing as 'left balls | right balls', e.g. 1 2 3 4 | 5 6 7 8")
    print("Add '?' at the end to only see the expected gain, e.g. 1 2 3 4 | 5 6 7 8 ?   (q = quit)")
    while len(game.possible) > 1:
        try:
            line = input("\nweighing> ").strip()
        except EOFError:
            break
        if line in ("q", "quit"):
            break
        try:
            preview = line.endswith("?")
            left, right = _parse(line.rstrip("?"))
            if preview:
                for o, p in game.outcome_probabilities(left, right).items():
                    print(f"  P({o}) = {p:.2f}")
                print(f"  expected gain: {game.expected_gain(left, right):.2f} bits")
                continue
            step = game.weigh(left, right)
        except (ValueError, IndexError) as error:
            print(f"  {error}")
            continue
        print(f"  -> {step['outcome']}")
        print(f"  expected {step['expected gain (bits)']} bits, gained {step['gained (bits)']} bits")
        print(f"  {step['possibilities left']} possibilities left ({step['uncertainty left (bits)']} bits)")
        if step["possibilities left"] <= 6:
            print("  still possible:", ", ".join(game.remaining()))
    ball, heavier = game.secret
    found = "found" if len(game.possible) == 1 else "not found yet"
    print(f"\nThe odd ball was {ball}, {'heavier' if heavier else 'lighter'} ({found} after {len(game.history)} weighings).")


if __name__ == "__main__":
    main()
