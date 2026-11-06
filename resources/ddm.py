"""Simulated 2AFC player: a drift-diffusion model (DDM) that produces the choice AND the response time.

Evidence starts halfway between two bounds (0 and a) and accumulates with drift v plus noise
(SD 1 per second) until it reaches a bound: the upper bound = "blue", the lower one = "orange".
    drift            v  = k · (ratio − threshold),   k = slope(scatteredness) / a
    slope            slope = slope_0 + slope_scat · scatteredness
    response time    RT = non-decision time + time to reach the bound
With an unbiased start the choices follow exactly the logistic curve of the course:
    P(blue) = 1 / (1 + exp(−a · v)) = 1 / (1 + exp(−slope · (ratio − threshold)))
so threshold and slope keep their meaning; the response times come from the same process.
The path is simulated in steps of 1 ms (Euler), which shifts P(blue) very slightly. A path that
has not reached a bound when the task's time-out (2.5 s) is over gives no answer.
"""

from typing import Optional

import numpy as np

DT = 0.001          # seconds per step
TIMEOUT = 2.5       # seconds, as in experiments/twoafc.py


class DDMPlayer:
    """Call with (ratio, scatteredness) -> 1 (blue) / 0 (orange) / -1 (no answer); RT in self.last_rt_ms."""

    def __init__(self, threshold: float = 0.5, slope: float = 15.0, slope_scat: float = 0.0,
                 bound: float = 1.5, non_decision: float = 0.3, rng: Optional[np.random.Generator] = None):
        self.threshold, self.slope, self.slope_scat = threshold, slope, slope_scat
        self.bound, self.non_decision = bound, non_decision
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_rt_ms = np.nan

    def drift(self, ratio, scat):
        slope = self.slope + self.slope_scat * scat
        return slope / self.bound * (ratio - self.threshold)

    def p_blue(self, ratio, scat=0.0):
        """Exact choice probability of the continuous process (the logistic curve)."""
        return 1 / (1 + np.exp(-self.bound * self.drift(np.asarray(ratio), np.asarray(scat))))

    def trial(self, ratio: float, scat: float = 0.0):
        """One trial: (choice, RT in ms); choice -1 and RT nan if the time-out is reached."""
        n = int((TIMEOUT - self.non_decision) / DT)
        v = self.drift(float(ratio), float(scat))
        x = self.bound / 2 + np.cumsum(v * DT + np.sqrt(DT) * self.rng.standard_normal(n))
        hit = np.flatnonzero((x >= self.bound) | (x <= 0))
        if hit.size == 0:
            return -1, np.nan
        i = hit[0]
        return int(x[i] >= self.bound), round(1000 * (self.non_decision + (i + 1) * DT), 1)

    def __call__(self, ratio, scat=0.0):
        choice, self.last_rt_ms = self.trial(ratio, scat)
        return choice
