"""Wake-word debouncing logic, factored out of the main loop so it can be
unit-tested without spinning up audio or model dependencies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class WakewordDecision:
    triggered: bool
    score: float
    avg_score: float
    consecutive: int
    in_cooldown: bool


class WakewordDetector:
    """Smooths and debounces raw openWakeWord scores.

    A trigger is reported only when:
      - the latest score exceeds `threshold`,
      - the rolling average over the last `window` scores exceeds 85% of
        threshold (rejects single-frame false positives),
      - we have seen `required_consecutive` such scores in a row, and
      - the time since the last trigger exceeds `cooldown_seconds`.
    """

    def __init__(
        self,
        threshold: float,
        cooldown_seconds: float = 1.0,
        required_consecutive: int = 2,
        window: int = 5,
        avg_factor: float = 0.85,
    ):
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        if window <= 0:
            raise ValueError("window must be > 0")

        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.required_consecutive = required_consecutive
        self.avg_factor = avg_factor

        self._window: deque[float] = deque(maxlen=window)
        self._consecutive = 0
        self._last_trigger_at: float | None = None

    def reset(self) -> None:
        self._window.clear()
        self._consecutive = 0

    def feed(self, score: float, now: float) -> WakewordDecision:
        self._window.append(score)
        avg_score = sum(self._window) / len(self._window)

        in_cooldown = (
            self._last_trigger_at is not None
            and (now - self._last_trigger_at) <= self.cooldown_seconds
        )

        if score <= self.threshold:
            self._consecutive = 0
            return WakewordDecision(False, score, avg_score, 0, in_cooldown)

        if in_cooldown:
            return WakewordDecision(False, score, avg_score, self._consecutive, True)

        self._consecutive += 1
        triggered = (
            self._consecutive >= self.required_consecutive
            and avg_score > self.threshold * self.avg_factor
        )

        if triggered:
            self._last_trigger_at = now
            self._consecutive = 0
            self._window.clear()

        return WakewordDecision(triggered, score, avg_score, self._consecutive, False)
