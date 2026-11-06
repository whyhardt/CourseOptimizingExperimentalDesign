"""Shared building blocks for the local experiments: window, text, timed key responses, saving."""

import os
import platform
import re
import secrets
import sys
import time
from datetime import datetime
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd

# hide the pygame greeting before importing it
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# Linux only: draw without OpenGL. conda's libstdc++ is often too old for the system GL drivers,
# which makes the window close immediately (GLX BadValue); 2D text does not need GL anyway.
# On macOS this setting can make window creation fail, so it is not set there.
if sys.platform.startswith("linux"):
    os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")
import pygame

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# participant IDs are assigned, not chosen: 4 characters without look-alikes (0/O, 1/I/L)
ID_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
ID_PATTERN = re.compile(f"^[{ID_ALPHABET}]{{4}}$")

BACKGROUND = (30, 30, 30)
FOREGROUND = (230, 230, 230)


class Abort(Exception):
    """Raised when the participant presses ESC or closes the window."""


class Display:
    """A pygame window with helpers for drawing text and collecting key presses.

    Args:
        fullscreen: open a fullscreen window instead of a 1024x768 one
        simulate: optional function (allowed_keys) -> (key, rt_seconds) that replaces
            real key presses. Used for testing the pipeline without a human.
    """

    def __init__(self, fullscreen: bool = False, simulate: Optional[Callable] = None):
        pygame.init()
        flags = pygame.FULLSCREEN if fullscreen else 0
        size = (0, 0) if fullscreen else (1024, 768)
        self.screen = self._open_window(size, flags)
        pygame.display.set_caption("Optimizing Experimental Design")
        pygame.mouse.set_visible(not fullscreen)
        self.width, self.height = self.screen.get_size()
        self.simulate = simulate
        self._fonts: Dict[int, pygame.font.Font] = {}

    @staticmethod
    def _open_window(size, flags):
        """Open the window; if that fails, retry once with the other framebuffer setting
        (with / without hardware acceleration), which is the usual cause on some machines."""
        try:
            return pygame.display.set_mode(size, flags)
        except pygame.error as first_error:
            previous = os.environ.get("SDL_FRAMEBUFFER_ACCELERATION")
            if previous is None:
                os.environ["SDL_FRAMEBUFFER_ACCELERATION"] = "0"
            else:
                del os.environ["SDL_FRAMEBUFFER_ACCELERATION"]
            pygame.display.quit()
            pygame.display.init()
            try:
                return pygame.display.set_mode(size, flags)
            except pygame.error as second_error:
                raise RuntimeError(
                    "Could not open the experiment window.\n"
                    f"  first try:  {str(first_error) or '(no message)'}\n"
                    f"  second try: {str(second_error) or '(no message)'}\n"
                    f"  system: {platform.platform()}, Python {platform.python_version()}, "
                    f"pygame {pygame.version.ver}, SDL {'.'.join(map(str, pygame.get_sdl_version()))}, "
                    f"video driver: {os.environ.get('SDL_VIDEODRIVER', 'default')}\n"
                    "Please send this message to the instructor."
                ) from second_error

    def font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def clear(self):
        self.screen.fill(BACKGROUND)

    def text(self, text: str, center: Tuple[int, int], size: int = 40, color=FOREGROUND):
        surface = self.font(size).render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=center))

    def paragraph(self, lines: Iterable[str], size: int = 36, spacing: float = 1.4):
        """Draw several lines of text centred on the screen."""
        lines = list(lines)
        line_height = int(size * spacing)
        top = self.height // 2 - (len(lines) - 1) * line_height // 2
        for i, line in enumerate(lines):
            self.text(line, (self.width // 2, top + i * line_height), size)

    def flip(self) -> float:
        """Show the drawn frame and return its onset time (perf_counter seconds)."""
        pygame.display.flip()
        return time.perf_counter()

    def _check_quit(self, event):
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            raise Abort()

    def wait(self, duration: float):
        """Wait for `duration` seconds while still reacting to ESC."""
        if self.simulate is not None:
            return
        end = time.perf_counter() + duration
        while time.perf_counter() < end:
            for event in pygame.event.get():
                self._check_quit(event)
            time.sleep(0.001)

    def wait_for_key(self, keys: Iterable[str], onset: Optional[float] = None,
                     timeout: Optional[float] = None) -> Tuple[Optional[str], Optional[float]]:
        """Wait for one of `keys` (pygame key names, e.g. 'd', 'space').

        Returns:
            (key, reaction time in seconds since `onset`) or (None, None) on timeout
        """
        keys = list(keys)
        if self.simulate is not None:
            key, rt = self.simulate(keys)
            if timeout is not None and rt > timeout:
                return None, None
            return key, rt

        if onset is None:
            onset = time.perf_counter()
        pygame.event.clear(pygame.KEYDOWN)
        while True:
            for event in pygame.event.get():
                self._check_quit(event)
                if event.type == pygame.KEYDOWN and pygame.key.name(event.key) in keys:
                    return pygame.key.name(event.key), time.perf_counter() - onset
            if timeout is not None and time.perf_counter() - onset > timeout:
                return None, None
            time.sleep(0.0005)

    def instructions(self, lines: Iterable[str]):
        """Show instructions until the participant presses SPACE."""
        self.clear()
        self.paragraph(list(lines) + ["", "Press SPACE to start"])
        self.flip()
        self.wait_for_key(["space"])

    def close(self):
        pygame.quit()


def save_data(data: pd.DataFrame, task: str, participant: str, block: str,
              out_dir: Optional[str] = None) -> str:
    """Save trial data as CSV to data/<task>/<task>_<participant>_<block>_<timestamp>.csv."""
    out_dir = out_dir or os.path.join(DATA_DIR, task)
    os.makedirs(out_dir, exist_ok=True)
    safe = lambda s: re.sub(r"[^A-Za-z0-9_-]", "-", str(s))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"{task}_{safe(participant)}_{safe(block)}_{timestamp}.csv")
    data.to_csv(path, index=False)
    return path


def new_participant_id() -> str:
    """Random anonymous participant ID, e.g. 'K7QM'."""
    return "".join(secrets.choice(ID_ALPHABET) for _ in range(4))


def check_participant_id(participant: str) -> str:
    """Normalise an ID given on the command line; raise ValueError if it was not assigned by us."""
    participant = participant.strip().upper()
    if not ID_PATTERN.match(participant):
        raise ValueError(f"'{participant}' is not a valid participant ID (4 characters, as shown by the program)")
    return participant


def known_participant_ids(task: str) -> list:
    """IDs that already have saved data of this task on this computer (from the file names)."""
    folder = os.path.join(DATA_DIR, task)
    names = os.listdir(folder) if os.path.isdir(folder) else []
    ids = {name.split("_")[1] for name in names if name.startswith(f"{task}_") and name.count("_") >= 3}
    return sorted(i for i in ids if ID_PATTERN.match(i))


def ask_participant_id(task: str) -> str:
    """Ask in the terminal for an existing ID; an empty answer assigns a new one."""
    known = known_participant_ids(task)
    if known:
        print(f"Participant IDs with saved data on this computer: {', '.join(known)}")
    while True:
        answer = input("Your participant ID (press Enter if you do not have one yet): ")
        if not answer.strip():
            return new_participant_id()
        try:
            return check_participant_id(answer)
        except ValueError as error:
            print(error)
