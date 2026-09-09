"""Repository paths shared by computation and plotting scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHED_RESULTS = ROOT / "cached_results"
FIGURES = ROOT / "figures"


def cached(name: str) -> Path:
    """Return a named cached-result directory, creating it when needed."""
    path = CACHED_RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    """Return the repository's figure directory, creating it when needed."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES
