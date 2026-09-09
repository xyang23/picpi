from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHED_RESULTS = ROOT / "cached_results"
FIGURES = ROOT / "figures"


def cached(name: str) -> Path:
    path = CACHED_RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES
