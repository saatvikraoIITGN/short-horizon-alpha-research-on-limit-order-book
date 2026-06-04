"""Project path utilities and config loading."""

from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return the project root directory (parent of config.yaml)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no config.yaml found)")


def load_config() -> dict:
    """Load the project configuration from config.yaml."""
    config_path = get_project_root() / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_data_dir(stage: str = "raw") -> Path:
    """Get data directory for a given stage ('raw', 'interim', 'processed')."""
    config = load_config()
    root = get_project_root()
    key = f"{stage}_dir"
    return root / config["data"].get(key, f"data/{stage}")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
