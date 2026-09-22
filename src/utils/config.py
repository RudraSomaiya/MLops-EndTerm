from pathlib import Path

import yaml

# walk up from this file to find project root where config.yaml lives
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    """Load project configuration from config.yaml."""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    return _PROJECT_ROOT
