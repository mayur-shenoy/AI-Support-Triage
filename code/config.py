from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_environment(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

