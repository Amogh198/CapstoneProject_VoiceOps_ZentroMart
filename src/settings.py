import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path=PROJECT_ROOT / ".env"):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


load_env_file()


def dataset_root():
    configured = os.environ.get("CAPSTONE_DATASET_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT


def required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in .env or the environment before running this command.")
    return value
