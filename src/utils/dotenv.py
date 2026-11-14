"""
Environment Variable Loader Utility.

Provides safe loading from .env without clobbering existing shell environment variables,
supporting inline comments, quotes, BOM, and export prefixes. Allows selective key loading
to prevent leaking irrelevant secrets into process memory.
"""

import os
from pathlib import Path
from typing import Dict, Iterable, Optional


def parse_dotenv_file(filepath: Path) -> Dict[str, str]:
    """
    Parse a .env file into a dictionary of key-value pairs.
    Handles 'export KEY=val', quotes, inline comments (#), empty lines, and BOM.
    """
    env_vars: Dict[str, str] = {}
    if not filepath.exists() or not filepath.is_file():
        return env_vars

    try:
        content = filepath.read_text(encoding="utf-8-sig")
    except Exception:
        return env_vars

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Strip leading 'export '
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()

        if not key:
            continue

        # Handle inline comments after quotes or unquoted values
        if val.startswith('"'):
            # Double quoted
            end_idx = val.find('"', 1)
            if end_idx != -1:
                val = val[1:end_idx]
            else:
                val = val.strip('"')
        elif val.startswith("'"):
            # Single quoted
            end_idx = val.find("'", 1)
            if end_idx != -1:
                val = val[1:end_idx]
            else:
                val = val.strip("'")
        else:
            # Unquoted: strip inline comments if present
            if "#" in val:
                val = val.split("#", 1)[0].strip()

        env_vars[key] = val

    return env_vars


def load_dotenv_keys(
    keys: Optional[Iterable[str]] = None,
    env_path: Optional[Path] = None,
) -> None:
    """
    Safely load specified keys from .env into os.environ using setdefault.
    Existing environment variables are NEVER overwritten.
    If `keys` is None, all keys parsed from .env are setdefault-loaded.
    """
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"

    parsed = parse_dotenv_file(env_path)
    filter_keys = set(keys) if keys is not None else None

    for k, v in parsed.items():
        if filter_keys is not None and k not in filter_keys:
            continue
        os.environ.setdefault(k, v)
