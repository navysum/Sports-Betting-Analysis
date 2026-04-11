"""
Team name normalisation — resolves football-data.org API names
to FDCO CSV names used by ELO and Dixon-Coles.
"""
import json
import os
from difflib import get_close_matches

_ALIASES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "team_aliases.json"
)
_aliases: dict[str, str] = {}


def _load():
    global _aliases
    if _aliases:
        return
    try:
        with open(_ALIASES_PATH) as f:
            raw = json.load(f)
        _aliases = {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception:
        _aliases = {}


def resolve(name: str) -> str:
    """Map an API team name to its FDCO equivalent. Returns original if no match."""
    _load()
    if name in _aliases:
        return _aliases[name]
    # Try fuzzy match as fallback
    candidates = list(_aliases.keys())
    close = get_close_matches(name, candidates, n=1, cutoff=0.75)
    if close:
        return _aliases[close[0]]
    return name
