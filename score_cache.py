"""
Cache for ML model scores.
Stores scores in a JSON file to avoid re-scoring photos on every curation run.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List

CACHE_FILENAME = ".photo_scores.json"

# Module-level cache to avoid reloading JSON repeatedly
_loaded_caches: Dict[str, Dict[str, float]] = {}


def get_cache_path(folder: Path) -> Path:
    """Get the path to the score cache file for a folder."""
    return folder / CACHE_FILENAME


def load_cache(folder: Path) -> Dict[str, float]:
    """Load the score cache from disk."""
    cache_path = get_cache_path(folder)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(folder: Path, cache: Dict[str, float]):
    """Save the score cache to disk."""
    cache_path = get_cache_path(folder)
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass


def get_cached_scores(folder: Path, filenames: List[str]) -> Dict[str, Optional[float]]:
    """
    Get cached scores for a list of filenames.

    Args:
        folder: The folder containing the score cache
        filenames: List of image filenames (not full paths)

    Returns:
        Dict mapping filename to score (or None if not cached)
    """
    # Use module-level cache
    folder_key = str(folder)
    if folder_key not in _loaded_caches:
        _loaded_caches[folder_key] = load_cache(folder)
    cache = _loaded_caches[folder_key]

    results = {}
    for filename in filenames:
        results[filename] = cache.get(filename)

    return results


def update_scores(folder: Path, scores: Dict[str, float]):
    """
    Update the score cache with new scores.

    Args:
        folder: The folder containing the score cache
        scores: Dict mapping filename to score
    """
    folder_key = str(folder)
    if folder_key not in _loaded_caches:
        _loaded_caches[folder_key] = load_cache(folder)

    cache = _loaded_caches[folder_key]
    cache.update(scores)
    save_cache(folder, cache)


def invalidate_cache(folder: Path = None):
    """Invalidate the in-memory cache for a folder, or all folders if None."""
    global _loaded_caches
    if folder is None:
        _loaded_caches = {}
    else:
        folder_key = str(folder)
        if folder_key in _loaded_caches:
            del _loaded_caches[folder_key]


def clear_cache(folder: Path):
    """Delete the cache file and clear in-memory cache."""
    cache_path = get_cache_path(folder)
    if cache_path.exists():
        cache_path.unlink()
    invalidate_cache(folder)
