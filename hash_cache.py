"""
Cache for perceptual hashes.
Stores hashes in a JSON file to avoid re-computing hashes on every deduplication run.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List

CACHE_FILENAME = ".photo_hashes.json"

# Module-level cache to avoid reloading JSON repeatedly
_loaded_caches: Dict[str, Dict[str, str]] = {}


def get_cache_path(folder: Path) -> Path:
    """Get the path to the hash cache file for a folder."""
    return folder / CACHE_FILENAME


def load_cache(folder: Path) -> Dict[str, str]:
    """Load the hash cache from disk."""
    cache_path = get_cache_path(folder)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(folder: Path, cache: Dict[str, str]):
    """Save the hash cache to disk."""
    cache_path = get_cache_path(folder)
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass


def get_cached_hashes(folder: Path, filenames: List[str]) -> Dict[str, Optional[str]]:
    """
    Get cached hashes for a list of filenames.

    Args:
        folder: The folder containing the hash cache
        filenames: List of image filenames (not full paths)

    Returns:
        Dict mapping filename to hash string (or None if not cached)
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


def update_hashes(folder: Path, hashes: Dict[str, str]):
    """
    Update the hash cache with new hashes.

    Args:
        folder: The folder containing the hash cache
        hashes: Dict mapping filename to hash string
    """
    folder_key = str(folder)
    if folder_key not in _loaded_caches:
        _loaded_caches[folder_key] = load_cache(folder)

    cache = _loaded_caches[folder_key]
    cache.update(hashes)
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
