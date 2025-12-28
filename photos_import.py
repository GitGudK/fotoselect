"""
Local Photos library import module for FotoSelect.

Uses osxphotos to access the macOS Photos library directly.
This is more reliable than iCloud API access.
"""

import io
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from datetime import datetime

from PIL import Image

# Register HEIC support with PIL
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False

try:
    import osxphotos
    OSXPHOTOS_AVAILABLE = True
except ImportError:
    OSXPHOTOS_AVAILABLE = False


class PhotosLibraryImporter:
    """Imports photos from macOS Photos library."""

    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize the Photos library importer.

        Args:
            library_path: Path to Photos library. If None, uses system default.
        """
        if not OSXPHOTOS_AVAILABLE:
            raise ImportError(
                "osxphotos is not installed. Install with: pip install osxphotos"
            )

        self.photosdb = osxphotos.PhotosDB(dbfile=library_path)
        self._photos_cache = None
        self._favorites_cache = None

    def _get_all_photos(self) -> List:
        """Get all photos (images only, no movies)."""
        if self._photos_cache is None:
            all_items = self.photosdb.photos(images=True, movies=False)
            self._photos_cache = all_items
        return self._photos_cache

    def _get_favorites(self) -> List:
        """Get favorited photos only."""
        if self._favorites_cache is None:
            all_photos = self._get_all_photos()
            self._favorites_cache = [p for p in all_photos if p.favorite]
        return self._favorites_cache

    def get_photo_count(self) -> int:
        """Get total number of photos in library."""
        return len(self._get_all_photos())

    def get_favorites_count(self) -> int:
        """Get number of favorited photos."""
        return len(self._get_favorites())

    def get_albums(self) -> List[str]:
        """Get list of album names."""
        return [album.title for album in self.photosdb.album_info]

    def get_library_info(self) -> dict:
        """Get library information."""
        all_photos = self._get_all_photos()
        local_count = sum(1 for p in all_photos if p.path is not None)
        icloud_only_count = len(all_photos) - local_count

        # Count photos with cached derivatives (thumbnails)
        cached_count = sum(
            1 for p in all_photos
            if p.path is None and p.path_derivatives and
            any(Path(d).exists() for d in p.path_derivatives)
        )

        favorites = self._get_favorites()
        local_favorites = sum(1 for p in favorites if p.path is not None)
        icloud_only_favorites = len(favorites) - local_favorites
        cached_favorites = sum(
            1 for p in favorites
            if p.path is None and p.path_derivatives and
            any(Path(d).exists() for d in p.path_derivatives)
        )

        return {
            "total_photos": len(all_photos),
            "local_photos": local_count,
            "cached_photos": cached_count,  # iCloud photos with local thumbnails
            "icloud_only_photos": icloud_only_count - cached_count,  # Truly unavailable
            "favorites": len(favorites),
            "local_favorites": local_favorites,
            "cached_favorites": cached_favorites,
            "icloud_only_favorites": icloud_only_favorites - cached_favorites,
            "albums": len(self.photosdb.album_info),
            "library_path": str(self.photosdb.library_path),
        }

    def _downsample_image(
        self,
        image_path: str,
        max_size: int = 512,
        quality: int = 85
    ) -> bytes:
        """
        Downsample an image to a maximum dimension while preserving aspect ratio.

        Args:
            image_path: Path to the image file
            max_size: Maximum dimension (width or height)
            quality: JPEG quality (1-100)

        Returns:
            Downsampled image as JPEG bytes
        """
        img = Image.open(image_path)

        # Convert to RGB if necessary
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        # Calculate new size preserving aspect ratio
        width, height = img.size
        if width > height:
            if width > max_size:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_width, new_height = width, height
        else:
            if height > max_size:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width, new_height = width, height

        # Resize using high-quality resampling
        if (new_width, new_height) != (width, height):
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save to bytes
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()

    def _get_photo_filename(self, photo) -> str:
        """Generate a unique filename for a photo using UUID."""
        # Always use UUID to ensure uniqueness (many photos have same original filename)
        # Use first 8 chars of UUID which is enough for uniqueness
        return f"{photo.uuid[:8]}.jpg"

    def export_photos(
        self,
        output_dir: str,
        favorites_only: bool = False,
        album_name: Optional[str] = None,
        max_photos: Optional[int] = None,
        max_size: int = 512,
        quality: int = 85,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        skip_existing: bool = True,
        download_missing: bool = True
    ) -> Tuple[int, int]:
        """
        Export photos from Photos library with downsampling.

        Args:
            output_dir: Directory to save photos
            favorites_only: Only export favorited photos
            album_name: Export from specific album (None for all photos)
            max_photos: Maximum number of photos to export
            max_size: Maximum image dimension
            quality: JPEG quality
            progress_callback: Callback function(current, total, filename)
            skip_existing: Skip photos that already exist in output_dir
            download_missing: If True, download photos from iCloud if not available locally

        Returns:
            Tuple of (exported_count, skipped_count)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get photos
        if album_name:
            albums = [a for a in self.photosdb.album_info if a.title == album_name]
            if not albums:
                raise ValueError(f"Album '{album_name}' not found")
            # Filter album photos to images only
            photos = [p for p in albums[0].photos if not p.ismovie]
        elif favorites_only:
            photos = self._get_favorites()
        else:
            photos = self._get_all_photos()

        total = len(photos)
        if max_photos:
            photos = photos[:max_photos]
            total = len(photos)

        exported = 0
        skipped = 0
        icloud_skipped = 0

        for i, photo in enumerate(photos):
            filename = self._get_photo_filename(photo)
            filepath = output_path / filename

            # Skip if exists
            if skip_existing and filepath.exists():
                skipped += 1
                if progress_callback:
                    progress_callback(i + 1, total, f"Skipped (exists): {filename}")
                continue

            try:
                # Get the photo path
                photo_path = photo.path

                if photo_path is None:
                    # Photo is in iCloud - try to use locally cached derivatives
                    derivatives = photo.path_derivatives
                    if derivatives:
                        # Use the largest available derivative (usually last in list)
                        # Filter to only existing files
                        existing_derivatives = [d for d in derivatives if Path(d).exists()]
                        if existing_derivatives:
                            # Sort by file size to get the largest/best quality
                            existing_derivatives.sort(key=lambda x: Path(x).stat().st_size, reverse=True)
                            photo_path = existing_derivatives[0]
                        else:
                            icloud_skipped += 1
                            if progress_callback:
                                progress_callback(i + 1, total, f"Skipped (no cache): {filename}")
                            continue
                    else:
                        icloud_skipped += 1
                        if progress_callback:
                            progress_callback(i + 1, total, f"Skipped (iCloud): {filename}")
                        continue

                # Downsample and save (for locally available photos)
                downsampled = self._downsample_image(photo_path, max_size, quality)

                with open(filepath, 'wb') as f:
                    f.write(downsampled)

                exported += 1

                if progress_callback:
                    progress_callback(i + 1, total, filename)

            except Exception as e:
                print(f"Error exporting {filename}: {e}")
                skipped += 1
                continue

        # Return total skipped (includes both existing and iCloud)
        return exported, skipped + icloud_skipped

    def export_all(
        self,
        raw_dir: str,
        curated_dir: str,
        max_size: int = 512,
        quality: int = 85,
        max_raw: Optional[int] = None,
        max_curated: Optional[int] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        download_missing: bool = True
    ) -> Tuple[int, int, int, int]:
        """
        Export all photos to raw folder and favorites to curated folder.

        Args:
            raw_dir: Directory for all photos
            curated_dir: Directory for favorited photos
            max_size: Maximum image dimension
            quality: JPEG quality
            max_raw: Maximum raw photos to export
            max_curated: Maximum curated photos to export
            progress_callback: Callback function(phase, current, total, filename)
            download_missing: If True, download photos from iCloud if not available locally

        Returns:
            Tuple of (raw_exported, raw_skipped, curated_exported, curated_skipped)
        """
        def raw_callback(current, total, filename):
            if progress_callback:
                progress_callback("raw", current, total, filename)

        def curated_callback(current, total, filename):
            if progress_callback:
                progress_callback("curated", current, total, filename)

        # Export all photos to raw
        raw_exported, raw_skipped = self.export_photos(
            output_dir=raw_dir,
            favorites_only=False,
            max_photos=max_raw,
            max_size=max_size,
            quality=quality,
            progress_callback=raw_callback,
            download_missing=download_missing
        )

        # Export favorites to curated
        curated_exported, curated_skipped = self.export_photos(
            output_dir=curated_dir,
            favorites_only=True,
            max_photos=max_curated,
            max_size=max_size,
            quality=quality,
            progress_callback=curated_callback,
            download_missing=download_missing
        )

        return raw_exported, raw_skipped, curated_exported, curated_skipped


def is_available() -> bool:
    """Check if osxphotos is available."""
    return OSXPHOTOS_AVAILABLE
