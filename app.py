"""
FotoSelect - Streamlit Web Application

A web interface for training and using photo curation models.
"""

import streamlit as st
import os
import shutil
from pathlib import Path
from PIL import Image
import torch
from typing import List, Tuple
import json

# Page config
st.set_page_config(
    page_title="FotoSelect",
    page_icon="📷",
    layout="wide"
)

# Constants
PHOTOS_DIR = Path("photos")
RAW_DIR = PHOTOS_DIR / "raw"
CURATED_DIR = PHOTOS_DIR / "curated"
INPUT_DIR = PHOTOS_DIR / "input"
OUTPUT_DIR = PHOTOS_DIR / "output"
CHECKPOINTS_DIR = Path("checkpoints")

# Ensure directories exist
for d in [RAW_DIR, CURATED_DIR, INPUT_DIR, OUTPUT_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_image_files(folder: Path) -> List[Path]:
    """Get all image files in a folder."""
    extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp',
                  '.nef', '.cr2', '.cr3', '.arw', '.raf', '.orf', '.rw2', '.dng'}
    files = []
    for f in folder.iterdir():
        if f.suffix.lower() in extensions:
            files.append(f)
    return sorted(files)


def save_uploaded_files(uploaded_files, destination: Path) -> int:
    """Save uploaded files to destination folder."""
    count = 0
    for uploaded_file in uploaded_files:
        dest_path = destination / uploaded_file.name
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        count += 1
    return count


def clear_folder(folder: Path, clear_staging: bool = True):
    """Remove all files from a folder and optionally clear staging directory."""
    import shutil

    for f in folder.iterdir():
        if f.is_file():
            f.unlink()

    # Also clear the staging directory used by osxphotos export
    # This removes the .osxphotos_export.db database so exports start fresh
    if clear_staging:
        staging_dir = folder / ".staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

        # Also clear the tracking file
        tracking_file = folder / ".downsampled.json"
        if tracking_file.exists():
            tracking_file.unlink()


def display_image_grid(images: List[Path], cols: int = 4, max_images: int = 20):
    """Display images in a grid."""
    images = images[:max_images]
    rows = (len(images) + cols - 1) // cols

    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(images):
                with columns[col]:
                    try:
                        img = Image.open(images[idx])
                        st.image(img, caption=images[idx].name, use_container_width=True)
                    except Exception as e:
                        st.error(f"Cannot load {images[idx].name}")


# Sidebar navigation
st.sidebar.title("📷 FotoSelect")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Import Photos", "Upload Photos", "Train Model", "Auto-Curate", "Faces", "Gallery"]
)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "Home":
    st.title("📷 FotoSelect")
    st.markdown("### AI-Powered Photo Curation")

    st.markdown("""
    FotoSelect learns your photo curation preferences and automatically selects your best shots.

    **How it works:**
    1. **Upload** your raw photos and the ones you've selected as "keepers"
    2. **Train** the AI model on your preferences
    3. **Auto-curate** new batches of photos instantly

    ---
    """)

    # Show current stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        raw_count = len(get_image_files(RAW_DIR))
        st.metric("Raw Photos", raw_count)

    with col2:
        curated_count = len(get_image_files(CURATED_DIR))
        st.metric("Curated Photos", curated_count)

    with col3:
        input_count = len(get_image_files(INPUT_DIR))
        st.metric("Pending Curation", input_count)

    with col4:
        has_model = (CHECKPOINTS_DIR / "best.pt").exists()
        st.metric("Model Status", "✅ Ready" if has_model else "❌ Not trained")

# ============================================================================
# PHOTOS IMPORT PAGE
# ============================================================================
elif page == "Import Photos":
    st.title("📸 Photos Library Import")

    st.markdown("""
    Import photos directly from your macOS Photos library.
    - **All photos** will be exported to the Raw folder
    - **Favorited photos** will be exported to the Curated folder
    - Photos are automatically downsampled for efficient training
    """)

    st.markdown("---")

    # Check if osxphotos is available
    try:
        from photos_import import PhotosLibraryImporter, is_available

        if not is_available():
            st.error("osxphotos is not installed. Run: `pip install osxphotos`")
        else:
            # Initialize importer in session state
            if 'photos_importer' not in st.session_state:
                try:
                    with st.spinner("Loading Photos library..."):
                        st.session_state.photos_importer = PhotosLibraryImporter()
                except Exception as e:
                    st.error(f"Failed to load Photos library: {e}")
                    st.session_state.photos_importer = None

            importer = st.session_state.photos_importer

            if importer:
                # Show library info
                info = importer.get_library_info()

                # Calculate totals
                available_photos = info["local_photos"] + info["cached_photos"]
                available_favorites = info["local_favorites"] + info["cached_favorites"]
                available_pct = (available_photos / info["total_photos"] * 100) if info["total_photos"] > 0 else 0

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Photos", info["total_photos"])
                with col2:
                    st.metric("Available", available_photos, delta=f"{available_pct:.0f}%")
                with col3:
                    st.metric("Favorites", info["favorites"])
                with col4:
                    st.metric("Albums", info["albums"])

                st.caption(f"Library: {info['library_path']}")

                # Show details about photo availability
                with st.expander("Photo availability details"):
                    st.markdown(f"""
                    **Photos ready to export:**
                    - 📁 Full resolution (local): {info['local_photos']}
                    - 🖼️ Cached thumbnails (iCloud): {info['cached_photos']}
                    - **Total available: {available_photos}**

                    **Favorites:**
                    - 📁 Full resolution: {info['local_favorites']}
                    - 🖼️ Cached thumbnails: {info['cached_favorites']}
                    - **Total available: {available_favorites}** / {info['favorites']}

                    ⚠️ **{info['icloud_only_photos']} photos have no local cache** and will be skipped by quick export.

                    ---
                    **To download ALL photos from iCloud**, run this in your terminal:
                    ```bash
                    cd {Path.cwd()}
                    python download_icloud.py --max-size 512
                    ```
                    This downloads photos from iCloud in batches and saves downsampled versions.
                    It can be interrupted and resumed (already-exported photos are skipped).
                    """)

                st.markdown("---")
                st.markdown("### Export Settings")

                col1, col2 = st.columns(2)

                with col1:
                    max_size = st.selectbox(
                        "Image Size (max dimension)",
                        [256, 512, 768, 1024],
                        index=1,
                        help="Larger = better quality but more storage"
                    )

                    quality = st.slider(
                        "JPEG Quality",
                        50, 100, 85,
                        help="Higher = better quality but larger files"
                    )

                with col2:
                    download_icloud = st.checkbox(
                        "Download from iCloud",
                        value=False,
                        help="Attempt to download photos stored in iCloud (slower)"
                    )

                st.markdown("---")
                st.markdown("### Photo Selection")

                selection_mode = st.radio(
                    "Selection Mode",
                    ["All Photos", "Fixed Number", "Percentage", "Date Range"],
                    horizontal=True,
                    help="Choose how to filter photos for export"
                )

                # Initialize filter variables
                max_raw = None
                max_curated = None
                percentage = None
                date_from = None
                date_to = None

                if selection_mode == "Fixed Number":
                    col1, col2 = st.columns(2)
                    with col1:
                        max_raw_input = st.number_input(
                            "Max raw photos to export",
                            min_value=1,
                            value=100,
                            help="Maximum number of photos to export"
                        )
                        max_raw = int(max_raw_input)
                    with col2:
                        max_curated_input = st.number_input(
                            "Max favorited photos to export",
                            min_value=1,
                            value=50,
                            help="Maximum number of favorites to export"
                        )
                        max_curated = int(max_curated_input)

                elif selection_mode == "Percentage":
                    percentage = st.slider(
                        "Percentage of photos to export",
                        1, 100, 25,
                        help="Randomly sample this percentage of photos"
                    )
                    st.info(f"Will randomly select ~{percentage}% of available photos")

                elif selection_mode == "Date Range":
                    from datetime import datetime, timedelta
                    col1, col2 = st.columns(2)
                    with col1:
                        date_from_input = st.date_input(
                            "From date",
                            value=datetime.now() - timedelta(days=365),
                            help="Export photos taken on or after this date"
                        )
                        date_from = datetime.combine(date_from_input, datetime.min.time())
                    with col2:
                        date_to_input = st.date_input(
                            "To date",
                            value=datetime.now(),
                            help="Export photos taken on or before this date"
                        )
                        date_to = datetime.combine(date_to_input, datetime.max.time())

                st.markdown("---")

                # Show current folder stats
                col1, col2 = st.columns(2)
                with col1:
                    raw_count = len(get_image_files(RAW_DIR))
                    st.metric("Current Raw Photos", raw_count)
                with col2:
                    curated_count = len(get_image_files(CURATED_DIR))
                    st.metric("Current Curated Photos", curated_count)

                clear_before = st.checkbox("Clear existing photos before export", value=False)

                # Tracking file management
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Sync Download Tracking", help="Sync tracking with existing files - next export will only download missing photos"):
                        from download_icloud import sync_tracking_with_folder
                        synced = []
                        if RAW_DIR.exists():
                            sync_tracking_with_folder(RAW_DIR)
                            synced.append("raw")
                        if CURATED_DIR.exists():
                            sync_tracking_with_folder(CURATED_DIR)
                            synced.append("curated")
                        if synced:
                            st.success(f"Synced tracking for: {', '.join(synced)}")
                        else:
                            st.info("No photo folders found")
                with col2:
                    if st.button("🗑️ Clear All & Start Over", type="secondary", help="Delete all exported photos and tracking data"):
                        cleared = []
                        if RAW_DIR.exists():
                            clear_folder(RAW_DIR)
                            cleared.append(f"raw ({raw_count} files)")
                        if CURATED_DIR.exists():
                            clear_folder(CURATED_DIR)
                            cleared.append(f"curated ({curated_count} files)")
                        if cleared:
                            st.success(f"Cleared: {', '.join(cleared)}")
                            st.rerun()
                        else:
                            st.info("No photo folders to clear")

                st.markdown("---")

                # Export buttons
                col1, col2, col3 = st.columns(3)

                with col1:
                    export_all = st.button("📥 Export All Photos", type="primary")

                with col2:
                    export_favorites = st.button("⭐ Export Favorites Only")

                with col3:
                    export_both = st.button("📥⭐ Export Both")

                if export_all or export_favorites or export_both:
                    if clear_before:
                        if export_all or export_both:
                            clear_folder(RAW_DIR)
                        if export_favorites or export_both:
                            clear_folder(CURATED_DIR)

                    # Create a prominent loading screen
                    loading_container = st.container()

                    with loading_container:
                        st.markdown("---")
                        st.markdown("## Importing Photos")

                        # Large status header
                        status_header = st.empty()
                        status_header.markdown("### Preparing to import...")

                        # Progress bar with percentage
                        progress_col1, progress_col2 = st.columns([4, 1])
                        with progress_col1:
                            progress_bar = st.progress(0)
                        with progress_col2:
                            progress_percent = st.empty()
                            progress_percent.markdown("**0%**")

                        # Current file status
                        status_text = st.empty()
                        status_text.info("Starting import process...")

                        # Live statistics
                        stats_cols = st.columns(4)
                        with stats_cols[0]:
                            exported_metric = st.empty()
                            exported_metric.metric("Exported", 0)
                        with stats_cols[1]:
                            skipped_metric = st.empty()
                            skipped_metric.metric("Skipped", 0)
                        with stats_cols[2]:
                            total_metric = st.empty()
                            total_metric.metric("Total", "...")
                        with stats_cols[3]:
                            phase_metric = st.empty()
                            phase_metric.metric("Phase", "Starting")

                        st.markdown("---")

                    # Track counts for live updates
                    export_counts = {"exported": 0, "skipped": 0}

                    try:
                        if export_both:
                            def progress_callback(phase, current, total, filename):
                                progress = int((current / total) * 100) if total > 0 else 0
                                progress_bar.progress(progress)
                                progress_percent.markdown(f"**{progress}%**")
                                phase_label = "Raw photos" if phase == "raw" else "Favorites"
                                status_header.markdown(f"### Importing {phase_label}...")
                                phase_metric.metric("Phase", phase_label)
                                total_metric.metric("Total", total)

                                # Update counts based on filename
                                if "Skipped" in filename:
                                    export_counts["skipped"] += 1
                                    skipped_metric.metric("Skipped", export_counts["skipped"])
                                else:
                                    export_counts["exported"] += 1
                                    exported_metric.metric("Exported", export_counts["exported"])

                                status_text.info(f"Processing: **{filename}** ({current}/{total})")

                            raw_exp, raw_skip, cur_exp, cur_skip = importer.export_all(
                                raw_dir=str(RAW_DIR),
                                curated_dir=str(CURATED_DIR),
                                max_size=max_size,
                                quality=quality,
                                max_raw=max_raw,
                                max_curated=max_curated,
                                progress_callback=progress_callback,
                                download_missing=download_icloud,
                                percentage=percentage,
                                date_from=date_from,
                                date_to=date_to
                            )

                            progress_bar.progress(100)
                            progress_percent.markdown("**100%**")
                            status_header.markdown("### Import Complete!")
                            status_text.success("All photos have been imported successfully!")
                            phase_metric.metric("Phase", "Done")
                            exported_metric.metric("Exported", raw_exp + cur_exp)
                            skipped_metric.metric("Skipped", raw_skip + cur_skip)

                            st.balloons()
                            st.success(f"Successfully imported {raw_exp} raw photos and {cur_exp} favorites!")

                        elif export_all:
                            def progress_callback(current, total, filename):
                                progress = int((current / total) * 100) if total > 0 else 0
                                progress_bar.progress(progress)
                                progress_percent.markdown(f"**{progress}%**")
                                status_header.markdown("### Importing All Photos...")
                                phase_metric.metric("Phase", "All Photos")
                                total_metric.metric("Total", total)

                                if "Skipped" in filename:
                                    export_counts["skipped"] += 1
                                    skipped_metric.metric("Skipped", export_counts["skipped"])
                                else:
                                    export_counts["exported"] += 1
                                    exported_metric.metric("Exported", export_counts["exported"])

                                status_text.info(f"Processing: **{filename}** ({current}/{total})")

                            exported, skipped = importer.export_photos(
                                output_dir=str(RAW_DIR),
                                favorites_only=False,
                                max_photos=max_raw,
                                max_size=max_size,
                                quality=quality,
                                progress_callback=progress_callback,
                                download_missing=download_icloud,
                                percentage=percentage,
                                date_from=date_from,
                                date_to=date_to
                            )

                            progress_bar.progress(100)
                            progress_percent.markdown("**100%**")
                            status_header.markdown("### Import Complete!")
                            status_text.success("All photos have been imported successfully!")
                            phase_metric.metric("Phase", "Done")
                            exported_metric.metric("Exported", exported)
                            skipped_metric.metric("Skipped", skipped)

                            st.balloons()
                            st.success(f"Successfully imported {exported} photos ({skipped} skipped)")

                        elif export_favorites:
                            def progress_callback(current, total, filename):
                                progress = int((current / total) * 100) if total > 0 else 0
                                progress_bar.progress(progress)
                                progress_percent.markdown(f"**{progress}%**")
                                status_header.markdown("### Importing Favorites...")
                                phase_metric.metric("Phase", "Favorites")
                                total_metric.metric("Total", total)

                                if "Skipped" in filename:
                                    export_counts["skipped"] += 1
                                    skipped_metric.metric("Skipped", export_counts["skipped"])
                                else:
                                    export_counts["exported"] += 1
                                    exported_metric.metric("Exported", export_counts["exported"])

                                status_text.info(f"Processing: **{filename}** ({current}/{total})")

                            exported, skipped = importer.export_photos(
                                output_dir=str(CURATED_DIR),
                                favorites_only=True,
                                max_photos=max_curated,
                                max_size=max_size,
                                quality=quality,
                                progress_callback=progress_callback,
                                download_missing=download_icloud,
                                percentage=percentage,
                                date_from=date_from,
                                date_to=date_to
                            )

                            progress_bar.progress(100)
                            progress_percent.markdown("**100%**")
                            status_header.markdown("### Import Complete!")
                            status_text.success("All favorite photos have been imported successfully!")
                            phase_metric.metric("Phase", "Done")
                            exported_metric.metric("Exported", exported)
                            skipped_metric.metric("Skipped", skipped)

                            st.balloons()
                            st.success(f"Successfully imported {exported} favorite photos ({skipped} skipped)")

                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")

    except ImportError:
        st.error(
            "The photos_import module is not available.\n\n"
            "Install osxphotos with: `pip install osxphotos`"
        )

# ============================================================================
# UPLOAD PAGE
# ============================================================================
elif page == "Upload Photos":
    st.title("📤 Upload Photos")

    tab1, tab2 = st.tabs(["Training Data", "Photos to Curate"])

    with tab1:
        st.markdown("### Upload Training Data")
        st.markdown("Upload your photos to train the model on your preferences.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Raw Photos (All)")
            st.caption("Upload all your original photos")
            raw_files = st.file_uploader(
                "Choose raw photos",
                type=['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'],
                accept_multiple_files=True,
                key="raw_upload"
            )
            if raw_files:
                if st.button("Upload Raw Photos", key="btn_raw"):
                    count = save_uploaded_files(raw_files, RAW_DIR)
                    st.success(f"Uploaded {count} photos to raw folder")
                    st.rerun()

            raw_count = len(get_image_files(RAW_DIR))
            st.info(f"Currently {raw_count} photos in raw folder")

            if raw_count > 0 and st.button("Clear Raw Folder", key="clear_raw"):
                clear_folder(RAW_DIR)
                st.rerun()

        with col2:
            st.markdown("#### Curated Photos (Selected)")
            st.caption("Upload only the photos you've selected as good")
            curated_files = st.file_uploader(
                "Choose curated photos",
                type=['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'],
                accept_multiple_files=True,
                key="curated_upload"
            )
            if curated_files:
                if st.button("Upload Curated Photos", key="btn_curated"):
                    count = save_uploaded_files(curated_files, CURATED_DIR)
                    st.success(f"Uploaded {count} photos to curated folder")
                    st.rerun()

            curated_count = len(get_image_files(CURATED_DIR))
            st.info(f"Currently {curated_count} photos in curated folder")

            if curated_count > 0 and st.button("Clear Curated Folder", key="clear_curated"):
                clear_folder(CURATED_DIR)
                st.rerun()

    with tab2:
        st.markdown("### Upload Photos to Auto-Curate")
        st.caption("Upload new photos that you want the AI to curate")

        input_files = st.file_uploader(
            "Choose photos to curate",
            type=['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'],
            accept_multiple_files=True,
            key="input_upload"
        )
        if input_files:
            if st.button("Upload for Curation", key="btn_input"):
                count = save_uploaded_files(input_files, INPUT_DIR)
                st.success(f"Uploaded {count} photos for curation")
                st.rerun()

        input_count = len(get_image_files(INPUT_DIR))
        st.info(f"Currently {input_count} photos pending curation")

        if input_count > 0 and st.button("Clear Input Folder", key="clear_input"):
            clear_folder(INPUT_DIR)
            st.rerun()

# ============================================================================
# TRAIN PAGE
# ============================================================================
elif page == "Train Model":
    st.title("🎯 Train Model")

    raw_count = len(get_image_files(RAW_DIR))
    curated_count = len(get_image_files(CURATED_DIR))

    st.markdown(f"""
    **Training Data Status:**
    - Raw photos: {raw_count}
    - Curated photos: {curated_count}
    """)

    if raw_count < 10 or curated_count < 5:
        st.warning("Please upload more photos. Recommended: at least 50 raw and 20 curated photos.")

    st.markdown("---")
    st.markdown("### Training Settings")

    # Training presets
    training_preset = st.radio(
        "Training Duration",
        ["Quick (10 epochs)", "Standard (30 epochs)", "Extended (100 epochs)", "Custom"],
        horizontal=True,
        index=1
    )

    preset_epochs = {"Quick (10 epochs)": 10, "Standard (30 epochs)": 30, "Extended (100 epochs)": 100}

    col1, col2 = st.columns(2)

    with col1:
        backbone = st.selectbox(
            "Model Architecture",
            ["resnet50", "resnet18", "efficientnet_b0", "mobilenet_v3"],
            help="ResNet50 is recommended for best accuracy"
        )

        if training_preset == "Custom":
            epochs = st.number_input("Training Epochs", min_value=1, max_value=500, value=50)
        else:
            epochs = preset_epochs[training_preset]
            st.info(f"Training for {epochs} epochs")

        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)

    with col2:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.00001, 0.0001, 0.001],
            value=0.0001,
            format_func=lambda x: f"{x:.5f}"
        )

        patience = st.slider("Early Stopping Patience", 3, 50, 10,
            help="Stop training if no improvement for this many epochs")

        freeze_backbone = st.checkbox(
            "Freeze Backbone",
            help="Train only the classifier head (faster, for small datasets)"
        )

    st.markdown("---")

    if st.button("🚀 Start Training", type="primary", disabled=(raw_count < 5 or curated_count < 3)):
        from train import train_model

        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()

        status_text.text("Initializing training...")

        # Track metrics across callbacks
        current_metrics = {'train_loss': 0, 'val_loss': 0, 'train_acc': 0, 'val_acc': 0}

        def update_progress(epoch, total_epochs, batch=None, total_batches=None,
                          phase='train', train_loss=None, val_loss=None,
                          train_acc=None, val_acc=None, batch_loss=None, batch_acc=None):
            # Calculate precise progress
            if phase == 'epoch_end':
                # End of epoch - update metrics
                current_metrics['train_loss'] = train_loss
                current_metrics['val_loss'] = val_loss
                current_metrics['train_acc'] = train_acc
                current_metrics['val_acc'] = val_acc
                progress = int((epoch / total_epochs) * 100)
                status_text.text(f"Epoch {epoch}/{total_epochs} complete")
            elif batch is not None and total_batches is not None:
                # Batch-level progress: each epoch has train + val phases
                # Train phase = first half of epoch, val phase = second half
                epoch_progress = (epoch - 1) / total_epochs
                if phase == 'train':
                    batch_progress = (batch / total_batches) * 0.8  # Training is ~80% of epoch
                    phase_text = "Training"
                    current_metrics['train_loss'] = batch_loss or current_metrics['train_loss']
                    current_metrics['train_acc'] = batch_acc or current_metrics['train_acc']
                else:  # val
                    batch_progress = 0.8 + (batch / total_batches) * 0.2  # Validation is ~20%
                    phase_text = "Validating"

                progress = int((epoch_progress + batch_progress / total_epochs) * 100)
                status_text.text(f"Epoch {epoch}/{total_epochs} - {phase_text} batch {batch}/{total_batches}")
            else:
                progress = 0

            progress_bar.progress(min(progress, 100))

            # Update metrics display
            with metrics_container.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train Loss", f"{current_metrics['train_loss']:.4f}")
                with col2:
                    st.metric("Val Loss", f"{current_metrics['val_loss']:.4f}")
                with col3:
                    st.metric("Train Acc", f"{current_metrics['train_acc']*100:.1f}%")
                with col4:
                    st.metric("Val Acc", f"{current_metrics['val_acc']*100:.1f}%")

        try:
            history = train_model(
                raw_folder=str(RAW_DIR),
                curated_folder=str(CURATED_DIR),
                output_dir=str(CHECKPOINTS_DIR),
                backbone=backbone,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                freeze_backbone=freeze_backbone,
                early_stopping_patience=patience,
                progress_callback=update_progress
            )

            progress_bar.progress(100)
            status_text.text("Training complete!")

            st.success("Model trained successfully!")

            # Show training results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Validation Accuracy", f"{max(history['val_acc'])*100:.1f}%")
            with col2:
                st.metric("Final Training Accuracy", f"{history['train_acc'][-1]*100:.1f}%")

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            raise e

    # Model Management Section
    st.markdown("---")
    st.markdown("### Model Management")

    # List existing saved models
    saved_models = list(CHECKPOINTS_DIR.glob("*.pt"))
    saved_models = [m for m in saved_models if m.name not in ["best.pt", "last.pt"]]

    col1, col2 = st.columns(2)

    with col1:
        # Save current model with custom name
        model_name = st.text_input(
            "Save model as",
            placeholder="my_model",
            help="Enter a name for your model (without .pt extension)"
        )

        if st.button("💾 Save Model", disabled=not has_model or not model_name):
            import shutil
            source = CHECKPOINTS_DIR / "best.pt"
            # Sanitize filename
            safe_name = "".join(c for c in model_name if c.isalnum() or c in "-_").strip()
            if safe_name:
                dest = CHECKPOINTS_DIR / f"{safe_name}.pt"
                shutil.copy2(source, dest)
                st.success(f"Model saved as '{safe_name}.pt'")
                st.rerun()
            else:
                st.error("Please enter a valid model name")

    with col2:
        if saved_models:
            st.markdown("**Saved Models:**")
            for model_path in sorted(saved_models):
                model_col1, model_col2 = st.columns([3, 1])
                with model_col1:
                    st.text(model_path.name)
                with model_col2:
                    if st.button("Load", key=f"load_{model_path.name}"):
                        import shutil
                        shutil.copy2(model_path, CHECKPOINTS_DIR / "best.pt")
                        st.success(f"Loaded '{model_path.name}' as active model")
                        st.rerun()
        else:
            st.info("No saved models yet. Train a model and save it with a custom name.")

# ============================================================================
# PREDICT PAGE
# ============================================================================
elif page == "Auto-Curate":
    st.title("🤖 Auto-Curate Photos")

    has_model = (CHECKPOINTS_DIR / "best.pt").exists()
    input_count = len(get_image_files(INPUT_DIR))

    if not has_model:
        st.error("No trained model found. Please train a model first.")
    elif input_count == 0:
        st.warning("No photos to curate. Please upload photos in the Upload section.")
    else:
        st.success(f"Model ready. {input_count} photos pending curation.")

        st.markdown("---")
        st.markdown("### Selection Mode")

        selection_mode = st.radio(
            "How do you want to select photos?",
            ["Fixed Number", "Percentage", "Score Threshold"],
            horizontal=True
        )

        # Selection parameters based on mode
        top_n = None
        top_percent = None
        threshold = 0.5

        if selection_mode == "Fixed Number":
            top_n = st.number_input(
                "Number of photos to select",
                min_value=1,
                max_value=input_count,
                value=min(10, input_count),
                help="Select exactly this many top-scoring photos"
            )
            st.info(f"Will select the top {top_n} photos out of {input_count}")

        elif selection_mode == "Percentage":
            top_percent = st.slider(
                "Percentage of photos to select",
                1, 100, 25,
                help="Select the top X% of photos"
            )
            num_selected = max(1, int(input_count * top_percent / 100))
            st.info(f"Will select ~{num_selected} photos ({top_percent}% of {input_count})")

        else:  # Score Threshold
            threshold = st.slider(
                "Score Threshold",
                0.0, 1.0, 0.5,
                help="Select photos with score above this threshold"
            )
            if threshold < 0.4:
                st.info("📈 Permissive - More photos will be selected")
            elif threshold > 0.6:
                st.info("📉 Strict - Fewer photos will be selected")
            else:
                st.info("⚖️ Balanced - Moderate selection")

        st.markdown("---")
        st.markdown("### Deduplication")

        deduplicate = st.checkbox(
            "Remove similar/duplicate photos",
            value=True,
            help="Compare selected photos and replace duplicates with different ones"
        )

        if deduplicate:
            similarity_threshold = st.slider(
                "Similarity threshold",
                0.0, 1.0, 0.75,
                help="Higher = only remove very similar photos, Lower = more aggressive deduplication"
            )
        else:
            similarity_threshold = 0.75

        st.markdown("---")

        # Options
        col1, col2 = st.columns(2)
        with col1:
            clear_output = st.checkbox("Clear output folder first", value=True)
        with col2:
            clear_input_after = st.checkbox("Clear input folder after curation", value=False)

        if st.button("🎯 Run Auto-Curation", type="primary"):
            from predict import PhotoCurator

            progress_bar = st.progress(0)
            status_text = st.empty()

            curator = PhotoCurator(
                checkpoint_path=str(CHECKPOINTS_DIR / "best.pt"),
                threshold=threshold,
                top_n=top_n,
                top_percent=top_percent,
                deduplicate=deduplicate,
                similarity_threshold=similarity_threshold
            )

            def update_progress(current, total, phase):
                if phase == 'scoring':
                    # Scoring is 0-50% (or 0-100% if no dedup)
                    max_pct = 50 if deduplicate else 100
                    pct = int((current / total) * max_pct)
                    status_text.text(f"Scoring photos: {current}/{total} batches")
                elif phase == 'features':
                    # Feature extraction is 50-75%
                    pct = 50 + int((current / total) * 25)
                    status_text.text(f"Extracting features: {current}/{total} batches")
                elif phase == 'dedup':
                    # Deduplication is 75-100%
                    pct = 75 + int((current / total) * 25)
                    status_text.text(f"Deduplicating: {current}/{total} photos")
                else:
                    pct = 0
                progress_bar.progress(min(pct, 100))

            status_text.text("Loading model...")
            results = curator.predict_folder(str(INPUT_DIR), progress_callback=update_progress)

            progress_bar.progress(100)
            status_text.text("Organizing files...")

            # Clear output folder if requested
            if clear_output:
                clear_folder(OUTPUT_DIR)

            # Pass pre-computed results to avoid duplicate processing
            num_curated, num_rejected = curator.curate_folder(
                input_folder=str(INPUT_DIR),
                output_folder=str(OUTPUT_DIR),
                copy_files=True,
                results=results
            )

            # Clear input folder if requested
            if clear_input_after:
                clear_folder(INPUT_DIR)

            status_text.empty()
            st.success(f"Curation complete! {num_curated} photos selected, {num_rejected} rejected.")

            # Show results
            st.markdown("### Results")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Selected", num_curated, delta=f"{num_curated/(num_curated+num_rejected)*100:.0f}%")
            with col2:
                st.metric("Rejected", num_rejected)

            # Show top predictions
            st.markdown("### Top Rated Photos")
            top_photos = [(Path(p), s) for p, s, c in results if c][:8]

            if top_photos:
                cols = st.columns(4)
                for i, (photo_path, score) in enumerate(top_photos):
                    with cols[i % 4]:
                        try:
                            img = Image.open(photo_path)
                            st.image(img, caption=f"{photo_path.name}\nScore: {score:.2f}", use_container_width=True)
                        except:
                            pass

# ============================================================================
# FACES PAGE
# ============================================================================
elif page == "Faces":
    st.title("👤 Face Recognition")

    from faces import FaceManager

    FACE_DATA_DIR = Path("face_data")
    FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize face manager in session state
    if 'face_manager' not in st.session_state:
        with st.spinner("Loading face recognition models..."):
            st.session_state.face_manager = FaceManager(data_dir=str(FACE_DATA_DIR))

    fm = st.session_state.face_manager

    tab1, tab2, tab3, tab4 = st.tabs(["Detect Faces", "Cluster & Name", "Find People", "Identify"])

    # ---- TAB 1: Detect Faces ----
    with tab1:
        st.markdown("### Detect Faces in Photos")
        st.markdown("Scan your photos to detect and extract faces for clustering.")

        summary = fm.get_summary()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Faces Detected", summary["total_faces"])
        with col2:
            st.metric("Images Processed", summary["total_images"])
        with col3:
            st.metric("Clusters Found", summary["total_clusters"])

        st.markdown("---")

        folder_choice = st.selectbox(
            "Select folder to scan",
            ["Raw Photos", "Curated Photos", "Input", "Output"],
            key="face_scan_folder"
        )

        folder_map = {
            "Raw Photos": RAW_DIR,
            "Curated Photos": CURATED_DIR,
            "Input": INPUT_DIR,
            "Output": OUTPUT_DIR
        }
        scan_folder = folder_map[folder_choice]

        image_count = len(get_image_files(scan_folder))
        st.info(f"{image_count} images in {folder_choice}")

        col1, col2 = st.columns(2)
        with col1:
            clear_existing = st.checkbox("Clear existing face data", value=False)
        with col2:
            pass

        if st.button("🔍 Scan for Faces", type="primary", disabled=(image_count == 0)):
            with st.spinner("Detecting faces... This may take a while."):
                face_count = fm.process_folder(str(scan_folder), clear_existing=clear_existing)
            st.success(f"Detected {face_count} faces!")
            st.rerun()

    # ---- TAB 2: Cluster & Name ----
    with tab2:
        st.markdown("### Cluster Faces & Assign Names")
        st.markdown("Group similar faces together and assign names to identify people.")

        summary = fm.get_summary()

        if summary["total_faces"] == 0:
            st.warning("No faces detected yet. Please scan photos first.")
        else:
            st.markdown("#### Clustering Settings")

            col1, col2 = st.columns(2)
            with col1:
                eps = st.slider(
                    "Clustering Sensitivity",
                    0.3, 0.8, 0.5,
                    help="Lower = stricter matching (more clusters), Higher = looser matching (fewer clusters)"
                )
            with col2:
                min_samples = st.slider(
                    "Minimum Faces per Cluster",
                    1, 10, 2,
                    help="Minimum number of face occurrences to form a cluster"
                )

            if st.button("🔄 Run Clustering"):
                with st.spinner("Clustering faces..."):
                    clusters = fm.cluster_faces(eps=eps, min_samples=min_samples)
                st.success(f"Found {len([c for c in clusters.keys() if c >= 0])} clusters!")
                st.rerun()

            st.markdown("---")
            st.markdown("#### Name Clusters")

            if fm.clusters:
                # Get clusters sorted by size
                cluster_sizes = [(cid, len(faces)) for cid, faces in fm.clusters.items() if cid >= 0]
                cluster_sizes.sort(key=lambda x: -x[1])

                for cluster_id, size in cluster_sizes[:20]:  # Show top 20 clusters
                    with st.expander(f"Cluster {cluster_id} ({size} faces) - {fm.get_cluster_name(cluster_id) or 'Unnamed'}"):
                        # Show sample faces
                        sample_images = fm.get_cluster_sample_images(cluster_id, max_samples=5)

                        if sample_images:
                            cols = st.columns(min(5, len(sample_images)))
                            for i, face_img in enumerate(sample_images):
                                with cols[i]:
                                    st.image(face_img, use_container_width=True)

                        # Name input
                        current_name = fm.get_cluster_name(cluster_id) or ""
                        new_name = st.text_input(
                            "Name this person",
                            value=current_name,
                            key=f"name_cluster_{cluster_id}"
                        )

                        if new_name != current_name:
                            if st.button(f"Save Name", key=f"save_name_{cluster_id}"):
                                fm.set_cluster_name(cluster_id, new_name)
                                st.success(f"Saved: {new_name}")
                                st.rerun()
            else:
                st.info("No clusters yet. Run clustering first.")

    # ---- TAB 3: Find People ----
    with tab3:
        st.markdown("### Find Photos by Person")

        people = fm.get_all_people()

        if not people:
            st.warning("No named people yet. Please name some clusters first.")
        else:
            person_options = [f"{name} ({count} photos)" for name, count in people]
            selected = st.selectbox("Select a person", person_options)

            if selected:
                person_name = selected.split(" (")[0]
                photos = fm.get_photos_by_person(person_name)

                st.markdown(f"### Photos of {person_name}")
                st.info(f"Found {len(photos)} photos")

                if photos:
                    max_display = st.slider("Max photos to display", 4, 50, 12, key="person_photos_max")
                    cols_count = st.slider("Columns", 2, 6, 4, key="person_photos_cols")

                    photos_to_show = photos[:max_display]
                    rows = (len(photos_to_show) + cols_count - 1) // cols_count

                    for row in range(rows):
                        cols = st.columns(cols_count)
                        for col in range(cols_count):
                            idx = row * cols_count + col
                            if idx < len(photos_to_show):
                                with cols[col]:
                                    try:
                                        img = Image.open(photos_to_show[idx])
                                        st.image(img, caption=Path(photos_to_show[idx]).name, use_container_width=True)
                                    except:
                                        pass

    # ---- TAB 4: Identify ----
    with tab4:
        st.markdown("### Identify People in Photos")
        st.markdown("Upload a photo to identify recognized people in it.")

        if not fm.cluster_names:
            st.warning("No named people yet. Please name some clusters first.")
        else:
            uploaded_file = st.file_uploader(
                "Upload a photo",
                type=['jpg', 'jpeg', 'png'],
                key="identify_upload"
            )

            threshold = st.slider(
                "Recognition Threshold",
                0.3, 0.8, 0.6,
                help="Lower = more matches but potentially more false positives"
            )

            if uploaded_file:
                # Save temporarily
                temp_path = FACE_DATA_DIR / "temp_identify.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Show original image
                img = Image.open(temp_path)
                st.image(img, caption="Uploaded Photo", use_container_width=True)

                if st.button("🔍 Identify Faces"):
                    with st.spinner("Identifying faces..."):
                        results = fm.identify_faces(str(temp_path), threshold=threshold)

                    if results:
                        st.markdown("### Identified People")
                        for bbox, name, confidence in results:
                            if name != "Unknown":
                                st.success(f"**{name}** (confidence: {confidence:.1%})")
                            else:
                                st.warning("Unknown person detected")
                    else:
                        st.info("No faces detected in this image.")

# ============================================================================
# GALLERY PAGE
# ============================================================================
elif page == "Gallery":
    st.title("🖼️ Photo Gallery")

    gallery_choice = st.selectbox(
        "View folder",
        ["Raw Photos", "Curated Photos", "Input (Pending)", "Output (Auto-Curated)"]
    )

    folder_map = {
        "Raw Photos": RAW_DIR,
        "Curated Photos": CURATED_DIR,
        "Input (Pending)": INPUT_DIR,
        "Output (Auto-Curated)": OUTPUT_DIR
    }

    folder = folder_map[gallery_choice]
    images = get_image_files(folder)

    st.markdown(f"**{len(images)} photos**")

    if images:
        max_display = st.slider("Max photos to display", 4, 50, 20)
        cols = st.slider("Columns", 2, 6, 4)
        display_image_grid(images, cols=cols, max_images=max_display)
    else:
        st.info("No photos in this folder yet.")
