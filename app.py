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


def clear_folder(folder: Path):
    """Remove all files from a folder."""
    for f in folder.iterdir():
        if f.is_file():
            f.unlink()


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
    ["Home", "Photos Import", "Upload Photos", "Train Model", "Auto-Curate", "Faces", "Gallery"]
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
elif page == "Photos Import":
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
                    max_raw = st.number_input(
                        "Max raw photos to export",
                        min_value=0,
                        value=0,
                        help="0 = export all"
                    )
                    max_raw = None if max_raw == 0 else int(max_raw)

                    max_curated = st.number_input(
                        "Max favorited photos to export",
                        min_value=0,
                        value=0,
                        help="0 = export all favorites"
                    )
                    max_curated = None if max_curated == 0 else int(max_curated)

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

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    stats_container = st.empty()

                    try:
                        if export_both:
                            def progress_callback(phase, current, total, filename):
                                progress = int((current / total) * 100) if total > 0 else 0
                                progress_bar.progress(progress)
                                phase_label = "Raw photos" if phase == "raw" else "Favorites"
                                status_text.text(f"{phase_label}: {current}/{total} - {filename}")

                            with st.spinner("Exporting photos from Photos library..."):
                                raw_exp, raw_skip, cur_exp, cur_skip = importer.export_all(
                                    raw_dir=str(RAW_DIR),
                                    curated_dir=str(CURATED_DIR),
                                    max_size=max_size,
                                    quality=quality,
                                    max_raw=max_raw,
                                    max_curated=max_curated,
                                    progress_callback=progress_callback
                                )

                            progress_bar.progress(100)
                            status_text.text("Export complete!")

                            with stats_container.container():
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Raw Photos Exported", raw_exp, delta=f"{raw_skip} skipped")
                                with col2:
                                    st.metric("Favorites Exported", cur_exp, delta=f"{cur_skip} skipped")

                            st.success("Photos imported successfully!")

                        elif export_all:
                            def progress_callback(current, total, filename):
                                progress = int((current / total) * 100) if total > 0 else 0
                                progress_bar.progress(progress)
                                status_text.text(f"Exporting: {current}/{total} - {filename}")

                            with st.spinner("Exporting all photos..."):
                                exported, skipped = importer.export_photos(
                                    output_dir=str(RAW_DIR),
                                    favorites_only=False,
                                    max_photos=max_raw,
                                    max_size=max_size,
                                    quality=quality,
                                    progress_callback=progress_callback
                                )

                            progress_bar.progress(100)
                            status_text.text("Export complete!")
                            st.success(f"Exported {exported} photos ({skipped} skipped)")

                        elif export_favorites:
                            def progress_callback(current, total, filename):
                                progress = int((current / total) * 100) if total > 0 else 0
                                progress_bar.progress(progress)
                                status_text.text(f"Exporting: {current}/{total} - {filename}")

                            with st.spinner("Exporting favorite photos..."):
                                exported, skipped = importer.export_photos(
                                    output_dir=str(CURATED_DIR),
                                    favorites_only=True,
                                    max_photos=max_curated,
                                    max_size=max_size,
                                    quality=quality,
                                    progress_callback=progress_callback,
                                    download_missing=download_icloud
                                )

                            progress_bar.progress(100)
                            status_text.text("Export complete!")
                            st.success(f"Exported {exported} favorite photos ({skipped} skipped)")

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

        def update_progress(epoch, total_epochs, train_loss, val_loss, train_acc, val_acc):
            progress = int((epoch / total_epochs) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch}/{total_epochs}")
            with metrics_container.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train Loss", f"{train_loss:.4f}")
                with col2:
                    st.metric("Val Loss", f"{val_loss:.4f}")
                with col3:
                    st.metric("Train Acc", f"{train_acc*100:.1f}%")
                with col4:
                    st.metric("Val Acc", f"{val_acc*100:.1f}%")

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

        # Options
        col1, col2 = st.columns(2)
        with col1:
            clear_output = st.checkbox("Clear output folder first", value=True)
        with col2:
            clear_input_after = st.checkbox("Clear input folder after curation", value=False)

        if st.button("🎯 Run Auto-Curation", type="primary"):
            from predict import PhotoCurator

            with st.spinner("Analyzing photos..."):
                curator = PhotoCurator(
                    checkpoint_path=str(CHECKPOINTS_DIR / "best.pt"),
                    threshold=threshold,
                    top_n=top_n,
                    top_percent=top_percent
                )

                results = curator.predict_folder(str(INPUT_DIR))

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
