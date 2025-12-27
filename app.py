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
    ["Home", "Upload Photos", "Train Model", "Auto-Curate", "Gallery"]
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

    col1, col2 = st.columns(2)

    with col1:
        backbone = st.selectbox(
            "Model Architecture",
            ["resnet50", "resnet18", "efficientnet_b0", "mobilenet_v3"],
            help="ResNet50 is recommended for best accuracy"
        )

        epochs = st.slider("Training Epochs", 10, 100, 30)

        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)

    with col2:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.00001, 0.0001, 0.001],
            value=0.0001,
            format_func=lambda x: f"{x:.5f}"
        )

        patience = st.slider("Early Stopping Patience", 3, 20, 10)

        freeze_backbone = st.checkbox(
            "Freeze Backbone",
            help="Train only the classifier head (faster, for small datasets)"
        )

    st.markdown("---")

    if st.button("🚀 Start Training", type="primary", disabled=(raw_count < 5 or curated_count < 3)):
        from train import train_model

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Initializing training...")

        try:
            with st.spinner("Training in progress... This may take a while."):
                history = train_model(
                    raw_folder=str(RAW_DIR),
                    curated_folder=str(CURATED_DIR),
                    output_dir=str(CHECKPOINTS_DIR),
                    backbone=backbone,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    freeze_backbone=freeze_backbone,
                    early_stopping_patience=patience
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
        st.markdown("### Curation Settings")

        threshold = st.slider(
            "Curation Threshold",
            0.0, 1.0, 0.5,
            help="Higher = more selective (fewer photos selected)"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if threshold < 0.4:
                st.info("📈 Permissive - More photos will be selected")
            elif threshold > 0.6:
                st.info("📉 Strict - Fewer photos will be selected")
            else:
                st.info("⚖️ Balanced - Moderate selection")

        st.markdown("---")

        if st.button("🎯 Run Auto-Curation", type="primary"):
            from predict import PhotoCurator

            with st.spinner("Analyzing photos..."):
                curator = PhotoCurator(
                    checkpoint_path=str(CHECKPOINTS_DIR / "best.pt"),
                    threshold=threshold
                )

                results = curator.predict_folder(str(INPUT_DIR))

                # Curate to output folder
                clear_folder(OUTPUT_DIR)
                num_curated, num_rejected = curator.curate_folder(
                    input_folder=str(INPUT_DIR),
                    output_folder=str(OUTPUT_DIR),
                    copy_files=True
                )

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
