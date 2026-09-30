import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from ImageLoader import UnifiedImageLoader
from new_utils import (
    ImageProcessor,
    DisplayTransform,
    apply_orientation,
    load_polygon_mask_2d,
    has_polygon_mask,
)
# Reuse the subprocess-backed prefetch cache from the draw step. Same slow
# call (pydicom decompresses the whole DBT volume for one slice) — so the
# same prefetch machinery applies unchanged. Module-level state is shared
# across steps because Python imports are cached, which is what we want.
from batch_draw_step import _prefetch, _load_slice_prefetched

# How many upcoming pending files to warm in the background after loading
# the current one. Each subprocess decode peaks ~1-2 GB RAM transiently,
# so 2 is a conservative "few" that won't stack up on modest machines.
_THRESHOLD_PREFETCH_AHEAD = 2


def _load_orientation(output_path: str, file_name: str) -> dict:
    """
    Orientation lookup order:
      1. Session state (set during the current run by the draw step)
      2. orientation.json written by the draw step on save
      3. Identity (rotate_k=0, no flips)
    """
    session = st.session_state.get("batch_orientations", {})
    orient = session.get(file_name)
    if orient is not None:
        return {
            "rotate_k": int(orient.get("rotate_k", 0)) % 4,
            "flip_h": bool(orient.get("flip_h", False)),
            "flip_v": bool(orient.get("flip_v", False)),
        }
    path = os.path.join(output_path, "orientation.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return {
                "rotate_k": int(data.get("rotate_k", 0)) % 4,
                "flip_h": bool(data.get("flip_h", False)),
                "flip_v": bool(data.get("flip_v", False)),
            }
        except Exception:
            pass
    return {"rotate_k": 0, "flip_h": False, "flip_v": False}


def _ensure_2d(arr):
    """
    Guarantee a 2D slice. Some loader paths (4D NIfTI, odd DICOM layouts)
    return a 3D slice; squeeze size-1 dims and, if still 3D, take the
    middle index along the smallest axis — same rule used by the draw step.
    """
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim == 3:
        slice_dim = int(np.argmin(arr.shape))
        idx = arr.shape[slice_dim] // 2
        if slice_dim == 0:
            arr = arr[idx]
        elif slice_dim == 1:
            arr = arr[:, idx, :]
        else:
            arr = arr[:, :, idx]
    return arr


@st.cache_data(show_spinner=False)
def _load_display_data(original_image_path, output_path, rotate_k: int, flip_h: bool, flip_v: bool):
    """Load + normalise images once per (file, orientation); cached so the slider doesn't re-read disk."""
    # _load_slice_prefetched returns from the shared subprocess cache when the
    # file was warmed by a previous rerun; otherwise it loads synchronously.
    img, _, _, _, img_type = _load_slice_prefetched(original_image_path)
    msk = load_polygon_mask_2d(output_path)
    if msk is None:
        raise FileNotFoundError(f"No saved polygon mask found for {output_path}")
    img = _ensure_2d(img)
    msk = _ensure_2d(msk)
    # The mask on disk is already in the oriented frame (draw step saved it
    # that way); only the original image needs the user orientation applied.
    img = apply_orientation(img, rotate_k, flip_h, flip_v)
    img_show = ImageProcessor.normalize_image(img)          # uint8 0-255
    msk_show = (msk > 0).astype(np.uint8)                   # binary
    dt = DisplayTransform(padding=0)
    dt.set_rotation_for_type(img_type)
    if dt.rotate_k % 4 != 0:
        img_show = np.rot90(img_show, k=dt.rotate_k)
        msk_show = np.rot90(msk_show, k=dt.rotate_k)
    return img_show, msk_show, img_type


def _render_threshold_pil(img_show, msk_show, threshold, width):
    """
    Fast PIL-based threshold overlay.
    Replaces matplotlib figure rendering (was ~4800×4800 px at dpi=300).
    Thresholded pixels get a red tint at 35% alpha; everything else stays grey.
    """
    norm         = img_show.astype(np.float32) / 255.0
    thresh_region = (norm > threshold) & (msk_show > 0)

    rgb = np.stack([img_show, img_show, img_show], axis=-1).astype(np.float32)
    rgb[thresh_region, 0] = rgb[thresh_region, 0] * 0.65 + 255 * 0.35  # red
    rgb[thresh_region, 1] = rgb[thresh_region, 1] * 0.65
    rgb[thresh_region, 2] = rgb[thresh_region, 2] * 0.65
    result = rgb.clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(result)
    h, w    = result.shape[:2]
    if w > 0:
        pil_img = pil_img.resize((width, max(1, int(width * h / w))), Image.LANCZOS)
    return pil_img


def batch_threshold_step():
    """
    Step 3: Batch threshold adjustment for all files.
    Organizes navigation, threshold selection, and saving for each file in the batch.
    """

    # =====================================================
    # UI header and styling
    # =====================================================
    st.header("🎯 Step 3: Adjust Thresholds")
    with open("static/batch_threshold_step.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # =====================================================
    # Session state and Data validation
    # =====================================================
    if "batch_files" not in st.session_state:
        st.error("No batch files selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return

    batch_files = st.session_state["batch_files"]
    current_index = st.session_state.get("batch_current_index", 0)
    completed_draw = st.session_state["batch_completed_files"]["draw"]
    all_completed_threshold = st.session_state["batch_completed_files"]["threshold"]
    completed_threshold = [f for f in all_completed_threshold if f in st.session_state["batch_files_without_extension"]] # only the set of batch files that already has a polygon
    
    # check if the threshold.json is in the output folder
    for file in completed_threshold:
        output_path = os.path.join(os.getcwd(), 'output', file)
        threshold_json = os.path.join(output_path, "threshold.json")
        if not os.path.exists(threshold_json):
            completed_threshold.remove(file)

    # =====================================================
    # Draw step completion check
    # =====================================================
    if len(completed_draw) == 0:
        st.error("No masks drawn yet. Please draw at least one mask first.")
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
        return

    if len(completed_draw) < len(batch_files):
        remaining = len(batch_files) - len(completed_draw)
        st.info(
            f"{len(completed_draw)}/{len(batch_files)} masks ready. "
            f"{remaining} file{'s' if remaining != 1 else ''} still need{'s' if remaining == 1 else ''} a mask — "
            "you can go back to draw them at any time."
        )
        if st.button("← Back to Draw Step", use_container_width=False):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()

    # =====================================================
    # Draw-then-threshold flow: if we were sent here from the draw step
    # after saving a single mask, bounce straight back to draw for the
    # next file WITHOUT showing any completion screen.
    # =====================================================
    draw_return = st.session_state.get("draw_return_index")
    if draw_return is not None:
        # Only redirect AFTER the threshold for the current file is saved.
        # If it isn't saved yet, fall through and show the threshold UI.
        current_file_for_check = batch_files[current_index] if current_index < len(batch_files) else None
        current_file_name_for_check = current_file_for_check.split('.')[0] if current_file_for_check else None
        if current_file_name_for_check and current_file_name_for_check in completed_threshold:
            st.session_state.pop("draw_return_index", None)
            st.session_state["batch_current_index"] = draw_return
            st.session_state["current_step"] = "batch_draw"
            st.rerun()

    # =====================================================
    # Progress bar and overall navigation
    # =====================================================
    # Use only files that have a mask as the denominator
    files_with_mask = len(completed_draw)
    st.write(f"### Progress: {len(completed_threshold)}/{files_with_mask} files thresholded")
    st.progress(len(completed_threshold) / files_with_mask if files_with_mask else 0)
    st.markdown('</div>', unsafe_allow_html=True)

    # "Done" when every masked file also has a threshold — show completion message
    batch_files_without_ext = st.session_state.get("batch_files_without_extension", [])
    all_done = (
        all(f in completed_draw for f in batch_files_without_ext)
        and all(f in completed_threshold for f in batch_files_without_ext)
    )
    if all_done:
        st.success("🎉 All files masked and thresholded! Ready to process.")
        st.session_state["current_step"] = "batch_process"
        st.rerun()

    # =====================================================
    # File selection and skipping logic 
    # =====================================================
    current_file = batch_files[current_index]
    current_file_name = current_file.split('.')[0]

    # Skip if already completed: go to next pending file
    if current_file_name in completed_threshold:
        next_index = None
        for idx in range(current_index + 1, len(batch_files)):
            if batch_files[idx].split('.')[0] not in completed_threshold:
                next_index = idx
                break
        if next_index is not None:
            st.session_state["batch_current_index"] = next_index
        else:
            st.session_state["batch_current_index"] = len(batch_files)
        st.rerun()
        return

    # Skip if not in completed_draw: go to next file
    if current_file_name not in completed_draw:
        st.session_state["batch_current_index"] = current_index + 1
        st.rerun()
        return

    # =====================================================
    # Current file UI and navigation 
    # =====================================================
    st.markdown(f"""
    <div class="current-file">
        <h4>📁 {current_file} ({current_index + 1}/{len(batch_files)})</h4>
        <p>💡 Navigate between files to adjust thresholds • Saved files are locked</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_index > 0 and st.button("← Previous", help="Navigate to previous file"):
            st.session_state["batch_current_index"] = current_index - 1
            st.rerun()
    with col3:
        next_file_index = current_index + 1
        if next_file_index < len(batch_files) and st.button("Next →", help="Navigate to next file"):
            st.session_state["batch_current_index"] = next_file_index
            st.rerun()

    # =====================================================
    # Completed files list 
    # =====================================================
    if len(completed_threshold) > 0:
        with st.expander(f"✅ Completed Files ({len(completed_threshold)})", expanded=False):
            st.caption("*These files have saved thresholds and are locked*")
            cols = st.columns(3)
            for i, file in enumerate(completed_threshold):
                with cols[i % 3]:
                    st.markdown(f"🔒 **{file}.nii**")

    # =====================================================
    # File paths for current file 
    # =====================================================
    # Use main input folder selected in Step 1
    input_folder = st.session_state.get("main_input_folder", os.path.join(os.getcwd(), "media"))
    output_path = os.path.join(os.getcwd(), "output", current_file_name)
    original_image_path = os.path.join(input_folder, current_file)

    if not os.path.exists(input_folder):
        st.error(f"The selected main input folder does not exist: {input_folder}")
        return


    # =====================================================
    # File existence and image loading
    # =====================================================
    if not has_polygon_mask(output_path):
        st.error("Mask file not found for this case.")
        st.write("This file may not have completed the draw step properly.")
        if st.button("Skip this file"):
            st.session_state["batch_current_index"] = current_index + 1
            st.rerun()
        return

    try:
        orient = _load_orientation(output_path, current_file_name)
        img_show, msk_show, img_type = _load_display_data(
            original_image_path, output_path,
            orient["rotate_k"], orient["flip_h"], orient["flip_v"],
        )
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        if st.button("Skip this file"):
            st.session_state["batch_current_index"] = current_index + 1
            st.rerun()
        return

    # Warm the next few pending files in background subprocesses so that
    # clicking Save & Continue / Next → lands on an already-decoded slice.
    # Guarded per index: every slider / number-input interaction reruns the
    # script, but we only want to kick the prefetch once per displayed file.
    prefetch_key = f"_threshold_prefetch_idx_{current_index}"
    if not st.session_state.get(prefetch_key):
        warmed = 0
        for idx in range(current_index + 1, len(batch_files)):
            if warmed >= _THRESHOLD_PREFETCH_AHEAD:
                break
            nxt_name = batch_files[idx].split('.')[0]
            if nxt_name in completed_draw and nxt_name not in completed_threshold:
                _prefetch(os.path.join(input_folder, batch_files[idx]))
                warmed += 1
        st.session_state[prefetch_key] = True

    # =====================================================
    # Threshold preview and controls 
    # =====================================================
    st.markdown("### Threshold Configuration")
    
    # Main configuration panel
    with st.container():
        st.markdown('<div class="threshold-controls">', unsafe_allow_html=True)
        
        # Top row: Threshold control and view options
        config_col1, config_col2, config_col3 = st.columns([2, 2, 1])
        
        with config_col1:
            st.markdown("**Threshold Value**")
            
            # Update the session state immediately when the threshold value changes
            if "batch_thresholds" not in st.session_state:
                st.session_state["batch_thresholds"] = {}

            # Update the session state directly when the number input changes
            def update_threshold():
                st.session_state["batch_thresholds"][current_file_name] = st.session_state[f"input_{current_file_name}"]

            threshold = st.number_input(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["batch_thresholds"].get(current_file_name, 0.38)),
                step=0.01,
                format="%.2f",
                key=f"input_{current_file_name}",
                help="Lower = more inclusive, Higher = more selective",
                label_visibility="collapsed",
                on_change=update_threshold
            )
            
            st.session_state["batch_thresholds"][current_file_name] = threshold
            
        with config_col2:
            st.markdown("**View Options**")
            comparison_mode = st.checkbox("Show side-by-side comparison",
                                        value=True, 
                                        key=f"comparison_{current_file_name}",
                                        help="Compare original image with thresholded result")
            
        with config_col3:
            st.markdown("**Image Width**")
            width_options = [400, 500, 600, 700, 800, 900, 1000, 1200, 1400]
            # Retrieve the last selected width index from session state or use default
            width_key = "last_selected_width_index"
            default_index = st.session_state.get(width_key, 4)
            
            # Update session state immediately when width changes
            def update_width():
                st.session_state[width_key] = width_options.index(st.session_state[f"width_{current_file}"])
            
            selected_width = st.selectbox("Width", width_options, index=default_index, 
                                        key=f"width_{current_file}", 
                                        label_visibility="collapsed",
                                        on_change=update_width)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Update session state
    #if "batch_thresholds" not in st.session_state:
        #st.session_state["batch_thresholds"] = {}
    #st.session_state["batch_thresholds"][current_file_name] = threshold

    # Image preview section
    st.markdown("### Image Preview")

    col_w = int(selected_width * 0.55)
    h, w  = img_show.shape
    orig_pil = Image.fromarray(img_show).resize(
        (col_w, max(1, int(col_w * h / w))), Image.LANCZOS
    )

    if comparison_mode:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("**Original Image**")
            st.image(orig_pil, width=col_w)
        with img_col2:
            st.markdown("**Thresholded Result**")
            st.image(_render_threshold_pil(img_show, msk_show, threshold, col_w), width=col_w)
    else:
        _, img_col, _ = st.columns([1, 2, 1])
        with img_col:
            st.image(_render_threshold_pil(img_show, msk_show, threshold, selected_width),
                     width=selected_width)

    # Action buttons section
    st.markdown("---")
    button_col1, button_col2, button_col3 = st.columns([2, 1, 2])
    
    with button_col1:
        if st.button("Save & Continue", type="primary", use_container_width=True):
            if "batch_final_thresholds" not in st.session_state:
                st.session_state["batch_final_thresholds"] = {}
            st.session_state["batch_final_thresholds"][current_file_name] = threshold
            try:
                threshold_json = os.path.join(output_path, "threshold.json")
                with open(threshold_json, "w") as f:
                    json.dump({"threshold": threshold}, f)
                st.session_state["batch_completed_files"]["threshold"].append(current_file_name)
                st.success(f"Threshold {threshold:.2f} saved successfully!")
            except Exception as e:
                st.error(f"Could not save threshold: {e}")
            # Rerun: the top-of-page draw_return_index check will bounce
            # back to the draw step if we're in one-file-at-a-time mode.
            st.rerun()
    
    with button_col3:
        if st.button("Back to Draw Step", use_container_width=True):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()

    # =====================================================
    # Show current thresholds table 
    # =====================================================
    if "batch_final_thresholds" in st.session_state and len(st.session_state["batch_final_thresholds"]) > 0:
        with st.expander(f"📊 Saved Thresholds ({len(st.session_state['batch_final_thresholds'])})", expanded=False):
            cols = st.columns(3)
            for i, (file_name, thresh) in enumerate(st.session_state["batch_final_thresholds"].items()):
                with cols[i % 3]:
                    st.metric(f"{file_name}", f"{thresh:.2f}")
