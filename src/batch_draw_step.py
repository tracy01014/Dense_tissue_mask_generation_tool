# =========================
# Imports
# =========================
import os
import streamlit as st
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events

from new_utils import MaskManager, DisplayTransform, resolve_dense_mask_path
from ImageLoader import UnifiedImageLoader


@st.cache_data(show_spinner=False)
def _load_slice_cached(file_path):
    return UnifiedImageLoader.load_slice(file_path)


def _get_display_rgb(file_path, image_slice, img_type, transform):
    if st.session_state.get("_disp_cache_key") == file_path:
        return (
            st.session_state["_disp_img_rgb"],
            st.session_state["_disp_w"],
            st.session_state["_disp_h"],
        )
    pil_image = transform.prepare_for_display(
        image_slice, img_type=img_type, max_width=1200, max_height=800
    )
    display_w, display_h = pil_image.size
    arr = np.array(pil_image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    st.session_state["_disp_cache_key"] = file_path
    st.session_state["_disp_img_rgb"]   = arr
    st.session_state["_disp_w"]         = display_w
    st.session_state["_disp_h"]         = display_h
    return arr, display_w, display_h


def _clear_draw_state():
    keys = [
        "display_transform", "mask", "result", "create_mask",
        "affine", "original_image_path", "slice_index", "output_path",
        "original_shape", "current_batch_file", "img_type",
        "native_polygons", "completed_polygons_display",
        "current_points", "last_click",
        "_disp_cache_key", "_disp_img_rgb", "_disp_w", "_disp_h",
    ]
    for k in keys:
        st.session_state.pop(k, None)


_STROKE = ["red", "lime", "royalblue", "gold", "magenta", "cyan", "orange"]
_FILL   = [
    "rgba(255,0,0,0.20)",   "rgba(0,255,0,0.20)",     "rgba(65,105,225,0.20)",
    "rgba(255,215,0,0.20)", "rgba(255,0,255,0.20)",   "rgba(0,255,255,0.20)",
    "rgba(255,165,0,0.20)",
]


def _build_figure(img_rgb, completed_display, current_points, display_w, display_h):
    # px.imshow with a 3-D RGB array produces go.Image - a real Plotly trace
    # that fires click events, unlike go.layout.Image which is a layout element.
    fig = px.imshow(img_rgb, aspect="equal")
    fig.update_layout(
        height=display_h, width=display_w,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    for i, poly in enumerate(completed_display):
        if len(poly) < 2:
            continue
        c  = _STROKE[i % len(_STROKE)]
        fc = _FILL[i % len(_FILL)]
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        fig.add_scatter(x=xs, y=ys, mode="lines",
                        line=dict(color=c, width=2),
                        fill="toself", fillcolor=fc,
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=[p[0] for p in poly], y=[p[1] for p in poly],
                        mode="markers", marker=dict(size=8, color=c),
                        showlegend=False, hoverinfo="skip")
        cx_a = sum(p[0] for p in poly) / len(poly)
        cy_a = sum(p[1] for p in poly) / len(poly)
        fig.add_annotation(x=cx_a, y=cy_a, text=f"#{i+1}",
                           font=dict(color=c, size=14), showarrow=False)

    if current_points:
        fig.add_scatter(
            x=[p[0] for p in current_points],
            y=[p[1] for p in current_points],
            mode="lines+markers",
            line=dict(color="yellow", width=2, dash="dot"),
            marker=dict(size=10, color="yellow", symbol="circle-open",
                        line=dict(width=2, color="yellow")),
            showlegend=False, hoverinfo="skip",
        )
    return fig


def batch_draw_step():
    st.header("Step 2: Draw Masks")
    with open("static/batch_draw_step.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    if "batch_files" not in st.session_state:
        st.error("No batch files selected. Please go back to file selection.")
        if st.button("Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return

    batch_files        = st.session_state["batch_files"]
    current_index      = st.session_state.get("batch_current_index", 0)
    all_completed_draw = st.session_state["batch_completed_files"]["draw"]
    completed_draw     = [
        f for f in all_completed_draw
        if f in st.session_state["batch_files_without_extension"]
    ]

    total_files = len(batch_files)
    st.write(f"### Progress: {len(completed_draw)}/{total_files} files completed")
    st.progress(len(completed_draw) / total_files)

    # Allow jumping to threshold step whenever at least one mask exists
    if len(completed_draw) > 0:
        if st.button(
            f"→ Go to Threshold Step ({len(completed_draw)} file{'s' if len(completed_draw) != 1 else ''} ready)",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["batch_current_index"] = 0
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()

    if current_index >= len(batch_files):
        st.success("All masks have been drawn!")
        if st.button("Continue to Threshold Step"):
            st.session_state["batch_current_index"] = 0
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        if st.button("Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return

    # ---- Jump-to dropdown ----
    def _make_label(i, f):
        name = f.split(".")[0]
        return f"✅ {f}" if name in completed_draw else f"⬜ {f}"

    dropdown_labels = [_make_label(i, f) for i, f in enumerate(batch_files)]
    selected_label  = st.selectbox(
        "Jump to file:",
        dropdown_labels,
        index=current_index,
        key="draw_jump_dropdown",
    )
    jumped_index = dropdown_labels.index(selected_label)
    if jumped_index != current_index:
        st.session_state["batch_current_index"] = jumped_index
        _clear_draw_state()
        st.rerun()

    current_index     = jumped_index
    current_file      = batch_files[current_index]
    current_file_name = current_file.split(".")[0]

    # ---- Header with free Prev / Next navigation ----
    is_completed = current_file_name in completed_draw

    status_label = "✅ Saved" if is_completed else f"({current_index + 1}/{total_files})"
    header_hint  = ("Viewing saved mask. Click Redraw to replace it."
                    if is_completed else
                    "Click the image to place vertices. Use the controls below to manage polygons.")

    st.markdown(f"""
    <div class="current-file">
        <h4>{current_file} &nbsp; <span style="font-size:0.85rem;">{status_label}</span></h4>
        <p>{header_hint}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, _mid, col3 = st.columns([1, 2, 1])
    with col1:
        if current_index > 0 and st.button("← Previous"):
            st.session_state["batch_current_index"] = current_index - 1
            _clear_draw_state()
            st.rerun()
    with col3:
        if current_index + 1 < len(batch_files) and st.button("Next →"):
            st.session_state["batch_current_index"] = current_index + 1
            _clear_draw_state()
            st.rerun()

    st.divider()

    # ---- Shared: load original image ----
    input_folder = st.session_state.get(
        "main_input_folder", os.path.join(os.getcwd(), "media")
    )
    file_path = os.path.join(input_folder, current_file)

    if not os.path.exists(input_folder):
        st.error(f"Input folder not found: {input_folder}")
        return
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    with st.spinner("Loading image..."):
        image_slice, affine, original_shape, slice_index, img_type = (
            _load_slice_cached(file_path)
        )

    # =========================================================
    # VIEW MODE  — file already has a saved mask
    # =========================================================
    if is_completed:
        output_path = os.path.join(os.getcwd(), "output", current_file_name)
        is_dicom    = os.path.isdir(file_path) or file_path.lower().endswith((".dcm", ".dicom"))
        mask_path   = resolve_dense_mask_path(output_path, prefer_dicom=is_dicom)

        if mask_path is None:
            st.warning("Saved mask file not found on disk. You can redraw it below.")
        else:
            st.markdown("### Saved Mask Preview")
            col1, col2 = st.columns(2)

            view_transform = DisplayTransform(padding=50)
            with st.spinner("Loading saved mask..."):
                mask_slice, _, _, _, _ = _load_slice_cached(mask_path)

            present_orig = view_transform.prepare_for_display(
                image_slice, img_type=img_type, max_width=900, max_height=600
            )
            present_mask = view_transform.prepare_for_display(
                mask_slice, img_type=img_type, max_width=900, max_height=600
            )

            with col1:
                st.image(present_orig, caption="Original Image",  use_container_width=True)
            with col2:
                st.image(present_mask, caption="Saved Mask",      use_container_width=True)

        st.divider()

        col_redraw, col_next = st.columns(2)
        with col_redraw:
            if st.button("✏️ Redraw Mask", type="primary", use_container_width=True,
                         help="Delete the saved mask and draw a new one"):
                # Remove from completed lists so drawing is unlocked
                st.session_state["batch_completed_files"]["draw"] = [
                    f for f in st.session_state["batch_completed_files"]["draw"]
                    if f != current_file_name
                ]
                # Also clear downstream completed states for this file
                for step in ("threshold", "process"):
                    st.session_state["batch_completed_files"][step] = [
                        f for f in st.session_state["batch_completed_files"][step]
                        if f != current_file_name
                    ]
                _clear_draw_state()
                st.rerun()
        with col_next:
            next_idx = current_index + 1
            if next_idx < len(batch_files) and st.button("Next →", key="view_next",
                                                          use_container_width=True):
                st.session_state["batch_current_index"] = next_idx
                _clear_draw_state()
                st.rerun()
        return

    # =========================================================
    # DRAW MODE  — file not yet completed
    # =========================================================

    if st.session_state.get("current_batch_file") != current_file:
        st.session_state.update({
            "affine":                     affine,
            "original_image_path":        file_path,
            "original_shape":             original_shape,
            "slice_index":                slice_index,
            "current_batch_file":         current_file,
            "img_type":                   img_type,
            "native_polygons":            [],
            "completed_polygons_display": [],
            "current_points":             [],
            "last_click":                 None,
        })
        for k in ("mask", "result", "create_mask", "display_transform",
                  "_disp_cache_key", "_disp_img_rgb", "_disp_w", "_disp_h"):
            st.session_state.pop(k, None)

    if "display_transform" not in st.session_state:
        st.session_state["display_transform"] = DisplayTransform(padding=50)
    transform = st.session_state["display_transform"]

    img_rgb, display_w, display_h = _get_display_rgb(
        file_path, image_slice, img_type, transform
    )

    completed_display = st.session_state.get("completed_polygons_display", [])
    current_points    = st.session_state.get("current_points", [])
    n_done = len(completed_display)
    n_pts  = len(current_points)

    fig = _build_figure(img_rgb, completed_display, current_points, display_w, display_h)

    if n_pts == 0 and n_done == 0:
        st.caption("Click anywhere on the image to start placing vertices.")
    elif n_pts > 0:
        need = max(0, 3 - n_pts)
        suffix = f"  Place {need} more to enable Close Polygon." if need else ""
        st.caption(f"{n_pts} vertex{'es' if n_pts != 1 else ''} placed.{suffix}")
    else:
        st.caption(f"{n_done} polygon{'s' if n_done != 1 else ''} completed.")

    st.markdown("### Drawing Canvas")

    # Safe key: forward-slash and dots from nested DICOM paths would break
    # Streamlit's widget key system on some versions
    canvas_key = "canvas_" + current_file.replace("/", "_").replace("\\", "_").replace(".", "_")

    clicked = plotly_events(
        fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        override_height=display_h,
        override_width=display_w,
        key=canvas_key,
    )

    # plotly_events replays the last stored event on every rerun;
    # deduplicate so button-triggered reruns don't add duplicate vertices.
    if clicked:
        ev = clicked[-1]
        new_click = (round(float(ev["x"])), round(float(ev["y"])))
        if new_click != st.session_state.get("last_click"):
            st.session_state["last_click"] = new_click
            st.session_state["current_points"].append(new_click)
            st.rerun()

    st.markdown("### Controls")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("Close Polygon", disabled=n_pts < 3,
                     use_container_width=True,
                     help="Finalise the current polygon (needs >= 3 points)"):
            disp_pts   = list(st.session_state["current_points"])
            native_pts = transform.canvas_to_native_coords(disp_pts)
            st.session_state["completed_polygons_display"].append(disp_pts)
            st.session_state["native_polygons"].append(native_pts)
            st.session_state["current_points"] = []
            st.rerun()

    with c2:
        if st.button("Undo Last Point", disabled=n_pts == 0,
                     use_container_width=True,
                     help="Remove the most recently placed vertex"):
            st.session_state["current_points"].pop()
            st.rerun()

    with c3:
        if st.button("Delete Last Polygon", disabled=n_done == 0,
                     use_container_width=True,
                     help="Remove the most recently closed polygon"):
            st.session_state["completed_polygons_display"].pop()
            st.session_state["native_polygons"].pop()
            st.rerun()

    with c4:
        if st.button("Clear All", use_container_width=True,
                     help="Delete all polygons and in-progress points"):
            st.session_state["current_points"]             = []
            st.session_state["completed_polygons_display"] = []
            st.session_state["native_polygons"]            = []
            st.session_state["last_click"]                 = None
            st.rerun()

    st.divider()

    col_mask, col_back = st.columns(2)

    with col_mask:
        if st.button("Create Mask", disabled=n_done == 0, type="primary",
                     use_container_width=True,
                     help="Generate the segmentation mask from all closed polygons"):
            st.session_state["create_mask"] = True

    with col_back:
        if st.button("Back to File Selection", use_container_width=True):
            st.session_state["current_step"] = "file_selection"
            st.rerun()

    if st.session_state.get("create_mask", False):
        valid = [p for p in st.session_state.get("native_polygons", []) if len(p) >= 3]
        if valid:
            result, combined_mask = MaskManager.create_combined_mask(image_slice, valid)
            st.session_state["mask"]        = combined_mask
            st.session_state["result"]      = result
            st.session_state["output_path"] = os.path.join(
                os.getcwd(), "output", current_file_name
            )
            st.session_state["create_mask"] = False
            st.rerun()
        else:
            st.warning("Close at least one polygon (>= 3 points) before creating a mask.")
            st.session_state["create_mask"] = False

    if "result" in st.session_state and "mask" in st.session_state:
        st.markdown("### Mask Preview")
        col1, col2 = st.columns(2)

        present_img    = transform.prepare_for_display(
            image_slice, img_type=st.session_state["img_type"]
        )
        present_result = transform.prepare_for_display(
            st.session_state["result"], img_type=st.session_state["img_type"]
        )

        with col1:
            st.image(present_img,    caption="Original Image", use_container_width=True)
        with col2:
            st.image(present_result, caption="Segmented Area", use_container_width=True)

        _, col_save, _ = st.columns([1, 1, 1])
        with col_save:
            if st.button("Save Mask & Continue", type="primary"):
                if current_file_name not in st.session_state["batch_completed_files"]["draw"]:
                    with st.spinner("Saving mask…"):
                        MaskManager.save_mask(
                            mask_2d=st.session_state["mask"],
                            original_shape=st.session_state["original_shape"],
                            affine=st.session_state["affine"],
                            img_type=st.session_state["img_type"],
                            file_path=st.session_state["output_path"],
                            original_image_path=st.session_state["original_image_path"],
                        )
                    st.session_state["batch_completed_files"]["draw"].append(current_file_name)
                st.session_state["batch_current_index"] = current_index + 1
                _clear_draw_state()
                st.toast(f"Mask saved for {current_file}!", icon="✅")
                st.rerun()
