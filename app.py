from html import escape
from math import isfinite
from pathlib import Path
import time
import uuid

import pandas as pd
import streamlit as st

from PIL import Image
import numpy as np

from database import initialize_database, save_scan, get_scans
from xray_model import predict_xray
from attention_visualization import (
    prepare_attention_views,
    rank_attention_regions,
    attention_focus_labels,
)

# ------------------------------------------------------------------------
# Application setup
# ------------------------------------------------------------------------

st.set_page_config(page_title="TRIGE / Static Imaging", layout="wide")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
initialize_database()
st.html(Path(__file__).with_name("styles.css"))

SCREENS = (
    "title",
    "intake",
    "loading",
    "result",
    "explanation",
    "record",
    "history",
)

SUPPORTING = (
    "Pneumonia",
    "Lung Opacity",
    "Consolidation",
    "Infiltration",
    "Effusion",
)

st.session_state.setdefault("screen", "title")
st.session_state.setdefault("trige_upload_version", 0)


# ------------------------------------------------------------------------
# Session state and navigation
# ------------------------------------------------------------------------


def navigate(screen):
    st.session_state.screen = screen
    st.rerun()


def new_scan():
    # Clear only the current UI/session state. Never clear model caches or SQLite.
    pending = st.session_state.get("pending_analysis")
    if pending:
        try:
            Path(pending["image_path"]).unlink(missing_ok=True)
        except (KeyError, OSError, TypeError):
            pass

    for key in (
        "trige_result",
        "trige_upload_bytes",
        "trige_upload_name",
        "pending_analysis",
        "analysis_error",
    ):
        st.session_state.pop(key, None)

    st.session_state.trige_upload_version += 1
    navigate("intake")


def capture_upload():
    key = f"trige_upload_{st.session_state.trige_upload_version}"
    upload = st.session_state.get(key)
    if upload is not None:
        st.session_state.trige_upload_bytes = upload.getvalue()
        st.session_state.trige_upload_name = upload.name
    else:
        st.session_state.pop("trige_upload_bytes", None)
        st.session_state.pop("trige_upload_name", None)


# ------------------------------------------------------------------------
# Shared presentation helpers
# ------------------------------------------------------------------------


def label(text):
    st.html(f'<div class="instrument-label">{escape(text)}</div>')


def heading(kicker, title):
    label(kicker)
    st.subheader(title)


def metadata(items):
    st.html(
        '<dl class="metadata">'
        + "".join(
            f"<div><dt>{escape(str(k))}</dt><dd>{escape(str(v))}</dd></div>"
            for k, v in items
        )
        + "</dl>"
    )


def render_workflow():
    current = {
        "intake": 0,
        "loading": 1,
        "result": 2,
        "explanation": 3,
        "record": 4,
    }.get(st.session_state.screen)

    stages = ("01 INTAKE", "02 ANALYZE", "03 RESULT", "04 EXPLAIN", "05 RECORD")
    spans = []

    for index, title in enumerate(stages):
        state = (
            "future"
            if current is None or index > current
            else ("current" if index == current else "complete")
        )
        spans.append(f'<span class="stage {state}">{title}</span>')

    st.html(
        '<nav class="workflow" aria-label="Analysis workflow">'
        + "".join(spans)
        + "</nav>"
    )


def render_header():
    brand, status, archive = st.columns([5, 2, 1.6])
    with brand:
        st.html(
            '<div class="brand">TRIGE</div><div class="instrument-label">'
            "STATIC IMAGING MODULE // REV 0.1</div>"
        )
    with status:
        if st.session_state.screen == "loading":
            status_text = "ANALYZING"
        elif st.session_state.get("trige_result"):
            status_text = "ANALYSIS COMPLETE"
        else:
            status_text = "SYSTEM READY"

        st.html(
            f'<div class="system-status"><span class="status-dot"></span>{status_text}</div>'
        )
    with archive:
        if st.button("ARCHIVE / HISTORY", width="stretch"):
            navigate("history")
    render_workflow()
    st.html(
        '<div class="research-notice">Research prototype only. '
        "Not clinically validated. Not for diagnosis or patient-care decisions.</div>"
    )


def navigation(left_text, left_screen, right_text=None, right_screen=None):
    left, right = st.columns(2)

    with left:
        if st.button(left_text, key=f"back_{st.session_state.screen}"):
            if left_screen == "new":
                new_scan()
            else:
                navigate(left_screen)

    if right_text and right_screen:
        with right:
            if st.button(right_text, type="primary", width="stretch"):
                navigate(right_screen)


def display_image(image, title):
    with st.container(border=True):
        label(title)
        st.image(image, width="stretch")


# ------------------------------------------------------------------------
# Computed output helpers
# ------------------------------------------------------------------------


def supporting_values(result):
    values = []
    for name in SUPPORTING:
        try:
            value = float(result.get("all_pathology_scores", {})[name])
            if isfinite(value):
                values.append((name, value))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return sorted(values, key=lambda item: item[1], reverse=True)


def regional_focus(result):
    return attention_focus_labels(rank_attention_regions(result["attention_map"]))


def prepare_views(result):
    if "attention_overlay" not in result:
        original, overlay = prepare_attention_views(
            result["image_path"], result["attention_map"]
        )
        result.update(display_original=original, attention_overlay=overlay)


# ------------------------------------------------------------------------
# Existing inference and SQLite integration
# ------------------------------------------------------------------------

def validate_radiograph_input(image_path):
    """
    Lightweight pre-inference screening.

    This does NOT prove that an image is a valid chest radiograph.
    It only rejects inputs that are clearly incompatible with the
    expected grayscale radiograph workflow.
    """

    try:
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image).astype(np.float32)

    except Exception:
        return False, "The uploaded file could not be read as an image."

    height, width = array.shape[:2]

    # ---------------------------------------------------------
    # 1. Minimum resolution
    # ---------------------------------------------------------
    if width < 224 or height < 224:
        return (
            False,
            "Image resolution is too small. Please upload a chest radiograph "
            "with dimensions of at least 224 × 224 pixels.",
        )

    # ---------------------------------------------------------
    # 2. Check whether image is approximately grayscale
    # ---------------------------------------------------------
    r = array[:, :, 0]
    g = array[:, :, 1]
    b = array[:, :, 2]

    channel_difference = (
        np.mean(np.abs(r - g))
        + np.mean(np.abs(g - b))
        + np.mean(np.abs(r - b))
    ) / 3.0

    if channel_difference > 12:
        return (
            False,
            "This image contains substantial color information and does not "
            "appear consistent with the expected radiograph input.",
        )

    # ---------------------------------------------------------
    # 3. Check intensity variation
    # ---------------------------------------------------------
    gray = np.mean(array, axis=2)

    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    dynamic_range = p95 - p5

    if dynamic_range < 35:
        return (
            False,
            "The image has insufficient intensity variation for the expected "
            "radiograph workflow.",
        )

    # ---------------------------------------------------------
    # 4. Reject extremely blank / saturated images
    # ---------------------------------------------------------
    dark_fraction = np.mean(gray < 5)
    bright_fraction = np.mean(gray > 250)

    if dark_fraction > 0.92 or bright_fraction > 0.92:
        return (
            False,
            "The uploaded image is mostly blank or saturated.",
        )

    return True, None

def begin_analysis():
    name = st.session_state.trige_upload_name

    path = UPLOAD_DIR / (
        f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"
    )

    try:
        path.write_bytes(
            st.session_state.trige_upload_bytes
        )

        # If you added your radiograph input validation,
        # perform it here BEFORE entering the loading screen.
        if "validate_radiograph_input" in globals():
            valid, validation_error = validate_radiograph_input(path)

            if not valid:
                try:
                    path.unlink()
                except OSError:
                    pass

                st.error(
                    "INPUT VALIDATION FAILED\n\n"
                    + validation_error
                    + "\n\nPlease upload a de-identified chest radiograph."
                )

                return

        st.session_state.pop("analysis_error", None)
        st.session_state.pending_analysis = {
            "filename": name,
            "image_path": str(path),
        }

        navigate("loading")

    except Exception as error:
        st.error(
            f"Could not prepare image for analysis: {error}"
        )

def complete_pending_analysis():
    pending = st.session_state.get(
        "pending_analysis"
    )

    if not pending:
        navigate("intake")

    name = pending["filename"]
    path = Path(pending["image_path"])

    try:
        # Snapshot existing IDs so the new database row
        # can still be matched exactly.
        try:
            old_ids = {
                row[0]
                for row in get_scans()
            }

        except Exception:
            old_ids = None

        # -------------------------------
        # REAL MODEL INFERENCE
        # -------------------------------

        score, all_scores, attention_map = (
            predict_xray(path)
        )

        flag = (
            "ELEVATED"
            if score >= 0.50
            else "LOW"
        )

        result = {
            "score": float(score),
            "flag": flag,
            "all_pathology_scores": all_scores,
            "attention_map": attention_map,
            "image_path": str(path),
            "filename": name,
            "record": None,
            "saved": False,
        }

        # -------------------------------
        # SQLITE SAVE
        # -------------------------------

        try:
            save_scan(
                name,
                score,
                flag,
            )

            result["saved"] = True

        except Exception as error:
            result["save_error"] = str(error)

        # -------------------------------
        # MATCH INSERTED RECORD
        # -------------------------------

        if (
            result["saved"]
            and old_ids is not None
        ):
            try:
                matches = [
                    tuple(row)
                    for row in get_scans()
                    if row[0] not in old_ids
                    and row[1] == name
                    and float(row[2])
                    == float(score)
                    and row[3] == flag
                ]

                if len(matches) == 1:
                    result["record"] = (
                        matches[0]
                    )

            except Exception:
                pass

        st.session_state.trige_result = (
            result
        )

        st.session_state.pop(
            "pending_analysis",
            None,
        )

        navigate("result")

    except Exception as error:
        st.session_state["analysis_error"] = str(error)
        return False

# ------------------------------------------------------------------------
# Project documentation
# ------------------------------------------------------------------------


def render_project_details():
    """Full project documentation, available without leaving the intake screen."""
    st.html("""
        <div class="trige-section-label">01 / PROJECT VISION</div>
        <div class="trige-result">
            <h3>Imaging today. Multimodal research ahead.</h3>
            <p>
                Trige is envisioned as a multimodal research system that will
                eventually combine medical imaging with longitudinal
                physiological data.
            </p>
            <p class="trige-score-unit">
                v0.1 intentionally implements only the static imaging layer.
            </p>
        </div>
        """)

    st.divider()
    st.html("""
        <div class="trige-section-label">02 / CURRENT IMPLEMENTATION</div>
        <span class="trige-pill">IMPLEMENTED · v0.1</span>
        <div class="trige-project-flow" role="group"
             aria-label="Implemented static imaging workflow">
            <div class="trige-project-node">Chest Radiograph</div>
            <div class="trige-project-arrow" aria-hidden="true">↓</div>
            <div class="trige-project-node">Preprocessing</div>
            <div class="trige-project-arrow" aria-hidden="true">↓</div>
            <div class="trige-project-node">DenseNet121</div>
            <div class="trige-project-arrow" aria-hidden="true">↓</div>
            <div class="trige-project-node">Pneumonia-Associated Model Score</div>
            <div class="trige-project-arrow" aria-hidden="true">↓</div>
            <div class="trige-project-node">SQLite Storage</div>
            <div class="trige-project-arrow" aria-hidden="true">↓</div>
            <div class="trige-project-node">Interactive Dashboard</div>
        </div>
        """)
    st.caption(
        "The current module integrates pretrained densenet121-res224-all "
        "weights. Its output is not a diagnosis or a clinically calibrated "
        "probability."
    )

    st.divider()
    st.html("""
        <div class="trige-section-label">03 / PLANNED ARCHITECTURE</div>
        <span class="trige-pill">PLANNED · NOT YET IMPLEMENTED</span>
        <div class="trige-project-flow" role="group"
             aria-label="Planned architecture: chest X-ray through a vision
             encoder and 48-hour vitals through a temporal encoder converge
             at multimodal fusion, followed by deterioration risk.">
            <div class="trige-project-branches">
                <div>
                    <div class="trige-project-node">Chest X-Ray</div>
                    <div class="trige-project-arrow" aria-hidden="true">↓</div>
                    <div class="trige-project-node">Vision Encoder</div>
                </div>
                <div>
                    <div class="trige-project-node">48h Vitals</div>
                    <div class="trige-project-arrow" aria-hidden="true">↓</div>
                    <div class="trige-project-node">Temporal Encoder</div>
                </div>
            </div>
            <div class="trige-project-join" aria-hidden="true"></div>
            <div class="trige-project-arrow" aria-hidden="true">↓</div>
            <div class="trige-project-node">Multimodal Fusion</div>
            <div class="trige-project-arrow" aria-hidden="true">↓</div>
            <div class="trige-project-node">Deterioration Risk</div>
        </div>
        """)
    st.info(
        "This is a proposed research architecture. Processing 48-hour vitals, "
        "temporal encoding, multimodal fusion, and deterioration-risk "
        "prediction are not implemented in v0.1."
    )

    st.divider()
    st.html("""
        <div class="trige-section-label">04 / TECHNOLOGY</div>
        <div class="trige-project-tech">
            <div class="trige-project-node">PyTorch</div>
            <div class="trige-project-node">TorchXRayVision</div>
            <div class="trige-project-node">DenseNet121</div>
            <div class="trige-project-node">Streamlit</div>
            <div class="trige-project-node">SQLite</div>
            <div class="trige-project-node">Python</div>
        </div>
        """)

    st.divider()
    st.html('<div class="trige-section-label">05 / LIMITATIONS</div>')
    with st.container(border=True):
        st.markdown("""
- **Pretrained model integration:** v0.1 uses existing model weights.
- **No original medical model training:** no medical model is trained in v0.1.
- **No clinical validation:** scores and demo flags are not validated for clinical use.
- **No real-time physiological data:** the current module analyzes static images only.
- **No multimodal fusion yet:** imaging and longitudinal vitals are not combined.
- **No clinical deployment:** v0.1 is not intended for diagnosis or patient-care decisions.
- **Research/demo purposes only:** the application demonstrates a research workflow.
            """)


# ------------------------------------------------------------------------
# Screen 0: title
# ------------------------------------------------------------------------

def render_title():
    st.html("""
        <section class="trige-title-screen">

            <div class="hud-noise"></div>
            <div class="hud-scanline"></div>

            <div class="hud-corner hud-corner-tl"></div>
            <div class="hud-corner hud-corner-tr"></div>
            <div class="hud-corner hud-corner-bl"></div>
            <div class="hud-corner hud-corner-br"></div>

            <div class="hud-topbar">
                <span>TRIGE // SYSTEM 01</span>
                <span>CLINICAL AI RESEARCH PROTOTYPE</span>
                <span class="hud-online">
                    <i></i>
                    ONLINE
                </span>
            </div>

            <div class="hud-main">

                <div class="hud-orbit">

                    <div class="hud-ring hud-ring-1"></div>
                    <div class="hud-ring hud-ring-2"></div>
                    <div class="hud-ring hud-ring-3"></div>
                    <div class="hud-ring hud-ring-4"></div>

                    <div class="hud-tick hud-tick-a"></div>
                    <div class="hud-tick hud-tick-b"></div>
                    <div class="hud-tick hud-tick-c"></div>
                    <div class="hud-tick hud-tick-d"></div>

                    <div class="hud-core">

                        <div class="hud-kicker">
                            TEMPORAL-SPATIAL CLINICAL TRIAGE ENGINE
                        </div>

                        <div class="hud-title">
                            TRIGE
                        </div>

                        <div class="hud-module">
                            STATIC IMAGING MODULE
                        </div>

                        <div class="hud-version">
                            REV 0.1
                        </div>

                    </div>

                </div>

                <div class="hud-system-line">
                    <span>
                        <i class="hud-dot"></i>
                        SYSTEM READY
                    </span>

                    <span>DENSENET121</span>
                    <span>GRAD-CAM</span>
                    <span>LOCAL ARCHIVE</span>
                </div>

                <div class="hud-description">
                    Explainable chest radiograph analysis using pretrained
                    deep-learning inference, model-attention visualization,
                    and local analysis history.
                </div>

            </div>

        </section>
        """)

    left, center, right = st.columns([1.55, 1, 1.55])

    with center:
        if st.button(
            "INITIALIZE TRIGE  →",
            type="primary",
            width="stretch",
            key="enter_trige",
        ):
            navigate("intake")

    st.html("""
        <div class="hud-footer">
            <span>RESEARCH PROTOTYPE</span>
            <span>NOT CLINICALLY VALIDATED</span>
            <span>NON-CLINICAL SYSTEM</span>
        </div>
        """)


# ------------------------------------------------------------------------
# Screen 1: intake
# ------------------------------------------------------------------------


@st.dialog("PROJECT / VISION, ARCHITECTURE & LIMITATIONS")
def show_project_dialog():
    render_project_details()


def render_intake():
    heading("01 / INTAKE", "CHEST RADIOGRAPH ANALYSIS")
    st.caption(
        "Load a de-identified chest radiograph to begin the static imaging analysis sequence."
    )
    image_col, controls = st.columns([1.5, 1], gap="large")
    with image_col:
        with st.container(border=True):
            label("DROP RADIOGRAPH / PNG / JPG / JPEG")
            st.file_uploader(
                "Select radiograph",
                type=["png", "jpg", "jpeg"],
                key=f"trige_upload_{st.session_state.trige_upload_version}",
                on_change=capture_upload,
                label_visibility="collapsed",
            )
            if st.session_state.get("trige_upload_bytes"):
                st.image(st.session_state.trige_upload_bytes, width="stretch")
                st.caption(st.session_state.trige_upload_name)
            else:
                st.html(
                    '<div class="empty-frame">RADIOGRAPH INPUT<span>AWAITING IMAGE</span></div>'
                )
    with controls:
        metadata(
            [
                ("INPUT", "Chest radiograph"),
                ("MODEL", "DenseNet121"),
                ("WEIGHTS", "res224-all"),
                ("MODULE", "Static Imaging"),
            ]
        )

        if st.button(
            "INITIATE ANALYSIS →",
            type="primary",
            width="stretch",
            disabled=not st.session_state.get("trige_upload_bytes"),
        ):
            begin_analysis()

        if st.button("CLEAR CURRENT SCAN"):
            new_scan()

        if st.button(
            "PROJECT / VISION, ARCHITECTURE & LIMITATIONS",
            width="stretch",
            key="open_project_dialog",
        ):
            show_project_dialog()


# ------------------------------------------------------------------------
# Screen 2: result
# ------------------------------------------------------------------------

def render_loading():
    st.html(
        """
        <section class="trige-loading-screen">

            <div class="loading-grid"></div>
            <div class="loading-scan"></div>

            <div class="loading-core">

                <div class="loading-orbit">

                    <div class="loading-ring loading-ring-1"></div>
                    <div class="loading-ring loading-ring-2"></div>
                    <div class="loading-ring loading-ring-3"></div>

                    <div class="loading-center">

                        <div class="loading-label">
                            TRIGE // ANALYSIS ENGINE
                        </div>

                        <div class="loading-title">
                            ANALYZING
                        </div>

                        <div class="loading-subtitle">
                            CHEST RADIOGRAPH
                        </div>

                    </div>

                </div>

                <div class="loading-status">

                    <span>
                        <i></i>
                        MODEL INFERENCE ACTIVE
                    </span>

                    <span>DENSENET121</span>
                    <span>STATIC IMAGING</span>

                </div>

                <div class="loading-message">
                    Processing the uploaded radiograph and generating model outputs.
                </div>

                <div class="loading-caution">
                    Please wait. Analysis time may vary depending on available hardware.
                </div>

            </div>

        </section>
        """
    )

    # Give Streamlit/browser a short moment to paint the loading screen.
    # The CSS animation then continues in the browser while Python inference blocks.
    time.sleep(0.20)

    existing_error = st.session_state.get("analysis_error")
    if existing_error:
        st.error(f"Analysis failed: {existing_error}")

        if st.button("← RETURN TO INTAKE", key="loading_return_after_error"):
            st.session_state.pop("analysis_error", None)
            st.session_state.pop("pending_analysis", None)
            navigate("intake")
        return

    completed = complete_pending_analysis()

    if completed is False:
        error = st.session_state.get("analysis_error", "Unknown analysis error.")
        st.error(f"Analysis failed: {error}")

        if st.button("← RETURN TO INTAKE", key="loading_return_after_failure"):
            st.session_state.pop("analysis_error", None)
            st.session_state.pop("pending_analysis", None)
            navigate("intake")

def render_result():
    result = st.session_state.trige_result
    heading("03 / RESULT", "STATIC IMAGING OUTPUT")
    navigation(
        "← NEW SCAN",
        "new",
        "UNDERSTAND RESULT →",
        "explanation",
    )
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        display_image(result["image_path"], "ORIGINAL RADIOGRAPH")
        st.caption(result.get("filename", "Uploaded radiograph"))
    with right:
        st.html(
            '<div class="score-panel"><div class="instrument-label">'
            "PNEUMONIA-ASSOCIATED MODEL SCORE</div>"
            f'<div class="score-reading">{result["score"]:.3f}</div>'
            '<div class="instrument-label">RAW CLASSIFIER OUTPUT</div></div>'
        )
        metadata(
            [
                ("DEMO FLAG", result["flag"]),
                ("MODEL", "DenseNet121"),
                ("WEIGHTS", "res224-all"),
                ("STATUS", "COMPLETE"),
            ]
        )
        st.caption(
            "This is a research model output and is not a clinically calibrated probability or diagnosis."
        )
        st.caption("Demo threshold: 0.50. This is not a clinically validated cutoff.")
        if result.get("save_error"):
            st.warning(
                "Analysis completed, but database saving failed: "
                + result["save_error"]
            )


# ------------------------------------------------------------------------
# Screen 3: explain
# ------------------------------------------------------------------------

def render_explanation():
    result = (
        st.session_state.trige_result
    )

    heading(
        "04 / EXPLANATION",
        "UNDERSTANDING THE RESULT",
    )

    navigation(
        "← RESULT",
        "result",
        "VIEW RECORD →",
        "record",
    )

    # ========================================================
    # MODEL ATTENTION
    # ========================================================

    st.html(
        """
        <div class="interpretation-heading">

            <div class="trige-section-label">
                MODEL EXPLAINABILITY
            </div>

            <h2>
                What influenced the model?
            </h2>

            <p>
                The images below show the original chest
                radiograph beside a visualization of the
                regions that had more influence on the
                pneumonia-associated model output.
            </p>

        </div>
        """
    )

    try:
        prepare_views(result)

        original_col, attention_col = (
            st.columns(
                2,
                gap="large",
            )
        )

        with original_col:
            display_image(
                result["display_original"],
                "ORIGINAL RADIOGRAPH",
            )

        with attention_col:
            display_image(
                result[
                    "attention_overlay"
                ],
                "MODEL ATTENTION MAP",
            )

    except Exception as error:
        st.warning(
            "Attention visualization "
            f"unavailable: {error}"
        )

    st.html(
        """
        <div class="attention-plain-language">

            <strong>
                What does this map mean?
            </strong>

            <p>
                Brighter highlighted areas had more influence
                on the model's output.
            </p>

            <p>
                This does not mean that pneumonia, infection,
                or another disease is located in those areas.
                The map explains model attention, not confirmed
                disease location.
            </p>

        </div>
        """
    )

    # ========================================================
    # SCORE EXPLANATION
    # ========================================================

    score = result["score"]

    st.html(
        f"""
        <div class="interpretation-score-card">

            <div class="interpretation-score-label">
                PNEUMONIA-ASSOCIATED MODEL SCORE
            </div>

            <div class="interpretation-score-value">
                {score:.3f}
            </div>

            <div class="interpretation-score-explanation">
                This number shows how strongly the pretrained
                model responded to image patterns connected
                with its Pneumonia output.
            </div>

            <div class="interpretation-score-warning">
                This is not the percentage chance that someone
                has pneumonia and it is not a diagnosis.
            </div>

        </div>
        """
    )

    # ========================================================
    # SUPPORTING MODEL OUTPUTS
    # ========================================================

    st.html(
        """
        <div class="interpretation-heading">

            <div class="trige-section-label">
                ADDITIONAL MODEL OUTPUTS
            </div>

            <h2>
                What about the other bars?
            </h2>

            <p>
                The pretrained model produces several outputs
                at the same time. The bars below show selected
                raw model scores.
            </p>

        </div>
        """
    )

    values = supporting_values(result)

    with st.container(border=True):
        for name, value in values:

            st.html(
                f"""
                <div class="output-row">
                    <span>
                        {escape(name)}
                    </span>

                    <strong>
                        {value:.3f}
                    </strong>
                </div>
                """
            )

            if 0 <= value <= 1:
                st.progress(value)

    st.caption(
        "Longer bars represent larger raw outputs from "
        "the pretrained model. They are not percentages "
        "and do not confirm that a condition is present."
    )

    # ========================================================
    # REGIONAL ATTENTION
    # ========================================================

    try:
        primary, secondary = (
            regional_focus(result)
        )

        st.html(
            f"""
            <div class="plain-result-card">

                <div class="plain-result-icon">
                    01
                </div>

                <div class="plain-result-content">

                    <div class="plain-result-title">
                        Where did the model focus?
                    </div>

                    <div class="plain-result-body">
                        The strongest average attention was
                        found in
                        <strong>
                            {escape(primary.lower())}
                        </strong>.

                        The next strongest region was
                        <strong>
                            {escape(secondary.lower())}
                        </strong>.
                    </div>

                    <div class="plain-result-note">
                        These are image display regions only.
                        They do not identify confirmed disease.
                    </div>

                </div>

            </div>
            """
        )

    except Exception:
        pass

    # ========================================================
    # OVERALL INTERPRETATION
    # ========================================================

    st.html(
        """
        <div class="plain-result-card important-result-card">

            <div class="plain-result-icon">
                02
            </div>

            <div class="plain-result-content">

                <div class="plain-result-title">
                    What does this mean overall?
                </div>

                <div class="plain-result-body">
                    Trige found image patterns that caused the
                    pretrained model to produce the outputs
                    shown above.
                </div>

                <div class="plain-result-note">
                    These results describe model behavior.
                    They cannot determine whether someone
                    actually has pneumonia or another medical
                    condition.
                </div>

            </div>

        </div>
        """
    )

    # ========================================================
    # HOW IT WORKS POPUP
    # ========================================================

    if st.button(
        "HOW TRIGE GENERATED THIS RESULT →",
        width="stretch",
        key="open_generation_dialog",
    ):
        show_generation_dialog()

    # ========================================================
    # NEXT STEPS LOWER DOWN PAGE
    # ========================================================

    st.html(
        """
        <div class="interpretation-scroll-break">

            <div class="scroll-break-line"></div>

            <div class="scroll-break-text">
                NEXT / USING THIS RESULT
            </div>

            <div class="scroll-break-arrow">
                ↓
            </div>

        </div>
        """
    )

    st.html(
        """
        <section class="next-steps-section">

            <div class="trige-section-label">
                NEXT STEPS
            </div>

            <h2>
                What should I do with this result?
            </h2>

            <p class="next-steps-lead">
                Trige cannot determine whether a person has
                pneumonia and should not be used to make
                treatment decisions.
            </p>

            <div class="next-step-card">

                <div class="next-step-number">
                    01
                </div>

                <div>
                    <h3>
                        Have the image professionally interpreted
                    </h3>

                    <p>
                        A real chest radiograph should be reviewed
                        by an appropriate healthcare professional
                        together with symptoms, medical history,
                        and other clinical information.
                    </p>
                </div>

            </div>

            <div class="next-step-card">

                <div class="next-step-number">
                    02
                </div>

                <div>
                    <h3>
                        Don't treat the score as a percentage
                    </h3>

                    <p>
                        A score such as 0.518 does not mean a
                        51.8% chance of pneumonia.
                    </p>
                </div>

            </div>

            <div class="next-step-card">

                <div class="next-step-number">
                    03
                </div>

                <div>
                    <h3>
                        Don't change treatment because of Trige
                    </h3>

                    <p>
                        Do not start, stop, or change medications
                        or treatment based on this research
                        prototype.
                    </p>
                </div>

            </div>

        </section>
        """
    )

# ------------------------------------------------------------------------
# Deterministic summary
# ------------------------------------------------------------------------


def summary_paragraphs(result):
    paragraphs = [
        "The pretrained DenseNet121 classifier generated a pneumonia-associated "
        f"model score of {result['score']:.3f}."
    ]
    values = supporting_values(result)
    if values:
        maximum = values[0][1]
        names = [name for name, value in values if value == maximum]
        verb = "produced" if len(names) == 1 else "shared"
        paragraphs.append(
            f"Among the displayed supporting outputs, {', '.join(names)} "
            f"{verb} the largest model score at {maximum:.3f}."
        )
    else:
        paragraphs.append("Supporting outputs are unavailable for this analysis.")
    try:
        primary, secondary = regional_focus(result)
        if primary == "No distinct regional focus":
            paragraphs.append(
                "Regional mean attention was equal or nearly equal; no distinct "
                "primary or secondary regional focus could be identified."
            )
        elif "(tied)" in primary or "(tied)" in secondary:
            paragraphs.append(
                f"The regional summary lists {primary} as primary model focus and "
                f"{secondary} as secondary model focus. Tied regions have equal or "
                "nearly equal means; their display order does not indicate stronger attention."
            )
        else:
            paragraphs.append(
                f"The highest regional mean Grad-CAM intensity was in the {primary.lower()}, "
                f"with the second highest in the {secondary.lower()}."
            )
    except (KeyError, TypeError, ValueError):
        paragraphs.append("Regional attention is unavailable for this analysis.")
    return paragraphs


# ------------------------------------------------------------------------
# Screen 4: interpretation
# ------------------------------------------------------------------------

@st.dialog("HOW TRIGE GENERATED THIS RESULT", width="large")
def show_generation_dialog():
    st.html(
        """
        <div class="generation-dialog-intro">
            <div class="trige-section-label">TRIGE / MODEL PROCESS</div>

            <h3>How the analysis was generated</h3>

            <p>
                Trige processes the uploaded chest radiograph through several
                steps before displaying the model output and attention map.
            </p>
        </div>
        """
    )

    steps = [
        (
            "01",
            "IMAGE PREPARATION",
            "Trige loads the chest radiograph and converts it into the format "
            "expected by the pretrained model.",
        ),
        (
            "02",
            "STANDARDIZATION",
            "The image is normalized, center-cropped, and resized to "
            "224 × 224 pixels so it matches the model's expected input.",
        ),
        (
            "03",
            "DENSENET121 ANALYSIS",
            "The pretrained DenseNet121 network examines visual patterns in "
            "the processed image and produces several chest X-ray model outputs.",
        ),
        (
            "04",
            "MODEL SCORE",
            "Trige selects the model output associated with the Pneumonia label. "
            "This is shown as the pneumonia-associated model score.",
        ),
        (
            "05",
            "MODEL ATTENTION",
            "Grad-CAM estimates which parts of the image influenced that model "
            "output more strongly. Trige displays this as the Model Attention Map.",
        ),
        (
            "06",
            "RESULT SUMMARY",
            "Trige organizes the raw model outputs, attention information, "
            "and analysis metadata into the interface you are viewing.",
        ),
    ]

    for number, title, text in steps:
        st.html(
            f"""
            <div class="generation-step">
                <div class="generation-number">{number}</div>

                <div class="generation-content">
                    <div class="generation-title">{title}</div>
                    <div class="generation-text">{text}</div>
                </div>
            </div>
            """
        )

    st.html(
        """
        <div class="generation-warning">
            These steps describe how the software produces its output.
            They do not establish whether pneumonia or another disease is present.
        </div>
        """
    )


# ------------------------------------------------------------------------
# Screen 5: record
# ------------------------------------------------------------------------


def render_record():
    result = st.session_state.trige_result
    heading("05 / RECORD", "TRIGE CASE RECORD")
    navigation("← EXPLANATION", "explanation", "VIEW ARCHIVE →", "history")
    row = result.get("record")
    with st.container(border=True):
        metadata(
            [
                ("CASE", f"TRG-{int(row[0]):05d}" if row else "Unavailable"),
                ("ANALYZED", row[5] if row else "Unavailable"),
                ("FILE", result.get("filename", "Unavailable")),
                ("MODEL", row[4] if row else "DenseNet121 / res224-all"),
                ("MODULE", "Static Imaging"),
                ("PNEUMONIA-ASSOCIATED MODEL SCORE", f"{result['score']:.3f}"),
                ("DEMO FLAG", result["flag"]),
                ("ANALYSIS STATUS", "COMPLETE"),
                ("DATABASE", "Saved" if result.get("saved") else "Save not confirmed"),
            ]
        )
    if not row:
        st.caption(
            "A unique SQLite record could not be confirmed for this session. "
            "No case ID or timestamp has been inferred; check the archive."
        )
    if result.get("save_error"):
        st.warning(result["save_error"])
    if st.button("START NEW ANALYSIS", type="primary"):
        new_scan()


# ------------------------------------------------------------------------
# Screen 6: archive
# ------------------------------------------------------------------------


def render_history():
    heading("ARCHIVE / LOCAL SQLITE", "ANALYSIS ARCHIVE")
    current = bool(st.session_state.get("trige_result"))
    if st.button("← RETURN TO CURRENT CASE" if current else "← RETURN TO INTAKE"):
        navigate("result" if current else "intake")
    try:
        history = pd.DataFrame(
            get_scans(),
            columns=["ID", "File", "Model Score", "Demo Flag", "Model", "Timestamp"],
        )
        history["_sort"] = pd.to_datetime(
            history["Timestamp"], errors="coerce", utc=True
        )
        history = history.sort_values(
            ["_sort", "ID"], ascending=False, na_position="last"
        ).reset_index(drop=True)
    except Exception as error:
        st.error(f"Could not load scan history: {error}")
        return
    a, b, c = st.columns([1, 1, 2])
    a.metric("TOTAL ANALYSES", len(history))
    b.metric("ELEVATED FLAGS", int(history["Demo Flag"].eq("ELEVATED").sum()))
    latest = (
        str(history.loc[0, "Timestamp"])
        if not history.empty and pd.notna(history.loc[0, "Timestamp"])
        else "--"
    )
    c.metric("LATEST ANALYSIS", latest)
    history = history.drop(columns="_sort")
    history["Model Score"] = history["Model Score"].map(
        lambda v: f"{v:.3f}" if pd.notna(v) else None
    )
    st.dataframe(history, width="stretch", hide_index=True, height=320)
    st.caption(
        "Model Score / Pneumonia-associated model score. Demo flags are not clinical classifications."
    )
    if history.empty:
        st.info("No scans have been analyzed yet.")


# ------------------------------------------------------------------------
# Main screen routing
# ------------------------------------------------------------------------

if st.session_state.screen not in SCREENS:
    st.session_state.screen = "title"

# The loading screen happens before a result exists, so it is valid as long
# as there is a prepared image waiting to be analyzed.
if (
    st.session_state.screen == "loading"
    and not st.session_state.get("pending_analysis")
):
    st.session_state.screen = "intake"

# These screens require a completed model result.
if (
    st.session_state.screen in ("result", "explanation", "record")
    and not st.session_state.get("trige_result")
):
    st.session_state.screen = "intake"


renderers = {
    "title": render_title,
    "intake": render_intake,
    "loading": render_loading,
    "result": render_result,
    "explanation": render_explanation,
    "record": render_record,
    "history": render_history,
}


if st.session_state.screen == "title":
    render_title()

else:
    render_header()
    renderers[st.session_state.screen]()

    st.html(
        '<footer class="instrument-footer">'
        "TRIGE v0.1 / STATIC IMAGING RESEARCH PROTOTYPE"
        "</footer>"
    )
