from html import escape
from math import isfinite
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

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

SCREENS = ("intake", "result", "explain", "interpretation", "record", "history")
SUPPORTING = ("Pneumonia", "Lung Opacity", "Consolidation", "Infiltration", "Effusion")
st.session_state.setdefault("screen", "intake")
st.session_state.setdefault("trige_upload_version", 0)


# ------------------------------------------------------------------------
# Session state and navigation
# ------------------------------------------------------------------------


def navigate(screen):
    st.session_state.screen = screen
    st.rerun()


def new_scan():
    # Only current UI state is cleared. Never clear caches, files, or SQLite.
    for key in ("trige_result", "trige_upload_bytes", "trige_upload_name"):
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
        "result": 2,
        "explain": 3,
        "interpretation": 3,
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
        text = (
            "ANALYSIS COMPLETE"
            if st.session_state.get("trige_result")
            else "SYSTEM READY"
        )
        st.html(
            f'<div class="system-status"><span class="status-dot"></span>{text}</div>'
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
            navigate(left_screen)
    if right_text:
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


def run_analysis():
    name = st.session_state.trige_upload_name
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"
    try:
        path.write_bytes(st.session_state.trige_upload_bytes)
        # A snapshot permits matching the actual inserted row without inventing IDs.
        try:
            old_ids = {row[0] for row in get_scans()}
        except Exception:
            old_ids = None
        with st.status("PROCESSING RADIOGRAPH", expanded=True) as progress:
            st.caption(
                "NORMALIZATION / FEATURE EXTRACTION / CLASSIFICATION / EXPLAINABILITY"
            )
            # These labels describe the call; they are not fabricated live milestones.
            score, all_scores, attention_map = predict_xray(path)
            flag = "ELEVATED" if score >= 0.50 else "LOW"
            result = dict(
                score=float(score),
                flag=flag,
                all_pathology_scores=all_scores,
                attention_map=attention_map,
                image_path=str(path),
                filename=name,
                record=None,
                saved=False,
            )
            st.session_state.trige_result = result
            # Preserve the original three-argument write. Navigation never calls it.
            try:
                save_scan(name, score, flag)
                result["saved"] = True
            except Exception as error:
                result["save_error"] = str(error)
            if result["saved"] and old_ids is not None:
                try:
                    matches = [
                        tuple(row)
                        for row in get_scans()
                        if row[0] not in old_ids
                        and row[1] == name
                        and float(row[2]) == float(score)
                        and row[3] == flag
                    ]
                    if len(matches) == 1:
                        result["record"] = matches[0]
                except Exception:
                    pass  # A history read failure must not repeat the successful write.
            progress.update(label="ANALYSIS COMPLETE", state="complete", expanded=False)
    except Exception as error:
        st.error(f"Analysis failed: {error}")
        return
    navigate("result")


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
# Screen 1: intake
# ------------------------------------------------------------------------


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
            run_analysis()
        if st.button("CLEAR CURRENT SCAN"):
            new_scan()
        with st.expander("PROJECT / VISION, ARCHITECTURE & LIMITATIONS"):
            render_project_details()


# ------------------------------------------------------------------------
# Screen 2: result
# ------------------------------------------------------------------------


def render_result():
    result = st.session_state.trige_result
    heading("03 / RESULT", "STATIC IMAGING OUTPUT")
    navigation("← NEW SCAN", "new", "EXPLAIN RESULT →", "explain")
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


def render_explain():
    result = st.session_state.trige_result
    heading("04 / EXPLAIN", "MODEL ATTENTION MAP")
    navigation("← RESULT", "result", "INTERPRETATION →", "interpretation")
    visual, support = st.columns([2, 1], gap="large")
    with visual:
        try:
            prepare_views(result)
            left, right = st.columns(2)
            with left:
                display_image(result["display_original"], "ORIGINAL RADIOGRAPH")
            with right:
                display_image(result["attention_overlay"], "MODEL ATTENTION MAP")
            st.caption("Aligned center crop / 224 × 224 / Inferno overlay")
        except Exception as error:
            st.warning(f"Attention visualization unavailable: {error}")
        try:
            primary, secondary = regional_focus(result)
            metadata(
                [("PRIMARY MODEL FOCUS", primary), ("SECONDARY MODEL FOCUS", secondary)]
            )
        except (KeyError, TypeError, ValueError):
            st.caption("Regional attention unavailable.")
        st.caption(
            "Highlighted regions indicate areas that contributed more strongly to the "
            "pneumonia-associated model output. This is an explainability visualization, "
            "not a confirmed location of disease."
        )
        st.caption(
            "Regional attention describes where the model concentrated when generating "
            "its output. Attention does not establish the presence or anatomical "
            "location of disease."
        )
        st.caption(
            "Image-left/right are display coordinates, not patient anatomy. "
            "Attention is normalized per scan; a blank map does not establish absence of disease."
        )
    with support:
        label("SUPPORTING MODEL OUTPUTS")
        values = supporting_values(result)
        with st.container(border=True):
            for name, value in values:
                st.html(
                    f'<div class="output-row"><span>{escape(name)}</span>'
                    f"<strong>{value:.3f}</strong></div>"
                )
                if 0 <= value <= 1:
                    st.progress(value)
            for name in SUPPORTING:
                if name not in dict(values):
                    st.caption(f"{name} — unavailable")
        st.caption(
            "These values are additional outputs generated by the pretrained chest X-ray "
            "classifier. They provide context for the primary pneumonia-associated output "
            "but do not constitute diagnoses. Bars show raw outputs, not probabilities."
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


def render_interpretation():
    heading("04 / EXPLAIN / INTERPRETATION", "MODEL INTERPRETATION")
    navigation("← EXPLAIN", "explain", "VIEW RECORD →", "record")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        label("ANALYSIS SUMMARY")
        with st.container(border=True):
            for paragraph in summary_paragraphs(st.session_state.trige_result):
                st.write(paragraph)
        st.caption(
            "These outputs characterize model behavior. They do not constitute a radiological diagnosis."
        )
    with right:
        label("NEXT STEPS")
        st.write(
            "Trige cannot determine whether a patient has pneumonia or make treatment decisions. "
            "A real chest radiograph should be interpreted by an appropriate healthcare professional "
            "together with symptoms and clinical history."
        )
        st.caption(
            "Seek urgent/emergency medical care for severe or rapidly worsening breathing difficulty, "
            "fainting, confusion, bluish/gray lips or skin, or other severe symptoms."
        )
        st.caption(
            "Do not start, stop, or change treatment based on this research prototype."
        )
        with st.expander("HOW TRIGE GENERATED THIS RESULT"):
            label("MODEL PROCESS")
            for title, text in (
                (
                    "01 / IMAGE PREPARATION",
                    "Load the chest radiograph using TorchXRayVision.",
                ),
                (
                    "02 / NORMALIZATION / STANDARDIZATION",
                    "Normalize, center-crop, and resize to 224 × 224.",
                ),
                (
                    "03 / DENSENET121 FEATURE EXTRACTION",
                    "Process image patterns through convolutional feature layers.",
                ),
                (
                    "04 / PATHOLOGY CLASSIFIER OUTPUT",
                    "Generate classifier scores and extract the Pneumonia output.",
                ),
                (
                    "05 / GRAD-CAM EXPLAINABILITY",
                    "Estimate positive influence on the selected output using gradients.",
                ),
            ):
                st.markdown(f"**{title}**")
                st.caption(text)


# ------------------------------------------------------------------------
# Screen 5: record
# ------------------------------------------------------------------------


def render_record():
    result = st.session_state.trige_result
    heading("05 / RECORD", "TRIGE CASE RECORD")
    navigation("← INTERPRETATION", "interpretation", "VIEW ARCHIVE →", "history")
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


# Exactly one major screen is rendered per run.
if st.session_state.screen not in SCREENS:
    st.session_state.screen = "intake"
if st.session_state.screen not in ("intake", "history") and not st.session_state.get(
    "trige_result"
):
    st.session_state.screen = "intake"
render_header()
renderers = {
    "intake": render_intake,
    "result": render_result,
    "explain": render_explain,
    "interpretation": render_interpretation,
    "record": render_record,
    "history": render_history,
}
renderers[st.session_state.screen]()
st.html(
    '<footer class="instrument-footer">TRIGE v0.1 / STATIC IMAGING RESEARCH PROTOTYPE</footer>'
)
