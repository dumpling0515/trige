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


# Application setup
st.set_page_config(page_title="Trige", page_icon="🩻", layout="wide")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
initialize_database()

st.html(Path("styles.css"))


def clear_analysis():
    """Prevent a previous result from appearing beside a new upload."""
    st.session_state.pop("trige_result", None)


def clear_current_scan():
    """Reset only the current upload widget and displayed analysis."""
    clear_analysis()
    # A new widget key creates an empty uploader on the automatic rerun.
    st.session_state["trige_upload_version"] = (
        st.session_state.get("trige_upload_version", 0) + 1
    )


# Hero and research notice
st.html(
    """
    <section class="trige-hero">
        <div class="trige-kicker">
            TEMPORAL-SPATIAL CLINICAL TRIAGE ENGINE
        </div>
        <h1 class="trige-title">TRIGE</h1>
        <p class="trige-subtitle">Static Imaging Module · v0.1</p>
        <p>
            A research prototype exploring AI-assisted analysis of
            chest radiographs as the first stage of the Trige
            multimodal triage architecture.
        </p>
        <div class="trige-status-row">
            <span class="trige-pill">
                <span class="trige-dot"></span> SYSTEM ONLINE
            </span>
            <span class="trige-pill">DenseNet121</span>
            <span class="trige-pill">STATIC IMAGING · v0.1</span>
        </div>
    </section>
    <div class="trige-disclaimer">
        <strong>Research prototype only. Not clinically validated.
        Not for diagnosis or patient-care decisions.</strong>
        <br>Only upload de-identified demonstration images.
    </div>
    """
)

analyze_tab, history_tab, project_tab = st.tabs(
    ["ANALYZE", "SCAN HISTORY", "PROJECT"]
)


# Analyze
with analyze_tab:
    st.html(
        '<div class="trige-section-label">CHEST RADIOGRAPH ANALYSIS</div>'
    )
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        uploaded_file = st.file_uploader(
            "Upload a de-identified chest X-ray",
            type=["png", "jpg", "jpeg"],
            key=f"trige_upload_{st.session_state.get('trige_upload_version', 0)}",
            on_change=clear_analysis,
        )

        if uploaded_file is not None:
            st.image(uploaded_file, width="stretch")
            st.text(uploaded_file.name)

        analyze_clicked = st.button(
            "RUN TRIGE ANALYSIS",
            type="primary",
            width="stretch",
            disabled=uploaded_file is None,
        )

        st.button(
            "CLEAR CURRENT SCAN",
            type="secondary",
            on_click=clear_current_scan,
        )

    if analyze_clicked and uploaded_file is not None:
        clear_analysis()
        try:
            suffix = Path(uploaded_file.name).suffix.lower()
            unique_filename = f"{uuid.uuid4().hex}{suffix}"
            image_path = UPLOAD_DIR / unique_filename

            with open(image_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            with st.spinner("Running chest X-ray model..."):
                score, all_pathology_scores, attention_map = predict_xray(image_path)

            # Preserve the existing demonstration threshold and database call.
            threshold = 0.50
            flag = "ELEVATED" if score >= threshold else "LOW"
            save_scan(uploaded_file.name, score, flag)

            # Store presentation state only; reruns do not repeat inference/saving.
            st.session_state["trige_result"] = {
                "score": float(score),
                "flag": flag,
                "all_pathology_scores": all_pathology_scores,
                "attention_map": attention_map,
                "image_path": str(image_path),
            }
        except Exception as error:
            st.error(f"Analysis or database saving failed: {error}")

    with right:
        result = st.session_state.get("trige_result")
        if uploaded_file is None or result is None:
            st.html(
                """
                <div class="trige-result">
                    <div class="trige-score-label">AI ANALYSIS</div>
                    <h3>Awaiting radiograph</h3>
                    <p class="trige-score-unit">
                        Upload a chest X-ray and initiate analysis to generate
                        a pneumonia-associated model score.
                    </p>
                </div>
                """
            )
        else:
            st.html(
                f"""
                <div class="trige-result">
                    <div class="trige-score-label">
                        PNEUMONIA-ASSOCIATED MODEL SCORE
                    </div>
                    <div class="trige-score">{result['score']:.3f}</div>
                    <div class="trige-score-unit">Raw model output</div>
                    <div class="trige-info-grid">
                        <div class="trige-info-item">
                            <div class="trige-info-name">DEMO ALERT</div>
                            <div class="trige-info-value">{result['flag']}</div>
                        </div>
                        <div class="trige-info-item">
                            <div class="trige-info-name">MODEL</div>
                            <div class="trige-info-value">DenseNet121</div>
                        </div>
                        <div class="trige-info-item">
                            <div class="trige-info-name">WEIGHTS</div>
                            <div class="trige-info-value">res224-all</div>
                        </div>
                        <div class="trige-info-item">
                            <div class="trige-info-name">INPUT</div>
                            <div class="trige-info-value">Chest radiograph</div>
                        </div>
                        <div class="trige-info-item">
                            <div class="trige-info-name">STATUS</div>
                            <div class="trige-info-value">Analysis complete</div>
                        </div>
                    </div>
                    <p class="trige-score-unit">
                        The displayed value is a research model output and is
                        not a clinically calibrated probability or diagnosis.
                    </p>
                </div>
                """
            )
            st.caption(
                "The 0.50 threshold is used only to demonstrate the interface. "
                "It is not a clinically validated diagnostic cutoff."
            )
            st.success("Analysis completed and saved to the local scan database.")

    # Display-only processing is separate from inference and database saving.
    result = st.session_state.get("trige_result")
    if uploaded_file is not None and result is not None:
        if "attention_map" in result and "image_path" in result:
            st.divider()
            st.html(
                '<div class="trige-section-label">MODEL ATTENTION MAP</div>'
            )
            try:
                if "attention_overlay" not in result:
                    original, overlay = prepare_attention_views(
                        result["image_path"], result["attention_map"]
                    )
                    result["display_original"] = original
                    result["attention_overlay"] = overlay

                original_col, attention_col = st.columns(2, gap="large")
                with original_col:
                    with st.container(border=True):
                        st.subheader("Original Radiograph")
                        st.image(result["display_original"], width="stretch")
                        st.caption("Center-cropped model view · 224 × 224")
                with attention_col:
                    with st.container(border=True):
                        st.subheader("Model Attention Map")
                        st.image(result["attention_overlay"], width="stretch")
                        st.caption("Inferno overlay · maximum opacity 45%")
                        st.caption(
                            "Highlighted regions indicate areas that contributed more "
                            "strongly to the pneumonia-associated model output. This is "
                            "an explainability visualization, not a confirmed location "
                            "of disease."
                        )
                st.caption(
                    "Both panels use the same center-cropped 224 × 224 view for "
                    "alignment. The full uploaded radiograph remains above. "
                    "Attention intensity is normalized per scan and should not "
                    "be compared across scans."
                )
                ranked_regions = rank_attention_regions(result["attention_map"])
                primary_focus, secondary_focus = attention_focus_labels(ranked_regions)
                primary_col, secondary_col = st.columns(2, gap="large")
                with primary_col:
                    with st.container(border=True):
                        st.html(
                            '<div class="trige-section-label">PRIMARY MODEL FOCUS</div>'
                        )
                        st.write(primary_focus)
                with secondary_col:
                    with st.container(border=True):
                        st.html(
                            '<div class="trige-section-label">SECONDARY MODEL FOCUS</div>'
                        )
                        st.write(secondary_focus)
                st.caption(
                    "Regional attention describes where the model concentrated "
                    "when generating its output. Attention does not establish "
                    "the presence or anatomical location of disease."
                )
                if result["attention_map"].max() == 0:
                    st.caption(
                        "No spatial attribution contrast was obtained; the overlay "
                        "shows the unchanged grayscale view. This does not establish "
                        "absence of disease."
                    )
            except Exception as error:
                st.warning(
                    f"Attention visualization could not be displayed: {error}. "
                    "The analysis score and saved database record are unaffected."
                )


    # Supporting outputs are display-only and come directly from this analysis.
    if uploaded_file is not None and result is not None:
        st.divider()
        st.html(
            '<div class="trige-section-label">SUPPORTING MODEL OUTPUTS</div>'
        )
        st.caption(
            "These values are additional outputs generated by the "
            "pretrained chest X-ray classifier. They provide context for "
            "the primary pneumonia-associated output but do not constitute "
            "diagnoses."
        )
        selected_names = (
            "Pneumonia", "Lung Opacity", "Consolidation", "Infiltration", "Effusion"
        )
        pathology_scores = result.get("all_pathology_scores", {})
        supporting_outputs = []
        unavailable_outputs = []
        for name in selected_names:
            try:
                value = float(pathology_scores[name])
                if not isfinite(value):
                    raise ValueError("Non-finite output")
            except (KeyError, TypeError, ValueError, OverflowError):
                unavailable_outputs.append(name)
            else:
                supporting_outputs.append((name, value))
        supporting_outputs.sort(key=lambda item: item[1], reverse=True)

        with st.container(border=True):
            for name, value in supporting_outputs:
                name_col, value_col = st.columns([3, 1])
                name_col.write(name)
                value_col.write(f"{value:.3f}")
                # Do not clip or rescale scores to fit a bar.
                if 0.0 <= value <= 1.0:
                    st.progress(value)
            for name in unavailable_outputs:
                st.caption(f"{name} — unavailable")
        st.caption(
            "Raw model outputs · bars use a 0–1 scale where applicable; "
            "they do not represent probabilities."
        )


        st.divider()
        st.html('<div class="trige-section-label">ANALYSIS SUMMARY</div>')
        with st.container(border=True):
            st.write(
                "The pretrained DenseNet121 classifier generated a "
                f"pneumonia-associated model score of {result['score']:.3f}."
            )
            if supporting_outputs:
                highest_score = supporting_outputs[0][1]
                highest_names = [
                    name for name, value in supporting_outputs
                    if value == highest_score
                ]
                if len(highest_names) == 1:
                    st.write(
                        "Among the displayed supporting outputs, "
                        f"{highest_names[0]} produced the largest model score "
                        f"at {highest_score:.3f}."
                    )
                else:
                    st.write(
                        "Among the displayed supporting outputs, "
                        + ", ".join(highest_names)
                        + f" shared the largest model score at {highest_score:.3f}."
                    )
            else:
                st.write("Supporting model outputs are unavailable for this analysis.")

            # Compute from the current map, independently of overlay rendering.
            try:
                summary_regions = rank_attention_regions(result["attention_map"])
                summary_primary, summary_secondary = attention_focus_labels(summary_regions)
            except (KeyError, TypeError, ValueError):
                st.write("Regional attention is unavailable for this analysis.")
            else:
                if summary_primary == "No distinct regional focus":
                    st.write(
                        "The six regions had equal or nearly equal mean Grad-CAM "
                        "intensity; no distinct primary or secondary regional "
                        "focus could be identified."
                    )
                elif "(tied)" in summary_primary or "(tied)" in summary_secondary:
                    st.write(
                        "The regional Grad-CAM summary lists "
                        f"{summary_primary} as primary model focus and "
                        f"{summary_secondary} as secondary model focus. "
                        "Tied labels indicate equal or nearly equal regional means; "
                        "their display order does not indicate stronger attention."
                    )
                else:
                    st.write(
                        "The Grad-CAM attention map had its highest regional mean "
                        f"intensity in the {summary_primary.lower()}, with the "
                        f"second highest in the {summary_secondary.lower()}."
                    )
            st.write(
                "These results describe classifier behavior and model "
                "attention. They do not confirm pneumonia or establish the "
                "anatomical location of infection."
            )


    with st.expander("HOW THE ANALYSIS IS GENERATED", expanded=False):
        analysis_steps = (
            ("IMAGE PREPARATION",
             "Chest radiograph loaded and normalized using the "
             "TorchXRayVision preprocessing standard."),
            ("STANDARDIZATION",
             "Image center-cropped and resized to 224 × 224 pixels."),
            ("FEATURE EXTRACTION",
             "DenseNet121 processes image patterns through its "
             "convolutional feature layers."),
            ("CLASSIFIER OUTPUTS",
             "The network produces scores for supported chest "
             "radiograph findings."),
            ("PNEUMONIA OUTPUT",
             "Trige extracts the classifier's Pneumonia output."),
            ("EXPLAINABILITY",
             "Grad-CAM estimates which image regions had greater "
             "positive influence on the pneumonia-associated model output."),
        )
        for step_number, (title, description) in enumerate(analysis_steps, start=1):
            with st.container(border=True):
                st.html(
                    f'<div class="trige-section-label">{step_number} / {title}</div>'
                )
                st.write(description)
            if step_number < len(analysis_steps):
                st.html(
                    '<div aria-hidden="true" style="text-align:center; '
                    'color:#78cfe2; font-size:1.25rem; padding:0.25rem;">↓</div>'
                )
        st.caption(
            "The Model Attention Map describes classifier behavior. "
            "It does not confirm disease or establish its anatomical location."
        )


    st.divider()
    st.html('<div class="trige-section-label">NEXT STEPS</div>')
    with st.container(border=True):
        st.markdown(
            """
Trige cannot determine whether a patient has pneumonia or decide treatment.

If this represents a real patient's radiograph:

- Have the image interpreted by a licensed healthcare professional together with symptoms and clinical history.
- Seek urgent/emergency medical care for severe or rapidly worsening breathing difficulty, fainting, confusion, bluish/gray lips or skin, or other severe symptoms.
- Do not start, stop, or change treatment based on this research prototype.
            """
        )


# Scan history: all summaries come from the existing database results.
with history_tab:
    st.html('<div class="trige-section-label">LOCAL DATABASE</div>')
    st.subheader("Scan History")

    try:
        rows = get_scans()
        history = pd.DataFrame(
            rows,
            columns=["ID", "File", "Model Score", "Demo Flag", "Model", "Timestamp"],
        )
    except Exception as error:
        st.error(f"Could not load scan history: {error}")
    else:
        # Sort using parsed timestamps while retaining the database's display text.
        # ID breaks ties when records share the same timestamp.
        history["_sort_timestamp"] = pd.to_datetime(
            history["Timestamp"], errors="coerce", utc=True
        )
        history = history.sort_values(
            ["_sort_timestamp", "ID"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

        total_analyses = len(history)
        elevated_flags = int(history["Demo Flag"].eq("ELEVATED").sum())
        latest_analysis = "--"
        if not history.empty and pd.notna(history.loc[0, "Timestamp"]):
            latest_analysis = str(history.loc[0, "Timestamp"])

        total_col, elevated_col, recent_col = st.columns([1, 1, 2])
        with total_col:
            with st.container(border=True):
                st.metric("TOTAL ANALYSES", total_analyses)
        with elevated_col:
            with st.container(border=True):
                st.metric("ELEVATED FLAGS", elevated_flags)
        with recent_col:
            with st.container(border=True):
                st.metric("LATEST ANALYSIS", latest_analysis)

        st.caption(
            "Model Score: Pneumonia-associated model score. "
            "Elevated flags use the demonstration threshold only."
        )
        history = history.drop(columns="_sort_timestamp")
        history["Model Score"] = history["Model Score"].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else None
        )
        st.dataframe(
            history,
            width="stretch",
            hide_index=True,
        )
        if history.empty:
            st.info("No scans have been analyzed yet.")


# Project: a transparent overview of implemented and planned capabilities.
with project_tab:
    st.html(
        """
        <style>
            .trige-project-flow {
                max-width: 720px;
                margin: 1.25rem auto;
                text-align: center;
            }
            .trige-project-node {
                padding: 0.85rem 1rem;
                border: 1px solid rgba(103, 207, 235, 0.22);
                border-radius: 12px;
                background: linear-gradient(
                    135deg, rgba(30, 86, 113, 0.18), rgba(65, 51, 111, 0.16)
                );
                color: #e2edf7;
                line-height: 1.5;
            }
            .trige-project-arrow {
                padding: 0.3rem;
                color: #78cfe2;
                font-size: 1.2rem;
                text-align: center;
            }
            .trige-project-branches {
                display: grid;
                grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                gap: 1rem;
            }
            .trige-project-join {
                height: 22px;
                width: calc(50% + 0.5rem);
                margin: 0 auto;
                border: 1px solid rgba(120, 207, 226, 0.65);
                border-top: none;
            }
            .trige-project-tech {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
                gap: 0.75rem;
                margin: 1rem 0;
            }
        </style>
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
        """
    )

    st.divider()
    st.html(
        """
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
        """
    )
    st.caption(
        "The current module integrates pretrained densenet121-res224-all "
        "weights. Its output is not a diagnosis or a clinically calibrated "
        "probability."
    )

    st.divider()
    st.html(
        """
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
        """
    )
    st.info(
        "This is a proposed research architecture. Processing 48-hour vitals, "
        "temporal encoding, multimodal fusion, and deterioration-risk "
        "prediction are not implemented in v0.1."
    )

    st.divider()
    st.html(
        """
        <div class="trige-section-label">04 / TECHNOLOGY</div>
        <div class="trige-project-tech">
            <div class="trige-project-node">PyTorch</div>
            <div class="trige-project-node">TorchXRayVision</div>
            <div class="trige-project-node">DenseNet121</div>
            <div class="trige-project-node">Streamlit</div>
            <div class="trige-project-node">SQLite</div>
            <div class="trige-project-node">Python</div>
        </div>
        """
    )

    st.divider()
    st.html('<div class="trige-section-label">05 / LIMITATIONS</div>')
    with st.container(border=True):
        st.markdown(
            """
- **Pretrained model integration:** v0.1 uses existing model weights.
- **No original medical model training:** no medical model is trained in v0.1.
- **No clinical validation:** scores and demo flags are not validated for clinical use.
- **No real-time physiological data:** the current module analyzes static images only.
- **No multimodal fusion yet:** imaging and longitudinal vitals are not combined.
- **No clinical deployment:** v0.1 is not intended for diagnosis or patient-care decisions.
- **Research/demo purposes only:** the application demonstrates a research workflow.
            """
        )


st.html(
    """
    <footer class="trige-footer">
        TRIGE v0.1 · Static Imaging Research Prototype
    </footer>
    """
)
