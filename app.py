from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from database import initialize_database, save_scan, get_scans
from xray_model import predict_xray


# Application setup
st.set_page_config(
    page_title="Trige · Static Imaging",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
                score = predict_xray(image_path)

            # Preserve the existing demonstration threshold and database call.
            threshold = 0.50
            flag = "ELEVATED" if score >= threshold else "LOW"
            save_scan(uploaded_file.name, score, flag)

            # Store presentation state only; reruns do not repeat inference/saving.
            st.session_state["trige_result"] = {
                "score": float(score),
                "flag": flag,
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
