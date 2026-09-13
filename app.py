from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from database import (
    initialize_database,
    save_scan,
    get_scans,
)
from xray_model import predict_xray


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

initialize_database()


st.set_page_config(
    page_title="Trige",
    page_icon="🩻",
    layout="wide"
)


st.title("TRIGE")
st.caption("Multimodal Temporal-Spatial Clinical Triage Engine")

st.subheader("v0.1 — Static Imaging Prototype")

st.warning(
    "Research prototype only. The model score is not a medical diagnosis "
    "or clinically validated probability. Do not use this application "
    "for patient-care decisions."
)


uploaded_file = st.file_uploader(
    "Upload a de-identified chest X-ray",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    left, right = st.columns(2)

    with left:
        st.subheader("Chest X-Ray")

        st.image(
            uploaded_file,
            use_container_width=True
        )

    if st.button(
        "Analyze X-Ray",
        type="primary"
    ):

        suffix = Path(uploaded_file.name).suffix.lower()

        unique_filename = (
            f"{uuid.uuid4().hex}{suffix}"
        )

        image_path = (
            UPLOAD_DIR / unique_filename
        )

        with open(image_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        try:

            with st.spinner(
                "Running chest X-ray model..."
            ):

                score = predict_xray(
                    image_path
                )

            # Demo threshold only.
            threshold = 0.50

            if score >= threshold:
                flag = "ELEVATED"
            else:
                flag = "LOW"

            save_scan(
                uploaded_file.name,
                score,
                flag
            )

            with right:
                st.subheader(
                    "Model Analysis"
                )

                st.metric(
                    "Pneumonia-associated score",
                    f"{score:.3f}"
                )

                st.metric(
                    "Demo alert flag",
                    flag
                )

                st.caption(
                    "The 0.50 alert threshold is "
                    "used only for this prototype "
                    "and is not a clinically "
                    "validated diagnostic cutoff."
                )

            st.success(
                "Analysis completed and "
                "saved to the scan database."
            )

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )


st.divider()

st.subheader("Scan History")

rows = get_scans()

if rows:

    history = pd.DataFrame(
        rows,
        columns=[
            "ID",
            "Filename",
            "Pneumonia Score",
            "Alert Flag",
            "Model",
            "Created At",
        ],
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info(
        "No scans have been analyzed yet."
    )