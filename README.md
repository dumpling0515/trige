TRIGE

Multimodal Temporal-Spatial Clinical Triage Engine
v0.1 — Static Imaging Module

Trige explores the integration of medical imaging and longitudinal physiological data in a research dashboard. The current release implements the static imaging layer: a Streamlit application that processes a chest radiograph with a pretrained model, displays a pneumonia-associated model score, and records analysis metadata locally.

Trige is a research and educational prototype. It has not been clinically validated and is not intended for diagnosis, treatment, triage decisions, or other patient-care use.

Dashboard



<!-- Add an actual screenshot of the dashboard at screenshots/trige_dashboard.png. Use only de-identified demonstration content. -->

What Trige is

Trige is a prototype for exploring how an imaging inference pipeline can be connected to a transparent research interface and local analysis history.

Its long-term vision is to combine a chest image with a 48-hour vital-sign trajectory, fuse information from both modalities, and investigate deterioration-risk prediction. Vital-sign processing, multimodal fusion, and deterioration-risk prediction are not implemented in v0.1.

The current release integrates an existing pretrained medical imaging model. It does not introduce an originally trained medical model or establish clinical performance.

Current v0.1 capabilities

Accept a de-identified chest X-ray in PNG, JPG, or JPEG format.

Preview the uploaded image and its filename.

Preprocess the image using TorchXRayVision.

Run pretrained DenseNet121 inference using densenet121-res224-all weights.

Extract and display the Pneumonia-associated model score, rounded to three decimal places without converting it to a percentage.

Display a LOW or ELEVATED demo flag using the existing 0.50 threshold.

Store the filename, model score, demo flag, model version, and timestamp in SQLite.

Display previous analyses and database-derived summary metrics in Streamlit.

The demo flag is an interface demonstration, not a validated clinical classification. The score is not a clinically calibrated probability of pneumonia or a diagnosis.

Architecture

Current — implemented in v0.1

The current workflow processes one static chest radiograph per analysis:

Chest X-Ray — accept and save the uploaded image locally.

Preprocessing — prepare the image using TorchXRayVision.

DenseNet121 — run the pretrained imaging model.

Pneumonia-associated model score — extract the relevant pathology output.

SQLite — store analysis metadata.

Streamlit Dashboard — display the result and analysis history.

Planned — not yet implemented

flowchart TD
    X[Chest X-Ray] --> V[Vision Encoder]
    T[48-hour Vitals] --> E[Temporal Encoder]
    V --> F[Multimodal Fusion]
    E --> F
    F --> R[Deterioration Risk]

This diagram describes a proposed research direction. The two-branch system and its deterioration-risk output do not exist in v0.1.

Technology stack

Technology

Current role

Python

Application and inference orchestration

PyTorch

Neural-network inference runtime

TorchXRayVision

Chest X-ray preprocessing and pretrained model integration

DenseNet121

Pretrained imaging model architecture

Streamlit

Upload interface, model-output display, and scan history

SQLite

Local storage of analysis metadata

pandas

Scan-history table preparation and summary calculations

Repository structure

Path

Purpose

app.py

Streamlit application with Analyze, Scan History, and Project tabs

xray_model.py

Existing preprocessing and predict_xray inference implementation

database.py

SQLite initialization, record saving, and history retrieval

styles.css

Custom dashboard styling

.streamlit/config.toml

Streamlit theme and configuration

requirements.txt

Python dependencies for the project

README.md

Project overview and local setup instructions

screenshots/trige_dashboard.png

Dashboard screenshot used in this README; add a captured image at this path

uploads/

Runtime directory for saved uploads; created by the application

The SQLite database location is determined by database.py. The app initializes storage when it starts. Uploaded image files and analysis metadata are stored separately; this prototype does not provide an automatic retention or deletion workflow.

Installation

Run the following commands from the repository root, where app.py and requirements.txt are located.

1. Create a virtual environment

python -m venv .venv

2. Activate the environment

Windows Command Prompt:

.venv\Scripts\activate.bat

macOS or Linux:

source .venv/bin/activate

3. Install the project dependencies

pip install -r requirements.txt

Use the project's existing dependency file. Initial dependency installation requires an internet connection.

Running locally

With the virtual environment activated, run this command from the repository root:

python -m streamlit run app.py

Open the local URL displayed by Streamlit in your browser.

Select the ANALYZE tab.

Upload a de-identified demonstration chest X-ray.

Select RUN TRIGE ANALYSIS.

Review the raw model score and demonstration flag.

Open SCAN HISTORY to view stored analyses.

Open PROJECT for the implemented scope, proposed architecture, and limitations.

Pretrained TorchXRayVision model weights may be downloaded on first inference if they are not already cached. The first analysis may therefore require internet access and take longer than subsequent analyses.

How the current inference pipeline works

Upload and local save. The user selects a PNG, JPG, or JPEG image. When analysis is requested, app.py saves its bytes under a generated unique filename in uploads/.

Preprocessing and inference. The app calls predict_xray(image_path) from xray_model.py. That module performs TorchXRayVision preprocessing and runs the pretrained densenet121-res224-all model.

Score extraction. The pneumonia pathology output is returned to the application as the pneumonia-associated model score.

Demo flag. Scores greater than or equal to 0.50 receive ELEVATED; scores below 0.50 receive LOW. This threshold is used only to demonstrate the interface.

Metadata persistence. The app calls save_scan(filename, score, flag). The database module handles storage of the associated record, including model version and timestamp.

Result and history display. The dashboard displays the raw score to three decimal places. get_scans() supplies the history table, ordered newest first, and the total-analysis, elevated-flag, and latest-analysis metrics.

Display rounding does not change the value passed to the database. The interface presents a model output, not an explanation of confirmed disease, an infection-localization result, or a patient-care recommendation.

Current limitations

Pretrained integration only: v0.1 uses existing DenseNet121 weights; no original medical model training is performed.

No clinical validation: this project does not establish diagnostic accuracy, calibration, or suitability for patient-care decisions.

Demonstration threshold: the 0.50 flag threshold has no established clinical meaning in this application.

Static imaging only: the app does not ingest real-time physiological data or process a 48-hour vital-sign trajectory.

No multimodal fusion: imaging and temporal physiological information are not combined.

No deterioration-risk prediction: the displayed score must not be interpreted as future patient deterioration risk.

No disease localization: v0.1 does not identify confirmed infection regions or provide validated explanatory overlays.

Limited input workflow: the upload interface accepts PNG/JPG/JPEG images; it does not provide a DICOM workflow or verify that an uploaded image is a suitable chest radiograph.

Local prototype storage: saved uploads and SQLite records do not constitute a clinical records system. De-identification must occur before upload.

No clinical deployment: the application is for research and demonstration only.

Future roadmap

The following are proposed research steps, not available features or committed release dates:

Evaluate the static imaging module on appropriate research datasets and document performance, calibration, and failure cases.

Investigate image-attribution visualizations and their limitations without presenting them as confirmed disease localization.

Develop preprocessing and missing-data handling for 48-hour vital-sign trajectories.

Explore a temporal encoder and alignment of physiological observations with chest imaging.

Investigate multimodal fusion of image and temporal representations.

Define and evaluate deterioration-risk research targets with patient-level data separation and leakage controls.

Document results and limitations before considering any expansion of the intended use.

Research disclaimer

Trige is a research and educational prototype. It has not been clinically validated and is not intended for diagnosis, treatment, triage decisions, or other patient-care use.

The pneumonia-associated model score is a research model output, not a clinically calibrated probability or medical diagnosis. LOW and ELEVATED are demonstration flags and must not guide clinical action. Use only de-identified demonstration images.