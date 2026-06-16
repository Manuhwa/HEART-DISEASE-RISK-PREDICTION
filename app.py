from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st


# Configure the Streamlit page and set the application title, icon, and layout.
st.set_page_config(page_title="HDRPS Minimal", page_icon=":heart:", layout="wide")

# Paths for saved model artifacts: the classifier, scaler, and optional metadata file.
ARTIFACT_DIR = Path("model_artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

# Order of features expected by the model and by the user input form.
FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Human-readable labels for each model feature used in reports and visual summaries.
FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting BP",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Major Vessels",
    "thal": "Thalassemia",
}

# Reference guide shown on the About page describing each model input.
VARIABLE_GUIDE = [
    ("age", "Age in years", "20-100"),
    ("sex", "Biological sex", "0=Female, 1=Male"),
    ("cp", "Chest pain type", "0,1,2,3"),
    ("trestbps", "Resting blood pressure (mmHg)", "80-220"),
    ("chol", "Serum cholesterol (mg/dl)", "100-600"),
    ("fbs", "Fasting blood sugar > 120 mg/dl", "0=No, 1=Yes"),
    ("restecg", "Resting electrocardiographic result", "0,1,2"),
    ("thalach", "Maximum heart rate achieved", "60-220"),
    ("exang", "Exercise induced angina", "0=No, 1=Yes"),
    ("oldpeak", "ST depression induced by exercise", "0.0-6.0"),
    ("slope", "Slope of peak exercise ST segment", "1,2,3"),
    ("ca", "Number of major vessels", "0-3"),
    ("thal", "Thalassemia status", "3,6,7"),
]


@st.cache_resource
# Load the trained model and scaler from disk once per session.
# Caching avoids repeated artifact loading across Streamlit reruns.
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Expected 'model_artifacts/model.pkl' and "
            "'model_artifacts/scaler.pkl'."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
# Load optional metadata such as dataset size and evaluation metrics.
# This is cached to reduce repeated file reads.
def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def default_patient_data():
    # Provide a default set of patient values for initial form population.
    # These values are used when the app loads and when the user resets inputs.
    return {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }


def estimate_risk_level(probability):
    # Categorize the model probability into a simple risk band.
    if probability < 0.30:
        return "Low"
    if probability < 0.70:
        return "Moderate"
    return "High"


def mild_value_checks(data):
    # Provide soft clinical warnings for extreme or unusual input values.
    warnings = []
    if data["trestbps"] > 180:
        warnings.append("Resting blood pressure is high; confirm entry and clinical context.")
    if data["chol"] > 300:
        warnings.append("Cholesterol is high; verify unit and recorded value.")
    if data["thalach"] < 90:
        warnings.append("Maximum heart rate is low; check if this reflects patient condition.")
    if data["oldpeak"] > 3.0:
        warnings.append("ST depression is elevated; review ECG/exercise findings.")
    return warnings


def format_feature_value(feature, value):
    # Convert raw numeric inputs into human-readable strings for display.
    if feature == "sex":
        return "Female" if int(value) == 0 else "Male"
    if feature == "exang":
        return "Yes" if int(value) == 1 else "No"
    if feature == "fbs":
        return "Yes" if int(value) == 1 else "No"
    if feature == "age":
        return f"{int(value)} years"
    if feature == "chol":
        return f"{int(value)} mg/dl"
    if feature == "trestbps":
        return f"{int(value)} mmHg"
    if feature == "thalach":
        return f"{int(value)} bpm"
    if feature == "oldpeak":
        return f"{float(value):.1f}"
    if feature == "thal":
        thal_map = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}
        return thal_map.get(int(value), str(int(value)))
    if feature == "cp":
        cp_map = {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal", 3: "Asymptomatic"}
        return cp_map.get(int(value), str(int(value)))
    if feature == "restecg":
        restecg_map = {0: "Normal", 1: "ST-T wave abnormality", 2: "LV hypertrophy"}
        return restecg_map.get(int(value), str(int(value)))
    if feature == "ca":
        return str(int(value))
    if feature == "slope":
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def compute_factor_analysis(input_df, input_scaled, model, top_risk=7, top_protective=3):
    # Analyze feature contributions using the model coefficients and scaled input.
    # Positive contributions increase predicted risk; negative contributions are protective.
    coefficients = model.coef_[0]
    contributions = coefficients * input_scaled[0]
    details = pd.DataFrame(
        {
            "feature": FEATURE_ORDER,
            "label": [FEATURE_LABELS[f] for f in FEATURE_ORDER],
            "value": input_df.iloc[0].values,
            "display_value": [
                format_feature_value(f, v) for f, v in zip(FEATURE_ORDER, input_df.iloc[0].values)
            ],
            "coefficient": coefficients,
            "contribution": contributions,
            "abs_contribution": np.abs(contributions),
        }
    )

    risk_factors = details[details["contribution"] > 0].sort_values("contribution", ascending=False)
    protective = details[details["contribution"] < 0].sort_values("contribution")

    # Compute relative impact percentages for the top contributing factors.
    if not risk_factors.empty:
        risk_total = risk_factors["contribution"].sum()
        risk_factors = risk_factors.copy()
        risk_factors["impact_pct"] = (risk_factors["contribution"] / risk_total * 100).round(0).astype(int)
    if not protective.empty:
        protective_total = protective["contribution"].abs().sum()
        protective = protective.copy()
        protective["impact_pct"] = (-protective["contribution"].abs() / protective_total * 100).round(0).astype(int)

    return risk_factors.head(top_risk), protective.head(top_protective)


def build_clinical_recommendations(prediction, probability, patient_data, risk_factors):
    # Create narrative recommendations based on prediction, probability, and encoded input risk indicators.
    recommendations = []
    if prediction == 1 or probability >= 0.5:
        recommendations.append(
            "High probability of coronary artery disease based on model prediction."
        )
        recommendations.append("Recommend cardiology consultation.")
        recommendations.append("Consider stress echocardiography or further cardiac workup.")
    else:
        recommendations.append("Low to moderate modeled probability of heart disease at this time.")
        recommendations.append("Continue routine cardiovascular risk assessment and prevention.")

    interventions = []
    if patient_data["chol"] >= 200:
        interventions.append(f"Elevated cholesterol ({patient_data['chol']} mg/dl) — target < 200 mg/dl.")
    if patient_data["trestbps"] >= 130:
        interventions.append(
            f"Manage hypertension ({patient_data['trestbps']} mmHg) — target BP < 130/80 mmHg."
        )
    if patient_data["fbs"] == 1:
        interventions.append("Fasting blood sugar > 120 mg/dl — evaluate glycemic control.")
    if patient_data["exang"] == 1:
        interventions.append("Exercise-induced angina reported — prioritize ischemia evaluation.")
    if patient_data["oldpeak"] >= 1.0:
        interventions.append(
            f"ST depression {patient_data['oldpeak']:.1f} — correlate with ECG and stress testing."
        )

    if not interventions and risk_factors is not None and not risk_factors.empty:
        top = risk_factors.iloc[0]
        interventions.append(
            f"Address primary driver: {top['label']} ({top['display_value']})."
        )

    if interventions:
        recommendations.append("Risk factors requiring intervention:")
    return recommendations, interventions



def render_factor_bars(factors, bar_class):
    # Render HTML bars for risk/protective factor contributions.
    if factors is None or factors.empty:
        return '<p class="pr-muted">No factors in this category for this prediction.</p>'

    max_pct = max(int(abs(row["impact_pct"])) for _, row in factors.iterrows()) or 1
    rows = []
    for _, row in factors.iterrows():
        width = max(8, int(abs(row["impact_pct"]) / max_pct * 100))
        sign = "+" if row["impact_pct"] > 0 else ""
        label = html_escape(str(row["label"]))
        display = html_escape(str(row["display_value"]))
        rows.append(
            f'<div class="pr-factor-row">'
            f'<div class="pr-factor-label"><strong>{label}</strong> '
            f'<span class="pr-factor-value">({display})</span></div>'
            f'<div class="pr-factor-bar-wrap">'
            f'<div class="pr-factor-bar {bar_class}" style="width:{width}%;"></div>'
            f"</div>"
            f'<div class="pr-factor-pct">{sign}{int(row["impact_pct"])}%</div>'
            f"</div>"
        )
    return "".join(rows)


def build_recommendations_html(recommendations, interventions):
    # Convert recommendation text and intervention details into safe HTML list items.
    items = []
    for rec in recommendations:
        if rec == "Risk factors requiring intervention:":
            continue
        items.append(f"<li>{html_escape(rec)}</li>")
    if interventions:
        sub_items = "".join(f"<li>{html_escape(item)}</li>" for item in interventions)
        items.append(
            "<li>Risk factors requiring intervention:"
            f'<ul class="pr-sublist">{sub_items}</ul></li>'
        )
    return f'<ul class="pr-rec-list">{"".join(items)}</ul>'


def render_prediction_dashboard(
    prediction,
    probability,
    patient_data,
    patient_id,
    risk_factors,
    protective_factors,
):
    # Assemble the prediction dashboard UI using HTML and CSS styling.
    outcome = "HEART DISEASE PRESENT" if prediction == 1 else "HEART DISEASE ABSENT"
    confidence = probability * 100 if prediction == 1 else (1 - probability) * 100
    gauge_pos = max(2, min(98, probability * 100))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    recommendations, interventions = build_clinical_recommendations(
        prediction, probability, patient_data, risk_factors
    )

    # Build the HTML content for recommendations and factor visualization.
    rec_html = build_recommendations_html(recommendations, interventions)
    risk_html = render_factor_bars(risk_factors, "pr-bar-risk")
    protective_html = render_factor_bars(protective_factors, "pr-bar-protect")
    safe_patient_id = html_escape(patient_id)

    dashboard_html = (
        f'<div class="pr-dashboard" id="prediction-results-print">'
        f'<div class="pr-header">PREDICTION RESULTS</div>'
        f'<div class="pr-info-bar">'
        f'<span>Patient: <strong>{safe_patient_id}</strong></span>'
        f'<span>Date: <strong>{now_str}</strong></span>'
        f"</div>"
        f'<div class="pr-grid">'
        f'<div class="pr-panel pr-outcome-panel">'
        f'<div class="pr-section-title">PREDICTION OUTCOME</div>'
        f'<div class="pr-outcome-main">{html_escape(outcome)}</div>'
        f'<div class="pr-confidence">Confidence: {confidence:.1f}%</div>'
        f'<div class="pr-gauge-labels"><span>LOW RISK</span><span>HIGH RISK</span></div>'
        f'<div class="pr-gauge-track"><div class="pr-gauge-marker" style="left: {gauge_pos}%;"></div></div>'
        f"</div>"
        f'<div class="pr-panel pr-factors-panel">'
        f'<div class="pr-section-title">Top Contributing Factors</div>'
        f"{risk_html}"
        f'<div class="pr-section-title pr-section-spaced">Protective Factors</div>'
        f"{protective_html}"
        f"</div>"
        f'<div class="pr-panel pr-rec-panel">'
        f'<div class="pr-section-title">CLINICAL RECOMMENDATIONS</div>'
        f"{rec_html}"
        f'<p class="pr-disclaimer">This tool provides decision support only and does not replace clinical judgment.</p>'
        f"</div></div></div>"
    )
    st.html(get_results_dashboard_css() + dashboard_html, width="stretch")



def get_results_dashboard_css():
    return """
    <style>
    .pr-dashboard {
        background: linear-gradient(180deg, #e8f1fb 0%, #f4f8fd 100%);
        border: 1px solid #c5d7ef;
        border-radius: 14px;
        overflow: hidden;
        margin: 1rem 0 0.5rem 0;
        box-shadow: 0 8px 24px rgba(30, 64, 120, 0.08);
        font-family: "Source Sans Pro", sans-serif;
    }
    .pr-header {
        background: linear-gradient(90deg, #1a4f8c, #2563a8);
        color: #ffffff;
        text-align: center;
        font-weight: 800;
        letter-spacing: 0.06em;
        padding: 0.85rem 1rem;
        font-size: 1.05rem;
    }
    .pr-info-bar {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        background: #e9eef5;
        color: #1e293b;
        padding: 0.55rem 1rem;
        font-size: 0.92rem;
        border-bottom: 1px solid #d5deea;
    }
    .pr-grid {
        display: grid;
        grid-template-columns: 1fr 1.15fr;
        gap: 0.75rem;
        padding: 0.75rem;
    }
    .pr-panel {
        background: #ffffff;
        border: 1px solid #d7e2f0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    .pr-rec-panel { grid-column: 1 / -1; }
    .pr-section-title {
        color: #1e4a7a;
        font-weight: 800;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        margin-bottom: 0.65rem;
        text-transform: uppercase;
    }
    .pr-section-spaced { margin-top: 1rem; }
    .pr-outcome-main {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.35rem;
        line-height: 1.25;
    }
    .pr-confidence {
        color: #334155;
        font-size: 0.98rem;
        margin-bottom: 0.85rem;
    }
    .pr-gauge-labels {
        display: flex;
        justify-content: space-between;
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .pr-gauge-track {
        position: relative;
        height: 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%);
        border: 1px solid #cbd5e1;
    }
    .pr-gauge-marker {
        position: absolute;
        top: 50%;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #facc15;
        border: 2px solid #ffffff;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
        transform: translate(-50%, -50%);
    }
    .pr-factor-row {
        display: grid;
        grid-template-columns: minmax(140px, 1.2fr) minmax(120px, 1.5fr) 52px;
        gap: 0.5rem;
        align-items: center;
        margin-bottom: 0.55rem;
    }
    .pr-factor-label { font-size: 0.82rem; color: #0f172a; line-height: 1.2; }
    .pr-factor-value { color: #64748b; font-weight: 500; }
    .pr-factor-bar-wrap {
        background: #edf2f7;
        border-radius: 999px;
        height: 12px;
        overflow: hidden;
    }
    .pr-factor-bar { height: 100%; border-radius: 999px; }
    .pr-bar-risk { background: linear-gradient(90deg, #fb923c, #f97316); }
    .pr-bar-protect { background: linear-gradient(90deg, #4ade80, #22c55e); }
    .pr-factor-pct {
        font-size: 0.82rem;
        font-weight: 700;
        color: #334155;
        text-align: right;
    }
    .pr-rec-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #1e293b;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .pr-sublist { margin-top: 0.35rem; }
    .pr-disclaimer {
        margin: 0.85rem 0 0;
        font-size: 0.78rem;
        color: #64748b;
    }
    .pr-muted { color: #64748b; font-size: 0.85rem; margin: 0; }
    @media (max-width: 900px) {
        .pr-grid { grid-template-columns: 1fr; }
        .pr-factor-row { grid-template-columns: 1fr; gap: 0.25rem; }
        .pr-factor-pct { text-align: left; }
    }
    </style>
    """


def apply_results_dashboard_styles():
    # Inject dashboard-specific CSS into the Streamlit app for the prediction results.
    st.markdown(
        """
        <style>
        .pr-dashboard {
            background: linear-gradient(180deg, #e8f1fb 0%, #f4f8fd 100%);
            border: 1px solid #c5d7ef;
            border-radius: 14px;
            overflow: hidden;
            margin: 1rem 0 0.5rem 0;
            box-shadow: 0 8px 24px rgba(30, 64, 120, 0.08);
        }
        .pr-header {
            background: linear-gradient(90deg, #1a4f8c, #2563a8);
            color: #ffffff;
            text-align: center;
            font-weight: 800;
            letter-spacing: 0.06em;
            padding: 0.85rem 1rem;
            font-size: 1.05rem;
        }
        .pr-info-bar {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            background: #e9eef5;
            color: #1e293b;
            padding: 0.55rem 1rem;
            font-size: 0.92rem;
            border-bottom: 1px solid #d5deea;
        }
        .pr-grid {
            display: grid;
            grid-template-columns: 1fr 1.15fr;
            gap: 0.75rem;
            padding: 0.75rem;
        }
        .pr-panel {
            background: #ffffff;
            border: 1px solid #d7e2f0;
            border-radius: 10px;
            padding: 0.9rem 1rem;
        }
        .pr-rec-panel {
            grid-column: 1 / -1;
        }
        .pr-section-title {
            color: #1e4a7a;
            font-weight: 800;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.65rem;
            text-transform: uppercase;
        }
        .pr-section-spaced {
            margin-top: 1rem;
        }
        .pr-outcome-main {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
            line-height: 1.25;
        }
        .pr-confidence {
            color: #334155;
            font-size: 0.98rem;
            margin-bottom: 0.85rem;
        }
        .pr-gauge-labels {
            display: flex;
            justify-content: space-between;
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .pr-gauge-track {
            position: relative;
            height: 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%);
            border: 1px solid #cbd5e1;
        }
        .pr-gauge-marker {
            position: absolute;
            top: 50%;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #facc15;
            border: 2px solid #ffffff;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
            transform: translate(-50%, -50%);
        }
        .pr-factor-row {
            display: grid;
            grid-template-columns: minmax(140px, 1.2fr) minmax(120px, 1.5fr) 52px;
            gap: 0.5rem;
            align-items: center;
            margin-bottom: 0.55rem;
        }
        .pr-factor-label {
            font-size: 0.82rem;
            color: #0f172a;
            line-height: 1.2;
        }
        .pr-factor-value {
            color: #64748b;
            font-weight: 500;
        }
        .pr-factor-bar-wrap {
            background: #edf2f7;
            border-radius: 999px;
            height: 12px;
            overflow: hidden;
        }
        .pr-factor-bar {
            height: 100%;
            border-radius: 999px;
        }
        .pr-bar-risk {
            background: linear-gradient(90deg, #fb923c, #f97316);
        }
        .pr-bar-protect {
            background: linear-gradient(90deg, #4ade80, #22c55e);
        }
        .pr-factor-pct {
            font-size: 0.82rem;
            font-weight: 700;
            color: #334155;
            text-align: right;
        }
        .pr-rec-list {
            margin: 0;
            padding-left: 1.1rem;
            color: #1e293b;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .pr-sublist {
            margin-top: 0.35rem;
        }
        .pr-disclaimer {
            margin: 0.85rem 0 0;
            font-size: 0.78rem;
            color: #64748b;
        }
        .pr-disclaimer a {
            color: #2563eb;
            text-decoration: none;
        }
        .pr-muted {
            color: #64748b;
            font-size: 0.85rem;
            margin: 0;
        }
        .pr-action-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin: 0.25rem 0 1rem;
        }
        .pr-actions div[data-testid="stDownloadButton"] > button {
            background: #16a34a !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            min-width: 110px;
            font-weight: 700 !important;
        }
        .pr-actions div[data-testid="stDownloadButton"] > button:hover {
            background: #15803d !important;
            color: #ffffff !important;
        }
        .pr-actions div[data-testid="stButton"]:nth-of-type(1) > button {
            background: #2563eb !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            min-width: 110px;
            font-weight: 700 !important;
        }
        .pr-actions div[data-testid="stButton"]:nth-of-type(1) > button:hover {
            background: #1d4ed8 !important;
            color: #ffffff !important;
        }
        .pr-actions div[data-testid="stButton"]:nth-of-type(2) > button {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            min-width: 150px;
            font-weight: 700 !important;
        }
        @media print {
            [data-testid="stSidebar"], [data-testid="stHeader"], .stApp > header {
                display: none !important;
            }
            .main .block-container {
                padding-top: 0 !important;
            }
            .pr-dashboard {
                box-shadow: none;
            }
        }
        @media (max-width: 900px) {
            .pr-grid {
                grid-template-columns: 1fr;
            }
            .pr-factor-row {
                grid-template-columns: 1fr;
                gap: 0.25rem;
            }
            .pr-factor-pct {
                text-align: left;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_theme(theme_mode):
    # Apply a light or dark theme to the Streamlit app using injected CSS.
    if theme_mode == "Dark":
        app_bg = "#1e293b"
        panel_bg = "#243447"
        panel_soft_bg = "#2f4055"
        card_bg = "#25364a"
        card_border = "#4a5f78"
        text_color = "#eef4ff"
        muted_color = "#d4ddee"
        header_bg = "rgba(30, 41, 59, 0.82)"
    else:
        app_bg = "#f6f8fc"
        panel_bg = "#ffffff"
        panel_soft_bg = "#f8fafc"
        card_bg = "#ffffff"
        card_border = "#d7dfeb"
        text_color = "#0f172a"
        muted_color = "#475569"
        header_bg = "rgba(246, 248, 252, 0.82)"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {app_bg};
            color: {text_color};
        }}
        [data-testid="stSidebar"] {{
            background-color: {panel_bg};
            border-right: 1px solid {card_border};
        }}
        [data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}
        [data-testid="stHeader"] {{
            background: {header_bg};
        }}
        [data-testid="stMetric"] {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 10px;
            padding: 0.4rem 0.6rem;
        }}
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] div,
        [data-testid="stMetricValue"] div,
        [data-testid="stMetricDelta"] div {{
            color: {text_color} !important;
        }}
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {panel_soft_bg} !important;
            color: {text_color} !important;
            border-color: {card_border} !important;
        }}
        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stCheckbox label,
        .stRadio label,
        .stSlider label,
        .stDateInput label,
        .stTimeInput label,
        .stTextArea label {{
            color: {text_color} !important;
            font-weight: 600 !important;
        }}
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stWidgetLabel"] label {{
            color: {text_color} !important;
            opacity: 1 !important;
        }}
        .stExpander {{
            border: 1px solid {card_border} !important;
            border-radius: 10px !important;
            background: {panel_soft_bg} !important;
        }}
        .stExpander > details {{
            background: {panel_soft_bg} !important;
            border-radius: 10px !important;
        }}
        .stExpander summary,
        .stExpander summary p,
        .stExpander details summary p,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] details summary p {{
            color: {text_color} !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }}
        .stExpander summary {{
            background: {card_bg} !important;
            border: 1px solid {card_border} !important;
            border-radius: 8px !important;
            padding: 0.4rem 0.75rem !important;
        }}
        .stExpander summary:hover {{
            background: {panel_bg} !important;
        }}
        .stExpander summary svg,
        [data-testid="stExpander"] summary svg {{
            fill: {text_color} !important;
            color: {text_color} !important;
        }}
        .stButton > button,
        .stDownloadButton > button {{
            background: {card_bg} !important;
            color: {text_color} !important;
            border: 1px solid {card_border} !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: {panel_bg} !important;
            color: {text_color} !important;
            border-color: {card_border} !important;
        }}
        .stButton > button:focus,
        .stDownloadButton > button:focus {{
            color: {text_color} !important;
            border-color: {card_border} !important;
            box-shadow: 0 0 0 0.15rem rgba(148, 163, 184, 0.25) !important;
        }}
        .hdrps-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 16px;
            margin-top: 8px;
            margin-bottom: 12px;
        }}
        .hdrps-card-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {text_color};
            margin-bottom: 0.35rem;
        }}
        .hdrps-card-main {{
            font-size: 1.6rem;
            font-weight: 800;
            color: {text_color};
            margin-bottom: 0.2rem;
        }}
        .hdrps-card-sub {{
            font-size: 0.95rem;
            color: {muted_color};
        }}
        .risk-badge {{
            display: inline-block;
            border-radius: 9999px;
            padding: 4px 10px;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            border: 1px solid transparent;
        }}
        .risk-low {{
            background: #0f2e1f;
            color: #86efac;
            border-color: #166534;
        }}
        .risk-moderate {{
            background: #3a2a0a;
            color: #fcd34d;
            border-color: #a16207;
        }}
        .risk-high {{
            background: #3a1113;
            color: #fca5a5;
            border-color: #991b1b;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def collect_inputs():
    # Render the input form for patient features and keep values in Streamlit session state.
    st.subheader("Patient Inputs")
    defaults = default_patient_data()

    if st.button("Reset to default values"):
        st.session_state["patient_data"] = defaults.copy()

    if "patient_data" not in st.session_state:
        st.session_state["patient_data"] = defaults.copy()

    data = st.session_state["patient_data"]

    with st.expander("Demographics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            data["age"] = int(
                st.number_input("Age", min_value=20, max_value=100, value=int(data["age"]), step=1, help="Patient age in completed years.")
            )
        with col2:
            data["sex"] = int(
                st.selectbox("Sex", options=[0, 1], index=[0, 1].index(int(data["sex"])), format_func=lambda x: "Female" if x == 0 else "Male")
            )

    with st.expander("Symptoms and Exercise", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            data["cp"] = int(st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], index=[0, 1, 2, 3].index(int(data["cp"]))))
            data["exang"] = int(
                st.selectbox(
                    "Exercise Angina (exang)",
                    options=[0, 1],
                    index=[0, 1].index(int(data["exang"])),
                    format_func=lambda x: "No" if x == 0 else "Yes",
                )
            )
            data["oldpeak"] = float(
                st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=float(data["oldpeak"]), step=0.1)
            )
        with col2:
            data["thalach"] = int(
                st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=220, value=int(data["thalach"]), step=1)
            )
            data["slope"] = int(st.selectbox("ST Slope (slope)", options=[1, 2, 3], index=[1, 2, 3].index(int(data["slope"]))))

    with st.expander("Clinical Tests", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            data["trestbps"] = int(
                st.number_input("Resting BP (trestbps)", min_value=80, max_value=220, value=int(data["trestbps"]), step=1)
            )
            data["chol"] = int(st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=int(data["chol"]), step=1))
            data["fbs"] = int(
                st.selectbox(
                    "Fasting Blood Sugar >120 (fbs)",
                    options=[0, 1],
                    index=[0, 1].index(int(data["fbs"])),
                    format_func=lambda x: "No" if x == 0 else "Yes",
                )
            )
        with col2:
            data["restecg"] = int(st.selectbox("Resting ECG (restecg)", options=[0, 1, 2], index=[0, 1, 2].index(int(data["restecg"]))))
            data["ca"] = int(st.selectbox("Major Vessels (ca)", options=[0, 1, 2, 3], index=[0, 1, 2, 3].index(int(data["ca"]))))
            data["thal"] = int(st.selectbox("Thalassemia (thal)", options=[3, 6, 7], index=[3, 6, 7].index(int(data["thal"]))))

    st.session_state["patient_data"] = data
    return data


def predict_page():
    # Build the primary prediction page with form inputs, model execution, and results display.
    st.title("Heart Disease Risk Prediction System")
    st.caption("Minimal HDRPS - Logistic Regression inference from saved artifacts")
    apply_results_dashboard_styles()

    metadata = load_metadata()

    try:
        model, scaler = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    if "patient_id" not in st.session_state:
        st.session_state["patient_id"] = "P00123456"

    patient_id = st.text_input(
        "Patient ID",
        value=st.session_state["patient_id"],
        help="Shown on the prediction results report.",
    )
    st.session_state["patient_id"] = patient_id.strip() or "P00123456"

    patient_data = collect_inputs()
    gentle_warnings = mild_value_checks(patient_data)
    for warning_text in gentle_warnings:
        st.warning(warning_text)

    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([patient_data], columns=FEATURE_ORDER)
        input_scaled = scaler.transform(input_df)
        probability = float(model.predict_proba(input_scaled)[0, 1])
        prediction = int(probability >= 0.5)
        risk_level = estimate_risk_level(probability)
        risk_factors, protective_factors = compute_factor_analysis(input_df, input_scaled, model)

        result_payload = {
            "patient_id": st.session_state["patient_id"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prediction": prediction,
            "probability": round(probability, 6),
            "risk_band": risk_level,
            "inputs": patient_data,
        }
        st.session_state["prediction_result"] = {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "patient_data": patient_data.copy(),
            "patient_id": st.session_state["patient_id"],
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "result_payload": result_payload,
        }

    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]
        render_prediction_dashboard(
            prediction=result["prediction"],
            probability=result["probability"],
            patient_data=result["patient_data"],
            patient_id=result["patient_id"],
            risk_factors=result["risk_factors"],
            protective_factors=result["protective_factors"],
        )

        st.markdown('<div class="pr-actions">', unsafe_allow_html=True)
        action_col1, action_col2, action_col3 = st.columns([1, 1, 1.2])
        with action_col1:
            st.download_button(
                "SAVE",
                data=json.dumps(result["result_payload"], indent=2),
                file_name=f"hdrps_{result['patient_id']}.json",
                mime="application/json",
                use_container_width=True,
            )
        with action_col2:
            if st.button("PRINT", use_container_width=True):
                st.components.v1.html(
                    "<script>window.parent.print();</script>",
                    height=0,
                )
        with action_col3:
            if st.button("NEW PREDICTION", use_container_width=True):
                del st.session_state["prediction_result"]
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if metadata:
            st.caption(
                "Model metadata: "
                f"rows={metadata.get('n_rows', 'NA')}, "
                f"features={metadata.get('n_features', 'NA')}, "
                f"test ROC-AUC={metadata.get('test_metrics', {}).get('roc_auc', 'NA')}"
            )


def about_page():
    # Render the informational About page with model description and input variable guide.
    st.title("About HDRPS")
    st.write(
        "The Heart Disease Risk Prediction System (HDRPS) is a lightweight decision-support "
        "tool that uses a trained Logistic Regression model to estimate the probability of "
        "heart disease from routine clinical variables."
    )
    st.write(
        "The model was trained using 13 structured features from the heart disease dataset, "
        "then saved as reusable artifacts (`model.pkl` and `scaler.pkl`)."
    )

    guide_df = pd.DataFrame(VARIABLE_GUIDE, columns=["Variable", "Meaning", "Allowed Values / Range"])
    st.subheader("Variables Used in Model Training")
    st.dataframe(guide_df, use_container_width=True)

    st.markdown(
        """
        **Prediction Output**
        - Class `1`: Heart disease likely present
        - Class `0`: Heart disease likely absent
        - Probability: confidence estimate from 0% to 100%
        """
    )


def main():
    # Main application entry point to render the sidebar and choose between pages.
    st.sidebar.title("Navigation")
    theme_mode = st.sidebar.selectbox("Display theme", ["Dark", "Light"], index=1)
    apply_theme(theme_mode)
    page = st.sidebar.radio("Go to", ["Predict", "About"])

    if page == "Predict":
        predict_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
