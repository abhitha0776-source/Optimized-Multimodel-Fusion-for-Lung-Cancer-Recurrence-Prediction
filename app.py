import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import cv2

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Lung Cancer Recurrence Prediction", layout="wide")

# ------------------ CSS Styling ------------------
st.markdown("""
<style>
@keyframes blink {
  0% { opacity: 1; }
  50% { opacity: 0; }
  100% { opacity: 1; }
}
.blink {
  color: darkred;
  font-size: 22px;
  font-weight: bold;
  animation: blink 1.2s infinite;
  text-align: center;
}
.hero {
    background: linear-gradient(90deg, #4CAF50, #45a049);
    color: #ffffff;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 20px;
}
.info-box {
    background-color: #fff3cd;
    color: #856404;
    padding: 15px;
    border-left: 6px solid #ffeeba;
    border-radius: 10px;
    margin-bottom: 15px;
}
.prob-bar {
    height: 22px;
    border-radius: 6px;
    margin-top: 5px;
}
.prob-clinical { background-color: #1f77b4; }
.prob-image { background-color: #ff7f0e; }
.prob-fusion { background-color: #2ca02c; }
</style>
""", unsafe_allow_html=True)

# ------------------ WARNING ------------------
st.markdown('<p class="blink">⚠ Research Use Only – Not for Clinical Decision Making ⚠</p>', unsafe_allow_html=True)

# ------------------ HERO ------------------
st.markdown('<div class="hero">OncoFusion: Lung Cancer Recurrence Risk Predictor</div>', unsafe_allow_html=True)

# ------------------ INFO ------------------
st.markdown("""
<div class="info-box">
<strong>About Lung Cancer Recurrence Prediction:</strong><br>
Lung cancer can recur even after successful treatment. Predicting recurrence early using clinical features and CT images helps identify high-risk patients, guide follow-up care, and improve survival outcomes.
</div>

<div class="info-box">
<strong>Project Features:</strong><br>
We use 17 clinical features including patient demographics, smoking history, histology, mutation status, treatment details, and pathological staging. CT scan images are analyzed using a VGG16-based model. A late fusion approach combines clinical and image predictions for robust assessment.
</div>
""", unsafe_allow_html=True)

# ------------------ LOAD MODELS ------------------
clinical_model = joblib.load("xgb_model_final.pkl")
image_model = load_model("vgg_model_fusion.h5")
scaler = joblib.load("scaler1.pkl")
feature_order = joblib.load("feature_names1.pkl")

# ------------------ CLINICAL INPUTS ------------------
st.header("📄 Clinical Features")

feature_dropdowns = {
    "Gender": ["Male", "Female"],
    "Smoking status": ["Former", "Nonsmoker", "Current"],
    "Histology": ["Adenocarcinoma", "Squamous cell carcinoma"],
    "Adjuvant Treatment": ["Yes", "No"],
    "Chemotherapy": ["Yes", "No"],
    "Radiation": ["Yes", "No"],
    "Pleural invasion (elastic, visceral, or parietal)": ["Yes", "No"],
    "EGFR mutation status": ["Wildtype", "Mutant"],
    "KRAS mutation status": ["Wildtype", "Mutant"],
    "ALK translocation status": ["Wildtype", "Translocated"],
    "Ethnicity": ["Asian", "Caucasian", "Hispanic/Latino", "Native Hawaiian/Pacific Islander"],
    "Pathological T Stage": ["T1a","T1b","T2a","T2b","T3","T4","Tis","Unknown"],
    "Pathological N Stage": ["N0","N1","N2"],
    "Pathological M Stage": ["M0","M1a","M1b","M1c"],
    "Lymphovascular invasion": ["Present","Absent","unknown"]
}

ui_inputs = {}
for k, v in feature_dropdowns.items():
    ui_inputs[k] = st.selectbox(k, v)

ui_inputs["Age at Histological Diagnosis"] = st.number_input(
    "Age at Histological Diagnosis", 0, 120, 60
)
ui_inputs["Days between CT and surgery"] = st.number_input(
    "Days between CT and surgery", 0, 10000, 30
)

# ================== CORRECT ENCODING ==================

# Initialize model input
model_input = {f: 0 for f in feature_order}

# Numeric features
model_input["Age at Histological Diagnosis"] = ui_inputs["Age at Histological Diagnosis"]
model_input["Days between CT and surgery"] = ui_inputs["Days between CT and surgery"]

# Label Encoded Features (exact training mapping)
label_encoders = {
    "Gender": {"Male": 1, "Female": 0},
    "Smoking status": {"Former": 1, "Nonsmoker": 0, "Current": 2},
    "Histology": {"Adenocarcinoma": 0, "Squamous cell carcinoma": 1},
    "Pleural invasion (elastic, visceral, or parietal)": {"No": 0, "Yes": 1},
    "EGFR mutation status": {"Wildtype": 1, "Mutant": 0},
    "KRAS mutation status": {"Wildtype": 1, "Mutant": 0},
    "ALK translocation status": {"Wildtype": 1, "Translocated": 0},
    "Adjuvant Treatment": {"No": 0, "Yes": 1},
    "Chemotherapy": {"No": 0, "Yes": 1},
    "Radiation": {"No": 0, "Yes": 1}
}

for feature, mapping in label_encoders.items():
    model_input[feature] = mapping[ui_inputs[feature]]

# -------- ONE-HOT ENCODING (MATCH TRAINING EXACTLY) --------
one_hot_prefix_map = {
    "Ethnicity": "Ethnicity",
    "Pathological T Stage": "Pathological T stage",
    "Pathological N Stage": "Pathological N stage",
    "Pathological M Stage": "Pathological M stage",
    "Lymphovascular invasion": "Lymphovascular invasion"
}

for ui_feature, train_prefix in one_hot_prefix_map.items():
    selected = ui_inputs[ui_feature]

    for col in feature_order:
        if col.startswith(train_prefix + "_") and col.endswith(selected):
            model_input[col] = 1

# Final dataframe (correct order)
final_input = pd.DataFrame(
    [[model_input[f] for f in feature_order]],
    columns=feature_order
)

# ------------------ IMAGE UPLOAD ------------------
st.header("🖼 Upload CT Image")
uploaded_image = st.file_uploader(
    "Upload CT Image (.jpg / .png)", type=["jpg","png","jpeg"]
)

if uploaded_image:
    img_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224,224))
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded CT Image")

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict"):
    try:
        clinical_scaled = scaler.transform(final_input)
        clinical_prob = clinical_model.predict_proba(clinical_scaled)[0][1]
        clinical_pred = "🔴 Recurrence" if clinical_prob >= 0.5 else "🟢 No Recurrence"

        if uploaded_image:
            img_input = np.expand_dims(img.astype("float32") / 255.0, axis=0)
            image_prob = image_model.predict(img_input)[0][0]
            image_pred = "🔴 Recurrence" if image_prob >= 0.5 else "🟢 No Recurrence"
        else:
            image_prob = 0.0
            image_pred = "Image Not Provided"

        fusion_prob = 0.5 * clinical_prob + 0.5 * image_prob
        fusion_pred = "🔴 Recurrence" if fusion_prob >= 0.5 else "🟢 No Recurrence"

        st.subheader("📊 Prediction Results")

        def show_block(title, prob, pred, css):
            width = max(prob * 100, 5)
            return f"""
            <div style="margin-bottom:30px;">
                <div style="font-size:20px; font-weight:bold;">{title}: {pred}</div>
                <div>Probability: {prob:.4f}</div>
                <div class="prob-bar {css}" style="width:{width}%"></div>
            </div>
            """

        st.markdown(
            show_block("🩺 Clinical Model", clinical_prob, clinical_pred, "prob-clinical") +
            show_block("🖼 Image Model", image_prob, image_pred, "prob-image") +
            show_block("🔗 Fusion Model", fusion_prob, fusion_pred, "prob-fusion"),
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

