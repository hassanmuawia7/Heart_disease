import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD FILES
# =========================
model = joblib.load("KNN_heart_disease.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# =========================
# HEADER
# =========================
st.title("❤️ Heart Disease Prediction App")
st.markdown(
    "Enter patient information below to estimate heart disease risk."
)

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("🩺 Patient Information")

age = st.sidebar.slider(
    "Age",
    min_value=16,
    max_value=100,
    value=30
)

sex = st.sidebar.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    ["NAP", "ATA", "ASY", "TA"]
)

resting_bp = st.sidebar.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=220,
    value=120
)

cholesterol = st.sidebar.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs = st.sidebar.selectbox(
    "Fasting Blood Sugar >120",
    ["0", "1"]
)

resting_ecg = st.sidebar.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.sidebar.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

exercise_angina = st.sidebar.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.sidebar.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

st_slope = st.sidebar.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# =========================
# SUMMARY CARD
# =========================
st.subheader("📋 Patient Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Age", age)
    st.metric("Blood Pressure", resting_bp)

with col2:
    st.metric("Cholesterol", cholesterol)
    st.metric("Max HR", max_hr)

with col3:
    st.metric("Sex", sex)
    st.metric("Oldpeak", oldpeak)

# =========================
# PREDICT BUTTON
# =========================
if st.button("🔍 Predict Heart Disease Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.divider()
    st.subheader("📊 Prediction Result")
    try:
            risk_score = model.predict_proba(scaled_input)[0][1]

            st.metric(
                "Estimated Risk",
                f"{risk_score * 100:.1f}% Chance of Heart Disease"
            )

            st.progress(float(risk_score))

    except:
        pass

    if prediction == 1:

        st.error("⚠️ HIGH Risk of Heart Disease")

        st.info("""
                ### Recommendations

                • Consult a cardiologist

                • Monitor blood pressure regularly

                • Reduce cholesterol intake

                • Exercise regularly

                • Maintain healthy body weight

                • Avoid smoking
                """)

    else:

        st.success("✅ LOW Risk of Heart Disease")

        st.info("""
                ### Recommendations

                • Continue healthy lifestyle

                • Maintain balanced diet

                • Exercise regularly

                • Get routine health checkups

                • Monitor cholesterol levels
                """)