import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1f4e79;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    '<div class="title">🩺 Diabetes Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based Diabetes Risk Prediction</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")


df = load_data()


# ---------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------

X = df.drop(columns=["Outcome"])
y = df["Outcome"]


# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)


# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.1,
    random_state=0,
    stratify=y
)


# ---------------------------------------------------
# SMOTE
# ---------------------------------------------------

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(
    X_train,
    y_train
)


# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

model = LogisticRegression()

model.fit(
    X_resampled,
    y_resampled
)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🩺 Patient Information")

st.sidebar.write(
    "Enter the patient's details below."
)

st.sidebar.markdown("---")

st.sidebar.subheader("🧪 Quick Test")

st.sidebar.info(
    "For testing the app, you can enter higher glucose "
    "and other sample values to test the diabetic prediction."
)

# ---------------------------------------------------
# INPUT FIELDS
# ---------------------------------------------------


# ---------------------------------------------------
# PATIENT INPUTS
# ---------------------------------------------------

st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.slider(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.slider(
        "Glucose Level",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.slider(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.slider(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )


with col2:

    insulin = st.slider(
        "Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=125.0,
        step=1.0
    )

    bmi = st.slider(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.slider(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.slider(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

st.markdown("---")

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:

    predict_button = st.button(
        "🔍 Predict Diabetes",
        use_container_width=True
    )


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)[0]


    # Probability
    probability = model.predict_proba(input_scaled)[0][1]


    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------

    st.markdown("---")

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Diabetes Detected"
        )

        st.markdown(
            f"""
            <div class="result-box">
                Patient Status: Diabetes Detected<br>
                Prediction Probability: {probability * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "This is an ML prediction, not a medical diagnosis. "
            "Please consult a qualified healthcare professional."
        )

    else:

        st.success(
            "✅ No Diabetes Detected"
        )

        st.markdown(
            f"""
            <div class="result-box">
                Patient Status: No Diabetes<br>
                Prediction Probability: {(1 - probability) * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "This is an ML prediction, not a medical diagnosis. "
            "For medical concerns, please consult a qualified healthcare professional."
        )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "⚕️ Diabetes Prediction System | Machine Learning Project"
)