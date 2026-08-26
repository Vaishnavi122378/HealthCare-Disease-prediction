
# ============================================================
# HEARTCARE AI - COMPLETE STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import hashlib
import hmac

from pathlib import Path
from datetime import datetime, timezone

from database import (
    create_table,
    save_patient,
    get_patients,
    clear_patients,
    get_user,
    log_action,
    get_recent_failed_attempts
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# REMOVE STREAMLIT DEFAULT HEADER / TOP BAR
# ============================================================

st.markdown(
    """
    <style>

    /* Hide Streamlit top header */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* Hide Streamlit toolbar */
    div[data-testid="stToolbar"] {
        display: none;
    }

    /* Remove top decoration */
    div[data-testid="stDecoration"] {
        display: none;
    }

    /* Remove extra top space */
    .stApp {
        margin-top: 0 !important;
    }

    /* Make main content start from the top */
    section[data-testid="stMain"] {
        padding-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       REMOVE STREAMLIT TOP WHITE SPACE / HEADER
       -------------------------------------------------------- */

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0rem !important;
        visibility: hidden !important;
    }

    div[data-testid="stToolbar"] {
        visibility: hidden !important;
        height: 0px !important;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }

    /* Remove top padding from main application */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }

    /* --------------------------------------------------------
       GENERAL
       -------------------------------------------------------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef6ff 50%,
            #f8fbff 100%
        );
    }

    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f172a 0%,
            #172554 50%,
            #1e3a8a 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stButton button {
        background: rgba(255,255,255,0.12);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 10px;
    }

    section[data-testid="stSidebar"] .stButton button:hover {
        background: rgba(255,255,255,0.22);
    }

    /* --------------------------------------------------------
       HEADINGS
       -------------------------------------------------------- */

    h1 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #1e3a8a !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #1e40af !important;
        font-weight: 700 !important;
    }

    /* --------------------------------------------------------
       INPUTS
       -------------------------------------------------------- */

    div[data-baseweb="input"],
    div[data-baseweb="select"] {
        border-radius: 10px !important;
    }

    input {
        border-radius: 10px !important;
    }

    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    .stButton button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 42px;
    }

    /* --------------------------------------------------------
       METRIC CARDS
       -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #dbeafe;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
    }

    /* --------------------------------------------------------
       DATAFRAME
       -------------------------------------------------------- */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* --------------------------------------------------------
       ALERTS
       -------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    /* --------------------------------------------------------
       LOGIN CARD
       -------------------------------------------------------- */

    .login-card {
        background: white;
        padding: 35px;
        border-radius: 18px;
        box-shadow: 0 10px 35px rgba(15, 23, 42, 0.12);
        border: 1px solid #e2e8f0;
    }

    .login-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 5px;
    }

    .login-subtitle {
        text-align: center;
        color: #64748b;
        margin-bottom: 25px;
    }

    /* --------------------------------------------------------
       DASHBOARD SECTIONS
       -------------------------------------------------------- */

    .section-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
    }

    /* --------------------------------------------------------
       FOOTER
       -------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"


# ============================================================
# LOAD MODEL
# IMPORTANT:
# The model was saved using joblib.dump()
# Therefore we MUST use joblib.load()
# ============================================================

try:

    if not MODEL_PATH.exists():

        st.error("❌ Heart disease model file was not found.")

        st.code(
            str(MODEL_PATH),
            language="text"
        )

        st.info(
            "Please run your model training script once so "
            "heart_disease_model.pkl is created."
        )

        st.stop()

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error("❌ Unable to load the heart disease model.")

    st.error(
        "The model file may be corrupted or incompatible."
    )

    st.code(
        str(e),
        language="text"
    )

    st.info(
        "Run the training script again to create a fresh "
        "heart_disease_model.pkl file."
    )

    st.stop()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:

    create_table()

except Exception as e:

    st.error("❌ Database initialization failed.")

    st.code(
        str(e),
        language="text"
    )

    st.stop()


# ============================================================
# SESSION SETTINGS
# ============================================================

SESSION_TIMEOUT_SECONDS = 60


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "authenticated": False,
    "username": None,
    "login_time": None,
    "prediction_done": False,
    "selected_patient_id": None
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# PASSWORD VERIFICATION
# ============================================================

def verify_password(password, stored_hash):

    try:

        algorithm, iterations, salt, stored_key = (
            stored_hash.split("$")
        )

        if algorithm != "pbkdf2_sha256":

            return False

        iterations = int(iterations)

        derived_key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            iterations
        )

        return hmac.compare_digest(
            derived_key.hex(),
            stored_key
        )

    except Exception:

        return False


# ============================================================
# LOGIN FUNCTION
# ============================================================

def login_user(username, password):

    username = username.strip().lower()

    if not username or not password:

        return False, "Please enter username and password."

    try:

        failed_attempts = get_recent_failed_attempts(
            username
        )

    except Exception:

        failed_attempts = 0

    if failed_attempts >= 5:

        return False, (
            "Too many failed login attempts. "
            "Please wait before trying again."
        )

    try:

        user = get_user(username)

    except Exception as e:

        return False, f"Database error: {e}"

    if user is None:

        try:

            log_action(
                username,
                "LOGIN_FAILED"
            )

        except Exception:

            pass

        return False, "Invalid username or password."

    stored_username = user[0]
    stored_hash = user[1]
    is_active = user[2]

    if not is_active:

        return False, "This account is inactive."

    if not verify_password(
        password,
        stored_hash
    ):

        try:

            log_action(
                username,
                "LOGIN_FAILED"
            )

        except Exception:

            pass

        return False, "Invalid username or password."

    try:

        log_action(
            stored_username,
            "LOGIN_SUCCESS"
        )

    except Exception:

        pass

    return True, stored_username


# ============================================================
# SESSION TIMER
# ============================================================

def get_remaining_seconds():

    if st.session_state.login_time is None:

        return 0

    now = datetime.now(timezone.utc)

    elapsed = (
        now -
        st.session_state.login_time
    ).total_seconds()

    remaining = (
        SESSION_TIMEOUT_SECONDS -
        elapsed
    )

    return max(
        0,
        int(remaining)
    )


@st.fragment(run_every=1)
def session_timer():

    remaining = get_remaining_seconds()

    if remaining <= 0:

        username = (
            st.session_state.username
            or "unknown"
        )

        try:

            log_action(
                username,
                "SESSION_EXPIRED"
            )

        except Exception:

            pass

        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.login_time = None

        st.rerun(scope="app")

    minutes = remaining // 60
    seconds = remaining % 60

    st.info(
        f"⏱️ Session: {minutes:02d}:{seconds:02d}"
    )


# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.authenticated:

    # Empty top spacing is intentionally minimized.
    # No custom HTML dashboard title/status lines.

    left, center, right = st.columns(
        [1, 1.5, 1]
    )

    with center:

        st.markdown(
            '<div class="login-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="login-title">'
            '❤️ HeartCare AI'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="login-subtitle">'
            'Secure Heart Disease Risk Assessment'
            '</div>',
            unsafe_allow_html=True
        )

        with st.form("login_form"):

            username = st.text_input(
                "Username",
                placeholder="Enter your username"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )

            login_button = st.form_submit_button(
                "🔐 Sign In",
                type="primary",
                use_container_width=True
            )

        if login_button:

            success, result = login_user(
                username,
                password
            )

            if success:

                st.session_state.authenticated = True

                st.session_state.username = result

                st.session_state.login_time = (
                    datetime.now(timezone.utc)
                )

                st.rerun()

            else:

                st.error(result)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("❤️ HeartCare AI")

    st.success(
        f"👤 {st.session_state.username}"
    )

    st.divider()

    st.subheader("⏱️ Session")

    session_timer()

    st.divider()

    st.subheader("🛡️ Security")

    st.write("✓ Secure Login")
    st.write("✓ Password Hashing")
    st.write("✓ Session Timeout")
    st.write("✓ Login Protection")
    st.write("✓ Audit Logging")

    st.divider()

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        try:

            log_action(
                st.session_state.username,
                "LOGOUT"
            )

        except Exception:

            pass

        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.login_time = None

        st.rerun()


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.title("❤️ HeartCare AI")

st.caption(
    "Secure Heart Disease Risk Assessment & Patient Analytics"
)

st.divider()


# ============================================================
# 1. PATIENT ASSESSMENT
# ============================================================

st.header("1️⃣ Patient Assessment")

st.write(
    "Enter patient information and clinical measurements "
    "to calculate heart disease risk."
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.subheader("👤 Patient Information")

c1, c2, c3 = st.columns(3)

with c1:

    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter full name"
    )

with c2:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45
    )

with c3:

    sex = st.selectbox(
        "Sex",
        [
            "Male",
            "Female"
        ]
    )


# ============================================================
# CLINICAL INFORMATION
# ============================================================

st.subheader("🩺 Clinical Information")

c1, c2, c3 = st.columns(3)

with c1:

    cp = st.selectbox(
        "Chest Pain Type",
        [1, 2, 3, 4]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=250
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?",
        [
            "No",
            "Yes"
        ]
    )

with c2:

    restecg = st.selectbox(
        "Resting ECG Result",
        [0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [
            "No",
            "Yes"
        ]
    )

    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=7.0,
        value=1.0,
        step=0.1
    )

with c3:

    slope = st.selectbox(
        "Slope",
        [1, 2, 3]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thalassemia",
        [3, 6, 7]
    )


# ============================================================
# CONVERT VALUES
# ============================================================

sex_value = 1 if sex == "Male" else 0

fbs_value = 1 if fbs == "Yes" else 0

exang_value = 1 if exang == "Yes" else 0


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Heart Disease Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    clean_name = patient_name.strip()

    if not clean_name:

        st.warning(
            "⚠️ Please enter the patient's name."
        )

        st.stop()

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    patient_data = pd.DataFrame(
        [{
            "age": age,
            "sex": sex_value,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs_value,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang_value,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }]
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            patient_data
        )

        probabilities = model.predict_proba(
            patient_data
        )

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            str(e),
            language="text"
        )

        st.stop()

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    disease_probability = (
        probabilities[0][1] * 100
    )

    no_disease_probability = (
        probabilities[0][0] * 100
    )

    confidence = (
        probabilities[0][prediction[0]] * 100
    )

    # --------------------------------------------------------
    # PREDICTION TEXT
    # --------------------------------------------------------

    if prediction[0] == 1:

        prediction_text = "Disease Detected"

    else:

        prediction_text = "No Disease Detected"

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if disease_probability < 30:

        risk_level = "Low Risk"

    elif disease_probability < 60:

        risk_level = "Medium Risk"

    else:

        risk_level = "High Risk"

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    prediction_date = datetime.now().strftime(
        "%d-%b-%Y %I:%M %p"
    )

    # --------------------------------------------------------
    # SAVE PATIENT
    # --------------------------------------------------------

    try:

        patient_id = save_patient(
            patient_name=clean_name,
            age=age,
            sex=sex,
            prediction=prediction_text,
            disease_probability=disease_probability,
            no_disease_probability=no_disease_probability,
            confidence=confidence,
            risk_level=risk_level,
            prediction_date=prediction_date,
            created_by=st.session_state.username
        )

    except Exception as e:

        st.error(
            "❌ Patient record could not be saved."
        )

        st.code(
            str(e),
            language="text"
        )

        st.stop()

    # --------------------------------------------------------
    # AUDIT LOG
    # --------------------------------------------------------

    try:

        log_action(
            st.session_state.username,
            f"PATIENT_PREDICTION PatientID={patient_id}"
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    st.session_state.prediction_done = True

    st.session_state.selected_patient_id = patient_id

    st.rerun()


# ============================================================
# GET PATIENT DATA
# ============================================================

patients = get_patients()


# ============================================================
# 2. PATIENT HISTORY
# ============================================================

if st.session_state.prediction_done:

    st.divider()

    st.header("2️⃣ Patient History")

    st.write(
        "Search patient records and view complete prediction details."
    )

    if patients:

        df = pd.DataFrame(
            patients,
            columns=[
                "Patient ID",
                "Patient Name",
                "Age",
                "Sex",
                "Prediction",
                "Disease Probability",
                "No Disease Probability",
                "Confidence",
                "Risk Level",
                "Prediction Date",
                "Created By"
            ]
        )

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        st.subheader("🔎 Search Patient")

        f1, f2, f3, f4 = st.columns(4)

        with f1:

            search_text = st.text_input(
                "Patient ID or Name",
                placeholder="Example: 2 or Apurva"
            )

        with f2:

            risk_filter = st.selectbox(
                "Risk Level",
                [
                    "All",
                    "Low Risk",
                    "Medium Risk",
                    "High Risk"
                ]
            )

        with f3:

            prediction_filter = st.selectbox(
                "Prediction",
                [
                    "All",
                    "Disease Detected",
                    "No Disease Detected"
                ]
            )

        with f4:

            sex_filter = st.selectbox(
                "Sex",
                [
                    "All",
                    "Male",
                    "Female"
                ]
            )

        # ----------------------------------------------------
        # FILTER
        # ----------------------------------------------------

        filtered_df = df.copy()

        if search_text.strip():

            search_value = (
                search_text.strip().lower()
            )

            name_match = (
                filtered_df["Patient Name"]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_value,
                    na=False
                )
            )

            id_match = (
                filtered_df["Patient ID"]
                .astype(str)
                .str.contains(
                    search_value,
                    na=False
                )
            )

            filtered_df = filtered_df[
                name_match | id_match
            ]

        if risk_filter != "All":

            filtered_df = filtered_df[
                filtered_df["Risk Level"]
                == risk_filter
            ]

        if prediction_filter != "All":

            filtered_df = filtered_df[
                filtered_df["Prediction"]
                == prediction_filter
            ]

        if sex_filter != "All":

            filtered_df = filtered_df[
                filtered_df["Sex"]
                == sex_filter
            ]

        st.caption(
            f"Showing {len(filtered_df)} "
            f"of {len(df)} patient records"
        )

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.subheader(
            "📋 Complete Patient Details"
        )

        table_df = filtered_df.copy()

        table_df["Patient ID"] = (
            table_df["Patient ID"]
            .apply(
                lambda x: f"Patient {x}"
            )
        )

        for column in [
            "Disease Probability",
            "No Disease Probability",
            "Confidence"
        ]:

            table_df[column] = (
                table_df[column]
                .apply(
                    lambda x: f"{float(x):.2f}%"
                )
            )

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # SELECT PATIENT
        # ----------------------------------------------------

        if len(filtered_df) > 0:

            st.subheader(
                "👤 Select Patient"
            )

            patient_options = {}

            for _, row in filtered_df.iterrows():

                patient_id = int(
                    row["Patient ID"]
                )

                name = str(
                    row["Patient Name"]
                )

                patient_options[
                    f"Patient {patient_id} — {name}"
                ] = patient_id

            option_labels = list(
                patient_options.keys()
            )

            selected_label = st.selectbox(
                "Choose a patient",
                option_labels
            )

            selected_id = patient_options[
                selected_label
            ]

            st.session_state.selected_patient_id = (
                selected_id
            )

            # ------------------------------------------------
            # SELECTED PATIENT
            # ------------------------------------------------

            selected_patient = df[
                df["Patient ID"]
                == selected_id
            ]

            patient = selected_patient.iloc[0]

            st.divider()

            st.header(
                "👤 Selected Patient Prediction Details"
            )

            st.subheader(
                f"Patient {int(patient['Patient ID'])}"
            )

            st.write(
                f"**Patient Name:** "
                f"{patient['Patient Name']}"
            )

            # ------------------------------------------------
            # BASIC INFORMATION
            # ------------------------------------------------

            p1, p2, p3, p4 = st.columns(4)

            with p1:

                st.metric(
                    "Patient ID",
                    f"Patient {int(patient['Patient ID'])}"
                )

            with p2:

                st.metric(
                    "Age",
                    f"{int(patient['Age'])} years"
                )

            with p3:

                st.metric(
                    "Sex",
                    patient["Sex"]
                )

            with p4:

                st.metric(
                    "Created By",
                    patient["Created By"]
                )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            st.subheader(
                "🧠 Prediction Result"
            )

            if patient["Prediction"] == "Disease Detected":

                st.error(
                    "⚠️ Disease Detected"
                )

            else:

                st.success(
                    "✅ No Disease Detected"
                )

            # ------------------------------------------------
            # RISK
            # ------------------------------------------------

            if patient["Risk Level"] == "High Risk":

                st.error(
                    "🔴 High Risk"
                )

            elif patient["Risk Level"] == "Medium Risk":

                st.warning(
                    "🟡 Medium Risk"
                )

            else:

                st.success(
                    "🟢 Low Risk"
                )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "📊 Prediction Details"
            )

            r1, r2, r3 = st.columns(3)

            with r1:

                st.metric(
                    "❤️ Disease Probability",
                    f"{float(patient['Disease Probability']):.2f}%"
                )

            with r2:

                st.metric(
                    "✅ No Disease Probability",
                    f"{float(patient['No Disease Probability']):.2f}%"
                )

            with r3:

                st.metric(
                    "🎯 Confidence",
                    f"{float(patient['Confidence']):.2f}%"
                )

            # ------------------------------------------------
            # PROBABILITY CHART
            # ------------------------------------------------

            st.subheader(
                "📈 Patient Probability"
            )

            profile_chart = pd.DataFrame(
                {
                    "Probability": [
                        float(
                            patient[
                                "Disease Probability"
                            ]
                        ),
                        float(
                            patient[
                                "No Disease Probability"
                            ]
                        )
                    ]
                },
                index=[
                    "Disease",
                    "No Disease"
                ]
            )

            st.bar_chart(
                profile_chart
            )

            st.info(
                f"🕒 Prediction Date: "
                f"{patient['Prediction Date']}"
            )

            st.warning(
                "⚕️ This machine-learning prediction "
                "is for educational/project purposes "
                "and is not a medical diagnosis."
            )

        else:

            st.warning(
                "No patient matches your search."
            )

    else:

        st.info(
            "No patient prediction has been recorded yet."
        )


# ============================================================
# 3. ANALYTICS DASHBOARD
# ============================================================

if st.session_state.prediction_done and patients:

    analytics_df = pd.DataFrame(
        patients,
        columns=[
            "Patient ID",
            "Patient Name",
            "Age",
            "Sex",
            "Prediction",
            "Disease Probability",
            "No Disease Probability",
            "Confidence",
            "Risk Level",
            "Prediction Date",
            "Created By"
        ]
    )

    st.divider()

    st.header(
        "3️⃣ Analytics Dashboard"
    )

    st.write(
        "Overview of patient predictions, risk levels, "
        "probabilities and model confidence."
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_patients = len(
        analytics_df
    )

    disease_count = (
        analytics_df["Prediction"]
        == "Disease Detected"
    ).sum()

    no_disease_count = (
        analytics_df["Prediction"]
        == "No Disease Detected"
    ).sum()

    low_risk = (
        analytics_df["Risk Level"]
        == "Low Risk"
    ).sum()

    medium_risk = (
        analytics_df["Risk Level"]
        == "Medium Risk"
    ).sum()

    high_risk = (
        analytics_df["Risk Level"]
        == "High Risk"
    ).sum()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        st.metric(
            "👥 Total Patients",
            total_patients
        )

    with a2:

        st.metric(
            "❤️ Disease Detected",
            disease_count
        )

    with a3:

        st.metric(
            "✅ No Disease",
            no_disease_count
        )

    with a4:

        st.metric(
            "🔴 High Risk",
            high_risk
        )

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    chart1, chart2 = st.columns(2)

    with chart1:

        st.subheader(
            "📊 Prediction Distribution"
        )

        prediction_chart = pd.Series(
            {
                "Disease Detected": disease_count,
                "No Disease Detected": no_disease_count
            }
        )

        st.bar_chart(
            prediction_chart
        )

    with chart2:

        st.subheader(
            "🚦 Risk Distribution"
        )

        risk_chart = pd.Series(
            {
                "Low Risk": low_risk,
                "Medium Risk": medium_risk,
                "High Risk": high_risk
            }
        )

        st.bar_chart(
            risk_chart
        )

    # --------------------------------------------------------
    # DISEASE PROBABILITY
    # --------------------------------------------------------

    st.subheader(
        "❤️ Disease Probability by Patient"
    )

    probability_chart = analytics_df[
        [
            "Patient ID",
            "Disease Probability"
        ]
    ].set_index(
        "Patient ID"
    )

    st.line_chart(
        probability_chart
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    st.subheader(
        "🎯 Model Confidence by Patient"
    )

    confidence_chart = analytics_df[
        [
            "Patient ID",
            "Confidence"
        ]
    ].set_index(
        "Patient ID"
    )

    st.line_chart(
        confidence_chart
    )

    # --------------------------------------------------------
    # AGE AND SEX
    # --------------------------------------------------------

    chart3, chart4 = st.columns(2)

    with chart3:

        st.subheader(
            "👥 Age Distribution"
        )

        age_chart = (
            analytics_df["Age"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            age_chart
        )

    with chart4:

        st.subheader(
            "👤 Sex Distribution"
        )

        sex_chart = (
            analytics_df["Sex"]
            .value_counts()
        )

        st.bar_chart(
            sex_chart
        )

    # --------------------------------------------------------
    # AVERAGES
    # --------------------------------------------------------

    st.divider()

    av1, av2, av3 = st.columns(3)

    with av1:

        st.metric(
            "Average Disease Probability",
            f"{analytics_df['Disease Probability'].mean():.2f}%"
        )

    with av2:

        st.metric(
            "Average Confidence",
            f"{analytics_df['Confidence'].mean():.2f}%"
        )

    with av3:

        st.metric(
            "Average Patient Age",
            f"{analytics_df['Age'].mean():.1f} years"
        )


# ============================================================
# DATABASE MANAGEMENT
# ============================================================

if st.session_state.prediction_done:

    st.divider()

    with st.expander(
        "⚙️ Database Management"
    ):

        st.warning(
            "Deleting patient records is permanent."
        )

        if st.button(
            "🗑️ Clear All Patient Records",
            type="secondary"
        ):

            try:

                clear_patients()

                log_action(
                    st.session_state.username,
                    "CLEARED_ALL_PATIENT_RECORDS"
                )

            except Exception as e:

                st.error(
                    f"Unable to clear records: {e}"
                )

                st.stop()

            st.session_state.prediction_done = False

            st.session_state.selected_patient_id = None

            st.success(
                "All patient records have been deleted."
            )

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        ❤️ <b>HeartCare AI</b><br>
        Secure Heart Disease Prediction & Patient Analytics<br>
        Machine-learning healthcare project • Educational purposes only
    </div>
    """,
    unsafe_allow_html=True
)
