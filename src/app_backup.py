import streamlit as st
import joblib
import pandas as pd
import hashlib
import hmac
import matplotlib.pyplot as plt

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
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"


# ============================================================
# DATABASE
# ============================================================

create_table()


# ============================================================
# SESSION TIMEOUT
# ============================================================

# TESTING: 60 seconds
#SESSION_TIMEOUT_SECONDS = 60

# AFTER TESTING, CHANGE TO:
SESSION_TIMEOUT_SECONDS = 30 * 60


# ============================================================
# SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "login_time" not in st.session_state:
    st.session_state.login_time = None

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False


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
# LOGIN
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

    user = get_user(username)

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
        now - st.session_state.login_time
    ).total_seconds()

    remaining = (
        SESSION_TIMEOUT_SECONDS - elapsed
    )

    return max(
        0,
        int(remaining)
    )


@st.fragment(run_every=1)
def session_timer():

    remaining_seconds = get_remaining_seconds()

    if remaining_seconds <= 0:

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
        st.session_state.prediction_done = False

        st.rerun(scope="app")

    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60

    st.info(
        f"⏱️ Session remaining: "
        f"{minutes:02d}:{seconds:02d}"
    )


# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.authenticated:

    st.title("❤️ HeartCare AI")

    st.subheader(
        "Secure Heart Disease Prediction System"
    )

    st.write(
        "Please sign in to continue."
    )

    st.divider()

    with st.form("login_form"):

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        login_button = st.form_submit_button(
            "🔐 Login",
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

            st.session_state.prediction_done = False

            st.rerun()

        else:

            st.error(result)

    st.divider()

    st.caption(
        "🔒 Authorized users only."
    )

    st.stop()


# ============================================================
# PROFESSIONAL LOGIN PAGE
# ============================================================

if not st.session_state.authenticated:

    st.markdown(
        """
        <style>

        .stApp {
            background:
                linear-gradient(
                    135deg,
                    #f4f8fc 0%,
                    #eaf3f8 50%,
                    #f8fbfd 100%
                );
        }

        section[data-testid="stMain"] {
            padding-top: 2rem !important;
        }

        .heart-icon {
            text-align: center;
            font-size: 52px;
            margin-bottom: 5px;
        }

        .brand-title {
            text-align: center;
            font-size: 36px;
            font-weight: 800;
            color: #0f4c5c;
            margin-bottom: 5px;
        }

        .brand-subtitle {
            text-align: center;
            font-size: 15px;
            color: #607d8b;
            margin-bottom: 30px;
        }

        .login-heading {
            text-align: center;
            font-size: 23px;
            font-weight: 700;
            color: #263238;
            margin-top: 10px;
            margin-bottom: 5px;
        }

        .login-description {
            text-align: center;
            color: #78909c;
            font-size: 14px;
            margin-bottom: 25px;
        }

        .security-box {
            background: #eef8fb;
            border-left: 4px solid #1597b5;
            border-radius: 8px;
            padding: 12px 15px;
            margin-top: 20px;
            color: #34515c;
            font-size: 13px;
        }

        .login-footer {
            text-align: center;
            color: #90a4ae;
            font-size: 12px;
            margin-top: 25px;
        }

        label {
            font-weight: 600 !important;
            color: #37474f !important;
        }

        .stFormSubmitButton > button {
            border-radius: 10px !important;
            font-weight: 700 !important;
            min-height: 45px !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )



    # ========================================================
    # BRANDING
    # ========================================================

    st.markdown(
        '<div class="heart-icon">❤️</div>',
           unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-title">HeartCare AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Secure Heart Disease Risk Assessment System'
        '</div>',
        unsafe_allow_html=True
    )
    # ========================================================
    # CENTER LOGIN
    # ========================================================
    left, center, right = st.columns(
        [1, 1.25, 1]
        )

    with center:
    
        st.markdown(
                '<div class="login-heading">'
                '🔐 Secure Login'
                '</div>',
                unsafe_allow_html=True
            )
        st.markdown(
                '<div class="login-description">'
                'Sign in to access the HeartCare AI dashboard'
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # LOGIN FORM
        # ====================================================
        
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
                "🔐  Sign In",
                type="primary",
                use_container_width=True
            )
        


# ====================================================
# LOGIN PROCESSING
# ====================================================

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


        # ====================================================
        # SECURITY INFORMATION
        # ====================================================

        st.markdown(
            """
            <div class="security-box">
                <div class="security-title">
                    🔒 Secure Access
                </div>
                Your account is protected with password hashing,
                login protection, session timeout, and audit logging.
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # LOGIN FOOTER
        # ====================================================

        st.markdown(
            """
            <div class="login-footer">
                ❤️ HeartCare AI<br>
                Machine Learning Healthcare Project<br>
                For educational and project purposes
            </div>
            """,
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
        st.session_state.prediction_done = False

        st.rerun()

    st.divider()

    st.subheader("🛡️ Security")

    st.write("✅ Secure Login")
    st.write("✅ Password Hashing")
    st.write("✅ Session Timeout")
    st.write("✅ Login Protection")
    st.write("✅ Audit Logging")


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "❤️ HeartCare AI"
)

st.subheader(
    "Heart Disease Prediction & Patient Analytics"
)

st.write(
    "Enter patient information below to generate "
    "a machine-learning based heart disease risk prediction."
)


# ============================================================
# STEP 1 — PATIENT DETAILS
# ============================================================

st.divider()

st.header(
    "👤 Step 1 — Patient Details"
)

patient_col1, patient_col2 = st.columns(2)


# ============================================================
# LEFT SIDE
# ============================================================

with patient_col1:

    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient's full name"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45
    )

    sex = st.selectbox(
        "Sex",
        [
            "Male",
            "Female"
        ]
    )

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


# ============================================================
# RIGHT SIDE
# ============================================================

with patient_col2:

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

sex_value = (
    1 if sex == "Male" else 0
)

fbs_value = (
    1 if fbs == "Yes" else 0
)

exang_value = (
    1 if exang == "Yes" else 0
)


# ============================================================
# STEP 2 — PREDICTION
# ============================================================

st.divider()

st.header(
    "🔍 Step 2 — Generate Prediction"
)

predict_button = st.button(
    "🔍 Predict Heart Disease Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION PROCESS
# ============================================================

if predict_button:

    clean_name = patient_name.strip()

    if not clean_name:

        st.warning(
            "⚠️ Please enter the patient's name."
        )

        st.stop()

    if len(clean_name) < 2:

        st.warning(
            "⚠️ Patient name must contain "
            "at least 2 characters."
        )

        st.stop()

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    patient_data = pd.DataFrame([{

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

    }])

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = model.predict(
        patient_data
    )

    probabilities = model.predict_proba(
        patient_data
    )

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
    # RESULT
    # --------------------------------------------------------

    if prediction[0] == 1:

        prediction_text = (
            "Disease Detected"
        )

    else:

        prediction_text = (
            "No Disease Detected"
        )

    # --------------------------------------------------------
    # RISK
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

    prediction_date = (
        datetime.now().strftime(
            "%d-%b-%Y %I:%M %p"
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    try:

        log_action(
            st.session_state.username,
            f"PATIENT_PREDICTION PatientID={patient_id}"
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # STORE RESULT IN SESSION
    # --------------------------------------------------------

    st.session_state.prediction_done = True

    st.session_state.last_patient_id = patient_id

    st.session_state.last_patient_name = clean_name

    st.session_state.last_prediction = prediction_text

    st.session_state.last_disease_probability = (
        disease_probability
    )

    st.session_state.last_no_disease_probability = (
        no_disease_probability
    )

    st.session_state.last_confidence = confidence

    st.session_state.last_risk_level = risk_level

    st.rerun()


# ============================================================
# SHOW RESULT ONLY AFTER PREDICTION
# ============================================================

if st.session_state.prediction_done:

    st.divider()

    st.header(
        "📊 Step 3 — Prediction Result"
    )

    st.success(
        f"Patient ID: "
        f"Patient {st.session_state.last_patient_id}"
    )

    st.write(
        f"**Patient Name:** "
        f"{st.session_state.last_patient_name}"
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    with result_col1:

        st.metric(
            "❤️ Disease Probability",
            f"{st.session_state.last_disease_probability:.2f}%"
        )

    with result_col2:

        st.metric(
            "✅ No Disease Probability",
            f"{st.session_state.last_no_disease_probability:.2f}%"
        )

    with result_col3:

        st.metric(
            "🎯 Confidence",
            f"{st.session_state.last_confidence:.2f}%"
        )

    st.divider()

    risk = st.session_state.last_risk_level

    if risk == "Low Risk":

        st.success(
            "🟢 LOW RISK"
        )

    elif risk == "Medium Risk":

        st.warning(
            "🟡 MEDIUM RISK"
        )

    else:

        st.error(
            "🔴 HIGH RISK"
        )

    if st.session_state.last_prediction == "Disease Detected":

        st.error(
            "⚠️ Heart Disease Detected"
        )

    else:

        st.success(
            "✅ No Heart Disease Detected"
        )

    st.info(
        "This prediction is generated by a machine-learning "
        "model and is intended for educational/project use. "
        "It is not a medical diagnosis."
    )


# ============================================================
# DASHBOARD ONLY AFTER PREDICTION
# ============================================================

if st.session_state.prediction_done:

    st.divider()

    st.header(
        "📈 Step 4 — Healthcare Analytics Dashboard"
    )

    st.write(
        "This dashboard summarizes the patient records "
        "stored in the system."
    )

    dashboard_patients = get_patients()

    if dashboard_patients:

        df = pd.DataFrame(
            dashboard_patients,
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

        # ====================================================
        # SUMMARY
        # ====================================================

        total = len(df)

        disease_count = (
            df["Prediction"]
            == "Disease Detected"
        ).sum()

        no_disease_count = (
            df["Prediction"]
            == "No Disease Detected"
        ).sum()

        high_risk = (
            df["Risk Level"]
            == "High Risk"
        ).sum()

        medium_risk = (
            df["Risk Level"]
            == "Medium Risk"
        ).sum()

        low_risk = (
            df["Risk Level"]
            == "Low Risk"
        ).sum()

        # ====================================================
        # SUMMARY CARDS
        # ====================================================

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "👥 Total Patients",
                total
            )

        with c2:

            st.metric(
                "❤️ Disease Detected",
                disease_count
            )

        with c3:

            st.metric(
                "✅ No Disease",
                no_disease_count
            )

        with c4:

            st.metric(
                "🔴 High Risk",
                high_risk
            )

        st.divider()

        # ====================================================
        # CHART 1 — DISEASE DISTRIBUTION
        # ====================================================

        chart1, chart2 = st.columns(2)

        with chart1:

            st.subheader(
                "📊 Disease Prediction Distribution"
            )

            prediction_counts = pd.Series(
                {
                    "Disease Detected": disease_count,
                    "No Disease Detected": no_disease_count
                }
            )

            st.bar_chart(
                prediction_counts
            )

        # ====================================================
        # CHART 2 — RISK DISTRIBUTION
        # ====================================================

        with chart2:

            st.subheader(
                "🚦 Risk Level Distribution"
            )

            risk_counts = pd.Series(
                {
                    "Low Risk": low_risk,
                    "Medium Risk": medium_risk,
                    "High Risk": high_risk
                }
            )

            st.bar_chart(
                risk_counts
            )

        st.divider()

        # ====================================================
        # CHART 3 — DISEASE PROBABILITY
        # ====================================================

        st.subheader(
            "📈 Disease Probability by Patient"
        )

        probability_data = df[
            [
                "Patient ID",
                "Disease Probability"
            ]
        ].copy()

        probability_data = (
            probability_data
            .set_index("Patient ID")
        )

        st.line_chart(
            probability_data
        )

        st.divider()

        # ====================================================
        # CHART 4 — MODEL CONFIDENCE
        # ====================================================

        st.subheader(
            "🎯 Model Confidence by Patient"
        )

        confidence_data = df[
            [
                "Patient ID",
                "Confidence"
            ]
        ].copy()

        confidence_data = (
            confidence_data
            .set_index("Patient ID")
        )

        st.line_chart(
            confidence_data
        )

        st.divider()

        # ====================================================
        # CHART 5 — AGE DISTRIBUTION
        # ====================================================

        age_col1, age_col2 = st.columns(2)

        with age_col1:

            st.subheader(
                "👥 Patient Age Distribution"
            )

            age_counts = (
                df["Age"]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(
                age_counts
            )

        # ====================================================
        # CHART 6 — SEX DISTRIBUTION
        # ====================================================

        with age_col2:

            st.subheader(
                "👤 Patient Sex Distribution"
            )

            sex_counts = (
                df["Sex"]
                .value_counts()
            )

            st.bar_chart(
                sex_counts
            )

        st.divider()

        # ====================================================
        # AVERAGE VALUES
        # ====================================================

        avg1, avg2, avg3 = st.columns(3)

        with avg1:

            avg_disease_probability = (
                df["Disease Probability"]
                .mean()
            )

            st.metric(
                "Average Disease Probability",
                f"{avg_disease_probability:.2f}%"
            )

        with avg2:

            avg_confidence = (
                df["Confidence"]
                .mean()
            )

            st.metric(
                "Average Model Confidence",
                f"{avg_confidence:.2f}%"
            )

        with avg3:

            avg_age = (
                df["Age"]
                .mean()
            )

            st.metric(
                "Average Patient Age",
                f"{avg_age:.1f} years"
            )


# ============================================================
# PATIENT HISTORY
# ============================================================

if st.session_state.prediction_done:

    st.divider()

    st.header(
        "📋 Step 5 — Patient Prediction History"
    )

    history_patients = get_patients()

    if history_patients:

        history_df = pd.DataFrame(
            history_patients,
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
        # FORMAT ID
        # ----------------------------------------------------

        history_df["Patient ID"] = (
            history_df["Patient ID"]
            .apply(
                lambda x:
                f"Patient {x}"
            )
        )

        # ----------------------------------------------------
        # FORMAT PERCENTAGES
        # ----------------------------------------------------

        history_df[
            "Disease Probability"
        ] = (
            history_df[
                "Disease Probability"
            ]
            .apply(
                lambda x:
                f"{x:.2f}%"
            )
        )

        history_df[
            "No Disease Probability"
        ] = (
            history_df[
                "No Disease Probability"
            ]
            .apply(
                lambda x:
                f"{x:.2f}%"
            )
        )

        history_df[
            "Confidence"
        ] = (
            history_df[
                "Confidence"
            ]
            .apply(
                lambda x:
                f"{x:.2f}%"
            )
        )

        # ----------------------------------------------------
        # HIDE INTERNAL ACCOUNT NAME
        # ----------------------------------------------------

        display_df = history_df.drop(
            columns=["Created By"],
            errors="ignore"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No patient predictions have been recorded yet."
        )


# ============================================================
# DATABASE MANAGEMENT
# ============================================================

if st.session_state.prediction_done:

    st.divider()

    st.subheader(
        "⚙️ Database Management"
    )

    st.warning(
        "Deleting patient records is permanent."
    )

    if st.button(
        "🗑️ Clear All Patient Records"
    ):

        clear_patients()

        try:

            log_action(
                st.session_state.username,
                "CLEARED_ALL_PATIENT_RECORDS"
            )

        except Exception:
            pass

        st.session_state.prediction_done = False

        st.success(
            "All patient records have been deleted."
        )

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "❤️ HeartCare AI | "
    "Secure Heart Disease Prediction System | "
    "Machine Learning Healthcare Project"
)
