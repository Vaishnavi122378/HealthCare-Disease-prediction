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
    get_recent_failed_attempts,
    clear_failed_login_attempts,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"


# ============================================================
# DATABASE
# ============================================================

try:
    create_table()
except Exception as e:
    st.error("Unable to initialize the database.")
    st.exception(e)
    st.stop()


# ============================================================
# SESSION CONFIGURATION
# ============================================================

SESSION_TIMEOUT_SECONDS = 30 * 60


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "authenticated": False,
    "username": None,
    "login_time": None,
    "prediction_done": False,
    "last_patient_id": None,
    "last_patient_name": None,
    "last_prediction": None,
    "last_disease_probability": 0.0,
    "last_no_disease_probability": 0.0,
    "last_confidence": 0.0,
    "last_risk_level": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PASSWORD VERIFICATION
# ============================================================

def verify_password(password, stored_hash):
    try:
        algorithm, iterations, salt, stored_key = stored_hash.split("$")

        if algorithm != "pbkdf2_sha256":
            return False

        iterations = int(iterations)

        derived_key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            iterations,
        )

        return hmac.compare_digest(
            derived_key.hex(),
            stored_key,
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
        failed_attempts = get_recent_failed_attempts(username)
    except Exception:
        failed_attempts = 0

    if failed_attempts >= 5:
        return (
            False,
            "Too many failed login attempts. "
            "Please use the reset_login.py script "
            "to reset the login protection.",
        )

    try:
        user = get_user(username)
    except Exception:
        return False, "Unable to access the user database."

    if user is None:
        try:
            log_action(username, "LOGIN_FAILED")
        except Exception:
            pass

        return False, "Invalid username or password."

    stored_username = user[0]
    stored_hash = user[1]
    is_active = user[2]

    if not is_active:
        return False, "This account is inactive."

    if not verify_password(password, stored_hash):
        try:
            log_action(username, "LOGIN_FAILED")
        except Exception:
            pass

        return False, "Invalid username or password."

    try:
        clear_failed_login_attempts(username)
        log_action(stored_username, "LOGIN_SUCCESS")
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

    remaining = SESSION_TIMEOUT_SECONDS - elapsed

    return max(0, int(remaining))


@st.fragment(run_every=1)
def session_timer():
    remaining_seconds = get_remaining_seconds()

    if remaining_seconds <= 0:
        username = st.session_state.username or "unknown"

        try:
            log_action(username, "SESSION_EXPIRED")
        except Exception:
            pass

        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.login_time = None
        st.session_state.prediction_done = False

        st.rerun(scope="app")

    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60

    st.caption(
        f"Session remaining: {minutes:02d}:{seconds:02d}"
    )


# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.authenticated:

    st.title("❤️ HeartCare AI")

    st.subheader("Secure Heart Disease Risk Assessment")

    st.write(
        "Sign in to access the HeartCare AI healthcare "
        "prediction dashboard."
    )

    st.divider()

    st.info(
        "🔒 Secure access with password protection, "
        "login monitoring, session timeout, and audit logging."
    )

    with st.form("login_form"):

        username = st.text_input(
            "Username",
            placeholder="Enter your username",
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
        )

        login_button = st.form_submit_button(
            "🔐 Sign In",
            type="primary",
            use_container_width=True,
        )

    if login_button:

        success, result = login_user(
            username,
            password,
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
        "❤️ HeartCare AI | Machine Learning Healthcare Project"
    )

    st.caption(
        "For educational and project purposes only. "
        "This system is not a medical diagnosis."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error(
        "Unable to load the heart disease model."
    )

    st.exception(e)
    st.stop()


# ============================================================
# LOAD PATIENT RECORDS
# ============================================================

PATIENT_COLUMNS = [
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
    "Created By",
]


try:

    all_patients = get_patients()

except Exception as e:

    st.error(
        "Unable to retrieve patient records."
    )

    st.exception(e)

    all_patients = []


if all_patients:

    all_patients_df = pd.DataFrame(
        all_patients,
        columns=PATIENT_COLUMNS,
    )

else:

    all_patients_df = pd.DataFrame(
        columns=PATIENT_COLUMNS
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("❤️ HeartCare AI")

    st.caption("Heart Disease Risk Assessment")

    st.success(
        f"👤 {st.session_state.username}"
    )

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select a page",
        [
            "🏠 Patient Prediction",
            "📊 Analytics Dashboard",
            "📋 Patient History",
            "⚙️ Database Management",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Patient Filters")

    search_patient = st.text_input(
        "Search by name",
        placeholder="Enter patient name",
    )

    sex_filter = st.selectbox(
        "Sex",
        ["All", "Male", "Female"],
    )

    prediction_filter = st.selectbox(
        "Prediction",
        [
            "All",
            "Disease Detected",
            "No Disease Detected",
        ],
    )

    risk_filter = st.selectbox(
        "Risk Level",
        [
            "All",
            "Low Risk",
            "Medium Risk",
            "High Risk",
        ],
    )

    st.divider()

    st.subheader("Session")

    session_timer()

    st.divider()

    if st.button(
        "🚪 Logout",
        use_container_width=True,
    ):

        try:
            log_action(
                st.session_state.username,
                "LOGOUT",
            )
        except Exception:
            pass

        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.login_time = None
        st.session_state.prediction_done = False

        st.rerun()

    st.divider()

    st.subheader("Security")

    st.caption("✅ Secure Login")
    st.caption("✅ Password Hashing")
    st.caption("✅ Session Timeout")
    st.caption("✅ Login Protection")
    st.caption("✅ Audit Logging")


# ============================================================
# FILTER PATIENT RECORDS
# ============================================================

filtered_df = all_patients_df.copy()

if not filtered_df.empty:

    if search_patient.strip():

        filtered_df = filtered_df[
            filtered_df["Patient Name"]
            .astype(str)
            .str.contains(
                search_patient.strip(),
                case=False,
                na=False,
            )
        ]

    if sex_filter != "All":

        filtered_df = filtered_df[
            filtered_df["Sex"] == sex_filter
        ]

    if prediction_filter != "All":

        filtered_df = filtered_df[
            filtered_df["Prediction"]
            == prediction_filter
        ]

    if risk_filter != "All":

        filtered_df = filtered_df[
            filtered_df["Risk Level"]
            == risk_filter
        ]


# ============================================================
# DISPLAY DATAFRAME HELPER
# ============================================================

def prepare_display_dataframe(dataframe):

    display_df = dataframe.copy()

    if display_df.empty:
        return display_df

    if "Patient ID" in display_df.columns:

        display_df["Patient ID"] = (
            display_df["Patient ID"]
            .apply(lambda x: f"Patient {x}")
        )

    if "Disease Probability" in display_df.columns:

        display_df["Disease Probability"] = (
            display_df["Disease Probability"]
            .apply(lambda x: f"{float(x):.2f}%")
        )

    if "No Disease Probability" in display_df.columns:

        display_df["No Disease Probability"] = (
            display_df["No Disease Probability"]
            .apply(lambda x: f"{float(x):.2f}%")
        )

    if "Confidence" in display_df.columns:

        display_df["Confidence"] = (
            display_df["Confidence"]
            .apply(lambda x: f"{float(x):.2f}%")
        )

    display_df = display_df.drop(
        columns=["Created By"],
        errors="ignore",
    )

    return display_df


# ============================================================
# ACTIVE FILTER DISPLAY
# ============================================================

def show_active_filters():

    active_filters = []

    if search_patient.strip():
        active_filters.append(
            f"Name: {search_patient}"
        )

    if sex_filter != "All":
        active_filters.append(
            f"Sex: {sex_filter}"
        )

    if prediction_filter != "All":
        active_filters.append(
            f"Prediction: {prediction_filter}"
        )

    if risk_filter != "All":
        active_filters.append(
            f"Risk: {risk_filter}"
        )

    if active_filters:

        st.info(
            "🔎 Active filters: "
            + " | ".join(active_filters)
        )

    else:

        st.caption(
            "Showing all patient records."
        )


# ============================================================
# PAGE 1 — PATIENT PREDICTION
# ============================================================

if page == "🏠 Patient Prediction":

    st.title("❤️ Patient Prediction")

    st.write(
        "Enter patient information to generate a "
        "machine-learning based heart disease risk assessment."
    )

    st.warning(
        "⚠️ Educational/project use only. "
        "This application does not provide a medical diagnosis."
    )

    st.divider()

    st.header("👤 Patient Information")

    patient_col1, patient_col2 = st.columns(2)

    with patient_col1:

        patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter patient's full name",
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45,
            step=1,
        )

    with patient_col2:

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"],
        )

    st.divider()

    st.header("🫀 Clinical Information")

    clinical_col1, clinical_col2 = st.columns(2)

    with clinical_col1:

        cp = st.selectbox(
            "Chest Pain Type",
            [1, 2, 3, 4],
            help="Chest pain category used by the trained model.",
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120,
            step=1,
            help="Resting blood pressure in mm Hg.",
        )

        chol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=700,
            value=250,
            step=1,
            help="Serum cholesterol in mg/dl.",
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl?",
            ["No", "Yes"],
        )

    with clinical_col2:

        restecg = st.selectbox(
            "Resting ECG Result",
            [0, 1, 2],
        )

        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=50,
            max_value=250,
            value=150,
            step=1,
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"],
        )

    st.divider()

    st.header("📈 Additional Heart Parameters")

    additional_col1, additional_col2 = st.columns(2)

    with additional_col1:

        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0,
            max_value=7.0,
            value=1.0,
            step=0.1,
        )

        slope = st.selectbox(
            "ST Segment Slope",
            [1, 2, 3],
        )

    with additional_col2:

        ca = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3],
        )

        thal = st.selectbox(
            "Thalassemia",
            [3, 6, 7],
        )

    st.divider()

    st.header("📋 Patient Summary")

    summary1, summary2, summary3, summary4 = st.columns(4)

    with summary1:
        st.metric(
            "Age",
            f"{age} years",
        )

    with summary2:
        st.metric(
            "Blood Pressure",
            f"{trestbps} mmHg",
        )

    with summary3:
        st.metric(
            "Cholesterol",
            f"{chol} mg/dl",
        )

    with summary4:
        st.metric(
            "Max Heart Rate",
            f"{thalach} bpm",
        )

    st.divider()

    st.header("🔍 Generate Prediction")

    predict_button = st.button(
        "🔍 Predict Heart Disease Risk",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        clean_name = patient_name.strip()

        if not clean_name:

            st.warning(
                "Please enter the patient's name."
            )

            st.stop()

        if len(clean_name) < 2:

            st.warning(
                "Patient name must contain at least 2 characters."
            )

            st.stop()

        sex_value = 1 if sex == "Male" else 0
        fbs_value = 1 if fbs == "Yes" else 0
        exang_value = 1 if exang == "Yes" else 0

        patient_data = pd.DataFrame(
            [
                {
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
                    "thal": thal,
                }
            ]
        )

        try:

            prediction = model.predict(
                patient_data
            )

            probabilities = model.predict_proba(
                patient_data
            )

        except Exception as e:

            st.error(
                "Unable to generate the prediction."
            )

            st.exception(e)
            st.stop()

        disease_probability = (
            probabilities[0][1] * 100
        )

        no_disease_probability = (
            probabilities[0][0] * 100
        )

        confidence = (
            probabilities[0][prediction[0]]
            * 100
        )

        if prediction[0] == 1:

            prediction_text = "Disease Detected"

        else:

            prediction_text = "No Disease Detected"

        if disease_probability < 30:

            risk_level = "Low Risk"

        elif disease_probability < 60:

            risk_level = "Medium Risk"

        else:

            risk_level = "High Risk"

        prediction_date = datetime.now().strftime(
            "%d-%b-%Y %I:%M %p"
        )

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
                created_by=st.session_state.username,
            )

        except Exception as e:

            st.error(
                "Prediction was generated, but the patient "
                "record could not be saved."
            )

            st.exception(e)
            st.stop()

        try:

            log_action(
                st.session_state.username,
                f"PATIENT_PREDICTION PatientID={patient_id}",
            )

        except Exception:
            pass

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

        st.success(
            "✅ Prediction generated and patient record saved successfully."
        )

    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    if st.session_state.prediction_done:

        st.divider()

        st.header("📊 Prediction Result")

        st.success(
            f"Patient ID: Patient {st.session_state.last_patient_id}"
        )

        st.write(
            f"**Patient Name:** "
            f"{st.session_state.last_patient_name}"
        )

        result1, result2, result3 = st.columns(3)

        with result1:

            st.metric(
                "❤️ Disease Probability",
                (
                    f"{st.session_state.last_disease_probability:.2f}%"
                ),
            )

        with result2:

            st.metric(
                "✅ No Disease Probability",
                (
                    f"{st.session_state.last_no_disease_probability:.2f}%"
                ),
            )

        with result3:

            st.metric(
                "🎯 Model Confidence",
                (
                    f"{st.session_state.last_confidence:.2f}%"
                ),
            )

        st.divider()

        risk = st.session_state.last_risk_level

        if risk == "Low Risk":

            st.success("🟢 LOW RISK")

        elif risk == "Medium Risk":

            st.warning("🟡 MEDIUM RISK")

        else:

            st.error("🔴 HIGH RISK")

        if (
            st.session_state.last_prediction
            == "Disease Detected"
        ):

            st.error(
                "⚠️ Heart Disease Detected"
            )

        else:

            st.success(
                "✅ No Heart Disease Detected"
            )

        st.info(
            "This prediction is generated by a machine-learning "
            "model for educational/project purposes and is not "
            "a medical diagnosis."
        )


# ============================================================
# PAGE 2 — ANALYTICS DASHBOARD
# ============================================================

elif page == "📊 Analytics Dashboard":

    st.title("📊 Analytics Dashboard")

    st.write(
        "Explore patient prediction statistics, risk levels, "
        "probabilities, and model confidence."
    )

    show_active_filters()

    st.divider()

    if not filtered_df.empty:

        total = len(filtered_df)

        disease_count = (
            filtered_df["Prediction"]
            == "Disease Detected"
        ).sum()

        no_disease_count = (
            filtered_df["Prediction"]
            == "No Disease Detected"
        ).sum()

        high_risk = (
            filtered_df["Risk Level"]
            == "High Risk"
        ).sum()

        medium_risk = (
            filtered_df["Risk Level"]
            == "Medium Risk"
        ).sum()

        low_risk = (
            filtered_df["Risk Level"]
            == "Low Risk"
        ).sum()

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "👥 Total Patients",
                total,
            )

        with c2:
            st.metric(
                "❤️ Disease Detected",
                disease_count,
            )

        with c3:
            st.metric(
                "✅ No Disease",
                no_disease_count,
            )

        with c4:
            st.metric(
                "🔴 High Risk",
                high_risk,
            )

        st.divider()

        chart1, chart2 = st.columns(2)

        with chart1:

            st.subheader(
                "📊 Disease Prediction Distribution"
            )

            prediction_counts = pd.Series(
                {
                    "Disease Detected": disease_count,
                    "No Disease Detected": no_disease_count,
                }
            )

            st.bar_chart(prediction_counts)

        with chart2:

            st.subheader(
                "🚦 Risk Level Distribution"
            )

            risk_counts = pd.Series(
                {
                    "Low Risk": low_risk,
                    "Medium Risk": medium_risk,
                    "High Risk": high_risk,
                }
            )

            st.bar_chart(risk_counts)

        st.divider()

        st.subheader(
            "📈 Disease Probability by Patient"
        )

        probability_data = filtered_df[
            [
                "Patient ID",
                "Disease Probability",
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

        st.subheader(
            "🎯 Model Confidence by Patient"
        )

        confidence_data = filtered_df[
            [
                "Patient ID",
                "Confidence",
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

        age_col1, age_col2 = st.columns(2)

        with age_col1:

            st.subheader(
                "👥 Patient Age Distribution"
            )

            age_counts = (
                filtered_df["Age"]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(age_counts)

        with age_col2:

            st.subheader(
                "👤 Patient Sex Distribution"
            )

            sex_counts = (
                filtered_df["Sex"]
                .value_counts()
            )

            st.bar_chart(sex_counts)

        st.divider()

        avg1, avg2, avg3 = st.columns(3)

        with avg1:

            avg_disease_probability = (
                filtered_df[
                    "Disease Probability"
                ].mean()
            )

            st.metric(
                "Average Disease Probability",
                f"{avg_disease_probability:.2f}%",
            )

        with avg2:

            avg_confidence = (
                filtered_df[
                    "Confidence"
                ].mean()
            )

            st.metric(
                "Average Model Confidence",
                f"{avg_confidence:.2f}%",
            )

        with avg3:

            avg_age = (
                filtered_df["Age"].mean()
            )

            st.metric(
                "Average Patient Age",
                f"{avg_age:.1f} years",
            )

        st.divider()

        st.subheader(
            "📋 Complete Patient Data"
        )

        dashboard_display = (
            prepare_display_dataframe(
                filtered_df
            )
        )

        st.dataframe(
            dashboard_display,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Showing {len(filtered_df)} of "
            f"{len(all_patients_df)} patient records."
        )

    else:

        st.warning(
            "🔎 No patients match the selected filters."
        )

        st.write(
            "Try clearing the sidebar search or changing the filters."
        )


# ============================================================
# PAGE 3 — PATIENT HISTORY
# ============================================================

elif page == "📋 Patient History":

    st.title("📋 Patient History")

    st.write(
        "Review previously generated heart disease risk predictions."
    )

    show_active_filters()

    st.divider()

    if not filtered_df.empty:

        history_df = (
            prepare_display_dataframe(
                filtered_df
            )
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Showing {len(filtered_df)} patient records."
        )

    else:

        st.info(
            "No patient records match the selected filters."
        )


# ============================================================
# PAGE 4 — DATABASE MANAGEMENT
# ============================================================

elif page == "⚙️ Database Management":

    st.title("⚙️ Database Management")

    st.write(
        "View and manage patient records stored in the "
        "HeartCare AI database."
    )

    st.warning(
        "⚠️ Deleting patient records is permanent."
    )

    st.divider()

    if not all_patients_df.empty:

        db1, db2, db3 = st.columns(3)

        with db1:

            st.metric(
                "Total Records",
                len(all_patients_df),
            )

        with db2:

            disease_total = (
                all_patients_df["Prediction"]
                == "Disease Detected"
            ).sum()

            st.metric(
                "Disease Detected",
                disease_total,
            )

        with db3:

            high_total = (
                all_patients_df["Risk Level"]
                == "High Risk"
            ).sum()

            st.metric(
                "High Risk",
                high_total,
            )

        st.divider()

        st.subheader(
            "📋 Current Patient Records"
        )

        database_display = (
            prepare_display_dataframe(
                all_patients_df
            )
        )

        st.dataframe(
            database_display,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No patient records are currently stored."
        )

    st.divider()

    st.subheader(
        "🗑️ Delete Patient Records"
    )

    st.write(
        "This action permanently removes all patient "
        "prediction records from the database."
    )

    confirm_delete = st.checkbox(
        "I understand that this action cannot be undone."
    )

    if confirm_delete:

        if st.button(
            "🗑️ Clear All Patient Records",
            type="secondary",
        ):

            try:

                clear_patients()

                try:
                    log_action(
                        st.session_state.username,
                        "CLEARED_ALL_PATIENT_RECORDS",
                    )
                except Exception:
                    pass

                st.session_state.prediction_done = False

                st.success(
                    "✅ All patient records have been deleted."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    "Unable to delete patient records."
                )

                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "❤️ HeartCare AI | Secure Heart Disease Prediction System"
)

st.caption(
    "Machine Learning Healthcare Project | "
    "For educational purposes only"
)