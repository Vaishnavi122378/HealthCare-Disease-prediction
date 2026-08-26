import sqlite3
from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# DATABASE LOCATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "patient_records.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

def create_table():

    connection = get_connection()

    cursor = connection.cursor()

    # ========================================================
    # PATIENT TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_name TEXT NOT NULL,

            age INTEGER NOT NULL,

            sex TEXT NOT NULL,

            prediction TEXT NOT NULL,

            disease_probability REAL NOT NULL,

            no_disease_probability REAL NOT NULL,

            confidence REAL NOT NULL,

            risk_level TEXT NOT NULL,

            prediction_date TEXT NOT NULL,

            created_by TEXT
        )
    """)

    # ========================================================
    # USERS TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT UNIQUE NOT NULL,

            password_hash TEXT NOT NULL,

            created_at TEXT NOT NULL,

            is_active INTEGER DEFAULT 1
        )
    """)

    # ========================================================
    # AUDIT LOG TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT,

            action TEXT NOT NULL,

            event_time TEXT NOT NULL
        )
    """)

    # ========================================================
    # PATIENT TABLE MIGRATION
    # ========================================================

    cursor.execute(
        "PRAGMA table_info(patients)"
    )

    columns = [
        column[1]
        for column in cursor.fetchall()
    ]

    if "patient_name" not in columns:

        cursor.execute("""
            ALTER TABLE patients
            ADD COLUMN patient_name TEXT
        """)

    if "created_by" not in columns:

        cursor.execute("""
            ALTER TABLE patients
            ADD COLUMN created_by TEXT
        """)

    connection.commit()

    connection.close()


# ============================================================
# SAVE PATIENT
# ============================================================

def save_patient(
    patient_name,
    age,
    sex,
    prediction,
    disease_probability,
    no_disease_probability,
    confidence,
    risk_level,
    prediction_date,
    created_by
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO patients (

            patient_name,
            age,
            sex,
            prediction,
            disease_probability,
            no_disease_probability,
            confidence,
            risk_level,
            prediction_date,
            created_by

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        patient_name,
        age,
        sex,
        prediction,
        disease_probability,
        no_disease_probability,
        confidence,
        risk_level,
        prediction_date,
        created_by

    ))

    patient_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return patient_id


# ============================================================
# GET PATIENTS
# ============================================================

def get_patients():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT

            id,
            patient_name,
            age,
            sex,
            prediction,
            disease_probability,
            no_disease_probability,
            confidence,
            risk_level,
            prediction_date,
            created_by

        FROM patients

        ORDER BY id ASC
    """)

    patients = cursor.fetchall()

    connection.close()

    return patients


# ============================================================
# CLEAR PATIENTS
# ============================================================

def clear_patients():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM patients
    """)

    cursor.execute("""
        DELETE FROM sqlite_sequence
        WHERE name = 'patients'
    """)

    connection.commit()

    connection.close()


# ============================================================
# SAVE AUDIT LOG
# ============================================================

def log_action(username, action):

    connection = get_connection()

    cursor = connection.cursor()

    event_time = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    cursor.execute("""
        INSERT INTO audit_logs (
            username,
            action,
            event_time
        )

        VALUES (?, ?, ?)
    """, (
        username,
        action,
        event_time
    ))

    connection.commit()

    connection.close()


# ============================================================
# GET USER
# ============================================================

def get_user(username):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT

            username,
            password_hash,
            is_active

        FROM users

        WHERE username = ?
    """, (
        username,
    ))

    user = cursor.fetchone()

    connection.close()

    return user


# ============================================================
# CREATE USER
# ============================================================

def create_user(username, password_hash):

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    cursor.execute("""
        INSERT INTO users (
            username,
            password_hash,
            created_at,
            is_active
        )

        VALUES (?, ?, ?, 1)
    """, (
        username,
        password_hash,
        created_at
    ))

    connection.commit()

    connection.close()


# ============================================================
# GET RECENT FAILED LOGIN ATTEMPTS
# ============================================================

def get_recent_failed_attempts(username):

    connection = get_connection()

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Find latest successful login
    # --------------------------------------------------------

    cursor.execute("""
        SELECT event_time

        FROM audit_logs

        WHERE username = ?

        AND action = 'LOGIN_SUCCESS'

        ORDER BY id DESC

        LIMIT 1
    """, (
        username,
    ))

    last_success = cursor.fetchone()

    # --------------------------------------------------------
    # Count failures after latest successful login
    # --------------------------------------------------------

    if last_success:

        last_success_time = last_success[0]

        cursor.execute("""
            SELECT COUNT(*)

            FROM audit_logs

            WHERE username = ?

            AND action = 'LOGIN_FAILED'

            AND event_time > ?

            AND event_time >= datetime(
                'now',
                '-10 minutes'
            )
        """, (
            username,
            last_success_time
        ))

    else:

        cursor.execute("""
            SELECT COUNT(*)

            FROM audit_logs

            WHERE username = ?

            AND action = 'LOGIN_FAILED'

            AND event_time >= datetime(
                'now',
                '-10 minutes'
            )
        """, (
            username,
        ))

    count = cursor.fetchone()[0]

    connection.close()

    return count


# ============================================================
# CLEAR FAILED LOGIN ATTEMPTS
# ============================================================

def clear_failed_login_attempts(username):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM audit_logs

        WHERE username = ?

        AND action = 'LOGIN_FAILED'
    """, (
        username,
    ))

    connection.commit()

    connection.close()