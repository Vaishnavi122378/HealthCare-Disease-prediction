# ❤️ HeartCare AI

### Intelligent Heart Disease Risk Assessment System

HeartCare AI is a machine-learning-based web application that predicts the risk of heart disease from patient health parameters.

The project uses multiple machine learning algorithms, compares their performance, tunes a Random Forest model using GridSearchCV, and provides the final prediction through an interactive Streamlit application.

> ⚠️ **Disclaimer:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 🚀 Features

* ❤️ Heart disease risk prediction
* 🤖 Multiple machine learning models
* 🌲 Random Forest model with hyperparameter tuning
* 📊 Interactive analytics dashboard
* 👤 Patient prediction and record management
* 📋 Patient history
* 🔎 Patient search and filtering
* 🔐 Secure login system
* 🔑 Password hashing using PBKDF2-SHA256
* 🛡️ Failed-login protection
* ⏱️ Automatic session timeout
* 📝 Audit logging
* 🗄️ Database-backed patient records
* 📈 Prediction probability and model confidence
* 🚦 Low, Medium, and High risk classification

---

## 🧠 Machine Learning

The project uses the UCI Heart Disease dataset.

### Dataset Features

The model uses the following 13 features:

| Feature    | Description                 |
| ---------- | --------------------------- |
| `age`      | Patient age                 |
| `sex`      | Patient sex                 |
| `cp`       | Chest pain type             |
| `trestbps` | Resting blood pressure      |
| `chol`     | Serum cholesterol           |
| `fbs`      | Fasting blood sugar         |
| `restecg`  | Resting ECG result          |
| `thalach`  | Maximum heart rate achieved |
| `exang`    | Exercise-induced angina     |
| `oldpeak`  | ST depression               |
| `slope`    | ST segment slope            |
| `ca`       | Number of major vessels     |
| `thal`     | Thalassemia                 |

The original target contains multiple disease severity levels. It is converted into a binary classification problem:

* `0` → No Disease
* `1` → Disease Detected

---

## 🔬 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Assigned meaningful column names.
3. Identified missing values.
4. Handled missing values in `ca` and `thal` using their respective modes.
5. Converted the original target into binary classification.
6. Separated features and target.
7. Split the dataset into training and testing sets using an 80/20 split.
8. Applied `StandardScaler` where required by the model.

---

## 🤖 Models Evaluated

Four machine learning algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

### Model Accuracy

| Model                   | Test Accuracy |
| ----------------------- | ------------: |
| Logistic Regression     |        86.89% |
| Decision Tree           |        73.77% |
| Random Forest           |        88.52% |
| SVM                     |        85.25% |
| **Tuned Random Forest** |    **90.16%** |

The tuned Random Forest model achieved the highest test accuracy and was selected as the final model.

---

## ⚙️ Hyperparameter Tuning

Random Forest hyperparameters were optimized using `GridSearchCV` with 5-fold cross-validation.

The best parameters were:

```text
n_estimators = 100
max_depth = 5
min_samples_split = 5
min_samples_leaf = 1
```

### Results

* Best Cross-Validation Accuracy: **83.44%**
* Tuned Random Forest Test Accuracy: **90.16%**

The final trained model is saved as:

```text
heart_disease_model.pkl
```

---

## 📊 Application Dashboard

The Streamlit application provides four main sections:

### 🏠 Patient Prediction

Users can enter patient health information and generate a heart disease risk assessment.

The application displays:

* Disease probability
* No-disease probability
* Model confidence
* Risk level
* Prediction result
* Patient ID

### 📊 Analytics Dashboard

Provides:

* Total patient count
* Disease prediction statistics
* Risk-level distribution
* Disease probability charts
* Model confidence charts
* Patient age distribution
* Patient sex distribution
* Average disease probability
* Average model confidence
* Average patient age

### 📋 Patient History

Displays previously generated patient predictions and allows users to filter records.

### ⚙️ Database Management

Provides database statistics and controlled deletion of stored patient prediction records.

---

## 🔐 Security Features

HeartCare AI includes several security-related features:

* Username/password authentication
* PBKDF2-SHA256 password verification
* Failed-login attempt tracking
* Login protection after repeated failures
* Session timeout
* Logout functionality
* Audit logging

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Science & Machine Learning

* Pandas
* NumPy
* Scikit-learn
* Joblib

### Web Application

* Streamlit

### Visualization

* Matplotlib
* Streamlit Charts

### Database

* SQLite

### Development Tools

* Git
* GitHub
* Visual Studio Code

---

## 📁 Project Structure

```text
HealthCare-Disease-prediction/
│
├── data/
│   └── processed.cleveland.data
│
├── src/
│   ├── app.py
│   ├── database.py
│   └── ...
│
├── heart_disease_model.pkl
├── requirements.txt
├── setup_admin.py
├── test_model.py
├── .gitignore
└── README.md
```

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

### 2. Navigate into the project

```bash
cd HealthCare-Disease-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run src/app.py
```

The application will open in your browser.

---

## 🔑 Initial Setup

Before using the login system, create the administrator account using:

```bash
python setup_admin.py
```

Follow the prompts to create the username and password.

Then start the application:

```bash
streamlit run src/app.py
```

---

## 🧪 Testing

The project includes a model testing script:

```bash
python test_model.py
```

This can be used to verify that the saved machine learning model loads and produces predictions correctly.

---

## 📈 Key Result

The final tuned Random Forest model achieved:

# **90.16% Test Accuracy**

This model is used by the HeartCare AI application for generating heart disease risk predictions.

---

## 🎯 Project Objective

The main objective of HeartCare AI is to demonstrate how machine learning can be integrated into a practical healthcare-oriented application.

The project combines:

**Data Processing → Machine Learning → Model Evaluation → Hyperparameter Tuning → Prediction → Database → Web Application**

---

## ⚠️ Disclaimer

HeartCare AI is an educational machine learning project.

The predictions generated by this application are **not medical diagnoses** and should not be used for medical decision-making.

Always consult a qualified healthcare professional for medical advice and diagnosis.

---

## 👩‍💻 Author

**Vaishnavi Nagiri**

Built as a Data Science / Machine Learning project using Python and Streamlit.
