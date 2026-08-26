import joblib
import pandas as pd

# ============================================
# LOAD SAVED MODEL
# ============================================

model = joblib.load("heart_disease_model.pkl")

print("\n============================================")
print("       HEART DISEASE PREDICTION SYSTEM")
print("============================================")
print("Model loaded successfully!")
print("\nPlease enter the patient's information.")
print("============================================")


# ============================================
# GET PATIENT INFORMATION
# ============================================

age = float(input("Enter Age: "))

sex = int(input(
    "Enter Sex (1 = Male, 0 = Female): "
))

cp = int(input(
    "Enter Chest Pain Type (1, 2, 3, 4): "
))

trestbps = float(input(
    "Enter Resting Blood Pressure: "
))

chol = float(input(
    "Enter Cholesterol Level: "
))

fbs = int(input(
    "Enter Fasting Blood Sugar (1 = Yes, 0 = No): "
))

restecg = int(input(
    "Enter Resting ECG Result (0, 1, 2): "
))

thalach = float(input(
    "Enter Maximum Heart Rate Achieved: "
))

exang = int(input(
    "Enter Exercise Induced Angina (1 = Yes, 0 = No): "
))

oldpeak = float(input(
    "Enter ST Depression (oldpeak): "
))

slope = int(input(
    "Enter Slope (1, 2, 3): "
))

ca = int(input(
    "Enter Number of Major Vessels (0, 1, 2, 3): "
))

thal = int(input(
    "Enter Thalassemia Value (3, 6, 7): "
))


# ============================================
# CREATE PATIENT DATA
# ============================================

patient = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])


# ============================================
# MAKE PREDICTION
# ============================================

prediction = model.predict(patient)

probability = model.predict_proba(patient)

disease_probability = probability[0][1] * 100
no_disease_probability = probability[0][0] * 100

confidence = probability[0][prediction[0]] * 100


# ============================================
# DISPLAY RESULT
# ============================================

print("\n\n============================================")
print("           PREDICTION RESULT")
print("============================================")

if prediction[0] == 1:

    print("Prediction           : Disease Detected")

else:

    print("Prediction           : No Disease Detected")


print(f"Disease Probability  : {disease_probability:.2f}%")
print(f"No Disease Probability: {no_disease_probability:.2f}%")
print(f"Confidence           : {confidence:.2f}%")

print("============================================")


# ============================================
# SIMPLE INTERPRETATION
# ============================================

if prediction[0] == 1:

    print("\n⚠️ The model predicts a higher likelihood")
    print("   of heart disease for this patient.")

else:

    print("\n✅ The model predicts a lower likelihood")
    print("   of heart disease for this patient.")

print("\nNote: This is an ML prediction for the")
print("project and is NOT a medical diagnosis.")
print("============================================")


