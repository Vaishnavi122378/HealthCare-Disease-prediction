import pandas as pd

file_path = "data/processed.cleveland.data"

columns = [
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
    "num"
]

df = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    na_values="?"
)

print(df.head())
print("\nShape of dataset:", df.shape)
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
# Handle missing values
df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# Handle missing values
df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
print("\nTarget Value Counts:")
print(df["num"].value_counts().sort_index())
# Convert target into binary classification
df["target"] = (df["num"] > 0).astype(int)

print("\nBinary Target Distribution:")
print(df["target"].value_counts().sort_index())
print("\nUnique Values:")
print("sex:", df["sex"].unique())
print("cp:", df["cp"].unique())
print("fbs:", df["fbs"].unique())
print("restecg:", df["restecg"].unique())
print("exang:", df["exang"].unique())
print("slope:", df["slope"].unique())
print("ca:", df["ca"].unique())
print("thal:", df["thal"].unique())
# Separate features and target

X = df.drop(columns=["num", "target"])
y = df["target"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaled Training Data Shape:", X_train_scaled.shape)
print("Scaled Testing Data Shape:", X_test_scaled.shape)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("Predicted Values:")
print(y_pred)
print("Actual Values:")
print(y_test.values)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================
# DECISION TREE MODEL
# ============================================

# Step 1: Import Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Step 2: Create the model
dt_model = DecisionTreeClassifier(random_state=42)

# Step 3: Train the model
dt_model.fit(X_train, y_train)

# Step 4: Make predictions
dt_pred = dt_model.predict(X_test)

# Step 5: Calculate accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Accuracy Percentage:", dt_accuracy * 100)

# Step 6: Classification Report
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred))

# Step 7: Confusion Matrix
print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))


# ============================================
# RANDOM FOREST MODEL
# ============================================

# Step 1: Import Random Forest
from sklearn.ensemble import RandomForestClassifier

# Step 2: Create the model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 3: Train the model
rf_model.fit(X_train, y_train)

# Step 4: Make predictions
rf_pred = rf_model.predict(X_test)

# Step 5: Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Accuracy Percentage:", rf_accuracy * 100)

# Step 6: Classification Report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Step 7: Confusion Matrix
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# ============================================
# SUPPORT VECTOR MACHINE (SVM) MODEL
# ============================================

# Step 1: Import SVM
from sklearn.svm import SVC

# Step 2: Create the model
svm_model = SVC(kernel="rbf", random_state=42)

# Step 3: Train the model
svm_model.fit(X_train_scaled, y_train)

# Step 4: Make predictions
svm_pred = svm_model.predict(X_test_scaled)

# Step 5: Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", svm_accuracy)
print("SVM Accuracy Percentage:", svm_accuracy * 100)

# Step 6: Classification Report
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred))

# Step 7: Confusion Matrix
print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))


# ============================================
# MODEL COMPARISON
# ============================================

model_results = {
    "Logistic Regression": accuracy,
    "Decision Tree": dt_accuracy,
    "Random Forest": rf_accuracy,
    "SVM": svm_accuracy
}

print("Model Accuracy Comparison:\n")

for model_name, model_accuracy in model_results.items():
    print(f"{model_name}: {model_accuracy * 100:.2f}%")


    feature_importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)




# ============================================
# RANDOM FOREST HYPERPARAMETER TUNING
# ============================================

from sklearn.model_selection import GridSearchCV

# Create Random Forest
rf_tuning = RandomForestClassifier(random_state=42)

# Parameters to test
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=rf_tuning,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train GridSearch
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:")
print(grid_search.best_params_)

# Best cross-validation score
print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)


# ============================================
# TEST THE TUNED RANDOM FOREST
# ============================================

# Get the best model
best_rf_model = grid_search.best_estimator_

# Make predictions on test data
best_rf_pred = best_rf_model.predict(X_test)

# Calculate accuracy
best_rf_accuracy = accuracy_score(y_test, best_rf_pred)

print("Tuned Random Forest Accuracy:", best_rf_accuracy)
print(
    "Tuned Random Forest Accuracy Percentage:",
    best_rf_accuracy * 100
)

# Classification Report
print("\nTuned Random Forest Classification Report:")
print(classification_report(y_test, best_rf_pred))

# Confusion Matrix
print("\nTuned Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, best_rf_pred))



# ============================================
# SAVE FINAL MODEL
# ============================================

import joblib

# Save the best Random Forest model
joblib.dump(best_rf_model, "heart_disease_model.pkl")

print("Final model saved successfully!")