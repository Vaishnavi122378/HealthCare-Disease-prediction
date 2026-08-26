import joblib
from pathlib import Path

model_path = Path("heart_disease_model.pkl")

print("Model path:", model_path.resolve())
print("Model exists:", model_path.exists())

if model_path.exists():
    print("Model size:", model_path.stat().st_size, "bytes")

    model = joblib.load(model_path)

    print("Model loaded successfully!")
    print("Model type:", type(model))
    print("Model classes:", model.classes_)
else:
    print("ERROR: Model file does not exist.")