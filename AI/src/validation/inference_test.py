import numpy as np
from inference import MotorFaultPredictor
from sklearn.metrics import classification_report

# 1. Initialize Predictor (Point to your improved model)
predictor = MotorFaultPredictor(
    model_path='../../artifacts/transformer/motor_fault_model.pth',
    metadata_path='../../artifacts/transformer/model_metadata.pkl'
)


# 2. Load the Test Split
data = np.load('../../artifacts/dataset.npz')
X_test = data['X_test']
y_test = data['y_test']

print(f"Running inference on {len(X_test)} samples from Test Split...")

# 3. Predict across the split
predictions = []
for i in range(len(X_test)):
    # predict_window expects a single (200, 6) window
    result = predictor.predict_window(X_test[i])
    predictions.append(result['predicted_class'])

# 4. Show Accuracy Report
print("\n--- Inference Report on Pre-split Test Data ---")
print(classification_report(y_test, predictions))