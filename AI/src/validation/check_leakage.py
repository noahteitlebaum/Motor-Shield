import numpy as np
import os

def check_leakage():
    dataset_path = '../../artifacts/dataset.npz'
    if not os.path.exists(dataset_path):
        dataset_path = 'artifacts/dataset.npz' # Fallback
    data = np.load(dataset_path)
    X_train = data['X_train']
    X_test = data['X_test']
    
    print(f"Checking for leakage between {len(X_train)} train and {len(X_test)} test samples...")
    
    # Check for exact duplicates (unlikely due to noise, but good sanity check)
    # We can't broadcast 3000x600x200x6 (memory), so loop
    
    duplicates = 0
    near_duplicates = 0
    threshold = 1e-6
    
    # Flatten windows for comparison: (N, 1200)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Use a small subset to check if any test sample is suspiciously close to any train sample
    # (If base signal is identical and only noise differs, distance should be related to noise variance)
    
    for i in range(len(X_test)):
        test_vec = X_test_flat[i]
        
        # Vectorized distance to all train
        # |a - b|
        dists = np.abs(X_train_flat - test_vec).max(axis=1) # Max diff per window
        
        if np.any(dists == 0):
            duplicates += 1
        if np.any(dists < threshold):
            near_duplicates += 1
            
        if i % 100 == 0:
            print(f"Checked {i}/{len(X_test)}...")

    print(f"\nResults:")
    print(f"Exact Duplicates: {duplicates}")
    print(f"Near Duplicates (Diff < {threshold}): {near_duplicates}")
    
    if duplicates == 0 and near_duplicates == 0:
        print("PASS: No leakage detected.")
    else:
        print("FAIL: Leakage detected!")

if __name__ == "__main__":
    check_leakage()
