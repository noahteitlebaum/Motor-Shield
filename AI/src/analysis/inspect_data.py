import numpy as np

# Load the file
data = np.load('../../artifacts/dataset.npz')

print(f"Files inside archive: {data.files}")

# Inspect each split
for split in ['train', 'val', 'test']:
    X_key = f'X_{split}'
    y_key = f'y_{split}'
    
    if X_key in data:
        X = data[X_key]
        y = data[y_key]
        print(f"\n--- {split.upper()} SPLIT ---")
        print(f"Features (X) shape: {X.shape}") # (Samples, Windows, Channels)
        print(f"Labels (y) shape:   {y.shape}")
        # print out all the feature names if they exist
        if 'feature_names' in data:
            feature_names = data['feature_names']
            print(f"Feature names: {feature_names}")
            
        
        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label}: {count}")