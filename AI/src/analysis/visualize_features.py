import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_sample():
    dataset_path = '../../artifacts/dataset.npz'
    if not os.path.exists(dataset_path):
        # Try fallback paths from different execution contexts
        paths = ['../../artifacts/dataset.npz', 'artifacts/dataset.npz', 'AI/artifacts/dataset.npz']
        for p in paths:
            if os.path.exists(p):
                dataset_path = p
                break
    
    try:
        data = np.load(dataset_path)
    except FileNotFoundError:
        print(f"Could not find dataset at {dataset_path}")
        return

    X_train = data['X_train'] # Shape: (N_samples, Window_Len, Channels)
    y_train = data['y_train']
    
    # Pick a random sample
    idx = np.random.randint(0, len(X_train))
    sample = X_train[idx] # (200, 6) or similar
    label = y_train[idx]
    
    # Check shape to confirm structure
    # Expected: (Window_Len, 6) based on generate_dataset.py df[features].values
    # But generate_dataset appends window_data = data[start:end], so it is (Window, Features)
    # train_model.py might transpose it. detailed check:
    # generate_dataset.py saves flat_windows = [w ...], so array is (N, Window, Features)
    
    print(f"Dataset Shape: {X_train.shape}")
    print(f"Sample Index: {idx}")
    print(f"Label: {label}")
    print(f"Window Shape: {sample.shape}")
    
    # Features as defined in generate_dataset.py
    feature_names = ['Ia', 'Ib', 'Ic', 'va', 'vb', 'vc']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Currents
    axes[0].plot(sample[:, 0], label='Ia (A)', color='r')
    axes[0].plot(sample[:, 1], label='Ib (A)', color='g')
    axes[0].plot(sample[:, 2], label='Ic (A)', color='b')
    axes[0].set_title(f"Currents - Label: {label}")
    axes[0].set_ylabel("Current (A)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Voltages
    axes[1].plot(sample[:, 3], label='va (V)', color='r', linestyle='--')
    axes[1].plot(sample[:, 4], label='vb (V)', color='g', linestyle='--')
    axes[1].plot(sample[:, 5], label='vc (V)', color='b', linestyle='--')
    axes[1].set_title("Phase Voltages (Derived)")
    axes[1].set_xlabel("Time Steps (Samples)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].legend()
    axes[1].grid(True)
    
    output_file = 'model_features_vis.png'
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    visualize_sample()
