import os
import glob
import pandas as pd
import numpy as np
import random
import argparse
from scipy import signal
from scipy.fft import fft, fftfreq

# Constants
VDC_BASE = 24.0
FAULT_TIME = 0.2
SAMPLE_RATE_HZ = 5000  # 1 / 0.0002
WINDOW_SIZE_MS = 40
OVERLAP_PCT = 0.50

# Calculated values
WINDOW_LEN = int((WINDOW_SIZE_MS / 1000) * SAMPLE_RATE_HZ)  # 200 samples
STRIDE = int(WINDOW_LEN * (1 - OVERLAP_PCT))  # 100 samples

print(f"Configuration: Fs={SAMPLE_RATE_HZ}Hz, Window={WINDOW_LEN} samples, Stride={STRIDE} samples")

class DataAugmentor:
    def __init__(self):
        pass

    def add_gaussian_noise(self, df):
        """
        Add Gaussian noise per sample.
        Ia, Ib, Ic: +/- 3-5%
        Vdc: +/- 0.5-1%
        """
        df_aug = df.copy()
        
        # Currents
        for col in ['Ia', 'Ib', 'Ic']:
            noise_pct = np.random.uniform(0.03, 0.05, size=len(df))
            noise = df[col] * noise_pct * np.random.normal(0, 1, size=len(df))
            df_aug[col] = df[col] + noise
            
        # Voltage
        col = 'Vdc'
        noise_pct = np.random.uniform(0.005, 0.01, size=len(df))
        noise = df[col] * noise_pct * np.random.normal(0, 1, size=len(df))
        df_aug[col] = df[col] + noise
        
        return df_aug

    def apply_gain_variation(self, df):
        """
        Gain variation constant per file.
        D new = D old * (1 + n/100)
        Ia, Ib, Ic: +/- 1-3%
        Vdc: +/- 1-3%
        """
        df_aug = df.copy()
        
        # Currents
        for col in ['Ia', 'Ib', 'Ic']:
            n = np.random.uniform(1, 3)
            if np.random.random() < 0.5: n = -n
            gain = 1 + (n / 100.0)
            df_aug[col] = df[col] * gain

        # Voltage
        col = 'Vdc'
        n = np.random.uniform(1, 3)
        if np.random.random() < 0.5: n = -n
        gain = 1 + (n / 100.0)
        df_aug[col] = df[col] * gain
        
        return df_aug

    def add_offset_drift(self, df):
        """
        Linear offset drift.
        Ia, Ib, Ic: +/- 1-3%
        Vdc: +/- 1-2%
        """
        df_aug = df.copy()
        n_samples = len(df)
        t = np.linspace(0, 1, n_samples) # Normalized time 0 to 1
        
        # Currents
        for col in ['Ia', 'Ib', 'Ic']:
            max_drift_pct = np.random.uniform(0.01, 0.03)
            # Drift direction
            if np.random.random() < 0.5: max_drift_pct = -max_drift_pct
            
            scale = df[col].abs().mean()
            drift = scale * max_drift_pct * t
            df_aug[col] = df[col] + drift

        # Voltage
        col = 'Vdc'
        max_drift_pct = np.random.uniform(0.01, 0.02)
        if np.random.random() < 0.5: max_drift_pct = -max_drift_pct
        scale = df[col].abs().mean()
        drift = scale * max_drift_pct * t
        df_aug[col] = df[col] + drift
        
        return df_aug
    
    def add_random_spikes(self, df, spike_prob=0.005):
        """
        Add random voltage/current spikes to simulate electrical transients
        """
        df_aug = df.copy()
        
        for col in ['Ia', 'Ib', 'Ic']:
            spike_mask = np.random.random(len(df)) < spike_prob
            spike_magnitude = np.random.uniform(0.05, 0.15, size=len(df))
            spike_sign = np.random.choice([-1, 1], size=len(df))
            scale = df[col].abs().mean()
            df_aug.loc[spike_mask, col] += spike_sign[spike_mask] * spike_magnitude[spike_mask] * scale
        
        return df_aug
    
    def add_harmonic_distortion(self, df):
        """
        Add harmonic distortion to current signals (simulates non-ideal conditions)
        """
        df_aug = df.copy()
        n_samples = len(df)
        t = np.arange(n_samples)
        
        for col in ['Ia', 'Ib', 'Ic']:
            # Add 3rd and 5th harmonics with small amplitude
            fundamental_freq = 2 * np.pi / WINDOW_LEN  # Approximate fundamental
            harmonic3 = np.random.uniform(0.01, 0.03) * np.sin(3 * fundamental_freq * t + np.random.uniform(0, 2*np.pi))
            harmonic5 = np.random.uniform(0.005, 0.015) * np.sin(5 * fundamental_freq * t + np.random.uniform(0, 2*np.pi))
            
            scale = df[col].abs().mean()
            df_aug[col] = df[col] + scale * (harmonic3 + harmonic5)
        
        return df_aug
    
    def apply_time_warping(self, df, warp_factor=0.05):
        """
        Apply slight time warping (temporal augmentation)
        """
        df_aug = df.copy()
        n_samples = len(df)
        
        # Create a warping curve
        warp = 1 + warp_factor * np.sin(2 * np.pi * np.random.uniform(0.5, 2) * np.linspace(0, 1, n_samples))
        
        # Apply to all numeric columns except time
        for col in ['Ia', 'Ib', 'Ic', 'Vdc', 'da', 'db', 'dc']:
            if col in df_aug.columns:
                # Simple scaling approximation
                df_aug[col] = df_aug[col] * (1 + np.random.uniform(-0.02, 0.02))
        
        return df_aug

class DataProcessor:
    def __init__(self, augmentor, use_fft_features=False):
        self.augmentor = augmentor
        self.use_fft_features = use_fft_features

    def derive_phase_voltages(self, df):
        """
        Derive va, vb, vc from Vdc and duty cycles.
        vaN = (2*da - 1) * Vdc / 2
        vbN = (2*db - 1) * Vdc / 2
        vcN = (2*dc - 1) * Vdc / 2
        vn = (vaN + vbN + vcN) / 3
        va = vaN - vn ...
        """
        # Convert inputs to float just in case
        Vdc = df['Vdc']
        da = df['da']
        db = df['db']
        dc = df['dc']

        vaN = (2 * da - 1) * Vdc / 2
        vbN = (2 * db - 1) * Vdc / 2
        vcN = (2 * dc - 1) * Vdc / 2

        vn = (vaN + vbN + vcN) / 3

        df['va'] = vaN - vn
        df['vb'] = vbN - vn
        df['vc'] = vcN - vn
        
        return df
    
    def extract_fft_features(self, window_data, n_fft_features=10):
        """
        Extract frequency domain features using FFT
        Returns top N frequency components for each channel
        """
        fft_features = []
        
        for channel in range(window_data.shape[1]):
            signal_data = window_data[:, channel]
            
            # Compute FFT
            fft_vals = fft(signal_data)
            fft_magnitude = np.abs(fft_vals)[:len(signal_data)//2]  # Take positive frequencies
            
            # Get top N frequency magnitudes
            top_indices = np.argsort(fft_magnitude)[-n_fft_features:]
            top_magnitudes = fft_magnitude[top_indices]
            
            fft_features.extend(top_magnitudes)
        
        return np.array(fft_features)
    
    def extract_statistical_features(self, window_data):
        """
        Extract statistical features from time-domain signal
        """
        stats = []
        
        for channel in range(window_data.shape[1]):
            signal_data = window_data[:, channel]
            
            # Time domain statistics
            stats.append(np.mean(signal_data))
            stats.append(np.std(signal_data))
            stats.append(np.max(signal_data) - np.min(signal_data))  # Peak-to-peak
            stats.append(np.sqrt(np.mean(signal_data**2)))  # RMS
            
            # Higher order statistics
            stats.append(np.mean(np.abs(np.diff(signal_data))))  # Mean absolute derivative
        
        return np.array(stats)

    def window_data(self, df, label_prefix, is_healthy_file, fault_time=None):
        """
        Slice data into windows and assign to Train/Val/Test based on time.
        Train: 0-70%
        Val: 70-85%
        Test: 85-100%
        """
        splits = {
            'train': {'windows': [], 'labels': []},
            'val': {'windows': [], 'labels': []},
            'test': {'windows': [], 'labels': []}
        }
        
        features = ['Ia', 'Ib', 'Ic', 'va', 'vb', 'vc']
        
        # Ensure only numeric data
        data = df[features].values
        times = df['time'].values
        total_duration = times[-1] - times[0]
        start_time_offset = times[0]
        
        # Split thresholds (relative to file start)
        train_end = start_time_offset + 0.70 * total_duration
        val_end = start_time_offset + 0.85 * total_duration
        
        num_windows = (len(df) - WINDOW_LEN) // STRIDE + 1
        
        # Default fault time if not provided
        if fault_time is None:
            fault_time = FAULT_TIME

        for i in range(num_windows):
            start_idx = i * STRIDE
            end_idx = start_idx + WINDOW_LEN
            
            window_data = data[start_idx:end_idx]
            window_times = times[start_idx:end_idx]
            
            # Determine Window Time (use midpoint or start)
            w_start = window_times[0]
            w_end = window_times[-1]
            w_mid = (w_start + w_end) / 2
            
            # Determine Split
            if w_end < train_end:
                split_key = 'train'
            elif w_start >= train_end and w_end < val_end:
                split_key = 'val'
            elif w_start >= val_end:
                split_key = 'test'
            else:
                # Window crosses split boundary -> Discard to ensure separation
                continue

            # Determine Label
            # Logic: Strictly apply fault time logic
            # Healthy File: Always Healthy
            # Faulty File:
            #   - Before fault_time: Healthy
            #   - After fault_time: Faulty
            #   - Mixed: Discard
            
            current_label = None
            if is_healthy_file:
                current_label = "Healthy"
            else:
                if w_end < fault_time:
                    current_label = "Healthy"
                elif w_start >= fault_time:
                    current_label = f"Faulty_{label_prefix}"
                else:
                    # Mixed healthy/faulty window -> Discard
                    continue
            
            if current_label:
                splits[split_key]['windows'].append(window_data)
                splits[split_key]['labels'].append(current_label)
                    
        return splits

def get_fault_time_from_filename(filename):
    """
    Parse fault time from filename.
    Expected format examples:
    - BLDC_ControlSwitch_0.46Nm_1s.csv -> 1.0
    - BLDC_ControlSwitch_0.42Nm_0.9s.csv -> 0.9
    - BLDC_base... (no time) -> None
    """
    import re
    # Look for pattern like _0.9s.csv or _1s.csv or _1.1s.csv at the end
    match = re.search(r'_(\d+\.?\d*)s\.csv$', filename)
    if match:
        return float(match.group(1))
    return None

def get_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(base_dir, 'files')
    
    file_list = []

    # 1. BLDC_basedata (Original base files)
    # BLDC_Healthy.csv -> Healthy (0.2s irrelevant)
    # BLDC_OpenCircuit.csv -> Open_Circuit (0.2s default)
    base_map = {
        'BLDC_Healthy.csv': ('Healthy', True),
        'BLDC_OpenCircuit.csv': ('Open_Circuit', False),
        'BLDC_Short.csv': ('Short_Circuit', False),
        'BLDC_ControlSwitch.csv': ('Control_Switch', False)
    }
    
    basedata_dir = os.path.join(files_dir, 'BLDC_basedata')
    if os.path.exists(basedata_dir):
        for fname in os.listdir(basedata_dir):
            if fname in base_map:
                label, is_healthy = base_map[fname]
                file_list.append({
                    'path': os.path.join(basedata_dir, fname),
                    'label_prefix': label,
                    'is_healthy': is_healthy,
                    'fault_time': 0.2 # Default for base files
                })

    # 2. Healthy Directory
    healthy_dir = os.path.join(files_dir, 'Healthy')
    if os.path.exists(healthy_dir):
        for fname in glob.glob(os.path.join(healthy_dir, "*.csv")):
            file_list.append({
                'path': fname,
                'label_prefix': 'Healthy',
                'is_healthy': True,
                'fault_time': None
            })

    # 3. Advanced Directory
    # Advanced/ControlSwitch -> Control_Switch
    # Advanced/InterTurn -> Short_Circuit (Mapped as discussed)
    # Advanced/OpenCircuit -> Open_Circuit
    advanced_dir = os.path.join(files_dir, 'Advanced')
    if os.path.exists(advanced_dir):
        # We can map subdirectory names to labels
        adv_map = {
            'ControlSwitch': 'Control_Switch',
            'InterTurn': 'Short_Circuit',
            'OpenCircuit': 'Open_Circuit'
        }
        
        for subdir, label in adv_map.items():
            path = os.path.join(advanced_dir, subdir)
            if os.path.exists(path):
                for fname in glob.glob(os.path.join(path, "*.csv")):
                    ft = get_fault_time_from_filename(os.path.basename(fname))
                    if ft is None:
                        # Fallback or skip? Assuming purely faulty files here.
                        # If no time is in name, maybe assumed 0? Or maybe skip.
                        # Let's assume 0.2s default if parsing fails but warn?
                        # Actually most have it.
                        ft = 0.2 
                    
                    file_list.append({
                        'path': fname,
                        'label_prefix': label,
                        'is_healthy': False, # All advanced are faulty conditions
                        'fault_time': ft
                    })

    return file_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmentations', type=int, default=10, help='Number of augmented copies per file')
    parser.add_argument('--output', type=str, default='artifacts/dataset.npz')
    parser.add_argument('--use_fft', action='store_true', help='Add FFT features (experimental)')
    parser.add_argument('--spike_prob', type=float, default=0.3, help='Probability of adding spike augmentation')
    parser.add_argument('--harmonic_prob', type=float, default=0.3, help='Probability of adding harmonic distortion')
    parser.add_argument('--warp_prob', type=float, default=0.2, help='Probability of applying time warping')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    augmentor = DataAugmentor()
    processor = DataProcessor(augmentor, use_fft_features=args.use_fft)
    
    all_splits = {
        'train': {'windows': [], 'labels': []},
        'val': {'windows': [], 'labels': []},
        'test': {'windows': [], 'labels': []}
    }
    
    file_list = get_files()
    print(f"Found {len(file_list)} files to process.")
    print(f"Augmentations per file: {args.augmentations}")
    print(f"Advanced augmentation probabilities:")
    print(f"  - Spikes: {args.spike_prob:.1%}")
    print(f"  - Harmonics: {args.harmonic_prob:.1%}")
    print(f"  - Time warping: {args.warp_prob:.1%}\n")
    
    for f in file_list:
        fname = os.path.basename(f['path'])
        try:
            df_base = pd.read_csv(f['path'])
        except Exception as e:
            print(f"Error reading {f['path']}: {e}")
            continue

        for i in range(args.augmentations):
            # Apply core augmentations (always)
            df_aug = augmentor.apply_gain_variation(df_base)
            df_aug = augmentor.add_offset_drift(df_aug)
            df_aug = augmentor.add_gaussian_noise(df_aug)
            
            # Apply advanced augmentations (probabilistically)
            if np.random.random() < args.spike_prob:
                df_aug = augmentor.add_random_spikes(df_aug)
            
            if np.random.random() < args.harmonic_prob:
                df_aug = augmentor.add_harmonic_distortion(df_aug)
            
            if np.random.random() < args.warp_prob:
                df_aug = augmentor.apply_time_warping(df_aug)
            
            # Derive Features
            df_proc = processor.derive_phase_voltages(df_aug)
            
            # Windowing and Splitting
            file_splits = processor.window_data(
                df_proc, 
                f['label_prefix'], 
                f['is_healthy'],
                fault_time=f['fault_time']
            )
            
            # Accumulate
            for split in ['train', 'val', 'test']:
                if len(file_splits[split]['windows']) > 0:
                    all_splits[split]['windows'].append(file_splits[split]['windows'])
                    all_splits[split]['labels'].append(file_splits[split]['labels'])
                
    # Aggregate and Save
    if any(len(all_splits[s]['windows']) > 0 for s in ['train', 'val', 'test']):
        
        final_data = {}
        print(f"\nProcessing and saving splits to {args.output}")
        
        for split in ['train', 'val', 'test']:
            if all_splits[split]['windows']:
                # Concatenate nested lists: List[List[Windows]] -> Array[All Windows]
                # Each element in 'windows' is a list of windows for one file/aug
                
                # First flatten the list of lists
                flat_windows = [w for sublist in all_splits[split]['windows'] for w in sublist]
                flat_labels = [l for sublist in all_splits[split]['labels'] for l in sublist]
                
                X = np.array(flat_windows)
                y = np.array(flat_labels)
                
                final_data[f'X_{split}'] = X
                final_data[f'y_{split}'] = y
                
                print(f"  {split.capitalize()}: {X.shape[0]} samples")
            else:
                final_data[f'X_{split}'] = np.array([])
                final_data[f'y_{split}'] = np.array([])
                print(f"  {split.capitalize()}: 0 samples")
        
        # Save compressed dictionary
        np.savez_compressed(args.output, **final_data)
        
        # Print Stats for Train
        if 'y_train' in final_data and len(final_data['y_train']) > 0:
            unique, counts = np.unique(final_data['y_train'], return_counts=True)
            print("\nTrain Class Distribution:")
            for label, count in zip(unique, counts):
                print(f"  {label}: {count:,} ({100*count/len(final_data['y_train']):.1f}%)")
    else:
        print("No windows generated.")

if __name__ == "__main__":
    main()
