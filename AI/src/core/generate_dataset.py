import os
import glob
import pandas as pd
import numpy as np
import random
import argparse
from scipy import signal
from scipy.interpolate import interp1d

# Constants
VDC_BASE = 24.0
FAULT_TIME = 0.2
WINDOW_SIZE_MS = 40
OVERLAP_PCT = 0.50

# Split Ratios
SPLIT_TRAIN = 0.6
SPLIT_VAL = 0.2
# SPLIT_TEST = 0.2 (Implicit)

class DataAugmentor:
    def __init__(self):
        pass

    def _get_scale_stats(self, series):
        """
        Compute robust scale stats.
        For file-based splitting, we can use the whole file stats since 
        if it's a test file, it's never seen in training anyway.
        """
        mean_abs = series.abs().mean()
        rms = np.sqrt(np.mean(series**2))
        return mean_abs, rms

    def add_gaussian_noise(self, df):
        """
        Add bounded uniform noise.
        """
        df_aug = df.copy()
        
        # Currents
        for col in ['Ia', 'Ib', 'Ic']:
            noise_level = np.random.uniform(0.03, 0.05) 
            noise = df[col] * np.random.uniform(-noise_level, noise_level, size=len(df))
            df_aug[col] = df[col] + noise
            
        # Voltage
        col = 'Vdc'
        noise_level = np.random.uniform(0.005, 0.01)
        noise = df[col] * np.random.uniform(-noise_level, noise_level, size=len(df))
        df_aug[col] = df[col] + noise
        
        return df_aug

    def apply_gain_variation(self, df):
        """
        Gain variation constant per file.
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
        """
        df_aug = df.copy()
        n_samples = len(df)
        t = np.linspace(0, 1, n_samples) 
        
        # Currents
        for col in ['Ia', 'Ib', 'Ic']:
            max_drift_pct = np.random.uniform(0.01, 0.03)
            if np.random.random() < 0.5: max_drift_pct = -max_drift_pct
            
            _, scale_rms = self._get_scale_stats(df[col])
            
            drift = scale_rms * max_drift_pct * t
            df_aug[col] = df[col] + drift

        # Voltage
        col = 'Vdc'
        max_drift_pct = np.random.uniform(0.001, 0.002)
        if np.random.random() < 0.5: max_drift_pct = -max_drift_pct
        
        _, scale_rms = self._get_scale_stats(df[col])
        drift = scale_rms * max_drift_pct * t
        df_aug[col] = df[col] + drift
        
        return df_aug
    
    def add_random_spikes(self, df, spike_prob=0.005):
        """
        Add random spikes.
        """
        df_aug = df.copy()
        
        for col in ['Ia', 'Ib', 'Ic']:
            spike_mask = np.random.random(len(df)) < spike_prob
            spike_magnitude = np.random.uniform(0.05, 0.15, size=len(df))
            spike_sign = np.random.choice([-1, 1], size=len(df))
            
            scale_mean, _ = self._get_scale_stats(df[col])
            
            df_aug.loc[spike_mask, col] += spike_sign[spike_mask] * spike_magnitude[spike_mask] * scale_mean
        
        return df_aug
    
    def add_harmonic_distortion(self, df):
        """
        Add harmonics.
        """
        df_aug = df.copy()
        n_samples = len(df)
        t_indices = np.arange(n_samples) 
        
        cycle_len = 100 
        fundamental_freq = 2 * np.pi / cycle_len

        for col in ['Ia', 'Ib', 'Ic']:
            harmonic3 = np.random.uniform(0.01, 0.03) * np.sin(3 * fundamental_freq * t_indices + np.random.uniform(0, 2*np.pi))
            harmonic5 = np.random.uniform(0.005, 0.015) * np.sin(5 * fundamental_freq * t_indices + np.random.uniform(0, 2*np.pi))
            
            scale_mean, _ = self._get_scale_stats(df[col])
            df_aug[col] = df[col] + scale_mean * (harmonic3 + harmonic5)
        
        return df_aug
    
    def apply_time_warping(self, df, warp_factor=0.05):
        """
        Apply REAL time warping via interpolation.
        """
        df_aug = df.copy()
        times = df['time'].values
        n_samples = len(times)
        
        # 1. Create a non-linear time map
        t_normalized = np.linspace(0, 1, n_samples)
        
        warp_curve = np.sin(2 * np.pi * t_normalized * np.random.uniform(0.5, 2)) 
        
        total_duration = times[-1] - times[0]
        max_shift = warp_factor * total_duration
        time_shifts = warp_curve * max_shift
        
        warped_times = times + time_shifts
        
        if not np.all(np.diff(warped_times) > 0):
             warped_times = np.sort(warped_times)

        # 2. Resample
        cols_to_warp = ['Ia', 'Ib', 'Ic', 'Vdc', 'da', 'db', 'dc']
        for col in cols_to_warp:
             if col in df_aug.columns:
                 f = interp1d(warped_times, df[col].values, kind='linear', fill_value="extrapolate")
                 df_aug[col] = f(times)
                 
        return df_aug

class DataProcessor:
    def __init__(self, augmentor):
        self.augmentor = augmentor

    def derive_phase_voltages(self, df):
        """
        Derive va, vb, vc from Vdc and duty cycles.
        """
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
    
    def window_data(self, df, label_prefix, is_healthy_file, fault_time=None):
        """
        Slice data into windows.
        NO splitting inside here. Returns ALL valid windows.
        """
        windows = []
        labels = []
        
        features = ['Ia', 'Ib', 'Ic', 'va', 'vb', 'vc']
        
        # Ensure only numeric data
        data = df[features].values
        times = df['time'].values
        
        # Dynamic Sample Rate Calculation
        dt = np.median(np.diff(times))
        sample_rate_hz = 1.0 / dt
        
        # Calculate Window Length in Samples
        window_len = int((WINDOW_SIZE_MS / 1000) * sample_rate_hz)
        stride = int(window_len * (1 - OVERLAP_PCT))
        
        if window_len < 2:
            return windows, labels
        
        start_time_offset = times[0]
        
        num_windows = (len(df) - window_len) // stride + 1
        
        if fault_time is None:
            fault_time = FAULT_TIME

        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_len
            
            window_data = data[start_idx:end_idx]
            window_times = times[start_idx:end_idx]
            
            w_start = window_times[0]
            w_end = window_times[-1]
            
            # Determine Label
            current_label = None
            if is_healthy_file:
                current_label = "Healthy"
            else:
                if (w_end - start_time_offset) < fault_time:
                    current_label = "Healthy"
                elif (w_start - start_time_offset) >= fault_time:
                    current_label = f"Faulty_{label_prefix}"
                else:
                    # Mixed healthy/faulty
                    continue
            
            if current_label:
                windows.append(window_data)
                labels.append(current_label)
                    
        return windows, labels

def get_fault_time_from_filename(filename):
    import re
    match = re.search(r'_(\d+\.?\d*)s\.csv$', filename)
    if match:
        return float(match.group(1))
    return None

def get_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from src/core up to AI directory
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    files_dir = os.path.join(base_dir, 'files')
    
    # Structure: {'ClassPrefix': [ {'path':..., 'label_prefix':..., ...} ] }
    grouped_files = {}

    def add_file(group_key, f_info):
        if group_key not in grouped_files:
            grouped_files[group_key] = []
        grouped_files[group_key].append(f_info)

    # 1. Healthy
    healthy_dir = os.path.join(files_dir, 'Healthy')
    if os.path.exists(healthy_dir):
        for fname in glob.glob(os.path.join(healthy_dir, "*.csv")):
            add_file('Healthy', {
                'path': fname,
                'label_prefix': 'Healthy',
                'is_healthy': True,
                'fault_time': None
            })

    # 2. Faulty
    faulty_dir = os.path.join(files_dir, 'Faulty')
    if os.path.exists(faulty_dir):
        fault_map = {
            'ControlSwitch': ('Control_Switch', 'Faulty_Control_Switch'),
            'InterTurn': ('Inter_Turn', 'Faulty_Inter_Turn'), 
            'OpenCircuit': ('Open_Circuit', 'Faulty_Open_Circuit')
        }
        
        for subdir, (label, group) in fault_map.items():
            path = os.path.join(faulty_dir, subdir)
            if os.path.exists(path):
                for fname in glob.glob(os.path.join(path, "*.csv")):
                    ft = get_fault_time_from_filename(os.path.basename(fname))
                    if ft is None: ft = 0.2
                    
                    add_file(group, {
                        'path': fname,
                        'label_prefix': label,
                        'is_healthy': False,
                        'fault_time': ft
                    })

    return grouped_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_augmentations', type=int, default=10, help='Number of augmented copies per TRAIN file')
    parser.add_argument('--val_augmentations', type=int, default=2, help='Number of augmented copies per VAL file (recommended: 0-2)')
    parser.add_argument('--test_augmentations', type=int, default=0, help='Number of augmented copies per TEST file (recommended: 0)')
    parser.add_argument('--output', type=str, default='../../artifacts/dataset.npz')
    parser.add_argument('--spike_prob', type=float, default=0.3)
    parser.add_argument('--harmonic_prob', type=float, default=0.3)
    parser.add_argument('--warp_prob', type=float, default=0.2)
    parser.add_argument('--allow_single_file_split', action='store_true', help='Allow time-based splitting for single-file groups (may cause leakage)')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    augmentor = DataAugmentor()
    processor = DataProcessor(augmentor)
    
    all_splits = {
        'train': {'windows': [], 'labels': []},
        'val': {'windows': [], 'labels': []},
        'test': {'windows': [], 'labels': []}
    }
    
    grouped_files = get_files()
    total_files = sum(len(v) for v in grouped_files.values())
    print(f"Found {total_files} files across {len(grouped_files)} groups.")
    
    required_cols = {'Ia', 'Ib', 'Ic', 'Vdc', 'da', 'db', 'dc', 'time'}
    
    # Process each group separately to ensure balanced splits
    for group, file_list in grouped_files.items():
        # Shuffle files for random assignment
        random.shuffle(file_list)
        n_files = len(file_list)
        
        if n_files == 1:
            # SPECIAL CASE: Single file - HIGH RISK OF LEAKAGE
            if not args.allow_single_file_split:
                print(f"WARNING: Group {group}: Only 1 file found. Skipping to prevent leakage.")
                print(f"   Use --allow_single_file_split to force inclusion (not recommended).")
                continue
            else:
                print(f"WARNING: Group {group}: Single file - using time-based split.")
                print(f"   This may cause data leakage due to temporal correlation!")
                
                f = file_list[0]
                try:
                    df_base = pd.read_csv(f['path'])
                except Exception as e:
                    print(f"  Error reading {f['path']}: {e}")
                    continue

                if not required_cols.issubset(df_base.columns):
                    continue
                
                # Process ORIGINAL data only, split windows, then augment each split separately
                df_proc = processor.derive_phase_voltages(df_base)
                w_full, l_full = processor.window_data(df_proc, f['label_prefix'], f['is_healthy'], fault_time=f['fault_time'])
                
                if not w_full:
                    continue
                
                # Split windows 60/20/20 (train/val/test) with gap to reduce correlation
                n_w = len(w_full)
                train_end = int(n_w * 0.6)
                val_end = int(n_w * 0.8)
                
                # Create a small gap (skip 5% of windows) to reduce correlation
                gap_size = max(1, int(n_w * 0.05))
                
                train_windows = w_full[:max(0, train_end - gap_size)]
                train_labels = l_full[:max(0, train_end - gap_size)]
                
                val_windows = w_full[train_end + gap_size:max(train_end + gap_size, val_end - gap_size)]
                val_labels = l_full[train_end + gap_size:max(train_end + gap_size, val_end - gap_size)]
                
                test_windows = w_full[val_end + gap_size:]
                test_labels = l_full[val_end + gap_size:]
                
                # Now augment each split according to its augmentation count
                # Original (i=0) plus augmentations
                for i in range(args.train_augmentations + 1):
                    if i == 0:
                        # Original data
                        all_splits['train']['windows'].append(train_windows)
                        all_splits['train']['labels'].append(train_labels)
                    else:
                        # Augment
                        df_aug = augmentor.apply_gain_variation(df_base)
                        df_aug = augmentor.add_offset_drift(df_aug)
                        df_aug = augmentor.add_gaussian_noise(df_aug)
                        if np.random.random() < args.spike_prob: df_aug = augmentor.add_random_spikes(df_aug)
                        if np.random.random() < args.harmonic_prob: df_aug = augmentor.add_harmonic_distortion(df_aug)
                        if np.random.random() < args.warp_prob: df_aug = augmentor.apply_time_warping(df_aug)
                        
                        df_aug_proc = processor.derive_phase_voltages(df_aug)
                        w_aug, l_aug = processor.window_data(df_aug_proc, f['label_prefix'], f['is_healthy'], fault_time=f['fault_time'])
                        
                        if w_aug:
                            w_aug_train = w_aug[:max(0, train_end - gap_size)]
                            l_aug_train = l_aug[:max(0, train_end - gap_size)]
                            all_splits['train']['windows'].append(w_aug_train)
                            all_splits['train']['labels'].append(l_aug_train)
                
                # Val: minimal augmentation
                for i in range(args.val_augmentations + 1):
                    if i == 0:
                        all_splits['val']['windows'].append(val_windows)
                        all_splits['val']['labels'].append(val_labels)
                    else:
                        df_aug = augmentor.add_gaussian_noise(df_base)  # Light augmentation only
                        df_aug_proc = processor.derive_phase_voltages(df_aug)
                        w_aug, l_aug = processor.window_data(df_aug_proc, f['label_prefix'], f['is_healthy'], fault_time=f['fault_time'])
                        if w_aug:
                            w_aug_val = w_aug[train_end + gap_size:max(train_end + gap_size, val_end - gap_size)]
                            l_aug_val = l_aug[train_end + gap_size:max(train_end + gap_size, val_end - gap_size)]
                            all_splits['val']['windows'].append(w_aug_val)
                            all_splits['val']['labels'].append(l_aug_val)
                
                # Test: NO augmentation by default (only original)
                for i in range(args.test_augmentations + 1):
                    if i == 0:
                        all_splits['test']['windows'].append(test_windows)
                        all_splits['test']['labels'].append(test_labels)
                
                continue

        # Normal Multi-File Logic
        n_train = int(n_files * SPLIT_TRAIN)
        if n_train == 0: n_train = 1 # Force at least 1 train if we have 2 files (1 train, 1 test)
        n_val = int(n_files * SPLIT_VAL)
        
        train_files = file_list[:n_train]
        val_files = file_list[n_train:n_train+n_val]
        test_files = file_list[n_train+n_val:]
        
        split_assignments = [
            ('train', train_files, args.train_augmentations),
            ('val', val_files, args.val_augmentations),
            ('test', test_files, args.test_augmentations)
        ]
        
        print(f"Group {group}: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")

        for split_name, files, n_augmentations in split_assignments:
            for f in files:
                try:
                    df_base = pd.read_csv(f['path'])
                except Exception as e:
                    print(f"  Error reading {f['path']}: {e}")
                    continue

                if not required_cols.issubset(df_base.columns):
                    continue

                # Process original data first (always include)
                df_proc = processor.derive_phase_voltages(df_base)
                w, l = processor.window_data(
                    df_proc, 
                    f['label_prefix'], 
                    f['is_healthy'],
                    fault_time=f['fault_time']
                )
                
                if w:
                    all_splits[split_name]['windows'].append(w)
                    all_splits[split_name]['labels'].append(l)
                
                # Apply augmentations based on split type
                for i in range(n_augmentations):
                    df_aug = df_base.copy()
                    
                    # Different augmentation strategies per split
                    if split_name == 'train':
                        # Full augmentation pipeline for training
                        df_aug = augmentor.apply_gain_variation(df_aug)
                        df_aug = augmentor.add_offset_drift(df_aug)
                        df_aug = augmentor.add_gaussian_noise(df_aug)
                        
                        if np.random.random() < args.spike_prob:
                            df_aug = augmentor.add_random_spikes(df_aug)
                        if np.random.random() < args.harmonic_prob:
                            df_aug = augmentor.add_harmonic_distortion(df_aug)
                        if np.random.random() < args.warp_prob:
                            df_aug = augmentor.apply_time_warping(df_aug)
                    
                    elif split_name == 'val':
                        # Light augmentation for validation (noise only)
                        df_aug = augmentor.add_gaussian_noise(df_aug)
                    
                    # Test: No augmentation by default (n_augmentations should be 0)
                    # If user explicitly sets test_augmentations > 0, apply minimal noise
                    elif split_name == 'test' and n_augmentations > 0:
                        df_aug = augmentor.add_gaussian_noise(df_aug)
                    
                    df_proc = processor.derive_phase_voltages(df_aug)
                    
                    w, l = processor.window_data(
                        df_proc, 
                        f['label_prefix'], 
                        f['is_healthy'],
                        fault_time=f['fault_time']
                    )
                    
                    if w:
                        all_splits[split_name]['windows'].append(w)
                        all_splits[split_name]['labels'].append(l)

    # Aggregate and Save
    if any(len(all_splits[s]['windows']) > 0 for s in ['train', 'val', 'test']):
        final_data = {}
        print(f"\nProcessing and saving splits to {args.output}")
        
        for split in ['train', 'val', 'test']:
            if all_splits[split]['windows']:
                # Flatten
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
        
        np.savez_compressed(args.output, **final_data)
        
        # Print statistics for all splits
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            y_key = f'y_{split}'
            if y_key in final_data and len(final_data[y_key]) > 0:
                unique, counts = np.unique(final_data[y_key], return_counts=True)
                print(f"\n{split.upper()} Class Distribution:")
                for label, count in zip(unique, counts):
                    print(f"  {label}: {count:,} ({100*count/len(final_data[y_key]):.1f}%)")
        
        # Data leakage check
        print("\n" + "="*60)
        print("DATA LEAKAGE CHECK")
        print("="*60)
        print("Checking for duplicate samples across splits...")
        
        # Quick check: compare a sample of test data against all train data
        if len(final_data.get('X_train', [])) > 0 and len(final_data.get('X_test', [])) > 0:
            X_train_flat = final_data['X_train'].reshape(len(final_data['X_train']), -1)
            X_test_flat = final_data['X_test'].reshape(len(final_data['X_test']), -1)
            
            # Check first 100 test samples
            n_check = min(100, len(X_test_flat))
            duplicates_found = 0
            
            for i in range(n_check):
                # Check if this test sample exists in train (exact match)
                if np.any(np.all(X_train_flat == X_test_flat[i], axis=1)):
                    duplicates_found += 1
            
            if duplicates_found == 0:
                print(f"PASS: No exact duplicates found (checked {n_check}/{len(X_test_flat)} test samples)")
            else:
                print(f"WARNING: Found {duplicates_found} exact duplicates in first {n_check} test samples!")
        
        print("\nDataset saved successfully to:", args.output)
    else:
        print("No windows generated.")

if __name__ == "__main__":
    main()
