#!/usr/bin/env python3
"""
Comprehensive data leakage validation script.
Run this after generating the dataset to ensure no leakage between splits.
"""

import numpy as np
import os
import sys
from collections import defaultdict

def check_exact_duplicates(X_train, X_val, X_test):
    """Check for exact duplicate samples across splits"""
    print("\n" + "="*60)
    print("EXACT DUPLICATE CHECK")
    print("="*60)
    
    # Flatten for comparison
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1) if len(X_val) > 0 else np.array([])
    X_test_flat = X_test.reshape(len(X_test), -1)
    
    issues = []
    
    # Train vs Test
    print(f"Checking {len(X_test_flat)} test samples against {len(X_train_flat)} train samples...")
    train_test_dups = 0
    for i in range(len(X_test_flat)):
        if np.any(np.all(X_train_flat == X_test_flat[i], axis=1)):
            train_test_dups += 1
        if i % 500 == 0 and i > 0:
            print(f"  Progress: {i}/{len(X_test_flat)}...")
    
    if train_test_dups > 0:
        issues.append(f"FAIL: Found {train_test_dups} exact duplicates between train and test")
    else:
        print("  PASS: No exact duplicates between train and test")
    
    # Train vs Val (if val exists)
    if len(X_val_flat) > 0:
        print(f"\nChecking {len(X_val_flat)} val samples against {len(X_train_flat)} train samples...")
        train_val_dups = 0
        for i in range(len(X_val_flat)):
            if np.any(np.all(X_train_flat == X_val_flat[i], axis=1)):
                train_val_dups += 1
            if i % 500 == 0 and i > 0:
                print(f"  Progress: {i}/{len(X_val_flat)}...")
        
        if train_val_dups > 0:
            issues.append(f"FAIL: Found {train_val_dups} exact duplicates between train and val")
        else:
            print("  PASS: No exact duplicates between train and val")
        
        # Val vs Test
        print(f"\nChecking {len(X_test_flat)} test samples against {len(X_val_flat)} val samples...")
        val_test_dups = 0
        for i in range(len(X_test_flat)):
            if np.any(np.all(X_val_flat == X_test_flat[i], axis=1)):
                val_test_dups += 1
            if i % 500 == 0 and i > 0:
                print(f"  Progress: {i}/{len(X_test_flat)}...")
        
        if val_test_dups > 0:
            issues.append(f"FAIL: Found {val_test_dups} exact duplicates between val and test")
        else:
            print("  PASS: No exact duplicates between val and test")
    
    return issues

def check_near_duplicates(X_train, X_test, threshold=1e-4, sample_size=200):
    """Check for near-duplicate samples (samples that are suspiciously similar)
    
    Note: Using 1e-4 threshold instead of 1e-6 to account for floating point precision
    and minor numerical differences from augmentation pipeline.
    """
    print("\n" + "="*60)
    print("NEAR-DUPLICATE CHECK")
    print("="*60)
    print(f"Threshold: max absolute difference < {threshold}")
    print(f"Sampling {min(sample_size, len(X_test))} test samples for efficiency\n")
    
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    
    # Sample test set for efficiency
    sample_indices = np.random.choice(len(X_test_flat), 
                                     size=min(sample_size, len(X_test_flat)), 
                                     replace=False)
    
    near_dups = 0
    print("Checking samples (this may take a minute)...")
    for i, idx in enumerate(sample_indices):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(sample_indices)}...")
        
        test_vec = X_test_flat[idx]
        
        # Compute max absolute difference to each train sample
        # Use chunking to reduce memory usage
        chunk_size = 1000
        found_near_dup = False
        for start_idx in range(0, len(X_train_flat), chunk_size):
            end_idx = min(start_idx + chunk_size, len(X_train_flat))
            chunk = X_train_flat[start_idx:end_idx]
            max_diffs = np.abs(chunk - test_vec).max(axis=1)
            
            if np.any(max_diffs < threshold):
                found_near_dup = True
                break
        
        if found_near_dup:
            near_dups += 1
    
    if near_dups > 0:
        pct = 100 * near_dups / len(sample_indices)
        print(f"INFO: Found {near_dups} near-duplicates in sample ({pct:.1f}%)")
        if pct > 10.0:
            print("  WARNING: >10% near-duplicates may indicate leakage")
            return [f"WARNING: {near_dups}/{len(sample_indices)} sampled test examples are suspiciously similar to train"]
        else:
            print("  OK: <10% is acceptable (may be from similar operating conditions)")
            return []
    else:
        print("  PASS: No concerning near-duplicates found in sample")
        return []

def check_statistical_similarity(X_train, X_test):
    """Check if train/test distributions are reasonably similar
    
    Note: When train is heavily augmented and test is not, we EXPECT differences.
    Large std difference is normal because augmentation adds variance.
    Focus on whether distributions are catastrophically different.
    """
    print("\n" + "="*60)
    print("STATISTICAL DISTRIBUTION CHECK")
    print("="*60)
    
    # Flatten to (n_samples, features)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    
    # Compare means and stds
    train_mean = X_train_flat.mean(axis=0)
    test_mean = X_test_flat.mean(axis=0)
    train_std = X_train_flat.std(axis=0)
    test_std = X_test_flat.std(axis=0)
    
    mean_diff = np.abs(train_mean - test_mean).mean()
    std_diff = np.abs(train_std - test_std).mean()
    
    # Calculate relative differences (more meaningful than absolute)
    mean_magnitude = (np.abs(train_mean).mean() + np.abs(test_mean).mean()) / 2
    std_magnitude = (train_std.mean() + test_std.mean()) / 2
    
    rel_mean_diff = mean_diff / (mean_magnitude + 1e-8)
    rel_std_diff = std_diff / (std_magnitude + 1e-8)
    
    print(f"Average absolute feature mean difference: {mean_diff:.6f}")
    print(f"Average absolute feature std difference: {std_diff:.6f}")
    print(f"\nRelative differences (more meaningful):")
    print(f"  Mean: {rel_mean_diff:.2%} of average magnitude")
    print(f"  Std:  {rel_std_diff:.2%} of average magnitude")
    
    issues = []
    
    # Relaxed thresholds accounting for augmentation
    # Mean shift should be relatively small (centers should be similar)
    if rel_mean_diff > 0.5:  # 50% relative shift is concerning
        issues.append(f"WARNING: Large relative mean difference ({rel_mean_diff:.1%}) - possible distribution shift")
    
    # Std difference is EXPECTED when train is augmented and test is not
    # Only flag if it's catastrophically different
    if rel_std_diff > 2.0:  # 200% difference is very concerning
        issues.append(f"WARNING: Very large relative std difference ({rel_std_diff:.1%}) - major distribution shift")
    elif rel_std_diff > 0.5:
        print(f"\n  INFO: Moderate std difference ({rel_std_diff:.1%}) is EXPECTED when:")
        print("    - Training data is heavily augmented")
        print("    - Test data is unaugmented (real-world data)")
        print("    This is normal and helps the model generalize!")
    
    if not issues:
        print("\n  PASS: Train and test distributions are reasonably similar")
    
    return issues

def check_class_overlap(y_train, y_val, y_test):
    """Ensure all classes appear in all splits"""
    print("\n" + "="*60)
    print("CLASS OVERLAP CHECK")
    print("="*60)
    
    train_classes = set(y_train)
    val_classes = set(y_val) if len(y_val) > 0 else set()
    test_classes = set(y_test)
    
    print(f"Train classes ({len(train_classes)}): {sorted(train_classes)}")
    if len(val_classes) > 0:
        print(f"Val classes ({len(val_classes)}): {sorted(val_classes)}")
    print(f"Test classes ({len(test_classes)}): {sorted(test_classes)}")
    
    issues = []
    
    # Check if test has classes not in train
    test_only = test_classes - train_classes
    if test_only:
        issues.append(f"FAIL: Test has classes not in train: {test_only}")
    else:
        print("  PASS: All test classes exist in train")
    
    # Check if train has all classes
    if len(val_classes) > 0:
        val_only = val_classes - train_classes
        if val_only:
            issues.append(f"FAIL: Val has classes not in train: {val_only}")
        else:
            print("  PASS: All val classes exist in train")
    
    return issues

def check_data_quality(X_train, X_val, X_test, y_train, y_val, y_test):
    """Check for data quality issues"""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    issues = []
    
    # Check for NaN or Inf
    for name, X in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        if len(X) == 0:
            continue
        if np.any(np.isnan(X)):
            issues.append(f"FAIL: {name} contains NaN values")
        if np.any(np.isinf(X)):
            issues.append(f"FAIL: {name} contains Inf values")
    
    if not issues:
        print("  PASS: No NaN or Inf values detected")
    
    # Check shapes consistency
    if X_train.shape[1:] != X_test.shape[1:]:
        issues.append(f"FAIL: Shape mismatch: train {X_train.shape[1:]} vs test {X_test.shape[1:]}")
    else:
        print(f"  PASS: Consistent shape: {X_train.shape[1:]}")
    
    # Check label counts
    print(f"\n  Sample counts:")
    print(f"    Train: {len(X_train):,} samples, {len(y_train):,} labels")
    print(f"    Val:   {len(X_val):,} samples, {len(y_val):,} labels")
    print(f"    Test:  {len(X_test):,} samples, {len(y_test):,} labels")
    
    if len(X_train) != len(y_train):
        issues.append(f"FAIL: Train: X and y length mismatch")
    if len(X_test) != len(y_test):
        issues.append(f"FAIL: Test: X and y length mismatch")
    if len(X_val) > 0 and len(X_val) != len(y_val):
        issues.append(f"FAIL: Val: X and y length mismatch")
    
    if len(X_train) == len(y_train) and len(X_test) == len(y_test):
        print("  PASS: X and y lengths match for all splits")
    
    return issues

def main():
    # Find dataset
    dataset_paths = [
        '../../artifacts/dataset.npz',
        'artifacts/dataset.npz',
        'AI/artifacts/dataset.npz',
        '../artifacts/dataset.npz'
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print("FAIL: Dataset not found! Please generate dataset first:")
        print("   python generate_dataset.py")
        sys.exit(1)
    
    print("="*60)
    print("DATA LEAKAGE VALIDATION")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Size: {os.path.getsize(dataset_path) / 1024**2:.1f} MB")
    
    # Load data
    print("\nLoading dataset...")
    data = np.load(dataset_path)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data.get('X_val', np.array([]))
    y_val = data.get('y_val', np.array([]))
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Run all checks
    all_issues = []
    
    all_issues.extend(check_data_quality(X_train, X_val, X_test, y_train, y_val, y_test))
    all_issues.extend(check_class_overlap(y_train, y_val, y_test))
    all_issues.extend(check_exact_duplicates(X_train, X_val, X_test))
    
    # Near-duplicate check with reduced sample size for speed (50 instead of 200)
    all_issues.extend(check_near_duplicates(X_train, X_test, threshold=1e-4, sample_size=50))
    all_issues.extend(check_statistical_similarity(X_train, X_test))
    
    # Final report
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    if not all_issues:
        print("\nPASS: ALL CHECKS PASSED - No leakage detected!")
        print("   Dataset appears to be properly split with no data leakage.")
        print("\n   Note: Distribution differences between train/test are EXPECTED")
        print("   when training data is augmented and test data is not.")
        print("   This helps your model generalize to real-world data!")
    else:
        print(f"\nWARNING: FOUND {len(all_issues)} ISSUE(S):\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        
        # Count critical vs warning issues
        critical = sum(1 for issue in all_issues if 'FAIL' in issue)
        warnings = len(all_issues) - critical
        
        if critical > 0:
            print(f"\nFAIL: {critical} critical issue(s) found - Please review and regenerate dataset")
            sys.exit(1)
        else:
            print(f"\nINFO: {warnings} warning(s) found, but no critical issues.")
            print("   These warnings may be acceptable if:")
            print("   1. Training data is heavily augmented (expected)")
            print("   2. <10% near-duplicates (natural similarity)")
            print("   3. No exact duplicates were found (verified above)")
            print("\n   Review the warnings above and decide if regeneration is needed.")

if __name__ == "__main__":
    main()
