#!/usr/bin/env python3
"""
Create sample dataset directly from data_filtered folder windows.
For each model and class, extract 10 random windows from the CSV files.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_FILTERED_DIR = "/home/kadir/sanal-makine/python/proje/fft_model/data_filtered"
OUTPUT_DIR = "/home/kadir/sanal-makine/python/proje/lstm_cnn_hybrid/sample_data"

MODELS = {
    "seq32": 32,
    "seq64": 64,
    "seq96": 96,
}

CLASSES = ["yukarı", "asagı", "araba"]


def create_sample_dataset():
    """Create sample dataset by extracting windows from data_filtered files"""
    
    # Create output directory structure
    for model_name in MODELS.keys():
        for class_name in CLASSES:
            class_dir = os.path.join(OUTPUT_DIR, model_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # Process each model
    for model_name, seq_len in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Processing {model_name.upper()} (sequence_length={seq_len})")
        print(f"{'='*60}")
        
        sample_count = {cls: 0 for cls in CLASSES}
        
        # For each class
        for class_name in CLASSES:
            class_dir = os.path.join(DATA_FILTERED_DIR, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠️  Class directory not found: {class_dir}")
                continue
            
            # Get all CSV files in this class directory
            csv_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.csv')])
            
            if not csv_files:
                print(f"⚠️  No CSV files found in {class_dir}")
                continue
            
            print(f"\n   📁 {class_name} ({len(csv_files)} files)")
            
            # Collect all possible windows from all files
            all_windows = []  # (file_path, start_row, end_row)
            
            for csv_file in csv_files:
                file_path = os.path.join(class_dir, csv_file)
                
                try:
                    # Load CSV to get row count
                    df = pd.read_csv(file_path)
                    num_rows = len(df)
                    
                    # For each possible window start position
                    # We want windows where start + seq_len <= num_rows
                    max_start = num_rows - seq_len + 1
                    
                    if max_start > 0:
                        # Create multiple possible windows (every stride positions)
                        stride = max(1, max_start // 20)  # Create ~20 candidate windows per file
                        
                        for start_idx in range(0, max_start, stride):
                            end_idx = start_idx + seq_len
                            all_windows.append((file_path, csv_file, start_idx, end_idx, num_rows))
                
                except Exception as e:
                    print(f"      ❌ Error reading {csv_file}: {e}")
                    continue
            
            if not all_windows:
                print(f"      ⚠️  No valid windows found for {class_name}")
                continue
            
            # Sample 10 windows randomly
            sampled_indices = np.random.choice(len(all_windows), min(10, len(all_windows)), replace=False)
            
            for sample_idx, window_idx in enumerate(sampled_indices, 1):
                file_path, csv_file, start_idx, end_idx, total_rows = all_windows[window_idx]
                
                try:
                    # Load the window
                    df = pd.read_csv(file_path)
                    window_data = df.iloc[start_idx:end_idx].copy()
                    
                    # Save the window
                    sample_name = f"{csv_file.replace('.csv', '')}_{sample_idx:02d}_r{start_idx}-{end_idx}.csv"
                    output_path = os.path.join(OUTPUT_DIR, model_name, class_name, sample_name)
                    
                    window_data.to_csv(output_path, index=False)
                    sample_count[class_name] += 1
                    
                    print(f"      ✓ {sample_name} (rows {start_idx}-{end_idx}/{total_rows})")
                    
                except Exception as e:
                    print(f"      ❌ Error saving sample: {e}")
        
        # Print summary
        print(f"\n✅ {model_name.upper()} Samples Created:")
        for class_name in CLASSES:
            count = sample_count[class_name]
            print(f"   {class_name}: {count} samples")


if __name__ == "__main__":
    print("🚀 Creating sample dataset from data_filtered windows...")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    create_sample_dataset()
    
    print(f"\n{'='*60}")
    print("✅ Sample dataset creation complete!")
    print(f"{'='*60}")
