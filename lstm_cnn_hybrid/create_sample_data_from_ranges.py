#!/usr/bin/env python3
"""
Extract sample data directly from data_filtered using high-confidence row ranges.
Creates samples for each model with clear start/end row positions.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re

# Configuration
DATA_FILTERED_DIR = "/home/kadir/sanal-makine/python/proje/fft_model/data_filtered"
OUTPUT_DIR = "/home/kadir/sanal-makine/python/proje/lstm_cnn_hybrid/sample_data"
MODEL_DIR = "/home/kadir/sanal-makine/python/proje/lstm_cnn_hybrid"

MODELS = {
    "seq32": {"seq_len": 32, "model_path": "seq32_best_model.pth"},
    "seq64": {"seq_len": 64, "model_path": "seq64_best_model.pth"},
    "seq96": {"seq_len": 96, "model_path": "seq96_best_model.pth"},
}

LABEL_MAP = {0: "yukarı", 1: "aşağı", 2: "araba"}
CONFIDENCE_THRESHOLD = 0.90

import torch
import sys
sys.path.insert(0, MODEL_DIR)

from realtime_predict import SimpleCNN_LSTM, PredictionEngine, load_scaler_and_labels


def parse_results_for_ranges(result_file):
    """Parse results file to extract row ranges with high confidence"""
    ranges_by_file = {}
    current_file = None
    
    with open(result_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Check for file header
        if line.startswith("##"):
            current_file = line.replace("##", "").strip()
            ranges_by_file[current_file] = []
        
        # Check for prediction lines with high confidence
        if current_file and ":" in line and "%" in line and "(" in line:
            try:
                match = re.match(r"\s*(\d+):\s+([\w]+)\s+\(([0-9.]+)%\)", line)
                if match:
                    row_idx = int(match.group(1))
                    class_name = match.group(2)
                    confidence = float(match.group(3)) / 100.0
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        ranges_by_file[current_file].append({
                            "row": row_idx,
                            "class": class_name,
                            "confidence": confidence
                        })
            except:
                pass
    
    return ranges_by_file


def find_csv_file(filename, base_dir):
    """Find CSV file with Turkish character handling"""
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower() == filename.lower():
                return os.path.join(root, file)
    return None


def create_sample_dataset():
    """Create sample dataset from high-confidence ranges"""
    
    # Create output directory structure
    for model_name in MODELS.keys():
        for class_name in LABEL_MAP.values():
            class_dir = os.path.join(OUTPUT_DIR, model_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # Process each model
    for model_name, model_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Processing {model_name.upper()}")
        print(f"{'='*60}")
        
        result_file = os.path.join(MODEL_DIR, f"son_sonuclar{'_' if model_name != 'seq32' else ''}{model_name.replace('seq', '') if model_name != 'seq32' else ''}.txt")
        
        # Handle file naming
        if model_name == "seq32":
            result_file = os.path.join(MODEL_DIR, "son_sonuclar.txt")
        elif model_name == "seq64":
            result_file = os.path.join(MODEL_DIR, "son_sonuclar_64.txt")
        elif model_name == "seq96":
            result_file = os.path.join(MODEL_DIR, "son_sonuclar_96.txt")
        
        if not os.path.exists(result_file):
            print(f"⚠️  Results file not found: {result_file}")
            continue
        
        # Parse results to get high-confidence ranges
        ranges_by_file = parse_results_for_ranges(result_file)
        print(f"📊 Found {len(ranges_by_file)} files with predictions")
        
        # Organize by class
        by_class = {}
        for class_name in LABEL_MAP.values():
            by_class[class_name] = []
        
        for csv_file, predictions in ranges_by_file.items():
            for pred in predictions:
                class_name = pred["class"]
                by_class[class_name].append({
                    "file": csv_file,
                    "row": pred["row"],
                    "confidence": pred["confidence"]
                })
        
        # Sample 10 per class
        sample_count = {cls: 0 for cls in LABEL_MAP.values()}
        
        for class_name in LABEL_MAP.values():
            candidates = by_class[class_name]
            if not candidates:
                print(f"   ⚠️  No high-confidence data for {class_name}")
                continue
            
            # Sort by confidence descending
            candidates.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Take top 20 candidates, then randomly sample 10
            top_candidates = candidates[:min(20, len(candidates))]
            if len(top_candidates) >= 10:
                sampled_indices = np.random.choice(len(top_candidates), 10, replace=False)
                samples = [top_candidates[i] for i in sampled_indices]
            else:
                samples = top_candidates
            
            # Extract and save samples
            for idx, sample in enumerate(samples, 1):
                csv_file_path = find_csv_file(sample["file"], DATA_FILTERED_DIR)
                if not csv_file_path:
                    print(f"   ❌ Could not find: {sample['file']}")
                    continue
                
                try:
                    # Load the CSV
                    df = pd.read_csv(csv_file_path)
                    
                    # Extract window around the high-confidence row
                    row = sample["row"]
                    seq_len = model_info["seq_len"]
                    
                    # Get the window
                    start_idx = max(0, row - seq_len + 1)
                    end_idx = min(len(df), row + 1)
                    
                    if end_idx - start_idx < seq_len:
                        start_idx = max(0, end_idx - seq_len)
                    
                    window_data = df.iloc[start_idx:end_idx].copy()
                    
                    # Save
                    sample_name = f"{sample['file'].replace('.csv', '')}_{idx:02d}.csv"
                    output_path = os.path.join(
                        OUTPUT_DIR, model_name, class_name, sample_name
                    )
                    window_data.to_csv(output_path, index=False)
                    sample_count[class_name] += 1
                    
                    print(f"   ✅ {class_name} #{idx}: {sample['file']} (Row {row}, Conf: {sample['confidence']:.1%})")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
        
        # Print summary
        print(f"\n✅ {model_name} Samples Summary:")
        total = sum(sample_count.values())
        for class_name in LABEL_MAP.values():
            count = sample_count[class_name]
            status = "✅" if count == 10 else "⚠️"
            print(f"   {status} {class_name}: {count}/10 samples")


if __name__ == "__main__":
    print("🚀 Creating sample dataset from high-confidence ranges...")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    create_sample_dataset()
    
    print(f"\n{'='*60}")
    print("✅ Sample dataset creation complete!")
    print(f"{'='*60}")
