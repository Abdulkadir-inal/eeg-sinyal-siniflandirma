#!/usr/bin/env python3
"""
Create sample dataset from test results.
Each model gets 10 random samples per class from high-confidence predictions.
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
CONFIDENCE_THRESHOLD = 0.90  # Only consider predictions with >= 90% confidence

MODELS = {
    "seq32": {"file": "son_sonuclar.txt", "seq_len": 32},
    "seq64": {"file": "son_sonuclar_64.txt", "seq_len": 64},
    "seq96": {"file": "son_sonuclar_96.txt", "seq_len": 96},
}

LABEL_MAP = {0: "yukarı", 1: "aşağı", 2: "araba"}
CLASS_MAP = {"yukarı": 0, "aşağı": 1, "araba": 2}


def parse_verbose_results(file_path):
    """Parse verbose results file to extract (filepath, row_index, class, confidence)"""
    results = []
    current_file = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Check for file header (starts with ##)
        if line.startswith("##"):
            current_file = line.replace("##", "").strip()
        
        # Check for prediction lines
        # Format: "    32: yukarı     (100.0%)"
        if current_file and ":" in line and "%" in line and "(" in line:
            try:
                # Extract row number, class name, and confidence
                match = re.match(r"\s*(\d+):\s+([\w]+)\s+\(([0-9.]+)%\)", line)
                if match:
                    row_idx = int(match.group(1))
                    class_name = match.group(2)
                    confidence = float(match.group(3)) / 100.0
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        results.append({
                            "file": current_file,
                            "row": row_idx,
                            "class": class_name,
                            "confidence": confidence
                        })
            except:
                pass
    
    return results


def parse_compact_results(file_path):
    """Parse compact results file"""
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_file = None
    for line in lines:
        line = line.strip()
        
        # Check for file header
        if line.startswith("##"):
            current_file = line.replace("##", "").strip()
        
        # Check for predictions
        if current_file and ":" in line and "%" in line and "(" in line:
            try:
                match = re.match(r"\s*(\d+):\s+([\w]+)\s+\(([0-9.]+)%\)", line)
                if match:
                    row_idx = int(match.group(1))
                    class_name = match.group(2)
                    confidence = float(match.group(3)) / 100.0
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        results.append({
                            "file": current_file,
                            "row": row_idx,
                            "class": class_name,
                            "confidence": confidence
                        })
            except:
                pass
    
    return results


def load_csv_data(filepath, row_idx, seq_len):
    """Load a single window from CSV file"""
    df = pd.read_csv(filepath)
    
    # Get 96 rows starting from row_idx (cover all seq_len variants)
    start = max(0, row_idx - seq_len + 1)
    end = min(len(df), row_idx + 1)
    
    if end - start < seq_len:
        start = max(0, end - seq_len)
    
    return df.iloc[start:end].copy()


def find_csv_file(filename, base_dir):
    """Find CSV file with Turkish character handling"""
    # Try direct path first
    for class_name in ["yukarı", "aşağı", "araba"]:
        candidate = os.path.join(base_dir, class_name, filename)
        if os.path.exists(candidate):
            return candidate
    
    # Try all files in base_dir recursively
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower() == filename.lower() or file == filename:
                return os.path.join(root, file)
    
    return None


def create_sample_dataset():
    """Create sample dataset for all models"""
    
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
        
        result_file = os.path.join(
            "/home/kadir/sanal-makine/python/proje/lstm_cnn_hybrid",
            model_info["file"]
        )
        
        if not os.path.exists(result_file):
            print(f"⚠️  Results file not found: {result_file}")
            continue
        
        # Parse results
        results = parse_verbose_results(result_file)
        
        print(f"📊 Found {len(results)} high-confidence predictions")
        
        # Organize by class
        by_class = {cls: [] for cls in LABEL_MAP.values()}
        for result in results:
            by_class[result["class"]].append(result)
        
        # Sample 10 per class
        samples_per_class = {}
        for class_name in LABEL_MAP.values():
            candidates = by_class[class_name]
            if candidates:
                # Sort by confidence descending, then sample randomly
                candidates.sort(key=lambda x: x["confidence"], reverse=True)
                # Take top 20, then random 10
                top_candidates = candidates[:min(20, len(candidates))]
                if top_candidates:
                    samples = np.random.choice(
                        len(top_candidates),
                        min(10, len(top_candidates)),
                        replace=False
                    )
                    samples_per_class[class_name] = [top_candidates[i] for i in samples]
        
        # If no detailed predictions found (compact mode), sample from files directly
        if not samples_per_class or all(len(v) == 0 for v in samples_per_class.values()):
            print(f"   ℹ️  Sampling from files (compact mode)")
            # Re-parse to get file->class mapping
            by_file_class = {}
            with open(result_file, 'r', encoding='utf-8') as f:
                current_file = None
                for line in f:
                    line = line.strip()
                    if line.startswith("##"):
                        current_file = line.replace("##", "").strip()
                    if "Dağılım:" in line:
                        # Parse distribution to get predicted class
                        import ast
                        try:
                            dist_str = line.split("Dağılım:")[1].strip()
                            dist_dict = ast.literal_eval(dist_str)
                            # Get the most predicted class
                            pred_class = max(dist_dict.items(), key=lambda x: x[1])[0]
                            if current_file:
                                if pred_class not in by_file_class:
                                    by_file_class[pred_class] = []
                                by_file_class[pred_class].append(current_file)
                        except:
                            pass
            
            # Create samples from files
            samples_per_class = {}
            for class_name in LABEL_MAP.values():
                files = by_file_class.get(class_name, [])
                if files:
                    # Need exactly 10 samples - replicate files if necessary
                    if len(files) < 10:
                        # Repeat some files to get 10 total
                        extra_needed = 10 - len(files)
                        repeated = list(np.random.choice(files, size=extra_needed, replace=True))
                        files = list(files) + repeated
                    
                    # Sample 10 files
                    sampled_indices = np.random.choice(len(files), 10, replace=False)
                    samples_per_class[class_name] = [
                        {"file": files[i], "row": 0, "class": class_name, "confidence": 0.95}
                        for i in sampled_indices
                    ]
        
        # Save samples
        sample_count = {cls: 0 for cls in LABEL_MAP.values()}
        
        for class_name, samples in samples_per_class.items():
            for idx, sample in enumerate(samples, 1):
                # Use flexible file finding
                csv_file = find_csv_file(sample["file"], DATA_FILTERED_DIR)
                
                if not csv_file:
                    continue
                
                try:
                    # Load data
                    if "row" in sample:
                        data = load_csv_data(csv_file, sample["row"], model_info["seq_len"])
                    else:
                        # For compact results without row info, load first seq_len rows
                        df = pd.read_csv(csv_file)
                        data = df.iloc[:model_info["seq_len"]].copy()
                    
                    # Save as CSV
                    sample_name = f"{sample['file'].replace('.csv', '')}_{idx:02d}.csv"
                    output_path = os.path.join(
                        OUTPUT_DIR, model_name, class_name, sample_name
                    )
                    data.to_csv(output_path, index=False)
                    sample_count[class_name] += 1
                    
                except Exception as e:
                    print(f"❌ Error processing {sample['file']}: {e}")
        
        # Print summary
        print(f"\n✅ {model_name} Samples Created:")
        for class_name in LABEL_MAP.values():
            count = sample_count[class_name]
            print(f"   {class_name}: {count} samples")


if __name__ == "__main__":
    print("🚀 Creating sample dataset from test results...")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    create_sample_dataset()
    
    print(f"\n{'='*60}")
    print("✅ Sample dataset creation complete!")
    print(f"{'='*60}")
