#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

DATA_DIR = "/home/kadir/sanal-makine/python/proje"
EEG_FEATURES = ["Electrode", "Delta", "Theta", "Low Alpha", "High Alpha", "Low Beta", "High Beta", "Low Gamma", "Mid Gamma"]
WINDOW_SIZE = 128
OVERLAP = 64
START_EVENT = 33025
END_EVENT = 33024

def load_csv_files(data_dir):
    csv_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                csv_files.append((filename, df))
                print(f"Yuklendi: {filename} ({len(df)} satir)")
            except Exception as e:
                print(f"Hata: {e}")
    return csv_files

def extract_segments(df):
    segments = []
    if "Event Id" not in df.columns:
        return segments
    start_indices = df[df["Event Id"] == START_EVENT].index.tolist()
    end_indices = df[df["Event Id"] == END_EVENT].index.tolist()
    print(f"   Baslangic: {len(start_indices)}, Bitis: {len(end_indices)}")
    for start_idx in start_indices:
        valid_ends = [end for end in end_indices if end > start_idx]
        if valid_ends:
            end_idx = valid_ends[0]
            segment = df.iloc[start_idx+1:end_idx].copy()
            if len(segment) > 0:
                segments.append(segment)
                print(f"   Segment: {len(segment)} satir")
    return segments

def create_windows(segment):
    data = segment[EEG_FEATURES].values
    data = np.nan_to_num(data, nan=0.0)
    windows = []
    step = WINDOW_SIZE - OVERLAP
    for i in range(0, len(data) - WINDOW_SIZE + 1, step):
        window = data[i:i + WINDOW_SIZE]
        windows.append(window)
    return np.array(windows)

def process_all_data(csv_files):
    all_windows = []
    all_labels = []
    label_map = {}
    current_label = 0
    for filename, df in csv_files:
        print(f"\nIsleniyor: {filename}")
        label_name = filename.replace(".csv", "")
        if label_name not in label_map:
            label_map[label_name] = current_label
            current_label += 1
        label = label_map[label_name]
        print(f"   Etiket: {label_name} -> {label}")
        segments = extract_segments(df)
        for seg_idx, segment in enumerate(segments):
            windows = create_windows(segment)
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.extend([label] * len(windows))
                print(f"   Segment {seg_idx+1}: {len(windows)} pencere")
    if all_windows:
        X = np.vstack(all_windows)
        y = np.array(all_labels)
        print(f"\n{'='*60}")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        for label_name, label_id in label_map.items():
            count = np.sum(y == label_id)
            print(f"   {label_name}: {count} pencere")
        print(f"{'='*60}")
        return X, y, label_map
    return None, None, None

def normalize_data(X):
    print("\nNormalizasyon...")
    original_shape = X.shape
    X_flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_normalized_flat = scaler.fit_transform(X_flat)
    X_normalized = X_normalized_flat.reshape(original_shape)
    print(f"   Mean: {X_normalized.mean():.4f}, Std: {X_normalized.std():.4f}")
    return X_normalized, scaler

def visualize_sample(X, y, label_map, sample_idx=0):
    label_names = {v: k for k, v in label_map.items()}
    label_name = label_names[y[sample_idx]]
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle(f'EEG - {label_name}', fontsize=16)
    for idx, (ax, feature_name) in enumerate(zip(axes.flat, EEG_FEATURES)):
        data = X[sample_idx, :, idx]
        ax.plot(data)
        ax.set_title(feature_name)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'sample_eeg_window.png'), dpi=150)
    print("   Grafik kaydedildi")
    plt.close()

def save_data(X, y, label_map):
    np.save(os.path.join(DATA_DIR, 'X.npy'), X)
    np.save(os.path.join(DATA_DIR, 'y.npy'), y)
    with open(os.path.join(DATA_DIR, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"\n   X.npy: {X.shape}")
    print(f"   y.npy: {y.shape}")
    print(f"   label_map.json kaydedildi")

def main():
    print("\n" + "="*60)
    print("EEG VERI ON ISLEME")
    print("="*60)
    csv_files = load_csv_files(DATA_DIR)
    if not csv_files:
        print("CSV dosyasi yok!")
        return
    X, y, label_map = process_all_data(csv_files)
    if X is None:
        print("Veri isleme basarisiz!")
        return
    X_normalized, scaler = normalize_data(X)
    visualize_sample(X_normalized, y, label_map, 0)
    save_data(X_normalized, y, label_map)
    print("\n" + "="*60)
    print("TAMAMLANDI!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
