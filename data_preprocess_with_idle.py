#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Veri Ön İşleme - "Durgun" Sınıfı ile
CSV dosyalarındaki START/END işaretleri dışındaki bölgeleri "durgun" olarak etiketle
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

DATA_DIR = "/home/kadir/sanal-makine/python/proje"
EEG_FEATURES = ["Electrode", "Delta", "Theta", "Low Alpha", "High Alpha", "Low Beta", "High Beta", "Low Gamma", "Mid Gamma"]
WINDOW_SIZE = 128
OVERLAP = 64
START_EVENT = 33025
END_EVENT = 33024

def load_csv_files(data_dir):
    """CSV dosyalarını yükle"""
    csv_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                csv_files.append((filename, df))
                print(f"✅ Yüklendi: {filename} ({len(df)} satır)")
            except Exception as e:
                print(f"❌ Hata: {e}")
    return csv_files

def extract_segments_with_idle(df):
    """
    START/END işaretleri arasındaki segmentleri ve dışındaki "durgun" bölgeleri ayır
    
    Returns:
        active_segments: Komut segmentleri (list of DataFrame)
        idle_segments: Durgun bölgeler (list of DataFrame)
    """
    active_segments = []
    idle_segments = []
    
    if "Event Id" not in df.columns:
        # Event Id yoksa tüm veri durgun
        idle_segments.append(df.copy())
        return active_segments, idle_segments
    
    # Başlangıç ve bitiş indekslerini bul
    start_indices = df[df["Event Id"] == START_EVENT].index.tolist()
    end_indices = df[df["Event Id"] == END_EVENT].index.tolist()
    
    print(f"   📍 Başlangıç işaretleri: {len(start_indices)}")
    print(f"   📍 Bitiş işaretleri: {len(end_indices)}")
    
    # Aktif bölgeleri çıkar (START ve END arasında)
    active_ranges = []
    for start_idx in start_indices:
        valid_ends = [end for end in end_indices if end > start_idx]
        if valid_ends:
            end_idx = valid_ends[0]
            segment = df.iloc[start_idx+1:end_idx].copy()
            if len(segment) > 0:
                active_segments.append(segment)
                active_ranges.append((start_idx, end_idx))
                print(f"   ✅ Aktif segment: {len(segment)} satır")
    
    # Durgun bölgeleri çıkar (START/END dışındaki her şey)
    if len(active_ranges) == 0:
        # Hiç aktif segment yoksa tüm veri durgun
        idle_segments.append(df.copy())
        print(f"   💤 Durgun bölge (tüm veri): {len(df)} satır")
    else:
        # İlk aktif segmentten öncesi
        if active_ranges[0][0] > 0:
            idle_seg = df.iloc[0:active_ranges[0][0]].copy()
            if len(idle_seg) > WINDOW_SIZE:  # Minimum pencere boyutu kadar olmalı
                idle_segments.append(idle_seg)
                print(f"   💤 Durgun bölge (başlangıç): {len(idle_seg)} satır")
        
        # Aktif segmentler arası
        for i in range(len(active_ranges) - 1):
            start = active_ranges[i][1]
            end = active_ranges[i+1][0]
            if end - start > WINDOW_SIZE:
                idle_seg = df.iloc[start:end].copy()
                idle_segments.append(idle_seg)
                print(f"   💤 Durgun bölge (ara): {len(idle_seg)} satır")
        
        # Son aktif segmentten sonrası
        if active_ranges[-1][1] < len(df):
            idle_seg = df.iloc[active_ranges[-1][1]:].copy()
            if len(idle_seg) > WINDOW_SIZE:
                idle_segments.append(idle_seg)
                print(f"   💤 Durgun bölge (son): {len(idle_seg)} satır")
    
    return active_segments, idle_segments

def create_windows(segment):
    """Veri segmentinden sliding window'lar oluştur"""
    data = segment[EEG_FEATURES].values
    data = np.nan_to_num(data, nan=0.0)
    
    windows = []
    step = WINDOW_SIZE - OVERLAP
    
    for i in range(0, len(data) - WINDOW_SIZE + 1, step):
        window = data[i:i + WINDOW_SIZE]
        windows.append(window)
    
    return np.array(windows)

def process_all_data(csv_files, include_idle=True):
    """
    Tüm CSV dosyalarını işle
    
    Args:
        csv_files: (filename, DataFrame) tuple'larının listesi
        include_idle: Durgun sınıfını dahil et (varsayılan: True)
    
    Returns:
        X: Özellik matrisi (N, 128, 9)
        y: Etiketler (N,)
        label_map: {'asagı': 0, 'yukarı': 1, 'durgun': 2}
    """
    all_windows = []
    all_labels = []
    label_map = {}
    current_label = 0
    
    # İlk olarak tüm komut sınıflarını işle
    for filename, df in csv_files:
        print(f"\n{'='*60}")
        print(f"📂 İşleniyor: {filename}")
        print(f"{'='*60}")
        
        # Etiket ismini dosya adından al
        label_name = filename.replace(".csv", "")
        
        if label_name not in label_map:
            label_map[label_name] = current_label
            current_label += 1
        
        label = label_map[label_name]
        print(f"🏷️  Etiket: '{label_name}' → {label}")
        
        # Aktif ve durgun segmentleri ayır
        active_segments, idle_segments = extract_segments_with_idle(df)
        
        # Aktif segmentler (komutlar)
        for seg_idx, segment in enumerate(active_segments):
            windows = create_windows(segment)
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.extend([label] * len(windows))
                print(f"   ✅ Aktif segment {seg_idx+1}: {len(windows)} pencere")
    
    # Durgun sınıfını ekle
    if include_idle:
        # "durgun" sınıfını label_map'e ekle
        label_map["durgun"] = current_label
        durgun_label = current_label
        
        print(f"\n{'='*60}")
        print(f"💤 DURGUN SINIFI EKLENIYOR")
        print(f"{'='*60}")
        print(f"🏷️  Etiket: 'durgun' → {durgun_label}")
        
        # Tüm CSV dosyalarındaki durgun bölgeleri topla
        total_idle_windows = 0
        for filename, df in csv_files:
            print(f"\n📂 {filename} - durgun bölgeler:")
            _, idle_segments = extract_segments_with_idle(df)
            
            for seg_idx, segment in enumerate(idle_segments):
                windows = create_windows(segment)
                if len(windows) > 0:
                    all_windows.append(windows)
                    all_labels.extend([durgun_label] * len(windows))
                    total_idle_windows += len(windows)
                    print(f"   💤 Durgun segment {seg_idx+1}: {len(windows)} pencere")
        
        print(f"\n✅ Toplam durgun pencere: {total_idle_windows}")
    
    # Tüm window'ları birleştir
    if all_windows:
        X = np.vstack(all_windows)
        y = np.array(all_labels)
        
        print(f"\n{'='*60}")
        print(f"📊 ÖZET")
        print(f"{'='*60}")
        print(f"📦 Toplam pencere: {len(X)}")
        print(f"📐 Pencere şekli: {X.shape}")
        print(f"\n🏷️  Sınıf dağılımı:")
        
        reverse_label_map = {v: k for k, v in label_map.items()}
        for label_idx in sorted(reverse_label_map.keys()):
            label_name = reverse_label_map[label_idx]
            count = np.sum(y == label_idx)
            percentage = (count / len(y)) * 100
            print(f"   {label_name:10s}: {count:5d} pencere ({percentage:5.1f}%)")
        
        return X, y, label_map
    else:
        print("\n❌ Hiç pencere oluşturulamadı!")
        return None, None, None

def normalize_data(X):
    """Verileri normalize et"""
    scaler = StandardScaler()
    
    # Her özelliği ayrı ayrı normalize et
    X_normalized = np.zeros_like(X)
    for feature_idx in range(X.shape[2]):
        feature_data = X[:, :, feature_idx].reshape(-1, 1)
        normalized = scaler.fit_transform(feature_data)
        X_normalized[:, :, feature_idx] = normalized.reshape(X.shape[0], X.shape[1])
    
    return X_normalized, scaler

def save_data(X, y, label_map, output_dir=DATA_DIR):
    """Verileri kaydet"""
    print(f"\n{'='*60}")
    print(f"💾 VERİLER KAYDEDİLİYOR")
    print(f"{'='*60}")
    
    # NumPy array'leri kaydet
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    print(f"✅ X.npy kayıt edildi: {X.shape}")
    print(f"✅ y.npy kayıt edildi: {y.shape}")
    
    # Label map'i JSON olarak kaydet
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"✅ label_map.json kayıt edildi")
    print(f"   Sınıflar: {label_map}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 EEG VERİ ÖN İŞLEME - DURGUN SINIFI İLE")
    print("="*60)
    
    # CSV dosyalarını yükle
    csv_files = load_csv_files(DATA_DIR)
    
    if not csv_files:
        print("\n❌ CSV dosyası bulunamadı!")
        exit(1)
    
    # Verileri işle (durgun sınıfı dahil)
    X, y, label_map = process_all_data(csv_files, include_idle=True)
    
    if X is not None and y is not None:
        # Verileri kaydet
        save_data(X, y, label_map)
        
        print(f"\n{'='*60}")
        print("✅ ÖN İŞLEME TAMAMLANDI!")
        print(f"{'='*60}")
        print(f"📂 Çıktı dosyaları: {DATA_DIR}")
        print(f"   - X.npy: {X.shape}")
        print(f"   - y.npy: {y.shape}")
        print(f"   - label_map.json")
        print(f"\n🎯 Şimdi modeli yeniden eğitin:")
        print(f"   python3 train_model.py")
    else:
        print("\n❌ Veri işleme başarısız!")
