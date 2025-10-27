#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Veri Ön İşleme - 4 Sınıf Sistemi
GÜNCELLEME (27 Ekim 2025): araba, yukarı, aşağı, boş sınıfları için
- START/END işaretleri arasındaki bölgeler → komut sınıfları (araba, yukarı, aşağı)
- START/END işaretleri dışındaki bölgeler → "boş" sınıfı
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

# Veri dizinleri - hem proje klasörü hem de proje-veri klasörü
DATA_DIRS = [
    "/home/kadir/sanal-makine/python/proje",
    "/home/kadir/sanal-makine/python/proje-veri/araba",
    "/home/kadir/sanal-makine/python/proje-veri/yukarı", 
    "/home/kadir/sanal-makine/python/proje-veri/asagı"
]
OUTPUT_DIR = "/home/kadir/sanal-makine/python/proje"

EEG_FEATURES = ["Electrode", "Delta", "Theta", "Low Alpha", "High Alpha", "Low Beta", "High Beta", "Low Gamma", "Mid Gamma"]
WINDOW_SIZE = 128
OVERLAP = 64
START_EVENT = 33025
END_EVENT = 33024

def load_csv_files(data_dirs):
    """
    Birden fazla dizindeki CSV dosyalarını yükle
    Her dosyanın hangi sınıfa ait olduğunu dosya adından veya dizinden belirle
    Test dosyalarını (test_*.csv) hariç tut
    """
    csv_files = []
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
        for filename in os.listdir(data_dir):
            # Test dosyalarını atla
            if filename.startswith("test_") or not filename.endswith(".csv"):
                continue
                
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                # Dosya adından veya dizin adından sınıf bilgisini al
                if "araba" in filename.lower() or "araba" in data_dir.lower():
                    class_name = "araba"
                elif "yukarı" in filename.lower() or "yukarı" in data_dir.lower():
                    class_name = "yukarı"
                elif "aşağı" in filename.lower() or "asagı" in filename.lower() or "asagı" in data_dir.lower():
                    class_name = "aşağı"
                else:
                    class_name = filename.replace(".csv", "")
                
                csv_files.append((filename, df, class_name))
                print(f"✅ Yüklendi: {filename} → {class_name} ({len(df)} satır)")
            except Exception as e:
                print(f"❌ Hata ({filename}): {e}")
    return csv_files

def extract_segments_with_idle(df):
    """
    START/END işaretleri arasındaki segmentleri ve dışındaki "boş" bölgeleri ayır
    
    Returns:
        active_segments: Komut segmentleri (list of DataFrame)
        idle_segments: Boş bölgeler (list of DataFrame)
    """
    active_segments = []
    idle_segments = []
    
    if "Event Id" not in df.columns:
        # Event Id yoksa tüm veri boş
        if len(df) > WINDOW_SIZE:
            idle_segments.append(df.copy())
        return active_segments, idle_segments
    
    # Başlangıç ve bitiş indekslerini bul
    start_indices = df[df["Event Id"] == START_EVENT].index.tolist()
    end_indices = df[df["Event Id"] == END_EVENT].index.tolist()
    
    print(f"   📍 Başlangıç: {len(start_indices)}, Bitiş: {len(end_indices)}")
    
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
    
    # Boş bölgeleri çıkar (START/END dışındaki her şey)
    if len(active_ranges) == 0:
        # Hiç aktif segment yoksa tüm veri boş
        if len(df) > WINDOW_SIZE:
            idle_segments.append(df.copy())
            print(f"   💤 Boş bölge (tüm veri): {len(df)} satır")
    else:
        # İlk aktif segmentten öncesi
        if active_ranges[0][0] > 0:
            idle_seg = df.iloc[0:active_ranges[0][0]].copy()
            if len(idle_seg) > WINDOW_SIZE:
                idle_segments.append(idle_seg)
                print(f"   💤 Boş bölge (başlangıç): {len(idle_seg)} satır")
        
        # Aktif segmentler arası
        for i in range(len(active_ranges) - 1):
            start = active_ranges[i][1]
            end = active_ranges[i+1][0]
            if end - start > WINDOW_SIZE:
                idle_seg = df.iloc[start:end].copy()
                idle_segments.append(idle_seg)
                print(f"   💤 Boş bölge (ara): {len(idle_seg)} satır")
        
        # Son aktif segmentten sonrası
        if active_ranges[-1][1] < len(df):
            idle_seg = df.iloc[active_ranges[-1][1]:].copy()
            if len(idle_seg) > WINDOW_SIZE:
                idle_segments.append(idle_seg)
                print(f"   💤 Boş bölge (son): {len(idle_seg)} satır")
    
    return active_segments, idle_segments

def extract_segments(df):
    """Eski fonksiyon - sadece aktif segmentleri döndür"""
    active_segments, _ = extract_segments_with_idle(df)
    return active_segments

def create_windows(segment):
    data = segment[EEG_FEATURES].values
    data = np.nan_to_num(data, nan=0.0)
    windows = []
    step = WINDOW_SIZE - OVERLAP
    for i in range(0, len(data) - WINDOW_SIZE + 1, step):
        window = data[i:i + WINDOW_SIZE]
        windows.append(window)
    return np.array(windows)

def process_all_data(csv_files, include_idle=False):
    """
    Tüm CSV dosyalarını işle - 3 sınıf sistemi (araba, yukarı, aşağı)
    
    Args:
        csv_files: (filename, DataFrame, class_name) tuple'larının listesi
        include_idle: Boş sınıfını dahil et (varsayılan: False - BOŞ SINIFI YOK)
    
    Returns:
        X: Özellik matrisi (N, 128, 9)
        y: Etiketler (N,)
        label_map: {'araba': 0, 'yukarı': 1, 'aşağı': 2}
    """
    all_windows = []
    all_labels = []
    label_map = {}
    current_label = 0
    
    # İlk olarak tüm komut sınıflarını işle
    for filename, df, class_name in csv_files:
        print(f"\n{'='*60}")
        print(f"📂 İşleniyor: {filename} → {class_name}")
        print(f"{'='*60}")
        
        if class_name not in label_map:
            label_map[class_name] = current_label
            current_label += 1
        
        label = label_map[class_name]
        print(f"🏷️  Etiket: '{class_name}' → {label}")
        
        # Aktif ve boş segmentleri ayır
        active_segments, idle_segments = extract_segments_with_idle(df)
        
        # Aktif segmentler (komutlar)
        for seg_idx, segment in enumerate(active_segments):
            windows = create_windows(segment)
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.extend([label] * len(windows))
                print(f"   ✅ Aktif segment {seg_idx+1}: {len(windows)} pencere")
    
    # Boş sınıfını ekle
    if include_idle:
        # "boş" sınıfını label_map'e ekle
        label_map["boş"] = current_label
        idle_label = current_label
        
        print(f"\n{'='*60}")
        print(f"💤 BOŞ SINIFI EKLENIYOR")
        print(f"{'='*60}")
        print(f"🏷️  Etiket: 'boş' → {idle_label}")
        
        # Tüm CSV dosyalarındaki boş bölgeleri topla
        total_idle_windows = 0
        for filename, df, class_name in csv_files:
            print(f"\n📂 {filename} - boş bölgeler:")
            _, idle_segments = extract_segments_with_idle(df)
            
            for seg_idx, segment in enumerate(idle_segments):
                windows = create_windows(segment)
                if len(windows) > 0:
                    all_windows.append(windows)
                    all_labels.extend([idle_label] * len(windows))
                    total_idle_windows += len(windows)
                    print(f"   💤 Boş segment {seg_idx+1}: {len(windows)} pencere")
        
        print(f"\n✅ Toplam boş pencere: {total_idle_windows}")
    
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_eeg_window.png'), dpi=150)
    print("   ✅ Grafik kaydedildi")
    plt.close()

def save_data(X, y, label_map):
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
    with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"\n   ✅ X.npy: {X.shape}")
    print(f"   ✅ y.npy: {y.shape}")
    print(f"   ✅ label_map.json kaydedildi")

def main():
    print("\n" + "="*60)
    print("🧠 EEG VERI ÖN İŞLEME - 3 SINIF SİSTEMİ (BOŞ SINIFI YOK)")
    print("="*60)
    print("📂 Veri dizinleri:")
    for d in DATA_DIRS:
        if os.path.exists(d):
            print(f"   ✅ {d}")
        else:
            print(f"   ⚠️  {d} (bulunamadı)")
    
    print("\n⚠️  NOT: Boş sınıfı DAHIL EDİLMEYECEK!")
    print("   → Sadece aktif komutlar: araba, yukarı, aşağı")
    print("   → Boş bölgeler atlanacak")
    
    csv_files = load_csv_files(DATA_DIRS)
    if not csv_files:
        print("\n❌ CSV dosyası bulunamadı!")
        return
    
    X, y, label_map = process_all_data(csv_files, include_idle=False)
    if X is None:
        print("\n❌ Veri işleme başarısız!")
        return
    
    X_normalized, scaler = normalize_data(X)
    visualize_sample(X_normalized, y, label_map, 0)
    save_data(X_normalized, y, label_map)
    
    print("\n" + "="*60)
    print("✅ TAMAMLANDI!")
    print("="*60)
    print(f"📂 Çıktı: {OUTPUT_DIR}")
    print("🎯 Şimdi modeli eğitin: python3 train_model.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
