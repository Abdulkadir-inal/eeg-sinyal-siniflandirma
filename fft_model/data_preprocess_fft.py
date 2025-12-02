#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT Tabanlı Veri Ön İşleme
==========================
Bu script, Raw EEG'den FFT ile hesaplanmış bant değerlerini kullanarak
model eğitimi için veri hazırlar.

ÖNEMLİ: Event Id sütunundaki START (33025) ve END (33024) işaretleri
kullanılarak sadece aktif (düşünme) bölgeleri alınır.

Veri kaynağı: ./data/ (convert_raw_to_fft.py çıktısı)
Çıktı: X_fft.npy, y_fft.npy, label_map_fft.json
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Veri dizinleri
SCRIPT_DIR = Path(__file__).parent
# Filtrelenmiş veri varsa onu kullan, yoksa normal veriyi
DATA_DIR_FILTERED = SCRIPT_DIR / "data_filtered"
DATA_DIR_NORMAL = SCRIPT_DIR / "data"
DATA_DIR = DATA_DIR_FILTERED if DATA_DIR_FILTERED.exists() else DATA_DIR_NORMAL
OUTPUT_DIR = SCRIPT_DIR  # fft_model/

# EEG özellikleri (FFT ile hesaplanmış)
EEG_FEATURES = ["Electrode", "Delta", "Theta", "Low Alpha", "High Alpha", 
                "Low Beta", "High Beta", "Low Gamma", "Mid Gamma"]

# Pencere ayarları
WINDOW_SIZE = 128  # Orijinal modelle aynı
OVERLAP = 64       # %50 overlap

# Event işaretleri
START_EVENT = 33025
END_EVENT = 33024


def load_csv_files():
    """
    fft_model/data klasöründeki tüm CSV dosyalarını yükle
    """
    csv_files = []
    
    if not DATA_DIR.exists():
        print(f"❌ Veri dizini bulunamadı: {DATA_DIR}")
        return csv_files
    
    # Kategori klasörlerini tara
    for category_dir in sorted(DATA_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        
        class_name = category_dir.name
        
        # Kategori düzeltmeleri
        if class_name == "asagı":
            class_name = "aşağı"
        
        # Bu kategorideki CSV dosyalarını yükle
        for csv_file in sorted(category_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
                csv_files.append((csv_file.name, df, class_name))
                print(f"✅ Yüklendi: {category_dir.name}/{csv_file.name} → {class_name} ({len(df)} satır)")
            except Exception as e:
                print(f"❌ Hata ({csv_file.name}): {e}")
    
    return csv_files


def extract_active_segments(df):
    """
    Event Id sütunundaki START/END işaretlerini kullanarak
    sadece aktif (düşünme) bölgelerini çıkar
    
    Returns:
        list of DataFrames: Aktif segmentler
    """
    active_segments = []
    
    if 'Event Id' not in df.columns:
        # Event Id yoksa tüm veriyi döndür (eski davranış)
        print("      ⚠ Event Id sütunu yok, tüm veri kullanılacak")
        return [df]
    
    # Event Id'leri sayısal değerlere çevir (NaN'ları 0 yap)
    event_ids = pd.to_numeric(df['Event Id'], errors='coerce').fillna(0).astype(int)
    
    # Başlangıç ve bitiş indekslerini bul
    start_indices = df.index[event_ids == START_EVENT].tolist()
    end_indices = df.index[event_ids == END_EVENT].tolist()
    
    if not start_indices:
        print("      ⚠ START işareti bulunamadı, tüm veri kullanılacak")
        return [df]
    
    print(f"      📍 {len(start_indices)} START, {len(end_indices)} END işareti bulundu")
    
    # Her START için en yakın END'i bul
    for start_idx in start_indices:
        # Bu START'tan sonraki END'leri bul
        valid_ends = [end for end in end_indices if end > start_idx]
        if valid_ends:
            end_idx = valid_ends[0]
            # START ve END arasındaki veriyi al
            segment = df.iloc[start_idx:end_idx+1].copy()
            if len(segment) > WINDOW_SIZE:
                active_segments.append(segment)
                print(f"      ✅ Aktif segment: {len(segment)} satır ({len(segment)/512:.1f}s)")
    
    if not active_segments:
        print("      ⚠ Aktif segment bulunamadı")
    
    return active_segments


def create_windows(df):
    """
    DataFrame'den sliding window'lar oluştur
    """
    # Sadece EEG özelliklerini al
    available_features = [f for f in EEG_FEATURES if f in df.columns]
    data = df[available_features].values
    data = np.nan_to_num(data, nan=0.0)
    
    windows = []
    step = WINDOW_SIZE - OVERLAP
    
    for i in range(0, len(data) - WINDOW_SIZE + 1, step):
        window = data[i:i + WINDOW_SIZE]
        windows.append(window)
    
    return np.array(windows) if windows else np.array([])


def process_all_data(csv_files):
    """
    Tüm CSV dosyalarını işle - sadece aktif bölgeleri kullan
    """
    all_windows = []
    all_labels = []
    label_map = {}
    current_label = 0
    
    for filename, df, class_name in csv_files:
        print(f"\n   📂 {filename} işleniyor...")
        
        # Sınıf etiketini ata
        if class_name not in label_map:
            label_map[class_name] = current_label
            current_label += 1
        
        label = label_map[class_name]
        
        # Aktif segmentleri çıkar
        active_segments = extract_active_segments(df)
        
        # Her segment için pencereler oluştur
        total_windows = 0
        for segment in active_segments:
            windows = create_windows(segment)
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.extend([label] * len(windows))
                total_windows += len(windows)
        
        if total_windows > 0:
            print(f"      📊 Toplam: {total_windows} pencere → etiket {label} ({class_name})")
    
    if all_windows:
        X = np.vstack(all_windows)
        y = np.array(all_labels)
        return X, y, label_map
    else:
        return None, None, None


def normalize_data(X):
    """
    StandardScaler ile normalizasyon
    """
    print("\n📐 Normalizasyon uygulanıyor...")
    
    original_shape = X.shape
    X_flat = X.reshape(X.shape[0], -1)
    
    scaler = StandardScaler()
    X_normalized_flat = scaler.fit_transform(X_flat)
    X_normalized = X_normalized_flat.reshape(original_shape)
    
    print(f"   Mean: {X_normalized.mean():.6f}")
    print(f"   Std:  {X_normalized.std():.6f}")
    
    # Scaler parametrelerini kaydet
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'feature_names': EEG_FEATURES,
        'window_size': WINDOW_SIZE
    }
    
    with open(OUTPUT_DIR / 'scaler_params_fft.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"   ✅ Scaler parametreleri kaydedildi: scaler_params_fft.json")
    
    return X_normalized, scaler


def visualize_comparison(X, y, label_map):
    """
    Sınıflar arası karşılaştırma görselleştirmesi
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('FFT Tabanlı EEG Bantları - Sınıf Karşılaştırması', fontsize=14)
    
    reverse_label_map = {v: k for k, v in label_map.items()}
    colors = ['blue', 'green', 'red']
    
    for idx, feature_name in enumerate(EEG_FEATURES):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for label_idx, color in enumerate(colors):
            if label_idx >= len(label_map):
                continue
            
            label_name = reverse_label_map[label_idx]
            mask = y == label_idx
            
            # Bu sınıfa ait tüm pencerelerin ortalaması
            class_data = X[mask, :, idx]
            mean_signal = class_data.mean(axis=0)
            std_signal = class_data.std(axis=0)
            
            x = np.arange(WINDOW_SIZE)
            ax.plot(x, mean_signal, color=color, label=label_name, linewidth=1.5)
            ax.fill_between(x, mean_signal - std_signal, mean_signal + std_signal, 
                          color=color, alpha=0.2)
        
        ax.set_title(feature_name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fft_class_comparison.png', dpi=150)
    print(f"\n   ✅ Görselleştirme kaydedildi: fft_class_comparison.png")
    plt.close()


def save_data(X, y, label_map):
    """
    İşlenmiş veriyi kaydet
    """
    np.save(OUTPUT_DIR / 'X_fft.npy', X)
    np.save(OUTPUT_DIR / 'y_fft.npy', y)
    
    with open(OUTPUT_DIR / 'label_map_fft.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Veriler kaydedildi:")
    print(f"   ✅ X_fft.npy: {X.shape}")
    print(f"   ✅ y_fft.npy: {y.shape}")
    print(f"   ✅ label_map_fft.json: {label_map}")


def main():
    print("\n" + "=" * 60)
    print("🧠 FFT TABANLI VERİ ÖN İŞLEME")
    print("=" * 60)
    print(f"📂 Veri dizini: {DATA_DIR}")
    print(f"📂 Çıktı dizini: {OUTPUT_DIR}")
    print(f"📐 Pencere boyutu: {WINDOW_SIZE}")
    print(f"📐 Overlap: {OVERLAP}")
    
    # CSV dosyalarını yükle
    print("\n📥 CSV dosyaları yükleniyor...")
    csv_files = load_csv_files()
    
    if not csv_files:
        print("\n❌ CSV dosyası bulunamadı!")
        return
    
    # Veriyi işle
    print("\n🔄 Pencereler oluşturuluyor...")
    X, y, label_map = process_all_data(csv_files)
    
    if X is None:
        print("\n❌ Veri işleme başarısız!")
        return
    
    # İstatistikler
    print("\n" + "=" * 60)
    print("📊 VERİ İSTATİSTİKLERİ")
    print("=" * 60)
    print(f"Toplam pencere: {len(X)}")
    print(f"Pencere şekli: {X.shape}")
    print(f"\n🏷️  Sınıf dağılımı:")
    
    reverse_label_map = {v: k for k, v in label_map.items()}
    for label_idx in sorted(reverse_label_map.keys()):
        label_name = reverse_label_map[label_idx]
        count = np.sum(y == label_idx)
        percentage = (count / len(y)) * 100
        print(f"   {label_name:10s}: {count:5d} pencere ({percentage:5.1f}%)")
    
    # Normalizasyon
    X_normalized, scaler = normalize_data(X)
    
    # Görselleştirme
    visualize_comparison(X_normalized, y, label_map)
    
    # Kaydet
    save_data(X_normalized, y, label_map)
    
    print("\n" + "=" * 60)
    print("✅ FFT VERİ ÖN İŞLEME TAMAMLANDI!")
    print("=" * 60)
    print("🎯 Sonraki adım: python3 train_model_fft.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
