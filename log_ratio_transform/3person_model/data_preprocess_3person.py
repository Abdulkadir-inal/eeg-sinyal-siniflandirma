#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log Transform + Oran Formülleri ile Veri Ön İşleme (3 Kişi: Apo, Bahadır, Canan)

Bu script FFT hesaplanmış verilere:
1. Log Transform (log1p) uygular
2. Basit Oran Formülleri ekler (8 yeni özellik)

SADECE APO, BAHADIR ve CANAN verileri kullanılır!

KAYNAK: ../../fft_model/data_filtered/
ÇIKTI: 3person_model/
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

# ============================================================================
# AYARLAR
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
FFT_MODEL_DIR = SCRIPT_DIR.parent.parent / "fft_model"

# Filtrelenmiş veri varsa onu kullan, yoksa normal FFT veriyi
DATA_DIR_FILTERED = FFT_MODEL_DIR / "data_filtered"
DATA_DIR_NORMAL = FFT_MODEL_DIR / "data"
DATA_DIR = DATA_DIR_FILTERED if DATA_DIR_FILTERED.exists() else DATA_DIR_NORMAL

OUTPUT_DIR = SCRIPT_DIR  # 3person_model/

# SADECE BU 3 KİŞİNİN VERİLERİ KULLANILACAK
ALLOWED_PERSONS = ["apo", "bahadır", "canan"]

# FFT ile hesaplanmış 9 özellik (fft_model/data/ çıktısı)
EEG_FEATURES = ["Electrode", "Delta", "Theta", "Low Alpha", "High Alpha", 
                "Low Beta", "High Beta", "Low Gamma", "Mid Gamma"]

# Yeni oran özellikleri (8 tane)
RATIO_NAMES = [
    "Delta_Theta",      # Delta / Theta
    "Theta_Alpha",      # Theta / Alpha
    "Alpha_Beta",       # Alpha / Beta
    "Beta_Gamma",       # Beta / Gamma
    "Slow_Fast",        # (Theta + Alpha) / (Beta + Gamma)
    "Delta_Alpha",      # Delta / Alpha
    "VeryLow_High",     # (Delta + Theta) / (Alpha + Beta + Gamma)
    "Mid_Low",          # (Alpha + Beta) / (Delta + Theta)
]

WINDOW_SIZE = 128
OVERLAP = 64
START_EVENT = 33025
END_EVENT = 33024

# ============================================================================
# TRANSFORMASYON FONKSİYONLARI
# ============================================================================

def apply_log_transform(data):
    """
    Log transform uygula: log1p(x) = log(1 + x)
    Büyük değerlerdeki küçük farkları vurgular
    Negatif değerler için: sign(x) * log1p(|x|)
    """
    return np.sign(data) * np.log1p(np.abs(data))

def calculate_band_ratios(window):
    """
    8 oran özelliği hesapla (her frame için)
    
    Input: (128, 9) - 128 frame, 9 özellik
    Output: (128, 8) - 128 frame, 8 oran
    """
    delta = window[:, 1] + 1e-8
    theta = window[:, 2] + 1e-8
    low_alpha = window[:, 3] + 1e-8
    high_alpha = window[:, 4] + 1e-8
    low_beta = window[:, 5] + 1e-8
    high_beta = window[:, 6] + 1e-8
    low_gamma = window[:, 7] + 1e-8
    mid_gamma = window[:, 8] + 1e-8
    
    # Kombine bantlar
    alpha = (low_alpha + high_alpha) / 2
    beta = (low_beta + high_beta) / 2
    gamma = (low_gamma + mid_gamma) / 2
    
    # 8 oran hesapla
    ratios = np.column_stack([
        delta / theta,                          # Delta_Theta
        theta / alpha,                          # Theta_Alpha
        alpha / beta,                           # Alpha_Beta
        beta / gamma,                           # Beta_Gamma
        (theta + alpha) / (beta + gamma),       # Slow_Fast
        delta / alpha,                          # Delta_Alpha
        (delta + theta) / (alpha + beta + gamma),  # VeryLow_High
        (alpha + beta) / (delta + theta),       # Mid_Low
    ])
    
    return ratios

def transform_window(window):
    """
    Tek bir window'a tüm transformasyonları uygula
    
    Input: (128, 9)
    Output: (128, 17) - 9 orijinal (log transformed) + 8 oran
    """
    # 1. Log transform uygula
    log_transformed = apply_log_transform(window)
    
    # 2. Oranları hesapla (orijinal veriden, log'dan değil)
    ratios = calculate_band_ratios(window)
    
    # 3. Log transform'u oranlara da uygula
    ratios_log = apply_log_transform(ratios)
    
    # 4. Birleştir
    combined = np.hstack([log_transformed, ratios_log])
    
    return combined

# ============================================================================
# VERİ YÜKLEME VE İŞLEME
# ============================================================================

def load_csv_files():
    """
    fft_model/data_filtered klasöründeki FFT hesaplanmış CSV dosyalarını yükle
    SADECE APO, BAHADIR, CANAN dosyaları!
    """
    csv_files = []
    
    if not DATA_DIR.exists():
        print(f"❌ Veri dizini bulunamadı: {DATA_DIR}")
        return csv_files
    
    print(f"\n🔍 SADECE {', '.join(ALLOWED_PERSONS).upper()} verileri yükleniyor...")
    
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
            # Dosya adından kişiyi çıkar
            filename = csv_file.stem.lower()
            
            # Önce canan_annane'yi exclude et (canan_ kontrolünden önce!)
            if "annane" in filename or "ırmak" in filename:
                print(f"⏭️  Atlandı: {category_dir.name}/{csv_file.name} (hariç tutulan kişi)")
                continue
            
            # Kişi kontrolü (dosya adı kişi ismi ile başlamalı)
            person_found = False
            for person in ALLOWED_PERSONS:
                # Dosya ismi "kişi_" ile başlamalı (örn: "apo_", "bahadır_", "canan_")
                if filename.startswith(person + "_"):
                    person_found = True
                    break
            
            if not person_found:
                print(f"⏭️  Atlandı: {category_dir.name}/{csv_file.name} (izinli kişi değil)")
                continue
            
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
        valid_ends = [end for end in end_indices if end > start_idx]
        if valid_ends:
            end_idx = valid_ends[0]
            segment = df.iloc[start_idx:end_idx].copy()
            if len(segment) > 0:
                active_segments.append(segment)
    
    if not active_segments:
        print("      ⚠ Aktif segment bulunamadı, tüm veri kullanılacak")
        return [df]
    
    return active_segments

def create_windows(features, window_size=WINDOW_SIZE, overlap=OVERLAP):
    """
    Kayan pencere ile window'lar oluştur
    
    Input:
        features: (N, 9) numpy array
    Output:
        windows: (M, window_size, 9) numpy array
    """
    stride = window_size - overlap
    windows = []
    
    for i in range(0, len(features) - window_size + 1, stride):
        window = features[i:i+window_size]
        windows.append(window)
    
    return np.array(windows)

# ============================================================================
# ANA İŞLEM
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("🧠 LOG TRANSFORM + ORAN FORMÜLLERİ - VERİ ÖN İŞLEME (3 KİŞİ)")
    print("=" * 80)
    print(f"📂 Veri Dizini: {DATA_DIR}")
    print(f"📁 Çıktı Dizini: {OUTPUT_DIR}")
    print(f"👥 İzinli Kişiler: {', '.join(ALLOWED_PERSONS).upper()}")
    print(f"🪟 Window: {WINDOW_SIZE} frame, Overlap: {OVERLAP}")
    print("-" * 80)
    
    # 1. CSV dosyalarını yükle
    csv_files = load_csv_files()
    
    if not csv_files:
        print("\n❌ Hiç veri yüklenemedi!")
        return
    
    print(f"\n✅ Toplam {len(csv_files)} dosya yüklendi (sadece apo, bahadır, canan)")
    
    # 2. Window'ları oluştur
    all_windows = []
    all_labels = []
    
    label_map = {"araba": 0, "aşağı": 1, "yukarı": 2}
    
    for filename, df, class_name in csv_files:
        print(f"\n🔄 İşleniyor: {filename} → {class_name}")
        
        # Aktif segmentleri çıkar
        segments = extract_active_segments(df)
        print(f"      📦 {len(segments)} aktif segment bulundu")
        
        for seg_idx, segment in enumerate(segments, 1):
            # Özellik sütunlarını al
            features = segment[EEG_FEATURES].values
            
            if len(features) < WINDOW_SIZE:
                print(f"      ⚠ Segment {seg_idx} çok kısa ({len(features)} < {WINDOW_SIZE}), atlanıyor")
                continue
            
            # Window'ları oluştur
            windows = create_windows(features)
            
            if len(windows) == 0:
                continue
            
            all_windows.extend(windows)
            all_labels.extend([label_map[class_name]] * len(windows))
            
            print(f"      ✅ Segment {seg_idx}: {len(features)} frame → {len(windows)} window")
    
    if not all_windows:
        print("\n❌ Hiç window oluşturulamadı!")
        return
    
    # 3. Numpy array'e çevir
    X = np.array(all_windows, dtype=np.float32)  # (N, 128, 9)
    y = np.array(all_labels, dtype=np.int64)     # (N,)
    
    print(f"\n📊 Ham Veri:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Sınıf dağılımı:")
    for class_name, label_idx in label_map.items():
        count = np.sum(y == label_idx)
        print(f"      {class_name:8s}: {count:5d} ({count/len(y)*100:.1f}%)")
    
    # 4. Log Transform + Oran Formülleri uygula
    print(f"\n🔄 Transform uygulanıyor...")
    X_transformed = []
    
    for i, window in enumerate(X):
        transformed = transform_window(window)  # (128, 9) → (128, 17)
        X_transformed.append(transformed)
        
        if (i + 1) % 1000 == 0:
            print(f"   {i+1}/{len(X)} window işlendi...")
    
    X_transformed = np.array(X_transformed, dtype=np.float32)
    
    print(f"\n✅ Transform tamamlandı!")
    print(f"   X_transformed shape: {X_transformed.shape}")
    print(f"   Özellikler: 9 FFT + 8 Oran = 17 toplam")
    
    # 5. Normalizasyon (StandardScaler)
    print(f"\n🔄 StandardScaler uygulanıyor...")
    
    # Window'ları flat'le: (N, 128, 17) → (N, 128*17)
    n_samples = X_transformed.shape[0]
    X_flat = X_transformed.reshape(n_samples, -1)
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_flat)
    
    # Tekrar reshape: (N, 128*17) → (N, 128, 17)
    X_final = X_normalized.reshape(n_samples, WINDOW_SIZE, 17)
    
    print(f"   ✅ Normalizasyon tamamlandı")
    
    # 6. Dosyalara kaydet
    print(f"\n💾 Dosyalar kaydediliyor...")
    
    X_path = OUTPUT_DIR / "X_3person.npy"
    y_path = OUTPUT_DIR / "y_3person.npy"
    scaler_path = OUTPUT_DIR / "scaler_3person.pkl"
    label_map_path = OUTPUT_DIR / "label_map_3person.json"
    
    np.save(X_path, X_final)
    np.save(y_path, y)
    
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"   ✅ X_3person.npy kaydedildi: {X_final.shape}")
    print(f"   ✅ y_3person.npy kaydedildi: {y.shape}")
    print(f"   ✅ scaler_3person.pkl kaydedildi")
    print(f"   ✅ label_map_3person.json kaydedildi")
    
    # 7. Özet istatistikler
    print("\n" + "=" * 80)
    print("📊 ÖZET İSTATİSTİKLER (3 KİŞİ)")
    print("=" * 80)
    print(f"👥 Kullanılan Kişiler: {', '.join(ALLOWED_PERSONS).upper()}")
    print(f"📁 Toplam dosya: {len(csv_files)}")
    print(f"🪟 Toplam window: {len(X_final)}")
    print(f"🔢 Özellik sayısı: {X_final.shape[2]} (9 FFT + 8 Oran)")
    print(f"📏 Window boyutu: {WINDOW_SIZE} frame")
    print(f"\n🎯 Sınıf dağılımı:")
    for class_name, label_idx in label_map.items():
        count = np.sum(y == label_idx)
        print(f"   {class_name:8s}: {count:5d} ({count/len(y)*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
