#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log Transform + Oran Formülleri ile Veri Ön İşleme (FFT Verileri)

Bu script FFT hesaplanmış verilere:
1. Log Transform (log1p) uygular
2. Basit Oran Formülleri ekler (8 yeni özellik)

Amaç: FFT bant güçlerindeki küçük farkları büyütmek
Performans Yükü: %0.05 (pratik 0)

KAYNAK: ../fft_model/data/ veya ../fft_model/data_filtered/
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
FFT_MODEL_DIR = SCRIPT_DIR.parent / "fft_model"

# Filtrelenmiş veri varsa onu kullan, yoksa normal FFT veriyi
DATA_DIR_FILTERED = FFT_MODEL_DIR / "data_filtered"
DATA_DIR_NORMAL = FFT_MODEL_DIR / "data"
DATA_DIR = DATA_DIR_FILTERED if DATA_DIR_FILTERED.exists() else DATA_DIR_NORMAL

OUTPUT_DIR = SCRIPT_DIR  # log_ratio_transform/

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
    # Bant indeksleri (EEG_FEATURES sırasına göre)
    # 0: Electrode, 1: Delta, 2: Theta, 3: Low Alpha, 4: High Alpha
    # 5: Low Beta, 6: High Beta, 7: Low Gamma, 8: Mid Gamma
    
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
    fft_model/data veya fft_model/data_filtered klasöründeki
    FFT hesaplanmış CSV dosyalarını yükle
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
        # Event Id yoksa tüm veriyi döndür
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

def create_windows_with_transform(segment):
    """Segment'ten window'lar oluştur ve transform uygula"""
    data = segment[EEG_FEATURES].values
    data = np.nan_to_num(data, nan=0.0)
    
    windows = []
    step = WINDOW_SIZE - OVERLAP
    
    for i in range(0, len(data) - WINDOW_SIZE + 1, step):
        window = data[i:i + WINDOW_SIZE]  # (128, 9)
        
        # Transform uygula
        transformed = transform_window(window)  # (128, 17)
        windows.append(transformed)
    
    return np.array(windows) if windows else np.array([])

def process_all_data(csv_files):
    """Tüm verileri işle - sadece aktif bölgeleri kullan"""
    all_windows = []
    all_labels = []
    label_map = {}
    current_label = 0
    
    for filename, df, class_name in csv_files:
        print(f"\n{'='*60}")
        print(f"📂 İşleniyor: {filename} → {class_name}")
        
        if class_name not in label_map:
            label_map[class_name] = current_label
            current_label += 1
        
        label = label_map[class_name]
        segments = extract_active_segments(df)
        
        for seg_idx, segment in enumerate(segments):
            windows = create_windows_with_transform(segment)
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.extend([label] * len(windows))
                print(f"   ✅ Segment {seg_idx+1}: {len(windows)} pencere")
    
    if all_windows:
        X = np.vstack(all_windows)
        y = np.array(all_labels)
        return X, y, label_map
    return None, None, None

# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("🧪 LOG TRANSFORM + ORAN FORMÜLLERİ DENEMESİ")
    print("=" * 70)
    
    print("\n📋 Uygulanan Transformasyonlar:")
    print("   1. Log Transform: log1p(x) = log(1 + x)")
    print("   2. Oran Formülleri: 8 yeni özellik")
    print(f"   → Girdi: 9 özellik → Çıktı: 17 özellik")
    
    print("\n📂 Veri kaynağı:")
    exists = "✅" if DATA_DIR.exists() else "❌"
    print(f"   {exists} {DATA_DIR}")
    if DATA_DIR == DATA_DIR_FILTERED:
        print(f"   (Filtrelenmiş FFT verileri kullanılıyor)")
    else:
        print(f"   (Normal FFT verileri kullanılıyor)")
    
    # Verileri yükle
    csv_files = load_csv_files()
    if not csv_files:
        print("\n❌ CSV dosyası bulunamadı!")
        return
    
    # Verileri işle
    X, y, label_map = process_all_data(csv_files)
    if X is None:
        print("\n❌ Veri işleme başarısız!")
        return
    
    print(f"\n{'='*70}")
    print("📊 SONUÇLAR")
    print("=" * 70)
    print(f"📦 X shape: {X.shape}")
    print(f"   → Orijinal: (N, 128, 9)")
    print(f"   → Yeni:     (N, 128, 17)")
    print(f"📦 y shape: {y.shape}")
    print(f"🏷️  Label map: {label_map}")
    
    # Sınıf dağılımı
    print(f"\n📊 Sınıf dağılımı:")
    reverse_map = {v: k for k, v in label_map.items()}
    for label_idx in sorted(reverse_map.keys()):
        count = np.sum(y == label_idx)
        pct = (count / len(y)) * 100
        print(f"   {reverse_map[label_idx]:10s}: {count:5d} ({pct:.1f}%)")
    
    # Normalizasyon
    print(f"\n🔄 Normalizasyon...")
    original_shape = X.shape
    X_flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_flat).reshape(original_shape)
    print(f"   Mean: {X_normalized.mean():.6f}")
    print(f"   Std:  {X_normalized.std():.6f}")
    
    # Kaydet
    print(f"\n💾 Kaydediliyor...")
    np.save(os.path.join(OUTPUT_DIR, 'X_transformed.npy'), X_normalized)
    np.save(os.path.join(OUTPUT_DIR, 'y_transformed.npy'), y)
    with open(os.path.join(OUTPUT_DIR, 'label_map_transformed.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    
    # Scaler'ı da kaydet
    import pickle
    with open(os.path.join(OUTPUT_DIR, 'scaler_transformed.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"   ✅ X_transformed.npy")
    print(f"   ✅ y_transformed.npy")
    print(f"   ✅ label_map_transformed.json")
    print(f"   ✅ scaler_transformed.pkl")
    
    # FFT verileriyle karşılaştır
    print(f"\n{'='*70}")
    print("📊 FFT vs TRANSFORMED KARŞILAŞTIRMASI")
    print("=" * 70)
    
    try:
        X_fft = np.load(FFT_MODEL_DIR / 'X_fft.npy')
        print(f"   FFT X shape:        {X_fft.shape}")
        print(f"   Transformed X shape: {X.shape}")
        print(f"   Özellik artışı: {X.shape[2] - X_fft.shape[2]} özellik (+{((X.shape[2] / X_fft.shape[2]) - 1) * 100:.0f}%)")
    except:
        print("   (FFT verileri bulunamadı)")
    
    print(f"\n{'='*70}")
    print("✅ TAMAMLANDI!")
    print("=" * 70)
    print("\n📌 Sonraki adım: Bu veriyle yeni model eğit")
    print(f"📌 Komut: python3 train_model_transformed.py")
    print("   python3 train_model_transformed.py")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
