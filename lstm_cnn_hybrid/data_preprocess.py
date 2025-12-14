#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM+CNN Hibrit Model için Veri Ön İşleme
==========================================

FFT bant güçleri (proje-veri/*.csv) kullanır.
Özellikler: Delta, Theta, Low Alpha, High Alpha, Low Beta, High Beta, Low Gamma, Mid Gamma
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import pickle

# ============================================================================
# AYARLAR
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Filtrelenmiş FFT verilerinin yolu
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'fft_model', 'data_filtered')
OUTPUT_DIR = SCRIPT_DIR

# FFT bant isimleri
BAND_NAMES = ['Delta', 'Theta', 'Low Alpha', 'High Alpha', 
              'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']

# Sequence uzunluğu (kaç frame bakacak)
SEQUENCE_LENGTH = 64  # ~0.5 saniye (128 Hz varsayımıyla)

# Sınıf eşleştirmeleri
CLASS_MAP = {
    'yukarı': 0,
    'yukari': 0,
    'aşağı': 1,
    'asagı': 1,  # Türkçe i harfi ile
    'asagi': 1,
    'araba': 2
}


def load_csv_files():
    """Tüm CSV dosyalarını yükle"""
    print("\n📂 CSV dosyaları yükleniyor...")
    
    all_data = []
    
    for class_name in ['yukarı', 'asagı', 'araba']:  # asagı (ı harfi ile)
        class_dir = os.path.join(DATA_DIR, class_name)
        
        if not os.path.exists(class_dir):
            # Alternatif isimleri dene
            alt_names = {
                'yukarı': 'yukari',
                'asagı': 'asagi',
                'aşağı': 'asagi'
            }
            if class_name in alt_names:
                class_dir = os.path.join(DATA_DIR, alt_names[class_name])
        
        if not os.path.exists(class_dir):
            print(f"   ⚠️  {class_name} klasörü bulunamadı!")
            continue
        
        csv_files = glob.glob(os.path.join(class_dir, '*.csv'))
        print(f"   📁 {class_name}: {len(csv_files)} dosya")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # FFT bantlarını kontrol et
                missing_bands = [b for b in BAND_NAMES if b not in df.columns]
                if missing_bands:
                    print(f"      ⚠️  {os.path.basename(csv_file)}: Eksik bantlar {missing_bands}")
                    continue
                
                # Sadece FFT bantlarını al
                band_data = df[BAND_NAMES].values
                
                # NaN ve sonsuz değerleri temizle
                band_data = np.nan_to_num(band_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Log transform (daha stabil)
                band_data = np.log1p(np.abs(band_data))
                
                all_data.append({
                    'file': os.path.basename(csv_file),
                    'class': class_name,
                    'label': CLASS_MAP.get(class_name, 0),
                    'data': band_data
                })
                
                print(f"      ✅ {os.path.basename(csv_file)}: {len(band_data)} örnek")
                
            except Exception as e:
                print(f"      ❌ {os.path.basename(csv_file)}: {e}")
    
    return all_data


def create_sequences(all_data):
    """Veriyi sequence'lara ayır"""
    print(f"\n🔄 Sequence'lar oluşturuluyor (uzunluk={SEQUENCE_LENGTH})...")
    
    X_list = []
    y_list = []
    
    for item in all_data:
        data = item['data']
        label = item['label']
        
        # Sliding window ile sequence'lar oluştur
        step = SEQUENCE_LENGTH // 2  # %50 overlap
        
        for i in range(0, len(data) - SEQUENCE_LENGTH, step):
            seq = data[i:i + SEQUENCE_LENGTH]
            X_list.append(seq)
            y_list.append(label)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    print(f"   ✅ X shape: {X.shape}")
    print(f"   ✅ y shape: {y.shape}")
    
    # Sınıf dağılımı
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Sınıf dağılımı:")
    for u, c in zip(unique, counts):
        class_name = [k for k, v in CLASS_MAP.items() if v == u][0]
        print(f"   {class_name}: {c} örnek ({100*c/len(y):.1f}%)")
    
    return X, y


def normalize_data(X):
    """Veriyi normalize et"""
    print("\n📐 Normalizasyon uygulanıyor...")
    
    # Her özellik için ayrı normalize et
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])  # (samples*seq_len, features)
    
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    
    X = X_flat.reshape(original_shape)
    
    print(f"   ✅ Mean: {scaler.mean_}")
    print(f"   ✅ Std: {scaler.scale_}")
    
    return X, scaler


def add_derived_features(X):
    """Türetilmiş özellikler ekle"""
    print("\n🧮 Türetilmiş özellikler ekleniyor...")
    
    # Orijinal: (samples, seq_len, 8)
    # Delta, Theta, Low Alpha, High Alpha, Low Beta, High Beta, Low Gamma, Mid Gamma
    
    delta = X[:, :, 0:1]
    theta = X[:, :, 1:2]
    low_alpha = X[:, :, 2:3]
    high_alpha = X[:, :, 3:4]
    low_beta = X[:, :, 4:5]
    high_beta = X[:, :, 5:6]
    low_gamma = X[:, :, 6:7]
    mid_gamma = X[:, :, 7:8]
    
    # Toplam Alpha ve Beta
    alpha_total = low_alpha + high_alpha
    beta_total = low_beta + high_beta
    gamma_total = low_gamma + mid_gamma
    
    # Oranlar (dikkat için önemli)
    eps = 1e-6
    theta_beta_ratio = theta / (beta_total + eps)
    alpha_beta_ratio = alpha_total / (beta_total + eps)
    theta_alpha_ratio = theta / (alpha_total + eps)
    
    # Engagement index (beta / (alpha + theta))
    engagement = beta_total / (alpha_total + theta + eps)
    
    # Birleştir
    X_extended = np.concatenate([
        X,  # 8 orijinal
        alpha_total,  # +1
        beta_total,   # +1
        gamma_total,  # +1
        theta_beta_ratio,  # +1
        alpha_beta_ratio,  # +1
        theta_alpha_ratio, # +1
        engagement         # +1
    ], axis=-1)
    
    print(f"   ✅ Orijinal: {X.shape[-1]} özellik")
    print(f"   ✅ Genişletilmiş: {X_extended.shape[-1]} özellik")
    
    return X_extended


def save_data(X, y, scaler, label_map):
    """Veriyi kaydet"""
    print("\n💾 Veri kaydediliyor...")
    
    np.save(os.path.join(OUTPUT_DIR, 'X_data.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_data.npy'), y)
    
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    # Config kaydet
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_features': X.shape[-1],
        'num_classes': len(label_map),
        'band_names': BAND_NAMES
    }
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ X_data.npy")
    print(f"   ✅ y_data.npy")
    print(f"   ✅ scaler.pkl")
    print(f"   ✅ label_map.json")
    print(f"   ✅ config.json")


def main():
    print("\n" + "=" * 60)
    print("🧠 LSTM+CNN Hibrit Model - Veri Ön İşleme")
    print("=" * 60)
    
    # 1. CSV dosyalarını yükle
    all_data = load_csv_files()
    
    if not all_data:
        print("\n❌ Veri bulunamadı!")
        return
    
    # 2. Sequence'lar oluştur
    X, y = create_sequences(all_data)
    
    # 3. Türetilmiş özellikler ekle
    X = add_derived_features(X)
    
    # 4. Normalize et
    X, scaler = normalize_data(X)
    
    # 5. Label map
    label_map = {v: k for k, v in CLASS_MAP.items()}
    # Düzelt: 0: yukarı, 1: aşağı, 2: araba
    label_map = {'0': 'yukarı', '1': 'aşağı', '2': 'araba'}
    
    # 6. Kaydet
    save_data(X, y, scaler, label_map)
    
    print("\n" + "=" * 60)
    print("✅ VERİ ÖN İŞLEME TAMAMLANDI!")
    print("=" * 60)
    print(f"\n📊 Özet:")
    print(f"   Toplam örnek: {len(X)}")
    print(f"   Sequence uzunluğu: {SEQUENCE_LENGTH}")
    print(f"   Özellik sayısı: {X.shape[-1]}")
    print(f"   Sınıf sayısı: {len(set(y))}")


if __name__ == "__main__":
    main()
