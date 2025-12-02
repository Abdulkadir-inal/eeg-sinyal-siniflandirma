#!/usr/bin/env python3
"""
Raw EEG'den FFT ile Frekans Bantları Hesaplama (NeuroSky Tarzı)
===============================================================
Bu script mevcut CSV dosyalarındaki Electrode (Raw EEG) sütununu okur,
FFT ile frekans bantlarını hesaplar ve NeuroSky gibi her 512 sample'a
aynı bant değerlerini yazar.

ÖNEMLİ: Event Id sütunundaki START (33025) ve END (33024) işaretleri
korunur, böylece data_preprocess sadece aktif bölgeleri kullanabilir.

Böylece:
- Satır sayısı korunur (512 Hz)
- Model yapısı değişmez
- Event işaretleri korunur
- Sadece bant değerleri bizim FFT hesabımızdan gelir

Kullanım:
    cd fft_model
    python convert_raw_to_fft.py

Giriş: ../proje-veri/ klasöründeki CSV dosyaları
Çıkış: ./data/ klasörüne yeni CSV dosyaları
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Dizin ayarları
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR.parent.parent / "proje-veri"  # ../proje-veri
OUTPUT_DIR = SCRIPT_DIR / "data"  # ./data (fft_model/data)

# FFT ayarları
SAMPLING_RATE = 512  # Hz
WINDOW_SIZE = 512    # 1 saniyelik pencere (512 sample) - FFT için
# Her 512 sample için 1 FFT hesapla, sonucu 512 satıra yaz (NeuroSky gibi)

# Event işaretleri
START_EVENT = 33025
END_EVENT = 33024

# NeuroSky frekans bantları (Hz)
FREQUENCY_BANDS = {
    'Delta': (0.5, 2.75),
    'Theta': (3.5, 6.75),
    'Low Alpha': (7.5, 9.25),
    'High Alpha': (10, 11.75),
    'Low Beta': (13, 16.75),
    'High Beta': (18, 29.75),
    'Low Gamma': (31, 39.75),
    'Mid Gamma': (41, 49.75)
}


def calculate_band_powers(raw_samples):
    """
    Raw EEG verisinden FFT ile frekans bant güçlerini hesapla
    NeuroSky'ın hesaplama yöntemine benzer şekilde
    """
    samples = np.array(raw_samples, dtype=np.float64)
    
    # DC offset'i kaldır
    samples = samples - np.mean(samples)
    
    # Hamming window uygula (spectral leakage azaltmak için)
    window = np.hamming(len(samples))
    samples = samples * window
    
    # FFT hesapla
    fft_vals = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / SAMPLING_RATE)
    
    # Güç spektrumu (magnitude squared)
    power_spectrum = fft_vals ** 2
    
    # Her bant için güç hesapla
    band_powers = {}
    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        # Toplam güç
        band_powers[band_name] = np.sum(power_spectrum[mask])
    
    return band_powers


def process_csv_file(input_path, output_path):
    """
    Tek bir CSV dosyasını işle - NeuroSky tarzı (512 satıra aynı değer)
    Event Id sütununu koruyarak aktif bölgelerin işaretlenmesini sağla
    """
    print(f"  İşleniyor: {input_path.name}")
    
    # CSV'yi oku
    df = pd.read_csv(input_path)
    
    # Electrode sütununu al (Raw EEG)
    if 'Electrode' not in df.columns:
        print(f"    ⚠ 'Electrode' sütunu bulunamadı, atlanıyor.")
        return False
    
    raw_eeg = df['Electrode'].values
    total_samples = len(raw_eeg)
    
    # Event Id sütununu kontrol et ve koru
    has_events = 'Event Id' in df.columns
    if has_events:
        event_ids = df['Event Id'].values
        start_count = np.sum(event_ids == START_EVENT)
        end_count = np.sum(event_ids == END_EVENT)
        print(f"    Toplam sample: {total_samples} ({total_samples/SAMPLING_RATE:.1f} saniye)")
        print(f"    Event işaretleri: {start_count} START, {end_count} END")
    else:
        event_ids = None
        print(f"    Toplam sample: {total_samples} ({total_samples/SAMPLING_RATE:.1f} saniye)")
        print(f"    ⚠ Event Id sütunu yok")
    
    # Sonuç dizileri - orijinal boyutta
    result_bands = {band: np.zeros(total_samples) for band in FREQUENCY_BANDS.keys()}
    result_electrode = raw_eeg.copy()  # Electrode değerlerini koru
    
    # Her 512 sample için FFT hesapla
    window_count = 0
    for start_idx in range(0, total_samples - WINDOW_SIZE + 1, WINDOW_SIZE):
        end_idx = start_idx + WINDOW_SIZE
        window_samples = raw_eeg[start_idx:end_idx]
        
        # Bant güçlerini hesapla
        band_powers = calculate_band_powers(window_samples)
        
        # Bu 512 sample'a aynı değerleri yaz (NeuroSky gibi)
        for band_name, power in band_powers.items():
            result_bands[band_name][start_idx:end_idx] = power
        
        window_count += 1
    
    # Son kısım (512'den az sample kaldıysa) - son hesaplanan değerleri kullan
    remaining_start = window_count * WINDOW_SIZE
    if remaining_start < total_samples:
        # Son 512 sample'dan FFT hesapla
        last_window = raw_eeg[max(0, total_samples - WINDOW_SIZE):total_samples]
        if len(last_window) == WINDOW_SIZE:
            band_powers = calculate_band_powers(last_window)
            for band_name, power in band_powers.items():
                result_bands[band_name][remaining_start:total_samples] = power
    
    # Yeni DataFrame oluştur - Event Id dahil
    new_df = pd.DataFrame({
        'Electrode': result_electrode,
        **result_bands
    })
    
    # Event Id sütununu ekle (varsa)
    if has_events:
        new_df['Event Id'] = event_ids
    
    # Sütun sırasını ayarla
    columns = ['Electrode', 'Delta', 'Theta', 'Low Alpha', 'High Alpha', 
               'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']
    if has_events:
        columns.append('Event Id')
    new_df = new_df[columns]
    
    # Çıktı dizinini oluştur
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV olarak kaydet
    new_df.to_csv(output_path, index=False)
    
    print(f"    ✓ {window_count} FFT hesaplandı → {total_samples} satır korundu")
    
    return True


def main():
    print("=" * 60)
    print("Raw EEG → FFT Bant Dönüştürücü (NeuroSky Tarzı)")
    print("=" * 60)
    print(f"\nGiriş dizini:  {INPUT_DIR}")
    print(f"Çıkış dizini:  {OUTPUT_DIR}")
    print(f"\n📌 Her 512 sample için FFT hesaplanıp")
    print(f"   aynı 512 satıra yazılacak (satır sayısı korunur)\n")
    
    # Giriş dizinini kontrol et
    if not INPUT_DIR.exists():
        print(f"✗ Hata: Giriş dizini bulunamadı: {INPUT_DIR}")
        return
    
    # Kategorileri bul (alt klasörler)
    categories = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    
    if not categories:
        print("✗ Hata: Kategori klasörleri bulunamadı")
        return
    
    print(f"Bulunan kategoriler: {[c.name for c in categories]}\n")
    
    total_files = 0
    processed_files = 0
    
    for category_dir in sorted(categories):
        category_name = category_dir.name
        print(f"\n📁 Kategori: {category_name}")
        print("-" * 40)
        
        # Bu kategorideki CSV dosyalarını bul
        csv_files = list(category_dir.glob("*.csv"))
        
        if not csv_files:
            print("  (CSV dosyası bulunamadı)")
            continue
        
        for csv_file in sorted(csv_files):
            total_files += 1
            
            # Çıktı yolunu oluştur
            output_file = OUTPUT_DIR / category_name / csv_file.name
            
            # Dosyayı işle
            if process_csv_file(csv_file, output_file):
                processed_files += 1
    
    print("\n" + "=" * 60)
    print(f"TAMAMLANDI!")
    print(f"İşlenen dosya: {processed_files}/{total_files}")
    print(f"Çıktı dizini:  {OUTPUT_DIR}")
    print("=" * 60)
    
    # İstatistikler
    print("\n📊 Çıktı dosyalarının satır sayıları:")
    for category_dir in sorted(OUTPUT_DIR.iterdir()):
        if category_dir.is_dir():
            for csv_file in sorted(category_dir.glob("*.csv"))[:1]:  # Her kategoriden 1 dosya
                df = pd.read_csv(csv_file)
                print(f"  {category_dir.name}/{csv_file.name}: {len(df)} satır")


if __name__ == '__main__':
    main()
