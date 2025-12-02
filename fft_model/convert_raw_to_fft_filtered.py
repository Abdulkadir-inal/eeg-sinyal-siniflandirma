#!/usr/bin/env python3
"""
Raw EEG Sinyal Filtreleme ve FFT Hesaplama
==========================================
Bu script NeuroSky'ın chip içinde yaptığı ön-işlemleri taklit eder:
1. Notch Filter (50 Hz) - Elektrik şebekesi gürültüsü
2. Bandpass Filter (0.5-50 Hz) - EEG frekans aralığı  
3. Artifact Removal - Aşırı değerleri temizle
4. FFT ile frekans bantları hesaplama

Kullanım:
    cd fft_model
    python convert_raw_to_fft_filtered.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.ndimage import median_filter

# Dizin ayarları
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR.parent.parent / "proje-veri"
OUTPUT_DIR = SCRIPT_DIR / "data_filtered"

# Sinyal işleme parametreleri
SAMPLING_RATE = 512  # Hz
WINDOW_SIZE = 512    # 1 saniyelik pencere

# Filtre parametreleri
NOTCH_FREQ = 50      # Hz (Türkiye elektrik şebekesi)
NOTCH_Q = 30         # Notch filter kalite faktörü
LOWCUT = 0.5         # Hz (EEG alt frekans)
HIGHCUT = 50         # Hz (EEG üst frekans)
FILTER_ORDER = 4     # Butterworth filter order

# Artifact rejection
ARTIFACT_THRESHOLD = 500  # µV üzeri değerler artifact sayılır
ARTIFACT_WINDOW = 50      # Artifact etrafında temizlenecek sample sayısı

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


def create_notch_filter(freq, fs, Q=30):
    """
    Notch (band-stop) filtre oluştur
    Elektrik şebekesi gürültüsünü (50/60 Hz) temizler
    """
    nyq = fs / 2
    w0 = freq / nyq
    b, a = signal.iirnotch(w0, Q)
    return b, a


def create_bandpass_filter(lowcut, highcut, fs, order=4):
    """
    Butterworth bandpass filtre oluştur
    Sadece EEG frekans aralığını (0.5-50 Hz) geçirir
    """
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def apply_notch_filter(data, fs=SAMPLING_RATE):
    """50 Hz notch filtre uygula"""
    b, a = create_notch_filter(NOTCH_FREQ, fs, NOTCH_Q)
    # İki yönlü filtreleme (sıfır faz kayması)
    filtered = signal.filtfilt(b, a, data)
    return filtered


def apply_bandpass_filter(data, fs=SAMPLING_RATE):
    """Bandpass filtre uygula (0.5-50 Hz)"""
    b, a = create_bandpass_filter(LOWCUT, HIGHCUT, fs, FILTER_ORDER)
    filtered = signal.filtfilt(b, a, data)
    return filtered


def remove_artifacts(data, threshold=ARTIFACT_THRESHOLD, window=ARTIFACT_WINDOW):
    """
    Artifact'ları tespit et ve temizle
    - Eşik değeri aşan noktaları bul
    - Bu noktaların etrafını median ile doldur
    """
    cleaned = data.copy()
    
    # Artifact noktalarını bul
    artifact_mask = np.abs(cleaned) > threshold
    artifact_indices = np.where(artifact_mask)[0]
    
    if len(artifact_indices) == 0:
        return cleaned, 0
    
    # Her artifact etrafındaki bölgeyi işaretle
    extended_mask = np.zeros(len(cleaned), dtype=bool)
    for idx in artifact_indices:
        start = max(0, idx - window)
        end = min(len(cleaned), idx + window)
        extended_mask[start:end] = True
    
    # Artifact bölgelerini median ile doldur
    # Önce tüm veri için median hesapla
    good_data = cleaned[~extended_mask]
    if len(good_data) > 0:
        median_val = np.median(good_data)
        # Artifact bölgelerini interpolasyon ile doldur
        cleaned[extended_mask] = median_val
    
    artifact_count = len(artifact_indices)
    return cleaned, artifact_count


def preprocess_signal(raw_data):
    """
    Tam ön-işleme pipeline'ı:
    1. DC offset kaldır
    2. Artifact temizle
    3. Notch filtre (50 Hz)
    4. Bandpass filtre (0.5-50 Hz)
    """
    # 1. DC offset kaldır
    data = raw_data - np.mean(raw_data)
    
    # 2. Artifact temizle
    data, artifact_count = remove_artifacts(data)
    
    # 3. Notch filtre (50 Hz elektrik gürültüsü)
    data = apply_notch_filter(data)
    
    # 4. Bandpass filtre (0.5-50 Hz EEG bandı)
    data = apply_bandpass_filter(data)
    
    return data, artifact_count


def calculate_band_powers(raw_samples):
    """
    Filtrelenmiş EEG'den FFT ile frekans bant güçlerini hesapla
    """
    samples = np.array(raw_samples, dtype=np.float64)
    
    # Hamming window uygula
    window = np.hamming(len(samples))
    samples = samples * window
    
    # FFT hesapla
    fft_vals = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / SAMPLING_RATE)
    
    # Güç spektrumu
    power_spectrum = fft_vals ** 2
    
    # Her bant için güç hesapla
    band_powers = {}
    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        band_powers[band_name] = np.sum(power_spectrum[mask])
    
    return band_powers


def process_csv_file(input_path, output_path):
    """CSV dosyasını filtrele ve FFT hesapla"""
    print(f"  İşleniyor: {input_path.name}")
    
    df = pd.read_csv(input_path)
    
    if 'Electrode' not in df.columns:
        print(f"    ⚠ 'Electrode' sütunu bulunamadı")
        return False
    
    raw_eeg = df['Electrode'].values
    total_samples = len(raw_eeg)
    
    # Event Id kontrolü
    has_events = 'Event Id' in df.columns
    if has_events:
        event_ids = df['Event Id'].values
        start_count = np.sum(pd.to_numeric(df['Event Id'], errors='coerce').fillna(0) == START_EVENT)
        end_count = np.sum(pd.to_numeric(df['Event Id'], errors='coerce').fillna(0) == END_EVENT)
        print(f"    Toplam: {total_samples} sample, {start_count} START, {end_count} END")
    else:
        event_ids = None
        print(f"    Toplam: {total_samples} sample")
    
    # TÜM VERİYİ FİLTRELE
    print(f"    🔧 Sinyal filtreleniyor...")
    filtered_eeg, total_artifacts = preprocess_signal(raw_eeg)
    print(f"    ✓ {total_artifacts} artifact temizlendi")
    
    # Sonuç dizileri
    result_bands = {band: np.zeros(total_samples) for band in FREQUENCY_BANDS.keys()}
    result_electrode = filtered_eeg.copy()
    
    # Her 512 sample için FFT hesapla
    window_count = 0
    for start_idx in range(0, total_samples - WINDOW_SIZE + 1, WINDOW_SIZE):
        end_idx = start_idx + WINDOW_SIZE
        window_samples = filtered_eeg[start_idx:end_idx]
        
        band_powers = calculate_band_powers(window_samples)
        
        for band_name, power in band_powers.items():
            result_bands[band_name][start_idx:end_idx] = power
        
        window_count += 1
    
    # Son kısım
    remaining_start = window_count * WINDOW_SIZE
    if remaining_start < total_samples:
        last_window = filtered_eeg[max(0, total_samples - WINDOW_SIZE):total_samples]
        if len(last_window) == WINDOW_SIZE:
            band_powers = calculate_band_powers(last_window)
            for band_name, power in band_powers.items():
                result_bands[band_name][remaining_start:total_samples] = power
    
    # DataFrame oluştur
    new_df = pd.DataFrame({
        'Electrode': result_electrode,
        **result_bands
    })
    
    if has_events:
        new_df['Event Id'] = event_ids
    
    columns = ['Electrode', 'Delta', 'Theta', 'Low Alpha', 'High Alpha', 
               'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']
    if has_events:
        columns.append('Event Id')
    new_df = new_df[columns]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_path, index=False)
    
    print(f"    ✓ {window_count} FFT hesaplandı")
    
    return True


def main():
    print("=" * 60)
    print("Raw EEG → Filtreleme → FFT Dönüştürücü")
    print("=" * 60)
    print(f"\n📌 Uygulanan filtreler:")
    print(f"   1. DC offset kaldırma")
    print(f"   2. Artifact removal (>{ARTIFACT_THRESHOLD} µV)")
    print(f"   3. Notch filter ({NOTCH_FREQ} Hz)")
    print(f"   4. Bandpass filter ({LOWCUT}-{HIGHCUT} Hz)")
    print(f"\nGiriş:  {INPUT_DIR}")
    print(f"Çıkış:  {OUTPUT_DIR}\n")
    
    if not INPUT_DIR.exists():
        print(f"✗ Giriş dizini bulunamadı: {INPUT_DIR}")
        return
    
    categories = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    print(f"Kategoriler: {[c.name for c in categories]}\n")
    
    total_files = 0
    processed_files = 0
    
    for category_dir in sorted(categories):
        category_name = category_dir.name
        print(f"\n📁 {category_name}")
        print("-" * 40)
        
        csv_files = list(category_dir.glob("*.csv"))
        
        for csv_file in sorted(csv_files):
            total_files += 1
            output_file = OUTPUT_DIR / category_name / csv_file.name
            if process_csv_file(csv_file, output_file):
                processed_files += 1
    
    print("\n" + "=" * 60)
    print(f"✅ TAMAMLANDI: {processed_files}/{total_files} dosya")
    print(f"📁 Çıktı: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
