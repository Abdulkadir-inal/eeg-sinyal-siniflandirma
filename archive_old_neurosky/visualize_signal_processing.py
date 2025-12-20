#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 EEG Sinyal İşleme Görselleştirme
====================================

Orijinal Raw EEG vs Senin FFT Filtreleme Yönteminle İşlenmiş FFT Bantları
Düşünme periyotlarını renkli olarak işaretler
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Türkçe karakterler için
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================================
# SİNYAL İŞLEME PARAMETRELERİ (windows_realtime_fft.py ile aynı)
# ============================================================================

SAMPLING_RATE = 512  # Hz
FFT_WINDOW_SIZE = 512  # 1 saniyelik FFT penceresi

# Filtre parametreleri
NOTCH_FREQ = 50      # Hz (Türkiye elektrik şebekesi)
NOTCH_Q = 30
LOWCUT = 0.5         # Hz
HIGHCUT = 50         # Hz
FILTER_ORDER = 4

# Artifact rejection
ARTIFACT_THRESHOLD = 500  # µV

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


# ============================================================================
# SİNYAL İŞLEME SINIFI
# ============================================================================

class SignalProcessor:
    """EEG sinyal işleme"""
    
    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs
        self.notch_b, self.notch_a = self._create_notch_filter()
        self.bandpass_b, self.bandpass_a = self._create_bandpass_filter()
    
    def _create_notch_filter(self):
        """50 Hz Notch filtre"""
        nyq = self.fs / 2
        w0 = NOTCH_FREQ / nyq
        return scipy_signal.iirnotch(w0, NOTCH_Q)
    
    def _create_bandpass_filter(self):
        """Bandpass filtre (0.5-50 Hz)"""
        nyq = self.fs / 2
        low = LOWCUT / nyq
        high = HIGHCUT / nyq
        return scipy_signal.butter(FILTER_ORDER, [low, high], btype='band')
    
    def filter_signal(self, raw_samples):
        """Raw EEG → Filtreli EEG"""
        samples = np.array(raw_samples, dtype=np.float64)
        
        # 1. DC offset kaldır
        samples = samples - np.mean(samples)
        
        # 2. Artifact'ları temizle
        artifact_mask = np.abs(samples) > ARTIFACT_THRESHOLD
        if np.any(artifact_mask):
            good_samples = samples[~artifact_mask]
            if len(good_samples) > 0:
                median_val = np.median(good_samples)
                samples[artifact_mask] = median_val
        
        # 3. Notch filtre (50 Hz)
        samples = scipy_signal.filtfilt(self.notch_b, self.notch_a, samples)
        
        # 4. Bandpass filtre (0.5-50 Hz)
        samples = scipy_signal.filtfilt(self.bandpass_b, self.bandpass_a, samples)
        
        return samples
    
    def calculate_fft_bands(self, filtered_samples):
        """FFT ile frekans bant güçleri"""
        samples = np.array(filtered_samples, dtype=np.float64)
        
        # Hamming window
        window = np.hamming(len(samples))
        samples = samples * window
        
        # FFT
        fft_vals = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / self.fs)
        
        # Güç spektrumu
        power_spectrum = fft_vals ** 2
        
        # Her bant için güç
        band_powers = {}
        for band_name in ['Delta', 'Theta', 'Low Alpha', 'High Alpha', 
                          'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']:
            low_freq, high_freq = FREQUENCY_BANDS[band_name]
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_powers[band_name] = np.sum(power_spectrum[mask])
        
        return band_powers, freqs, power_spectrum


# ============================================================================
# VERİ YÜKLEME VE İŞARETLEME
# ============================================================================

def load_and_mark_periods(csv_path, duration_seconds=30):
    """
    CSV'den veri yükle ve düşünme periyotlarını işaretle
    
    Gerçek NeuroSky formatı:
    - Header: Time:512Hz,Epoch,Electrode,Attention,Meditation,Delta,Theta,Low Alpha,High Alpha,Low Beta,High Beta,Low Gamma,Mid Gamma,Event Id,Event Date,Event Duration
    - Her satır: ~512 Hz sampling rate (0.00195 sn = ~1.95ms)
    - Electrode: Raw EEG değeri
    - FFT bantları: Delta, Theta, Low Alpha, High Alpha, Low Beta, High Beta, Low Gamma, Mid Gamma
    """
    
    print(f"📂 Yükleniyor: {csv_path}")
    
    # CSV'yi oku (header var)
    df = pd.read_csv(csv_path)
    
    # Kolon isimlerini temizle
    df.columns = df.columns.str.strip()
    
    print(f"📋 Kolonlar: {list(df.columns)}")
    print(f"✅ Toplam satır: {len(df)}")
    
    # İlk N saniyeyi al (her satır ~0.00195 sn = 512 Hz)
    # duration_seconds * 512 satır
    num_rows = min(int(duration_seconds * 512), len(df))
    df_segment = df.iloc[:num_rows]
    
    print(f"📊 Seçilen segment: {num_rows} satır (~{num_rows/512:.1f} saniye)")
    
    # Raw EEG (Electrode kolonu)
    raw_segment = df_segment['Electrode'].values
    
    # FFT bantları
    fft_bands = {
        'Delta': df_segment['Delta'].values,
        'Theta': df_segment['Theta'].values,
        'Low Alpha': df_segment['Low Alpha'].values,
        'High Alpha': df_segment['High Alpha'].values,
        'Low Beta': df_segment['Low Beta'].values,
        'High Beta': df_segment['High Beta'].values,
        'Low Gamma': df_segment['Low Gamma'].values,
        'Mid Gamma': df_segment['Mid Gamma'].values
    }
    
    # Zaman ekseni
    time_axis = df_segment['Time:512Hz'].values
    
    # Düşünme periyotlarını otomatik tespit et (Attention değerinden)
    # Attention > 60 → Düşünme
    attention = df_segment['Attention'].values
    thinking_periods = []
    
    in_thinking = False
    think_start = 0
    
    for i, att in enumerate(attention):
        t = time_axis[i]
        if att > 60 and not in_thinking:
            # Düşünme başladı
            in_thinking = True
            think_start = t
        elif att <= 60 and in_thinking:
            # Düşünme bitti
            in_thinking = False
            thinking_periods.append((think_start, t))
    
    # Son periyot hala açıksa kapat
    if in_thinking:
        thinking_periods.append((think_start, time_axis[-1]))
    
    # Eğer hiç düşünme periyodu yoksa, varsayılan patern kullan
    if len(thinking_periods) == 0:
        print("⚠️ Attention'dan düşünme periyodu tespit edilemedi, varsayılan patern kullanılıyor")
        current_time = 0
        max_time = time_axis[-1]
        while current_time < max_time:
            think_start = current_time + 3
            think_end = min(current_time + 8, max_time)
            if think_start < max_time:
                thinking_periods.append((think_start, think_end))
            current_time = think_end + 3
    
    return raw_segment, time_axis, thinking_periods, fft_bands


# ============================================================================
# GÖRSELLEŞTİRME
# ============================================================================

def visualize_signal_processing(raw_segment, time_axis, thinking_periods, fft_bands_original):
    """
    Raw EEG ve işlenmiş sinyali görselleştir
    """
    
    processor = SignalProcessor()
    
    # Filtreleme
    print("\n🔧 Sinyal filtreleniyor...")
    filtered_segment = processor.filter_signal(raw_segment)
    
    # FFT (her 1 saniyelik pencerede - kendi hesaplamamız)
    print("📊 FFT hesaplanıyor (kendi hesaplamamız)...")
    fft_times = []
    fft_band_powers = {band: [] for band in FREQUENCY_BANDS.keys()}
    
    window_size = FFT_WINDOW_SIZE
    step = window_size  # Her 1 saniye
    
    for i in range(0, len(filtered_segment) - window_size, step):
        window = filtered_segment[i:i+window_size]
        band_powers, _, _ = processor.calculate_fft_bands(window)
        
        fft_time = (i + window_size/2) / SAMPLING_RATE  # Pencerenin ortası
        fft_times.append(fft_time)
        
        for band_name, power in band_powers.items():
            fft_band_powers[band_name].append(power)
    
    # Görselleştirme
    print("🎨 Görselleştirme oluşturuluyor...")
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 14))
    fig.suptitle('EEG Sinyal İşleme Görselleştirmesi', fontsize=16, fontweight='bold')
    
    # Düşünme periyotlarını tüm grafiklerde göster
    for ax in axes:
        for think_start, think_end in thinking_periods:
            ax.axvspan(think_start, think_end, alpha=0.2, color='green', label='Düşünme' if ax == axes[0] else '')
    
    # 1. Raw EEG
    axes[0].plot(time_axis, raw_segment, color='blue', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Amplitüd (µV)', fontweight='bold')
    axes[0].set_title('1. Raw EEG (512 Hz) - Orijinal', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[0].set_xlim([time_axis[0], time_axis[-1]])
    
    # 2. Filtreli EEG
    axes[1].plot(time_axis, filtered_segment, color='red', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('Amplitüd (µV)', fontweight='bold')
    axes[1].set_title('2. Filtreli EEG (Notch 50Hz + Bandpass 0.5-50Hz) - İşlenmiş', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([time_axis[0], time_axis[-1]])
    
    # 3. Orijinal FFT Bantları (NeuroSky'dan gelen)
    colors_orig = ['purple', 'blue', 'cyan', 'green', 'orange', 'red', 'darkred', 'brown']
    for idx, (band_name, powers) in enumerate(fft_bands_original.items()):
        axes[2].plot(time_axis, powers, label=band_name, linewidth=1.5, alpha=0.8, color=colors_orig[idx])
    axes[2].set_ylabel('Güç (µV²)', fontweight='bold')
    axes[2].set_title('3. Orijinal FFT Bantları (NeuroSky\'dan)', fontweight='bold')
    axes[2].legend(loc='upper right', ncol=4, fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([time_axis[0], time_axis[-1]])
    
    # 4. Bizim hesapladığımız FFT Bantları (Low Freq)
    colors_low = ['purple', 'blue', 'cyan', 'green']
    for idx, band_name in enumerate(['Delta', 'Theta', 'Low Alpha', 'High Alpha']):
        axes[3].plot(fft_times, fft_band_powers[band_name], 
                    label=band_name, linewidth=2, color=colors_low[idx])
    axes[3].set_ylabel('Güç (µV²)', fontweight='bold')
    axes[3].set_title('4. Bizim FFT - Düşük Frekans (Filtreli EEG\'den)', fontweight='bold')
    axes[3].legend(loc='upper right', ncol=2)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim([time_axis[0], time_axis[-1]])
    
    # 5. Bizim hesapladığımız FFT Bantları (High Freq)
    colors_high = ['orange', 'red', 'darkred', 'brown']
    for idx, band_name in enumerate(['Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']):
        axes[4].plot(fft_times, fft_band_powers[band_name], 
                    label=band_name, linewidth=2, color=colors_high[idx])
    axes[4].set_xlabel('Zaman (saniye)', fontweight='bold')
    axes[4].set_ylabel('Güç (µV²)', fontweight='bold')
    axes[4].set_title('5. Bizim FFT - Yüksek Frekans (Filtreli EEG\'den)', fontweight='bold')
    axes[4].legend(loc='upper right', ncol=2)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xlim([time_axis[0], time_axis[-1]])
    
    plt.tight_layout()
    
    # Kaydet
    output_path = 'signal_processing_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Görselleştirme kaydedildi: {output_path}")
    
    plt.show()
    
    # İstatistikler
    print("\n" + "=" * 60)
    print("📊 İSTATİSTİKLER")
    print("=" * 60)
    print(f"Raw EEG:")
    print(f"  Ortalama: {np.mean(raw_segment):.2f} µV")
    print(f"  Std: {np.std(raw_segment):.2f} µV")
    print(f"  Min: {np.min(raw_segment):.2f} µV")
    print(f"  Max: {np.max(raw_segment):.2f} µV")
    print()
    print(f"Filtreli EEG:")
    print(f"  Ortalama: {np.mean(filtered_segment):.2f} µV")
    print(f"  Std: {np.std(filtered_segment):.2f} µV")
    print(f"  Min: {np.min(filtered_segment):.2f} µV")
    print(f"  Max: {np.max(filtered_segment):.2f} µV")
    print()
    print("Düşünme Periyotları:")
    for i, (start, end) in enumerate(thinking_periods, 1):
        print(f"  {i}. {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("🎨 EEG Sinyal İşleme Görselleştirme")
    print("=" * 60)
    
    # Veri yolu
    csv_path = '../proje-veri/asagı/apo_asagı.csv'
    
    # Veri yükle (30 saniye)
    raw_segment, time_axis, thinking_periods, fft_bands_original = load_and_mark_periods(csv_path, duration_seconds=30)
    
    # Görselleştir
    visualize_signal_processing(raw_segment, time_axis, thinking_periods, fft_bands_original)


if __name__ == "__main__":
    main()
