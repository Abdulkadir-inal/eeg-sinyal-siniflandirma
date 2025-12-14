#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinyal İşleme Modülü
====================

Raw EEG sinyalinden FFT bant güçleri hesaplar.
Eğitim verileriyle aynı pipeline kullanılır.

Pipeline:
1. DC offset kaldır
2. Artifact removal (>500 µV)
3. Notch filter (50 Hz)
4. Bandpass filter (0.5-50 Hz)
5. FFT ile bant güçleri hesapla
"""

import numpy as np
from scipy import signal
from collections import deque

# ============================================================================
# PARAMETRELER (convert_raw_to_fft_filtered.py ile aynı)
# ============================================================================
SAMPLING_RATE = 512  # Hz
WINDOW_SIZE = 512    # 1 saniyelik FFT penceresi

# Filtre parametreleri
NOTCH_FREQ = 50      # Hz (Türkiye elektrik şebekesi)
NOTCH_Q = 30         # Notch filter kalite faktörü
LOWCUT = 0.5         # Hz (EEG alt frekans)
HIGHCUT = 50         # Hz (EEG üst frekans)
FILTER_ORDER = 4     # Butterworth filter order

# Artifact rejection
ARTIFACT_THRESHOLD = 500  # µV üzeri değerler artifact

# NeuroSky frekans bantları (Hz) - Eğitimle aynı
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

BAND_NAMES = ['Delta', 'Theta', 'Low Alpha', 'High Alpha', 
              'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']


# ============================================================================
# FİLTRE FONKSİYONLARI
# ============================================================================

class SignalProcessor:
    """
    Gerçek zamanlı sinyal işleme sınıfı.
    Eğitim verilerinde kullanılan aynı filtreleri uygular.
    """
    
    def __init__(self, sampling_rate=SAMPLING_RATE):
        self.fs = sampling_rate
        self.window_size = WINDOW_SIZE
        
        # Filtre katsayılarını önceden hesapla
        self._create_filters()
        
        # Raw sample buffer (512 sample = 1 saniye)
        self.raw_buffer = deque(maxlen=WINDOW_SIZE)
        
        # Filtrelenmiş sinyal için buffer
        self.filtered_buffer = deque(maxlen=WINDOW_SIZE)
        
        # Son hesaplanan FFT değerleri
        self.last_fft_values = {band: 0.0 for band in BAND_NAMES}
        
        # İstatistikler
        self.total_samples = 0
        self.artifact_count = 0
    
    def _create_filters(self):
        """Filtre katsayılarını hesapla"""
        nyq = self.fs / 2
        
        # Notch filter (50 Hz)
        w0 = NOTCH_FREQ / nyq
        self.notch_b, self.notch_a = signal.iirnotch(w0, NOTCH_Q)
        
        # Bandpass filter (0.5-50 Hz)
        low = LOWCUT / nyq
        high = HIGHCUT / nyq
        self.bp_b, self.bp_a = signal.butter(FILTER_ORDER, [low, high], btype='band')
    
    def add_sample(self, raw_value):
        """
        Yeni bir raw EEG sample ekle.
        Buffer dolduğunda otomatik FFT hesaplar.
        
        Args:
            raw_value: Raw EEG değeri (µV)
        
        Returns:
            dict or None: FFT bant güçleri (buffer doluysa) veya None
        """
        self.raw_buffer.append(raw_value)
        self.total_samples += 1
        
        # Buffer doldu mu?
        if len(self.raw_buffer) >= self.window_size:
            # FFT hesapla ve döndür
            return self.compute_fft()
        
        return None
    
    def add_samples(self, raw_values):
        """
        Birden fazla sample ekle.
        
        Args:
            raw_values: Raw EEG değerleri listesi
        
        Returns:
            dict or None: Son FFT değerleri
        """
        result = None
        for val in raw_values:
            fft = self.add_sample(val)
            if fft is not None:
                result = fft
        return result
    
    def compute_fft(self):
        """
        Buffer'daki veriyi filtrele ve FFT hesapla.
        Eğitim pipeline'ı ile aynı işlemler.
        
        Returns:
            dict: Bant güçleri {'Delta': value, 'Theta': value, ...}
        """
        if len(self.raw_buffer) < self.window_size:
            return self.last_fft_values
        
        # Buffer'ı numpy array'e çevir
        raw_data = np.array(self.raw_buffer, dtype=np.float64)
        
        # 1. DC offset kaldır
        data = raw_data - np.mean(raw_data)
        
        # 2. Artifact kontrolü (basit versiyon)
        artifact_mask = np.abs(data) > ARTIFACT_THRESHOLD
        if np.any(artifact_mask):
            self.artifact_count += np.sum(artifact_mask)
            # Artifact'ları median ile değiştir
            good_data = data[~artifact_mask]
            if len(good_data) > 0:
                median_val = np.median(good_data)
                data[artifact_mask] = median_val
            else:
                data[artifact_mask] = 0
        
        # 3. Notch filter (50 Hz)
        try:
            data = signal.filtfilt(self.notch_b, self.notch_a, data)
        except:
            pass  # Filtre başarısız olursa devam et
        
        # 4. Bandpass filter (0.5-50 Hz)
        try:
            data = signal.filtfilt(self.bp_b, self.bp_a, data)
        except:
            pass
        
        # 5. FFT hesapla
        band_powers = self._calculate_band_powers(data)
        
        self.last_fft_values = band_powers
        return band_powers
    
    def _calculate_band_powers(self, filtered_samples):
        """
        Filtrelenmiş EEG'den FFT ile bant güçleri hesapla.
        (convert_raw_to_fft_filtered.py ile aynı)
        """
        samples = np.array(filtered_samples, dtype=np.float64)
        
        # Hamming window uygula
        window = np.hamming(len(samples))
        samples = samples * window
        
        # FFT hesapla
        fft_vals = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / self.fs)
        
        # Güç spektrumu
        power_spectrum = fft_vals ** 2
        
        # Her bant için güç hesapla
        band_powers = {}
        for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_powers[band_name] = np.sum(power_spectrum[mask])
        
        return band_powers
    
    def get_last_fft(self):
        """Son hesaplanan FFT değerlerini döndür"""
        return self.last_fft_values.copy()
    
    def reset(self):
        """Buffer'ları temizle"""
        self.raw_buffer.clear()
        self.filtered_buffer.clear()
        self.last_fft_values = {band: 0.0 for band in BAND_NAMES}
        self.total_samples = 0
        self.artifact_count = 0
    
    def is_ready(self):
        """Buffer dolu mu?"""
        return len(self.raw_buffer) >= self.window_size
    
    def get_buffer_progress(self):
        """Buffer doluluk yüzdesi"""
        return len(self.raw_buffer) / self.window_size * 100


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("SignalProcessor Test")
    print("=" * 50)
    
    # Test verisi oluştur (1 saniyelik sinüs dalgası)
    t = np.linspace(0, 1, SAMPLING_RATE)
    # 10 Hz Alpha + 20 Hz Beta + 50 Hz noise
    test_signal = (
        50 * np.sin(2 * np.pi * 10 * t) +  # Alpha
        30 * np.sin(2 * np.pi * 20 * t) +  # Beta
        20 * np.sin(2 * np.pi * 50 * t) +  # 50 Hz noise
        5 * np.random.randn(len(t))         # Random noise
    )
    
    processor = SignalProcessor()
    
    # Sample'ları ekle
    for i, sample in enumerate(test_signal):
        result = processor.add_sample(sample)
        
        if result is not None:
            print(f"\nFFT hesaplandı ({i+1} sample sonra):")
            for band, power in result.items():
                print(f"  {band:12s}: {power:12.2f}")
    
    print(f"\nBuffer durumu: {processor.get_buffer_progress():.1f}%")
    print(f"Toplam sample: {processor.total_samples}")
    print(f"Artifact sayısı: {processor.artifact_count}")
