#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 EEG Data Viewer - OpenViBE Benzeri Veri Görüntüleyici
========================================================

Kaydedilmiş EEG verilerini OpenViBE benzeri arayüzde gösterir.

Özellikler:
    • Real-time animasyon (kaydı oynatır)
    • Raw EEG dalga formu
    • FFT power spectrum
    • Event marker'lar (sınıf değişimleri)
    • Filtreli/Ham sinyal seçimi
    • Hız kontrolü
    • Dosya seçimi

Kullanım:
    python3 eeg_data_viewer.py
    
Gereksinimler:
    pip install matplotlib numpy scipy pandas
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, CheckButtons
from scipy import signal as scipy_signal
from collections import deque
import tkinter as tk
from tkinter import filedialog

# ============================================================================
# AYARLAR
# ============================================================================
SAMPLING_RATE = 512  # Hz
DISPLAY_WINDOW = 2.0  # saniye (ekranda gösterilen süre)
DISPLAY_SAMPLES = int(SAMPLING_RATE * DISPLAY_WINDOW)

# Filtre parametreleri
NOTCH_FREQ = 50
NOTCH_Q = 30
LOWCUT = 0.5
HIGHCUT = 50
FILTER_ORDER = 4

# FFT parametreleri
FFT_WINDOW = 512
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
# SİNYAL İŞLEME
# ============================================================================

class SignalProcessor:
    """EEG sinyal filtreleme"""
    
    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs
        self.notch_b, self.notch_a = self._create_notch_filter()
        self.bandpass_b, self.bandpass_a = self._create_bandpass_filter()
    
    def _create_notch_filter(self):
        nyq = self.fs / 2
        w0 = NOTCH_FREQ / nyq
        return scipy_signal.iirnotch(w0, NOTCH_Q)
    
    def _create_bandpass_filter(self):
        nyq = self.fs / 2
        low = LOWCUT / nyq
        high = HIGHCUT / nyq
        return scipy_signal.butter(FILTER_ORDER, [low, high], btype='band')
    
    def filter_signal(self, raw_samples):
        """Sinyal filtrele"""
        samples = np.array(raw_samples, dtype=np.float64)
        samples = samples - np.mean(samples)
        samples = scipy_signal.filtfilt(self.notch_b, self.notch_a, samples)
        samples = scipy_signal.filtfilt(self.bandpass_b, self.bandpass_a, samples)
        return samples
    
    def calculate_fft(self, samples):
        """FFT hesapla"""
        window = np.hamming(len(samples))
        samples_windowed = samples * window
        fft_vals = np.abs(np.fft.rfft(samples_windowed))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / self.fs)
        power_spectrum = fft_vals ** 2
        return freqs, power_spectrum


# ============================================================================
# VERİ YÜKLEYICI
# ============================================================================

class EEGDataLoader:
    """CSV dosyasından EEG verisi yükle"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.class_name = None
        self.person_name = None
        self.load_data()
    
    def load_data(self):
        """Veriyi yükle"""
        try:
            # CSV oku
            self.data = pd.read_csv(self.csv_path)
            
            # Dosya adından bilgi çıkar
            filename = os.path.basename(self.csv_path)
            parts = filename.replace('.csv', '').split('_')
            self.person_name = parts[0] if len(parts) > 0 else "Unknown"
            self.class_name = parts[1] if len(parts) > 1 else "Unknown"
            
            print(f"✅ Veri yüklendi: {filename}")
            print(f"   👤 Kişi: {self.person_name}")
            print(f"   🎯 Sınıf: {self.class_name}")
            print(f"   📊 Örnek sayısı: {len(self.data)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return False
    
    def get_raw_eeg(self):
        """Raw EEG kolonunu al"""
        if 'Raw' in self.data.columns:
            return self.data['Raw'].values
        elif 'raw' in self.data.columns:
            return self.data['raw'].values
        else:
            # İlk sayısal kolon
            return self.data.iloc[:, 0].values


# ============================================================================
# VİZÜALİZER
# ============================================================================

class EEGViewer:
    """EEG veri görüntüleyici"""
    
    def __init__(self, data_loader):
        self.loader = data_loader
        self.signal_processor = SignalProcessor()
        
        # Veri
        self.raw_eeg = self.loader.get_raw_eeg()
        self.total_samples = len(self.raw_eeg)
        self.current_index = 0
        
        # Display buffer
        self.display_buffer = deque(maxlen=DISPLAY_SAMPLES)
        
        # Durum
        self.is_playing = False
        self.use_filter = True
        self.playback_speed = 1.0
        
        # Figure oluştur
        self.setup_figure()
        
        # Animation
        self.anim = None
    
    def setup_figure(self):
        """Matplotlib figure'ı oluştur"""
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(f'EEG Viewer - {self.loader.person_name} - {self.loader.class_name}')
        
        # Grid layout
        gs = self.fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # Raw EEG plot (üst, geniş)
        self.ax_eeg = self.fig.add_subplot(gs[0:2, :])
        self.ax_eeg.set_title(f'🧠 EEG Signal - {self.loader.person_name} ({self.loader.class_name})', 
                              fontsize=12, fontweight='bold')
        self.ax_eeg.set_ylabel('Amplitude (µV)', fontsize=10)
        self.ax_eeg.set_xlabel('Time (s)', fontsize=10)
        self.ax_eeg.grid(True, alpha=0.3)
        self.line_eeg, = self.ax_eeg.plot([], [], 'b-', linewidth=1)
        
        # FFT plot (alt sol)
        self.ax_fft = self.fig.add_subplot(gs[2, 0])
        self.ax_fft.set_title('📊 Power Spectrum (FFT)', fontsize=10, fontweight='bold')
        self.ax_fft.set_xlabel('Frequency (Hz)', fontsize=9)
        self.ax_fft.set_ylabel('Power', fontsize=9)
        self.ax_fft.set_xlim(0, 50)
        self.ax_fft.grid(True, alpha=0.3)
        self.line_fft, = self.ax_fft.plot([], [], 'r-', linewidth=1.5)
        
        # Band powers (alt orta)
        self.ax_bands = self.fig.add_subplot(gs[2, 1])
        self.ax_bands.set_title('🎵 Frequency Bands', fontsize=10, fontweight='bold')
        self.ax_bands.set_ylabel('Power', fontsize=9)
        self.ax_bands.set_xlabel('Band', fontsize=9)
        self.band_names = list(FREQUENCY_BANDS.keys())
        self.band_bars = self.ax_bands.bar(range(len(self.band_names)), 
                                            [0]*len(self.band_names), 
                                            color='teal', alpha=0.7)
        self.ax_bands.set_xticks(range(len(self.band_names)))
        self.ax_bands.set_xticklabels([name.replace(' ', '\n') for name in self.band_names], 
                                       fontsize=7, rotation=0)
        
        # Kontrol paneli (alt sağ)
        self.ax_control = self.fig.add_subplot(gs[3, :])
        self.ax_control.axis('off')
        
        # Progress text
        self.progress_text = self.ax_control.text(0.5, 0.9, '', 
                                                    transform=self.ax_control.transAxes,
                                                    fontsize=10, ha='center',
                                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Butonlar
        ax_play = plt.axes([0.15, 0.02, 0.08, 0.04])
        ax_reset = plt.axes([0.25, 0.02, 0.08, 0.04])
        ax_filter = plt.axes([0.75, 0.02, 0.12, 0.04])
        
        self.btn_play = Button(ax_play, '▶ Play')
        self.btn_play.on_clicked(self.toggle_play)
        
        self.btn_reset = Button(ax_reset, '⏮ Reset')
        self.btn_reset.on_clicked(self.reset)
        
        self.check_filter = CheckButtons(ax_filter, ['Filter'], [self.use_filter])
        self.check_filter.on_clicked(self.toggle_filter)
        
        # Speed slider
        ax_speed = plt.axes([0.15, 0.08, 0.7, 0.02])
        self.slider_speed = Slider(ax_speed, 'Speed', 0.1, 5.0, 
                                     valinit=1.0, valstep=0.1)
        self.slider_speed.on_changed(self.update_speed)
    
    def toggle_play(self, event):
        """Play/Pause"""
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('⏸ Pause' if self.is_playing else '▶ Play')
    
    def reset(self, event):
        """Başa sar"""
        self.current_index = 0
        self.display_buffer.clear()
        self.is_playing = False
        self.btn_play.label.set_text('▶ Play')
    
    def toggle_filter(self, label):
        """Filtre aç/kapa"""
        self.use_filter = not self.use_filter
    
    def update_speed(self, val):
        """Hız güncelle"""
        self.playback_speed = val
    
    def update(self, frame):
        """Animation frame güncelleme"""
        if not self.is_playing:
            return
        
        # Hıza göre adım
        step = max(1, int(self.playback_speed * 10))
        
        # Veri ekle
        for _ in range(step):
            if self.current_index >= self.total_samples:
                self.current_index = 0
                self.display_buffer.clear()
                break
            
            self.display_buffer.append(self.raw_eeg[self.current_index])
            self.current_index += 1
        
        # Buffer'da yeterli veri varsa güncelle
        if len(self.display_buffer) > 100:
            # Raw EEG plot
            buffer_array = np.array(self.display_buffer)
            
            # Filtre uygula
            if self.use_filter and len(buffer_array) > 100:
                try:
                    display_signal = self.signal_processor.filter_signal(buffer_array)
                except:
                    display_signal = buffer_array
            else:
                display_signal = buffer_array
            
            # Time axis
            time_axis = np.arange(len(display_signal)) / SAMPLING_RATE
            
            # EEG güncelle
            self.line_eeg.set_data(time_axis, display_signal)
            self.ax_eeg.set_xlim(0, DISPLAY_WINDOW)
            self.ax_eeg.set_ylim(np.min(display_signal) - 100, 
                                  np.max(display_signal) + 100)
            
            # FFT güncelle (son 512 sample)
            if len(buffer_array) >= FFT_WINDOW:
                fft_samples = buffer_array[-FFT_WINDOW:]
                if self.use_filter:
                    try:
                        fft_samples = self.signal_processor.filter_signal(fft_samples)
                    except:
                        pass
                
                freqs, power = self.signal_processor.calculate_fft(fft_samples)
                
                # FFT plot
                mask = freqs <= 50
                self.line_fft.set_data(freqs[mask], power[mask])
                self.ax_fft.set_ylim(0, np.max(power[mask]) * 1.1)
                
                # Band powers
                band_powers = []
                for band_name in self.band_names:
                    low, high = FREQUENCY_BANDS[band_name]
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_power = np.sum(power[band_mask])
                    band_powers.append(band_power)
                
                # Normalize
                max_power = max(band_powers) if max(band_powers) > 0 else 1
                band_powers = [p / max_power for p in band_powers]
                
                for bar, height in zip(self.band_bars, band_powers):
                    bar.set_height(height)
                
                self.ax_bands.set_ylim(0, 1.2)
        
        # Progress
        progress = (self.current_index / self.total_samples) * 100
        time_elapsed = self.current_index / SAMPLING_RATE
        time_total = self.total_samples / SAMPLING_RATE
        
        self.progress_text.set_text(
            f'Progress: {progress:.1f}% | '
            f'Time: {time_elapsed:.1f}s / {time_total:.1f}s | '
            f'Sample: {self.current_index}/{self.total_samples}'
        )
    
    def run(self):
        """Animasyonu başlat"""
        # Interval: daha hızlı update (10ms)
        self.anim = FuncAnimation(self.fig, self.update, interval=10, 
                                   blit=False, cache_frame_data=False)
        plt.show()


# ============================================================================
# DOSYA SEÇİCİ
# ============================================================================

def select_csv_file():
    """Tkinter ile CSV dosyası seç"""
    root = tk.Tk()
    root.withdraw()
    
    # Varsayılan klasör
    default_dir = "/home/kadir/sanal-makine/python/proje-veri"
    
    file_path = filedialog.askopenfilename(
        title="EEG CSV Dosyası Seç",
        initialdir=default_dir if os.path.exists(default_dir) else "~",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("🎬 EEG DATA VIEWER - OpenViBE Benzeri Görüntüleyici")
    print("=" * 60)
    print("📁 CSV dosyası seçin...")
    
    # Dosya seç
    csv_path = select_csv_file()
    
    if not csv_path:
        print("❌ Dosya seçilmedi!")
        return
    
    print(f"\n📂 Seçilen dosya: {os.path.basename(csv_path)}")
    
    # Veri yükle
    loader = EEGDataLoader(csv_path)
    
    if loader.data is None:
        return
    
    # Viewer başlat
    print("\n🎬 Görüntüleyici başlatılıyor...")
    print("\n💡 Kontroller:")
    print("   ▶ Play/Pause - Oynatma kontrolü")
    print("   ⏮ Reset - Başa sar")
    print("   Filter - Sinyal filtreleme aç/kapa")
    print("   Speed - Oynatma hızı (0.1x - 5.0x)")
    print("\n🎯 Kapatmak için pencereyi kapat")
    print("-" * 60)
    
    viewer = EEGViewer(loader)
    viewer.run()


if __name__ == "__main__":
    main()
