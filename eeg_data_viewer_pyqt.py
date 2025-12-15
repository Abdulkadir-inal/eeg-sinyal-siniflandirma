#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 EEG Data Viewer - OpenViBE Benzeri Veri Görüntüleyici (PyQtGraph)
====================================================================

Kaydedilmiş EEG verilerini OpenViBE benzeri arayüzde gösterir.
PyQtGraph ile yüksek performanslı real-time görüntüleme.

Özellikler:
    • Real-time animasyon (kaydı oynatır)
    • Raw EEG dalga formu (512 Hz)
    • FFT Power Spectrum (8 bant)
    • Event marker'lar
    • Filtreli/Ham sinyal seçimi
    • Hız kontrolü (0.25x - 4x)
    • Zoom in/out
    • Dosya seçimi

Kullanım:
    python3 eeg_data_viewer_pyqt.py
    
Gereksinimler:
    pip install pyqtgraph PyQt5 numpy scipy pandas
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from collections import deque

# PyQt5 ve PyQtGraph
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                  QHBoxLayout, QPushButton, QLabel, QSlider, 
                                  QFileDialog, QComboBox, QCheckBox, QGroupBox,
                                  QSplitter, QFrame, QStatusBar)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont
    import pyqtgraph as pg
except ImportError:
    print("❌ PyQt5 veya PyQtGraph kurulu değil!")
    print("   Kurulum: pip install PyQt5 pyqtgraph")
    sys.exit(1)

# ============================================================================
# AYARLAR
# ============================================================================
SAMPLING_RATE = 512  # Hz
DISPLAY_WINDOW = 4.0  # saniye (ekranda gösterilen süre)
DISPLAY_SAMPLES = int(SAMPLING_RATE * DISPLAY_WINDOW)
UPDATE_INTERVAL = 20  # ms (50 FPS)

# Filtre parametreleri
NOTCH_FREQ = 50
NOTCH_Q = 30
LOWCUT = 0.5
HIGHCUT = 50
FILTER_ORDER = 4

# Renkler
COLORS = {
    'bg': '#1e1e1e',
    'fg': '#ffffff',
    'eeg': '#00ff88',
    'eeg_raw': '#ffaa00',
    'grid': '#333333',
    'bands': ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', 
              '#54a0ff', '#5f27cd', '#00d2d3', '#1dd1a1']
}

# Frekans bantları
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
    
    def filter_signal(self, samples):
        """Sinyal filtrele"""
        if len(samples) < 50:
            return samples
        samples = np.array(samples, dtype=np.float64)
        samples = samples - np.mean(samples)
        try:
            samples = scipy_signal.filtfilt(self.notch_b, self.notch_a, samples)
            samples = scipy_signal.filtfilt(self.bandpass_b, self.bandpass_a, samples)
        except:
            pass
        return samples
    
    def calculate_band_powers(self, samples):
        """FFT ile bant güçlerini hesapla"""
        if len(samples) < 256:
            return {band: 0 for band in FREQUENCY_BANDS}
        
        window = np.hamming(len(samples))
        samples_windowed = samples * window
        fft_vals = np.abs(np.fft.rfft(samples_windowed))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / self.fs)
        power_spectrum = fft_vals ** 2
        
        band_powers = {}
        for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_powers[band_name] = np.sum(power_spectrum[mask])
        
        return band_powers


# ============================================================================
# ANA UYGULAMA
# ============================================================================

class EEGViewer(QMainWindow):
    """EEG Veri Görüntüleyici Ana Pencere"""
    
    def __init__(self):
        super().__init__()
        
        # Veri
        self.data = None
        self.raw_eeg = None
        self.current_pos = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.show_filtered = True
        
        # Signal processor
        self.signal_processor = SignalProcessor()
        
        # Buffer
        self.display_buffer = deque(maxlen=DISPLAY_SAMPLES)
        
        # UI
        self.init_ui()
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        
    def init_ui(self):
        """UI oluştur"""
        self.setWindowTitle("🧠 EEG Data Viewer - OpenViBE Style")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet(f"background-color: {COLORS['bg']}; color: {COLORS['fg']};")
        
        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        
        # ==================== ÜST KONTROL PANELİ ====================
        control_panel = QHBoxLayout()
        
        # Dosya seçimi
        file_group = QGroupBox("📁 Dosya")
        file_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding-top: 10px; }")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("Dosya seçilmedi")
        self.file_label.setStyleSheet("color: #888;")
        file_layout.addWidget(self.file_label)
        
        self.btn_open = QPushButton("📂 Aç")
        self.btn_open.setStyleSheet("QPushButton { background: #4a4a4a; padding: 8px 15px; border-radius: 5px; } QPushButton:hover { background: #5a5a5a; }")
        self.btn_open.clicked.connect(self.open_file)
        file_layout.addWidget(self.btn_open)
        
        control_panel.addWidget(file_group)
        
        # Oynatma kontrolü
        play_group = QGroupBox("▶️ Oynatma")
        play_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding-top: 10px; }")
        play_layout = QHBoxLayout(play_group)
        
        self.btn_play = QPushButton("▶️ Oynat")
        self.btn_play.setStyleSheet("QPushButton { background: #2d7d46; padding: 8px 15px; border-radius: 5px; } QPushButton:hover { background: #3d8d56; }")
        self.btn_play.clicked.connect(self.toggle_play)
        play_layout.addWidget(self.btn_play)
        
        self.btn_stop = QPushButton("⏹️ Dur")
        self.btn_stop.setStyleSheet("QPushButton { background: #7d2d2d; padding: 8px 15px; border-radius: 5px; } QPushButton:hover { background: #8d3d3d; }")
        self.btn_stop.clicked.connect(self.stop_playback)
        play_layout.addWidget(self.btn_stop)
        
        self.btn_reset = QPushButton("⏮️ Başa")
        self.btn_reset.setStyleSheet("QPushButton { background: #4a4a4a; padding: 8px 15px; border-radius: 5px; } QPushButton:hover { background: #5a5a5a; }")
        self.btn_reset.clicked.connect(self.reset_playback)
        play_layout.addWidget(self.btn_reset)
        
        control_panel.addWidget(play_group)
        
        # Hız kontrolü
        speed_group = QGroupBox("⚡ Hız")
        speed_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding-top: 10px; }")
        speed_layout = QHBoxLayout(speed_group)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(16)
        self.speed_slider.setValue(4)
        self.speed_slider.setStyleSheet("QSlider::groove:horizontal { background: #333; height: 8px; border-radius: 4px; } QSlider::handle:horizontal { background: #00ff88; width: 16px; margin: -4px 0; border-radius: 8px; }")
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        self.speed_label.setStyleSheet("color: #00ff88; font-weight: bold;")
        speed_layout.addWidget(self.speed_label)
        
        control_panel.addWidget(speed_group)
        
        # Filtre kontrolü
        filter_group = QGroupBox("🔧 Filtre")
        filter_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding-top: 10px; }")
        filter_layout = QHBoxLayout(filter_group)
        
        self.filter_check = QCheckBox("Filtreli Sinyal")
        self.filter_check.setChecked(True)
        self.filter_check.setStyleSheet("QCheckBox { color: #00ff88; } QCheckBox::indicator { width: 18px; height: 18px; }")
        self.filter_check.stateChanged.connect(self.toggle_filter)
        filter_layout.addWidget(self.filter_check)
        
        control_panel.addWidget(filter_group)
        
        control_panel.addStretch()
        main_layout.addLayout(control_panel)
        
        # ==================== GRAFİK ALANI ====================
        splitter = QSplitter(Qt.Vertical)
        
        # EEG Sinyali Grafiği
        eeg_widget = QWidget()
        eeg_layout = QVBoxLayout(eeg_widget)
        eeg_layout.setContentsMargins(0, 0, 0, 0)
        
        eeg_label = QLabel("📊 Raw EEG Signal (512 Hz)")
        eeg_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ff88; padding: 5px;")
        eeg_layout.addWidget(eeg_label)
        
        # PyQtGraph Plot
        pg.setConfigOptions(antialias=True, background=COLORS['bg'], foreground=COLORS['fg'])
        
        self.eeg_plot = pg.PlotWidget()
        self.eeg_plot.setLabel('left', 'Amplitude', units='µV')
        self.eeg_plot.setLabel('bottom', 'Time', units='s')
        self.eeg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.eeg_plot.setYRange(-500, 500)
        self.eeg_plot.setXRange(0, DISPLAY_WINDOW)
        self.eeg_curve = self.eeg_plot.plot(pen=pg.mkPen(color=COLORS['eeg'], width=1.5))
        eeg_layout.addWidget(self.eeg_plot)
        
        splitter.addWidget(eeg_widget)
        
        # FFT Bant Güçleri Grafiği
        fft_widget = QWidget()
        fft_layout = QVBoxLayout(fft_widget)
        fft_layout.setContentsMargins(0, 0, 0, 0)
        
        fft_label = QLabel("📈 FFT Band Powers")
        fft_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffaa00; padding: 5px;")
        fft_layout.addWidget(fft_label)
        
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setLabel('left', 'Power', units='µV²')
        self.fft_plot.setLabel('bottom', 'Frequency Band')
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Bar graph için
        self.band_names = list(FREQUENCY_BANDS.keys())
        self.fft_bars = pg.BarGraphItem(x=range(len(self.band_names)), height=[0]*len(self.band_names), 
                                         width=0.6, brushes=[pg.mkBrush(c) for c in COLORS['bands']])
        self.fft_plot.addItem(self.fft_bars)
        
        # X ekseni etiketleri
        ax = self.fft_plot.getAxis('bottom')
        ax.setTicks([[(i, name) for i, name in enumerate(self.band_names)]])
        
        fft_layout.addWidget(self.fft_plot)
        
        splitter.addWidget(fft_widget)
        
        # Splitter oranları
        splitter.setSizes([500, 300])
        main_layout.addWidget(splitter)
        
        # ==================== PROGRESS BAR ====================
        progress_layout = QHBoxLayout()
        
        self.progress_label = QLabel("0:00 / 0:00")
        self.progress_label.setStyleSheet("color: #888;")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setStyleSheet("QSlider::groove:horizontal { background: #333; height: 8px; border-radius: 4px; } QSlider::handle:horizontal { background: #00ff88; width: 16px; margin: -4px 0; border-radius: 8px; }")
        self.progress_slider.sliderPressed.connect(self.pause_for_seek)
        self.progress_slider.sliderReleased.connect(self.seek_position)
        progress_layout.addWidget(self.progress_slider)
        
        main_layout.addLayout(progress_layout)
        
        # ==================== STATUS BAR ====================
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("color: #888;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("📂 Bir EEG dosyası açın...")
        
    def open_file(self):
        """Dosya aç dialogu"""
        # WSL'de çalışıyorsak varsayılan dizin
        default_dir = "/home/kadir/sanal-makine/python/proje-veri"
        if not os.path.exists(default_dir):
            default_dir = os.path.expanduser("~")
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "EEG Dosyası Seç", 
            default_dir,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path):
        """Dosyayı yükle"""
        try:
            self.data = pd.read_csv(file_path)
            
            # Electrode kolonunu bul (raw EEG)
            if 'Electrode' in self.data.columns:
                self.raw_eeg = self.data['Electrode'].values
            elif 'electrode' in self.data.columns:
                self.raw_eeg = self.data['electrode'].values
            else:
                # İlk sayısal kolon
                self.raw_eeg = self.data.iloc[:, 2].values
            
            # Float'a çevir
            self.raw_eeg = np.array(self.raw_eeg, dtype=np.float64)
            
            # Dosya bilgisi
            filename = os.path.basename(file_path)
            parts = filename.replace('.csv', '').split('_')
            person = parts[0] if len(parts) > 0 else "?"
            task = parts[1] if len(parts) > 1 else "?"
            
            self.file_label.setText(f"👤 {person} | 🎯 {task}")
            
            # Progress bar ayarla
            self.progress_slider.setMaximum(len(self.raw_eeg))
            
            # Buffer'ı temizle
            self.display_buffer.clear()
            self.current_pos = 0
            
            # Toplam süre
            total_seconds = len(self.raw_eeg) / SAMPLING_RATE
            self.total_time = total_seconds
            
            self.status_bar.showMessage(f"✅ Yüklendi: {filename} | {len(self.raw_eeg):,} örnek | {total_seconds:.1f} saniye")
            
            # İlk veriyi göster
            self.update_display()
            
        except Exception as e:
            self.status_bar.showMessage(f"❌ Hata: {e}")
    
    def toggle_play(self):
        """Oynat/Duraklat"""
        if self.raw_eeg is None:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.btn_play.setText("⏸️ Duraklat")
            self.btn_play.setStyleSheet("QPushButton { background: #7d7d2d; padding: 8px 15px; border-radius: 5px; }")
            self.timer.start(UPDATE_INTERVAL)
        else:
            self.btn_play.setText("▶️ Oynat")
            self.btn_play.setStyleSheet("QPushButton { background: #2d7d46; padding: 8px 15px; border-radius: 5px; }")
            self.timer.stop()
    
    def stop_playback(self):
        """Oynatmayı durdur"""
        self.is_playing = False
        self.btn_play.setText("▶️ Oynat")
        self.btn_play.setStyleSheet("QPushButton { background: #2d7d46; padding: 8px 15px; border-radius: 5px; }")
        self.timer.stop()
    
    def reset_playback(self):
        """Başa sar"""
        self.stop_playback()
        self.current_pos = 0
        self.display_buffer.clear()
        self.update_display()
    
    def update_speed(self, value):
        """Hız güncelle"""
        # 1-16 arası değer -> 0.25x - 4x
        self.playback_speed = value / 4.0
        self.speed_label.setText(f"{self.playback_speed:.2f}x")
    
    def toggle_filter(self, state):
        """Filtre aç/kapa"""
        self.show_filtered = state == Qt.Checked
        self.update_display()
    
    def pause_for_seek(self):
        """Seek için duraklat"""
        self.was_playing = self.is_playing
        self.stop_playback()
    
    def seek_position(self):
        """Yeni pozisyona git"""
        self.current_pos = self.progress_slider.value()
        self.display_buffer.clear()
        
        # Önceki verileri buffer'a ekle
        start = max(0, self.current_pos - DISPLAY_SAMPLES)
        for i in range(start, self.current_pos):
            self.display_buffer.append(self.raw_eeg[i])
        
        self.update_display()
        
        if hasattr(self, 'was_playing') and self.was_playing:
            self.toggle_play()
    
    def update_display(self):
        """Ekranı güncelle"""
        if self.raw_eeg is None:
            return
        
        # Veri al
        samples_per_update = int(SAMPLING_RATE * (UPDATE_INTERVAL / 1000) * self.playback_speed)
        
        for _ in range(samples_per_update):
            if self.current_pos < len(self.raw_eeg):
                self.display_buffer.append(self.raw_eeg[self.current_pos])
                self.current_pos += 1
        
        # Buffer'ı array'e çevir
        if len(self.display_buffer) < 10:
            return
        
        display_data = np.array(self.display_buffer)
        
        # Filtrele
        if self.show_filtered and len(display_data) > 100:
            display_data = self.signal_processor.filter_signal(display_data)
            self.eeg_curve.setPen(pg.mkPen(color=COLORS['eeg'], width=1.5))
        else:
            self.eeg_curve.setPen(pg.mkPen(color=COLORS['eeg_raw'], width=1.5))
        
        # Zaman ekseni
        t = np.linspace(0, len(display_data) / SAMPLING_RATE, len(display_data))
        
        # EEG grafiği güncelle
        self.eeg_curve.setData(t, display_data)
        
        # Y ekseni otomatik ayarla
        y_range = max(abs(display_data.min()), abs(display_data.max())) * 1.1
        if y_range > 10:
            self.eeg_plot.setYRange(-y_range, y_range)
        
        # FFT bant güçleri
        if len(display_data) >= 512:
            fft_data = display_data[-512:]
            band_powers = self.signal_processor.calculate_band_powers(fft_data)
            heights = [np.log1p(band_powers[band]) for band in self.band_names]
            self.fft_bars.setOpts(height=heights)
        
        # Progress güncelle
        self.progress_slider.setValue(self.current_pos)
        
        current_time = self.current_pos / SAMPLING_RATE
        total_time = len(self.raw_eeg) / SAMPLING_RATE
        self.progress_label.setText(f"{int(current_time//60)}:{int(current_time%60):02d} / {int(total_time//60)}:{int(total_time%60):02d}")
        
        # Sonuna gelindi mi?
        if self.current_pos >= len(self.raw_eeg):
            self.stop_playback()
            self.status_bar.showMessage("✅ Oynatma tamamlandı!")


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    # WSL için display ayarı
    if 'microsoft' in os.uname().release.lower():
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
    app = QApplication(sys.argv)
    
    # Font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Ana pencere
    viewer = EEGViewer()
    viewer.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
