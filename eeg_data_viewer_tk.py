#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 EEG Data Viewer - OpenViBE Benzeri Veri Görüntüleyici
========================================================

Kaydedilmiş EEG verilerini OpenViBE benzeri arayüzde gösterir.
Tkinter + Matplotlib ile WSL uyumlu.

Veri Kaynağı: proje-veri/ klasöründeki CSV dosyaları
    - Electrode: Raw EEG sinyali (512 Hz)
    - Delta, Theta, Alpha, Beta, Gamma: FFT bant güçleri

Kullanım:
    python3 eeg_data_viewer_tk.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

# ============================================================================
# AYARLAR
# ============================================================================
SAMPLING_RATE = 512  # Hz
DISPLAY_WINDOW = 4.0  # saniye
DISPLAY_SAMPLES = int(SAMPLING_RATE * DISPLAY_WINDOW)
UPDATE_INTERVAL = 50  # ms

# Filtre parametreleri
NOTCH_FREQ = 50
NOTCH_Q = 30
LOWCUT = 0.5
HIGHCUT = 50
FILTER_ORDER = 4

# Renkler (OpenViBE benzeri)
COLORS = {
    'bg': '#1a1a2e',
    'panel': '#16213e',
    'accent': '#0f3460',
    'text': '#e94560',
    'eeg': '#00ff88',
    'eeg_raw': '#ffaa00',
    'grid': '#333355'
}

BAND_COLORS = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', 
               '#54a0ff', '#5f27cd', '#00d2d3', '#1dd1a1']

BAND_NAMES = ['Delta', 'Theta', 'Low Alpha', 'High Alpha', 
              'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']

# ============================================================================
# SİNYAL İŞLEME
# ============================================================================

class SignalProcessor:
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
        """Sinyal filtreleme - kenar efektlerini minimize et"""
        if len(samples) < 50:
            return samples
        samples = np.array(samples, dtype=np.float64)
        
        # 1. Linear detrend (düzgün baseline çıkarma - mean'den daha iyi)
        samples = scipy_signal.detrend(samples, type='linear')
        
        # 2. Padding ekle (kenar efektlerini azaltır)
        pad_len = min(len(samples) // 4, 256)  # Max 256 örnek padding
        
        # Reflect padding (ayna yansıması)
        padded = np.pad(samples, pad_len, mode='reflect')
        
        try:
            # Notch filtre (50 Hz)
            filtered = scipy_signal.filtfilt(self.notch_b, self.notch_a, padded)
            # Bandpass filtre (0.5-50 Hz)
            filtered = scipy_signal.filtfilt(self.bandpass_b, self.bandpass_a, filtered)
            # Padding'i kaldır
            filtered = filtered[pad_len:-pad_len]
        except:
            filtered = samples
        
        return filtered


# ============================================================================
# ANA UYGULAMA
# ============================================================================

class EEGViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 EEG Data Viewer - OpenViBE Style")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['bg'])
        
        # Veri
        self.data = None
        self.raw_eeg = None
        self.filtered_eeg = None  # Önceden filtrelenımiş EEG
        self.band_powers = {}
        self.current_pos = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.show_filtered = True
        
        # Event marker'ları (START=33025, END=33024)
        self.events = []  # [(sample_index, event_type), ...]
        self.event_lines = []  # Grafikteki çizgiler
        
        # Y ekseni sabit aralık
        self.y_range_fixed = 300  # ±300 µV
        
        # Signal processor
        self.processor = SignalProcessor()
        
        # Buffer
        self.display_buffer = deque(maxlen=DISPLAY_SAMPLES)
        
        # UI oluştur
        self.create_ui()
        
        # Timer
        self.update_id = None
        
    def create_ui(self):
        """UI oluştur"""
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 10), padding=5)
        style.configure('TLabel', font=('Segoe UI', 10), background=COLORS['bg'], foreground='white')
        style.configure('TFrame', background=COLORS['bg'])
        style.configure('TCheckbutton', background=COLORS['bg'], foreground='white')
        
        # Ana frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ==================== ÜST KONTROL PANELİ ====================
        control_frame = tk.Frame(main_frame, bg=COLORS['panel'], height=80)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        control_frame.pack_propagate(False)
        
        # Dosya grubu
        file_frame = tk.Frame(control_frame, bg=COLORS['panel'])
        file_frame.pack(side=tk.LEFT, padx=20, pady=15)
        
        tk.Label(file_frame, text="📁 Dosya:", bg=COLORS['panel'], fg='white', 
                 font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT)
        
        self.file_label = tk.Label(file_frame, text="Dosya seçilmedi", bg=COLORS['panel'], 
                                    fg='#888', font=('Segoe UI', 10))
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        self.btn_open = tk.Button(file_frame, text="📂 Aç", command=self.open_file,
                                   bg='#4a4a6a', fg='white', font=('Segoe UI', 10),
                                   relief=tk.FLAT, padx=15, pady=5)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        # Oynatma grubu
        play_frame = tk.Frame(control_frame, bg=COLORS['panel'])
        play_frame.pack(side=tk.LEFT, padx=20, pady=15)
        
        tk.Label(play_frame, text="▶️ Oynatma:", bg=COLORS['panel'], fg='white',
                 font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT)
        
        self.btn_play = tk.Button(play_frame, text="▶️ Oynat", command=self.toggle_play,
                                   bg='#2d7d46', fg='white', font=('Segoe UI', 10),
                                   relief=tk.FLAT, padx=15, pady=5)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(play_frame, text="⏹️ Dur", command=self.stop_playback,
                                   bg='#7d2d2d', fg='white', font=('Segoe UI', 10),
                                   relief=tk.FLAT, padx=15, pady=5)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.btn_reset = tk.Button(play_frame, text="⏮️ Başa", command=self.reset_playback,
                                    bg='#4a4a6a', fg='white', font=('Segoe UI', 10),
                                    relief=tk.FLAT, padx=15, pady=5)
        self.btn_reset.pack(side=tk.LEFT, padx=5)
        
        # Hız grubu
        speed_frame = tk.Frame(control_frame, bg=COLORS['panel'])
        speed_frame.pack(side=tk.LEFT, padx=20, pady=15)
        
        tk.Label(speed_frame, text="⚡ Hız:", bg=COLORS['panel'], fg='white',
                 font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT)
        
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = tk.Scale(speed_frame, from_=0.25, to=4.0, resolution=0.25,
                                     orient=tk.HORIZONTAL, variable=self.speed_var,
                                     bg=COLORS['panel'], fg='#00ff88', 
                                     highlightthickness=0, length=120,
                                     command=self.update_speed)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        self.speed_label = tk.Label(speed_frame, text="1.0x", bg=COLORS['panel'], 
                                     fg='#00ff88', font=('Segoe UI', 11, 'bold'))
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        # Filtre grubu
        filter_frame = tk.Frame(control_frame, bg=COLORS['panel'])
        filter_frame.pack(side=tk.LEFT, padx=20, pady=15)
        
        self.filter_var = tk.BooleanVar(value=True)
        self.filter_check = tk.Checkbutton(filter_frame, text="🔧 Filtreli Sinyal",
                                            variable=self.filter_var, 
                                            bg=COLORS['panel'], fg='#00ff88',
                                            selectcolor=COLORS['accent'],
                                            font=('Segoe UI', 10),
                                            command=self.toggle_filter)
        self.filter_check.pack(side=tk.LEFT)
        
        # Y ekseni aralık kontrolü
        y_range_frame = tk.Frame(control_frame, bg=COLORS['panel'])
        y_range_frame.pack(side=tk.LEFT, padx=20, pady=15)
        
        tk.Label(y_range_frame, text="📏 Y Aralık:", bg=COLORS['panel'], fg='white',
                 font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT)
        
        self.y_range_var = tk.IntVar(value=300)
        self.y_range_scale = tk.Scale(y_range_frame, from_=50, to=1000, resolution=50,
                                        orient=tk.HORIZONTAL, variable=self.y_range_var,
                                        bg=COLORS['panel'], fg='#ffaa00',
                                        highlightthickness=0, length=100,
                                        command=self.update_y_range)
        self.y_range_scale.pack(side=tk.LEFT, padx=5)
        
        self.y_range_label = tk.Label(y_range_frame, text="±300 µV", bg=COLORS['panel'],
                                       fg='#ffaa00', font=('Segoe UI', 10, 'bold'))
        self.y_range_label.pack(side=tk.LEFT, padx=5)
        
        # ==================== GRAFİK ALANI ====================
        graph_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(14, 8), facecolor=COLORS['bg'])
        self.fig.subplots_adjust(hspace=0.3, left=0.06, right=0.98, top=0.95, bottom=0.08)
        
        # EEG grafiği (üst)
        self.ax_eeg = self.fig.add_subplot(211)
        self.ax_eeg.set_facecolor(COLORS['bg'])
        self.ax_eeg.set_title('📊 Raw EEG Signal (512 Hz)', color='#00ff88', fontsize=12, fontweight='bold')
        self.ax_eeg.set_xlabel('Time (s)', color='white')
        self.ax_eeg.set_ylabel('Amplitude (µV)', color='white')
        self.ax_eeg.tick_params(colors='white')
        self.ax_eeg.grid(True, alpha=0.2, color=COLORS['grid'])
        self.ax_eeg.set_xlim(0, DISPLAY_WINDOW)
        self.ax_eeg.set_ylim(-300, 300)
        for spine in self.ax_eeg.spines.values():
            spine.set_color('#444')
        
        self.eeg_line, = self.ax_eeg.plot([], [], color=COLORS['eeg'], linewidth=1)
        
        # Event marker lejantı
        self.ax_eeg.axhline(y=0, color='#444', linewidth=0.5, alpha=0.5)  # Sıfır çizgisi
        # Lejant için dummy çizgiler
        self.ax_eeg.plot([], [], color='#00ff00', linewidth=2, label='▶ START (33025)')
        self.ax_eeg.plot([], [], color='#ff0000', linewidth=2, label='⏹ END (33024)')
        self.ax_eeg.legend(loc='upper right', fontsize=8, facecolor=COLORS['bg'], 
                           edgecolor='#444', labelcolor='white')
        
        # FFT Bar grafiği (alt)
        self.ax_fft = self.fig.add_subplot(212)
        self.ax_fft.set_facecolor(COLORS['bg'])
        self.ax_fft.set_title('📈 FFT Band Powers', color='#ffaa00', fontsize=12, fontweight='bold')
        self.ax_fft.set_xlabel('Frequency Band', color='white')
        self.ax_fft.set_ylabel('Power (log)', color='white')
        self.ax_fft.tick_params(colors='white')
        self.ax_fft.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
        for spine in self.ax_fft.spines.values():
            spine.set_color('#444')
        
        # Bar chart
        x_pos = np.arange(len(BAND_NAMES))
        self.fft_bars = self.ax_fft.bar(x_pos, [0]*len(BAND_NAMES), color=BAND_COLORS, width=0.6)
        self.ax_fft.set_xticks(x_pos)
        self.ax_fft.set_xticklabels(BAND_NAMES, rotation=30, ha='right', fontsize=9)
        self.ax_fft.set_ylim(0, 20)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # ==================== PROGRESS BAR ====================
        progress_frame = tk.Frame(main_frame, bg=COLORS['panel'], height=50)
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        progress_frame.pack_propagate(False)
        
        self.progress_label = tk.Label(progress_frame, text="0:00 / 0:00", 
                                        bg=COLORS['panel'], fg='white',
                                        font=('Segoe UI', 10))
        self.progress_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_scale = tk.Scale(progress_frame, from_=0, to=100,
                                        orient=tk.HORIZONTAL, variable=self.progress_var,
                                        bg=COLORS['panel'], fg='#00ff88',
                                        highlightthickness=0, showvalue=False,
                                        command=self.seek_position)
        self.progress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20, pady=10)
        
        # ==================== STATUS BAR ====================
        self.status_label = tk.Label(main_frame, text="📂 Bir EEG dosyası açın...", 
                                      bg=COLORS['bg'], fg='#888',
                                      font=('Segoe UI', 10), anchor='w')
        self.status_label.pack(fill=tk.X, pady=(10, 0))
        
    def open_file(self):
        """Dosya aç"""
        default_dir = "/home/kadir/sanal-makine/python/proje-veri"
        if not os.path.exists(default_dir):
            default_dir = os.path.expanduser("~")
        
        file_path = filedialog.askopenfilename(
            title="EEG Dosyası Seç",
            initialdir=default_dir,
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path):
        """Dosyayı yükle"""
        try:
            self.data = pd.read_csv(file_path)
            
            # Electrode kolonunu bul (raw EEG)
            if 'Electrode' in self.data.columns:
                self.raw_eeg = self.data['Electrode'].values.astype(np.float64)
            else:
                self.raw_eeg = self.data.iloc[:, 2].values.astype(np.float64)
            
            # Filtrelenmiş EEG'yi ÖNCEDEN hesapla (titreme olmaz)
            print("   🔧 Tüm sinyal filtreleniyor...")
            self.filtered_eeg = self.processor.filter_signal(self.raw_eeg.copy())
            print("   ✅ Filtreleme tamamlandı!")
            
            # FFT bant güçleri (varsa)
            for band in BAND_NAMES:
                if band in self.data.columns:
                    self.band_powers[band] = self.data[band].values.astype(np.float64)
            
            # Event marker'ları oku (START=33025, END=33024)
            self.events = []
            if 'Event Id' in self.data.columns:
                event_col = self.data['Event Id'].values
                for i, event_id in enumerate(event_col):
                    try:
                        eid = int(float(event_id))
                        if eid == 33025:  # START
                            self.events.append((i, 'START'))
                        elif eid == 33024:  # END
                            self.events.append((i, 'END'))
                    except (ValueError, TypeError):
                        continue
                print(f"   📍 {len(self.events)} event bulundu (START/END)")
            
            # Dosya bilgisi
            filename = os.path.basename(file_path)
            parts = filename.replace('.csv', '').split('_')
            person = parts[0] if len(parts) > 0 else "?"
            task = parts[1] if len(parts) > 1 else "?"
            
            self.file_label.config(text=f"👤 {person} | 🎯 {task}", fg='#00ff88')
            
            # Progress bar ayarla
            self.progress_scale.config(to=len(self.raw_eeg))
            
            # Buffer'ı temizle
            self.display_buffer.clear()
            self.current_pos = 0
            
            # Toplam süre
            total_seconds = len(self.raw_eeg) / SAMPLING_RATE
            
            self.status_label.config(text=f"✅ Yüklendi: {filename} | {len(self.raw_eeg):,} örnek | {total_seconds:.1f} saniye")
            
            # İlk veriyi göster
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya yüklenemedi: {e}")
    
    def toggle_play(self):
        """Oynat/Duraklat"""
        if self.raw_eeg is None:
            messagebox.showwarning("Uyarı", "Önce bir dosya açın!")
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.btn_play.config(text="⏸️ Duraklat", bg='#7d7d2d')
            self.animate()
        else:
            self.btn_play.config(text="▶️ Oynat", bg='#2d7d46')
            if self.update_id:
                self.root.after_cancel(self.update_id)
    
    def stop_playback(self):
        """Dur"""
        self.is_playing = False
        self.btn_play.config(text="▶️ Oynat", bg='#2d7d46')
        if self.update_id:
            self.root.after_cancel(self.update_id)
    
    def reset_playback(self):
        """Başa sar"""
        self.stop_playback()
        self.current_pos = 0
        self.display_buffer.clear()
        self.update_display()
    
    def update_speed(self, val):
        """Hız güncelle"""
        self.playback_speed = float(val)
        self.speed_label.config(text=f"{self.playback_speed:.2f}x")
    
    def toggle_filter(self):
        """Filtre toggle"""
        self.show_filtered = self.filter_var.get()
        self.update_display()
    
    def update_y_range(self, val):
        """Y ekseni aralığını güncelle"""
        self.y_range_fixed = int(float(val))
        self.y_range_label.config(text=f"±{self.y_range_fixed} µV")
        self.ax_eeg.set_ylim(-self.y_range_fixed, self.y_range_fixed)
        self.canvas.draw_idle()
    
    def seek_position(self, val):
        """Pozisyon değiştir"""
        if self.raw_eeg is None:
            return
        
        self.current_pos = int(float(val))
        self.update_display()
    
    def animate(self):
        """Animasyon döngüsü"""
        if not self.is_playing:
            return
        
        self.update_display()
        
        # Bir sonraki frame
        if self.current_pos < len(self.raw_eeg):
            self.update_id = self.root.after(UPDATE_INTERVAL, self.animate)
        else:
            self.stop_playback()
            self.status_label.config(text="✅ Oynatma tamamlandı!")
    
    def update_display(self):
        """Ekranı güncelle"""
        if self.raw_eeg is None:
            return
        
        # Pozisyonu ilerlet (sadece oynatma sırasında)
        if self.is_playing:
            samples_per_update = int(SAMPLING_RATE * (UPDATE_INTERVAL / 1000) * self.playback_speed)
            self.current_pos = min(self.current_pos + samples_per_update, len(self.raw_eeg))
        
        # Görüntülenecek aralık
        start_idx = max(0, self.current_pos - DISPLAY_SAMPLES)
        end_idx = self.current_pos
        
        if end_idx - start_idx < 10:
            return
        
        # Önceden filtrelenmiş veya raw veriyi al
        if self.show_filtered and self.filtered_eeg is not None:
            display_data = self.filtered_eeg[start_idx:end_idx].copy()
            self.eeg_line.set_color(COLORS['eeg'])
        else:
            display_data = self.raw_eeg[start_idx:end_idx].copy()
            # Raw sinyal için DC offset düzeltmesi
            display_data = display_data - np.mean(display_data)
            self.eeg_line.set_color(COLORS['eeg_raw'])
        
        # Zaman ekseni
        t = np.linspace(0, len(display_data) / SAMPLING_RATE, len(display_data))
        
        # EEG grafiği güncelle
        self.eeg_line.set_data(t, display_data)
        self.ax_eeg.set_xlim(0, max(DISPLAY_WINDOW, t[-1] if len(t) > 0 else DISPLAY_WINDOW))
        
        # Y ekseni SABİT (sallanmaz)
        self.ax_eeg.set_ylim(-self.y_range_fixed, self.y_range_fixed)
        
        # Eski event çizgilerini temizle
        for line in self.event_lines:
            try:
                line.remove()
            except:
                pass
        self.event_lines = []
        
        # Event marker'ları çiz (görünür aralıkta olanlar)
        if self.events:
            # Görünür zaman aralığı
            start_sample = max(0, self.current_pos - DISPLAY_SAMPLES)
            end_sample = self.current_pos
            
            for event_sample, event_type in self.events:
                if start_sample <= event_sample <= end_sample:
                    # Zaman pozisyonu hesapla
                    relative_pos = (event_sample - start_sample) / SAMPLING_RATE
                    
                    if event_type == 'START':
                        line = self.ax_eeg.axvline(x=relative_pos, color='#00ff00', 
                                                    linewidth=2, alpha=0.8, linestyle='--')
                    else:  # END
                        line = self.ax_eeg.axvline(x=relative_pos, color='#ff0000', 
                                                    linewidth=2, alpha=0.8, linestyle='--')
                    self.event_lines.append(line)
        
        # FFT bant güçleri (CSV'den)
        if self.band_powers and self.current_pos > 0:
            idx = min(self.current_pos - 1, len(list(self.band_powers.values())[0]) - 1)
            heights = []
            for band in BAND_NAMES:
                if band in self.band_powers:
                    val = self.band_powers[band][idx]
                    heights.append(np.log1p(val))
                else:
                    heights.append(0)
            
            for bar, h in zip(self.fft_bars, heights):
                bar.set_height(h)
            
            max_h = max(heights) if heights else 20
            self.ax_fft.set_ylim(0, max(20, max_h * 1.2))
        
        # Canvas güncelle
        self.canvas.draw_idle()
        
        # Progress güncelle
        self.progress_var.set(self.current_pos)
        
        current_time = self.current_pos / SAMPLING_RATE
        total_time = len(self.raw_eeg) / SAMPLING_RATE
        self.progress_label.config(
            text=f"{int(current_time//60)}:{int(current_time%60):02d} / {int(total_time//60)}:{int(total_time%60):02d}"
        )


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    root = tk.Tk()
    app = EEGViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
