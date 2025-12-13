#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Log Transform + Oran Formülleri - GUI Versiyonu
==================================================

Tkinter + Matplotlib ile gerçek zamanlı EEG görselleştirmesi.

Özellikler:
    - Raw EEG dalga formu (gerçek zamanlı)
    - FFT bant güçleri (çubuk grafik)
    - Tahmin sonuçları (büyük gösterge)
    - Güven skoru göstergesi
    - Sinyal kalitesi göstergesi
    - Tahmin istatistikleri
    - Start/Stop/Kalibrasyon butonları

Kullanım:
    python realtime_transformed_gui.py

Gereksinimler:
    pip install torch numpy scipy matplotlib
"""

import os
import sys
import time
import socket
import json
import numpy as np
from collections import deque
from datetime import datetime
import pickle
import threading
import queue

# Tkinter
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("❌ Tkinter kurulu değil!")
    sys.exit(1)

# Matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.animation import FuncAnimation
except ImportError:
    print("❌ Matplotlib kurulu değil!")
    print("   Kurulum: pip install matplotlib")
    sys.exit(1)

# SciPy
try:
    from scipy import signal as scipy_signal
except ImportError:
    print("❌ SciPy kurulu değil!")
    print("   Kurulum: pip install scipy")
    sys.exit(1)

# PyTorch
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("❌ PyTorch kurulu değil!")
    print("   Kurulum: pip install torch")
    sys.exit(1)


# ============================================================================
# AYARLAR
# ============================================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLING_RATE = 512
FFT_WINDOW_SIZE = 512
MODEL_WINDOW = 128

NOTCH_FREQ = 50
NOTCH_Q = 30
LOWCUT = 0.5
HIGHCUT = 50
FILTER_ORDER = 4

ARTIFACT_THRESHOLD = 500

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['araba', 'aşağı', 'yukarı']
LABEL_COLORS = {'araba': '#FF6B6B', 'aşağı': '#4ECDC4', 'yukarı': '#95E1D3'}
LABEL_EMOJI = {'araba': '🚗', 'aşağı': '⬇️', 'yukarı': '⬆️'}


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
    
    def filter_signal(self, raw_samples):
        samples = np.array(raw_samples, dtype=np.float64)
        samples = samples - np.mean(samples)
        
        artifact_mask = np.abs(samples) > ARTIFACT_THRESHOLD
        if np.any(artifact_mask):
            good_samples = samples[~artifact_mask]
            if len(good_samples) > 0:
                median_val = np.median(good_samples)
                samples[artifact_mask] = median_val
        
        samples = scipy_signal.filtfilt(self.notch_b, self.notch_a, samples)
        samples = scipy_signal.filtfilt(self.bandpass_b, self.bandpass_a, samples)
        
        return samples
    
    def calculate_fft_bands(self, filtered_samples):
        samples = np.array(filtered_samples, dtype=np.float64)
        window = np.hamming(len(samples))
        samples = samples * window
        
        fft_vals = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / self.fs)
        power_spectrum = fft_vals ** 2
        
        band_powers = []
        for band_name in ['Delta', 'Theta', 'Low Alpha', 'High Alpha', 
                          'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']:
            low_freq, high_freq = FREQUENCY_BANDS[band_name]
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_powers.append(np.sum(power_spectrum[mask]))
        
        return band_powers
    
    def process_raw_to_fft(self, raw_samples):
        filtered = self.filter_signal(raw_samples)
        band_powers = self.calculate_fft_bands(filtered)
        return band_powers


# ============================================================================
# TRANSFORMASYON
# ============================================================================

def apply_log_transform(data):
    return np.sign(data) * np.log1p(np.abs(data))

def calculate_band_ratios(window):
    delta = window[:, 1] + 1e-8
    theta = window[:, 2] + 1e-8
    low_alpha = window[:, 3] + 1e-8
    high_alpha = window[:, 4] + 1e-8
    low_beta = window[:, 5] + 1e-8
    high_beta = window[:, 6] + 1e-8
    low_gamma = window[:, 7] + 1e-8
    mid_gamma = window[:, 8] + 1e-8
    
    alpha = (low_alpha + high_alpha) / 2
    beta = (low_beta + high_beta) / 2
    gamma = (low_gamma + mid_gamma) / 2
    
    ratios = np.column_stack([
        delta / theta,
        theta / alpha,
        alpha / beta,
        beta / gamma,
        (theta + alpha) / (beta + gamma),
        delta / alpha,
        (delta + theta) / (alpha + beta + gamma),
        (alpha + beta) / (delta + theta),
    ])
    
    return ratios

def transform_window(window):
    log_transformed = apply_log_transform(window)
    ratios = calculate_band_ratios(window)
    ratios_log = apply_log_transform(ratios)
    combined = np.hstack([log_transformed, ratios_log])
    return combined


# ============================================================================
# TCN MODEL
# ============================================================================

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        
        if out.size(2) != res.size(2):
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        
        return self.relu(out + res)


class TCN_Model(nn.Module):
    def __init__(self, input_channels=17, num_classes=3, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(TCN_Model, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size,
                                       padding=padding, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y.transpose(1, 2)
        y = torch.mean(y, dim=1)
        return self.fc(y)


# ============================================================================
# THINKGEAR CONNECTOR
# ============================================================================

class ThinkGearConnector:
    def __init__(self, host='127.0.0.1', port=13854):
        self.host = host
        self.port = port
        self.socket = None
        self.raw_buffer = deque(maxlen=2048)
        self.poor_signal = 200
        self.raw_count = 0
        self.running = False
    
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            
            config = json.dumps({
                "enableRawOutput": True,
                "format": "Json"
            })
            self.socket.send((config + '\n').encode())
            self.running = True
            return True
        except Exception as e:
            print(f"❌ Bağlantı hatası: {e}")
            return False
    
    def disconnect(self):
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
    
    def read_data(self):
        if not self.socket:
            return None
        
        try:
            data = self.socket.recv(8192).decode('utf-8')
            
            for line in data.split('\r\n'):
                if not line.strip():
                    continue
                
                try:
                    packet = json.loads(line)
                    
                    if 'poorSignalLevel' in packet:
                        self.poor_signal = packet['poorSignalLevel']
                    
                    if 'rawEeg' in packet:
                        self.raw_buffer.append(packet['rawEeg'])
                        self.raw_count += 1
                        return 'raw'
                    
                except json.JSONDecodeError:
                    continue
            
            return None
            
        except Exception as e:
            return None
    
    def get_raw_samples(self, n):
        if len(self.raw_buffer) >= n:
            return list(self.raw_buffer)[-n:]
        return list(self.raw_buffer)
    
    def get_buffer_size(self):
        return len(self.raw_buffer)


# ============================================================================
# GUI APPLICATION
# ============================================================================

class EEGVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 EEG Tahmin Sistemi - GUI")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2C3E50')
        
        # Değişkenler
        self.is_running = False
        self.is_calibrated = False
        self.calibration_mean = None
        self.calibration_std = None
        
        # MindWave
        self.thinkgear = ThinkGearConnector()
        self.signal_processor = SignalProcessor()
        
        # Model
        self.model = None
        self.scaler = None
        self.device = DEVICE
        
        # Veri buffer'ları
        self.raw_eeg_data = deque(maxlen=512)
        self.fft_buffer = deque(maxlen=MODEL_WINDOW)
        self.current_fft_bands = np.zeros(8)
        
        # Tahmin verileri
        self.current_prediction = None
        self.current_confidence = 0.0
        self.prediction_counts = {label: 0 for label in LABELS}
        self.prediction_history = deque(maxlen=50)
        
        # Thread kontrolü
        self.data_thread = None
        self.stop_thread = threading.Event()
        self.data_queue = queue.Queue()
        
        # GUI bileşenleri
        self.create_widgets()
        self.load_model()
        
        # MindWave'e otomatik bağlan
        self.connect_mindwave()
        
        # GUI güncelleme
        self.update_gui()
    
    def create_widgets(self):
        # Üst panel - Tahmin göstergesi
        top_frame = tk.Frame(self.root, bg='#34495E', height=150)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        self.prediction_label = tk.Label(
            top_frame,
            text="Bekleniyor...",
            font=('Arial', 36, 'bold'),
            bg='#34495E',
            fg='white'
        )
        self.prediction_label.pack(pady=10)
        
        self.confidence_label = tk.Label(
            top_frame,
            text="Güven: ---%",
            font=('Arial', 16),
            bg='#34495E',
            fg='#BDC3C7'
        )
        self.confidence_label.pack()
        
        info_frame = tk.Frame(top_frame, bg='#34495E')
        info_frame.pack(pady=5)
        
        self.signal_label = tk.Label(
            info_frame,
            text="Sinyal: --",
            font=('Arial', 12),
            bg='#34495E',
            fg='#BDC3C7'
        )
        self.signal_label.pack(side='left', padx=10)
        
        self.status_label = tk.Label(
            info_frame,
            text="● Durdu",
            font=('Arial', 12),
            bg='#34495E',
            fg='#E74C3C'
        )
        self.status_label.pack(side='left', padx=10)
        
        # Grafik paneli
        graph_frame = tk.Frame(self.root, bg='#2C3E50')
        graph_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6), 
                                                        facecolor='#2C3E50')
        self.fig.tight_layout(pad=3)
        
        # Raw EEG grafiği
        self.ax1.set_facecolor('#34495E')
        self.ax1.set_title('Raw EEG Sinyali', color='white', fontsize=12)
        self.ax1.set_ylabel('Amplitüd (µV)', color='white')
        self.ax1.tick_params(colors='white')
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], color='#3498DB', linewidth=1)
        self.ax1.set_xlim(0, 512)
        self.ax1.set_ylim(-200, 200)
        
        # FFT bantları grafiği
        self.ax2.set_facecolor('#34495E')
        self.ax2.set_title('FFT Bant Güçleri', color='white', fontsize=12)
        self.ax2.set_ylabel('Güç', color='white')
        self.ax2.tick_params(colors='white')
        self.ax2.grid(True, alpha=0.3, axis='y')
        
        band_names = ['Delta', 'Theta', 'L.Alpha', 'H.Alpha', 
                      'L.Beta', 'H.Beta', 'L.Gamma', 'M.Gamma']
        self.bar_colors = ['#E74C3C', '#E67E22', '#F39C12', '#F1C40F',
                          '#2ECC71', '#1ABC9C', '#3498DB', '#9B59B6']
        x_pos = np.arange(len(band_names))
        self.bars = self.ax2.bar(x_pos, np.zeros(8), color=self.bar_colors, alpha=0.8)
        self.ax2.set_xticks(x_pos)
        self.ax2.set_xticklabels(band_names, rotation=45, ha='right', color='white')
        self.ax2.set_ylim(0, 1)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Alt panel - İstatistikler
        stats_frame = tk.Frame(self.root, bg='#34495E', height=80)
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        self.stats_label = tk.Label(
            stats_frame,
            text="İstatistikler: -- | Toplam: 0 tahmin",
            font=('Arial', 12),
            bg='#34495E',
            fg='white',
            justify='left'
        )
        self.stats_label.pack(pady=10, padx=10)
        
        # Kontrol butonları
        button_frame = tk.Frame(self.root, bg='#2C3E50')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.calibrate_btn = tk.Button(
            button_frame,
            text="📊 Kalibrasyon",
            font=('Arial', 12, 'bold'),
            bg='#F39C12',
            fg='white',
            activebackground='#E67E22',
            command=self.start_calibration,
            width=15,
            height=2
        )
        self.calibrate_btn.pack(side='left', padx=5)
        
        self.start_btn = tk.Button(
            button_frame,
            text="▶ Başlat",
            font=('Arial', 12, 'bold'),
            bg='#27AE60',
            fg='white',
            activebackground='#229954',
            command=self.start_prediction,
            width=15,
            height=2
        )
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏸ Durdur",
            font=('Arial', 12, 'bold'),
            bg='#E74C3C',
            fg='white',
            activebackground='#C0392B',
            command=self.stop_prediction,
            width=15,
            height=2,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=5)
        
        self.quit_btn = tk.Button(
            button_frame,
            text="❌ Çıkış",
            font=('Arial', 12, 'bold'),
            bg='#95A5A6',
            fg='white',
            activebackground='#7F8C8D',
            command=self.quit_app,
            width=15,
            height=2
        )
        self.quit_btn.pack(side='right', padx=5)
    
    def load_model(self):
        """Model ve scaler'ı yükle"""
        try:
            # Scaler
            scaler_path = os.path.join(MODEL_DIR, 'scaler_transformed.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Model
            model_path = os.path.join(MODEL_DIR, 'best_model_transformed.pth')
            if os.path.exists(model_path):
                self.model = TCN_Model(input_channels=17, num_classes=3).to(self.device)
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print("✅ Model yüklendi")
            else:
                messagebox.showerror("Hata", f"Model bulunamadı:\n{model_path}")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Model yükleme hatası:\n{str(e)}")
    
    def connect_mindwave(self):
        """MindWave'e bağlan"""
        def try_connect():
            if self.thinkgear.connect():
                self.root.after(0, lambda: self.signal_label.config(
                    text="✅ MindWave Bağlandı",
                    fg='#27AE60'
                ))
                print("✅ MindWave'e bağlanıldı")
            else:
                self.root.after(0, lambda: self.signal_label.config(
                    text="❌ MindWave Bağlı Değil",
                    fg='#E74C3C'
                ))
                self.root.after(0, lambda: messagebox.showwarning(
                    "Bağlantı Hatası",
                    "MindWave'e bağlanılamadı!\n\n"
                    "Lütfen kontrol edin:\n"
                    "• ThinkGear Connector açık mı?\n"
                    "• MindWave açık ve eşleşmiş mi?"
                ))
        
        # Thread'de bağlan (GUI donmasın)
        threading.Thread(target=try_connect, daemon=True).start()
    
    def start_calibration(self):
        """Kalibrasyon başlat"""
        if self.is_running:
            messagebox.showwarning("Uyarı", "Önce tahmini durdurun!")
            return
        
        # Bağlan
        if not self.thinkgear.running:
            if not self.thinkgear.connect():
                messagebox.showerror("Hata", "MindWave'e bağlanılamadı!")
                return
        
        response = messagebox.askokcancel(
            "Kalibrasyon",
            "15 saniye boyunca:\n• Gözlerinizi kapatın\n• Hiçbir şey düşünmeyin\n\nHazır mısınız?"
        )
        
        if not response:
            return
        
        self.calibrate_btn.config(state='disabled', text='⏳ Kalibrasyon...')
        
        # Thread'de kalibrasyon yap
        def calibrate():
            calibration_data = []
            start_time = time.time()
            last_raw_count = 0
            
            while (time.time() - start_time) < 15:
                result = self.thinkgear.read_data()
                
                if result == 'raw':
                    raw_buffer_size = self.thinkgear.get_buffer_size()
                    new_samples = self.thinkgear.raw_count - last_raw_count
                    
                    if raw_buffer_size >= FFT_WINDOW_SIZE and new_samples >= 256:
                        last_raw_count = self.thinkgear.raw_count
                        raw_samples = self.thinkgear.get_raw_samples(FFT_WINDOW_SIZE)
                        band_powers = self.signal_processor.process_raw_to_fft(raw_samples)
                        calibration_data.append([0] + band_powers)
                
                time.sleep(0.001)
            
            if len(calibration_data) >= 10:
                cal_array = np.array(calibration_data, dtype=np.float32)
                if len(cal_array) >= MODEL_WINDOW:
                    cal_transformed = transform_window(cal_array[:MODEL_WINDOW])
                    self.calibration_mean = np.mean(cal_transformed.flatten())
                    self.calibration_std = np.std(cal_transformed.flatten())
                else:
                    self.calibration_mean = np.mean(cal_array.flatten())
                    self.calibration_std = np.std(cal_array.flatten())
                
                self.is_calibrated = True
                self.root.after(0, lambda: messagebox.showinfo("Başarılı", "✅ Kalibrasyon tamamlandı!"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Hata", "❌ Yeterli veri toplanamadı!"))
            
            self.root.after(0, lambda: self.calibrate_btn.config(state='normal', text='📊 Kalibrasyon'))
        
        threading.Thread(target=calibrate, daemon=True).start()
    
    def start_prediction(self):
        """Tahmin başlat"""
        if self.model is None:
            messagebox.showerror("Hata", "Model yüklenmedi!")
            return
        
        if not self.is_calibrated:
            response = messagebox.askyesno(
                "Uyarı",
                "Kalibrasyon yapılmadı!\nYine de devam etmek istiyor musunuz?"
            )
            if not response:
                return
        
        # Bağlan
        if not self.thinkgear.running:
            if not self.thinkgear.connect():
                messagebox.showerror("Hata", "MindWave'e bağlanılamadı!")
                return
        
        self.is_running = True
        self.stop_thread.clear()
        
        # Buton durumları
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.calibrate_btn.config(state='disabled')
        self.status_label.config(text="● Çalışıyor", fg='#27AE60')
        
        # Veri toplama thread'i
        self.data_thread = threading.Thread(target=self.data_collection_loop, daemon=True)
        self.data_thread.start()
    
    def stop_prediction(self):
        """Tahmini durdur"""
        self.is_running = False
        self.stop_thread.set()
        
        # Buton durumları
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.calibrate_btn.config(state='normal')
        self.status_label.config(text="● Durdu", fg='#E74C3C')
    
    def data_collection_loop(self):
        """Veri toplama döngüsü"""
        last_raw_count = 0
        
        while not self.stop_thread.is_set():
            result = self.thinkgear.read_data()
            
            if result == 'raw':
                # Raw EEG buffer'a ekle
                if len(self.thinkgear.raw_buffer) > 0:
                    self.raw_eeg_data.extend(list(self.thinkgear.raw_buffer)[-50:])
                
                # FFT hesapla
                raw_buffer_size = self.thinkgear.get_buffer_size()
                new_samples = self.thinkgear.raw_count - last_raw_count
                
                if raw_buffer_size >= FFT_WINDOW_SIZE and new_samples >= 256:
                    last_raw_count = self.thinkgear.raw_count
                    
                    raw_samples = self.thinkgear.get_raw_samples(FFT_WINDOW_SIZE)
                    band_powers = self.signal_processor.process_raw_to_fft(raw_samples)
                    
                    self.current_fft_bands = np.array(band_powers)
                    self.fft_buffer.append([0] + band_powers)
                    
                    # Tahmin yap
                    if len(self.fft_buffer) >= MODEL_WINDOW:
                        self.make_prediction()
            
            time.sleep(0.001)
    
    def make_prediction(self):
        """Tahmin yap"""
        try:
            window_data = list(self.fft_buffer)[-MODEL_WINDOW:]
            x = np.array(window_data, dtype=np.float32)
            
            # Transform
            x_transformed = transform_window(x)
            
            # Kalibrasyon
            if self.is_calibrated and self.calibration_mean is not None:
                x_flat = x_transformed.flatten()
                x_flat = x_flat - self.calibration_mean
                x_flat = x_flat / (self.calibration_std + 1e-8)
                x_transformed = x_flat.reshape(x_transformed.shape)
            
            # Scaler
            if self.scaler is not None:
                x_flat = x_transformed.reshape(1, -1)
                x_normalized = self.scaler.transform(x_flat)
                x_transformed = x_normalized.reshape(MODEL_WINDOW, 17)
            
            # Model
            x_tensor = torch.FloatTensor(x_transformed).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(x_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
            label = LABELS[predicted.item()]
            conf = confidence.item()
            
            # Güncelle
            self.current_prediction = label
            self.current_confidence = conf
            
            if conf >= 0.6:  # Güven eşiği
                self.prediction_counts[label] += 1
                self.prediction_history.append(label)
        
        except Exception as e:
            print(f"Tahmin hatası: {e}")
    
    def update_gui(self):
        """GUI'yi güncelle"""
        # Tahmin göstergesi
        if self.current_prediction:
            emoji = LABEL_EMOJI.get(self.current_prediction, '')
            color = LABEL_COLORS.get(self.current_prediction, 'white')
            self.prediction_label.config(
                text=f"{emoji} {self.current_prediction.upper()}",
                fg=color
            )
            self.confidence_label.config(
                text=f"Güven: {self.current_confidence*100:.1f}%"
            )
        
        # Sinyal kalitesi
        if self.thinkgear.running:
            ps = self.thinkgear.poor_signal
            sig_text = f"Sinyal: {'✅ İyi' if ps < 50 else f'⚠️ Zayıf ({ps})'}"
            self.signal_label.config(text=sig_text)
        
        # İstatistikler
        total = sum(self.prediction_counts.values())
        stats_text = " | ".join([f"{label}: {count}" for label, count in self.prediction_counts.items()])
        stats_text += f" | Toplam: {total} tahmin"
        self.stats_label.config(text=stats_text)
        
        # Grafikleri güncelle
        if len(self.raw_eeg_data) > 0:
            self.line1.set_data(range(len(self.raw_eeg_data)), list(self.raw_eeg_data))
            self.ax1.set_xlim(0, max(512, len(self.raw_eeg_data)))
        
        if np.any(self.current_fft_bands):
            # Normalize et
            max_val = np.max(self.current_fft_bands)
            if max_val > 0:
                normalized = self.current_fft_bands / max_val
            else:
                normalized = self.current_fft_bands
            
            for bar, height in zip(self.bars, normalized):
                bar.set_height(height)
        
        self.canvas.draw_idle()
        
        # 50ms'de bir güncelle
        self.root.after(50, self.update_gui)
    
    def quit_app(self):
        """Uygulamadan çık"""
        self.stop_prediction()
        self.thinkgear.disconnect()
        self.root.quit()


# ============================================================================
# ANA PROGRAM
# ============================================================================

def main():
    print("="*60)
    print("🧠 EEG TAHMİN SİSTEMİ - GUI")
    print("="*60)
    print(f"📱 Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)
    
    root = tk.Tk()
    app = EEGVisualizerGUI(root)
    
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()


if __name__ == '__main__':
    main()
