#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM+CNN Hibrit Model - GUI ile Canlı Tahmin (Raw EEG → FFT)
============================================================

Raw EEG sinyalinden FFT hesaplar (eğitimle aynı pipeline).
MindWave'in kendi FFT'sini DEĞİL, bizim hesapladığımız FFT'yi kullanır.

Pipeline:
    Raw EEG → Notch (50Hz) → Bandpass (0.5-50Hz) → FFT → Model

Model Seçenekleri:
    --model seq64   : Baseline model (sequence_length=64)
    --model seq96   : Genişletilmiş görüş (sequence_length=96)
    --model seq128  : En geniş görüş (sequence_length=128)

Kullanım:
    python realtime_gui.py --simulation        (Test için - varsayılan seq64)
    python realtime_gui.py --simulation --model seq96   (seq96 ile test)
    python realtime_gui.py --port COM5         (Seri port)
    python realtime_gui.py --thinkgear --model seq96    (ThinkGear + seq96)
"""

import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
from collections import deque
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import torch
import torch.nn as nn

# Sinyal işleme modülü
from signal_processor import SignalProcessor, BAND_NAMES, SAMPLING_RATE, WINDOW_SIZE, FREQUENCY_BANDS

# ============================================================================
# AYARLAR
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BAND_COLORS = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1',
               '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3']

COLORS = {
    'bg': '#1a1a2e',
    'panel': '#16213e',
    'text': 'white',
    'yukarı': '#00ff88',
    'aşağı': '#ff6b6b',
    'asagı': '#ff6b6b',
    'araba': '#feca57'
}

# Model seçenekleri
AVAILABLE_MODELS = {
    'seq32': {
        'name': 'Hızlı (seq32)',
        'model': 'seq32_best_model.pth',
        'config': 'seq32_config.json',
        'scaler': 'seq32_scaler.pkl',
        'label_map': 'seq32_label_map.json',
        'description': 'En hızlı tepki, sequence_length=32 (~4s gecikme)'
    },
    'seq64': {
        'name': 'Baseline (seq64)',
        'model': 'best_model.pth',
        'config': 'config.json',
        'scaler': 'scaler.pkl',
        'label_map': 'label_map.json',
        'description': 'Varsayılan model, sequence_length=64'
    },
    'seq96': {
        'name': 'Genişletilmiş (seq96)',
        'model': 'seq96_best_model.pth',
        'config': 'seq96_config.json',
        'scaler': 'seq96_scaler.pkl',
        'label_map': 'seq96_label_map.json',
        'description': 'Daha geniş görüş alanı, sequence_length=96'
    },
    'seq128': {
        'name': 'En Geniş (seq128)',
        'model': 'seq128_best_model.pth',
        'config': 'seq128_config.json',
        'scaler': 'seq128_scaler.pkl',
        'label_map': 'seq128_label_map.json',
        'description': 'En geniş görüş alanı, sequence_length=128'
    }
}


# ============================================================================
# MODEL (train_model.py ile aynı)
# ============================================================================

class SimpleCNN_LSTM(nn.Module):
    def __init__(self, input_features=15, num_classes=3, dropout=0.4):
        super(SimpleCNN_LSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
        )
        
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(x)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        out = self.fc(hidden)
        return out


# ============================================================================
# SIMÜLASYON MODU (Test için - Raw EEG üretir)
# ============================================================================

class SimulatedMindWave:
    """Test için simüle edilmiş MindWave - RAW EEG üretir"""
    
    def __init__(self):
        self.running = True
        self.poor_signal = 0
        self.current_class = 0
        self.sample_count = 0
        self.last_time = time.time()
        
        # Raw sample buffer
        self.raw_buffer = deque(maxlen=2048)
        self.lock = threading.Lock()
    
    def _generate_raw_eeg(self, class_idx, num_samples):
        """Sınıfa göre raw EEG üret"""
        t = np.linspace(0, num_samples / SAMPLING_RATE, num_samples)
        
        # Temel EEG bileşenleri
        if class_idx == 0:  # yukarı: Alpha dominant
            signal = (
                30 * np.sin(2 * np.pi * 10 * t) +   # Alpha (10 Hz)
                15 * np.sin(2 * np.pi * 5 * t) +    # Theta
                10 * np.sin(2 * np.pi * 20 * t) +   # Beta
                5 * np.random.randn(num_samples)
            )
        elif class_idx == 1:  # aşağı: Theta/Delta dominant
            signal = (
                40 * np.sin(2 * np.pi * 2 * t) +    # Delta (2 Hz)
                25 * np.sin(2 * np.pi * 5 * t) +    # Theta (5 Hz)
                5 * np.sin(2 * np.pi * 10 * t) +    # Alpha
                5 * np.random.randn(num_samples)
            )
        else:  # araba: Beta dominant
            signal = (
                10 * np.sin(2 * np.pi * 2 * t) +    # Delta
                15 * np.sin(2 * np.pi * 5 * t) +    # Theta
                20 * np.sin(2 * np.pi * 10 * t) +   # Alpha
                35 * np.sin(2 * np.pi * 20 * t) +   # Beta (dominant)
                5 * np.random.randn(num_samples)
            )
        
        return signal.astype(np.float32)
    
    def get_raw_samples(self):
        """Simüle edilmiş raw sample'ları al"""
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        
        # Her 5 saniyede sınıf değiştir
        self.current_class = int(time.time() / 5) % 3
        
        # Geçen süreye göre sample üret (512 Hz)
        num_samples = int(elapsed * SAMPLING_RATE)
        num_samples = min(num_samples, 512)  # Max 1 saniye
        
        if num_samples > 0:
            samples = self._generate_raw_eeg(self.current_class, num_samples)
            self.sample_count += num_samples
            return list(samples)
        
        return []
    
    def connect(self):
        return True
    
    def disconnect(self):
        self.running = False
    
    def start(self):
        pass


# ============================================================================
# TAHMİN MOTORU
# ============================================================================

class PredictionEngine:
    """Raw EEG'den FFT hesaplar ve model ile tahmin yapar"""
    
    def __init__(self, model_path, scaler_path, config_path, label_map_path=None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        if label_map_path is None:
            label_map_path = os.path.join(os.path.dirname(config_path), 'label_map.json')
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.model = SimpleCNN_LSTM(
            input_features=self.config['num_features'],
            num_classes=self.config['num_classes'],
            dropout=0.0
        ).to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.val_acc = checkpoint.get('val_acc', 0)
        
        # Sinyal işleyici
        self.signal_processor = SignalProcessor()
        
        # Feature buffer
        self.feature_buffer = deque(maxlen=self.config['sequence_length'])
        
        # Son FFT değerleri (grafik için)
        self.last_fft_values = {band: 0.0 for band in BAND_NAMES}
        
        # Prediction history
        self.prediction_history = deque(maxlen=5)
    
    def process_raw_samples(self, raw_samples):
        """Raw sample'ları işle"""
        fft_count = 0
        
        for sample in raw_samples:
            result = self.signal_processor.add_sample(sample)
            
            if result is not None:
                self.last_fft_values = result.copy()
                self._add_fft_to_buffer(result)
                fft_count += 1
        
        return fft_count
    
    def _add_fft_to_buffer(self, fft_dict):
        """FFT'yi feature buffer'a ekle"""
        features = [fft_dict.get(band, 0) for band in BAND_NAMES]
        features = np.array(features, dtype=np.float32)
        features = np.log1p(np.abs(features))
        
        delta, theta = features[0], features[1]
        low_alpha, high_alpha = features[2], features[3]
        low_beta, high_beta = features[4], features[5]
        low_gamma, mid_gamma = features[6], features[7]
        
        alpha_total = low_alpha + high_alpha
        beta_total = low_beta + high_beta
        gamma_total = low_gamma + mid_gamma
        
        eps = 1e-6
        theta_beta_ratio = theta / (beta_total + eps)
        alpha_beta_ratio = alpha_total / (beta_total + eps)
        theta_alpha_ratio = theta / (alpha_total + eps)
        engagement = beta_total / (alpha_total + theta + eps)
        
        extended = np.concatenate([
            features,
            [alpha_total, beta_total, gamma_total,
             theta_beta_ratio, alpha_beta_ratio, theta_alpha_ratio, engagement]
        ])
        
        self.feature_buffer.append(extended)
    
    def predict(self):
        """Tahmin yap"""
        if len(self.feature_buffer) < self.config['sequence_length']:
            return None, 0.0, {}
        
        sequence = np.array(self.feature_buffer, dtype=np.float32)
        
        original_shape = sequence.shape
        seq_flat = sequence.reshape(-1, sequence.shape[-1])
        seq_flat = self.scaler.transform(seq_flat)
        sequence = seq_flat.reshape(original_shape)
        
        x = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, predicted = probs.max(0)
            
            pred_class = predicted.item()
            conf = confidence.item()
            
            class_probs = {self.label_map[str(i)]: probs[i].item() for i in range(len(probs))}
        
        self.prediction_history.append((pred_class, conf))
        
        from collections import Counter
        recent_preds = [p[0] for p in self.prediction_history if p[1] > 0.4]
        if recent_preds:
            smoothed_pred = Counter(recent_preds).most_common(1)[0][0]
        else:
            smoothed_pred = pred_class
        
        label = self.label_map.get(str(smoothed_pred), f"Class {smoothed_pred}")
        
        return label, conf, class_probs
    
    def get_fft_values(self):
        """Son FFT değerleri"""
        return self.last_fft_values.copy()
    
    def get_buffer_status(self):
        """Buffer durumu"""
        return {
            'raw': self.signal_processor.get_buffer_progress(),
            'features': len(self.feature_buffer),
            'max_features': self.config['sequence_length'],
            'samples': self.signal_processor.total_samples,
            'artifacts': self.signal_processor.artifact_count
        }


# ============================================================================
# GUI
# ============================================================================

class RealtimeGUI:
    def __init__(self, root, use_simulation=False, port=None, use_thinkgear=False, model_name='seq64'):
        self.root = root
        self.root.title("🧠 LSTM+CNN Canlı Tahmin (Raw → FFT)")
        self.root.geometry("1200x800")
        self.root.configure(bg=COLORS['bg'])
        
        self.use_simulation = use_simulation
        self.use_thinkgear = use_thinkgear
        self.port = port
        self.model_name = model_name  # Seçilen model
        self.running = False
        self.monitoring = False  # Sinyal izleme modu
        self.connector = None
        self.engine = None
        
        # FFT geçmişi
        self.fft_history = {band: deque(maxlen=100) for band in BAND_NAMES}
        
        self.create_ui()
        self.load_model()
    
    def create_ui(self):
        # Üst - Tahmin
        top_frame = tk.Frame(self.root, bg=COLORS['panel'], height=200)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        top_frame.pack_propagate(False)
        
        # Model bilgisi
        model_info = AVAILABLE_MODELS.get(self.model_name, AVAILABLE_MODELS['seq64'])
        tk.Label(top_frame, text=f"Model: {model_info['name']} | Pipeline: Raw EEG → Filtre → FFT → Model",
                bg=COLORS['panel'], fg='#888', font=('Segoe UI', 9)).pack(pady=(5,0))
        
        self.pred_label = tk.Label(top_frame, text="⏳ Bekleniyor...",
                                   font=('Segoe UI', 32, 'bold'),
                                   bg=COLORS['panel'], fg='white')
        self.pred_label.pack(pady=5)
        
        # Güven barları
        self.conf_frame = tk.Frame(top_frame, bg=COLORS['panel'])
        self.conf_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.conf_bars = {}
        for class_name in ['yukarı', 'asagı', 'araba']:
            frame = tk.Frame(self.conf_frame, bg=COLORS['panel'])
            frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
            
            display_name = class_name.upper()
            color = COLORS.get(class_name, 'white')
            tk.Label(frame, text=display_name, bg=COLORS['panel'],
                    fg=color, font=('Segoe UI', 10, 'bold')).pack()
            
            bar = ttk.Progressbar(frame, length=120, mode='determinate')
            bar.pack(pady=3)
            self.conf_bars[class_name] = bar
            
            pct_label = tk.Label(frame, text="0%", bg=COLORS['panel'],
                                fg='white', font=('Segoe UI', 9))
            pct_label.pack(pady=1)
            self.conf_bars[f"{class_name}_label"] = pct_label
        
        # Orta - Grafikler
        mid_frame = tk.Frame(self.root, bg=COLORS['bg'])
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        self.fig = Figure(figsize=(12, 5), facecolor=COLORS['bg'])
        
        self.ax_bands = self.fig.add_subplot(121)
        self.ax_bands.set_facecolor(COLORS['bg'])
        self.ax_bands.set_title('FFT Bant Güçleri (Hesaplanan)', color='white')
        self.ax_bands.tick_params(colors='white')
        
        self.ax_time = self.fig.add_subplot(122)
        self.ax_time.set_facecolor(COLORS['bg'])
        self.ax_time.set_title('Zaman Serisi', color='white')
        self.ax_time.tick_params(colors='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Alt - Kontroller
        bottom_frame = tk.Frame(self.root, bg=COLORS['panel'], height=100)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        bottom_frame.pack_propagate(False)
        
        # BAĞLAN butonu (Aşama 1)
        self.btn_connect = tk.Button(bottom_frame, text="🔌 BAĞLAN",
                                     font=('Segoe UI', 12, 'bold'),
                                     bg='#3498db', fg='white',
                                     width=10, height=1,
                                     command=self.connect_device)
        self.btn_connect.pack(side=tk.LEFT, padx=5, pady=5)
        
        # BAŞLAT butonu (Aşama 2)
        self.btn_start = tk.Button(bottom_frame, text="▶️ BAŞLAT",
                                   font=('Segoe UI', 12, 'bold'),
                                   bg='#95a5a6', fg='white',
                                   width=10, height=1,
                                   state=tk.DISABLED,
                                   command=self.start_prediction)
        self.btn_start.pack(side=tk.LEFT, padx=5, pady=5)
        
        # DURDUR butonu
        self.btn_stop = tk.Button(bottom_frame, text="⏹️ DURDUR",
                                  font=('Segoe UI', 12, 'bold'),
                                  bg='#95a5a6', fg='white',
                                  width=10, height=1,
                                  state=tk.DISABLED,
                                  command=self.stop_prediction)
        self.btn_stop.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Mod seçimi
        mode_frame = tk.Frame(bottom_frame, bg=COLORS['panel'])
        mode_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(mode_frame, text="Mod:", bg=COLORS['panel'],
                fg='white', font=('Segoe UI', 11)).pack(side=tk.LEFT)
        
        # Varsayılan mod seç
        if self.use_simulation:
            default_mode = 'simulation'
        elif self.use_thinkgear:
            default_mode = 'thinkgear'
        else:
            default_mode = 'thinkgear'  # ThinkGear varsayılan
        
        self.mode_var = tk.StringVar(value=default_mode)
        
        tk.Radiobutton(mode_frame, text="Simülasyon", variable=self.mode_var,
                      value='simulation', bg=COLORS['panel'], fg='white',
                      selectcolor=COLORS['bg'], font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Radiobutton(mode_frame, text="ThinkGear", variable=self.mode_var,
                      value='thinkgear', bg=COLORS['panel'], fg='white',
                      selectcolor=COLORS['bg'], font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Radiobutton(mode_frame, text="Seri Port", variable=self.mode_var,
                      value='serial', bg=COLORS['panel'], fg='white',
                      selectcolor=COLORS['bg'], font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=5)
        
        # Port
        port_frame = tk.Frame(bottom_frame, bg=COLORS['panel'])
        port_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(port_frame, text="Port:", bg=COLORS['panel'],
                fg='white', font=('Segoe UI', 11)).pack(side=tk.LEFT)
        
        self.port_entry = tk.Entry(port_frame, width=15, font=('Segoe UI', 11))
        self.port_entry.insert(0, self.port or "COM5")
        self.port_entry.pack(side=tk.LEFT, padx=5)
        
        # Durum
        status_frame = tk.Frame(bottom_frame, bg=COLORS['panel'])
        status_frame.pack(side=tk.RIGHT, padx=20)
        
        self.status_label = tk.Label(status_frame, text="Hazır",
                                     bg=COLORS['panel'], fg='#888',
                                     font=('Segoe UI', 10))
        self.status_label.pack()
        
        self.buffer_label = tk.Label(status_frame, text="Raw: 0% | Features: 0/64",
                                     bg=COLORS['panel'], fg='#888',
                                     font=('Segoe UI', 9))
        self.buffer_label.pack()
        
        self.sample_label = tk.Label(status_frame, text="Samples: 0 | Artifacts: 0",
                                     bg=COLORS['panel'], fg='#888',
                                     font=('Segoe UI', 9))
        self.sample_label.pack()
    
    def load_model(self):
        # Model bilgilerini al
        model_info = AVAILABLE_MODELS.get(self.model_name, AVAILABLE_MODELS['seq64'])
        model_path = os.path.join(SCRIPT_DIR, model_info['model'])
        scaler_path = os.path.join(SCRIPT_DIR, model_info['scaler'])
        config_path = os.path.join(SCRIPT_DIR, model_info['config'])
        label_map_path = os.path.join(SCRIPT_DIR, model_info['label_map'])
        
        missing_files = []
        for path, name in [(model_path, 'Model'), (scaler_path, 'Scaler'), 
                          (config_path, 'Config'), (label_map_path, 'Label Map')]:
            if not os.path.exists(path):
                missing_files.append(os.path.basename(path))
        
        if missing_files:
            self.status_label.config(text=f"⚠️ {self.model_name} bulunamadı!", fg='red')
            messagebox.showwarning("Uyarı", 
                f"'{self.model_name}' model dosyaları bulunamadı!\n\n"
                f"Eksik: {', '.join(missing_files)}\n\n"
                f"Eğitmek için:\n"
                f"python train_experiment.py --seq-len {self.model_name.replace('seq', '')}")
            return False
        
        self.engine = PredictionEngine(model_path, scaler_path, config_path, label_map_path)
        seq_len = self.engine.config.get('sequence_length', 64)
        self.status_label.config(
            text=f"✅ {model_info['name']} (Acc: {self.engine.val_acc:.1f}%, seq={seq_len})", 
            fg='#00ff88'
        )
        return True
    
    def connect_device(self):
        """AŞAMA 1: Cihaza bağlan ve sinyal kalitesini göster"""
        if self.engine is None:
            if not self.load_model():
                return
        
        mode = self.mode_var.get()
        
        if mode == 'simulation':
            self.connector = SimulatedMindWave()
            self.connector.connect()
            self.status_label.config(text="✅ Simülasyon bağlandı", fg='#00ff88')
        
        elif mode == 'thinkgear':
            try:
                from realtime_predict import ThinkGearConnector
                self.connector = ThinkGearConnector(host='127.0.0.1', port=13854)
                if not self.connector.connect():
                    messagebox.showerror("Hata", 
                        "ThinkGear Connector'a bağlanılamadı!\n\n"
                        "ThinkGear Connector uygulamasının çalıştığından emin olun.")
                    return
                self.connector.start()
                self.status_label.config(text="✅ ThinkGear bağlandı", fg='#00ff88')
            except Exception as e:
                messagebox.showerror("Hata", f"ThinkGear bağlantı hatası: {e}")
                return
        
        elif mode == 'serial':
            try:
                from realtime_predict import MindWaveRawConnector
                port = self.port_entry.get()
                self.connector = MindWaveRawConnector(port)
                if not self.connector.connect():
                    messagebox.showerror("Hata", f"MindWave'e bağlanılamadı: {port}")
                    return
                self.connector.start()
                self.status_label.config(text=f"✅ Seri port bağlandı ({port})", fg='#00ff88')
            except Exception as e:
                messagebox.showerror("Hata", f"Seri port bağlantı hatası: {e}")
                return
        
        # Butonları güncelle
        self.btn_connect.config(text="🔌 BAĞLANDI", bg='#27ae60', state=tk.DISABLED)
        self.btn_start.config(state=tk.NORMAL, bg='#2ecc71')
        
        # Sinyal kalitesi izleme başlat
        self.monitoring = True
        self.monitor_signal()
    
    def monitor_signal(self):
        """Sinyal kalitesini izle (bağlantı sonrası, tahmin öncesi)"""
        if not self.monitoring or self.running:
            return
        
        if self.connector:
            signal_quality = self.connector.poor_signal
            
            if signal_quality == 0:
                status = "✅ Mükemmel"
                color = '#00ff88'
            elif signal_quality < 50:
                status = "👍 İyi"
                color = '#00ff88'
            elif signal_quality < 100:
                status = "⚠️ Orta"
                color = '#feca57'
            else:
                status = "❌ Zayıf"
                color = '#ff6b6b'
            
            self.pred_label.config(text=f"📊 Sinyal: {signal_quality}", fg=color)
            self.buffer_label.config(text=f"Sinyal Kalitesi: {status}")
        
        self.root.after(500, self.monitor_signal)
    
    def start_prediction(self):
        """AŞAMA 2: Tahmine başla"""
        if self.connector is None:
            messagebox.showwarning("Uyarı", "Önce cihaza bağlanın!")
            return
        
        self.monitoring = False  # Sinyal izlemeyi durdur
        self.running = True
        
        # Butonları güncelle
        self.btn_start.config(text="▶️ ÇALIŞIYOR", bg='#27ae60', state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL, bg='#e74c3c')
        self.status_label.config(text="🔴 Tahmin yapılıyor...", fg='#ff6b6b')
        
        self.update_loop()
    
    def stop_prediction(self):
        """Tahmini durdur"""
        self.running = False
        self.monitoring = False
        
        if self.connector:
            self.connector.disconnect()
            self.connector = None
        
        # Butonları sıfırla
        self.btn_connect.config(text="🔌 BAĞLAN", bg='#3498db', state=tk.NORMAL)
        self.btn_start.config(text="▶️ BAŞLAT", bg='#95a5a6', state=tk.DISABLED)
        self.btn_stop.config(bg='#95a5a6', state=tk.DISABLED)
        self.status_label.config(text="⏸️ Durduruldu", fg='#888')
        self.pred_label.config(text="⏳ Bekleniyor...", fg='white')
    
    def update_loop(self):
        if not self.running:
            return
        
        try:
            # Raw sample'ları al
            raw_samples = self.connector.get_raw_samples()
            
            if raw_samples:
                # FFT hesapla
                self.engine.process_raw_samples(raw_samples)
            
            # FFT değerlerini güncelle
            fft_values = self.engine.get_fft_values()
            for band in BAND_NAMES:
                self.fft_history[band].append(fft_values.get(band, 0))
            
            # Buffer durumu
            status = self.engine.get_buffer_status()
            self.buffer_label.config(text=f"Raw: {status['raw']:.0f}% | Features: {status['features']}/{status['max_features']}")
            self.sample_label.config(text=f"Samples: {status['samples']} | Artifacts: {status['artifacts']}")
            
            # Tahmin
            label, confidence, class_probs = self.engine.predict()
            
            if label:
                color = COLORS.get(label, 'white')
                self.pred_label.config(text=f"🎯 {label.upper()}", fg=color)
                
                for class_name in ['yukarı', 'asagı', 'araba']:
                    prob = class_probs.get(class_name, 0) * 100
                    self.conf_bars[class_name]['value'] = prob
                    self.conf_bars[f"{class_name}_label"].config(text=f"{prob:.1f}%")
            
            # Grafik güncelle
            self.update_plots(fft_values)
            
        except Exception as e:
            print(f"Güncelleme hatası: {e}")
        
        self.root.after(50, self.update_loop)
    
    def update_plots(self, fft_values):
        # Bant grafiği
        self.ax_bands.clear()
        self.ax_bands.set_facecolor(COLORS['bg'])
        
        values = [np.log1p(fft_values.get(band, 0)) for band in BAND_NAMES]
        self.ax_bands.bar(range(len(BAND_NAMES)), values, color=BAND_COLORS)
        self.ax_bands.set_xticks(range(len(BAND_NAMES)))
        self.ax_bands.set_xticklabels(['δ', 'θ', 'Lα', 'Hα', 'Lβ', 'Hβ', 'Lγ', 'Mγ'], color='white')
        self.ax_bands.set_ylim(0, 20)  # Y eksenini 20'de sabitle
        self.ax_bands.set_title('FFT Bant Güçleri (log)', color='white')
        self.ax_bands.tick_params(colors='white')
        
        # Zaman serisi
        self.ax_time.clear()
        self.ax_time.set_facecolor(COLORS['bg'])
        
        for i, band in enumerate(['Delta', 'Theta', 'Low Alpha', 'Low Beta']):
            if len(self.fft_history[band]) > 0:
                data = [np.log1p(v) for v in self.fft_history[band]]
                self.ax_time.plot(data, color=BAND_COLORS[BAND_NAMES.index(band)],
                                 label=band, alpha=0.8)
        
        self.ax_time.set_title('Zaman Serisi (log)', color='white')
        self.ax_time.legend(loc='upper right', fontsize=8)
        self.ax_time.tick_params(colors='white')
        
        self.canvas.draw()
    
    def on_closing(self):
        self.stop_prediction()
        self.root.destroy()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LSTM+CNN GUI (Raw → FFT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  python realtime_gui.py --simulation                 # Test için simülasyon (seq64)
  python realtime_gui.py --simulation --model seq96   # seq96 ile simülasyon
  python realtime_gui.py --thinkgear --model seq96    # ThinkGear + seq96
  python realtime_gui.py --port COM5 --model seq128   # Seri port + seq128
  
  python realtime_gui.py --list-models                # Mevcut modelleri listele
        """
    )
    
    # Model seçimi
    parser.add_argument('--model', default='seq64', choices=list(AVAILABLE_MODELS.keys()),
                       help='Kullanılacak model (varsayılan: seq64)')
    parser.add_argument('--list-models', action='store_true',
                       help='Mevcut modelleri listele ve çık')
    
    # Bağlantı modu
    parser.add_argument('--simulation', action='store_true', help='Simülasyon modu')
    parser.add_argument('--thinkgear', action='store_true', help='ThinkGear Connector kullan')
    parser.add_argument('--port', default='COM5', help='COM port (seri port modu için)')
    args = parser.parse_args()
    
    # Model listesi göster
    if args.list_models:
        print("\n" + "=" * 60)
        print("📦 MEVCUT MODELLER")
        print("=" * 60)
        for key, info in AVAILABLE_MODELS.items():
            model_path = os.path.join(SCRIPT_DIR, info['model'])
            exists = "✅" if os.path.exists(model_path) else "❌"
            print(f"\n{exists} {key}: {info['name']}")
            print(f"   {info['description']}")
        print()
        return
    
    root = tk.Tk()
    app = RealtimeGUI(root, use_simulation=args.simulation, port=args.port, 
                      use_thinkgear=args.thinkgear, model_name=args.model)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
