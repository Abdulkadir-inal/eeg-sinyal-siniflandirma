#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM+CNN Hibrit Model - GUI ile Canlı Tahmin (Raw EEG → FFT)
============================================================

Raw EEG sinyalinden FFT hesaplar (eğitimle aynı pipeline).
MindWave'in kendi FFT'sini DEĞİL, bizim hesapladığımız FFT'yi kullanır.

Pipeline:
    Raw EEG → Notch (50Hz) → Bandpass (0.5-50Hz) → FFT → Model

Kullanım:
    python realtime_gui.py --simulation   (Test için)
    python realtime_gui.py --port COM5    (Gerçek cihaz)
"""

import os
import sys
import time
import json
import pickle
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
    
    def __init__(self, model_path, scaler_path, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
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
    def __init__(self, root, use_simulation=False, port=None):
        self.root = root
        self.root.title("🧠 LSTM+CNN Canlı Tahmin (Raw → FFT)")
        self.root.geometry("1200x800")
        self.root.configure(bg=COLORS['bg'])
        
        self.use_simulation = use_simulation
        self.port = port
        self.running = False
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
        
        # Pipeline bilgisi
        tk.Label(top_frame, text="Pipeline: Raw EEG → Filtre → FFT → Model",
                bg=COLORS['panel'], fg='#888', font=('Segoe UI', 9)).pack(pady=(5,0))
        
        self.pred_label = tk.Label(top_frame, text="⏳ Bekleniyor...",
                                   font=('Segoe UI', 48, 'bold'),
                                   bg=COLORS['panel'], fg='white')
        self.pred_label.pack(pady=10)
        
        # Güven barları
        self.conf_frame = tk.Frame(top_frame, bg=COLORS['panel'])
        self.conf_frame.pack(fill=tk.X, padx=50)
        
        self.conf_bars = {}
        for class_name in ['yukarı', 'asagı', 'araba']:
            frame = tk.Frame(self.conf_frame, bg=COLORS['panel'])
            frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
            
            display_name = class_name.upper()
            color = COLORS.get(class_name, 'white')
            tk.Label(frame, text=display_name, bg=COLORS['panel'],
                    fg=color, font=('Segoe UI', 12, 'bold')).pack()
            
            bar = ttk.Progressbar(frame, length=150, mode='determinate')
            bar.pack(pady=5)
            self.conf_bars[class_name] = bar
            
            pct_label = tk.Label(frame, text="0%", bg=COLORS['panel'],
                                fg='white', font=('Segoe UI', 10))
            pct_label.pack()
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
        
        # Başlat butonu
        self.btn_start = tk.Button(bottom_frame, text="▶️ BAŞLAT",
                                   font=('Segoe UI', 14, 'bold'),
                                   bg='#2ecc71', fg='white',
                                   width=15, height=2,
                                   command=self.toggle_prediction)
        self.btn_start.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Mod seçimi
        mode_frame = tk.Frame(bottom_frame, bg=COLORS['panel'])
        mode_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(mode_frame, text="Mod:", bg=COLORS['panel'],
                fg='white', font=('Segoe UI', 11)).pack(side=tk.LEFT)
        
        self.mode_var = tk.StringVar(value='simulation' if self.use_simulation else 'real')
        
        tk.Radiobutton(mode_frame, text="Simülasyon", variable=self.mode_var,
                      value='simulation', bg=COLORS['panel'], fg='white',
                      selectcolor=COLORS['bg'], font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Radiobutton(mode_frame, text="Gerçek Cihaz", variable=self.mode_var,
                      value='real', bg=COLORS['panel'], fg='white',
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
        model_path = os.path.join(SCRIPT_DIR, 'best_model.pth')
        scaler_path = os.path.join(SCRIPT_DIR, 'scaler.pkl')
        config_path = os.path.join(SCRIPT_DIR, 'config.json')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, config_path]):
            self.status_label.config(text="⚠️ Model bulunamadı!", fg='red')
            messagebox.showwarning("Uyarı", 
                "Model dosyaları bulunamadı!\n\n"
                "Önce şunları çalıştırın:\n"
                "1. python data_preprocess.py\n"
                "2. python train_model.py")
            return False
        
        self.engine = PredictionEngine(model_path, scaler_path, config_path)
        self.status_label.config(text=f"✅ Model yüklendi (Acc: {self.engine.val_acc:.1f}%)", fg='#00ff88')
        return True
    
    def toggle_prediction(self):
        if not self.running:
            self.start_prediction()
        else:
            self.stop_prediction()
    
    def start_prediction(self):
        if self.engine is None:
            if not self.load_model():
                return
        
        if self.mode_var.get() == 'simulation':
            self.connector = SimulatedMindWave()
            self.connector.connect()
        else:
            try:
                from realtime_predict import MindWaveRawConnector
                port = self.port_entry.get()
                self.connector = MindWaveRawConnector(port)
                if not self.connector.connect():
                    messagebox.showerror("Hata", f"MindWave'e bağlanılamadı: {port}")
                    return
                self.connector.start()
            except Exception as e:
                messagebox.showerror("Hata", f"Bağlantı hatası: {e}")
                return
        
        self.running = True
        self.btn_start.config(text="⏹️ DURDUR", bg='#e74c3c')
        self.status_label.config(text="🔴 Çalışıyor...", fg='#ff6b6b')
        
        self.update_loop()
    
    def stop_prediction(self):
        self.running = False
        
        if self.connector:
            self.connector.disconnect()
            self.connector = None
        
        self.btn_start.config(text="▶️ BAŞLAT", bg='#2ecc71')
        self.status_label.config(text="⏸️ Durduruldu", fg='#888')
    
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
    import argparse
    parser = argparse.ArgumentParser(description='LSTM+CNN GUI (Raw → FFT)')
    parser.add_argument('--simulation', action='store_true', help='Simülasyon modu')
    parser.add_argument('--port', default='COM5', help='COM port')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = RealtimeGUI(root, use_simulation=args.simulation, port=args.port)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
