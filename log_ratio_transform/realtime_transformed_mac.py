#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Log Transform + Oran Formülleri ile Gerçek Zamanlı EEG Tahmin (macOS)
========================================================================

macOS için optimize edilmiş versiyon.

Bu script:
1. MindWave'den Raw EEG alır (512 Hz)
2. Sinyal filtreleme yapar (Notch 50Hz, Bandpass 0.5-50Hz)
3. FFT ile bant güçleri hesaplar
4. Log Transform + Oran Formülleri uygular (9 → 17 özellik)
5. TCN model ile tahmin yapar (%99.43 accuracy)

🍎 macOS Özel Notlar:
    - Tuş kontrolü için Accessibility izni gerekir
    - System Settings > Privacy & Security > Accessibility
    - Terminal'i listeye ekleyin

Kullanım:
    python3 realtime_transformed_mac.py

Gereksinimler:
    pip install torch numpy scipy pynput
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
import platform

# macOS kontrolü
if platform.system() != 'Darwin':
    print("⚠️ Bu script macOS için optimize edilmiştir.")
    print(f"   Mevcut sistem: {platform.system()}")
    response = input("   Yine de devam etmek istiyor musunuz? (y/n): ")
    if response.lower() not in ['y', 'yes', 'e', 'evet']:
        sys.exit(0)

# Tuş kontrolü
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
    print("✅ pynput yüklü")
    print("💡 macOS İpucu: Accessibility izni gerekiyorsa:")
    print("   System Settings > Privacy & Security > Accessibility")
    print("   Terminal veya Python'u listeye ekleyin\n")
except ImportError:
    PYNPUT_AVAILABLE = False
    print("⚠️ pynput bulunamadı. Tuş kontrolü devre dışı.")
    print("   Yüklemek için: pip3 install pynput\n")

# SciPy (filtreleme için)
try:
    from scipy import signal as scipy_signal
except ImportError:
    print("❌ SciPy kurulu değil!")
    print("   Kurulum: pip3 install scipy")
    sys.exit(1)

# PyTorch
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("❌ PyTorch kurulu değil!")
    print("   Kurulum: pip3 install torch")
    sys.exit(1)


# ============================================================================
# AYARLAR
# ============================================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Sinyal işleme parametreleri
SAMPLING_RATE = 512
FFT_WINDOW_SIZE = 512
MODEL_WINDOW = 128

# Filtre parametreleri
NOTCH_FREQ = 50
NOTCH_Q = 30
LOWCUT = 0.5
HIGHCUT = 50
FILTER_ORDER = 4

ARTIFACT_THRESHOLD = 500

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

# macOS için MPS (Metal Performance Shaders) desteği
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Apple Silicon GPU (MPS) bulundu!")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🎮 CUDA GPU bulundu!")
else:
    DEVICE = torch.device("cpu")
    print("💻 CPU kullanılacak")

LABELS = ['araba', 'aşağı', 'yukarı']


# ============================================================================
# SİNYAL FİLTRELEME
# ============================================================================

class SignalProcessor:
    """EEG sinyal işleme sınıfı"""
    
    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs
        self.notch_b, self.notch_a = self._create_notch_filter()
        self.bandpass_b, self.bandpass_a = self._create_bandpass_filter()
    
    def _create_notch_filter(self):
        """50 Hz Notch filtre oluştur"""
        nyq = self.fs / 2
        w0 = NOTCH_FREQ / nyq
        return scipy_signal.iirnotch(w0, NOTCH_Q)
    
    def _create_bandpass_filter(self):
        """Bandpass filtre oluştur (0.5-50 Hz)"""
        nyq = self.fs / 2
        low = LOWCUT / nyq
        high = HIGHCUT / nyq
        return scipy_signal.butter(FILTER_ORDER, [low, high], btype='band')
    
    def filter_signal(self, raw_samples):
        """Raw EEG sinyalini filtrele"""
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
        """FFT ile frekans bant güçlerini hesapla"""
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
        """Raw EEG → Filtreleme → FFT"""
        filtered = self.filter_signal(raw_samples)
        band_powers = self.calculate_fft_bands(filtered)
        return band_powers


# ============================================================================
# TRANSFORMASYON FONKSİYONLARI (Log + Oran)
# ============================================================================

def apply_log_transform(data):
    """Log transform: log1p(x) = log(1 + x)"""
    return np.sign(data) * np.log1p(np.abs(data))


def calculate_band_ratios(window):
    """8 oran özelliği hesapla"""
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
    """Window transformasyonu (128, 9) → (128, 17)"""
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
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                       dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                       dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


# ============================================================================
# THINKGEAR BAĞLANTISI
# ============================================================================

class ThinkGearConnector:
    """ThinkGear Connector'a bağlanır (macOS uyumlu)"""
    
    def __init__(self, host='127.0.0.1', port=13854):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""
        self.raw_buffer = deque(maxlen=FFT_WINDOW_SIZE * 2)
        self.poor_signal = 200
        self.raw_count = 0
    
    def connect(self):
        """ThinkGear Connector'a bağlan"""
        try:
            print(f"🔵 ThinkGear Connector'a bağlanılıyor: {self.host}:{self.port}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.host, self.port))
            
            self.sock.send(b'{"enableRawOutput": true, "format": "Json"}\n')
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(0.05)
            
            print("✅ Bağlantı başarılı!")
            print("📡 Raw EEG çıktısı: AKTİF (512 Hz)")
            return True
            
        except ConnectionRefusedError:
            print("❌ ThinkGear Connector çalışmıyor!")
            print("\n💡 macOS'ta Çözüm:")
            print("   1. ThinkGear Connector'ı indirin:")
            print("      http://developer.neurosky.com/docs/doku.php?id=thinkgear_connector_tgc")
            print("   2. Uygulamayı açın")
            print("   3. MindWave'i Bluetooth ile eşleştirin")
            print("   4. Bu scripti tekrar çalıştırın")
            return False
        except Exception as e:
            print(f"❌ Bağlantı hatası: {e}")
            return False
    
    def disconnect(self):
        """Bağlantıyı kapat"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        print("🔌 Bağlantı kapatıldı")
    
    def read_data(self):
        """ThinkGear'dan veri oku"""
        if not self.sock:
            return None
        
        try:
            data = self.sock.recv(16384).decode('utf-8')
            if not data:
                return None
            
            self.buffer += data
            lines = self.buffer.split('\r')
            self.buffer = lines[-1]
            
            got_raw = False
            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parsed = json.loads(line)
                    
                    if 'rawEeg' in parsed:
                        self.raw_buffer.append(parsed['rawEeg'])
                        self.raw_count += 1
                        got_raw = True
                    
                    if 'poorSignalLevel' in parsed:
                        self.poor_signal = parsed['poorSignalLevel']
                    
                except json.JSONDecodeError:
                    continue
            
            return 'raw' if got_raw else None
            
        except socket.timeout:
            return None
        except Exception:
            return None
    
    def get_raw_samples(self, n_samples):
        """Son n sample'ı al"""
        if len(self.raw_buffer) < n_samples:
            return None
        return list(self.raw_buffer)[-n_samples:]
    
    def get_buffer_size(self):
        return len(self.raw_buffer)


# ============================================================================
# GERÇEK ZAMANLI TAHMİN
# ============================================================================

class RealtimeTransformedPredictor:
    """Log Transform + Oran Formülleri ile gerçek zamanlı tahmin (macOS)"""
    
    CONFIDENCE_THRESHOLD = 0.70
    
    def __init__(self, model_window=MODEL_WINDOW, fft_window=FFT_WINDOW_SIZE, prediction_interval=0.25):
        self.model_window = model_window
        self.fft_window = fft_window
        self.prediction_interval = prediction_interval
        
        self.device = DEVICE
        self.model = None
        self.signal_processor = SignalProcessor()
        self.fft_buffer = deque(maxlen=model_window)
        self.thinkgear = ThinkGearConnector()
        self.scaler = None
        
        self.calibration_mean = None
        self.calibration_std = None
        self.is_calibrated = False
        
        self.recording = False
        self.should_quit = False
        
        self.predictions = {label: 0 for label in LABELS}
        self.total_predictions = 0
        self.uncertain_count = 0
    
    def load_model(self):
        """Model ve scaler'ı yükle"""
        print("\n📂 Model yükleniyor...")
        
        scaler_path = os.path.join(MODEL_DIR, 'scaler_transformed.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"   ✅ Scaler yüklendi")
        else:
            print(f"   ⚠️ Scaler bulunamadı: {scaler_path}")
        
        model_path = os.path.join(MODEL_DIR, 'best_model_transformed.pth')
        if not os.path.exists(model_path):
            print(f"   ❌ Model bulunamadı: {model_path}")
            return False
        
        try:
            self.model = TCN_Model(input_channels=17, num_classes=3).to(self.device)
            
            # macOS için weights_only parametresi eski PyTorch versiyonlarında olmayabilir
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location=self.device)
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print(f"   ✅ TCN Model yüklendi (%99.43 accuracy)")
            print(f"   ⚡ Cihaz: {self.device}")
            
            if self.device.type == 'mps':
                print(f"   🍎 Apple Silicon GPU (Metal)")
            elif self.device.type == 'cuda':
                print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Model yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess(self, fft_window_data):
        """FFT verilerini transform et ve normalize et"""
        x = np.array(fft_window_data, dtype=np.float32)
        x_transformed = transform_window(x)
        
        if self.is_calibrated and self.calibration_mean is not None:
            x_flat = x_transformed.flatten()
            x_flat = x_flat - self.calibration_mean
            x_flat = x_flat / (self.calibration_std + 1e-8)
            x_transformed = x_flat.reshape(x_transformed.shape)
        
        if self.scaler is not None:
            x_flat = x_transformed.reshape(1, -1)
            x_normalized = self.scaler.transform(x_flat)
            x_transformed = x_normalized.reshape(self.model_window, 17)
        
        return torch.FloatTensor(x_transformed).unsqueeze(0).to(self.device)
    
    def predict(self, fft_window_data):
        """Tahmin yap"""
        if self.model is None:
            return None, None, 0
        
        start_time = time.time()
        
        with torch.no_grad():
            x = self.preprocess(fft_window_data)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            # MPS için sync gerekli değil ama CUDA için gerekli
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = (time.time() - start_time) * 1000
            
            return LABELS[predicted.item()], confidence.item(), inference_time
    
    def setup_keyboard_listener(self):
        """Klavye dinleyicisini başlat (macOS uyumlu)"""
        if not PYNPUT_AVAILABLE:
            print("⚠️ pynput yüklü değil, tuş kontrolü devre dışı")
            print("   Manuel kontrol: Ctrl+C ile durdurun")
            return
        
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char:
                    if key.char.lower() == 's':
                        self.recording = True
                        print("\n🔴 TAHMİN BAŞLADI (S tuşu)")
                    elif key.char.lower() == 'e':
                        self.recording = False
                        print("\n⏸️  TAHMİN DURAKLATILDI (E tuşu)")
                    elif key.char.lower() == 'q':
                        self.should_quit = True
                        print("\n⛔ ÇIKIŞ (Q tuşu)")
                        return False
            except AttributeError:
                if key == keyboard.Key.space:
                    self.recording = not self.recording
                    print(f"\n{'🔴 TAHMİN AKTİF' if self.recording else '⏸️  TAHMİN PASIF'} (SPACE)")
        
        try:
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
            print("✅ Tuş kontrolü aktif: [S]tart, [E]nd, [SPACE]toggle, [Q]uit")
        except Exception as e:
            print(f"⚠️ Tuş kontrolü başlatılamadı: {e}")
            print("   macOS'ta Accessibility izni gerekebilir")
            print("   System Settings > Privacy & Security > Accessibility")
    
    def calibrate(self, duration=15):
        """Kullanıcıya özel kalibrasyon"""
        print("\n" + "=" * 60)
        print("🎯 KALİBRASYON AŞAMASI")
        print("=" * 60)
        print(f"⏱️  {duration} saniye boyunca:")
        print("   • Rahat oturun")
        print("   • Gözlerinizi kapatın")
        print("   • Hiçbir şey düşünmeyin (nötr durum)")
        print("-" * 60)
        
        input("Hazır olduğunuzda ENTER'a basın...")
        
        print("\n🔴 KALİBRASYON BAŞLADI...")
        
        calibration_data = []
        start_time = time.time()
        last_raw_count = 0
        raw_samples_for_fft = 256
        
        while (time.time() - start_time) < duration:
            result = self.thinkgear.read_data()
            
            if result == 'raw':
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                sig = "✅" if self.thinkgear.poor_signal < 50 else f"⚠️({self.thinkgear.poor_signal})"
                print(f"\r⏳ Kalan: {remaining:.1f}s | Veri: {len(calibration_data)} | {sig}   ", end='')
                
                raw_buffer_size = self.thinkgear.get_buffer_size()
                new_samples = self.thinkgear.raw_count - last_raw_count
                
                if raw_buffer_size >= self.fft_window and new_samples >= raw_samples_for_fft:
                    last_raw_count = self.thinkgear.raw_count
                    raw_samples = self.thinkgear.get_raw_samples(self.fft_window)
                    band_powers = self.signal_processor.process_raw_to_fft(raw_samples)
                    calibration_data.append([0] + band_powers)
            
            time.sleep(0.001)
        
        if len(calibration_data) < 10:
            print("\n\n❌ Yeterli kalibrasyon verisi toplanamadı!")
            return False
        
        cal_array = np.array(calibration_data, dtype=np.float32)
        if len(cal_array) >= self.model_window:
            cal_transformed = transform_window(cal_array[:self.model_window])
            self.calibration_mean = np.mean(cal_transformed.flatten())
            self.calibration_std = np.std(cal_transformed.flatten())
        else:
            self.calibration_mean = np.mean(cal_array.flatten())
            self.calibration_std = np.std(cal_array.flatten())
        
        self.is_calibrated = True
        
        print("\n\n✅ KALİBRASYON TAMAMLANDI")
        print(f"   📊 {len(calibration_data)} FFT frame toplandı")
        print(f"   📈 Baseline: {self.calibration_mean:.2f} (std: {self.calibration_std:.2f})")
        
        return True
    
    def run(self):
        """Ana döngü"""
        print("\n" + "=" * 60)
        print("🧠 LOG TRANSFORM + ORAN FORMÜLLERİ (macOS)")
        print("   Gerçek Zamanlı EEG Tahmin (%99.43 accuracy)")
        print("=" * 60)
        
        if not self.load_model():
            return
        
        print("\n" + "-" * 60)
        if not self.thinkgear.connect():
            return
        
        print("\n" + "=" * 60)
        do_cal = input("Kalibrasyon yapmak ister misiniz? (y/n) [önerilen]: ").strip().lower()
        
        if do_cal in ['y', 'yes', 'e', 'evet', '']:
            if not self.calibrate():
                return
        else:
            print("⚠️ Kalibrasyon atlandı")
        
        if PYNPUT_AVAILABLE:
            self.setup_keyboard_listener()
        else:
            self.recording = True
        
        print("\n" + "=" * 60)
        print(f"📊 Model: TCN (%99.43 accuracy)")
        print(f"🔧 Özellik: 17 (9 FFT + 8 Oran)")
        print(f"⚡ Cihaz: {self.device}")
        print(f"🎯 Sınıflar: {', '.join(LABELS)}")
        print(f"🎚️  Kalibrasyon: {'✅ Aktif' if self.is_calibrated else '❌ Yok'}")
        print("=" * 60)
        print("\n💡 MindWave'i takın!")
        if PYNPUT_AVAILABLE:
            print("🎹 Tuşlar: [S]başla [E]dur [SPACE]toggle [Q]çık")
        print("⏸️  Durdurmak için Ctrl+C")
        print("-" * 60)
        
        print("\n⏳ FFT buffer dolduruluyor...")
        last_raw_count = 0
        raw_samples_for_fft = 256
        
        try:
            while not self.should_quit:
                result = self.thinkgear.read_data()
                
                if result == 'raw':
                    raw_buffer_size = self.thinkgear.get_buffer_size()
                    new_samples = self.thinkgear.raw_count - last_raw_count
                    
                    if raw_buffer_size >= self.fft_window and new_samples >= raw_samples_for_fft:
                        last_raw_count = self.thinkgear.raw_count
                        
                        raw_samples = self.thinkgear.get_raw_samples(self.fft_window)
                        band_powers = self.signal_processor.process_raw_to_fft(raw_samples)
                        self.fft_buffer.append([0] + band_powers)
                        
                        if len(self.fft_buffer) >= self.model_window:
                            if self.recording:
                                window_data = list(self.fft_buffer)[-self.model_window:]
                                label, confidence, inference_time = self.predict(window_data)
                                
                                self.total_predictions += 1
                                emoji = {"araba": "🚗", "yukarı": "⬆️", "aşağı": "⬇️"}.get(label, "❓")
                                
                                if confidence >= self.CONFIDENCE_THRESHOLD:
                                    self.predictions[label] += 1
                                    sig = "✅" if self.thinkgear.poor_signal < 50 else f"⚠️"
                                    print(f"\r[{self.total_predictions:4d}] {emoji} {label:8s} | "
                                          f"Güven: {confidence*100:5.1f}% | "
                                          f"{inference_time:.1f}ms | {sig}   ", end='')
                                else:
                                    self.uncertain_count += 1
                                    print(f"\r[{self.total_predictions:4d}] ❓ belirsiz | "
                                          f"Güven: {confidence*100:5.1f}%   ", end='')
                            else:
                                print(f"\r⏸️  Bekleniyor... Buffer: {len(self.fft_buffer)}/{self.model_window}   ", end='')
                        else:
                            print(f"\r⏳ Buffer: {len(self.fft_buffer)}/{self.model_window}   ", end='')
                
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Durduruldu.")
        finally:
            self.thinkgear.disconnect()
            self.print_stats()
    
    def print_stats(self):
        """İstatistikleri yazdır"""
        print("\n" + "=" * 60)
        print("📊 İSTATİSTİKLER")
        print("=" * 60)
        print(f"   Toplam tahmin: {self.total_predictions}")
        print(f"   Belirsiz: {self.uncertain_count}")
        print("\n   Sınıf dağılımı:")
        for label, count in self.predictions.items():
            if self.total_predictions > 0:
                pct = (count / self.total_predictions) * 100
                bar = "█" * int(pct / 5)
                print(f"      {label:8s}: {count:4d} ({pct:5.1f}%) {bar}")
        print("=" * 60)


# ============================================================================
# DEMO MODU
# ============================================================================

def demo_mode():
    """ThinkGear olmadan demo test"""
    print("\n" + "=" * 60)
    print("🧪 DEMO MODU - Rastgele Veri ile Test (macOS)")
    print("=" * 60)
    
    predictor = RealtimeTransformedPredictor()
    
    if not predictor.load_model():
        return
    
    print("\n🎲 Rastgele veri ile 10 tahmin yapılıyor...\n")
    
    for i in range(10):
        window = np.random.randn(MODEL_WINDOW, 9) * 50000 + 100000
        window = np.abs(window)
        
        label, confidence, inference_time = predictor.predict(window)
        
        emoji = {"araba": "🚗", "yukarı": "⬆️", "aşağı": "⬇️"}.get(label, "❓")
        print(f"[{i+1:2d}] {emoji} {label:8s} | Güven: {confidence*100:5.1f}% | {inference_time:.1f}ms")
    
    print("\n✅ Demo tamamlandı!")


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("🧠 LOG TRANSFORM + ORAN FORMÜLLERİ (macOS)")
    print("   Gerçek Zamanlı EEG Tahmin Sistemi")
    print("=" * 60)
    print(f"🍎 Platform: {platform.system()} {platform.release()}")
    print(f"📱 Device: {DEVICE}")
    print(f"📂 Model: {MODEL_DIR}")
    
    print("\n📋 Seçenekler:")
    print("   1. Canlı Tahmin (ThinkGear Connector gerekli)")
    print("   2. Demo Modu (rastgele veri ile test)")
    print("   3. Çıkış")
    
    try:
        choice = input("\nSeçiminiz (1/2/3): ").strip()
        
        if choice == "1":
            predictor = RealtimeTransformedPredictor()
            predictor.run()
        elif choice == "2":
            demo_mode()
        else:
            print("Çıkış...")
            
    except KeyboardInterrupt:
        print("\n\nÇıkış...")


if __name__ == "__main__":
    main()
