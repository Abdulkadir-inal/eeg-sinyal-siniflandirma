#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 FFT Tabanlı Gerçek Zamanlı EEG Tahmin Sistemi
================================================

MindWave'den sadece Raw EEG alır, filtreleme ve FFT'yi
bilgisayarda hesaplayarak daha hızlı tahmin yapar.

NeuroSky EEG Power: 1 Hz (saniyede 1 tahmin)
Bu sistem: ~2-4 Hz (saniyede 2-4 tahmin)

Kullanım:
    1. Windows'ta: python thinkgear_proxy.py
    2. WSL2'de:    python realtime_fft_predict.py

Özellikler:
    - Raw EEG'den kendi FFT hesaplama
    - Sinyal filtreleme (Notch 50Hz, Bandpass 0.5-50Hz)
    - Artifact rejection
    - CUDA/GPU desteği
    - Hızlı tahmin (~250-500ms aralıklarla)
"""

import os
import sys
import time
import socket
import json
import subprocess
import numpy as np
from collections import deque
from datetime import datetime
from scipy import signal as scipy_signal

# PyTorch
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("❌ PyTorch kurulu değil!")
    print("   Kurulum: pip install torch")
    sys.exit(1)


# ============================================================================
# SİNYAL İŞLEME PARAMETRELERİ
# ============================================================================

SAMPLING_RATE = 512  # Hz
FFT_WINDOW_SIZE = 512  # 1 saniyelik FFT penceresi

# Filtre parametreleri (NeuroSky benzeri)
NOTCH_FREQ = 50      # Hz (Türkiye elektrik şebekesi)
NOTCH_Q = 30         # Notch filter kalite faktörü
LOWCUT = 0.5         # Hz (EEG alt frekans)
HIGHCUT = 50         # Hz (EEG üst frekans)
FILTER_ORDER = 4     # Butterworth filter order

# Artifact rejection
ARTIFACT_THRESHOLD = 500  # µV üzeri değerler artifact sayılır

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
# SİNYAL FİLTRELEME FONKSİYONLARI
# ============================================================================

class SignalProcessor:
    """EEG sinyal işleme sınıfı"""
    
    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs
        
        # Filtreleri önceden oluştur (hız için)
        self.notch_b, self.notch_a = self._create_notch_filter()
        self.bandpass_b, self.bandpass_a = self._create_bandpass_filter()
        
        # Filtre durumları (gerçek zamanlı filtreleme için)
        self.notch_zi = None
        self.bandpass_zi = None
    
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
    
    def reset_filter_states(self, n_samples=1):
        """Filtre durumlarını sıfırla"""
        self.notch_zi = scipy_signal.lfilter_zi(self.notch_b, self.notch_a) * 0
        self.bandpass_zi = scipy_signal.lfilter_zi(self.bandpass_b, self.bandpass_a) * 0
    
    def filter_signal(self, raw_samples):
        """
        Raw EEG sinyalini filtrele
        1. DC offset kaldır
        2. Artifact temizle
        3. Notch filtre (50 Hz)
        4. Bandpass filtre (0.5-50 Hz)
        """
        samples = np.array(raw_samples, dtype=np.float64)
        
        # 1. DC offset kaldır
        samples = samples - np.mean(samples)
        
        # 2. Artifact'ları temizle (basit threshold)
        artifact_mask = np.abs(samples) > ARTIFACT_THRESHOLD
        if np.any(artifact_mask):
            # Artifact noktalarını medyan ile değiştir
            good_samples = samples[~artifact_mask]
            if len(good_samples) > 0:
                median_val = np.median(good_samples)
                samples[artifact_mask] = median_val
        
        # 3. Notch filtre (50 Hz elektrik gürültüsü)
        samples = scipy_signal.filtfilt(self.notch_b, self.notch_a, samples)
        
        # 4. Bandpass filtre (0.5-50 Hz EEG bandı)
        samples = scipy_signal.filtfilt(self.bandpass_b, self.bandpass_a, samples)
        
        return samples
    
    def calculate_fft_bands(self, filtered_samples):
        """
        Filtrelenmiş sinyalden FFT ile frekans bant güçlerini hesapla
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
        band_powers = []
        for band_name in ['Delta', 'Theta', 'Low Alpha', 'High Alpha', 
                          'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']:
            low_freq, high_freq = FREQUENCY_BANDS[band_name]
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_powers.append(np.sum(power_spectrum[mask]))
        
        return band_powers
    
    def process_raw_to_fft(self, raw_samples):
        """
        Raw EEG → Filtreleme → FFT (tek adımda)
        Returns: [Delta, Theta, LowAlpha, HighAlpha, LowBeta, HighBeta, LowGamma, MidGamma]
        """
        # Filtrele
        filtered = self.filter_signal(raw_samples)
        
        # FFT hesapla
        band_powers = self.calculate_fft_bands(filtered)
        
        return band_powers


# ============================================================================
# MODEL TANIMLARI (FFT modeli için)
# ============================================================================

class TCN_Block(nn.Module):
    """Temporal Convolutional Network Bloğu"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCN_Block, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        if residual.size(2) != out.size(2):
            residual = residual[:, :, :out.size(2)]
        return self.relu(out + residual)


class TCN_EEG_Model(nn.Module):
    """TCN Model - FFT için (9 kanal: Electrode + 8 bant)"""
    def __init__(self, input_channels=9, num_classes=3, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(TCN_EEG_Model, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TCN_Block(in_ch, out_ch, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_channels[-1], 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    """Transformer için Pozisyonel Encoding"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEEG(nn.Module):
    """Transformer Model - FFT için"""
    def __init__(self, input_channels=9, num_classes=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.3):
        super(TransformerEEG, self).__init__()
        self.input_projection = nn.Linear(input_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        self.d_model = d_model
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN_LSTM_Model(nn.Module):
    """CNN + LSTM Hibrit Model"""
    def __init__(self, input_channels=9, num_classes=3):
        super(CNN_LSTM_Model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# THINKGEAR RAW OKUYUCU
# ============================================================================

class ThinkGearRawReader:
    """
    Windows'taki ThinkGear Proxy'den RAW EEG verisi okur.
    NeuroSky EEG Power yerine Raw Electrode değerini alır.
    """
    
    def __init__(self, host=None, port=5555):
        if host is None:
            host = self._find_windows_ip()
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""
        
        # Raw EEG buffer (512 Hz)
        self.raw_buffer = deque(maxlen=FFT_WINDOW_SIZE * 2)  # 2 saniyelik buffer
        
        # Son durum
        self.poor_signal = 200
        self.last_raw_time = 0
        self.raw_count = 0
    
    def _find_windows_ip(self):
        """WSL2'nin Windows gateway IP'sini bul"""
        try:
            result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'default' in line:
                    return line.split()[2]
        except:
            pass
        return '172.31.240.1'
    
    def connect(self):
        """Proxy'ye bağlan"""
        try:
            print(f"🔵 Windows Proxy'ye bağlanılıyor: {self.host}:{self.port}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            
            # TCP optimizasyonları
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(0.05)  # 50ms timeout (çok hızlı)
            
            print("✅ Bağlantı başarılı!")
            return True
            
        except ConnectionRefusedError:
            print(f"❌ Bağlantı reddedildi: {self.host}:{self.port}")
            print("\n💡 Windows'ta proxy'yi başlatın:")
            print("   python thinkgear_proxy.py")
            return False
        except socket.timeout:
            print("❌ Bağlantı zaman aşımı")
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
        """
        ThinkGear verisi oku - sadece Raw EEG'ye odaklan
        Returns: 'raw' if new raw data, None otherwise
        """
        if not self.sock:
            return None
        
        try:
            data = self.sock.recv(16384).decode('utf-8')
            if not data:
                return None
            
            self.buffer += data
            
            # Satır satır JSON parse et
            lines = self.buffer.split('\r')
            self.buffer = lines[-1]
            
            got_raw = False
            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parsed = json.loads(line)
                    
                    # Raw EEG verisi (512 Hz)
                    if 'rawEeg' in parsed:
                        raw_value = parsed['rawEeg']
                        self.raw_buffer.append(raw_value)
                        self.raw_count += 1
                        self.last_raw_time = time.time()
                        got_raw = True
                    
                    # Sinyal kalitesi
                    if 'poorSignalLevel' in parsed:
                        self.poor_signal = parsed['poorSignalLevel']
                    
                except json.JSONDecodeError:
                    continue
            
            return 'raw' if got_raw else None
            
        except socket.timeout:
            return None
        except Exception as e:
            return None
    
    def get_raw_samples(self, n_samples):
        """Son n sample'ı al"""
        if len(self.raw_buffer) < n_samples:
            return None
        return list(self.raw_buffer)[-n_samples:]
    
    def get_buffer_size(self):
        """Buffer boyutunu döndür"""
        return len(self.raw_buffer)


# ============================================================================
# CANLI FFT TAHMİN SİSTEMİ
# ============================================================================

class RealtimeFFTPredictor:
    """
    Raw EEG → FFT → Tahmin pipeline'ı
    NeuroSky'ın 1 Hz limitini aşar!
    """
    
    MODELS = {
        '1': {
            'name': 'TCN (En İyi - %95.70)',
            'class': TCN_EEG_Model,
            'file': 'tcn_model_fft.pth',
            'params': {'input_channels': 9, 'num_classes': 3}
        },
        '2': {
            'name': 'Transformer (%93.49)',
            'class': TransformerEEG,
            'file': 'transformer_model_fft.pth',
            'params': {'input_channels': 9, 'num_classes': 3}
        },
        '3': {
            'name': 'CNN-LSTM (%81.57)',
            'class': CNN_LSTM_Model,
            'file': 'cnn_lstm_model_fft.pth',
            'params': {'input_channels': 9, 'num_classes': 3}
        }
    }
    
    LABELS = ['araba', 'aşağı', 'yukarı']
    
    # Confidence threshold
    CONFIDENCE_THRESHOLD = 0.70  # %70
    
    def __init__(self, model_window=128, fft_window=512, prediction_interval=0.25):
        """
        model_window: Model için pencere boyutu (128 sample = 128 FFT frame)
        fft_window: FFT pencere boyutu (512 sample = 1 saniye)
        prediction_interval: Tahminler arası süre (saniye)
        """
        self.model_window = model_window
        self.fft_window = fft_window
        self.prediction_interval = prediction_interval
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = None
        
        # Signal processor (filtreleme + FFT)
        self.signal_processor = SignalProcessor()
        
        # FFT sonuçları buffer'ı
        self.fft_buffer = deque(maxlen=model_window)
        
        # ThinkGear bağlantısı
        self.thinkgear = ThinkGearRawReader()
        
        # Scaler parametreleri (eğitimden)
        self.scaler_mean = None
        self.scaler_std = None
        
        # İstatistikler
        self.predictions = {label: 0 for label in self.LABELS}
        self.total_predictions = 0
        self.uncertain_count = 0
        self.inference_times = []
        self.fft_times = []
    
    def load_scaler_params(self):
        """Scaler parametrelerini yükle"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(script_dir, 'scaler_params_fft.json')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                params = json.load(f)
            self.scaler_mean = np.array(params['mean'])
            self.scaler_std = np.array(params['std'])
            self.window_size_from_scaler = params.get('window_size', 128)
            print(f"✅ Scaler parametreleri yüklendi ({len(self.scaler_mean)} değer)")
            return True
        else:
            print(f"⚠️ Scaler dosyası bulunamadı: {scaler_path}")
            return False
    
    def select_model(self):
        """Model seçimi"""
        print("\n" + "=" * 60)
        print("🧠 FFT MODEL SEÇİMİ")
        print("=" * 60)
        
        for key, info in self.MODELS.items():
            print(f"   {key}. {info['name']}")
        
        print("   q. Çıkış")
        print("-" * 60)
        
        while True:
            choice = input("Model seçin (1-3): ").strip()
            
            if choice.lower() == 'q':
                return False
            
            if choice in self.MODELS:
                return self.load_model(choice)
            
            print("❌ Geçersiz seçim!")
    
    def load_model(self, choice):
        """Model yükle"""
        info = self.MODELS[choice]
        model_path = info['file']
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, model_path)
        
        if not os.path.exists(full_path):
            print(f"❌ Model dosyası bulunamadı: {full_path}")
            return False
        
        try:
            print(f"\n📥 Model yükleniyor: {info['name']}")
            
            self.model = info['class'](**info['params'])
            state_dict = torch.load(full_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_name = info['name']
            print(f"✅ Model yüklendi!")
            print(f"⚡ Cihaz: {self.device}")
            
            if self.device.type == 'cuda':
                print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess(self, fft_window_data):
        """FFT verilerini model için hazırla"""
        x = np.array(fft_window_data, dtype=np.float32)
        
        # StandardScaler normalizasyonu (flatten edilmiş formda)
        # Eğitimde X_flat = X.reshape(X.shape[0], -1) yapılıyor
        if self.scaler_mean is not None and self.scaler_std is not None:
            x_flat = x.flatten()
            if len(x_flat) == len(self.scaler_mean):
                x_normalized = (x_flat - self.scaler_mean) / np.where(self.scaler_std > 0, self.scaler_std, 1)
                x = x_normalized.reshape(x.shape)
            else:
                # Boyut uyuşmazlığı - per-channel normalize
                for i in range(x.shape[1]):
                    x[:, i] = (x[:, i] - np.mean(x[:, i])) / (np.std(x[:, i]) + 1e-8)
        
        x = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        return x
    
    def predict(self, fft_window_data):
        """Tahmin yap"""
        if self.model is None:
            return None, None, 0
        
        start_time = time.time()
        
        with torch.no_grad():
            x = self.preprocess(fft_window_data)
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            return self.LABELS[predicted.item()], confidence.item(), inference_time
    
    def run(self):
        """Ana döngü"""
        print("\n" + "=" * 60)
        print("🧠 FFT Tabanlı Gerçek Zamanlı EEG Tahmin")
        print("   Raw EEG → Filtreleme → FFT → Tahmin")
        print("=" * 60)
        
        # Scaler yükle
        if not self.load_scaler_params():
            print("⚠️ Scaler olmadan devam ediliyor (düşük performans)")
        
        # Model seç
        if not self.select_model():
            return
        
        # Proxy'ye bağlan
        print("\n" + "-" * 60)
        if not self.thinkgear.connect():
            return
        
        print("\n" + "=" * 60)
        print(f"📊 Model: {self.model_name}")
        print(f"⚡ Cihaz: {self.device}")
        print(f"🎯 Sınıflar: {', '.join(self.LABELS)}")
        print(f"📦 FFT Pencere: {self.fft_window} sample (1 saniye)")
        print(f"📦 Model Pencere: {self.model_window} FFT frame")
        print(f"⏱️ Tahmin Aralığı: {self.prediction_interval}s ({1/self.prediction_interval:.1f} Hz)")
        print("=" * 60)
        print("\n💡 MindWave'i başınıza takın!")
        print("⏸️  Durdurmak için Ctrl+C")
        print("-" * 60)
        
        try:
            last_fft_time = 0
            last_prediction_time = 0
            fft_interval = 1.0 / 4  # 4 Hz FFT hesaplama (her 256 sample'da bir)
            raw_samples_for_fft = 256  # Her 256 sample'da FFT hesapla (overlap)
            
            raw_received = False
            prediction_started = False
            last_raw_count = 0
            
            while True:
                result = self.thinkgear.read_data()
                
                if result == 'raw':
                    raw_received = True
                    
                    # Buffer durumunu göster (tahmin başlamadan önce)
                    if not prediction_started:
                        raw_count = self.thinkgear.get_buffer_size()
                        fft_count = len(self.fft_buffer)
                        signal_status = "✅" if self.thinkgear.poor_signal < 50 else f"⚠️({self.thinkgear.poor_signal})"
                        print(f"\r📦 Raw: {raw_count}/{self.fft_window} | FFT: {fft_count}/{self.model_window} | Sinyal: {signal_status}   ", end='')
                    
                    current_time = time.time()
                    
                    # Yeterli raw sample varsa FFT hesapla
                    raw_buffer_size = self.thinkgear.get_buffer_size()
                    new_samples = self.thinkgear.raw_count - last_raw_count
                    
                    if raw_buffer_size >= self.fft_window and new_samples >= raw_samples_for_fft:
                        last_raw_count = self.thinkgear.raw_count
                        
                        # FFT hesapla
                        fft_start = time.time()
                        raw_samples = self.thinkgear.get_raw_samples(self.fft_window)
                        band_powers = self.signal_processor.process_raw_to_fft(raw_samples)
                        fft_time = (time.time() - fft_start) * 1000
                        self.fft_times.append(fft_time)
                        
                        # FFT sonucunu buffer'a ekle (Electrode=0 + 8 bant)
                        fft_vector = [0] + band_powers  # [Electrode, Delta, Theta, ...]
                        self.fft_buffer.append(fft_vector)
                    
                    # Tahmin zamanı mı?
                    if len(self.fft_buffer) >= self.model_window and (current_time - last_prediction_time) >= self.prediction_interval:
                        last_prediction_time = current_time
                        prediction_started = True
                        
                        # Tahmin yap
                        fft_window_data = list(self.fft_buffer)[-self.model_window:]
                        label, confidence, inference_time = self.predict(fft_window_data)
                        
                        self.inference_times.append(inference_time)
                        
                        if label:
                            avg_fft = sum(self.fft_times[-10:]) / min(len(self.fft_times), 10) if self.fft_times else 0
                            avg_inference = sum(self.inference_times[-10:]) / min(len(self.inference_times), 10)
                            
                            print()
                            print("\n" + "=" * 60)
                            
                            if confidence >= self.CONFIDENCE_THRESHOLD:
                                self.predictions[label] += 1
                                self.total_predictions += 1
                                
                                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | #{self.total_predictions}")
                                print(f"⚡ FFT: {avg_fft:.1f}ms | Inference: {inference_time:.1f}ms")
                                print(f"🎯 {label.upper()} ({confidence*100:.1f}%)")
                            else:
                                self.uncertain_count += 1
                                
                                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | ❓ Belirsiz #{self.uncertain_count}")
                                print(f"⚡ FFT: {avg_fft:.1f}ms | Inference: {inference_time:.1f}ms")
                                print(f"🤔 {label.upper()} ({confidence*100:.1f}% < {self.CONFIDENCE_THRESHOLD*100:.0f}%)")
                            
                            print("-" * 60)
                            
                            for l in self.LABELS:
                                count = self.predictions[l]
                                total = self.total_predictions if self.total_predictions > 0 else 1
                                pct = (count / total * 100)
                                bar = "█" * int(pct / 5)
                                marker = "👉" if (l == label and confidence >= self.CONFIDENCE_THRESHOLD) else "  "
                                print(f"{marker} {l:8}: {bar:<20} {pct:.1f}%")
                            
                            if self.uncertain_count > 0:
                                uncertain_pct = self.uncertain_count / (self.total_predictions + self.uncertain_count) * 100
                                print(f"   {'belirsiz':8}: {'░' * int(uncertain_pct / 5):<20} {uncertain_pct:.1f}%")
                            
                            print("=" * 60)
                
                elif not raw_received:
                    print("\r⏳ Raw EEG bekleniyor...", end='')
                
                time.sleep(0.001)  # 1ms polling
                
        except KeyboardInterrupt:
            print("\n\n⛔ Durduruldu")
        finally:
            self.thinkgear.disconnect()
            self._print_summary()
    
    def _print_summary(self):
        """Özet yazdır"""
        if self.total_predictions > 0:
            avg_inference = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
            avg_fft = sum(self.fft_times) / len(self.fft_times) if self.fft_times else 0
            
            print("\n" + "=" * 60)
            print("📊 ÖZET")
            print("=" * 60)
            print(f"Toplam tahmin: {self.total_predictions}")
            print(f"Belirsiz: {self.uncertain_count}")
            print(f"Ortalama FFT: {avg_fft:.2f}ms")
            print(f"Ortalama inference: {avg_inference:.2f}ms")
            print(f"Toplam gecikme: {avg_fft + avg_inference:.2f}ms")
            print()
            for label in self.LABELS:
                count = self.predictions[label]
                pct = count / self.total_predictions * 100 if self.total_predictions > 0 else 0
                print(f"   {label}: {count} ({pct:.1f}%)")
            print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("🧠 FFT Tabanlı Gerçek Zamanlı EEG Tahmin")
    print("   Raw EEG → Filtreleme → FFT → Tahmin")
    print("=" * 60)
    
    # CUDA kontrolü
    if torch.cuda.is_available():
        print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA yok, CPU kullanılacak")
    
    print("\n📋 Gereksinimler:")
    print("   1. Windows'ta: python thinkgear_proxy.py")
    print("   2. ThinkGear Connector çalışıyor")
    print("   3. MindWave bağlı")
    
    print("\n🚀 Avantajlar:")
    print("   - NeuroSky 1 Hz → Bu sistem ~2-4 Hz")
    print("   - Kendi filtreleme (Notch + Bandpass)")
    print("   - Kendi FFT hesaplama")
    print("   - %95.70 doğruluk (TCN)")
    
    predictor = RealtimeFFTPredictor(
        model_window=128,      # 128 FFT frame
        fft_window=512,        # 512 sample = 1 saniye
        prediction_interval=0.25  # 4 Hz tahmin
    )
    predictor.run()


if __name__ == "__main__":
    main()
