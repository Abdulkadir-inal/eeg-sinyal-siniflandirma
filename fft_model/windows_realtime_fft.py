#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Windows FFT Tabanlı Gerçek Zamanlı EEG Tahmin
================================================

MindWave'den Raw EEG alır, filtreleme ve FFT'yi bilgisayarda
hesaplayarak hızlı tahmin yapar.

⚠️ BİLİNEN SORUNLAR:
- Canlı tahmin performansı offline test sonuçlarından düşük
- Model sınıflar arası geçişlerde zorlanıyor

🔧 TODO: KALİBRASYON SİSTEMİ EKLENMELİ
Scaler uyumsuzluğu çözümü için:
1. Program başında 10-30 sn kalibrasyon
2. Kullanıcının nötr/dinlenme durumu ölçülecek  
3. Kişiye özel mean/std hesaplanacak
4. Eğitim scaler'ına oranlanarak adaptif normalizasyon

🔧 TODO: TRANSFER LEARNING (Kişiye Özel Model)
Daha iyi bireysel tahmin için:
1. Karma model (tüm kullanıcılar) ile temel EEG örüntüleri öğretilmiş
2. Son katmanlar dondurulup, sadece son katmanlar yeniden eğitilecek
3. Sadece hedef kullanıcının verisi (örn: apo_*.csv) ile fine-tune
4. Avantajları:
   - Az veri ile yüksek doğruluk
   - Kişisel EEG desenlerine uyum
   - Scaler uyumsuzluğu sorunu azalır
   
Uygulama:
    # Karma modeli yükle
    model = load_model("karma_model.pth")
    
    # Erken katmanları dondur (genel EEG bilgisi korunsun)
    for param in model.tcn.parameters():
        param.requires_grad = False
    
    # Sadece son FC katmanları eğitilebilir bırak
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Kişisel veri ile fine-tune (az epoch yeterli, örn: 10-20)
    train(model, personal_data, epochs=20, lr=0.0001)

🔧 TODO: TUŞ KONTROLÜ (Start/Stop Tahmin)
Tahmin yapma zamanlamasını kullanıcı kontrol edebilsin:
1. pynput kütüphanesi kullanılacak (cross-platform: Windows, Mac, Linux)
2. Kurulum: pip install pynput
3. Mac'te Accessibility izni gerekli (System Preferences > Security & Privacy > Accessibility)

Tuş atamaları:
    - S tuşu → Tahmin başlat (Start)
    - E tuşu → Tahmin durdur (End)
    - SPACE  → Toggle (aç/kapat)
    - Q tuşu → Programdan çık (Quit)

Uygulama:
    from pynput import keyboard
    
    recording = False
    
    def on_press(key):
        global recording
        try:
            if key.char == 's':
                recording = True
                print("🔴 TAHMİN BAŞLADI")
            elif key.char == 'e':
                recording = False
                print("⏸️ TAHMİN DURAKLATILDI")
            elif key.char == 'q':
                return False  # Listener'ı durdur
        except AttributeError:
            if key == keyboard.Key.space:
                recording = not recording
                print(f"{'🔴 AKTİF' if recording else '⏸️ PASIF'}")
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    # Ana döngüde:
    if recording:
        # Tahmin yap
        pass

🔧 TODO: DAHA HIZLI TAHMİN İÇİN PENCERE BOYUTU KÜÇÜLTME
Şu anki model_window = 128 frame → Daha küçük yapılabilir (64, 32)

Avantajları:
    ✅ Daha hızlı tepki süresi (gecikme azalır)
    ✅ Daha az veri biriktirme bekleme süresi
    ✅ Gerçek zamanlı kontrol için daha uygun

Dezavantajları:
    ❌ Daha az temporal context → Model daha az bilgiyle karar verir
    ❌ Doğruluk düşebilir (daha az veri = daha az güvenilir patern)
    ❌ Gürültüye daha hassas (küçük pencere = noise'dan daha çok etkilenir)
    ❌ MODEL YENİDEN EĞİTİLMELİ! (eğitim ve test aynı pencere boyutunda olmalı)

Uygulama adımları:
    1. train_model_fft.py'de SEQUENCE_LENGTH değiştir (128 → 64 veya 32)
    2. Modeli yeniden eğit
    3. windows_realtime_fft.py'de model_window değiştir
    4. Test et ve doğruluk karşılaştır

Önerilen deney:
    | Pencere | Tahmini Gecikme | Beklenen Doğruluk |
    |---------|-----------------|-------------------|
    | 128     | ~1-2 sn         | En yüksek (%95)   |
    | 64      | ~0.5-1 sn       | Orta (%85-90?)    |
    | 32      | ~0.25-0.5 sn    | Düşük (%75-85?)   |

------------------------------------------------------------------------    

NeuroSky EEG Power: 1 Hz (saniyede 1 tahmin)
Bu sistem: ~2-4 Hz (saniyede 2-4 tahmin)

Kullanım:
    1. ThinkGear Connector'ı başlatın
    2. MindWave'i bağlayın
    3. Gerekli paketleri yükleyin:
       pip install torch numpy scipy pynput
    4. Bu scripti çalıştırın:
       python windows_realtime_fft.py
    5. Model seçin (TCN önerilen: %95.70)
    6. Kalibrasyon yapın (15 sn dinlenme durumu)
    7. Tuşlarla kontrol edin:
       - S: Tahmin başlat
       - E: Tahmin durdur
       - SPACE: Toggle (aç/kapat)
       - Q: Programdan çık

Gereksinimler:
    pip install torch numpy scipy pynput

Yeni Özellikler:
    ✨ Kalibrasyon sistemi - Kişiye özel normalizasyon
    ✨ Tuş kontrolü - İstediğiniz zaman tahmin başlatın/durdurun

Özellikler:
    - Raw EEG'den kendi FFT hesaplama
    - Sinyal filtreleme (Notch 50Hz, Bandpass 0.5-50Hz)
    - Artifact rejection
    - CUDA/GPU desteği (varsa)
    - Hızlı tahmin (~250-500ms aralıklarla)
"""

import os
import sys
import time
import socket
import json
import numpy as np
from collections import deque
from datetime import datetime

# Tuş kontrolü
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("⚠️ pynput bulunamadı. Tuş kontrolü devre dışı.")
    print("   Yüklemek için: pip install pynput")

# SciPy (filtreleme için)
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
# SİNYAL FİLTRELEME
# ============================================================================

class SignalProcessor:
    """EEG sinyal işleme sınıfı"""
    
    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs
        
        # Filtreleri önceden oluştur (hız için)
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
        """FFT ile frekans bant güçlerini hesapla"""
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
# MODEL TANIMLARI
# ============================================================================

class TemporalBlock(nn.Module):
    """TCN için Temporal Block - Causal Convolution"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.padding = padding
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Causal: padding'i kes
        out = self.dropout1(self.relu1(self.bn1(out)))
        
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Causal: padding'i kes
        out = self.dropout2(self.relu2(self.bn2(out)))
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN_Model(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, input_channels=9, num_classes=3, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(TCN_Model, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=padding, dropout=dropout))
        
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


class PositionalEncoding(nn.Module):
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


class TransformerModel(nn.Module):
    """Transformer tabanlı model"""
    def __init__(self, input_channels=9, num_classes=3, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_channels, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 128, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=d_model*4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_channels=9, num_classes=3):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, dropout=0.3)
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
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# THINKGEAR BAĞLANTISI
# ============================================================================

class ThinkGearConnector:
    """ThinkGear Connector'a doğrudan bağlanır ve Raw EEG okur"""
    
    def __init__(self, host='127.0.0.1', port=13854):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""
        
        # Raw EEG buffer
        self.raw_buffer = deque(maxlen=FFT_WINDOW_SIZE * 2)
        
        # Durum
        self.poor_signal = 200
        self.raw_count = 0
    
    def connect(self):
        """ThinkGear Connector'a bağlan"""
        try:
            print(f"🔵 ThinkGear Connector'a bağlanılıyor: {self.host}:{self.port}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.host, self.port))
            
            # Raw EEG çıktısı iste (512 Hz)
            self.sock.send(b'{"enableRawOutput": true, "format": "Json"}\n')
            
            # TCP optimizasyonları
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(0.05)  # 50ms timeout
            
            print("✅ Bağlantı başarılı!")
            print("📡 Raw EEG çıktısı: AKTİF (512 Hz)")
            return True
            
        except ConnectionRefusedError:
            print("❌ ThinkGear Connector çalışmıyor!")
            print("\n💡 Çözüm:")
            print("   1. ThinkGear Connector'ı başlatın")
            print("   2. MindWave cihazını bağlayın")
            print("   3. Bu scripti tekrar çalıştırın")
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
                    
                    # Raw EEG (512 Hz)
                    if 'rawEeg' in parsed:
                        self.raw_buffer.append(parsed['rawEeg'])
                        self.raw_count += 1
                        got_raw = True
                    
                    # Sinyal kalitesi
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

class WindowsFFTPredictor:
    """Windows'ta FFT tabanlı gerçek zamanlı tahmin"""
    
    MODELS = {
        '1': ('TCN (%95.70)', TCN_Model, 'tcn_model_fft.pth'),
        '2': ('Transformer (%93.49)', TransformerModel, 'transformer_model_fft.pth'),
        '3': ('CNN-LSTM (%81.57)', CNN_LSTM_Model, 'cnn_lstm_model_fft.pth')
    }
    
    LABELS = ['araba', 'aşağı', 'yukarı']
    CONFIDENCE_THRESHOLD = 0.70
    
    def __init__(self, model_window=128, fft_window=512, prediction_interval=0.25):
        self.model_window = model_window
        self.fft_window = fft_window
        self.prediction_interval = prediction_interval
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = None
        
        # Signal processor
        self.signal_processor = SignalProcessor()
        
        # FFT buffer
        self.fft_buffer = deque(maxlen=model_window)
        
        # ThinkGear
        self.thinkgear = ThinkGearConnector()
        
        # Scaler (eğitim verisi)
        self.scaler_mean = None
        self.scaler_std = None
        
        # Kalibrasyon (kullanıcıya özel)
        self.calibration_mean = None
        self.calibration_std = None
        self.is_calibrated = False
        
        # Tuş kontrolü
        self.recording = False  # Tahmin yapılsın mı?
        self.should_quit = False
        
        # Stats
        self.predictions = {label: 0 for label in self.LABELS}
        self.total_predictions = 0
        self.uncertain_count = 0
        self.inference_times = []
        self.fft_times = []
    
    def find_model_path(self, filename):
        """Model dosyasının yolunu bul"""
        # Olası konumlar
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, filename),
            os.path.join(script_dir, 'fft_model', filename),
            os.path.join(os.path.dirname(script_dir), 'fft_model', filename),
            filename
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def load_scaler_params(self):
        """Scaler parametrelerini yükle"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Olası konumlar
        possible_paths = [
            os.path.join(script_dir, 'scaler_params_fft.json'),
            os.path.join(script_dir, 'fft_model', 'scaler_params_fft.json'),
            'scaler_params_fft.json'
        ]
        
        for scaler_path in possible_paths:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'r') as f:
                    params = json.load(f)
                self.scaler_mean = np.array(params['mean'])
                self.scaler_std = np.array(params['std'])
                print(f"✅ Scaler yüklendi: {scaler_path}")
                return True
        
        print("⚠️ Scaler dosyası bulunamadı")
        return False
    
    def select_model(self):
        """Model seç"""
        print("\n" + "=" * 60)
        print("🧠 FFT MODEL SEÇİMİ")
        print("=" * 60)
        
        for key, (name, _, _) in self.MODELS.items():
            print(f"   {key}. {name}")
        
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
        name, model_class, filename = self.MODELS[choice]
        
        model_path = self.find_model_path(filename)
        if not model_path:
            print(f"❌ Model bulunamadı: {filename}")
            return False
        
        try:
            print(f"\n📥 Yükleniyor: {name}")
            
            self.model = model_class(input_channels=9, num_classes=3)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_name = name
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
        """FFT verilerini normalize et (kalibrasyonlu)"""
        x = np.array(fft_window_data, dtype=np.float32)
        
        # Kalibrasyon uygula
        if self.is_calibrated and self.calibration_mean is not None:
            x_flat = x.flatten()
            # Önce kullanıcının baseline'ını çıkar
            x_flat = x_flat - self.calibration_mean
            # Sonra eğitim scaler'ı ile normalize et
            if self.scaler_mean is not None and len(x_flat) == len(self.scaler_mean):
                x_normalized = x_flat / np.where(self.scaler_std > 0, self.scaler_std, 1)
                x = x_normalized.reshape(x.shape)
            else:
                x = x_flat.reshape(x.shape)
        elif self.scaler_mean is not None and self.scaler_std is not None:
            # Kalibrasyon yoksa klasik normalize
            x_flat = x.flatten()
            if len(x_flat) == len(self.scaler_mean):
                x_normalized = (x_flat - self.scaler_mean) / np.where(self.scaler_std > 0, self.scaler_std, 1)
                x = x_normalized.reshape(x.shape)
            else:
                # Fallback: per-channel normalize
                for i in range(x.shape[1]):
                    col = x[:, i]
                    x[:, i] = (col - np.mean(col)) / (np.std(col) + 1e-8)
        
        return torch.FloatTensor(x).unsqueeze(0).to(self.device)
    
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
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = (time.time() - start_time) * 1000
            
            return self.LABELS[predicted.item()], confidence.item(), inference_time
    
    def setup_keyboard_listener(self):
        """Klavye dinleyicisini başlat"""
        if not PYNPUT_AVAILABLE:
            print("⚠️ pynput yüklü değil, tuş kontrolü devre dışı")
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
                # Space tuşu
                if key == keyboard.Key.space:
                    self.recording = not self.recording
                    print(f"\n{'🔴 TAHMİN AKTİF' if self.recording else '⏸️  TAHMİN PASIF'} (SPACE)")
        
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print("✅ Tuş kontrolü aktif: [S]tart, [E]nd, [SPACE]toggle, [Q]uit")
    
    def calibrate(self, duration=15):
        """Kullanıcıya özel kalibrasyon"""
        print("\n" + "=" * 60)
        print("🎯 KALİBRASYON AŞAMASI")
        print("=" * 60)
        print(f"⏱️  {duration} saniye boyunca:")
        print("   • Rahat oturun")
        print("   • Gözlerinizi kapatın")
        print("   • Hiçbir şey düşünmeyin (nötr durum)")
        print("   • MindWave'in sinyali iyi olmalı")
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
            print("   Sinyal kalitesini kontrol edin ve tekrar deneyin.")
            return False
        
        # Kalibrasyon istatistikleri hesapla
        cal_array = np.array(calibration_data, dtype=np.float32)
        self.calibration_mean = np.mean(cal_array.flatten())
        self.calibration_std = np.std(cal_array.flatten())
        self.is_calibrated = True
        
        print("\n\n✅ KALİBRASYON TAMAMLANDI")
        print(f"   📊 {len(calibration_data)} FFT frame toplandı")
        print(f"   📈 Baseline: {self.calibration_mean:.2f} (std: {self.calibration_std:.2f})")
        print("-" * 60)
        
        return True
    
    def run(self):
        """Ana döngü"""
        print("\n" + "=" * 60)
        print("🧠 Windows FFT Gerçek Zamanlı EEG Tahmin")
        print("   Raw EEG → Filtreleme → FFT → Tahmin")
        print("=" * 60)
        
        # Scaler yükle
        self.load_scaler_params()
        
        # Model seç
        if not self.select_model():
            return
        
        # Bağlan
        print("\n" + "-" * 60)
        if not self.thinkgear.connect():
            return
        
        # Kalibrasyon sor
        print("\n" + "=" * 60)
        do_calibration = input("Kalibrasyon yapmak ister misiniz? (y/n) [önerilen]: ").strip().lower()
        
        if do_calibration in ['y', 'yes', 'e', 'evet', '']:
            if not self.calibrate():
                return
        else:
            print("⚠️ Kalibrasyon atlandı - tahmin doğruluğu düşük olabilir")
        
        # Tuş kontrolünü başlat
        if PYNPUT_AVAILABLE:
            self.setup_keyboard_listener()
        else:
            print("⚠️ Tuş kontrolü yok - sürekli tahmin modu")
            self.recording = True
        
        print("\n" + "=" * 60)
        print(f"📊 Model: {self.model_name}")
        print(f"⚡ Cihaz: {self.device}")
        print(f"🎯 Sınıflar: {', '.join(self.LABELS)}")
        print(f"📦 FFT: {self.fft_window} sample (1 saniye)")
        print(f"📦 Model: {self.model_window} frame")
        print(f"⏱️ Tahmin: {1/self.prediction_interval:.1f} Hz")
        print(f"🎚️  Kalibrasyon: {'✅ Aktif' if self.is_calibrated else '❌ Yok'}")
        print("=" * 60)
        print("\n💡 MindWave'i takın!")
        if PYNPUT_AVAILABLE:
            print("🎹 Tuşlar: [S]başla [E]dur [SPACE]toggle [Q]çık")
        print("⏸️  Durdurmak için Ctrl+C")
        print("-" * 60)
        
        try:
            last_prediction_time = 0
            last_raw_count = 0
            raw_samples_for_fft = 256  # Her 256 sample'da FFT
            
            raw_received = False
            prediction_started = False
            
            while True:
                # Çıkış kontrolü
                if self.should_quit:
                    break
                
                result = self.thinkgear.read_data()
                
                if result == 'raw':
                    raw_received = True
                    
                    # Buffer durumu
                    if not prediction_started:
                        raw_count = self.thinkgear.get_buffer_size()
                        fft_count = len(self.fft_buffer)
                        sig = "✅" if self.thinkgear.poor_signal < 50 else f"⚠️({self.thinkgear.poor_signal})"
                        rec_status = "🔴" if self.recording else "⏸️"
                        print(f"\r{rec_status} Raw: {raw_count}/{self.fft_window} | FFT: {fft_count}/{self.model_window} | {sig}   ", end='')
                    
                    current_time = time.time()
                    
                    # FFT hesapla
                    raw_buffer_size = self.thinkgear.get_buffer_size()
                    new_samples = self.thinkgear.raw_count - last_raw_count
                    
                    if raw_buffer_size >= self.fft_window and new_samples >= raw_samples_for_fft:
                        last_raw_count = self.thinkgear.raw_count
                        
                        fft_start = time.time()
                        raw_samples = self.thinkgear.get_raw_samples(self.fft_window)
                        band_powers = self.signal_processor.process_raw_to_fft(raw_samples)
                        fft_time = (time.time() - fft_start) * 1000
                        self.fft_times.append(fft_time)
                        
                        # FFT buffer'a ekle [Electrode=0, Delta, Theta, ...]
                        self.fft_buffer.append([0] + band_powers)
                    
                    # Tahmin zamanı (sadece recording=True ise)
                    if self.recording and len(self.fft_buffer) >= self.model_window and (current_time - last_prediction_time) >= self.prediction_interval:
                        last_prediction_time = current_time
                        prediction_started = True
                        
                        fft_data = list(self.fft_buffer)[-self.model_window:]
                        label, confidence, inf_time = self.predict(fft_data)
                        
                        self.inference_times.append(inf_time)
                        
                        if label:
                            avg_fft = sum(self.fft_times[-10:]) / min(len(self.fft_times), 10) if self.fft_times else 0
                            
                            print()
                            print("\n" + "=" * 60)
                            
                            if confidence >= self.CONFIDENCE_THRESHOLD:
                                self.predictions[label] += 1
                                self.total_predictions += 1
                                
                                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | #{self.total_predictions}")
                                print(f"⚡ FFT: {avg_fft:.1f}ms | Model: {inf_time:.1f}ms")
                                print(f"🎯 {label.upper()} ({confidence*100:.1f}%)")
                            else:
                                self.uncertain_count += 1
                                
                                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | ❓ Belirsiz #{self.uncertain_count}")
                                print(f"⚡ FFT: {avg_fft:.1f}ms | Model: {inf_time:.1f}ms")
                                print(f"🤔 {label} ({confidence*100:.1f}% < {self.CONFIDENCE_THRESHOLD*100:.0f}%)")
                            
                            print("-" * 60)
                            
                            for l in self.LABELS:
                                count = self.predictions[l]
                                total = max(self.total_predictions, 1)
                                pct = count / total * 100
                                bar = "█" * int(pct / 5)
                                marker = "👉" if (l == label and confidence >= self.CONFIDENCE_THRESHOLD) else "  "
                                print(f"{marker} {l:8}: {bar:<20} {pct:.1f}%")
                            
                            if self.uncertain_count > 0:
                                all_total = self.total_predictions + self.uncertain_count
                                u_pct = self.uncertain_count / all_total * 100
                                print(f"   {'belirsiz':8}: {'░' * int(u_pct / 5):<20} {u_pct:.1f}%")
                            
                            print("=" * 60)
                
                elif not raw_received:
                    print("\r⏳ Raw EEG bekleniyor...", end='')
                
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n\n⛔ Durduruldu")
        finally:
            self.thinkgear.disconnect()
            self._print_summary()
    
    def _print_summary(self):
        """Özet"""
        if self.total_predictions > 0:
            avg_inf = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
            avg_fft = sum(self.fft_times) / len(self.fft_times) if self.fft_times else 0
            
            print("\n" + "=" * 60)
            print("📊 ÖZET")
            print("=" * 60)
            print(f"Toplam tahmin: {self.total_predictions}")
            print(f"Belirsiz: {self.uncertain_count}")
            print(f"Ortalama FFT: {avg_fft:.2f}ms")
            print(f"Ortalama model: {avg_inf:.2f}ms")
            print(f"Toplam gecikme: {avg_fft + avg_inf:.2f}ms")
            print()
            for label in self.LABELS:
                count = self.predictions[label]
                pct = count / self.total_predictions * 100 if self.total_predictions > 0 else 0
                print(f"   {label}: {count} ({pct:.1f}%)")
            print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("🧠 Windows FFT Gerçek Zamanlı EEG Tahmin")
    print("=" * 60)
    
    # CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️ CPU kullanılacak")
    
    print("\n📋 Gereksinimler:")
    print("   1. ThinkGear Connector çalışıyor")
    print("   2. MindWave bağlı")
    
    print("\n🚀 Avantajlar:")
    print("   - NeuroSky 1 Hz → Bu sistem ~2-4 Hz")
    print("   - Kendi filtreleme")
    print("   - %95.70 doğruluk (TCN)")
    
    predictor = WindowsFFTPredictor(
        model_window=128,
        fft_window=512,
        prediction_interval=0.25
    )
    predictor.run()


if __name__ == "__main__":
    main()
