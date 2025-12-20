#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Log Transform + Oran Formülleri ile Gerçek Zamanlı EEG Tahmin
================================================================

windows_realtime_fft.py'nin Log Transform + Oran Formülleri versiyonu.

Bu script:
1. MindWave'den Raw EEG alır (512 Hz)
2. Sinyal filtreleme yapar (Notch 50Hz, Bandpass 0.5-50Hz)
3. FFT ile bant güçleri hesaplar
4. Log Transform + Oran Formülleri uygular (9 → 17 özellik)
5. TCN model ile tahmin yapar (%99.43 accuracy)

Yeni Özellikler:
    ✨ Log Transform - Küçük farkları büyütür
    ✨ 8 Oran Formülü - Bantlar arası ilişkileri yakalar
    ✨ Kalibrasyon sistemi - Kişiye özel normalizasyon
    ✨ Tuş kontrolü - [S]tart [E]nd [SPACE]toggle [Q]uit
    ✨ Direkt Bağlantı - ThinkGear Connector'a ihtiyaç yok!

Bağlantı Seçenekleri:
    1. ThinkGear Connector (Port 13854)
    2. Direkt Seri Port (Bluetooth SPP) - DAHA KARARLI!

Kullanım:
    python3 realtime_transformed.py

Gereksinimler:
    pip install torch numpy scipy pynput pyserial
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

# Tuş kontrolü
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("⚠️ pynput bulunamadı. Tuş kontrolü devre dışı.")
    print("   Yüklemek için: pip install pynput")

# Serial Port (direkt bağlantı için)
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️ pyserial bulunamadı. Direkt bağlantı devre dışı.")
    print("   Yüklemek için: pip install pyserial")

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
# AYARLAR
# ============================================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Sinyal işleme parametreleri
SAMPLING_RATE = 512  # Hz
FFT_WINDOW_SIZE = 512  # 1 saniyelik FFT penceresi
MODEL_WINDOW = 128  # Model için frame sayısı

# Filtre parametreleri
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ['araba', 'aşağı', 'yukarı']


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
        """Raw EEG sinyalini filtrele"""
        samples = np.array(raw_samples, dtype=np.float64)
        
        # DC offset kaldır
        samples = samples - np.mean(samples)
        
        # Artifact'ları temizle
        artifact_mask = np.abs(samples) > ARTIFACT_THRESHOLD
        if np.any(artifact_mask):
            good_samples = samples[~artifact_mask]
            if len(good_samples) > 0:
                median_val = np.median(good_samples)
                samples[artifact_mask] = median_val
        
        # Notch filtre (50 Hz)
        samples = scipy_signal.filtfilt(self.notch_b, self.notch_a, samples)
        
        # Bandpass filtre (0.5-50 Hz)
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
# TRANSFORMASYON FONKSİYONLARI (Log + Oran)
# ============================================================================

def apply_log_transform(data):
    """Log transform: log1p(x) = log(1 + x)"""
    return np.sign(data) * np.log1p(np.abs(data))


def calculate_band_ratios(window):
    """
    8 oran özelliği hesapla
    
    Input: (128, 9) - 128 frame, 9 özellik
    Output: (128, 8) - 128 frame, 8 oran
    """
    # Bant indeksleri: 0=Electrode, 1=Delta, 2=Theta, 3=LowAlpha, 4=HighAlpha
    # 5=LowBeta, 6=HighBeta, 7=LowGamma, 8=MidGamma
    
    delta = window[:, 1] + 1e-8
    theta = window[:, 2] + 1e-8
    low_alpha = window[:, 3] + 1e-8
    high_alpha = window[:, 4] + 1e-8
    low_beta = window[:, 5] + 1e-8
    high_beta = window[:, 6] + 1e-8
    low_gamma = window[:, 7] + 1e-8
    mid_gamma = window[:, 8] + 1e-8
    
    # Kombine bantlar
    alpha = (low_alpha + high_alpha) / 2
    beta = (low_beta + high_beta) / 2
    gamma = (low_gamma + mid_gamma) / 2
    
    # 8 oran hesapla
    ratios = np.column_stack([
        delta / theta,                          # Delta_Theta
        theta / alpha,                          # Theta_Alpha
        alpha / beta,                           # Alpha_Beta
        beta / gamma,                           # Beta_Gamma
        (theta + alpha) / (beta + gamma),       # Slow_Fast
        delta / alpha,                          # Delta_Alpha
        (delta + theta) / (alpha + beta + gamma),  # VeryLow_High
        (alpha + beta) / (delta + theta),       # Mid_Low
    ])
    
    return ratios


def transform_window(window):
    """
    Tek bir window'a tüm transformasyonları uygula
    
    Input: (128, 9)
    Output: (128, 17) - 9 orijinal (log transformed) + 8 oran (log transformed)
    """
    # Log transform uygula
    log_transformed = apply_log_transform(window)
    
    # Oranları hesapla (orijinal veriden)
    ratios = calculate_band_ratios(window)
    
    # Log transform'u oranlara da uygula
    ratios_log = apply_log_transform(ratios)
    
    # Birleştir
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
        
        # Boyut uyumu için kırp
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

class DirectMindWaveConnector:
    """MindWave'e direkt seri port üzerinden bağlanır (ThinkGear Connector gerekmez!)"""
    
    def __init__(self, port=None):
        self.port = port
        self.serial = None
        self.buffer = bytearray()
        
        # Raw EEG buffer
        self.raw_buffer = deque(maxlen=FFT_WINDOW_SIZE * 2)
        
        # Durum
        self.poor_signal = 200
        self.raw_count = 0
    
    @staticmethod
    def list_ports():
        """Kullanılabilir seri portları listele"""
        if not SERIAL_AVAILABLE:
            return []
        
        ports = []
        for port in serial.tools.list_ports.comports():
            # MindWave portlarını filtrele
            if any(keyword in port.description.lower() for keyword in ['mindwave', 'neurosky', 'bluetooth', 'rfcomm', 'tty.']):
                ports.append({
                    'device': port.device,
                    'description': port.description,
                    'hwid': port.hwid
                })
        
        return ports
    
    def connect(self):
        """MindWave'e direkt bağlan"""
        if not SERIAL_AVAILABLE:
            print("❌ pyserial kurulu değil!")
            print("   Kurulum: pip install pyserial")
            return False
        
        try:
            # Port otomatik seçimi
            if self.port is None:
                print("🔍 Kullanılabilir portlar aranıyor...")
                ports = self.list_ports()
                
                if not ports:
                    print("❌ MindWave portu bulunamadı!")
                    print("\n💡 Çözüm:")
                    print("   1. MindWave'i Bluetooth ile eşleştirin")
                    print("   2. Cihazın 'Bağlı' durumda olduğundan emin olun")
                    print("   3. Bu scripti tekrar çalıştırın")
                    return False
                
                if len(ports) == 1:
                    self.port = ports[0]['device']
                    print(f"✅ Port bulundu: {self.port}")
                    print(f"   {ports[0]['description']}")
                else:
                    print(f"\n📋 {len(ports)} port bulundu:")
                    for i, port in enumerate(ports, 1):
                        print(f"   {i}. {port['device']} - {port['description']}")
                    
                    choice = input(f"\nHangi portu kullanmak istersiniz? (1-{len(ports)}): ").strip()
                    try:
                        idx = int(choice) - 1
                        self.port = ports[idx]['device']
                    except (ValueError, IndexError):
                        print("❌ Geçersiz seçim!")
                        return False
            
            # Serial bağlantı aç
            print(f"\n🔵 MindWave'e bağlanılıyor: {self.port}")
            self.serial = serial.Serial(
                port=self.port,
                baudrate=57600,  # MindWave standart baud rate
                timeout=0.1,     # 100ms timeout
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Buffer temizle
            time.sleep(0.5)
            self.serial.reset_input_buffer()
            
            print("✅ Bağlantı başarılı!")
            print("📡 Raw EEG çıktısı: AKTİF (512 Hz)")
            print("🎉 ThinkGear Connector gerekmedi!")
            return True
            
        except serial.SerialException as e:
            print(f"❌ Bağlantı hatası: {e}")
            print("\n💡 Olası Çözümler:")
            print("   • Port başka bir uygulama tarafından kullanılıyor olabilir")
            print("   • MindWave'in Bluetooth bağlantısını kontrol edin")
            print("   • Cihazı kapatıp tekrar açın")
            return False
        except Exception as e:
            print(f"❌ Beklenmeyen hata: {e}")
            return False
    
    def disconnect(self):
        """Bağlantıyı kapat"""
        if self.serial and self.serial.is_open:
            try:
                self.serial.close()
            except:
                pass
        print("🔌 Bağlantı kapatıldı")
    
    def _parse_packet(self):
        """ThinkGear paketini parse et"""
        while len(self.buffer) >= 4:
            # Sync bytes ara (0xAA 0xAA)
            if self.buffer[0] != 0xAA or self.buffer[1] != 0xAA:
                self.buffer.pop(0)
                continue
            
            # Packet uzunluğu
            plength = self.buffer[2]
            
            # Tam paket gelene kadar bekle
            if len(self.buffer) < plength + 4:  # AA AA LEN [DATA...] CKSUM
                break
            
            # Checksum kontrolü
            payload = self.buffer[3:3+plength]
            checksum = self.buffer[3+plength]
            
            calc_sum = sum(payload) & 0xFF
            calc_sum = (~calc_sum) & 0xFF
            
            if checksum != calc_sum:
                # Checksum hatası, ilk byte'ı at ve devam et
                self.buffer.pop(0)
                continue
            
            # Payload'ı parse et
            i = 0
            while i < len(payload):
                code = payload[i]
                i += 1
                
                # Extended code level check
                while code == 0x55 and i < len(payload):
                    code = payload[i]
                    i += 1
                
                # Value uzunluğu
                if code >= 0x80:
                    if i >= len(payload):
                        break
                    vlength = payload[i]
                    i += 1
                else:
                    vlength = 1
                
                # Value oku
                if i + vlength > len(payload):
                    break
                
                value = payload[i:i+vlength]
                i += vlength
                
                # Raw EEG (0x80, 2 bytes)
                if code == 0x80 and len(value) == 2:
                    raw_value = int.from_bytes(value, byteorder='big', signed=True)
                    self.raw_buffer.append(raw_value)
                    self.raw_count += 1
                
                # Poor Signal Quality (0x02, 1 byte)
                elif code == 0x02 and len(value) == 1:
                    self.poor_signal = value[0]
            
            # İşlenen paketi buffer'dan kaldır
            del self.buffer[:3+plength+1]
    
    def read_data(self):
        """Serial porttan veri oku"""
        if not self.serial or not self.serial.is_open:
            return None
        
        try:
            # Mevcut veriyi oku
            if self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting)
                self.buffer.extend(data)
            
            # Buffer'daki paketleri parse et
            old_count = self.raw_count
            self._parse_packet()
            
            # Yeni raw veri geldi mi?
            if self.raw_count > old_count:
                return 'raw'
            
            return None
            
        except serial.SerialException:
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

class RealtimeTransformedPredictor:
    """Log Transform + Oran Formülleri ile gerçek zamanlı tahmin"""
    
    CONFIDENCE_THRESHOLD = 0.70
    
    def __init__(self, model_window=MODEL_WINDOW, fft_window=FFT_WINDOW_SIZE, prediction_interval=0.25, use_direct_connection=False, use_3person_model=False):
        self.model_window = model_window
        self.fft_window = fft_window
        self.prediction_interval = prediction_interval
        self.use_direct_connection = use_direct_connection
        self.use_3person_model = use_3person_model
        
        # Device
        self.device = DEVICE
        self.model = None
        
        # Signal processor
        self.signal_processor = SignalProcessor()
        
        # FFT buffer (9 özellik: Electrode + 8 bant)
        self.fft_buffer = deque(maxlen=model_window)
        
        # MindWave bağlantısı (direkt veya ThinkGear Connector)
        if use_direct_connection:
            self.thinkgear = DirectMindWaveConnector()
        else:
            self.thinkgear = ThinkGearConnector()
        
        # Scaler (eğitim verisi)
        self.scaler = None
        
        # Kalibrasyon
        self.calibration_mean = None
        self.calibration_std = None
        self.is_calibrated = False
        
        # Tuş kontrolü
        self.recording = False
        self.should_quit = False
        
        # Stats
        self.predictions = {label: 0 for label in LABELS}
        self.total_predictions = 0
        self.uncertain_count = 0
    
    def load_model(self):
        """Model ve scaler'ı yükle"""
        print("\n📂 Model yükleniyor...")
        
        # Model dizini ve dosya adlarını belirle
        if self.use_3person_model:
            model_dir = os.path.join(MODEL_DIR, '3person_model')
            scaler_name = 'scaler_3person.pkl'
            model_name = 'best_model_3person.pth'
            accuracy = '%99.35'
            model_desc = '(3 Kişi: Apo, Bahadır, Canan)'
        else:
            model_dir = MODEL_DIR
            scaler_name = 'scaler_transformed.pkl'
            model_name = 'best_model_transformed.pth'
            accuracy = '%99.43'
            model_desc = '(Tüm Veri)'
        
        # Scaler yükle
        scaler_path = os.path.join(model_dir, scaler_name)
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"   ✅ Scaler yüklendi {model_desc}")
        else:
            print(f"   ⚠️ Scaler bulunamadı: {scaler_path}")
        
        # Model yükle
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"   ❌ Model bulunamadı: {model_path}")
            return False
        
        try:
            self.model = TCN_Model(input_channels=17, num_classes=3).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print(f"   ✅ TCN Model yüklendi {accuracy} {model_desc}")
            print(f"   ⚡ Cihaz: {self.device}")
            
            if self.device.type == 'cuda':
                print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Model yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess(self, fft_window_data):
        """FFT verilerini transform et ve normalize et"""
        # numpy array'e çevir (128, 9)
        x = np.array(fft_window_data, dtype=np.float32)
        
        # Log Transform + Oran Formülleri uygula (128, 9) → (128, 17)
        x_transformed = transform_window(x)
        
        # Kalibrasyon uygula
        if self.is_calibrated and self.calibration_mean is not None:
            x_flat = x_transformed.flatten()
            x_flat = x_flat - self.calibration_mean
            x_flat = x_flat / (self.calibration_std + 1e-8)
            x_transformed = x_flat.reshape(x_transformed.shape)
        
        # Scaler uygula
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
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = (time.time() - start_time) * 1000
            
            return LABELS[predicted.item()], confidence.item(), inference_time
    
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
                    # 9 özellik: Electrode (0) + 8 bant
                    calibration_data.append([0] + band_powers)
            
            time.sleep(0.001)
        
        if len(calibration_data) < 10:
            print("\n\n❌ Yeterli kalibrasyon verisi toplanamadı!")
            return False
        
        # Kalibrasyon istatistikleri hesapla
        cal_array = np.array(calibration_data, dtype=np.float32)
        # Log + Oran transform uygula
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
        accuracy = "%99.35" if self.use_3person_model else "%99.43"
        model_desc = "(3 Kişi)" if self.use_3person_model else "(Tüm Veri)"
        
        print("\n" + "=" * 60)
        print("🧠 LOG TRANSFORM + ORAN FORMÜLLERİ")
        print(f"   Gerçek Zamanlı EEG Tahmin {accuracy} {model_desc}")
        print("=" * 60)
        
        # Model yükle
        if not self.load_model():
            return
        
        # Bağlan
        print("\n" + "-" * 60)
        if not self.thinkgear.connect():
            return
        
        # Kalibrasyon sor
        print("\n" + "=" * 60)
        do_cal = input("Kalibrasyon yapmak ister misiniz? (y/n) [önerilen]: ").strip().lower()
        
        if do_cal in ['y', 'yes', 'e', 'evet', '']:
            if not self.calibrate():
                return
        else:
            print("⚠️ Kalibrasyon atlandı")
        
        # Tuş kontrolünü başlat
        if PYNPUT_AVAILABLE:
            self.setup_keyboard_listener()
        else:
            self.recording = True
        
        accuracy = "%99.35" if self.use_3person_model else "%99.43"
        model_desc = "(3 Kişi: Apo, Bahadır, Canan)" if self.use_3person_model else "(Tüm Veri)"
        
        print("\n" + "=" * 60)
        print(f"📊 Model: TCN {accuracy} {model_desc}")
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
        
        # Buffer doldur
        print("\n⏳ FFT buffer dolduruluyor...")
        last_raw_count = 0
        raw_samples_for_fft = 256
        
        try:
            while not self.should_quit:
                # Veri oku
                result = self.thinkgear.read_data()
                
                if result == 'raw':
                    raw_buffer_size = self.thinkgear.get_buffer_size()
                    new_samples = self.thinkgear.raw_count - last_raw_count
                    
                    # Yeni FFT hesapla
                    if raw_buffer_size >= self.fft_window and new_samples >= raw_samples_for_fft:
                        last_raw_count = self.thinkgear.raw_count
                        
                        # FFT hesapla
                        raw_samples = self.thinkgear.get_raw_samples(self.fft_window)
                        band_powers = self.signal_processor.process_raw_to_fft(raw_samples)
                        
                        # Buffer'a ekle (9 özellik)
                        self.fft_buffer.append([0] + band_powers)
                        
                        # Buffer doluysa ve recording aktifse tahmin yap
                        if len(self.fft_buffer) >= self.model_window:
                            if self.recording:
                                # Tahmin yap
                                window_data = list(self.fft_buffer)[-self.model_window:]
                                label, confidence, inference_time = self.predict(window_data)
                                
                                self.total_predictions += 1
                                
                                # Emoji
                                emoji = {"araba": "🚗", "yukarı": "⬆️", "aşağı": "⬇️"}.get(label, "❓")
                                
                                # Güven kontrolü
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
                                # Recording pasif
                                print(f"\r⏸️  Bekleniyor... Buffer: {len(self.fft_buffer)}/{self.model_window}   ", end='')
                        else:
                            # Buffer dolmuyor
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
    print("🧪 DEMO MODU - Rastgele Veri ile Test")
    print("=" * 60)
    
    predictor = RealtimeTransformedPredictor()
    
    if not predictor.load_model():
        return
    
    print("\n🎲 Rastgele veri ile 10 tahmin yapılıyor...\n")
    
    for i in range(10):
        # Rastgele window oluştur (128, 9)
        window = np.random.randn(MODEL_WINDOW, 9) * 50000 + 100000
        window = np.abs(window)
        
        # Tahmin
        label, confidence, inference_time = predictor.predict(window)
        
        emoji = {"araba": "🚗", "yukarı": "⬆️", "aşağı": "⬇️"}.get(label, "❓")
        print(f"[{i+1:2d}] {emoji} {label:8s} | Güven: {confidence*100:5.1f}% | {inference_time:.1f}ms")
    
    print("\n✅ Demo tamamlandı!")


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("🧠 LOG TRANSFORM + ORAN FORMÜLLERİ")
    print("   Gerçek Zamanlı EEG Tahmin Sistemi")
    print("=" * 60)
    print(f"📱 Device: {DEVICE}")
    print(f"📂 Model: {MODEL_DIR}")
    
    print("\n📋 Bağlantı Türü:")
    print("   1. 🔌 Direkt Bağlantı (Seri Port - ÖNERİLEN!)")
    print("   2. 🌐 ThinkGear Connector (Port 13854)")
    print("   3. 🧪 Demo Modu (rastgele veri ile test)")
    print("   4. ❌ Çıkış")
    
    print("\n💡 İpucu:")
    print("   • Direkt Bağlantı daha kararlı ve kolay!")
    print("   • ThinkGear Connector gerekmez")
    print("   • Sadece Bluetooth eşleştirmesi yeterli")
    
    if SERIAL_AVAILABLE:
        print("   ✅ pyserial yüklü - Direkt bağlantı kullanılabilir")
    else:
        print("   ⚠️ pyserial yok - Sadece ThinkGear Connector kullanılabilir")
        print("      Yüklemek için: pip install pyserial")
    
    try:
        choice = input("\nBağlantı türü seçin (1/2/3/4): ").strip()
        
        if choice == "4":
            print("Çıkış...")
            return
        elif choice == "3":
            demo_mode()
            return
        
        # Bağlantı türü belirlendi, şimdi model seçimi
        use_direct = (choice == "1")
        
        if choice == "1" and not SERIAL_AVAILABLE:
            print("\n❌ pyserial kurulu değil!")
            print("   Kurulum: pip install pyserial")
            return
        
        # Model seçimi
        print("\n" + "=" * 60)
        print("📊 MODEL SEÇİMİ")
        print("=" * 60)
        print("   1. 📈 Tüm Veri Modeli (%99.43 accuracy)")
        print("      • Tüm katılımcılar dahil")
        print("      • 20,207 window ile eğitildi")
        print("")
        print("   2. 👥 3 Kişi Modeli (%99.35 accuracy)")
        print("      • Sadece: Apo, Bahadır, Canan")
        print("      • 13,144 window ile eğitildi")
        print("      • Daha spesifik tahmin")
        
        model_choice = input("\nModel seçin (1/2): ").strip()
        use_3person = (model_choice == "2")
        
        # Bağlantı türüne göre mesaj
        if choice == "1":
            print("\n🔌 DİREKT BAĞLANTI MODU")
            print("=" * 60)
            print("✨ ThinkGear Connector gerekmez!")
            print("🎯 Sadece MindWave'i Bluetooth ile eşleştirin")
        else:
            print("\n🌐 THINKGEAR CONNECTOR MODU")
            print("=" * 60)
            print("⚠️ ThinkGear Connector uygulaması çalışıyor olmalı")
            print("📡 Port 13854 dinleniyor...")
        
        # Model bilgisi
        if use_3person:
            print("👥 Model: 3 Kişi (Apo, Bahadır, Canan) - %99.35")
        else:
            print("📈 Model: Tüm Veri - %99.43")
        print("-" * 60)
        
        predictor = RealtimeTransformedPredictor(
            use_direct_connection=use_direct,
            use_3person_model=use_3person
        )
        predictor.run()
            
    except KeyboardInterrupt:
        print("\n\nÇıkış...")


if __name__ == "__main__":
    main()
