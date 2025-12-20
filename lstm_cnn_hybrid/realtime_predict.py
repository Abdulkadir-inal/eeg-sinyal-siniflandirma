#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM+CNN Hibrit Model - Canlı/Dosya Tahmin
==========================================

Üç Mod:
    1. Canlı (Raw EEG → FFT → Model)
    2. Canlı (ThinkGear Connector)
    3. Dosya (FFT CSV → Model)

Dosya Modu:
    FFT bant güçleri hesaplanmış CSV dosyalarını okur.
    Sütunlar: delta, theta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma, midGamma
    
    Kullanım:
        python realtime_predict.py --file data.csv --model seq64
        python realtime_predict.py --file data.csv --model seq32 --output results.txt
        python realtime_predict.py --file folder/ --model seq64  # Klasördeki tüm CSV'ler

Canlı Mod:
    Raw EEG sinyalinden FFT hesaplar (eğitimle aynı pipeline).
    MindWave'in kendi FFT'sini DEĞİL, bizim hesapladığımız FFT'yi kullanır.
    
    Pipeline: Raw EEG → Notch Filter (50Hz) → Bandpass (0.5-50Hz) → FFT → Model
    
    Kullanım:
        python realtime_predict.py --port COM5
        python realtime_predict.py --thinkgear --model seq96

Model Seçenekleri:
    --model seq32   : Hızlı tepki (sequence_length=32)
    --model seq64   : Baseline model (sequence_length=64)
    --model seq96   : Genişletilmiş görüş (sequence_length=96)
    --model seq128  : En geniş görüş (sequence_length=128)
"""

import os
import sys
import argparse
import time
import json
import pickle
import numpy as np
from collections import deque
import threading
import signal as sig

import torch
import torch.nn as nn
import pandas as pd
import glob
import serial  # Arduino servo control için

# Sinyal işleme modülü
from signal_processor import SignalProcessor, BAND_NAMES, SAMPLING_RATE, WINDOW_SIZE

# ============================================================================
# AYARLAR
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# MindWave bağlantı
BAUD_RATE = 57600

# Tahmin ayarları
PREDICTION_INTERVAL = 0.5  # Her 0.5 saniyede bir tahmin
CONFIDENCE_THRESHOLD = 0.6  # Minimum güven skoru

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
# MINDWAVE RAW EEG BAĞLANTISI
# ============================================================================

class MindWaveRawConnector:
    """
    MindWave'den RAW EEG verisi okur.
    ThinkGear protokolü ile 512 Hz raw sinyal alır.
    """
    
    def __init__(self, port, baud_rate=57600):
        self.port = port
        self.baud_rate = baud_rate
        self.serial = None
        self.running = False
        
        # Raw EEG buffer
        self.raw_samples = deque(maxlen=2048)  # ~4 saniye
        self.lock = threading.Lock()
        
        # Sinyal kalitesi
        self.poor_signal = 0
        self.last_raw_time = 0
    
    def connect(self):
        """Seri porta bağlan"""
        try:
            import serial
            self.serial = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)
            print(f"✅ MindWave bağlandı: {self.port}")
            return True
        except Exception as e:
            print(f"❌ Bağlantı hatası: {e}")
            return False
    
    def disconnect(self):
        """Bağlantıyı kapat"""
        self.running = False
        if self.serial:
            self.serial.close()
            self.serial = None
    
    def parse_packet(self, packet):
        """ThinkGear paketi parse et - RAW değerleri çıkar"""
        i = 0
        while i < len(packet):
            code = packet[i]
            
            if code == 0x80:  # RAW Wave Value (2 bytes)
                if i + 2 < len(packet):
                    # Big-endian signed 16-bit
                    high = packet[i + 1]
                    low = packet[i + 2]
                    raw_value = (high << 8) | low
                    # İşaretli değere çevir
                    if raw_value >= 32768:
                        raw_value -= 65536
                    
                    with self.lock:
                        self.raw_samples.append(raw_value)
                    self.last_raw_time = time.time()
                i += 3
            
            elif code == 0x02:  # Poor Signal
                if i + 1 < len(packet):
                    self.poor_signal = packet[i + 1]
                i += 2
            
            elif code == 0x83:  # ASIC_EEG_POWER (24 bytes) - Atlıyoruz
                i += 26
            
            elif code == 0x04:  # Attention
                i += 2
            elif code == 0x05:  # Meditation
                i += 2
            else:
                i += 1
    
    def read_loop(self):
        """Sürekli veri okuma"""
        self.running = True
        sync_count = 0
        
        while self.running:
            try:
                byte = self.serial.read(1)
                if not byte:
                    continue
                
                b = byte[0]
                
                if b == 0xAA:
                    sync_count += 1
                    if sync_count >= 2:
                        sync_count = 0
                        length_byte = self.serial.read(1)
                        if length_byte:
                            length = length_byte[0]
                            if length < 170:
                                payload = list(self.serial.read(length))
                                checksum = self.serial.read(1)
                                if payload and checksum:
                                    calc_checksum = (~sum(payload) & 0xFF)
                                    if calc_checksum == checksum[0]:
                                        self.parse_packet(payload)
                else:
                    sync_count = 0
                    
            except Exception as e:
                if self.running:
                    print(f"Okuma hatası: {e}")
                    time.sleep(0.1)
    
    def get_raw_samples(self, count=None):
        """Raw sample'ları al ve buffer'dan temizle"""
        with self.lock:
            if count is None:
                samples = list(self.raw_samples)
                self.raw_samples.clear()
            else:
                samples = []
                for _ in range(min(count, len(self.raw_samples))):
                    samples.append(self.raw_samples.popleft())
        return samples
    
    def get_buffer_size(self):
        """Buffer'daki sample sayısı"""
        with self.lock:
            return len(self.raw_samples)
    
    def start(self):
        """Okuma thread'ini başlat"""
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()


# ============================================================================
# THINKGEAR CONNECTOR BAĞLANTISI (TCP/JSON)
# ============================================================================

class ThinkGearConnector:
    """
    ThinkGear Connector üzerinden RAW EEG verisi okur.
    TCP Socket ile localhost:13854'e bağlanır.
    JSON formatında veri alır.
    
    NOT: ThinkGear Connector uygulamasının çalışıyor olması gerekir!
    """
    
    def __init__(self, host='127.0.0.1', port=13854):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
        # Raw EEG buffer
        self.raw_samples = deque(maxlen=2048)  # ~4 saniye
        self.lock = threading.Lock()
        
        # Sinyal kalitesi
        self.poor_signal = 0
        self.last_raw_time = 0
        
        # Buffer for incomplete JSON
        self.buffer = ""
    
    def connect(self):
        """ThinkGear Connector'a TCP bağlantısı kur"""
        import socket
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(1.0)
            
            # Raw EEG output'u etkinleştir
            config = json.dumps({
                "enableRawOutput": True,
                "format": "Json"
            })
            self.socket.send(config.encode('utf-8'))
            
            print(f"✅ ThinkGear Connector bağlandı: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ ThinkGear Connector bağlantı hatası: {e}")
            print("   ThinkGear Connector uygulamasının çalıştığından emin olun!")
            return False
    
    def disconnect(self):
        """Bağlantıyı kapat"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def read_loop(self):
        """Sürekli veri okuma (TCP)"""
        self.running = True
        
        while self.running:
            try:
                data = self.socket.recv(4096)
                if not data:
                    continue
                
                self.buffer += data.decode('utf-8')
                
                # JSON objeleri ayır (her satır bir JSON)
                while '\r' in self.buffer:
                    line, self.buffer = self.buffer.split('\r', 1)
                    line = line.strip()
                    if line:
                        try:
                            packet = json.loads(line)
                            self.parse_json(packet)
                        except json.JSONDecodeError:
                            pass
                            
            except Exception as e:
                if self.running and "timed out" not in str(e):
                    print(f"ThinkGear okuma hatası: {e}")
                    time.sleep(0.1)
    
    def parse_json(self, packet):
        """JSON paketi parse et"""
        # Raw EEG değeri
        if 'rawEeg' in packet:
            raw_value = packet['rawEeg']
            with self.lock:
                self.raw_samples.append(raw_value)
            self.last_raw_time = time.time()
        
        # Sinyal kalitesi
        if 'poorSignalLevel' in packet:
            self.poor_signal = packet['poorSignalLevel']
    
    def get_raw_samples(self, count=None):
        """Raw sample'ları al ve buffer'dan temizle"""
        with self.lock:
            if count is None:
                samples = list(self.raw_samples)
                self.raw_samples.clear()
            else:
                samples = []
                for _ in range(min(count, len(self.raw_samples))):
                    samples.append(self.raw_samples.popleft())
        return samples
    
    def get_buffer_size(self):
        """Buffer'daki sample sayısı"""
        with self.lock:
            return len(self.raw_samples)
    
    def start(self):
        """Okuma thread'ini başlat"""
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()


# ============================================================================
# TAHMİN MOTORU (Raw EEG → FFT → Model)
# ============================================================================

class PredictionEngine:
    """Raw EEG'den FFT hesaplar ve model ile tahmin yapar"""
    
    def __init__(self, model_path, scaler_path, config_path, label_map_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Config yükle
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Label map yükle
        if label_map_path is None:
            label_map_path = os.path.join(os.path.dirname(config_path), 'label_map.json')
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        # Scaler yükle
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Model yükle
        self.model = SimpleCNN_LSTM(
            input_features=self.config['num_features'],
            num_classes=self.config['num_classes'],
            dropout=0.0
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model yüklendi: {os.path.basename(model_path)}")
        print(f"   Sequence Length: {self.config['sequence_length']}")
        print(f"   Validation Accuracy: {checkpoint.get('val_acc', 0):.2f}%")
        
        # Sinyal işleyici (Raw → FFT)
        self.signal_processor = SignalProcessor()
        
        # FFT feature buffer (model için)
        self.feature_buffer = deque(maxlen=self.config['sequence_length'])
        
        # Tahmin smoothing
        self.prediction_history = deque(maxlen=5)
    
    def process_raw_samples(self, raw_samples):
        """
        Raw EEG sample'larını işle.
        Her 512 sample'da bir FFT hesaplar.
        """
        fft_count = 0
        
        for sample in raw_samples:
            result = self.signal_processor.add_sample(sample)
            
            if result is not None:
                # FFT hesaplandı - feature'lara ekle
                self._add_fft_to_buffer(result)
                fft_count += 1
        
        return fft_count
    
    def _add_fft_to_buffer(self, fft_dict):
        """FFT değerlerini özellik buffer'ına ekle"""
        # 8 orijinal özellik
        features = [fft_dict.get(band, 0) for band in BAND_NAMES]
        features = np.array(features, dtype=np.float32)
        
        # Log transform (eğitimle aynı)
        features = np.log1p(np.abs(features))
        
        # Türetilmiş özellikler (data_preprocess.py ile aynı)
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
        
        # 15 özellik
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
        
        # Buffer'ı numpy array'e çevir
        sequence = np.array(self.feature_buffer, dtype=np.float32)
        
        # Normalize (eğitimle aynı scaler)
        original_shape = sequence.shape
        seq_flat = sequence.reshape(-1, sequence.shape[-1])
        seq_flat = self.scaler.transform(seq_flat)
        sequence = seq_flat.reshape(original_shape)
        
        # Tensor'a çevir
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Tahmin
        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, predicted = probs.max(0)
            
            pred_class = predicted.item()
            conf = confidence.item()
            
            class_probs = {self.label_map[str(i)]: probs[i].item() for i in range(len(probs))}
        
        # Smoothing
        self.prediction_history.append((pred_class, conf))
        
        from collections import Counter
        recent_preds = [p[0] for p in self.prediction_history if p[1] > 0.4]
        if recent_preds:
            smoothed_pred = Counter(recent_preds).most_common(1)[0][0]
        else:
            smoothed_pred = pred_class
        
        label = self.label_map.get(str(smoothed_pred), f"Class {smoothed_pred}")
        
        return label, conf, class_probs
    
    def is_ready(self):
        """Tahmin için hazır mı?"""
        return len(self.feature_buffer) >= self.config['sequence_length']
    
    def get_buffer_status(self):
        """Buffer durumu"""
        return {
            'raw_buffer': f"{self.signal_processor.get_buffer_progress():.1f}%",
            'feature_buffer': f"{len(self.feature_buffer)}/{self.config['sequence_length']}",
            'total_samples': self.signal_processor.total_samples,
            'artifacts': self.signal_processor.artifact_count
        }
    
    def add_fft_row(self, fft_values):
        """
        Dışarıdan FFT değerlerini ekle (dosya modu için).
        fft_values: 8 bant güçü (delta, theta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma, midGamma)
        """
        fft_dict = dict(zip(BAND_NAMES, fft_values))
        self._add_fft_to_buffer(fft_dict)
    
    def predict_from_fft_sequence(self, fft_data, stride=1):
        """
        FFT veri serisinden tahmin yap (dosya modu için).
        fft_data: numpy array shape (N, 8) - N satır, 8 bant
        stride: kaç satırda bir tahmin yapılacak (varsayılan: 1)
        Returns: list of (label, confidence, class_probs) tuples
        """
        # Buffer'ı temizle
        self.feature_buffer.clear()
        self.prediction_history.clear()
        
        results = []
        seq_len = self.config['sequence_length']
        last_pred_idx = -stride  # İlk tahmini hemen yapabilsin
        
        for i, row in enumerate(fft_data):
            self.add_fft_row(row)
            
            # Yeterli veri biriktiğinde ve stride aralığında tahmin yap
            if len(self.feature_buffer) >= seq_len and (i - last_pred_idx) >= stride:
                label, conf, probs = self.predict()
                results.append({
                    'row': i + 1,
                    'label': label,
                    'confidence': conf,
                    'probabilities': probs
                })
                last_pred_idx = i
        
        return results


# ============================================================================
# DOSYA MODU
# ============================================================================

# Desteklenen FFT sütun isimleri (farklı kaynaklardan gelen dosyalar için)
FFT_COLUMN_VARIANTS = {
    'delta': ['delta', 'Delta', 'DELTA'],
    'theta': ['theta', 'Theta', 'THETA'],
    'lowAlpha': ['lowAlpha', 'low_alpha', 'LowAlpha', 'LOW_ALPHA', 'lowalpha', 'Low Alpha'],
    'highAlpha': ['highAlpha', 'high_alpha', 'HighAlpha', 'HIGH_ALPHA', 'highalpha', 'High Alpha'],
    'lowBeta': ['lowBeta', 'low_beta', 'LowBeta', 'LOW_BETA', 'lowbeta', 'Low Beta'],
    'highBeta': ['highBeta', 'high_beta', 'HighBeta', 'HIGH_BETA', 'highbeta', 'High Beta'],
    'lowGamma': ['lowGamma', 'low_gamma', 'LowGamma', 'LOW_GAMMA', 'lowgamma', 'Low Gamma'],
    'midGamma': ['midGamma', 'mid_gamma', 'MidGamma', 'MID_GAMMA', 'midgamma', 'Mid Gamma']
}

def find_fft_columns(df):
    """DataFrame'deki FFT sütunlarını bul ve eşleştir"""
    found_columns = {}
    
    for band, variants in FFT_COLUMN_VARIANTS.items():
        for variant in variants:
            if variant in df.columns:
                found_columns[band] = variant
                break
    
    return found_columns

def load_fft_csv(file_path):
    """FFT CSV dosyasını yükle ve 8 bant değerlerini döndür"""
    df = pd.read_csv(file_path)
    
    # Sütunları bul
    col_map = find_fft_columns(df)
    
    # Gerekli bantları kontrol et
    required_bands = list(FFT_COLUMN_VARIANTS.keys())
    missing = [b for b in required_bands if b not in col_map]
    
    if missing:
        # Eğer sütunlar yoksa, indeks sıralı olabilir
        if len(df.columns) >= 8:
            # İlk 8 sütunu al (veya timestamp sonrası)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 8:
                print(f"   ⚠️ Bant isimleri bulunamadı, ilk 8 sayısal sütun kullanılıyor")
                return df[numeric_cols[:8]].values
        
        raise ValueError(f"Eksik FFT bantları: {missing}")
    
    # Sıralı olarak al
    data = df[[col_map[b] for b in required_bands]].values
    return data

def run_file_mode(args):
    """FFT CSV dosyasından tahmin yap"""
    
    # Model bilgilerini al
    model_info = AVAILABLE_MODELS[args.model]
    model_path = os.path.join(SCRIPT_DIR, model_info['model'])
    scaler_path = os.path.join(SCRIPT_DIR, model_info['scaler'])
    config_path = os.path.join(SCRIPT_DIR, model_info['config'])
    label_map_path = os.path.join(SCRIPT_DIR, model_info['label_map'])
    
    print("\n" + "=" * 60)
    print("📂 DOSYA MODU - FFT CSV → Model")
    print("=" * 60)
    print(f"\n📦 Seçilen Model: {model_info['name']}")
    print(f"   {model_info['description']}")
    
    # Dosya kontrolü
    missing_files = []
    for path, name in [(model_path, 'Model'), (scaler_path, 'Scaler'), 
                       (config_path, 'Config'), (label_map_path, 'Label Map')]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {os.path.basename(path)}")
    
    if missing_files:
        print(f"\n❌ Eksik model dosyaları:")
        for f in missing_files:
            print(f"   - {f}")
        return
    
    # Engine oluştur
    print("\n📦 Model yükleniyor...")
    engine = PredictionEngine(model_path, scaler_path, config_path, label_map_path)
    seq_len = engine.config['sequence_length']
    
    # Dosya/klasör yolu
    input_path = args.file
    files_to_process = []
    
    if os.path.isdir(input_path):
        # Klasördeki tüm CSV'leri bul
        pattern = os.path.join(input_path, '**', '*.csv')
        files_to_process = glob.glob(pattern, recursive=True)
        print(f"\n📁 Klasör: {input_path}")
        print(f"   {len(files_to_process)} CSV dosyası bulundu")
    elif os.path.isfile(input_path):
        files_to_process = [input_path]
    else:
        print(f"\n❌ Dosya/klasör bulunamadı: {input_path}")
        return
    
    if not files_to_process:
        print("\n❌ İşlenecek CSV dosyası bulunamadı!")
        return
    
    # Çıktı dosyası
    output_file = None
    if args.output:
        output_file = open(args.output, 'w', encoding='utf-8')
        output_file.write("# LSTM+CNN FFT Tahmin Sonuçları\n")
        output_file.write(f"# Model: {model_info['name']}\n")
        output_file.write(f"# Sequence Length: {seq_len}\n\n")
    
    # Her dosyayı işle
    total_predictions = 0
    all_results = {}
    stride = args.stride
    
    print(f"   Stride: her {stride} satırda bir tahmin")
    print("\n" + "-" * 60)
    
    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        print(f"\n📄 {file_name}")
        
        try:
            # FFT verilerini yükle
            fft_data = load_fft_csv(file_path)
            print(f"   {len(fft_data)} satır yüklendi")
            
            if len(fft_data) < seq_len:
                print(f"   ⚠️ Yetersiz veri! En az {seq_len} satır gerekli.")
                continue
            
            # Tahmin yap (stride ile)
            results = engine.predict_from_fft_sequence(fft_data, stride=stride)
            total_predictions += len(results)
            
            # Özet istatistikler
            if results:
                # En sık tahmin
                from collections import Counter
                label_counts = Counter(r['label'] for r in results)
                most_common = label_counts.most_common()
                
                # Ortalama güven
                avg_conf = sum(r['confidence'] for r in results) / len(results)
                
                print(f"   ✅ {len(results)} tahmin yapıldı")
                print(f"   📊 Dağılım: ", end='')
                for label, count in most_common:
                    pct = count / len(results) * 100
                    print(f"{label}={pct:.1f}% ", end='')
                print(f"\n   🎯 Ortalama güven: {avg_conf*100:.1f}%")
                
                # Detaylı çıktı
                if args.verbose:
                    print("\n   Detaylı tahminler:")
                    for r in results[:10]:  # İlk 10
                        print(f"      Satır {r['row']:4d}: {r['label']:10s} ({r['confidence']*100:.1f}%)")
                    if len(results) > 10:
                        print(f"      ... ve {len(results)-10} daha")
                
                # Dosyaya yaz
                if output_file:
                    output_file.write(f"\n## {file_name}\n")
                    output_file.write(f"Satır sayısı: {len(fft_data)}\n")
                    output_file.write(f"Tahmin sayısı: {len(results)}\n")
                    output_file.write(f"Dağılım: {dict(label_counts)}\n")
                    output_file.write(f"Ortalama güven: {avg_conf*100:.1f}%\n")
                    
                    if args.verbose:
                        output_file.write("\nDetaylı tahminler:\n")
                        for r in results:
                            output_file.write(f"  {r['row']:4d}: {r['label']:10s} ({r['confidence']*100:.1f}%)\n")
                
                all_results[file_name] = {
                    'predictions': results,
                    'distribution': dict(label_counts),
                    'avg_confidence': avg_conf
                }
            
        except Exception as e:
            print(f"   ❌ Hata: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Özet
    print("\n" + "=" * 60)
    print("📊 ÖZET")
    print("=" * 60)
    print(f"   İşlenen dosya: {len(files_to_process)}")
    print(f"   Toplam tahmin: {total_predictions}")
    
    if output_file:
        output_file.write(f"\n\n# ÖZET\n")
        output_file.write(f"İşlenen dosya: {len(files_to_process)}\n")
        output_file.write(f"Toplam tahmin: {total_predictions}\n")
        output_file.close()
        print(f"   💾 Sonuçlar kaydedildi: {args.output}")
    
    print()


# ============================================================================
# ANA UYGULAMA
# ============================================================================

# ============================================================================
# ARDUINO SERVO CONTROLLER
# ============================================================================

class ArduinoController:
    """
    Arduino ile servo motor kontrolü.
    Tahmin sonucuna göre servo pozisyonunu değiştirir.
    
    Komutlar:
        b'Y' -> yukarı (servo yukarı pozisyon)
        b'A' -> aşağı (servo aşağı pozisyon)  
        b'R' -> araba (servo orta pozisyon)
    """
    
    def __init__(self, port, baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.connected = False
    
    def connect(self):
        """Arduino'ya seri port bağlantısı kur"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Arduino reset için bekle
            self.connected = True
            print(f"✅ Arduino bağlandı: {self.port} @ {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"❌ Arduino bağlantı hatası: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"❌ Arduino hatası: {e}")
            self.connected = False
            return False
    
    def send_command(self, label):
        """
        Tahmin etiketine göre Arduino'ya komut gönder.
        
        Args:
            label: Tahmin etiketi ('yukarı', 'aşağı', 'asagı', 'araba')
        """
        if not self.connected or self.serial_conn is None:
            return False
        
        try:
            if 'yukarı' in label.lower() or 'yukari' in label.lower():
                self.serial_conn.write(b'Y')
                return True
            elif 'aşağı' in label.lower() or 'asagı' in label.lower() or 'asagi' in label.lower():
                self.serial_conn.write(b'A')
                return True
            elif 'araba' in label.lower():
                self.serial_conn.write(b'R')
                return True
            else:
                return False
        except serial.SerialException as e:
            print(f"❌ Arduino yazma hatası: {e}")
            return False
    
    def close(self):
        """Seri port bağlantısını kapat"""
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
                print("✅ Arduino bağlantısı kapatıldı")
            except:
                pass
        self.connected = False


def main():
    parser = argparse.ArgumentParser(
        description='LSTM+CNN Canlı/Dosya Tahmin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  # Dosya modu - FFT CSV
  python realtime_predict.py --file data.csv --model seq64
  python realtime_predict.py --file folder/ --model seq32 -v
  python realtime_predict.py --file data.csv --output results.txt
  
  # Canlı mod - seri port
  python realtime_predict.py --port COM5
  
  # Canlı mod - ThinkGear
  python realtime_predict.py --thinkgear --model seq96
  
  # Mevcut modelleri listele
  python realtime_predict.py --list-models
        """
    )
    
    # Model seçimi
    parser.add_argument('--model', default='seq64', choices=list(AVAILABLE_MODELS.keys()),
                       help='Kullanılacak model (varsayılan: seq64)')
    parser.add_argument('--list-models', action='store_true',
                       help='Mevcut modelleri listele ve çık')
    
    # Dosya modu
    parser.add_argument('--file', metavar='PATH',
                       help='FFT CSV dosyası veya klasör (dosya modu)')
    parser.add_argument('--output', '-o', metavar='FILE',
                       help='Sonuçları kaydet (varsayılan: stdout)')
    parser.add_argument('--stride', type=int, default=64,
                       help='Dosya modunda kaç satırda bir tahmin (varsayılan: 64)')
    
    # Bağlantı türü (canlı mod)
    conn_group = parser.add_mutually_exclusive_group()
    conn_group.add_argument('--port', help='Seri port (örn: COM5 veya /dev/ttyUSB0)')
    conn_group.add_argument('--thinkgear', action='store_true', 
                           help='ThinkGear Connector kullan (TCP/JSON)')
    
    # ThinkGear ayarları
    parser.add_argument('--tg-host', default='127.0.0.1', 
                       help='ThinkGear Connector host (varsayılan: 127.0.0.1)')
    parser.add_argument('--tg-port', type=int, default=13854, 
                       help='ThinkGear Connector port (varsayılan: 13854)')
    
    # Genel ayarlar
    parser.add_argument('--threshold', type=float, default=0.6, help='Güven eşiği (0-1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Detaylı çıktı (dosya modu)')
    
    # Arduino servo kontrolü
    parser.add_argument('--arduino', metavar='PORT',
                       help='Arduino seri port (örn: COM3, /dev/ttyACM0). Belirtilmezse servo kontrolü devre dışı.')
    
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
            print(f"   Dosyalar: {info['model']}, {info['config']}")
        print()
        return
    
    # Dosya modu mu?
    if args.file:
        run_file_mode(args)
        return
    
    # Canlı mod: Bağlantı kontrolü
    if not args.port and not args.thinkgear:
        parser.error("--port, --thinkgear veya --file belirtmelisiniz. --list-models ile modelleri görebilirsiniz.")
    
    # Model bilgilerini al
    model_info = AVAILABLE_MODELS[args.model]
    model_path = os.path.join(SCRIPT_DIR, model_info['model'])
    scaler_path = os.path.join(SCRIPT_DIR, model_info['scaler'])
    config_path = os.path.join(SCRIPT_DIR, model_info['config'])
    label_map_path = os.path.join(SCRIPT_DIR, model_info['label_map'])
    
    print("\n" + "=" * 60)
    print("🧠 LSTM+CNN Hibrit Model - Canlı Tahmin")
    print("   (Raw EEG → Filtre → FFT → Model)")
    print("=" * 60)
    print(f"\n📦 Seçilen Model: {model_info['name']}")
    print(f"   {model_info['description']}")
    
    # Dosya kontrolü
    missing_files = []
    for path, name in [(model_path, 'Model'), (scaler_path, 'Scaler'), 
                       (config_path, 'Config'), (label_map_path, 'Label Map')]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {os.path.basename(path)}")
    
    if missing_files:
        print(f"\n❌ Eksik dosyalar:")
        for f in missing_files:
            print(f"   - {f}")
        print(f"\n💡 İpucu: '{args.model}' modeli henüz eğitilmemiş olabilir.")
        print(f"   Eğitmek için: python train_experiment.py --seq-len {args.model.replace('seq', '')}")
        return
    
    # Engine oluştur
    print("\n📦 Model yükleniyor...")
    engine = PredictionEngine(model_path, scaler_path, config_path, label_map_path)
    
    # Arduino kontrolcü oluştur (opsiyonel)
    arduino_controller = None
    if args.arduino:
        print(f"\n🤖 Arduino servo kontrolü: {args.arduino}")
        arduino_controller = ArduinoController(args.arduino)
        if not arduino_controller.connect():
            print("⚠️  Arduino bağlanamadı, servo kontrolü devre dışı")
            arduino_controller = None
    
    # =====================================================================
    # AŞAMA 1: Bağlantı için kullanıcı komutu bekle
    # =====================================================================
    connection_type = "ThinkGear Connector (TCP)" if args.thinkgear else f"Seri Port ({args.port})"
    print("\n" + "=" * 60)
    print("📋 KOMUTLAR:")
    print("   'baglan' veya 'b' → Cihaza bağlan")
    print("   'q' → Çıkış")
    print("=" * 60)
    print(f"\n🔌 Bağlantı modu: {connection_type}")
    print("   MindWave başlığını hazırlayın...")
    print()
    
    connector = None
    
    while True:
        try:
            user_input = input(">> ").strip().lower()
            if user_input in ['baglan', 'bağlan', 'connect', 'b', 'c']:
                break
            elif user_input in ['q', 'quit', 'exit', 'cik', 'çık']:
                print("👋 Çıkış yapılıyor...")
                return
            else:
                print("   'baglan' yazıp ENTER'a basın (çıkmak için 'q')")
        except EOFError:
            return
    
    # Bağlantı kur
    if args.thinkgear:
        print(f"\n📡 ThinkGear Connector'a bağlanılıyor: {args.tg_host}:{args.tg_port}")
        print("   (ThinkGear Connector uygulamasının çalıştığından emin olun!)")
        connector = ThinkGearConnector(host=args.tg_host, port=args.tg_port)
    else:
        print(f"\n📡 MindWave'e bağlanılıyor (Seri port): {args.port}")
        connector = MindWaveRawConnector(args.port)
    
    if not connector.connect():
        print("❌ Bağlantı kurulamadı!")
        if args.thinkgear:
            print("\n💡 İpucu: ThinkGear Connector uygulamasını başlattınız mı?")
            print("   Windows: Başlat menüsünden 'ThinkGear Connector' arayın")
        return
    
    connector.start()
    
    # =====================================================================
    # AŞAMA 2: Sinyal kalitesini göster ve tahmin için bekle
    # =====================================================================
    print("\n" + "=" * 60)
    print("✅ Bağlantı kuruldu!")
    print(f"   Mod: {connection_type}")
    print("=" * 60)
    
    print("\n🎧 MindWave başlığını takın...")
    print("   Sinyal kalitesi izleniyor (0 = iyi, 200 = kötü)")
    print()
    print("📋 KOMUTLAR:")
    print("   'basla' veya 's' → Tahmine başla")
    print("   'q' → Çıkış")
    print()
    
    # Sinyal kalitesini göster (arka planda)
    import sys
    last_signal_time = 0
    
    while True:
        try:
            # Sinyal kalitesini her saniye göster
            current_time = time.time()
            if current_time - last_signal_time >= 1.0:
                last_signal_time = current_time
                signal_quality = connector.poor_signal
                buffer_size = connector.get_buffer_size()
                
                if signal_quality == 0:
                    status = "✅ Mükemmel"
                    color = "\033[92m"
                elif signal_quality < 50:
                    status = "👍 İyi"
                    color = "\033[92m"
                elif signal_quality < 100:
                    status = "⚠️ Orta"
                    color = "\033[93m"
                else:
                    status = "❌ Zayıf"
                    color = "\033[91m"
                
                sys.stdout.write(f"\r{color}📊 Sinyal: {signal_quality:3d} ({status}) | Buffer: {buffer_size} sample\033[0m   ")
                sys.stdout.flush()
            
            # Non-blocking input check (Windows compatible)
            import select
            if sys.platform == 'win32':
                import msvcrt
                if msvcrt.kbhit():
                    char = msvcrt.getwch()
                    if char == '\r':  # Enter
                        sys.stdout.write('\n')
                        user_input = input(">> ").strip().lower()
                        if user_input in ['basla', 'başla', 'start', 's']:
                            break
                        elif user_input in ['q', 'quit', 'exit', 'cik', 'çık']:
                            print("👋 Çıkış yapılıyor...")
                            connector.disconnect()
                            return
                        else:
                            print("   'basla' yazıp ENTER'a basın (çıkmak için 'q')")
            else:
                # Linux/Mac
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input in ['basla', 'başla', 'start', 's']:
                        break
                    elif user_input in ['q', 'quit', 'exit', 'cik', 'çık']:
                        print("\n👋 Çıkış yapılıyor...")
                        connector.disconnect()
                        return
                    elif user_input:
                        print("\n   'basla' yazıp ENTER'a basın (çıkmak için 'q')")
            
            time.sleep(0.1)
            
        except EOFError:
            break
        except Exception as e:
            # Fallback: blocking input
            print()
            user_input = input(">> ").strip().lower()
            if user_input in ['basla', 'başla', 'start', 's']:
                break
            elif user_input in ['q', 'quit', 'exit', 'cik', 'çık']:
                print("👋 Çıkış yapılıyor...")
                connector.disconnect()
                return
    
    print("\n")  # Yeni satır
    
    # Ctrl+C handler
    running = True
    def signal_handler(s, frame):
        nonlocal running
        running = False
        print("\n\n⏹️  Durduruluyor...")
    
    sig.signal(sig.SIGINT, signal_handler)
    
    print("=" * 60)
    print("🎯 TAHMİN BAŞLADI!")
    print(f"   Bağlantı: {connection_type}")
    print("   Pipeline: Raw EEG → Filtre → FFT → Model")
    print("   Ctrl+C ile çıkış")
    print("=" * 60)
    
    last_prediction_time = 0
    last_status_time = 0
    
    while running:
        try:
            # Raw sample'ları al
            raw_samples = connector.get_raw_samples()
            
            if raw_samples:
                # İşle (FFT hesapla)
                fft_count = engine.process_raw_samples(raw_samples)
            
            current_time = time.time()
            
            # Durum göster
            if current_time - last_status_time >= 1.0:
                last_status_time = current_time
                status = engine.get_buffer_status()
                
                if connector.poor_signal > 50:
                    print(f"\r⚠️  Sinyal zayıf: {connector.poor_signal} | Raw: {status['raw_buffer']} | Features: {status['feature_buffer']}", end='', flush=True)
                else:
                    print(f"\r📊 Raw: {status['raw_buffer']} | Features: {status['feature_buffer']} | Samples: {status['total_samples']}", end='', flush=True)
            
            # Tahmin
            if current_time - last_prediction_time >= PREDICTION_INTERVAL:
                last_prediction_time = current_time
                
                if engine.is_ready():
                    label, confidence, class_probs = engine.predict()
                    
                    if confidence >= args.threshold:
                        color = {'yukarı': '\033[92m', 'aşağı': '\033[91m', 'araba': '\033[93m', 'asagı': '\033[91m'}
                        c = color.get(label, '\033[0m')
                        print(f"\n{c}🎯 Tahmin: {label:10s} | Güven: {confidence*100:.1f}%\033[0m")
                        
                        # Arduino servo kontrolü
                        if arduino_controller is not None:
                            if arduino_controller.send_command(label):
                                print(f"   🤖 Arduino: {label} komutu gönderildi")
                    else:
                        print(f"\n⏳ Belirsiz... | Güven: {confidence*100:.1f}%")
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"\n❌ Hata: {e}")
            import traceback
            traceback.print_exc()
            break
    
    connector.disconnect()
    
    # Arduino bağlantısını kapat
    if arduino_controller is not None:
        arduino_controller.close()
    
    print("\n✅ Program sonlandı.")


if __name__ == "__main__":
    main()
