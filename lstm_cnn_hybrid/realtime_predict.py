#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM+CNN Hibrit Model - Canlı Tahmin (Raw EEG → FFT)
====================================================

Raw EEG sinyalinden FFT hesaplar (eğitimle aynı pipeline).
MindWave'in kendi FFT'sini DEĞİL, bizim hesapladığımız FFT'yi kullanır.

Pipeline:
    Raw EEG → Notch Filter (50Hz) → Bandpass (0.5-50Hz) → FFT → Model

Kullanım:
    python realtime_predict.py --port COM5    (Windows)
    python realtime_predict.py --port /dev/ttyUSB0  (Linux)
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

# Sinyal işleme modülü
from signal_processor import SignalProcessor, BAND_NAMES, SAMPLING_RATE, WINDOW_SIZE

# ============================================================================
# AYARLAR
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# MindWave bağlantı
BAUD_RATE = 57600

# Tahmin ayarları
SEQUENCE_LENGTH = 64  # Model'in beklediği sequence uzunluğu
PREDICTION_INTERVAL = 0.5  # Her 0.5 saniyede bir tahmin
CONFIDENCE_THRESHOLD = 0.6  # Minimum güven skoru


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
# TAHMİN MOTORU (Raw EEG → FFT → Model)
# ============================================================================

class PredictionEngine:
    """Raw EEG'den FFT hesaplar ve model ile tahmin yapar"""
    
    def __init__(self, model_path, scaler_path, config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Config yükle
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Label map yükle
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
        
        print(f"✅ Model yüklendi: {model_path}")
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


# ============================================================================
# ANA UYGULAMA
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTM+CNN Canlı Tahmin (Raw EEG → FFT)')
    parser.add_argument('--port', required=True, help='COM port (örn: COM5 veya /dev/ttyUSB0)')
    parser.add_argument('--threshold', type=float, default=0.6, help='Güven eşiği')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🧠 LSTM+CNN Hibrit Model - Canlı Tahmin")
    print("   (Raw EEG → Filtre → FFT → Model)")
    print("=" * 60)
    
    # Dosya yolları
    model_path = os.path.join(SCRIPT_DIR, 'best_model.pth')
    scaler_path = os.path.join(SCRIPT_DIR, 'scaler.pkl')
    config_path = os.path.join(SCRIPT_DIR, 'config.json')
    
    for path, name in [(model_path, 'Model'), (scaler_path, 'Scaler'), (config_path, 'Config')]:
        if not os.path.exists(path):
            print(f"❌ {name} bulunamadı: {path}")
            return
    
    # Engine
    print("\n📦 Model yükleniyor...")
    engine = PredictionEngine(model_path, scaler_path, config_path)
    
    # MindWave bağlantısı
    print(f"\n📡 MindWave'e bağlanılıyor: {args.port}")
    connector = MindWaveRawConnector(args.port)
    
    if not connector.connect():
        print("❌ MindWave bağlantısı kurulamadı!")
        return
    
    connector.start()
    
    # Ctrl+C handler
    running = True
    def signal_handler(s, frame):
        nonlocal running
        running = False
        print("\n\n⏹️  Durduruluyor...")
    
    sig.signal(sig.SIGINT, signal_handler)
    
    print("\n" + "=" * 60)
    print("🎯 TAHMİN BAŞLADI!")
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
                    else:
                        print(f"\n⏳ Belirsiz... | Güven: {confidence*100:.1f}%")
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"\n❌ Hata: {e}")
            import traceback
            traceback.print_exc()
            break
    
    connector.disconnect()
    print("\n✅ Program sonlandı.")


if __name__ == "__main__":
    main()
