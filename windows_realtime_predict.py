#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 MindWave Canlı EEG Tahmin Sistemi - Windows Edition
======================================================

ThinkGear Connector üzerinden MindWave Mobile 2 cihazından 
canlı veri alır ve gerçek zamanlı tahmin yapar.

Gereksinimler:
    1. ThinkGear Connector kurulu ve çalışıyor olmalı
    2. pip install torch numpy

Kullanım:
    python windows_realtime_predict.py

Özellikler:
    - Birden fazla model seçeneği (TCN, Transformer, CNN-LSTM)
    - CPU üzerinde çalışır (CUDA gerekmez)
    - ThinkGear Connector'dan JSON veri okur
    - Canlı tahmin ve görselleştirme
"""

import os
import sys
import time
import socket
import json
import numpy as np
from collections import deque
from datetime import datetime

# PyTorch (CPU modu yeterli)
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("❌ PyTorch kurulu değil!")
    print("   Kurulum: pip install torch")
    sys.exit(1)


# ============================================================================
# MODEL TANIMLARI
# ============================================================================

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
    """TCN Model - En iyi performans (%92.44)"""
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
    """Transformer Model (%87.99 @ 80 epoch)"""
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


# ============================================================================
# THINKGEAR CONNECTOR BAĞLANTISI
# ============================================================================

class ThinkGearConnector:
    """ThinkGear Connector'a bağlanır ve JSON veri okur"""
    
    def __init__(self, host='127.0.0.1', port=13854):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""
        
        # Son veriler
        self.last_data = {
            'delta': 0, 'theta': 0,
            'lowAlpha': 0, 'highAlpha': 0,
            'lowBeta': 0, 'highBeta': 0,
            'lowGamma': 0, 'highGamma': 0,
            'attention': 0, 'meditation': 0,
            'poorSignalLevel': 200
        }
    
    def connect(self):
        """ThinkGear Connector'a bağlan"""
        try:
            print(f"\n🔵 ThinkGear Connector'a bağlanılıyor: {self.host}:{self.port}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.host, self.port))
            
            # JSON format iste
            self.sock.send(b'{"enableRawOutput": false, "format": "Json"}\n')
            
            print("✅ ThinkGear Connector'a bağlandı!")
            self.sock.settimeout(1)
            return True
            
        except ConnectionRefusedError:
            print(f"❌ Bağlantı reddedildi: {self.host}:{self.port}")
            print("\n💡 ThinkGear Connector çalışmıyor!")
            print("   1. ThinkGear Connector'ı başlatın")
            print("   2. Sistem tray'de ThinkGear ikonunu kontrol edin")
            print("   3. MindWave cihazının bağlı olduğundan emin olun")
            return False
        except socket.timeout:
            print(f"❌ Bağlantı zaman aşımı")
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
        """ThinkGear'dan JSON veri oku"""
        if not self.sock:
            return None
        
        try:
            data = self.sock.recv(2048).decode('utf-8')
            if not data:
                return None
            
            self.buffer += data
            
            # Satır satır JSON parse et
            lines = self.buffer.split('\r')
            self.buffer = lines[-1]
            
            result = None
            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parsed = json.loads(line)
                    
                    # EEG Power verisi
                    if 'eegPower' in parsed:
                        eeg = parsed['eegPower']
                        self.last_data['delta'] = eeg.get('delta', 0)
                        self.last_data['theta'] = eeg.get('theta', 0)
                        self.last_data['lowAlpha'] = eeg.get('lowAlpha', 0)
                        self.last_data['highAlpha'] = eeg.get('highAlpha', 0)
                        self.last_data['lowBeta'] = eeg.get('lowBeta', 0)
                        self.last_data['highBeta'] = eeg.get('highBeta', 0)
                        self.last_data['lowGamma'] = eeg.get('lowGamma', 0)
                        self.last_data['highGamma'] = eeg.get('highGamma', 0)
                        result = 'eeg'
                    
                    # eSense verisi
                    if 'eSense' in parsed:
                        esense = parsed['eSense']
                        self.last_data['attention'] = esense.get('attention', 0)
                        self.last_data['meditation'] = esense.get('meditation', 0)
                    
                    # Sinyal kalitesi
                    if 'poorSignalLevel' in parsed:
                        self.last_data['poorSignalLevel'] = parsed['poorSignalLevel']
                    
                except json.JSONDecodeError:
                    continue
            
            return result
            
        except socket.timeout:
            return None
        except Exception as e:
            return None
    
    def get_eeg_vector(self):
        """Model için EEG vektörü döndür (9 özellik)"""
        d = self.last_data
        return [
            0,  # Electrode (kullanılmıyor)
            d['delta'],
            d['theta'],
            d['lowAlpha'],
            d['highAlpha'],
            d['lowBeta'],
            d['highBeta'],
            d['lowGamma'],
            d['highGamma']
        ]
    
    def is_signal_good(self):
        """Sinyal kalitesi iyi mi?"""
        return self.last_data['poorSignalLevel'] < 50


# ============================================================================
# CANLI TAHMİN SİSTEMİ
# ============================================================================

class RealtimePredictor:
    """Canlı EEG tahmin sistemi"""
    
    MODELS = {
        '1': {
            'name': 'TCN (En İyi - %92.44)',
            'class': TCN_EEG_Model,
            'file': 'model_experiments/TCN/tcn_best_model.pth',
            'params': {'input_channels': 9, 'num_classes': 3}
        },
        '2': {
            'name': 'Transformer 80 epoch (%87.99)',
            'class': TransformerEEG,
            'file': 'model_experiments/Transformer/transformer_80epoch_best_model.pth',
            'params': {'input_channels': 9, 'num_classes': 3}
        },
        '3': {
            'name': 'Transformer 50 epoch (%86.25)',
            'class': TransformerEEG,
            'file': 'model_experiments/Transformer/transformer_best_model.pth',
            'params': {'input_channels': 9, 'num_classes': 3}
        },
        '4': {
            'name': 'CNN-LSTM (%84.86)',
            'class': CNN_LSTM_Model,
            'file': 'model_experiments/CNN_LSTM/cnn_lstm_best_model.pth',
            'params': {'input_channels': 9, 'num_classes': 3}
        }
    }
    
    LABELS = ['araba', 'yukarı', 'aşağı']  # label_map: araba=0, yukarı=1, aşağı=2
    
    # Eğitim verisinden hesaplanan StandardScaler parametreleri
    # Bu değerler ham EEG verilerinin mean ve std'si
    SCALER_MEAN = [51.84, 451729.19, 100606.08, 29922.44, 30237.58, 
                   19389.26, 16738.54, 9701.50, 4930.83]
    SCALER_STD = [209.64, 564932.61, 150896.39, 42604.21, 42522.26,
                  33606.18, 31252.04, 17717.86, 10128.90]
    
    def __init__(self, window_size=128):
        self.window_size = window_size
        self.device = torch.device('cpu')
        self.model = None
        self.model_name = None
        self.buffer = deque(maxlen=window_size)
        self.thinkgear = ThinkGearConnector()
        
        # İstatistikler
        self.predictions = {label: 0 for label in self.LABELS}
        self.total_predictions = 0
    
    def select_model(self):
        """Model seçimi"""
        print("\n" + "=" * 60)
        print("🧠 MODEL SEÇİMİ")
        print("=" * 60)
        
        for key, info in self.MODELS.items():
            print(f"   {key}. {info['name']}")
        
        print("   q. Çıkış")
        print("-" * 60)
        
        while True:
            choice = input("Model seçin (1-4): ").strip()
            
            if choice.lower() == 'q':
                return False
            
            if choice in self.MODELS:
                return self.load_model(choice)
            
            print("❌ Geçersiz seçim!")
    
    def load_model(self, choice):
        """Model yükle"""
        info = self.MODELS[choice]
        model_path = info['file']
        
        # Script'in bulunduğu dizini baz al
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, model_path)
        
        if not os.path.exists(full_path):
            full_path = model_path
        
        if not os.path.exists(full_path):
            print(f"❌ Model dosyası bulunamadı: {model_path}")
            return False
        
        try:
            print(f"\n📥 Model yükleniyor: {info['name']}")
            
            self.model = info['class'](**info['params'])
            state_dict = torch.load(full_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.model_name = info['name']
            print(f"✅ Model yüklendi!")
            
            return True
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False
    
    def preprocess(self, data_window):
        """Veriyi model için hazırla - StandardScaler normalizasyonu"""
        x = np.array(data_window, dtype=np.float32)
        
        # StandardScaler normalizasyonu: (x - mean) / std
        # Eğitim verisiyle AYNI normalizasyon kullanılmalı!
        for i in range(x.shape[1]):
            x[:, i] = (x[:, i] - self.SCALER_MEAN[i]) / self.SCALER_STD[i]
        
        x = torch.FloatTensor(x).unsqueeze(0)
        return x
    
    def predict(self, data_window):
        """Tahmin yap"""
        if self.model is None:
            return None, None
        
        with torch.no_grad():
            x = self.preprocess(data_window)
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return self.LABELS[predicted.item()], confidence.item()
    
    def run(self):
        """Ana döngü"""
        print("\n" + "=" * 60)
        print("🧠 MindWave Canlı EEG Tahmin Sistemi")
        print("   ThinkGear Connector Edition")
        print("=" * 60)
        
        # Model seç
        if not self.select_model():
            return
        
        # ThinkGear'a bağlan
        print("\n" + "-" * 60)
        if not self.thinkgear.connect():
            return
        
        print("\n" + "=" * 60)
        print(f"📊 Model: {self.model_name}")
        print(f"🎯 Sınıflar: {', '.join(self.LABELS)}")
        print(f"📦 Pencere boyutu: {self.window_size}")
        print("=" * 60)
        print("\n💡 MindWave'i başınıza takın!")
        print("   - Kulak kıskacı kulak memesine")
        print("   - Alın sensörü alnınıza")
        print("\n⏸️  Durdurmak için Ctrl+C")
        print("-" * 60)
        
        try:
            last_prediction_time = 0
            prediction_interval = 1.0
            eeg_received = False
            
            while True:
                # Veri oku
                result = self.thinkgear.read_data()
                
                if result == 'eeg':
                    eeg_received = True
                    # EEG verisi geldi - buffer'a ekle
                    eeg_vector = self.thinkgear.get_eeg_vector()
                    self.buffer.append(eeg_vector)
                    
                    # Sinyal durumu
                    poor = self.thinkgear.last_data['poorSignalLevel']
                    if poor == 0:
                        signal_status = "✅ Mükemmel"
                    elif poor < 50:
                        signal_status = f"⚠️ Orta ({poor})"
                    else:
                        signal_status = f"❌ Zayıf ({poor})"
                    
                    attention = self.thinkgear.last_data['attention']
                    meditation = self.thinkgear.last_data['meditation']
                    
                    print(f"\r📦 Buffer: {len(self.buffer)}/{self.window_size} | Sinyal: {signal_status} | Dikkat: {attention} | Meditasyon: {meditation}   ", end='')
                    
                    # Yeterli veri var mı?
                    current_time = time.time()
                    if len(self.buffer) >= self.window_size and (current_time - last_prediction_time) >= prediction_interval:
                        last_prediction_time = current_time
                        
                        # Tahmin yap
                        data_window = list(self.buffer)[-self.window_size:]
                        label, confidence = self.predict(data_window)
                        
                        if label:
                            self.predictions[label] += 1
                            self.total_predictions += 1
                            
                            print()
                            print("\n" + "=" * 60)
                            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Tahmin #{self.total_predictions}")
                            print(f"🎯 Sonuç: {label.upper()} ({confidence*100:.1f}%)")
                            print("-" * 60)
                            
                            for l in self.LABELS:
                                count = self.predictions[l]
                                pct = (count / self.total_predictions * 100) if self.total_predictions > 0 else 0
                                bar = "█" * int(pct / 5)
                                marker = "👉" if l == label else "  "
                                print(f"{marker} {l:8} : {bar:<20} {pct:.1f}% ({count})")
                            
                            print("=" * 60)
                
                elif not eeg_received:
                    print("\r⏳ EEG verisi bekleniyor... (MindWave'i başınıza takın)", end='')
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n⛔ Kullanıcı tarafından durduruldu")
        finally:
            self.thinkgear.disconnect()
            
            if self.total_predictions > 0:
                print("\n" + "=" * 60)
                print("📊 ÖZET")
                print("=" * 60)
                print(f"Toplam tahmin: {self.total_predictions}")
                for label in self.LABELS:
                    count = self.predictions[label]
                    pct = count / self.total_predictions * 100
                    print(f"   {label}: {count} ({pct:.1f}%)")
                print("=" * 60)


# ============================================================================
# ANA PROGRAM
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("🧠 MindWave Canlı EEG Tahmin Sistemi")
    print("   ThinkGear Connector Edition")
    print("=" * 60)
    print("\n📋 Gereksinimler:")
    print("   1. ThinkGear Connector kurulu ve çalışıyor")
    print("   2. MindWave Mobile 2 bağlı")
    print("   3. Model dosyaları (.pth)")
    
    predictor = RealtimePredictor()
    predictor.run()


if __name__ == "__main__":
    main()
