#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canlı MindWave Verisi ile Gerçek Zamanlı EEG Sınıflandırma
WSL2 üzerinden Windows proxy sunucusu ile MindWave cihazını okur
"""

import sys
import os
import time
import numpy as np
import torch
import json
from collections import deque
from datetime import datetime

# Parent dizinini ekle (mindwave_wsl2.py için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindwave_wsl2 import MindWaveWSL2
from train_model import CNN_LSTM_Model

class RealtimeMindWaveClassifier:
    """Canlı MindWave verisi ile gerçek zamanlı sınıflandırma"""
    
    def __init__(self, window_size=128, host='10.255.255.254', port=5555):
        """
        Args:
            window_size: Model için gerekli pencere boyutu (varsayılan: 128)
            host: Windows proxy IP adresi
            port: TCP port numarası
        """
        self.window_size = window_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # MindWave bağlantısı
        self.mindwave = MindWaveWSL2(host=host, port=port)
        
        # Veri tamponu (sliding window için)
        self.data_buffer = deque(maxlen=window_size * 2)  # Extra buffer
        
        # EEG özellik sırası (preprocessing ile aynı)
        self.eeg_features = [
            "Electrode",
            "Delta", "Theta", 
            "Low Alpha", "High Alpha",
            "Low Beta", "High Beta",
            "Low Gamma", "Mid Gamma"
        ]
        
        # Model ve etiketleri yükle
        self.model = None
        self.label_map = None
        self.reverse_label_map = None
        self.scaler_mean = None
        self.scaler_std = None
        
        # İstatistikler
        self.total_predictions = 0
        self.class_counts = {}
        
        self.load_model_and_labels()
        
    def load_model_and_labels(self):
        """Eğitilmiş modeli ve normalizasyon parametrelerini yükle"""
        model_path = 'best_model.pth'
        label_path = 'label_map.json'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model dosyası bulunamadı: {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"❌ Etiket dosyası bulunamadı: {label_path}")
            
        # Label map yükle
        with open(label_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # İstatistik için class_counts initialize et
        for class_name in self.label_map.keys():
            self.class_counts[class_name] = 0
        
        # Model yükle
        num_classes = len(self.label_map)
        self.model = CNN_LSTM_Model(input_channels=9, num_classes=num_classes).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        # Checkpoint direkt state_dict ise
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direkt state_dict kaydedilmiş
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Normalizasyon parametrelerini yükle (eğitim verisinden)
        if os.path.exists('X.npy'):
            X_train = np.load('X.npy')
            # Her özellik için ayrı ayrı mean ve std
            self.scaler_mean = X_train.reshape(-1, 9).mean(axis=0)
            self.scaler_std = X_train.reshape(-1, 9).std(axis=0)
        else:
            print("⚠️  X.npy bulunamadı, normalizasyon uygulanmayacak")
            self.scaler_mean = np.zeros(9)
            self.scaler_std = np.ones(9)
        
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
        print(f"✅ Model yüklendi ({gpu_info})")
        print(f"📊 Sınıflar: {list(self.label_map.keys())}")
        print(f"🔢 Normalizasyon: mean={self.scaler_mean[0]:.4f}, std={self.scaler_std[0]:.4f}")
        
    def extract_features(self, mindwave_data):
        """MindWave data dictionary'den 9 özelliği çıkar"""
        features = []
        
        # Electrode (raw_value'den)
        features.append(float(mindwave_data.get('raw_value', 0)))
        
        # 8 EEG bandı
        features.append(float(mindwave_data.get('delta', 0)))
        features.append(float(mindwave_data.get('theta', 0)))
        features.append(float(mindwave_data.get('low_alpha', 0)))
        features.append(float(mindwave_data.get('high_alpha', 0)))
        features.append(float(mindwave_data.get('low_beta', 0)))
        features.append(float(mindwave_data.get('high_beta', 0)))
        features.append(float(mindwave_data.get('low_gamma', 0)))
        features.append(float(mindwave_data.get('mid_gamma', 0)))
        
        return np.array(features, dtype=np.float32)
    
    def normalize_window(self, window):
        """Pencereyi normalize et (eğitim verisi ile aynı)"""
        # window shape: (128, 9)
        normalized = (window - self.scaler_mean) / (self.scaler_std + 1e-8)
        return normalized
    
    def predict(self, window):
        """Tek bir pencere için tahmin yap"""
        # Normalize et
        window_normalized = self.normalize_window(window)
        
        # Torch tensor'e çevir (batch_size=1)
        x = torch.FloatTensor(window_normalized).unsqueeze(0).to(self.device)
        
        # Tahmin
        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()
        
        return predicted_class, confidence, probs[0].cpu().numpy()
    
    def print_prediction(self, predicted_class, confidence, all_probs):
        """Tahmin sonucunu görsel olarak yazdır"""
        class_name = self.reverse_label_map[predicted_class]
        
        # İstatistikleri güncelle
        self.total_predictions += 1
        self.class_counts[class_name] += 1
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print("\n" + "="*60)
        print(f"⏰ {timestamp} | Tahmin #{self.total_predictions}")
        print(f"🎯 Sonuç: {class_name.upper()} ({confidence*100:.2f}%)")
        print("-"*60)
        
        # Tüm sınıfların olasılıklarını göster
        for class_idx in sorted(self.reverse_label_map.keys()):
            name = self.reverse_label_map[class_idx]
            prob = all_probs[class_idx]
            
            # Progress bar
            bar_length = 30
            filled = int(bar_length * prob)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            marker = "👉" if class_idx == predicted_class else "  "
            print(f"{marker} {name:10s}: {bar} {prob*100:5.2f}%")
        
        print("="*60)
        
        # Özet istatistikler
        print(f"\n📈 İstatistikler:")
        for name, count in self.class_counts.items():
            percentage = (count / self.total_predictions * 100) if self.total_predictions > 0 else 0
            print(f"   {name}: {count} ({percentage:.1f}%)")
    
    def run(self, prediction_interval=1.0, min_signal_quality=50):
        """
        Canlı veri okuma ve tahmin döngüsü
        
        Args:
            prediction_interval: Tahminler arası minimum süre (saniye)
            min_signal_quality: Minimum sinyal kalitesi (0-200, düşük=iyi)
        """
        print("\n" + "="*60)
        print("🧠 CANLI MINDWAVE EEG SINIFLANDIRMA")
        print("="*60)
        print(f"📡 MindWave bağlantısı kuruluyor...")
        print(f"🖥️  Host: {self.mindwave.host}:{self.mindwave.port}")
        print(f"🪟  Pencere boyutu: {self.window_size}")
        print(f"⚡ Cihaz: {self.device}")
        print("="*60)
        
        # MindWave'e bağlan
        if not self.mindwave.connect():
            print("\n❌ MindWave bağlantısı kurulamadı!")
            print("\n💡 Kontrol listesi:")
            print("1. Windows'ta proxy sunucusu çalışıyor mu?")
            print("   → python windows_proxy.py")
            print("2. MindWave cihazı açık ve bilgisayara bağlı mı?")
            print("3. IP adresi doğru mu? (Varsayılan: 10.255.255.254)")
            return
        
        print("\n✅ MindWave bağlantısı başarılı!")
        print("📊 Veri toplama başladı...\n")
        print("⏸️  Çıkmak için Ctrl+C")
        print("-"*60)
        
        last_prediction_time = 0
        
        try:
            while True:
                # Veri paketini oku
                if self.mindwave.parse_packet():
                    data = self.mindwave.data
                    
                    # Sinyal kalitesini kontrol et
                    signal_quality = data.get('signal_quality', 200)
                    
                    if signal_quality > min_signal_quality:
                        print(f"⚠️  Zayıf sinyal: {signal_quality}/200 (iyi=0)", end='\r')
                        continue
                    
                    # Özellikleri çıkar ve buffer'a ekle
                    features = self.extract_features(data)
                    self.data_buffer.append(features)
                    
                    # Buffer'ı göster
                    buffer_fill = len(self.data_buffer)
                    print(f"📦 Buffer: {buffer_fill}/{self.window_size} | Sinyal: {signal_quality}/200", end='\r')
                    
                    # Yeterli veri toplandı mı?
                    if len(self.data_buffer) >= self.window_size:
                        # Tahmin zamanı geldi mi?
                        current_time = time.time()
                        if current_time - last_prediction_time >= prediction_interval:
                            # En son 128 örneği al
                            window = np.array(list(self.data_buffer)[-self.window_size:])
                            
                            # Tahmin yap
                            predicted_class, confidence, all_probs = self.predict(window)
                            
                            # Sonucu yazdır
                            self.print_prediction(predicted_class, confidence, all_probs)
                            
                            last_prediction_time = current_time
                
                time.sleep(0.01)  # CPU kullanımını azalt
                
        except KeyboardInterrupt:
            print("\n\n🛑 Kullanıcı tarafından durduruldu")
        except Exception as e:
            print(f"\n\n❌ Hata: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.mindwave.disconnect()
            print("\n✅ Bağlantı kapatıldı")
            print(f"📊 Toplam tahmin sayısı: {self.total_predictions}")

def main():
    """Ana program"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Canlı MindWave EEG Sınıflandırma')
    parser.add_argument('--host', type=str, default='10.255.255.254',
                        help='Windows proxy IP adresi (varsayılan: 10.255.255.254)')
    parser.add_argument('--port', type=int, default=5555,
                        help='TCP port (varsayılan: 5555)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Tahminler arası süre (saniye, varsayılan: 1.0)')
    parser.add_argument('--min-quality', type=int, default=50,
                        help='Minimum sinyal kalitesi (0-200, varsayılan: 50)')
    
    args = parser.parse_args()
    
    # Sınıflandırıcıyı başlat
    classifier = RealtimeMindWaveClassifier(
        window_size=128,
        host=args.host,
        port=args.port
    )
    
    # Çalıştır
    classifier.run(
        prediction_interval=args.interval,
        min_signal_quality=args.min_quality
    )

if __name__ == '__main__':
    main()
