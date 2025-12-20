#!/usr/bin/env python3
"""
Gerçek zamanlı sınıflandırma test scripti
MindWave olmadan çalışır (test verisi kullanır)
"""

import sys
import os
import time
import numpy as np

# Proje dizinini ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtime_mindwave_predict import RealtimeMindWaveClassifier

def test_prediction_system():
    """Test verisi ile tahmin sistemini test et"""
    print("\n🧪 CANLI TAHMİN SİSTEMİ TEST")
    print("="*60)
    
    # Classifier oluştur (MindWave bağlantısı olmadan)
    classifier = RealtimeMindWaveClassifier()
    
    # Test verisini yükle
    if not os.path.exists('X.npy'):
        print("❌ Test verisi bulunamadı: X.npy")
        return
    
    X = np.load('X.npy')
    y = np.load('y.npy')
    
    print(f"✅ Test verisi yüklendi: {X.shape}")
    print(f"📊 Sınıflar: {list(classifier.label_map.keys())}")
    print("-"*60)
    
    # 5 rastgele örnek seç
    test_count = 5
    indices = np.random.choice(len(X), test_count, replace=False)
    
    correct = 0
    for i, idx in enumerate(indices, 1):
        window = X[idx]
        true_label = y[idx]
        true_class = classifier.reverse_label_map[true_label]
        
        print(f"\n📦 Test #{i} (Gerçek: {true_class})")
        print("   Sliding window simülasyonu...")
        
        # Her 10 timestep'te bir göster (sliding window simülasyonu)
        for t in range(0, len(window), 10):
            progress = (t / len(window)) * 100
            print(f"   Buffer: {t}/{len(window)} ({progress:.0f}%)", end='\r')
            time.sleep(0.05)  # Görsel efekt
        
        print(f"   Buffer: {len(window)}/{len(window)} (100%)✅")
        
        # Tahmin yap
        predicted_class, confidence, all_probs = classifier.predict(window)
        predicted_name = classifier.reverse_label_map[predicted_class]
        
        # Sonuç
        is_correct = (predicted_class == true_label)
        if is_correct:
            correct += 1
        
        print(f"\n   🎯 Tahmin: {predicted_name} ({confidence*100:.2f}%)")
        print(f"   {'✅ DOĞRU' if is_correct else '❌ YANLIŞ'}")
        
        # Probability bars
        for class_idx in sorted(classifier.reverse_label_map.keys()):
            name = classifier.reverse_label_map[class_idx]
            prob = all_probs[class_idx]
            bar_length = 20
            filled = int(bar_length * prob)
            bar = "█" * filled + "░" * (bar_length - filled)
            marker = "👉" if class_idx == predicted_class else "  "
            print(f"   {marker} {name:10s}: {bar} {prob*100:5.2f}%")
    
    # Özet
    accuracy = (correct / test_count) * 100
    print("\n" + "="*60)
    print(f"📊 TEST SONUCU")
    print(f"   Doğru: {correct}/{test_count}")
    print(f"   Doğruluk: {accuracy:.1f}%")
    print("="*60)
    
    if accuracy >= 80:
        print("✅ Sistem başarıyla test edildi!")
        print("🚀 Canlı MindWave ile kullanıma hazır:")
        print("   ./start_realtime.sh")
    else:
        print("⚠️  Düşük doğruluk! Model yeniden eğitilmeli.")

if __name__ == '__main__':
    try:
        test_prediction_system()
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
