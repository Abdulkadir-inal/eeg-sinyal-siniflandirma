#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini Tahmin Testi - Eğitilmiş Modelleri Test Et

Bu script, eğitilmiş modelleri 10 rastgele örnek üzerinde test eder.
Eğitim yapmadan sadece inference için kullanılır.

Kullanım:
    python3 mini_test.py TCN
    python3 mini_test.py EGGnet
    python3 mini_test.py [model_klasoru]
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import json

# Model klasörünü command line'dan al
if len(sys.argv) < 2:
    print("❌ Kullanım: python3 mini_test.py MODEL_KLASORU")
    print("Örnek: python3 mini_test.py TCN")
    print("       python3 mini_test.py EGGnet")
    sys.exit(1)

MODEL_FOLDER = sys.argv[1]
DATA_DIR = "/home/kadir/sanal-makine/python/proje"
MODEL_DIR = f"/home/kadir/sanal-makine/python/proje/model_experiments/{MODEL_FOLDER}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data():
    """Test verilerini yükle"""
    print("\n" + "="*70)
    print("VERİ YÜKLEME")
    print("="*70)
    
    X = np.load(os.path.join(DATA_DIR, 'X.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    
    with open(os.path.join(DATA_DIR, 'label_map.json'), 'r') as f:
        label_map = json.load(f)
    
    # Test set oluştur (aynı random seed ile)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"✓ Test seti: {X_test.shape[0]} örnek")
    print(f"✓ Sınıflar: {label_map}")
    
    return X_test, y_test, label_map


def load_model(model_path):
    """Model yükle - otomatik olarak model tipini algıla"""
    print("\n" + "="*70)
    print("MODEL YÜKLEME")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        sys.exit(1)
    
    # Model tipine göre import
    if MODEL_FOLDER == "TCN":
        from TCN.tcn_model import TCN_EEG_Model
        model = TCN_EEG_Model(input_channels=9, num_classes=3)
    elif MODEL_FOLDER == "EGGnet":
        from EGGnet.eegnet_model import EEGNet
        model = EEGNet(num_channels=9, num_classes=3, samples=128)
    elif MODEL_FOLDER == "Transformer":
        from Transformer.transformer_model import TransformerEEG
        model = TransformerEEG(input_channels=9, num_classes=3)
    else:
        print(f"❌ Bilinmeyen model klasörü: {MODEL_FOLDER}")
        print("Desteklenen modeller: TCN, EGGnet, Transformer")
        sys.exit(1)
    
    # Model ağırlıklarını yükle
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model yüklendi: {os.path.basename(model_path)}")
    print(f"✓ Model tipi: {MODEL_FOLDER}")
    print(f"✓ Device: {DEVICE}")
    
    return model


def mini_prediction_test(model, X_test, y_test, label_map, num_samples=10):
    """
    🎯 MİNİ TAHMİN TESTİ
    10 rastgele örnek üzerinde model performansını test et
    """
    print("\n" + "="*70)
    print(f"🎯 MİNİ TAHMİN TESTİ - {MODEL_FOLDER.upper()} MODEL")
    print("="*70)
    
    # Random 10 örnek seç
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Reverse label map (id -> name)
    id_to_label = {v: k for k, v in label_map.items()}
    
    correct_count = 0
    
    print("\n{:<5} {:<15} {:<15} {:<20}".format("No", "Gerçek", "Tahmin", "Sonuç"))
    print("-" * 60)
    
    with torch.no_grad():
        for i, idx in enumerate(indices, 1):
            # Tek örnek al
            sample = torch.FloatTensor(X_test[idx:idx+1]).to(DEVICE)
            true_label = y_test[idx]
            
            # Tahmin yap
            output = model(sample)
            probabilities = F.softmax(output, dim=1)
            predicted_id = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_id].item() * 100
            
            # Label isimlerini al
            true_name = id_to_label[true_label]
            pred_name = id_to_label[predicted_id]
            
            # Doğru mu kontrol et
            is_correct = true_label == predicted_id
            result = f"✓ DOĞRU ({confidence:.1f}%)" if is_correct else f"✗ YANLIŞ ({confidence:.1f}%)"
            
            if is_correct:
                correct_count += 1
            
            # Renkli çıktı için
            color = "\033[92m" if is_correct else "\033[91m"  # Green or Red
            reset = "\033[0m"
            
            print("{:<5} {:<15} {:<15} {}{}{}".format(
                i, true_name, pred_name, color, result, reset
            ))
    
    print("-" * 60)
    accuracy = correct_count * 10
    
    # Renk seç
    if accuracy >= 80:
        color = "\033[92m"  # Green
    elif accuracy >= 60:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[91m"  # Red
    
    print(f"Mini Test Accuracy: {color}{correct_count}/{num_samples} ({accuracy}%)\033[0m")
    print("="*70)
    
    return accuracy


def full_test_evaluation(model, X_test, y_test, label_map):
    """Tüm test seti üzerinde detaylı değerlendirme"""
    print("\n" + "="*70)
    print("DETAYLI TEST DEĞERLENDİRME (Tüm Test Seti)")
    print("="*70)
    
    model.eval()
    all_predictions = []
    
    # Batch batch test et
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.FloatTensor(X_test[i:i+batch_size]).to(DEVICE)
            outputs = model(batch)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    # Accuracy hesapla
    correct = sum([1 for p, l in zip(all_predictions, y_test) if p == l])
    total_accuracy = 100 * correct / len(y_test)
    
    print(f"\n✓ Toplam Test Accuracy: {total_accuracy:.2f}% ({correct}/{len(y_test)})")
    
    # Sınıf bazlı accuracy
    id_to_label = {v: k for k, v in label_map.items()}
    print("\n📊 Sınıf Bazlı Performans:")
    print("-" * 60)
    
    for class_id, class_name in sorted(id_to_label.items()):
        class_mask = y_test == class_id
        class_preds = [all_predictions[i] for i in range(len(y_test)) if class_mask[i]]
        class_true = y_test[class_mask]
        
        class_correct = sum([1 for p, t in zip(class_preds, class_true) if p == t])
        class_total = len(class_true)
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        
        print(f"{class_name:15} : {class_acc:6.2f}% ({class_correct:4d}/{class_total:4d})")
    
    print("="*70)


def main():
    print("\n" + "🎯" + "="*68 + "🎯")
    print(f"   MİNİ TAHMİN TEST ARACI - {MODEL_FOLDER.upper()} MODEL")
    print("🎯" + "="*68 + "🎯")
    
    # Veri yükle
    X_test, y_test, label_map = load_test_data()
    
    # En iyi modeli yükle
    model_name = MODEL_FOLDER.lower()
    # EGGnet klasörü için dosya ismi 'eegnet' (küçük e)
    if MODEL_FOLDER == "EGGnet":
        model_name = "eegnet"
    
    best_model_path = os.path.join(MODEL_DIR, f"{model_name}_best_model.pth")
    model = load_model(best_model_path)
    
    # Mini test (10 örnek)
    mini_accuracy = mini_prediction_test(model, X_test, y_test, label_map, num_samples=10)
    
    # Detaylı test değerlendirmesi (tüm test seti)
    full_test_evaluation(model, X_test, y_test, label_map)
    
    print("\n✅ Test tamamlandı!")
    print(f"Model: {MODEL_FOLDER}")
    print(f"Model dosyası: {best_model_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
