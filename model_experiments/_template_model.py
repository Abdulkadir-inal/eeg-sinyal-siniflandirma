#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODEL_ADI - [Model açıklaması]

Bu dosya yeni model eklemek için template olarak kullanılabilir.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import time

# ========================================
# KONFIGURASYON - HER MODEL İÇİN DEĞİŞTİR
# ========================================
DATA_DIR = "/home/kadir/sanal-makine/python/proje"
SAVE_DIR = "/home/kadir/sanal-makine/python/proje/model_experiments/MODEL_KLASORU"  # DEĞİŞTİR!
MODEL_NAME = "model_adi"  # DEĞİŞTİR! (örn: "tcn", "eegnet", "transformer")
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# MODEL TANIMI - BURAYA KENDİ MODELİNİ YAZ
# ========================================
class YourModel(nn.Module):
    """
    Kendi model mimarinizi buraya tanımlayın
    """
    def __init__(self, input_channels=9, num_classes=3, samples=128):
        super(YourModel, self).__init__()
        
        # Model katmanlarını buraya ekle
        # Örnek:
        # self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3)
        # self.fc = nn.Linear(hidden_size, num_classes)
        
        pass
    
    def forward(self, x):
        """
        Forward pass
        Input: (batch, seq_len, features)
        Output: (batch, num_classes)
        """
        # Forward pass mantığını buraya yaz
        pass


# ========================================
# VERİ YÜKLEME VE HAZIRLIK
# ========================================
def load_data():
    """Önceden işlenmiş X ve y verilerini yükle"""
    print("\n" + "="*70)
    print("VERİ YÜKLEME")
    print("="*70)
    
    X = np.load(os.path.join(DATA_DIR, 'X.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    
    with open(os.path.join(DATA_DIR, 'label_map.json'), 'r') as f:
        label_map = json.load(f)
    
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    print(f"✓ Sınıflar: {label_map}")
    print(f"✓ Sınıf dağılımı:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = [k for k, v in label_map.items() if v == label][0]
        print(f"   - {label_name} ({label}): {count} örnek")
    
    return X, y, label_map


def prepare_dataloaders(X, y, test_size=0.2, val_size=0.1):
    """Veriyi train, validation ve test setlerine ayır"""
    print("\n" + "="*70)
    print("VERİ SETLERİ HAZIRLANIYOR")
    print("="*70)
    
    # Önce train+val ve test'e ayır
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Sonra train+val'i train ve val'e ayır
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Eğitim seti: {X_train.shape[0]} örnek ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"✓ Doğrulama seti: {X_val.shape[0]} örnek ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"✓ Test seti: {X_test.shape[0]} örnek ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # PyTorch tensorlerine çevir
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # DataLoader oluştur
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, (X_test, y_test)


# ========================================
# EĞİTİM VE DOĞRULAMA
# ========================================
def train_epoch(model, train_loader, criterion, optimizer):
    """Tek epoch eğitim"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion):
    """Tek epoch doğrulama"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


# ========================================
# TEST VE DEĞERLENDİRME
# ========================================
def test_model(model, test_loader, label_map):
    """Model performansını test setinde değerlendir"""
    print("\n" + "="*70)
    print("TEST SETİ DEĞERLENDİRME")
    print("="*70)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Accuracy hesapla
    correct = sum([1 for p, l in zip(all_predictions, all_labels) if p == l])
    accuracy = 100 * correct / len(all_labels)
    
    print(f"\n✓ Test Accuracy: {accuracy:.2f}%")
    
    # Classification report
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    print("\n" + "-"*70)
    print("Classification Report:")
    print("-"*70)
    print(classification_report(all_labels, all_predictions, target_names=label_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, cm, label_names


def mini_prediction_test(model, X_test, y_test, label_map, num_samples=10):
    """
    🎯 MİNİ TAHMİN TESTİ
    Eğitim sonrası 10 örneklik gerçek zamanlı tahmin testi
    Her yeni model için ZORUNLU!
    """
    print("\n" + "="*70)
    print("🎯 MİNİ TAHMİN TESTİ (10 Örnek)")
    print("="*70)
    
    model.eval()
    
    # Random 10 örnek seç
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Reverse label map (id -> name)
    id_to_label = {v: k for k, v in label_map.items()}
    
    correct_count = 0
    
    print("\n{:<5} {:<15} {:<15} {:<10}".format("No", "Gerçek", "Tahmin", "Sonuç"))
    print("-" * 50)
    
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
            result = "✓ DOĞRU" if is_correct else "✗ YANLIŞ"
            
            if is_correct:
                correct_count += 1
            
            print("{:<5} {:<15} {:<15} {} ({:.1f}%)".format(
                i, true_name, pred_name, result, confidence
            ))
    
    print("-" * 50)
    print(f"Mini Test Accuracy: {correct_count}/{num_samples} ({correct_count*10}%)")
    print("="*70)


# ========================================
# GRAFİKLER VE RAPORLAMA
# ========================================
def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Eğitim grafiklerini çiz"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss grafiği
    ax1.plot(epochs, train_losses, 'b-', label='Eğitim Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='Doğrulama Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{MODEL_NAME.upper()} Model - Loss Değişimi', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy grafiği
    ax2.plot(epochs, train_accs, 'b-', label='Eğitim Accuracy', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_accs, 'r-', label='Doğrulama Accuracy', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{MODEL_NAME.upper()} Model - Accuracy Değişimi', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Eğitim grafikleri kaydedildi: {os.path.basename(save_path)}")
    plt.close()


def plot_confusion_matrix(cm, label_names, save_path):
    """Confusion matrix çiz"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, 
                yticklabels=label_names, cbar_kws={'label': 'Örnek Sayısı'})
    plt.title(f'{MODEL_NAME.upper()} Model - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix kaydedildi: {os.path.basename(save_path)}")
    plt.close()


def save_training_log(log_data, save_path):
    """Eğitim logunu kaydet"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"{MODEL_NAME.upper()} MODEL EĞİTİM LOGU\n")
        f.write("="*70 + "\n\n")
        f.write(f"Tarih: {log_data['timestamp']}\n")
        f.write(f"Device: {log_data['device']}\n\n")
        
        f.write("MODEL YAPISI:\n")
        f.write("-"*70 + "\n")
        f.write(f"Model: {log_data['model_name']}\n")
        f.write(f"Toplam Parametreler: {log_data['total_params']:,}\n")
        f.write(f"Eğitilebilir Parametreler: {log_data['trainable_params']:,}\n\n")
        
        f.write("HİPERPARAMETRELER:\n")
        f.write("-"*70 + "\n")
        f.write(f"Batch Size: {log_data['batch_size']}\n")
        f.write(f"Epochs: {log_data['epochs']}\n")
        f.write(f"Learning Rate: {log_data['learning_rate']}\n")
        f.write(f"Optimizer: {log_data['optimizer']}\n\n")
        
        f.write("SONUÇLAR:\n")
        f.write("-"*70 + "\n")
        f.write(f"En İyi Doğrulama Accuracy: {log_data['best_val_acc']:.2f}%\n")
        f.write(f"Final Test Accuracy: {log_data['test_acc']:.2f}%\n")
        f.write(f"Toplam Eğitim Süresi: {log_data['training_time']:.2f} saniye\n\n")
        
        f.write("DETAYLI EPOCH LOGU:\n")
        f.write("-"*70 + "\n")
        for i, (tl, ta, vl, va) in enumerate(zip(log_data['train_losses'], 
                                                  log_data['train_accs'],
                                                  log_data['val_losses'],
                                                  log_data['val_accs']), 1):
            f.write(f"Epoch {i:3d} | Train Loss: {tl:.4f} | Train Acc: {ta:.2f}% | "
                   f"Val Loss: {vl:.4f} | Val Acc: {va:.2f}%\n")
    
    print(f"✓ Eğitim logu kaydedildi: {os.path.basename(save_path)}")


# ========================================
# MAIN FONKSİYON
# ========================================
def main():
    start_time = time.time()
    
    print("\n" + "="*70)
    print(f"{MODEL_NAME.upper()} MODEL EĞİTİMİ")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Veri yükleme
    X, y, label_map = load_data()
    num_classes = len(label_map)
    num_channels = X.shape[2]
    samples = X.shape[1]
    
    # Data loaders
    train_loader, val_loader, test_loader, (X_test, y_test) = prepare_dataloaders(X, y)
    
    # Model oluştur
    print("\n" + "="*70)
    print("MODEL OLUŞTURULUYOR")
    print("="*70)
    
    model = YourModel(
        input_channels=num_channels,
        num_classes=num_classes,
        samples=samples
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model oluşturuldu")
    print(f"✓ Toplam parametreler: {total_params:,}")
    print(f"✓ Eğitilebilir parametreler: {trainable_params:,}")
    
    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Eğitim
    print("\n" + "="*70)
    print(f"EĞİTİM BAŞLIYOR - {EPOCHS} EPOCH")
    print("="*70)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        # Progress bar
        bar_length = 30
        progress = (epoch + 1) / EPOCHS
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"[{bar}] Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_acc:6.2f}% | Val: {val_acc:6.2f}% | Loss: {val_loss:.4f}", end='')
        
        # En iyi model kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{MODEL_NAME}_best_model.pth'))
            print(" ✓ BEST", end='')
        
        print()
    
    # Final model kaydet
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{MODEL_NAME}_final_model.pth'))
    
    # Test seti değerlendirmesi
    test_acc, cm, label_names = test_model(model, test_loader, label_map)
    
    # 🎯 Mini Tahmin Testi
    # Not: Eğitim sonrası hızlı kontrol için
    # Detaylı test için: python3 ../mini_test.py MODEL_KLASORU
    mini_prediction_test(model, X_test, y_test, label_map, num_samples=10)
    
    # Grafikleri kaydet
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                         os.path.join(SAVE_DIR, f'{MODEL_NAME}_training_history.png'))
    
    plot_confusion_matrix(cm, label_names,
                         os.path.join(SAVE_DIR, f'{MODEL_NAME}_confusion_matrix.png'))
    
    # Eğitim logu kaydet
    training_time = time.time() - start_time
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(DEVICE),
        'model_name': MODEL_NAME.upper(),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'optimizer': 'Adam',
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'training_time': training_time,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    
    save_training_log(log_data, os.path.join(SAVE_DIR, f'{MODEL_NAME}_training_log.txt'))
    
    # Özet
    print("\n" + "="*70)
    print("EĞİTİM TAMAMLANDI! 🎉")
    print("="*70)
    print(f"✓ En iyi doğrulama accuracy: {best_val_acc:.2f}%")
    print(f"✓ Test accuracy: {test_acc:.2f}%")
    print(f"✓ Toplam süre: {training_time:.2f} saniye ({training_time/60:.2f} dakika)")
    print(f"\nKaydedilen dosyalar ({SAVE_DIR}):")
    print(f"  📁 {MODEL_NAME}_best_model.pth")
    print(f"  📁 {MODEL_NAME}_final_model.pth")
    print(f"  📊 {MODEL_NAME}_training_history.png")
    print(f"  📊 {MODEL_NAME}_confusion_matrix.png")
    print(f"  📝 {MODEL_NAME}_training_log.txt")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
