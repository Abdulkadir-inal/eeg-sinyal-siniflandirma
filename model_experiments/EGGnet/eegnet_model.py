#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEGNet Model - EEG Sinyalleri için Özel Tasarlanmış Mimari
Referans: Lawhern et al. (2018) - EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
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

# Konfigurasyon
DATA_DIR = "/home/kadir/sanal-makine/python/proje"
SAVE_DIR = "/home/kadir/sanal-makine/python/proje/model_experiments/EGGnet"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EEGNet(nn.Module):
    """
    EEGNet: Kompakt CNN mimarisi, özellikle EEG-BCI için tasarlanmış
    
    Özellikler:
    - Depthwise ve Separable Convolutions (az parametre)
    - Temporal ve spatial filtering
    - Batch normalization ve dropout
    """
    def __init__(self, num_channels=9, num_classes=3, samples=128, 
                 F1=8, F2=16, D=2, dropout_rate=0.5):
        """
        Args:
            num_channels: EEG kanal sayısı (bizim durumumuzda 9 feature)
            num_classes: Sınıf sayısı
            samples: Zaman noktası sayısı (sequence length)
            F1: İlk temporal filter sayısı
            F2: İkinci temporal filter sayısı
            D: Depthwise convolution depth multiplier
            dropout_rate: Dropout oranı
        """
        super(EEGNet, self).__init__()
        
        # Block 1: Temporal Convolution
        self.temporal_conv = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Block 2: Depthwise Spatial Convolution
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (num_channels, 1), 
                                        groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 3: Separable Convolution
        # Depthwise
        self.separable_conv1 = nn.Conv2d(F1 * D, F1 * D, (1, 16), 
                                         padding=(0, 8), groups=F1 * D, bias=False)
        # Pointwise
        self.separable_conv2 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Flatten boyutunu hesapla
        self.flatten_size = self._get_flatten_size(samples, num_channels)
        
        # Classification layer
        self.fc = nn.Linear(self.flatten_size, num_classes)
    
    def _get_flatten_size(self, samples, num_channels):
        """Flatten sonrası boyutu hesapla"""
        # Dummy input ile boyut hesaplama
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_channels, samples)
            x = self.temporal_conv(dummy_input)
            x = self.bn1(x)
            x = self.depthwise_conv(x)
            x = self.bn2(x)
            x = self.pool1(x)
            x = self.separable_conv1(x)
            x = self.separable_conv2(x)
            x = self.bn3(x)
            x = self.pool2(x)
            return x.numel()
    
    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        # EEGNet expects: (batch, 1, channels, samples)
        x = x.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, features, seq_len)
        
        # Block 1
        x = self.temporal_conv(x)
        x = self.bn1(x)
        
        # Block 2
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 3
        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten ve classification
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        
        return x


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
    """Eğitim sonrası 10 örneklik mini tahmin testi"""
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


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Eğitim grafiklerini çiz"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss grafiği
    ax1.plot(epochs, train_losses, 'b-', label='Eğitim Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='Doğrulama Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('EEGNet Model - Loss Değişimi', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy grafiği
    ax2.plot(epochs, train_accs, 'b-', label='Eğitim Accuracy', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_accs, 'r-', label='Doğrulama Accuracy', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('EEGNet Model - Accuracy Değişimi', fontsize=14, fontweight='bold')
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
    plt.title('EEGNet Model - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
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
        f.write("EEGNet MODEL EĞİTİM LOGU\n")
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


def main():
    start_time = time.time()
    
    print("\n" + "="*70)
    print("EEGNet MODEL EĞİTİMİ")
    print("EEG-BCI için Özel Tasarlanmış Kompakt CNN")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Veri yükleme
    X, y, label_map = load_data()
    num_classes = len(label_map)
    num_channels = X.shape[2]  # Feature sayısı
    samples = X.shape[1]  # Sequence length
    
    # Data loaders
    train_loader, val_loader, test_loader, (X_test, y_test) = prepare_dataloaders(X, y)
    
    # Model oluştur
    print("\n" + "="*70)
    print("MODEL OLUŞTURULUYOR")
    print("="*70)
    
    model = EEGNet(
        num_channels=num_channels,
        num_classes=num_classes,
        samples=samples,
        F1=8,      # İlk temporal filter sayısı
        F2=16,     # İkinci temporal filter sayısı
        D=2,       # Depthwise multiplier
        dropout_rate=0.5
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ EEGNet Model oluşturuldu")
    print(f"✓ Toplam parametreler: {total_params:,}")
    print(f"✓ Eğitilebilir parametreler: {trainable_params:,}")
    print(f"✓ Model özelliği: Depthwise & Separable Convolutions (Kompakt!)")
    
    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                      patience=5)
    
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
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'eegnet_best_model.pth'))
            print(" ✓ BEST", end='')
        
        print()
    
    # Final model kaydet
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'eegnet_final_model.pth'))
    
    # Test seti değerlendirmesi
    test_acc, cm, label_names = test_model(model, test_loader, label_map)
    
    # 🎯 Mini Tahmin Testi
    mini_prediction_test(model, X_test, y_test, label_map, num_samples=10)
    
    # Grafikleri kaydet
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                         os.path.join(SAVE_DIR, 'eegnet_training_history.png'))
    
    plot_confusion_matrix(cm, label_names,
                         os.path.join(SAVE_DIR, 'eegnet_confusion_matrix.png'))
    
    # Eğitim logu kaydet
    training_time = time.time() - start_time
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(DEVICE),
        'model_name': 'EEGNet (Compact Convolutional Neural Network)',
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
    
    save_training_log(log_data, os.path.join(SAVE_DIR, 'eegnet_training_log.txt'))
    
    # Özet
    print("\n" + "="*70)
    print("EĞİTİM TAMAMLANDI! 🎉")
    print("="*70)
    print(f"✓ En iyi doğrulama accuracy: {best_val_acc:.2f}%")
    print(f"✓ Test accuracy: {test_acc:.2f}%")
    print(f"✓ Toplam süre: {training_time:.2f} saniye ({training_time/60:.2f} dakika)")
    print(f"\nKaydedilen dosyalar ({SAVE_DIR}):")
    print(f"  📁 eegnet_best_model.pth")
    print(f"  📁 eegnet_final_model.pth")
    print(f"  📊 eegnet_training_history.png")
    print(f"  📊 eegnet_confusion_matrix.png")
    print(f"  📝 eegnet_training_log.txt")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
