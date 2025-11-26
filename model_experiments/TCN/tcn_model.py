#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCN (Temporal Convolutional Network) Model
EEG sinyal sınıflandırması için optimize edilmiş TCN mimarisi
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

# Konfigurasyon
DATA_DIR = "/home/kadir/sanal-makine/python/proje"
SAVE_DIR = "/home/kadir/sanal-makine/python/proje/model_experiments/TCN"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TCN_Block(nn.Module):
    """
    Temporal Convolutional Network Bloğu
    - Dilated causal convolution kullanır
    - Residual connection ile gradient flow iyileştirir
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCN_Block, self).__init__()
        
        # Causal padding hesaplama
        self.padding = (kernel_size - 1) * dilation
        
        # İlk convolution katmanı
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        # İkinci convolution katmanı
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection için boyut eşitleme
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Residual bağlantı için orijinal girdi
        residual = x if self.downsample is None else self.downsample(x)
        
        # İlk convolution
        out = self.conv1(x)
        # Causal padding için sağ tarafı kes
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # İkinci convolution
        out = self.conv2(out)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Residual connection ekle
        # Boyut uyumsuzluğu varsa residual'ı kes
        if residual.size(2) != out.size(2):
            residual = residual[:, :, :out.size(2)]
        
        return self.relu(out + residual)


class TCN_EEG_Model(nn.Module):
    """
    EEG sinyalleri için TCN Model
    - Çoklu dilated convolution katmanları
    - Giderek artan dilation rate ile uzun vadeli bağımlılıkları yakalar
    - Global average pooling ile boyut azaltma
    """
    def __init__(self, input_channels=9, num_classes=3, num_channels=[64, 128, 256], 
                 kernel_size=3, dropout=0.2):
        super(TCN_EEG_Model, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TCN_Block(in_channels, out_channels, kernel_size, 
                                   dilation_size, dropout))
        
        self.tcn = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_channels[-1], 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Input: (batch, seq_len, features)
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        
        # TCN layers
        x = self.tcn(x)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
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
    ax1.set_title('TCN Model - Loss Değişimi', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy grafiği
    ax2.plot(epochs, train_accs, 'b-', label='Eğitim Accuracy', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_accs, 'r-', label='Doğrulama Accuracy', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('TCN Model - Accuracy Değişimi', fontsize=14, fontweight='bold')
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
    plt.title('TCN Model - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
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
        f.write("TCN MODEL EĞİTİM LOGU\n")
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
    import time
    start_time = time.time()
    
    print("\n" + "="*70)
    print("TCN (TEMPORAL CONVOLUTIONAL NETWORK) MODEL EĞİTİMİ")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Veri yükleme
    X, y, label_map = load_data()
    num_classes = len(label_map)
    input_channels = X.shape[2]  # Feature sayısı
    
    # Data loaders
    train_loader, val_loader, test_loader, (X_test, y_test) = prepare_dataloaders(X, y)
    
    # Model oluştur
    print("\n" + "="*70)
    print("MODEL OLUŞTURULUYOR")
    print("="*70)
    
    model = TCN_EEG_Model(
        input_channels=input_channels,
        num_classes=num_classes,
        num_channels=[64, 128, 256],  # TCN katman boyutları
        kernel_size=3,
        dropout=0.2
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ TCN Model oluşturuldu")
    print(f"✓ Toplam parametreler: {total_params:,}")
    print(f"✓ Eğitilebilir parametreler: {trainable_params:,}")
    
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
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'tcn_best_model.pth'))
            print(" ✓ BEST", end='')
        
        print()
    
    # Final model kaydet
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'tcn_final_model.pth'))
    
    # Test seti değerlendirmesi
    test_acc, cm, label_names = test_model(model, test_loader, label_map)
    
    # 🎯 Mini Tahmin Testi
    mini_prediction_test(model, X_test, y_test, label_map, num_samples=10)
    
    # Grafikleri kaydet
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                         os.path.join(SAVE_DIR, 'tcn_training_history.png'))
    
    plot_confusion_matrix(cm, label_names,
                         os.path.join(SAVE_DIR, 'tcn_confusion_matrix.png'))
    
    # Eğitim logu kaydet
    training_time = time.time() - start_time
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(DEVICE),
        'model_name': 'TCN (Temporal Convolutional Network)',
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
    
    save_training_log(log_data, os.path.join(SAVE_DIR, 'tcn_training_log.txt'))
    
    # Özet
    print("\n" + "="*70)
    print("EĞİTİM TAMAMLANDI! 🎉")
    print("="*70)
    print(f"✓ En iyi doğrulama accuracy: {best_val_acc:.2f}%")
    print(f"✓ Test accuracy: {test_acc:.2f}%")
    print(f"✓ Toplam süre: {training_time:.2f} saniye ({training_time/60:.2f} dakika)")
    print(f"\nKaydedilen dosyalar ({SAVE_DIR}):")
    print(f"  📁 tcn_best_model.pth")
    print(f"  📁 tcn_final_model.pth")
    print(f"  📊 tcn_training_history.png")
    print(f"  📊 tcn_confusion_matrix.png")
    print(f"  📝 tcn_training_log.txt")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
