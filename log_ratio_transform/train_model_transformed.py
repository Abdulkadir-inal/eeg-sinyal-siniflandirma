#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformed Data (Log + Oran Formülleri) için Model Eğitimi

Girdi: 17 özellik (9 orijinal + 8 oran)
Model: TCN (En iyi performans gösteren model)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# ============================================================================
# AYARLAR
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR  # log_ratio_transform/ klasörü
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# TCN MODEL (En iyi performans)
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
            # Daha yakın geçmişe odaklan: [1, 2, 2] yerine [1, 2, 4]
            dilation_size = 1 if i == 0 else 2
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                       dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                       dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


# ============================================================================
# VERİ YÜKLEME
# ============================================================================

def load_data():
    print("\n📂 Transformed veri yükleniyor...")
    
    X_path = os.path.join(DATA_DIR, 'X_transformed.npy')
    y_path = os.path.join(DATA_DIR, 'y_transformed.npy')
    label_path = os.path.join(DATA_DIR, 'label_map_transformed.json')
    
    if not os.path.exists(X_path):
        print("❌ X_transformed.npy bulunamadı!")
        print("   Önce data_preprocess_transformed.py çalıştırın.")
        return None, None, None
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    with open(label_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    print(f"   ✅ X shape: {X.shape}")
    print(f"   ✅ y shape: {y.shape}")
    print(f"   ✅ Sınıflar: {label_map}")
    
    return X, y, label_map


def prepare_dataloaders(X, y):
    print("\n🔀 Veri setleri ayrılıyor...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"   Eğitim: {X_train.shape[0]} örnek")
    print(f"   Doğrulama: {X_val.shape[0]} örnek")
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader


# ============================================================================
# EĞİTİM FONKSİYONLARI
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer):
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
    
    return running_loss / len(train_loader), 100 * correct / total


def validate_epoch(model, val_loader, criterion):
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
    
    return running_loss / len(val_loader), 100 * correct / total


def plot_training_history(train_losses, val_losses, train_accs, val_accs, filename='training_history_transformed.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Eğitim Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Doğrulama Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Eğitim ve Doğrulama Loss (Transformed Data)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, 'b-', label='Eğitim Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Doğrulama Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Eğitim ve Doğrulama Accuracy (Transformed Data)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, filename), dpi=150)
    print(f"\n   📊 Grafik kaydedildi: {filename}")
    plt.close()


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("🧪 TRANSFORMED VERİ İLE TCN MODEL EĞİTİMİ")
    print("=" * 70)
    print(f"📱 Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Veri yükle
    X, y, label_map = load_data()
    if X is None:
        return
    
    num_classes = len(label_map)
    input_channels = X.shape[2]  # 17 özellik
    
    print(f"\n📊 Veri özellikleri:")
    print(f"   Girdi kanalları: {input_channels}")
    print(f"   Sınıf sayısı: {num_classes}")
    
    # DataLoader'ları hazırla
    train_loader, val_loader = prepare_dataloaders(X, y)
    
    # Model oluştur
    print(f"\n🏗️  TCN Model oluşturuluyor...")
    model = TCN_Model(input_channels=input_channels, num_classes=num_classes).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Toplam parametreler: {total_params:,}")
    print(f"   Eğitilecek parametreler: {trainable_params:,}")
    
    # Eğitim ayarları
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Eğitim
    print(f"\n🚀 Eğitim başlıyor... ({EPOCHS} epoch)")
    print("=" * 70)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Val: {val_loss:.4f} / {val_acc:.2f}%", end="")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(DATA_DIR, 'best_model_transformed.pth'))
            print(f" ✅ Best!", end="")
        print()
    
    # Son modeli kaydet
    torch.save(model.state_dict(), os.path.join(DATA_DIR, 'final_model_transformed.pth'))
    
    # Grafik çiz
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Sonuçlar
    print("\n" + "=" * 70)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("=" * 70)
    print(f"🏆 En iyi doğrulama accuracy: {best_val_acc:.2f}%")
    
    # Karşılaştırma
    print(f"\n📊 KARŞILAŞTIRMA:")
    print(f"   Önceki model (9 özellik): ~95.70% (TCN)")
    print(f"   Yeni model (17 özellik):  {best_val_acc:.2f}%")
    
    if best_val_acc > 95.70:
        print(f"   🎉 İYİLEŞME: +{best_val_acc - 95.70:.2f}%")
    elif best_val_acc < 95.70:
        print(f"   ⚠️  DÜŞÜŞ: {best_val_acc - 95.70:.2f}%")
    else:
        print(f"   ➖ AYNI")
    
    print(f"\n💾 Kaydedilen dosyalar:")
    print(f"   - best_model_transformed.pth")
    print(f"   - final_model_transformed.pth")
    print(f"   - training_history_transformed.png")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
