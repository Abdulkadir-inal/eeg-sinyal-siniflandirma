#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3 Kişi (Apo, Bahadır, Canan) için Model Eğitimi

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
DATA_DIR = SCRIPT_DIR  # 3person_model/ klasörü
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# TCN MODEL
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
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


# ============================================================================
# EĞİTİM FONKSİYONLARI
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(loader), 100 * correct / total


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("🧠 TCN MODEL EĞİTİMİ - 3 KİŞİ (APO, BAHADIR, CANAN)")
    print("=" * 80)
    print(f"📂 Veri Dizini: {DATA_DIR}")
    print(f"🎮 Device: {DEVICE}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"📈 Learning Rate: {LEARNING_RATE}")
    print("-" * 80)
    
    # 1. Veriyi yükle
    X_path = os.path.join(DATA_DIR, 'X_3person.npy')
    y_path = os.path.join(DATA_DIR, 'y_3person.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("❌ Veri dosyaları bulunamadı!")
        print(f"   Önce data_preprocess_3person.py'yi çalıştırın")
        return
    
    print("📁 Veri yükleniyor...")
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Özellik sayısı: {X.shape[2]}")
    
    # Sınıf dağılımı
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Sınıf Dağılımı:")
    label_names = {0: "araba", 1: "aşağı", 2: "yukarı"}
    for label, count in zip(unique, counts):
        print(f"   {label_names[label]:8s}: {count:5d} ({count/len(y)*100:.1f}%)")
    
    # 2. Train/Val split
    print(f"\n🔀 Train/Validation split (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    
    # 3. PyTorch dataset ve dataloader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Model oluştur
    print(f"\n🏗️  TCN Model oluşturuluyor...")
    input_channels = X.shape[2]  # 17
    model = TCN_Model(input_channels=input_channels, num_classes=3).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✅ Model oluşturuldu")
    print(f"   📊 Toplam parametre: {total_params:,}")
    print(f"   🎯 Eğitilebilir: {trainable_params:,}")
    
    # 5. Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 6. Eğitim döngüsü
    print(f"\n🚀 Eğitim başlıyor...")
    print("-" * 80)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Kaydet
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # En iyi model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(DATA_DIR, 'best_model_3person.pth'))
        
        # Print
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"{'✨' if val_acc == best_val_acc else ''}")
    
    print("-" * 80)
    print(f"✅ Eğitim tamamlandı!")
    print(f"🏆 En iyi validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # 7. Final model kaydet
    torch.save(model.state_dict(), os.path.join(DATA_DIR, 'final_model_3person.pth'))
    print(f"💾 Final model kaydedildi: final_model_3person.pth")
    print(f"💾 Best model kaydedildi: best_model_3person.pth")
    
    # 8. Grafik çiz
    print(f"\n📊 Eğitim grafikleri çiziliyor...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss History (3 Person Model)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy History (3 Person Model)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'training_history_3person.png'), dpi=150)
    print(f"   ✅ Grafik kaydedildi: training_history_3person.png")
    
    # 9. Özet
    print("\n" + "=" * 80)
    print("📊 EĞİTİM ÖZETİ (3 KİŞİ)")
    print("=" * 80)
    print(f"👥 Kullanılan Kişiler: APO, BAHADIR, CANAN")
    print(f"📦 Toplam veri: {len(X)}")
    print(f"🎓 Train: {len(X_train)}")
    print(f"✅ Validation: {len(X_val)}")
    print(f"🏆 Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"📈 Final Train Acc: {history['train_acc'][-1]:.2f}%")
    print(f"📉 Final Val Acc: {history['val_acc'][-1]:.2f}%")
    print(f"💾 Model dosyaları:")
    print(f"   • best_model_3person.pth")
    print(f"   • final_model_3person.pth")
    print(f"   • training_history_3person.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
