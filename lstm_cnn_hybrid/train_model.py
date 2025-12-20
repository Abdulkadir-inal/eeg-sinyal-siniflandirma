#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM+CNN Hibrit Model Eğitimi
=============================

Architecture:
- CNN: Spatial/spektral pattern'ları yakalar
- LSTM: Temporal dependencies (zaman bağımlılıkları)
- Attention: Önemli kısımlara odaklanır

Overfitting Önlemleri:
- Dropout (0.4)
- L2 Regularization (weight_decay)
- Early Stopping
- Data Augmentation
- Batch Normalization
"""

import os
import numpy as np
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================================
# AYARLAR
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparametreler
BATCH_SIZE = 64
EPOCHS = 50  # Class weights ile daha hızlı yakınsar
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # L2 regularization
DROPOUT = 0.4
EARLY_STOP_PATIENCE = 15


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_data(X, y, noise_std=0.02, scale_range=(0.95, 1.05)):
    """Zaman serisi için data augmentation"""
    X_aug = X.copy()
    
    # 1. Gaussian noise
    noise = np.random.normal(0, noise_std, X_aug.shape).astype(np.float32)
    X_noisy = X + noise
    
    # 2. Random scaling
    X_scaled = X.copy()
    for i in range(len(X_scaled)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        X_scaled[i] = X_scaled[i] * scale
    
    # 3. Time shift (circular)
    X_shifted = np.roll(X, shift=np.random.randint(1, 5), axis=1)
    
    # Birleştir
    X_combined = np.concatenate([X, X_noisy, X_scaled, X_shifted], axis=0)
    y_combined = np.concatenate([y, y, y, y], axis=0)
    
    return X_combined, y_combined


# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

class Attention(nn.Module):
    """Self-Attention for sequence"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        weighted = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_size)
        return weighted, weights


# ============================================================================
# CNN+LSTM HİBRİT MODEL
# ============================================================================

class CNN_LSTM_Hybrid(nn.Module):
    """
    Hibrit Model Mimarisi:
    
    Input -> CNN (spatial features) -> LSTM (temporal) -> Attention -> FC -> Output
    
    CNN: 1D Convolutions for feature extraction
    LSTM: Bidirectional for temporal context
    Attention: Focus on important timesteps
    """
    
    def __init__(self, input_features=15, num_classes=3, 
                 cnn_channels=[32, 64], lstm_hidden=128, dropout=0.4):
        super(CNN_LSTM_Hybrid, self).__init__()
        
        # CNN Layers
        self.cnn = nn.Sequential(
            # Conv1: input_features -> 32
            nn.Conv1d(input_features, cnn_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # Conv2: 32 -> 64
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # Conv3: 64 -> 64 (dilated for larger receptive field)
            nn.Conv1d(cnn_channels[1], cnn_channels[1], kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
        )
        
        # LSTM (Bidirectional)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[1],
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Attention
        self.attention = Attention(lstm_hidden * 2)  # *2 for bidirectional
        
        # Fully Connected
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.BatchNorm1d(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        
        # CNN expects (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        
        # Back to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention
        context, attention_weights = self.attention(lstm_out)  # (batch, hidden*2)
        
        # Classification
        out = self.fc(context)
        
        return out


# ============================================================================
# DAHA BASİT ALTERNATİF MODEL (Overfitting riski düşük)
# ============================================================================

class SimpleCNN_LSTM(nn.Module):
    """Daha basit versiyon - overfitting riski düşük"""
    
    def __init__(self, input_features=15, num_classes=3, dropout=0.4):
        super(SimpleCNN_LSTM, self).__init__()
        
        # Basit CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
        )
        
        # Basit LSTM
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # Output
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
        # Son hidden state'leri birleştir (forward + backward)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        out = self.fc(hidden)
        return out


# ============================================================================
# EĞİTİM FONKSİYONLARI
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping (exploding gradients önle)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100 * correct / total


def validate_epoch(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100 * correct / total


def plot_history(train_losses, val_losses, train_accs, val_accs, filename='training_history.png'):
    """Eğitim geçmişini çiz"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overfitting göstergesi
    gap = [t - v for t, v in zip(train_accs, val_accs)]
    if max(gap) > 10:
        ax2.annotate(f'⚠️ Max gap: {max(gap):.1f}%', 
                    xy=(0.7, 0.1), xycoords='axes fraction',
                    fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, filename), dpi=150)
    plt.close()
    print(f"   📊 Grafik kaydedildi: {filename}")


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("🧠 LSTM+CNN HİBRİT MODEL EĞİTİMİ")
    print("=" * 70)
    print(f"📱 Device: {DEVICE}")
    
    # Veri yükle
    print("\n📂 Veri yükleniyor...")
    
    X_path = os.path.join(SCRIPT_DIR, 'X_data.npy')
    y_path = os.path.join(SCRIPT_DIR, 'y_data.npy')
    
    if not os.path.exists(X_path):
        print("❌ X_data.npy bulunamadı!")
        print("   Önce data_preprocess.py çalıştırın.")
        return
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    with open(os.path.join(SCRIPT_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    
    with open(os.path.join(SCRIPT_DIR, 'label_map.json'), 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    print(f"   ✅ X shape: {X.shape}")
    print(f"   ✅ y shape: {y.shape}")
    print(f"   ✅ Özellik sayısı: {config['num_features']}")
    print(f"   ✅ Sınıf sayısı: {config['num_classes']}")
    
    # Train/Val split
    print("\n🔀 Veri ayrılıyor...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} örnek")
    print(f"   Val: {len(X_val)} örnek")
    
    # Data Augmentation (sadece train)
    print("\n🔧 Data Augmentation uygulanıyor...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    print(f"   ✅ Train genişletildi: {len(X_train)} -> {len(X_train_aug)}")
    
    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_aug),
        torch.LongTensor(y_train_aug)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model seçimi
    print("\n🏗️  Model oluşturuluyor...")
    USE_SIMPLE_MODEL = True  # Basit model daha az overfitting yapar
    
    if USE_SIMPLE_MODEL:
        print("   📌 Simple CNN+LSTM seçildi (overfitting riski düşük)")
        model = SimpleCNN_LSTM(
            input_features=config['num_features'],
            num_classes=config['num_classes'],
            dropout=DROPOUT
        ).to(DEVICE)
    else:
        print("   📌 Full CNN+LSTM+Attention seçildi")
        model = CNN_LSTM_Hybrid(
            input_features=config['num_features'],
            num_classes=config['num_classes'],
            dropout=DROPOUT
        ).to(DEVICE)
    
    # Model özeti
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Toplam parametreler: {total_params:,}")
    print(f"   Eğitilecek parametreler: {trainable_params:,}")
    
    # Sınıf ağırlıklarını hesapla (imbalance için)
    print(f"\n⚖️  Sınıf ağırlıkları hesaplanıyor...")
    unique, counts = np.unique(y_train_aug, return_counts=True)
    total = len(y_train_aug)
    class_weights = []
    for i in range(config['num_classes']):
        count = counts[np.where(unique == i)[0][0]] if i in unique else 1
        weight = total / (config['num_classes'] * count)
        class_weights.append(weight)
        class_name = label_map[str(i)]
        print(f"   {class_name:10s}: ağırlık = {weight:.3f} ({count}/{total})")
    
    # Optimizer ve Loss
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )
    
    # Eğitim
    print(f"\n🚀 Eğitim başlıyor... ({EPOCHS} epoch)")
    print("=" * 70)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        # Overfitting gap hesapla
        gap = train_acc - val_acc
        
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Val: {val_loss:.4f} / {val_acc:.2f}% | "
              f"Gap: {gap:.1f}%", end="")
        
        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(SCRIPT_DIR, 'best_model.pth'))
            print(f" ✅ Best!", end="")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"\n\n⏸️  Early Stopping: {EARLY_STOP_PATIENCE} epoch iyileşme yok")
                break
        
        print()
    
    # Son model kaydet
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, os.path.join(SCRIPT_DIR, 'final_model.pth'))
    
    # Grafik
    plot_history(train_losses, val_losses, train_accs, val_accs)
    
    # Sonuçlar
    print("\n" + "=" * 70)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("=" * 70)
    print(f"🏆 En iyi doğrulama accuracy: {best_val_acc:.2f}%")
    print(f"📊 Son train-val gap: {train_accs[-1] - val_accs[-1]:.1f}%")
    
    if train_accs[-1] - val_accs[-1] > 10:
        print("⚠️  UYARI: Hala overfitting olabilir!")
        print("   Öneriler:")
        print("   - Daha fazla veri topla")
        print("   - Dropout'u artır (0.5)")
        print("   - Model karmaşıklığını azalt")
    
    print(f"\n💾 Kaydedilen dosyalar:")
    print(f"   - best_model.pth (en iyi)")
    print(f"   - final_model.pth (son)")
    print(f"   - training_history.png")


if __name__ == "__main__":
    main()
