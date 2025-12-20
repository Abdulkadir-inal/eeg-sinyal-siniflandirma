#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM+CNN Hibrit Model - Parametreli Eğitim Pipeline
====================================================

Farklı sequence_length değerleri ile deney yapmak için.
Modelleri ayrı dosyalarda saklar, karşılaştırma yapılabilir.

Kullanım:
    python train_experiment.py --seq-len 96
    python train_experiment.py --seq-len 128
    python train_experiment.py --seq-len 64   # baseline tekrarı
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# AYARLAR
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'fft_model', 'data_filtered')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FFT bant isimleri
BAND_NAMES = ['Delta', 'Theta', 'Low Alpha', 'High Alpha', 
              'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']

# Sınıf eşleştirmeleri
CLASS_MAP = {
    'yukarı': 0, 'yukari': 0,
    'aşağı': 1, 'asagı': 1, 'asagi': 1,
    'araba': 2
}

# Hiperparametreler
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
EARLY_STOP_PATIENCE = 10


# ============================================================================
# MODEL
# ============================================================================

class SimpleCNN_LSTM(nn.Module):
    """SimpleCNN_LSTM - Overfitting riski düşük basit model"""
    
    def __init__(self, input_features=15, num_classes=3, dropout=0.4):
        super(SimpleCNN_LSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
        )
        
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
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
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        out = self.fc(hidden)
        return out


# ============================================================================
# VERİ İŞLEME
# ============================================================================

def load_csv_files():
    """CSV dosyalarını yükle"""
    print("\n📂 CSV dosyaları yükleniyor...")
    
    all_data = []
    
    for class_name in ['yukarı', 'asagı', 'araba']:
        class_dir = os.path.join(DATA_DIR, class_name)
        
        if not os.path.exists(class_dir):
            alt_names = {'yukarı': 'yukari', 'asagı': 'asagi', 'aşağı': 'asagi'}
            if class_name in alt_names:
                class_dir = os.path.join(DATA_DIR, alt_names[class_name])
        
        if not os.path.exists(class_dir):
            print(f"   ⚠️  {class_name} klasörü bulunamadı: {class_dir}")
            continue
        
        csv_files = glob.glob(os.path.join(class_dir, '*.csv'))
        print(f"   📁 {class_name}: {len(csv_files)} dosya")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                missing_bands = [b for b in BAND_NAMES if b not in df.columns]
                if missing_bands:
                    print(f"      ⚠️  {os.path.basename(csv_file)}: Eksik bantlar")
                    continue
                
                band_data = df[BAND_NAMES].values
                band_data = np.nan_to_num(band_data, nan=0.0, posinf=0.0, neginf=0.0)
                band_data = np.log1p(np.abs(band_data))
                
                all_data.append({
                    'file': os.path.basename(csv_file),
                    'class': class_name,
                    'label': CLASS_MAP.get(class_name, 0),
                    'data': band_data
                })
                
            except Exception as e:
                print(f"      ❌ {os.path.basename(csv_file)}: {e}")
    
    return all_data


def create_sequences(all_data, sequence_length):
    """Sequence'lar oluştur"""
    print(f"\n🔄 Sequence'lar oluşturuluyor (uzunluk={sequence_length})...")
    
    X_list, y_list = [], []
    step = sequence_length // 2  # %50 overlap
    
    for item in all_data:
        data = item['data']
        label = item['label']
        
        for i in range(0, len(data) - sequence_length, step):
            seq = data[i:i + sequence_length]
            X_list.append(seq)
            y_list.append(label)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    print(f"   ✅ X shape: {X.shape}")
    print(f"   ✅ y shape: {y.shape}")
    
    return X, y


def add_derived_features(X):
    """Türetilmiş özellikler ekle (8 → 15)"""
    delta, theta = X[:, :, 0:1], X[:, :, 1:2]
    low_alpha, high_alpha = X[:, :, 2:3], X[:, :, 3:4]
    low_beta, high_beta = X[:, :, 4:5], X[:, :, 5:6]
    low_gamma, mid_gamma = X[:, :, 6:7], X[:, :, 7:8]
    
    alpha_total = low_alpha + high_alpha
    beta_total = low_beta + high_beta
    gamma_total = low_gamma + mid_gamma
    
    eps = 1e-6
    theta_beta_ratio = theta / (beta_total + eps)
    alpha_beta_ratio = alpha_total / (beta_total + eps)
    theta_alpha_ratio = theta / (alpha_total + eps)
    engagement = beta_total / (alpha_total + theta + eps)
    
    X_extended = np.concatenate([
        X, alpha_total, beta_total, gamma_total,
        theta_beta_ratio, alpha_beta_ratio, theta_alpha_ratio, engagement
    ], axis=-1)
    
    print(f"   ✅ Özellikler: {X.shape[-1]} → {X_extended.shape[-1]}")
    return X_extended


def normalize_data(X):
    """Normalize et"""
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(original_shape)
    
    return X, scaler


def augment_data(X, y, noise_std=0.02, scale_range=(0.95, 1.05)):
    """Data augmentation"""
    noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
    X_noisy = X + noise
    
    X_scaled = X.copy()
    for i in range(len(X_scaled)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        X_scaled[i] = X_scaled[i] * scale
    
    X_shifted = np.roll(X, shift=np.random.randint(1, 5), axis=1)
    
    X_combined = np.concatenate([X, X_noisy, X_scaled, X_shifted], axis=0)
    y_combined = np.concatenate([y, y, y, y], axis=0)
    
    return X_combined, y_combined


# ============================================================================
# EĞİTİM
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100 * correct / total


def validate_epoch(model, val_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
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


def plot_history(train_losses, val_losses, train_accs, val_accs, filename):
    """Eğitim grafiği"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTM+CNN Deney Eğitimi')
    parser.add_argument('--seq-len', type=int, default=96, 
                        help='Sequence uzunluğu (varsayılan: 96)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Epoch sayısı (varsayılan: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (varsayılan: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (varsayılan: {LEARNING_RATE})')
    args = parser.parse_args()
    
    seq_len = args.seq_len
    prefix = f"seq{seq_len}"
    
    print("\n" + "=" * 70)
    print(f"🧠 LSTM+CNN DENEY EĞİTİMİ - Sequence Length: {seq_len}")
    print("=" * 70)
    print(f"📱 Device: {DEVICE}")
    print(f"📁 Prefix: {prefix}_*.pth")
    
    # 1. Veri yükle
    all_data = load_csv_files()
    if not all_data:
        print("❌ Veri bulunamadı!")
        return
    
    # 2. Sequence'lar oluştur
    X, y = create_sequences(all_data, seq_len)
    
    # 3. Türetilmiş özellikler
    X = add_derived_features(X)
    
    # 4. Normalize
    X, scaler = normalize_data(X)
    
    # 5. Label map
    label_map = {'0': 'yukarı', '1': 'aşağı', '2': 'araba'}
    
    # 6. Config kaydet
    config = {
        'sequence_length': seq_len,
        'num_features': X.shape[-1],
        'num_classes': len(label_map),
        'band_names': BAND_NAMES
    }
    
    config_path = os.path.join(SCRIPT_DIR, f'{prefix}_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Config: {prefix}_config.json")
    
    # Scaler kaydet
    scaler_path = os.path.join(SCRIPT_DIR, f'{prefix}_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler: {prefix}_scaler.pkl")
    
    # Label map kaydet
    label_map_path = os.path.join(SCRIPT_DIR, f'{prefix}_label_map.json')
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"💾 Label map: {prefix}_label_map.json")
    
    # 7. Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n🔀 Train: {len(X_train)} | Val: {len(X_val)}")
    
    # 8. Augmentation
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    print(f"🔧 Augmented: {len(X_train)} → {len(X_train_aug)}")
    
    # 9. DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 10. Model
    model = SimpleCNN_LSTM(
        input_features=config['num_features'],
        num_classes=config['num_classes'],
        dropout=DROPOUT
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🏗️  Model parametreleri: {total_params:,}")
    
    # 11. Sınıf ağırlıkları (imbalance için)
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
    
    # 11. Optimizer
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    # 12. Eğitim
    print(f"\n🚀 Eğitim başlıyor... ({args.epochs} epoch)")
    print("=" * 70)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        gap = train_acc - val_acc
        
        print(f"Epoch [{epoch+1:2d}/{args.epochs}] "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Val: {val_loss:.4f} / {val_acc:.2f}% | "
              f"Gap: {gap:.1f}%", end="")
        
        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'sequence_length': seq_len,
            }, os.path.join(SCRIPT_DIR, f'{prefix}_best_model.pth'))
            print(f" ✅ Best!", end="")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"\n\n⏸️  Early Stopping: {EARLY_STOP_PATIENCE} epoch iyileşme yok")
                break
        
        print()
    
    # Final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'sequence_length': seq_len,
    }, os.path.join(SCRIPT_DIR, f'{prefix}_final_model.pth'))
    
    # Grafik
    plot_path = os.path.join(SCRIPT_DIR, f'{prefix}_training_history.png')
    plot_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # Sonuçlar
    print("\n" + "=" * 70)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("=" * 70)
    print(f"🏆 En iyi validation accuracy: {best_val_acc:.2f}%")
    print(f"📊 Son train-val gap: {train_accs[-1] - val_accs[-1]:.1f}%")
    
    print(f"\n💾 Kaydedilen dosyalar ({prefix}_*):")
    print(f"   - {prefix}_best_model.pth")
    print(f"   - {prefix}_final_model.pth")
    print(f"   - {prefix}_config.json")
    print(f"   - {prefix}_scaler.pkl")
    print(f"   - {prefix}_label_map.json")
    print(f"   - {prefix}_training_history.png")
    
    # Log dosyasını güncelle
    log_path = os.path.join(SCRIPT_DIR, 'EXPERIMENT_LOG.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    log_entry = f"""
--- DENEY SONUCU ({timestamp}) ---
Sequence Length: {seq_len}
Best Validation Accuracy: {best_val_acc:.2f}%
Final Train-Val Gap: {train_accs[-1] - val_accs[-1]:.1f}%
Epoch Sayısı: {epoch + 1}
Model Dosyası: {prefix}_best_model.pth
"""
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    print(f"\n📝 Log güncellendi: EXPERIMENT_LOG.txt")
    
    return best_val_acc


if __name__ == "__main__":
    main()
