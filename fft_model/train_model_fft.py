#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT Tabanlı Model Eğitimi
=========================
Bu script, Raw EEG'den FFT ile hesaplanmış verilerle
CNN-LSTM, TCN ve Transformer modellerini eğitir.

Giriş: X_fft.npy, y_fft.npy (data_preprocess_fft.py çıktısı)
Çıkış: best_model_fft.pth, training_history_fft.png

Karşılaştırma için orijinal modellerle ayrı dosyalarda tutuluyor.
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
from pathlib import Path
import time

# Dizin ayarları
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR  # fft_model/
OUTPUT_DIR = SCRIPT_DIR  # fft_model/

# Eğitim parametreleri
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== MODEL TANIMLARI ====================

class CNN_LSTM_Model(nn.Module):
    """CNN + LSTM hibrit modeli"""
    def __init__(self, input_channels=9, num_classes=3):
        super(CNN_LSTM_Model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TemporalBlock(nn.Module):
    """TCN için Temporal Block - Causal Convolution"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.padding = padding
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Causal: padding'i kes
        out = self.dropout1(self.relu1(self.bn1(out)))
        
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Causal: padding'i kes
        out = self.dropout2(self.relu2(self.bn2(out)))
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN_Model(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, input_channels=9, num_classes=3, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(TCN_Model, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=padding, dropout=dropout))
        
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


class TransformerModel(nn.Module):
    """Transformer tabanlı model"""
    def __init__(self, input_channels=9, num_classes=3, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_channels, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 128, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=d_model*4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==================== EĞİTİM FONKSİYONLARI ====================

def load_data():
    """FFT verilerini yükle"""
    print("\n📥 FFT verileri yükleniyor...")
    
    X = np.load(DATA_DIR / 'X_fft.npy')
    y = np.load(DATA_DIR / 'y_fft.npy')
    
    with open(DATA_DIR / 'label_map_fft.json', 'r') as f:
        label_map = json.load(f)
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Sınıflar: {label_map}")
    
    return X, y, label_map


def prepare_dataloaders(X, y):
    """DataLoader'ları hazırla"""
    print("\n🔄 Veri setleri ayrılıyor...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Eğitim: {X_train.shape[0]} örnek")
    print(f"   Doğrulama: {X_val.shape[0]} örnek")
    
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
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer):
    """Bir epoch eğitim"""
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
    """Bir epoch doğrulama"""
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


def train_model(model, model_name, train_loader, val_loader, num_classes):
    """Modeli eğit"""
    print(f"\n{'='*60}")
    print(f"🚀 {model_name} EĞİTİMİ")
    print(f"{'='*60}")
    
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Toplam parametre: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch [{epoch+1:2d}/{EPOCHS}] "
                  f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    elapsed = time.time() - start_time
    print(f"   ⏱️  Süre: {elapsed:.1f}s | En iyi: {best_val_acc:.2f}%")
    
    return {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_model_state': best_model_state,
        'elapsed_time': elapsed
    }


def plot_comparison(results):
    """Tüm modellerin karşılaştırma grafiği"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Loss grafikleri
    ax1 = axes[0, 0]
    for idx, result in enumerate(results):
        ax1.plot(result['train_losses'], color=colors[idx], linestyle='-', 
                label=f"{result['model_name']} (Train)", alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Eğitim Loss Karşılaştırması')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for idx, result in enumerate(results):
        ax2.plot(result['val_losses'], color=colors[idx], linestyle='-',
                label=f"{result['model_name']}", linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Doğrulama Loss Karşılaştırması')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Accuracy grafikleri
    ax3 = axes[1, 0]
    for idx, result in enumerate(results):
        ax3.plot(result['train_accs'], color=colors[idx], linestyle='-',
                label=f"{result['model_name']}", linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Eğitim Accuracy Karşılaştırması')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    for idx, result in enumerate(results):
        ax4.plot(result['val_accs'], color=colors[idx], linestyle='-',
                label=f"{result['model_name']} ({result['best_val_acc']:.1f}%)", linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Doğrulama Accuracy Karşılaştırması')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history_fft.png', dpi=150)
    print(f"\n📊 Grafik kaydedildi: training_history_fft.png")
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("🧠 FFT TABANLI MODEL EĞİTİMİ")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    
    # Veriyi yükle
    X, y, label_map = load_data()
    num_classes = len(label_map)
    input_channels = X.shape[2]
    
    # DataLoader'ları hazırla
    train_loader, val_loader = prepare_dataloaders(X, y)
    
    # Modelleri tanımla
    models = [
        (CNN_LSTM_Model(input_channels=input_channels, num_classes=num_classes), "CNN-LSTM"),
        (TCN_Model(input_channels=input_channels, num_classes=num_classes), "TCN"),
        (TransformerModel(input_channels=input_channels, num_classes=num_classes), "Transformer"),
    ]
    
    # Her modeli eğit
    results = []
    for model, model_name in models:
        result = train_model(model, model_name, train_loader, val_loader, num_classes)
        results.append(result)
    
    # Sonuçları karşılaştır
    print("\n" + "=" * 60)
    print("📊 FFT MODEL KARŞILAŞTIRMASI")
    print("=" * 60)
    print(f"{'Model':<15} {'Val Acc':>10} {'Süre':>10}")
    print("-" * 40)
    
    best_result = None
    for result in results:
        print(f"{result['model_name']:<15} {result['best_val_acc']:>9.2f}% {result['elapsed_time']:>9.1f}s")
        if best_result is None or result['best_val_acc'] > best_result['best_val_acc']:
            best_result = result
    
    # En iyi modeli kaydet
    print(f"\n🏆 En iyi model: {best_result['model_name']} ({best_result['best_val_acc']:.2f}%)")
    
    # Model oluştur ve kaydet
    if best_result['model_name'] == "CNN-LSTM":
        best_model = CNN_LSTM_Model(input_channels=input_channels, num_classes=num_classes)
    elif best_result['model_name'] == "TCN":
        best_model = TCN_Model(input_channels=input_channels, num_classes=num_classes)
    else:
        best_model = TransformerModel(input_channels=input_channels, num_classes=num_classes)
    
    best_model.load_state_dict(best_result['best_model_state'])
    torch.save(best_model.state_dict(), OUTPUT_DIR / 'best_model_fft.pth')
    print(f"💾 Model kaydedildi: best_model_fft.pth")
    
    # Tüm modelleri ayrı ayrı kaydet
    for result in results:
        model_filename = f"{result['model_name'].lower().replace('-', '_')}_model_fft.pth"
        
        if result['model_name'] == "CNN-LSTM":
            model = CNN_LSTM_Model(input_channels=input_channels, num_classes=num_classes)
        elif result['model_name'] == "TCN":
            model = TCN_Model(input_channels=input_channels, num_classes=num_classes)
        else:
            model = TransformerModel(input_channels=input_channels, num_classes=num_classes)
        
        model.load_state_dict(result['best_model_state'])
        torch.save(model.state_dict(), OUTPUT_DIR / model_filename)
        print(f"💾 {result['model_name']} kaydedildi: {model_filename}")
    
    # Grafik oluştur
    plot_comparison(results)
    
    # Sonuçları JSON olarak kaydet
    summary = {
        'models': [
            {
                'name': r['model_name'],
                'val_accuracy': r['best_val_acc'],
                'elapsed_time': r['elapsed_time']
            }
            for r in results
        ],
        'best_model': best_result['model_name'],
        'best_accuracy': best_result['best_val_acc'],
        'data_source': 'FFT (Raw EEG -> FFT bands)',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }
    
    with open(OUTPUT_DIR / 'training_results_fft.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ FFT MODEL EĞİTİMİ TAMAMLANDI!")
    print("=" * 60)
    print(f"\n📁 Çıktı dosyaları:")
    print(f"   - best_model_fft.pth (en iyi model)")
    print(f"   - cnn_lstm_model_fft.pth")
    print(f"   - tcn_model_fft.pth")
    print(f"   - transformer_model_fft.pth")
    print(f"   - training_history_fft.png")
    print(f"   - training_results_fft.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
