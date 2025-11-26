#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Model - 80 EPOCH VERSION

Orijinal transformer_model.py ile aynı ama 80 epoch ile eğitim.
Epoch 50'de 86.25% elde ettik, 80'de daha iyi sonuç bekliyoruz.
Tüm dosyalar "_80epoch" suffix ile kaydedilir.
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
import math

# Konfigurasyon
DATA_DIR = "/home/kadir/sanal-makine/python/proje"
SAVE_DIR = "/home/kadir/sanal-makine/python/proje/model_experiments/Transformer"
MODEL_NAME = "transformer_80epoch"  # ✨ 80 epoch versiyonu
BATCH_SIZE = 32
EPOCHS = 80  # ✨ 50'den 80'e çıkarıldı
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# MODEL TANIMLARI
# ============================================================================

class PositionalEncoding(nn.Module):
    """Pozisyonel encoding - sequence içindeki pozisyon bilgisi"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEEG(nn.Module):
    """
    Transformer model - EEG için optimize edilmiş küçük versiyonu
    Az veri için: daha az katman, daha az head, yüksek dropout
    """
    def __init__(self, input_channels=9, num_classes=3, d_model=64, 
                 nhead=4, num_layers=2, dim_feedforward=256, dropout=0.3):
        super(TransformerEEG, self).__init__()
        
        self.input_projection = nn.Linear(input_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.d_model = d_model
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# EĞİTİM FONKSİYONLARI
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Eğitim epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validation epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ============================================================================
# TEST VE GÖRSELLEŞTIRME
# ============================================================================

def test_model(model, test_loader, device, class_names):
    """Test seti değerlendirme"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    
    print("\n" + "="*70)
    print("TEST SETİ DEĞERLENDİRME")
    print("="*70)
    print(f"\n✓ Test Accuracy: {accuracy:.2f}%")
    print("\n" + "-"*70)
    print("Classification Report:")
    print("-"*70)
    print(classification_report(all_labels, all_preds, 
                                target_names=class_names, digits=2))
    
    return accuracy, all_preds, all_labels

def plot_training_history(history, save_path):
    """Eğitim grafikleri"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss grafiği
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Model Loss (80 Epochs)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy grafiği
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy (80 Epochs)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    plt.title('Confusion Matrix (80 Epochs)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Gerçek Sınıf', fontsize=13)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def mini_prediction_test(model, X_test, y_test, class_names, device, n_samples=10):
    """Mini tahmin testi"""
    model.eval()
    
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    print("\n" + "="*70)
    print("🎯 MİNİ TAHMİN TESTİ (10 Örnek) - 80 EPOCH MODEL")
    print("="*70)
    print(f"\n{'No':<6}{'Gerçek':<16}{'Tahmin':<16}{'Sonuç':<10}")
    print("-"*50)
    
    correct = 0
    with torch.no_grad():
        for i, idx in enumerate(indices, 1):
            x = torch.FloatTensor(X_test[idx:idx+1]).to(device)
            y_true = y_test[idx]
            
            output = model(x)
            probs = F.softmax(output, dim=1)
            confidence = probs.max().item() * 100
            y_pred = output.argmax(dim=1).item()
            
            true_label = class_names[y_true]
            pred_label = class_names[y_pred]
            
            if y_pred == y_true:
                result = f"✓ DOĞRU ({confidence:.1f}%)"
                correct += 1
            else:
                result = f"✗ YANLIŞ ({confidence:.1f}%)"
            
            print(f"{i:<6}{true_label:<16}{pred_label:<16}{result:<10}")
    
    print("-"*50)
    print(f"Mini Test Accuracy: {correct}/{n_samples} ({100*correct/n_samples:.0f}%)")
    print("="*70)


# ============================================================================
# MAIN EĞİTİM LOOP
# ============================================================================

def main():
    print("="*70)
    print("TRANSFORMER MODEL EĞİTİMİ - 80 EPOCH VERSION 🚀")
    print("Attention Mechanism - Extended Training")
    print("="*70)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # ========================================
    # 1. VERİ YÜKLEME
    # ========================================
    print("\n" + "="*70)
    print("VERİ YÜKLEME")
    print("="*70)
    
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    
    with open(os.path.join(DATA_DIR, "label_map.json"), "r") as f:
        label_map = json.load(f)
    
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    print(f"✓ Sınıflar: {label_map}")
    print(f"✓ Sınıf dağılımı:")
    for class_name in class_names:
        class_idx = label_map[class_name]
        count = np.sum(y == class_idx)
        print(f"   - {class_name} ({class_idx}): {count} örnek")
    
    # ========================================
    # 2. VERİ SETLERİNİ HAZIRLAMA
    # ========================================
    print("\n" + "="*70)
    print("VERİ SETLERİ HAZIRLANIYOR")
    print("="*70)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Eğitim seti: {len(X_train)} örnek ({100*len(X_train)/len(X):.1f}%)")
    print(f"✓ Doğrulama seti: {len(X_val)} örnek ({100*len(X_val)/len(X):.1f}%)")
    print(f"✓ Test seti: {len(X_test)} örnek ({100*len(X_test)/len(X):.1f}%)")
    
    # PyTorch tensörlere çevir
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ========================================
    # 3. MODEL OLUŞTURMA
    # ========================================
    print("\n" + "="*70)
    print("MODEL OLUŞTURULUYOR")
    print("="*70)
    
    model = TransformerEEG(
        input_channels=9,
        num_classes=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.3
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Transformer Model oluşturuldu (80 EPOCH VERSION)")
    print(f"✓ Model boyutu: KÜÇÜK (az veri için optimize)")
    print(f"✓ d_model: 64, nhead: 4, layers: 2")
    print(f"✓ Toplam parametreler: {total_params:,}")
    print(f"✓ Eğitilebilir parametreler: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )
    
    # ========================================
    # 4. EĞİTİM
    # ========================================
    print("\n" + "="*70)
    print(f"EĞİTİM BAŞLIYOR - {EPOCHS} EPOCH 🚀")
    print("✨ 50 epoch'ta 86.25% elde ettik, 80'de daha iyisini bekliyoruz!")
    print("="*70)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        progress = int(30 * (epoch + 1) / EPOCHS)
        bar = "█" * progress + "░" * (30 - progress)
        
        best_marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(SAVE_DIR, f"{MODEL_NAME}_best_model.pth"))
            best_marker = "✓ BEST"
        
        # Önemli milestone'ları vurgula
        milestone = ""
        if epoch + 1 == 50:
            milestone = "📍 50 EPOCH (Önceki sonuç: 86.23%)"
        elif epoch + 1 == 60:
            milestone = "📍 60 EPOCH"
        elif epoch + 1 == 70:
            milestone = "📍 70 EPOCH"
        
        print(f"[{bar}] Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_acc:6.2f}% | Val: {val_acc:6.2f}% | "
              f"Loss: {val_loss:.4f} {best_marker} {milestone}")
    
    training_time = time.time() - start_time
    
    # ========================================
    # 5. FINAL MODEL KAYDET
    # ========================================
    torch.save(model.state_dict(), 
              os.path.join(SAVE_DIR, f"{MODEL_NAME}_final_model.pth"))
    
    # ========================================
    # 6. TEST
    # ========================================
    model.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, f"{MODEL_NAME}_best_model.pth")
    ))
    
    test_acc, y_pred, y_true = test_model(model, test_loader, DEVICE, class_names)
    
    # ========================================
    # 7. MİNİ TAHMİN TESTİ
    # ========================================
    mini_prediction_test(model, X_test, y_test, class_names, DEVICE, n_samples=10)
    
    # ========================================
    # 8. GÖRSELLEŞTİRME
    # ========================================
    plot_training_history(history, 
                         os.path.join(SAVE_DIR, f"{MODEL_NAME}_training_history.png"))
    plot_confusion_matrix(y_true, y_pred, class_names,
                         os.path.join(SAVE_DIR, f"{MODEL_NAME}_confusion_matrix.png"))
    
    print(f"\n✓ Eğitim grafikleri kaydedildi: {MODEL_NAME}_training_history.png")
    print(f"✓ Confusion matrix kaydedildi: {MODEL_NAME}_confusion_matrix.png")
    
    # ========================================
    # 9. EĞİTİM LOGU KAYDET
    # ========================================
    log_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_training_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TRANSFORMER MODEL - 80 EPOCH VERSION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Model: Transformer (d_model=64, nhead=4, layers=2)\n")
        f.write(f"Toplam Parametreler: {total_params:,}\n\n")
        f.write(f"Veri:\n")
        f.write(f"- Train: {len(X_train)} samples\n")
        f.write(f"- Validation: {len(X_val)} samples\n")
        f.write(f"- Test: {len(X_test)} samples\n\n")
        f.write(f"Hiperparametreler:\n")
        f.write(f"- Batch Size: {BATCH_SIZE}\n")
        f.write(f"- Epochs: {EPOCHS} (50'den artırıldı)\n")
        f.write(f"- Learning Rate: {LEARNING_RATE}\n")
        f.write(f"- Optimizer: AdamW (weight_decay=0.01)\n")
        f.write(f"- Scheduler: ReduceLROnPlateau (patience=7)\n\n")
        f.write(f"Karşılaştırma:\n")
        f.write(f"- 50 Epoch Sonucu: 86.23% validation, 86.25% test\n")
        f.write(f"- 80 Epoch Sonucu: {best_val_acc:.2f}% validation, {test_acc:.2f}% test\n")
        f.write(f"- İyileşme: {test_acc - 86.25:+.2f}%\n\n")
        f.write(f"Sonuçlar:\n")
        f.write(f"- En İyi Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"- Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"- Toplam Eğitim Süresi: {training_time:.2f} saniye ({training_time/60:.2f} dakika)\n\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Eğitim logu kaydedildi: {MODEL_NAME}_training_log.txt")
    
    # ========================================
    # 10. ÖZET VE KARŞILAŞTIRMA
    # ========================================
    print("\n" + "="*70)
    print("EĞİTİM TAMAMLANDI! 🎉")
    print("="*70)
    print(f"✓ En iyi doğrulama accuracy: {best_val_acc:.2f}%")
    print(f"✓ Test accuracy: {test_acc:.2f}%")
    print(f"✓ Toplam süre: {training_time:.2f} saniye ({training_time/60:.2f} dakika)")
    print(f"\n📊 KARŞILAŞTIRMA:")
    print(f"   50 Epoch: 86.25% test accuracy")
    print(f"   80 Epoch: {test_acc:.2f}% test accuracy")
    print(f"   İyileşme: {test_acc - 86.25:+.2f}%")
    print(f"\nKaydedilen dosyalar ({SAVE_DIR}):")
    print(f"  📁 {MODEL_NAME}_best_model.pth")
    print(f"  📁 {MODEL_NAME}_final_model.pth")
    print(f"  📊 {MODEL_NAME}_training_history.png")
    print(f"  📊 {MODEL_NAME}_confusion_matrix.png")
    print(f"  📝 {MODEL_NAME}_training_log.txt")
    print("="*70)


if __name__ == "__main__":
    main()
