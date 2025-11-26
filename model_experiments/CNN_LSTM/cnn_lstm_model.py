#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN+LSTM Model - Original Baseline Model

Konvolüsyonel katmanlar + LSTM katmanları kombinasyonu.
Spatial features (CNN) + Temporal patterns (LSTM).
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
SAVE_DIR = "/home/kadir/sanal-makine/python/proje/model_experiments/CNN_LSTM"
MODEL_NAME = "cnn_lstm"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# MODEL TANIMI
# ============================================================================

class CNN_LSTM_Model(nn.Module):
    """
    CNN+LSTM Hybrid Model
    - CNN katmanları: Spatial/feature extraction
    - LSTM katmanları: Temporal sequence modeling
    """
    def __init__(self, input_channels=9, num_classes=3):
        super(CNN_LSTM_Model, self).__init__()
        
        # CNN Layers - Feature Extraction
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # LSTM Layers - Temporal Modeling
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3
        )
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Input: (batch, seq_len, channels)
        # CNN expects: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # CNN Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # CNN Block 3
        x = self.relu(self.bn3(self.conv3(x)))
        
        # LSTM expects: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        x, (hn, cn) = self.lstm(x)
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Fully Connected
        x = self.relu(self.fc1(x))
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
    axes[0].set_title('Model Loss (CNN+LSTM)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy grafiği
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy (CNN+LSTM)', fontsize=14, fontweight='bold')
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
    plt.title('Confusion Matrix (CNN+LSTM)', fontsize=16, fontweight='bold', pad=20)
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
    print("🎯 MİNİ TAHMİN TESTİ (10 Örnek)")
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
    print("CNN+LSTM MODEL EĞİTİMİ")
    print("Spatial Features (CNN) + Temporal Patterns (LSTM)")
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
    
    model = CNN_LSTM_Model(
        input_channels=9,
        num_classes=3
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ CNN+LSTM Model oluşturuldu")
    print(f"✓ Toplam parametreler: {total_params:,}")
    print(f"✓ Eğitilebilir parametreler: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # ========================================
    # 4. EĞİTİM
    # ========================================
    print("\n" + "="*70)
    print(f"EĞİTİM BAŞLIYOR - {EPOCHS} EPOCH")
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
        
        print(f"[{bar}] Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_acc:6.2f}% | Val: {val_acc:6.2f}% | "
              f"Loss: {val_loss:.4f} {best_marker}")
    
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
        f.write("CNN+LSTM MODEL EĞİTİM LOGU\n")
        f.write("="*70 + "\n\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write(f"MODEL YAPISI:\n")
        f.write("-"*70 + "\n")
        f.write(f"Model: CNN+LSTM Hybrid\n")
        f.write(f"Toplam Parametreler: {total_params:,}\n")
        f.write(f"Eğitilebilir Parametreler: {trainable_params:,}\n\n")
        f.write(f"HİPERPARAMETRELER:\n")
        f.write("-"*70 + "\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Optimizer: Adam\n\n")
        f.write(f"SONUÇLAR:\n")
        f.write("-"*70 + "\n")
        f.write(f"En İyi Doğrulama Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Toplam Eğitim Süresi: {training_time:.2f} saniye\n\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Eğitim logu kaydedildi: {MODEL_NAME}_training_log.txt")
    
    # ========================================
    # 10. ÖZET
    # ========================================
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
    print("="*70)


if __name__ == "__main__":
    main()
