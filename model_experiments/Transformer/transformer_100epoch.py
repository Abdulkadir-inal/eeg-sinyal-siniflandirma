#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Model - 100 EPOCH VERSION with DETAILED METRICS

80 epoch'ta 87.99% elde ettik, 100 epoch'ta daha da iyi olmalı.
Tüm dosyalar "_100epoch" suffix ile kaydedilir.
DETAYLI METRİK RAPORLAMA ile!
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import time
import math

# Konfigurasyon
DATA_DIR = "/home/kadir/sanal-makine/python/proje"
SAVE_DIR = "/home/kadir/sanal-makine/python/proje/model_experiments/Transformer"
MODEL_NAME = "transformer_100epoch"
BATCH_SIZE = 32
EPOCHS = 100  # ✨ 100 epoch
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
    """Transformer model - EEG için optimize edilmiş küçük versiyonu"""
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
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# EĞİTİM FONKSİYONLARI (Detaylı Metriklerle)
# ============================================================================

def train_epoch_detailed(model, train_loader, criterion, optimizer, device, class_names):
    """Eğitim epoch - detaylı metriklerle"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
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
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Sınıf bazlı metrikler
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        }
    
    return epoch_loss, epoch_acc, class_metrics

def validate_epoch_detailed(model, val_loader, criterion, device, class_names):
    """Validation epoch - detaylı metriklerle"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Sınıf bazlı metrikler
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        }
    
    return epoch_loss, epoch_acc, class_metrics


# ============================================================================
# TEST VE GÖRSELLEŞTIRME
# ============================================================================

def test_model_detailed(model, test_loader, device, class_names):
    """Test seti - ÇOK DETAYLI değerlendirme"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    
    print("\n" + "="*80)
    print("📊 DETAYLI TEST SETİ DEĞERLENDİRME")
    print("="*80)
    
    # Genel Accuracy
    print(f"\n{'GENEL METRİKLER':-^80}")
    print(f"✓ Test Accuracy: {accuracy:.2f}%")
    print(f"✓ Doğru Tahmin: {np.sum(np.array(all_preds) == np.array(all_labels))}/{len(all_labels)}")
    print(f"✓ Yanlış Tahmin: {np.sum(np.array(all_preds) != np.array(all_labels))}/{len(all_labels)}")
    
    # Sınıf bazlı detaylı analiz
    print(f"\n{'SINIF BAZLI DETAYLI ANALİZ':-^80}")
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    for i, class_name in enumerate(class_names):
        print(f"\n📌 {class_name.upper()} Sınıfı:")
        print(f"   • Precision:  {precision[i]:.4f} ({precision[i]*100:.2f}%)")
        print(f"   • Recall:     {recall[i]:.4f} ({recall[i]*100:.2f}%)")
        print(f"   • F1-Score:   {f1[i]:.4f} ({f1[i]*100:.2f}%)")
        print(f"   • Support:    {support[i]} örnek")
        print(f"   • Doğru:      {np.sum((np.array(all_labels) == i) & (np.array(all_preds) == i))}/{support[i]}")
    
    # Macro ve Weighted Average
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    print(f"\n{'ORTALAMA METRİKLER':-^80}")
    print(f"Macro Average:")
    print(f"   • Precision:  {precision_macro:.4f} ({precision_macro*100:.2f}%)")
    print(f"   • Recall:     {recall_macro:.4f} ({recall_macro*100:.2f}%)")
    print(f"   • F1-Score:   {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"\nWeighted Average:")
    print(f"   • Precision:  {precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
    print(f"   • Recall:     {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
    print(f"   • F1-Score:   {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    
    # Güven skorları analizi
    all_probs = np.array(all_probs)
    avg_confidence = np.mean(np.max(all_probs, axis=1))
    print(f"\n{'GÜVEN SKORU ANALİZİ':-^80}")
    print(f"✓ Ortalama Güven: {avg_confidence*100:.2f}%")
    print(f"✓ Min Güven: {np.min(np.max(all_probs, axis=1))*100:.2f}%")
    print(f"✓ Max Güven: {np.max(np.max(all_probs, axis=1))*100:.2f}%")
    
    # Sklearn classification report
    print(f"\n{'SKLEARN CLASSIFICATION REPORT':-^80}")
    print(classification_report(all_labels, all_preds, 
                                target_names=class_names, digits=4))
    
    return accuracy, all_preds, all_labels

def plot_training_history(history, save_path):
    """Eğitim grafikleri"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss grafiği
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Model Loss (100 Epochs)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy grafiği
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy (100 Epochs)', fontsize=14, fontweight='bold')
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
    plt.title('Confusion Matrix (100 Epochs)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Gerçek Sınıf', fontsize=13)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def mini_prediction_test(model, X_test, y_test, class_names, device, n_samples=10):
    """Mini tahmin testi"""
    model.eval()
    
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    print("\n" + "="*80)
    print("🎯 MİNİ TAHMİN TESTİ (10 Örnek)")
    print("="*80)
    print(f"\n{'No':<6}{'Gerçek':<16}{'Tahmin':<16}{'Güven':<12}{'Sonuç':<10}")
    print("-"*60)
    
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
                result = "✓ DOĞRU"
                correct += 1
            else:
                result = "✗ YANLIŞ"
            
            print(f"{i:<6}{true_label:<16}{pred_label:<16}{confidence:>6.2f}%    {result:<10}")
    
    print("-"*60)
    print(f"Mini Test Accuracy: {correct}/{n_samples} ({100*correct/n_samples:.0f}%)")
    print("="*80)


# ============================================================================
# MAIN EĞİTİM LOOP
# ============================================================================

def main():
    print("="*80)
    print("TRANSFORMER MODEL EĞİTİMİ - 100 EPOCH VERSION 🚀")
    print("Detaylı Metrik Raporlama ile Extended Training")
    print("="*80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # VERİ YÜKLEME
    print("\n" + "="*80)
    print("VERİ YÜKLEME")
    print("="*80)
    
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    
    with open(os.path.join(DATA_DIR, "label_map.json"), "r") as f:
        label_map = json.load(f)
    
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    print(f"✓ Sınıflar: {label_map}")
    
    # VERİ SETLERİNİ HAZIRLAMA
    print("\n" + "="*80)
    print("VERİ SETLERİ HAZIRLANIYOR")
    print("="*80)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Eğitim: {len(X_train)} örnek ({100*len(X_train)/len(X):.1f}%)")
    print(f"✓ Validation: {len(X_val)} örnek ({100*len(X_val)/len(X):.1f}%)")
    print(f"✓ Test: {len(X_test)} örnek ({100*len(X_test)/len(X):.1f}%)")
    
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
    
    # MODEL OLUŞTURMA
    print("\n" + "="*80)
    print("MODEL OLUŞTURULUYOR")
    print("="*80)
    
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
    
    print(f"✓ Transformer Model (100 EPOCH VERSION)")
    print(f"✓ Parametreler: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )
    
    # EĞİTİM
    print("\n" + "="*80)
    print(f"EĞİTİM BAŞLIYOR - {EPOCHS} EPOCH 🚀")
    print("✨ 50 epoch: 86.25% | 80 epoch: 87.99% | 100 epoch: ???")
    print("="*80)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_metrics = train_epoch_detailed(
            model, train_loader, criterion, optimizer, DEVICE, class_names
        )
        val_loss, val_acc, val_metrics = validate_epoch_detailed(
            model, val_loader, criterion, DEVICE, class_names
        )
        
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
        if epoch + 1 in [50, 80, 90]:
            milestone = f"📍 {epoch+1} EPOCH"
        
        # Her 10 epoch'ta detaylı metrikler
        if (epoch + 1) % 10 == 0:
            print(f"\n{' EPOCH '+str(epoch+1)+' DETAY ':-^80}")
            print(f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Loss: {val_loss:.4f} {best_marker}")
            print(f"Val Metrics:")
            for cls in class_names:
                m = val_metrics[cls]
                print(f"  • {cls:8s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        else:
            print(f"[{bar}] Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {train_acc:6.2f}% | Val: {val_acc:6.2f}% | "
                  f"Loss: {val_loss:.4f} {best_marker} {milestone}")
    
    training_time = time.time() - start_time
    
    # FINAL MODEL KAYDET
    torch.save(model.state_dict(), 
              os.path.join(SAVE_DIR, f"{MODEL_NAME}_final_model.pth"))
    
    # TEST - DETAYLI
    model.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, f"{MODEL_NAME}_best_model.pth")
    ))
    
    test_acc, y_pred, y_true = test_model_detailed(model, test_loader, DEVICE, class_names)
    
    # MİNİ TAHMİN TESTİ
    mini_prediction_test(model, X_test, y_test, class_names, DEVICE, n_samples=10)
    
    # GÖRSELLEŞTİRME
    plot_training_history(history, 
                         os.path.join(SAVE_DIR, f"{MODEL_NAME}_training_history.png"))
    plot_confusion_matrix(y_true, y_pred, class_names,
                         os.path.join(SAVE_DIR, f"{MODEL_NAME}_confusion_matrix.png"))
    
    print(f"\n✓ Grafikler kaydedildi")
    
    # ÖZET VE KARŞILAŞTIRMA
    print("\n" + "="*80)
    print("EĞİTİM TAMAMLANDI! 🎉")
    print("="*80)
    print(f"✓ Best validation: {best_val_acc:.2f}%")
    print(f"✓ Test accuracy: {test_acc:.2f}%")
    print(f"✓ Süre: {training_time:.2f}s ({training_time/60:.2f}dk)")
    print(f"\n📊 KARŞILAŞTIRMA:")
    print(f"   50 Epoch: 86.25%")
    print(f"   80 Epoch: 87.99% (+1.74%)")
    print(f"  100 Epoch: {test_acc:.2f}% ({test_acc - 87.99:+.2f}%)")
    print("="*80)


if __name__ == "__main__":
    main()
