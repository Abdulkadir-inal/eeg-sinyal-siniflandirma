"""
Alternatif Model Mimarileri - Transformer, TCN, EEGNet
CNN-LSTM modeli ile karşılaştırma için
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class TransformerEEG(nn.Module):
    """
    Transformer tabanlı EEG sınıflandırıcısı
    
    Avantajlar:
    - Uzun bağımlılıkları iyi yakalar
    - Paralel işleme uygun
    - Self-attention ile önemli noktaları vurgular
    
    Dezavantajlar:
    - Daha fazla parametre (yüksek karmaşıklık)
    - Daha yavaş tahmin
    """
    
    def __init__(self, input_size=15, seq_len=96, num_classes=3, d_model=64, nhead=4, num_layers=2):
        super(TransformerEEG, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc1 = nn.Linear(d_model * seq_len, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def _create_positional_encoding(self, seq_len, d_model):
        """Positional encoding oluştur"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        if self.pos_encoding.device != x.device:
            self.pos_encoding = self.pos_encoding.to(x.device)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)  # [batch, seq_len, d_model]
        
        # Flatten
        x = x.reshape(batch_size, -1)
        
        # Classification
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        
        return x


# ============================================================================
# TCN (TEMPORAL CONVOLUTIONAL NETWORK)
# ============================================================================

class ResidualBlock(nn.Module):
    """TCN Residual Block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.3)
        
        # 1x1 conv for residual connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        
        # First conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        
        # Second conv
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.skip:
            residual = self.skip(residual)
        
        out = out + residual
        out = torch.relu(out)
        
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network
    
    Avantajlar:
    - Çok hızlı (paralel işleme)
    - Düşük latency
    - Az parametre
    
    Dezavantajlar:
    - Kısa-vadeli bağımlılıkları daha iyi yakalar
    """
    
    def __init__(self, input_size=15, seq_len=96, num_classes=3):
        super(TCN, self).__init__()
        
        # Input projection
        self.input_proj = nn.Conv1d(input_size, 64, kernel_size=1)
        
        # TCN layers with increasing dilation
        self.tcn1 = ResidualBlock(64, 64, kernel_size=3, dilation=1)
        self.tcn2 = ResidualBlock(64, 128, kernel_size=3, dilation=2)
        self.tcn3 = ResidualBlock(128, 256, kernel_size=3, dilation=4)
        
        # Classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: [batch, seq_len, features] → [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # TCN blocks
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        
        # Pooling
        x = self.pool(x)  # [batch, 256, 1]
        x = x.squeeze(-1)  # [batch, 256]
        
        # Classification
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# EEGNET
# ============================================================================

class EEGNet(nn.Module):
    """
    EEGNet - EEG sınıflandırması için özel tasarlanmış hafif model
    
    Avantajlar:
    - Çok az parametre (hafif)
    - Hızlı (gömülü sistemler için uygun)
    - Depthwise separable convolution
    
    Dezavantajlar:
    - Daha düşük doğruluk
    - Kısıtlı model kapasitesi
    """
    
    def __init__(self, input_size=15, seq_len=96, num_classes=3):
        super(EEGNet, self).__init__()
        
        # Block 1: Spatial convolution
        self.conv1 = nn.Conv2d(1, 8, (1, input_size), padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Block 2: Temporal + Depthwise
        self.conv2 = nn.Conv2d(8, 16, (3, 1), padding=(1, 0), groups=8)
        self.bn2 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((4, 1))
        self.dropout1 = nn.Dropout(0.25)
        
        # Block 3: Separable
        self.conv3 = nn.Conv2d(16, 16, (3, 1), padding=(1, 0), groups=16)
        self.conv3_sep = nn.Conv2d(16, 32, (1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((8, 1))
        self.dropout2 = nn.Dropout(0.25)
        
        # Flatten size
        self.flatten_size = self._get_flatten_size(seq_len)
        
        # Classification
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def _get_flatten_size(self, seq_len):
        """Flatten boyutunu hesapla"""
        # After conv1: [batch, 8, seq_len, 1]
        # After pool1: [batch, 16, seq_len//4, 1]
        # After pool2: [batch, 32, seq_len//32, 1]
        after_pool = (seq_len // 4) // 8
        return 32 * after_pool
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        # Convert to [batch, 1, seq_len, features]
        x = x.unsqueeze(1)
        
        # Block 1
        x = self.conv1(x)  # [batch, 8, seq_len, 1]
        x = self.bn1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.conv3_sep(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=100, device='cpu', model_name='model'):
    """Model eğit"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{'='*60}")
    print(f"🧠 {model_name} eğitiliyor...")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # LR scheduler
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️  Early stopping at epoch {epoch+1}")
                break
    
    return history, best_val_acc


def calculate_metrics(model, data_loader, device='cpu'):
    """Metrikleri hesapla"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Accuracy
    accuracy = (all_preds == all_labels).mean() * 100
    
    # F1 Score (macro)
    from sklearn.metrics import f1_score, classification_report
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return accuracy, f1


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def main():
    """Tüm modelleri eğit ve karşılaştır"""
    
    print("\n" + "="*70)
    print("🧬 ALTERNATİF EEG MODELLERİ BENCHMARK")
    print("="*70)
    
    # Veri yükle (CNN-LSTM'den kullanılan aynı veri)
    print("\n📂 Veri yükleniyor...")
    
    try:
        X_data = np.load('X_data.npy')
        y_data = np.load('y_data.npy')
        print(f"✅ Veri yüklendi: {X_data.shape}")
    except:
        print("❌ Veri dosyaları bulunamadı. Lütfen X_data.npy ve y_data.npy oluşturun.")
        return
    
    # Train-Val-Test split
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    # Models to test
    models_config = {
        'Transformer': TransformerEEG(input_size=15, seq_len=X_train.shape[1], num_classes=3),
        'TCN': TCN(input_size=15, seq_len=X_train.shape[1], num_classes=3),
        'EEGNet': EEGNet(input_size=15, seq_len=X_train.shape[1], num_classes=3)
    }
    
    results = {}
    
    # Train each model
    for model_name, model in models_config.items():
        print(f"\n{'='*70}")
        print(f"Training {model_name}...")
        print(f"{'='*70}")
        
        model = model.to(device)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Parameters: {params:,}")
        
        # Train
        history, best_val_acc = train_model(
            model, train_loader, val_loader, 
            epochs=100, device=device, model_name=model_name
        )
        
        # Test
        test_acc, test_f1 = calculate_metrics(model, test_loader, device)
        
        results[model_name] = {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'best_val_acc': best_val_acc,
            'parameters': params
        }
        
        print(f"\n✅ {model_name} Results:")
        print(f"   Test Accuracy: {test_acc:.2f}%")
        print(f"   Test F1 Score: {test_f1:.2f}")
    
    # Summary table
    print(f"\n\n{'='*70}")
    print("📊 BENCHMARK SONUÇLARI")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Accuracy':<12} {'F1 Score':<12} {'Parameters':<12}")
    print("-"*70)
    
    for model_name, metrics in results.items():
        acc = metrics['test_accuracy']
        f1 = metrics['test_f1']
        params = metrics['parameters']
        print(f"{model_name:<15} {acc:.2f}%{'':<7} {f1:.2f}{'':<8} {params:,}")
    
    # Save results
    with open('alternative_models_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Sonuçlar kaydedildi: alternative_models_results.json")


if __name__ == "__main__":
    main()
