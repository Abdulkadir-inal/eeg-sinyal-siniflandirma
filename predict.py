#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import json
from collections import deque
import time

DATA_DIR = "/home/kadir/sanal-makine/python/proje"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 128
EEG_FEATURES = ["Electrode", "Delta", "Theta", "Low Alpha", "High Alpha", "Low Beta", "High Beta", "Low Gamma", "Mid Gamma"]

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_channels=9, num_classes=2):
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

def load_model_and_labels():
    print("\nModel yukleniyor...")
    
    with open(os.path.join(DATA_DIR, 'label_map.json'), 'r') as f:
        label_map = json.load(f)
    
    reverse_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    
    model = CNN_LSTM_Model(input_channels=len(EEG_FEATURES), num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'best_model.pth')))
    model.eval()
    
    print(f"   Model yuklendi: best_model.pth")
    print(f"   Siniflar: {label_map}")
    print(f"   Device: {DEVICE}")
    
    return model, reverse_label_map

def predict_single_window(model, window_data, reverse_label_map):
    with torch.no_grad():
        window_tensor = torch.FloatTensor(window_data).unsqueeze(0).to(DEVICE)
        output = model(window_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
        
        label_name = reverse_label_map[predicted_class]
        
        return label_name, confidence, probabilities[0].cpu().numpy()

def simulate_realtime_prediction(model, reverse_label_map, num_predictions=10):
    print(f"\nSimulasyon: Test verisinden {num_predictions} ornek seciliyor...")
    
    X_test = np.load(os.path.join(DATA_DIR, 'X.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y.npy'))
    
    indices = np.random.choice(len(X_test), num_predictions, replace=False)
    
    print("\n" + "="*70)
    print("GERCEK ZAMANLI TAHMIN SIMULASYONU")
    print("="*70)
    
    correct = 0
    for idx, test_idx in enumerate(indices):
        window = X_test[test_idx]
        true_label = reverse_label_map[y_test[test_idx]]
        
        predicted_label, confidence, probs = predict_single_window(model, window, reverse_label_map)
        
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct += 1
            status = "✅ DOGRU"
        else:
            status = "❌ YANLIS"
        
        print(f"\nOrnek {idx+1}:")
        print(f"   Gercek etiket: {true_label}")
        print(f"   Tahmin: {predicted_label} ({confidence:.2f}%)")
        
        for class_idx, prob in enumerate(probs):
            class_name = reverse_label_map[class_idx]
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"   {class_name:10s}: {bar} {prob*100:.2f}%")
        
        print(f"   {status}")
        time.sleep(0.5)
    
    accuracy = (correct / num_predictions) * 100
    print("\n" + "="*70)
    print(f"Test Accuracy: {correct}/{num_predictions} = {accuracy:.2f}%")
    print("="*70)

def realtime_csv_prediction():
    print("\n" + "="*70)
    print("GERCEK ZAMANLI CSV TAHMINI")
    print("="*70)
    print("\nCSV dosyasi bekleniyor...")
    print("Ornekler:")
    print("   - Dosya yolu: /home/kadir/sanal-makine/python/proje/test_data.csv")
    print("   - Veya 'q' yazarak cikis")
    
    model, reverse_label_map = load_model_and_labels()
    
    while True:
        csv_path = input("\nCSV dosya yolu (veya 'q' cikis): ").strip()
        
        if csv_path.lower() == 'q':
            print("Cikiliyor...")
            break
        
        if not os.path.exists(csv_path):
            print(f"❌ Dosya bulunamadi: {csv_path}")
            continue
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            if len(df) < WINDOW_SIZE:
                print(f"❌ Yetersiz veri! En az {WINDOW_SIZE} satir gerekli, bulundu: {len(df)}")
                continue
            
            print(f"\n✅ CSV yuklendi: {len(df)} satir")
            
            window_data = df[EEG_FEATURES].values[:WINDOW_SIZE]
            window_data = np.nan_to_num(window_data, nan=0.0)
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            window_data = scaler.fit_transform(window_data.reshape(-1, len(EEG_FEATURES))).reshape(WINDOW_SIZE, -1)
            
            predicted_label, confidence, probs = predict_single_window(model, window_data, reverse_label_map)
            
            print("\n" + "-"*70)
            print(f"TAHMIN SONUCU:")
            print(f"   Sinif: {predicted_label}")
            print(f"   Guven: {confidence:.2f}%")
            print("\nTum siniflar:")
            for class_idx, prob in enumerate(probs):
                class_name = reverse_label_map[class_idx]
                bar_length = int(prob * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"   {class_name:10s}: {bar} {prob*100:.2f}%")
            print("-"*70)
            
        except Exception as e:
            print(f"❌ Hata: {e}")

def main():
    print("\n" + "="*70)
    print("EEG SINIFLANDIRMA - GERCEK ZAMANLI TAHMIN")
    print("="*70)
    
    model, reverse_label_map = load_model_and_labels()
    
    while True:
        print("\nMod Secin:")
        print("   1. Simulasyon (test verisinden tahmin)")
        print("   2. CSV dosyasindan tahmin")
        print("   3. Cikis")
        
        choice = input("\nSeciminiz (1/2/3): ").strip()
        
        if choice == '1':
            num = input("Kac ornek test edilsin? (varsayilan: 10): ").strip()
            num = int(num) if num.isdigit() else 10
            simulate_realtime_prediction(model, reverse_label_map, num)
        
        elif choice == '2':
            realtime_csv_prediction()
        
        elif choice == '3':
            print("\nCikiliyor...")
            break
        
        else:
            print("❌ Gecersiz secim!")

if __name__ == "__main__":
    main()
