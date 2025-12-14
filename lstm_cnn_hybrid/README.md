# LSTM+CNN Hibrit Model

MindWave EEG verileri için hibrit derin öğrenme modeli.

## Mimari

```
FFT Bantları (8) → Türetilmiş Özellikler (15) → CNN → LSTM → Attention → Tahmin
```

### Özellikler:
- **8 FFT Bandı**: Delta, Theta, Low Alpha, High Alpha, Low Beta, High Beta, Low Gamma, Mid Gamma
- **7 Türetilmiş Özellik**: Alpha Total, Beta Total, Gamma Total, Theta/Beta Ratio, Alpha/Beta Ratio, Theta/Alpha Ratio, Engagement Index

### Overfitting Önlemleri:
- Dropout (0.4)
- L2 Regularization (weight_decay=1e-4)
- Early Stopping (patience=15)
- Data Augmentation (4x veri)
- Batch Normalization
- Gradient Clipping

## Kullanım

### 1. Veri Ön İşleme
```bash
python data_preprocess.py
```

Çıktılar:
- `X_data.npy` - Özellik matrisi
- `y_data.npy` - Etiketler
- `scaler.pkl` - Normalizasyon parametreleri
- `config.json` - Model konfigürasyonu
- `label_map.json` - Sınıf isimleri

### 2. Model Eğitimi
```bash
python train_model.py
```

Çıktılar:
- `best_model.pth` - En iyi model
- `final_model.pth` - Son model
- `training_history.png` - Eğitim grafiği

### 3. Canlı Tahmin (Terminal)
```bash
# Windows
python realtime_predict.py --port COM5

# Linux
python realtime_predict.py --port /dev/ttyUSB0
```

### 4. Canlı Tahmin (GUI)
```bash
# Simülasyon modu (test için)
python realtime_gui.py --simulation

# Gerçek cihaz
python realtime_gui.py --port COM5
```

## Model Parametreleri

| Parametre | Değer |
|-----------|-------|
| Sequence Length | 64 frame (~0.5 sn) |
| CNN Channels | 32 |
| LSTM Hidden | 64 |
| LSTM Layers | 1 |
| Bidirectional | Evet |
| Dropout | 0.4 |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Early Stop Patience | 15 |

## Dosya Yapısı

```
lstm_cnn_hybrid/
├── data_preprocess.py    # Veri ön işleme
├── train_model.py        # Model eğitimi
├── realtime_predict.py   # Terminal tahmin
├── realtime_gui.py       # GUI tahmin
├── README.md             # Bu dosya
├── X_data.npy            # (oluşturulacak)
├── y_data.npy            # (oluşturulacak)
├── scaler.pkl            # (oluşturulacak)
├── config.json           # (oluşturulacak)
├── label_map.json        # (oluşturulacak)
├── best_model.pth        # (oluşturulacak)
├── final_model.pth       # (oluşturulacak)
└── training_history.png  # (oluşturulacak)
```

## TCN vs LSTM+CNN Karşılaştırması

| Özellik | TCN | LSTM+CNN |
|---------|-----|----------|
| Temporal Receptive Field | Sabit (dilation ile) | Dinamik (LSTM) |
| Paralel İşlem | Evet | Kısmen |
| Uzun Vade Bağımlılık | Dilation gerekli | Doğal |
| Eğitim Hızı | Hızlı | Orta |
| Overfitting Riski | Düşük | Orta |
| Interpretability | Düşük | Orta |

## Troubleshooting

### Model yüklenmiyor
```
Önce data_preprocess.py ve train_model.py çalıştırın.
```

### MindWave bağlanmıyor
```
- USB kablo kontrolü
- COM port numarasını kontrol et (Device Manager)
- Driver kurulumunu kontrol et
```

### Overfitting var
```
- Dropout'u artır (0.5)
- Daha fazla veri topla
- Model karmaşıklığını azalt
```
