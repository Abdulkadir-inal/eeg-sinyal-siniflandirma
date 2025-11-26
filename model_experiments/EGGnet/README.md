# EEGNet Model

EEG sinyalleri için özel olarak tasarlanmış kompakt CNN mimarisi.

## 📋 Model Özellikleri

**Mimari:**
- Depthwise ve Separable Convolutions
- Temporal ve Spatial Filtering
- Batch Normalization ve Dropout
- Kompakt yapı (çok az parametre)

**Referans:**
Lawhern et al. (2018) - EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces

## 🚀 Kullanım

```bash
cd /home/kadir/sanal-makine/python/proje/model_experiments/EGGnet
python3 eegnet_model.py
```

## ⚙️ Hiperparametreler

- **Batch Size:** 32
- **Epochs:** 50
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **F1 (Temporal Filters):** 8
- **F2 (Separable Filters):** 16
- **D (Depthwise Multiplier):** 2
- **Dropout Rate:** 0.5

## 📊 Model Yapısı

```
Input (batch, 128, 9)
    ↓
Temporal Conv (1×64)
    ↓
Depthwise Spatial Conv
    ↓
Average Pooling (1×4)
    ↓
Separable Conv (1×16)
    ↓
Average Pooling (1×8)
    ↓
Fully Connected
    ↓
Output (3 classes)
```

## 📈 Performans Metrikleri

| Metrik | Değer |
|--------|-------|
| Toplam Parametreler | 1,443 |
| Test Accuracy | ~50% |
| Eğitim Süresi | ~1-2 dakika |

## 📁 Çıktı Dosyaları

- `eegnet_best_model.pth` - En iyi validation accuracy'ye sahip model
- `eegnet_final_model.pth` - Son epoch'taki model
- `eegnet_training_history.png` - Loss ve accuracy grafikleri
- `eegnet_confusion_matrix.png` - Test seti confusion matrix
- `eegnet_training_log.txt` - Detaylı eğitim raporu

## 🎯 Mini Tahmin Testi

Her eğitim sonunda 10 rastgele örnek üzerinde gerçek zamanlı tahmin testi yapılır.

## 📝 Notlar

**Neden düşük performans?**
- EEGNet, raw EEG elektrodu sinyalleri için tasarlandı
- Bizim verilerimiz önceden işlenmiş feature'lar (Delta, Theta, Alpha, vb.)
- Spatial filtering bekleniyor ama elimizde EEG kanalları değil, feature'lar var

**Öneriler:**
- Raw EEG sinyalleri ile kullanılmalı
- TCN veya CNN+LSTM modelleri bu veri tipi için daha uygun

## 🔗 İlgili Modeller

- [TCN](../TCN/) - Temporal Convolutional Network (Önerilen)
- CNN+LSTM - Ana proje klasöründe
