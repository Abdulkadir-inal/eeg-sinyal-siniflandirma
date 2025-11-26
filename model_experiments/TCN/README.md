# TCN (Temporal Convolutional Network)

Zaman serisi verileri için optimize edilmiş, dilated causal convolution kullanan modern mimari.

## 📋 Model Özellikleri

**Mimari:**
- Dilated Causal Convolution
- Residual Connections
- Exponential Dilation (1, 2, 4, 8, ...)
- Global Average Pooling
- Paralel işlem desteği

**Avantajlar:**
- LSTM'den daha hızlı
- Uzun vadeli bağımlılıkları yakalayabilir
- Gradient problemi yok
- Eğitimi paralelize edilebilir

## 🚀 Kullanım

```bash
cd /home/kadir/sanal-makine/python/proje/model_experiments/TCN
python3 tcn_model.py
```

## ⚙️ Hiperparametreler

- **Batch Size:** 32
- **Epochs:** 50
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **TCN Channels:** [64, 128, 256]
- **Kernel Size:** 3
- **Dropout:** 0.2

## 📊 Model Yapısı

```
Input (batch, 128, 9)
    ↓
TCN Block 1 (9→64, dilation=1)
    ↓
TCN Block 2 (64→128, dilation=2)
    ↓
TCN Block 3 (128→256, dilation=4)
    ↓
Global Average Pooling
    ↓
FC (256→128)
    ↓
Dropout (0.5)
    ↓
FC (128→3)
    ↓
Output (3 classes)
```

## 📈 Performans Metrikleri

| Metrik | Değer |
|--------|-------|
| Toplam Parametreler | 460,611 |
| Test Accuracy | **~89.41%** ⭐ |
| Eğitim Süresi | ~2-3 dakika |
| Sınıf F1-Scores | araba: 88%, yukarı: 90%, aşağı: 90% |

## 📁 Çıktı Dosyaları

- `tcn_best_model.pth` - En iyi validation accuracy'ye sahip model
- `tcn_final_model.pth` - Son epoch'taki model
- `tcn_training_history.png` - Loss ve accuracy grafikleri
- `tcn_confusion_matrix.png` - Test seti confusion matrix
- `tcn_training_log.txt` - Detaylı eğitim raporu

## 🎯 Mini Tahmin Testi

Her eğitim sonunda 10 rastgele örnek üzerinde gerçek zamanlı tahmin testi yapılır:
- Gerçek label vs Tahmin edilen label
- Confidence skorları (%)
- Doğru/Yanlış işaretleri

## 💡 Neden TCN?

**EEG Sinyalleri için ideal çünkü:**
1. ✅ Temporal patterns'i çok iyi yakalar
2. ✅ Uzun sekansları işleyebilir
3. ✅ Hızlı eğitim ve inference
4. ✅ Az memory kullanımı
5. ✅ Stabil gradient flow

## 🔗 İlgili Modeller

- [EEGNet](../EGGnet/) - EEG için özel CNN
- CNN+LSTM - Ana proje klasöründe

## 📝 Notlar

- TCN bu proje için **en iyi performansı** gösterdi
- Tüm sınıflar için dengeli sonuçlar
- Production için önerilen model
