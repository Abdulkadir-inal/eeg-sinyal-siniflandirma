# Transformer Model

Modern attention mechanism kullanan Transformer mimarisi. Az veri için optimize edilmiş küçük versiyonu.

## 📋 Model Özellikleri

**Mimari:**
- Self-Attention Mechanism
- Positional Encoding
- Multi-Head Attention (4 heads)
- Feedforward Neural Network
- Layer Normalization
- Residual Connections

**Optimizasyon (Az Veri İçin):**
- Küçük model boyutu (d_model=64)
- Az katman (2 encoder layer)
- Az attention head (4 head)
- Yüksek dropout (0.3)
- Gradient clipping

## 🚀 Kullanım

```bash
cd /home/kadir/sanal-makine/python/proje/model_experiments/Transformer
python3 transformer_model.py
```

## ⚙️ Hiperparametreler

- **Batch Size:** 32
- **Epochs:** 50
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **d_model:** 64 (embedding dimension)
- **nhead:** 4 (attention heads)
- **num_layers:** 2 (encoder layers)
- **dim_feedforward:** 256
- **Dropout:** 0.3

## 📊 Model Yapısı

```
Input (batch, 128, 9)
    ↓
Input Projection (9→64)
    ↓
Positional Encoding
    ↓
Transformer Encoder Layer 1
  ├─ Multi-Head Self-Attention (4 heads)
  ├─ Add & Norm
  ├─ Feedforward (64→256→64)
  └─ Add & Norm
    ↓
Transformer Encoder Layer 2
  └─ (same structure)
    ↓
Global Average Pooling
    ↓
FC (64→128)
    ↓
Dropout (0.3)
    ↓
FC (128→3)
    ↓
Output (3 classes)
```

## 📈 Beklenen Performans

| Metrik | Tahmini Değer |
|--------|---------------|
| Toplam Parametreler | ~50,000-100,000 |
| Test Accuracy | %60-80 (az veri) |
| Eğitim Süresi | ~3-5 dakika |

**Not:** Transformer modelleri genellikle daha fazla veri gerektirir (50k+ örnek). 14k örnekle sınırlı performans beklenir.

## ⚠️ Az Veri Problemi

**Neden düşük performans olabilir?**
- Transformer'lar veri açlığı çeker
- Self-attention çok fazla parametre öğrenir
- 14k örnek ideal değil (50k+ önerilir)

**Alınan önlemler:**
1. ✅ Küçük model boyutu (d_model=64)
2. ✅ Az katman (2 layer)
3. ✅ Yüksek dropout (0.3)
4. ✅ Gradient clipping
5. ✅ Learning rate scheduling

## 📁 Çıktı Dosyaları

- `transformer_best_model.pth` - En iyi validation accuracy
- `transformer_final_model.pth` - Son epoch modeli
- `transformer_training_history.png` - Loss ve accuracy grafikleri
- `transformer_confusion_matrix.png` - Test seti confusion matrix
- `transformer_training_log.txt` - Detaylı eğitim raporu

## 🎯 Mini Tahmin Testi

Eğitim sonunda 10 rastgele örnek üzerinde test yapılır.

**Ayrıca test etmek için:**
```bash
cd ..
python3 mini_test.py Transformer
```

## 💡 Ne Zaman Transformer Kullanılmalı?

**✅ Transformer iyidir eğer:**
- Çok fazla veri varsa (50k+ örnek)
- Uzun vadeli bağımlılıklar varsa
- Paralel işlem önemliyse
- SOTA performans gerekiyorsa

**❌ Transformer kötüdür eğer:**
- Az veri varsa (<20k örnek) ← Bizim durum
- Hızlı inference gerekiyorsa
- Küçük model isteniyorsa
- Basit pattern'ler yeterliyse

## 🔗 İlgili Modeller

- [TCN](../TCN/) - Önerilen! (%92.44 accuracy) ⭐
- [EEGNet](../EGGnet/) - EEG özel ama veri uyumsuz
- CNN+LSTM - Ana projede

## 📝 Notlar

- Bu implementasyon az veri için optimize edilmiştir
- Daha fazla veriyle (data augmentation) performans artabilir
- TCN bu veri miktarı için daha uygun
- Transformer'ın gücünü görmek için 50k+ örnek gerekir

## 🧪 Deneysel Sonuçlar

Eğitim tamamlandığında buraya eklenecek:
- Test Accuracy: ?
- Mini Test: ?/10
- Sınıf Performansları: ?
