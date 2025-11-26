# Model Denemeleri / Model Experiments

Bu klasör, EEG beyin dalgası sınıflandırması için farklı deep learning modellerinin test edildiği deneysel çalışma alanıdır.

## 📁 Klasör Yapısı

```
model_experiments/
├── README.md              # Bu dosya
├── TCN/                   # Temporal Convolutional Network
│   ├── tcn_model.py
│   ├── tcn_best_model.pth
│   ├── tcn_final_model.pth
│   ├── tcn_training_history.png
│   ├── tcn_confusion_matrix.png
│   ├── tcn_training_log.txt
│   └── README.md
│
└── EGGnet/                # EEGNet (EEG için özel CNN)
    ├── eegnet_model.py
    ├── eegnet_best_model.pth
    ├── eegnet_final_model.pth
    ├── eegnet_training_history.png
    ├── eegnet_confusion_matrix.png
    ├── eegnet_training_log.txt
    └── README.md
```

## 🎯 Proje Hedefi

MindWave EEG cihazından alınan beyin dalgası sinyallerini kullanarak 3 farklı düşünceyi sınıflandırmak:
- **Araba** (araç düşüncesi)
- **Yukarı** (yukarı hareket)
- **Aşağı** (aşağı hareket)

## 📊 Model Karşılaştırması

| Model | Test Accuracy | Parametreler | Eğitim Süresi | Durum |
|-------|---------------|--------------|---------------|-------|
| **TCN** ⭐ | **89.41%** | 460,611 | ~3 dk | ✅ Production Ready |
| **EEGNet** | 50.58% | 1,443 | ~1 dk | ⚠️ Veri uyumsuzluğu |
| **CNN+LSTM** | ? | ? | ? | 📋 Ana proje klasöründe |

## 🏆 En İyi Model: TCN

**Neden TCN?**
- ✅ En yüksek accuracy (%89.41)
- ✅ Dengeli sınıf performansı (tüm sınıflar ~%90)
- ✅ Hızlı eğitim
- ✅ Stabil öğrenme
- ✅ Overfitting yok

## 🚀 Yeni Model Ekleme

Her yeni model için:

1. **Klasör oluştur:** `model_experiments/MODEL_ADI/`
2. **Model dosyası:** `model_adi_model.py`
3. **README ekle:** Model özellikleri ve kullanımı
4. **Standart çıktılar:**
   - `{model}_best_model.pth`
   - `{model}_final_model.pth`
   - `{model}_training_history.png`
   - `{model}_confusion_matrix.png`
   - `{model}_training_log.txt`

### ✅ Model Template Özellikleri:

```python
# Zorunlu bileşenler:
1. SAVE_DIR = "model_experiments/MODEL_ADI/"
2. mini_prediction_test() fonksiyonu (10 örneklik test)
3. Aynı random seed (42) kullan
4. Train/Val/Test split: 70/10/20
5. Progress bar ile epoch takibi
6. Best model kaydetme
```

## 📝 Veri Seti

**Kaynak:** `/home/kadir/sanal-makine/python/proje/`
- `X.npy` - (20114, 128, 9) EEG features
- `y.npy` - (20114,) Labels
- `label_map.json` - Sınıf isimleri

**Features:**
1. Delta (0.5-3 Hz)
2. Theta (4-7 Hz)
3. Low Alpha (8-9 Hz)
4. High Alpha (10-12 Hz)
5. Low Beta (13-17 Hz)
6. High Beta (18-30 Hz)
7. Low Gamma (31-40 Hz)
8. Mid Gamma (41-50 Hz)
9. Attention & Meditation metrikleri

## 🔬 Gelecek Denemeler

- [ ] **Transformer** - Attention mechanism (veri augmentation gerekli)
- [ ] **GRU** - Daha hafif RNN alternatifi
- [ ] **CNN+Attention** - Hybrid model
- [ ] **Ensemble** - TCN + CNN+LSTM kombinasyonu
- [ ] **ResNet-1D** - Residual connections
- [ ] **Bidirectional LSTM** - İki yönlü temporal analiz

## 📖 Kullanım Kılavuzu

### Model Eğitimi:

```bash
# TCN modelini çalıştır
cd model_experiments/TCN
python3 tcn_model.py

# EEGNet modelini çalıştır
cd model_experiments/EGGnet
python3 eegnet_model.py
```

### 🎯 Mini Test (Eğitilmiş Modeli Test Et):

Eğitim yapmadan, sadece eğitilmiş modeli test etmek için:

```bash
# TCN modelini test et (10 rastgele örnek)
cd model_experiments
python3 mini_test.py TCN

# EEGNet modelini test et
python3 mini_test.py EGGnet

# Herhangi bir modeli test et
python3 mini_test.py MODEL_KLASORU
```

**mini_test.py özellikleri:**
- ✅ Eğitim yapmaz, sadece inference
- ✅ 10 rastgele örnek üzerinde hızlı test
- ✅ Tüm test seti üzerinde detaylı değerlendirme
- ✅ Sınıf bazlı performans analizi
- ✅ Renkli çıktı (✓ yeşil, ✗ kırmızı)
- ✅ Confidence skorları

### Model Yükleme (Inference):

```python
import torch
from tcn_model import TCN_EEG_Model

# Model oluştur
model = TCN_EEG_Model(input_channels=9, num_classes=3)

# En iyi modeli yükle
model.load_state_dict(torch.load('TCN/tcn_best_model.pth'))
model.eval()

# Tahmin yap
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
```

## 🎯 Mini Tahmin Testi

Eğitim sırasında otomatik olarak 10 örnek test edilir.

**Eğitilmiş modeli ayrıca test etmek için:**
```bash
python3 mini_test.py MODEL_KLASORU
```

Örnek çıktı:
```
🎯 MİNİ TAHMİN TESTİ (10 Örnek)
======================================================================

No    Gerçek          Tahmin          Sonuç     
--------------------------------------------------
1     yukarı          yukarı          ✓ DOĞRU (92.3%)
2     araba           araba           ✓ DOĞRU (88.5%)
3     aşağı           aşağı           ✓ DOĞRU (91.7%)
...
--------------------------------------------------
Mini Test Accuracy: 9/10 (90%)
```

## 📈 Performans İyileştirme İpuçları

1. **Data Augmentation**
   - Time warping
   - Gaussian noise ekleme
   - Amplitude scaling

2. **Hyperparameter Tuning**
   - Learning rate scheduling
   - Batch size optimizasyonu
   - Dropout oranı ayarlama

3. **Ensemble Methods**
   - Birden fazla modeli birleştir
   - Voting veya averaging kullan

4. **Transfer Learning**
   - Benzer EEG datasetlerinden pre-training

## 🔗 Bağlantılar

- Ana Proje: `/home/kadir/sanal-makine/python/proje/`
- Veri Seti: `/home/kadir/sanal-makine/python/proje-veri/`
- Real-time Tahmin: `/home/kadir/sanal-makine/python/proje/realtime_mindwave_predict.py`

## 📞 Not

Her model için ayrı README dosyası bulunmaktadır. Detaylı bilgi için ilgili model klasörüne bakınız.
