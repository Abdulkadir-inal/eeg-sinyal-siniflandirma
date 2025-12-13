# Log Transform + Oran Formülleri Tekniği

## 📊 Sonuçlar

| Metrik | FFT Modeli | Transform Modeli |
|--------|------------|------------------|
| Özellik Sayısı | 8 FFT bantı | 17 (8 log + 8 oran + Electrode) |
| Veri Kaynağı | Raw EEG 512Hz → FFT | FFT bantları → Transform |
| Doğrulama Accuracy | ~96-98% | **99.70%** |
| İyileşme | - | **+~2-4%** |

## 🔗 Veri Akışı

```
../fft_model/data/ veya ../fft_model/data_filtered/
    ↓
FFT ile hesaplanmış 8 bant gücü
    ↓
Log Transform + Oran Formülleri
    ↓
17 özellik (8 log + 8 oran + Electrode)
    ↓
TCN Model → %99.70 Doğruluk
```

## 🔧 Uygulanan Transformasyonlar

### 1. Log Transform
```python
log1p(x) = log(1 + x)
```
- Büyük değerlerdeki küçük farkları vurgular
- Negatif değerler için: `sign(x) * log1p(|x|)`

### 2. Oran Formülleri (8 yeni özellik)
```python
Delta_Theta   = Delta / Theta
Theta_Alpha   = Theta / Alpha
Alpha_Beta    = Alpha / Beta
Beta_Gamma    = Beta / Gamma
Slow_Fast     = (Theta + Alpha) / (Beta + Gamma)
Delta_Alpha   = Delta / Alpha
VeryLow_High  = (Delta + Theta) / (Alpha + Beta + Gamma)
Mid_Low       = (Alpha + Beta) / (Delta + Theta)
```

## 📁 Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `data_preprocess_transformed.py` | FFT verilerini yükleyip transform uygular |
| `train_model_transformed.py` | TCN model eğitim scripti |
| `realtime_transformed.py` | Gerçek zamanlı tahmin (FFT + Transform) |
| `X_transformed.npy` | İşlenmiş özellik matrisi (N, 128, 17) |
| `y_transformed.npy` | Etiketler (N,) |
| `label_map_transformed.json` | Sınıf etiketleri |
| `scaler_transformed.pkl` | StandardScaler (pickle) |
| `best_model_transformed.pth` | En iyi model ağırlıkları |
| `final_model_transformed.pth` | Son epoch model ağırlıkları |
| `training_history_transformed.png` | Eğitim grafiği |

## ⚠️ Önemli Notlar

- **Veri Kaynağı**: Bu sistem `../fft_model/data/` veya `../fft_model/data_filtered/` klasöründeki FFT hesaplanmış CSV dosyalarını kullanır
- **Eski Veriler**: Önceki NeuroSky ham verileri `../archive_old_neurosky/` klasöründe arşivlendi
- **FFT Bağımlılığı**: FFT hesaplaması `fft_model/` klasöründe yapılıyor, bu klasör ona transformasyon ekliyor

## 🚀 Kullanım

### 1. FFT Verilerini Hazırla (önce)
```bash
cd ../fft_model
python3 convert_raw_to_fft_filtered.py  # veya convert_raw_to_fft.py
```

### 2. Transform Veri İşleme
```bash
cd ../log_ratio_transform
python3 data_preprocess_transformed.py
```

### 3. Model Eğitimi
```bash
python3 train_model_transformed.py
```

### 4. Gerçek Zamanlı Tahmin
```bash
python3 realtime_transformed.py
```

## ⚡ Performans Yükü

- Log Transform: **0.003 ms** (%0.3 sistem yükü)
- Oran Formülleri: **0.002 ms** (%0.2 sistem yükü)
- **Toplam: ~0.005 ms** (pratik 0)

Canlı sistemde 2-4 Hz tahmin hızı korunur.

## 📅 Tarih
9 Aralık 2025
