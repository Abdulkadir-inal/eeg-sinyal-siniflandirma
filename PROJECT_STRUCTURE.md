# EEG Sinyal Sınıflandırma - Proje Yapısı

## 📁 Aktif Klasörler

### 1. `fft_model/` ⭐ Ana Sistem
Raw EEG 512Hz + FFT hesaplaması ile çalışan ana sistem.

**Özellikler:**
- Raw EEG 512Hz alınıyor
- Notch filter (50Hz) + Bandpass filter (0.5-50Hz)
- FFT ile 8 bant gücü hesaplanıyor
- TCN, Transformer, CNN-LSTM modelleri
- Windows realtime tahmin desteği

**Önemli Dosyalar:**
- `windows_realtime_fft.py` - Windows realtime tahmin (Ana)
- `train_model_fft.py` - Model eğitimi
- `data_preprocess_fft.py` - Veri ön işleme
- `X_fft.npy`, `y_fft.npy` - İşlenmiş veri
- `best_model_fft.pth` - En iyi TCN modeli

**Kullanım:**
```bash
cd fft_model
python3 windows_realtime_fft.py
```

### 2. `log_ratio_transform/` 🚀 Gelişmiş Sistem
FFT + Log Transform + Oran Formülleri ile %99.70 doğruluk.

**Özellikler:**
- FFT'den 9 bant gücü alınıyor
- Log Transform uygulanıyor
- 8 Oran Formülleri hesaplanıyor (Delta/Theta, vb.)
- 9 → 17 özellik genişletmesi
- %99.70 doğruluk (FFT'den +4.00%)

**Önemli Dosyalar:**
- `realtime_transformed.py` - Realtime tahmin
- `train_model_transformed.py` - Model eğitimi
- `data_preprocess_transformed.py` - Transform ön işleme
- `X_transformed.npy`, `y_transformed.npy` - Transform veri
- `best_model_transformed.pth` - En iyi model

**Kullanım:**
```bash
cd log_ratio_transform
python3 realtime_transformed.py
```

### 3. `model_experiments/` 🧪 Deneysel Modeller
Farklı model mimarilerinin test edildiği klasör.

**İçerik:**
- `TCN/` - Temporal Convolutional Network denemeleri
- `Transformer/` - Transformer mimarisi denemeleri
- `CNN_LSTM/` - CNN-LSTM hibrit modeller
- `EGGnet/` - EEGNet mimarisi
- `_template_model.py` - Yeni model şablonu

### 4. `archive_old_neurosky/` 📦 Arşiv
Eski NeuroSky ham veri sistemi dosyaları (KULLANILMIYOR).

## 📄 Kök Dizindeki Dosyalar

- `README.md` - Ana proje dokümantasyonu
- `WINDOWS_REALTIME_README.md` - Windows realtime kurulum
- `FFT_BAND_ANALIZ_SONUCLARI.md` - FFT bant analiz sonuçları
- `LICENSE` - MIT Lisansı
- `.gitignore` - Git ignore kuralları

## 🎯 Hangi Sistemi Kullanmalıyım?

| Durum | Önerilen Sistem |
|-------|-----------------|
| Genel kullanım | `fft_model/` |
| En yüksek doğruluk | `log_ratio_transform/` |
| Model geliştirme | `model_experiments/` |
| Eski veri erişimi | `archive_old_neurosky/` |

## 🔄 Veri Akışı

### fft_model/
```
MindWave (512Hz) 
  → Raw EEG Buffer 
  → Notch + Bandpass Filter 
  → FFT (8 bant) 
  → Model 
  → Tahmin
```

### log_ratio_transform/
```
MindWave (512Hz) 
  → Raw EEG Buffer 
  → Notch + Bandpass Filter 
  → FFT (8 bant) 
  → Log Transform + Oran Formülleri (17 özellik)
  → Model 
  → Tahmin
```

## 📊 Performans Karşılaştırması

| Sistem | Doğruluk | Özellik Sayısı | Hız |
|--------|----------|----------------|-----|
| **archive_old_neurosky** | 95.70% | 9 | ~0ms |
| **fft_model** | ~96-98% | 8 | ~2-4Hz |
| **log_ratio_transform** | 99.70% | 17 | ~2-4Hz |

## 🚀 Hızlı Başlangıç

### FFT Model (Önerilen)
```bash
cd /home/kadir/sanal-makine/python/proje/fft_model
python3 windows_realtime_fft.py
```

### Log Ratio Transform (En İyi Doğruluk)
```bash
cd /home/kadir/sanal-makine/python/proje/log_ratio_transform
python3 realtime_transformed.py
```

## 📝 Notlar

- **Yapay Zeka Kullanımı**: Artık sadece aktif klasörler (`fft_model/`, `log_ratio_transform/`) kullanılacak
- **Eski Sistem**: `archive_old_neurosky/` yalnızca referans amaçlı saklanıyor
- **Model Geliştirme**: Yeni modeller `model_experiments/` içinde test edilmeli
- **Veri Kaynağı**: Her iki aktif sistem de `../proje-veri/` klasöründeki ham CSV'leri kullanıyor

---

**Son Güncelleme**: 9 Aralık 2025  
**Aktif Sistemler**: `fft_model/` (Ana), `log_ratio_transform/` (Gelişmiş)
