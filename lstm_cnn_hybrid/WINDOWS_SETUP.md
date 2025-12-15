# Windows Kurulum Rehberi - LSTM+CNN Model

## 📁 Gerekli Dosyalar (C:\Users\abdul\Desktop\code\python\biyo_proje\lstm+cnn)

### ✅ MUTLAKA GEREKLI (Canlı Tahmin İçin):
```
signal_processor.py      (11K)  - Raw EEG → FFT hesaplama
realtime_predict.py      (17K)  - Terminal tabanlı canlı tahmin (GUI YOK)
best_model.pth          (736K)  - Eğitilmiş model (%99.21 accuracy)
scaler.pkl              (810B)  - Normalizasyon parametreleri
config.json             (219B)  - Model konfigürasyonu
label_map.json           (55B)  - Sınıf etiketleri (yukarı/asagı/araba)
README.md               (3.1K)  - Açıklamalar
```

### 🎨 İSTEĞE BAĞLI (GUI İsterseniz):
```
realtime_gui.py          (23K)  - Tkinter GUI (grafik gösterim)
```

### 🔧 EĞITIM İÇIN (Yeniden Eğitmek İsterseniz):
```
data_preprocess.py       (8.4K)  - Veri hazırlama
train_model.py           (16K)   - Model eğitimi
```

### ❌ KOPYALANMAMALI (Çok Büyük / Gereksiz):
```
X_data.npy              (301M)  - Eğitim verisi (sadece yeniden eğitmek için)
y_data.npy              (642K)  - Etiketler (sadece yeniden eğitmek için)
final_model.pth         (736K)  - Son model (best_model kullanılmalı)
__pycache__/                    - Python cache
```

---

## 🚀 Windows'ta Kullanım

### 1. Python Kurulumu
```bash
# Python 3.8+ gerekli
python --version
```

### 2. Gerekli Kütüphaneleri Kur
```bash
cd C:\Users\abdul\Desktop\code\python\biyo_proje\lstm+cnn
pip install torch numpy scipy pyserial
```

### 3. MindWave'i Bağla
- MindWave USB dongle'ı tak
- Cihaz Yöneticisi'nden COM port'u kontrol et (örn: COM5)

### 4. Canlı Tahmin (Terminal - GUI YOK)
```bash
python realtime_predict.py --port COM5 --threshold 0.5
```

**Çıktı:**
```
🎯 Tahmin: YUKARI     | Güven: 85.3%
🎯 Tahmin: ASAGI      | Güven: 92.1%
🎯 Tahmin: ARABA      | Güven: 78.5%
```

### 5. GUI ile Canlı Tahmin (İsteğe Bağlı)
```bash
python realtime_gui.py --port COM5
```

### 6. Simülasyon Modu (Cihaz Olmadan Test)
```bash
python realtime_predict.py --simulation
# veya
python realtime_gui.py --simulation
```

---

## ⚙️ Parametreler

### realtime_predict.py Argümanları:
```
--port COM5          : MindWave COM port (varsayılan: COM5)
--threshold 0.5      : Minimum güven skoru (0-1 arası)
--simulation         : Simülasyon modu (cihaz olmadan test)
```

### Stride Ayarı:
`signal_processor.py` içinde:
```python
DEFAULT_STRIDE = 64  # 512Hz / 64 = ~8 FFT/saniye (~125ms)
```

---

## 📊 Sistem Özellikleri

**Pipeline:**
```
MindWave (512 Hz Raw EEG)
    ↓
signal_processor.py
  - DC offset removal
  - Artifact rejection (>500 µV)
  - Notch filter (50 Hz)
  - Bandpass filter (0.5-50 Hz)
  - FFT → 8 bant gücü
    ↓
Feature Engineering
  - 8 FFT bant + 7 türetilmiş özellik = 15 özellik
    ↓
Model (SimpleCNN_LSTM)
  - CNN (feature extraction)
  - Bidirectional LSTM (temporal)
  - FC layers (classification)
    ↓
Tahmin: yukarı / asagı / araba
```

**Model Performansı:**
- Validation Accuracy: 99.21%
- Train Accuracy: 93.51%
- Sequence Length: 64 frames
- FFT Rate: ~8/saniye (~125ms interval)

---

## 🔍 Sorun Giderme

### "Module not found" Hatası:
```bash
pip install torch numpy scipy pyserial
```

### COM Port Bulunamıyor:
```bash
# Cihaz Yöneticisi → Bağlantı Noktaları → COM5 kontrolü
python realtime_predict.py --port COM7  # Farklı port dene
```

### Sinyal Zayıf:
- MindWave başlığını düzgün tak
- Elektrotların temiz olduğundan emin ol
- Başlığı ıslatabilirsin (daha iyi iletkenlik)

### Model Yüklenemiyor:
- `best_model.pth` dosyasının aynı klasörde olduğundan emin ol
- Dosyanın bozuk olmadığını kontrol et

---

## 📝 Notlar

- **GUI kullanmak zorunda değilsiniz**: `realtime_predict.py` terminal'de çalışır
- **Stride mekanizması**: CPU kullanımını optimize eder, her sample'da değil her 64 sample'da FFT hesaplar
- **Real-time**: ~125ms aralıklarla tahmin yapar (LSTM'in beklediği temporal resolution)
- **Eğitim verileri**: Windows'a kopyalamaya gerek yok (sadece tahmin için)

---

## 🎯 Hızlı Başlangıç

**Minimum kurulum (sadece tahmin için):**
```bash
# 1. Dosyaları kopyala (7 dosya, toplam ~1.5 MB)
signal_processor.py
realtime_predict.py
best_model.pth
scaler.pkl
config.json
label_map.json
README.md

# 2. Kütüphaneleri kur
pip install torch numpy scipy pyserial

# 3. Çalıştır
python realtime_predict.py --port COM5
```

---

## 📞 Yardım

Sorun yaşarsan hata mesajını gönder!
