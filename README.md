# 🧠 EEG Sinyal Sınıflandırma Projesi

Neurosky MindWave EEG cihazından alınan beyin dalgalarını CNN+LSTM hibrit deep learning modeli ile sınıflandırma.

## 📊 Proje Özeti

- **Veri Kaynağı:** Neurosky MindWave Mobile 2 EEG cihazı
- **Özellikler:** 9 kanal (Electrode + 8 EEG bandı)
- **Model:** CNN+LSTM Hibrit Mimari
- **Performans:** %98.20 doğruluk (validation)
- **GPU:** NVIDIA RTX 4050 (CUDA 11.8)

## 🗂️ Dosya Yapısı

```
proje/
├── data_preprocess.py           # Veri ön işleme
├── train_model.py               # Model eğitimi
├── predict.py                   # Offline tahmin (CSV/simulasyon)
├── realtime_mindwave_predict.py # 🔥 Canlı MindWave tahmini
├── start_realtime.sh            # 🚀 Hızlı başlatma scripti
├── X.npy (22M)                  # İşlenmiş özellikler
├── y.npy (20K)                  # Etiketler
├── label_map.json               # Sınıf haritası
├── best_model.pth (1.8M)        # En iyi model
├── final_model.pth (1.8M)       # Son model
├── training_history.png         # Eğitim grafikleri
├── sample_eeg_window.png        # Örnek veri görselleştirme
├── asagı.csv                    # Ham EEG verisi (sınıf 0)
└── yukarı.csv                   # Ham EEG verisi (sınıf 1)
```

## 🚀 Kullanım

### 1. Veri Ön İşleme

```bash
python3 data_preprocess.py
```

**Çıktılar:**
- `X.npy`: (2500, 128, 9) - Normalize edilmiş pencereler
- `y.npy`: (2500,) - Etiketler
- `label_map.json`: Sınıf haritası
- `sample_eeg_window.png`: Görselleştirme

**İşlemler:**
1. CSV dosyalarını oku
2. Event ID 33025-33024 arası segmentleri çıkar
3. 128 örneklik pencerelere böl (50% overlap)
4. StandardScaler ile normalize et

### 2. Model Eğitimi

```bash
python3 train_model.py
```

**Model Mimarisi:**
```
Conv1D(9→64) → BatchNorm → MaxPool
    ↓
Conv1D(64→128) → BatchNorm → MaxPool
    ↓
Conv1D(128→256) → BatchNorm
    ↓
LSTM(256→128, 2 layers)
    ↓
FC(128→64) → Dropout(0.5) → FC(64→2)
```

**Hiperparametreler:**
- Batch Size: 32
- Epochs: 50
- Learning Rate: 0.001
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau

**Sonuçlar:**
- Train Accuracy: 96.95%
- Validation Accuracy: 98.20%
- Toplam Parametreler: 465,218

### 3. Tahmin (Offline)

```bash
python3 predict.py
```

**Mod 1: Simulasyon**
```python
# Test verisinden rastgele örnekler seç ve tahmin et
1. Simulasyon seç
2. Örnek sayısını gir (varsayılan: 10)
```

**Mod 2: CSV Tahmini**
```python
# Yeni CSV dosyasından tahmin
2. CSV dosyasından tahmin
# Dosya yolunu gir (en az 128 satır gerekli)
```

### 4. 🔥 Canlı MindWave ile Gerçek Zamanlı Tahmin

**Hazırlık (Windows):**
```bash
# 1. MindWave cihazını bilgisayara bağlayın (Bluetooth/USB)
# 2. Proxy sunucusunu başlatın
cd python
python windows_proxy.py
```

**WSL2'de Çalıştırma:**
```bash
# Basit kullanım
./start_realtime.sh

# Ya da manuel
python3 realtime_mindwave_predict.py

# Özel ayarlar
python3 realtime_mindwave_predict.py --host 192.168.1.100 --interval 2.0 --min-quality 30
```

**Parametreler:**
- `--host`: Windows IP adresi (varsayılan: 10.255.255.254)
- `--port`: TCP port (varsayılan: 5555)
- `--interval`: Tahminler arası süre (saniye, varsayılan: 1.0)
- `--min-quality`: Minimum sinyal kalitesi (0-200, varsayılan: 50)

**Örnek Çıktı:**
```
============================================================
⏰ 14:23:45 | Tahmin #23
🎯 Sonuç: YUKARI (98.76%)
------------------------------------------------------------
   asagı     : ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.24%
👉 yukarı    : ████████████████████████████ 98.76%
============================================================

📈 İstatistikler:
   asagı: 10 (43.5%)
   yukarı: 13 (56.5%)
```

## 📈 EEG Özellikler

### 1. **Electrode** (Ham EEG)
   - Volt cinsinden ham beyin sinyali
   - En yüksek çözünürlük

### 2. **Delta** (0.5-4 Hz)
   - Derin uyku, meditasyon

### 3. **Theta** (4-8 Hz)
   - Hafif uyku, yaratıcılık

### 4. **Low Alpha** (8-10 Hz)
   - Rahat uyanıklık

### 5. **High Alpha** (10-12 Hz)
   - Zihinsel dinlenme

### 6. **Low Beta** (12-18 Hz)
   - Aktif düşünme

### 7. **High Beta** (18-30 Hz)
   - Yoğun konsantrasyon

### 8. **Low Gamma** (30-40 Hz)
   - Bilgi işleme

### 9. **Mid Gamma** (40-50 Hz)
   - Yüksek bilişsel aktivite

## 🔬 Teknik Detaylar

### Veri İşleme Pipeline
```
CSV → Event Segmentation → Windowing → Normalization → X.npy
                              ↓
                        (128, 9) pencereler
                         50% overlap
```

### Model Akışı
```
Input (128, 9)
    ↓
CNN (Temporal feature extraction)
    ↓
LSTM (Sequential dependencies)
    ↓
FC Layers (Classification)
    ↓
Softmax (Probability distribution)
```

## 📊 Sonuçlar

### Test Performansı (5 örnek)
```
Test Accuracy: 5/5 = 100.00%
Ortalama Güven: >99%
```

### Eğitim Metrikleri
- En düşük validation loss: 0.0324
- En yüksek validation accuracy: 98.20%
- Overfitting yok (dropout ve regularization sayesinde)

## 🛠️ Gereksinimler

```bash
# Python paketleri
pandas==2.3.3
numpy==2.2.6
scikit-learn==1.7.2
torch==2.7.1+cu118
matplotlib==3.10.6

# Sistem
- CUDA 11.8+ (GPU için)
- Python 3.10+
```

## 🎯 Gelecek İyileştirmeler

1. ✅ **Canlı MindWave Entegrasyonu** → TAMAMLANDI!
   - WSL2 → Windows proxy → MindWave cihazı
   - Gerçek zamanlı sliding window
   - Her saniye tahmin

2. **Daha Fazla Sınıf**
   - Sol/sağ el hareketi
   - Farklı zihinsel görevler
   - Dikkat/meditasyon seviyeleri

3. **Model Optimizasyonu**
   - Attention mechanism
   - Transformer architecture
   - Model pruning/quantization

4. **Deployment**
   - REST API (Flask/FastAPI)
   - Web dashboard (real-time chart)
   - Mobil uygulama

5. **Veri Toplama Araçları**
   - Otomatik etiketleme GUI
   - Event marker ekleme
   - Dataset genişletme

## 🔧 Troubleshooting

### MindWave Bağlantı Sorunları

**Problem:** "Bağlantı reddedildi"
```bash
# Çözüm:
1. Windows'ta proxy çalışıyor mu kontrol et
2. Firewall ayarlarını kontrol et
3. IP adresini doğrula: ipconfig (Windows)
```

**Problem:** "Zayıf sinyal"
```bash
# Çözüm:
1. Elektrotları ıslatın (hafifçe)
2. Cihazı doğru takın (alın ortası)
3. --min-quality değerini artırın (örn: 100)
```

**Problem:** Buffer dolmuyor
```bash
# Çözüm:
1. MindWave cihazının pil seviyesini kontrol edin
2. Bluetooth/USB bağlantısını yeniden başlatın
3. Proxy sunucusunu yeniden başlatın
```

### Model Sorunları

**Problem:** Düşük doğruluk
```bash
# Çözüm:
1. Daha fazla veri toplayın
2. Pencere boyutunu ayarlayın (WINDOW_SIZE)
3. Model hiperparametrelerini tune edin
```

**Problem:** CUDA/GPU hatası
```bash
# Çözüm:
1. CUDA kurulumunu kontrol edin: nvidia-smi
2. PyTorch CUDA versiyonunu kontrol edin
3. CPU'da çalıştırın (otomatik fallback)

## � Lisans

Bu proje **MIT License** altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

### MIT License ile yapabilecekleriniz:

- ✅ **Ticari kullanım** - Ticari projelerde kullanabilirsiniz
- ✅ **Değiştirme** - Kodu değiştirebilirsiniz
- ✅ **Dağıtım** - Paylaşabilirsiniz
- ✅ **Özel kullanım** - Kendi projelerinizde kullanabilirsiniz
- ✅ **Patent kullanımı** - Patent haklarını kullanabilirsiniz

**Tek şart:** Lisans ve telif hakkı bildirimini koruyun.

```
Copyright (c) 2025 Kadir
MIT License
```

## 👤 İletişim

Kadir - EEG Beyin-Bilgisayar Arayüzü Projesi

---

**Oluşturulma Tarihi:** 17 Ekim 2025
**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU
**Framework:** PyTorch 2.7.1
