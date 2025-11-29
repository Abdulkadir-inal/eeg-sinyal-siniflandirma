# 🧠 EEG Sinyal Sınıflandırma Projesi

MindWave Mobile 2 EEG cihazından alınan beyin dalgalarını deep learning modelleri ile sınıflandırma.

> **Son Güncelleme:** 29 Kasım 2025 - Canlı tahmin sistemi ve WSL2/CUDA desteği eklendi.

## 📊 Proje Özeti

| Özellik | Değer |
|---------|-------|
| **Cihaz** | NeuroSky MindWave Mobile 2 |
| **Sınıflar** | araba, yukarı, aşağı (3 sınıf) |
| **En İyi Model** | TCN (%92.44 accuracy) |
| **GPU** | NVIDIA GeForce RTX 5070 (12GB VRAM) |
| **CUDA** | 12.8 |
| **Framework** | PyTorch 2.9.1 |
| **Python** | 3.10.12 |
| **OS** | Ubuntu 22.04.5 LTS (WSL2) |

## 🗂️ Proje Yapısı

```
proje/
├── 📁 Veri İşleme
│   ├── data_preprocess.py        # CSV → NumPy dönüşümü
│   └── X.npy, y.npy              # İşlenmiş veri
│
├── 📁 Model Eğitimi
│   ├── train_model.py            # Model eğitim scripti
│   └── model_experiments/        # Eğitilmiş modeller
│       ├── TCN/                  # %92.44 (En iyi)
│       ├── Transformer/          # %87.99
│       └── CNN_LSTM/             # %84.86
│
├── 📁 Canlı Tahmin (Windows)
│   ├── windows_realtime_predict.py  # ThinkGear Connector ile tahmin
│   └── WINDOWS_REALTIME_README.md   # Windows kullanım kılavuzu
│
├── 📁 Canlı Tahmin (WSL2 + CUDA)
│   ├── thinkgear_proxy.py        # Windows → WSL2 proxy
│   └── wsl_realtime_predict.py   # CUDA hızlandırmalı tahmin
│
└── 📁 Veri Seti
    └── ../proje-veri/            # Ham EEG verileri
        ├── araba/                # Araba düşüncesi
        ├── yukarı/               # Yukarı yön düşüncesi
        └── aşağı/                # Aşağı yön düşüncesi
```

## 🚀 Hızlı Başlangıç

### Windows'ta Canlı Tahmin (Kolay Yol)
```bash
# ThinkGear Connector çalışıyor olmalı
python windows_realtime_predict.py
```

### WSL2'de CUDA ile Hızlı Tahmin
```bash
# 1. Windows'ta proxy başlat
python thinkgear_proxy.py

# 2. WSL2'de tahmin başlat
python3 wsl_realtime_predict.py
```

## 📊 Model Performansları

| Model | Accuracy | Parametre | Özellik |
|-------|----------|-----------|---------|
| **TCN** | %92.44 | 460K | Temporal patterns, dilated conv |
| **Transformer** | %87.99 | 109K | Attention mechanism |
| **CNN-LSTM** | %84.86 | 465K | Hibrit mimari |

## 📈 EEG Özellikleri (9 Kanal)

| Band | Frekans | Açıklama |
|------|---------|----------|
| Delta | 0.5-4 Hz | Derin uyku |
| Theta | 4-8 Hz | Hafif uyku, yaratıcılık |
| Alpha | 8-12 Hz | Rahat uyanıklık |
| Beta | 12-30 Hz | Aktif düşünme |
| Gamma | 30-50 Hz | Yüksek bilişsel aktivite |

## 🛠️ Gereksinimler

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

- Python 3.10+
- CUDA 12.x (GPU için, opsiyonel)
- ThinkGear Connector (Windows, canlı tahmin için)

## 📄 Lisans

MIT License - Özgürce kullanabilir ve değiştirebilirsiniz.

---

**Geliştirici:** Kadir  
**GPU:** NVIDIA GeForce RTX 5070 (12GB)  
**Son Güncelleme:** 29 Kasım 2025
