## LSTM+CNN Hibrit Sistem: İşlem Sırası ve Ayrıntılar

Bu bölüm canlı tahmin sisteminin uçtan uca nasıl çalıştığını, model katmanlarını, sinyal işleme adımlarını, özellik (feature) üretimini, "sliding window" mantığını ve tahmin akışını ayrıntılı olarak açıklar. Kod başlıca şu dosyalardadır: [lstm_cnn_hybrid/realtime_predict.py](lstm_cnn_hybrid/realtime_predict.py), [lstm_cnn_hybrid/realtime_gui.py](lstm_cnn_hybrid/realtime_gui.py) ve [lstm_cnn_hybrid/signal_processor.py](lstm_cnn_hybrid/signal_processor.py).

### Genel Akış
- **Giriş:** MindWave’dan 512 Hz ham EEG (`rawEeg`) + sinyal kalitesi (`poorSignalLevel`).
- **Ön-İşleme:** DC offset kaldırma → Artifact düzeltme → 50 Hz notch → 0.5–50 Hz bandpass.
- **FFT ve Bant Güçleri:** 1 saniyelik pencere (`WINDOW_SIZE=512`) üzerinde Hamming + `rfft` → NeuroSky bant güçleri (8 bant).
- **Özellik Üretimi (15 boyut):** 8 bant (log1p) + 3 toplam (alpha/beta/gamma) + 4 oran (theta/beta, alpha/beta, theta/alpha, engagement).
- **Sliding Window (Zamansal Sekans):** Her `stride=64` örnekte (~125 ms) yeni FFT; 64 çerçevelik (64×15) sekans oluşturulur.
- **Normalizasyon:** Eğitimde kaydedilmiş `scaler.pkl` ile 64×15 sekans normalize edilir.
- **Model:** `SimpleCNN_LSTM` (CNN → BiLSTM → FC) sınıf olasılıklarını üretir.
- **Smoothing:** Son tahminlerden çoğunluk oylaması (güven > 0.4) ile stabil çıktı.
- **Çıkış:** Etiket (`label_map.json`), güven skoru ve sınıf yüzdeleri GUI/CLI’de gösterilir.

### Model Katmanları (SimpleCNN_LSTM)
- **Girdi:** `seq_len=64`, `num_features=15` → tensör şekli `[batch, 64, 15]`.
- **`Conv1d(15→32, kernel=5, padding=2)`:** Zamansal komşuluk boyunca kısa vadeli paternleri çıkarır; feature kanallarını 32’ye genişletir.
- **`BatchNorm1d(32)` + `ReLU`:** Aktivasyonları stabilize eder ve doğrusal olmayan temsil sağlar.
- **`MaxPool1d(2)`:** Zaman boyutunu 2× azaltır; gürültüyü bastırıp özetler.
- **`BiLSTM(input=32, hidden=64, bidirectional=True)`:** İleri/geri yönde uzun menzilli zaman bağımlılıklarını modeller; son gizli durumlar birleştirilir (`128`).
- **`FC Head (128→64→num_classes)`:** Son temsili sınıf uzayına projeler; `Dropout` ile genelleme.
- **Çıktı:** Sınıf lojitleri → `softmax` → olasılıklar. Üç sınıf: `yukarı`, `asagı`, `araba` (etiketler `label_map.json`).

### Sinyal İşleme ve Özellikler
- **Örnekleme/Pencere:** `SAMPLING_RATE=512 Hz`, `WINDOW_SIZE=512` (1 s). İlk FFT için pencere dolmalı.
- **Stride:** `DEFAULT_STRIDE=64` örnek → ~8 FFT/saniye (125 ms). Pencere tam dolduktan sonra her 64 yeni örnekte bir FFT.
- **Adımlar:**
    - **DC Kaldırma:** Ortalama çıkarılır; sinyal merkezlenir.
    - **Artifact Düzeltme:** `±500 µV` üzeri değerler median ile değiştirilir; spike etkisi azaltılır.
    - **50 Hz Notch:** `iirnotch(w0=50/nyq, Q=30)` ile şebeke paraziti bastırılır.
    - **0.5–50 Hz Bandpass:** `butter(order=4)` ile EEG ilgili frekanslar izole edilir.
    - **Hamming + FFT:** Pencerelenmiş `rfft`; güç spektrumu `|FFT|^2`.
- **Bantlar (8):** Delta (0.5–2.75), Theta (3.5–6.75), Low Alpha (7.5–9.25), High Alpha (10–11.75), Low Beta (13–16.75), High Beta (18–29.75), Low Gamma (31–39.75), Mid Gamma (41–49.75).
- **15 Özellik:**
    - **8 bant gücü:** log ölçeğe `log1p(abs(power))`.
    - **3 toplam:** `alpha_total=low_alpha+high_alpha`, `beta_total=low_beta+high_beta`, `gamma_total=low_gamma+mid_gamma`.
    - **4 oran:** `theta_beta_ratio=theta/(beta_total+eps)`, `alpha_beta_ratio=alpha_total/(beta_total+eps)`, `theta_alpha_ratio=theta/(alpha_total+eps)`, `engagement=beta_total/(alpha_total+theta+eps)`.

### Sliding Window ve Tahmin
- **Ham pencere kaydırma:** 1 sn’lik ham pencere sabit; her `64` örnekte pencere ileri kayar ve yeni FFT hesaplanır.
- **Sekans pencere:** Her yeni FFT → 15 boyutlu vektör **sekansa eklenir**; `sequence_length=64` olduğunda model girişine hazırdır.
- **Normalize etme:** 64×15 sekans, eğitimdeki scaler ile dönüştürülür (önce `reshape`, sonra geri `reshape`).
- **Model çalıştırma:** CNN→BiLSTM→FC ile lojitler; `softmax` ile olasılıklar; en yüksek olasılıklı sınıf + güven skoru seçilir.
- **Smoothing:** Son 5 tahminin (güven > 0.4) çoğunluk oylaması ile etiket salınımı azaltılır.
- **Aralıklar:** GUI/CLI görünürlüğü için tahmin gösterimleri genelde ~0.5 s aralıkla güncellenir (FFT ~125 ms aralıkla üretilir).

### Bağlantı ve Çalıştırma
- **Modlar:**
    - **ThinkGear Connector (önerilen):** TCP/JSON `127.0.0.1:13854`; `{"enableRawOutput": true, "format": "Json"}` ile ham çıktı açılır; `rawEeg` ve `poorSignalLevel` okunur.
    - **Seri Port (doğrudan):** 57600 baud; ThinkGear ikili protokol paketlerinden `0x80` kodlu ham EEG çıkarılır.
    - **Simülasyon (GUI):** Sınıfa göre sentetik ham EEG üretir; test için uygundur.
- **İki Aşama:**
    - **Aşama 1 (Bağlan):** Cihaza bağlan, sinyal kalitesini izle, ham buffer dolumunu başlat.
    - **Aşama 2 (Başlat):** Tahmin döngüsünü başlat; ham → filtre → FFT → özellik → normalize → model → smoothing.

### Sistem Tasarım Notları
- **Eğitim-Uyum:** Canlıda kullanılan tüm filtre ve özellik dönüşümleri eğitimdekiyle birebir aynı tutulur; `scaler.pkl` ve `config.json` yüklenir.
- **Zamansal Kadans:** `stride=64` (125 ms) ve `sequence_length=64` ile LSTM’in beklediği zaman çözünürlüğü sağlanır.
- **Görsel Stabilite:** GUI bant grafiği log ölçekli ve sabit Y-ekseni ile daha az salınım gösterir; güven barları yüzdelik metinleriyle sunulur.

### Hızlı Komutlar
- **CLI (ThinkGear):**
    ```bash
    python lstm_cnn_hybrid/realtime_predict.py --thinkgear
    # Komut akışı: 'baglan' → sinyal izleme → 'basla' → tahmin
    ```
- **CLI (Seri):**
    ```bash
    python lstm_cnn_hybrid/realtime_predict.py --port /dev/ttyUSB0
    ```
- **GUI:**
    ```bash
    python lstm_cnn_hybrid/realtime_gui.py --thinkgear
    # Alternatif: --simulation ya da --port COM5
    ```

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
