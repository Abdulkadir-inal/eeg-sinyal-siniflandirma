# 🧠 MindWave Canlı EEG Tahmin Sistemi

Windows'ta ThinkGear Connector üzerinden MindWave Mobile 2 cihazından canlı EEG verisi alarak gerçek zamanlı tahmin yapan sistem.

## ✨ Özellikler

- 🖥️ **Windows'ta Doğrudan Çalışır** - WSL2 veya Linux gerekmez
- 🧮 **CPU Modu** - CUDA/GPU gerekmez, herhangi bir bilgisayarda çalışır
- 🎯 **4 Farklı Model Seçeneği** - İstediğiniz modeli seçebilirsiniz
- 🔌 **ThinkGear Connector Entegrasyonu** - Güvenilir JSON veri akışı
- 📊 **Canlı İstatistikler** - Tahmin sonuçlarını anlık görüntüler

## 📋 Gereksinimler

### Donanım
- MindWave Mobile 2 cihazı
- Bluetooth destekli Windows bilgisayar

### Yazılım
1. **ThinkGear Connector** (NeuroSky resmi yazılımı)
2. Python kütüphaneleri:
```bash
pip install torch numpy
```

## 🔧 ThinkGear Connector Kurulumu

### ThinkGear Connector Nedir?
ThinkGear Connector, NeuroSky'ın MindWave cihazları için geliştirdiği resmi yazılımdır. MindWave'den gelen ham Bluetooth verisini işler ve uygulamalara düzgün JSON formatında sunar.

### İndirme ve Kurulum

1. **İndirin**: 
   - [NeuroSky Store](http://store.neurosky.com/products/thinkgear-connector)
   - veya MindWave kutusuyla gelen CD'den

2. **Kurun**: `ThinkGear Connector.exe` dosyasını çalıştırın

3. **Başlatın**: Kurulum sonrası otomatik başlar veya:
   - Başlat Menüsü → ThinkGear Connector
   - Sistem tray'de (sağ alt köşe) ThinkGear ikonu görünür

4. **MindWave'i Bağlayın**:
   - MindWave cihazını açın
   - Windows Bluetooth ayarlarından "MindWave Mobile" eşleştirin
   - ThinkGear Connector otomatik olarak bağlanır
   - Tray ikonunda yeşil ışık = bağlı

### ThinkGear Connector Portu
- **Host**: `127.0.0.1` (localhost)
- **Port**: `13854`
- **Format**: JSON stream

## 🚀 Kurulum

### 1. ThinkGear Connector'ı Kurun
Yukarıdaki "ThinkGear Connector Kurulumu" bölümüne bakın.

### 2. Dosyaları İndirin

Aşağıdaki dosya yapısını Windows'a kopyalayın:

```
proje/
├── windows_realtime_predict.py    # Ana program
├── label_map.json                 # Sınıf etiketleri
└── model_experiments/
    ├── TCN/
    │   └── tcn_best_model.pth     # TCN model (%92.44)
    ├── Transformer/
    │   ├── transformer_80epoch_best_model.pth   # Transformer (%87.99)
    │   └── transformer_best_model.pth           # Transformer (%86.25)
    └── CNN_LSTM/
        └── cnn_lstm_best_model.pth              # CNN-LSTM (%84.86)
```

### 3. Python Kütüphanelerini Kurun
```bash
pip install torch numpy
```

## 🎮 Kullanım

### Adım 1: ThinkGear Connector'ı Başlatın
1. Sistem tray'de ThinkGear ikonuna çift tıklayın
2. MindWave cihazını açın
3. Bağlantı kurulduğunda ikon yeşile döner

### Adım 2: Scripti Çalıştırın
```bash
python windows_realtime_predict.py
```

### Adım 3: Model Seçin
```
🧠 MODEL SEÇİMİ
============================================================
   1. TCN (En İyi - %92.44)
   2. Transformer 80 epoch (%87.99)
   3. Transformer 50 epoch (%86.25)
   4. CNN-LSTM (%84.86)
   q. Çıkış
------------------------------------------------------------
Model seçin (1-4): 
```

### Adım 4: MindWave'i Takın
- Kulak kıskacını kulak memenize takın
- Alın sensörünü alnınıza yerleştirin
- 5-10 saniye bekleyin (sinyal stabilize olsun)

### Adım 5: Canlı Tahminleri İzleyin
```
📦 Buffer: 128/128 | Sinyal: ✅ Mükemmel | Dikkat: 67 | Meditasyon: 43

============================================================
⏰ 14:23:45 | Tahmin #5
🎯 Sonuç: YUKARI (98.76%)
------------------------------------------------------------
👉 araba    : ████                 15.2% (1)
   yukarı   : ████████████████████ 78.5% (4)
   aşağı    : █                    6.3% (0)
============================================================
```

### Adım 6: Durdurmak İçin
`Ctrl+C` tuşlarına basın

## 📊 Modeller

| Model | Accuracy | Parametre | Açıklama |
|-------|----------|-----------|----------|
| **TCN** | %92.44 | 460K | En iyi performans, temporal patterns |
| **Transformer 80** | %87.99 | 109K | Attention-based, optimal epoch |
| **Transformer 50** | %86.25 | 109K | Baseline transformer |
| **CNN-LSTM** | %84.86 | 465K | Hibrit model |

## 🎯 Sınıflar

| Sınıf | Açıklama |
|-------|----------|
| `araba` | Araba düşünme/hayal etme |
| `yukarı` | Yukarı yön düşüncesi |
| `aşağı` | Aşağı yön düşüncesi |

## ⚙️ Teknik Detaylar

### Veri İşleme
- **Pencere boyutu**: 128 örnek
- **Özellikler**: 9 (Delta, Theta, Low/High Alpha, Low/High Beta, Low/High Gamma, Electrode)
- **Tahmin aralığı**: 1 saniye

### ThinkGear Connector Veri Formatı
ThinkGear Connector'dan gelen JSON verisi:
```json
{
  "eSense": {
    "attention": 67,
    "meditation": 43
  },
  "eegPower": {
    "delta": 123456,
    "theta": 234567,
    "lowAlpha": 34567,
    "highAlpha": 45678,
    "lowBeta": 56789,
    "highBeta": 67890,
    "lowGamma": 78901,
    "highGamma": 89012
  },
  "poorSignalLevel": 0
}
```

### Sinyal Kalitesi (poorSignalLevel)
- `0` = Mükemmel sinyal
- `1-50` = Kabul edilebilir
- `51-200` = Zayıf sinyal (tahminler güvenilir olmayabilir)

## 🛠️ Sorun Giderme

### ❌ "Bağlantı reddedildi: 127.0.0.1:13854"
- ThinkGear Connector çalışmıyor
- Sistem tray'de ThinkGear ikonunu kontrol edin
- ThinkGear Connector'ı yeniden başlatın

### ❌ "ThinkGear Connector bağlı değil"
- MindWave cihazı kapalı olabilir
- Bluetooth eşleştirmesi yapılmamış olabilir
- ThinkGear tray ikonunda kırmızı = bağlı değil

### ❌ "Model dosyası bulunamadı"
- `model_experiments/` klasörünün doğru konumda olduğundan emin olun
- `.pth` dosyalarının mevcut olduğunu kontrol edin

### ⚠️ Sinyal kalitesi düşük (poorSignalLevel yüksek)
- Kulak kıskacının cilde temas ettiğinden emin olun
- Alın sensörünü temiz cilde yerleştirin
- Saç sensör ile cilt arasında olmamalı
- Cihazı çıkarıp tekrar takın

## 📝 Notlar

- İlk tahmin için 128 EEG örneği toplanması gerekir (~10-15 saniye)
- Sinyal kalitesi düşükken tahminler güvenilir olmayabilir
- Model CPU'da çalışır, GPU olmadan da performans yeterlidir

## 📄 Lisans

MIT License

## 🤝 Katkı

Pull request'ler kabul edilir. Büyük değişiklikler için önce issue açınız.
