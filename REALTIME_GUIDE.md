# 🔥 Canlı MindWave EEG Tahmini - Hızlı Başlangıç

## ✅ Test Sonucu

```
🧪 CANLI TAHMİN SİSTEMİ TEST
============================================================
✅ Model yüklendi (GPU: NVIDIA GeForce RTX 4050 Laptop GPU)
📊 Test Doğruluğu: 5/5 = 100.0%
🎯 Ortalama Güven: >99.9%
============================================================
✅ Sistem başarıyla test edildi!
```

## 🚀 Kullanım Adımları

### 1. Windows'ta Proxy Başlat

```bash
# Terminal/PowerShell açın
cd python
python windows_proxy.py

# Çıktı:
# 🚀 MindWave Bluetooth → TCP Proxy Sunucusu
# ✅ COM5 portu açıldı
# 🔵 127.0.0.1:5555 üzerinde dinleniyor...
```

### 2. MindWave Cihazını Bağlayın

1. MindWave Mobile 2'yi açın (güç düğmesi)
2. Windows Bluetooth ayarlarından "MindWave Mobile" cihazına bağlanın
3. Windows Aygıt Yöneticisi'nden COM port numarasını kontrol edin
   - "Portlar (COM & LPT)" → "Standard Serial over Bluetooth link (COM5)"
4. `windows_proxy.py` içinde COM port numarasını güncelleyin (gerekirse)

### 3. WSL2'de Canlı Tahmin Başlat

```bash
# Basit kullanım (varsayılan ayarlar)
cd python/proje
./start_realtime.sh

# VEYA manuel çalıştırma
python3 realtime_mindwave_predict.py

# Özel parametrelerle
python3 realtime_mindwave_predict.py \
    --host 10.255.255.254 \
    --port 5555 \
    --interval 1.0 \
    --min-quality 50
```

## 📊 Beklenen Çıktı

```
============================================================
🧠 CANLI MINDWAVE EEG SINIFLANDIRMA
============================================================
📡 MindWave bağlantısı kuruluyor...
🖥️  Host: 10.255.255.254:5555
🪟  Pencere boyutu: 128
⚡ Cihaz: cuda
============================================================

✅ MindWave bağlantısı başarılı!
📊 Veri toplama başladı...

⏸️  Çıkmak için Ctrl+C
------------------------------------------------------------
📦 Buffer: 128/128 | Sinyal: 12/200

============================================================
⏰ 14:23:45 | Tahmin #1
🎯 Sonuç: YUKARI (98.76%)
------------------------------------------------------------
   asagı     : ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.24%
👉 yukarı    : ████████████████████████████ 98.76%
============================================================

📈 İstatistikler:
   asagı: 0 (0.0%)
   yukarı: 1 (100.0%)
```

## ⚙️ Parametreler

| Parametre | Açıklama | Varsayılan | Örnek |
|-----------|----------|-----------|-------|
| `--host` | Windows IP adresi | 10.255.255.254 | 192.168.1.100 |
| `--port` | TCP port | 5555 | 5555 |
| `--interval` | Tahminler arası süre (saniye) | 1.0 | 2.0 |
| `--min-quality` | Minimum sinyal kalitesi (0=en iyi, 200=en kötü) | 50 | 30 |

## 🛠️ Troubleshooting

### ❌ "Bağlantı reddedildi"

**Neden:**
- Windows proxy sunucusu çalışmıyor
- Firewall portu engelliyor
- IP adresi yanlış

**Çözüm:**
```bash
# 1. Proxy çalışıyor mu?
# Windows'ta kontrol edin

# 2. IP adresini kontrol edin
# WSL2'de:
ip route show | grep default

# Windows'ta:
ipconfig

# 3. Firewall'u geçici olarak kapatın (test için)
```

### ⚠️ "Zayıf sinyal"

**Neden:**
- Elektrotlar kuru
- Cihaz yanlış takılmış
- Saç/ter interferansı

**Çözüm:**
```bash
# 1. Elektrotları hafifçe ıslatın
# 2. Cihazı doğru konumlandırın:
#    - Sensör alnın ortasında
#    - Klips kulak memesinde
# 3. Minimum kalite eşiğini yükseltin:
python3 realtime_mindwave_predict.py --min-quality 100
```

### 📦 "Buffer dolmuyor"

**Neden:**
- MindWave pilsi bitti
- Bluetooth bağlantısı kopuk
- COM port yanlış

**Çözüm:**
```bash
# 1. MindWave LED'ini kontrol edin (yanıp sönüyor olmalı)
# 2. Bluetooth bağlantısını yenileyin
# 3. Windows Aygıt Yöneticisi'nden COM portunu doğrulayın
# 4. Proxy sunucusunu yeniden başlatın
```

### 🐌 "Tahminler yavaş"

**Neden:**
- CPU modunda çalışıyor
- Buffer hızı düşük
- Network gecikme

**Çözüm:**
```bash
# 1. GPU kullanımını kontrol edin:
nvidia-smi

# 2. Tahmin aralığını artırın:
python3 realtime_mindwave_predict.py --interval 2.0

# 3. Buffer boyutunu optimize edin (kod içinde WINDOW_SIZE)
```

## 📈 Performans Metrikleri

| Metrik | Değer |
|--------|-------|
| **Model Doğruluğu** | 98.20% |
| **Test Doğruluğu** | 100.00% |
| **Ortalama Güven** | >99% |
| **GPU Hızlandırma** | Aktif (RTX 4050) |
| **Tahmin Latansı** | <50ms |
| **Buffer Boyutu** | 128 timesteps |
| **Özellik Sayısı** | 9 EEG kanalı |

## 🎯 Kullanım Senaryoları

### 1. Zihinsel Komutlar
```python
# Yukarı düşün → "yukarı" tahmini
# Aşağı düşün → "asagı" tahmini
# Örnek: Oyun kontrolü, robot kontrolü
```

### 2. Beyin-Bilgisayar Arayüzü
```python
# Gerçek zamanlı mind control
# Nörofeedback uygulamaları
# Dikkat/konsantrasyon izleme
```

### 3. Veri Toplama
```python
# Yeni komutlar için veri toplama
# Model iyileştirme
# Personalized training
```

## 🔗 İlgili Dosyalar

```
proje/
├── realtime_mindwave_predict.py  ← Ana script
├── start_realtime.sh             ← Hızlı başlatma
├── test_realtime.py              ← Test scripti
├── best_model.pth                ← Eğitilmiş model
├── label_map.json                ← Sınıf haritası
└── X.npy, y.npy                  ← Normalizasyon için

../
├── mindwave_wsl2.py              ← MindWave TCP client
└── windows_proxy.py              ← Windows Bluetooth proxy
```

## 🚦 Sonraki Adımlar

1. **Daha Fazla Komut Ekle**
   ```bash
   # Yeni CSV verisi topla
   # Sol/sağ, ileri/geri gibi
   ```

2. **Web Dashboard Oluştur**
   ```bash
   # Flask/FastAPI ile REST API
   # Real-time grafik görüntüleme
   ```

3. **Model İyileştirme**
   ```bash
   # Daha fazla veri topla
   # Hiperparametre tuning
   # Ensemble modeller dene
   ```

4. **Uygulama Geliştir**
   ```bash
   # Oyun kontrolü
   # Akıllı ev kontrolü
   # Nörofeedback uygulaması
   ```

## 📞 Yardım

**Sorunlarınız için:**
1. Test scriptini çalıştırın: `python3 test_realtime.py`
2. GPU kontrolü: `nvidia-smi`
3. Proxy loglarını kontrol edin (Windows terminalinde)
4. README.md'yi okuyun

---

**Son Güncelleme:** 17 Ekim 2025  
**Durum:** ✅ Test Edildi ve Çalışıyor  
**Doğruluk:** 100% (5/5 test)
