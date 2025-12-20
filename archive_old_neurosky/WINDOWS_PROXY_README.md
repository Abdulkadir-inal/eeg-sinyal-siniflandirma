# 🪟 Windows MindWave Proxy Dosyaları

Bu klasördeki dosyalar **Windows'ta** çalıştırılmalıdır (WSL2'de değil!).

## 📁 Dosyalar

### 1. `windows_proxy_auto.py` ⭐ (Önerilen)
**Amaç:** MindWave Bluetooth → TCP/IP proxy (otomatik COM port tespiti)

**Kullanım:**
```cmd
# Windows PowerShell veya CMD:
cd proje
python windows_proxy_auto.py
```

**Özellikler:**
- ✅ Otomatik COM port taraması
- ✅ MindWave cihazını otomatik bulur
- ✅ Exclusive access (başka program engellemez)
- ✅ Detaylı hata mesajları

---

### 2. `test_windows_com.py`
**Amaç:** COM portlarını test et ve MindWave'den veri gelip gelmediğini kontrol et

**Kullanım:**
```cmd
# Windows'ta:
python test_windows_com.py
```

**Ne yapar:**
- Tüm COM portlarını tarar
- Her birinden veri gelip gelmediğini test eder
- MindWave hangi portta bağlı olduğunu söyler

---

### 3. `mindwave_thinkgear_binary.py`
**Amaç:** ThinkGear Connector üzerinden MindWave verisi oku

**Kullanım:**
```cmd
# Windows'ta (ThinkGear Connector çalışıyor olmalı):
python mindwave_thinkgear_binary.py
```

**Gereksinim:**
- ThinkGear Connector yüklü ve çalışıyor olmalı
- İndirme: https://store.neurosky.com/pages/mindwave

---

## 🚀 Hızlı Başlangıç

### Adım 1: Windows'ta Proxy Başlat

```cmd
# Terminal/PowerShell açın
cd C:\Users\<Kullanıcı>\Desktop\code\sanal makine\python\proje
python windows_proxy_auto.py
```

**Beklenen Çıktı:**
```
📋 Bulunan COM portları (3 adet):
  1. COM3 - Bluetooth bağlantısı üzerinden Standart Seri
     ⭐ MindWave olabilir!
  ...
  
✅ COM3 seçildi
🌐 TCP sunucu başlatıldı: 0.0.0.0:5555
⏳ Bağlantı bekleniyor...
```

### Adım 2: WSL2'de Canlı Tahmin Başlat

```bash
# WSL2 terminalinde:
cd /home/kadir/sanal-makine/python/proje
python3 realtime_mindwave_predict.py
```

---

## 🔧 Sorun Giderme

### ❌ "COM port açılamıyor" Hatası

**Çözüm:**
```cmd
1. Task Manager → ThinkGear Connector'ı kapat
2. Device Manager → COM portunu Disable/Enable
3. Proxy'yi tekrar başlat
```

### ❌ "Hiç veri gelmiyor"

**Çözüm:**
```cmd
1. MindWave açık mı kontrol edin (mavi LED)
2. test_windows_com.py ile doğru portu bulun
3. MindWave'i başınıza takın (sensör teması olmalı)
```

### ❌ "Bağlantı reddedildi" (WSL2'de)

**Çözüm:**
```bash
# WSL2'de Windows IP'yi kontrol edin:
ip route show | grep default | awk '{print $3}'

# realtime_mindwave_predict.py'de bu IP'yi kullanın:
python3 realtime_mindwave_predict.py --host 172.20.16.1
```

---

## 📌 Önemli Notlar

1. **Proxy Windows'ta çalışmalı** - WSL2'nin Bluetooth desteği yok
2. **Tek bağlantı** - MindWave aynı anda sadece bir programa bağlanır
3. **COM portu** - Her bilgisayarda farklı olabilir (COM3, COM4, COM5...)
4. **Firewall** - Windows Firewall 5555 portunu açık bırakmalı

---

## 🔗 İlgili Dosyalar

**WSL2'de:**
- `realtime_mindwave_predict.py` - Canlı EEG sınıflandırma
- `mindwave_wsl2.py` - TCP client (Windows proxy'ye bağlanır)

**Model Dosyaları:**
- `best_model.pth` - Eğitilmiş model
- `label_map.json` - Sınıf haritası (aşağı, yukarı, durgun)

---

**Son Güncelleme:** 18 Ekim 2025  
**Durum:** ✅ Test Edildi (Windows 11 + WSL2 Ubuntu)
