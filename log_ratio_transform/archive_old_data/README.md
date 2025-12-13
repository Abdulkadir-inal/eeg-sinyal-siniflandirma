# Arşiv: Eski NeuroSky Verisi ile Transform

Bu klasör, **eski NeuroSky ham verileri** kullanılarak oluşturulmuş transform verilerini içerir.

## ⚠️ Neden Arşivlendi?

**SORUN**: Yanlış veri kaynağı kullanıldı!

Log Transform + Oran Formülleri tekniği geliştirilirken, **eski NeuroSky ham verileri** 
(`/home/kadir/sanal-makine/python/proje-veri/`) üzerinden çalışıldı. 

Ancak proje artık **FFT hesaplamasını kendimiz yapıyoruz**:
- ✅ Doğru yol: `../fft_model/` → Raw EEG 512Hz → FFT → 8 bant gücü
- ❌ Kullanılan: Eski NeuroSky'dan gelen hazır bant güçleri

**Sonuç**: Bu verilerle eğitilen modeller FFT tabanlı realtime sistemle uyumlu değil!

## 🔧 Çözüm

Transform tekniği doğru, sadece veri kaynağı değiştirildi:
- `data_preprocess_transformed.py` → Artık `../fft_model/data/` kullanıyor
- Yeni verilerle model yeniden eğitilecek
- FFT + Transform pipeline tam uyumlu olacak

## Arşivlenen Dosyalar

- `X_transformed.npy` - Eski verilerle transform edilmiş özellikler
- `y_transformed.npy` - Eski etiketler
- `scaler_transformed.pkl` - Eski scaler
- `best_model_transformed.pth` - Eski model (yanlış veriyle eğitilmiş)
- `final_model_transformed.pth` - Eski model (yanlış veriyle eğitilmiş)
- `training_history_transformed.png` - Eski eğitim grafiği

## Yeni Veri Oluşturma

Doğru FFT verilerini kullanarak yeni transform verileri oluşturmak için:

```bash
cd /home/kadir/sanal-makine/python/proje/log_ratio_transform
python3 data_preprocess_transformed.py
```

---

**Arşivlenme Tarihi**: 11 Aralık 2025  
**Sebep**: Yanlış veri kaynağı (NeuroSky ham → FFT hesaplanan)
