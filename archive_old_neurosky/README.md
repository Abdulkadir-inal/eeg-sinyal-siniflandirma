# Arşiv: Eski NeuroSky Ham Veri Sistemi

Bu klasör, **eski NeuroSky ham veri** (Attention, Meditation, 8 bant gücü) ile çalışan dosyaları içerir.

## ⚠️ Önemli Not

Bu dosyalar **ARTIK KULLANILMIYOR**. Aktif sistemler:

- **`fft_model/`** - Raw EEG 512Hz + FFT hesaplaması (Ana sistem)
- **`log_ratio_transform/`** - FFT + Log Transform + Oran Formülleri

## Arşivlenen Dosyalar

### Veri ve Modeller
- `X.npy`, `y.npy` - Eski NeuroSky ham verileri
- `best_model.pth`, `final_model.pth` - Eski eğitilmiş modeller
- `scaler_params.json`, `label_map.json` - Eski yapılandırma dosyaları
- `training_history.png`, `training_log_3class.txt` - Eski eğitim sonuçları

### Kod Dosyaları
- `data_preprocess.py` - Eski veri ön işleme
- `train_model.py` - Eski model eğitimi
- `predict.py` - Eski tahmin kodu
- `realtime_mindwave_predict.py` - Eski realtime tahmin
- `test_realtime.py`, `start_realtime.sh`, `verify_setup.py` - Eski test dosyaları

### MindWave Bağlantı Testleri
- `mindwave_thinkgear_binary.py` - ThinkGear binary protokol testi
- `test_mindwave_connection.py` - Bağlantı testi
- `test_windows_com.py` - Windows COM port testi
- `mindwave_wsl2.py` - WSL2 bağlantı testi
- `thinkgear_proxy.py` - ThinkGear proxy

### Proxy ve Realtime
- `windows_proxy_auto.py`, `windows_proxy_raw.py` - Eski proxy sistemleri
- `windows_realtime_predict.py` - Eski Windows realtime
- `wsl_realtime_predict.py` - Eski WSL realtime

### Analiz ve Görselleştirme
- `analyze_fft_bands.py` - FFT bant analizi
- `performance_analysis.py` - Performans analizi
- `visualize_eeg_comparison.py` - EEG karşılaştırma görselleştirmesi
- `visualize_signal_processing.py` - Sinyal işleme görselleştirmesi
- `correlation_matrix.png`, `eeg_comparison_visualization.png`, `fft_band_comparison.png`, `sample_eeg_window.png`, `variance_analysis.png` - Görselleştirmeler

### Dokümantasyon
- `REALTIME_GUIDE.md` - Eski realtime kılavuzu
- `WINDOWS_PROXY_README.md` - Eski proxy dokümantasyonu

## Neden Arşivlendi?

1. **Düşük Doğruluk**: NeuroSky ham verileri 95.70% doğruluk sağlıyordu
2. **FFT Geçiş**: Raw EEG 512Hz + FFT hesaplaması daha iyi sonuç verdi
3. **Karışıklık**: Yapay zeka eski verileri kullanarak yanlış kod üretiyordu
4. **Aktif Sistem**: `fft_model/` ve `log_ratio_transform/` aktif olarak kullanılıyor

## Geri Yükleme

Bu dosyaları tekrar kullanmak isterseniz:

```bash
# Tek dosya geri yükleme
cp archive_old_neurosky/[dosya_adı] .

# Tüm dosyaları geri yükleme (ÖNERİLMEZ)
cp -r archive_old_neurosky/* .
```

---

**Arşivlenme Tarihi**: 9 Aralık 2025  
**Sebep**: FFT tabanlı sistem geçişi ve kod karışıklığı önleme
