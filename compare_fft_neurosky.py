#!/usr/bin/env python3
"""
NeuroSky EEG Power vs Bilgisayar FFT Karşılaştırma Scripti
==========================================================
Bu script Raw EEG verisinden FFT ile bant değerlerini hesaplar
ve NeuroSky'ın kendi hesapladığı değerlerle karşılaştırır.

Windows'ta çalıştırma:
    python compare_fft_neurosky.py

Gereksinimler:
    pip install numpy scipy

Kullanım:
    1. MindWave Mobile 2'yi Bluetooth ile bağlayın
    2. ThinkGear Connector'ı başlatın
    3. Bu scripti çalıştırın
    4. 30 saniye bekleyin (veri toplama)
    5. Korelasyon sonuçlarını görün
"""

import socket
import json
import time
import threading
from collections import deque
import numpy as np

# ThinkGear Connector ayarları
THINKGEAR_HOST = '127.0.0.1'
THINKGEAR_PORT = 13854

# FFT ayarları
SAMPLING_RATE = 512  # Raw EEG sampling rate (Hz)
FFT_WINDOW_SIZE = 512  # 1 saniyelik pencere
FFT_OVERLAP = 256  # %50 overlap

# NeuroSky frekans bantları (Hz)
FREQUENCY_BANDS = {
    'delta': (0.5, 2.75),
    'theta': (3.5, 6.75),
    'lowAlpha': (7.5, 9.25),
    'highAlpha': (10, 11.75),
    'lowBeta': (13, 16.75),
    'highBeta': (18, 29.75),
    'lowGamma': (31, 39.75),
    'highGamma': (41, 49.75)
}

# Veri depolama
raw_eeg_buffer = deque(maxlen=FFT_WINDOW_SIZE * 2)
neurosky_powers = []  # NeuroSky'dan gelen EEG Power değerleri
computed_powers = []  # Bizim FFT ile hesapladığımız değerler
comparison_count = 0
running = True


def calculate_band_powers(raw_samples):
    """
    Raw EEG verisinden FFT ile frekans bant güçlerini hesapla
    """
    if len(raw_samples) < FFT_WINDOW_SIZE:
        return None
    
    # Son FFT_WINDOW_SIZE sample'ı al
    samples = np.array(list(raw_samples)[-FFT_WINDOW_SIZE:], dtype=np.float64)
    
    # DC offset'i kaldır
    samples = samples - np.mean(samples)
    
    # Hamming window uygula (spectral leakage azaltmak için)
    window = np.hamming(len(samples))
    samples = samples * window
    
    # FFT hesapla
    fft_vals = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / SAMPLING_RATE)
    
    # Güç spektrumu (magnitude squared)
    power_spectrum = fft_vals ** 2
    
    # Her bant için güç hesapla
    band_powers = {}
    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        # Toplam güç (NeuroSky'ın yaptığı gibi)
        band_powers[band_name] = np.sum(power_spectrum[mask])
    
    return band_powers


def connect_thinkgear():
    """ThinkGear Connector'a bağlan"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(5.0)
    
    print(f"ThinkGear Connector'a bağlanılıyor ({THINKGEAR_HOST}:{THINKGEAR_PORT})...")
    sock.connect((THINKGEAR_HOST, THINKGEAR_PORT))
    
    # JSON formatında veri iste (Raw EEG dahil)
    config = json.dumps({
        "enableRawOutput": True,
        "format": "Json"
    })
    sock.send(config.encode('utf-8'))
    
    print("✓ Bağlantı başarılı!")
    return sock


def process_packet(packet):
    """Gelen paketi işle"""
    global comparison_count
    
    # Raw EEG verisi
    if 'rawEeg' in packet:
        raw_eeg_buffer.append(packet['rawEeg'])
    
    # EEG Power verisi (NeuroSky'dan)
    if 'eegPower' in packet:
        neurosky_power = packet['eegPower']
        
        # Aynı anda FFT ile hesapla
        if len(raw_eeg_buffer) >= FFT_WINDOW_SIZE:
            computed_power = calculate_band_powers(raw_eeg_buffer)
            
            if computed_power:
                neurosky_powers.append(neurosky_power)
                computed_powers.append(computed_power)
                comparison_count += 1
                
                # Her 5 karşılaştırmada bir göster
                if comparison_count % 5 == 0:
                    print(f"\n--- Karşılaştırma #{comparison_count} ---")
                    print(f"{'Bant':<12} {'NeuroSky':>12} {'FFT':>12} {'Oran':>10}")
                    print("-" * 48)
                    for band in FREQUENCY_BANDS.keys():
                        ns_val = neurosky_power.get(band, 0)
                        fft_val = computed_power.get(band, 0)
                        if ns_val > 0:
                            ratio = fft_val / ns_val
                            print(f"{band:<12} {ns_val:>12.0f} {fft_val:>12.0f} {ratio:>10.2f}")


def calculate_correlations():
    """Toplanan veriler için korelasyon hesapla"""
    if len(neurosky_powers) < 10:
        print("\n⚠ Yeterli veri toplanamadı (en az 10 gerekli)")
        return
    
    print("\n" + "=" * 60)
    print("KORELASYON ANALİZİ")
    print("=" * 60)
    
    correlations = {}
    scale_factors = {}
    
    for band in FREQUENCY_BANDS.keys():
        ns_vals = np.array([p.get(band, 0) for p in neurosky_powers])
        fft_vals = np.array([p.get(band, 0) for p in computed_powers])
        
        # Pearson korelasyonu
        if np.std(ns_vals) > 0 and np.std(fft_vals) > 0:
            correlation = np.corrcoef(ns_vals, fft_vals)[0, 1]
        else:
            correlation = 0
        
        correlations[band] = correlation
        
        # Ortalama ölçek faktörü
        valid_mask = ns_vals > 0
        if np.any(valid_mask):
            scale_factors[band] = np.median(fft_vals[valid_mask] / ns_vals[valid_mask])
        else:
            scale_factors[band] = 1.0
    
    print(f"\n{'Bant':<12} {'Korelasyon':>12} {'Ölçek Faktörü':>15} {'Durum':>12}")
    print("-" * 55)
    
    good_count = 0
    for band in FREQUENCY_BANDS.keys():
        corr = correlations[band]
        scale = scale_factors[band]
        
        if corr > 0.9:
            status = "✓ Mükemmel"
            good_count += 1
        elif corr > 0.7:
            status = "○ İyi"
            good_count += 1
        elif corr > 0.5:
            status = "△ Orta"
        else:
            status = "✗ Düşük"
        
        print(f"{band:<12} {corr:>12.3f} {scale:>15.4f} {status:>12}")
    
    avg_correlation = np.mean(list(correlations.values()))
    
    print("\n" + "=" * 60)
    print(f"Ortalama Korelasyon: {avg_correlation:.3f}")
    print("=" * 60)
    
    if avg_correlation > 0.8:
        print("\n✓ SONUÇ: FFT değerleri NeuroSky ile yüksek korelasyon gösteriyor!")
        print("  Mevcut modeli FFT tabanlı verilerle kullanabilirsiniz.")
        print("  Sadece ölçek faktörlerini uygulayarak uyumluluk sağlanabilir.")
    elif avg_correlation > 0.6:
        print("\n○ SONUÇ: Orta düzeyde korelasyon var.")
        print("  FFT kullanılabilir ama doğruluk düşebilir.")
        print("  Model yeniden eğitimi önerilir.")
    else:
        print("\n✗ SONUÇ: Düşük korelasyon.")
        print("  NeuroSky farklı bir algoritma kullanıyor olabilir.")
        print("  FFT tabanlı verilerle model yeniden eğitilmeli.")
    
    # Ölçek faktörlerini kaydet
    print("\n\nÖlçek faktörleri (FFT → NeuroSky dönüşümü için):")
    print("-" * 40)
    scale_dict = {band: float(scale_factors[band]) for band in FREQUENCY_BANDS.keys()}
    print(json.dumps(scale_dict, indent=2))
    
    # Dosyaya kaydet
    with open('fft_scale_factors.json', 'w') as f:
        json.dump({
            'correlations': {k: float(v) for k, v in correlations.items()},
            'scale_factors': scale_dict,
            'average_correlation': float(avg_correlation),
            'sample_count': len(neurosky_powers)
        }, f, indent=2)
    print("\n✓ Sonuçlar 'fft_scale_factors.json' dosyasına kaydedildi.")


def wait_for_device_connection(sock):
    """Cihazın bağlanmasını bekle (poorSignalLevel < 200)"""
    print("\n⏳ MindWave cihazının bağlanması bekleniyor...")
    print("   (Cihazı takın ve kulak klipsini bağlayın)\n")
    
    buffer = ""
    connected = False
    last_signal = 200
    
    while not connected:
        try:
            data = sock.recv(4096).decode('utf-8')
            if not data:
                continue
            
            buffer += data
            
            while '\r' in buffer:
                line, buffer = buffer.split('\r', 1)
                line = line.strip()
                if line:
                    try:
                        packet = json.loads(line)
                        if 'poorSignalLevel' in packet:
                            signal = packet['poorSignalLevel']
                            last_signal = signal
                            
                            if signal == 200:
                                status = "Cihaz bağlı değil"
                            elif signal > 50:
                                status = "Sinyal zayıf - Elektrotları kontrol edin"
                            elif signal > 0:
                                status = "Sinyal orta - Biraz bekleyin"
                            else:
                                status = "Sinyal mükemmel!"
                                connected = True
                            
                            # Sinyal çubuğu göster
                            bar_len = 20
                            filled = int((200 - signal) / 200 * bar_len)
                            bar = "█" * filled + "░" * (bar_len - filled)
                            print(f"\r   Sinyal: [{bar}] {200-signal}/200 - {status}     ", end="", flush=True)
                            
                            # Sinyal 50'nin altındaysa bağlı say
                            if signal < 50:
                                connected = True
                    except json.JSONDecodeError:
                        pass
        except socket.timeout:
            print(f"\r   Sinyal: Bekleniyor... (poorSignalLevel: {last_signal})     ", end="", flush=True)
            continue
    
    print("\n\n✓ Cihaz bağlandı! Test başlıyor...\n")
    return buffer


def main():
    global running
    
    print("=" * 60)
    print("NeuroSky vs FFT Karşılaştırma Testi")
    print("=" * 60)
    print("\nBu test, Raw EEG'den hesaplanan FFT değerlerinin")
    print("NeuroSky'ın chip üzerinde hesapladığı değerlerle")
    print("ne kadar uyumlu olduğunu ölçer.\n")
    
    try:
        sock = connect_thinkgear()
    except Exception as e:
        print(f"\n✗ Bağlantı hatası: {e}")
        print("\nKontrol edin:")
        print("1. ThinkGear Connector çalışıyor mu?")
        print("2. MindWave Bluetooth ile bağlı mı?")
        return
    
    # Cihaz bağlantısını bekle
    remaining_buffer = wait_for_device_connection(sock)
    
    print("⏳ Veri toplanıyor... (30 saniye)")
    print("   Raw EEG buffer dolduğunda karşılaştırma başlayacak.\n")
    
    buffer = remaining_buffer
    start_time = time.time()
    test_duration = 30  # 30 saniye test
    
    try:
        while running and (time.time() - start_time) < test_duration:
            try:
                data = sock.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                while '\r' in buffer:
                    line, buffer = buffer.split('\r', 1)
                    line = line.strip()
                    if line:
                        try:
                            packet = json.loads(line)
                            process_packet(packet)
                        except json.JSONDecodeError:
                            pass
                
                # İlerleme göster
                elapsed = time.time() - start_time
                raw_count = len(raw_eeg_buffer)
                print(f"\r[{elapsed:5.1f}s] Raw buffer: {raw_count}/{FFT_WINDOW_SIZE} | "
                      f"Karşılaştırma: {comparison_count}", end="", flush=True)
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"\nHata: {e}")
                break
        
        print("\n\n⏳ Test tamamlandı, sonuçlar hesaplanıyor...")
        calculate_correlations()
        
    except KeyboardInterrupt:
        print("\n\n⏹ Test durduruldu.")
        if comparison_count > 10:
            calculate_correlations()
    finally:
        sock.close()


if __name__ == '__main__':
    main()
