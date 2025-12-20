#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 EEG Sinyal Karşılaştırma Görselleştirmesi
============================================

Orijinal Raw EEG vs Kendi FFT Filtreleme Yönteminle İşlenmiş Bantlar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Türkçe karakterler için
plt.rcParams['font.family'] = 'DejaVu Sans'


def load_data(original_csv, filtered_csv, duration_seconds=180):
    """
    Orijinal ve filtrelenmiş veriyi yükle
    
    Args:
        original_csv: Orijinal veri (Time:512Hz,Epoch,Electrode,...)
        filtered_csv: Filtrelenmiş veri (Electrode,Delta,Theta,...)
        duration_seconds: Kaç saniye gösterilecek
    """
    
    print(f"📂 Orijinal veri: {original_csv}")
    print(f"📂 Filtrelenmiş veri: {filtered_csv}")
    
    # 1. Orijinal veri
    df_orig = pd.read_csv(original_csv)
    df_orig.columns = df_orig.columns.str.strip()
    
    # İlk 'start' eventini bul
    event_id_all = df_orig['Event Id'].values
    start_index = None
    
    for i, evt in enumerate(event_id_all):
        if pd.notna(evt) and str(evt).strip().lower() == 'start':
            start_index = i
            print(f"📍 İlk 'start' eventi bulundu: {i}. satır ({df_orig.iloc[i]['Time:512Hz']:.2f}s)")
            break
    
    # Event bulunamazsa baştan başla
    if start_index is None:
        print("⚠️ 'start' eventi bulunamadı, baştan başlanıyor")
        start_index = 0
    
    # Event'ten itibaren duration_seconds kadar al
    num_rows_orig = int(duration_seconds * 512)
    end_index = min(start_index + num_rows_orig, len(df_orig))
    df_orig_segment = df_orig.iloc[start_index:end_index]
    
    raw_eeg = df_orig_segment['Electrode'].values
    time_axis_raw = df_orig_segment['Time:512Hz'].values - df_orig.iloc[start_index]['Time:512Hz']  # 0'dan başlat
    attention = df_orig_segment['Attention'].values
    event_id = df_orig_segment['Event Id'].values
    
    print(f"✅ Orijinal: {len(raw_eeg)} sample (~{len(raw_eeg)/512:.1f}s)")
    
    # 2. Filtrelenmiş veri - aynı başlangıç noktasından
    df_filt = pd.read_csv(filtered_csv)
    df_filt.columns = df_filt.columns.str.strip()
    
    # FFT verisi de 512 Hz formatında (her 512 satır = 1 saniyelik FFT)
    # Her 512 satırdan birini al (downsampling)
    fft_start_row = start_index  # Raw ile aynı başlangıç
    fft_end_row = min(fft_start_row + num_rows_orig, len(df_filt))
    
    # Her 512 satırdan birini al (1 Hz'e düşür)
    fft_indices = np.arange(fft_start_row, fft_end_row, 512)
    df_filt_segment = df_filt.iloc[fft_indices]
    
    fft_bands = {
        'Delta': df_filt_segment['Delta'].values,
        'Theta': df_filt_segment['Theta'].values,
        'Low Alpha': df_filt_segment['Low Alpha'].values,
        'High Alpha': df_filt_segment['High Alpha'].values,
        'Low Beta': df_filt_segment['Low Beta'].values,
        'High Beta': df_filt_segment['High Beta'].values,
        'Low Gamma': df_filt_segment['Low Gamma'].values,
        'Mid Gamma': df_filt_segment['Mid Gamma'].values
    }
    
    # Zaman ekseni (her nokta = 1 saniye)
    time_axis_fft = np.arange(len(fft_bands['Delta']))
    
    print(f"✅ Filtrelenmiş: {len(fft_bands['Delta'])} frame (~{len(fft_bands['Delta'])}s)")
    
    # 3. Düşünme periyotlarını tespit et (Event Id'den)
    thinking_periods = []
    event_duration = 10  # 10 saniyelik alanlar
    
    for i, evt in enumerate(event_id):
        if pd.notna(evt) and str(evt).strip().lower() == 'start':
            start_time = time_axis_raw[i]
            end_time = min(start_time + event_duration, time_axis_raw[-1])
            thinking_periods.append((start_time, end_time))
            print(f"📍 Düşünme periyodu: {start_time:.2f}s → {end_time:.2f}s")
    
    # Hiç event yoksa - ilk event'i kullandık, manuel 10 saniye işaretle
    if len(thinking_periods) == 0:
        print("💡 Event başlangıcından 10 saniye işaretleniyor")
        thinking_periods = [(0, min(10, time_axis_raw[-1]))]
    
    return raw_eeg, time_axis_raw, fft_bands, time_axis_fft, thinking_periods, attention


def visualize_comparison(raw_eeg, time_axis_raw, fft_bands, time_axis_fft, thinking_periods, attention):
    """
    Orijinal Raw EEG ve İşlenmiş FFT Bantlarını Karşılaştır
    """
    
    print("\n🎨 Görselleştirme oluşturuluyor...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('🧠 EEG Sinyal İşleme Karşılaştırması - Orijinal vs Filtrelenmiş (Kendi FFT)', 
                 fontsize=16, fontweight='bold')
    
    # Düşünme periyotlarını işaretle fonksiyonu
    def mark_thinking(ax):
        for think_start, think_end in thinking_periods:
            ax.axvspan(think_start, think_end, alpha=0.15, color='green', zorder=0)
    
    # ============= SOL TARAF: ORİJİNAL VERİ =============
    
    # 1. Raw EEG
    ax1 = fig.add_subplot(gs[0, 0])
    mark_thinking(ax1)
    ax1.plot(time_axis_raw, raw_eeg, color='blue', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Amplitüd (µV)', fontweight='bold')
    ax1.set_title('📊 Orijinal Raw EEG (512 Hz)', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([time_axis_raw[0], time_axis_raw[-1]])
    
    # İlk yeşil alana label ekle
    if thinking_periods:
        ax1.axvspan(thinking_periods[0][0], thinking_periods[0][1], 
                   alpha=0.15, color='green', label='Düşünme (Att>60)', zorder=0)
        ax1.legend(loc='upper right', fontsize=9)
    
    # 2. Attention
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    mark_thinking(ax2)
    ax2.plot(time_axis_raw, attention, color='purple', linewidth=1.5)
    ax2.axhline(y=60, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (60)')
    ax2.set_ylabel('Attention', fontweight='bold')
    ax2.set_title('🎯 Attention Seviyesi', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim([time_axis_raw[0], time_axis_raw[-1]])
    
    # 3. Orijinal FFT (NeuroSky) - Low Freq
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    mark_thinking(ax3)
    # NeuroSky'dan gelen orijinal bantları kullan (eğer varsa)
    ax3.text(0.5, 0.5, 'Orijinal NeuroSky FFT\nverisi yok', 
            ha='center', va='center', transform=ax3.transAxes, 
            fontsize=12, color='gray', style='italic')
    ax3.set_ylabel('Güç', fontweight='bold')
    ax3.set_title('📉 Orijinal NeuroSky FFT (Yok)', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([time_axis_raw[0], time_axis_raw[-1]])
    
    # 4. İstatistikler
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.axis('off')
    
    stats_text = f"""
    📊 ORİJİNAL VERİ İSTATİSTİKLERİ
    {"="*35}
    
    Raw EEG:
      • Ortalama: {np.mean(raw_eeg):.2f} µV
      • Std: {np.std(raw_eeg):.2f} µV  
      • Min: {np.min(raw_eeg):.2f} µV
      • Max: {np.max(raw_eeg):.2f} µV
      • Sample: {len(raw_eeg)} (~{len(raw_eeg)/512:.1f}s)
    
    Attention:
      • Ortalama: {np.mean(attention):.1f}
      • Max: {np.max(attention):.0f}
      • Düşünme periyodu: {len(thinking_periods)}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ============= SAĞ TARAF: FİLTRELENMİŞ VERİ (KENDİ FFT) =============
    
    # 5. FFT Bantları - Düşük Frekans
    ax5 = fig.add_subplot(gs[0, 1])
    mark_thinking(ax5)
    colors_low = ['#9b59b6', '#3498db', '#1abc9c', '#2ecc71']
    for idx, band_name in enumerate(['Delta', 'Theta', 'Low Alpha', 'High Alpha']):
        ax5.plot(time_axis_fft, fft_bands[band_name], 
                label=band_name, linewidth=2, color=colors_low[idx], alpha=0.8)
    ax5.set_ylabel('Güç (µV²)', fontweight='bold')
    ax5.set_title('🌊 Filtrelenmiş FFT - Düşük Frekans (0.5-13 Hz)', fontweight='bold', fontsize=12)
    ax5.legend(loc='upper right', ncol=2, fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, time_axis_fft[-1]])
    ax5.set_yscale('log')  # Log scale - daha iyi görünüm
    
    # 6. FFT Bantları - Yüksek Frekans  
    ax6 = fig.add_subplot(gs[1, 1], sharex=ax5)
    mark_thinking(ax6)
    colors_high = ['#f39c12', '#e74c3c', '#c0392b', '#8b4513']
    for idx, band_name in enumerate(['Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']):
        ax6.plot(time_axis_fft, fft_bands[band_name], 
                label=band_name, linewidth=2, color=colors_high[idx], alpha=0.8)
    ax6.set_ylabel('Güç (µV²)', fontweight='bold')
    ax6.set_title('⚡ Filtrelenmiş FFT - Yüksek Frekans (13-50 Hz)', fontweight='bold', fontsize=12)
    ax6.legend(loc='upper right', ncol=2, fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, time_axis_fft[-1]])
    ax6.set_yscale('log')
    
    # 7. Tüm Bantların Karşılaştırması
    ax7 = fig.add_subplot(gs[2, 1], sharex=ax5)
    mark_thinking(ax7)
    all_colors = colors_low + colors_high
    for idx, (band_name, powers) in enumerate(fft_bands.items()):
        ax7.plot(time_axis_fft, powers, label=band_name, 
                linewidth=1.5, color=all_colors[idx], alpha=0.7)
    ax7.set_xlabel('Zaman (saniye)', fontweight='bold')
    ax7.set_ylabel('Güç (µV²)', fontweight='bold')
    ax7.set_title('🎼 Tüm FFT Bantları (Normalized)', fontweight='bold', fontsize=12)
    ax7.legend(loc='upper right', ncol=4, fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, time_axis_fft[-1]])
    ax7.set_yscale('log')
    
    # 8. FFT İstatistikleri
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis('off')
    
    # Her bandın ortalama gücü
    stats_fft = "📊 FİLTRELENMİŞ FFT İSTATİSTİKLERİ\n" + "="*35 + "\n\n"
    stats_fft += "Ortalama Bant Güçleri (µV²):\n"
    for band_name, powers in fft_bands.items():
        avg_power = np.mean(powers)
        stats_fft += f"  • {band_name:12s}: {avg_power:>12.2e}\n"
    
    stats_fft += f"\nToplam Frame: {len(time_axis_fft)}"
    stats_fft += f"\nSampling: 1 Hz (her saniye 1 FFT)"
    
    ax8.text(0.05, 0.95, stats_fft, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Kaydet
    output_path = 'eeg_comparison_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Görselleştirme kaydedildi: {output_path}")
    
    plt.show()
    
    # Terminal istatistikleri
    print("\n" + "=" * 70)
    print("📊 KARŞILAŞTIRMA İSTATİSTİKLERİ")
    print("=" * 70)
    print(f"\n{'ORİJİNAL RAW EEG':^35} | {'FİLTRELENMİŞ FFT':^35}")
    print("-" * 70)
    print(f"{'Sample sayısı:':<25} {len(raw_eeg):>10} | {'Frame sayısı:':<25} {len(time_axis_fft):>10}")
    print(f"{'Sampling rate:':<25} {'512 Hz':>10} | {'Sampling rate:':<25} {'1 Hz':>10}")
    print(f"{'Ortalama:':<25} {np.mean(raw_eeg):>10.2f} | {'Ortalama güç (Delta):':<25} {np.mean(fft_bands['Delta']):>10.2e}")
    print(f"{'Std:':<25} {np.std(raw_eeg):>10.2f} | {'Std (Delta):':<25} {np.std(fft_bands['Delta']):>10.2e}")
    print()
    print(f"Düşünme periyotları: {len(thinking_periods)}")
    for i, (start, end) in enumerate(thinking_periods, 1):
        print(f"  {i}. {start:.2f}s - {end:.2f}s (süre: {end-start:.2f}s)")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("🎨 EEG Sinyal Karşılaştırma Görselleştirmesi")
    print("   Orijinal Raw EEG vs Kendi FFT Filtreleme Yöntemi")
    print("=" * 70)
    
    # Veri yolları
    original_csv = '../proje-veri/asagı/apo_asagı.csv'
    filtered_csv = 'fft_model/data_filtered/asagı/apo_asagı.csv'
    
    # Veri yükle (30 saniye)
    raw_eeg, time_axis_raw, fft_bands, time_axis_fft, thinking_periods, attention = \
        load_data(original_csv, filtered_csv, duration_seconds=30)
    
    # Görselleştir
    visualize_comparison(raw_eeg, time_axis_raw, fft_bands, time_axis_fft, thinking_periods, attention)


if __name__ == "__main__":
    main()
