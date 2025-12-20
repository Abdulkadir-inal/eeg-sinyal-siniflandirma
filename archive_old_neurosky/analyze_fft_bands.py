import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Dosya yolları
class_dirs = {
    'araba': '/home/kadir/sanal-makine/python/proje-veri/araba/',
    'asagı': '/home/kadir/sanal-makine/python/proje-veri/asagı/',
    'yukarı': '/home/kadir/sanal-makine/python/proje-veri/yukarı/'
}

band_columns = ['Delta', 'Theta', 'Low Alpha', 'High Alpha', 
                'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma']

print("=" * 70)
print("FFT BANT GÜÇLERI ANALİZİ")
print("=" * 70)

# ============================================================================
# ADIM 1: Tüm verileri yükle ve sınıf başına ortalamalar hesapla
# ============================================================================
print("\n[1/3] FFT Bant Güçleri Yükleniyor...")

classes_data = {}
all_data = {class_name: [] for class_name in class_dirs.keys()}

for class_name, class_path in class_dirs.items():
    print(f"  → {class_name}: ", end="")
    
    csv_files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
    print(f"{len(csv_files)} dosya", end=" ... ")
    
    class_band_data = []
    
    for file in csv_files:
        file_path = os.path.join(class_path, file)
        df = pd.read_csv(file_path)
        
        # Sadece bant sütunlarını al
        bands = df[band_columns]
        class_band_data.append(bands)
        all_data[class_name].append(df)
    
    # Tüm dosyaları birleştir
    if class_band_data:
        classes_data[class_name] = pd.concat(class_band_data, ignore_index=True)
        print(f"Toplam {len(classes_data[class_name])} satır")

# ============================================================================
# ANALIZ 1: FFT Bant Güçleri Görselleştirmesi
# ============================================================================
print("\n" + "=" * 70)
print("ANALİZ 1: FFT BANT GÜÇLERI KARŞILAŞTIRMASI")
print("=" * 70)

# Her sınıf için ortalama hesapla
class_means = {}
for class_name, df in classes_data.items():
    class_means[class_name] = df[band_columns].mean()
    
    print(f"\n{class_name.upper()}:")
    for band in band_columns:
        print(f"  {band:12} = {class_means[class_name][band]:8.3f} μV")

# Görselleştir
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Grafik 1: Bar Chart
classes_df = pd.DataFrame(class_means).T
ax = axes[0]
classes_df.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Sınıflar Arasında FFT Bant Güçleri Farkı (Ortalama)', fontsize=14, fontweight='bold')
ax.set_ylabel('Ortalama Güç (μV)', fontsize=11)
ax.set_xlabel('Sınıf', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

# Grafik 2: Heatmap
ax = axes[1]
im = ax.imshow(classes_df.values, cmap='RdYlGn', aspect='auto')
ax.set_xticks(np.arange(len(band_columns)))
ax.set_yticks(np.arange(len(class_means)))
ax.set_xticklabels(band_columns, rotation=45, ha='right')
ax.set_yticklabels(class_means.keys())
ax.set_title('FFT Bant Güçleri Heatmap', fontsize=14, fontweight='bold')

# Değerleri yazdir
for i in range(len(class_means)):
    for j in range(len(band_columns)):
        text = ax.text(j, i, f'{classes_df.values[i, j]:.1f}',
                      ha="center", va="center", color="black", fontsize=9)

fig.colorbar(im, ax=ax, label='Güç (μV)')
plt.tight_layout()
plt.savefig('/home/kadir/sanal-makine/python/proje/fft_band_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Grafik kaydedildi: fft_band_comparison.png")
plt.close()

# ============================================================================
# ANALIZ 2: Korelasyon Analizi
# ============================================================================
print("\n" + "=" * 70)
print("ANALİZ 2: KORELASYON ANALİZİ")
print("=" * 70)

# Sınıflar arası korelasyon hesapla (her sınıfın bant vektörü)
class_vectors = {name: vector.values for name, vector in class_means.items()}
correlation_data = {}

print("\nSınıflar Arasında Korelasyon (Band Vektörleri):")

class_names = list(class_vectors.keys())
for i, class1 in enumerate(class_names):
    for j, class2 in enumerate(class_names):
        if i <= j:
            vec1 = class_vectors[class1]
            vec2 = class_vectors[class2]
            # Pearson korelasyonu
            corr = np.corrcoef(vec1, vec2)[0, 1]
            correlation_data[f"{class1}-{class2}"] = corr
            
            if i < j:
                if corr > 0.9:
                    status = "🔴 PROBLEM: Çok benzer!"
                elif corr > 0.7:
                    status = "🟡 UYARI: Benzer"
                elif corr > 0.5:
                    status = "🟢 ORTA"
                else:
                    status = "✅ İYİ: Yeterince farklı"
                print(f"  {class1} ↔ {class2}: {corr:.3f} {status}")

# Korelasyon matrisi
corr_matrix = np.zeros((len(class_names), len(class_names)))
for i, class1 in enumerate(class_names):
    for j, class2 in enumerate(class_names):
        vec1 = class_vectors[class1]
        vec2 = class_vectors[class2]
        corr_matrix[i, j] = np.corrcoef(vec1, vec2)[0, 1]

# Korelasyon heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='RdYlGn_r', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticklabels(class_names)
ax.set_title('Sınıflar Arası Korelasyon Matrisi\n(1.0=Benzer, -1.0=Farklı)', 
             fontsize=12, fontweight='bold')

for i in range(len(class_names)):
    for j in range(len(class_names)):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontweight='bold')

fig.colorbar(im, ax=ax, label='Korelasyon')
plt.tight_layout()
plt.savefig('/home/kadir/sanal-makine/python/proje/correlation_matrix.png', dpi=150, bbox_inches='tight')
print("\n✓ Grafik kaydedildi: correlation_matrix.png")
plt.close()

# ============================================================================
# ANALIZ 3: Sınıf İçi vs Sınıflar Arası Varyans
# ============================================================================
print("\n" + "=" * 70)
print("ANALİZ 3: SINIF İÇİ vs SINIFLAR ARASI VARYANS")
print("=" * 70)

variance_analysis = {}

for class_name, df in classes_data.items():
    # Sınıf içi standart sapma (ne kadar değişken?)
    within_class_std = df[band_columns].std().mean()
    variance_analysis[class_name] = {'within_std': within_class_std}

# Sınıflar arası standart sapma (ne kadar farklı?)
between_class_std = classes_df.std().mean()

print(f"\nSınıflar Arası Standart Sapma (Between-Class): {between_class_std:.3f} μV")
print("\nSınıf İçi Standart Sapma (Within-Class):")

for class_name in variance_analysis.keys():
    within_std = variance_analysis[class_name]['within_std']
    ratio = within_std / between_class_std
    
    if ratio > 1.5:
        status = "🔴 PROBLEM: Sınıf içi varyans çok yüksek!"
    elif ratio > 1.0:
        status = "🟡 UYARI: Sınıf içi varyans > sınıflar arası"
    elif ratio > 0.5:
        status = "🟢 ORTA: Makul"
    else:
        status = "✅ İYİ: Düşük sınıf içi varyans"
    
    print(f"  {class_name:8} = {within_std:.3f} μV (Oran: {ratio:.2f}x) {status}")

# Görselleştir
fig, ax = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(variance_analysis))
within_stds = [variance_analysis[c]['within_std'] for c in variance_analysis.keys()]

bars1 = ax.bar(x_pos - 0.2, within_stds, 0.4, label='Sınıf İçi Varyans', color='#FF6B6B')
bars2 = ax.bar(x_pos + 0.2, [between_class_std] * len(variance_analysis), 0.4, 
               label='Sınıflar Arası Varyans', color='#4ECDC4')

ax.set_ylabel('Standart Sapma (μV)', fontsize=11)
ax.set_xlabel('Sınıf', fontsize=11)
ax.set_title('Sınıf İçi vs Sınıflar Arası Varyans\n(Düşük oran = İyi ayırılabilirlik)', 
             fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(variance_analysis.keys())
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Değerleri yazdır
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/home/kadir/sanal-makine/python/proje/variance_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Grafik kaydedildi: variance_analysis.png")
plt.close()

# ============================================================================
# ÖZET VE SONUÇ
# ============================================================================
print("\n" + "=" * 70)
print("ÖZET VE ÖNERİLER")
print("=" * 70)

# Ortalama korelasyon
indices = np.triu_indices(len(class_names), k=1)
avg_corr = corr_matrix[indices].mean()

print(f"\n✓ Ortalama Sınıf Benzerliği: {avg_corr:.3f}")

if avg_corr > 0.85:
    print("  🔴 SONUÇ: Sınıflar ÇOK BENZER - Model ayırt edemeyebilir!")
    print("  Öneriler:")
    print("    1. Yeni frekans bandları dene (mü-ritmi, beta ağırlıklı)")
    print("    2. İlave özellikler ekle (faz fark, asimetri)")
    print("    3. Transfer Learning yap (kişiye özel model)")
    print("    4. Daha iyi sensör kullan")
elif avg_corr > 0.75:
    print("  🟡 UYARI: Sınıflar BENZER - Ayırılabilir ama sıkıntılı")
    print("  Öneriler:")
    print("    1. Yeni özellikler ekleyerek denemeler yap")
    print("    2. Hyperparameter tuning (learning rate, window size)")
    print("    3. Transfer Learning düşün")
else:
    print("  ✅ İYİ: Sınıflar YETERINCE FARKI")
    print("  Çıkmazın sebebi başka yerdedir:")
    print("    1. Model overfitting olmuş olabilir")
    print("    2. Calibration/Scaler uyumsuzluğu devam ediyor")
    print("    3. Sinyal kalitesi sorunları (gürültü, artifact)")

print(f"\nOrtalama Sınıf İçi/Arası Varyans Oranı: {np.mean(within_stds) / between_class_std:.2f}x")
if np.mean(within_stds) / between_class_std > 1.0:
    print("  🔴 Sınıf içi varyans çok yüksek!")
else:
    print("  ✅ Sınıf içi varyans kontrol altında")

print("\n" + "=" * 70)
print("✓ Analiz Tamamlandı. 3 grafik oluşturuldu:")
print("  1. fft_band_comparison.png")
print("  2. correlation_matrix.png")
print("  3. variance_analysis.png")
print("=" * 70)
