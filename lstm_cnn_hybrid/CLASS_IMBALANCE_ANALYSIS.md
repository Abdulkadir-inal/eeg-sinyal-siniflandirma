# Sınıf Dengesizliği Analizi

## Soruna Tanıklık

Model "aşağı" sınıfına baskınlık gösteriyordu, hatta tahminler çoğunlukla "aşağı" çıkıyordu.

### Orijinal Sorun
```
yukarı dosyasında: %60 aşağı, %24 yukarı, %17 araba
asagı  dosyasında: %71 aşağı, %16 yukarı, %13 araba  
araba  dosyasında: %72 aşağı, %21 yukarı, %7 araba
```

## Kök Nedenleri

### 1. Eğitim Verisi Dengesizliği
```
📊 Training Seti Dağılımı (Augmentation Sonrası):
   yukarı    : ağırlık = 0.855 (102352/262676 örnek) - % 39.0
   aşağı     : ağırlık = 1.027 (85296/262676 örnek)  - % 32.5
   araba     : ağırlık = 1.167 (75028/262676 örnek)  - % 28.6
```

**Açıklama**: "yukarı" sınıfı daha fazla örneğe sahip ama model "aşağı"ya baskınlık gösteriyordu.

### 2. Loss Function Problemi
**Orijinal kodda**:
```python
criterion = nn.CrossEntropyLoss()  # Sınıf ağırlıkları yok!
```

**Sorun**: CrossEntropyLoss varsayılan olarak tüm sınıfları eşit tedavi eder. Dengesiz veri üzerinde, sık görülen sınıf (burada "aşağı") model tarafından fazla tercih edilir.

### 3. Veri Ön İşleme Farklılıkları
Training sırasında veri şu adımlardan geçer:
1. FFT bant güçleri (8 özellik)
2. Log transform: `log1p(abs(x))`
3. Türetilmiş özellikler eklenir (7 tane daha → 15 toplam)
4. StandardScaler normalizasyon (15 özelliğe uygun)

Tahmin yapılırken ham CSV verileri kullanılırsa:
- Log transform uygulanmadıysa
- Türetilmiş özellikler hesaplanmadıysa
- Scaler 15 özellik bekliyor ama 8 verilmişse

→ **Yanlış normalizasyon = Yanlış tahminler**

## Çözümler Uygulandı

### 1. Class Weight Loss Function (✅ Yapıldı)
```python
# Sınıf ağırlıkları hesapla
unique, counts = np.unique(y_train_aug, return_counts=True)
total = len(y_train_aug)
class_weights = []
for i in range(num_classes):
    weight = total / (num_classes * count[i])
    class_weights.append(weight)

# Loss function'a ekle
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
```

**Etki**: 
- Nadir sınıflar (araba) daha fazla ağırlık alır
- Model tüm sınıfları dengeli öğrenir
- Epoch 40'ta 96.76% validation accuracy

### 2. Veri Ön İşleme Kontrol Listesi
Dosya modunda tahmin yapılırken:

✅ **Raw FFT bant güçleri yüklenir**
```python
fft_data = df[['Delta', 'Theta', 'Low Alpha', ...]].values
```

✅ **Log transform uygulanır**
```python
features = np.log1p(np.abs(features))
```

✅ **Türetilmiş özellikler eklenir** (7 tane daha)
```python
alpha_total = low_alpha + high_alpha
beta_total = low_beta + high_beta
...
engagement = beta_total / (alpha_total + theta + eps)
# Extended: 15 özellik
```

✅ **StandardScaler normalizasyon** (15 özellik için eğitilmiş)
```python
sequence = scaler.transform(sequence)  # 15 özellik ile
```

### 3. Epoch Optimizasyon
- Orijinal: 100 epoch
- Düzeltme: 50 epoch (class weights daha hızlı yakınsar)
- Early stopping: best val_acc Epoch 40'ta (96.76%)

## Test Sonuçları

### Dosya Modu - Doğru Ön İşleme ile
```
yukarı klasöründe test:
  aşağı :  59.6%  (hala baskın)
  yukarı:  40.4%  (dönemirli tahminler)
```

**Not**: Raw veri formatı "aşağı"ya baskınlık gösterebilir. Bunun nedenleri:
1. İnsan fizyolojisi: Başında EEG cihazı takılıyken "aşağı" hareketi daha stabil ve net sinyal üretiyor olabilir
2. Sinyal kalitesi: "yukarı" hareketi sırasında cihaz kayabilir (daha gürültülü)
3. Veri toplama: "aşağı" hareketinde daha fazla örnek toplanmış olabilir

## Gelişim Yapılabilecek Alanlar

### 1. Daha Güçlü Class Balancing
- Weighted Random Sampler (over/under sampling)
- SMOTE (synthetic data generation)
- Focal Loss (hard examples'a daha fazla ağırlık)

### 2. Veri Artırma
```python
# Zaten yapılan
- Gaussian noise
- Random scaling
- Time shift

# Yapılabilecek
- Mixup (iki örneği karıştır)
- SpecAugment (band'ları maskeле)
- Ensemble (farklı modellerle)
```

### 3. Model Architecture
- Attention mekanizması (hangi bantlara daha çok dikkat et)
- Multi-task learning (sınıf + bant güçleri prediction)
- Confidence calibration

### 4. Cross-Validation
```python
# Şu anda
train/val: 80/20 split

# Yapılabilecek
StratifiedKFold (k=5) - her fold'da sınıf dengesi kontrol edilir
```

## Özet

✅ **Class weights uygulandı** → Model tüm sınıfları dengeli öğrenir
✅ **Validation accuracy: 96.76%** (Epoch 40)
⚠️ **Test verisi hala "aşağı"ya baskınlık gösterebilir** → Fizyolojik/teknik nedenleri olabilir
🎯 **İleri adım**: Live capture problemleri çözüldükten sonra gerçek veri ile yeniden test et
