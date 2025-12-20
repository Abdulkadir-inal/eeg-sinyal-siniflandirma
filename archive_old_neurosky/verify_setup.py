#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dizin Yapısı Doğrulama Scripti
Kodlardaki dizin yapısının doğru kurulduğunu kontrol eder
"""

import os
import sys

def check_directory(path, description):
    """Dizinin varlığını kontrol et"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description:40s} : {path}")
    return exists

def check_file(path, description):
    """Dosyanın varlığını kontrol et"""
    exists = os.path.isfile(path)
    if exists:
        size = os.path.getsize(path)
        size_str = f"{size / (1024*1024):.1f} MB" if size > 1024*1024 else f"{size / 1024:.1f} KB"
        print(f"✅ {description:40s} : {size_str}")
    else:
        print(f"❌ {description:40s} : Bulunamadı")
    return exists

def count_csv_files(directory):
    """Dizindeki CSV dosyalarını say"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith('.csv')])

def main():
    print("\n" + "="*70)
    print("🔍 DİZİN YAPISI DOĞRULAMA")
    print("="*70)
    
    # Ana dizinler
    print("\n📁 ANA DİZİNLER:")
    print("-"*70)
    base_dir = "/home/kadir/sanal-makine/python"
    proje_dir = os.path.join(base_dir, "proje")
    proje_veri_dir = os.path.join(base_dir, "proje-veri")
    
    all_ok = True
    all_ok &= check_directory(base_dir, "Base dizin")
    all_ok &= check_directory(proje_dir, "Proje dizini (DATA_DIR)")
    all_ok &= check_directory(proje_veri_dir, "Veri dizini")
    
    # Veri alt dizinleri
    print("\n📂 VERİ ALT DİZİNLERİ:")
    print("-"*70)
    for subdir in ["araba", "yukarı", "asagı"]:
        path = os.path.join(proje_veri_dir, subdir)
        exists = check_directory(path, f"Veri/{subdir}")
        all_ok &= exists
        
        if exists:
            csv_count = count_csv_files(path)
            print(f"   └─ CSV dosyası sayısı: {csv_count}")
    
    # Kritik dosyalar
    print("\n📄 KRİTİK DOSYALAR:")
    print("-"*70)
    all_ok &= check_file(os.path.join(proje_dir, "best_model.pth"), "En iyi model")
    all_ok &= check_file(os.path.join(proje_dir, "final_model.pth"), "Son model")
    all_ok &= check_file(os.path.join(proje_dir, "label_map.json"), "Etiket haritası")
    
    # İşlenmiş veri dosyaları (opsiyonel)
    print("\n📊 İŞLENMİŞ VERİ DOSYALARI (Opsiyonel):")
    print("-"*70)
    x_exists = check_file(os.path.join(proje_dir, "X.npy"), "Özellik matrisi (X.npy)")
    y_exists = check_file(os.path.join(proje_dir, "y.npy"), "Etiketler (y.npy)")
    
    if not x_exists or not y_exists:
        print("\n💡 NOT: X.npy ve y.npy yoksa, data_preprocess.py çalıştırın")
    
    # Python script kontrolleri
    print("\n🐍 PYTHON SCRIPTLER:")
    print("-"*70)
    scripts = [
        "train_model.py",
        "predict.py", 
        "data_preprocess.py",
        "realtime_mindwave_predict.py"
    ]
    
    current_dir = "/home/kadir/eeg-sinyal-siniflandirma"
    for script in scripts:
        check_file(os.path.join(current_dir, script), script)
    
    # Özet
    print("\n" + "="*70)
    if all_ok:
        print("✅ TÜM DİZİNLER ve DOSYALAR HAZIR!")
        print("="*70)
        print("\n🎯 SONRAKI ADIMLAR:")
        print("-"*70)
        print("1. Veri işleme (eğer X.npy/y.npy yoksa):")
        print("   cd /home/kadir/eeg-sinyal-siniflandirma")
        print("   python3 data_preprocess.py")
        print()
        print("2. Model eğitimi (yeni veri ile):")
        print("   python3 train_model.py")
        print()
        print("3. Tahmin:")
        print("   python3 predict.py")
        print()
        print("4. Canlı MindWave:")
        print("   python3 realtime_mindwave_predict.py")
    else:
        print("⚠️  BAZI DİZİNLER VEYA DOSYALAR EKSİK!")
        print("="*70)
        print("\nYukarıdaki ❌ işaretli öğeleri kontrol edin.")
    
    print("="*70 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
