#!/usr/bin/env python3
"""
Windows COM Port Test - MindWave veri kontrolü
Bu scripti Windows'ta çalıştırın!
"""

import serial
import serial.tools.list_ports
import time

def test_com_port(port, baudrate=57600, duration=10):
    """COM portunu test et ve ham veriyi göster"""
    
    print("="*60)
    print(f"🔍 {port} Test Ediliyor...")
    print("="*60)
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"✅ {port} açıldı")
        print(f"⏱️  {duration} saniye boyunca veri bekleniyor...\n")
        
        start_time = time.time()
        byte_count = 0
        
        while (time.time() - start_time) < duration:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                byte_count += len(data)
                
                # İlk 20 byte'ı göster
                hex_data = ' '.join([f'{b:02X}' for b in data[:20]])
                print(f"📦 {len(data)} byte: [{hex_data}...]")
                
                # MindWave sync byte (0xAA 0xAA)
                if len(data) >= 2 and data[0] == 0xAA and data[1] == 0xAA:
                    print("   ✅ MindWave paketi tespit edildi!")
            else:
                time.sleep(0.1)
        
        ser.close()
        
        print("\n" + "="*60)
        print(f"📊 Toplam: {byte_count} byte")
        
        if byte_count > 0:
            print("✅ VERİ GELİYOR! Bu port doğru.")
            return True
        else:
            print("❌ VERİ GELMİYOR!")
            return False
            
    except serial.SerialException as e:
        print(f"❌ Hata: {e}")
        return False
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 MindWave COM Port Tarayıcı (Windows)")
    print("="*60)
    print("\n⚠️  Bu scripti Windows'ta çalıştırın!\n")
    
    # Tüm COM portlarını listele
    ports = list(serial.tools.list_ports.comports())
    
    if not ports:
        print("❌ Hiç COM port bulunamadı!")
        print("💡 MindWave Bluetooth ile eşleştirilmiş mi?")
        exit(1)
    
    print(f"📋 Bulunan COM portları ({len(ports)} adet):\n")
    for i, port in enumerate(ports, 1):
        print(f"  {i}. {port.device} - {port.description}")
        if "bluetooth" in port.description.lower():
            print(f"     ⭐ Bluetooth - MindWave olabilir!")
    
    print("\n" + "="*60)
    
    # Kullanıcıdan port seç
    choice = input(f"\nTest edilecek portu seçin (1-{len(ports)}) veya Enter (hepsini test et): ").strip()
    
    print("\n⚠️  MindWave cihazının:")
    print("   - Açık olduğundan")
    print("   - Başınıza takıldığından")
    print("   - Kulak kıskacı ve alın sensörü temas ettiğinden")
    print("   emin olun!\n")
    input("Hazır olduğunuzda Enter'a basın...")
    
    if choice.isdigit() and 1 <= int(choice) <= len(ports):
        # Tek port test et
        port = ports[int(choice) - 1].device
        test_com_port(port, duration=10)
    else:
        # Hepsini test et
        print("\n🔄 Tüm portlar test ediliyor...\n")
        working_ports = []
        
        for port in ports:
            if "bluetooth" in port.description.lower():
                print(f"\n{'='*60}")
                result = test_com_port(port.device, duration=5)
                if result:
                    working_ports.append(port.device)
                time.sleep(1)
        
        if working_ports:
            print("\n" + "="*60)
            print("✅ ÇALIŞAN PORTLAR:")
            print("="*60)
            for port in working_ports:
                print(f"   → {port}")
            print("\n💡 windows_proxy_auto.py'de bu portu kullanın!")
        else:
            print("\n" + "="*60)
            print("❌ HİÇBİR PORTTAN VERİ GELMEDİ!")
            print("="*60)
            print("\n💡 Kontrol edin:")
            print("   1. MindWave açık mı? (Mavi LED)")
            print("   2. Başınıza takılı mı?")
            print("   3. Kulak kıskacı ve alın sensörü temas ediyor mu?")
            print("   4. Pil dolu mu?")
