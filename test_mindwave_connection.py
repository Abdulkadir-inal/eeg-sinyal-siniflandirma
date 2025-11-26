#!/usr/bin/env python3
"""
MindWave Bağlantı ve Veri Akışı Test Scripti
Ham veriyi gösterir - sorun tespiti için
"""

import socket
import time
import sys

def test_connection(host='172.20.16.1', port=5555, timeout=30):
    """MindWave proxy'ye bağlan ve ham veriyi göster"""
    
    print("="*60)
    print("🔍 MindWave Bağlantı Testi")
    print("="*60)
    print(f"📡 Bağlanıyor: {host}:{port}")
    
    try:
        # Bağlantı kur
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))
        print("✅ Bağlantı başarılı!")
        print(f"⏱️  {timeout} saniye boyunca veri dinlenecek...")
        print("="*60)
        
        sock.settimeout(1)  # 1 saniye timeout
        
        start_time = time.time()
        byte_count = 0
        packet_count = 0
        last_data_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                data = sock.recv(1024)
                
                if data:
                    byte_count += len(data)
                    packet_count += 1
                    last_data_time = time.time()
                    
                    # İlk 20 byte'ı hex formatında göster
                    hex_data = ' '.join([f'{b:02X}' for b in data[:20]])
                    print(f"📦 Paket #{packet_count}: {len(data)} byte - [{hex_data}...]")
                    
                    # Sync byte kontrolü (0xAA 0xAA)
                    if len(data) >= 2 and data[0] == 0xAA and data[1] == 0xAA:
                        print("   ✅ MindWave sync byte'ı tespit edildi!")
                    
                else:
                    # Bağlantı kesildi
                    print("\n❌ Bağlantı sunucu tarafından kapatıldı")
                    break
                    
            except socket.timeout:
                # Timeout - veri yok
                elapsed = time.time() - last_data_time
                if elapsed > 5:
                    print(f"⚠️  Son {elapsed:.1f} saniyedir veri gelmiyor...")
                continue
            except Exception as e:
                print(f"\n❌ Veri okuma hatası: {e}")
                break
        
        print("\n" + "="*60)
        print("📊 Test Özeti:")
        print("="*60)
        print(f"⏱️  Süre: {time.time() - start_time:.1f} saniye")
        print(f"📦 Toplam Paket: {packet_count}")
        print(f"📊 Toplam Byte: {byte_count}")
        print(f"📈 Veri Hızı: {byte_count/(time.time()-start_time):.1f} byte/saniye")
        
        if packet_count == 0:
            print("\n❌ HİÇ VERİ GELMEDİ!")
            print("\n💡 Olası Nedenler:")
            print("   1. MindWave cihazı kapalı veya pil bitmiş")
            print("   2. MindWave başınıza takılı değil")
            print("   3. Kulak kıskacı ve alın sensörü teması yok")
            print("   4. Windows proxy MindWave'den veri alamıyor")
            print("   5. COM portu yanlış seçilmiş")
        elif packet_count < 10:
            print("\n⚠️  ÇOK AZ VERİ GELDİ!")
            print("   MindWave düzgün bağlı olmayabilir")
        else:
            print("\n✅ Veri akışı normal görünüyor!")
        
    except socket.timeout:
        print(f"❌ Bağlantı zaman aşımına uğradı ({host}:{port})")
        print("💡 Windows'ta proxy sunucusu çalışıyor mu?")
    except ConnectionRefusedError:
        print(f"❌ Bağlantı reddedildi ({host}:{port})")
        print("💡 Windows'ta proxy sunucusu çalışıyor mu?")
    except Exception as e:
        print(f"❌ Hata: {e}")
    finally:
        try:
            sock.close()
        except:
            pass
    
    print("="*60)


if __name__ == "__main__":
    print("\n🧪 MindWave Ham Veri Testi Başlıyor...")
    print("⚠️  Windows'ta proxy sunucusu çalışıyor olmalı!\n")
    
    # Kullanıcıdan IP almak isterseniz:
    if len(sys.argv) > 1:
        host = sys.argv[1]
    else:
        host = '172.20.16.1'
    
    test_connection(host=host, timeout=30)
