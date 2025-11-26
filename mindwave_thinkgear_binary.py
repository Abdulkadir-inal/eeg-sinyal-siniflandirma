#!/usr/bin/env python3
"""
ThinkGear Connector - Binary Protokol Parser
MindWave'den ham binary veri okur ve parse eder
"""

import socket
import struct
import time

class ThinkGearBinary:
    """ThinkGear binary protokol parser"""
    
    # ThinkGear paket tipleri
    POOR_SIGNAL = 0x02
    ATTENTION = 0x04
    MEDITATION = 0x05
    RAW_VALUE = 0x80
    EEG_POWER = 0x83
    
    def __init__(self, host='localhost', port=13854):
        self.host = host
        self.port = port
        self.sock = None
        
    def connect(self):
        """ThinkGear Connector'a bağlan"""
        try:
            print(f"🔵 ThinkGear Connector'a bağlanıyor: {self.host}:{self.port}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.host, self.port))
            
            print("✅ ThinkGear Connector'a bağlandı!")
            print("📡 Binary protokol dinleniyor...")
            return True
            
        except ConnectionRefusedError:
            print(f"❌ Bağlantı reddedildi!")
            print("\n💡 ThinkGear Connector çalışmıyor!")
            print("   1. ThinkGear Connector'ı başlatın")
            print("   2. MindWave'i bağlayın")
            print("   3. Bu scripti tekrar çalıştırın")
            return False
        except Exception as e:
            print(f"❌ Hata: {e}")
            return False
    
    def parse_packet(self, data):
        """ThinkGear binary paketini parse et"""
        result = {}
        i = 0
        
        while i < len(data):
            # Sync bytes kontrol (0xAA 0xAA)
            if i + 1 < len(data) and data[i] == 0xAA and data[i+1] == 0xAA:
                i += 2
                
                # Payload length
                if i < len(data):
                    payload_length = data[i]
                    i += 1
                    
                    if payload_length > 169:  # Invalid
                        continue
                    
                    # Payload parse
                    payload_end = i + payload_length
                    while i < payload_end and i < len(data):
                        code = data[i]
                        i += 1
                        
                        if code == self.POOR_SIGNAL and i < len(data):
                            result['signal_quality'] = 200 - data[i]
                            result['poor_signal'] = data[i]
                            i += 1
                            
                        elif code == self.ATTENTION and i < len(data):
                            result['attention'] = data[i]
                            i += 1
                            
                        elif code == self.MEDITATION and i < len(data):
                            result['meditation'] = data[i]
                            i += 1
                            
                        elif code == self.RAW_VALUE and i + 1 < len(data):
                            raw_value = struct.unpack('>h', bytes([data[i], data[i+1]]))[0]
                            result['raw'] = raw_value
                            i += 2
                            
                        elif code == self.EEG_POWER and i + 23 < len(data):
                            # 8 x 3-byte values
                            eeg = {}
                            eeg['delta'] = (data[i] << 16) | (data[i+1] << 8) | data[i+2]
                            eeg['theta'] = (data[i+3] << 16) | (data[i+4] << 8) | data[i+5]
                            eeg['low_alpha'] = (data[i+6] << 16) | (data[i+7] << 8) | data[i+8]
                            eeg['high_alpha'] = (data[i+9] << 16) | (data[i+10] << 8) | data[i+11]
                            eeg['low_beta'] = (data[i+12] << 16) | (data[i+13] << 8) | data[i+14]
                            eeg['high_beta'] = (data[i+15] << 16) | (data[i+16] << 8) | data[i+17]
                            eeg['low_gamma'] = (data[i+18] << 16) | (data[i+19] << 8) | data[i+20]
                            eeg['mid_gamma'] = (data[i+21] << 16) | (data[i+22] << 8) | data[i+23]
                            result['eeg'] = eeg
                            i += 24
                        else:
                            break
                    
                    # Checksum (skip)
                    i += 1
            else:
                i += 1
        
        return result
    
    def read_data(self, duration=60):
        """Veri oku ve göster"""
        if not self.sock:
            print("❌ Bağlantı yok!")
            return
        
        print(f"\n📊 {duration} saniye veri okunuyor...\n")
        print("="*60)
        
        start_time = time.time()
        packet_count = 0
        eeg_count = 0
        
        try:
            self.sock.settimeout(1)
            
            while (time.time() - start_time) < duration:
                try:
                    data = self.sock.recv(1024)
                    
                    if not data:
                        print("❌ Bağlantı kesildi")
                        break
                    
                    # Parse
                    result = self.parse_packet(data)
                    
                    if result:
                        packet_count += 1
                        
                        # Sinyal kalitesi
                        if 'signal_quality' in result:
                            quality = result['signal_quality']
                            poor = result['poor_signal']
                            
                            if poor == 0:
                                status = "✅ Mükemmel"
                            elif poor < 50:
                                status = f"⚠️  Orta (Zayıf: {poor})"
                            else:
                                status = f"❌ Kötü (Zayıf: {poor})"
                            
                            print(f"📶 Sinyal Kalitesi: {status}")
                        
                        # eSense değerleri
                        if 'attention' in result or 'meditation' in result:
                            print(f"🧠 eSense:")
                            if 'attention' in result:
                                print(f"   Dikkat: {result['attention']}/100")
                            if 'meditation' in result:
                                print(f"   Meditasyon: {result['meditation']}/100")
                        
                        # EEG güç değerleri
                        if 'eeg' in result:
                            eeg_count += 1
                            eeg = result['eeg']
                            print(f"\n📡 EEG Paketi #{eeg_count}:")
                            print(f"   Delta:      {eeg['delta']:>8}")
                            print(f"   Theta:      {eeg['theta']:>8}")
                            print(f"   Low Alpha:  {eeg['low_alpha']:>8}")
                            print(f"   High Alpha: {eeg['high_alpha']:>8}")
                            print(f"   Low Beta:   {eeg['low_beta']:>8}")
                            print(f"   High Beta:  {eeg['high_beta']:>8}")
                            print(f"   Low Gamma:  {eeg['low_gamma']:>8}")
                            print(f"   Mid Gamma:  {eeg['mid_gamma']:>8}")
                        
                        if result:
                            print("-" * 60)
                
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"❌ Parse hatası: {e}")
                    continue
        
        except KeyboardInterrupt:
            print("\n⛔ Kullanıcı tarafından durduruldu")
        except Exception as e:
            print(f"\n❌ Hata: {e}")
        
        print("\n" + "="*60)
        print(f"📊 İstatistikler:")
        print(f"   Toplam paket: {packet_count}")
        print(f"   EEG paketi: {eeg_count}")
        print("="*60)
        
        if packet_count == 0:
            print("\n❌ HİÇ VERİ GELMEDİ!")
            print("💡 MindWave'i ThinkGear Connector'da bağlayın!")
        elif eeg_count > 0:
            print("\n✅ MindWave düzgün çalışıyor!")
        
    def close(self):
        """Bağlantıyı kapat"""
        if self.sock:
            self.sock.close()
        print("\n✅ Bağlantı kapatıldı")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 ThinkGear Connector - Binary Parser")
    print("="*60)
    print("\n⚠️  ThinkGear Connector çalışıyor olmalı!")
    print("   Windows: C:\\Program Files (x86)\\ThinkGear Connector")
    print("   veya sistem tray'de ThinkGear ikonu\n")
    
    connector = ThinkGearBinary()
    
    if connector.connect():
        print("\n💡 MindWave'i başınıza takın:")
        print("   - Kulak kıskacı kulak memesine")
        print("   - Alın sensörü alnınıza")
        print("   - 5 saniye bekleyin (sinyal kalitesi düzelsin)\n")
        
        input("Hazır olduğunuzda Enter'a basın...")
        
        try:
            connector.read_data(duration=60)
        finally:
            connector.close()
