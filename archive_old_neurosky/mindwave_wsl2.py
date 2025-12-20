#!/usr/bin/env python3
"""
MindWave WSL2 Bağlantı Modülü - ThinkGear Binary Parser
Windows'ta çalışan proxy sunucusundan gelen ham ThinkGear 
binary verisini parse eder.

Kullanım:
1. Windows'ta windows_proxy_auto.py çalıştırın
2. WSL2'de bu modülü import edin
"""

import socket
import struct
import time
import subprocess


class MindWaveWSL2:
    """WSL2'den Windows proxy sunucusuna bağlanarak MindWave verisi okur"""
    
    # ThinkGear paket tipleri
    POOR_SIGNAL = 0x02
    ATTENTION = 0x04
    MEDITATION = 0x05
    RAW_VALUE = 0x80
    EEG_POWER = 0x83
    
    def __init__(self, host=None, port=5555):
        """
        Args:
            host: Windows proxy IP adresi (None = otomatik tespit)
            port: TCP port numarası
        """
        if host is None:
            host = self._find_windows_ip()
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.buffer = b""
        
        # Son okunan veriler
        self.last_data = {
            'delta': 0,
            'theta': 0,
            'lowAlpha': 0,
            'highAlpha': 0,
            'lowBeta': 0,
            'highBeta': 0,
            'lowGamma': 0,
            'highGamma': 0,
            'attention': 0,
            'meditation': 0,
            'poorSignal': 200,
            'timestamp': 0
        }
    
    def _find_windows_ip(self):
        """WSL2'nin Windows gateway IP'sini otomatik bul"""
        try:
            result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'default' in line:
                    gateway = line.split()[2]
                    return gateway
        except:
            pass
        return '172.31.240.1'  # Varsayılan
        
    def connect(self):
        """Proxy sunucusuna bağlan"""
        try:
            print(f"🔵 Windows proxy'ye bağlanılıyor: {self.host}:{self.port}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(2)
            self.connected = True
            print("✅ Windows proxy'ye bağlandı!")
            print("📡 ThinkGear binary protokol dinleniyor...")
            return True
            
        except ConnectionRefusedError:
            print(f"❌ Bağlantı reddedildi: {self.host}:{self.port}")
            print("\n💡 Windows'ta proxy sunucusunu başlatın:")
            print("   python windows_proxy_auto.py")
            return False
        except socket.timeout:
            print(f"❌ Bağlantı zaman aşımı")
            return False
        except Exception as e:
            print(f"❌ Bağlantı hatası: {e}")
            return False
    
    def disconnect(self):
        """Bağlantıyı kapat"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.connected = False
        print("🔌 Bağlantı kapatıldı")
    
    def _parse_thinkgear_packet(self, data):
        """
        ThinkGear binary paketini parse et
        
        Returns:
            dict: Parse edilmiş veriler veya None
        """
        result = {}
        i = 0
        
        while i < len(data) - 3:
            # Sync bytes kontrol (0xAA 0xAA)
            if data[i] == 0xAA and data[i+1] == 0xAA:
                i += 2
                
                # Payload length
                payload_length = data[i]
                i += 1
                
                if payload_length > 169:  # Invalid
                    continue
                
                # Payload parse
                payload_end = min(i + payload_length, len(data) - 1)
                
                while i < payload_end:
                    if i >= len(data):
                        break
                        
                    code = data[i]
                    i += 1
                    
                    if code == self.POOR_SIGNAL:
                        if i < len(data):
                            result['poorSignal'] = data[i]
                            i += 1
                            
                    elif code == self.ATTENTION:
                        if i < len(data):
                            result['attention'] = data[i]
                            i += 1
                            
                    elif code == self.MEDITATION:
                        if i < len(data):
                            result['meditation'] = data[i]
                            i += 1
                            
                    elif code == self.RAW_VALUE:
                        if i + 1 < len(data):
                            i += 2  # Skip raw value (we don't need it)
                            
                    elif code == self.EEG_POWER:
                        if i + 23 < len(data):
                            # 8 x 3-byte values (big-endian)
                            result['delta'] = (data[i] << 16) | (data[i+1] << 8) | data[i+2]
                            result['theta'] = (data[i+3] << 16) | (data[i+4] << 8) | data[i+5]
                            result['lowAlpha'] = (data[i+6] << 16) | (data[i+7] << 8) | data[i+8]
                            result['highAlpha'] = (data[i+9] << 16) | (data[i+10] << 8) | data[i+11]
                            result['lowBeta'] = (data[i+12] << 16) | (data[i+13] << 8) | data[i+14]
                            result['highBeta'] = (data[i+15] << 16) | (data[i+16] << 8) | data[i+17]
                            result['lowGamma'] = (data[i+18] << 16) | (data[i+19] << 8) | data[i+20]
                            result['highGamma'] = (data[i+21] << 16) | (data[i+22] << 8) | data[i+23]
                            i += 24
                        else:
                            break
                    elif code >= 0x80:
                        # Extended code - length byte
                        if i < len(data):
                            extended_length = data[i]
                            i += 1 + extended_length
                    else:
                        # Unknown single-byte code
                        if i < len(data):
                            i += 1
                
                # Checksum skip
                i += 1
                
                if result:
                    result['timestamp'] = time.time()
                    return result
            else:
                i += 1
        
        return None
    
    def read_data(self):
        """
        Proxy'den EEG verisi oku ve parse et
        
        Returns:
            dict: EEG verileri içeren dictionary veya None
        """
        if not self.connected:
            return None
            
        try:
            # Veri oku
            data = self.sock.recv(1024)
            if not data:
                return None
            
            # Buffer'a ekle
            self.buffer += data
            
            # Parse et
            result = self._parse_thinkgear_packet(self.buffer)
            
            if result:
                # Buffer'ı temizle (sync byte'lardan sonrası)
                # Son 10 byte'ı sakla (kısmi paket olabilir)
                if len(self.buffer) > 100:
                    self.buffer = self.buffer[-50:]
                
                # Son veriyi güncelle
                for key, value in result.items():
                    if key in self.last_data:
                        self.last_data[key] = value
                
                return self.last_data.copy()
            
            # Buffer çok büyükse temizle
            if len(self.buffer) > 1024:
                self.buffer = self.buffer[-100:]
            
            return None
            
        except socket.timeout:
            return None
        except Exception as e:
            print(f"⚠️ Veri okuma hatası: {e}")
            return None
    
    def get_eeg_features(self):
        """
        EEG özelliklerini döndür
        
        Returns:
            dict: EEG özellikleri veya None
        """
        return self.read_data()
    
    def is_signal_good(self, data=None):
        """Sinyal kalitesini kontrol et (0=en iyi, 200=en kötü)"""
        if data is None:
            data = self.last_data
        return data.get('poorSignal', 200) < 50
    
    def __enter__(self):
        """Context manager desteği"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager desteği"""
        self.disconnect()
        return False


def test_connection():
    """Bağlantı testi"""
    print("=" * 60)
    print("🧠 MindWave WSL2 Bağlantı Testi")
    print("=" * 60)
    
    mindwave = MindWaveWSL2()  # IP otomatik bulunacak
    print(f"🔍 Tespit edilen IP: {mindwave.host}")
    
    if mindwave.connect():
        print("\n📡 Veri bekleniyor (30 saniye)...")
        print("💡 MindWave'i başınıza takın!")
        print("-" * 60)
        
        start_time = time.time()
        eeg_count = 0
        
        while (time.time() - start_time) < 30:
            data = mindwave.read_data()
            
            if data and 'delta' in data and data['delta'] > 0:
                eeg_count += 1
                signal_status = "✅ İyi" if data['poorSignal'] < 50 else f"⚠️ Zayıf ({data['poorSignal']})"
                
                print(f"\n📦 EEG Paketi #{eeg_count}:")
                print(f"   Sinyal: {signal_status}")
                print(f"   Delta:     {data['delta']:>8}")
                print(f"   Theta:     {data['theta']:>8}")
                print(f"   Alpha:     {data['lowAlpha'] + data['highAlpha']:>8}")
                print(f"   Beta:      {data['lowBeta'] + data['highBeta']:>8}")
                print(f"   Gamma:     {data['lowGamma'] + data['highGamma']:>8}")
                print(f"   Attention: {data.get('attention', 0)}")
                print(f"   Meditation:{data.get('meditation', 0)}")
            
            time.sleep(0.1)
        
        mindwave.disconnect()
        
        print("\n" + "=" * 60)
        if eeg_count > 0:
            print(f"✅ Test başarılı! {eeg_count} EEG paketi alındı.")
        else:
            print("❌ EEG paketi alınamadı!")
            print("\n💡 Kontrol edin:")
            print("   1. MindWave cihazı açık mı?")
            print("   2. MindWave başınızda takılı mı?")
            print("   3. Kulak kıskacı ve alın sensörü temas ediyor mu?")
        print("=" * 60)
    else:
        print("\n❌ Bağlantı başarısız!")
        print("\n💡 Çözüm:")
        print("   1. Windows'ta: python windows_proxy_auto.py")
        print("   2. MindWave cihazını Bluetooth ile eşleştirin")
        print("   3. Bu testi tekrar çalıştırın")


if __name__ == "__main__":
    test_connection()
