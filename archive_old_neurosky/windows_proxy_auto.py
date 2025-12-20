#!/usr/bin/env python3
"""
MindWave Mobile 2 - TCP/IP Proxy Sunucu (Windows'ta çalışır)
Otomatik COM port tespiti ile - Kullanıcı girişi gerektirmez
"""

import serial
import serial.tools.list_ports
import socket
import threading
import sys
import time

class SerialToTCPProxy:
    """COM portunu TCP/IP'ye yönlendirir"""
    
    def __init__(self, com_port='COM3', tcp_port=5555, baudrate=57600):
        self.com_port = com_port
        self.tcp_port = tcp_port
        self.baudrate = baudrate
        self.ser = None
        self.server_socket = None
        self.client_socket = None
        self.running = False
        
    def start(self):
        """Proxy sunucusunu başlat"""
        try:
            # Seri port aç
            print(f"📡 {self.com_port} portu açılıyor...")
            print(f"⚠️  MindWave başka bir uygulama tarafından kullanılıyorsa bağlantı kopar!")
            self.ser = serial.Serial(
                self.com_port, 
                self.baudrate, 
                timeout=0.1,
                exclusive=True  # Exclusive access - başka uygulama erişemez
            )
            print(f"✅ {self.com_port} açıldı (baudrate: {self.baudrate})")
            
            # TCP sunucu oluştur
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.tcp_port))
            self.server_socket.listen(1)
            
            print(f"🌐 TCP sunucu başlatıldı: 0.0.0.0:{self.tcp_port}")
            print(f"💡 WSL2'den bağlanmak için: host.docker.internal:{self.tcp_port}")
            print("⏳ Bağlantı bekleniyor...\n")
            
            # İstemci bekle
            self.client_socket, addr = self.server_socket.accept()
            print(f"✅ İstemci bağlandı: {addr}\n")
            
            self.running = True
            
            # İki yönlü veri aktarımı
            serial_thread = threading.Thread(target=self.serial_to_tcp, daemon=True)
            tcp_thread = threading.Thread(target=self.tcp_to_serial, daemon=True)
            
            serial_thread.start()
            tcp_thread.start()
            
            # Ana thread bekle
            try:
                print("📊 Veri aktarımı başladı... (Durdurmak için Ctrl+C)")
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⛔ Kullanıcı tarafından durduruldu")
                
        except serial.SerialException as e:
            print(f"\n❌ Seri Port Hatası: {e}")
            print(f"\n💡 MindWave bağlantısı koptu! Olası nedenler:")
            print(f"   1. {self.com_port} başka bir program tarafından kullanılıyor")
            print(f"   2. MindWave cihazının pili bitti veya kapatıldı")
            print(f"   3. Bluetooth bağlantısı kesintiye uğradı")
            print(f"   4. COM portu Windows tarafından serbest bırakılmadı")
            print(f"\n🔄 Çözüm önerileri:")
            print(f"   - MindWave'i Bluetooth ayarlarından disconnect/reconnect yapın")
            print(f"   - ThinkGear Connector gibi uygulamalar kapalı olmalı")
            print(f"   - Device Manager'da COM portunu Disable/Enable deneyin")
        except Exception as e:
            print(f"❌ Hata: {e}")
        finally:
            self.stop()
            
    def serial_to_tcp(self):
        """Seri porttan TCP'ye veri aktar"""
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    if data and self.client_socket:
                        self.client_socket.sendall(data)
                        print(f"→ TCP: {len(data)} byte", end='\r')
            except Exception as e:
                print(f"\n❌ Serial→TCP hatası: {e}")
                self.running = False
                
    def tcp_to_serial(self):
        """TCP'den seri porta veri aktar"""
        while self.running:
            try:
                if self.client_socket:
                    data = self.client_socket.recv(1024)
                    if data:
                        self.ser.write(data)
                        print(f"← Serial: {len(data)} byte", end='\r')
                    else:
                        print("\n🔌 İstemci bağlantıyı kesti")
                        self.running = False
            except Exception as e:
                print(f"\n❌ TCP→Serial hatası: {e}")
                self.running = False
                
    def stop(self):
        """Sunucuyu durdur"""
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except:
                pass
        print("\n✅ Proxy sunucu kapatıldı")


def find_mindwave_port():
    """MindWave cihazını otomatik bul"""
    ports = list(serial.tools.list_ports.comports())
    
    print(f"📋 Bulunan COM portları ({len(ports)} adet):")
    for i, port in enumerate(ports, 1):
        print(f"  {i}. {port.device} - {port.description}")
        # MindWave genelde "Standard Serial over Bluetooth" olarak görünür
        if "bluetooth" in port.description.lower() or "mindwave" in port.description.lower():
            print(f"     ⭐ MindWave olabilir!")
    
    return ports


if __name__ == "__main__":
    print("=" * 60)
    print("🔄 MindWave COM Port → TCP/IP Proxy (Otomatik)")
    print("=" * 60)
    
    # COM portları listele
    ports = find_mindwave_port()
    
    if not ports:
        print("\n❌ Hiç COM port bulunamadı!")
        print("💡 MindWave Bluetooth ile eşleştirilmiş mi kontrol edin")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    
    # Kullanıcıdan seçim al veya otomatik seç
    if len(ports) == 1:
        # Tek port varsa otomatik seç
        selected_port = ports[0].device
        print(f"✅ Otomatik seçildi: {selected_port}")
    else:
        # Birden fazla port varsa sor
        choice = input(f"\nCOM portu seçin (1-{len(ports)}) veya Enter (otomatik COM3): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(ports):
            selected_port = ports[int(choice) - 1].device
        else:
            selected_port = "COM3"  # Varsayılan
        print(f"✅ Seçildi: {selected_port}")
    
    # TCP port
    tcp_port_input = input("TCP portu (varsayılan: 5555): ").strip()
    tcp_port = int(tcp_port_input) if tcp_port_input else 5555
    
    print("\n" + "=" * 60)
    print(f"🚀 Başlatılıyor: {selected_port} → TCP:{tcp_port}")
    print("=" * 60 + "\n")
    
    # Proxy başlat
    proxy = SerialToTCPProxy(com_port=selected_port, tcp_port=tcp_port)
    proxy.start()
