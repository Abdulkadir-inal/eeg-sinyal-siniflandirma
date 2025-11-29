#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThinkGear → WSL2 Proxy
======================

ThinkGear Connector'dan gelen veriyi WSL2'ye yönlendirir.

Windows'ta çalıştırın:
    python thinkgear_proxy.py

WSL2'de:
    python wsl_realtime_predict.py
"""

import socket
import threading
import sys
import time

class ThinkGearProxy:
    """ThinkGear Connector verisini WSL2'ye yönlendirir"""
    
    def __init__(self, thinkgear_host='127.0.0.1', thinkgear_port=13854, proxy_port=5555):
        self.thinkgear_host = thinkgear_host
        self.thinkgear_port = thinkgear_port
        self.proxy_port = proxy_port
        
        self.thinkgear_sock = None
        self.server_sock = None
        self.client_sock = None
        self.running = False
    
    def connect_thinkgear(self):
        """ThinkGear Connector'a bağlan"""
        try:
            print(f"🔵 ThinkGear Connector'a bağlanılıyor: {self.thinkgear_host}:{self.thinkgear_port}")
            self.thinkgear_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.thinkgear_sock.settimeout(5)
            self.thinkgear_sock.connect((self.thinkgear_host, self.thinkgear_port))
            
            # JSON format iste
            self.thinkgear_sock.send(b'{"enableRawOutput": false, "format": "Json"}\n')
            
            # TCP optimizasyonları
            self.thinkgear_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            print("✅ ThinkGear Connector'a bağlandı!")
            self.thinkgear_sock.settimeout(0.1)  # 100ms timeout (daha hızlı)
            return True
            
        except ConnectionRefusedError:
            print(f"❌ ThinkGear Connector çalışmıyor!")
            print("   1. ThinkGear Connector'ı başlatın")
            print("   2. MindWave cihazını bağlayın")
            return False
        except Exception as e:
            print(f"❌ Bağlantı hatası: {e}")
            return False
    
    def start_server(self):
        """WSL2 için TCP sunucu başlat"""
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind(('0.0.0.0', self.proxy_port))
            self.server_sock.listen(1)
            
            print(f"🌐 Proxy sunucu başlatıldı: 0.0.0.0:{self.proxy_port}")
            print(f"💡 WSL2'den bağlanmak için gateway IP'yi kullanın")
            return True
            
        except Exception as e:
            print(f"❌ Sunucu başlatılamadı: {e}")
            return False
    
    def run(self):
        """Ana döngü"""
        print("\n" + "=" * 60)
        print("🔄 ThinkGear → WSL2 Proxy")
        print("=" * 60)
        
        # ThinkGear'a bağlan
        if not self.connect_thinkgear():
            return
        
        # Sunucu başlat
        if not self.start_server():
            return
        
        print("\n⏳ WSL2 bağlantısı bekleniyor...")
        print("   WSL2'de çalıştırın: python wsl_realtime_predict.py")
        print("-" * 60)
        
        try:
            # İstemci bekle
            self.client_sock, addr = self.server_sock.accept()
            print(f"\n✅ WSL2 bağlandı: {addr}")
            
            # TCP optimizasyonları - hızlı veri iletimi
            self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            self.running = True
            
            # Veri aktarımı
            print("📊 Veri aktarımı başladı... (Ctrl+C ile durdurun)")
            print("-" * 60)
            
            byte_count = 0
            packet_count = 0
            start_time = time.time()
            
            while self.running:
                try:
                    # ThinkGear'dan oku (daha büyük buffer)
                    data = self.thinkgear_sock.recv(8192)
                    
                    if data:
                        # WSL2'ye hemen gönder
                        self.client_sock.sendall(data)
                        byte_count += len(data)
                        packet_count += 1
                        
                        # İstatistik (her 10 pakette bir güncelle - daha az overhead)
                        if packet_count % 10 == 0:
                            elapsed = time.time() - start_time
                            rate = byte_count / elapsed if elapsed > 0 else 0
                            print(f"\r📦 Paket: {packet_count} | Byte: {byte_count} | Hız: {rate:.0f} B/s   ", end='', flush=True)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"\n❌ Veri aktarım hatası: {e}")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⛔ Kullanıcı tarafından durduruldu")
        finally:
            self.stop()
    
    def stop(self):
        """Bağlantıları kapat"""
        self.running = False
        
        if self.client_sock:
            try:
                self.client_sock.close()
            except:
                pass
        
        if self.server_sock:
            try:
                self.server_sock.close()
            except:
                pass
        
        if self.thinkgear_sock:
            try:
                self.thinkgear_sock.close()
            except:
                pass
        
        print("✅ Proxy kapatıldı")


def main():
    proxy = ThinkGearProxy()
    proxy.run()


if __name__ == "__main__":
    main()
