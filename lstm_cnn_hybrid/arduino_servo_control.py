#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arduino Servo Control Script
=============================

Bu script, EEG tahmin sonuçlarına göre Arduino'ya bağlı servo motorları kontrol eder.

Bağımsız kullanım:
    python arduino_servo_control.py --port COM3 --test     # Test modu
    python arduino_servo_control.py --port /dev/ttyACM0    # Linux port
    
realtime_predict.py ile kullanım:
    python realtime_predict.py --thinkgear --model seq64 --arduino COM3
    python realtime_predict.py --port COM5 --arduino /dev/ttyACM0

Arduino Kodu:
    Arduino'da şu kodun yüklü olması gerekir:
    
    #include <Servo.h>
    
    Servo myServo;
    bool servoAttached = true;
    
    void setup() {
      Serial.begin(9600);
      myServo.attach(9);  // PWM pin
      myServo.write(90);  // Başlangıç: orta
    }
    
    void loop() {
      if (Serial.available() > 0) {
        char cmd = Serial.read();
        
        if (cmd == 'Y') {        // Yukarı
          if (!servoAttached) { myServo.attach(9); servoAttached = true; }
          myServo.write(180);
        }
        else if (cmd == 'A') {   // Aşağı
          if (!servoAttached) { myServo.attach(9); servoAttached = true; }
          myServo.write(0);
        }
        else if (cmd == 'R') {   // Araba (orta)
          if (!servoAttached) { myServo.attach(9); servoAttached = true; }
          myServo.write(90);
        }
        else if (cmd == 'S') {   // Stop (servo serbest)
          myServo.detach();
          servoAttached = false;
        }
      }
    }

Komutlar:
    'Y' -> Yukarı (servo 180°)
    'A' -> Aşağı (servo 0°)
    'R' -> Araba/Reset (servo 90°)
    'S' -> Stop (servo serbest bırak - motor durdur)
"""

import serial
import time
import argparse
import sys


class ArduinoController:
    """
    Arduino ile servo motor kontrolü.
    Tahmin sonucuna göre servo pozisyonunu değiştirir.
    
    Komutlar:
        b'Y' -> yukarı (servo yukarı pozisyon - 180°)
        b'A' -> aşağı (servo aşağı pozisyon - 0°)  
        b'R' -> araba (servo orta pozisyon - 90°)
        b'S' -> stop (servo durdur - detach, motor serbest)
    """
    
    def __init__(self, port, baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.connected = False
    
    def connect(self):
        """Arduino'ya seri port bağlantısı kur"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Arduino reset için bekle
            self.connected = True
            print(f"✅ Arduino bağlandı: {self.port} @ {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"❌ Arduino bağlantı hatası: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"❌ Arduino hatası: {e}")
            self.connected = False
            return False
    
    def send_command(self, label):
        """
        Tahmin etiketine göre Arduino'ya komut gönder.
        
        Args:
            label: Tahmin etiketi ('yukarı', 'aşağı', 'asagı', 'araba')
        
        Returns:
            True: Komut başarıyla gönderildi
            False: Gönderme başarısız
        """
        if not self.connected or self.serial_conn is None:
            return False
        
        try:
            label_lower = label.lower()
            
            if 'yukarı' in label_lower or 'yukari' in label_lower:
                self.serial_conn.write(b'Y')
                print(f"   🔼 Servo: YUKARI (180°)")
                return True
            elif 'aşağı' in label_lower or 'asagı' in label_lower or 'asagi' in label_lower:
                self.serial_conn.write(b'A')
                print(f"   🔽 Servo: AŞAĞI (0°)")
                return True
            elif 'araba' in label_lower:
                self.serial_conn.write(b'R')
                print(f"   🚗 Servo: ARABA/ORTA (90°)")
                return True
            elif 'stop' in label_lower or 'dur' in label_lower:
                self.serial_conn.write(b'S')
                print(f"   ⏹️ Servo: DURDURULDU (detach)")
                return True
            else:
                print(f"   ⚠️ Bilinmeyen etiket: {label}")
                return False
                
        except serial.SerialException as e:
            print(f"❌ Arduino yazma hatası: {e}")
            return False
    
    def send_raw(self, command):
        """
        Ham komut gönder (test için).
        
        Args:
            command: Tek karakter ('Y', 'A', 'R', 'S')
        """
        if not self.connected or self.serial_conn is None:
            print("❌ Arduino bağlı değil!")
            return False
        
        try:
            self.serial_conn.write(command.encode())
            print(f"   📤 Gönderildi: {command}")
            return True
        except serial.SerialException as e:
            print(f"❌ Yazma hatası: {e}")
            return False
    
    def stop_servo(self):
        """Servo motoru durdur (serbest bırak)"""
        if not self.connected or self.serial_conn is None:
            return False
        try:
            self.serial_conn.write(b'S')
            print("   ⏹️ Servo DURDURULDU (detach)")
            return True
        except serial.SerialException as e:
            print(f"❌ Arduino yazma hatası: {e}")
            return False
    
    def close(self):
        """Seri port bağlantısını kapat"""
        if self.serial_conn is not None:
            try:
                # Kapatmadan önce servo'yu durdur
                self.serial_conn.write(b'S')
                time.sleep(0.1)
                self.serial_conn.close()
                print("✅ Arduino bağlantısı kapatıldı (servo durduruldu)")
            except:
                pass
        self.connected = False


def list_ports():
    """Mevcut seri portları listele"""
    import serial.tools.list_ports
    
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ Hiç seri port bulunamadı!")
        return
    
    print("\n📋 Mevcut Seri Portlar:")
    print("=" * 50)
    for port in ports:
        print(f"   {port.device}")
        print(f"      Açıklama: {port.description}")
        if port.manufacturer:
            print(f"      Üretici: {port.manufacturer}")
        print()


def test_mode(controller):
    """Interaktif test modu"""
    print("\n" + "=" * 50)
    print("🧪 ARDUINO TEST MODU")
    print("=" * 50)
    print("Komutlar:")
    print("   y, yukari  -> Servo yukarı (180°)")
    print("   a, asagi   -> Servo aşağı (0°)")
    print("   r, araba   -> Servo orta (90°)")
    print("   s, stop    -> Servo durdur (detach)")
    print("   Y, A, R, S -> Ham komut gönder")
    print("   q, quit    -> Çıkış")
    print("=" * 50)
    print()
    
    while True:
        try:
            cmd = input(">> ").strip()
            
            if cmd.lower() in ['q', 'quit', 'exit', 'çık']:
                print("👋 Çıkış...")
                break
            elif cmd.lower() in ['y', 'yukari', 'yukarı']:
                controller.send_command('yukarı')
            elif cmd.lower() in ['a', 'asagi', 'aşağı']:
                controller.send_command('asagı')
            elif cmd.lower() in ['r', 'araba', 'reset']:
                controller.send_command('araba')
            elif cmd.lower() in ['s', 'stop', 'dur']:
                controller.stop_servo()
            elif cmd in ['Y', 'A', 'R', 'S']:
                controller.send_raw(cmd)
            elif cmd:
                print(f"   ⚠️ Bilinmeyen komut: {cmd}")
                print("   Kullanım: y/a/r/s veya Y/A/R/S veya q")
        
        except KeyboardInterrupt:
            print("\n👋 Çıkış...")
            break
        except EOFError:
            break


def demo_sequence(controller):
    """Demo sekansı - her pozisyonu test et"""
    print("\n" + "=" * 50)
    print("🎬 DEMO SEKVANSI")
    print("=" * 50)
    
    sequence = [
        ('yukarı', 2),
        ('araba', 1),
        ('asagı', 2),
        ('araba', 1),
        ('yukarı', 2),
        ('asagı', 2),
        ('araba', 1),
    ]
    
    for label, duration in sequence:
        print(f"\n🎯 {label.upper()}")
        controller.send_command(label)
        time.sleep(duration)
    
    print("\n✅ Demo tamamlandı!")


def main():
    parser = argparse.ArgumentParser(
        description='Arduino Servo Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  python arduino_servo_control.py --list-ports          # Portları listele
  python arduino_servo_control.py --port COM3 --test    # İnteraktif test
  python arduino_servo_control.py --port COM3 --demo    # Demo sekansı
  
  # Linux port örneği
  python arduino_servo_control.py --port /dev/ttyACM0 --test
        """
    )
    
    parser.add_argument('--port', metavar='PORT',
                       help='Arduino seri port (örn: COM3, /dev/ttyACM0)')
    parser.add_argument('--baud', type=int, default=9600,
                       help='Baud rate (varsayılan: 9600)')
    parser.add_argument('--list-ports', action='store_true',
                       help='Mevcut seri portları listele')
    parser.add_argument('--test', action='store_true',
                       help='İnteraktif test modu')
    parser.add_argument('--demo', action='store_true',
                       help='Demo sekansı çalıştır')
    
    args = parser.parse_args()
    
    # Port listele
    if args.list_ports:
        list_ports()
        return
    
    # Port gerekli
    if not args.port:
        print("❌ --port belirtmelisiniz!")
        print("   Örnek: python arduino_servo_control.py --port COM3 --test")
        print("   Portları görmek için: --list-ports")
        return
    
    # Bağlan
    controller = ArduinoController(args.port, args.baud)
    
    if not controller.connect():
        print("\n💡 İpucu:")
        print("   - Arduino'nun bağlı olduğundan emin olun")
        print("   - Doğru portu seçtiğinizden emin olun (--list-ports)")
        print("   - Arduino IDE seri monitörü kapalı olmalı")
        return
    
    try:
        if args.demo:
            demo_sequence(controller)
        elif args.test:
            test_mode(controller)
        else:
            # Varsayılan: test modu
            test_mode(controller)
    
    finally:
        controller.close()


if __name__ == "__main__":
    main()
