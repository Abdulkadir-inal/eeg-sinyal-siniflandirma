#!/bin/bash
# MindWave Canlı Sınıflandırma - Hızlı Başlangıç

echo "🧠 MindWave Canlı EEG Sınıflandırma"
echo "===================================="
echo ""

# Proje dizinine git
cd "$(dirname "$0")"

# Model kontrolü
if [ ! -f "best_model.pth" ]; then
    echo "❌ Model dosyası bulunamadı!"
    echo "   Önce modeli eğitin: python3 train_model.py"
    exit 1
fi

# Virtual environment aktif mi?
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔵 Virtual environment aktif edilecek..."
    source ../../.venv/bin/activate
fi

echo "✅ Hazırlıklar tamamlandı"
echo ""
echo "📋 KULLANIM:"
echo "   • Varsayılan ayarlar:"
echo "     python3 realtime_mindwave_predict.py"
echo ""
echo "   • Özel IP adresi:"
echo "     python3 realtime_mindwave_predict.py --host 192.168.1.100"
echo ""
echo "   • Tahmin sıklığını ayarla (2 saniyede bir):"
echo "     python3 realtime_mindwave_predict.py --interval 2.0"
echo ""
echo "   • Minimum sinyal kalitesi (0=mükemmel, 200=çok kötü):"
echo "     python3 realtime_mindwave_predict.py --min-quality 30"
echo ""
echo "💡 ÖNEMLİ:"
echo "   1. Windows'ta proxy sunucusunu başlatın:"
echo "      python windows_proxy.py"
echo ""
echo "   2. MindWave cihazını açın ve bilgisayara bağlayın"
echo ""
echo "   3. Ctrl+C ile durdurun"
echo ""
echo "🚀 Başlatılıyor..."
echo ""

# Scripti çalıştır
python3 realtime_mindwave_predict.py "$@"
