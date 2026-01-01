#!/bin/bash
echo "=== كاشف كاميرات التجسس اللاسلكية ==="
echo "المسح يبدأ..."

# مسح متعدد النطاقات
frequencies=("2400" "1200" "5800")
band_names=("2.4GHz" "1.2GHz" "5.8GHz")

for i in ${!frequencies[@]}; do
    echo "المسح في نطاق ${band_names[$i]}..."
    rtl_power -f ${frequencies[$i]}M:$((frequencies[$i]+100))M:50k \
              -g 50 -i 30s scan_${band_names[$i]}.csv
    
    # تحليل أولي
    echo "تحليل النتائج..."
    python3 << EOF
import pandas as pd
import numpy as np
data = pd.read_csv('scan_${band_names[$i]}.csv', header=None)
if len(data) > 0:
    signal_strength = data.iloc[:, 2].mean()
    if signal_strength > -60:  # إشارة قوية
        print(f"⚠️  إشارة قوية في نطاق {band_names[$i]}: {signal_strength} dB")
        print("احتمال وجود كاميرا لاسلكية!")
EOF
done

echo "المسح اكتمل. تحقق من الملفات:"
ls -la scan_*.csv
