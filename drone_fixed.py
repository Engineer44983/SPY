#!/usr/bin/env python3
"""
إطار عمل تعليمي للكشف عن إشارات RF
تحذير: نظام تعليمي للتدريب فقط
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from enum import Enum
import warnings
import time
import sys

warnings.filterwarnings('ignore')

class SignalType(Enum):
    """أنواع الإشارات المعروفة"""
    UNKNOWN = "unknown"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"
    ISM_433 = "ism_433"
    ISM_868 = "ism_868"
    ISM_915 = "ism_915"
    CUSTOM = "custom"

@dataclass
class SignalDetection:
    """فئة تمثل اكتشاف إشارة"""
    timestamp: str
    frequency: float
    bandwidth: float
    power: float
    signal_type: SignalType
    confidence: float
    location: Tuple[float, float]
    signature: str

class EducationalRFDetector:
    """نظام تعليمي لتحليل إشارات RF"""
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.known_signatures = self.load_known_signatures()
        self.detections_history: List[SignalDetection] = []
        self.alerts: List[Dict] = []
        
    def load_config(self, config_path: Optional[str]) -> Dict:
        """تحميل إعدادات النظام"""
        default_config = {
            "frequency_ranges": {
                "ISM_433": (433.05, 434.79),
                "ISM_868": (868.0, 868.6),
                "ISM_915": (902.0, 928.0),
                "WIFI_2G": (2400.0, 2483.5),
                "WIFI_5G": (5150.0, 5850.0),
                "BLUETOOTH": (2402.0, 2480.0)
            },
            "detection_threshold": -70,
            "scan_interval": 1.0,
            "location": (33.3152, 44.3661),
            "max_history": 1000,
            "sample_rate": 2.4e6  # إضافة معدل العينات
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except:
                print("⚠️  استخدام الإعدادات الافتراضية")
        
        return default_config
    
    def load_known_signatures(self) -> Dict:
        """تحميل توقيعات إشارات معروفة"""
        return {
            "EDU_WIFI_BEACON": {
                "frequency_range": (2412, 2472),
                "bandwidth": 20,
                "pattern": "periodic_beacon",
                "type": SignalType.WIFI
            },
            "EDU_BT_ADVERT": {
                "frequency_range": (2402, 2480),
                "bandwidth": 2,
                "pattern": "frequency_hopping",
                "type": SignalType.BLUETOOTH
            },
            "EDU_ISM_CONTROL": {
                "frequency_range": (433.05, 434.79),
                "bandwidth": 0.1,
                "pattern": "control_signal",
                "type": SignalType.ISM_433
            }
        }
    
    def simulate_rtl_sdr_scan(self) -> List[Dict]:
        """محاكاة مسح RTL-SDR"""
        simulations = []
        
        for _ in range(np.random.randint(1, 5)):
            freq_range = np.random.choice(list(self.config["frequency_ranges"].values()))
            freq = np.random.uniform(freq_range[0], freq_range[1])
            
            # إنشاء عينات إشارة حقيقية
            n_samples = 1024
            t = np.linspace(0, 0.001, n_samples)
            
            # إشارة جيبية مع ضوضاء
            signal_freq = np.random.uniform(1000, 10000)
            real_part = np.sin(2 * np.pi * signal_freq * t)
            imag_part = np.cos(2 * np.pi * signal_freq * t)
            
            # إضافة ضوضاء
            noise_real = 0.1 * np.random.randn(n_samples)
            noise_imag = 0.1 * np.random.randn(n_samples)
            
            simulation = {
                "frequency": freq,
                "power": np.random.uniform(-90, -30),
                "bandwidth": np.random.uniform(0.1, 20),
                "samples_real": real_part + noise_real,
                "samples_imag": imag_part + noise_imag,
                "timestamp": datetime.now().isoformat()
            }
            simulations.append(simulation)
        
        return simulations
    
    def analyze_signal_characteristics(self, signal_data: Dict) -> Dict:
        """تحليل خصائص الإشارة - الإصدار المصحح"""
        samples_real = signal_data.get("samples_real", np.array([]))
        samples_imag = signal_data.get("samples_imag", np.array([]))
        
        if len(samples_real) == 0 or len(samples_imag) == 0:
            return {"error": "لا توجد عينات كافية"}
        
        # إنشاء إشارة مركبة (معقدة)
        complex_signal = samples_real + 1j * samples_imag
        
        # تحويل فورييه
        fft_result = np.fft.fft(complex_signal)
        n = len(complex_signal)
        
        # التأكد من أن n موجب
        if n <= 0:
            return {"error": "عدد العينات غير صحيح"}
        
        # حساب الترددات
        frequencies = np.fft.fftfreq(n, d=1/self.config["sample_rate"])
        
        # حساب طيف القدرة
        power_spectrum = np.abs(fft_result) ** 2
        
        # العثور على ذروة التردد
        peak_idx = np.argmax(power_spectrum[:n//2])  # نأخذ النصف الموجب فقط
        peak_freq = np.abs(frequencies[peak_idx])
        
        # حساب عرض النطاق الترددي
        max_power = np.max(power_spectrum)
        threshold = 0.5 * max_power
        bandwidth_samples = np.sum(power_spectrum > threshold)
        bandwidth_hz = bandwidth_samples * self.config["sample_rate"] / n
        
        characteristics = {
            "peak_frequency": float(peak_freq / 1e6),  # تحويل إلى MHz
            "total_power": float(10 * np.log10(np.mean(power_spectrum) + 1e-10)),
            "bandwidth_estimate": float(bandwidth_hz / 1e3),  # تحويل إلى kHz
            "spectral_flatness": float(np.exp(np.mean(np.log(power_spectrum + 1e-10))) / np.mean(power_spectrum)),
            "modulation_score": np.random.random()
        }
        
        return characteristics
    
    def classify_signal(self, characteristics: Dict) -> Tuple[SignalType, float]:
        """تصنيف الإشارة"""
        freq = characteristics.get("peak_frequency", 0)
        bandwidth = characteristics.get("bandwidth_estimate", 0)
        
        if 2400 <= freq <= 2483.5:
            if 20 <= bandwidth <= 40:
                return SignalType.WIFI, 0.8
            elif bandwidth < 2:
                return SignalType.BLUETOOTH, 0.7
        
        elif 433 <= freq <= 434.79:
            return SignalType.ISM_433, 0.6
        
        elif 868 <= freq <= 868.6:
            return SignalType.ISM_868, 0.6
        
        elif 902 <= freq <= 928:
            return SignalType.ISM_915, 0.6
        
        return SignalType.UNKNOWN, 0.3
    
    def detect_anomalies(self, signal_data: Dict, characteristics: Dict) -> Optional[List[Dict]]:
        """اكتشاف إشارات غير عادية"""
        anomalies = []
        freq = characteristics.get("peak_frequency", 0)
        power = characteristics.get("total_power", -100)
        bandwidth = characteristics.get("bandwidth_estimate", 0)
        
        in_known_band = False
        for band_name, (f_low, f_high) in self.config["frequency_ranges"].items():
            if f_low <= freq <= f_high:
                in_known_band = True
                break
        
        if not in_known_band:
            anomalies.append({
                "type": "UNKNOWN_FREQUENCY",
                "severity": "MEDIUM",
                "message": f"إشارة على تردد غير معتاد: {freq:.2f} MHz"
            })
        
        if power > self.config["detection_threshold"]:
            anomalies.append({
                "type": "HIGH_POWER_SIGNAL",
                "severity": "LOW",
                "message": f"إشارة عالية الطاقة: {power:.1f} dBm"
            })
        
        if bandwidth > 50:
            anomalies.append({
                "type": "WIDE_BANDWIDTH",
                "severity": "MEDIUM",
                "message": f"عرض نطاق غير معتاد: {bandwidth:.1f} kHz"
            })
        
        return anomalies if anomalies else None
    
    def generate_signal_signature(self, signal_data: Dict) -> str:
        """إنشاء توقيع فريد للإشارة"""
        import hashlib
        freq = signal_data.get("frequency", 0)
        power = signal_data.get("power", 0)
        timestamp = signal_data.get("timestamp", "")
        signature_str = f"{freq:.3f}_{power:.1f}_{timestamp}"
        signature_hash = hashlib.md5(signature_str.encode()).hexdigest()[:8]
        return f"SIG_{signature_hash}"
    
    def scan_and_analyze(self) -> List[SignalDetection]:
        """تنفيذ دورة مسح وتحليل"""
        print(f"\n{'='*60}")
        print(f"جولة مسح RF - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        detections = []
        scanned_signals = self.simulate_rtl_sdr_scan()
        
        for i, signal in enumerate(scanned_signals):
            print(f"\n🔍 تحليل الإشارة #{i+1}")
            print(f"   التردد: {signal['frequency']:.2f} MHz")
            print(f"   القوة: {signal['power']:.1f} dBm")
            
            try:
                characteristics = self.analyze_signal_characteristics(signal)
                
                if "error" in characteristics:
                    print(f"   ❌ خطأ في التحليل: {characteristics['error']}")
                    continue
                    
                signal_type, confidence = self.classify_signal(characteristics)
                print(f"   النوع: {signal_type.value} (ثقة: {confidence:.1%})")
                
                anomalies = self.detect_anomalies(signal, characteristics)
                if anomalies:
                    print(f"   ⚠️  تم اكتشاف {len(anomalies)} شذوذ:")
                    for anomaly in anomalies:
                        print(f"      - {anomaly['message']}")
                        self.alerts.append({
                            **anomaly,
                            "frequency": signal['frequency'],
                            "timestamp": signal['timestamp']
                        })
                
                detection = SignalDetection(
                    timestamp=signal['timestamp'],
                    frequency=signal['frequency'],
                    bandwidth=characteristics.get('bandwidth_estimate', 0),
                    power=signal['power'],
                    signal_type=signal_type,
                    confidence=confidence,
                    location=self.config['location'],
                    signature=self.generate_signal_signature(signal)
                )
                
                detections.append(detection)
                self.detections_history.append(detection)
                
                if len(self.detections_history) > self.config['max_history']:
                    self.detections_history = self.detections_history[-self.config['max_history']:]
                    
            except Exception as e:
                print(f"   ❌ خطأ في معالجة الإشارة: {str(e)}")
                continue
        
        return detections
    
    def generate_report(self, period_hours: int = 24) -> Dict:
        """توليد تقرير عن الفترة المحددة"""
        cutoff_time = datetime.now().timestamp() - (period_hours * 3600)
        
        recent_detections = [
            d for d in self.detections_history
            if datetime.fromisoformat(d.timestamp).timestamp() > cutoff_time
        ]
        
        recent_alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']).timestamp() > cutoff_time
        ]
        
        report = {
            "report_time": datetime.now().isoformat(),
            "period_hours": period_hours,
            "total_detections": len(recent_detections),
            "total_alerts": len(recent_alerts),
            "signal_type_distribution": {},
            "alerts_by_severity": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            "frequency_coverage": {
                "known_bands": 0,
                "unknown_bands": 0
            },
            "top_anomalies": []
        }
        
        for detection in recent_detections:
            sig_type = detection.signal_type.value
            report["signal_type_distribution"][sig_type] = \
                report["signal_type_distribution"].get(sig_type, 0) + 1
            
            in_known_band = False
            for f_low, f_high in self.config["frequency_ranges"].values():
                if f_low <= detection.frequency <= f_high:
                    in_known_band = True
                    break
            
            if in_known_band:
                report["frequency_coverage"]["known_bands"] += 1
            else:
                report["frequency_coverage"]["unknown_bands"] += 1
        
        for alert in recent_alerts[-10:]:
            severity = alert.get("severity", "LOW")
            report["alerts_by_severity"][severity] += 1
            
            report["top_anomalies"].append({
                "time": alert['timestamp'],
                "type": alert['type'],
                "message": alert['message'],
                "frequency": alert.get('frequency', 0)
            })
        
        return report
    
    def run_continuous_monitoring(self, duration_minutes: int = 5):
        """تشغيل المراقبة المستمرة"""
        print("\n" + "="*60)
        print("بدء المراقبة المستمرة للطيف الترددي")
        print(f"المدة: {duration_minutes} دقيقة")
        print("="*60 + "\n")
        
        start_time = time.time()
        scan_count = 0
        
        try:
            while time.time() - start_time < duration_minutes * 60:
                scan_count += 1
                print(f"\n📡 جولة المسح #{scan_count}")
                self.scan_and_analyze()
                
                if scan_count % 3 == 0:
                    report = self.generate_report(period_hours=1)
                    print("\n📊 ملخص سريع:")
                    print(f"   الإجمالي: {report['total_detections']} اكتشاف")
                    print(f"   التنبيهات: {report['total_alerts']}")
                    for severity, count in report['alerts_by_severity'].items():
                        if count > 0:
                            print(f"   {severity}: {count}")
                
                time.sleep(self.config['scan_interval'])
                
        except KeyboardInterrupt:
            print("\n\n⏹️  توقف المراقبة بناءً على طلب المستخدم")
        
        print("\n" + "="*60)
        print("تقرير المراقبة النهائي")
        print("="*60)
        
        final_report = self.generate_report(period_hours=24)
        
        print(f"\nالمسوحات المكتملة: {scan_count}")
        print(f"إجمالي الاكتشافات: {final_report['total_detections']}")
        print(f"إجمالي التنبيهات: {final_report['total_alerts']}")
        
        print("\nتوزيع أنواع الإشارات:")
        for sig_type, count in final_report['signal_type_distribution'].items():
            print(f"  {sig_type}: {count}")
        
        if final_report['top_anomalies']:
            print("\nأهم الشذوذات المكتشفة:")
            for anomaly in final_report['top_anomalies'][-5:]:
                print(f"  [{anomaly['time'][11:19]}] {anomaly['message']}")

def show_menu():
    """عرض القائمة الرئيسية"""
    print("\n" + "="*50)
    print("نظام كشف إشارات RF التعليمي")
    print("="*50)
    print("1. مسح ترددي واحد")
    print("2. مراقبة مستمرة (5 دقائق)")
    print("3. عرض التقرير")
    print("4. معلومات النظام")
    print("5. الخروج")

def main():
    """الدالة الرئيسية"""
    print("="*70)
    print("نظام كشف إشارات RF التعليمي - الإصدار 1.1 (مصحح)")
    print("="*70)
    print("\n⚠️  تحذير: هذا نظام تعليمي للتدريب فقط")
    print("   لأغراض البحث والتعليم المشروع\n")
    
    detector = EducationalRFDetector()
    
    while True:
        show_menu()
        
        try:
            choice = input("\nاختر الخيار (1-5): ").strip()
            
            if choice == "1":
                detections = detector.scan_and_analyze()
                if detections:
                    print(f"\n✅ تم اكتشاف {len(detections)} إشارة")
                else:
                    print("\nℹ️  لم يتم اكتشاف أي إشارات")
                    
            elif choice == "2":
                detector.run_continuous_monitoring(duration_minutes=5)
                
            elif choice == "3":
                report = detector.generate_report(period_hours=24)
                print("\n📈 تقرير الـ24 ساعة الماضية:")
                print(f"وقت التقرير: {report['report_time']}")
                print(f"فترة التقرير: {report['period_hours']} ساعة")
                print(f"إجمالي الاكتشافات: {report['total_detections']}")
                print(f"إجمالي التنبيهات: {report['total_alerts']}")
                
                if report['signal_type_distribution']:
                    print("\nتوزيع أنواع الإشارات:")
                    for sig_type, count in report['signal_type_distribution'].items():
                        print(f"  {sig_type}: {count}")
                
                if report['top_anomalies']:
                    print("\nأحدث التنبيهات:")
                    for anomaly in report['top_anomalies'][-5:]:
                        print(f"  [{anomaly['time'][11:19]}] {anomaly['message']}")
                        
            elif choice == "4":
                print("\n📋 معلومات النظام:")
                print("الإصدار: 1.1 (مصحح)")
                print("الغرض: تدريب على تحليل إشارات RF")
                print("المتطلبات: numpy")
                print("\nتعديلات هذا الإصدار:")
                print("- إصلاح مشكلة تحويل فورييه")
                print("- تحسين معالجة الإشارات")
                print("- إضافة معالجة الأخطاء")
                
            elif choice == "5":
                print("\nشكراً لاستخدام النظام التعليمي")
                print("التزم دائمًا بالقوانين واللوائح المحلية")
                break
            else:
                print("❌ خيار غير صالح")
                
        except KeyboardInterrupt:
            print("\n\nتم الخروج من البرنامج")
            break
        except Exception as e:
            print(f"\n❌ خطأ غير متوقع: {e}")
            print("يرجى إعادة تشغيل البرنامج")

if __name__ == "__main__":
    main()
