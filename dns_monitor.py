#!/usr/bin/env python3
# dns_monitor.py

from scapy.all import *
import json
from datetime import datetime

class DNSMonitor:
    def init(self):
        self.dns_queries = []
        self.blocked_domains = [
            'mossad', 'CIA', 'Mi6', 'gambling',
            'violence', 'hacking', 'warez'
        ]
    
    def dns_callback(self, pkt):
        """معالجة حزم DNS"""
        try:
            if pkt.haslayer(DNSQR):  # استعلام DNS
                domain = pkt[DNSQR].qname.decode('utf-8', errors='ignore')
                src_ip = pkt[IP].src
                
                # تسجيل الاستعلام
                record = {
                    'timestamp': str(datetime.now()),
                    'source_ip': src_ip,
                    'domain': domain,
                    'blocked': any(b in domain.lower() for b in self.blocked_domains)
                }
                
                self.dns_queries.append(record)
                
                # عرض التنبيهات
                if record['blocked']:
                    print(f"[!] تحذير: تم الوصول إلى موقع محظور من {src_ip}")
                    print(f"    الموقع: {domain}")
                
                # عرض جميع الاستعلامات
                print(f"[DNS] {src_ip} -> {domain}")
                
                # حفظ دوري
                if len(self.dns_queries) % 10 == 0:
                    self.save_logs()
                    
        except Exception as e:
            pass
    
    def save_logs(self):
        """حفظ سجلات DNS"""
        with open('dns_log.json', 'w') as f:
            json.dump(self.dns_queries, f, indent=4)
    
    def start_monitoring(self, interface='eth0'):
        """بدء مراقبة حركة DNS"""
        print(f"[*] بدء مراقبة DNS على واجهة {interface}")
        print("[*] جاري تسجيل استعلامات DNS...")
        print("[*] اضغط Ctrl+C لإيقاف المراقبة\n")
        
        try:
            # تصفية حركة DNS (منفذ 53)
            sniff(filter='udp port 53', 
                  prn=self.dns_callback,
                  store=0,
                  iface=interface)
                  
        except KeyboardInterrupt:
            print("\n[*] إيقاف المراقبة...")
            self.save_logs()
            print(f"[+] تم حفظ {len(self.dns_queries)} سجل DNS")

if name == "main":
    monitor = DNSMonitor()
    
    # تأكد من تغيير eth0 إلى واجهة الشبكة الصحيحة
    # استخدم الأمر: ip addr show لمعرفة الواجهة الصحيحة
    monitor.start_monitoring(interface='eth0')
