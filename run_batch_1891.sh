#!/bin/bash
cd /Users/chrisruano/Research/HEB_91r
echo "=== Starting 1891 multi-group batch ==="
echo "=== Mysore 1891 ===" >> batch_log.txt
python3 batch_run.py --province Mysore --year 1891 >> batch_log.txt 2>&1
echo "=== Assam 1891 ===" >> batch_log.txt
python3 batch_run.py --province Assam --year 1891 >> batch_log.txt 2>&1
echo "=== Burma 1891 ===" >> batch_log.txt
python3 batch_run.py --province Burma --year 1891 >> batch_log.txt 2>&1
echo "=== Berar 1901 ===" >> batch_log.txt
python3 batch_run.py --province Berar --year 1901 >> batch_log.txt 2>&1
echo "=== Madras 1891 ===" >> batch_log.txt
python3 batch_run.py --province Madras --year 1891 >> batch_log.txt 2>&1
echo "=== NW Provinces 1891 ===" >> batch_log.txt
python3 batch_run.py --province North_Western_Provinces_Oudh --year 1891 >> batch_log.txt 2>&1
echo "=== Punjab 1891 ===" >> batch_log.txt
python3 batch_run.py --province Punjab --year 1891 >> batch_log.txt 2>&1
echo "=== Central India 1891 ===" >> batch_log.txt
python3 batch_run.py --province Central_India --year 1891 >> batch_log.txt 2>&1
echo "=== DONE ===" >> batch_log.txt
date >> batch_log.txt
