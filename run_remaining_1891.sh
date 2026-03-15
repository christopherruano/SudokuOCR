#!/bin/bash
# Process remaining 1891 pages with 3.1-pro (fallback to 2.5-pro)
cd /Users/chrisruano/Research/HEB_91r
export GEMINI_MODEL=gemini-3.1-pro-preview

echo "=== Processing remaining 1891 pages ==="
date

# Assam pages 1, 2 (already have 3,4,5,6)
for page in age_tables/Assam/1891/Assam_age_1891-1.png age_tables/Assam/1891/Assam_age_1891-2.png; do
    if [ -f "$page" ]; then
        stem=$(basename "$page" .png)
        if ! ls results/${stem}*_oneshot.json 1>/dev/null 2>&1; then
            echo "--- Processing: $page ---"
            python3 oneshot.py "$page"
        else
            echo "--- Skipping (exists): $page ---"
        fi
    fi
done

# Burma pages 2-10 (already have 01)
for i in $(seq -w 2 10); do
    page="age_tables/Burma/1891/Burma_age_1891-$(printf '%02d' $i).png"
    if [ -f "$page" ]; then
        stem=$(basename "$page" .png)
        if ! ls results/${stem}*_oneshot.json 1>/dev/null 2>&1; then
            echo "--- Processing: $page ---"
            python3 oneshot.py "$page"
        else
            echo "--- Skipping (exists): $page ---"
        fi
    fi
done

# Madras, NW Provinces, Punjab, Central India, Berar, Bombay
for province_year in "Madras/1891" "North_Western_Provinces_Oudh/1891" "Punjab/1891" "Central_India/1891" "Berar/1901" "Bombay/1891"; do
    for page in age_tables/${province_year}/*.png; do
        if [ -f "$page" ]; then
            stem=$(basename "$page" .png)
            if ! ls results/${stem}*_oneshot.json 1>/dev/null 2>&1; then
                echo "--- Processing: $page ---"
                python3 oneshot.py "$page"
            else
                echo "--- Skipping (exists): $page ---"
            fi
        fi
    done
done

echo ""
echo "=== DONE ==="
date
