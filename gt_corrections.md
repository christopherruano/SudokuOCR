# Ground Truth Corrections

Known errors in the Excel ground truth files, discovered by cross-referencing model outputs with scanned originals.

## Travancore Eastern Division 1901

### Row 20-25
- **persons**: Excel says 112040, correct value is **111950**
- **males**: Excel says 53092, correct value is **53002**
- Source: Confirmed by all 3 models reading the scan consistently

### Row 30-35
- **persons**: Excel says 96124, correct value is **95124**
- **females**: Excel says 46867, correct value is **45867**
- Source: Confirmed by all 3 models reading the scan consistently

## Coorg 1901

Per-1000 table. Corrections confirmed by user visual inspection of original scan.

### Row 0-5 (aggregated from 0-1 through 4-5)
- **persons**: Excel individual years sum to 219, scan subtotal reads **218**
- **males**: Excel individual years sum to 96, scan subtotal reads **95**
- Source: All models consistently read Total 0-5 as 218/95/123

### Row 50-55
- **persons**: Excel says 78, correct value is **67** (= 32 + 35)
- **males**: Excel says 43, correct value is **32**
- Source: User confirmed M=32 by visual inspection; P=M+F=32+35=67

## Hyderabad State 1901

All corrections confirmed by visual inspection of the original scan. In every case, models read internally consistent values (M+F=P) that differ from the Excel GT.

### Row 5-10
- **persons**: Excel 1452869 → **1452854**
- **males**: Excel 755968 → **755963**
- **females**: Excel 696901 → **696891**

### Row 10-15
- **persons**: Excel 1344321 → **1350321**
- **males**: Excel 733509 → **739509**

### Row 15-20
- **persons**: Excel 852380 → **852389**
- **males**: Excel 428400 → **428409**

### Row 20-25
- **persons**: Excel 893929 → **894027**
- **males**: Excel 405709 → **405798**
- **females**: Excel 488220 → **488229**

### Row 25-30
- **persons**: Excel 1051408 → **1051503**
- **males**: Excel 523518 → **523613**

### Row 35-40
- **persons**: Excel 628689 → **628589**
- **females**: Excel 286076 → **285976**

### Row 40-45
- **persons**: Excel 818045 → **818675**
- **females**: Excel 392292 → **392922**

### Row 45-50
- **persons**: Excel 351028 → **357058**
- **females**: Excel 150214 → **156244**

### Row 60 and over
- **persons**: Excel 571011 → **570951**
- **males**: Excel 271092 → **271002**
- **females**: Excel 299919 → **299949**
