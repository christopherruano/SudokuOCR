"""Export standardized, data-science-ready flat files from validated OCR results.

Produces:
  standardized/primary_age_civil_condition.csv  — main age×sex×civil condition tables
  standardized/religion_age_civil_condition.csv — religion-specific breakdowns
  standardized/all_tables.csv                   — everything in one flat file
  standardized/codebook.txt                     — column definitions

Standardization applied:
  - Column names normalized (lowercase, no trailing dots)
  - Region names normalized (title case, no trailing dots)
  - Age labels cleaned (stripped, consistent casing for totals)
  - Split-page halves joined where possible
  - Null numeric values → empty (not 0)
"""

import json
import os
import re
import csv
import glob
from collections import defaultdict
from pathlib import Path

ORGANIZED_ROOT = os.path.expanduser("~/Research/SudokuOCR/organized_results")
OUTPUT_DIR = os.path.expanduser("~/Research/SudokuOCR/export/csv")

RELIGION_KEYWORDS = [
    'hindu', 'muslim', 'musalman', 'muhammadan', 'mohammedan',
    'christian', 'sikh', 'buddhist', 'jain', 'parsi', 'zoroastrian',
    'animist', 'arya', 'brahmo', 'tribal', 'jewish',
]


def normalize_region(region):
    if not region:
        return ""
    region = region.strip().rstrip('.')
    region = re.sub(r'\s+', ' ', region)
    region = region.replace('-concld.', '').replace('-concld', '')
    region = region.replace('-concluded.', '').replace('-concluded', '')
    region = region.replace('-contd.', '').replace('-contd', '')
    region = region.replace('-continued.', '').replace('-continued', '')
    region = region.strip().rstrip('.-– ')
    return region.title() if region.isupper() else region


def normalize_col_group(name):
    return name.strip().rstrip('.').strip()


def normalize_age_label(label):
    if not label:
        return ""
    label = str(label).strip().rstrip('.')
    label = label.replace('—', '-').replace('–', '-')  # em/en dash → hyphen
    label = re.sub(r'\s+', ' ', label).strip()
    low = label.lower().strip()
    if low in ('total', 'all ages', 'all age', 'total all ages', 'total all ages...', 'grand total'):
        return "Total"
    if re.match(r'^total\s+0[\s-]', low):
        num = re.sub(r'^total\s+', '', label, flags=re.IGNORECASE)
        return f"Total {num}"
    if low in ('all religions', 'all religions...'):
        return "All Religions"
    if low in ('all races',):
        return "All Races"
    if low in ('age not stated', 'unspecified', 'not stated', 'age unspecified'):
        return "Age not stated"
    return label


def extract_religion_from_title(title):
    """Try to extract the specific religion from the table title."""
    title_lower = title.lower()
    patterns = [
        (r'religion[\s.:-]*hindu', 'Hindu'),
        (r'religion[\s.:-]*musal?man', 'Muslim'),
        (r'religion[\s.:-]*muhammadan', 'Muslim'),
        (r'religion[\s.:-]*mohammedan', 'Muslim'),
        (r'religion[\s.:-]*christian', 'Christian'),
        (r'religion[\s.:-]*sikh', 'Sikh'),
        (r'religion[\s.:-]*buddhist', 'Buddhist'),
        (r'religion[\s.:-]*jain', 'Jain'),
        (r'religion[\s.:-]*parsi', 'Parsi'),
        (r'religion[\s.:-]*zoroastrian', 'Parsi'),
        (r'religion[\s.:-]*arya', 'Arya'),
        (r'religion[\s.:-]*brahmo', 'Brahmo'),
        (r'religion[\s.:-]*animis', 'Animist'),
        (r'religion[\s.:-]*tribal', 'Tribal'),
        (r'religion[\s.:-]*jewish', 'Jewish'),
        (r'religion[\s.:-]*other', 'Others'),
        (r'\bhindus?\b', 'Hindu'),
        (r'\bmusal?mans?\b', 'Muslim'),
        (r'\bmuhammadans?\b', 'Muslim'),
        (r'\bchristians?\b', 'Christian'),
        (r'\bsikhs?\b', 'Sikh'),
        (r'\bbuddhists?\b', 'Buddhist'),
        (r'\bjains?\b', 'Jain'),
        (r'\bparsis?\b', 'Parsi'),
        (r'\baryas?\b', 'Arya'),
    ]
    for pattern, religion in patterns:
        if re.search(pattern, title_lower):
            return religion
    return ""


def classify_table(meta, data_sections):
    title = (meta.get('title') or '').lower()
    col_groups = meta.get('column_groups', [])
    col_str = ' '.join(g.lower() for g in col_groups)
    data_type = meta.get('data_type', '')

    is_civil = ('civil condition' in title or 'civil condition' in col_str
                or 'dition and age' in title
                or 'marital' in title or 'marital' in col_str
                or any(x in col_str for x in ['unmarried', 'married', 'widowed', 'divorced']))

    is_all_religions = 'all religions' in title
    has_religion = any(kw in title for kw in RELIGION_KEYWORDS)

    is_life_table = 'life table' in title
    is_education = 'education' in title or 'literate' in title or 'literate' in col_str
    is_occupation = 'occupation' in title or 'industri' in title
    is_infirmity = any(kw in title for kw in ['infirmit', 'blind', 'deaf', 'leper', 'insane'])
    is_subsidiary = 'subsidiary' in title
    is_proportional = (data_type == 'proportional' or 'proportion' in title
                       or '10,000' in title or '100,000' in title
                       or 'per mille' in title or 'per cent' in title)
    is_single_year = 'single year' in title
    is_language = 'bilingual' in title or 'language' in title
    is_birth_death = 'birth' in title or 'death' in title or 'fertility' in title

    if is_life_table:
        return "life_table", ""
    if is_infirmity:
        return "infirmity", ""
    if is_education:
        return ("education_religion" if has_religion else "education"), ""
    if is_occupation:
        return "occupation", ""
    if is_language:
        return "language", ""
    if is_birth_death:
        return "birth_death", ""
    if is_civil:
        if has_religion and not is_all_religions:
            religion = extract_religion_from_title(meta.get('title', ''))
            return "age_civil_condition_religion", religion
        return "age_civil_condition", ""
    if is_single_year:
        return "single_year_age", ""
    if is_subsidiary:
        return "subsidiary", ""
    if is_proportional:
        return "proportional", ""
    return "age_sex_other", ""


def detect_column_type(col_groups):
    """Determine what kind of column structure this table has."""
    norm = [normalize_col_group(c).upper() for c in col_groups]
    norm_set = set(norm)

    if norm_set >= {'POPULATION', 'UNMARRIED', 'MARRIED', 'WIDOWED'}:
        return 'full_civil'
    if norm_set == {'POPULATION', 'UNMARRIED'}:
        return 'left_half'
    if norm_set == {'MARRIED', 'WIDOWED'}:
        return 'right_half'
    if norm_set >= {'TOTAL', 'UNMARRIED', 'MARRIED', 'WIDOWED'}:
        return 'full_civil'
    if 'POPULATION' in norm_set or 'TOTAL' in norm_set or 'TOTAL POPULATION' in norm_set:
        return 'population_only'
    return 'custom'


def flatten_civil_condition_row(row, col_groups, col_type):
    """Flatten a row from a civil condition table into standardized columns."""
    age_label = ""
    for k, v in row.items():
        if not isinstance(v, dict):
            age_label = normalize_age_label(v)
            break

    result = {'age': age_label}

    col_mapping = {}
    for cg in col_groups:
        norm = normalize_col_group(cg).upper()
        if norm in ('POPULATION', 'TOTAL', 'TOTAL POPULATION'):
            col_mapping[cg] = 'population'
        elif norm == 'UNMARRIED':
            col_mapping[cg] = 'unmarried'
        elif norm in ('MARRIED',):
            col_mapping[cg] = 'married'
        elif norm in ('WIDOWED',):
            col_mapping[cg] = 'widowed'
        elif norm == 'SINGLE':
            col_mapping[cg] = 'unmarried'
        elif norm == 'DIVORCED':
            col_mapping[cg] = 'divorced'

    for cg, prefix in col_mapping.items():
        val = row.get(cg, {})
        if isinstance(val, dict):
            result[f'{prefix}_persons'] = val.get('persons')
            result[f'{prefix}_males'] = val.get('males')
            result[f'{prefix}_females'] = val.get('females')
        else:
            result[f'{prefix}_total'] = val

    return result


def flatten_generic_row(row, col_groups):
    """Flatten a row from a non-civil-condition table."""
    age_label = ""
    row_label_key = None
    for k, v in row.items():
        if not isinstance(v, dict):
            age_label = normalize_age_label(v)
            row_label_key = k
            break

    result = {'row_label': age_label}

    for cg in col_groups:
        norm_cg = normalize_col_group(cg)
        val = row.get(cg, {})
        if isinstance(val, dict):
            for sk, sv in val.items():
                result[f'{norm_cg}_{sk}'] = sv
        else:
            result[norm_cg] = val

    return result


def load_all_validated():
    jsons = glob.glob(os.path.join(ORGANIZED_ROOT, '**', '*_oneshot.json'), recursive=True)
    records = []

    for jf in sorted(jsons):
        with open(jf) as f:
            d = json.load(f)

        constraints = d.get('constraints', {})
        total_checks = constraints.get('total_checks', 0)
        failed = constraints.get('failed', 0)
        if total_checks == 0 or failed > 0:
            continue

        meta = d.get('metadata', {})
        rel = os.path.relpath(os.path.dirname(jf), ORGANIZED_ROOT)
        parts = rel.split('/')
        province = parts[0].replace('_', ' ') if len(parts) >= 1 else 'Unknown'

        try:
            year = int(meta.get('year', 0))
        except (ValueError, TypeError):
            year = 0
        if year == 0:
            try:
                year = int(parts[1]) if len(parts) >= 2 else 0
            except ValueError:
                year = 0

        region = normalize_region(meta.get('region', ''))
        title = meta.get('title', '')
        data_sections = d.get('data', [])
        table_type, religion = classify_table(meta, data_sections)
        col_groups = meta.get('column_groups', [])
        col_type = detect_column_type(col_groups)
        data_type = meta.get('data_type', 'absolute')
        source = os.path.relpath(jf, ORGANIZED_ROOT)
        trust = d.get('acceptance', {}).get('trust_score', '')

        records.append({
            'province': province,
            'year': year,
            'region': region,
            'title': title,
            'table_type': table_type,
            'religion': religion,
            'col_groups': col_groups,
            'col_type': col_type,
            'data_type': data_type,
            'data_sections': data_sections,
            'source': source,
            'trust_score': trust,
            'n_checks': total_checks,
        })

    return records


CIVIL_COLUMNS = [
    'province', 'year', 'region', 'title', 'section', 'section_type', 'age',
    'population_persons', 'population_males', 'population_females',
    'unmarried_persons', 'unmarried_males', 'unmarried_females',
    'married_persons', 'married_males', 'married_females',
    'widowed_persons', 'widowed_males', 'widowed_females',
    'data_type', 'trust_score', 'source',
]

ALL_COLUMNS = [
    'province', 'year', 'region', 'table_type', 'title', 'religion',
    'section', 'section_type', 'row_label', 'column_group', 'sub_column', 'value',
    'data_type', 'trust_score', 'source',
]


RELIGION_SECTION_KEYWORDS = [
    'hindu', 'muslim', 'musalman', 'muhammadan', 'mohammedan',
    'christian', 'sikh', 'buddhist', 'jain', 'parsi', 'zoroastrian',
    'animist', 'animistic', 'arya', 'brahmo', 'tribal', 'jewish', 'jew',
    'nat-worshipper', 'nat worshipper',
]


def classify_section(section_name):
    """Classify a data section as 'Total', a specific religion, or 'Other'."""
    if not section_name:
        return "Total"
    name_lower = section_name.lower().strip().rstrip('.')
    # Strip leading markers like "(a) ", "A.-", "B-", but only single-char prefixes
    name_clean = re.sub(r'^[\(\[]?[a-z][\)\]]\s*[-–—.]?\s*', '', name_lower).strip()
    name_clean = re.sub(r'^[a-z][-–—.]\s*', '', name_clean).strip()

    if 'all religions' in name_clean or 'all races' in name_clean:
        return "Total"
    if name_clean in ('total population', 'total', 'all castes and tribes',
                       'all castes', 'general population'):
        return "Total"

    for kw in RELIGION_SECTION_KEYWORDS:
        if kw in name_clean:
            return "Religion"

    if name_clean in ('others', 'other religions', 'other', 'minor religions',
                       'minor and unspecified religions', 'minor and unspecified',
                       'other indigenous races', 'other races'):
        return "Religion"
    if 'minor' in name_clean and ('religion' in name_clean or 'unspecified' in name_clean):
        return "Religion"

    if re.match(r'^(table|part|subsidiary)', name_clean):
        return "Total"

    return "Other"


def try_join_halves(records):
    """Try to join left-half (Pop+Unm) and right-half (Mar+Wid) tables."""
    left = defaultdict(list)
    right = defaultdict(list)
    others = []

    for r in records:
        if r['col_type'] == 'left_half':
            key = (r['province'], r['year'], r['region'])
            left[key].append(r)
        elif r['col_type'] == 'right_half':
            key = (r['province'], r['year'], r['region'])
            right[key].append(r)
        else:
            others.append(r)

    joined = []
    used_right = set()

    for key, lefts in left.items():
        rights = right.get(key, [])
        for li, l_rec in enumerate(lefts):
            if li < len(rights):
                r_rec = rights[li]
                merged = merge_halves(l_rec, r_rec)
                if merged:
                    joined.append(merged)
                    used_right.add((key, li))
                    continue
            others.append(l_rec)

    for key, rights in right.items():
        for ri, r_rec in enumerate(rights):
            if (key, ri) not in used_right:
                others.append(r_rec)

    return others + joined


def merge_halves(left_rec, right_rec):
    """Merge a left-half and right-half record."""
    merged = dict(left_rec)
    merged['col_groups'] = left_rec['col_groups'] + right_rec['col_groups']
    merged['col_type'] = 'full_civil'
    merged['source'] = f"{left_rec['source']} + {right_rec['source']}"

    merged_sections = []
    for ls, rs in zip(left_rec['data_sections'], right_rec['data_sections']):
        merged_rows = []
        for lr, rr in zip(ls.get('rows', []), rs.get('rows', [])):
            row = dict(lr)
            for k, v in rr.items():
                if isinstance(v, dict) and k not in row:
                    row[k] = v
            merged_rows.append(row)
        merged_sections.append({'name': ls['name'], 'rows': merged_rows})

    merged['data_sections'] = merged_sections
    return merged


def export_civil_condition(records, output_path, filter_religion=False):
    """Export civil condition tables to flat CSV."""
    civil_recs = [r for r in records
                  if r['table_type'] in ('age_civil_condition', 'age_civil_condition_religion')]

    if filter_religion:
        civil_recs = [r for r in civil_recs if r['table_type'] == 'age_civil_condition_religion']
    else:
        civil_recs = [r for r in civil_recs if r['table_type'] == 'age_civil_condition']

    civil_recs = try_join_halves(civil_recs)

    rows_written = 0
    with open(output_path, 'w', newline='') as f:
        extra_cols = ['religion'] if filter_religion else []
        cols = CIVIL_COLUMNS[:5] + extra_cols + CIVIL_COLUMNS[5:]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()

        for rec in sorted(civil_recs, key=lambda r: (r['province'], r['year'], r['region'])):
            for section in rec['data_sections']:
                section_name = section.get('name', '')
                sec_type = classify_section(section_name)
                for row in section.get('rows', []):
                    flat = flatten_civil_condition_row(row, rec['col_groups'], rec['col_type'])
                    flat['province'] = rec['province']
                    flat['year'] = rec['year']
                    flat['region'] = rec['region']
                    flat['title'] = rec['title']
                    flat['section'] = section_name
                    flat['section_type'] = sec_type
                    flat['data_type'] = rec['data_type']
                    flat['trust_score'] = rec['trust_score']
                    flat['source'] = rec['source']
                    if filter_religion:
                        flat['religion'] = rec['religion']
                    writer.writerow(flat)
                    rows_written += 1

    return rows_written


def export_all_tables_long(records, output_path):
    """Export ALL tables in long/tidy format (one row per cell value)."""
    rows_written = 0
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS, extrasaction='ignore')
        writer.writeheader()

        for rec in sorted(records, key=lambda r: (r['province'], r['year'], r['region'])):
            for section in rec['data_sections']:
                section_name = section.get('name', '')
                sec_type = classify_section(section_name)
                for row in section.get('rows', []):
                    row_label = ""
                    for k, v in row.items():
                        if not isinstance(v, dict):
                            row_label = normalize_age_label(v)
                            break

                    for cg in rec['col_groups']:
                        val = row.get(cg, {})
                        norm_cg = normalize_col_group(cg)
                        base_row = {
                            'province': rec['province'],
                            'year': rec['year'],
                            'region': rec['region'],
                            'table_type': rec['table_type'],
                            'title': rec['title'],
                            'religion': rec['religion'],
                            'section': section_name,
                            'section_type': sec_type,
                            'row_label': row_label,
                            'column_group': norm_cg,
                            'data_type': rec['data_type'],
                            'trust_score': rec['trust_score'],
                            'source': rec['source'],
                        }
                        if isinstance(val, dict):
                            for sk, sv in val.items():
                                if sv is not None:
                                    out = dict(base_row)
                                    out['sub_column'] = sk.strip().rstrip('.')
                                    out['value'] = sv
                                    writer.writerow(out)
                                    rows_written += 1
                        elif val is not None:
                            out = dict(base_row)
                            out['sub_column'] = ''
                            out['value'] = val
                            writer.writerow(out)
                            rows_written += 1

    return rows_written


def write_codebook(output_path, primary_rows, religion_rows, all_rows):
    with open(output_path, 'w') as f:
        f.write("CENSUS OCR DATA — STANDARDIZED EXPORT CODEBOOK\n")
        f.write("=" * 60 + "\n\n")

        f.write("FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"primary_age_civil_condition.csv   {primary_rows:>8,} rows  General population age×sex×civil condition tables\n")
        f.write(f"religion_age_civil_condition.csv  {religion_rows:>8,} rows  Religion-specific age×sex×civil condition tables\n")
        f.write(f"all_tables_long.csv               {all_rows:>8,} rows  All validated tables in long/tidy format\n\n")

        f.write("PRIMARY / RELIGION CSV COLUMNS\n")
        f.write("-" * 40 + "\n")
        f.write("province              Province or state name (title case)\n")
        f.write("year                  Census year (integer)\n")
        f.write("region                District, division, or region name (normalized)\n")
        f.write("title                 Original table title from the census page\n")
        f.write("section               Data section within the table (often religion or district name)\n")
        f.write("section_type          'Total' (general pop), 'Religion' (specific religion), or 'Other'\n")
        f.write("age                   Age group label (e.g. '0-1', '5-10', 'Total')\n")
        f.write("religion              (religion CSV only) Religion name\n")
        f.write("population_persons    Total persons in age group\n")
        f.write("population_males      Male persons in age group\n")
        f.write("population_females    Female persons in age group\n")
        f.write("unmarried_persons     Unmarried persons\n")
        f.write("unmarried_males       Unmarried males\n")
        f.write("unmarried_females     Unmarried females\n")
        f.write("married_persons       Married persons\n")
        f.write("married_males         Married males\n")
        f.write("married_females       Married females\n")
        f.write("widowed_persons       Widowed persons\n")
        f.write("widowed_males         Widowed males\n")
        f.write("widowed_females       Widowed females\n")
        f.write("data_type             'absolute' (raw counts) or 'proportional'\n")
        f.write("trust_score           Automated trust score (0-100)\n")
        f.write("source                Path to source JSON in organized_results/\n\n")

        f.write("ALL TABLES LONG CSV COLUMNS\n")
        f.write("-" * 40 + "\n")
        f.write("province              Province or state name\n")
        f.write("year                  Census year\n")
        f.write("region                District/region name\n")
        f.write("table_type            Category (age_civil_condition, education, etc.)\n")
        f.write("title                 Original table title from the census page\n")
        f.write("religion              Religion name (if religion-specific table)\n")
        f.write("section               Data section within the table\n")
        f.write("section_type          'Total' (general pop), 'Religion', or 'Other'\n")
        f.write("row_label             Row label (age group, district name, etc.)\n")
        f.write("column_group          Column group name (Population, Unmarried, etc.)\n")
        f.write("sub_column            Sub-column (persons, males, females)\n")
        f.write("value                 Numeric value\n")
        f.write("data_type             'absolute' or 'proportional'\n")
        f.write("trust_score           Automated trust score\n")
        f.write("source                Path to source JSON\n\n")

        f.write("TABLE TYPES\n")
        f.write("-" * 40 + "\n")
        f.write("age_civil_condition          Age × Sex × Civil Condition (the primary census table)\n")
        f.write("age_civil_condition_religion  Same, but for a specific religion\n")
        f.write("age_sex_other                Age × Sex tables (no civil condition)\n")
        f.write("education                    Literacy and education tables\n")
        f.write("education_religion           Education tables by religion\n")
        f.write("proportional                 Per-10,000 or percentage distributions\n")
        f.write("subsidiary                   Subsidiary analytical tables\n")
        f.write("single_year_age              Single-year-of-age returns\n")
        f.write("infirmity                    Disability/infirmity tables\n")
        f.write("life_table                   Actuarial life tables\n")
        f.write("occupation                   Occupation/industry tables\n")
        f.write("language                     Bilingualism and language tables\n")
        f.write("birth_death                  Vital statistics\n\n")

        f.write("STANDARDIZATION APPLIED\n")
        f.write("-" * 40 + "\n")
        f.write("- Region names: normalized casing, trailing dots/dashes removed\n")
        f.write("- Column groups: trailing dots removed, consistent casing\n")
        f.write("- Age labels: 'Total'/'TOTAL'/'All ages' → 'Total'; 'Age not stated' standardized\n")
        f.write("- Split-page tables joined where left (Pop+Unm) and right (Mar+Wid) halves match\n")
        f.write("- All values are integers (raw counts) unless data_type='proportional'\n")
        f.write("- Empty cells are blank (not 0) — zeros are real zeros\n\n")

        f.write("NOTES FOR ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("- Age group bins differ across census years. Common patterns:\n")
        f.write("    (a) 0-1, 1-2, ..., 4-5, Total 0-5, 5-10, 10-15, ... (fine-grained)\n")
        f.write("    (b) 0-5, 5-10, 10-15, 15-20, 20-40, 40-60, 60+ (coarse)\n")
        f.write("    (c) 0-10, 10-15, 15-40, 40+ (very coarse, early censuses)\n")
        f.write("  Harmonization across years requires rebinning to a common set.\n")
        f.write("- Region names are not fully standardized across census years.\n")
        f.write("  The same district may appear under different spellings or administrative units.\n")
        f.write("- Multi-section tables: some image scans contain multiple districts.\n")
        f.write("  The 'section' column identifies which district within the table.\n")
        f.write("- Constraint validation: M + F = Persons, and vertical sums (age groups sum to total)\n")
        f.write("  were verified for all tables. Trust_score=100 means all checks passed.\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading validated records...")
    records = load_all_validated()
    print(f"  {len(records)} validated tables")

    print("Exporting primary civil condition tables...")
    primary_rows = export_civil_condition(
        records,
        os.path.join(OUTPUT_DIR, 'primary_age_civil_condition.csv'),
        filter_religion=False,
    )
    print(f"  {primary_rows:,} rows written")

    print("Exporting religion civil condition tables...")
    religion_rows = export_civil_condition(
        records,
        os.path.join(OUTPUT_DIR, 'religion_age_civil_condition.csv'),
        filter_religion=True,
    )
    print(f"  {religion_rows:,} rows written")

    print("Exporting all tables (long format)...")
    all_rows = export_all_tables_long(
        records,
        os.path.join(OUTPUT_DIR, 'all_tables_long.csv'),
    )
    print(f"  {all_rows:,} rows written")

    print("Writing codebook...")
    write_codebook(
        os.path.join(OUTPUT_DIR, 'codebook.txt'),
        primary_rows, religion_rows, all_rows,
    )

    print(f"\nDone! Files in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
