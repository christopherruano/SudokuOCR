"""Build a comprehensive data guide for the Census OCR export.

Generates an Excel workbook (data_guide.xlsx) with:
  - Summary: high-level stats on what we have
  - Inventory: one row per validated table with province, year, region, category, etc.
  - Coverage: province × year matrix showing table counts and gaps
  - Format Notes: consistency findings for data science use
"""

import argparse
import json
import os
import glob
import re
from collections import defaultdict, Counter
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill, numbers
    from openpyxl.utils import get_column_letter
except ImportError:
    print("pip install openpyxl")
    raise SystemExit(1)

from build_gap_fillability_guide import (
    append_summary_section,
    coverage_from_records,
    load_gap_rows,
    write_detail_sheet as write_gap_detail_sheet,
    write_matrix_sheet as write_gap_matrix_sheet,
)

ORGANIZED_ROOT = os.path.expanduser("~/Research/SudokuOCR/organized_results")
EXPORT_ROOT = os.path.expanduser("~/Research/SudokuOCR/export")
OUTPUT_PATH = os.path.expanduser("~/Research/SudokuOCR/export/data_guide.xlsx")
DEFAULT_GAP_JSON = os.path.expanduser("~/census_toc/gap_candidates.json")

RELIGION_KEYWORDS = [
    'hindu', 'muslim', 'musalman', 'muhammadan', 'mohammedan',
    'christian', 'sikh', 'buddhist', 'jain', 'parsi', 'zoroastrian',
    'animist', 'arya', 'brahmo', 'tribal', 'jewish',
    'all religions', 'religion',
]

# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

def classify_table(meta, data_sections):
    title = (meta.get('title') or '').lower()
    col_groups = meta.get('column_groups', [])
    col_str = ' '.join(g.lower() for g in col_groups)
    sections = [s['name'] for s in data_sections]
    sections_str = ' '.join(sections).lower()
    data_type = meta.get('data_type', '')

    # "All Religions" means the general population table, not a religion-specific one
    is_all_religions = 'all religions' in title
    specific_religion_keywords = [kw for kw in RELIGION_KEYWORDS if kw != 'all religions' and kw != 'religion']
    has_religion = (any(kw in title for kw in specific_religion_keywords)
                    or ('religion' in title and not is_all_religions and 'by religion' not in title))
    has_religion_sections = any(kw in sections_str for kw in specific_religion_keywords)
    has_religion_cols = any(kw in col_str for kw in specific_religion_keywords)

    is_civil = ('civil condition' in title or 'civil condition' in col_str
                or 'dition and age' in title  # truncated OCR: "Con[dition and Age]"
                or 'marital' in title or 'marital' in col_str
                or any(x in col_str for x in ['unmarried', 'married', 'widowed', 'divorced']))
    is_life_table = 'life table' in title
    is_education = 'education' in title or 'literate' in title or 'literate' in col_str
    is_occupation = 'occupation' in title or 'industri' in title
    is_infirmity = any(kw in title for kw in ['infirmit', 'blind', 'deaf', 'leper', 'insane'])
    is_subsidiary = 'subsidiary' in title
    is_proportional = (data_type == 'proportional'
                       or 'proportion' in title or '10,000' in title
                       or '100,000' in title or 'per mille' in title
                       or 'per cent' in title)
    is_single_year = 'single year' in title
    is_age_returns = 'age return' in title or ('age' in title and 'group' in title)
    is_language = 'bilingual' in title or 'language' in title
    is_birth_death = 'birth' in title or 'death' in title or 'fertility' in title or 'mortality' in title

    # Primary category (broad)
    if is_life_table:
        primary = "LifeTable"
    elif is_infirmity:
        primary = "Infirmity"
    elif is_education:
        primary = "Education"
    elif is_occupation:
        primary = "Occupation"
    elif is_civil:
        primary = "AgeSexCivilCondition"
    elif is_single_year:
        primary = "SingleYearAge"
    elif is_language:
        primary = "Language"
    elif is_birth_death:
        primary = "BirthDeath"
    elif is_subsidiary:
        primary = "Subsidiary"
    elif is_proportional:
        primary = "Proportional"
    elif is_age_returns:
        primary = "AgeSex"
    else:
        primary = "AgeSex"

    # Sub-category
    if has_religion or has_religion_cols:
        sub = "Religion"
    elif is_proportional and primary not in ("Proportional",):
        sub = "Proportional"
    elif primary in ("Subsidiary",):
        if is_proportional:
            sub = "Proportional"
        elif has_religion or has_religion_sections:
            sub = "Religion"
        else:
            sub = "Other"
    else:
        sub = "General"

    return primary, sub


def detect_column_shape(meta, data_sections):
    """Return a short string describing the column structure."""
    col_groups = [c.strip('.').strip().upper() for c in meta.get('column_groups', [])]
    col_set = set(col_groups)

    if col_set >= {'POPULATION', 'UNMARRIED', 'MARRIED', 'WIDOWED'}:
        shape = "Full (Pop+Unm+Mar+Wid)"
    elif col_set == {'POPULATION', 'UNMARRIED'}:
        shape = "Left half (Pop+Unm)"
    elif col_set == {'MARRIED', 'WIDOWED'}:
        shape = "Right half (Mar+Wid)"
    elif 'POPULATION' in col_set or 'TOTAL' in col_set:
        shape = "Population only"
    else:
        shape = "Custom"

    # Sub-column structure
    if data_sections:
        rows = data_sections[0].get('rows', [])
        if rows:
            first_row = rows[0]
            for k, v in first_row.items():
                if isinstance(v, dict):
                    sub_keys = sorted(v.keys())
                    return shape, sub_keys
    return shape, []


def count_data_rows(data_sections):
    return sum(len(s.get('rows', [])) for s in data_sections)


def extract_age_groups(data_sections):
    """Return the list of age labels from the first section."""
    if not data_sections:
        return []
    rows = data_sections[0].get('rows', [])
    ages = []
    for r in rows:
        for k, v in r.items():
            if not isinstance(v, dict):
                ages.append(str(v) if v else '')
                break
    return ages


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def load_all_validated():
    """Load all validated JSONs and return parsed records."""
    jsons = glob.glob(os.path.join(ORGANIZED_ROOT, '**', '*_oneshot.json'), recursive=True)
    records = []

    for jf in sorted(jsons):
        with open(jf) as f:
            d = json.load(f)

        meta = d.get('metadata', {})
        constraints = d.get('constraints', {})
        total_checks = constraints.get('total_checks', 0)
        failed = constraints.get('failed', 0)
        is_validated = total_checks > 0 and failed == 0

        if not is_validated:
            continue

        rel = os.path.relpath(os.path.dirname(jf), ORGANIZED_ROOT)
        parts = rel.split('/')
        province = parts[0] if len(parts) >= 1 else 'Unknown'
        # Prefer metadata year (matches export_results.py behavior)
        meta_year = meta.get('year', 0)
        try:
            year = int(meta_year)
        except (ValueError, TypeError):
            year = 0
        if year == 0:
            year_str = parts[1] if len(parts) >= 2 else ''
            try:
                year = int(year_str)
            except ValueError:
                year = 0

        title = meta.get('title', '')
        region = (meta.get('region') or '').strip()
        data_sections = d.get('data', [])
        primary_cat, sub_cat = classify_table(meta, data_sections)
        col_shape, sub_cols = detect_column_shape(meta, data_sections)
        n_rows = count_data_rows(data_sections)
        n_sections = len(data_sections)
        section_names = [s['name'] for s in data_sections]
        age_groups = extract_age_groups(data_sections)
        col_groups = meta.get('column_groups', [])
        data_type = meta.get('data_type', '')
        acceptance = d.get('acceptance', {}).get('decision', '')
        trust = d.get('acceptance', {}).get('trust_score', '')

        # Find corresponding export file
        export_province_dir = os.path.join(EXPORT_ROOT, province, str(year))
        export_file = ''
        if os.path.isdir(export_province_dir):
            # Try to find matching file by looking at all xlsx files
            # (exact matching is complex due to slugification, just note the directory)
            export_file = f"export/{province}/{year}/"

        records.append({
            'province': province,
            'year': year,
            'region': region,
            'title': title,
            'primary_category': primary_cat,
            'sub_category': sub_cat,
            'full_category': f"{primary_cat} ({sub_cat})",
            'column_shape': col_shape,
            'sub_columns': ', '.join(sub_cols),
            'column_groups': ', '.join(col_groups),
            'n_rows': n_rows,
            'n_sections': n_sections,
            'section_names': ' | '.join(section_names),
            'data_type': data_type,
            'n_checks': total_checks,
            'acceptance': acceptance,
            'trust_score': trust,
            'age_groups_preview': ' | '.join(age_groups[:8]),
            'source_json': os.path.relpath(jf, ORGANIZED_ROOT),
            'export_dir': export_file,
        })

    return records


# ---------------------------------------------------------------------------
# Excel output
# ---------------------------------------------------------------------------

HEADER_FONT = Font(bold=True, size=11, color='FFFFFF')
HEADER_FILL = PatternFill(start_color='2F5496', end_color='2F5496', fill_type='solid')
SECTION_FILL = PatternFill(start_color='D6E4F0', end_color='D6E4F0', fill_type='solid')
SECTION_FONT = Font(bold=True, size=11, color='2F5496')
GAP_FILL = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
OK_FILL = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
WARN_FILL = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
THIN_BORDER = Border(
    left=Side(style='thin'), right=Side(style='thin'),
    top=Side(style='thin'), bottom=Side(style='thin'),
)


def style_header_row(ws, row, ncols):
    for ci in range(1, ncols + 1):
        cell = ws.cell(row=row, column=ci)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.border = THIN_BORDER
        cell.alignment = Alignment(horizontal='center', wrap_text=True)


def auto_width(ws, max_width=30):
    for col_cells in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col_cells[0].column)
        for cell in col_cells[:100]:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 2, max_width)


def write_summary_sheet(wb, records):
    ws = wb.active
    ws.title = "Summary"

    total = len(records)
    provinces = sorted(set(r['province'] for r in records))
    years = sorted(set(r['year'] for r in records))
    cat_counts = Counter(r['primary_category'] for r in records)
    sub_counts = Counter(r['full_category'] for r in records)

    # Title
    ws.merge_cells('A1:D1')
    title_cell = ws['A1']
    title_cell.value = "Census OCR Data Guide — Validated Tables"
    title_cell.font = Font(bold=True, size=16, color='2F5496')

    ws.merge_cells('A2:D2')
    ws['A2'].value = f"Generated from {total} validated tables across {len(provinces)} provinces, {len(years)} census years"
    ws['A2'].font = Font(size=11, italic=True, color='666666')

    row = 4
    ws.cell(row=row, column=1, value="Overview").font = Font(bold=True, size=13, color='2F5496')
    row += 1
    stats = [
        ("Total validated tables", total),
        ("Provinces/States", len(provinces)),
        ("Census years covered", f"{min(years)}–{max(years)} ({len(years)} years)"),
        ("Year list", ', '.join(str(y) for y in years)),
        ("", ""),
        ("Standardized CSV Exports", "csv/"),
        ("  Primary tables (wide format)", "csv/primary_age_civil_condition.csv"),
        ("  Religion tables (wide format)", "csv/religion_age_civil_condition.csv"),
        ("  All tables (long/tidy format)", "csv/all_tables_long.csv"),
        ("  Column definitions", "csv/codebook.txt"),
    ]
    for label, value in stats:
        ws.cell(row=row, column=1, value=label).font = Font(bold=True)
        ws.cell(row=row, column=2, value=value)
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Tables by Primary Category").font = Font(bold=True, size=13, color='2F5496')
    row += 1
    ws.cell(row=row, column=1, value="Category").font = SECTION_FONT
    ws.cell(row=row, column=2, value="Count").font = SECTION_FONT
    ws.cell(row=row, column=3, value="Description").font = SECTION_FONT
    row += 1

    descriptions = {
        'AgeSexCivilCondition': 'Age × Sex × Civil Condition (Population, Unmarried, Married, Widowed) — the primary census table',
        'AgeSex': 'Age × Sex population tables (without civil condition breakdown)',
        'Education': 'Literacy and education tables by age/sex/religion',
        'Proportional': 'Per-10,000 or per-cent distributions (derived, not raw counts)',
        'Subsidiary': 'Subsidiary analytical tables (smoothing, birth/death rates, etc.)',
        'SingleYearAge': 'Single-year-of-age returns (age 0, 1, 2, ... instead of grouped)',
        'Infirmity': 'Disability/infirmity tables (blind, deaf-mute, insane, lepers)',
        'LifeTable': 'Actuarial life tables (mortality rates, life expectancy)',
        'Occupation': 'Occupation/industry tables by age',
        'Language': 'Bilingualism and language tables',
        'BirthDeath': 'Birth, death, fertility, and mortality statistics',
    }
    for cat, cnt in cat_counts.most_common():
        ws.cell(row=row, column=1, value=cat)
        ws.cell(row=row, column=2, value=cnt)
        ws.cell(row=row, column=3, value=descriptions.get(cat, ''))
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Tables by Detailed Category").font = Font(bold=True, size=13, color='2F5496')
    row += 1
    ws.cell(row=row, column=1, value="Category").font = SECTION_FONT
    ws.cell(row=row, column=2, value="Count").font = SECTION_FONT
    row += 1
    for cat, cnt in sub_counts.most_common():
        ws.cell(row=row, column=1, value=cat)
        ws.cell(row=row, column=2, value=cnt)
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Tables by Province").font = Font(bold=True, size=13, color='2F5496')
    row += 1
    ws.cell(row=row, column=1, value="Province").font = SECTION_FONT
    ws.cell(row=row, column=2, value="Total").font = SECTION_FONT
    ws.cell(row=row, column=3, value="Primary (General)").font = SECTION_FONT
    ws.cell(row=row, column=4, value="Religion").font = SECTION_FONT
    ws.cell(row=row, column=5, value="Other").font = SECTION_FONT
    ws.cell(row=row, column=6, value="Years").font = SECTION_FONT
    row += 1

    for prov in provinces:
        prov_recs = [r for r in records if r['province'] == prov]
        general = sum(1 for r in prov_recs if r['sub_category'] == 'General')
        religion = sum(1 for r in prov_recs if r['sub_category'] == 'Religion')
        other = len(prov_recs) - general - religion
        prov_years = sorted(set(r['year'] for r in prov_recs))
        ws.cell(row=row, column=1, value=prov.replace('_', ' '))
        ws.cell(row=row, column=2, value=len(prov_recs))
        ws.cell(row=row, column=3, value=general)
        ws.cell(row=row, column=4, value=religion)
        ws.cell(row=row, column=5, value=other)
        ws.cell(row=row, column=6, value=', '.join(str(y) for y in prov_years))
        row += 1

    ws.column_dimensions['A'].width = 35
    ws.column_dimensions['B'].width = 15
    ws.column_dimensions['C'].width = 70
    ws.column_dimensions['D'].width = 15
    ws.column_dimensions['E'].width = 15
    ws.column_dimensions['F'].width = 40


def write_inventory_sheet(wb, records):
    ws = wb.create_sheet(title="Full Inventory")

    headers = [
        'Province', 'Year', 'Region', 'Primary Category', 'Sub-Category',
        'Column Shape', 'Sub-Columns', 'Column Groups', 'Rows', 'Sections',
        'Section Names', 'Data Type', 'Checks', 'Trust Score',
        'Age Groups (preview)', 'Title', 'Source JSON',
    ]

    for ci, h in enumerate(headers, 1):
        ws.cell(row=1, column=ci, value=h)
    style_header_row(ws, 1, len(headers))

    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

    for ri, rec in enumerate(records, 2):
        values = [
            rec['province'].replace('_', ' '),
            rec['year'],
            rec['region'],
            rec['primary_category'],
            rec['sub_category'],
            rec['column_shape'],
            rec['sub_columns'],
            rec['column_groups'],
            rec['n_rows'],
            rec['n_sections'],
            rec['section_names'],
            rec['data_type'],
            rec['n_checks'],
            rec['trust_score'],
            rec['age_groups_preview'],
            rec['title'],
            rec['source_json'],
        ]
        for ci, v in enumerate(values, 1):
            cell = ws.cell(row=ri, column=ci, value=v)
            cell.border = THIN_BORDER
            if isinstance(v, (int, float)) and v != '':
                cell.number_format = '#,##0'

    ws.freeze_panes = 'A2'
    auto_width(ws, max_width=40)


def write_primary_inventory(wb, records):
    """Separate sheet for just the primary AgeSexCivilCondition General tables."""
    ws = wb.create_sheet(title="Primary Tables")

    primary_recs = [r for r in records
                    if r['primary_category'] == 'AgeSexCivilCondition'
                    and r['sub_category'] == 'General']
    primary_recs.sort(key=lambda r: (r['province'], r['year'], r['region']))

    headers = [
        'Province', 'Year', 'Region/District', 'Column Shape',
        'Sub-Columns', 'Rows', 'Sections', 'Age Groups (preview)',
        'Checks', 'Trust', 'Title',
    ]

    for ci, h in enumerate(headers, 1):
        ws.cell(row=1, column=ci, value=h)
    style_header_row(ws, 1, len(headers))
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

    for ri, rec in enumerate(primary_recs, 2):
        values = [
            rec['province'].replace('_', ' '),
            rec['year'],
            rec['region'],
            rec['column_shape'],
            rec['sub_columns'],
            rec['n_rows'],
            rec['n_sections'],
            rec['age_groups_preview'],
            rec['n_checks'],
            rec['trust_score'],
            rec['title'],
        ]
        for ci, v in enumerate(values, 1):
            cell = ws.cell(row=ri, column=ci, value=v)
            cell.border = THIN_BORDER

    ws.freeze_panes = 'A2'
    auto_width(ws)
    ws.cell(row=len(primary_recs) + 3, column=1,
            value=f"Total primary tables: {len(primary_recs)}").font = Font(bold=True, size=11)


def write_religion_inventory(wb, records):
    ws = wb.create_sheet(title="Religion Tables")

    rel_recs = [r for r in records if r['sub_category'] == 'Religion']
    rel_recs.sort(key=lambda r: (r['province'], r['year'], r['region']))

    headers = [
        'Province', 'Year', 'Region/District', 'Primary Category',
        'Rows', 'Title',
    ]

    for ci, h in enumerate(headers, 1):
        ws.cell(row=1, column=ci, value=h)
    style_header_row(ws, 1, len(headers))
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

    for ri, rec in enumerate(rel_recs, 2):
        values = [
            rec['province'].replace('_', ' '),
            rec['year'],
            rec['region'],
            rec['primary_category'],
            rec['n_rows'],
            rec['title'],
        ]
        for ci, v in enumerate(values, 1):
            cell = ws.cell(row=ri, column=ci, value=v)
            cell.border = THIN_BORDER

    ws.freeze_panes = 'A2'
    auto_width(ws, max_width=50)
    ws.cell(row=len(rel_recs) + 3, column=1,
            value=f"Total religion tables: {len(rel_recs)}").font = Font(bold=True, size=11)


def write_coverage_sheet(wb, records):
    ws = wb.create_sheet(title="Coverage Matrix")

    provinces = sorted(set(r['province'] for r in records))
    all_years = sorted(set(r['year'] for r in records))
    # Filter to major census years
    census_years = [y for y in all_years if y >= 1871]

    # Build matrix: for each province×year, count primary general tables
    primary_recs = [r for r in records
                    if r['primary_category'] == 'AgeSexCivilCondition'
                    and r['sub_category'] == 'General']

    matrix = defaultdict(lambda: defaultdict(int))
    regions_matrix = defaultdict(lambda: defaultdict(set))
    for r in primary_recs:
        matrix[r['province']][r['year']] += 1
        if r['region']:
            regions_matrix[r['province']][r['year']].add(r['region'])

    # Also count ALL tables per province×year
    all_matrix = defaultdict(lambda: defaultdict(int))
    for r in records:
        all_matrix[r['province']][r['year']] += 1

    # Header row
    ws.cell(row=1, column=1, value="Province / Year")
    for ci, year in enumerate(census_years, 2):
        ws.cell(row=1, column=ci, value=year)
    style_header_row(ws, 1, len(census_years) + 1)

    for ri, prov in enumerate(provinces, 2):
        prov_display = prov.replace('_', ' ')
        ws.cell(row=ri, column=1, value=prov_display).font = Font(bold=True)
        ws.cell(row=ri, column=1).border = THIN_BORDER

        for ci, year in enumerate(census_years, 2):
            primary_count = matrix[prov][year]
            total_count = all_matrix[prov][year]
            n_regions = len(regions_matrix[prov][year])
            cell = ws.cell(row=ri, column=ci)
            cell.border = THIN_BORDER
            cell.alignment = Alignment(horizontal='center')

            if primary_count > 0:
                cell.value = f"{primary_count}p/{total_count}t"
                cell.fill = OK_FILL
            elif total_count > 0:
                cell.value = f"0p/{total_count}t"
                cell.fill = WARN_FILL
            else:
                cell.value = "—"
                cell.fill = GAP_FILL if year in [1881, 1891, 1901, 1911, 1921, 1931] else PatternFill()

    # Legend
    legend_row = len(provinces) + 3
    ws.cell(row=legend_row, column=1, value="Legend:").font = Font(bold=True)
    ws.cell(row=legend_row + 1, column=1, value="Np/Mt = N primary tables / M total tables")
    c = ws.cell(row=legend_row + 2, column=1, value="Green = has primary tables")
    c.fill = OK_FILL
    c = ws.cell(row=legend_row + 3, column=1, value="Yellow = has other tables but no primary")
    c.fill = WARN_FILL
    c = ws.cell(row=legend_row + 4, column=1, value="Red = no tables (gap in major census year)")
    c.fill = GAP_FILL

    ws.freeze_panes = 'B2'
    ws.column_dimensions['A'].width = 30
    for ci in range(2, len(census_years) + 2):
        ws.column_dimensions[get_column_letter(ci)].width = 12


def write_district_coverage(wb, records):
    """Show which regions/districts we have for each province × year (primary tables only)."""
    ws = wb.create_sheet(title="District Detail")

    primary_recs = [r for r in records
                    if r['primary_category'] == 'AgeSexCivilCondition'
                    and r['sub_category'] == 'General']
    primary_recs.sort(key=lambda r: (r['province'], r['year'], r['region']))

    # Group by province → year
    prov_year = defaultdict(lambda: defaultdict(list))
    for r in primary_recs:
        prov_year[r['province']][r['year']].append(r['region'])

    headers = ['Province', 'Year', '# Regions', 'Regions/Districts']
    for ci, h in enumerate(headers, 1):
        ws.cell(row=1, column=ci, value=h)
    style_header_row(ws, 1, len(headers))

    row = 2
    for prov in sorted(prov_year.keys()):
        for year in sorted(prov_year[prov].keys()):
            regions = prov_year[prov][year]
            unique_regions = sorted(set(regions))
            ws.cell(row=row, column=1, value=prov.replace('_', ' ')).border = THIN_BORDER
            ws.cell(row=row, column=2, value=year).border = THIN_BORDER
            ws.cell(row=row, column=3, value=len(unique_regions)).border = THIN_BORDER
            ws.cell(row=row, column=4, value=', '.join(
                r if r else '(unnamed)' for r in unique_regions
            )).border = THIN_BORDER
            row += 1

    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = f"A1:D1"
    ws.column_dimensions['A'].width = 30
    ws.column_dimensions['B'].width = 10
    ws.column_dimensions['C'].width = 12
    ws.column_dimensions['D'].width = 120


def write_format_notes(wb, records):
    ws = wb.create_sheet(title="Format Notes")

    primary_recs = [r for r in records
                    if r['primary_category'] == 'AgeSexCivilCondition'
                    and r['sub_category'] == 'General']

    ws.merge_cells('A1:D1')
    ws['A1'].value = "Format Consistency Analysis"
    ws['A1'].font = Font(bold=True, size=14, color='2F5496')

    row = 3
    ws.cell(row=row, column=1, value="Column Structure").font = Font(bold=True, size=12, color='2F5496')
    row += 1

    col_shapes = Counter(r['column_shape'] for r in primary_recs)
    for shape, cnt in col_shapes.most_common():
        ws.cell(row=row, column=1, value=shape)
        ws.cell(row=row, column=2, value=cnt)
        pct = cnt / len(primary_recs) * 100
        ws.cell(row=row, column=3, value=f"{pct:.1f}%")
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Sub-Column Structure").font = Font(bold=True, size=12, color='2F5496')
    row += 1
    sub_col_patterns = Counter(r['sub_columns'] for r in primary_recs)
    for pat, cnt in sub_col_patterns.most_common():
        ws.cell(row=row, column=1, value=pat if pat else '(none)')
        ws.cell(row=row, column=2, value=cnt)
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Row Counts (per first section)").font = Font(bold=True, size=12, color='2F5496')
    row += 1
    row_counts = Counter(r['n_rows'] for r in primary_recs)
    for rc, cnt in row_counts.most_common(10):
        ws.cell(row=row, column=1, value=f"{rc} rows")
        ws.cell(row=row, column=2, value=cnt)
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Age Group Patterns").font = Font(bold=True, size=12, color='2F5496')
    row += 1
    age_pats = Counter(r['age_groups_preview'] for r in primary_recs)
    for pat, cnt in age_pats.most_common(15):
        ws.cell(row=row, column=1, value=pat if pat else '(empty)')
        ws.cell(row=row, column=2, value=cnt)
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Data Science Notes").font = Font(bold=True, size=12, color='2F5496')
    row += 1
    n_primary = len(primary_recs)
    n_full = sum(1 for r in primary_recs if 'Full' in r['column_shape'])
    n_left = sum(1 for r in primary_recs if 'Left' in r['column_shape'])
    n_right = sum(1 for r in primary_recs if 'Right' in r['column_shape'])

    notes = [
        f"CONSISTENCY: {n_full}/{n_primary} primary tables ({n_full/n_primary*100:.0f}%) have the full 4-group column structure (Population, Unmarried, Married, Widowed).",
        f"SPLIT PAGES: {n_left + n_right} tables are split across two page scans — {n_left} have Population+Unmarried, {n_right} have Married+Widowed. These share the same province/year/region and can be joined.",
        "SUB-COLUMNS: Virtually all tables use {{persons, males, females}} as sub-columns within each column group. Highly consistent.",
        "AGE GROUPS: Age grouping varies by census year. Common patterns: (a) 0-1/1-2/.../4-5/0-5/5-10/... (single-year then 5-year), (b) 0-5/5-10/.../60+, (c) 0-10/10-15/.../60+. Harmonization needed for cross-year analysis.",
        "ROW LABELS: The age label key varies (AGE., Age., age, AGE, or unlabeled). The first non-dict column in each row is always the age label.",
        "SECTIONS: Tables have 1–28 sections. Multi-section tables often represent multi-district pages from a single scan.",
        "DATA TYPE: Most tables are 'absolute' (raw counts). Some are 'proportional' (per 10,000). Check the data_type field in the JSON.",
        "REGION NAMES: Not standardized — same district may appear as 'Ajmer', 'AJMER', 'Ajmer District', 'Ajmer district'. Normalize before joining.",
        "VALIDATION: All tables in this guide passed arithmetic constraint checks (M+F=Persons, vertical sums). 283 additional unvalidated tables exist (0 checks — table structure didn't have verifiable constraints).",
        "SOURCE JSONs: The source_json column in the inventory links to the full JSON with complete data, metadata, constraint details, and source image paths.",
    ]
    for note in notes:
        ws.cell(row=row, column=1, value=note)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
        row += 1

    ws.column_dimensions['A'].width = 50
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 12


def main():
    parser = argparse.ArgumentParser(description="Build the Census OCR data guide workbook.")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help=f"Output workbook path (default: {OUTPUT_PATH})")
    parser.add_argument("--gap-json", default=None,
                        help="Optional gap_candidates.json path to append gap fillability sheets")
    args = parser.parse_args()

    print("Loading validated JSONs...")
    records = load_all_validated()
    print(f"  {len(records)} validated tables loaded")

    print("Building workbook...")
    wb = Workbook()

    write_summary_sheet(wb, records)
    summary_ws = wb["Summary"]
    write_inventory_sheet(wb, records)
    write_primary_inventory(wb, records)
    write_religion_inventory(wb, records)
    write_coverage_sheet(wb, records)
    write_district_coverage(wb, records)
    write_format_notes(wb, records)

    gap_json = args.gap_json or (DEFAULT_GAP_JSON if os.path.exists(DEFAULT_GAP_JSON) else None)
    if gap_json:
        print(f"Loading gap fillability report from {gap_json}...")
        gap_rows = load_gap_rows(gap_json)
        provinces, total_counts, primary_counts = coverage_from_records(records)
        append_summary_section(summary_ws, gap_rows)
        write_gap_matrix_sheet(
            wb,
            provinces,
            total_counts,
            primary_counts,
            gap_rows,
            title="Gap Fillability Matrix",
        )
        write_gap_detail_sheet(wb, gap_rows, title="Gap Fillability Detail")
        print(f"  Added gap fillability sheets for {len(gap_rows)} gaps")

    output_path = os.path.expanduser(args.output)
    print(f"Saving to {output_path}...")
    wb.save(output_path)
    print("Done!")

    # Print quick summary
    cats = Counter(r['primary_category'] for r in records)
    print(f"\n=== Quick Summary ===")
    print(f"Total validated: {len(records)}")
    for cat, cnt in cats.most_common():
        print(f"  {cat}: {cnt}")

    primary = [r for r in records
               if r['primary_category'] == 'AgeSexCivilCondition'
               and r['sub_category'] == 'General']
    print(f"\nPrimary tables (AgeSexCivilCondition, General): {len(primary)}")
    provs = sorted(set(r['province'] for r in primary))
    print(f"  Across {len(provs)} provinces")


if __name__ == '__main__':
    main()
