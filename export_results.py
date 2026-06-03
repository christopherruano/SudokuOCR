"""Export organized OCR results as prettified, intuitively-named Excel files.

Creates an export directory with:
  {Province}/{Year}/{Province}_{Year}_{Region}_{TableType}_{NNN}.xlsx

Each Excel file has:
  - Formatted headers with bold + borders
  - Auto-sized columns
  - A metadata sheet with source info and constraint status
  - Conditional formatting: green if validated, yellow if unvalidated
"""

import json
import os
import re
import sys
import shutil
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:
    print("pip install openpyxl")
    sys.exit(1)


def slugify(text, max_len=60):
    if not text:
        return ""
    text = re.sub(r'[^\w\s\-.]', '', text)
    text = re.sub(r'[\s\-_.]+', '_', text).strip('_')
    text = '_'.join(w.capitalize() for w in text.split('_'))
    return text[:max_len]


def extract_region_short(metadata, data):
    region = metadata.get('region', '') or ''
    if not region and data:
        region = data[0].get('name', '') or ''
    region = re.sub(r'(?i)^(table\s+\w+[\.\s]*[-–—]?\s*)', '', region)
    region = re.sub(r'(?i)(age,?\s*sex\s*(and)?\s*civil\s*condition\.?\s*[-–—]?\s*)', '', region)
    return region.strip().strip('.-– ')


def extract_table_type(metadata):
    title = (metadata.get('title') or '').lower()
    groups = metadata.get('column_groups', [])
    group_str = ' '.join(g.lower() for g in groups)

    if 'civil condition' in title or 'civil condition' in group_str:
        if 'unmarried' in group_str or 'married' in group_str:
            return 'AgeSexCivilCondition'
    if 'marital' in title:
        return 'AgeSexMaritalStatus'
    if 'population' in group_str and ('male' in group_str or 'female' in group_str):
        return 'AgeSexPopulation'
    if 'age' in title and 'sex' in title:
        return 'AgeSex'
    if 'life table' in title:
        return 'LifeTable'
    if 'single year' in title:
        return 'SingleYearAge'
    return 'AgeTable'


def make_pretty_xlsx(data_dict, output_path):
    wb = Workbook()

    meta = data_dict.get('metadata', {})
    constraints = data_dict.get('constraints', {})
    acceptance = data_dict.get('acceptance', {})
    sections = data_dict.get('data', [])

    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
    header_font_white = Font(bold=True, size=11, color='FFFFFF')
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin'),
    )
    number_fmt = '#,##0'

    checks = constraints.get('total_checks', 0)
    failed = constraints.get('failed', 0)
    validated = checks > 0 and failed == 0

    for si, section in enumerate(sections):
        section_name = section.get('name', f'Section {si+1}')
        sheet_name = slugify(section_name, 31) or f'Section_{si+1}'
        if si == 0:
            ws = wb.active
            ws.title = sheet_name
        else:
            ws = wb.create_sheet(title=sheet_name)

        rows = section.get('rows', [])
        if not rows:
            ws['A1'] = 'No data'
            continue

        first_row = rows[0]
        row_label_key = None
        group_keys = []
        for k, v in first_row.items():
            if not isinstance(v, dict):
                row_label_key = k
            else:
                group_keys.append(k)

        headers = [row_label_key or 'Label']
        sub_headers = ['']
        for gk in group_keys:
            sample_val = first_row.get(gk, {})
            sub_cols = list(sample_val.keys()) if isinstance(sample_val, dict) else []
            if sub_cols:
                headers.append(gk)
                headers.extend([''] * (len(sub_cols) - 1))
                sub_headers.extend(sub_cols)
            else:
                headers.append(gk)
                sub_headers.append('')

        for ci, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=ci, value=h)
            cell.font = header_font_white
            cell.fill = header_fill
            cell.border = thin_border
            cell.alignment = Alignment(horizontal='center')

        has_sub = any(s for s in sub_headers)
        if has_sub:
            for ci, sh in enumerate(sub_headers, 1):
                cell = ws.cell(row=2, column=ci, value=sh.capitalize() if sh else '')
                cell.font = header_font
                cell.border = thin_border
                cell.alignment = Alignment(horizontal='center')
            data_start_row = 3
        else:
            data_start_row = 2

        for ri, row in enumerate(rows, data_start_row):
            col = 1
            label = row.get(row_label_key, '') if row_label_key else ''
            if isinstance(label, dict):
                label = str(label)
            cell = ws.cell(row=ri, column=col, value=label)
            cell.border = thin_border
            if label and ('total' in str(label).lower() or 'all age' in str(label).lower()):
                cell.font = Font(bold=True)
            col += 1

            for gk in group_keys:
                val = row.get(gk, {})
                if isinstance(val, dict):
                    for sk in (list(first_row.get(gk, {}).keys()) if isinstance(first_row.get(gk), dict) else []):
                        v = val.get(sk)
                        if isinstance(v, (dict, list)):
                            v = str(v)
                        cell = ws.cell(row=ri, column=col, value=v)
                        cell.border = thin_border
                        if isinstance(v, (int, float)):
                            cell.number_format = number_fmt
                            cell.alignment = Alignment(horizontal='right')
                        col += 1
                else:
                    if isinstance(val, (dict, list)):
                        val = str(val)
                    cell = ws.cell(row=ri, column=col, value=val)
                    cell.border = thin_border
                    col += 1

        for ci in range(1, len(headers) + 1):
            max_width = 10
            for row_cells in ws.iter_rows(min_col=ci, max_col=ci, max_row=min(len(rows) + 3, 50)):
                for cell in row_cells:
                    if cell.value:
                        max_width = max(max_width, len(str(cell.value)) + 2)
            ws.column_dimensions[get_column_letter(ci)].width = min(max_width, 20)

    ws_meta = wb.create_sheet(title='_Metadata')
    meta_rows = [
        ('Source', data_dict.get('source_image', '')),
        ('Title', meta.get('title', '')),
        ('Region', meta.get('region', '')),
        ('Year', meta.get('year', '')),
        ('Data Type', meta.get('data_type', '')),
        ('Column Groups', ', '.join(meta.get('column_groups', []))),
        ('Extraction Method', data_dict.get('extraction_method', '')),
        ('API Calls', data_dict.get('api_calls', '')),
        ('', ''),
        ('Constraint Checks', checks),
        ('Constraints Failed', failed),
        ('Status', 'VALIDATED' if validated else 'UNVALIDATED' if checks == 0 else f'FAILED ({failed} failures)'),
        ('Acceptance', acceptance.get('decision', '')),
    ]

    status_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid') if validated else \
                  PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')

    for ri, (label, value) in enumerate(meta_rows, 1):
        ws_meta.cell(row=ri, column=1, value=label).font = Font(bold=True)
        cell = ws_meta.cell(row=ri, column=2, value=value)
        if label == 'Status':
            cell.fill = status_fill
            cell.font = Font(bold=True)

    ws_meta.column_dimensions['A'].width = 20
    ws_meta.column_dimensions['B'].width = 80

    wb.save(output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export organized results as pretty, renamed Excel files')
    parser.add_argument('--input', default=os.path.expanduser('~/Research/SudokuOCR/organized_results'),
                        help='Organized results root')
    parser.add_argument('--output', default=os.path.expanduser('~/Research/SudokuOCR/export'),
                        help='Export directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    args = parser.parse_args()

    root = args.input
    export_root = args.output

    seen_names = {}
    exported = 0
    skipped = 0
    errors = 0

    for p, ds, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if not f.endswith('_oneshot.json'):
                continue

            fp = os.path.join(p, f)
            try:
                d = json.load(open(fp))
            except (json.JSONDecodeError, OSError):
                errors += 1
                continue

            meta = d.get('metadata', {})
            rel = os.path.relpath(p, root)
            parts = rel.split('/')
            province = parts[0] if len(parts) >= 1 else 'Unknown'
            year = str(meta.get('year', 0))
            if year == '0' or year == 'None':
                year = parts[1] if len(parts) >= 2 else 'Unknown'

            region_short = slugify(extract_region_short(meta, d.get('data', [])), 40)
            table_type = extract_table_type(meta)

            base_name = f"{province}_{year}_{table_type}"
            if region_short:
                base_name += f"_{region_short}"

            key = f"{province}/{year}/{base_name}"
            count = seen_names.get(key, 0) + 1
            seen_names[key] = count
            if count > 1:
                base_name += f"_{count:03d}"

            out_dir = os.path.join(export_root, province, year)
            out_path = os.path.join(out_dir, f"{base_name}.xlsx")

            if args.dry_run:
                print(f"  {os.path.relpath(out_path, export_root)}")
                exported += 1
                continue

            os.makedirs(out_dir, exist_ok=True)
            try:
                make_pretty_xlsx(d, out_path)
                exported += 1
            except Exception as e:
                print(f"  ERROR: {base_name}: {e}")
                errors += 1

    print(f"\nExported: {exported}")
    print(f"Errors: {errors}")
    if not args.dry_run:
        print(f"Output: {export_root}")


if __name__ == '__main__':
    main()
