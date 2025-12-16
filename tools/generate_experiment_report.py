"""
Simple experiment results summarizer.
Writes SUMMARY_REPORT.md into the target directory.

Usage (PowerShell):
    python .\tools\generate_experiment_report.py --dir programmingAssignment2\results\run_20251103_013059

"""
import os
import argparse
import json
import csv
from datetime import datetime


def safe_load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {'_load_error': str(e)}


def read_csv_head(path, n=10):
    rows = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                rows.append(row)
                if i+1 >= n:
                    break
    except Exception as e:
        return {'_load_error': str(e)}
    return rows


def main(target_dir):
    report_lines = []
    report_lines.append(f"# Experiment run summary\nGenerated: {datetime.utcnow().isoformat()} UTC\n")

    if not os.path.isdir(target_dir):
        report_lines.append(f"Directory not found: `{target_dir}`\n")
        out = os.path.join(target_dir, 'SUMMARY_REPORT.md')
        os.makedirs(target_dir, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"Wrote empty report to {out}")
        return

    files = sorted(os.listdir(target_dir))
    report_lines.append(f"## Files in `{target_dir}` ({len(files)})\n")

    json_files = [f for f in files if f.lower().endswith('.json')]
    csv_files = [f for f in files if f.lower().endswith('.csv')]
    png_files = [f for f in files if f.lower().endswith('.png') or f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    txt_files = [f for f in files if f.lower().endswith('.txt')]
    other = [f for f in files if f not in json_files + csv_files + png_files + txt_files]

    report_lines.append(f"- JSON files: {len(json_files)}")
    report_lines.append(f"- CSV files: {len(csv_files)}")
    report_lines.append(f"- Images: {len(png_files)}")
    report_lines.append(f"- Text logs: {len(txt_files)}")
    report_lines.append(f"- Other: {len(other)}\n")

    if json_files:
        report_lines.append('\n---\n## JSON summaries')
        for jf in json_files:
            path = os.path.join(target_dir, jf)
            data = safe_load_json(path)
            report_lines.append(f"\n### {jf}")
            report_lines.append('```json')
            try:
                report_lines.append(json.dumps(data, indent=2, default=str))
            except Exception:
                report_lines.append(str(data))
            report_lines.append('```')

    if csv_files:
        report_lines.append('\n---\n## CSV previews')
        for cf in csv_files:
            path = os.path.join(target_dir, cf)
            rows = read_csv_head(path, n=10)
            report_lines.append(f"\n### {cf}")
            if isinstance(rows, dict) and rows.get('_load_error'):
                report_lines.append(f"Could not read CSV: {rows['_load_error']}")
            else:
                report_lines.append('```')
                for r in rows:
                    report_lines.append(','.join(map(str, r)))
                report_lines.append('```')

    if png_files:
        report_lines.append('\n---\n## Images')
        for p in png_files:
            report_lines.append(f"- {p}")

    if txt_files:
        report_lines.append('\n---\n## Logs')
        for t in txt_files:
            report_lines.append(f"- {t}")

    if other:
        report_lines.append('\n---\n## Other files')
        for o in other:
            report_lines.append(f"- {o}")

    out_path = os.path.join(target_dir, 'SUMMARY_REPORT.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"Wrote report to {out_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='experiment_results', help='target results directory')
    args = p.parse_args()
    main(args.dir)
