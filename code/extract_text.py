"""Extract plain text from downloaded BAI reports (pdf / hwp / zip of those) into data/text/<srno>.txt.
Usage: python3 -I extract_text.py <download_dir>
"""
import os, re, sys, subprocess, zipfile, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hwp5_text import extract as hwp_extract

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'text')
os.makedirs(OUT, exist_ok=True)


def pdf_text(path):
    r = subprocess.run(['pdftotext', '-enc', 'UTF-8', path, '-'], capture_output=True)
    return r.stdout.decode('utf-8', errors='ignore')


def file_text(path):
    ext = path.lower().rsplit('.', 1)[-1]
    if ext == 'pdf':
        return pdf_text(path)
    if ext == 'hwp':
        try:
            return hwp_extract(path)
        except Exception as e:
            return ''
    if ext == 'zip':
        out = []
        with tempfile.TemporaryDirectory() as td:
            try:
                with zipfile.ZipFile(path) as z:
                    z.extractall(td)
            except Exception:
                return ''
            for root, _, files in os.walk(td):
                for f in sorted(files):
                    if f.lower().endswith(('.pdf', '.hwp')):
                        out.append(file_text(os.path.join(root, f)))
        return '\n'.join(out)
    return ''


def main(dl):
    files = sorted(f for f in os.listdir(dl) if f.startswith('bai_'))
    by_srno = {}
    for f in files:
        m = re.match(r'bai_(\d+)_', f)
        if not m:
            continue
        by_srno.setdefault(m.group(1), []).append(os.path.join(dl, f))
    stats = {}
    for srno, paths in by_srno.items():
        txt = '\n'.join(file_text(p) for p in paths)
        txt = re.sub(r'\x00', '', txt)
        with open(os.path.join(OUT, f'{srno}.txt'), 'w', encoding='utf-8') as fh:
            fh.write(txt)
        stats[srno] = {'files': [os.path.basename(p) for p in paths], 'chars': len(txt), 'hangul': len(re.findall(r'[가-힣]', txt))}
    with open(os.path.join(OUT, '_stats.json'), 'w', encoding='utf-8') as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=1)
    n = len(stats)
    low = [s for s, v in stats.items() if v['hangul'] < 1000]
    print(f'docs={n} low_text={len(low)} {low[:20]}')


if __name__ == '__main__':
    main(sys.argv[1])
