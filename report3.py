import argparse, glob, json
from common import load_cfg

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
rows = [r for r in (json.load(open(f)) for f in sorted(glob.glob(f"{cfg['out_dir']}/*.json"))) if 'sensors' in r]
keys = list(rows[0]['all']['hit1'])
conds = [c for c in ('day', 'night', 'rain') if c in rows[0]]
head = ['run', 'step', 'mean'] + keys + [f'mean ({c})' for c in conds]
lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
for r in rows:
    vals = [r['name'], r['step'], r['all']['hit1_mean']] + [r['all']['hit1'][k] for k in keys] + [r[c]['hit1_mean'] for c in conds]
    lines.append('| ' + ' | '.join(str(v) for v in vals) + ' |')
lines.append(f"\nhit@1 a→b: a token of sensor a finds the same spot in sensor b (within 1 token). Chance ≈ {rows[0]['chance_hit1']}.")
open(f"{cfg['out_dir']}/table.md", 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
