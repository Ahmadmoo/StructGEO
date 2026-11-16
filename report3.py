import argparse, glob, json
from common import load_cfg

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
rows = [r for r in (json.load(open(f)) for f in sorted(glob.glob(f"{cfg['out_dir']}/*.json"))) if 'sensors' in r]
keys = list(dict.fromkeys(k for r in rows for k in r['all']['hit1']))
conds = [c for c, v in rows[0].items() if c != 'all' and isinstance(v, dict) and 'hit1_mean' in v]
head = ['run', 'step', 'mean'] + keys + [f'mean ({c})' for c in conds]
lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
for r in rows:
    vals = [r['name'], r['step'], r['all']['hit1_mean']] + [r['all']['hit1'].get(k) for k in keys] + [r.get(c, {}).get('hit1_mean') for c in conds]
    lines.append('| ' + ' | '.join(str(v) for v in vals) + ' |')
lines.append(f"\nhit@1 a→b: a token of sensor a finds the same spot in sensor b (within 1 token). Chance ≈ {rows[0]['chance_hit1']}.")
open(f"{cfg['out_dir']}/table.md", 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
