import argparse, glob, json
from common import load_cfg

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{cfg['out_dir']}/*.json"))]
conds = sorted({k for r in rows for k, v in r.items() if isinstance(v, dict) and k != 'all'})
head = ['run', 'step', 'hit@1 T→RGB', 'hit@1 RGB→T', 'med. rank T→RGB', 'med. rank RGB→T'] + [f'hit@1 T→RGB ({c})' for c in conds]
lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
for r in rows:
    a = r['all']
    vals = [r['name'], r['step'], a['hit1_th2rgb'], a['hit1_rgb2th'], a['medrank_th2rgb'], a['medrank_rgb2th']]
    lines.append('| ' + ' | '.join(str(v) for v in vals + [r.get(c, {}).get('hit1_th2rgb', '') for c in conds]) + ' |')
if rows:
    lines.append(f"\nChance hit@1 ≈ {rows[0]['chance_hit1']} (a hit = best match within 1 token of the true spot).")
open(f"{cfg['out_dir']}/table.md", 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
