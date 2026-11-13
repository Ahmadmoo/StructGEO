"""Figures for the PoC experiments (matplotlib, PNG for slides)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

PLAIN = FuncFormatter(lambda v, _: f'{v:g}')

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2de', '#fcfcfb'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']                    # fixed categorical order: blue, orange, aqua
GOOD, BAD = '#0ca30c', '#d03b3b'
BLUES = LinearSegmentedColormap.from_list('blues', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95',
                                                    '#0d366b'])


def style():
    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                         'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11,
                         'axes.titlesize': 12, 'legend.frameon': False})


def fusion_curve(path, ks, series, title):
    """series: {label: dict(r25=[...], median=[...])} over ks (None = no frames). Two panels: share within 25 m, median error."""
    style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for j, (lab, v) in enumerate(series.items()):
        c = SERIES[j % len(SERIES)]
        for a, key, scale in ((ax[0], 'r25', 100), (ax[1], 'median', 1)):
            y = [np.nan if x is None else x * scale for x in v[key]]
            a.plot(ks, y, color=c, lw=2, marker='o', ms=8, mec=SURF, mew=2, label=lab)
            if np.isfinite(y[-1]):
                a.annotate(f'{y[-1]:.0f}{"%" if key == "r25" else " m"}', (ks[-1], y[-1]), xytext=(6, 0),
                           textcoords='offset points', va='center', color=INK2, fontsize=10)
    ax[0].set(xlabel='frames fused (k)', ylabel='frames within 25 m (%)', ylim=(0, 100), xscale='log')
    ax[1].set(xlabel='frames fused (k)', ylabel='median error (m)', xscale='log', yscale='log')
    ys = [x for v in series.values() for x in v['median'] if x]
    if ys:
        lo, hi = max(0.5, min(ys) / 1.5), max(ys) * 1.5
        ax[1].set_ylim(lo, hi)
        ax[1].yaxis.set_major_locator(FixedLocator([t for t in (1, 2, 5, 10, 20, 50, 100, 200) if lo <= t <= hi]))
        ax[1].yaxis.set_major_formatter(PLAIN)
        ax[1].yaxis.set_minor_locator(NullLocator())
    for a in ax:
        a.set_xticks(ks)
        a.set_xticklabels([str(k) for k in ks])
        a.xaxis.set_minor_locator(NullLocator())
    if len(series) > 1:
        ax[0].legend(loc='lower right')
    fig.suptitle(title, x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def stanford(path, panels, al):
    """panels: list of (title, err, pl). Error vs protection level, log-log, counts per bin; alert limit al.
    Regions: nominal (err <= PL <= AL), misleading (PL < err), hazardous (PL <= AL < err), unavailable (PL > AL)."""
    style()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), squeeze=False)
    lo, hi = 0.5, 400
    bins = np.logspace(np.log10(lo), np.log10(hi), 45)
    for a, (title, e, p) in zip(axes[0], panels):
        e, p = np.clip(np.asarray(e, float), lo, hi), np.clip(np.asarray(p, float), lo, hi)
        h = a.hist2d(e, p, bins=[bins, bins], cmap=BLUES, norm=LogNorm(vmin=1), cmin=1)
        a.plot([lo, hi], [lo, hi], color=INK2, lw=1)
        a.axvline(al, color=INK2, lw=1, ls='--')
        a.axhline(al, color=INK2, lw=1, ls='--')
        a.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi), xlabel='position error (m)',
              ylabel='protection level (m)', title=title)
        for ax_ in (a.xaxis, a.yaxis):
            ax_.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
            ax_.set_major_formatter(PLAIN)
            ax_.set_minor_locator(NullLocator())
        n = len(e)
        reg = dict(nominal=((e <= p) & (p <= al)).sum(), misleading=((e > p) & (p <= al) & (e <= al)).sum(),
                   hazardous=((e > al) & (p <= al)).sum(), unavailable=(p > al).sum())
        a.text(0.03, 0.97, f"nominal {reg['nominal'] / n:.0%}\nunavailable {reg['unavailable'] / n:.0%}",
               transform=a.transAxes, va='top', fontsize=9, color=INK2)
        a.text(0.97, 0.03, f"misleading {reg['misleading'] / n:.1%}\nhazardous {reg['hazardous'] / n:.1%}",
               transform=a.transAxes, va='bottom', ha='right', fontsize=9, color=INK2)
        cb = fig.colorbar(h[3], ax=a, label='frames per cell', shrink=0.8)
        cb.ax.yaxis.set_major_formatter(PLAIN)
        cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def loc_map(path, score, extent, bg=None, pts=None, title='', label='map distinctiveness'):
    """score (H, W) grid over extent (E0, E1, N0, N1); bg: RGB image of the same extent; pts: (E, N, ok) frames."""
    style()
    fig, ax = plt.subplots(1, 2 if bg is not None else 1, figsize=(13 if bg is not None else 7, 5.4), squeeze=False)
    ax = ax[0]
    if bg is not None:
        ax[0].imshow(bg, extent=extent, origin='upper')
        ax[0].set_title('map (OpenStreetMap layers)')
    a = ax[-1]
    im = a.imshow(np.ma.masked_invalid(score), extent=extent, origin='upper', cmap=BLUES, vmin=0,
                  vmax=np.nanpercentile(score, 99) if np.isfinite(score).any() else 1)
    fig.colorbar(im, ax=a, label=label, shrink=0.85)
    a.set_title(title)
    if pts is not None and len(pts):
        e, n, ok = (np.asarray(x) for x in pts)
        for sel, col, lab in ((ok, GOOD, 'matched (≤ 25 m)'), (~ok, BAD, 'failed (> 25 m)')):
            a.scatter(e[sel], n[sel], s=6, c=col, lw=0, label=lab)
        a.legend(loc='lower left', markerscale=3, fontsize=9)
    for x in ax:
        x.set(xlabel='easting (m)', ylabel='northing (m)')
        x.grid(False)
        x.ticklabel_format(useOffset=False, style='plain')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def binned(path, score, fail, nbins, title, xlabel):
    """Failure rate per score bin (equal-count bins)."""
    style()
    score, fail = np.asarray(score, float), np.asarray(fail, bool)
    edges = np.unique(np.quantile(score, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(score, edges[1:-1]), 0, len(edges) - 2)
    rate = [fail[idx == j].mean() * 100 if (idx == j).any() else np.nan for j in range(len(edges) - 1)]
    fig, a = plt.subplots(figsize=(5.6, 3.6))
    x = np.arange(len(rate))
    a.bar(x, rate, color=SERIES[0], width=0.8)
    for xi, r in zip(x, rate):
        if np.isfinite(r):
            a.text(xi, r + 1, f'{r:.0f}%', ha='center', va='bottom', fontsize=9, color=INK2)
    a.set_xticks(x)
    a.set_xticklabels([f'{edges[j]:.2f}–{edges[j + 1]:.2f}' for j in range(len(rate))], rotation=30, ha='right', fontsize=9)
    a.set(ylabel='frames failing (> 25 m), %', xlabel=xlabel, title=title, ylim=(0, 100))
    a.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
