"""
F1 Race Winner Prediction - Visualization Module
Draws the README charts in docs/ from the test predictions and metrics
written by model_trainer.py
"""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# Colors follow the method, never its rank: model, pole sitter, championship leader
COLORS = {'model': '#2a78d6', 'pole': '#eb6834', 'leader': '#1baf7a'}
LABELS = {'model': 'Model', 'pole': 'Pole sitter wins', 'leader': 'Championship leader wins'}
INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_SOFT, 'axes.titlecolor': INK,
    'xtick.color': INK_SOFT, 'ytick.color': INK_SOFT, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'font.size': 10, 'axes.titlesize': 11,
    'axes.titleweight': 'bold', 'legend.frameon': False,
})


def _save(fig, name):
    path = os.path.join(config.DOCS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_by_season(metrics):
    """Winner hit rate and log loss per test season, model against baselines."""
    seasons = [str(s) for s in config.TEST_SEASONS]
    labels = [s if int(s) in config.FULL_TEST_SEASONS else f'{s}\n(partial)' for s in seasons]
    m = metrics[metrics['season'].isin(seasons)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [3, 2]})
    x = np.arange(len(seasons))
    width = 0.26
    for i, method in enumerate(['model', 'pole', 'leader']):
        vals = [100 * m.loc[(m['season'] == s) & (m['method'] == method), 'winner_hit'].iloc[0]
                for s in seasons]
        ax1.bar(x + (i - 1) * width, vals, width * 0.92, color=COLORS[method], label=LABELS[method])
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Races where the top pick won (%)')
    ax1.set_title('Race-winner hit rate')
    ax1.grid(axis='x', visible=False)
    ax1.legend(loc='upper right', fontsize=9)

    for i, method in enumerate(['model', 'pole']):
        vals = [m.loc[(m['season'] == s) & (m['method'] == method), 'log_loss'].iloc[0]
                for s in seasons]
        ax2.bar(x + (i - 0.5) * width, vals, width * 0.92, color=COLORS[method], label=LABELS[method])
    ax2.set_xticks(x, labels)
    ax2.set_ylabel('Mean -log p(actual winner)')
    ax2.set_title('Log loss (lower is better)')
    ax2.grid(axis='x', visible=False)
    fig.text(0.01, -0.04, 'Model fit on 2014-2022 only. Pole sitter log loss uses grid-slot '
             'win rates from the same training seasons.', fontsize=8.5, color=INK_SOFT)
    fig.tight_layout()
    _save(fig, 'results_by_season.png')


def plot_season_heatmap(preds, season):
    """Model win probability for each race of one season, actual winner marked."""
    s = preds[preds['season'] == season]
    rounds = sorted(s['round'].unique())
    winners = set(s.loc[s['won'] == 1, 'driver_code'])
    peak = s.groupby('driver_code')['p_model'].max()
    drivers = sorted(set(peak[peak >= 0.10].index) | winners,
                     key=lambda d: -s.loc[s['driver_code'] == d, 'p_model'].sum())

    grid = s.pivot_table(index='driver_code', columns='round', values='p_model').reindex(drivers)
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(drivers) + 2.2))
    im = ax.imshow(grid.to_numpy() * 100, cmap='Blues', vmin=0, vmax=80, aspect='auto')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    names = s.groupby('round')['race_name'].first().str.replace(' Grand Prix', '')
    ax.set_xticks(range(len(rounds)), [f'{r} {names[r]}' for r in rounds],
                  rotation=60, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(drivers)), drivers)

    hits = 0
    for j, r in enumerate(rounds):
        race = s[s['round'] == r]
        winner = race.loc[race['won'] == 1].iloc[0]
        pick = race.sort_values(['p_model', 'grid_pos'], ascending=[False, True]).iloc[0]
        hits += int(pick['driver_code'] == winner['driver_code'])
        if winner['driver_code'] in drivers:
            i = drivers.index(winner['driver_code'])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=INK, linewidth=1.6))
            p = winner['p_model'] * 100
            ax.text(j, i, f'{p:.0f}', ha='center', va='center', fontsize=7.5,
                    color='white' if p > 45 else INK)
        if pick['driver_code'] in drivers:
            i = drivers.index(pick['driver_code'])
            ax.plot(j + 0.3, i - 0.27, marker='o', markersize=4, color=INK,
                    markeredgecolor='white', markeredgewidth=0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label('Model win probability (%)', color=INK_SOFT)
    cbar.outline.set_visible(False)
    ax.set_title(f'{season} test season: model win probability by race', loc='left', pad=24)
    ax.text(0, 1.015, 'Box = actual winner, with the model\'s probability for them. '
            f'Dot = model\'s top pick. Top pick won {hits} of {len(rounds)} races.',
            transform=ax.transAxes, fontsize=9, color=INK_SOFT, va='bottom')
    fig.tight_layout()
    _save(fig, f'win_probability_{season}.png')


def plot_calibration(preds):
    """Predicted win probability against how often those drivers won."""
    data = preds[preds['season'].isin(config.FULL_TEST_SEASONS)]
    edges = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0001]

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=INK_SOFT, linewidth=1, label='Perfect calibration')
    for method, col in [('model', 'p_model'), ('pole', 'p_grid')]:
        bins = pd.cut(data[col], edges, right=False)
        g = data.groupby(bins, observed=True).agg(p=(col, 'mean'), won=('won', 'mean'),
                                                  n=('won', 'size'))
        ax.plot(g['p'], g['won'], marker='o', markersize=7, linewidth=2,
                color=COLORS[method],
                label='Model' if method == 'model' else 'Grid-slot win rates (pole baseline)')
        if method == 'model':
            print('Model calibration bins (mean predicted, share won, rows):')
            print(g.round(3).to_string())
            # Label only the upper bins; the low bins hold most rows and sit on top of each other
            for _, row in g[g['p'] >= 0.1].iterrows():
                ax.annotate(f'n={int(row["n"])}', (row['p'], row['won']), fontsize=7.5,
                            color=INK_SOFT, xytext=(-10, 7), textcoords='offset points', ha='right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted win probability (binned)')
    ax.set_ylabel('Share of those drivers who won')
    first, last = config.FULL_TEST_SEASONS[0], config.FULL_TEST_SEASONS[-1]
    ax.set_title(f'Calibration on {first}-{last} test races')
    ax.legend(loc='upper left', fontsize=8.5)
    fig.tight_layout()
    _save(fig, 'calibration.png')


def create_all():
    metrics = pd.read_csv(config.METRICS_FILE, dtype={'season': str})
    preds = pd.read_csv(config.PREDICTIONS_FILE)
    plot_by_season(metrics)
    plot_season_heatmap(preds, config.FULL_TEST_SEASONS[-1])
    plot_calibration(preds)


if __name__ == '__main__':
    create_all()
