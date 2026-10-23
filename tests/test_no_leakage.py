"""
The features for a race must come out the same whether or not the results of
that race (and every later race) exist. If they differ, the feature builder
is reading something that is only known after the start.
"""

import os

import numpy as np
import pandas as pd
import pytest

import config
from feature_engineering import FEATURE_COLUMNS, build_features

POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
RESULT_COLUMNS = ['position', 'points', 'status']


def make_synthetic(seasons=3, rounds=6, drivers=8, seed=0):
    """Random races with 4 teams of 2 drivers and a sprint every third round."""
    rng = np.random.default_rng(seed)
    races, sprints = [], []
    for season in range(2020, 2020 + seasons):
        for rnd in range(1, rounds + 1):
            grid = rng.permutation(drivers) + 1
            finish = rng.permutation(drivers) + 1
            for d in range(drivers):
                races.append({
                    'season': season, 'round': rnd, 'race_name': f'R{rnd}',
                    'circuit_id': f'c{rnd}', 'date': f'{season}-01-{rnd:02d}',
                    'driver_id': f'd{d}', 'driver_code': f'D{d:02d}',
                    'driver_name': f'Driver {d}', 'constructor_id': f't{d // 2}',
                    'constructor_name': f'Team {d // 2}',
                    'grid': float(grid[d]) if rng.random() > 0.05 else 0.0,
                    'position': float(finish[d]),
                    'points': float(POINTS[finish[d] - 1]) if finish[d] <= 10 else 0.0,
                    'status': 'Finished' if rng.random() > 0.15 else 'Engine',
                    'quali_position': float(grid[d]),
                })
            if rnd % 3 == 0:
                sprint_order = rng.permutation(drivers) + 1
                for d in range(drivers):
                    sprints.append({
                        'season': season, 'round': rnd, 'driver_id': f'd{d}',
                        'constructor_id': f't{d // 2}',
                        'points': float(max(0, 9 - sprint_order[d])),
                    })
    return pd.DataFrame(races), pd.DataFrame(sprints)


def features_before_start(races, sprints, season, rnd):
    """Features for one race built from data as it stood before the start:
    later races removed and this race's results blanked."""
    key = (races['season'] * 100 + races['round'])
    target = season * 100 + rnd
    past = races[key <= target].copy()
    past.loc[key[key <= target] == target, RESULT_COLUMNS] = np.nan
    sprint_key = sprints['season'] * 100 + sprints['round']
    feats = build_features(past, sprints[sprint_key <= target])
    return feats[(feats['season'] == season) & (feats['round'] == rnd)]


def assert_same_features(full, races, sprints, season, rnd):
    expected = full[(full['season'] == season) & (full['round'] == rnd)]
    got = features_before_start(races, sprints, season, rnd)
    expected = expected.set_index('driver_id').sort_index()[FEATURE_COLUMNS]
    got = got.set_index('driver_id').sort_index()[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(got, expected, check_exact=False, rtol=1e-12)


def test_synthetic_features_ignore_current_and_future_results():
    races, sprints = make_synthetic()
    full = build_features(races, sprints)
    for (season, rnd) in races[['season', 'round']].drop_duplicates().itertuples(index=False):
        assert_same_features(full, races, sprints, season, rnd)


def test_scrambled_result_does_not_change_its_features():
    races, sprints = make_synthetic(seed=1)
    full = build_features(races, sprints)
    scrambled = races.copy()
    mask = (scrambled['season'] == 2021) & (scrambled['round'] == 4)
    scrambled.loc[mask, 'position'] = scrambled.loc[mask, 'position'].values[::-1]
    scrambled.loc[mask, 'points'] = scrambled.loc[mask, 'points'].values[::-1]
    after = build_features(scrambled, sprints)

    pick = lambda df: df[(df['season'] == 2021) & (df['round'] == 4)].set_index('driver_id')[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(pick(after), pick(full))
    # ...while the next race does see the new result
    nxt = lambda df: df[(df['season'] == 2021) & (df['round'] == 5)].set_index('driver_id')['drv_pts_last5']
    assert not nxt(after).equals(nxt(full))


@pytest.mark.skipif(not os.path.exists(config.RACES_FILE), reason='run data_collector.py first')
def test_real_data_features_ignore_current_and_future_results():
    races = pd.read_csv(config.RACES_FILE)
    sprints = pd.read_csv(config.SPRINTS_FILE)
    full = build_features(races, sprints)
    # A season opener, sprint weekends (2021 R14 also had two pit lane starts)
    # and late-season races
    for season, rnd in [(2014, 1), (2021, 14), (2023, 4), (2024, 21), (2025, 24), (2026, 5)]:
        assert_same_features(full, races, sprints, season, rnd)
