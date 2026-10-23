"""
F1 Race Winner Prediction - Feature Engineering Module
Builds one row per driver per race using only what is known before the
race starts: the starting grid, qualifying, and results of earlier races.
"""

import numpy as np
import pandas as pd

import config

# Every feature below is either set before the start (grid, qualifying) or
# computed from earlier races only. tests/test_no_leakage.py checks this.
FEATURE_COLUMNS = [
    'grid_pos',             # starting slot; pit lane starts go to the back
    'quali_pos',            # qualifying result
    'drv_points_share',     # driver's championship points / leader's, before this round
    'team_points_share',    # constructor's championship points / leader's, before this round
    'drv_pts_last5',        # mean points over the driver's last 5 races
    'drv_finish_last5',     # mean finishing position over the last 5 races
    'drv_grid_last5',       # mean starting slot over the last 5 races (one-lap pace)
    'drv_wins_last10',      # wins in the last 10 races
    'drv_podiums_last10',   # podiums in the last 10 races
    'drv_dnf_rate_last10',  # share of the last 10 races not finished
    'team_pts_last5',       # constructor's mean points per race over its last 5 races
]

ID_COLUMNS = ['season', 'round', 'race_key', 'race_name', 'circuit_id', 'driver_id',
              'driver_code', 'driver_name', 'constructor_id', 'constructor_name']
OUTCOME_COLUMNS = ['position', 'points', 'status', 'won', 'podium']


def load_raw():
    races = pd.read_csv(config.RACES_FILE)
    sprints = pd.read_csv(config.SPRINTS_FILE)
    return races, sprints


def _finished(status):
    """'Finished', '+1 Lap', 'Lapped' and so on count as finishing."""
    return status.eq('Finished') | status.str.startswith('+') | status.eq('Lapped')


def _past(df, by, col, window, how):
    """Rolling stat of `col` over the previous `window` rows of each group.

    shift(1) drops the current row, so a race never sees its own result.
    """
    return df.groupby(by, sort=False)[col].transform(
        lambda s: getattr(s.shift(1).rolling(window, min_periods=1), how)()
    )


def _points_before(df, points, key):
    """Championship points of each `key` (driver or constructor) from earlier
    rounds of the same season, race and sprint points together."""
    per_round = (points.groupby(['season', key, 'round'], as_index=False)['points'].sum()
                 .sort_values('round'))
    per_round['total'] = per_round.groupby(['season', key])['points'].cumsum()

    left = df[['season', 'round', key]].reset_index().sort_values('round')
    merged = pd.merge_asof(left, per_round[['season', key, 'round', 'total']],
                           on='round', by=['season', key],
                           allow_exact_matches=False)  # rounds strictly before this one
    return merged.set_index('index')['total'].reindex(df.index).fillna(0.0)


def build_features(races, sprints):
    """Return one row per driver per race with FEATURE_COLUMNS filled in.

    Races without results (not run yet) get features too, so the predictor
    can score them; their outcome columns stay empty.
    """
    df = races.sort_values(['season', 'round', 'driver_id']).reset_index(drop=True)
    df['race_key'] = df['season'] * 100 + df['round']

    has_result = df['position'].notna()
    df['won'] = (df['position'] == 1).astype(float).where(has_result)
    df['podium'] = (df['position'] <= 3).astype(float).where(has_result)
    df['dnf'] = (~_finished(df['status'].fillna(''))).astype(float).where(has_result)

    # Starting grid. 0 means a pit lane start; a race not yet run uses qualifying.
    field = df.groupby('race_key')['driver_id'].transform('size').astype(float)
    grid = df['grid'].astype(float)
    grid = grid.mask(grid == 0, field).fillna(df['quali_position']).fillna(field)
    df['grid_pos'] = grid
    df['quali_pos'] = df['quali_position'].fillna(grid)

    # Driver form from earlier races (any season)
    df['drv_pts_last5'] = _past(df, 'driver_id', 'points', 5, 'mean')
    df['drv_finish_last5'] = _past(df, 'driver_id', 'position', 5, 'mean')
    df['drv_grid_last5'] = _past(df, 'driver_id', 'grid_pos', 5, 'mean')
    df['drv_wins_last10'] = _past(df, 'driver_id', 'won', 10, 'sum')
    df['drv_podiums_last10'] = _past(df, 'driver_id', 'podium', 10, 'sum')
    df['drv_dnf_rate_last10'] = _past(df, 'driver_id', 'dnf', 10, 'mean')

    # Constructor form: total points per race first, so a teammate's result in
    # the same race can never feed in.
    team_race = (df.groupby(['constructor_id', 'race_key'], as_index=False)['points']
                 .sum(min_count=1).sort_values('race_key'))
    team_race['team_pts_last5'] = _past(team_race, 'constructor_id', 'points', 5, 'mean')
    df = df.merge(team_race[['constructor_id', 'race_key', 'team_pts_last5']],
                  on=['constructor_id', 'race_key'], how='left')

    # Championship standings as they stood before this round
    point_cols = ['season', 'round', 'driver_id', 'constructor_id', 'points']
    points = pd.concat([df.loc[df['position'].notna(), point_cols], sprints[point_cols]],
                       ignore_index=True)
    df['drv_points'] = _points_before(df, points, 'driver_id')
    df['team_points'] = _points_before(df, points, 'constructor_id')
    for col, share in [('drv_points', 'drv_points_share'), ('team_points', 'team_points_share')]:
        leader = df.groupby('race_key')[col].transform('max')
        df[share] = np.where(leader > 0, df[col] / leader.where(leader > 0, 1.0), 0.0)

    return df


if __name__ == '__main__':
    races, sprints = load_raw()
    features = build_features(races, sprints)
    features.to_csv(config.FEATURES_FILE, index=False)

    print(f'Built {len(FEATURE_COLUMNS)} features for {len(features)} driver rows')
    print(f'Saved to {config.FEATURES_FILE}')
    print('\nMissing values per feature (drivers with no earlier races):')
    print(features[FEATURE_COLUMNS].isna().sum().to_string())
