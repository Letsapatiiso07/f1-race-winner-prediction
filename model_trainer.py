"""
F1 Race Winner Prediction - Model Training Module
Fits candidate models on 2014-2021, picks one on 2022, then scores it on
2023 onward against simple baselines. Each driver gets a win score; scores
are scaled within each race so they sum to 1.
"""

import os

import config

os.environ.setdefault('OMP_NUM_THREADS', str(config.N_JOBS))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import FEATURE_COLUMNS, build_features, load_raw

CANDIDATES = {
    'logreg_C1': make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000)),
    'logreg_C0.1': make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000)),
    'hist_gb_shallow': HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
        random_state=config.RANDOM_STATE),
    'hist_gb_deeper': HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.05, max_iter=300, min_samples_leaf=20,
        random_state=config.RANDOM_STATE),
    'random_forest': make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                               n_jobs=config.N_JOBS, random_state=config.RANDOM_STATE)),
}

METHOD_LABELS = {
    'model': 'Model',
    'pole': 'Pole sitter wins',
    'leader': 'Championship leader wins',
}


# ============================================================================
# PROBABILITIES
# ============================================================================

def normalise_by_race(df, raw):
    """Scale raw scores so they sum to 1 within each race."""
    raw = pd.Series(raw, index=df.index)
    return raw / raw.groupby(df['race_key']).transform('sum')


def model_probs(model, df):
    return normalise_by_race(df, model.predict_proba(df[FEATURE_COLUMNS])[:, 1])


def grid_win_rates(train):
    """Share of starts won from each grid slot in the training seasons.

    This gives the pole sitter baseline a probability, so it gets a log loss.
    """
    slot = train['grid_pos'].clip(upper=20)
    stats = train.groupby(slot)['won'].agg(['sum', 'count'])
    return (stats['sum'] + 0.5) / (stats['count'] + 1.0)   # light smoothing


def grid_probs(rates, df):
    raw = df['grid_pos'].clip(upper=20).map(rates).fillna(rates.min())
    return normalise_by_race(df, raw)


# ============================================================================
# SCORING
# ============================================================================

def score_races(df, rank_col, prob_col=None):
    """One row per race: did the top pick win, was the winner in the top 3,
    how much of the podium the top 3 named, and (with probabilities) the
    log loss and Brier score. Ties go to the better grid slot."""
    rows = []
    for race_key, race in df.groupby('race_key'):
        order = race.sort_values([rank_col, 'grid_pos'], ascending=[False, True])
        top3 = set(order['driver_id'].iloc[:3])
        winner = race.loc[race['won'] == 1, 'driver_id'].iloc[0]
        podium = set(race.loc[race['podium'] == 1, 'driver_id'])
        row = {
            'race_key': race_key,
            'season': race['season'].iloc[0],
            'pick': order['driver_id'].iloc[0],
            'winner_hit': float(order['driver_id'].iloc[0] == winner),
            'winner_in_top3': float(winner in top3),
            'podium_overlap': len(top3 & podium) / 3,
        }
        if prob_col is not None:
            p = race[prob_col].to_numpy()
            y = race['won'].to_numpy()
            row['log_loss'] = -np.log(max(p[y == 1][0], 1e-15))
            row['brier'] = float(((p - y) ** 2).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(per_race, method, season_label):
    out = {'season': season_label, 'method': method, 'races': len(per_race)}
    for col in ['winner_hit', 'winner_in_top3', 'podium_overlap', 'log_loss', 'brier']:
        out[col] = per_race[col].mean() if col in per_race else np.nan
    return out


def evaluate(test, rates):
    """Score the model and the baselines on the test rows, by season and
    for the full test seasons together."""
    test = test.copy()
    test['p_grid'] = grid_probs(rates, test)
    test['pole_score'] = -test['grid_pos']

    per_race = {
        'model': score_races(test, 'p_model', 'p_model'),
        # The pole pick is the grid order; its probabilities are the grid win rates.
        'pole': score_races(test, 'pole_score', 'p_grid'),
        'leader': score_races(test, 'drv_points'),
    }
    rows = []
    groups = [(str(s), [s]) for s in config.TEST_SEASONS]
    full = config.FULL_TEST_SEASONS
    groups.insert(len(full), (f'{full[0]}-{full[-1]}', full))
    for label, seasons in groups:
        for method, races in per_race.items():
            part = races[races['season'].isin(seasons)]
            if len(part):
                rows.append(summarise(part, method, label))

    # Where the model and the pole pick name different drivers, who was right?
    both = per_race['model'].merge(per_race['pole'], on=['race_key', 'season'],
                                   suffixes=('_model', '_pole'))
    both = both[both['season'].isin(full) & (both['pick_model'] != both['pick_pole'])]
    model_right = int(both['winner_hit_model'].sum())
    pole_right = int(both['winner_hit_pole'].sum())
    sign_p = binomtest(model_right, model_right + pole_right).pvalue if model_right + pole_right else 1.0
    print(f'\nIn {full[0]}-{full[-1]} the model and the pole pick differ in {len(both)} races: '
          f'model right {model_right}, pole right {pole_right}, '
          f'neither {len(both) - model_right - pole_right} (sign test p = {sign_p:.2f})')
    return pd.DataFrame(rows), test


# ============================================================================
# TRAINING
# ============================================================================

def completed(df, seasons):
    return df[df['season'].isin(seasons) & df['won'].notna()]


def choose_model(features):
    train = completed(features, config.TRAIN_SEASONS)
    valid = completed(features, [config.VALID_SEASON])
    print(f'Choosing a model: train {config.TRAIN_SEASONS[0]}-{config.TRAIN_SEASONS[-1]} '
          f'({len(train)} rows), validate {config.VALID_SEASON} ({valid["race_key"].nunique()} races)')

    rows = []
    for name, template in CANDIDATES.items():
        model = clone(template).fit(train[FEATURE_COLUMNS], train['won'])
        valid = valid.assign(p=model_probs(model, valid))
        races = score_races(valid, 'p', 'p')
        rows.append({'candidate': name, 'log_loss': races['log_loss'].mean(),
                     'brier': races['brier'].mean(), 'winner_hit': races['winner_hit'].mean()})
    table = pd.DataFrame(rows).sort_values('log_loss').reset_index(drop=True)
    print(table.to_string(index=False, float_format='%.3f'))
    best = table['candidate'].iloc[0]
    print(f'Picked {best} (lowest {config.VALID_SEASON} log loss)')
    return best, table


def print_results(metrics):
    show = metrics.copy()
    show['method'] = show['method'].map(METHOD_LABELS)
    for col in ['winner_hit', 'winner_in_top3', 'podium_overlap']:
        show[col] = (100 * show[col]).map('{:.1f}%'.format)
    for col in ['log_loss', 'brier']:
        show[col] = show[col].map(lambda v: '-' if pd.isna(v) else f'{v:.3f}')
    print(show.to_string(index=False))


def run(features=None):
    if features is None:
        features = build_features(*load_raw())

    best, validation = choose_model(features)

    # Refit the chosen setup on 2014-2022 and score every later race.
    fit_seasons = config.TRAIN_SEASONS + [config.VALID_SEASON]
    train = completed(features, fit_seasons)
    model = clone(CANDIDATES[best]).fit(train[FEATURE_COLUMNS], train['won'])
    test = completed(features, config.TEST_SEASONS)
    test = test.assign(p_model=model_probs(model, test))
    metrics, scored = evaluate(test, grid_win_rates(train))

    print(f'\nTest results: model fit on {fit_seasons[0]}-{fit_seasons[-1]}, '
          f'scored on races it never saw')
    print('(pole sitter log loss and Brier use grid-slot win rates from the training seasons)')
    print_results(metrics)

    os.makedirs(config.DOCS_DIR, exist_ok=True)
    metrics.round(4).to_csv(config.METRICS_FILE, index=False)
    validation.round(4).to_csv(config.VALIDATION_FILE, index=False)
    keep = ['season', 'round', 'race_key', 'race_name', 'driver_id', 'driver_code',
            'constructor_name', 'grid_pos', 'drv_points', 'position', 'won', 'podium',
            'p_model', 'p_grid']
    scored[keep].to_csv(config.PREDICTIONS_FILE, index=False)
    print(f'\nSaved {config.METRICS_FILE}, {config.VALIDATION_FILE} and {config.PREDICTIONS_FILE}')

    # The predictor uses the same setup refit on every full season we have.
    final_train = completed(features, config.FINAL_SEASONS)
    final = clone(CANDIDATES[best]).fit(final_train[FEATURE_COLUMNS], final_train['won'])
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump({'model': final, 'name': best, 'features': FEATURE_COLUMNS,
                 'seasons': config.FINAL_SEASONS}, config.MODEL_FILE)
    print(f'Saved {best} refit on {config.FINAL_SEASONS[0]}-{config.FINAL_SEASONS[-1]} '
          f'to {config.MODEL_FILE}')
    return metrics


if __name__ == '__main__':
    run()
