"""
F1 Race Winner Prediction - Prediction Module
Win probabilities for any race in the data, or for a grid you type in.
Features come from the same builder used in training, so a prediction only
uses what was known before the start.
"""

import argparse

import joblib
import pandas as pd

import config
from feature_engineering import build_features, load_raw
from model_trainer import model_probs


class F1Predictor:
    def __init__(self, model_path=config.MODEL_FILE):
        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.model_name = bundle['name']
        self.trained_seasons = bundle['seasons']
        self.races, self.sprints = load_raw()
        self.features = build_features(self.races, self.sprints)
        print(f'Loaded {self.model_name} trained on '
              f'{self.trained_seasons[0]}-{self.trained_seasons[-1]}')

    def _score(self, race):
        race = race.assign(win_probability=model_probs(self.model, race))
        return race.sort_values(['win_probability', 'grid_pos'], ascending=[False, True])

    def latest_race(self):
        last = self.features.sort_values('race_key').iloc[-1]
        return int(last['season']), int(last['round'])

    def predict_race(self, season, rnd):
        """Predict a race already in the data (run or with qualifying done)."""
        race = self.features[(self.features['season'] == season) & (self.features['round'] == rnd)]
        if race.empty:
            raise ValueError(f'No data for {season} round {rnd}. '
                             f'Run data_collector.py --refresh {season} after qualifying.')
        return self._score(race)

    def predict_from_grid(self, codes):
        """Predict a new race that follows the latest completed race.

        codes: driver codes in grid order, e.g. ['RUS', 'ANT', 'LEC'].
        Each driver keeps the team of their latest race.
        """
        done = self.races[self.races['position'].notna()].sort_values(['season', 'round'])
        last = done.iloc[-1]
        season, rnd = int(last['season']), int(last['round']) + 1
        latest = done.groupby('driver_code').tail(1).set_index('driver_code')

        rows = []
        for slot, code in enumerate(codes, 1):
            if code not in latest.index:
                raise ValueError(f'Unknown driver code: {code}')
            d = latest.loc[code]
            rows.append({
                'season': season, 'round': rnd, 'race_name': 'Next race',
                'circuit_id': 'unknown', 'date': '',
                'driver_id': d['driver_id'], 'driver_code': code,
                'driver_name': d['driver_name'], 'constructor_id': d['constructor_id'],
                'constructor_name': d['constructor_name'],
                'grid': float(slot), 'quali_position': float(slot),
            })
        races = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        feats = build_features(races, self.sprints)
        return self._score(feats[(feats['season'] == season) & (feats['round'] == rnd)])

    def driver_stats(self, code):
        """Career numbers in the data for the latest driver with this code.

        Codes get reused (VER was Vergne before Verstappen), so match on driver_id.
        """
        done = self.races[self.races['position'].notna()].sort_values(['season', 'round'])
        matches = done[done['driver_code'] == code]
        if matches.empty:
            return None
        rows = done[done['driver_id'] == matches['driver_id'].iloc[-1]]
        return {
            'Name': rows['driver_name'].iloc[-1],
            'Seasons': f"{rows['season'].min()}-{rows['season'].max()}",
            'Starts': len(rows),
            'Wins': int((rows['position'] == 1).sum()),
            'Podiums': int((rows['position'] <= 3).sum()),
            'Front row starts': int(rows['grid'].between(1, 2).sum()),
            'Win rate': f"{(rows['position'] == 1).mean():.1%}",
        }


def print_prediction(race, top_n=10):
    first = race.iloc[0]
    print(f"\n{int(first['season'])} round {int(first['round'])}: {first['race_name']}")
    has_result = race['position'].notna().any()
    header = f"{'#':<4}{'Driver':<8}{'Team':<24}{'Grid':>5}{'Win %':>8}"
    print(header + (f"{'Finish':>8}" if has_result else ''))
    print('-' * (len(header) + (8 if has_result else 0)))
    for i, (_, row) in enumerate(race.head(top_n).iterrows(), 1):
        line = (f"{i:<4}{row['driver_code']:<8}{row['constructor_name'][:22]:<24}"
                f"{int(row['grid_pos']):>5}{row['win_probability']:>8.1%}")
        if has_result:
            line += f"{int(row['position']):>8}"
        print(line)
    if has_result:
        winner = race.loc[race['position'] == 1].iloc[0]
        print(f"Actual winner: {winner['driver_code']} "
              f"(model gave {winner['win_probability']:.1%})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict an F1 race winner')
    parser.add_argument('--season', type=int)
    parser.add_argument('--round', type=int)
    parser.add_argument('--grid', help='driver codes in grid order, e.g. RUS,ANT,LEC,NOR')
    args = parser.parse_args()

    predictor = F1Predictor()
    if args.grid:
        codes = [c.strip().upper() for c in args.grid.split(',') if c.strip()]
        print_prediction(predictor.predict_from_grid(codes), top_n=len(codes))
    else:
        if args.season and args.round:
            season, rnd = args.season, args.round
        else:
            season, rnd = predictor.latest_race()
        if season in predictor.trained_seasons:
            print(f'Note: the model was trained on {season}, so this is not a fair test.')
        print_prediction(predictor.predict_race(season, rnd))
