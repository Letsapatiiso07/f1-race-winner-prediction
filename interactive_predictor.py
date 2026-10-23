"""
F1 Race Winner Prediction - Interactive CLI
Simple command-line menu around predictor.py
"""

import os
import sys

import config
from predictor import F1Predictor, print_prediction


class InteractiveF1Predictor:
    def __init__(self):
        if not os.path.exists(config.MODEL_FILE):
            print('No trained model found. Run: python main_pipeline.py')
            sys.exit(1)
        self.predictor = F1Predictor()

    def show_menu(self):
        print('\n' + '=' * 60)
        print('F1 RACE WINNER PREDICTION')
        print('=' * 60)
        print('  1. Predict a race in the data (season and round)')
        print('  2. Predict a race from a grid you type in')
        print('  3. Driver statistics')
        print('  4. Exit')

    def predict_known_race(self):
        default_season, default_round = self.predictor.latest_race()
        season = input(f'Season [{default_season}]: ').strip()
        rnd = input(f'Round [{default_round}]: ').strip()
        season = int(season) if season else default_season
        rnd = int(rnd) if rnd else default_round
        if season in self.predictor.trained_seasons:
            print(f'Note: the model was trained on {season}, so this is not a fair test.')
        print_prediction(self.predictor.predict_race(season, rnd))

    def predict_typed_grid(self):
        print('Enter driver codes in grid order, separated by commas.')
        codes = input('Grid (e.g. RUS,ANT,LEC,NOR): ')
        codes = [c.strip().upper() for c in codes.split(',') if c.strip()]
        if not codes:
            print('No drivers entered.')
            return
        print_prediction(self.predictor.predict_from_grid(codes), top_n=len(codes))

    def view_driver_stats(self):
        code = input('Driver code (e.g. VER, HAM, LEC): ').strip().upper()
        stats = self.predictor.driver_stats(code)
        if stats is None:
            print(f'No races found for {code}')
            return
        for key, value in stats.items():
            print(f'{key:<18}{value}')

    def run(self):
        actions = {'1': self.predict_known_race, '2': self.predict_typed_grid,
                   '3': self.view_driver_stats}
        while True:
            self.show_menu()
            choice = input('\nChoice (1-4): ').strip()
            if choice == '4':
                break
            if choice not in actions:
                print('Please enter 1-4.')
                continue
            try:
                actions[choice]()
            except ValueError as err:
                print(f'Error: {err}')


if __name__ == '__main__':
    try:
        InteractiveF1Predictor().run()
    except (KeyboardInterrupt, EOFError):
        print()
