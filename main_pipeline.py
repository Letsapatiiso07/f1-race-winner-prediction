"""
F1 Race Winner Prediction - Main Pipeline
Fetch data, build features, train and evaluate, draw the charts, then
predict the latest race in the data.
"""

import argparse
import os

import config
import model_trainer
import visualizer
from data_collector import clear_cache, collect_all
from feature_engineering import FEATURE_COLUMNS, build_features
from predictor import F1Predictor, print_prediction


def step(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def run_full_pipeline(refresh=()):
    step('STEP 1: DATA COLLECTION (cached API pages are reused)')
    clear_cache(refresh)
    races, sprints = collect_all()

    step('STEP 2: FEATURE ENGINEERING')
    features = build_features(races, sprints)
    features.to_csv(config.FEATURES_FILE, index=False)
    print(f'Built {len(FEATURE_COLUMNS)} features for {len(features)} driver rows')

    step('STEP 3: MODEL TRAINING AND EVALUATION')
    model_trainer.run(features)

    step('STEP 4: CHARTS')
    visualizer.create_all()

    step('STEP 5: PREDICT THE LATEST RACE')
    quick_predict()


def quick_predict():
    if not os.path.exists(config.MODEL_FILE):
        print('No trained model found. Run: python main_pipeline.py')
        return
    predictor = F1Predictor()
    season, rnd = predictor.latest_race()
    print_prediction(predictor.predict_race(season, rnd))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F1 Race Winner Prediction')
    parser.add_argument('--predict', action='store_true',
                        help='only predict the latest race (needs a trained model)')
    parser.add_argument('--refresh', type=int, nargs='*', default=[],
                        help='seasons to download again, e.g. --refresh 2026')
    args = parser.parse_args()

    if args.predict:
        quick_predict()
    else:
        run_full_pipeline(args.refresh)
