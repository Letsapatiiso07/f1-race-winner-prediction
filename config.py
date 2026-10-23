"""
F1 Race Winner Prediction - Configuration File
Seasons, file paths and model settings in one place
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# DATA COLLECTION SETTINGS
# ============================================================================

# Jolpica serves the old Ergast API format. No key needed.
API_BASE = 'https://api.jolpi.ca/ergast/f1'
PAGE_SIZE = 100          # largest page the API returns
REQUEST_PAUSE = 0.5      # seconds between calls; the API allows 4 per second and 500 per hour
MAX_RETRIES = 5

# 2013 is fetched only so the form features for early 2014 races have history.
FIRST_SEASON = 2013
LAST_SEASON = 2026       # 2026 is still running, so it holds only the races held so far
SPRINT_FIRST_SEASON = 2021

# ============================================================================
# FILE PATHS
# ============================================================================

DATA_DIR = os.path.join(ROOT_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
RACES_FILE = os.path.join(DATA_DIR, 'races.csv')
SPRINTS_FILE = os.path.join(DATA_DIR, 'sprints.csv')
FEATURES_FILE = os.path.join(DATA_DIR, 'features.csv')
PREDICTIONS_FILE = os.path.join(DATA_DIR, 'test_predictions.csv')

MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.joblib')

DOCS_DIR = os.path.join(ROOT_DIR, 'docs')
METRICS_FILE = os.path.join(DOCS_DIR, 'metrics.csv')
VALIDATION_FILE = os.path.join(DOCS_DIR, 'validation.csv')

# ============================================================================
# EVALUATION SPLIT
# ============================================================================

TRAIN_SEASONS = list(range(2014, 2022))   # 2014-2021: fit candidate models
VALID_SEASON = 2022                       # pick one candidate by its log loss here
TEST_SEASONS = [2023, 2024, 2025, 2026]   # never seen in training or model choice
FULL_TEST_SEASONS = [2023, 2024, 2025]    # the headline numbers
FINAL_SEASONS = list(range(2014, 2026))   # refit on these for the predictor

# ============================================================================
# MODEL SETTINGS
# ============================================================================

RANDOM_STATE = 42
N_JOBS = 2
