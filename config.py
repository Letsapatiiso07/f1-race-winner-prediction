"""
F1 Race Winner Prediction - Configuration File
Centralized configuration for easy customization
"""

# ============================================================================
# DATA COLLECTION SETTINGS
# ============================================================================

# Year range for historical data
DATA_START_YEAR = 2018
DATA_END_YEAR = 2024

# FastF1 cache directory (speeds up subsequent runs)
CACHE_DIR = 'f1_cache'

# ============================================================================
# FILE PATHS
# ============================================================================

RAW_DATA_FILE = 'f1_race_data.csv'
ENGINEERED_DATA_FILE = 'f1_engineered_data.csv'
MODEL_FILE = 'best_model.pkl'
FEATURE_IMPORTANCE_PLOT = 'feature_importance.png'

# ============================================================================
# FEATURE ENGINEERING SETTINGS
# ============================================================================

# Number of recent races to consider for form features
FORM_WINDOW_SHORT = 3  # Last 3 races
FORM_WINDOW_LONG = 5   # Last 5 races

# Feature columns for model training
FEATURE_COLUMNS = [
    'grid_position',
    'quali_position',
    'driver_last_3_avg_position',
    'driver_last_5_avg_points',
    'driver_wins_last_5',
    'driver_podiums_last_5',
    'driver_dnf_rate_last_5',
    'driver_season_points',
    'team_last_3_avg_position',
    'team_last_3_points',
    'team_season_points',
    'driver_circuit_avg_position',
    'driver_circuit_wins',
    'team_circuit_avg_position',
    'driver_avg_quali_race_diff',
    'driver_encoded',
    'team_encoded',
    'circuit_encoded',
    'year',
    'round'
]

# Target variable
TARGET_COLUMN = 'won'

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

# Test set configuration
TEST_YEAR = 2024  # Use this year as test set
RANDOM_STATE = 42  # For reproducibility

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# XGBoost parameters
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Gradient Boosting parameters
GB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE
}

# ============================================================================
# PREDICTION SETTINGS
# ============================================================================

# Number of top drivers to display in predictions
DEFAULT_TOP_N = 10

# Minimum win probability to consider (for filtering)
MIN_WIN_PROBABILITY = 0.01  # 1%

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Console output formatting
DIVIDER_LENGTH = 70
SHOW_EMOJI = True

# Feature importance plot settings
FEATURE_PLOT_TOP_N = 15
FEATURE_PLOT_FIGSIZE = (10, 8)
FEATURE_PLOT_DPI = 300

# ============================================================================
# CURRENT SEASON INFO (Update as needed)
# ============================================================================

# Current F1 season
CURRENT_SEASON = 2025

# Current driver abbreviations (update if needed)
CURRENT_DRIVERS = {
    'VER': 'Max Verstappen',
    'PER': 'Sergio Perez',
    'HAM': 'Lewis Hamilton',
    'RUS': 'George Russell',
    'LEC': 'Charles Leclerc',
    'SAI': 'Carlos Sainz',
    'NOR': 'Lando Norris',
    'PIA': 'Oscar Piastri',
    'ALO': 'Fernando Alonso',
    'STR': 'Lance Stroll',
    'GAS': 'Pierre Gasly',
    'OCO': 'Esteban Ocon',
    'ALB': 'Alexander Albon',
    'SAR': 'Logan Sargeant',
    'TSU': 'Yuki Tsunoda',
    'RIC': 'Daniel Ricciardo',
    'HUL': 'Nico Hulkenberg',
    'MAG': 'Kevin Magnussen',
    'BOT': 'Valtteri Bottas',
    'ZHO': 'Zhou Guanyu'
}

# Current team names (update if needed)
CURRENT_TEAMS = {
    'Red Bull Racing': ['VER', 'PER'],
    'Mercedes': ['HAM', 'RUS'],
    'Ferrari': ['LEC', 'SAI'],
    'McLaren': ['NOR', 'PIA'],
    'Aston Martin': ['ALO', 'STR'],
    'Alpine': ['GAS', 'OCO'],
    'Williams': ['ALB', 'SAR'],
    'RB': ['TSU', 'RIC'],
    'Haas F1 Team': ['HUL', 'MAG'],
    'Kick Sauber': ['BOT', 'ZHO']
}

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Enable/disable specific features
ENABLE_CIRCUIT_HISTORY = True
ENABLE_TEAM_FEATURES = True
ENABLE_FORM_FEATURES = True
ENABLE_QUALIFYING_FEATURES = True

# Data validation settings
VALIDATE_DATA = True
HANDLE_MISSING_VALUES = True
FILL_STRATEGY = 'median'  # Options: 'median', 'mean', 'zero'

# Model evaluation settings
USE_CROSS_VALIDATION = False  # Set to True for more robust evaluation
CV_FOLDS = 5

# Logging level
VERBOSE = True  # Set to False for minimal output

# ============================================================================
# CIRCUIT INFORMATION (Optional - for future enhancements)
# ============================================================================

CIRCUITS = {
    'Bahrain': {'type': 'street', 'characteristics': 'high-speed'},
    'Saudi Arabia': {'type': 'street', 'characteristics': 'high-speed'},
    'Australia': {'type': 'street', 'characteristics': 'medium-speed'},
    'Monaco': {'type': 'street', 'characteristics': 'low-speed'},
    'Spain': {'type': 'permanent', 'characteristics': 'high-speed'},
    'Canada': {'type': 'street', 'characteristics': 'medium-speed'},
    'Austria': {'type': 'permanent', 'characteristics': 'high-speed'},
    'Great Britain': {'type': 'permanent', 'characteristics': 'high-speed'},
    'Hungary': {'type': 'permanent', 'characteristics': 'low-speed'},
    'Belgium': {'type': 'permanent', 'characteristics': 'high-speed'},
    'Netherlands': {'type': 'permanent', 'characteristics': 'high-speed'},
    'Italy': {'type': 'permanent', 'characteristics': 'high-speed'},
    'Singapore': {'type': 'street', 'characteristics': 'low-speed'},
    'Japan': {'type': 'permanent', 'characteristics': 'high-speed'},
    'Qatar': {'type': 'permanent', 'characteristics': 'high-speed'},
    'United States': {'type': 'permanent', 'characteristics': 'high-speed'},
    'Mexico': {'type': 'permanent', 'characteristics': 'medium-speed'},
    'Brazil': {'type': 'permanent', 'characteristics': 'medium-speed'},
    'Las Vegas': {'type': 'street', 'characteristics': 'high-speed'},
    'Abu Dhabi': {'type': 'permanent', 'characteristics': 'medium-speed'}
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_team_for_driver(driver_abbr):
    """Get current team for a driver"""
    for team, drivers in CURRENT_TEAMS.items():
        if driver_abbr in drivers:
            return team
    return "Unknown Team"


def get_driver_name(driver_abbr):
    """Get full name for driver abbreviation"""
    return CURRENT_DRIVERS.get(driver_abbr, "Unknown Driver")


def get_all_drivers():
    """Get list of all current driver abbreviations"""
    return list(CURRENT_DRIVERS.keys())


def get_all_teams():
    """Get list of all current team names"""
    return list(CURRENT_TEAMS.keys())


def validate_driver(driver_abbr):
    """Check if driver abbreviation is valid"""
    return driver_abbr in CURRENT_DRIVERS


def validate_team(team_name):
    """Check if team name is valid"""
    return team_name in CURRENT_TEAMS


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("🏎️  F1 Prediction Model - Configuration")
    print("=" * 70)
    
    print(f"\n📊 Data Settings:")
    print(f"   Year Range: {DATA_START_YEAR} - {DATA_END_YEAR}")
    print(f"   Cache Directory: {CACHE_DIR}")
    
    print(f"\n🔧 Feature Settings:")
    print(f"   Number of Features: {len(FEATURE_COLUMNS)}")
    print(f"   Form Windows: {FORM_WINDOW_SHORT}, {FORM_WINDOW_LONG} races")
    
    print(f"\n🤖 Model Settings:")
    print(f"   Test Year: {TEST_YEAR}")
    print(f"   Random State: {RANDOM_STATE}")
    
    print(f"\n🏁 Current Season: {CURRENT_SEASON}")
    print(f"   Active Drivers: {len(CURRENT_DRIVERS)}")
    print(f"   Active Teams: {len(CURRENT_TEAMS)}")
    
    print(f"\n📋 Sample Drivers:")
    for i, (abbr, name) in enumerate(list(CURRENT_DRIVERS.items())[:5]):
        team = get_team_for_driver(abbr)
        print(f"   {abbr}: {name} ({team})")
    
    print("\n✅ Configuration loaded successfully!")