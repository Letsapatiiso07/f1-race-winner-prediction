"""
F1 Race Winner Prediction - Feature Engineering Module
Creates meaningful features from raw race data
"""

import pandas as pd
import numpy as np


class F1FeatureEngineer:
    def __init__(self, df):
        """
        Initialize feature engineer with race data
        
        Args:
            df: DataFrame with raw race data
        """
        self.df = df.copy()
        self.df = self.df.sort_values(['year', 'round']).reset_index(drop=True)
        
    def create_all_features(self):
        """Create all features for the model"""
        print("🔧 Creating features...")
        
        self.create_driver_form_features()
        self.create_team_features()
        self.create_circuit_features()
        self.create_qualifying_features()
        self.encode_categorical_features()
        self.handle_missing_values()
        
        print("✅ Feature engineering complete!")
        return self.df
    
    def create_driver_form_features(self):
        """Create features based on recent driver performance"""
        print("  - Creating driver form features...")
        
        # Sort by driver and race order
        self.df['race_id'] = self.df['year'].astype(str) + '_' + self.df['round'].astype(str)
        
        # Last 3 races average finish position
        self.df['driver_last_3_avg_position'] = self.df.groupby('driver')['finish_position'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        
        # Last 5 races average points
        self.df['driver_last_5_avg_points'] = self.df.groupby('driver')['points'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        # Wins in last 5 races
        self.df['driver_wins_last_5'] = self.df.groupby('driver')['won'].transform(
            lambda x: x.rolling(window=5, min_periods=1).sum().shift(1)
        )
        
        # Podiums in last 5 races
        self.df['driver_podiums_last_5'] = self.df.groupby('driver')['podium'].transform(
            lambda x: x.rolling(window=5, min_periods=1).sum().shift(1)
        )
        
        # DNF rate in last 5 races
        self.df['driver_dnf_rate_last_5'] = self.df.groupby('driver')['dnf'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        # Season points so far
        self.df['driver_season_points'] = self.df.groupby(['driver', 'year'])['points'].transform(
            lambda x: x.expanding().sum().shift(1)
        ).fillna(0)
        
    def create_team_features(self):
        """Create features based on team performance"""
        print("  - Creating team features...")
        
        # Team average finish in last 3 races
        self.df['team_last_3_avg_position'] = self.df.groupby('team')['finish_position'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        
        # Team points in last 3 races
        self.df['team_last_3_points'] = self.df.groupby('team')['points'].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum().shift(1)
        )
        
        # Team season points
        self.df['team_season_points'] = self.df.groupby(['team', 'year'])['points'].transform(
            lambda x: x.expanding().sum().shift(1)
        ).fillna(0)
        
    def create_circuit_features(self):
        """Create circuit-specific features"""
        print("  - Creating circuit features...")
        
        # Driver's average finish at this circuit
        self.df['driver_circuit_avg_position'] = self.df.groupby(['driver', 'circuit'])['finish_position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Driver wins at this circuit
        self.df['driver_circuit_wins'] = self.df.groupby(['driver', 'circuit'])['won'].transform(
            lambda x: x.expanding().sum().shift(1)
        ).fillna(0)
        
        # Team performance at circuit
        self.df['team_circuit_avg_position'] = self.df.groupby(['team', 'circuit'])['finish_position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
    def create_qualifying_features(self):
        """Create qualifying-related features"""
        print("  - Creating qualifying features...")
        
        # Quali to race position change (historical average for driver)
        self.df['quali_race_diff'] = self.df['finish_position'] - self.df['quali_position']
        self.df['driver_avg_quali_race_diff'] = self.df.groupby('driver')['quali_race_diff'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Fill grid position with quali position if missing
        self.df['grid_position'] = self.df['grid_position'].fillna(self.df['quali_position'])
        
    def encode_categorical_features(self):
        """Encode categorical variables"""
        print("  - Encoding categorical features...")
        
        # Create numeric encodings for driver and team
        self.df['driver_encoded'] = pd.Categorical(self.df['driver']).codes
        self.df['team_encoded'] = pd.Categorical(self.df['team']).codes
        self.df['circuit_encoded'] = pd.Categorical(self.df['circuit']).codes
        
    def handle_missing_values(self):
        """Handle any remaining missing values"""
        print("  - Handling missing values...")
        
        # Fill NaN values with reasonable defaults
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
    def get_feature_columns(self):
        """Return list of feature columns for modeling"""
        feature_cols = [
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
        return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    df = pd.read_csv('f1_race_data.csv')
    
    engineer = F1FeatureEngineer(df)
    df_engineered = engineer.create_all_features()
    
    print(f"\n📊 Engineered dataset shape: {df_engineered.shape}")
    print(f"\n🔍 Feature columns: {engineer.get_feature_columns()}")
    print(f"\n📈 Sample engineered data:\n{df_engineered[engineer.get_feature_columns()].head()}")
    
    # Save engineered data
    df_engineered.to_csv('f1_engineered_data.csv', index=False)
    print("\n💾 Engineered data saved!")