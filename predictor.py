"""
F1 Race Winner Prediction - Prediction Module
Makes predictions for upcoming races
"""

import pandas as pd
import numpy as np
import joblib


class F1Predictor:
    def __init__(self, model_path='best_model.pkl', data_path='f1_engineered_data.csv'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model
            data_path: Path to historical data for context
        """
        # Load model
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_cols = model_data['feature_cols']
        self.accuracy = model_data['accuracy']
        
        # Load historical data for feature calculation
        self.historical_data = pd.read_csv(data_path)
        
        print(f"🏎️  Loaded {self.model_name} model (Accuracy: {self.accuracy:.4f})")
        
    def predict_race(self, race_data):
        """
        Predict winner for a race
        
        Args:
            race_data: DataFrame with race information for all drivers
            
        Returns:
            DataFrame with predictions and probabilities
        """
        # Ensure all features are present
        X = race_data[self.feature_cols]
        
        # Get predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Create results dataframe
        results = race_data.copy()
        results['win_prediction'] = predictions
        results['win_probability'] = probabilities
        
        # Sort by probability
        results = results.sort_values('win_probability', ascending=False)
        
        return results
    
    def predict_podium(self, race_data):
        """Predict top 3 finishers"""
        results = self.predict_race(race_data)
        
        print("\n🏆 RACE WINNER PREDICTION")
        print("="*70)
        
        top_3 = results.head(3)
        for idx, (_, driver) in enumerate(top_3.iterrows(), 1):
            print(f"{idx}. {driver['driver']:5s} ({driver['team']:25s}) - "
                  f"Win Probability: {driver['win_probability']:.1%}")
            
        print("="*70)
        
        return top_3
    
    def analyze_race_predictions(self, race_data, top_n=10):
        """Detailed analysis of race predictions"""
        results = self.predict_race(race_data)
        
        print("\n📊 TOP 10 PREDICTIONS")
        print("="*100)
        print(f"{'Pos':<5}{'Driver':<8}{'Team':<30}{'Grid':<6}{'Win %':<10}{'Recent Form':<15}")
        print("="*100)
        
        for idx, (_, driver) in enumerate(results.head(top_n).iterrows(), 1):
            grid_pos = int(driver.get('grid_position', 0))
            win_prob = driver['win_probability']
            recent_form = driver.get('driver_last_3_avg_position', 0)
            
            print(f"{idx:<5}{driver['driver']:<8}{driver['team']:<30}"
                  f"{grid_pos:<6}{win_prob:>6.1%}{recent_form:>12.1f}")
        
        print("="*100)
        
        return results
    
    def get_driver_stats(self, driver_abbr):
        """Get historical stats for a driver"""
        driver_data = self.historical_data[self.historical_data['driver'] == driver_abbr]
        
        if len(driver_data) == 0:
            print(f"❌ No data found for driver: {driver_abbr}")
            return None
        
        stats = {
            'total_races': len(driver_data),
            'wins': driver_data['won'].sum(),
            'podiums': driver_data['podium'].sum(),
            'avg_finish': driver_data['finish_position'].mean(),
            'dnf_rate': driver_data['dnf'].mean(),
            'win_rate': driver_data['won'].mean(),
            'podium_rate': driver_data['podium'].mean(),
        }
        
        print(f"\n📈 {driver_abbr} CAREER STATS")
        print("="*50)
        print(f"Total Races:     {stats['total_races']}")
        print(f"Wins:            {stats['wins']} ({stats['win_rate']:.1%})")
        print(f"Podiums:         {stats['podiums']} ({stats['podium_rate']:.1%})")
        print(f"Avg Finish:      {stats['avg_finish']:.2f}")
        print(f"DNF Rate:        {stats['dnf_rate']:.1%}")
        print("="*50)
        
        return stats


def create_sample_race_input():
    """
    Create a sample race input for predictions
    This would normally come from upcoming race data
    """
    # Sample data for an upcoming race (you'd replace with real data)
    drivers = [
        {'driver': 'VER', 'team': 'Red Bull Racing', 'grid_position': 1, 'quali_position': 1},
        {'driver': 'NOR', 'team': 'McLaren', 'grid_position': 2, 'quali_position': 2},
        {'driver': 'LEC', 'team': 'Ferrari', 'grid_position': 3, 'quali_position': 3},
        {'driver': 'PIA', 'team': 'McLaren', 'grid_position': 4, 'quali_position': 4},
        {'driver': 'SAI', 'team': 'Ferrari', 'grid_position': 5, 'quali_position': 5},
        {'driver': 'HAM', 'team': 'Mercedes', 'grid_position': 6, 'quali_position': 6},
        {'driver': 'RUS', 'team': 'Mercedes', 'grid_position': 7, 'quali_position': 7},
        {'driver': 'PER', 'team': 'Red Bull Racing', 'grid_position': 8, 'quali_position': 8},
        {'driver': 'ALO', 'team': 'Aston Martin', 'grid_position': 9, 'quali_position': 9},
        {'driver': 'STR', 'team': 'Aston Martin', 'grid_position': 10, 'quali_position': 10},
    ]
    
    # You would need to add all the feature columns here
    # For now, this is a placeholder
    race_df = pd.DataFrame(drivers)
    
    # Add placeholder values for other features
    # In production, these would be calculated from recent race data
    race_df['driver_last_3_avg_position'] = [2.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    race_df['driver_last_5_avg_points'] = [22.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0]
    race_df['driver_wins_last_5'] = [3, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    race_df['driver_podiums_last_5'] = [5, 4, 3, 2, 2, 1, 1, 0, 0, 0]
    race_df['driver_dnf_rate_last_5'] = [0.0, 0.2, 0.0, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2]
    race_df['driver_season_points'] = [350, 280, 250, 220, 210, 180, 170, 140, 50, 45]
    race_df['team_last_3_avg_position'] = [2.0, 3.0, 4.0, 3.0, 4.0, 7.0, 7.0, 2.0, 10.0, 10.0]
    race_df['team_last_3_points'] = [60, 50, 45, 50, 45, 25, 25, 60, 8, 8]
    race_df['team_season_points'] = [600, 500, 460, 500, 460, 350, 350, 600, 95, 95]
    race_df['driver_circuit_avg_position'] = [1.5, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    race_df['driver_circuit_wins'] = [2, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    race_df['team_circuit_avg_position'] = [2.0, 3.5, 4.5, 3.5, 4.5, 6.5, 6.5, 2.0, 9.5, 9.5]
    race_df['driver_avg_quali_race_diff'] = [0.5, -0.5, 0.0, 1.0, 0.5, 1.0, 0.5, 2.0, 1.5, 1.0]
    
    # Encoded values (these would match your training data)
    driver_encoding = {'VER': 0, 'NOR': 1, 'LEC': 2, 'PIA': 3, 'SAI': 4, 
                      'HAM': 5, 'RUS': 6, 'PER': 7, 'ALO': 8, 'STR': 9}
    team_encoding = {'Red Bull Racing': 0, 'McLaren': 1, 'Ferrari': 2, 
                    'Mercedes': 3, 'Aston Martin': 4}
    
    race_df['driver_encoded'] = race_df['driver'].map(driver_encoding)
    race_df['team_encoded'] = race_df['team'].map(team_encoding)
    race_df['circuit_encoded'] = 0  # Would be actual circuit
    race_df['year'] = 2025
    race_df['round'] = 1
    
    return race_df


if __name__ == "__main__":
    # Load predictor
    predictor = F1Predictor()
    
    # Create sample race data
    print("\n🏎️  Creating sample race prediction...")
    race_data = create_sample_race_input()
    
    # Make predictions
    predictor.predict_podium(race_data)
    predictor.analyze_race_predictions(race_data)
    
    # Get stats for a driver
    predictor.get_driver_stats('VER')