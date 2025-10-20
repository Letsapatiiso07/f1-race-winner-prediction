"""
F1 Race Winner Prediction - Data Collection Module
Collects historical F1 data for model training
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Create cache directory if it doesn't exist
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"✅ Created cache directory: {cache_dir}")

# Enable FastF1 cache for faster data loading
fastf1.Cache.enable_cache(cache_dir)


class F1DataCollector:
    def __init__(self, start_year=2018, end_year=2024):
        """
        Initialize the data collector
        
        Args:
            start_year: Starting year for data collection
            end_year: Ending year for data collection
        """
        self.start_year = start_year
        self.end_year = end_year
        self.race_data = []
        
    def collect_season_data(self, year):
        """Collect all race data for a given season"""
        print(f"\n📊 Collecting data for {year} season...")
        
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'] != 'testing']
            
            for idx, race in races.iterrows():
                try:
                    # Skip if race hasn't happened yet
                    if pd.isna(race['EventDate']) or race['EventDate'] > datetime.now():
                        continue
                    
                    print(f"  - Processing: {race['EventName']}")
                    
                    # Load race session
                    session = fastf1.get_session(year, race['RoundNumber'], 'R')
                    session.load()
                    
                    # Load qualifying session
                    quali = fastf1.get_session(year, race['RoundNumber'], 'Q')
                    quali.load()
                    
                    # Get results
                    results = session.results
                    quali_results = quali.results
                    
                    # Process each driver's result
                    for driver_idx, driver in results.iterrows():
                        race_record = self._create_race_record(
                            year, race, driver, results, quali_results
                        )
                        if race_record:
                            self.race_data.append(race_record)
                            
                except Exception as e:
                    print(f"    ⚠️  Error processing {race['EventName']}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error loading {year} season: {str(e)}")
            
    def _create_race_record(self, year, race, driver, results, quali_results):
        """Create a single race record for a driver"""
        try:
            # Get driver's qualifying position
            quali_driver = quali_results[quali_results['Abbreviation'] == driver['Abbreviation']]
            quali_pos = quali_driver['Position'].values[0] if len(quali_driver) > 0 else np.nan
            
            record = {
                'year': year,
                'round': race['RoundNumber'],
                'race_name': race['EventName'],
                'circuit': race['Location'],
                'driver': driver['Abbreviation'],
                'driver_number': driver['DriverNumber'],
                'team': driver['TeamName'],
                'grid_position': driver['GridPosition'],
                'quali_position': quali_pos,
                'finish_position': driver['Position'],
                'points': driver['Points'],
                'status': driver['Status'],
                'won': 1 if driver['Position'] == 1 else 0,
                'podium': 1 if driver['Position'] <= 3 else 0,
                'dnf': 0 if driver['Status'] == 'Finished' else 1,
            }
            
            return record
            
        except Exception as e:
            return None
            
    def collect_all_data(self):
        """Collect data for all seasons"""
        print(f"🏎️  Starting F1 data collection from {self.start_year} to {self.end_year}")
        
        for year in range(self.start_year, self.end_year + 1):
            self.collect_season_data(year)
            
        print(f"\n✅ Data collection complete! Collected {len(self.race_data)} race records")
        
    def get_dataframe(self):
        """Convert collected data to pandas DataFrame"""
        return pd.DataFrame(self.race_data)
    
    def save_data(self, filename='f1_race_data.csv'):
        """Save collected data to CSV"""
        df = self.get_dataframe()
        df.to_csv(filename, index=False)
        print(f"💾 Data saved to {filename}")
        return df


if __name__ == "__main__":
    # Collect data from 2018-2024
    collector = F1DataCollector(start_year=2018, end_year=2024)
    collector.collect_all_data()
    df = collector.save_data()
    
    print(f"\n📈 Dataset shape: {df.shape}")
    print(f"📊 Sample data:\n{df.head()}")
    print(f"\n🏆 Winners distribution:\n{df[df['won']==1]['driver'].value_counts()}")