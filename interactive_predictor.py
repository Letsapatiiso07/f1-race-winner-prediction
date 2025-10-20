"""
F1 Race Winner Prediction - Interactive CLI
Simple command-line interface for making predictions
"""

import pandas as pd
import os
from predictor import F1Predictor


class InteractiveF1Predictor:
    def __init__(self):
        """Initialize interactive predictor"""
        if not os.path.exists('best_model.pkl'):
            print("❌ No trained model found!")
            print("   Please run: python main_pipeline.py --full")
            exit(1)
        
        self.predictor = F1Predictor()
        self.current_race = None
        
    def show_menu(self):
        """Display main menu"""
        print("\n" + "🏎️ " * 30)
        print("     F1 RACE WINNER PREDICTION - INTERACTIVE MODE")
        print("🏎️ " * 30)
        print("\nOptions:")
        print("  1. Predict race winner (manual input)")
        print("  2. View driver statistics")
        print("  3. Quick prediction with sample data")
        print("  4. Compare drivers")
        print("  5. Exit")
        print("\n" + "-" * 60)
        
    def get_race_input(self):
        """Get race information from user"""
        print("\n📝 ENTER RACE INFORMATION")
        print("-" * 60)
        
        race_name = input("Race name (e.g., Monaco GP): ").strip()
        if not race_name:
            race_name = "Upcoming Race"
        
        num_drivers = input("Number of drivers to predict (default 10): ").strip()
        num_drivers = int(num_drivers) if num_drivers else 10
        
        drivers_data = []
        print(f"\nEnter data for {num_drivers} drivers:")
        print("(Press Enter to use default values)\n")
        
        for i in range(num_drivers):
            print(f"\n--- Driver {i+1} ---")
            
            driver = input("  Driver code (e.g., VER, HAM, LEC): ").strip().upper()
            if not driver:
                print("  ⚠️  Skipping driver (no code provided)")
                continue
                
            team = input(f"  Team for {driver}: ").strip()
            if not team:
                team = "Unknown Team"
            
            grid = input(f"  Grid position for {driver} (1-20): ").strip()
            grid = int(grid) if grid else i + 1
            
            quali = input(f"  Quali position for {driver} (1-20): ").strip()
            quali = int(quali) if quali else i + 1
            
            # Use defaults for complex features
            driver_data = {
                'driver': driver,
                'team': team,
                'grid_position': grid,
                'quali_position': quali,
                'driver_last_3_avg_position': grid * 1.2,
                'driver_last_5_avg_points': max(0, 25 - grid * 2),
                'driver_wins_last_5': max(0, 3 - grid // 3),
                'driver_podiums_last_5': max(0, 5 - grid // 2),
                'driver_dnf_rate_last_5': min(0.4, grid * 0.02),
                'driver_season_points': max(0, 400 - grid * 20),
                'team_last_3_avg_position': grid * 1.1,
                'team_last_3_points': max(0, 50 - grid * 3),
                'team_season_points': max(0, 700 - grid * 30),
                'driver_circuit_avg_position': grid * 1.15,
                'driver_circuit_wins': max(0, 2 - grid // 5),
                'team_circuit_avg_position': grid * 1.1,
                'driver_avg_quali_race_diff': 0.5,
                'driver_encoded': hash(driver) % 100,
                'team_encoded': hash(team) % 50,
                'circuit_encoded': 0,
                'year': 2025,
                'round': 1
            }
            
            drivers_data.append(driver_data)
        
        if not drivers_data:
            print("❌ No valid driver data entered!")
            return None
        
        return pd.DataFrame(drivers_data)
    
    def predict_winner(self):
        """Get race input and predict winner"""
        race_data = self.get_race_input()
        
        if race_data is None:
            return
        
        print("\n🔮 GENERATING PREDICTIONS...")
        print("=" * 70)
        
        self.predictor.predict_podium(race_data)
        self.predictor.analyze_race_predictions(race_data, top_n=len(race_data))
        
    def view_driver_stats(self):
        """View statistics for a specific driver"""
        print("\n📊 DRIVER STATISTICS")
        print("-" * 60)
        
        driver = input("Enter driver code (e.g., VER, HAM, LEC): ").strip().upper()
        
        if not driver:
            print("❌ No driver code entered!")
            return
        
        self.predictor.get_driver_stats(driver)
        
    def quick_prediction(self):
        """Quick prediction with sample data"""
        from predictor import create_sample_race_input
        
        print("\n🚀 QUICK PREDICTION (Sample Data)")
        print("=" * 70)
        
        race_data = create_sample_race_input()
        self.predictor.predict_podium(race_data)
        self.predictor.analyze_race_predictions(race_data)
        
    def compare_drivers(self):
        """Compare statistics of multiple drivers"""
        print("\n⚖️  COMPARE DRIVERS")
        print("-" * 60)
        
        drivers = input("Enter driver codes separated by commas (e.g., VER,HAM,LEC): ")
        drivers = [d.strip().upper() for d in drivers.split(',') if d.strip()]
        
        if len(drivers) < 2:
            print("❌ Please enter at least 2 drivers to compare!")
            return
        
        print("\n" + "=" * 100)
        print(f"{'Metric':<30}", end="")
        for driver in drivers:
            print(f"{driver:>15}", end="")
        print("\n" + "=" * 100)
        
        metrics = {}
        for driver in drivers:
            driver_data = self.predictor.historical_data[
                self.predictor.historical_data['driver'] == driver
            ]
            
            if len(driver_data) == 0:
                metrics[driver] = None
            else:
                metrics[driver] = {
                    'Races': len(driver_data),
                    'Wins': driver_data['won'].sum(),
                    'Win Rate': driver_data['won'].mean(),
                    'Podiums': driver_data['podium'].sum(),
                    'Podium Rate': driver_data['podium'].mean(),
                    'Avg Finish': driver_data['finish_position'].mean(),
                    'DNF Rate': driver_data['dnf'].mean(),
                }
        
        # Print comparison table
        metric_names = ['Races', 'Wins', 'Win Rate', 'Podiums', 'Podium Rate', 'Avg Finish', 'DNF Rate']
        
        for metric in metric_names:
            print(f"{metric:<30}", end="")
            for driver in drivers:
                if metrics[driver] is None:
                    print(f"{'N/A':>15}", end="")
                else:
                    value = metrics[driver][metric]
                    if metric in ['Win Rate', 'Podium Rate', 'DNF Rate']:
                        print(f"{value:>14.1%}", end="")
                    elif metric == 'Avg Finish':
                        print(f"{value:>14.2f}", end="")
                    else:
                        print(f"{value:>15.0f}", end="")
            print()
        
        print("=" * 100)
        
    def run(self):
        """Run interactive CLI"""
        while True:
            self.show_menu()
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self.predict_winner()
            elif choice == '2':
                self.view_driver_stats()
            elif choice == '3':
                self.quick_prediction()
            elif choice == '4':
                self.compare_drivers()
            elif choice == '5':
                print("\n👋 Thanks for using F1 Predictor! See you at the track! 🏁")
                break
            else:
                print("❌ Invalid choice! Please enter 1-5.")
            
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        cli = InteractiveF1Predictor()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted! Goodbye! 🏁")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure you've run the pipeline first: python main_pipeline.py --full")