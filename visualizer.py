"""
F1 Race Winner Prediction - Visualization Module
Creates charts and graphs for model analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from predictor import F1Predictor
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class F1Visualizer:
    def __init__(self, data_path='f1_engineered_data.csv'):
        """Initialize visualizer with data"""
        self.df = pd.read_csv(data_path)
        self.output_dir = 'visualizations'
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def plot_driver_win_distribution(self):
        """Plot distribution of wins by driver"""
        wins = self.df[self.df['won'] == 1]['driver'].value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(wins.index, wins.values, color='#FF1E00', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('Top 10 Race Winners (2018-2024)', fontsize=16, fontweight='bold')
        plt.xlabel('Driver', fontsize=12)
        plt.ylabel('Number of Wins', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/driver_wins.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir}/driver_wins.png")
        plt.close()
        
    def plot_team_performance(self):
        """Plot team performance over years"""
        team_wins = self.df[self.df['won'] == 1].groupby(['year', 'team']).size().reset_index(name='wins')
        
        # Get top teams
        top_teams = self.df[self.df['won'] == 1]['team'].value_counts().head(5).index
        team_wins_filtered = team_wins[team_wins['team'].isin(top_teams)]
        
        plt.figure(figsize=(14, 7))
        for team in top_teams:
            team_data = team_wins_filtered[team_wins_filtered['team'] == team]
            plt.plot(team_data['year'], team_data['wins'], marker='o', label=team, linewidth=2)
        
        plt.title('Team Performance Evolution (2018-2024)', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Race Wins', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/team_performance.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir}/team_performance.png")
        plt.close()
        
    def plot_grid_position_vs_results(self):
        """Analyze relationship between grid position and race results"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Grid vs Finish Position
        grid_finish = self.df.groupby('grid_position')['finish_position'].mean()
        axes[0].plot(grid_finish.index, grid_finish.values, marker='o', color='#FF1E00', linewidth=2)
        axes[0].plot([1, 20], [1, 20], 'k--', alpha=0.3, label='Perfect correlation')
        axes[0].set_title('Grid Position vs Average Finish Position', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Grid Position', fontsize=12)
        axes[0].set_ylabel('Average Finish Position', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Win probability by grid position
        grid_wins = self.df.groupby('grid_position')['won'].mean() * 100
        axes[1].bar(grid_wins.index, grid_wins.values, color='#FF1E00', alpha=0.8, edgecolor='black')
        axes[1].set_title('Win Probability by Grid Position', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Grid Position', fontsize=12)
        axes[1].set_ylabel('Win Probability (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/grid_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir}/grid_analysis.png")
        plt.close()
        
    def plot_dnf_analysis(self):
        """Analyze DNF (Did Not Finish) patterns"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: DNF rate by year
        dnf_by_year = self.df.groupby('year')['dnf'].mean() * 100
        axes[0].plot(dnf_by_year.index, dnf_by_year.values, marker='o', color='#FF1E00', linewidth=2)
        axes[0].fill_between(dnf_by_year.index, dnf_by_year.values, alpha=0.3, color='#FF1E00')
        axes[0].set_title('DNF Rate by Year', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('DNF Rate (%)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Top 10 drivers by DNF rate
        driver_dnf = self.df.groupby('driver')['dnf'].agg(['sum', 'count'])
        driver_dnf['rate'] = (driver_dnf['sum'] / driver_dnf['count']) * 100
        driver_dnf = driver_dnf[driver_dnf['count'] > 20].sort_values('rate', ascending=False).head(10)
        
        axes[1].barh(driver_dnf.index, driver_dnf['rate'], color='#FF1E00', alpha=0.8, edgecolor='black')
        axes[1].set_title('Top 10 Drivers by DNF Rate (min 20 races)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('DNF Rate (%)', fontsize=12)
        axes[1].set_ylabel('Driver', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dnf_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir}/dnf_analysis.png")
        plt.close()
        
    def plot_podium_analysis(self):
        """Analyze podium finish patterns"""
        podium_drivers = self.df[self.df['podium'] == 1]['driver'].value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(podium_drivers)))
        bars = plt.barh(podium_drivers.index, podium_drivers.values, color=colors, edgecolor='black')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}',
                    ha='left', va='center', fontweight='bold', fontsize=10)
        
        plt.title('Top 10 Drivers by Podium Finishes (2018-2024)', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Podiums', fontsize=12)
        plt.ylabel('Driver', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/podium_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir}/podium_analysis.png")
        plt.close()
        
    def plot_season_progression(self):
        """Plot season points progression for top drivers"""
        latest_year = self.df['year'].max()
        season_data = self.df[self.df['year'] == latest_year]
        
        # Get top 5 drivers by final points
        top_drivers = season_data.groupby('driver')['driver_season_points'].max().nlargest(5).index
        
        plt.figure(figsize=(14, 7))
        for driver in top_drivers:
            driver_data = season_data[season_data['driver'] == driver].sort_values('round')
            plt.plot(driver_data['round'], driver_data['driver_season_points'], 
                    marker='o', label=driver, linewidth=2, markersize=6)
        
        plt.title(f'Championship Battle - {latest_year} Season', fontsize=16, fontweight='bold')
        plt.xlabel('Race Round', fontsize=12)
        plt.ylabel('Championship Points', fontsize=12)
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/season_progression.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir}/season_progression.png")
        plt.close()
        
    def plot_circuit_difficulty(self):
        """Analyze circuit difficulty based on overtaking"""
        circuit_stats = self.df.groupby('circuit').agg({
            'quali_race_diff': 'mean',
            'won': 'count'
        }).reset_index()
        
        # Filter circuits with enough data
        circuit_stats = circuit_stats[circuit_stats['won'] > 20]
        circuit_stats = circuit_stats.sort_values('quali_race_diff')
        
        plt.figure(figsize=(14, 8))
        colors = ['green' if x < 0 else 'red' for x in circuit_stats['quali_race_diff']]
        plt.barh(circuit_stats['circuit'], circuit_stats['quali_race_diff'], 
                color=colors, alpha=0.7, edgecolor='black')
        
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.title('Circuit Overtaking Difficulty\n(Negative = Hard to overtake, Positive = Easy to overtake)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Average Position Change (Quali to Race)', fontsize=12)
        plt.ylabel('Circuit', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/circuit_difficulty.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir}/circuit_difficulty.png")
        plt.close()
        
    def plot_prediction_confidence(self, predictions_df):
        """Plot confidence distribution for predictions"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(predictions_df['win_probability'], bins=20, color='#FF1E00', 
                alpha=0.7, edgecolor='black')
        plt.title('Distribution of Win Probabilities', fontsize=14, fontweight='bold')
        plt.xlabel('Win Probability', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.subplot(1, 2, 2)
        top_5 = predictions_df.nlargest(5, 'win_probability')
        plt.bar(top_5['driver'], top_5['win_probability'] * 100, 
               color='#FF1E00', alpha=0.8, edgecolor='black')
        plt.title('Top 5 Win Probabilities', fontsize=14, fontweight='bold')
        plt.xlabel('Driver', fontsize=12)
        plt.ylabel('Win Probability (%)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/prediction_confidence.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir}/prediction_confidence.png")
        plt.close()
        
    def create_all_visualizations(self):
        """Generate all visualization reports"""
        print("\n📊 GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        self.plot_driver_win_distribution()
        self.plot_team_performance()
        self.plot_grid_position_vs_results()
        self.plot_dnf_analysis()
        self.plot_podium_analysis()
        self.plot_season_progression()
        self.plot_circuit_difficulty()
        
        print("\n✅ All visualizations created!")
        print(f"📁 Check the '{self.output_dir}/' folder for all charts")


def create_prediction_report(race_data, predictions):
    """Create a detailed prediction report with visualizations"""
    
    output_dir = 'visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Top 10 Win Probabilities
    ax1 = fig.add_subplot(gs[0, :])
    top_10 = predictions.head(10)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_10)))
    bars = ax1.barh(top_10['driver'], top_10['win_probability'] * 100, 
                    color=colors, edgecolor='black')
    ax1.set_xlabel('Win Probability (%)', fontsize=12)
    ax1.set_ylabel('Driver', fontsize=12)
    ax1.set_title('Top 10 Race Winner Predictions', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%',
                ha='left', va='center', fontweight='bold')
    
    # Plot 2: Grid Position vs Probability
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(predictions['grid_position'], predictions['win_probability'] * 100,
               s=100, alpha=0.6, c=predictions['win_probability'], cmap='RdYlGn')
    ax2.set_xlabel('Grid Position', fontsize=11)
    ax2.set_ylabel('Win Probability (%)', fontsize=11)
    ax2.set_title('Grid Position vs Win Probability', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Team Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    team_probs = predictions.groupby('team')['win_probability'].sum() * 100
    team_probs = team_probs.nlargest(5)
    ax3.pie(team_probs.values, labels=team_probs.index, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 9})
    ax3.set_title('Team Win Probability Distribution', fontsize=12, fontweight='bold')
    
    # Plot 4: Prediction Table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for idx, row in predictions.head(10).iterrows():
        table_data.append([
            f"{len(table_data) + 1}",
            row['driver'],
            row['team'][:20],  # Truncate long team names
            f"{row['grid_position']:.0f}",
            f"{row['win_probability']*100:.1f}%"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Rank', 'Driver', 'Team', 'Grid', 'Win Prob'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.1, 0.15, 0.35, 0.1, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(5):
        table[(0, i)].set_facecolor('#FF1E00')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('F1 Race Prediction Report', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(f'{output_dir}/prediction_report.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Prediction report saved: {output_dir}/prediction_report.png")
    plt.close()


if __name__ == "__main__":
    print("🎨 F1 Prediction Visualizer")
    print("=" * 70)
    
    # Check if data exists
    if not os.path.exists('f1_engineered_data.csv'):
        print("❌ No data found! Please run the pipeline first:")
        print("   python main_pipeline.py --full")
        exit(1)
    
    # Create visualizer
    visualizer = F1Visualizer()
    
    # Generate all visualizations
    visualizer.create_all_visualizations()
    
    # If model exists, create prediction visualization
    if os.path.exists('best_model.pkl'):
        print("\n📊 Creating prediction visualization...")
        from predictor import F1Predictor, create_sample_race_input
        
        predictor = F1Predictor()
        race_data = create_sample_race_input()
        predictions = predictor.predict_race(race_data)
        
        create_prediction_report(race_data, predictions)
        visualizer.plot_prediction_confidence(predictions)
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print(f"📁 All charts saved in: visualizations/")
    print("\nGenerated files:")
    print("  - driver_wins.png")
    print("  - team_performance.png")
    print("  - grid_analysis.png")
    print("  - dnf_analysis.png")
    print("  - podium_analysis.png")
    print("  - season_progression.png")
    print("  - circuit_difficulty.png")
    print("  - prediction_report.png")
    print("  - prediction_confidence.png")
    print("=" * 70)