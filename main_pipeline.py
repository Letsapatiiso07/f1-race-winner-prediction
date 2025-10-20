"""
F1 Race Winner Prediction - Main Pipeline
Complete pipeline from data collection to prediction
"""

import os
import sys
from data_collector import F1DataCollector
from feature_engineering import F1FeatureEngineer
from model_trainer import F1ModelTrainer
from predictor import F1Predictor, create_sample_race_input


def run_full_pipeline():
    """Run the complete ML pipeline"""
    
    print("🏁" * 35)
    print("     F1 RACE WINNER PREDICTION MODEL - FULL PIPELINE")
    print("🏁" * 35)
    
    # Step 1: Data Collection
    print("\n" + "="*70)
    print("STEP 1: DATA COLLECTION")
    print("="*70)
    
    if not os.path.exists('f1_race_data.csv'):
        print("📥 Collecting historical F1 data...")
        collector = F1DataCollector(start_year=2018, end_year=2024)
        collector.collect_all_data()
        df_raw = collector.save_data()
    else:
        print("✅ Found existing data file, skipping collection")
        print("   (Delete 'f1_race_data.csv' to re-collect)")
        import pandas as pd
        df_raw = pd.read_csv('f1_race_data.csv')
    
    print(f"📊 Dataset: {len(df_raw)} records collected")
    
    # Step 2: Feature Engineering
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    if not os.path.exists('f1_engineered_data.csv'):
        print("🔧 Engineering features...")
        engineer = F1FeatureEngineer(df_raw)
        df_engineered = engineer.create_all_features()
        df_engineered.to_csv('f1_engineered_data.csv', index=False)
        feature_cols = engineer.get_feature_columns()
    else:
        print("✅ Found existing engineered data, skipping feature engineering")
        print("   (Delete 'f1_engineered_data.csv' to re-engineer)")
        import pandas as pd
        df_engineered = pd.read_csv('f1_engineered_data.csv')
        engineer = F1FeatureEngineer(df_engineered)
        feature_cols = engineer.get_feature_columns()
    
    print(f"📊 Features created: {len(feature_cols)} features")
    
    # Step 3: Model Training
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING & EVALUATION")
    print("="*70)
    
    if not os.path.exists('best_model.pkl'):
        print("🤖 Training machine learning models...")
        trainer = F1ModelTrainer(df_engineered, feature_cols)
        trainer.train_all_models()
        trainer.evaluate_best_model()
        trainer.save_best_model()
    else:
        print("✅ Found existing trained model, skipping training")
        print("   (Delete 'best_model.pkl' to retrain)")
    
    # Step 4: Make Predictions
    print("\n" + "="*70)
    print("STEP 4: MAKING PREDICTIONS")
    print("="*70)
    
    predictor = F1Predictor()
    
    # Create sample race for prediction
    race_data = create_sample_race_input()
    
    print("\n🏁 Predicting next race winner...")
    predictor.predict_podium(race_data)
    
    print("\n📊 Detailed race analysis:")
    predictor.analyze_race_predictions(race_data, top_n=10)
    
    # Show some driver stats
    print("\n📈 Sample driver statistics:")
    for driver in ['VER', 'HAM', 'LEC']:
        predictor.get_driver_stats(driver)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   - f1_race_data.csv         (Raw race data)")
    print("   - f1_engineered_data.csv   (Engineered features)")
    print("   - best_model.pkl           (Trained model)")
    print("   - feature_importance.png   (Feature analysis)")
    print("\n💡 Next steps:")
    print("   - Use predictor.py to make predictions for upcoming races")
    print("   - Update data with new races as the season progresses")
    print("   - Retrain model periodically with fresh data")
    

def quick_predict():
    """Quick prediction without full pipeline"""
    print("🏎️  QUICK PREDICTION MODE")
    print("="*70)
    
    if not os.path.exists('best_model.pkl'):
        print("❌ No trained model found!")
        print("   Run: python main_pipeline.py --full")
        return
    
    predictor = F1Predictor()
    race_data = create_sample_race_input()
    
    predictor.predict_podium(race_data)
    predictor.analyze_race_predictions(race_data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Race Winner Prediction')
    parser.add_argument('--full', action='store_true', 
                       help='Run full pipeline (data collection, training, prediction)')
    parser.add_argument('--predict', action='store_true',
                       help='Quick prediction only (requires trained model)')
    
    args = parser.parse_args()
    
    if args.predict:
        quick_predict()
    elif args.full:
        run_full_pipeline()
    else:
        # Default: run full pipeline
        run_full_pipeline()