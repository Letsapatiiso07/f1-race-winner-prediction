"""
F1 Race Winner Prediction - Model Training Module
Trains and evaluates multiple ML models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class F1ModelTrainer:
    def __init__(self, df, feature_cols, target_col='won'):
        """
        Initialize model trainer
        
        Args:
            df: DataFrame with engineered features
            feature_cols: List of feature column names
            target_col: Target variable column name
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.models = {}
        self.results = {}
        
        # Prepare data
        self.X = self.df[feature_cols]
        self.y = self.df[target_col]
        
        # Split data - use temporal split (last season for testing)
        self.temporal_split()
        
    def temporal_split(self):
        """Split data temporally - train on older races, test on recent"""
        # Check what years we have
        available_years = sorted(self.df['year'].unique())
        
        if len(available_years) == 0:
            raise ValueError("No data available for training!")
        
        # Use most recent year as test set, or last 20% of data if only one year
        if len(available_years) > 1:
            test_year = available_years[-1]
            test_mask = self.df['year'] == test_year
        else:
            # If only one year, use last 20% of races as test set
            n_test = int(len(self.df) * 0.2)
            test_mask = pd.Series([False] * len(self.df))
            test_mask.iloc[-n_test:] = True
            test_year = available_years[0]
        
        train_mask = ~test_mask
        
        self.X_train = self.X[train_mask]
        self.X_test = self.X[test_mask]
        self.y_train = self.y[train_mask]
        self.y_test = self.y[test_mask]
        
        # Check if we have enough data
        if len(self.X_test) == 0:
            print("⚠️  Warning: No test data available. Using 80/20 random split instead.")
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
        
        print(f"📊 Training set: {len(self.X_train)} samples")
        print(f"📊 Test set: {len(self.X_test)} samples")
        print(f"🎯 Target distribution (train): {self.y_train.value_counts().to_dict()}")
        print(f"🎯 Target distribution (test): {self.y_test.value_counts().to_dict()}")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf
        
        # Predictions
        y_pred = rf.predict(self.X_test)
        y_pred_proba = rf.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        self.results['random_forest'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"✅ Random Forest Accuracy: {self.results['random_forest']['accuracy']:.4f}")
        
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n🚀 Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced classes
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        
        xgb.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb
        
        # Predictions
        y_pred = xgb.predict(self.X_test)
        y_pred_proba = xgb.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        self.results['xgboost'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"✅ XGBoost Accuracy: {self.results['xgboost']['accuracy']:.4f}")
        
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n📈 Training Gradient Boosting...")
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        gb.fit(self.X_train, self.y_train)
        self.models['gradient_boosting'] = gb
        
        # Predictions
        y_pred = gb.predict(self.X_test)
        y_pred_proba = gb.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        self.results['gradient_boosting'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"✅ Gradient Boosting Accuracy: {self.results['gradient_boosting']['accuracy']:.4f}")
        
    def train_all_models(self):
        """Train all models"""
        print("🏎️  Training all models...\n")
        
        self.train_random_forest()
        self.train_xgboost()
        self.train_gradient_boosting()
        
        print("\n" + "="*50)
        print("📊 MODEL COMPARISON")
        print("="*50)
        for model_name, results in self.results.items():
            print(f"{model_name.upper()}: {results['accuracy']:.4f}")
            
    def evaluate_best_model(self):
        """Detailed evaluation of best performing model"""
        # Find best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]
        best_results = self.results[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name.upper()}")
        print("="*50)
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, best_results['predictions'], 
                                   target_names=['No Win', 'Win']))
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n🔝 Top 10 Important Features:")
            print(feature_importance_df.head(10).to_string(index=False))
            
            # Save feature importance plot
            self.plot_feature_importance(feature_importance_df)
            
        return best_model_name, best_model
    
    def plot_feature_importance(self, feature_importance_df):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(15)
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n💾 Feature importance plot saved as 'feature_importance.png'")
        plt.close()
        
    def save_best_model(self, model_name='best_model.pkl'):
        """Save the best model"""
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]
        
        model_data = {
            'model': best_model,
            'model_name': best_model_name,
            'feature_cols': self.feature_cols,
            'accuracy': self.results[best_model_name]['accuracy']
        }
        
        joblib.dump(model_data, model_name)
        print(f"\n💾 Best model saved as '{model_name}'")
        
        return model_name


if __name__ == "__main__":
    # Load engineered data
    df = pd.read_csv('f1_engineered_data.csv')
    
    # Define features
    feature_cols = [
        'grid_position', 'quali_position',
        'driver_last_3_avg_position', 'driver_last_5_avg_points',
        'driver_wins_last_5', 'driver_podiums_last_5',
        'driver_dnf_rate_last_5', 'driver_season_points',
        'team_last_3_avg_position', 'team_last_3_points',
        'team_season_points', 'driver_circuit_avg_position',
        'driver_circuit_wins', 'team_circuit_avg_position',
        'driver_avg_quali_race_diff', 'driver_encoded',
        'team_encoded', 'circuit_encoded', 'year', 'round'
    ]
    
    # Train models
    trainer = F1ModelTrainer(df, feature_cols)
    trainer.train_all_models()
    trainer.evaluate_best_model()
    trainer.save_best_model()
    
    print("\n🏁 Training complete!")