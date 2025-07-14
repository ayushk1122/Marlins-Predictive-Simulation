import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_data_and_model():
    """Load the holdout data and outcome model"""
    print("Loading data and model...")
    
    # Load holdout data
    df = pd.read_csv('ronald_acuna_jr_holdout_statcast.csv')
    print(f"✓ Loaded holdout data with {len(df)} pitches")
    
    # Load outcome model
    with open('sequential_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    outcome_model = models.get('swing_outcome_model')
    outcome_preprocessor = models.get('swing_outcome_preprocessor')
    outcome_le = models.get('swing_outcome_le')
    
    if outcome_model is None:
        print("✗ Outcome model not found")
        return None, None, None, None
    
    print("✓ Loaded outcome model")
    return df, outcome_model, outcome_preprocessor, outcome_le

def prepare_features_for_analysis(df):
    """Prepare features for analysis - focusing on hitting events"""
    print("Preparing features for analysis...")
    
    # Focus only on hitting events
    hitting_events = [
        'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
        'missed_bunt', 'bunt_foul_tip', 'single', 'double', 'triple', 'home_run',
        'groundout', 'force_out', 'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
        'field_error', 'fielders_choice', 'fielders_choice_out', 'sac_fly_double_play',
        'sac_bunt_double_play', 'grounded_into_double_play', 'batter_interference',
        'catcher_interference', 'fan_interference', 'strikeout', 'strikeout_double_play', 
        'strikeout_triple_play', 'walk', 'intent_walk', 'hit_by_pitch',
        'sacrifice_bunt_double_play', 'sacrifice_bunt_triple_play', 'umpire_interference'
    ]
    
    # Filter to only hitting events
    swing_df = df[df['events'].isin(hitting_events)].copy()
    print(f"Found {len(swing_df)} hitting events")
    
    # Create outcome labels
    outcome_mapping = {
        'swinging_strike': 'whiff',
        'swinging_strike_blocked': 'whiff',
        'foul': 'field_out',
        'foul_tip': 'whiff',
        'foul_bunt': 'field_out',
        'missed_bunt': 'whiff',
        'bunt_foul_tip': 'whiff',
        'single': 'hit_safely',
        'double': 'hit_safely',
        'triple': 'hit_safely',
        'home_run': 'hit_safely',
        'groundout': 'field_out',
        'force_out': 'field_out',
        'double_play': 'field_out',
        'triple_play': 'field_out',
        'sac_fly': 'field_out',
        'sac_bunt': 'field_out',
        'field_error': 'field_out',
        'fielders_choice': 'field_out',
        'fielders_choice_out': 'field_out',
        'sac_fly_double_play': 'field_out',
        'sac_bunt_double_play': 'field_out',
        'grounded_into_double_play': 'field_out',
        'batter_interference': 'field_out',
        'catcher_interference': 'field_out',
        'fan_interference': 'field_out',
        'strikeout': 'whiff',
        'strikeout_double_play': 'whiff',
        'strikeout_triple_play': 'whiff',
        'walk': 'field_out',
        'intent_walk': 'field_out',
        'hit_by_pitch': 'field_out',
        'sacrifice_bunt_double_play': 'field_out',
        'sacrifice_bunt_triple_play': 'field_out',
        'umpire_interference': 'field_out'
    }
    
    # Map outcomes
    swing_df['outcome'] = swing_df['events'].map(outcome_mapping)
    swing_df = swing_df.dropna(subset=['outcome'])
    
    # Engineer basic features
    swing_df = engineer_basic_features(swing_df)
    
    return swing_df

def engineer_basic_features(df):
    """Engineer basic features for analysis"""
    print("Engineering basic features...")
    
    # Movement calculation
    if 'horizontal_break' in df.columns and 'vertical_break' in df.columns:
        df['movement_magnitude'] = np.sqrt(df['horizontal_break']**2 + df['vertical_break']**2)
    elif 'pfx_x' in df.columns and 'pfx_z' in df.columns:
        df['movement_magnitude'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    else:
        df['movement_magnitude'] = df['release_spin_rate'] / 1000
    
    # Basic features
    df['count_total'] = df['balls'] + df['strikes']
    df['zone_distance'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - 2.5)**2)
    df['in_strike_zone'] = ((df['plate_x'].abs() <= 0.7) & (df['plate_z'] >= 1.5) & (df['plate_z'] <= 3.5)).astype(int)
    df['two_strikes'] = (df['strikes'] >= 2).astype(int)
    df['three_balls'] = (df['balls'] >= 3).astype(int)
    df['full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
    
    # Pitch type features
    df['is_fastball'] = df['pitch_type'].isin(['FF', 'SI', 'FC', 'FT']).astype(int)
    df['is_breaking_ball'] = df['pitch_type'].isin(['SL', 'CU', 'KC', 'SV']).astype(int)
    df['is_offspeed'] = df['pitch_type'].isin(['CH', 'FS', 'FO']).astype(int)
    
    # Velocity features
    df['high_velocity'] = (df['release_speed'] > 95).astype(int)
    df['low_velocity'] = (df['release_speed'] < 85).astype(int)
    
    # Location features
    df['high_pitch'] = (df['plate_z'] > 3.0).astype(int)
    df['low_pitch'] = (df['plate_z'] < 2.0).astype(int)
    df['inside_pitch'] = (df['plate_x'] < -0.5).astype(int)
    df['outside_pitch'] = (df['plate_x'] > 0.5).astype(int)
    
    return df

def analyze_feature_distributions(df, y_true, y_pred):
    """Analyze feature distributions by outcome and prediction accuracy"""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Add prediction columns
    df['true_outcome'] = y_true
    df['predicted_outcome'] = y_pred
    df['correct_prediction'] = (df['true_outcome'] == df['predicted_outcome']).astype(int)
    
    # Key features to analyze
    key_features = [
        'release_speed', 'movement_magnitude', 'zone_distance', 'plate_x', 'plate_z',
        'balls', 'strikes', 'count_total', 'in_strike_zone', 'two_strikes', 'three_balls',
        'is_fastball', 'is_breaking_ball', 'is_offspeed', 'high_velocity', 'low_velocity',
        'high_pitch', 'low_pitch', 'inside_pitch', 'outside_pitch'
    ]
    
    print("\nFEATURE DISTRIBUTIONS BY TRUE OUTCOME:")
    print("-" * 40)
    
    for feature in key_features:
        if feature in df.columns:
            print(f"\n{feature.upper()}:")
            for outcome in ['whiff', 'hit_safely', 'field_out']:
                outcome_data = df[df['true_outcome'] == outcome][feature]
                if len(outcome_data) > 0:
                    mean_val = outcome_data.mean()
                    std_val = outcome_data.std()
                    print(f"  {outcome}: {mean_val:.3f} ± {std_val:.3f} (n={len(outcome_data)})")
    
    print("\nFEATURE DISTRIBUTIONS BY PREDICTION ACCURACY:")
    print("-" * 40)
    
    for feature in key_features:
        if feature in df.columns:
            print(f"\n{feature.upper()}:")
            correct_data = df[df['correct_prediction'] == 1][feature]
            incorrect_data = df[df['correct_prediction'] == 0][feature]
            
            if len(correct_data) > 0:
                correct_mean = correct_data.mean()
                correct_std = correct_data.std()
                print(f"  Correct predictions: {correct_mean:.3f} ± {correct_std:.3f} (n={len(correct_data)})")
            
            if len(incorrect_data) > 0:
                incorrect_mean = incorrect_data.mean()
                incorrect_std = incorrect_data.std()
                print(f"  Incorrect predictions: {incorrect_mean:.3f} ± {incorrect_std:.3f} (n={len(incorrect_data)})")

def analyze_misclassification_patterns(df, y_true, y_pred):
    """Analyze specific misclassification patterns"""
    print("\n" + "="*60)
    print("MISCLASSIFICATION PATTERN ANALYSIS")
    print("="*60)
    
    # Add prediction columns
    df['true_outcome'] = y_true
    df['predicted_outcome'] = y_pred
    df['correct_prediction'] = (df['true_outcome'] == df['predicted_outcome']).astype(int)
    
    # Get misclassified data
    misclassified = df[df['correct_prediction'] == 0].copy()
    
    print(f"\nTotal misclassifications: {len(misclassified)}")
    
    # Analyze by pitch type
    print("\nMISCLASSIFICATIONS BY PITCH TYPE:")
    print("-" * 40)
    pitch_errors = misclassified.groupby(['pitch_type', 'true_outcome', 'predicted_outcome']).size().sort_values(ascending=False)
    for (pitch_type, true_out, pred_out), count in pitch_errors.head(15).items():
        print(f"  {pitch_type}: {true_out} → {pred_out} ({count} times)")
    
    # Analyze by zone
    print("\nMISCLASSIFICATIONS BY ZONE:")
    print("-" * 40)
    zone_errors = misclassified.groupby(['zone', 'true_outcome', 'predicted_outcome']).size().sort_values(ascending=False)
    for (zone, true_out, pred_out), count in zone_errors.head(15).items():
        print(f"  Zone {zone}: {true_out} → {pred_out} ({count} times)")
    
    # Analyze by count
    print("\nMISCLASSIFICATIONS BY COUNT:")
    print("-" * 40)
    count_errors = misclassified.groupby(['balls', 'strikes', 'true_outcome', 'predicted_outcome']).size().sort_values(ascending=False)
    for (balls, strikes, true_out, pred_out), count in count_errors.head(15).items():
        print(f"  {balls}-{strikes}: {true_out} → {pred_out} ({count} times)")
    
    # Analyze by velocity ranges
    print("\nMISCLASSIFICATIONS BY VELOCITY RANGE:")
    print("-" * 40)
    misclassified['velocity_range'] = pd.cut(misclassified['release_speed'], 
                                           bins=[0, 80, 85, 90, 95, 100, 110], 
                                           labels=['<80', '80-85', '85-90', '90-95', '95-100', '>100'])
    vel_errors = misclassified.groupby(['velocity_range', 'true_outcome', 'predicted_outcome']).size().sort_values(ascending=False)
    for (vel_range, true_out, pred_out), count in vel_errors.head(15).items():
        print(f"  {vel_range}: {true_out} → {pred_out} ({count} times)")

def analyze_class_imbalance(df, y_true, y_pred):
    """Analyze class imbalance issues"""
    print("\n" + "="*60)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*60)
    
    # True outcome distribution
    true_counts = pd.Series(y_true).value_counts()
    print(f"\nTRUE OUTCOME DISTRIBUTION:")
    for outcome, count in true_counts.items():
        print(f"  {outcome}: {count} ({count/len(y_true)*100:.1f}%)")
    
    # Predicted outcome distribution
    pred_counts = pd.Series(y_pred).value_counts()
    print(f"\nPREDICTED OUTCOME DISTRIBUTION:")
    for outcome, count in pred_counts.items():
        print(f"  {outcome}: {count} ({count/len(y_pred)*100:.1f}%)")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_true, y_pred, labels=['whiff', 'hit_safely', 'field_out'])
    print(f"\nCONFUSION MATRIX:")
    print("True\Pred\twhiff\thit_safely\tfield_out")
    outcomes = ['whiff', 'hit_safely', 'field_out']
    for i, true_out in enumerate(outcomes):
        print(f"{true_out}\t" + "\t".join([str(cm[i][j]) for j in range(3)]))
    
    # Calculate per-class accuracy
    print(f"\nPER-CLASS ACCURACY:")
    for i, outcome in enumerate(outcomes):
        true_pos = cm[i][i]
        total = cm[i].sum()
        accuracy = true_pos / total if total > 0 else 0
        print(f"  {outcome}: {accuracy:.3f} ({true_pos}/{total})")

def suggest_improvements(df, y_true, y_pred):
    """Suggest specific improvements based on analysis"""
    print("\n" + "="*60)
    print("IMPROVEMENT SUGGESTIONS")
    print("="*60)
    
    # Add prediction columns
    df['true_outcome'] = y_true
    df['predicted_outcome'] = y_pred
    df['correct_prediction'] = (df['true_outcome'] == df['predicted_outcome']).astype(int)
    
    print("\n1. CLASS IMBALANCE ISSUES:")
    print("-" * 30)
    
    # Analyze class imbalance
    true_counts = pd.Series(y_true).value_counts()
    pred_counts = pd.Series(y_pred).value_counts()
    
    print("The model shows severe class imbalance:")
    for outcome in ['whiff', 'hit_safely', 'field_out']:
        true_pct = true_counts.get(outcome, 0) / len(y_true) * 100
        pred_pct = pred_counts.get(outcome, 0) / len(y_pred) * 100
        print(f"  {outcome}: True={true_pct:.1f}%, Predicted={pred_pct:.1f}%")
    
    print("\nSUGGESTIONS:")
    print("  - Use class weights in training")
    print("  - Implement SMOTE or other oversampling techniques")
    print("  - Use balanced accuracy metrics")
    
    print("\n2. FEATURE ENGINEERING ISSUES:")
    print("-" * 30)
    
    # Analyze feature importance for misclassifications
    misclassified = df[df['correct_prediction'] == 0]
    
    if len(misclassified) > 0:
        print("Key features that may need improvement:")
        
        # Check for missing or problematic features
        key_features = ['release_speed', 'movement_magnitude', 'zone_distance', 'pitch_type']
        for feature in key_features:
            if feature in misclassified.columns:
                missing_pct = misclassified[feature].isna().sum() / len(misclassified) * 100
                if missing_pct > 10:
                    print(f"  - {feature}: {missing_pct:.1f}% missing values")
        
        # Check for extreme values
        for feature in ['release_speed', 'movement_magnitude']:
            if feature in misclassified.columns:
                q99 = misclassified[feature].quantile(0.99)
                q01 = misclassified[feature].quantile(0.01)
                print(f"  - {feature}: 99th percentile = {q99:.2f}, 1st percentile = {q01:.2f}")
    
    print("\nSUGGESTIONS:")
    print("  - Add more granular pitch type features")
    print("  - Improve movement calculation")
    print("  - Add sequence-based features")
    print("  - Include more situational features")
    
    print("\n3. MODEL ARCHITECTURE ISSUES:")
    print("-" * 30)
    
    print("Current issues identified:")
    print("  - Model is heavily biased toward certain predictions")
    print("  - Poor performance on minority classes")
    print("  - Inadequate feature representation")
    
    print("\nSUGGESTIONS:")
    print("  - Try ensemble methods (Random Forest, Gradient Boosting)")
    print("  - Use neural networks with dropout")
    print("  - Implement probability calibration")
    print("  - Add regularization to prevent overfitting")
    
    print("\n4. DATA QUALITY ISSUES:")
    print("-" * 30)
    
    # Check for data quality issues
    print("Data quality checks:")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    high_missing = missing_counts[missing_counts > len(df) * 0.1]
    if len(high_missing) > 0:
        print("  - High missing values in:", list(high_missing.index))
    
    # Check for extreme values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            if q99 - q01 > 100:  # Large range
                print(f"  - {col}: Large value range ({q01:.2f} to {q99:.2f})")
    
    print("\nSUGGESTIONS:")
    print("  - Clean missing values more carefully")
    print("  - Normalize features properly")
    print("  - Remove outliers if appropriate")
    print("  - Add data validation checks")

def main():
    """Main analysis function"""
    print("OUTCOME MODEL ERROR ANALYSIS")
    print("="*60)
    
    # Load data and model
    df, model, preprocessor, le = load_data_and_model()
    if df is None:
        return
    
    # Prepare features
    swing_df = prepare_features_for_analysis(df)
    if swing_df is None:
        return
    
    # Make predictions
    try:
        # Prepare features for model
        available_feats = [f for f in preprocessor.feature_names_in_ if f in swing_df.columns]
        missing_feats = [f for f in preprocessor.feature_names_in_ if f not in swing_df.columns]
        
        if missing_feats:
            for feat in missing_feats:
                if feat in ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team', 'zone_quadrant', 'location_quadrant', 'count_advantage']:
                    swing_df[feat] = 'unknown'
                else:
                    swing_df[feat] = 0.0
        
        # Transform features
        X = preprocessor.transform(swing_df[preprocessor.feature_names_in_])
        X = np.nan_to_num(X, nan=0.0)
        
        # Get predictions
        probabilities = model.predict_proba(X)
        predictions = model.predict(X)
        
        # Decode predictions
        y_true = swing_df['outcome'].values
        y_pred = [le.inverse_transform([pred])[0] for pred in predictions]
        
        # Clean target variables
        y_true = [str(val) if val is not None and not pd.isna(val) else 'field_out' for val in y_true]
        y_pred = [str(val) if val is not None and not pd.isna(val) else 'field_out' for val in y_pred]
        
        print(f"\nPREDICTION SUMMARY:")
        print(f"Total predictions: {len(y_pred)}")
        print(f"Overall accuracy: {(np.array(y_true) == np.array(y_pred)).mean():.3f}")
        
        # Run analyses
        analyze_feature_distributions(swing_df, y_true, y_pred)
        analyze_misclassification_patterns(swing_df, y_true, y_pred)
        analyze_class_imbalance(swing_df, y_true, y_pred)
        suggest_improvements(swing_df, y_true, y_pred)
        
    except Exception as e:
        print(f"✗ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
 
 