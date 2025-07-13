import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import warnings
import joblib
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_career_data():
    """Load Acuna Jr.'s complete career data"""
    try:
        df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
        print(f"✓ Loaded Acuna Jr. data with {len(df)} pitches")
        return df
    except Exception as e:
        print(f"✗ Error loading career data: {e}")
        return None

def load_babip_data():
    """Load BABIP data with whiff rates"""
    try:
        babip_df = pd.read_csv('pitch_type_zone_batting_averages.csv')
        print(f"✓ Loaded BABIP data with {len(babip_df)} pitch type x zone combinations")
        
        # Create a dictionary for quick lookup
        babip_lookup = {}
        for _, row in babip_df.iterrows():
            key = (row['Pitch Type'], row['Zone'])
            babip_lookup[key] = {
                'batting_average_bip': row['Batting Average (BIP)'],
                'whiff_rate': row['Whiff Rate'],
                'field_out_rate_bip': row['Field Out Rate (BIP)'],
                'balls_in_play': row['Balls in Play'],
                'total_swings': row['Total Swings'],
                'total_whiffs': row['Total Whiffs']
            }
        
        return babip_lookup
    except FileNotFoundError:
        print("⚠️ BABIP data file not found. Run calculate_pitch_type_zone_batting_averages.py first.")
        return {}
    except Exception as e:
        print(f"✗ Error loading BABIP data: {e}")
        return {}

def get_babip_features(pitch_type, zone, babip_lookup):
    """Get BABIP and whiff rate features for a specific pitch type and zone"""
    key = (pitch_type, zone)
    if key in babip_lookup:
        return babip_lookup[key]
    else:
        # Return default values if no data available
        return {
            'batting_average_bip': 0.25,  # Default BABIP
            'whiff_rate': 0.35,           # Default whiff rate
            'field_out_rate_bip': 0.40,   # Default field out rate
            'balls_in_play': 0,           # No balls in play data
            'total_swings': 0,            # No swing data
            'total_whiffs': 0             # No whiff data
        }

def calculate_zone(plate_x, plate_z):
    """Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates."""
    # Strike zone boundaries (approximate)
    sz_left = -0.85
    sz_right = 0.85
    sz_bot = 1.5
    sz_top = 3.5
    
    # Check if pitch is in strike zone
    in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
    
    if in_strike_zone:
        # Calculate zone within strike zone (1-9)
        x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
        z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
        
        # Clamp to valid ranges
        x_section = max(0, min(2, x_section))
        z_section = max(0, min(2, z_section))
        
        # Convert to zone number (1-9)
        zone = z_section * 3 + x_section + 1
    else:
        # Outside strike zone (11-14)
        if plate_x < sz_left:  # Left side
            zone = 11 if plate_z > sz_top else 13
        else:  # Right side
            zone = 12 if plate_z > sz_top else 14
    
    return zone

def prepare_features(df):
    """Prepare features for whiff vs contact modeling"""
    df = df.copy()
    
    # Only calculate zones for pitches that don't have valid zone data
    if 'zone' not in df.columns or df['zone'].isna().any() or (df['zone'] <= 0).any():
        print("Calculating zones for pitches with missing/invalid zone data...")
        zone_mask = df['zone'].isna() | (df['zone'] <= 0) if 'zone' in df.columns else pd.Series([True] * len(df))
        df.loc[zone_mask, 'zone'] = df[zone_mask].apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    else:
        print("Using original Statcast zone data")
    
    # Create swing and whiff columns
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    whiff_events = ['swinging_strike', 'swinging_strike_blocked']
    
    df['is_swing'] = df['description'].isin(swing_events).astype(int)
    df['is_whiff'] = df['description'].isin(whiff_events).astype(int)
    
    # Filter for swings only
    swing_df = df[df['is_swing'] == 1].copy()
    
    if len(swing_df) == 0:
        print("✗ No swing data found!")
        return None, None, None
    
    print(f"✓ Found {len(swing_df)} swings for analysis")
    
    # Create whiff vs contact target
    swing_df['is_whiff_binary'] = swing_df['is_whiff'].astype(int)
    
    # ENGINEERED FEATURES
    
    # Zone distance
    swing_df['zone_center_x'] = 0
    swing_df['zone_center_z'] = (swing_df['sz_top'] + swing_df['sz_bot']) / 2
    swing_df['zone_distance'] = np.sqrt(
        (swing_df['plate_x'] - swing_df['zone_center_x'])**2 + 
        (swing_df['plate_z'] - swing_df['zone_center_z'])**2
    )
    
    # Movement features
    swing_df['horizontal_break'] = swing_df['api_break_x_batter_in'].fillna(0)
    swing_df['vertical_break'] = swing_df['api_break_z_with_gravity'].fillna(0)
    swing_df['movement_magnitude'] = np.sqrt(swing_df['horizontal_break']**2 + swing_df['vertical_break']**2)
    
    # Count features
    swing_df['count_pressure'] = swing_df['balls'] - swing_df['strikes']
    swing_df['count_total'] = swing_df['balls'] + swing_df['strikes']
    swing_df['behind_in_count'] = (swing_df['strikes'] > swing_df['balls']).astype(int)
    swing_df['ahead_in_count'] = (swing_df['balls'] > swing_df['strikes']).astype(int)
    swing_df['two_strikes'] = (swing_df['strikes'] >= 2).astype(int)
    swing_df['three_balls'] = (swing_df['balls'] >= 3).astype(int)
    
    # Zone features
    swing_df['in_strike_zone'] = ((swing_df['plate_x'] >= -0.85) & (swing_df['plate_x'] <= 0.85) & 
                                 (swing_df['plate_z'] >= swing_df['sz_bot']) & (swing_df['plate_z'] <= swing_df['sz_top'])).astype(int)
    swing_df['far_from_zone'] = (swing_df['zone_distance'] > 1.0).astype(int)
    swing_df['high_pitch'] = (swing_df['plate_z'] > swing_df['sz_top']).astype(int)
    swing_df['low_pitch'] = (swing_df['plate_z'] < swing_df['sz_bot']).astype(int)
    swing_df['inside_pitch'] = (swing_df['plate_x'] < -0.85).astype(int)
    swing_df['outside_pitch'] = (swing_df['plate_x'] > 0.85).astype(int)
    
    # Pitch type features
    swing_df['is_fastball'] = swing_df['pitch_type'].isin(['FF', 'SI', 'FC']).astype(int)
    swing_df['is_breaking_ball'] = swing_df['pitch_type'].isin(['SL', 'CU', 'KC']).astype(int)
    swing_df['is_offspeed'] = swing_df['pitch_type'].isin(['CH', 'FS']).astype(int)
    
    # Velocity features
    swing_df['high_velocity'] = (swing_df['release_speed'] > 95).astype(int)
    swing_df['low_velocity'] = (swing_df['release_speed'] < 85).astype(int)
    swing_df['velocity_movement_ratio'] = swing_df['release_speed'] / (swing_df['movement_magnitude'] + 0.1)
    
    # Movement features
    swing_df['high_movement'] = (swing_df['movement_magnitude'] > 6).astype(int)
    swing_df['low_movement'] = (swing_df['movement_magnitude'] < 2).astype(int)
    swing_df['movement_ratio'] = np.abs(swing_df['horizontal_break']) / (np.abs(swing_df['vertical_break']) + 0.1)
    
    # Interaction features
    swing_df['zone_distance_x_count_pressure'] = swing_df['zone_distance'] * swing_df['count_pressure']
    swing_df['movement_x_count_pressure'] = swing_df['movement_magnitude'] * swing_df['count_pressure']
    swing_df['in_zone_x_two_strikes'] = swing_df['in_strike_zone'] * swing_df['two_strikes']
    swing_df['far_from_zone_x_ahead'] = swing_df['far_from_zone'] * swing_df['ahead_in_count']
    
    # Advanced features
    swing_df['velocity_diff_from_avg'] = swing_df['release_speed'] - swing_df['release_speed'].mean()
    swing_df['movement_diff_from_avg'] = swing_df['movement_magnitude'] - swing_df['movement_magnitude'].mean()
    
    # Zone-specific features
    swing_df['zone_corner'] = ((swing_df['zone'] == 1) | (swing_df['zone'] == 3) | 
                               (swing_df['zone'] == 7) | (swing_df['zone'] == 9)).astype(int)
    swing_df['zone_heart'] = ((swing_df['zone'] == 2) | (swing_df['zone'] == 5) | 
                              (swing_df['zone'] == 8)).astype(int)
    swing_df['zone_shadow'] = ((swing_df['zone'] == 4) | (swing_df['zone'] == 6)).astype(int)
    
    # BABIP features
    print("Loading BABIP data for whiff prediction features...")
    babip_lookup = load_babip_data()
    
    if babip_lookup:
        # Add BABIP features for each pitch
        babip_features = []
        for idx, row in swing_df.iterrows():
            pitch_type = row['pitch_type']
            zone = row['zone']
            babip_data = get_babip_features(pitch_type, zone, babip_lookup)
            babip_features.append(babip_data)
        
        # Convert to DataFrame and add to main DataFrame
        babip_df = pd.DataFrame(babip_features, index=swing_df.index)
        swing_df = pd.concat([swing_df, babip_df], axis=1)
        
        print(f"✓ Added BABIP features for {len(babip_lookup)} pitch type x zone combinations")
    else:
        # Add default BABIP features if no data available
        swing_df['batting_average_bip'] = 0.25
        swing_df['whiff_rate'] = 0.35
        swing_df['field_out_rate_bip'] = 0.40
        swing_df['balls_in_play'] = 0
        swing_df['total_swings'] = 0
        swing_df['total_whiffs'] = 0
        print("Added default BABIP features (no BABIP data available)")
    
    # Define features
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'age_pit',
        # Engineered features
        'zone_distance', 'movement_magnitude', 'horizontal_break', 'vertical_break',
        'count_pressure', 'count_total', 'behind_in_count', 'ahead_in_count', 'two_strikes', 'three_balls',
        'in_strike_zone', 'far_from_zone', 'high_pitch', 'low_pitch', 'inside_pitch', 'outside_pitch',
        'is_fastball', 'is_breaking_ball', 'is_offspeed',
        'zone_distance_x_count_pressure', 'movement_x_count_pressure', 'in_zone_x_two_strikes', 'far_from_zone_x_ahead',
        'high_velocity', 'low_velocity', 'velocity_movement_ratio',
        'high_movement', 'low_movement', 'movement_ratio',
        'velocity_diff_from_avg', 'movement_diff_from_avg',
        'zone_corner', 'zone_heart', 'zone_shadow',
        # BABIP features
        'batting_average_bip', 'whiff_rate', 'field_out_rate_bip', 'balls_in_play', 'total_swings', 'total_whiffs'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in swing_df.columns]
    cat_feats = [f for f in cat_feats if f in swing_df.columns]
    
    return swing_df, num_feats, cat_feats

def create_whiff_vs_contact_model(df, num_feats, cat_feats):
    """Create a binary whiff vs contact classifier"""
    print("\n=== WHIFF VS CONTACT BINARY CLASSIFIER ===")
    
    # Prepare target
    y = df['is_whiff_binary']
    
    # Prepare features
    X = df[num_feats + cat_feats].copy()
    
    # Handle categorical features - encode them
    for feat in cat_feats:
        X[feat] = X[feat].fillna('unknown').astype(str)
    
    # Handle numeric features
    for feat in num_feats:
        X[feat] = pd.to_numeric(X[feat], errors='coerce').fillna(0)
    
    print(f"Features: {len(num_feats)} numeric, {len(cat_feats)} categorical")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_feats),
            ('cat', categorical_transformer, cat_feats)
        ],
        remainder='drop'
    )
    
    # Create full pipeline with XGBoost
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Grid search for hyperparameter tuning
    param_grid = {
        'xgbclassifier__n_estimators': [100, 200],
        'xgbclassifier__max_depth': [6, 8],
        'xgbclassifier__learning_rate': [0.1, 0.2],
        'xgbclassifier__subsample': [0.8, 1.0],
        'xgbclassifier__colsample_bytree': [0.8, 1.0]
    }
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgbclassifier', xgb_model)
    ])
    
    print("Starting grid search...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Contact', 'Whiff']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("          Predicted")
    print("          Contact  Whiff")
    print(f"Actual Contact  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"      Whiff     {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Get feature names after preprocessing
    feature_names = []
    feature_names.extend(num_feats)  # Numeric features keep their names
    # Add encoded categorical feature names
    for i, cat_feat in enumerate(cat_feats):
        unique_values = X[cat_feat].unique()
        for val in unique_values[1:]:  # Skip first value (dropped by OneHotEncoder)
            feature_names.append(f"{cat_feat}_{val}")
    
    # Feature importance (if available)
    if hasattr(best_model.named_steps['xgbclassifier'], 'feature_importances_'):
        importances = best_model.named_steps['xgbclassifier'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    model_filename = 'whiff_vs_contact_model.pkl'
    preprocessor_filename = 'whiff_vs_contact_preprocessor.pkl'
    
    joblib.dump(best_model, model_filename)
    
    # Create and save preprocessor info
    preprocessor_info = {
        'feature_names_in_': feature_names,
        'num_features': num_feats,
        'cat_features': cat_feats,
        'preprocessor': preprocessor
    }
    joblib.dump(preprocessor_info, preprocessor_filename)
    
    print(f"\n✓ Model saved as {model_filename}")
    print(f"✓ Preprocessor info saved as {preprocessor_filename}")
    
    return best_model, preprocessor_info

def main():
    """Main function to train whiff vs contact model"""
    print("=== WHIFF VS CONTACT MODEL TRAINING ===")
    
    # Load data
    df = load_career_data()
    if df is None:
        return
    
    # Prepare features
    swing_df, num_feats, cat_feats = prepare_features(df)
    if swing_df is None:
        return
    
    # Create and train model
    model, preprocessor = create_whiff_vs_contact_model(swing_df, num_feats, cat_feats)
    
    print("\n" + "="*60)
    print("WHIFF VS CONTACT MODEL TRAINING COMPLETE")
    print("="*60)
    print("Model files created:")
    print("  - whiff_vs_contact_model.pkl")
    print("  - whiff_vs_contact_preprocessor.pkl")
    print("\nUse these files for making whiff vs contact predictions!")

if __name__ == "__main__":
    main() 